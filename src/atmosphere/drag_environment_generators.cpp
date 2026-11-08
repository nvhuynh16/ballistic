// =============================================================================
// DragEnvironment builders and the live model columns — implementation
// =============================================================================
//
// The atmosphere models (NRLMSIS 2.1 density, HWM14 wind) are not compiled
// into the library: they run in the ballistic_nrl plugin, loaded at run time
// (nrl_models.hpp). Every builder that evaluates them first calls
// require_nrl_models, which raises ConfigError with installation instructions
// when the plugin is missing; the measured-density builder and
// extend_density_below need no plugin.
// =============================================================================

#include <cstdio>
#include <ballistic/core/defaults.hpp>
#include "ballistic/atmosphere/drag_environment_generators.hpp"
#include "ballistic/atmosphere/nrl_models.hpp"
#include "ballistic/atmosphere/ussa76.hpp"
#include "ballistic/atmosphere/space_weather.hpp"
#include "ballistic/coordinates/geodetic.hpp"
#include "ballistic/core/types.hpp"
#include "ballistic/trajectory.hpp"
#include "ballistic/ephemeris/spk_ephemeris.hpp"  // provides the inline
                                                   // definitions for
                                                   // ephemeris::detail::*
                                                   // declared in
                                                   // jpl_ephemeris.hpp.
#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

namespace ballistic {
namespace atmosphere {

std::vector<double> standard_leo_altitude_grid_km() {
    std::vector<double> grid;
    grid.reserve(57);
    for (double h = 100.0; h <= 1500.0; h += 25.0) grid.push_back(h);
    return grid;
}

namespace {

// Local solar time [hours, 0-24) at a longitude, the way NRLMSIS 2.1 and
// HWM14 both take it: stl = UT hours + lon_deg / 15 (mean solar time).
double solar_local_time_hours(const Epoch& t, double lon_deg) {
    auto [y, m, d, h, min, sec] = t.to_calendar(TimeScale::UTC);
    (void)y; (void)m; (void)d;
    double ut_hours = static_cast<double>(h) + static_cast<double>(min) / 60.0
                       + sec / 3600.0;
    double stl = ut_hours + lon_deg / 15.0;
    while (stl < 0.0)  stl += 24.0;
    while (stl >= 24.0) stl -= 24.0;
    return stl;
}

// HWM14's date and time inputs: iyd = year * 1000 + day of year, and the
// seconds since 00:00 UTC. Returns (iyd, sec_of_day).
std::pair<int, double> epoch_to_hwm14_iyd_sec(const Epoch& t) {
    auto [y, m, d, h, min, sec] = t.to_calendar(TimeScale::UTC);
    // Day-of-year: count days from Jan 1.
    int doy = 0;
    static const int days_in_month[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    for (int i = 0; i < m - 1; ++i) doy += days_in_month[i];
    // Leap year correction for months after February.
    auto is_leap = [](int year) {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    };
    if (m > 2 && is_leap(y)) doy += 1;
    doy += d;
    int iyd = y * 1000 + doy;
    double sec_of_day = static_cast<double>(h) * 3600.0
                         + static_cast<double>(min) * 60.0
                         + sec;
    return {iyd, sec_of_day};
}

// HWM14 horizontal wind at one point: meridional (northward +) and zonal
// (eastward +) components [m/s]. Returns {meridional_ms, zonal_ms}.
std::pair<double, double> hwm14_wind_enu(const Epoch& t,
                                          double alt_km,
                                          double lat_deg, double lon_deg,
                                          const WeatherConditions& w)
{
    auto [iyd, sec] = epoch_to_hwm14_iyd_sec(t);
    float stl = static_cast<float>(solar_local_time_hours(t, lon_deg));
    BallisticNrlHwmInput in{};
    in.iyd = iyd;
    in.sec = static_cast<float>(sec);
    in.alt_km = static_cast<float>(alt_km);
    in.glat_deg = static_cast<float>(lat_deg);
    in.glon_deg = static_cast<float>(lon_deg);
    in.stl_h = stl;
    in.f107a = static_cast<float>(w.f107_average);
    in.f107 = static_cast<float>(w.f107_daily);
    in.ap[0] = static_cast<float>(w.ap_daily);
    in.ap[1] = static_cast<float>(w.ap_history[0]);
    float wind_out[2] = {0.0f, 0.0f};
    nrl::hwm14(in, wind_out);
    // HWM14 output: wind[0] = meridional (north +), wind[1] = zonal (east +).
    return {static_cast<double>(wind_out[0]), static_cast<double>(wind_out[1])};
}

// NRLMSIS 2.1 at one point, with the date and time taken from the epoch the
// way NRLMSIS21::full_output takes them (fractional day of year, UT seconds).
BallisticNrlMsisOutput msis_at(double alt_km, double lat_deg, double lon_deg,
                               const Epoch& epoch, const WeatherConditions& w) {
    BallisticNrlMsisInput in{};
    auto [year, month, day_of_month, hour, minute, second] = epoch.to_calendar();
    (void)year; (void)month; (void)day_of_month;
    in.day = static_cast<double>(epoch.day_of_year());
    in.utsec = hour * 3600.0 + minute * 60.0 + second;
    in.alt_km = alt_km;
    in.lat_deg = lat_deg;
    in.lon_deg = lon_deg;
    in.f107a = w.f107_average;
    in.f107 = w.f107_daily;
    for (int i = 0; i < 7; ++i) in.ap[i] = w.ap_history[i];
    return nrl::msis(in);
}

// Total mass density [kg/m^3] from NRLMSIS 2.1 (dn[1]).
double msis_density(double alt_km, double lat_deg, double lon_deg,
                    const Epoch& epoch, const WeatherConditions& w) {
    return msis_at(alt_km, lat_deg, lon_deg, epoch, w).dn[1];
}

}  // namespace

DragEnvironment from_models_snapshot(
    const Epoch& t,
    double ref_lat_deg,
    double ref_lon_deg,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km)
{
    require_nrl_models("from_models_snapshot");
    if (altitude_grid_km.empty()) {
        altitude_grid_km = standard_leo_altitude_grid_km();
    }

    const WeatherConditions weather = sw.at(t);

    const std::size_t n = altitude_grid_km.size();
    std::vector<double> density_kg_m3(n);
    std::vector<double> meridional_ms(n);
    std::vector<double> zonal_ms(n);

    for (std::size_t k = 0; k < n; ++k) {
        const double alt_km = altitude_grid_km[k];
        density_kg_m3[k] = msis_density(alt_km, ref_lat_deg, ref_lon_deg, t, weather);
        auto [m, z] = hwm14_wind_enu(t, alt_km, ref_lat_deg, ref_lon_deg, weather);
        meridional_ms[k] = m;
        zonal_ms[k] = z;
    }

    return DragEnvironment(
        DensityProfile1D(altitude_grid_km, std::move(density_kg_m3)),
        WindProfile1D_ENU(altitude_grid_km, std::move(meridional_ms), std::move(zonal_ms)),
        std::move(beta_prior));
}

namespace {

// Rotate HWM14's local east/north wind into ECEF axes at geodetic (lat, lon)
// [rad].
Vector3d enu_to_ecef(double meridional_ms, double zonal_ms,
                      double lat_rad, double lon_rad)
{
    Matrix3d R = coords::ecef_to_enu_rotation(lat_rad, lon_rad);
    Vector3d wind_enu(zonal_ms, meridional_ms, 0.0);  // (east, north, up)
    return R.transpose() * wind_enu;
}

// Weather for a live-model evaluation. Throws std::out_of_range, naming the
// loaded span (as MJD), when the epoch is outside the source's window.
WeatherConditions live_weather(const LiveModelSource& src, const Epoch& epoch) {
    // Tolerance as in Trajectory's edge handling: a step sized to land on the
    // edge can overshoot it by rounding (~1e-11 s with exact Epoch storage).
    constexpr double tol = defaults::EPOCH_EQUALITY_SECONDS;
    if (src.valid && (src.valid->start - epoch > tol || epoch - src.valid->end > tol)) {
        throw std::out_of_range(
            "live drag model: no space weather at this epoch (the loaded weather "
            "covers MJD " + std::to_string(src.valid->start.mjd(TimeScale::UTC)) +
            " to " + std::to_string(src.valid->end.mjd(TimeScale::UTC)) +
            "); load a newer SW-All.csv or pass the weather explicitly");
    }
    return src.weather.at(epoch);
}

// NRLMSIS 2.1 runs from the ground to the exobase; a state more than 1 km
// below the ellipsoid has passed through the ground (a ground-impact event
// stops a propagation at 0 km), so density there is an error, not a value.
constexpr double kLowestAltitudeKm = -1.0;

[[noreturn]] void throw_below_ground(double alt_km) {
    throw DataCoverageError(
        "drag environment: altitude " + std::to_string(alt_km) + " km is below the ground "
        "(the atmosphere is defined down to -1 km). The trajectory passed through the Earth's "
        "surface: stop it with a ground-impact event (events::ground_impact(), armed by "
        "Trajectory::propagate_until).");
}

}  // namespace

// ---- Live model columns (live_model.hpp) -----------------------------------

AtmosphereState LiveModelDensity::evaluate_state(const Vector3d& position_ecef, const Epoch& epoch) const {
    if (!src_) return {};
    const auto geo = coords::ecef_to_geodetic(position_ecef);
    const double alt_km = geo.altitude / 1000.0;
    if (alt_km < kLowestAltitudeKm) throw_below_ground(alt_km);
    // USSA76 only where extend_density_below asked for it (floor > 0). Swapping
    // in the standard atmosphere at the ellipsoid by default would make density
    // jump 4.5% at the ground, and a step that straddled it would fail its
    // error test down to the minimum step size.
    if (src_->isa_floor_km > 0.0 && alt_km < src_->isa_floor_km) {
        const auto u = ussa76::evaluate(geo.altitude);
        return {u.density_kg_m3, u.temperature_k, u.molar_mass_kg_kmol};
    }
    constexpr double RAD_TO_DEG = 57.29577951308232;
    // full_output runs the same calculation as density() (dn[1] is its total
    // mass density) and also gives the temperature and the number densities.
    const auto out = msis_at(alt_km, geo.latitude * RAD_TO_DEG, geo.longitude * RAD_TO_DEG,
                             epoch, live_weather(*src_, epoch));
    // Mean molar mass of the species in the total mass density (N2, O2, O,
    // He, H, Ar, N, anomalous O; NO is excluded from it, as in NRLMSIS).
    double n_total = 0.0;
    for (int i = 2; i <= 9; ++i) {
        if (out.dn[i] > 0.0) n_total += out.dn[i];
    }
    const double molar_mass = n_total > 0.0 ? out.dn[1] / n_total * aero::kAvogadroKmol : 0.0;
    return {out.dn[1], out.tn, molar_mass};
}

Vector3d LiveModelWind::evaluate(const Vector3d& position_ecef, const Epoch& epoch) const {
    if (!src_) return Vector3d::Zero();
    const auto geo = coords::ecef_to_geodetic(position_ecef);
    constexpr double RAD_TO_DEG = 57.29577951308232;
    // HWM14 is defined from the ground up. Between 0 and -1 km (a step's
    // trial points near a ground impact) the 0 km wind is used; deeper is an
    // error, as for the density.
    if (geo.altitude / 1000.0 < kLowestAltitudeKm) throw_below_ground(geo.altitude / 1000.0);
    auto [m, z] = hwm14_wind_enu(epoch, std::max(geo.altitude, 0.0) / 1000.0,
                                 geo.latitude * RAD_TO_DEG, geo.longitude * RAD_TO_DEG,
                                 live_weather(*src_, epoch));
    return enu_to_ecef(m, z, geo.latitude, geo.longitude);
}

DragEnvironment from_models_along(
    const std::vector<Epoch>& sample_epochs,
    const std::vector<Vector3d>& sample_positions_ecef,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km,
    IterationDiag* diag_out)
{
    require_nrl_models("from_models_along");
    if (sample_epochs.size() != sample_positions_ecef.size()) {
        throw std::invalid_argument(
            "from_models_along: sample_epochs / sample_positions_ecef size mismatch");
    }
    if (sample_epochs.size() < 2) {
        throw std::invalid_argument(
            "from_models_along: need at least 2 sample points to build a 2D profile");
    }
    if (altitude_grid_km.empty()) {
        altitude_grid_km = standard_leo_altitude_grid_km();
    }

    const std::size_t n_t = sample_epochs.size();
    const std::size_t n_alt = altitude_grid_km.size();

    constexpr double RAD_TO_DEG = 57.29577951308232;
    std::vector<double> density_grid(n_t * n_alt);
    std::vector<Vector3d> wind_grid(n_t * n_alt);

    for (std::size_t ti = 0; ti < n_t; ++ti) {
        const Epoch& t = sample_epochs[ti];
        const WeatherConditions weather = sw.at(t);
        const auto geo = coords::ecef_to_geodetic(sample_positions_ecef[ti]);
        const double lat_deg = geo.latitude * RAD_TO_DEG;
        const double lon_deg = geo.longitude * RAD_TO_DEG;

        for (std::size_t k = 0; k < n_alt; ++k) {
            const double alt_km = altitude_grid_km[k];
            const std::size_t idx = ti * n_alt + k;
            density_grid[idx] = msis_density(alt_km, lat_deg, lon_deg, t, weather);
            auto [m, z] = hwm14_wind_enu(t, alt_km, lat_deg, lon_deg, weather);
            wind_grid[idx] = enu_to_ecef(m, z, geo.latitude, geo.longitude);
        }
    }

    DensityProfile2D density(altitude_grid_km, sample_epochs, std::move(density_grid));
    WindProfile2D_ECEF wind(altitude_grid_km, sample_epochs, std::move(wind_grid));

    if (diag_out) {
        diag_out->iterations = 1;
        diag_out->converged = true;
        diag_out->final_traj_delta_m = 0.0;  // single pass: no delta to report
    }

    // Single pass by design; from_models_along_iterated re-propagates with
    // the env and re-samples until the trajectory delta converges.

    return DragEnvironment(std::move(density), std::move(wind), std::move(beta_prior));
}

DragEnvironment from_models_4d(
    const std::vector<double>& altitude_grid_km,
    const std::vector<double>& latitude_grid_deg,
    const std::vector<double>& longitude_grid_deg,
    const std::vector<Epoch>& epoch_grid,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior)
{
    require_nrl_models("from_models_4d");
    if (altitude_grid_km.empty() || latitude_grid_deg.empty()
        || longitude_grid_deg.empty() || epoch_grid.empty()) {
        throw std::invalid_argument(
            "from_models_4d: all four axis grids must be non-empty");
    }

    const std::size_t n_alt = altitude_grid_km.size();
    const std::size_t n_lat = latitude_grid_deg.size();
    const std::size_t n_lon = longitude_grid_deg.size();
    const std::size_t n_t   = epoch_grid.size();

    // Storage layout matches DensityField4D / WindField4D_ECEF:
    //   index = ((t * n_lon + j) * n_lat + i) * n_alt + k
    std::vector<half>    log_rho_storage(n_alt * n_lat * n_lon * n_t);
    std::vector<Vector3d> wind_storage(n_alt * n_lat * n_lon * n_t);

    // Parallelize over the epoch axis: each thread writes to a disjoint slice
    // of the storage (the plugin keeps one HWM14 instance per thread).
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (std::ptrdiff_t ti = 0; ti < static_cast<std::ptrdiff_t>(n_t); ++ti) {
            const Epoch& t = epoch_grid[static_cast<std::size_t>(ti)];
            const WeatherConditions weather = sw.at(t);
            for (std::size_t j = 0; j < n_lon; ++j) {
                const double lon_deg = longitude_grid_deg[j];
                for (std::size_t i = 0; i < n_lat; ++i) {
                    const double lat_deg = latitude_grid_deg[i];
                    constexpr double DEG_TO_RAD = 0.017453292519943295;
                    const double lat_rad = lat_deg * DEG_TO_RAD;
                    const double lon_rad = lon_deg * DEG_TO_RAD;
                    for (std::size_t k = 0; k < n_alt; ++k) {
                        const double alt_km = altitude_grid_km[k];
                        const std::size_t idx =
                            ((static_cast<std::size_t>(ti) * n_lon + j) * n_lat + i) * n_alt + k;
                        const double rho = msis_density(alt_km, lat_deg, lon_deg, t, weather);
                        log_rho_storage[idx] = encode_log_rho(rho);
                        auto [m, z] = hwm14_wind_enu(t, alt_km, lat_deg, lon_deg, weather);
                        wind_storage[idx] = enu_to_ecef(m, z, lat_rad, lon_rad);
                    }
                }
            }
        }
    }

    DensityField4D density(altitude_grid_km, latitude_grid_deg, longitude_grid_deg,
                            epoch_grid, std::move(log_rho_storage));
    WindField4D_ECEF wind(altitude_grid_km, latitude_grid_deg, longitude_grid_deg,
                           epoch_grid, std::move(wind_storage));
    return DragEnvironment(std::move(density), std::move(wind), std::move(beta_prior));
}


// -----------------------------------------------------------------------------
// from_accelerometer_density
// -----------------------------------------------------------------------------
DragEnvironment from_accelerometer_density(
    const std::vector<Epoch>& sample_epochs,
    const std::vector<double>& measured_density_kg_m3,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km)
{
    if (sample_epochs.size() != measured_density_kg_m3.size()) {
        throw std::invalid_argument(
            "from_accelerometer_density: sample_epochs.size() must equal "
            "measured_density_kg_m3.size()");
    }
    if (sample_epochs.empty()) {
        throw std::invalid_argument(
            "from_accelerometer_density: at least one sample is required");
    }
    if (altitude_grid_km.empty()) {
        altitude_grid_km = standard_leo_altitude_grid_km();
    }
    if (altitude_grid_km.size() < 2) {
        throw std::invalid_argument(
            "from_accelerometer_density: altitude_grid_km needs >= 2 knots");
    }

    const std::size_t n_t = sample_epochs.size();
    const std::size_t n_h = altitude_grid_km.size();

    // density_grid is row-major time × altitude. At each epoch row, repeat
    // the measured density across all altitudes — the env's altitude
    // dimension becomes degenerate (log-spline is flat), so evaluate(pos,t)
    // returns the time-interpolated measured density regardless of altitude.
    std::vector<double> density_grid(n_t * n_h);
    for (std::size_t ti = 0; ti < n_t; ++ti) {
        const double rho = measured_density_kg_m3[ti];
        // DensityProfile2D's LogRhoSpline rejects rho <= 0; surface a clearer
        // error before that throws.
        if (!(rho > 0.0)) {
            throw std::invalid_argument(
                "from_accelerometer_density: all measured densities must be "
                "strictly positive (log-spline storage requirement)");
        }
        for (std::size_t hi = 0; hi < n_h; ++hi) {
            density_grid[ti * n_h + hi] = rho;
        }
    }

    DensityProfile2D density(std::move(altitude_grid_km),
                              std::vector<Epoch>(sample_epochs),
                              std::move(density_grid));
    WindProfile1D_ENU wind;  // empty — no HWM14 mixed in
    return DragEnvironment(std::move(density), std::move(wind),
                            std::move(beta_prior));
}

// -----------------------------------------------------------------------------
// extend_density_below
// -----------------------------------------------------------------------------
namespace {

// The table's altitude knots with U.S. Standard Atmosphere 1976 knots every
// 2 km from 0 km to the floor: knots at or below the floor take the USSA76
// value, knots above keep the table's. Only replacing EXISTING knots below
// the floor would change nothing on the standard grid, which starts at
// 100 km (trap USSA-MISLABEL).
std::vector<double> extended_altitudes(const std::vector<double>& table_km, double floor_km) {
    std::vector<double> alt;
    for (double z = 0.0; z < floor_km; z += 2.0) alt.push_back(z);
    alt.push_back(floor_km);
    for (double z : table_km) if (z > floor_km) alt.push_back(z);
    std::sort(alt.begin(), alt.end());
    alt.erase(std::unique(alt.begin(), alt.end()), alt.end());
    return alt;
}

// Index of `z` in `table_km` (exact knot match), or -1.
long knot_index(const std::vector<double>& table_km, double z) {
    for (std::size_t i = 0; i < table_km.size(); ++i)
        if (table_km[i] == z) return static_cast<long>(i);
    return -1;
}

}  // namespace

DragEnvironment extend_density_below(const DragEnvironment& env, double floor_km) {
    if (!(floor_km > 0.0 && floor_km <= ussa76::kMaxAltitudeM / 1000.0)) {
        throw ConfigError(
            "extend_density_below: floor_km must be in (0, 86] km - the U.S. Standard "
            "Atmosphere 1976 is implemented through its well-mixed layers only (got " +
            std::to_string(floor_km) + " km)");
    }
    DragEnvironment patched = env;

    std::visit([&](const auto& d) {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, DensityProfile1D>) {
            const auto& tab_alt = d.altitude_km();
            const auto& tab_rho = d.density_kg_m3();
            std::vector<double> alt = extended_altitudes(tab_alt, floor_km);
            std::vector<double> rho;
            for (double z : alt)
                rho.push_back(z <= floor_km ? ussa76::density(z * 1000.0)
                                            : tab_rho[knot_index(tab_alt, z)]);
            patched.set_density(DensityProfile1D(std::move(alt), std::move(rho)));
        } else if constexpr (std::is_same_v<T, DensityProfile2D>) {
            const auto& tab_alt = d.altitude_km();
            const auto& tab_grid = d.density_grid();
            std::vector<Epoch> epochs = d.epochs();
            std::vector<double> alt = extended_altitudes(tab_alt, floor_km);
            const std::size_t n_tab = tab_alt.size(), n_alt = alt.size();
            std::vector<double> grid(n_alt * epochs.size());
            for (std::size_t ti = 0; ti < epochs.size(); ++ti)
                for (std::size_t k = 0; k < n_alt; ++k)
                    grid[ti * n_alt + k] = alt[k] <= floor_km
                        ? ussa76::density(alt[k] * 1000.0)   // no time dependence
                        : tab_grid[ti * n_tab + knot_index(tab_alt, alt[k])];
            patched.set_density(DensityProfile2D(std::move(alt), std::move(epochs), std::move(grid)));
        } else if constexpr (std::is_same_v<T, LiveModelDensity>) {
            auto src = std::make_shared<LiveModelSource>(*d.source());
            src->isa_floor_km = floor_km;
            patched.set_density(LiveModelDensity(src));
            if (std::holds_alternative<LiveModelWind>(env.wind()))
                patched.set_wind(LiveModelWind(src));
        } else {
            throw ConfigError(
                "extend_density_below: a 4D density field cannot be extended; build it with an "
                "altitude grid that reaches the ground (from_models_4d) instead.");
        }
    }, env.density());

    return patched;
}

// -----------------------------------------------------------------------------
// from_models_along_iterated
// -----------------------------------------------------------------------------
namespace {

// Propagate a Trajectory with the given config from t0 over `duration_s`
// (negative = backward) and return (epochs, positions_ecef) sampled every
// `sample_dt_s`, in ascending time order either way. The samples are read
// by interpolating the trajectory (traj(t)) at each target time.
std::pair<std::vector<Epoch>, std::vector<Vector3d>>
sample_arc(const Epoch& t0, const Vector3d& pos_ecef, const Vector3d& vel_ecef,
            double duration_s, double sample_dt_s,
            const TrajectoryConfig& cfg)
{
    Trajectory traj(t0, pos_ecef, vel_ecef, cfg);
    traj.mark_internal();
    traj.propagate(t0 + duration_s);  // build the history once

    const double span = std::abs(duration_s);
    const double sign = duration_s < 0.0 ? -1.0 : 1.0;
    const int n_steps = std::max(int(span / sample_dt_s), 1);
    std::vector<Epoch> epochs;
    std::vector<Vector3d> positions;
    epochs.reserve(n_steps + 1);
    positions.reserve(n_steps + 1);
    for (int i = 0; i <= n_steps; ++i) {
        const double dt = sign * std::min(double(i) * sample_dt_s, span);
        const Epoch t = t0 + dt;
        const PropagationResult r = traj(t);  // ECEF
        epochs.push_back(t);
        positions.emplace_back(r.position);
    }
    if (sign < 0.0) {
        std::reverse(epochs.begin(), epochs.end());
        std::reverse(positions.begin(), positions.end());
    }
    return {std::move(epochs), std::move(positions)};
}

double max_pair_delta_m(const std::vector<Vector3d>& a,
                        const std::vector<Vector3d>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double max_d = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_d = std::max(max_d, (a[i] - b[i]).norm());
    }
    return max_d;
}

double max_pair_delta_relative(const std::vector<Vector3d>& a,
                                const std::vector<Vector3d>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double max_d = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double abs_d = (a[i] - b[i]).norm();
        const double ref = b[i].norm();
        if (ref > 0.0) max_d = std::max(max_d, abs_d / ref);
    }
    return max_d;
}

}  // namespace

DragEnvironment from_models_along_iterated(
    const Epoch& initial_epoch,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    double duration_s,
    const TrajectoryConfig& base_config,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km,
    int max_iter,
    double tol_relative,
    double sample_dt_s,
    IterationDiag* diag_out)
{
    require_nrl_models("from_models_along_iterated");
    if (duration_s == 0.0) {
        throw std::invalid_argument(
            "from_models_along_iterated: duration_s must be non-zero "
            "(negative = backward from initial_epoch)");
    }
    if (max_iter < 1) {
        throw std::invalid_argument(
            "from_models_along_iterated: max_iter must be >= 1");
    }

    // Iteration 0: drag-free nominal arc.
    TrajectoryConfig cfg = base_config;
    cfg.drag_env = nullptr;

    auto [epochs_prev, positions_prev] = sample_arc(
        initial_epoch, pos_ecef, vel_ecef,
        duration_s, sample_dt_s, cfg);

    // `from_models_along` consumes its `beta_prior` arg by value; pass
    // copies inside the loop so the original is preserved for re-use.
    DragEnvironment env = from_models_along(
        epochs_prev, positions_prev, sw,
        BetaProfile1D{beta_prior}, altitude_grid_km);

    int iterations = 1;
    bool converged = false;
    double final_delta_m = 0.0;

    for (int iter = 1; iter < max_iter; ++iter) {
        cfg.drag_env = std::make_shared<const DragEnvironment>(env);

        auto sampled = sample_arc(
            initial_epoch, pos_ecef, vel_ecef,
            duration_s, sample_dt_s, cfg);
        std::vector<Epoch> epochs_new = std::move(sampled.first);
        std::vector<Vector3d> positions_new = std::move(sampled.second);

        ++iterations;
        const double abs_delta_m = max_pair_delta_m(positions_new, positions_prev);
        const double rel_delta = max_pair_delta_relative(positions_new, positions_prev);
        final_delta_m = abs_delta_m;

        if (rel_delta < tol_relative) {
            converged = true;
            // Final env is already built from the previous iteration's
            // samples; the just-propagated trajectory matches within
            // tol so no further env-rebuild is needed.
            break;
        }

        env = from_models_along(
            epochs_new, positions_new, sw,
            BetaProfile1D{beta_prior}, altitude_grid_km);
        epochs_prev = std::move(epochs_new);
        positions_prev = std::move(positions_new);
    }

    if (diag_out) {
        diag_out->iterations = iterations;
        diag_out->converged = converged;
        diag_out->final_traj_delta_m = final_delta_m;
    }

    return env;
}

std::shared_ptr<const DragEnvironment> make_drag_environment(
    const Epoch& t0,
    const Vector3d& /*pos_ecef*/,
    const Vector3d& /*vel_ecef*/,
    double /*duration_s*/,
    const TrajectoryConfig& /*base_config*/,
    const DragSetup& setup,
    IterationDiag* diag_out)
{
    require_nrl_models("make_drag_environment");
    auto src = std::make_shared<LiveModelSource>();
    src->allow_predicted = setup.allow_predicted_weather;
    if (setup.weather) {
        src->weather = *setup.weather;
        if (!src->weather.is_flat()) src->valid = src->weather.window();
    } else {
        // Default: the MEASURED space weather loaded into SpaceWeatherData —
        // the whole file, then CelesTrak's forecast unless refused. Never a
        // silent substitute of quiet conditions (which bias density low
        // whenever the Sun or the geomagnetic field is active): nothing
        // loaded is an error.
        const auto& loaded = SpaceWeatherData::instance();
        if (!loaded.is_loaded()) {
            throw std::runtime_error(
                "make_drag_environment: no space weather passed and none loaded. "
                "Load measured F10.7/Ap first (Python: ballistic.load(), or "
                "ballistic.load(space_weather=\"path/to/SW-All.csv\") offline; C++: "
                "SpaceWeatherData::instance().load_celestrak_csv(\"SW-All.csv\")), "
                "or pass DragSetup::weather explicitly — "
                "SpaceWeatherHistory::from_constant(WeatherConditions::quiet()) "
                "opts into quiet conditions on purpose.");
        }
        const auto full = loaded.full_history(setup.allow_predicted_weather);
        src->weather = full->history;
        src->valid = full->valid;
        src->measured_until = full->measured_until;
        src->loaded_weather = true;
    }
    // Fail at setup, not mid-propagation, when the start itself has no weather.
    if (src->valid && (t0 < src->valid->start || t0 > src->valid->end)) {
        // Dates in TT: the loaded history starts in 1957, before UTC existed
        // (formatting it in UTC would throw "TAI-UTC is undefined before
        // 1961" and hide this message).
        auto day = [](const Epoch& e) {
            const auto [y, m, d, h, mi, sec] = e.to_calendar(TimeScale::TT);
            (void)h; (void)mi; (void)sec;
            char b[16];
            std::snprintf(b, sizeof b, "%04d-%02d-%02d", y, m, d);
            return std::string(b);
        };
        const auto [y, m, d, h, mi, sec] = t0.to_calendar(TimeScale::TT);
        (void)h; (void)mi; (void)sec;
        char start[16];
        std::snprintf(start, sizeof start, "%04d-%02d-%02d", y, m, d);
        throw DataCoverageError(
            "make_drag_environment: the start (" + std::string(start) + ") has no space "
            "weather: the loaded weather covers " + day(src->valid->start) + " to " +
            day(src->valid->end) + (setup.allow_predicted_weather ? " (TT)" : " (TT), forecasts refused") +
            ". Load a newer SW-All.csv (Python: ballistic.load()) or pass the weather explicitly.");
    }
    if (diag_out) *diag_out = IterationDiag{0, true, 0.0};
    return std::make_shared<const DragEnvironment>(
        LiveModelDensity(src), LiveModelWind(src),
        std::visit([](const auto& b) { return DragEnvironment::BetaVariant(b); }, setup.beta));
}

}  // namespace atmosphere
}  // namespace ballistic
