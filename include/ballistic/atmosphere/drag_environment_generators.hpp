#pragma once

// =============================================================================
// DragEnvironment builders
// =============================================================================
//
// Functions that build a DragEnvironment (drag_environment.hpp) from the
// atmosphere models plus a ballistic coefficient β = m/(Cd·A) [kg/m²] that the
// caller supplies (β describes the satellite, not the air):
//   - NRLMSIS 2.1: an empirical model of the neutral atmosphere's density,
//     temperature and composition (U.S. Naval Research Laboratory).
//   - HWM14: the Horizontal Wind Model 2014, the matching empirical model of
//     the upper-atmosphere wind.
// Both take the space weather — the solar radio flux F10.7 (a proxy for solar
// heating) and the geomagnetic index Ap — from a SpaceWeatherHistory.
//
//   make_drag_environment : evaluates the models LIVE at the satellite's
//                           position (the recommended default)
//   from_models_snapshot  : table over altitude at one (lat, lon, time)
//   from_models_along     : table over altitude × time along a sampled
//                           ground track
//   from_models_4d        : table over altitude × lat × lon × time
//   from_accelerometer_density : measured density instead of a model
//   extend_density_below  : adds the 1976 standard atmosphere near the ground
//
// The tables are built once at setup; the propagator only reads them. The
// model code is compiled in the .cpp so callers do not include its 1+ MB of
// coefficient tables.
//
// =============================================================================

#include "../core/types.hpp"
#include "../time/epoch.hpp"
#include "drag_environment.hpp"
#include "beta_profile.hpp"
#include "space_weather_history.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace ballistic {
namespace atmosphere {

/// The default altitude grid of the table builders: 100 to 1500 km in 25 km
/// steps (low Earth orbit, LEO).
[[nodiscard]] std::vector<double> standard_leo_altitude_grid_km();

/// Table over altitude only, evaluated at one place and time: NRLMSIS 2.1
/// density and HWM14 wind at each altitude of `altitude_grid_km` [km] (empty
/// = standard_leo_altitude_grid_km()), at geodetic latitude `ref_lat_deg` and
/// longitude `ref_lon_deg` [degrees], epoch `t`, with the weather `sw` gives
/// for `t`. `beta_prior` becomes the β column unchanged.
///
/// Fits short arcs (much less than one orbit) that stay near the reference
/// point; for longer arcs use make_drag_environment or from_models_along.
/// Queries below the lowest grid altitude throw std::out_of_range.
[[nodiscard]] DragEnvironment from_models_snapshot(
    const Epoch& t,
    double ref_lat_deg,
    double ref_lon_deg,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km = {});

/// How the table-building loop ended (optional output of the builders).
///
/// from_models_along always reports {iterations=1, converged=true,
/// final_traj_delta_m=0}: it builds once from the track the caller gives it.
/// make_drag_environment reports {0, true, 0}: it builds no table.
///
/// from_models_along_iterated reports the real loop: the number of
/// propagations run, whether the change between the last two sampled
/// trajectories fell below the relative tolerance, and that change as the
/// largest position difference [m].
struct IterationDiag {
    int    iterations = 1;
    bool   converged  = true;
    double final_traj_delta_m = 0.0;
};

/// Table over altitude × time along a trajectory the caller already has.
/// For each sample i (epoch `sample_epochs[i]`, ECEF position
/// `sample_positions_ecef[i]` [m]) it takes the latitude and longitude below
/// the satellite and evaluates NRLMSIS 2.1 and HWM14 at every altitude of the
/// grid there. Wind is stored in ECEF axes because the place changes from
/// sample to sample.
///
/// For arcs of several orbits. Typically the samples come from a drag-free
/// propagation; when drag moves the satellite far from that track, use
/// from_models_along_iterated, which re-propagates with the table and
/// re-samples. Throws std::invalid_argument if the two vectors differ in
/// length or hold fewer than 2 samples. `diag_out`, if given, is set to
/// {1, true, 0}.
[[nodiscard]] DragEnvironment from_models_along(
    const std::vector<Epoch>& sample_epochs,
    const std::vector<Vector3d>& sample_positions_ecef,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km = {},
    IterationDiag* diag_out = nullptr);

}  // namespace atmosphere

// Forward declaration: TrajectoryConfig lives in <ballistic/trajectory.hpp>,
// which includes this header, so including it here would be circular. The
// .cpp that implements the functions below includes trajectory.hpp.
struct TrajectoryConfig;

namespace atmosphere {

/// Like from_models_along, but builds its own track and repeats until the
/// track stops changing. from_models_along's table follows whatever track it
/// was given (usually a drag-free one); where drag is strong (a reentry, a
/// very low orbit) the real track drifts away from it. This function
/// propagates without drag from (`initial_epoch`, `pos_ecef` [m],
/// `vel_ecef` [m/s], both Earth-fixed) for `duration_s` [s] (negative =
/// backward), samples every `sample_dt_s` [s], builds the table, propagates
/// again with that table's drag, and repeats up to `max_iter` propagations
/// or until the largest position change over the track, divided by the
/// distance from the Earth's center, is below `tol_relative`.
///
/// `base_config` supplies every propagation setting (gravity degree,
/// integrator, frame, ...); any drag_env it carries is ignored. `diag_out`,
/// if given, records how the loop ended. Throws std::invalid_argument for
/// duration_s == 0 or max_iter < 1.
[[nodiscard]] DragEnvironment from_models_along_iterated(
    const Epoch& initial_epoch,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    double duration_s,
    const TrajectoryConfig& base_config,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km = {},
    int max_iter = 3,
    double tol_relative = 0.01,
    double sample_dt_s = 60.0,
    IterationDiag* diag_out = nullptr);

/// Wrap a DragEnvironment in the shared_ptr<const> that
/// TrajectoryConfig::drag_env takes:
///   config.drag_env = share(from_models_along(...));
[[nodiscard]] inline std::shared_ptr<const DragEnvironment> share(DragEnvironment env) {
    return std::make_shared<const DragEnvironment>(std::move(env));
}

/// Settings for make_drag_environment. Only `beta` is required: it describes
/// the satellite, which the atmosphere models cannot supply (see
/// BetaProfile1D::constant / from_mass_cd_area).
struct DragSetup {
    // The body: a constant (or altitude-dependent) β, or an aero::AeroBeta
    // whose drag coefficient follows the flow (FlowRegimeCd, for reentry).
    std::variant<BetaProfile1D, aero::AeroBeta> beta;
    std::optional<SpaceWeatherHistory> weather;  // nullopt → the MEASURED weather loaded
                                                 // into SpaceWeatherData (the whole file);
                                                 // throws if none is loaded
    bool   allow_predicted_weather = true;  // past the last measured day, use
                                 // CelesTrak's daily FORECASTS (~45 days); a
                                 // Trajectory warns once it gets there. false =
                                 // measurements only (the Trajectory stops at
                                 // the last measured day instead)
    // Unused: make_drag_environment evaluates the models live. Kept so
    // existing initializers compile.
    std::vector<double> altitude_grid_km = {};
    int    max_iter = 3;
    double tol_relative = 0.01;
    double sample_dt_s = 60.0;
};

/// Turn drag on in one call: NRLMSIS 2.1 density and HWM14 wind evaluated
/// LIVE at the satellite's position on every force evaluation
/// (live_model.hpp), using the whole loaded space-weather file. No arc, no
/// duration, no table: the same environment serves any trajectory at any
/// epoch the weather covers, forward or backward. A Trajectory stops with an
/// error at the edge of the weather (past CelesTrak's forecast, or past the
/// last measured day with allow_predicted_weather = false) and warns once
/// when it enters forecast days.
///
///   SpaceWeatherData::instance().load_celestrak_csv("SW-All.csv");  // once
///   config.drag_env = make_drag_environment(
///       t0, pos_ecef, vel_ecef, 3600.0, config,
///       {.beta = BetaProfile1D::from_mass_cd_area(2200.0, 2.2, 6.35)});
///
/// Only `t0` and `setup` are used; pos/vel/duration/base_config are accepted
/// so existing calls compile. Throws std::runtime_error when no weather was
/// passed and none is loaded (it never substitutes quiet conditions), and
/// DataCoverageError when `t0` itself has no weather, so a bad start fails
/// here rather than mid-propagation. It costs far less than building an
/// iterated table. The table builders remain for measured density, shared
/// fields and comparisons.
[[nodiscard]] std::shared_ptr<const DragEnvironment> make_drag_environment(
    const Epoch& t0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    double duration_s,
    const TrajectoryConfig& base_config,
    const DragSetup& setup,
    IterationDiag* diag_out = nullptr);

/// Full 4D table (altitude × latitude × longitude × time) of NRLMSIS 2.1
/// density and HWM14 wind. Grids are in km, degrees and epochs; longitudes
/// should lie in [-180, 180) (queries are wrapped to that range). Heavy: a
/// grid of 200 altitudes × 36 latitudes × 72 longitudes × 24 hours is ~12
/// million model evaluations and, stored in 16-bit floats, ~25 MB of density
/// (log(ρ), one value per cell) plus ~75 MB of wind (three values per cell).
/// Computed in parallel over the time axis with OpenMP.
///
/// For many satellites sharing one atmosphere; a single satellite is better
/// served by make_drag_environment. Throws std::invalid_argument if any grid
/// is empty.
[[nodiscard]] DragEnvironment from_models_4d(
    const std::vector<double>& altitude_grid_km,
    const std::vector<double>& latitude_grid_deg,
    const std::vector<double>& longitude_grid_deg,
    const std::vector<Epoch>& epoch_grid,
    const SpaceWeatherHistory& sw,
    BetaProfile1D beta_prior);

/// Environment whose density is a measured time series: density
/// `measured_density_kg_m3[i]` [kg/m³] at `sample_epochs[i]` (e.g. derived
/// from a satellite's accelerometer, such as the TU Delft GOCE density
/// product). Every altitude of `altitude_grid_km` gets the same value at a
/// given time, so evaluate() returns the measured density interpolated in
/// time whatever the altitude — right for a satellite that stays near one
/// altitude over the arc. Wind is empty; add one with set_wind() if needed.
///
/// Throws std::invalid_argument if the two vectors differ in length or are
/// empty, if any density is not strictly positive, or if the altitude grid
/// has fewer than 2 knots. Empty grid = standard_leo_altitude_grid_km(); a
/// tighter grid around the satellite's altitudes saves memory.
[[nodiscard]] DragEnvironment from_accelerometer_density(
    const std::vector<Epoch>& sample_epochs,
    const std::vector<double>& measured_density_kg_m3,
    BetaProfile1D beta_prior,
    std::vector<double> altitude_grid_km = {});

/// Copy of `env` whose density below `floor_km` [km] (0 < floor_km <= 86)
/// comes from the U.S. Standard Atmosphere 1976 (USSA76, atmosphere/
/// ussa76.hpp), implemented exactly through its lower, well-mixed layers.
/// For deep reentry and sub-orbital flight. β passes through unchanged.
///
///   - Live environment (make_drag_environment): NRLMSIS 2.1 already runs to
///     the ground, so this CHOOSES the fixed standard atmosphere below the
///     floor instead of the weather-driven model.
///   - 1D / 2D tables: knots every 2 km from 0 km to the floor are ADDED
///     (USSA76 values, the same at every time); table knots above the floor
///     keep their values, and the log-density spline bridges the floor to the
///     table's lowest knot. (Replacing only existing knots below the floor
///     would change nothing on the standard 100-1500 km grid, which has
///     none — trap USSA-MISLABEL.)
///   - 4D tables: throws ConfigError (build the table down to the ground).
///
/// Throws ConfigError for a floor outside (0, 86] km. Wind is NOT changed: a
/// live environment keeps HWM14 (defined to the ground) below the floor; a
/// table's wind follows that table's own altitude rule.
[[nodiscard]] DragEnvironment extend_density_below(
    const DragEnvironment& env,
    double floor_km = 80.0);

}  // namespace atmosphere
}  // namespace ballistic
