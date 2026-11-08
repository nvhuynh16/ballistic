#pragma once

// Helpers shared by several C++ tests (the pass/fail framework itself is in
// test_utils.hpp):
//   - IsolatedDataDir: hides the user's downloaded data files from a test;
//   - compute_pos_rmse / compute_arg_latitude / load_test_ephemeris;
//   - namespace glide: the "glide fixture", a synthetic reentry-glide
//     trajectory with a known empirical acceleration, used to test the
//     empirical-acceleration (EA) estimators;
//   - namespace radar: synthetic radar and camera sensors and their samples.
//
// Frames: ECEF / ITRF = Earth-fixed; GCRS = the inertial (non-rotating)
// frame the propagator integrates in.

#include <ballistic/ballistic.hpp>
#include <ballistic/calibration/multisensor.hpp>
#include <ballistic/calibration/detail/run_propagation.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/ephemeris/spk_ephemeris.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ballistic::test::helpers {

// The current process id, for per-process scratch directory names.
inline long long current_pid() {
#if defined(_WIN32)
    return static_cast<long long>(::_getpid());
#else
    return static_cast<long long>(::getpid());
#endif
}

// The environment variable that names the platform data root (the library
// looks for <root>/ballistic): LOCALAPPDATA on Windows, XDG_DATA_HOME
// elsewhere.
inline constexpr const char* data_root_env_var() {
#if defined(_WIN32)
    return "LOCALAPPDATA";
#else
    return "XDG_DATA_HOME";
#endif
}

// RAII helper: point XDG_DATA_HOME (Linux) / LOCALAPPDATA (Windows) at an
// empty temp directory for the duration of the scope, so initialize({})
// can't auto-discover finals2000A.data / leap-seconds.list / SW-All.csv
// from the user's real data directory. Use it in every test whose results
// must not depend on cached user data.
class IsolatedDataDir {
public:
    IsolatedDataDir() {
        constexpr const char* kVar = data_root_env_var();
        const char* orig = std::getenv(kVar);
        if (orig) original_value_ = orig;
        original_set_ = (orig != nullptr);
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("ballistic_test_isolated_" +
                    std::to_string(current_pid()));
        std::filesystem::create_directories(tmp_dir_);
        var_name_ = kVar;
#if defined(_WIN32)
        _putenv_s(kVar, tmp_dir_.string().c_str());
#else
        ::setenv(kVar, tmp_dir_.c_str(), 1);
#endif
    }
    ~IsolatedDataDir() {
#if defined(_WIN32)
        if (original_set_) _putenv_s(var_name_, original_value_.c_str());
        else _putenv_s(var_name_, "");
#else
        if (original_set_) ::setenv(var_name_, original_value_.c_str(), 1);
        else ::unsetenv(var_name_);
#endif
        std::error_code ec;
        std::filesystem::remove_all(tmp_dir_, ec);
    }
private:
    std::filesystem::path tmp_dir_;
    std::string original_value_;
    bool original_set_ = false;
    const char* var_name_ = nullptr;
};

// Position RMSE (root-mean-square of the 3-D position difference) between two
// interpolated trajectories, sampled every `dt_sample` seconds over
// [t_start, t_end] relative to t0. Samples where either interpolator has no
// state are skipped; returns 1e9 if none pair up.
inline double compute_pos_rmse(
    TrajectoryInterpolator& test_interp,
    TrajectoryInterpolator& truth_interp,
    const Epoch& t0,
    double t_start, double t_end, double dt_sample)
{
    double sse = 0.0;
    int count = 0;
    for (double t = t_start; t <= t_end; t += dt_sample) {
        auto s_test = test_interp.get_state(t0 + t);
        auto s_truth = truth_interp.get_state(t0 + t);
        if (!s_test || !s_truth) continue;
        sse += (s_test->position - s_truth->position).squaredNorm();
        ++count;
    }
    if (count == 0) return 1e9;
    return std::sqrt(sse / count);
}

// Argument of latitude (angle from the ascending node to the satellite,
// measured in the orbit plane, 0-360 degrees) from an Earth-fixed position and
// velocity. The state is first rotated to the inertial GCRS frame.
inline double compute_arg_latitude(
    const Vector3d& pos, const Vector3d& vel, const Epoch& epoch)
{
    const StateVector gcrs = coords::itrf_to_gcrs_full(StateVector(pos, vel), epoch);
    const Vector3d& pos_eci = gcrs.position;
    const Vector3d& vel_eci = gcrs.velocity;

    Vector3d h = pos_eci.cross(vel_eci);
    Vector3d n(-h.y(), h.x(), 0.0);
    double n_mag = n.norm();
    double r = pos_eci.norm();

    if (n_mag > 1e-10) {
        double cos_u = n.dot(pos_eci) / (n_mag * r);
        cos_u = std::clamp(cos_u, -1.0, 1.0);
        double arg_lat = std::acos(cos_u) * constants::RAD_TO_DEG;
        if (pos_eci.z() < 0) arg_lat = 360.0 - arg_lat;
        return arg_lat;
    }
    return 0.0;
}

// Load a JPL planetary ephemeris file (SPK format), trying DE421 first (the one
// ESA's orbit products use), then DE440. Returns nullptr if neither file is in
// data/; callers then run on the library's embedded DE440.
inline std::shared_ptr<ephemeris::EphemerisProvider> load_test_ephemeris(
    const std::string& de421_path = BALLISTIC_PROJECT_ROOT "/data/de421.bsp",
    const std::string& de440_path = BALLISTIC_PROJECT_ROOT "/data/de440.bsp")
{
    namespace fs = std::filesystem;

    if (fs::exists(de421_path)) {
        auto spk = std::make_shared<ephemeris::SPKEphemeris>();
        if (spk->load(de421_path)) {
            std::cout << "Loaded JPL DE421 ephemeris" << std::endl;
            return spk;
        }
    }

    if (fs::exists(de440_path)) {
        auto spk = std::make_shared<ephemeris::SPKEphemeris>();
        if (spk->load(de440_path)) {
            std::cout << "Loaded JPL DE440 ephemeris" << std::endl;
            return spk;
        }
    }

    std::cout << "WARNING: JPL ephemeris not found - using analytical (degraded accuracy)" << std::endl;
    return nullptr;
}

// =============================================================================
// Glide fixture: synthetic reentry glide (reference test bed for the EA
// estimators)
// =============================================================================
//
// An empirical acceleration (EA) is an extra acceleration, beyond the modeled
// forces, that an estimator recovers from observations. Here the truth
// trajectory is flown with a known, hand-made EA so a test can compare what an
// estimator recovers with the truth. The EA has two Gaussian bumps in time: a
// "forced flight" lift bump centered at FC = 70 s and a "recoil" bump centered
// at RC = 420 s. The start is at ~120 km altitude at 85% of circular speed, so
// the object descends; the arc used by the tests ends 10 s before the
// trajectory first drops below 1 km altitude (about 192 s after the start for
// these initial conditions; the arc is about 182 s), so only the rising flank of
// the recoil bump falls inside
// it.
namespace glide {

constexpr double LIFT = 8;
constexpr double LAT = 3;
constexpr double FC = 70;
constexpr double FW = 35;
constexpr double LP = 100;
constexpr double RET = 6;
constexpr double RC = 420;
constexpr double RW = 80;

// Force model for the glide baseline: Earth gravity to degree 20 only. Drag,
// solar radiation pressure (SRP), third bodies, relativity and solid tides are
// off, so the only departure from that gravity field is the EA added by
// make_forced_config().
inline TrajectoryConfig base_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 20;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

// The gravity baseline plus the truth EA (lift and lateral bump + recoil bump),
// tabulated every 1 s over 0-700 s in GCRS. `rd` is the recoil direction: zero
// for the first (probe) pass of make_glide(), then a unit vector pointing
// mostly toward the Earth's center, tilted toward +z (north), for the truth
// pass.
inline TrajectoryConfig make_forced_config(const Vector3d& rd) {
    TrajectoryConfig cfg = base_config();
    std::vector<forces::EmpiricalAccelEntry> entries;
    for (int i = 0; i <= 700; ++i) {
        double t = i;
        double g  = std::exp(-0.5 * std::pow((t - FC) / FW, 2));
        double rg = std::exp(-0.5 * std::pow((t - RC) / RW, 2));
        entries.emplace_back(
            t,
            Vector3d(0,
                     LIFT * g,
                     LAT * g * std::sin(2 * constants::PI * t / LP))
            + RET * rg * rd);
    }
    cfg.empirical_accel = std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(
            std::move(entries), forces::EmpiricalAccelFrame::GCRS));
    return cfg;
}

// Adds zero-mean Gaussian noise of standard deviation `sigma` to each component
// of a list of (Epoch, Vector3d) observations; one random stream seeded by
// `seed` supplies x, y, z in turn, so the result is reproducible.
inline std::vector<std::pair<Epoch, Vector3d>> add_noise(
    const std::vector<std::pair<Epoch, Vector3d>>& clean,
    double sigma, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> n(0.0, sigma);
    std::vector<std::pair<Epoch, Vector3d>> out;
    out.reserve(clean.size());
    for (auto& p : clean)
        out.emplace_back(p.first, p.second + Vector3d(n(rng), n(rng), n(rng)));
    return out;
}

struct Setup {
    Epoch t0;
    Vector3d pos, vel;     // initial ECEF state of the truth trajectory
    double eval_end = 0.0; // arc end time [s] (10 s before re-entry crossing)
    std::unique_ptr<Trajectory> truth_traj;
    std::unique_ptr<TrajectoryInterpolator> truth_interp;

    // Clean observations sampled from truth_interp (ECEF position; ECEF
    // velocity; ECEF acceleration). The 100/200 suffix denotes Hz.
    std::vector<std::pair<Epoch, Vector3d>> pos_100, pos_200;
    std::vector<std::pair<Epoch, Vector3d>> vel_100, vel_200;
    std::vector<std::pair<Epoch, Vector3d>> acc_100, acc_200;

    // Pre-noised variants (default seeds 42 / 43 / 44 for POS / VEL / ACC).
    // For multi-seed sweeps, call add_noise() directly with a custom seed.
    std::vector<std::pair<Epoch, Vector3d>> pos_n100, pos_n200;
    std::vector<std::pair<Epoch, Vector3d>> vel_n100, vel_n200;
    std::vector<std::pair<Epoch, Vector3d>> acc_n100, acc_n200;
};

namespace detail {

inline void sample_ecef(const Setup& s,
                        TrajectoryInterpolator& ecef_i,
                        TrajectoryInterpolator& gcrs_i,
                        double dt,
                        std::vector<std::pair<Epoch, Vector3d>>& po,
                        std::vector<std::pair<Epoch, Vector3d>>& vo,
                        std::vector<std::pair<Epoch, Vector3d>>& ao)
{
    for (double t = 0; t <= s.eval_end; t += dt) {
        auto se = ecef_i.get_state(s.t0 + t);
        auto sg = gcrs_i.get_state(s.t0 + t);
        if (!se || !sg) continue;
        po.emplace_back(s.t0 + t, se->position);
        vo.emplace_back(s.t0 + t, se->velocity);
        StateVector svg{sg->position, sg->velocity};
        Vector3d a_g = s.truth_traj->force_model().acceleration(s.t0 + t, svg);
        StateVector sv{se->position, se->velocity};
        ao.emplace_back(
            s.t0 + t,
            util::acceleration_to_itrf(a_g, sv, s.t0 + t));
    }
}

}  // namespace detail

// Builds the glide fixture: a probe pass finds the recoil direction from the
// position at RC, the truth pass flies the full EA, a bisection finds the
// 1 km crossing, then clean and noisy samples are drawn at 100 Hz and 200 Hz
// (noise σ: position 1 m, velocity 0.25 m/s, acceleration 0.1 m/s²).
inline Setup make_glide() {
    Setup s;
    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    const double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    const double vc = std::sqrt(constants::earth::GM / R);
    const double wr = constants::earth::ANGULAR_VELOCITY * R;
    s.pos = Vector3d(R, 0, 0);
    s.vel = Vector3d(-30.0, 0.85 * vc - wr, 0);

    // Probe pass: identify recoil direction from peak position.
    Trajectory sp(s.t0, s.pos, s.vel, make_forced_config(Vector3d::Zero()));
    for (int i = 0; i < 50000; ++i) {
        auto sr = sp.step();
        if (sr.epoch - s.t0 >= RC + 10) break;
    }
    TrajectoryInterpolator si(sp);
    auto pk = si.get_state(s.t0 + RC);
    Vector3d rd = pk
        ? (-pk->position.normalized() + 0.4 * Vector3d(0, 0, 1)).normalized()
        : Vector3d::Zero();

    // Truth pass: full trajectory with recoil-direction-aware EA.
    s.truth_traj = std::make_unique<Trajectory>(
        s.t0, s.pos, s.vel, make_forced_config(rd));
    double fd = 0;
    for (int i = 0; i < 100000; ++i) {
        auto sr = s.truth_traj->step();
        if (coords::ecef_to_geodetic(sr.position).altitude < 1000) {
            fd = sr.epoch - s.t0;
            break;
        }
        if (sr.epoch - s.t0 > 700) {
            fd = sr.epoch - s.t0;
            break;
        }
    }
    s.truth_interp = std::make_unique<TrajectoryInterpolator>(*s.truth_traj);

    // Bisect to find the precise re-entry crossing; back off 10 s.
    double lo = fd - 2, hi = fd;
    for (int i = 0; i < 50; ++i) {
        double mid = 0.5 * (lo + hi);
        auto st = s.truth_interp->get_state(s.t0 + mid);
        if (st && coords::ecef_to_geodetic(st->position).altitude < 1000) hi = mid;
        else lo = mid;
    }
    s.eval_end = hi - 10.0;

    InterpolatorConfig icfg;
    icfg.output_frame = InterpolationFrame::GCRS;
    TrajectoryInterpolator gcrs_i(*s.truth_traj, icfg);

    detail::sample_ecef(s, *s.truth_interp, gcrs_i, 0.01,
                        s.pos_100, s.vel_100, s.acc_100);
    detail::sample_ecef(s, *s.truth_interp, gcrs_i, 0.005,
                        s.pos_200, s.vel_200, s.acc_200);

    s.pos_n100 = add_noise(s.pos_100, 1.0, 42);
    s.vel_n100 = add_noise(s.vel_100, 0.25, 43);
    s.acc_n100 = add_noise(s.acc_100, 0.1, 44);
    s.pos_n200 = add_noise(s.pos_200, 1.0, 42);
    s.vel_n200 = add_noise(s.vel_200, 0.25, 43);
    s.acc_n200 = add_noise(s.acc_200, 0.1, 44);

    std::cout << "Glide reentry: arc=" << std::fixed << std::setprecision(1)
              << s.eval_end << "s  100Hz=" << s.pos_100.size()
              << " obs  200Hz=" << s.pos_200.size() << " obs\n\n";
    return s;
}

// Re-propagates from the glide initial state with the supplied EA and returns
// the position RMSE against truth, sampled at 1 Hz over [10 s, eval_end].
// max_step_s > 0 caps the integrator step, for an EA that was estimated with
// a fixed integrator step and must be replayed with the same step.
inline double reprop_rmse(const Setup& s,
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea,
    double max_step_s = 0.0)
{
    if (!ea) return 1e9;
    TrajectoryConfig rc = base_config();
    rc.empirical_accel = ea;
    if (max_step_s > 0.0) rc.rk45.max_step_size = max_step_s;
    Trajectory rt(s.t0, s.pos, s.vel, rc);
    while (rt.epoch() - s.t0 < s.eval_end) {
        (void)rt.step();
    }
    TrajectoryInterpolator ri(rt);
    double sse = 0.0;
    int n = 0;
    for (double t = 10.0; t <= s.eval_end; t += 1.0) {
        auto a = ri.get_state(s.t0 + t);
        auto b = s.truth_interp->get_state(s.t0 + t);
        if (a && b) {
            sse += (a->position - b->position).squaredNorm();
            ++n;
        }
    }
    return n > 0 ? std::sqrt(sse / n) : 1e9;
}

}  // namespace glide

// =============================================================================
// Radar and camera helpers for the sensor-fusion EA tests
// (test_ea_smoother.cpp, test_ea_problem_spec.cpp). Angles are azimuth
// (from north, toward east) and elevation in the station's local
// east-north-up (ENU) frame.
// =============================================================================
namespace radar {

// Place a sensor `distance_m` east of `impact_ecef` at altitude=0 on the
// WGS84 ellipsoid. Used to anchor radars relative to a known impact spot.
[[nodiscard]] inline Vector3d place_sensor_east(
    const Vector3d& impact_ecef, double distance_m)
{
    auto geo = coords::ecef_to_geodetic(impact_ecef);
    Matrix3d R_enu_to_ecef =
        coords::ecef_to_enu_rotation(geo.latitude, geo.longitude).transpose();
    Vector3d east_enu(distance_m, 0.0, 0.0);
    Vector3d raw = impact_ecef + R_enu_to_ecef * east_enu;
    auto rg = coords::ecef_to_geodetic(raw);
    return coords::geodetic_to_ecef(rg.latitude, rg.longitude, 0.0);
}

// Build a fully-configured RadarObs with the given station ECEF position
// and noise σs (range [m], angle [deg], range-rate [m/s]). Fills in
// lat/lon and the cached ECEF→ENU rotation matrix.
[[nodiscard]] inline calibration::RadarObs make_radar(
    const Vector3d& position_ecef,
    double sigma_range = 2.0,
    double sigma_angle_deg = 0.02,
    double sigma_rate = 0.02)
{
    calibration::RadarObs r;
    r.position_ecef = position_ecef;
    r.sigma_range = sigma_range;
    r.sigma_angle = sigma_angle_deg * constants::DEG_TO_RAD;
    r.sigma_rate = sigma_rate;
    auto geo = coords::ecef_to_geodetic(position_ecef);
    r.lat_rad = geo.latitude;
    r.lon_rad = geo.longitude;
    r.ecef_to_enu = coords::ecef_to_enu_rotation(geo.latitude, geo.longitude);
    return r;
}

// Sample (range, azimuth, elevation, range-rate) from `truth_ecef` at
// `sample_rate_hz` over [epoch0, epoch0 + duration_s]. Range-rate is the speed
// along the line of sight. With `add_noise` (the default) each channel gets
// Gaussian noise with the radar's σ, reproducible for a given `seed`; pass
// false for clean data. Returns the list ready to assign to RadarObs::samples.
[[nodiscard]] inline std::vector<std::pair<Epoch, Eigen::Vector4d>>
generate_radar_samples(
    const calibration::RadarObs& radar_spec,
    TrajectoryInterpolator& truth_ecef,
    const Epoch& epoch0,
    double sample_rate_hz,
    double duration_s,
    unsigned seed = 42,
    bool add_noise = true)
{
    std::vector<std::pair<Epoch, Eigen::Vector4d>> out;
    const double dt = 1.0 / sample_rate_hz;
    out.reserve(static_cast<size_t>(std::ceil(duration_s / dt)) + 1);

    std::mt19937 rng(seed);
    std::normal_distribution<double> n_range(0.0, radar_spec.sigma_range);
    std::normal_distribution<double> n_angle(0.0, radar_spec.sigma_angle);
    std::normal_distribution<double> n_rate(0.0, radar_spec.sigma_rate);

    for (double t = 0.0; t <= duration_s + 1e-9; t += dt) {
        auto state = truth_ecef.get_state(epoch0 + t);
        if (!state) continue;
        Vector3d d = state->position - radar_spec.position_ecef;
        double range = d.norm();
        if (range < 1.0) continue;
        Vector3d d_hat = d / range;
        Vector3d d_enu = radar_spec.ecef_to_enu * d;
        double E = d_enu.x(), N = d_enu.y(), U = d_enu.z();
        double rho_h = std::sqrt(E * E + N * N);
        double az = std::atan2(E, N);
        double el = std::atan2(U, std::max(rho_h, 1e-9));
        double rate = d_hat.dot(state->velocity);

        // Keep this exact order of random draws (range, az, el, rate): tests
        // pin results computed from these seeded samples, and a different
        // draw order would change every noisy sample.

        Eigen::Vector4d obs;
        if (add_noise) {
            obs << range + n_range(rng),
                   az    + n_angle(rng),
                   el    + n_angle(rng),
                   rate  + n_rate(rng);
        } else {
            obs << range, az, el, rate;
        }
        out.emplace_back(epoch0 + t, obs);
    }
    return out;
}

// A radar for the glide fixture, placed on the ground `east_offset_m` east of
// the point where the glide's start state would hit the ground with no EA
// (pure gravity), with samples of the truth trajectory attached.
[[nodiscard]] calibration::RadarObs build_radar_for_glide(
    const glide::Setup& s,
    double east_offset_m,
    double sample_rate_hz,
    double duration_s,
    bool add_noise,
    unsigned seed)
{
    // Find the impact spot using the no-EA ballistic trajectory.
    auto base = glide::base_config();
    auto [traj, interp] = calibration::detail::run_propagation_arc(
        s.t0, s.pos, s.vel, base, /*ea=*/nullptr,
        /*max_duration_s=*/1000.0, /*min_altitude_m=*/-1000.0);
    double t_lo = -1.0, t_hi = -1.0;
    for (double t = 1.0; t <= 1000.0; t += 1.0) {
        auto state = interp->get_state(s.t0 + t);
        if (!state) break;
        if (coords::ecef_to_geodetic(state->position).altitude <= 0.0) {
            t_lo = t - 1.0;
            t_hi = t;
            break;
        }
    }
    if (t_hi < 0.0) throw std::runtime_error("no impact crossing");
    for (int i = 0; i < 50; ++i) {
        double mid = 0.5 * (t_lo + t_hi);
        auto st = interp->get_state(s.t0 + mid);
        if (!st) break;
        if (coords::ecef_to_geodetic(st->position).altitude <= 0.0) t_hi = mid;
        else t_lo = mid;
    }
    auto fs = interp->get_state(s.t0 + t_hi);
    Vector3d impact = fs ? fs->position : Vector3d::Zero();
    Vector3d radar_pos = place_sensor_east(impact, east_offset_m);
    auto radar = make_radar(radar_pos);
    radar.samples = generate_radar_samples(
        radar, *s.truth_interp, s.t0, sample_rate_hz, duration_s,
        seed, add_noise);
    return radar;
}


// A camera (optical sensor giving azimuth/elevation bearings only) at the
// given ECEF site. Default σ 50 µrad, the camera used in the fusion tests.
[[nodiscard]] inline calibration::OpticsObs make_optics(
    const Vector3d& position_ecef,
    double sigma_angle_urad = 50.0)
{
    calibration::OpticsObs c;
    c.position_ecef = position_ecef;
    c.sigma_angle = sigma_angle_urad * 1e-6;
    auto geo = coords::ecef_to_geodetic(position_ecef);
    c.lat_rad = geo.latitude;
    c.lon_rad = geo.longitude;
    c.ecef_to_enu = coords::ecef_to_enu_rotation(geo.latitude, geo.longitude);
    return c;
}

// Sample (az, el) from `truth_ecef` at `sample_rate_hz`, optionally adding
// Gaussian noise per the spec's σ (deterministic for `seed`).
[[nodiscard]] inline std::vector<std::pair<Epoch, Eigen::Vector2d>>
generate_optics_samples(
    const calibration::OpticsObs& cam_spec,
    TrajectoryInterpolator& truth_ecef,
    const Epoch& epoch0,
    double sample_rate_hz,
    double duration_s,
    unsigned seed = 43,
    bool add_noise = true)
{
    std::vector<std::pair<Epoch, Eigen::Vector2d>> out;
    const double dt = 1.0 / sample_rate_hz;
    out.reserve(static_cast<size_t>(std::ceil(duration_s / dt)) + 1);
    std::mt19937 rng(seed);
    std::normal_distribution<double> n_angle(0.0, cam_spec.sigma_angle);
    for (double t = 0.0; t <= duration_s + 1e-9; t += dt) {
        auto state = truth_ecef.get_state(epoch0 + t);
        if (!state) continue;
        StateVector sv{state->position, state->velocity};
        Eigen::Vector2d obs = cam_spec.predict(sv);
        if (add_noise) {
            obs(0) += n_angle(rng);
            obs(1) += n_angle(rng);
        }
        out.emplace_back(epoch0 + t, obs);
    }
    return out;
}

}  // namespace radar

} // namespace ballistic::test::helpers
