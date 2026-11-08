// B-spline empirical-acceleration (EA) estimator with position, velocity and
// acceleration observations of the glide fixture, at 1, 10, 100 and 300 Hz.
//
// EA = the extra acceleration, beyond the force model, that an estimator fits so
// that a propagated trajectory matches the observations. The glide fixture is a
// synthetic reentry glide from 120 km: a known "truth" force (lift, a rotating
// lateral force and a late return force, parameters below) is applied as an EA
// table, and the estimator must recover it from noise-free samples of the truth.
//
// Every cell asserts one thing: re-propagating with the estimated EA lands
// closer to the truth (position RMSE, root-mean-square error, over the forced
// arc) than the baseline trajectory without any EA. The printed RMSEs are for
// the log; the "diag_*" cells only print comparisons.

#include "../test_utils.hpp"
#include "../test_helpers.hpp"

using ballistic::test::helpers::compute_pos_rmse;
#include <ballistic/init.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/calibration/bspline_accel_estimator.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;
using namespace ballistic::calibration::detail;
using namespace ballistic::calibration;

// Glide force parameters (same as test_bspline_estimator.cpp)
constexpr double LIFT_PEAK = 8.0;
constexpr double LATERAL_AMP = 3.0;
constexpr double FORCE_CENTER = 70.0;
constexpr double FORCE_WIDTH = 35.0;
constexpr double LATERAL_PERIOD = 100.0;
constexpr double RETURN_MAGNITUDE = 6.0;
constexpr double RETURN_CENTER = 420.0;
constexpr double RETURN_WIDTH = 80.0;

static TrajectoryConfig fast_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 20;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

// The glide truth force as an EA table in GCRS (the inertial frame), one entry
// per second.
static TrajectoryConfig make_forced_config(const Vector3d& return_dir) {
    TrajectoryConfig cfg = fast_config();

    auto ea = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;

    for (int i = 0; i <= 700; ++i) {
        double t = static_cast<double>(i);
        double gauss = std::exp(-0.5 * std::pow((t - FORCE_CENTER) / FORCE_WIDTH, 2));
        double lift = LIFT_PEAK * gauss;
        double lateral = LATERAL_AMP * gauss * std::sin(2.0 * constants::PI * t / LATERAL_PERIOD);
        double return_gauss = std::exp(-0.5 * std::pow((t - RETURN_CENTER) / RETURN_WIDTH, 2));
        double ret = RETURN_MAGNITUDE * return_gauss;

        Vector3d force(0.0, lift, lateral);
        force += ret * return_dir;
        entries.emplace_back(t, force);
    }

    *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
    cfg.empirical_accel = ea;
    return cfg;
}

static TrajectoryConfig make_ballistic_config() {
    return fast_config();
}

static std::pair<double, bool> propagate_to_ground(
    Trajectory& traj, const Epoch& t0, double max_duration)
{
    double last_t = 0.0;
    for (int step = 0; step < 100000; ++step) {
        auto sr = traj.step();
        double elapsed = sr.epoch - t0;
        auto geo = coords::ecef_to_geodetic(sr.position);
        if (geo.altitude < 1000.0) return {elapsed, true};
        if (elapsed > max_duration) return {elapsed, false};
        last_t = elapsed;
    }
    return {last_t, false};
}

static double bisect_impact(TrajectoryInterpolator& interp,
    const Epoch& t0, double approx_t)
{
    double lo = approx_t - 2.0, hi = approx_t;
    for (int i = 0; i < 50; ++i) {
        double mid = 0.5 * (lo + hi);
        auto state = interp.get_state(t0 + mid);
        if (!state) break;
        auto geo = coords::ecef_to_geodetic(state->position);
        if (geo.altitude < 1000.0) hi = mid;
        else lo = mid;
    }
    return hi;
}


struct Setup {
    Epoch t0;
    Vector3d pos, vel;
    double forced_duration;

    TrajectoryInterpolator* truth_interp;
    TrajectoryInterpolator* base_interp;

    std::vector<std::pair<Epoch, Vector3d>> pos_data_1hz, pos_data_10hz, pos_data_100hz, pos_data_300hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_data_1hz, vel_data_10hz, vel_data_100hz, vel_data_300hz;
    std::vector<std::pair<Epoch, Vector3d>> accel_data_1hz, accel_data_10hz, accel_data_100hz, accel_data_300hz;

    std::unique_ptr<Trajectory> truth_traj_ptr;
    std::unique_ptr<Trajectory> base_traj_ptr;
    std::unique_ptr<TrajectoryInterpolator> truth_interp_ptr;
    std::unique_ptr<TrajectoryInterpolator> base_interp_ptr;
};

static Setup& get_setup() {
    static Setup s;
    static bool initialized = false;
    if (initialized) return s;
    initialized = true;

    std::cout << "  [setup] Building glide reentry truth trajectory..." << std::endl;

    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    double v_circ = std::sqrt(constants::earth::GM / R);
    double omega_r = constants::earth::ANGULAR_VELOCITY * R;
    double v_tangential = 0.85 * v_circ - omega_r;
    double v_radial = -30.0;

    s.pos = Vector3d(R, 0.0, 0.0);
    s.vel = Vector3d(v_radial, v_tangential, 0.0);

    // Ballistic trajectory -> impact point
    auto ball_traj = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, make_ballistic_config());
    auto [ball_duration, ball_ok] = propagate_to_ground(*ball_traj, s.t0, 600.0);
    auto ball_interp_tmp = std::make_unique<TrajectoryInterpolator>(*ball_traj);
    if (ball_ok) ball_duration = bisect_impact(*ball_interp_tmp, s.t0, ball_duration);

    // Glide-only -> get position at return center
    Trajectory glide_only(s.t0, s.pos, s.vel, make_forced_config(Vector3d::Zero()));
    for (int step = 0; step < 50000; ++step) {
        auto sr = glide_only.step();
        if ((sr.epoch - s.t0) >= RETURN_CENTER + 10.0) break;
    }
    TrajectoryInterpolator glide_interp(glide_only);
    auto glide_at_peak = glide_interp.get_state(s.t0 + RETURN_CENTER);

    Vector3d return_dir = Vector3d::Zero();
    if (glide_at_peak) {
        Vector3d descent = -glide_at_peak->position.normalized();
        Vector3d northward(0.0, 0.0, 1.0);
        return_dir = (descent + 0.4 * northward).normalized();
    }

    // Full forced trajectory (truth)
    s.truth_traj_ptr = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, make_forced_config(return_dir));
    auto [fd, forced_ok] = propagate_to_ground(*s.truth_traj_ptr, s.t0, 700.0);
    s.truth_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.truth_traj_ptr);
    if (forced_ok) fd = bisect_impact(*s.truth_interp_ptr, s.t0, fd);
    s.forced_duration = fd;
    s.truth_interp = s.truth_interp_ptr.get();

    std::cout << "  [setup] Forced duration: " << std::fixed << std::setprecision(1)
              << s.forced_duration << "s" << std::endl;

    // Baseline trajectory
    s.base_traj_ptr = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, make_ballistic_config());
    while (s.base_traj_ptr->epoch() - s.t0 < s.forced_duration) {
        s.base_traj_ptr->step();
    }
    s.base_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.base_traj_ptr);
    s.base_interp = s.base_interp_ptr.get();

    double sample_end = s.forced_duration - 10.0;

    // Position sampling (ECEF)
    auto sample_pos = [&](double dt_sample) {
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = 0.0; t <= sample_end; t += dt_sample) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                data.emplace_back(s.t0 + t, state->position);
            }
        }
        return data;
    };

    s.pos_data_1hz = sample_pos(1.0);
    s.pos_data_10hz = sample_pos(0.1);
    s.pos_data_100hz = sample_pos(0.01);
    s.pos_data_300hz = sample_pos(1.0 / 300.0);

    std::cout << "  [setup] Sampled "
              << s.pos_data_1hz.size() << " (1Hz) / "
              << s.pos_data_10hz.size() << " (10Hz) / "
              << s.pos_data_100hz.size() << " (100Hz) / "
              << s.pos_data_300hz.size() << " (300Hz) position points" << std::endl;

    // Velocity sampling (GCRS)
    auto sample_vel = [&](double dt_sample) {
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = 0.0; t <= sample_end; t += dt_sample) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                StateVector s_ecef;
                s_ecef.position = state->position;
                s_ecef.velocity = state->velocity;
                StateVector s_eci = util::itrf_to_gcrs(
                    s_ecef, s.t0 + t);
                data.emplace_back(s.t0 + t, s_eci.velocity);
            }
        }
        return data;
    };

    s.vel_data_1hz = sample_vel(1.0);
    s.vel_data_10hz = sample_vel(0.1);
    s.vel_data_100hz = sample_vel(0.01);
    s.vel_data_300hz = sample_vel(1.0 / 300.0);

    std::cout << "  [setup] Sampled "
              << s.vel_data_1hz.size() << " (1Hz) / "
              << s.vel_data_10hz.size() << " (10Hz) / "
              << s.vel_data_100hz.size() << " (100Hz) / "
              << s.vel_data_300hz.size() << " (300Hz) velocity points" << std::endl;

    // Acceleration sampling (GCRS, via truth force model)
    auto sample_accel = [&](double dt_sample) {
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = 0.0; t <= sample_end; t += dt_sample) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                StateVector s_ecef;
                s_ecef.position = state->position;
                s_ecef.velocity = state->velocity;
                StateVector s_eci = util::itrf_to_gcrs(
                    s_ecef, s.t0 + t);
                Vector3d a_gcrs = s.truth_traj_ptr->force_model().acceleration(
                    s.t0 + t, s_eci);
                data.emplace_back(s.t0 + t, a_gcrs);
            }
        }
        return data;
    };

    s.accel_data_1hz = sample_accel(1.0);
    s.accel_data_10hz = sample_accel(0.1);
    s.accel_data_100hz = sample_accel(0.01);
    s.accel_data_300hz = sample_accel(1.0 / 300.0);

    std::cout << "  [setup] Sampled "
              << s.accel_data_1hz.size() << " (1Hz) / "
              << s.accel_data_10hz.size() << " (10Hz) / "
              << s.accel_data_100hz.size() << " (100Hz) / "
              << s.accel_data_300hz.size() << " (300Hz) acceleration points" << std::endl;

    return s;
}

int main() {
    ballistic::initialize({});

    TestRunner runner("B-Spline Velocity & Acceleration Estimators");

    auto run_vel_test = [](
        const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& vel_data,
        double knot_spacing, double lambda, int max_iterations)
    {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;
        double base_rmse = compute_pos_rmse(*s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knot_spacing;
        cfg.lambda = lambda;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = max_iterations;

        auto result = estimate_empirical_acceleration_bspline_from_velocity(
            s.t0, s.pos, s.vel, vel_data, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) recon_traj.step();
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        double improvement = (1.0 - recon_rmse / base_rmse) * 100.0;

        std::cout << "    [" << label << "] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << recon_rmse << "m, improvement: "
                  << improvement << "%, iters: " << result.iterations_used << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse, label + " did not beat baseline");
        return recon_rmse;
    };

    auto run_accel_test = [](
        const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& accel_data,
        double knot_spacing, double lambda, int max_iterations)
    {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;
        double base_rmse = compute_pos_rmse(*s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knot_spacing;
        cfg.lambda = lambda;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = max_iterations;

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.t0, s.pos, s.vel, accel_data, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) recon_traj.step();
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        double improvement = (1.0 - recon_rmse / base_rmse) * 100.0;

        std::cout << "    [" << label << "] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << recon_rmse << "m, improvement: "
                  << improvement << "%, iters: " << result.iterations_used << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse, label + " did not beat baseline");
        return recon_rmse;
    };

    // Position tests (baseline comparison)
    auto run_pos_test = [](
        const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& pos_data,
        double knot_spacing, double lambda, int max_iterations)
    {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;
        double base_rmse = compute_pos_rmse(*s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knot_spacing;
        cfg.lambda = lambda;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = max_iterations;

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, pos_data, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) recon_traj.step();
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        double improvement = (1.0 - recon_rmse / base_rmse) * 100.0;

        std::cout << "    [" << label << "] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << recon_rmse << "m, improvement: "
                  << improvement << "%, iters: " << result.iterations_used << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse, label + " did not beat baseline");
        return recon_rmse;
    };

    runner.add_test("pos_1hz", [&]() { run_pos_test("pos_1hz", get_setup().pos_data_1hz, 20.0, 1e-6, 5); });
    runner.add_test("pos_10hz", [&]() { run_pos_test("pos_10hz", get_setup().pos_data_10hz, 5.0, 1e-8, 5); });
    runner.add_test("pos_100hz", [&]() { run_pos_test("pos_100hz", get_setup().pos_data_100hz, 2.0, 1e-10, 10); });
    runner.add_test("pos_300hz", [&]() { run_pos_test("pos_300hz", get_setup().pos_data_300hz, 2.0, 1e-10, 10); });

    // Position with a larger regularization weight λ (1e-8, 1e-6) at 100 and
    // 300 Hz; each must still beat the baseline.
    runner.add_test("pos_100hz_lam8", [&]() { run_pos_test("pos_100hz_lam8", get_setup().pos_data_100hz, 2.0, 1e-8, 10); });
    runner.add_test("pos_300hz_lam8", [&]() { run_pos_test("pos_300hz_lam8", get_setup().pos_data_300hz, 2.0, 1e-8, 10); });
    runner.add_test("pos_300hz_lam6", [&]() { run_pos_test("pos_300hz_lam6", get_setup().pos_data_300hz, 2.0, 1e-6, 10); });

    // Velocity tests
    runner.add_test("vel_1hz", [&]() { run_vel_test("vel_1hz", get_setup().vel_data_1hz, 20.0, 1e-6, 5); });
    runner.add_test("vel_10hz", [&]() { run_vel_test("vel_10hz", get_setup().vel_data_10hz, 5.0, 1e-8, 5); });
    runner.add_test("vel_100hz", [&]() { run_vel_test("vel_100hz", get_setup().vel_data_100hz, 2.0, 1e-10, 10); });
    runner.add_test("vel_300hz", [&]() { run_vel_test("vel_300hz", get_setup().vel_data_300hz, 2.0, 1e-10, 10); });

    // Velocity with a drift penalty (lambda_drift) added to the fit.
    auto run_vel_drift_test = [](
        const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& vel_data,
        double knot_spacing, double lambda, double lambda_drift, int max_iterations)
    {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;
        double base_rmse = compute_pos_rmse(*s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knot_spacing;
        cfg.lambda = lambda;
        cfg.lambda_drift = lambda_drift;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = max_iterations;

        auto result = estimate_empirical_acceleration_bspline_from_velocity(
            s.t0, s.pos, s.vel, vel_data, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) recon_traj.step();
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        std::cout << "    [" << label << "] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << recon_rmse << "m, iters: "
                  << result.iterations_used << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse, label + " did not beat baseline");
        return recon_rmse;
    };

    // Drift-penalty sweep 1e-10 .. 1 at 10, 100 and 300 Hz; each value must still
    // beat the baseline.
    for (double ld : {1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}) {
        std::string tag = [](double v) {
            char buf[32]; std::snprintf(buf, sizeof(buf), "%.0e", v); return std::string(buf);
        }(ld);

        runner.add_test("vel_10hz_drift_" + tag, [&, ld]() {
            run_vel_drift_test("vel_10hz_drift", get_setup().vel_data_10hz, 5.0, 1e-8, ld, 5);
        });
        runner.add_test("vel_100hz_drift_" + tag, [&, ld]() {
            run_vel_drift_test("vel_100hz_drift", get_setup().vel_data_100hz, 2.0, 1e-10, ld, 10);
        });
        runner.add_test("vel_300hz_drift2_" + tag, [&, ld]() {
            run_vel_drift_test("vel_300hz_drift", get_setup().vel_data_300hz, 2.0, 1e-10, ld, 10);
        });
    }

    // Acceleration tests
    runner.add_test("accel_1hz", [&]() { run_accel_test("accel_1hz", get_setup().accel_data_1hz, 20.0, 1e-6, 5); });
    runner.add_test("accel_10hz", [&]() { run_accel_test("accel_10hz", get_setup().accel_data_10hz, 5.0, 1e-8, 5); });
    runner.add_test("accel_100hz", [&]() { run_accel_test("accel_100hz", get_setup().accel_data_100hz, 2.0, 1e-10, 10); });
    runner.add_test("accel_300hz", [&]() { run_accel_test("accel_300hz", get_setup().accel_data_300hz, 2.0, 1e-10, 10); });

    // Acceleration with a larger λ, for reference (the result should barely change).
    runner.add_test("accel_300hz_lam8", [&]() { run_accel_test("accel_300hz_lam8", get_setup().accel_data_300hz, 2.0, 1e-8, 10); });

    // Acceleration with the 10 Hz tuning (5 s knots, λ 1e-8) at higher rates.
    runner.add_test("accel_100hz_5s", [&]() { run_accel_test("accel_100hz_5s", get_setup().accel_data_100hz, 5.0, 1e-8, 5); });
    runner.add_test("accel_300hz_5s", [&]() { run_accel_test("accel_300hz_5s", get_setup().accel_data_300hz, 5.0, 1e-8, 5); });

    // =========================================================================
    // Diagnostic: velocity coefficient comparison 10 Hz vs 100 Hz vs 300 Hz
    // =========================================================================
    runner.add_test("diag_vel_coefficients", [&]() {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        // Propagate model trajectory (no EA) for residual computation
        Trajectory model_traj(s.t0, s.pos, s.vel, fast_config());
        while (model_traj.epoch() - s.t0 < s.forced_duration) model_traj.step();
        TrajectoryInterpolator model_interp(model_traj);

        auto compute_vel_residuals = [&](
            const std::vector<std::pair<Epoch, Vector3d>>& vel_data)
        {
            std::vector<double> tau;
            std::vector<Vector3d> dv;
            for (const auto& obs : vel_data) {
                double t_rel = obs.first - s.t0;
                if (t_rel > eval_end) continue;
                auto ms = model_interp.get_state(obs.first);
                if (!ms) continue;
                StateVector s_ecef;
                s_ecef.position = ms->position;
                s_ecef.velocity = ms->velocity;
                StateVector s_eci = util::itrf_to_gcrs(
                    s_ecef, obs.first);
                tau.push_back(t_rel);
                dv.push_back(obs.second - s_eci.velocity);
            }
            return std::make_pair(tau, dv);
        };

        auto [tau_10, dv_10] = compute_vel_residuals(s.vel_data_10hz);
        auto [tau_100, dv_100] = compute_vel_residuals(s.vel_data_100hz);
        auto [tau_300, dv_300] = compute_vel_residuals(s.vel_data_300hz);

        // Fit B-spline at each rate (velocity uses reg_deriv_order=1)
        auto fit_10 = detail::fit_bspline(tau_10, dv_10, 5.0, 1e-8, 1);
        auto fit_100 = detail::fit_bspline(tau_100, dv_100, 2.0, 1e-10, 1);
        auto fit_300 = detail::fit_bspline(tau_300, dv_300, 2.0, 1e-10, 1);

        std::cout << "    10hz:  n_obs=" << tau_10.size() << " n_basis=" << fit_10.n_basis << std::endl;
        std::cout << "    100hz: n_obs=" << tau_100.size() << " n_basis=" << fit_100.n_basis << std::endl;
        std::cout << "    300hz: n_obs=" << tau_300.size() << " n_basis=" << fit_300.n_basis << std::endl;

        // Compare 100 Hz vs 300 Hz coefficients (same knot spacing)
        if (fit_100.n_basis == fit_300.n_basis) {
            double max_cx = 0, max_cy = 0, max_cz = 0;
            for (int j = 0; j < fit_100.n_basis; ++j) {
                max_cx = (std::max)(max_cx, std::abs(fit_100.cx(j) - fit_300.cx(j)));
                max_cy = (std::max)(max_cy, std::abs(fit_100.cy(j) - fit_300.cy(j)));
                max_cz = (std::max)(max_cz, std::abs(fit_100.cz(j) - fit_300.cz(j)));
            }
            std::cout << "    100hz vs 300hz coeff max diff: cx=" << std::scientific
                      << std::setprecision(4) << max_cx << " cy=" << max_cy
                      << " cz=" << max_cz << std::endl;
        }

        // Compare EA (d/dt) at common times
        if (fit_100.n_basis == fit_300.n_basis) {
            double max_ea_diff = 0;
            for (int i = 0; i <= 90; ++i) {
                double t = 2.0 * i;
                if (t > eval_end) break;
                Vector3d ea_100 = detail::eval_d1(fit_100, t);
                Vector3d ea_300 = detail::eval_d1(fit_300, t);
                max_ea_diff = (std::max)(max_ea_diff, (ea_100 - ea_300).norm());
            }
            std::cout << "    max EA (d/dt) diff 100vs300: " << std::scientific
                      << std::setprecision(6) << max_ea_diff << " m/s^2" << std::endl;
        }

        // Test: velocity with linear EA table vs cubic
        auto make_vel_ea = [&](const detail::BSplineFitResult& fit, bool use_linear) {
            std::vector<double> times;
            times.push_back(fit.t_start);
            for (size_t k = fit.degree + 1;
                 k < fit.knots.size() - fit.degree - 1; ++k) {
                double tk = fit.knots[k];
                if (tk > fit.t_start + 1e-10 && tk < fit.t_end - 1e-10)
                    times.push_back(tk);
            }
            times.push_back(fit.t_end);

            std::vector<forces::EmpiricalAccelEntry> entries;
            for (double t : times) {
                entries.emplace_back(t, detail::eval_d1(fit, t));
            }
            auto ea = std::make_shared<forces::EmpiricalAcceleration>();
            if (use_linear)
                *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS, forces::EAInterp::Linear);
            else
                *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
            return ea;
        };

        // Test: velocity with cubic Hermite + explicit derivatives (d²/dt²)
        auto make_vel_ea_with_derivs = [&](const detail::BSplineFitResult& fit) {
            std::vector<double> times;
            times.push_back(fit.t_start);
            for (size_t k = fit.degree + 1;
                 k < fit.knots.size() - fit.degree - 1; ++k) {
                double tk = fit.knots[k];
                if (tk > fit.t_start + 1e-10 && tk < fit.t_end - 1e-10)
                    times.push_back(tk);
            }
            times.push_back(fit.t_end);

            std::vector<forces::EmpiricalAccelEntry> entries;
            std::vector<Vector3d> derivs;
            for (double t : times) {
                entries.emplace_back(t, detail::eval_d1(fit, t));
                derivs.push_back(detail::eval_d2(fit, t));
            }
            auto ea = std::make_shared<forces::EmpiricalAcceleration>();
            *ea = forces::EmpiricalAcceleration::from_entries_hermite(std::move(entries), std::move(derivs), forces::EmpiricalAccelFrame::GCRS);
            return ea;
        };

        // Test: velocity with denser output (midpoints)
        auto make_vel_ea_dense = [&](const detail::BSplineFitResult& fit) {
            std::vector<double> times;
            times.push_back(fit.t_start);
            for (size_t k = fit.degree + 1;
                 k < fit.knots.size() - fit.degree - 1; ++k) {
                double tk = fit.knots[k];
                if (tk > fit.t_start + 1e-10 && tk < fit.t_end - 1e-10) {
                    // Add midpoint before this knot
                    if (!times.empty()) {
                        double mid = 0.5 * (times.back() + tk);
                        times.push_back(mid);
                    }
                    times.push_back(tk);
                }
            }
            // Add midpoint before end
            if (!times.empty()) {
                double mid = 0.5 * (times.back() + fit.t_end);
                times.push_back(mid);
            }
            times.push_back(fit.t_end);

            std::vector<forces::EmpiricalAccelEntry> entries;
            for (double t : times) {
                entries.emplace_back(t, detail::eval_d1(fit, t));
            }
            auto ea = std::make_shared<forces::EmpiricalAcceleration>();
            *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS, forces::EAInterp::Linear);
            return ea;
        };

        auto eval_rmse = [&](const std::shared_ptr<forces::EmpiricalAcceleration>& ea) {
            TrajectoryConfig cfg = fast_config();
            cfg.empirical_accel = ea;
            Trajectory traj(s.t0, s.pos, s.vel, cfg);
            while (traj.epoch() - s.t0 < s.forced_duration) traj.step();
            TrajectoryInterpolator interp(traj);
            return compute_pos_rmse(interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        };

        std::cout << "\n    --- Single-iter velocity EA table comparison ---" << std::endl;
        for (int rate_idx = 0; rate_idx < 3; ++rate_idx) {
            const auto& fit = (rate_idx == 0) ? fit_10 : (rate_idx == 1) ? fit_100 : fit_300;
            const char* rlabel = (rate_idx == 0) ? "10hz" : (rate_idx == 1) ? "100hz" : "300hz";

            double rmse_cubic = eval_rmse(make_vel_ea(fit, false));
            double rmse_linear = eval_rmse(make_vel_ea(fit, true));
            double rmse_derivs = eval_rmse(make_vel_ea_with_derivs(fit));
            double rmse_dense = eval_rmse(make_vel_ea_dense(fit));

            std::cout << "    " << rlabel << ": cubic=" << std::fixed << std::setprecision(2)
                      << rmse_cubic << "m  linear=" << rmse_linear
                      << "m  hermite+derivs=" << rmse_derivs
                      << "m  dense_linear=" << rmse_dense << "m" << std::endl;
        }
    });

    // =========================================================================
    // Diagnostic: deep coefficient comparison 100 Hz vs 300 Hz
    // =========================================================================
    runner.add_test("diag_coefficient_sensitivity", [&]() {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        // Step 1: Propagate the same model trajectory (no EA) and compute
        // residuals at both 100 Hz and 300 Hz observation times
        Trajectory model_traj(s.t0, s.pos, s.vel, fast_config());
        while (model_traj.epoch() - s.t0 < s.forced_duration) model_traj.step();
        TrajectoryInterpolator model_interp(model_traj);

        auto compute_residuals = [&](const std::vector<std::pair<Epoch, Vector3d>>& pos_data) {
            std::vector<double> tau;
            std::vector<Vector3d> dp;
            for (const auto& obs : pos_data) {
                double t_rel = obs.first - s.t0;
                if (t_rel > eval_end) continue;
                auto ms = model_interp.get_state(obs.first);
                if (!ms) continue;
                Vector3d residual = obs.second - ms->position;
                residual = coords::itrf_to_gcrs_vector_full(residual, obs.first);
                tau.push_back(t_rel);
                dp.push_back(residual);
            }
            return std::make_pair(tau, dp);
        };

        auto [tau_100, dp_100] = compute_residuals(s.pos_data_100hz);
        auto [tau_300, dp_300] = compute_residuals(s.pos_data_300hz);

        std::cout << "    --- Iter 1 B-spline fit comparison ---" << std::endl;
        std::cout << "    100hz: n_obs=" << tau_100.size()
                  << " t=[" << std::fixed << std::setprecision(6) << tau_100.front()
                  << ", " << tau_100.back() << "]" << std::endl;
        std::cout << "    300hz: n_obs=" << tau_300.size()
                  << " t=[" << tau_300.front() << ", " << tau_300.back() << "]"
                  << std::endl;

        // Step 2: Call fit_bspline directly at both rates
        auto fit_100 = detail::fit_bspline(tau_100, dp_100, 2.0, 1e-10, 2);
        auto fit_300 = detail::fit_bspline(tau_300, dp_300, 2.0, 1e-10, 2);

        std::cout << "    100hz: n_basis=" << fit_100.n_basis
                  << " n_knots=" << fit_100.knots.size() << std::endl;
        std::cout << "    300hz: n_basis=" << fit_300.n_basis
                  << " n_knots=" << fit_300.knots.size() << std::endl;

        // Step 3: Compare knot vectors
        if (fit_100.knots.size() == fit_300.knots.size()) {
            double max_knot_diff = 0.0;
            for (size_t i = 0; i < fit_100.knots.size(); ++i) {
                max_knot_diff = (std::max)(max_knot_diff,
                    std::abs(fit_100.knots[i] - fit_300.knots[i]));
            }
            std::cout << "    max knot diff: " << std::scientific
                      << std::setprecision(6) << max_knot_diff << " s" << std::endl;
        } else {
            std::cout << "    WARNING: different knot vector sizes!" << std::endl;
        }

        // Step 4: Compare B-spline coefficients
        if (fit_100.n_basis == fit_300.n_basis) {
            double max_cx_diff = 0.0, max_cy_diff = 0.0, max_cz_diff = 0.0;
            double rms_cx_diff = 0.0, rms_cy_diff = 0.0, rms_cz_diff = 0.0;
            for (int j = 0; j < fit_100.n_basis; ++j) {
                double dx = std::abs(fit_100.cx(j) - fit_300.cx(j));
                double dy = std::abs(fit_100.cy(j) - fit_300.cy(j));
                double dz = std::abs(fit_100.cz(j) - fit_300.cz(j));
                max_cx_diff = (std::max)(max_cx_diff, dx);
                max_cy_diff = (std::max)(max_cy_diff, dy);
                max_cz_diff = (std::max)(max_cz_diff, dz);
                rms_cx_diff += dx * dx;
                rms_cy_diff += dy * dy;
                rms_cz_diff += dz * dz;
            }
            rms_cx_diff = std::sqrt(rms_cx_diff / fit_100.n_basis);
            rms_cy_diff = std::sqrt(rms_cy_diff / fit_100.n_basis);
            rms_cz_diff = std::sqrt(rms_cz_diff / fit_100.n_basis);

            std::cout << "    coeff max diff: cx=" << std::scientific << std::setprecision(4)
                      << max_cx_diff << " cy=" << max_cy_diff << " cz=" << max_cz_diff
                      << std::endl;
            std::cout << "    coeff rms diff: cx=" << rms_cx_diff
                      << " cy=" << rms_cy_diff << " cz=" << rms_cz_diff
                      << std::endl;

            // Print first 10 coefficients
            std::cout << "    first 10 cx coefficients:" << std::endl;
            for (int j = 0; j < (std::min)(10, fit_100.n_basis); ++j) {
                std::cout << "      c[" << j << "]: 100hz=" << std::setprecision(8)
                          << fit_100.cx(j) << " 300hz=" << fit_300.cx(j)
                          << " diff=" << std::scientific << std::setprecision(4)
                          << (fit_300.cx(j) - fit_100.cx(j)) << std::endl;
            }
        }

        // Step 5: Compare d²/dt² (EA) at common evaluation times
        std::cout << "    --- EA (d²/dt²) comparison at common times ---" << std::endl;
        double max_ea_diff = 0.0;
        int max_ea_diff_idx = 0;
        for (int i = 0; i <= 90; ++i) {
            double t = 2.0 * i;  // Every 2s
            if (t > eval_end) break;
            Vector3d ea_100 = detail::eval_d2(fit_100, t);
            Vector3d ea_300 = detail::eval_d2(fit_300, t);
            double diff = (ea_100 - ea_300).norm();
            if (diff > max_ea_diff) { max_ea_diff = diff; max_ea_diff_idx = i; }
            if (i < 5 || i == max_ea_diff_idx) {
                std::cout << "      t=" << std::fixed << std::setprecision(1) << t
                          << ": 100hz=(" << std::setprecision(6) << ea_100.x()
                          << "," << ea_100.y() << "," << ea_100.z()
                          << ") 300hz=(" << ea_300.x() << "," << ea_300.y()
                          << "," << ea_300.z() << ") diff=" << std::scientific
                          << std::setprecision(4) << diff << std::endl;
            }
        }
        std::cout << "    max EA diff: " << std::scientific << std::setprecision(6)
                  << max_ea_diff << " m/s^2 at t="
                  << std::fixed << std::setprecision(1)
                  << (2.0 * max_ea_diff_idx) << "s" << std::endl;

        // Step 6: Propagate both EA tables and compare RMSE
        // Use the d²/dt² directly from the fit (bypassing the EA table storage)
        auto make_ea_from_fit = [&](const detail::BSplineFitResult& fit) {
            std::vector<double> times;
            times.push_back(fit.t_start);
            for (size_t k = fit.degree + 1;
                 k < fit.knots.size() - fit.degree - 1; ++k) {
                double tk = fit.knots[k];
                if (tk > fit.t_start + 1e-10 && tk < fit.t_end - 1e-10)
                    times.push_back(tk);
            }
            times.push_back(fit.t_end);

            std::vector<forces::EmpiricalAccelEntry> entries;
            for (double t : times) {
                entries.emplace_back(t, detail::eval_d2(fit, t));
            }
            auto ea = std::make_shared<forces::EmpiricalAcceleration>();
            *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
            return ea;
        };

        auto ea_100 = make_ea_from_fit(fit_100);
        auto ea_300 = make_ea_from_fit(fit_300);

        // Also make linear versions
        auto make_ea_linear = [&](const detail::BSplineFitResult& fit) {
            std::vector<double> times;
            times.push_back(fit.t_start);
            for (size_t k = fit.degree + 1;
                 k < fit.knots.size() - fit.degree - 1; ++k) {
                double tk = fit.knots[k];
                if (tk > fit.t_start + 1e-10 && tk < fit.t_end - 1e-10)
                    times.push_back(tk);
            }
            times.push_back(fit.t_end);

            std::vector<forces::EmpiricalAccelEntry> entries;
            for (double t : times) {
                entries.emplace_back(t, detail::eval_d2(fit, t));
            }
            auto ea = std::make_shared<forces::EmpiricalAcceleration>();
            *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS, forces::EAInterp::Linear);
            return ea;
        };

        auto ea_100_lin = make_ea_linear(fit_100);
        auto ea_300_lin = make_ea_linear(fit_300);

        // Also: what if we use 100hz coefficients for 300hz evaluation?
        // (cross-test: same EA, different SSE eval)

        auto eval_rmse = [&](const std::shared_ptr<forces::EmpiricalAcceleration>& ea) {
            TrajectoryConfig cfg = fast_config();
            cfg.empirical_accel = ea;
            Trajectory traj(s.t0, s.pos, s.vel, cfg);
            while (traj.epoch() - s.t0 < s.forced_duration) traj.step();
            TrajectoryInterpolator interp(traj);
            return compute_pos_rmse(interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        };

        double rmse_100_cubic = eval_rmse(ea_100);
        double rmse_300_cubic = eval_rmse(ea_300);
        double rmse_100_linear = eval_rmse(ea_100_lin);
        double rmse_300_linear = eval_rmse(ea_300_lin);

        std::cout << "\n    --- Single-iteration reconstruction RMSE ---" << std::endl;
        std::cout << "    100hz coeffs, cubic table:  " << std::fixed << std::setprecision(4)
                  << rmse_100_cubic << " m" << std::endl;
        std::cout << "    300hz coeffs, cubic table:  " << rmse_300_cubic << " m" << std::endl;
        std::cout << "    100hz coeffs, linear table: " << rmse_100_linear << " m" << std::endl;
        std::cout << "    300hz coeffs, linear table: " << rmse_300_linear << " m" << std::endl;
    });

    runner.run();
    return print_final_summary();
}
