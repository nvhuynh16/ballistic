// Integration test: the B-spline empirical-acceleration (EA) estimator on the
// glide fixture (a synthetic reentry glide with known extra forces: lift,
// a lateral oscillation and a late "return" push, defined below).
//
// EA = the extra acceleration the force model is missing; the estimator fits
// it as a B-spline so that a trajectory propagated with it matches the
// observations. Each cell samples the truth trajectory at 1/10/100/300 Hz
// (positions, velocities or accelerations), estimates the EA, re-propagates
// with it, and requires the position RMSE (root-mean-square error) against
// the truth to beat the baseline (the same start propagated with no EA).
// The point of the rate sweep: a spline fit should NOT get worse as the
// observation rate rises, unlike estimators that difference neighboring
// samples (their conditioning degrades as the sample spacing shrinks).

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

// =============================================================================
// Glide force parameters: Gaussian lift / lateral / return pulses [m/s^2, s]
// =============================================================================

constexpr double LIFT_PEAK = 8.0;
constexpr double LATERAL_AMP = 3.0;
constexpr double FORCE_CENTER = 70.0;
constexpr double FORCE_WIDTH = 35.0;
constexpr double LATERAL_PERIOD = 100.0;
constexpr double RETURN_MAGNITUDE = 6.0;
constexpr double RETURN_CENTER = 420.0;
constexpr double RETURN_WIDTH = 80.0;

// =============================================================================
// Config builders
// =============================================================================

static TrajectoryConfig make_ballistic_config() {
    TrajectoryConfig c;
    c.gravity_degree = 8;
    c.enable_third_body = false;
    c.enable_relativistic = false;
    c.enable_solid_tides = false;

    return c;
}

static TrajectoryConfig fast_config() { return make_ballistic_config(); }

static TrajectoryConfig make_forced_config(const Vector3d& return_dir_gcrs) {
    TrajectoryConfig c = make_ballistic_config();
    c.rk45.max_step_size = 0.5;

    auto accel = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;

    auto glide_bump = [](double t) {
        double x = (t - FORCE_CENTER) / FORCE_WIDTH;
        return std::exp(-0.5 * x * x);
    };

    auto return_bump = [](double t) {
        double x = (t - RETURN_CENTER) / RETURN_WIDTH;
        return std::exp(-0.5 * x * x);
    };

    const double omega = 2.0 * constants::PI / LATERAL_PERIOD;

    for (double t = 0.0; t <= 700.0; t += 2.0) {
        double sb = glide_bump(t);
        double lift = LIFT_PEAK * sb;
        double lat_y = LATERAL_AMP * sb * std::sin(omega * t);
        double lat_z = LATERAL_AMP * sb * std::cos(omega * t);
        Vector3d glide_force(lift, lat_y, lat_z);

        double rb = (t >= 280.0) ? return_bump(t) : 0.0;
        Vector3d return_force = RETURN_MAGNITUDE * rb * return_dir_gcrs;

        entries.emplace_back(t, glide_force + return_force);
    }

    *accel = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
    c.empirical_accel = accel;
    return c;
}

// =============================================================================
// Helpers
// =============================================================================

static std::pair<double, bool> propagate_to_ground(
    Trajectory& traj, const Epoch& t0, double timeout_s)
{
    for (int step = 0; step < 50000; ++step) {
        auto sr = traj.step();
        auto geo = coords::ecef_to_geodetic(sr.position);
        double dt = sr.epoch - t0;
        if (geo.altitude <= 1000.0) return {dt, true};
        if (dt > timeout_s) return {dt, false};
    }
    return {0.0, false};
}

static double bisect_impact(
    TrajectoryInterpolator& interp, const Epoch& t0, double duration)
{
    double t_lo = (std::max)(0.0, duration - 30.0);
    double t_hi = duration;
    for (int iter = 0; iter < 50; ++iter) {
        double t_mid = 0.5 * (t_lo + t_hi);
        auto state = interp.get_state(t0 + t_mid);
        if (!state) break;
        auto geo = coords::ecef_to_geodetic(state->position);
        if (geo.altitude > 1000.0) t_lo = t_mid;
        else t_hi = t_mid;
        if (t_hi - t_lo < 0.001) break;
    }
    return 0.5 * (t_lo + t_hi);
}


// =============================================================================
// Lazy-initialized shared setup (B-spline needs only position data)
// =============================================================================

struct BSplineSetup {
    Epoch t0;
    Vector3d pos;
    Vector3d vel;
    double forced_duration;

    TrajectoryInterpolator* truth_interp;
    TrajectoryInterpolator* base_interp;

    std::vector<std::pair<Epoch, Vector3d>> pos_data_1hz;
    std::vector<std::pair<Epoch, Vector3d>> pos_data_10hz;
    std::vector<std::pair<Epoch, Vector3d>> pos_data_100hz;
    std::vector<std::pair<Epoch, Vector3d>> pos_data_300hz;

    std::vector<std::pair<Epoch, Vector3d>> vel_data_1hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_data_10hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_data_100hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_data_300hz;

    std::vector<std::pair<Epoch, Vector3d>> accel_data_1hz;
    std::vector<std::pair<Epoch, Vector3d>> accel_data_10hz;
    std::vector<std::pair<Epoch, Vector3d>> accel_data_100hz;
    std::vector<std::pair<Epoch, Vector3d>> accel_data_300hz;

    // Backward estimation fields
    Epoch impact_epoch;
    Vector3d impact_pos, impact_vel;
    double sample_end;
    std::vector<std::pair<Epoch, Vector3d>> bw_pos_data_300hz;
    TrajectoryInterpolator* bw_base_interp;

    std::unique_ptr<Trajectory> truth_traj_ptr;
    std::unique_ptr<Trajectory> base_traj_ptr;
    std::unique_ptr<TrajectoryInterpolator> truth_interp_ptr;
    std::unique_ptr<TrajectoryInterpolator> base_interp_ptr;
    std::unique_ptr<Trajectory> bw_base_traj_ptr;
    std::unique_ptr<TrajectoryInterpolator> bw_base_interp_ptr;
};

static BSplineSetup& get_setup() {
    static BSplineSetup s;
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

    // Return direction
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

    // Sample position data at multiple rates
    double sample_end = s.forced_duration - 10.0;

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

    // === Backward setup ===
    double T = s.forced_duration;
    s.sample_end = T - 10.0;

    auto impact_state = s.truth_interp->get_state(s.t0 + s.sample_end);
    if (impact_state) {
        s.impact_epoch = s.t0 + s.sample_end;
        s.impact_pos = impact_state->position;
        s.impact_vel = impact_state->velocity;

        // Backward observations: same positions, sorted DECREASING by epoch
        for (double t = s.sample_end; t >= 0.0; t -= (1.0 / 300.0)) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                s.bw_pos_data_300hz.emplace_back(s.t0 + t, state->position);
            }
        }

        // Backward baseline
        s.bw_base_traj_ptr = std::make_unique<Trajectory>(
            s.impact_epoch, s.impact_pos, s.impact_vel, make_ballistic_config());
        s.bw_base_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.bw_base_traj_ptr);
        s.bw_base_interp_ptr->extend_backward_to(s.impact_epoch - s.sample_end);
        s.bw_base_interp = s.bw_base_interp_ptr.get();

        std::cout << "  [setup] Backward: " << s.bw_pos_data_300hz.size()
                  << " obs, impact at "
                  << std::fixed << std::setprecision(1)
                  << coords::ecef_to_geodetic(s.impact_pos).altitude / 1000.0
                  << " km" << std::endl;
    }

    return s;
}

// =============================================================================
// Test runner
// =============================================================================

int main() {
    ballistic::initialize({});

    TestRunner runner("B-Spline Collocation Estimator");

    // -------------------------------------------------------------------------
    // Helper: estimate from position samples, re-propagate with the EA, and
    // require the position RMSE vs truth to beat the no-EA baseline.
    // knot_spacing [s] sets the spline resolution; lambda is the smoothing
    // (regularization) weight; max_iterations caps the fit-and-repropagate
    // loop.
    // -------------------------------------------------------------------------
    auto run_bspline_test = [](
        const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& pos_data,
        double knot_spacing,
        double lambda,
        int max_iterations)
    {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        double base_rmse = compute_pos_rmse(
            *s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

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

        TEST_ASSERT_MSG(result.empirical_accel != nullptr,
            label + ": estimator returned null empirical_accel");

        // Reconstruct
        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) {
            recon_traj.step();
        }
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(
            recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        double improvement = (1.0 - recon_rmse / base_rmse) * 100.0;
        std::cout << "    [" << label << "] base_rmse:  " << std::fixed
                  << std::setprecision(1) << base_rmse << " m" << std::endl;
        std::cout << "    [" << label << "] recon_rmse: " << recon_rmse << " m" << std::endl;
        std::cout << "    [" << label << "] improvement: " << std::setprecision(1)
                  << improvement << "%" << std::endl;
        std::cout << "    [" << label << "] iterations: " << result.iterations_used
                  << ", nodes: " << result.empirical_accel->keys().size() << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse,
            label + " reconstruction (" + std::to_string(recon_rmse)
            + "m) did not beat baseline (" + std::to_string(base_rmse) + "m)");

        return recon_rmse;
    };

    // -------------------------------------------------------------------------
    // Test 1: B-spline 1Hz
    // -------------------------------------------------------------------------
    runner.add_test("bspline_1hz", [&]() {
        run_bspline_test("bspline_1hz", get_setup().pos_data_1hz,
                         20.0, 1e-6, 1);
    });

    // -------------------------------------------------------------------------
    // Test 2: B-spline 10Hz
    // -------------------------------------------------------------------------
    runner.add_test("bspline_10hz", [&]() {
        run_bspline_test("bspline_10hz", get_setup().pos_data_10hz,
                         20.0, 1e-6, 1);
    });

    // -------------------------------------------------------------------------
    // Test 3: B-spline 100Hz — key: should not degrade
    // -------------------------------------------------------------------------
    runner.add_test("bspline_100hz", [&]() {
        run_bspline_test("bspline_100hz", get_setup().pos_data_100hz,
                         20.0, 1e-6, 1);
    });

    // -------------------------------------------------------------------------
    // Test 4: B-spline 300Hz — key: should match or beat 100Hz
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz", [&]() {
        run_bspline_test("bspline_300hz", get_setup().pos_data_300hz,
                         20.0, 1e-6, 1);
    });

    // -------------------------------------------------------------------------
    // Test 5: B-spline 300Hz with 3 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_iter3", [&]() {
        run_bspline_test("bspline_300hz_iter3", get_setup().pos_data_300hz,
                         20.0, 1e-6, 3);
    });

    // -------------------------------------------------------------------------
    // Test 6: B-spline 300Hz with smaller knot spacing (10s)
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_10s_knots", [&]() {
        run_bspline_test("bspline_300hz_10s", get_setup().pos_data_300hz,
                         10.0, 1e-6, 1);
    });

    // -------------------------------------------------------------------------
    // Test 7: B-spline 300Hz, 5s knots, 5 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_5s_iter5", [&]() {
        run_bspline_test("bspline_300hz_5s_iter5", get_setup().pos_data_300hz,
                         5.0, 1e-8, 5);
    });

    // -------------------------------------------------------------------------
    // Test 8: B-spline 300Hz, 2s knots, 5 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_2s_iter5", [&]() {
        run_bspline_test("bspline_300hz_2s_iter5", get_setup().pos_data_300hz,
                         2.0, 1e-10, 5);
    });

    // -------------------------------------------------------------------------
    // Test 9: B-spline 1Hz, 20s knots, 5 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_1hz_iter5", [&]() {
        run_bspline_test("bspline_1hz_iter5", get_setup().pos_data_1hz,
                         20.0, 1e-6, 5);
    });

    // -------------------------------------------------------------------------
    // Test 10: B-spline 10Hz, 5s knots, 5 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_10hz_5s_iter5", [&]() {
        run_bspline_test("bspline_10hz_5s_iter5", get_setup().pos_data_10hz,
                         5.0, 1e-8, 5);
    });

    // -------------------------------------------------------------------------
    // Test 11: B-spline 300Hz, 2s knots, 10 iterations (push convergence)
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_2s_iter10", [&]() {
        run_bspline_test("bspline_300hz_2s_iter10", get_setup().pos_data_300hz,
                         2.0, 1e-10, 10);
    });

    // -------------------------------------------------------------------------
    // Test 12: B-spline 100Hz, 2s knots, 10 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_100hz_2s_iter10", [&]() {
        run_bspline_test("bspline_100hz_2s_iter10", get_setup().pos_data_100hz,
                         2.0, 1e-10, 10);
    });

    // -------------------------------------------------------------------------
    // Test 13: B-spline 300Hz, 1s knots, 10 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_1s_iter10", [&]() {
        run_bspline_test("bspline_300hz_1s_iter10", get_setup().pos_data_300hz,
                         1.0, 1e-12, 10);
    });

    // -------------------------------------------------------------------------
    // Test 14: B-spline 300Hz, 0.5s knots, lambda=1e-6, 10 iterations
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_05s_iter10", [&]() {
        run_bspline_test("bspline_300hz_05s_iter10", get_setup().pos_data_300hz,
                         0.5, 1e-6, 10);
    });

    // -------------------------------------------------------------------------
    // Test 15: B-spline 300Hz, 1s knots, 20 iterations (push sub-meter)
    // -------------------------------------------------------------------------
    runner.add_test("bspline_300hz_1s_iter20", [&]() {
        run_bspline_test("bspline_300hz_1s_iter20", get_setup().pos_data_300hz,
                         1.0, 1e-12, 20);
    });

    // =========================================================================
    // Diagnostic: what RMSE does the EXACT truth EA table give? Prints the
    // floor set by integration and table interpolation alone (forward and
    // backward); the assertion is only a sanity bound.
    // =========================================================================

    runner.add_test("diag_exact_ea", [&]() {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        // Build the EXACT same EA table as the truth trajectory uses
        // (same formula, same 2s sampling, same set_entries call)
        auto truth_cfg = s.truth_traj_ptr->config();

        // Propagate FORWARD with this exact EA — use truth step size!
        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = truth_cfg.empirical_accel;
        recon_cfg.rk45.max_step_size = 0.5;  // match truth
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) {
            recon_traj.step();
        }
        TrajectoryInterpolator recon_interp(recon_traj);

        double rmse = compute_pos_rmse(
            recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        std::cout << "    [diag] Exact truth EA table RMSE: " << std::fixed
                  << std::setprecision(6) << rmse << " m" << std::endl;

        std::cout << "    [diag] Forward exact EA, matched steps: "
                  << std::fixed << std::setprecision(6) << rmse << " m" << std::endl;

        // Now test BACKWARD with truth EA (need to re-key for backward epoch0)
        // Include ALL truth EA entries (full 0-700s range) to preserve cubic
        // spline boundary conditions. Keys shift by -(sample_end).
        {
            auto& truth_ea = truth_cfg.empirical_accel;
            const auto& truth_keys = truth_ea->keys();
            std::vector<forces::EmpiricalAccelEntry> bw_entries;
            for (double key : truth_keys) {
                double bw_key = key - s.sample_end;  // shift: 0→-484, 700→216
                Vector3d val = truth_ea->accel_at((s.t0 + key) - (s.t0));
                bw_entries.emplace_back(bw_key, val);
            }
            auto bw_ea = std::make_shared<forces::EmpiricalAcceleration>();
            *bw_ea = forces::EmpiricalAcceleration::from_entries(std::move(bw_entries), forces::EmpiricalAccelFrame::GCRS);

            // Propagate backward with exact truth EA
            TrajectoryConfig bw_cfg = fast_config();
            bw_cfg.empirical_accel = bw_ea;
            bw_cfg.rk45.max_step_size = 0.5;
            Trajectory bw_traj(s.impact_epoch, s.impact_pos, s.impact_vel, bw_cfg);
            TrajectoryInterpolator bw_interp(bw_traj);
            bw_interp.extend_backward_to(s.impact_epoch - s.sample_end);

            double bw_sse = 0.0; int bw_count = 0;
            for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
                Epoch real_ep = s.impact_epoch - sec;
                auto r = bw_interp.get_state(real_ep);
                auto t = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
                if (r && t) {
                    bw_sse += (r->position - t->position).squaredNorm();
                    ++bw_count;
                }
            }
            double bw_rmse = (bw_count > 0) ? std::sqrt(bw_sse / bw_count) : 1e9;
            std::cout << "    [diag] Backward exact EA, matched steps: "
                      << std::fixed << std::setprecision(6) << bw_rmse << " m"
                      << " (" << bw_count << " points)" << std::endl;

            // Test with ea_reference_epoch override: use ORIGINAL truth EA, no re-keying
            {
                TrajectoryConfig bw_orig_cfg = fast_config();
                bw_orig_cfg.empirical_accel = truth_ea;  // original truth EA (keys 0-700)
                bw_orig_cfg.ea_reference_epoch = s.t0;            // look up keys relative to fwd_t0
                bw_orig_cfg.rk45.max_step_size = 0.5;
                Trajectory bw_orig_traj(s.impact_epoch, s.impact_pos, s.impact_vel, bw_orig_cfg);
                TrajectoryInterpolator bw_orig_interp(bw_orig_traj);
                bw_orig_interp.extend_backward_to(s.impact_epoch - s.sample_end);

                double bw_orig_sse = 0.0; int bw_orig_count = 0;
                for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
                    Epoch real_ep = s.impact_epoch - sec;
                    auto r = bw_orig_interp.get_state(real_ep);
                    auto t3 = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
                    if (r && t3) {
                        bw_orig_sse += (r->position - t3->position).squaredNorm();
                        ++bw_orig_count;
                    }
                }
                double bw_orig_rmse = (bw_orig_count > 0) ? std::sqrt(bw_orig_sse / bw_orig_count) : 1e9;
                std::cout << "    [diag] Backward ORIGINAL EA + ea_reference_epoch, matched: "
                          << std::fixed << std::setprecision(6) << bw_orig_rmse << " m" << std::endl;
            }

            // Also test with DEFAULT steps
            TrajectoryConfig bw_def_cfg = fast_config();
            bw_def_cfg.empirical_accel = bw_ea;
            Trajectory bw_def_traj(s.impact_epoch, s.impact_pos, s.impact_vel, bw_def_cfg);
            TrajectoryInterpolator bw_def_interp(bw_def_traj);
            bw_def_interp.extend_backward_to(s.impact_epoch - s.sample_end);

            double bw_def_sse = 0.0; int bw_def_count = 0;
            for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
                Epoch real_ep = s.impact_epoch - sec;
                auto r = bw_def_interp.get_state(real_ep);
                auto t2 = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
                if (r && t2) {
                    bw_def_sse += (r->position - t2->position).squaredNorm();
                    ++bw_def_count;
                }
            }
            double bw_def_rmse = (bw_def_count > 0) ? std::sqrt(bw_def_sse / bw_def_count) : 1e9;
            std::cout << "    [diag] Backward exact EA, default steps: "
                      << std::fixed << std::setprecision(6) << bw_def_rmse << " m" << std::endl;
        }

        TEST_ASSERT_MSG(rmse < 1000.0, "Sanity check failed");
    });

    // =========================================================================
    // Key test: B-spline with the integrator step matched to the truth's
    // (0.5 s max step), so step-size differences do not limit the fit. The
    // bound 141000 m is a loose no-worse-than-baseline check.
    // =========================================================================

    runner.add_test("bspline_300hz_2s_matched", [&]() {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.trajectory_config.rk45.max_step_size = 0.5;  // match truth!
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 10;

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, s.pos_data_300hz, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, "null EA");

        // Reconstruct with SAME step size
        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        recon_cfg.rk45.max_step_size = 0.5;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) {
            recon_traj.step();
        }
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(
            recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        std::cout << "    [matched] recon_rmse: " << std::fixed
                  << std::setprecision(6) << recon_rmse << " m" << std::endl;

        TEST_ASSERT_MSG(recon_rmse < 141000.0, "Did not beat baseline");
    });

    // =========================================================================
    // Backward with matched step size: estimate from the impact point
    // backwards in time. Bound 176000 m: loose no-worse-than-baseline check.
    // =========================================================================

    runner.add_test("bspline_bw_300hz_2s_matched", [&]() {
        auto& s = get_setup();

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.trajectory_config.rk45.max_step_size = 0.5;  // match truth
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 10;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline(
            s.impact_epoch, s.impact_pos, s.impact_vel,
            s.bw_pos_data_300hz, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, "null EA");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        recon_cfg.rk45.max_step_size = 0.5;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
            Epoch real_ep = s.impact_epoch - sec;
            auto r = recon_interp.get_state(real_ep);
            auto t = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
            if (r && t) { sse += (r->position - t->position).squaredNorm(); ++count; }
        }
        double recon_rmse = (count > 0) ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [bw_matched] recon_rmse: " << std::fixed
                  << std::setprecision(6) << recon_rmse << " m" << std::endl;

        TEST_ASSERT_MSG(recon_rmse < 176000.0, "Did not beat baseline");
    });

    // =========================================================================
    // 300 Hz, 2 s knots, 10 iterations; must beat the no-EA baseline.
    // =========================================================================

    runner.add_test("bspline_300hz_2s_gn3", [&]() {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        double base_rmse = compute_pos_rmse(
            *s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 10;
        // No Gauss-Newton (GN) refinement is run here despite the cell name:
        // this low-level estimator has no inline GN, and the cell does not call
        // refine_ea_gn. It checks the plain fit with these settings.

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, s.pos_data_300hz, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr,
            "GN: null empirical_accel");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) {
            recon_traj.step();
        }
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(
            recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        std::cout << "    [gn3] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << std::setprecision(3)
                  << recon_rmse << "m" << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse, "GN did not beat baseline");
    });

    // =========================================================================
    // Backward B-spline tests: estimate starting from the impact state and
    // propagating backwards; must beat the backward no-EA baseline.
    // =========================================================================

    runner.add_test("bspline_bw_300hz_2s_iter10", [&]() {
        auto& s = get_setup();

        // Backward baseline RMSE
        double base_rmse = 0.0;
        {
            double sse = 0.0;
            int count = 0;
            for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
                Epoch real_ep = s.impact_epoch - sec;
                double fwd_t = real_ep - s.t0;
                auto recon_st = s.bw_base_interp->get_state(real_ep);
                auto truth_st = s.truth_interp->get_state(s.t0 + fwd_t);
                if (!recon_st || !truth_st) continue;
                sse += (recon_st->position - truth_st->position).squaredNorm();
                ++count;
            }
            base_rmse = (count > 0) ? std::sqrt(sse / count) : 1e9;
        }

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 10;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline(
            s.impact_epoch, s.impact_pos, s.impact_vel,
            s.bw_pos_data_300hz, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr,
            "backward bspline: null empirical_accel");

        // Backward reconstruction
        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0;
        int count = 0;
        for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
            Epoch real_ep = s.impact_epoch - sec;
            double fwd_t = real_ep - s.t0;
            auto recon_st = recon_interp.get_state(real_ep);
            auto truth_st = s.truth_interp->get_state(s.t0 + fwd_t);
            if (!recon_st || !truth_st) continue;
            sse += (recon_st->position - truth_st->position).squaredNorm();
            ++count;
        }
        double recon_rmse = (count > 0) ? std::sqrt(sse / count) : 1e9;

        double improvement = (1.0 - recon_rmse / base_rmse) * 100.0;
        std::cout << "    [bw_300hz_2s] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << recon_rmse
                  << "m, improvement: " << improvement << "%" << std::endl;
        std::cout << "    iterations: " << result.iterations_used
                  << ", nodes: " << result.empirical_accel->keys().size() << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse,
            "Backward bspline did not beat baseline");
    });

    runner.add_test("bspline_bw_300hz_1s_iter10", [&]() {
        auto& s = get_setup();

        double base_rmse = 0.0;
        {
            double sse = 0.0; int count = 0;
            for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
                Epoch real_ep = s.impact_epoch - sec;
                auto r = s.bw_base_interp->get_state(real_ep);
                auto t = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
                if (r && t) { sse += (r->position - t->position).squaredNorm(); ++count; }
            }
            base_rmse = (count > 0) ? std::sqrt(sse / count) : 1e9;
        }

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 1.0;
        cfg.lambda = 1e-12;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 10;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline(
            s.impact_epoch, s.impact_pos, s.impact_vel,
            s.bw_pos_data_300hz, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, "null EA");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
            Epoch real_ep = s.impact_epoch - sec;
            auto r = recon_interp.get_state(real_ep);
            auto t = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
            if (r && t) { sse += (r->position - t->position).squaredNorm(); ++count; }
        }
        double recon_rmse = (count > 0) ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [bw_1s] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << recon_rmse << "m" << std::endl;
        TEST_ASSERT_MSG(recon_rmse < base_rmse, "backward bspline 1s did not beat baseline");
    });

    runner.add_test("bspline_bw_300hz_2s_iter20", [&]() {
        auto& s = get_setup();

        double base_rmse = 0.0;
        {
            double sse = 0.0; int count = 0;
            for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
                Epoch real_ep = s.impact_epoch - sec;
                auto r = s.bw_base_interp->get_state(real_ep);
                auto t = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
                if (r && t) { sse += (r->position - t->position).squaredNorm(); ++count; }
            }
            base_rmse = (count > 0) ? std::sqrt(sse / count) : 1e9;
        }

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 20;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline(
            s.impact_epoch, s.impact_pos, s.impact_vel,
            s.bw_pos_data_300hz, cfg);

        TEST_ASSERT_MSG(result.empirical_accel != nullptr, "null EA");

        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double sec = 10.0; sec <= s.sample_end - 10.0; sec += 1.0) {
            Epoch real_ep = s.impact_epoch - sec;
            auto r = recon_interp.get_state(real_ep);
            auto t = s.truth_interp->get_state(s.t0 + (real_ep - s.t0));
            if (r && t) { sse += (r->position - t->position).squaredNorm(); ++count; }
        }
        double recon_rmse = (count > 0) ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [bw_2s_20] base: " << std::fixed << std::setprecision(1)
                  << base_rmse << "m, recon: " << recon_rmse << "m" << std::endl;
        TEST_ASSERT_MSG(recon_rmse < base_rmse, "backward bspline 2s/20 did not beat baseline");
    });

    // =========================================================================
    // Velocity B-spline tests: the spline is fitted to velocity observations;
    // each must beat the no-EA baseline.
    // =========================================================================

    auto run_bspline_vel_test = [](
        const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& vel_data,
        double knot_spacing,
        double lambda,
        int max_iterations)
    {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        double base_rmse = compute_pos_rmse(
            *s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

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

        TEST_ASSERT_MSG(result.empirical_accel != nullptr,
            label + ": estimator returned null empirical_accel");

        // Reconstruct
        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) {
            recon_traj.step();
        }
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(
            recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        double improvement = (1.0 - recon_rmse / base_rmse) * 100.0;
        std::cout << "    [" << label << "] base_rmse:  " << std::fixed
                  << std::setprecision(1) << base_rmse << " m" << std::endl;
        std::cout << "    [" << label << "] recon_rmse: " << recon_rmse << " m" << std::endl;
        std::cout << "    [" << label << "] improvement: " << std::setprecision(1)
                  << improvement << "%" << std::endl;
        std::cout << "    [" << label << "] iterations: " << result.iterations_used
                  << ", nodes: " << result.empirical_accel->keys().size() << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse,
            label + " reconstruction (" + std::to_string(recon_rmse)
            + "m) did not beat baseline (" + std::to_string(base_rmse) + "m)");

        return recon_rmse;
    };

    runner.add_test("bspline_vel_1hz", [&]() {
        run_bspline_vel_test("bspline_vel_1hz", get_setup().vel_data_1hz,
                             20.0, 1e-6, 5);
    });

    runner.add_test("bspline_vel_10hz", [&]() {
        run_bspline_vel_test("bspline_vel_10hz", get_setup().vel_data_10hz,
                             5.0, 1e-8, 5);
    });

    runner.add_test("bspline_vel_100hz", [&]() {
        run_bspline_vel_test("bspline_vel_100hz", get_setup().vel_data_100hz,
                             2.0, 1e-10, 10);
    });

    runner.add_test("bspline_vel_300hz", [&]() {
        run_bspline_vel_test("bspline_vel_300hz", get_setup().vel_data_300hz,
                             2.0, 1e-10, 10);
    });

    // =========================================================================
    // Acceleration B-spline tests: the spline is fitted to acceleration
    // observations; each must beat the no-EA baseline.
    // =========================================================================

    auto run_bspline_accel_test = [](
        const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& accel_data,
        double knot_spacing,
        double lambda,
        int max_iterations)
    {
        auto& s = get_setup();
        double eval_end = s.forced_duration - 10.0;

        double base_rmse = compute_pos_rmse(
            *s.base_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

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

        TEST_ASSERT_MSG(result.empirical_accel != nullptr,
            label + ": estimator returned null empirical_accel");

        // Reconstruct
        TrajectoryConfig recon_cfg = fast_config();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) {
            recon_traj.step();
        }
        TrajectoryInterpolator recon_interp(recon_traj);

        double recon_rmse = compute_pos_rmse(
            recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);

        double improvement = (1.0 - recon_rmse / base_rmse) * 100.0;
        std::cout << "    [" << label << "] base_rmse:  " << std::fixed
                  << std::setprecision(1) << base_rmse << " m" << std::endl;
        std::cout << "    [" << label << "] recon_rmse: " << recon_rmse << " m" << std::endl;
        std::cout << "    [" << label << "] improvement: " << std::setprecision(1)
                  << improvement << "%" << std::endl;
        std::cout << "    [" << label << "] iterations: " << result.iterations_used
                  << ", nodes: " << result.empirical_accel->keys().size() << std::endl;

        TEST_ASSERT_MSG(recon_rmse < base_rmse,
            label + " reconstruction (" + std::to_string(recon_rmse)
            + "m) did not beat baseline (" + std::to_string(base_rmse) + "m)");

        return recon_rmse;
    };

    runner.add_test("bspline_accel_1hz", [&]() {
        run_bspline_accel_test("bspline_accel_1hz", get_setup().accel_data_1hz,
                               20.0, 1e-6, 5);
    });

    runner.add_test("bspline_accel_10hz", [&]() {
        run_bspline_accel_test("bspline_accel_10hz", get_setup().accel_data_10hz,
                               5.0, 1e-8, 5);
    });

    runner.add_test("bspline_accel_100hz", [&]() {
        run_bspline_accel_test("bspline_accel_100hz", get_setup().accel_data_100hz,
                               2.0, 1e-10, 10);
    });

    runner.add_test("bspline_accel_300hz", [&]() {
        run_bspline_accel_test("bspline_accel_300hz", get_setup().accel_data_300hz,
                               2.0, 1e-10, 10);
    });

    runner.run();
    return print_final_summary();
}
