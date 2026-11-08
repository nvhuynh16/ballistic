// B-spline empirical-acceleration (EA) estimator on the glide fixture with the
// truth and the estimator integrating with the same 0.5 s RK45 maximum step
// ("matched steps"), forward from the start and backward from the end of the
// arc, with position, velocity and acceleration samples at 10/100/300 Hz.
//
// EA = the extra acceleration, beyond the force model, that the estimator fits so
// that a propagated trajectory matches the observations. The glide fixture is a
// synthetic reentry glide from 120 km driven by a known lift / lateral / return
// force (parameters below), sampled without noise.
//
// The cells assert only that an EA is returned; the re-propagation RMSE
// (root-mean-square position error against the truth) is printed for comparison.
// Heavy tier (about 70 s).

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

static TrajectoryConfig fast_config_matched() {
    auto cfg = fast_config();
    cfg.rk45.max_step_size = 0.5;
    cfg.rk45.initial_step_size = 0.5;
    return cfg;
}

static TrajectoryConfig make_forced_config(const Vector3d& return_dir) {
    TrajectoryConfig cfg = fast_config();
    cfg.rk45.max_step_size = 0.5;   // truth uses 0.5s steps
    cfg.rk45.initial_step_size = 0.5;

    auto ea = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;
    for (int i = 0; i <= 700; ++i) {
        double t = static_cast<double>(i);
        double gauss = std::exp(-0.5 * std::pow((t - FORCE_CENTER) / FORCE_WIDTH, 2));
        double lift = LIFT_PEAK * gauss;
        double lateral = LATERAL_AMP * gauss *
            std::sin(2.0 * constants::PI * t / LATERAL_PERIOD);
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

static std::pair<double, bool> propagate_to_ground(
    Trajectory& traj, const Epoch& t0, double max_duration)
{
    for (int step = 0; step < 2000000; ++step) {
        auto sr = traj.step();
        double elapsed = sr.epoch - t0;
        if (coords::ecef_to_geodetic(sr.position).altitude < 1000.0) return {elapsed, true};
        if (elapsed > max_duration) return {elapsed, false};
    }
    return {0.0, false};
}

static double bisect_impact(TrajectoryInterpolator& interp,
    const Epoch& t0, double approx_t)
{
    double lo = approx_t - 2.0, hi = approx_t;
    for (int i = 0; i < 50; ++i) {
        double mid = 0.5 * (lo + hi);
        auto state = interp.get_state(t0 + mid);
        if (!state) break;
        if (coords::ecef_to_geodetic(state->position).altitude < 1000.0) hi = mid;
        else lo = mid;
    }
    return hi;
}


struct MatchedSetup {
    Epoch t0;
    Vector3d pos, vel;
    double forced_duration;
    double sample_end;

    TrajectoryInterpolator* truth_interp;

    std::vector<std::pair<Epoch, Vector3d>> pos_data_10hz;
    std::vector<std::pair<Epoch, Vector3d>> pos_data_100hz;
    std::vector<std::pair<Epoch, Vector3d>> pos_data_300hz;

    // Backward
    Epoch impact_epoch;
    Vector3d impact_pos, impact_vel;
    std::vector<std::pair<Epoch, Vector3d>> bw_pos_data_10hz;
    std::vector<std::pair<Epoch, Vector3d>> bw_pos_data_100hz;
    std::vector<std::pair<Epoch, Vector3d>> bw_pos_data_300hz;

    std::unique_ptr<Trajectory> truth_traj_ptr;
    std::unique_ptr<TrajectoryInterpolator> truth_interp_ptr;
};

static MatchedSetup& get_setup() {
    static MatchedSetup s;
    static bool initialized = false;
    if (initialized) return s;
    initialized = true;

    std::cout << "  [setup] Building glide reentry truth (matched steps=0.5s)..." << std::endl;

    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    double v_circ = std::sqrt(constants::earth::GM / R);
    double omega_r = constants::earth::ANGULAR_VELOCITY * R;
    s.pos = Vector3d(R, 0.0, 0.0);
    s.vel = Vector3d(-30.0, 0.85 * v_circ - omega_r, 0.0);

    // Glide direction
    Trajectory glide_only(s.t0, s.pos, s.vel, make_forced_config(Vector3d::Zero()));
    for (int step = 0; step < 2000000; ++step) {
        auto sr = glide_only.step();
        if ((sr.epoch - s.t0) >= RETURN_CENTER + 10.0) break;
    }
    TrajectoryInterpolator glide_interp(glide_only);
    auto glide_at_peak = glide_interp.get_state(s.t0 + RETURN_CENTER);
    Vector3d return_dir = Vector3d::Zero();
    if (glide_at_peak) {
        Vector3d descent = -glide_at_peak->position.normalized();
        return_dir = (descent + 0.4 * Vector3d(0, 0, 1)).normalized();
    }

    // Truth trajectory (0.5 s steps, the same as the estimator)
    s.truth_traj_ptr = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, make_forced_config(return_dir));
    auto [fd, ok] = propagate_to_ground(*s.truth_traj_ptr, s.t0, 700.0);
    s.truth_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.truth_traj_ptr);
    if (ok) fd = bisect_impact(*s.truth_interp_ptr, s.t0, fd);
    s.forced_duration = fd;
    s.truth_interp = s.truth_interp_ptr.get();
    s.sample_end = fd - 10.0;

    std::cout << "  [setup] Forced duration: " << std::fixed << std::setprecision(1)
              << s.forced_duration << "s" << std::endl;

    // Forward position data
    auto sample_pos = [&](double dt) {
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = 0.0; t <= s.sample_end; t += dt) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) data.emplace_back(s.t0 + t, state->position);
        }
        return data;
    };
    s.pos_data_10hz = sample_pos(0.1);
    s.pos_data_100hz = sample_pos(0.01);
    s.pos_data_300hz = sample_pos(1.0 / 300.0);

    // Backward setup: start from the truth state 10 s before impact.
    auto impact_state = s.truth_interp->get_state(s.t0 + s.sample_end);
    if (impact_state) {
        s.impact_epoch = s.t0 + s.sample_end;
        s.impact_pos = impact_state->position;
        s.impact_vel = impact_state->velocity;

        auto sample_bw = [&](double dt) {
            std::vector<std::pair<Epoch, Vector3d>> data;
            for (double t = s.sample_end; t >= 0.0; t -= dt) {
                auto state = s.truth_interp->get_state(s.t0 + t);
                if (state) data.emplace_back(s.t0 + t, state->position);
            }
            return data;
        };
        s.bw_pos_data_10hz = sample_bw(0.1);
        s.bw_pos_data_100hz = sample_bw(0.01);
        s.bw_pos_data_300hz = sample_bw(1.0 / 300.0);
    }

    std::cout << "  [setup] Forward: " << s.pos_data_10hz.size() << " (10Hz) / "
              << s.pos_data_100hz.size() << " (100Hz) / "
              << s.pos_data_300hz.size() << " (300Hz)" << std::endl;
    std::cout << "  [setup] Backward: " << s.bw_pos_data_10hz.size() << " (10Hz) / "
              << s.bw_pos_data_100hz.size() << " (100Hz) / "
              << s.bw_pos_data_300hz.size() << " (300Hz)" << std::endl;

    return s;
}

int main() {
    ballistic::initialize({});
    TestRunner runner("B-Spline Matched Steps");

    // Forward test
    auto run_fwd = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters)
    {
        auto& s = get_setup();
        double eval_end = s.sample_end;

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config_matched();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config_matched();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) recon_traj.step();
        TrajectoryInterpolator recon_interp(recon_traj);

        double rmse = compute_pos_rmse(recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        std::cout << "    [" << label << "] recon_rmse: " << std::fixed
                  << std::setprecision(6) << rmse << " m, iters: "
                  << result.iterations_used << std::endl;
        return rmse;
    };

    // Backward test
    auto run_bwd = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters)
    {
        auto& s = get_setup();

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config_matched();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline(
            s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config_matched();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double t = 10.0; t <= s.sample_end; t += 1.0) {
            auto s_test = recon_interp.get_state(s.impact_epoch - t);
            auto s_truth = s.truth_interp->get_state(s.impact_epoch - t);
            if (!s_test || !s_truth) continue;
            sse += (s_test->position - s_truth->position).squaredNorm();
            ++count;
        }
        double rmse = count > 0 ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [" << label << "] recon_rmse: " << std::fixed
                  << std::setprecision(6) << rmse << " m, iters: "
                  << result.iterations_used << std::endl;
        return rmse;
    };

    // Velocity and acceleration data
    auto sample_vel = [&](double dt) {
        auto& s = get_setup();
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = 0.0; t <= s.sample_end; t += dt) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                StateVector s_ecef{state->position, state->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, s.t0 + t);
                data.emplace_back(s.t0 + t, s_eci.velocity);
            }
        }
        return data;
    };

    auto sample_accel = [&](double dt) {
        auto& s = get_setup();
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = 0.0; t <= s.sample_end; t += dt) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                StateVector s_ecef{state->position, state->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, s.t0 + t);
                Vector3d a_gcrs = s.truth_traj_ptr->force_model().acceleration(s.t0 + t, s_eci);
                data.emplace_back(s.t0 + t, a_gcrs);
            }
        }
        return data;
    };

    // Velocity/accel forward test
    auto run_vel_fwd = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters)
    {
        auto& s = get_setup();
        double eval_end = s.sample_end;

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.lambda_drift = 1e-2;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config_matched();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;

        auto result = estimate_empirical_acceleration_bspline_from_velocity(
            s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config_matched();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) recon_traj.step();
        TrajectoryInterpolator recon_interp(recon_traj);

        double rmse = compute_pos_rmse(recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        std::cout << "    [" << label << "] recon_rmse: " << std::fixed
                  << std::setprecision(6) << rmse << " m, iters: "
                  << result.iterations_used << std::endl;
        return rmse;
    };

    auto run_accel_fwd = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters)
    {
        auto& s = get_setup();
        double eval_end = s.sample_end;

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.max_duration_s = eval_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config_matched();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config_matched();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.t0, s.pos, s.vel, recon_cfg);
        while (recon_traj.epoch() - s.t0 < s.forced_duration) recon_traj.step();
        TrajectoryInterpolator recon_interp(recon_traj);

        double rmse = compute_pos_rmse(recon_interp, *s.truth_interp, s.t0, 10.0, eval_end, 1.0);
        std::cout << "    [" << label << "] recon_rmse: " << std::fixed
                  << std::setprecision(6) << rmse << " m, iters: "
                  << result.iterations_used << std::endl;
        return rmse;
    };

    // Forward position matched
    runner.add_test("fwd_10hz", [&]() {
        run_fwd("fwd_10hz", get_setup().pos_data_10hz, 5.0, 1e-8, 10);
    });
    runner.add_test("fwd_100hz", [&]() {
        run_fwd("fwd_100hz", get_setup().pos_data_100hz, 2.0, 1e-10, 10);
    });
    runner.add_test("fwd_300hz", [&]() {
        run_fwd("fwd_300hz", get_setup().pos_data_300hz, 2.0, 1e-10, 10);
    });

    // Backward matched
    runner.add_test("bwd_10hz", [&]() {
        run_bwd("bwd_10hz", get_setup().bw_pos_data_10hz, 5.0, 1e-8, 10);
    });
    runner.add_test("bwd_100hz", [&]() {
        run_bwd("bwd_100hz", get_setup().bw_pos_data_100hz, 2.0, 1e-10, 10);
    });
    runner.add_test("bwd_300hz", [&]() {
        run_bwd("bwd_300hz", get_setup().bw_pos_data_300hz, 2.0, 1e-10, 10);
    });

    // Backward velocity/acceleration data
    auto sample_bw_vel = [&](double dt) {
        auto& s = get_setup();
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = s.sample_end; t >= 0.0; t -= dt) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                StateVector s_ecef{state->position, state->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, s.t0 + t);
                data.emplace_back(s.t0 + t, s_eci.velocity);
            }
        }
        return data;
    };

    auto sample_bw_accel = [&](double dt) {
        auto& s = get_setup();
        std::vector<std::pair<Epoch, Vector3d>> data;
        for (double t = s.sample_end; t >= 0.0; t -= dt) {
            auto state = s.truth_interp->get_state(s.t0 + t);
            if (state) {
                StateVector s_ecef{state->position, state->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, s.t0 + t);
                Vector3d a_gcrs = s.truth_traj_ptr->force_model().acceleration(s.t0 + t, s_eci);
                data.emplace_back(s.t0 + t, a_gcrs);
            }
        }
        return data;
    };

    // Backward velocity/acceleration test helpers
    auto run_vel_bwd = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters)
    {
        auto& s = get_setup();

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.lambda_drift = 1e-2;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config_matched();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline_from_velocity(
            s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config_matched();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double t = 10.0; t <= s.sample_end; t += 1.0) {
            auto s_test = recon_interp.get_state(s.impact_epoch - t);
            auto s_truth = s.truth_interp->get_state(s.impact_epoch - t);
            if (!s_test || !s_truth) continue;
            sse += (s_test->position - s_truth->position).squaredNorm();
            ++count;
        }
        double rmse = count > 0 ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [" << label << "] recon_rmse: " << std::fixed
                  << std::setprecision(6) << rmse << " m, iters: "
                  << result.iterations_used << std::endl;
        return rmse;
    };

    auto run_accel_bwd = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters)
    {
        auto& s = get_setup();

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config_matched();
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = fast_config_matched();
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double t = 10.0; t <= s.sample_end; t += 1.0) {
            auto s_test = recon_interp.get_state(s.impact_epoch - t);
            auto s_truth = s.truth_interp->get_state(s.impact_epoch - t);
            if (!s_test || !s_truth) continue;
            sse += (s_test->position - s_truth->position).squaredNorm();
            ++count;
        }
        double rmse = count > 0 ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [" << label << "] recon_rmse: " << std::fixed
                  << std::setprecision(6) << rmse << " m, iters: "
                  << result.iterations_used << std::endl;
        return rmse;
    };

    // Forward velocity matched
    runner.add_test("vel_fwd_10hz", [&]() {
        run_vel_fwd("vel_fwd_10hz", sample_vel(0.1), 5.0, 1e-8, 10);
    });
    runner.add_test("vel_fwd_100hz", [&]() {
        run_vel_fwd("vel_fwd_100hz", sample_vel(0.01), 2.0, 1e-10, 10);
    });
    runner.add_test("vel_fwd_300hz", [&]() {
        run_vel_fwd("vel_fwd_300hz", sample_vel(1.0 / 300.0), 2.0, 1e-10, 10);
    });

    // Forward acceleration matched
    runner.add_test("acc_fwd_10hz", [&]() {
        run_accel_fwd("acc_fwd_10hz", sample_accel(0.1), 5.0, 1e-8, 10);
    });
    runner.add_test("acc_fwd_100hz", [&]() {
        run_accel_fwd("acc_fwd_100hz", sample_accel(0.01), 2.0, 1e-10, 10);
    });
    runner.add_test("acc_fwd_300hz", [&]() {
        run_accel_fwd("acc_fwd_300hz", sample_accel(1.0 / 300.0), 2.0, 1e-10, 10);
    });

    // Backward velocity matched
    runner.add_test("vel_bwd_10hz", [&]() {
        run_vel_bwd("vel_bwd_10hz", sample_bw_vel(0.1), 5.0, 1e-8, 10);
    });
    runner.add_test("vel_bwd_100hz", [&]() {
        run_vel_bwd("vel_bwd_100hz", sample_bw_vel(0.01), 2.0, 1e-10, 10);
    });
    runner.add_test("vel_bwd_300hz", [&]() {
        run_vel_bwd("vel_bwd_300hz", sample_bw_vel(1.0 / 300.0), 2.0, 1e-10, 10);
    });

    // Backward acceleration matched
    runner.add_test("acc_bwd_10hz", [&]() {
        run_accel_bwd("acc_bwd_10hz", sample_bw_accel(0.1), 5.0, 1e-8, 10);
    });
    runner.add_test("acc_bwd_100hz", [&]() {
        run_accel_bwd("acc_bwd_100hz", sample_bw_accel(0.01), 2.0, 1e-10, 10);
    });
    runner.add_test("acc_bwd_300hz", [&]() {
        run_accel_bwd("acc_bwd_300hz", sample_bw_accel(1.0 / 300.0), 2.0, 1e-10, 10);
    });

    // Step-size sweep for backward velocity (the best backward observation type
    // in the printed results).
    auto run_vel_bwd_step = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters, double step_size)
    {
        auto& s = get_setup();

        TrajectoryConfig tcfg = fast_config();
        tcfg.rk45.max_step_size = step_size;
        tcfg.rk45.initial_step_size = step_size;

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.lambda_drift = 1e-2;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = tcfg;
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline_from_velocity(
            s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = tcfg;
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double t = 10.0; t <= s.sample_end; t += 1.0) {
            auto s_test = recon_interp.get_state(s.impact_epoch - t);
            auto s_truth = s.truth_interp->get_state(s.impact_epoch - t);
            if (!s_test || !s_truth) continue;
            sse += (s_test->position - s_truth->position).squaredNorm();
            ++count;
        }
        double rmse = count > 0 ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [" << label << "] step=" << std::fixed << std::setprecision(1)
                  << step_size << "s rmse: " << std::setprecision(3) << rmse
                  << " m, iters: " << result.iterations_used << std::endl;
        return rmse;
    };

    for (double step : {0.5, 1.0, 2.0, 5.0, 10.0, 30.0}) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f", step);
        std::string tag(buf);

        runner.add_test("vel_bwd_300hz_step" + tag, [&, step]() {
            run_vel_bwd_step("vel_bwd_300hz", sample_bw_vel(1.0 / 300.0),
                             2.0, 1e-10, 10, step);
        });
    }

    // Also sweep step size for backward position at 300 Hz
    auto run_pos_bwd_step = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters, double step_size)
    {
        auto& s = get_setup();

        TrajectoryConfig tcfg = fast_config();
        tcfg.rk45.max_step_size = step_size;
        tcfg.rk45.initial_step_size = step_size;

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = tcfg;
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline(
            s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = tcfg;
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double t = 10.0; t <= s.sample_end; t += 1.0) {
            auto s_test = recon_interp.get_state(s.impact_epoch - t);
            auto s_truth = s.truth_interp->get_state(s.impact_epoch - t);
            if (!s_test || !s_truth) continue;
            sse += (s_test->position - s_truth->position).squaredNorm();
            ++count;
        }
        double rmse = count > 0 ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [" << label << "] step=" << std::fixed << std::setprecision(1)
                  << step_size << "s rmse: " << std::setprecision(3) << rmse
                  << " m, iters: " << result.iterations_used << std::endl;
        return rmse;
    };

    for (double step : {0.5, 1.0, 2.0, 5.0, 10.0, 30.0}) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f", step);
        std::string tag(buf);

        runner.add_test("pos_bwd_300hz_step" + tag, [&, step]() {
            run_pos_bwd_step("pos_bwd_300hz", get_setup().bw_pos_data_300hz,
                             2.0, 1e-10, 10, step);
        });
    }

    // Backward acceleration step size sweep
    auto run_accel_bwd_step = [](const std::string& label,
        const std::vector<std::pair<Epoch, Vector3d>>& data,
        double knots, double lambda, int iters, double step_size)
    {
        auto& s = get_setup();

        TrajectoryConfig tcfg = fast_config();
        tcfg.rk45.max_step_size = step_size;
        tcfg.rk45.initial_step_size = step_size;

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knots;
        cfg.lambda = lambda;
        cfg.max_duration_s = s.sample_end;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = tcfg;
        cfg.min_altitude_m = 500.0;
        cfg.max_iterations = iters;
        cfg.backward = true;

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null EA");

        TrajectoryConfig recon_cfg = tcfg;
        recon_cfg.empirical_accel = result.empirical_accel;
        Trajectory recon_traj(s.impact_epoch, s.impact_pos, s.impact_vel, recon_cfg);
        TrajectoryInterpolator recon_interp(recon_traj);
        recon_interp.extend_backward_to(s.impact_epoch - s.sample_end);

        double sse = 0.0; int count = 0;
        for (double t = 10.0; t <= s.sample_end; t += 1.0) {
            auto s_test = recon_interp.get_state(s.impact_epoch - t);
            auto s_truth = s.truth_interp->get_state(s.impact_epoch - t);
            if (!s_test || !s_truth) continue;
            sse += (s_test->position - s_truth->position).squaredNorm();
            ++count;
        }
        double rmse = count > 0 ? std::sqrt(sse / count) : 1e9;

        std::cout << "    [" << label << "] step=" << std::fixed << std::setprecision(1)
                  << step_size << "s rmse: " << std::setprecision(3) << rmse
                  << " m, iters: " << result.iterations_used << std::endl;
        return rmse;
    };

    for (double step : {0.5, 1.0, 2.0, 5.0, 10.0, 30.0}) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f", step);
        std::string tag(buf);

        runner.add_test("acc_bwd_300hz_step" + tag, [&, step]() {
            run_accel_bwd_step("acc_bwd_300hz", sample_bw_accel(1.0 / 300.0),
                               2.0, 1e-10, 10, step);
        });
    }

    runner.run();
    return print_final_summary();
}
