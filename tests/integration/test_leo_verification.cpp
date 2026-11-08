// B-spline empirical-acceleration (EA) estimator on a synthetic 400 km circular
// orbit, with 100 Hz noise-free position, velocity and acceleration samples,
// alone and followed by Gauss-Newton (GN) refinement.
//
// EA = the extra acceleration, beyond the force model, fitted so the propagated
// trajectory matches the observations. The truth carries a smooth EA of
// 1e-7..1e-6 m/s² over a 300 s arc. Metric: position RMSE (root-mean-square
// error) of the trajectory re-propagated with the estimated EA, against the
// truth. Gate: < 1 m for the three B-spline cells and the position + GN cell;
// the velocity + GN and acceleration cells print their RMSE only.

#include "../test_utils.hpp"
#include "../test_helpers.hpp"

using ballistic::test::helpers::compute_pos_rmse;
#include <ballistic/init.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/calibration/bspline_accel_estimator.hpp>
#include <ballistic/calibration/gn_refine.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;
using namespace ballistic::calibration::detail;
using namespace ballistic::calibration;

static TrajectoryConfig base_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 8;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}


struct Setup {
    Epoch t0;
    Vector3d pos, vel;
    double arc_s;
    TrajectoryInterpolator* truth_interp;
    std::vector<std::pair<Epoch, Vector3d>> pos_100hz, vel_100hz, accel_100hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_ecef_100hz, accel_ecef_100hz;  // ECEF for integration
    std::unique_ptr<Trajectory> truth_traj;
    std::unique_ptr<TrajectoryInterpolator> truth_interp_ptr;
};

static Setup& get_setup() {
    static Setup s;
    static bool init = false;
    if (init) return s;
    init = true;

    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
    double vc = std::sqrt(constants::earth::GM / R);
    s.pos = Vector3d(R, 0, 0);
    // Earth-fixed velocity of the circular orbit: inertial vc minus omega x r
    // (the inertial vc read as Earth-fixed would fly an orbit of eccentricity
    // 0.133 with apogee ~2480 km; the start-state frame check flags that).
    s.vel = Vector3d(0, vc - constants::earth::ANGULAR_VELOCITY * R, 0);
    s.arc_s = 300.0;

    // Truth with smooth EA
    TrajectoryConfig truth_cfg = base_config();
    auto ea = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;
    for (double t = 0; t <= s.arc_s; t += 10) {
        double f = t / s.arc_s;
        entries.emplace_back(t, Vector3d(
            1e-6 * std::sin(f * 3.14159),
            -5e-7 * std::cos(f * 2.0),
            2e-7 * f));
    }
    *ea = forces::EmpiricalAcceleration::from_entries(entries, forces::EmpiricalAccelFrame::GCRS);
    truth_cfg.empirical_accel = ea;

    s.truth_traj = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, truth_cfg);
    while (s.truth_traj->epoch() - s.t0 < s.arc_s) s.truth_traj->step();
    s.truth_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.truth_traj);
    s.truth_interp = s.truth_interp_ptr.get();

    // Interpolator in GCRS (the inertial frame) for velocity/acceleration sampling
    InterpolatorConfig icfg;
    icfg.output_frame = InterpolationFrame::GCRS;
    TrajectoryInterpolator gcrs_interp(*s.truth_traj, icfg);

    for (double t = 0; t <= s.arc_s - 10; t += 0.01) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        auto sg = gcrs_interp.get_state(s.t0 + t);
        if (st && sg) {
            s.pos_100hz.emplace_back(s.t0 + t, st->position);      // ECEF
            s.vel_100hz.emplace_back(s.t0 + t, sg->velocity);       // GCRS
            s.vel_ecef_100hz.emplace_back(s.t0 + t, st->velocity);  // ECEF
            StateVector s_gcrs{sg->position, sg->velocity};
            Vector3d a_gcrs = s.truth_traj->force_model().acceleration(s.t0 + t, s_gcrs);
            s.accel_100hz.emplace_back(s.t0 + t, a_gcrs);           // GCRS
            // ECEF acceleration via frame transform
            StateVector se{st->position, st->velocity};
            s.accel_ecef_100hz.emplace_back(s.t0 + t,
                util::acceleration_to_itrf(a_gcrs, se, s.t0 + t));
        }
    }

    std::cout << "  [setup] LEO 400km, " << s.pos_100hz.size() << " obs at 100Hz, "
              << s.arc_s << "s arc" << std::endl;
    return s;
}

int main() {
    ballistic::initialize({});
    TestRunner runner("LEO Verification");

    // B-spline position 100 Hz
    runner.add_test("bspline_pos_100hz", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10;
        cfg.max_duration_s = s.arc_s - 10; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 50000; cfg.max_iterations = 5;

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, s.pos_100hz, cfg);
        TEST_ASSERT(result.empirical_accel != nullptr);

        TrajectoryConfig rc = base_config();
        rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.arc_s - 10) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 5, s.arc_s - 15, 1.0);
        std::cout << "    B-spline pos: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
        TEST_ASSERT_MSG(rmse < 1.0, "RMSE: " + std::to_string(rmse));
    });

    // B-spline velocity 100 Hz
    runner.add_test("bspline_vel_100hz", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10; cfg.lambda_drift = 1e-2;
        cfg.max_duration_s = s.arc_s - 10; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 50000; cfg.max_iterations = 5;

        auto result = estimate_empirical_acceleration_bspline_from_velocity(
            s.t0, s.pos, s.vel, s.vel_100hz, cfg);
        TEST_ASSERT(result.empirical_accel != nullptr);

        TrajectoryConfig rc = base_config();
        rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.arc_s - 10) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 5, s.arc_s - 15, 1.0);
        std::cout << "    B-spline vel: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
        TEST_ASSERT_MSG(rmse < 1.0, "RMSE: " + std::to_string(rmse));
    });

    // B-spline acceleration 100 Hz
    runner.add_test("bspline_accel_100hz", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
        cfg.max_duration_s = s.arc_s - 10; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 50000; cfg.max_iterations = 5;

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.t0, s.pos, s.vel, s.accel_100hz, cfg);
        TEST_ASSERT(result.empirical_accel != nullptr);

        TrajectoryConfig rc = base_config();
        rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.arc_s - 10) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 5, s.arc_s - 15, 1.0);
        std::cout << "    B-spline acc: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
        TEST_ASSERT_MSG(rmse < 1.0, "RMSE: " + std::to_string(rmse));
    });

    // =========================================================================
    // B-spline + GN with finite-difference Jacobians (position obs, 100 Hz)
    // =========================================================================
    runner.add_test("bspline_gn_pos_100hz", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10;
        cfg.max_duration_s = s.arc_s - 10; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 50000; cfg.max_iterations = 5;
        cfg.trajectory_config.rk45.max_step_size = 0.5;  // Must match re-propagation step

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, s.pos_100hz, cfg);
        TEST_ASSERT(result.empirical_accel != nullptr);

        GNRefineConfig gc;
        gc.max_iterations = 5;
        gc.fd_delta = 1e-4;
        gc.sample_stride = 100;
        gc.use_stm = false;  // finite differences, not the state transition matrix
        gc.trajectory_config = cfg.trajectory_config;
        gc.max_duration_s = cfg.max_duration_s;
        if (auto refined = refine_ea_gn(s.t0, s.pos, s.vel, s.pos_100hz,
                                        result.empirical_accel, gc))
            result.empirical_accel = refined;

        TrajectoryConfig rc = base_config();
        rc.rk45.max_step_size = 0.5;  // Match GN optimization step size
        rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.arc_s - 10) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 5, s.arc_s - 15, 1.0);
        std::cout << "    B-spline+GN pos: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
        TEST_ASSERT_MSG(rmse < 1.0, "RMSE: " + std::to_string(rmse));
    });

    // =========================================================================
    // B-spline + GN velocity + acceleration
    // =========================================================================
    runner.add_test("bspline_gn_vel_100hz", [&]() {
        auto& s = get_setup();
        // Integrate the Earth-fixed (ECEF) velocity samples once (trapezoid
        // rule) into positions, so GN has position targets built from the
        // velocity data only, not from the truth positions.
        std::vector<std::pair<Epoch, Vector3d>> pos_from_vel;
        Vector3d rr = s.pos;
        pos_from_vel.emplace_back(s.vel_ecef_100hz[0].first, rr);
        for (size_t i = 0; i + 1 < s.vel_ecef_100hz.size(); ++i) {
            double dt_i = s.vel_ecef_100hz[i+1].first - s.vel_ecef_100hz[i].first;
            rr += 0.5 * (s.vel_ecef_100hz[i].second + s.vel_ecef_100hz[i+1].second) * dt_i;
            pos_from_vel.emplace_back(s.vel_ecef_100hz[i+1].first, rr);
        }
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10; cfg.lambda_drift = 1e-2;
        cfg.max_duration_s = s.arc_s - 10; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 50000; cfg.max_iterations = 5;
        cfg.trajectory_config.rk45.max_step_size = 0.5;

        auto result = estimate_empirical_acceleration_bspline_from_velocity(
            s.t0, s.pos, s.vel, s.vel_100hz, cfg);
        TEST_ASSERT(result.empirical_accel != nullptr);

        GNRefineConfig gc;
        gc.max_iterations = 5;
        gc.fd_delta = 1e-4;
        gc.sample_stride = 100;
        gc.use_stm = false;
        gc.trajectory_config = cfg.trajectory_config;
        gc.max_duration_s = cfg.max_duration_s;
        if (auto refined = refine_ea_gn(s.t0, s.pos, s.vel, pos_from_vel,
                                        result.empirical_accel, gc))
            result.empirical_accel = refined;

        TrajectoryConfig rc = base_config();
        rc.rk45.max_step_size = 0.5;
        rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.arc_s - 10) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 5, s.arc_s - 15, 1.0);
        std::cout << "    B-spline+GN vel: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    runner.add_test("bspline_gn_accel_100hz", [&]() {
        auto& s = get_setup();
        // No GN for acceleration: its position targets would come from
        // re-propagating the estimated EA itself, so GN would only fit its own
        // output.
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
        cfg.max_duration_s = s.arc_s - 10; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 50000; cfg.max_iterations = 5;

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.t0, s.pos, s.vel, s.accel_100hz, cfg);
        TEST_ASSERT(result.empirical_accel != nullptr);

        TrajectoryConfig rc = base_config();
        rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.arc_s - 10) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 5, s.arc_s - 15, 1.0);
        std::cout << "    B-spline acc (no GN): " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    runner.run();
    return print_final_summary();
}
