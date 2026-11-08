// B-spline empirical-acceleration (EA) estimator on the glide fixture at sample
// rates from 10 to 1000 Hz, forward from the start and backward from the end,
// with position, velocity and acceleration samples.
//
// EA = the extra acceleration, beyond the force model, that the estimator fits so
// that a propagated trajectory matches the observations. The glide fixture is a
// synthetic reentry glide from 120 km driven by a known lift / lateral / return
// force (parameters below), sampled without noise.
//
// Purpose: show how the re-propagation RMSE (root-mean-square position error
// against the truth) changes as the rate grows. The cells assert only that an EA
// is returned; the RMSEs are printed for comparison. Heavy tier (about 520 s).

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
#include <functional>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;
using namespace ballistic::calibration::detail;
using namespace ballistic::calibration;

constexpr double LIFT_PEAK = 8.0, LATERAL_AMP = 3.0;
constexpr double FORCE_CENTER = 70.0, FORCE_WIDTH = 35.0, LATERAL_PERIOD = 100.0;
constexpr double RETURN_MAGNITUDE = 6.0, RETURN_CENTER = 420.0, RETURN_WIDTH = 80.0;

static TrajectoryConfig fast_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 20;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

static TrajectoryConfig make_forced_config(const Vector3d& return_dir) {
    TrajectoryConfig cfg = fast_config();
    auto ea = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;
    for (int i = 0; i <= 700; ++i) {
        double t = static_cast<double>(i);
        double gauss = std::exp(-0.5 * std::pow((t - FORCE_CENTER) / FORCE_WIDTH, 2));
        double lift = LIFT_PEAK * gauss;
        double lateral = LATERAL_AMP * gauss * std::sin(2.0 * constants::PI * t / LATERAL_PERIOD);
        double ret = RETURN_MAGNITUDE * std::exp(-0.5 * std::pow((t - RETURN_CENTER) / RETURN_WIDTH, 2));
        entries.emplace_back(t, Vector3d(0, lift, lateral) + ret * return_dir);
    }
    *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
    cfg.empirical_accel = ea;
    return cfg;
}

static std::pair<double, bool> propagate_to_ground(Trajectory& traj, const Epoch& t0, double max_dur) {
    for (int s = 0; s < 100000; ++s) {
        auto sr = traj.step();
        if (coords::ecef_to_geodetic(sr.position).altitude < 1000.0) return {sr.epoch - t0, true};
        if (sr.epoch - t0 > max_dur) return {sr.epoch - t0, false};
    }
    return {0, false};
}

static double bisect_impact(TrajectoryInterpolator& interp, const Epoch& t0, double approx_t) {
    double lo = approx_t - 2.0, hi = approx_t;
    for (int i = 0; i < 50; ++i) {
        double mid = 0.5 * (lo + hi);
        auto st = interp.get_state(t0 + mid);
        if (!st) break;
        if (coords::ecef_to_geodetic(st->position).altitude < 1000.0) hi = mid; else lo = mid;
    }
    return hi;
}


struct HRSetup {
    Epoch t0;
    Vector3d pos, vel;
    double forced_duration, sample_end;
    Epoch impact_epoch;
    Vector3d impact_pos, impact_vel;

    TrajectoryInterpolator* truth_interp;
    TrajectoryInterpolator* base_interp;

    std::unique_ptr<Trajectory> truth_traj_ptr, base_traj_ptr;
    std::unique_ptr<TrajectoryInterpolator> truth_interp_ptr, base_interp_ptr;
};

static HRSetup& get_setup() {
    static HRSetup s;
    static bool init = false;
    if (init) return s;
    init = true;

    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    double v_circ = std::sqrt(constants::earth::GM / R);
    s.pos = Vector3d(R, 0, 0);
    s.vel = Vector3d(-30.0, 0.85 * v_circ - constants::earth::ANGULAR_VELOCITY * R, 0);

    Trajectory glide(s.t0, s.pos, s.vel, make_forced_config(Vector3d::Zero()));
    for (int i = 0; i < 50000; ++i) { auto sr = glide.step(); if (sr.epoch - s.t0 >= RETURN_CENTER + 10) break; }
    TrajectoryInterpolator si(glide);
    auto peak = si.get_state(s.t0 + RETURN_CENTER);
    Vector3d rdir = peak ? ((-peak->position.normalized()) + 0.4 * Vector3d(0,0,1)).normalized() : Vector3d::Zero();

    s.truth_traj_ptr = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, make_forced_config(rdir));
    auto [fd, ok] = propagate_to_ground(*s.truth_traj_ptr, s.t0, 700.0);
    s.truth_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.truth_traj_ptr);
    if (ok) fd = bisect_impact(*s.truth_interp_ptr, s.t0, fd);
    s.forced_duration = fd;
    s.sample_end = fd - 10.0;
    s.truth_interp = s.truth_interp_ptr.get();

    s.base_traj_ptr = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, fast_config());
    while (s.base_traj_ptr->epoch() - s.t0 < s.forced_duration) s.base_traj_ptr->step();
    s.base_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.base_traj_ptr);
    s.base_interp = s.base_interp_ptr.get();

    auto imp = s.truth_interp->get_state(s.t0 + s.sample_end);
    if (imp) { s.impact_epoch = s.t0 + s.sample_end; s.impact_pos = imp->position; s.impact_vel = imp->velocity; }

    std::cout << "  [setup] duration=" << std::fixed << std::setprecision(1) << fd << "s" << std::endl;
    return s;
}

// Sample helpers
static auto sample_pos(double dt) {
    auto& s = get_setup();
    std::vector<std::pair<Epoch, Vector3d>> d;
    for (double t = 0; t <= s.sample_end; t += dt) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) d.emplace_back(s.t0 + t, st->position);
    }
    return d;
}

static auto sample_vel(double dt) {
    auto& s = get_setup();
    std::vector<std::pair<Epoch, Vector3d>> d;
    for (double t = 0; t <= s.sample_end; t += dt) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) {
            StateVector se{st->position, st->velocity};
            auto si = util::itrf_to_gcrs(se, s.t0 + t);
            d.emplace_back(s.t0 + t, si.velocity);
        }
    }
    return d;
}

static auto sample_accel(double dt) {
    auto& s = get_setup();
    std::vector<std::pair<Epoch, Vector3d>> d;
    for (double t = 0; t <= s.sample_end; t += dt) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) {
            StateVector se{st->position, st->velocity};
            auto si = util::itrf_to_gcrs(se, s.t0 + t);
            d.emplace_back(s.t0 + t, s.truth_traj_ptr->force_model().acceleration(s.t0 + t, si));
        }
    }
    return d;
}

// Backward sample helpers (sorted decreasing by epoch)
static auto sample_bw_pos(double dt) {
    auto& s = get_setup();
    std::vector<std::pair<Epoch, Vector3d>> d;
    for (double t = s.sample_end; t >= 0; t -= dt) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) d.emplace_back(s.t0 + t, st->position);
    }
    return d;
}

static auto sample_bw_vel(double dt) {
    auto& s = get_setup();
    std::vector<std::pair<Epoch, Vector3d>> d;
    for (double t = s.sample_end; t >= 0; t -= dt) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) {
            StateVector se{st->position, st->velocity};
            auto si = util::itrf_to_gcrs(se, s.t0 + t);
            d.emplace_back(s.t0 + t, si.velocity);
        }
    }
    return d;
}

static auto sample_bw_accel(double dt) {
    auto& s = get_setup();
    std::vector<std::pair<Epoch, Vector3d>> d;
    for (double t = s.sample_end; t >= 0; t -= dt) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) {
            StateVector se{st->position, st->velocity};
            auto si = util::itrf_to_gcrs(se, s.t0 + t);
            d.emplace_back(s.t0 + t, s.truth_traj_ptr->force_model().acceleration(s.t0 + t, si));
        }
    }
    return d;
}

int main() {
    ballistic::initialize({});
    TestRunner runner("B-Spline High-Rate Validation");

    // Forward position
    auto run_fwd_pos = [](const std::string& label, double dt) {
        auto& s = get_setup();
        auto data = sample_pos(dt);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline(s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
        std::cout << "    [" << label << "] n=" << data.size() << " rmse=" << std::fixed
                  << std::setprecision(3) << rmse << "m" << std::endl;
    };

    // Forward velocity (λ_drift=1e-2)
    auto run_fwd_vel = [](const std::string& label, double dt) {
        auto& s = get_setup();
        auto data = sample_vel(dt);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10; cfg.lambda_drift = 1e-2;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline_from_velocity(s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
        std::cout << "    [" << label << "] n=" << data.size() << " rmse=" << std::fixed
                  << std::setprecision(3) << rmse << "m" << std::endl;
    };

    // Forward acceleration
    auto run_fwd_acc = [](const std::string& label, double dt) {
        auto& s = get_setup();
        auto data = sample_accel(dt);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline_from_acceleration(s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
        std::cout << "    [" << label << "] n=" << data.size() << " rmse=" << std::fixed
                  << std::setprecision(3) << rmse << "m" << std::endl;
    };

    // Backward helpers
    auto run_bwd_pos = [](const std::string& label, double dt) {
        auto& s = get_setup();
        auto data = sample_bw_pos(dt);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10; cfg.backward = true;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline(s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.impact_epoch, s.impact_pos, s.impact_vel, rc);
        TrajectoryInterpolator ri(rt); ri.extend_backward_to(s.impact_epoch - s.sample_end);
        double sse = 0; int n = 0;
        for (double t = 10; t <= s.sample_end; t += 1) {
            auto a = ri.get_state(s.impact_epoch - t), b = s.truth_interp->get_state(s.impact_epoch - t);
            if (a && b) { sse += (a->position - b->position).squaredNorm(); ++n; }
        }
        double rmse = n > 0 ? std::sqrt(sse / n) : 1e9;
        std::cout << "    [" << label << "] n=" << data.size() << " rmse=" << std::fixed
                  << std::setprecision(3) << rmse << "m" << std::endl;
    };

    auto run_bwd_vel = [](const std::string& label, double dt) {
        auto& s = get_setup();
        auto data = sample_bw_vel(dt);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10; cfg.lambda_drift = 1e-2; cfg.backward = true;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline_from_velocity(s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.impact_epoch, s.impact_pos, s.impact_vel, rc);
        TrajectoryInterpolator ri(rt); ri.extend_backward_to(s.impact_epoch - s.sample_end);
        double sse = 0; int n = 0;
        for (double t = 10; t <= s.sample_end; t += 1) {
            auto a = ri.get_state(s.impact_epoch - t), b = s.truth_interp->get_state(s.impact_epoch - t);
            if (a && b) { sse += (a->position - b->position).squaredNorm(); ++n; }
        }
        double rmse = n > 0 ? std::sqrt(sse / n) : 1e9;
        std::cout << "    [" << label << "] n=" << data.size() << " rmse=" << std::fixed
                  << std::setprecision(3) << rmse << "m" << std::endl;
    };

    auto run_bwd_acc = [](const std::string& label, double dt) {
        auto& s = get_setup();
        auto data = sample_bw_accel(dt);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10; cfg.backward = true;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline_from_acceleration(s.impact_epoch, s.impact_pos, s.impact_vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, label + ": null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.impact_epoch, s.impact_pos, s.impact_vel, rc);
        TrajectoryInterpolator ri(rt); ri.extend_backward_to(s.impact_epoch - s.sample_end);
        double sse = 0; int n = 0;
        for (double t = 10; t <= s.sample_end; t += 1) {
            auto a = ri.get_state(s.impact_epoch - t), b = s.truth_interp->get_state(s.impact_epoch - t);
            if (a && b) { sse += (a->position - b->position).squaredNorm(); ++n; }
        }
        double rmse = n > 0 ? std::sqrt(sse / n) : 1e9;
        std::cout << "    [" << label << "] n=" << data.size() << " rmse=" << std::fixed
                  << std::setprecision(3) << rmse << "m" << std::endl;
    };

    // Forward tests
    for (auto [hz, dt] : std::vector<std::pair<int,double>>{{10,0.1},{100,0.01},{300,1./300},{500,0.002},{1000,0.001}}) {
        std::string tag = std::to_string(hz) + "hz";
        runner.add_test("fwd_pos_" + tag, [&, dt, tag]() { run_fwd_pos("fwd_pos_" + tag, dt); });
        runner.add_test("fwd_vel_" + tag, [&, dt, tag]() { run_fwd_vel("fwd_vel_" + tag, dt); });
        runner.add_test("fwd_acc_" + tag, [&, dt, tag]() { run_fwd_acc("fwd_acc_" + tag, dt); });
    }

    // Backward tests
    for (auto [hz, dt] : std::vector<std::pair<int,double>>{{300,1./300},{500,0.002},{1000,0.001}}) {
        std::string tag = std::to_string(hz) + "hz";
        runner.add_test("bwd_pos_" + tag, [&, dt, tag]() { run_bwd_pos("bwd_pos_" + tag, dt); });
        runner.add_test("bwd_vel_" + tag, [&, dt, tag]() { run_bwd_vel("bwd_vel_" + tag, dt); });
        runner.add_test("bwd_acc_" + tag, [&, dt, tag]() { run_bwd_acc("bwd_acc_" + tag, dt); });
    }

    // All rates again with regularization weight λ = 1e-8 (a candidate default).
    for (auto [hz, dt] : std::vector<std::pair<int,double>>{{10,0.1},{100,0.01},{300,1./300},{500,0.002},{1000,0.001}}) {
        std::string tag = std::to_string(hz) + "hz";
        runner.add_test("fwd_pos_lam8_" + tag, [&, dt, tag]() {
            auto& s = get_setup();
            auto data = sample_pos(dt);
            BSplineAccelConfig cfg;
            cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
            cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
            cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
            auto result = estimate_empirical_acceleration_bspline(s.t0, s.pos, s.vel, data, cfg);
            TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
            Trajectory rt(s.t0, s.pos, s.vel, rc);
            while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
            TrajectoryInterpolator ri(rt);
            double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
            std::cout << "    [fwd_pos_lam8_" << tag << "] rmse=" << std::fixed
                      << std::setprecision(3) << rmse << "m" << std::endl;
        });
        runner.add_test("fwd_acc_lam8_" + tag, [&, dt, tag]() {
            auto& s = get_setup();
            auto data = sample_accel(dt);
            BSplineAccelConfig cfg;
            cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
            cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
            cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0; cfg.max_iterations = 10;
            auto result = estimate_empirical_acceleration_bspline_from_acceleration(s.t0, s.pos, s.vel, data, cfg);
            TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
            Trajectory rt(s.t0, s.pos, s.vel, rc);
            while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
            TrajectoryInterpolator ri(rt);
            double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
            std::cout << "    [fwd_acc_lam8_" << tag << "] rmse=" << std::fixed
                      << std::setprecision(3) << rmse << "m" << std::endl;
        });
    }

    // Diagnostic: 500 Hz vs 300 Hz iteration detail
    // Run manually with verbose logging at each alpha
    runner.add_test("diag_500hz_linesearch", [&]() {
        auto& s = get_setup();

        for (double dt : {1.0/300, 0.002, 0.001}) {
            int hz = static_cast<int>(1.0 / dt + 0.5);
            auto data = sample_pos(dt);

            // Run B-spline for 3 iterations, then manually try more alphas
            BSplineAccelConfig cfg;
            cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10;
            cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
            cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0;
            cfg.max_iterations = 3;

            auto result = estimate_empirical_acceleration_bspline(
                s.t0, s.pos, s.vel, data, cfg);

            // Evaluate reconstruction RMSE
            if (result.empirical_accel) {
                TrajectoryConfig rc = fast_config();
                rc.empirical_accel = result.empirical_accel;
                Trajectory rt(s.t0, s.pos, s.vel, rc);
                while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
                TrajectoryInterpolator ri(rt);
                double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
                std::cout << "    [" << hz << "Hz] 3-iter recon_rmse=" << std::fixed
                          << std::setprecision(4) << rmse << "m (EA entries="
                          << result.empirical_accel->keys().size() << ")" << std::endl;

                // Print a few EA values to compare
                const auto& keys = result.empirical_accel->keys();
                for (size_t i = 0; i < (std::min)(keys.size(), size_t(3)); ++i) {
                    Vector3d ea = result.empirical_accel->accel_at((s.t0 + keys[i]) - (s.t0));
                    std::cout << "      t=" << std::setprecision(2) << keys[i]
                              << ": ea=(" << std::setprecision(6)
                              << ea.x() << ", " << ea.y() << ", " << ea.z() << ")"
                              << std::endl;
                }
            }
        }
    });

    // Diagnostic: 500 Hz position with more iterations (20)
    runner.add_test("fwd_pos_500hz_20iter", [&]() {
        auto& s = get_setup();
        auto data = sample_pos(0.002);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 20;
        auto result = estimate_empirical_acceleration_bspline(s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, "null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
        std::cout << "    [fwd_pos_500hz_20iter] rmse=" << std::fixed << std::setprecision(3)
                  << rmse << "m" << std::endl;
    });

    // Diagnostic: 500 Hz with λ=1e-8 (more regularization)
    runner.add_test("fwd_pos_500hz_lam8", [&]() {
        auto& s = get_setup();
        auto data = sample_pos(0.002);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline(s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, "null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
        std::cout << "    [fwd_pos_500hz_lam8] rmse=" << std::fixed << std::setprecision(3)
                  << rmse << "m" << std::endl;
    });

    // Diagnostic: 500 Hz acceleration with λ=1e-8
    runner.add_test("fwd_acc_500hz_lam8", [&]() {
        auto& s = get_setup();
        auto data = sample_accel(0.002);
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
        cfg.max_duration_s = s.sample_end; cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = fast_config(); cfg.min_altitude_m = 500.0;
        cfg.max_iterations = 10;
        auto result = estimate_empirical_acceleration_bspline_from_acceleration(s.t0, s.pos, s.vel, data, cfg);
        TEST_ASSERT_MSG(result.empirical_accel != nullptr, "null");
        TrajectoryConfig rc = fast_config(); rc.empirical_accel = result.empirical_accel;
        Trajectory rt(s.t0, s.pos, s.vel, rc);
        while (rt.epoch() - s.t0 < s.forced_duration) rt.step();
        TrajectoryInterpolator ri(rt);
        double rmse = compute_pos_rmse(ri, *s.truth_interp, s.t0, 10, s.sample_end, 1);
        std::cout << "    [fwd_acc_500hz_lam8] rmse=" << std::fixed << std::setprecision(3)
                  << rmse << "m" << std::endl;
    });

    runner.run();
    return print_final_summary();
}
