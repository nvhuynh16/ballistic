// Glide reentry verification (long-running): estimates the empirical acceleration
// (EA, the unmodeled force) with the B-spline estimator, alone and followed by
// Gauss-Newton (GN) refinement, from each of three observation types sampled at
// 100 Hz: positions (ECEF, Earth-fixed), velocities and accelerations (GCRS,
// inertial). The truth is a synthetic glide: degree-20 gravity plus a tabulated
// acceleration (a lift/lateral pulse centered at 70 s and a return pulse at 420 s)
// from 120 km altitude, stopped 10 s before it falls below 1 km.
// Metric: position RMSE (root-mean-square error) of a fresh propagation with the
// returned EA against the truth. The cells are report-only: they assert that an
// EA was returned and print the RMSE.

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

constexpr double LIFT_PEAK = 8.0, LATERAL_AMP = 3.0, FORCE_CENTER = 70.0;
constexpr double FORCE_WIDTH = 35.0, LATERAL_PERIOD = 100.0;
constexpr double RETURN_MAGNITUDE = 6.0, RETURN_CENTER = 420.0, RETURN_WIDTH = 80.0;

static TrajectoryConfig base_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 20;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    return cfg;
}

static TrajectoryConfig make_forced_config(const Vector3d& rd) {
    TrajectoryConfig cfg = base_config();
    auto ea = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;
    for (int i = 0; i <= 700; ++i) {
        double t = i;
        double g = std::exp(-0.5 * std::pow((t - FORCE_CENTER) / FORCE_WIDTH, 2));
        double rg = std::exp(-0.5 * std::pow((t - RETURN_CENTER) / RETURN_WIDTH, 2));
        entries.emplace_back(t, Vector3d(0, LIFT_PEAK * g,
            LATERAL_AMP * g * std::sin(2 * constants::PI * t / LATERAL_PERIOD))
            + RETURN_MAGNITUDE * rg * rd);
    }
    *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
    cfg.empirical_accel = ea;
    return cfg;
}


struct Setup {
    Epoch t0; Vector3d pos, vel;
    double eval_end;
    TrajectoryInterpolator* truth_interp;
    std::vector<std::pair<Epoch, Vector3d>> pos_100hz, vel_100hz, accel_100hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_ecef_100hz, accel_ecef_100hz;
    std::unique_ptr<Trajectory> truth_traj;
    std::unique_ptr<TrajectoryInterpolator> truth_interp_ptr;
};

static Setup& get_setup() {
    static Setup s;
    static bool init = false;
    if (init) return s;
    init = true;

    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    double vc = std::sqrt(constants::earth::GM / R);
    double wr = constants::earth::ANGULAR_VELOCITY * R;
    s.pos = Vector3d(R, 0, 0);
    s.vel = Vector3d(-30.0, 0.85 * vc - wr, 0);

    Trajectory sp(s.t0, s.pos, s.vel, make_forced_config(Vector3d::Zero()));
    for (int i = 0; i < 50000; i++) {
        auto sr = sp.step();
        if (sr.epoch - s.t0 >= RETURN_CENTER + 10) break;
    }
    TrajectoryInterpolator si(sp);
    auto pk = si.get_state(s.t0 + RETURN_CENTER);
    Vector3d rd = pk ? (-pk->position.normalized() + 0.4 * Vector3d(0, 0, 1)).normalized() : Vector3d::Zero();

    s.truth_traj = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, make_forced_config(rd));
    double fd = 0;
    for (int i = 0; i < 100000; i++) {
        auto sr = s.truth_traj->step();
        if (coords::ecef_to_geodetic(sr.position).altitude < 1000) { fd = sr.epoch - s.t0; break; }
        if (sr.epoch - s.t0 > 700) { fd = sr.epoch - s.t0; break; }
    }
    s.truth_interp_ptr = std::make_unique<TrajectoryInterpolator>(*s.truth_traj);
    double lo = fd - 2, hi = fd;
    for (int i = 0; i < 50; i++) {
        double mid = 0.5 * (lo + hi);
        auto st = s.truth_interp_ptr->get_state(s.t0 + mid);
        if (st && coords::ecef_to_geodetic(st->position).altitude < 1000) hi = mid; else lo = mid;
    }
    s.eval_end = hi - 10.0;
    s.truth_interp = s.truth_interp_ptr.get();

    InterpolatorConfig icfg;
    icfg.output_frame = InterpolationFrame::GCRS;
    TrajectoryInterpolator gcrs_interp(*s.truth_traj, icfg);

    for (double t = 0; t <= s.eval_end; t += 0.01) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        auto sg = gcrs_interp.get_state(s.t0 + t);
        if (st && sg) {
            s.pos_100hz.emplace_back(s.t0 + t, st->position);       // ECEF
            s.vel_100hz.emplace_back(s.t0 + t, sg->velocity);        // GCRS
            s.vel_ecef_100hz.emplace_back(s.t0 + t, st->velocity);   // ECEF
            StateVector s_gcrs{sg->position, sg->velocity};
            Vector3d a_gcrs = s.truth_traj->force_model().acceleration(s.t0 + t, s_gcrs);
            s.accel_100hz.emplace_back(s.t0 + t, a_gcrs);            // GCRS
            StateVector se{st->position, st->velocity};
            s.accel_ecef_100hz.emplace_back(s.t0 + t,
                util::acceleration_to_itrf(a_gcrs, se, s.t0 + t));
        }
    }

    std::cout << "  [setup] Glide reentry, " << s.pos_100hz.size() << " obs at 100Hz, "
              << std::fixed << std::setprecision(1) << s.eval_end << "s" << std::endl;
    return s;
}

// Helper: re-propagate and measure RMSE
static double reprop_rmse(const Setup& s,
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea,
    double max_step = 0.0)
{
    TrajectoryConfig rc = base_config();
    if (max_step > 0) rc.rk45.max_step_size = max_step;
    rc.empirical_accel = ea;
    Trajectory rt(s.t0, s.pos, s.vel, rc);
    while (rt.epoch() - s.t0 < s.eval_end) rt.step();
    TrajectoryInterpolator ri(rt);
    return compute_pos_rmse(ri, *s.truth_interp, s.t0, 10.0, s.eval_end, 1.0);
}

int main() {
    ballistic::initialize({});
    TestRunner runner("Glide Verification");

    // =========================================================================
    // B-spline only (3 obs types)
    // =========================================================================
    runner.add_test("bspline_pos", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;  // Position needs stronger regularization
        cfg.max_duration_s = s.eval_end; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 500; cfg.max_iterations = 10;
        auto r = estimate_empirical_acceleration_bspline(s.t0, s.pos, s.vel, s.pos_100hz, cfg);
        TEST_ASSERT(r.empirical_accel);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    B-spline pos: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    runner.add_test("bspline_vel", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-10; cfg.lambda_drift = 1e-2;
        cfg.max_duration_s = s.eval_end; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 500; cfg.max_iterations = 10;
        auto r = estimate_empirical_acceleration_bspline_from_velocity(s.t0, s.pos, s.vel, s.vel_100hz, cfg);
        TEST_ASSERT(r.empirical_accel);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    B-spline vel: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    runner.add_test("bspline_acc", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
        cfg.max_duration_s = s.eval_end; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 500; cfg.max_iterations = 10;
        auto r = estimate_empirical_acceleration_bspline_from_acceleration(s.t0, s.pos, s.vel, s.accel_100hz, cfg);
        TEST_ASSERT(r.empirical_accel);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    B-spline acc: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    // =========================================================================
    // B-spline + GN (3 obs types)
    // =========================================================================
    runner.add_test("bspline_gn_pos", [&]() {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;  // Position needs stronger regularization
        cfg.max_duration_s = s.eval_end; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 500; cfg.max_iterations = 5;
        cfg.trajectory_config.rk45.max_step_size = 0.5;
        auto r = estimate_empirical_acceleration_bspline(s.t0, s.pos, s.vel, s.pos_100hz, cfg);
        TEST_ASSERT(r.empirical_accel);

        GNRefineConfig gc;
        gc.max_iterations = 30;
        gc.fd_delta = 1e-2;
        gc.sample_stride = 10;
        gc.use_stm = false;
        gc.trajectory_config = cfg.trajectory_config;
        gc.max_duration_s = cfg.max_duration_s;
        if (auto refined = refine_ea_gn(s.t0, s.pos, s.vel, s.pos_100hz,
                                        r.empirical_accel, gc))
            r.empirical_accel = refined;

        double rmse = reprop_rmse(s, r.empirical_accel, 0.5);
        std::cout << "    B-spline+GN pos: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    runner.add_test("bspline_gn_vel", [&]() {
        auto& s = get_setup();
        // Integrate the ECEF velocities once to get positions for GN to fit (a user
        // with velocity data would have only these, not the true positions)
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
        cfg.max_duration_s = s.eval_end; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 500; cfg.max_iterations = 5;
        cfg.trajectory_config.rk45.max_step_size = 0.5;
        auto r = estimate_empirical_acceleration_bspline_from_velocity(s.t0, s.pos, s.vel, s.vel_100hz, cfg);
        TEST_ASSERT(r.empirical_accel);

        GNRefineConfig gc;
        gc.max_iterations = 30;
        gc.fd_delta = 1e-2;
        gc.sample_stride = 10;
        gc.use_stm = false;
        gc.trajectory_config = cfg.trajectory_config;
        gc.max_duration_s = cfg.max_duration_s;
        if (auto refined = refine_ea_gn(s.t0, s.pos, s.vel, pos_from_vel,
                                        r.empirical_accel, gc))
            r.empirical_accel = refined;

        double rmse = reprop_rmse(s, r.empirical_accel, 0.5);
        std::cout << "    B-spline+GN vel: " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    runner.add_test("bspline_gn_acc", [&]() {
        auto& s = get_setup();
        // B-spline GN for acceleration is circular: the reference trajectory IS
        // re-propagated with estimated EA, so GN targets = GN output. GN disabled.
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0; cfg.lambda = 1e-8;
        cfg.max_duration_s = s.eval_end; cfg.trajectory_config = base_config();
        cfg.min_altitude_m = 500; cfg.max_iterations = 5;
        auto r = estimate_empirical_acceleration_bspline_from_acceleration(s.t0, s.pos, s.vel, s.accel_100hz, cfg);
        TEST_ASSERT(r.empirical_accel);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    B-spline acc (no GN): " << std::fixed << std::setprecision(3) << rmse << " m" << std::endl;
    });

    runner.run();
    return print_final_summary();
}
