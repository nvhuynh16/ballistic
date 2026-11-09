// Empirical acceleration (EA) estimation on a synthetic reentry glide.
// EA is the extra acceleration, beyond the force model, that makes a propagated
// trajectory fit observations; estimate_ea_bspline fits it as a B-spline from
// position (POS), velocity (VEL) or acceleration (ACC) samples, all in ECEF
// (Earth-fixed). The truth trajectory is a glide driven by a known smooth
// lift/lateral/return acceleration tabulated over 700 s (set up below); the
// scored arc ends 10 s before the 1 km crossing, about 182 s.
//
// Score: RMSE (root-mean-square position error) of the trajectory re-propagated
// with the returned EA, against the truth. Checks:
//   (1) clean data at 100 Hz and 200 Hz: RMSE < 2 m (POS at 200 Hz is
//       report-only: differentiating positions twice amplifies small errors);
//   (2) the regularization weight is scaled by the number of samples, so the
//       100 Hz and 200 Hz results are comparable (printed, not compared);
//   (3) noisy data (POS sigma 1 m, VEL 0.25 m/s, ACC 0.1 m/s^2): loose bounds
//       that catch divergence (accuracy is gated in test_ea_accuracy_claims);
//   (4) the default config equals an explicit Gauss-Newton (GN) refinement bit
//       for bit, and GN is ignored for ACC.

#include "../portable_random.hpp"
#include "../test_utils.hpp"
#include "../test_helpers.hpp"

using ballistic::test::helpers::compute_pos_rmse;
#include <ballistic/init.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/calibration/ea_estimator.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <cmath>
#include <iomanip>
#include <random>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;

// =============================================================================
// Glide setup — identical to test_glide_verification.cpp
// =============================================================================

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

    // Clean observations at 100 Hz and 200 Hz (all ECEF)
    std::vector<std::pair<Epoch, Vector3d>> pos_100hz, pos_200hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_100hz, vel_200hz;
    std::vector<std::pair<Epoch, Vector3d>> acc_100hz, acc_200hz;

    // Noisy observations at 100 Hz
    std::vector<std::pair<Epoch, Vector3d>> pos_noisy;   // σ=1m
    std::vector<std::pair<Epoch, Vector3d>> vel_noisy;   // σ=0.25 m/s
    std::vector<std::pair<Epoch, Vector3d>> acc_noisy;   // σ=0.1 m/s²

    std::unique_ptr<Trajectory> truth_traj;
    std::unique_ptr<TrajectoryInterpolator> truth_interp_ptr;
};

static void sample_ecef(
    const Setup& s, TrajectoryInterpolator& ecef_interp,
    TrajectoryInterpolator& gcrs_interp, double dt,
    std::vector<std::pair<Epoch, Vector3d>>& pos_out,
    std::vector<std::pair<Epoch, Vector3d>>& vel_out,
    std::vector<std::pair<Epoch, Vector3d>>& acc_out)
{
    for (double t = 0; t <= s.eval_end; t += dt) {
        auto st = ecef_interp.get_state(s.t0 + t);
        auto sg = gcrs_interp.get_state(s.t0 + t);
        if (!st || !sg) continue;
        pos_out.emplace_back(s.t0 + t, st->position);
        vel_out.emplace_back(s.t0 + t, st->velocity);
        StateVector s_gcrs{sg->position, sg->velocity};
        Vector3d a_gcrs = s.truth_traj->force_model().acceleration(s.t0 + t, s_gcrs);
        StateVector se{st->position, st->velocity};
        acc_out.emplace_back(s.t0 + t,
            util::acceleration_to_itrf(a_gcrs, se, s.t0 + t));
    }
}

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
    Vector3d rd = pk ? (-pk->position.normalized() + 0.4 * Vector3d(0, 0, 1)).normalized()
                     : Vector3d::Zero();

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

    // Clean observations at 100 Hz and 200 Hz
    sample_ecef(s, *s.truth_interp, gcrs_interp, 0.01,
                s.pos_100hz, s.vel_100hz, s.acc_100hz);
    sample_ecef(s, *s.truth_interp, gcrs_interp, 0.005,
                s.pos_200hz, s.vel_200hz, s.acc_200hz);

    // Noisy observations at 100 Hz
    std::mt19937 rng(42);
    ballistic::test::Normal pos_noise(0.0, 1.0);
    ballistic::test::Normal vel_noise(0.0, 0.25);
    ballistic::test::Normal acc_noise(0.0, 0.1);

    s.pos_noisy.reserve(s.pos_100hz.size());
    s.vel_noisy.reserve(s.vel_100hz.size());
    s.acc_noisy.reserve(s.acc_100hz.size());
    for (size_t i = 0; i < s.pos_100hz.size(); ++i) {
        Vector3d pn(pos_noise(rng), pos_noise(rng), pos_noise(rng));
        Vector3d vn(vel_noise(rng), vel_noise(rng), vel_noise(rng));
        Vector3d an(acc_noise(rng), acc_noise(rng), acc_noise(rng));
        s.pos_noisy.emplace_back(s.pos_100hz[i].first, s.pos_100hz[i].second + pn);
        s.vel_noisy.emplace_back(s.vel_100hz[i].first, s.vel_100hz[i].second + vn);
        s.acc_noisy.emplace_back(s.acc_100hz[i].first, s.acc_100hz[i].second + an);
    }

    std::cout << "  [setup] Glide reentry, 100Hz=" << s.pos_100hz.size()
              << " 200Hz=" << s.pos_200hz.size() << " obs, "
              << std::fixed << std::setprecision(1) << s.eval_end << "s arc\n";
    return s;
}

static double reprop_rmse(const Setup& s,
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea)
{
    TrajectoryConfig rc = base_config();
    rc.empirical_accel = ea;
    Trajectory rt(s.t0, s.pos, s.vel, rc);
    while (rt.epoch() - s.t0 < s.eval_end) rt.step();
    TrajectoryInterpolator ri(rt);
    return compute_pos_rmse(ri, *s.truth_interp, s.t0, 10.0, s.eval_end, 1.0);
}

// =============================================================================

int main() {
    ballistic::initialize({});
    TestRunner runner("EA Facade — Glide");
    auto fm = base_config();

    // =====================================================================
    // B-spline clean (100 Hz + 200 Hz)
    // =====================================================================

    runner.add_test("bspline_pos_100hz_clean", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        // Threshold 2.0 m gives headroom over the measured result (about
        // 0.01 m with the default Brent line search, state-transition-matrix
        // (STM) Jacobian and GN refinement).
        TEST_ASSERT_MSG(rmse < 2.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_pos_200hz_clean", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_200hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse
                  << " m  (d²/dt² amplification at 200Hz)\n";
    });

    runner.add_test("bspline_vel_100hz_clean", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, VelocitySamples{s.vel_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 2.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_vel_200hz_clean", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, VelocitySamples{s.vel_200hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 2.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_acc_100hz_clean", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, AccelSamples{s.acc_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 2.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_acc_200hz_clean", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, AccelSamples{s.acc_200hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 2.0, "RMSE: " + std::to_string(rmse));
    });

    // =====================================================================
    // B-spline noisy (100 Hz)
    // =====================================================================

    runner.add_test("bspline_pos_100hz_noisy", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        cfg.noise_sigma = 1.0;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_noisy}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse
                  << " m  (σ=1m)\n";
        // Default config (GN refinement on). The loose 100 m bound only checks
        // that the estimator does not diverge; accuracy on noisy POS is gated
        // in test_ea_accuracy_claims.
        TEST_ASSERT_MSG(rmse < 100.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_vel_100hz_noisy", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        cfg.noise_sigma = 0.25;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, VelocitySamples{s.vel_noisy}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse
                  << " m  (σ=0.25m/s)\n";
        TEST_ASSERT_MSG(rmse < 10.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_acc_100hz_noisy", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.eval_end;
        cfg.noise_sigma = 0.1;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, AccelSamples{s.acc_noisy}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse
                  << " m  (σ=0.1m/s²)\n";
        TEST_ASSERT_MSG(rmse < 15.0, "RMSE: " + std::to_string(rmse));
    });

    // =====================================================================
    // B-spline + GN
    // =====================================================================

    runner.add_test("bspline_pos_gn_default_byte_identical", [&]() {
        // The default config runs GN refinement (it measured best). Pin that
        // the default equals an explicit gn = GNRefine{} bit for bit. Short arc
        // and coarse knots keep the run time down.
        auto& s = get_setup();
        EABSplineConfig cfg_default;
        cfg_default.max_duration_s = 60.0;
        cfg_default.knot_spacing_s = 5.0;
        auto r0 = estimate_ea_bspline(
            s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg_default);
        TEST_ASSERT(r0.empirical_accel != nullptr);
        double rmse0 = reprop_rmse(s, r0.empirical_accel);

        EABSplineConfig cfg_explicit = cfg_default;
        cfg_explicit.gn = GNRefine{};
        auto r1 = estimate_ea_bspline(
            s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg_explicit);
        TEST_ASSERT(r1.empirical_accel != nullptr);
        double rmse1 = reprop_rmse(s, r1.empirical_accel);
        TEST_ASSERT_NEAR(rmse0, rmse1, 1e-12);
    });

    // GN convergence is verified by test_ea_accuracy_claims. Running full GN
    // here would multiply the run time; the wiring is verified by the
    // bit-identical default above and the ACC cell below.

    runner.add_test("bspline_acc_100hz_gn_skipped_safely", [&]() {
        // The ACC path ignores GN on purpose: acceleration samples give no
        // independent position targets to refine against. GN with 30
        // iterations must give the same result as the default, bit for bit.
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = 60.0;
        cfg.knot_spacing_s = 5.0;
        auto r0 = estimate_ea_bspline(
            s.t0, s.pos, s.vel, AccelSamples{s.acc_100hz}, fm, cfg);
        cfg.gn = GNRefine{30};
        auto r1 = estimate_ea_bspline(
            s.t0, s.pos, s.vel, AccelSamples{s.acc_100hz}, fm, cfg);
        TEST_ASSERT(r0.empirical_accel != nullptr);
        TEST_ASSERT(r1.empirical_accel != nullptr);
        double rmse0 = reprop_rmse(s, r0.empirical_accel);
        double rmse1 = reprop_rmse(s, r1.empirical_accel);
        TEST_ASSERT_NEAR(rmse0, rmse1, 1e-12);
    });

    // No collocation ACC cell: collocation is not a public estimator. For
    // noisy ACC use estimate_ea_bspline (AccelSamples) with hv-block CV.


    runner.run();
    return print_final_summary();
}
