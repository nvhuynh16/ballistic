// Integration test for the empirical-acceleration (EA) estimator front end
// (ea_estimator.hpp). An EA is the extra acceleration the force model is
// missing; the estimator recovers it from observations of the orbit.
//
// Setup (synthetic, no data files): a 400 km circular orbit flown for 300 s with
// a known smooth EA (~1e-6 m/s^2) added, sampled at 100 Hz as Earth-fixed (ECEF)
// positions, velocities and accelerations. Each cell estimates the EA, flies the
// orbit again with it ("re-propagation") and requires the position RMSE (root-
// mean-square error) against the truth to stay under a loose bound (1-5 m): a
// regression gate for the plumbing, not an accuracy claim. Also covers the
// smoothing-weight variants (fixed lambda, IRGN = iteratively regularized
// Gauss-Newton schedule, GCV = generalized cross-validation), the hv-block
// cross-validation (CV) settings, the K-fold split and the integration helpers.
// Mirrors the test_leo_verification.cpp setup.

#include "../test_utils.hpp"
#include "../test_helpers.hpp"

using ballistic::test::helpers::compute_pos_rmse;
#include <ballistic/init.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/calibration/ea_estimator.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <cmath>
#include <iomanip>
#include <set>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;

// =============================================================================
// Setup — identical to test_leo_verification.cpp pattern
// =============================================================================

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
    // 100 Hz observations
    std::vector<std::pair<Epoch, Vector3d>> pos_100hz;         // ECEF positions
    std::vector<std::pair<Epoch, Vector3d>> vel_ecef_100hz;    // ECEF velocities
    // ECEF coordinate acceleration d²r_ecef/dt² (includes Coriolis + centrifugal).
    // The facade handles full ECEF→GCRS conversion using a reference trajectory.
    std::vector<std::pair<Epoch, Vector3d>> accel_ecef_100hz;
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
    // (the inertial vc read as Earth-fixed would fly eccentricity 0.133, apogee
    // ~2480 km; the start-state frame check flags that mix-up).
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

    InterpolatorConfig icfg;
    icfg.output_frame = InterpolationFrame::GCRS;
    TrajectoryInterpolator gcrs_interp(*s.truth_traj, icfg);

    for (double t = 0; t <= s.arc_s - 10; t += 0.01) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        auto sg = gcrs_interp.get_state(s.t0 + t);
        if (st && sg) {
            s.pos_100hz.emplace_back(s.t0 + t, st->position);
            s.vel_ecef_100hz.emplace_back(s.t0 + t, st->velocity);
            // ECEF coordinate acceleration (d²r_ecef/dt², with Coriolis/centrifugal)
            StateVector s_gcrs{sg->position, sg->velocity};
            Vector3d a_gcrs = s.truth_traj->force_model().acceleration(s.t0 + t, s_gcrs);
            StateVector se{st->position, st->velocity};
            s.accel_ecef_100hz.emplace_back(s.t0 + t,
                util::acceleration_to_itrf(a_gcrs, se, s.t0 + t));
        }
    }

    std::cout << "  [setup] LEO 400km, " << s.pos_100hz.size() << " obs at 100Hz, "
              << s.arc_s << "s arc" << std::endl;
    return s;
}

// Re-propagate with estimated EA and measure RMSE against truth
static double reprop_rmse(const Setup& s,
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea)
{
    TrajectoryConfig rc = base_config();
    rc.empirical_accel = ea;
    Trajectory rt(s.t0, s.pos, s.vel, rc);
    while (rt.epoch() - s.t0 < s.arc_s - 10) rt.step();
    TrajectoryInterpolator ri(rt);
    return compute_pos_rmse(ri, *s.truth_interp, s.t0, 5, s.arc_s - 15, 1.0);
}

// =============================================================================

int main() {
    ballistic::initialize({});
    TestRunner runner("EA Estimator Facade");
    auto fm = base_config();

    // =====================================================================
    // 1. B-spline facade: 3 observation types
    // =====================================================================

    runner.add_test("bspline_position", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.arc_s - 10;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 1.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_velocity", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.arc_s - 10;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, VelocitySamples{s.vel_ecef_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 5.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("bspline_acceleration", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.arc_s - 10;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, AccelSamples{s.accel_ecef_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 1.0, "RMSE: " + std::to_string(rmse));
    });

    // No collocation cell for accelerometer data: collocation is not a public
    // estimator (B-spline + hv-block CV is the noisy-accelerometer path; the plain
    // B-spline fit is best on clean accelerometer data).

    // =====================================================================
    // 3. Lambda variant dispatch (all use position — fastest)
    // =====================================================================

    runner.add_test("lambda_fixed", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.regularization = ea::FixedLambda{1e-8};
        cfg.max_duration_s = s.arc_s - 10;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 5.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("lambda_irgn_explicit", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.regularization = ea::IRGN{1e-2, 0.3, 1e-8};
        cfg.max_duration_s = s.arc_s - 10;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 5.0, "RMSE: " + std::to_string(rmse));
    });

    runner.add_test("lambda_gcv", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.regularization = ea::GCV{};
        cfg.max_duration_s = s.arc_s - 10;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
        double rmse = reprop_rmse(s, r.empirical_accel);
        std::cout << "    RMSE: " << std::fixed << std::setprecision(3) << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 5.0, "RMSE: " + std::to_string(rmse));
    });

    // =====================================================================
    // 4. Accelerometer front end honors the caller's degree / knot spacing /
    //    regularization (no silent override)
    // =====================================================================

    // ea::GCV{lam_min, lam_max} is a search interval: the fit must pick λ
    // inside it (the bounds used to be ignored in favor of 1e-12 .. 0.1).
    runner.add_test("gcv_interval_honored", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.arc_s - 10;
        cfg.regularization = ea::GCV{};
        auto wide = estimate_ea_bspline(
            s.t0, s.pos, s.vel, AccelSamples{s.accel_ecef_100hz}, fm, cfg);
        cfg.regularization = ea::GCV{1e-3, 1e-2};
        auto narrow = estimate_ea_bspline(
            s.t0, s.pos, s.vel, AccelSamples{s.accel_ecef_100hz}, fm, cfg);
        TEST_ASSERT(wide.lambda_used.has_value() && narrow.lambda_used.has_value());
        const double lw = *wide.lambda_used, ln = *narrow.lambda_used;
        std::cout << "    GCV lambda: default interval " << std::scientific
                  << std::setprecision(2) << lw << ", [1e-3, 1e-2] " << ln
                  << std::fixed << "\n";
        TEST_ASSERT_MSG(ln >= 1e-3 * (1 - 1e-9) && ln <= 1e-2 * (1 + 1e-9),
                        "lambda " + std::to_string(ln) + " outside [1e-3, 1e-2]");
        // The default interval chooses a λ outside the narrow one, so the
        // narrow result is the interval's doing.
        TEST_ASSERT_MSG(lw < 1e-3 || lw > 1e-2,
                        "default-interval lambda already inside [1e-3, 1e-2]: "
                        "the check cannot tell the interval was used");

        bool threw = false;
        try {
            cfg.regularization = ea::GCV{1e-2, 1e-3};
            (void)estimate_ea_bspline(
                s.t0, s.pos, s.vel, AccelSamples{s.accel_ecef_100hz}, fm, cfg);
        } catch (const ConfigError&) { threw = true; }
        TEST_ASSERT_MSG(threw, "an empty GCV interval must raise ConfigError");
    });

    runner.add_test("acc_no_facade_override", [&]() {
        auto& s = get_setup();
        // The front end must pass the caller's spline settings to the fitter
        // unchanged (a coercion such as degree 5 -> 3 or knot spacing 2.0 ->
        // 3.0 once hid in it). Acceleration data skip GN, so the returned EA
        // is the fitted B-spline itself and its degree and knot vector can be
        // read back directly. Comparing the accuracy of the two fits is not a
        // test of this: once both reach the noise floor they agree to ~1e-6 m.
        EABSplineConfig deg5_cfg;
        deg5_cfg.degree = 5;
        deg5_cfg.knot_spacing_s = 2.0;
        deg5_cfg.regularization = ea::IRGN{1e-2, 0.5, 1e-12};
        deg5_cfg.max_duration_s = s.arc_s - 10;
        auto r5 = estimate_ea_bspline(
            s.t0, s.pos, s.vel, AccelSamples{s.accel_ecef_100hz}, fm, deg5_cfg);

        EABSplineConfig deg3_cfg = deg5_cfg;
        deg3_cfg.degree = 3;
        deg3_cfg.knot_spacing_s = 3.0;
        auto r3 = estimate_ea_bspline(
            s.t0, s.pos, s.vel, AccelSamples{s.accel_ecef_100hz}, fm, deg3_cfg);

        TEST_ASSERT(r5.empirical_accel && r3.empirical_accel);
        const auto b5 = r5.empirical_accel->as_bspline();
        const auto b3 = r3.empirical_accel->as_bspline();
        TEST_ASSERT_MSG(b5 && b3, "acceleration fits must return the B-spline itself");
        TEST_ASSERT_MSG(b5->degree == 5, "degree 5 requested, got " + std::to_string(b5->degree));
        TEST_ASSERT_MSG(b3->degree == 3, "degree 3 requested, got " + std::to_string(b3->degree));
        // Knot count = the one the requested spacing gives over the fitted
        // span: floor(span / spacing) - 1 interior knots plus degree + 1
        // clamped knots at each end.
        auto expected_knots = [](const std::vector<double>& k, double spacing, int degree) {
            const double span = k.back() - k.front();
            return static_cast<size_t>(std::floor(span / spacing) - 1 + 2 * (degree + 1));
        };
        TEST_ASSERT_MSG(b5->knots.size() == expected_knots(b5->knots, 2.0, 5),
            "degree-5 fit: " + std::to_string(b5->knots.size()) + " knots, not the 2.0 s spacing");
        TEST_ASSERT_MSG(b3->knots.size() == expected_knots(b3->knots, 3.0, 3),
            "degree-3 fit: " + std::to_string(b3->knots.size()) + " knots, not the 3.0 s spacing");

        const double d5 = reprop_rmse(s, r5.empirical_accel);
        const double d3 = reprop_rmse(s, r3.empirical_accel);
        std::cout << "    deg=5: " << std::fixed << std::setprecision(6) << d5
                  << " m, deg=3: " << d3 << " m\n";
        // Both configs must be functional (finite RMSE on clean LEO).
        TEST_ASSERT_MSG(d5 < 5.0, "deg=5 RMSE: " + std::to_string(d5));
        TEST_ASSERT_MSG(d3 < 5.0, "deg=3 RMSE: " + std::to_string(d3));
    });

    // =====================================================================
    // 4b. hv-block CV schedule plumbing (translate_cv_dep_config)
    // =====================================================================

    runner.add_test("cv_translate_default", []() {
        EABSplineConfig cfg;
        auto dep = detail::translate_cv_dep_config(cfg, ea::CV{});
        TEST_ASSERT(dep.hv_n_blocks == 18);
        TEST_ASSERT(dep.hv_gap_blocks == 1);
        const std::vector<int> expected = {2, 7, 12};
        TEST_ASSERT(dep.hv_test_blocks == expected);
    });

    runner.add_test("cv_translate_custom_n_blocks", []() {
        EABSplineConfig cfg;
        ea::CV cv; cv.hv_n_blocks = 30;
        auto dep = detail::translate_cv_dep_config(cfg, cv);
        TEST_ASSERT(dep.hv_n_blocks == 30);
        const std::vector<int> expected = {2, 7, 12, 17, 22, 27};
        TEST_ASSERT(dep.hv_test_blocks == expected);
    });

    runner.add_test("cv_translate_short_arc", []() {
        EABSplineConfig cfg;
        ea::CV cv; cv.hv_n_blocks = 10;
        auto dep = detail::translate_cv_dep_config(cfg, cv);
        const std::vector<int> expected = {2, 7};
        TEST_ASSERT(dep.hv_test_blocks == expected);
    });

    runner.add_test("cv_translate_explicit_overrides_auto", []() {
        EABSplineConfig cfg;
        ea::CV cv; cv.hv_n_blocks = 18; cv.hv_test_blocks = {3, 8, 13};
        auto dep = detail::translate_cv_dep_config(cfg, cv);
        const std::vector<int> expected = {3, 8, 13};
        TEST_ASSERT(dep.hv_test_blocks == expected);
    });

    runner.add_test("cv_translate_passes_gn_fields", []() {
        EABSplineConfig cfg;
        cfg.gn = GNRefine{.max_iterations = 30, .sample_stride = 25,
                          .ea_subsample = 2, .fd_delta = 5e-3,
                          .rk45_max_step = 1.0};
        cfg.gn->reg_lambda = 0.25;
        cfg.gn->reg_deriv_order = 2;
        ea::CV cv; cv.hv_gap_blocks = 2; cv.grid = ea::CV::Grid::Coarse;
        auto dep = detail::translate_cv_dep_config(cfg, cv);
        TEST_ASSERT(dep.gn_iterations == 30);
        TEST_ASSERT(dep.gn_sample_stride == 25);
        TEST_ASSERT(dep.gn_ea_subsample == 2);
        TEST_ASSERT_NEAR(dep.fd_delta, 5e-3, 1e-12);
        TEST_ASSERT_NEAR(dep.rk45_max_step_size, 1.0, 1e-12);
        TEST_ASSERT(dep.hv_gap_blocks == 2);
        TEST_ASSERT(dep.coarse_grid);
        // The GN smoothing penalty reaches the GN runs inside CV.
        TEST_ASSERT_NEAR(dep.gn_reg_lambda, 0.25, 0.0);
        TEST_ASSERT(dep.gn_reg_order == 2);
        auto gc = detail::make_gn_config_from_dep(dep, TrajectoryConfig{}, 2.0, 100.0,
                                                  /*reg_order_auto=*/1);
        TEST_ASSERT_NEAR(gc.reg_lambda, 0.25, 0.0);
        TEST_ASSERT(gc.reg_order == 2);
        // Unset order = automatic: the data kind's order (3 POS, 1 VEL).
        EABSplineConfig dflt;
        auto dep0 = detail::translate_cv_dep_config(dflt, ea::CV{});
        TEST_ASSERT(dep0.gn_reg_order == -1);
        TEST_ASSERT_NEAR(dep0.gn_reg_lambda, 1.0, 0.0);
        TEST_ASSERT(detail::make_gn_config_from_dep(dep0, TrajectoryConfig{}, 2.0, 100.0, 1)
                        .reg_order == 1);
        TEST_ASSERT(detail::make_gn_config_from_dep(dep0, TrajectoryConfig{}, 2.0, 100.0, 3)
                        .reg_order == 3);
    });

    // A layout that holds out nothing cannot cross-validate: it used to fall
    // back to fitting the first grid point silently.
    runner.add_test("cv_layout_without_held_out_block_raises", []() {
        EABSplineConfig cfg;
        auto raises = [&](const ea::CV& cv) {
            try { (void)detail::translate_cv_dep_config(cfg, cv); }
            catch (const ConfigError&) { return true; }
            return false;
        };
        TEST_ASSERT(raises(ea::CV{2}));
        TEST_ASSERT(raises(ea::CV{3}));
        ea::CV out_of_range; out_of_range.hv_test_blocks = {18};
        TEST_ASSERT(raises(out_of_range));
        TEST_ASSERT(!raises(ea::CV{4}));   // block 2 held out
    });

    // =====================================================================
    // 5. ECEF velocity front end = direct call with inertial (GCRS) velocities
    //    (same conversion, so the RMSEs agree to 1 cm)
    // =====================================================================

    runner.add_test("ecef_vel_facade_equals_direct", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = s.arc_s - 10;

        auto r_fac = estimate_ea_bspline(
            s.t0, s.pos, s.vel, VelocitySamples{s.vel_ecef_100hz}, fm, cfg);

        // Manual: same conversion the facade does internally
        auto vel_gcrs = detail::vel_ecef_to_gcrs(s.vel_ecef_100hz, s.pos);
        auto ic = detail::translate_config(cfg, fm, -1);
        ic.lambda_drift = 1e-2;
        auto r_dir = estimate_empirical_acceleration_bspline_from_velocity(
            s.t0, s.pos, s.vel, vel_gcrs, ic);

        TEST_ASSERT(r_fac.empirical_accel && r_dir.empirical_accel);
        double f = reprop_rmse(s, r_fac.empirical_accel);
        double d = reprop_rmse(s, r_dir.empirical_accel);
        std::cout << "    facade: " << std::fixed << std::setprecision(3) << f
                  << " m, direct: " << d << " m\n";
        TEST_ASSERT_NEAR(f, d, 0.01);
    });

    // =====================================================================
    // 6. Config propagation
    // =====================================================================

    runner.add_test("noise_sigma_auto_scales", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.noise_sigma = 1.0;  // early stopping at 1.5σ = 1.5m
        cfg.max_duration_s = s.arc_s - 10;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
    });

    runner.add_test("max_duration_respected", [&]() {
        auto& s = get_setup();
        EABSplineConfig cfg;
        cfg.max_duration_s = 30.0;  // only first 30s of data
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{s.pos_100hz}, fm, cfg);
        TEST_ASSERT(r.empirical_accel != nullptr);
    });

    // =====================================================================
    // 7. K-fold CV split logic (pure unit tests)
    // =====================================================================

    runner.add_test("kfold_split_sizes", [&]() {
        using Obs = std::pair<Epoch, Vector3d>;
        Epoch t0 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
        std::vector<Obs> data;
        for (int i = 0; i < 102; ++i)
            data.emplace_back(t0 + i, Vector3d(i, 0, 0));
        auto split = detail::split_kfold(data, 5);
        TEST_ASSERT(split.endpoints.size() == 2);
        size_t total = 0;
        for (const auto& f : split.folds) total += f.size();
        TEST_ASSERT(total == 100);
        for (int f = 0; f < 5; ++f) TEST_ASSERT(split.folds[f].size() == 20);
    });

    runner.add_test("kfold_endpoints_always_in_training", [&]() {
        using Obs = std::pair<Epoch, Vector3d>;
        Epoch t0 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
        std::vector<Obs> data;
        for (int i = 0; i < 20; ++i)
            data.emplace_back(t0 + i, Vector3d(i, 0, 0));
        auto split = detail::split_kfold(data, 5);

        for (int f = 0; f < 5; ++f)
            for (const auto& [ep, v] : split.folds[f])
                TEST_ASSERT(v.x() > 0.5 && v.x() < 18.5);

        for (int f = 0; f < 5; ++f) {
            auto train = detail::merge_train_folds(split, f);
            bool first = false, last = false;
            for (const auto& [ep, v] : train) {
                if (v.x() < 0.5) first = true;
                if (v.x() > 18.5) last = true;
            }
            TEST_ASSERT(first && last);
        }
    });

    runner.add_test("kfold_no_overlap", [&]() {
        using Obs = std::pair<Epoch, Vector3d>;
        Epoch t0 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
        std::vector<Obs> data;
        for (int i = 0; i < 50; ++i)
            data.emplace_back(t0 + i, Vector3d(i, 0, 0));
        auto split = detail::split_kfold(data, 3);
        std::set<int> seen;
        for (const auto& fold : split.folds)
            for (const auto& [ep, v] : fold) {
                int x = static_cast<int>(std::round(v.x()));
                TEST_ASSERT(seen.find(x) == seen.end());
                seen.insert(x);
            }
        TEST_ASSERT(seen.size() == 48);
    });

    runner.add_test("kfold_merge_sorted", [&]() {
        using Obs = std::pair<Epoch, Vector3d>;
        Epoch t0 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
        std::vector<Obs> data;
        for (int i = 0; i < 30; ++i)
            data.emplace_back(t0 + i, Vector3d(i, 0, 0));
        auto split = detail::split_kfold(data, 4);
        for (int f = 0; f < 4; ++f) {
            auto train = detail::merge_train_folds(split, f);
            for (size_t i = 1; i < train.size(); ++i)
                TEST_ASSERT(train[i].first >= train[i - 1].first);
        }
    });

    // =====================================================================
    // 8. Integration proxies
    // =====================================================================

    runner.add_test("integrate_vel_to_pos", [&]() {
        auto& s = get_setup();
        auto ip = detail::integrate_ecef_vel_to_pos(s.vel_ecef_100hz, s.pos);
        TEST_ASSERT(ip.size() == s.vel_ecef_100hz.size());
        TEST_ASSERT((ip[0].second - s.pos).norm() < 1e-6);
    });

    runner.add_test("integrate_acc_to_pos", [&]() {
        auto& s = get_setup();
        auto ip = detail::integrate_ecef_acc_to_pos(s.accel_ecef_100hz, s.vel, s.pos);
        TEST_ASSERT(ip.size() == s.accel_ecef_100hz.size());
        TEST_ASSERT((ip[0].second - s.pos).norm() < 1e-6);
    });

    // =====================================================================
    // 9. CV grid
    // =====================================================================

    runner.add_test("cv_grid_495_combos", [&]() {
        // The grid includes knot spacings ks ∈ {8, 10, 12, 16} s (495 combinations):
        // ks = 10 s was best for noisy position data with the B-spline estimator
        // plus Gauss-Newton refinement (GN), a regime a grid ending at ks = 5 s
        // misses entirely.

        auto grid = detail::build_cv_grid();
        TEST_ASSERT(grid.size() == 495);
        for (const auto& hp : grid) {
            TEST_ASSERT(hp.knot_spacing > 0 && hp.lambda_0 > 0);
            TEST_ASSERT(hp.decay > 0 && hp.decay < 1);
            TEST_ASSERT(hp.lambda_min <= hp.lambda_0);
        }
    });

    // =====================================================================
    // 10. Short arc edge case
    // =====================================================================

    runner.add_test("short_arc_no_crash", [&]() {
        auto& s = get_setup();
        std::vector<std::pair<Epoch, Vector3d>> short_data(
            s.pos_100hz.begin(), s.pos_100hz.begin() + 50);
        EABSplineConfig cfg;
        cfg.max_duration_s = 5.0;
        auto r = estimate_ea_bspline(s.t0, s.pos, s.vel, PositionSamples{short_data}, fm, cfg);
        std::cout << "    EA " << (r.empirical_accel ? "valid" : "null") << "\n";
    });

    runner.run();
    return print_final_summary();
}
