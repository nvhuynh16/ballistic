// Integration test for cross-validation (CV) in the empirical-acceleration (EA)
// estimator. The EA is the extra acceleration the estimator fits so that a
// propagated orbit matches observations; CV picks the estimator's tuning
// parameters (knot spacing, regularization) by holding out part of the data.
// Cells: (1) the B-spline estimator with ea::CV (coarse grid, 10 blocks) on
// clean 10 Hz synthetic positions recovers a known EA and reports its
// grid-edge choice; (2) velocity CV refines each candidate with GN on the
// training points only, never on the held-out block it is scored on; (3) GN
// inside velocity CV uses the velocity-data penalty order and the caller's
// reg_lambda; (4) the grid-edge report text; (5) velocity CV end to end on the
// coarse grid; (6) the K-fold split helpers in detail/ea_cv.hpp
// (split_kfold / merge_train_folds) keep the arc endpoints in every training
// set. The heavy build adds velocity CV end to end on the full grid.
// Production CV is hv-block CV (blocks of consecutive data
// held out with a gap), not plain K-fold; see detail/ea_cv.hpp
// (cv_bspline_position / _velocity / _acceleration).


#include "../portable_random.hpp"
#include "../test_utils.hpp"
#include <ballistic/init.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/calibration/ea_estimator.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <ballistic/core/diagnostics.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;

// =============================================================================
// Simple low-Earth-orbit scenario with a known EA
// =============================================================================

static TrajectoryConfig base_config() {
    TrajectoryConfig c;
    c.gravity_degree = 8;
    c.enable_third_body = false;
    c.enable_relativistic = false;
    c.enable_solid_tides = false;

    return c;
}

static TrajectoryConfig make_truth_config() {
    auto c = base_config();

    // Truth EA: a 5 m/s^2 Gaussian bump (40 s wide, centered at 150 s) along the
    // inertial (GCRS) y axis
    auto ea = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;
    for (int i = 0; i <= 300; ++i) {
        double t = static_cast<double>(i);
        double bump = 5.0 * std::exp(-0.5 * std::pow((t - 150.0) / 40.0, 2));
        entries.emplace_back(t, Vector3d(0, bump, 0));
    }
    *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
    c.empirical_accel = ea;
    return c;
}

// Noise-free Earth-fixed (ECEF) position observations from the truth trajectory
static std::vector<std::pair<Epoch, Vector3d>> generate_pos_obs(
    const Epoch& t0, const Vector3d& pos, const Vector3d& vel,
    double hz, double duration_s)
{
    auto cfg = make_truth_config();
    Trajectory truth(t0, pos, vel, cfg);
    while (truth.epoch() - t0 < duration_s + 10) truth.step();
    TrajectoryInterpolator interp(truth);

    std::vector<std::pair<Epoch, Vector3d>> obs;
    double dt = 1.0 / hz;
    for (double t = 0; t <= duration_s; t += dt) {
        auto st = interp.get_state(t0 + t);
        if (st) obs.emplace_back(t0 + t, st->position);
    }
    return obs;
}

// Noise-free Earth-fixed (ECEF) velocity observations from the truth trajectory
static std::vector<std::pair<Epoch, Vector3d>> generate_vel_obs(
    const Epoch& t0, const Vector3d& pos, const Vector3d& vel,
    double hz, double duration_s)
{
    auto cfg = make_truth_config();
    Trajectory truth(t0, pos, vel, cfg);
    while (truth.epoch() - t0 < duration_s + 10) truth.step();
    TrajectoryInterpolator interp(truth);

    std::vector<std::pair<Epoch, Vector3d>> obs;
    const int n = static_cast<int>(std::lround(duration_s * hz));
    for (int i = 0; i <= n; ++i) {
        const double t = i / hz;
        auto st = interp.get_state(t0 + t);
        if (st) obs.emplace_back(t0 + t, st->velocity);
    }
    return obs;
}

// Position RMSE between the orbit re-propagated with the estimated EA and the truth
// (1 s samples, skipping the first 10 s and last 5 s)
static double eval_truth_rmse(
    const EAResult& result,
    const Epoch& t0, const Vector3d& pos, const Vector3d& vel,
    double duration_s)
{
    auto truth_cfg = make_truth_config();
    Trajectory truth(t0, pos, vel, truth_cfg);
    while (truth.epoch() - t0 < duration_s + 10) truth.step();
    TrajectoryInterpolator truth_interp(truth);

    auto recon_cfg = base_config();
    if (result.empirical_accel)
        recon_cfg.empirical_accel = result.empirical_accel;
    Trajectory recon(t0, pos, vel, recon_cfg);
    while (recon.epoch() - t0 < duration_s + 10) recon.step();
    TrajectoryInterpolator recon_interp(recon);

    double sse = 0;
    int n = 0;
    for (double t = 10; t <= duration_s - 5; t += 1.0) {
        auto sa = recon_interp.get_state(t0 + t);
        auto sb = truth_interp.get_state(t0 + t);
        if (sa && sb) {
            sse += (sa->position - sb->position).squaredNorm();
            ++n;
        }
    }
    return n > 0 ? std::sqrt(sse / n) : 1e9;
}

// =============================================================================
// Tests
// =============================================================================

int main() {
    ballistic::initialize({});

    TestRunner runner("EA Cross-Validation");

    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
    double v_circ = std::sqrt(constants::earth::GM / R);
    Vector3d pos(R, 0, 0);
    // Earth-fixed velocity of the circular orbit: inertial v_circ minus Earth
    // rotation (taking v_circ itself as Earth-fixed would give eccentricity 0.133).
    Vector3d vel(0, v_circ - constants::earth::ANGULAR_VELOCITY * R, 0);
    double arc = 300.0;  // 5-minute LEO arc

    runner.add_test("cv_position_selects_reasonable_params", [&]() {
        // Generate 10 Hz position observations (3000 points)
        auto pos_data = generate_pos_obs(t0, pos, vel, 10.0, arc);
        TEST_ASSERT(pos_data.size() > 100);

        // hv-block CV on the coarse grid (12 candidates), 10 blocks of 30 s
        // with blocks 2 and 7 held out, GN on every candidate as in the final
        // fit. (This cell used ea::CV{2} until that layout was found to hold
        // out nothing: it fell back to one fit and cross-validated nothing.)
        EABSplineConfig cv_cfg;
        ea::CV cv;
        cv.hv_n_blocks = 10;
        cv.grid = ea::CV::Grid::Coarse;
        cv_cfg.regularization = cv;
        cv_cfg.max_duration_s = arc + 10;

        // Cross-validation really happens: the layout leaves data on both sides.
        auto dep = detail::translate_cv_dep_config(cv_cfg, cv);
        auto [train, test] = detail::split_hv_block(
            pos_data, t0, pos_data.back().first - t0,
            dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
        TEST_ASSERT(!train.empty() && !test.empty());

        std::vector<std::string> hints;
        diagnostics::set_handler([&](const diagnostics::Diagnostic& d) {
            if (d.category == diagnostics::Category::ConfigHint &&
                d.message.rfind("ea::CV", 0) == 0)
                hints.push_back(d.message);
        });
        const auto wall0 = std::chrono::steady_clock::now();
        auto result = estimate_ea_bspline(
            t0, pos, vel, PositionSamples{pos_data}, base_config(), cv_cfg);
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();
        diagnostics::set_handler(nullptr);

        TEST_ASSERT(result.succeeded());
        TEST_ASSERT(result.empirical_accel != nullptr);
        // GN ran on every candidate and on the final fit, and the result says so.
        TEST_ASSERT(result.gn_status == GNStatus::Ran);

        double rmse = eval_truth_rmse(result, t0, pos, vel, arc);
        std::cout << "  CV position RMSE: " << std::fixed << std::setprecision(3)
                  << rmse << " m in " << std::setprecision(1) << wall << " s; "
                  << (hints.empty() ? std::string("interior choice")
                                    : "edge: " + hints.front()) << "\n";

        // Measured: 0.141 m in ~8 s, choosing knot spacing 10 s and
        // lambda_min 1e-12, both grid edges (GCC); MSVC chose another
        // lambda_min-edge candidate, 0.004 m. On clean data the held-out
        // scores of many candidates agree within the 1 mm tie tolerance, so
        // which one wins depends on rounding; ties go to the larger knot
        // spacing (pick_best_grid_idx). Whichever edge it is, the report must
        // name it.
        TEST_ASSERT_MSG(hints.size() == 1, "expected one grid-edge report, got " +
                                               std::to_string(hints.size()));
        TEST_ASSERT_MSG(hints.front().find("knot spacing 10 s (the largest") != std::string::npos ||
                            hints.front().find("lambda_min 1e-12 (the smallest") != std::string::npos,
                        hints.front());
        // Bound: about 1.8x the measured 0.141 m.
        TEST_ASSERT_MSG(rmse < 0.25,
            "CV position RMSE too large: " + std::to_string(rmse) + " m");
    });

    runner.add_test("cv_velocity_gn_never_sees_the_held_out_block", [&]() {
        // Velocity CV scores each candidate at held-out proxy positions (the
        // velocities integrated from the start). Its GN refinement used to
        // take exactly those held-out points as targets, so the score was
        // measured on data the candidate had been fitted to. One candidate,
        // scored three ways: by the driver's helper, by hand with training
        // targets, and by hand with the held-out targets (the old leak). The
        // helper must equal the training-target score bit for bit and differ
        // from the leaky one.
        auto vel_data = generate_vel_obs(t0, pos, vel, 1.0, arc);
        EABSplineConfig cfg;   // GN on by default
        auto dep = detail::translate_cv_dep_config(cfg, ea::CV{});
        TEST_ASSERT(dep.gn_iterations > 0);
        const double max_dur = (vel_data.back().first - t0) + 10.0;
        const double arc_blocks = vel_data.back().first - t0;
        auto pos_proxy = detail::integrate_ecef_vel_to_pos(vel_data, pos);
        auto vel_gcrs = detail::vel_ecef_to_gcrs(vel_data, pos);
        auto vsplit = detail::split_hv_block(vel_gcrs, t0, arc_blocks,
            dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
        auto psplit = detail::split_hv_block(pos_proxy, t0, arc_blocks,
            dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
        TEST_ASSERT(!vsplit.first.empty() && !psplit.first.empty() &&
                    !psplit.second.empty());
        const detail::CVHyperParams hp{4.0, 1.0, 0.5, 1e-10};
        const int degree = 5;

        const double helper = detail::cv_velocity_candidate_score(
            t0, pos, vel, vsplit.first, psplit.first, psplit.second,
            base_config(), hp, degree, arc + 10, max_dur, 0.0, dep, false);

        auto by_hand = [&](const std::vector<std::pair<Epoch, Vector3d>>& gn_targets) {
            auto bcfg = detail::make_cv_bspline_config(
                hp, base_config(), detail::CVObsType::VELOCITY, degree, arc + 10,
                0.0, dep.cv_inner_max_iterations, false, dep.drop_lambda_drift);
            auto ar = detail::estimate_empirical_acceleration_bspline_from_velocity(
                t0, pos, vel, vsplit.first, bcfg);
            TEST_ASSERT(ar.empirical_accel != nullptr);
            auto ea = detail::apply_gn_for_cv(t0, pos, vel, gn_targets, base_config(),
                                              ar.empirical_accel, dep, hp.knot_spacing,
                                              arc + 10, /*reg_order_auto=*/1);
            return detail::eval_holdout_position_rmse(
                ea, t0, pos, vel, base_config(), max_dur, psplit.second);
        };
        const double clean = by_hand(psplit.first);
        const double leaky = by_hand(psplit.second);
        std::cout << "  VEL CV held-out score: helper " << helper
                  << " m, training targets " << clean << " m, held-out targets "
                  << leaky << " m\n";
        TEST_ASSERT(std::isfinite(helper));
        TEST_ASSERT_MSG(helper == clean,
                        "the CV helper's GN must use the training targets");
        TEST_ASSERT_MSG(helper != leaky,
                        "GN on the held-out block scores differently: the "
                        "check cannot tell the two apart");
        // Which of the two scores is lower depends on GN's smoothing penalty:
        // with the third-derivative penalty CV used to apply to velocity data
        // the leaky score was the lower one; with the first-derivative penalty
        // (the velocity-data order) GN on the few held-out points alone fits
        // worse (measured 16.7 m with training targets vs 169 m). The
        // direction is therefore not asserted.
    });

    // Inside velocity cross-validation, GN's smoothing penalty acts on the
    // first derivative of the EA, the order the estimator uses for velocity
    // data outside CV (it was always the third). One candidate scored with
    // the automatic order, with order 3 forced, and with GN's penalty off
    // (reg_lambda = 0, which CV used to ignore): all three must differ.
    runner.add_test("cv_velocity_gn_penalty_order_and_weight", [&]() {
        auto vel_data = generate_vel_obs(t0, pos, vel, 1.0, arc);
        EABSplineConfig cfg;   // GN on by default, reg_deriv_order automatic
        auto dep = detail::translate_cv_dep_config(cfg, ea::CV{});
        TEST_ASSERT(dep.gn_reg_order == -1);
        TEST_ASSERT(detail::make_gn_config_from_dep(dep, base_config(), 4.0, arc + 10, 1)
                        .reg_order == 1);
        const double max_dur = (vel_data.back().first - t0) + 10.0;
        const double arc_blocks = vel_data.back().first - t0;
        auto pos_proxy = detail::integrate_ecef_vel_to_pos(vel_data, pos);
        auto vel_gcrs = detail::vel_ecef_to_gcrs(vel_data, pos);
        auto vsplit = detail::split_hv_block(vel_gcrs, t0, arc_blocks,
            dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
        auto psplit = detail::split_hv_block(pos_proxy, t0, arc_blocks,
            dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
        const detail::CVHyperParams hp{4.0, 1.0, 0.5, 1e-10};
        auto score = [&](const detail::CVDeploymentConfig& d) {
            return detail::cv_velocity_candidate_score(
                t0, pos, vel, vsplit.first, psplit.first, psplit.second,
                base_config(), hp, 5, arc + 10, max_dur, 0.0, d, false);
        };
        auto dep3 = dep;  dep3.gn_reg_order = 3;
        EABSplineConfig off = cfg;  off.gn->reg_lambda = 0.0;
        auto dep_off = detail::translate_cv_dep_config(off, ea::CV{});
        TEST_ASSERT_NEAR(dep_off.gn_reg_lambda, 0.0, 0.0);
        const double s_auto = score(dep), s3 = score(dep3), s_off = score(dep_off);
        std::cout << "  VEL CV held-out score: order 1 (automatic) " << std::fixed
                  << std::setprecision(3) << s_auto << " m, order 3 " << s3
                  << " m, reg_lambda 0 " << s_off << " m\n";
        TEST_ASSERT(std::isfinite(s_auto) && std::isfinite(s3) && std::isfinite(s_off));
        TEST_ASSERT_MSG(s_auto != s3, "the automatic order must not be order 3 for velocity data");
        TEST_ASSERT_MSG(s_auto != s_off, "reg_lambda = 0 must reach the GN runs inside CV");
    });

    // The grid-edge report: names the axis and the edge, stays quiet for an
    // interior choice.
    runner.add_test("cv_grid_edge_text", []() {
        const auto grid = detail::build_cv_grid_coarse();
        TEST_ASSERT(grid.size() == 12);
        TEST_ASSERT(detail::build_cv_grid_fixed_lambda_coarse().size() == 16);
        auto find = [&](double ks, double lm) {
            for (int i = 0; i < static_cast<int>(grid.size()); ++i)
                if (grid[i].knot_spacing == ks && grid[i].lambda_min == lm) return i;
            return -1;
        };
        TEST_ASSERT(detail::cv_grid_edges(grid, find(2.0, 1e-8), false).empty());
        const std::string lo = detail::cv_grid_edges(grid, find(1.0, 1e-8), false);
        TEST_ASSERT_MSG(lo.find("knot spacing 1 s (the smallest") != std::string::npos, lo);
        TEST_ASSERT(lo.find("lambda_min") == std::string::npos);
        const std::string both = detail::cv_grid_edges(grid, find(10.0, 1e-12), false);
        TEST_ASSERT_MSG(both.find("largest") != std::string::npos &&
                        both.find("lambda_min 1e-12 (the smallest") != std::string::npos, both);
        const auto fl = detail::build_cv_grid_fixed_lambda_coarse();
        TEST_ASSERT_MSG(detail::cv_grid_edges(fl, 15, true).find("lambda 1 (the largest")
                            != std::string::npos, detail::cv_grid_edges(fl, 15, true));
    });

    // Default tier: the public velocity path with settings chosen by
    // cross-validation, end to end, on the coarse grid -- the same noisy 1 Hz
    // velocities and block layout as the heavy cell below, which searches the
    // full grid.
    runner.add_test("cv_velocity_coarse_end_to_end", [&]() {
        auto clean = generate_vel_obs(t0, pos, vel, 1.0, arc);
        std::mt19937 rng(7);
        ballistic::test::Normal noise(0.0, 0.05);
        std::vector<std::pair<Epoch, Vector3d>> vel_data;
        for (const auto& [ep, v] : clean)
            vel_data.emplace_back(ep, v + Vector3d(noise(rng), noise(rng), noise(rng)));

        EABSplineConfig cfg;
        cfg.gn = std::nullopt;
        ea::CV cv;
        cv.hv_n_blocks = 8;
        cv.hv_test_blocks = {2, 5};
        cv.grid = ea::CV::Grid::Coarse;
        cfg.regularization = cv;
        std::vector<std::string> hints;
        diagnostics::set_handler([&](const diagnostics::Diagnostic& d) {
            if (d.category == diagnostics::Category::ConfigHint &&
                d.message.rfind("ea::CV", 0) == 0)
                hints.push_back(d.message);
        });
        const auto wall0 = std::chrono::steady_clock::now();
        auto result = estimate_ea_bspline(
            t0, pos, vel, VelocitySamples{vel_data}, base_config(), cfg);
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();
        diagnostics::set_handler(nullptr);
        TEST_ASSERT(result.succeeded());
        TEST_ASSERT(result.empirical_accel != nullptr);
        const double rmse = eval_truth_rmse(result, t0, pos, vel, arc);
        std::cout << "  VEL CV (coarse): RMSE " << std::fixed << std::setprecision(3) << rmse
                  << " m in " << std::setprecision(1) << wall << " s, final lambda "
                  << std::scientific << std::setprecision(1)
                  << result.lambda_used.value_or(-1.0) << std::fixed << "; "
                  << (hints.empty() ? std::string("interior choice")
                                    : "edge: " + hints.front()) << "\n";
        // Measured: 3.395 m in ~3.5 s (the full grid in the heavy cell: 2.687 m
        // in ~10 min). The choice is the grid's largest knot spacing (10 s) and
        // smallest lambda_min (1e-12), both edges, and the report says so.
        TEST_ASSERT_MSG(hints.size() == 1, "expected one grid-edge report, got " +
                                               std::to_string(hints.size()));
        TEST_ASSERT_MSG(hints.front().find("lambda_min 1e-12 (the smallest") !=
                            std::string::npos, hints.front());
        // Bound: 1.5x the measured 3.395 m.
        TEST_ASSERT_MSG(rmse < 5.1, "VEL CV RMSE " + std::to_string(rmse) + " m");
    });

    // After GN, collocation_history belongs to the returned (refined) EA:
    // refine_beta_from_ea / decompose_to_beta pair the two, and a history of
    // the pre-GN EA would split the EA along another orbit's velocity.
    runner.add_test("history_follows_the_gn_refined_ea", [&]() {
        auto pos_data = generate_pos_obs(t0, pos, vel, 1.0, arc);
        EABSplineConfig cfg;               // GN on (default)
        cfg.max_duration_s = arc + 10;
        auto with_gn = estimate_ea_bspline(t0, pos, vel, PositionSamples{pos_data},
                                           base_config(), cfg);
        cfg.gn = std::nullopt;
        auto no_gn = estimate_ea_bspline(t0, pos, vel, PositionSamples{pos_data},
                                         base_config(), cfg);
        TEST_ASSERT(with_gn.gn_status == GNStatus::Ran);
        TEST_ASSERT(with_gn.has_correction() && no_gn.has_correction());
        TEST_ASSERT(!with_gn.collocation_history.empty());

        auto max_gap = [&](const EAResult& r,
                           const std::shared_ptr<forces::EmpiricalAcceleration>& ea) {
            auto c = base_config();
            c.empirical_accel = ea;
            Trajectory tr(t0, pos, vel, c);
            while (tr.epoch() - t0 < arc + 10) (void)tr.step();
            TrajectoryInterpolator ti(tr);
            double worst = 0.0;
            for (const auto& h : r.collocation_history) {
                auto st = ti.get_state(h.epoch);
                if (st) worst = std::max(worst, (st->position - h.position).norm());
            }
            return worst;
        };
        const double own = max_gap(with_gn, with_gn.empirical_accel);
        const double pre = max_gap(with_gn, no_gn.empirical_accel);
        std::cout << "  history vs orbit of the refined EA " << std::scientific
                  << std::setprecision(2) << own << " m, vs orbit of the pre-GN EA "
                  << pre << " m" << std::fixed << "\n";
        TEST_ASSERT_MSG(own < 1e-3, "history is not the orbit of the returned EA: " +
                                        std::to_string(own) + " m");
        TEST_ASSERT_MSG(pre > 10.0 * own, "the check cannot tell the two orbits apart");
    });

#ifdef BALLISTIC_EA_CV_HEAVY
    // Heavy tier (test_ea_cv_heavy): the public velocity path with settings
    // chosen by cross-validation, end to end -- hv-block CV over the full IRGN
    // grid (495 candidates), then the final fit.
    // Noisy 1 Hz velocities (σ 0.05 m/s, within the Doppler range the
    // estimator is meant for); 8 blocks with blocks 2 and 5 held out; no GN
    // inside CV (GN on every candidate multiplies the run time several-fold).
    // About 10 min on 8 threads.
    //
    // Measured: re-propagated position RMSE 2.687 m. The final λ is the
    // schedule floor 1e-12, the grid's smallest λ_min -- an edge choice, so
    // whether λ is interior is printed, not asserted. The fixed-λ grid
    // (ea::CV::Mode::FixedLambda) is worse on velocity data: it chooses the
    // other edge, λ = 1, and the result is 1971 m off.
    runner.add_test("cv_velocity_end_to_end", [&]() {
        auto clean = generate_vel_obs(t0, pos, vel, 1.0, arc);
        std::mt19937 rng(7);
        ballistic::test::Normal noise(0.0, 0.05);
        std::vector<std::pair<Epoch, Vector3d>> vel_data;
        for (const auto& [ep, v] : clean)
            vel_data.emplace_back(ep, v + Vector3d(noise(rng), noise(rng), noise(rng)));

        EABSplineConfig cfg;
        cfg.gn = std::nullopt;
        ea::CV cv;
        cv.hv_n_blocks = 8;
        cv.hv_test_blocks = {2, 5};
        cfg.regularization = cv;
        auto result = estimate_ea_bspline(
            t0, pos, vel, VelocitySamples{vel_data}, base_config(), cfg);
        TEST_ASSERT(result.succeeded());
        TEST_ASSERT(result.empirical_accel != nullptr);
        TEST_ASSERT(result.lambda_used.has_value());
        const double rmse = eval_truth_rmse(result, t0, pos, vel, arc);
        const double lam = result.lambda_used.value_or(-1.0);
        std::cout << "  VEL CV: RMSE " << std::fixed << std::setprecision(3) << rmse
                  << " m, final lambda " << std::scientific << std::setprecision(1) << lam
                  << std::fixed << (lam > 1.5e-12 && lam < 1e-2 / 1.5 ? " (interior)" : " (grid edge)")
                  << "\n";
        // Bound: 1.5x the measured 2.687 m.
        TEST_ASSERT_MSG(rmse < 4.0, "VEL CV RMSE " + std::to_string(rmse) + " m");
    });
#endif

    runner.add_test("kfold_endpoints_always_in_training", [&]() {
        // Verify the K-fold split keeps endpoints in training
        using Obs = std::pair<Epoch, Vector3d>;
        std::vector<Obs> data;
        for (int i = 0; i < 20; ++i)
            data.emplace_back(t0 + i, Vector3d(i, 0, 0));

        auto split = detail::split_kfold(data, 5);

        // Endpoints should be in the endpoints set
        TEST_ASSERT(split.endpoints.size() == 2);
        TEST_ASSERT((split.endpoints[0].second - Vector3d(0, 0, 0)).norm() < 1e-10);
        TEST_ASSERT((split.endpoints[1].second - Vector3d(19, 0, 0)).norm() < 1e-10);

        // No fold should contain the first or last observation
        for (int f = 0; f < 5; ++f) {
            for (const auto& [ep, v] : split.folds[f]) {
                TEST_ASSERT_MSG(v.x() > 0.5 && v.x() < 18.5,
                    "Fold " + std::to_string(f) + " contains endpoint x=" +
                    std::to_string(v.x()));
            }
        }

        // Merged training for any fold should contain endpoints
        for (int f = 0; f < 5; ++f) {
            auto train = detail::merge_train_folds(split, f);
            bool has_first = false, has_last = false;
            for (const auto& [ep, v] : train) {
                if (v.x() < 0.5) has_first = true;
                if (v.x() > 18.5) has_last = true;
            }
            TEST_ASSERT_MSG(has_first, "Fold " + std::to_string(f) + " training missing first endpoint");
            TEST_ASSERT_MSG(has_last, "Fold " + std::to_string(f) + " training missing last endpoint");
        }
    });

    runner.run();
    return print_final_summary();
}
