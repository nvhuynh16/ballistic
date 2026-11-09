// =============================================================================
// estimate_ea_smoother — integration gate
// =============================================================================
//
// Gates estimate_ea_smoother, the multi-sensor empirical-acceleration (EA)
// estimator. EA = the extra acceleration, beyond the force model, that is
// fitted so the propagated trajectory matches the observations. The smoother
// is a Gauss-Markov Rauch-Tung-Striebel (GM-RTS) smoother: a Kalman filter run
// forward, then a backward pass, with the EA modeled as a first-order
// Gauss-Markov process (strength σ_a, correlation time τ).
//
// Data: the glide fixture (the synthetic reentry glide from 120 km in
// tests/test_helpers.hpp, with a known truth EA) observed by one radar at
// 160 Hz (range, azimuth, elevation, range rate; noisy, seed 42) and one
// position fix near impact. Metric: "reprop" = position RMSE (root-mean-square
// error) of the trajectory re-propagated with the returned EA, against the truth.
//
// Measured on this setup: reprop 12.7 m with σ_a/τ picked automatically by NIS
// (normalized innovation squared per degree of freedom; about 1 when the
// declared noise matches the residuals).
// The information limit for this geometry is about 9-17 m (posterior /
// Cramér-Rao bound, `information_bound`). Gates carry margin for compiler and
// platform differences rather than sitting at the measured values; the smoother
// is not sensitive to floating-point rounding, so a large move means a real
// regression.
// =============================================================================

#include "../portable_random.hpp"
#include "../test_helpers.hpp"
#include "../test_utils.hpp"

#include <ballistic/init.hpp>
#include <ballistic/calibration/ea_smoother.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/coordinates/iau_transforms.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>

#include <cmath>
#include <iostream>
#include <random>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;
namespace glide_helpers = ballistic::test::helpers::glide;
namespace radar_helpers = ballistic::test::helpers::radar;

int main() {
    ballistic::initialize({});
    TestRunner runner("EA Smoother (GM-RTS) integration");

    auto glide = glide_helpers::make_glide();

    // Baseline scenario: one radar placed relative to the impact point (seed 42)
    // plus one 3 m position fix 0.5 s before the end of the arc.
    MultiSensorScenario scenario;
    scenario.epoch0 = glide.t0;
    scenario.pos0_ecef = glide.pos;
    scenario.vel0_ecef = glide.vel;
    scenario.arc_duration_s = glide.eval_end;
    scenario.add_radar(radar_helpers::build_radar_for_glide(
        glide, 20'000.0, 160.0, glide.eval_end, /*noise=*/true, /*seed=*/42));
    {
        PositionFixObs fix;
        fix.sigma_pos = 3.0;
        auto st = glide.truth_interp->get_state(glide.t0 + glide.eval_end - 0.5);
        std::mt19937 rng(44);
        ballistic::test::Normal n(0.0, fix.sigma_pos);
        if (st)
            fix.samples.emplace_back(
                glide.t0 + glide.eval_end - 0.5,
                Vector3d(st->position + Vector3d(n(rng), n(rng), n(rng))));
        scenario.add_position_fix(std::move(fix));
    }
    SensorMask mask = SensorMask::all_on(scenario);
    TrajectoryConfig fm = glide_helpers::base_config();

    EASmootherResult result;
    double combo1_rmse = -1.0;

    runner.add_test("smoother_combo1_reprop_and_health", [&]() {
        EASmootherConfig cfg;   // all-auto: NIS grid + measured rate budget
        result = estimate_ea_smoother(scenario, mask, fm, cfg);
        TEST_ASSERT_MSG(result.error_message.empty(),
                        "smoother error: " + result.error_message);
        TEST_ASSERT(result.succeeded());

        double rmse = glide_helpers::reprop_rmse(glide, result.empirical_accel);
        combo1_rmse = rmse;
        std::cout << "  smoother combo1: reprop = " << rmse
                  << " m  (σ_a=" << result.sigma_a_used
                  << ", τ=" << result.tau_used
                  << ", σ_rate_eff=" << result.sigma_rate_eff
                  << ", NIS/dof=" << result.nis << ")\n";
        // Measured 12.7 m; information limit about 9-17 m; gate 25 m.
        TEST_ASSERT_MSG(rmse < 25.0, "smoother reprop: " + std::to_string(rmse));

        // Health signals that need no truth: NIS/dof in [0.3, 3].
        TEST_ASSERT_MSG(result.nis > 0.3 && result.nis < 3.0,
                        "NIS/dof out of band: " + std::to_string(result.nis));
        // The range-rate noise the smoother measures from the data must exceed
        // the declared sensor σ (0.02 m/s): the forward model cannot match
        // range rate to the sensor's precision, so the declared σ is never used
        // as-is.
        TEST_ASSERT_MSG(result.sigma_rate_eff > scenario.radars[0].sigma_rate,
                        "rate budget not measured/honest");
        // Posterior σ present, finite, parallel to the returned EA keys.
        TEST_ASSERT(!result.ea_sigma.empty());
        for (const auto& s : result.ea_sigma)
            TEST_ASSERT(s.allFinite() && s.minCoeff() >= 0.0);
        // Per-channel residual RMS ("honored σ", compared with the declared σ)
        // is filled in.
        TEST_ASSERT(!result.sigma_honored.empty());
        for (const auto& [ch, v] : result.sigma_honored) {
            std::cout << "    honored " << ch << " = " << v << "\n";
            TEST_ASSERT(std::isfinite(v) && v >= 0.0);
        }
    });

    runner.add_test("known_bias_exact_neutralization", [&]() {
        // A KNOWN bias declared on the sensor must be exactly invisible: add
        // +25 m to every range sample AND declare it. Subtracting it restores the
        // original observations bit for bit, so the fit (same automatic config as
        // the first cell) must reproduce the first cell's reprop to 1e-9 m.
        MultiSensorScenario s2 = scenario;
        for (auto& [t, v] : s2.radars[0].samples) v(0) += 25.0;
        s2.radars[0].known_bias(0) = 25.0;
        auto r2 = estimate_ea_smoother(s2, SensorMask::all_on(s2), fm, {});
        TEST_ASSERT(r2.error_message.empty());
        double rmse = glide_helpers::reprop_rmse(glide, r2.empirical_accel);
        std::cout << "  known-bias neutralization: reprop = " << rmse
                  << " m (combo1 " << combo1_rmse << ")\n";
        TEST_ASSERT_MSG(std::abs(rmse - combo1_rmse) < 1e-9,
                        "declared bias not exactly neutralized: " +
                            std::to_string(rmse));
    });

    runner.add_test("sigma_misdeclaration_flagged", [&]() {
        // Declare the range σ 10× too small (0.2 m vs the true 2.0 m noise). The
        // estimate may degrade; what is gated is that the health signals FLAG
        // it: NIS/dof > 3 and the honored range RMS > 5× the declared σ. A bad
        // result that is flagged is acceptable; a silently wrong one is not.
        MultiSensorScenario s2 = scenario;
        s2.radars[0].sigma_range = 0.2;
        EASmootherConfig cfg2;             // pinned prior: single fit
        cfg2.sigma_a = 3.0;
        cfg2.tau_s = 35.0;
        auto r2 = estimate_ea_smoother(s2, SensorMask::all_on(s2), fm, cfg2);
        TEST_ASSERT(r2.error_message.empty());
        double honored_range = -1.0;
        for (const auto& [ch, v] : r2.sigma_honored)
            if (ch == "radar0_range") honored_range = v;
        std::cout << "  σ mis-declared: NIS/dof = " << r2.nis
                  << "  honored range RMS = " << honored_range
                  << " m (declared 0.2)\n";
        TEST_ASSERT_MSG(r2.nis > 3.0, "mis-declared σ not flagged by NIS");
        TEST_ASSERT_MSG(honored_range > 5.0 * 0.2,
                        "honored σ does not expose the mis-declaration");
    });

    runner.add_test("smoother_camera_fusion", [&]() {
        // Add a co-located 50 µrad optical tracker (angles only). The
        // information bound predicted the limit moves from about 9 to about
        // 2.3 m; measured 3.0 m, within 1.3× of the bound. The camera alone
        // cannot work (angles leave range unobservable), it complements the
        // radar. Gate 6 m.
        MultiSensorScenario s2 = scenario;
        {
            auto st = glide.truth_interp->get_state(
                glide.t0 + glide.eval_end - 0.5);
            Vector3d site = radar_helpers::place_sensor_east(
                st ? st->position : glide.pos, 20'000.0);
            auto cam = radar_helpers::make_optics(site);
            cam.samples = radar_helpers::generate_optics_samples(
                cam, *glide.truth_interp, glide.t0, 160.0, glide.eval_end, 43);
            s2.add_optics(std::move(cam));
        }
        auto r2 = estimate_ea_smoother(s2, SensorMask::all_on(s2), fm, {});
        TEST_ASSERT_MSG(r2.error_message.empty(),
                        "camera fusion error: " + r2.error_message);
        double rmse = glide_helpers::reprop_rmse(glide, r2.empirical_accel);
        std::cout << "  smoother +camera: reprop = " << rmse
                  << " m  NIS/dof = " << r2.nis << "\n";
        TEST_ASSERT_MSG(rmse < 6.0, "camera-fusion reprop: " + std::to_string(rmse));
        TEST_ASSERT(r2.nis > 0.3 && r2.nis < 3.0);
    });

    runner.add_test("own_clock_isolates_biased_radar", [&]() {
        // Radar time tags shifted by +5 ms and declared clock=Own (its own clock
        // offset state), fused with a CLEAN camera declared clock=None (the
        // default). The radar's clock state must absorb its timing error without
        // polluting the optics channel: the honored optics angle RMS stays at the
        // sensor level.
        MultiSensorScenario s2 = scenario;
        for (auto& [t, v] : s2.radars[0].samples) t = t + 0.005;
        s2.radars[0].clock = SensorClock::Own;
        {
            auto st = glide.truth_interp->get_state(
                glide.t0 + glide.eval_end - 0.5);
            Vector3d site = radar_helpers::place_sensor_east(
                st ? st->position : glide.pos, 20'000.0);
            auto cam = radar_helpers::make_optics(site);
            cam.samples = radar_helpers::generate_optics_samples(
                cam, *glide.truth_interp, glide.t0, 160.0, glide.eval_end, 43);
            s2.add_optics(std::move(cam));
        }
        auto r2 = estimate_ea_smoother(s2, SensorMask::all_on(s2), fm, {});
        TEST_ASSERT(r2.error_message.empty());
        double rmse = glide_helpers::reprop_rmse(glide, r2.empirical_accel);
        double clock_radar = 0.0, honored_cam = -1.0;
        for (const auto& [lbl, v] : r2.clock_biases)
            if (lbl == "radar0") clock_radar = v;
        for (const auto& [ch, v] : r2.sigma_honored)
            if (ch == "optics0_angle") honored_cam = v;
        std::cout << "  own-clock: reprop = " << rmse
                  << " m  radar0 clock = " << clock_radar * 1e3
                  << " ms  honored cam angle = " << honored_cam * 1e6
                  << " µrad  NIS/dof = " << r2.nis << "\n";
        // Measured 44.9 m: the clock state rescues most of the timing damage
        // but not all of it; the camera's benefit here is that its channel
        // stays clean, not a better trajectory. Gate 60 m.
        TEST_ASSERT_MSG(rmse < 60.0, "own-clock reprop: " + std::to_string(rmse));
        // The radar's clock state engaged (nonzero estimate; its scale and sign
        // are not checked).
        TEST_ASSERT_MSG(std::abs(clock_radar) > 1e-4,
                        "radar Own clock state did not engage");
        // Optics unpolluted: honored angle within 3× the 50 µrad sensor σ.
        TEST_ASSERT_MSG(honored_cam > 0.0 && honored_cam < 150e-6,
                        "optics channel polluted: " + std::to_string(honored_cam));
    });

    runner.add_test("smoother_accel_fusion_with_bias", [&]() {
        // Add a 160 Hz accelerometer: truth EA + constant bias + noise, rotated
        // into ECEF (Earth-fixed) with the full GCRS-to-ITRF rotation, the same
        // convention the smoother's observation model uses.
        MultiSensorScenario s2 = scenario;
        AccelObs acc;
        acc.sigma_accel = 0.01;
        acc.sigma_bias_prior = 1e-3;
        const Vector3d true_bias(1.0e-4, -2.0e-4, 5.0e-5);
        auto truth_ea = glide.truth_traj->force_model().empirical_acceleration();
        std::mt19937 rng2(45);
        ballistic::test::Normal na(0.0, acc.sigma_accel);
        for (double t = 0.0; t <= glide.eval_end; t += 1.0 / 160.0) {
            Matrix3d R_g2e = coords::detail::get_cached_transform()
                                 .gcrs_to_itrf_matrix(glide.t0 + t);
            Vector3d obs = R_g2e * truth_ea->accel_at(t) + true_bias
                         + Vector3d(na(rng2), na(rng2), na(rng2));
            acc.samples.emplace_back(glide.t0 + t, obs);
        }
        s2.add_accelerometer(std::move(acc));

        // Pin σ_a/τ: the first cell already gates the automatic grid; this
        // cell gates the accelerometer observation model with a single fit.
        EASmootherConfig cfg2;
        cfg2.sigma_a = 3.0;
        cfg2.tau_s = 35.0;
        auto r2 = estimate_ea_smoother(s2, SensorMask::all_on(s2), fm, cfg2);
        TEST_ASSERT_MSG(r2.error_message.empty(),
                        "accel fusion error: " + r2.error_message);
        double rmse = glide_helpers::reprop_rmse(glide, r2.empirical_accel);
        double bias_err = (r2.recovered_bias - true_bias).norm();
        std::cout << "  smoother +accel: reprop = " << rmse
                  << " m  bias_err = " << bias_err
                  << "  NIS/dof = " << r2.nis << "\n";
        // Measured 5.3 m. Not sub-meter: there is no camera, and a constant
        // accelerometer bias cannot be told apart from a constant EA offset; at
        // the bias prior σ_b = 1e-3 m/s² that ambiguity allows up to about 17 m
        // of reprop error. The accelerometer still improves the radar + fix
        // result (12.7 m) by 2.4×. Gate 8 m.
        TEST_ASSERT_MSG(rmse < 8.0, "accel-fusion reprop: " + std::to_string(rmse));
        TEST_ASSERT_MSG(bias_err < 5e-4, "bias error: " + std::to_string(bias_err));
        TEST_ASSERT(r2.nis > 0.3 && r2.nis < 3.0);
    });

    runner.add_test("regime_aware_prior_no_glide_regression", [&]() {
        // regime_aware_prior anchors the σ_a candidate grid on a scale taken from
        // the data (a fit with the EA frozen) instead of fixed literals tuned on
        // glides; it is meant for orbital arcs. It must
        // not cost anything on the glide: gate at 1.3× the default result here
        // and the documented 3.1 m (camera) and 5.3 m (accelerometer) values.
        EASmootherConfig cfg;
        cfg.regime_aware_prior = true;
        auto r = estimate_ea_smoother(scenario, mask, fm, cfg);
        TEST_ASSERT_MSG(r.error_message.empty(), r.error_message);
        double rmse = glide_helpers::reprop_rmse(glide, r.empirical_accel);
        std::cout << "  regime-aware glide: anchor=" << r.prior_anchor
                  << " σ_a=" << r.sigma_a_used << " τ=" << r.tau_used
                  << " reprop=" << rmse << " m (default " << combo1_rmse
                  << ") NIS=" << r.nis << "\n";
        TEST_ASSERT_MSG(rmse < 1.3 * combo1_rmse,
                        "regime-aware prior regressed the glide fixture: " +
                            std::to_string(rmse) + " vs " +
                            std::to_string(combo1_rmse));
        TEST_ASSERT(r.nis > 0.3 && r.nis < 3.0);
        TEST_ASSERT_MSG(r.prior_anchor > 0.0, "anchor not computed");

        // The same rule for the two fusion cases: camera and accelerometer.
        MultiSensorScenario s_cam = scenario;
        {
            auto st = glide.truth_interp->get_state(
                glide.t0 + glide.eval_end - 0.5);
            Vector3d site = radar_helpers::place_sensor_east(
                st ? st->position : glide.pos, 20'000.0);
            auto cam = radar_helpers::make_optics(site);
            cam.samples = radar_helpers::generate_optics_samples(
                cam, *glide.truth_interp, glide.t0, 160.0, glide.eval_end, 43);
            s_cam.add_optics(std::move(cam));
        }
        auto r_cam = estimate_ea_smoother(s_cam, SensorMask::all_on(s_cam),
                                          fm, cfg);
        TEST_ASSERT(r_cam.error_message.empty());
        double rmse_cam = glide_helpers::reprop_rmse(glide,
                                                     r_cam.empirical_accel);
        std::cout << "  regime-aware +camera: reprop=" << rmse_cam
                  << " m (documented 3.1)\n";
        TEST_ASSERT_MSG(rmse_cam < 1.3 * 3.1,
                        "regime-aware regressed the camera arm: " +
                            std::to_string(rmse_cam));

        // Accel arm (the fusion case with a joint bias state).
        MultiSensorScenario s_acc = scenario;
        {
            AccelObs acc;
            acc.sigma_accel = 0.01;
            acc.sigma_bias_prior = 1e-3;
            const Vector3d true_bias(1.0e-4, -2.0e-4, 5.0e-5);
            auto truth_ea =
                glide.truth_traj->force_model().empirical_acceleration();
            std::mt19937 rng2(45);
            ballistic::test::Normal na(0.0, acc.sigma_accel);
            for (double t = 0.0; t <= glide.eval_end; t += 1.0 / 160.0) {
                Matrix3d R_g2e = coords::detail::get_cached_transform()
                                     .gcrs_to_itrf_matrix(glide.t0 + t);
                Vector3d obs = R_g2e * truth_ea->accel_at(t) + true_bias
                             + Vector3d(na(rng2), na(rng2), na(rng2));
                acc.samples.emplace_back(glide.t0 + t, obs);
            }
            s_acc.add_accelerometer(std::move(acc));
        }
        auto r_acc = estimate_ea_smoother(s_acc, SensorMask::all_on(s_acc),
                                          fm, cfg);
        TEST_ASSERT(r_acc.error_message.empty());
        double rmse_acc = glide_helpers::reprop_rmse(glide,
                                                     r_acc.empirical_accel);
        std::cout << "  regime-aware +accel: reprop=" << rmse_acc
                  << " m (documented 5.3)\n";
        TEST_ASSERT_MSG(rmse_acc < 1.3 * 5.3,
                        "regime-aware regressed the accel arm: " +
                            std::to_string(rmse_acc));
    });

    runner.add_test("grid_edge_warning_semantics", [&]() {
        // Grid-edge warning (report only): when the automatic grid picks a
        // BOUNDARY value, the best value may lie outside the grid. On an orbital
        // arc this gave 61 m reprop vs 0.157 m with no EA, while NIS looked
        // clean. These checks gate WHEN the warning appears, not what it changes.
        auto has_edge_warning = [](const EASmootherResult& r) {
            for (const auto& w : r.warnings)
                if (w.find("BOUNDARY") != std::string::npos) return true;
            return false;
        };

        // (a) Both axes pinned → grid.size() == 1 → never warns.
        EASmootherConfig pinned;
        pinned.sigma_a = 3.0;
        pinned.tau_s = 35.0;
        auto r_pinned = estimate_ea_smoother(scenario, mask, fm, pinned);
        TEST_ASSERT(r_pinned.error_message.empty());
        TEST_ASSERT_MSG(!has_edge_warning(r_pinned),
                        "pinned config must not emit a grid-edge warning");

        // (b) One axis pinned → only the auto axis can warn; a pinned axis
        //     is degenerate (single candidate) and must never be reported.
        EASmootherConfig half;
        half.sigma_a = 3.0;   // pinned to the grid's INTERIOR value
        auto r_half = estimate_ea_smoother(scenario, mask, fm, half);
        TEST_ASSERT(r_half.error_message.empty());
        for (const auto& w : r_half.warnings)
            TEST_ASSERT_MSG(w.find("sigma_a=") == std::string::npos,
                            "pinned sigma_a must not be reported as an edge");

        // (c) All-auto on the glide fixture: whatever it selects, the
        //     warning must agree with the reported values — no false
        //     alarm on an interior winner (3.0 / 35.0 are interior).
        const bool interior = (std::abs(result.sigma_a_used - 3.0) < 1e-12 &&
                               std::abs(result.tau_used - 35.0) < 1e-12);
        if (interior)
            TEST_ASSERT_MSG(!has_edge_warning(result),
                            "interior winner must not warn");
        std::cout << "  grid-edge: glide all-auto σ_a=" << result.sigma_a_used
                  << " τ=" << result.tau_used << " warned="
                  << has_edge_warning(result) << "\n";
    });

    runner.add_test("closed_loop_data_consistency", [&]() {
        // Feed the RETURNED EA back through Trajectory and require the
        // re-propagated trajectory to sit near the observations at the declared
        // noise level. Measured on this fixture: range about 1.65× declared, fix
        // about 1.2×, angle 1.00×, rate 0.57× of the measured rate budget. The
        // excess over the smoothed fit's residuals comes from how densely the
        // returned EA is tabulated (output_dt_s).
        TEST_ASSERT(result.succeeded());   // reuses the first cell's fit
        TrajectoryConfig rc = fm;
        rc.empirical_accel = result.empirical_accel;
        Trajectory rt(scenario.epoch0, scenario.pos0_ecef, scenario.vel0_ecef,
                      rc);
        while (rt.epoch() - scenario.epoch0 < scenario.arc_duration_s)
            (void)rt.step();
        InterpolatorConfig ic;
        ic.output_frame = InterpolationFrame::GCRS;
        TrajectoryInterpolator ri(rt, ic);
        calibration::detail::SmootherRun fake;
        auto obs = calibration::detail::build_timeline(
            scenario, SensorMask::all_on(scenario));
        for (const auto& o : obs) {
            auto st = ri.get_state(scenario.epoch0 + o.t);
            if (!st) continue;
            Eigen::VectorXd z = Eigen::VectorXd::Zero(9);
            z.segment<3>(0) = st->position;
            z.segment<3>(3) = st->velocity;
            fake.t.push_back(o.t);
            fake.z_smooth.push_back(std::move(z));
        }
        auto hr = calibration::detail::measure_honored_sigmas(
            scenario, SensorMask::all_on(scenario),
            coords::ITRFRealization::ITRF2020, fake);
        double rng_r = -1, ang_r = -1, rate_r = -1, fix_r = -1;
        for (const auto& [ch, v] : hr) {
            if (ch == "radar0_range") rng_r = v;
            if (ch == "radar0_angle") ang_r = v;
            if (ch == "radar0_rate") rate_r = v;
            if (ch == "fix0_pos") fix_r = v;
        }
        std::cout << "  closed loop: range " << rng_r << " (σ=2)  angle "
                  << ang_r << " (σ=3.5e-4)  rate " << rate_r << " (budget "
                  << result.sigma_rate_eff << ")  fix " << fix_r
                  << " (σ=3)\n";
        TEST_ASSERT_MSG(rng_r > 0 && rng_r < 2.0 * scenario.radars[0].sigma_range,
                        "reprop range residuals exceed 2x declared σ");
        TEST_ASSERT_MSG(ang_r > 0 && ang_r < 2.0 * scenario.radars[0].sigma_angle,
                        "reprop angle residuals exceed 2x declared σ");
        TEST_ASSERT_MSG(rate_r > 0 && rate_r < 1.5 * result.sigma_rate_eff,
                        "reprop rate residuals exceed 1.5x the honest budget");
        TEST_ASSERT_MSG(fix_r > 0 && fix_r < 2.0 * scenario.fixes[0].sigma_pos,
                        "reprop fix residuals exceed 2x declared σ");
    });

    runner.add_test("gap_is_flagged_and_covered", [&]() {
        // A 40 s radar outage ruins the reprop without NIS noticing (NIS/dof
        // stays in band, measured 0.8): the EA in the gap is simply
        // unobservable, not an estimator defect. This cell locks three measured
        // facts: (1) the "prior-dominated" WARNING fires, the only signal;
        // (2) NIS stays in band (it cannot see the gap); (3) the posterior σ at
        // the gap center covers the actual error (measured z = 1.6).
        MultiSensorScenario s2 = scenario;
        auto& samples = s2.radars[0].samples;
        samples.erase(std::remove_if(samples.begin(), samples.end(),
                                     [&](const auto& s) {
                                         double t = s.first - glide.t0;
                                         return t >= 60.0 && t <= 100.0;
                                     }),
                      samples.end());
        auto r = estimate_ea_smoother(s2, SensorMask::all_on(s2), fm, {});
        TEST_ASSERT(r.error_message.empty());

        bool flagged = false;
        for (const auto& w : r.warnings)
            if (w.find("prior-dominated") != std::string::npos) flagged = true;
        TEST_ASSERT_MSG(flagged, "gap warning did not fire");
        TEST_ASSERT_MSG(r.nis > 0.3 && r.nis < 3.0,
                        "NIS out of band — the gap-blindness premise changed; "
                        "re-measure the radar-gap case");

        // Gap-center calibration (t_c = 80 s; measured z = 1.6).
        auto truth_ea = glide.truth_traj->force_model().empirical_acceleration();
        const auto& keys = r.empirical_accel->keys();
        size_t ic = 0;
        for (size_t i = 0; i < keys.size(); ++i)
            if (std::abs(keys[i] - 80.0) < std::abs(keys[ic] - 80.0)) ic = i;
        const double z =
            (r.empirical_accel->accel_at(keys[ic]) -
             truth_ea->accel_at(keys[ic])).norm() /
            std::max(1e-12, r.ea_sigma[ic].norm());
        std::cout << "  gap 40 s: NIS = " << r.nis << "  z(80 s) = " << z
                  << "  warning fired = " << flagged << "\n";
        TEST_ASSERT_MSG(z < 2.5, "gap-center sigma coverage lost: z = " +
                                     std::to_string(z));
    });

    runner.add_test("robust_accel_mixture_update", [&]() {
        // Robust accelerometer update: 5% of samples are outliers at 10σ
        // (production path, accelerometer bias state active). The outlier
        // mixture is declared WRONG on purpose (p = 0.02, k = 5 vs the true
        // 0.05, 10), to show it helps even when mis-declared.
        // Measured here: plain least squares (L2) 21.8 m, mis-declared mixture
        // 11.2 m.
        // Gate: robust < 0.75 × L2 (measured ratio about 0.5).
        auto truth_ea = glide.truth_traj->force_model().empirical_acceleration();
        AccelObs acc;
        acc.sigma_accel = 0.01;
        acc.sigma_bias_prior = 1e-3;
        std::mt19937 rng(18);
        ballistic::test::Normal na(0.0, acc.sigma_accel);
        ballistic::test::UniformReal u01(0.0, 1.0);
        for (double t = 0.0; t <= glide.eval_end; t += 1.0 / 160.0) {
            Matrix3d R_g2e = coords::detail::get_cached_transform()
                                 .gcrs_to_itrf_matrix(glide.t0 + t);
            const double s = (u01(rng) < 0.05) ? 10.0 : 1.0;
            Vector3d obs = R_g2e * truth_ea->accel_at(t)
                         + s * Vector3d(na(rng), na(rng), na(rng));
            acc.samples.emplace_back(glide.t0 + t, obs);
        }
        MultiSensorScenario s_l2 = scenario;
        s_l2.add_accelerometer(acc);
        MultiSensorScenario s_rob = scenario;
        acc.outlier_prob = 0.02;    // mis-declared on purpose (true 0.05)
        acc.outlier_scale = 5.0;    // mis-declared on purpose (true 10)
        s_rob.add_accelerometer(std::move(acc));

        EASmootherConfig cfg;
        cfg.sigma_a = 3.0;   // pinned prior: variants on identical footing
        cfg.tau_s = 35.0;
        auto r_l2 = estimate_ea_smoother(s_l2, SensorMask::all_on(s_l2), fm, cfg);
        auto r_rob = estimate_ea_smoother(s_rob, SensorMask::all_on(s_rob), fm, cfg);
        TEST_ASSERT(r_l2.error_message.empty() && r_rob.error_message.empty());
        double rp_l2 = glide_helpers::reprop_rmse(glide, r_l2.empirical_accel);
        double rp_rob = glide_helpers::reprop_rmse(glide, r_rob.empirical_accel);
        std::cout << "  robust accel: L2 = " << rp_l2 << " m  mixture-ψ = "
                  << rp_rob << " m  (ratio " << rp_rob / rp_l2 << ")\n";
        TEST_ASSERT_MSG(rp_rob < 0.75 * rp_l2,
                        "robust update did not beat L2: " +
                            std::to_string(rp_rob) + " vs " +
                            std::to_string(rp_l2));
    });

    runner.add_test("rate_budget_reads_every_radar", [&]() {
        // The measured range-rate budget is the RMS rate residual over the
        // samples of ALL active radars. Two radars sampling interleaved times
        // (0, 1, 2, ... s and 0.5, 1.5, ... s): radar 0's rates match the
        // state exactly, radar 1's are 1 m/s high. The RMS must be
        // sqrt(1/2). It used to come out sqrt(1/11) = 0.30: one search
        // position into the timeline was shared by all radars, so of radar 1's
        // samples only the one after radar 0's last sample was counted.
        const Epoch e0 = glide.t0;
        const Vector3d r_g(6'500'000.0, 1'000'000.0, 500'000.0);
        const Vector3d v_g(-1'000.0, 6'000.0, 2'000.0);
        MultiSensorScenario two;
        two.epoch0 = e0;
        RadarObs r0 = scenario.radars[0];
        RadarObs r1 = scenario.radars[0];
        r1.position_ecef += Vector3d(50'000.0, 0.0, 0.0);
        r0.samples.clear();
        r1.samples.clear();
        calibration::detail::SmootherRun run;
        Eigen::VectorXd z = Eigen::VectorXd::Zero(9);
        z.segment<3>(0) = r_g;
        z.segment<3>(3) = v_g;
        for (int k = 0; k < 20; ++k) {
            const double t = 0.5 * k;
            const StateVector st_e = util::gcrs_to_itrf(
                StateVector{r_g, v_g}, e0 + t, coords::ITRFRealization::ITRF2020);
            RadarObs& rad = (k % 2 == 0) ? r0 : r1;
            Eigen::Vector4d h = rad.predict(st_e);
            if (k % 2 == 1) h(3) += 1.0;
            rad.samples.emplace_back(e0 + t, h);
            run.t.push_back(t);
            run.z_smooth.push_back(z);
        }
        two.add_radar(std::move(r0));
        two.add_radar(std::move(r1));
        const double rms = calibration::detail::smoother_rate_residual_rms(
            two, SensorMask::all_on(two), coords::ITRFRealization::ITRF2020, run);
        std::cout << "  two-radar rate RMS = " << rms << " m/s (expect "
                  << std::sqrt(0.5) << ")\n";
        TEST_ASSERT_NEAR(rms, std::sqrt(0.5), 1e-6);
    });

    runner.add_test("two_radars_beat_one", [&]() {
        // The baseline scenario plus a second radar of the same type and noise
        // (σ 2 m / 0.02 deg / 0.02 m/s, 160 Hz, its own noise seed) 20 km WEST
        // of the impact point, so the two lines of sight cross. A second
        // viewing direction constrains the transverse position the first radar
        // sees only to σ_angle × range, so the fit must not get worse.
        MultiSensorScenario two = scenario;
        two.add_radar(radar_helpers::build_radar_for_glide(
            glide, -20'000.0, 160.0, glide.eval_end, /*noise=*/true, /*seed=*/43));
        auto r2 = estimate_ea_smoother(two, SensorMask::all_on(two), fm, {});
        TEST_ASSERT_MSG(r2.error_message.empty(), "smoother error: " + r2.error_message);
        TEST_ASSERT(r2.succeeded());
        double one = combo1_rmse;
        if (one < 0.0) {   // first cell filtered out: fit the one-radar case here
            one = glide_helpers::reprop_rmse(
                glide, estimate_ea_smoother(scenario, mask, fm, {}).empirical_accel);
        }
        const double rmse = glide_helpers::reprop_rmse(glide, r2.empirical_accel);
        std::cout << "  two radars: reprop = " << rmse << " m (one radar " << one
                  << " m; σ_a=" << r2.sigma_a_used << ", τ=" << r2.tau_used
                  << ", NIS/dof=" << r2.nis << ")\n";
        TEST_ASSERT_MSG(rmse <= one, "two radars worse than one: " +
                        std::to_string(rmse) + " vs " + std::to_string(one) + " m");
        // Measured 10.6 m (one radar 12.7 m); gate 1.5x the measured value.
        TEST_ASSERT_MSG(rmse < 16.0, "two-radar reprop: " + std::to_string(rmse));
    });

    runner.run();
    return print_final_summary();
}
