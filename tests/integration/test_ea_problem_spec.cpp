// =============================================================================
// estimate_ea(spec) — the declarative front door
// =============================================================================
//
// estimate_ea takes the observations plus a declared error model and picks the
// empirical-acceleration (EA) estimator itself. EA = the extra acceleration,
// beyond the force model, fitted so the propagated trajectory matches the
// observations. This file gates the routing and the audit of the declarations,
// not estimator accuracy (that has its own tests):
//   - a multi-sensor scenario goes to estimate_ea_smoother and gives the same
//     result as calling it directly;
//   - typed samples (positions, velocities, accelerations) go to the B-spline
//     estimator, with the declared-σ audit filled in;
//   - a missing noise declaration is an immediate error.
// Data: the glide fixture (the synthetic reentry glide from 120 km in
// tests/test_helpers.hpp) and synthetic radar / camera / accelerometer samples.
// =============================================================================

#include "../portable_random.hpp"
#include "../test_helpers.hpp"
#include "../test_utils.hpp"

#include <ballistic/init.hpp>
#include <ballistic/calibration/ea_problem_spec.hpp>

#include <iomanip>
#include <iostream>
#include <random>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;
namespace glide_helpers = ballistic::test::helpers::glide;
namespace radar_helpers = ballistic::test::helpers::radar;

int main() {
    ballistic::initialize({});
    TestRunner runner("EAProblemSpec front door");

    auto glide = glide_helpers::make_glide();
    TrajectoryConfig fm = glide_helpers::base_config();

    runner.add_test("scenario_dispatches_to_smoother_identically", [&]() {
        MultiSensorScenario scenario;
        scenario.epoch0 = glide.t0;
        scenario.pos0_ecef = glide.pos;
        scenario.vel0_ecef = glide.vel;
        scenario.arc_duration_s = glide.eval_end;
        scenario.add_radar(radar_helpers::build_radar_for_glide(
            glide, 20'000.0, 160.0, glide.eval_end, true, 42));

        EAProblemSpec spec;
        spec.observations = scenario;
        spec.force_model = fm;
        auto r = estimate_ea(spec);
        TEST_ASSERT_MSG(r.error_message.empty(), r.error_message);
        TEST_ASSERT(r.method_used == EAMethod::Smoother);
        TEST_ASSERT(r.smoother_detail.has_value());
        TEST_ASSERT(!r.sigma_honored.empty());

        // Identical (to 1e-15) to calling estimate_ea_smoother directly.
        auto direct = estimate_ea_smoother(
            scenario, SensorMask::all_on(scenario), fm, {});
        TEST_ASSERT(std::abs(r.nis - direct.nis) < 1e-15);
        TEST_ASSERT(std::abs(r.smoother_detail->sigma_a_used -
                             direct.sigma_a_used) < 1e-15);
        std::cout << "  spec→smoother: NIS/dof = " << r.nis
                  << "  warnings = " << r.audit_warnings.size() << "\n";
        for (const auto& w : r.audit_warnings) std::cout << "    ! " << w << "\n";
    });

    runner.add_test("accel_samples_dispatch_with_audit", [&]() {
        auto truth_ea = glide.truth_traj->force_model().empirical_acceleration();
        AccelSamples acc;
        std::mt19937 rng(7);
        ballistic::test::Normal n(0.0, 0.01);
        for (double t = 0.0; t <= 60.0; t += 0.1) {
            auto st = glide.truth_interp->get_state(glide.t0 + t);
            if (!st) continue;
            // Truth EA plus noise, not rotated into the Earth-fixed frame:
            // this cell only needs a nonzero signal to check the routing, not
            // a physically exact accelerometer reading.
            Vector3d a = truth_ea->accel_at(t) +
                         Vector3d(n(rng), n(rng), n(rng));
            acc.ecef.emplace_back(glide.t0 + t, a);
        }
        EAProblemSpec spec;
        spec.epoch0 = glide.t0;
        spec.pos0_ecef = glide.pos;
        spec.vel0_ecef = glide.vel;
        spec.observations = acc;
        spec.force_model = fm;
        spec.noise_sigma = 0.01;
        auto r = estimate_ea(spec);
        TEST_ASSERT_MSG(r.error_message.empty(), r.error_message);
        TEST_ASSERT(r.method_used == EAMethod::BSplineAcceleration);
        TEST_ASSERT(r.bspline_detail.has_value());
        TEST_ASSERT(r.succeeded());
        std::cout << "  spec→bspline ACC: warnings = "
                  << r.audit_warnings.size() << "\n";
    });

    runner.add_test("custom_sensor_hook_matches_builtin_fix", [&]() {
        // A CustomSensor whose callback replicates the built-in position-fix
        // update must reproduce the PositionFixObs result exactly: the hook
        // runs through the same code path, not a parallel one.
        MultiSensorScenario base;
        base.epoch0 = glide.t0;
        base.pos0_ecef = glide.pos;
        base.vel0_ecef = glide.vel;
        base.arc_duration_s = glide.eval_end;
        base.add_radar(radar_helpers::build_radar_for_glide(
            glide, 20'000.0, 160.0, glide.eval_end, true, 42));

        PositionFixObs fix;
        fix.sigma_pos = 3.0;
        auto st = glide.truth_interp->get_state(glide.t0 + glide.eval_end - 0.5001);
        TEST_ASSERT(st.has_value());
        std::mt19937 rng(44);
        ballistic::test::Normal nn(0.0, fix.sigma_pos);
        Vector3d obs_pos = st->position + Vector3d(nn(rng), nn(rng), nn(rng));
        fix.samples.emplace_back(glide.t0 + glide.eval_end - 0.5001, obs_pos);

        MultiSensorScenario with_fix = base;
        with_fix.add_position_fix(fix);
        EASmootherConfig cfg;
        cfg.sigma_a = 3.0;   // pinned prior: single fit each side
        cfg.tau_s = 35.0;
        auto r_builtin = estimate_ea_smoother(
            with_fix, SensorMask::all_on(with_fix), fm, cfg);

        EASmootherConfig cfg2 = cfg;
        calibration::detail::CustomSensor cs;
        // Use the time after the Epoch round trip, as the built-in timeline
        // does: (epoch0 + t) - epoch0 is rounded to the last digit of the
        // Epoch's seconds field (far below a microsecond), and using the raw
        // double instead shifts the observation time enough to break the
        // bit-for-bit match gated below.
        cs.times_s = {fix.samples[0].first - base.epoch0};
        cs.update = [&fix](const calibration::detail::EpochContext& ctx,
                           const Eigen::VectorXd&,
                           const calibration::detail::StateLayout& L,
                           size_t i) {
            return calibration::detail::fix_update(ctx, fix, i, L);
        };
        cfg2.custom_sensors.push_back(std::move(cs));
        auto r_custom = estimate_ea_smoother(
            base, SensorMask::all_on(base), fm, cfg2);

        TEST_ASSERT(r_builtin.error_message.empty() &&
                    r_custom.error_message.empty());
        std::cout << "  custom-vs-builtin fix: NIS "
                  << std::scientific << std::setprecision(3) << r_custom.nis
                  << " vs " << r_builtin.nis << " (|Δ| = "
                  << std::abs(r_custom.nis - r_builtin.nis) << ")\n"
                  << std::defaultfloat;
        // With a timestamp off the sample grid (so its sort position is unique)
        // and the round-tripped time, the hook must match the built-in path
        // bit for bit.
        TEST_ASSERT_MSG(std::abs(r_custom.nis - r_builtin.nis) < 1e-12,
                        "custom hook diverged from built-in path");
    });

    runner.add_test("information_bound_service", [&]() {
        // The information bound: the best reprop accuracy the sensors allow
        // (posterior covariance and Cramér-Rao bound, CRB), evaluated along the
        // trajectory with zero EA because a user has no truth. Measured here:
        // single-radar posterior about 10-17 m, with a camera about 1.3 m.
        // Gated loosely, not at the exact values: radar-only bound in [3, 30] m,
        // and the camera must cut it by at least 2×. For context (not asserted):
        // the smoother achieves 12.7 m and 3.1 m on the same setups, and a bound
        // above the achieved accuracy would mean the bound is built wrong.
        MultiSensorScenario s1;
        s1.epoch0 = glide.t0;
        s1.pos0_ecef = glide.pos;
        s1.vel0_ecef = glide.vel;
        s1.arc_duration_s = glide.eval_end;
        s1.add_radar(radar_helpers::build_radar_for_glide(
            glide, 20'000.0, 160.0, glide.eval_end, true, 42));

        EAProblemSpec spec;
        spec.observations = s1;
        spec.force_model = fm;
        auto b1 = information_bound(spec);
        TEST_ASSERT_MSG(b1.ok(), b1.error_message);

        MultiSensorScenario s2 = s1;
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
        auto b2 = information_bound(s2, SensorMask::all_on(s2), fm);
        TEST_ASSERT_MSG(b2.ok(), b2.error_message);

        std::cout << "  bound radar+fix: posterior = " << b1.posterior_bound_m
                  << " m  CRB = " << b1.crb_bound_m << " m  (rows = "
                  << b1.n_obs_rows << ")\n"
                  << "  bound +camera:   posterior = " << b2.posterior_bound_m
                  << " m  CRB = " << b2.crb_bound_m << " m\n";
        TEST_ASSERT_MSG(b1.posterior_bound_m > 3.0 &&
                        b1.posterior_bound_m < 30.0,
                        "radar bound out of class: " +
                            std::to_string(b1.posterior_bound_m));
        TEST_ASSERT_MSG(b2.posterior_bound_m < 0.5 * b1.posterior_bound_m,
                        "camera does not tighten the bound 2x");
        TEST_ASSERT(b1.crb_bound_m > 0.0 && b2.crb_bound_m > 0.0);

        // Typed-sample specs are refused.
        EAProblemSpec bad;
        bad.observations = PositionSamples{};
        TEST_ASSERT(!information_bound(bad).ok());
    });

    runner.add_test("heavy_tail_prefilter_acc", [&]() {
        // Heavy-tail prefilter: 100 Hz acceleration samples, 5% of them outliers
        // at 10σ. Measured on this seed: plain least squares (L2) 2.5 m, with
        // outlier rejection 0.7 m (ratio 0.3). Gate: filtered < 0.6 × plain, and
        // the audit reports
        // how many samples were rejected.
        std::mt19937 rng(18);
        ballistic::test::Normal n(0.0, 0.01);
        ballistic::test::UniformReal u01(0.0, 1.0);
        AccelSamples acc;
        acc.ecef = glide.acc_100;
        for (auto& [t, v] : acc.ecef) {
            const double sc = (u01(rng) < 0.05) ? 10.0 : 1.0;
            v += sc * Vector3d(n(rng), n(rng), n(rng));
        }
        EAProblemSpec spec;
        spec.epoch0 = glide.t0;
        spec.pos0_ecef = glide.pos;
        spec.vel0_ecef = glide.vel;
        spec.observations = acc;
        spec.force_model = fm;
        spec.noise_sigma = 0.01;

        auto r_plain = estimate_ea(spec);
        spec.outlier_prob = 0.05;
        spec.outlier_scale = 10.0;
        auto r_filt = estimate_ea(spec);
        TEST_ASSERT(r_plain.succeeded() && r_filt.succeeded());
        double rp_plain =
            glide_helpers::reprop_rmse(glide, r_plain.empirical_accel);
        double rp_filt =
            glide_helpers::reprop_rmse(glide, r_filt.empirical_accel);
        bool reported = false;
        for (const auto& w : r_filt.audit_warnings)
            if (w.find("prefilter rejected") != std::string::npos)
                reported = true;
        std::cout << "  heavy-tail ACC: plain = " << rp_plain
                  << " m  filtered = " << rp_filt << " m  (ratio "
                  << rp_filt / rp_plain << ")\n";
        TEST_ASSERT_MSG(rp_filt < 0.6 * rp_plain,
                        "prefilter did not recover: " +
                            std::to_string(rp_filt) + " vs " +
                            std::to_string(rp_plain));
        TEST_ASSERT_MSG(reported, "rejected-count audit line missing");
    });

    runner.add_test("custom_sensor_bad_dimensions_fail_loudly", [&]() {
        // A callback returning inconsistent {r, H, R} sizes must throw
        // a descriptive invalid_argument (naming the sizes), not corrupt
        // memory via an unchecked Eigen product.

        MultiSensorScenario base;
        base.epoch0 = glide.t0;
        base.pos0_ecef = glide.pos;
        base.vel0_ecef = glide.vel;
        base.arc_duration_s = 20.0;
        base.add_radar(radar_helpers::build_radar_for_glide(
            glide, 20'000.0, 10.0, 20.0, true, 42));
        EASmootherConfig cfg;
        cfg.sigma_a = 3.0;
        cfg.tau_s = 35.0;
        calibration::detail::CustomSensor bad;
        bad.times_s = {5.0};
        bad.update = [](const calibration::detail::EpochContext&,
                        const Eigen::VectorXd&,
                        const calibration::detail::StateLayout&, size_t) {
            calibration::detail::SensorUpdate u;
            u.r = Eigen::VectorXd::Zero(3);
            u.H = Eigen::MatrixXd::Zero(3, 5);   // wrong: nx is 10
            u.R = Eigen::MatrixXd::Identity(3, 3);
            return u;
        };
        cfg.custom_sensors.push_back(std::move(bad));
        bool threw = false;
        try {
            (void)estimate_ea_smoother(base, SensorMask::all_on(base), fm, cfg);
        } catch (const std::invalid_argument& e) {
            threw = true;
            std::string msg = e.what();
            TEST_ASSERT_MSG(msg.find("inconsistent") != std::string::npos &&
                                msg.find("nx=") != std::string::npos,
                            "error message must name the sizes: " + msg);
            std::cout << "  bad-dimension callback: " << msg << "\n";
        }
        TEST_ASSERT_MSG(threw, "bad-dimension callback did not throw");
    });

    runner.add_test("missing_noise_declaration_is_an_error", [&]() {
        EAProblemSpec spec;
        spec.epoch0 = glide.t0;
        spec.pos0_ecef = glide.pos;
        spec.vel0_ecef = glide.vel;
        spec.observations = PositionSamples{};
        spec.force_model = fm;
        auto r = estimate_ea(spec);
        TEST_ASSERT(!r.error_message.empty());
        TEST_ASSERT(!r.succeeded());
    });

    runner.run();
    return print_final_summary();
}
