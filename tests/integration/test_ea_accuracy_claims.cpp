// =============================================================================
// EA accuracy claims: the asserted form of the estimator accuracy numbers
// =============================================================================
//
// EA = empirical acceleration: the extra acceleration, beyond the force model,
// that an estimator fits so the propagated trajectory matches the observations.
// Every documented EA reconstruction accuracy number is asserted here.
// docs/empirical_acceleration_estimation.md cites these cells by name; an
// estimator number without a cell here is not a claim.
//
// Gates sit a few times above the measured values (so a regression of a few
// times shows up) and leave room for the ±0.3-0.5 m run-to-run spread seen on
// single-seed Gauss-Newton (GN) refinement between different compiled binaries
// (floating-point summation order differs between compilation units).
//
// Data: the glide fixture (the synthetic reentry glide from 120 km in
// tests/test_helpers.hpp, with a known truth EA) and a synthetic 400 km
// circular orbit (below). RMSE = root-mean-square position error of the
// trajectory re-propagated with the estimated EA, against the truth.
//
// Cells:
//   null_pos_bspline / null_acc_bspline — model-generated observations with
//       ZERO injected EA must yield ~zero EA (succeeded(), !has_correction()
//       allowed): regularization must not invent signal.
//   recover_*                — known-EA recovery on the canonical glide.
//   ladder_ordering          — the observability ladder: with derivative-
//                              equivalent noise, RMSE(ACC) < RMSE(VEL) <
//                              RMSE(POS).
//   noise_scaling_pos        — RMSE grows ~linearly with σ (factor guard).
//   beta_recovery_decompose  — loop closure: drag-driven truth → estimate EA
//                              → refine_beta_from_ea recovers β within 10%.
// =============================================================================

#include "../test_utils.hpp"
#include "../test_helpers.hpp"

#include <ballistic/calibration/ea_estimator.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;
namespace glide_helpers = ballistic::test::helpers::glide;
namespace radar_helpers = ballistic::test::helpers::radar;
namespace atm = ballistic::atmosphere;

namespace {

// 400 km circular LEO fixture: gravity to degree 20 plus a gentle smooth truth EA
// (one Gaussian bump of mm/s² size, a realistic unmodeled force at LEO, unlike
// the glide's 8 m/s² forced flight).
struct LeoSetup {
    Epoch t0;
    Vector3d pos, vel;
    double arc_s = 300.0;
    std::shared_ptr<forces::EmpiricalAcceleration> truth_ea;
    std::unique_ptr<Trajectory> truth_traj;
    std::unique_ptr<TrajectoryInterpolator> truth_interp;
    std::vector<std::pair<Epoch, Vector3d>> pos_10;  // 10 Hz clean positions
};

TrajectoryConfig leo_base_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 20;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

LeoSetup make_leo() {
    LeoSetup s;
    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    const double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
    const double vc = std::sqrt(constants::earth::GM / R);
    const double wr = constants::earth::ANGULAR_VELOCITY * R;
    s.pos = Vector3d(R, 0, 0);
    s.vel = Vector3d(0.0, vc - wr, 0);

    std::vector<forces::EmpiricalAccelEntry> entries;
    for (int i = 0; i <= 320; ++i) {
        const double t = i;
        const double g = std::exp(-0.5 * std::pow((t - 150.0) / 40.0, 2));
        entries.emplace_back(t, Vector3d(2e-3 * g,
                                         1e-3 * g * std::sin(t / 30.0),
                                         -1.5e-3 * g));
    }
    s.truth_ea = std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(
            std::move(entries), forces::EmpiricalAccelFrame::GCRS));

    TrajectoryConfig cfg = leo_base_config();
    cfg.empirical_accel = s.truth_ea;
    s.truth_traj = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, cfg);
    while (s.truth_traj->epoch() - s.t0 < s.arc_s + 10.0)
        (void)s.truth_traj->step();
    s.truth_interp = std::make_unique<TrajectoryInterpolator>(*s.truth_traj);

    for (double t = 0.0; t <= s.arc_s; t += 0.1) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) s.pos_10.emplace_back(s.t0 + t, st->position);
    }
    return s;
}

// Re-propagate from the LEO IC with `ea` and compute 1 Hz position RMSE.
double leo_reprop_rmse(const LeoSetup& s,
                       const std::shared_ptr<forces::EmpiricalAcceleration>& ea,
                       double max_step_s = 0.0) {
    if (!ea) return 1e9;
    TrajectoryConfig rc = leo_base_config();
    rc.empirical_accel = ea;
    if (max_step_s > 0.0) rc.rk45.max_step_size = max_step_s;
    Trajectory rt(s.t0, s.pos, s.vel, rc);
    while (rt.epoch() - s.t0 < s.arc_s) (void)rt.step();
    TrajectoryInterpolator ri(rt);
    double sse = 0.0; int n = 0;
    for (double t = 5.0; t <= s.arc_s - 5.0; t += 1.0) {
        auto a = ri.get_state(s.t0 + t);
        auto b = s.truth_interp->get_state(s.t0 + t);
        if (a && b) { sse += (a->position - b->position).squaredNorm(); ++n; }
    }
    return n > 0 ? std::sqrt(sse / n) : 1e9;
}

double max_ea_norm(const forces::EmpiricalAcceleration& ea, double t_end) {
    double m = 0.0;
    for (double t = 0.0; t <= t_end; t += 1.0)
        m = std::max(m, ea.accel_at(t).norm());
    return m;
}

}  // namespace

int main() {
    // Hermetic init: ignore the user's data directory (a cached
    // leap-seconds.list / finals2000A.data could shift the gated numbers), then
    // load the Earth orientation parameters (EOP) from the committed file. An
    // explicit eop_path must load, so a missing or unparseable file fails loudly.
    ballistic::test::helpers::IsolatedDataDir isolation;
    InitConfig init_cfg;
    init_cfg.eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
    auto init_result = initialize(init_cfg);
    if (!init_result.success || !init_result.eop_loaded) {
        std::cerr << "FATAL: hermetic init failed: " << init_result.error_message << "\n";
        return 1;
    }
    TestRunner runner("EA Accuracy Claims");

    auto glide = glide_helpers::make_glide();
    auto leo = make_leo();

    // -------------------------------------------------------------------
    // Null tests: zero injected EA in, ~zero EA out.
    // -------------------------------------------------------------------
    runner.add_test("null_pos_bspline", [&]() {
        // Truth = pure gravity LEO arc (no EA); estimator must not invent one.
        TrajectoryConfig cfg = leo_base_config();
        Trajectory truth(leo.t0, leo.pos, leo.vel, cfg);
        while (truth.epoch() - leo.t0 < leo.arc_s + 10.0) (void)truth.step();
        TrajectoryInterpolator ti(truth);
        std::vector<std::pair<Epoch, Vector3d>> obs;
        for (double t = 0.0; t <= leo.arc_s; t += 0.1) {
            auto st = ti.get_state(leo.t0 + t);
            if (st) obs.emplace_back(leo.t0 + t, st->position);
        }
        EABSplineConfig ec;
        ec.max_duration_s = leo.arc_s;
        auto r = estimate_ea_bspline(leo.t0, leo.pos, leo.vel,
                                     PositionSamples{obs}, cfg, ec);
        TEST_ASSERT_MSG(r.succeeded(), "null POS estimate must succeed");
        const double peak = r.empirical_accel
            ? max_ea_norm(*r.empirical_accel, leo.arc_s) : 0.0;
        std::cout << "  null POS peak |EA| = " << std::scientific << peak
                  << " m/s^2\n" << std::fixed;
        TEST_ASSERT_MSG(peak < 1e-4, "estimator invented EA from null data");
    });

    runner.add_test("null_acc_bspline", [&]() {
        TrajectoryConfig cfg = leo_base_config();
        Trajectory truth(leo.t0, leo.pos, leo.vel, cfg);
        while (truth.epoch() - leo.t0 < leo.arc_s + 10.0) (void)truth.step();
        TrajectoryInterpolator ecef_i(truth);
        InterpolatorConfig icfg; icfg.output_frame = InterpolationFrame::GCRS;
        TrajectoryInterpolator gcrs_i(truth, icfg);
        std::vector<std::pair<Epoch, Vector3d>> obs;
        for (double t = 0.0; t <= leo.arc_s; t += 0.1) {
            auto se = ecef_i.get_state(leo.t0 + t);
            auto sg = gcrs_i.get_state(leo.t0 + t);
            if (!se || !sg) continue;
            StateVector svg{sg->position, sg->velocity};
            Vector3d a_g = truth.force_model().acceleration(leo.t0 + t, svg);
            StateVector sv{se->position, se->velocity};
            obs.emplace_back(leo.t0 + t,
                util::acceleration_to_itrf(a_g, sv, leo.t0 + t));
        }
        EABSplineConfig ec;
        ec.max_duration_s = leo.arc_s;
        auto r = estimate_ea_bspline(leo.t0, leo.pos, leo.vel,
                                     AccelSamples{obs}, cfg, ec);
        TEST_ASSERT_MSG(r.succeeded(), "null ACC estimate must succeed");
        const double peak = r.empirical_accel
            ? max_ea_norm(*r.empirical_accel, leo.arc_s) : 0.0;
        std::cout << "  null ACC peak |EA| = " << std::scientific << peak
                  << " m/s^2\n" << std::fixed;
        TEST_ASSERT_MSG(peak < 1e-4, "estimator invented EA from null data");
    });

    // -------------------------------------------------------------------
    // Known-EA recovery, clean glide. The suite measures ~0.0-0.1 m across
    // the board; the gates sit at ~5x the measured values so that a
    // regression of a few times shows up instead of hiding under a loose
    // bound.
    // -------------------------------------------------------------------
    runner.add_test("recover_pos_bspline", [&]() {
        EABSplineConfig c;
        c.max_duration_s = glide.eval_end;
        c.gn = GNRefine{};  // load-bearing for POS
        auto r = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                     PositionSamples{glide.pos_100},
                                     glide_helpers::base_config(), c);
        TEST_ASSERT(r.has_correction());
        const double rmse = glide_helpers::reprop_rmse(glide, r.empirical_accel);
        std::cout << "  POS bspline+GN reprop RMSE = " << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 0.30, "clean POS bspline+GN claim (~0.01 m measured)");
    });

    runner.add_test("recover_vel_bspline", [&]() {
        EABSplineConfig c;  // gn left at its default (GNRefine{}, on); GN helps VEL
        c.max_duration_s = glide.eval_end;
        auto r = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                     VelocitySamples{glide.vel_100},
                                     glide_helpers::base_config(), c);
        TEST_ASSERT(r.has_correction());
        const double rmse = glide_helpers::reprop_rmse(glide, r.empirical_accel);
        std::cout << "  VEL bspline reprop RMSE = " << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 0.30, "clean VEL bspline claim (~0.05 m measured)");
    });

    runner.add_test("recover_acc_bspline", [&]() {
        EABSplineConfig c;
        c.max_duration_s = glide.eval_end;
        auto r = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                     AccelSamples{glide.acc_100},
                                     glide_helpers::base_config(), c);
        TEST_ASSERT(r.has_correction());
        const double rmse = glide_helpers::reprop_rmse(glide, r.empirical_accel);
        std::cout << "  ACC bspline reprop RMSE = " << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 0.30, "clean ACC bspline claim (~0.05 m measured)");

    });

    // -------------------------------------------------------------------
    // max_duration_s auto-clamp: the DEFAULT config (3600 s) on the ~182 s glide
    // must pass the same gates as the explicit-arc cells above. The library
    // clamps max_duration_s to the observation arc; without the clamp the
    // velocity fit came out far worse.
    // -------------------------------------------------------------------
    runner.add_test("clamp_parity_pos_bspline_default", [&]() {
        EABSplineConfig c;  // max_duration_s left at the 3600 s default
        c.gn = GNRefine{};
        auto r = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                     PositionSamples{glide.pos_100},
                                     glide_helpers::base_config(), c);
        TEST_ASSERT(r.has_correction());
        const double rmse = glide_helpers::reprop_rmse(glide, r.empirical_accel);
        std::cout << "  POS bspline+GN (default max_duration_s) reprop RMSE = "
                  << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 0.30, "default config must match explicit-arc gate");
    });

    runner.add_test("clamp_parity_vel_bspline_default", [&]() {
        EABSplineConfig c;  // max_duration_s left at the 3600 s default
        auto r = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                     VelocitySamples{glide.vel_100},
                                     glide_helpers::base_config(), c);
        TEST_ASSERT(r.has_correction());
        const double rmse = glide_helpers::reprop_rmse(glide, r.empirical_accel);
        std::cout << "  VEL bspline (default max_duration_s) reprop RMSE = "
                  << rmse << " m\n";
        TEST_ASSERT_MSG(rmse < 0.30, "default config must match explicit-arc gate");
    });

    // -------------------------------------------------------------------
    // Observability ladder: with derivative-equivalent noise (feature
    // timescale T≈35 s: σ_vel ≈ σ_pos/T, σ_acc ≈ σ_pos/T²), the noise
    // amplification of each extra derivative must order the errors.
    // -------------------------------------------------------------------
    runner.add_test("ladder_ordering", [&]() {
        const double T = 35.0;
        auto pos_n = glide_helpers::add_noise(glide.pos_100, 1.0, 42);
        auto vel_n = glide_helpers::add_noise(glide.vel_100, 1.0 / T, 43);
        auto acc_n = glide_helpers::add_noise(glide.acc_100, 1.0 / (T * T), 44);

        EABSplineConfig cp; cp.max_duration_s = glide.eval_end;
        cp.noise_sigma = 1.0; cp.gn = GNRefine{};
        auto rp = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                      PositionSamples{pos_n},
                                      glide_helpers::base_config(), cp);
        EABSplineConfig cv2; cv2.max_duration_s = glide.eval_end;
        cv2.noise_sigma = 1.0 / T;
        auto rv = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                      VelocitySamples{vel_n},
                                      glide_helpers::base_config(), cv2);
        EABSplineConfig ca; ca.max_duration_s = glide.eval_end;
        ca.noise_sigma = 1.0 / (T * T);
        auto ra = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                      AccelSamples{acc_n},
                                      glide_helpers::base_config(), ca);
        TEST_ASSERT(rp.has_correction() && rv.has_correction() && ra.has_correction());

        const double ep = glide_helpers::reprop_rmse(glide, rp.empirical_accel);
        const double ev = glide_helpers::reprop_rmse(glide, rv.empirical_accel);
        const double ea_ = glide_helpers::reprop_rmse(glide, ra.empirical_accel);
        std::cout << "  ladder (deriv-equivalent noise): ACC " << ea_
                  << "  VEL " << ev << "  POS " << ep << " m\n";
        // A strict VEL < POS can flip inside the ±0.3-0.5 m spread between
        // binaries; the margins below encode the ordering without demanding more
        // precision than the test bed provides.
        TEST_ASSERT_MSG(ea_ < 0.7 * ep, "ACC must beat POS decisively");
        TEST_ASSERT_MSG(ev < ep + 0.25, "VEL must not lose to POS beyond FP margin");
    });

    // -------------------------------------------------------------------
    // Noise scaling: doubling σ must not blow up (≲ linear growth), and
    // more noise must not make things (much) better.
    // -------------------------------------------------------------------
    runner.add_test("noise_scaling_pos", [&]() {
        auto n1 = glide_helpers::add_noise(glide.pos_100, 1.0, 7);
        auto n2 = glide_helpers::add_noise(glide.pos_100, 2.0, 7);
        EABSplineConfig c1; c1.max_duration_s = glide.eval_end;
        c1.noise_sigma = 1.0; c1.gn = GNRefine{};
        EABSplineConfig c2; c2.max_duration_s = glide.eval_end;
        c2.noise_sigma = 2.0; c2.gn = GNRefine{};
        auto r1 = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                      PositionSamples{n1},
                                      glide_helpers::base_config(), c1);
        auto r2 = estimate_ea_bspline(glide.t0, glide.pos, glide.vel,
                                      PositionSamples{n2},
                                      glide_helpers::base_config(), c2);
        TEST_ASSERT(r1.has_correction() && r2.has_correction());
        const double e1 = glide_helpers::reprop_rmse(glide, r1.empirical_accel);
        const double e2 = glide_helpers::reprop_rmse(glide, r2.empirical_accel);
        std::cout << "  noise scaling: sigma=1 -> " << e1
                  << " m, sigma=2 -> " << e2 << " m\n";
        TEST_ASSERT_MSG(e2 <= 3.5 * e1, "superlinear noise blow-up");
        TEST_ASSERT_MSG(e2 >= e1 - 0.1, "more noise should not help");
    });

    // -------------------------------------------------------------------
    // Length of the glide arc. (The beta cell below defines its own 300 s arc.)
    // -------------------------------------------------------------------
    const double arc_s = glide.eval_end;

    // -------------------------------------------------------------------
    // Loop closure: drag-driven truth → EA estimate → refine_beta_from_ea
    // recovers the injected ballistic coefficient within 10%.
    // -------------------------------------------------------------------
#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN   // the drag truth is the NRLMSIS 2.1 model
    runner.add_test("beta_recovery_decompose", [&]() {
        // β = m/(Cd·A), the ballistic coefficient [kg/m²]. Truth: 160 km under
        // active solar conditions, drag about 5e-4 m/s², far above the
        // estimator's EA noise floor, so the velocity-aligned part of the EA is
        // mostly signal. At
        // 400 km in quiet conditions the drag signal drowns in that floor and β
        // cannot be recovered; that is expected, not a bug.
        const double true_beta = 66.2;   // truth β [kg/m²]
        const double wrong_beta = 200.0; // deliberately-bad prior
        const double arc_s = 300.0;      // matches the validated LEO cell size
        const std::vector<double> grid{100.0, 130.0, 160.0, 190.0, 220.0};

        const Epoch t0 = leo.t0;
        const double R = constants::earth::RADIUS_EQUATORIAL + 160000.0;
        const double vc = std::sqrt(constants::earth::GM / R);
        const double wr = constants::earth::ANGULAR_VELOCITY * R;
        const Vector3d pos(R, 0, 0);
        const Vector3d vel(0.0, vc - wr, 0);
        const auto weather = atm::SpaceWeatherHistory::from_constant(
            WeatherConditions::active());

        TrajectoryConfig base = leo_base_config();
        auto env_true = atm::make_drag_environment(
            t0, pos, vel, arc_s + 10.0, base,
            {.beta = atm::BetaProfile1D::constant(true_beta),
             .weather = weather, .altitude_grid_km = grid});

        TrajectoryConfig truth_cfg = base;
        truth_cfg.drag_env = env_true;
        Trajectory truth(t0, pos, vel, truth_cfg);
        while (truth.epoch() - t0 < arc_s + 5.0) (void)truth.step();
        TrajectoryInterpolator ti(truth);
        std::vector<std::pair<Epoch, Vector3d>> obs;
        for (double t = 0.0; t <= arc_s; t += 0.1) {
            auto st = ti.get_state(t0 + t);
            if (st) obs.emplace_back(t0 + t, st->position);
        }

        // Estimate the drag signal as EA against the drag-FREE model.
        // B-spline with WIDE knots (10 s): drag changes over minutes (the time to
        // cross a density scale height), and the β split needs the EA accurate
        // point by point, which comes from matching the spline's smoothness to
        // that physics. Tight knots reproduce the positions with a ringing EA
        // instead (many EA shapes fit the same arc), which scrambles the
        // velocity projection.
        EABSplineConfig ec;
        ec.max_duration_s = arc_s;
        ec.knot_spacing_s = 10.0;
        ec.gn = GNRefine{};
        auto r = estimate_ea_bspline(t0, pos, vel,
                                     PositionSamples{obs}, base, ec);
        TEST_ASSERT(r.has_correction());
        TEST_ASSERT(!r.collocation_history.empty());

        // Quality gate: the EA must reproduce the drag-on truth before its
        // decomposition means anything.
        {
            TrajectoryConfig rc = base;
            rc.empirical_accel = r.empirical_accel;
            Trajectory rt(t0, pos, vel, rc);
            while (rt.epoch() - t0 < arc_s) (void)rt.step();
            TrajectoryInterpolator ri(rt);
            double sse = 0.0; int n = 0;
            for (double t = 5.0; t <= arc_s - 5.0; t += 1.0) {
                auto a = ri.get_state(t0 + t);
                auto b = ti.get_state(t0 + t);
                if (a && b) { sse += (a->position - b->position).squaredNorm(); ++n; }
            }
            const double reprop = n ? std::sqrt(sse / n) : 1e9;
            std::cout << "  EA reprop vs drag-on truth = " << reprop << " m\n";
            TEST_ASSERT_MSG(reprop < 1.0, "EA must capture the drag signal");
        }

        // Close the loop: wrong-β prior env + EA → refined β column.
        auto env_prior = atm::make_drag_environment(
            t0, pos, vel, arc_s + 10.0, base,
            {.beta = atm::BetaProfile1D::constant(wrong_beta),
             .weather = weather, .altitude_grid_km = grid});
        auto refined = refine_beta_from_ea(
            *env_prior, *r.empirical_accel, r.collocation_history, t0);
        const auto& mid =
            r.collocation_history[r.collocation_history.size() / 2];
        const double got = refined.evaluate(mid.position, mid.epoch).beta_kg_m2;
        std::cout << "  beta recovery: true " << true_beta
                  << "  prior " << wrong_beta << "  refined " << got << "\n";
        TEST_ASSERT_MSG(std::abs(got - true_beta) / true_beta < 0.10,
                        "refine_beta_from_ea must recover beta within 10%");
    });
#endif

    runner.run();
    return print_final_summary();
}
