// Unit tests for refine_beta_from_ea, which turns an estimated empirical
// acceleration (EA: the extra acceleration fitted beyond the force model) into a
// new ballistic coefficient column (β = m/(Cd·A), kg/m²) of a DragEnvironment.
// With a zero EA the prior must come back unchanged, with a ModelSubstitution
// diagnostic saying so, and density and wind must always pass through untouched.

#include "../test_utils.hpp"
#include <ballistic/calibration/ea_decomposition.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/atmosphere/density_profile.hpp>
#include <ballistic/atmosphere/wind_profile.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/init.hpp>
#include <ballistic/core/diagnostics.hpp>
#include <string>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;
using namespace ballistic::calibration;

namespace {

Epoch t0_test() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

// Collects the diagnostics reported while it lives (an installed handler also
// owns strictness, so a strict run does not throw on the expected note).
struct CaptureDiagnostics {
    std::vector<diagnostics::Diagnostic> seen;
    CaptureDiagnostics() {
        diagnostics::set_handler(
            [this](const diagnostics::Diagnostic& d) { seen.push_back(d); });
    }
    ~CaptureDiagnostics() { diagnostics::set_handler(nullptr); }
};

}  // namespace

void test_density_and_wind_pass_through() {
    TestRunner runner("DensityAndWindPassThrough");

    runner.add_test("refine_with_zero_ea_preserves_prior", []() {
        ballistic::initialize({});

        // Prior with uniform density / wind / beta.
        std::vector<double> alt{200.0, 400.0, 600.0};
        std::vector<double> rho{1e-10, 1e-12, 1e-14};
        DragEnvironment prior(
            DensityProfile1D(alt, rho),
            WindProfile1D_ENU(alt, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}),
            BetaProfile1D::constant(50.0));

        // Zero EA -> no refinement; refine_beta_from_ea should return the
        // prior unchanged (the inner decompose_to_beta short-circuits on
        // is_zero()).
        auto ea_zero = forces::EmpiricalAcceleration::zero();
        std::vector<HistoryEntry> traj;  // empty traj

        CaptureDiagnostics cap;
        BetaResult report;
        DragEnvironment refined = refine_beta_from_ea(
            prior, ea_zero, traj, t0_test(), {}, &report);

        // Keeping the prior is reported, not silent, and the report says why.
        TEST_ASSERT(cap.seen.size() == 1);
        TEST_ASSERT(cap.seen[0].category ==
                    diagnostics::Category::ModelSubstitution);
        TEST_ASSERT(cap.seen[0].message.find("prior beta is returned unchanged") !=
                    std::string::npos);
        TEST_ASSERT(!report.error_message.empty());
        TEST_ASSERT(report.steps_solved == 0);

        // Beta should still be the prior's 50.0.
        constexpr double DEG_TO_RAD = 0.017453292519943295;
        Vector3d pos = coords::geodetic_to_ecef(
            0.0 * DEG_TO_RAD, 0.0 * DEG_TO_RAD, 400e3);
        const DragSample s = refined.evaluate(pos, t0_test());
        TEST_ASSERT_NEAR(s.beta_kg_m2, 50.0, 1e-6);

        // Density should match prior at the second altitude knot (1e-12).
        TEST_ASSERT(std::abs(s.rho_kg_m3 - 1e-12) / 1e-12 < 5e-3);
    });

    runner.run();
}

void test_refined_env_swaps_only_beta() {
    TestRunner runner("RefinedEnvSwapsOnlyBeta");

    runner.add_test("density_pointer_or_value_preserved_through_refinement", []() {
        ballistic::initialize({});

        std::vector<double> alt{200.0, 400.0, 600.0};
        std::vector<double> rho{1e-10, 1e-12, 1e-14};
        DragEnvironment prior(
            DensityProfile1D(alt, rho),
            WindProfile1D_ENU(alt, {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}),
            BetaProfile1D::constant(50.0));

        // Sample prior density+wind before refinement.
        constexpr double DEG_TO_RAD = 0.017453292519943295;
        Vector3d pos = coords::geodetic_to_ecef(
            0.0 * DEG_TO_RAD, 0.0 * DEG_TO_RAD, 400e3);
        const DragSample prior_s = prior.evaluate(pos, t0_test());

        auto ea_zero = forces::EmpiricalAcceleration::zero();
        std::vector<HistoryEntry> traj;
        CaptureDiagnostics cap;
        DragEnvironment refined = refine_beta_from_ea(
            prior, ea_zero, traj, t0_test());

        const DragSample refined_s = refined.evaluate(pos, t0_test());

        // Density and wind unchanged.
        TEST_ASSERT_NEAR(refined_s.rho_kg_m3, prior_s.rho_kg_m3, 1e-20);
        TEST_ASSERT_NEAR((refined_s.wind_ecef_m_s - prior_s.wind_ecef_m_s).norm(), 0.0, 1e-9);
        // Beta unchanged because EA was zero (early return).
        TEST_ASSERT_NEAR(refined_s.beta_kg_m2, prior_s.beta_kg_m2, 1e-6);
    });

    runner.run();
}

void test_fit_or_report() {
    TestRunner runner("FitOrReport");

    // Three points at 400 km on the equator, moving east at 7.2 km/s (ECEF),
    // with a 1D density table; the EA is given in ECEF.
    std::vector<double> alt{200.0, 400.0, 600.0};
    std::vector<double> rho{1e-10, 1e-12, 1e-14};
    auto make_prior = [&]() {
        return DragEnvironment(
            DensityProfile1D(alt, rho),
            WindProfile1D_ENU(alt, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}),
            BetaProfile1D::constant(50.0));
    };
    auto make_traj = []() {
        std::vector<HistoryEntry> traj;
        for (int k = 0; k < 3; ++k) {
            HistoryEntry h{};
            h.epoch = t0_test() + 10.0 * k;
            h.position = coords::geodetic_to_ecef(0.0, 0.0, 400e3);
            h.velocity = Vector3d(0.0, 7200.0, 0.0);
            traj.push_back(h);
        }
        return traj;
    };
    auto ea_along = [](double ay) {
        std::vector<forces::EmpiricalAccelEntry> e;
        for (int k = 0; k <= 3; ++k) e.emplace_back(10.0 * k, Vector3d(0.0, ay, 0.0));
        return forces::EmpiricalAcceleration::from_entries(
            std::move(e), forces::EmpiricalAccelFrame::ITRF);
    };

    runner.add_test("ea_along_the_motion_is_reported", [&]() {
        // A non-zero EA that pushes WITH the motion has no drag part: the
        // prior is kept and the diagnostic carries the reason.
        ballistic::initialize({});
        auto prior = make_prior();
        auto traj = make_traj();
        CaptureDiagnostics cap;
        BetaResult report;
        auto refined = refine_beta_from_ea(prior, ea_along(+1e-5), traj,
                                           t0_test(), {}, &report);
        TEST_ASSERT(cap.seen.size() == 1);
        TEST_ASSERT(cap.seen[0].message.find("3 trajectory points") !=
                    std::string::npos);
        TEST_ASSERT(report.error_message == "No valid beta estimates");
        const Vector3d pos = coords::geodetic_to_ecef(0.0, 0.0, 400e3);
        TEST_ASSERT_NEAR(refined.evaluate(pos, t0_test()).beta_kg_m2, 50.0, 1e-9);
    });

    runner.add_test("drag_like_ea_is_fitted_silently", [&]() {
        // EA = drag at β 100: fitted, no diagnostic, report filled.
        ballistic::initialize({});
        auto prior = make_prior();
        auto traj = make_traj();
        const double a = 1e-12 * 7200.0 * 7200.0 / (2.0 * 100.0);
        CaptureDiagnostics cap;
        BetaResult report;
        auto refined = refine_beta_from_ea(prior, ea_along(-a), traj,
                                           t0_test(), {}, &report);
        TEST_ASSERT(cap.seen.empty());
        TEST_ASSERT(report.error_message.empty());
        TEST_ASSERT(report.steps_solved == 3);
        TEST_ASSERT(std::abs(report.constant_beta - 100.0) < 1.0);
        const Vector3d pos = coords::geodetic_to_ecef(0.0, 0.0, 400e3);
        TEST_ASSERT(std::abs(refined.evaluate(pos, t0_test()).beta_kg_m2 - 100.0) < 1.0);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "refine_beta_from_ea Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_density_and_wind_pass_through();
    test_refined_env_swaps_only_beta();
    test_fit_or_report();

    return print_final_summary();
}
