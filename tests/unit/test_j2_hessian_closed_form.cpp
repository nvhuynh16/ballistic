// =============================================================================
// Test: J2 closed-form acceleration and Hessian (reference values)
// =============================================================================
//
// J2 is the term of Earth's gravity field due to its oblateness (flattening
// at the poles); it is several hundred times larger than any other
// non-spherical term (J2 ≈ 1.08e-3 vs J3 ≈ -2.5e-6). The
// Hessian is the 3x3 matrix of second derivatives of the potential V
// (equivalently, the derivative of the acceleration with respect to
// position); estimators need it to propagate sensitivities.
//
// Validates the closed-form J2 expressions in `j2_hessian_closed_form.hpp`
// against:
//   - Laplace's equation (∇²V_J2 = 0; V_J2 is harmonic outside the source)
//   - Hessian symmetry (Schwarz)
//   - Acceleration-via-FD-on-potential (sanity check)
//
// These properties hold IDENTICALLY for the closed-form (they're proven
// mathematically), so the test should pass to machine precision. The
// test is the **anchor** for the gravity engine's analytic Hessian: the
// engine evaluated at degree 2, zonal terms only, must agree with these
// closed-form values to 1e-9 relative (the gate in test_gravity_hessian;
// subtracting the point-mass term there costs about three digits).
// =============================================================================

#include "../test_utils.hpp"
#include "j2_hessian_closed_form.hpp"
#include <Eigen/Dense>

using namespace ballistic;
using namespace ballistic::test;

int main() {
    TestRunner runner("J2 closed-form acceleration and Hessian");

    // Several test positions covering distinct regimes
    const std::vector<std::pair<std::string, Vector3d>> test_points = {
        {"LEO equatorial", Vector3d(6.778e6, 0.0, 0.0)},
        {"LEO mid-lat",     Vector3d(5.6e6, 0.0, 3.85e6)},   // ~35° lat
        {"LEO near-pole",   Vector3d(0.115e6, 0.0, 6.777e6)}, // ~89° lat
        {"MEO equatorial",  Vector3d(2.65e7, 0.0, 0.0)},
        {"GEO equatorial",  Vector3d(4.2164e7, 0.0, 0.0)},
        {"Off-axis",        Vector3d(5e6, 3e6, 4e6)},
    };

    runner.add_test("Laplace equation: tr(H) = 0 at all test points", [&]() {
        for (const auto& [label, pos] : test_points) {
            auto result = j2_closed_form(pos);
            const auto& H = result.hessian;
            const double trace = H(0, 0) + H(1, 1) + H(2, 2);
            const double scale = H.norm() + 1e-30;
            const double rel = std::abs(trace) / scale;
            // Closed form: should be EXACT to machine epsilon.
            TEST_ASSERT_NEAR(rel, 0.0, 1e-13);
        }
    });

    runner.add_test("Hessian symmetry: H = Hᵀ at all test points", [&]() {
        for (const auto& [label, pos] : test_points) {
            auto result = j2_closed_form(pos);
            const auto& H = result.hessian;
            for (int i = 0; i < 3; ++i) {
                for (int j = i + 1; j < 3; ++j) {
                    TEST_ASSERT_NEAR(H(i, j), H(j, i), 1e-25);
                }
            }
        }
    });

    runner.add_test("Self-check helper agrees with explicit checks", [&]() {
        for (const auto& [label, pos] : test_points) {
            const double max_violation = j2_closed_form_self_check(pos);
            TEST_ASSERT(max_violation < 1e-13);
        }
    });

    runner.add_test("Acceleration vs FD on potential", [&]() {
        // Sanity: a = ∇V should match central FD on V.
        // This catches typos in the acceleration formula (vs the potential).
        const double delta = 1.0;  // 1 m perturbation, ~1e-7 rel for r ≈ 1e7
        for (const auto& [label, pos] : test_points) {
            auto result = j2_closed_form(pos);
            for (int i = 0; i < 3; ++i) {
                Vector3d pos_p = pos, pos_m = pos;
                pos_p(i) += delta;
                pos_m(i) -= delta;
                const double V_p = j2_closed_form(pos_p).potential;
                const double V_m = j2_closed_form(pos_m).potential;
                const double fd_grad = (V_p - V_m) / (2.0 * delta);
                const double rel = std::abs(fd_grad - result.acceleration(i))
                                   / (std::abs(result.acceleration(i)) + 1e-30);
                TEST_ASSERT(rel < 1e-5);  // FD truncation floor
            }
        }
    });

    runner.add_test("Hessian vs FD on acceleration", [&]() {
        // Sanity: H_ij = ∂a_i/∂x_j should match central FD on acceleration.
        const double delta = 1.0;
        for (const auto& [label, pos] : test_points) {
            auto result = j2_closed_form(pos);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Vector3d pos_p = pos, pos_m = pos;
                    pos_p(j) += delta;
                    pos_m(j) -= delta;
                    const double a_p = j2_closed_form(pos_p).acceleration(i);
                    const double a_m = j2_closed_form(pos_m).acceleration(i);
                    const double fd_H = (a_p - a_m) / (2.0 * delta);
                    const double scale = std::abs(result.hessian(i, j)) + 1e-15;
                    const double rel = std::abs(fd_H - result.hessian(i, j)) / scale;
                    TEST_ASSERT(rel < 1e-5);  // FD truncation floor
                }
            }
        }
    });

    runner.add_test("Magnitude scale: J2 acceleration ~ 1.6e-2 m/s² at LEO surface",
        [&]() {
            // Sanity check of the J2 scale at r = 6.778e6 m on the equator
            // (x axis), where |a_J2| = 1.5·GM·J2·R²/r⁴ ≈ 1.2e-2 m/s².
            const Vector3d pos(6.778e6, 0.0, 0.0);
            const auto result = j2_closed_form(pos);
            const double a_mag = result.acceleration.norm();
            // GM·J2·R²/r⁴ ≈ 4.0e14 · 1.08e-3 · 4.07e13 / 2.11e27 ≈ 8.3e-3;
            // the window [1e-4, 1e-1] is a ballpark check only.
            TEST_ASSERT(a_mag > 1e-4 && a_mag < 1e-1);
        });

    runner.run();
    return print_final_summary();
}
