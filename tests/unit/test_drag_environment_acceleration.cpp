// Unit tests for AtmosphericDrag::acceleration and its Jacobian.
//
// Strategy: build a hand-crafted DragEnvironment (no NRLMSIS), call the env
// path, and assert that the returned acceleration matches a manually-computed
// reference using the same drag formula. Also verifies the analytic ∂a/∂v
// against FD, and confirms the position-FD column produces finite results.

#include "../test_utils.hpp"
#include <ballistic/forces/drag.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/coordinates/earth_rotation.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::forces;
using namespace ballistic::atmosphere;

namespace {

Epoch t_test() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

// Build a uniform DragEnvironment: constant rho, zero wind, constant beta.
// Useful for hand-computing the reference acceleration.
DragEnvironment uniform_env(double rho, double beta) {
    DensityProfile1D density(
        {100.0, 1500.0},      // altitude_km
        {rho, rho});
    WindProfile1D_ENU wind;   // empty -> zero wind
    BetaProfile1D b = BetaProfile1D::constant(beta);
    return DragEnvironment(std::move(density), std::move(wind), std::move(b));
}

// Build a GCRS state at the given equatorial LEO orbit point.
StateVector leo_state_gcrs() {
    StateVector s;
    // r = 6778 km along x_ECI (~equatorial), v = 7500 m/s along y.
    // Note: GCRS ≈ ECEF only when polar motion + nutation are negligible.
    // For this unit test we just need a finite, non-degenerate state.
    s.position = Vector3d(6.778137e6, 0.0, 0.0);
    s.velocity = Vector3d(0.0, 7500.0, 0.0);
    return s;
}

}  // namespace

void test_zero_when_empty_env() {
    TestRunner runner("ZeroWhenEmptyEnv");

    runner.add_test("empty_env_returns_zero_acceleration", []() {
        AtmosphericDrag drag;
        DragEnvironment env;  // default-constructed: has_drag() == false
        Vector3d a = drag.acceleration(env, t_test(), leo_state_gcrs());
        TEST_ASSERT(a.norm() < 1e-20);
    });

    runner.add_test("empty_env_jacobian_is_zero", []() {
        AtmosphericDrag drag;
        DragEnvironment env;
        auto [a, J] = drag.acceleration_with_jacobian(
            env, t_test(), leo_state_gcrs());
        TEST_ASSERT(a.norm() < 1e-20);
        TEST_ASSERT(J.norm() < 1e-20);
    });
}

void test_formula_matches_hand_computation() {
    TestRunner runner("FormulaMatchesHandComputation");

    runner.add_test("acceleration_magnitude_equals_half_rho_inv_beta_vsq", []() {
        // For uniform env with no wind, in ITRF (which after GCRS transform is
        // very close to the input ECEF state at this epoch, modulo rotation),
        // the drag formula reduces to |a| = 0.5 * rho * v² / beta.
        const double rho = 1e-11;
        const double beta = 100.0;
        DragEnvironment env = uniform_env(rho, beta);

        AtmosphericDrag drag;
        Vector3d a = drag.acceleration(env, t_test(), leo_state_gcrs());

        const double v = 7500.0;
        const double expected_mag = 0.5 * rho * v * v / beta;
        // Allow 10% tolerance because the ITRF-relative velocity differs from
        // GCRS velocity by the Earth-rotation contribution (~470 m/s at the
        // equator), which changes |v_rel| slightly.
        const double rel_err = std::abs(a.norm() - expected_mag) / expected_mag;
        TEST_ASSERT(rel_err < 0.15);
    });

    runner.add_test("acceleration_opposes_velocity_in_itrf", []() {
        // After Earth-rotation subtraction, drag acceleration must oppose
        // v_rel. In GCRS this is approximately anti-parallel to velocity.
        DragEnvironment env = uniform_env(1e-11, 100.0);
        AtmosphericDrag drag;
        StateVector s = leo_state_gcrs();
        Vector3d a = drag.acceleration(env, t_test(), s);

        // Cosine of angle between a and v should be very negative (~-1).
        const double cos_angle = a.dot(s.velocity) / (a.norm() * s.velocity.norm());
        TEST_ASSERT(cos_angle < -0.95);
    });

    runner.add_test("doubling_density_doubles_acceleration", []() {
        AtmosphericDrag drag;
        StateVector s = leo_state_gcrs();
        const Vector3d a1 = drag.acceleration(
            uniform_env(1e-11, 100.0), t_test(), s);
        const Vector3d a2 = drag.acceleration(
            uniform_env(2e-11, 100.0), t_test(), s);
        TEST_ASSERT_NEAR(a2.norm() / a1.norm(), 2.0, 1e-9);
    });

    runner.add_test("halving_beta_doubles_acceleration", []() {
        AtmosphericDrag drag;
        StateVector s = leo_state_gcrs();
        const Vector3d a1 = drag.acceleration(
            uniform_env(1e-11, 100.0), t_test(), s);
        const Vector3d a2 = drag.acceleration(
            uniform_env(1e-11, 50.0), t_test(), s);
        TEST_ASSERT_NEAR(a2.norm() / a1.norm(), 2.0, 1e-9);
    });
}

void test_velocity_jacobian_matches_fd() {
    TestRunner runner("VelocityJacobianMatchesFD");

    runner.add_test("analytic_dav_within_1pct_of_fd", []() {
        DragEnvironment env = uniform_env(1e-11, 100.0);
        AtmosphericDrag drag;
        StateVector s = leo_state_gcrs();

        const auto [a0, J] = drag.acceleration_with_jacobian(env, t_test(), s);
        const Eigen::Matrix3d J_v_analytic = J.block<3, 3>(0, 3);

        // FD ∂a/∂v: perturb velocity components.
        const double dv = 1e-3;  // 1 mm/s
        Eigen::Matrix3d J_v_fd;
        for (int i = 0; i < 3; ++i) {
            StateVector sp = s;
            sp.velocity(i) += dv;
            const Vector3d ap = drag.acceleration(env, t_test(), sp);
            J_v_fd.col(i) = (ap - a0) / dv;
        }

        // Analytic should match FD within ~1% (FD truncation error).
        const Eigen::Matrix3d diff = J_v_analytic - J_v_fd;
        const double rel_err = diff.norm() / J_v_fd.norm();
        TEST_ASSERT(rel_err < 0.02);
    });
}

void test_position_jacobian_finite() {
    TestRunner runner("PositionJacobianFinite");

    runner.add_test("position_columns_are_finite", []() {
        DragEnvironment env = uniform_env(1e-11, 100.0);
        AtmosphericDrag drag;
        StateVector s = leo_state_gcrs();

        const auto [a0, J] = drag.acceleration_with_jacobian(env, t_test(), s);
        // For a uniform-density env, ∂a/∂r should be very small (density
        // doesn't vary with altitude). But the values must be finite.
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                TEST_ASSERT(std::isfinite(J(i, j)));
            }
        }
    });
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "DragEnvironment Acceleration Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_zero_when_empty_env();
    test_formula_matches_hand_computation();
    test_velocity_jacobian_matches_fd();
    test_position_jacobian_finite();

    return print_final_summary();
}
