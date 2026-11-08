// =============================================================================
// Unit Tests for Quintic Hermite Interpolation
// =============================================================================
//
// Tests math/quintic_hermite.hpp: the degree-5 polynomial on one interval that
// matches position, velocity and acceleration at both ends (used to interpolate
// a trajectory between integrator steps). The variable s in [0, 1] is time
// scaled by the interval length dt. Checks: the basis functions' endpoint
// values and derivatives (exact), endpoint matching, analytic vs
// finite-difference derivatives, exact reproduction of straight-line and
// constant-acceleration motion, and the precomputed-coefficient form.
//
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/math/quintic_hermite.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::math;

int main() {
    // Initialize the library
    ballistic::initialize({});

    TestRunner runner("Quintic Hermite Tests");

    // =========================================================================
    // Basis Function Endpoint Tests
    // =========================================================================

    runner.add_test("basis_at_s_0", []() {
        // At s=0, only h00, h10, h20 should be active
        auto basis = quintic_hermite_basis(0.0);

        TEST_ASSERT_NEAR(basis.h00, 1.0, 1e-15);   // p0 coefficient = 1
        TEST_ASSERT_NEAR(basis.h10, 0.0, 1e-15);   // v0 coefficient = 0 (but derivative = 1)
        TEST_ASSERT_NEAR(basis.h20, 0.0, 1e-15);   // a0 coefficient = 0 (but 2nd deriv = 1)
        TEST_ASSERT_NEAR(basis.h01, 0.0, 1e-15);   // p1 coefficient = 0
        TEST_ASSERT_NEAR(basis.h11, 0.0, 1e-15);   // v1 coefficient = 0
        TEST_ASSERT_NEAR(basis.h21, 0.0, 1e-15);   // a1 coefficient = 0
    });

    runner.add_test("basis_at_s_1", []() {
        // At s=1, only h01, h11, h21 should be active
        auto basis = quintic_hermite_basis(1.0);

        TEST_ASSERT_NEAR(basis.h00, 0.0, 1e-15);   // p0 coefficient = 0
        TEST_ASSERT_NEAR(basis.h10, 0.0, 1e-15);   // v0 coefficient = 0
        TEST_ASSERT_NEAR(basis.h20, 0.0, 1e-15);   // a0 coefficient = 0
        TEST_ASSERT_NEAR(basis.h01, 1.0, 1e-15);   // p1 coefficient = 1
        TEST_ASSERT_NEAR(basis.h11, 0.0, 1e-15);   // v1 coefficient = 0 (but derivative = 1)
        TEST_ASSERT_NEAR(basis.h21, 0.0, 1e-15);   // a1 coefficient = 0 (but 2nd deriv = 1)
    });

    runner.add_test("deriv1_at_s_0", []() {
        // First derivative at s=0
        auto basis = quintic_hermite_deriv1(0.0);

        // dh00/ds at s=0 should be 0 (position is flat)
        TEST_ASSERT_NEAR(basis.h00, 0.0, 1e-15);
        // dh10/ds at s=0 should be 1 (velocity basis)
        TEST_ASSERT_NEAR(basis.h10, 1.0, 1e-15);
        // dh20/ds at s=0 should be 0 (acceleration basis has zero first derivative)
        TEST_ASSERT_NEAR(basis.h20, 0.0, 1e-15);
        // All endpoint 1 derivatives should be 0 at s=0
        TEST_ASSERT_NEAR(basis.h01, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h11, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h21, 0.0, 1e-15);
    });

    runner.add_test("deriv1_at_s_1", []() {
        // First derivative at s=1
        auto basis = quintic_hermite_deriv1(1.0);

        // All endpoint 0 derivatives should be 0 at s=1
        TEST_ASSERT_NEAR(basis.h00, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h10, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h20, 0.0, 1e-15);
        // dh01/ds at s=1 should be 0 (position is flat)
        TEST_ASSERT_NEAR(basis.h01, 0.0, 1e-15);
        // dh11/ds at s=1 should be 1 (velocity basis)
        TEST_ASSERT_NEAR(basis.h11, 1.0, 1e-15);
        // dh21/ds at s=1 should be 0
        TEST_ASSERT_NEAR(basis.h21, 0.0, 1e-15);
    });

    runner.add_test("deriv2_at_s_0", []() {
        // Second derivative at s=0
        auto basis = quintic_hermite_deriv2(0.0);

        TEST_ASSERT_NEAR(basis.h00, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h10, 0.0, 1e-15);
        // d²h20/ds² at s=0 should be 1 (acceleration basis)
        TEST_ASSERT_NEAR(basis.h20, 1.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h01, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h11, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h21, 0.0, 1e-15);
    });

    runner.add_test("deriv2_at_s_1", []() {
        // Second derivative at s=1
        auto basis = quintic_hermite_deriv2(1.0);

        TEST_ASSERT_NEAR(basis.h00, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h10, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h20, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h01, 0.0, 1e-15);
        TEST_ASSERT_NEAR(basis.h11, 0.0, 1e-15);
        // d²h21/ds² at s=1 should be 1 (acceleration basis)
        TEST_ASSERT_NEAR(basis.h21, 1.0, 1e-15);
    });

    // =========================================================================
    // Position Interpolation Tests
    // =========================================================================

    runner.add_test("position_matches_endpoints", []() {
        // Test that P(0) = p0 and P(1) = p1
        double dt = 10.0;  // 10 second interval

        Vector3d p0(1000.0, 2000.0, 3000.0);
        Vector3d v0(10.0, 20.0, 30.0);
        Vector3d a0(0.1, 0.2, 0.3);
        Vector3d p1(1100.0, 2200.0, 3300.0);
        Vector3d v1(11.0, 22.0, 33.0);
        Vector3d a1(0.15, 0.25, 0.35);

        Vector3d pos_at_0 = quintic_hermite_position(0.0, dt, p0, v0, a0, p1, v1, a1);
        Vector3d pos_at_1 = quintic_hermite_position(1.0, dt, p0, v0, a0, p1, v1, a1);

        TEST_ASSERT_VEC3_NEAR(pos_at_0, p0, 1e-12);
        TEST_ASSERT_VEC3_NEAR(pos_at_1, p1, 1e-12);
    });

    runner.add_test("velocity_matches_endpoints", []() {
        // Test that P'(0)/dt = v0 and P'(1)/dt = v1
        double dt = 10.0;

        Vector3d p0(1000.0, 2000.0, 3000.0);
        Vector3d v0(10.0, 20.0, 30.0);
        Vector3d a0(0.1, 0.2, 0.3);
        Vector3d p1(1100.0, 2200.0, 3300.0);
        Vector3d v1(11.0, 22.0, 33.0);
        Vector3d a1(0.15, 0.25, 0.35);

        Vector3d vel_at_0 = quintic_hermite_velocity(0.0, dt, p0, v0, a0, p1, v1, a1);
        Vector3d vel_at_1 = quintic_hermite_velocity(1.0, dt, p0, v0, a0, p1, v1, a1);

        TEST_ASSERT_VEC3_NEAR(vel_at_0, v0, 1e-12);
        TEST_ASSERT_VEC3_NEAR(vel_at_1, v1, 1e-12);
    });

    runner.add_test("acceleration_matches_endpoints", []() {
        // Test that P''(0)/dt² = a0 and P''(1)/dt² = a1
        double dt = 10.0;

        Vector3d p0(1000.0, 2000.0, 3000.0);
        Vector3d v0(10.0, 20.0, 30.0);
        Vector3d a0(0.1, 0.2, 0.3);
        Vector3d p1(1100.0, 2200.0, 3300.0);
        Vector3d v1(11.0, 22.0, 33.0);
        Vector3d a1(0.15, 0.25, 0.35);

        Vector3d acc_at_0 = quintic_hermite_acceleration(0.0, dt, p0, v0, a0, p1, v1, a1);
        Vector3d acc_at_1 = quintic_hermite_acceleration(1.0, dt, p0, v0, a0, p1, v1, a1);

        TEST_ASSERT_VEC3_NEAR(acc_at_0, a0, 1e-12);
        TEST_ASSERT_VEC3_NEAR(acc_at_1, a1, 1e-12);
    });

    // =========================================================================
    // Numerical Derivative Consistency
    // =========================================================================

    runner.add_test("velocity_matches_numerical_derivative", []() {
        // Verify that analytical velocity matches numerical derivative of position
        double dt = 10.0;
        double eps = 1e-8;

        Vector3d p0(6778137.0, 0.0, 0.0);
        Vector3d v0(0.0, 7500.0, 0.0);
        Vector3d a0(-9.0, 0.0, 0.0);
        Vector3d p1(6778100.0, 75000.0, 0.0);
        Vector3d v1(-100.0, 7498.0, 0.0);
        Vector3d a1(-8.9, 0.1, 0.0);

        double s = 0.5;  // Test at midpoint

        // Analytical velocity
        Vector3d vel_analytical = quintic_hermite_velocity(s, dt, p0, v0, a0, p1, v1, a1);

        // Numerical derivative: v = dP/dt = dP/ds * ds/dt = dP/ds / dt
        // where dP/ds = (P(s+eps) - P(s-eps)) / (2*eps)
        Vector3d pos_plus = quintic_hermite_position(s + eps, dt, p0, v0, a0, p1, v1, a1);
        Vector3d pos_minus = quintic_hermite_position(s - eps, dt, p0, v0, a0, p1, v1, a1);
        Vector3d vel_numerical = (pos_plus - pos_minus) / (2.0 * eps) / dt;

        // Should match to reasonable precision (finite difference has inherent error)
        TEST_ASSERT_VEC3_NEAR(vel_analytical, vel_numerical, 0.01);
    });

    runner.add_test("acceleration_matches_numerical_derivative", []() {
        // Verify that analytical acceleration matches numerical derivative of velocity
        double dt = 10.0;
        double eps = 1e-6;

        Vector3d p0(6778137.0, 0.0, 0.0);
        Vector3d v0(0.0, 7500.0, 0.0);
        Vector3d a0(-9.0, 0.0, 0.0);
        Vector3d p1(6778100.0, 75000.0, 0.0);
        Vector3d v1(-100.0, 7498.0, 0.0);
        Vector3d a1(-8.9, 0.1, 0.0);

        double s = 0.5;  // Test at midpoint

        // Analytical acceleration
        Vector3d acc_analytical = quintic_hermite_acceleration(s, dt, p0, v0, a0, p1, v1, a1);

        // Numerical second derivative: dv/dt = dv/ds * ds/dt = dv/ds / dt
        // where dv/ds = (vel_plus - vel_minus) / (2*eps)
        Vector3d vel_plus = quintic_hermite_velocity(s + eps, dt, p0, v0, a0, p1, v1, a1);
        Vector3d vel_minus = quintic_hermite_velocity(s - eps, dt, p0, v0, a0, p1, v1, a1);
        Vector3d acc_numerical = (vel_plus - vel_minus) / (2.0 * eps) / dt;

        // Should match to several significant digits
        TEST_ASSERT_VEC3_NEAR(acc_analytical, acc_numerical, 1e-3);
    });

    // =========================================================================
    // Combined State Function
    // =========================================================================

    runner.add_test("state_function_matches_individual", []() {
        // Verify that quintic_hermite_state gives same results as individual functions
        double dt = 15.0;

        Vector3d p0(6778137.0, 100.0, 200.0);
        Vector3d v0(10.0, 7500.0, 5.0);
        Vector3d a0(-8.5, 0.1, -0.05);
        Vector3d p1(6778000.0, 112500.0, 275.0);
        Vector3d v1(5.0, 7490.0, 4.9);
        Vector3d a1(-8.6, 0.05, -0.04);

        // Test at several points
        for (double s : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            auto state = quintic_hermite_state(s, dt, p0, v0, a0, p1, v1, a1);

            Vector3d pos_individual = quintic_hermite_position(s, dt, p0, v0, a0, p1, v1, a1);
            Vector3d vel_individual = quintic_hermite_velocity(s, dt, p0, v0, a0, p1, v1, a1);
            Vector3d acc_individual = quintic_hermite_acceleration(s, dt, p0, v0, a0, p1, v1, a1);

            TEST_ASSERT_VEC3_NEAR(state.position, pos_individual, 1e-12);
            TEST_ASSERT_VEC3_NEAR(state.velocity, vel_individual, 1e-12);
            TEST_ASSERT_VEC3_NEAR(state.acceleration, acc_individual, 1e-12);
        }
    });

    // =========================================================================
    // Polynomial Coefficient Tests
    // =========================================================================

    runner.add_test("segment_coefficients_match_direct", []() {
        // Verify that precomputed coefficients give same results as direct evaluation
        double dt = 20.0;

        Vector3d p0(6778137.0, 0.0, 0.0);
        Vector3d v0(0.0, 7500.0, 100.0);
        Vector3d a0(-9.0, 0.0, 0.01);
        Vector3d p1(6777500.0, 150000.0, 2000.0);
        Vector3d v1(-100.0, 7480.0, 99.0);
        Vector3d a1(-8.8, 0.2, 0.015);

        auto coeffs = compute_segment_coefficients(dt, p0, v0, a0, p1, v1, a1);

        double inv_dt = 1.0 / dt;
        double inv_dt2 = inv_dt * inv_dt;

        // Test at several points
        for (double s : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}) {
            Vector3d pos_coeff = coeffs.position(s);
            Vector3d vel_coeff = coeffs.velocity(s, inv_dt);
            Vector3d acc_coeff = coeffs.acceleration(s, inv_dt2);

            Vector3d pos_direct = quintic_hermite_position(s, dt, p0, v0, a0, p1, v1, a1);
            Vector3d vel_direct = quintic_hermite_velocity(s, dt, p0, v0, a0, p1, v1, a1);
            Vector3d acc_direct = quintic_hermite_acceleration(s, dt, p0, v0, a0, p1, v1, a1);

            // Positions in this test are ~6.78e6 m; 1e-6 absolute = 1.5e-13
            // relative, well above floating-point rounding (~3e-16) but tight
            // enough to catch real coefficient bugs. (A 1e-9 m bound would be
            // below the rounding error of numbers this large.)
            TEST_ASSERT_VEC3_NEAR(pos_coeff, pos_direct, 1e-6);
            TEST_ASSERT_VEC3_NEAR(vel_coeff, vel_direct, 1e-6);
            TEST_ASSERT_VEC3_NEAR(acc_coeff, acc_direct, 1e-6);
        }
    });

    // =========================================================================
    // Known Analytical Solution Test
    // =========================================================================

    runner.add_test("constant_velocity_trajectory", []() {
        // For constant velocity (zero acceleration), the path is a straight line
        // P(t) = p0 + v*t, so P(s*dt) should give p0 + v*s*dt
        double dt = 10.0;

        Vector3d p0(1000.0, 2000.0, 3000.0);
        Vector3d v(100.0, 200.0, 300.0);  // Constant velocity
        Vector3d a(0.0, 0.0, 0.0);        // Zero acceleration

        // Endpoints for constant velocity
        Vector3d p1 = p0 + v * dt;  // p1 = p0 + v*dt

        // Test interpolation
        for (double s : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            Vector3d expected_pos = p0 + v * (s * dt);
            Vector3d interpolated = quintic_hermite_position(s, dt, p0, v, a, p1, v, a);

            TEST_ASSERT_VEC3_NEAR(interpolated, expected_pos, 1e-10);

            // Velocity should be constant
            Vector3d vel = quintic_hermite_velocity(s, dt, p0, v, a, p1, v, a);
            TEST_ASSERT_VEC3_NEAR(vel, v, 1e-10);

            // Acceleration should be zero
            Vector3d acc = quintic_hermite_acceleration(s, dt, p0, v, a, p1, v, a);
            TEST_ASSERT_VEC3_NEAR(acc, a, 1e-10);
        }
    });

    runner.add_test("constant_acceleration_trajectory", []() {
        // For constant acceleration, P(t) = p0 + v0*t + 0.5*a*t²
        double dt = 10.0;

        Vector3d p0(1000.0, 2000.0, 3000.0);
        Vector3d v0(100.0, 200.0, 300.0);
        Vector3d a(1.0, 2.0, 3.0);  // Constant acceleration

        // Compute exact endpoints
        Vector3d p1 = p0 + v0 * dt + 0.5 * a * dt * dt;
        Vector3d v1 = v0 + a * dt;

        // Test interpolation
        for (double s : {0.0, 0.25, 0.5, 0.75, 1.0}) {
            double t = s * dt;
            Vector3d expected_pos = p0 + v0 * t + 0.5 * a * t * t;
            Vector3d expected_vel = v0 + a * t;

            Vector3d interpolated_pos = quintic_hermite_position(s, dt, p0, v0, a, p1, v1, a);
            Vector3d interpolated_vel = quintic_hermite_velocity(s, dt, p0, v0, a, p1, v1, a);
            Vector3d interpolated_acc = quintic_hermite_acceleration(s, dt, p0, v0, a, p1, v1, a);

            TEST_ASSERT_VEC3_NEAR(interpolated_pos, expected_pos, 1e-9);
            TEST_ASSERT_VEC3_NEAR(interpolated_vel, expected_vel, 1e-9);
            TEST_ASSERT_VEC3_NEAR(interpolated_acc, a, 1e-9);  // Constant acceleration
        }
    });

    // =========================================================================
    // Edge Cases
    // =========================================================================

    runner.add_test("very_small_dt", []() {
        // Test with very small time step (should still work)
        double dt = 0.001;  // 1 millisecond

        Vector3d p0(6778137.0, 0.0, 0.0);
        Vector3d v0(0.0, 7500.0, 0.0);
        Vector3d a0(-9.0, 0.0, 0.0);
        Vector3d p1 = p0 + v0 * dt + 0.5 * a0 * dt * dt;
        Vector3d v1 = v0 + a0 * dt;
        Vector3d a1(-9.0, 0.0, 0.0);

        // Should still match endpoints
        Vector3d pos_at_0 = quintic_hermite_position(0.0, dt, p0, v0, a0, p1, v1, a1);
        Vector3d pos_at_1 = quintic_hermite_position(1.0, dt, p0, v0, a0, p1, v1, a1);

        TEST_ASSERT_VEC3_NEAR(pos_at_0, p0, 1e-9);
        TEST_ASSERT_VEC3_NEAR(pos_at_1, p1, 1e-9);

        // Velocity at endpoints should match
        Vector3d vel_at_0 = quintic_hermite_velocity(0.0, dt, p0, v0, a0, p1, v1, a1);
        Vector3d vel_at_1 = quintic_hermite_velocity(1.0, dt, p0, v0, a0, p1, v1, a1);

        TEST_ASSERT_VEC3_NEAR(vel_at_0, v0, 1e-6);
        TEST_ASSERT_VEC3_NEAR(vel_at_1, v1, 1e-6);
    });

    runner.add_test("basis_sum_equals_one", []() {
        // For any s, the position basis functions should sum to 1 when
        // p0 = p1 = (1,1,1) and velocities/accelerations are zero
        // Actually, h00 + h01 should equal 1 (partition of unity for position)
        for (double s = 0.0; s <= 1.0; s += 0.1) {
            auto basis = quintic_hermite_basis(s);
            double pos_sum = basis.h00 + basis.h01;
            TEST_ASSERT_NEAR(pos_sum, 1.0, 1e-14);
        }
    });

    runner.add_test("symmetry_check", []() {
        // h00(s) = h01(1-s) (symmetric endpoints)
        // h10(s) = -h11(1-s) (antisymmetric for velocity)
        // h20(s) = h21(1-s) (symmetric for acceleration)
        for (double s = 0.0; s <= 0.5; s += 0.1) {
            auto basis_s = quintic_hermite_basis(s);
            auto basis_1ms = quintic_hermite_basis(1.0 - s);

            TEST_ASSERT_NEAR(basis_s.h00, basis_1ms.h01, 1e-14);
            TEST_ASSERT_NEAR(basis_s.h10, -basis_1ms.h11, 1e-14);
            TEST_ASSERT_NEAR(basis_s.h20, basis_1ms.h21, 1e-14);
        }
    });

    runner.run();
    return print_final_summary();
}
