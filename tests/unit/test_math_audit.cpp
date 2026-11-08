// Math module audit tests: each routine is checked against an independent
// reference (a hand-computed value, a polynomial it must reproduce exactly, an
// analytic derivative, or its theoretical convergence order) rather than against
// its own earlier output. Tolerances are near machine precision for exact cases
// and derived from the method's error order otherwise. No data files.
// Tests: math/chebyshev.hpp, math/interpolation.hpp, math/quintic_hermite.hpp

#define _USE_MATH_DEFINES
#include <cmath>

#include "../test_utils.hpp"
#include <ballistic/math/chebyshev.hpp>
#include <ballistic/math/interpolation.hpp>
#include <ballistic/math/quintic_hermite.hpp>
#include <Eigen/Dense>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::math;

// =============================================================================
// Chebyshev Derivative Tests
// (derivatives of Chebyshev series; the first derivative coefficient c'_0 is
// halved by the primed-sum convention, Numerical Recipes eq. 5.8.11)
// =============================================================================

void test_chebyshev_audit() {
    TestRunner runner("ChebyshevAudit");

    runner.add_test("chebyshev_derivative_T1_correct", []() {
        // f(x) = T_1(x) = x -> f'(x) = 1 -> f'(t) = 1/h
        double coeffs[] = {0.0, 1.0};
        double h = 5.0;

        // Derivative must be 1/h everywhere, to rounding (1e-14).
        for (double x : {-0.9, -0.5, 0.0, 0.3, 0.7, 1.0}) {
            double d = chebyshev_deriv(coeffs, 2, x, h);
            TEST_ASSERT_NEAR(d, 1.0 / h, 1e-14);
        }

        // Also verify via chebyshev_eval_with_derivative
        auto [val, deriv] = chebyshev_eval_with_derivative(coeffs, 2, 0.5, h);
        TEST_ASSERT_NEAR(val, 0.5, 1e-14);
        TEST_ASSERT_NEAR(deriv, 1.0 / h, 1e-14);
    });

    runner.add_test("chebyshev_derivative_T3_correct", []() {
        // f(x) = T_3(x) = 4x^3 - 3x -> f'(x) = 12x^2 - 3
        // Chebyshev coeffs for T_3: {0, 0, 0, 1}
        double coeffs[] = {0.0, 0.0, 0.0, 1.0};
        double h = 1.0;  // h=1 so df/dt = df/dx

        // f'(0) = -3
        double d0 = chebyshev_deriv(coeffs, 4, 0.0, h);
        TEST_ASSERT_NEAR(d0, -3.0, 1e-13);

        // f'(0.5) = 12*0.25 - 3 = 0
        double d05 = chebyshev_deriv(coeffs, 4, 0.5, h);
        TEST_ASSERT_NEAR(d05, 0.0, 1e-13);

        // f'(1) = 12 - 3 = 9
        double d1 = chebyshev_deriv(coeffs, 4, 1.0, h);
        TEST_ASSERT_NEAR(d1, 9.0, 1e-13);

        // f'(-1) = 12 - 3 = 9
        double dm1 = chebyshev_deriv(coeffs, 4, -1.0, h);
        TEST_ASSERT_NEAR(dm1, 9.0, 1e-13);
    });

    runner.add_test("chebyshev_eval_3d_hand_computed", []() {
        // 3-component function:
        //   X(x) = 1 + 2x  (T_0 + 2*T_1, coeffs = {1, 2})
        //   Y(x) = 3 - x   (3*T_0 - T_1, coeffs = {3, -1})
        //   Z(x) = x^2 = 0.5*T_0 + 0.5*T_2  (coeffs = {0.5, 0, 0.5})
        // Using 3 coefficients per component (pad X,Y with zero)
        double coeffs[] = {
            1.0, 2.0, 0.0,   // X
            3.0, -1.0, 0.0,  // Y
            0.5, 0.0, 0.5    // Z (T_2 = 2x^2 - 1, so 0.5*1 + 0.5*(2x^2-1) = x^2)
        };
        double h = 2.0;
        double x = 0.5;

        auto [pos, vel] = chebyshev_eval_3d(coeffs, 3, x, h);

        // Position: X=1+2*0.5=2, Y=3-0.5=2.5, Z=0.25
        TEST_ASSERT_NEAR(pos[0], 2.0, 1e-14);
        TEST_ASSERT_NEAR(pos[1], 2.5, 1e-14);
        TEST_ASSERT_NEAR(pos[2], 0.25, 1e-14);

        // Velocity (df/dt = df/dx / h):
        //   X'(x) = 2 -> 2/h = 1
        //   Y'(x) = -1 -> -1/h = -0.5
        //   Z'(x) = 2x = 1 -> 1/h = 0.5
        TEST_ASSERT_NEAR(vel[0], 1.0, 1e-14);
        TEST_ASSERT_NEAR(vel[1], -0.5, 1e-14);
        TEST_ASSERT_NEAR(vel[2], 0.5, 1e-14);
    });

    runner.add_test("chebyshev_eval_3d_vs_separate_eval", []() {
        // Verify chebyshev_eval_3d matches 3 separate chebyshev_eval_with_derivative calls
        double coeffs[] = {
            1.5, -0.3, 0.7, 0.2,   // X: 4 coeffs
            2.1, 0.5, -0.4, 0.1,   // Y
            -0.8, 1.2, 0.3, -0.6   // Z
        };
        int n = 4;
        double x = 0.37;
        double h = 7.5;

        auto [pos3d, vel3d] = chebyshev_eval_3d(coeffs, n, x, h);

        for (int i = 0; i < 3; ++i) {
            auto [p, v] = chebyshev_eval_with_derivative(coeffs + i * n, n, x, h);
            TEST_ASSERT_NEAR(pos3d[i], p, 1e-15);
            TEST_ASSERT_NEAR(vel3d[i], v, 1e-15);
        }
    });

    runner.run();
}

// =============================================================================
// Lagrange Interpolation Tests
// =============================================================================

void test_lagrange_audit() {
    TestRunner runner("LagrangeAudit");

    runner.add_test("lagrange_non_uniform_spacing", []() {
        // Chebyshev nodes on [0, 10], interpolate f(t) = t^4
        // With 9 points (degree 8 polynomial), t^4 should be reproduced exactly
        const int n = 9;
        std::vector<double> times(n);
        std::vector<Eigen::Vector3d> values(n);

        for (int i = 0; i < n; ++i) {
            // Chebyshev nodes on [0, 10]: t_i = 5 + 5*cos((2i+1)*pi/(2n))
            double t = 5.0 + 5.0 * std::cos((2.0 * i + 1.0) * M_PI / (2.0 * n));
            times[i] = t;
            double t4 = t * t * t * t;
            values[i] = Eigen::Vector3d(t4, 2.0 * t4, -t4);
        }

        // Sort by time (Chebyshev nodes aren't monotonic as generated)
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (times[j] < times[i]) {
                    std::swap(times[i], times[j]);
                    std::swap(values[i], values[j]);
                }
            }
        }

        // Interpolate at t = 3.7 (not a node); exact up to rounding of values ~1e2-1e4.
        double t_eval = 3.7;
        double t4_exact = t_eval * t_eval * t_eval * t_eval;
        auto result = lagrange_interpolate(times, values, t_eval, n);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_NEAR(result->x(), t4_exact, 1e-8);
        TEST_ASSERT_NEAR(result->y(), 2.0 * t4_exact, 1e-8);
        TEST_ASSERT_NEAR(result->z(), -t4_exact, 1e-8);
    });

    runner.add_test("lagrange_polynomial_exactness", []() {
        // p(t) = t^7 - 2t^5 + 3t^3 - t + 5, degree 7
        // 9 uniform points -> degree 8 polynomial can reproduce degree 7 exactly
        const int n = 9;
        std::vector<double> times(n);
        std::vector<Eigen::Vector3d> values(n);

        auto poly = [](double t) -> double {
            double t2 = t * t;
            double t3 = t2 * t;
            double t5 = t3 * t2;
            double t7 = t5 * t2;
            return t7 - 2.0 * t5 + 3.0 * t3 - t + 5.0;
        };

        for (int i = 0; i < n; ++i) {
            double t = -2.0 + i * 0.5;  // [-2, 2] uniform spacing
            times[i] = t;
            double p = poly(t);
            values[i] = Eigen::Vector3d(p, 0.0, 0.0);
        }

        double t_eval = 1.73;
        double expected = poly(t_eval);
        auto result = lagrange_interpolate(times, values, t_eval, n);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_NEAR(result->x(), expected, 1e-6);
    });

    runner.add_test("lagrange_derivative_analytic", []() {
        // f(t) = sin(t) on [0, 2*pi], 20 points
        // f'(pi/4) = cos(pi/4) = sqrt(2)/2; a 9-point stencil's truncation error
        // is far below the 1e-6 tolerance.
        const int n = 20;
        std::vector<double> times(n);
        std::vector<Eigen::Vector3d> values(n);

        for (int i = 0; i < n; ++i) {
            double t = i * 2.0 * M_PI / (n - 1);
            times[i] = t;
            values[i] = Eigen::Vector3d(std::sin(t), 0.0, 0.0);
        }

        double t_eval = M_PI / 4.0;
        double expected_deriv = std::cos(t_eval);
        auto result = lagrange_derivative(times, values, t_eval, 9);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_NEAR(result->x(), expected_deriv, 1e-6);
    });

    runner.run();
}

// =============================================================================
// Quintic Hermite Tests
// =============================================================================

void test_quintic_hermite_audit() {
    TestRunner runner("QuinticHermiteAudit");

    runner.add_test("quintic_hermite_6x6_system_verification", []() {
        // Independently derive basis function coefficients by solving 6x6 systems
        // Constraint matrix M for polynomial p(s) = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4 + c5*s^5
        Eigen::Matrix<double, 6, 6> M;
        // Row 0: p(0) = c0
        M.row(0) << 1, 0, 0, 0, 0, 0;
        // Row 1: p(1) = c0+c1+c2+c3+c4+c5
        M.row(1) << 1, 1, 1, 1, 1, 1;
        // Row 2: p'(0) = c1
        M.row(2) << 0, 1, 0, 0, 0, 0;
        // Row 3: p'(1) = c1+2c2+3c3+4c4+5c5
        M.row(3) << 0, 1, 2, 3, 4, 5;
        // Row 4: p''(0) = 2*c2
        M.row(4) << 0, 0, 2, 0, 0, 0;
        // Row 5: p''(1) = 2c2+6c3+12c4+20c5
        M.row(5) << 0, 0, 2, 6, 12, 20;

        // Expected basis function coefficients (from header):
        // h00: [1, 0, 0, -10, 15, -6]
        // h10: [0, 1, 0, -6, 8, -3]
        // h20: [0, 0, 0.5, -1.5, 1.5, -0.5]
        // h01: [0, 0, 0, 10, -15, 6]
        // h11: [0, 0, 0, -4, 7, -3]
        // h21: [0, 0, 0, 0.5, -1, 0.5]

        // RHS vectors for each basis function
        struct BasisCase {
            Eigen::Matrix<double, 6, 1> rhs;
            double expected[6];
        };

        BasisCase cases[] = {
            // h00: p(0)=1, p(1)=0, p'(0)=0, p'(1)=0, p''(0)=0, p''(1)=0
            {{1, 0, 0, 0, 0, 0}, {1, 0, 0, -10, 15, -6}},
            // h10: p(0)=0, p(1)=0, p'(0)=1, p'(1)=0, p''(0)=0, p''(1)=0
            {{0, 0, 1, 0, 0, 0}, {0, 1, 0, -6, 8, -3}},
            // h20: p(0)=0, p(1)=0, p'(0)=0, p'(1)=0, p''(0)=1, p''(1)=0
            {{0, 0, 0, 0, 1, 0}, {0, 0, 0.5, -1.5, 1.5, -0.5}},
            // h01: p(0)=0, p(1)=1, p'(0)=0, p'(1)=0, p''(0)=0, p''(1)=0
            {{0, 1, 0, 0, 0, 0}, {0, 0, 0, 10, -15, 6}},
            // h11: p(0)=0, p(1)=0, p'(0)=0, p'(1)=1, p''(0)=0, p''(1)=0
            {{0, 0, 0, 1, 0, 0}, {0, 0, 0, -4, 7, -3}},
            // h21: p(0)=0, p(1)=0, p'(0)=0, p'(1)=0, p''(0)=0, p''(1)=1
            {{0, 0, 0, 0, 0, 1}, {0, 0, 0, 0.5, -1, 0.5}},
        };

        for (int b = 0; b < 6; ++b) {
            Eigen::Matrix<double, 6, 1> solved = M.colPivHouseholderQr().solve(cases[b].rhs);
            for (int j = 0; j < 6; ++j) {
                TEST_ASSERT_NEAR(solved(j), cases[b].expected[j], 1e-12);
            }
        }

        // Cross-check: evaluate basis functions at s=0.5 and compare
        auto basis = quintic_hermite_basis(0.5);
        // h00(0.5) = 1 - 10*0.125 + 15*0.0625 - 6*0.03125 = 1 - 1.25 + 0.9375 - 0.1875 = 0.5
        TEST_ASSERT_NEAR(basis.h00, 0.5, 1e-14);
        // h01(0.5) = 10*0.125 - 15*0.0625 + 6*0.03125 = 1.25 - 0.9375 + 0.1875 = 0.5
        TEST_ASSERT_NEAR(basis.h01, 0.5, 1e-14);
    });

    runner.add_test("quintic_hermite_quintic_polynomial_exact", []() {
        // f(t) = t^5 - 2t^4 + 3t^3 - t^2 + t - 1 on [0, 2]
        // Quintic Hermite with 6 DOF reproduces any degree-5 polynomial exactly
        auto f = [](double t) -> double {
            return t * t * t * t * t - 2.0 * t * t * t * t + 3.0 * t * t * t - t * t + t - 1.0;
        };
        auto fp = [](double t) -> double {
            return 5.0 * t * t * t * t - 8.0 * t * t * t + 9.0 * t * t - 2.0 * t + 1.0;
        };
        auto fpp = [](double t) -> double {
            return 20.0 * t * t * t - 24.0 * t * t + 18.0 * t - 2.0;
        };

        double t0 = 0.0, t1 = 2.0;
        double dt = t1 - t0;

        Vector3d p0(f(t0), 0, 0), v0(fp(t0), 0, 0), a0(fpp(t0), 0, 0);
        Vector3d p1(f(t1), 0, 0), v1(fp(t1), 0, 0), a1(fpp(t1), 0, 0);

        for (double s : {0.3, 0.5, 0.7}) {
            double t = t0 + s * dt;
            Vector3d pos = quintic_hermite_position(s, dt, p0, v0, a0, p1, v1, a1);
            TEST_ASSERT_NEAR(pos.x(), f(t), 1e-10);

            Vector3d vel = quintic_hermite_velocity(s, dt, p0, v0, a0, p1, v1, a1);
            TEST_ASSERT_NEAR(vel.x(), fp(t), 1e-10);

            Vector3d acc = quintic_hermite_acceleration(s, dt, p0, v0, a0, p1, v1, a1);
            TEST_ASSERT_NEAR(acc.x(), fpp(t), 1e-10);
        }
    });

    runner.add_test("quintic_hermite_convergence_order", []() {
        // f(t) = A*sin(omega*t), orbital-scale amplitude
        // Use omega large enough that error is well above machine epsilon at all step sizes
        const double A = 6778137.0;  // LEO orbit radius in meters
        const double omega = 0.01;   // rad/s

        auto f = [&](double t) -> double { return A * std::sin(omega * t); };
        auto fp = [&](double t) -> double { return A * omega * std::cos(omega * t); };
        auto fpp = [&](double t) -> double { return -A * omega * omega * std::sin(omega * t); };

        // Test at step sizes h = 160, 80, 40, 20 seconds
        double step_sizes[] = {160.0, 80.0, 40.0, 20.0};
        double errors[4];

        for (int k = 0; k < 4; ++k) {
            double h = step_sizes[k];
            double t0 = 100.0;  // arbitrary start
            double t1 = t0 + h;
            double t_mid = t0 + h / 2.0;  // evaluate at midpoint

            Vector3d p0(f(t0), 0, 0), v0(fp(t0), 0, 0), a0(fpp(t0), 0, 0);
            Vector3d p1(f(t1), 0, 0), v1(fp(t1), 0, 0), a1(fpp(t1), 0, 0);

            Vector3d interp = quintic_hermite_position(0.5, h, p0, v0, a0, p1, v1, a1);
            errors[k] = std::abs(interp.x() - f(t_mid));
        }

        // Verify O(h^6) convergence: error ratio between consecutive halvings ~ 2^6 = 64
        for (int k = 0; k < 3; ++k) {
            double ratio = errors[k] / errors[k + 1];
            // Allow factor-of-2 tolerance on the ratio (32 to 128)
            TEST_ASSERT_MSG(ratio > 32.0 && ratio < 128.0,
                "Convergence ratio " + std::to_string(ratio) + " outside [32, 128] (expected ~64)");
        }
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Math Module Audit Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_chebyshev_audit();
    test_lagrange_audit();
    test_quintic_hermite_audit();

    return print_final_summary();
}
