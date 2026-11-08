// Unit tests for Chebyshev polynomial evaluation (math/chebyshev.hpp), used to
// evaluate the JPL planetary ephemeris. Checks values and derivatives against
// the closed forms T0 = 1, T1 = x, T2 = 2x^2 - 1, T3 = 4x^3 - 3x and the
// endpoint identities T_n(1) = 1, T_n(-1) = (-1)^n, to machine precision. The
// derivative is with respect to time t, where x = (t - mid) / half_interval,
// so it carries a 1/half_interval factor.

#include "../test_utils.hpp"
#include <ballistic/math/chebyshev.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::math;

// =============================================================================
// Chebyshev Evaluation Tests
// =============================================================================

void test_chebyshev_eval() {
    TestRunner runner("ChebyshevEval");

    runner.add_test("T0_equals_1", []() {
        // T_0(x) = 1 for all x -> coeffs = {1.0}
        double coeffs[] = {1.0};
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 1, 0.0), 1.0, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 1, 0.5), 1.0, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 1, -1.0), 1.0, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 1, 1.0), 1.0, 1e-15);
    });

    runner.add_test("T1_equals_x", []() {
        // f(x) = T_1(x) = x -> coeffs = {0.0, 1.0}
        double coeffs[] = {0.0, 1.0};
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 2, 0.0), 0.0, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 2, 0.5), 0.5, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 2, -0.7), -0.7, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 2, 1.0), 1.0, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 2, -1.0), -1.0, 1e-15);
    });

    runner.add_test("T2_equals_2x2_minus_1", []() {
        // f(x) = T_2(x) = 2x^2 - 1 -> coeffs = {0.0, 0.0, 1.0}
        double coeffs[] = {0.0, 0.0, 1.0};
        // T_2(0) = -1
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 3, 0.0), -1.0, 1e-15);
        // T_2(0.5) = 2*0.25 - 1 = -0.5
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 3, 0.5), -0.5, 1e-15);
        // T_2(1) = 2 - 1 = 1
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 3, 1.0), 1.0, 1e-15);
        // T_2(-1) = 2 - 1 = 1
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 3, -1.0), 1.0, 1e-15);
    });

    runner.add_test("known_polynomial_sum", []() {
        // f(x) = 3*T_0(x) + 2*T_1(x) + T_2(x)
        //       = 3 + 2x + (2x^2 - 1) = 2x^2 + 2x + 2
        double coeffs[] = {3.0, 2.0, 1.0};

        double x = 0.5;
        double expected = 2.0 * x * x + 2.0 * x + 2.0;
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 3, x), expected, 1e-14);

        x = -0.3;
        expected = 2.0 * x * x + 2.0 * x + 2.0;
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 3, x), expected, 1e-14);
    });

    runner.add_test("boundary_x_plus_1", []() {
        // T_n(1) = 1 for all n
        // f(1) = sum(coeffs[i]) for coeffs of T_i at x=1
        double coeffs[] = {1.0, 2.0, 3.0, 4.0};
        // f(1) = 1*T_0(1) + 2*T_1(1) + 3*T_2(1) + 4*T_3(1) = 1+2+3+4 = 10
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 4, 1.0), 10.0, 1e-13);
    });

    runner.add_test("boundary_x_minus_1", []() {
        // T_n(-1) = (-1)^n
        // f(-1) = sum(coeffs[i] * (-1)^i)
        double coeffs[] = {1.0, 2.0, 3.0, 4.0};
        // f(-1) = 1*1 + 2*(-1) + 3*1 + 4*(-1) = 1-2+3-4 = -2
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 4, -1.0), -2.0, 1e-13);
    });

    runner.add_test("zero_coefficients", []() {
        TEST_ASSERT_NEAR(chebyshev_eval(nullptr, 0, 0.5), 0.0, 1e-15);
    });

    runner.add_test("single_coefficient", []() {
        double coeffs[] = {42.0};
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 1, 0.0), 42.0, 1e-15);
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 1, 1.0), 42.0, 1e-15);
    });

    runner.add_test("higher_order_T3", []() {
        // T_3(x) = 4x^3 - 3x
        double coeffs[] = {0.0, 0.0, 0.0, 1.0};
        double x = 0.5;
        double expected = 4.0 * x * x * x - 3.0 * x;
        TEST_ASSERT_NEAR(chebyshev_eval(coeffs, 4, x), expected, 1e-14);
    });

    runner.run();
}

// =============================================================================
// Chebyshev Derivative Tests
// =============================================================================

void test_chebyshev_deriv() {
    TestRunner runner("ChebyshevDeriv");

    runner.add_test("constant_derivative_is_zero", []() {
        // f(x) = c (constant) -> f'(t) = 0
        double coeffs[] = {5.0};
        double half_interval = 100.0;
        TEST_ASSERT_NEAR(chebyshev_deriv(coeffs, 1, 0.0, half_interval), 0.0, 1e-15);
    });

    runner.add_test("linear_derivative_is_constant", []() {
        // f(x) = T_1(x) = x -> df/dx = 1 -> df/dt = 1/h
        double coeffs[] = {0.0, 1.0};
        double half_interval = 10.0;

        double deriv = chebyshev_deriv(coeffs, 2, 0.0, half_interval);
        TEST_ASSERT_NEAR(deriv, 1.0 / half_interval, 1e-14);

        // Should be constant (independent of x)
        double deriv2 = chebyshev_deriv(coeffs, 2, 0.5, half_interval);
        TEST_ASSERT_NEAR(deriv, deriv2, 1e-14);
    });

    runner.add_test("quadratic_derivative", []() {
        // f(x) = T_2(x) = 2x^2 - 1 -> df/dx = 4x
        // Derivative coefficients: c'_1 = 2*2*c_2 = 4, c'_0 = 2*1*c_1 + c'_2 = 0
        // So deriv = chebyshev_eval({0, 4}, 2, x) / h = (4x) / h
        // This happens to be correct because c'_0 = 0 here
        double coeffs[] = {0.0, 0.0, 1.0};
        double half_interval = 5.0;

        double x = 0.5;
        double expected = 4.0 * x / half_interval;
        TEST_ASSERT_NEAR(chebyshev_deriv(coeffs, 3, x, half_interval), expected, 1e-13);

        x = -0.3;
        expected = 4.0 * x / half_interval;
        TEST_ASSERT_NEAR(chebyshev_deriv(coeffs, 3, x, half_interval), expected, 1e-13);
    });

    runner.add_test("half_interval_scaling", []() {
        // Derivative should scale inversely with half_interval
        double coeffs[] = {0.0, 1.0};
        double x = 0.5;

        double d1 = chebyshev_deriv(coeffs, 2, x, 1.0);
        double d10 = chebyshev_deriv(coeffs, 2, x, 10.0);

        TEST_ASSERT_NEAR_REL(d1 / d10, 10.0, 1e-14);
    });

    runner.add_test("at_zero", []() {
        // T_1'(0) = 1 -> df/dt = 1/h
        double coeffs[] = {0.0, 1.0};
        double h = 3.0;
        TEST_ASSERT_NEAR(chebyshev_deriv(coeffs, 2, 0.0, h), 1.0 / h, 1e-14);
    });

    runner.run();
}

// =============================================================================
// Chebyshev Eval With Derivative Tests
// =============================================================================

void test_chebyshev_eval_with_derivative() {
    TestRunner runner("ChebyshevEvalWithDeriv");

    runner.add_test("value_matches_eval", []() {
        double coeffs[] = {3.0, 2.0, 1.0};
        double x = 0.5;
        double half_interval = 10.0;

        auto [val, deriv] = chebyshev_eval_with_derivative(coeffs, 3, x, half_interval);

        double expected_val = chebyshev_eval(coeffs, 3, x);
        TEST_ASSERT_NEAR(val, expected_val, 1e-15);
    });

    runner.add_test("derivative_matches_deriv", []() {
        double coeffs[] = {3.0, 2.0, 1.0};
        double x = 0.5;
        double half_interval = 10.0;

        auto [val, deriv] = chebyshev_eval_with_derivative(coeffs, 3, x, half_interval);

        double expected_deriv = chebyshev_deriv(coeffs, 3, x, half_interval);
        TEST_ASSERT_NEAR(deriv, expected_deriv, 1e-14);
    });

    runner.add_test("zero_coefficients", []() {
        auto [val, deriv] = chebyshev_eval_with_derivative(nullptr, 0, 0.5, 1.0);
        TEST_ASSERT_NEAR(val, 0.0, 1e-15);
        TEST_ASSERT_NEAR(deriv, 0.0, 1e-15);
    });

    runner.add_test("single_coefficient", []() {
        double coeffs[] = {7.0};
        auto [val, deriv] = chebyshev_eval_with_derivative(coeffs, 1, 0.5, 1.0);
        TEST_ASSERT_NEAR(val, 7.0, 1e-15);
        TEST_ASSERT_NEAR(deriv, 0.0, 1e-15);
    });

    runner.add_test("long_series_derivative_is_computed", []() {
        // 70 coefficients (more than the stack buffer holds): the derivative used to come
        // back as 0. Series = T_69, so d/dx = 69 U_68(x) = 69 sin(69 t) / sin(t), x = cos(t).
        std::vector<double> c(70, 0.0);
        c[69] = 1.0;
        for (double x : {-0.83, -0.2, 0.37, 0.91}) {
            const double t = std::acos(x);
            const double expected = 69.0 * std::sin(69.0 * t) / std::sin(t);
            const double half = 2.5;
            auto [v, d] = chebyshev_eval_with_derivative(c.data(), 70, x, half);
            TEST_ASSERT_NEAR(v, std::cos(69.0 * t), 1e-12);
            TEST_ASSERT_NEAR(d, expected / half, 1e-9 * std::max(1.0, std::abs(expected)));
            TEST_ASSERT_NEAR(chebyshev_deriv(c.data(), 70, x, half), d, 0.0);
        }
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Chebyshev Polynomial Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_chebyshev_eval();
    test_chebyshev_deriv();
    test_chebyshev_eval_with_derivative();

    return print_final_summary();
}
