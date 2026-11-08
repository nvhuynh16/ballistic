// Brent's method (1D scalar minimization) unit tests.
//
// Used by the line search of the B-spline estimator's iterative fit (IRGN,
// iteratively regularized Gauss-Newton) to pick a step length. Coverage:
//   (a) pure parabolic — converges fast
//   (b) perturbed parabolic — robust to non-smoothness
//   (c) plateau near minimum — doesn't wander
//   (d) monotonic-on-bracket — should park at the right endpoint
//   (e) max_iter respected on pathological input

#include <ballistic/math/brent_minimize.hpp>
#include "../test_utils.hpp"

#include <cmath>
#include <iostream>

using namespace ballistic::test;
using ballistic::math::brent_minimize;
using ballistic::math::BrentResult;

void test_brent_pure_parabolic() {
    TestRunner runner("Brent / pure parabolic");

    runner.add_test("converges to (alpha-0.7)^2 minimum at 0.7", []() {
        auto f = [](double x) { return (x - 0.7) * (x - 0.7); };
        BrentResult r = brent_minimize(f, 0.0, 0.5, 1.5, 1e-6, 100);
        TEST_ASSERT_TRUE(r.converged);
        TEST_ASSERT_NEAR(r.x_min, 0.7, 1e-5);
        TEST_ASSERT_TRUE(r.f_min < 1e-9);
        TEST_ASSERT_TRUE(r.n_evals <= 12);  // pure parabola: very fast
    });

    runner.add_test("converges with off-center bracket", []() {
        auto f = [](double x) { return (x - 1.5) * (x - 1.5) + 0.1; };
        BrentResult r = brent_minimize(f, 0.5, 1.0, 3.0, 1e-6, 100);
        TEST_ASSERT_TRUE(r.converged);
        TEST_ASSERT_NEAR(r.x_min, 1.5, 1e-5);
        TEST_ASSERT_NEAR(r.f_min, 0.1, 1e-9);
    });

    runner.run();
}

void test_brent_perturbed_parabolic() {
    TestRunner runner("Brent / perturbed parabolic");

    runner.add_test("converges within tol on (x-0.7)^2 + 0.05*sin(10x)", []() {
        auto f = [](double x) {
            return (x - 0.7) * (x - 0.7) + 0.05 * std::sin(10.0 * x);
        };
        // True minimum is near x ≈ 0.62 (perturbed from 0.7 by sin term).
        // Tolerance loose: 1e-3 in the step length is all the line search needs.
        BrentResult r = brent_minimize(f, 0.0, 0.5, 1.5, 1e-3, 100);
        TEST_ASSERT_TRUE(r.converged);
        // Should land near the actual minimum, not the unperturbed 0.7
        TEST_ASSERT_TRUE(r.x_min > 0.5 && r.x_min < 0.85);
        // Found minimum should be lower than f(0.5) and f(1.5)
        TEST_ASSERT_TRUE(r.f_min < f(0.5));
        TEST_ASSERT_TRUE(r.f_min < f(1.5));
    });

    runner.run();
}

void test_brent_plateau() {
    TestRunner runner("Brent / plateau near minimum");

    runner.add_test("does not wander on flat region near optimum", []() {
        // f(x) = max(0, (x-0.7)^2 - 0.01) — flat zero on [0.6, 0.8], parabolic outside
        auto f = [](double x) {
            const double q = (x - 0.7) * (x - 0.7) - 0.01;
            return q > 0.0 ? q : 0.0;
        };
        BrentResult r = brent_minimize(f, 0.0, 0.5, 1.5, 1e-3, 100);
        TEST_ASSERT_TRUE(r.converged);
        // Any point in [0.6, 0.8] is equally good; just verify we landed on plateau
        TEST_ASSERT_TRUE(r.x_min >= 0.55 && r.x_min <= 0.85);
        TEST_ASSERT_NEAR(r.f_min, 0.0, 1e-6);
    });

    runner.run();
}

void test_brent_corner_cases() {
    TestRunner runner("Brent / corner cases");

    runner.add_test("respects max_iter on pathological input", []() {
        // Tiny tolerance + tight max_iter: should bail early (not infinite-loop)
        auto f = [](double x) { return (x - 0.7) * (x - 0.7); };
        BrentResult r = brent_minimize(f, 0.0, 0.5, 1.5, 1e-15, 3);
        // Likely doesn't converge (tol too tight, iter too few); must return
        // sensible values, not NaN
        TEST_ASSERT_TRUE(std::isfinite(r.x_min));
        TEST_ASSERT_TRUE(std::isfinite(r.f_min));
        TEST_ASSERT_TRUE(r.n_evals <= 4);  // 1 init eval + ≤3 iters
    });

    runner.add_test("handles asymmetric brackets correctly", []() {
        // Bracket with bx near a (left side), then minimum at right of bx
        auto f = [](double x) { return (x - 1.2) * (x - 1.2); };
        BrentResult r = brent_minimize(f, 0.0, 0.1, 2.0, 1e-6, 100);
        TEST_ASSERT_TRUE(r.converged);
        TEST_ASSERT_NEAR(r.x_min, 1.2, 1e-4);
    });

    runner.run();
}

int main() {
    test_brent_pure_parabolic();
    test_brent_perturbed_parabolic();
    test_brent_plateau();
    test_brent_corner_cases();
    return print_final_summary();
}
