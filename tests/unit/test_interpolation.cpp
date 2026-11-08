// Unit tests for Lagrange polynomial interpolation (math/interpolation.hpp),
// used to interpolate tabulated orbits. Checks: polynomials up to the
// interpolation degree are reproduced exactly (to rounding), derivatives of
// known functions, refusal of times outside the data, and the choice of the
// window of points around the target time.

#include "../test_utils.hpp"
#include <ballistic/math/interpolation.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::math;
using Eigen::Vector3d;

// =============================================================================
// Lagrange Interpolation Tests
// =============================================================================

void test_lagrange_interpolation() {
    TestRunner runner("LagrangeInterpolation");

    runner.add_test("linear_exact", []() {
        // Linear data: f(t) = (2t, 3t, t)
        std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        std::vector<Vector3d> values;
        for (double t : times) {
            values.push_back(Vector3d(2 * t, 3 * t, t));
        }

        auto result = lagrange_interpolate(times, values, 4.5);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_VEC3_NEAR(*result, Vector3d(9.0, 13.5, 4.5), 1e-10);
    });

    runner.add_test("quadratic_exact", []() {
        // Quadratic data: f(t) = (t^2, 2t^2, 3t^2)
        std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        std::vector<Vector3d> values;
        for (double t : times) {
            values.push_back(Vector3d(t * t, 2 * t * t, 3 * t * t));
        }

        auto result = lagrange_interpolate(times, values, 3.5);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_VEC3_NEAR(*result, Vector3d(12.25, 24.5, 36.75), 1e-10);
    });

    runner.add_test("cubic_exact", []() {
        // Cubic data: f(t) = (t^3, 0, 0)
        std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        std::vector<Vector3d> values;
        for (double t : times) {
            values.push_back(Vector3d(t * t * t, 0.0, 0.0));
        }

        auto result = lagrange_interpolate(times, values, 2.7);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_NEAR((*result)(0), 2.7 * 2.7 * 2.7, 1e-8);
    });

    runner.add_test("at_data_point", []() {
        std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        std::vector<Vector3d> values;
        for (double t : times) {
            values.push_back(Vector3d(t * t, t, 1.0));
        }

        // Interpolate exactly at a data point
        auto result = lagrange_interpolate(times, values, 4.0);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_VEC3_NEAR(*result, Vector3d(16.0, 4.0, 1.0), 1e-10);
    });

    runner.add_test("insufficient_data_returns_nullopt", []() {
        // With order=9 (default), we need at least 9 points
        std::vector<double> times = {0.0, 1.0, 2.0};
        std::vector<Vector3d> values = {
            Vector3d(0, 0, 0), Vector3d(1, 1, 1), Vector3d(2, 2, 2)
        };

        auto result = lagrange_interpolate(times, values, 1.5);
        TEST_ASSERT_FALSE(result.has_value());
    });

    runner.add_test("lower_order_with_less_data", []() {
        // Use order=3 with just 3 data points
        std::vector<double> times = {0.0, 1.0, 2.0};
        std::vector<Vector3d> values = {
            Vector3d(0, 0, 0), Vector3d(1, 2, 3), Vector3d(4, 8, 12)
        };

        auto result = lagrange_interpolate(times, values, 0.5, 3);
        TEST_ASSERT(result.has_value());
    });

    runner.add_test("size_mismatch_returns_nullopt", []() {
        std::vector<double> times = {0.0, 1.0, 2.0};
        std::vector<Vector3d> values = {Vector3d(0, 0, 0), Vector3d(1, 1, 1)};

        auto result = lagrange_interpolate(times, values, 0.5);
        TEST_ASSERT_FALSE(result.has_value());
    });

    runner.add_test("extrapolation_refused", []() {
        // The public overloads refuse a time outside the data instead of
        // extrapolating the degree-8 polynomial (which can be far off there).
        // The raw templates without the check live in math::detail.
        std::vector<double> times;
        std::vector<Vector3d> values;
        for (int i = 0; i < 9; i++) {
            times.push_back(i);
            values.push_back(Vector3d(i * 2.0, 0.0, 0.0));
        }
        TEST_ASSERT_THROWS(lagrange_interpolate(times, values, 9.5), ballistic::DataCoverageError);
        TEST_ASSERT_THROWS(lagrange_derivative(times, values, -0.5), ballistic::DataCoverageError);
        TEST_ASSERT(lagrange_interpolate(times, values, 8.0).has_value());
    });

    runner.run();
}

// =============================================================================
// Lagrange Derivative Tests
// =============================================================================

void test_lagrange_derivative() {
    TestRunner runner("LagrangeDerivative");

    runner.add_test("constant_slope_for_linear", []() {
        // f(t) = (3t, 0, 0) -> f'(t) = (3, 0, 0)
        std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        std::vector<Vector3d> values;
        for (double t : times) {
            values.push_back(Vector3d(3.0 * t, 0.0, 0.0));
        }

        auto result = lagrange_derivative(times, values, 4.0);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_NEAR((*result)(0), 3.0, 1e-8);
        TEST_ASSERT_NEAR((*result)(1), 0.0, 1e-10);
    });

    runner.add_test("quadratic_derivative_is_2x", []() {
        // f(t) = (t^2, 0, 0) -> f'(t) = (2t, 0, 0)
        std::vector<double> times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        std::vector<Vector3d> values;
        for (double t : times) {
            values.push_back(Vector3d(t * t, 0.0, 0.0));
        }

        double t = 3.0;
        auto result = lagrange_derivative(times, values, t);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_NEAR((*result)(0), 2.0 * t, 1e-8);
    });

    runner.add_test("derivative_at_midpoint", []() {
        // f(t) = (sin(t), cos(t), 0) -> f'(t) = (cos(t), -sin(t), 0)
        std::vector<double> times;
        std::vector<Vector3d> values;
        for (int i = 0; i < 20; i++) {
            double t = i * 0.1;
            times.push_back(t);
            values.push_back(Vector3d(std::sin(t), std::cos(t), 0.0));
        }

        double t = 0.95;
        auto result = lagrange_derivative(times, values, t);
        TEST_ASSERT(result.has_value());
        TEST_ASSERT_NEAR((*result)(0), std::cos(t), 1e-6);
        TEST_ASSERT_NEAR((*result)(1), -std::sin(t), 1e-6);
    });

    runner.add_test("insufficient_data_returns_nullopt", []() {
        std::vector<double> times = {0.0, 1.0};
        std::vector<Vector3d> values = {Vector3d::Zero(), Vector3d::Ones()};

        auto result = lagrange_derivative(times, values, 0.5);
        TEST_ASSERT_FALSE(result.has_value());
    });

    runner.run();
}

// =============================================================================
// find_interpolation_window Tests
// =============================================================================

void test_find_interpolation_window() {
    TestRunner runner("FindInterpolationWindow");

    runner.add_test("centers_on_exact_match", []() {
        std::vector<double> data = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        auto get_time = [&](size_t i) { return data[i]; };

        // order=5, center=4 -> start = 4 - 2 = 2
        int start = find_interpolation_window(get_time, data.size(), 4.0, 5);
        TEST_ASSERT_EQ(start, 2);
    });

    runner.add_test("clamps_at_start", []() {
        std::vector<double> data = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        auto get_time = [&](size_t i) { return data[i]; };

        // target near start, order=5 -> should clamp to 0
        int start = find_interpolation_window(get_time, data.size(), 0.1, 5);
        TEST_ASSERT_EQ(start, 0);
    });

    runner.add_test("clamps_at_end", []() {
        std::vector<double> data = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        auto get_time = [&](size_t i) { return data[i]; };

        // target near end, order=5 -> should clamp to data_size - order = 4
        int start = find_interpolation_window(get_time, data.size(), 7.9, 5);
        TEST_ASSERT_EQ(start, 4);
    });

    runner.add_test("between_points_picks_nearest", []() {
        std::vector<double> data = {0.0, 1.0, 2.0, 3.0, 4.0};
        auto get_time = [&](size_t i) { return data[i]; };

        // t=2.3, nearest is index 2, order=3 -> start = 2 - 1 = 1
        int start = find_interpolation_window(get_time, data.size(), 2.3, 3);
        TEST_ASSERT_EQ(start, 1);
    });

    runner.add_test("beyond_end_clamps", []() {
        std::vector<double> data = {0.0, 1.0, 2.0, 3.0, 4.0};
        auto get_time = [&](size_t i) { return data[i]; };

        // target beyond all data, order=3 -> should clamp to data_size - order = 2
        int start = find_interpolation_window(get_time, data.size(), 10.0, 3);
        TEST_ASSERT_EQ(start, 2);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Lagrange Interpolation Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_lagrange_interpolation();
    test_lagrange_derivative();
    test_find_interpolation_window();

    return print_final_summary();
}
