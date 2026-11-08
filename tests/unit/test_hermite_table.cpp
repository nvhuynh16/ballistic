// Unit tests for HermiteTable interpolation (math/hermite_table.hpp). With
// derivatives given at the knots it is a cubic Hermite interpolant (exact for
// cubics); without them it falls back to a natural cubic spline (exact for
// straight lines). Expected values are polynomial evaluations, so tolerances
// are at rounding level.

#include "../test_utils.hpp"
#include <ballistic/math/hermite_table.hpp>
#include <cmath>

using namespace ballistic::test;
using namespace ballistic::math;

int main() {
    TestRunner runner("HermiteTable");

    // =========================================================================
    // Scalar HermiteTable (without derivatives => CubicSpline)
    // =========================================================================

    runner.add_test("scalar no-derivs: delegates to CubicSpline", []() {
        // Linear data: natural cubic spline recovers exactly
        HermiteTable<double> h({0, 1, 2, 3}, {0, 2, 4, 6});
        TEST_ASSERT_FALSE(h.has_derivatives());
        TEST_ASSERT_NEAR(h(0.5), 1.0, 1e-12);
        TEST_ASSERT_NEAR(h(1.5), 3.0, 1e-12);
        TEST_ASSERT_NEAR(h(2.5), 5.0, 1e-12);
    });

    runner.add_test("scalar with-derivs: cubic Hermite", []() {
        // f(x) = x^3, f'(x) = 3x^2 on [0, 1, 2]
        HermiteTable<double> h(
            {0.0, 1.0, 2.0},
            {0.0, 1.0, 8.0},
            {0.0, 3.0, 12.0}
        );
        TEST_ASSERT(h.has_derivatives());
        // At x=0.5: f(0.5) = 0.125
        TEST_ASSERT_NEAR(h(0.5), 0.125, 1e-12);
        // At x=1.5: f(1.5) = 3.375
        TEST_ASSERT_NEAR(h(1.5), 3.375, 1e-12);
    });

    runner.add_test("scalar Hermite matches at knots", []() {
        HermiteTable<double> h(
            {0.0, 1.0, 2.0},
            {0.0, 1.0, 8.0},
            {0.0, 3.0, 12.0}
        );
        TEST_ASSERT_NEAR(h(0.0), 0.0, 1e-15);
        TEST_ASSERT_NEAR(h(1.0), 1.0, 1e-15);
        TEST_ASSERT_NEAR(h(2.0), 8.0, 1e-15);
    });

    runner.add_test("scalar boundary clamping", []() {
        HermiteTable<double> h({1.0, 2.0, 3.0}, {10.0, 20.0, 30.0});
        TEST_ASSERT_NEAR(h(0.0), 10.0, 1e-12);
        TEST_ASSERT_NEAR(h(5.0), 30.0, 1e-12);
    });

    runner.add_test("empty table", []() {
        HermiteTable<double> h;
        TEST_ASSERT(h.empty());
        TEST_ASSERT_EQ(h.size(), 0);
    });

    // =========================================================================
    // Vector3d HermiteTable (without derivatives => CubicSpline)
    // =========================================================================

    runner.add_test("Vector3d no-derivs: CubicSpline delegation", []() {
        using V = Eigen::Vector3d;
        // Linear data: natural cubic spline recovers exactly
        HermiteTable<V> h(
            {0, 1, 2, 3},
            {V(0,0,0), V(1,2,3), V(2,4,6), V(3,6,9)}
        );
        TEST_ASSERT_FALSE(h.has_derivatives());
        V val = h(1.5);
        TEST_ASSERT_NEAR(val.x(), 1.5, 1e-12);
        TEST_ASSERT_NEAR(val.y(), 3.0, 1e-12);
        TEST_ASSERT_NEAR(val.z(), 4.5, 1e-12);
    });

    runner.add_test("Vector3d with-derivs: cubic Hermite", []() {
        using V = Eigen::Vector3d;
        // f(t) = (t^3, t^2, t), f'(t) = (3t^2, 2t, 1)
        HermiteTable<V> h(
            {0.0, 1.0, 2.0},
            {V(0,0,0), V(1,1,1), V(8,4,2)},
            {V(0,0,1), V(3,2,1), V(12,4,1)}
        );
        TEST_ASSERT(h.has_derivatives());

        // At t=0.5: (0.125, 0.25, 0.5)
        V val = h(0.5);
        TEST_ASSERT_NEAR(val.x(), 0.125, 1e-12);
        TEST_ASSERT_NEAR(val.y(), 0.25, 1e-12);
        TEST_ASSERT_NEAR(val.z(), 0.5, 1e-12);
    });

    runner.add_test("Vector3d Hermite continuity at knot", []() {
        using V = Eigen::Vector3d;
        // f(t) = (t^3, t^2, t), check both sides of t=1.0
        HermiteTable<V> h(
            {0.0, 1.0, 2.0},
            {V(0,0,0), V(1,1,1), V(8,4,2)},
            {V(0,0,1), V(3,2,1), V(12,4,1)}
        );

        V left = h(1.0 - 1e-8);
        V right = h(1.0 + 1e-8);
        V at_knot = h(1.0);

        TEST_ASSERT_NEAR((left - at_knot).norm(), 0.0, 1e-5);
        TEST_ASSERT_NEAR((right - at_knot).norm(), 0.0, 1e-5);
    });

    runner.run();
    return print_final_summary();
}
