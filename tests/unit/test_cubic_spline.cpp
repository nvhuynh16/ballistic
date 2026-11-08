// CubicSpline (math/cubic_spline.hpp): natural cubic spline interpolation
// (second derivative zero at both ends) of scalars and 3-vectors. Linear data must
// be reproduced exactly (1e-10); curved data only approximately, with the larger
// error near the ends where the natural end condition is wrong for the function.

#include "../test_utils.hpp"
#include <ballistic/math/cubic_spline.hpp>
#include <cmath>

using namespace ballistic::test;
using namespace ballistic::math;

int main() {
    TestRunner runner("CubicSpline");

    // =========================================================================
    // Scalar spline tests
    // =========================================================================

    runner.add_test("empty spline returns zero", []() {
        CubicSpline<double> s;
        TEST_ASSERT(s.empty());
        TEST_ASSERT_EQ(s.size(), 0);
        TEST_ASSERT_NEAR(s(42.0), 0.0, 1e-15);
    });

    runner.add_test("single point returns constant", []() {
        CubicSpline<double> s({5.0}, {3.14});
        TEST_ASSERT_EQ(s.size(), 1);
        TEST_ASSERT_NEAR(s(0.0), 3.14, 1e-15);
        TEST_ASSERT_NEAR(s(100.0), 3.14, 1e-15);
    });

    runner.add_test("two points: linear interpolation", []() {
        CubicSpline<double> s({0.0, 10.0}, {0.0, 20.0});
        TEST_ASSERT_EQ(s.size(), 2);
        TEST_ASSERT_NEAR(s(5.0), 10.0, 1e-12);
        TEST_ASSERT_NEAR(s(2.5), 5.0, 1e-12);
        // Boundary clamping
        TEST_ASSERT_NEAR(s(-1.0), 0.0, 1e-12);
        TEST_ASSERT_NEAR(s(15.0), 20.0, 1e-12);
    });

    runner.add_test("knot interpolation exact", []() {
        // Cubic spline passes through knots exactly regardless of data
        std::vector<double> x = {-2, -1, 0, 1, 2};
        std::vector<double> y = {-8, -1, 0, 1, 8};
        CubicSpline<double> s(x, y);

        TEST_ASSERT_NEAR(s(0.0), 0.0, 1e-12);
        TEST_ASSERT_NEAR(s(1.0), 1.0, 1e-12);
        TEST_ASSERT_NEAR(s(-1.0), -1.0, 1e-12);
        TEST_ASSERT_NEAR(s(-2.0), -8.0, 1e-12);
        TEST_ASSERT_NEAR(s(2.0), 8.0, 1e-12);
    });

    runner.add_test("linear data exact recovery", []() {
        // Natural cubic spline recovers linear functions exactly
        // f(x) = 3x + 1
        std::vector<double> x = {0, 1, 2, 3, 4};
        std::vector<double> y = {1, 4, 7, 10, 13};
        CubicSpline<double> s(x, y);

        TEST_ASSERT_NEAR(s(0.5), 2.5, 1e-12);
        TEST_ASSERT_NEAR(s(1.5), 5.5, 1e-12);
        TEST_ASSERT_NEAR(s(2.5), 8.5, 1e-12);
        TEST_ASSERT_NEAR(s(3.5), 11.5, 1e-12);
    });

    runner.add_test("quadratic approximation at interior", []() {
        // f(x) = x^2 on [0, 1, 2, 3, 4]
        // Natural BC (f''=0 at endpoints) doesn't match true f''=2,
        // so recovery is approximate near boundaries but good at interior
        std::vector<double> x = {0, 1, 2, 3, 4};
        std::vector<double> y = {0, 1, 4, 9, 16};
        CubicSpline<double> s(x, y);

        // Interior point (away from boundary): good approximation
        TEST_ASSERT_NEAR(s(2.5), 6.25, 0.1);
        // Near boundary: larger error from natural BC
        TEST_ASSERT_NEAR(s(0.5), 0.25, 0.15);
        // Knots are exact
        TEST_ASSERT_NEAR(s(2.0), 4.0, 1e-12);
    });

    runner.add_test("derivative at interior", []() {
        // f(x) = x^2 => f'(x) = 2x
        // Derivative at interior points should be approximate
        std::vector<double> x = {0, 1, 2, 3, 4};
        std::vector<double> y = {0, 1, 4, 9, 16};
        CubicSpline<double> s(x, y);

        // Interior: better accuracy
        TEST_ASSERT_NEAR(s.derivative(2.0), 4.0, 0.2);
        TEST_ASSERT_NEAR(s.derivative(2.5), 5.0, 0.2);
    });

    runner.add_test("boundary clamping", []() {
        CubicSpline<double> s({1.0, 2.0, 3.0}, {10.0, 20.0, 30.0});
        // Below domain
        TEST_ASSERT_NEAR(s(0.0), 10.0, 1e-12);
        TEST_ASSERT_NEAR(s(-100.0), 10.0, 1e-12);
        // Above domain
        TEST_ASSERT_NEAR(s(5.0), 30.0, 1e-12);
        TEST_ASSERT_NEAR(s(1000.0), 30.0, 1e-12);
    });

    runner.add_test("domain query", []() {
        CubicSpline<double> s({1.5, 3.7, 8.2}, {0, 0, 0});
        auto [lo, hi] = s.domain();
        TEST_ASSERT_NEAR(lo, 1.5, 1e-15);
        TEST_ASSERT_NEAR(hi, 8.2, 1e-15);
    });

    runner.add_test("many points smooth sine", []() {
        // Spline should approximate sin(x) well with enough points
        const int N = 20;
        std::vector<double> x(N), y(N);
        for (int i = 0; i < N; ++i) {
            x[i] = i * 3.14159265358979323846 / (N - 1);
            y[i] = std::sin(x[i]);
        }
        CubicSpline<double> s(x, y);

        // Check at quarter-period midpoints
        double mid = 3.14159265358979323846 / 4.0;
        double err = std::abs(s(mid) - std::sin(mid));
        TEST_ASSERT(err < 1e-3);  // Should be very close with 20 points on [0, pi]
    });

    // =========================================================================
    // Vector3d spline tests
    // =========================================================================

    runner.add_test("Vector3d: two points linear", []() {
        using V = Eigen::Vector3d;
        CubicSpline<V> s({0.0, 1.0}, {V(0, 0, 0), V(3, 6, 9)});
        V val = s(0.5);
        TEST_ASSERT_NEAR(val.x(), 1.5, 1e-12);
        TEST_ASSERT_NEAR(val.y(), 3.0, 1e-12);
        TEST_ASSERT_NEAR(val.z(), 4.5, 1e-12);
    });

    runner.add_test("Vector3d: quadratic approximation", []() {
        using V = Eigen::Vector3d;
        // f(t) = (t^2, 2t, 3) on [0,1,2,3]
        // Natural BC: quadratic component approximate, linear/constant exact
        std::vector<double> x = {0, 1, 2, 3};
        std::vector<V> y = {V(0,0,3), V(1,2,3), V(4,4,3), V(9,6,3)};
        CubicSpline<V> s(x, y);

        V val = s(1.5);
        TEST_ASSERT_NEAR(val.x(), 2.25, 0.1);  // quadratic: approximate
        TEST_ASSERT_NEAR(val.y(), 3.0, 1e-10);  // linear: exact
        TEST_ASSERT_NEAR(val.z(), 3.0, 1e-10);  // constant: exact
    });

    runner.add_test("Vector3d: derivative", []() {
        using V = Eigen::Vector3d;
        // f(t) = (t^2, 2t, 3) => f'(t) = (2t, 2, 0)
        std::vector<double> x = {0, 1, 2, 3};
        std::vector<V> y = {V(0,0,3), V(1,2,3), V(4,4,3), V(9,6,3)};
        CubicSpline<V> s(x, y);

        V d = s.derivative(1.5);
        TEST_ASSERT_NEAR(d.x(), 3.0, 1e-10);
        TEST_ASSERT_NEAR(d.y(), 2.0, 1e-10);
        TEST_ASSERT_NEAR(d.z(), 0.0, 1e-10);
    });

    runner.run();
    return print_final_summary();
}
