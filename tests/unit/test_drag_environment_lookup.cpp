// Unit tests for DragEnvironment, the value that holds the three drag columns
// (air density, wind, ballistic coefficient β = m/(Cd·A) in kg/m²) and returns
// them at a position and time. Uses a small hand-built table: node values are
// recovered, values between nodes are interpolated, below the table throws,
// above it clamps, and set_beta replaces only the β column.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <stdexcept>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;

namespace {

Vector3d position_at(double lat_deg, double lon_deg, double alt_km) {
    constexpr double DEG_TO_RAD = 0.017453292519943295;
    return coords::geodetic_to_ecef(lat_deg * DEG_TO_RAD,
                                     lon_deg * DEG_TO_RAD,
                                     alt_km * 1000.0);
}

Epoch t0() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

DragEnvironment build_hand_env() {
    DensityProfile1D rho(
        {200.0, 300.0, 400.0, 500.0},
        {2.5e-10, 3.0e-11, 4.0e-12, 1.0e-12});
    WindProfile1D_ENU wind(
        {200.0, 400.0},
        {0.0, 0.0},      // no north-south wind
        {30.0, 50.0});   // 30 -> 50 m/s east
    BetaProfile1D beta = BetaProfile1D::constant(45.45);
    return DragEnvironment(std::move(rho), std::move(wind), std::move(beta));
}

}  // namespace

void test_default_construction() {
    TestRunner runner("DefaultConstruction");

    runner.add_test("empty_env_has_drag_false", []() {
        DragEnvironment env;
        TEST_ASSERT(!env.has_drag());
    });

    runner.add_test("empty_env_evaluate_returns_zero", []() {
        DragEnvironment env;
        // Need a position that won't hit the below-table throw — empty
        // density profiles have spline_.empty() and return 0 without checks.
        Vector3d pos = position_at(0.0, 0.0, 400.0);
        DragSample s = env.evaluate(pos, t0());
        TEST_ASSERT_NEAR(s.rho_kg_m3, 0.0, 1e-15);
        TEST_ASSERT(s.wind_ecef_m_s.norm() < 1e-12);
        TEST_ASSERT_NEAR(s.beta_kg_m2, 0.0, 1e-15);
    });
}

void test_hand_built_env() {
    TestRunner runner("HandBuiltEnv");

    runner.add_test("has_drag_true_after_populating_all_columns", []() {
        DragEnvironment env = build_hand_env();
        TEST_ASSERT(env.has_drag());
    });

    runner.add_test("evaluate_returns_node_values_at_nodes", []() {
        DragEnvironment env = build_hand_env();
        Vector3d pos = position_at(0.0, 0.0, 200.0);
        DragSample s = env.evaluate(pos, t0());
        // Density at 200 km altitude should be ~2.5e-10 (within 1%).
        TEST_ASSERT(std::abs(s.rho_kg_m3 - 2.5e-10) / 2.5e-10 < 1e-2);
        // Beta is constant 45.45.
        TEST_ASSERT_NEAR(s.beta_kg_m2, 45.45, 1e-9);
        // Wind at 200 km: 30 m/s east, transformed to ECEF at (lat=0, lon=0)
        // gives wind in +y direction.
        TEST_ASSERT(std::abs(s.wind_ecef_m_s.y() - 30.0) < 1.0);
    });

    runner.add_test("evaluate_interpolates_between_nodes", []() {
        DragEnvironment env = build_hand_env();
        Vector3d pos = position_at(0.0, 0.0, 350.0);
        DragSample s = env.evaluate(pos, t0());
        // 350 km: density should be between 3e-11 and 4e-12 (cubic spline in log).
        TEST_ASSERT(s.rho_kg_m3 < 3e-11);
        TEST_ASSERT(s.rho_kg_m3 > 4e-12);
    });

    runner.add_test("below_table_throws", []() {
        DragEnvironment env = build_hand_env();
        Vector3d pos = position_at(0.0, 0.0, 100.0);  // below table's 200 km
        bool threw = false;
        try {
            (void)env.evaluate(pos, t0());
        } catch (const std::out_of_range&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("above_table_clamps_to_top", []() {
        DragEnvironment env = build_hand_env();
        Vector3d pos = position_at(0.0, 0.0, 800.0);  // above 500 km
        DragSample s = env.evaluate(pos, t0());
        // Should clamp to the top altitude's density.
        TEST_ASSERT_NEAR(s.rho_kg_m3, 1.0e-12, 1e-13);
    });
}

void test_setters_for_calibration_loop() {
    TestRunner runner("CalibrationSetters");

    runner.add_test("set_beta_replaces_only_beta_column", []() {
        DragEnvironment env = build_hand_env();
        // Sample baseline at 250 km.
        Vector3d pos = position_at(0.0, 0.0, 250.0);
        DragSample before = env.evaluate(pos, t0());

        // Replace beta with a different profile.
        env.set_beta(BetaProfile1D::constant(100.0));
        DragSample after = env.evaluate(pos, t0());

        // Density and wind should be unchanged.
        TEST_ASSERT_NEAR(after.rho_kg_m3, before.rho_kg_m3, 1e-20);
        TEST_ASSERT_NEAR((after.wind_ecef_m_s - before.wind_ecef_m_s).norm(), 0.0, 1e-9);
        // Beta should be the new value.
        TEST_ASSERT_NEAR(after.beta_kg_m2, 100.0, 1e-9);
    });
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "DragEnvironment Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_default_construction();
    test_hand_built_env();
    test_setters_for_calibration_loop();

    return print_final_summary();
}
