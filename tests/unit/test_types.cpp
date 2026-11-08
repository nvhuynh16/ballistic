// Unit tests for the core value types in types.hpp: StateVector (position and
// velocity), PropagationResult, BallisticModel (holds the ballistic coefficient
// beta = m / (Cd * A), mass over drag coefficient times area, in kg/m^2),
// WeatherConditions (solar flux F10.7 and geomagnetic index Ap for the atmosphere
// model) and GeodeticCoord (latitude, longitude, altitude). Values are stored and
// returned unchanged, so tolerances are at floating-point level.

#include "../test_utils.hpp"
#include <ballistic/core/types.hpp>
#include <ballistic/core/constants.hpp>
#include <cmath>
#include <limits>

using namespace ballistic;
using namespace ballistic::test;

static TestRunner runner("Types Tests");

//=============================================================================
// StateVector Tests
//=============================================================================

void test_statevector_default_constructor() {
    StateVector s;
    TEST_ASSERT_NEAR(s.position.norm(), 0.0, 1e-15);
    TEST_ASSERT_NEAR(s.velocity.norm(), 0.0, 1e-15);
}

void test_statevector_parameterized_constructor() {
    Vector3d pos(1.0, 2.0, 3.0);
    Vector3d vel(4.0, 5.0, 6.0);
    StateVector s(pos, vel);

    TEST_ASSERT_NEAR(s.position.x(), 1.0, 1e-15);
    TEST_ASSERT_NEAR(s.position.y(), 2.0, 1e-15);
    TEST_ASSERT_NEAR(s.position.z(), 3.0, 1e-15);
    TEST_ASSERT_NEAR(s.velocity.x(), 4.0, 1e-15);
    TEST_ASSERT_NEAR(s.velocity.y(), 5.0, 1e-15);
    TEST_ASSERT_NEAR(s.velocity.z(), 6.0, 1e-15);
}

void test_statevector_to_vector() {
    Vector3d pos(1.0, 2.0, 3.0);
    Vector3d vel(4.0, 5.0, 6.0);
    StateVector s(pos, vel);

    Vector6d v = s.to_vector();
    TEST_ASSERT_NEAR(v(0), 1.0, 1e-15);
    TEST_ASSERT_NEAR(v(1), 2.0, 1e-15);
    TEST_ASSERT_NEAR(v(2), 3.0, 1e-15);
    TEST_ASSERT_NEAR(v(3), 4.0, 1e-15);
    TEST_ASSERT_NEAR(v(4), 5.0, 1e-15);
    TEST_ASSERT_NEAR(v(5), 6.0, 1e-15);
}

void test_statevector_from_vector() {
    Vector6d v;
    v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;

    StateVector s = StateVector::from_vector(v);
    TEST_ASSERT_NEAR(s.position.x(), 1.0, 1e-15);
    TEST_ASSERT_NEAR(s.position.y(), 2.0, 1e-15);
    TEST_ASSERT_NEAR(s.position.z(), 3.0, 1e-15);
    TEST_ASSERT_NEAR(s.velocity.x(), 4.0, 1e-15);
    TEST_ASSERT_NEAR(s.velocity.y(), 5.0, 1e-15);
    TEST_ASSERT_NEAR(s.velocity.z(), 6.0, 1e-15);
}

void test_statevector_roundtrip_conversion() {
    Vector3d pos(123.456, -789.012, 345.678);
    Vector3d vel(-0.001, 0.002, -0.003);
    StateVector s1(pos, vel);

    Vector6d v = s1.to_vector();
    StateVector s2 = StateVector::from_vector(v);

    TEST_ASSERT_VEC3_NEAR(s2.position, s1.position, 1e-15);
    TEST_ASSERT_VEC3_NEAR(s2.velocity, s1.velocity, 1e-15);
}

void test_statevector_addition() {
    StateVector s1(Vector3d(1, 2, 3), Vector3d(4, 5, 6));
    StateVector s2(Vector3d(10, 20, 30), Vector3d(40, 50, 60));

    StateVector result = s1 + s2;
    TEST_ASSERT_VEC3_NEAR(result.position, Vector3d(11, 22, 33), 1e-15);
    TEST_ASSERT_VEC3_NEAR(result.velocity, Vector3d(44, 55, 66), 1e-15);
}

void test_statevector_subtraction() {
    StateVector s1(Vector3d(10, 20, 30), Vector3d(40, 50, 60));
    StateVector s2(Vector3d(1, 2, 3), Vector3d(4, 5, 6));

    StateVector result = s1 - s2;
    TEST_ASSERT_VEC3_NEAR(result.position, Vector3d(9, 18, 27), 1e-15);
    TEST_ASSERT_VEC3_NEAR(result.velocity, Vector3d(36, 45, 54), 1e-15);
}

void test_statevector_scalar_multiplication() {
    StateVector s(Vector3d(1, 2, 3), Vector3d(4, 5, 6));

    StateVector result = s * 2.0;
    TEST_ASSERT_VEC3_NEAR(result.position, Vector3d(2, 4, 6), 1e-15);
    TEST_ASSERT_VEC3_NEAR(result.velocity, Vector3d(8, 10, 12), 1e-15);
}

void test_statevector_scalar_multiplication_left() {
    StateVector s(Vector3d(1, 2, 3), Vector3d(4, 5, 6));

    StateVector result = 3.0 * s;
    TEST_ASSERT_VEC3_NEAR(result.position, Vector3d(3, 6, 9), 1e-15);
    TEST_ASSERT_VEC3_NEAR(result.velocity, Vector3d(12, 15, 18), 1e-15);
}

void test_statevector_compound_addition() {
    StateVector s1(Vector3d(1, 2, 3), Vector3d(4, 5, 6));
    StateVector s2(Vector3d(10, 20, 30), Vector3d(40, 50, 60));

    s1 += s2;
    TEST_ASSERT_VEC3_NEAR(s1.position, Vector3d(11, 22, 33), 1e-15);
    TEST_ASSERT_VEC3_NEAR(s1.velocity, Vector3d(44, 55, 66), 1e-15);
}

void test_statevector_compound_multiplication() {
    StateVector s(Vector3d(1, 2, 3), Vector3d(4, 5, 6));

    s *= 2.5;
    TEST_ASSERT_VEC3_NEAR(s.position, Vector3d(2.5, 5.0, 7.5), 1e-15);
    TEST_ASSERT_VEC3_NEAR(s.velocity, Vector3d(10.0, 12.5, 15.0), 1e-15);
}

void test_statevector_large_values() {
    // Test with Earth-scale positions
    Vector3d pos(6.378137e6, 0, 0);  // Equatorial radius in meters
    Vector3d vel(0, 7.8e3, 0);       // Orbital velocity in m/s

    StateVector s(pos, vel);
    Vector6d v = s.to_vector();
    StateVector s2 = StateVector::from_vector(v);

    // Should maintain precision at Earth scale
    TEST_ASSERT_NEAR_REL(s2.position.x(), pos.x(), 1e-14);
    TEST_ASSERT_NEAR_REL(s2.velocity.y(), vel.y(), 1e-14);
}

void test_statevector_small_values() {
    // Test with very small values (might appear in derivatives)
    Vector3d pos(1e-10, -2e-10, 3e-10);
    Vector3d vel(1e-12, 2e-12, -3e-12);

    StateVector s(pos, vel);
    StateVector doubled = s * 2.0;

    TEST_ASSERT_NEAR(doubled.position.x(), 2e-10, 1e-25);
    TEST_ASSERT_NEAR(doubled.velocity.z(), -6e-12, 1e-27);
}

void test_statevector_zero_operations() {
    StateVector zero;
    StateVector s(Vector3d(1, 2, 3), Vector3d(4, 5, 6));

    // Adding zero
    StateVector result = s + zero;
    TEST_ASSERT_VEC3_NEAR(result.position, s.position, 1e-15);

    // Multiplying by zero
    StateVector zeroed = s * 0.0;
    TEST_ASSERT_NEAR(zeroed.position.norm(), 0.0, 1e-15);
    TEST_ASSERT_NEAR(zeroed.velocity.norm(), 0.0, 1e-15);
}

//=============================================================================
// PropagationResult Tests
//=============================================================================

void test_propagationresult_default_constructor() {
    PropagationResult pr;

    TEST_ASSERT_NEAR(pr.position.norm(), 0.0, 1e-15);
    TEST_ASSERT_NEAR(pr.velocity.norm(), 0.0, 1e-15);
    TEST_ASSERT_NEAR(pr.acceleration.norm(), 0.0, 1e-15);
    TEST_ASSERT_FALSE(pr.covariance.has_value());
}

void test_propagationresult_parameterized_constructor() {
    Vector3d pos(1, 2, 3);
    Vector3d vel(4, 5, 6);
    Vector3d acc(7, 8, 9);

    PropagationResult pr(pos, vel, acc);

    TEST_ASSERT_VEC3_NEAR(pr.position, pos, 1e-15);
    TEST_ASSERT_VEC3_NEAR(pr.velocity, vel, 1e-15);
    TEST_ASSERT_VEC3_NEAR(pr.acceleration, acc, 1e-15);
    TEST_ASSERT_FALSE(pr.covariance.has_value());
}

void test_propagationresult_covariance_optional() {
    PropagationResult pr;

    // Initially no covariance
    TEST_ASSERT_FALSE(pr.covariance.has_value());

    // Can set covariance
    Matrix6d cov = Matrix6d::Identity();
    pr.covariance = cov;
    TEST_ASSERT_TRUE(pr.covariance.has_value());

    // Can clear covariance
    pr.covariance = std::nullopt;
    TEST_ASSERT_FALSE(pr.covariance.has_value());
}

//=============================================================================
// BallisticModel Tests
//=============================================================================

void test_ballisticmodel_default_constructor() {
    BallisticModel bm;

    TEST_ASSERT_NEAR(bm.beta, 45.45, 1e-2);
    TEST_ASSERT_NEAR(bm.mass, 0.0, 1e-10);
    TEST_ASSERT_NEAR(bm.drag_area, 0.0, 1e-10);
}

void test_ballisticmodel_parameterized_constructor() {
    // e.g. 500 kg, 5 m^2, Cd 2.5 -> beta = 500/(2.5*5) = 40.0
    BallisticModel bm(40.0);

    TEST_ASSERT_NEAR(bm.beta, 40.0, 1e-10);
    TEST_ASSERT_NEAR(bm.mass, 0.0, 1e-10);
    TEST_ASSERT_NEAR(bm.drag_area, 0.0, 1e-10);
}

void test_ballisticmodel_ballistic_coefficient() {
    // Default: beta = 45.45
    BallisticModel bm;
    double bc = bm.ballistic_coefficient();
    TEST_ASSERT_NEAR(bc, 45.45, 1e-2);

    // Sentinel-3A: beta = 1250 / (2.2 * 4.8) = 118.371...
    double sentinel_beta = 1250.0 / (2.2 * 4.8);
    BallisticModel sentinel(sentinel_beta);
    double bc_sentinel = sentinel.ballistic_coefficient();
    TEST_ASSERT_NEAR(bc_sentinel, sentinel_beta, 1e-10);
}

void test_ballisticmodel_inverse_ballistic_coefficient() {
    // beta = 1000/(2.2*10) = 45.4545..., ibc = 1/beta = 0.022
    double beta_val = 1000.0 / 22.0;
    BallisticModel bm(beta_val);

    double ibc = bm.inverse_ballistic_coefficient();
    double bc = bm.ballistic_coefficient();

    // inverse_bc * bc = 1
    TEST_ASSERT_NEAR(ibc * bc, 1.0, 1e-14);

    // ibc = 1/beta = 22/1000 = 0.022
    TEST_ASSERT_NEAR(ibc, 0.022, 1e-10);
}

void test_ballisticmodel_sentinel3a_values() {
    // Sentinel-3A mass and drag area (the satellite registry values)
    // beta = 1250 / (2.2 * 4.8) = 118.371...
    double sentinel_beta = 1250.0 / (2.2 * 4.8);
    BallisticModel sentinel(sentinel_beta, 1250.0, 4.8);

    double bc = sentinel.ballistic_coefficient();
    TEST_ASSERT_NEAR(bc, sentinel_beta, 1e-10);
    TEST_ASSERT_NEAR(sentinel.mass, 1250.0, 1e-12);
    TEST_ASSERT_NEAR(sentinel.drag_area, 4.8, 1e-12);
}

void test_ballisticmodel_iss_values() {
    // ISS parameters (approximate): beta = 420000 / (2.3 * 2500) = 73.043...
    double iss_beta = 420000.0 / (2.3 * 2500.0);
    BallisticModel iss(iss_beta);

    // BC for ISS is relatively high due to large mass
    double bc = iss.ballistic_coefficient();
    TEST_ASSERT_NEAR(bc, iss_beta, 1e-10);
}

void test_ballisticmodel_extreme_values() {
    // Very light, high-drag (balloon-like): beta = 10/(2.5*100) = 0.04
    BallisticModel balloon(0.04);
    double bc_balloon = balloon.ballistic_coefficient();
    TEST_ASSERT_NEAR(bc_balloon, 0.04, 1e-10);

    // Very heavy, low-drag (dense sphere): beta = 10000/1.5 = 6666.67
    double dense_beta = 10000.0 / 1.5;
    BallisticModel dense(dense_beta);
    double bc_dense = dense.ballistic_coefficient();
    TEST_ASSERT_NEAR(bc_dense, dense_beta, 1e-10);
}

void test_ballisticmodel_near_zero_beta() {
    // Very small beta (high-drag, low-mass object)
    // e.g. mass=0.001, area=1, cd=2 -> beta = 0.001/2 = 0.0005
    BallisticModel tiny(0.0005);
    double bc = tiny.ballistic_coefficient();
    TEST_ASSERT_NEAR(bc, 0.0005, 1e-10);
    TEST_ASSERT_TRUE(std::isfinite(bc));
}

void test_ballisticmodel_large_beta() {
    // Very large beta (low-drag, high-mass object like a needle)
    // e.g. mass=100, area=0.001, cd=2 -> beta = 100/0.002 = 50000
    BallisticModel needle(50000.0);
    double bc = needle.ballistic_coefficient();
    TEST_ASSERT_TRUE(std::isfinite(bc));
    TEST_ASSERT_NEAR(bc, 50000.0, 1e-10);
}

//=============================================================================
// WeatherConditions Tests
//=============================================================================

void test_weatherconditions_default_constructor() {
    WeatherConditions w;

    TEST_ASSERT_NEAR(w.f107_daily, 150.0, 1e-10);
    TEST_ASSERT_NEAR(w.f107_average, 150.0, 1e-10);
    TEST_ASSERT_NEAR(w.ap_daily, 4.0, 1e-10);
    TEST_ASSERT_EQ(w.ap_history.size(), 7u);
}

void test_weatherconditions_quiet() {
    WeatherConditions w = WeatherConditions::quiet();

    TEST_ASSERT_NEAR(w.f107_daily, 70.0, 1e-10);
    TEST_ASSERT_NEAR(w.f107_average, 70.0, 1e-10);
    TEST_ASSERT_NEAR(w.ap_daily, 4.0, 1e-10);

    // All ap_history should be 4.0
    for (const auto& ap : w.ap_history) {
        TEST_ASSERT_NEAR(ap, 4.0, 1e-10);
    }
}

void test_weatherconditions_active() {
    WeatherConditions w = WeatherConditions::active();

    TEST_ASSERT_NEAR(w.f107_daily, 200.0, 1e-10);
    TEST_ASSERT_NEAR(w.f107_average, 200.0, 1e-10);
    TEST_ASSERT_NEAR(w.ap_daily, 15.0, 1e-10);

    // All ap_history should be 15.0
    for (const auto& ap : w.ap_history) {
        TEST_ASSERT_NEAR(ap, 15.0, 1e-10);
    }
}

void test_weatherconditions_storm() {
    WeatherConditions w = WeatherConditions::storm();

    TEST_ASSERT_NEAR(w.f107_daily, 250.0, 1e-10);
    TEST_ASSERT_NEAR(w.f107_average, 200.0, 1e-10);
    TEST_ASSERT_NEAR(w.ap_daily, 100.0, 1e-10);

    // Storm has decreasing Ap history
    TEST_ASSERT_NEAR(w.ap_history[0], 100.0, 1e-10);
    TEST_ASSERT_NEAR(w.ap_history[1], 80.0, 1e-10);
    TEST_ASSERT_NEAR(w.ap_history[6], 15.0, 1e-10);
}

void test_weatherconditions_f107_range() {
    // Quiet: F10.7 around 70 (solar minimum)
    WeatherConditions quiet = WeatherConditions::quiet();
    TEST_ASSERT_TRUE(quiet.f107_daily >= 65.0);
    TEST_ASSERT_TRUE(quiet.f107_daily <= 80.0);

    // Active: F10.7 around 200 (solar maximum)
    WeatherConditions active = WeatherConditions::active();
    TEST_ASSERT_TRUE(active.f107_daily >= 150.0);
    TEST_ASSERT_TRUE(active.f107_daily <= 250.0);
}

void test_weatherconditions_ap_range() {
    // Ap typically ranges from 0 to 400+
    // Quiet: Ap ~ 4
    WeatherConditions quiet = WeatherConditions::quiet();
    TEST_ASSERT_TRUE(quiet.ap_daily >= 0.0);
    TEST_ASSERT_TRUE(quiet.ap_daily <= 10.0);

    // Storm: Ap can exceed 100
    WeatherConditions storm = WeatherConditions::storm();
    TEST_ASSERT_TRUE(storm.ap_daily >= 50.0);
}

//=============================================================================
// GeodeticCoord Tests
//=============================================================================

void test_geodeticcoord_default_constructor() {
    GeodeticCoord g;

    TEST_ASSERT_NEAR(g.latitude, 0.0, 1e-15);
    TEST_ASSERT_NEAR(g.longitude, 0.0, 1e-15);
    TEST_ASSERT_NEAR(g.altitude, 0.0, 1e-15);
}

void test_geodeticcoord_parameterized_constructor() {
    double lat = constants::PI / 4.0;  // 45 degrees in radians
    double lon = -constants::PI / 2.0; // -90 degrees in radians
    double alt = 10000.0;              // 10 km

    GeodeticCoord g(lat, lon, alt);

    TEST_ASSERT_NEAR(g.latitude, lat, 1e-15);
    TEST_ASSERT_NEAR(g.longitude, lon, 1e-15);
    TEST_ASSERT_NEAR(g.altitude, alt, 1e-10);
}

void test_geodeticcoord_from_degrees() {
    GeodeticCoord g = GeodeticCoord::from_degrees(45.0, -90.0, 10000.0);

    TEST_ASSERT_NEAR(g.latitude, constants::PI / 4.0, 1e-10);
    TEST_ASSERT_NEAR(g.longitude, -constants::PI / 2.0, 1e-10);
    TEST_ASSERT_NEAR(g.altitude, 10000.0, 1e-10);
}

void test_geodeticcoord_pole_values() {
    // North pole
    GeodeticCoord north(constants::PI / 2.0, 0.0, 0.0);
    TEST_ASSERT_NEAR(north.latitude, constants::PI / 2.0, 1e-15);

    // South pole
    GeodeticCoord south(-constants::PI / 2.0, 0.0, 0.0);
    TEST_ASSERT_NEAR(south.latitude, -constants::PI / 2.0, 1e-15);
}

void test_geodeticcoord_high_altitude() {
    // GEO altitude
    GeodeticCoord geo(0.0, 0.0, 35786000.0);
    TEST_ASSERT_NEAR(geo.altitude, 35786000.0, 1.0);

    // Moon distance
    GeodeticCoord moon_dist(0.0, 0.0, 384400000.0);
    TEST_ASSERT_NEAR(moon_dist.altitude, 384400000.0, 1.0);
}

void test_geodeticcoord_negative_altitude() {
    // Below sea level (e.g., Dead Sea)
    GeodeticCoord dead_sea(0.55, 0.62, -430.0);
    TEST_ASSERT_NEAR(dead_sea.altitude, -430.0, 1e-10);
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "BALLISTIC LIBRARY - TYPES TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    // StateVector tests
    runner.add_test("statevector_default_constructor", test_statevector_default_constructor);
    runner.add_test("statevector_parameterized_constructor", test_statevector_parameterized_constructor);
    runner.add_test("statevector_to_vector", test_statevector_to_vector);
    runner.add_test("statevector_from_vector", test_statevector_from_vector);
    runner.add_test("statevector_roundtrip_conversion", test_statevector_roundtrip_conversion);
    runner.add_test("statevector_addition", test_statevector_addition);
    runner.add_test("statevector_subtraction", test_statevector_subtraction);
    runner.add_test("statevector_scalar_multiplication", test_statevector_scalar_multiplication);
    runner.add_test("statevector_scalar_multiplication_left", test_statevector_scalar_multiplication_left);
    runner.add_test("statevector_compound_addition", test_statevector_compound_addition);
    runner.add_test("statevector_compound_multiplication", test_statevector_compound_multiplication);
    runner.add_test("statevector_large_values", test_statevector_large_values);
    runner.add_test("statevector_small_values", test_statevector_small_values);
    runner.add_test("statevector_zero_operations", test_statevector_zero_operations);

    // PropagationResult tests
    runner.add_test("propagationresult_default_constructor", test_propagationresult_default_constructor);
    runner.add_test("propagationresult_parameterized_constructor", test_propagationresult_parameterized_constructor);
    runner.add_test("propagationresult_covariance_optional", test_propagationresult_covariance_optional);

    // BallisticModel tests
    runner.add_test("ballisticmodel_default_constructor", test_ballisticmodel_default_constructor);
    runner.add_test("ballisticmodel_parameterized_constructor", test_ballisticmodel_parameterized_constructor);
    runner.add_test("ballisticmodel_ballistic_coefficient", test_ballisticmodel_ballistic_coefficient);
    runner.add_test("ballisticmodel_inverse_ballistic_coefficient", test_ballisticmodel_inverse_ballistic_coefficient);
    runner.add_test("ballisticmodel_sentinel3a_values", test_ballisticmodel_sentinel3a_values);
    runner.add_test("ballisticmodel_iss_values", test_ballisticmodel_iss_values);
    runner.add_test("ballisticmodel_extreme_values", test_ballisticmodel_extreme_values);
    runner.add_test("ballisticmodel_near_zero_beta", test_ballisticmodel_near_zero_beta);
    runner.add_test("ballisticmodel_large_beta", test_ballisticmodel_large_beta);

    // WeatherConditions tests
    runner.add_test("weatherconditions_default_constructor", test_weatherconditions_default_constructor);
    runner.add_test("weatherconditions_quiet", test_weatherconditions_quiet);
    runner.add_test("weatherconditions_active", test_weatherconditions_active);
    runner.add_test("weatherconditions_storm", test_weatherconditions_storm);
    runner.add_test("weatherconditions_f107_range", test_weatherconditions_f107_range);
    runner.add_test("weatherconditions_ap_range", test_weatherconditions_ap_range);

    // GeodeticCoord tests
    runner.add_test("geodeticcoord_default_constructor", test_geodeticcoord_default_constructor);
    runner.add_test("geodeticcoord_parameterized_constructor", test_geodeticcoord_parameterized_constructor);
    runner.add_test("geodeticcoord_from_degrees", test_geodeticcoord_from_degrees);
    runner.add_test("geodeticcoord_pole_values", test_geodeticcoord_pole_values);
    runner.add_test("geodeticcoord_high_altitude", test_geodeticcoord_high_altitude);
    runner.add_test("geodeticcoord_negative_altitude", test_geodeticcoord_negative_altitude);

    int result = runner.run();
    return print_final_summary();
}
