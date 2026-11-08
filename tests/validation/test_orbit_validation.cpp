// =============================================================================
// Orbit Propagation Validation Tests
// =============================================================================
//
// Sanity checks on synthetic circular orbits (no real orbit data): energy and
// radius after one period, J2 nodal drift and the sun-synchronous inclination
// from closed forms, 1-hour stability, GCRS <-> ITRF round trips, geodetic
// round trips, and parsing of a small SP3 text (SP3 = the standard precise-orbit
// file format). Bounds are physical plausibility ranges, not accuracy claims.
//
// Terms: ECI = Earth-centered inertial (here GCRS, the frame the propagator
// integrates in); ECEF = Earth-centered Earth-fixed (here ITRF); J2 = the
// Earth-oblateness term of the gravity field; RAAN = right ascension of the
// ascending node.
// =============================================================================

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include "../sp3_truth.hpp"
#include <vector>
#include <string>

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;
using namespace ballistic::test;

// =============================================================================
// Synthetic Test Orbits
// =============================================================================

// Generate a circular LEO orbit for testing
struct TestOrbit {
    Epoch epoch;
    Vector3d position_ecef;
    Vector3d velocity_ecef;
    double altitude_km;
    double inclination_deg;
    std::string name;
};

TestOrbit create_circular_leo(double altitude_km, double inclination_deg,
                              double raan_deg, double mean_anomaly_deg,
                              const Epoch& epoch, const std::string& name) {
    using namespace constants;

    TestOrbit orbit;
    orbit.epoch = epoch;
    orbit.altitude_km = altitude_km;
    orbit.inclination_deg = inclination_deg;
    orbit.name = name;

    // Compute orbital elements
    double a = earth::RADIUS_EQUATORIAL + altitude_km * 1000.0;  // Semi-major axis (m)
    double v = std::sqrt(earth::GM / a);  // Orbital velocity (m/s)

    // Convert angles to radians
    double inc = inclination_deg * DEG_TO_RAD;
    double raan = raan_deg * DEG_TO_RAD;
    double ma = mean_anomaly_deg * DEG_TO_RAD;

    // Position in orbital plane
    double x_orb = a * std::cos(ma);
    double y_orb = a * std::sin(ma);

    // Velocity in orbital plane
    double vx_orb = -v * std::sin(ma);
    double vy_orb = v * std::cos(ma);

    // Rotation matrices
    // R3(RAAN) * R1(inc) * R3(arg_perigee=0)
    double cos_raan = std::cos(raan);
    double sin_raan = std::sin(raan);
    double cos_inc = std::cos(inc);
    double sin_inc = std::sin(inc);

    // Position in ECI
    Vector3d pos_eci;
    pos_eci.x() = cos_raan * x_orb - sin_raan * cos_inc * y_orb;
    pos_eci.y() = sin_raan * x_orb + cos_raan * cos_inc * y_orb;
    pos_eci.z() = sin_inc * y_orb;

    // Velocity in ECI
    Vector3d vel_eci;
    vel_eci.x() = cos_raan * vx_orb - sin_raan * cos_inc * vy_orb;
    vel_eci.y() = sin_raan * vx_orb + cos_raan * cos_inc * vy_orb;
    vel_eci.z() = sin_inc * vy_orb;

    // Convert the GCRS state to ITRF (Earth-fixed), which the propagator takes
    StateVector state_eci(pos_eci, vel_eci);
    StateVector state_ecef = coords::gcrs_to_itrf_full(state_eci, epoch);
    orbit.position_ecef = state_ecef.position;
    orbit.velocity_ecef = state_ecef.velocity;

    return orbit;
}

// =============================================================================
// Test Functions
// =============================================================================

// Test: two-body energy changes by < 1e-4 (relative) over one orbit
void test_kepler_energy_conservation() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // Create 400 km circular orbit
    auto orbit = create_circular_leo(400.0, 51.6, 0.0, 0.0, t0, "ISS-like");

    // Initial energy
    double r0 = orbit.position_ecef.norm();
    double v0 = orbit.velocity_ecef.norm();
    double E0 = 0.5 * v0 * v0 - constants::earth::GM / r0;

    // Default force model (EGM2008 gravity field, not two-body); third body off.
    // No DragEnvironment is attached, so drag is off.
    BallisticModel model(1000.0, 10.0, 0.0);  // No drag
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = create_rk45_propagator(t0, orbit.position_ecef, orbit.velocity_ecef,
                                        model, weather, false);  // No drag

    // Propagate one orbit (~90 minutes)
    double period = 2.0 * constants::PI * std::sqrt(
        std::pow(constants::earth::RADIUS_EQUATORIAL + 400000.0, 3) / constants::earth::GM);

    Epoch t1 = t0 + period;
    auto result = (*prop)(t1);

    // Final energy
    double r1 = result.position.norm();
    double v1 = result.velocity.norm();
    double E1 = 0.5 * v1 * v1 - constants::earth::GM / r1;

    // Two-body energy computed from Earth-fixed position and velocity is not a
    // conserved quantity under the full gravity field and Earth rotation, so
    // the bound is a loose 1e-4 relative.
    double dE = std::abs((E1 - E0) / E0);
    TEST_ASSERT(dE < 1e-4);  // Relative energy error < 0.01%
}

// Test: after one two-body period the radius is back within 200 m
void test_kepler_period_accuracy() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // Create orbit
    auto orbit = create_circular_leo(400.0, 0.0, 0.0, 0.0, t0, "Equatorial");

    // Theoretical period
    double a = constants::earth::RADIUS_EQUATORIAL + 400000.0;
    double T_theory = 2.0 * constants::PI * std::sqrt(a * a * a / constants::earth::GM);

    BallisticModel model(1000.0, 10.0, 0.0);
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = create_rk45_propagator(t0, orbit.position_ecef, orbit.velocity_ecef,
                                        model, weather, false);

    // Propagate exactly one period
    Epoch t1 = t0 + T_theory;
    auto result = (*prop)(t1);

    // Only the radius is compared: the gravity field beyond point mass moves
    // the satellite along the orbit and shifts the plane over one period.
    double r_error = std::abs(result.position.norm() - orbit.position_ecef.norm());
    TEST_ASSERT(r_error < 200.0);  // < 200 m allows for J2 and tides
}

// Test: J2 causes RAAN drift for inclined orbits
void test_j2_raan_drift() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // Create 400 km, 51.6 inclination orbit (ISS-like)
    auto orbit = create_circular_leo(400.0, 51.6, 0.0, 0.0, t0, "ISS-like");

    // Theoretical RAAN drift rate from J2
    double a = constants::earth::RADIUS_EQUATORIAL + 400000.0;
    double n = std::sqrt(constants::earth::GM / (a * a * a));
    double Re_a = constants::earth::RADIUS_EQUATORIAL / a;
    double cos_i = std::cos(51.6 * constants::DEG_TO_RAD);
    double raan_rate = -1.5 * constants::earth::J2 * n * Re_a * Re_a * cos_i;

    // Expected RAAN drift over 24 hours
    double dt = 86400.0;  // 24 hours
    double expected_drift_rad = raan_rate * dt;
    double expected_drift_deg = expected_drift_rad * constants::RAD_TO_DEG;

    // ISS RAAN drifts about -5 deg/day
    TEST_ASSERT(std::abs(expected_drift_deg - (-5.0)) < 1.0);  // Within 1 deg/day
}

// Test: Sun-synchronous orbit maintains constant RAAN with respect to Sun
void test_sun_synchronous_orbit() {
    // Sun-synchronous requires specific inclination for given altitude
    // For ~700 km altitude, inclination ~ 98.2
    double alt_km = 700.0;
    double a = constants::earth::RADIUS_EQUATORIAL + alt_km * 1000.0;
    double n = std::sqrt(constants::earth::GM / (a * a * a));
    double Re_a = constants::earth::RADIUS_EQUATORIAL / a;

    // Required RAAN rate = Earth's angular velocity around Sun
    double raan_rate_required = 2.0 * constants::PI / (365.25 * 86400.0);  // rad/s

    // Solve for inclination: cos(i) = -raan_rate / (1.5 * J2 * n * (Re/a)^2)
    double cos_i = -raan_rate_required / (1.5 * constants::earth::J2 * n * Re_a * Re_a);
    double inc_deg = std::acos(cos_i) * constants::RAD_TO_DEG;

    // Should be around 98.2 for 700 km
    TEST_ASSERT(std::abs(inc_deg - 98.2) < 1.0);
}

// Test: Short-term propagation accuracy (1 minute)
void test_short_term_accuracy() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    auto orbit = create_circular_leo(400.0, 51.6, 0.0, 0.0, t0, "LEO");

    BallisticModel model(1000.0, 10.0, 2.2);
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = create_rk45_propagator(t0, orbit.position_ecef, orbit.velocity_ecef,
                                        model, weather, true);

    // Propagate 1 minute
    Epoch t1 = t0 + 60.0;
    auto result = (*prop)(t1);

    // Check position is reasonable
    TEST_ASSERT(result.position.norm() > 6.7e6);  // Above Earth surface
    TEST_ASSERT(result.position.norm() < 7.0e6);  // Below 630 km

    // Velocity should be ~7.7 km/s for LEO
    // Note: ECEF velocity includes Earth rotation contribution
    double v = result.velocity.norm();
    TEST_ASSERT(v > 7000.0);  // Adjusted for ECEF frame
    TEST_ASSERT(v < 8500.0);
}

// Test: One-hour propagation stability
void test_one_hour_stability() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    auto orbit = create_circular_leo(400.0, 51.6, 0.0, 0.0, t0, "LEO");

    BallisticModel model(1000.0, 10.0, 2.2);
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = create_rk45_propagator(t0, orbit.position_ecef, orbit.velocity_ecef,
                                        model, weather, true);

    // Propagate 1 hour with intermediate checks
    double dt = 60.0;  // 1-minute steps
    Epoch t = t0;

    for (int i = 0; i < 60; ++i) {
        t = t0 + (i + 1) * dt;
        auto result = (*prop)(t);

        // Verify position stays reasonable
        double r = result.position.norm();
        TEST_ASSERT(r > 6.5e6);  // Above Earth
        TEST_ASSERT(r < 7.5e6);  // Not escaped

        // Verify velocity stays reasonable
        double v = result.velocity.norm();
        TEST_ASSERT(v > 7000.0);
        TEST_ASSERT(v < 8500.0);
    }
}

// Test: radius decreases over a day at 300 km.
// NOTE: create_rk45_propagator attaches no DragEnvironment and the
// BallisticModel/weather arguments are not read, so drag is OFF here; the
// radius change this cell asserts comes from the rest of the force model, not
// from drag. Drag decay is tested where a DragEnvironment is attached.
void test_drag_altitude_decay() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    auto orbit = create_circular_leo(300.0, 51.6, 0.0, 0.0, t0, "Low-LEO");

    // Not read for drag (no DragEnvironment)
    BallisticModel model(1000.0, 20.0, 2.5);  // 20 m^2 area
    WeatherConditions weather = WeatherConditions::active();  // Higher density

    auto prop = create_rk45_propagator(t0, orbit.position_ecef, orbit.velocity_ecef,
                                        model, weather, true);

    double r0 = orbit.position_ecef.norm();

    // Propagate 1 day
    Epoch t1 = t0 + 86400.0;
    auto result = (*prop)(t1);

    double r1 = result.position.norm();

    TEST_ASSERT(r1 < r0);
    TEST_ASSERT(r0 - r1 > 100.0);
}

// Test: GCRS -> ITRF -> GCRS round trip returns the start state (1 m, 1 cm/s
// bounds; the transforms are exact inverses, so this is far looser than needed)
void test_eci_ecef_roundtrip() {
    Epoch epoch = Epoch::from_utc(2024, 6, 21, 12, 0, 0);  // Summer solstice

    // Test point in ECI
    Vector3d pos_eci(6778137.0, 0.0, 0.0);
    Vector3d vel_eci(0.0, 7500.0, 0.0);

    // Convert to ECEF
    StateVector state_eci_orig(pos_eci, vel_eci);
    StateVector state_ecef = coords::gcrs_to_itrf_full(state_eci_orig, epoch);
    Vector3d pos_ecef = state_ecef.position;
    Vector3d vel_ecef = state_ecef.velocity;

    // Convert back to ECI
    StateVector state_eci2 = coords::itrf_to_gcrs_full(state_ecef, epoch);
    Vector3d pos_eci2 = state_eci2.position;
    Vector3d vel_eci2 = state_eci2.velocity;

    // Should match original
    Vector3d pos_err = pos_eci - pos_eci2;
    Vector3d vel_err = vel_eci - vel_eci2;

    TEST_ASSERT(pos_err.norm() < 1.0);   // < 1m position error
    TEST_ASSERT(vel_err.norm() < 0.01);  // < 0.01 m/s velocity error
}

// Test: Geodetic-ECEF round-trip consistency
void test_geodetic_ecef_roundtrip() {
    // Test various locations
    std::vector<Vector3d> test_points = {
        coords::geodetic_to_ecef(0.0, 0.0, 0.0),           // Equator/Prime meridian
        coords::geodetic_to_ecef(45.0, 90.0, 10000.0),     // Mid-latitude
        coords::geodetic_to_ecef(89.9, 0.0, 1000.0),       // Near North pole
        coords::geodetic_to_ecef(-45.0, -180.0, 50000.0),  // Southern hemisphere
    };

    for (const auto& pos_ecef : test_points) {
        auto [lat, lon, alt] = coords::ecef_to_geodetic(pos_ecef);
        Vector3d pos_ecef2 = coords::geodetic_to_ecef(lat, lon, alt);

        Vector3d err = pos_ecef - pos_ecef2;
        TEST_ASSERT(err.norm() < 0.001);  // < 1mm error
    }
}

// Test: SP3 parser handles embedded test data
void test_sp3_parser_basic() {
    // Create minimal SP3-like content
    std::string sp3_content = R"(#cP2024  1 15  0  0  0.00000000      97 ORBIT IGS14 HLM  IGS
## 2295 432000.00000000   900.00000000 60323 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* Example SP3 file for testing
/* This is not real data
/* Created for unit testing
/* End of comments
*  2024  1 15  0  0  0.00000000
PG01  15600.000000   6778.137000      0.000000    999.999999
*  2024  1 15  0 15  0.00000000
PG01  15550.000000   6800.000000    100.000000    999.999999
EOF
)";

    auto sp3 = ballistic::test::parse_sp3_truth(sp3_content);   // throws if malformed
    TEST_ASSERT(sp3.records.size() == 2);

    // Check first record
    TEST_ASSERT(sp3.records[0].satellite_id == "G01");
    TEST_ASSERT_NEAR(sp3.records[0].position.x(), 15600000.0, 1.0);
    TEST_ASSERT_NEAR(sp3.records[0].position.y(), 6778137.0, 1.0);
}

// Test: SP3 interpolation
void test_sp3_interpolation() {
    // An empty file returns no interpolated position (nullopt), not a value
    ballistic::test::TruthFile empty_file;

    auto result = empty_file.interpolate_position("G01", Epoch::j2000());
    TEST_ASSERT(!result.has_value());  // Should return nullopt for empty file
}

// Test: an ISS-like orbit (400 km, 51.6 deg) stays between 350 and 450 km
// altitude after ~one orbit. Synthetic; no TLE is involved.
void test_iss_reference_orbit() {
    // ISS-like orbit using our orbit generator for consistency
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // Create ISS-like orbit (400 km, 51.6 inclination)
    auto orbit = create_circular_leo(400.0, 51.6, 30.0, 45.0, t0, "ISS-like");

    // Arguments are (beta, mass, area); the model is not read for drag here
    // (no DragEnvironment), so these values have no effect.
    BallisticModel model(420000.0, 2500.0, 2.2);
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = create_rk45_propagator(t0, orbit.position_ecef, orbit.velocity_ecef,
                                        model, weather, true);

    // Propagate 1 orbit (~93 minutes for ISS)
    double period = 93.0 * 60.0;
    Epoch t1 = t0 + period;
    auto result = (*prop)(t1);

    // Verify orbit didn't diverge
    double r1 = result.position.norm();
    double alt1 = r1 - constants::earth::RADIUS_EQUATORIAL;

    // ISS altitude should remain in reasonable LEO range
    TEST_ASSERT(alt1 > 350000.0);  // Above 350 km
    TEST_ASSERT(alt1 < 450000.0);  // Below 450 km
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    // Runs without an Earth orientation parameter (EOP) table on purpose:
    // synthetic, self-consistent checks with UT1-UTC and polar motion zero.
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    TestRunner kepler("Kepler Two-Body Tests");
    kepler.add_test("kepler_energy_conservation", test_kepler_energy_conservation);
    kepler.add_test("kepler_period_accuracy", test_kepler_period_accuracy);
    kepler.run();

    TestRunner j2("J2 Perturbation Tests");
    j2.add_test("j2_raan_drift", test_j2_raan_drift);
    j2.add_test("sun_synchronous_orbit", test_sun_synchronous_orbit);
    j2.run();

    TestRunner accuracy("Propagation Accuracy Tests");
    accuracy.add_test("short_term_accuracy", test_short_term_accuracy);
    accuracy.add_test("one_hour_stability", test_one_hour_stability);
    accuracy.add_test("drag_altitude_decay", test_drag_altitude_decay);
    accuracy.run();

    TestRunner coords_tests("Coordinate Transform Tests");
    coords_tests.add_test("eci_ecef_roundtrip", test_eci_ecef_roundtrip);
    coords_tests.add_test("geodetic_ecef_roundtrip", test_geodetic_ecef_roundtrip);
    coords_tests.run();

    TestRunner sp3("SP3 Parser Tests");
    sp3.add_test("sp3_parser_basic", test_sp3_parser_basic);
    sp3.add_test("sp3_interpolation", test_sp3_interpolation);
    sp3.run();

    TestRunner reference("Reference Orbit Tests");
    reference.add_test("iss_reference_orbit", test_iss_reference_orbit);
    reference.run();

    return print_final_summary();
}
