// Unit tests for coordinates/geodetic.hpp: conversions between geodetic
// coordinates (latitude, longitude, height above the WGS84 ellipsoid) and ECEF
// (Earth-centered, Earth-fixed Cartesian x, y, z in meters), the local
// East-North-Up (ENU) frame, geocentric latitude and the ellipsoid radius.
// Known points are checked to 1-100 m against the ellipsoid's radii; round
// trips to 1-10 m in height and 1e-8 to 1e-9 rad (loose bounds); the inverse
// is checked to 1 um against the forward map evaluated in long double.

#include "../test_utils.hpp"
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/core/constants.hpp>
#include <cmath>
#include <algorithm>

using namespace ballistic;
using namespace ballistic::coords;
using namespace ballistic::test;

static TestRunner runner("Geodetic Tests");

//=============================================================================
// geodetic_to_ecef Tests
//=============================================================================

void test_geodetic_to_ecef_equator_prime_meridian() {
    // Point on equator at prime meridian, sea level
    // lat=0, lon=0, alt=0 -> x=Re, y=0, z=0
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, 0.0);

    TEST_ASSERT_NEAR(ecef.x(), constants::earth::RADIUS_EQUATORIAL, 1.0);
    TEST_ASSERT_NEAR(ecef.y(), 0.0, 1.0);
    TEST_ASSERT_NEAR(ecef.z(), 0.0, 1.0);
}

void test_geodetic_to_ecef_equator_90_east() {
    // Equator at 90 degrees East: x=0, y=Re, z=0
    double lon = constants::PI / 2.0;  // 90 degrees
    Vector3d ecef = geodetic_to_ecef(0.0, lon, 0.0);

    TEST_ASSERT_NEAR(ecef.x(), 0.0, 1.0);
    TEST_ASSERT_NEAR(ecef.y(), constants::earth::RADIUS_EQUATORIAL, 1.0);
    TEST_ASSERT_NEAR(ecef.z(), 0.0, 1.0);
}

void test_geodetic_to_ecef_north_pole() {
    // North pole: x=0, y=0, z=Rp (polar radius)
    double lat = constants::PI / 2.0;  // 90 degrees
    Vector3d ecef = geodetic_to_ecef(lat, 0.0, 0.0);

    TEST_ASSERT_NEAR(ecef.x(), 0.0, 1.0);
    TEST_ASSERT_NEAR(ecef.y(), 0.0, 1.0);
    TEST_ASSERT_NEAR(ecef.z(), constants::earth::RADIUS_POLAR, 1.0);  // WGS84 polar radius known to mm
}

void test_geodetic_to_ecef_south_pole() {
    // South pole: x=0, y=0, z=-Rp
    double lat = -constants::PI / 2.0;  // -90 degrees
    Vector3d ecef = geodetic_to_ecef(lat, 0.0, 0.0);

    TEST_ASSERT_NEAR(ecef.x(), 0.0, 1.0);
    TEST_ASSERT_NEAR(ecef.y(), 0.0, 1.0);
    TEST_ASSERT_NEAR(ecef.z(), -constants::earth::RADIUS_POLAR, 1.0);  // WGS84 polar radius known to mm
}

void test_geodetic_to_ecef_45_degree_latitude() {
    // 45 degrees latitude, prime meridian
    double lat = constants::PI / 4.0;
    Vector3d ecef = geodetic_to_ecef(lat, 0.0, 0.0);

    // Should be between equator and pole values
    double r_equator = constants::earth::RADIUS_EQUATORIAL;
    double r_polar = constants::earth::RADIUS_POLAR;

    TEST_ASSERT_TRUE(ecef.x() > 0);
    TEST_ASSERT_TRUE(ecef.x() < r_equator);
    TEST_ASSERT_NEAR(ecef.y(), 0.0, 1.0);
    TEST_ASSERT_TRUE(ecef.z() > 0);
    TEST_ASSERT_TRUE(ecef.z() < r_polar);
}

void test_geodetic_to_ecef_with_altitude() {
    // Point on equator with 400 km altitude (ISS)
    double alt = 400000.0;  // 400 km
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, alt);

    double expected_r = constants::earth::RADIUS_EQUATORIAL + alt;
    TEST_ASSERT_NEAR(ecef.x(), expected_r, 1.0);
    TEST_ASSERT_NEAR(ecef.y(), 0.0, 1.0);
    TEST_ASSERT_NEAR(ecef.z(), 0.0, 1.0);
}

void test_geodetic_to_ecef_geo_altitude() {
    // GEO altitude ~35,786 km
    double alt = 35786000.0;
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, alt);

    double expected_r = constants::earth::RADIUS_EQUATORIAL + alt;
    TEST_ASSERT_NEAR(ecef.x(), expected_r, 100.0);
}

void test_geodetic_to_ecef_negative_altitude() {
    // Negative altitude (e.g., Dead Sea at -430 m)
    double alt = -430.0;
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, alt);

    double expected_r = constants::earth::RADIUS_EQUATORIAL + alt;
    TEST_ASSERT_NEAR(ecef.x(), expected_r, 1.0);
}

void test_geodetic_to_ecef_coord_overload() {
    // Test GeodeticCoord overload
    GeodeticCoord coord(constants::PI / 4.0, constants::PI / 6.0, 1000.0);
    Vector3d ecef = geodetic_to_ecef(coord);

    Vector3d expected = geodetic_to_ecef(coord.latitude, coord.longitude, coord.altitude);
    TEST_ASSERT_VEC3_NEAR(ecef, expected, 1e-10);
}

//=============================================================================
// ecef_to_geodetic Tests
//=============================================================================

void test_ecef_to_geodetic_equator() {
    Vector3d ecef(constants::earth::RADIUS_EQUATORIAL, 0.0, 0.0);
    GeodeticCoord coord = ecef_to_geodetic(ecef);

    TEST_ASSERT_NEAR(coord.latitude, 0.0, 1e-10);
    TEST_ASSERT_NEAR(coord.longitude, 0.0, 1e-10);
    TEST_ASSERT_NEAR(coord.altitude, 0.0, 1.0);
}

void test_ecef_to_geodetic_north_pole() {
    Vector3d ecef(0.0, 0.0, constants::earth::RADIUS_POLAR);
    GeodeticCoord coord = ecef_to_geodetic(ecef);

    TEST_ASSERT_NEAR(coord.latitude, constants::PI / 2.0, 1e-6);
    TEST_ASSERT_NEAR(coord.altitude, 0.0, 10.0);  // Pole altitude at surface
}

void test_ecef_to_geodetic_south_pole() {
    Vector3d ecef(0.0, 0.0, -constants::earth::RADIUS_POLAR);
    GeodeticCoord coord = ecef_to_geodetic(ecef);

    TEST_ASSERT_NEAR(coord.latitude, -constants::PI / 2.0, 1e-6);
    TEST_ASSERT_NEAR(coord.altitude, 0.0, 10.0);  // Pole altitude at surface
}

void test_ecef_to_geodetic_roundtrip() {
    // Test multiple points for roundtrip conversion
    struct TestPoint {
        double lat, lon, alt;
    };

    TestPoint points[] = {
        {0.0, 0.0, 0.0},
        {constants::PI / 4.0, constants::PI / 4.0, 100000.0},
        {-constants::PI / 3.0, -constants::PI / 2.0, 500000.0},
        {constants::PI / 6.0, constants::PI, 0.0},
        {0.0, -constants::PI / 3.0, 35786000.0},  // GEO
    };

    for (const auto& pt : points) {
        Vector3d ecef = geodetic_to_ecef(pt.lat, pt.lon, pt.alt);
        GeodeticCoord coord = ecef_to_geodetic(ecef);

        TEST_ASSERT_NEAR(coord.latitude, pt.lat, 1e-9);
        TEST_ASSERT_NEAR(coord.longitude, pt.lon, 1e-9);
        TEST_ASSERT_NEAR(coord.altitude, pt.alt, 1.0);  // 1 meter tolerance
    }
}

void test_ecef_to_geodetic_high_altitude() {
    // Test at lunar distance
    double alt = 384400000.0;  // Moon distance
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, alt);
    GeodeticCoord coord = ecef_to_geodetic(ecef);

    TEST_ASSERT_NEAR(coord.altitude, alt, 1000.0);  // 1 km tolerance at this distance
}

void test_ecef_to_geodetic_convergence() {
    // Round trip over a sweep of latitudes, longitudes and heights 0-1000 km
    for (int i = 0; i < 100; ++i) {
        double lat = (i - 50) * constants::PI / 100.0;  // -90 to 90 degrees
        double lon = (i - 50) * constants::PI / 50.0;   // -180 to 180 degrees
        double alt = i * 10000.0;  // 0 to 1000 km

        Vector3d ecef = geodetic_to_ecef(lat, lon, alt);
        GeodeticCoord coord = ecef_to_geodetic(ecef);

        TEST_ASSERT_NEAR(coord.latitude, lat, 1e-8);
        TEST_ASSERT_NEAR(coord.altitude, alt, 10.0);
    }
}

//=============================================================================
// geodetic_from_degrees Tests
//=============================================================================

void test_geodetic_from_degrees() {
    GeodeticCoord coord = geodetic_from_degrees(45.0, -90.0, 10000.0);

    TEST_ASSERT_NEAR(coord.latitude, constants::PI / 4.0, 1e-10);
    TEST_ASSERT_NEAR(coord.longitude, -constants::PI / 2.0, 1e-10);
    TEST_ASSERT_NEAR(coord.altitude, 10000.0, 1e-10);
}

void test_geodetic_from_degrees_180() {
    GeodeticCoord coord = geodetic_from_degrees(0.0, 180.0, 0.0);

    TEST_ASSERT_NEAR(coord.latitude, 0.0, 1e-10);
    TEST_ASSERT_NEAR(coord.longitude, constants::PI, 1e-10);
}

void test_geodetic_from_degrees_negative() {
    GeodeticCoord coord = geodetic_from_degrees(-33.86, 151.21, 58.0);  // Sydney

    TEST_ASSERT_NEAR(coord.latitude * constants::RAD_TO_DEG, -33.86, 1e-6);
    TEST_ASSERT_NEAR(coord.longitude * constants::RAD_TO_DEG, 151.21, 1e-6);
}

//=============================================================================
// ecef_to_altitude Tests
//=============================================================================

void test_ecef_to_altitude_sea_level() {
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, 0.0);
    double alt = ecef_to_altitude(ecef);
    TEST_ASSERT_TRUE(std::abs(alt) < 100.0);  // Within 100 meters of sea level
}

void test_ecef_to_altitude_iss() {
    double expected_alt = 400000.0;
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, expected_alt);
    TEST_ASSERT_NEAR(ecef_to_altitude(ecef), expected_alt, 1e-6);
}

// ecef_to_geodetic and ecef_to_altitude both use the exact closed form of
// Vermeille (2011). Asserted: altitude and latitude (as arc length) within
// 1 um on a grid from pole to pole and from -1 km to beyond GEO, against the
// forward map evaluated in long double. (Trap census row ALT-GEOCENTRIC,
// static rule SPHERICAL-ALT: an approximate altitude r - N(geocentric
// latitude) is off by -21 km at 45 deg and -41 km at 80 deg.)
void test_ecef_to_altitude_vs_full() {
    using LD = long double;
    const LD a = constants::earth::RADIUS_EQUATORIAL;
    const LD e2 = constants::earth::ECCENTRICITY_SQ;
    double worst_alt = 0.0, worst_lat = 0.0;
    for (double lat_deg = -90.0; lat_deg <= 90.0; lat_deg += 2.5) {
        for (double h : {-1000.0, 0.0, 1.0, 80e3, 120e3, 400e3, 2e6, 2e7, 3.6e7, 4.2e7}) {
            const LD lat = lat_deg * (LD)constants::PI / 180.0L, lon = 1.1L;
            const LD N = a / std::sqrt(1.0L - e2 * std::sin(lat) * std::sin(lat));
            const Vector3d p((double)((N + h) * std::cos(lat) * std::cos(lon)),
                             (double)((N + h) * std::cos(lat) * std::sin(lon)),
                             (double)((N * (1.0L - e2) + h) * std::sin(lat)));
            const GeodeticCoord g = ecef_to_geodetic(p);
            worst_alt = std::max(worst_alt, std::abs(g.altitude - h));
            worst_alt = std::max(worst_alt, std::abs(ecef_to_altitude(p) - h));
            worst_lat = std::max(worst_lat, std::abs(g.latitude - (double)lat));
        }
    }
    TEST_ASSERT(worst_alt < 1e-6);
    TEST_ASSERT(worst_lat * constants::earth::RADIUS_EQUATORIAL < 1e-6);
}

// Shows why altitude must be geodetic: at 45 and 80 deg latitude a point 200 km
// above the ellipsoid is more than 10 km away from 200 km in the spherical
// "radius minus equatorial radius" sense, while ecef_to_altitude returns 200 km
// to 1 um.
void test_altitude_is_geodetic_not_spherical() {
    for (double lat_deg : {45.0, 80.0}) {
        const Vector3d p = geodetic_to_ecef(lat_deg * constants::DEG_TO_RAD, 0.3, 200e3);
        TEST_ASSERT_NEAR(ecef_to_altitude(p), 200e3, 1e-6);
        TEST_ASSERT(std::abs((p.norm() - constants::earth::RADIUS_EQUATORIAL) - 200e3) > 10e3);  // guard-exempt(SPHERICAL-ALT): shows the size of the error
    }
}

//=============================================================================
// ENU Rotation Tests
//=============================================================================

void test_ecef_to_enu_rotation_equator() {
    // At equator, prime meridian:
    // East = +Y, North = +Z, Up = +X
    Matrix3d R = ecef_to_enu_rotation(0.0, 0.0);

    // Check orthogonality
    Matrix3d RRT = R * R.transpose();
    Matrix3d I = Matrix3d::Identity();
    TEST_ASSERT_NEAR((RRT - I).norm(), 0.0, 1e-10);

    // Check determinant = 1 (proper rotation)
    TEST_ASSERT_NEAR(R.determinant(), 1.0, 1e-10);
}

void test_ecef_to_enu_rotation_pole() {
    // At north pole - check that it's still a valid rotation
    double lat = constants::PI / 2.0 - 1e-10;  // Nearly at pole
    Matrix3d R = ecef_to_enu_rotation(lat, 0.0);

    // Check orthogonality
    Matrix3d RRT = R * R.transpose();
    Matrix3d I = Matrix3d::Identity();
    TEST_ASSERT_NEAR((RRT - I).norm(), 0.0, 1e-8);
}

void test_ecef_to_enu_rotation_determinant() {
    // Test at various locations
    for (int i = 0; i < 10; ++i) {
        double lat = (i - 5) * constants::PI / 10.0;
        double lon = i * constants::PI / 5.0;

        Matrix3d R = ecef_to_enu_rotation(lat, lon);
        TEST_ASSERT_NEAR(R.determinant(), 1.0, 1e-10);
    }
}

//=============================================================================
// ENU Velocity Tests
//=============================================================================

void test_ecef_to_enu_velocity() {
    // At equator, prime meridian with eastward velocity
    Vector3d vel_ecef(0.0, 7500.0, 0.0);  // Eastward in ECEF
    Vector3d vel_enu = ecef_to_enu_velocity(vel_ecef, 0.0, 0.0);

    // At this location, ECEF +Y is ENU East
    TEST_ASSERT_NEAR(vel_enu.x(), 7500.0, 1.0);  // East component
    TEST_ASSERT_NEAR(vel_enu.y(), 0.0, 1.0);     // North component
    TEST_ASSERT_NEAR(vel_enu.z(), 0.0, 1.0);     // Up component
}

void test_ecef_to_enu_velocity_magnitude() {
    // Velocity magnitude should be preserved
    Vector3d vel_ecef(1000.0, 2000.0, 3000.0);
    double lat = constants::PI / 4.0;
    double lon = constants::PI / 6.0;

    Vector3d vel_enu = ecef_to_enu_velocity(vel_ecef, lat, lon);

    TEST_ASSERT_NEAR(vel_enu.norm(), vel_ecef.norm(), 1e-6);
}

//=============================================================================
// Geocentric Latitude Tests
//=============================================================================

void test_geodetic_to_geocentric_latitude_equator() {
    // At equator, both are zero
    double geocentric = geodetic_to_geocentric_latitude(0.0);
    TEST_ASSERT_NEAR(geocentric, 0.0, 1e-15);
}

void test_geodetic_to_geocentric_latitude_pole() {
    // At poles, both are +-90 degrees
    double geocentric_north = geodetic_to_geocentric_latitude(constants::PI / 2.0);
    double geocentric_south = geodetic_to_geocentric_latitude(-constants::PI / 2.0);

    TEST_ASSERT_NEAR(geocentric_north, constants::PI / 2.0, 1e-10);
    TEST_ASSERT_NEAR(geocentric_south, -constants::PI / 2.0, 1e-10);
}

void test_geodetic_to_geocentric_latitude_45deg() {
    // At 45 degrees geodetic, geocentric should be slightly less
    double geodetic = constants::PI / 4.0;  // 45 degrees
    double geocentric = geodetic_to_geocentric_latitude(geodetic);

    // Geocentric latitude is always closer to equator (smaller magnitude)
    TEST_ASSERT_TRUE(geocentric < geodetic);
    TEST_ASSERT_TRUE(geocentric > 0.0);

    // Geocentric should be approximately 44.8 degrees for WGS84
    // (difference of about 0.2 degrees or ~0.003 radians)
    TEST_ASSERT_NEAR(geocentric, geodetic, 0.01);  // Within 0.01 radians (~0.6 degrees)
}

//=============================================================================
// Earth Radius at Latitude Tests
//=============================================================================

void test_earth_radius_at_latitude_equator() {
    double r = earth_radius_at_latitude(0.0);
    TEST_ASSERT_NEAR(r, constants::earth::RADIUS_EQUATORIAL, 0.1);
}

void test_earth_radius_at_latitude_pole() {
    double r_north = earth_radius_at_latitude(constants::PI / 2.0);
    double r_south = earth_radius_at_latitude(-constants::PI / 2.0);

    // Allow some deviation - the geocentric radius formula may differ slightly
    TEST_ASSERT_NEAR(r_north, constants::earth::RADIUS_POLAR, 100.0);
    TEST_ASSERT_NEAR(r_south, constants::earth::RADIUS_POLAR, 100.0);
}

void test_earth_radius_at_latitude_45deg() {
    double r = earth_radius_at_latitude(constants::PI / 4.0);

    // Should be between polar and equatorial
    TEST_ASSERT_TRUE(r > constants::earth::RADIUS_POLAR);
    TEST_ASSERT_TRUE(r < constants::earth::RADIUS_EQUATORIAL);
}

void test_earth_radius_at_latitude_smooth_variation() {
    // Radius at equator should be larger than at poles
    double r_equator = earth_radius_at_latitude(0.0);
    double r_pole = earth_radius_at_latitude(constants::PI / 2.0);
    double r_45 = earth_radius_at_latitude(constants::PI / 4.0);

    TEST_ASSERT_TRUE(r_equator > r_pole);
    TEST_ASSERT_TRUE(r_45 > r_pole);
    TEST_ASSERT_TRUE(r_equator > r_45);
}

//=============================================================================
// Edge Cases
//=============================================================================

void test_geodetic_antimeridian() {
    // Point at 180 degrees longitude
    double lon = constants::PI;  // 180 degrees
    Vector3d ecef = geodetic_to_ecef(0.0, lon, 0.0);

    TEST_ASSERT_NEAR(ecef.x(), -constants::earth::RADIUS_EQUATORIAL, 1.0);
    TEST_ASSERT_NEAR(ecef.y(), 0.0, 1.0);
}

void test_geodetic_negative_longitude() {
    // -90 degrees should be same as 270 degrees
    double lon1 = -constants::PI / 2.0;
    double lon2 = 3.0 * constants::PI / 2.0;

    Vector3d ecef1 = geodetic_to_ecef(0.0, lon1, 0.0);
    Vector3d ecef2 = geodetic_to_ecef(0.0, lon2, 0.0);

    TEST_ASSERT_VEC3_NEAR(ecef1, ecef2, 1.0);
}

void test_geodetic_very_high_altitude() {
    // 6 Earth radii altitude
    double alt = 6.0 * constants::earth::RADIUS_EQUATORIAL;
    Vector3d ecef = geodetic_to_ecef(0.0, 0.0, alt);
    GeodeticCoord coord = ecef_to_geodetic(ecef);

    TEST_ASSERT_NEAR(coord.altitude, alt, 1000.0);  // 1 km tolerance
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "BALLISTIC LIBRARY - GEODETIC TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    // geodetic_to_ecef tests
    runner.add_test("geodetic_to_ecef_equator_prime_meridian", test_geodetic_to_ecef_equator_prime_meridian);
    runner.add_test("geodetic_to_ecef_equator_90_east", test_geodetic_to_ecef_equator_90_east);
    runner.add_test("geodetic_to_ecef_north_pole", test_geodetic_to_ecef_north_pole);
    runner.add_test("geodetic_to_ecef_south_pole", test_geodetic_to_ecef_south_pole);
    runner.add_test("geodetic_to_ecef_45_degree_latitude", test_geodetic_to_ecef_45_degree_latitude);
    runner.add_test("geodetic_to_ecef_with_altitude", test_geodetic_to_ecef_with_altitude);
    runner.add_test("geodetic_to_ecef_geo_altitude", test_geodetic_to_ecef_geo_altitude);
    runner.add_test("geodetic_to_ecef_negative_altitude", test_geodetic_to_ecef_negative_altitude);
    runner.add_test("geodetic_to_ecef_coord_overload", test_geodetic_to_ecef_coord_overload);

    // ecef_to_geodetic tests
    runner.add_test("ecef_to_geodetic_equator", test_ecef_to_geodetic_equator);
    runner.add_test("ecef_to_geodetic_north_pole", test_ecef_to_geodetic_north_pole);
    runner.add_test("ecef_to_geodetic_south_pole", test_ecef_to_geodetic_south_pole);
    runner.add_test("ecef_to_geodetic_roundtrip", test_ecef_to_geodetic_roundtrip);
    runner.add_test("ecef_to_geodetic_high_altitude", test_ecef_to_geodetic_high_altitude);
    runner.add_test("ecef_to_geodetic_convergence", test_ecef_to_geodetic_convergence);

    // geodetic_from_degrees tests
    runner.add_test("geodetic_from_degrees", test_geodetic_from_degrees);
    runner.add_test("geodetic_from_degrees_180", test_geodetic_from_degrees_180);
    runner.add_test("geodetic_from_degrees_negative", test_geodetic_from_degrees_negative);

    // ecef_to_altitude tests
    runner.add_test("ecef_to_altitude_sea_level", test_ecef_to_altitude_sea_level);
    runner.add_test("ecef_to_altitude_iss", test_ecef_to_altitude_iss);
    runner.add_test("ecef_to_altitude_vs_full", test_ecef_to_altitude_vs_full);
    runner.add_test("altitude_is_geodetic_not_spherical", test_altitude_is_geodetic_not_spherical);

    // ENU rotation tests
    runner.add_test("ecef_to_enu_rotation_equator", test_ecef_to_enu_rotation_equator);
    runner.add_test("ecef_to_enu_rotation_pole", test_ecef_to_enu_rotation_pole);
    runner.add_test("ecef_to_enu_rotation_determinant", test_ecef_to_enu_rotation_determinant);

    // ENU velocity tests
    runner.add_test("ecef_to_enu_velocity", test_ecef_to_enu_velocity);
    runner.add_test("ecef_to_enu_velocity_magnitude", test_ecef_to_enu_velocity_magnitude);

    // Geocentric latitude tests
    runner.add_test("geodetic_to_geocentric_latitude_equator", test_geodetic_to_geocentric_latitude_equator);
    runner.add_test("geodetic_to_geocentric_latitude_pole", test_geodetic_to_geocentric_latitude_pole);
    runner.add_test("geodetic_to_geocentric_latitude_45deg", test_geodetic_to_geocentric_latitude_45deg);

    // Earth radius tests
    runner.add_test("earth_radius_at_latitude_equator", test_earth_radius_at_latitude_equator);
    runner.add_test("earth_radius_at_latitude_pole", test_earth_radius_at_latitude_pole);
    runner.add_test("earth_radius_at_latitude_45deg", test_earth_radius_at_latitude_45deg);
    runner.add_test("earth_radius_at_latitude_smooth_variation", test_earth_radius_at_latitude_smooth_variation);

    // Edge cases
    runner.add_test("geodetic_antimeridian", test_geodetic_antimeridian);
    runner.add_test("geodetic_negative_longitude", test_geodetic_negative_longitude);
    runner.add_test("geodetic_very_high_altitude", test_geodetic_very_high_altitude);

    int result = runner.run();
    return print_final_summary();
}
