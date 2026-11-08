// =============================================================================
// Geodetic Function External Validation Tests
// =============================================================================
//
// Checks the conversions between WGS84 geodetic coordinates (latitude,
// longitude, height above the ellipsoid) and ECEF (Earth-centered, Earth-fixed
// Cartesian x, y, z) against values computed independently with GeographicLib
// (C. Karney), rather than only against the library's own inverse.
//
// Reference points: the equator, poles and 45 deg at sea level, orbital heights
// (ISS, 400 km, GPS, GEO) and two observatories (Paris, Greenwich; approximate
// coordinates, used only as test inputs). Also: round trips from 11 km below sea
// level to lunar distance, latitudes near the poles, and the WGS84 constants
// themselves (NIMA TR8350.2).
//
// Tolerances: 1 mm per point (1 cm for the points at GPS/GEO height and on the
// equator at 400 km), and 1 mm for the round trips.
//
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <vector>
#include <string>

using namespace ballistic;
using namespace ballistic::test;

// =============================================================================
// Reference Data: Known Geodetic/ECEF Points
// =============================================================================

struct GeodeticTestPoint {
    std::string name;
    double lat_deg;     // Geodetic latitude (degrees)
    double lon_deg;     // Longitude (degrees)
    double alt_m;       // Height above ellipsoid (meters)
    double x_m;         // ECEF X (meters)
    double y_m;         // ECEF Y (meters)
    double z_m;         // ECEF Z (meters)
    double tolerance_m; // Expected accuracy
};

// Reference points computed using GeographicLib (verified)
// These serve as ground truth for our implementation
const std::vector<GeodeticTestPoint> REFERENCE_POINTS = {
    // Equator on prime meridian at sea level
    {"Equator-PM-0m", 0.0, 0.0, 0.0,
     6378137.0, 0.0, 0.0, 0.001},

    // North pole at sea level
    {"North Pole-0m", 90.0, 0.0, 0.0,
     0.0, 0.0, 6356752.3142, 0.001},

    // South pole at sea level
    {"South Pole-0m", -90.0, 0.0, 0.0,
     0.0, 0.0, -6356752.3142, 0.001},

    // 45 degrees on prime meridian at sea level
    {"45N-PM-0m", 45.0, 0.0, 0.0,
     4517590.8789, 0.0, 4487348.4087, 0.001},

    // GPS constellation altitude (20,200 km)
    {"GPS Altitude", 0.0, 0.0, 20200000.0,
     26578137.0, 0.0, 0.0, 0.01},

    // Geostationary altitude
    {"GEO Altitude", 0.0, 0.0, 35786000.0,
     42164137.0, 0.0, 0.0, 0.01},

    // ISS orbital altitude (typical)
    {"ISS Altitude", 51.6, 0.0, 408000.0,
     4223363.5819, 0.0, 5295027.8082, 0.001},

    // Various latitudes at 400 km altitude
    {"400km-Equator", 0.0, 0.0, 400000.0,
     6778137.0, 0.0, 0.0, 0.01},

    {"400km-45N", 45.0, 0.0, 400000.0,
     4800433.5913, 0.0, 4770191.1213, 0.001},

    {"400km-60N", 60.0, 0.0, 400000.0,
     3397104.5869, 0.0, 5846887.2955, 0.001},

    // International test points (approximate geodetic coordinates)
    {"Paris Observatory", 48.8361, 2.3366, 67.0,
     4202720.7831, 171487.7619, 4778631.5299, 0.001},

    {"Greenwich Observatory", 51.4769, -0.0005, 31.2,
     3980678.9830, -34.7380, 4966786.5671, 0.001},
};

// =============================================================================
// Test Functions
// =============================================================================

// Test: Geodetic to ECEF conversion accuracy
void test_geodetic_to_ecef() {
    std::cout << "\n";
    std::cout << "    Geodetic to ECEF Conversion:\n";
    std::cout << "    Point                 Error (m)    Tolerance (m)\n";
    std::cout << "    --------------------------------------------------\n";

    for (const auto& pt : REFERENCE_POINTS) {
        double lat_rad = pt.lat_deg * constants::DEG_TO_RAD;
        double lon_rad = pt.lon_deg * constants::DEG_TO_RAD;

        Vector3d ecef = coords::geodetic_to_ecef(lat_rad, lon_rad, pt.alt_m);

        Vector3d ref(pt.x_m, pt.y_m, pt.z_m);
        double error = (ecef - ref).norm();

        std::cout << "    " << std::left << std::setw(20) << pt.name
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(12) << error << "    " << std::setw(12) << pt.tolerance_m << "\n";

        TEST_ASSERT_MSG(error <= pt.tolerance_m,
            "Geodetic->ECEF error " + std::to_string(error) + "m exceeds tolerance "
            + std::to_string(pt.tolerance_m) + "m for " + pt.name);
    }

    std::cout << "    --------------------------------------------------\n";
    std::cout << "    All " << REFERENCE_POINTS.size() << " points within tolerance\n";
    std::cout << "    ";
}

// Test: ECEF to Geodetic conversion accuracy
void test_ecef_to_geodetic() {
    std::cout << "\n";
    std::cout << "    ECEF to Geodetic Conversion:\n";
    std::cout << "    Point                 Lat Err(deg)  Lon Err(deg)  Alt Err(m)\n";
    std::cout << "    ----------------------------------------------------------------\n";

    for (const auto& pt : REFERENCE_POINTS) {
        Vector3d ecef(pt.x_m, pt.y_m, pt.z_m);
        auto [lat_rad, lon_rad, alt] = coords::ecef_to_geodetic(ecef);

        double lat_deg = lat_rad * constants::RAD_TO_DEG;
        double lon_deg = lon_rad * constants::RAD_TO_DEG;

        double lat_err = std::abs(lat_deg - pt.lat_deg);
        double lon_err = std::abs(lon_deg - pt.lon_deg);
        double alt_err = std::abs(alt - pt.alt_m);

        // Handle pole singularity (longitude undefined at poles)
        if (std::abs(pt.lat_deg) > 89.9) {
            lon_err = 0.0;
        }

        bool pass = (lat_err < 1e-8) && (lon_err < 1e-8 || std::abs(pt.lat_deg) > 89.9) && (alt_err < 0.001);

        std::cout << "    " << std::left << std::setw(20) << pt.name
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(12) << lat_err
                  << std::setw(14) << lon_err
                  << std::setw(12) << alt_err << "\n";

        TEST_ASSERT_MSG(pass,
            "ECEF->Geodetic failed for " + pt.name +
            " (lat_err=" + std::to_string(lat_err) +
            ", lon_err=" + std::to_string(lon_err) +
            ", alt_err=" + std::to_string(alt_err) + ")");
    }

    std::cout << "    ----------------------------------------------------------------\n";
    std::cout << "    All " << REFERENCE_POINTS.size() << " points passed\n";
    std::cout << "    ";
}

// Test: Round-trip conversion accuracy
void test_round_trip_conversion() {
    std::cout << "\n";
    std::cout << "    Round-Trip Conversion (Geodetic -> ECEF -> Geodetic):\n";
    std::cout << "    Point                 Position Error (m)\n";
    std::cout << "    --------------------------------------------------\n";

    double max_error = 0.0;

    for (const auto& pt : REFERENCE_POINTS) {
        // Original geodetic
        double lat_rad = pt.lat_deg * constants::DEG_TO_RAD;
        double lon_rad = pt.lon_deg * constants::DEG_TO_RAD;
        double alt = pt.alt_m;

        // Convert to ECEF
        Vector3d ecef = coords::geodetic_to_ecef(lat_rad, lon_rad, alt);

        // Convert back to geodetic
        auto [lat2_rad, lon2_rad, alt2] = coords::ecef_to_geodetic(ecef);

        // Convert back to ECEF again
        Vector3d ecef2 = coords::geodetic_to_ecef(lat2_rad, lon2_rad, alt2);

        // Round-trip error
        double error = (ecef2 - ecef).norm();
        max_error = (std::max)(max_error, error);

        TEST_ASSERT_MSG(error < 1e-6,
            "Round-trip error " + std::to_string(error) + "m for " + pt.name);

        std::cout << "    " << std::left << std::setw(20) << pt.name
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(18) << error << "\n";
    }

    std::cout << "    --------------------------------------------------\n";
    std::cout << "    Max round-trip error: " << std::scientific << std::setprecision(3) << max_error << " m\n";
    std::cout << "    ";
}

// Test: Extreme altitude cases
void test_extreme_altitudes() {
    std::cout << "\n";
    std::cout << "    Extreme Altitude Cases:\n";
    std::cout << "    Altitude            Round-Trip Error (m)\n";
    std::cout << "    --------------------------------------------------\n";

    struct AltitudeTest {
        std::string name;
        double alt_m;
    };

    std::vector<AltitudeTest> altitudes = {
        {"Deep ocean (-11km)", -11000.0},
        {"Sea level (0 m)", 0.0},
        {"Mount Everest (9 km)", 8850.0},
        {"LEO (400 km)", 400000.0},
        {"MEO (20,200 km)", 20200000.0},
        {"GEO (35,786 km)", 35786000.0},
        {"Lunar distance (384,400 km)", 384400000.0}
    };

    for (const auto& test : altitudes) {
        double lat_rad = 45.0 * constants::DEG_TO_RAD;
        double lon_rad = 90.0 * constants::DEG_TO_RAD;

        Vector3d ecef = coords::geodetic_to_ecef(lat_rad, lon_rad, test.alt_m);
        auto [lat2_rad, lon2_rad, alt2] = coords::ecef_to_geodetic(ecef);
        Vector3d ecef2 = coords::geodetic_to_ecef(lat2_rad, lon2_rad, alt2);

        double error = (ecef2 - ecef).norm();

        TEST_ASSERT_MSG(error < 1e-3,
            "Extreme altitude error " + std::to_string(error) + "m for " + test.name);

        std::cout << "    " << std::left << std::setw(24) << test.name
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(18) << error << "\n";
    }

    std::cout << "    --------------------------------------------------\n";
    std::cout << "    ";
}

// Test: Latitude edge cases
void test_latitude_edge_cases() {
    std::cout << "\n";
    std::cout << "    Latitude Edge Cases:\n";
    std::cout << "    Latitude            Round-Trip Error (m)\n";
    std::cout << "    --------------------------------------------------\n";

    std::vector<double> latitudes = {
        -90.0, -89.99999, -89.0, -45.0, -1.0, 0.0, 1.0, 45.0, 89.0, 89.99999, 90.0
    };

    for (double lat_deg : latitudes) {
        double lat_rad = lat_deg * constants::DEG_TO_RAD;
        double lon_rad = 45.0 * constants::DEG_TO_RAD;
        double alt = 400000.0;  // 400 km

        Vector3d ecef = coords::geodetic_to_ecef(lat_rad, lon_rad, alt);
        auto [lat2_rad, lon2_rad, alt2] = coords::ecef_to_geodetic(ecef);
        Vector3d ecef2 = coords::geodetic_to_ecef(lat2_rad, lon2_rad, alt2);

        double error = (ecef2 - ecef).norm();

        TEST_ASSERT(error < 1e-3);

        std::cout << "    " << std::right << std::fixed << std::setprecision(5)
                  << std::setw(12) << lat_deg << " deg    "
                  << std::scientific << std::setprecision(3)
                  << std::setw(18) << error << "\n";
    }

    std::cout << "    --------------------------------------------------\n";
    std::cout << "    ";
}

// Test: WGS84 ellipsoid constants
void test_wgs84_constants() {
    std::cout << "\n";
    std::cout << "    WGS84 Ellipsoid Constants Verification:\n";

    // WGS84 defining parameters (from NIMA TR8350.2)
    const double WGS84_A = 6378137.0;              // Semi-major axis (m)
    const double WGS84_F = 1.0 / 298.257223563;    // Flattening
    const double WGS84_B = WGS84_A * (1.0 - WGS84_F);  // Semi-minor axis (m)
    const double WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;  // First eccentricity squared

    std::cout << "    Parameter              Expected          Actual            Diff\n";
    std::cout << "    -------------------------------------------------------------------\n";

    auto check_param = [](const std::string& name, double expected, double actual, double tol = 1e-10) {
        double diff = std::abs(expected - actual);
        std::cout << "    " << std::left << std::setw(20) << name
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(16) << expected
                  << std::setw(18) << actual
                  << std::scientific << std::setprecision(3)
                  << std::setw(14) << diff << "\n";
        return diff < tol;
    };

    bool pass = true;
    pass &= check_param("Semi-major axis (m)", WGS84_A, constants::earth::RADIUS_EQUATORIAL);
    pass &= check_param("Flattening", WGS84_F, constants::earth::FLATTENING);
    // Semi-minor axis truncated to 4 decimals in constants - use 0.001m tolerance
    pass &= check_param("Semi-minor axis (m)", WGS84_B, constants::earth::RADIUS_POLAR, 0.001);
    pass &= check_param("Eccentricity^2", WGS84_E2, constants::earth::ECCENTRICITY_SQ);

    std::cout << "    -------------------------------------------------------------------\n";
    std::cout << "    ";

    TEST_ASSERT(pass);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    TestRunner conversion("Geodetic Conversion Accuracy");
    conversion.add_test("geodetic_to_ecef", test_geodetic_to_ecef);
    conversion.add_test("ecef_to_geodetic", test_ecef_to_geodetic);
    conversion.add_test("round_trip_conversion", test_round_trip_conversion);
    conversion.run();

    TestRunner extreme("Extreme Cases");
    extreme.add_test("extreme_altitudes", test_extreme_altitudes);
    extreme.add_test("latitude_edge_cases", test_latitude_edge_cases);
    extreme.run();

    TestRunner wgs84("WGS84 Constants");
    wgs84.add_test("wgs84_constants", test_wgs84_constants);
    wgs84.run();

    return print_final_summary();
}
