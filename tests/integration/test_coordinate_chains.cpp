// Integration tests for coordinate transformation chains: geodetic (latitude,
// longitude, height) <-> ECEF (Earth-centered, Earth-fixed x, y, z), and GCRS
// (the inertial frame) <-> ITRF (the Earth-fixed frame) through the IAU
// 2006/2000A model. Round trips must return positions to within 1 micrometer
// (velocities within 1 mm/s; the transforms are exact inverses up to rounding);
// the rotation matrices must be orthogonal to 1e-14 with determinant 1 (to
// 1e-10). EOP = Earth orientation parameters
// (polar motion xp, yp and UT1-UTC).

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/coordinates/iau_transforms.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::coords;
using namespace ballistic::constants;

// =============================================================================
// Geodetic Roundtrip
// =============================================================================

void test_geodetic_roundtrip() {
    // Original geodetic coordinates
    double lat = 37.7749 * DEG_TO_RAD;  // San Francisco
    double lon = -122.4194 * DEG_TO_RAD;
    double alt = 100.0;

    // Convert to ECEF
    Vector3d pos_ecef = geodetic_to_ecef(lat, lon, alt);

    // Convert back to geodetic
    auto [lat_out, lon_out, alt_out] = ecef_to_geodetic(pos_ecef);

    // Check roundtrip accuracy
    TEST_ASSERT_NEAR(lat_out, lat, 1e-10);
    TEST_ASSERT_NEAR(lon_out, lon, 1e-10);
    TEST_ASSERT_NEAR(alt_out, alt, 1e-6);
}

// =============================================================================
// Full IAU Transform Chain Tests (GCRS <-> ITRF)
// =============================================================================

void test_gcrs_itrf_roundtrip_position() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_gcrs(6778137.0, 0.0, 0.0);

    // GCRS -> ITRF -> GCRS
    Vector3d pos_itrf = gcrs_to_itrf_full(pos_gcrs, epoch);
    Vector3d pos_gcrs_back = itrf_to_gcrs_full(pos_itrf, epoch);

    double error = (pos_gcrs_back - pos_gcrs).norm();
    TEST_ASSERT(error < 1e-6);
}

void test_gcrs_itrf_roundtrip_state() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_gcrs(6778137.0, 0.0, 0.0);
    Vector3d vel_gcrs(0.0, 7500.0, 0.0);
    StateVector state_gcrs(pos_gcrs, vel_gcrs);

    // Roundtrip
    StateVector state_itrf = gcrs_to_itrf_full(state_gcrs, epoch);
    StateVector state_gcrs_back = itrf_to_gcrs_full(state_itrf, epoch);

    double pos_error = (state_gcrs_back.position - pos_gcrs).norm();
    double vel_error = (state_gcrs_back.velocity - vel_gcrs).norm();

    TEST_ASSERT(pos_error < 1e-6);
    TEST_ASSERT(vel_error < 1e-3);
}

void test_gcrs_itrf_with_eop() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_gcrs(6778137.0, 0.0, 0.0);

    // With zero EOP
    EarthOrientationParameters eop_zero = EarthOrientationParameters::zero();
    IAUTransform transform_zero(eop_zero);
    Vector3d pos_itrf_zero = transform_zero.gcrs_to_itrf(pos_gcrs, epoch);

    // With non-zero EOP (typical values)
    EarthOrientationParameters eop;
    eop.xp = 0.1;  // arcseconds
    eop.yp = 0.3;  // arcseconds
    eop.dut1 = 0.1;  // seconds
    IAUTransform transform(eop);
    Vector3d pos_itrf = transform.gcrs_to_itrf(pos_gcrs, epoch);

    // Positions should differ: 0.1 s of UT1-UTC rotates the Earth by ~50 m at
    // this radius and 0.3" of polar motion moves it by ~10 m, so the change
    // must be well above 1 cm and below 100 m.
    double diff = (pos_itrf - pos_itrf_zero).norm();
    TEST_ASSERT(diff > 0.01);
    TEST_ASSERT(diff < 100.0);
}

// =============================================================================
// Polar and Equatorial Geodetic Tests
// =============================================================================

void test_geodetic_at_poles() {
    // North pole geodetic
    double lat_north = PI / 2;
    double lon = 0.0;
    double alt = 0.0;

    Vector3d pos_north = geodetic_to_ecef(lat_north, lon, alt);
    TEST_ASSERT(std::abs(pos_north.x()) < 1e-6);
    TEST_ASSERT(std::abs(pos_north.y()) < 1e-6);
    TEST_ASSERT(pos_north.z() > earth::RADIUS_POLAR - 1.0);

    // South pole
    double lat_south = -PI / 2;
    Vector3d pos_south = geodetic_to_ecef(lat_south, lon, alt);
    TEST_ASSERT(pos_south.z() < -(earth::RADIUS_POLAR - 1.0));
}

void test_geodetic_at_equator() {
    double lat = 0.0;  // Equator
    double lon = 0.0;
    double alt = 0.0;

    Vector3d pos = geodetic_to_ecef(lat, lon, alt);

    // Should be at equatorial radius on x-axis
    TEST_ASSERT(std::abs(pos.x() - earth::RADIUS_EQUATORIAL) < 1.0);
    TEST_ASSERT(std::abs(pos.y()) < 1e-6);
    TEST_ASSERT(std::abs(pos.z()) < 1e-6);
}

// =============================================================================
// Matrix Properties Tests
// =============================================================================

void test_gcrs_to_itrf_matrix_orthogonal() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    IAUTransform transform;
    Matrix3d R = transform.gcrs_to_itrf_matrix(epoch);

    // Check orthogonality: R * R^T = I
    Matrix3d I = R * R.transpose();

    double off_diag_max = 0.0;
    double diag_error_max = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) {
                diag_error_max = std::max(diag_error_max, std::abs(I(i, j) - 1.0));
            } else {
                off_diag_max = std::max(off_diag_max, std::abs(I(i, j)));
            }
        }
    }

    TEST_ASSERT(diag_error_max < 1e-14);
    TEST_ASSERT(off_diag_max < 1e-14);
}

void test_rotation_matrices_determinant() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double t_tt = epoch.julian_centuries_tt();
    double ut1_jd = epoch.ut1_jd();

    // Test individual matrices
    Matrix3d Q = gcrs_to_cirs_matrix(t_tt);
    Matrix3d R = cirs_to_tirs_matrix(ut1_jd);
    Matrix3d W = tirs_to_itrf_matrix(EarthOrientationParameters::zero());

    // All should have determinant = 1 (proper rotations)
    TEST_ASSERT(std::abs(Q.determinant() - 1.0) < 1e-10);
    TEST_ASSERT(std::abs(R.determinant() - 1.0) < 1e-10);
    TEST_ASSERT(std::abs(W.determinant() - 1.0) < 1e-10);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Runs without an Earth orientation parameter (EOP) table on purpose:
    // self-consistent checks with UT1-UTC and polar motion zero.
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    TestRunner chain_tests("Geodetic-ECEF Chain");
    chain_tests.add_test("Geodetic roundtrip", test_geodetic_roundtrip);
    chain_tests.run();

    TestRunner iau_tests("Full IAU Transform Chain");
    iau_tests.add_test("GCRS-ITRF position roundtrip", test_gcrs_itrf_roundtrip_position);
    iau_tests.add_test("GCRS-ITRF state roundtrip", test_gcrs_itrf_roundtrip_state);
    iau_tests.add_test("With EOP", test_gcrs_itrf_with_eop);
    iau_tests.run();

    TestRunner polar_tests("Polar Regions");
    polar_tests.add_test("Geodetic at poles", test_geodetic_at_poles);
    polar_tests.run();

    TestRunner equator_tests("Equatorial Regions");
    equator_tests.add_test("Geodetic at equator", test_geodetic_at_equator);
    equator_tests.run();

    TestRunner matrix_tests("Matrix Properties");
    matrix_tests.add_test("GCRS-ITRF orthogonal", test_gcrs_to_itrf_matrix_orthogonal);
    matrix_tests.add_test("Determinant = 1", test_rotation_matrices_determinant);
    matrix_tests.run();

    return print_final_summary();
}
