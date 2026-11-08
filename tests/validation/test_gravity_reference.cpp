// Independent gravity field reference validation
// Tests: egm_gravity.hpp, clenshaw_engine.hpp
//
// This test compares our EGM2008 Clenshaw gravity against independently-derived
// reference values to catch systemic errors in the coefficient transfer or
// summation algorithm.
//
// Reference 1: Somigliana normal gravity formula (WGS84 closed-form, independent
//   of spherical harmonics). Agreement expected to ~0.001 m/s² (main field).
//   Source: Moritz, H. (2000). "Geodetic Reference System 1980".
//          Journal of Geodesy, 74, 128-133.
//
// Reference 2: Degree-2-only analytical gravity (hand-verifiable, no code reuse).
//   The gravitational acceleration from C20 alone can be computed analytically.
//   This isolates coefficient loading and Clenshaw recursion correctness for n=2.

#include "../test_utils.hpp"
#include <ballistic/gravity/egm_gravity.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/init.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::gravity;
using namespace ballistic::constants;

// =============================================================================
// Somigliana Normal Gravity (closed-form, WGS84)
// =============================================================================
// The Somigliana formula gives normal gravity on the WGS84 ellipsoid surface
// as a function of geodetic latitude only. It is derived from the theory of
// the normal gravity field and is completely independent of spherical harmonics.
//
// gamma(phi) = (a * gamma_e * cos²(phi) + b * gamma_p * sin²(phi))
//              / sqrt(a² * cos²(phi) + b² * sin²(phi))
//
// WGS84 values (Moritz 2000):
//   gamma_e = 9.7803253141 m/s² (equatorial normal gravity)
//   gamma_p = 9.8321849379 m/s² (polar normal gravity)
//   a = 6378137.0 m, b = 6356752.3142 m
// =============================================================================

static double somigliana_normal_gravity(double lat_rad) {
    constexpr double a = earth::RADIUS_EQUATORIAL;
    constexpr double b = earth::RADIUS_POLAR;
    constexpr double gamma_e = 9.7803253141;  // m/s², equatorial
    constexpr double gamma_p = 9.8321849379;  // m/s², polar

    double sin_lat = std::sin(lat_rad);
    double cos_lat = std::cos(lat_rad);

    double num = a * gamma_e * cos_lat * cos_lat + b * gamma_p * sin_lat * sin_lat;
    double den = std::sqrt(a * a * cos_lat * cos_lat + b * b * sin_lat * sin_lat);

    return num / den;
}

// =============================================================================
// Test: Somigliana reference comparison
// =============================================================================
// The Somigliana formula gives total GRAVITY (gravitational + centrifugal) on the
// ellipsoid. Our model gives GRAVITATIONAL acceleration only. To compare, we
// compute the centrifugal acceleration vector and add it to the gravitational
// vector, then compare the resulting magnitude with Somigliana.
//
// Centrifugal acceleration vector (ECEF): a_cf = omega² * [x, y, 0]
// Gravity vector: g = a_grav + a_cf
// =============================================================================

void test_somigliana_reference() {
    TestRunner runner("SomiglianaReference");

    auto& model = EGM2008Gravity::instance();
    model.init(360);

    runner.add_test("equator_surface", [&model]() {
        // Equator (0°N, 0°E) on ellipsoid surface
        double lat = 0.0;
        double lon = 0.0;
        Vector3d pos = coords::geodetic_to_ecef(lat, lon, 0.0);

        Vector3d acc_grav = model.acceleration(pos, 360);

        // Centrifugal acceleration vector: omega² * [x, y, 0]
        double omega = earth::ANGULAR_VELOCITY;
        Vector3d acc_cf(omega * omega * pos.x(), omega * omega * pos.y(), 0.0);

        // Gravity = gravitational + centrifugal (vector sum)
        Vector3d gravity = acc_grav + acc_cf;
        double g_total = gravity.norm();

        double g_somigliana = somigliana_normal_gravity(lat);

        // Agreement to ~0.002 m/s² (anomalous gravity ~few mGal)
        TEST_ASSERT_NEAR(g_total, g_somigliana, 0.002);

        std::cout << "[g_grav=" << acc_grav.norm() << ", g_total=" << g_total
                  << ", g_som=" << g_somigliana << "] ";
    });

    runner.add_test("north_pole_surface", [&model]() {
        // North Pole (90°N): no centrifugal contribution (on rotation axis)
        double lat = 90.0 * DEG_TO_RAD;
        double lon = 0.0;
        Vector3d pos = coords::geodetic_to_ecef(lat, lon, 0.0);

        Vector3d acc_grav = model.acceleration(pos, 360);

        // At pole, x=y=0 so centrifugal is zero
        double omega = earth::ANGULAR_VELOCITY;
        Vector3d acc_cf(omega * omega * pos.x(), omega * omega * pos.y(), 0.0);
        Vector3d gravity = acc_grav + acc_cf;
        double g_total = gravity.norm();

        double g_somigliana = somigliana_normal_gravity(lat);
        TEST_ASSERT_NEAR(g_total, g_somigliana, 0.002);

        std::cout << "[g_total=" << g_total << ", g_som=" << g_somigliana << "] ";
    });

    runner.add_test("midlatitude_45N", [&model]() {
        // 45°N, 0°E on surface — proper vector centrifugal correction
        double lat = 45.0 * DEG_TO_RAD;
        double lon = 0.0;
        Vector3d pos = coords::geodetic_to_ecef(lat, lon, 0.0);

        Vector3d acc_grav = model.acceleration(pos, 360);

        double omega = earth::ANGULAR_VELOCITY;
        Vector3d acc_cf(omega * omega * pos.x(), omega * omega * pos.y(), 0.0);
        Vector3d gravity = acc_grav + acc_cf;
        double g_total = gravity.norm();

        double g_somigliana = somigliana_normal_gravity(lat);
        TEST_ASSERT_NEAR(g_total, g_somigliana, 0.002);

        std::cout << "[g_total=" << g_total << ", g_som=" << g_somigliana << "] ";
    });

    runner.add_test("south_pole_surface", [&model]() {
        double lat = -90.0 * DEG_TO_RAD;
        double lon = 0.0;
        Vector3d pos = coords::geodetic_to_ecef(lat, lon, 0.0);

        Vector3d acc_grav = model.acceleration(pos, 360);
        double omega = earth::ANGULAR_VELOCITY;
        Vector3d acc_cf(omega * omega * pos.x(), omega * omega * pos.y(), 0.0);
        Vector3d gravity = acc_grav + acc_cf;
        double g_total = gravity.norm();

        double g_somigliana = somigliana_normal_gravity(lat);
        TEST_ASSERT_NEAR(g_total, g_somigliana, 0.002);
    });

    runner.add_test("latitude_30N", [&model]() {
        double lat = 30.0 * DEG_TO_RAD;
        double lon = 45.0 * DEG_TO_RAD;  // Non-zero longitude
        Vector3d pos = coords::geodetic_to_ecef(lat, lon, 0.0);

        Vector3d acc_grav = model.acceleration(pos, 360);
        double omega = earth::ANGULAR_VELOCITY;
        Vector3d acc_cf(omega * omega * pos.x(), omega * omega * pos.y(), 0.0);
        Vector3d gravity = acc_grav + acc_cf;
        double g_total = gravity.norm();

        double g_somigliana = somigliana_normal_gravity(lat);
        TEST_ASSERT_NEAR(g_total, g_somigliana, 0.002);

        std::cout << "[g_total=" << g_total << ", g_som=" << g_somigliana << "] ";
    });

    runner.run();
}

// =============================================================================
// Test: Degree-2 analytical gravity (hand-computed, fully independent)
// =============================================================================
// For a degree-2-only gravity field, the acceleration can be computed analytically.
// At the pole (on the rotation axis), tesseral terms (C21, C22, S22) contribute
// zero because cos(m*lambda) and sin(m*lambda) vanish or Legendre functions vanish.
// This gives a clean comparison for the zonal C20 term.
//
// At the equator on the X-axis, C22 and S22 contribute non-trivially, so we use
// a wider tolerance there.
// =============================================================================

void test_degree2_analytical() {
    TestRunner runner("Degree2Analytical");

    runner.add_test("equator_c20_analytical", []() {
        // At the equator (φ=0) on X-axis
        double r = earth::RADIUS_EQUATORIAL;
        Vector3d pos(r, 0.0, 0.0);

        auto& model = EGM2008Gravity::instance();
        model.init(360);
        Vector3d acc_clenshaw = model.acceleration(pos, 2);

        // Hand-computed C20-only radial acceleration at equator:
        // J2 = -sqrt(5) * C20 (denormalization)
        double C20 = -0.484165143790815e-03;
        double J2_from_C20 = -std::sqrt(5.0) * C20;
        double ar2 = (earth::RADIUS_EQUATORIAL / r) * (earth::RADIUS_EQUATORIAL / r);  // = 1.0

        // At equator (sin φ = 0): a_r = -(GM/r²) * [1 + 3/2 * J2 * (a/r)²]
        double a_r_expected = -(earth::GM / (r * r)) * (1.0 + 1.5 * J2_from_C20 * ar2);

        double a_r_clenshaw = acc_clenshaw.x();

        // C22/S22 contribute ~1.4e-4 m/s² at the equator on the X-axis
        // (C22 ≈ 2.44e-6, contributes 6*GM/r² * C22 * sqrt(5/12) ≈ 1.4e-4)
        TEST_ASSERT_NEAR(a_r_clenshaw, a_r_expected, 2e-4);

        std::cout << "[a_r_clenshaw=" << a_r_clenshaw
                  << ", a_r_c20=" << a_r_expected
                  << ", diff=" << std::abs(a_r_clenshaw - a_r_expected) << "] ";
    });

    runner.add_test("pole_c20_analytical", []() {
        // At the pole (φ=π/2), tesseral terms vanish, so C20 dominates
        double r = earth::RADIUS_POLAR;
        Vector3d pos(0.0, 0.0, r);

        auto& model = EGM2008Gravity::instance();
        Vector3d acc_clenshaw = model.acceleration(pos, 2);

        double C20 = -0.484165143790815e-03;
        double J2_from_C20 = -std::sqrt(5.0) * C20;
        double ar2 = (earth::RADIUS_EQUATORIAL / r) * (earth::RADIUS_EQUATORIAL / r);

        // At pole (sin φ = 1): a_r = -(GM/r²) * [1 - 3 * J2 * (a/r)²]
        double a_r_expected = -(earth::GM / (r * r)) * (1.0 - 3.0 * J2_from_C20 * ar2);

        double a_z_clenshaw = acc_clenshaw.z();

        // At pole, C21/C22 tesseral terms contribute negligibly
        TEST_ASSERT_NEAR(a_z_clenshaw, a_r_expected, 1e-4);

        std::cout << "[a_z_clenshaw=" << a_z_clenshaw
                  << ", a_r_hand=" << a_r_expected << "] ";
    });

    runner.run();
}

// =============================================================================
// Test: Gravity magnitude ordering (sanity checks)
// =============================================================================

void test_gravity_physical_constraints() {
    TestRunner runner("GravityPhysicalConstraints");

    auto& model = EGM2008Gravity::instance();
    model.init(360);

    runner.add_test("polar_gt_equatorial_gravitational", [&model]() {
        // Gravitational acceleration (no centrifugal) at pole vs equator.
        // The pole is closer to Earth's center (b < a), so gravitational
        // acceleration is stronger at the pole.
        Vector3d pos_eq(earth::RADIUS_EQUATORIAL, 0.0, 0.0);
        Vector3d pos_pole(0.0, 0.0, earth::RADIUS_POLAR);

        double g_eq = model.acceleration(pos_eq, 70).norm();
        double g_pole = model.acceleration(pos_pole, 70).norm();

        TEST_ASSERT_MSG(g_pole > g_eq,
            "Polar gravitational (" + std::to_string(g_pole) +
            ") should exceed equatorial (" + std::to_string(g_eq) + ")");

        // Gravitational-only difference: ~0.018 m/s²
        // (The full gravity difference including centrifugal is ~0.052 m/s²)
        double diff = g_pole - g_eq;
        TEST_ASSERT(diff > 0.010 && diff < 0.030);

        std::cout << "[diff=" << diff * 1000 << " mGal] ";
    });

    runner.add_test("gravity_decreases_with_altitude", [&model]() {
        Vector3d pos_surface(earth::RADIUS_EQUATORIAL, 0.0, 0.0);
        Vector3d pos_200km(earth::RADIUS_EQUATORIAL + 200000.0, 0.0, 0.0);
        Vector3d pos_400km(earth::RADIUS_EQUATORIAL + 400000.0, 0.0, 0.0);

        double g_surface = model.acceleration(pos_surface, 70).norm();
        double g_200km = model.acceleration(pos_200km, 70).norm();
        double g_400km = model.acceleration(pos_400km, 70).norm();

        TEST_ASSERT(g_surface > g_200km);
        TEST_ASSERT(g_200km > g_400km);

        // At 400 km, gravity should be ~8.69 m/s² (1 - 2h/r of surface)
        TEST_ASSERT_NEAR(g_400km, 8.69, 0.03);
    });

    runner.add_test("j2_dominates_higher_harmonics", [&model]() {
        Vector3d pos(6778137.0, 1000000.0, 3000000.0);

        Vector3d acc_2 = model.acceleration(pos, 2);    // degree 2 only
        Vector3d acc_70 = model.acceleration(pos, 70);   // full field

        // Point-mass (degree 0) contribution
        double r = pos.norm();
        Vector3d acc_0 = -(earth::GM / (r * r * r)) * pos;

        double j2_contribution = (acc_2 - acc_0).norm();
        double higher_contribution = (acc_70 - acc_2).norm();

        // J2 should be at least 100x larger than degrees 3-70 combined
        TEST_ASSERT_MSG(j2_contribution > higher_contribution * 100,
            "J2 contribution (" + std::to_string(j2_contribution) +
            ") should dominate higher harmonics (" + std::to_string(higher_contribution) + ")");
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Gravity Field Reference Validation" << std::endl;
    std::cout << "============================================" << std::endl;

    initialize({});

    test_somigliana_reference();
    test_degree2_analytical();
    test_gravity_physical_constraints();

    return print_final_summary();
}
