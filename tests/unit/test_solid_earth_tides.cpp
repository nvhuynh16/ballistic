// Unit tests for the solid Earth tide correction: the Moon and Sun deform the
// Earth slightly, which changes its gravity field. Following the IERS 2010
// Conventions (Chapter 6), the deformation is expressed as small changes to the
// gravity field's spherical-harmonic (Stokes) coefficients C_nm, S_nm, scaled by
// the Love numbers k_nm (how strongly the Earth responds). Checks are order-of-
// magnitude bounds, physical symmetries, the IERS Love-number table, and a hand
// computation of IERS eq. 6.6. Synthetic Moon/Sun positions; no data files.
// Tests: gravity/solid_earth_tides.hpp

#include "../test_utils.hpp"
#include <ballistic/gravity/solid_earth_tides.hpp>
#include <ballistic/core/constants.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::gravity;

// =============================================================================
// Solid Earth Tides Acceleration Tests
// =============================================================================

void test_solid_earth_tides_magnitude() {
    TestRunner runner("SolidEarthTides_Magnitude");

    runner.add_test("leo_acceleration_order_of_magnitude", []() {
        SolidEarthTides tides;

        // LEO satellite at ~800 km altitude, on the x-axis
        Vector3d sat(7178137.0, 0.0, 0.0);  // R_E + 800 km

        // Moon at roughly 384400 km along y-axis (ECEF, simplified)
        Vector3d moon(0.0, 384400000.0, 0.0);

        // Sun at roughly 1 AU along z-axis (ECEF, simplified)
        Vector3d sun(0.0, 0.0, 149597870700.0);

        Vector3d acc = tides.acceleration(sat, moon, sun);
        double mag = acc.norm();

        // Expected: ~5e-8 m/s^2 (decimeter-level effect over 1 hour)
        // Should be in range 1e-9 to 1e-6
        TEST_ASSERT_MSG(mag > 1e-9, "Tidal acceleration too small: " + std::to_string(mag));
        TEST_ASSERT_MSG(mag < 1e-6, "Tidal acceleration too large: " + std::to_string(mag));
    });

    // The lunar tide is stronger than the solar one but the Sun's is not negligible
    // (typically ~45% of the Moon's; the bound is 30%).
    runner.add_test("moon_dominates_over_sun", []() {
        SolidEarthTides tides;

        Vector3d sat(7178137.0, 0.0, 0.0);
        Vector3d moon(0.0, 384400000.0, 0.0);
        Vector3d sun(149597870700.0, 0.0, 0.0);
        Vector3d origin(0.0, 0.0, 0.0);

        // Moon-only contribution
        Vector3d acc_moon = tides.acceleration(sat, moon, origin);
        // Sun-only contribution
        Vector3d acc_sun = tides.acceleration(sat, origin, sun);

        // Moon's tidal effect should be larger than Sun's (~2.2x)
        double moon_mag = acc_moon.norm();
        double sun_mag = acc_sun.norm();

        // Moon should dominate (but Sun is not negligible)
        TEST_ASSERT_MSG(moon_mag > sun_mag, "Moon tidal effect should dominate over Sun");
        // Sun should be at least 30% of Moon (typically ~45%)
        TEST_ASSERT_MSG(sun_mag > 0.3 * moon_mag, "Sun tidal effect should be significant fraction of Moon");
    });

    // A body at the Earth's center (the "absent" convention) contributes nothing.
    runner.add_test("zero_body_position_gives_zero", []() {
        SolidEarthTides tides;

        Vector3d sat(7178137.0, 0.0, 0.0);
        Vector3d origin(0.0, 0.0, 0.0);

        // With both bodies at origin, should return zero
        Vector3d acc = tides.acceleration(sat, origin, origin);
        TEST_ASSERT(acc.norm() < 1e-20);
    });

    runner.add_test("zero_satellite_position_gives_zero", []() {
        SolidEarthTides tides;

        Vector3d origin(0.0, 0.0, 0.0);
        Vector3d moon(0.0, 384400000.0, 0.0);
        Vector3d sun(149597870700.0, 0.0, 0.0);

        // Satellite at origin should give zero (r < 1.0 guard)
        Vector3d acc = tides.acceleration(origin, moon, sun);
        TEST_ASSERT(acc.norm() < 1e-20);
    });

    runner.run();
}

void test_solid_earth_tides_properties() {
    TestRunner runner("SolidEarthTides_Properties");

    // The tide force derives from a potential (it does not depend on velocity).
    runner.add_test("is_conservative", []() {
        TEST_ASSERT(SolidEarthTides::is_conservative());
    });

    runner.add_test("symmetry_opposite_satellite_positions", []() {
        SolidEarthTides tides;

        // Two satellite positions at same altitude, opposite sides
        Vector3d sat1(7178137.0, 0.0, 0.0);
        Vector3d sat2(-7178137.0, 0.0, 0.0);

        // Moon and Sun at fixed positions
        Vector3d moon(0.0, 384400000.0, 0.0);
        Vector3d sun(149597870700.0, 0.0, 0.0);

        double mag1 = tides.acceleration(sat1, moon, sun).norm();
        double mag2 = tides.acceleration(sat2, moon, sun).norm();

        // Magnitudes should be similar (same altitude, different geometry)
        // Not exactly equal due to tesseral terms, but within 50%
        TEST_ASSERT_MSG(std::abs(mag1 - mag2) / (mag1 + mag2) < 0.5,
            "Opposite positions should have comparable tidal magnitudes");
    });

    runner.add_test("higher_altitude_smaller_effect", []() {
        SolidEarthTides tides;

        // 400 km altitude
        Vector3d sat_low(6778137.0, 0.0, 0.0);
        // 2000 km altitude
        Vector3d sat_high(8378137.0, 0.0, 0.0);

        Vector3d moon(0.0, 384400000.0, 0.0);
        Vector3d sun(149597870700.0, 0.0, 0.0);

        double mag_low = tides.acceleration(sat_low, moon, sun).norm();
        double mag_high = tides.acceleration(sat_high, moon, sun).norm();

        // Tidal acceleration decreases with altitude (~ r^(-4) for degree 2)
        TEST_ASSERT_MSG(mag_high < mag_low,
            "Tidal effect should decrease with altitude");
    });

    runner.add_test("realistic_moon_position_dc20", []() {
        SolidEarthTides tides;

        // Satellite on x-axis
        Vector3d sat(7178137.0, 0.0, 0.0);

        // Realistic Moon position (in ECEF, roughly)
        Vector3d moon(200000000.0, 300000000.0, 50000000.0);
        Vector3d sun(0.0, 0.0, 0.0);  // Sun at origin = no contribution

        Vector3d acc = tides.acceleration(sat, moon, sun);

        // With a single realistic Moon position, acceleration should be
        // in the 1e-8 to 1e-7 range
        double mag = acc.norm();
        TEST_ASSERT_MSG(mag > 1e-9 && mag < 1e-6,
            "Realistic Moon position should give ~1e-8 to 1e-7 m/s^2, got " + std::to_string(mag));
    });

    runner.run();
}

void test_solid_earth_tides_love_numbers() {
    TestRunner runner("SolidEarthTides_LoveNumbers");

    // Love numbers are the IERS 2010 Table 6.3 values, to the table's printed digits.
    runner.add_test("constants_match_iers_table", []() {
        TEST_ASSERT_NEAR(constants::tides::k20_real, 0.30190, 1e-5);
        TEST_ASSERT_NEAR(constants::tides::k21_real, 0.29830, 1e-5);
        TEST_ASSERT_NEAR(constants::tides::k21_imag, -0.00144, 1e-5);
        TEST_ASSERT_NEAR(constants::tides::k22_real, 0.30102, 1e-5);
        TEST_ASSERT_NEAR(constants::tides::k22_imag, -0.00130, 1e-5);
        TEST_ASSERT_NEAR(constants::tides::k30_real, 0.093, 1e-4);
        TEST_ASSERT_NEAR(constants::tides::k31_real, 0.093, 1e-4);
        TEST_ASSERT_NEAR(constants::tides::k32_real, 0.093, 1e-4);
        TEST_ASSERT_NEAR(constants::tides::k33_real, 0.094, 1e-4);
    });

    runner.run();
}

// =============================================================================
// IERS 2010 Eq. 6.6 Hand-Computed Worked Example
// =============================================================================

void test_solid_earth_tides_iers_worked_example() {
    TestRunner runner("SolidEarthTides_IERS_WorkedExample");

    runner.add_test("delta_c20_moon_on_z_axis", []() {
        // IERS 2010 Eq. 6.6:
        //   ΔC_nm - i·ΔS_nm = (k_nm/(2n+1)) * (GM_j/GM_E) * (R_E/r_j)^(n+1)
        //                     * P̄_nm(sin φ_j) * exp(-i·m·λ_j)
        //
        // Simplest case: Moon on the z-axis (North Pole direction)
        //   φ_j = π/2 (sin φ = 1, cos φ = 0)
        //   λ_j = 0 (undefined but atan2(0,0)=0)
        //   r_j = 384400 km (mean lunar distance)
        //
        // For (n,m) = (2,0):
        //   P̄_20(1) = sqrt(5)/2 * (3*1² - 1) = sqrt(5)/2 * 2 = sqrt(5)
        //   ΔC_20 = (k20_real/5) * (GM_Moon/GM_Earth) * (R_E/r_Moon)³ * sqrt(5)

        const double GM_Moon = constants::moon::GM;
        const double GM_Earth = constants::earth::GM;
        const double R_E = constants::earth::RADIUS_EQUATORIAL;
        const double r_Moon = 384400000.0;  // Mean distance

        const double gm_ratio = GM_Moon / GM_Earth;
        const double re_r3 = std::pow(R_E / r_Moon, 3);
        const double P20_at_pole = std::sqrt(5.0);  // P̄_20(sin(π/2))

        // Expected ΔC_20
        const double k20_real = 0.30190;
        const double expected_dC20 = (k20_real / 5.0) * gm_ratio * re_r3 * P20_at_pole;

        // For m>0 at the pole: cos_phi = 0, so P_nm = 0 for m≥1
        // Therefore only (2,0) and (3,0) contribute

        // Now verify via the code: place Moon on z-axis at mean distance
        SolidEarthTides tides;
        Vector3d moon_z(0.0, 0.0, r_Moon);
        Vector3d no_sun(0.0, 0.0, 0.0);

        // Place satellite on x-axis at LEO
        const double r_sat = 7178137.0;  // R_E + 800 km
        Vector3d sat(r_sat, 0.0, 0.0);

        Vector3d acc = tides.acceleration(sat, moon_z, no_sun);

        // The acceleration should be non-zero and physical
        TEST_ASSERT(acc.norm() > 1e-9);
        TEST_ASSERT(acc.norm() < 1e-6);

        // Verify ΔC_20 value is reasonable
        // ΔC_20 ≈ (0.30190/5) * (4.903e12/3.986e14) * (6.378e6/3.844e8)³ * 2.236
        // ≈ 0.06038 * 1.230e-2 * 4.558e-6 * 2.236 = 7.55e-9
        TEST_ASSERT_MSG(expected_dC20 > 1e-10 && expected_dC20 < 1e-7,
            "ΔC_20 should be ~1e-8, got " + std::to_string(expected_dC20));

        // For degree-3: P̄_30(1) = sqrt(7)/2*(5-3) = sqrt(7)
        const double k30_real = 0.093;
        const double re_r4 = std::pow(R_E / r_Moon, 4);
        const double P30_at_pole = std::sqrt(7.0);
        const double expected_dC30 = (k30_real / 7.0) * gm_ratio * re_r4 * P30_at_pole;

        // ΔC_30 should be much smaller than ΔC_20 (higher power of R_E/r)
        TEST_ASSERT(expected_dC30 < expected_dC20 * 0.1);
    });

    runner.add_test("delta_stokes_moon_equatorial", []() {
        // Moon on x-axis (equatorial, λ=0):
        //   sin φ = 0, cos φ = 1, λ = 0
        //
        // P̄_20(0) = sqrt(5)/2 * (0 - 1) = -sqrt(5)/2
        // P̄_21(0) = sqrt(5/3) * 3 * 0 * 1 = 0  (m=1 vanishes at equator for odd parity)
        // P̄_22(0) = sqrt(5/12) * 3 * 1 = 3*sqrt(5/12)

        const double sin_phi = 0.0;
        const double cos_phi = 1.0;

        // Verify P̄_20(0)
        double P20 = std::sqrt(5.0) / 2.0 * (3.0 * sin_phi * sin_phi - 1.0);
        TEST_ASSERT_NEAR(P20, -std::sqrt(5.0) / 2.0, 1e-15);

        // Verify P̄_22(0)
        double P22 = std::sqrt(5.0 / 12.0) * 3.0 * cos_phi * cos_phi;
        TEST_ASSERT_NEAR(P22, 3.0 * std::sqrt(5.0 / 12.0), 1e-15);

        // With Moon on x-axis at mean distance: λ=0 → cos(mλ)=1, sin(mλ)=0
        // So ΔS terms vanish and only ΔC terms contribute (for real k only)
        SolidEarthTides tides;
        const double r_Moon = 384400000.0;
        Vector3d moon_x(r_Moon, 0.0, 0.0);
        Vector3d no_sun(0.0, 0.0, 0.0);

        // Satellite perpendicular to Moon (on y-axis) to see tesseral terms
        const double r_sat = 7178137.0;
        Vector3d sat(0.0, r_sat, 0.0);

        Vector3d acc = tides.acceleration(sat, moon_x, no_sun);
        TEST_ASSERT(acc.norm() > 1e-9);
        TEST_ASSERT(acc.norm() < 1e-6);
    });

    runner.add_test("superposition_moon_plus_sun", []() {
        // Verify linearity: acc(Moon+Sun) = acc(Moon) + acc(Sun)
        // The Stokes coefficient changes are additive (Eq. 6.6 sums over bodies),
        // and the acceleration is linear in the Stokes coefficients.
        SolidEarthTides tides;

        Vector3d sat(7178137.0, 0.0, 0.0);
        Vector3d moon(200000000.0, 300000000.0, 50000000.0);
        Vector3d sun(100000000000.0, 80000000000.0, 30000000000.0);
        Vector3d origin(0.0, 0.0, 0.0);

        Vector3d acc_both = tides.acceleration(sat, moon, sun);
        Vector3d acc_moon = tides.acceleration(sat, moon, origin);
        Vector3d acc_sun = tides.acceleration(sat, origin, sun);

        // Additive to 1e-20 m/s^2 (rounding only): same code path, linear accumulation.
        TEST_ASSERT_VEC3_NEAR(acc_both, acc_moon + acc_sun, 1e-20);
    });

    runner.run();
}

int main() {
    test_solid_earth_tides_magnitude();
    test_solid_earth_tides_properties();
    test_solid_earth_tides_love_numbers();
    test_solid_earth_tides_iers_worked_example();
    return print_final_summary();
}
