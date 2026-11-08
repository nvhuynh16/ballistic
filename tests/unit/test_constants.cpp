// Constants (core/constants.hpp): mathematical, time, WGS84 Earth, Sun/Moon/Jupiter
// gravitational parameters (GM, from JPL DE440's gm_de440.tpc), Love numbers (IERS
// Conventions 2010) and the zonal harmonics J2..J15, which must equal the embedded
// EGM2008 gravity model's coefficients. Each value is checked against its published
// source and against an independent physical cross-check where one exists.

#include "../test_utils.hpp"
#include <ballistic/core/constants.hpp>
#include <ballistic/gravity/egm2008_coefficients.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::test;

// Test runner for this suite
static TestRunner runner("Constants Tests");

//=============================================================================
// Mathematical Constants
//=============================================================================

void test_pi_value() {
    // Compare against std::numbers::pi (C++20) or M_PI
    TEST_ASSERT_NEAR(constants::PI, 3.14159265358979323846, 1e-18);

    // Verify it's actually pi using trigonometric identity
    TEST_ASSERT_NEAR(std::sin(constants::PI), 0.0, 1e-15);
    TEST_ASSERT_NEAR(std::cos(constants::PI), -1.0, 1e-15);
}

void test_two_pi_value() {
    TEST_ASSERT_NEAR(constants::TWO_PI, 2.0 * constants::PI, 1e-18);
    TEST_ASSERT_NEAR(constants::TWO_PI, 6.28318530717958647692, 1e-15);

    // Full rotation
    TEST_ASSERT_NEAR(std::sin(constants::TWO_PI), 0.0, 1e-14);
    TEST_ASSERT_NEAR(std::cos(constants::TWO_PI), 1.0, 1e-15);
}

void test_deg_to_rad_conversion() {
    // 180 degrees = PI radians
    TEST_ASSERT_NEAR(180.0 * constants::DEG_TO_RAD, constants::PI, 1e-15);

    // 90 degrees = PI/2 radians
    TEST_ASSERT_NEAR(90.0 * constants::DEG_TO_RAD, constants::PI / 2.0, 1e-15);

    // 360 degrees = 2*PI radians
    TEST_ASSERT_NEAR(360.0 * constants::DEG_TO_RAD, constants::TWO_PI, 1e-15);

    // 1 degree in radians
    TEST_ASSERT_NEAR(constants::DEG_TO_RAD, constants::PI / 180.0, 1e-18);
}

void test_rad_to_deg_conversion() {
    // PI radians = 180 degrees
    TEST_ASSERT_NEAR(constants::PI * constants::RAD_TO_DEG, 180.0, 1e-12);

    // Inverse relationship
    TEST_ASSERT_NEAR(constants::DEG_TO_RAD * constants::RAD_TO_DEG, 1.0, 1e-15);

    // Roundtrip conversion
    double angle_deg = 45.0;
    double angle_rad = angle_deg * constants::DEG_TO_RAD;
    double back_deg = angle_rad * constants::RAD_TO_DEG;
    TEST_ASSERT_NEAR(back_deg, angle_deg, 1e-14);
}

void test_arcsec_to_rad_conversion() {
    // 3600 arcsec = 1 degree
    TEST_ASSERT_NEAR(3600.0 * constants::ARCSEC_TO_RAD, constants::DEG_TO_RAD, 1e-18);

    // 1 arcsec = PI / (180 * 3600) radians
    TEST_ASSERT_NEAR(constants::ARCSEC_TO_RAD, constants::PI / 648000.0, 1e-20);

    // 1 degree = 3600 arcsec
    TEST_ASSERT_NEAR(constants::DEG_TO_RAD / constants::ARCSEC_TO_RAD, 3600.0, 1e-10);
}

//=============================================================================
// Time Constants
//=============================================================================

void test_seconds_per_day() {
    TEST_ASSERT_EQ(constants::SECONDS_PER_DAY, 86400.0);
    TEST_ASSERT_EQ(constants::SECONDS_PER_DAY, 24.0 * 60.0 * 60.0);
}

void test_seconds_per_hour() {
    TEST_ASSERT_EQ(constants::SECONDS_PER_HOUR, 3600.0);
    TEST_ASSERT_EQ(constants::SECONDS_PER_HOUR, 60.0 * 60.0);
}

void test_seconds_per_minute() {
    TEST_ASSERT_EQ(constants::SECONDS_PER_MINUTE, 60.0);
}

void test_time_constants_consistency() {
    // Day = 24 hours
    TEST_ASSERT_EQ(constants::SECONDS_PER_DAY / constants::SECONDS_PER_HOUR, 24.0);

    // Hour = 60 minutes
    TEST_ASSERT_EQ(constants::SECONDS_PER_HOUR / constants::SECONDS_PER_MINUTE, 60.0);
}

//=============================================================================
// Julian Date Constants
//=============================================================================

void test_j2000_jd() {
    // J2000.0 = January 1, 2000, 12:00:00 TT
    TEST_ASSERT_EQ(constants::J2000_JD, 2451545.0);
}

void test_mjd_offset() {
    // MJD = JD - 2400000.5
    TEST_ASSERT_EQ(constants::MJD_OFFSET, 2400000.5);

    // J2000.0 in MJD
    double j2000_mjd = constants::J2000_JD - constants::MJD_OFFSET;
    TEST_ASSERT_NEAR(j2000_mjd, 51544.5, 1e-10);
}

void test_unix_epoch_jd() {
    // Unix epoch = January 1, 1970, 00:00:00 UTC
    TEST_ASSERT_EQ(constants::UNIX_EPOCH_JD, 2440587.5);
}

//=============================================================================
// Time Scale Offsets
//=============================================================================

void test_tt_tai_offset() {
    // TT = TAI + 32.184 seconds (exact, by definition)
    TEST_ASSERT_EQ(constants::TT_TAI_OFFSET, 32.184);
}

void test_gps_tai_offset() {
    // GPS = TAI - 19 seconds (fixed offset since GPS epoch)
    TEST_ASSERT_EQ(constants::GPS_TAI_OFFSET, -19.0);
}

//=============================================================================
// Earth Parameters (WGS84)
//=============================================================================

void test_earth_gm() {
    // WGS84 GM value
    TEST_ASSERT_NEAR(constants::earth::GM, 3.986004418e14, 1e6);

    // Should give ~9.8 m/s^2 surface gravity
    double r = constants::earth::RADIUS_EQUATORIAL;
    double g = constants::earth::GM / (r * r);
    TEST_ASSERT_NEAR(g, 9.8, 0.2);
}

void test_earth_radius_equatorial() {
    // WGS84 semi-major axis
    TEST_ASSERT_EQ(constants::earth::RADIUS_EQUATORIAL, 6378137.0);

    // About 6378 km
    TEST_ASSERT_NEAR(constants::earth::RADIUS_EQUATORIAL / 1000.0, 6378.137, 0.001);
}

void test_earth_radius_polar() {
    // WGS84 semi-minor axis
    TEST_ASSERT_NEAR(constants::earth::RADIUS_POLAR, 6356752.3142, 0.1);

    // Polar radius < equatorial radius
    TEST_ASSERT_TRUE(constants::earth::RADIUS_POLAR < constants::earth::RADIUS_EQUATORIAL);

    // Difference is about 21 km
    double diff = constants::earth::RADIUS_EQUATORIAL - constants::earth::RADIUS_POLAR;
    TEST_ASSERT_NEAR(diff / 1000.0, 21.385, 0.01);
}

void test_earth_flattening() {
    // WGS84 flattening = 1/298.257223563
    TEST_ASSERT_NEAR(constants::earth::FLATTENING, 1.0 / 298.257223563, 1e-15);

    // Flattening = (a - b) / a
    double f_calc = (constants::earth::RADIUS_EQUATORIAL - constants::earth::RADIUS_POLAR)
                    / constants::earth::RADIUS_EQUATORIAL;
    TEST_ASSERT_NEAR(constants::earth::FLATTENING, f_calc, 1e-10);
}

void test_earth_eccentricity_squared() {
    // e^2 = 2f - f^2
    double f = constants::earth::FLATTENING;
    double e2_calc = 2.0 * f - f * f;
    TEST_ASSERT_NEAR(constants::earth::ECCENTRICITY_SQ, e2_calc, 1e-15);

    // e^2 should be positive and small (about 0.00669)
    TEST_ASSERT_TRUE(constants::earth::ECCENTRICITY_SQ > 0);
    TEST_ASSERT_TRUE(constants::earth::ECCENTRICITY_SQ < 0.01);
}

void test_earth_angular_velocity() {
    // Earth rotation rate ~7.29e-5 rad/s
    TEST_ASSERT_NEAR(constants::earth::ANGULAR_VELOCITY, 7.292115e-5, 1e-10);

    // Should give ~24 hour rotation period
    double period_hours = constants::TWO_PI / constants::earth::ANGULAR_VELOCITY / 3600.0;
    TEST_ASSERT_NEAR(period_hours, 23.9345, 0.01);  // Sidereal day
}

void test_earth_j2() {
    // J2 is the dominant zonal harmonic (~1e-3)
    TEST_ASSERT_NEAR(constants::earth::J2, 1.0826261738522e-3, 1e-15);  // EGM2008

    // J2 should be positive
    TEST_ASSERT_TRUE(constants::earth::J2 > 0);
}

void test_earth_j3() {
    // J3 is much smaller and negative
    TEST_ASSERT_NEAR(constants::earth::J3, -2.5324105185677e-6, 1e-17);  // EGM2008
    TEST_ASSERT_TRUE(constants::earth::J3 < 0);

    // |J3| << |J2|
    TEST_ASSERT_TRUE(std::abs(constants::earth::J3) < std::abs(constants::earth::J2) * 0.01);
}

void test_earth_j4() {
    // Exact values are checked against the EGM2008 table (test_earth_j7_to_j15).
    TEST_ASSERT_TRUE(constants::earth::J4 < 0);
}

void test_earth_j5() {
    TEST_ASSERT_TRUE(constants::earth::J5 < 0);
}

void test_earth_j6() {
    TEST_ASSERT_TRUE(constants::earth::J6 > 0);  // J6 is positive
}

void test_earth_zonal_harmonics_magnitude_order() {
    // |J2| > |J3| > |J4| > |J5| > |J6| (approximately)
    TEST_ASSERT_TRUE(std::abs(constants::earth::J2) > std::abs(constants::earth::J3));
    TEST_ASSERT_TRUE(std::abs(constants::earth::J3) > std::abs(constants::earth::J4));
    // J5 and J6 are similar magnitude, no strict ordering
}

//=============================================================================
// Sun Parameters
//=============================================================================

void test_sun_gm() {
    // Sun GM from DE440 (gm_de440.tpc)
    TEST_ASSERT_NEAR(constants::sun::GM, 1.32712440041279419e20, 1e8);

    // Sun GM >> Earth GM
    TEST_ASSERT_TRUE(constants::sun::GM > constants::earth::GM * 1e5);

    // Cross-check: Sun/Earth GM ratio ≈ 332946.05 (well-known mass ratio)
    double ratio = constants::sun::GM / constants::earth::GM;
    TEST_ASSERT_NEAR(ratio, 332946.05, 1.0);
}

void test_sun_au() {
    // Astronomical Unit = ~149.6 million km (IAU 2012, exact)
    TEST_ASSERT_NEAR(constants::sun::AU, 149597870700.0, 1.0);

    // About 150 million km
    TEST_ASSERT_NEAR(constants::sun::AU / 1e9, 149.6, 0.1);

    // Cross-check: Earth orbital period T = 2*pi*sqrt(AU^3/GM_sun) ≈ 365.25 days
    double T_sec = constants::TWO_PI * std::sqrt(
        constants::sun::AU * constants::sun::AU * constants::sun::AU / constants::sun::GM);
    double T_days = T_sec / constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(T_days, 365.25, 0.3);
}

//=============================================================================
// Moon Parameters
//=============================================================================

void test_moon_gm() {
    // Moon GM from DE440 (gm_de440.tpc)
    TEST_ASSERT_NEAR(constants::moon::GM, 4.9028001184575496e12, 1e4);

    // Moon GM << Earth GM
    TEST_ASSERT_TRUE(constants::moon::GM < constants::earth::GM * 0.02);

    // Cross-check: Earth/Moon GM ratio (EMRAT) ≈ 81.3006 (DE440)
    double emrat = constants::earth::GM / constants::moon::GM;
    TEST_ASSERT_NEAR(emrat, 81.3006, 0.01);
}

void test_moon_mean_distance() {
    // Mean Earth-Moon distance ~384,400 km
    TEST_ASSERT_NEAR(constants::moon::MEAN_DISTANCE, 384400000.0, 1.0);
    TEST_ASSERT_NEAR(constants::moon::MEAN_DISTANCE / 1e6, 384.4, 0.1);
}

//=============================================================================
// Jupiter Parameters
//=============================================================================

void test_jupiter_gm() {
    // Jupiter system barycenter GM from DE440 (gm_de440.tpc)
    TEST_ASSERT_NEAR(constants::jupiter::GM, 1.26712764e17, 1e9);

    // Jupiter GM > Earth GM
    TEST_ASSERT_TRUE(constants::jupiter::GM > constants::earth::GM * 100);

    // Cross-check: Jupiter/Sun GM ratio ≈ 9.547e-4 (well-known)
    double ratio = constants::jupiter::GM / constants::sun::GM;
    TEST_ASSERT_NEAR(ratio, 9.547e-4, 1e-5);
}

//=============================================================================
// Physical Constants
//=============================================================================

void test_speed_of_light() {
    // Speed of light = 299,792,458 m/s (exact by definition)
    TEST_ASSERT_EQ(constants::SPEED_OF_LIGHT, 299792458.0);
}

//=============================================================================
// Atmospheric Constants
//=============================================================================

void test_sea_level_density() {
    // Standard sea level density = 1.225 kg/m^3
    TEST_ASSERT_NEAR(constants::atmosphere::SEA_LEVEL_DENSITY, 1.225, 0.001);
}

void test_scale_height() {
    // Approximate scale height ~8.5 km
    TEST_ASSERT_NEAR(constants::atmosphere::SCALE_HEIGHT, 8500.0, 100.0);
}

void test_karman_line() {
    // Karman line = 100 km
    TEST_ASSERT_EQ(constants::atmosphere::KARMAN_LINE, 100000.0);
}

//=============================================================================
// J7-J15 Zonal Harmonics
//=============================================================================

void test_earth_j7_to_j15() {  // all of J2-J15, despite the name
    // Every J_n must be the embedded EGM2008 model's -C_n0 * sqrt(2n+1),
    // derived here from the table rather than pinned literals (literal pins
    // can freeze a mistyped coefficient).

    for (const auto& c : gravity::egm2008_data::COEFFICIENTS) {
        if (c.m != 0 || c.n < 2 || c.n > 15) continue;
        const double j = -c.C * std::sqrt(2.0 * c.n + 1.0);
        TEST_ASSERT_NEAR(constants::earth::J_COEFFS[c.n], j, 1e-15 * std::abs(j));
    }

    // J_COEFFS array consistency: J_COEFFS[n] == Jn
    TEST_ASSERT_EQ(constants::earth::J_COEFFS[0], 0.0);
    TEST_ASSERT_EQ(constants::earth::J_COEFFS[1], 0.0);
    TEST_ASSERT_EQ(constants::earth::J_COEFFS[2], constants::earth::J2);
    TEST_ASSERT_EQ(constants::earth::J_COEFFS[3], constants::earth::J3);
    TEST_ASSERT_EQ(constants::earth::J_COEFFS[7], constants::earth::J7);
    TEST_ASSERT_EQ(constants::earth::J_COEFFS[10], constants::earth::J10);
    TEST_ASSERT_EQ(constants::earth::J_COEFFS[15], constants::earth::J15);
}

//=============================================================================
// Love Numbers
//=============================================================================

void test_love_numbers() {
    // IERS 2010 Table 6.3 (anelastic, nominal)
    // Degree 2 Love numbers
    TEST_ASSERT_NEAR(constants::tides::k20_real, 0.30190, 1e-5);
    TEST_ASSERT_NEAR(constants::tides::k21_real, 0.29830, 1e-5);
    TEST_ASSERT_NEAR(constants::tides::k21_imag, -0.00144, 1e-5);
    TEST_ASSERT_NEAR(constants::tides::k22_real, 0.30102, 1e-5);
    TEST_ASSERT_NEAR(constants::tides::k22_imag, -0.00130, 1e-5);

    // Degree 3 Love numbers (all ~0.093)
    TEST_ASSERT_NEAR(constants::tides::k30_real, 0.093, 1e-3);
    TEST_ASSERT_NEAR(constants::tides::k31_real, 0.093, 1e-3);
    TEST_ASSERT_NEAR(constants::tides::k32_real, 0.093, 1e-3);
    TEST_ASSERT_NEAR(constants::tides::k33_real, 0.094, 1e-3);

    // k2 real parts should all be ~0.3 (basic sanity)
    TEST_ASSERT_TRUE(constants::tides::k20_real > 0.29 && constants::tides::k20_real < 0.31);
    TEST_ASSERT_TRUE(constants::tides::k21_real > 0.29 && constants::tides::k21_real < 0.31);
    TEST_ASSERT_TRUE(constants::tides::k22_real > 0.29 && constants::tides::k22_real < 0.31);

    // k3 should be much smaller than k2
    TEST_ASSERT_TRUE(constants::tides::k30_real < constants::tides::k20_real);
}

//=============================================================================
// Independent Cross-Checks
//=============================================================================

void test_days_per_julian_century() {
    TEST_ASSERT_EQ(constants::DAYS_PER_JULIAN_CENTURY, 36525.0);
}

void test_sidereal_day_crosscheck() {
    // Earth angular velocity should give sidereal day ≈ 86164.1 seconds
    double sidereal_day = constants::TWO_PI / constants::earth::ANGULAR_VELOCITY;
    TEST_ASSERT_NEAR(sidereal_day, 86164.1, 0.1);
}

void test_j2_denormalization_crosscheck() {
    // -J2 / sqrt(5) gives the normalized C20 ≈ -4.84e-4
    // J2 = -C20 * sqrt(2*2+1) = -C20 * sqrt(5)
    // So C20 = -J2 / sqrt(5)
    double C20 = -constants::earth::J2 / std::sqrt(5.0);
    TEST_ASSERT_NEAR(C20, -4.84e-4, 1e-6);
}

void test_emrat_crosscheck() {
    // Earth/Moon GM ratio should match DE440 EMRAT = 81.3005690699
    double emrat = constants::earth::GM / constants::moon::GM;
    // Note: we use WGS84 Earth GM (not DE440), so ratio won't be exact DE440 EMRAT
    // but should be within ~0.01 of the well-known value
    TEST_ASSERT_NEAR(emrat, 81.3006, 0.01);
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "BALLISTIC LIBRARY - CONSTANTS TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    // Mathematical constants
    runner.add_test("pi_value", test_pi_value);
    runner.add_test("two_pi_value", test_two_pi_value);
    runner.add_test("deg_to_rad_conversion", test_deg_to_rad_conversion);
    runner.add_test("rad_to_deg_conversion", test_rad_to_deg_conversion);
    runner.add_test("arcsec_to_rad_conversion", test_arcsec_to_rad_conversion);

    // Time constants
    runner.add_test("seconds_per_day", test_seconds_per_day);
    runner.add_test("seconds_per_hour", test_seconds_per_hour);
    runner.add_test("seconds_per_minute", test_seconds_per_minute);
    runner.add_test("time_constants_consistency", test_time_constants_consistency);

    // Julian date constants
    runner.add_test("j2000_jd", test_j2000_jd);
    runner.add_test("mjd_offset", test_mjd_offset);
    runner.add_test("unix_epoch_jd", test_unix_epoch_jd);

    // Time scale offsets
    runner.add_test("tt_tai_offset", test_tt_tai_offset);
    runner.add_test("gps_tai_offset", test_gps_tai_offset);

    // Earth parameters
    runner.add_test("earth_gm", test_earth_gm);
    runner.add_test("earth_radius_equatorial", test_earth_radius_equatorial);
    runner.add_test("earth_radius_polar", test_earth_radius_polar);
    runner.add_test("earth_flattening", test_earth_flattening);
    runner.add_test("earth_eccentricity_squared", test_earth_eccentricity_squared);
    runner.add_test("earth_angular_velocity", test_earth_angular_velocity);
    runner.add_test("earth_j2", test_earth_j2);
    runner.add_test("earth_j3", test_earth_j3);
    runner.add_test("earth_j4", test_earth_j4);
    runner.add_test("earth_j5", test_earth_j5);
    runner.add_test("earth_j6", test_earth_j6);
    runner.add_test("earth_zonal_harmonics_magnitude_order", test_earth_zonal_harmonics_magnitude_order);

    // Sun parameters
    runner.add_test("sun_gm", test_sun_gm);
    runner.add_test("sun_au", test_sun_au);

    // Moon parameters
    runner.add_test("moon_gm", test_moon_gm);
    runner.add_test("moon_mean_distance", test_moon_mean_distance);

    // Jupiter parameters
    runner.add_test("jupiter_gm", test_jupiter_gm);

    // Physical constants
    runner.add_test("speed_of_light", test_speed_of_light);

    // Atmospheric constants
    runner.add_test("sea_level_density", test_sea_level_density);
    runner.add_test("scale_height", test_scale_height);
    runner.add_test("karman_line", test_karman_line);

    // J7-J15 and array consistency
    runner.add_test("earth_j7_to_j15", test_earth_j7_to_j15);

    // Love numbers
    runner.add_test("love_numbers", test_love_numbers);

    // Independent cross-checks
    runner.add_test("days_per_julian_century", test_days_per_julian_century);
    runner.add_test("sidereal_day_crosscheck", test_sidereal_day_crosscheck);
    runner.add_test("j2_denormalization_crosscheck", test_j2_denormalization_crosscheck);
    runner.add_test("emrat_crosscheck", test_emrat_crosscheck);

    int result = runner.run();
    return print_final_summary();
}
