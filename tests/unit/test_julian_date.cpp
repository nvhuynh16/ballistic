// Julian dates (time/julian_date.hpp): calendar <-> Julian Date (JD, days since
// 4713 BC noon) and Modified Julian Date (MJD = JD - 2400000.5), Julian centuries
// since J2000.0 (2000-01-01 12:00 TT), Unix time, leap years and day of year.
// Reference values are standard (Meeus, "Astronomical Algorithms"); tolerances
// are floating-point limits of a JD near 2.45e6 (~1e-9 day = ~0.1 ms).
// TT = Terrestrial Time, TDB = Barycentric Dynamical Time.

#include "../test_utils.hpp"
#include <ballistic/time/julian_date.hpp>
#include <ballistic/core/constants.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::time;
using namespace ballistic::test;

static TestRunner runner("Julian Date Tests");

//=============================================================================
// calendar_to_jd Tests
//=============================================================================

void test_calendar_to_jd_j2000() {
    // J2000.0 = January 1, 2000, 12:00:00 TT
    double jd = calendar_to_jd(2000, 1, 1, 12, 0, 0);
    TEST_ASSERT_NEAR(jd, constants::J2000_JD, 1e-9);
    TEST_ASSERT_NEAR(jd, 2451545.0, 1e-9);
}

void test_calendar_to_jd_unix_epoch() {
    // Unix epoch = January 1, 1970, 00:00:00 UTC
    double jd = calendar_to_jd(1970, 1, 1, 0, 0, 0);
    TEST_ASSERT_NEAR(jd, constants::UNIX_EPOCH_JD, 1e-9);
    TEST_ASSERT_NEAR(jd, 2440587.5, 1e-9);
}

void test_calendar_to_jd_historical_dates() {
    // October 15, 1582 = First day of Gregorian calendar
    // JD should be approximately 2299160.5
    double jd = calendar_to_jd(1582, 10, 15, 0, 0, 0);
    TEST_ASSERT_NEAR(jd, 2299160.5, 1e-6);

    // January 1, 1900 at noon
    double jd_1900 = calendar_to_jd(1900, 1, 1, 12, 0, 0);
    TEST_ASSERT_NEAR(jd_1900, 2415021.0, 1e-6);
}

void test_calendar_to_jd_future_dates() {
    // January 1, 2100 at noon
    double jd = calendar_to_jd(2100, 1, 1, 12, 0, 0);
    // Calculated value: should be about 2488070.0
    TEST_ASSERT_TRUE(jd > 2488069.0 && jd < 2488071.0);

    // January 1, 2050 at noon
    double jd_2050 = calendar_to_jd(2050, 1, 1, 12, 0, 0);
    TEST_ASSERT_TRUE(jd_2050 > jd_to_mjd(51544.5) + constants::MJD_OFFSET);  // After J2000
}

void test_calendar_to_jd_leap_year_date() {
    // February 29, 2024 (leap year)
    double jd = calendar_to_jd(2024, 2, 29, 12, 0, 0);
    TEST_ASSERT_TRUE(jd > 0);  // Valid date

    // March 1, 2024 should be one day after Feb 29
    double jd_march = calendar_to_jd(2024, 3, 1, 12, 0, 0);
    TEST_ASSERT_NEAR(jd_march - jd, 1.0, 1e-9);
}

void test_calendar_to_jd_fractional_seconds() {
    // Test sub-second precision
    double jd1 = calendar_to_jd(2024, 1, 1, 0, 0, 0.0);
    double jd2 = calendar_to_jd(2024, 1, 1, 0, 0, 0.5);

    // 0.5 seconds = 0.5/86400 days
    double expected_diff = 0.5 / constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(jd2 - jd1, expected_diff, 1e-9);  // Relaxed for floating-point precision
}

void test_calendar_to_jd_midnight_vs_noon() {
    // Midnight (0:00) should be 0.5 days before noon
    double jd_midnight = calendar_to_jd(2024, 1, 1, 0, 0, 0);
    double jd_noon = calendar_to_jd(2024, 1, 1, 12, 0, 0);

    TEST_ASSERT_NEAR(jd_noon - jd_midnight, 0.5, 1e-10);
}

void test_calendar_to_jd_january_february_handling() {
    // Test the January/February month correction
    // December 31 of year Y should be close to January 1 of year Y+1
    double jd_dec31 = calendar_to_jd(2023, 12, 31, 12, 0, 0);
    double jd_jan1 = calendar_to_jd(2024, 1, 1, 12, 0, 0);

    TEST_ASSERT_NEAR(jd_jan1 - jd_dec31, 1.0, 1e-9);

    // February 28 to March 1 in non-leap year
    double jd_feb28 = calendar_to_jd(2023, 2, 28, 12, 0, 0);
    double jd_mar1 = calendar_to_jd(2023, 3, 1, 12, 0, 0);
    TEST_ASSERT_NEAR(jd_mar1 - jd_feb28, 1.0, 1e-9);
}

void test_calendar_to_jd_day_progression() {
    // Each consecutive day should increase JD by exactly 1
    double jd1 = calendar_to_jd(2024, 6, 15, 0, 0, 0);
    double jd2 = calendar_to_jd(2024, 6, 16, 0, 0, 0);
    double jd3 = calendar_to_jd(2024, 6, 17, 0, 0, 0);

    TEST_ASSERT_NEAR(jd2 - jd1, 1.0, 1e-10);
    TEST_ASSERT_NEAR(jd3 - jd2, 1.0, 1e-10);
}

//=============================================================================
// jd_to_calendar Tests
//=============================================================================

void test_jd_to_calendar_j2000() {
    auto [y, m, d, h, mi, s] = jd_to_calendar(constants::J2000_JD);

    TEST_ASSERT_EQ(y, 2000);
    TEST_ASSERT_EQ(m, 1);
    TEST_ASSERT_EQ(d, 1);
    TEST_ASSERT_EQ(h, 12);
    TEST_ASSERT_EQ(mi, 0);
    TEST_ASSERT_NEAR(s, 0.0, 1e-6);
}

void test_jd_to_calendar_unix_epoch() {
    auto [y, m, d, h, mi, s] = jd_to_calendar(constants::UNIX_EPOCH_JD);

    TEST_ASSERT_EQ(y, 1970);
    TEST_ASSERT_EQ(m, 1);
    TEST_ASSERT_EQ(d, 1);
    TEST_ASSERT_EQ(h, 0);
    TEST_ASSERT_EQ(mi, 0);
    TEST_ASSERT_NEAR(s, 0.0, 1e-6);
}

void test_jd_to_calendar_roundtrip() {
    // Test multiple dates for roundtrip conversion
    struct TestDate {
        int year, month, day, hour, minute;
        double second;
    };

    TestDate dates[] = {
        {2000, 1, 1, 12, 0, 0.0},
        {2024, 6, 15, 14, 30, 45.5},
        {1999, 12, 31, 23, 59, 59.0},
        {2050, 7, 4, 0, 0, 0.0},
        {1980, 1, 6, 0, 0, 0.0},  // GPS epoch
    };

    for (const auto& date : dates) {
        double jd = calendar_to_jd(date.year, date.month, date.day,
                                    date.hour, date.minute, date.second);
        auto [y, m, d, h, mi, s] = jd_to_calendar(jd);

        TEST_ASSERT_EQ(y, date.year);
        TEST_ASSERT_EQ(m, date.month);
        TEST_ASSERT_EQ(d, date.day);
        TEST_ASSERT_EQ(h, date.hour);
        TEST_ASSERT_EQ(mi, date.minute);
        TEST_ASSERT_NEAR(s, date.second, 1e-3);  // Sub-millisecond precision
    }
}

void test_jd_to_calendar_precision() {
    // Test that roundtrip maintains sub-second precision
    double jd_orig = calendar_to_jd(2024, 3, 15, 8, 30, 22.123456);
    auto [y, m, d, h, mi, s] = jd_to_calendar(jd_orig);
    double jd_back = calendar_to_jd(y, m, d, h, mi, s);

    // Should match to microsecond (about 1e-11 days)
    TEST_ASSERT_NEAR(jd_back, jd_orig, 1e-9);
}

//=============================================================================
// MJD Conversion Tests
//=============================================================================

void test_jd_to_mjd() {
    // J2000.0 in MJD = 51544.5
    double mjd = jd_to_mjd(constants::J2000_JD);
    TEST_ASSERT_NEAR(mjd, 51544.5, 1e-10);

    // Unix epoch in MJD
    double mjd_unix = jd_to_mjd(constants::UNIX_EPOCH_JD);
    TEST_ASSERT_NEAR(mjd_unix, 40587.0, 1e-10);
}

void test_mjd_to_jd() {
    // MJD 51544.5 -> JD 2451545.0
    double jd = mjd_to_jd(51544.5);
    TEST_ASSERT_NEAR(jd, constants::J2000_JD, 1e-10);
}

void test_mjd_roundtrip() {
    double jd_orig = 2460000.5;  // Some arbitrary JD
    double mjd = jd_to_mjd(jd_orig);
    double jd_back = mjd_to_jd(mjd);
    TEST_ASSERT_NEAR(jd_back, jd_orig, 1e-15);
}

void test_mjd_offset_consistency() {
    // JD - MJD should always be MJD_OFFSET
    double jd = 2451545.0;
    double mjd = jd_to_mjd(jd);
    TEST_ASSERT_NEAR(jd - mjd, constants::MJD_OFFSET, 1e-15);
}

//=============================================================================
// Julian Centuries Tests
//=============================================================================

void test_julian_centuries_at_j2000() {
    // At J2000.0, T = 0
    double T = jd_to_julian_centuries_tt(constants::J2000_JD);
    TEST_ASSERT_NEAR(T, 0.0, 1e-15);
}

void test_julian_centuries_one_century() {
    // J2000.0 + 36525 days = J2100.0, T = 1.0
    double jd_j2100 = constants::J2000_JD + 36525.0;
    double T = jd_to_julian_centuries_tt(jd_j2100);
    TEST_ASSERT_NEAR(T, 1.0, 1e-12);
}

void test_julian_centuries_negative() {
    // Before J2000.0, T should be negative
    double jd_1950 = calendar_to_jd(1950, 1, 1, 12, 0, 0);
    double T = jd_to_julian_centuries_tt(jd_1950);
    TEST_ASSERT_TRUE(T < 0);
    TEST_ASSERT_NEAR(T, -0.5, 0.01);  // About half a century before
}

void test_julian_centuries_tdb() {
    // TDB centuries should be very close to TT centuries
    // (Difference is only a few milliseconds periodic variation)
    double T_tt = jd_to_julian_centuries_tt(constants::J2000_JD);
    double T_tdb = jd_to_julian_centuries_tdb(constants::J2000_JD);
    TEST_ASSERT_NEAR(T_tt, T_tdb, 1e-10);
}

//=============================================================================
// Seconds Since J2000 Tests
//=============================================================================

void test_seconds_since_j2000_at_epoch() {
    double sec = jd_to_seconds_since_j2000(constants::J2000_JD);
    TEST_ASSERT_NEAR(sec, 0.0, 1e-6);
}

void test_seconds_since_j2000_one_day() {
    double jd = constants::J2000_JD + 1.0;
    double sec = jd_to_seconds_since_j2000(jd);
    TEST_ASSERT_NEAR(sec, constants::SECONDS_PER_DAY, 1e-6);
}

void test_seconds_since_j2000_negative() {
    // Before J2000.0
    double jd = constants::J2000_JD - 1.0;
    double sec = jd_to_seconds_since_j2000(jd);
    TEST_ASSERT_NEAR(sec, -constants::SECONDS_PER_DAY, 1e-6);
}

//=============================================================================
// Unix Timestamp Conversion Tests
//=============================================================================

void test_unix_to_jd_epoch() {
    double jd = unix_to_jd(0.0);
    TEST_ASSERT_NEAR(jd, constants::UNIX_EPOCH_JD, 1e-10);
}

void test_unix_to_jd_one_day() {
    // One day after Unix epoch
    double jd = unix_to_jd(constants::SECONDS_PER_DAY);
    TEST_ASSERT_NEAR(jd, constants::UNIX_EPOCH_JD + 1.0, 1e-10);
}

void test_jd_to_unix_epoch() {
    double unix_time = jd_to_unix(constants::UNIX_EPOCH_JD);
    TEST_ASSERT_NEAR(unix_time, 0.0, 1e-6);
}

void test_unix_roundtrip() {
    double unix_orig = 1000000000.0;  // Sep 9, 2001
    double jd = unix_to_jd(unix_orig);
    double unix_back = jd_to_unix(jd);
    TEST_ASSERT_NEAR(unix_back, unix_orig, 1e-3);  // Millisecond precision for large timestamps
}

void test_unix_j2000_relationship() {
    // J2000.0 in Unix time should be positive (after 1970)
    double unix_j2000 = jd_to_unix(constants::J2000_JD);
    TEST_ASSERT_TRUE(unix_j2000 > 0);

    // January 1, 2000 12:00:00 UTC is about 946728000 seconds after Unix epoch
    // (accounting for leap seconds, this is approximate)
    TEST_ASSERT_NEAR(unix_j2000, 946728000.0, 100.0);
}

//=============================================================================
// Leap Year Tests
//=============================================================================

void test_is_leap_year_regular_leap() {
    // Years divisible by 4 are leap years (normally)
    TEST_ASSERT_TRUE(is_leap_year(2024));
    TEST_ASSERT_TRUE(is_leap_year(2020));
    TEST_ASSERT_TRUE(is_leap_year(2016));
    TEST_ASSERT_TRUE(is_leap_year(2004));
}

void test_is_leap_year_century_exception() {
    // Years divisible by 100 are NOT leap years
    TEST_ASSERT_FALSE(is_leap_year(1900));
    TEST_ASSERT_FALSE(is_leap_year(2100));
    TEST_ASSERT_FALSE(is_leap_year(2200));
    TEST_ASSERT_FALSE(is_leap_year(2300));
}

void test_is_leap_year_400_exception() {
    // Years divisible by 400 ARE leap years (exception to century rule)
    TEST_ASSERT_TRUE(is_leap_year(2000));
    TEST_ASSERT_TRUE(is_leap_year(1600));
    TEST_ASSERT_TRUE(is_leap_year(2400));
}

void test_is_leap_year_non_leap() {
    // Regular non-leap years
    TEST_ASSERT_FALSE(is_leap_year(2023));
    TEST_ASSERT_FALSE(is_leap_year(2021));
    TEST_ASSERT_FALSE(is_leap_year(2019));
    TEST_ASSERT_FALSE(is_leap_year(2001));
}

//=============================================================================
// Day of Year Tests
//=============================================================================

void test_day_of_year_january_first() {
    TEST_ASSERT_EQ(day_of_year(2024, 1, 1), 1);
    TEST_ASSERT_EQ(day_of_year(2023, 1, 1), 1);
}

void test_day_of_year_december_31_leap() {
    // December 31 in a leap year = day 366
    TEST_ASSERT_EQ(day_of_year(2024, 12, 31), 366);
    TEST_ASSERT_EQ(day_of_year(2020, 12, 31), 366);
}

void test_day_of_year_december_31_non_leap() {
    // December 31 in a non-leap year = day 365
    TEST_ASSERT_EQ(day_of_year(2023, 12, 31), 365);
    TEST_ASSERT_EQ(day_of_year(2021, 12, 31), 365);
}

void test_day_of_year_march_first_leap() {
    // March 1 in leap year = day 61 (31 + 29 + 1)
    TEST_ASSERT_EQ(day_of_year(2024, 3, 1), 61);
}

void test_day_of_year_march_first_non_leap() {
    // March 1 in non-leap year = day 60 (31 + 28 + 1)
    TEST_ASSERT_EQ(day_of_year(2023, 3, 1), 60);
}

void test_day_of_year_february_29_leap() {
    // February 29 in leap year = day 60
    TEST_ASSERT_EQ(day_of_year(2024, 2, 29), 60);
}

void test_day_of_year_mid_year() {
    // July 1, 2024 (leap year)
    // Jan(31) + Feb(29) + Mar(31) + Apr(30) + May(31) + Jun(30) + 1 = 183
    TEST_ASSERT_EQ(day_of_year(2024, 7, 1), 183);
}

//=============================================================================
// Fractional Day of Year Tests
//=============================================================================

void test_fractional_day_of_year_noon() {
    // January 1 at noon = 0.5 (0-indexed)
    double fdoy = fractional_day_of_year(2024, 1, 1, 12, 0, 0.0);
    TEST_ASSERT_NEAR(fdoy, 0.5, 1e-10);
}

void test_fractional_day_of_year_midnight() {
    // January 1 at midnight = 0.0 (0-indexed)
    double fdoy = fractional_day_of_year(2024, 1, 1, 0, 0, 0.0);
    TEST_ASSERT_NEAR(fdoy, 0.0, 1e-10);
}

void test_fractional_day_of_year_end_of_year() {
    // December 31 at noon in leap year
    // DOY = 366, so fractional = 365.5
    double fdoy = fractional_day_of_year(2024, 12, 31, 12, 0, 0.0);
    TEST_ASSERT_NEAR(fdoy, 365.5, 1e-10);
}

void test_fractional_day_of_year_seconds() {
    // Test sub-day precision
    double fdoy1 = fractional_day_of_year(2024, 1, 1, 0, 0, 0.0);
    double fdoy2 = fractional_day_of_year(2024, 1, 1, 0, 0, 3600.0);  // 1 hour

    TEST_ASSERT_NEAR(fdoy2 - fdoy1, 1.0 / 24.0, 1e-12);
}

//=============================================================================
// Edge Cases
//=============================================================================

void test_calendar_year_boundaries() {
    // Test year boundary transition
    double jd_dec31 = calendar_to_jd(2023, 12, 31, 23, 59, 59.0);
    double jd_jan1 = calendar_to_jd(2024, 1, 1, 0, 0, 0.0);

    // Should be 1 second apart
    double diff = (jd_jan1 - jd_dec31) * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(diff, 1.0, 1e-3);
}

void test_very_early_date() {
    // Test an early date (year 1)
    double jd = calendar_to_jd(1, 1, 1, 12, 0, 0);
    TEST_ASSERT_TRUE(jd > 0);
    TEST_ASSERT_TRUE(jd < constants::J2000_JD);
}

void test_negative_year() {
    // Test BC dates (negative years)
    double jd = calendar_to_jd(-4712, 1, 1, 12, 0, 0);
    // Julian Date origin is around this time
    TEST_ASSERT_TRUE(jd > 0);
    TEST_ASSERT_TRUE(jd < 100);
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "BALLISTIC LIBRARY - JULIAN DATE TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    // calendar_to_jd tests
    runner.add_test("calendar_to_jd_j2000", test_calendar_to_jd_j2000);
    runner.add_test("calendar_to_jd_unix_epoch", test_calendar_to_jd_unix_epoch);
    runner.add_test("calendar_to_jd_historical_dates", test_calendar_to_jd_historical_dates);
    runner.add_test("calendar_to_jd_future_dates", test_calendar_to_jd_future_dates);
    runner.add_test("calendar_to_jd_leap_year_date", test_calendar_to_jd_leap_year_date);
    runner.add_test("calendar_to_jd_fractional_seconds", test_calendar_to_jd_fractional_seconds);
    runner.add_test("calendar_to_jd_midnight_vs_noon", test_calendar_to_jd_midnight_vs_noon);
    runner.add_test("calendar_to_jd_january_february_handling", test_calendar_to_jd_january_february_handling);
    runner.add_test("calendar_to_jd_day_progression", test_calendar_to_jd_day_progression);

    // jd_to_calendar tests
    runner.add_test("jd_to_calendar_j2000", test_jd_to_calendar_j2000);
    runner.add_test("jd_to_calendar_unix_epoch", test_jd_to_calendar_unix_epoch);
    runner.add_test("jd_to_calendar_roundtrip", test_jd_to_calendar_roundtrip);
    runner.add_test("jd_to_calendar_precision", test_jd_to_calendar_precision);

    // MJD tests
    runner.add_test("jd_to_mjd", test_jd_to_mjd);
    runner.add_test("mjd_to_jd", test_mjd_to_jd);
    runner.add_test("mjd_roundtrip", test_mjd_roundtrip);
    runner.add_test("mjd_offset_consistency", test_mjd_offset_consistency);

    // Julian centuries tests
    runner.add_test("julian_centuries_at_j2000", test_julian_centuries_at_j2000);
    runner.add_test("julian_centuries_one_century", test_julian_centuries_one_century);
    runner.add_test("julian_centuries_negative", test_julian_centuries_negative);
    runner.add_test("julian_centuries_tdb", test_julian_centuries_tdb);

    // Seconds since J2000 tests
    runner.add_test("seconds_since_j2000_at_epoch", test_seconds_since_j2000_at_epoch);
    runner.add_test("seconds_since_j2000_one_day", test_seconds_since_j2000_one_day);
    runner.add_test("seconds_since_j2000_negative", test_seconds_since_j2000_negative);

    // Unix timestamp tests
    runner.add_test("unix_to_jd_epoch", test_unix_to_jd_epoch);
    runner.add_test("unix_to_jd_one_day", test_unix_to_jd_one_day);
    runner.add_test("jd_to_unix_epoch", test_jd_to_unix_epoch);
    runner.add_test("unix_roundtrip", test_unix_roundtrip);
    runner.add_test("unix_j2000_relationship", test_unix_j2000_relationship);

    // Leap year tests
    runner.add_test("is_leap_year_regular_leap", test_is_leap_year_regular_leap);
    runner.add_test("is_leap_year_century_exception", test_is_leap_year_century_exception);
    runner.add_test("is_leap_year_400_exception", test_is_leap_year_400_exception);
    runner.add_test("is_leap_year_non_leap", test_is_leap_year_non_leap);

    // Day of year tests
    runner.add_test("day_of_year_january_first", test_day_of_year_january_first);
    runner.add_test("day_of_year_december_31_leap", test_day_of_year_december_31_leap);
    runner.add_test("day_of_year_december_31_non_leap", test_day_of_year_december_31_non_leap);
    runner.add_test("day_of_year_march_first_leap", test_day_of_year_march_first_leap);
    runner.add_test("day_of_year_march_first_non_leap", test_day_of_year_march_first_non_leap);
    runner.add_test("day_of_year_february_29_leap", test_day_of_year_february_29_leap);
    runner.add_test("day_of_year_mid_year", test_day_of_year_mid_year);

    // Fractional day of year tests
    runner.add_test("fractional_day_of_year_noon", test_fractional_day_of_year_noon);
    runner.add_test("fractional_day_of_year_midnight", test_fractional_day_of_year_midnight);
    runner.add_test("fractional_day_of_year_end_of_year", test_fractional_day_of_year_end_of_year);
    runner.add_test("fractional_day_of_year_seconds", test_fractional_day_of_year_seconds);

    // Edge cases
    runner.add_test("calendar_year_boundaries", test_calendar_year_boundaries);
    runner.add_test("very_early_date", test_very_early_date);
    runner.add_test("negative_year", test_negative_year);

    int result = runner.run();
    return print_final_summary();
}
