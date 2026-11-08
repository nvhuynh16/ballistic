// Unit tests for epoch.hpp: the Epoch class, conversions between time scales, and
// leap seconds. Time scales: UTC (civil time, with leap seconds), TAI (atomic
// time, UTC + leap seconds), TT (Terrestrial Time = TAI + 32.184 s), GPS time
// (TAI - 19 s), UT1 (Earth-rotation time, UTC + DUT1) and TDB (Barycentric
// Dynamical Time, within ~2 ms of TT). Expected values are definitions or the
// published IERS leap-second table, so most tolerances are at floating-point level.

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/time/epoch.hpp>
#include <ballistic/core/constants.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::time;
using namespace ballistic::test;

static TestRunner runner("Epoch Tests");

//=============================================================================
// Epoch Constructor Tests
//=============================================================================

void test_epoch_default_constructor() {
    Epoch e;
    TEST_ASSERT_NEAR(e.tt_jd(), constants::J2000_JD, 1e-10);
}

void test_epoch_from_jd_tt() {
    double jd = 2451545.5;  // 0.5 days after J2000.0
    Epoch e = Epoch::from_jd(jd, TimeScale::TT);
    TEST_ASSERT_NEAR(e.tt_jd(), jd, 1e-15);
}

void test_epoch_from_jd_tai() {
    // TAI = TT - 32.184s
    double tai_jd = constants::J2000_JD;
    Epoch e = Epoch::from_jd(tai_jd, TimeScale::TAI);

    // TT should be TAI + 32.184 seconds
    double expected_tt = tai_jd + constants::TT_TAI_OFFSET / constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(e.tt_jd(), expected_tt, 1e-10);
}

void test_epoch_from_utc() {
    // January 1, 2024, 12:00:00 UTC
    Epoch e = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    // Check calendar round-trip
    auto [y, m, d, h, mi, s] = e.to_calendar(TimeScale::UTC);
    TEST_ASSERT_EQ(y, 2024);
    TEST_ASSERT_EQ(m, 1);
    TEST_ASSERT_EQ(d, 1);
    TEST_ASSERT_EQ(h, 12);
    TEST_ASSERT_EQ(mi, 0);
    TEST_ASSERT_NEAR(s, 0.0, 1e-3);
}

void test_epoch_from_mjd() {
    // J2000.0 in MJD = 51544.5
    Epoch e = Epoch::from_mjd(51544.5, TimeScale::TT);
    TEST_ASSERT_NEAR(e.tt_jd(), constants::J2000_JD, 1e-10);
}

void test_epoch_from_unix() {
    // Unix epoch = 0
    Epoch e = Epoch::from_unix(0.0);

    auto [y, m, d, h, mi, s] = e.to_calendar(TimeScale::UTC);
    TEST_ASSERT_EQ(y, 1970);
    TEST_ASSERT_EQ(m, 1);
    TEST_ASSERT_EQ(d, 1);
    TEST_ASSERT_EQ(h, 0);
}

void test_epoch_j2000_factory() {
    Epoch e = Epoch::j2000();
    TEST_ASSERT_NEAR(e.tt_jd(), constants::J2000_JD, 1e-15);
    TEST_ASSERT_NEAR(e.seconds_since_j2000(), 0.0, 1e-6);
}

//=============================================================================
// Time Scale Conversion Tests
//=============================================================================

void test_epoch_tt_tai_offset() {
    // TT = TAI + 32.184 seconds (exact by definition)
    Epoch e = Epoch::j2000();

    double tt_jd = e.tt_jd();
    double tai_jd = e.tai_jd();

    double diff_seconds = (tt_jd - tai_jd) * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(diff_seconds, constants::TT_TAI_OFFSET, 1e-3);  // Relaxed for roundtrip precision
}

void test_epoch_utc_tai_offset_2020() {
    // After January 1, 2017: TAI - UTC = 37 seconds
    Epoch e = Epoch::from_utc(2020, 6, 15, 12, 0, 0);

    double utc_jd = e.utc_jd();
    double tai_jd = e.tai_jd();

    double diff_seconds = (tai_jd - utc_jd) * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(diff_seconds, 37.0, 1e-6);
}

void test_epoch_utc_tai_offset_2010() {
    // After January 1, 2009: TAI - UTC = 34 seconds
    Epoch e = Epoch::from_utc(2010, 6, 15, 12, 0, 0);

    double utc_jd = e.utc_jd();
    double tai_jd = e.tai_jd();

    double diff_seconds = (tai_jd - utc_jd) * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(diff_seconds, 34.0, 1e-3);
}

void test_epoch_utc_tai_offset_1980() {
    // After January 1, 1980: TAI - UTC = 19 seconds
    Epoch e = Epoch::from_utc(1980, 6, 15, 12, 0, 0);

    double utc_jd = e.utc_jd();
    double tai_jd = e.tai_jd();

    double diff_seconds = (tai_jd - utc_jd) * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(diff_seconds, 19.0, 1e-3);
}

void test_epoch_utc_tai_offset_pre1972() {
    // 1961-1972: the USNO rubber-second formula; 1968-02-01 12:00 UTC:
    // 4.2131700 + (39887.5 - 39126) * 0.002592 = 6.1869780 s.
    Epoch e = Epoch::from_utc(1968, 2, 1, 12, 0, 0);
    TEST_ASSERT_NEAR(Epoch::from_tai(1968, 2, 1, 12, 0, 6.1869780) - e, 0.0, 1e-6);
    // Before 1961 UTC did not exist: refused rather than guessed.
    TEST_ASSERT_THROWS(Epoch::from_utc(1960, 1, 1, 12, 0, 0), ballistic::DataCoverageError);
}

void test_epoch_gps_tai_offset() {
    // GPS = TAI - 19 seconds (fixed offset)
    Epoch e = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    double gps_jd = e.gps_jd();
    double tai_jd = e.tai_jd();

    double diff_seconds = (gps_jd - tai_jd) * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(diff_seconds, constants::GPS_TAI_OFFSET, 1e-3);
}

void test_epoch_tdb_tt_periodic() {
    // TDB - TT has a periodic variation of ~1.7ms maximum
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 7, 1, 12, 0, 0);

    double diff1 = (e1.tdb_jd() - e1.tt_jd()) * constants::SECONDS_PER_DAY;
    double diff2 = (e2.tdb_jd() - e2.tt_jd()) * constants::SECONDS_PER_DAY;

    // Both should be small (< 2ms)
    TEST_ASSERT_TRUE(std::abs(diff1) < 0.002);
    TEST_ASSERT_TRUE(std::abs(diff2) < 0.002);
}

void test_epoch_ut1_with_dut1() {
    Epoch e = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    // Set DUT1 = +0.3 seconds (UT1 is ahead of UTC)
    e.set_dut1(0.3);

    double utc_jd = e.utc_jd();
    double ut1_jd = e.ut1_jd();

    double diff_seconds = (ut1_jd - utc_jd) * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(diff_seconds, 0.3, 1e-3);
}

void test_epoch_ut1_default_zero() {
    Epoch e = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    // Default DUT1 = 0
    TEST_ASSERT_NEAR(e.dut1(), 0.0, 1e-15);

    // UT1 should equal UTC when DUT1 = 0
    double utc_jd = e.utc_jd();
    double ut1_jd = e.ut1_jd();
    TEST_ASSERT_NEAR(ut1_jd, utc_jd, 1e-15);
}

//=============================================================================
// Epoch Arithmetic Tests
//=============================================================================

void test_epoch_addition() {
    Epoch e = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
    Epoch e2 = e + 3600.0;  // Add 1 hour

    // Verify that the difference is exactly 1 hour
    double diff = e2 - e;
    TEST_ASSERT_NEAR(diff, 3600.0, 1e-3);
}

void test_epoch_subtraction_seconds() {
    Epoch e = Epoch::from_utc(2024, 1, 1, 1, 0, 0);
    Epoch e2 = e - 3600.0;  // Subtract 1 hour

    auto [y, m, d, h, mi, s] = e2.to_calendar(TimeScale::UTC);
    TEST_ASSERT_EQ(h, 0);
}

void test_epoch_difference() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 1, 1, 1, 0, 0);

    double diff = e2 - e1;
    TEST_ASSERT_NEAR(diff, 3600.0, 1e-3);
}

void test_epoch_compound_addition() {
    Epoch e = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
    e += 7200.0;  // Add 2 hours

    auto [y, m, d, h, mi, s] = e.to_calendar(TimeScale::UTC);
    TEST_ASSERT_EQ(h, 2);
}

void test_epoch_compound_subtraction() {
    Epoch e = Epoch::from_utc(2024, 1, 1, 2, 0, 0);
    e -= 3600.0;  // Subtract 1 hour

    auto [y, m, d, h, mi, s] = e.to_calendar(TimeScale::UTC);
    TEST_ASSERT_EQ(h, 1);
}

void test_epoch_day_addition() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);
    Epoch e2 = e1 + constants::SECONDS_PER_DAY;  // Add 1 day

    auto [y, m, d, h, mi, s] = e2.to_calendar(TimeScale::UTC);
    TEST_ASSERT_EQ(d, 2);
}

void test_epoch_year_boundary() {
    Epoch e1 = Epoch::from_utc(2023, 12, 31, 23, 0, 0);
    Epoch e2 = e1 + 3600.0;  // Add 1 hour (crosses year boundary)

    auto [y, m, d, h, mi, s] = e2.to_calendar(TimeScale::UTC);
    TEST_ASSERT_EQ(y, 2024);
    TEST_ASSERT_EQ(m, 1);
    TEST_ASSERT_EQ(d, 1);
    TEST_ASSERT_EQ(h, 0);
}

//=============================================================================
// Epoch Comparison Tests
//=============================================================================

void test_epoch_less_than() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 1, 1, 1, 0, 0);

    TEST_ASSERT_TRUE(e1 < e2);
    TEST_ASSERT_FALSE(e2 < e1);
}

void test_epoch_greater_than() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 1, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);

    TEST_ASSERT_TRUE(e1 > e2);
    TEST_ASSERT_FALSE(e2 > e1);
}

void test_epoch_less_equal() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 1, 1, 1, 0, 0);
    Epoch e3 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);

    TEST_ASSERT_TRUE(e1 <= e2);
    TEST_ASSERT_TRUE(e1 <= e3);
}

void test_epoch_greater_equal() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 1, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
    Epoch e3 = Epoch::from_utc(2024, 1, 1, 1, 0, 0);

    TEST_ASSERT_TRUE(e1 >= e2);
    TEST_ASSERT_TRUE(e1 >= e3);
}

void test_epoch_equality() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    TEST_ASSERT_TRUE(e1 == e2);
}

void test_epoch_inequality() {
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);
    Epoch e2 = Epoch::from_utc(2024, 1, 1, 12, 0, 1);  // 1 second later

    TEST_ASSERT_TRUE(e1 != e2);
}

//=============================================================================
// Convenience Method Tests
//=============================================================================

void test_epoch_seconds_since_j2000() {
    Epoch e = Epoch::j2000();
    TEST_ASSERT_NEAR(e.seconds_since_j2000(), 0.0, 1e-6);

    Epoch e2 = e + 86400.0;  // 1 day later
    TEST_ASSERT_NEAR(e2.seconds_since_j2000(), 86400.0, 1e-3);
}

void test_epoch_julian_centuries() {
    Epoch e = Epoch::j2000();
    TEST_ASSERT_NEAR(e.julian_centuries_tt(), 0.0, 1e-15);

    // 36525 days = 1 Julian century
    double days_per_century = 36525.0;
    Epoch e2 = e + days_per_century * constants::SECONDS_PER_DAY;
    TEST_ASSERT_NEAR(e2.julian_centuries_tt(), 1.0, 1e-10);
}

void test_epoch_unix_timestamp() {
    Epoch e = Epoch::from_unix(1000000000.0);
    TEST_ASSERT_NEAR(e.unix_timestamp(), 1000000000.0, 1.0);
}

void test_epoch_day_of_year() {
    // January 1 = day 0 (0-indexed for atmospheric models)
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);
    TEST_ASSERT_NEAR(e1.day_of_year(), 0.0, 1e-10);

    // January 2 at noon = day 1.5 (0-indexed)
    Epoch e2 = Epoch::from_utc(2024, 1, 2, 12, 0, 0);
    TEST_ASSERT_NEAR(e2.day_of_year(), 1.5, 1e-10);
}

void test_epoch_year() {
    Epoch e = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
    TEST_ASSERT_EQ(e.year(), 2024);
}

void test_epoch_to_calendar() {
    Epoch e = Epoch::from_utc(2024, 3, 15, 14, 30, 45.5);
    auto [y, m, d, h, mi, s] = e.to_calendar(TimeScale::UTC);

    TEST_ASSERT_EQ(y, 2024);
    TEST_ASSERT_EQ(m, 3);
    TEST_ASSERT_EQ(d, 15);
    TEST_ASSERT_EQ(h, 14);
    TEST_ASSERT_EQ(mi, 30);
    TEST_ASSERT_NEAR(s, 45.5, 1e-3);
}

//=============================================================================
// Leap Second Table Tests
//=============================================================================

void test_leapsecondtable_1972() {
    auto& table = LeapSecondTable::instance();

    // After July 1, 1972: TAI - UTC = 11
    double jd_1972_jul = calendar_to_jd(1972, 7, 15, 0, 0, 0);
    TEST_ASSERT_EQ(table.tai_utc(jd_1972_jul), 11);
}

void test_leapsecondtable_2017() {
    auto& table = LeapSecondTable::instance();

    // After January 1, 2017: TAI - UTC = 37
    double jd_2017 = calendar_to_jd(2017, 6, 1, 0, 0, 0);
    TEST_ASSERT_EQ(table.tai_utc(jd_2017), 37);
}

void test_leapsecondtable_future() {
    auto& table = LeapSecondTable::instance();

    // After 2017, use last known value (37)
    double jd_future = calendar_to_jd(2030, 1, 1, 0, 0, 0);
    TEST_ASSERT_EQ(table.tai_utc(jd_future), 37);
}

void test_leapsecondtable_pre1972() {
    auto& table = LeapSecondTable::instance();

    // 1961-1972: USNO rubber-second formulas (SOFA iauDat); before 1961 UTC
    // did not exist and the table refuses.
    TEST_ASSERT_NEAR(table.tai_utc(calendar_to_jd(1968, 2, 1)), 6.1856820, 1e-7);
    TEST_ASSERT_NEAR(table.tai_utc(calendar_to_jd(1970, 1, 1)), 8.0000820, 1e-7);
    TEST_ASSERT_NEAR(table.tai_utc(calendar_to_jd(1961, 1, 1)), 1.4228180, 1e-7);
    TEST_ASSERT_THROWS(table.tai_utc(calendar_to_jd(1960, 1, 1)), ballistic::DataCoverageError);
}

//=============================================================================
// Edge Cases
//=============================================================================

void test_epoch_precision() {
    // Test that epoch maintains sub-second precision
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 12, 30, 45.123456);
    auto [y, m, d, h, mi, s] = e1.to_calendar(TimeScale::UTC);

    TEST_ASSERT_NEAR(s, 45.123456, 1e-3);  // Millisecond precision
}

void test_epoch_large_time_span() {
    // Test epoch 100 years from J2000
    Epoch e1 = Epoch::j2000();
    Epoch e2 = e1 + (100.0 * 365.25 * constants::SECONDS_PER_DAY);

    double diff_years = (e2 - e1) / (365.25 * constants::SECONDS_PER_DAY);
    TEST_ASSERT_NEAR(diff_years, 100.0, 0.01);
}

void test_epoch_negative_time_span() {
    // Test going backward in time
    Epoch e1 = Epoch::j2000();
    Epoch e2 = e1 - (30.0 * 365.25 * constants::SECONDS_PER_DAY);  // 30 years before

    TEST_ASSERT_TRUE(e2 < e1);
    TEST_ASSERT_TRUE(e2.year() < 2000);
}

void test_epoch_dut1_preservation() {
    // DUT1 should be preserved through arithmetic
    Epoch e1 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);
    e1.set_dut1(0.5);

    Epoch e2 = e1 + 3600.0;
    TEST_ASSERT_NEAR(e2.dut1(), 0.5, 1e-15);
}

//=============================================================================
// Main
//=============================================================================

// --- Exact representation (trap EPOCH-JD-JITTER) ---------------------------
// A single-double TT Julian date puts every label on a ~40 us grid, which
// corrupts derivatives and interpolation built on Epoch differences.


void test_epoch_microsecond_offsets_are_exact() {
    for (int year = 1950; year <= 2100; year += 10) {
        const Epoch t0 = Epoch::from_tt(year, 6, 15, 7, 31, 12.0);   // TT: defined for all years
        for (double dt : {1e-6, 3e-6, 0.5, 30.0, 86400.0 + 1e-6, -1e-6}) {
            const Epoch t1 = t0 + dt;
            TEST_ASSERT_NEAR(t1 - t0, dt, 1e-10);
        }
    }
}

void test_epoch_calendar_round_trip_is_clean() {
    // Exact calendar seconds come back exactly (a single-double Julian date
    // would return 59.99999).
    const auto [y, mo, d, h, mi, s] = Epoch::from_utc(2026, 1, 1, 10, 0, 0.0).to_calendar();
    TEST_ASSERT_EQ(y, 2026); TEST_ASSERT_EQ(mo, 1); TEST_ASSERT_EQ(d, 1);
    TEST_ASSERT_EQ(h, 10); TEST_ASSERT_EQ(mi, 0);
    TEST_ASSERT_NEAR(s, 0.0, 1e-9);
    const auto [y2, mo2, d2, h2, mi2, s2] = Epoch::from_utc(2024, 2, 29, 23, 59, 59.123456).to_calendar();
    TEST_ASSERT_EQ(y2, 2024); TEST_ASSERT_EQ(mo2, 2); TEST_ASSERT_EQ(d2, 29);
    TEST_ASSERT_EQ(h2, 23); TEST_ASSERT_EQ(mi2, 59);
    TEST_ASSERT_NEAR(s2, 59.123456, 1e-9);
}

void test_epoch_scale_offsets_are_exact() {
    // GPS - UTC = leap seconds - 19 s exactly; TT - TAI = 32.184 s exactly.
    const Epoch utc = Epoch::from_utc(2026, 1, 1, 12, 0, 0.0);
    const Epoch gps = Epoch::from_jd(utc.jd(TimeScale::GPS), TimeScale::GPS);
    TEST_ASSERT_NEAR(gps - utc, 0.0, 5e-5);   // through a double JD: 40 us class
    auto [gy, gmo, gd, gh, gmi, gs] = utc.to_calendar(TimeScale::GPS);
    TEST_ASSERT_EQ(gh, 12); TEST_ASSERT_EQ(gmi, 0);
    TEST_ASSERT_NEAR(gs, 18.0, 1e-9);         // 37 leap seconds - 19
    auto [ty, tmo, td, th, tmi, ts] = utc.to_calendar(TimeScale::TT);
    TEST_ASSERT_EQ(th, 12); TEST_ASSERT_EQ(tmi, 1);
    TEST_ASSERT_NEAR(ts, 9.184, 1e-9);        // 37 + 32.184 s
    (void)gy; (void)gmo; (void)gd; (void)ty; (void)tmo; (void)td;
}

void test_epoch_split_round_trip() {
    const Epoch e = Epoch::from_utc(2013, 11, 11, 0, 16, 3.25);
    const Epoch back = Epoch::from_tt_split(e.tt_day(), e.tt_seconds());
    TEST_ASSERT(back == e);
    TEST_ASSERT(e.tt_seconds() >= 0.0 && e.tt_seconds() < 86400.0);
}

int main() {
    // Runs without an Earth orientation table on purpose (synthetic or
    // self-consistent checks; pinned values assume zero UT1-UTC/polar motion).
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    std::cout << "\n========================================" << std::endl;
    std::cout << "BALLISTIC LIBRARY - EPOCH TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    // Constructor tests
    runner.add_test("epoch_default_constructor", test_epoch_default_constructor);
    runner.add_test("epoch_from_jd_tt", test_epoch_from_jd_tt);
    runner.add_test("epoch_from_jd_tai", test_epoch_from_jd_tai);
    runner.add_test("epoch_from_utc", test_epoch_from_utc);
    runner.add_test("epoch_from_mjd", test_epoch_from_mjd);
    runner.add_test("epoch_from_unix", test_epoch_from_unix);
    runner.add_test("epoch_j2000_factory", test_epoch_j2000_factory);

    // Time scale conversion tests
    runner.add_test("epoch_tt_tai_offset", test_epoch_tt_tai_offset);
    runner.add_test("epoch_utc_tai_offset_2020", test_epoch_utc_tai_offset_2020);
    runner.add_test("epoch_utc_tai_offset_2010", test_epoch_utc_tai_offset_2010);
    runner.add_test("epoch_utc_tai_offset_1980", test_epoch_utc_tai_offset_1980);
    runner.add_test("epoch_utc_tai_offset_pre1972", test_epoch_utc_tai_offset_pre1972);
    runner.add_test("epoch_gps_tai_offset", test_epoch_gps_tai_offset);
    runner.add_test("epoch_tdb_tt_periodic", test_epoch_tdb_tt_periodic);
    runner.add_test("epoch_ut1_with_dut1", test_epoch_ut1_with_dut1);
    runner.add_test("epoch_ut1_default_zero", test_epoch_ut1_default_zero);

    // Arithmetic tests
    runner.add_test("epoch_addition", test_epoch_addition);
    runner.add_test("epoch_subtraction_seconds", test_epoch_subtraction_seconds);
    runner.add_test("epoch_difference", test_epoch_difference);
    runner.add_test("epoch_compound_addition", test_epoch_compound_addition);
    runner.add_test("epoch_compound_subtraction", test_epoch_compound_subtraction);
    runner.add_test("epoch_day_addition", test_epoch_day_addition);
    runner.add_test("epoch_year_boundary", test_epoch_year_boundary);

    // Comparison tests
    runner.add_test("epoch_less_than", test_epoch_less_than);
    runner.add_test("epoch_greater_than", test_epoch_greater_than);
    runner.add_test("epoch_less_equal", test_epoch_less_equal);
    runner.add_test("epoch_greater_equal", test_epoch_greater_equal);
    runner.add_test("epoch_equality", test_epoch_equality);
    runner.add_test("epoch_inequality", test_epoch_inequality);

    // Convenience method tests
    runner.add_test("epoch_seconds_since_j2000", test_epoch_seconds_since_j2000);
    runner.add_test("epoch_julian_centuries", test_epoch_julian_centuries);
    runner.add_test("epoch_unix_timestamp", test_epoch_unix_timestamp);
    runner.add_test("epoch_day_of_year", test_epoch_day_of_year);
    runner.add_test("epoch_year", test_epoch_year);
    runner.add_test("epoch_to_calendar", test_epoch_to_calendar);

    // Leap second table tests
    runner.add_test("leapsecondtable_1972", test_leapsecondtable_1972);
    runner.add_test("leapsecondtable_2017", test_leapsecondtable_2017);
    runner.add_test("leapsecondtable_future", test_leapsecondtable_future);
    runner.add_test("leapsecondtable_pre1972", test_leapsecondtable_pre1972);

    // Edge cases
    runner.add_test("epoch_precision", test_epoch_precision);
    runner.add_test("epoch_large_time_span", test_epoch_large_time_span);
    runner.add_test("epoch_negative_time_span", test_epoch_negative_time_span);
    runner.add_test("epoch_dut1_preservation", test_epoch_dut1_preservation);
    runner.add_test("epoch_microsecond_offsets_are_exact", test_epoch_microsecond_offsets_are_exact);
    runner.add_test("epoch_calendar_round_trip_is_clean", test_epoch_calendar_round_trip_is_clean);
    runner.add_test("epoch_scale_offsets_are_exact", test_epoch_scale_offsets_are_exact);
    runner.add_test("epoch_split_round_trip", test_epoch_split_round_trip);

    int result = runner.run();
    return print_final_summary();
}
