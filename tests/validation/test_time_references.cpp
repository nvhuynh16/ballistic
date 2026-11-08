// Time module reference validation tests
// Checks Julian Dates (JD; MJD = JD - 2400000.5), the leap-second table, the
// conversions between time scales (UTC, TAI = atomic time, TT = TAI + 32.184 s,
// GPS = TAI - 19 s, TDB = barycentric dynamical time) and the Earth orientation
// parameters (EOP: UT1-UTC and polar motion) against independently published values.
// Where a test subtracts two double-precision Julian dates near 2.45e6 days, the
// result carries ~50 us of rounding, so those tolerances are 1e-4 s.
// Data: tests/data/eop/finals2000A.all (IERS).
//
// References:
//   [1] Meeus, "Astronomical Algorithms" 2nd ed., Ch.7 (Julian Date)
//   [2] IERS Bulletin C 69 (leap-seconds.list)
//   [3] IAU 1991 Resolution A4 (TT = TAI + 32.184s)
//   [4] IS-GPS-200 (GPS = TAI - 19s)
//   [5] USNO Circular 179 (TDB-TT)
//   [6] IERS Technical Note 36, Ch.5 (EOP)

#include "../test_utils.hpp"

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::time;

// ============================================================================
// Suite 1: Julian Date — Meeus/USNO references
// ============================================================================

void suite_julian_date_references() {
    TestRunner runner("Julian Date References (Meeus Ch.7 / USNO)");

    // Meeus Ch.7, Example 7.a: 1957 Oct 4.81 (Sputnik launch)
    // JD = 2436116.31
    runner.add_test("meeus_sputnik_1957_oct_4_81", []() {
        // 0.81 day = 19h 26m 24s
        double jd = calendar_to_jd(1957, 10, 4, 19, 26, 24.0);
        TEST_ASSERT_NEAR(jd, 2436116.31, 1e-6);
    });

    // Meeus Ch.7, Example 7.b: 1988 Jan 27.0
    // JD = 2447187.5
    runner.add_test("meeus_1988_jan_27_0", []() {
        double jd = calendar_to_jd(1988, 1, 27);
        TEST_ASSERT_NEAR(jd, 2447187.5, 1e-9);
    });

    // Meeus Ch.7, Example 7.c: 1988 Jun 19.5
    // JD = 2447332.0
    runner.add_test("meeus_1988_jun_19_5", []() {
        double jd = calendar_to_jd(1988, 6, 19, 12, 0, 0.0);
        TEST_ASSERT_NEAR(jd, 2447332.0, 1e-9);
    });

    // Meeus Ch.7, Example 7.d: 333 Jan 27 12:00
    // Meeus gives JD = 1842713.0 using the Julian calendar.
    // Our code uses proleptic Gregorian throughout (B = 2 - A + A/4 always applied).
    // For year 333: A=3, B=2-3+0=-1, so proleptic Gregorian gives JD = 1842712.0
    // (1 day less than Julian calendar result). This is correct behavior.
    runner.add_test("meeus_333_jan_27_12h_proleptic_gregorian", []() {
        double jd = calendar_to_jd(333, 1, 27, 12, 0, 0.0);
        TEST_ASSERT_NEAR(jd, 1842712.0, 1e-9);
    });

    // USNO: 2024 Jan 1 00:00:00 UTC -> JD 2460310.5
    runner.add_test("usno_2024_jan_1", []() {
        double jd = calendar_to_jd(2024, 1, 1, 0, 0, 0.0);
        TEST_ASSERT_NEAR(jd, 2460310.5, 1e-9);
    });

    // J2000.0: 2000 Jan 1.5 TT -> JD 2451545.0
    runner.add_test("j2000_epoch", []() {
        double jd = calendar_to_jd(2000, 1, 1, 12, 0, 0.0);
        TEST_ASSERT_NEAR(jd, 2451545.0, 1e-9);
    });

    // Roundtrip: JD -> calendar -> JD for Meeus dates (Gregorian era)
    runner.add_test("roundtrip_meeus_dates", []() {
        double jds[] = {2436116.31, 2447187.5, 2447332.0, 2460310.5, 2451545.0};
        for (double jd : jds) {
            auto [y, m, d, h, min, sec] = jd_to_calendar(jd);
            double jd_back = calendar_to_jd(y, m, d, h, min, sec);
            TEST_ASSERT_NEAR(jd_back, jd, 1e-6);
        }
    });

    // MJD conversion: MJD = JD - 2400000.5
    runner.add_test("mjd_conversion", []() {
        double jd = 2451545.0;
        double mjd = jd_to_mjd(jd);
        TEST_ASSERT_NEAR(mjd, 51544.5, 1e-9);
        TEST_ASSERT_NEAR(mjd_to_jd(mjd), jd, 1e-9);
    });

    runner.run();
}

// ============================================================================
// Suite 2: Leap second table — complete IERS verification
// ============================================================================

void suite_leap_second_table() {
    TestRunner runner("Leap Second Table (IERS Bulletin C 69)");

    // Complete IERS leap second table: date -> TAI-UTC after that date
    // Source: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
    struct LeapEntry { int y, m, d; int tai_utc; };
    static const LeapEntry iers_table[] = {
        {1972, 1, 1, 10}, {1972, 7, 1, 11}, {1973, 1, 1, 12},
        {1974, 1, 1, 13}, {1975, 1, 1, 14}, {1976, 1, 1, 15},
        {1977, 1, 1, 16}, {1978, 1, 1, 17}, {1979, 1, 1, 18},
        {1980, 1, 1, 19}, {1981, 7, 1, 20}, {1982, 7, 1, 21},
        {1983, 7, 1, 22}, {1985, 7, 1, 23}, {1988, 1, 1, 24},
        {1990, 1, 1, 25}, {1991, 1, 1, 26}, {1992, 7, 1, 27},
        {1993, 7, 1, 28}, {1994, 7, 1, 29}, {1996, 1, 1, 30},
        {1997, 7, 1, 31}, {1999, 1, 1, 32}, {2006, 1, 1, 33},
        {2009, 1, 1, 34}, {2012, 7, 1, 35}, {2015, 7, 1, 36},
        {2017, 1, 1, 37},
    };

    // Verify entry count
    runner.add_test("entry_count_28", []() {
        auto& table = LeapSecondTable::instance();
        TEST_ASSERT_EQ(table.size(), size_t(28));
    });

    // Verify each entry returns correct TAI-UTC
    runner.add_test("all_28_entries_correct", []() {
        auto& table = LeapSecondTable::instance();
        for (const auto& e : iers_table) {
            double jd = calendar_to_jd(e.y, e.m, e.d);
            int got = table.tai_utc(jd);
            if (got != e.tai_utc) {
                std::ostringstream oss;
                oss << "At " << e.y << "-" << e.m << "-" << e.d
                    << ": expected TAI-UTC=" << e.tai_utc << ", got " << got;
                throw std::runtime_error(oss.str());
            }
        }
    });

    // Boundary test: day before each entry returns previous offset
    runner.add_test("day_before_transitions", []() {
        auto& table = LeapSecondTable::instance();
        for (int i = 1; i < 28; i++) {
            const auto& e = iers_table[i];
            // 1 second before the transition
            double jd_before = calendar_to_jd(e.y, e.m, e.d) - 1.0 / 86400.0;
            int got = table.tai_utc(jd_before);
            int expected = iers_table[i - 1].tai_utc;
            if (got != expected) {
                std::ostringstream oss;
                oss << "Before " << e.y << "-" << e.m << "-" << e.d
                    << ": expected TAI-UTC=" << expected << ", got " << got;
                throw std::runtime_error(oss.str());
            }
        }
    });

    // Current (latest) TAI-UTC
    runner.add_test("current_tai_utc_37", []() {
        auto& table = LeapSecondTable::instance();
        TEST_ASSERT_EQ(table.current_tai_utc(), 37);
    });

    // Pre-1972: USNO rubber-second formula (1970-01-01: 4.2131700 +
    // (40587 - 39126) * 0.002592 = 8.0000820 s)
    runner.add_test("pre_1972_rubber_seconds", []() {
        auto& table = LeapSecondTable::instance();
        double jd_1970 = calendar_to_jd(1970, 1, 1);
        TEST_ASSERT_NEAR(table.tai_utc(jd_1970), 8.0000820, 1e-7);
    });

    runner.run();
}

// ============================================================================
// Suite 3: TDB-TT reference values
// ============================================================================

void suite_tdb_tt() {
    TestRunner runner("TDB-TT References (USNO Circular 179)");

    // TDB-TT at J2000.0: verify sign and magnitude.
    // g ≈ 357.53° at T=0, so TDB-TT ≈ -70 to -80 μs (negative, small).
    // Note: JD subtraction at epoch ~2.45e6 has ~47 μs precision limit,
    // so we use a wide tolerance for the exact value.
    runner.add_test("tdb_tt_at_j2000_sign_and_magnitude", []() {
        Epoch j2000 = Epoch::j2000();
        double tdb_jd = j2000.tdb_jd();
        double tt_jd = j2000.tt_jd();
        double tdb_tt_sec = (tdb_jd - tt_jd) * 86400.0;
        // Should be small and negative at J2000.0
        TEST_ASSERT(tdb_tt_sec < 0.0);
        TEST_ASSERT(std::abs(tdb_tt_sec) < 0.001);  // < 1 ms
        // ~-70 to -80 μs (with JD precision uncertainty)
        TEST_ASSERT_NEAR(tdb_tt_sec, -7.5e-5, 5e-5);
    });

    // Physical bound: |TDB-TT| < 1.7 ms at all times
    // (the two-term approximation's maximum is 0.001657 + 0.000014 = 0.001671 s)
    runner.add_test("tdb_tt_physical_bound", []() {
        double epochs_jd[] = {
            calendar_to_jd(2000, 1, 1, 12, 0, 0.0),
            calendar_to_jd(2000, 7, 1, 12, 0, 0.0),
            calendar_to_jd(2010, 1, 15, 0, 0, 0.0),
            calendar_to_jd(2010, 7, 15, 0, 0, 0.0),
            calendar_to_jd(2024, 3, 20, 12, 0, 0.0),
            calendar_to_jd(2024, 9, 22, 12, 0, 0.0),
        };
        for (double jd : epochs_jd) {
            Epoch e(jd, TimeScale::TT);
            double tdb_tt = (e.tdb_jd() - e.tt_jd()) * 86400.0;
            TEST_ASSERT(std::abs(tdb_tt) < 0.0017);
        }
    });

    // TDB-TT has both positive and negative values over a year.
    // The annual oscillation should produce a range > 0.5 ms.
    // Maximum |TDB-TT| occurs near g ≈ 90° and 270° (roughly April and October).
    runner.add_test("tdb_tt_annual_oscillation", []() {
        // Sample monthly over a year and check for sign changes
        bool found_positive = false;
        bool found_negative = false;
        double max_val = -1e10, min_val = 1e10;
        for (int m = 1; m <= 12; m++) {
            Epoch e(calendar_to_jd(2020, m, 15, 12, 0, 0.0), TimeScale::TT);
            double tdb_tt = (e.tdb_jd() - e.tt_jd()) * 86400.0;
            if (tdb_tt > 0) found_positive = true;
            if (tdb_tt < 0) found_negative = true;
            if (tdb_tt > max_val) max_val = tdb_tt;
            if (tdb_tt < min_val) min_val = tdb_tt;
        }
        TEST_ASSERT(found_positive);
        TEST_ASSERT(found_negative);
        // Range should be > 0.5 ms (theoretical max range ~3.3 ms)
        TEST_ASSERT(max_val - min_val > 0.0005);
    });

    // Roundtrip: TDB -> TT -> TDB at 4 epochs
    runner.add_test("tdb_tt_roundtrip", []() {
        double test_jds[] = {
            2451545.0,   // J2000.0
            2455000.0,   // ~2009
            2459000.0,   // ~2020
            2460310.5,   // 2024-01-01  guard-exempt(PUB-DATE): data
        };
        for (double tdb_jd : test_jds) {
            Epoch from_tdb(tdb_jd, TimeScale::TDB);
            double tdb_back = from_tdb.tdb_jd();
            double err_us = std::abs(tdb_back - tdb_jd) * 86400.0e6;  // microseconds
            TEST_ASSERT(err_us < 1.0);  // < 1 μs roundtrip error
        }
    });

    runner.run();
}

// ============================================================================
// Suite 4: UTC-TT chain references
// ============================================================================

void suite_utc_tt_chain() {
    TestRunner runner("UTC-TT Chain References (IAU/GPS-ICD)");

    // TT - UTC = TAI-UTC + 32.184s
    // Note: JD subtraction at epoch ~2.45e6 has ~50 μs precision limit
    // due to double precision. Tolerance of 1e-4 s (100 μs) accounts for this.
    //
    // At epochs with known TAI-UTC:
    //   2024: TAI-UTC=37, TT-UTC=69.184
    //   2010: TAI-UTC=34, TT-UTC=66.184
    //   2000: TAI-UTC=32, TT-UTC=64.184
    //   1972 Jan 1: TAI-UTC=10, TT-UTC=42.184
    runner.add_test("tt_minus_utc_2024", []() {
        Epoch e = Epoch::from_utc(2024, 6, 15, 12, 0, 0.0);
        double tt_utc = (e.tt_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT_NEAR(tt_utc, 69.184, 1e-4);
    });

    runner.add_test("tt_minus_utc_2010", []() {
        Epoch e = Epoch::from_utc(2010, 6, 15, 12, 0, 0.0);
        double tt_utc = (e.tt_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT_NEAR(tt_utc, 66.184, 1e-4);
    });

    runner.add_test("tt_minus_utc_2000", []() {
        Epoch e = Epoch::from_utc(2000, 6, 15, 12, 0, 0.0);
        double tt_utc = (e.tt_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT_NEAR(tt_utc, 64.184, 1e-4);
    });

    runner.add_test("tt_minus_utc_1972", []() {
        Epoch e = Epoch::from_utc(1972, 1, 15, 12, 0, 0.0);
        double tt_utc = (e.tt_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT_NEAR(tt_utc, 42.184, 1e-4);
    });

    // GPS epoch: 1980-01-06 00:00:00 UTC
    // GPS = TAI - 19s, and TAI-UTC was 19 at GPS epoch, so GPS-UTC = 0
    runner.add_test("gps_utc_at_gps_epoch", []() {
        Epoch e = Epoch::from_utc(1980, 1, 6, 0, 0, 0.0);
        double gps_utc = (e.gps_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT_NEAR(gps_utc, 0.0, 1e-4);
    });

    // GPS-UTC in 2024: TAI-UTC=37, GPS=TAI-19, so GPS-UTC = 37-19 = 18s
    runner.add_test("gps_utc_2024", []() {
        Epoch e = Epoch::from_utc(2024, 6, 15, 12, 0, 0.0);
        double gps_utc = (e.gps_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT_NEAR(gps_utc, 18.0, 1e-4);
    });

    // TAI-TT is constant: -32.184s (by definition, IAU 1991 Res A4)
    // JD precision limits accuracy to ~50 μs
    runner.add_test("tai_tt_constant", []() {
        double test_jds[] = {2451545.0, 2455000.0, 2460310.5};
        for (double jd : test_jds) {
            Epoch e(jd, TimeScale::TT);
            double tai_tt = (e.tai_jd() - e.tt_jd()) * 86400.0;
            TEST_ASSERT_NEAR(tai_tt, -32.184, 1e-4);
        }
    });

    runner.run();
}

// ============================================================================
// Suite 5: EOP real data (conditional)
// ============================================================================

void suite_eop_data() {
    TestRunner runner("EOP Data (committed finals2000A.all)");

    runner.add_test("eop_load_and_sanity", []() {
        // The committed IERS file, found through the project root so the cell
        // cannot pass without running.
        const std::filesystem::path p = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
        if (!std::filesystem::exists(p)) TEST_SKIP("file not found: " << p);
        auto& eop = EOPData::instance();
        TEST_ASSERT(eop.load(p.string()));

        // Should have > 1000 records (decades of daily data)
        TEST_ASSERT(eop.size() > 1000);

        // Sanity at MJD 60000 (~2023-Feb-25): |UT1-UTC| < 0.9 s by definition,
        // polar motion |xp|, |yp| < 0.5" (typically < 0.3").
        TEST_ASSERT(eop.in_range(60000.0));
        TEST_ASSERT(std::abs(eop.ut1_utc(60000.0)) < 0.9);
        TEST_ASSERT(std::abs(eop.polar_x(60000.0)) < 0.5);
        TEST_ASSERT(std::abs(eop.polar_y(60000.0)) < 0.5);
    });

    runner.run();
}

// ============================================================================
// Suite 6: Boundary & cross-validation tests
// ============================================================================

void suite_boundaries() {
    TestRunner runner("Boundary & Cross-Validation Tests");

    // Pre-1972 boundary: the USNO rubber-second formula, not a flat 10 s.
    // 1971-12-31 23:59:59 UTC: TAI-UTC = 4.2131700 + (41316.999988 - 39126)
    // x 0.002592 = 9.892242 s, so TT-UTC = 42.076242 s (UTC then stepped by
    // 0.107758 s to exactly 10 s on 1972-01-01).
    runner.add_test("pre_1972_tt_utc", []() {
        Epoch e = Epoch::from_utc(1971, 12, 31, 23, 59, 59.0);
        const auto [y, mo, d, h, mi, s] = e.to_calendar(TimeScale::TT);
        TEST_ASSERT_EQ(y, 1972); TEST_ASSERT_EQ(h, 0); TEST_ASSERT_EQ(mi, 0);
        TEST_ASSERT_NEAR(s, 41.076242, 1e-5);    // 23:59:59 + 42.076242 s
    });

    // 1972-Jul-1: TAI-UTC jumps from 10 to 11
    runner.add_test("1972_jul_1_transition", []() {
        auto& table = LeapSecondTable::instance();
        double jd_before = calendar_to_jd(1972, 6, 30, 23, 59, 59.0);
        double jd_after = calendar_to_jd(1972, 7, 1, 0, 0, 0.0);
        TEST_ASSERT_EQ(table.tai_utc(jd_before), 10);
        TEST_ASSERT_EQ(table.tai_utc(jd_after), 11);
    });

    // GPS epoch cross-validation: GPS time = UTC at 1980-01-06 00:00:00 UTC
    // because TAI-UTC=19 and GPS=TAI-19 at that moment
    runner.add_test("gps_epoch_cross_validation", []() {
        Epoch e = Epoch::from_utc(1980, 1, 6, 0, 0, 0.0);
        double diff = std::abs(e.gps_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT(diff < 1e-4);
    });

    // GPS week 1000: 1999-03-22 00:00:00 UTC
    // TAI-UTC = 32 (after 1999-Jan-1 leap second)
    // GPS-UTC = TAI-UTC - 19 = 13s
    runner.add_test("gps_week_1000", []() {
        Epoch e = Epoch::from_utc(1999, 3, 22, 0, 0, 0.0);
        double gps_utc = (e.gps_jd() - e.utc_jd()) * 86400.0;
        TEST_ASSERT_NEAR(gps_utc, 13.0, 1e-4);
    });

    // Unix epoch: 1970-01-01 00:00:00 UTC -> JD 2440587.5
    runner.add_test("unix_epoch", []() {
        double jd = unix_to_jd(0.0);
        TEST_ASSERT_NEAR(jd, 2440587.5, 1e-9);
        TEST_ASSERT_NEAR(jd_to_unix(2440587.5), 0.0, 1e-6);
    });

    // Day-of-year: known dates
    runner.add_test("day_of_year_known", []() {
        TEST_ASSERT_EQ(day_of_year(2024, 1, 1), 1);
        TEST_ASSERT_EQ(day_of_year(2024, 2, 29), 60);
        TEST_ASSERT_EQ(day_of_year(2024, 12, 31), 366);
        TEST_ASSERT_EQ(day_of_year(2023, 12, 31), 365);
    });

    // Leap year checks
    runner.add_test("leap_year_rules", []() {
        TEST_ASSERT_TRUE(is_leap_year(2024));   // divisible by 4
        TEST_ASSERT_FALSE(is_leap_year(2023));  // not divisible by 4
        TEST_ASSERT_FALSE(is_leap_year(1900));  // divisible by 100 but not 400
        TEST_ASSERT_TRUE(is_leap_year(2000));   // divisible by 400
    });

    runner.run();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "Time Module Reference Validation Tests" << std::endl;
    std::cout << "=======================================" << std::endl;

    suite_julian_date_references();
    suite_leap_second_table();
    suite_tdb_tt();
    suite_utc_tt_chain();
    suite_eop_data();
    suite_boundaries();

    return print_final_summary();
}
