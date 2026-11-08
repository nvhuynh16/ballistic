// Unit tests for LeapSecondTable, the table of TAI-UTC offsets (the leap seconds
// added to UTC). Values are the published IERS/USNO ones; whole-second offsets
// are checked exactly, the pre-1972 formula to 1e-7 s.
// Uses the embedded table and small temporary files written by the test.
// Tests: time/leap_seconds.hpp (LeapSecondTable class)

#include "../test_utils.hpp"
#include <ballistic/core/diagnostics.hpp>

#include <filesystem>
#include <fstream>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::time;

// =============================================================================
// Embedded Leap Seconds Tests
// =============================================================================

void test_embedded_leap_seconds() {
    TestRunner runner("EmbeddedLeapSeconds");

    runner.add_test("singleton_same_address", []() {
        auto& t1 = LeapSecondTable::instance();
        auto& t2 = LeapSecondTable::instance();
        TEST_ASSERT_EQ(&t1, &t2);
    });

    runner.add_test("has_embedded_entries", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        TEST_ASSERT(table.size() > 0);
        TEST_ASSERT(table.size() >= 28);  // At least 28 historical leap seconds
    });

    runner.add_test("not_loaded_from_file_initially", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        TEST_ASSERT_FALSE(table.is_loaded_from_file());
    });

    runner.add_test("tai_utc_before_1972", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        // Before 1972 UTC used fractional "rubber seconds": TAI-UTC follows the USNO
        // formula (8.0000820 s on 1970-01-01), not a whole number.
        double jd_1970 = calendar_to_jd(1970, 1, 1);
        TEST_ASSERT_NEAR(table.tai_utc(jd_1970), 8.0000820, 1e-7);
    });

    // Exact published offsets: 10 s at the start of 1972, 11 s after the first leap
    // second, 37 s from 2017-01-01 on.
    runner.add_test("tai_utc_at_1972_jan_1", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        double jd = calendar_to_jd(1972, 1, 1);
        int offset = table.tai_utc(jd);
        TEST_ASSERT_EQ(offset, 10);
    });

    runner.add_test("tai_utc_at_1972_jul_1", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        double jd = calendar_to_jd(1972, 7, 1);
        int offset = table.tai_utc(jd);
        TEST_ASSERT_EQ(offset, 11);
    });

    runner.add_test("tai_utc_at_2017_jan_1", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        double jd = calendar_to_jd(2017, 1, 1);
        int offset = table.tai_utc(jd);
        TEST_ASSERT_EQ(offset, 37);
    });

    runner.add_test("tai_utc_after_last_entry", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        // After 2017, still 37 (no new leap seconds)
        double jd = calendar_to_jd(2024, 6, 15);
        int offset = table.tai_utc(jd);
        TEST_ASSERT_EQ(offset, 37);
    });

    runner.add_test("tai_utc_boundary_just_before", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        // Just before 2017-01-01, should still be 36
        double jd = calendar_to_jd(2016, 12, 31, 23, 59, 59.0);
        int offset = table.tai_utc(jd);
        TEST_ASSERT_EQ(offset, 36);
    });

    runner.add_test("current_tai_utc", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        int current = table.current_tai_utc();
        TEST_ASSERT_EQ(current, 37);
    });

    runner.add_test("last_entry_jd_nonzero", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        double jd = table.last_entry_jd();
        TEST_ASSERT(jd > 0);
        // Should be approximately 2017-01-01
        double expected = calendar_to_jd(2017, 1, 1);
        TEST_ASSERT_NEAR(jd, expected, 1.0);
    });

    runner.run();
}

// =============================================================================
// File Loading Tests
// =============================================================================

void test_load_leap_seconds() {
    TestRunner runner("LoadLeapSeconds");

    // A missing file returns false and keeps the embedded table.
    runner.add_test("load_nonexistent_file", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        bool ok = table.load("nonexistent_leap_seconds.list");
        TEST_ASSERT_FALSE(ok);
        // Should still have embedded data
        TEST_ASSERT(table.size() > 0);
        TEST_ASSERT_FALSE(table.is_loaded_from_file());
    });

    runner.add_test("last_error_after_failure", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        table.load("nonexistent_file.list");
        std::string err = table.last_error();
        TEST_ASSERT(err.length() > 0);
    });

    // A well-formed IETF-style file (comments, '#@' expiry, trailing '# date' comments) loads.
    runner.add_test("load_wellformed_file", []() {
        auto& table = LeapSecondTable::instance();
        const auto p = std::filesystem::temp_directory_path() / "ballistic_leap_ok.list";
        std::ofstream(p) << "# comment\n#$\t 3929093563\n#@\t3991593600\n\n"
                            "2272060800\t10\t# 1 Jan 1972\n"
                            "3692217600\t37\t# 1 Jan 2017\n#h\tabc def\n";
        TEST_ASSERT_TRUE(table.load(p));
        TEST_ASSERT_TRUE(table.is_loaded_from_file());
        TEST_ASSERT_EQ(table.size(), static_cast<size_t>(2));
        table.reset_to_embedded();
        std::filesystem::remove(p);
    });

    // A non-comment line that does not parse refuses the whole file (it used to be skipped
    // silently, and a skipped row is a 1 s TAI-UTC error on every later date). The table
    // stays as it was and last_error() names the line.
    runner.add_test("malformed_line_refuses_the_file", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        const auto before = table.size();
        const auto p = std::filesystem::temp_directory_path() / "ballistic_leap_bad.list";
        for (const char* bad : {"2287785600 eleven\n", "x2287785600 11\n", "2287785600 11 12\n",
                                "#@ never\n"}) {
            std::ofstream(p) << "# header\n2272060800\t10\t# 1 Jan 1972\n" << bad
                             << "3692217600\t37\t# 1 Jan 2017\n";
            TEST_ASSERT_FALSE(table.load(p));
            TEST_ASSERT_FALSE(table.is_loaded_from_file());
            TEST_ASSERT_EQ(table.size(), before);
            TEST_ASSERT(table.last_error().find("line 3") != std::string::npos);
        }
        std::filesystem::remove(p);
    });

    runner.add_test("reset_to_embedded", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        TEST_ASSERT_FALSE(table.is_loaded_from_file());
        TEST_ASSERT(table.size() >= 28);
    });

    runner.run();
}

// =============================================================================
// Expiry Tests
// =============================================================================

void test_warning_callback() {
    TestRunner runner("Expiry");

    runner.add_test("embedded_table_has_the_iers_expiry", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        // The embedded copy of IERS leap-seconds.list says "File expires on 28 June 2027".

        TEST_ASSERT_NEAR(table.expiration_jd(), calendar_to_jd(2027, 6, 28), 1e-9);
    });

    // Past the expiry date the last offset (37 s) is held and a warning goes to the
    // diagnostics channel exactly once per process, however many lookups follow.
    runner.add_test("past_expiry_reports_once_through_the_channel", []() {
        auto& table = LeapSecondTable::instance();
        table.reset_to_embedded();
        ballistic::diagnostics::Sink::instance().reset_dedup();
        int reports = 0;
        ballistic::diagnostics::set_handler([&](const ballistic::diagnostics::Diagnostic& d) {
            if (d.message.find("leap-second table expires") != std::string::npos) ++reports;
        });
        const double later = calendar_to_jd(2030, 1, 1);
        TEST_ASSERT_EQ(table.tai_utc(later), 37.0);   // last value held
        (void)table.tai_utc(later + 10.0);
        TEST_ASSERT_EQ(table.tai_utc(calendar_to_jd(2026, 1, 1)), 37.0);
        ballistic::diagnostics::set_handler(nullptr);
        TEST_ASSERT_EQ(reports, 1);
    });

    runner.run();
}

// =============================================================================
// Global accessor Tests
// =============================================================================

void test_global_accessor() {
    TestRunner runner("GlobalAccessor");

    runner.add_test("get_leap_second_table", []() {
        auto& table = get_leap_second_table();
        auto& instance = LeapSecondTable::instance();
        TEST_ASSERT_EQ(&table, &instance);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "LeapSecondTable Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_embedded_leap_seconds();
    test_load_leap_seconds();
    test_warning_callback();
    test_global_accessor();

    return print_final_summary();
}
