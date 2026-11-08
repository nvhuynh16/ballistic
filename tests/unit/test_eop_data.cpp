// Unit tests for the Earth Orientation Parameters (EOP) table
// (time/eop_data.hpp). EOP describe how the real Earth's rotation departs from
// the model: polar motion (x, y, in arcseconds) and UT1-UTC (the difference
// between Earth-rotation time and atomic civil time, in seconds). They are
// read from IERS finals2000A files. Cells use small synthetic files in that
// fixed-column format (values chosen so interpolation results are exact) and
// check: loading, linear interpolation, the policy outside the table (edge
// value held for up to 30 days with one warning, then an error), the
// requirement to opt in to zero EOP when no table is loaded, UT1-UTC across a
// leap second, and blank rows past the IERS predictions.

#include "../test_utils.hpp"
#include <ballistic/time/eop_data.hpp>
#include <ballistic/core/diagnostics.hpp>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::time;

// =============================================================================
// Singleton Tests
// =============================================================================

void test_eop_singleton() {
    TestRunner runner("EOPSingleton");

    runner.add_test("instance_same_address", []() {
        auto& inst1 = EOPData::instance();
        auto& inst2 = EOPData::instance();
        TEST_ASSERT_EQ(&inst1, &inst2);
    });

    runner.add_test("initial_state_not_loaded", []() {
        // Clear to ensure clean state
        EOPData::instance().clear();
        TEST_ASSERT_FALSE(EOPData::instance().is_loaded());
        TEST_ASSERT_EQ(EOPData::instance().size(), static_cast<size_t>(0));
    });

    runner.run();
}

// =============================================================================
// Load/Clear Tests
// =============================================================================

void test_eop_load_clear() {
    TestRunner runner("EOPLoadClear");

    runner.add_test("load_nonexistent_file", []() {
        EOPData::instance().clear();
        bool result = EOPData::instance().load("nonexistent_file.data");
        TEST_ASSERT_FALSE(result);
        TEST_ASSERT_FALSE(EOPData::instance().is_loaded());
    });

    runner.add_test("clear_resets_state", []() {
        EOPData::instance().clear();
        TEST_ASSERT_FALSE(EOPData::instance().is_loaded());
        TEST_ASSERT_EQ(EOPData::instance().size(), static_cast<size_t>(0));
    });

    runner.add_test("empty_data_requires_a_policy", []() {
        EOPData::instance().clear();
        // Default: no table is an error, not silent zeros.
        TEST_ASSERT_THROWS(EOPData::instance().ut1_utc(58000.0), DataCoverageError);
        // Explicit opt-in: zeros.
        EOPData::instance().set_policy(EopPolicy::AssumeZero);
        double ut1 = EOPData::instance().ut1_utc(58000.0);
        TEST_ASSERT_NEAR(ut1, 0.0, 1e-15);

        double px = EOPData::instance().polar_x(58000.0);
        TEST_ASSERT_NEAR(px, 0.0, 1e-15);

        double py = EOPData::instance().polar_y(58000.0);
        TEST_ASSERT_NEAR(py, 0.0, 1e-15);
        EOPData::instance().set_policy(EopPolicy::Require);
    });

    runner.run();
}

// =============================================================================
// Interpolation Tests (with synthetic data)
// =============================================================================

static std::string make_line(double mjd, double x, double y, double ut1) {
    // One finals2000A row: fields at fixed columns (0-indexed below).
    std::string line(80, ' ');
    // MJD (Modified Julian Date) at 7-14
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%8.1f", mjd);
    line.replace(7, 8, buf);
    // x flag at 16 ('I' = IERS final value)
    line[16] = 'I';
    // x [arcsec] at 17-26
    std::snprintf(buf, sizeof(buf), "%10.6f", x);
    line.replace(17, 10, buf);
    // y flag at 37
    line[37] = 'I';
    // y [arcsec] at 38-47
    std::snprintf(buf, sizeof(buf), "%10.6f", y);
    line.replace(38, 10, buf);
    // UT1-UTC flag at 57
    line[57] = 'I';
    // UT1-UTC [s] at 58-67
    std::snprintf(buf, sizeof(buf), "%10.7f", ut1);
    line.replace(58, 10, buf);
    return line;
}

// Helper: create a temporary EOP file with known data
static std::string create_temp_eop_file(const std::string& path) {
    // Five daily rows, MJD 58000-58004: x, y and UT1-UTC each rise by 0.010
    // per day from 0.100 / 0.200 / 0.300 (column layout in make_line).
    std::ofstream file(path);

    file << make_line(58000.0, 0.100000, 0.200000, 0.3000000) << "\n";
    file << make_line(58001.0, 0.110000, 0.210000, 0.3100000) << "\n";
    file << make_line(58002.0, 0.120000, 0.220000, 0.3200000) << "\n";
    file << make_line(58003.0, 0.130000, 0.230000, 0.3300000) << "\n";
    file << make_line(58004.0, 0.140000, 0.240000, 0.3400000) << "\n";

    file.close();
    return path;
}

void test_eop_interpolation() {
    TestRunner runner("EOPInterpolation");

    // Create temp file
    std::string temp_path = "test_eop_temp.data";
    create_temp_eop_file(temp_path);

    runner.add_test("load_valid_file", [&]() {
        EOPData::instance().clear();
        bool ok = EOPData::instance().load(temp_path);
        TEST_ASSERT_TRUE(ok);
        TEST_ASSERT_TRUE(EOPData::instance().is_loaded());
        TEST_ASSERT_EQ(EOPData::instance().size(), static_cast<size_t>(5));
    });

    runner.add_test("exact_point_interpolation", [&]() {
        // At exact data point, should return exact value
        double x = EOPData::instance().polar_x(58000.0);
        TEST_ASSERT_NEAR(x, 0.100, 0.001);
    });

    runner.add_test("midpoint_interpolation", [&]() {
        // At midpoint, should be linear interpolation
        double x = EOPData::instance().polar_x(58000.5);
        TEST_ASSERT_NEAR(x, 0.105, 0.001);  // Midpoint between 0.100 and 0.110

        double ut1 = EOPData::instance().ut1_utc(58001.5);
        TEST_ASSERT_NEAR(ut1, 0.315, 0.001);  // Midpoint between 0.310 and 0.320
    });

    runner.add_test("held_within_30_days_reported_once", [&]() {
        diagnostics::Sink::instance().reset_dedup();
        int reports = 0;
        diagnostics::set_handler([&](const diagnostics::Diagnostic& d) {
            if (d.message.find("beyond the loaded table") != std::string::npos) ++reports;
        });
        const EopSample lo = EOPData::instance().lookup(57990.0);   // 10 d before
        const EopSample hi = EOPData::instance().lookup(58030.0);   // 26 d after
        (void)EOPData::instance().lookup(58031.0);
        diagnostics::set_handler(nullptr);
        TEST_ASSERT_NEAR(lo.xp_arcsec, 0.100, 1e-9);
        TEST_ASSERT_NEAR(hi.xp_arcsec, 0.140, 1e-9);
        TEST_ASSERT(hi.coverage == EopCoverage::Held);
        TEST_ASSERT_NEAR(hi.days_beyond, 26.0, 1e-9);
        TEST_ASSERT_EQ(reports, 1);
    });

    runner.add_test("refused_beyond_30_days", [&]() {
        // More than 30 days outside the table (either side): an error, not
        // the edge value held indefinitely.
        TEST_ASSERT_THROWS(EOPData::instance().polar_x(57000.0), DataCoverageError);
        TEST_ASSERT_THROWS(EOPData::instance().lookup(58034.5), DataCoverageError);
    });

    runner.add_test("mjd_range", [&]() {
        auto [lo, hi] = EOPData::instance().mjd_range();
        TEST_ASSERT_NEAR(lo, 58000.0, 0.1);
        TEST_ASSERT_NEAR(hi, 58004.0, 0.1);
    });

    runner.add_test("in_range", [&]() {
        TEST_ASSERT_TRUE(EOPData::instance().in_range(58002.0));
        TEST_ASSERT_FALSE(EOPData::instance().in_range(57000.0));
        TEST_ASSERT_FALSE(EOPData::instance().in_range(60000.0));
    });

    runner.add_test("polar_motion_rad_conversion", [&]() {
        auto [xr, yr] = EOPData::instance().polar_motion_rad(58000.0);
        double x_arcsec = EOPData::instance().polar_x(58000.0);
        double expected_xr = x_arcsec * constants::ARCSEC_TO_RAD;
        TEST_ASSERT_NEAR(xr, expected_xr, 1e-15);
    });

    runner.add_test("ut1_utc_accessor", [&]() {
        double ut1 = EOPData::instance().ut1_utc(58002.0);
        TEST_ASSERT_NEAR(ut1, 0.320, 0.001);
    });

    runner.add_test("lod_default_zero", [&]() {
        // LOD column may not be in our test data (line < 86 chars)
        // Should default to 0
        double lod = EOPData::instance().lod(58000.0);
        TEST_ASSERT_NEAR(lod, 0.0, 0.001);
    });

    runner.add_test("get_record", [&]() {
        auto record = EOPData::instance().get_record(58001.0);
        TEST_ASSERT(record.has_value());
        TEST_ASSERT_NEAR(record->mjd, 58001.0, 0.1);
    });

    runner.add_test("convenience_functions", [&]() {
        double ut1 = time::get_ut1_utc(58000.0);
        TEST_ASSERT_NEAR(ut1, 0.300, 0.001);

        auto [xr, yr] = time::get_polar_motion(58000.0);
        TEST_ASSERT(std::abs(xr) > 0);
    });

    runner.run();

    // Cleanup
    EOPData::instance().clear();
    std::filesystem::remove(temp_path);
}

// =============================================================================
// UT1-UTC across a leap second; rows past the predictions
// =============================================================================

void test_eop_leap_and_blank_rows() {
    TestRunner runner("EOPLeapAndBlankRows");
    const std::string path = "test_eop_leap.data";
    {
        // 2017-01-01 (MJD 57754): TAI-UTC 36 -> 37, so UT1-UTC steps +1 s.
        std::ofstream f(path);
        f << make_line(57753.0, 0.1, 0.2, -0.4083) << "\n";
        f << make_line(57754.0, 0.1, 0.2, 0.5917) << "\n";
        f << make_line(57755.0, 0.1, 0.2, 0.5910) << "\n";
        // A row past the IERS predictions: MJD present, x/y/UT1 blank. It
        // must be skipped and counted, not loaded as zeros (which would also
        // extend the "in range" span).
        std::string blank(80, ' ');
        blank.replace(7, 8, "57756.00");
        f << blank << "\n";
    }
    EOPData::instance().clear();
    TEST_ASSERT_TRUE(EOPData::instance().load(path));
    LeapSecondTable::instance().reset_to_embedded();

    runner.add_test("ut1_utc_is_interpolated_as_ut1_tai", []() {
        // UT1-UTC jumps by 1 s at the leap second, so it is interpolated as
        // UT1-TAI, which is smooth. Interpolating UT1-UTC linearly across the
        // step would give +0.0917 s at noon (0.5 s wrong: ~230 m at the
        // equator).
        TEST_ASSERT_NEAR(EOPData::instance().ut1_utc(57753.5), -0.4083, 1e-9);
        TEST_ASSERT_NEAR(EOPData::instance().ut1_utc(57754.5), 0.59135, 1e-9);
    });

    runner.add_test("blank_rows_are_skipped_and_counted", []() {
        TEST_ASSERT_EQ(EOPData::instance().size(), static_cast<size_t>(3));
        TEST_ASSERT_EQ(EOPData::instance().blank_rows_skipped(), static_cast<size_t>(1));
        TEST_ASSERT_NEAR(EOPData::instance().mjd_range().second, 57755.0, 1e-9);
    });

    runner.run();
    EOPData::instance().clear();
    std::filesystem::remove(path);
}

// =============================================================================
// Re-entrancy: a diagnostics handler that reads EOP data must not deadlock
// =============================================================================

void test_eop_reentrant_handler() {
    TestRunner runner("EOPReentrantHandler");

    runner.add_test("handler_reading_eop_during_lookup_does_not_deadlock", []() {
        // Rows past the leap-second table's expiry: the lookup's TAI-UTC call reports
        // "leap-second table expires", and the handler calls lookup() again. With the
        // leap-second call made under the EOP mutex this deadlocked (the lookup runs on a
        // worker thread here so the cell fails instead of hanging).
        const std::string path = "test_eop_reentrant.data";
        {
            std::ofstream f(path);
            f << make_line(62000.0, 0.10, 0.20, 0.30) << "\n"
              << make_line(62001.0, 0.11, 0.21, 0.31) << "\n"
              << make_line(62002.0, 0.12, 0.22, 0.32) << "\n";
        }
        EOPData::instance().clear();
        TEST_ASSERT_TRUE(EOPData::instance().load(path));
        LeapSecondTable::instance().reset_to_embedded();
        diagnostics::Sink::instance().reset_dedup();
        auto inner = std::make_shared<std::atomic<int>>(0);
        diagnostics::set_handler([inner](const diagnostics::Diagnostic& d) {
            if (d.message.find("leap-second table expires") != std::string::npos) {
                (void)EOPData::instance().lookup(62001.0);
                ++*inner;
            }
        });
        auto done = std::make_shared<std::atomic<bool>>(false);
        auto value = std::make_shared<std::atomic<double>>(0.0);
        std::thread([done, value]() {
            value->store(EOPData::instance().lookup(62000.5).ut1_utc_s);
            done->store(true);
        }).detach();
        for (int i = 0; i < 2000 && !done->load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const bool finished = done->load();
        if (finished) {
            diagnostics::set_handler(nullptr);
            EOPData::instance().clear();
            std::filesystem::remove(path);
        }
        TEST_ASSERT_TRUE(finished);
        TEST_ASSERT_EQ(inner->load(), 1);
        TEST_ASSERT_NEAR(value->load(), 0.305, 1e-9);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "EOP Data Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_eop_singleton();
    test_eop_load_clear();
    test_eop_interpolation();
    test_eop_leap_and_blank_rows();
    test_eop_reentrant_handler();

    return print_final_summary();
}
