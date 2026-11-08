// Test framework shared by every C++ test in tests/.
//
// A test file builds a TestRunner, registers named cells with add_test(), runs
// them, and returns print_final_summary() from main(). A cell passes if it
// returns normally and fails if it throws; the TEST_ASSERT_* macros below throw
// with the file, line and the values involved. The file also holds a few
// reference constants and synthetic space-weather inputs used across tests.
#pragma once

#include <ballistic/ballistic.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <cstdlib>
#include <sstream>

namespace ballistic {
namespace test {

// Counts across every TestRunner in the process (read by print_final_summary).
struct TestStats {
    int passed = 0;
    int failed = 0;
    int xfailed = 0;       // expected to fail, did fail (counted as success for CI)
    int xpassed = 0;       // expected to fail, but passed (regression — investigate)
    int skipped = 0;       // TEST_SKIP / TEST_SKIP_OPT_IN: did not run (NOT a pass)
    int total = 0;
    std::vector<std::string> failures;
    std::vector<std::string> skips;    // "suite::test - reason"
    std::vector<std::string> xpasses;  // surprises: tracked failures that started passing
};

inline TestStats& global_stats() {
    static TestStats stats;
    return stats;
}

// ANSI color codes (disabled on Windows by default)
#ifdef _WIN32
    inline constexpr const char* GREEN = "";
    inline constexpr const char* RED = "";
    inline constexpr const char* YELLOW = "";
    inline constexpr const char* RESET = "";
#else
    inline constexpr const char* GREEN = "\033[32m";
    inline constexpr const char* RED = "\033[31m";
    inline constexpr const char* YELLOW = "\033[33m";
    inline constexpr const char* RESET = "\033[0m";
#endif

// Skipping a cell. A skip is counted and listed separately from passes: a
// cell that prints "SKIPPED" and returns would count as a pass, so a wrong
// data path would hide it forever (trap TEST-SKIP-AS-PASS).

//   TEST_SKIP(reason)          data or a precondition is missing. With
//                              BALLISTIC_REQUIRE_TEST_DATA=1 (the local gate,
//                              where every data file is present) it FAILS.
//   TEST_SKIP_OPT_IN(reason)   a cell the caller opts into (heavy tiers);
//                              always a skip, never a failure.
// `reason` is a stream expression: TEST_SKIP("only " << h << " h of data").
struct SkipTest {
    std::string reason;
    bool opt_in;
};

inline bool require_test_data() {
    const char* v = std::getenv("BALLISTIC_REQUIRE_TEST_DATA");
    return v && std::string(v) == "1";
}

#define TEST_SKIP(reason_expr) \
    do { \
        std::ostringstream skip_oss_; \
        skip_oss_ << reason_expr; \
        throw ::ballistic::test::SkipTest{skip_oss_.str(), false}; \
    } while(0)

#define TEST_SKIP_OPT_IN(reason_expr) \
    do { \
        std::ostringstream skip_oss_; \
        skip_oss_ << reason_expr; \
        throw ::ballistic::test::SkipTest{skip_oss_.str(), true}; \
    } while(0)

// Assertion macros. Each throws std::runtime_error with the location and the
// values compared, which the runner reports as a failed cell.
#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ << " - Assertion failed: " << #condition; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_MSG(condition, msg) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ << " - " << msg; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_NEAR(actual, expected, tolerance) \
    do { \
        double _a = static_cast<double>(actual); \
        double _e = static_cast<double>(expected); \
        double _t = static_cast<double>(tolerance); \
        if (std::abs(_a - _e) > _t) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Expected " << std::setprecision(15) << _e \
                << " +/- " << _t << ", got " << _a \
                << " (diff=" << std::abs(_a - _e) << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

// Relative tolerance: |actual - expected| <= |expected| * rel_tolerance, with an
// absolute floor of 1e-15 so that expected == 0 does not demand exact equality.
#define TEST_ASSERT_NEAR_REL(actual, expected, rel_tolerance) \
    do { \
        double _a = static_cast<double>(actual); \
        double _e = static_cast<double>(expected); \
        double _t = std::abs(_e * rel_tolerance); \
        if (_t < 1e-15) _t = 1e-15; \
        if (std::abs(_a - _e) > _t) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Expected " << std::setprecision(15) << _e \
                << " +/- " << (rel_tolerance * 100) << "%, got " << _a \
                << " (rel_diff=" << std::abs((_a - _e) / _e) * 100 << "%)"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_VEC3_NEAR(actual, expected, tolerance) \
    do { \
        Eigen::Vector3d _a = (actual); \
        Eigen::Vector3d _e = (expected); \
        double _diff = (_a - _e).norm(); \
        if (_diff > (tolerance)) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Vector difference " << _diff << " exceeds tolerance " << (tolerance) \
                << "\n  Expected: [" << _e.x() << ", " << _e.y() << ", " << _e.z() << "]" \
                << "\n  Actual:   [" << _a.x() << ", " << _a.y() << ", " << _a.z() << "]"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define TEST_ASSERT_TRUE(condition) TEST_ASSERT(condition)
#define TEST_ASSERT_FALSE(condition) TEST_ASSERT(!(condition))

#define TEST_ASSERT_EQ(actual, expected) \
    do { \
        auto _a = (actual); \
        auto _e = (expected); \
        if (_a != _e) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Expected " << _e << ", got " << _a; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

// Passes only if `expr` throws `exception_type` (or a subclass); any other
// exception counts as "did not throw the expected type".
#define TEST_ASSERT_THROWS(expr, exception_type) \
    do { \
        bool threw = false; \
        try { expr; } \
        catch (const exception_type&) { threw = true; } \
        catch (...) { } \
        if (!threw) { \
            std::ostringstream oss; \
            oss << __FILE__ << ":" << __LINE__ \
                << " - Expected exception " << #exception_type; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

// Runs a suite of named cells, prints PASSED / FAILED / SKIP / XFAIL per cell
// and a suite summary.
class TestRunner {
public:
    using TestFunc = std::function<void()>;

    TestRunner(const std::string& suite_name) : suite_name_(suite_name) {
        std::cout << "\n=== " << suite_name_ << " ===" << std::endl;
    }

    void add_test(const std::string& name, TestFunc func) {
        tests_.push_back({name, func, /*xfail*/ false, /*reason*/ ""});
    }

    /// Mark a test as expected-to-fail. The runner counts xfail as success
    /// for CI purposes; if the test starts passing (xpass), the runner flags
    /// that as a regression so the xfail mark can be retired.
    /// `reason` is printed on every run to keep the failure visible.
    void add_xfail(const std::string& name, const std::string& reason, TestFunc func) {
        tests_.push_back({name, func, /*xfail*/ true, reason});
    }

    int run() {
        int passed = 0;
        int failed = 0;
        int xfailed = 0;
        int xpassed = 0;
        int skipped = 0;

        // TEST_FILTER / TEST_FILTER_EXCLUDE: ','-separated substring lists
        // (comma, not semicolon: these are set via CTest ENVIRONMENT
        // properties, where semicolons are CMake list separators).
        // A test runs if its name contains any TEST_FILTER substring (or the
        // variable is unset) and no TEST_FILTER_EXCLUDE substring. A single
        // substring (no ',') behaves as before.
        const char* filter = std::getenv("TEST_FILTER");
        const char* exclude = std::getenv("TEST_FILTER_EXCLUDE");
        auto matches_any = [](const std::string& name, const char* spec) {
            std::string s(spec);
            std::size_t pos = 0;
            while (pos <= s.size()) {
                const std::size_t semi = s.find(',', pos);
                const std::string tok = s.substr(
                    pos, semi == std::string::npos ? std::string::npos : semi - pos);
                if (!tok.empty() && name.find(tok) != std::string::npos) return true;
                if (semi == std::string::npos) break;
                pos = semi + 1;
            }
            return false;
        };

        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& test : tests_) {
            if (filter && !matches_any(test.name, filter)) continue;
            if (exclude && matches_any(test.name, exclude)) continue;
            std::cout << "  " << test.name << "... " << std::flush;
            bool threw = false;
            std::string err;
            try {
                test.func();
            } catch (const SkipTest& skip) {
                if (skip.opt_in || !require_test_data()) {
                    std::cout << YELLOW << "SKIP" << RESET << " (" << skip.reason << ")" << std::endl;
                    skipped++;
                    global_stats().skips.push_back(suite_name_ + "::" + test.name + " - " + skip.reason);
                    continue;
                }
                threw = true;
                err = "skipped with BALLISTIC_REQUIRE_TEST_DATA=1: " + skip.reason;
            } catch (const std::exception& e) {
                threw = true;
                err = e.what();
            }

            if (test.xfail) {
                if (threw) {
                    std::cout << YELLOW << "XFAIL" << RESET
                              << " (expected: " << test.reason << ")" << std::endl;
                    xfailed++;
                } else {
                    std::cout << YELLOW << "XPASS" << RESET
                              << " (was expected to fail: " << test.reason
                              << " — retire the xfail mark)" << std::endl;
                    xpassed++;
                    global_stats().xpasses.push_back(suite_name_ + "::" + test.name);
                }
            } else {
                if (threw) {
                    std::cout << RED << "FAILED" << RESET << std::endl;
                    std::cout << "    " << err << std::endl;
                    failed++;
                    global_stats().failures.push_back(suite_name_ + "::" + test.name);
                } else {
                    std::cout << GREEN << "PASSED" << RESET << std::endl;
                    passed++;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "\n  " << suite_name_ << " Summary: "
                  << passed << "/" << (passed + failed + xfailed + xpassed + skipped) << " passed";
        if (skipped > 0) std::cout << ", " << YELLOW << skipped << " skipped" << RESET;
        if (xfailed > 0) std::cout << ", " << xfailed << " xfail";
        if (xpassed > 0) std::cout << ", " << YELLOW << xpassed << " xpass" << RESET;
        std::cout << " (" << duration.count() << "ms)" << std::endl;

        global_stats().passed += passed;
        global_stats().failed += failed;
        global_stats().xfailed += xfailed;
        global_stats().xpassed += xpassed;
        global_stats().skipped += skipped;
        global_stats().total += passed + failed + xfailed + xpassed + skipped;

        // Returns the number of hard failures only; an unexpected pass of an
        // xfail cell makes print_final_summary() return non-zero instead.
        return failed;
    }

private:
    struct Test {
        std::string name;
        TestFunc func;
        bool xfail;
        std::string reason;
    };

    std::string suite_name_;
    std::vector<Test> tests_;
};

// Alternative registration: defines a function and adds it to `runner` at
// static-initialization time.
#define TEST_CASE(runner, test_name) \
    void test_##test_name(); \
    struct TestRegistrar_##test_name { \
        TestRegistrar_##test_name() { runner.add_test(#test_name, test_##test_name); } \
    } registrar_##test_name; \
    void test_##test_name()

// Prints the process-wide summary; main() returns its value (0 = all good).
inline int print_final_summary() {
    auto& stats = global_stats();
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total: " << stats.total << " tests" << std::endl;
    std::cout << GREEN << "Passed: " << stats.passed << RESET << std::endl;
    if (stats.xfailed > 0) {
        std::cout << YELLOW << "XFail (expected failures): " << stats.xfailed
                  << RESET << std::endl;
    }
    if (stats.skipped > 0) {
        std::cout << YELLOW << "Skipped (did not run): " << stats.skipped << RESET << std::endl;
        for (const auto& f : stats.skips) {
            std::cout << "  ~ " << f << std::endl;
        }
    }
    if (stats.xpassed > 0) {
        std::cout << YELLOW << "XPass (UNEXPECTED success — retire xfail): "
                  << stats.xpassed << RESET << std::endl;
        std::cout << "\nUnexpected successes (retire the xfail mark):" << std::endl;
        for (const auto& f : stats.xpasses) {
            std::cout << "  + " << f << std::endl;
        }
    }
    if (stats.failed > 0) {
        std::cout << RED << "Failed: " << stats.failed << RESET << std::endl;
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& f : stats.failures) {
            std::cout << "  - " << f << std::endl;
        }
    }
    std::cout << "========================================" << std::endl;
    // Non-zero on hard failures or on XPasses (regression-tracked tests should
    // be retired explicitly when they start passing).
    return (stats.failed > 0 || stats.xpassed > 0) ? 1 : 0;
}

// Reference values from published sources, for tests that check the library's
// own constants and conversions against them.
namespace reference {

// J2000.0 epoch (2000-01-01 12:00:00 TT)
constexpr double J2000_JD = 2451545.0;
constexpr double J2000_MJD = 51544.5;

// Unix epoch (1970-01-01 00:00:00 UTC)
constexpr double UNIX_EPOCH_JD = 2440587.5;

// GMST at J2000.0 (radians) - from IERS Conventions
constexpr double GMST_J2000_RAD = 4.894961212823756;  // ~280.46 degrees

// Time offsets
constexpr double TT_TAI_OFFSET = 32.184;  // seconds
constexpr double GPS_TAI_OFFSET = -19.0;  // seconds

// TAI-UTC (accumulated leap seconds) at two epochs
constexpr int TAI_UTC_1972 = 10;   // value when the leap-second system began (1972-01-01)
constexpr int TAI_UTC_2017 = 37;   // after the leap second of 2017-01-01, still current

// Sun position at J2000.0 TDB (meters, GCRS) - from JPL Horizons
constexpr double SUN_J2000_X = 26500628134.0;
constexpr double SUN_J2000_Y = 132756139614.0;
constexpr double SUN_J2000_Z = 57556500184.0;

// Moon position at J2000.0 TDB (meters, GCRS) - from JPL Horizons
constexpr double MOON_J2000_X = -291608384.0;
constexpr double MOON_J2000_Y = -266716833.0;
constexpr double MOON_J2000_Z = -76102488.0;

// Earth parameters (WGS84)
constexpr double EARTH_GM = 3.986004418e14;  // m^3/s^2
constexpr double EARTH_RE = 6378137.0;       // m (equatorial radius)
constexpr double EARTH_RP = 6356752.3142;    // m (polar radius)
constexpr double EARTH_J2 = 1.08262668355e-3;

// Surface gravity
constexpr double SURFACE_GRAVITY = 9.80665;  // m/s^2 (standard)

// Atmospheric density at sea level
constexpr double SEA_LEVEL_DENSITY = 1.225;  // kg/m^3

// Obliquity of ecliptic at J2000.0
constexpr double OBLIQUITY_J2000_DEG = 23.439291111;  // degrees

}  // namespace reference

// Synthetic space-weather inputs for atmosphere tests. F10.7 is the solar
// 10.7 cm radio flux (solar flux units; daily and 81-day average) and Ap the
// geomagnetic activity index (daily value plus the history array NRLMSIS
// reads). Quiet, active and storm levels are round illustrative values, not a
// particular day.
inline WeatherConditions make_quiet_weather() {
    WeatherConditions w;
    w.f107_daily = 70.0;
    w.f107_average = 72.0;
    w.ap_daily = 4.0;
    w.ap_history = {4, 4, 4, 4, 4, 4, 4};
    return w;
}

inline WeatherConditions make_active_weather() {
    WeatherConditions w;
    w.f107_daily = 200.0;
    w.f107_average = 190.0;
    w.ap_daily = 15.0;
    w.ap_history = {15, 15, 15, 15, 15, 15, 15};
    return w;
}

inline WeatherConditions make_storm_weather() {
    WeatherConditions w;
    w.f107_daily = 250.0;
    w.f107_average = 220.0;
    w.ap_daily = 100.0;
    w.ap_history = {100, 80, 60, 40, 30, 20, 15};
    return w;
}

}  // namespace test
}  // namespace ballistic
