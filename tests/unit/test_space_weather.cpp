// Unit tests for the space-weather loader (atmosphere/space_weather.hpp),
// which reads CelesTrak's SW-All.csv. The atmosphere model NRLMSIS 2.1 needs
// two inputs from it: F10.7 (the Sun's 10.7 cm radio flux, in solar flux
// units, a proxy for solar heating) and Ap (a daily / 3-hourly geomagnetic
// activity index; Kp is its logarithmic sibling, stored in tenths). MJD =
// Modified Julian Date. Most cells use a small synthetic CSV written by
// create_test_csv() with made-up values of realistic size dated 2005-06-29 to
// 2005-07-03 (not the real file's values for those days); RealDataFile and the pinned-date cells read the committed
// tests/data/weather/SW-All.csv.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/space_weather.hpp>
#include <ballistic/time/epoch.hpp>
#include <filesystem>
#include <fstream>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;
using namespace ballistic::time;

// =============================================================================
// Helper Functions
// =============================================================================

// Create a minimal test CSV file
std::filesystem::path create_test_csv() {
    std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "test_space_weather.csv";

    std::ofstream file(temp_path);
    file << "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n";
    file << "2005-06-29,2344,15,20,17,17,17,17,23,30,30,171,7,6,6,6,6,9,18,18,10,0.3,2,25,112.3,116.2,OBS,93.8,93.6,96.7,95.7\n";
    file << "2005-06-30,2345,16,30,27,17,17,20,17,23,30,181,18,12,6,6,7,6,9,18,10,0.3,2,28,113.4,117.3,OBS,94.2,94.0,97.1,96.1\n";
    file << "2005-07-01,2346,17,27,20,30,23,23,27,17,13,180,12,9,18,12,12,15,6,5,11,0.4,2,40,114.6,118.5,OBS,94.6,94.4,97.6,96.6\n";
    file << "2005-07-02,2347,18,13,20,17,20,10,17,13,17,127,5,7,6,7,4,6,5,6,6,0.3,1,35,115.2,119.1,OBS,95.0,94.8,98.0,97.0\n";
    file << "2005-07-03,2348,19,23,20,20,17,17,17,20,17,151,9,7,7,6,6,6,7,6,7,0.4,2,42,116.1,120.0,OBS,95.4,95.2,98.4,97.4\n";
    file.close();

    return temp_path;
}

// =============================================================================
// Test Suites
// =============================================================================

void test_space_weather_record() {
    TestRunner runner("SpaceWeatherRecord");

    runner.add_test("default_construction", []() {
        SpaceWeatherRecord rec;

        TEST_ASSERT_NEAR(rec.mjd, 0.0, 1e-10);
        TEST_ASSERT_NEAR(rec.f107_obs, 150.0, 1e-10);
        TEST_ASSERT_NEAR(rec.ap_daily, 4.0, 1e-10);
        TEST_ASSERT_EQ(rec.ap_3hr.size(), 8);
    });

    runner.run();
}

void test_csv_parsing() {
    TestRunner runner("CSVParsing");

    runner.add_test("load_test_file", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        bool loaded = SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        TEST_ASSERT(loaded);
        TEST_ASSERT(SpaceWeatherData::instance().is_loaded());
        TEST_ASSERT_EQ(SpaceWeatherData::instance().size(), 5);

        std::filesystem::remove(temp_path);
    });

    runner.add_test("f107_values", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        // July 1, 2005: F10.7 = 114.6
        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        auto rec = SpaceWeatherData::instance().get_record(epoch.mjd(TimeScale::UTC));

        TEST_ASSERT(rec.has_value());
        TEST_ASSERT_NEAR(rec->f107_obs, 114.6, 0.1);
        TEST_ASSERT_NEAR(rec->f107_81_center, 94.6, 0.1);

        std::filesystem::remove(temp_path);
    });

    runner.add_test("ap_values", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        // July 1, 2005: Ap_avg = 11, AP values = 12,9,18,12,12,15,6,5
        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        auto rec = SpaceWeatherData::instance().get_record(epoch.mjd(TimeScale::UTC));

        TEST_ASSERT(rec.has_value());
        TEST_ASSERT_NEAR(rec->ap_daily, 11.0, 0.1);
        TEST_ASSERT_NEAR(rec->ap_3hr[0], 12.0, 0.1);  // AP1
        TEST_ASSERT_NEAR(rec->ap_3hr[2], 18.0, 0.1);  // AP3
        TEST_ASSERT_NEAR(rec->ap_3hr[7], 5.0, 0.1);   // AP8

        std::filesystem::remove(temp_path);
    });

    runner.add_test("kp_values", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        // July 1, 2005: KP values = 27,20,30,23,23,27,17,13 (in 10ths)
        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        auto rec = SpaceWeatherData::instance().get_record(epoch.mjd(TimeScale::UTC));

        TEST_ASSERT(rec.has_value());
        TEST_ASSERT_NEAR(rec->kp_3hr[0], 2.7, 0.01);  // KP1
        TEST_ASSERT_NEAR(rec->kp_3hr[2], 3.0, 0.01);  // KP3

        std::filesystem::remove(temp_path);
    });

    runner.add_test("mjd_range", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        auto [min_mjd, max_mjd] = SpaceWeatherData::instance().mjd_range();

        // June 29 to July 3, 2005
        TEST_ASSERT(min_mjd > 53548.0);  // June 29, 2005
        TEST_ASSERT(max_mjd < 53555.0);  // July 3, 2005
        TEST_ASSERT(min_mjd < max_mjd);

        std::filesystem::remove(temp_path);
    });

    runner.run();
}

void test_get_conditions() {
    TestRunner runner("GetConditions");

    runner.add_test("exact_date", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        // At noon the value is the day's row (2005-07-02: F10.7 115.2, Ap 6);
        // the 57-hour Ap lookback reaches 06-30, inside the file.
        Epoch epoch = Epoch::from_utc(2005, 7, 2, 12, 0, 0);
        WeatherConditions conditions = SpaceWeatherData::instance().get_conditions(epoch);
        TEST_ASSERT_NEAR(conditions.f107_daily, 115.2, 1e-9);
        TEST_ASSERT_NEAR(conditions.f107_average, 95.0, 1e-9);
        TEST_ASSERT_NEAR(conditions.ap_daily, 6.0, 1e-9);
        // 07-01 00:00 needs 06-28, before the file: raises rather than
        // filling the missing days with the current day's mean.
        TEST_ASSERT_THROWS(SpaceWeatherData::instance().get_conditions(
                               Epoch::from_utc(2005, 7, 1, 0, 0, 0)), std::out_of_range);

        std::filesystem::remove(temp_path);
    });

    runner.add_test("interpolation", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        // Mid-day on July 1, 2005
        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        WeatherConditions conditions = SpaceWeatherData::instance().get_conditions(epoch);

        // Should interpolate between July 1 and July 2
        // July 1: 114.6, July 2: 115.2
        // At noon: ~114.9
        TEST_ASSERT(conditions.f107_daily >= 114.0);
        TEST_ASSERT(conditions.f107_daily <= 116.0);

        std::filesystem::remove(temp_path);
    });

    runner.add_test("ap_history_structure", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        WeatherConditions conditions = SpaceWeatherData::instance().get_conditions(epoch);

        // ap_history should have 7 elements
        TEST_ASSERT_EQ(conditions.ap_history.size(), 7);

        // All values should be positive
        for (int i = 0; i < 7; ++i) {
            TEST_ASSERT(conditions.ap_history[i] > 0.0);
        }

        std::filesystem::remove(temp_path);
    });

    runner.add_test("convenience_function", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        WeatherConditions conditions = get_weather_conditions(epoch);

        TEST_ASSERT(conditions.f107_daily > 100.0);
        TEST_ASSERT(conditions.ap_daily > 0.0);

        std::filesystem::remove(temp_path);
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("before_data_range", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        // Before the file: an error, not the first record
        Epoch epoch = Epoch::from_utc(2005, 1, 1, 0, 0, 0);
        TEST_ASSERT_THROWS((void)SpaceWeatherData::instance().get_conditions(epoch),
                           DataCoverageError);

        std::filesystem::remove(temp_path);
    });

    runner.add_test("after_data_range", []() {
        auto temp_path = create_test_csv();

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);

        // After the file: an error, not the last record
        Epoch epoch = Epoch::from_utc(2005, 12, 31, 0, 0, 0);
        TEST_ASSERT_THROWS((void)SpaceWeatherData::instance().get_conditions(epoch),
                           DataCoverageError);

        std::filesystem::remove(temp_path);
    });

    runner.add_test("empty_data", []() {
        SpaceWeatherData::instance().clear();

        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        // Nothing loaded: an error, not quiet defaults
        TEST_ASSERT_THROWS((void)SpaceWeatherData::instance().get_conditions(epoch),
                           DataCoverageError);
    });

    runner.add_test("invalid_file", []() {
        SpaceWeatherData::instance().clear();
        bool loaded = SpaceWeatherData::instance().load_celestrak_csv("nonexistent_file.csv");

        TEST_ASSERT_FALSE(loaded);
        TEST_ASSERT_FALSE(SpaceWeatherData::instance().is_loaded());
    });

    runner.run();
}

void test_real_data_file() {
    TestRunner runner("RealDataFile");

    std::filesystem::path sw_path = BALLISTIC_PROJECT_ROOT "/tests/data/weather/SW-All.csv";

    runner.add_test("load_celestrak_file", [&sw_path]() {
        if (!std::filesystem::exists(sw_path)) {
            TEST_SKIP("file not found: " << sw_path);
        }

        SpaceWeatherData::instance().clear();
        bool loaded = SpaceWeatherData::instance().load_celestrak_csv(sw_path);

        TEST_ASSERT(loaded);
        TEST_ASSERT(SpaceWeatherData::instance().size() > 1000);

        auto [min_mjd, max_mjd] = SpaceWeatherData::instance().mjd_range();
        std::cout << "[" << SpaceWeatherData::instance().size() << " records] ";
    });

    runner.add_test("champ_validation_date", [&sw_path]() {
        if (!std::filesystem::exists(sw_path)) {
            TEST_SKIP("file not found: " << sw_path);
        }

        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(sw_path);

        // July 1, 2005 - CHAMP validation date
        Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
        WeatherConditions conditions = SpaceWeatherData::instance().get_conditions(epoch);

        // CelesTrak SW-All.csv for 2005-07-01: F10.7 = 114.6, Ap = 14
        TEST_ASSERT_NEAR(conditions.f107_daily, 114.6, 1.0);
        TEST_ASSERT_NEAR(conditions.ap_daily, 14.0, 2.0);  // Allow some tolerance

        std::cout << "[F10.7=" << conditions.f107_daily << ", Ap=" << conditions.ap_daily << "] ";
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

// =============================================================================
// Pinned dates against the COMMITTED real
// SW-All.csv, verified against the INDEPENDENT DRAO/Penticton observed-flux
// tables (NOAA NGDC, drao_noontime-flux-observed_<year>.txt, retrieved
// 2026-07-19). Guards the parser end-to-end: column selection (F10.7_OBS —  guard-exempt(PUB-DATE): data
// the Sun-Earth-distance flux NRLMSIS 2.1 requires per its own source
// comments), date indexing, and Ap wiring. Three dates match DRAO exactly;
// the 2003 Halloween storm day pins the famous Ap=204 with an F10.7 BAND
// (DRAO 2000 UT logs 279.1 flare-starred vs CelesTrak's 291.7 — flare-day
// daily-value conventions differ between archives; both in [270, 295]).
// =============================================================================
void test_pinned_historical_dates() {
    TestRunner runner("PinnedHistoricalDates");

    runner.add_test("independent_archive_pins", []() {
        auto csv = std::filesystem::path(__FILE__).parent_path()
                       .parent_path() / "data" / "weather" / "SW-All.csv";
        TEST_ASSERT(std::filesystem::exists(csv));
        SpaceWeatherData::instance().clear();
        TEST_ASSERT(SpaceWeatherData::instance().load_celestrak_csv(csv.string()));

        struct Pin { int y, m, d; double f107, ap; };
        // DRAO-exact pins (independent archive matches to the digit).
        const Pin exact[] = {
            {2005, 7, 1, 114.6, 14.0},    // CHAMP drag validation day
            {2008, 12, 20, 69.1, 2.0},    // solar minimum
            {2014, 2, 24, 170.7, 5.0},    // cycle-24 maximum
        };
        for (const auto& pin : exact) {
            Epoch e = Epoch::from_utc(pin.y, pin.m, pin.d, 12, 0, 0);
            auto c = SpaceWeatherData::instance().get_conditions(e);
            TEST_ASSERT_NEAR(c.f107_daily, pin.f107, 0.15);
            TEST_ASSERT_NEAR(c.ap_daily, pin.ap, 0.5);
        }
        // Halloween storm 2003-10-29: Ap = 204 (canonical), F10.7 banded.
        {
            Epoch e = Epoch::from_utc(2003, 10, 29, 12, 0, 0);
            auto c = SpaceWeatherData::instance().get_conditions(e);
            TEST_ASSERT_NEAR(c.ap_daily, 204.0, 0.5);
            TEST_ASSERT(c.f107_daily > 270.0 && c.f107_daily < 295.0);
        }
        SpaceWeatherData::instance().clear();
    });

    runner.run();
}

// =============================================================================
// SpaceWeatherData::history — the measured default weather of
// make_drag_environment.
// =============================================================================
namespace {
// Six observed days (Ap rises by 10 per 3-h bin across the series, F10.7 = 100
// + day index) plus one CelesTrak monthly PREDICTED row (no 3-hourly Ap).
std::filesystem::path create_history_csv() {
    auto p = std::filesystem::temp_directory_path() / "sw_history.csv";
    std::ofstream f(p);
    f << "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n";
    for (int d = 0; d < 6; ++d) {
        f << "2024-03-0" << (d + 1) << ",0,0,0,0,0,0,0,0,0,0,0";
        for (int k = 0; k < 8; ++k) f << "," << (10 * (8 * d + k) + 1);
        f << ",0,0,0,0," << (100 + d) << ",0,OBS," << (200 + d) << ",0,0,0\n";
    }
    f << "2024-04-01,,,,,,,,,,,,,,,,,,,,,,,,120.0,,PRM,121.0,,,\n";
    return p;
}
}  // namespace

void test_measured_history() {
    TestRunner runner("MeasuredHistory");

    // Window inside day index 4 (2024-03-05), 06:00: NRLMSIS wants the  guard-exempt(PUB-DATE): data
    // PREVIOUS day's F10.7 and the 24-h mean of the 8 bins ending with the
    // current one; ap_history follows NRLMSIS's AP-array schedule.
    runner.add_test("previous_day_f107_and_nrlmsis_ap_schedule", []() {
        auto p = create_history_csv();
        auto& sw = SpaceWeatherData::instance();
        sw.clear();
        TEST_ASSERT(sw.load_celestrak_csv(p));
        const Epoch t = Epoch::from_utc(2024, 3, 5, 6, 0, 0);
        auto h = sw.history(TimeWindow{t, t + 3600.0});
        auto w = h.at(t);
        auto ap = [](int day, int bin) { return 10.0 * (8 * day + bin) + 1.0; };
        TEST_ASSERT_NEAR(w.f107_daily, 103.0, 1e-12);          // day 3's obs
        double mean8 = 0.0;
        for (int k = 3; k < 8; ++k) mean8 += ap(3, k);
        for (int k = 0; k < 3; ++k) mean8 += ap(4, k);
        mean8 /= 8.0;
        TEST_ASSERT_NEAR(w.ap_daily, mean8, 1e-12);
        TEST_ASSERT_NEAR(w.ap_history[0], mean8, 1e-12);       // daily Ap slot
        TEST_ASSERT_NEAR(w.ap_history[1], ap(4, 2), 1e-12);    // current 3-h
        TEST_ASSERT_NEAR(w.ap_history[2], ap(4, 1), 1e-12);    // -3 h
        TEST_ASSERT_NEAR(w.ap_history[4], ap(3, 7), 1e-12);    // -9 h
        // 12..33 h before 06:00 on day 4 = day 3 bins 6..0 and day 2 bin 7
        double m12_expected = ap(2, 7);
        for (int k = 0; k <= 6; ++k) m12_expected += ap(3, k);
        TEST_ASSERT_NEAR(w.ap_history[5], m12_expected / 8.0, 1e-12);
        sw.clear();
        std::filesystem::remove(p);
    });

    // Predicted rows carry no 3-hourly Ap: refusing them is the point
    // (silently using Ap = 4 there biases density low, as quiet weather does).

    runner.add_test("refuses_predicted_rows_and_gaps", []() {
        auto p = create_history_csv();
        auto& sw = SpaceWeatherData::instance();
        sw.clear();
        sw.load_celestrak_csv(p);
        const Epoch pred = Epoch::from_utc(2024, 4, 1, 12, 0, 0);
        TEST_ASSERT_THROWS((void)sw.history(TimeWindow{pred, pred + 3600.0}),
                           std::out_of_range);
        // Day 1 needs 3 days of Ap lookback that the file does not have.
        const Epoch early = Epoch::from_utc(2024, 3, 2, 12, 0, 0);
        TEST_ASSERT_THROWS((void)sw.history(TimeWindow{early, early + 3600.0}),
                           std::out_of_range);
        sw.clear();
        std::filesystem::remove(p);
    });

    runner.run();
}

// =============================================================================
// Load warnings: a malformed CSV must not SILENTLY substitute quiet weather.
// Failure mode guarded: a renamed column → whole file defaults to quiet
// 150/4, density biased low, undetected.
// =============================================================================
void test_load_warnings() {
    TestRunner runner("LoadWarnings");

    // Clean data must produce NO warnings (guard against false positives).
    runner.add_test("clean_csv_no_warnings", []() {
        auto temp_path = create_test_csv();
        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(temp_path);
        TEST_ASSERT(SpaceWeatherData::instance().warnings().empty());
        std::filesystem::remove(temp_path);
    });

    // A renamed F10.7 column → col stays -1 → every record silently quiet.
    runner.add_test("missing_f107_column_refused", []() {
        auto p = std::filesystem::temp_directory_path() / "sw_bad_header.csv";
        std::ofstream f(p);
        f << "DATE,KP_SUM,AP_AVG,F107_RENAMED\n";
        f << "2005-07-01,180,11,114.6\n";
        f.close();
        SpaceWeatherData::instance().clear();
        // Refused (loading it would mean all-quiet weather with a note).
        TEST_ASSERT_THROWS(SpaceWeatherData::instance().load_celestrak_csv(p), ballistic::Error);
        TEST_ASSERT(!SpaceWeatherData::instance().is_loaded());
        std::filesystem::remove(p);
    });

    // A present-but-unparseable value → counted + surfaced.
    runner.add_test("unparseable_value_warns", []() {
        auto p = std::filesystem::temp_directory_path() / "sw_bad_value.csv";
        std::ofstream f(p);
        f << "DATE,KP_SUM,AP_AVG,F10.7_OBS\n";
        f << "2005-07-01,180,11,not_a_number\n";
        f.close();
        SpaceWeatherData::instance().clear();
        SpaceWeatherData::instance().load_celestrak_csv(p);
        bool found = false;
        for (const auto& s : SpaceWeatherData::instance().warnings())
            if (s.find("unparseable") != std::string::npos) found = true;
        TEST_ASSERT(found);
        std::filesystem::remove(p);
    });

    runner.run();
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Space Weather Data Tests" << std::endl;
    std::cout << "============================================" << std::endl;

    test_space_weather_record();
    test_csv_parsing();
    test_get_conditions();
    test_pinned_historical_dates();
    test_edge_cases();
    test_load_warnings();
    test_measured_history();
    test_real_data_file();

    auto& stats = global_stats();
    std::cout << "\n============================================" << std::endl;
    std::cout << "Total: " << stats.total << " tests, "
              << GREEN << stats.passed << " passed" << RESET << ", "
              << RED << stats.failed << " failed" << RESET << std::endl;

    if (!stats.failures.empty()) {
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& f : stats.failures) {
            std::cout << "  - " << f << std::endl;
        }
    }

    return stats.failed > 0 ? 1 : 0;
}
