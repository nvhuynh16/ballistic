// Unit tests for SpaceWeatherHistory: the time series of space-weather inputs the
// atmosphere model reads (F10.7 = daily 10.7 cm solar radio flux and its 81-day
// average; Ap = daily geomagnetic index) over a validity window. Checks lookups
// (nearest prior sample), the auto-computed 81-day mean, and that bad inputs and
// epochs outside the window raise. Values are copied, so tolerances are exact.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/core/errors.hpp>
#include <stdexcept>
#include <string>

using namespace ballistic;
using namespace ballistic::test;
using ballistic::atmosphere::SpaceWeatherHistory;
using ballistic::atmosphere::TimeWindow;

namespace {

WeatherConditions sample_quiet() { return WeatherConditions::quiet(); }
WeatherConditions sample_active() { return WeatherConditions::active(); }

TimeWindow one_day_window() {
    Epoch start = Epoch::from_utc(2024, 6, 15, 0, 0, 0);
    Epoch end   = Epoch::from_utc(2024, 6, 16, 0, 0, 0);
    return TimeWindow{start, end};
}

}  // namespace

void test_default_construction() {
    TestRunner runner("DefaultConstruction");

    runner.add_test("empty_history_reports_empty", []() {
        SpaceWeatherHistory h;
        TEST_ASSERT(h.empty());
    });

    runner.add_test("at_throws_on_empty", []() {
        SpaceWeatherHistory h;
        bool threw = false;
        try {
            (void)h.at(Epoch::from_utc(2024, 1, 1, 0, 0, 0));
        } catch (const std::out_of_range&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_from_constant() {
    TestRunner runner("FromConstant");

    runner.add_test("flat_quiet_history_returns_quiet_anywhere_in_window", []() {
        const WeatherConditions w = sample_quiet();
        SpaceWeatherHistory h = SpaceWeatherHistory::from_constant(w, one_day_window());
        TEST_ASSERT(!h.empty());

        // Midpoint
        Epoch t = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        WeatherConditions got = h.at(t);
        TEST_ASSERT_NEAR(got.f107_daily, w.f107_daily, 1e-9);
        TEST_ASSERT_NEAR(got.f107_average, w.f107_average, 1e-9);
        TEST_ASSERT_NEAR(got.ap_daily, w.ap_daily, 1e-9);
        for (std::size_t i = 0; i < 7; ++i) {
            TEST_ASSERT_NEAR(got.ap_history[i], w.ap_history[i], 1e-9);
        }
    });

    runner.add_test("flat_active_history_returns_active", []() {
        const WeatherConditions w = sample_active();
        SpaceWeatherHistory h = SpaceWeatherHistory::from_constant(w, one_day_window());

        Epoch t = Epoch::from_utc(2024, 6, 15, 18, 0, 0);
        WeatherConditions got = h.at(t);
        TEST_ASSERT_NEAR(got.f107_daily, 200.0, 1e-9);
        TEST_ASSERT_NEAR(got.ap_daily, 15.0, 1e-9);
    });

    runner.add_test("invalid_window_rejected", []() {
        bool threw = false;
        try {
            Epoch t = Epoch::from_utc(2024, 6, 15, 0, 0, 0);
            (void)SpaceWeatherHistory::from_constant(sample_quiet(), TimeWindow{t, t});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_explicit_series() {
    TestRunner runner("ExplicitSeries");

    runner.add_test("f107_daily_nearest_prior_lookup", []() {
        // Three daily samples: 100, 150, 200 on consecutive days.
        std::vector<Epoch> td{
            Epoch::from_utc(2024, 6, 15, 0, 0, 0),
            Epoch::from_utc(2024, 6, 16, 0, 0, 0),
            Epoch::from_utc(2024, 6, 17, 0, 0, 0),
        };
        std::vector<double> f107{100.0, 150.0, 200.0};
        std::vector<double> avg{};  // auto-compute
        std::vector<Epoch> tap{td.front(), td.back()};
        std::vector<double> ap{4.0, 4.0};

        SpaceWeatherHistory h(td, f107, avg, tap, ap);

        // Mid-day on day 1 (June 15 12:00) — nearest-prior is day-1 sample (100).
        Epoch t1 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        TEST_ASSERT_NEAR(h.at(t1).f107_daily, 100.0, 1e-9);

        // Mid-day on day 2 (June 16 12:00) — nearest-prior is day-2 sample (150).
        Epoch t2 = Epoch::from_utc(2024, 6, 16, 12, 0, 0);
        TEST_ASSERT_NEAR(h.at(t2).f107_daily, 150.0, 1e-9);
    });

    runner.add_test("f107_81day_avg_auto_computed_matches_mean", []() {
        // Three samples → 81-day window is degenerate; avg should equal local mean.
        std::vector<Epoch> td{
            Epoch::from_utc(2024, 6, 15, 0, 0, 0),
            Epoch::from_utc(2024, 6, 16, 0, 0, 0),
            Epoch::from_utc(2024, 6, 17, 0, 0, 0),
        };
        std::vector<double> f107{100.0, 150.0, 200.0};
        std::vector<Epoch> tap{td.front(), td.back()};
        std::vector<double> ap{4.0, 4.0};

        SpaceWeatherHistory h(td, f107, /*avg*/{}, tap, ap);

        // Auto-computed 81-day mean for 3 samples is the global mean: 150.
        const double expected_mean = (100.0 + 150.0 + 200.0) / 3.0;
        for (std::size_t i = 0; i < h.f107_81day_avg().size(); ++i) {
            TEST_ASSERT_NEAR(h.f107_81day_avg()[i], expected_mean, 1e-9);
        }
    });

    runner.add_test("no_ap_values_raises_instead_of_quiet_default", []() {
        // Daily F10.7 but no 3-hourly Ap: at() used to fill the daily Ap with
        // 4 (quiet) without a word. It must raise, naming the epoch.
        std::vector<Epoch> td{
            Epoch::from_utc(2024, 6, 15, 0, 0, 0),
            Epoch::from_utc(2024, 6, 16, 0, 0, 0),
        };
        SpaceWeatherHistory h(td, {120.0, 130.0}, {}, {}, {});
        bool threw = false;
        try {
            (void)h.at(Epoch::from_utc(2024, 6, 15, 12, 0, 0));
        } catch (const DataCoverageError& e) {
            const std::string msg = e.what();
            threw = msg.find("no 3-hourly Ap") != std::string::npos &&
                    msg.find("2024-06-15") != std::string::npos;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("size_mismatch_rejected", []() {
        std::vector<Epoch> td{Epoch::from_utc(2024, 6, 15, 0, 0, 0)};
        std::vector<double> f107{100.0, 150.0};  // wrong length
        std::vector<Epoch> tap{td.front()};
        std::vector<double> ap{4.0};

        bool threw = false;
        try {
            SpaceWeatherHistory h(td, f107, {}, tap, ap);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("non_monotonic_times_rejected", []() {
        std::vector<Epoch> td{
            Epoch::from_utc(2024, 6, 16, 0, 0, 0),
            Epoch::from_utc(2024, 6, 15, 0, 0, 0),  // out of order
        };
        std::vector<double> f107{100.0, 150.0};
        std::vector<Epoch> tap{td.front()};
        std::vector<double> ap{4.0};

        bool threw = false;
        try {
            SpaceWeatherHistory h(td, f107, {}, tap, ap);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_at_out_of_range() {
    TestRunner runner("AtOutOfRange");

    runner.add_test("epoch_before_window_throws", []() {
        SpaceWeatherHistory h = SpaceWeatherHistory::from_constant(
            sample_quiet(), one_day_window());

        Epoch before = Epoch::from_utc(2024, 6, 14, 0, 0, 0);
        bool threw = false;
        try {
            (void)h.at(before);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("epoch_after_window_throws", []() {
        SpaceWeatherHistory h = SpaceWeatherHistory::from_constant(
            sample_quiet(), one_day_window());

        Epoch after = Epoch::from_utc(2024, 6, 17, 0, 0, 0);
        bool threw = false;
        try {
            (void)h.at(after);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "SpaceWeatherHistory Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_default_construction();
    test_from_constant();
    test_explicit_series();
    test_at_out_of_range();

    return print_final_summary();
}
