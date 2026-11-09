// =============================================================================
// Staleness / clamp reporting tests
// =============================================================================
//
// Checks that out-of-date data and clamped lookups are reported, not hidden:
//  - InitResult.eop_table_stale / eop_table_end_mjd: the loaded Earth
//    orientation parameter (EOP) table ends before today;
//  - InitResult.leap_seconds_expired / leap_seconds_expiration_jd: the
//    leap-second list is past its stated expiry;
//  - DragSample.time_clamped + DragEnvironment::domain(): a tabulated drag
//    environment was queried outside its time span and returned its edge value.
//
// These are flags and warnings only: initialization still succeeds and values
// are still returned. Data: small files written by the test (a truncated copy of
// the committed finals2000A.all, a two-entry leap-seconds.list) and a hand-built
// density table.
//
// =============================================================================

#include "../test_utils.hpp"
#include "../test_helpers.hpp"
#include <ballistic/ballistic.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace ballistic;
using namespace ballistic::test;

namespace fs = std::filesystem;

namespace {

fs::path scratch_dir() {
    fs::path p = fs::temp_directory_path() /
                 ("ballistic_staleness_test_" +
                  std::to_string(helpers::current_pid()));
    fs::create_directories(p);
    return p;
}

fs::path pinned_eop_path() {
    return fs::path(__FILE__).parent_path().parent_path() /
           "data" / "eop" / "finals2000A.all";
}

// First `n_lines` of the pinned finals2000A.all — a valid EOP file whose
// table ends in 1973, i.e. unambiguously stale forever.
fs::path make_stale_eop_file(const fs::path& dir) {
    fs::path out = dir / "finals_stale.all";
    std::ifstream in(pinned_eop_path());
    std::ofstream o(out);
    std::string line;
    for (int i = 0; i < 100 && std::getline(in, line); ++i) {
        o << line << "\n";
    }
    return out;
}

// Minimal leap-seconds.list: two real entries + a chosen #@ expiry (NTP s).
fs::path make_leap_file(const fs::path& dir, const char* name,
                        long long expiry_ntp) {
    fs::path out = dir / name;
    std::ofstream o(out);
    o << "#  synthetic leap-seconds.list for staleness tests\n";
    o << "#@ " << expiry_ntp << "\n";
    o << "2272060800 10\n";   // 1972-01-01, TAI-UTC = 10
    o << "3692217600 37\n";   // 2017-01-01, TAI-UTC = 37
    return out;
}

// NTP = Unix + 2208988800. 2024-01-01 (past) and 2033-05-18 (future).  guard-exempt(PUB-DATE): data
constexpr long long kExpiredNtp = 3913056000LL;  // 2024-01-01
constexpr long long kFutureNtp  = 4208988800LL;  // 2033-05-18

}  // namespace

int main() {
    TestRunner runner("Staleness Warning Tests");

    const fs::path dir = scratch_dir();

    runner.add_test("eop_stale_sets_flag_and_warning", [&]() {
        InitConfig cfg;
        cfg.eop_path = make_stale_eop_file(dir);
        auto r = ballistic::initialize(cfg);
        TEST_ASSERT(r.success);
        TEST_ASSERT(r.eop_loaded);
        TEST_ASSERT(r.eop_table_stale);
        TEST_ASSERT(r.eop_table_end_mjd > 41000.0 && r.eop_table_end_mjd < 42000.0);
        bool has_warning = false;
        for (const auto& w : r.warnings) {
            if (w.find("EOP table ends") != std::string::npos) has_warning = true;
        }
        TEST_ASSERT(has_warning);
    });

    runner.add_test("auto_discovered_unparseable_eop_is_reported", []() {
        // Trap INIT-SKIP-BAD-FILE (docs/trap_census.md): a corrupt finals2000A in
        // the data directory must produce a warning, not be skipped silently.
        helpers::IsolatedDataDir isolated;
        const fs::path data = fs::path(std::getenv(helpers::data_dir_env_var()));
        fs::create_directories(data);
        std::ofstream(data / "finals2000A.data") << "<html>rate limited</html>\n";
        time::EOPData::instance().clear();
        auto r = ballistic::initialize({});
        TEST_ASSERT(r.success);
        TEST_ASSERT(!r.eop_loaded);
        bool reported = false;
        for (const auto& w : r.warnings)
            if (w.find("could not parse it") != std::string::npos) reported = true;
        TEST_ASSERT(reported);
    });

    runner.add_test("eop_fresh_flag_matches_table_end", [&]() {
        InitConfig cfg;
        cfg.eop_path = pinned_eop_path();
        auto r = ballistic::initialize(cfg);
        TEST_ASSERT(r.success);
        TEST_ASSERT(r.eop_loaded);
        // Pinned file's table (observations + IERS predictions) ends 2027-03.
        TEST_ASSERT(r.eop_table_end_mjd > 61000.0);
        // Wiring check that stays valid after the pinned table's end passes:
        TEST_ASSERT(r.eop_table_stale == (r.eop_table_end_mjd < current_utc_mjd()));
    });

    runner.add_test("leap_expired_sets_flag_and_warning", [&]() {
        InitConfig cfg;
        cfg.eop_path = pinned_eop_path();
        cfg.leap_seconds_path = make_leap_file(dir, "leap_expired.list", kExpiredNtp);
        auto r = ballistic::initialize(cfg);
        TEST_ASSERT(r.success);
        TEST_ASSERT(r.leap_seconds_loaded);
        TEST_ASSERT(r.leap_seconds_expired);
        TEST_ASSERT(r.leap_seconds_expiration_jd > 0.0);
        bool has_warning = false;
        for (const auto& w : r.warnings) {
            if (w.find("Leap-second table expired") != std::string::npos) has_warning = true;
        }
        TEST_ASSERT(has_warning);
    });

    runner.add_test("leap_future_expiry_no_flag", [&]() {
        InitConfig cfg;
        cfg.eop_path = pinned_eop_path();
        cfg.leap_seconds_path = make_leap_file(dir, "leap_future.list", kFutureNtp);
        auto r = ballistic::initialize(cfg);
        TEST_ASSERT(r.success);
        TEST_ASSERT(r.leap_seconds_loaded);
        TEST_ASSERT(!r.leap_seconds_expired);
        TEST_ASSERT(r.leap_seconds_expiration_jd > 0.0);
    });

    runner.add_test("leap_embedded_reports_its_iers_expiry", [&]() {
        // The embedded table carries the expiry of the IERS file it matches
        // (28 June 2027), not 0 = unknown, so it can warn once that date passes.
        time::LeapSecondTable::instance().reset_to_embedded();
        auto r = ballistic::get_init_status();
        TEST_ASSERT_NEAR(r.leap_seconds_expiration_jd, time::calendar_to_jd(2027, 6, 28), 1e-9);
    });

    runner.add_test("drag_env_time_clamp_flag_and_domain", []() {
        using namespace ballistic::atmosphere;
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Epoch t1 = t0 + 600.0;

        std::vector<double> alt{200.0, 400.0, 600.0};
        // Two time slices, three altitudes each.
        DensityProfile2D rho2d(alt, {t0, t1},
                               {1e-10, 1e-12, 1e-14,
                                2e-10, 2e-12, 2e-14});
        DragEnvironment env(rho2d,
                            WindProfile1D_ENU(alt, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}),
                            BetaProfile1D::constant(100.0));

        auto w = env.domain();
        TEST_ASSERT(w.has_value());
        TEST_ASSERT(std::abs(w->start - t0) < 1e-9);
        TEST_ASSERT(std::abs(w->end - t1) < 1e-9);

        const Vector3d pos(constants::earth::RADIUS_EQUATORIAL + 300000.0, 0.0, 0.0);

        auto inside = env.evaluate(pos, t0 + 300.0);
        TEST_ASSERT(!inside.time_clamped);
        TEST_ASSERT(inside.rho_kg_m3 > 0.0);

        auto past = env.evaluate(pos, t1 + 3600.0);
        TEST_ASSERT(past.time_clamped);
        TEST_ASSERT(past.rho_kg_m3 > 0.0);  // values still returned (clamped)

        auto before = env.evaluate(pos, t0 - 3600.0);
        TEST_ASSERT(before.time_clamped);
    });

    runner.add_test("drag_env_1d_is_time_invariant", []() {
        using namespace ballistic::atmosphere;
        std::vector<double> alt{200.0, 400.0, 600.0};
        DragEnvironment env(DensityProfile1D(alt, {1e-10, 1e-12, 1e-14}),
                            WindProfile1D_ENU(alt, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}),
                            BetaProfile1D::constant(100.0));
        TEST_ASSERT(!env.domain().has_value());
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        auto s = env.evaluate(
            Vector3d(constants::earth::RADIUS_EQUATORIAL + 300000.0, 0.0, 0.0),
            t0 + 1e6);
        TEST_ASSERT(!s.time_clamped);
    });

    runner.run();

    std::error_code ec;
    fs::remove_all(dir, ec);

    return print_final_summary();
}
