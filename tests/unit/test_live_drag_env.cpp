// make_drag_environment evaluates the atmosphere models live -- NRLMSIS 2.1
// (density) and HWM14 (horizontal wind) -- at the satellite's position on
// every force evaluation, using the whole loaded space-weather file: no arc,
// no duration. A Trajectory warns once on entering forecast weather and stops
// with an error at the edge of the weather. Also: forecasts are never taken
// for measurements, tabulated drag environments ("envs") warn once when
// frozen past their time span, and an empirical acceleration (EA) warns once
// past its time arc. Uses a small synthetic weather CSV written below (no
// real data). EOP (Earth Orientation Parameters) come from the committed
// table so warning counts are reproducible.

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather.hpp>
#include <filesystem>
#include <fstream>
#include <thread>

using namespace ballistic;
using namespace ballistic::test;
namespace atm = ballistic::atmosphere;

namespace {

Epoch t0() { return Epoch::from_utc(2024, 1, 15, 12, 0, 0); }
Epoch day(int d, int h = 0) { return Epoch::from_utc(2024, 1, d, h, 0, 0); }

TrajectoryConfig fast_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 8;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

const Vector3d kPos(6778137.0, 0.0, 0.0);   // ~400 km
const Vector3d kVel(0.0, 1000.0 - 494.27, 7600.0);   // Earth-fixed: inertial (0, 1000, 7600) minus omega x r

// Measured days 2024-01-08..16 (varying Ap so time matters), then two  guard-exempt(PUB-DATE): data
// CelesTrak-style forecast (PRD) days that DO carry 3-hourly Ap.
std::filesystem::path write_csv() {
    auto p = std::filesystem::temp_directory_path() / "sw_live_drag_env.csv";
    std::ofstream f(p);
    f << "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n";
    for (int d = 8; d <= 18; ++d) {
        const char* type = d <= 16 ? "OBS" : "PRD";
        const int ap = 5 + 9 * (d % 4);
        f << "2024-01-" << (d < 10 ? "0" : "") << d
          << ",0,0,0,0,0,0,0,0,0,0,0";
        for (int k = 0; k < 8; ++k) f << "," << ap + 3 * k;
        f << "," << ap << ",0,0,0," << 150 + d << ".0,0," << type << ",160.0,0,0,0\n";
    }
    return p;
}

std::shared_ptr<const atm::DragEnvironment> live_env(const Epoch& start = t0(),
                                                     bool allow_predicted = true) {
    atm::DragSetup s{.beta = atm::BetaProfile1D::constant(120.0)};
    s.allow_predicted_weather = allow_predicted;
    return atm::make_drag_environment(start, kPos, kVel, 0.0, fast_config(), s);
}

TrajectoryConfig with_env(std::shared_ptr<const atm::DragEnvironment> env) {
    TrajectoryConfig cfg = fast_config();
    cfg.drag_env = std::move(env);
    return cfg;
}

bool throws_with(const std::function<void()>& f, const std::string& needle) {
    try { f(); } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

}  // namespace

int main() {
    // Pinned EOP, so the "no Earth orientation data" note never fires and the
    // warning counts below hold on a clean checkout too.
    InitConfig init;
    init.eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
    initialize(init);
    const auto csv = write_csv();
    auto& sw = atm::SpaceWeatherData::instance();
    sw.clear();
    if (!sw.load_celestrak_csv(csv)) {
        std::cerr << "could not load " << csv << "\n";
        return 1;
    }
    TestRunner runner("live drag environment");

    runner.add_test("forecast_rows_are_not_measurements", []() {
        auto& w = atm::SpaceWeatherData::instance();
        TEST_ASSERT_NEAR(*w.last_measured_mjd(), day(16).mjd(time::TimeScale::UTC), 1e-9);
        const Epoch a = day(17, 6);
        TEST_ASSERT_THROWS((void)w.history(atm::TimeWindow{a, a + 3600.0}, false),
                           std::out_of_range);
        bool used = false;
        (void)w.history(atm::TimeWindow{a, a + 3600.0}, true, &used);
        TEST_ASSERT_TRUE(used);
        const Epoch b = day(19, 6);   // past the last row
        TEST_ASSERT_THROWS((void)w.history(atm::TimeWindow{b, b + 3600.0}), std::out_of_range);
    });

    runner.add_test("full_history_spans_the_file", []() {
        auto& w = atm::SpaceWeatherData::instance();
        const auto all = w.full_history(true);
        TEST_ASSERT_NEAR(all->valid.start - day(11), 0.0, 1e-3);   // 3-day Ap lookback
        TEST_ASSERT_NEAR(all->valid.end - day(19), 0.0, 1e-3);
        TEST_ASSERT_NEAR(*all->measured_until - day(17), 0.0, 1e-3);
        const auto measured = w.full_history(false);
        TEST_ASSERT_NEAR(measured->valid.end - day(17), 0.0, 1e-3);
        TEST_ASSERT_FALSE(measured->measured_until.has_value());
        TEST_ASSERT_TRUE(w.full_history(true) == all);              // cached
    });

    runner.add_test("live_density_is_nrlmsis_at_the_point", []() {
        auto env = live_env();
        TEST_ASSERT_TRUE(env->live_source() != nullptr);
        TEST_ASSERT_TRUE(env->uses_predicted_weather());
        // A 1D snapshot at the point's lat/lon and a node altitude is NRLMSIS
        // there, so the live value must agree to 1e-6 relative.
        const Vector3d p = coords::geodetic_to_ecef(0.3, 1.1, 400e3);
        const auto geo = coords::ecef_to_geodetic(p);
        const auto weather = atm::SpaceWeatherData::instance().full_history()->history;
        auto snap = atm::from_models_snapshot(t0(), geo.latitude * 57.29577951308232,
                                              geo.longitude * 57.29577951308232, weather,
                                              atm::BetaProfile1D::constant(120.0),
                                              {300.0, 400.0, 500.0});
        const double live = env->evaluate(p, t0()).rho_kg_m3;
        const double tab = snap.evaluate(p, t0()).rho_kg_m3;
        TEST_ASSERT_NEAR_REL(live, tab, 1e-6);
        TEST_ASSERT_TRUE(env->evaluate(p, t0()).wind_ecef_m_s.norm() > 1.0);
    });

    // Over 4 h the live env and a table built along the same path must give
    // positions within 2% of the total drag effect (which must exceed 10 m).
    runner.add_test("live_matches_a_trajectory_following_table", []() {
        const double H = 4 * 3600.0;
        Trajectory live(t0(), kPos, kVel, with_env(live_env()));
        TrajectoryConfig tcfg = fast_config();
        tcfg.drag_env = atm::share(atm::from_models_along_iterated(
            t0(), kPos, kVel, H, fast_config(),
            atm::SpaceWeatherData::instance().history(atm::TimeWindow{t0(), t0() + H}),
            atm::BetaProfile1D::constant(120.0)));
        Trajectory table(t0(), kPos, kVel, tcfg);
        Trajectory nodrag(t0(), kPos, kVel, fast_config());
        const auto rl = live.propagate(t0() + H);
        const auto rt = table.propagate(t0() + H);
        const auto rn = nodrag.propagate(t0() + H);
        const double diff = (rl.position - rt.position).norm();
        const double drag = (rn.position - rl.position).norm();
        std::cout << "    live vs table " << diff << " m (drag effect " << drag << " m)\n";
        TEST_ASSERT_TRUE(drag > 10.0);
        TEST_ASSERT_TRUE(diff < 0.02 * drag);
        TEST_ASSERT_TRUE(live.warnings().empty());
    });

    runner.add_test("entering_forecast_days_warns_once", []() {
        const Epoch start = day(16, 20);
        Trajectory tr(start, kPos, kVel, with_env(live_env(start)));
        TEST_ASSERT_TRUE(tr.warnings().empty());
        (void)tr.propagate(start + 3 * 3600.0);
        (void)tr.propagate(start + 6 * 3600.0);
        TEST_ASSERT_EQ(tr.warnings().size(), std::size_t{1});
        TEST_ASSERT_TRUE(tr.warnings()[0].find("2024-01-17 00:00:00") != std::string::npos);
    });

    runner.add_test("starting_in_forecast_days_warns_at_construction", []() {
        const Epoch start = day(17, 6);
        Trajectory tr(start, kPos, kVel, with_env(live_env(start)));
        TEST_ASSERT_EQ(tr.warnings().size(), std::size_t{1});
    });

    runner.add_test("refusing_forecasts_stops_at_the_last_measured_day", []() {
        const Epoch start = day(16, 20);
        Trajectory tr(start, kPos, kVel, with_env(live_env(start, false)));
        TEST_ASSERT_TRUE(throws_with([&] { (void)tr.propagate(start + 8 * 3600.0); },
                                     "forecasts were refused"));
        TEST_ASSERT_NEAR(tr.epoch() - day(17), 0.0, 1e-3);   // state kept at the edge
    });

    runner.add_test("past_all_weather_stops_at_its_end", []() {
        const Epoch start = day(18, 12);
        Trajectory tr(start, kPos, kVel, with_env(live_env(start)));
        TEST_ASSERT_TRUE(throws_with([&] { (void)tr.propagate(start + 24 * 3600.0); },
                                     "including CelesTrak's ~45-day forecast"));
        TEST_ASSERT_NEAR(tr.epoch() - day(19), 0.0, 1e-3);
    });

    runner.add_test("backward_before_the_weather_stops_at_its_start", []() {
        const Epoch start = day(11, 6);
        Trajectory tr(start, kPos, kVel, with_env(live_env(start)));
        TEST_ASSERT_TRUE(throws_with([&] { (void)tr.propagate(start - 12 * 3600.0); },
                                     "starts here"));
        TEST_ASSERT_NEAR(tr.epoch() - day(11), 0.0, 1e-3);
    });

    runner.add_test("steps_land_on_the_edge_then_stop", []() {
        const Epoch start = day(18, 22);
        Trajectory tr(start, kPos, kVel, with_env(live_env(start)));
        bool stopped = false;
        for (int i = 0; i < 10000 && !stopped; ++i) {
            try { (void)tr.step(); } catch (const std::runtime_error&) { stopped = true; }
        }
        TEST_ASSERT_TRUE(stopped);
        TEST_ASSERT_NEAR(tr.epoch() - day(19), 0.0, 1e-3);
    });

    runner.add_test("start_without_weather_fails_at_setup", []() {
        TEST_ASSERT_THROWS((void)live_env(day(20)), DataCoverageError);
        try { (void)live_env(day(20)); } catch (const DataCoverageError& e) {
            TEST_ASSERT(std::string(e.what()).find("has no space weather") != std::string::npos);
        }
    });

    runner.add_test("constant_weather_works_at_any_epoch", []() {
        atm::DragSetup s{.beta = atm::BetaProfile1D::constant(120.0)};
        s.weather = atm::SpaceWeatherHistory::from_constant(WeatherConditions::quiet());
        // Past the loaded weather file (ends 2024-01-18), inside the EOP table.  guard-exempt(PUB-DATE): data
        const Epoch late = Epoch::from_utc(2025, 6, 1, 0, 0, 0);
        auto env = atm::make_drag_environment(late, kPos, kVel, 0.0, fast_config(), s);
        Trajectory tr(late, kPos, kVel, with_env(env));
        (void)tr.propagate(late + 6 * 3600.0);
        TEST_ASSERT_TRUE(tr.warnings().empty());
    });

    runner.add_test("ussa76_below_the_floor", []() {
        auto env = atm::extend_density_below(*live_env(), 80.0);
        const Vector3d low = coords::geodetic_to_ecef(0.0, 0.0, 60e3);
        const Vector3d high = coords::geodetic_to_ecef(0.0, 0.0, 400e3);
        TEST_ASSERT_TRUE(env.live_source() != nullptr);
        // U.S. Standard Atmosphere 1976 (USSA76), Table I at 60 km geometric:
        // 3.0968e-4 kg/m^3 (checked to 1e-4 relative).
        TEST_ASSERT_NEAR_REL(env.evaluate(low, t0()).rho_kg_m3, 3.0968e-4, 1e-4);
        TEST_ASSERT_NEAR_REL(env.evaluate(high, t0()).rho_kg_m3,
                             live_env()->evaluate(high, t0()).rho_kg_m3, 1e-12);
    });

    runner.add_test("below_ground_raises", []() {
        const Vector3d under = coords::geodetic_to_ecef(0.3, 0.2, -2000.0);
        TEST_ASSERT_THROWS(live_env()->evaluate(under, t0()), DataCoverageError);
        const Vector3d sea = coords::geodetic_to_ecef(0.3, 0.2, 0.0);
        const double rho0 = live_env()->evaluate(sea, t0()).rho_kg_m3;   // NRLMSIS to the ground
        TEST_ASSERT(rho0 > 0.9 && rho0 < 1.5);
        // Just below the ground (a step's stages near impact) must return a
        // finite value: HWM14's span search must not loop below its lowest
        // knot.
        const Vector3d just_below = coords::geodetic_to_ecef(0.3, 0.2, -500.0);
        const auto s = live_env()->evaluate(just_below, t0());
        TEST_ASSERT(std::isfinite(s.wind_ecef_m_s.norm()) && s.rho_kg_m3 > rho0);
        // Continuous through the ground: NRLMSIS on both sides (no switch to
        // another model at 0 km, which would make a density jump no step
        // size could cross).
        const double above = live_env()->evaluate(coords::geodetic_to_ecef(0.3, 0.2, 0.01), t0()).rho_kg_m3;
        const double below = live_env()->evaluate(coords::geodetic_to_ecef(0.3, 0.2, -0.01), t0()).rho_kg_m3;
        TEST_ASSERT(std::abs(above / below - 1.0) < 1e-5);
    });

    // Two threads propagating with one shared env must give bit-identical
    // results to a serial run (the HWM14 caches are per thread).
    runner.add_test("threads_agree_with_serial", []() {
        auto env = live_env();
        auto run = [&](Vector3d& out) {
            Trajectory tr(t0(), kPos, kVel, with_env(env));
            out = tr.propagate(t0() + 3600.0).position;
        };
        Vector3d serial, a, b;
        run(serial);
        std::thread ta([&] { run(a); }), tb([&] { run(b); });
        ta.join(); tb.join();
        TEST_ASSERT_NEAR((a - serial).norm(), 0.0, 0.0);
        TEST_ASSERT_NEAR((b - serial).norm(), 0.0, 0.0);
    });

    runner.add_test("tabulated_env_warns_once_when_frozen", []() {
        auto fixed = atm::share(atm::from_models_along_iterated(
            t0(), kPos, kVel, 600.0, fast_config(),
            atm::SpaceWeatherData::instance().history(atm::TimeWindow{t0(), t0() + 600.0}),
            atm::BetaProfile1D::constant(120.0), {300.0, 400.0, 500.0}, 1));
        Trajectory tr(t0(), kPos, kVel, with_env(fixed));
        (void)tr.propagate(t0() + 1800.0);
        (void)tr.propagate(t0() + 3600.0);
        TEST_ASSERT_EQ(tr.warnings().size(), std::size_t{1});
        TEST_ASSERT_TRUE(tr.warnings()[0].find("frozen") != std::string::npos);
    });

    runner.add_test("empirical_accel_past_its_arc_warns_once", []() {
        std::vector<forces::EmpiricalAccelEntry> entries;
        for (double t : {0.0, 50.0, 100.0}) {
            forces::EmpiricalAccelEntry e;
            e.key = t;
            e.accel = Vector3d(1e-6, 0.0, 0.0);
            entries.push_back(e);
        }
        TrajectoryConfig cfg = fast_config();
        cfg.empirical_accel = std::make_shared<forces::EmpiricalAcceleration>(
            forces::EmpiricalAcceleration::from_entries(entries));
        Trajectory inside(t0(), kPos, kVel, cfg);
        (void)inside.propagate(t0() + 100.0);
        TEST_ASSERT_TRUE(inside.warnings().empty());
        Trajectory tr(t0(), kPos, kVel, cfg);
        (void)tr.propagate(t0() + 300.0);
        (void)tr.propagate(t0() + 600.0);
        TEST_ASSERT_EQ(tr.warnings().size(), std::size_t{1});
        TEST_ASSERT_TRUE(tr.warnings()[0].find("empirical acceleration") != std::string::npos);
    });

    runner.run();
    sw.clear();
    std::filesystem::remove(csv);
    return print_final_summary();
}
