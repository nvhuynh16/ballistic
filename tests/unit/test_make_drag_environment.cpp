// Tests for make_drag_environment, the one call that turns drag on (live
// NRLMSIS 2.1 density + HWM14 wind evaluated at the satellite's position), and
// its helpers: SpaceWeatherHistory::from_constant without a time window,
// BetaProfile1D::from_mass_cd_area (β = m/(Cd·A), the ballistic coefficient in
// kg/m²) and share(). Space weather (F10.7 solar flux, Ap geomagnetic index) is
// either a constant or a small CSV written by the test.

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather.hpp>
#include <filesystem>
#include <fstream>

using namespace ballistic;
using namespace ballistic::test;
namespace atm = ballistic::atmosphere;

namespace {

Epoch t0() { return Epoch::from_utc(2024, 1, 15, 12, 0, 0); }

TrajectoryConfig fast_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 8;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

const Vector3d kPos(6778137.0, 0.0, 0.0);   // ~400 km
const Vector3d kVel(0.0, 7668.55, 0.0);
const std::vector<double> kGrid{300.0, 350.0, 400.0, 450.0, 500.0};

// Synthetic storm-level weather around t0 (2024-01-15), rows marked as observed  guard-exempt(PUB-DATE): data
// (OBS): Ap 80 in every bin, F10.7 200 — far from quiet, so "loaded weather was
// used" is unambiguous.
std::filesystem::path write_active_csv() {
    auto p = std::filesystem::temp_directory_path() / "sw_make_drag_env.csv";
    std::ofstream f(p);
    f << "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n";
    for (int d = 10; d <= 17; ++d)
        f << "2024-01-" << d << ",0,0,0,0,0,0,0,0,0,0,0,80,80,80,80,80,80,80,80,80,0,0,0,"
          << "200.0,0,OBS,180.0,0,0,0\n";
    return p;
}

}  // namespace

int main() {
    initialize({});
    TestRunner runner("make_drag_environment");

    runner.add_test("windowless_from_constant_never_throws", []() {
        auto sw = atm::SpaceWeatherHistory::from_constant(WeatherConditions::quiet());
        TEST_ASSERT_TRUE(sw.is_flat());
        TEST_ASSERT_FALSE(sw.empty());
        // Epochs far outside any plausible window, including exactly at the end
        // of one (a windowed history would need the window padded there).
        auto w1 = sw.at(t0());
        auto w2 = sw.at(t0() + 86400.0 * 365.0);
        auto w3 = sw.at(t0() - 86400.0 * 365.0);
        TEST_ASSERT_NEAR(w1.f107_daily, 70.0, 1e-12);
        TEST_ASSERT_NEAR(w2.f107_daily, w1.f107_daily, 0.0);
        TEST_ASSERT_NEAR(w3.ap_daily, 4.0, 1e-12);
        // ap_history flattened to ap_daily, matching the windowed overload.
        for (double a : w1.ap_history) TEST_ASSERT_NEAR(a, 4.0, 1e-12);
    });

    runner.add_test("windowless_matches_windowed_values", []() {
        WeatherConditions w = WeatherConditions::active();
        auto flat = atm::SpaceWeatherHistory::from_constant(w);
        auto windowed = atm::SpaceWeatherHistory::from_constant(
            w, atm::TimeWindow{t0(), t0() + 3600.0});
        auto a = flat.at(t0() + 1800.0);
        auto b = windowed.at(t0() + 1800.0);
        TEST_ASSERT_NEAR(a.f107_daily, b.f107_daily, 0.0);
        TEST_ASSERT_NEAR(a.f107_average, b.f107_average, 0.0);
        TEST_ASSERT_NEAR(a.ap_daily, b.ap_daily, 0.0);
        for (int i = 0; i < 7; ++i)
            TEST_ASSERT_NEAR(a.ap_history[i], b.ap_history[i], 0.0);
    });

    runner.add_test("from_mass_cd_area", []() {
        // Satellite registry S1A: 2200 kg / (2.2 * 6.35 m^2) = 157.5 kg/m^2
        const auto& s1 = ballistic::satellites::get("S1A");
        auto beta = atm::BetaProfile1D::from_mass_cd_area(s1.mass_kg, s1.cd, s1.drag_area_m2);
        double b = beta.evaluate(kPos, t0());
        TEST_ASSERT_NEAR(b, s1.beta_kg_m2(), 1e-9);
        TEST_ASSERT_THROWS(atm::BetaProfile1D::from_mass_cd_area(0.0, s1.cd, s1.drag_area_m2),  // guard-exempt(SAT-LITERAL): invalid mass on purpose
                           std::invalid_argument);
    });

    // The registry says how each value was obtained (FitKind): names round
    // trip, fitted values are marked as fits, legacy rows exist and are named.
    runner.add_test("satellite_fit_kinds", []() {
        namespace sat = ballistic::satellites;
        using K = sat::FitKind;
        TEST_ASSERT(sat::fit_kind_name(K::Physical) == "physical");
        TEST_ASSERT(sat::fit_kind_name(K::Fitted) == "fitted");
        TEST_ASSERT(sat::fit_kind_name(K::Nominal) == "nominal");
        TEST_ASSERT(sat::fit_kind_name(K::Legacy) == "legacy");
        static_assert(sat::get("CHAMP").drag_kind == K::Fitted);
        static_assert(sat::get("GOCE").drag_kind == K::Fitted);
        static_assert(sat::get("GRACE-FO-1").drag_kind == K::Physical);
        static_assert(sat::get("S1A").srp_kind == K::Fitted);
        int fitted_srp = 0, fitted_drag = 0, legacy = 0;
        for (const auto& s : sat::all()) {
            fitted_srp += s.srp_kind == K::Fitted;
            fitted_drag += s.drag_kind == K::Fitted;
            legacy += (s.drag_kind == K::Legacy || s.srp_kind == K::Legacy);
        }
        TEST_ASSERT_EQ(fitted_srp, static_cast<int>(sat::kSrpFit.size()));
        TEST_ASSERT_EQ(fitted_drag, 2);   // CHAMP, GOCE
        TEST_ASSERT(legacy > 0);
    });

    // The one-call env evaluates the models live rather than tabulating
    // them along an arc: at a grid altitude it equals a snapshot
    // profile taken at the same point, at every epoch.
    runner.add_test("one_call_is_the_models_at_the_point", []() {
        TrajectoryConfig cfg = fast_config();
        auto beta = atm::BetaProfile1D::constant(66.2);
        auto sw = atm::SpaceWeatherHistory::from_constant(WeatherConditions::quiet());
        atm::IterationDiag diag;
        auto env = atm::make_drag_environment(
            t0(), kPos, kVel, 600.0, cfg, {.beta = beta, .weather = sw}, &diag);
        TEST_ASSERT_TRUE(env != nullptr && env->live_source() != nullptr);
        TEST_ASSERT_EQ(diag.iterations, 0);
        const auto geo = coords::ecef_to_geodetic(kPos);
        for (double dt : {0.0, 300.0, 86400.0}) {
            const Epoch t = t0() + dt;
            auto snap = atm::from_models_snapshot(t, geo.latitude * 57.29577951308232,
                                                  geo.longitude * 57.29577951308232,
                                                  sw, beta, kGrid);
            TEST_ASSERT_NEAR_REL(env->evaluate(kPos, t).rho_kg_m3,
                                 snap.evaluate(kPos, t).rho_kg_m3, 1e-6);
            TEST_ASSERT_NEAR(env->evaluate(kPos, t).beta_kg_m2, 66.2, 0.0);
        }
    });

    // The default weather is the loaded MEASURED weather, and nothing loaded
    // is an error rather than a substitution (a silent QUIET snapshot would
    // bias density 2-3x low).

    runner.add_test("default_weather_without_loaded_data_throws", []() {
        atm::SpaceWeatherData::instance().clear();
        TEST_ASSERT_THROWS(
            (void)atm::make_drag_environment(
                t0(), kPos, kVel, 300.0, fast_config(),
                {.beta = atm::BetaProfile1D::constant(100.0),
                 .altitude_grid_km = kGrid}),
            std::runtime_error);
    });

    runner.add_test("default_weather_is_loaded_measured_history", []() {
        auto p = write_active_csv();
        auto& sw = atm::SpaceWeatherData::instance();
        sw.clear();
        TEST_ASSERT(sw.load_celestrak_csv(p));
        TrajectoryConfig cfg = fast_config();
        auto env = atm::make_drag_environment(
            t0(), kPos, kVel, 300.0, cfg,
            {.beta = atm::BetaProfile1D::constant(100.0),
             .altitude_grid_km = kGrid});
        auto env_loaded = atm::make_drag_environment(
            t0(), kPos, kVel, 300.0, cfg,
            {.beta = atm::BetaProfile1D::constant(100.0),
             .weather = sw.history(atm::TimeWindow{t0(), t0() + 300.0}),
             .altitude_grid_km = kGrid});
        auto env_quiet = atm::make_drag_environment(
            t0(), kPos, kVel, 300.0, cfg,
            {.beta = atm::BetaProfile1D::constant(100.0),
             .weather = atm::SpaceWeatherHistory::from_constant(
                 WeatherConditions::quiet()),
             .altitude_grid_km = kGrid});
        const double rho = env->evaluate(kPos, t0() + 150.0).rho_kg_m3;
        TEST_ASSERT_NEAR(rho, env_loaded->evaluate(kPos, t0() + 150.0).rho_kg_m3, 0.0);
        TEST_ASSERT(rho > 2.0 * env_quiet->evaluate(kPos, t0() + 150.0).rho_kg_m3);
        sw.clear();
        std::filesystem::remove(p);
    });

    runner.add_test("env_drives_trajectory_drag", []() {
        TrajectoryConfig cfg = fast_config();
        cfg.drag_env = atm::make_drag_environment(
            t0(), kPos, kVel, 600.0, cfg,
            {.beta = atm::BetaProfile1D::constant(66.2),
             .weather = atm::SpaceWeatherHistory::from_constant(
                 WeatherConditions::quiet()),
             .altitude_grid_km = kGrid});

        Trajectory with_drag(t0(), kPos, kVel, cfg);
        auto flags = with_drag.force_model().force_flags();
        TEST_ASSERT_TRUE(flags.drag);  // auto-derived from the env

        TrajectoryConfig nodrag = fast_config();
        Trajectory without(t0(), kPos, kVel, nodrag);

        auto r1 = with_drag.propagate(t0() + 600.0);
        auto r2 = without.propagate(t0() + 600.0);
        double delta = (r1.position - r2.position).norm();
        // Drag at 400 km over 10 min is small but must be nonzero.
        TEST_ASSERT_TRUE(delta > 1e-4);
        TEST_ASSERT_TRUE(delta < 100.0);
    });

    runner.add_test("share_wraps_generator_output", []() {
        auto sw = atm::SpaceWeatherHistory::from_constant(WeatherConditions::quiet());
        auto env = atm::share(atm::from_models_snapshot(
            t0(), 0.0, 0.0, sw, atm::BetaProfile1D::constant(50.0), kGrid));
        TEST_ASSERT_TRUE(env != nullptr);
        auto s = env->evaluate(kPos, t0());
        TEST_ASSERT_TRUE(s.rho_kg_m3 > 0.0);
        TEST_ASSERT_NEAR(s.beta_kg_m2, 50.0, 1e-9);
    });

    runner.run();
    return print_final_summary();
}
