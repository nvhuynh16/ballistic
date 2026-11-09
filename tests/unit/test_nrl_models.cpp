// The NRL models plugin (atmosphere/nrl_models.hpp): NRLMSIS 2.1 and HWM14 are
// not compiled into the library but loaded at run time from ballistic_nrl.
//
// The cells run in order on one process state: first with no plugin anywhere
// (the environment variable is cleared and the data directory is an empty
// temporary one), where every model-based drag builder must refuse with a
// ConfigError that says how to install the models; then the plugin built with
// the tests is loaded, and calls through it must give exactly the numbers of
// the model code called directly.

#include "../test_utils.hpp"
#include "../test_helpers.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/atmosphere/nrl_models.hpp>
#include <ballistic/atmosphere/internal/hwm14.hpp>
#include <ballistic/atmosphere/internal/nrlmsis21_full.hpp>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
namespace atm = ballistic::atmosphere;
namespace nrl = ballistic::atmosphere::nrl;

namespace {

Epoch t0() { return Epoch::from_utc(2024, 1, 15, 12, 0, 0); }

TrajectoryConfig fast_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 8;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    cfg.ocean_tide_degree = 0;
    return cfg;
}

// ~400 km circular orbit, Earth-fixed velocity.
const Vector3d kPos(6778137.0, 0.0, 0.0);
const Vector3d kVel(0.0, 7175.0, 0.0);

atm::DragSetup quiet_setup() {
    atm::DragSetup setup{.beta = atm::BetaProfile1D::constant(150.0)};
    setup.weather = atm::SpaceWeatherHistory::from_constant(WeatherConditions::quiet());
    return setup;
}

void unset_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    ::unsetenv(name);
#endif
}

bool contains(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

}  // namespace

int main() {
    // The plugin the tests were built with; then make sure nothing points at it
    // yet (test_env.cpp set BALLISTIC_NRL_MODELS during static init).
    const std::string built_plugin = BALLISTIC_TEST_NRL_PLUGIN;
    unset_env("BALLISTIC_NRL_MODELS");
    helpers::IsolatedDataDir isolated;   // no plugin in the data directory either

    InitConfig init;
    init.eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
    initialize(init);

    TestRunner runner("NRL models plugin");

    runner.add_test("not_installed_model_drag_says_how_to_install", []() {
        TEST_ASSERT(!nrl::try_autoload());
        TEST_ASSERT(!nrl::status().loaded);
        TEST_ASSERT(!get_init_status().nrl_models_loaded);
        const TrajectoryConfig cfg = fast_config();
        bool threw = false;
        try {
            (void)atm::make_drag_environment(t0(), kPos, kVel, 0.0, cfg, quiet_setup());
        } catch (const ConfigError& e) {
            threw = true;
            const std::string msg = e.what();
            TEST_ASSERT_MSG(contains(msg, "make_drag_environment"), msg);
            TEST_ASSERT_MSG(contains(msg, "install_nrl_models(accept_license=True)"), msg);
            TEST_ASSERT_MSG(contains(msg, "NRL-SOF-014-1"), msg);
            TEST_ASSERT_MSG(contains(msg, "non-profit"), msg);
            TEST_ASSERT_MSG(contains(msg, nrl::default_install_path().string().c_str()), msg);
            TEST_ASSERT_MSG(contains(msg, "BALLISTIC_NRL_MODELS"), msg);
            TEST_ASSERT_MSG(contains(msg, "from_accelerometer_density"), msg);
        }
        TEST_ASSERT(threw);
        const auto sw = atm::SpaceWeatherHistory::from_constant(WeatherConditions::quiet());
        const auto beta = atm::BetaProfile1D::constant(150.0);
        TEST_ASSERT_THROWS((void)atm::from_models_snapshot(t0(), 0.0, 0.0, sw, beta), ConfigError);
        TEST_ASSERT_THROWS((void)atm::from_models_along({t0(), t0() + 60.0}, {kPos, kPos}, sw, beta),
                           ConfigError);
        TEST_ASSERT_THROWS((void)atm::from_models_4d({400.0}, {0.0}, {0.0}, {t0()}, sw, beta),
                           ConfigError);
        TEST_ASSERT_THROWS((void)atm::from_models_along_iterated(t0(), kPos, kVel, 600.0, cfg, sw, beta),
                           ConfigError);
        TEST_ASSERT_THROWS((void)atm::LiveModelDensity(std::make_shared<atm::LiveModelSource>()),
                           ConfigError);
        TEST_ASSERT_THROWS((void)atm::LiveModelWind(std::make_shared<atm::LiveModelSource>()),
                           ConfigError);
    });

    runner.add_test("measured_density_drag_needs_no_plugin", []() {
        std::vector<Epoch> epochs;
        std::vector<double> rho;
        for (int i = 0; i <= 12; ++i) {
            epochs.push_back(t0() + 600.0 * i);
            rho.push_back(3e-12 * (1.0 + 0.2 * std::sin(0.5 * i)));
        }
        const auto env = std::make_shared<const atm::DragEnvironment>(
            atm::from_accelerometer_density(epochs, rho, atm::BetaProfile1D::constant(150.0)));
        TrajectoryConfig with_drag = fast_config();
        with_drag.drag_env = env;
        Trajectory a(InitialState::from_itrf(t0(), kPos, kVel), with_drag);
        Trajectory b(InitialState::from_itrf(t0(), kPos, kVel), fast_config());
        const double moved = (a.propagate(t0() + 3600.0).position -
                              b.propagate(t0() + 3600.0).position).norm();
        std::cout << "(measured-density drag moved the 1-h position by " << moved << " m) ";
        TEST_ASSERT(moved > 0.1 && moved < 1000.0);
        TEST_ASSERT(!nrl::status().loaded);
    });

    runner.add_test("unloadable_files_are_refused_with_the_reason", []() {
        const auto dir = std::filesystem::temp_directory_path() /
                         ("ballistic_nrl_test_" + std::to_string(helpers::current_pid()));
        std::filesystem::create_directories(dir);
        const auto missing = dir / "absent.so";
        const auto text = dir / "not_a_library.so";
        { std::ofstream(text) << "not a shared library\n"; }
        try {
            nrl::load(missing);
            TEST_ASSERT_MSG(false, "loading a missing file did not throw");
        } catch (const ConfigError& e) {
            TEST_ASSERT_MSG(contains(e.what(), "no file at"), e.what());
        }
        try {
            nrl::load(text);
            TEST_ASSERT_MSG(false, "loading a text file did not throw");
        } catch (const ConfigError& e) {
            TEST_ASSERT_MSG(contains(e.what(), "not a loadable library"), e.what());
        }
        InitConfig bad;
        bad.nrl_models_path = missing;
        const InitResult r = initialize(bad);
        TEST_ASSERT(!r.success);
        TEST_ASSERT_MSG(contains(r.error_message, "no file at"), r.error_message);
        TEST_ASSERT(!nrl::status().loaded);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    });

    runner.add_test("initialize_loads_the_plugin_from_its_path", [&built_plugin]() {
        InitConfig cfg;
        cfg.eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
        cfg.nrl_models_path = built_plugin;
        const InitResult r = initialize(cfg);
        TEST_ASSERT_MSG(r.success, r.error_message);
        TEST_ASSERT(r.nrl_models_loaded);
        const auto st = nrl::status();
        TEST_ASSERT(st.loaded);
        TEST_ASSERT(st.models == "NRLMSIS 2.1, HWM14");
        TEST_ASSERT(std::filesystem::equivalent(st.path, built_plugin));
        nrl::load(built_plugin);   // the same file again: no-op
        const auto other = std::filesystem::temp_directory_path() / "elsewhere_ballistic_nrl.so";
        TEST_ASSERT_THROWS(nrl::load(other), ConfigError);
    });

    runner.add_test("plugin_matches_the_models_called_directly", []() {
        const atm::NRLMSIS21 msis;
        atm::HWM14 hwm;
        hwm.init();
        const double alts[] = {0.0, 85.0, 160.0, 400.0, 800.0};
        const double lats[] = {-75.0, 0.0, 45.0};
        int n = 0;
        for (double alt : alts) {
            for (double lat : lats) {
                atm::NRLMSIS21::Input in{};
                in.day = 172.5;
                in.utsec = 29000.0;
                in.alt = alt;
                in.lat = lat;
                in.lon = -70.0;
                in.f107a = 150.0;
                in.f107 = 140.0;
                in.ap = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
                const auto direct = msis.msiscalc(in);
                BallisticNrlMsisInput pin{in.day, in.utsec, in.alt, in.lat, in.lon, in.f107a,
                                          in.f107, {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}};
                const auto via = nrl::msis(pin);
                TEST_ASSERT(via.tn == direct.tn && via.tex == direct.tex);
                for (int i = 1; i <= 10; ++i) TEST_ASSERT(via.dn[i] == direct.dn[i]);

                float ap[2] = {4.0f, 27.0f};
                float w_direct[2], w_via[2];
                hwm.hwm14_calc(2024172, 29000.0f, static_cast<float>(alt), static_cast<float>(lat),
                               -70.0f, 3.4f, 150.0f, 140.0f, ap, w_direct);
                BallisticNrlHwmInput hin{2024172, 29000.0f, static_cast<float>(alt),
                                         static_cast<float>(lat), -70.0f, 3.4f, 150.0f, 140.0f,
                                         {4.0f, 27.0f}};
                nrl::hwm14(hin, w_via);
                TEST_ASSERT(w_via[0] == w_direct[0] && w_via[1] == w_direct[1]);
                ++n;
            }
        }
        std::cout << "(" << n << " points bit-identical) ";
    });

    runner.add_test("model_drag_works_once_loaded", []() {
        const auto env = atm::make_drag_environment(t0(), kPos, kVel, 0.0, fast_config(),
                                                    quiet_setup());
        TEST_ASSERT(env != nullptr && env->live_source() != nullptr);
        TEST_ASSERT(env->evaluate(kPos, t0()).rho_kg_m3 > 0.0);
    });

    runner.run();
    return print_final_summary();
}
