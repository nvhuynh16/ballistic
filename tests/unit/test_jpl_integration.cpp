// How the planetary ephemeris (Sun/Moon/Jupiter positions from JPL's DE
// ephemerides) is wired into initialization, the force model and the
// propagators: InitConfig's optional JPL file path, the embedded DE440 default,
// physical distance and acceleration ranges, the older constructors without an
// ephemeris argument, and loading an external DE440 file when one is present
// (skipped otherwise). Bounds are order-of-magnitude physical sanity ranges.

#include "../test_utils.hpp"
#include <ballistic/ephemeris/jpl_ephemeris.hpp>
#include <ballistic/ephemeris/jpl_ephemeris.hpp>
#include <ballistic/forces/force_model.hpp>
#include <ballistic/propagators/propagator.hpp>
#include <ballistic/init.hpp>
#include <filesystem>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::ephemeris;
using namespace ballistic::forces;
using namespace ballistic::propagators;

// =============================================================================
// InitConfig JPL Path Tests
// =============================================================================

void test_init_config_jpl() {
    TestRunner runner("InitConfigJPL");

    runner.add_test("default_has_no_jpl_path", []() {
        InitConfig config;
        TEST_ASSERT_FALSE(config.jpl_ephemeris_path.has_value());
    });

    runner.add_test("can_set_jpl_path", []() {
        InitConfig config;
        config.jpl_ephemeris_path = "de440.bsp";
        TEST_ASSERT(config.jpl_ephemeris_path.has_value());
        TEST_ASSERT_EQ(config.jpl_ephemeris_path.value().string(), "de440.bsp");
    });

    runner.add_test("init_result_has_jpl_fields", []() {
        InitResult result;
        TEST_ASSERT_FALSE(result.jpl_ephemeris_loaded);
        TEST_ASSERT(result.jpl_de_version.empty());
    });

    runner.run();
}

// =============================================================================
// Initialize Without JPL Tests
// =============================================================================

void test_init_without_jpl() {
    TestRunner runner("InitWithoutJPL");

    runner.add_test("init_works_without_jpl", []() {
        InitConfig config;
        auto result = initialize(config);

        TEST_ASSERT(result.success);
        TEST_ASSERT_FALSE(result.jpl_ephemeris_loaded);  // No external file loaded
        // Embedded DE440 should be loaded by default
        TEST_ASSERT_TRUE(result.embedded_ephemeris_loaded);
        TEST_ASSERT_EQ(result.jpl_de_version, std::string("DE440 (embedded)"));
    });

    runner.add_test("missing_jpl_file_fails", []() {
        // Explicit paths are strict — missing file fails init.
        InitConfig config;
        config.jpl_ephemeris_path = "nonexistent_de440.bsp";

        auto result = initialize(config);

        TEST_ASSERT_FALSE(result.success);
        TEST_ASSERT(result.error_message.find("not found") != std::string::npos ||
                    result.error_message.find("JPL") != std::string::npos);
    });

    runner.run();
}

// =============================================================================
// Default Ephemeris Provider Tests
// =============================================================================

void test_default_ephemeris_provider() {
    TestRunner runner("DefaultEphemerisProvider");

    runner.add_test("returns_non_null", []() {
        auto provider = get_default_ephemeris_provider();
        TEST_ASSERT(provider != nullptr);
    });

    runner.add_test("returns_non_null_provider", []() {
        // After initialize({}) the SPK singleton is loaded; provider is SPK-backed.
        auto p1 = get_default_ephemeris_provider();
        auto p2 = get_default_ephemeris_provider();
        TEST_ASSERT(p1 != nullptr);
        TEST_ASSERT(p2 != nullptr);
    });

    runner.add_test("provides_sun_position", []() {
        auto provider = get_default_ephemeris_provider();
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        Vector3d sun_pos = provider->sun_position(epoch);

        // Sun should be ~150 million km away
        double sun_dist = sun_pos.norm();
        TEST_ASSERT(sun_dist > 1.4e11);  // > 140 million km
        TEST_ASSERT(sun_dist < 1.55e11); // < 155 million km
    });

    runner.add_test("provides_moon_position", []() {
        auto provider = get_default_ephemeris_provider();
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        Vector3d moon_pos = provider->moon_position(epoch);

        // Moon should be ~384,000 km away
        double moon_dist = moon_pos.norm();
        TEST_ASSERT(moon_dist > 3.5e8);  // > 350,000 km
        TEST_ASSERT(moon_dist < 4.1e8);  // < 410,000 km
    });

    runner.add_test("provides_jupiter_position", []() {
        auto provider = get_default_ephemeris_provider();
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        Vector3d jupiter_pos = provider->jupiter_position(epoch);

        // Jupiter should be 4-6 AU away (varies with orbit)
        double jupiter_dist = jupiter_pos.norm();
        TEST_ASSERT(jupiter_dist > 5e11);  // > 500 million km (3.3 AU)
        TEST_ASSERT(jupiter_dist < 1e12);  // < 1 billion km (6.7 AU)
    });

    runner.run();
}

// =============================================================================
// ForceModel Ephemeris Tests
// =============================================================================

void test_force_model_ephemeris() {
    TestRunner runner("ForceModelEphemeris");

    runner.add_test("default_constructor_uses_global", []() {
        ForceModel fm;

        // Should work without crashing
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        StateVector state(Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0));

        Vector3d third_body_acc = fm.third_body_acceleration(epoch, state);

        // Third body acceleration at LEO: ~1e-6 m/s² (Sun + Moon)
        TEST_ASSERT(third_body_acc.norm() > 1e-8);
        TEST_ASSERT(third_body_acc.norm() < 1e-4);
    });

    runner.add_test("explicit_ephemeris_constructor", []() {
        auto analytical = get_default_ephemeris_provider();
        ForceModel fm(analytical);

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        StateVector state(Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0));

        Vector3d third_body_acc = fm.third_body_acceleration(epoch, state);

        // Should produce physical acceleration
        TEST_ASSERT(third_body_acc.norm() > 1e-8);
        TEST_ASSERT(third_body_acc.norm() < 1e-4);
    });

    runner.add_test("acceleration_magnitude_reasonable", []() {
        ForceModel fm;

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        // LEO orbit position and velocity
        StateVector state(Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0));

        Vector3d third_body_acc = fm.third_body_acceleration(epoch, state);

        // Third body acceleration at LEO should be ~1e-6 m/s^2 (dominated by Moon)
        double acc_mag = third_body_acc.norm();
        TEST_ASSERT(acc_mag > 1e-8);  // Non-negligible
        TEST_ASSERT(acc_mag < 1e-4);  // But small compared to Earth gravity
    });

    runner.run();
}

// =============================================================================
// Propagator Ephemeris Tests
// =============================================================================

void test_propagator_ephemeris() {
    TestRunner runner("PropagatorEphemeris");

    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    runner.add_test("default_propagator_works", [&]() {
        RK45Propagator prop(t0, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

        // Propagate 60 seconds
        PropagationResult result = prop(t0 + 60.0);

        // Should have moved
        double dist = (result.position - pos_ecef).norm();
        TEST_ASSERT(dist > 100.0);  // Moved at least 100m in 60s
    });

    runner.add_test("propagator_with_explicit_ephemeris", [&]() {
        auto analytical = get_default_ephemeris_provider();

        RK45Propagator prop(t0, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet(),
                           true, RK45Config{}, analytical);

        PropagationResult result = prop(t0 + 60.0);

        double dist = (result.position - pos_ecef).norm();
        TEST_ASSERT(dist > 100.0);
    });

    runner.add_test("factory_function_accepts_ephemeris", [&]() {
        auto analytical = get_default_ephemeris_provider();

        auto prop = create_rk45_propagator(t0, pos_ecef, vel_ecef,
                                           BallisticModel(), WeatherConditions::quiet(),
                                           true, analytical);

        PropagationResult result = (*prop)(t0 + 60.0);

        double dist = (result.position - pos_ecef).norm();
        TEST_ASSERT(dist > 100.0);
    });

    runner.add_test("convenience_propagate_accepts_ephemeris", [&]() {
        auto analytical = get_default_ephemeris_provider();

        PropagationResult result = propagate(t0, pos_ecef, vel_ecef, t0 + 60.0,
                                             BallisticModel(), WeatherConditions::quiet(),
                                             true, analytical);

        double dist = (result.position - pos_ecef).norm();
        TEST_ASSERT(dist > 100.0);
    });

    runner.run();
}

// =============================================================================
// Backward Compatibility Tests
// =============================================================================

void test_backward_compatibility() {
    TestRunner runner("BackwardCompatibility");

    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    runner.add_test("old_propagator_constructor_works", [&]() {
        // Old-style constructor without ephemeris parameter should still work
        RK45Propagator prop(t0, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet(), true);

        PropagationResult result = prop(t0 + 60.0);
        // LEO position should be ~6-8 million meters from center
        TEST_ASSERT(result.position.norm() > 6e6);
        TEST_ASSERT(result.position.norm() < 8e6);
    });

    runner.add_test("old_factory_function_works", [&]() {
        // Factory function without ephemeris parameter
        auto prop = create_rk45_propagator(t0, pos_ecef, vel_ecef);

        PropagationResult result = (*prop)(t0 + 60.0);
        TEST_ASSERT(result.position.norm() > 6e6);
        TEST_ASSERT(result.position.norm() < 8e6);
    });

    runner.add_test("old_convenience_function_works", [&]() {
        // Convenience function without ephemeris parameter
        PropagationResult result = propagate(t0, pos_ecef, vel_ecef, t0 + 60.0);

        TEST_ASSERT(result.position.norm() > 6e6);
        TEST_ASSERT(result.position.norm() < 8e6);
    });

    runner.add_test("force_model_default_constructor_works", []() {
        // Default ForceModel should work
        ForceModel fm;

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        StateVector state(Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0));

        Vector3d acc = fm.acceleration(epoch, state);
        // LEO gravity ~8-10 m/s²
        TEST_ASSERT(acc.norm() > 8.0);
        TEST_ASSERT(acc.norm() < 10.0);
    });

    runner.run();
}

// =============================================================================
// Get Init Status Tests
// =============================================================================

void test_get_init_status() {
    TestRunner runner("GetInitStatus");

    runner.add_test("reports_jpl_status", []() {
        auto status = get_init_status();

        // Should report JPL status (loaded or not)
        // Just check that the fields are accessible
        TEST_ASSERT(status.success);

        // Either external file or embedded ephemeris should be available
        if (!status.jpl_ephemeris_loaded && !status.embedded_ephemeris_loaded) {
            // Neither loaded - version should be empty
            TEST_ASSERT(status.jpl_de_version.empty());
        } else {
            // Something is loaded - version should be set
            TEST_ASSERT_FALSE(status.jpl_de_version.empty());
        }
    });

    runner.run();
}

// =============================================================================
// JPL File Loading Tests (Skip if file not available)
// =============================================================================

void test_jpl_file_loading() {
    TestRunner runner("JPLFileLoading");

    // Common paths where DE440 might be located
    std::vector<std::filesystem::path> possible_paths = {
        BALLISTIC_PROJECT_ROOT "/data/de440.bsp",
        "data/de440.bsp",
        "../data/de440.bsp",
        "de440.bsp"
    };

    std::filesystem::path jpl_path;
    for (const auto& p : possible_paths) {
        if (std::filesystem::exists(p)) {
            jpl_path = p;
            break;
        }
    }

    runner.add_test("load_de440_via_init", [&jpl_path]() {
        if (jpl_path.empty()) {
            TEST_SKIP("DE440 file not found");
        }

        InitConfig config;
        config.jpl_ephemeris_path = jpl_path;

        auto result = initialize(config);

        TEST_ASSERT(result.success);
        TEST_ASSERT(result.jpl_ephemeris_loaded);
        TEST_ASSERT(result.jpl_de_version == "DE440" || result.jpl_de_version == "DE441");
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "JPL Ephemeris Integration Tests" << std::endl;
    std::cout << "============================================" << std::endl;

    // Required: embedded DE440 must be loaded before get_default_ephemeris_provider works.
    ballistic::initialize({});

    test_init_config_jpl();
    test_init_without_jpl();
    test_default_ephemeris_provider();
    test_force_model_ephemeris();
    test_propagator_ephemeris();
    test_backward_compatibility();
    test_get_init_status();
    test_jpl_file_loading();

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
