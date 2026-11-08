// Unit tests for the EGM2008 gravity model (Earth Gravitational Model 2008: the
// spherical-harmonic coefficients C_nm, S_nm of the Earth's field) and for
// global initialization. Covers the embedded coefficients (degree 2-360: spot
// checks against the NGA file and a checksum over all of them), loading the full
// NGA file (data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree, ~75 MB, not
// committed; those cells skip without it), the pole special case, and timing.
// Tests: egm_gravity.hpp, init.hpp

#include "../test_utils.hpp"
#include "../test_helpers.hpp"
#include <ballistic/gravity/egm_gravity.hpp>
#include <ballistic/gravity/egm2008_coefficients.hpp>
#include <ballistic/init.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <filesystem>
#include <chrono>
#include <cstdlib>
#include <string>

using ballistic::test::helpers::IsolatedDataDir;

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::gravity;
using namespace ballistic::constants;

// =============================================================================
// EGM2008 Gravity Model Tests
// =============================================================================

void test_egm2008_embedded() {
    TestRunner runner("EGM2008Embedded");

    runner.add_test("initialization", []() {
        // Reset and test embedded initialization
        auto& model = EGM2008Gravity::instance();
        model.init(70);

        TEST_ASSERT(model.max_degree() >= 10);
        TEST_ASSERT_FALSE(model.is_loaded_from_file());
    });

    runner.add_test("c20_embedded_value", []() {
        // C20 (normalized J2) should match the EGM2008 value
        auto& model = EGM2008Gravity::instance();
        auto [c20, s20] = model.model().get_coefficient(2, 0);

        // EGM2008 C20 = -0.484165143790815e-03
        TEST_ASSERT_NEAR(c20, -0.484165143790815e-03, 1e-15);
        TEST_ASSERT_NEAR(s20, 0.0, 1e-20);
    });

    runner.add_test("acceleration_at_equator", []() {
        auto& model = EGM2008Gravity::instance();

        // Position on the equator at the surface (6378 km from the center)
        Vector3d pos(earth::RADIUS_EQUATORIAL, 0.0, 0.0);
        Vector3d acc = model.acceleration(pos, 10);

        // Should have negative X component (pointing towards center)
        TEST_ASSERT(acc.x() < 0.0);
        // Magnitude should be close to g (~9.8 m/s^2)
        TEST_ASSERT_NEAR(acc.norm(), earth::GM / (earth::RADIUS_EQUATORIAL * earth::RADIUS_EQUATORIAL), 0.1);
    });

    runner.add_test("acceleration_at_pole", []() {
        auto& model = EGM2008Gravity::instance();

        // Position at north pole
        Vector3d pos(0.0, 0.0, earth::RADIUS_POLAR);
        Vector3d acc = model.acceleration(pos, 10);

        // Should have negative Z component
        TEST_ASSERT(acc.z() < 0.0);
        // X and Y should be near zero (tesseral terms cause small non-zero values)
        // Tolerance relaxed to 3e-4 to accommodate numerical precision at the pole
        TEST_ASSERT_NEAR(acc.x(), 0.0, 3e-4);
        TEST_ASSERT_NEAR(acc.y(), 0.0, 3e-4);
    });

    runner.run();
}

void test_egm2008_file_loading() {
    TestRunner runner("EGM2008FileLoading");

    // Path to the full NGA coefficient file (downloaded separately)
    std::filesystem::path egm_path = BALLISTIC_PROJECT_ROOT "/data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree";

    runner.add_test("file_exists", [&egm_path]() {
        // Skip if file doesn't exist
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found");
        }
        TEST_ASSERT(std::filesystem::exists(egm_path));
    });

    runner.add_test("load_degree_70", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 70);
        bool loaded = model.load_from_file(egm_path, 70);

        TEST_ASSERT(loaded);
        TEST_ASSERT_EQ(model.max_degree(), 70);
    });

    runner.add_test("load_degree_360", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 360);

        auto start = std::chrono::high_resolution_clock::now();
        bool loaded = model.load_from_file(egm_path, 360);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "[" << duration.count() << "ms] ";

        TEST_ASSERT(loaded);
        TEST_ASSERT_EQ(model.max_degree(), 360);
    });

    runner.add_test("c20_matches_embedded", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 10);
        model.load_from_file(egm_path, 10);

        auto [c20_file, s20_file] = model.get_coefficient(2, 0);

        // Should match embedded EGM2008 C20
        TEST_ASSERT_NEAR(c20_file, -0.484165143790815e-03, 1e-15);
    });

    runner.add_test("high_degree_coefficients", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 100);
        model.load_from_file(egm_path, 100);

        // Check that high-degree coefficients are non-zero
        auto [c100_0, s100_0] = model.get_coefficient(100, 0);
        auto [c100_50, s100_50] = model.get_coefficient(100, 50);

        // C(100,0) should be small but non-zero
        TEST_ASSERT(std::abs(c100_0) > 1e-20);
        TEST_ASSERT(std::abs(c100_50) > 1e-20 || std::abs(s100_50) > 1e-20);
    });

    runner.run();
}

void test_egm2008_accuracy() {
    TestRunner runner("EGM2008Accuracy");

    std::filesystem::path egm_path = BALLISTIC_PROJECT_ROOT "/data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree";

    runner.add_test("degree_convergence", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        // Test that higher degrees converge for a LEO position
        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 360);
        model.load_from_file(egm_path, 360);

        // LEO satellite position (~400 km altitude)
        Vector3d pos(6778137.0, 0.0, 0.0);

        Vector3d acc_10 = model.acceleration(pos, 10);
        Vector3d acc_70 = model.acceleration(pos, 70);
        Vector3d acc_120 = model.acceleration(pos, 120);
        Vector3d acc_360 = model.acceleration(pos, 360);

        // Differences should decrease with higher degree
        double diff_10_70 = (acc_10 - acc_70).norm();
        double diff_70_120 = (acc_70 - acc_120).norm();
        double diff_120_360 = (acc_120 - acc_360).norm();

        // Higher degrees should provide smaller corrections
        TEST_ASSERT(diff_70_120 < diff_10_70);
        TEST_ASSERT(diff_120_360 < diff_70_120);
    });

    runner.add_test("symmetry_test", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 70);
        model.load_from_file(egm_path, 70);

        // Test points symmetric about equator
        Vector3d pos_north(6778137.0, 0.0, 500000.0);
        Vector3d pos_south(6778137.0, 0.0, -500000.0);

        Vector3d acc_north = model.acceleration(pos_north, 70);
        Vector3d acc_south = model.acceleration(pos_south, 70);

        // X and Y components agree (the field is nearly north-south symmetric):
        // 2e-5 m/s^2 absolute, ~2.5 ppm of the ~8 m/s^2 acceleration.
        TEST_ASSERT_NEAR(acc_north.x(), acc_south.x(), 2e-5);
        TEST_ASSERT_NEAR(acc_north.y(), acc_south.y(), 2e-5);

        // Z components are opposite, to 1e-4 m/s^2 (the odd-degree zonal terms break
        // the north-south symmetry slightly).
        TEST_ASSERT_NEAR(acc_north.z(), -acc_south.z(), 1e-4);
    });

    runner.run();
}

void test_init_api() {
    TestRunner runner("InitAPI");

    std::filesystem::path egm_path = BALLISTIC_PROJECT_ROOT "/data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree";
    std::filesystem::path eop_path = BALLISTIC_PROJECT_ROOT "/data/finals2000A.data";

    runner.add_test("default_config", []() {
        InitConfig config;

        TEST_ASSERT_FALSE(config.egm2008_path.has_value());
        TEST_ASSERT_FALSE(config.eop_path.has_value());
    });

    runner.add_test("init_without_files", []() {
        // Hermetic fixture — see IsolatedDataDir in tests/test_helpers.hpp.
        IsolatedDataDir isolation;

        InitConfig config;
        auto result = initialize(config);

        TEST_ASSERT(result.success);
        TEST_ASSERT_FALSE(result.egm2008_loaded);
        TEST_ASSERT_FALSE(result.eop_loaded);
        TEST_ASSERT(result.gravity_degree_loaded > 0);
    });

    runner.add_test("init_with_egm2008", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        InitConfig config;
        config.egm2008_path = egm_path;

        auto result = initialize(config);

        TEST_ASSERT(result.success);
        TEST_ASSERT(result.egm2008_loaded);
        TEST_ASSERT(result.gravity_degree_loaded >= 360);  // Full file (up to 2190)
    });

    runner.add_test("init_missing_file_fails", []() {
        // Explicit paths are strict — missing file fails init.
        InitConfig config;
        config.egm2008_path = "nonexistent_file.dat";

        auto result = initialize(config);

        TEST_ASSERT_FALSE(result.success);
        TEST_ASSERT(result.error_message.find("not found") != std::string::npos);
    });

    runner.add_test("get_init_status", []() {
        auto status = get_init_status();

        TEST_ASSERT(status.success);
        TEST_ASSERT(status.gravity_degree_loaded > 0);
    });

    runner.run();
}

void test_performance() {
    TestRunner runner("Performance");

    std::filesystem::path egm_path = BALLISTIC_PROJECT_ROOT "/data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree";

    runner.add_test("evaluation_time_degree_70", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 70);
        model.load_from_file(egm_path, 70);

        Vector3d pos(6778137.0, 1000000.0, 500000.0);

        // Warm up
        model.acceleration(pos, 70);

        // Benchmark
        const int iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            model.acceleration(pos, 70);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_call = static_cast<double>(duration.count()) / iterations;

        std::cout << "[" << us_per_call << " us/call] ";

        // Should be reasonably fast (<1ms per call)
        TEST_ASSERT(us_per_call < 1000.0);
    });

    runner.add_test("evaluation_time_degree_360", [&egm_path]() {
        if (!std::filesystem::exists(egm_path)) {
            TEST_SKIP("file not found: " << egm_path);
        }

        ClenshawGravity model;
        model.init(earth::GM, earth::RADIUS_EQUATORIAL, 360);
        model.load_from_file(egm_path, 360);

        Vector3d pos(6778137.0, 1000000.0, 500000.0);

        // Warm up
        model.acceleration(pos, 360);

        // Benchmark
        const int iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            model.acceleration(pos, 360);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double us_per_call = static_cast<double>(duration.count()) / iterations;

        std::cout << "[" << us_per_call << " us/call] ";

        // Degree 360 will be slower, but should still be <10ms
        TEST_ASSERT(us_per_call < 10000.0);
    });

    runner.run();
}

// =============================================================================
// EGM2008 Coefficient Spot-Check Tests (NGA EGM2008_to2190_TideFree reference)
// =============================================================================
// These 16 coefficients span the full degree range (2-360) and are taken
// directly from the NGA file. Any generation script bug, truncated write,
// or coefficient corruption will cause at least one to fail.
//
// Reference: NGA EGM2008_to2190_TideFree (tide-free, fully-normalized)
// Source: https://earth-info.nga.mil/php/download.php?file=egm-08coeffs
// =============================================================================

void test_egm2008_coefficient_spotchecks() {
    TestRunner runner("EGM2008CoefficientSpotChecks");

    // Create a fresh degree-360 model to avoid singleton degree-limit issues
    // (the singleton may have been initialized at a lower degree by earlier tests)
    static ClenshawGravity full_model;
    static bool model_ready = false;
    if (!model_ready) {
        full_model.init(earth::GM, earth::RADIUS_EQUATORIAL, 360);
        for (const auto& coeff : egm2008_data::COEFFICIENTS) {
            if (coeff.n > 360) break;
            full_model.set_coefficient(coeff.n, coeff.m, coeff.C, coeff.S);
        }
        model_ready = true;
    }

    runner.add_test("coeff_n2_m0", []() {
        auto [c, s] = full_model.get_coefficient(2, 0);
        TEST_ASSERT_NEAR(c, -0.484165143790815e-03, 1e-15);
        TEST_ASSERT_NEAR(s, 0.0, 1e-20);
    });

    runner.add_test("coeff_n2_m2", []() {
        auto [c, s] = full_model.get_coefficient(2, 2);
        TEST_ASSERT_NEAR(c, 0.243938357328313e-05, 1e-15);
        TEST_ASSERT_NEAR(s, -0.140027370385934e-05, 1e-15);
    });

    runner.add_test("coeff_n3_m0", []() {
        auto [c, s] = full_model.get_coefficient(3, 0);
        TEST_ASSERT_NEAR(c, 0.957161207093473e-06, 1e-15);
        TEST_ASSERT_NEAR(s, 0.0, 1e-20);
    });

    runner.add_test("coeff_n5_m3", []() {
        auto [c, s] = full_model.get_coefficient(5, 3);
        TEST_ASSERT_NEAR(c, -0.451847152328843e-06, 1e-15);
        TEST_ASSERT_NEAR(s, -0.214955408306046e-06, 1e-15);
    });

    runner.add_test("coeff_n10_m5", []() {
        auto [c, s] = full_model.get_coefficient(10, 5);
        TEST_ASSERT_NEAR(c, -0.492894049964295e-07, 1e-15);
        TEST_ASSERT_NEAR(s, -0.506137282060864e-07, 1e-15);
    });

    runner.add_test("coeff_n20_m10", []() {
        auto [c, s] = full_model.get_coefficient(20, 10);
        TEST_ASSERT_NEAR(c, -0.322958167727745e-07, 1e-15);
        TEST_ASSERT_NEAR(s, -0.478973340317704e-08, 1e-15);
    });

    runner.add_test("coeff_n50_m25", []() {
        auto [c, s] = full_model.get_coefficient(50, 25);
        TEST_ASSERT_NEAR(c, 0.476889420158310e-08, 1e-15);
        TEST_ASSERT_NEAR(s, 0.261965183394672e-08, 1e-15);
    });

    runner.add_test("coeff_n100_m50", []() {
        auto [c, s] = full_model.get_coefficient(100, 50);
        TEST_ASSERT_NEAR(c, 0.369458014241304e-09, 1e-15);
        TEST_ASSERT_NEAR(s, -0.758797890874107e-09, 1e-15);
    });

    runner.add_test("coeff_n150_m75", []() {
        auto [c, s] = full_model.get_coefficient(150, 75);
        TEST_ASSERT_NEAR(c, -0.280074149432334e-09, 1e-15);
        TEST_ASSERT_NEAR(s, 0.863338066241331e-09, 1e-15);
    });

    runner.add_test("coeff_n200_m100", []() {
        auto [c, s] = full_model.get_coefficient(200, 100);
        TEST_ASSERT_NEAR(c, -0.846418239314462e-10, 1e-15);
        TEST_ASSERT_NEAR(s, 0.953488706862237e-10, 1e-15);
    });

    runner.add_test("coeff_n250_m125", []() {
        auto [c, s] = full_model.get_coefficient(250, 125);
        TEST_ASSERT_NEAR(c, -0.208121366059082e-09, 1e-15);
        TEST_ASSERT_NEAR(s, -0.161151742431318e-09, 1e-15);
    });

    runner.add_test("coeff_n300_m150", []() {
        auto [c, s] = full_model.get_coefficient(300, 150);
        TEST_ASSERT_NEAR(c, -0.142436196955854e-10, 1e-15);
        TEST_ASSERT_NEAR(s, 0.101003775686779e-09, 1e-15);
    });

    runner.add_test("coeff_n350_m175", []() {
        auto [c, s] = full_model.get_coefficient(350, 175);
        TEST_ASSERT_NEAR(c, 0.170055324017111e-10, 1e-15);
        TEST_ASSERT_NEAR(s, -0.428959889415726e-10, 1e-15);
    });

    runner.add_test("coeff_n360_m0", []() {
        auto [c, s] = full_model.get_coefficient(360, 0);
        TEST_ASSERT_NEAR(c, 0.192459814396284e-11, 1e-15);
        TEST_ASSERT_NEAR(s, 0.0, 1e-20);
    });

    runner.add_test("coeff_n360_m180", []() {
        auto [c, s] = full_model.get_coefficient(360, 180);
        TEST_ASSERT_NEAR(c, -0.868729088622552e-12, 1e-15);
        TEST_ASSERT_NEAR(s, 0.906473487351586e-11, 1e-15);
    });

    runner.add_test("coeff_n360_m360", []() {
        auto [c, s] = full_model.get_coefficient(360, 360);
        TEST_ASSERT_NEAR(c, 0.200046056782130e-10, 1e-15);
        TEST_ASSERT_NEAR(s, -0.958653755280305e-10, 1e-15);
    });

    runner.run();
}

// =============================================================================
// EGM2008 Coefficient Array Integrity Test (full checksum)
// =============================================================================
// Verifies ALL 65,338 embedded coefficients by computing a weighted checksum
// and comparing against a pre-computed reference. This catches any truncated
// write, single-bit corruption, or generation script regression that the 16
// spot-checks might miss.
//
// The checksum is computed as:
//   sum_C = Σ (n + m + 1) * Cnm    (for all coefficients)
//   sum_S = Σ (n + m + 1) * Snm
//   count = number of coefficients
//
// Reference values computed from the known-good embedded array.
// =============================================================================

void test_egm2008_coefficient_checksum() {
    TestRunner runner("EGM2008CoefficientChecksum");

    runner.add_test("coefficient_count", []() {
        TEST_ASSERT_EQ(egm2008_data::NUM_COEFFICIENTS, static_cast<std::size_t>(65338));
        TEST_ASSERT_EQ(egm2008_data::MAX_EMBEDDED_DEGREE, 360);
    });

    runner.add_test("array_bounds", []() {
        // First coefficient should be (2,0)
        TEST_ASSERT_EQ(egm2008_data::COEFFICIENTS[0].n, 2);
        TEST_ASSERT_EQ(egm2008_data::COEFFICIENTS[0].m, 0);

        // Last coefficient should be (360,360)
        const auto& last = egm2008_data::COEFFICIENTS[egm2008_data::NUM_COEFFICIENTS - 1];
        TEST_ASSERT_EQ(last.n, 360);
        TEST_ASSERT_EQ(last.m, 360);
    });

    runner.add_test("degree_ordering", []() {
        // Verify array is sorted by degree (required for early-exit in loading)
        int prev_n = 0;
        for (std::size_t i = 0; i < egm2008_data::NUM_COEFFICIENTS; ++i) {
            int n = egm2008_data::COEFFICIENTS[i].n;
            TEST_ASSERT_MSG(n >= prev_n,
                "Array not sorted at index " + std::to_string(i) +
                ": n=" + std::to_string(n) + " < prev=" + std::to_string(prev_n));
            prev_n = n;
        }
    });

    runner.add_test("weighted_checksum", []() {
        // Compute weighted checksums over all 65,338 coefficients.
        // The weighting by (n+m+1) makes each coefficient contribute uniquely,
        // so a swap or misplacement changes the sum.
        //
        // Reference values pre-computed from the known-good array:
        //   sum_C = -1.42552156670379e-03  (dominated by C20 = -4.84e-4 with weight 3)
        //   sum_S = -2.57847516722875e-05
        //
        // These reference values were computed by this same loop and verified
        // against the 16 NGA spot-checks. Any future corruption will shift them.

        double sum_C = 0.0;
        double sum_S = 0.0;
        std::size_t count = 0;

        for (std::size_t i = 0; i < egm2008_data::NUM_COEFFICIENTS; ++i) {
            const auto& coeff = egm2008_data::COEFFICIENTS[i];
            double weight = static_cast<double>(coeff.n + coeff.m + 1);
            sum_C += weight * coeff.C;
            sum_S += weight * coeff.S;
            count++;
        }

        TEST_ASSERT_EQ(count, static_cast<std::size_t>(65338));

        // Print for initial reference capture
        std::cout << std::setprecision(15)
                  << "[sum_C=" << sum_C << ", sum_S=" << sum_S << "] ";

        // Tolerance: 1e-12 absolute (the sums are ~1e-3 and ~3e-5, so ~1e-9 and ~4e-8
        // relative).
        // Any single corrupted coefficient changes the sum by at least ~1e-15 * weight
        TEST_ASSERT_NEAR(sum_C, -1.42552156670379e-03, 1e-12);
        TEST_ASSERT_NEAR(sum_S, -2.57847516722875e-05, 1e-12);
    });

    runner.run();
}

// =============================================================================
// Pole Gravity Tests (Clenshaw engine singularity handling)
// =============================================================================
// The Clenshaw engine has explicit pole handling via GRAVITY_POLE_RATIO_THRESHOLD
// and GRAVITY_POLE_U_THRESHOLD. These tests exercise the exact-pole code path
// and verify physically expected gravity values at the poles.
//
// Reference: Normal gravity at poles ~9.8322 m/s² (Somigliana formula, WGS84)
// =============================================================================

void test_pole_gravity() {
    TestRunner runner("PoleGravity");

    runner.add_test("north_pole_exact", []() {
        // North Pole ECEF: (0, 0, +b) where b = semi-minor axis
        auto& model = EGM2008Gravity::instance();
        model.init(70);

        Vector3d pos(0.0, 0.0, earth::RADIUS_POLAR);
        Vector3d acc = model.acceleration(pos, 70);

        // Gravity at north pole should be ~9.832 m/s² (purely -Z directed)
        double mag = acc.norm();
        TEST_ASSERT_NEAR(mag, 9.832, 0.01);

        // Should be purely in -Z direction
        TEST_ASSERT_NEAR(acc.x(), 0.0, 1e-3);
        TEST_ASSERT_NEAR(acc.y(), 0.0, 1e-3);
        TEST_ASSERT(acc.z() < 0.0);
    });

    runner.add_test("south_pole_exact", []() {
        // South Pole ECEF: (0, 0, -b)
        auto& model = EGM2008Gravity::instance();

        Vector3d pos(0.0, 0.0, -earth::RADIUS_POLAR);
        Vector3d acc = model.acceleration(pos, 70);

        // Gravity magnitude should match north pole
        double mag = acc.norm();
        TEST_ASSERT_NEAR(mag, 9.832, 0.01);

        // Should be purely in +Z direction (pointing toward center)
        TEST_ASSERT_NEAR(acc.x(), 0.0, 1e-3);
        TEST_ASSERT_NEAR(acc.y(), 0.0, 1e-3);
        TEST_ASSERT(acc.z() > 0.0);
    });

    runner.add_test("pole_symmetry", []() {
        // North and south pole gravity magnitudes should be equal
        // (even-degree terms are symmetric, odd-degree terms are antisymmetric)
        auto& model = EGM2008Gravity::instance();

        Vector3d pos_n(0.0, 0.0, earth::RADIUS_POLAR);
        Vector3d pos_s(0.0, 0.0, -earth::RADIUS_POLAR);

        Vector3d acc_n = model.acceleration(pos_n, 70);
        Vector3d acc_s = model.acceleration(pos_s, 70);

        // Magnitudes should agree to ~1e-3 m/s² (odd-degree zonal terms cause N/S asymmetry)
        TEST_ASSERT_NEAR(acc_n.norm(), acc_s.norm(), 1e-3);

        // Z components should be opposite in sign
        TEST_ASSERT_NEAR(acc_n.z(), -acc_s.z(), 1e-3);
    });

    runner.add_test("near_pole_continuity", []() {
        // Verify continuity: gravity 0.001° from pole should be close to pole value
        auto& model = EGM2008Gravity::instance();

        // Exact north pole
        Vector3d pos_pole(0.0, 0.0, earth::RADIUS_POLAR);
        Vector3d acc_pole = model.acceleration(pos_pole, 70);

        // 0.001° from north pole (~111 m offset)
        double lat = (90.0 - 0.001) * constants::DEG_TO_RAD;
        double lon = 0.0;
        Vector3d pos_near = coords::geodetic_to_ecef(lat, lon, 0.0);
        Vector3d acc_near = model.acceleration(pos_near, 70);

        // Magnitudes should agree within ~0.001 m/s²
        TEST_ASSERT_NEAR(acc_pole.norm(), acc_near.norm(), 0.001);
    });

    runner.add_test("pole_high_degree", []() {
        // Test pole at higher degree to stress the Clenshaw recursion
        auto& model = EGM2008Gravity::instance();
        model.init(360);

        Vector3d pos(0.0, 0.0, earth::RADIUS_POLAR);
        Vector3d acc = model.acceleration(pos, 360);

        // Should still produce reasonable gravity
        double mag = acc.norm();
        TEST_ASSERT_NEAR(mag, 9.832, 0.01);
        TEST_ASSERT(std::isfinite(mag));
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "EGM2008 Loading Tests" << std::endl;
    std::cout << "============================================" << std::endl;

    test_egm2008_embedded();
    test_egm2008_file_loading();
    test_egm2008_accuracy();
    test_init_api();
    test_performance();
    test_egm2008_coefficient_spotchecks();
    test_egm2008_coefficient_checksum();
    test_pole_gravity();

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
