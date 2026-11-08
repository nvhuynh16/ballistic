// =============================================================================
// Sentinel-1 Orbit Propagation Validation Tests
// =============================================================================
//
// Propagates Sentinel-1 start states and compares the result with ESA's Precise
// Orbit Ephemerides (POE), the reference orbit ESA distributes as EOF ("Earth
// Observation File") XML files. The POE is accurate to centimeters, so it serves as truth.
//
// Data: tests/data/eof/sentinel1_2016/2014-10 (Sentinel-1A POE files, October 2014,
// guard-exempt(PUB-DATE): data).
//
// Sentinel-1 orbit characteristics:
// - Altitude: ~693 km (sun-synchronous)
// - Inclination: 98.18 degrees
// - Period: ~98.6 minutes
// - Mass, drag area and Cd: the S1A entry of the satellite registry
//   (include/ballistic/satellites.hpp)
//
// Two configurations are exercised:
// - "Optimal": the Trajectory class in the GCRS frame (the inertial frame of the
//   IAU 2006/2000A transforms) with the JPL DE421 ephemeris and ITRF2014; target
//   <1 m RMSE (root-mean-square position error) over 1 hour. With DE421 this
//   configuration reaches 0.24 m.
// - "Standard": the older RK45 propagator (no DragEnvironment is attached, so
//   drag is off; the weather it is given is unused); its gates are loose (they
//   catch breakage, not accuracy regressions).
// =============================================================================

#include <ballistic/ballistic.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/validation/eof_parser.hpp>
#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN
#include <ballistic/atmosphere/internal/nrlmsis21_full.hpp>
#endif
#include <ballistic/forces/drag.hpp>
#include <ballistic/ephemeris/spk_ephemeris.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numbers>
#include <vector>
#include <string>
#include <filesystem>

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;

#include "../test_utils.hpp"
using namespace ballistic::test;

namespace fs = std::filesystem;

// =============================================================================
// Test Configuration
// =============================================================================

// Sentinel-1 spacecraft parameters
struct Sentinel1Config {
    static constexpr double MASS = ballistic::satellites::get("S1A").mass_kg;          // registry (satellites.hpp)
    static constexpr double DRAG_AREA = ballistic::satellites::get("S1A").drag_area_m2;  // frontal
    static constexpr double CD = ballistic::satellites::get("S1A").cd;
    static constexpr double ALTITUDE = 693.0;        // km nominal
    static constexpr double INCLINATION = 98.18;     // degrees

    // Round values for the test data period (early October 2014): F10.7 is the
    // 10.7 cm solar radio flux, Ap the daily geomagnetic index. The committed
    // CelesTrak file gives F10.7 149-155, 81-day mean ~146, Ap 7-11 on Oct 1-2.
    static constexpr double F107_DAILY = 145.0;      // Moderate solar conditions
    static constexpr double F107_AVERAGE = 140.0;    // 81-day average
    static constexpr double AP_DAILY = 10.0;         // Quiet geomagnetic
};

// JPL ephemeris paths for optimal configuration
namespace ephemeris_paths {
    constexpr const char* DE421 = BALLISTIC_PROJECT_ROOT "/data/de421.bsp";
    constexpr const char* DE440 = BALLISTIC_PROJECT_ROOT "/data/de440.bsp";
}

// =============================================================================
// Utility Functions
// =============================================================================

// Directory of the October 2014 POE files; tries paths relative to common
// build locations.
std::string get_test_data_path() {
    std::vector<std::string> candidates = {
        BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel1_2016/2014-10",
        "../tests/data/eof/sentinel1_2016/2014-10",
        "../../tests/data/eof/sentinel1_2016/2014-10",
        "tests/data/eof/sentinel1_2016/2014-10"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }

    return BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel1_2016/2014-10";  // Default
}

// Find and parse an EOF file (handles both .EOF and .EOF.zip files)
std::optional<EOFFile> load_eof_file(const std::string& pattern = "", bool debug = false) {
    std::string data_path = get_test_data_path();

    if (debug) std::cout << "  [DEBUG] Looking for EOF files in: " << data_path << std::endl;

    // find_and_parse_eof also extracts zipped files
    auto result = find_and_parse_eof(data_path, pattern);

    if (result && debug) {
        std::cout << "  [DEBUG] Parsed OK, " << result->state_vectors.size() << " vectors" << std::endl;
    }

    return result;
}

// Propagate with the standard RK45 propagator and sample the state every
// `sample_interval` seconds. No DragEnvironment is attached, so drag is off
// (the propagator does not use the weather or the BallisticModel).
std::vector<std::pair<Epoch, StateVector>> propagate_with_sampling(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    double sample_interval = 10.0)
{
    std::vector<std::pair<Epoch, StateVector>> results;

    BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, Sentinel1Config::CD);

    // Weather conditions for October 2014
    WeatherConditions weather;
    weather.f107_daily = Sentinel1Config::F107_DAILY;
    weather.f107_average = Sentinel1Config::F107_AVERAGE;
    weather.ap_daily = Sentinel1Config::AP_DAILY;
    weather.ap_history.fill(Sentinel1Config::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos0, vel0, model, weather, true);

    int num_samples = static_cast<int>(duration_seconds / sample_interval) + 1;
    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * sample_interval;
        if (dt > duration_seconds) dt = duration_seconds;

        Epoch t = t0 + dt;
        auto result = (*prop)(t);

        StateVector state(result.position, result.velocity);
        results.emplace_back(t, state);
    }

    return results;
}

// =============================================================================
// EOF Parser Tests
// =============================================================================

// The EOF parser reads the file: mission name, Earth-fixed frame, and a first
// state at Sentinel-1's radius and speed (wide physical bounds).
void test_eof_parser_basic() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    TEST_ASSERT(!eof->state_vectors.empty());
    TEST_ASSERT(eof->mission.find("Sentinel") != std::string::npos);
    TEST_ASSERT(eof->reference_frame == "EARTH_FIXED");

    const auto& sv0 = eof->state_vectors[0];
    double r = sv0.position.norm();
    double v = sv0.velocity.norm();

    // Sentinel-1 at ~693 km altitude
    TEST_ASSERT(r > 7.0e6 && r < 7.2e6);  // ~7071 km radius
    TEST_ASSERT(v > 7400 && v < 7600);     // ~7.5 km/s
}

// The file holds many states (a POE has ~9360: 10 s spacing over ~26 h) at 10 s spacing.
void test_eof_state_vector_count() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    TEST_ASSERT(eof->state_vectors.size() > 1000);

    if (eof->state_vectors.size() >= 2) {
        double dt = eof->state_vectors[1].epoch - eof->state_vectors[0].epoch;
        TEST_ASSERT(std::abs(dt - 10.0) < 0.1);
    }
}

// =============================================================================
// Short-Term Propagation Tests
// =============================================================================

// 1-minute propagation, standard propagator. Loose 10 m bound: catches breakage only.
void test_1_minute_propagation() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    // Start 10 records in, safely inside the file's interpolation range
    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    auto results = propagate_with_sampling(t0, sv0.position, sv0.velocity, 60.0, 10.0);
    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "(RMSE=" << std::fixed << std::setprecision(2)
              << test_result.position_rmse << "m) ";

    TEST_ASSERT(test_result.position_rmse < 10.0);
}

// 10-minute propagation, standard propagator. Loose 50 m bound: catches breakage only.
void test_10_minute_propagation() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    auto results = propagate_with_sampling(t0, sv0.position, sv0.velocity, 600.0, 10.0);
    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "(RMSE=" << std::fixed << std::setprecision(2)
              << test_result.position_rmse << "m) ";

    TEST_ASSERT(test_result.position_rmse < 50.0);
}

// =============================================================================
// One-Hour Propagation Tests (Primary Target)
// =============================================================================

// 1-hour propagation, standard propagator. Prints the error split into radial,
// along-track and cross-track parts and grades it against 10 m / 100 m; the
// assertion is only a 1000 m breakage bound (the sub-meter claim is gated by the
// optimal-configuration tests).
void test_1_hour_propagation() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    auto results = propagate_with_sampling(t0, sv0.position, sv0.velocity, 3600.0, 10.0);
    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << test_result.position_rmse << " m\n";
    std::cout << "    Position Max:  " << test_result.position_max_error << " m\n";
    std::cout << "    Radial RMS:    " << test_result.radial_rms << " m\n";
    std::cout << "    Along-trk RMS: " << test_result.along_track_rms << " m\n";
    std::cout << "    Cross-trk RMS: " << test_result.cross_track_rms << " m\n";
    std::cout << "    Velocity RMSE: " << test_result.velocity_rmse * 1000 << " mm/s\n";
    std::cout << "    ";

    if (test_result.position_rmse < 10.0) {
        std::cout << "TARGET MET (<10m)... ";
    } else if (test_result.position_rmse < 100.0) {
        std::cout << "ACCEPTABLE (<100m)... ";
    } else {
        std::cout << "NEEDS IMPROVEMENT... ";
    }

    TEST_ASSERT(test_result.position_rmse < 1000.0);
}

// 1-hour propagations from 8 starts 15 minutes apart (first 2 hours of the file);
// the mean RMSE must stay under a 1000 m breakage bound.
void test_1_hour_multiple_starts() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    std::vector<double> rmse_values;
    int skip = 90;  // 90 * 10s = 15 minutes between starts

    std::cout << "\n";
    for (int start_idx = 10; start_idx < 730 && start_idx < static_cast<int>(eof->state_vectors.size()) - 360; start_idx += skip) {
        const auto& sv0 = eof->state_vectors[start_idx];
        Epoch t0 = sv0.epoch;

        auto results = propagate_with_sampling(t0, sv0.position, sv0.velocity, 3600.0, 60.0);
        auto test_result = compute_propagation_result(results, *eof);

        rmse_values.push_back(test_result.position_rmse);
        std::cout << "    Start " << (start_idx - 10) / skip + 1 << ": RMSE = "
                  << std::fixed << std::setprecision(2) << test_result.position_rmse << " m\n";
    }

    double sum = 0.0, max_rmse = 0.0;
    for (double r : rmse_values) {
        sum += r;
        max_rmse = (std::max)(max_rmse, r);
    }
    double mean_rmse = sum / rmse_values.size();

    std::cout << "    Mean RMSE: " << std::fixed << std::setprecision(2) << mean_rmse << " m\n";
    std::cout << "    Max RMSE:  " << max_rmse << " m\n";
    std::cout << "    ";

    TEST_ASSERT(mean_rmse < 1000.0);
}

// =============================================================================
// Extended Propagation Tests
// =============================================================================

// One full orbit (~99 minutes), standard propagator; measured 0.55 m RMSE (max 1.12 m)
// with every sample on a 10 s orbit record; bound 2x the measured value.
void test_full_orbit_propagation() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    // Sentinel-1 orbital period ~98.6 minutes (~5916 s). The span is rounded
    // to 5910 s so every sample lands on a 10 s POE record: the truth is not
    // interpolated, and a sample 4 s off its record scores ~30 km of motion.
    double orbital_period = 5910.0;

    auto results = propagate_with_sampling(t0, sv0.position, sv0.velocity, orbital_period, 10.0);
    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "(RMSE=" << std::fixed << std::setprecision(2)
              << test_result.position_rmse << "m, max="
              << test_result.position_max_error << "m) ";

    TEST_ASSERT(test_result.position_rmse < 1.2);
}

// =============================================================================
// Error Analysis Tests
// =============================================================================

// Diagnostic (no assertion): prints RMSE and its growth rate every 5 minutes up to 1 hour.
void test_error_growth_rate() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    std::cout << "\n";
    std::cout << "    Time(min)  RMSE(m)  Rate(m/min)\n";
    std::cout << "    --------------------------------\n";

    double prev_rmse = 0.0;
    for (int minutes = 5; minutes <= 60; minutes += 5) {
        double duration = minutes * 60.0;
        auto results = propagate_with_sampling(t0, sv0.position, sv0.velocity, duration, 10.0);
        auto test_result = compute_propagation_result(results, *eof);

        double rate = (test_result.position_rmse - prev_rmse) / 5.0;
        std::cout << "    " << std::setw(4) << minutes << "       "
                  << std::fixed << std::setprecision(1) << std::setw(7) << test_result.position_rmse
                  << "  " << std::setw(6) << rate << "\n";

        prev_rmse = test_result.position_rmse;
    }
    std::cout << "    ";
}

// =============================================================================
// Gravity Degree Sweep Tests
// =============================================================================

// Same as propagate_with_sampling, with the gravity field truncated at `gravity_degree`.
std::vector<std::pair<Epoch, StateVector>> propagate_with_gravity_degree(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    int gravity_degree,
    double sample_interval = 10.0)
{
    std::vector<std::pair<Epoch, StateVector>> results;

    BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, Sentinel1Config::CD);

    // Weather conditions for October 2014
    WeatherConditions weather;
    weather.f107_daily = Sentinel1Config::F107_DAILY;
    weather.f107_average = Sentinel1Config::F107_AVERAGE;
    weather.ap_daily = Sentinel1Config::AP_DAILY;
    weather.ap_history.fill(Sentinel1Config::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos0, vel0, model, weather, true);
    prop->force_model().set_gravity_max_degree(gravity_degree);

    int num_samples = static_cast<int>(duration_seconds / sample_interval) + 1;
    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * sample_interval;
        if (dt > duration_seconds) dt = duration_seconds;

        Epoch t = t0 + dt;
        auto result = (*prop)(t);

        StateVector state(result.position, result.velocity);
        results.emplace_back(t, state);
    }

    return results;
}

// Diagnostic sweep of the gravity degree (10 to 200) for a 1-hour propagation. Needs the
// full NGA EGM2008 file under data/ for degrees above 10; asserts only a 1000 m
// breakage bound on the best degree.
void test_gravity_degree_sweep() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file");
    }

    // Optional full EGM2008 coefficient file (degree 2190), loaded to degree 200
    fs::path egm_path = BALLISTIC_PROJECT_ROOT "/data/EGM2008_Spherical_Harmonics/EGM2008_to2190_TideFree";
    bool egm_loaded = false;
    if (fs::exists(egm_path)) {
        egm_loaded = gravity::EGM2008Gravity::instance().load_full(egm_path, 200);
        if (egm_loaded) {
            std::cout << "\n    Loaded EGM2008 to degree "
                      << gravity::EGM2008Gravity::instance().loaded_degree() << "\n";
        }
    }

    if (!egm_loaded) {
        std::cout << "\n    Using embedded EGM2008 fallback (degree ~10)\n";
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;
    double duration = 3600.0;  // 1 hour

    std::cout << "\n";
    std::cout << "    Gravity Degree vs 1-Hour RMSE:\n";
    std::cout << "    Degree    RMSE(m)   Cross-Track  Along-Track  Radial\n";
    std::cout << "    ----------------------------------------------------------\n";

    std::vector<int> degrees = {10, 20, 40, 70, 100, 120, 150, 200};
    double best_rmse = 1e9;
    int best_degree = 10;

    for (int degree : degrees) {
        if (egm_loaded && degree > gravity::EGM2008Gravity::instance().loaded_degree()) {
            continue;
        }
        if (!egm_loaded && degree > 10) {
            continue;  // without the full file this sweep runs degree 10 only
                       // (the embedded model itself reaches degree 360)
        }

        auto results = propagate_with_gravity_degree(t0, sv0.position, sv0.velocity, duration, degree, 10.0);
        auto test_result = compute_propagation_result(results, *eof);

        std::cout << "    " << std::setw(4) << degree << "       "
                  << std::fixed << std::setprecision(2) << std::setw(7) << test_result.position_rmse
                  << "     " << std::setw(7) << test_result.cross_track_rms
                  << "      " << std::setw(7) << test_result.along_track_rms
                  << "     " << std::setw(6) << test_result.radial_rms << "\n";

        if (test_result.position_rmse < best_rmse) {
            best_rmse = test_result.position_rmse;
            best_degree = degree;
        }
    }

    std::cout << "    ----------------------------------------------------------\n";
    std::cout << "    Best: degree " << best_degree << " with RMSE = "
              << std::fixed << std::setprecision(2) << best_rmse << " m\n";

    TEST_ASSERT(best_rmse < 1000.0);
}

// =============================================================================
// Optimal Configuration Tests (Sub-Meter Accuracy)
// =============================================================================

// 1-hour propagation with the Trajectory class, gated at <1 m RMSE when a JPL
// ephemeris file (data/de421.bsp or data/de440.bsp) is present, else <2 m. Setup:
// - GCRS frame (IAU 2006/2000A)
// - JPL DE421 ephemeris
// - ITRF2014 realization (ESA's frame for 2014 data)
// - Third-body perturbations (Sun/Moon/Jupiter)
// - Relativistic corrections (Schwarzschild + Lense-Thirring)
// - No drag and no solar radiation pressure (SRP): this test pins the
//   force-model-only baseline. A POE start state does NOT already include those
//   forces; both improve the RMSE [`SENTINEL-YEAR-2025`].
void test_optimal_trajectory_class() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    bool ephemeris_loaded = false;
    if (fs::exists(ephemeris_paths::DE421)) {
        ephemeris_loaded = ephemeris::load_spk_ephemeris(ephemeris_paths::DE421);
        if (ephemeris_loaded) {
            std::cout << "\n    JPL DE421 ephemeris loaded\n";
        }
    }
    if (!ephemeris_loaded && fs::exists(ephemeris_paths::DE440)) {
        ephemeris_loaded = ephemeris::load_spk_ephemeris(ephemeris_paths::DE440);
        if (ephemeris_loaded) {
            std::cout << "\n    JPL DE440 ephemeris loaded\n";
        }
    }
    if (!ephemeris_loaded) {
        std::cout << "\n    No JPL ephemeris file in data/ - using the embedded DE440 "
                     "(0.24 m RMSE was measured with data/de421.bsp)\n";
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    TrajectoryConfig config;
    config.gravity_degree = 70;           // EGM2008 degree 70
    config.enable_third_body = true;      // Sun/Moon/Jupiter
    config.enable_relativistic = true;    // Schwarzschild + Lense-Thirring
    config.itrf = coords::ITRFRealization::ITRF2014;  // Match ESA's ITRF for 2014 data

    Trajectory traj(t0, sv0.position, sv0.velocity, config);

    // Sample every 10 s
    std::vector<std::pair<Epoch, StateVector>> results;
    double duration = 3600.0;  // 1 hour
    double sample_interval = 10.0;
    int num_samples = static_cast<int>(duration / sample_interval) + 1;

    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * sample_interval;
        if (dt > duration) dt = duration;
        Epoch t = t0 + dt;

        auto result = traj.propagate(t);
        results.emplace_back(t, StateVector(result.position, result.velocity));
    }

    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "\n";
    std::cout << "    =====================================================\n";
    std::cout << "    OPTIMAL CONFIGURATION (Trajectory Class)\n";
    std::cout << "    =====================================================\n";
    std::cout << "    Config: GCRS + JPL + ITRF2014 + third-body + relativistic\n";
    std::cout << "            No drag, No SRP (force-model-only baseline)\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    Position RMSE:   " << std::fixed << std::setprecision(4)
              << test_result.position_rmse << " m\n";
    std::cout << "    Position Max:    " << test_result.position_max_error << " m\n";
    std::cout << "    Radial RMS:      " << test_result.radial_rms << " m\n";
    std::cout << "    Along-track RMS: " << test_result.along_track_rms << " m\n";
    std::cout << "    Cross-track RMS: " << test_result.cross_track_rms << " m\n";
    std::cout << "    Velocity RMSE:   " << test_result.velocity_rmse * 1000 << " mm/s\n";
    std::cout << "    -----------------------------------------------------\n";

    if (test_result.position_rmse < 1.0) {
        std::cout << "    *** SUB-METER ACCURACY ACHIEVED! ***\n";
    } else if (test_result.position_rmse < 2.0) {
        std::cout << "    Note: above 1 m (0.24 m is measured with data/de421.bsp)\n";
    }
    std::cout << "    =====================================================\n";
    std::cout << "    ";

    // <1 m with a JPL ephemeris file, <2 m without one
    double threshold = ephemeris_loaded ? 1.0 : 2.0;
    TEST_ASSERT_MSG(test_result.position_rmse < threshold,
        "RMSE " + std::to_string(test_result.position_rmse) + "m exceeds threshold " +
        std::to_string(threshold) + "m");
}

// The same no-drag setup through the low-level create_gcrs_propagator; must stay
// under 2 m like the Trajectory-class result.
void test_optimal_gcrs_propagator() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    std::shared_ptr<ephemeris::EphemerisProvider> jpl_ephemeris;
    if (fs::exists(ephemeris_paths::DE421)) {
        if (ephemeris::load_spk_ephemeris(ephemeris_paths::DE421)) {
            jpl_ephemeris = std::shared_ptr<ephemeris::EphemerisProvider>(
                std::shared_ptr<void>{}, &ephemeris::SPKEphemerisManager::instance().ephemeris());
        }
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, 0.0);  // No drag
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = propagators::create_gcrs_propagator(
        t0, sv0.position, sv0.velocity, model, weather, true,
        jpl_ephemeris, coords::ITRFRealization::ITRF2014);

    prop->force_model().set_gravity_max_degree(70);
    {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.relativistic = true;
        flags.drag = false;
        prop->force_model().set_force_flags(flags);
    }

    std::vector<std::pair<Epoch, StateVector>> results;
    double duration = 3600.0;
    int num_samples = static_cast<int>(duration / 10.0) + 1;

    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * 10.0;
        if (dt > duration) dt = duration;
        Epoch t = t0 + dt;
        auto result = (*prop)(t);
        results.emplace_back(t, StateVector(result.position, result.velocity));
    }

    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "\n";
    std::cout << "    create_gcrs_propagator (low-level API):\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(4)
              << test_result.position_rmse << " m\n";
    std::cout << "    ";

    TEST_ASSERT(test_result.position_rmse < 2.0);
}

// =============================================================================
// IAU Transform Tests
// =============================================================================

// Diagnostic (no assertion): converts the start state from ITRF (the Earth-fixed
// frame, also called ECEF here) to GCRS and back with the IAU 2006 transforms,
// prints the round-trip error, then propagates 1 hour and prints the error against
// the nearest POE record.
void test_iau_end_to_end() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;
    double duration = 3600.0;  // 1 hour

    std::cout << "\n    Testing IAU transforms end-to-end...\n";

    Vector3d pos_ecef = sv0.position;
    Vector3d vel_ecef = sv0.velocity;

    // IAU 2006 transform (not the older sidereal-time-only rotation)
    StateVector state_ecef(pos_ecef, vel_ecef);
    StateVector state_gcrs = coords::itrf_to_gcrs_full(state_ecef, t0);

    std::cout << "    Initial ECEF pos: [" << pos_ecef.transpose() << "] m\n";
    std::cout << "    Initial GCRS pos: [" << state_gcrs.position.transpose() << "] m\n";

    StateVector roundtrip_ecef = coords::gcrs_to_itrf_full(state_gcrs, t0);
    Vector3d roundtrip_error = roundtrip_ecef.position - pos_ecef;
    std::cout << "    IAU round-trip error:  " << roundtrip_error.norm() << " m\n";

    BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, Sentinel1Config::CD);
    WeatherConditions weather;
    weather.f107_daily = Sentinel1Config::F107_DAILY;
    weather.f107_average = Sentinel1Config::F107_AVERAGE;
    weather.ap_daily = Sentinel1Config::AP_DAILY;
    weather.ap_history.fill(Sentinel1Config::AP_DAILY);

    // The propagator takes the Earth-fixed start state and integrates in GCRS
    auto prop = create_rk45_propagator(t0, pos_ecef, vel_ecef, model, weather, true);

    Epoch t_end = t0 + duration;
    prop->propagate_to(t_end);

    // Result in GCRS (the propagation frame), converted with IAU 2006
    StateVector result_gcrs = prop->current_state_gcrs();
    StateVector result_ecef_iau = coords::gcrs_to_itrf_full(result_gcrs, t_end);

    // Nearest POE record to the end time
    double min_diff = 1e9;
    StateVector truth_ecef;
    for (const auto& sv : eof->state_vectors) {
        double dt = std::abs(sv.epoch - t_end);
        if (dt < min_diff) {
            min_diff = dt;
            truth_ecef = StateVector(sv.position, sv.velocity);
        }
    }

    Vector3d error_iau = result_ecef_iau.position - truth_ecef.position;

    std::cout << "\n    After 1-hour propagation:\n";
    std::cout << "    Error with IAU output:  " << error_iau.norm() << " m\n";
}

// Diagnostic (no assertion): prints the start epoch in UTC/TT/UT1/TAI, checks
// UT1 = UTC + (UT1-UTC from the Earth orientation parameters, EOP), and compares
// Greenwich apparent sidereal time (GAST) with the Earth rotation angle (ERA).
void test_time_scale_diagnostic() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    std::cout << "\n    Time Scale Diagnostic for epoch:\n";
    std::cout << "    UTC JD:  " << std::fixed << std::setprecision(8) << t0.utc_jd() << "\n";
    std::cout << "    TT JD:   " << t0.tt_jd() << "\n";
    std::cout << "    UT1 JD:  " << t0.ut1_jd() << "\n";
    std::cout << "    TAI JD:  " << t0.tai_jd() << "\n";

    double mjd = t0.utc_jd() - 2400000.5;
    std::cout << "    MJD:     " << mjd << "\n";

    if (time::EOPData::instance().is_loaded()) {
        double dut1 = time::EOPData::instance().ut1_utc(mjd);
        std::cout << "    DUT1 (UT1-UTC): " << std::setprecision(6) << dut1 << " seconds\n";

        double ut1_calc = t0.utc_jd() + dut1 / 86400.0;
        double ut1_diff = (t0.ut1_jd() - ut1_calc) * 86400.0;
        std::cout << "    UT1 verification error: " << ut1_diff << " seconds\n";
    }

    double gast_angle = coords::gast(t0.ut1_jd(), t0.tt_jd());
    double gast_full_angle = coords::gast_full(t0.ut1_jd(), t0.tt_jd(), mjd);

    // ERA (IERS 2010 eq. 5.15)
    double era = 2.0 * constants::PI * (0.7790572732640 + 1.00273781191135448 * (t0.ut1_jd() - constants::J2000_JD));
    era = std::fmod(era, 2.0 * constants::PI);

    std::cout << "\n    Rotation angles:\n";
    std::cout << "    GAST (simplified): " << std::setprecision(10) << gast_angle << " rad\n";
    std::cout << "    GAST (full):       " << gast_full_angle << " rad\n";
    std::cout << "    ERA:               " << era << " rad\n";
    std::cout << "    GAST - ERA:        " << (gast_angle - era) * 3600.0 * 180.0 / constants::PI << " arcsec\n";
}

// Diagnostic (no assertion): 1-hour RMSE and relative energy drift for the adaptive
// RK45 (Dormand-Prince) integrator and fixed-step symplectic integrators (leapfrog,
// Yoshida 6th/8th order) at several step sizes, then a one-orbit comparison.
void test_rk45_vs_symplectic() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    std::cout << "\n    RK45 vs Symplectic Integrator Comparison:\n";
    std::cout << "    Integrator      Step(s)  1hr RMSE   Cross-Trk  Along-Trk  Radial   Energy Drift\n";
    std::cout << "    ---------------------------------------------------------------------------------\n";

    BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, Sentinel1Config::CD);
    WeatherConditions weather;
    weather.f107_daily = Sentinel1Config::F107_DAILY;
    weather.f107_average = Sentinel1Config::F107_AVERAGE;
    weather.ap_daily = Sentinel1Config::AP_DAILY;
    weather.ap_history.fill(Sentinel1Config::AP_DAILY);

    double duration = 3600.0;  // 1 hour
    double sample_interval = 10.0;

    // RK45 (adaptive step)
    {
        auto prop = create_rk45_propagator(t0, sv0.position, sv0.velocity, model, weather, true);

        // Point-mass orbital energy from the Earth-fixed start state (a rough
        // reference; the final energy uses the GCRS state)
        double r0 = sv0.position.norm();
        double v0 = sv0.velocity.norm();
        double E0 = 0.5 * v0 * v0 - constants::earth::GM / r0;

        std::vector<std::pair<Epoch, StateVector>> results;
        int num_samples = static_cast<int>(duration / sample_interval) + 1;
        for (int i = 0; i <= num_samples; ++i) {
            double dt = i * sample_interval;
            if (dt > duration) dt = duration;
            Epoch t = t0 + dt;
            auto result = (*prop)(t);
            StateVector state(result.position, result.velocity);
            results.emplace_back(t, state);
        }

        StateVector final_eci = prop->current_state_gcrs();
        double rf = final_eci.position.norm();
        double vf = final_eci.velocity.norm();
        double Ef = 0.5 * vf * vf - constants::earth::GM / rf;
        double energy_drift = std::abs((Ef - E0) / E0) * 100.0;  // percent

        auto test_result = compute_propagation_result(results, *eof);
        std::cout << "    RK45 (adaptive)  ~10    "
                  << std::fixed << std::setprecision(2) << std::setw(7) << test_result.position_rmse << "   "
                  << std::setw(7) << test_result.cross_track_rms << "    "
                  << std::setw(7) << test_result.along_track_rms << "    "
                  << std::setw(6) << test_result.radial_rms << "   "
                  << std::scientific << std::setprecision(2) << energy_drift << "%\n";
    }

    // Symplectic integrators at several orders and step sizes
    std::vector<std::tuple<SymplecticOrder, double, std::string>> configs = {
        // Leapfrog (2nd order) - needs small steps for accuracy
        {SymplecticOrder::LEAPFROG, 0.1, "Leapfrog-2"},
        {SymplecticOrder::LEAPFROG, 0.5, "Leapfrog-2"},
        {SymplecticOrder::LEAPFROG, 1.0, "Leapfrog-2"},
        {SymplecticOrder::LEAPFROG, 5.0, "Leapfrog-2"},
        // Higher order methods - efficient with larger steps
        {SymplecticOrder::YOSHIDA6, 10.0, "Yoshida-6"},
        {SymplecticOrder::YOSHIDA6, 30.0, "Yoshida-6"},
        {SymplecticOrder::YOSHIDA6, 30.0, "Yoshida-6"},
        {SymplecticOrder::YOSHIDA8, 30.0, "Yoshida-8"},
    };

    for (const auto& [order, step, name] : configs) {
        SymplecticConfig config;
        config.order = order;
        config.step_size = step;
        config.nc_strategy = NonConservativeStrategy::SLIMPLECTIC;

        auto prop = create_symplectic_propagator(t0, sv0.position, sv0.velocity, model, weather, true, config);

        double r0 = sv0.position.norm();
        double v0 = sv0.velocity.norm();
        double E0 = 0.5 * v0 * v0 - constants::earth::GM / r0;

        std::vector<std::pair<Epoch, StateVector>> results;
        int num_samples = static_cast<int>(duration / sample_interval) + 1;
        for (int i = 0; i <= num_samples; ++i) {
            double dt = i * sample_interval;
            if (dt > duration) dt = duration;
            Epoch t = t0 + dt;
            auto result = (*prop)(t);
            StateVector state(result.position, result.velocity);
            results.emplace_back(t, state);
        }

        StateVector final_eci = prop->current_state_gcrs();
        double rf = final_eci.position.norm();
        double vf = final_eci.velocity.norm();
        double Ef = 0.5 * vf * vf - constants::earth::GM / rf;
        double energy_drift = std::abs((Ef - E0) / E0) * 100.0;

        auto test_result = compute_propagation_result(results, *eof);
        std::cout << "    " << std::left << std::setw(16) << name
                  << std::right << std::setw(6) << std::fixed << std::setprecision(1) << step << "  "
                  << std::fixed << std::setprecision(2) << std::setw(7) << test_result.position_rmse << "   "
                  << std::setw(7) << test_result.cross_track_rms << "    "
                  << std::setw(7) << test_result.along_track_rms << "    "
                  << std::setw(6) << test_result.radial_rms << "   "
                  << std::scientific << std::setprecision(2) << energy_drift << "%\n";
    }

    std::cout << "    ---------------------------------------------------------------------------------\n";

    // One full orbit (~98 minutes)
    std::cout << "\n    Extended Propagation (1 orbit = 98 min):\n";
    std::cout << "    Integrator      RMSE(m)   Energy Drift\n";
    std::cout << "    ----------------------------------------\n";

    double orbit_duration = 98.0 * 60.0;  // 98 minutes

    // RK45
    {
        auto prop = create_rk45_propagator(t0, sv0.position, sv0.velocity, model, weather, true);
        double r0 = sv0.position.norm();
        double v0 = sv0.velocity.norm();
        double E0 = 0.5 * v0 * v0 - constants::earth::GM / r0;

        std::vector<std::pair<Epoch, StateVector>> results;
        for (double dt = 0; dt <= orbit_duration; dt += sample_interval) {
            Epoch t = t0 + dt;
            auto result = (*prop)(t);
            results.emplace_back(t, StateVector(result.position, result.velocity));
        }

        StateVector final_eci = prop->current_state_gcrs();
        double rf = final_eci.position.norm();
        double vf = final_eci.velocity.norm();
        double Ef = 0.5 * vf * vf - constants::earth::GM / rf;
        double energy_drift = std::abs((Ef - E0) / E0) * 100.0;

        auto test_result = compute_propagation_result(results, *eof);
        std::cout << "    RK45           " << std::fixed << std::setprecision(1) << std::setw(8)
                  << test_result.position_rmse << "   "
                  << std::scientific << std::setprecision(2) << energy_drift << "%\n";
    }

    // Yoshida 6th order at 10 s (the printed label says "Yoshida-4")
    {
        SymplecticConfig config;
        config.order = SymplecticOrder::YOSHIDA6;
        config.step_size = 10.0;
        config.nc_strategy = NonConservativeStrategy::SLIMPLECTIC;

        auto prop = create_symplectic_propagator(t0, sv0.position, sv0.velocity, model, weather, true, config);
        double r0 = sv0.position.norm();
        double v0 = sv0.velocity.norm();
        double E0 = 0.5 * v0 * v0 - constants::earth::GM / r0;

        std::vector<std::pair<Epoch, StateVector>> results;
        for (double dt = 0; dt <= orbit_duration; dt += sample_interval) {
            Epoch t = t0 + dt;
            auto result = (*prop)(t);
            results.emplace_back(t, StateVector(result.position, result.velocity));
        }

        StateVector final_eci = prop->current_state_gcrs();
        double rf = final_eci.position.norm();
        double vf = final_eci.velocity.norm();
        double Ef = 0.5 * vf * vf - constants::earth::GM / rf;
        double energy_drift = std::abs((Ef - E0) / E0) * 100.0;

        auto test_result = compute_propagation_result(results, *eof);
        std::cout << "    Yoshida-4      " << std::fixed << std::setprecision(1) << std::setw(8)
                  << test_result.position_rmse << "   "
                  << std::scientific << std::setprecision(2) << energy_drift << "%\n";
    }

    std::cout << "    ----------------------------------------\n";
}

#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN   // the NRLMSIS 2.1 cells below need the model
// =============================================================================
// NRLMSIS 2.1 Atmosphere Model Tests
// =============================================================================

// NRLMSIS 2.1 density at Sentinel-1's altitude is positive and below 1e-10 kg/m^3
// (a loose physical sanity bound). Only NRLMSIS 2.1 is evaluated, despite the name.
void test_msis21_vs_msise00_density() {
    double altitude_km = 693.0;  // Sentinel-1 altitude
    Epoch epoch = Epoch::from_utc(2016, 7, 15, 12, 0, 0);

    WeatherConditions weather;
    weather.f107_daily = Sentinel1Config::F107_DAILY;
    weather.f107_average = Sentinel1Config::F107_AVERAGE;
    weather.ap_daily = Sentinel1Config::AP_DAILY;
    weather.ap_history.fill(Sentinel1Config::AP_DAILY);

    atmosphere::NRLMSIS21 msis21;
    double rho_msis21 = msis21.density(altitude_km, 0.0, 0.0, epoch, weather);

    std::cout << "\n";
    std::cout << "    At " << altitude_km << " km (Sentinel-1 altitude):\n";
    std::cout << "    NRLMSIS 2.1:  " << std::scientific << std::setprecision(3)
              << rho_msis21 << " kg/m^3\n";
    std::cout << "    ";

    TEST_ASSERT(rho_msis21 > 0);
    TEST_ASSERT(rho_msis21 < 1e-10);
}

// Propagate with the standard RK45 propagator and one of three drag set-ups: drag off,
// drag on without an environment (use_msis21 = false: nothing is attached, so
// this is also drag-free), or a NRLMSIS 2.1
// DragEnvironment tabulated at the start (use_msis21 = true).
std::vector<std::pair<Epoch, StateVector>> propagate_with_msis21(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    double cd,
    bool enable_drag = true,
    bool use_msis21 = true,
    double sample_interval = 10.0)
{
    std::vector<std::pair<Epoch, StateVector>> results;

    BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, cd);

    WeatherConditions weather;
    weather.f107_daily = Sentinel1Config::F107_DAILY;
    weather.f107_average = Sentinel1Config::F107_AVERAGE;
    weather.ap_daily = Sentinel1Config::AP_DAILY;
    weather.ap_history.fill(Sentinel1Config::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos0, vel0, model, weather, true);

    if (!enable_drag) {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.drag = false;
        prop->force_model().set_force_flags(flags);
    } else if (use_msis21) {
        // One density-vs-altitude profile at the start time and the latitude and
        // longitude below the satellite. For a 1 h arc (0.61 orbits) at ~700 km this
        // is adequate here: the weather inputs (F10.7, Ap) are held constant anyway.
        // pos0 is the POE's Earth-fixed position, so it converts to geodetic
        // coordinates directly (treating it as inertial would misplace the profile).
        const auto geo = coords::ecef_to_geodetic(pos0);
        atmosphere::TimeWindow window{t0, t0 + duration_seconds + 1.0};
        auto sw = atmosphere::SpaceWeatherHistory::from_constant(weather, window);
        auto beta = atmosphere::BetaProfile1D::constant(model.beta);
        // Altitude grid spanning Sentinel-1's range (650–750 km in 10 km steps)
        std::vector<double> alt_grid;
        for (int alt = 650; alt <= 750; alt += 10) alt_grid.push_back(static_cast<double>(alt));
        const double lat_deg = geo.latitude * 180.0 / std::numbers::pi;
        const double lon_deg = geo.longitude * 180.0 / std::numbers::pi;
        auto env = std::make_shared<const atmosphere::DragEnvironment>(
            atmosphere::from_models_snapshot(
                t0, lat_deg, lon_deg, sw, std::move(beta),
                std::move(alt_grid)));
        prop->force_model().set_drag_environment(env);
    }

    int num_samples = static_cast<int>(duration_seconds / sample_interval) + 1;
    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * sample_interval;
        if (dt > duration_seconds) dt = duration_seconds;

        Epoch t = t0 + dt;
        auto result = (*prop)(t);

        StateVector state(result.position, result.velocity);
        results.emplace_back(t, state);
    }

    return results;
}

// Diagnostic (no assertion): 1-hour RMSE with drag off, the "MSISE-00" row
// (no environment attached, so also drag-free), and NRLMSIS 2.1 at
// Cd 2.2 / 3.0 / 5.0.
void test_1_hour_propagation_msis21() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    std::cout << "\n";
    std::cout << "    NRLMSIS 2.1 vs NRLMSISE-00 Comparison (1-hour):\n";
    std::cout << "    Configuration            RMSE(m)\n";
    std::cout << "    ----------------------------------\n";

    struct Config {
        std::string name;
        double cd;
        bool use_msis21;
        bool enable_drag;
    };

    std::vector<Config> configs = {
        {"Drag OFF", 2.2, false, false},
        {"MSISE-00, Cd=2.2", 2.2, false, true},
        {"MSIS 2.1, Cd=2.2", 2.2, true, true},
        {"MSIS 2.1, Cd=3.0", 3.0, true, true},
        {"MSIS 2.1, Cd=5.0", 5.0, true, true},
    };

    double best_rmse = 1e9;
    std::string best_config;

    for (const auto& cfg : configs) {
        auto results = propagate_with_msis21(t0, sv0.position, sv0.velocity,
                                             3600.0, cfg.cd, cfg.enable_drag, cfg.use_msis21);
        auto test_result = compute_propagation_result(results, *eof);

        std::cout << "    " << std::left << std::setw(24) << cfg.name
                  << std::right << std::fixed << std::setprecision(2)
                  << test_result.position_rmse << "\n";

        if (test_result.position_rmse < best_rmse) {
            best_rmse = test_result.position_rmse;
            best_config = cfg.name;
        }
    }

    std::cout << "    ----------------------------------\n";
    std::cout << "    Best: " << best_config << " (" << std::fixed << std::setprecision(2)
              << best_rmse << " m)\n";
    std::cout << "    ";

}

// Diagnostic (no assertion): 1-hour RMSE with NRLMSIS 2.1 for drag coefficients
// Cd from 1.0 to 5.0.
void test_msis21_cd_sweep() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    std::cout << "\n";
    std::cout << "    NRLMSIS 2.1 Cd Sweep (1-hour):\n";
    std::cout << "    Cd       RMSE(m)\n";
    std::cout << "    ------------------\n";

    std::vector<double> cd_values = {1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0};
    double best_rmse = 1e9;
    double best_cd = 2.2;

    for (double cd : cd_values) {
        auto results = propagate_with_msis21(t0, sv0.position, sv0.velocity,
                                             3600.0, cd, true, true);
        auto test_result = compute_propagation_result(results, *eof);

        std::cout << "    " << std::fixed << std::setprecision(1) << std::setw(4) << cd
                  << "     " << std::setprecision(2) << std::setw(7)
                  << test_result.position_rmse << "\n";

        if (test_result.position_rmse < best_rmse) {
            best_rmse = test_result.position_rmse;
            best_cd = cd;
        }
    }

    std::cout << "    ------------------\n";
    std::cout << "    Best: Cd=" << std::setprecision(1) << best_cd
              << " (" << std::setprecision(2) << best_rmse << " m)\n";
    std::cout << "    ";
}

#endif  // BALLISTIC_TEST_NO_NRL_PLUGIN

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "======================================\n";
    std::cout << "Sentinel-1 Orbit Propagation Validation\n";
    std::cout << "======================================\n\n";

    std::cout << "Test data path: " << get_test_data_path() << "\n\n";

    // Earth orientation parameters (EOP: polar motion, UT1-UTC) from the committed
    // IERS finals2000A file
    fs::path eop_path = fs::path(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all");
    if (time::EOPData::instance().load(eop_path)) {
        auto [mjd_min, mjd_max] = time::EOPData::instance().mjd_range();
        std::cout << "EOP data loaded: " << time::EOPData::instance().size()
                  << " records (MJD " << mjd_min << " - " << mjd_max << ")\n";

        // Print polar motion at one date (MJD 57580, July 2016) as a load check
        double test_mjd = 57580.0;
        auto [xp, yp] = time::EOPData::instance().polar_motion_rad(test_mjd);
        std::cout << "Polar motion at MJD " << test_mjd << ": xp="
                  << (xp / constants::ARCSEC_TO_RAD) << " arcsec, yp="
                  << (yp / constants::ARCSEC_TO_RAD) << " arcsec\n\n";
    } else {
        std::cout << "WARNING: Could not load EOP data from " << eop_path << "\n";
        std::cout << "Polar motion corrections will be skipped.\n\n";
    }

    // ==========================================================================
    // OPTIMAL CONFIGURATION TESTS (Sub-Meter Accuracy)
    // ==========================================================================
    // Trajectory class, GCRS, JPL DE421, ITRF2014, third-body + relativistic.
    // Target: <1 m RMSE with a JPL ephemeris file, <2 m without one.

    TestRunner optimal_tests("OPTIMAL CONFIGURATION (Target: <1m RMSE)");
    optimal_tests.add_test("optimal_trajectory_class", []() { test_optimal_trajectory_class(); });
    optimal_tests.add_test("optimal_gcrs_propagator", []() { test_optimal_gcrs_propagator(); });
    optimal_tests.run();

    // ==========================================================================
    // STANDARD TESTS
    // ==========================================================================

    // EOF Parser Tests
    TestRunner eof_tests("EOF Parser Tests");
    eof_tests.add_test("eof_parser_basic", []() { test_eof_parser_basic(); });
    eof_tests.add_test("eof_state_vector_count", []() { test_eof_state_vector_count(); });
    eof_tests.run();

    // Short-Term Propagation Tests
    TestRunner short_term_tests("Short-Term Propagation Tests");
    short_term_tests.add_test("1_minute_propagation", []() { test_1_minute_propagation(); });
    short_term_tests.add_test("10_minute_propagation", []() { test_10_minute_propagation(); });
    short_term_tests.run();

    // 1-Hour Propagation Tests (standard config)
    TestRunner hour_tests("1-Hour Propagation Tests (Standard Config, Target: <10m)");
    hour_tests.add_test("1_hour_propagation", []() { test_1_hour_propagation(); });
    hour_tests.add_test("1_hour_multiple_starts", []() { test_1_hour_multiple_starts(); });
    hour_tests.run();

    // Extended Propagation Tests
    TestRunner extended_tests("Extended Propagation Tests");
    extended_tests.add_test("full_orbit_propagation", []() { test_full_orbit_propagation(); });
    extended_tests.run();

    // Error Analysis Tests
    TestRunner error_tests("Error Analysis");
    error_tests.add_test("error_growth_rate", []() { test_error_growth_rate(); });
    error_tests.run();

    // Gravity Degree Analysis
    TestRunner gravity_tests("Gravity Degree Analysis");
    gravity_tests.add_test("gravity_degree_sweep", []() { test_gravity_degree_sweep(); });
    gravity_tests.run();

    // IAU End-to-End Test
    TestRunner iau_tests("IAU End-to-End Test");
    iau_tests.add_test("iau_end_to_end", []() { test_iau_end_to_end(); });
    iau_tests.run();

    // Time Scale Diagnostic
    TestRunner time_tests("Time Scale Diagnostic");
    time_tests.add_test("time_scale_diagnostic", []() { test_time_scale_diagnostic(); });
    time_tests.run();

    // Integrator Comparison
    TestRunner integrator_tests("Integrator Comparison");
    integrator_tests.add_test("rk45_vs_symplectic", []() { test_rk45_vs_symplectic(); });
    integrator_tests.run();

#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN
    // NRLMSIS 2.1 Atmosphere Tests
    TestRunner msis_tests("NRLMSIS 2.1 Atmosphere Tests");
    msis_tests.add_test("msis21_vs_msise00_density", []() { test_msis21_vs_msise00_density(); });
    msis_tests.add_test("1_hour_propagation_msis21", []() { test_1_hour_propagation_msis21(); });
    msis_tests.add_test("msis21_cd_sweep", []() { test_msis21_cd_sweep(); });
    msis_tests.run();
#endif

    return print_final_summary();
}
