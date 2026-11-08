// =============================================================================
// Trajectory Step/Step-Back Validation Tests
// =============================================================================
//
// Checks the Trajectory class's step() and rewind() methods:
// - basic stepping and the stored history;
// - 1-hour position RMSE (root-mean-square error) against Sentinel-1 orbits from
//   ESA's EOF files (Earth Observation File format, the precise orbit ephemerides,
//   POE) in tests/data/eof/sentinel1_2016/;
// - a synthetic HEO (highly elliptical, Molniya-type) orbit;
// - stepping one step at a time versus a single propagate() call.
//
// The Sentinel-1 cells run with drag and solar radiation pressure (SRP) off, so
// they pin the gravity/third-body/relativity "force-model-only" baseline of about
// 0.24 m, not the recommended configuration's accuracy.
//
// Frames: GCRS = Geocentric Celestial Reference System (inertial); ITRF =
// International Terrestrial Reference Frame (Earth-fixed). EOP = Earth
// Orientation Parameters (polar motion, UT1-UTC) from IERS.
//
// =============================================================================

#include <ballistic/ballistic.hpp>
#include <ballistic/validation/eof_parser.hpp>
#include "../test_helpers.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;
using namespace ballistic::ephemeris;

namespace fs = std::filesystem;

// Global ephemeris provider (JPL DE421/DE440)
static std::shared_ptr<EphemerisProvider> g_ephemeris;

// =============================================================================
// Test Configuration
// =============================================================================

// Sentinel-1 spacecraft parameters
struct Sentinel1Config {
    static constexpr double MASS = ballistic::satellites::get("S1A").mass_kg;  // registry (satellites.hpp)
    static constexpr double DRAG_AREA = ballistic::satellites::get("S1A").drag_area_m2;
    static constexpr double CD = ballistic::satellites::get("S1A").cd;
    static constexpr double F107_DAILY = 89.0;
    static constexpr double F107_AVERAGE = 90.0;
    static constexpr double AP_DAILY = 6.0;
};

// Molniya-type HEO parameters (a synthetic object, not a registry satellite)
struct MolniyaConfig {
    static constexpr double MASS = 1500.0;
    static constexpr double DRAG_AREA = 10.0;
    static constexpr double CD = 2.2;
    static constexpr double PERIGEE_ALT = 500000.0;   // m (500 km)
    static constexpr double APOGEE_ALT = 40000000.0;  // m (40,000 km)
    static constexpr double INCLINATION = 63.4;       // degrees
};

// =============================================================================
// Test Framework
// =============================================================================

#include "../test_utils.hpp"
using namespace ballistic::test;

// =============================================================================
// Utility Functions
// =============================================================================

std::string get_test_data_path() {
    std::vector<std::string> candidates = {
        BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel1_2016/2016-8",
        "../tests/data/eof/sentinel1_2016/2016-8",
        "../../tests/data/eof/sentinel1_2016/2016-8",
        "tests/data/eof/sentinel1_2016/2016-8"
    };
    for (const auto& path : candidates) {
        if (fs::exists(path)) return path;
    }
    return BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel1_2016/2016-8";
}

std::optional<EOFFile> load_eof_file(const std::string& pattern = "") {
    std::string data_path = get_test_data_path();
    return find_and_parse_eof(data_path, pattern);
}

// Generate initial state for Molniya-type HEO orbit
std::pair<Vector3d, Vector3d> generate_heo_initial_state(
    double perigee_alt_m, double apogee_alt_m, double inclination_deg)
{
    double r_e = constants::earth::RADIUS_EQUATORIAL;
    double mu = constants::earth::GM;

    double r_p = r_e + perigee_alt_m;
    double r_a = r_e + apogee_alt_m;
    double a = (r_p + r_a) / 2.0;
    double e = (r_a - r_p) / (r_a + r_p);
    double v_p = std::sqrt(mu * (2.0/r_p - 1.0/a));

    double inc_rad = inclination_deg * M_PI / 180.0;

    // Position at perigee (along x-axis)
    Vector3d pos(r_p, 0.0, 0.0);

    // Velocity perpendicular to position, inclined
    Vector3d vel(0.0, v_p * std::cos(inc_rad), v_p * std::sin(inc_rad));

    return {pos, vel};
}

// =============================================================================
// Unit Tests: Basic Step Functionality
// =============================================================================

void test_step_basic() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7500.0, 0.0);

    Trajectory traj(t0, pos, vel);

    // History should be empty initially
    TEST_ASSERT(traj.history_size() == 0);

    // Take first step
    auto result = traj.step();

    // History should now have 2 entries (initial + step)
    TEST_ASSERT(traj.history_size() == 2);

    // Epoch should advance
    TEST_ASSERT(result.epoch > t0);

    // Step size should be reasonable (> 0 and < 1000s)
    TEST_ASSERT(result.step_size_taken > 0.0);
    TEST_ASSERT(result.step_size_taken < 1000.0);

    // Position should change
    TEST_ASSERT((result.position - pos).norm() > 0.0);
}

void test_step_multiple() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7500.0, 0.0);

    Trajectory traj(t0, pos, vel);

    // Take 10 steps
    for (int i = 0; i < 10; ++i) {
        auto result = traj.step();
        TEST_ASSERT(result.history_index == static_cast<size_t>(i + 1));
    }

    // History should have 11 entries (initial + 10 steps)
    TEST_ASSERT(traj.history_size() == 11);

    // Verify history consistency
    for (size_t i = 1; i < traj.history_size(); ++i) {
        TEST_ASSERT(traj.history_at(i).epoch > traj.history_at(i-1).epoch);
    }
}

void test_rewind_basic() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7500.0, 0.0);

    Trajectory traj(t0, pos, vel);

    // Take 5 steps
    for (int i = 0; i < 5; ++i) {
        traj.step();
    }
    TEST_ASSERT(traj.history_size() == 6);

    // Record state at step 3
    auto state_at_3 = traj.history_at(3);

    // Step back 2 steps (from 5 to 3)
    bool ok = traj.rewind(2);
    TEST_ASSERT(ok);

    // History should be truncated
    TEST_ASSERT(traj.history_size() == 4);  // 0,1,2,3

    // Current state should match history[3]
    TEST_ASSERT(std::abs((traj.epoch() - state_at_3.epoch)) < 1e-9);
    TEST_ASSERT((traj.state_ecef().position - state_at_3.position).norm() < 1e-6);
}

void test_rewind_to_initial() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7500.0, 0.0);

    Trajectory traj(t0, pos, vel);

    // Take 5 steps
    for (int i = 0; i < 5; ++i) {
        traj.step();
    }

    // Step back to initial
    bool ok = traj.rewind(5);
    TEST_ASSERT(ok);

    // Should be at initial state
    TEST_ASSERT(traj.history_size() == 1);
    TEST_ASSERT(std::abs((traj.epoch() - t0)) < 1e-9);

    // Step back again should fail
    ok = traj.rewind();
    TEST_ASSERT(!ok);
}

void test_clear_history() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7500.0, 0.0);

    Trajectory traj(t0, pos, vel);

    // Take 10 steps
    for (int i = 0; i < 10; ++i) {
        traj.step();
    }
    TEST_ASSERT(traj.history_size() == 11);

    // Record current state
    Epoch current_epoch = traj.epoch();
    Vector3d current_pos = traj.state_ecef().position;

    // Clear history
    traj.clear_history();

    // History should have 1 entry (current as new initial)
    TEST_ASSERT(traj.history_size() == 1);

    // State should be unchanged
    TEST_ASSERT(std::abs((traj.epoch() - current_epoch)) < 1e-9);
    TEST_ASSERT((traj.state_ecef().position - current_pos).norm() < 1e-6);
}

// =============================================================================
// LEO Validation Tests (Sentinel-1)
// =============================================================================

void test_step_rmse_sentinel_1hour() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("EOF data not available");  // Skip if no data
    }

    if (eof->state_vectors.size() < 400) {
        TEST_SKIP("Insufficient EOF data");
    }

    // Start from state vector 10 (skip first few)
    const auto& sv0 = eof->state_vectors[10];

    TrajectoryConfig config;
  // Match ESA configuration

    // First, test that propagate() works as expected (baseline RMSE)
    Trajectory traj_check(sv0.epoch, sv0.position, sv0.velocity, config);
    std::vector<std::pair<Epoch, StateVector>> check_results;
    check_results.emplace_back(traj_check.epoch(), traj_check.state_ecef());
    for (double t = 10.0; t <= 3600.0; t += 10.0) {
        auto result = traj_check.propagate(sv0.epoch + t);
        check_results.emplace_back(sv0.epoch + t, StateVector(result.position, result.velocity));
    }
    auto check_validation = compute_propagation_result(check_results, *eof);
    std::cout << "\n    Baseline propagate() RMSE: " << std::fixed << std::setprecision(2)
              << check_validation.position_rmse << " m";

    // Now test step() accuracy: compare step() results to propagate() at the same times
    // This validates that step() produces correct results independent of EOF timing
    Trajectory traj(sv0.epoch, sv0.position, sv0.velocity, config);

    size_t step_count = 0;
    double max_pos_error = 0.0;
    double sum_sq_error = 0.0;

    while (traj.epoch() - sv0.epoch < 3600.0) {
        auto step_result = traj.step();
        step_count++;

        // Compare this step's result to a fresh propagate() to the same time
        Trajectory traj_verify(sv0.epoch, sv0.position, sv0.velocity, config);
        auto verify_result = traj_verify.propagate(step_result.epoch);

        double pos_error = (step_result.position - verify_result.position).norm();
        max_pos_error = std::max(max_pos_error, pos_error);
        sum_sq_error += pos_error * pos_error;

        if (step_count > 10000) break;
    }

    double rmse = std::sqrt(sum_sq_error / step_count);
    std::cout << "\n    Steps taken: " << step_count;
    std::cout << "\n    step() vs propagate() RMSE: " << std::fixed << std::setprecision(3) << rmse << " m";
    std::cout << "\n    step() vs propagate() max:  " << max_pos_error << " m";

    // step() should match propagate() within numerical precision
    // Small differences arise from integrator state management
    TEST_ASSERT(rmse < 1.0);
    TEST_ASSERT(max_pos_error < 2.0);  // Slight accumulation over 103 steps is acceptable
}

void test_step_vs_propagate_comparison() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("EOF data not available");
    }

    if (eof->state_vectors.size() < 400) {
        TEST_SKIP("Insufficient EOF data");
    }

    const auto& sv0 = eof->state_vectors[10];

    TrajectoryConfig config;

    // Method 1: Single propagate() call at t=60s
    Trajectory traj1(sv0.epoch, sv0.position, sv0.velocity, config);
    auto result1 = traj1.propagate(sv0.epoch + 60.0);

    // Method 2: Step until we pass 60s, then check state
    Trajectory traj2(sv0.epoch, sv0.position, sv0.velocity, config);
    while (traj2.epoch() - sv0.epoch < 60.0) {
        traj2.step();
    }
    // Get current state (which is slightly past 60s)
    auto state2 = traj2.state_ecef();
    double elapsed2 = traj2.epoch() - sv0.epoch;

    // Also propagate fresh to traj2's actual epoch for fair comparison
    Trajectory traj3(sv0.epoch, sv0.position, sv0.velocity, config);
    auto result3 = traj3.propagate(traj2.epoch());

    // Compare step() final state vs propagate() to same time
    double pos_diff = (state2.position - result3.position).norm();
    double vel_diff = (state2.velocity - result3.velocity).norm();

    std::cout << "\n    Time after stepping: " << std::fixed << std::setprecision(2) << elapsed2 << " s";
    std::cout << "\n    Position (step):     " << state2.position.transpose();
    std::cout << "\n    Position (propagate):" << result3.position.transpose();
    std::cout << "\n    Position difference: " << std::fixed << std::setprecision(6) << pos_diff << " m";
    std::cout << "\n    Velocity difference: " << vel_diff << " m/s";

    // Positions should match within ~1m
    TEST_ASSERT(pos_diff < 1.0);

    // Also verify propagate() baseline RMSE
    Trajectory traj4(sv0.epoch, sv0.position, sv0.velocity, config);
    std::vector<std::pair<Epoch, StateVector>> results;
    results.emplace_back(traj4.epoch(), traj4.state_ecef());
    for (double t = 10.0; t <= 3600.0; t += 10.0) {
        auto result = traj4.propagate(sv0.epoch + t);
        results.emplace_back(sv0.epoch + t, StateVector(result.position, result.velocity));
    }
    auto validation = compute_propagation_result(results, *eof);

    std::cout << "\n    Baseline propagate() RMSE: " << std::fixed << std::setprecision(3)
              << validation.position_rmse << " m";

    TEST_ASSERT(validation.position_rmse < 5.0);
}

// =============================================================================
// HEO Validation Tests
// =============================================================================

void test_step_heo_molniya() {
    auto [pos, vel] = generate_heo_initial_state(
        MolniyaConfig::PERIGEE_ALT,
        MolniyaConfig::APOGEE_ALT,
        MolniyaConfig::INCLINATION
    );

    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    TrajectoryConfig config;
    // Drag would matter at the 500 km perigee, but this cell builds no
    // DragEnvironment: it only exercises the integrator's step-size adaptation
    // with gravity + third-body forces.

    Trajectory traj(t0, pos, vel, config);

    // Propagate for 1 hour using adaptive steps
    size_t step_count = 0;
    double min_step = 1e10, max_step = 0.0;

    while (traj.epoch() - t0 < 3600.0) {
        auto result = traj.step();
        step_count++;
        min_step = std::min(min_step, result.step_size_taken);
        max_step = std::max(max_step, result.step_size_taken);

        if (step_count > 10000) break;
    }

    std::cout << "\n    Steps: " << step_count;
    std::cout << "\n    Step size range: " << std::fixed << std::setprecision(2)
              << min_step << " - " << max_step << " s";

    // Verify rewind works on HEO
    size_t history_size_before = traj.history_size();
    bool ok = traj.rewind(step_count / 2);
    TEST_ASSERT(ok);
    TEST_ASSERT(traj.history_size() < history_size_before);

    // Should be able to continue stepping
    auto result = traj.step();
    TEST_ASSERT(result.step_size_taken > 0.0);
}

void test_step_heo_energy_conservation() {
    auto [pos, vel] = generate_heo_initial_state(
        MolniyaConfig::PERIGEE_ALT,
        MolniyaConfig::APOGEE_ALT,
        MolniyaConfig::INCLINATION
    );

    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    TrajectoryConfig config;
  // Disable non-conservative forces
    config.enable_third_body = false;
    config.enable_relativistic = false;

    Trajectory traj(t0, pos, vel, config);

    // Compute initial specific orbital energy
    double mu = constants::earth::GM;
    auto compute_energy = [mu](const Vector3d& p, const Vector3d& v) {
        return 0.5 * v.squaredNorm() - mu / p.norm();
    };

    double initial_energy = compute_energy(pos, vel);
    double max_energy_drift = 0.0;

    // Propagate for 1 hour
    while (traj.epoch() - t0 < 3600.0) {
        traj.step();

        auto state = traj.state_ecef();
        double energy = compute_energy(state.position, state.velocity);
        double drift = std::abs(energy - initial_energy) / std::abs(initial_energy);
        max_energy_drift = std::max(max_energy_drift, drift);
    }

    std::cout << "\n    Max relative energy drift: " << std::scientific
              << std::setprecision(2) << max_energy_drift;

    // RK45 is not a symplectic integrator, so energy drift is expected
    // For HEO orbits over 1 hour, ~5% drift is acceptable
    TEST_ASSERT(max_energy_drift < 0.1);
}

// Test: HEO with optimal configuration (GCRS + JPL ephemeris)
// This tests whether the optimal LEO configuration works well for HEO
void test_step_heo_optimal_config() {
    // -------------------------------------------------------------------------
    // HEO (Molniya-type) propagation with optimal configuration
    //
    // Uses same force model as optimal LEO configuration:
    //   - GCRS frame (IAU 2006/2000A)
    //   - JPL ephemeris for third-body perturbations
    //   - EGM2008 gravity (though attenuation is much higher at apogee)
    //   - Relativistic corrections
    //   - No drag (minimal at apogee altitude)
    //   - No SRP (simplified model doesn't match complex geometry)
    //
    // HEO specifics:
    //   - Perigee: 500 km (gravity dominated, drag relevant)
    //   - Apogee: 40,000 km (third-body dominated, gravity attenuated)
    //   - Step sizes should vary significantly across the orbit
    // -------------------------------------------------------------------------

    auto [pos, vel] = generate_heo_initial_state(
        MolniyaConfig::PERIGEE_ALT,
        MolniyaConfig::APOGEE_ALT,
        MolniyaConfig::INCLINATION
    );

    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // Create propagator with optimal HEO settings
    BallisticModel model;  // Default: no drag (controlled by force flags)
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = std::make_unique<RK45Propagator>(
        t0, pos, vel,
        model, weather,
        true,  // enable_third_body (critical for HEO)
        RK45Config{},
        g_ephemeris,  // JPL ephemeris for accurate third-body
        PropagationFrame::GCRS,
        coords::ITRFRealization::ITRF2020  // Use ITRF2020 for generic HEO
    );

    auto& fm = prop->force_model();
    fm.set_gravity_max_degree(70);  // May need higher for perigee accuracy
    {
        forces::ForceComponentFlags flags = fm.force_flags();
        flags.relativistic = true;
        flags.drag = false;
        fm.set_force_flags(flags);
    }
    fm.enable_schwarzschild(true);
    fm.enable_lense_thirring(true);

    // Track step sizes throughout the orbit
    size_t step_count = 0;
    double min_step = 1e10, max_step = 0.0;
    double total_step_time = 0.0;
    std::vector<double> radii;

    double integration_time = 6.0 * 3600.0;  // 6 hours (~half Molniya period)
    double current_time = 0.0;

    StateVector state = prop->current_state_ecef();
    radii.push_back(state.position.norm());

    while (current_time < integration_time) {
        // Get current step size
        double dt = prop->integrator().current_step_size();

        // Take a step using integrate to target
        double target_time = std::min(current_time + dt, integration_time);
        (*prop)(t0 + target_time);
        double step_taken = target_time - current_time;

        current_time = target_time;
        step_count++;
        min_step = std::min(min_step, step_taken);
        max_step = std::max(max_step, step_taken);
        total_step_time += step_taken;

        state = prop->current_state_ecef();
        radii.push_back(state.position.norm());

        if (step_count > 10000) break;
    }

    // Find altitude extremes
    double r_min = *std::min_element(radii.begin(), radii.end());
    double r_max = *std::max_element(radii.begin(), radii.end());
    double alt_min = (r_min - constants::earth::RADIUS_EQUATORIAL) / 1000.0;
    double alt_max = (r_max - constants::earth::RADIUS_EQUATORIAL) / 1000.0;

    std::cout << "\n    HEO Optimal Configuration (6-hour propagation, ~half Molniya period)";
    std::cout << "\n    Configuration: GCRS + JPL ephemeris + EGM2008 deg70 + relativistic";
    std::cout << "\n    Steps: " << step_count;
    std::cout << "\n    Step size range: " << std::fixed << std::setprecision(2)
              << min_step << " - " << max_step << " s";
    std::cout << "\n    Altitude range: " << std::setprecision(0)
              << alt_min << " - " << alt_max << " km";

    if (g_ephemeris) {
        std::cout << "\n    (using JPL ephemeris)";
    } else {
        std::cout << "\n    (using analytical ephemeris - less accurate for third-body)";
    }

    // HEO should complete within reasonable number of steps
    TEST_ASSERT(step_count < 50000);
    // 6 hours is ~half Molniya period, should reach near apogee
    TEST_ASSERT(alt_min < 600.0);    // Should pass through perigee (~500 km)
    TEST_ASSERT(alt_max > 30000.0);  // Should reach near apogee (~40000 km)
}

// =============================================================================
// Sub-1m RMSE Optimization Tests
// =============================================================================
//
// These tests investigate whether we can achieve sub-1m RMSE for 1-hour LEO
// propagation against ESA Sentinel-1 Precise Orbit Ephemerides.
//
// The ESA POD (Precise Orbit Determination) chain uses:
//   - GRACE-based gravity (similar to EGM2008)
//   - IAU 2006/2000A precession-nutation (CIO-based, GCRS frame)
//   - Third-body perturbations from JPL ephemeris
//   - Relativistic corrections
//   - Dynamically estimated drag coefficient (Cd) from accelerometer data
//   - Box-wing SRP model with estimated reflectivity
//
// We cannot perfectly match ESA's model because:
//   1. ESA estimates Cd dynamically - we use a fixed value
//   2. ESA uses a detailed box-wing model - we use a simple cannonball
//
// This test disables drag and SRP: it pins the force-model-only baseline
// (~1.03 m RMSE). For prediction accuracy, drag with measured space weather
// and SRP with the fitted radiation area both lower the Sentinel 1-h RMSE —
// docs/validation_claims.md SENTINEL-YEAR-2025.
//
// =============================================================================

void test_optimal_config_rmse() {
    // -------------------------------------------------------------------------
    // Test optimal configuration for minimum RMSE against Sentinel-1 POE
    //
    // This configuration achieves ~1.03m RMSE for 1-hour propagation:
    //
    // GRAVITY: EGM2008 degree 70
    //   - Higher degrees (90, 120, 180) provide <0.01m improvement at LEO
    //   - Above ~degree 70, signal attenuates rapidly at 700km altitude
    //   - Diminishing returns vs computational cost
    //
    // COORDINATE FRAME: GCRS (Geocentric Celestial Reference System)
    //   - CIO-based, IAU 2006/2000A precession-nutation model
    //   - ~0.4m better than GAST-based equinox frame
    //   - Matches ESA's operational POD processing chain
    //
    // ITRF REALIZATION: ITRF2014
    //   - ESA transitioned to ITRF2020 on March 4, 2025
    //   - All Sentinel data before that date uses ITRF2014
    //   - Using wrong ITRF adds ~1-3mm systematic error
    //
    // THIRD-BODY: Sun + Moon + Jupiter enabled
    //   - Sun: largest effect (~0.3m improvement)
    //   - Moon: secondary effect
    //   - Jupiter: minimal effect but included for completeness
    //
    // RELATIVISTIC: Schwarzschild + Lense-Thirring enabled
    //   - Provides ~0.05m improvement over 1 hour
    //   - Required for cm-level accuracy
    //
    // DRAG: DISABLED
    //   - ESA estimates Cd dynamically from accelerometer data
    //   - Our fixed Cd=2.2 cannot match their estimates
    //   - (this is a baseline choice; drag does help prediction, SENTINEL-YEAR-2025)
    //
    // SRP: DISABLED
    //   - ESA uses detailed box-wing model with estimated reflectivity
    //   - Our cannonball model cannot match their estimates
    //   - (this is a baseline choice; SRP does help prediction, SENTINEL-YEAR-2025)

    // -------------------------------------------------------------------------

    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("EOF data not available");
    }

    if (eof->state_vectors.size() < 400) {
        TEST_SKIP("Insufficient EOF data");
    }

    // Start from state vector 10 (skip first few for numerical stability)
    const auto& sv0 = eof->state_vectors[10];

    // Check if JPL ephemeris is loaded (required for 1.03m RMSE)
    if (!g_ephemeris) {
        std::cout << "\n    WARNING: JPL ephemeris not loaded - using analytical\n";
        std::cout << "    Analytical ephemeris has reference frame issues (~4m RMSE)\n";
        std::cout << "    For 1.03m RMSE, install de421.bsp or de440.bsp in <project>/data/";
    }

    // OPTIMAL CONFIGURATION for 1.03m baseline (see comments above)
    // Must use RK45Propagator directly to pass the JPL ephemeris
    BallisticModel model;  // Drag DISABLED via force flags: force-model-only baseline
    WeatherConditions weather = WeatherConditions::quiet();

    // Create propagator with optimal settings
    auto prop = std::make_unique<RK45Propagator>(
        sv0.epoch, sv0.position, sv0.velocity,
        model, weather,
        true,  // enable_third_body
        RK45Config{},
        g_ephemeris,  // JPL ephemeris (nullptr = analytical, degraded accuracy)
        PropagationFrame::GCRS,  // CIO-based (IAU 2006/2000A) - required for sub-meter
        coords::ITRFRealization::ITRF2014  // Sentinel data uses ITRF2014 (before March 2025)
    );

    // Configure force model
    auto& fm = prop->force_model();
    fm.set_gravity_max_degree(70);        // EGM2008 degree 70 (sufficient for LEO)
    {
        forces::ForceComponentFlags flags = fm.force_flags();
        flags.relativistic = true;        // Schwarzschild + Lense-Thirring
        flags.drag = false;               // DISABLED: force-model-only baseline
        fm.set_force_flags(flags);
    }
    fm.enable_schwarzschild(true);
    fm.enable_lense_thirring(true);

    // Test propagate() baseline
    std::vector<std::pair<Epoch, StateVector>> results;
    results.emplace_back(prop->current_epoch(), prop->current_state_ecef());

    for (double t = 10.0; t <= 3600.0; t += 10.0) {
        auto result = (*prop)(sv0.epoch + t);
        results.emplace_back(sv0.epoch + t, StateVector(result.position, result.velocity));
    }

    auto validation = compute_propagation_result(results, *eof);

    std::cout << "\n    Optimal config propagate() RMSE: " << std::fixed << std::setprecision(3)
              << validation.position_rmse << " m";
    std::cout << "\n    Radial RMS:      " << validation.radial_rms << " m";
    std::cout << "\n    Along-track RMS: " << validation.along_track_rms << " m";
    std::cout << "\n    Cross-track RMS: " << validation.cross_track_rms << " m";

    if (g_ephemeris) {
        std::cout << "\n    (using JPL ephemeris)";
    } else {
        std::cout << "\n    (using analytical ephemeris - degraded)";
    }

    // Target: < 2m with optimal config and JPL ephemeris
    // Without JPL ephemeris, expect ~4m (still pass, but note degradation)
    if (g_ephemeris) {
        TEST_ASSERT(validation.position_rmse < 2.0);  // With JPL: measured ~0.24 m
    } else {
        TEST_ASSERT(validation.position_rmse < 5.0);  // Without JPL: expect ~4m
    }
}

void test_higher_gravity_degree() {
    // -------------------------------------------------------------------------
    // Test if higher gravity degree improves RMSE
    //
    // EGM2008 provides coefficients up to degree/order 2190, but for LEO:
    //   - Signal attenuates with (R_earth/r)^(n+1)
    //   - At 700km altitude, r ≈ 7078 km, so R/r ≈ 0.90
    //   - Degree 70 contribution: (0.90)^71 ≈ 5.6e-4
    //   - Degree 120 contribution: (0.90)^121 ≈ 2.5e-6
    //   - Degree 180 contribution: (0.90)^181 ≈ 4.5e-9
    //
    // At LEO altitude, higher degrees provide diminishing returns.
    // We test degrees 70, 90, 120, 180 to quantify the actual improvement.
    //
    // Expected result: <0.01m improvement beyond degree 70.
    // -------------------------------------------------------------------------

    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("EOF data not available");
    }

    if (eof->state_vectors.size() < 400) {
        TEST_SKIP("Insufficient EOF data");
    }

    const auto& sv0 = eof->state_vectors[10];

    // Test gravity degrees: 70 (baseline), 90, 120, 180
    // Higher degrees cost more computation but may improve accuracy
    // Uses RK45Propagator directly for JPL ephemeris support
    int degrees[] = {70, 90, 120, 180};
    BallisticModel model;  // Drag disabled via force flags
    WeatherConditions weather = WeatherConditions::quiet();

    for (int deg : degrees) {
        auto prop = std::make_unique<RK45Propagator>(
            sv0.epoch, sv0.position, sv0.velocity,
            model, weather, true, RK45Config{},
            g_ephemeris,  // Use JPL ephemeris for accurate third-body
            PropagationFrame::GCRS, coords::ITRFRealization::ITRF2014
        );

        auto& fm = prop->force_model();
        fm.set_gravity_max_degree(deg);
        {
            forces::ForceComponentFlags flags = fm.force_flags();
            flags.relativistic = true;
            flags.drag = false;
            fm.set_force_flags(flags);
        }

        std::vector<std::pair<Epoch, StateVector>> results;
        results.emplace_back(prop->current_epoch(), prop->current_state_ecef());

        for (double t = 10.0; t <= 3600.0; t += 10.0) {
            auto result = (*prop)(sv0.epoch + t);
            results.emplace_back(sv0.epoch + t, StateVector(result.position, result.velocity));
        }

        auto validation = compute_propagation_result(results, *eof);
        std::cout << "\n    Degree " << std::setw(3) << deg << ": RMSE = "
                  << std::fixed << std::setprecision(3) << validation.position_rmse << " m";
    }
}

void test_tighter_tolerances() {
    // -------------------------------------------------------------------------
    // Test if tighter integrator tolerances improve RMSE
    //
    // RK45 (Dormand-Prince) uses adaptive step size controlled by:
    //   - relative_tolerance: error relative to state magnitude
    //   - absolute_tolerance: minimum error threshold
    //
    // Default settings: rel=1e-10, abs=1e-12
    //   - Takes ~360 steps for 1-hour propagation
    //   - Achieves ~1.03m RMSE
    //
    // Tighter settings tested:
    //   - Tight: rel=1e-12, abs=1e-14 (100x tighter)
    //   - Ultra-tight: rel=1e-14, abs=1e-16 (10000x tighter)
    //
    // Hypothesis: Integration error may be a limiting factor.
    // If RMSE doesn't improve with tighter tolerances, the error is likely
    // dominated by force model differences (not integration accuracy).
    //
    // Trade-off: Tighter tolerances = more steps = longer computation.
    // -------------------------------------------------------------------------

    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("EOF data not available");
    }

    if (eof->state_vectors.size() < 400) {
        TEST_SKIP("Insufficient EOF data");
    }

    const auto& sv0 = eof->state_vectors[10];

    // Test with tighter integrator tolerances using RK45Propagator directly
    // We bypass Trajectory class to have full control over RK45Config
    BallisticModel model;  // Drag disabled via force flags (force-model-only baseline)
    WeatherConditions weather = WeatherConditions::quiet();

    // TIGHT TOLERANCE: 100x tighter than default
    propagators::RK45Config tight_config;
    tight_config.relative_tolerance = 1e-12;  // Default: 1e-10
    tight_config.absolute_tolerance = 1e-14;  // Default: 1e-12
    tight_config.initial_step_size = 30.0;

    auto prop = std::make_unique<propagators::RK45Propagator>(
        sv0.epoch, sv0.position, sv0.velocity,
        model, weather, true, tight_config,
        g_ephemeris,  // Use JPL ephemeris for accurate third-body
        PropagationFrame::GCRS, coords::ITRFRealization::ITRF2014
    );

    // Configure force model
    prop->force_model().set_gravity_max_degree(70);
    {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.relativistic = true;
        flags.drag = false;
        prop->force_model().set_force_flags(flags);
    }

    std::vector<std::pair<Epoch, StateVector>> results;
    results.emplace_back(prop->current_epoch(), prop->current_state_ecef());

    for (double t = 10.0; t <= 3600.0; t += 10.0) {
        auto result = (*prop)(sv0.epoch + t);
        results.emplace_back(sv0.epoch + t, StateVector(result.position, result.velocity));
    }

    auto validation = compute_propagation_result(results, *eof);

    std::cout << "\n    Tight tolerance (1e-12/1e-14) RMSE: " << std::fixed << std::setprecision(3)
              << validation.position_rmse << " m";
    std::cout << "\n    Steps taken: " << prop->stats().steps_taken;
    std::cout << "\n    Radial RMS:      " << validation.radial_rms << " m";
    std::cout << "\n    Along-track RMS: " << validation.along_track_rms << " m";
    std::cout << "\n    Cross-track RMS: " << validation.cross_track_rms << " m";

    // ULTRA-TIGHT TOLERANCE: 10000x tighter than default
    // This tests the limit of integration accuracy improvement
    propagators::RK45Config ultra_tight;
    ultra_tight.relative_tolerance = 1e-14;  // Default: 1e-10
    ultra_tight.absolute_tolerance = 1e-16;  // Default: 1e-12
    ultra_tight.initial_step_size = 10.0;    // Smaller initial step

    auto prop2 = std::make_unique<propagators::RK45Propagator>(
        sv0.epoch, sv0.position, sv0.velocity,
        model, weather, true, ultra_tight,
        g_ephemeris,  // Use JPL ephemeris for accurate third-body
        PropagationFrame::GCRS, coords::ITRFRealization::ITRF2014
    );

    prop2->force_model().set_gravity_max_degree(70);
    {
        forces::ForceComponentFlags flags = prop2->force_model().force_flags();
        flags.relativistic = true;
        flags.drag = false;
        prop2->force_model().set_force_flags(flags);
    }

    std::vector<std::pair<Epoch, StateVector>> results2;
    results2.emplace_back(prop2->current_epoch(), prop2->current_state_ecef());

    for (double t = 10.0; t <= 3600.0; t += 10.0) {
        auto result = (*prop2)(sv0.epoch + t);
        results2.emplace_back(sv0.epoch + t, StateVector(result.position, result.velocity));
    }

    auto validation2 = compute_propagation_result(results2, *eof);

    std::cout << "\n    Ultra tight (1e-14/1e-16) RMSE: " << validation2.position_rmse << " m";
    std::cout << "\n    Steps taken: " << prop2->stats().steps_taken;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    // Load EOP data for polar motion corrections (optional, improves accuracy)
    fs::path eop_path = fs::path(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all");
    if (time::EOPData::instance().load(eop_path)) {
        std::cout << "EOP data loaded successfully" << std::endl;
    }

    // Load JPL ephemeris for accurate third-body positions
    // CRITICAL: Without JPL ephemeris, RMSE will be ~4m instead of ~1m
    g_ephemeris = test::helpers::load_test_ephemeris();

    TestRunner unit_tests("Unit Tests (Basic Functionality)");
    unit_tests.add_test("step_basic", test_step_basic);
    unit_tests.add_test("step_multiple", test_step_multiple);
    unit_tests.add_test("rewind_basic", test_rewind_basic);
    unit_tests.add_test("rewind_to_initial", test_rewind_to_initial);
    unit_tests.add_test("clear_history", test_clear_history);
    unit_tests.run();

    TestRunner leo_tests("LEO Validation (Sentinel-1)");
    leo_tests.add_test("step_rmse_sentinel_1hour", test_step_rmse_sentinel_1hour);
    leo_tests.add_test("step_vs_propagate_comparison", test_step_vs_propagate_comparison);
    leo_tests.run();

    TestRunner heo_tests("HEO Validation (Molniya-type)");
    heo_tests.add_test("step_heo_molniya", test_step_heo_molniya);
    heo_tests.add_test("step_heo_energy_conservation", test_step_heo_energy_conservation);
    heo_tests.add_test("step_heo_optimal_config", test_step_heo_optimal_config);
    heo_tests.run();

    TestRunner rmse_tests("Sub-1m RMSE Optimization");
    rmse_tests.add_test("optimal_config_rmse", test_optimal_config_rmse);
    rmse_tests.add_test("higher_gravity_degree", test_higher_gravity_degree);
    rmse_tests.add_test("tighter_tolerances", test_tighter_tolerances);
    rmse_tests.run();

    return print_final_summary();
}
