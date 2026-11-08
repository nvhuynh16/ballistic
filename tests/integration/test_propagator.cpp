// Integration tests for RK45Propagator (the older propagator interface;
// Trajectory is the recommended one): construction, state access, forward /
// backward propagation, reset, statistics, energy conservation and the
// conversion between the Earth-fixed frame ECEF (Earth-Centered Earth-Fixed,
// the frame of the inputs and outputs) and the inertial frame it integrates
// in (GCRS, called ECI = Earth-Centered Inertial in some names below).
// RK45 = the adaptive Dormand-Prince Runge-Kutta 4(5) integrator.
// Thresholds are physical sanity bounds unless a cell says otherwise.

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/propagators/propagator.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::propagators;
using namespace ballistic::coords;

// =============================================================================
// RK45Propagator Construction Tests
// =============================================================================

void test_propagator_construction_default() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);  // LEO altitude
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    TEST_ASSERT_MSG(std::abs(prop.current_epoch() - epoch) < 1e-3,
                "Epoch should match");
}

void test_propagator_construction_with_ballistic() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);
    BallisticModel ballistic(45.45, 1000.0, 10.0);

    // The BallisticModel/WeatherConditions arguments remain in the signature
    // but do not configure drag (drag comes only from a DragEnvironment).
    // This test just confirms the constructor accepts them.
    RK45Propagator prop(epoch, pos_ecef, vel_ecef, ballistic, WeatherConditions::quiet());
    TEST_ASSERT_MSG(prop.current_epoch() == epoch, "Propagator constructed");
}

void test_propagator_construction_third_body_enabled() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), true);

    // Third body should be enabled in force model
    TEST_ASSERT_MSG(prop.force_model().third_body().sun_enabled(), "Sun should be enabled");
    TEST_ASSERT_MSG(prop.force_model().third_body().moon_enabled(), "Moon should be enabled");
}

void test_propagator_construction_third_body_disabled() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), false);

    // Third body should be disabled
    TEST_ASSERT_MSG(!prop.force_model().third_body().sun_enabled(), "Sun should be disabled");
    TEST_ASSERT_MSG(!prop.force_model().third_body().moon_enabled(), "Moon should be disabled");
}

void test_propagator_construction_with_config() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Config config;
    config.relative_tolerance = 1e-12;
    config.absolute_tolerance = 1e-14;

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), true, config);

    // Verify propagator was constructed with correct epoch
    TEST_ASSERT_MSG(std::abs(prop.current_epoch() - epoch) < 1e-3,
                "Construction with config should preserve epoch");
}

// =============================================================================
// State Access Tests
// =============================================================================

void test_propagator_current_epoch() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    TEST_ASSERT_MSG(std::abs(prop.current_epoch() - epoch) < 1e-3,
                "Current epoch should match initial");
}

void test_propagator_current_state_eci() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    StateVector state_eci = prop.current_state_gcrs();

    // State should be converted from ECEF
    double r = state_eci.position.norm();
    TEST_ASSERT_MSG(std::abs(r - 6778137.0) < 1.0, "Position magnitude should be preserved");
}

void test_propagator_current_state_ecef() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    StateVector state_ecef = prop.current_state_ecef();

    // Should match initial state (approximately, due to roundtrip)
    TEST_ASSERT_MSG((state_ecef.position - pos_ecef).norm() < 1.0,
                "ECEF position should be preserved");
}

void test_propagator_force_model_access() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    // Should be able to access and modify force model
    forces::ForceModel& fm = prop.force_model();
    forces::ForceComponentFlags flags = fm.force_flags();
    flags.drag = true;
    fm.set_force_flags(flags);

    TEST_ASSERT_MSG(prop.force_model().force_flags().drag, "Force model should be modifiable");
}

// =============================================================================
// Basic Propagation Tests
// =============================================================================

void test_propagator_propagate_zero_time() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    StateVector result = prop.propagate_to(epoch);

    // Should return current state
    TEST_ASSERT_MSG((result.position - prop.current_state_gcrs().position).norm() < 1e-6,
                "Zero time propagation should return same state");
}

void test_propagator_propagate_short_time() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    // Propagate 60 seconds
    Epoch target = epoch + 60.0;
    StateVector result = prop.propagate_to(target);

    // Position should change
    double r = result.position.norm();
    TEST_ASSERT_MSG(r > 6700000.0 && r < 6900000.0, "Position should remain in LEO range");

    // Epoch should update
    TEST_ASSERT_MSG(std::abs(prop.current_epoch() - target) < 1e-3,
                "Epoch should update after propagation");
}

void test_propagator_propagate_one_orbit() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // Circular orbit at 400km — account for Earth rotation in ECEF velocity
    double r = constants::earth::RADIUS_EQUATORIAL + 400000.0;
    double v_eci = std::sqrt(constants::earth::GM / r);
    // ECEF velocity must subtract Earth rotation: v_ecef = v_eci - omega * r
    double omega_earth = 7.2921150e-5;  // rad/s
    double v_ecef = v_eci - omega_earth * r;

    Vector3d pos_ecef(r, 0.0, 0.0);
    Vector3d vel_ecef(0.0, v_ecef, 0.0);

    // Disable drag, solid tides, relativistic for clean orbital test
    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), false);
    {
        forces::ForceComponentFlags flags = prop.force_model().force_flags();
        flags.drag = false;
        flags.solid_earth_tides = false;
        flags.relativistic = false;
        prop.force_model().set_force_flags(flags);
    }

    // Compute actual orbital period from ECI state energy
    StateVector initial = prop.current_state_gcrs();
    double E = 0.5 * initial.velocity.squaredNorm() - constants::earth::GM / initial.position.norm();
    double a = -constants::earth::GM / (2.0 * E);  // semi-major axis
    double T = constants::TWO_PI * std::sqrt(a * a * a / constants::earth::GM);

    prop.propagate_to(epoch + T);
    StateVector final_state = prop.current_state_gcrs();

    // After one period the position should return to within 150 km of the
    // start: J2 (Earth's oblateness) shifts the orbit along-track each revolution.
    double pos_error = (final_state.position - initial.position).norm();
    TEST_ASSERT_MSG(pos_error < 150000.0, "Position should return near initial after one orbit (J2 drift ~120 km)");
}

void test_propagator_call_operator() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    PropagationResult result = prop(epoch + 60.0);

    // Result should be in ECEF
    TEST_ASSERT_MSG(result.position.norm() > 6700000.0, "Position should be reasonable");
    TEST_ASSERT_MSG(result.velocity.norm() > 7000.0, "Velocity should be reasonable");
    TEST_ASSERT_MSG(result.acceleration.norm() > 5.0, "Acceleration should be present");
}

// =============================================================================
// Forward and Backward Propagation
// =============================================================================

void test_propagator_forward_propagation() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), false);
    {
        forces::ForceComponentFlags flags = prop.force_model().force_flags();
        flags.drag = false;
        prop.force_model().set_force_flags(flags);
    }

    // Forward 1 hour
    PropagationResult result = prop(epoch + 3600.0);

    TEST_ASSERT_MSG(result.position.norm() > 6700000.0, "Position should be valid");
}

void test_propagator_backward_propagation() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), false);
    {
        forces::ForceComponentFlags flags = prop.force_model().force_flags();
        flags.drag = false;
        prop.force_model().set_force_flags(flags);
    }

    // Backward 1 hour
    PropagationResult result = prop(epoch - 3600.0);

    TEST_ASSERT_MSG(result.position.norm() > 6700000.0, "Position should be valid for backward prop");
}

void test_propagator_roundtrip() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), false);
    {
        forces::ForceComponentFlags flags = prop.force_model().force_flags();
        flags.drag = false;
        prop.force_model().set_force_flags(flags);
    }

    StateVector initial = prop.current_state_gcrs();

    // Forward 10 minutes, then back; must return to within 0.1 m (integrator
    // tolerance over 20 minutes of propagation).
    prop.propagate_to(epoch + 600.0);
    prop.propagate_to(epoch);

    StateVector final_state = prop.current_state_gcrs();

    // Should return close to initial
    double pos_error = (final_state.position - initial.position).norm();
    TEST_ASSERT_MSG(pos_error < 0.1, "Roundtrip should preserve position within 0.1m");
}

// =============================================================================
// Reset Tests
// =============================================================================

void test_propagator_reset() {
    Epoch epoch1 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef1(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef1(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch1, pos_ecef1, vel_ecef1, BallisticModel(), WeatherConditions::quiet());

    // Propagate forward
    prop.propagate_to(epoch1 + 600.0);

    // Reset to new state
    Epoch epoch2 = Epoch::from_utc(2024, 1, 16, 12, 0, 0);
    Vector3d pos_ecef2(7000000.0, 0.0, 0.0);
    Vector3d vel_ecef2(0.0, 7300.0, 0.0);

    prop.reset(epoch2, StateVector(pos_ecef2, vel_ecef2));

    TEST_ASSERT_MSG(std::abs(prop.current_epoch() - epoch2) < 1e-3,
                "Epoch should be reset");

    StateVector state = prop.current_state_ecef();
    TEST_ASSERT_MSG((state.position - pos_ecef2).norm() < 1.0, "Position should be reset");
}

void test_propagator_reset_clears_integration() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    // Propagate
    prop.propagate_to(epoch + 600.0);

    // Reset to same state
    prop.reset(epoch, StateVector(pos_ecef, vel_ecef));

    // Stats should be reset
    const IntegrationStats& stats = prop.stats();
    TEST_ASSERT_MSG(stats.steps_taken == 0,
                "Stats should be reset");
}

// =============================================================================
// Statistics Tests
// =============================================================================

void test_propagator_stats_after_propagation() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    prop.propagate_to(epoch + 600.0);

    const IntegrationStats& stats = prop.stats();

    TEST_ASSERT_MSG(stats.steps_taken > 0, "Should have taken steps");
    TEST_ASSERT_MSG(stats.function_evaluations >= 6 * stats.steps_taken,
                "RK45 should use at least 6 function evaluations per step");
}

void test_propagator_reset_stats() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    prop.propagate_to(epoch + 600.0);
    prop.reset_stats();

    const IntegrationStats& stats = prop.stats();

    TEST_ASSERT_MSG(stats.steps_taken == 0, "Stats should be reset");
}

// =============================================================================
// Factory Function Tests
// =============================================================================

void test_create_rk45_propagator_default() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    auto prop = create_rk45_propagator(epoch, pos_ecef, vel_ecef);

    TEST_ASSERT_MSG(prop != nullptr, "Should create propagator");
    TEST_ASSERT_MSG(std::abs(prop->current_epoch() - epoch) < 1e-3,
                "Epoch should match");
}

void test_create_rk45_propagator_with_options() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);
    BallisticModel ballistic(1000.0, 10.0, 2.2);
    WeatherConditions weather = WeatherConditions::active();

    auto prop = create_rk45_propagator(epoch, pos_ecef, vel_ecef, ballistic, weather, false);

    TEST_ASSERT_MSG(prop != nullptr, "Should create propagator");
    TEST_ASSERT_MSG(!prop->force_model().third_body().sun_enabled(), "Third body should be disabled");
}

// =============================================================================
// Convenience Propagate Function Tests
// =============================================================================

void test_propagate_convenience_function() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);
    Epoch target = epoch + 60.0;

    PropagationResult result = propagate(epoch, pos_ecef, vel_ecef, target);

    TEST_ASSERT_MSG(result.position.norm() > 6700000.0, "Position should be valid");
    TEST_ASSERT_MSG(result.velocity.norm() > 7000.0, "Velocity should be valid");
}

void test_propagate_with_ballistic() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);
    BallisticModel ballistic(1000.0, 10.0, 2.2);
    Epoch target = epoch + 60.0;

    PropagationResult result = propagate(epoch, pos_ecef, vel_ecef, target,
                                         ballistic, WeatherConditions::quiet());

    TEST_ASSERT_MSG(result.position.norm() > 6700000.0, "Position should be valid");
}

// =============================================================================
// Energy Conservation Tests
// =============================================================================

void test_propagator_energy_conservation_keplerian() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // 400 km; the inertial circular speed is passed as an Earth-fixed velocity,
    // so the orbit is eccentric (e ~ 0.13) -- irrelevant for energy conservation.
    double r = constants::earth::RADIUS_EQUATORIAL + 400000.0;
    double v = std::sqrt(constants::earth::GM / r);

    Vector3d pos_ecef(r, 0.0, 0.0);
    Vector3d vel_ecef(0.0, v, 0.0);

    // Keplerian only (no J2, no drag, no third body, no tides, no relativistic).
    // Specific energy must be conserved to 1e-9 relative over 1 hour
    // (integrator tolerance; a real drift would be far larger).
    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), false);
    {
        forces::ForceComponentFlags flags = prop.force_model().force_flags();
        flags.drag = false;
        flags.solid_earth_tides = false;
        flags.relativistic = false;
        prop.force_model().set_force_flags(flags);
    }
    prop.force_model().set_gravity_max_degree(0);  // Point mass only

    StateVector initial = prop.current_state_gcrs();
    double E0 = 0.5 * initial.velocity.squaredNorm() - constants::earth::GM / initial.position.norm();

    // Propagate 1 hour
    prop.propagate_to(epoch + 3600.0);

    StateVector final_state = prop.current_state_gcrs();
    double E1 = 0.5 * final_state.velocity.squaredNorm() - constants::earth::GM / final_state.position.norm();

    double relative_error = std::abs((E1 - E0) / E0);
    TEST_ASSERT_MSG(relative_error < 1e-9, "Energy should be conserved for Keplerian orbit");
}

// =============================================================================
// ECEF/ECI Coordinate Handling Tests
// =============================================================================

void test_propagator_ecef_input_output() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    PropagationResult result = prop(epoch + 60.0);

    // Result should be ECEF (position on rotating Earth)
    // Velocity should include Earth rotation effect
    TEST_ASSERT_MSG(result.position.norm() > 6700000.0, "ECEF position valid");
}

void test_propagator_eci_internal_state() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(), WeatherConditions::quiet());

    // Internal state is ECI
    StateVector state_eci = prop.current_state_gcrs();
    StateVector state_ecef = prop.current_state_ecef();

    // Magnitudes should match (rotation doesn't change magnitude)
    double r_eci = state_eci.position.norm();
    double r_ecef = state_ecef.position.norm();
    TEST_ASSERT_MSG(std::abs(r_eci - r_ecef) < 1.0, "Position magnitude should match");
}

// =============================================================================
// Long Duration Propagation Tests
// =============================================================================

void test_propagator_one_hour_leo() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // Sentinel-like orbit at 814.5 km — account for Earth rotation in ECEF velocity
    double r = constants::earth::RADIUS_EQUATORIAL + 814500.0;
    double v_eci = std::sqrt(constants::earth::GM / r);
    double omega_earth = 7.2921150e-5;  // rad/s
    double v_ecef = v_eci - omega_earth * r;

    Vector3d pos_ecef(r, 0.0, 0.0);
    Vector3d vel_ecef(0.0, v_ecef, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), true);

    PropagationResult result = prop(epoch + 3600.0);

    // Position should remain in reasonable orbit range (J2 causes altitude oscillation ±20 km)
    double r_final = result.position.norm();
    TEST_ASSERT_MSG(r_final > constants::earth::RADIUS_EQUATORIAL + 790000.0 && r_final < constants::earth::RADIUS_EQUATORIAL + 840000.0,
                "Orbit altitude should be maintained");
}

void test_propagator_multiple_sequential_propagations() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos_ecef(6778137.0, 0.0, 0.0);
    Vector3d vel_ecef(0.0, 7500.0, 0.0);

    RK45Propagator prop(epoch, pos_ecef, vel_ecef, BallisticModel(),
                        WeatherConditions::quiet(), false);
    {
        forces::ForceComponentFlags flags = prop.force_model().force_flags();
        flags.drag = false;
        prop.force_model().set_force_flags(flags);
    }

    // Propagate in 10-minute increments
    for (int i = 1; i <= 6; ++i) {
        prop.propagate_to(epoch + i * 600.0);
    }

    // Should have propagated 1 hour total
    TEST_ASSERT_MSG(std::abs(prop.current_epoch() - (epoch + 3600.0)) < 1e-6,
                "Should reach target time");

    StateVector state = prop.current_state_gcrs();
    TEST_ASSERT_MSG(state.position.norm() > 6700000.0, "Position should be valid");
}

// =============================================================================
// Drag Effect Tests
// =============================================================================

void test_propagator_drag_effect_on_orbit() {
    Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

    // NOTE: no DragEnvironment is attached, so drag is effectively off in both
    // runs (setting flags.drag without one computes zero drag). The cell only
    // checks that the "drag" run ends no more than 1 km above the drag-free
    // run; it does not measure decay.
    double r = constants::earth::RADIUS_EQUATORIAL + 300000.0;  // 300 km - significant drag
    double v = std::sqrt(constants::earth::GM / r);

    Vector3d pos_ecef(r, 0.0, 0.0);
    Vector3d vel_ecef(0.0, v, 0.0);

    BallisticModel ballistic(100.0, 10.0, 2.2);  // High area-to-mass ratio

    // Propagate with drag
    RK45Propagator prop_drag(epoch, pos_ecef, vel_ecef, ballistic,
                              WeatherConditions::active(), false);
    {
        forces::ForceComponentFlags flags = prop_drag.force_model().force_flags();
        flags.drag = true;
        prop_drag.force_model().set_force_flags(flags);
    }

    // Propagate without drag
    RK45Propagator prop_no_drag(epoch, pos_ecef, vel_ecef, ballistic,
                                 WeatherConditions::quiet(), false);
    {
        forces::ForceComponentFlags flags = prop_no_drag.force_model().force_flags();
        flags.drag = false;
        prop_no_drag.force_model().set_force_flags(flags);
    }

    // Propagate 1 hour
    prop_drag.propagate_to(epoch + 3600.0);
    prop_no_drag.propagate_to(epoch + 3600.0);

    double r_drag = prop_drag.current_state_gcrs().position.norm();
    double r_no_drag = prop_no_drag.current_state_gcrs().position.norm();

    TEST_ASSERT_MSG(r_drag < r_no_drag + 1000.0, "Drag should cause orbit to decay or stay similar");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Runs without an Earth orientation table on purpose (synthetic or
    // self-consistent checks; pinned values assume zero UT1-UTC/polar motion).
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    TestRunner construction_tests("RK45Propagator Construction");
    construction_tests.add_test("Default construction", test_propagator_construction_default);
    construction_tests.add_test("With ballistic model", test_propagator_construction_with_ballistic);
    construction_tests.add_test("Third body enabled", test_propagator_construction_third_body_enabled);
    construction_tests.add_test("Third body disabled", test_propagator_construction_third_body_disabled);
    construction_tests.add_test("With config", test_propagator_construction_with_config);

    TestRunner state_tests("State Access");
    state_tests.add_test("Current epoch", test_propagator_current_epoch);
    state_tests.add_test("Current state ECI", test_propagator_current_state_eci);
    state_tests.add_test("Current state ECEF", test_propagator_current_state_ecef);
    state_tests.add_test("Force model access", test_propagator_force_model_access);

    TestRunner propagation_tests("Basic Propagation");
    propagation_tests.add_test("Zero time propagation", test_propagator_propagate_zero_time);
    propagation_tests.add_test("Short time propagation", test_propagator_propagate_short_time);
    propagation_tests.add_test("One orbit propagation", test_propagator_propagate_one_orbit);
    propagation_tests.add_test("Call operator", test_propagator_call_operator);

    TestRunner direction_tests("Forward/Backward Propagation");
    direction_tests.add_test("Forward propagation", test_propagator_forward_propagation);
    direction_tests.add_test("Backward propagation", test_propagator_backward_propagation);
    direction_tests.add_test("Roundtrip propagation", test_propagator_roundtrip);

    TestRunner reset_tests("Reset");
    reset_tests.add_test("Reset to new state", test_propagator_reset);
    reset_tests.add_test("Reset clears integration", test_propagator_reset_clears_integration);

    TestRunner stats_tests("Statistics");
    stats_tests.add_test("Stats after propagation", test_propagator_stats_after_propagation);
    stats_tests.add_test("Reset stats", test_propagator_reset_stats);

    TestRunner factory_tests("Factory Functions");
    factory_tests.add_test("create_rk45_propagator default", test_create_rk45_propagator_default);
    factory_tests.add_test("create_rk45_propagator with options", test_create_rk45_propagator_with_options);

    TestRunner convenience_tests("Convenience Functions");
    convenience_tests.add_test("propagate function", test_propagate_convenience_function);
    convenience_tests.add_test("propagate with ballistic", test_propagate_with_ballistic);

    TestRunner energy_tests("Energy Conservation");
    energy_tests.add_test("Keplerian energy conservation", test_propagator_energy_conservation_keplerian);

    TestRunner coord_tests("Coordinate Handling");
    coord_tests.add_test("ECEF input/output", test_propagator_ecef_input_output);
    coord_tests.add_test("ECI internal state", test_propagator_eci_internal_state);

    TestRunner duration_tests("Long Duration");
    duration_tests.add_test("One hour LEO", test_propagator_one_hour_leo);
    duration_tests.add_test("Sequential propagations", test_propagator_multiple_sequential_propagations);

    TestRunner drag_tests("Drag Effects");
    drag_tests.add_test("Drag effect on orbit", test_propagator_drag_effect_on_orbit);

    // Run all tests
    construction_tests.run();
    state_tests.run();
    propagation_tests.run();
    direction_tests.run();
    reset_tests.run();
    stats_tests.run();
    factory_tests.run();
    convenience_tests.run();
    energy_tests.run();
    coord_tests.run();
    duration_tests.run();
    drag_tests.run();

    return print_final_summary();
}
