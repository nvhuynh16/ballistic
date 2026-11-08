// Unit tests for the Trajectory class: construction, propagation, the call
// operator, the default configuration, reset and the factory function. Synthetic
// 400 km circular orbit; the checks are coarse sanity bounds (still in orbit,
// orbital speed), not accuracy claims. No data files.
//
// Positions and velocities are Earth-fixed (ECEF/ITRF, rotating with the Earth).

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"

using namespace ballistic;
using namespace ballistic::test;

// Test basic Trajectory construction and propagation
void test_trajectory_basic() {
    // Initial state (typical LEO)
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);  // ~400 km altitude
    // Circular orbit: inertial speed 7668.55 m/s minus the Earth's rotation
    // (omega x r = 494.27 m/s here), because the velocity is Earth-fixed.
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    // Create trajectory with default config
    Trajectory traj(t0, pos, vel);

    // Check initial state
    auto state = traj.state_ecef();
    TEST_ASSERT_NEAR(state.position.x(), pos.x(), 1.0);
    TEST_ASSERT_NEAR(state.position.y(), pos.y(), 1.0);
    TEST_ASSERT_NEAR(state.position.z(), pos.z(), 1.0);

    // Check epoch
    TEST_ASSERT_NEAR(traj.epoch() - t0, 0.0, 1e-9);
}

// Test Trajectory propagation
void test_trajectory_propagation() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    Trajectory traj(t0, pos, vel);

    // Propagate 10 minutes
    Epoch t1 = t0 + 600.0;
    auto result = traj.propagate(t1);

    // Position should have changed
    double pos_diff = (result.position - pos).norm();
    TEST_ASSERT_MSG(pos_diff > 1000.0, "Position changed after propagation (>1km)");

    // Velocity magnitude should be in orbital range (5-12 km/s for LEO)
    double vel_mag = result.velocity.norm();
    TEST_ASSERT_MSG(vel_mag > 5000.0 && vel_mag < 12000.0, "Velocity in orbital range");

    // Position should stay in reasonable orbital range
    double r = result.position.norm();
    TEST_ASSERT_MSG(r > constants::earth::RADIUS_EQUATORIAL && r < 1e8, "Position in orbital range");
}

// Test Trajectory call operator
void test_trajectory_call_operator() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    Trajectory traj(t0, pos, vel);
    Epoch t1 = t0 + 300.0;

    // Both should give the same result
    auto result1 = traj.propagate(t1);

    // Reset and use operator()
    traj.reset(t0, pos, vel);
    auto result2 = traj(t1);

    double pos_diff = (result1.position - result2.position).norm();
    TEST_ASSERT_MSG(pos_diff < 1.0, "propagate() and operator() give same result");
}

// Test TrajectoryConfig
void test_trajectory_config() {
    TrajectoryConfig config;

    // Check defaults
    TEST_ASSERT_MSG(config.gravity_degree == 70, "Default gravity degree is 70");
    TEST_ASSERT_MSG(config.enable_third_body == true, "Third body enabled by default");
    TEST_ASSERT_MSG(static_cast<bool>(config.drag_env) == false, "Drag disabled by default (drag_env is null)");
    TEST_ASSERT_MSG(!config.srp.has_value(), "SRP disabled by default");
    TEST_ASSERT_MSG(config.enable_relativistic == true, "Relativistic enabled by default");
}

// Test Trajectory with custom config
void test_trajectory_custom_config() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    // Custom config with custom gravity degree + third_body off.
    TrajectoryConfig config;
    config.gravity_degree = 36;
    config.enable_third_body = false;

    Trajectory traj(t0, pos, vel, config);

    TEST_ASSERT_MSG(traj.config().gravity_degree == 36, "Custom gravity degree applied");
    TEST_ASSERT_MSG(traj.config().enable_third_body == false, "Custom third body setting applied");
}

// Test force model access
void test_trajectory_force_model() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    Trajectory traj(t0, pos, vel);

    // The force model is reachable and has Earth gravity on.

    auto& fm = traj.force_model();
    TEST_ASSERT_MSG(fm.force_flags().earth_gravity, "Force model defaults have earth gravity enabled");
}

// Test Trajectory reset
void test_trajectory_reset() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    Trajectory traj(t0, pos, vel);

    // Propagate forward
    traj.propagate(t0 + 600.0);

    // Reset to new state
    Vector3d new_pos(7000000.0, 0.0, 0.0);
    Vector3d new_vel(0.0, 7500.0, 0.0);
    Epoch new_t0 = t0 + 3600.0;

    traj.reset(new_t0, new_pos, new_vel);

    auto state = traj.state_ecef();
    TEST_ASSERT_NEAR(state.position.x(), new_pos.x(), 1.0);
    TEST_ASSERT_NEAR(traj.epoch() - new_t0, 0.0, 1e-9);
}

// Test create_trajectory factory function
void test_create_trajectory() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    auto traj = create_trajectory(t0, pos, vel);

    TEST_ASSERT_MSG(traj != nullptr, "Factory creates valid trajectory");
    TEST_ASSERT_NEAR(traj->epoch() - t0, 0.0, 1e-9);
}

// Test Trajectory 1-hour propagation returns reasonable results
void test_trajectory_one_hour() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    Vector3d vel(0.0, 7668.55 - 494.27, 0.0);

    Trajectory traj(t0, pos, vel);

    // Propagate 1 hour
    Epoch t1 = t0 + 3600.0;
    auto result = traj.propagate(t1);

    // Check position is reasonable (still in orbit)
    double r = result.position.norm();
    TEST_ASSERT_MSG(r > constants::earth::RADIUS_EQUATORIAL, "Position above Earth surface");
    TEST_ASSERT_MSG(r < 10000000.0, "Position within reasonable altitude");

    // Check velocity is reasonable
    double v = result.velocity.norm();
    TEST_ASSERT_MSG(v > 5000.0, "Velocity above minimum orbital");
    TEST_ASSERT_MSG(v < 12000.0, "Velocity below escape velocity");

    std::cout << "  Final position: " << result.position.transpose() << " m\n";
    std::cout << "  Final velocity: " << result.velocity.transpose() << " m/s\n";
    std::cout << "  Altitude: " << (r - constants::earth::RADIUS_EQUATORIAL) / 1000.0 << " km\n";
}

// Main entry point
// Restarting from a recorded node continues the same orbit. The Earth-fixed
// node converts back to the inertial state the run had there (the GCRS <->
// ITRF conversion is a function of the epoch alone; it used to hold the
// precession-nutation matrix for up to 10 minutes and round the Earth
// rotation angle to a ~40 us grid, so the round trip missed by up to ~1 cm and
// a restarted run drifted 1-2 cm from the original). Started with the step
// size the original run was about to take, the restarted run repeats its
// steps: same path to < 1e-5 m (measured 4e-7 m; the step sizes differ by
// ~1e-7 s because they respond to rounding). Started with the default first
// step, the steps differ and so does the truncation error: measured ~1 mm
// after an hour (documented on Trajectory).
void test_trajectory_restart_from_node() {
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    Vector3d pos(6778137.0, 0.0, 0.0);
    // Inclined circular orbit, Earth-fixed velocity (inertial 7669 m/s minus
    // omega x r = 494.3 m/s along y)
    Vector3d vel(0.0, 7669.0 * 0.6 - 494.3, 7669.0 * 0.8);
    TrajectoryConfig cfg;
    cfg.ocean_tide_degree = 0;
    Trajectory a(InitialState::from_itrf(t0, pos, vel), cfg);
    std::vector<StateVector> gcrs{a.state_gcrs()};
    std::vector<double> next{cfg.rk45.initial_step_size};
    while (a.epoch() - t0 < 1200.0) {
        const StepResult s = a.step();
        gcrs.push_back(a.state_gcrs());
        next.push_back(s.next_step_size);
    }
    const auto& H = a.history();
    const size_t k = H.size() / 3;

    // The node converts back to the run's inertial state
    Trajectory b(InitialState::from_itrf(H[k].epoch, H[k].position, H[k].velocity, cfg.itrf), cfg);
    TEST_ASSERT((b.state_gcrs().position - gcrs[k].position).norm() < 1e-7);
    TEST_ASSERT((b.state_gcrs().velocity - gcrs[k].velocity).norm() < 1e-10);

    // ... and, with the same next step, repeats the run's path
    TrajectoryConfig c2 = cfg;
    c2.rk45.initial_step_size = next[k];
    Trajectory c(InitialState::from_itrf(H[k].epoch, H[k].position, H[k].velocity, cfg.itrf), c2);
    double worst = 0.0;
    for (size_t j = k + 1; j < H.size(); ++j) {
        (void)c.step();
        const double dt = c.epoch() - H[j].epoch;   // step sizes differ by rounding
        TEST_ASSERT(std::abs(dt) < 1e-5);
        worst = std::max(worst, (c.state_ecef().position - H[j].position - H[j].velocity * dt).norm());
    }
    TEST_ASSERT(worst < 1e-5);
}

int main() {
    // Runs without an Earth orientation table on purpose (synthetic or
    // self-consistent checks; pinned values assume zero UT1-UTC/polar motion).
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    TestRunner runner("Trajectory API Unit Tests");

    runner.add_test("trajectory_basic", test_trajectory_basic);
    runner.add_test("trajectory_propagation", test_trajectory_propagation);
    runner.add_test("trajectory_call_operator", test_trajectory_call_operator);
    runner.add_test("trajectory_config", test_trajectory_config);
    runner.add_test("trajectory_custom_config", test_trajectory_custom_config);
    runner.add_test("trajectory_force_model", test_trajectory_force_model);
    runner.add_test("trajectory_reset", test_trajectory_reset);
    runner.add_test("create_trajectory", test_create_trajectory);
    runner.add_test("trajectory_one_hour", test_trajectory_one_hour);
    runner.add_test("trajectory_restart_from_node", test_trajectory_restart_from_node);

    runner.run();
    return print_final_summary();
}
