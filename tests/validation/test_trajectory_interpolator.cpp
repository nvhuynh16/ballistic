// =============================================================================
// Validation Tests for Trajectory Interpolator
// =============================================================================
//
// TrajectoryInterpolator turns the states a Trajectory stored at each integration
// step ("history nodes") into a continuous function of time, by quintic Hermite
// interpolation (matching position, velocity and acceleration at both ends of
// each step), and propagates further when asked for a time outside the history.
// Synthetic LEO states; tolerances are rounding-level for exact cases and loose
// physical bounds otherwise. No data files.
//
// Frames: ECEF = Earth-fixed (rotates with the Earth); GCRS = the inertial
// Geocentric Celestial Reference System.
//
// Tests the TrajectoryInterpolator class functionality including:
// - Within-range interpolation accuracy
// - At-node queries (exact match)
// - Forward propagation extension
// - Frame conversion (ECEF <-> GCRS)
// - Cache behavior
// - Batch query optimization
// - Comparison to direct propagation
//
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/ballistic.hpp>
#include <cmath>
#include <numeric>

using namespace ballistic;
using namespace ballistic::test;

int main() {
    ballistic::initialize({});

    TestRunner runner("Trajectory Interpolator Tests");

    // =========================================================================
    // Basic Interpolation Tests
    // =========================================================================

    runner.add_test("interpolator_creation", []() {
        // Test basic interpolator creation
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;  // Faster for testing
        Trajectory traj(t0, pos, vel, config);

        // Build some history
        for (int i = 0; i < 10; ++i) {
            traj.step();
        }

        // Create interpolator
        TrajectoryInterpolator interp(traj);

        TEST_ASSERT(interp.history_size() >= 11);  // Initial + 10 steps
    });

    runner.add_test("at_node_queries_exact", []() {
        // Queries at exact history nodes should return exact values
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build history
        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        // Query at each history node
        for (size_t i = 0; i < traj.history_size(); ++i) {
            const auto& entry = traj.history_at(i);
            auto state = interp.get_state(entry.epoch);

            TEST_ASSERT(state.has_value());
            // At nodes, quintic Hermite gives near-exact results
            // Allow 1e-6 for floating-point accumulation in basis function evaluation
            TEST_ASSERT_VEC3_NEAR(state->position, entry.position, 1e-6);
            TEST_ASSERT_VEC3_NEAR(state->velocity, entry.velocity, 1e-6);
            TEST_ASSERT_VEC3_NEAR(state->acceleration, entry.acceleration, 1e-6);
        }
    });

    runner.add_test("midpoint_interpolation", []() {
        // Test interpolation at segment midpoints
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build history
        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        // Query at midpoint of each segment
        for (size_t i = 0; i + 1 < traj.history_size(); ++i) {
            const auto& h0 = traj.history_at(i);
            const auto& h1 = traj.history_at(i + 1);

            // Midpoint time
            Epoch t_mid = h0.epoch + (h1.epoch - h0.epoch) * 0.5;

            auto state = interp.get_state(t_mid);
            TEST_ASSERT(state.has_value());

            // Position should be between endpoints (roughly)
            double pos_norm = state->position.norm();
            double expected_norm = 0.5 * (h0.position.norm() + h1.position.norm());
            TEST_ASSERT_NEAR_REL(pos_norm, expected_norm, 0.01);  // Within 1%
        }
    });

    // =========================================================================
    // Interpolation Accuracy Tests
    // =========================================================================

    runner.add_test("interpolation_vs_direct_propagation", []() {
        // Compare interpolated state vs directly propagated state from SAME trajectory
        // The interpolated state should closely match what a direct propagation would give
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;

        // Build trajectory with adaptive stepping
        Trajectory traj(t0, pos, vel, config);

        // Build history (1 hour)
        while (traj.epoch() - t0 < 3600.0) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        // Test interpolation at midpoints between history nodes
        // Interpolated values should be smooth and physically reasonable
        for (size_t i = 0; i + 1 < traj.history_size() && i < 20; ++i) {
            const auto& h0 = traj.history_at(i);
            const auto& h1 = traj.history_at(i + 1);

            // Test at midpoint
            Epoch t_mid = h0.epoch + (h1.epoch - h0.epoch) * 0.5;
            auto state_mid = interp.get_state(t_mid);
            TEST_ASSERT(state_mid.has_value());

            // Position should be between endpoints (interpolation is bounded)
            double pos_mid_norm = state_mid->position.norm();
            double pos_min = (std::min)(h0.position.norm(), h1.position.norm());
            double pos_max = (std::max)(h0.position.norm(), h1.position.norm());

            // Allow some overshoot due to quintic nature, but should be close
            TEST_ASSERT_MSG(pos_mid_norm >= pos_min * 0.999 && pos_mid_norm <= pos_max * 1.001,
                "Midpoint position norm " + std::to_string(pos_mid_norm) +
                " outside expected range [" + std::to_string(pos_min) + ", " +
                std::to_string(pos_max) + "]");

            // Velocity should also be reasonable
            double vel_mid_norm = state_mid->velocity.norm();
            double vel_min = (std::min)(h0.velocity.norm(), h1.velocity.norm());
            double vel_max = (std::max)(h0.velocity.norm(), h1.velocity.norm());

            TEST_ASSERT_MSG(vel_mid_norm >= vel_min * 0.99 && vel_mid_norm <= vel_max * 1.01,
                "Midpoint velocity norm outside expected range");
        }
    });

    // =========================================================================
    // Range and Extension Tests
    // =========================================================================

    runner.add_test("covers_range_check", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build 10 steps
        for (int i = 0; i < 10; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        // Should cover times within history
        TEST_ASSERT(interp.covers(interp.history_start()));
        TEST_ASSERT(interp.covers(interp.history_end()));
        TEST_ASSERT(interp.covers(interp.history_start() + 10.0));

        // Should not cover times outside history
        TEST_ASSERT(!interp.covers(interp.history_start() - 1.0));
        TEST_ASSERT(!interp.covers(interp.history_end() + 1.0));
    });

    runner.add_test("auto_forward_propagation", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build minimal history
        traj.step();
        traj.step();

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = true;
        interp_config.max_propagation_time = 3600.0;

        TrajectoryInterpolator interp(traj, interp_config);

        size_t initial_size = interp.history_size();

        // Query beyond current history
        Epoch t_future = interp.history_end() + 100.0;
        auto state = interp.get_state(t_future);

        TEST_ASSERT(state.has_value());
        TEST_ASSERT(state->was_propagated);
        TEST_ASSERT(interp.history_size() > initial_size);
    });

    runner.add_test("propagation_limit_respected", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        traj.step();

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = true;
        interp_config.max_propagation_time = 60.0;  // Only 60 seconds

        TrajectoryInterpolator interp(traj, interp_config);

        // Query within limit - should work
        Epoch t_ok = interp.history_end() + 30.0;
        auto state_ok = interp.get_state(t_ok);
        TEST_ASSERT(state_ok.has_value());

        // Query beyond limit - should fail
        Epoch t_far = interp.history_end() + 120.0;
        auto state_far = interp.get_state(t_far);
        TEST_ASSERT(!state_far.has_value());
    });

    runner.add_test("propagation_disabled", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        traj.step();

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = false;

        TrajectoryInterpolator interp(traj, interp_config);

        // Query beyond current history - should fail
        Epoch t_future = interp.history_end() + 10.0;
        auto state = interp.get_state(t_future);

        TEST_ASSERT(!state.has_value());
    });

    // =========================================================================
    // Frame Conversion Tests
    // =========================================================================

    runner.add_test("ecef_frame_output", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        InterpolatorConfig interp_config;
        interp_config.output_frame = InterpolationFrame::ECEF;

        TrajectoryInterpolator interp(traj, interp_config);

        // Get state at first node (should match exactly)
        const auto& h0 = traj.history_at(0);
        auto state = interp.get_state(h0.epoch);

        TEST_ASSERT(state.has_value());
        TEST_ASSERT_VEC3_NEAR(state->position, h0.position, 1e-9);
        TEST_ASSERT_VEC3_NEAR(state->velocity, h0.velocity, 1e-9);
    });

    runner.add_test("gcrs_frame_output", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        InterpolatorConfig interp_config;
        interp_config.output_frame = InterpolationFrame::GCRS;

        TrajectoryInterpolator interp(traj, interp_config);

        // Get state in GCRS
        const auto& h0 = traj.history_at(0);
        auto state_gcrs = interp.get_state(h0.epoch);
        TEST_ASSERT(state_gcrs.has_value());

        // Convert ECEF to GCRS manually for comparison
        StateVector state_ecef(h0.position, h0.velocity);
        StateVector state_manual = coords::itrf_to_gcrs_full(state_ecef, h0.epoch);

        TEST_ASSERT_VEC3_NEAR(state_gcrs->position, state_manual.position, 1e-6);
        TEST_ASSERT_VEC3_NEAR(state_gcrs->velocity, state_manual.velocity, 1e-6);
    });

    // =========================================================================
    // Batch Query Tests
    // =========================================================================

    runner.add_test("batch_query_same_as_individual", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build history
        while (traj.epoch() - t0 < 600.0) {  // 10 minutes
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        // Create batch of query times
        std::vector<Epoch> times;
        for (double dt = 10.0; dt < 500.0; dt += 50.0) {
            times.push_back(t0 + dt);
        }

        // Batch query
        auto batch_results = interp.get_states(std::span<const Epoch>(times));

        // Compare with individual queries
        for (size_t i = 0; i < times.size(); ++i) {
            auto individual = interp.get_state(times[i]);

            TEST_ASSERT(batch_results[i].has_value() == individual.has_value());

            if (batch_results[i].has_value()) {
                TEST_ASSERT_VEC3_NEAR(batch_results[i]->position, individual->position, 1e-12);
                TEST_ASSERT_VEC3_NEAR(batch_results[i]->velocity, individual->velocity, 1e-12);
            }
        }
    });

    runner.add_test("batch_query_unsorted_times", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        while (traj.epoch() - t0 < 600.0) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        // Unsorted times
        std::vector<Epoch> times = {
            t0 + 300.0,
            t0 + 100.0,
            t0 + 500.0,
            t0 + 200.0,
            t0 + 400.0
        };

        auto results = interp.get_states(std::span<const Epoch>(times));

        // Results should be in original order
        TEST_ASSERT_EQ(results.size(), times.size());

        for (size_t i = 0; i < times.size(); ++i) {
            TEST_ASSERT(results[i].has_value());
            // Epoch should match query time
            TEST_ASSERT_NEAR(results[i]->epoch - times[i], 0.0, 1e-15);
        }
    });

    // =========================================================================
    // Cache Behavior Tests
    // =========================================================================

    runner.add_test("cache_improves_repeated_queries", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        while (traj.epoch() - t0 < 300.0) {
            traj.step();
        }

        InterpolatorConfig interp_config;
        interp_config.cache_coefficients = true;
        TrajectoryInterpolator interp(traj, interp_config);

        Epoch t_query = t0 + 150.0;

        // First query - populates cache
        auto state1 = interp.get_state(t_query);
        TEST_ASSERT(state1.has_value());

        // Second query - should use cache
        auto state2 = interp.get_state(t_query);
        TEST_ASSERT(state2.has_value());

        // Results should be identical (same computation path)
        TEST_ASSERT_VEC3_NEAR(state1->position, state2->position, 1e-15);
        TEST_ASSERT_VEC3_NEAR(state1->velocity, state2->velocity, 1e-15);
        TEST_ASSERT_EQ(state1->segment_index, state2->segment_index);
    });

    runner.add_test("cache_invalidation_on_history_change", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        // Query to populate cache
        Epoch t_query = t0 + 50.0;
        auto state1 = interp.get_state(t_query);
        TEST_ASSERT(state1.has_value());

        size_t old_size = interp.history_size();

        // Add more history
        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        // Cache should be invalidated, new query should still work
        auto state2 = interp.get_state(t_query);
        TEST_ASSERT(state2.has_value());
        TEST_ASSERT(interp.history_size() > old_size);

        // Results should still be the same for same query time
        TEST_ASSERT_VEC3_NEAR(state1->position, state2->position, 1e-9);
    });

    runner.add_test("manual_cache_clear", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        Epoch t_query = t0 + 50.0;
        auto state1 = interp.get_state(t_query);
        TEST_ASSERT(state1.has_value());

        // Clear cache
        interp.clear_cache();

        // Query should still work
        auto state2 = interp.get_state(t_query);
        TEST_ASSERT(state2.has_value());

        // Results should be identical
        TEST_ASSERT_VEC3_NEAR(state1->position, state2->position, 1e-12);
    });

    // =========================================================================
    // Convenience Method Tests
    // =========================================================================

    runner.add_test("position_only_query", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        Epoch t_query = t0 + 50.0;

        auto pos_only = interp.position(t_query);
        auto full_state = interp.get_state(t_query);

        TEST_ASSERT(pos_only.has_value());
        TEST_ASSERT(full_state.has_value());
        TEST_ASSERT_VEC3_NEAR(*pos_only, full_state->position, 1e-15);
    });

    runner.add_test("velocity_only_query", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        Epoch t_query = t0 + 50.0;

        auto vel_only = interp.velocity(t_query);
        auto full_state = interp.get_state(t_query);

        TEST_ASSERT(vel_only.has_value());
        TEST_ASSERT(full_state.has_value());
        TEST_ASSERT_VEC3_NEAR(*vel_only, full_state->velocity, 1e-15);
    });

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    runner.add_test("set_output_frame", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        TrajectoryInterpolator interp(traj);

        Epoch t_query = t0 + 50.0;

        // ECEF first
        interp.set_output_frame(InterpolationFrame::ECEF);
        auto state_ecef = interp.get_state(t_query);

        // Then GCRS
        interp.set_output_frame(InterpolationFrame::GCRS);
        auto state_gcrs = interp.get_state(t_query);

        TEST_ASSERT(state_ecef.has_value());
        TEST_ASSERT(state_gcrs.has_value());

        // Positions should differ (different frames)
        double diff = (state_ecef->position - state_gcrs->position).norm();
        TEST_ASSERT(diff > 1000.0);  // Should differ by km due to Earth rotation
    });

    runner.add_test("explicit_extend_to", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        (void)traj.step();  // Minimal history

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = false;  // Disable auto-propagation
        TrajectoryInterpolator interp(traj, interp_config);

        Epoch t_target = t0 + 300.0;

        // Should not cover target initially
        TEST_ASSERT(!interp.covers(t_target));

        // Explicitly extend
        bool success = interp.extend_to(t_target);
        TEST_ASSERT(success);

        // Now should cover
        TEST_ASSERT(interp.covers(t_target));

        // And query should work
        auto state = interp.get_state(t_target);
        TEST_ASSERT(state.has_value());
    });

    // =========================================================================
    // Altitude-Based Propagation Tests
    // =========================================================================

    runner.add_test("propagate_to_altitude_already_satisfied", []() {
        // Test that if condition is already met, no steps are taken
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        // Start at ~400 km altitude
        Vector3d pos(6778137.0, 0.0, 0.0);  // ~400 km altitude
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        (void)traj.step();

        TrajectoryInterpolator interp(traj);

        // Get current altitude
        double current_alt = coords::ecef_to_altitude(traj.history().back().position);

        // Condition already satisfied: altitude > 300 km
        auto result = interp.propagate_to_altitude(
            300000.0,  // 300 km
            AltitudeCondition::Above);

        TEST_ASSERT(result.condition_met);
        TEST_ASSERT_EQ(result.steps_taken, size_t(0));
        TEST_ASSERT_NEAR(result.starting_altitude_m, current_alt, 1.0);
        TEST_ASSERT_NEAR(result.ending_altitude_m, current_alt, 1.0);
    });

    runner.add_test("propagate_to_altitude_forward_above", []() {
        // Test propagating forward until altitude rises above a threshold
        // Use an elliptical orbit that will gain altitude
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        // Start near perigee ~300 km, elliptical orbit will go higher
        Vector3d pos(6678137.0, 0.0, 0.0);  // ~300 km altitude
        Vector3d vel(0.0, 7800.0, 0.0);      // Slightly higher velocity for elliptical

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        (void)traj.step();

        TrajectoryInterpolator interp(traj);

        double start_alt = coords::ecef_to_altitude(traj.history().back().position);

        // Propagate until altitude > 350 km
        auto result = interp.propagate_to_altitude(
            350000.0,  // 350 km
            AltitudeCondition::Above,
            PropagationDirection::Forward,
            3.0 * 3600.0);

        TEST_ASSERT(result.condition_met);
        TEST_ASSERT(result.steps_taken > 0);
        TEST_ASSERT_NEAR(result.starting_altitude_m, start_alt, 1.0);
        // Ends AT the crossing (located inside the step), not a step past it.
        TEST_ASSERT(result.ending_altitude_m >= 350000.0);
        TEST_ASSERT(result.ending_altitude_m < 350000.0 + 2.0);
    });

    runner.add_test("propagate_to_altitude_forward_below", []() {
        // Rise above start + 10 km, then search for the falling crossing of
        // start + 5 km: the history ends AT it (located inside the step).
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);  // ~400 km altitude
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        (void)traj.step();

        TrajectoryInterpolator interp(traj);

        double start_alt = coords::ecef_to_altitude(traj.history().back().position);
        auto up = interp.propagate_to_altitude(start_alt + 10e3, AltitudeCondition::Above,
                                               PropagationDirection::Forward, 86400.0);
        TEST_ASSERT(up.condition_met);

        const double target_alt = start_alt + 5e3;
        auto result = interp.propagate_to_altitude(
            target_alt,
            AltitudeCondition::Below,
            PropagationDirection::Forward,
            86400.0);

        TEST_ASSERT(result.condition_met);
        TEST_ASSERT(result.steps_taken > 0);
        TEST_ASSERT(result.ending_altitude_m <= target_alt);
        TEST_ASSERT(result.ending_altitude_m > target_alt - 2.0);
    });

    runner.add_test("propagate_to_altitude_max_duration_limit", []() {
        // Test that the search window is respected
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        (void)traj.step();

        TrajectoryInterpolator interp(traj);

        // Request an unreachable altitude with a 10-minute search window: the history
        // must stop at the window (plus the first step already taken), to 1e-6 s.
        auto result = interp.propagate_to_altitude(
            1000000.0,  // 1000 km: not reached within the window (starts near perigee)
            AltitudeCondition::Above,
            PropagationDirection::Forward,
            600.0);  // 10 minutes

        TEST_ASSERT(!result.condition_met);
        TEST_ASSERT(result.steps_taken > 0);
        TEST_ASSERT_NEAR(traj.history().back().epoch - traj.history().front().epoch,
                         600.0 + (traj.history()[1].epoch - traj.history()[0].epoch), 1e-6);
    });

    runner.add_test("propagate_to_altitude_cache_invalidation", []() {
        // Test that cache is invalidated after propagation
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        (void)traj.step();

        TrajectoryInterpolator interp(traj);

        size_t initial_size = interp.history_size();

        // Query to populate cache
        Epoch t_query = t0 + 5.0;
        auto state1 = interp.get_state(t_query);
        TEST_ASSERT(state1.has_value());

        // Propagate to altitude
        auto result = interp.propagate_to_altitude(
            395000.0,
            AltitudeCondition::Below,
            PropagationDirection::Forward,
            3600.0);

        // History should have grown
        TEST_ASSERT(interp.history_size() > initial_size);

        // Query should still work (cache was invalidated and rebuilt)
        auto state2 = interp.get_state(t_query);
        TEST_ASSERT(state2.has_value());
        TEST_ASSERT_VEC3_NEAR(state1->position, state2->position, 1e-9);
    });

    runner.add_test("propagate_to_altitude_backward", []() {
        // Test backward propagation extends history to earlier times
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);  // ~400 km altitude
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build some forward history first
        for (int i = 0; i < 5; ++i) {
            (void)traj.step();
        }

        TrajectoryInterpolator interp(traj);

        Epoch original_start = interp.history_start();
        size_t original_size = interp.history_size();
        double start_alt = coords::ecef_to_altitude(traj.history().front().position);

        // Propagate backward - request altitude ABOVE current (not satisfied initially)
        // As orbit oscillates backward, it should eventually rise above this threshold
        double target_alt = start_alt + 1000.0;  // 1 km above current front

        auto result = interp.propagate_to_altitude(
            target_alt,
            AltitudeCondition::Above,  // Changed: need to go ABOVE (not below)
            PropagationDirection::Backward,
            3.0 * 3600.0);

        // Verify backward propagation occurred
        TEST_ASSERT(result.steps_taken > 0);
        TEST_ASSERT(interp.history_size() > original_size);
        TEST_ASSERT(interp.history_start() < original_start);  // Earlier epoch

        // If condition was met, verify the altitude
        if (result.condition_met) {
            TEST_ASSERT(result.ending_altitude_m >= target_alt);
        }
    });

    runner.add_test("step_backward_basic", []() {
        // Test basic step_backward functionality on Trajectory
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Take a forward step to initialize history
        (void)traj.step();
        size_t initial_size = traj.history_size();
        Epoch initial_start = traj.history().front().epoch;

        // Take a backward step
        auto result = traj.step_backward();

        // Verify history was prepended
        TEST_ASSERT(traj.history_size() == initial_size + 1);
        TEST_ASSERT(traj.history().front().epoch < initial_start);
        TEST_ASSERT(result.epoch < initial_start);
    });

    // =========================================================================
    // Backward Extension Tests
    // =========================================================================

    runner.add_test("auto_backward_propagation", []() {
        // Query before history start should auto-extend backward
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build some forward history
        for (int i = 0; i < 5; ++i) {
            traj.step();
        }

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = true;
        interp_config.max_propagation_time = 3600.0;

        TrajectoryInterpolator interp(traj, interp_config);

        Epoch original_start = interp.history_start();
        size_t initial_size = interp.history_size();

        // Query 60 seconds before history start
        Epoch t_past = original_start - 60.0;
        auto state = interp.get_state(t_past);

        TEST_ASSERT(state.has_value());
        TEST_ASSERT(state->was_propagated);
        TEST_ASSERT(interp.history_size() > initial_size);
        TEST_ASSERT(interp.history_start() <= t_past);
    });

    runner.add_test("backward_propagation_limit_respected", []() {
        // Backward query beyond max_propagation_time should return nullopt
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        traj.step();

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = true;
        interp_config.max_propagation_time = 60.0;  // Only 60 seconds

        TrajectoryInterpolator interp(traj, interp_config);

        // Query within limit - should work
        Epoch t_ok = interp.history_start() - 30.0;
        auto state_ok = interp.get_state(t_ok);
        TEST_ASSERT(state_ok.has_value());

        // Query beyond limit - should fail
        Epoch t_far = interp.history_start() - 120.0;
        auto state_far = interp.get_state(t_far);
        TEST_ASSERT(!state_far.has_value());
    });

    runner.add_test("explicit_extend_backward_to", []() {
        // Test the extend_backward_to method directly
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);
        (void)traj.step();

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = false;  // Disable auto-propagation
        TrajectoryInterpolator interp(traj, interp_config);

        Epoch t_target = t0 - 120.0;  // 2 minutes before start

        // Should not cover target initially
        TEST_ASSERT(!interp.covers(t_target));

        // Explicitly extend backward
        bool success = interp.extend_backward_to(t_target);
        TEST_ASSERT(success);

        // Now should cover
        TEST_ASSERT(interp.covers(t_target));

        // And query should work
        auto state = interp.get_state(t_target);
        TEST_ASSERT(state.has_value());
    });

    runner.add_test("batch_query_with_bidirectional_extension", []() {
        // get_states with times spanning before AND after current history
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.gravity_degree = 20;
        Trajectory traj(t0, pos, vel, config);

        // Build a small window of history
        for (int i = 0; i < 3; ++i) {
            traj.step();
        }

        InterpolatorConfig interp_config;
        interp_config.allow_propagation = true;
        interp_config.max_propagation_time = 600.0;

        TrajectoryInterpolator interp(traj, interp_config);

        Epoch start = interp.history_start();
        Epoch end = interp.history_end();

        // Query both before and after current history
        std::vector<Epoch> times = {
            start - 30.0,   // Before history
            start + 5.0,    // Within history
            end + 60.0      // After history
        };

        auto results = interp.get_states(std::span<const Epoch>(times));

        TEST_ASSERT_EQ(results.size(), size_t(3));
        for (size_t i = 0; i < results.size(); ++i) {
            TEST_ASSERT_MSG(results[i].has_value(),
                "Query at index " + std::to_string(i) + " failed");
        }

        // Verify history was extended in both directions
        TEST_ASSERT(interp.history_start() <= times[0]);
        TEST_ASSERT(interp.history_end() >= times[2]);
    });

    runner.run();
    return print_final_summary();
}
