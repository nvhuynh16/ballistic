// =============================================================================
// Epoch-label drift regression tests
// =============================================================================
//
// Building each step's epoch label from the previous label accumulates any
// rounding in the time representation. With a single-double TT (Terrestrial
// Time) Julian date, whose spacing is ~40.23 µs, this measured −1.41 ms after
// 1 h of 8 s steps (≈10.6 m along-track at 7.5 km/s) and −0.41 s after 184 s of
// 5 ms steps.
//
// Trajectory therefore keeps an exact elapsed-seconds accumulator and builds
// every label as epoch0 + elapsed, so the label error does not accumulate. These
// cells compare each label with epoch0 + (sum of step sizes) done in one addition.
//
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::test;

namespace {

// Tolerance: one spacing of a single-double Julian date (~40.23 µs) plus slack.
// Epoch now stores whole days plus seconds and is far more precise, so this is
// a loose bound; it stays 30× below the millisecond-scale drift guarded against.
constexpr double LABEL_TOL_S = 4.5e-5;

Epoch test_epoch0() {
    return Epoch::from_utc(2026, 1, 15, 12, 0, 0);
}

// Cheap force configuration: label bookkeeping is integrator-side, so the
// force fidelity is irrelevant here.
TrajectoryConfig cheap_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 2;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    return cfg;
}

} // namespace

int main() {
    ballistic::initialize({});

    TestRunner runner("Epoch Label Drift Tests");

    // 1 h of 8 s RK45 steps (accumulated labels drifted −1.41 ms here)
    runner.add_test("rk45_8s_steps_1h", []() {
        TrajectoryConfig cfg = cheap_config();
        cfg.rk45.initial_step_size = 8.0;
        cfg.rk45.max_step_size = 8.0;

        Epoch t0 = test_epoch0();
        Trajectory traj(t0, Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0), cfg);

        for (int i = 0; i < 450; ++i) {
            (void)traj.step();
        }

        Epoch expected = t0 + 450 * 8.0;
        TEST_ASSERT_NEAR(traj.epoch() - expected, 0.0, LABEL_TOL_S);

        // Labels in history must agree with the same rule at every step
        Epoch mid_expected = t0 + 225 * 8.0;
        TEST_ASSERT_NEAR(traj.history_at(225).epoch - mid_expected, 0.0, LABEL_TOL_S);
    });

    // 184 s of 5 ms symplectic steps (accumulated labels drifted −0.41 s here)
    runner.add_test("symplectic_5ms_steps_184s", []() {
        TrajectoryConfig cfg = cheap_config();
        cfg.integrator = IntegratorType::YOSHIDA6;
        cfg.symplectic.step_size = 0.005;

        Epoch t0 = test_epoch0();
        Trajectory traj(t0, Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0), cfg);

        constexpr int n_steps = 36800;
        for (int i = 0; i < n_steps; ++i) {
            (void)traj.step();
        }

        Epoch expected = t0 + n_steps * 0.005;
        TEST_ASSERT_NEAR(traj.epoch() - expected, 0.0, LABEL_TOL_S);
    });

    // Backward stepping accumulates through the history front — same rule
    runner.add_test("rk45_backward_steps", []() {
        TrajectoryConfig cfg = cheap_config();
        cfg.rk45.initial_step_size = 8.0;
        cfg.rk45.max_step_size = 8.0;

        Epoch t0 = test_epoch0();
        Trajectory traj(t0, Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0), cfg);

        for (int i = 0; i < 100; ++i) {
            (void)traj.step_backward();
        }

        Epoch expected_front = t0 + (-100 * 8.0);
        TEST_ASSERT_NEAR(traj.history_at(0).epoch - expected_front, 0.0, LABEL_TOL_S);
        // Forward end of history is still the initial state
        TEST_ASSERT_NEAR(traj.epoch() - t0, 0.0, 1e-9);
    });

    // propagate() labels exactly at the target; subsequent steps stay exact
    runner.add_test("propagate_then_step_consistency", []() {
        TrajectoryConfig cfg = cheap_config();
        cfg.rk45.initial_step_size = 8.0;
        cfg.rk45.max_step_size = 8.0;

        Epoch t0 = test_epoch0();
        Trajectory traj(t0, Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0), cfg);

        (void)traj.propagate(t0 + 600.0);
        TEST_ASSERT_NEAR(traj.epoch() - (t0 + 600.0), 0.0, 1e-9);

        // After propagate() the adaptive step size ramps back up (landing on the
        // target can force a short final step), so accumulate the actual steps
        // taken rather than assuming 8 s each.
        Epoch resume = traj.epoch();
        double total = 0.0;
        for (int i = 0; i < 10; ++i) {
            total += traj.step().step_size_taken;
        }
        Epoch expected = resume + total;
        TEST_ASSERT_NEAR(traj.epoch() - expected, 0.0, LABEL_TOL_S);
    });

    // rewind() must preserve the surviving history entries (failure mode guarded:
    // clearing the history and refilling it with default-constructed entries)
    runner.add_test("rewind_preserves_history", []() {
        TrajectoryConfig cfg = cheap_config();
        cfg.rk45.initial_step_size = 8.0;
        cfg.rk45.max_step_size = 8.0;

        Epoch t0 = test_epoch0();
        Trajectory traj(t0, Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 7500.0, 0.0), cfg);

        for (int i = 0; i < 5; ++i) {
            (void)traj.step();
        }

        // Snapshot surviving entries before rewinding
        std::vector<Epoch> epochs;
        std::vector<Vector3d> positions;
        for (size_t i = 0; i <= 3; ++i) {
            epochs.push_back(traj.history_at(i).epoch);
            positions.push_back(traj.history_at(i).position);
        }

        TEST_ASSERT(traj.rewind(2));
        TEST_ASSERT(traj.history_size() == 4);

        for (size_t i = 0; i <= 3; ++i) {
            TEST_ASSERT_NEAR(traj.history_at(i).epoch - epochs[i], 0.0, 1e-12);
            TEST_ASSERT_NEAR((traj.history_at(i).position - positions[i]).norm(), 0.0, 1e-9);
        }
        TEST_ASSERT_NEAR(traj.epoch() - epochs[3], 0.0, 1e-9);

        // Stepping after rewind continues on the exact-label rule
        for (int i = 0; i < 2; ++i) {
            (void)traj.step();
        }
        Epoch expected = t0 + 5 * 8.0;
        TEST_ASSERT_NEAR(traj.epoch() - expected, 0.0, LABEL_TOL_S);
    });

    runner.run();
    return print_final_summary();
}
