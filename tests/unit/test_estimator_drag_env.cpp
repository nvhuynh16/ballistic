// Unit test: the empirical-acceleration estimators pick up drag through
// TrajectoryConfig::drag_env, with no separate drag argument. They propagate
// through calibration::detail::run_propagation_arc with the caller's full
// TrajectoryConfig, so setting drag_env on that config is enough. Checks that
// the arc's force model holds the same DragEnvironment, that drag is switched on
// by its presence, and that a 600 s arc with a uniform-density atmosphere runs.

#include "../test_utils.hpp"
#include <ballistic/trajectory.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/density_profile.hpp>
#include <ballistic/atmosphere/wind_profile.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/calibration/detail/run_propagation.hpp>
#include <ballistic/init.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;
using namespace ballistic::calibration;

namespace {

// Constant density at every altitude from 100 to 1500 km, no wind, constant
// ballistic coefficient beta = m / (Cd * A).
std::shared_ptr<const DragEnvironment> make_uniform_env(double rho, double beta) {
    DensityProfile1D density({100.0, 1500.0}, {rho, rho});
    WindProfile1D_ENU wind;
    BetaProfile1D b = BetaProfile1D::constant(beta);
    return std::make_shared<const DragEnvironment>(
        std::move(density), std::move(wind), std::move(b));
}

}  // namespace

void test_run_propagation_arc_picks_up_drag_env() {
    TestRunner runner("RunPropagationArcPicksUpDragEnv");

    runner.add_test("trajectory_config_drag_env_wires_into_force_model", []() {
        ballistic::initialize({});

        TrajectoryConfig cfg;
        cfg.gravity_degree = 4;
        cfg.drag_env = make_uniform_env(1e-11, 100.0);

        Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        Vector3d pos(6.778137e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        auto [traj, ti] = calibration::detail::run_propagation_arc(
            t0, pos, vel, cfg,
            /*ea*/ nullptr,
            /*max_duration_s*/ 60.0,
            /*min_altitude_m*/ 100000.0);

        // The Trajectory's ForceModel should hold the env.
        auto held = traj->force_model().drag_environment();
        TEST_ASSERT(held);
        TEST_ASSERT(held.get() == cfg.drag_env.get());

        // Drag should be enabled implicitly via drag_env.
        TEST_ASSERT(traj->force_model().force_flags().drag);
    });

    runner.add_test("propagation_with_drag_env_completes", []() {
        ballistic::initialize({});

        TrajectoryConfig cfg;
        cfg.gravity_degree = 4;
        cfg.drag_env = make_uniform_env(1e-11, 100.0);

        Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        Vector3d pos(6.778137e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        auto [traj, ti] = calibration::detail::run_propagation_arc(
            t0, pos, vel, cfg, nullptr, 600.0, 100000.0);

        // Should have stepped without throwing; history should be non-trivial.
        TEST_ASSERT(traj->history().size() > 1);
        // Position should have advanced.
        const double dist = (traj->history().back().position - pos).norm();
        TEST_ASSERT(dist > 1000.0);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Estimator DragEnv Path Tests (A13)" << std::endl;
    std::cout << "======================================" << std::endl;

    test_run_propagation_arc_picks_up_drag_env();

    return print_final_summary();
}
