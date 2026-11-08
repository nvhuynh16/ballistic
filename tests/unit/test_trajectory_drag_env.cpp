// Unit test: TrajectoryConfig::drag_env wires the propagator's drag through
// the env-based path. Validates that:
//   - Setting drag_env enables flags.drag automatically.
//   - The propagator's ForceModel.drag_environment() returns the same object.
//   - A short propagation runs without crashing through the env path.

#include "../test_utils.hpp"
#include <ballistic/trajectory.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/init.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;

namespace {

std::shared_ptr<const DragEnvironment> make_uniform_env(double rho, double beta) {
    DensityProfile1D density({100.0, 1500.0}, {rho, rho});
    WindProfile1D_ENU wind;  // no wind
    BetaProfile1D b = BetaProfile1D::constant(beta);
    return std::make_shared<const DragEnvironment>(
        std::move(density), std::move(wind), std::move(b));
}

}  // namespace

void test_drag_env_sets_drag_flag() {
    TestRunner runner("DragEnvSetsDragFlag");

    runner.add_test("setting_drag_env_enables_drag_flag", []() {
        ballistic::initialize({});

        TrajectoryConfig config;
        config.drag_env = make_uniform_env(1e-11, 100.0);

        Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        Vector3d pos(6.778137e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);
        Trajectory traj(t0, pos, vel, config);

        // The propagator's force flags should have drag enabled because
        // drag_env was set.
        TEST_ASSERT(traj.force_model().force_flags().drag);
    });

    runner.add_test("force_model_holds_the_env_shared_ptr", []() {
        ballistic::initialize({});

        TrajectoryConfig config;
        auto env = make_uniform_env(1e-11, 100.0);
        config.drag_env = env;

        Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        Vector3d pos(6.778137e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);
        Trajectory traj(t0, pos, vel, config);

        auto held = traj.force_model().drag_environment();
        TEST_ASSERT(held);
        TEST_ASSERT(held.get() == env.get());
    });

    runner.run();
}

void test_drag_env_short_propagation() {
    TestRunner runner("DragEnvShortPropagation");

    runner.add_test("propagates_one_minute_without_crash", []() {
        ballistic::initialize({});

        TrajectoryConfig config;
        config.drag_env = make_uniform_env(1e-11, 100.0);
        config.gravity_degree = 4;  // keep test fast

        Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        Vector3d pos(6.778137e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.propagate(t0 + 60.0);
        // Should have moved roughly velocity*time ~ 450 km in 60 s; the
        // 100-1000 km bounds only check that the propagation ran.
        const double dist = (result.position - pos).norm();
        TEST_ASSERT(dist > 100000.0);  // > 100 km
        TEST_ASSERT(dist < 1000000.0); // < 1000 km
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Trajectory + DragEnvironment Integration Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_drag_env_sets_drag_flag();
    test_drag_env_short_propagation();

    return print_final_summary();
}
