// Integration test for the from_models_along_iterated generator, which
// builds a tabulated drag environment ("env": density, wind and ballistic
// coefficient along the satellite's path, from the NRLMSIS 2.1 / HWM14
// atmosphere models) and refines it because the path itself depends on drag.
//
// Verifies the iteration loop:
//   - Builds an initial env from a drag-free nominal arc.
//   - Re-propagates with the new env's drag, re-samples, compares the
//     new trajectory to the previous.
//   - Stops when max(|p_new - p_prev|) / |p_ref| < tol_relative, or
//     when max_iter is reached.
//
// On a stable LEO arc the drag delta is small and convergence happens
// in 2 iterations (drag-free baseline + one env-driven re-propagation).

#include "../test_utils.hpp"
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/init.hpp>
#include <ballistic/trajectory.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;

namespace {

Epoch t0_test() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

SpaceWeatherHistory quiet_sw(const Epoch& t0, double window_s) {
    return SpaceWeatherHistory::from_constant(
        WeatherConditions::quiet(),
        TimeWindow{t0 - 86400.0, t0 + window_s + 86400.0});
}

TrajectoryConfig nominal_base_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 4;          // light grav for test speed
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    return cfg;
}

}  // namespace

int main() {
    initialize({});
    TestRunner runner("FromModelsAlongIterated");

    runner.add_test("converges_in_two_iterations_on_low_drag_LEO", []() {
        const Epoch t0 = t0_test();
        const double r = constants::earth::RADIUS_EQUATORIAL + 400e3;
        const Vector3d pos(r, 0.0, 0.0);
        const Vector3d vel(0.0, std::sqrt(constants::earth::GM / r), 0.0);
        const double duration_s = 1200.0;  // 20 minutes, ~0.2 orbits

        IterationDiag diag;
        auto env = from_models_along_iterated(
            t0, pos, vel, duration_s,
            nominal_base_config(),
            quiet_sw(t0, duration_s),
            BetaProfile1D::constant(256.4),
            std::vector<double>{350, 380, 400, 420, 450},
            /*max_iter=*/3,
            /*tol_relative=*/0.01,
            /*sample_dt_s=*/60.0,
            &diag);

        TEST_ASSERT_MSG(env.has_drag(),
            "iterated env should be drag-capable");
        TEST_ASSERT_MSG(diag.iterations >= 2,
            "expected at least one re-propagation past the drag-free baseline");
        TEST_ASSERT_MSG(diag.iterations <= 3,
            "respect max_iter cap");
        TEST_ASSERT_MSG(diag.converged,
            "high-altitude quiet-weather LEO should converge inside tol");
        // Drag at 400 km / quiet weather is ~1e-7 m/s², so delta over
        // 20 min is sub-millimeter relative — well inside 1%.
        TEST_ASSERT_MSG(diag.final_traj_delta_m < 100.0,
            "final delta should be modest at 400 km");

        std::cout << "    iterations=" << diag.iterations
                  << " converged=" << diag.converged
                  << " final_delta_m=" << diag.final_traj_delta_m << "\n";
    });

    runner.add_test("rejects_bad_inputs", []() {
        const Epoch t0 = t0_test();
        const Vector3d pos(6.778e6, 0.0, 0.0);
        const Vector3d vel(0.0, 7500.0, 0.0);

        bool threw = false;
        try {
            from_models_along_iterated(
                t0, pos, vel,
                /*duration_s=*/0.0,  // bad (negative = backward, valid)
                nominal_base_config(),
                quiet_sw(t0, 1000.0),
                BetaProfile1D::constant(256.4));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT_MSG(threw, "expected throw on zero duration");

        threw = false;
        try {
            from_models_along_iterated(
                t0, pos, vel, 60.0,
                nominal_base_config(),
                quiet_sw(t0, 60.0),
                BetaProfile1D::constant(256.4),
                {},
                /*max_iter=*/0);  // bad
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT_MSG(threw, "expected throw on max_iter=0");
    });

    runner.run();
    return print_final_summary();
}
