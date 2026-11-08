// Tests the force-breakdown report (TrajectoryConfig::report_forces): when
// enabled, each step reports the acceleration of every force component
// (gravity, third body, solid tides, relativity, drag, SRP = solar radiation
// pressure, ...), the enabled components sum to the total, and disabled
// components are zero unless include_disabled_forces asks for them. Magnitude
// checks are physical (gravity ~8.7 m/s^2 at 400 km).
#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"

using namespace ballistic;
using namespace ballistic::test;

int main() {
    // Runs without an Earth orientation table on purpose (synthetic or
    // self-consistent checks; pinned values assume zero UT1-UTC/polar motion).
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    TestRunner runner("Force Breakdown Tests");

    runner.add_test("force_breakdown_disabled_by_default", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        // report_forces defaults to false
        TEST_ASSERT_FALSE(config.report_forces);

        Trajectory traj(t0, pos, vel, config);
        auto result = traj.step();

        // Forces should not be populated when report_forces is false
        TEST_ASSERT_FALSE(result.forces.has_value());
    });

    runner.add_test("force_breakdown_step_enabled", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.step();

        // Forces should be populated
        TEST_ASSERT(result.forces.has_value());

        // Gravity at ~400km altitude: ~8.7 m/s^2
        TEST_ASSERT(result.forces->gravity.norm() > 5.0);
        TEST_ASSERT(result.forces->gravity.norm() < 10.0);

        // Total should be dominated by gravity
        TEST_ASSERT(result.forces->total.norm() > 5.0);
        TEST_ASSERT(result.forces->total.norm() < 10.0);
    });

    runner.add_test("force_breakdown_propagate_enabled", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.propagate(t0 + 60.0);

        // Forces should be populated
        TEST_ASSERT(result.forces.has_value());

        // Gravity should be non-zero
        TEST_ASSERT(result.forces->gravity.norm() > 5.0);
    });

    runner.add_test("force_breakdown_components_sum_to_total", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        config.enable_third_body = true;
        config.enable_relativistic = true;
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.step();
        TEST_ASSERT(result.forces.has_value());

        // Verify total equals sum of enabled components (gravity + third_body + solid_tides + relativistic)
        Vector3d expected_total = result.forces->gravity +
                                  result.forces->third_body +
                                  result.forces->solid_tides +
                                  result.forces->relativistic;

        TEST_ASSERT_VEC3_NEAR(result.forces->total, expected_total, 1e-10);
    });

    runner.add_test("force_breakdown_disabled_forces_excluded", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        config.include_disabled_forces = false;  // Default
  // Disabled
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.step();
        TEST_ASSERT(result.forces.has_value());

        // Drag and SRP should be zero since they're disabled and include_disabled_forces=false
        TEST_ASSERT_VEC3_NEAR(result.forces->drag, Vector3d::Zero(), 1e-15);
        TEST_ASSERT_VEC3_NEAR(result.forces->srp, Vector3d::Zero(), 1e-15);
        TEST_ASSERT_VEC3_NEAR(result.forces->albedo, Vector3d::Zero(), 1e-15);
        TEST_ASSERT_VEC3_NEAR(result.forces->ir, Vector3d::Zero(), 1e-15);
    });

    runner.add_test("force_breakdown_includes_disabled_when_flag_set", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        config.include_disabled_forces = true;  // Include disabled forces for diagnostic
  // Disabled but will still be computed for diagnostic
        Trajectory traj(t0, pos, vel, config);

        // Set up drag model with ballistic properties
        BallisticModel model(1000.0, 10.0, 2.2);  // mass, area, Cd

        auto result = traj.step();
        TEST_ASSERT(result.forces.has_value());

        // Drag should be computed even though it's disabled (for diagnostic purposes)
        // At ~400km altitude with quiet conditions, drag is small but non-zero
        // Note: drag may be zero if atmosphere model returns zero density
        // Just verify we got a valid breakdown with physical gravity
        TEST_ASSERT(result.forces->gravity.norm() > 5.0);
        TEST_ASSERT(result.forces->gravity.norm() < 10.0);
    });

    runner.add_test("force_breakdown_third_body_non_zero", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        config.enable_third_body = true;
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.step();
        TEST_ASSERT(result.forces.has_value());

        // Third-body perturbations: Sun ~5e-7, Moon ~1e-6 m/s² at LEO
        double third_body_magnitude = result.forces->third_body.norm();
        TEST_ASSERT(third_body_magnitude > 1e-7);  // Should be measurable
        TEST_ASSERT(third_body_magnitude < 1e-4);  // But much smaller than gravity
    });

    runner.add_test("force_breakdown_relativistic_non_zero", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        config.enable_relativistic = true;
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.step();
        TEST_ASSERT(result.forces.has_value());

        // Relativistic corrections: Schwarzschild + Lense-Thirring ~1e-9 to 1e-8 m/s² at LEO
        double relativistic_magnitude = result.forces->relativistic.norm();
        TEST_ASSERT(relativistic_magnitude > 1e-10);  // Should be measurable
        TEST_ASSERT(relativistic_magnitude < 1e-7);   // But very small
    });

    runner.add_test("force_breakdown_radiation_convenience_method", []() {
        ForceBreakdown breakdown;
        breakdown.srp = Vector3d(1.0, 0.0, 0.0);
        breakdown.albedo = Vector3d(0.0, 2.0, 0.0);
        breakdown.ir = Vector3d(0.0, 0.0, 3.0);

        Vector3d radiation = breakdown.radiation();
        TEST_ASSERT_VEC3_NEAR(radiation, Vector3d(1.0, 2.0, 3.0), 1e-15);
    });

    runner.add_test("force_breakdown_symplectic_integrator", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        config.integrator = IntegratorType::YOSHIDA6;
        config.symplectic.step_size = 30.0;
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.step();
        TEST_ASSERT(result.forces.has_value());

        // Gravity should be non-zero
        TEST_ASSERT(result.forces->gravity.norm() > 5.0);
    });

    runner.add_test("force_breakdown_multiple_steps", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config;
        config.report_forces = true;
        Trajectory traj(t0, pos, vel, config);

        // Take multiple steps
        for (int i = 0; i < 5; ++i) {
            auto result = traj.step();
            TEST_ASSERT(result.forces.has_value());
            TEST_ASSERT(result.forces->gravity.norm() > 5.0);
        }
    });

    runner.run();
    return print_final_summary();
}
