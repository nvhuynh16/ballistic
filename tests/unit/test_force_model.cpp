// Unit tests for ForceModel (forces/force_model.hpp), which adds up the individual
// accelerations: Earth gravity, third bodies (Sun/Moon/Jupiter), solid Earth tides,
// relativity, drag and radiation pressure. Checks the on/off switches, the split
// into conservative and non-conservative parts, and the expected size ordering of
// the forces in low Earth orbit (LEO).
//
// Force magnitude ordering at LEO (Montenbruck & Gill, "Satellite Orbits", Table 3.1):
// Gravity ~8 m/s², third-body ~1e-6, relativistic ~1e-8, drag ~1e-6 (varies with altitude)

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/forces/force_model.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::forces;
using namespace ballistic::constants;

// =============================================================================
// Test Suites
// =============================================================================

void test_force_model_construction() {
    TestRunner runner("ForceModelConstruction");

    runner.add_test("default_constructor", []() {
        ForceModel model;

        // Default flags (drag and radiation off, see defaults_values below)
        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d acc = model.acceleration(epoch, state);
        // At LEO, total acceleration dominated by gravity: ~8-10 m/s²
        TEST_ASSERT(acc.norm() > 8.0);
        TEST_ASSERT(acc.norm() < 10.0);
    });

    runner.run();
}

void test_enable_disable_forces() {
    TestRunner runner("EnableDisableForces");

    runner.add_test("disable_earth_gravity", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.earth_gravity = false;
        model.set_force_flags(flags);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d earth_acc = model.earth_gravity_acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(earth_acc, Vector3d::Zero(), 1e-15);
    });

    runner.add_test("disable_third_body", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.third_body = false;
        model.set_force_flags(flags);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d tb_acc = model.third_body_acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(tb_acc, Vector3d::Zero(), 1e-15);
    });

    runner.add_test("disable_drag", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.drag = false;
        model.set_force_flags(flags);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(6778137.0, 0.0, 0.0);  // LEO
        state.velocity = Vector3d(0.0, 7670.0, 0.0);

        Vector3d drag_acc = model.drag_acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(drag_acc, Vector3d::Zero(), 1e-15);
    });

    runner.add_test("all_disabled", []() {
        ForceModel model;
        ForceComponentFlags flags;
        flags.earth_gravity = false;
        flags.third_body = false;
        flags.drag = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        model.set_force_flags(flags);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d acc = model.acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(acc, Vector3d::Zero(), 1e-15);
    });

    runner.add_test("re_enable", []() {
        ForceModel model;

        // Disable then re-enable
        ForceComponentFlags flags = model.force_flags();
        flags.earth_gravity = false;
        model.set_force_flags(flags);
        flags.earth_gravity = true;
        model.set_force_flags(flags);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d earth_acc = model.earth_gravity_acceleration(epoch, state);
        TEST_ASSERT(earth_acc.norm() > 5.0);  // Should be significant
    });

    runner.run();
}

void test_gravity_configuration() {
    TestRunner runner("GravityConfiguration");

    runner.add_test("set_gravity_max_degree", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.third_body = false;
        flags.drag = false;
        model.set_force_flags(flags);

        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 3000000.0);
        state.velocity = Vector3d(0.0, 7000.0, 0.0);
        Epoch epoch = Epoch::j2000();

        // Gravity field to degree 6
        model.set_gravity_max_degree(6);
        Vector3d acc_j6 = model.acceleration(epoch, state);

        // Degree 2 only (the J2 oblateness term)
        model.set_gravity_max_degree(2);
        Vector3d acc_j2 = model.acceleration(epoch, state);

        // Degrees 3-6 change the acceleration, but by less than 0.1%
        double diff = (acc_j6 - acc_j2).norm();
        TEST_ASSERT(diff > 0);
        TEST_ASSERT(diff < acc_j2.norm() * 0.001);  // Small difference
    });

    runner.add_test("point_mass_only", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.third_body = false;
        flags.drag = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        model.set_force_flags(flags);
        model.set_gravity_max_degree(1);  // Point mass only

        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);
        Epoch epoch = Epoch::j2000();

        Vector3d acc = model.acceleration(epoch, state);
        Vector3d pm_acc = gravity::point_mass_acceleration(state.position);

        TEST_ASSERT_VEC3_NEAR(acc, pm_acc, 1e-10);
    });

    runner.run();
}

void test_third_body_configuration() {
    TestRunner runner("ThirdBodyConfiguration");

    runner.add_test("enable_sun", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.earth_gravity = false;
        flags.drag = false;
        model.set_force_flags(flags);

        model.third_body().enable_sun(true);
        model.third_body().enable_moon(false);
        model.third_body().enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d acc = model.acceleration(epoch, state);

        // Sun perturbation at LEO is ~5e-7 m/s²; 1e-5 gives 20x headroom
        TEST_ASSERT(acc.norm() > 1e-7);
        TEST_ASSERT(acc.norm() < 1e-5);
    });

    runner.add_test("enable_moon", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.earth_gravity = false;
        flags.drag = false;
        model.set_force_flags(flags);

        model.third_body().enable_sun(false);
        model.third_body().enable_moon(true);
        model.third_body().enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d acc = model.acceleration(epoch, state);
        TEST_ASSERT(acc.norm() > 1e-7);
    });

    runner.add_test("enable_jupiter", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.earth_gravity = false;
        flags.drag = false;
        model.set_force_flags(flags);

        model.third_body().enable_sun(false);
        model.third_body().enable_moon(false);
        model.third_body().enable_jupiter(true);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        Vector3d acc = model.acceleration(epoch, state);

        // Jupiter perturbation is very small (~1e-8 at LEO)
        TEST_ASSERT(acc.norm() > 1e-12);
        TEST_ASSERT(acc.norm() < 1e-6);
    });

    runner.run();
}

void test_total_acceleration() {
    TestRunner runner("TotalAcceleration");

    runner.add_test("sum_of_components", []() {
        ForceModel model;

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(6778137.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7670.0, 0.0);

        Vector3d total = model.acceleration(epoch, state);
        Vector3d earth = model.earth_gravity_acceleration(epoch, state);
        Vector3d tb = model.third_body_acceleration(epoch, state);
        Vector3d drag = model.drag_acceleration(epoch, state);
        Vector3d rel = model.relativistic_acceleration(epoch, state);
        Vector3d tides = model.solid_tides_acceleration(epoch, state);

        Vector3d expected = earth + tb + drag + rel + tides;
        TEST_ASSERT_VEC3_NEAR(total, expected, 1e-10);
    });

    runner.add_test("leo_dominated_by_gravity", []() {
        ForceModel model;

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(6778137.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7670.0, 0.0);

        Vector3d total = model.acceleration(epoch, state);
        Vector3d earth = model.earth_gravity_acceleration(epoch, state);

        // Total should be dominated by Earth gravity
        TEST_ASSERT_NEAR(total.norm(), earth.norm(), earth.norm() * 0.001);
    });

    runner.run();
}

void test_split_acceleration() {
    TestRunner runner("SplitAcceleration");

    runner.add_test("conservative_non_conservative", []() {
        ForceModel model;

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(6778137.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7670.0, 0.0);

        auto components = model.split_acceleration(epoch, state);

        // Conservative = gravity + third-body + solid_tides + relativistic
        Vector3d expected_cons = model.earth_gravity_acceleration(epoch, state)
                               + model.third_body_acceleration(epoch, state)
                               + model.solid_tides_acceleration(epoch, state)
                               + model.relativistic_acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(components.conservative, expected_cons, 1e-10);

        // Non-conservative = drag
        Vector3d expected_nc = model.drag_acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(components.non_conservative, expected_nc, 1e-10);
    });

    runner.add_test("sum_equals_total", []() {
        ForceModel model;

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(6778137.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7670.0, 0.0);

        auto components = model.split_acceleration(epoch, state);
        Vector3d total = model.acceleration(epoch, state);

        Vector3d sum = components.conservative + components.non_conservative;
        TEST_ASSERT_VEC3_NEAR(sum, total, 1e-10);
    });

    runner.add_test("drag_disabled", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.drag = false;
        model.set_force_flags(flags);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(6778137.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7670.0, 0.0);

        auto components = model.split_acceleration(epoch, state);

        // Non-conservative should be zero when drag disabled
        TEST_ASSERT_VEC3_NEAR(components.non_conservative, Vector3d::Zero(), 1e-15);
    });

    runner.run();
}

void test_has_non_conservative_forces() {
    TestRunner runner("HasNonConservativeForces");

    runner.add_test("drag_enabled", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.drag = true;
        model.set_force_flags(flags);

        TEST_ASSERT_TRUE(model.has_non_conservative_forces());
    });

    runner.add_test("drag_disabled", []() {
        ForceModel model;
        ForceComponentFlags flags = model.force_flags();
        flags.drag = false;
        model.set_force_flags(flags);

        TEST_ASSERT_FALSE(model.has_non_conservative_forces());
    });

    runner.run();
}

void test_component_accessors() {
    TestRunner runner("ComponentAccessors");

    runner.add_test("third_body_accessor", []() {
        ForceModel model;

        const auto& tb = model.third_body();

        // Should be able to check configuration
        TEST_ASSERT_TRUE(tb.sun_enabled());
        TEST_ASSERT_TRUE(tb.moon_enabled());
        TEST_ASSERT_FALSE(tb.jupiter_enabled());
    });

    runner.add_test("drag_accessor", []() {
        ForceModel model;
        const auto& drag = model.drag();
        // Only checks that the accessor exists. Drag inputs come from a
        // DragEnvironment, not from this component; the accessor remains for
        // its non-drag setting (the ITRF realization).

        (void)drag;
    });

    runner.run();
}

void test_derivative_function() {
    TestRunner runner("DerivativeFunction");

    runner.add_test("make_derivative_function", []() {
        auto force_model = std::make_shared<ForceModel>();

        auto deriv = make_derivative_function(force_model);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        StateVector derivative = deriv(epoch, state);

        // Derivative.position should be velocity
        TEST_ASSERT_VEC3_NEAR(derivative.position, state.velocity, 1e-10);

        // Derivative.velocity should be acceleration
        Vector3d expected_acc = force_model->acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(derivative.velocity, expected_acc, 1e-10);
    });

    runner.add_test("use_in_integration", []() {
        auto force_model = std::make_shared<ForceModel>();
        ForceComponentFlags flags = force_model->force_flags();
        flags.third_body = false;
        flags.drag = false;
        force_model->set_force_flags(flags);

        auto deriv = make_derivative_function(force_model);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);  // ~Circular

        // Simple Euler step (just to test the derivative)
        double dt = 1.0;
        StateVector d = deriv(epoch, state);
        StateVector new_state = state + dt * d;

        // Should have moved slightly
        TEST_ASSERT((new_state.position - state.position).norm() > 0);
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("high_altitude", []() {
        ForceModel model;

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(42164000.0, 0.0, 0.0);  // GEO
        state.velocity = Vector3d(0.0, 3075.0, 0.0);

        Vector3d acc = model.acceleration(epoch, state);

        // Should be finite and reasonable
        TEST_ASSERT(std::isfinite(acc.norm()));
        TEST_ASSERT(acc.norm() > 0.1);
        TEST_ASSERT(acc.norm() < 1.0);
    });

    runner.add_test("various_positions", []() {
        ForceModel model;

        Epoch epoch = Epoch::j2000();

        std::vector<Vector3d> positions = {
            Vector3d(7000000.0, 0.0, 0.0),
            Vector3d(0.0, 7000000.0, 0.0),
            Vector3d(0.0, 0.0, 7000000.0),
            Vector3d(5000000.0, 5000000.0, 3000000.0),
        };

        for (const auto& pos : positions) {
            StateVector state;
            state.position = pos;
            state.velocity = Vector3d(0.0, 7500.0, 0.0);

            Vector3d acc = model.acceleration(epoch, state);
            TEST_ASSERT(std::isfinite(acc.norm()));
            // All LEO positions: gravity ~5-10 m/s²
            TEST_ASSERT(acc.norm() > 5.0);
            TEST_ASSERT(acc.norm() < 10.0);
        }
    });

    runner.add_test("different_epochs", []() {
        ForceModel model;

        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        // Test various epochs
        for (double days = 0; days < 365; days += 30) {
            Epoch epoch = Epoch::j2000() + days * constants::SECONDS_PER_DAY;
            Vector3d acc = model.acceleration(epoch, state);

            TEST_ASSERT(std::isfinite(acc.norm()));
            TEST_ASSERT(acc.norm() > 5.0);  // Reasonable gravity
        }
    });

    runner.run();
}

// =============================================================================
// Montenbruck Table 3.1 Force Magnitude Ordering Test
// =============================================================================

void test_montenbruck_table_3_1() {
    TestRunner runner("Montenbruck_Table3_1");

    runner.add_test("force_ordering_at_400km_leo", []() {
        // Montenbruck & Gill (2000), Table 3.1: Perturbative accelerations
        // acting on a satellite at 400 km altitude.
        //
        // Expected ordering (approximate magnitudes):
        //   Central gravity:  ~8.4 m/s²
        //   J2 oblateness:    ~4.4e-2 m/s²
        //   J3 pear-shape:    ~1e-4 m/s²
        //   Moon third-body:  ~1.2e-6 m/s²
        //   Sun third-body:   ~5.3e-7 m/s²
        //   SRP:              ~9e-8 m/s² (A/m dependent)
        //   Relativistic:     ~3e-9 m/s²

        // Setup: 400 km circular orbit on x-axis
        const double r = earth::RADIUS_EQUATORIAL + 400000.0;  // 6778137 m
        const double v = std::sqrt(earth::GM / r);  // ECI circular velocity

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(r, 0.0, 0.0);
        state.velocity = Vector3d(0.0, v, 0.0);

        // --- Central gravity ---
        ForceModel model_pm;
        ForceComponentFlags flags_pm;
        flags_pm.earth_gravity = true;
        flags_pm.third_body = false;
        flags_pm.drag = false;
        flags_pm.relativistic = false;
        flags_pm.solid_earth_tides = false;
        flags_pm.radiation = false;
        model_pm.set_force_flags(flags_pm);
        model_pm.set_gravity_max_degree(1);  // Point mass only
        double a_pm = model_pm.acceleration(epoch, state).norm();

        // Central gravity ≈ GM/r² ≈ 8.67 m/s²
        TEST_ASSERT_NEAR_REL(a_pm, earth::GM / (r * r), 0.01);
        TEST_ASSERT_MSG(a_pm > 8.0 && a_pm < 9.0,
            "Point mass gravity should be ~8.7 m/s², got " + std::to_string(a_pm));

        // --- J2 perturbation ---
        // Need position off equator for J2 to show up clearly
        // Use 45° latitude: position = r*(cos45, 0, sin45)
        double cos45 = std::sqrt(2.0) / 2.0;
        StateVector state_45;
        state_45.position = Vector3d(r * cos45, 0.0, r * cos45);
        state_45.velocity = Vector3d(0.0, v, 0.0);

        ForceModel model_j2;
        ForceComponentFlags flags_j2 = flags_pm;
        model_j2.set_force_flags(flags_j2);

        model_j2.set_gravity_max_degree(2);
        double a_j2_full = model_j2.acceleration(epoch, state_45).norm();
        model_j2.set_gravity_max_degree(1);
        double a_j1 = model_j2.acceleration(epoch, state_45).norm();
        double a_j2_pert = std::abs(a_j2_full - a_j1);  // J2 perturbation magnitude

        // J2 perturbation ~4.4e-2 m/s² at 400 km (Table 3.1)
        // Allow order of magnitude: 1e-3 to 1e-1
        TEST_ASSERT_MSG(a_j2_pert > 1e-3 && a_j2_pert < 1e-1,
            "J2 perturbation should be ~4e-2 m/s², got " + std::to_string(a_j2_pert));

        // --- Third-body accelerations ---
        ForceModel model_tb;
        ForceComponentFlags flags_tb;
        flags_tb.earth_gravity = false;
        flags_tb.third_body = true;
        flags_tb.drag = false;
        flags_tb.relativistic = false;
        flags_tb.solid_earth_tides = false;
        flags_tb.radiation = false;
        model_tb.set_force_flags(flags_tb);
        model_tb.third_body().enable_sun(true);
        model_tb.third_body().enable_moon(true);
        model_tb.third_body().enable_jupiter(false);

        // Individual bodies
        model_tb.third_body().enable_sun(true);
        model_tb.third_body().enable_moon(false);
        double a_sun = model_tb.acceleration(epoch, state).norm();

        model_tb.third_body().enable_sun(false);
        model_tb.third_body().enable_moon(true);
        double a_moon = model_tb.acceleration(epoch, state).norm();

        // Sun ~5e-7, Moon ~1e-6 (Table 3.1)
        TEST_ASSERT_MSG(a_sun > 1e-7 && a_sun < 5e-6,
            "Sun perturbation should be ~5e-7 m/s², got " + std::to_string(a_sun));
        TEST_ASSERT_MSG(a_moon > 1e-7 && a_moon < 5e-6,
            "Moon perturbation should be ~1e-6 m/s², got " + std::to_string(a_moon));

        // --- Relativistic (Schwarzschild) ---
        ForceModel model_rel;
        ForceComponentFlags flags_rel;
        flags_rel.earth_gravity = false;
        flags_rel.third_body = false;
        flags_rel.drag = false;
        flags_rel.relativistic = true;
        flags_rel.solid_earth_tides = false;
        flags_rel.radiation = false;
        model_rel.set_force_flags(flags_rel);
        double a_rel = model_rel.acceleration(epoch, state).norm();

        // Relativistic ~3e-9 to 2e-8 m/s² at LEO (Table 3.1)
        TEST_ASSERT_MSG(a_rel > 1e-9 && a_rel < 1e-7,
            "Relativistic should be ~1e-8 m/s², got " + std::to_string(a_rel));

        // --- Verify ordering: gravity >> J2 >> third-body >> relativistic ---
        TEST_ASSERT_MSG(a_pm > a_j2_pert * 100,
            "Central gravity should dominate J2 by >100x");
        TEST_ASSERT_MSG(a_j2_pert > a_moon * 100,
            "J2 should dominate third-body by >100x");
        TEST_ASSERT_MSG(a_moon > a_rel * 10,
            "Third-body should dominate relativistic by >10x");
    });

    runner.run();
}

// =============================================================================
// Radiation Master Switch Tests
// =============================================================================

void test_radiation_master_switch() {
    TestRunner runner("RadiationMasterSwitch");

    runner.add_test("radiation_master_switch_blocks_srp", []() {
        ForceModel model;

        // Disable everything; radiation sub-switches (on the component) stay
        // at their default-enabled state but must be blocked by the master.
        ForceComponentFlags flags;
        flags.earth_gravity = false;
        flags.third_body = false;
        flags.drag = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.radiation = false;
        model.radiation().enable_srp(true);
        model.radiation().enable_albedo(true);
        model.radiation().enable_ir(true);
        model.set_force_flags(flags);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        // With radiation master switch off, all radiation should be zero
        Vector3d acc = model.radiation_acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(acc, Vector3d::Zero(), 1e-15);

        // Total acceleration should also be zero
        Vector3d total = model.acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(total, Vector3d::Zero(), 1e-15);
    });

    runner.run();
}

// =============================================================================
// ForceComponentFlags Builder Tests
// =============================================================================

void test_force_component_flags_builders() {
    TestRunner runner("ForceComponentFlagsBuilders");

    runner.add_test("defaults_values", []() {
        auto flags = ForceComponentFlags::defaults();
        TEST_ASSERT_TRUE(flags.earth_gravity);
        TEST_ASSERT_TRUE(flags.third_body);
        TEST_ASSERT_FALSE(flags.drag);      // drag derives from the DragEnvironment
        TEST_ASSERT_FALSE(flags.radiation);
        TEST_ASSERT_TRUE(flags.relativistic);
        TEST_ASSERT_TRUE(flags.solid_earth_tides);
        TEST_ASSERT_FALSE(flags.empirical);
    });

    runner.add_test("all_enabled", []() {
        auto flags = ForceComponentFlags::all_enabled();
        TEST_ASSERT_TRUE(flags.earth_gravity);
        TEST_ASSERT_TRUE(flags.third_body);
        TEST_ASSERT_TRUE(flags.drag);
        TEST_ASSERT_TRUE(flags.radiation);
        TEST_ASSERT_TRUE(flags.relativistic);
        TEST_ASSERT_TRUE(flags.solid_earth_tides);
        TEST_ASSERT_TRUE(flags.empirical);
    });

    runner.add_test("direct_member_assignment", []() {
        auto flags = ForceComponentFlags::defaults();
        flags.earth_gravity = false;
        TEST_ASSERT_FALSE(flags.earth_gravity);
        TEST_ASSERT_TRUE(flags.third_body);  // Others unchanged
    });

    runner.add_test("multiple_member_assignment", []() {
        auto flags = ForceComponentFlags::defaults();
        flags.third_body = false;
        flags.drag = false;
        flags.radiation = true;

        TEST_ASSERT_TRUE(flags.earth_gravity);
        TEST_ASSERT_FALSE(flags.third_body);
        TEST_ASSERT_FALSE(flags.drag);
        TEST_ASSERT_TRUE(flags.radiation);
        TEST_ASSERT_TRUE(flags.relativistic);
    });

    // The sub-switches live on the RadiationPressure component; verify they
    // gate both the breakdown fields and the enabled total.
    runner.add_test("radiation_subswitch_gates_component", []() {
        ForceModel model;
        SatelliteProperties props(10.0, 0.3);
        model.set_satellite_properties(props, 1000.0);

        ForceComponentFlags flags;
        flags.earth_gravity = false;
        flags.third_body = false;
        flags.drag = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.radiation = true;
        model.set_force_flags(flags);
        model.radiation().enable_albedo(false);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7500.0, 0.0);

        auto fb = model.compute_force_breakdown(epoch, state, false);
        TEST_ASSERT_VEC3_NEAR(fb.albedo, Vector3d::Zero(), 1e-30);
        // Total must equal SRP + IR only (albedo gated off)
        TEST_ASSERT_VEC3_NEAR(fb.total, Vector3d(fb.srp + fb.ir), 1e-18);
    });

    runner.add_test("copy_independence", []() {
        auto original = ForceComponentFlags::defaults();
        auto modified = original;
        modified.earth_gravity = false;

        // Original should be unchanged
        TEST_ASSERT_TRUE(original.earth_gravity);
        TEST_ASSERT_FALSE(modified.earth_gravity);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Runs without an Earth orientation table on purpose (synthetic or
    // self-consistent checks; pinned values assume zero UT1-UTC/polar motion).
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    std::cout << "======================================" << std::endl;
    std::cout << "Force Model Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_force_model_construction();
    test_enable_disable_forces();
    test_gravity_configuration();
    test_third_body_configuration();
    test_total_acceleration();
    test_split_acceleration();
    test_has_non_conservative_forces();
    test_component_accessors();
    test_derivative_function();
    test_edge_cases();
    test_montenbruck_table_3_1();
    test_radiation_master_switch();
    test_force_component_flags_builders();

    return print_final_summary();
}
