// Unit tests for the symplectic integrators in propagators/symplectic.hpp:
// leapfrog (2nd order), Yoshida 6th and 8th order, and the ways of adding
// non-conservative forces such as drag (splitting, "slimplectic", or ignoring
// them). A symplectic integrator preserves the structure of Hamiltonian
// motion, so its energy error stays bounded instead of drifting over many orbits.
//
// The orbits are circular two-body orbits (point-mass gravity only). The energy
// and reversibility bounds are loose upper limits chosen to separate bounded
// error from secular drift; they are not tight measured values.

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/propagators/symplectic.hpp>
#include <ballistic/gravity/point_mass.hpp>
#include <ballistic/propagators/rk45.hpp>
#include <cmath>
#include <iomanip>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::propagators;
using namespace ballistic::constants;

// =============================================================================
// Helper functions
// =============================================================================

// Simple two-body conservative force
ConservativeForce make_two_body_conservative() {
    return [](const Epoch& /* epoch */, const Vector3d& pos) -> Vector3d {
        return gravity::point_mass_acceleration(pos);
    };
}

// Simple drag force (for testing non-conservative)
NonConservativeForce make_simple_drag(double drag_coeff = 1e-10) {
    return [drag_coeff](const Epoch& /* epoch */, const StateVector& state) -> Vector3d {
        double v = state.velocity.norm();
        if (v < 1e-10) return Vector3d::Zero();
        return -drag_coeff * v * state.velocity;
    };
}

// =============================================================================
// Test Suites
// =============================================================================

void test_symplectic_config() {
    TestRunner runner("SymplecticConfig");

    runner.add_test("default_values", []() {
        SymplecticConfig config;

        TEST_ASSERT_NEAR(config.step_size, 30.0, 1e-10);
        TEST_ASSERT_EQ(static_cast<int>(config.order), static_cast<int>(SymplecticOrder::YOSHIDA6));
        TEST_ASSERT_EQ(static_cast<int>(config.nc_strategy),
                       static_cast<int>(NonConservativeStrategy::SLIMPLECTIC));
    });

    runner.add_test("custom_values", []() {
        SymplecticConfig config;
        config.step_size = 60.0;
        config.order = SymplecticOrder::LEAPFROG;
        config.nc_strategy = NonConservativeStrategy::SPLITTING;

        TEST_ASSERT_NEAR(config.step_size, 60.0, 1e-10);
        TEST_ASSERT_EQ(static_cast<int>(config.order), static_cast<int>(SymplecticOrder::LEAPFROG));
    });

    runner.run();
}

void test_symplectic_orders() {
    TestRunner runner("SymplecticOrders");

    runner.add_test("leapfrog", []() {
        SymplecticConfig config;
        config.order = SymplecticOrder::LEAPFROG;
        config.nc_strategy = NonConservativeStrategy::NONE;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force, config);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector result = integrator.step(epoch, state, 30.0);
        TEST_ASSERT(std::isfinite(result.position.norm()));
        // LEO orbit radius should stay ~6e6 to 8e6 m after one step
        TEST_ASSERT(result.position.norm() > 6e6);
        TEST_ASSERT(result.position.norm() < 8e6);
    });

    runner.add_test("yoshida6", []() {
        SymplecticConfig config;
        config.order = SymplecticOrder::YOSHIDA6;
        config.nc_strategy = NonConservativeStrategy::NONE;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force, config);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector result = integrator.step(epoch, state, 30.0);
        TEST_ASSERT(std::isfinite(result.position.norm()));
        TEST_ASSERT(result.position.norm() > 6e6);
        TEST_ASSERT(result.position.norm() < 8e6);
    });

    runner.add_test("yoshida8", []() {
        SymplecticConfig config;
        config.order = SymplecticOrder::YOSHIDA8;
        config.nc_strategy = NonConservativeStrategy::NONE;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force, config);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector result = integrator.step(epoch, state, 30.0);
        TEST_ASSERT(std::isfinite(result.position.norm()));
        TEST_ASSERT(result.position.norm() > 6e6);
        TEST_ASSERT(result.position.norm() < 8e6);
    });

    runner.run();
}

void test_conservative_only() {
    TestRunner runner("ConservativeOnly");

    runner.add_test("construction", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        // Should have NONE strategy for non-conservative
        TEST_ASSERT_EQ(static_cast<int>(integrator.config().nc_strategy),
                       static_cast<int>(NonConservativeStrategy::NONE));
    });

    runner.add_test("integration", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(r, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, v, 0.0);

        StateVector state1 = integrator.integrate(t0, state0, t0 + 600.0);

        TEST_ASSERT(std::isfinite(state1.position.norm()));
        TEST_ASSERT(state1.position.norm() > 0);
    });

    runner.run();
}

void test_non_conservative_strategies() {
    TestRunner runner("NonConservativeStrategies");

    runner.add_test("splitting_strategy", []() {
        SymplecticConfig config;
        config.nc_strategy = NonConservativeStrategy::SPLITTING;
        config.step_size = 30.0;

        auto cons_force = make_two_body_conservative();
        auto nc_force = make_simple_drag();

        SymplecticIntegrator integrator(cons_force, nc_force, config);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector state1 = integrator.step(t0, state0, 30.0);
        TEST_ASSERT(std::isfinite(state1.position.norm()));
    });

    runner.add_test("slimplectic_strategy", []() {
        SymplecticConfig config;
        config.nc_strategy = NonConservativeStrategy::SLIMPLECTIC;
        config.step_size = 30.0;

        auto cons_force = make_two_body_conservative();
        auto nc_force = make_simple_drag();

        SymplecticIntegrator integrator(cons_force, nc_force, config);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector state1 = integrator.step(t0, state0, 30.0);
        TEST_ASSERT(std::isfinite(state1.position.norm()));
    });

    runner.add_test("none_strategy_ignores_nc", []() {
        SymplecticConfig config;
        config.nc_strategy = NonConservativeStrategy::NONE;
        config.step_size = 30.0;

        auto cons_force = make_two_body_conservative();
        auto nc_force = make_simple_drag(1.0);  // Strong drag

        SymplecticIntegrator integrator(cons_force, nc_force, config);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        // With NONE strategy, drag should be ignored
        // So energy should be (nearly) conserved
        double E0 = gravity::specific_orbital_energy(state0.position, state0.velocity);

        StateVector state1 = integrator.integrate(t0, state0, t0 + 600.0);

        double E1 = gravity::specific_orbital_energy(state1.position, state1.velocity);

        double rel_error = std::abs((E1 - E0) / E0);
        TEST_ASSERT(rel_error < 1e-6);  // Energy conserved
    });

    runner.run();
}

void test_energy_conservation() {
    TestRunner runner("EnergyConservation");

    runner.add_test("leapfrog_one_orbit", []() {
        SymplecticConfig config;
        config.order = SymplecticOrder::LEAPFROG;
        config.nc_strategy = NonConservativeStrategy::NONE;
        config.step_size = 10.0;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force, config);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);
        double period = 2.0 * PI * r / v;

        StateVector state;
        state.position = Vector3d(r, 0.0, 0.0);
        state.velocity = Vector3d(0.0, v, 0.0);

        double E0 = gravity::specific_orbital_energy(state.position, state.velocity);

        Epoch t0 = Epoch::j2000();
        StateVector final_state = integrator.integrate(t0, state, t0 + period);

        double E1 = gravity::specific_orbital_energy(final_state.position, final_state.velocity);

        // Bounded energy error (no drift); 1e-4 is a loose limit for a 10 s leapfrog step.
        double rel_error = std::abs((E1 - E0) / E0);
        TEST_ASSERT(rel_error < 1e-4);  // Bounded, not necessarily tiny
    });

    runner.add_test("yoshida6_better_conservation", []() {
        SymplecticConfig config_2;
        config_2.order = SymplecticOrder::LEAPFROG;
        config_2.nc_strategy = NonConservativeStrategy::NONE;
        config_2.step_size = 30.0;

        SymplecticConfig config_6;
        config_6.order = SymplecticOrder::YOSHIDA6;
        config_6.nc_strategy = NonConservativeStrategy::NONE;
        config_6.step_size = 30.0;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator_2(cons_force, config_2);
        SymplecticIntegrator integrator_6(cons_force, config_6);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        StateVector state;
        state.position = Vector3d(r, 0.0, 0.0);
        state.velocity = Vector3d(0.0, v, 0.0);

        double E0 = gravity::specific_orbital_energy(state.position, state.velocity);

        Epoch t0 = Epoch::j2000();
        StateVector final_2 = integrator_2.integrate(t0, state, t0 + 3600.0);
        StateVector final_6 = integrator_6.integrate(t0, state, t0 + 3600.0);

        double E1_2 = gravity::specific_orbital_energy(final_2.position, final_2.velocity);
        double E1_6 = gravity::specific_orbital_energy(final_6.position, final_6.velocity);

        double err_2 = std::abs((E1_2 - E0) / E0);
        double err_6 = std::abs((E1_6 - E0) / E0);

        // At same step size, Yoshida6 (6th order) must outperform Leapfrog (2nd order)
        TEST_ASSERT(err_6 < err_2);
    });

    runner.add_test("long_term_stability", []() {
        SymplecticConfig config;
        config.order = SymplecticOrder::YOSHIDA6;
        config.nc_strategy = NonConservativeStrategy::NONE;
        config.step_size = 60.0;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force, config);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);
        double period = 2.0 * PI * r / v;

        StateVector state;
        state.position = Vector3d(r, 0.0, 0.0);
        state.velocity = Vector3d(0.0, v, 0.0);

        double E0 = gravity::specific_orbital_energy(state.position, state.velocity);

        // Integrate for 10 orbits
        Epoch t0 = Epoch::j2000();
        StateVector final_state = integrator.integrate(t0, state, t0 + 10.0 * period);

        double E_final = gravity::specific_orbital_energy(final_state.position, final_state.velocity);

        // Key property: error should be BOUNDED (no secular drift)
        double rel_error = std::abs((E_final - E0) / E0);
        TEST_ASSERT(rel_error < 1e-3);  // Bounded over 10 orbits
    });

    runner.run();
}

void test_hamiltonian() {
    TestRunner runner("Hamiltonian");

    runner.add_test("compute_hamiltonian_formula", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        double H = integrator.hamiltonian(epoch, state);

        // H = T + V = 0.5*v² - GM/r
        double expected = 0.5 * state.velocity.squaredNorm() - earth::GM / state.position.norm();
        TEST_ASSERT_NEAR(H, expected, 1e-6);
    });

    runner.add_test("hamiltonian_conservation", []() {
        SymplecticConfig config;
        config.order = SymplecticOrder::YOSHIDA6;
        config.nc_strategy = NonConservativeStrategy::NONE;
        config.step_size = 30.0;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force, config);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(r, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, v, 0.0);

        double H0 = integrator.hamiltonian(t0, state0);

        // Integrate for 1 hour
        StateVector state1 = integrator.integrate(t0, state0, t0 + 3600.0);

        double H1 = integrator.hamiltonian(t0 + 3600.0, state1);

        // Symplectic integrator should conserve Hamiltonian well
        double rel_error = std::abs((H1 - H0) / H0);
        TEST_ASSERT(rel_error < 1e-4);
    });

    runner.run();
}

void test_statistics() {
    TestRunner runner("Statistics");

    runner.add_test("steps_counted", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        Epoch t0 = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        integrator.integrate(t0, state, t0 + 600.0);

        auto& stats = integrator.stats();
        TEST_ASSERT(stats.steps_taken > 0);
    });

    runner.add_test("reset_stats", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        Epoch t0 = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        integrator.integrate(t0, state, t0 + 600.0);
        integrator.reset_stats();

        auto& stats = integrator.stats();
        TEST_ASSERT_EQ(stats.steps_taken, static_cast<size_t>(0));
    });

    runner.run();
}

void test_symplectic_propagator() {
    TestRunner runner("SymplecticPropagator");

    runner.add_test("construction", []() {
        Epoch t0 = Epoch::j2000();
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7670.0, 0.0);
        BallisticModel ballistic(1000.0, 10.0, 2.2);
        WeatherConditions weather = WeatherConditions::quiet();

        auto prop = create_symplectic_propagator(t0, pos, vel, ballistic, weather);

        TEST_ASSERT(prop != nullptr);
    });

    runner.add_test("propagate_to", []() {
        Epoch t0 = Epoch::j2000();
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7670.0, 0.0);
        BallisticModel ballistic(1000.0, 10.0, 2.2);
        WeatherConditions weather = WeatherConditions::quiet();

        auto prop = create_symplectic_propagator(t0, pos, vel, ballistic, weather, false);

        Epoch t1 = t0 + 600.0;
        StateVector state1 = prop->propagate_to(t1);

        TEST_ASSERT(std::isfinite(state1.position.norm()));
        TEST_ASSERT((state1.position - pos).norm() > 0);
    });

    runner.add_test("call_operator", []() {
        Epoch t0 = Epoch::j2000();
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7670.0, 0.0);
        BallisticModel ballistic(1000.0, 10.0, 2.2);
        WeatherConditions weather = WeatherConditions::quiet();

        auto prop = create_symplectic_propagator(t0, pos, vel, ballistic, weather, false);

        Epoch t1 = t0 + 600.0;
        PropagationResult result = (*prop)(t1);

        // Result should be in ECEF
        TEST_ASSERT(std::isfinite(result.position.norm()));
        TEST_ASSERT(std::isfinite(result.velocity.norm()));
        TEST_ASSERT(std::isfinite(result.acceleration.norm()));
    });

    runner.add_test("energy_tracking", []() {
        Epoch t0 = Epoch::j2000();
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7670.0, 0.0);
        BallisticModel ballistic(1000.0, 10.0, 2.2);
        WeatherConditions weather = WeatherConditions::quiet();

        // Disable drag for energy test
        SymplecticConfig config;
        config.nc_strategy = NonConservativeStrategy::NONE;

        auto prop = create_symplectic_propagator(t0, pos, vel, ballistic, weather, false, config);

        double E0 = prop->initial_energy();

        // Propagate
        prop->propagate_to(t0 + 600.0);

        double E1 = prop->current_energy();
        double drift = prop->energy_drift();

        // Symplectic with NONE strategy on conservative forces: < 0.1% drift
        TEST_ASSERT(std::abs(drift) < 0.001);
    });

    runner.run();
}

void test_backward_integration() {
    TestRunner runner("BackwardIntegration");

    runner.add_test("reversibility", []() {
        SymplecticConfig config;
        config.nc_strategy = NonConservativeStrategy::NONE;
        config.step_size = 30.0;

        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force, config);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        StateVector state0;
        state0.position = Vector3d(r, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, v, 0.0);

        Epoch t0 = Epoch::j2000();

        // Forward
        StateVector state1 = integrator.integrate(t0, state0, t0 + 600.0);

        integrator.reset_stats();

        // Backward
        StateVector state_back = integrator.integrate(t0 + 600.0, state1, t0);

        // Symmetric compositions are time-reversible: integrating 600 s forward and
        // back must return near the start (loose 100 m / 0.1 m/s limits).
        double pos_error = (state_back.position - state0.position).norm();
        double vel_error = (state_back.velocity - state0.velocity).norm();

        TEST_ASSERT(pos_error < 100.0);   // Within 100 m
        TEST_ASSERT(vel_error < 0.1);     // Within 0.1 m/s
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("zero_time_step", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector result = integrator.step(epoch, state, 0.0);

        // Zero step should return same state
        TEST_ASSERT_VEC3_NEAR(result.position, state.position, 1e-10);
        TEST_ASSERT_VEC3_NEAR(result.velocity, state.velocity, 1e-10);
    });

    runner.add_test("negative_time_step", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector result = integrator.step(epoch, state, -30.0);

        // Should step backward
        TEST_ASSERT(std::isfinite(result.position.norm()));
    });

    runner.add_test("elliptical_orbit", []() {
        auto cons_force = make_two_body_conservative();
        SymplecticIntegrator integrator(cons_force);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r) * 1.2;  // Elliptical

        StateVector state;
        state.position = Vector3d(r, 0.0, 0.0);
        state.velocity = Vector3d(0.0, v, 0.0);

        Epoch t0 = Epoch::j2000();
        StateVector final_state = integrator.integrate(t0, state, t0 + 3600.0);

        TEST_ASSERT(std::isfinite(final_state.position.norm()));
    });

    runner.run();
}

// Accuracy against references, not just finiteness: one circular LEO orbit
// at a 10 s step compared with the closed-form Kepler position (point-mass
// gravity), and a J2 orbit compared with RK45 at a tight tolerance. Both are
// in an inertial frame with the z axis on the Earth's pole, where J2 is
// time-independent.
void test_accuracy_vs_references() {
    TestRunner runner("AccuracyVsReferences");

    // Circular orbit at 7000 km radius, in the x-y plane, run for the whole
    // number of 10 s steps closest to one period; returns the distance to the
    // analytic position (the start rotated by n*t).
    auto kepler_error = [](SymplecticOrder order) {
        SymplecticConfig config;
        config.order = order;
        config.nc_strategy = NonConservativeStrategy::NONE;
        config.step_size = 10.0;
        SymplecticIntegrator integrator(make_two_body_conservative(), config);

        const double r = 7000000.0;
        const double mu = earth::GM;
        const double n = std::sqrt(mu / (r * r * r));
        const double period = 2.0 * PI / n;
        const double t = 10.0 * std::round(period / 10.0);
        StateVector s0{Vector3d(r, 0.0, 0.0), Vector3d(0.0, r * n, 0.0)};
        const Epoch t0 = Epoch::j2000();
        const StateVector s1 = integrator.integrate(t0, s0, t0 + t);
        const Vector3d exact(r * std::cos(n * t), r * std::sin(n * t), 0.0);
        return (s1.position - exact).norm();
    };

    // Bound: 1 mm after one orbit at 10 s. Measured 5.0e-6 m (Yoshida 6) and
    // 1.2e-7 m (Yoshida 8); the bound is the requirement, not a measured value.
    runner.add_test("yoshida6_matches_kepler", [&]() {
        const double err = kepler_error(SymplecticOrder::YOSHIDA6);
        std::cout << "  Yoshida 6 vs Kepler, one orbit at 10 s: "
                  << std::scientific << std::setprecision(3) << err << " m\n";
        TEST_ASSERT_MSG(err < 1e-3, "Yoshida 6 error " + std::to_string(err) + " m");
    });

    runner.add_test("yoshida8_matches_kepler", [&]() {
        const double err = kepler_error(SymplecticOrder::YOSHIDA8);
        std::cout << "  Yoshida 8 vs Kepler, one orbit at 10 s: "
                  << std::scientific << std::setprecision(3) << err << " m\n";
        TEST_ASSERT_MSG(err < 1e-3, "Yoshida 8 error " + std::to_string(err) + " m");
    });

    // Bound: 1 cm after one inclined (51.6 deg) orbit with J2, Yoshida 6 at
    // 10 s vs RK45 at relative tolerance 1e-13. Measured 1.9e-6 m; the bound
    // is the requirement, not a measured value.
    runner.add_test("yoshida6_matches_rk45_j2", []() {
        const double r = 6778137.0;
        const double v = std::sqrt(earth::GM / r);
        const double inc = 51.6 * PI / 180.0;
        StateVector s0{Vector3d(r, 0.0, 0.0),
                       Vector3d(0.0, v * std::cos(inc), v * std::sin(inc))};
        const double t = 10.0 * std::round(2.0 * PI * std::sqrt(r * r * r / earth::GM) / 10.0);
        const Epoch t0 = Epoch::j2000();

        SymplecticConfig config;
        config.order = SymplecticOrder::YOSHIDA6;
        config.nc_strategy = NonConservativeStrategy::NONE;
        config.step_size = 10.0;
        SymplecticIntegrator y6(
            [](const Epoch&, const Vector3d& p) { return gravity::gravity_with_j2(p); }, config);
        const StateVector a = y6.integrate(t0, s0, t0 + t);

        RK45Config rk;
        rk.relative_tolerance = 1e-13;
        rk.absolute_tolerance = 1e-9;
        rk.initial_step_size = 1.0;
        rk.max_step_size = 30.0;
        const StateVector b = integrate_rk45(t0, s0, t0 + t,
            [](const Epoch&, const StateVector& s) {
                return StateVector{s.velocity, gravity::gravity_with_j2(s.position)};
            }, rk);
        const double err = (a.position - b.position).norm();
        std::cout << "  Yoshida 6 vs RK45 (J2), one orbit at 10 s: "
                  << std::scientific << std::setprecision(3) << err << " m\n";
        TEST_ASSERT_MSG(err < 1e-2, "Yoshida 6 vs RK45 " + std::to_string(err) + " m");
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
    std::cout << "Symplectic Integrator Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_symplectic_config();
    test_symplectic_orders();
    test_conservative_only();
    test_non_conservative_strategies();
    test_energy_conservation();
    test_hamiltonian();
    test_statistics();
    test_symplectic_propagator();
    test_backward_integration();
    test_edge_cases();
    test_accuracy_vs_references();

    return print_final_summary();
}
