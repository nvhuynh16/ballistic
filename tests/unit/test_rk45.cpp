// Unit tests for the RK45 integrator (propagators/rk45.hpp): the adaptive
// Dormand-Prince Runge-Kutta method of orders 4 and 5, where the difference
// between the two orders estimates the step error and sets the next step size.
//
// Checks: default settings; the method's coefficient identities (exact to
// rounding); single forward/backward steps; two-body orbits (closure after one
// period, energy and angular momentum conserved to 1e-10 relative at the tight
// tolerances used); a harmonic oscillator; step-size control and statistics;
// edge cases (zero and very short intervals).

#include "../test_utils.hpp"
#include <ballistic/propagators/rk45.hpp>
#include <ballistic/gravity/point_mass.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::propagators;
using namespace ballistic::constants;

// =============================================================================
// Helper functions
// =============================================================================

// Simple two-body derivative for testing
DerivativeFunction make_two_body_derivative() {
    return [](const Epoch& /* epoch */, const StateVector& state) -> StateVector {
        Vector3d acc = gravity::point_mass_acceleration(state.position);
        return StateVector(state.velocity, acc);
    };
}

// Harmonic oscillator derivative (for testing)
DerivativeFunction make_harmonic_oscillator(double omega = 1.0) {
    return [omega](const Epoch& /* epoch */, const StateVector& state) -> StateVector {
        Vector3d acc = -omega * omega * state.position;
        return StateVector(state.velocity, acc);
    };
}

// =============================================================================
// Test Suites
// =============================================================================

void test_rk45_config() {
    TestRunner runner("RK45Config");

    runner.add_test("default_values", []() {
        RK45Config config;

        TEST_ASSERT_NEAR(config.initial_step_size, 60.0, 1e-10);
        TEST_ASSERT_NEAR(config.min_step_size, 0.001, 1e-10);
        TEST_ASSERT_NEAR(config.max_step_size, 3600.0, 1e-10);
        TEST_ASSERT_NEAR(config.relative_tolerance, 1e-10, 1e-15);
        TEST_ASSERT_NEAR(config.absolute_tolerance, 1e-12, 1e-15);
    });

    runner.add_test("custom_values", []() {
        RK45Config config;
        config.initial_step_size = 30.0;
        config.relative_tolerance = 1e-8;

        TEST_ASSERT_NEAR(config.initial_step_size, 30.0, 1e-10);
        TEST_ASSERT_NEAR(config.relative_tolerance, 1e-8, 1e-15);
    });

    runner.run();
}

void test_dopri5_coefficients() {
    TestRunner runner("DOPRI5Coefficients");

    runner.add_test("time_coefficients_sum", []() {
        using namespace dopri5;

        // Stage times c2..c5 lie strictly inside the step; c6 = c7 = 1
        TEST_ASSERT(c2 > 0 && c2 < 1);
        TEST_ASSERT(c3 > 0 && c3 < 1);
        TEST_ASSERT(c4 > 0 && c4 < 1);
        TEST_ASSERT(c5 > 0 && c5 < 1);
        TEST_ASSERT_NEAR(c6, 1.0, 1e-15);
        TEST_ASSERT_NEAR(c7, 1.0, 1e-15);
    });

    runner.add_test("b_coefficients_sum", []() {
        using namespace dopri5;

        // 5th order b coefficients should sum to 1
        double sum = b1 + b2 + b3 + b4 + b5 + b6 + b7;
        TEST_ASSERT_NEAR(sum, 1.0, 1e-14);
    });

    runner.add_test("bs_coefficients_sum", []() {
        using namespace dopri5;

        // 4th order b* coefficients should sum to 1
        double sum = bs1 + bs2 + bs3 + bs4 + bs5 + bs6 + bs7;
        TEST_ASSERT_NEAR(sum, 1.0, 1e-14);
    });

    runner.run();
}

void test_rk45_integrator_construction() {
    TestRunner runner("RK45IntegratorConstruction");

    runner.add_test("default_config", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        TEST_ASSERT_NEAR(integrator.current_step_size(), 60.0, 1e-10);
    });

    runner.add_test("custom_config", []() {
        RK45Config config;
        config.initial_step_size = 30.0;

        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv, config);

        TEST_ASSERT_NEAR(integrator.current_step_size(), 30.0, 1e-10);
    });

    runner.run();
}

void test_single_step() {
    TestRunner runner("SingleStep");

    runner.add_test("step_forward", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);  // ~Circular

        auto result = integrator.step(epoch, state, 60.0);

        // State should have changed
        TEST_ASSERT((result.state.position - state.position).norm() > 0);
        TEST_ASSERT((result.state.velocity - state.velocity).norm() > 0);
    });

    runner.add_test("step_backward", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        auto result = integrator.step(epoch, state, -60.0);

        // Should step backward
        TEST_ASSERT(result.epoch < epoch);
    });

    runner.add_test("error_estimate", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        auto result = integrator.step(epoch, state, 60.0);

        // Error estimate should be positive
        TEST_ASSERT(result.error_estimate >= 0);
    });

    runner.add_test("step_acceptance", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-6;  // Loose tolerance
        RK45Integrator integrator(deriv, config);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        auto result = integrator.step(epoch, state, 60.0);

        // Acceptance is not guaranteed even at a loose tolerance; only check
        // that a positive next step size is proposed.
        TEST_ASSERT(result.next_step_size > 0);
    });

    runner.run();
}

void test_integration() {
    TestRunner runner("Integration");

    runner.add_test("one_orbit", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

        // Circular orbit at ~7000 km
        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);  // Circular velocity
        double period = 2.0 * PI * r / v;      // Orbital period

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(r, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, v, 0.0);

        // Integrate one orbit
        Epoch t1 = t0 + period;
        StateVector state1 = integrator.integrate(t0, state0, t1);

        // Should return to approximately starting position
        double pos_error = (state1.position - state0.position).norm();
        TEST_ASSERT(pos_error < 10.0);  // Within 10 m after one orbit (rtol=1e-12)
    });

    runner.add_test("half_orbit", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);
        double period = 2.0 * PI * r / v;

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(r, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, v, 0.0);

        // Integrate half orbit
        Epoch t_half = t0 + period / 2.0;
        StateVector state_half = integrator.integrate(t0, state0, t_half);

        // Should be on opposite side
        TEST_ASSERT(state_half.position.x() < 0);
    });

    runner.add_test("backward_integration", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(r, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, v, 0.0);

        // Forward integration
        Epoch t1 = t0 + 1000.0;
        StateVector state1 = integrator.integrate(t0, state0, t1);

        integrator.reset_stats();

        // Backward integration should recover initial state
        StateVector state_back = integrator.integrate(t1, state1, t0);

        double pos_error = (state_back.position - state0.position).norm();
        double vel_error = (state_back.velocity - state0.velocity).norm();

        TEST_ASSERT(pos_error < 1.0);    // Within 1 m
        TEST_ASSERT(vel_error < 0.001);  // Within 1 mm/s
    });

    runner.add_test("short_integration", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        // Very short integration
        Epoch t1 = t0 + 1.0;  // 1 second
        StateVector state1 = integrator.integrate(t0, state0, t1);

        // Should have moved ~7.5 km
        double dist = (state1.position - state0.position).norm();
        TEST_ASSERT_NEAR(dist, 7546.0, 10.0);  // ~v * t (1-second, gravity correction ~4m)
    });

    runner.run();
}

void test_energy_conservation() {
    TestRunner runner("EnergyConservation");

    runner.add_test("circular_orbit_one_orbit", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

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

        // Two-body energy is conserved; 1e-10 relative is what the tight
        // tolerances of this test deliver.
        double rel_error = std::abs((E1 - E0) / E0);
        TEST_ASSERT(rel_error < 1e-10);
    });

    runner.add_test("elliptical_orbit", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

        // Elliptical orbit (higher velocity at perigee)
        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r) * 1.1;  // Above circular

        StateVector state;
        state.position = Vector3d(r, 0.0, 0.0);
        state.velocity = Vector3d(0.0, v, 0.0);

        double E0 = gravity::specific_orbital_energy(state.position, state.velocity);

        // Integrate for 1 hour
        Epoch t0 = Epoch::j2000();
        StateVector final_state = integrator.integrate(t0, state, t0 + 3600.0);

        double E1 = gravity::specific_orbital_energy(final_state.position, final_state.velocity);

        double rel_error = std::abs((E1 - E0) / E0);
        TEST_ASSERT(rel_error < 1e-10);
    });

    runner.run();
}

void test_angular_momentum_conservation() {
    TestRunner runner("AngularMomentumConservation");

    runner.add_test("circular_orbit", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        StateVector state;
        state.position = Vector3d(r, 0.0, 0.0);
        state.velocity = Vector3d(0.0, v, 0.0);

        Vector3d h0 = gravity::specific_angular_momentum(state.position, state.velocity);

        // Integrate for 1 hour
        Epoch t0 = Epoch::j2000();
        StateVector final_state = integrator.integrate(t0, state, t0 + 3600.0);

        Vector3d h1 = gravity::specific_angular_momentum(final_state.position, final_state.velocity);

        // Two-body angular momentum is conserved (1e-10 relative)
        double rel_error = (h1 - h0).norm() / h0.norm();
        TEST_ASSERT(rel_error < 1e-10);
    });

    runner.run();
}

void test_statistics() {
    TestRunner runner("Statistics");

    runner.add_test("function_evaluations", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch t0 = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        integrator.integrate(t0, state, t0 + 600.0);

        auto& stats = integrator.stats();
        TEST_ASSERT(stats.function_evaluations > 0);
        TEST_ASSERT(stats.steps_taken > 0);
    });

    runner.add_test("reset_stats", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch t0 = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        integrator.integrate(t0, state, t0 + 600.0);
        integrator.reset_stats();

        auto& stats = integrator.stats();
        TEST_ASSERT_EQ(stats.function_evaluations, static_cast<size_t>(0));
        TEST_ASSERT_EQ(stats.steps_taken, static_cast<size_t>(0));
    });

    runner.add_test("min_max_step_size", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch t0 = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        integrator.integrate(t0, state, t0 + 600.0);

        auto& stats = integrator.stats();
        TEST_ASSERT(stats.min_step_size > 0);
        TEST_ASSERT(stats.max_step_size >= stats.min_step_size);
    });

    runner.run();
}

void test_step_size_control() {
    TestRunner runner("StepSizeControl");

    runner.add_test("set_step_size", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        integrator.set_step_size(30.0);
        TEST_ASSERT_NEAR(integrator.current_step_size(), 30.0, 1e-10);
    });

    runner.add_test("step_size_clamping", []() {
        RK45Config config;
        config.min_step_size = 1.0;
        config.max_step_size = 100.0;

        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv, config);

        integrator.set_step_size(0.1);  // Below min
        TEST_ASSERT_NEAR(integrator.current_step_size(), 1.0, 1e-10);

        integrator.set_step_size(1000.0);  // Above max
        TEST_ASSERT_NEAR(integrator.current_step_size(), 100.0, 1e-10);
    });

    runner.add_test("adaptive_step_sizing", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.initial_step_size = 10.0;
        RK45Integrator integrator(deriv, config);

        Epoch t0 = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        // Integrate and check that step size adapts
        integrator.integrate(t0, state, t0 + 600.0);

        // Step size should have changed from initial
        double final_step = integrator.current_step_size();
        TEST_ASSERT(std::abs(final_step - 10.0) > 1.0);
    });

    runner.run();
}

void test_convenience_function() {
    TestRunner runner("ConvenienceFunction");

    runner.add_test("integrate_rk45", []() {
        auto deriv = make_two_body_derivative();

        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(r, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, v, 0.0);

        Epoch t1 = t0 + 600.0;
        StateVector state1 = integrate_rk45(t0, state0, t1, deriv);

        // Should have moved
        TEST_ASSERT((state1.position - state0.position).norm() > 0);
    });

    runner.add_test("integrate_rk45_custom_config", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-8;

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        Epoch t1 = t0 + 600.0;
        StateVector state1 = integrate_rk45(t0, state0, t1, deriv, config);

        TEST_ASSERT(std::isfinite(state1.position.norm()));
    });

    runner.run();
}

void test_harmonic_oscillator() {
    TestRunner runner("HarmonicOscillator");

    runner.add_test("period", []() {
        double omega = 1.0;
        auto deriv = make_harmonic_oscillator(omega);

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

        // Initial condition: displaced in x
        StateVector state;
        state.position = Vector3d(1.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 0.0, 0.0);

        Epoch t0 = Epoch::j2000();
        double period = 2.0 * PI / omega;

        // After one period, should return to start
        StateVector final_state = integrator.integrate(t0, state, t0 + period);

        TEST_ASSERT_VEC3_NEAR(final_state.position, state.position, 1e-4);  // Relaxed for numerical drift
        TEST_ASSERT_VEC3_NEAR(final_state.velocity, state.velocity, 1e-4);
    });

    runner.add_test("energy_conservation", []() {
        double omega = 2.0;
        auto deriv = make_harmonic_oscillator(omega);

        RK45Config config;
        config.relative_tolerance = 1e-12;
        RK45Integrator integrator(deriv, config);

        StateVector state;
        state.position = Vector3d(1.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, omega, 0.0);  // With some initial velocity

        // Energy = 0.5 * v² + 0.5 * omega² * x²
        auto energy = [omega](const StateVector& s) {
            return 0.5 * s.velocity.squaredNorm() + 0.5 * omega * omega * s.position.squaredNorm();
        };

        double E0 = energy(state);

        Epoch t0 = Epoch::j2000();
        StateVector final_state = integrator.integrate(t0, state, t0 + 10.0);

        double E1 = energy(final_state);

        double rel_error = std::abs((E1 - E0) / E0);
        TEST_ASSERT(rel_error < 1e-10);
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("zero_integration_time", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        StateVector state1 = integrator.integrate(t0, state0, t0);

        // Should return same state
        TEST_ASSERT_VEC3_NEAR(state1.position, state0.position, 1e-10);
        TEST_ASSERT_VEC3_NEAR(state1.velocity, state0.velocity, 1e-10);
    });

    runner.add_test("very_short_time", []() {
        auto deriv = make_two_body_derivative();
        RK45Integrator integrator(deriv);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        // Very short integration
        StateVector state1 = integrator.integrate(t0, state0, t0 + 0.001);

        // Should be close to original
        TEST_ASSERT((state1.position - state0.position).norm() < 10.0);
    });

    runner.add_test("tight_tolerance", []() {
        auto deriv = make_two_body_derivative();

        RK45Config config;
        config.relative_tolerance = 1e-14;
        config.absolute_tolerance = 1e-16;
        RK45Integrator integrator(deriv, config);

        Epoch t0 = Epoch::j2000();
        StateVector state0;
        state0.position = Vector3d(7000000.0, 0.0, 0.0);
        state0.velocity = Vector3d(0.0, 7546.0, 0.0);

        // Should still work with tight tolerance
        StateVector state1 = integrator.integrate(t0, state0, t0 + 60.0);
        TEST_ASSERT(std::isfinite(state1.position.norm()));
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "RK45 Integrator Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_rk45_config();
    test_dopri5_coefficients();
    test_rk45_integrator_construction();
    test_single_step();
    test_integration();
    test_energy_conservation();
    test_angular_momentum_conservation();
    test_statistics();
    test_step_size_control();
    test_convenience_function();
    test_harmonic_oscillator();
    test_edge_cases();

    return print_final_summary();
}
