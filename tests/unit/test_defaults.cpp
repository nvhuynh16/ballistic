// Unit tests for defaults.hpp
// Tests that the default constants (integrator step sizes and step-control
// factors, altitude limits, time tolerances) are in sensible ranges and ordered
// consistently; values are design choices, so the checks are relations, not pins.

#include "../test_utils.hpp"
#include <ballistic/core/defaults.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::test;

static TestRunner runner("Defaults Tests");

void test_rk45_step_sizes_ordered() {
    // MIN < INITIAL < MAX
    TEST_ASSERT_TRUE(defaults::RK45_MIN_STEP_SIZE < defaults::RK45_INITIAL_STEP_SIZE);
    TEST_ASSERT_TRUE(defaults::RK45_INITIAL_STEP_SIZE < defaults::RK45_MAX_STEP_SIZE);

    // All positive
    TEST_ASSERT_TRUE(defaults::RK45_MIN_STEP_SIZE > 0.0);
    TEST_ASSERT_TRUE(defaults::RK45_INITIAL_STEP_SIZE > 0.0);
    TEST_ASSERT_TRUE(defaults::RK45_MAX_STEP_SIZE > 0.0);
}

void test_rk45_scale_factors() {
    // MIN_SCALE < 1.0 < MAX_SCALE
    TEST_ASSERT_TRUE(defaults::RK45_MIN_SCALE < 1.0);
    TEST_ASSERT_TRUE(defaults::RK45_MAX_SCALE > 1.0);

    // Both positive
    TEST_ASSERT_TRUE(defaults::RK45_MIN_SCALE > 0.0);
    TEST_ASSERT_TRUE(defaults::RK45_MAX_SCALE > 0.0);
}

void test_rk45_safety_factor() {
    // Safety factor in (0, 1)
    TEST_ASSERT_TRUE(defaults::RK45_SAFETY_FACTOR > 0.0);
    TEST_ASSERT_TRUE(defaults::RK45_SAFETY_FACTOR < 1.0);
}

void test_rk45_controller_exponent() {
    // Standard value for 5th-order method: 1/(p+1) = 1/5 = 0.2
    TEST_ASSERT_NEAR(defaults::RK45_STEP_CONTROLLER_EXPONENT, 0.2, 1e-15);
    TEST_ASSERT_NEAR(defaults::RK45_STEP_CONTROLLER_EXPONENT, 1.0 / 5.0, 1e-15);
}

void test_rk45_tolerances_positive() {
    TEST_ASSERT_TRUE(defaults::RK45_RELATIVE_TOLERANCE > 0.0);
    TEST_ASSERT_TRUE(defaults::RK45_ABSOLUTE_TOLERANCE > 0.0);
    TEST_ASSERT_TRUE(defaults::RK45_ERROR_FLOOR > 0.0);
}

void test_force_model_epsilons() {
    TEST_ASSERT_TRUE(defaults::VELOCITY_EPSILON > 0.0);
    TEST_ASSERT_TRUE(defaults::VISIBILITY_EPSILON > 0.0);
}

void test_atmosphere_threshold() {
    // 1000 km = 1e6 m (NRLMSIS 2.1 validity ceiling)
    TEST_ASSERT_NEAR(defaults::ATMOSPHERE_ALTITUDE_THRESHOLD_M, 1e6, 1.0);
}

void test_epsilon_ordering() {
    // Integration epsilon should be tighter than epoch equality
    TEST_ASSERT_TRUE(defaults::INTEGRATION_EPSILON_SECONDS < defaults::EPOCH_EQUALITY_SECONDS);
}

void test_max_iterations_positive() {
    TEST_ASSERT_TRUE(defaults::RK45_MAX_ITERATIONS > 0);
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "BALLISTIC LIBRARY - DEFAULTS TESTS" << std::endl;
    std::cout << "========================================" << std::endl;

    runner.add_test("rk45_step_sizes_ordered", test_rk45_step_sizes_ordered);
    runner.add_test("rk45_scale_factors", test_rk45_scale_factors);
    runner.add_test("rk45_safety_factor", test_rk45_safety_factor);
    runner.add_test("rk45_controller_exponent", test_rk45_controller_exponent);
    runner.add_test("rk45_tolerances_positive", test_rk45_tolerances_positive);
    runner.add_test("force_model_epsilons", test_force_model_epsilons);
    runner.add_test("atmosphere_threshold", test_atmosphere_threshold);
    runner.add_test("epsilon_ordering", test_epsilon_ordering);
    runner.add_test("max_iterations_positive", test_max_iterations_positive);

    int result = runner.run();
    return print_final_summary();
}
