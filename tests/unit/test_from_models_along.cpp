// Unit tests for from_models_along, which tabulates NRLMSIS 2.1 density (2D:
// altitude × time) under the sub-satellite point of each sample of a trajectory.
// Checks construction and input validation, that the table reproduces a direct
// NRLMSIS 2.1 call at the first sample to 0.5%, and that the β column (β = m/(Cd·A), kg/m²) is passed through.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/internal/nrlmsis21_full.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/init.hpp>
#include <stdexcept>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;

namespace {

Epoch t0_test() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

Vector3d position_at(double lat_deg, double lon_deg, double alt_km) {
    constexpr double DEG_TO_RAD = 0.017453292519943295;
    return coords::geodetic_to_ecef(lat_deg * DEG_TO_RAD,
                                     lon_deg * DEG_TO_RAD,
                                     alt_km * 1000.0);
}

}  // namespace

void test_basic_construction() {
    TestRunner runner("BasicConstruction");

    runner.add_test("two_sample_trajectory_builds_env", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<Epoch> epochs{t0_test(), t0_test() + 3600.0};
        std::vector<Vector3d> positions{
            position_at(0.0, 0.0, 400.0),    // equator, prime meridian
            position_at(30.0, 90.0, 400.0),  // mid-latitude, 90E
        };

        IterationDiag diag;
        DragEnvironment env = from_models_along(
            epochs, positions, sw, BetaProfile1D::constant(100.0),
            std::vector<double>{300.0, 400.0, 500.0},
            &diag);

        TEST_ASSERT(env.has_drag());
        TEST_ASSERT_EQ(diag.iterations, 1);
        TEST_ASSERT(diag.converged);
    });

    runner.add_test("size_mismatch_rejected", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<Epoch> epochs{t0_test(), t0_test() + 3600.0};
        std::vector<Vector3d> positions{position_at(0.0, 0.0, 400.0)};  // wrong size

        bool threw = false;
        try {
            (void)from_models_along(epochs, positions, sw,
                                     BetaProfile1D::constant(100.0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("single_sample_rejected", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<Epoch> epochs{t0_test()};
        std::vector<Vector3d> positions{position_at(0.0, 0.0, 400.0)};

        bool threw = false;
        try {
            (void)from_models_along(epochs, positions, sw,
                                     BetaProfile1D::constant(100.0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_density_along_trajectory() {
    TestRunner runner("DensityAlongTrajectory");

    runner.add_test("density_at_first_sample_matches_direct_nrlmsis", []() {
        ballistic::initialize({});

        WeatherConditions w = WeatherConditions::quiet();
        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            w, TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<Epoch> epochs{t0_test(), t0_test() + 3600.0};
        std::vector<Vector3d> positions{
            position_at(0.0, 0.0, 400.0),
            position_at(0.0, 0.0, 400.0),  // hold position constant
        };

        DragEnvironment env = from_models_along(
            epochs, positions, sw, BetaProfile1D::constant(100.0),
            std::vector<double>{400.0});

        // At the first sample (t=t0, lat=0, lon=0, alt=400 km) the density must
        // match a direct NRLMSIS 2.1 call to 0.5%.
        NRLMSIS21 nrlmsis;
        const double expected = nrlmsis.density(400.0, 0.0, 0.0, t0_test(), w);
        const DragSample s = env.evaluate(positions[0], t0_test());
        const double rel_err = std::abs(s.rho_kg_m3 - expected) / expected;
        TEST_ASSERT(rel_err < 5e-3);
    });

    runner.run();
}

void test_beta_passthrough() {
    TestRunner runner("BetaPassthrough");

    runner.add_test("beta_passed_through_unchanged", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<Epoch> epochs{t0_test(), t0_test() + 3600.0};
        std::vector<Vector3d> positions{
            position_at(0.0, 0.0, 400.0),
            position_at(10.0, 30.0, 400.0),
        };

        DragEnvironment env = from_models_along(
            epochs, positions, sw, BetaProfile1D::constant(123.4));

        const DragSample s = env.evaluate(positions[0], t0_test());
        TEST_ASSERT_NEAR(s.beta_kg_m2, 123.4, 1e-9);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "from_models_along Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_basic_construction();
    test_density_along_trajectory();
    test_beta_passthrough();

    return print_final_summary();
}
