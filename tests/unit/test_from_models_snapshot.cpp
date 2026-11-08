// Unit tests for the from_models_snapshot generator, which tabulates one
// altitude profile of the NRLMSIS 2.1 atmosphere model at a fixed latitude,
// longitude and time.
//
// Cross-checks that the generated DragEnvironment's density column matches
// direct NRLMSIS 2.1 evaluations at the same (alt, lat, lon, t). Synthetic
// constant quiet weather; no data files.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/internal/nrlmsis21_full.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/init.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;

namespace {

Epoch t_test() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

Vector3d position_at(double lat_deg, double lon_deg, double alt_km) {
    constexpr double DEG_TO_RAD = 0.017453292519943295;
    return coords::geodetic_to_ecef(lat_deg * DEG_TO_RAD,
                                     lon_deg * DEG_TO_RAD,
                                     alt_km * 1000.0);
}

}  // namespace

void test_basic_construction() {
    TestRunner runner("BasicConstruction");

    // The generator returns an environment that turns drag on.
    runner.add_test("generator_returns_drag_capable_env", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t_test() - 86400.0, t_test() + 86400.0});

        BetaProfile1D beta_prior = BetaProfile1D::constant(100.0);
        DragEnvironment env = from_models_snapshot(
            t_test(), 0.0, 0.0, sw, std::move(beta_prior));

        TEST_ASSERT(env.has_drag());
    });

    runner.run();
}

void test_density_matches_direct_nrlmsis() {
    TestRunner runner("DensityMatchesDirectNRLMSIS");

    // At the table's altitude nodes the density equals a direct model call to 0.5%
    // (a loose bound: the 1D profile keeps log(rho) in double precision, so at a
    // node only rounding remains).
    runner.add_test("matches_direct_nrlmsis_at_grid_nodes", []() {
        ballistic::initialize({});

        WeatherConditions w = WeatherConditions::quiet();
        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            w, TimeWindow{t_test() - 86400.0, t_test() + 86400.0});

        BetaProfile1D beta_prior = BetaProfile1D::constant(100.0);
        std::vector<double> grid{300.0, 400.0, 500.0};
        DragEnvironment env = from_models_snapshot(
            t_test(), 0.0, 0.0, sw, std::move(beta_prior), grid);

        NRLMSIS21 nrlmsis;
        for (double h : grid) {
            const double expected = nrlmsis.density(h, 0.0, 0.0, t_test(), w);
            const Vector3d pos = position_at(0.0, 0.0, h);
            const DragSample s = env.evaluate(pos, t_test());
            // At an exact grid node the log-rho spline returns the tabulated
            // value (double precision), so only rounding remains.
            const double rel_err = std::abs(s.rho_kg_m3 - expected) / expected;
            TEST_ASSERT(rel_err < 5e-3);
        }
    });

    runner.run();
}

void test_density_decreases_with_altitude() {
    TestRunner runner("DensityDecreasesWithAltitude");

    // Physical sanity: density falls with altitude from 200 to 800 km.
    runner.add_test("monotonic_decrease_in_atmosphere", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t_test() - 86400.0, t_test() + 86400.0});

        BetaProfile1D beta_prior = BetaProfile1D::constant(100.0);
        DragEnvironment env = from_models_snapshot(
            t_test(), 0.0, 0.0, sw, std::move(beta_prior),
            std::vector<double>{200.0, 400.0, 600.0, 800.0});

        const Vector3d p200 = position_at(0.0, 0.0, 200.0);
        const Vector3d p400 = position_at(0.0, 0.0, 400.0);
        const Vector3d p600 = position_at(0.0, 0.0, 600.0);
        const Vector3d p800 = position_at(0.0, 0.0, 800.0);
        const double r200 = env.evaluate(p200, t_test()).rho_kg_m3;
        const double r400 = env.evaluate(p400, t_test()).rho_kg_m3;
        const double r600 = env.evaluate(p600, t_test()).rho_kg_m3;
        const double r800 = env.evaluate(p800, t_test()).rho_kg_m3;
        TEST_ASSERT(r200 > r400);
        TEST_ASSERT(r400 > r600);
        TEST_ASSERT(r600 > r800);
    });

    runner.run();
}

void test_beta_passthrough() {
    TestRunner runner("BetaPassthrough");

    // The ballistic coefficient (beta = m / (Cd A), kg/m^2) supplied by the caller comes
    // back unchanged (exact up to 1e-9).
    runner.add_test("beta_column_matches_supplied_profile", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t_test() - 86400.0, t_test() + 86400.0});

        DragEnvironment env = from_models_snapshot(
            t_test(), 0.0, 0.0, sw, BetaProfile1D::constant(75.5));

        const Vector3d pos = position_at(0.0, 0.0, 400.0);
        const DragSample s = env.evaluate(pos, t_test());
        TEST_ASSERT_NEAR(s.beta_kg_m2, 75.5, 1e-9);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "from_models_snapshot Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_basic_construction();
    test_density_matches_direct_nrlmsis();
    test_density_decreases_with_altitude();
    test_beta_passthrough();

    return print_final_summary();
}
