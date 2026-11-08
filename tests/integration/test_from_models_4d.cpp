// Integration test for the from_models_4d generator, which tabulates the
// NRLMSIS 2.1 atmosphere model on a full altitude x latitude x longitude x time grid.
//
// Uses a small grid (5 alt × 4 lat × 8 lon × 2 epochs = 320 cells) to keep
// CI runtime in a few seconds. Synthetic constant quiet weather; no data files.
// Validates basic constructional sanity:
//   - Generator builds an env with has_drag() == true.
//   - Density at the (0, 0, 400 km, t0) corner matches direct NRLMSIS 2.1.
//   - Density monotonically decreases with altitude at fixed (lat, lon, t).
//
// Full-scale production grids (12M+ cells) are not exercised here — those
// belong in the BALLISTIC_RUN_HEAVY_TESTS tier. This test runs in the
// default CI tier.

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

Epoch t0_test() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

Vector3d position_at(double lat_deg, double lon_deg, double alt_km) {
    constexpr double DEG_TO_RAD = 0.017453292519943295;
    return coords::geodetic_to_ecef(lat_deg * DEG_TO_RAD,
                                     lon_deg * DEG_TO_RAD,
                                     alt_km * 1000.0);
}

}  // namespace

void test_small_grid_construction() {
    TestRunner runner("SmallGridConstruction");

    // A small grid builds and turns drag on.
    runner.add_test("small_4d_grid_has_drag", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<double> alt{200.0, 300.0, 400.0, 500.0, 600.0};
        std::vector<double> lat{-30.0, 0.0, 30.0, 60.0};
        std::vector<double> lon{-90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0, -135.0};
        std::vector<Epoch>  epochs{t0_test(), t0_test() + 3600.0};

        DragEnvironment env = from_models_4d(
            alt, lat, lon, epochs, sw, BetaProfile1D::constant(100.0));

        TEST_ASSERT(env.has_drag());
    });

    // An empty altitude grid is refused with std::invalid_argument.
    runner.add_test("empty_grid_rejected", []() {
        ballistic::initialize({});
        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        bool threw = false;
        try {
            (void)from_models_4d({}, {0.0}, {0.0}, {t0_test()}, sw,
                                  BetaProfile1D::constant(100.0));
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_density_matches_direct_nrlmsis() {
    TestRunner runner("DensityMatchesDirectNRLMSIS");

    // At a grid corner the density equals a direct model call to 1.5% (float16
    // log(rho) storage plus bilinear lat/lon interpolation, ~1% expected).
    runner.add_test("density_at_grid_corner_matches_direct_call", []() {
        ballistic::initialize({});

        WeatherConditions w = WeatherConditions::quiet();
        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            w, TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<double> alt{300.0, 400.0, 500.0};
        std::vector<double> lat{0.0, 30.0};
        std::vector<double> lon{0.0, 60.0};
        std::vector<Epoch>  epochs{t0_test(), t0_test() + 3600.0};

        DragEnvironment env = from_models_4d(
            alt, lat, lon, epochs, sw, BetaProfile1D::constant(100.0));

        // Sample at (lat=0, lon=0, alt=400 km, t=t0).
        Vector3d pos = position_at(0.0, 0.0, 400.0);
        DragSample s = env.evaluate(pos, t0_test());

        NRLMSIS21 nrlmsis;
        const double expected = nrlmsis.density(400.0, 0.0, 0.0, t0_test(), w);
        // Float16 round-trip + bilinear interp: tolerate ~1% at corners.
        const double rel_err = std::abs(s.rho_kg_m3 - expected) / expected;
        TEST_ASSERT(rel_err < 1.5e-2);
    });

    runner.run();
}

void test_density_decreases_with_altitude() {
    TestRunner runner("DensityDecreasesWithAltitude");

    // Physical sanity: density falls with altitude from 200 to 800 km.
    runner.add_test("monotonic_at_fixed_lat_lon_t", []() {
        ballistic::initialize({});

        SpaceWeatherHistory sw = SpaceWeatherHistory::from_constant(
            WeatherConditions::quiet(),
            TimeWindow{t0_test() - 86400.0, t0_test() + 86400.0});

        std::vector<double> alt{200.0, 400.0, 600.0, 800.0};
        std::vector<double> lat{0.0, 30.0};
        std::vector<double> lon{0.0, 60.0};
        std::vector<Epoch>  epochs{t0_test(), t0_test() + 3600.0};

        DragEnvironment env = from_models_4d(
            alt, lat, lon, epochs, sw, BetaProfile1D::constant(100.0));

        const double r200 = env.evaluate(position_at(0.0, 0.0, 200.0), t0_test()).rho_kg_m3;
        const double r400 = env.evaluate(position_at(0.0, 0.0, 400.0), t0_test()).rho_kg_m3;
        const double r600 = env.evaluate(position_at(0.0, 0.0, 600.0), t0_test()).rho_kg_m3;
        const double r800 = env.evaluate(position_at(0.0, 0.0, 800.0), t0_test()).rho_kg_m3;
        TEST_ASSERT(r200 > r400);
        TEST_ASSERT(r400 > r600);
        TEST_ASSERT(r600 > r800);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "from_models_4d Integration Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_small_grid_construction();
    test_density_matches_direct_nrlmsis();
    test_density_decreases_with_altitude();

    return print_final_summary();
}
