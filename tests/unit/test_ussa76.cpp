// U.S. Standard Atmosphere 1976, 0-86 km (atmosphere/ussa76.hpp), against the
// standard's Table I (NOAA-S/T 76-1562, geometric altitudes). The model is
// computed from the defining constants, so agreement to 0.01% tests the
// layer table, the hydrostatic integration and the M/M0 correction at once.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/ussa76.hpp>

using namespace ballistic;
using namespace ballistic::test;
namespace u76 = ballistic::atmosphere::ussa76;

int main() {
    TestRunner runner("USSA76");

    runner.add_test("table_I_density_pressure_temperature", []() {
        // Density, pressure and temperature agree with the standard's Table I to 0.01%
        // (the table's own printed precision).
        struct Row { double z_m, rho, p, t; };
        // Table I of the standard (geometric altitude).
        const Row rows[] = {
            {0.0, 1.2250, 101325.0, 288.150},   {10e3, 4.1351e-1, 26499.9, 223.252},
            {20e3, 8.8910e-2, 5529.3, 216.650}, {30e3, 1.8410e-2, 1197.0, 226.509},
            {40e3, 3.9957e-3, 287.14, 250.350}, {50e3, 1.0269e-3, 79.779, 270.650},
            {70e3, 8.2829e-5, 5.2209, 219.585}, {80e3, 1.8458e-5, 1.0524, 198.639},
            {86e3, 6.958e-6, 0.37338, 186.87},
        };
        for (const auto& r : rows) {
            const auto s = u76::evaluate(r.z_m);
            TEST_ASSERT(std::abs(s.density_kg_m3 / r.rho - 1.0) < 1e-4);
            TEST_ASSERT(std::abs(s.pressure_pa / r.p - 1.0) < 1e-4);
            TEST_ASSERT(std::abs(s.temperature_k / r.t - 1.0) < 1e-4);
        }
    });

    // Sea-level speed of sound is the standard's 340.294 m/s.
    runner.add_test("speed_of_sound_sea_level", []() {
        TEST_ASSERT_NEAR(u76::evaluate(0.0).speed_of_sound_m_s, 340.294, 0.01);
    });

    // Density has no jump at the layer boundaries: 1 mm either side of each base agrees to
    // 1e-6 (relative), i.e. the layers join up.
    runner.add_test("continuous_across_layer_bases", []() {
        for (const auto& L : u76::kLayers) {
            if (L.base_km == 0.0) continue;
            // geometric altitude of the geopotential base
            const double z = L.base_km * 1e3 * u76::kR0 / (u76::kR0 - L.base_km * 1e3);
            const double lo = u76::density(z - 1e-3), hi = u76::density(z + 1e-3);
            TEST_ASSERT(std::abs(lo / hi - 1.0) < 1e-6);
        }
    });

    // Above 86 km or below -5 km the model raises DataCoverageError instead of extrapolating;
    // -5 km itself is still inside and denser than sea level.
    runner.add_test("outside_range_raises", []() {
        TEST_ASSERT_THROWS(u76::evaluate(86001.0), DataCoverageError);
        TEST_ASSERT_THROWS(u76::evaluate(-5001.0), DataCoverageError);
        TEST_ASSERT(u76::density(-5000.0) > 1.225);
    });

    runner.run();
    return print_final_summary();
}
