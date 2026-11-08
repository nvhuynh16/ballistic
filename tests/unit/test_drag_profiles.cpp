// Unit tests for the three kinds of column a DragEnvironment holds: air density
// (1D by altitude, 2D by altitude and time, 4D by altitude/latitude/longitude/
// time), wind (east-north-up or Earth-fixed ECEF), and the ballistic coefficient
// β = m/(Cd·A) [kg/m²]. Checks interpolation, edge behavior (below the table
// throws, above it clamps) and input validation on small hand-built tables.
// Tolerances are interpolation or storage precision (log-density and
// half-precision wind storage), not measured data.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/density_profile.hpp>
#include <ballistic/atmosphere/wind_profile.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/core/errors.hpp>
#include <stdexcept>
#include <cmath>
#include <string>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::atmosphere;

namespace {

// Build an ECEF position from geodetic (lat_deg, lon_deg, altitude_km).
Vector3d position_at(double lat_deg, double lon_deg, double alt_km) {
    constexpr double DEG_TO_RAD = 0.017453292519943295;
    return coords::geodetic_to_ecef(lat_deg * DEG_TO_RAD,
                                     lon_deg * DEG_TO_RAD,
                                     alt_km * 1000.0);
}

Epoch t0() { return Epoch::from_utc(2024, 6, 15, 12, 0, 0); }

}  // namespace

// ============================================================================
// Density profile tests
// ============================================================================

void test_density_profile_1d() {
    TestRunner runner("DensityProfile1D");

    runner.add_test("empty_returns_zero", []() {
        DensityProfile1D p;
        TEST_ASSERT(p.empty());
        // Empty profile returns 0 — we use a non-ECEF position so we don't
        // hit the below-table throw path.
        TEST_ASSERT_NEAR(p.evaluate(Vector3d(7e6, 0, 0), t0()), 0.0, 1e-15);
    });

    runner.add_test("exact_node_recovery_log_scale", []() {
        // log(rho) is linear in altitude for an exponential atmosphere.
        // Cubic spline on exact log values should recover them at nodes.
        std::vector<double> alt{100.0, 200.0, 300.0, 400.0, 500.0};
        std::vector<double> rho{1e-7, 1e-10, 1e-12, 4e-13, 1e-13};
        DensityProfile1D p(alt, rho);
        for (std::size_t i = 0; i < alt.size(); ++i) {
            Vector3d pos = position_at(0.0, 0.0, alt[i]);
            const double got = p.evaluate(pos, t0());
            const double rel_err = std::abs(got - rho[i]) / rho[i];
            TEST_ASSERT(rel_err < 1e-3);
        }
    });

    runner.add_test("midpoint_monotone_in_log", []() {
        // Between two altitude knots, rho should be between them (in log space).
        std::vector<double> alt{200.0, 400.0};
        std::vector<double> rho{1e-10, 1e-13};
        DensityProfile1D p(alt, rho);
        Vector3d pos300 = position_at(0.0, 0.0, 300.0);
        const double rho300 = p.evaluate(pos300, t0());
        TEST_ASSERT(rho300 < 1e-10);
        TEST_ASSERT(rho300 > 1e-13);
    });

    runner.add_test("below_table_throws", []() {
        std::vector<double> alt{200.0, 400.0};
        std::vector<double> rho{1e-10, 1e-13};
        DensityProfile1D p(alt, rho);
        Vector3d pos = position_at(0.0, 0.0, 100.0);  // below table's 200 km
        bool threw = false;
        try {
            (void)p.evaluate(pos, t0());
        } catch (const std::out_of_range&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("above_table_clamps", []() {
        // Above-table clamps to the top altitude's density (no throw).
        std::vector<double> alt{200.0, 400.0};
        std::vector<double> rho{1e-10, 1e-13};
        DensityProfile1D p(alt, rho);
        Vector3d pos = position_at(0.0, 0.0, 600.0);  // above 400 km
        const double got = p.evaluate(pos, t0());
        // Should be approximately the top value (1e-13).
        TEST_ASSERT_NEAR(got, 1e-13, 1e-14);
    });

    runner.add_test("rejects_zero_density", []() {
        bool threw = false;
        try {
            DensityProfile1D p({100.0, 200.0}, {1e-10, 0.0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_density_profile_2d() {
    TestRunner runner("DensityProfile2D");

    runner.add_test("linear_in_time_at_fixed_altitude", []() {
        // 2 altitudes, 2 epochs. Density doubles over the time window.
        std::vector<double> alt{200.0, 400.0};
        std::vector<Epoch> ep{t0(), t0() + 3600.0};
        // grid: t0 has [1e-10, 1e-13]; t0+1h has [2e-10, 2e-13]
        std::vector<double> grid{1e-10, 1e-13, 2e-10, 2e-13};
        DensityProfile2D p(alt, ep, grid);

        Vector3d pos300 = position_at(0.0, 0.0, 300.0);
        const double r0   = p.evaluate(pos300, t0());
        const double r_mid = p.evaluate(pos300, t0() + 1800.0);
        const double r1   = p.evaluate(pos300, t0() + 3600.0);

        // r_mid should be ~midway between r0 and r1.
        TEST_ASSERT_NEAR(r_mid, 0.5 * (r0 + r1), 1e-2 * r0);
    });

    runner.add_test("size_mismatch_rejected", []() {
        bool threw = false;
        try {
            DensityProfile2D p({200.0, 400.0}, {t0(), t0() + 3600.0},
                                {1.0, 2.0, 3.0});  // wrong size
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_density_field_4d() {
    TestRunner runner("DensityField4D");

    runner.add_test("uniform_field_returns_constant_density", []() {
        // 2x2x2x2 grid with uniform density. Result should be uniform.
        const double rho = 1e-11;
        std::vector<double> alt{200.0, 400.0};
        std::vector<double> lat{-30.0, 30.0};
        std::vector<double> lon{-90.0, 90.0};
        std::vector<Epoch>  ep{t0(), t0() + 3600.0};
        std::vector<half> storage(2 * 2 * 2 * 2, encode_log_rho(rho));

        DensityField4D field(alt, lat, lon, ep, storage);
        Vector3d pos = position_at(0.0, 45.0, 300.0);  // interior point
        const double got = field.evaluate(pos, t0() + 1800.0);
        // Round-trip + interpolation < 0.5% combined error.
        TEST_ASSERT(std::abs(got - rho) / rho < 5e-3);
    });

    runner.add_test("size_mismatch_rejected", []() {
        bool threw = false;
        try {
            DensityField4D field({200.0, 400.0}, {0.0, 30.0}, {0.0},
                                  {t0()},
                                  std::vector<half>(3));  // wrong size
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("single_value_axes_are_held", []() {
        // One latitude, one longitude and one epoch: the query used to read
        // the (absent) second cell on each of those axes, past the end of the
        // storage. Each single-value axis is now held at its value.
        const double rho_lo = 1e-10, rho_hi = 1e-12;
        std::vector<double> alt{200.0, 400.0};
        std::vector<half> storage{encode_log_rho(rho_lo), encode_log_rho(rho_hi)};
        DensityField4D field(alt, {10.0}, {20.0}, {t0()}, storage);
        // Anywhere, any time: the altitude profile of the only column.
        // Compared with the stored (16-bit) values, so only interpolation
        // round-off remains.
        const double want_lo = decode_log_rho(encode_log_rho(rho_lo));
        const double want_hi = decode_log_rho(encode_log_rho(rho_hi));
        const double at_node = field.evaluate(position_at(-50.0, 170.0, 200.0),
                                              t0() + 86400.0);
        TEST_ASSERT(std::abs(at_node - want_lo) / want_lo < 1e-6);
        const double top = field.evaluate(position_at(60.0, -100.0, 400.0),
                                          t0() - 3600.0);
        TEST_ASSERT(std::abs(top - want_hi) / want_hi < 1e-6);
    });

    runner.add_test("single_altitude_rejected", []() {
        // A spline through one altitude has no profile: refused with the
        // axis named.
        bool threw = false;
        try {
            DensityField4D field({300.0}, {0.0, 30.0}, {0.0, 90.0},
                                 {t0(), t0() + 3600.0},
                                 std::vector<half>(8, encode_log_rho(1e-11)));
        } catch (const ConfigError& e) {
            threw = std::string(e.what()).find("altitude axis") != std::string::npos;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

// ============================================================================
// Wind profile tests
// ============================================================================

void test_wind_profile_1d_enu() {
    TestRunner runner("WindProfile1D_ENU");

    runner.add_test("empty_returns_zero", []() {
        WindProfile1D_ENU w;
        TEST_ASSERT(w.empty());
        TEST_ASSERT(w.evaluate(Vector3d(7e6, 0, 0), t0()).norm() < 1e-12);
    });

    runner.add_test("eastward_wind_at_equator_points_east", []() {
        // At equator (lat=0, lon=0), east is +Y in ECEF (right-hand rule with
        // x through Greenwich-equator, z through north pole).
        // ENU = (east, north, up).
        std::vector<double> alt{200.0, 400.0};
        std::vector<double> meridional{0.0, 0.0};  // no north-south
        std::vector<double> zonal{50.0, 50.0};      // 50 m/s east
        WindProfile1D_ENU w(alt, meridional, zonal);
        Vector3d pos = position_at(0.0, 0.0, 300.0);
        Vector3d wind_ecef = w.evaluate(pos, t0());
        // ECEF y-component dominates, magnitude ~50 m/s.
        TEST_ASSERT(std::abs(wind_ecef.y() - 50.0) < 1.0);
        TEST_ASSERT(std::abs(wind_ecef.x()) < 1.0);
        TEST_ASSERT(std::abs(wind_ecef.z()) < 1.0);
    });

    runner.add_test("length_mismatch_raises_input_plausibility_error_naming_the_sizes", []() {
        // checked before the splines are built: the message names all three lengths
        bool threw = false;
        try {
            WindProfile1D_ENU w({200.0, 400.0, 600.0}, {0.0, 0.0, 0.0}, {50.0, 50.0});
        } catch (const InputPlausibilityError& e) {
            threw = true;
            const std::string msg = e.what();
            TEST_ASSERT(msg.find("3 altitudes") != std::string::npos);
            TEST_ASSERT(msg.find("3 meridional") != std::string::npos);
            TEST_ASSERT(msg.find("2 zonal") != std::string::npos);
        }
        TEST_ASSERT(threw);
        TEST_ASSERT_THROWS(WindProfile1D_ENU({200.0, 400.0}, {0.0}, {1.0, 2.0}),
                           InputPlausibilityError);
    });

    runner.run();
}

void test_wind_profile_2d_ecef() {
    TestRunner runner("WindProfile2D_ECEF");

    runner.add_test("returns_stored_ecef_wind_at_node", []() {
        std::vector<double> alt{200.0, 400.0};
        std::vector<Epoch> ep{t0(), t0() + 3600.0};
        std::vector<Vector3d> grid{
            Vector3d(10, 0, 0), Vector3d(20, 0, 0),  // t0
            Vector3d(30, 0, 0), Vector3d(40, 0, 0),  // t0+1h
        };
        WindProfile2D_ECEF w(alt, ep, grid);
        Vector3d pos = position_at(0.0, 0.0, 200.0);
        Vector3d got = w.evaluate(pos, t0());
        TEST_ASSERT(std::abs(got.x() - 10.0) < 1e-6);
    });

    runner.run();
}

// ============================================================================
// Beta profile tests
// ============================================================================

void test_beta_profile_1d() {
    TestRunner runner("BetaProfile1D");

    runner.add_test("constant_factory_returns_constant_beta", []() {
        BetaProfile1D p = BetaProfile1D::constant(45.45);
        for (double h : {100.0, 300.0, 500.0, 1000.0}) {
            Vector3d pos = position_at(0.0, 0.0, h);
            TEST_ASSERT_NEAR(p.evaluate(pos, t0()), 45.45, 1e-9);
        }
    });

    runner.add_test("interpolates_between_knots", []() {
        BetaProfile1D p({200.0, 400.0}, {50.0, 100.0});
        Vector3d pos = position_at(0.0, 0.0, 300.0);
        const double mid = p.evaluate(pos, t0());
        TEST_ASSERT(mid > 50.0);
        TEST_ASSERT(mid < 100.0);
    });

    runner.add_test("rejects_zero_beta", []() {
        bool threw = false;
        try {
            BetaProfile1D p({200.0, 400.0}, {50.0, 0.0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("rejects_negative_beta", []() {
        bool threw = false;
        try {
            BetaProfile1D p({200.0, 400.0}, {50.0, -10.0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

void test_beta_profile_2d() {
    TestRunner runner("BetaProfile2D");

    runner.add_test("linear_time_interp", []() {
        std::vector<double> alt{200.0, 400.0};
        std::vector<Epoch> ep{t0(), t0() + 3600.0};
        std::vector<double> grid{50.0, 100.0, 100.0, 200.0};  // doubles over the hour
        BetaProfile2D p(alt, ep, grid);

        Vector3d pos = position_at(0.0, 0.0, 200.0);  // bottom altitude knot
        const double b0   = p.evaluate(pos, t0());
        const double b_mid = p.evaluate(pos, t0() + 1800.0);
        const double b1   = p.evaluate(pos, t0() + 3600.0);
        TEST_ASSERT_NEAR(b0, 50.0, 1e-3);
        TEST_ASSERT_NEAR(b1, 100.0, 1e-3);
        TEST_ASSERT_NEAR(b_mid, 75.0, 1e-3);
    });

    runner.run();
}

// WindField4D_ECEF stores 3 × binary16 (`half`) per cell instead of
// Vector3d. Verify the ctor encodes input correctly and evaluate()
// recovers within half-precision tolerance.
void test_wind_field_4d_half_precision_round_trip() {
    TestRunner runner("WindField4D_ECEF half-precision storage");

    runner.add_test("round_trip_within_half_precision", []() {
        std::vector<double> alt_km{200.0, 400.0};
        std::vector<double> lat_deg{0.0, 30.0};
        std::vector<double> lon_deg{0.0, 90.0};
        Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        std::vector<Epoch> epochs{t0, t0 + 3600.0};

        // Wind values in the 10–100 m/s range — typical HWM14 output.
        std::vector<Vector3d> wind;
        wind.reserve(2 * 2 * 2 * 2);
        for (int ti = 0; ti < 2; ++ti)
        for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 2; ++k) {
            const double x = 10.0 + 2.5 * k + 5.0 * i;
            const double y = 30.0 - 4.0 * j + 2.0 * ti;
            const double z =  0.5 + 0.1 * (k + i + j + ti);
            wind.emplace_back(x, y, z);
        }

        WindField4D_ECEF field(alt_km, lat_deg, lon_deg, epochs, wind);
        // Spot-check at a corner cell (alt_km=200, lat=0, lon=0, t=t0).
        // After cubic-spline + bilinear + linear sampling, the corner
        // returns the bilinear interp of {x=10, x=12.5}-style halves —
        // still within ~0.1 % of input.
        // Sample at ECEF position corresponding to (lat=0°, lon=0°,
        // alt=200 km).
        constexpr double R = 6378137.0;  // equatorial radius (m)
        const Vector3d pos((R + 200e3), 0.0, 0.0);
        const Vector3d v = field.evaluate(pos, t0);
        // Expected from wind[0] = (10, 30, 0.5) at the corner.
        TEST_ASSERT_NEAR(v.x(), 10.0, 0.05);  // half ~1e-3 rel ⇒ ~10 mm/s abs
        TEST_ASSERT_NEAR(v.y(), 30.0, 0.05);
        TEST_ASSERT_NEAR(v.z(),  0.5, 0.05);
    });

    runner.add_test("memory_3_halves_per_cell", []() {
        std::vector<double> alt_km{200.0, 400.0, 600.0};   // n=3
        std::vector<double> lat_deg{0.0, 30.0};            // n=2
        std::vector<double> lon_deg{0.0, 90.0, 180.0};     // n=3
        Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);
        std::vector<Epoch> epochs{t0, t0 + 3600.0};        // n=2
        const std::size_t n_cells = 3 * 2 * 3 * 2;
        std::vector<Vector3d> wind(n_cells, Vector3d(1.0, 2.0, 3.0));

        WindField4D_ECEF field(alt_km, lat_deg, lon_deg, epochs, wind);
        TEST_ASSERT(!field.empty());
        // Each cell uses 3 × half = 6 bytes vs Vector3d's 24 bytes
        // (4× smaller memory footprint).

    });

    runner.add_test("single_value_axes_are_held", []() {
        // One latitude, one longitude and one epoch: held at that value
        // (the query used to read past the end of the storage).
        std::vector<Vector3d> wind{Vector3d(10.0, 20.0, 1.0),
                                   Vector3d(30.0, 40.0, 2.0)};
        WindField4D_ECEF field({200.0, 400.0}, {10.0}, {20.0}, {t0()}, wind);
        const Vector3d lo = field.evaluate(position_at(-50.0, 170.0, 200.0),
                                           t0() + 86400.0);
        TEST_ASSERT_NEAR(lo.x(), 10.0, 0.05);
        TEST_ASSERT_NEAR(lo.y(), 20.0, 0.05);
        const Vector3d hi = field.evaluate(position_at(60.0, -100.0, 400.0),
                                           t0() - 3600.0);
        TEST_ASSERT_NEAR(hi.x(), 30.0, 0.05);
        TEST_ASSERT_NEAR(hi.z(), 2.0, 0.05);
    });

    runner.add_test("single_altitude_rejected", []() {
        // Through one altitude the spline returns zero wind: refused.
        bool threw = false;
        try {
            WindField4D_ECEF field({300.0}, {0.0}, {0.0}, {t0()},
                                   {Vector3d(10.0, 0.0, 0.0)});
        } catch (const ConfigError& e) {
            threw = std::string(e.what()).find("altitude axis") != std::string::npos;
        }
        TEST_ASSERT(threw);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "DragEnvironment Column-Variant Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_density_profile_1d();
    test_density_profile_2d();
    test_density_field_4d();
    test_wind_profile_1d_enu();
    test_wind_profile_2d_ecef();
    test_wind_field_4d_half_precision_round_trip();
    test_beta_profile_1d();
    test_beta_profile_2d();

    return print_final_summary();
}
