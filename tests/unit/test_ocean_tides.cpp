// Tests for the time-variable part of Earth's gravity field: the Doodson arguments
// (the six astronomical angles that every tidal constituent is a combination of),
// the IERS 2010 Conventions "Step-2" solid-tide corrections (frequency-dependent
// corrections to the tidal change of the gravity coefficients, Tables 6.5a-c), and
// the ocean-tide model in the FES2004 file format (IERS 2010 eq. 6.15), plus their
// wiring into ForceModel. The ocean-tide cells use a tiny hand-written file, so
// expected values are computed in closed form and checked to rounding.

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/gravity/ocean_tides.hpp>
#include <ballistic/gravity/tide_arguments.hpp>
#include <ballistic/forces/force_model.hpp>
#include <filesystem>
#include <fstream>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::gravity;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Period [h] of a Doodson constituent from a 1-minute finite difference.
double period_hours(const std::array<int, 6>& n, const Epoch& t) {
    const auto b1 = doodson_arguments(t), b2 = doodson_arguments(t + 60.0);
    return 2.0 * kPi / std::abs(doodson_angle(n, b2) - doodson_angle(n, b1)) / 60.0;
}

std::filesystem::path write_tide_file() {
    auto p = std::filesystem::temp_directory_path() / "ocean_tides_test.dat";
    std::ofstream f(p);
    f << "Coefficients to compute variations in normalized Stokes coefficients (unit = 10^-11)\n";
    f << "Doodson Darw  l   m    DelC+     DelS+       DelC-     DelS-\n";
    f << "255.555 M2    2   2   1.00000   2.00000     3.00000   4.00000\n";
    f << " 55.565 Om1   2   0  -6.58128   0.00000    -0.00000  -0.00000\n";
    f << "165.555 K1   99   1   9.0       9.0         9.0       9.0\n";   // above max_degree
    return p;
}

}  // namespace

int main() {
    initialize({});
    TestRunner runner("OceanTidesAndStep2");
    const Epoch t0 = Epoch::from_utc(2025, 6, 1, 0, 0, 0);

    // Periods of the main constituents from the Doodson arguments match the
    // textbook values to 1e-3 h (the finite difference is the only error).
    runner.add_test("doodson_periods", [&]() {
        TEST_ASSERT_NEAR(period_hours({2, 0, 0, 0, 0, 0}, t0), 12.4206, 1e-3);   // M2
        TEST_ASSERT_NEAR(period_hours({2, 2, -2, 0, 0, 0}, t0), 12.0000, 1e-3);  // S2
        TEST_ASSERT_NEAR(period_hours({1, 1, 0, 0, 0, 0}, t0), 23.9345, 1e-3);   // K1
        TEST_ASSERT_NEAR(period_hours({1, -1, 0, 0, 0, 0}, t0), 25.8193, 1e-3);  // O1
    });

    // The tables have the IERS row counts (Tables 6.5b / 6.5a / 6.5c: 21 / 48 /
    // 2), no constituent appears twice, rows run in the published frequency
    // order (Doodson number ascending), and the K1 row has the published
    // in-phase / out-of-phase amplitudes.
    runner.add_test("step2_tables_match_iers", []() {
        TEST_ASSERT_EQ(std::size(kStep2LongPeriod), size_t{21});
        TEST_ASSERT_EQ(std::size(kStep2Diurnal), size_t{48});
        TEST_ASSERT_EQ(std::size(kStep2Semidiurnal), size_t{2});
        auto doodson_number = [](const std::array<int, 6>& n) {
            // tau s h p N' ps -> digits n1 (n2+5) (n3+5) (n4+5) (n5+5) (n6+5)
            long d = n[0];
            for (int i = 1; i < 6; ++i) d = 10 * d + (n[i] + 5);
            return d;
        };
        auto check_order = [&](const auto& table) {
            for (std::size_t i = 1; i < std::size(table); ++i)
                TEST_ASSERT(doodson_number(table[i - 1].n) < doodson_number(table[i].n));
        };
        check_order(kStep2LongPeriod);
        check_order(kStep2Diurnal);
        check_order(kStep2Semidiurnal);
        // first and last diurnal rows: 2Q1 (125,755) and 195,465
        TEST_ASSERT_EQ(doodson_number(kStep2Diurnal[0].n), 125755L);
        TEST_ASSERT_EQ(doodson_number(kStep2Diurnal[47].n), 195465L);
        bool k1 = false;
        for (const auto& t : kStep2Diurnal)
            if (t.n == std::array<int, 6>{1, 1, 0, 0, 0, 0}) {
                k1 = true;
                TEST_ASSERT_NEAR(t.ip, 470.9, 1e-9);   // IERS worked example
                TEST_ASSERT_NEAR(t.op, -30.2, 1e-9);
            }
        TEST_ASSERT(k1);
    });

    // K1 dominates: over one day |dC21 + i dS21| stays near |A_K1| = 4.7e-10
    // (gate 3.0e-10 to 6.5e-10 leaves room for the smaller diurnal terms).
    runner.add_test("step2_diurnal_amplitude", [&]() {
        double lo = 1.0, hi = 0.0;
        for (int k = 0; k < 48; ++k) {
            double dC[4][4] = {}, dS[4][4] = {};
            add_step2_corrections(doodson_arguments(t0 + 1800.0 * k), dC, dS);
            const double a = std::hypot(dC[2][1], dS[2][1]);
            lo = std::min(lo, a); hi = std::max(hi, a);
        }
        TEST_ASSERT(lo > 3.0e-10);
        TEST_ASSERT(hi < 6.5e-10);
    });

    // The parser reads the file, ignores rows above the maximum degree, and the
    // Stokes-coefficient changes follow eq. 6.15 exactly for the M2 and Om1 rows.
    runner.add_test("ocean_file_parsing_and_eq_6_15", [&]() {
        auto& m = OceanTideModel::instance();
        m.clear();
        const auto p = write_tide_file();
        TEST_ASSERT(m.load(p, 50));
        TEST_ASSERT_EQ(m.max_degree(), 2);
        const auto beta = doodson_arguments(t0);
        std::vector<double> dC, dS;
        m.stokes(beta, 2, dC, dS);
        const double th = 2.0 * beta[0];                          // M2
        const double c = std::cos(th), s = std::sin(th);
        TEST_ASSERT_NEAR(dC[2 * 3 + 2], 1e-11 * ((1 + 3) * c + (2 + 4) * s), 1e-24);
        TEST_ASSERT_NEAR(dS[2 * 3 + 2], 1e-11 * ((2 - 4) * c - (1 - 3) * s), 1e-24);
        const double om = beta[4];                                // Om1 = N'
        TEST_ASSERT_NEAR(dC[2 * 3 + 0], -6.58128e-11 * std::cos(om), 1e-24);
        TEST_ASSERT_NEAR(dS[2 * 3 + 0], 0.0, 0.0);
        m.clear();
        std::filesystem::remove(p);
    });

    // Ocean tides act only once a file is loaded; a requested degree above the
    // file's raises; the force breakdown and the Jacobian path agree to 1e-12.
    runner.add_test("force_model_ocean_tides_gated_and_consistent", [&]() {
        auto& m = OceanTideModel::instance();
        m.clear();
        forces::ForceModel fm;
        const StateVector s(Vector3d(7078137.0, 0.0, 0.0), Vector3d(0.0, 1000.0, 7430.0));
        TEST_ASSERT(!fm.ocean_tides_active());
        TEST_ASSERT_NEAR(fm.compute_force_breakdown(t0, s).ocean_tides.norm(), 0.0, 0.0);

        const auto p = write_tide_file();
        TEST_ASSERT(m.load(p, 50));
        TEST_ASSERT(fm.ocean_tides_active());
        // The file stops at degree 2: the default degree 50 raises instead of
        // being silently reduced.
        TEST_ASSERT_THROWS(fm.compute_force_breakdown(t0, s), ConfigError);
        fm.set_ocean_tide_degree(m.max_degree());
        const auto fb = fm.compute_force_breakdown(t0, s);
        TEST_ASSERT(fb.ocean_tides.norm() > 1e-12);
        // The Jacobian path must give the same total acceleration.
        const auto [a_j, J] = fm.acceleration_with_jacobian(t0, s);
        TEST_ASSERT_VEC3_NEAR(a_j, fb.total, 1e-12);
        fm.set_ocean_tide_degree(0);
        TEST_ASSERT(!fm.ocean_tides_active());
        m.clear();
        std::filesystem::remove(p);
    });

    runner.run();
    return print_final_summary();
}
