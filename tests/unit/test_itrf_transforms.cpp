// Unit tests for transformations between realizations of the International
// Terrestrial Reference Frame (ITRF2000, 2005, 2008, 2014, 2020: successive
// solutions for the Earth-fixed frame, differing by millimeters). Each step is a
// 7-parameter Helmert transformation (translation T, scale D, rotation R, each
// with a yearly rate). Parameter values are checked against the IERS published
// tables; displacements against their expected millimeter size. No data files.
// Tests: coordinates/itrf_transforms.hpp

#include "../test_utils.hpp"
#include <ballistic/coordinates/itrf_transforms.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::coords;

// =============================================================================
// HelmertParams::at_epoch Tests
// =============================================================================

void test_helmert_at_epoch() {
    TestRunner runner("HelmertAtEpoch");

    runner.add_test("reference_epoch_dt_zero", []() {
        // At reference epoch, dt=0, so params equal base values
        auto params = itrf_params::itrf2020_to_itrf2014();
        Vector3d T;
        double D;
        Vector3d R;

        params.at_epoch(params.t0, T, D, R);

        // T should be base values converted mm -> m
        TEST_ASSERT_NEAR(T(0), params.Tx * 1e-3, 1e-15);
        TEST_ASSERT_NEAR(T(1), params.Ty * 1e-3, 1e-15);
        TEST_ASSERT_NEAR(T(2), params.Tz * 1e-3, 1e-15);

        // D: ppb -> dimensionless
        TEST_ASSERT_NEAR(D, params.D * 1e-9, 1e-20);
    });

    runner.add_test("linear_extrapolation", []() {
        auto params = itrf_params::itrf2020_to_itrf2014();
        Vector3d T0, T1;
        double D0, D1;
        Vector3d R0, R1;

        double dt = 5.0;  // 5 years
        params.at_epoch(params.t0, T0, D0, R0);
        params.at_epoch(params.t0 + dt, T1, D1, R1);

        // Check that translation changed by dT*dt (mm->m)
        TEST_ASSERT_NEAR(T1(0), T0(0) + params.dTx * dt * 1e-3, 1e-15);
        TEST_ASSERT_NEAR(T1(1), T0(1) + params.dTy * dt * 1e-3, 1e-15);
        TEST_ASSERT_NEAR(T1(2), T0(2) + params.dTz * dt * 1e-3, 1e-15);
    });

    runner.add_test("unit_conversions_ppb", []() {
        auto params = itrf_params::itrf2020_to_itrf2014();
        Vector3d T;
        double D;
        Vector3d R;

        params.at_epoch(params.t0, T, D, R);

        // D = -0.42 ppb = -0.42e-9
        TEST_ASSERT_NEAR(D, -0.42e-9, 1e-20);
    });

    runner.add_test("unit_conversions_mas_to_rad", []() {
        // The published edges carry no rotation; a constructed one checks the
        // mas -> rad conversion and its rate
        HelmertParams p{0, 0, 0, 0, 0.0, 0.0, 0.02, 0, 0, 0, 0, 0.0, 0.0, 0.003, 2010.0};
        Vector3d T;
        double D;
        Vector3d R;

        p.at_epoch(2020.0, T, D, R);

        // Rz = 0.02 + 0.003 * 10 mas -> radians
        double MAS_TO_RAD = constants::PI / (180.0 * 3600.0 * 1000.0);
        TEST_ASSERT_NEAR(R(2), 0.05 * MAS_TO_RAD, 1e-20);
    });

    runner.run();
}

// =============================================================================
// transform_itrf Tests
// =============================================================================

void test_transform_itrf() {
    TestRunner runner("TransformITRF");

    runner.add_test("identity_same_realization", []() {
        Vector3d pos(6378137.0, 0.0, 0.0);
        Vector3d result = transform_itrf(pos, ITRFRealization::ITRF2014, ITRFRealization::ITRF2014, 2020.0);
        TEST_ASSERT_VEC3_NEAR(result, pos, 1e-15);
    });

    runner.add_test("identity_all_realizations", []() {
        Vector3d pos(6378137.0, 1000000.0, 5000000.0);
        std::vector<ITRFRealization> frames = {
            ITRFRealization::ITRF2000,
            ITRFRealization::ITRF2005,
            ITRFRealization::ITRF2008,
            ITRFRealization::ITRF2014,
            ITRFRealization::ITRF2020,
        };

        for (auto frame : frames) {
            Vector3d result = transform_itrf(pos, frame, frame, 2020.0);
            TEST_ASSERT_VEC3_NEAR(result, pos, 1e-15);
        }
    });

    runner.add_test("itrf2014_to_itrf2020_mm_displacement", []() {
        // Displacement between ITRF2014 and ITRF2020 should be ~1-3 mm
        Vector3d pos(6378137.0, 0.0, 0.0);  // on the equator at the surface
        Vector3d result = transform_itrf(pos, ITRFRealization::ITRF2014, ITRFRealization::ITRF2020, 2015.0);

        double displacement = (result - pos).norm();
        // Should be mm-level
        TEST_ASSERT(displacement > 1e-4);   // > 0.1 mm
        TEST_ASSERT(displacement < 0.01);   // < 10 mm
    });

    runner.add_test("roundtrip_itrf2014_itrf2020", []() {
        Vector3d pos(6378137.0, 1000000.0, 5000000.0);
        double epoch = 2020.0;

        Vector3d forward = transform_itrf(pos, ITRFRealization::ITRF2014, ITRFRealization::ITRF2020, epoch);
        Vector3d roundtrip = transform_itrf(forward, ITRFRealization::ITRF2020, ITRFRealization::ITRF2014, epoch);

        // The forward and inverse transforms cancel to better than a micrometer.
        TEST_ASSERT_VEC3_NEAR(roundtrip, pos, 1e-6);
    });

    runner.add_test("roundtrip_itrf2008_itrf2020", []() {
        Vector3d pos(6378137.0, 2000000.0, 4000000.0);
        double epoch = 2020.0;

        Vector3d forward = transform_itrf(pos, ITRFRealization::ITRF2008, ITRFRealization::ITRF2020, epoch);
        Vector3d roundtrip = transform_itrf(forward, ITRFRealization::ITRF2020, ITRFRealization::ITRF2008, epoch);

        TEST_ASSERT_VEC3_NEAR(roundtrip, pos, 1e-5);  // Micrometer level
    });

    runner.add_test("chained_itrf2005_to_itrf2020", []() {
        // ITRF2005 -> ITRF2020 goes through ITRF2014 -> ITRF2020
        Vector3d pos(6378137.0, 0.0, 0.0);
        double epoch = 2015.0;

        Vector3d result = transform_itrf(pos, ITRFRealization::ITRF2005, ITRFRealization::ITRF2020, epoch);
        double displacement = (result - pos).norm();

        // Should be a few mm
        TEST_ASSERT(displacement > 1e-4);
        TEST_ASSERT(displacement < 0.1);  // < 100 mm
    });

    runner.add_test("itrf2000_to_itrf2020", []() {
        // ITRF2000 -> ITRF2014 -> ITRF2020
        Vector3d pos(6378137.0, 0.0, 0.0);
        double epoch = 2015.0;

        Vector3d result = transform_itrf(pos, ITRFRealization::ITRF2000, ITRFRealization::ITRF2020, epoch);
        double displacement = (result - pos).norm();

        // ITRF2000 to ITRF2020 has larger translation (~26mm Tz), so displacement is larger
        TEST_ASSERT(displacement > 1e-3);  // > 1 mm
        TEST_ASSERT(displacement < 1.0);   // < 1 m
    });

    runner.add_test("epoch_dependence", []() {
        // Transform should change with epoch due to rates
        Vector3d pos(6378137.0, 0.0, 0.0);

        Vector3d r1 = transform_itrf(pos, ITRFRealization::ITRF2014, ITRFRealization::ITRF2020, 2015.0);
        Vector3d r2 = transform_itrf(pos, ITRFRealization::ITRF2014, ITRFRealization::ITRF2020, 2025.0);

        // Should be different due to non-zero rates
        double diff = (r2 - r1).norm();
        // dTy=-0.1 mm/yr, dTz=0.2 mm/yr -> over 10 years ~2mm change
        TEST_ASSERT(diff > 1e-5);  // At least some change
    });

    runner.add_test("transform_magnitude_function", []() {
        double mag = itrf_transform_magnitude(
            ITRFRealization::ITRF2014, ITRFRealization::ITRF2020, 2020.0);
        // Should be mm-level
        TEST_ASSERT(mag > 1e-4);
        TEST_ASSERT(mag < 0.01);
    });

    runner.add_test("to_decimal_year", []() {
        double dy = to_decimal_year(2020, 1, 1);
        TEST_ASSERT_NEAR(dy, 2020.0, 0.01);

        double dy_mid = to_decimal_year(2020, 7, 1);
        TEST_ASSERT_NEAR(dy_mid, 2020.5, 0.05);
    });

    runner.run();
}

// =============================================================================
// Helmert Parameter Value Verification
// =============================================================================
// Every edge pinned to the IERS ITRS Centre tables, which give the
// transformation FROM the newer frame TO the older one
// (X_old = X_new + T + D*X_new + R x X_new):
//   https://itrf.ign.fr/docs/solutions/itrf2020/Transfo-ITRF2020_TRFs.txt
//     ("Transformation parameters from ITRF2020 to past ITRFs")
//   https://itrf.ign.fr/doc_ITRF/Transfo-ITRF2014_ITRFs.txt
//     ("Transformation parameters from ITRF2014 to past ITRFs")
// Units: mm, ppb, mas (0.001"), and the same per year. The code once held
// these rows under older->newer names, i.e. with every sign flipped, and two
// rows carried rotations (Rz 0.02 and 0.06 mas) that the tables do not have.

namespace {
struct Row { double T[3], D, R[3], dT[3], dD, dR[3], t0; };
void check_row(const HelmertParams& p, const Row& r) {
    TEST_ASSERT_NEAR(p.Tx, r.T[0], 1e-12);
    TEST_ASSERT_NEAR(p.Ty, r.T[1], 1e-12);
    TEST_ASSERT_NEAR(p.Tz, r.T[2], 1e-12);
    TEST_ASSERT_NEAR(p.D, r.D, 1e-12);
    TEST_ASSERT_NEAR(p.Rx, r.R[0], 1e-12);
    TEST_ASSERT_NEAR(p.Ry, r.R[1], 1e-12);
    TEST_ASSERT_NEAR(p.Rz, r.R[2], 1e-12);
    TEST_ASSERT_NEAR(p.dTx, r.dT[0], 1e-12);
    TEST_ASSERT_NEAR(p.dTy, r.dT[1], 1e-12);
    TEST_ASSERT_NEAR(p.dTz, r.dT[2], 1e-12);
    TEST_ASSERT_NEAR(p.dD, r.dD, 1e-12);
    TEST_ASSERT_NEAR(p.dRx, r.dR[0], 1e-12);
    TEST_ASSERT_NEAR(p.dRy, r.dR[1], 1e-12);
    TEST_ASSERT_NEAR(p.dRz, r.dR[2], 1e-12);
    TEST_ASSERT_NEAR(p.t0, r.t0, 1e-12);
}
}  // namespace

void test_helmert_parameter_values() {
    TestRunner runner("HelmertParameterValues");

    runner.add_test("itrf2020_to_itrf2014_matches_iers_table", []() {
        // ITRF2020 table, row ITRF2014
        check_row(itrf_params::itrf2020_to_itrf2014(),
                  {{-1.4, -0.9, 1.4}, -0.42, {0, 0, 0}, {0.0, -0.1, 0.2}, 0.0, {0, 0, 0}, 2015.0});
    });

    runner.add_test("itrf2014_to_itrf2008_matches_iers_table", []() {
        // ITRF2014 table, row ITRF2008
        check_row(itrf_params::itrf2014_to_itrf2008(),
                  {{1.6, 1.9, 2.4}, -0.02, {0, 0, 0}, {0.0, 0.0, -0.1}, 0.03, {0, 0, 0}, 2010.0});
    });

    runner.add_test("itrf2014_to_itrf2005_matches_iers_table", []() {
        // ITRF2014 table, row ITRF2005
        check_row(itrf_params::itrf2014_to_itrf2005(),
                  {{2.6, 1.0, -2.3}, 0.92, {0, 0, 0}, {0.3, 0.0, -0.1}, 0.03, {0, 0, 0}, 2010.0});
    });

    runner.add_test("itrf2014_to_itrf2000_matches_iers_table", []() {
        // ITRF2014 table, row ITRF2000
        check_row(itrf_params::itrf2014_to_itrf2000(),
                  {{0.7, 1.2, -26.1}, 2.12, {0, 0, 0}, {0.1, 0.1, -1.9}, 0.11, {0, 0, 0}, 2010.0});
    });

    // The direction: transform_itrf(ITRF2020 -> ITRF2014) applies the table row
    // as published (X_2014 = X_2020 + T + D*X_2020), and the reverse call
    // undoes it.
    runner.add_test("itrf2020_to_itrf2014_applies_the_published_direction", []() {
        const Vector3d x2020(4027894.006, 307045.600, 4919474.910);   // a station-like point
        const double t = 2015.0;  // reference epoch: rates drop out
        const Vector3d expected = x2020 + Vector3d(-1.4e-3, -0.9e-3, 1.4e-3) - 0.42e-9 * x2020;
        const Vector3d x2014 = transform_itrf(x2020, ITRFRealization::ITRF2020,
                                              ITRFRealization::ITRF2014, t);
        TEST_ASSERT_VEC3_NEAR(x2014, expected, 1e-9);
        // and the reverse direction is its inverse to second order (< 1e-12 m here)
        const Vector3d back = transform_itrf(x2014, ITRFRealization::ITRF2014,
                                             ITRFRealization::ITRF2020, t);
        TEST_ASSERT_VEC3_NEAR(back, x2020, 1e-9);
    });

    // Round trip through every edge and the ITRF2014 hub, at a LEO radius.
    runner.add_test("roundtrip_every_pair", []() {
        const Vector3d pos(6878137.0 * 0.6, 6878137.0 * 0.48, 6878137.0 * 0.64);
        const std::vector<ITRFRealization> frames = {
            ITRFRealization::ITRF2000, ITRFRealization::ITRF2005, ITRFRealization::ITRF2008,
            ITRFRealization::ITRF2014, ITRFRealization::ITRF2020};
        for (auto a : frames) {
            for (auto b : frames) {
                const Vector3d there = transform_itrf(pos, a, b, 2025.0);
                const Vector3d back = transform_itrf(there, b, a, 2025.0);
                TEST_ASSERT_VEC3_NEAR(back, pos, 1e-9);
            }
        }
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "ITRF Transforms Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_helmert_at_epoch();
    test_transform_itrf();
    test_helmert_parameter_values();

    return print_final_summary();
}
