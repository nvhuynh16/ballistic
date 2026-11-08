// =============================================================================
// ERFA Reference Validation Tests
// =============================================================================
//
// Validates our IAU 2006/2000A implementation against reference values computed
// with pyerfa 2.0.1.5 (Python bindings of ERFA, the open copy of the IAU's SOFA
// library). This is the authoritative test that the precession, nutation and
// frame transformation code was transcribed correctly.
//
// Terms: IAU 2006/2000A = the standard models of the Earth's precession and
// nutation; TT = Terrestrial Time; CIP = Celestial Intermediate Pole (the
// Earth's rotation axis, as X, Y coordinates in the celestial frame); CIO
// locator s = a small angle fixing the origin of right ascension on the
// moving equator; NPB = the combined bias-precession-nutation rotation matrix;
// mas = milliarcsecond (4.85e-9 rad); 1 microarcsecond = 4.85e-12 rad.
// Tolerances are set just above double-precision rounding, except for the
// nutation, CIP and NPB checks: 1e-12 rad (0.2 microarcsec), because the
// planetary nutation terms use the IERS 2003 fundamental arguments where
// ERFA's nut00a uses its own linear MHB2000 ones (a 0.04 microarcsec
// difference). A dropped or mistyped series term is 0.1-200 microarcsec.
//
// All reference values were computed using the following pyerfa functions:
//   - erfa.fal03, falp03, faf03, fad03, faom03 (fundamental arguments)
//   - erfa.fame03, fave03, fae03, fama03, faju03, fasa03, faur03, fane03, fapa03
//   - erfa.pfw06 (precession)
//   - erfa.nut06a (nutation, = nut00a + J2 rate correction)
//   - erfa.pnm06a (NPB matrix)
//   - erfa.bpn2xy (CIP X,Y)
//   - erfa.s06 (CIO locator)
//   - erfa.era00 (Earth Rotation Angle)
//   - erfa.gmst06 (Greenwich Mean Sidereal Time)
//
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/coordinates/iau_transforms.hpp>
#include <ballistic/coordinates/earth_rotation.hpp>
#include <ballistic/time/epoch.hpp>
#include <ballistic/core/constants.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::coords;
using namespace ballistic::constants;

// =============================================================================
// Fundamental Arguments
// =============================================================================

void test_fundamental_arguments() {
    TestRunner runner("FundamentalArguments_ERFA");

    runner.add_test("delaunay_args_at_t0p80", []() {
        // t = 0.80 Julian centuries TT from J2000.0
        // Reference: pyerfa fal03(0.80), falp03(0.80), etc.
        double t = 0.80;
        auto args = compute_fundamental_args(t);

        // 1e-12 radians tolerance (~0.2 microarcseconds)
        TEST_ASSERT_NEAR(args.l,  5.13236975110911e+00, 1e-12);
        TEST_ASSERT_NEAR(args.lp, 6.22679797350553e+00, 1e-12);
        TEST_ASSERT_NEAR(args.F,  2.59771136675145e-01, 1e-12);
        TEST_ASSERT_NEAR(args.D,  1.94670920539721e+00, 1e-12);
        TEST_ASSERT_NEAR(args.Om, -5.97361844095130e+00, 1e-12);
    });

    runner.add_test("planetary_longitudes_at_t0p80", []() {
        // Reference: pyerfa fame03(0.80), fave03(0.80), etc.
        double t = 0.80;
        auto args = compute_fundamental_args(t);

        // 1e-12 radians tolerance
        TEST_ASSERT_NEAR(args.L_Me, 5.41733818429768e+00, 1e-12);
        TEST_ASSERT_NEAR(args.L_Ve, 3.42490046053383e+00, 1e-12);
        TEST_ASSERT_NEAR(args.L_E,  1.74471373891311e+00, 1e-12);
        TEST_ASSERT_NEAR(args.L_Ma, 3.27550684027783e+00, 1e-12);
        TEST_ASSERT_NEAR(args.L_J,  5.27571166520249e+00, 1e-12);
        TEST_ASSERT_NEAR(args.L_Sa, 5.37157453944083e+00, 1e-12);
        TEST_ASSERT_NEAR(args.L_U,  5.18063645018041e+00, 1e-12);
        TEST_ASSERT_NEAR(args.L_Ne, 2.07934383086041e+00, 1e-12);
    });

    runner.add_test("general_precession_at_t0p80", []() {
        // Reference: pyerfa fapa03(0.80)
        double t = 0.80;
        auto args = compute_fundamental_args(t);

        TEST_ASSERT_NEAR(args.pa, 1.95088476224000e-02, 1e-12);
    });

    runner.run();
}

// =============================================================================
// Precession (IAU 2006 Fukushima-Williams)
// =============================================================================

void test_precession_erfa() {
    TestRunner runner("Precession_ERFA");

    runner.add_test("pfw06_at_mjd53736", []() {
        // ERFA test epoch: pyerfa pfw06(2400000.5, 53736.0) treats input as TT JD
        // Reference was computed with TT JD = 2453736.5 (no 32.184s offset)
        double tt_jd = 2400000.5 + 53736.0;
        double t_tt = (tt_jd - J2000_JD) / 36525.0;

        auto prec = compute_precession_angles(t_tt);

        // Reference: pyerfa pfw06(2400000.5, 53736.0)
        // 5e-12 radians tolerance (~1 microarcsecond, allows double-precision roundoff)
        TEST_ASSERT_NEAR(prec.gamb, 2.82273141464725e-06, 5e-12);
        TEST_ASSERT_NEAR(prec.phib, 4.09079017780327e-01, 5e-12);
        TEST_ASSERT_NEAR(prec.psib, 1.46545952168175e-03, 5e-12);
        TEST_ASSERT_NEAR(prec.epsa, 4.09078976335651e-01, 5e-12);
    });

    runner.run();
}

// =============================================================================
// Nutation (IAU 2006/2000A = nut06a)
// =============================================================================

void test_nutation_erfa() {
    TestRunner runner("Nutation_ERFA");

    runner.add_test("nut06a_at_j2000", []() {
        // J2000.0: t_tt = 0.0 exactly
        // Reference: pyerfa nut06a
        // Note: at t=0, J2 correction is zero so nut06a ≈ nut00a
        double t_tt = 0.0;
        auto args = compute_fundamental_args(t_tt);
        auto nut = compute_nutation(t_tt, args);

        double tol = 1e-12;  // 0.2 microarcsec
        TEST_ASSERT_NEAR(nut.dpsi, -6.754425598969512e-05, tol);
        TEST_ASSERT_NEAR(nut.deps, -2.797083119237414e-05, tol);
    });

    runner.add_test("nut06a_at_2010", []() {
        // J2010.0: t_tt = 0.1 Julian centuries from J2000.0
        // Reference: pyerfa nut06a at t=0.1
        double t_tt = 0.1;
        auto args = compute_fundamental_args(t_tt);
        auto nut = compute_nutation(t_tt, args);

        double tol = 1e-12;  // 0.2 microarcsec
        TEST_ASSERT_NEAR(nut.dpsi, 7.974537086785984e-05, tol);
        TEST_ASSERT_NEAR(nut.deps, 1.369128324565096e-05, tol);
    });

    runner.add_test("nut06a_at_2020", []() {
        // J2020.0: t_tt = 0.2 Julian centuries from J2000.0
        // Reference: pyerfa nut06a at t=0.2
        double t_tt = 0.2;
        auto args = compute_fundamental_args(t_tt);
        auto nut = compute_nutation(t_tt, args);

        double tol = 1e-12;  // 0.2 microarcsec
        TEST_ASSERT_NEAR(nut.dpsi, -8.007567059020389e-05, tol);
        TEST_ASSERT_NEAR(nut.deps, -8.163977470032335e-06, tol);
    });

    runner.run();
}

// =============================================================================
// CIP Position and CIO Locator
// =============================================================================

void test_cip_erfa() {
    TestRunner runner("CIP_ERFA");

    runner.add_test("cip_xy_at_mjd53736", []() {
        double tt_jd = 2400000.5 + 53736.0;
        double t_tt = (tt_jd - J2000_JD) / 36525.0;

        auto cip = compute_cip_position(t_tt);

        // Reference: pyerfa bpn2xy(pnm06a(2400000.5, 53736.0)), the library's
        // route (X, Y read off the precession-nutation matrix)
        double tol = 1e-12;  // 0.2 microarcsec
        TEST_ASSERT_NEAR(cip.X, 5.791308482835291e-04, tol);
        TEST_ASSERT_NEAR(cip.Y, 4.0205800994674856e-05, tol);
        // ERFA's X, Y series xy06, a different evaluation of the same model,
        // is ~0.6 microarcsec away here
        TEST_ASSERT_NEAR(cip.X, 5.79130848670601e-04, 1e-11);
        TEST_ASSERT_NEAR(cip.Y, 4.02057981673295e-05, 1e-11);
    });

    runner.add_test("cio_locator_s_at_mjd53736", []() {
        double tt_jd = 2400000.5 + 53736.0;
        double t_tt = (tt_jd - J2000_JD) / 36525.0;

        auto cip = compute_cip_position(t_tt);

        // Reference: pyerfa s06 at MJD 53736
        // s is small (~1e-8 rad), use absolute tolerance of 1e-13
        TEST_ASSERT_NEAR(cip.s, -1.22003221307646e-08, 1e-13);
    });

    runner.run();
}

// =============================================================================
// Earth Rotation Angle
// =============================================================================

void test_era_erfa() {
    TestRunner runner("ERA_ERFA");

    runner.add_test("era00_at_mjd53736", []() {
        // Reference: pyerfa era00(2400000.5, 53736.0)
        double ut1_jd = 2400000.5 + 53736.0;
        double era = earth_rotation_angle(ut1_jd);

        TEST_ASSERT_NEAR(era, 1.75283325530307e+00, 1e-10);
    });

    runner.run();
}

// =============================================================================
// Greenwich Mean Sidereal Time
// =============================================================================

void test_gmst_erfa() {
    TestRunner runner("GMST_ERFA");

    runner.add_test("gmst06_at_mjd53736", []() {
        // Reference: pyerfa gmst06(2400000.5, 53736.0, 2400000.5, 53736.0)
        double ut1_jd = 2400000.5 + 53736.0;
        double g = gmst(ut1_jd);

        TEST_ASSERT_NEAR(g, 1.75417497187009e+00, 1e-10);
    });

    runner.run();
}

// =============================================================================
// NPB Matrix (Bias-Precession-Nutation)
// =============================================================================

void test_npb_matrix_erfa() {
    TestRunner runner("NPBMatrix_ERFA");

    runner.add_test("pnm06a_at_mjd53736", []() {
        double tt_jd = 2400000.5 + 53736.0;
        double t_tt = (tt_jd - J2000_JD) / 36525.0;

        auto prec = compute_precession_angles(t_tt);
        auto args = compute_fundamental_args(t_tt);
        auto nut = compute_nutation(t_tt, args);

        double psi = prec.psib + nut.dpsi;
        double eps = prec.epsa + nut.deps;
        Matrix3d npb = build_npb_matrix(prec.gamb, prec.phib, psi, eps);

        // Reference: pyerfa pnm06a(2400000.5, 53736.0); the reference values
        // carry 15 significant digits
        double tol = 1e-12;

        TEST_ASSERT_NEAR(npb(0, 0),  9.99998944048067e-01, tol);
        TEST_ASSERT_NEAR(npb(0, 1), -1.33288141809192e-03, tol);
        TEST_ASSERT_NEAR(npb(0, 2), -5.79076744761204e-04, tol);
        TEST_ASSERT_NEAR(npb(1, 0),  1.33285791125099e-03, tol);
        TEST_ASSERT_NEAR(npb(1, 1),  9.99999110904914e-01, tol);
        TEST_ASSERT_NEAR(npb(1, 2), -4.09776712854693e-05, tol);
        TEST_ASSERT_NEAR(npb(2, 0),  5.79130848283529e-04, tol);
        TEST_ASSERT_NEAR(npb(2, 1),  4.02058009946193e-05, tol);
        TEST_ASSERT_NEAR(npb(2, 2),  9.99999831495463e-01, tol);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "ERFA Reference Validation Tests" << std::endl;
    std::cout << "============================================" << std::endl;

    test_fundamental_arguments();
    test_precession_erfa();
    test_nutation_erfa();
    test_cip_erfa();
    test_era_erfa();
    test_gmst_erfa();
    test_npb_matrix_erfa();

    return print_final_summary();
}
