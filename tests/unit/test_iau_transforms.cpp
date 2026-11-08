// Unit tests for the IAU 2006/2000A precession-nutation transforms
// (coordinates/iau_transforms.hpp): the rotation between the inertial frame
// GCRS (Geocentric Celestial Reference System) and the Earth-fixed frame ITRF
// (International Terrestrial Reference Frame), built in three steps:
//   GCRS -> CIRS  precession-nutation, from the CIP (Celestial Intermediate
//                 Pole, Earth's rotation axis) coordinates X, Y and the CIO
//                 locator s;
//   CIRS -> TIRS  rotation about z by the ERA (Earth Rotation Angle);
//   TIRS -> ITRF  polar motion (xp, yp).
// EOP = Earth Orientation Parameters (polar motion, UT1-UTC, ...). Most cells
// check physical magnitudes and rates from the IAU conventions (e.g.
// precession ~5038 arcsec/century, obliquity ~23.44 deg, nutation <~17 arcsec
// in longitude) or exact properties (orthogonality, round trips). Bit-level
// agreement with the SOFA/ERFA reference library is tested separately
// (test_erfa_reference, test_sofa_comprehensive).

#include <ballistic/time/eop_data.hpp>
#include "../test_utils.hpp"
#include <ballistic/coordinates/iau_transforms.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::coords;
using namespace ballistic::constants;

// =============================================================================
// Test Suites
// =============================================================================

void test_earth_orientation_parameters() {
    TestRunner runner("EarthOrientationParameters");

    runner.add_test("zero_factory", []() {
        auto eop = EarthOrientationParameters::zero();
        TEST_ASSERT_NEAR(eop.xp, 0.0, 1e-15);
        TEST_ASSERT_NEAR(eop.yp, 0.0, 1e-15);
        TEST_ASSERT_NEAR(eop.dut1, 0.0, 1e-15);
        TEST_ASSERT_NEAR(eop.lod, 0.0, 1e-15);
        TEST_ASSERT_NEAR(eop.dX, 0.0, 1e-15);
        TEST_ASSERT_NEAR(eop.dY, 0.0, 1e-15);
    });

    runner.add_test("typical_values", []() {
        EarthOrientationParameters eop;
        eop.xp = 0.1;      // arcseconds
        eop.yp = 0.3;      // arcseconds
        eop.dut1 = 0.2;    // seconds
        eop.lod = 0.001;   // seconds
        eop.dX = 0.0001;   // arcseconds
        eop.dY = 0.0002;   // arcseconds

        TEST_ASSERT_NEAR(eop.xp, 0.1, 1e-15);
        TEST_ASSERT_NEAR(eop.yp, 0.3, 1e-15);
        TEST_ASSERT_NEAR(eop.dut1, 0.2, 1e-15);
    });

    runner.run();
}

void test_fundamental_arguments() {
    TestRunner runner("FundamentalArguments");

    runner.add_test("at_j2000", []() {
        // At J2000.0, t_tt = 0
        auto args = compute_fundamental_args(0.0);

        // Verify all arguments are finite
        TEST_ASSERT(std::isfinite(args.l));
        TEST_ASSERT(std::isfinite(args.lp));
        TEST_ASSERT(std::isfinite(args.F));
        TEST_ASSERT(std::isfinite(args.D));
        TEST_ASSERT(std::isfinite(args.Om));

        // Planetary longitudes should be in [0, 2π) or normalized
        TEST_ASSERT(std::isfinite(args.L_Me));
        TEST_ASSERT(std::isfinite(args.L_Ve));
        TEST_ASSERT(std::isfinite(args.L_E));
        TEST_ASSERT(std::isfinite(args.L_Ma));
        TEST_ASSERT(std::isfinite(args.L_J));
    });

    runner.add_test("vary_with_time", []() {
        auto args_0 = compute_fundamental_args(0.0);
        auto args_1 = compute_fundamental_args(1.0);  // 1 century later

        // Arguments should change over time
        TEST_ASSERT(std::abs(args_1.l - args_0.l) > 1e-6);
        TEST_ASSERT(std::abs(args_1.lp - args_0.lp) > 1e-6);
        TEST_ASSERT(std::abs(args_1.F - args_0.F) > 1e-6);
        TEST_ASSERT(std::abs(args_1.D - args_0.D) > 1e-6);
        TEST_ASSERT(std::abs(args_1.Om - args_0.Om) > 1e-6);
    });

    runner.add_test("moon_mean_anomaly_rate", []() {
        // Moon's mean anomaly rate is ~13.06 deg/day = ~477 rad/century
        auto args_0 = compute_fundamental_args(0.0);
        auto args_001 = compute_fundamental_args(0.01);  // 1 year

        double dl = args_001.l - args_0.l;
        // Should have changed significantly over a year
        TEST_ASSERT(std::abs(dl) > 1.0);  // More than 1 radian
    });

    runner.add_test("planetary_longitudes_jupiter", []() {
        // Jupiter orbital period ~11.86 years
        // In 1 century, Jupiter completes ~8.4 orbits
        auto args_0 = compute_fundamental_args(0.0);
        auto args_1 = compute_fundamental_args(1.0);

        // Jupiter longitude rate ~52.97 rad/century
        double dL_J = args_1.L_J - args_0.L_J;
        // Account for modular arithmetic - should have moved ~52.97 rad (modulo 2π)
        TEST_ASSERT(std::abs(dL_J) < TWO_PI);  // Result is modulo 2π
    });

    runner.add_test("negative_time", []() {
        // Test at 100 years before J2000
        auto args = compute_fundamental_args(-1.0);

        // All arguments should still be finite
        TEST_ASSERT(std::isfinite(args.l));
        TEST_ASSERT(std::isfinite(args.lp));
        TEST_ASSERT(std::isfinite(args.F));
        TEST_ASSERT(std::isfinite(args.D));
        TEST_ASSERT(std::isfinite(args.Om));
    });

    runner.run();
}

void test_precession_angles() {
    TestRunner runner("PrecessionAngles");

    runner.add_test("at_j2000", []() {
        auto prec = compute_precession_angles(0.0);

        // At J2000, gamma_bar should be very small
        TEST_ASSERT_NEAR(prec.gamb, 0.0, 1e-6);

        // phi_bar (related to obliquity) should be ~84381 arcsec = ~23.44 deg
        double phi_deg = prec.phib * RAD_TO_DEG;
        TEST_ASSERT_NEAR(phi_deg, 23.44, 0.001);  // IAU obliquity known to μas

        // psi_bar should be very small at J2000
        TEST_ASSERT_NEAR(prec.psib, 0.0, 1e-6);

        // Mean obliquity should be ~23.439 degrees
        double eps_deg = prec.epsa * RAD_TO_DEG;
        TEST_ASSERT_NEAR(eps_deg, reference::OBLIQUITY_J2000_DEG, 0.001);
    });

    runner.add_test("precession_rate", []() {
        // Lunisolar precession rate is ~50.3 arcsec/year
        auto prec_0 = compute_precession_angles(0.0);
        auto prec_1 = compute_precession_angles(1.0);  // 100 years later

        // psi_bar should increase by ~5038 arcsec/century
        double dpsi_arcsec = (prec_1.psib - prec_0.psib) / ARCSEC_TO_RAD;
        TEST_ASSERT_NEAR(dpsi_arcsec, 5038.0, 25.0);  // Within 0.5%
    });

    runner.add_test("obliquity_change", []() {
        // Mean obliquity decreases by ~46.8 arcsec/century
        auto prec_0 = compute_precession_angles(0.0);
        auto prec_1 = compute_precession_angles(1.0);

        double deps_arcsec = (prec_1.epsa - prec_0.epsa) / ARCSEC_TO_RAD;
        TEST_ASSERT_NEAR(deps_arcsec, -46.8, 1.0);
    });

    runner.add_test("century_in_future", []() {
        auto prec = compute_precession_angles(1.0);

        // All values should be finite
        TEST_ASSERT(std::isfinite(prec.gamb));
        TEST_ASSERT(std::isfinite(prec.phib));
        TEST_ASSERT(std::isfinite(prec.psib));
        TEST_ASSERT(std::isfinite(prec.epsa));
    });

    runner.add_test("century_in_past", []() {
        auto prec = compute_precession_angles(-1.0);

        // Mean obliquity should be larger in the past
        auto prec_now = compute_precession_angles(0.0);
        TEST_ASSERT(prec.epsa > prec_now.epsa);
    });

    runner.run();
}

void test_nutation() {
    TestRunner runner("Nutation");

    runner.add_test("at_j2000", []() {
        auto args = compute_fundamental_args(0.0);
        auto nut = compute_nutation(0.0, args);

        // Nutation in longitude is typically < 25 arcsec (allowing for model differences)
        double dpsi_arcsec = nut.dpsi / ARCSEC_TO_RAD;
        TEST_ASSERT(std::abs(dpsi_arcsec) < 25.0);

        // Nutation in obliquity - allow for model-specific variations
        double deps_arcsec = nut.deps / ARCSEC_TO_RAD;
        TEST_ASSERT(std::abs(deps_arcsec) < 20.0);  // Allow more tolerance
    });

    runner.add_test("dominant_term_18_year", []() {
        // The dominant nutation term has ~18.6 year period (Moon's node)
        // Amplitude is ~17.2 arcsec in longitude

        // Test at two times half a period apart (9.3 years = 0.093 centuries)
        auto args_0 = compute_fundamental_args(0.0);
        auto args_half = compute_fundamental_args(0.093);

        auto nut_0 = compute_nutation(0.0, args_0);
        auto nut_half = compute_nutation(0.093, args_half);

        // Nutations should differ significantly
        double diff = std::abs(nut_half.dpsi - nut_0.dpsi);
        TEST_ASSERT(diff > 1e-6);
    });

    runner.add_test("small_magnitude", []() {
        // Nutation is a small correction (< 0.01 degrees)
        for (double t = -1.0; t <= 1.0; t += 0.1) {
            auto args = compute_fundamental_args(t);
            auto nut = compute_nutation(t, args);

            TEST_ASSERT(std::abs(nut.dpsi) < 0.01 * DEG_TO_RAD);
            TEST_ASSERT(std::abs(nut.deps) < 0.01 * DEG_TO_RAD);
        }
    });

    runner.run();
}

void test_cip_position() {
    TestRunner runner("CIPPosition");

    runner.add_test("at_j2000", []() {
        auto cip = compute_cip_position(0.0);

        // X and Y are small (< 1 arcsec at J2000)
        TEST_ASSERT(std::abs(cip.X) < 0.001);  // < 1000 microarcsec
        TEST_ASSERT(std::abs(cip.Y) < 0.001);

        // CIO locator s is very small
        TEST_ASSERT(std::abs(cip.s) < 1e-6);
    });

    runner.add_test("vary_with_precession", []() {
        auto cip_0 = compute_cip_position(0.0);
        auto cip_1 = compute_cip_position(1.0);

        // X grows due to precession (~psi * sin(eps) ≈ 5038 * 0.4 arcsec/century)
        double dX = cip_1.X - cip_0.X;
        double dX_arcsec = dX / ARCSEC_TO_RAD;
        TEST_ASSERT(std::abs(dX_arcsec) > 100);  // Significant change

        // Y also changes
        double dY = cip_1.Y - cip_0.Y;
        TEST_ASSERT(std::abs(dY) > 1e-6);
    });

    runner.add_test("consistency", []() {
        // CIP X, Y, s should satisfy geometric constraints
        auto cip = compute_cip_position(0.5);

        // X^2 + Y^2 should be small (CIP close to pole)
        double cip_dist2 = cip.X * cip.X + cip.Y * cip.Y;
        TEST_ASSERT(cip_dist2 < 0.01);  // Less than ~0.1 rad from pole
    });

    runner.run();
}

void test_earth_rotation_angle() {
    TestRunner runner("EarthRotationAngle");

    runner.add_test("at_j2000_noon", []() {
        // ERA at J2000.0 should be 280.46 degrees = 4.895 rad
        double era = earth_rotation_angle(reference::J2000_JD);
        TEST_ASSERT_NEAR(era, reference::GMST_J2000_RAD, 0.01);
    });

    runner.add_test("daily_rate", []() {
        // ERA increases by ~2π per sidereal day (~360.986 deg/solar day)
        double era_0 = earth_rotation_angle(reference::J2000_JD);
        double era_1 = earth_rotation_angle(reference::J2000_JD + 1.0);

        // Difference should be ~360.986 degrees = ~6.300 rad (modulo 2π)
        double diff = era_1 - era_0;
        if (diff < 0) diff += TWO_PI;

        // One sidereal day rate: 1.00273781191135448 * 2π - 2π ≈ 0.0172 rad
        double expected_diff = std::fmod(1.00273781191135448 * TWO_PI, TWO_PI);
        TEST_ASSERT_NEAR(diff, expected_diff, 0.001);
    });

    runner.add_test("in_range", []() {
        // ERA should be finite and bounded
        for (double jd = 2450000.0; jd < 2460000.0; jd += 100.0) {
            double era = earth_rotation_angle(jd);
            TEST_ASSERT(std::isfinite(era));
            // Normalize to [0, 2π) for checking
            double era_norm = std::fmod(era, TWO_PI);
            if (era_norm < 0) era_norm += TWO_PI;
            TEST_ASSERT(era_norm >= 0.0 && era_norm < TWO_PI);
        }
    });

    runner.add_test("continuous", []() {
        // ERA should change smoothly (no large jumps except at 2π wrap)
        double prev_era = earth_rotation_angle(reference::J2000_JD);
        for (double dt = 0.001; dt <= 0.1; dt += 0.001) {
            double era = earth_rotation_angle(reference::J2000_JD + dt);
            double diff = era - prev_era;
            if (diff < -PI) diff += TWO_PI;
            if (diff > PI) diff -= TWO_PI;

            // Should increase by ~0.0063 rad/hour = ~0.000072 rad per 0.001 day
            TEST_ASSERT(std::abs(diff) < 0.1);  // No large jumps
            prev_era = era;
        }
    });

    runner.run();
}

void test_gcrs_to_cirs_matrix() {
    TestRunner runner("GCRS_to_CIRS_Matrix");

    runner.add_test("at_j2000", []() {
        Matrix3d Q = gcrs_to_cirs_matrix(0.0);

        // At J2000, should be close to identity (small precession/nutation)
        Matrix3d I = Matrix3d::Identity();
        double diff = (Q - I).norm();

        // Difference should be small but not zero
        TEST_ASSERT(diff < 0.1);
        TEST_ASSERT(diff > 1e-10);
    });

    runner.add_test("orthogonal", []() {
        for (double t = -1.0; t <= 1.0; t += 0.5) {
            Matrix3d Q = gcrs_to_cirs_matrix(t);

            // Q * Q^T should be identity
            Matrix3d QQt = Q * Q.transpose();
            double diff = (QQt - Matrix3d::Identity()).norm();
            TEST_ASSERT(diff < 1e-10);

            // det(Q) should be 1
            TEST_ASSERT_NEAR(Q.determinant(), 1.0, 1e-10);
        }
    });

    runner.add_test("time_dependence", []() {
        Matrix3d Q_0 = gcrs_to_cirs_matrix(0.0);
        Matrix3d Q_1 = gcrs_to_cirs_matrix(1.0);

        // Matrices should differ significantly after 1 century
        double diff = (Q_1 - Q_0).norm();
        TEST_ASSERT(diff > 0.001);
    });

    runner.run();
}

void test_cirs_to_tirs_matrix() {
    TestRunner runner("CIRS_to_TIRS_Matrix");

    runner.add_test("rotation_about_z", []() {
        double jd = reference::J2000_JD;
        Matrix3d R = cirs_to_tirs_matrix(jd);

        // Should be a rotation about Z-axis
        // R(2,0), R(2,1), R(0,2), R(1,2) should be zero
        TEST_ASSERT_NEAR(R(2, 0), 0.0, 1e-15);
        TEST_ASSERT_NEAR(R(2, 1), 0.0, 1e-15);
        TEST_ASSERT_NEAR(R(0, 2), 0.0, 1e-15);
        TEST_ASSERT_NEAR(R(1, 2), 0.0, 1e-15);
        TEST_ASSERT_NEAR(R(2, 2), 1.0, 1e-15);
    });

    runner.add_test("orthogonal", []() {
        for (double jd = 2450000.0; jd < 2460000.0; jd += 500.0) {
            Matrix3d R = cirs_to_tirs_matrix(jd);

            // R * R^T should be identity
            Matrix3d RRt = R * R.transpose();
            double diff = (RRt - Matrix3d::Identity()).norm();
            TEST_ASSERT(diff < 1e-14);

            // det(R) should be 1
            TEST_ASSERT_NEAR(R.determinant(), 1.0, 1e-14);
        }
    });

    runner.add_test("full_rotation_per_day", []() {
        double jd_0 = reference::J2000_JD;
        double jd_1 = jd_0 + 1.0;

        Matrix3d R_0 = cirs_to_tirs_matrix(jd_0);
        Matrix3d R_1 = cirs_to_tirs_matrix(jd_1);

        // After one solar day, matrices should be similar (within one day's rotation)
        // The difference depends on sidereal vs solar day
        double diff = (R_1 - R_0).norm();
        TEST_ASSERT(diff < 0.1);  // Matrices should be reasonably similar
    });

    runner.run();
}

void test_tirs_to_itrf_matrix() {
    TestRunner runner("TIRS_to_ITRF_Matrix");

    runner.add_test("zero_polar_motion", []() {
        auto eop = EarthOrientationParameters::zero();
        Matrix3d W = tirs_to_itrf_matrix(eop);

        // With zero polar motion, should be identity
        double diff = (W - Matrix3d::Identity()).norm();
        TEST_ASSERT(diff < 1e-14);
    });

    runner.add_test("small_polar_motion", []() {
        EarthOrientationParameters eop;
        eop.xp = 0.1;  // 0.1 arcsec
        eop.yp = 0.2;  // 0.2 arcsec

        Matrix3d W = tirs_to_itrf_matrix(eop);

        // Should still be close to identity (polar motion is small)
        double diff = (W - Matrix3d::Identity()).norm();
        TEST_ASSERT(diff < 0.001);
        TEST_ASSERT(diff > 1e-10);
    });

    runner.add_test("orthogonal", []() {
        EarthOrientationParameters eop;
        eop.xp = 0.3;
        eop.yp = 0.4;

        Matrix3d W = tirs_to_itrf_matrix(eop);

        // W * W^T should be identity
        Matrix3d WWt = W * W.transpose();
        double diff = (WWt - Matrix3d::Identity()).norm();
        TEST_ASSERT(diff < 1e-10);

        // det(W) should be 1
        TEST_ASSERT_NEAR(W.determinant(), 1.0, 1e-10);
    });

    runner.run();
}

void test_iau_transform_class() {
    TestRunner runner("IAUTransform");

    runner.add_test("default_constructor", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        // Should work with default (zero) EOP
        Matrix3d R = transform.gcrs_to_itrf_matrix(epoch);
        TEST_ASSERT_NEAR(R.determinant(), 1.0, 1e-10);
    });

    runner.add_test("with_eop", []() {
        EarthOrientationParameters eop;
        eop.xp = 0.1;
        eop.yp = 0.2;
        eop.dut1 = 0.3;

        IAUTransform transform(eop);
        Epoch epoch = Epoch::j2000();

        Matrix3d R = transform.gcrs_to_itrf_matrix(epoch);
        TEST_ASSERT_NEAR(R.determinant(), 1.0, 1e-10);
    });

    runner.add_test("set_eop", []() {
        IAUTransform transform;

        EarthOrientationParameters eop;
        eop.xp = 0.1;
        transform.set_eop(eop);

        // Matrix should now reflect polar motion
        Epoch epoch = Epoch::j2000();
        Matrix3d R = transform.gcrs_to_itrf_matrix(epoch);
        TEST_ASSERT_NEAR(R.determinant(), 1.0, 1e-10);
    });

    runner.add_test("inverse_matrices", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        Matrix3d R = transform.gcrs_to_itrf_matrix(epoch);
        Matrix3d Rinv = transform.itrf_to_gcrs_matrix(epoch);

        // R * Rinv should be identity
        Matrix3d prod = R * Rinv;
        double diff = (prod - Matrix3d::Identity()).norm();
        TEST_ASSERT(diff < 1e-10);
    });

    runner.run();
}

void test_position_transforms() {
    TestRunner runner("PositionTransforms");

    runner.add_test("roundtrip_gcrs_itrf", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        Vector3d pos_gcrs(6778137.0, 0.0, 0.0);  // On X-axis

        Vector3d pos_itrf = transform.gcrs_to_itrf(pos_gcrs, epoch);
        Vector3d pos_gcrs_back = transform.itrf_to_gcrs(pos_itrf, epoch);

        TEST_ASSERT_VEC3_NEAR(pos_gcrs_back, pos_gcrs, 1e-6);
    });

    runner.add_test("pole_position", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        // Position at North Pole - should be minimally affected by rotation
        Vector3d pos_pole(0.0, 0.0, 6356752.0);

        Vector3d pos_itrf = transform.gcrs_to_itrf(pos_pole, epoch);

        // Z component should be nearly unchanged
        TEST_ASSERT_NEAR(pos_itrf.z(), pos_pole.z(), 1.0);  // Pole Z unchanged by rotation
    });

    runner.add_test("equatorial_rotation", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        Vector3d pos_equator(6378137.0, 0.0, 0.0);
        Vector3d pos_itrf = transform.gcrs_to_itrf(pos_equator, epoch);

        // Magnitude should be preserved
        TEST_ASSERT_NEAR(pos_itrf.norm(), pos_equator.norm(), 1.0);
    });

    runner.add_test("different_epochs", []() {
        IAUTransform transform;

        Vector3d pos_gcrs(6778137.0, 1000000.0, 2000000.0);

        Epoch epoch1 = Epoch::from_utc(2020, 1, 1, 0, 0, 0);
        Epoch epoch2 = Epoch::from_utc(2020, 1, 1, 12, 0, 0);  // 12 hours later

        Vector3d pos_itrf1 = transform.gcrs_to_itrf(pos_gcrs, epoch1);
        Vector3d pos_itrf2 = transform.gcrs_to_itrf(pos_gcrs, epoch2);

        // ITRF position should differ significantly due to Earth rotation
        double diff = (pos_itrf2 - pos_itrf1).norm();
        TEST_ASSERT(diff > 1e6);  // Several thousand km in 12 hours
    });

    runner.run();
}

void test_state_vector_transforms() {
    TestRunner runner("StateVectorTransforms");

    runner.add_test("roundtrip_with_velocity", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        StateVector state_gcrs;
        state_gcrs.position = Vector3d(6778137.0, 0.0, 0.0);
        state_gcrs.velocity = Vector3d(0.0, 7500.0, 0.0);

        StateVector state_itrf = transform.gcrs_to_itrf(state_gcrs, epoch);
        StateVector state_gcrs_back = transform.itrf_to_gcrs(state_itrf, epoch);

        TEST_ASSERT_VEC3_NEAR(state_gcrs_back.position, state_gcrs.position, 1e-5);
        TEST_ASSERT_VEC3_NEAR(state_gcrs_back.velocity, state_gcrs.velocity, 1e-5);
    });

    // The velocity transform must equal the time derivative of the position
    // transform, INCLUDING the drift of the CIP (Qdot, ~3e-12 rad/s). A
    // stationary ITRF point on the polar axis isolates it: Earth's spin adds
    // ~nothing there, so its GCRS velocity (~2e-5 m/s) is the CIP drift. A
    // +-5 min central difference of positions is exact to ~1e-11 m/s here
    // (even with polar motion loaded: the point then circles the CIP at ~13 m).
    // Omitting the term costs ~2e-5 m/s, a +-0.3 m seasonal 1-hour
    // along-track error on Sentinel orbits.

    runner.add_test("velocity_matches_position_derivative_incl_cip_drift", []() {
        IAUTransform transform;
        const Epoch t0 = Epoch::from_utc(2025, 6, 1, 0, 0, 0);
        const Vector3d r_itrf(0.0, 0.0, 6356752.0);
        StateVector s_itrf; s_itrf.position = r_itrf; s_itrf.velocity = Vector3d::Zero();
        const Vector3d v = transform.itrf_to_gcrs(s_itrf, t0).velocity;
        const Epoch tp = t0 + 300.0, tm = t0 - 300.0;
        // (Fresh instances or the same one: the results depend on the epoch only.)
        const Vector3d v_fd = (IAUTransform{}.itrf_to_gcrs(r_itrf, tp) -
                               IAUTransform{}.itrf_to_gcrs(r_itrf, tm)) / (tp - tm);
        TEST_ASSERT(v_fd.norm() > 1e-5);                 // the drift is real
        TEST_ASSERT_VEC3_NEAR(v, v_fd, 1e-9);            // measured 2e-11 m/s
        // and the inverse transform returns zero ITRF velocity
        StateVector s_gcrs; s_gcrs.position = transform.itrf_to_gcrs(r_itrf, t0); s_gcrs.velocity = v;
        TEST_ASSERT_VEC3_NEAR(transform.gcrs_to_itrf(s_gcrs, t0).velocity, Vector3d::Zero(), 1e-9);
    });

    runner.add_test("velocity_includes_rotation", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        // Stationary in GCRS at equator
        StateVector state_gcrs;
        state_gcrs.position = Vector3d(6378137.0, 0.0, 0.0);
        state_gcrs.velocity = Vector3d(0.0, 0.0, 0.0);

        StateVector state_itrf = transform.gcrs_to_itrf(state_gcrs, epoch);

        // In ITRF, should have velocity due to Earth rotation
        // v = omega × r ≈ 7.29e-5 × 6.378e6 ≈ 465 m/s
        double v_itrf = state_itrf.velocity.norm();
        TEST_ASSERT_NEAR(v_itrf, 465.0, 10.0);
    });


    runner.add_test("geostationary_orbit", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        // Geostationary orbit in GCRS has velocity matching Earth rotation
        double r_geo = 42164000.0;  // meters
        double v_geo = earth::ANGULAR_VELOCITY * r_geo;  // ~3075 m/s

        StateVector state_gcrs;
        state_gcrs.position = Vector3d(r_geo, 0.0, 0.0);
        state_gcrs.velocity = Vector3d(0.0, v_geo, 0.0);

        StateVector state_itrf = transform.gcrs_to_itrf(state_gcrs, epoch);

        // In ITRF, geostationary should be nearly stationary
        double v_itrf = state_itrf.velocity.norm();
        TEST_ASSERT(v_itrf < 10.0);  // Nearly zero velocity in ITRF
    });

    runner.add_test("high_inclination_orbit", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        // Polar orbit position
        StateVector state_gcrs;
        state_gcrs.position = Vector3d(0.0, 0.0, 6778137.0);  // Over North Pole
        state_gcrs.velocity = Vector3d(7500.0, 0.0, 0.0);

        StateVector state_itrf = transform.gcrs_to_itrf(state_gcrs, epoch);

        // Velocity magnitude should be similar (Earth rotation minimal at pole)
        TEST_ASSERT_NEAR(state_itrf.velocity.norm(), state_gcrs.velocity.norm(), 10.0);
    });

    // Earth's spin: a point fixed on the equator moves at the rate of the
    // Earth rotation angle, 2*pi*1.00273781191135448/86400 rad/s (no EOP table:
    // LOD 0). A 5-point central difference (h = 60 s, truncation ~6e-9 m/s)
    // resolves the 1.47e-12 rad/s (9.4e-6 m/s here) by which the nominal WGS84
    // rate 7.292115e-5 rad/s falls short.
    runner.add_test("velocity_matches_position_derivative_earth_rotation_rate", []() {
        IAUTransform transform;
        const Epoch t0 = Epoch::from_utc(2025, 6, 1, 3, 0, 0);
        const Vector3d r_itrf(6378137.0, 0.0, 0.0);
        StateVector s_itrf; s_itrf.position = r_itrf; s_itrf.velocity = Vector3d::Zero();
        const Vector3d v = transform.itrf_to_gcrs(s_itrf, t0).velocity;
        const double h = 60.0;
        auto p = [&](double k) { return transform.itrf_to_gcrs(r_itrf, t0 + k * h); };
        const Vector3d v_fd = (p(-2) - 8.0 * p(-1) + 8.0 * p(1) - p(2)) / (12.0 * h);
        TEST_ASSERT_VEC3_NEAR(v, v_fd, 3e-8);
        TEST_ASSERT_NEAR(transform.earth_rotation_rate(t0),
                         2.0 * constants::PI * 1.00273781191135448 / 86400.0, 1e-20);
    });

    runner.run();
}

void test_epoch_only_results() {
    TestRunner runner("EpochOnlyResults");

    // A conversion depends on the epoch alone, not on what the instance
    // converted before: the precession-nutation series sits on a fixed TT grid
    // and everything else is evaluated at the epoch. (A matrix held for up to
    // 10 minutes made the same epoch convert differently by up to ~1 cm at a
    // LEO radius, depending on the call history.)
    runner.add_test("same_epoch_same_matrix_whatever_came_before", []() {
        const Epoch e = Epoch::from_utc(2025, 3, 10, 6, 7, 8.25);
        IAUTransform fresh, used;
        for (double dt : {-3000.0, -590.0, -1.0, 250.0, 599.0}) {
            (void)used.gcrs_to_itrf_matrix(e + dt);
        }
        const Matrix3d a = fresh.gcrs_to_itrf_matrix(e);
        const Matrix3d b = used.gcrs_to_itrf_matrix(e);
        TEST_ASSERT((a - b).cwiseAbs().maxCoeff() == 0.0);
        StateVector s; s.position = Vector3d(6.9e6, 1.0e5, 2.0e5); s.velocity = Vector3d(10.0, 7.0e3, 2.0e3);
        (void)used.gcrs_to_itrf(s, e - 400.0);
        const StateVector f = fresh.gcrs_to_itrf(s, e), u = used.gcrs_to_itrf(s, e);
        TEST_ASSERT((f.position - u.position).norm() == 0.0);
        TEST_ASSERT((f.velocity - u.velocity).norm() == 0.0);
    });

    // The interpolated precession-nutation matches the series evaluated at the
    // epoch itself (Catmull-Rom on a 600 s grid: < 1e-15 rad).
    runner.add_test("interpolated_precession_nutation_matches_the_series", []() {
        IAUTransform transform;
        const Epoch e0 = Epoch::from_utc(2024, 9, 1, 0, 0, 0);
        double worst = 0.0;
        for (double dt = 0.0; dt < 1800.0; dt += 137.0) {
            const Epoch e = e0 + dt;
            // No EOP table (AssumeZero): W and R known, so Q follows from the product
            const Matrix3d Q = gcrs_to_cirs_matrix(e.julian_centuries_tt());
            const Matrix3d full = transform.gcrs_to_itrf_matrix(e);
            const Matrix3d R = cirs_to_tirs_matrix(e.ut1_jd());
            const Matrix3d W = tirs_to_itrf_matrix(0.0, 0.0, e.julian_centuries_tt());
            // R from a one-double UT1 date is only good to ~3e-9 rad: compare Q via
            // the third row, which R (a rotation about Z) does not touch
            const Matrix3d Q_from_full = (W * R).transpose() * full;
            worst = std::max(worst, (Q.row(2) - Q_from_full.row(2)).cwiseAbs().maxCoeff());
        }
        TEST_ASSERT(worst < 1e-14);
    });

    // The Earth rotation angle comes from the exact UT1 day and seconds: a
    // point's Earth-fixed position moves smoothly at the microsecond level (a
    // one-double UT1 Julian date moves it in ~2 cm jumps every ~40 us).
    runner.add_test("earth_rotation_angle_has_no_40_us_steps", []() {
        IAUTransform transform;
        const Epoch e0 = Epoch::from_utc(2025, 1, 20, 18, 30, 0);
        const Vector3d r(7.0e6, 0.0, 0.0);
        std::vector<Vector3d> p;
        for (int k = 0; k < 200; ++k) p.push_back(transform.gcrs_to_itrf(r, e0 + 1e-6 * k));
        double worst = 0.0;   // second differences: ~0 for a smooth path
        for (size_t k = 1; k + 1 < p.size(); ++k) {
            worst = std::max(worst, (p[k + 1] - 2.0 * p[k] + p[k - 1]).norm());
        }
        TEST_ASSERT(worst < 1e-5);   // one 40 us step would be ~2e-2 m
    });

    runner.run();
}

void test_explicit_eop() {
    TestRunner runner("ExplicitEOP");

    // With no table loaded, explicit EOP are used, all of them: UT1-UTC turns
    // the Earth by dut1 * rotation rate, LOD sets the rotation rate. They work
    // under the default EopPolicy::Require (no table lookup happens).
    runner.add_test("explicit_dut1_and_lod_are_used_without_a_table", []() {
        auto& eop_data = ballistic::time::EOPData::instance();
        eop_data.set_policy(ballistic::time::EopPolicy::Require);
        const Epoch e = Epoch::from_utc(2025, 2, 1, 0, 0, 0);
        EarthOrientationParameters base; base.xp = 1e-9;   // explicit, ~zero
        EarthOrientationParameters shifted = base; shifted.dut1 = 0.3; shifted.lod = 0.002;
        bool threw = false;
        Matrix3d m0, m1;
        double rate = 0.0;
        try {
            m0 = IAUTransform(base).gcrs_to_itrf_matrix(e);
            IAUTransform t1(shifted);
            m1 = t1.gcrs_to_itrf_matrix(e);
            rate = t1.earth_rotation_rate(e);
        } catch (const std::exception&) {
            threw = true;
        }
        eop_data.set_policy(ballistic::time::EopPolicy::AssumeZero);
        TEST_ASSERT(!threw);
        // m1 * m0^T = Rz(angle) up to the (tiny) polar motion: the extra turn
        const Matrix3d d = m1 * m0.transpose();
        const double angle = std::atan2(d(0, 1), d(0, 0));
        const double era_rate = 2.0 * constants::PI * 1.00273781191135448 / 86400.0;
        TEST_ASSERT_NEAR(angle, 0.3 * era_rate, 1e-12);
        TEST_ASSERT_NEAR(rate, era_rate * (1.0 - 0.002 / 86400.0), 1e-20);
    });

    runner.run();
}

void test_convenience_functions() {
    TestRunner runner("ConvenienceFunctions");

    runner.add_test("gcrs_to_itrf_full_position", []() {
        Epoch epoch = Epoch::j2000();
        Vector3d pos_gcrs(6778137.0, 1000000.0, 0.0);

        Vector3d pos_itrf = gcrs_to_itrf_full(pos_gcrs, epoch);

        // Magnitude should be preserved
        TEST_ASSERT_NEAR(pos_itrf.norm(), pos_gcrs.norm(), 1.0);
    });

    runner.add_test("itrf_to_gcrs_full_position", []() {
        Epoch epoch = Epoch::j2000();
        Vector3d pos_itrf(6778137.0, 0.0, 0.0);

        Vector3d pos_gcrs = itrf_to_gcrs_full(pos_itrf, epoch);

        // Magnitude should be preserved
        TEST_ASSERT_NEAR(pos_gcrs.norm(), pos_itrf.norm(), 1.0);
    });

    runner.add_test("gcrs_to_itrf_full_state", []() {
        Epoch epoch = Epoch::j2000();
        StateVector state_gcrs;
        state_gcrs.position = Vector3d(6778137.0, 0.0, 0.0);
        state_gcrs.velocity = Vector3d(0.0, 7500.0, 0.0);

        StateVector state_itrf = gcrs_to_itrf_full(state_gcrs, epoch);

        // Position magnitude preserved
        TEST_ASSERT_NEAR(state_itrf.position.norm(), state_gcrs.position.norm(), 1.0);
    });

    runner.add_test("itrf_to_gcrs_full_state", []() {
        Epoch epoch = Epoch::j2000();
        StateVector state_itrf;
        state_itrf.position = Vector3d(6778137.0, 0.0, 0.0);
        state_itrf.velocity = Vector3d(0.0, 500.0, 0.0);

        StateVector state_gcrs = itrf_to_gcrs_full(state_itrf, epoch);

        // Position magnitude preserved
        TEST_ASSERT_NEAR(state_gcrs.position.norm(), state_itrf.position.norm(), 1.0);
    });

    runner.add_test("roundtrip_full", []() {
        Epoch epoch = Epoch::from_utc(2024, 7, 15, 6, 30, 0);
        StateVector original;
        original.position = Vector3d(5000000.0, 4000000.0, 3000000.0);
        original.velocity = Vector3d(-1000.0, 7000.0, 500.0);

        StateVector itrf = gcrs_to_itrf_full(original, epoch);
        StateVector back = itrf_to_gcrs_full(itrf, epoch);

        TEST_ASSERT_VEC3_NEAR(back.position, original.position, 1e-5);
        TEST_ASSERT_VEC3_NEAR(back.velocity, original.velocity, 1e-5);
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("very_distant_past", []() {
        // Year 1900
        double t_tt = -1.0;  // 100 years before J2000

        auto args = compute_fundamental_args(t_tt);
        auto prec = compute_precession_angles(t_tt);
        auto nut = compute_nutation(t_tt, args);
        auto cip = compute_cip_position(t_tt);

        // All should be finite
        TEST_ASSERT(std::isfinite(prec.epsa));
        TEST_ASSERT(std::isfinite(nut.dpsi));
        TEST_ASSERT(std::isfinite(cip.X));
    });

    runner.add_test("very_distant_future", []() {
        // Year 2100
        double t_tt = 1.0;  // 100 years after J2000

        auto args = compute_fundamental_args(t_tt);
        auto prec = compute_precession_angles(t_tt);
        auto nut = compute_nutation(t_tt, args);
        auto cip = compute_cip_position(t_tt);

        // All should be finite
        TEST_ASSERT(std::isfinite(prec.epsa));
        TEST_ASSERT(std::isfinite(nut.dpsi));
        TEST_ASSERT(std::isfinite(cip.X));
    });

    runner.add_test("large_eop_values", []() {
        // Extreme but plausible EOP values
        EarthOrientationParameters eop;
        eop.xp = 1.0;    // 1 arcsec
        eop.yp = 1.0;
        eop.dut1 = 0.9;  // Near leap second threshold

        Matrix3d W = tirs_to_itrf_matrix(eop);

        // Should still be orthogonal
        Matrix3d WWt = W * W.transpose();
        double diff = (WWt - Matrix3d::Identity()).norm();
        TEST_ASSERT(diff < 1e-8);
    });

    runner.add_test("zero_position", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        Vector3d pos_zero(0.0, 0.0, 0.0);
        Vector3d result = transform.gcrs_to_itrf(pos_zero, epoch);

        // Zero in = zero out
        TEST_ASSERT_VEC3_NEAR(result, pos_zero, 1e-15);
    });

    runner.add_test("very_large_position", []() {
        IAUTransform transform;
        Epoch epoch = Epoch::j2000();

        // Moon distance
        Vector3d pos_moon(384400000.0, 0.0, 0.0);
        Vector3d result = transform.gcrs_to_itrf(pos_moon, epoch);

        // Magnitude should be preserved
        TEST_ASSERT_NEAR(result.norm(), pos_moon.norm(), 1e6);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    // Runs without an Earth orientation table on purpose (synthetic or
    // self-consistent checks; pinned values assume zero UT1-UTC/polar motion).
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    std::cout << "======================================" << std::endl;
    std::cout << "IAU 2006/2000A Transform Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_earth_orientation_parameters();
    test_fundamental_arguments();
    test_precession_angles();
    test_nutation();
    test_cip_position();
    test_earth_rotation_angle();
    test_gcrs_to_cirs_matrix();
    test_cirs_to_tirs_matrix();
    test_tirs_to_itrf_matrix();
    test_iau_transform_class();
    test_position_transforms();
    test_state_vector_transforms();
    test_explicit_eop();
    test_epoch_only_results();
    test_convenience_functions();
    test_edge_cases();

    return print_final_summary();
}
