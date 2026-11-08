// Tests for two util/ helpers:
//   util/time_utils.hpp      epoch_to_decimal_year (e.g. 2010-01-01 -> 2010.0),
//                            used for ITRF (International Terrestrial Reference
//                            Frame) epoch-dependent transformations;
//   util/frame_transform.hpp itrf_to_gcrs / gcrs_to_itrf (Earth-fixed ITRF,
//                            also called ECEF, <-> inertial GCRS) and the
//                            acceleration transform acceleration_to_itrf.
// Decimal-year tolerances reflect that the function is an approximation
// (months as twelfths of a year); frame round trips must close to < 1 mm.

#include "../test_utils.hpp"
#include <ballistic/util/time_utils.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <ballistic/core/constants.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::time;

// =============================================================================
// Suite 1: epoch_to_decimal_year reference values
// =============================================================================

void test_epoch_to_decimal_year() {
    TestRunner runner("EpochToDecimalYear");

    // --- Reference epoch anchors (exact integer years) ---

    runner.add_test("jan1_midnight_gives_exact_year_2020", []() {
        Epoch e = Epoch::from_utc(2020, 1, 1, 0, 0, 0.0);
        double dy = util::epoch_to_decimal_year(e);
        TEST_ASSERT_NEAR(dy, 2020.0, 1e-10);
    });

    runner.add_test("jan1_midnight_gives_exact_year_2000", []() {
        Epoch e = Epoch::from_utc(2000, 1, 1, 0, 0, 0.0);
        double dy = util::epoch_to_decimal_year(e);
        TEST_ASSERT_NEAR(dy, 2000.0, 1e-10);
    });

    // --- ITRF reference epochs (critical for Helmert transforms) ---

    runner.add_test("itrf_reference_epoch_2010", []() {
        // ITRF2008 reference epoch is 2010.0 = Jan 1 2010 00:00 UTC
        Epoch e = Epoch::from_utc(2010, 1, 1, 0, 0, 0.0);
        double dy = util::epoch_to_decimal_year(e);
        TEST_ASSERT_NEAR(dy, 2010.0, 1e-10);
    });

    runner.add_test("itrf_reference_epoch_2015", []() {
        // ITRF2014 reference epoch is 2015.0 = Jan 1 2015 00:00 UTC
        Epoch e = Epoch::from_utc(2015, 1, 1, 0, 0, 0.0);
        double dy = util::epoch_to_decimal_year(e);
        TEST_ASSERT_NEAR(dy, 2015.0, 1e-10);
    });

    // --- Mid-year value ---

    runner.add_test("mid_year_approximately_half", []() {
        // July 2 is day 183/184 of the year, approximately 0.5
        Epoch e = Epoch::from_utc(2020, 7, 2, 0, 0, 0.0);
        double dy = util::epoch_to_decimal_year(e);
        // Approximate formula: (7-1)/12 + (2-1)/365.25 = 0.5 + 0.00274 = 0.5027
        // True fractional year: 183/366 = 0.5 (2020 is leap year)
        // Allow 0.01 tolerance for approximate formula
        TEST_ASSERT_NEAR(dy, 2020.5, 0.01);
    });

    // --- Intraday component ---

    runner.add_test("noon_vs_midnight_differ_by_half_day", []() {
        Epoch midnight = Epoch::from_utc(2024, 6, 15, 0, 0, 0.0);
        Epoch noon = Epoch::from_utc(2024, 6, 15, 12, 0, 0.0);
        double dy_midnight = util::epoch_to_decimal_year(midnight);
        double dy_noon = util::epoch_to_decimal_year(noon);
        // 12 hours / (24 * 365.25 hours) = 0.001369
        double expected_diff = 12.0 / (24.0 * 365.25);
        TEST_ASSERT_NEAR(dy_noon - dy_midnight, expected_diff, 1e-10);
    });

    // --- Monotonicity across months ---

    runner.add_test("12_monthly_samples_strictly_increasing", []() {
        double prev = 0.0;
        for (int m = 1; m <= 12; ++m) {
            Epoch e = Epoch::from_utc(2023, m, 1, 0, 0, 0.0);
            double dy = util::epoch_to_decimal_year(e);
            TEST_ASSERT_TRUE(dy > prev);
            prev = dy;
        }
    });

    // --- Range: result in [year, year+1) ---

    runner.add_test("result_within_year_range", []() {
        // Test several dates in 2024
        for (int m = 1; m <= 12; ++m) {
            Epoch e = Epoch::from_utc(2024, m, 15, 12, 30, 45.0);
            double dy = util::epoch_to_decimal_year(e);
            TEST_ASSERT_TRUE(dy >= 2024.0);
            TEST_ASSERT_TRUE(dy < 2025.0);
        }
    });

    // --- J2000 epoch ---

    runner.add_test("j2000_epoch_near_2000_point_0", []() {
        // J2000.0 = 2000-01-01 12:00:00 TT ≈ 2000-01-01 11:58:55.816 UTC
        // Using Jan 1 2000 12:00 UTC (close enough for approximate formula)
        Epoch e = Epoch::from_utc(2000, 1, 1, 12, 0, 0.0);
        double dy = util::epoch_to_decimal_year(e);
        // Expected: 2000 + 0 + 0 + 12/(24*365.25) = 2000.001369
        TEST_ASSERT_NEAR(dy, 2000.0 + 12.0 / (24.0 * 365.25), 1e-10);
    });

    runner.run();
}

// =============================================================================
// Suite 2: frame_transform routing tests
// =============================================================================

void test_frame_transform() {
    TestRunner runner("FrameTransform");

    // LEO state for testing: ~400 km altitude circular orbit
    const double r = 6778137.0; // meters
    const double v_eci = std::sqrt(constants::earth::GM / r);
    const double omega_r = constants::earth::ANGULAR_VELOCITY * r;
    const Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0.0);

    // --- GCRS roundtrip ---

    runner.add_test("gcrs_roundtrip_position_submm", [&]() {
        StateVector ecef;
        ecef.position = Eigen::Vector3d(r, 0.0, 0.0);
        ecef.velocity = Eigen::Vector3d(0.0, v_eci - omega_r, 0.0);

        auto eci = util::itrf_to_gcrs(ecef, epoch);
        auto back = util::gcrs_to_itrf(eci, epoch);

        double pos_err = (back.position - ecef.position).norm();
        double vel_err = (back.velocity - ecef.velocity).norm();
        TEST_ASSERT_TRUE(pos_err < 1e-3);  // < 1 mm
        TEST_ASSERT_TRUE(vel_err < 1e-6);  // < 0.001 mm/s
    });

    // --- GCRS acceleration includes Coriolis/centrifugal (magnitude NOT preserved) ---

    runner.add_test("gcrs_acceleration_includes_coriolis", [&]() {
        Eigen::Vector3d acc_eci(0.001, -0.002, 0.0005);
        StateVector state_ecef;
        state_ecef.position = Eigen::Vector3d(r, 0.0, 0.0);
        state_ecef.velocity = Eigen::Vector3d(0.0, v_eci - omega_r, 0.0);

        auto acc_ecef = util::acceleration_to_itrf(
            acc_eci, state_ecef, epoch);

        // acceleration_to_itrf converts force accel to coordinate accel
        // (d²r_ecef/dt²) by subtracting Coriolis (-2ω×v) and centrifugal
        // (-ω×(ω×r)) — magnitude should differ from input
        double mag_diff = std::abs(acc_ecef.norm() - acc_eci.norm());
        TEST_ASSERT_TRUE(mag_diff > 1e-10);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    ballistic::initialize({});

    test_epoch_to_decimal_year();
    test_frame_transform();

    return print_final_summary();
}
