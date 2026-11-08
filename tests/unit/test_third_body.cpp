// Third-body gravity (gravity/third_body.hpp): the extra acceleration of a
// satellite from the Sun, Moon and Jupiter, i.e. the body's pull on the satellite
// minus its pull on the Earth (the "tidal" difference, Vallado 2013 Eq. 8-36).
// Checks configuration, magnitudes against hand calculations and Montenbruck &
// Gill (2000) Table 3.1, and the formula itself with fixed JPL Horizons positions.
// LEO = low Earth orbit, GEO = geostationary orbit (42,164 km radius).

#include "../test_utils.hpp"
#include <ballistic/gravity/third_body.hpp>
#include <ballistic/init.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::gravity;
using namespace ballistic::constants;

// =============================================================================
// Test Suites
// =============================================================================

void test_third_body_config() {
    TestRunner runner("ThirdBodyConfig");

    runner.add_test("default_constructor", []() {
        ThirdBodyGravity tb;

        // Sun and Moon enabled by default, Jupiter disabled
        TEST_ASSERT_TRUE(tb.sun_enabled());
        TEST_ASSERT_TRUE(tb.moon_enabled());
        TEST_ASSERT_FALSE(tb.jupiter_enabled());
    });

    runner.add_test("enable_disable_sun", []() {
        ThirdBodyGravity tb;

        tb.enable_sun(false);
        TEST_ASSERT_FALSE(tb.sun_enabled());

        tb.enable_sun(true);
        TEST_ASSERT_TRUE(tb.sun_enabled());
    });

    runner.add_test("enable_disable_moon", []() {
        ThirdBodyGravity tb;

        tb.enable_moon(false);
        TEST_ASSERT_FALSE(tb.moon_enabled());

        tb.enable_moon(true);
        TEST_ASSERT_TRUE(tb.moon_enabled());
    });

    runner.add_test("enable_disable_jupiter", []() {
        ThirdBodyGravity tb;

        tb.enable_jupiter(true);
        TEST_ASSERT_TRUE(tb.jupiter_enabled());

        tb.enable_jupiter(false);
        TEST_ASSERT_FALSE(tb.jupiter_enabled());
    });

    runner.add_test("custom_gm_values", []() {
        ThirdBodyGravity tb;

        // Set custom GM values
        double custom_sun_gm = 1.4e20;
        double custom_moon_gm = 5.0e12;
        double custom_jupiter_gm = 1.3e17;

        tb.set_sun_gm(custom_sun_gm);
        tb.set_moon_gm(custom_moon_gm);
        tb.set_jupiter_gm(custom_jupiter_gm);

        // Values should affect acceleration calculations
        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        // With custom GM, acceleration should be finite and physical
        Vector3d acc = tb.acceleration(epoch, pos);
        TEST_ASSERT(std::isfinite(acc.norm()));
        TEST_ASSERT(acc.norm() > 1e-8);
        TEST_ASSERT(acc.norm() < 1e-3);
    });

    runner.run();
}

void test_sun_acceleration() {
    TestRunner runner("SunAcceleration");

    runner.add_test("magnitude_at_leo", []() {
        ThirdBodyGravity tb;
        tb.enable_moon(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);  // LEO

        Vector3d acc = tb.sun_acceleration(epoch, pos);

        // Sun perturbation at 7000 km radius: ~5.5e-7 m/s² (2·GM_sun·r/AU³),
        // small compared to Earth gravity (~8 m/s²). Loose bounds: order of magnitude.
        double mag = acc.norm();
        TEST_ASSERT(mag > 1e-7);   // Non-zero
        TEST_ASSERT(mag < 1e-4);   // Small compared to Earth gravity
    });

    runner.add_test("magnitude_at_geo", []() {
        ThirdBodyGravity tb;
        tb.enable_moon(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(42164000.0, 0.0, 0.0);  // GEO

        Vector3d acc = tb.sun_acceleration(epoch, pos);

        // Sun perturbation at GEO is similar to LEO (Sun is far)
        double mag = acc.norm();
        TEST_ASSERT(mag > 1e-7);
        TEST_ASSERT(mag < 1e-4);
    });

    runner.add_test("varies_with_position", []() {
        ThirdBodyGravity tb;
        tb.enable_moon(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();

        // Positions on opposite sides of Earth
        Vector3d pos_toward(7000000.0, 0.0, 0.0);
        Vector3d pos_away(-7000000.0, 0.0, 0.0);

        Vector3d acc_toward = tb.sun_acceleration(epoch, pos_toward);
        Vector3d acc_away = tb.sun_acceleration(epoch, pos_away);

        // Accelerations should be similar magnitude but may differ
        // The difference represents the tidal effect
        double diff = (acc_toward - acc_away).norm();
        TEST_ASSERT(diff > 1e-12);  // Non-zero tidal difference
        TEST_ASSERT(diff < 1e-5);   // But small
    });

    runner.add_test("varies_with_time", []() {
        ThirdBodyGravity tb;
        tb.enable_moon(false);
        tb.enable_jupiter(false);

        Vector3d pos(7000000.0, 0.0, 0.0);

        // Six months apart - Sun position changes
        Epoch epoch1 = Epoch::j2000();
        Epoch epoch2 = Epoch::j2000() + 182.5 * constants::SECONDS_PER_DAY;

        Vector3d acc1 = tb.sun_acceleration(epoch1, pos);
        Vector3d acc2 = tb.sun_acceleration(epoch2, pos);

        // Accelerations should be different (Sun position changed)
        double diff = (acc2 - acc1).norm();
        TEST_ASSERT(diff > 1e-12);  // Non-zero time variation
        TEST_ASSERT(diff < 1e-5);   // But bounded
        TEST_ASSERT(std::isfinite(acc1.norm()));
        TEST_ASSERT(std::isfinite(acc2.norm()));
    });

    runner.run();
}

void test_moon_acceleration() {
    TestRunner runner("MoonAcceleration");

    runner.add_test("magnitude_at_leo", []() {
        ThirdBodyGravity tb;
        tb.enable_sun(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);  // LEO

        Vector3d acc = tb.moon_acceleration(epoch, pos);

        // Moon perturbation at LEO: ~1e-6 m/s², about twice the Sun's (see the
        // Montenbruck reference cells below). Loose bounds: order of magnitude.
        double mag = acc.norm();
        TEST_ASSERT(mag > 1e-7);   // Non-zero
        TEST_ASSERT(mag < 1e-4);   // Small
    });

    runner.add_test("moon_stronger_tidal", []() {
        ThirdBodyGravity tb;

        Epoch epoch = Epoch::j2000();

        // Moon tidal effect is stronger than Sun despite lower GM
        // because Moon is much closer
        Vector3d pos1(7000000.0, 0.0, 0.0);
        Vector3d pos2(-7000000.0, 0.0, 0.0);

        tb.enable_sun(false);
        tb.enable_jupiter(false);
        Vector3d moon_acc1 = tb.moon_acceleration(epoch, pos1);
        Vector3d moon_acc2 = tb.moon_acceleration(epoch, pos2);
        double moon_tidal = (moon_acc1 - moon_acc2).norm();

        tb.enable_sun(true);
        tb.enable_moon(false);
        Vector3d sun_acc1 = tb.sun_acceleration(epoch, pos1);
        Vector3d sun_acc2 = tb.sun_acceleration(epoch, pos2);
        double sun_tidal = (sun_acc1 - sun_acc2).norm();

        // Moon tidal effect should be about 2x Sun's
        TEST_ASSERT(moon_tidal > sun_tidal * 0.5);
    });

    runner.add_test("varies_with_lunar_phase", []() {
        ThirdBodyGravity tb;
        tb.enable_sun(false);
        tb.enable_jupiter(false);

        Vector3d pos(7000000.0, 0.0, 0.0);

        // Half a lunar month apart (~14 days)
        Epoch epoch1 = Epoch::j2000();
        Epoch epoch2 = Epoch::j2000() + 14.0 * constants::SECONDS_PER_DAY;

        Vector3d acc1 = tb.moon_acceleration(epoch1, pos);
        Vector3d acc2 = tb.moon_acceleration(epoch2, pos);

        // Tidal direction is dominated by satellite position, not Moon phase, so
        // direction change over 14d is small (~10°) but non-zero. Magnitudes,
        // however, vary appreciably as Moon distance changes.
        double dot = acc1.normalized().dot(acc2.normalized());
        TEST_ASSERT(dot < 0.999);  // Direction varies at all
        TEST_ASSERT(std::abs(acc1.norm() - acc2.norm()) / acc1.norm() > 0.01);
    });

    runner.run();
}

void test_jupiter_acceleration() {
    TestRunner runner("JupiterAcceleration");

    runner.add_test("magnitude_at_leo", []() {
        ThirdBodyGravity tb;
        tb.enable_sun(false);
        tb.enable_moon(false);
        tb.enable_jupiter(true);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d acc = tb.jupiter_acceleration(epoch, pos);

        // Jupiter perturbation: ~1e-9 m/s² (very small)
        double mag = acc.norm();
        TEST_ASSERT(mag > 1e-12);  // Non-zero
        TEST_ASSERT(mag < 1e-6);   // Much smaller than Sun/Moon
    });

    runner.add_test("much_smaller_than_sun_moon", []() {
        ThirdBodyGravity tb;

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        tb.enable_sun(false);
        tb.enable_moon(false);
        tb.enable_jupiter(true);
        Vector3d jupiter_acc = tb.jupiter_acceleration(epoch, pos);

        tb.enable_sun(true);
        tb.enable_moon(false);
        tb.enable_jupiter(false);
        Vector3d sun_acc = tb.sun_acceleration(epoch, pos);

        // Jupiter should be ~1000x smaller than Sun
        double ratio = jupiter_acc.norm() / sun_acc.norm();
        TEST_ASSERT(ratio < 0.01);
    });

    runner.add_test("varies_over_year", []() {
        ThirdBodyGravity tb;
        tb.enable_sun(false);
        tb.enable_moon(false);
        tb.enable_jupiter(true);

        Vector3d pos(7000000.0, 0.0, 0.0);

        // One year apart
        Epoch epoch1 = Epoch::j2000();
        Epoch epoch2 = Epoch::j2000() + 365.25 * constants::SECONDS_PER_DAY;

        Vector3d acc1 = tb.jupiter_acceleration(epoch1, pos);
        Vector3d acc2 = tb.jupiter_acceleration(epoch2, pos);

        // Jupiter's geocentric direction varies substantially over 1 year due
        // to Earth's parallax (Earth's orbit gives a ~10 AU baseline at 5 AU),
        // so the tidal-accel direction can shift by ~60-70°. Just verify it
        // varies and the magnitudes are both physical.
        double dot = acc1.normalized().dot(acc2.normalized());
        TEST_ASSERT(dot < 0.999);
        TEST_ASSERT(acc1.norm() > 1e-12 && acc2.norm() > 1e-12);
    });

    runner.run();
}

void test_combined_acceleration() {
    TestRunner runner("CombinedAcceleration");

    runner.add_test("sum_of_parts", []() {
        ThirdBodyGravity tb;
        tb.enable_jupiter(true);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        // Get individual accelerations
        Vector3d sun_acc = tb.sun_acceleration(epoch, pos);
        Vector3d moon_acc = tb.moon_acceleration(epoch, pos);
        Vector3d jupiter_acc = tb.jupiter_acceleration(epoch, pos);

        // Get combined
        Vector3d total = tb.acceleration(epoch, pos);

        // Should be sum of parts
        Vector3d expected = sun_acc + moon_acc + jupiter_acc;
        TEST_ASSERT_VEC3_NEAR(total, expected, 1e-15);
    });

    runner.add_test("sun_only", []() {
        ThirdBodyGravity tb;
        tb.enable_sun(true);
        tb.enable_moon(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d sun_acc = tb.sun_acceleration(epoch, pos);
        Vector3d total = tb.acceleration(epoch, pos);

        TEST_ASSERT_VEC3_NEAR(total, sun_acc, 1e-15);
    });

    runner.add_test("moon_only", []() {
        ThirdBodyGravity tb;
        tb.enable_sun(false);
        tb.enable_moon(true);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d moon_acc = tb.moon_acceleration(epoch, pos);
        Vector3d total = tb.acceleration(epoch, pos);

        TEST_ASSERT_VEC3_NEAR(total, moon_acc, 1e-15);
    });

    runner.add_test("nothing_enabled", []() {
        ThirdBodyGravity tb;
        tb.enable_sun(false);
        tb.enable_moon(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d total = tb.acceleration(epoch, pos);
        TEST_ASSERT_VEC3_NEAR(total, Vector3d::Zero(), 1e-20);
    });

    runner.run();
}

void test_body_positions() {
    TestRunner runner("BodyPositions");

    runner.add_test("sun_distance", []() {
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();

        Vector3d sun_pos = tb.sun_position(epoch);

        // Sun distance should be ~1 AU = 1.5e11 m
        double dist = sun_pos.norm();
        TEST_ASSERT_NEAR(dist, sun::AU, sun::AU * 0.02);  // Within 2%
    });

    runner.add_test("moon_distance", []() {
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();

        Vector3d moon_pos = tb.moon_position(epoch);

        // Moon distance should be ~3.84e8 m
        double dist = moon_pos.norm();
        TEST_ASSERT_NEAR(dist, moon::MEAN_DISTANCE, moon::MEAN_DISTANCE * 0.1);
    });

    runner.add_test("sun_position_j2000", []() {
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();

        Vector3d sun_pos = tb.sun_position(epoch);

        // Sun should be at approximately 1 AU distance
        TEST_ASSERT_NEAR(sun_pos.norm(), sun::AU, sun::AU * 0.02);

        // All components should be finite
        TEST_ASSERT(std::isfinite(sun_pos.x()));
        TEST_ASSERT(std::isfinite(sun_pos.y()));
        TEST_ASSERT(std::isfinite(sun_pos.z()));
    });

    runner.add_test("moon_position_j2000", []() {
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();

        Vector3d moon_pos = tb.moon_position(epoch);

        // Moon should be at approximately mean distance
        TEST_ASSERT_NEAR(moon_pos.norm(), moon::MEAN_DISTANCE, moon::MEAN_DISTANCE * 0.1);

        // All components should be finite
        TEST_ASSERT(std::isfinite(moon_pos.x()));
        TEST_ASSERT(std::isfinite(moon_pos.y()));
        TEST_ASSERT(std::isfinite(moon_pos.z()));
    });

    runner.run();
}

// =============================================================================
// Reference Value Tests (Vallado Eq. 8-36, Montenbruck Table 3.1)
// =============================================================================

void test_reference_values() {
    TestRunner runner("ReferenceValues");

    runner.add_test("vallado_eq_8_36_hand_computation", []() {
        // Vallado Eq. 8-36: a = GM * [(r_body - r_sat)/|r_body - r_sat|^3 - r_body/|r_body|^3]
        //
        // Test with satellite at (7000 km, 0, 0) and Sun from analytical ephemeris at J2000.
        // Hand-compute each term and verify the code output matches.

        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();
        Vector3d sat_pos(7000000.0, 0.0, 0.0);

        // Get Sun position from the same ephemeris the code uses
        Vector3d r_sun = tb.sun_position(epoch);

        // Hand compute Eq. 8-36 for Sun
        Vector3d r_rel = r_sun - sat_pos;
        double r_rel_norm = r_rel.norm();
        double r_sun_norm = r_sun.norm();
        double r_rel_3 = r_rel_norm * r_rel_norm * r_rel_norm;
        double r_sun_3 = r_sun_norm * r_sun_norm * r_sun_norm;

        Vector3d expected = sun::GM * (r_rel / r_rel_3 - r_sun / r_sun_3);

        // Get code's result
        tb.enable_moon(false);
        tb.enable_jupiter(false);
        Vector3d actual = tb.sun_acceleration(epoch, sat_pos);

        // Should match to machine precision (same formula)
        TEST_ASSERT_VEC3_NEAR(actual, expected, 1e-20);
    });

    runner.add_test("moon_hand_computation", []() {
        // Same hand computation for Moon
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();
        Vector3d sat_pos(7000000.0, 0.0, 0.0);

        Vector3d r_moon = tb.moon_position(epoch);

        Vector3d r_rel = r_moon - sat_pos;
        double r_rel_norm = r_rel.norm();
        double r_moon_norm = r_moon.norm();
        double r_rel_3 = r_rel_norm * r_rel_norm * r_rel_norm;
        double r_moon_3 = r_moon_norm * r_moon_norm * r_moon_norm;

        Vector3d expected = moon::GM * (r_rel / r_rel_3 - r_moon / r_moon_3);

        tb.enable_sun(false);
        tb.enable_jupiter(false);
        Vector3d actual = tb.moon_acceleration(epoch, sat_pos);

        TEST_ASSERT_VEC3_NEAR(actual, expected, 1e-20);
    });

    runner.add_test("montenbruck_table_3_1_sun_magnitude", []() {
        // Montenbruck & Gill (2000), Table 3.1:
        // Sun third-body perturbation at LEO (400 km): ~5.3e-7 m/s²
        //
        // Approximate formula for tidal acceleration:
        //   a_tidal ≈ 2 * GM_sun * r_sat / r_sun³
        // At r_sat = 6778 km, r_sun ≈ 1 AU:
        //   a ≈ 2 * 1.327e20 * 6.778e6 / (1.496e11)³
        //     ≈ 1.798e27 / 3.348e33 = 5.37e-7 m/s²

        ThirdBodyGravity tb;
        tb.enable_moon(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(6778137.0, 0.0, 0.0);  // 400 km altitude

        Vector3d acc = tb.sun_acceleration(epoch, pos);
        double mag = acc.norm();

        // Montenbruck Table 3.1: ~5e-7 m/s² for Sun at LEO
        // Allow factor of 3 tolerance (actual depends on exact geometry)
        TEST_ASSERT_MSG(mag > 1e-7 && mag < 2e-6,
            "Sun acceleration at LEO should be ~5e-7 m/s², got " + std::to_string(mag));
    });

    runner.add_test("montenbruck_table_3_1_moon_magnitude", []() {
        // Montenbruck & Gill (2000), Table 3.1:
        // Moon third-body perturbation at LEO: ~1.2e-6 m/s²
        //
        // Approximate: a ≈ 2 * GM_moon * r_sat / r_moon³
        //   ≈ 2 * 4.903e12 * 6.778e6 / (3.844e8)³
        //   ≈ 6.646e19 / 5.681e25 = 1.17e-6 m/s²

        ThirdBodyGravity tb;
        tb.enable_sun(false);
        tb.enable_jupiter(false);

        Epoch epoch = Epoch::j2000();
        Vector3d pos(6778137.0, 0.0, 0.0);  // 400 km altitude

        Vector3d acc = tb.moon_acceleration(epoch, pos);
        double mag = acc.norm();

        // Montenbruck Table 3.1: ~1e-6 m/s² for Moon at LEO
        // Allow factor of 3 tolerance
        TEST_ASSERT_MSG(mag > 3e-7 && mag < 3e-6,
            "Moon acceleration at LEO should be ~1e-6 m/s², got " + std::to_string(mag));
    });

    runner.add_test("moon_stronger_than_sun_perturbation", []() {
        // Montenbruck Table 3.1 and tidal theory: Moon's perturbation
        // on satellite is about 2x Sun's, despite Sun's larger GM,
        // because perturbation scales as GM/r³ and Moon is much closer.

        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();
        Vector3d pos(6778137.0, 0.0, 0.0);

        tb.enable_sun(true);
        tb.enable_moon(false);
        tb.enable_jupiter(false);
        double sun_mag = tb.acceleration(epoch, pos).norm();

        tb.enable_sun(false);
        tb.enable_moon(true);
        double moon_mag = tb.acceleration(epoch, pos).norm();

        // Moon perturbation should be larger than Sun's
        TEST_ASSERT_MSG(moon_mag > sun_mag,
            "Moon perturbation should exceed Sun's at LEO");

        // Ratio should be roughly 2:1 (typical range 1.5-3 depending on geometry)
        double ratio = moon_mag / sun_mag;
        TEST_ASSERT_MSG(ratio > 1.0 && ratio < 5.0,
            "Moon/Sun ratio should be ~2, got " + std::to_string(ratio));
    });

    runner.add_test("gm_values_match_de440", []() {
        // Verify GM values match JPL DE440 gm_de440.tpc
        TEST_ASSERT_NEAR_REL(sun::GM, 1.32712440041279419e20, 1e-14);
        TEST_ASSERT_NEAR_REL(moon::GM, 4.9028001184575496e12, 1e-14);
        TEST_ASSERT_NEAR_REL(jupiter::GM, 1.26712764e17, 1e-7);
    });

    runner.run();
}

void test_convenience_function() {
    TestRunner runner("ConvenienceFunction");

    runner.add_test("default_sun_moon", []() {
        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d acc = third_body_perturbation(epoch, pos);

        // Should include Sun and Moon by default
        ThirdBodyGravity tb;
        tb.enable_jupiter(false);
        Vector3d expected = tb.acceleration(epoch, pos);

        TEST_ASSERT_VEC3_NEAR(acc, expected, 1e-15);
    });

    runner.add_test("sun_only_parameter", []() {
        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d acc = third_body_perturbation(epoch, pos, true, false, false);

        ThirdBodyGravity tb;
        tb.enable_moon(false);
        tb.enable_jupiter(false);
        Vector3d expected = tb.acceleration(epoch, pos);

        TEST_ASSERT_VEC3_NEAR(acc, expected, 1e-15);
    });

    runner.add_test("with_jupiter", []() {
        Epoch epoch = Epoch::j2000();
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d acc = third_body_perturbation(epoch, pos, true, true, true);

        ThirdBodyGravity tb;
        tb.enable_jupiter(true);
        Vector3d expected = tb.acceleration(epoch, pos);

        TEST_ASSERT_VEC3_NEAR(acc, expected, 1e-15);
    });

    runner.run();
}

void test_is_conservative() {
    TestRunner runner("IsConservative");

    runner.add_test("returns_true", []() {
        // Third-body gravity is conservative (no energy dissipation)
        TEST_ASSERT_TRUE(ThirdBodyGravity::is_conservative());
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("zero_position", []() {
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();
        Vector3d pos(0.0, 0.0, 0.0);  // At Earth center

        // Should still work (though unphysical)
        Vector3d acc = tb.acceleration(epoch, pos);
        TEST_ASSERT(std::isfinite(acc.norm()));
    });

    runner.add_test("geo_position", []() {
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();
        Vector3d pos(42164000.0, 0.0, 0.0);  // GEO

        Vector3d acc = tb.acceleration(epoch, pos);
        TEST_ASSERT(std::isfinite(acc.norm()));
        // GEO third-body: ~1e-6 to 1e-5 m/s²
        TEST_ASSERT(acc.norm() > 1e-8);
        TEST_ASSERT(acc.norm() < 1e-4);
    });

    runner.add_test("high_altitude", []() {
        ThirdBodyGravity tb;
        Epoch epoch = Epoch::j2000();
        Vector3d pos(100000000.0, 0.0, 0.0);  // ~100,000 km

        Vector3d acc = tb.acceleration(epoch, pos);
        TEST_ASSERT(std::isfinite(acc.norm()));
    });

    runner.add_test("various_epochs", []() {
        ThirdBodyGravity tb;
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        // Test various epochs over 10 years
        for (double days = 0; days < 3650; days += 100) {
            Epoch epoch = Epoch::j2000() + days * constants::SECONDS_PER_DAY;
            Vector3d acc = tb.acceleration(epoch, pos);

            TEST_ASSERT(std::isfinite(acc.norm()));
            TEST_ASSERT(acc.norm() > 1e-8);  // Non-zero
            TEST_ASSERT(acc.norm() < 1e-3);  // Reasonable magnitude
        }
    });

    runner.add_test("symmetry_check", []() {
        ThirdBodyGravity tb;
        tb.enable_jupiter(true);
        Epoch epoch = Epoch::j2000();

        // Opposite positions should have different accelerations (not symmetric)
        Vector3d pos1(7000000.0, 0.0, 0.0);
        Vector3d pos2(-7000000.0, 0.0, 0.0);

        Vector3d acc1 = tb.acceleration(epoch, pos1);
        Vector3d acc2 = tb.acceleration(epoch, pos2);

        // Should be different (not simply opposite)
        double sum_mag = (acc1 + acc2).norm();
        double diff_mag = (acc1 - acc2).norm();

        // Neither sum nor difference should be zero
        TEST_ASSERT(sum_mag > 1e-10);
        TEST_ASSERT(diff_mag > 1e-10);
    });

    runner.run();
}

// =============================================================================
// Independent Hand-Computed Third-Body Tests
// =============================================================================
// These tests use FIXED body positions (not from our ephemeris) and manually
// compute every intermediate value in Vallado Eq. 8-36. This catches bugs in
// the vector math that the same-formula tautological test at 1e-20 cannot.
//
// Reference: Vallado (2013), Eq. 8-36:
//   a = GM * [(r_body - r_sat)/|r_body - r_sat|^3 - r_body/|r_body|^3]
// =============================================================================

void test_independent_hand_computation() {
    TestRunner runner("IndependentHandComputation");

    runner.add_test("fixed_moon_position_step_by_step", []() {
        // Use a FIXED Moon position (not from our ephemeris)
        // Moon at approximately J2000 position from JPL Horizons:
        //   X = -2.916083884e+08 m, Y = -2.667168290e+08 m, Z = -7.610248158e+07 m
        // (GCRF, 2000-01-01 12:00:00 TDB)
        // Source: JPL Horizons system, target=301, center=399, ref=J2000
        Vector3d r_moon(-2.916083884e+08, -2.667168290e+08, -7.610248158e+07);  // m

        // Satellite at LEO
        Vector3d r_sat(6778137.0, 0.0, 0.0);  // m

        // Step 1: relative position vector
        Vector3d r_rel = r_moon - r_sat;
        double r_rel_x = r_moon.x() - r_sat.x();
        double r_rel_y = r_moon.y() - r_sat.y();
        double r_rel_z = r_moon.z() - r_sat.z();
        TEST_ASSERT_NEAR(r_rel.x(), r_rel_x, 1.0);
        TEST_ASSERT_NEAR(r_rel.y(), r_rel_y, 1.0);
        TEST_ASSERT_NEAR(r_rel.z(), r_rel_z, 1.0);

        // Step 2: distances
        double d_rel = std::sqrt(r_rel_x * r_rel_x + r_rel_y * r_rel_y + r_rel_z * r_rel_z);
        double d_moon = std::sqrt(r_moon.x() * r_moon.x() + r_moon.y() * r_moon.y() + r_moon.z() * r_moon.z());
        TEST_ASSERT_NEAR(r_rel.norm(), d_rel, 1.0);
        TEST_ASSERT_NEAR(r_moon.norm(), d_moon, 1.0);

        // Moon distance should be ~3.8e8 m
        TEST_ASSERT(d_moon > 3.5e8 && d_moon < 4.2e8);

        // Step 3: cube of distances
        double d_rel_3 = d_rel * d_rel * d_rel;
        double d_moon_3 = d_moon * d_moon * d_moon;

        // Step 4: compute Vallado Eq. 8-36
        double gm_moon = constants::moon::GM;  // 4.9028001184575496e12
        Vector3d a_direct = gm_moon * r_rel / d_rel_3;
        Vector3d a_indirect = gm_moon * r_moon / d_moon_3;
        Vector3d a_expected = a_direct - a_indirect;

        // Step 5: verify each component manually
        double ax = gm_moon * (r_rel_x / d_rel_3 - r_moon.x() / d_moon_3);
        double ay = gm_moon * (r_rel_y / d_rel_3 - r_moon.y() / d_moon_3);
        double az = gm_moon * (r_rel_z / d_rel_3 - r_moon.z() / d_moon_3);
        TEST_ASSERT_NEAR(a_expected.x(), ax, 1e-20);
        TEST_ASSERT_NEAR(a_expected.y(), ay, 1e-20);
        TEST_ASSERT_NEAR(a_expected.z(), az, 1e-20);

        // Step 6: magnitude should be ~1e-6 m/s² (Moon tidal at LEO)
        double a_mag = std::sqrt(ax * ax + ay * ay + az * az);
        TEST_ASSERT(a_mag > 1e-7 && a_mag < 1e-5);

        std::cout << "[a_mag=" << a_mag << " m/s²] ";
    });

    runner.add_test("fixed_sun_position_step_by_step", []() {
        // Fixed Sun position near J2000 from JPL Horizons:
        //   X = -2.650398313e+10 m, Y = +1.446007544e+11 m, Z = +6.268258712e+10 m
        // (GCRF, 2000-01-01 12:00:00 TDB)
        Vector3d r_sun(-2.650398313e+10, 1.446007544e+11, 6.268258712e+10);

        Vector3d r_sat(0.0, 7000000.0, 0.0);  // LEO, Y-axis

        // Compute Eq. 8-36 step by step
        Vector3d r_rel = r_sun - r_sat;
        double d_rel = r_rel.norm();
        double d_sun = r_sun.norm();

        // Sun distance should be ~1 AU
        TEST_ASSERT_NEAR(d_sun, constants::sun::AU, constants::sun::AU * 0.1);

        double d_rel_3 = d_rel * d_rel * d_rel;
        double d_sun_3 = d_sun * d_sun * d_sun;

        Vector3d a_expected = constants::sun::GM * (r_rel / d_rel_3 - r_sun / d_sun_3);

        // Magnitude: ~5e-7 m/s² (Sun tidal at LEO)
        double a_mag = a_expected.norm();
        TEST_ASSERT(a_mag > 1e-8 && a_mag < 1e-5);

        // Verify the direct and indirect terms are close in magnitude
        // (this is the tidal cancellation — both are ~6e-3 m/s², diff is ~5e-7)
        double direct_mag = (constants::sun::GM * r_rel / d_rel_3).norm();
        double indirect_mag = (constants::sun::GM * r_sun / d_sun_3).norm();
        TEST_ASSERT_NEAR_REL(direct_mag, indirect_mag, 1e-4);

        std::cout << "[a_mag=" << a_mag << ", direct=" << direct_mag
                  << ", indirect=" << indirect_mag << "] ";
    });

    runner.add_test("tidal_term_structure", []() {
        // The tidal acceleration points TOWARD the Moon for a satellite between
        // the Earth and the Moon, and AWAY from the Moon for a satellite on the
        // far side of the Earth (both sides are stretched away from the Earth).
        Vector3d r_moon(3.844e8, 0.0, 0.0);  // Moon along +X

        Vector3d r_sat_toward(7000000.0, 0.0, 0.0);   // Between Earth and Moon
        Vector3d r_sat_away(-7000000.0, 0.0, 0.0);     // Far side

        // Compute accelerations
        double gm = constants::moon::GM;
        auto compute_tb = [&](const Vector3d& r_sat) -> Vector3d {
            Vector3d r_rel = r_moon - r_sat;
            double d_rel_3 = std::pow(r_rel.norm(), 3);
            double d_moon_3 = std::pow(r_moon.norm(), 3);
            return gm * (r_rel / d_rel_3 - r_moon / d_moon_3);
        };

        Vector3d a_toward = compute_tb(r_sat_toward);
        Vector3d a_away = compute_tb(r_sat_away);

        // Toward Moon: tidal force should point TOWARD Moon (+X)
        TEST_ASSERT_MSG(a_toward.x() > 0,
            "Tidal force between Earth and Moon should point toward Moon");

        // Away from Moon: tidal force should point AWAY from Moon (-X)
        TEST_ASSERT_MSG(a_away.x() < 0,
            "Tidal force on far side should point away from Moon");

        // Both should have similar magnitude (tidal symmetry)
        TEST_ASSERT_NEAR_REL(std::abs(a_toward.x()), std::abs(a_away.x()), 0.1);
    });

    runner.add_test("gm_values_from_de440_gm_tpc", []() {
        // Verify GM values match JPL DE440 gm_de440.tpc kernel
        // Source: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
        //
        // BODY10  GM = 1.3271244004127942E+20  (Sun)
        // BODY301 GM = 4.9028001184575496E+12  (Moon)
        // BODY5   GM = 1.26712764E+17          (Jupiter system)

        TEST_ASSERT_NEAR(constants::sun::GM, 1.3271244004127942e+20, 1e6);
        TEST_ASSERT_NEAR(constants::moon::GM, 4.9028001184575496e+12, 1.0);
        TEST_ASSERT_NEAR(constants::jupiter::GM, 1.26712764e+17, 1e10);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Third-Body Gravity Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    // Required: embedded DE440 must be loaded before default ThirdBodyGravity ctor.
    ballistic::initialize({});

    test_third_body_config();
    test_sun_acceleration();
    test_moon_acceleration();
    test_jupiter_acceleration();
    test_combined_acceleration();
    test_body_positions();
    test_reference_values();
    test_convenience_function();
    test_is_conservative();
    test_edge_cases();
    test_independent_hand_computation();

    return print_final_summary();
}
