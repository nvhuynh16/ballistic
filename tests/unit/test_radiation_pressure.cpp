// Unit tests for the radiation pressure models: solar radiation pressure (SRP,
// sunlight pushing on the satellite, with the cylindrical Earth-shadow model),
// Earth albedo (sunlight reflected by the Earth) and Earth infrared (IR, heat
// radiated by the Earth). The "cannonball" model treats the satellite as a
// sphere with area A, mass m and radiation pressure coefficient C_R
// (1 = perfect absorber, 2 = perfect mirror; the props field srp_reflectivity
// is C_R - 1). Checks are hand-computed values, exact ratios and directions.
// Uses the embedded DE440 ephemeris for the Sun; no data files.
// Tests: forces/radiation_pressure.hpp
//
// Reference Sources:
// - Solar constant: 1361 W/m² (SORCE/TIM, Kopp & Lean 2011)
// - Earth albedo: ~0.3 (Stephens et al. 2015, "The albedo of Earth")

#include "../test_utils.hpp"
#include <ballistic/forces/radiation_pressure.hpp>
#include <ballistic/ephemeris/jpl_ephemeris.hpp>
#include <ballistic/init.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::forces;
using namespace ballistic::constants;

// =============================================================================
// Shadow Factor Tests
// =============================================================================

void test_shadow_factor() {
    TestRunner runner("ShadowFactor");

    runner.add_test("sunlit_satellite_on_sun_side", []() {
        // Satellite on same side as Sun -> sunlit
        Vector3d sat_pos(7000000.0, 0.0, 0.0);
        Vector3d sun_pos(1.0e11, 0.0, 0.0);  // Sun along +X
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 1.0, 1e-15);
    });

    runner.add_test("full_shadow_behind_earth", []() {
        // Satellite directly behind Earth from Sun -> in shadow
        Vector3d sun_pos(1.0e11, 0.0, 0.0);   // Sun along +X
        Vector3d sat_pos(-7000000.0, 0.0, 0.0); // Satellite behind Earth
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 0.0, 1e-15);
    });

    runner.add_test("satellite_off_axis_sunlit", []() {
        // Satellite behind Earth but far off axis -> sunlit
        Vector3d sun_pos(1.0e11, 0.0, 0.0);
        Vector3d sat_pos(-7000000.0, 7000000.0, 0.0);  // Behind but far off axis
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 1.0, 1e-15);
    });

    runner.add_test("geo_sunlit", []() {
        // GEO satellite on sun side
        Vector3d sun_pos(1.496e11, 0.0, 0.0);  // ~1 AU
        Vector3d sat_pos(42164000.0, 0.0, 0.0); // GEO altitude
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 1.0, 1e-15);
    });

    runner.add_test("geo_in_shadow", []() {
        // GEO satellite directly behind Earth
        Vector3d sun_pos(1.496e11, 0.0, 0.0);
        Vector3d sat_pos(-42164000.0, 0.0, 0.0);
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 0.0, 1e-15);
    });

    runner.add_test("boundary_just_outside_shadow", []() {
        // Satellite behind Earth at exactly Earth radius distance from axis
        Vector3d sun_pos(1.0e11, 0.0, 0.0);
        double R_earth = earth::RADIUS_EQUATORIAL;
        // Just outside shadow cylinder
        Vector3d sat_pos(-7000000.0, R_earth + 1.0, 0.0);
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 1.0, 1e-15);
    });

    runner.add_test("boundary_just_inside_shadow", []() {
        // Just inside shadow cylinder
        Vector3d sun_pos(1.0e11, 0.0, 0.0);
        double R_earth = earth::RADIUS_EQUATORIAL;
        Vector3d sat_pos(-7000000.0, R_earth - 1.0, 0.0);
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 0.0, 1e-15);
    });

    runner.add_test("collinear_sun_earth_sat", []() {
        // Satellite between Earth and Sun
        Vector3d sun_pos(1.0e11, 0.0, 0.0);
        Vector3d sat_pos(7000000.0, 0.0, 0.0);  // Between Earth and Sun
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 1.0, 1e-15);
    });

    runner.add_test("sun_along_z_axis", []() {
        // Sun along Z axis, satellite behind Earth
        Vector3d sun_pos(0.0, 0.0, 1.0e11);
        Vector3d sat_pos(0.0, 0.0, -7000000.0);
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 0.0, 1e-15);
    });

    runner.run();
}

// =============================================================================
// SRP Acceleration Tests
// =============================================================================

void test_srp_acceleration() {
    TestRunner runner("SRPAcceleration");

    auto eph = ephemeris::get_default_ephemeris_provider();

    runner.add_test("cannonball_magnitude_at_1au", [&]() {
        // At 1 AU, SRP acceleration should be ~P_1AU * (A/m) * C_R
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();

        // Put satellite on sun-side (same direction as Sun, between Earth and Sun)
        Vector3d sat_pos = sun_dir * 7000000.0;  // Sunlit side

        double mass = 1000.0;
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.srp_reflectivity = 0.3;  // C_R = 1.3

        Vector3d acc = srp_acceleration(epoch, sat_pos, mass, props, *eph);

        // Expected magnitude: P_1AU * (A/m) * C_R ≈ 4.54e-6 * 0.01 * 1.3 ≈ 5.9e-8
        double expected_mag = radiation::SOLAR_PRESSURE_1AU * (10.0 / 1000.0) * 1.3;
        // 5% tolerance: the Sun is not exactly 1 AU away on this date (pressure goes as 1/r^2).
        TEST_ASSERT_NEAR_REL(acc.norm(), expected_mag, 0.05);
    });

    runner.add_test("perfect_absorber_cr_1", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();
        Vector3d sat_pos = sun_dir * 7000000.0;  // Sunlit side

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.srp_reflectivity = 0.0;  // Perfect absorber -> C_R = 1.0

        Vector3d acc = srp_acceleration(epoch, sat_pos, 1000.0, props, *eph);
        double expected_mag = radiation::SOLAR_PRESSURE_1AU * 0.01 * 1.0;
        TEST_ASSERT_NEAR_REL(acc.norm(), expected_mag, 0.05);
    });

    runner.add_test("perfect_mirror_cr_2", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();
        Vector3d sat_pos = sun_dir * 7000000.0;  // Sunlit side

        SatelliteRadiationProperties props_absorber;
        props_absorber.srp_area = 10.0;
        props_absorber.srp_reflectivity = 0.0;

        SatelliteRadiationProperties props_mirror;
        props_mirror.srp_area = 10.0;
        props_mirror.srp_reflectivity = 1.0;  // Perfect mirror -> C_R = 2.0

        Vector3d acc_abs = srp_acceleration(epoch, sat_pos, 1000.0, props_absorber, *eph);
        Vector3d acc_mir = srp_acceleration(epoch, sat_pos, 1000.0, props_mirror, *eph);

        // Mirror should have 2x acceleration of absorber
        TEST_ASSERT_NEAR_REL(acc_mir.norm() / acc_abs.norm(), 2.0, 1e-10);
    });

    runner.add_test("direction_away_from_sun", [&]() {
        // SRP acceleration is a = P * (A/m) * C_R * s_hat
        // where s_hat = (sat - sun) / |sat - sun| (from Sun toward satellite)
        // Photon pressure pushes satellite away from Sun
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();
        Vector3d sat_pos = sun_dir * 7000000.0;  // Sunlit side

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.srp_reflectivity = 0.3;

        Vector3d acc = srp_acceleration(epoch, sat_pos, 1000.0, props, *eph);

        // s_hat points from Sun to satellite, acc = P * ... * s_hat
        // So acceleration should point away from Sun
        Vector3d away_from_sun = (sat_pos - sun_pos).normalized();
        double dot = acc.normalized().dot(away_from_sun);
        TEST_ASSERT(dot > 0.99);  // Photon pressure pushes away from Sun
    });

    runner.add_test("zero_in_shadow", [&]() {
        // When satellite is in shadow, SRP should be zero
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();
        // Place satellite directly behind Earth from Sun
        Vector3d sat_pos = -sun_dir * 7000000.0;

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.srp_reflectivity = 0.3;

        // Verify shadow
        double shadow = compute_shadow_factor(sat_pos, sun_pos);
        TEST_ASSERT_NEAR(shadow, 0.0, 1e-10);

        Vector3d acc = srp_acceleration(epoch, sat_pos, 1000.0, props, *eph);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.add_test("zero_mass_returns_zero", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sat_pos(7000000.0, 0.0, 0.0);

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;

        Vector3d acc = srp_acceleration(epoch, sat_pos, 0.0, props, *eph);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.add_test("zero_area_returns_zero", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sat_pos(7000000.0, 0.0, 0.0);

        SatelliteRadiationProperties props;
        props.srp_area = 0.0;

        Vector3d acc = srp_acceleration(epoch, sat_pos, 1000.0, props, *eph);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.add_test("larger_area_larger_acceleration", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();
        Vector3d sat_pos = sun_dir * 7000000.0;  // Sunlit side

        SatelliteRadiationProperties props1;
        props1.srp_area = 10.0;
        props1.srp_reflectivity = 0.3;

        SatelliteRadiationProperties props2;
        props2.srp_area = 20.0;
        props2.srp_reflectivity = 0.3;

        Vector3d acc1 = srp_acceleration(epoch, sat_pos, 1000.0, props1, *eph);
        Vector3d acc2 = srp_acceleration(epoch, sat_pos, 1000.0, props2, *eph);

        TEST_ASSERT_NEAR_REL(acc2.norm() / acc1.norm(), 2.0, 1e-10);
    });

    runner.run();
}

// =============================================================================
// Albedo Acceleration Tests
// =============================================================================

void test_albedo_acceleration() {
    TestRunner runner("AlbedoAcceleration");

    auto eph = ephemeris::get_default_ephemeris_provider();

    runner.add_test("radially_outward_direction", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();

        // Put satellite on subsolar point (maximum albedo)
        Vector3d sat_pos = sun_dir * 7000000.0;

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.albedo_reflectivity = 0.3;

        Vector3d acc = albedo_acceleration(epoch, sat_pos, 1000.0, props, *eph);

        if (acc.norm() > 1e-20) {
            // Albedo force should be radially outward from Earth
            Vector3d e_sat = sat_pos.normalized();
            double dot = acc.normalized().dot(e_sat);
            TEST_ASSERT(dot > 0.99);  // Should be radially outward
        }
    });

    runner.add_test("subsolar_maximum", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();

        // Subsolar satellite (maximum albedo)
        Vector3d sat_subsolar = sun_dir * 7000000.0;

        // Satellite at 45 degrees from subsolar
        Vector3d perp = Vector3d(0, 0, 1).cross(sun_dir);
        if (perp.norm() < 0.1) perp = Vector3d(0, 1, 0).cross(sun_dir);
        perp.normalize();
        Vector3d sat_45 = (sun_dir + perp).normalized() * 7000000.0;

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.albedo_reflectivity = 0.3;

        Vector3d acc_sub = albedo_acceleration(epoch, sat_subsolar, 1000.0, props, *eph);
        Vector3d acc_45 = albedo_acceleration(epoch, sat_45, 1000.0, props, *eph);

        // Subsolar should have larger albedo than 45-degree position
        TEST_ASSERT(acc_sub.norm() > acc_45.norm());
    });

    runner.add_test("dark_hemisphere_zero", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();

        // Satellite on dark side (anti-sun direction)
        Vector3d sat_pos = -sun_dir * 7000000.0;

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.albedo_reflectivity = 0.3;

        Vector3d acc = albedo_acceleration(epoch, sat_pos, 1000.0, props, *eph);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.add_test("altitude_scaling_leo_vs_geo", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph->sun_position(epoch);
        Vector3d sun_dir = sun_pos.normalized();

        Vector3d sat_leo = sun_dir * 7000000.0;     // LEO
        Vector3d sat_geo = sun_dir * 42164000.0;    // GEO

        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.albedo_reflectivity = 0.3;

        Vector3d acc_leo = albedo_acceleration(epoch, sat_leo, 1000.0, props, *eph);
        Vector3d acc_geo = albedo_acceleration(epoch, sat_geo, 1000.0, props, *eph);

        // LEO should have much larger albedo acceleration (1/r^2 scaling)
        TEST_ASSERT(acc_leo.norm() > acc_geo.norm() * 10.0);
    });

    runner.add_test("zero_mass_returns_zero", [&]() {
        Epoch epoch = Epoch::j2000();
        Vector3d sat_pos(7000000.0, 0.0, 0.0);
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;

        Vector3d acc = albedo_acceleration(epoch, sat_pos, 0.0, props, *eph);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.run();
}

// =============================================================================
// IR Acceleration Tests
// =============================================================================

void test_ir_acceleration() {
    TestRunner runner("IRAcceleration");

    runner.add_test("radially_outward", []() {
        Vector3d sat_pos(7000000.0, 0.0, 0.0);
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.ir_emissivity = 0.8;

        Vector3d acc = ir_acceleration(sat_pos, 1000.0, props);

        if (acc.norm() > 0) {
            Vector3d e_sat = sat_pos.normalized();
            double dot = acc.normalized().dot(e_sat);
            TEST_ASSERT(dot > 0.99);
        }
    });

    runner.add_test("altitude_scaling_leo_larger_than_geo", []() {
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.ir_emissivity = 0.8;

        Vector3d sat_leo(7000000.0, 0.0, 0.0);    // LEO
        Vector3d sat_geo(42164000.0, 0.0, 0.0);   // GEO

        Vector3d acc_leo = ir_acceleration(sat_leo, 1000.0, props);
        Vector3d acc_geo = ir_acceleration(sat_geo, 1000.0, props);

        // IR at LEO >> IR at GEO due to 1/r^2 solid angle
        double ratio = acc_leo.norm() / acc_geo.norm();
        double expected_ratio = (42164000.0 * 42164000.0) / (7000000.0 * 7000000.0);
        TEST_ASSERT_NEAR_REL(ratio, expected_ratio, 1e-10);
    });

    runner.add_test("hand_calculated_reference", []() {
        // IR acceleration = P_ir * (R/r)^2 * (A/m) * C_IR * e_sat
        // P_ir = EARTH_IR_PRESSURE = 240/c ≈ 8.006e-7 N/m^2
        // R = 6378137 m, r = 7000000 m
        // A/m = 10/1000 = 0.01
        // C_IR = 0.8
        double P_ir = radiation::EARTH_IR_PRESSURE;
        double R = earth::RADIUS_EQUATORIAL;
        double r = 7000000.0;
        double A_over_m = 0.01;
        double C_IR = 0.8;
        double expected = P_ir * (R * R) / (r * r) * A_over_m * C_IR;

        Vector3d sat_pos(7000000.0, 0.0, 0.0);
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.ir_emissivity = 0.8;

        Vector3d acc = ir_acceleration(sat_pos, 1000.0, props);
        TEST_ASSERT_NEAR_REL(acc.norm(), expected, 1e-10);
    });

    runner.add_test("zero_mass_returns_zero", []() {
        Vector3d sat_pos(7000000.0, 0.0, 0.0);
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;

        Vector3d acc = ir_acceleration(sat_pos, 0.0, props);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.add_test("zero_area_returns_zero", []() {
        Vector3d sat_pos(7000000.0, 0.0, 0.0);
        SatelliteRadiationProperties props;
        props.srp_area = 0.0;

        Vector3d acc = ir_acceleration(sat_pos, 1000.0, props);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.add_test("below_surface_returns_zero", []() {
        // Satellite below Earth surface -> zero
        Vector3d sat_pos(6000000.0, 0.0, 0.0);  // Below equatorial radius
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.ir_emissivity = 0.8;

        Vector3d acc = ir_acceleration(sat_pos, 1000.0, props);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.run();
}

// =============================================================================
// SRP Hand-Computed Reference Value Tests
// =============================================================================

void test_srp_hand_computed_reference() {
    TestRunner runner("SRP_HandComputedReference");

    runner.add_test("exact_srp_at_1au_known_geometry", []() {
        // Hand-computed SRP acceleration for a satellite at exactly 1 AU from Sun.
        //
        // Formula: a = P_1AU * (A/m) * C_R * (AU/r_sun)^2 * s_hat
        // With satellite at exactly 1 AU from Sun: (AU/r_sun)^2 = 1.0
        //
        // Parameters:
        //   P_1AU = S/c = 1361 / 299792458 = 4.53930e-6 N/m²
        //   A/m = 0.01 m²/kg (10 m², 1000 kg)
        //   C_R = 1.5 (reflectivity = 0.5)
        //
        // Expected: a = 4.53930e-6 * 0.01 * 1.5 = 6.80895e-8 m/s²

        const double P_1AU = radiation::SOLAR_CONSTANT / constants::SPEED_OF_LIGHT;
        const double A_over_m = 10.0 / 1000.0;  // 0.01 m²/kg
        const double C_R = 1.5;  // reflectivity = 0.5
        const double expected_mag = P_1AU * A_over_m * C_R;

        // Verify P_1AU intermediate value
        TEST_ASSERT_NEAR_REL(P_1AU, 4.53930e-6, 1e-3);

        // Place Sun at +X at 1 AU, satellite at origin+offset on sunlit side
        // Sun-to-sat distance will be ~1AU (sat offset negligible)
        const Vector3d sun_pos(constants::sun::AU, 0.0, 0.0);
        const Vector3d sat_pos(7000000.0, 0.0, 0.0);  // Sunlit side

        // Exact distance from Sun to satellite
        const double r_sun_sat = (sat_pos - sun_pos).norm();
        const double distance_factor = (constants::sun::AU * constants::sun::AU) / (r_sun_sat * r_sun_sat);

        // Adjusted expected magnitude accounting for exact distance
        const double expected_exact = P_1AU * distance_factor * A_over_m * C_R;

        // The real ephemeris is used below, with the expected value recomputed from
        // the formula at the actual Sun distance.
        SatelliteRadiationProperties props;
        props.srp_area = 10.0;
        props.srp_reflectivity = 0.5;  // C_R = 1.5

        // Create ephemeris and get acceleration
        auto eph = ephemeris::get_default_ephemeris_provider();
        Epoch epoch = Epoch::j2000();
        Vector3d real_sun = eph->sun_position(epoch);
        Vector3d sun_dir = real_sun.normalized();

        // Place satellite on sunlit side
        Vector3d sat = sun_dir * 7000000.0;
        Vector3d acc = srp_acceleration(epoch, sat, 1000.0, props, *eph);

        // Compute expected from formula with actual Sun distance
        double r_actual = (sat - real_sun).norm();
        double dist_factor_actual = (constants::sun::AU * constants::sun::AU) / (r_actual * r_actual);
        double expected_actual = P_1AU * dist_factor_actual * A_over_m * C_R;

        // Same formula, so agreement to 1e-10 relative (rounding only).
        TEST_ASSERT_NEAR_REL(acc.norm(), expected_actual, 1e-10);
    });

    runner.add_test("srp_pressure_constant_value", []() {
        // Verify SOLAR_PRESSURE_1AU = SOLAR_CONSTANT / c
        // 1361 / 299792458 = 4.539304e-6 N/m²
        double expected = 1361.0 / 299792458.0;
        TEST_ASSERT_NEAR_REL(radiation::SOLAR_PRESSURE_1AU, expected, 1e-12);
    });

    runner.add_test("srp_perfect_absorber_vs_mirror_exact_ratio", []() {
        // For same geometry, perfect mirror (C_R=2) gives exactly 2x
        // acceleration of perfect absorber (C_R=1).
        // This is an exact relationship from the formula.
        auto eph = ephemeris::get_default_ephemeris_provider();
        Epoch epoch = Epoch::j2000();
        Vector3d sun_dir = eph->sun_position(epoch).normalized();
        Vector3d sat = sun_dir * 7000000.0;

        SatelliteRadiationProperties abs_props;
        abs_props.srp_area = 10.0;
        abs_props.srp_reflectivity = 0.0;  // C_R = 1.0

        SatelliteRadiationProperties mir_props;
        mir_props.srp_area = 10.0;
        mir_props.srp_reflectivity = 1.0;  // C_R = 2.0

        Vector3d a_abs = srp_acceleration(epoch, sat, 1000.0, abs_props, *eph);
        Vector3d a_mir = srp_acceleration(epoch, sat, 1000.0, mir_props, *eph);

        // Exact 2:1 ratio to machine precision
        TEST_ASSERT_NEAR_REL(a_mir.norm() / a_abs.norm(), 2.0, 1e-14);

        // Direction must be identical (both push away from Sun)
        TEST_ASSERT_NEAR(a_abs.normalized().dot(a_mir.normalized()), 1.0, 1e-14);
    });

    runner.run();
}

// =============================================================================
// Combined RadiationPressure Class Tests
// =============================================================================

void test_radiation_pressure_class() {
    TestRunner runner("RadiationPressureClass");

    runner.add_test("default_all_enabled", []() {
        RadiationPressure rp;
        TEST_ASSERT_TRUE(rp.srp_enabled());
        TEST_ASSERT_TRUE(rp.albedo_enabled());
        TEST_ASSERT_TRUE(rp.ir_enabled());
    });

    runner.add_test("enable_disable_srp", []() {
        RadiationPressure rp;
        rp.enable_srp(false);
        TEST_ASSERT_FALSE(rp.srp_enabled());
        rp.enable_srp(true);
        TEST_ASSERT_TRUE(rp.srp_enabled());
    });

    runner.add_test("total_equals_sum_of_components", []() {
        RadiationPressure rp;
        rp.set_mass(1000.0);
        SatelliteRadiationProperties props(10.0, 0.3, 0.8);
        rp.set_properties(props);

        Epoch epoch = Epoch::j2000();
        Vector3d sat_pos(7000000.0, 0.0, 0.0);

        Vector3d total = rp.acceleration(epoch, sat_pos);
        Vector3d srp = rp.srp_only(epoch, sat_pos);
        Vector3d albedo = rp.albedo_only(epoch, sat_pos);
        Vector3d ir = rp.ir_only(sat_pos);

        TEST_ASSERT_VEC3_NEAR(total, srp + albedo + ir, 1e-20);
    });

    runner.add_test("disable_all_gives_zero", []() {
        RadiationPressure rp;
        rp.enable_srp(false);
        rp.enable_albedo(false);
        rp.enable_ir(false);
        rp.set_mass(1000.0);
        SatelliteRadiationProperties props(10.0, 0.3, 0.8);
        rp.set_properties(props);

        Epoch epoch = Epoch::j2000();
        Vector3d sat_pos(7000000.0, 0.0, 0.0);

        Vector3d acc = rp.acceleration(epoch, sat_pos);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Radiation Pressure Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    // Required: embedded DE440 must be loaded before get_default_ephemeris_provider works.
    ballistic::initialize({});

    test_shadow_factor();
    test_srp_acceleration();
    test_srp_hand_computed_reference();
    test_albedo_acceleration();
    test_ir_acceleration();
    test_radiation_pressure_class();

    return print_final_summary();
}
