// Unit tests for the general-relativistic corrections to the acceleration
// (forces/relativistic.hpp), following IERS Conventions 2010 Chapter 10:
//   Schwarzschild    the main correction from Earth's mass (~1.5e-8 m/s^2 at
//                    LEO, low Earth orbit), eq. 10.12;
//   Lense-Thirring   frame dragging from Earth's rotation (~2e-10 m/s^2),
//                    eq. 10.13.
// Reference values are worked out by hand in the comments (circular orbit:
// a = 3 GM^2 / (c^2 r^3)) and checked to machine precision; magnitude cells
// use order-of-magnitude windows.

#include "../test_utils.hpp"
#include <ballistic/forces/relativistic.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::forces;
using namespace ballistic::constants;

// =============================================================================
// Schwarzschild Acceleration Tests
// =============================================================================

void test_schwarzschild() {
    TestRunner runner("SchwarzschildAcceleration");

    runner.add_test("leo_magnitude_order", []() {
        // LEO: r ~ 7000 km, v ~ 7.5 km/s
        // Expected: ~1.5e-8 m/s^2 (value quoted in forces/relativistic.hpp)
        Vector3d pos(7000000.0, 0.0, 0.0);
        Vector3d vel(0.0, 7546.0, 0.0);  // Circular velocity

        Vector3d acc = schwarzschild_acceleration(pos, vel);
        double mag = acc.norm();

        // Should be in range 1e-9 to 1e-7
        TEST_ASSERT(mag > 1e-9);
        TEST_ASSERT(mag < 1e-7);
    });

    runner.add_test("geo_smaller_than_leo", []() {
        // Schwarzschild decreases with altitude
        Vector3d pos_leo(7000000.0, 0.0, 0.0);
        Vector3d vel_leo(0.0, 7546.0, 0.0);

        Vector3d pos_geo(42164000.0, 0.0, 0.0);
        Vector3d vel_geo(0.0, 3075.0, 0.0);

        double mag_leo = schwarzschild_acceleration(pos_leo, vel_leo).norm();
        double mag_geo = schwarzschild_acceleration(pos_geo, vel_geo).norm();

        TEST_ASSERT(mag_geo < mag_leo);
    });

    runner.add_test("reference_value_montenbruck", []() {
        // Reference: Montenbruck & Gill (2000), Section 3.7
        // For LEO at r=7000 km, v=7.5 km/s:
        // Full formula: a = GM/(c^2*r^3) * [(4GM/r - v^2)*r + 4(r·v)*v]
        // For circular orbit (r·v=0): a = GM/(c^2*r^3) * (4GM/r - v^2) * r
        //                             = GM/(c^2*r^2) * (4GM/r - GM/r)
        //                             = 3*GM^2/(c^2*r^3)
        double r = 7000000.0;
        double GM = earth::GM;
        double c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

        // For circular orbit: a = 3*GM^2/(c^2*r^3)
        double expected = 3.0 * GM * GM / (c2 * r * r * r);

        double v = std::sqrt(GM / r);
        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        double actual = schwarzschild_acceleration(pos, vel).norm();

        // Should match to a few percent (exact for circular orbit)
        TEST_ASSERT_NEAR_REL(actual, expected, 0.01);
    });

    runner.add_test("symmetry_opposite_position", []() {
        // Magnitude should be the same for opposite positions
        Vector3d pos1(7000000.0, 0.0, 0.0);
        Vector3d pos2(-7000000.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        double mag1 = schwarzschild_acceleration(pos1, vel).norm();
        double mag2 = schwarzschild_acceleration(pos2, vel).norm();

        TEST_ASSERT_NEAR_REL(mag1, mag2, 1e-10);
    });

    runner.add_test("formula_components_match", []() {
        // Verify the formula: a = coeff * [(4GM/r - v^2)r + 4(r·v)v]
        Vector3d pos(6778137.0, 1000000.0, 500000.0);
        Vector3d vel(1000.0, 7000.0, 500.0);

        double r = pos.norm();
        double r2 = r * r;
        double r3 = r2 * r;
        double v2 = vel.squaredNorm();
        double rdotv = pos.dot(vel);

        double coeff = earth::GM / (C_SQUARED * r3);
        double term1 = 4.0 * earth::GM / r - v2;
        double term2 = 4.0 * rdotv;

        Vector3d expected = coeff * (term1 * pos + term2 * vel);
        Vector3d actual = schwarzschild_acceleration(pos, vel);

        TEST_ASSERT_VEC3_NEAR(actual, expected, 1e-25);
    });

    runner.add_test("circular_orbit_radial_component", []() {
        // For circular orbit, r·v = 0, so only the (4GM/r - v^2)*r term survives
        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);  // Circular velocity
        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        double rdotv = pos.dot(vel);
        TEST_ASSERT_NEAR(rdotv, 0.0, 1e-5);  // Should be zero for circular orbit

        Vector3d acc = schwarzschild_acceleration(pos, vel);
        // For circular orbit: v^2 = GM/r, so 4GM/r - v^2 = 3GM/r
        double coeff = earth::GM / (C_SQUARED * r * r * r);
        double expected_radial = coeff * 3.0 * earth::GM / r * r;
        TEST_ASSERT_NEAR_REL(std::abs(acc.x()), expected_radial, 1e-10);
    });

    runner.add_test("estimate_matches_circular_orbit", []() {
        // Estimate should match circular-orbit Schwarzschild: 3*GM^2/(c^2*r^3)
        double alt = 600000.0;  // 600 km LEO
        double r = earth::RADIUS_EQUATORIAL + alt;
        double expected = 3.0 * earth::GM * earth::GM / (C_SQUARED * r * r * r);
        double estimated = estimate_schwarzschild_magnitude(alt);
        TEST_ASSERT_NEAR_REL(estimated, expected, 0.01);  // 1% tolerance

        // ~1.5e-8 m/s^2 at LEO — sanity check order of magnitude
        TEST_ASSERT(estimated > 1e-9);
        TEST_ASSERT(estimated < 1e-7);
    });

    runner.add_test("estimate_monotonicity", []() {
        // Stronger gravity at lower altitude
        double est_low = estimate_schwarzschild_magnitude(400000.0);
        double est_high = estimate_schwarzschild_magnitude(800000.0);
        TEST_ASSERT(est_low > est_high);
    });

    runner.run();
}

// =============================================================================
// Lense-Thirring Acceleration Tests
// =============================================================================

void test_lense_thirring() {
    TestRunner runner("LenseThirringAcceleration");

    runner.add_test("leo_magnitude_order", []() {
        // Expected: ~1e-10 to 1e-11 m/s^2 at LEO
        Vector3d pos(7000000.0, 0.0, 0.0);
        Vector3d vel(0.0, 7546.0, 0.0);

        Vector3d acc = lense_thirring_acceleration(pos, vel);
        double mag = acc.norm();

        // Should be in range ~1e-12 to 1e-9 (2 orders of magnitude window)
        TEST_ASSERT(mag > 1e-12);
        TEST_ASSERT(mag < 1e-9);
    });

    runner.add_test("much_smaller_than_schwarzschild", []() {
        Vector3d pos(7000000.0, 0.0, 0.0);
        Vector3d vel(0.0, 7546.0, 0.0);

        double sch = schwarzschild_acceleration(pos, vel).norm();
        double lt = lense_thirring_acceleration(pos, vel).norm();

        // Lense-Thirring should be significantly smaller than Schwarzschild
        double ratio = lt / sch;
        TEST_ASSERT(ratio < 0.1);  // At least 10x smaller
        TEST_ASSERT(lt > 0);       // But non-zero
    });

    runner.add_test("equatorial_vs_polar_coupling", []() {
        // Lense-Thirring depends on angular momentum direction (Z axis)
        // For pos=(r,0,0) and vel=(0,v,0): v×J = (v*J,0,0), r·J=0 -> non-zero
        // For pos=(r,0,0) and vel=(0,0,v): v×J = (0,0,0) since v||J -> zero!

        // Equatorial orbit (in X-Y plane): v perpendicular to J
        Vector3d pos_eq(7000000.0, 0.0, 0.0);
        Vector3d vel_eq(0.0, 7546.0, 0.0);
        double mag_eq = lense_thirring_acceleration(pos_eq, vel_eq).norm();

        // Polar orbit velocity along Z (parallel to J): v×J = 0
        Vector3d pos_po(7000000.0, 0.0, 0.0);
        Vector3d vel_po(0.0, 0.0, 7546.0);
        double mag_po = lense_thirring_acceleration(pos_po, vel_po).norm();

        // Equatorial should have non-zero LT; polar (v||J, r·J=0) gives zero
        TEST_ASSERT(mag_eq > 0);
        // mag_po is zero because both terms vanish: r·J=0 and v×J=0
        TEST_ASSERT_NEAR(mag_po, 0.0, 1e-20);
    });

    runner.add_test("z_component_coupling", []() {
        // J is along Z, so position along Z should couple via r·J
        Vector3d pos(0.0, 0.0, 7000000.0);
        Vector3d vel(7546.0, 0.0, 0.0);

        Vector3d acc = lense_thirring_acceleration(pos, vel);
        TEST_ASSERT(acc.norm() > 0);
    });

    runner.run();
}

// =============================================================================
// IERS Reference Value Tests
// =============================================================================

void test_iers_reference_values() {
    TestRunner runner("IERS_ReferenceValues");

    runner.add_test("schwarzschild_circular_leo_iers_eq_10_12", []() {
        // IERS 2010 Eq. 10.12 for Schwarzschild:
        //   a = (GM/(c²r³)) * [(4GM/r - v²)r + 4(r·v)v]
        //
        // For circular orbit: v² = GM/r, r·v = 0
        // Simplifies to: a = (GM/(c²r³)) * (4GM/r - GM/r) * r
        //                  = (GM/(c²r³)) * 3GM/r * r
        //                  = 3GM²/(c²r³)  (radially outward)
        //
        // At r = 7000 km:
        //   GM = 3.986004418e14 m³/s²
        //   c  = 299792458 m/s
        //   r  = 7.0e6 m
        //   a  = 3 * (3.986004418e14)² / ((299792458)² * (7.0e6)³)
        //      = 3 * 1.58882e29 / (8.98755e16 * 3.43e20)
        //      = 4.76646e29 / 3.08273e37
        //      = 1.5463e-8 m/s²

        const double r = 7000000.0;
        const double GM = earth::GM;
        const double c = SPEED_OF_LIGHT;
        const double c2 = c * c;

        // Exact expected value for circular orbit
        const double expected = 3.0 * GM * GM / (c2 * r * r * r);

        // Verify the expected value is in the right ballpark (~1.5e-8)
        TEST_ASSERT(expected > 1e-8);
        TEST_ASSERT(expected < 2e-8);

        // Circular orbit velocity
        const double v = std::sqrt(GM / r);
        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        // Verify r·v = 0 (circular orbit)
        TEST_ASSERT_NEAR(pos.dot(vel), 0.0, 1e-5);

        Vector3d acc = schwarzschild_acceleration(pos, vel);

        // Magnitude should match exact formula to machine precision
        TEST_ASSERT_NEAR_REL(acc.norm(), expected, 1e-12);

        // Direction should be radially outward (+x for pos on +x axis)
        // For circular orbit: a = coeff * (3GM/r) * r_hat
        TEST_ASSERT(acc.x() > 0);  // Outward
        TEST_ASSERT_NEAR(acc.y(), 0.0, acc.norm() * 1e-12);
        TEST_ASSERT_NEAR(acc.z(), 0.0, acc.norm() * 1e-12);
    });

    runner.add_test("schwarzschild_different_radii", []() {
        // For circular orbits, a = 3GM²/(c²r³) → a ∝ 1/r³
        // Verify scaling between two altitudes
        const double GM = earth::GM;
        const double c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

        const double r1 = 6778137.0;  // 400 km altitude
        const double r2 = 42164000.0; // GEO

        const double expected1 = 3.0 * GM * GM / (c2 * r1 * r1 * r1);
        const double expected2 = 3.0 * GM * GM / (c2 * r2 * r2 * r2);

        Vector3d pos1(r1, 0.0, 0.0);
        Vector3d vel1(0.0, std::sqrt(GM / r1), 0.0);
        Vector3d pos2(r2, 0.0, 0.0);
        Vector3d vel2(0.0, std::sqrt(GM / r2), 0.0);

        double actual1 = schwarzschild_acceleration(pos1, vel1).norm();
        double actual2 = schwarzschild_acceleration(pos2, vel2).norm();

        TEST_ASSERT_NEAR_REL(actual1, expected1, 1e-12);
        TEST_ASSERT_NEAR_REL(actual2, expected2, 1e-12);

        // Verify 1/r³ scaling
        double ratio_expected = (r2 * r2 * r2) / (r1 * r1 * r1);
        double ratio_actual = actual1 / actual2;
        TEST_ASSERT_NEAR_REL(ratio_actual, ratio_expected, 1e-10);
    });

    runner.add_test("lense_thirring_magnitude_estimate", []() {
        // IERS 2010 Eq. 10.13 for Lense-Thirring:
        //   a_LT = (2GM/(c²r³)) * [(3/r²)(r×v)(r·J) + (v×J)]
        //
        // For equatorial circular orbit (pos on x, vel on y):
        //   r = (r, 0, 0), v = (0, v, 0), J = J_z * (0, 0, 1)
        //   r·J = 0
        //   v×J = (0,v,0)×(0,0,J_z) = (v*J_z, 0, 0)
        //   a_LT = (2GM/(c²r³)) * (v*J_z, 0, 0)
        //
        // Magnitude: |a_LT| = 2GM*v*J_z/(c²r³)
        //   ≈ 2 * 3.986e14 * 7546 * 9.8e8 / (8.988e16 * 3.43e20)
        //   ≈ 5.9e27 / 3.08e37 ≈ 1.9e-10 m/s²

        const double r = 7000000.0;
        const double v = std::sqrt(earth::GM / r);
        const double c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;

        // J/M for Earth (from relativistic.hpp)
        const double J_per_M = EARTH_J_PER_M;

        // Expected: 2*GM*v*J/(c²*r³) for equatorial orbit with r·J=0
        const double expected = 2.0 * earth::GM * v * J_per_M / (c2 * r * r * r);

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);
        Vector3d acc = lense_thirring_acceleration(pos, vel);

        TEST_ASSERT_NEAR_REL(acc.norm(), expected, 1e-10);

        // For this geometry, acceleration should be in +x direction
        TEST_ASSERT(acc.x() > 0);
        TEST_ASSERT_NEAR(acc.y(), 0.0, acc.norm() * 1e-10);
        TEST_ASSERT_NEAR(acc.z(), 0.0, acc.norm() * 1e-10);
    });

    runner.run();
}

// =============================================================================
// Combined Relativistic Tests
// =============================================================================

void test_combined_relativistic() {
    TestRunner runner("CombinedRelativistic");

    runner.add_test("total_equals_sum_of_components", []() {
        RelativisticConfig config;
        config.enable_schwarzschild = true;
        config.enable_lense_thirring = true;
        RelativisticCorrections rel(config);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        Vector3d total = rel.acceleration(epoch, state);
        Vector3d sch = schwarzschild_acceleration(state.position, state.velocity);
        Vector3d lt = lense_thirring_acceleration(state.position, state.velocity);

        TEST_ASSERT_VEC3_NEAR(total, sch + lt, 1e-25);
    });

    runner.add_test("disable_both_gives_zero", []() {
        RelativisticConfig config;
        config.enable_schwarzschild = false;
        config.enable_lense_thirring = false;
        RelativisticCorrections rel(config);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        Vector3d acc = rel.acceleration(epoch, state);
        TEST_ASSERT_VEC3_NEAR(acc, Vector3d::Zero(), 1e-25);
    });

    runner.add_test("schwarzschild_only", []() {
        RelativisticConfig config;
        config.enable_schwarzschild = true;
        config.enable_lense_thirring = false;
        RelativisticCorrections rel(config);

        Epoch epoch = Epoch::j2000();
        StateVector state;
        state.position = Vector3d(7000000.0, 0.0, 0.0);
        state.velocity = Vector3d(0.0, 7546.0, 0.0);

        Vector3d total = rel.acceleration(epoch, state);
        Vector3d sch = schwarzschild_acceleration(state.position, state.velocity);

        TEST_ASSERT_VEC3_NEAR(total, sch, 1e-25);
    });

    runner.add_test("is_conservative_schwarzschild_only", []() {
        // Schwarzschild alone is conservative (derivable from potential)
        RelativisticConfig cfg;
        cfg.enable_schwarzschild = true;
        cfg.enable_lense_thirring = false;
        RelativisticCorrections rel(cfg);
        TEST_ASSERT_TRUE(rel.is_conservative());
    });

    runner.add_test("is_not_conservative_with_lense_thirring", []() {
        // Lense-Thirring is velocity-dependent -> non-conservative
        RelativisticConfig cfg;
        cfg.enable_schwarzschild = true;
        cfg.enable_lense_thirring = true;
        RelativisticCorrections rel(cfg);
        TEST_ASSERT_TRUE(!rel.is_conservative());
    });

    runner.add_test("config_access", []() {
        RelativisticCorrections rel;
        rel.enable_schwarzschild(true);
        rel.enable_lense_thirring(true);

        TEST_ASSERT_TRUE(rel.config().enable_schwarzschild);
        TEST_ASSERT_TRUE(rel.config().enable_lense_thirring);
    });

    runner.add_test("error_estimate_consistency", []() {
        double alt = 400000.0;
        double duration = 3600.0;
        double error = estimate_schwarzschild_error(alt, duration);

        // Error = 0.5 * a * t^2 where a = 3*GM^2/(c^2*r^3) ~ 2e-8 m/s^2
        // Over 1 hour: ~ 0.5 * 2e-8 * 3600^2 ~ 0.13 m
        TEST_ASSERT(error > 0.01);   // At least 1 cm
        TEST_ASSERT(error < 10.0);   // Less than 10 m

        double error_short = estimate_schwarzschild_error(alt, 60.0);
        TEST_ASSERT(error > error_short);
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Relativistic Corrections Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_schwarzschild();
    test_lense_thirring();
    test_iers_reference_values();
    test_combined_relativistic();

    return print_final_summary();
}
