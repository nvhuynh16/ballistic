// Unit tests for point-mass gravity (GM/r^2) and the zonal harmonics J2..J15
// (the Earth's oblateness and other north-south-symmetric shape terms), plus the
// potential, orbital energy and angular momentum helpers. Expected values are
// textbook GM/r^2 numbers, symmetries, and exact identities (to rounding).
// No data files.
// Tests: point_mass.hpp

#include "../test_utils.hpp"
#include <cstdio>
#include <ballistic/gravity/point_mass.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::gravity;
using namespace ballistic::constants;

// =============================================================================
// Test Suites
// =============================================================================

void test_point_mass_acceleration() {
    TestRunner runner("PointMassAcceleration");

    runner.add_test("surface_gravity", []() {
        // Surface of Earth at equator
        Vector3d pos(earth::RADIUS_EQUATORIAL, 0.0, 0.0);
        Vector3d acc = point_mass_acceleration(pos);

        // Should be ~9.8 m/s^2 toward Earth center
        // Note: Point mass gives GM/r^2 ≈ 9.798, slightly different from standard 9.80665 (includes centrifugal)
        double g = acc.norm();
        TEST_ASSERT_NEAR(g, 9.8, 0.2);  // Within 0.2 m/s^2 of ~9.8

        // Direction should be toward center (negative X)
        TEST_ASSERT(acc.x() < 0);
        TEST_ASSERT_NEAR(acc.y(), 0.0, 1e-10);
        TEST_ASSERT_NEAR(acc.z(), 0.0, 1e-10);
    });

    runner.add_test("leo_gravity", []() {
        // LEO at 400 km altitude
        double alt = 400000.0;  // meters
        Vector3d pos(earth::RADIUS_EQUATORIAL + alt, 0.0, 0.0);
        Vector3d acc = point_mass_acceleration(pos);

        // g at 400 km ≈ GM/r² = 3.986e14 / (6778e3)² ≈ 8.674 m/s²
        double g = acc.norm();
        TEST_ASSERT_NEAR(g, 8.674, 0.01);
    });

    runner.add_test("geo_gravity", []() {
        // GEO at 35786 km altitude (42164 km from center)
        double r_geo = 42164000.0;
        Vector3d pos(r_geo, 0.0, 0.0);
        Vector3d acc = point_mass_acceleration(pos);

        // g at GEO ≈ GM/r² = 3.986e14 / (42164e3)² ≈ 0.2242 m/s²
        double g = acc.norm();
        TEST_ASSERT_NEAR(g, 0.2242, 0.001);
    });

    runner.add_test("moon_distance_gravity", []() {
        // Moon distance (~384,400 km)
        double r_moon = 384400000.0;
        Vector3d pos(r_moon, 0.0, 0.0);
        Vector3d acc = point_mass_acceleration(pos);

        // g at Moon distance ≈ GM/r² = 3.986e14 / (3.844e8)² ≈ 0.0027 m/s²
        double g = acc.norm();
        TEST_ASSERT_NEAR(g, 0.0027, 0.0005);
    });

    runner.add_test("direction_toward_center", []() {
        // Various positions - acceleration always toward origin
        std::vector<Vector3d> positions = {
            Vector3d(7000000.0, 0.0, 0.0),
            Vector3d(0.0, 7000000.0, 0.0),
            Vector3d(0.0, 0.0, 7000000.0),
            Vector3d(5000000.0, 5000000.0, 0.0),
            Vector3d(4000000.0, 4000000.0, 4000000.0),
        };

        for (const auto& pos : positions) {
            Vector3d acc = point_mass_acceleration(pos);

            // Acceleration should point opposite to position (toward center)
            double dot = acc.dot(pos);
            TEST_ASSERT(dot < 0);

            // Acceleration magnitude should depend only on distance
            double r = pos.norm();
            double expected_g = earth::GM / (r * r);
            TEST_ASSERT_NEAR(acc.norm(), expected_g, 0.001);  // Within 1 mm/s^2
        }
    });

    runner.add_test("inverse_square_law", []() {
        // Verify inverse square relationship
        Vector3d pos1(7000000.0, 0.0, 0.0);
        Vector3d pos2(14000000.0, 0.0, 0.0);  // 2x distance

        Vector3d acc1 = point_mass_acceleration(pos1);
        Vector3d acc2 = point_mass_acceleration(pos2);

        // acc2 should be 1/4 of acc1
        TEST_ASSERT_NEAR(acc2.norm() / acc1.norm(), 0.25, 1e-10);
    });

    runner.add_test("custom_mu", []() {
        // Use Moon's GM
        Vector3d pos(1738000.0, 0.0, 0.0);  // Moon radius
        Vector3d acc = point_mass_acceleration(pos, moon::GM);

        // Surface gravity on Moon ≈ 1.623 m/s²
        TEST_ASSERT_NEAR(acc.norm(), 1.623, 0.01);
    });

    runner.run();
}

void test_j2_acceleration() {
    TestRunner runner("J2Acceleration");

    runner.add_test("equatorial_j2", []() {
        // Equatorial position
        Vector3d pos(7000000.0, 0.0, 0.0);
        Vector3d acc_j2 = j2_acceleration(pos);

        // J2 effect at equator should be primarily radial
        // ay and az should be near zero
        TEST_ASSERT(acc_j2.x() != 0);  // Non-zero effect
        TEST_ASSERT_NEAR(acc_j2.y(), 0.0, 1e-10);
        TEST_ASSERT_NEAR(acc_j2.z(), 0.0, 1e-10);
    });

    runner.add_test("polar_j2", []() {
        // Polar position
        Vector3d pos(0.0, 0.0, 7000000.0);
        Vector3d acc_j2 = j2_acceleration(pos);

        // J2 effect at pole should be primarily in z-direction
        TEST_ASSERT(acc_j2.z() != 0);  // Non-zero effect
        TEST_ASSERT_NEAR(acc_j2.x(), 0.0, 1e-10);
        TEST_ASSERT_NEAR(acc_j2.y(), 0.0, 1e-10);
    });

    runner.add_test("j2_magnitude", []() {
        // J2 perturbation magnitude at LEO
        double r = 7000000.0;
        Vector3d pos(r, 0.0, 0.0);
        Vector3d acc_j2 = j2_acceleration(pos);

        // J2 perturbation is small compared to point-mass gravity
        double g = earth::GM / (r * r);
        double j2_frac = acc_j2.norm() / g;
        // Should be < 1% of total gravity
        TEST_ASSERT(j2_frac < 0.01);
        TEST_ASSERT(j2_frac > 0.0001);  // But not negligible
    });

    runner.add_test("j2_decreases_with_altitude", []() {
        // J2 effect decreases faster than point-mass (Re/r)^2 factor
        Vector3d pos1(7000000.0, 0.0, 0.0);
        Vector3d pos2(14000000.0, 0.0, 0.0);

        Vector3d acc1 = j2_acceleration(pos1);
        Vector3d acc2 = j2_acceleration(pos2);

        // The J2 potential goes as Re^2/r^3, so its acceleration goes as 1/r^4:
        // at 2x distance acc2/acc1 ≈ (1/2)^4 = 1/16 (bound 0.1).
        double ratio = acc2.norm() / acc1.norm();
        TEST_ASSERT(ratio < 0.1);
    });

    runner.add_test("symmetry_about_equator", []() {
        // J2 is symmetric about equator
        Vector3d pos1(6000000.0, 0.0, 3000000.0);
        Vector3d pos2(6000000.0, 0.0, -3000000.0);

        Vector3d acc1 = j2_acceleration(pos1);
        Vector3d acc2 = j2_acceleration(pos2);

        // X and Y components should be identical
        TEST_ASSERT_NEAR(acc1.x(), acc2.x(), 1e-15);
        TEST_ASSERT_NEAR(acc1.y(), acc2.y(), 1e-15);

        // Z components should be opposite
        TEST_ASSERT_NEAR(acc1.z(), -acc2.z(), 1e-15);
    });

    runner.add_test("azimuthal_symmetry", []() {
        // J2 is symmetric about Z-axis
        Vector3d pos1(6000000.0, 3000000.0, 2000000.0);
        Vector3d pos2(3000000.0, 6000000.0, 2000000.0);  // Same r_xy, different angle

        double r_xy1 = std::sqrt(pos1.x() * pos1.x() + pos1.y() * pos1.y());
        double r_xy2 = std::sqrt(pos2.x() * pos2.x() + pos2.y() * pos2.y());
        TEST_ASSERT_NEAR(r_xy1, r_xy2, 1e-10);

        Vector3d acc1 = j2_acceleration(pos1);
        Vector3d acc2 = j2_acceleration(pos2);

        // Magnitude should be the same
        TEST_ASSERT_NEAR(acc1.norm(), acc2.norm(), 1e-15);
    });

    runner.run();
}

void test_gravity_with_j2() {
    TestRunner runner("GravityWithJ2");

    runner.add_test("combines_correctly", []() {
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d pm = point_mass_acceleration(pos);
        Vector3d j2 = j2_acceleration(pos);
        Vector3d combined = gravity_with_j2(pos);

        TEST_ASSERT_VEC3_NEAR(combined, pm + j2, 1e-15);
    });

    runner.add_test("equatorial_weaker", []() {
        // At equator, J2 modifies gravity
        Vector3d pos(7000000.0, 0.0, 0.0);

        Vector3d pm = point_mass_acceleration(pos);
        Vector3d combined = gravity_with_j2(pos);

        // Combined should be different from point-mass (J2 has effect)
        double diff = std::abs(combined.norm() - pm.norm());
        TEST_ASSERT(diff > 1e-6);  // J2 makes a measurable difference
        TEST_ASSERT(diff < pm.norm() * 0.01);  // But < 1% difference
    });

    runner.add_test("polar_stronger", []() {
        // At pole, J2 modifies gravity
        Vector3d pos(0.0, 0.0, 7000000.0);

        Vector3d pm = point_mass_acceleration(pos);
        Vector3d combined = gravity_with_j2(pos);

        // Combined should be different from point-mass (J2 has effect)
        double diff = std::abs(combined.norm() - pm.norm());
        TEST_ASSERT(diff > 1e-6);  // J2 makes a measurable difference
        TEST_ASSERT(diff < pm.norm() * 0.01);  // But < 1% difference
    });

    runner.add_test("custom_parameters", []() {
        Vector3d pos(7000000.0, 0.0, 0.0);

        // Use different parameters
        double custom_mu = 4.0e14;
        double custom_re = 6400000.0;
        double custom_j2 = 0.001;

        Vector3d acc = gravity_with_j2(pos, custom_mu, custom_re, custom_j2);

        // Should be different from default
        Vector3d acc_default = gravity_with_j2(pos);
        double diff = (acc - acc_default).norm();
        TEST_ASSERT(diff > 1e-6);
    });

    runner.run();
}

void test_jn_acceleration() {
    TestRunner runner("JnAcceleration");

    runner.add_test("j2_via_jn", []() {
        // J2 through jn_acceleration should match j2_acceleration
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d j2_direct = j2_acceleration(pos);
        Vector3d j2_via_jn = jn_acceleration(pos, 2, earth::J2);

        TEST_ASSERT_VEC3_NEAR(j2_via_jn, j2_direct, 1e-15);
    });

    runner.add_test("j3_magnitude", []() {
        // J3 should be much smaller than J2
        Vector3d pos(7000000.0, 0.0, 3000000.0);

        Vector3d j2_acc = jn_acceleration(pos, 2, earth::J2);
        Vector3d j3_acc = jn_acceleration(pos, 3, earth::J3);

        // J3/J2 ≈ 2.5e-6 / 1.08e-3 ≈ 0.002
        TEST_ASSERT(j3_acc.norm() < j2_acc.norm() * 0.01);
    });

    runner.add_test("j4_magnitude", []() {
        Vector3d pos(7000000.0, 0.0, 3000000.0);

        Vector3d j2_acc = jn_acceleration(pos, 2, earth::J2);
        Vector3d j4_acc = jn_acceleration(pos, 4, earth::J4);

        // J4/J2 ≈ 1.6e-6 / 1.08e-3 ≈ 0.0015
        TEST_ASSERT(j4_acc.norm() < j2_acc.norm() * 0.01);
    });

    runner.add_test("j5_magnitude", []() {
        Vector3d pos(7000000.0, 0.0, 3000000.0);

        Vector3d j2_acc = jn_acceleration(pos, 2, earth::J2);
        Vector3d j5_acc = jn_acceleration(pos, 5, earth::J5);

        // J5 should be very small
        TEST_ASSERT(j5_acc.norm() < j2_acc.norm() * 0.001);
    });

    runner.add_test("j6_magnitude", []() {
        Vector3d pos(7000000.0, 0.0, 3000000.0);

        Vector3d j2_acc = jn_acceleration(pos, 2, earth::J2);
        Vector3d j6_acc = jn_acceleration(pos, 6, earth::J6);

        // J6 should be very small
        TEST_ASSERT(j6_acc.norm() < j2_acc.norm() * 0.001);
    });

    runner.add_test("unsupported_degree", []() {
        // Degrees 2-15 are supported; any other degree raises ConfigError
        // (as zonal_harmonics_acceleration does), never a silent zero.
        Vector3d pos(7000000.0, 0.0, 3000000.0);

        // J7-J15 give a non-zero acceleration
        Vector3d j7_acc = jn_acceleration(pos, 7, earth::J7);
        TEST_ASSERT(j7_acc.norm() > 0.0);  // J7 should be non-zero

        TEST_ASSERT_THROWS((void)jn_acceleration(pos, 16, 1e-6), ConfigError);
        TEST_ASSERT_THROWS((void)jn_acceleration(pos, 1, 1e-6), ConfigError);
    });

    runner.add_test("every_degree_matches_numerical_gradient_of_potential", []() {
        // a_n = grad U_n with U_n = -(mu/r) Jn (Re/r)^n Pn(z/r). The gradient is
        // taken by fourth-order central differences of U_n (Legendre recurrence,
        // independent of the closed forms); 1e-9 relative per component scale.
        auto U = [](const Vector3d& p, int n, double Jn) {
            const double r = p.norm();
            const double s = p.z() / r;
            double P0 = 1.0, P1 = s, Pn = (n == 1) ? s : 1.0;
            for (int k = 2; k <= n; ++k) {
                Pn = ((2.0 * k - 1.0) * s * P1 - (k - 1.0) * P0) / k;
                P0 = P1;
                P1 = Pn;
            }
            return -(earth::GM / r) * Jn * std::pow(earth::RADIUS_EQUATORIAL / r, n) * Pn;
        };
        const std::vector<Vector3d> points = {
            Vector3d(7.0e6, 2.0e6, 3.0e6), Vector3d(5.0e6, -4.0e6, -2.5e6),
            Vector3d(1.0e6, 5.0e5, 6.9e6), Vector3d(-3.0e6, 1.0e6, 6.0e6)};
        double worst = 0.0;
        for (int n = 2; n <= 15; ++n) {
            const double Jn = 1e-6;
            for (const auto& p : points) {
                const Vector3d a = jn_acceleration(p, n, Jn);
                const double h = 50.0;
                Vector3d g;
                for (int j = 0; j < 3; ++j) {
                    Vector3d d = Vector3d::Zero();
                    d(j) = h;
                    g(j) = (8.0 * (U(p + d, n, Jn) - U(p - d, n, Jn))
                            - (U(p + 2.0 * d, n, Jn) - U(p - 2.0 * d, n, Jn))) / (12.0 * h);
                }
                const double rel = (a - g).cwiseAbs().maxCoeff() / g.cwiseAbs().maxCoeff();
                if (rel > worst) worst = rel;
                if (rel > 1e-9) {
                    std::printf("    n = %d: relative difference %.2e\n", n, rel);
                }
            }
        }
        std::printf("    worst relative difference over n = 2..15: %.2e\n", worst);
        TEST_ASSERT(worst < 1e-9);
    });

    runner.add_test("odd_even_symmetry", []() {
        // Odd harmonics (J3, J5) break N/S symmetry
        // Even harmonics (J2, J4, J6) are symmetric

        Vector3d pos_n(5000000.0, 0.0, 4000000.0);   // Northern
        Vector3d pos_s(5000000.0, 0.0, -4000000.0);  // Southern

        Vector3d j2_n = jn_acceleration(pos_n, 2, earth::J2);
        Vector3d j2_s = jn_acceleration(pos_s, 2, earth::J2);
        Vector3d j3_n = jn_acceleration(pos_n, 3, earth::J3);
        Vector3d j3_s = jn_acceleration(pos_s, 3, earth::J3);

        // J2: X,Y same, Z opposite
        TEST_ASSERT_NEAR(j2_n.x(), j2_s.x(), 1e-15);
        TEST_ASSERT_NEAR(j2_n.z(), -j2_s.z(), 1e-15);

        // J3: different (asymmetric)
        TEST_ASSERT(std::abs(j3_n.x() - j3_s.x()) > 1e-15 ||
                    std::abs(j3_n.z() + j3_s.z()) > 1e-15);
    });

    runner.run();
}

void test_zonal_harmonics_acceleration() {
    TestRunner runner("ZonalHarmonicsAcceleration");

    runner.add_test("j2_only", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d zonal_2 = zonal_harmonics_acceleration(pos, earth::GM, earth::RADIUS_EQUATORIAL, 2);
        Vector3d j2_direct = j2_acceleration(pos);

        TEST_ASSERT_VEC3_NEAR(zonal_2, j2_direct, 1e-15);
    });

    runner.add_test("j2_j3", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d zonal_3 = zonal_harmonics_acceleration(pos, earth::GM, earth::RADIUS_EQUATORIAL, 3);
        Vector3d expected = j2_acceleration(pos) + jn_acceleration(pos, 3, earth::J3);

        TEST_ASSERT_VEC3_NEAR(zonal_3, expected, 1e-15);
    });

    runner.add_test("j2_to_j6", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d zonal_6 = zonal_harmonics_acceleration(pos, earth::GM, earth::RADIUS_EQUATORIAL, 6);
        Vector3d expected = j2_acceleration(pos)
                          + jn_acceleration(pos, 3, earth::J3)
                          + jn_acceleration(pos, 4, earth::J4)
                          + jn_acceleration(pos, 5, earth::J5)
                          + jn_acceleration(pos, 6, earth::J6);

        TEST_ASSERT_VEC3_NEAR(zonal_6, expected, 1e-15);
    });

    runner.add_test("higher_degree_small_effect", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d zonal_2 = zonal_harmonics_acceleration(pos, earth::GM, earth::RADIUS_EQUATORIAL, 2);
        Vector3d zonal_6 = zonal_harmonics_acceleration(pos, earth::GM, earth::RADIUS_EQUATORIAL, 6);

        // J3-J6 contribution should be small compared to J2
        double diff = (zonal_6 - zonal_2).norm();
        double j2_mag = zonal_2.norm();
        TEST_ASSERT(diff < j2_mag * 0.01);  // Less than 1% of J2
    });

    runner.add_test("max_degree_1_is_zero", []() {
        // Degree 1 is usually zero (center of mass)
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);
        Vector3d zonal_1 = zonal_harmonics_acceleration(pos, earth::GM, earth::RADIUS_EQUATORIAL, 1);

        TEST_ASSERT_VEC3_NEAR(zonal_1, Vector3d::Zero(), 1e-20);
    });

    runner.run();
}

void test_gravity_with_zonal() {
    TestRunner runner("GravityWithZonal");

    runner.add_test("combines_point_mass_and_zonal", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d pm = point_mass_acceleration(pos);
        Vector3d zonal = zonal_harmonics_acceleration(pos, earth::GM, earth::RADIUS_EQUATORIAL, 6);
        Vector3d full = gravity_with_zonal(pos, earth::GM, earth::RADIUS_EQUATORIAL, 6);

        TEST_ASSERT_VEC3_NEAR(full, pm + zonal, 1e-15);
    });

    runner.add_test("default_degree_6", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d full_default = gravity_with_zonal(pos);
        Vector3d full_explicit = gravity_with_zonal(pos, earth::GM, earth::RADIUS_EQUATORIAL, 6);

        TEST_ASSERT_VEC3_NEAR(full_default, full_explicit, 1e-15);
    });

    runner.add_test("dominance_of_point_mass", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        Vector3d pm = point_mass_acceleration(pos);
        Vector3d full = gravity_with_zonal(pos);

        // Point-mass dominates - full should be within 0.2% of point-mass magnitude
        double pm_mag = pm.norm();
        double full_mag = full.norm();
        TEST_ASSERT_NEAR(full_mag, pm_mag, pm_mag * 0.002);
    });

    runner.run();
}

void test_gravitational_potential() {
    TestRunner runner("GravitationalPotential");

    runner.add_test("surface_potential", []() {
        Vector3d pos(earth::RADIUS_EQUATORIAL, 0.0, 0.0);
        double U = gravitational_potential(pos);

        // U = -GM/r ≈ -6.26e7 J/kg
        double expected = -earth::GM / earth::RADIUS_EQUATORIAL;
        TEST_ASSERT_NEAR(U, expected, 1e-6);
    });

    runner.add_test("potential_negative", []() {
        // Potential is always negative (bound system)
        std::vector<Vector3d> positions = {
            Vector3d(7000000.0, 0.0, 0.0),
            Vector3d(42164000.0, 0.0, 0.0),
            Vector3d(384400000.0, 0.0, 0.0),
        };

        for (const auto& pos : positions) {
            double U = gravitational_potential(pos);
            TEST_ASSERT(U < 0);
        }
    });

    runner.add_test("potential_increases_with_distance", []() {
        // Potential becomes less negative (increases) with distance
        Vector3d pos1(7000000.0, 0.0, 0.0);
        Vector3d pos2(14000000.0, 0.0, 0.0);

        double U1 = gravitational_potential(pos1);
        double U2 = gravitational_potential(pos2);

        // U2 > U1 (less negative)
        TEST_ASSERT(U2 > U1);
    });

    runner.add_test("spherical_symmetry", []() {
        // Potential depends only on distance
        double r = 7000000.0;
        std::vector<Vector3d> positions = {
            Vector3d(r, 0.0, 0.0),
            Vector3d(0.0, r, 0.0),
            Vector3d(0.0, 0.0, r),
            Vector3d(r / std::sqrt(3.0), r / std::sqrt(3.0), r / std::sqrt(3.0)),
        };

        double U_ref = gravitational_potential(positions[0]);
        for (const auto& pos : positions) {
            double U = gravitational_potential(pos);
            TEST_ASSERT_NEAR(U, U_ref, 1e-10);
        }
    });

    runner.add_test("custom_mu", []() {
        Vector3d pos(1738000.0, 0.0, 0.0);  // Moon radius
        double U = gravitational_potential(pos, moon::GM);

        double expected = -moon::GM / pos.norm();
        TEST_ASSERT_NEAR(U, expected, 1e-6);
    });

    runner.run();
}

void test_j2_potential() {
    TestRunner runner("J2Potential");

    runner.add_test("equatorial_potential", []() {
        // At equator, sin(lat) = 0, so P2 = -0.5
        Vector3d pos(7000000.0, 0.0, 0.0);
        double U_j2 = j2_potential(pos);

        // U_j2 = -J2 * GM * Re² / r³ * P2
        // P2(0) = -0.5
        double r = pos.norm();
        double expected = -earth::J2 * earth::GM * earth::RADIUS_EQUATORIAL * earth::RADIUS_EQUATORIAL
                         / (r * r * r) * (-0.5);
        TEST_ASSERT_NEAR(U_j2, expected, 1e-6);
    });

    runner.add_test("polar_potential", []() {
        // At pole, sin(lat) = 1, so P2 = 1
        Vector3d pos(0.0, 0.0, 7000000.0);
        double U_j2 = j2_potential(pos);

        // P2(1) = (3*1 - 1)/2 = 1
        double r = pos.norm();
        double expected = -earth::J2 * earth::GM * earth::RADIUS_EQUATORIAL * earth::RADIUS_EQUATORIAL
                         / (r * r * r) * 1.0;
        TEST_ASSERT_NEAR(U_j2, expected, 1e-6);
    });

    runner.add_test("j2_much_smaller_than_central", []() {
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        double U_central = gravitational_potential(pos);
        double U_j2 = j2_potential(pos);

        // J2 contribution should be ~0.1% of central
        TEST_ASSERT(std::abs(U_j2) < std::abs(U_central) * 0.002);
    });

    runner.run();
}

void test_total_potential_with_j2() {
    TestRunner runner("TotalPotentialWithJ2");

    runner.add_test("combines_correctly", []() {
        // Positive convention: U = mu/r + (J2 term) = -gravitational_potential() + j2_potential()
        Vector3d pos(7000000.0, 2000000.0, 3000000.0);

        double U_central = gravitational_potential(pos);
        double U_j2 = j2_potential(pos);
        double U_total = total_potential_with_j2(pos);

        TEST_ASSERT_NEAR(U_total, -U_central + U_j2, 1e-6);
    });

    runner.add_test("always_positive", []() {
        std::vector<Vector3d> positions = {
            Vector3d(7000000.0, 0.0, 0.0),
            Vector3d(0.0, 7000000.0, 0.0),
            Vector3d(0.0, 0.0, 7000000.0),
            Vector3d(5000000.0, 5000000.0, 3000000.0),
        };

        for (const auto& pos : positions) {
            double U = total_potential_with_j2(pos);
            TEST_ASSERT(U > 0);
        }
    });

    runner.add_test("gradient_is_point_mass_plus_j2_acceleration", []() {
        // a = +grad(U): the central-difference gradient of the potential must equal the
        // analytic point-mass + J2 acceleration (with the two terms in opposite sign
        // conventions the gradient was neither +a nor -a).
        const std::vector<Vector3d> positions = {
            Vector3d(7000000.0, 2000000.0, 3000000.0),
            Vector3d(0.0, 0.0, 7000000.0),
            Vector3d(-4000000.0, 5000000.0, -2500000.0),
        };
        const double h = 10.0;  // m (round-off ~1e-9 m/s^2, truncation ~1e-11 m/s^2)
        for (const auto& pos : positions) {
            Vector3d grad;
            for (int i = 0; i < 3; ++i) {
                Vector3d dp = Vector3d::Zero();
                dp[i] = h;
                grad[i] = (total_potential_with_j2(pos + dp) - total_potential_with_j2(pos - dp)) /
                          (2.0 * h);
            }
            const Vector3d a = point_mass_acceleration(pos) + j2_acceleration(pos);
            // The J2 part is ~1e-2 m/s^2: a sign error in it is ~1e6 times this tolerance.
            TEST_ASSERT((grad - a).norm() < 1e-6 * j2_acceleration(pos).norm());
        }
    });

    runner.run();
}

void test_specific_orbital_energy() {
    TestRunner runner("SpecificOrbitalEnergy");

    runner.add_test("circular_leo", []() {
        // Circular LEO at 400 km
        double r = earth::RADIUS_EQUATORIAL + 400000.0;
        double v = std::sqrt(earth::GM / r);  // Circular velocity

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        double E = specific_orbital_energy(pos, vel);

        // For circular orbit, E = -GM/(2r)
        double expected = -earth::GM / (2 * r);
        TEST_ASSERT_NEAR(E, expected, 1e-6);
    });

    runner.add_test("circular_geo", []() {
        // GEO circular orbit
        double r = 42164000.0;
        double v = std::sqrt(earth::GM / r);

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        double E = specific_orbital_energy(pos, vel);
        double expected = -earth::GM / (2 * r);
        TEST_ASSERT_NEAR(E, expected, 1e-6);
    });

    runner.add_test("escape_velocity", []() {
        // At escape velocity, E = 0
        double r = 7000000.0;
        double v_esc = std::sqrt(2 * earth::GM / r);

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v_esc, 0.0);

        double E = specific_orbital_energy(pos, vel);
        TEST_ASSERT_NEAR(E, 0.0, 1e-6);
    });

    runner.add_test("hyperbolic", []() {
        // Above escape velocity, E > 0
        double r = 7000000.0;
        double v = 12000.0;  // Above escape velocity

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        double E = specific_orbital_energy(pos, vel);
        TEST_ASSERT(E > 0);
    });

    runner.add_test("bound_orbit", []() {
        // Below escape velocity, E < 0
        double r = 7000000.0;
        double v = 7000.0;  // Below escape velocity

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        double E = specific_orbital_energy(pos, vel);
        TEST_ASSERT(E < 0);
    });

    runner.add_test("direction_independent", []() {
        // Energy depends only on magnitudes, not directions
        double r = 7000000.0;
        double v = 7500.0;

        std::vector<std::pair<Vector3d, Vector3d>> states = {
            {Vector3d(r, 0, 0), Vector3d(0, v, 0)},
            {Vector3d(0, r, 0), Vector3d(v, 0, 0)},
            {Vector3d(r/std::sqrt(2), r/std::sqrt(2), 0), Vector3d(-v/std::sqrt(2), v/std::sqrt(2), 0)},
        };

        double E_ref = specific_orbital_energy(states[0].first, states[0].second);
        for (const auto& state : states) {
            double E = specific_orbital_energy(state.first, state.second);
            TEST_ASSERT_NEAR(E, E_ref, 1e-6);
        }
    });

    runner.run();
}

void test_specific_angular_momentum() {
    TestRunner runner("SpecificAngularMomentum");

    runner.add_test("circular_prograde", []() {
        // Circular prograde orbit in XY plane
        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, v, 0.0);

        Vector3d h = specific_angular_momentum(pos, vel);

        // h should point in +Z direction
        TEST_ASSERT_NEAR(h.x(), 0.0, 1e-6);
        TEST_ASSERT_NEAR(h.y(), 0.0, 1e-6);
        TEST_ASSERT(h.z() > 0);

        // |h| = r * v
        TEST_ASSERT_NEAR(h.norm(), r * v, 1e-6);
    });

    runner.add_test("circular_retrograde", []() {
        // Circular retrograde orbit
        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, -v, 0.0);  // Retrograde

        Vector3d h = specific_angular_momentum(pos, vel);

        // h should point in -Z direction
        TEST_ASSERT(h.z() < 0);
    });

    runner.add_test("polar_orbit", []() {
        // Polar orbit (velocity in Z direction)
        double r = 7000000.0;
        double v = std::sqrt(earth::GM / r);

        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(0.0, 0.0, v);

        Vector3d h = specific_angular_momentum(pos, vel);

        // h should be in XY plane (perpendicular to both pos and vel)
        TEST_ASSERT_NEAR(h.z(), 0.0, 1e-6);
        TEST_ASSERT(h.y() < 0);  // -Y direction
    });

    runner.add_test("radial_velocity", []() {
        // Pure radial velocity (degenerate case)
        Vector3d pos(7000000.0, 0.0, 0.0);
        Vector3d vel(1000.0, 0.0, 0.0);  // Radial

        Vector3d h = specific_angular_momentum(pos, vel);

        // h = r × v = 0 for parallel vectors
        TEST_ASSERT_VEC3_NEAR(h, Vector3d::Zero(), 1e-6);
    });

    runner.add_test("magnitude_formula", []() {
        // |h| = r * v * sin(angle)
        double r = 7000000.0;
        double v = 7500.0;

        // 45 degree angle between pos and vel
        Vector3d pos(r, 0.0, 0.0);
        Vector3d vel(v * 0.5, v * std::sqrt(0.75), 0.0);

        Vector3d h = specific_angular_momentum(pos, vel);

        // sin(angle) = v_perp / v
        double v_perp = v * std::sqrt(0.75);  // Perpendicular component
        double expected_h = r * v_perp;
        TEST_ASSERT_NEAR(h.norm(), expected_h, 1e-6);
    });

    runner.run();
}

void test_edge_cases() {
    TestRunner runner("EdgeCases");

    runner.add_test("very_close_position", []() {
        // Very close to Earth center - should still work (though unphysical)
        Vector3d pos(100000.0, 0.0, 0.0);  // 100 km from center

        Vector3d pm = point_mass_acceleration(pos);
        Vector3d j2 = j2_acceleration(pos);

        // Both should be finite and large
        TEST_ASSERT(std::isfinite(pm.norm()));
        TEST_ASSERT(std::isfinite(j2.norm()));
        TEST_ASSERT(pm.norm() > 100.0);  // Very strong gravity
    });

    runner.add_test("very_far_position", []() {
        // Very far from Earth - Sun distance
        Vector3d pos(1.5e11, 0.0, 0.0);  // 1 AU

        Vector3d pm = point_mass_acceleration(pos);
        Vector3d j2 = j2_acceleration(pos);

        // Point mass should be tiny (GM/r^2 ~ 1.77e-8), J2 should be negligible
        TEST_ASSERT(pm.norm() < 1e-7);  // Small but not zero
        TEST_ASSERT(pm.norm() > 1e-10);  // Not negligible
        TEST_ASSERT(j2.norm() < pm.norm() * 0.001);  // J2 much smaller than point mass
    });

    runner.add_test("zero_velocity_energy", []() {
        // Zero velocity - maximum potential energy
        Vector3d pos(7000000.0, 0.0, 0.0);
        Vector3d vel(0.0, 0.0, 0.0);

        double E = specific_orbital_energy(pos, vel);

        // E = 0 - GM/r = -GM/r
        double expected = -earth::GM / pos.norm();
        TEST_ASSERT_NEAR(E, expected, 1e-6);
    });

    runner.add_test("high_velocity", []() {
        // Very high velocity - hyperbolic
        Vector3d pos(7000000.0, 0.0, 0.0);
        Vector3d vel(0.0, 50000.0, 0.0);  // 50 km/s

        double E = specific_orbital_energy(pos, vel);

        // Should be positive (unbound)
        TEST_ASSERT(E > 0);
    });

    runner.add_test("different_gravity_sources", []() {
        // Test with Moon's parameters
        Vector3d pos(1738000.0, 0.0, 0.0);  // Moon surface

        // Moon surface gravity
        Vector3d acc = point_mass_acceleration(pos, moon::GM);
        TEST_ASSERT_NEAR(acc.norm(), 1.62, 0.1);

        // Sun at 1 AU
        Vector3d pos_sun(1.496e11, 0.0, 0.0);
        Vector3d acc_sun = point_mass_acceleration(pos_sun, sun::GM);
        TEST_ASSERT_NEAR(acc_sun.norm(), 0.006, 0.001);  // ~6 mm/s²
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Point Mass and Zonal Harmonics Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_point_mass_acceleration();
    test_j2_acceleration();
    test_gravity_with_j2();
    test_jn_acceleration();
    test_zonal_harmonics_acceleration();
    test_gravity_with_zonal();
    test_gravitational_potential();
    test_j2_potential();
    test_total_potential_with_j2();
    test_specific_orbital_energy();
    test_specific_angular_momentum();
    test_edge_cases();

    return print_final_summary();
}
