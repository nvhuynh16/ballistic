// =============================================================================
// Test: gravity Hessian (second derivatives of the gravity potential)
// =============================================================================
//
// The Hessian H = ∂²V/∂x_i∂x_j of the Earth gravity potential, computed by the
// Clenshaw spherical-harmonic engine. Two implementations exist:
//   - "Strategy 1", compute_with_hessian_analytic: an extended Clenshaw
//     recurrence that differentiates analytically. compute_with_hessian (the
//     production entry point) calls it.
//   - "Strategy 2", compute_with_hessian_fd: finite differences (FD) of the
//     analytic gradient, kept for cross-checking.
// Checked against:
//   - the closed-form J2 Hessian at degree 2 (J2 = the Earth-flattening term,
//     j2_hessian_closed_form.hpp);
//   - properties every gravity Hessian has: Laplace's equation tr(H) = 0 outside
//     the Earth, and symmetry H = Hᵀ;
//   - fourth-order central finite differences of acceleration(), for every
//     single (n, m, C or S) term to degree 6 and for the full EGM2008 field at
//     degrees 2, 20 and 70, in both storage layouts (5e-10 relative);
//   - agreement of the two implementations to FD precision;
//   - several altitudes and latitudes.
// =============================================================================

#include "../test_utils.hpp"
#include "j2_hessian_closed_form.hpp"
#include "../../include/ballistic/gravity/clenshaw_engine.hpp"
#include "../../include/ballistic/gravity/egm_gravity.hpp"
#include "../../include/ballistic/core/constants.hpp"
#include <Eigen/Dense>
#include <cstdio>

using namespace ballistic;
using namespace ballistic::gravity;
using namespace ballistic::test;

// Helper: configure an engine with J2 zonal coefficient only (deg=2 m=0).
// J2 normalized coefficient C_20_norm = -J2 / sqrt(5) per geodesy convention.
ClenshawTriangular make_j2_engine(double J2 = 1.0826267e-3) {
    ClenshawTriangular engine;
    engine.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 2);
    // Set C_20 normalized
    const double C20_norm = -J2 / std::sqrt(5.0);
    engine.set_coefficient(2, 0, C20_norm, 0.0);
    return engine;
}

// Hessian by fourth-order central differences of the engine's acceleration:
// truncation O(delta^4), so with delta = 100 m the reference is limited by
// round-off (~5e-11 of |H| at LEO), not by the stencil.
template <typename Engine>
Eigen::Matrix3d fd4_hessian(const Engine& engine, const Vector3d& pos, int degree,
                            double delta = 100.0) {
    Eigen::Matrix3d H;
    for (int j = 0; j < 3; ++j) {
        Vector3d dx = Vector3d::Zero();
        dx(j) = delta;
        const Vector3d a1 = engine.acceleration(pos + dx, degree);
        const Vector3d am1 = engine.acceleration(pos - dx, degree);
        const Vector3d a2 = engine.acceleration(pos + 2.0 * dx, degree);
        const Vector3d am2 = engine.acceleration(pos - 2.0 * dx, degree);
        H.col(j) = (8.0 * (a1 - am1) - (a2 - am2)) / (12.0 * delta);
    }
    return H;
}

// max |H_a - H_ref| / max |H_ref| (element-wise)
double rel_max_diff(const Eigen::Matrix3d& H, const Eigen::Matrix3d& ref) {
    return (H - ref).cwiseAbs().maxCoeff() / ref.cwiseAbs().maxCoeff();
}

// Points off the equator (where the r-latitude cross term of the Hessian is
// non-zero), including two within 2 degrees of the poles.
const std::vector<Vector3d> kFdPoints = {
    Vector3d(6.778e6, 0.0, 0.0),
    Vector3d(5.6e6, 0.0, 3.85e6),
    Vector3d(5e6, 3e6, 4e6),
    Vector3d(1e6, 5e5, 6.7e6),
    Vector3d(-3e6, -4e6, -4.5e6),
    Vector3d(2e5, 1e5, 6.9e6),
    Vector3d(-1e5, 2e5, -6.95e6),
    Vector3d(2.0e7, 1.0e7, -1.2e7),
};

int main() {
    TestRunner runner("Gravity Hessian (Stage F Strategy 2)");

    const std::vector<std::pair<std::string, Vector3d>> test_points = {
        {"LEO equatorial", Vector3d(6.778e6, 0.0, 0.0)},
        {"LEO mid-lat",    Vector3d(5.6e6, 0.0, 3.85e6)},
        {"LEO 45-deg lat", Vector3d(5.0e6 * std::sqrt(0.5), 0.0, 5.0e6 * std::sqrt(0.5))},
        {"MEO equatorial", Vector3d(2.65e7, 0.0, 0.0)},
        {"GEO equatorial", Vector3d(4.2164e7, 0.0, 0.0)},
        {"Off-axis",       Vector3d(5e6, 3e6, 4e6)},
    };

    runner.add_test("J2 only at deg=2: matches closed form to FD precision", [&]() {
        // Compare engine deg=2 zonal Hessian to closed-form J2 Hessian.
        // The engine includes the point-mass term too (l=0); the closed
        // form is J2-only. So the test point compares H_engine_deg2 −
        // H_engine_deg0 (= J2-only contribution) against H_J2_closed.
        auto engine_full = make_j2_engine();
        ClenshawTriangular engine_pointmass;
        engine_pointmass.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 0);

        for (const auto& [label, pos] : test_points) {
            // Engine with J2: full deg=2 result
            auto h_full = engine_full.compute_with_hessian(pos, 2);
            // Engine point-mass-only: deg=0 result
            auto h_pm = engine_pointmass.compute_with_hessian(pos, 0);
            // J2-only Hessian = full - point-mass
            const Eigen::Matrix3d H_engine_J2 = h_full.hessian_ecef - h_pm.hessian_ecef;
            // Closed-form J2 reference
            auto j2_ref = j2_closed_form(pos);
            const Eigen::Matrix3d& H_ref = j2_ref.hessian;

            const double scale = H_ref.norm() + 1e-30;
            const double rel = (H_engine_J2 - H_ref).norm() / scale;
            // Production now uses Strategy 1 (analytic recurrence). Tolerance
            // 1e-9 accommodates condition-number amplification: the
            // (H_full - H_pointmass) subtraction loses ~3 digits because
            // |H_pointmass| / |H_J2| ≈ 1000.
            TEST_ASSERT(rel < 1e-9);
        }
    });

    runner.add_test("Laplace's equation: tr(H_ecef) = 0 outside source", [&]() {
        // Gravitational Hessian is traceless outside the source mass; 1e-9 of |H|
        // leaves room for round-off.
        auto engine = make_j2_engine();
        for (const auto& [label, pos] : test_points) {
            auto result = engine.compute_with_hessian(pos, 2);
            const auto& H = result.hessian_ecef;
            const double trace = H(0, 0) + H(1, 1) + H(2, 2);
            const double scale = H.norm() + 1e-30;
            const double rel = std::abs(trace) / scale;
            TEST_ASSERT(rel < 1e-9);
        }
    });

    runner.add_test("Hessian symmetry: H = Hᵀ to machine precision", [&]() {
        // Schwarz: enforced by 0.5·(H + Hᵀ) symmetrization in
        // compute_with_hessian. Should be EXACT.
        auto engine = make_j2_engine();
        for (const auto& [label, pos] : test_points) {
            auto result = engine.compute_with_hessian(pos, 2);
            const auto& H = result.hessian_ecef;
            for (int i = 0; i < 3; ++i) {
                for (int j = i + 1; j < 3; ++j) {
                    TEST_ASSERT_NEAR(H(i, j), H(j, i), 1e-25);
                }
            }
        }
    });

    runner.add_test("Acceleration sanity: matches engine's gradient method", [&]() {
        // compute_with_hessian's acceleration field must equal compute()'s
        // (i.e., the central-evaluation gradient — no FD applied to it).
        auto engine = make_j2_engine();
        for (const auto& [label, pos] : test_points) {
            auto h = engine.compute_with_hessian(pos, 2);
            auto a = engine.acceleration(pos, 2);
            for (int i = 0; i < 3; ++i) {
                TEST_ASSERT_NEAR(h.acceleration(i), a(i), 1e-15);
            }
        }
    });

    runner.add_test("FD floor: tr(H_full)/|H_full| at LEO is < 1e-9", [&]() {
        // Even with the point mass (l=0) included, Laplace must hold —
        // it's a consequence of ∇²(GM/r) = 0 plus harmonic spherical
        // harmonics. Our deg=2 J2-augmented gravity remains harmonic.
        auto engine = make_j2_engine();
        Vector3d pos(6.778e6, 0.0, 0.0);
        auto result = engine.compute_with_hessian(pos, 2);
        const auto& H = result.hessian_ecef;
        const double trace = H(0, 0) + H(1, 1) + H(2, 2);
        const double scale = H.norm();
        const double rel = std::abs(trace) / scale;
        TEST_ASSERT(rel < 1e-9);
    });

    runner.add_test("Hessian magnitude scale: J2 contributes ~10^-9 1/s² at LEO", [&]() {
        // Sanity check that we're computing actual gravity Hessian
        // (not, e.g., zeros).
        auto engine = make_j2_engine();
        Vector3d pos(6.778e6, 0.0, 0.0);
        auto result = engine.compute_with_hessian(pos, 2);
        const double H_norm = result.hessian_ecef.norm();
        // Point mass: |H| ~ GM/r³ ~ 4e14/3e20 ~ 1.3e-6 1/s²
        // J2 modifies this by ~1e-3 → still ~1e-6 magnitude
        TEST_ASSERT(H_norm > 1e-7 && H_norm < 1e-5);
    });

    // ========================================================================
    // Strategy 1 (compute_with_hessian_analytic) tests
    // ========================================================================

    runner.add_test("Strategy 1: J2 only at deg=2 matches closed form to machine precision",
        [&]() {
            // Strategy 1 should be analytically EXACT at deg=2 J2-only —
            // matches closed-form Hessian to machine epsilon (modulo
            // condition-number amplification of round-off).
            auto engine_full = make_j2_engine();
            ClenshawTriangular engine_pointmass;
            engine_pointmass.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 0);

            for (const auto& [label, pos] : test_points) {
                auto h_full = engine_full.compute_with_hessian_analytic(pos, 2);
                auto h_pm = engine_pointmass.compute_with_hessian_analytic(pos, 0);
                const Eigen::Matrix3d H_engine_J2 = h_full.hessian_ecef - h_pm.hessian_ecef;
                auto j2_ref = j2_closed_form(pos);
                const Eigen::Matrix3d& H_ref = j2_ref.hessian;
                const double scale = H_ref.norm() + 1e-30;
                const double rel = (H_engine_J2 - H_ref).norm() / scale;
                // Strategy 1: analytic → ~1e-12 expected. Loose 1e-9 to
                // accommodate condition-number amplification.
                TEST_ASSERT(rel < 1e-9);
            }
        });

    runner.add_test("Strategy 1 vs Strategy 2: agree at the FD floor (full field, degree 70)", [&]() {
        // Strategy 2 differences the acceleration with a 1 m step (second-order
        // central), so the two agree to its round-off. If they differ by more,
        // one of them has a math bug.
        auto& egm = EGM2008Gravity::instance();
        egm.init(70);
        const auto& engine = egm.colmajor_model();
        double worst = 0.0;
        for (const auto& pos : kFdPoints) {
            auto h1 = engine.compute_with_hessian_analytic(pos, 70);
            auto h2 = engine.compute_with_hessian_fd(pos, 70);
            worst = std::max(worst, rel_max_diff(h1.hessian_ecef, h2.hessian_ecef));

            // Acceleration must be identical (both use the same
            // analytic gradient code path)
            for (int i = 0; i < 3; ++i) {
                TEST_ASSERT_NEAR(h1.acceleration(i), h2.acceleration(i), 1e-15);
            }
        }
        std::printf("    worst Strategy 1 vs 2 relative difference: %.2e\n", worst);
        // measured 3.7e-9 (round-off of the 1 m step); bound 10x
        TEST_ASSERT(worst < 4e-8);
    });

    runner.add_test("Strategy 1: Laplace's equation tr(H) = 0 at machine precision",
        [&]() {
            // Strategy 1 is analytic, so Laplace must hold to ~machine
            // epsilon (no FD truncation). Loose 1e-12 for condition-number.
            auto engine = make_j2_engine();
            for (const auto& [label, pos] : test_points) {
                auto result = engine.compute_with_hessian_analytic(pos, 2);
                const auto& H = result.hessian_ecef;
                const double trace = H(0, 0) + H(1, 1) + H(2, 2);
                const double scale = H.norm() + 1e-30;
                const double rel = std::abs(trace) / scale;
                TEST_ASSERT(rel < 1e-12);
            }
        });

    runner.add_test("Strategy 1: Hessian symmetric to machine precision", [&]() {
        auto engine = make_j2_engine();
        for (const auto& [label, pos] : test_points) {
            auto result = engine.compute_with_hessian_analytic(pos, 2);
            const auto& H = result.hessian_ecef;
            for (int i = 0; i < 3; ++i) {
                for (int j = i + 1; j < 3; ++j) {
                    TEST_ASSERT_NEAR(H(i, j), H(j, i), 1e-25);
                }
            }
        }
    });

    runner.add_test("Strategy 1: acceleration matches engine compute()", [&]() {
        // The 1st-deriv part of Strategy 1's recurrence is identical
        // to compute(); they must produce the same acceleration.
        auto engine = make_j2_engine();
        for (const auto& [label, pos] : test_points) {
            auto h_strat1 = engine.compute_with_hessian_analytic(pos, 2);
            auto a_native = engine.acceleration(pos, 2);
            for (int i = 0; i < 3; ++i) {
                const double rel = std::abs(h_strat1.acceleration(i) - a_native(i))
                                   / (std::abs(a_native(i)) + 1e-30);
                TEST_ASSERT(rel < 1e-12);
            }
        }
    });

    // Laplace's equation and symmetry do not involve the r-latitude cross term,
    // so an error confined to it (e.g. a sectorial correction applied twice,
    // O(1) on every order m >= 1 term, 4e-5 of the full field) passes them.
    // These cells compare each term separately and the full field with
    // finite differences of the acceleration.
    runner.add_test("analytic Hessian: each single (n, m) term vs 4th-order FD of acceleration",
        [&]() {
            double worst = 0.0;
            for (int n = 2; n <= 6; ++n) {
                for (int m = 0; m <= n; ++m) {
                    for (int sine = 0; sine <= (m > 0 ? 1 : 0); ++sine) {
                        ClenshawTriangular engine;
                        engine.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 6);
                        engine.set_coefficient(0, 0, 0.0, 0.0);  // the term alone
                        engine.set_coefficient(n, m, sine ? 0.0 : 1e-3, sine ? 1e-3 : 0.0);
                        for (const auto& pos : kFdPoints) {
                            const double rel = rel_max_diff(
                                engine.compute_with_hessian_analytic(pos).hessian_ecef,
                                fd4_hessian(engine, pos, -1));
                            worst = std::max(worst, rel);
                        }
                    }
                }
            }
            std::printf("    worst single-term relative difference: %.2e\n", worst);
            // measured 7.4e-11 (FD round-off, worst near the poles); bound 10x
            TEST_ASSERT(worst < 7.5e-10);
        });

    runner.add_test("analytic Hessian: full EGM2008 field vs 4th-order FD, degrees 2/20/70, both layouts",
        [&]() {
            auto& egm = EGM2008Gravity::instance();
            egm.init(70);
            const auto& colmajor = egm.colmajor_model();
            const auto& triangular = egm.model();
            double worst = 0.0;
            for (int degree : {2, 20, 70}) {
                for (const auto& pos : kFdPoints) {
                    const Eigen::Matrix3d ref = fd4_hessian(colmajor, pos, degree);
                    worst = std::max(worst, rel_max_diff(
                        colmajor.compute_with_hessian(pos, degree).hessian_ecef, ref));
                    worst = std::max(worst, rel_max_diff(
                        triangular.compute_with_hessian(pos, degree).hessian_ecef,
                        fd4_hessian(triangular, pos, degree)));
                }
            }
            std::printf("    worst full-field relative difference: %.2e\n", worst);
            // measured 5.4e-11 (FD round-off); bound 10x
            TEST_ASSERT(worst < 5e-10);
        });

    runner.add_test("engine of degree < 5: the m = 0 step stays inside its sqrt table", [&]() {
        // The m = 0 step reads sqrt(15); a table sized 2N + 6 was too short for
        // N <= 4 and read past its end. A degree-2 engine must equal a degree-6
        // engine holding the same coefficients.
        ClenshawTriangular small, large;
        small.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 2);
        large.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 6);
        for (auto* e : {&small, &large}) {
            e->set_coefficient(2, 0, -4.84e-4, 0.0);
            e->set_coefficient(2, 2, 2.4e-6, -1.4e-6);
        }
        for (const auto& pos : kFdPoints) {
            const auto hs = small.compute_with_hessian(pos);
            const auto hl = large.compute_with_hessian(pos);
            TEST_ASSERT((hs.acceleration - hl.acceleration).norm()
                        <= 1e-14 * hl.acceleration.norm());
            TEST_ASSERT((hs.hessian_ecef - hl.hessian_ecef).norm()
                        <= 1e-13 * hl.hessian_ecef.norm());
        }
    });

    runner.run();
    return print_final_summary();
}
