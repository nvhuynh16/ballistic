// Per-component analytic Jacobian unit tests. The Jacobian is the matrix of
// partial derivatives of a force's acceleration with respect to the state
// (∂a/∂r, and ∂a/∂v where the force depends on velocity); estimators use it
// to propagate sensitivities.
//
// Each force component that exposes acceleration_with_jacobian gets two kinds
// of test:
//   (a) agreement of the analytic Jacobian with a finite-difference (FD)
//       Jacobian (perturb the state, difference the accelerations) at several
//       representative states in LEO / MEO / GEO (low, medium, geostationary
//       Earth orbit);
//   (b) algebraic properties: symmetry, tracelessness, expected magnitudes.
// Forces covered: third body (Sun, Moon, Jupiter), SRP (solar radiation
// pressure), Earth infrared (IR) and albedo radiation, relativistic
// corrections, and drag.
//
// Tolerances chosen so that FD truncation error (O(fd_dx² · second-deriv)) is
// well below the threshold. For third-body at LEO/MEO/GEO with fd_dx = 1 m,
// truncation is ~1e-25 vs analytic magnitude ~1e-14, so relative agreement
// should be 1e-10 or better. We test at 1e-4 — defensive but catches gross
// math errors.

#include <ballistic/init.hpp>
#include <ballistic/gravity/third_body.hpp>
#include <ballistic/forces/radiation_pressure.hpp>
#include <ballistic/forces/relativistic.hpp>
#include <ballistic/forces/drag.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/time/epoch.hpp>
#include "../test_utils.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

using namespace ballistic;
using namespace ballistic::test;

namespace {

// Build the FD Jacobian ∂a/∂r at r_sat with the given perturbation. The
// callable `accel_fn(r)` returns the acceleration at position r.
template <typename AccelFn>
Eigen::Matrix3d fd_position_jacobian(AccelFn&& accel_fn, const Vector3d& r_sat,
                                      double fd_dx)
{
    const Vector3d a0 = accel_fn(r_sat);
    Eigen::Matrix3d J;
    for (int i = 0; i < 3; ++i) {
        Vector3d rp = r_sat;
        rp(i) += fd_dx;
        J.col(i) = (accel_fn(rp) - a0) / fd_dx;
    }
    return J;
}

double rel_frobenius_diff(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B) {
    const double num = (A - B).norm();
    const double den = std::max(B.norm(), 1e-30);
    return num / den;
}

// Run the third-body FD-vs-analytic test for a given body configuration.
//
// Default tolerance 1e-3 reflects the floor from catastrophic cancellation in
// the existing third_body_acceleration formula `(Δ/d³ - r_body/D³)`: for
// distant bodies (Jupiter especially), |Δ/d³| ≈ |r_body/D³| and their
// difference loses ~5 digits of precision. This affects FD by the same
// factor, so analytic and FD agree to ~1e-4 relative for Jupiter, ~1e-5 for
// Sun, ~1e-7 for Moon. Math is correct in all cases — looser tolerance just
// accommodates the formula's own precision floor. Production impact is
// negligible (Jupiter perturbation absolute error is sub-femtometer over a
// 184 s glide).
void test_third_body_at(
    const char* label,
    bool sun, bool moon, bool jupiter,
    const Vector3d& r_sat,
    double tol = 1e-3)
{
    gravity::ThirdBodyGravity tb;
    tb.enable_sun(sun);
    tb.enable_moon(moon);
    tb.enable_jupiter(jupiter);

    const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    auto [a_an, J_an] = tb.acceleration_with_jacobian(t0, r_sat);

    // Sanity: analytic acceleration matches the existing acceleration() method
    const Vector3d a_existing = tb.acceleration(t0, r_sat);
    const double a_diff = (a_an - a_existing).norm();
    TEST_ASSERT_MSG(a_diff < 1e-15 * std::max(a_existing.norm(), 1e-30),
        "analytic acceleration disagrees with tb.acceleration()");

    // FD Jacobian (1 m perturbation; truncation tiny at body distances ≥ 1e8 m)
    const Eigen::Matrix3d J_fd = fd_position_jacobian(
        [&](const Vector3d& r) { return tb.acceleration(t0, r); },
        r_sat, /*fd_dx=*/1.0);

    const double rel = rel_frobenius_diff(J_an, J_fd);
    std::cout << "    [" << label << "] |J_an|=" << std::scientific
              << std::setprecision(2) << J_an.norm()
              << " |J_fd|=" << J_fd.norm()
              << " rel=" << rel << std::endl;
    TEST_ASSERT(rel < tol);

    // Symmetry (analytic should be exactly symmetric — outer-product + I).
    const double sym = (J_an - J_an.transpose()).norm()
                     / std::max(J_an.norm(), 1e-30);
    TEST_ASSERT_MSG(sym < 1e-12, "third-body J should be symmetric");

    // Tracelessness (each per-body tide tensor is traceless; sum is too).
    const double trace_rel = std::abs(J_an.trace())
                           / std::max(J_an.norm(), 1e-30);
    TEST_ASSERT_MSG(trace_rel < 1e-12, "third-body J should be traceless");
}

}  // namespace

int main() {
    initialize({});
    TestRunner runner("Force Component Jacobians");

    // Standard satellite positions.
    const Vector3d r_leo  (6.778e6,  0.0,      0.0);       // ~400 km altitude
    const Vector3d r_leo_off(5e6, 3e6, 2e6);                // off-axis LEO
    const Vector3d r_meo  (2.65e7, 0.0,      0.0);          // ~20 Mm (GPS-like)
    const Vector3d r_geo  (4.2164e7, 0.0,    0.0);          // ~36 Mm GEO

    runner.add_test("3rd-body Sun: analytic vs FD at LEO/MEO/GEO", []() {
        test_third_body_at("Sun-LEO",     /*sun*/true, false, false, Vector3d(6.778e6,  0.0, 0.0));
        test_third_body_at("Sun-LEO-off", /*sun*/true, false, false, Vector3d(5e6, 3e6, 2e6));
        test_third_body_at("Sun-MEO",     /*sun*/true, false, false, Vector3d(2.65e7, 0.0, 0.0));
        test_third_body_at("Sun-GEO",     /*sun*/true, false, false, Vector3d(4.2164e7, 0.0, 0.0));
    });

    runner.add_test("3rd-body Moon: analytic vs FD at LEO/MEO/GEO", []() {
        test_third_body_at("Moon-LEO",     false, /*moon*/true, false, Vector3d(6.778e6, 0.0, 0.0));
        test_third_body_at("Moon-LEO-off", false, /*moon*/true, false, Vector3d(5e6, 3e6, 2e6));
        test_third_body_at("Moon-MEO",     false, /*moon*/true, false, Vector3d(2.65e7, 0.0, 0.0));
        test_third_body_at("Moon-GEO",     false, /*moon*/true, false, Vector3d(4.2164e7, 0.0, 0.0));
    });

    runner.add_test("3rd-body Jupiter: analytic vs FD at LEO/GEO", []() {
        test_third_body_at("Jup-LEO", false, false, /*jup*/true, Vector3d(6.778e6, 0.0, 0.0));
        test_third_body_at("Jup-GEO", false, false, /*jup*/true, Vector3d(4.2164e7, 0.0, 0.0));
    });

    runner.add_test("3rd-body Sun+Moon+Jupiter combined", []() {
        test_third_body_at("All-LEO",     true, true, true, Vector3d(6.778e6,  0.0, 0.0));
        test_third_body_at("All-LEO-off", true, true, true, Vector3d(5e6, 3e6, 2e6));
        test_third_body_at("All-MEO",     true, true, true, Vector3d(2.65e7, 0.0, 0.0));
        test_third_body_at("All-GEO",     true, true, true, Vector3d(4.2164e7, 0.0, 0.0));
    });

    // ===== SRP =====
    auto make_srp_only = []() {
        forces::RadiationPressure rad;
        rad.enable_srp(true);
        rad.enable_albedo(false);
        rad.enable_ir(false);
        forces::SatelliteRadiationProperties props;
        props.srp_area = 20.0;            // m² (synthetic)
        props.srp_reflectivity = 1.0;     // perfect mirror, C_R = 2
        rad.set_properties(props);
        rad.set_mass(2200.0);             // kg
        return rad;
    };

    runner.add_test("SRP: analytic vs FD at LEO/MEO/GEO (sunlit)", [&]() {
        auto rad = make_srp_only();
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        // Sat positions to test. All in sunlight (we'll verify by checking
        // |a_srp| > 0). The Jan 15 12:00 epoch has Sun roughly along the
        // ecliptic plane; positions chosen to avoid umbra (anti-sun side).
        const std::vector<std::pair<const char*, Vector3d>> cases = {
            {"LEO",     Vector3d(6.778e6, 0.0,    0.0)},
            {"LEO-off", Vector3d(5e6,     3e6,    2e6)},
            {"MEO",     Vector3d(2.65e7,  0.0,    0.0)},
            {"GEO",     Vector3d(4.2164e7, 0.0,   0.0)},
        };

        int sunlit = 0;
        for (const auto& [label, r_sat] : cases) {
            auto [a_an, J_an] = rad.acceleration_with_jacobian(t0, r_sat);

            // Skip shadowed positions (Sun direction at Jan 15 12:00 UTC may
            // place some test points behind Earth in cylindrical shadow).
            if (a_an.norm() < 1e-15) {
                std::cout << "    [SRP-" << label << "] (shadowed; skip)\n";
                continue;
            }
            ++sunlit;

            // Sanity: analytic acceleration matches existing acceleration().
            const Vector3d a_existing = rad.acceleration(t0, r_sat);
            TEST_ASSERT_MSG((a_an - a_existing).norm() <
                              1e-15 * std::max(a_existing.norm(), 1e-30),
                "analytic SRP a disagrees with rad.acceleration()");

            // FD vs analytic.
            const Eigen::Matrix3d J_fd = fd_position_jacobian(
                [&](const Vector3d& r) { return rad.acceleration(t0, r); },
                r_sat, /*fd_dx=*/1.0);

            const double rel = rel_frobenius_diff(J_an, J_fd);
            std::cout << "    [SRP-" << label << "] |J_an|=" << std::scientific
                      << std::setprecision(2) << J_an.norm()
                      << " |J_fd|=" << J_fd.norm()
                      << " rel=" << rel << std::endl;
            TEST_ASSERT(rel < 1e-3);

            // Symmetric and traceless.
            const double sym = (J_an - J_an.transpose()).norm()
                             / std::max(J_an.norm(), 1e-30);
            TEST_ASSERT_MSG(sym < 1e-12, "SRP J should be symmetric");

            const double trace_rel = std::abs(J_an.trace())
                                   / std::max(J_an.norm(), 1e-30);
            TEST_ASSERT_MSG(trace_rel < 1e-12, "SRP J should be traceless");
        }
        TEST_ASSERT_MSG(sunlit >= 2,
            "expected at least 2 sunlit test positions to validate against");
    });

    // ===== IR =====
    auto make_ir_only = []() {
        forces::RadiationPressure rad;
        rad.enable_srp(false);
        rad.enable_albedo(false);
        rad.enable_ir(true);
        forces::SatelliteRadiationProperties props;
        props.srp_area = 20.0;
        props.ir_emissivity = 0.8;       // typical thermal emissivity
        rad.set_properties(props);
        rad.set_mass(2200.0);
        return rad;
    };

    runner.add_test("IR: analytic vs FD at LEO/MEO/GEO", [&]() {
        auto rad = make_ir_only();
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        // LEO-off chosen above Earth radius (sqrt(50)·1e6 ≈ 7.07e6 m) to avoid
        // the r ≤ R_E zero-return branch.
        const std::vector<std::pair<const char*, Vector3d>> cases = {
            {"LEO",     Vector3d(6.778e6, 0.0,    0.0)},
            {"LEO-off", Vector3d(5e6,     3e6,    4e6)},
            {"MEO",     Vector3d(2.65e7,  0.0,    0.0)},
            {"GEO",     Vector3d(4.2164e7, 0.0,   0.0)},
        };

        for (const auto& [label, r_sat] : cases) {
            auto [a_an, J_an] = rad.acceleration_with_jacobian(t0, r_sat);
            TEST_ASSERT_MSG(a_an.norm() > 1e-15, "IR should be nonzero above Earth");

            // Sanity: analytic acceleration matches existing rad.acceleration().
            const Vector3d a_existing = rad.acceleration(t0, r_sat);
            TEST_ASSERT_MSG((a_an - a_existing).norm() <
                              1e-15 * std::max(a_existing.norm(), 1e-30),
                "IR analytic disagrees with rad.acceleration()");

            const Eigen::Matrix3d J_fd = fd_position_jacobian(
                [&](const Vector3d& r) { return rad.acceleration(t0, r); },
                r_sat, /*fd_dx=*/1.0);

            const double rel = rel_frobenius_diff(J_an, J_fd);
            std::cout << "    [IR-" << label << "] |J_an|=" << std::scientific
                      << std::setprecision(2) << J_an.norm()
                      << " rel=" << rel << std::endl;
            TEST_ASSERT(rel < 1e-3);

            // IR is symmetric and traceless (same form as SRP/third-body).
            const double sym = (J_an - J_an.transpose()).norm()
                             / std::max(J_an.norm(), 1e-30);
            TEST_ASSERT_MSG(sym < 1e-12, "IR J should be symmetric");
            const double trace_rel = std::abs(J_an.trace())
                                   / std::max(J_an.norm(), 1e-30);
            TEST_ASSERT_MSG(trace_rel < 1e-12, "IR J should be traceless");
        }
    });

    // ===== Albedo =====
    auto make_albedo_only = []() {
        forces::RadiationPressure rad;
        rad.enable_srp(false);
        rad.enable_albedo(true);
        rad.enable_ir(false);
        forces::SatelliteRadiationProperties props;
        props.srp_area = 20.0;
        props.albedo_reflectivity = 0.3;   // typical
        rad.set_properties(props);
        rad.set_mass(2200.0);
        return rad;
    };

    runner.add_test("Albedo: analytic vs FD on sunlit positions", [&]() {
        auto rad = make_albedo_only();
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        const std::vector<std::pair<const char*, Vector3d>> cases = {
            {"LEO",     Vector3d(6.778e6, 0.0,    0.0)},
            {"LEO-off", Vector3d(5e6,     3e6,    2e6)},
            {"MEO",     Vector3d(2.65e7,  0.0,    0.0)},
            {"GEO",     Vector3d(4.2164e7, 0.0,   0.0)},
        };

        int sunlit = 0;
        for (const auto& [label, r_sat] : cases) {
            auto [a_an, J_an] = rad.acceleration_with_jacobian(t0, r_sat);
            if (a_an.norm() < 1e-18) {
                std::cout << "    [Alb-" << label << "] (anti-sun; skip)\n";
                continue;
            }
            ++sunlit;

            // Sanity: analytic matches existing acceleration().
            const Vector3d a_existing = rad.acceleration(t0, r_sat);
            TEST_ASSERT_MSG((a_an - a_existing).norm() <
                              1e-15 * std::max(a_existing.norm(), 1e-30),
                "Albedo analytic disagrees with rad.acceleration()");

            const Eigen::Matrix3d J_fd = fd_position_jacobian(
                [&](const Vector3d& r) { return rad.acceleration(t0, r); },
                r_sat, /*fd_dx=*/1.0);

            const double rel = rel_frobenius_diff(J_an, J_fd);
            std::cout << "    [Alb-" << label << "] |J_an|=" << std::scientific
                      << std::setprecision(2) << J_an.norm()
                      << " rel=" << rel << std::endl;
            TEST_ASSERT(rel < 1e-3);

            // Albedo J is NOT symmetric in general (cos_phi factor breaks
            // gradient structure). Don't enforce symmetry. Don't enforce
            // tracelessness either.
        }
        TEST_ASSERT_MSG(sunlit >= 2, "expected ≥2 sunlit positions to validate");
    });

    runner.add_test("SRP Jacobian sign convention (radial vs perpendicular)", [&]() {
        // SRP weakens with distance from Sun, so radial sensitivity is
        // negative: ∂a_srp_radial / ∂r_radial < 0.
        // Eigenvalues: -2/r³ along ŝ (sat-from-sun direction), +1/r³
        // perpendicular. Verify a positive radial perturbation along ŝ
        // gives a negative change in a_srp.
        auto rad = make_srp_only();
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        const Vector3d r_sat(6.778e6, 0.0, 0.0);

        auto [a_an, J_an] = rad.acceleration_with_jacobian(t0, r_sat);

        // ŝ direction: from Sun toward satellite. Use a's direction (which is
        // along ŝ since a = K · s/r³ has K > 0).
        const Vector3d s_hat = a_an.normalized();

        // J · ŝ should be approximately -2/r³ * ŝ (eigenvector along ŝ with
        // eigenvalue -2/r³). Verify it points OPPOSITE to ŝ.
        const Vector3d J_s = J_an * s_hat;
        TEST_ASSERT_MSG(J_s.dot(s_hat) < 0.0,
            "SRP J along ŝ direction should give negative eigenvalue");
    });

    // ===== Relativistic =====
    //
    // Magnitudes at LEO: Schwarzschild ~1.5e-8 m/s², Lense-Thirring ~1e-11 m/s².
    // Both depend on r AND v. Use full 3×6 Jacobian comparison.
    auto fd_position_velocity_jacobian = [](auto&& accel_fn,
                                             const Vector3d& pos, const Vector3d& vel,
                                             double fd_dx, double fd_dv) {
        Eigen::Matrix<double, 3, 6> J;
        const Vector3d a0 = accel_fn(pos, vel);
        for (int i = 0; i < 3; ++i) {
            Vector3d rp = pos; rp(i) += fd_dx;
            J.col(i) = (accel_fn(rp, vel) - a0) / fd_dx;
        }
        for (int i = 0; i < 3; ++i) {
            Vector3d vp = vel; vp(i) += fd_dv;
            J.col(3 + i) = (accel_fn(pos, vp) - a0) / fd_dv;
        }
        return J;
    };

    runner.add_test("Schwarzschild: analytic vs FD at LEO", [&]() {
        // LEO orbit: r = (R+400km, 0, 0), v = circular speed in y-direction.
        const Vector3d r_sat(6.778e6, 0.0, 0.0);
        const double v_circ = std::sqrt(constants::earth::GM / r_sat.norm());
        const Vector3d v_sat(0.0, v_circ, 0.0);

        auto [a_an, J_an] = forces::schwarzschild_acceleration_with_jacobian(r_sat, v_sat);

        // Sanity: |a| ~ 1.5e-8 m/s² at LEO per docstring.
        TEST_ASSERT_MSG(a_an.norm() > 1e-9 && a_an.norm() < 1e-7,
            "Schwarzschild magnitude at LEO out of expected range");

        const auto J_fd = fd_position_velocity_jacobian(
            [&](const Vector3d& r, const Vector3d& v) {
                return forces::schwarzschild_acceleration(r, v);
            },
            r_sat, v_sat, /*fd_dx=*/1.0, /*fd_dv=*/1e-3);

        const double rel = (J_an - J_fd).norm() / std::max(J_fd.norm(), 1e-30);
        std::cout << "    [Sch-LEO] |J_an|=" << std::scientific << std::setprecision(2)
                  << J_an.norm() << " |J_fd|=" << J_fd.norm()
                  << " rel=" << rel << std::endl;
        // Generous tolerance — Schwarzschild magnitudes are tiny so FD has
        // wider noise floor. 1e-3 is plenty (we expect ~1e-5 in practice).
        TEST_ASSERT(rel < 1e-3);
    });

    runner.add_test("Schwarzschild: analytic vs FD at MEO and elliptical orbit", [&]() {
        // MEO (~GPS altitude)
        const Vector3d r_meo(2.65e7, 0.0, 0.0);
        const double v_meo = std::sqrt(constants::earth::GM / r_meo.norm());
        const Vector3d v_meo_circ(0.0, v_meo, 0.0);

        auto [a1, J1] = forces::schwarzschild_acceleration_with_jacobian(r_meo, v_meo_circ);
        const auto J_fd1 = fd_position_velocity_jacobian(
            [&](const Vector3d& r, const Vector3d& v) {
                return forces::schwarzschild_acceleration(r, v);
            },
            r_meo, v_meo_circ, 1.0, 1e-3);
        const double rel1 = (J1 - J_fd1).norm() / std::max(J_fd1.norm(), 1e-30);
        std::cout << "    [Sch-MEO-circ] rel=" << std::scientific << std::setprecision(2)
                  << rel1 << std::endl;
        TEST_ASSERT(rel1 < 1e-3);

        // Elliptical orbit (r and v not perpendicular, r·v ≠ 0)
        const Vector3d r_ell(7.0e6, 1.5e6, -2e5);
        const Vector3d v_ell(2.0e3, 7.5e3, 0.5e3);
        auto [a2, J2] = forces::schwarzschild_acceleration_with_jacobian(r_ell, v_ell);
        const auto J_fd2 = fd_position_velocity_jacobian(
            [&](const Vector3d& r, const Vector3d& v) {
                return forces::schwarzschild_acceleration(r, v);
            },
            r_ell, v_ell, 1.0, 1e-3);
        const double rel2 = (J2 - J_fd2).norm() / std::max(J_fd2.norm(), 1e-30);
        std::cout << "    [Sch-elliptical] rel=" << std::scientific << std::setprecision(2)
                  << rel2 << std::endl;
        TEST_ASSERT(rel2 < 1e-3);
    });

    runner.add_test("Lense-Thirring: analytic vs FD at LEO and elliptical orbit", [&]() {
        const Vector3d r_leo(6.778e6, 0.0, 0.0);
        const double v_circ = std::sqrt(constants::earth::GM / r_leo.norm());
        const Vector3d v_leo(0.0, v_circ, 0.0);

        auto [a1, J1] = forces::lense_thirring_acceleration_with_jacobian(r_leo, v_leo);
        // Sanity: |a_LT| ~1e-11 at LEO per docstring.
        TEST_ASSERT_MSG(a1.norm() > 1e-13 && a1.norm() < 1e-9,
            "Lense-Thirring magnitude at LEO out of expected range");

        const auto J_fd1 = fd_position_velocity_jacobian(
            [&](const Vector3d& r, const Vector3d& v) {
                return forces::lense_thirring_acceleration(r, v);
            },
            r_leo, v_leo, 1.0, 1e-3);
        const double rel1 = (J1 - J_fd1).norm() / std::max(J_fd1.norm(), 1e-30);
        std::cout << "    [LT-LEO-circ] |J_an|=" << std::scientific << std::setprecision(2)
                  << J1.norm() << " rel=" << rel1 << std::endl;
        // Lense-Thirring is even tinier than Schwarzschild — looser tolerance.
        TEST_ASSERT(rel1 < 1e-2);

        // Elliptical, off-z-axis state to exercise all terms (r·J ≠ 0, etc.)
        const Vector3d r_ell(5e6, 4e6, 3e6);
        const Vector3d v_ell(-3e3, 5e3, 4e3);
        auto [a2, J2] = forces::lense_thirring_acceleration_with_jacobian(r_ell, v_ell);
        const auto J_fd2 = fd_position_velocity_jacobian(
            [&](const Vector3d& r, const Vector3d& v) {
                return forces::lense_thirring_acceleration(r, v);
            },
            r_ell, v_ell, 1.0, 1e-3);
        const double rel2 = (J2 - J_fd2).norm() / std::max(J_fd2.norm(), 1e-30);
        std::cout << "    [LT-elliptical] |J_an|=" << std::scientific << std::setprecision(2)
                  << J2.norm() << " rel=" << rel2 << std::endl;
        TEST_ASSERT(rel2 < 1e-2);
    });

    // ===== Drag =====

    //
    // Drag's analytic Jacobian: ∂a/∂v is closed form (Rᵀ·M_v·R), ∂a/∂r is FD.
    // The interesting validation is the ∂a/∂v block — it should agree with
    // FD-on-velocity to high precision. Position columns are identical to FD
    // by construction (both paths FD position).
    runner.add_test("Drag: analytic ∂a/∂v vs FD at low LEO (env path)", [&]() {
        // Build a uniform DragEnvironment with NRLMSIS21-like density at
        // ~200 km (~3e-10 kg/m³) and a representative ballistic coefficient
        // β = m/(Cd·A) of 256.4 kg/m² (not any registry satellite's value).
        // The Jacobian math is independent of the density model — uniform
        // ρ is sufficient to exercise the M_v analytic block.
        atmosphere::DensityProfile1D density({100.0, 1500.0}, {3.0e-10, 3.0e-10});
        atmosphere::WindProfile1D_ENU wind;  // empty → zero wind
        atmosphere::BetaProfile1D beta = atmosphere::BetaProfile1D::constant(256.4);
        atmosphere::DragEnvironment env(std::move(density), std::move(wind),
                                        std::move(beta));

        forces::AtmosphericDrag drag;

        // ~200 km altitude, near-circular orbit.
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        const Vector3d r_sat(constants::earth::RADIUS_EQUATORIAL + 200e3, 0.0, 0.0);
        const double v_circ = std::sqrt(constants::earth::GM / r_sat.norm());
        const Vector3d v_sat(0.0, v_circ, 0.0);
        const StateVector state{r_sat, v_sat};

        auto [a_an, J_an] = drag.acceleration_with_jacobian(env, t0, state);
        const double a_mag = a_an.norm();
        std::cout << "    [Drag-LEO200] |a|=" << std::scientific << std::setprecision(2)
                  << a_mag << std::endl;
        TEST_ASSERT_MSG(a_mag > 1e-10, "drag should be non-negligible at 200 km");

        // FD ∂a/∂v (3 perturbations) for comparison.
        Eigen::Matrix3d J_fd_v;
        const double fd_dv = 1e-3;
        for (int i = 0; i < 3; ++i) {
            StateVector xp = state;
            xp.velocity(i) += fd_dv;
            J_fd_v.col(i) = (drag.acceleration(env, t0, xp) - a_an) / fd_dv;
        }

        // Compare analytic ∂a/∂v block to FD.
        const Eigen::Matrix3d J_an_v = J_an.block<3, 3>(0, 3);
        const double rel = rel_frobenius_diff(J_an_v, J_fd_v);
        std::cout << "    [Drag-LEO200] |∂a/∂v_an|=" << std::scientific
                  << std::setprecision(2) << J_an_v.norm()
                  << " rel=" << rel << std::endl;
        TEST_ASSERT(rel < 1e-3);

        // Sanity: analytic acceleration matches plain acceleration.
        const Vector3d a_existing = drag.acceleration(env, t0, state);
        TEST_ASSERT_MSG((a_an - a_existing).norm() <
                          1e-15 * std::max(a_existing.norm(), 1e-30),
            "drag analytic disagrees with drag.acceleration()");
    });

    runner.add_test("Drag: zero Jacobian when drag env is empty", [&]() {
        // Empty DragEnvironment (has_drag() == false) ⇒ zero acceleration
        // and zero Jacobian regardless of position.
        atmosphere::DragEnvironment env;  // default-constructed
        forces::AtmosphericDrag drag;

        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        const Vector3d r_sat(4.2164e7, 0.0, 0.0);  // GEO altitude
        const double v_geo = std::sqrt(constants::earth::GM / r_sat.norm());
        const Vector3d v_sat(0.0, v_geo, 0.0);
        const StateVector state{r_sat, v_sat};

        auto [a, J] = drag.acceleration_with_jacobian(env, t0, state);
        TEST_ASSERT_MSG(a.norm() < 1e-15, "drag should be zero when env is empty");
        TEST_ASSERT_MSG(J.norm() < 1e-30, "Jacobian should be zero when env is empty");
    });

    runner.add_test("3rd-body Jacobian magnitudes match expectation", []() {
        // Order-of-magnitude sanity: |J| ~ GM/d³ where d ~ body distance.
        // Sun at LEO: GM=1.327e20, d~1.5e11 → ~4e-14 /s²
        // Moon at LEO: GM=4.9e12, d~3.84e8 → ~8.6e-14 /s²
        gravity::ThirdBodyGravity tb_sun, tb_moon;
        tb_sun.enable_sun(true); tb_sun.enable_moon(false);
        tb_moon.enable_sun(false); tb_moon.enable_moon(true);

        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        const Vector3d r_sat(6.778e6, 0.0, 0.0);

        auto [a_sun, J_sun] = tb_sun.acceleration_with_jacobian(t0, r_sat);
        auto [a_moon, J_moon] = tb_moon.acceleration_with_jacobian(t0, r_sat);

        // Magnitudes (loose bounds — solar/lunar geometry at this epoch may
        // shift these slightly, but order-of-magnitude is robust).
        TEST_ASSERT(J_sun.norm() > 1e-15 && J_sun.norm() < 1e-12);
        TEST_ASSERT(J_moon.norm() > 1e-14 && J_moon.norm() < 1e-12);
    });

    runner.run();
    return print_final_summary();
}
