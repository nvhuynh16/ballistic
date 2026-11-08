#pragma once

// =============================================================================
// J2 Closed-Form Acceleration and Hessian (test-only reference)
// =============================================================================
//
// Provides closed-form analytic expressions for the J2 (l=2, m=0) zonal
// gravity contribution: potential, acceleration, and Hessian. Used as an
// independent ground truth to anchor the deg=2 limit of any analytic
// gravity Hessian implementation.
//
// Convention: geodesy potential V = +GM/r (positive, so a = +∇V; for
// physical gravity acting toward center, ∇V points inward at the
// pointmass leading term).
//
// Reference:
// ----------
// Vallado, D. A. (2013). "Fundamentals of Astrodynamics and
// Applications", 4th ed., Algorithm 65 / Eq. 8-37.
//
// Math:
// -----
// V_J2(x, y, z) = −GM·J2·R² · (3z² − r²) / (2·r⁵)
//               = −GM·J2·R² · (2z² − x² − y²) / (2·r⁵)
//
// Let K = −GM·J2·R²/2, N(x,y,z) = 2z² − x² − y² (so V_J2 = K·N/r⁵).
// Define basic partials:
//     ∂N/∂x = −2x,   ∂N/∂y = −2y,   ∂N/∂z = +4z
//     ∂²N/∂x² = −2,  ∂²N/∂y² = −2,  ∂²N/∂z² = +4,   all cross zero
//
// Acceleration a = ∇V (per geodesy convention):
//     a_i = ∂V/∂x_i = K · [(∂N/∂x_i)/r⁵ − 5·N·x_i/r⁷]
//
// Hessian H_ij = ∂²V/∂x_i∂x_j:
//     H_ij = K · [(∂²N/∂x_i∂x_j)/r⁵
//                  − 5·((∂N/∂x_i)·x_j + (∂N/∂x_j)·x_i)/r⁷
//                  − 5·δ_{ij}·N/r⁷
//                  + 35·N·x_i·x_j/r⁹]
//
// Properties verified by tests using these closed forms:
//   - tr(H) = 0 (Laplace's equation: J2 is harmonic)
//   - H = Hᵀ (Schwarz)
//   - All entries finite for any r > 0 (J2 is regular off the origin)
//
// Acceptance criterion when comparing against analytic Hessian engine:
//   ‖H_engine − H_J2‖_F / ‖H_J2‖_F  <  1e-14   (essentially exact)
// =============================================================================

#include "../../include/ballistic/core/types.hpp"
#include "../../include/ballistic/core/constants.hpp"
#include <Eigen/Dense>

namespace ballistic {
namespace test {

struct J2Result {
    double potential;
    Vector3d acceleration;
    Eigen::Matrix3d hessian;
};

// J2_norm: the unnormalized J2 (≈ 1.0826e-3 for Earth).
// Note: if you have C̄_20 (fully-normalized), use J2 = −√5 · C̄_20.
[[nodiscard]] inline J2Result j2_closed_form(
    const Vector3d& pos_ecef,
    double J2 = 1.0826267e-3,
    double R = constants::earth::RADIUS_EQUATORIAL,
    double GM = constants::earth::GM)
{
    const double x = pos_ecef.x();
    const double y = pos_ecef.y();
    const double z = pos_ecef.z();

    const double r2 = x*x + y*y + z*z;
    const double r = std::sqrt(r2);
    const double r3 = r * r2;
    const double r5 = r3 * r2;
    const double r7 = r5 * r2;
    const double r9 = r7 * r2;

    // V_J2 = K · N / r⁵
    const double K = -GM * J2 * R * R / 2.0;
    const double N = 2.0 * z * z - x * x - y * y;

    // Partials of N
    const double Nx = -2.0 * x;
    const double Ny = -2.0 * y;
    const double Nz =  4.0 * z;
    // Second partials of N (constants)
    const double Nxx = -2.0, Nyy = -2.0, Nzz = 4.0;
    // Off-diagonal Nxy = Nxz = Nyz = 0.

    J2Result out;
    out.potential = K * N / r5;
    out.acceleration.x() = K * (Nx / r5 - 5.0 * N * x / r7);
    out.acceleration.y() = K * (Ny / r5 - 5.0 * N * y / r7);
    out.acceleration.z() = K * (Nz / r5 - 5.0 * N * z / r7);

    // Hessian
    auto H_entry = [&](double Nij, double Ni, double Nj, double xi, double xj, double delta_ij) {
        return K * (Nij / r5
                    - 5.0 * (Ni * xj + Nj * xi) / r7
                    - 5.0 * delta_ij * N / r7
                    + 35.0 * N * xi * xj / r9);
    };

    out.hessian(0, 0) = H_entry(Nxx, Nx, Nx, x, x, 1.0);
    out.hessian(1, 1) = H_entry(Nyy, Ny, Ny, y, y, 1.0);
    out.hessian(2, 2) = H_entry(Nzz, Nz, Nz, z, z, 1.0);
    out.hessian(0, 1) = H_entry(0.0, Nx, Ny, x, y, 0.0);
    out.hessian(0, 2) = H_entry(0.0, Nx, Nz, x, z, 0.0);
    out.hessian(1, 2) = H_entry(0.0, Ny, Nz, y, z, 0.0);
    // Symmetric
    out.hessian(1, 0) = out.hessian(0, 1);
    out.hessian(2, 0) = out.hessian(0, 2);
    out.hessian(2, 1) = out.hessian(1, 2);

    return out;
}

// Assertion helper: closed-form J2 satisfies Laplace + symmetry.
// Returns max violation; should be ≤ 1e-15 for any test point.
[[nodiscard]] inline double j2_closed_form_self_check(const Vector3d& pos_ecef) {
    auto result = j2_closed_form(pos_ecef);
    const auto& H = result.hessian;
    const double trace = H(0, 0) + H(1, 1) + H(2, 2);
    const double sym_xy = std::abs(H(0, 1) - H(1, 0));
    const double sym_xz = std::abs(H(0, 2) - H(2, 0));
    const double sym_yz = std::abs(H(1, 2) - H(2, 1));
    const double scale = H.norm() + 1e-30;
    return std::max({std::abs(trace) / scale, sym_xy, sym_xz, sym_yz});
}

} // namespace test
} // namespace ballistic
