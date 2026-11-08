#pragma once

// =============================================================================
// Quintic Hermite Interpolation
// =============================================================================
//
// Quintic (5th degree) Hermite interpolation: a polynomial between two time points that
// matches the position, velocity and acceleration at both ends. Its error shrinks as h^6
// (written O(h^6)) for a spacing h between the points, which gives sub-millimeter accuracy for
// typical orbital propagation step sizes (10-30 seconds). Used to read a propagated trajectory
// between its integration steps.
//
// The interpolation polynomial P(s) satisfies:
//   P(0)   = p0      P(1)   = p1       (position match)
//   P'(0)  = v0*dt   P'(1)  = v1*dt    (velocity match)
//   P''(0) = a0*dt²  P''(1) = a1*dt²   (acceleration match)
//
// where s ∈ [0,1] is the normalized parameter and dt is the time interval.
//
// Basis functions (evaluated at normalized parameter s):
//   h00(s) = 1 - 10s³ + 15s⁴ - 6s⁵     (position at s=0)
//   h10(s) = s - 6s³ + 8s⁴ - 3s⁵       (first derivative at s=0)
//   h20(s) = 0.5s² - 1.5s³ + 1.5s⁴ - 0.5s⁵  (second derivative at s=0)
//   h01(s) = 10s³ - 15s⁴ + 6s⁵         (position at s=1)
//   h11(s) = -4s³ + 7s⁴ - 3s⁵          (first derivative at s=1)
//   h21(s) = 0.5s³ - s⁴ + 0.5s⁵        (second derivative at s=1)
//
// References:
// -----------
// [1] Burden, R.L. & Faires, J.D. "Numerical Analysis", 9th Ed., Chapter 3.
//     - Hermite polynomial interpolation theory
// [2] de Boor, C. "A Practical Guide to Splines", Springer, 2001.
//     - Spline interpolation and basis functions
// [3] Hairer, E., Norsett, S.P. & Wanner, G. "Solving Ordinary Differential
//     Equations I", 2nd Ed., Springer, 1993. Chapter II.6: Dense Output.
//     - Hermite interpolation using derivative data from ODE solvers
//
// =============================================================================

#include "../core/types.hpp"
#include <cmath>

namespace ballistic {
namespace math {

// =============================================================================
// Basis Function Structures
// =============================================================================

/// Values of the six quintic Hermite basis functions (or of their first or second
/// derivatives, depending on which function filled it) at one parameter s. The interpolant is
/// h00*p0 + h10*(v0*dt) + h20*(a0*dt²) + h01*p1 + h11*(v1*dt) + h21*(a1*dt²).
struct QuinticHermiteBasis {
    double h00;  // Coefficient for p0 (position at endpoint 0)
    double h10;  // Coefficient for v0*dt (first derivative at endpoint 0)
    double h20;  // Coefficient for a0*dt² (second derivative at endpoint 0)
    double h01;  // Coefficient for p1 (position at endpoint 1)
    double h11;  // Coefficient for v1*dt (first derivative at endpoint 1)
    double h21;  // Coefficient for a1*dt² (second derivative at endpoint 1)
};

// =============================================================================
// Basis Function Evaluation
// =============================================================================

/// Quintic Hermite basis functions at normalized parameter s ∈ [0,1] (0 = start, 1 = end).
//
// These basis functions satisfy the following endpoint conditions:
//   h00: h00(0)=1, h00(1)=0, derivatives at both ends = 0
//   h10: h10'(0)=1, h10(0)=h10(1)=0, h10'(1)=0, h10''(both)=0
//   h20: h20''(0)=1, h20(0)=h20(1)=0, h20'(both)=0, h20''(1)=0
//   h01: h01(1)=1, h01(0)=0, derivatives at both ends = 0
//   h11: h11'(1)=1, h11(0)=h11(1)=0, h11'(0)=0, h11''(both)=0
//   h21: h21''(1)=1, h21(0)=h21(1)=0, h21'(both)=0, h21''(0)=0
[[nodiscard]] inline QuinticHermiteBasis quintic_hermite_basis(double s) noexcept {
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;
    const double s5 = s4 * s;

    QuinticHermiteBasis basis;

    // Basis for p0: h00(s) = 1 - 10s³ + 15s⁴ - 6s⁵
    basis.h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5;

    // Basis for v0*dt: h10(s) = s - 6s³ + 8s⁴ - 3s⁵
    basis.h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5;

    // Basis for a0*dt²: h20(s) = 0.5s² - 1.5s³ + 1.5s⁴ - 0.5s⁵
    basis.h20 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5;

    // Basis for p1: h01(s) = 10s³ - 15s⁴ + 6s⁵
    basis.h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;

    // Basis for v1*dt: h11(s) = -4s³ + 7s⁴ - 3s⁵
    basis.h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5;

    // Basis for a1*dt²: h21(s) = 0.5s³ - s⁴ + 0.5s⁵
    basis.h21 = 0.5 * s3 - s4 + 0.5 * s5;

    return basis;
}

/// First derivatives dh/ds of the basis functions at s ∈ [0,1].
/// Used for computing velocity: dP/dt = (dP/ds) / dt
[[nodiscard]] inline QuinticHermiteBasis quintic_hermite_deriv1(double s) noexcept {
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;

    QuinticHermiteBasis basis;

    // d/ds h00 = -30s² + 60s³ - 30s⁴
    basis.h00 = -30.0 * s2 + 60.0 * s3 - 30.0 * s4;

    // d/ds h10 = 1 - 18s² + 32s³ - 15s⁴
    basis.h10 = 1.0 - 18.0 * s2 + 32.0 * s3 - 15.0 * s4;

    // d/ds h20 = s - 4.5s² + 6s³ - 2.5s⁴
    basis.h20 = s - 4.5 * s2 + 6.0 * s3 - 2.5 * s4;

    // d/ds h01 = 30s² - 60s³ + 30s⁴
    basis.h01 = 30.0 * s2 - 60.0 * s3 + 30.0 * s4;

    // d/ds h11 = -12s² + 28s³ - 15s⁴
    basis.h11 = -12.0 * s2 + 28.0 * s3 - 15.0 * s4;

    // d/ds h21 = 1.5s² - 4s³ + 2.5s⁴
    basis.h21 = 1.5 * s2 - 4.0 * s3 + 2.5 * s4;

    return basis;
}

/// Second derivatives d²h/ds² of the basis functions at s ∈ [0,1].
/// Used for computing acceleration: d²P/dt² = (d²P/ds²) / dt²
[[nodiscard]] inline QuinticHermiteBasis quintic_hermite_deriv2(double s) noexcept {
    const double s2 = s * s;
    const double s3 = s2 * s;

    QuinticHermiteBasis basis;

    // d²/ds² h00 = -60s + 180s² - 120s³
    basis.h00 = -60.0 * s + 180.0 * s2 - 120.0 * s3;

    // d²/ds² h10 = -36s + 96s² - 60s³
    basis.h10 = -36.0 * s + 96.0 * s2 - 60.0 * s3;

    // d²/ds² h20 = 1 - 9s + 18s² - 10s³
    basis.h20 = 1.0 - 9.0 * s + 18.0 * s2 - 10.0 * s3;

    // d²/ds² h01 = 60s - 180s² + 120s³
    basis.h01 = 60.0 * s - 180.0 * s2 + 120.0 * s3;

    // d²/ds² h11 = -24s + 84s² - 60s³
    basis.h11 = -24.0 * s + 84.0 * s2 - 60.0 * s3;

    // d²/ds² h21 = 3s - 12s² + 10s³
    basis.h21 = 3.0 * s - 12.0 * s2 + 10.0 * s3;

    return basis;
}

// =============================================================================
// Interpolation Functions
// =============================================================================

/// Interpolated position at normalized parameter s ∈ [0,1]. Values of s outside [0,1]
/// extrapolate the polynomial (no check).
//
// Parameters:
//   s  - Normalized parameter (0 = start, 1 = end)
//   dt - Time interval between endpoints (seconds)
//   p0, v0, a0 - Position (m), velocity (m/s), acceleration (m/s²) at start
//   p1, v1, a1 - Position (m), velocity (m/s), acceleration (m/s²) at end
//
// Returns:
//   Interpolated position at parameter s
//
// Note: the basis is normalized so that h20''(0) = h21''(1) = 1, so the acceleration enters as
// a*dt² (not a*dt²/2); that gives d²P/ds² = a*dt² at the endpoints, as required.
[[nodiscard]] inline Vector3d quintic_hermite_position(
    double s, double dt,
    const Vector3d& p0, const Vector3d& v0, const Vector3d& a0,
    const Vector3d& p1, const Vector3d& v1, const Vector3d& a1) noexcept
{
    const auto basis = quintic_hermite_basis(s);
    const double dt2 = dt * dt;

    return basis.h00 * p0
         + basis.h10 * (v0 * dt)
         + basis.h20 * (a0 * dt2)
         + basis.h01 * p1
         + basis.h11 * (v1 * dt)
         + basis.h21 * (a1 * dt2);
}

/// Interpolated velocity [m/s] at normalized parameter s ∈ [0,1].
//
// Velocity is the first derivative of position with respect to time:
//   v(t) = dP/dt = (dP/ds) * (ds/dt) = (dP/ds) / dt
//
// Parameters:
//   s  - Normalized parameter (0 = start, 1 = end)
//   dt - Time interval between endpoints (seconds)
//   p0, v0, a0 - Position (m), velocity (m/s), acceleration (m/s²) at start
//   p1, v1, a1 - Position (m), velocity (m/s), acceleration (m/s²) at end
//
// Returns:
//   Interpolated velocity at parameter s (m/s)
[[nodiscard]] inline Vector3d quintic_hermite_velocity(
    double s, double dt,
    const Vector3d& p0, const Vector3d& v0, const Vector3d& a0,
    const Vector3d& p1, const Vector3d& v1, const Vector3d& a1) noexcept
{
    const auto basis = quintic_hermite_deriv1(s);
    const double dt2 = dt * dt;

    // dP/ds
    Vector3d dP_ds = basis.h00 * p0
                   + basis.h10 * (v0 * dt)
                   + basis.h20 * (a0 * dt2)
                   + basis.h01 * p1
                   + basis.h11 * (v1 * dt)
                   + basis.h21 * (a1 * dt2);

    // dP/dt = dP/ds / dt
    return dP_ds / dt;
}

/// Interpolated acceleration [m/s²] at normalized parameter s ∈ [0,1].
//
// Acceleration is the second derivative of position with respect to time:
//   a(t) = d²P/dt² = (d²P/ds²) * (ds/dt)² = (d²P/ds²) / dt²
//
// Parameters:
//   s  - Normalized parameter (0 = start, 1 = end)
//   dt - Time interval between endpoints (seconds)
//   p0, v0, a0 - Position (m), velocity (m/s), acceleration (m/s²) at start
//   p1, v1, a1 - Position (m), velocity (m/s), acceleration (m/s²) at end
//
// Returns:
//   Interpolated acceleration at parameter s (m/s²)
[[nodiscard]] inline Vector3d quintic_hermite_acceleration(
    double s, double dt,
    const Vector3d& p0, const Vector3d& v0, const Vector3d& a0,
    const Vector3d& p1, const Vector3d& v1, const Vector3d& a1) noexcept
{
    const auto basis = quintic_hermite_deriv2(s);
    const double dt2 = dt * dt;

    // d²P/ds²
    Vector3d d2P_ds2 = basis.h00 * p0
                     + basis.h10 * (v0 * dt)
                     + basis.h20 * (a0 * dt2)
                     + basis.h01 * p1
                     + basis.h11 * (v1 * dt)
                     + basis.h21 * (a1 * dt2);

    // d²P/dt² = d²P/ds² / dt²
    return d2P_ds2 / dt2;
}

// =============================================================================
// Combined State Evaluation
// =============================================================================

/// Interpolated position, velocity and acceleration at one point.
struct QuinticHermiteState {
    Vector3d position;      // Interpolated position (m)
    Vector3d velocity;      // Interpolated velocity (m/s)
    Vector3d acceleration;  // Interpolated acceleration (m/s²)
};

/// Position, velocity and acceleration at normalized parameter s ∈ [0,1] in one call.
//
// This is more efficient than calling position/velocity/acceleration separately
// because common basis function powers are computed only once.
//
// Parameters:
//   s  - Normalized parameter (0 = start, 1 = end)
//   dt - Time interval between endpoints (seconds)
//   p0, v0, a0 - Position (m), velocity (m/s), acceleration (m/s²) at start
//   p1, v1, a1 - Position (m), velocity (m/s), acceleration (m/s²) at end
//
// Returns:
//   Full interpolated state (position, velocity, acceleration)
[[nodiscard]] inline QuinticHermiteState quintic_hermite_state(
    double s, double dt,
    const Vector3d& p0, const Vector3d& v0, const Vector3d& a0,
    const Vector3d& p1, const Vector3d& v1, const Vector3d& a1) noexcept
{
    // Precompute powers of s
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double s4 = s3 * s;
    const double s5 = s4 * s;

    // Time scaling factors
    const double dt2 = dt * dt;
    const double inv_dt = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;

    // Scaled endpoint data
    const Vector3d v0_dt = v0 * dt;
    const Vector3d v1_dt = v1 * dt;
    const Vector3d a0_dt2 = a0 * dt2;
    const Vector3d a1_dt2 = a1 * dt2;

    QuinticHermiteState state;

    // Position basis functions
    const double h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5;
    const double h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5;
    const double h20 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5;
    const double h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
    const double h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5;
    const double h21 = 0.5 * s3 - s4 + 0.5 * s5;

    state.position = h00 * p0 + h10 * v0_dt + h20 * a0_dt2
                   + h01 * p1 + h11 * v1_dt + h21 * a1_dt2;

    // First derivative basis functions
    const double dh00 = -30.0 * s2 + 60.0 * s3 - 30.0 * s4;
    const double dh10 = 1.0 - 18.0 * s2 + 32.0 * s3 - 15.0 * s4;
    const double dh20 = s - 4.5 * s2 + 6.0 * s3 - 2.5 * s4;
    const double dh01 = 30.0 * s2 - 60.0 * s3 + 30.0 * s4;
    const double dh11 = -12.0 * s2 + 28.0 * s3 - 15.0 * s4;
    const double dh21 = 1.5 * s2 - 4.0 * s3 + 2.5 * s4;

    Vector3d dP_ds = dh00 * p0 + dh10 * v0_dt + dh20 * a0_dt2
                   + dh01 * p1 + dh11 * v1_dt + dh21 * a1_dt2;
    state.velocity = dP_ds * inv_dt;

    // Second derivative basis functions
    const double d2h00 = -60.0 * s + 180.0 * s2 - 120.0 * s3;
    const double d2h10 = -36.0 * s + 96.0 * s2 - 60.0 * s3;
    const double d2h20 = 1.0 - 9.0 * s + 18.0 * s2 - 10.0 * s3;
    const double d2h01 = 60.0 * s - 180.0 * s2 + 120.0 * s3;
    const double d2h11 = -24.0 * s + 84.0 * s2 - 60.0 * s3;
    const double d2h21 = 3.0 * s - 12.0 * s2 + 10.0 * s3;

    Vector3d d2P_ds2 = d2h00 * p0 + d2h10 * v0_dt + d2h20 * a0_dt2
                     + d2h01 * p1 + d2h11 * v1_dt + d2h21 * a1_dt2;
    state.acceleration = d2P_ds2 * inv_dt2;

    return state;
}

// =============================================================================
// Precomputed Polynomial Coefficients
// =============================================================================

/// One segment's interpolant in ordinary polynomial form, for repeated evaluation:
/// P(s) = c0 + c1*s + c2*s² + c3*s³ + c4*s⁴ + c5*s⁵ for each component (x, y, z), evaluated
/// with Horner's method (nested multiplication). Build it with `compute_segment_coefficients`.
struct QuinticSegmentCoefficients {
    // Coefficients for x, y, z components: coeff[component][power]
    // coeff[0] = x coefficients, coeff[1] = y, coeff[2] = z
    double coeff[3][6];

    /// Position [m] at normalized parameter s.
    [[nodiscard]] Vector3d position(double s) const noexcept {
        Vector3d result;
        for (int i = 0; i < 3; ++i) {
            // Horner's method: c0 + s*(c1 + s*(c2 + s*(c3 + s*(c4 + s*c5))))
            double val = coeff[i][5];
            val = val * s + coeff[i][4];
            val = val * s + coeff[i][3];
            val = val * s + coeff[i][2];
            val = val * s + coeff[i][1];
            val = val * s + coeff[i][0];
            result[i] = val;
        }
        return result;
    }

    /// Velocity [m/s] at s; `inv_dt` = 1 / segment length [1/s].
    [[nodiscard]] Vector3d velocity(double s, double inv_dt) const noexcept {
        Vector3d result;
        for (int i = 0; i < 3; ++i) {
            // dP/ds = c1 + 2*c2*s + 3*c3*s² + 4*c4*s³ + 5*c5*s⁴
            double val = 5.0 * coeff[i][5];
            val = val * s + 4.0 * coeff[i][4];
            val = val * s + 3.0 * coeff[i][3];
            val = val * s + 2.0 * coeff[i][2];
            val = val * s + coeff[i][1];
            result[i] = val * inv_dt;
        }
        return result;
    }

    /// Acceleration [m/s²] at s; `inv_dt2` = 1 / (segment length)² [1/s²].
    [[nodiscard]] Vector3d acceleration(double s, double inv_dt2) const noexcept {
        Vector3d result;
        for (int i = 0; i < 3; ++i) {
            // d²P/ds² = 2*c2 + 6*c3*s + 12*c4*s² + 20*c5*s³
            double val = 20.0 * coeff[i][5];
            val = val * s + 12.0 * coeff[i][4];
            val = val * s + 6.0 * coeff[i][3];
            val = val * s + 2.0 * coeff[i][2];
            result[i] = val * inv_dt2;
        }
        return result;
    }
};

/// Polynomial coefficients of the segment of length `dt` [s] between (p0, v0, a0) and
/// (p1, v1, a1) (meters, m/s, m/s²).
//
// Converts the Hermite form to standard polynomial form for efficient
// repeated evaluation. The polynomial is:
//   P(s) = sum_{i=0}^{5} c_i * s^i
//
// where the coefficients are derived from:
//   P(s) = h00(s)*p0 + h10(s)*v0*dt + h20(s)*a0*dt²
//        + h01(s)*p1 + h11(s)*v1*dt + h21(s)*a1*dt²
[[nodiscard]] inline QuinticSegmentCoefficients compute_segment_coefficients(
    double dt,
    const Vector3d& p0, const Vector3d& v0, const Vector3d& a0,
    const Vector3d& p1, const Vector3d& v1, const Vector3d& a1) noexcept
{
    const double dt2 = dt * dt;

    // Scaled inputs
    Vector3d v0_dt = v0 * dt;
    Vector3d v1_dt = v1 * dt;
    Vector3d a0_dt2 = a0 * dt2;
    Vector3d a1_dt2 = a1 * dt2;

    QuinticSegmentCoefficients seg;

    for (int i = 0; i < 3; ++i) {
        // Collect contributions from basis functions:
        // h00(s) = 1 - 10s³ + 15s⁴ - 6s⁵     → contributes p0[i] to c0, -10*p0[i] to c3, etc.
        // h10(s) = s - 6s³ + 8s⁴ - 3s⁵       → contributes v0_dt[i] to c1, -6*v0_dt[i] to c3, etc.
        // h20(s) = 0.5s² - 1.5s³ + 1.5s⁴ - 0.5s⁵
        // h01(s) = 10s³ - 15s⁴ + 6s⁵
        // h11(s) = -4s³ + 7s⁴ - 3s⁵
        // h21(s) = 0.5s³ - s⁴ + 0.5s⁵

        // c0: constant term (s⁰)
        seg.coeff[i][0] = p0[i];

        // c1: linear term (s¹)
        seg.coeff[i][1] = v0_dt[i];

        // c2: quadratic term (s²)
        seg.coeff[i][2] = 0.5 * a0_dt2[i];

        // c3: cubic term (s³)
        seg.coeff[i][3] = -10.0 * p0[i] - 6.0 * v0_dt[i] - 1.5 * a0_dt2[i]
                        + 10.0 * p1[i] - 4.0 * v1_dt[i] + 0.5 * a1_dt2[i];

        // c4: quartic term (s⁴)
        seg.coeff[i][4] = 15.0 * p0[i] + 8.0 * v0_dt[i] + 1.5 * a0_dt2[i]
                        - 15.0 * p1[i] + 7.0 * v1_dt[i] - a1_dt2[i];

        // c5: quintic term (s⁵)
        seg.coeff[i][5] = -6.0 * p0[i] - 3.0 * v0_dt[i] - 0.5 * a0_dt2[i]
                        + 6.0 * p1[i] - 3.0 * v1_dt[i] + 0.5 * a1_dt2[i];
    }

    return seg;
}

} // namespace math
} // namespace ballistic
