#pragma once

// =============================================================================
// Relativistic Corrections for Orbit Propagation
// =============================================================================
//
// Small corrections from general relativity to Newtonian gravity
// ("post-Newtonian": terms of order 1/c², c = speed of light). At LEO they move
// the predicted position by about 0.1 m per hour, so they matter for
// sub-meter accuracy.
//
// All functions take Earth-centered inertial (GCRS) position (m) and velocity
// (m/s) and return accelerations in m/s² in the same frame.
//
// References:
// -----------
// [1] IERS Conventions (2010), Chapter 10: General relativistic models
//     https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html
//
// [2] Montenbruck, O. & Gill, E. (2000). "Satellite Orbits: Models, Methods,
//     and Applications". Springer. Section 3.7: Relativistic Effects.
//
// [3] Petit, G. & Luzum, B. (2010). "IERS Conventions (2010)", Section 10.3:
//     "Equations of motion for an artificial Earth satellite"
//
// Corrections:
// ------------
// 1. Schwarzschild term (the main 1/c² correction to Earth's gravity)
//    - Acceleration at LEO: ~1.5e-8 m/s²
//    - Position effect: ~0.1 m over 1 hour at LEO
//    - Ratio to gravity: ~2e-9
//
// 2. Lense-Thirring term (frame dragging: the spinning Earth drags spacetime
//    around with it) [optional, off by default]
//    - Acceleration at LEO: ~2e-10 m/s²
//    - Position effect: ~0.001 m over 1 hour at LEO
//    - Uses the Earth's angular momentum
//
// 3. De Sitter term (geodetic precession, from the Earth-satellite system
//    moving through the Sun's gravity field) [NOT implemented]
//    - Negligible for short-term propagation
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../time/epoch.hpp"
#include <cmath>

namespace ballistic {
namespace forces {

/// Speed of light squared (m²/s²).
constexpr double C_SQUARED = constants::SPEED_OF_LIGHT * constants::SPEED_OF_LIGHT;

/// Earth's gravitational radius GM/c² ≈ 4.435 mm (the length scale of the
/// relativistic terms).
constexpr double EARTH_GRAVITATIONAL_RADIUS = constants::earth::GM / C_SQUARED;

// =============================================================================
// Schwarzschild Correction
// =============================================================================
//
// The largest relativistic correction for Earth satellites: the effect of
// spacetime curvature near the Earth.
//
// Formula (IERS Conventions 2010, Eq. 10.12):
// a_Sch = (GM/c²r³) * { (4GM/r - v²)r + 4(r·v)v }
//
// Magnitude at LEO (~7000 km):
// - Acceleration: ~1.5e-8 m/s²
// - Position effect over 1 hour: ~0.1 m (estimate 0.5·a·t²)
//
// =============================================================================

/// Schwarzschild acceleration (m/s², GCRS) for a GCRS position (m) and
/// velocity (m/s).
[[nodiscard]] inline Vector3d schwarzschild_acceleration(
    const Vector3d& position,
    const Vector3d& velocity) noexcept
{
    double r = position.norm();
    double r2 = r * r;
    double r3 = r2 * r;
    double v2 = velocity.squaredNorm();
    double rdotv = position.dot(velocity);

    // GM/c² ≈ 4.435e-3 m for Earth
    double coeff = constants::earth::GM / (C_SQUARED * r3);

    // Schwarzschild acceleration components
    // a = coeff * [(4GM/r - v²)r + 4(r·v)v]
    double term1 = 4.0 * constants::earth::GM / r - v2;
    double term2 = 4.0 * rdotv;

    return coeff * (term1 * position + term2 * velocity);
}

/// Schwarzschild acceleration AND its analytic 3×6 Jacobian [∂a/∂r | ∂a/∂v]
/// (the partial derivatives with respect to position and velocity).
///
/// Derivation (split a_Sch = a_A + a_B + a_C, µα ≡ GM/c²):
///   a_A = 4µ²α · r/r⁴               (radial scale)
///   a_B = -µα · v² · r/r³           (kinetic-energy scale)
///   a_C = 4µα · (r·v) · v/r³        (velocity-coupled)
///
/// Using ∂(r/rⁿ)/∂r = I/rⁿ - n·r·rᵀ/r^(n+2):
///
///   ∂a_Sch/∂r = 4µ²α·[I/r⁴ - 4·rrᵀ/r⁶]
///             - µα·v²·[I/r³ - 3·rrᵀ/r⁵]
///             + 4µα·[vvᵀ/r³ - 3·(r·v)·vrᵀ/r⁵]
///
///   ∂a_Sch/∂v = -2µα/r³·rvᵀ + 4µα/r³·[vrᵀ + (r·v)·I]
///
/// The position block is not symmetric in general, because the acceleration
/// depends on velocity.
[[nodiscard]] inline std::pair<Vector3d, Eigen::Matrix<double, 3, 6>>
schwarzschild_acceleration_with_jacobian(
    const Vector3d& position,
    const Vector3d& velocity) noexcept
{
    using Mat3 = Eigen::Matrix3d;
    using Mat36 = Eigen::Matrix<double, 3, 6>;

    const double mu = constants::earth::GM;
    const double mualpha = mu / C_SQUARED;        // µα = GM/c²
    const double r = position.norm();
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r4 = r2 * r2;
    const double r5 = r3 * r2;
    const double r6 = r3 * r3;
    const double v2 = velocity.squaredNorm();
    const double rdotv = position.dot(velocity);

    // Acceleration (matches schwarzschild_acceleration above).
    const double term1 = 4.0 * mu / r - v2;
    const double term2 = 4.0 * rdotv;
    const Vector3d a = (mu / (C_SQUARED * r3)) * (term1 * position + term2 * velocity);

    // Outer products needed multiple times below.
    const Mat3 rrT = position * position.transpose();
    const Mat3 vvT = velocity * velocity.transpose();
    const Mat3 vrT = velocity * position.transpose();
    const Mat3 rvT = position * velocity.transpose();
    const Mat3 I = Mat3::Identity();

    // ∂a_A/∂r = 4µ²α · [I/r⁴ - 4·rrᵀ/r⁶]
    const Mat3 dA_dr = 4.0 * mualpha * mu * (I / r4 - 4.0 * rrT / r6);

    // ∂a_B/∂r = -µα·v²·[I/r³ - 3·rrᵀ/r⁵]
    const Mat3 dB_dr = -mualpha * v2 * (I / r3 - 3.0 * rrT / r5);

    // ∂a_C/∂r = 4µα·[vvᵀ/r³ - 3·(r·v)·vrᵀ/r⁵]
    const Mat3 dC_dr = 4.0 * mualpha * (vvT / r3 - 3.0 * rdotv * vrT / r5);

    // ∂a_B/∂v = -2µα/r³·rvᵀ
    const Mat3 dB_dv = -2.0 * mualpha / r3 * rvT;

    // ∂a_C/∂v = 4µα/r³·[vrᵀ + (r·v)·I]
    const Mat3 dC_dv = 4.0 * mualpha / r3 * (vrT + rdotv * I);

    Mat36 J;
    J.block<3, 3>(0, 0) = dA_dr + dB_dr + dC_dr;
    J.block<3, 3>(0, 3) = dB_dv + dC_dv;          // dA_dv = 0
    return {a, J};
}

// =============================================================================
// Lense-Thirring (Frame-Dragging) Correction
// =============================================================================
//
// Caused by the Earth's rotation dragging spacetime around with it. Much
// smaller than Schwarzschild.
//
// Formula (IERS Conventions 2010, Eq. 10.13):
// a_LT = (2GM/c²r³) * { (3/r²)(r × v)(r · J) + (v × J) }
//
// where J is the Earth's angular momentum per unit mass (m²/s).
//
// Magnitude at LEO:
// - Acceleration: ~2e-10 m/s²
// - Position effect over 1 hour: ~0.001 m
//
// =============================================================================

/// Earth's angular momentum per unit mass (J/M) in m²/s:
/// J/M = C/(MR²) * R² * ω, where C/(MR²) = 0.3307 is the Earth's
/// moment-of-inertia factor (IERS TN36 Table 1.1), R the equatorial radius
/// and ω the rotation rate: J/M ≈ 0.3307 * (6.378137e6)² * 7.292115e-5 ≈ 9.8e8 m²/s.
/// A uniform sphere would have 0.4 = 2/5; the Earth's denser core makes it
/// 0.3307 (a ~21% difference).
constexpr double EARTH_J_PER_M = 0.3307 * constants::earth::RADIUS_EQUATORIAL *
                                  constants::earth::RADIUS_EQUATORIAL *
                                  constants::earth::ANGULAR_VELOCITY;

/// Lense-Thirring acceleration (m/s², GCRS) for a GCRS position (m) and
/// velocity (m/s). Assumes the Earth's spin axis is the GCRS z axis.
[[nodiscard]] inline Vector3d lense_thirring_acceleration(
    const Vector3d& position,
    const Vector3d& velocity) noexcept
{
    double r = position.norm();
    double r2 = r * r;
    double r3 = r2 * r;
    double r5 = r3 * r2;

    // Earth's angular momentum direction: the rotation axis, taken as GCRS z
    Vector3d J_hat(0, 0, 1);
    Vector3d J = EARTH_J_PER_M * J_hat;

    // r × v
    Vector3d rxv = position.cross(velocity);

    // r · J
    double rdotJ = position.dot(J);

    // v × J
    Vector3d vxJ = velocity.cross(J);

    double coeff = 2.0 * constants::earth::GM / (C_SQUARED * r3);

    // a_LT = coeff * { (3/r²)(r × v)(r · J) + (v × J) }
    return coeff * ((3.0 / r2) * rdotJ * rxv + vxJ);
}

/// Cross-product matrix: skew(a)·u = a × u.
[[nodiscard]] inline Eigen::Matrix3d skew_symmetric_(const Vector3d& a) noexcept {
    Eigen::Matrix3d S;
    S <<  0,    -a(2),  a(1),
          a(2),  0,    -a(0),
         -a(1),  a(0),  0;
    return S;
}

/// Lense-Thirring acceleration AND its analytic 3×6 Jacobian [∂a/∂r | ∂a/∂v].
///
/// Derivation (a_LT = a_D + a_E, where µα ≡ GM/c²):
///   a_D = (6µα/r⁵)·(r·J)·(r×v)
///   a_E = (2µα/r³)·(v×J)
///
/// Using:
///   ∂(r×v)/∂r = -skew(v)        ((r×v) = -(v×r))
///   ∂(r×v)/∂v = +skew(r)
///   ∂(v×J)/∂v = -skew(J)
///   ∂(r·J)/∂r = J,  ∂(r⁻ⁿ)/∂r = -n·r/r^(n+2)
///
///   ∂a_D/∂r = 6µα · [(-5/r⁷)(r·J)(r×v)·rᵀ + (1/r⁵)(r×v)·Jᵀ - ((r·J)/r⁵)·skew(v)]
///   ∂a_D/∂v = (6µα·(r·J)/r⁵) · skew(r)
///   ∂a_E/∂r = -(6µα/r⁵) · (v×J)·rᵀ
///   ∂a_E/∂v = -(2µα/r³) · skew(J)
///
/// Checked by hand on (r,v,J) = ((R,0,0), (0,V,0), (0,0,Jz)):
///   diag(∂a_LT/∂r) = (-6µαVJ_z/R⁴, 0, +6µαVJ_z/R⁴), which matches finite differences.
[[nodiscard]] inline std::pair<Vector3d, Eigen::Matrix<double, 3, 6>>
lense_thirring_acceleration_with_jacobian(
    const Vector3d& position,
    const Vector3d& velocity) noexcept
{
    using Mat3 = Eigen::Matrix3d;
    using Mat36 = Eigen::Matrix<double, 3, 6>;

    const double mu = constants::earth::GM;
    const double mualpha = mu / C_SQUARED;
    const double r = position.norm();
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double r5 = r3 * r2;
    const double r7 = r5 * r2;

    const Vector3d J_vec(0.0, 0.0, EARTH_J_PER_M);
    const Vector3d rxv = position.cross(velocity);
    const Vector3d vxJ = velocity.cross(J_vec);
    const double rdotJ = position.dot(J_vec);

    // Acceleration (matches lense_thirring_acceleration above).
    const Vector3d a = (2.0 * mualpha / r3) * ((3.0 / r2) * rdotJ * rxv + vxJ);

    // ∂a_D/∂r = 6µα·[(-5/r⁷)(r·J)(r×v)·rᵀ + (1/r⁵)(r×v)·Jᵀ - ((r·J)/r⁵)·skew(v)]
    const Mat3 dD_dr =
          6.0 * mualpha * (-5.0 / r7) * rdotJ * (rxv * position.transpose())
        + 6.0 * mualpha * (1.0 / r5) * (rxv * J_vec.transpose())
        - 6.0 * mualpha * (rdotJ / r5) * skew_symmetric_(velocity);

    // ∂a_D/∂v = (6µα·(r·J)/r⁵) · skew(r)
    const Mat3 dD_dv = 6.0 * mualpha * (rdotJ / r5) * skew_symmetric_(position);

    // ∂a_E/∂r = -(6µα/r⁵) · (v×J)·rᵀ
    const Mat3 dE_dr = -6.0 * mualpha / r5 * (vxJ * position.transpose());

    // ∂a_E/∂v = -(2µα/r³) · skew(J)
    const Mat3 dE_dv = -(2.0 * mualpha / r3) * skew_symmetric_(J_vec);

    Mat36 J;
    J.block<3, 3>(0, 0) = dD_dr + dE_dr;
    J.block<3, 3>(0, 3) = dD_dv + dE_dv;
    return {a, J};
}

// =============================================================================
// Combined Relativistic Acceleration
// =============================================================================

/// Which relativistic corrections are on.
struct RelativisticConfig {
    bool enable_schwarzschild = true;   // Main term (~0.1 m in 1 h at LEO)
    bool enable_lense_thirring = false; // Frame dragging (~0.001 m in 1 h at LEO)
    bool enable_de_sitter = false;      // Geodetic precession (negligible; not implemented)
};

/// Sum of the enabled relativistic corrections (Schwarzschild on, Lense-Thirring
/// off by default).
class RelativisticCorrections {
public:
    RelativisticCorrections() = default;

    /// Starts from the given set of switches.
    explicit RelativisticCorrections(const RelativisticConfig& config)
        : config_(config) {}

    /// The current switches.
    [[nodiscard]] const RelativisticConfig& config() const noexcept {
        return config_;
    }

    /// Switch each correction on or off.
    void enable_schwarzschild(bool enable) noexcept {
        config_.enable_schwarzschild = enable;
    }

    void enable_lense_thirring(bool enable) noexcept {
        config_.enable_lense_thirring = enable;
    }

    /// Total relativistic acceleration (m/s², GCRS) for a GCRS state. The epoch
    /// is unused (the terms depend only on the state).
    [[nodiscard]] Vector3d acceleration(
        const Epoch& /* epoch */,
        const StateVector& state) const noexcept
    {
        Vector3d acc = Vector3d::Zero();

        if (config_.enable_schwarzschild) {
            acc += schwarzschild_acceleration(state.position, state.velocity);
        }

        if (config_.enable_lense_thirring) {
            acc += lense_thirring_acceleration(state.position, state.velocity);
        }

        // The de Sitter term is negligible for Earth satellites over short
        // periods and needs the Sun's position, so it is not implemented.

        return acc;
    }

    /// True when only Schwarzschild is on: it is treated as conservative
    /// (derivable from a potential). Lense-Thirring depends on velocity through
    /// v × J and is treated as non-conservative.
    [[nodiscard]] bool is_conservative() const noexcept {
        return !config_.enable_lense_thirring;
    }

    /// Total relativistic acceleration AND its 3×6 Jacobian [∂a/∂r | ∂a/∂v],
    /// summing the enabled corrections. The derivations are with the free
    /// functions above.
    [[nodiscard]] std::pair<Vector3d, Eigen::Matrix<double, 3, 6>>
    acceleration_with_jacobian(
        const Epoch& /* epoch */,
        const StateVector& state) const noexcept
    {
        Vector3d a_total = Vector3d::Zero();
        Eigen::Matrix<double, 3, 6> J_total =
            Eigen::Matrix<double, 3, 6>::Zero();

        if (config_.enable_schwarzschild) {
            auto [a, J] = schwarzschild_acceleration_with_jacobian(
                state.position, state.velocity);
            a_total += a;
            J_total += J;
        }
        if (config_.enable_lense_thirring) {
            auto [a, J] = lense_thirring_acceleration_with_jacobian(
                state.position, state.velocity);
            a_total += a;
            J_total += J;
        }
        return {a_total, J_total};
    }

private:
    RelativisticConfig config_;
};

// =============================================================================
// Utility: rough size of the relativistic effect
// =============================================================================

/// Schwarzschild acceleration magnitude (m/s²) on a circular orbit at
/// `altitude_m` above the equatorial radius:
/// |a_Sch| = 3*GM²/(c²*r³)  [IERS 2010 Eq. 10.12 with v² = GM/r, r·v = 0].
[[nodiscard]] inline double estimate_schwarzschild_magnitude(double altitude_m) noexcept {
    double r = constants::earth::RADIUS_EQUATORIAL + altitude_m;
    return 3.0 * constants::earth::GM * constants::earth::GM / (C_SQUARED * r * r * r);
}

/// Rough position error (m) after `duration_s` seconds from leaving out the
/// Schwarzschild correction: 0.5 · a · t².
[[nodiscard]] inline double estimate_schwarzschild_error(
    double altitude_m,
    double duration_s) noexcept
{
    double a = estimate_schwarzschild_magnitude(altitude_m);
    // Position error ≈ 0.5 * a * t²
    return 0.5 * a * duration_s * duration_s;
}

} // namespace forces
} // namespace ballistic
