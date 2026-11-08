#pragma once

// =============================================================================
// Zonal Harmonic Gravity Model (Point Mass + J2-J15)
// =============================================================================
//
// References:
// -----------
// [1] Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications",
//     4th Edition. Microcosm Press.
//     - Two-body equation: Section 1.4, Equation 1-6
//     - Zonal harmonics J2-J6: Section 8.7.1, Equations 8-25 through 8-30
//
// [2] NIMA (2000). "Department of Defense World Geodetic System 1984".
//     NIMA TR8350.2, 3rd Edition. Table 3.1 (the zonal formulas; the values
//     used, constants::earth::J2..J15, come from the embedded EGM2008 model
//     and are checked against it in test_constants.cpp):
//     - J2 = 1.08262668355e-3
//     - J3 = -2.53265648533e-6
//     - J4 = -1.61989759991e-6
//     - J5 = -2.27279221460e-7
//     - J6 = 5.40681239107e-7
//
// [3] Montenbruck, O. & Gill, E. (2000). "Satellite Orbits: Models, Methods,
//     and Applications". Springer-Verlag. Chapter 3.2.
//     - Legendre polynomial formulation
//
// Mathematical Background:
// ------------------------
// "Zonal" harmonics are the part of the Earth's gravity field that depends only
// on latitude (it is symmetric about the rotation axis). J2 describes the
// Earth's equatorial bulge and is by far the largest; J3..J15 are smaller
// corrections. The gravitational potential with zonal harmonics is:
//   U = (GM/r) * [1 - sum_{n=2}^{N} Jn * (Re/r)^n * Pn(sin(phi))]
// where GM is the Earth's gravitational parameter [m^3/s^2], Re its equatorial
// radius [m], Pn the Legendre polynomials and phi the geocentric latitude.
//
// With this sign convention (U positive) the acceleration is a = +grad(U),
// computed analytically below. Caution: gravitational_potential() further down
// uses the opposite convention (-GM/r).
//
// All functions here work in any Earth-centered frame whose z axis is the
// Earth's rotation axis (the zonal terms do not depend on longitude, so an
// Earth-fixed and an inertial frame give the same answer, up to the small tilt
// between the inertial z axis and the true rotation axis). Positions in meters,
// accelerations in m/s^2. A position at the origin gives inf/NaN.
//
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../core/defaults.hpp"
#include <cmath>

namespace ballistic {
namespace gravity {

/// Point-mass (two-body) gravitational acceleration a = -mu/r^3 * r_vec [m/s^2],
/// in the frame of `position` [m]. Reference: Vallado [1], Equation 1-6.
/// @param mu  gravitational parameter [m^3/s^2] (default: the Earth's)
[[nodiscard]] inline Vector3d point_mass_acceleration(
    const Vector3d& position,
    double mu = constants::earth::GM) noexcept
{
    const double r = position.norm();
    const double r3 = r * r * r;

    return -mu / r3 * position;
}

/// Acceleration from J2, the Earth's oblateness (equatorial bulge) [m/s^2].
/// Reference: Vallado [1], Equation 8-25; derived from
/// P2(sin(phi)) = (3*sin^2(phi) - 1)/2. The largest perturbation for Earth
/// satellites.
/// @param position  Earth-centered position with z along the rotation axis [m]
/// @param Re        reference (equatorial) radius [m]
[[nodiscard]] inline Vector3d j2_acceleration(
    const Vector3d& position,
    double mu = constants::earth::GM,
    double Re = constants::earth::RADIUS_EQUATORIAL,
    double J2 = constants::earth::J2) noexcept
{
    const double x = position.x();
    const double y = position.y();
    const double z = position.z();

    const double r = position.norm();
    const double r2 = r * r;
    const double r5 = r2 * r2 * r;
    const double Re2 = Re * Re;

    const double z2_r2 = (z * z) / r2;
    const double factor = 1.5 * J2 * mu * Re2 / r5;

    // J2 perturbation components
    const double ax = factor * x * (5.0 * z2_r2 - 1.0);
    const double ay = factor * y * (5.0 * z2_r2 - 1.0);
    const double az = factor * z * (5.0 * z2_r2 - 3.0);

    return Vector3d(ax, ay, az);
}

/// point_mass_acceleration() + j2_acceleration() [m/s^2].
[[nodiscard]] inline Vector3d gravity_with_j2(
    const Vector3d& position,
    double mu = constants::earth::GM,
    double Re = constants::earth::RADIUS_EQUATORIAL,
    double J2 = constants::earth::J2) noexcept
{
    return point_mass_acceleration(position, mu) + j2_acceleration(position, mu, Re, J2);
}

/// Acceleration [m/s^2] from one zonal term Jn, 2 <= n <= 15.
///
/// n = 2..6 use the closed forms of Vallado [1], Equations 8-25 to 8-30; n = 7..15
/// use the Legendre recurrence.
/// @param n   degree
/// @param Jn  the coefficient (e.g. constants::earth::J_COEFFS[n])
/// @throws ConfigError if n is outside 2..15 (as zonal_harmonics_acceleration).
///
/// Derivation notes:
/// Each Jn term is derived from the Legendre polynomial Pn(sin(phi)) where
/// phi = geocentric latitude = asin(z/r). The acceleration components are:
///   ax = dU/dx, ay = dU/dy, az = dU/dz
/// where U is the gravitational potential perturbation.
///
/// The polynomial coefficients (e.g., 5, 7, 3, 35 in J3) arise from:
///   P3(x) = (5x^3 - 3x)/2
///   P4(x) = (35x^4 - 30x^2 + 3)/8
///   P5(x) = (63x^5 - 70x^3 + 15x)/8
///   P6(x) = (231x^6 - 315x^4 + 105x^2 - 5)/16
/// and their derivatives with respect to Cartesian coordinates.
[[nodiscard]] inline Vector3d jn_acceleration(
    const Vector3d& position,
    int n,
    double Jn,
    double mu = constants::earth::GM,
    double Re = constants::earth::RADIUS_EQUATORIAL)
{
    // Only J2..J15 have a formula here; any other degree raises rather than
    // returning a zero acceleration that looks like a valid (tiny) term.
    if (n < 2 || n > 15) {
        throw ConfigError("jn_acceleration: degree " + std::to_string(n) +
                          " requested, but only zonal degrees 2..15 are implemented; use "
                          "EGM2008 (gravity::EGM2008Gravity) for higher degrees");
    }

    const double x = position.x();
    const double y = position.y();
    const double z = position.z();

    const double r = position.norm();
    const double r2 = r * r;
    const double z_r = z / r;

    // Common factor
    const double Re_r = Re / r;
    double Re_r_n = std::pow(Re_r, n);

    Vector3d acc = Vector3d::Zero();

    switch (n) {
        case 2: // J2 - use optimized version above
            return j2_acceleration(position, mu, Re, Jn);

        case 3: {
            // J3 perturbation
            const double factor = 0.5 * Jn * mu * Re_r_n / r2;
            const double z_r2 = z_r * z_r;

            acc.x() = factor * (5.0 * x / r) * (7.0 * z_r2 * z_r - 3.0 * z_r);
            acc.y() = factor * (5.0 * y / r) * (7.0 * z_r2 * z_r - 3.0 * z_r);
            acc.z() = factor * (3.0 - 30.0 * z_r2 + 35.0 * z_r2 * z_r2);
            break;
        }

        case 4: {
            // J4 perturbation
            const double factor = 0.625 * Jn * mu * Re_r_n / r2;
            const double z_r2 = z_r * z_r;
            const double z_r4 = z_r2 * z_r2;

            const double term_xy = 3.0 - 42.0 * z_r2 + 63.0 * z_r4;
            const double term_z = 15.0 - 70.0 * z_r2 + 63.0 * z_r4;

            acc.x() = factor * (x / r) * term_xy;
            acc.y() = factor * (y / r) * term_xy;
            acc.z() = factor * z_r * term_z;
            break;
        }

        case 5: {
            // J5 perturbation
            // Reference: Vallado [1], Equation 8-29
            // Derived from P5(x) = (63x^5 - 70x^3 + 15x)/8
            // Factor 0.125 = 1/8 from Legendre polynomial normalization
            // Coefficients verified against Montenbruck & Gill [3], Table 3.1
            const double factor = -0.125 * Jn * mu * Re_r_n / r2;
            const double z_r2 = z_r * z_r;
            const double z_r4 = z_r2 * z_r2;
            const double z_r6 = z_r4 * z_r2;

            // Polynomial coefficients from Legendre derivative expansion
            // P5(s) = (63s^5 - 70s^3 + 15s)/8
            // 8 * [(n+1)*Pn + s*Pn'] = 693s^5 - 630s^3 + 105s enters a_x/(x/r)
            // with +1/8, the opposite sign of `factor`, hence the minus here
            // (a_z's 8*[(n+1)*s*Pn - (1-s^2)*Pn'] = -term_z takes factor's sign).
            const double term_xy = -z_r * (693.0 * z_r4 - 630.0 * z_r2 + 105.0);
            const double term_z = 15.0 - 315.0 * z_r2 + 945.0 * z_r4 - 693.0 * z_r6;

            acc.x() = factor * (x / r) * term_xy;
            acc.y() = factor * (y / r) * term_xy;
            acc.z() = factor * term_z;
            break;
        }

        case 6: {
            // J6 perturbation
            // Reference: Vallado [1], Equation 8-30
            // Derived from P6(x) = (231x^6 - 315x^4 + 105x^2 - 5)/16
            // Factor -0.0625 = -1/16 from Legendre polynomial normalization
            // Negative sign from potential derivative
            const double factor = -0.0625 * Jn * mu * Re_r_n / r2;
            const double z_r2 = z_r * z_r;
            const double z_r4 = z_r2 * z_r2;
            const double z_r6 = z_r4 * z_r2;

            // Polynomial coefficients from Legendre derivative expansion
            // 3465 = 3*5*7*33, 3003 = 3*7*11*13, 945 = 3^3*5*7
            const double term_xy = 35.0 - 945.0 * z_r2 + 3465.0 * z_r4 - 3003.0 * z_r6;
            const double term_z = 245.0 - 2205.0 * z_r2 + 4851.0 * z_r4 - 3003.0 * z_r6;

            acc.x() = factor * (x / r) * term_xy;
            acc.y() = factor * (y / r) * term_xy;
            acc.z() = factor * z_r * term_z;
            break;
        }

        default: {
            // Generic implementation for J7-J15 using Legendre polynomial recursion
            // Reference: Montenbruck & Gill [3], Section 3.2
            // Pn(x) computed recursively: (n+1)*P_{n+1} = (2n+1)*x*Pn - n*P_{n-1}
            // dPn/dx computed recursively: dP_{n+1}/dx = x*dPn/dx + (n+1)*Pn
            // Compute Legendre polynomial Pn and its derivative dPn at sin(lat) = z/r
            double P_nm2 = 1.0;      // P_0
            double P_nm1 = z_r;      // P_1
            double dP_nm1 = 1.0;     // dP_1/dx

            double Pn = 0.0;
            double dPn = 0.0;

            for (int k = 2; k <= n; ++k) {
                // Legendre recursion
                Pn = ((2.0 * k - 1.0) * z_r * P_nm1 - (k - 1.0) * P_nm2) / k;
                dPn = z_r * dP_nm1 + k * P_nm1;

                P_nm2 = P_nm1;
                P_nm1 = Pn;
                dP_nm1 = dPn;
            }

            // Acceleration from Jn term
            // U_n = -(GM/r) * Jn * (Re/r)^n * Pn(sin(lat))
            // a = +grad(U_n) computed via direct Cartesian derivatives:
            //   ax = factor * (x/r) * [(n+1)*Pn + s*dPn]
            //   az = factor * [(n+1)*s*Pn - (1-s²)*dPn]
            const double factor = Jn * mu * Re_r_n / r2;
            const double bracket_xy = (n + 1.0) * Pn + z_r * dPn;
            const double cos2_lat = 1.0 - z_r * z_r;

            acc.x() = factor * (x / r) * bracket_xy;
            acc.y() = factor * (y / r) * bracket_xy;
            acc.z() = factor * ((n + 1.0) * z_r * Pn - cos2_lat * dPn);
            break;
        }
    }

    return acc;
}

/// Sum of the zonal accelerations J2..J{max_degree} [m/s^2], without the
/// point-mass term. Reference: Vallado [1], Section 8.7.1; Montenbruck & Gill
/// [3], Chapter 3.2.
/// @param max_degree  highest zonal degree (default 6, at most 15); below 2
///                    the result is zero
/// @throws ConfigError if max_degree > 15 (use gravity::EGM2008Gravity).
[[nodiscard]] inline Vector3d zonal_harmonics_acceleration(
    const Vector3d& position,
    double mu = constants::earth::GM,
    double Re = constants::earth::RADIUS_EQUATORIAL,
    int max_degree = 6)
{
    using namespace constants::earth;

    Vector3d acc = Vector3d::Zero();

    // Return zero for invalid degree (J0, J1 are not physical perturbations)
    if (max_degree < 2) {
        return acc;
    }

    // constants::earth holds J2..J15; a higher degree raises (never clamped).
    if (max_degree > 15) {
        throw ConfigError("zonal_harmonics_acceleration: degree " + std::to_string(max_degree) +
                          " requested, but only J2..J15 are tabulated; use EGM2008 "
                          "(gravity::EGM2008Gravity) for higher degrees");
    }

    // J2 has optimized implementation
    if (max_degree >= 2) {
        acc += j2_acceleration(position, mu, Re, J2);
    }

    // J3-J15 use general implementation
    for (int n = 3; n <= max_degree; ++n) {
        acc += jn_acceleration(position, n, J_COEFFS[n], mu, Re);
    }

    return acc;
}

/// point_mass_acceleration() + zonal_harmonics_acceleration() [m/s^2].
/// @throws ConfigError if max_degree > 15.
[[nodiscard]] inline Vector3d gravity_with_zonal(
    const Vector3d& position,
    double mu = constants::earth::GM,
    double Re = constants::earth::RADIUS_EQUATORIAL,
    int max_degree = 6)
{
    return point_mass_acceleration(position, mu)
         + zonal_harmonics_acceleration(position, mu, Re, max_degree);
}

/// Point-mass potential -mu/r [m^2/s^2] (physics convention: negative,
/// acceleration = -grad). Used for energy checks.
[[nodiscard]] inline double gravitational_potential(
    const Vector3d& position,
    double mu = constants::earth::GM) noexcept
{
    return -mu / position.norm();
}

/// J2 term of the potential [m^2/s^2]: -J2 * mu * Re^2 / r^3 * P2(sin(phi)).
/// This is the J2 term of U in the header (positive convention, a = +grad(U)),
/// so its sign convention is the OPPOSITE of gravitational_potential().
[[nodiscard]] inline double j2_potential(
    const Vector3d& position,
    double mu = constants::earth::GM,
    double Re = constants::earth::RADIUS_EQUATORIAL,
    double J2 = constants::earth::J2) noexcept
{
    const double r = position.norm();
    const double z = position.z();
    const double sin_lat = z / r;  // sin(geocentric latitude)

    // Legendre polynomial P2(sin_lat) = (3*sin^2(lat) - 1) / 2
    const double P2 = 0.5 * (3.0 * sin_lat * sin_lat - 1.0);

    return -J2 * mu * Re * Re / (r * r * r) * P2;
}

/// Point-mass + J2 potential U = (mu/r) * [1 - J2 (Re/r)^2 P2(sin(phi))] [m^2/s^2], in the
/// header's convention (U positive, a = +grad(U) = point_mass_acceleration() +
/// j2_acceleration()). Note the sign: this is -gravitational_potential() + j2_potential().
[[nodiscard]] inline double total_potential_with_j2(
    const Vector3d& position,
    double mu = constants::earth::GM,
    double Re = constants::earth::RADIUS_EQUATORIAL,
    double J2 = constants::earth::J2) noexcept
{
    return mu / position.norm() + j2_potential(position, mu, Re, J2);
}

/// Two-body orbital energy per unit mass, v^2/2 - mu/r [m^2/s^2] (constant
/// along a pure two-body orbit; negative for a bound orbit). Position [m] and
/// velocity [m/s] must be in the same inertial frame.
[[nodiscard]] inline double specific_orbital_energy(
    const Vector3d& position,
    const Vector3d& velocity,
    double mu = constants::earth::GM) noexcept
{
    const double v2 = velocity.squaredNorm();
    const double r = position.norm();

    return 0.5 * v2 - mu / r;
}

/// Angular momentum per unit mass, r x v [m^2/s].
[[nodiscard]] inline Vector3d specific_angular_momentum(
    const Vector3d& position,
    const Vector3d& velocity) noexcept
{
    return position.cross(velocity);
}

} // namespace gravity
} // namespace ballistic
