#pragma once

// =============================================================================
// Earth rotation: sidereal time (GMST/GAST), polar motion, ITRF realizations
// =============================================================================
//
// Building blocks for converting between the inertial frame (GCRS) and the
// Earth-fixed frame (ITRF). The full conversion every propagation uses lives
// in iau_transforms.hpp; this header defines no frame transform itself.
//
// Terms used below:
// - Sidereal time: the Earth's rotation angle measured against the stars
//   rather than the Sun. GMST (Greenwich mean sidereal time) uses the mean,
//   precessing equinox; GAST (Greenwich apparent sidereal time) adds the
//   nutation of the equinox ("equation of the equinoxes"). Both are angles in
//   radians here.
// - UT1: the time scale that follows the Earth's actual rotation; TT
//   (Terrestrial Time): the uniform time scale of the ephemerides.
// - Polar motion: the wander of the Earth's spin axis over its surface,
//   given as pole coordinates xp, yp.
// - ITRF realization: successive editions of the Earth-fixed frame (ITRF2005,
//   ITRF2014, ITRF2020, ...), which differ by millimeters to centimeters.
//
// References:
// -----------
// [1] Petit, G. & Luzum, B. (2010). "IERS Conventions (2010)".
//     IERS Technical Note No. 36.
//     - ERA: Equation 5.15
//     - GMST: Equation 5.32
//
// [2] US Naval Observatory Circular No. 163 (1981). Kaplan, G.H.
//     "The IAU Resolutions on Astronomical Reference Systems, Time Scales,
//      and Earth Rotation Models".
//
// [3] Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications",
//     4th Edition. Microcosm Press. Section 3.7.
//
// Notes:
// ------
// - This uses the classical equinox-based (GMST/GAST) approach.
// - For the modern IERS method (Earth rotation angle measured from the
//   Celestial Intermediate Origin, CIO), use iau_transforms.hpp.
// - gast() uses Meeus's 4-term nutation (~0.5 arcsec); gast_full() the 678 luni-solar
//   terms of IAU 2000A (the full IAU 2006/2000A series has 1365 terms).
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../time/epoch.hpp"
#include "../time/eop_data.hpp"
#include "../util/time_utils.hpp"
#include "nutation_coefficients.hpp"
#include "itrf_transforms.hpp"
#include <cmath>

namespace ballistic {
namespace coords {

namespace detail {
    // Nutation in longitude dpsi (radians) from the 678 luni-solar IAU 2000A
    // terms, at t_tt Julian centuries of TT since J2000.0 (linear fundamental
    // arguments only). Used by gast_full().
    [[nodiscard]] inline double compute_nutation_dpsi(double t_tt) noexcept {
        using namespace constants;
        using namespace nutation;

        double dpsi = 0.0;

        // Fundamental arguments (Delaunay parameters)
        // l = mean anomaly of Moon
        double l = DEG_TO_RAD * std::fmod(485868.249036 + 1717915923.2178 * t_tt, 1296000.0) / 3600.0;
        // l' = mean anomaly of Sun
        double lp = DEG_TO_RAD * std::fmod(1287104.793048 + 129596581.0481 * t_tt, 1296000.0) / 3600.0;
        // F = Moon's argument of latitude
        double F = DEG_TO_RAD * std::fmod(335779.526232 + 1739527262.8478 * t_tt, 1296000.0) / 3600.0;
        // D = Mean elongation of Moon from Sun
        double D = DEG_TO_RAD * std::fmod(1072260.703692 + 1602961601.2090 * t_tt, 1296000.0) / 3600.0;
        // Om = Longitude of Moon's ascending node
        double Om = DEG_TO_RAD * std::fmod(450160.398036 - 6962890.5431 * t_tt, 1296000.0) / 3600.0;

        // Luni-solar nutation (major contribution)
        for (size_t i = 0; i < LUNISOLAR_TERMS_COUNT; ++i) {
            const auto& term = LUNISOLAR_TERMS[i];

            if (term.S == 0.0 && term.Sdot == 0.0 && term.Cp == 0.0) continue;

            double arg = term.nl * l + term.nlp * lp + term.nF * F
                       + term.nD * D + term.nOm * Om;

            double sin_arg = std::sin(arg);
            double cos_arg = std::cos(arg);

            // Nutation in longitude only
            dpsi += (term.S + term.Sdot * t_tt) * sin_arg + term.Cp * cos_arg;
        }

        // Convert from 0.1 microarcseconds to radians
        return dpsi * 0.1e-6 * ARCSEC_TO_RAD;
    }
} // namespace detail

/// Greenwich mean sidereal time (GMST, radians in [0, 2*pi)) at UT1 Julian date
/// `ut1_jd` (days). GMST = Earth rotation angle (ERA) + the accumulated
/// precession in right ascension, with IAU 2006 coefficients.
/// Reference: IERS TN 36 [1], Equation 5.32. The IERS formula evaluates the
/// polynomial in TT; this uses UT1 for both (they differ by about a minute,
/// a negligible change in the polynomial).
[[nodiscard]] inline double gmst(double ut1_jd) noexcept {
    using namespace constants;

    // Julian centuries of UT1 from J2000.0
    const double Tu = (ut1_jd - J2000_JD) / 36525.0;

    // Earth Rotation Angle (ERA) - IERS TN 36, Equation 5.15
    // Coefficients from IERS Conventions 2010:
    //   0.7790572732640 = fractional rotation at J2000.0
    //   1.00273781191135448 = sidereal/solar day ratio
    const double Du = ut1_jd - J2000_JD;
    const double theta = TWO_PI * (0.7790572732640 + 1.00273781191135448 * Du);

    // GMST = ERA + polynomial (accumulated precession in right ascension)
    // Coefficients from IERS TN 36 [1], Equation 5.32; ERFA gmst06()
    // Units: arcseconds per century^n
    const double Tu2 = Tu * Tu;
    const double gmst_rad = theta
        + ARCSEC_TO_RAD * (0.014506                  // Constant offset (arcsec)
        + 4612.156534 * Tu                            // Linear term
        + 1.3915817 * Tu2                             // Quadratic term
        - 0.00000044 * Tu2 * Tu                       // Cubic term
        - 0.000029956 * Tu2 * Tu2                     // Quartic term (ERFA)
        - 0.0000000368 * Tu2 * Tu2 * Tu);             // Quintic term (ERFA)

    // Normalize to [0, 2π)
    double result = std::fmod(gmst_rad, TWO_PI);
    if (result < 0.0) result += TWO_PI;

    return result;
}

/// Greenwich apparent sidereal time (GAST, radians; not wrapped to [0, 2*pi))
/// at UT1 Julian date `ut1_jd` and TT Julian date `tt_jd` (days).
/// GAST = GMST + equation of the equinoxes, where the equation of the
/// equinoxes = dpsi * cos(eps) (nutation in longitude times the cosine of the
/// mean tilt of the equator to the ecliptic).
/// Reference: Meeus, "Astronomical Algorithms", 2nd ed., Chapter 12
/// Reference: Vallado [3], Section 3.7.2
///
/// This uses Meeus's low-accuracy 4-term nutation in longitude (good to about
/// 0.5 arcsec). The full IAU 2006/2000A model has 1365 terms. For high
/// precision, use gast_full() or iau_transforms.hpp.
[[nodiscard]] inline double gast(double ut1_jd, double tt_jd) noexcept {
    // Julian centuries of TT from J2000.0
    const double T = (tt_jd - constants::J2000_JD) / 36525.0;

    // Mean obliquity of the ecliptic (IAU 2006)
    // Reference: IERS TN 36, Equation 5.40
    // Value at J2000.0: 84381.406 arcseconds = 23°26'21.406"
    const double eps0 = constants::ARCSEC_TO_RAD * (84381.406
        - 46.836769 * T
        - 0.0001831 * T * T
        + 0.00200340 * T * T * T);

    // Mean longitude of the Moon's ascending node
    // Reference: Meeus, Chapter 22 (simplified expression)
    const double Omega = constants::DEG_TO_RAD * (125.04452
        - 1934.136261 * T);

    // Mean longitudes of the Sun (L) and of the Moon (L')
    // Reference: Meeus, Chapter 22 (the expressions given with Equation 22.1)
    const double L_sun = constants::DEG_TO_RAD * (280.4665 + 36000.7698 * T);
    const double L_moon = constants::DEG_TO_RAD * (218.3165 + 481267.8813 * T);

    // Low-accuracy nutation in longitude (delta_psi), Meeus Chapter 22,
    // Equation 22.1 (accuracy 0.5 arcsec; the full model has 1365 terms):
    //   -17.20" sin(Omega) - 1.32" sin(2L) - 0.23" sin(2L') + 0.21" sin(2 Omega)
    const double dpsi = constants::ARCSEC_TO_RAD * (
        -17.20 * std::sin(Omega)          // Moon's node (18.6 years)
        - 1.32 * std::sin(2.0 * L_sun)    // half-year (Sun)
        - 0.23 * std::sin(2.0 * L_moon)   // half-month (Moon)
        + 0.21 * std::sin(2.0 * Omega));  // Moon's node, 9.3 years

    // Equation of the equinoxes: dpsi * cos(mean_obliquity)
    // Reference: Vallado [3], Equation 3-84
    const double ee = dpsi * std::cos(eps0);

    return gmst(ut1_jd) + ee;
}

/// Greenwich apparent sidereal time (radians; not wrapped to [0, 2*pi)) with the
/// 678 luni-solar nutation terms of IAU 2000A, at UT1 Julian date `ut1_jd` and
/// TT Julian date `tt_jd` (days). More precise than gast() (which keeps only
/// three nutation terms) while keeping the equinox-based method; for
/// comparisons with data in frames defined by GAST.
///
/// If `mjd` (UTC Modified Julian Date) is positive, the celestial pole offset
/// dX from the EOP table is added. That lookup can throw
/// DataCoverageError (no EOP table loaded under the default policy, or an
/// epoch more than 30 days beyond the table); pass mjd <= 0 to skip it.
[[nodiscard]] inline double gast_full(double ut1_jd, double tt_jd, double mjd = -1.0) {
    using namespace constants;

    // Julian centuries of TT from J2000.0
    const double T = (tt_jd - J2000_JD) / 36525.0;

    // Mean obliquity of the ecliptic (IAU 2006)
    const double eps0 = ARCSEC_TO_RAD * (84381.406
        - 46.836769 * T
        - 0.0001831 * T * T
        + 0.00200340 * T * T * T);

    // Full IAU 2006/2000A nutation in longitude (678 luni-solar terms)
    double dpsi = detail::compute_nutation_dpsi(T);

    // Apply CIP offset correction (dX) from EOP data if available
    // Reference: IERS TN 36, Section 5.5.4
    // The correction to dpsi is: delta_dpsi ≈ dX / sin(eps)
    if (mjd > 0.0) {
        auto [dX_rad, dY_rad] = time::EOPData::instance().cip_offsets_rad(mjd);
        double sin_eps = std::sin(eps0);
        if (std::abs(sin_eps) > 1e-10) {
            dpsi += dX_rad / sin_eps;
        }
    }

    // Equation of the equinoxes: dpsi * cos(mean_obliquity)
    const double ee = dpsi * std::cos(eps0);

    return gmst(ut1_jd) + ee;
}

/// Polar-motion matrix, used as [ITRF] = M · [TIRS]: it maps coordinates in
/// the Terrestrial Intermediate Reference System (TIRS: Earth-fixed axes with
/// Z along the true spin axis) to the ITRF.
/// Reference: IERS TN 36, Section 5.4.1, Equation 5.3, which writes the
/// polar-motion matrix as W = R₃(-s') · R₂(xp) · R₁(yp) for the opposite
/// direction (ITRF -> TIRS). This function returns its transpose exactly,
/// M = R₁(-yp) · R₂(-xp) · R₃(s'), the matrix ERFA/SOFA's eraPom00 builds
/// (R_i are rotations of the coordinate axes).
///
/// Parameters:
/// - xp_rad: Pole x coordinate (radians), positive toward Greenwich
/// - yp_rad: Pole y coordinate (radians), positive toward 270° longitude
/// - sp_rad: TIO locator s' (radians: the tiny angle fixing the zero of
///   longitude in the TIRS); about -12 microarcseconds at present
///   (-47 microarcseconds per century since 2000), often set to 0
[[nodiscard]] inline Matrix3d polar_motion_matrix(
    double xp_rad,
    double yp_rad,
    double sp_rad = 0.0) noexcept
{
    const double cx = std::cos(xp_rad), sx = std::sin(xp_rad);
    const double cy = std::cos(yp_rad), sy = std::sin(yp_rad);
    const double cs = std::cos(sp_rad), ss = std::sin(sp_rad);

    // R₁(-yp) · R₂(-xp) · R₃(s'), multiplied out
    Matrix3d W;
    W(0, 0) = cx * cs;
    W(0, 1) = cx * ss;
    W(0, 2) = sx;
    W(1, 0) = -cy * ss + sy * sx * cs;
    W(1, 1) = cy * cs + sy * sx * ss;
    W(1, 2) = -sy * cx;
    W(2, 0) = -sy * ss - cy * sx * cs;
    W(2, 1) = sy * cs - cy * sx * ss;
    W(2, 2) = cy * cx;

    return W;
}

/// Converts a position `pos` (m, Earth-fixed) from ITRF realization `from` to
/// `to` at `epoch`, with the published 14-parameter Helmert transformation
/// (itrf_transforms.hpp). Returns the input unchanged when from == to.
/// Throws DataCoverageError for an epoch before 1961 when from != to (the
/// parameters are evaluated at the UTC decimal year, and UTC starts in 1961).
[[nodiscard]] inline Vector3d convert_itrf(
    const Vector3d& pos,
    ITRFRealization from,
    ITRFRealization to,
    const Epoch& epoch)
{
    if (from == to) return pos;
    return transform_itrf(pos, from, to, util::epoch_to_decimal_year(epoch));
}

/// The Earth's rotation rate about the CIP [rad per second of UT1]: the rate
/// of the Earth rotation angle, 2*pi * 1.00273781191135448 / 86400 (IERS TN 36
/// eq. 5.15). Per second of TT/UTC it is smaller by the factor
/// (1 - LOD / 86400 s), LOD being the excess length of day (~1 ms: a 1.2e-8
/// relative change, ~6e-6 m/s at LEO); IAUTransform::earth_rotation_rate
/// applies that factor. The nominal WGS84 value
/// constants::earth::ANGULAR_VELOCITY (7.292115e-5 rad/s) is 1.47e-12 rad/s
/// smaller (~1e-5 m/s at LEO).
inline constexpr double EARTH_ROTATION_ANGLE_RATE =
    constants::TWO_PI * 1.00273781191135448 / constants::SECONDS_PER_DAY;

/// The Earth's rotation vector (rad/s): (0, 0, EARTH_ROTATION_ANGLE_RATE),
/// i.e. about the Z axis of a frame whose Z is the spin axis (CIRS/TIRS),
/// without the length-of-day correction (coords::earth_rotation_rate(epoch)
/// in iau_transforms.hpp includes it).
[[nodiscard]] inline Vector3d earth_rotation_vector() noexcept {
    return Vector3d(0.0, 0.0, EARTH_ROTATION_ANGLE_RATE);
}

/// Rate of change of GMST (rad/s), taken as the nominal WGS84 Earth rotation
/// rate (GMST's true rate is nearly constant and very close to it).
[[nodiscard]] inline double gmst_rate() noexcept {
    return constants::earth::ANGULAR_VELOCITY;  // rad/s
}

} // namespace coords
} // namespace ballistic
