#pragma once

// =============================================================================
// Inertial <-> Earth-fixed frame conversion (IAU 2006/2000A model)
// =============================================================================
//
// Converts positions and velocities between the two frames the library uses:
//
//   GCRS - the Geocentric Celestial Reference System: an inertial
//          (non-rotating) frame centered on the Earth, with axes fixed relative
//          to distant quasars. Orbits are integrated in it. (Sometimes loosely
//          called "ECI", Earth-centered inertial.)
//   ITRF - the International Terrestrial Reference Frame: the Earth-fixed frame
//          that rotates with the ground (also called "ECEF", Earth-centered
//          Earth-fixed). Orbit products, ground stations and the gravity field
//          are given in it.
//
// Going from one to the other needs three rotations:
//   - precession and nutation: the slow wobble of the Earth's spin axis in
//     space (precession: a 26,000-year cone; nutation: small periodic nods,
//     the largest with an 18.6-year period), computed here with the IAU
//     2006 precession and IAU 2000A nutation models (the combination SOFA
//     calls "nut06a");
//   - the Earth's daily spin, through the Earth rotation angle (ERA), which
//     needs UT1 (the time scale that follows the Earth's actual rotation);
//   - polar motion: the spin axis also wanders by several meters across the
//     Earth's surface.
// The last two depend on measured Earth orientation parameters (EOP: UT1-UTC,
// polar motion, celestial pole offsets) published by the IERS (International
// Earth Rotation and Reference Systems Service); see time/eop_data.hpp.
//
// References:
// -----------
// [1] Petit, G. & Luzum, B. (2010). "IERS Conventions (2010)".
//     IERS Technical Note No. 36. Chapter 5: Transformation between the
//     celestial and terrestrial reference systems.
//     URL: https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html
//
// [2] McCarthy, D.D. & Petit, G. (2004). "IERS Conventions (2003)".
//     IERS Technical Note No. 32.
//     - Fundamental arguments: Table 5.2a
//     - Nutation series: Table 5.3a
//
// [3] Capitaine, N., Wallace, P.T., & Chapront, J. (2003). "Expressions for
//     IAU 2000 precession quantities". A&A, 412, 567-586.
//     DOI: 10.1051/0004-6361:20031539
//
// [4] Fukushima, T. (2003). "A New Precession Formula".
//     The Astronomical Journal, 126, 494-534.
//     - Fukushima-Williams parameterization of precession
//
// [5] SOFA (Standards of Fundamental Astronomy) Library.
//     http://www.iausofa.org/
//     - Reference implementation for verification
//     - Functions: iauNut06a, iauPnm06a, iauEra00, iauFal03, etc.
//
// Frame Definitions:
// ------------------
// GCRF: the realization of the GCRS (inertial; axes aligned with J2000 to
//       within a small fixed "frame bias")
// CIRS: Celestial Intermediate Reference System - the GCRS turned so that its
//       Z axis is the Earth's true spin axis (the Celestial Intermediate Pole,
//       CIP); i.e. after precession-nutation
// TIRS: Terrestrial Intermediate Reference System - CIRS turned by the Earth's
//       spin (rotates with the Earth, Z still along the CIP)
// ITRF: TIRS corrected for polar motion (the Earth-fixed frame)
//
// Transformation chain: GCRF -> Q(t) -> CIRS -> R(t) -> TIRS -> W(t) -> ITRF
//   Q(t) = precession-nutation matrix, built from the CIP coordinates X, Y and
//          the "CIO locator" s (the CIO, Celestial Intermediate Origin, is the
//          reference point on the CIRS equator from which the Earth rotation
//          angle is measured)
//   R(t) = Earth rotation by the Earth rotation angle (ERA)
//   W(t) = polar motion
// In this file each matrix maps the frame on its left to the frame on its
// right. (IERS Conventions write the chain in the opposite direction,
// [GCRS] = Q R W [ITRS], so their Q, R, W are the transposes of these.)
//
// Implementation Notes:
// ---------------------
// - Nutation uses the full IAU 2000A series (678 luni-solar + 687 planetary
//   terms) with the IAU 2006 J2 rate correction, equivalent to SOFA iauNut06a.
// - Precession uses IAU 2006 Fukushima-Williams angles (SOFA iauPfw06).
// - Coefficients are sourced from IERS Conventions and verified against SOFA.
// - Maximum omitted contribution: <0.01 milliarcseconds (sub-mas precision).
// - Angles: 1 arcsecond (arcsec, ") = 1/3600 degree = 4.85e-6 rad; 1 mas
//   (milliarcsecond) = 1e-3 arcsec, about 3.4 cm at a low-Earth-orbit radius.
// - Time arguments named t_tt are Julian centuries of TT (Terrestrial Time,
//   the uniform time scale of the ephemerides) since J2000.0
//   (2000-01-01 12:00 TT); one Julian century = 36525 days.
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../time/epoch.hpp"
#include "../time/eop_data.hpp"
#include "../util/time_utils.hpp"
#include "nutation_coefficients.hpp"
#include "itrf_transforms.hpp"
#include "earth_rotation.hpp"  // for coords::polar_motion_matrix (canonical impl)
#include <cmath>
#include <array>
#include <cstdint>
#include <limits>

namespace ballistic {
namespace coords {

// Frame transformation chain: GCRF -> CIRS -> TIRS -> ITRF
// Reference: IERS TN 36 [1], Chapter 5

/// Earth orientation parameters (EOP) for one instant: the measured
/// corrections that the models above cannot predict. Used by IAUTransform only
/// when no EOP table is loaded (a loaded table always wins); then every field
/// is used, held constant in time.
struct EarthOrientationParameters {
    double xp = 0.0;        // Polar motion x (arcseconds)
    double yp = 0.0;        // Polar motion y (arcseconds)
    double dut1 = 0.0;      // UT1 - UTC (seconds): the Earth rotation angle
    double lod = 0.0;       // Length of day minus 86400 s (SECONDS; the table's
                            // column is in ms): the Earth's rotation rate
    double dX = 0.0;        // Celestial pole offset X (MILLIarcseconds, as in finals2000A)
    double dY = 0.0;        // Celestial pole offset Y (milliarcseconds)

    /// All corrections zero.
    static EarthOrientationParameters zero() {
        return EarthOrientationParameters{};
    }
};

/// The angles that drive the nutation series (radians): the five Delaunay
/// arguments, which describe where the Moon and Sun are in their orbits, plus
/// the mean longitudes of the planets.
struct FundamentalArguments {
    double l;     // Mean anomaly of the Moon
    double lp;    // Mean anomaly of the Sun
    double F;     // Moon's mean argument of latitude
    double D;     // Mean elongation of the Moon from the Sun
    double Om;    // Mean longitude of the Moon's ascending node

    // Planetary longitudes
    double L_Me;  // Mercury
    double L_Ve;  // Venus
    double L_E;   // Earth
    double L_Ma;  // Mars
    double L_J;   // Jupiter
    double L_Sa;  // Saturn
    double L_U;   // Uranus
    double L_Ne;  // Neptune
    double pa;    // General precession in longitude
};

/// Computes the nutation arguments at time t_tt (Julian centuries of TT since
/// J2000.0). Returns every angle in radians.
/// Reference: IERS TN 32 [2], Table 5.2a; SOFA iauFal03(), iauFalp03(), etc.
[[nodiscard]] inline FundamentalArguments compute_fundamental_args(double t_tt) {
    using namespace constants;

    FundamentalArguments args;

    // Mean anomaly of the Moon (l)
    // Reference: IERS TN 32 [2], Table 5.2a, row 1
    // SOFA equivalent: iauFal03()
    args.l = 485868.249036 + t_tt * (1717915923.2178
           + t_tt * (31.8792 + t_tt * (0.051635 + t_tt * (-0.00024470))));
    args.l = std::fmod(args.l, 1296000.0) * ARCSEC_TO_RAD;

    // Mean anomaly of the Sun (l')
    // Reference: IERS Conventions (2010), Table 5.2a, row 2
    // ERFA equivalent: eraFalp03() - coefficient 1287104.793048 from ERFA source
    args.lp = 1287104.793048 + t_tt * (129596581.0481
            + t_tt * (-0.5532 + t_tt * (0.000136 + t_tt * (-0.00001149))));
    args.lp = std::fmod(args.lp, 1296000.0) * ARCSEC_TO_RAD;

    // Mean argument of latitude of the Moon (F)
    // Reference: IERS TN 32 [2], Table 5.2a, row 3
    // SOFA equivalent: iauFaf03()
    args.F = 335779.526232 + t_tt * (1739527262.8478
           + t_tt * (-12.7512 + t_tt * (-0.001037 + t_tt * (0.00000417))));
    args.F = std::fmod(args.F, 1296000.0) * ARCSEC_TO_RAD;

    // Mean elongation of the Moon from the Sun (D)
    // Reference: IERS Conventions (2010), Table 5.2a, row 4
    // ERFA equivalent: eraFad03() - coefficient 1072260.703692 from ERFA source
    args.D = 1072260.703692 + t_tt * (1602961601.2090
           + t_tt * (-6.3706 + t_tt * (0.006593 + t_tt * (-0.00003169))));
    args.D = std::fmod(args.D, 1296000.0) * ARCSEC_TO_RAD;

    // Mean longitude of the Moon's ascending node (Omega)
    // Reference: IERS TN 32 [2], Table 5.2a, row 5
    // SOFA equivalent: iauFaom03()
    args.Om = 450160.398036 + t_tt * (-6962890.5431
            + t_tt * (7.4722 + t_tt * (0.007702 + t_tt * (-0.00005939))));
    args.Om = std::fmod(args.Om, 1296000.0) * ARCSEC_TO_RAD;

    // Planetary mean longitudes (radians)
    // Reference: SOFA iauFame03(), iauFave03(), iauFae03(), etc.
    // These are used for planetary nutation terms in the full IAU 2006/2000A model
    args.L_Me = std::fmod(4.402608842 + 2608.7903141574 * t_tt, TWO_PI);   // Mercury
    args.L_Ve = std::fmod(3.176146697 + 1021.3285546211 * t_tt, TWO_PI);   // Venus
    args.L_E = std::fmod(1.753470314 + 628.3075849991 * t_tt, TWO_PI);     // Earth
    args.L_Ma = std::fmod(6.203480913 + 334.0612426700 * t_tt, TWO_PI);    // Mars
    args.L_J = std::fmod(0.599546497 + 52.9690962641 * t_tt, TWO_PI);      // Jupiter
    args.L_Sa = std::fmod(0.874016757 + 21.3299104960 * t_tt, TWO_PI);     // Saturn
    args.L_U = std::fmod(5.481293872 + 7.4781598567 * t_tt, TWO_PI);       // Uranus
    args.L_Ne = std::fmod(5.311886287 + 3.8133035638 * t_tt, TWO_PI);      // Neptune

    // General precession in longitude
    // Reference: SOFA iauFapa03()
    args.pa = std::fmod(0.024381750 * t_tt + 0.00000538691 * t_tt * t_tt, TWO_PI);

    return args;
}

/// The four IAU 2006 precession angles in the Fukushima-Williams form
/// (radians). Together they give the orientation of the mean (precessing, not
/// nutating) equator and equinox of date relative to the GCRS.
struct PrecessionAngles {
    double gamb;   // gamma_bar
    double phib;   // phi_bar
    double psib;   // psi_bar
    double epsa;   // epsilon_A (mean obliquity)
};

/// Computes the IAU 2006 precession angles at time t_tt (Julian centuries of TT
/// since J2000.0). Returns radians.
/// Reference: Fukushima [4], Table 1; IERS TN 36 [1], Equations 5.38-5.41
/// SOFA equivalent: iauPfw06()
[[nodiscard]] inline PrecessionAngles compute_precession_angles(double t_tt) {
    using namespace constants;

    PrecessionAngles angles;

    // gamma_bar (arcseconds) - ICRS to mean equator of date
    // Reference: IERS TN 36 [1], Equation 5.38
    angles.gamb = -0.052928 + t_tt * (10.556378
                + t_tt * (0.4932044 + t_tt * (-0.00031238
                + t_tt * (-0.000002788 + t_tt * 0.0000000260))));
    angles.gamb *= ARCSEC_TO_RAD;

    // phi_bar (arcseconds) - obliquity of the ecliptic on J2000.0 ecliptic
    // Reference: IERS TN 36 [1], Equation 5.39
    angles.phib = 84381.412819 + t_tt * (-46.811016
                + t_tt * (0.0511268 + t_tt * (0.00053289
                + t_tt * (-0.000000440 + t_tt * (-0.0000000176)))));
    angles.phib *= ARCSEC_TO_RAD;

    // psi_bar (arcseconds) - precession angle
    // Reference: IERS TN 36 [1], Equation 5.40
    angles.psib = -0.041775 + t_tt * (5038.481484
                + t_tt * (1.5584175 + t_tt * (-0.00018522
                + t_tt * (-0.000026452 + t_tt * (-0.0000000148)))));
    angles.psib *= ARCSEC_TO_RAD;

    // Mean obliquity of the ecliptic (epsilon_A)
    // Reference: IERS TN 36 [1], Equation 5.40; also SOFA iauObl06()
    // Value at J2000.0: 84381.406" = 23°26'21.406"
    angles.epsa = 84381.406 + t_tt * (-46.836769
                + t_tt * (-0.0001831 + t_tt * (0.00200340
                + t_tt * (-0.000000576 + t_tt * (-0.0000000434)))));
    angles.epsa *= ARCSEC_TO_RAD;

    return angles;
}

/// Nutation angles: dpsi, the shift of the equinox along the ecliptic
/// ("nutation in longitude"), and deps, the change in the tilt of the equator
/// to the ecliptic ("nutation in obliquity").
struct NutationAngles {
    double dpsi;   // Nutation in longitude (radians)
    double deps;   // Nutation in obliquity (radians)
};

/// Computes the nutation angles (radians) at time t_tt (Julian centuries of TT
/// since J2000.0) from the precomputed arguments `args` (compute_fundamental_args
/// at the same t_tt), with the IAU 2006/2000A model.
/// Base series: IERS TN 32 [2], Table 5.3a (luni-solar) and 5.3b (planetary)
/// J2 correction: IAU 2006 rate adjustment per IERS TN 36 [1], Section 5.6.3
/// SOFA equivalent: iauNut06a()
///
/// The IAU 2006 J2 rate correction accounts for the slow steady change in the
/// Earth's flattening (its J2 gravity coefficient) that the original IAU 2000A
/// model left out. This makes the nutation consistent with IAU 2006 precession.
[[nodiscard]] inline NutationAngles compute_nutation(double t_tt, const FundamentalArguments& args) {
    using namespace constants;
    using namespace nutation;

    NutationAngles nut{0.0, 0.0};

    // Luni-solar nutation (primary contribution)
    for (size_t i = 0; i < LUNISOLAR_TERMS_COUNT; ++i) {
        const auto& term = LUNISOLAR_TERMS[i];

        // Every term is summed: 17 of them have only the out-of-phase
        // coefficients Cp, Sp (up to 0.2 mas), so a test on S, Sdot, C, Cdot
        // alone would drop them.
        double arg = term.nl * args.l + term.nlp * args.lp + term.nF * args.F
                   + term.nD * args.D + term.nOm * args.Om;

        double sin_arg = std::sin(arg);
        double cos_arg = std::cos(arg);

        // Nutation in longitude: dpsi = S*sin(arg) + Sdot*t*sin(arg) + Cp*cos(arg)
        nut.dpsi += (term.S + term.Sdot * t_tt) * sin_arg + term.Cp * cos_arg;

        // Nutation in obliquity: deps = C*cos(arg) + Cdot*t*cos(arg) + Sp*sin(arg)
        nut.deps += (term.C + term.Cdot * t_tt) * cos_arg + term.Sp * sin_arg;
    }

    // Planetary nutation (smaller contribution)
    for (size_t i = 0; i < PLANETARY_TERMS_COUNT; ++i) {
        const auto& term = PLANETARY_TERMS[i];

        // Skip zero terms
        if (term.S == 0.0 && term.C == 0.0 && term.Sp == 0.0 && term.Cp == 0.0) {
            continue;
        }

        // Compute argument from lunar + planetary terms
        double arg = term.nl * args.l + term.nF * args.F
                   + term.nD * args.D + term.nOm * args.Om
                   + term.nMe * args.L_Me + term.nVe * args.L_Ve
                   + term.nE * args.L_E + term.nMa * args.L_Ma
                   + term.nJ * args.L_J + term.nSa * args.L_Sa
                   + term.nU * args.L_U + term.nN * args.L_Ne
                   + term.npa * args.pa;

        double sin_arg = std::sin(arg);
        double cos_arg = std::cos(arg);

        // Add planetary nutation contributions
        nut.dpsi += term.S * sin_arg + term.C * cos_arg;
        nut.deps += term.Sp * sin_arg + term.Cp * cos_arg;
    }

    // Convert from 0.1 microarcseconds to radians
    nut.dpsi *= 0.1e-6 * ARCSEC_TO_RAD;
    nut.deps *= 0.1e-6 * ARCSEC_TO_RAD;

    // IAU 2006 adjustments (nut00a -> nut06a), as SOFA iauNut06a: the J2 rate
    // factor fj2 = -2.7774e-6 per Julian century on both angles, and on dpsi
    // also the constant 0.4697e-6 that makes the IAU 2000A nutation consistent
    // with the IAU 2006 precession (~8 microarcseconds).
    // Reference: IERS TN 36, Section 5.6.3
    const double fj2 = -2.7774e-6 * t_tt;
    nut.dpsi *= 1.0 + 0.4697e-6 + fj2;
    nut.deps *= 1.0 + fj2;

    return nut;
}

/// Direction of the Earth's spin axis (the Celestial Intermediate Pole, CIP)
/// in the GCRS, plus the CIO locator s.
struct CIPPosition {
    double X;   // X component of the CIP unit vector (dimensionless, = radians for small angles)
    double Y;   // Y component of the CIP unit vector
    double s;   // CIO locator: small angle fixing the CIRS origin on the equator (radians)
};

/// TIO locator s' (radians) at time t_tt (Julian centuries of TT since
/// J2000.0): a tiny angle (-47 microarcseconds per century) that fixes the
/// zero of longitude in the TIRS, the Earth-fixed counterpart of the CIO
/// locator. TIO = Terrestrial Intermediate Origin.
/// Reference: IERS TN 36 [1], Equation 5.13 (first-order formula).
[[nodiscard]] inline double compute_tio_locator(double t_tt) noexcept {
    using namespace constants;
    return -47.0e-6 * ARCSEC_TO_RAD * t_tt;
}

/// CIO locator s (radians) at time t_tt (Julian centuries of TT since J2000.0),
/// given the nutation arguments `args` and the CIP coordinates X, Y at the same
/// time. s = polynomial + periodic terms - XY/2.
/// Reference: IERS TN 36 [1], Section 5.5.4, Table 5.2d
[[nodiscard]] inline double compute_cio_locator_s(double t_tt,
    const FundamentalArguments& args, double X, double Y) noexcept
{
    using namespace constants;

    // Polynomial part of s (microarcseconds)
    // Reference: IERS TN 36, Eq. 5.16; ERFA s06() sp[0..5] coefficients
    // Note: This is a 5th-order polynomial (6 terms) per IERS Conventions 2010 and ERFA
    double s_poly = 94.0                                           // sp[0]
                  + 3808.65 * t_tt                                 // sp[1]
                  - 122.68 * t_tt * t_tt                           // sp[2]
                  - 72574.11 * t_tt * t_tt * t_tt                  // sp[3]
                  + 27.98 * t_tt * t_tt * t_tt * t_tt              // sp[4]
                  + 15.62 * t_tt * t_tt * t_tt * t_tt * t_tt;      // sp[5]

    // Periodic terms from ERFA s06.c (verified against IERS TN36 Table 5.2d)
    // s0: 33 terms (t^0), s1: 3 terms (t^1), s2: 25 terms (t^2),
    // s3: 4 terms (t^3), s4: 1 term (t^4)
    // Two s0 terms (#16, #24) require planetary longitudes (Venus, Earth, pa).
    struct SPeriodicTerm {
        int nl, nlp, nF, nD, nOm;
        int nVe, nE, npa;  // Planetary: Venus, Earth, general precession
        double S, C;  // sin and cos coefficients in microarcseconds
    };

    // s0 group: 33 terms (coefficient of t^0)
    // Reference: ERFA s06.c (verified term by term)
    static constexpr std::array<SPeriodicTerm, 33> s0_terms = {{
        {0, 0, 0, 0, 1,  0, 0, 0, -2640.73, 0.39},
        {0, 0, 0, 0, 2,  0, 0, 0,   -63.53, 0.02},
        {0, 0, 2,-2, 3,  0, 0, 0,   -11.75,-0.01},
        {0, 0, 2,-2, 1,  0, 0, 0,   -11.21,-0.01},
        {0, 0, 2,-2, 2,  0, 0, 0,     4.57, 0.00},
        {0, 0, 2, 0, 3,  0, 0, 0,    -2.02, 0.00},
        {0, 0, 2, 0, 1,  0, 0, 0,    -1.98, 0.00},
        {0, 0, 0, 0, 3,  0, 0, 0,     1.72, 0.00},
        {0, 1, 0, 0, 1,  0, 0, 0,     1.41, 0.01},
        {0, 1, 0, 0,-1,  0, 0, 0,     1.26, 0.01},
        {1, 0, 0, 0,-1,  0, 0, 0,     0.63, 0.00},
        {1, 0, 0, 0, 1,  0, 0, 0,     0.63, 0.00},
        {0, 1, 2,-2, 3,  0, 0, 0,    -0.46, 0.00},
        {0, 1, 2,-2, 1,  0, 0, 0,    -0.45, 0.00},
        {0, 0, 4,-4, 4,  0, 0, 0,    -0.36, 0.00},
        {0, 0, 1,-1, 1, -8,12, 0,     0.24, 0.12},  // Planetary: -8*Ve+12*E
        {0, 0, 2, 0, 0,  0, 0, 0,    -0.32, 0.00},
        {0, 0, 2, 0, 2,  0, 0, 0,    -0.28, 0.00},
        {1, 0, 2, 0, 3,  0, 0, 0,    -0.27, 0.00},
        {1, 0, 2, 0, 1,  0, 0, 0,    -0.26, 0.00},
        {0, 0, 2,-2, 0,  0, 0, 0,     0.21, 0.00},
        {0, 1,-2, 2,-3,  0, 0, 0,    -0.19, 0.00},
        {0, 1,-2, 2,-1,  0, 0, 0,    -0.18, 0.00},
        {0, 0, 0, 0, 0,  8,-13,-1,    0.10,-0.05},  // Planetary only: 8*Ve-13*E-pa
        {0, 0, 0, 2, 0,  0, 0, 0,    -0.15, 0.00},
        {2, 0,-2, 0,-1,  0, 0, 0,     0.14, 0.00},
        {0, 1, 2,-2, 2,  0, 0, 0,     0.14, 0.00},
        {1, 0, 0,-2, 1,  0, 0, 0,    -0.14, 0.00},
        {1, 0, 0,-2,-1,  0, 0, 0,    -0.14, 0.00},
        {0, 0, 4,-2, 4,  0, 0, 0,    -0.13, 0.00},
        {0, 0, 2,-2, 4,  0, 0, 0,     0.11, 0.00},
        {1, 0,-2, 0,-3,  0, 0, 0,    -0.11, 0.00},
        {1, 0,-2, 0,-1,  0, 0, 0,    -0.11, 0.00},
    }};

    // s1 group: 3 terms (coefficient of t^1)
    static constexpr std::array<SPeriodicTerm, 3> s1_terms = {{
        {0, 0, 0, 0, 2, 0, 0, 0, -0.07, 3.57},
        {0, 0, 0, 0, 1, 0, 0, 0,  1.73,-0.03},
        {0, 0, 2,-2, 3, 0, 0, 0,  0.00, 0.48},
    }};

    // s2 group: 25 terms (coefficient of t^2)
    static constexpr std::array<SPeriodicTerm, 25> s2_terms = {{
        {0, 0, 0, 0, 1, 0, 0, 0, 743.52,-0.17},
        {0, 0, 2,-2, 2, 0, 0, 0,  56.91, 0.06},
        {0, 0, 2, 0, 2, 0, 0, 0,   9.84,-0.01},
        {0, 0, 0, 0, 2, 0, 0, 0,  -8.85, 0.01},
        {0, 1, 0, 0, 0, 0, 0, 0,  -6.38,-0.05},
        {1, 0, 0, 0, 0, 0, 0, 0,  -3.07, 0.00},
        {0, 1, 2,-2, 2, 0, 0, 0,   2.23, 0.00},
        {0, 0, 2, 0, 1, 0, 0, 0,   1.67, 0.00},
        {1, 0, 2, 0, 2, 0, 0, 0,   1.30, 0.00},
        {0, 1,-2, 2,-2, 0, 0, 0,   0.93, 0.00},
        {1, 0, 0,-2, 0, 0, 0, 0,   0.68, 0.00},
        {0, 0, 2,-2, 1, 0, 0, 0,  -0.55, 0.00},
        {1, 0,-2, 0,-2, 0, 0, 0,   0.53, 0.00},
        {0, 0, 0, 2, 0, 0, 0, 0,  -0.27, 0.00},
        {1, 0, 0, 0, 1, 0, 0, 0,  -0.27, 0.00},
        {1, 0,-2,-2,-2, 0, 0, 0,  -0.26, 0.00},
        {1, 0, 0, 0,-1, 0, 0, 0,  -0.25, 0.00},
        {1, 0, 2, 0, 1, 0, 0, 0,   0.22, 0.00},
        {2, 0, 0,-2, 0, 0, 0, 0,  -0.21, 0.00},
        {2, 0,-2, 0,-1, 0, 0, 0,   0.20, 0.00},
        {0, 0, 2, 2, 2, 0, 0, 0,   0.17, 0.00},
        {2, 0, 2, 0, 2, 0, 0, 0,   0.13, 0.00},
        {2, 0, 0, 0, 0, 0, 0, 0,  -0.13, 0.00},
        {1, 0, 2,-2, 2, 0, 0, 0,  -0.12, 0.00},
        {0, 0, 2, 0, 0, 0, 0, 0,  -0.11, 0.00},
    }};

    // s3 group: 4 terms (coefficient of t^3)
    static constexpr std::array<SPeriodicTerm, 4> s3_terms = {{
        {0, 0, 0, 0, 1, 0, 0, 0,  0.30,-23.42},
        {0, 0, 2,-2, 2, 0, 0, 0, -0.03, -1.46},
        {0, 0, 2, 0, 2, 0, 0, 0, -0.01, -0.25},
        {0, 0, 0, 0, 2, 0, 0, 0,  0.00,  0.23},
    }};

    // s4 group: 1 term (coefficient of t^4)
    static constexpr std::array<SPeriodicTerm, 1> s4_terms = {{
        {0, 0, 0, 0, 1, 0, 0, 0, -0.26,-0.01},
    }};

    // Evaluate periodic terms for a given group
    auto eval_periodic = [&args](const auto& terms) -> double {
        double sum = 0.0;
        for (const auto& term : terms) {
            double arg = term.nl * args.l + term.nlp * args.lp + term.nF * args.F
                       + term.nD * args.D + term.nOm * args.Om
                       + term.nVe * args.L_Ve + term.nE * args.L_E
                       + term.npa * args.pa;
            sum += term.S * std::sin(arg) + term.C * std::cos(arg);
        }
        return sum;
    };

    double s_periodic = eval_periodic(s0_terms)
                      + eval_periodic(s1_terms) * t_tt
                      + eval_periodic(s2_terms) * t_tt * t_tt
                      + eval_periodic(s3_terms) * t_tt * t_tt * t_tt
                      + eval_periodic(s4_terms) * t_tt * t_tt * t_tt * t_tt;

    // Total s in radians
    // s = polynomial + periodic - XY/2
    double s = (s_poly + s_periodic) * 1e-6 * ARCSEC_TO_RAD - X * Y / 2.0;

    return s;
}

/// Builds the combined frame-bias, precession and nutation matrix (NPB) from
/// Fukushima-Williams angles (all radians): gamb, phib as from
/// compute_precession_angles, psi = psib + dpsi, eps = epsa + deps. The matrix
/// maps GCRS vectors to the true equator and equinox of date (precession and
/// nutation applied). Equivalent to ERFA's fw2m.
/// Reference: IERS TN 36, Section 5.2; SOFA iauFw2m()
///
/// ERFA fw2m applies rotations as: Rz(gamb) -> Rx(phib) -> Rz(-psi) -> Rx(-eps)
/// where each rotation PRE-multiplies the accumulated result.
/// Final: NPB = Rx(-eps) * Rz(-psi) * Rx(phib) * Rz(gamb)
[[nodiscard]] inline Matrix3d build_npb_matrix(
    double gamb, double phib, double psi, double eps) noexcept
{
    // Start with identity
    Matrix3d r = Matrix3d::Identity();

    // Helper to apply Rz(angle) as pre-multiplication: r = Rz * r
    auto apply_rz = [](Matrix3d& m, double angle) {
        double c = std::cos(angle), s = std::sin(angle);
        double a00 = c * m(0, 0) + s * m(1, 0);
        double a01 = c * m(0, 1) + s * m(1, 1);
        double a02 = c * m(0, 2) + s * m(1, 2);
        double a10 = -s * m(0, 0) + c * m(1, 0);
        double a11 = -s * m(0, 1) + c * m(1, 1);
        double a12 = -s * m(0, 2) + c * m(1, 2);
        m(0, 0) = a00; m(0, 1) = a01; m(0, 2) = a02;
        m(1, 0) = a10; m(1, 1) = a11; m(1, 2) = a12;
    };

    // Helper to apply Rx(angle) as pre-multiplication: r = Rx * r
    auto apply_rx = [](Matrix3d& m, double angle) {
        double c = std::cos(angle), s = std::sin(angle);
        double a10 = c * m(1, 0) + s * m(2, 0);
        double a11 = c * m(1, 1) + s * m(2, 1);
        double a12 = c * m(1, 2) + s * m(2, 2);
        double a20 = -s * m(1, 0) + c * m(2, 0);
        double a21 = -s * m(1, 1) + c * m(2, 1);
        double a22 = -s * m(1, 2) + c * m(2, 2);
        m(1, 0) = a10; m(1, 1) = a11; m(1, 2) = a12;
        m(2, 0) = a20; m(2, 1) = a21; m(2, 2) = a22;
    };

    // Apply rotations in order (each pre-multiplies)
    apply_rz(r, gamb);    // r = Rz(gamb)
    apply_rx(r, phib);    // r = Rx(phib) * Rz(gamb)
    apply_rz(r, -psi);    // r = Rz(-psi) * Rx(phib) * Rz(gamb)
    apply_rx(r, -eps);    // r = Rx(-eps) * Rz(-psi) * Rx(phib) * Rz(gamb)

    return r;
}

/// Position of the Celestial Intermediate Pole (X, Y) and the CIO locator s at
/// time t_tt (Julian centuries of TT since J2000.0), from the IAU 2006/2000A
/// model alone (no measured pole offsets dX, dY).
[[nodiscard]] inline CIPPosition compute_cip_position(double t_tt) {
    using namespace constants;

    auto prec = compute_precession_angles(t_tt);
    auto args = compute_fundamental_args(t_tt);
    auto nut = compute_nutation(t_tt, args);

    CIPPosition cip;

    // CIP X, Y from NPB (bias-precession-nutation) matrix
    // Reference: IERS TN 36, Section 5.2; SOFA iauBpn2xy(), iauPnm06a()
    //
    // Approach:
    // 1. Build NPB matrix from F-W angles: NPB = fw2m(gamb, phib, psi, eps)
    //    where psi = psib + dpsi and eps = epsa + deps
    // 2. Extract X = NPB[2,0] and Y = NPB[2,1]

    double psi = prec.psib + nut.dpsi;  // psi_bar + nutation in longitude
    double eps = prec.epsa + nut.deps;  // epsilon_A + nutation in obliquity

    Matrix3d npb = build_npb_matrix(prec.gamb, prec.phib, psi, eps);

    // X and Y are the first two elements of the third row
    // This is how ERFA's bpn2xy extracts CIP coordinates
    cip.X = npb(2, 0);
    cip.Y = npb(2, 1);

    // CIO locator s with full polynomial and periodic terms
    cip.s = compute_cio_locator_s(t_tt, args, cip.X, cip.Y);

    return cip;
}

/// Earth rotation angle (ERA, radians in [0, 2*pi)) for a UT1 Julian date
/// `ut1_jd` (days). ERA is how far the Earth has turned about its spin axis,
/// measured from the CIO to the TIO; it grows uniformly with UT1. It replaces
/// Greenwich apparent sidereal time (GAST) in the modern IERS method.
/// Reference: IERS TN 36 [1], Equation 5.15
/// SOFA/ERFA equivalent: eraEra00()
///
/// This follows ERFA's era00() formula, splitting the date into its integer
/// and fractional day. The split gains nothing over the one double passed in:
/// a Julian date near 2.46e6 resolves only ~40 microseconds, about 3e-9 rad of
/// Earth rotation (~2 cm at a low-Earth-orbit radius). The two-part overload
/// below keeps full precision; IAUTransform uses it. The formula is:
///   ERA = 2π * (f + 0.7790572732640 + 0.00273781191135448 * Du)
/// where:
///   Du = UT1 Julian date - J2000.0
///   f = fractional day component of UT1 JD
///
/// The coefficient 0.00273781191135448 = 1.00273781191135448 - 1, with the
/// "1" accounted for by the fractional day f.
[[nodiscard]] inline double earth_rotation_angle(double ut1_jd) {
    using namespace constants;

    // Split JD into integer and fractional parts for precision
    // This matches ERFA's two-part JD handling
    double dj1 = std::floor(ut1_jd);
    double dj2 = ut1_jd - dj1;

    // Days since J2000.0 (ERFA: t = d1 + (d2 - DJ00))
    double t = dj1 + (dj2 - J2000_JD);

    // Fractional part of UT1 day (ERFA: f = fmod(d1, 1.0) + fmod(d2, 1.0))
    double f = std::fmod(dj1, 1.0) + std::fmod(dj2, 1.0);

    // Earth rotation angle (ERFA formula)
    double theta = TWO_PI * (f + 0.7790572732640 + 0.00273781191135448 * t);

    // Normalize to [0, 2π) - equivalent to ERFA's eraAnpm followed by + 2π if negative
    theta = std::fmod(theta, TWO_PI);
    if (theta < 0.0) {
        theta += TWO_PI;
    }

    return theta;
}

/// Earth rotation angle (radians in [0, 2*pi)) for a UT1 Julian date given in
/// two parts, ut1_jd1 + ut1_jd2 (days, in either order), which keeps more
/// precision than one double. Matches ERFA's era00 exactly.
[[nodiscard]] inline double earth_rotation_angle(double ut1_jd1, double ut1_jd2) {
    using namespace constants;

    // Order so that d1 < d2 (ERFA convention)
    double d1, d2;
    if (ut1_jd1 < ut1_jd2) {
        d1 = ut1_jd1;
        d2 = ut1_jd2;
    } else {
        d1 = ut1_jd2;
        d2 = ut1_jd1;
    }

    // Days since J2000.0
    double t = d1 + (d2 - J2000_JD);

    // Fractional part of UT1 day
    double f = std::fmod(d1, 1.0) + std::fmod(d2, 1.0);

    // Earth rotation angle (ERFA formula)
    double theta = TWO_PI * (f + 0.7790572732640 + 0.00273781191135448 * t);

    // Normalize to [0, 2π)
    theta = std::fmod(theta, TWO_PI);
    if (theta < 0.0) {
        theta += TWO_PI;
    }

    return theta;
}

// Build rotation matrices for full transformation

/// GCRS -> CIRS rotation matrix Q (precession and nutation) at time t_tt
/// (Julian centuries of TT since J2000.0), including the measured celestial
/// pole offsets dX_mas, dY_mas (milliarcseconds, from the EOP table) that
/// correct the model's CIP position.
/// Reference: IERS TN 36 [1] Sections 5.4.4 and 5.5.4, ERFA c2ixys
///
/// Built from the CIP coordinates (X, Y) and the CIO locator s with the
/// closed-form matrix of IERS TN 36 eq. 5.10 (exact, no small-angle
/// approximation): third row = (X, Y, sqrt(1 - X^2 - Y^2)), followed by a
/// rotation by s about Z.
[[nodiscard]] inline Matrix3d gcrs_to_cirs_matrix(double t_tt, double dX_mas, double dY_mas) {
    auto cip = compute_cip_position(t_tt);

    constexpr double MAS_TO_RAD = constants::ARCSEC_TO_RAD / 1000.0;
    double X = cip.X + dX_mas * MAS_TO_RAD;
    double Y = cip.Y + dY_mas * MAS_TO_RAD;
    double s = cip.s;

    double r2 = X * X + Y * Y;
    double a = 1.0 / (1.0 + std::sqrt(1.0 - r2));

    Matrix3d Q;
    Q(0, 0) = 1.0 - a * X * X;
    Q(0, 1) = -a * X * Y;
    Q(0, 2) = -X;
    Q(1, 0) = -a * X * Y;
    Q(1, 1) = 1.0 - a * Y * Y;
    Q(1, 2) = -Y;
    Q(2, 0) = X;
    Q(2, 1) = Y;
    Q(2, 2) = 1.0 - a * (X * X + Y * Y);

    double cos_s = std::cos(s);
    double sin_s = std::sin(s);

    Matrix3d Rs;
    Rs << cos_s, -sin_s, 0,
          sin_s,  cos_s, 0,
              0,      0, 1;

    return Rs * Q;
}

/// GCRS -> CIRS rotation matrix Q at time t_tt (Julian centuries of TT since
/// J2000.0), from the model alone (no measured pole offsets).
[[nodiscard]] inline Matrix3d gcrs_to_cirs_matrix(double t_tt) {
    auto cip = compute_cip_position(t_tt);

    double X = cip.X;
    double Y = cip.Y;
    double s = cip.s;

    // Closed-form matrix as in ERFA c2ixys (exact for any X, Y).
    double r2 = X * X + Y * Y;
    double a = 1.0 / (1.0 + std::sqrt(1.0 - r2));

    // Build the basic Q matrix (before s rotation)
    // ERFA c2ixys convention:
    //   Third column = [-X, -Y, 1-a*(X²+Y²)]
    //   Third row = [X, Y, 1-a*(X²+Y²)]
    // Note: Q[0,2]=-X and Q[2,0]=+X (verified against ERFA reference values)
    Matrix3d Q;
    Q(0, 0) = 1.0 - a * X * X;
    Q(0, 1) = -a * X * Y;
    Q(0, 2) = -X;
    Q(1, 0) = -a * X * Y;
    Q(1, 1) = 1.0 - a * Y * Y;
    Q(1, 2) = -Y;
    Q(2, 0) = X;
    Q(2, 1) = Y;
    Q(2, 2) = 1.0 - a * (X * X + Y * Y);

    // Apply the s rotation about Z. [cos -sin; sin cos] is ERFA's Rz(-s)
    // (ERFA's Rz(a) is [cos sin; -sin cos]).
    double cos_s = std::cos(s);
    double sin_s = std::sin(s);

    Matrix3d Rs;
    Rs << cos_s, -sin_s, 0,
          sin_s,  cos_s, 0,
          0,      0,     1;

    return Rs * Q;
}

/// CIRS -> TIRS rotation matrix R for an Earth rotation angle `era` (radians).
/// R = Rz(ERA) in ERFA's convention.
[[nodiscard]] inline Matrix3d cirs_to_tirs_matrix_for_angle(double era) {
    double cos_era = std::cos(era);
    double sin_era = std::sin(era);

    Matrix3d R;
    R << cos_era,  sin_era, 0,
        -sin_era,  cos_era, 0,
         0,        0,       1;

    return R;
}

/// CIRS -> TIRS rotation matrix R: the Earth's spin by the Earth rotation angle
/// at UT1 Julian date `ut1_jd` (days; one double resolves ~40 microseconds,
/// ~2 cm at a LEO radius).
[[nodiscard]] inline Matrix3d cirs_to_tirs_matrix(double ut1_jd) {
    return cirs_to_tirs_matrix_for_angle(earth_rotation_angle(ut1_jd));
}

/// CIRS -> TIRS rotation matrix R at the UT1 Julian day number `ut1_day` plus
/// `ut1_day_fraction` (days; the two parts keep full precision).
[[nodiscard]] inline Matrix3d cirs_to_tirs_matrix(std::int64_t ut1_day, double ut1_day_fraction) {
    return cirs_to_tirs_matrix_for_angle(
        earth_rotation_angle(static_cast<double>(ut1_day), ut1_day_fraction));
}

/// TIRS -> ITRF rotation matrix W (polar motion), from the pole coordinates
/// eop.xp, eop.yp (arcseconds; the other EOP fields are not used) and the TIO
/// locator s' at time t_tt (Julian centuries of TT since J2000.0; the default
/// 0 means J2000.0, where s' = 0).
/// Reference: IERS TN 36 [1], Equation 5.3.
/// Delegates to coords::polar_motion_matrix (earth_rotation.hpp), which takes
/// radians.
[[nodiscard]] inline Matrix3d tirs_to_itrf_matrix(
    const EarthOrientationParameters& eop,
    double t_tt = 0.0) noexcept
{
    using namespace constants;
    return coords::polar_motion_matrix(
        eop.xp * ARCSEC_TO_RAD,
        eop.yp * ARCSEC_TO_RAD,
        compute_tio_locator(t_tt));
}

/// Same as above with the pole coordinates passed directly (arcseconds).
[[nodiscard]] inline Matrix3d tirs_to_itrf_matrix(
    double xp_arcsec, double yp_arcsec, double t_tt = 0.0) noexcept
{
    EarthOrientationParameters eop;
    eop.xp = xp_arcsec;
    eop.yp = yp_arcsec;
    return tirs_to_itrf_matrix(eop, t_tt);
}

/// The complete GCRS <-> ITRF conversion for positions, velocities and plain
/// vectors.
///
/// Earth orientation: when an EOP table is loaded (time::EOPData), it is
/// always used; otherwise the parameters passed to the constructor or
/// set_eop() are used (all of them: pole, UT1 - UTC, LOD, dX, dY); when those
/// are all zero too, the table lookup still runs so that its "no EOP loaded"
/// policy applies (by default it raises DataCoverageError;
/// EopPolicy::AssumeZero accepts zeros, costing about 3-10 m at low Earth
/// orbit).
///
/// ITRF realization: the IERS EOP define ITRF2020 axes; when a different
/// realization (e.g. ITRF2014) is selected, positions are converted with the
/// published 14-parameter Helmert transformation (itrf_transforms.hpp).
///
/// Every result is a function of the epoch alone (not of earlier calls), so a
/// state converted at the same epoch gives the same numbers whatever the
/// instance converted before. The precession-nutation series (1365 terms) is
/// evaluated on a fixed 600 s grid of TT and the CIP coordinates X, Y, s are
/// interpolated between the grid nodes with a cubic (Catmull-Rom; error
/// below 1e-15 rad); the EOP, the polar motion and the Earth rotation angle
/// are evaluated at the epoch itself. The Earth rotation angle comes from the
/// exact UT1 day and seconds (a one-double UT1 Julian date resolves only
/// ~40 microseconds, ~2 cm at a LEO radius).
/// Not thread-safe: use one instance per thread (the free functions below do).
///
/// Every conversion can throw DataCoverageError from the EOP lookup (no table
/// loaded under the default policy, or an epoch more than 30 days beyond the
/// table).
class IAUTransform {
public:
    /// Zero explicit EOP, ITRF2020 output.
    IAUTransform()
        : eop_(EarthOrientationParameters::zero())
        , target_itrf_(ITRFRealization::ITRF2020) {}

    /// Explicit EOP (used only when no EOP table is loaded), ITRF2020 output.
    explicit IAUTransform(const EarthOrientationParameters& eop)
        : eop_(eop)
        , target_itrf_(ITRFRealization::ITRF2020) {}

    /// Explicit EOP (used only when no EOP table is loaded) and the ITRF
    /// realization of the Earth-fixed side.
    IAUTransform(const EarthOrientationParameters& eop, ITRFRealization itrf)
        : eop_(eop)
        , target_itrf_(itrf) {}

    /// Replaces the explicit EOP (used only when no EOP table is loaded).
    void set_eop(const EarthOrientationParameters& eop) {
        eop_ = eop;
        memo_valid_ = false;
    }

    /// Selects the ITRF realization of the Earth-fixed side.
    /// Default: ITRF2020. Use ITRF2014 for ESA Sentinel data before March 4, 2025.
    void set_target_itrf(ITRFRealization itrf) { target_itrf_ = itrf; }
    [[nodiscard]] ITRFRealization target_itrf() const { return target_itrf_; }

    /// Rotation matrix from GCRS to ITRF2020 axes at `epoch`:
    /// [ITRF] = W(t) * R(t) * Q(t) * [GCRS] (IERS TN 36 [1], Equation 5.1).
    /// No realization change (that applies to positions only).
    /// Throws DataCoverageError from the EOP lookup (see the class comment).
    [[nodiscard]] Matrix3d gcrs_to_itrf_matrix(const Epoch& epoch) const {
        const Frame& f = frame_at(epoch);
        return f.W * f.R * f.Q;
    }

    /// The Earth's rotation rate about the CIP at `epoch` [rad/s of TT]:
    /// EARTH_ROTATION_ANGLE_RATE * (1 + d(UT1-UTC)/dt). With a table, the
    /// rate of its interpolated UT1-UTC (about -LOD / 86400 s; the rate the
    /// converted positions turn at, so converted velocities are their
    /// derivative); with explicit EOP, -eop.lod / 86400 s. Throws
    /// DataCoverageError from the EOP lookup.
    [[nodiscard]] double earth_rotation_rate(const Epoch& epoch) const {
        return frame_at(epoch).omega;
    }

private:
    // Grid of the precession-nutation nodes [s of TT]. 86400 is a multiple,
    // so the node index follows exactly from the Epoch's (day, seconds).
    static constexpr double kCipGridS = 600.0;
    static constexpr std::int64_t kCipNodesPerDay = 144;
    static constexpr std::int64_t kJ2000Day = 2451545;

    struct CipNode {
        std::int64_t k = std::numeric_limits<std::int64_t>::min();
        double X = 0.0, Y = 0.0, s = 0.0;
    };

    struct Frame {
        Matrix3d Q, R, W;
        double omega = EARTH_ROTATION_ANGLE_RATE;
        double xp = 0.0, yp = 0.0;                  // pole [arcsec]
        double xp_rate = 0.0, yp_rate = 0.0;        // [arcsec/s]
        double t_tt = 0.0;
        double X = 0.0, Y = 0.0, s = 0.0;           // incl. dX, dY
        double Xdot = 0.0, Ydot = 0.0, sdot = 0.0;  // [1/s]
    };

    // CIP X, Y, s of the model alone at grid node k (TT = J2000 + k * 600 s).
    [[nodiscard]] const CipNode& cip_node(std::int64_t k) const {
        for (const auto& n : nodes_) {
            if (n.k == k) return n;
        }
        CipNode& n = nodes_[next_node_];
        next_node_ = (next_node_ + 1) % nodes_.size();
        const double t_tt = static_cast<double>(k) /
            (static_cast<double>(kCipNodesPerDay) * constants::DAYS_PER_JULIAN_CENTURY);
        const CIPPosition cip = compute_cip_position(t_tt);
        n.k = k;
        n.X = cip.X;
        n.Y = cip.Y;
        n.s = cip.s;
        return n;
    }

    // Q from CIP coordinates (radians), closed form of IERS TN 36 eq. 5.10
    // followed by the rotation by s (as gcrs_to_cirs_matrix).
    [[nodiscard]] static Matrix3d q_from_xys(double X, double Y, double s) {
        const double r2 = X * X + Y * Y;
        const double a = 1.0 / (1.0 + std::sqrt(1.0 - r2));
        Matrix3d Q;
        Q << 1.0 - a * X * X, -a * X * Y,       -X,
             -a * X * Y,      1.0 - a * Y * Y,  -Y,
             X,               Y,                1.0 - a * r2;
        const double cs = std::cos(s), ss = std::sin(s);
        Matrix3d Rs;
        Rs << cs, -ss, 0,
              ss,  cs, 0,
              0,   0,  1;
        return Rs * Q;
    }

    // Everything the conversions need at `epoch`, computed once per epoch.
    [[nodiscard]] const Frame& frame_at(const Epoch& epoch) const {
        const std::uint64_t gen = time::EOPData::instance().generation();
        if (memo_valid_ && epoch.tt_day() == memo_day_ && epoch.tt_seconds() == memo_sec_ &&
            gen == memo_gen_) {
            return memo_;
        }
        Frame f;

        // UTC (day, seconds) of this instant; the EOP table is indexed by UTC.
        std::int64_t day = epoch.tt_day();
        double sec = epoch.tt_seconds() + epoch.seconds_from_tt(TimeScale::UTC);
        auto normalize = [](std::int64_t& d, double& s) {
            const double whole = std::floor(s / constants::SECONDS_PER_DAY);
            d += static_cast<std::int64_t>(whole);
            s -= whole * constants::SECONDS_PER_DAY;
        };
        normalize(day, sec);
        const double utc_mjd = static_cast<double>(day - 2400000) +
                               (sec / constants::SECONDS_PER_DAY - 0.5);

        // The loaded table (via the one EOP lookup and its policy) wins;
        // without one, explicitly set parameters are used. All-zero
        // parameters with no table go through the lookup too, so the
        // "no EOP loaded" note is reported.
        EarthOrientationParameters eop = eop_;
        double ut1_rate = -eop_.lod / constants::SECONDS_PER_DAY;   // d(UT1-UTC)/dt [s/s]
        const bool explicit_eop = eop_.xp != 0.0 || eop_.yp != 0.0 || eop_.dut1 != 0.0 ||
                                  eop_.dX != 0.0 || eop_.dY != 0.0 || eop_.lod != 0.0;
        if (time::EOPData::instance().is_loaded() || !explicit_eop) {
            const time::EopSample s = time::EOPData::instance().lookup(utc_mjd);
            eop.xp   = s.xp_arcsec;
            eop.yp   = s.yp_arcsec;
            eop.dut1 = s.ut1_utc_s;
            eop.dX   = s.dX_mas;
            eop.dY   = s.dY_mas;
            ut1_rate  = s.ut1_utc_rate_s_per_day / constants::SECONDS_PER_DAY;
            f.xp_rate = s.xp_rate_arcsec_per_day / constants::SECONDS_PER_DAY;
            f.yp_rate = s.yp_rate_arcsec_per_day / constants::SECONDS_PER_DAY;
        }

        // Precession-nutation: CIP X, Y, s interpolated on the node grid
        // (Catmull-Rom through nodes k-1 .. k+2), plus the measured offsets.
        const double grid_in_day = std::floor(epoch.tt_seconds() / kCipGridS);
        const std::int64_t k = (epoch.tt_day() - kJ2000Day) * kCipNodesPerDay +
                               static_cast<std::int64_t>(grid_in_day);
        const double u = (epoch.tt_seconds() - grid_in_day * kCipGridS) / kCipGridS;
        const CipNode n0 = cip_node(k - 1), n1 = cip_node(k), n2 = cip_node(k + 1),
                      n3 = cip_node(k + 2);
        auto cubic = [u](double p0, double p1, double p2, double p3, double& rate) {
            const double c1 = 0.5 * (p2 - p0);
            const double c2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
            const double c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2);
            rate = (c1 + u * (2.0 * c2 + 3.0 * u * c3)) / kCipGridS;
            return p1 + u * (c1 + u * (c2 + u * c3));
        };
        constexpr double MAS_TO_RAD = constants::ARCSEC_TO_RAD / 1000.0;
        f.X = cubic(n0.X, n1.X, n2.X, n3.X, f.Xdot) + eop.dX * MAS_TO_RAD;
        f.Y = cubic(n0.Y, n1.Y, n2.Y, n3.Y, f.Ydot) + eop.dY * MAS_TO_RAD;
        f.s = cubic(n0.s, n1.s, n2.s, n3.s, f.sdot);
        f.Q = q_from_xys(f.X, f.Y, f.s);

        // Earth rotation angle from the exact UT1 (day, seconds).
        std::int64_t ut1_day = day;
        double ut1_sec = sec + eop.dut1;
        normalize(ut1_day, ut1_sec);
        f.R = cirs_to_tirs_matrix(ut1_day, ut1_sec / constants::SECONDS_PER_DAY);

        f.t_tt = epoch.julian_centuries_tt();
        f.xp = eop.xp;
        f.yp = eop.yp;
        f.W = tirs_to_itrf_matrix(eop, f.t_tt);
        f.omega = EARTH_ROTATION_ANGLE_RATE * (1.0 + ut1_rate);

        memo_ = f;
        memo_day_ = epoch.tt_day();
        memo_sec_ = epoch.tt_seconds();
        memo_gen_ = gen;
        memo_valid_ = true;
        memo_qdot_valid_ = false;
        memo_wdot_valid_ = false;
        return memo_;
    }

    // dW/dt [1/s]: the polar motion's rate (~1 mas/day, ~4e-7 m/s at LEO),
    // the derivative of the interpolated pole. Computed only when a VELOCITY
    // is transformed.
    [[nodiscard]] const Matrix3d& wdot(const Epoch& epoch) const {
        const Frame& f = frame_at(epoch);
        if (!memo_wdot_valid_) {
            constexpr double h = 300.0;
            memo_wdot_ = (tirs_to_itrf_matrix(f.xp + f.xp_rate * h, f.yp + f.yp_rate * h, f.t_tt) -
                          tirs_to_itrf_matrix(f.xp - f.xp_rate * h, f.yp - f.yp_rate * h, f.t_tt)) /
                         (2.0 * h);
            memo_wdot_valid_ = true;
        }
        return memo_wdot_;
    }

    // dQ/dt [1/s]: how fast the CIP drifts in the GCRS (precession-nutation,
    // ~3e-12 rad/s), the derivative of the interpolated Q (central difference
    // along the interpolated X, Y, s rates; Q is linear in them to ~1e-18).
    // Computed only when a VELOCITY is transformed.
    [[nodiscard]] const Matrix3d& qdot(const Epoch& epoch) const {
        const Frame& f = frame_at(epoch);
        if (!memo_qdot_valid_) {
            constexpr double h = 300.0;
            memo_qdot_ = (q_from_xys(f.X + f.Xdot * h, f.Y + f.Ydot * h, f.s + f.sdot * h) -
                          q_from_xys(f.X - f.Xdot * h, f.Y - f.Ydot * h, f.s - f.sdot * h)) /
                         (2.0 * h);
            memo_qdot_valid_ = true;
        }
        return memo_qdot_;
    }

public:

    /// Rotation matrix from ITRF2020 axes to GCRS at `epoch` (transpose of
    /// gcrs_to_itrf_matrix). Throws DataCoverageError from the EOP lookup.
    [[nodiscard]] Matrix3d itrf_to_gcrs_matrix(const Epoch& epoch) const {
        return gcrs_to_itrf_matrix(epoch).transpose();
    }

    /// Position (m) from GCRS to ITRF in the selected realization at `epoch`.
    /// Throws DataCoverageError from the EOP lookup.
    [[nodiscard]] Vector3d gcrs_to_itrf(const Vector3d& pos_gcrs, const Epoch& epoch) const {
        Vector3d pos_itrf2020 = gcrs_to_itrf_matrix(epoch) * pos_gcrs;
        return apply_itrf_realization(pos_itrf2020, epoch);
    }

    /// Position (m) from ITRF in the selected realization to GCRS at `epoch`.
    /// Throws DataCoverageError from the EOP lookup.
    [[nodiscard]] Vector3d itrf_to_gcrs(const Vector3d& pos_itrf, const Epoch& epoch) const {
        Vector3d pos_itrf2020 = apply_itrf_realization_inverse(pos_itrf, epoch);
        return itrf_to_gcrs_matrix(epoch) * pos_itrf2020;
    }

    /// Position (m) and velocity (m/s) from GCRS to ITRF (selected
    /// realization) at `epoch`. The ITRF velocity is the time derivative of
    /// the ITRF position. Throws DataCoverageError from the EOP lookup.
    ///
    /// Reference: IERS TN 36, Chapter 5. With r_gcrs = Q^T R^T W^T r_itrf, the
    /// velocity has THREE terms: the rotated ITRF velocity, Earth's spin about
    /// the CIP (omega_gcrs = Q^T omega_cirs, at the Earth rotation angle's
    /// rate corrected by the length of day), and the drift of the CIP itself,
    /// Qdot^T Q r_gcrs (precession-nutation, ~3e-12 rad/s -> ~2e-5 m/s at LEO,
    /// low Earth orbit). Omitting the third term leaves every converted start
    /// velocity off by ~2e-5 m/s, up to ~0.2 m of 1-hour along-track error that
    /// varies with the orbit's orientation in space (seen as a seasonal swing
    /// in Sentinel orbit predictions; docs/validation_claims.md,
    /// SENTINEL-YEAR-2025). The polar motion's rate (~1 mas/day, ~4e-7 m/s)
    /// is included too.
    [[nodiscard]] StateVector gcrs_to_itrf(const StateVector& state_gcrs, const Epoch& epoch) const {
        const Matrix3d full_R = gcrs_to_itrf_matrix(epoch);
        const Frame& f = frame_at(epoch);

        // Earth rotation rate along the CIP (the CIRS Z axis), in GCRS
        const Vector3d omega_gcrs = f.Q.transpose() * Vector3d(0.0, 0.0, f.omega);

        // Position transformation (to ITRF2020)
        Vector3d pos_itrf2020 = full_R * state_gcrs.position;

        // Velocity transformation: v_ITRF = R * (v_GCRS - omega_gcrs x r_GCRS - Qdot^T Q r_GCRS)
        Vector3d v_inertial = state_gcrs.velocity - omega_gcrs.cross(state_gcrs.position)
                            - qdot(epoch).transpose() * (f.Q * state_gcrs.position);
        // ... plus the polar motion's rate, Wdot W^T r_itrf
        Vector3d vel_itrf = full_R * v_inertial + wdot(epoch) * (f.W.transpose() * pos_itrf2020);

        // Apply ITRF realization conversion to position
        // Note: Helmert transformation for velocity is negligible (mm-level effect on velocity)
        Vector3d pos_itrf = apply_itrf_realization(pos_itrf2020, epoch);

        return StateVector(pos_itrf, vel_itrf);
    }

    /// The part of an ITRF velocity that comes from the CIP drift,
    /// R * Qdot^T Q r_gcrs [m/s] (~2e-5 m/s at LEO), for an ITRF position
    /// `pos_itrf` (m, selected realization) at `epoch`. A product whose ITRF
    /// velocity records were written without the CIP drift carries exactly
    /// this extra term: v_true = v_file - cip_drift_velocity(). Check before
    /// correcting: check_velocity_records() (velocity_check.hpp) compares
    /// records with the derivative of the positions. Measured (pinned in
    /// tests/python/test_velocity_records.py): GFZ CHAMP/GRACE-FO SP3 records
    /// are true derivatives; ILRS SLR (satellite laser ranging) records fit
    /// neither convention. Throws DataCoverageError from the EOP lookup.
    [[nodiscard]] Vector3d cip_drift_velocity(const Vector3d& pos_itrf, const Epoch& epoch) const {
        const Matrix3d full_R = gcrs_to_itrf_matrix(epoch);
        const Frame& f = frame_at(epoch);
        const Vector3d pos_gcrs = full_R.transpose() * apply_itrf_realization_inverse(pos_itrf, epoch);
        return full_R * (qdot(epoch).transpose() * (f.Q * pos_gcrs));
    }

    /// Position (m) and velocity (m/s) from ITRF (selected realization; the
    /// velocity must be the true time derivative of the position) to GCRS at
    /// `epoch`; the inverse of gcrs_to_itrf(StateVector). Throws
    /// DataCoverageError from the EOP lookup.
    [[nodiscard]] StateVector itrf_to_gcrs(const StateVector& state_itrf, const Epoch& epoch) const {
        // Convert input position from target realization to ITRF2020
        Vector3d pos_itrf2020 = apply_itrf_realization_inverse(state_itrf.position, epoch);

        const Matrix3d full_Rt = gcrs_to_itrf_matrix(epoch).transpose();
        const Frame& f = frame_at(epoch);

        // Position transformation
        Vector3d pos_gcrs = full_Rt * pos_itrf2020;

        // Earth rotation rate along the CIP, in GCRS
        const Vector3d omega_gcrs = f.Q.transpose() * Vector3d(0.0, 0.0, f.omega);

        // Velocity transformation:
        //   v_GCRS = R^T (v_ITRF - Wdot W^T r_ITRF) + omega_gcrs x r_GCRS + Qdot^T Q r_GCRS
        // Note: Using original velocity - Helmert effect on velocity is negligible
        const Vector3d v_tirs_frame = state_itrf.velocity -
                                      wdot(epoch) * (f.W.transpose() * pos_itrf2020);
        Vector3d vel_gcrs = full_Rt * v_tirs_frame + omega_gcrs.cross(pos_gcrs)
                          + qdot(epoch).transpose() * (f.Q * pos_gcrs);

        return StateVector(pos_gcrs, vel_gcrs);
    }

private:
    EarthOrientationParameters eop_;
    ITRFRealization target_itrf_;

    // Caches (results depend on the epoch only, see the class comment):
    // precession-nutation grid nodes, and the last epoch's matrices.
    // Thread-safety: This class is NOT thread-safe. Each thread should use its own
    // IAUTransform instance, or use the thread-local convenience functions
    // (gcrs_to_itrf_position, gcrs_to_itrf_full, etc.) which provide automatic
    // per-thread caching via get_cached_transform().
    mutable std::array<CipNode, 8> nodes_{};
    mutable std::size_t next_node_ = 0;
    mutable Frame memo_;
    mutable std::int64_t memo_day_ = 0;
    mutable double memo_sec_ = 0.0;
    mutable std::uint64_t memo_gen_ = 0;
    mutable bool memo_valid_ = false;
    mutable Matrix3d memo_qdot_;
    mutable bool memo_qdot_valid_ = false;
    mutable Matrix3d memo_wdot_;
    mutable bool memo_wdot_valid_ = false;

    // Apply ITRF realization transformation if needed.
    // The IAU transforms output ITRF2020-aligned coordinates when using
    // modern EOP data; convert_itrf is a no-op when target is ITRF2020.
    [[nodiscard]] Vector3d apply_itrf_realization(const Vector3d& pos_itrf2020, const Epoch& epoch) const {
        return convert_itrf(pos_itrf2020, ITRFRealization::ITRF2020, target_itrf_, epoch);
    }

    [[nodiscard]] Vector3d apply_itrf_realization_inverse(const Vector3d& pos_target, const Epoch& epoch) const {
        return convert_itrf(pos_target, target_itrf_, ITRFRealization::ITRF2020, epoch);
    }
};

// One IAUTransform per thread, so the free functions below keep their caches
// and are safe to call from several threads at once.
namespace detail {
    inline IAUTransform& get_cached_transform() {
        thread_local IAUTransform transform;
        return transform;
    }
}

// Free functions on the per-thread cached IAUTransform. `itrf` selects the
// realization of the Earth-fixed side (default ITRF2020; use ITRF2014 for ESA
// Sentinel data before March 4, 2025). All throw DataCoverageError from the
// EOP lookup (see IAUTransform).

/// Position (m) from GCRS to ITRF (realization `itrf`) at `epoch`.
[[nodiscard]] inline Vector3d gcrs_to_itrf_full(
    const Vector3d& pos_gcrs,
    const Epoch& epoch,
    ITRFRealization itrf = ITRFRealization::ITRF2020)
{
    auto& transform = detail::get_cached_transform();
    if (transform.target_itrf() != itrf) {
        transform.set_target_itrf(itrf);
    }
    return transform.gcrs_to_itrf(pos_gcrs, epoch);
}

/// Position (m) from ITRF (realization `itrf`) to GCRS at `epoch`.
[[nodiscard]] inline Vector3d itrf_to_gcrs_full(
    const Vector3d& pos_itrf,
    const Epoch& epoch,
    ITRFRealization itrf = ITRFRealization::ITRF2020)
{
    auto& transform = detail::get_cached_transform();
    if (transform.target_itrf() != itrf) {
        transform.set_target_itrf(itrf);
    }
    return transform.itrf_to_gcrs(pos_itrf, epoch);
}

/// CIP-drift part of an ITRF velocity (m/s) at ITRF position `pos_itrf` (m);
/// see IAUTransform::cip_drift_velocity.
[[nodiscard]] inline Vector3d cip_drift_velocity(
    const Vector3d& pos_itrf, const Epoch& epoch,
    ITRFRealization itrf = ITRFRealization::ITRF2020)
{
    auto& transform = detail::get_cached_transform();
    if (transform.target_itrf() != itrf) transform.set_target_itrf(itrf);
    return transform.cip_drift_velocity(pos_itrf, epoch);
}

/// The Earth's rotation rate about the CIP at `epoch` [rad/s]; see
/// IAUTransform::earth_rotation_rate. Throws DataCoverageError from the EOP
/// lookup.
[[nodiscard]] inline double earth_rotation_rate(const Epoch& epoch) {
    return detail::get_cached_transform().earth_rotation_rate(epoch);
}

/// Position and velocity from GCRS to ITRF (realization `itrf`) at `epoch`;
/// see IAUTransform::gcrs_to_itrf(StateVector).
[[nodiscard]] inline StateVector gcrs_to_itrf_full(
    const StateVector& state_gcrs,
    const Epoch& epoch,
    ITRFRealization itrf = ITRFRealization::ITRF2020)
{
    auto& transform = detail::get_cached_transform();
    if (transform.target_itrf() != itrf) {
        transform.set_target_itrf(itrf);
    }
    return transform.gcrs_to_itrf(state_gcrs, epoch);
}

/// Position and velocity from ITRF (realization `itrf`) to GCRS at `epoch`;
/// see IAUTransform::itrf_to_gcrs(StateVector).
[[nodiscard]] inline StateVector itrf_to_gcrs_full(
    const StateVector& state_itrf,
    const Epoch& epoch,
    ITRFRealization itrf = ITRFRealization::ITRF2020)
{
    auto& transform = detail::get_cached_transform();
    if (transform.target_itrf() != itrf) {
        transform.set_target_itrf(itrf);
    }
    return transform.itrf_to_gcrs(state_itrf, epoch);
}

/// Rotation matrix mapping a GCRS vector to ITRF2020 axes at `epoch`. No ITRF
/// realization change (that affects positions only).
[[nodiscard]] inline Matrix3d gcrs_to_itrf_rotation(const Epoch& epoch) {
    return detail::get_cached_transform().gcrs_to_itrf_matrix(epoch);
}

/// Rotation matrix mapping an ITRF2020 vector to GCRS axes at `epoch` (the
/// transpose of gcrs_to_itrf_rotation).
[[nodiscard]] inline Matrix3d itrf_to_gcrs_rotation(const Epoch& epoch) {
    return detail::get_cached_transform().itrf_to_gcrs_matrix(epoch);
}

/// Rotates a free vector (an acceleration, a direction; not a position, not a
/// velocity - no Earth-spin term is added) from GCRS to ITRF2020 axes at
/// `epoch`. No ITRF realization change (that affects positions only).
[[nodiscard]] inline Vector3d gcrs_to_itrf_vector_full(const Vector3d& vec_gcrs, const Epoch& epoch) {
    return detail::get_cached_transform().gcrs_to_itrf_matrix(epoch) * vec_gcrs;
}

/// Rotates a free vector from ITRF2020 axes to GCRS at `epoch` (inverse of
/// gcrs_to_itrf_vector_full).
[[nodiscard]] inline Vector3d itrf_to_gcrs_vector_full(const Vector3d& vec_itrf, const Epoch& epoch) {
    return detail::get_cached_transform().itrf_to_gcrs_matrix(epoch) * vec_itrf;
}

} // namespace coords
} // namespace ballistic
