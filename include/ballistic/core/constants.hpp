#pragma once

#include <numbers>

// Physical and mathematical constants, SI units (meters, seconds, kilograms,
// radians) unless the name says otherwise. GM is a body's gravitational
// parameter (Newton's G times its mass) [m^3/s^2].

namespace ballistic {
namespace constants {

// Mathematical constants
constexpr double PI = std::numbers::pi;
constexpr double TWO_PI = 2.0 * PI;
constexpr double DEG_TO_RAD = PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / PI;
constexpr double ARCSEC_TO_RAD = PI / (180.0 * 3600.0);

// Time constants
constexpr double SECONDS_PER_DAY = 86400.0;
constexpr double SECONDS_PER_HOUR = 3600.0;
constexpr double SECONDS_PER_MINUTE = 60.0;
constexpr double MINUTES_PER_HOUR = 60.0;
constexpr double HOURS_PER_DAY = 24.0;
constexpr double DAYS_PER_JULIAN_CENTURY = 36525.0;

// Reference instants as Julian dates (continuous day counts; days start at noon)
constexpr double J2000_JD = 2451545.0;           // J2000.0 epoch (2000-01-01 12:00:00 TT)
constexpr double MJD_OFFSET = 2400000.5;         // JD - MJD (modified Julian date) offset
constexpr double UNIX_EPOCH_JD = 2440587.5;      // Unix epoch (1970-01-01 00:00:00 UTC)

// Time scale offsets: TT = Terrestrial Time, TAI = International Atomic Time
constexpr double TT_TAI_OFFSET = 32.184;         // TT = TAI + 32.184 seconds (IAU 1991 Resolution A4, exact)
constexpr double GPS_TAI_OFFSET = -19.0;         // GPS = TAI - 19 seconds (GPS ICD, exact since 1980-01-06)

// Earth parameters (NGA WGS84, NIMA TR8350.2)
namespace earth {
    constexpr double GM = 3.986004418e14;        // Gravitational parameter [m^3/s^2] (WGS84)
    constexpr double RADIUS_EQUATORIAL = 6378137.0;  // Semi-major axis [m] (WGS84 defining parameter)
    constexpr double RADIUS_POLAR = 6356752.3142;    // Semi-minor axis [m] (WGS84 derived)
    constexpr double FLATTENING = 1.0 / 298.257223563;  // Flattening (WGS84 defining parameter)
    constexpr double ECCENTRICITY_SQ = 2.0 * FLATTENING - FLATTENING * FLATTENING;
    constexpr double ANGULAR_VELOCITY = 7.292115e-5;  // Earth rotation rate [rad/s] (WGS84)

    // Zonal harmonic coefficients J_n: the parts of the Earth's gravity field
    // that depend only on latitude (J2 = equatorial bulge). J_n = -C_n0 *
    // sqrt(2n+1), from the fully normalized C_n0 of the EGM2008 gravity model
    // the library embeds
    // (src/gravity/egm2008_coefficients.cpp), so the analytic zonal helpers
    // and the full gravity field agree. tests/unit/test_constants.cpp checks
    // every J_n against that table.
    constexpr double J2 = 1.082626173852223e-03;   // Oblateness (dominant perturbation)
    constexpr double J3 = -2.532410518567722e-06;  // Pear-shape term
    constexpr double J4 = -1.619897599916973e-06;
    constexpr double J5 = -2.277535907308362e-07;
    constexpr double J6 = 5.406665762838132e-07;
    constexpr double J7 = -3.505517957137420e-07;
    constexpr double J8 = -2.039931259298844e-07;
    constexpr double J9 = -1.221279589194960e-07;
    constexpr double J10 = -2.443907697726934e-07;
    constexpr double J11 = 2.434765909981471e-07;
    constexpr double J12 = -1.821809613072860e-07;
    constexpr double J13 = -2.168318145607224e-07;
    constexpr double J14 = 1.220715373469818e-07;
    constexpr double J15 = -1.220543892828698e-08;

    // The constants EGM2008's coefficients are published with (NGA EGM2008
    // release: the TT-compatible GM and the model's reference radius).
    // The gravity engine does NOT use them: it evaluates EGM2008 with the
    // WGS84 GM and RADIUS_EQUATORIAL above. Measured by changing only these
    // two values: switching to EGM2008's own made the Sentinel, CHAMP and
    // LAGEOS/Etalon validations all worse (Sentinel 1-hour position RMSE
    // 0.216 -> 0.224 m; trap EGM-GM-CONVENTION in docs/trap_census.md).
    // The IERS Conventions tie the scale of the terrestrial
    // reference system (ITRS) to TCG (Geocentric Coordinate Time), whose GM is
    // the WGS84 value, and the reference orbits we validate against appear
    // to be computed that way. Kept here, named, so the choice is visible.
    constexpr double EGM2008_GM = 3.986004415e14;     // [m^3/s^2] (not used by the engine)
    constexpr double EGM2008_RADIUS = 6378136.3;      // [m] (not used by the engine)

    // J_COEFFS[n] = J_n for n = 2..15 (entries 0 and 1 are zero)
    constexpr double J_COEFFS[16] = {
        0.0, 0.0, J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13, J14, J15
    };
}

// Sun parameters
namespace sun {
    constexpr double GM = 1.32712440041279419e20;  // Gravitational parameter [m^3/s^2] (JPL DE440, gm_de440.tpc)
    constexpr double AU = 149597870700.0;          // Astronomical unit [m] (IAU 2012 Resolution B2, exact)
}

// Moon parameters
namespace moon {
    constexpr double GM = 4.9028001184575496e12;   // Gravitational parameter [m^3/s^2] (JPL DE440, gm_de440.tpc)
    constexpr double MEAN_DISTANCE = 384400000.0;  // Mean Earth-Moon distance [m]
}

// Jupiter parameters
namespace jupiter {
    constexpr double GM = 1.26712764e17;           // System barycenter GM [m^3/s^2] (JPL DE440, gm_de440.tpc)
}

// Physical constants
constexpr double SPEED_OF_LIGHT = 299792458.0;   // Speed of light [m/s] (SI definition, exact)

// Solid Earth tide Love numbers k_nm (IERS Conventions 2010 Table 6.3,
// anelastic, nominal): how strongly the solid Earth deforms under the Sun's
// and Moon's pull, and so how much that deformation changes its gravity
// field, per degree n and order m.
namespace tides {
    constexpr double k20_real = 0.30190;
    constexpr double k21_real = 0.29830;
    constexpr double k21_imag = -0.00144;
    constexpr double k22_real = 0.30102;
    constexpr double k22_imag = -0.00130;
    constexpr double k30_real = 0.093;
    constexpr double k31_real = 0.093;
    constexpr double k32_real = 0.093;
    constexpr double k33_real = 0.094;
}

// Atmospheric constants (USSA76 = U.S. Standard Atmosphere 1976 / standard definitions)
namespace atmosphere {
    constexpr double SEA_LEVEL_DENSITY = 1.225;  // kg/m^3 at sea level (USSA76)
    constexpr double SCALE_HEIGHT = 8500.0;      // Approximate scale height [m] (USSA76)
    constexpr double KARMAN_LINE = 100000.0;     // Karman line altitude [m] (FAI definition)
}

} // namespace constants
} // namespace ballistic
