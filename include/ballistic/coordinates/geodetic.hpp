#pragma once

// =============================================================================
// Geodetic <-> ECEF Coordinate Transformations
// =============================================================================
//
// This module provides conversions between geodetic (latitude, longitude,
// altitude) and Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates
// using the WGS84 reference ellipsoid.
//
// "Geodetic" means measured against the ellipsoid that models the Earth's
// flattened shape: geodetic latitude is the angle between the equator and the
// local vertical (the normal to the ellipsoid), and geodetic altitude is the
// height above the ellipsoid along that normal. Geocentric latitude, the
// angle of the position vector from the equatorial plane, differs from it by
// up to 0.19 degrees. WGS84 is the World Geodetic System 1984, the standard
// GPS ellipsoid.
//
// References:
// -----------
// [1] NIMA (2000). "Department of Defense World Geodetic System 1984".
//     NIMA TR8350.2, 3rd Edition.
//     - WGS84 defining parameters: Table 3.1
//     - Coordinate conversion equations: Section 4.4
//
// [2] Vermeille, H. (2011). "An analytical method to transform geocentric
//     into geodetic coordinates". J. Geodesy 85, 105-117.
//     - Closed-form ECEF-to-geodetic algorithm
//
// [3] Vallado, D.A. (2013). "Fundamentals of Astrodynamics and Applications",
//     4th Edition. Microcosm Press. Section 3.4.
//
// WGS84 Ellipsoid Parameters (from constants.hpp):
// - Semi-major axis (a): 6378137.0 m
// - Flattening (f): 1/298.257223563
// - Eccentricity squared (e²): 0.00669437999014
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../core/defaults.hpp"
#include <cmath>

namespace ballistic {
namespace coords {

/// Converts geodetic latitude `lat` and longitude `lon` (radians) and altitude
/// `alt` above the WGS84 ellipsoid (m) to an ECEF position (m).
/// Reference: NIMA TR8350.2 [1], Section 4.4.4, Equations 4-14 through 4-16
[[nodiscard]] inline Vector3d geodetic_to_ecef(double lat, double lon, double alt) noexcept {
    using namespace constants::earth;

    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);

    // Radius of curvature in the prime vertical
    const double N = RADIUS_EQUATORIAL / std::sqrt(1.0 - ECCENTRICITY_SQ * sin_lat * sin_lat);

    // ECEF coordinates
    const double x = (N + alt) * cos_lat * cos_lon;
    const double y = (N + alt) * cos_lat * sin_lon;
    const double z = (N * (1.0 - ECCENTRICITY_SQ) + alt) * sin_lat;

    return Vector3d(x, y, z);
}

/// Same as above, from a GeodeticCoord (radians, radians, meters).
[[nodiscard]] inline Vector3d geodetic_to_ecef(const GeodeticCoord& coord) noexcept {
    return geodetic_to_ecef(coord.latitude, coord.longitude, coord.altitude);
}

/// Converts an ECEF position (m) to WGS84 geodetic latitude and longitude
/// (radians) and altitude (m), exactly (closed form, no iteration).
/// Reference: Vermeille, H. (2004/2011). "Computing geodetic coordinates from
/// geocentric coordinates", J. Geodesy 78:94-95; "An analytical method to
/// transform geocentric into geodetic coordinates", J. Geodesy 85:105-117.
/// The closed form is valid everywhere outside the ellipsoid's evolute (a
/// small region within ~43 km of the Earth's center); inside it the function
/// falls back to a fixed-point iteration (nothing physical is that deep).
/// Altitude and latitude (as arc length) within 1 micrometer from 1 km below
/// the ground to beyond geostationary orbit, poles included (asserted in
/// test_geodetic.cpp).
[[nodiscard]] inline GeodeticCoord ecef_to_geodetic(const Vector3d& ecef) noexcept {
    using namespace constants::earth;
    constexpr double a = RADIUS_EQUATORIAL;
    constexpr double e2 = ECCENTRICITY_SQ;
    constexpr double e4 = e2 * e2;

    const double x = ecef.x();
    const double y = ecef.y();
    const double z = ecef.z();
    const double lon = std::atan2(y, x);
    const double rho = std::hypot(x, y);

    const double p = (rho / a) * (rho / a);
    const double q = (1.0 - e2) * (z / a) * (z / a);
    const double r = (p + q - e4) / 6.0;

    if (r > 0.0) {
        const double s = e4 * p * q / (4.0 * r * r * r);
        const double t = std::cbrt(1.0 + s + std::sqrt(s * (2.0 + s)));
        const double u = r * (1.0 + t + 1.0 / t);
        const double v = std::sqrt(u * u + e4 * q);
        const double w = e2 * (u + v - q) / (2.0 * v);
        const double k = std::sqrt(u + v + w * w) - w;
        const double D = k * rho / (k + e2);
        const double Dz = std::hypot(D, z);
        const double lat = 2.0 * std::atan2(z, D + Dz);
        const double alt = (k + e2 - 1.0) / k * Dz;
        return GeodeticCoord(lat, lon, alt);
    }

    // Inside the evolute (|r| < ~43 km): fixed-point on the latitude.
    double lat = std::atan2(z, rho * (1.0 - e2));
    for (int i = 0; i < 50; ++i) {
        const double sl = std::sin(lat);
        const double N = a / std::sqrt(1.0 - e2 * sl * sl);
        const double next = std::atan2(z + e2 * N * sl, rho);
        if (std::abs(next - lat) < 1e-15) { lat = next; break; }
        lat = next;
    }
    const double sl = std::sin(lat);
    const double cl = std::cos(lat);
    const double N = a / std::sqrt(1.0 - e2 * sl * sl);
    // Projection onto the normal: valid at every latitude, no pole branch.
    const double alt = rho * cl + z * sl - N * (1.0 - e2 * sl * sl);
    return GeodeticCoord(lat, lon, alt);
}

/// Builds a GeodeticCoord from latitude and longitude in degrees and altitude
/// in meters (stored as radians, radians, meters).
[[nodiscard]] inline GeodeticCoord geodetic_from_degrees(
    double lat_deg, double lon_deg, double alt_m) noexcept
{
    return GeodeticCoord(
        lat_deg * constants::DEG_TO_RAD,
        lon_deg * constants::DEG_TO_RAD,
        alt_m
    );
}

/// Geodetic altitude (m) above the WGS84 ellipsoid of an ECEF position (m);
/// exact, the same as ecef_to_geodetic(ecef).altitude. Shortcuts such as
/// r - N(geocentric latitude) are wrong by -21 km at 45 deg and -41 km at
/// 80 deg. This is the altitude propagate_to_altitude and at_alt use.
[[nodiscard]] inline double ecef_to_altitude(const Vector3d& ecef) noexcept {
    return ecef_to_geodetic(ecef).altitude;
}

/// Rotation matrix from ECEF axes to the local East-North-Up (ENU) axes at
/// geodetic latitude `lat` and longitude `lon` (radians). Its rows are the
/// east, north and up unit vectors expressed in ECEF.
[[nodiscard]] inline Matrix3d ecef_to_enu_rotation(double lat, double lon) noexcept {
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);

    Matrix3d R;
    // East vector
    R(0, 0) = -sin_lon;
    R(0, 1) = cos_lon;
    R(0, 2) = 0.0;
    // North vector
    R(1, 0) = -sin_lat * cos_lon;
    R(1, 1) = -sin_lat * sin_lon;
    R(1, 2) = cos_lat;
    // Up vector
    R(2, 0) = cos_lat * cos_lon;
    R(2, 1) = cos_lat * sin_lon;
    R(2, 2) = sin_lat;

    return R;
}

/// Rotates an ECEF velocity (m/s) into local East-North-Up components at
/// geodetic latitude `lat` and longitude `lon` (radians).
[[nodiscard]] inline Vector3d ecef_to_enu_velocity(
    const Vector3d& vel_ecef, double lat, double lon) noexcept
{
    return ecef_to_enu_rotation(lat, lon) * vel_ecef;
}

/// Geocentric latitude (radians) of a point ON the WGS84 ellipsoid with
/// geodetic latitude `geodetic_lat` (radians). For points above the surface
/// the geocentric latitude also depends on altitude; use ecef_to_geodetic.
[[nodiscard]] inline double geodetic_to_geocentric_latitude(double geodetic_lat) noexcept {
    // tan(geocentric) = (1 - e^2) * tan(geodetic)
    return std::atan((1.0 - constants::earth::ECCENTRICITY_SQ) * std::tan(geodetic_lat));
}

/// Distance (m) from the Earth's center to the WGS84 ellipsoid surface at
/// geodetic latitude `lat` (radians).
[[nodiscard]] inline double earth_radius_at_latitude(double lat) noexcept {
    using namespace constants::earth;
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);

    const double a2 = RADIUS_EQUATORIAL * RADIUS_EQUATORIAL;
    const double b2 = RADIUS_POLAR * RADIUS_POLAR;

    return std::sqrt(
        (a2 * a2 * cos_lat * cos_lat + b2 * b2 * sin_lat * sin_lat) /
        (a2 * cos_lat * cos_lat + b2 * sin_lat * sin_lat)
    );
}

} // namespace coords

// GeodeticCoord::from_degrees (declared in core/types.hpp): degrees in,
// radians stored.
inline GeodeticCoord GeodeticCoord::from_degrees(double lat_deg, double lon_deg, double alt_m) {
    return coords::geodetic_from_degrees(lat_deg, lon_deg, alt_m);
}

} // namespace ballistic
