#pragma once

// =============================================================================
// TEME: the output frame of SGP4
// =============================================================================
//
// "True Equator, Mean Equinox": the z axis is the true pole of date (the
// Earth's rotation axis including precession, its slow centuries-long drift,
// and nutation, its small periodic nodding), the x axis the MEAN equinox of
// date (the direction where the mean equator crosses the ecliptic). It is not
// the GCRS (the inertial frame of the IAU 2006 transforms), not J2000 and not
// true-of-date; the only well-defined way out of it is through the Earth:
// rotate by the IAU 1982 Greenwich MEAN sidereal time (GMST82, the Earth's
// rotation angle measured from that mean equinox along the true equator) to
// the pseudo-Earth-fixed frame (PEF), then apply polar motion (the small
// offset between the rotation axis and the crust; Vallado, Crawford,
// Hujsak & Kelso 2006, AIAA 2006-6753, sec. "TEME"; Vallado, Fundamentals of
// Astrodynamics and Applications, 4th ed., sec. 3.7):
//
//     r_ITRF = W(xp, yp) R3(GMST82(UT1)) r_TEME
//     v_ITRF = W [ R3(GMST82) v_TEME - omega x r_PEF ],  omega = 7.29211514670698e-5 (1 - LOD/86400) rad/s
//
// UT1 (Earth rotation time), polar motion xp, yp and the length of day LOD
// come from the Earth orientation parameter (EOP) lookup (its hold/raise
// policy applies) or are passed explicitly. No "kinematic"
// equation-of-equinoxes terms are added: Vallado's published example
// (AIAA 2006-6753) is reproduced without them (to < 1 mm, test_sgp4).
// From the ITRF (the Earth-fixed frame) the library's IAU 2006 transform
// reaches the GCRS.
// The ITRF realization of the result is that of the EOP series (finals2000A:
// ITRF2020 currently) -- millimeters, far below a TLE's kilometer error.
// =============================================================================

#include "../coordinates/earth_rotation.hpp"
#include "../core/constants.hpp"
#include "../core/types.hpp"
#include "../time/eop_data.hpp"
#include "../time/epoch.hpp"
#include "sgp4.hpp"

namespace ballistic::sgp4 {

namespace detail_teme {

struct Rotation {
    Eigen::Matrix3d r3;   // TEME -> PEF
    Eigen::Matrix3d w;    // PEF -> ITRF
    double omega_rad_s;   // Earth rotation rate of date
};

// `eop` supplies UT1-UTC, polar motion and LOD (normally the EOP lookup's
// sample at t).
[[nodiscard]] inline Rotation rotation(const Epoch& t, const time::EopSample& e) {
    // UT1 Julian date as an exact split: JD_TT = tt_day + tt_seconds / 86400.
    const double ut1_sec = t.tt_seconds() + t.seconds_from_tt(TimeScale::UTC) + e.ut1_utc_s;
    const double gmst = detail_sgp4::gmst1982(static_cast<double>(t.tt_day()) +
                                              ut1_sec / constants::SECONDS_PER_DAY);
    const double c = std::cos(gmst), s = std::sin(gmst);
    Rotation rot;
    rot.r3 << c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0;
    rot.w = coords::polar_motion_matrix(e.xp_arcsec * constants::ARCSEC_TO_RAD,
                                        e.yp_arcsec * constants::ARCSEC_TO_RAD, 0.0);
    // The IAU 1982 rotation rate, part of the TEME -> PEF definition.
    rot.omega_rad_s = 7.29211514670698e-5 *  // guard-exempt(CONST-LITERAL): defines TEME
                      (1.0 - e.lod_ms * 1e-3 / constants::SECONDS_PER_DAY);
    return rot;
}

[[nodiscard]] inline time::EopSample eop_at(const Epoch& t) {
    return time::EOPData::instance().lookup(t.mjd(TimeScale::UTC));
}

}  // namespace detail_teme

/// Convert a TEME state (position m, velocity m/s) at epoch `t` to ITRF, using the Earth
/// orientation values `eop` (UT1-UTC, xp, yp, LOD). The returned velocity is relative to the
/// rotating Earth. Throws DataCoverageError only through the UTC conversion of `t`
/// (before 1961).
[[nodiscard]] inline StateVector teme_to_itrf(const StateVector& teme, const Epoch& t,
                                              const time::EopSample& eop) {
    const auto rot = detail_teme::rotation(t, eop);
    const Vector3d r_pef = rot.r3 * teme.position;
    const Vector3d v_pef = rot.r3 * teme.velocity - Vector3d(0.0, 0.0, rot.omega_rad_s).cross(r_pef);
    return StateVector(rot.w * r_pef, rot.w * v_pef);
}

/// Same, with the Earth orientation from EOPData::lookup at `t`. Throws DataCoverageError
/// when no EOP table is loaded (unless EopPolicy::AssumeZero) or `t` is more than 30 days
/// outside it.
[[nodiscard]] inline StateVector teme_to_itrf(const StateVector& teme, const Epoch& t) {
    return teme_to_itrf(teme, t, detail_teme::eop_at(t));
}

/// Convert an ITRF state (position m, velocity m/s relative to the rotating Earth) at epoch
/// `t` to TEME, using `eop`; the inverse of teme_to_itrf.
[[nodiscard]] inline StateVector itrf_to_teme(const StateVector& itrf, const Epoch& t,
                                              const time::EopSample& eop) {
    const auto rot = detail_teme::rotation(t, eop);
    const Vector3d r_pef = rot.w.transpose() * itrf.position;
    const Vector3d v_pef = rot.w.transpose() * itrf.velocity;
    const Vector3d v_inert = v_pef + Vector3d(0.0, 0.0, rot.omega_rad_s).cross(r_pef);
    return StateVector(rot.r3.transpose() * r_pef, rot.r3.transpose() * v_inert);
}

/// Same, with the Earth orientation from EOPData::lookup at `t` (same errors as the
/// teme_to_itrf overload without `eop`).
[[nodiscard]] inline StateVector itrf_to_teme(const StateVector& itrf, const Epoch& t) {
    return itrf_to_teme(itrf, t, detail_teme::eop_at(t));
}

}  // namespace ballistic::sgp4
