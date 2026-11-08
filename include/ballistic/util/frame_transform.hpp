#pragma once

// =============================================================================
// Frame Transformation Utilities
// =============================================================================
//
// Conversions between the two frames the library works in:
//   - ITRF (International Terrestrial Reference Frame): Earth-fixed, rotates
//     with the Earth; also called ECEF (Earth-centered, Earth-fixed).
//   - GCRS (Geocentric Celestial Reference System): Earth-centered but not
//     rotating; the equations of motion are integrated here.
//
// Every function here is GCRS <-> ITRF by the IAU 2006/2000A model (with
// Earth orientation parameters, EOP). The names avoid "ECI" (Earth-centered
// inertial) on purpose: it also names the true-of-date frame, which differs
// from GCRS by the precession since 2000 (~22 km at LEO in 2013).
//
// All functions look up the EOP table and throw DataCoverageError when none
// is loaded (unless EopPolicy::AssumeZero) or the date is more than 30 days
// outside it.
//
// =============================================================================

#include "../core/types.hpp"
#include "../time/epoch.hpp"
#include "../coordinates/earth_rotation.hpp"
#include "../coordinates/iau_transforms.hpp"

namespace ballistic {
namespace util {

// =============================================================================
// State Vector Transformations
// =============================================================================

/// Convert an ITRF state (position m, velocity m/s relative to the rotating Earth) at
/// `epoch` to GCRS. `itrf` names the ITRF realization the input is expressed in.
[[nodiscard]] inline StateVector itrf_to_gcrs(
    const StateVector& state_ecef,
    const Epoch& epoch,
    coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020)
{
    return coords::itrf_to_gcrs_full(state_ecef, epoch, itrf);
}

/// Convert a GCRS state at `epoch` to ITRF (velocity relative to the rotating Earth), in
/// realization `itrf`.
[[nodiscard]] inline StateVector gcrs_to_itrf(
    const StateVector& state_gcrs,
    const Epoch& epoch,
    coords::ITRFRealization itrf = coords::ITRFRealization::ITRF2020)
{
    return coords::gcrs_to_itrf_full(state_gcrs, epoch, itrf);
}

// =============================================================================
// Acceleration Transformations
// =============================================================================

/// Convert a force acceleration in GCRS [m/s^2] to the second time derivative of the ITRF
/// position (d²r_itrf/dt²), given the ITRF state at `epoch`. Adds the Coriolis (-2ω×v) and
/// centrifugal (-ω×(ω×r)) terms of the rotating frame (ω = the Earth's rotation rate at
/// `epoch` about the ITRF Z axis, the same rate the state transforms use; the spin axis is
/// within ~0.5 arcsec of Z), so the result is the true second derivative of the Earth-fixed
/// position, which accurate quintic Hermite interpolation of Earth-fixed trajectories needs.
[[nodiscard]] inline Vector3d acceleration_to_itrf(
    const Vector3d& acc_gcrs,
    const StateVector& state_ecef,
    const Epoch& epoch)
{
    Vector3d acc_ecef = coords::gcrs_to_itrf_vector_full(acc_gcrs, epoch);
    const Vector3d omega(0.0, 0.0, coords::earth_rotation_rate(epoch));
    acc_ecef -= 2.0 * omega.cross(state_ecef.velocity);
    acc_ecef -= omega.cross(omega.cross(state_ecef.position));
    return acc_ecef;
}

/// Inverse of acceleration_to_itrf: convert d²r_itrf/dt² [m/s^2] to a GCRS force
/// acceleration, given the ITRF state at `epoch` (removes the Coriolis and centrifugal
/// terms, then rotates to GCRS).
[[nodiscard]] inline Vector3d acceleration_to_gcrs(
    const Vector3d& acc_ecef,
    const StateVector& state_ecef,
    const Epoch& epoch)
{
    const Vector3d omega(0.0, 0.0, coords::earth_rotation_rate(epoch));
    Vector3d acc_itrf = acc_ecef
        + 2.0 * omega.cross(state_ecef.velocity)
        + omega.cross(omega.cross(state_ecef.position));
    return coords::itrf_to_gcrs_vector_full(acc_itrf, epoch);
}

} // namespace util
} // namespace ballistic
