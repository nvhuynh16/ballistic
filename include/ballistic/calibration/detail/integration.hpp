#pragma once

// =============================================================================
// Trapezoidal integration helpers for the empirical-acceleration (EA) estimators
// =============================================================================
//
// The estimators sometimes need to turn velocity samples into positions, or acceleration
// samples into velocities or positions, by the trapezoidal rule. Frames: ECEF is the
// Earth-fixed, rotating frame (here the ITRF); GCRS is the non-rotating (inertial) frame.
//
//   trapezoidal_vel_to_pos_ecef
//       ECEF velocity → ECEF position, summed in the rotating frame. Measured to drift much
//       more than the GCRS version below over a glide re-entry. Fine for cross-validation
//       hold-out points or short arcs; do NOT use the result as position targets for a
//       Gauss-Newton (GN) refinement.
//
//   trapezoidal_acc_to_vel_ecef
//   trapezoidal_acc_to_pos_ecef
//       The same for ECEF accelerations (single and double integration). Same drift; same
//       uses (cross-validation stand-ins, starting guesses).
//
//   trapezoidal_vel_gcrs_to_pos_ecef
//       Drift-free: integrates GCRS velocity in GCRS and converts each running position back
//       to ECEF, as the velocity branch of estimate_bspline_impl
//       (bspline_accel_estimator.hpp) does. Use this for GN position targets.
//
// All inputs are ECEF except for `trapezoidal_vel_gcrs_to_pos_ecef`, whose velocities must
// already be in GCRS (e.g. from `detail::vel_ecef_to_gcrs` in ea_estimator.hpp).
// =============================================================================

#include "../../core/types.hpp"
#include "../../time/epoch.hpp"
#include "../../coordinates/iau_transforms.hpp"
#include "../../coordinates/itrf_transforms.hpp"

#include <utility>
#include <vector>

namespace ballistic {
namespace calibration {
namespace detail {

/// Integrate ECEF velocity samples [m/s] to ECEF positions [m] by the trapezoidal rule,
/// starting from `pos0_ecef` at the first sample's time. Returns one position per sample
/// (empty for empty input). Drifts more than trapezoidal_vel_gcrs_to_pos_ecef (measured).
[[nodiscard]] inline std::vector<std::pair<Epoch, Vector3d>>
trapezoidal_vel_to_pos_ecef(
    const std::vector<std::pair<Epoch, Vector3d>>& vel_ecef,
    const Vector3d& pos0_ecef)
{
    std::vector<std::pair<Epoch, Vector3d>> out;
    if (vel_ecef.empty()) return out;
    out.reserve(vel_ecef.size());

    Vector3d p = pos0_ecef;
    out.emplace_back(vel_ecef.front().first, p);
    for (std::size_t i = 1; i < vel_ecef.size(); ++i) {
        const double dt = vel_ecef[i].first - vel_ecef[i - 1].first;
        p += 0.5 * (vel_ecef[i - 1].second + vel_ecef[i].second) * dt;
        out.emplace_back(vel_ecef[i].first, p);
    }
    return out;
}

/// Integrate ECEF acceleration samples [m/s²] to ECEF velocities [m/s] by the trapezoidal
/// rule, starting from `vel0_ecef`. One velocity per sample. Drifts like
/// trapezoidal_vel_to_pos_ecef.
[[nodiscard]] inline std::vector<std::pair<Epoch, Vector3d>>
trapezoidal_acc_to_vel_ecef(
    const std::vector<std::pair<Epoch, Vector3d>>& acc_ecef,
    const Vector3d& vel0_ecef)
{
    std::vector<std::pair<Epoch, Vector3d>> out;
    if (acc_ecef.empty()) return out;
    out.reserve(acc_ecef.size());

    Vector3d v = vel0_ecef;
    out.emplace_back(acc_ecef.front().first, v);
    for (std::size_t i = 1; i < acc_ecef.size(); ++i) {
        const double dt = acc_ecef[i].first - acc_ecef[i - 1].first;
        v += 0.5 * (acc_ecef[i - 1].second + acc_ecef[i].second) * dt;
        out.emplace_back(acc_ecef[i].first, v);
    }
    return out;
}

/// Integrate ECEF acceleration samples [m/s²] twice to ECEF positions [m], from `vel0_ecef`
/// and `pos0_ecef`. The drift of a double integration grows with the square of time; short
/// arcs only.
[[nodiscard]] inline std::vector<std::pair<Epoch, Vector3d>>
trapezoidal_acc_to_pos_ecef(
    const std::vector<std::pair<Epoch, Vector3d>>& acc_ecef,
    const Vector3d& vel0_ecef,
    const Vector3d& pos0_ecef)
{
    auto vel = trapezoidal_acc_to_vel_ecef(acc_ecef, vel0_ecef);
    return trapezoidal_vel_to_pos_ecef(vel, pos0_ecef);
}

/// Drift-free velocity → position: integrate GCRS velocity samples [m/s] trapezoidally in
/// GCRS, starting from `pos0_ecef` [m] (ECEF, converted to GCRS at the first sample's time),
/// and convert each running position back to ECEF in the ITRF realization `itrf`. Returns one
/// ECEF position per sample. Same computation as the `vel_integrated_positions` block in
/// estimate_bspline_impl. Use it for GN position targets: the plain ECEF sum was measured to
/// drift much more on a glide re-entry.

[[nodiscard]] inline std::vector<std::pair<Epoch, Vector3d>>
trapezoidal_vel_gcrs_to_pos_ecef(
    const std::vector<std::pair<Epoch, Vector3d>>& vel_gcrs,
    const Vector3d& pos0_ecef,
    coords::ITRFRealization itrf)
{
    std::vector<std::pair<Epoch, Vector3d>> out;
    if (vel_gcrs.empty()) return out;
    out.reserve(vel_gcrs.size());

    const Epoch t0 = vel_gcrs.front().first;
    Vector3d p_gcrs = coords::itrf_to_gcrs_full(pos0_ecef, t0, itrf);
    out.emplace_back(t0, pos0_ecef);
    for (std::size_t i = 1; i < vel_gcrs.size(); ++i) {
        const double dt = vel_gcrs[i].first - vel_gcrs[i - 1].first;
        p_gcrs += 0.5 * (vel_gcrs[i - 1].second + vel_gcrs[i].second) * dt;
        const Vector3d p_ecef = coords::gcrs_to_itrf_full(p_gcrs, vel_gcrs[i].first, itrf);
        out.emplace_back(vel_gcrs[i].first, p_ecef);
    }
    return out;
}

}  // namespace detail
}  // namespace calibration
}  // namespace ballistic
