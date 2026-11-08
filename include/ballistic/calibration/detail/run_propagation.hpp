#pragma once

// =============================================================================
// Shared trajectory-propagation primitive
// =============================================================================
//
// One helper used by every calibration routine that needs a trial trajectory: propagate a
// start state with a candidate empirical acceleration (EA: a fitted correction acceleration
// added to the force model) for a given duration, stopping early below an altitude, and
// return an interpolator over the result.
//
// Every call is limited to `MAX_PROP_STEPS` steps. The cap guards against a bad EA that
// forces tiny steps and never advances time; healthy runs stay well under it.

//
// The trajectory and interpolator are returned on the heap (unique_ptr) because some callers
// keep them beyond the call; the interpolator refers to the trajectory, so keep both alive.
// =============================================================================

#include "calibration_constants.hpp"
#include "../../core/types.hpp"
#include "../../coordinates/geodetic.hpp"
#include "../../forces/empirical_acceleration.hpp"
#include "../../time/epoch.hpp"
#include "../../trajectory.hpp"
#include "../../trajectory_interpolator.hpp"

#include <memory>
#include <utility>

namespace ballistic {
namespace calibration {
namespace detail {

/// Propagate a trial trajectory.
/// `epoch0`, `pos_ecef` [m], `vel_ecef` [m/s]: start state, Earth-fixed (ITRF) position and
/// velocity. `base_cfg`: force model; `ea` (may be null = keep the config's own EA) replaces
/// its empirical acceleration. Propagation stops after `max_duration_s` [s], when geodetic
/// altitude drops below `min_altitude_m` [m], or after `MAX_PROP_STEPS` steps, whichever comes
/// first; `backward = true` propagates backward in time. Returns the trajectory and an
/// interpolator over its history. The trajectory is marked internal, so its notes about
/// running past the EA's time span stay on the trajectory instead of reaching the user.
/// Throws whatever the `Trajectory` constructor throws for an unusable start state.
[[nodiscard]] inline std::pair<std::unique_ptr<Trajectory>, std::unique_ptr<TrajectoryInterpolator>>
run_propagation_arc(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const TrajectoryConfig& base_cfg,
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea,
    double max_duration_s,
    double min_altitude_m,
    bool backward = false)
{
    TrajectoryConfig cfg = base_cfg;
    if (ea) cfg.empirical_accel = ea;
    auto traj = std::make_unique<Trajectory>(epoch0, pos_ecef, vel_ecef, cfg);
    traj->mark_internal();

    if (backward) {
        for (int si = 0; si < MAX_PROP_STEPS; ++si) {
            auto sr = traj->step_backward();
            if (epoch0 - sr.epoch >= max_duration_s) break;
            if (coords::ecef_to_geodetic(sr.position).altitude < min_altitude_m) break;
        }
    } else {
        for (int si = 0; si < MAX_PROP_STEPS; ++si) {
            if (traj->epoch() - epoch0 >= max_duration_s) break;
            auto sr = traj->step();
            if (coords::ecef_to_geodetic(sr.position).altitude < min_altitude_m) break;
        }
    }

    auto ti = std::make_unique<TrajectoryInterpolator>(*traj);
    return {std::move(traj), std::move(ti)};
}

}  // namespace detail
}  // namespace calibration
}  // namespace ballistic
