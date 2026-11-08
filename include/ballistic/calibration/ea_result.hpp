#pragma once

// =============================================================================
// EAResult — what `estimate_ea_bspline` returns
// =============================================================================
//
// EA = empirical acceleration: a small correction acceleration fitted to
// observations and added to the force model. The types that split an EA into
// a drag part and a remainder live in ea_decomposition.hpp.
// =============================================================================

#include "ea_result_core.hpp"
#include "../forces/empirical_acceleration.hpp"
#include "../trajectory.hpp"
#include "../trajectory_interpolator.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace ballistic {
namespace calibration {

/// What happened to the optional Gauss-Newton (GN) refinement step. GN
/// refinement re-fits the EA by repeatedly propagating the orbit with it and
/// correcting it toward the observations; the estimator reports whether it ran
/// and, if not, why.
enum class GNStatus {
    Disabled,                        // Caller did not request GN
    Ran,                             // GN executed
    SkippedAccelerationCircular      // Accelerometer data: there are no positions to fit, so GN
                                     // would only fit the EA to a propagation of itself
};

/// Result of `estimate_ea_bspline`. The first three fields and the success
/// tests come from `ResultCore`.
///
///   empirical_accel       — the estimated EA (null on failure)
///   iterations_used       — number of outer iterations run
///   collocation_history   — the trajectory that goes with the EA, sampled on
///                           the spline's knot times; filled when the estimator
///                           completes with a non-zero EA. It always belongs to
///                           the returned `empirical_accel`: after GN refinement
///                           it is the orbit propagated with the refined EA.
///                           Without GN, for position data it is "force model +
///                           fitted residual spline" (follows the data without
///                           a re-propagation's drift); otherwise the orbit
///                           re-propagated with the EA. This is the trajectory
///                           refine_beta_from_ea / decompose_to_beta expect.

///   final_trajectory      — the propagated trajectory behind
///                           `collocation_history`. Kept alive because
///                           `final_interp` refers to it.
///   final_interp          — interpolator over `final_trajectory`. Cross-validation
///                           scoring reuses it instead of propagating again.
///                           Null when no final propagation ran (an empty
///                           result, or a solver that does not build one).
///   gn_status             — what happened to the optional GN refinement
///   lambda_used           — the regularization weight λ at the end, when the
///                           method used one (larger λ = smoother EA)
///   sigma_sq_estimate     — per-component observation noise variance estimated
///                           by generalized cross-validation (GCV), when used
///   obs_rmse              — root-mean-square observation residual at the end
///   error_message         — non-empty on failure
///
/// Check `succeeded()` before using the result and `has_correction()` before
/// GN refinement or a ballistic-coefficient fit, which need a non-zero EA.
struct EAResult : ResultCore {
    std::vector<HistoryEntry> collocation_history;
    std::shared_ptr<Trajectory> final_trajectory;
    std::shared_ptr<TrajectoryInterpolator> final_interp;
    GNStatus gn_status = GNStatus::Disabled;

    std::optional<double> lambda_used;
    std::optional<double> sigma_sq_estimate;
    std::optional<double> obs_rmse;
};

}  // namespace calibration
}  // namespace ballistic
