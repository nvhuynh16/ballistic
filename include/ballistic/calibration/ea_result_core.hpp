#pragma once

// =============================================================================
// ResultCore — the fields every empirical-acceleration estimator result shares
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration, fitted to
// observations of a satellite, that is added to the physical force model to
// account for whatever the model gets wrong. The library has several EA
// estimators; their result types — `EAResult` (`estimate_ea_bspline`),
// `EASmootherResult` (`estimate_ea_smoother`) and `EASpecResult`
// (`estimate_ea`) — all derive from `ResultCore`, so a caller can test success
// the same way whichever estimator produced the result.
//
// Derived types add their estimator-specific fields. The Python bindings refer
// to inherited members as `&Derived::member`, which works for base-class members.
// =============================================================================

#include "../forces/empirical_acceleration.hpp"

#include <memory>
#include <string>

namespace ballistic {
namespace calibration {

/// Fields shared by every EA-estimator result.
///
/// Success is split into two questions because "the estimator ran" and "it
/// found something to correct" differ: an EA of exactly zero (the force model
/// already fits the data) is a success, not a failure.
///   - `succeeded()`      — the estimator ran without error and produced an EA.
///   - `has_correction()` — ...and that EA is not zero.
/// Check `succeeded()` before using a result at all; check `has_correction()`
/// before steps that need a non-zero EA, such as Gauss-Newton refinement or
/// fitting a ballistic coefficient from the EA.
struct ResultCore {
    /// The estimated correction acceleration; null when the estimator failed.
    std::shared_ptr<forces::EmpiricalAcceleration> empirical_accel;
    /// Number of outer iterations the estimator ran.
    int iterations_used = 0;
    /// Why the estimator failed; empty on success.
    std::string error_message;

    /// True when the estimator ran without error and returned an EA.
    [[nodiscard]] bool succeeded() const noexcept {
        return error_message.empty() && empirical_accel != nullptr;
    }

    /// True when `succeeded()` and the EA is not identically zero.
    [[nodiscard]] bool has_correction() const noexcept {
        return succeeded() && !empirical_accel->is_zero();
    }
};

} // namespace calibration
} // namespace ballistic
