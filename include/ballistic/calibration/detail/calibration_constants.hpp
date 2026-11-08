#pragma once

// =============================================================================
// Tuning constants used across the EA estimators
// =============================================================================
//
// Tuning constants shared by the empirical-acceleration (EA) estimators. An EA is a small
// correction acceleration fitted to observations and added to the force model. Keeping the
// constants here documents each reason once and makes a tuning change a one-line edit.
// Constants that belong to a single call site (e.g. Eigen sparse-zero tolerances) stay there.
//
// Terms used below:
//   IRGN  - iteratively regularized Gauss-Newton: the outer loop of the B-spline EA fit, a
//           Gauss-Newton (GN) least-squares solve whose regularization weight λ shrinks
//           every iteration.
//   GN refine - a final Gauss-Newton pass that adjusts the EA against the observations.
//   RK45  - the adaptive Dormand-Prince Runge-Kutta 4(5) integrator.
//   POS / VEL - estimator runs fitted to position or velocity observations.
//   SSE   - sum of squared errors.
// =============================================================================

namespace ballistic {
namespace calibration {
namespace detail {

/// Maximum number of RK45 steps in one trial propagation, well above what a healthy arc
/// needs. The cap stops an oscillating EA candidate, which
/// can force ever-smaller steps, from looping without advancing time.
inline constexpr int MAX_PROP_STEPS = 1000;

/// A trial propagation that covers less than this fraction of the requested arc is rejected.
/// It catches runs cut short (e.g. by the altitude cutoff), whose error would otherwise be
/// scored over a shorter arc than its competitors and look unfairly good.
inline constexpr double TRUNCATION_GUARD = 0.9;

/// The IRGN loop stops once the observation RMS error falls to
/// `EARLY_STOP_SIGMA_MULT * noise_sigma` (the caller's stated noise level) or below. 1.5 was
/// chosen empirically: below it the fit starts chasing noise, above it real signal is still
/// left in the residual.
inline constexpr double EARLY_STOP_SIGMA_MULT = 1.5;

/// Number of consecutive non-improving iterations after which IRGN and GN refine stop. The
/// tracked measure is the observation SSE for IRGN, the SSE of the position proxy (integrated
/// velocity) for VEL runs, and the position SSE for GN refine.
inline constexpr int STALL_LIMIT = 5;

/// On VEL runs, IRGN keeps the best iteration by the SSE of a position proxy (the velocity
/// observations integrated to positions). Only every `VEL_PROXY_STRIDE`-th sample is stored:
/// at 200–300 Hz data that is 2–3 Hz, enough to rank iterations without large storage.
inline constexpr int VEL_PROXY_STRIDE = 100;

/// Weight of a Tikhonov (ridge) penalty on the size of the spline values, ||f||², used on VEL
/// runs. It keeps the integrated-velocity position proxy from drifting without bound when the
/// main regularization is weak. Tuned for a decreasing IRGN schedule starting at λ₀ = 1,
/// decay = 0.5 (the public default starts at λ₀ = 1e-4). Do not change it casually: with a
/// fixed λ below 1e-2 this penalty dominates and suppresses the EA (measured: a large proxy
/// RMS error). Every VEL run, cross-validated or IRGN, depends on it.

inline constexpr double VEL_LAMBDA_DRIFT = 1e-2;

/// Smallest Jacobian column norm used when scaling the normal equations of a GN or
/// Levenberg-Marquardt (LM, damped Gauss-Newton) solve. A column with ‖J_j‖ below it is
/// treated as zero (its parameter is not updated) instead of producing a huge, ill-conditioned
/// step.
inline constexpr double COL_NORM_FLOOR = 1e-30;

/// Index of one EA parameter in the flat parameter vector used by the GN Jacobians: all x
/// values first, then all y, then all z, so `j = comp * n_nodes + node` with comp 0/1/2 = x/y/z
/// and `node` the EA table node. gn_refine.hpp and gn_stm.hpp share this layout.
[[nodiscard]] inline constexpr int param_idx(int comp, int node, int n_nodes) noexcept {
    return comp * n_nodes + node;
}

}  // namespace detail
}  // namespace calibration
}  // namespace ballistic
