#pragma once

// =============================================================================
// estimate_ea_bspline — fit an empirical acceleration to dense observations
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration fitted to
// observations of a satellite and added to the physical force model, to absorb
// whatever the model gets wrong (mis-modeled drag, unknown thrust, ...).
//
// estimate_ea_bspline(...) is one overload set; the kind of data is the type of
// the samples argument (PositionSamples / VelocitySamples / AccelSamples), so
// passing velocities to a position fit does not compile. It fits a B-spline to
// the residuals with a regularization schedule — a penalty that keeps the EA
// smooth, with weight λ — chosen by one of ea::IRGN, ea::FixedLambda, ea::GCV,
// ea::CV (below), then by default runs Gauss-Newton (GN) refinement
// (gn_refine.hpp). For several sensors at once, or noisy data, use
// estimate_ea_smoother (ea_smoother.hpp) instead.
//
// Frames: all observations and the start state are ECEF (Earth-centered,
// Earth-fixed, i.e. the ITRF); the returned EA is in GCRS (the Earth-centered
// inertial frame the propagator integrates in). The force model
// (TrajectoryConfig) is a separate argument.
//
// Settings (EABSplineConfig) include:
//   - `regularization`: ea::IRGN{}, ea::FixedLambda{}, ea::GCV{} or ea::CV{}
//   - `gn`: std::optional<GNRefine> — empty = no GN refinement; present = run
//     it with these settings (the default)
// =============================================================================

#include "ea_method_advisor.hpp"
#include "bspline_accel_estimator.hpp"
#include "gn_refine.hpp"
#include "ea_decomposition.hpp"
#include "multisensor.hpp"
#include "ea_smoother.hpp"
#include "detail/calibration_constants.hpp"
#include "detail/integration.hpp"
#include "../util/frame_transform.hpp"
#include "../core/errors.hpp"
#include <variant>
#include <optional>
#include <algorithm>
#include <iostream>

namespace ballistic {
namespace calibration {

// =============================================================================
// How the regularization weight λ is chosen (larger λ = smoother EA)
// =============================================================================

namespace ea {

/// Constant λ for every iteration.
struct FixedLambda {
    double lambda = 1e-4;
};

/// IRGN: iteratively regularized Gauss-Newton. λ starts large and shrinks
/// each iteration, λ_k = max(lambda_min, lambda_0 · decay^k), so the first
/// iterations fit only the broad shape and later ones the detail. The default.
struct IRGN {
    double lambda_0 = 1e-4;
    double decay = 0.5;
    double lambda_min = 1e-12;
};

/// GCV: generalized cross-validation picks λ at each iteration from the data
/// alone (it estimates how well each λ would predict left-out points), searched
/// within [lam_min, lam_max] (0 < lam_min < lam_max, else ConfigError). Needs no
/// noise level.
struct GCV {
    double lam_min = 1e-12;
    double lam_max = 1e-1;
};

/// CV: cross-validation grid search. Every combination of settings on a grid
/// is fitted on part of the data and scored on how well it predicts the rest;
/// the best combination is then fitted on all data. The data are split into
/// contiguous time blocks with gaps around each held-out block ("hv-block"
/// cross-validation, Racine 2000), because neighboring samples are correlated
/// and would otherwise leak into the score. Slow, but the best automatic
/// choice for noisy data when the right settings are unknown.
///
/// `mode` selects what is searched:
///   IRGN          — 4D grid (knot spacing × λ₀ × decay × λ_min), 495
///                   combinations; λ shrinks over iterations. Default.
///   FixedLambda   — 2D grid (knot spacing × λ), 99 combinations, λ constant.
///                   About 5× fewer fits, but measured (without GN) to fail on
///                   noisy position and velocity data; only acceleration data
///                   was on par with IRGN. Use it only for acceleration data,
///                   and check the result on your own data.
///
///
/// `grid` selects the grid size:
///   Full (default) — the grids above.
///   Coarse         — knot spacing {1, 2, 5, 10} s × λ_min {1e-12, 1e-8, 1e-4}
///                    with λ₀ = 0.1, decay 0.5 (12 combinations; FixedLambda:
///                    the same spacings × λ {1e-12, 1e-8, 1e-4, 1}, 16). About
///                    40× faster than Full; it finds the right region of the
///                    grid, not the best point inside it.
/// When the chosen knot spacing or final λ lies on the edge of its grid, a
/// ConfigHint diagnostic names the edge: the best setting may lie outside the
/// grid.
///
/// Block layout (the defaults were checked on a 184 s reentry-glide arc; each
/// block, arc length / hv_n_blocks, must be longer than the time over which
/// the residual errors stay correlated):
///   hv_n_blocks      — number of contiguous time blocks (default 18, ≈ 10 s each)
///   hv_test_blocks   — indices of the held-out blocks; empty → blocks 2, 7, 12, ...
///                      (every 5th, never the last)
///   hv_gap_blocks    — blocks left out of training on each side of a held-out block
/// The layout must hold out at least one block: the automatic pattern needs
/// hv_n_blocks ≥ 4, and explicit indices must lie in [0, hv_n_blocks)
/// (ConfigError otherwise).
struct CV {
    enum class Mode { IRGN, FixedLambda };
    enum class Grid { Full, Coarse };

    int hv_n_blocks = 18;
    std::vector<int> hv_test_blocks;
    int hv_gap_blocks = 1;
    Mode mode = Mode::IRGN;
    Grid grid = Grid::Full;
};

} // namespace ea

using LambdaSelection = std::variant<ea::IRGN, ea::FixedLambda, ea::GCV, ea::CV>;

// =============================================================================
// Optional Gauss-Newton refinement
// =============================================================================

/// Settings for Gauss-Newton (GN) refinement: after the B-spline fit, the EA
/// is adjusted so that the orbit propagated with it matches the observed
/// positions as closely as possible (see gn_refine.hpp). Set through
/// `EABSplineConfig::gn`; a present value means "run GN" (the default).
///
/// Per kind of data (docs/empirical_acceleration_estimation.md):
///   positions     — keep it on (large improvement)
///   velocities    — keep it on (measured better over several noise seeds);
///                   fitted against positions integrated from the velocities
///   accelerations — not applicable; the result reports `SkippedAccelerationCircular`
struct GNRefine {
    /// Marker: limit the integrator step to the estimator's knot spacing —
    /// one of the three choices of `rk45_max_step` below.
    struct MatchKnotSpacing {};

    int max_iterations = 30;   // maximum GN iterations
    int sample_stride = 20;    // fit every N-th observation
    int ea_subsample = 1;      // keep every N-th EA node as a parameter (1 = all)
    double fd_delta = 1e-2;    // finite-difference perturbation of an EA value [m/s²]

    /// Largest step of the RK45 integrator (adaptive Runge-Kutta, Dormand-Prince
    /// 4(5)) during GN propagations:
    ///   std::monostate (default) — no override; use force_model.rk45
    ///                              (fully adaptive — recommended)
    ///   MatchKnotSpacing{}       — the estimator's knot_spacing_s (measured
    ///                              to make results worse)
    ///   double                   — this value [s]
    std::variant<std::monostate, MatchKnotSpacing, double> rk45_max_step = {};

    /// How GN gets its derivatives. true (default): the state transition
    /// matrix (STM) — the sensitivity S(t) = ∂(state at t)/∂(EA node values),
    /// integrated alongside the orbit in one propagation. false: finite
    /// differences (FD), one extra propagation per parameter.
    /// STM is faster but requires linear interpolation between EA nodes;
    /// for the default degree-5 spline, FD can use cubic interpolation and was
    /// measured more accurate on position data. Set false for a one-off run
    /// where accuracy matters most. Cross-validation grids use STM unless told
    /// otherwise.
    bool use_stm = true;
    double stm_dt_s = 0.5;       // step [s] for integrating the sensitivity equations
    double stm_fd_dx = 1.0;      // position perturbation [m] for ∂a/∂r
    double stm_fd_dv = 1e-3;     // velocity perturbation [m/s] for ∂a/∂v
    bool stm_verify = false;     // first iteration: also compute FD derivatives, print difference
    bool stm_verbose = false;    // print the STM computation's internal trace

    /// Smoothness penalty during refinement, the same kind as in the B-spline
    /// fit: λ_r·N·h^(2k+1)·Σ((k-th difference of the node values)/hᵏ)²·h
    /// (h = node spacing, N = observations used), added to both the solved
    /// equations and the step-acceptance test, so refinement minimizes the same
    /// objective as the fit before it. k = reg_deriv_order (empty = automatic:
    /// 3 for positions, 1 for velocities, as in EABSplineConfig).
    /// reg_lambda = 0 turns it off; on noisy data that still keeps the
    /// propagated orbit close, but the EA itself was measured to go badly
    /// wrong point by point.
    double reg_lambda = 1.0;
    std::optional<int> reg_deriv_order;   // empty = automatic (see above)
};

// =============================================================================
// B-spline estimator settings
// =============================================================================

/// Settings for `estimate_ea_bspline`.
struct EABSplineConfig {
    /// How λ is chosen. Default IRGN. For noisy acceleration data without a
    /// known noise level use `ea::GCV{}`; for noisy data whose best settings
    /// are unknown, `ea::CV{}` (cross-validation grid search).
    LambdaSelection regularization = ea::IRGN{};

    /// Observation noise standard deviation σ in the data's units (m, m/s or
    /// m/s²). If > 0, the fit stops once the residual RMS is ≤ 1.5σ.
    double noise_sigma = 0.0;

    /// Time between spline knots [s]; smaller = the EA can change faster.
    /// For noisy acceleration data, widen it as σ grows.
    double knot_spacing_s = 2.0;

    /// B-spline degree: 5 (quintic, the default for all data kinds) or 3 (cubic).
    int degree = 5;

    /// Maximum outer iterations; the fit also stops after 5 iterations in a
    /// row without improvement.
    int max_iterations = 50;

    /// Longest arc fitted [s]. Reduced automatically to the last observation
    /// time + 10 s, so the default is safe on short arcs; set it below the
    /// data span only to fit deliberately less of the data.
    double max_duration_s = 3600.0;

    /// Which derivative of the spline the smoothness penalty acts on. Empty =
    /// automatic: 3 for positions, 1 for velocities/accelerations. Must not
    /// exceed `degree`.
    std::optional<int> reg_deriv_order;

    /// Gauss-Newton refinement. On by default because it measured better on
    /// noisy and clean positions and on velocities. Set std::nullopt to skip it
    /// (about half the run time). Not applied to acceleration data, which has
    /// no positions to fit; the result then reports
    /// `gn_status = SkippedAccelerationCircular`.
    std::optional<GNRefine> gn = GNRefine{};

    /// How the step length of each update is searched: true = bracket the best
    /// length from 5 trials, then refine with Brent's method (parabolic
    /// interpolation); false = try 24 fixed lengths. Brent usually needs fewer
    /// propagations per iteration. Measured:
    ///   positions:     more accurate and faster
    ///   accelerations: more accurate and faster
    ///   velocities:    same mean error over several seeds but more spread
    ///                  from seed to seed. Harmless in
    ///                  cross-validation, which averages many fits; consider
    ///                  false for a single velocity fit.
    bool brent_line_search = true;

    /// Print one diagnostic line per iteration (fit and GN) to stdout, and a
    /// note when max_duration_s is reduced to the data.
    bool verbose = false;
};

// =============================================================================
// Internal: convert the public settings to the internal ones
// =============================================================================

namespace detail {

// Copy the public `LambdaSelection` choice into an internal config's
// (lambda_method, lambda, lambda_decay, lambda_min) fields:
//   FixedLambda → FIXED with λ
//   IRGN        → DECREASING with λ₀ / decay / λ_min
//   GCV         → GCV searched within [lam_min, lam_max]; with
//                 radar_fallback = true, IRGN defaults instead
//   CV          → IRGN defaults (CV is handled before this is called; this
//                 is only a fallback)
template <typename IC>
inline void apply_regularization(const LambdaSelection& reg, IC& ic,
                                 bool radar_fallback) {
    if (std::holds_alternative<ea::FixedLambda>(reg)) {
        ic.lambda_method = detail::LambdaMethod::FIXED;
        ic.lambda = std::get<ea::FixedLambda>(reg).lambda;
    } else if (std::holds_alternative<ea::IRGN>(reg)) {
        const auto& irgn = std::get<ea::IRGN>(reg);
        ic.lambda_method = detail::LambdaMethod::DECREASING;
        ic.lambda = irgn.lambda_0;
        ic.lambda_decay = irgn.decay;
        ic.lambda_min = irgn.lambda_min;
    } else if (!radar_fallback && std::holds_alternative<ea::GCV>(reg)) {
        const auto& gcv = std::get<ea::GCV>(reg);
        if (!(gcv.lam_min > 0.0) || !(gcv.lam_max > gcv.lam_min)) {
            throw ConfigError(
                "ea::GCV{lam_min = " + std::to_string(gcv.lam_min) + ", lam_max = " +
                std::to_string(gcv.lam_max) + "}: GCV searches lambda on a log scale "
                "between the two, so it needs 0 < lam_min < lam_max (default 1e-12 to 0.1)");
        }
        ic.lambda_method = detail::LambdaMethod::GCV;
        ic.gcv_lambda_min = gcv.lam_min;
        ic.gcv_lambda_max = gcv.lam_max;
    } else {
        // Non-radar CV (dispatched before translate; safety fallback)
        // OR radar GCV/CV (not implemented): IRGN defaults.
        ic.lambda_method = detail::LambdaMethod::DECREASING;
        ic.lambda = 1e-4;
        ic.lambda_decay = 0.5;
        ic.lambda_min = 1e-12;
    }
}

inline detail::BSplineAccelConfig translate_config(
    const EABSplineConfig& cfg,
    const TrajectoryConfig& force_model,
    int default_reg_order)
{
    detail::BSplineAccelConfig ic;
    ic.trajectory_config = force_model;
    ic.knot_spacing_s = cfg.knot_spacing_s;
    ic.degree = cfg.degree;
    ic.max_iterations = cfg.max_iterations;
    ic.max_duration_s = cfg.max_duration_s;
    ic.noise_sigma = cfg.noise_sigma;
    ic.min_altitude_m = 0.0;
    ic.reg_deriv_order = cfg.reg_deriv_order.value_or(default_reg_order);
    ic.brent_line_search = cfg.brent_line_search;
    ic.verbose = cfg.verbose;

    apply_regularization(cfg.regularization, ic, /*radar_fallback=*/false);

    return ic;
}

// Build the internal GNRefineConfig (gn_refine.hpp) from a public GNRefine,
// the force model and the estimator's knot spacing (needed when
// rk45_max_step is MatchKnotSpacing).
inline GNRefineConfig translate_gn_config(
    const GNRefine& gn,
    const TrajectoryConfig& force_model,
    double knot_spacing_s,
    double max_duration_s,
    bool verbose = false,
    int reg_order_auto = 3)   // penalty derivative order used when reg_deriv_order is empty
{
    GNRefineConfig gc;
    gc.reg_lambda = gn.reg_lambda;
    gc.reg_order = gn.reg_deriv_order.value_or(reg_order_auto);
    gc.max_iterations = gn.max_iterations;
    gc.fd_delta = gn.fd_delta;
    gc.sample_stride = gn.sample_stride;
    gc.ea_subsample = gn.ea_subsample;
    gc.max_duration_s = max_duration_s;
    gc.min_altitude_m = 0.0;

    gc.trajectory_config = force_model;
    if (std::holds_alternative<GNRefine::MatchKnotSpacing>(gn.rk45_max_step))
        gc.trajectory_config.rk45.max_step_size = knot_spacing_s;
    else if (const double* step = std::get_if<double>(&gn.rk45_max_step))
        gc.trajectory_config.rk45.max_step_size = *step;
    // std::monostate → no override (default; adaptive step)

    gc.use_stm = gn.use_stm;
    gc.stm_dt_s = gn.stm_dt_s;
    gc.stm_fd_dx = gn.stm_fd_dx;
    gc.stm_fd_dv = gn.stm_fd_dv;
    gc.stm_verify = gn.stm_verify;
    gc.stm_verbose = gn.stm_verbose;
    gc.verbose = verbose;
    return gc;
}

// Convert Earth-fixed (ECEF) velocity observations to inertial (GCRS)
// velocities. Converting a velocity between a rotating and a fixed frame
// needs the position too (the Earth-rotation term ω×r, about 465 m/s at the
// equator), so positions are obtained by integrating the velocities from
// pos0_ecef with the trapezoidal rule. This needs no reference trajectory and
// works even when the EA is large (a reentry glide).
inline std::vector<std::pair<Epoch, Vector3d>> vel_ecef_to_gcrs(
    const std::vector<std::pair<Epoch, Vector3d>>& vel_ecef,
    const Vector3d& pos0_ecef)
{
    std::vector<std::pair<Epoch, Vector3d>> vel_gcrs;
    vel_gcrs.reserve(vel_ecef.size());
    if (vel_ecef.empty()) return vel_gcrs;

    Vector3d p = pos0_ecef;
    for (size_t i = 0; i < vel_ecef.size(); ++i) {
        if (i > 0) {
            double dt = vel_ecef[i].first - vel_ecef[i - 1].first;
            p += 0.5 * (vel_ecef[i - 1].second + vel_ecef[i].second) * dt;
        }
        StateVector sv_ecef{p, vel_ecef[i].second};
        auto sv_gcrs = coords::itrf_to_gcrs_full(sv_ecef, vel_ecef[i].first);
        vel_gcrs.emplace_back(vel_ecef[i].first, sv_gcrs.velocity);
    }
    return vel_gcrs;
}

// Convert accelerations observed in Earth-fixed coordinates (d²r_ecef/dt²,
// which include the apparent Coriolis and centrifugal terms of the rotating
// frame) to inertial GCRS accelerations. The velocities and positions those
// terms need come from integrating the data from vel0_ecef / pos0_ecef.
inline std::vector<std::pair<Epoch, Vector3d>> acc_ecef_to_gcrs(
    const std::vector<std::pair<Epoch, Vector3d>>& acc_ecef,
    const Vector3d& vel0_ecef,
    const Vector3d& pos0_ecef)
{
    std::vector<std::pair<Epoch, Vector3d>> acc_gcrs;
    acc_gcrs.reserve(acc_ecef.size());
    if (acc_ecef.empty()) return acc_gcrs;

    Vector3d v = vel0_ecef;
    Vector3d p = pos0_ecef;
    for (size_t i = 0; i < acc_ecef.size(); ++i) {
        if (i > 0) {
            double dt = acc_ecef[i].first - acc_ecef[i - 1].first;
            Vector3d v_new = v + 0.5 * (acc_ecef[i - 1].second + acc_ecef[i].second) * dt;
            p += 0.5 * (v + v_new) * dt;
            v = v_new;
        }
        StateVector sv_ecef{p, v};
        acc_gcrs.emplace_back(acc_ecef[i].first,
            util::acceleration_to_gcrs(acc_ecef[i].second, sv_ecef,
                                      acc_ecef[i].first));
    }
    return acc_gcrs;
}

} // namespace detail
} // namespace calibration
} // namespace ballistic

// Cross-validation implementation (needs the types above; included at file scope)
#include "detail/ea_cv.hpp"

namespace ballistic {
namespace calibration {
namespace detail {

// Build the cross-validation settings from EABSplineConfig + ea::CV. The
// block layout comes from `cv` (empty hv_test_blocks → blocks 2, 7, 12, ...);
// a layout that holds out no block, or names a block outside
// [0, hv_n_blocks), throws ConfigError (it could not cross-validate anything).
// Every GNRefine field is passed on to the GN runs inside cross-validation;
// an empty cfg.gn means no GN there either.
inline CVDeploymentConfig translate_cv_dep_config(
    const EABSplineConfig& cfg, const ea::CV& cv)
{
    CVDeploymentConfig dep;
    if (cfg.gn) {
        dep.gn_iterations      = cfg.gn->max_iterations;
        dep.gn_sample_stride   = cfg.gn->sample_stride;
        dep.gn_ea_subsample    = cfg.gn->ea_subsample;
        dep.fd_delta           = cfg.gn->fd_delta;
        // Internal encoding: 0 = adaptive, -1 = match knot spacing, > 0 = this step [s].
        if (std::holds_alternative<GNRefine::MatchKnotSpacing>(cfg.gn->rk45_max_step))
            dep.rk45_max_step_size = -1.0;
        else if (const double* step = std::get_if<double>(&cfg.gn->rk45_max_step))
            dep.rk45_max_step_size = *step;
        else
            dep.rk45_max_step_size = 0.0;
        dep.gn_use_stm    = cfg.gn->use_stm;
        dep.gn_stm_dt_s   = cfg.gn->stm_dt_s;
        dep.gn_stm_fd_dx  = cfg.gn->stm_fd_dx;
        dep.gn_stm_fd_dv  = cfg.gn->stm_fd_dv;
        dep.gn_stm_verify = cfg.gn->stm_verify;
        dep.gn_stm_verbose = cfg.gn->stm_verbose;
        dep.gn_reg_lambda  = cfg.gn->reg_lambda;
        // -1 = automatic (3 for positions, 1 for velocities), as outside CV.
        dep.gn_reg_order   = cfg.gn->reg_deriv_order.value_or(-1);
    } else {
        dep.gn_iterations = 0;
    }
    dep.hv_n_blocks   = cv.hv_n_blocks;
    dep.hv_gap_blocks = cv.hv_gap_blocks;
    if (cv.hv_test_blocks.empty()) {
        // CVDeploymentConfig::hv_test_blocks defaults to {2, 7, 12} (for 18
        // blocks); clear it first so the pattern follows hv_n_blocks.
        dep.hv_test_blocks.clear();
        for (int i = 2; i < cv.hv_n_blocks - 1; i += 5)
            dep.hv_test_blocks.push_back(i);
    } else {
        dep.hv_test_blocks = cv.hv_test_blocks;
    }
    if (dep.hv_test_blocks.empty()) {
        throw ConfigError(
            "ea::CV with hv_n_blocks = " + std::to_string(cv.hv_n_blocks) +
            " holds out no block (the automatic pattern takes every 5th block from block "
            "2, never the last), so nothing would be cross-validated: use hv_n_blocks >= 4 "
            "or list the held-out blocks in hv_test_blocks");
    }
    for (int b : dep.hv_test_blocks) {
        if (b < 0 || b >= cv.hv_n_blocks) {
            throw ConfigError(
                "ea::CV hv_test_blocks names block " + std::to_string(b) +
                ", outside the " + std::to_string(cv.hv_n_blocks) +
                " blocks (indices 0 to " + std::to_string(cv.hv_n_blocks - 1) + ")");
        }
    }
    dep.coarse_grid = (cv.grid == ea::CV::Grid::Coarse);
    dep.cv_mode = (cv.mode == ea::CV::Mode::FixedLambda)
        ? CVDeploymentConfig::CVMode::FixedLambda
        : CVDeploymentConfig::CVMode::IRGN;
    dep.verbose = cfg.verbose;
    return dep;
}

/// Run GN refinement on an estimator result if `gn` is set and the result
/// has a non-zero EA; otherwise do nothing. Replaces `result.empirical_accel`,
/// rebuilds `collocation_history` from an orbit propagated with the refined EA
/// (one extra propagation), and sets `result.gn_status = GNStatus::Ran`.
///
/// `targets` are the ECEF positions [m] to fit: the observations themselves
/// for position data, positions integrated from the velocities for velocity data.
inline void apply_optional_gn_refinement(
    EAResult& result,
    const std::optional<GNRefine>& gn,
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& targets,
    const TrajectoryConfig& force_model,
    double knot_spacing_s,
    double max_duration_s,
    bool verbose,
    int reg_order_auto = 3)
{
    if (!gn || !result.empirical_accel || result.empirical_accel->is_zero())
        return;
    auto gc = translate_gn_config(*gn, force_model, knot_spacing_s,
                                  max_duration_s, verbose, reg_order_auto);
    auto refined = refine_ea_gn(epoch0, position_ecef, velocity_ecef, targets,
                                result.empirical_accel, gc);
    if (refined) {
        result.empirical_accel = refined;
        result.gn_status = GNStatus::Ran;
        refresh_history_with_ea(result, epoch0, position_ecef, velocity_ecef,
                                force_model, max_duration_s);
    }
}

} // namespace detail

// =============================================================================
// Observation samples, one type per kind of data
// =============================================================================
//
// The kind of data is carried by the type, so passing velocities to a position
// fit does not compile. All samples are ECEF (Earth-fixed) and in time order;
// each carries its own time. Radar observations (RadarObs, multisensor.hpp)
// go to estimate_ea_smoother instead.

/// Observed positions: (time, ECEF position [m]).
struct PositionSamples {
    std::vector<std::pair<Epoch, Vector3d>> ecef;  // positions [m]
};

/// Observed velocities: (time, ECEF velocity [m/s], the rate of change of the
/// ECEF position).
struct VelocitySamples {
    std::vector<std::pair<Epoch, Vector3d>> ecef;  // velocities [m/s]
};

/// Observed accelerations: (time, d²r/dt² of the ECEF position [m/s²]) — the
/// satellite's full acceleration including gravity, as seen in Earth-fixed
/// coordinates. (Not an accelerometer's specific force; that is AccelObs.)
struct AccelSamples {
    std::vector<std::pair<Epoch, Vector3d>> ecef;  // accelerations [m/s²]
};

/// Remove outliers from dense samples before fitting.
///
/// Assumes the noise is mostly Gaussian with std dev `sigma` (data units)
/// but a fraction p = `outlier_prob` of samples are outliers with std dev
/// k·sigma, k = `outlier_scale`: (1−p)·N(0,σ²) + p·N(0,(kσ)²). Each sample is
/// compared with the median of the samples within ±0.25 s; it is dropped when
/// any component differs by more than z*·σ, where z* = k·√(2·ln((1−p)k/p)/(k²−1))
/// is the point at which the sample is more likely an outlier than not (≈ 3.3σ
/// for the defaults, and nearly independent of p and k, so a rough guess is fine).
/// Measured with 5% of samples at 10σ, 100 Hz, it brought acceleration-data fits
/// close to removing exactly the true outliers, and improved position data.
/// Outliers are dropped, not clipped: pulling them toward the local median was
/// measured to corrupt position data that curves noticeably within the window.
///
/// Below ~4 samples per second the ±0.25 s window holds too few samples to
/// judge; the input is then returned unchanged and `*out_of_envelope` is set.
/// `*n_rejected` (optional) receives the number dropped. Never throws.
[[nodiscard]] inline std::vector<std::pair<Epoch, Vector3d>>
reject_heavy_tails(const std::vector<std::pair<Epoch, Vector3d>>& samples,
                   double sigma, double outlier_prob = 0.05,
                   double outlier_scale = 10.0,
                   size_t* n_rejected = nullptr,
                   bool* out_of_envelope = nullptr)
{
    // The ±0.25 s median needs several samples per window. Below ~4 Hz the
    // window holds only the sample itself, every difference is 0, and the
    // filter would silently do nothing. A wider window is not the fix (the
    // curvature problem above returns), so return the input unchanged and
    // say so through out_of_envelope.
    if (out_of_envelope) *out_of_envelope = false;
    if (samples.size() >= 2) {
        const double span =
            samples.back().first - samples.front().first;
        const double median_rate =
            span > 0.0 ? (static_cast<double>(samples.size()) - 1.0) / span
                       : 0.0;
        if (median_rate < 4.0) {
            if (out_of_envelope) *out_of_envelope = true;
            if (n_rejected) *n_rejected = 0;
            return samples;
        }
    }
    const double p = outlier_prob, k = outlier_scale;
    const double z_star =
        k * std::sqrt(2.0 * std::log((1.0 - p) * k / p) / (k * k - 1.0));
    std::vector<std::pair<Epoch, Vector3d>> out;
    out.reserve(samples.size());
    if (n_rejected) *n_rejected = 0;
    std::vector<double> buf;
    for (size_t i = 0; i < samples.size(); ++i) {
        // Samples within ±0.25 s of sample i (assumes time order: scans
        // outward by index while the time difference allows).
        const Epoch& ti = samples[i].first;
        size_t lo = i, hi = i + 1;
        while (lo > 0 && ti - samples[lo - 1].first <= 0.25) --lo;
        while (hi < samples.size() && samples[hi].first - ti <= 0.25) ++hi;
        Vector3d med;
        for (int c = 0; c < 3; ++c) {
            buf.clear();
            for (size_t j = lo; j < hi; ++j)
                buf.push_back(samples[j].second(c));
            std::nth_element(buf.begin(), buf.begin() + buf.size() / 2,
                             buf.end());
            med(c) = buf[buf.size() / 2];
        }
        const Vector3d z = (samples[i].second - med) / sigma;
        if (z.cwiseAbs().maxCoeff() > z_star) {
            if (n_rejected) (*n_rejected)++;
            continue;
        }
        out.push_back(samples[i]);
    }
    return out;
}

namespace detail {



// Margin [s] added to the last observation time when reducing max_duration_s,
// so the "propagation reached 0.9 × max_duration_s" check keeps some room
// (same value as in ea_cv.hpp).
inline constexpr double MAX_DURATION_SPAN_SLACK_S = 10.0;

// Latest observation time, in seconds after epoch0 (any input order). 0 for
// empty data.
[[nodiscard]] inline double obs_end_rel_s(
    const std::vector<std::pair<Epoch, Vector3d>>& data, const Epoch& epoch0)
{
    double end = 0.0;
    for (const auto& [t, v] : data) end = std::max(end, t - epoch0);
    return end;
}

// Reduce cfg.max_duration_s to the last observation time + slack. Why: the fit
// rejects any trial EA whose propagation covers less than 0.9 × max_duration_s
// (to catch propagations that stopped early). Left at 3600 s on a short arc,
// every trial would be rejected, the fit would stay at its first step, and
// velocity/acceleration estimates would be 100–1000× off. Never increases the
// value: a deliberately shorter setting is kept.
template <typename Config>
[[nodiscard]] Config clamp_max_duration(Config cfg, double obs_end_s)
{
    const double clamped = obs_end_s + MAX_DURATION_SPAN_SLACK_S;
    if (obs_end_s > 0.0 && clamped < cfg.max_duration_s) {
        if (cfg.verbose) {
            std::cerr << "[EA] max_duration_s " << cfg.max_duration_s
                      << " exceeds observation arc; clamping to " << clamped
                      << " s (obs end " << obs_end_s << " s + "
                      << MAX_DURATION_SPAN_SLACK_S << " s slack)\n";
        }
        cfg.max_duration_s = clamped;
    }
    return cfg;
}

} // namespace detail

// =============================================================================
// Gauss-Newton refinement on its own
// =============================================================================

/// Refine any EA (from any source) so that the orbit started from
/// (`pos_ecef` [m], `vel_ecef` [m/s], ECEF, at `epoch0`) and propagated with
/// force model `force_model` plus the EA best matches `position_data` (time,
/// ECEF position [m]). Observations more than `max_duration_s` from epoch0 are
/// ignored. Uses the public GNRefine settings; GNRefine::MatchKnotSpacing has
/// no knot spacing here and acts as "no override". Returns the refined
/// tabulated EA, or `initial_ea` unchanged when there is nothing to refine
/// (see gn_refine.hpp). Exceptions from propagation propagate.
[[nodiscard]] inline std::shared_ptr<forces::EmpiricalAcceleration> refine_ea_gn(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& position_data,
    std::shared_ptr<forces::EmpiricalAcceleration> initial_ea,
    const TrajectoryConfig& force_model,
    const GNRefine& gn = {},
    double max_duration_s = 3600.0)
{
    GNRefine g = gn;
    if (std::holds_alternative<GNRefine::MatchKnotSpacing>(g.rk45_max_step))
        g.rk45_max_step = std::monostate{};
    return refine_ea_gn(epoch0, pos_ecef, vel_ecef, position_data,
                        std::move(initial_ea),
                        detail::translate_gn_config(g, force_model,
                                                    /*knot_spacing_s=*/0.0,
                                                    max_duration_s));
}

// =============================================================================
// B-spline estimator — public functions
// =============================================================================
//
// Common arguments: `epoch0` start time; `position_ecef` [m] and
// `velocity_ecef` [m/s] the start state in ECEF (Earth-fixed); `force_model`
// the dynamics; `config_in` the settings. Return an EAResult whose EA is in
// GCRS; failures (too few observations, ...) are reported in
// `error_message`, not thrown. Exceptions from propagation (e.g. missing
// Earth-orientation or space-weather data) propagate.

/// Estimate an EA from ECEF position observations. Defaults: degree 5, knots
/// every 2 s, IRGN λ schedule, then GN refinement.
[[nodiscard]] inline EAResult estimate_ea_bspline(
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const PositionSamples& obs,
    const TrajectoryConfig& force_model,
    const EABSplineConfig& config_in = {})
{
    const EABSplineConfig config = detail::clamp_max_duration(
        config_in, detail::obs_end_rel_s(obs.ecef, epoch0));
    const auto& position_data_ecef = obs.ecef;
    // Cross-validation: grid search; each candidate is scored after GN, as it
    // will be used.
    if (std::holds_alternative<ea::CV>(config.regularization)) {
        const auto& cv = std::get<ea::CV>(config.regularization);
        return detail::cv_bspline_position(
            epoch0, position_ecef, velocity_ecef, position_data_ecef,
            force_model, config.degree, config.max_iterations,
            config.max_duration_s, config.noise_sigma,
            detail::translate_cv_dep_config(config, cv));
    }

    auto ic = detail::translate_config(config, force_model, /*reg_order=*/3);
    auto result = detail::estimate_empirical_acceleration_bspline(
        epoch0, position_ecef, velocity_ecef, position_data_ecef, ic);

    detail::apply_optional_gn_refinement(
        result, config.gn, epoch0, position_ecef, velocity_ecef,
        position_data_ecef, force_model, config.knot_spacing_s,
        config.max_duration_s, config.verbose);
    return result;
}

/// Estimate an EA from ECEF velocity observations. GN refinement (on by
/// default) fits positions integrated from the velocities and measurably helps
/// (over several noise seeds).
[[nodiscard]] inline EAResult estimate_ea_bspline(
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const VelocitySamples& obs,
    const TrajectoryConfig& force_model,
    const EABSplineConfig& config_in = {})
{
    const EABSplineConfig config = detail::clamp_max_duration(
        config_in, detail::obs_end_rel_s(obs.ecef, epoch0));
    const auto& velocity_data_ecef = obs.ecef;
    if (std::holds_alternative<ea::CV>(config.regularization)) {
        const auto& cv = std::get<ea::CV>(config.regularization);
        return detail::cv_bspline_velocity(
            epoch0, position_ecef, velocity_ecef, velocity_data_ecef,
            force_model, config.degree, config.max_iterations,
            config.max_duration_s, config.noise_sigma,
            detail::translate_cv_dep_config(config, cv));
    }

    auto ic = detail::translate_config(config, force_model, /*reg_order=*/-1);
    ic.lambda_drift = detail::VEL_LAMBDA_DRIFT;

    auto vel_gcrs = detail::vel_ecef_to_gcrs(velocity_data_ecef, position_ecef);

    auto result = detail::estimate_empirical_acceleration_bspline_from_velocity(
        epoch0, position_ecef, velocity_ecef, vel_gcrs, ic);

    if (config.gn && !velocity_data_ecef.empty()) {
        // Integrate the velocities to positions in the inertial GCRS frame
        // (reusing vel_gcrs from above): integrating the ECEF velocities
        // directly was measured to drift much more over a reentry glide.


        auto integrated_positions = detail::trapezoidal_vel_gcrs_to_pos_ecef(
            vel_gcrs, position_ecef, force_model.itrf);
        detail::apply_optional_gn_refinement(
            result, config.gn, epoch0, position_ecef, velocity_ecef,
            integrated_positions, force_model, config.knot_spacing_s,
            config.max_duration_s, config.verbose, /*reg_order_auto=*/1);
    }
    return result;
}

/// Estimate an EA from ECEF acceleration observations (AccelSamples). GN
/// refinement is not run — there are no observed positions to fit, only a
/// propagation of the EA itself — and, if `config.gn` is set, the result says
/// so with `gn_status = SkippedAccelerationCircular`.
[[nodiscard]] inline EAResult estimate_ea_bspline(
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const AccelSamples& obs,
    const TrajectoryConfig& force_model,
    const EABSplineConfig& config_in = {})
{
    const EABSplineConfig config = detail::clamp_max_duration(
        config_in, detail::obs_end_rel_s(obs.ecef, epoch0));
    const auto& acceleration_data_ecef = obs.ecef;
    if (std::holds_alternative<ea::CV>(config.regularization)) {
        const auto& cv = std::get<ea::CV>(config.regularization);
        auto result = detail::cv_bspline_acceleration(
            epoch0, position_ecef, velocity_ecef, acceleration_data_ecef,
            force_model, config.degree, config.max_iterations,
            config.max_duration_s, config.noise_sigma,
            detail::translate_cv_dep_config(config, cv));
        if (config.gn) result.gn_status = GNStatus::SkippedAccelerationCircular;
        return result;
    }

    auto ic = detail::translate_config(config, force_model, /*reg_order=*/-1);

    auto acc_gcrs = detail::acc_ecef_to_gcrs(
        acceleration_data_ecef, velocity_ecef, position_ecef);

    auto result = detail::estimate_empirical_acceleration_bspline_from_acceleration(
        epoch0, position_ecef, velocity_ecef, acc_gcrs, ic);

    // No GN for acceleration data: its only targets would be a propagation of
    // the EA itself. Report that on the result.
    if (config.gn) result.gn_status = GNStatus::SkippedAccelerationCircular;

    return result;
}


} // namespace calibration
} // namespace ballistic
