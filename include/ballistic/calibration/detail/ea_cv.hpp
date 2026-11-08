#pragma once

// Cross-validation (CV) for choosing the settings of the B-spline empirical-acceleration
// fit. An empirical acceleration (EA) is a small correction acceleration fitted to
// observations and added to the force model. The B-spline fit runs an IRGN loop (iteratively
// regularized Gauss-Newton: a Gauss-Newton least-squares fit whose smoothing weight λ starts
// at λ₀ and is multiplied by `decay` each iteration down to λ_min). Included from
// ea_estimator.hpp after the config structs are defined.
//
// The search tries every combination of (knot spacing, λ₀, decay, λ_min) on a grid. Each
// candidate is fitted on training data and scored on held-out data, using hv-block CV
// (Racine 2000, J. Econometrics 99): the arc is cut into time blocks, a few interior blocks
// are held out, and the blocks next to them are dropped as a gap so that errors correlated
// in time cannot leak from training into test. The score is the RMS position error of the
// re-propagated trajectory at the held-out times; for velocity data the held-out "positions"
// are the velocities integrated once, for acceleration data integrated twice.
//
// `split_kfold` / `merge_train_folds` are simple test utilities (used by
// tests/integration/test_ea_cv.cpp and test_ea_estimator.cpp), not part of the CV used by the
// estimator, whose entry points are cv_bspline_position / cv_bspline_velocity /
// cv_bspline_acceleration.
//
// How well the CV score ranked candidates by true error, measured on a simulated glide
// re-entry:
//   Position:     almost perfectly
//   Velocity:     almost perfectly (via the integrated position proxy)
//   Acceleration: only partly on clean data, not at all on noisy data (use generalized
//                 cross-validation, GCV, instead)

#include "../bspline_accel_estimator.hpp"
#include "../gn_refine.hpp"
#include "integration.hpp"
#include "run_propagation.hpp"
#include "../../trajectory.hpp"
#include "../../trajectory_interpolator.hpp"
#include "../../util/frame_transform.hpp"
#include "../../core/diagnostics.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ballistic {
namespace calibration {
namespace detail {


// =========================================================================
// Hyperparameter grid
// =========================================================================

/// One candidate setting of the B-spline fit: knot spacing [s] and the IRGN λ schedule
/// (start value λ₀, per-iteration decay factor, floor λ_min).
struct CVHyperParams {
    double knot_spacing;
    double lambda_0;
    double decay;
    double lambda_min;
};

/// Two held-out scores closer than this [m of position RMS error] count as tied. It is far
/// above floating-point rounding differences (~1e-13 relative) and far below the noise of
/// any realistic data set. See `pick_best_grid_idx`.
constexpr double CV_TIE_EPS_M = 1e-3;

/// Index of the best (lowest) held-out score in `holdout`, with a deterministic tie-break.
///
/// A plain "strictly less than" choice would depend on the last bits of the scores: the same
/// code compiled into different programs can produce scores that differ in the final digit
/// (different floating-point summation order from OpenMP or compiler optimization). When two
/// grid points score almost the same, such a flip changes the chosen settings and visibly
/// moved the final error between builds with identical inputs.
///
/// Rule: scores within `tie_eps` [m] are ties, and a tie goes to the larger knot spacing
/// (when the data cannot tell two fits apart, take the smoother one). Ties at equal knot
/// spacing keep the earlier grid entry.
[[nodiscard]] inline int pick_best_grid_idx(
    const std::vector<double>& holdout,
    const std::vector<CVHyperParams>& grid,
    double tie_eps = CV_TIE_EPS_M) noexcept
{
    int best_idx = 0;
    for (int i = 1; i < static_cast<int>(grid.size()); ++i) {
        const double diff = holdout[i] - holdout[best_idx];
        if (diff < -tie_eps) {
            best_idx = i;                              // strictly better by margin
        } else if (std::abs(diff) <= tie_eps) {
            if (grid[i].knot_spacing > grid[best_idx].knot_spacing) {
                best_idx = i;                          // Occam's razor on tie
            }
            // else keep first-seen (deterministic sub-tie fallback)
        }
    }
    return best_idx;
}

/// The default search grid for the IRGN fit: 11 knot spacings × λ₀ × decay × λ_min, keeping
/// only λ_min ≤ λ₀ (495 combinations). Checked on a simulated glide re-entry at 100 and 200 Hz
/// for position, velocity and acceleration data.
/// Why the coarse knot spacings 8–16 s: on noisy position data with Gauss-Newton (GN)
/// refinement afterwards, 10 s was best over many random seeds, clearly better than 5 s.
/// Without GN, finer knots win; GN removes the bias of a coarse spline, so the optimum moves
/// coarser, and a grid stopping at 5 s left the choice stuck at the grid edge. Above ~16 s a
/// degree-5 spline can no longer follow the test EA's 35 s wide bumps (20 s left a bias and
/// an error floor).
[[nodiscard]] inline std::vector<CVHyperParams> build_cv_grid() {
    static constexpr double knots[]      = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0, 12.0, 16.0};
    static constexpr double lambda_0s[]  = {1e0, 1e-1, 1e-2};
    static constexpr double decays[]     = {0.3, 0.5, 0.7};
    static constexpr double lambda_mins[] = {1e-12, 1e-8, 1e-6, 1e-4, 1e-2};

    std::vector<CVHyperParams> grid;
    grid.reserve(495);
    for (double ks : knots)
        for (double l0 : lambda_0s)
            for (double q : decays)
                for (double lm : lambda_mins)
                    if (lm <= l0)
                        grid.push_back({ks, l0, q, lm});
    return grid;
}

/// The search grid when λ is held constant instead of following a schedule: 11 knot
/// spacings × 9 log-spaced λ values from 1e-12 (where IRGN usually ends) to 1 (heavy
/// smoothing) = 99 combinations, about 5× fewer than the IRGN grid. `decay` and `lambda_min`
/// are not used on this path; they are set to the neutral values 1.0 and λ.
[[nodiscard]] inline std::vector<CVHyperParams> build_cv_grid_fixed_lambda() {
    static constexpr double knots[]   = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 8.0, 10.0, 12.0, 16.0};
    static constexpr double lambdas[] = {1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1e0};

    std::vector<CVHyperParams> grid;
    grid.reserve(99);
    for (double ks : knots)
        for (double lam : lambdas)
            grid.push_back({ks, /*lambda_0=*/lam, /*decay=*/1.0, /*lambda_min=*/lam});
    return grid;
}

/// The coarse IRGN grid (ea::CV::Grid::Coarse): 4 knot spacings × λ_min ∈ {1e-12, 1e-8, 1e-4}
/// with λ₀ = 1e-1 and decay 0.5 = 12 combinations, about 40× fewer fits than the full grid.
/// It spans the same knot-spacing range (1–10 s) and the same λ_min decades as the full grid,
/// so it finds the right region but not the best point inside it.
[[nodiscard]] inline std::vector<CVHyperParams> build_cv_grid_coarse() {
    static constexpr double knots[]       = {1.0, 2.0, 5.0, 10.0};
    static constexpr double lambda_mins[] = {1e-12, 1e-8, 1e-4};
    std::vector<CVHyperParams> grid;
    grid.reserve(12);
    for (double ks : knots)
        for (double lm : lambda_mins)
            grid.push_back({ks, /*lambda_0=*/1e-1, /*decay=*/0.5, lm});
    return grid;
}

/// The coarse fixed-λ grid: 4 knot spacings × λ ∈ {1e-12, 1e-8, 1e-4, 1e0} = 16 combinations.
[[nodiscard]] inline std::vector<CVHyperParams> build_cv_grid_fixed_lambda_coarse() {
    static constexpr double knots[]   = {1.0, 2.0, 5.0, 10.0};
    static constexpr double lambdas[] = {1e-12, 1e-8, 1e-4, 1e0};
    std::vector<CVHyperParams> grid;
    grid.reserve(16);
    for (double ks : knots)
        for (double lam : lambdas)
            grid.push_back({ks, /*lambda_0=*/lam, /*decay=*/1.0, /*lambda_min=*/lam});
    return grid;
}

/// The grid a cross-validation run searches: IRGN or fixed λ, full or coarse.
[[nodiscard]] inline std::vector<CVHyperParams> select_cv_grid(bool fixed_lambda, bool coarse) {
    if (fixed_lambda) return coarse ? build_cv_grid_fixed_lambda_coarse() : build_cv_grid_fixed_lambda();
    return coarse ? build_cv_grid_coarse() : build_cv_grid();
}

/// The grid edges the selected candidate sits on, as text ("" when it is interior on both
/// axes checked). Two axes are checked: knot spacing and the final smoothing weight (λ_min
/// for IRGN, λ for fixed λ), the two that set how smooth the EA ends up. λ₀ and decay only
/// shape the path of the IRGN schedule and are not checked. An axis with a single value has
/// no edge.
[[nodiscard]] inline std::string cv_grid_edges(
    const std::vector<CVHyperParams>& grid, int best_idx, bool fixed_lambda)
{
    if (grid.empty() || best_idx < 0 || best_idx >= static_cast<int>(grid.size())) return {};
    auto edge = [&](auto get, const char* name, const char* unit) -> std::string {
        double lo = get(grid[0]), hi = lo;
        for (const auto& g : grid) { lo = std::min(lo, get(g)); hi = std::max(hi, get(g)); }
        const double v = get(grid[best_idx]);
        if (lo == hi || (v != lo && v != hi)) return {};
        std::ostringstream os;
        os << name << " " << v << unit << " (the " << (v == lo ? "smallest" : "largest")
           << " on the grid)";
        return os.str();
    };
    const std::string ks = edge([](const CVHyperParams& g) { return g.knot_spacing; },
                                "knot spacing", " s");
    const std::string lam = edge([](const CVHyperParams& g) { return g.lambda_min; },
                                 fixed_lambda ? "lambda" : "lambda_min", "");
    if (ks.empty()) return lam;
    if (lam.empty()) return ks;
    return ks + " and " + lam;
}

/// Report (ConfigHint, never an error) when cross-validation chose a candidate on a grid
/// edge: the best setting may then lie outside the grid, and the score cannot say how far.
inline void report_cv_grid_edge(
    const std::vector<CVHyperParams>& grid, int best_idx, bool fixed_lambda,
    const char* data_kind)
{
    const std::string edges = cv_grid_edges(grid, best_idx, fixed_lambda);
    if (edges.empty()) return;
    diagnostics::report(
        diagnostics::Category::ConfigHint,
        std::string("ea::CV on ") + data_kind + " data chose " + edges + " of its " +
        std::to_string(grid.size()) + "-candidate grid. The best setting may lie outside "
        "the grid, where cross-validation did not look: if the result matters, check it "
        "with the parameter set by hand beyond that edge (EABSplineConfig::knot_spacing_s "
        "or ea::IRGN / ea::FixedLambda).");
}

// =========================================================================
// Deployment-aware CV configuration
// =========================================================================

/// Settings passed from the public EABSplineConfig into the cross-validation loop: the GN
/// refinement settings and the hv-block layout. Each candidate is scored after running the
/// same steps the final fit will run (IRGN, then GN if enabled); otherwise CV would choose
/// settings that are best for a different procedure (Racine 2000; confirmed by measurement).
/// translate_cv_dep_config (ea_estimator.hpp) fills it, including the state-transition-matrix
/// (STM) settings; make_gn_config_from_dep turns it back into a GNRefineConfig.
struct CVDeploymentConfig {
    /// Number of GN refinement iterations after IRGN (0 = no GN). When > 0, every candidate
    /// is GN-refined before it is scored, as in the final fit. Never applied to acceleration
    /// data (see cv_bspline_acceleration).
    int gn_iterations = 0;
    int gn_sample_stride = 20;
    int gn_ea_subsample = 1;
    double fd_delta = 1e-2;
    /// Order of the time derivative of the EA that GN's smoothing (Tikhonov) penalty acts on,
    /// so GN minimizes the same kind of objective as the spline fit. -1 (default) = automatic,
    /// the rule the estimator uses outside cross-validation: 3 for position data, 1 for
    /// velocity data. 0 = no smoothing penalty in GN. Copied from GNRefine::reg_deriv_order.
    int gn_reg_order = -1;
    /// Weight of GN's smoothing penalty (GNRefine::reg_lambda; 0 = off).
    double gn_reg_lambda = 1.0;
    /// Largest RK45 integrator step [s] in GN's propagations. 0 = keep the force model's own
    /// setting (recommended); -1 = the knot spacing (measured to make results worse); > 0 =
    /// that value.
    double rk45_max_step_size = 0.0;

    /// hv-block layout (Racine 2000). Defaults checked on a 184 s glide re-entry: 18 blocks of
    /// ~10 s, blocks 2, 7 and 12 held out (all interior), one block (10 s) dropped on each side
    /// of them. What matters is that the gap is longer than the time over which residual
    /// errors stay correlated. For other arc lengths, scale n_blocks so blocks
    /// stay that long, and keep the test blocks interior with a one-block gap.
    int hv_n_blocks = 18;
    std::vector<int> hv_test_blocks = {2, 7, 12};
    int hv_gap_blocks = 1;

    /// Fit used for each candidate: IRGN searches the 4-D schedule grid (495 combinations);
    /// FixedLambda holds λ constant and searches a 2-D grid (99 combinations, 5× fewer fits).
    /// Set through ea::CV::mode.
    enum class CVMode { IRGN, FixedLambda };
    CVMode cv_mode = CVMode::IRGN;

    /// Grid size (ea::CV::grid): false = the full grid (495 / 99 candidates), true = the
    /// coarse grid (build_cv_grid_coarse / build_cv_grid_fixed_lambda_coarse).
    bool coarse_grid = false;

    /// Experimental: remove the drift penalty (`lambda_drift`) on velocity data. Normally
    /// `lambda_drift = 1e-2` penalizes the size of the spline values so the integrated-velocity
    /// position proxy cannot drift. It was tuned for the IRGN schedule (λ₀ = 1, decay 0.5),
    /// where the smoothing penalty dominates it in early iterations. With a fixed λ below 1e-2
    /// the drift penalty dominates instead, pulls the EA towards zero, and gave a large proxy
    /// RMS error on 100 and 200 Hz velocity data (measured). This flag exists to test that
    /// explanation; leave it false.
    bool drop_lambda_drift = false;

    /// Compute GN's Jacobian from the state transition matrix (STM: the sensitivity of the
    /// state at time t to the EA parameters, integrated with the orbit) instead of by finite
    /// differences (FD). Default true: faster than FD, which matters over a grid of hundreds of
    /// candidate fits.
    /// refine_ea_gn builds a linearly interpolated EA table on this path.
    bool gn_use_stm = true;
    /// STM settings, copied from the public GNRefine (defaults equal GNRefineConfig's).
    double gn_stm_dt_s   = 0.5;
    double gn_stm_fd_dx  = 1.0;
    double gn_stm_fd_dv  = 1e-3;
    bool   gn_stm_verify = false;
    bool   gn_stm_verbose = false;

    /// Print per-iteration logs from each candidate fit (set from EABSplineConfig::verbose).
    bool verbose = false;

    /// Maximum IRGN iterations for each candidate fit (default 10). The final fit on all data
    /// uses `EABSplineConfig::max_iterations`. Raise it if candidates do not converge within 10
    /// iterations (rare); lower it to make CV faster at some cost in reliability.
    int cv_inner_max_iterations = 10;
};

// =========================================================================
// hv-block split (Racine 2000)
// =========================================================================

/// Split time-stamped samples (pairs whose `.first` is an Epoch) into training and test sets.
/// The interval [epoch0, epoch0 + arc_s] is cut into `n_blocks` equal blocks; blocks listed
/// in `test_block_idx` are test, their `gap_blocks` neighbors on each side are dropped, the
/// rest is training (samples outside the interval go to the first or last block; invalid
/// indices are ignored). Choose interior test blocks, with training data on both sides, so
/// the score measures interpolation rather than extrapolation. Returns {train, test}.
template<typename T>
[[nodiscard]] inline std::pair<std::vector<T>, std::vector<T>>
split_hv_block(
    const std::vector<T>& data,
    const Epoch& epoch0,
    double arc_s,
    int n_blocks,
    const std::vector<int>& test_block_idx,
    int gap_blocks)
{
    std::vector<T> train, test;
    if (data.empty() || n_blocks <= 0 || arc_s <= 0) return {train, test};

    enum class Status : int { TRAIN, GAP, TEST };
    std::vector<Status> blocks(n_blocks, Status::TRAIN);
    for (int t : test_block_idx) {
        if (t < 0 || t >= n_blocks) continue;
        blocks[t] = Status::TEST;
        for (int g = 1; g <= gap_blocks; ++g) {
            if (t - g >= 0 && blocks[t - g] != Status::TEST)
                blocks[t - g] = Status::GAP;
            if (t + g < n_blocks && blocks[t + g] != Status::TEST)
                blocks[t + g] = Status::GAP;
        }
    }

    const double block_dur = arc_s / n_blocks;
    for (const auto& obs : data) {
        const double rel = obs.first - epoch0;
        int bi = static_cast<int>(rel / block_dur);
        if (bi < 0) bi = 0;
        if (bi >= n_blocks) bi = n_blocks - 1;
        switch (blocks[bi]) {
        case Status::TRAIN: train.push_back(obs); break;
        case Status::TEST:  test.push_back(obs);  break;
        case Status::GAP:   break;
        }
    }
    return {train, test};
}

/// Build the GNRefineConfig used inside CV from a CVDeploymentConfig, the force model, the
/// candidate's knot spacing [s] and the arc length [s]. Counterpart of ea_estimator.hpp's
/// `translate_gn_config`, which reads the public GNRefine; this header cannot include
/// EABSplineConfig without a circular include.
[[nodiscard]] inline GNRefineConfig make_gn_config_from_dep(
    const CVDeploymentConfig& dep,
    const TrajectoryConfig& force_model,
    double knot_spacing_s,
    double max_duration_s,
    int reg_order_auto)   // penalty order when dep.gn_reg_order < 0: 3 position, 1 velocity
{
    GNRefineConfig gc;
    gc.max_iterations = dep.gn_iterations;
    gc.fd_delta = dep.fd_delta;
    gc.sample_stride = dep.gn_sample_stride;
    gc.ea_subsample = dep.gn_ea_subsample;
    gc.max_duration_s = max_duration_s;
    gc.min_altitude_m = 0.0;
    gc.use_stm = dep.gn_use_stm;
    gc.stm_dt_s = dep.gn_stm_dt_s;
    gc.stm_fd_dx = dep.gn_stm_fd_dx;
    gc.stm_fd_dv = dep.gn_stm_fd_dv;
    gc.stm_verify = dep.gn_stm_verify;
    gc.stm_verbose = dep.gn_stm_verbose;
    // GN smoothing penalty, same kind as the spline fit's.
    gc.reg_order = dep.gn_reg_order >= 0 ? dep.gn_reg_order : reg_order_auto;
    gc.reg_lambda = dep.gn_reg_lambda;

    gc.trajectory_config = force_model;
    if (dep.rk45_max_step_size < 0.0)
        gc.trajectory_config.rk45.max_step_size = knot_spacing_s;
    else if (dep.rk45_max_step_size > 0.0)
        gc.trajectory_config.rk45.max_step_size = dep.rk45_max_step_size;
    return gc;
}

/// After GN has replaced `result.empirical_accel`, rebuild `collocation_history` (and
/// `final_trajectory` / `final_interp`) from an orbit propagated with the refined EA, at the
/// same times as before. Without this the history would still describe the pre-GN EA, while
/// refine_beta_from_ea / decompose_to_beta pair it with the refined one: the velocity used to
/// split the EA into drag and remainder would come from a different orbit. Costs one
/// propagation of the arc. Entries the new orbit does not reach are dropped.
inline void refresh_history_with_ea(
    EAResult& result,
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const TrajectoryConfig& force_model,
    double max_duration_s)
{
    if (!result.empirical_accel || result.empirical_accel->is_zero() ||
        result.collocation_history.empty())
        return;
    auto [traj_u, interp_u] = detail::run_propagation_arc(
        epoch0, pos_ecef, vel_ecef, force_model, result.empirical_accel,
        max_duration_s, /*min_altitude_m=*/0.0, /*backward=*/false);
    std::shared_ptr<Trajectory> traj = std::move(traj_u);
    std::shared_ptr<TrajectoryInterpolator> interp = std::move(interp_u);

    std::vector<HistoryEntry> fresh;
    fresh.reserve(result.collocation_history.size());
    for (const auto& old : result.collocation_history) {
        auto st = interp->get_state(old.epoch);
        if (!st) continue;
        StateVector se{st->position, st->velocity};
        StateVector si = util::itrf_to_gcrs(se, old.epoch, force_model.itrf);
        Vector3d a_gcrs = traj->force_model().acceleration(old.epoch, si);
        Vector3d a_ecef = util::acceleration_to_itrf(a_gcrs, se, old.epoch);
        fresh.emplace_back(old.epoch, old.t_rel_s, st->position, st->velocity, a_ecef,
                           old.step_size_used);
    }
    result.collocation_history = std::move(fresh);
    result.final_trajectory = std::move(traj);
    result.final_interp = std::move(interp);
}

/// GN-refine an IRGN result against `position_targets` (ECEF [m]) and return the refined EA.
/// `reg_order_auto` is the GN penalty order used when dep.gn_reg_order is automatic (3 for
/// position data, 1 for velocity data). Returns `initial_ea` unchanged when GN is off
/// (gn_iterations ≤ 0), the EA is null, or there are no targets. Never returns null for a
/// non-null input.
[[nodiscard]] inline std::shared_ptr<forces::EmpiricalAcceleration>
apply_gn_for_cv(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& position_targets,
    const TrajectoryConfig& force_model,
    const std::shared_ptr<forces::EmpiricalAcceleration>& initial_ea,
    const CVDeploymentConfig& dep,
    double knot_spacing_s,
    double max_duration_s,
    int reg_order_auto)
{
    if (!initial_ea || dep.gn_iterations <= 0 || position_targets.empty())
        return initial_ea;
    auto gc = make_gn_config_from_dep(dep, force_model, knot_spacing_s, max_duration_s,
                                      reg_order_auto);
    return refine_ea_gn(epoch0, pos_ecef, vel_ecef, position_targets, initial_ea, gc);
}

// =========================================================================
// K-fold splitting — a simple utility for tests
// =========================================================================
//
// Used by `tests/integration/test_ea_cv.cpp` and `test_ea_estimator.cpp` to check basic
// splitting properties (folds are disjoint, merging gives the data back). The estimator uses
// the hv-block split above instead: K-fold interleaves neighboring samples between training
// and test, so errors correlated in time leak across and the score is too optimistic.
// =========================================================================

/// K-fold split result: endpoints always in training, interior in folds.
template<typename T>
struct KFoldSplit {
    std::vector<T> endpoints;          ///< First and last obs (always in training)
    std::vector<std::vector<T>> folds; ///< K folds of interior observations
};

/// Split data into K folds by systematic thinning of interior observations.
/// Endpoints (first and last) are excluded from all folds — they appear in
/// every training set but never in any test set.
template<typename T>
[[nodiscard]] inline KFoldSplit<T> split_kfold(const std::vector<T>& data, int k)
{
    KFoldSplit<T> result;
    result.folds.resize(k);
    for (int f = 0; f < k; ++f)
        result.folds[f].reserve(data.size() / k + 1);

    if (data.empty()) return result;

    // Endpoints → always-train
    result.endpoints.push_back(data.front());
    if (data.size() > 1) result.endpoints.push_back(data.back());

    // Interior → systematic K-fold thinning
    for (size_t i = 1; i + 1 < data.size(); ++i)
        result.folds[static_cast<int>(i % k)].push_back(data[i]);

    return result;
}

/// Merge all folds except fold_idx + endpoints into a training set.
template<typename T>
[[nodiscard]] inline std::vector<T> merge_train_folds(
    const KFoldSplit<T>& split, int fold_idx)
{
    size_t total = split.endpoints.size();
    for (int i = 0; i < static_cast<int>(split.folds.size()); ++i)
        if (i != fold_idx) total += split.folds[i].size();

    std::vector<T> train;
    train.reserve(total);

    // Always include endpoints
    train.insert(train.end(), split.endpoints.begin(), split.endpoints.end());

    // Include all folds except the test fold
    for (int i = 0; i < static_cast<int>(split.folds.size()); ++i)
        if (i != fold_idx)
            train.insert(train.end(), split.folds[i].begin(), split.folds[i].end());

    // Re-sort by epoch (systematic thinning interleaves folds)
    std::sort(train.begin(), train.end(),
        [](const T& a, const T& b) { return a.first < b.first; });
    return train;
}

// =========================================================================
// Holdout evaluation
// =========================================================================

/// Re-propagate from the start state with EA `ea` and return the RMS position error [m]
/// against `test_positions` (ECEF). Infinity if `ea` is null, there are no test points, or
/// none falls inside the propagated arc.
[[nodiscard]] inline double eval_holdout_position_rmse(
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea,
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const TrajectoryConfig& force_model,
    double max_duration_s,
    const std::vector<std::pair<Epoch, Vector3d>>& test_positions)
{
    if (!ea || test_positions.empty())
        return std::numeric_limits<double>::infinity();

    auto [traj, interp] = detail::run_propagation_arc(
        epoch0, pos_ecef, vel_ecef, force_model, ea,
        max_duration_s, /*min_altitude_m=*/0.0, /*backward=*/false);

    double sse = 0;
    int n = 0;
    for (const auto& [ep, obs] : test_positions) {
        auto st = interp->get_state(ep);
        if (st) {
            sse += (st->position - obs).squaredNorm();
            ++n;
        }
    }
    return n > 0 ? std::sqrt(sse / n) : std::numeric_limits<double>::infinity();
}

/// Same score as `eval_holdout_position_rmse`, but from an existing interpolator, saving a
/// propagation. Valid only if the interpolator was propagated with the EA being scored (i.e.
/// GN did not change it afterwards).
[[nodiscard]] inline double eval_holdout_position_rmse_fused(
    TrajectoryInterpolator& interp,
    const std::vector<std::pair<Epoch, Vector3d>>& test_positions)
{
    if (test_positions.empty()) return std::numeric_limits<double>::infinity();

    double sse = 0;
    int n = 0;
    for (const auto& [ep, obs] : test_positions) {
        auto st = interp.get_state(ep);
        if (st) {
            sse += (st->position - obs).squaredNorm();
            ++n;
        }
    }
    return n > 0 ? std::sqrt(sse / n) : std::numeric_limits<double>::infinity();
}

// =========================================================================
// Integration proxies (ECEF space)
// =========================================================================

/// ECEF velocities → ECEF positions by the trapezoidal rule (see
/// trapezoidal_vel_to_pos_ecef in integration.hpp; it drifts, acceptable for scoring).
[[nodiscard]] inline std::vector<std::pair<Epoch, Vector3d>> integrate_ecef_vel_to_pos(
    const std::vector<std::pair<Epoch, Vector3d>>& vel_ecef,
    const Vector3d& pos0_ecef)
{
    return detail::trapezoidal_vel_to_pos_ecef(vel_ecef, pos0_ecef);
}

/// ECEF accelerations → velocities → positions by the trapezoidal rule (see
/// trapezoidal_acc_to_pos_ecef in integration.hpp).
[[nodiscard]] inline std::vector<std::pair<Epoch, Vector3d>> integrate_ecef_acc_to_pos(
    const std::vector<std::pair<Epoch, Vector3d>>& acc_ecef,
    const Vector3d& vel0_ecef,
    const Vector3d& pos0_ecef)
{
    return detail::trapezoidal_acc_to_pos_ecef(acc_ecef, vel0_ecef, pos0_ecef);
}

// =========================================================================
// Build internal config from hyperparams
// =========================================================================

/// Kind of observation being cross-validated.
enum class CVObsType { POSITION, VELOCITY, ACCELERATION };

/// Build the B-spline fit configuration for one candidate `hp`. The smoothing penalty acts on
/// the 3rd derivative for position data and on the automatic order (-1, which the fitter
/// resolves to the 1st derivative) for velocity and acceleration data. Velocity data also get
/// the drift penalty 1e-2 unless `drop_lambda_drift`. `fixed_lambda` holds λ constant instead
/// of decreasing it.
[[nodiscard]] inline BSplineAccelConfig make_cv_bspline_config(
    const CVHyperParams& hp,
    const TrajectoryConfig& force_model,
    CVObsType obs_type,
    int degree,
    double max_duration_s,
    double noise_sigma,
    int max_iterations,
    bool fixed_lambda = false,
    bool drop_lambda_drift = false,
    bool verbose = false)
{
    BSplineAccelConfig cfg;
    cfg.trajectory_config = force_model;
    cfg.knot_spacing_s = hp.knot_spacing;
    cfg.lambda = hp.lambda_0;
    cfg.lambda_decay = hp.decay;
    cfg.lambda_min = hp.lambda_min;
    cfg.lambda_method = fixed_lambda
        ? detail::LambdaMethod::FIXED
        : detail::LambdaMethod::DECREASING;
    cfg.max_iterations = max_iterations;
    cfg.degree = degree;
    cfg.max_duration_s = max_duration_s;
    cfg.noise_sigma = noise_sigma;
    cfg.min_altitude_m = 0.0;
    cfg.verbose = verbose;

    switch (obs_type) {
    case CVObsType::POSITION:
        cfg.reg_deriv_order = 3;
        break;
    case CVObsType::VELOCITY:
        cfg.reg_deriv_order = -1;
        cfg.lambda_drift = drop_lambda_drift ? 0.0 : 1e-2;
        break;
    case CVObsType::ACCELERATION:
        cfg.reg_deriv_order = -1;
        break;
    }

    return cfg;
}

// =========================================================================
// CV entry points
// =========================================================================

/// Fit an EA to position observations (`pos_data_ecef`, ECEF [m]) with settings chosen by
/// hv-block CV. Each grid candidate is fitted on the training blocks (IRGN, then GN if
/// `dep.gn_iterations > 0`) and scored by the RMS position error of the re-propagated orbit at
/// the held-out times; the best candidate is then refitted on all data with
/// `max_iterations_final` iterations and GN. Start state `epoch0`, `pos_ecef` [m], `vel_ecef`
/// [m/s] (ECEF); `degree` of the spline; `max_duration_s` [s] caps the propagation;
/// `noise_sigma` [m] enables early stopping. The candidates run in parallel (OpenMP). A
/// single held-out set is used (no rotation over several folds). If the split leaves no
/// training or no test data, the first grid point is fitted on all data instead.
[[nodiscard]] inline EAResult cv_bspline_position(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& pos_data_ecef,
    const TrajectoryConfig& force_model,
    int degree,
    int max_iterations_final,
    double max_duration_s,
    double noise_sigma,
    const CVDeploymentConfig& dep = {})
{
    const bool fixed_lambda = (dep.cv_mode == CVDeploymentConfig::CVMode::FixedLambda);
    const auto grid = select_cv_grid(fixed_lambda, dep.coarse_grid);

    // Propagation length: up to the last observation + 10 s, at most max_duration_s.
    double max_dur = pos_data_ecef.empty() ? max_duration_s
        : (pos_data_ecef.back().first - epoch0) + 10.0;
    if (max_dur > max_duration_s) max_dur = max_duration_s;

    // hv-block split into training and held-out samples.
    const double arc_for_blocks = pos_data_ecef.empty() ? max_dur
        : (pos_data_ecef.back().first - epoch0);
    auto [train, test] = split_hv_block(
        pos_data_ecef, epoch0, arc_for_blocks,
        dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);

    // If the split left no training or no test data (e.g. a very short arc with many
    // blocks), fit the first grid point on all data.
    if (train.empty() || test.empty()) {
        auto final_cfg = make_cv_bspline_config(
            grid[0], force_model, CVObsType::POSITION,
            degree, max_duration_s, noise_sigma, max_iterations_final,
            fixed_lambda, /*drop_lambda_drift=*/false, dep.verbose);
        return estimate_empirical_acceleration_bspline(
            epoch0, pos_ecef, vel_ecef, pos_data_ecef, final_cfg);
    }

    std::vector<double> holdout(grid.size(), std::numeric_limits<double>::infinity());

    #pragma omp parallel for schedule(dynamic)
    for (int gi = 0; gi < static_cast<int>(grid.size()); ++gi) {
        auto cfg = make_cv_bspline_config(
            grid[gi], force_model, CVObsType::POSITION,
            degree, max_duration_s, noise_sigma, dep.cv_inner_max_iterations,
            fixed_lambda, /*drop_lambda_drift=*/false, dep.verbose);

        auto ar = estimate_empirical_acceleration_bspline(
            epoch0, pos_ecef, vel_ecef, train, cfg);

        if (!ar.empirical_accel) continue;

        // GN refinement if requested, as in the final fit.
        auto ea = apply_gn_for_cv(
            epoch0, pos_ecef, vel_ecef, train, force_model,
            ar.empirical_accel, dep,
            grid[gi].knot_spacing, max_duration_s, /*reg_order_auto=*/3);

        // Without GN the fit's own final interpolator still matches `ea` and can be scored
        // directly; after GN the EA changed, so re-propagate.
        const bool gn_was_noop = (dep.gn_iterations <= 0);
        const double rmse = (gn_was_noop && ar.final_interp)
            ? eval_holdout_position_rmse_fused(*ar.final_interp, test)
            : eval_holdout_position_rmse(
                  ea, epoch0, pos_ecef, vel_ecef,
                  force_model, max_dur, test);
        if (std::isfinite(rmse)) holdout[gi] = rmse;
    }

    int best_idx = pick_best_grid_idx(holdout, grid);
    report_cv_grid_edge(grid, best_idx, fixed_lambda, "position");

    // Refit on all data with the best settings, full iterations, then GN.
    auto final_cfg = make_cv_bspline_config(
        grid[best_idx], force_model, CVObsType::POSITION,
        degree, max_duration_s, noise_sigma, max_iterations_final,
            fixed_lambda, /*drop_lambda_drift=*/false, dep.verbose);

    auto final_ar = estimate_empirical_acceleration_bspline(
        epoch0, pos_ecef, vel_ecef, pos_data_ecef, final_cfg);
    if (final_ar.empirical_accel) {
        final_ar.empirical_accel = apply_gn_for_cv(
            epoch0, pos_ecef, vel_ecef, pos_data_ecef, force_model,
            final_ar.empirical_accel, dep,
            grid[best_idx].knot_spacing, max_duration_s, /*reg_order_auto=*/3);
        if (dep.gn_iterations > 0) {
            final_ar.gn_status = GNStatus::Ran;
            refresh_history_with_ea(final_ar, epoch0, pos_ecef, vel_ecef,
                                    force_model, max_duration_s);
        }
    }
    return final_ar;
}

/// Held-out score of one candidate `hp` for velocity data: fit the B-spline to the training
/// velocities `vel_train_gcrs`, GN-refine it (when `dep.gn_iterations > 0`) against the
/// training proxy positions `pos_train_proxy`, and return the RMS position error of the
/// re-propagated orbit at the held-out proxy positions `pos_test_proxy` [m] (infinity if the
/// fit fails). GN must not see `pos_test_proxy`: refining on the points that are then scored
/// makes every candidate look better than it is and biases the choice toward settings GN can
/// bend onto the test block. The training proxies downstream of the held-out block still
/// carry the integral of its velocities (the proxy is one cumulative integration over the
/// whole arc); that is a much weaker coupling than fitting the scored points themselves.
[[nodiscard]] inline double cv_velocity_candidate_score(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& vel_train_gcrs,
    const std::vector<std::pair<Epoch, Vector3d>>& pos_train_proxy,
    const std::vector<std::pair<Epoch, Vector3d>>& pos_test_proxy,
    const TrajectoryConfig& force_model,
    const CVHyperParams& hp,
    int degree,
    double max_duration_s,
    double max_dur,
    double noise_sigma,
    const CVDeploymentConfig& dep,
    bool fixed_lambda)
{
    auto cfg = make_cv_bspline_config(
        hp, force_model, CVObsType::VELOCITY,
        degree, max_duration_s, noise_sigma, dep.cv_inner_max_iterations,
        fixed_lambda, dep.drop_lambda_drift, dep.verbose);

    auto ar = estimate_empirical_acceleration_bspline_from_velocity(
        epoch0, pos_ecef, vel_ecef, vel_train_gcrs, cfg);
    if (!ar.empirical_accel) return std::numeric_limits<double>::infinity();

    auto ea = apply_gn_for_cv(
        epoch0, pos_ecef, vel_ecef, pos_train_proxy, force_model,
        ar.empirical_accel, dep, hp.knot_spacing, max_duration_s, /*reg_order_auto=*/1);

    // Without GN the fit's own interpolator still matches `ea`; after GN, re-propagate.
    const bool gn_was_noop = (dep.gn_iterations <= 0);
    return (gn_was_noop && ar.final_interp)
        ? eval_holdout_position_rmse_fused(*ar.final_interp, pos_test_proxy)
        : eval_holdout_position_rmse(
              ea, epoch0, pos_ecef, vel_ecef, force_model, max_dur, pos_test_proxy);
}

/// Fit an EA to velocity observations (`vel_data_ecef`, ECEF [m/s]) with settings chosen by
/// hv-block CV, as cv_bspline_position does. The held-out "positions" are the velocities
/// integrated from `pos_ecef` (a drifting proxy, see integration.hpp); the fit itself uses the
/// velocities rotated to GCRS. With `dep.gn_iterations > 0`, each candidate is GN-refined
/// against the training proxy positions (never the held-out ones it is scored on) and the
/// final fit against all proxy positions.
[[nodiscard]] inline EAResult cv_bspline_velocity(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& vel_data_ecef,
    const TrajectoryConfig& force_model,
    int degree,
    int max_iterations_final,
    double max_duration_s,
    double noise_sigma,
    const CVDeploymentConfig& dep = {})
{
    auto all_pos_proxy = integrate_ecef_vel_to_pos(vel_data_ecef, pos_ecef);
    auto all_vel_gcrs  = vel_ecef_to_gcrs(vel_data_ecef, pos_ecef);

    double max_dur = vel_data_ecef.empty() ? max_duration_s
        : (vel_data_ecef.back().first - epoch0) + 10.0;
    if (max_dur > max_duration_s) max_dur = max_duration_s;

    const double arc_for_blocks = vel_data_ecef.empty() ? max_dur
        : (vel_data_ecef.back().first - epoch0);
    auto vel_split = split_hv_block(
        all_vel_gcrs,  epoch0, arc_for_blocks,
        dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
    auto pos_split = split_hv_block(
        all_pos_proxy, epoch0, arc_for_blocks,
        dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
    const auto& vel_train = vel_split.first;
    const auto& pos_train = pos_split.first;
    const auto& pos_test  = pos_split.second;

    const bool fixed_lambda = (dep.cv_mode == CVDeploymentConfig::CVMode::FixedLambda);
    const auto grid = select_cv_grid(fixed_lambda, dep.coarse_grid);
    if (vel_train.empty() || pos_test.empty()) {
        auto fc = make_cv_bspline_config(
            grid[0], force_model, CVObsType::VELOCITY,
            degree, max_duration_s, noise_sigma, max_iterations_final,
            fixed_lambda, dep.drop_lambda_drift, dep.verbose);
        return estimate_empirical_acceleration_bspline_from_velocity(
            epoch0, pos_ecef, vel_ecef, all_vel_gcrs, fc);
    }

    std::vector<double> holdout(grid.size(), std::numeric_limits<double>::infinity());

    #pragma omp parallel for schedule(dynamic)
    for (int gi = 0; gi < static_cast<int>(grid.size()); ++gi) {
        const double rmse = cv_velocity_candidate_score(
            epoch0, pos_ecef, vel_ecef, vel_train, pos_train, pos_test,
            force_model, grid[gi], degree, max_duration_s, max_dur, noise_sigma,
            dep, fixed_lambda);
        if (std::isfinite(rmse)) holdout[gi] = rmse;
    }

    int best_idx = pick_best_grid_idx(holdout, grid);
    report_cv_grid_edge(grid, best_idx, fixed_lambda, "velocity");

    auto final_cfg = make_cv_bspline_config(
        grid[best_idx], force_model, CVObsType::VELOCITY,
        degree, max_duration_s, noise_sigma, max_iterations_final,
        fixed_lambda, dep.drop_lambda_drift, dep.verbose);
    auto final_ar = estimate_empirical_acceleration_bspline_from_velocity(
        epoch0, pos_ecef, vel_ecef, all_vel_gcrs, final_cfg);
    if (final_ar.empirical_accel && dep.gn_iterations > 0) {
        final_ar.empirical_accel = apply_gn_for_cv(
            epoch0, pos_ecef, vel_ecef, all_pos_proxy, force_model,
            final_ar.empirical_accel, dep,
            grid[best_idx].knot_spacing, max_duration_s, /*reg_order_auto=*/1);
        final_ar.gn_status = GNStatus::Ran;
        refresh_history_with_ea(final_ar, epoch0, pos_ecef, vel_ecef,
                                force_model, max_duration_s);
    }
    return final_ar;
}

/// Fit an EA to acceleration observations (`acc_data_ecef`, ECEF [m/s²]) with settings chosen
/// by hv-block CV, scored at positions obtained by integrating the accelerations twice. GN is
/// never applied: its position targets would come from the same accelerations (circular).
/// On noisy data the score barely tracks the true error; prefer GCV

/// with an explicit `noise_sigma` there.
[[nodiscard]] inline EAResult cv_bspline_acceleration(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& acc_data_ecef,
    const TrajectoryConfig& force_model,
    int degree,
    int max_iterations_final,
    double max_duration_s,
    double noise_sigma,
    const CVDeploymentConfig& dep = {})
{
    auto all_pos_proxy = integrate_ecef_acc_to_pos(acc_data_ecef, vel_ecef, pos_ecef);
    auto all_acc_gcrs  = acc_ecef_to_gcrs(acc_data_ecef, vel_ecef, pos_ecef);

    double max_dur = acc_data_ecef.empty() ? max_duration_s
        : (acc_data_ecef.back().first - epoch0) + 10.0;
    if (max_dur > max_duration_s) max_dur = max_duration_s;

    const double arc_for_blocks = acc_data_ecef.empty() ? max_dur
        : (acc_data_ecef.back().first - epoch0);
    auto acc_split = split_hv_block(
        all_acc_gcrs,  epoch0, arc_for_blocks,
        dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
    auto pos_split = split_hv_block(
        all_pos_proxy, epoch0, arc_for_blocks,
        dep.hv_n_blocks, dep.hv_test_blocks, dep.hv_gap_blocks);
    const auto& acc_train = acc_split.first;
    const auto& pos_test  = pos_split.second;

    const bool fixed_lambda = (dep.cv_mode == CVDeploymentConfig::CVMode::FixedLambda);
    const auto grid = select_cv_grid(fixed_lambda, dep.coarse_grid);
    if (acc_train.empty() || pos_test.empty()) {
        auto fc = make_cv_bspline_config(
            grid[0], force_model, CVObsType::ACCELERATION,
            degree, max_duration_s, noise_sigma, max_iterations_final,
            fixed_lambda, /*drop_lambda_drift=*/false, dep.verbose);
        return estimate_empirical_acceleration_bspline_from_acceleration(
            epoch0, pos_ecef, vel_ecef, all_acc_gcrs, fc);
    }

    std::vector<double> holdout(grid.size(), std::numeric_limits<double>::infinity());

    #pragma omp parallel for schedule(dynamic)
    for (int gi = 0; gi < static_cast<int>(grid.size()); ++gi) {
        auto cfg = make_cv_bspline_config(
            grid[gi], force_model, CVObsType::ACCELERATION,
            degree, max_duration_s, noise_sigma, dep.cv_inner_max_iterations,
            fixed_lambda, /*drop_lambda_drift=*/false, dep.verbose);

        auto ar = estimate_empirical_acceleration_bspline_from_acceleration(
            epoch0, pos_ecef, vel_ecef, acc_train, cfg);
        if (!ar.empirical_accel) continue;

        // No GN for acceleration data, so the fit's own interpolator always matches its EA.
        const double rmse = ar.final_interp
            ? eval_holdout_position_rmse_fused(*ar.final_interp, pos_test)
            : eval_holdout_position_rmse(
                  ar.empirical_accel, epoch0, pos_ecef, vel_ecef,
                  force_model, max_dur, pos_test);
        if (std::isfinite(rmse)) holdout[gi] = rmse;
    }

    int best_idx = pick_best_grid_idx(holdout, grid);
    report_cv_grid_edge(grid, best_idx, fixed_lambda, "acceleration");

    auto final_cfg = make_cv_bspline_config(
        grid[best_idx], force_model, CVObsType::ACCELERATION,
        degree, max_duration_s, noise_sigma, max_iterations_final,
            fixed_lambda, /*drop_lambda_drift=*/false, dep.verbose);
    return estimate_empirical_acceleration_bspline_from_acceleration(
        epoch0, pos_ecef, vel_ecef, all_acc_gcrs, final_cfg);
}

} // namespace detail
} // namespace calibration
} // namespace ballistic
