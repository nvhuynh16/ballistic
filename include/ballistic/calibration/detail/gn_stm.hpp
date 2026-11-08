#pragma once

// =============================================================================
// Sensitivity (variational-equation) Jacobian for refine_ea_gn
// =============================================================================
//
// Gauss-Newton (GN) refinement of an empirical acceleration (EA: a correction acceleration
// fitted to observations and added to the force model) needs J = how each predicted position
// changes when one EA node value changes. Instead of re-propagating once per parameter
// (finite differences, FD), this computes all columns in one pass by integrating the
// sensitivity matrix S(t) = ∂x(t)/∂p alongside the nominal trajectory (x = position and
// velocity, p = EA node values). S is the parameter part of the state transition matrix
// (STM: the sensitivity of the state at time t to the initial conditions and parameters).
//
//   dS/dt = A(t) S + B(t),   S(t0) = 0
//   A(t)  = ∂f/∂x   6×6, from ForceModel::acceleration_with_jacobian (analytic where
//                   available, finite differences otherwise)
//   B(t)  = ∂f/∂p   6×n_params, exact for a linearly interpolated EA table, nonzero only for
//                   the two nodes around t
//   J_ij  = R_gcrs_to_ecef(t_i) · S[position rows, j] at observation time t_i
//
// S is stepped with Heun's method (second-order Runge-Kutta) at a fixed step `dt_var_s`; the
// A(t) computed at the end of one step is reused at the start of the next.
//
// Frames: the trajectory is integrated in the inertial GCRS frame. An EA stored in GCRS
// enters B(t) directly; an EA stored in the Earth-fixed ITRF frame is rotated into GCRS
// first (see `add_B_to`).
//
// Assumptions (not checked here): the EA table uses Linear interpolation (refine_ea_gn builds
// it that way on this path; Cubic interpolation is not supported), and the propagation frame
// is GCRS (the default).
//
// =============================================================================

#include "../../core/types.hpp"
#include "../../time/epoch.hpp"
#include "../../trajectory.hpp"
#include "../../trajectory_interpolator.hpp"
#include "../../forces/empirical_acceleration.hpp"
#include "../../forces/force_model.hpp"
#include "../../propagators/variational.hpp"
#include "../../coordinates/iau_transforms.hpp"
#include "../../coordinates/geodetic.hpp"
#include "calibration_constants.hpp"

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>

namespace ballistic {
namespace calibration {
namespace detail {

/// Output of build_stm_jacobian_and_pnom: predicted positions and their Jacobian with respect
/// to the EA node values. Columns are ordered j = comp * n_nodes + node (comp 0/1/2 = x/y/z).
struct StmResult {
    Eigen::VectorXd p_nom;       // size 3*n_samples; positions at sample times (ECEF)
    Eigen::MatrixXd J;           // size (3*n_samples) × n_params [m per m/s²]; zero rows past a truncated arc
};

/// Propagate the trajectory once with EA `ea` and return the ECEF positions at the selected
/// observation times plus their Jacobian J with respect to the EA node values.
/// `epoch0`, `pos_ecef` [m], `vel_ecef` [m/s]: Earth-fixed start state. `sample_indices` pick
/// the observations of `position_data` used. `ea_keys` [s from epoch0] are the EA node times
/// (`n_nodes` of them, `n_params` = 3·n_nodes); `ea_frame` is the frame the EA is stored in.
/// Propagation stops after `max_duration_s` [s] or below `min_altitude_m` [m]; observation
/// times past the end get zero Jacobian rows (and a zero position if not interpolable).
/// `dt_var_s` [s] is the sensitivity step; `fd_dx` [m] and `fd_dv` [m/s] are the finite-
/// difference perturbations for force terms without analytic Jacobians. If the state cannot be
/// interpolated at t = 0, prints a warning to stderr and returns an all-zero J.
/// Assumes a Linear-interpolation EA table: `add_B_to` encodes that interpolation.
[[nodiscard]] inline StmResult build_stm_jacobian_and_pnom(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const TrajectoryConfig& trajectory_config_in,
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea,
    const std::vector<size_t>& sample_indices,
    const std::vector<std::pair<Epoch, Vector3d>>& position_data,
    const std::vector<double>& ea_keys,
    int n_nodes,
    int n_params,
    forces::EmpiricalAccelFrame ea_frame,
    double max_duration_s,
    double min_altitude_m,
    double dt_var_s,
    double fd_dx,
    double fd_dv,
    bool verbose)
{
    StmResult out;
    const int n_samples = static_cast<int>(sample_indices.size());
    out.p_nom = Eigen::VectorXd::Zero(3 * n_samples);

    // 1. Forward-propagate the nominal trajectory.
    TrajectoryConfig cfg = trajectory_config_in;
    if (ea) cfg.empirical_accel = ea;
    Trajectory traj(epoch0, pos_ecef, vel_ecef, cfg);
    traj.mark_internal();
    while (traj.epoch() - epoch0 < max_duration_s) {
        auto sr = traj.step();
        if (coords::ecef_to_geodetic(sr.position).altitude < min_altitude_m) break;
    }

    // Interpolators: ECEF for the returned positions (same as the FD path), GCRS for the
    // states at which A(t) is evaluated.
    InterpolatorConfig ic_ecef;
    ic_ecef.output_frame = InterpolationFrame::ECEF;
    TrajectoryInterpolator ti_ecef(traj, ic_ecef);

    InterpolatorConfig ic_gcrs;
    ic_gcrs.output_frame = InterpolationFrame::GCRS;
    TrajectoryInterpolator ti_gcrs(traj, ic_gcrs);

    // 2. Sample p_nom (ECEF).
    for (int i = 0; i < n_samples; ++i) {
        auto ms = ti_ecef.get_state(position_data[sample_indices[i]].first);
        out.p_nom.segment<3>(3 * i) = ms ? ms->position : Vector3d::Zero();
    }

    const double t_end = traj.epoch() - epoch0;
    if (t_end <= 0.0) {
        out.J = Eigen::MatrixXd::Zero(3 * n_samples, n_params);
        return out;
    }

    // 3. Force model for A(t): the one that produced the nominal trajectory (same EA, same
    //    switches).
    auto& fm = traj.force_model();

    // 4. A(t) = ∂f/∂x with f(x) = [v; a(x, t)] at the interpolated GCRS state
    //    (propagators/variational.hpp).
    auto compute_A = [&](double t, Eigen::Matrix<double, 6, 6>& A) -> bool {
        Epoch ep = epoch0 + t;
        auto ms = ti_gcrs.get_state(ep);
        if (!ms) return false;
        A = ::ballistic::propagators::detail::variational_matrix(
            fm, ep, StateVector{ms->position, ms->velocity}, fd_dx, fd_dv);
        return true;
    };

    // 5. Add B(t) to dS/dt in place. Only the two EA nodes around t have nonzero columns;
    //    with linear interpolation their weights are
    //      w_left  = (key_right - t) / (key_right - key_left),  w_right = 1 - w_left,
    //    and before the first / after the last node the end value is held (as
    //    LinearVector3d::evaluate does).
    //    The EA acceleration in GCRS is R(t)·EA(t), with R = identity for a GCRS EA and
    //    R = ITRF→GCRS rotation for an ITRF EA, so ∂a/∂(node i, component c) = w_i · R.col(c).
    auto add_B_to = [&](double t, Eigen::MatrixXd& AS_plus_B) {
        if (ea_keys.empty()) return;
        const int n = static_cast<int>(ea_keys.size());

        int i_left, i_right;
        double w_left, w_right;
        if (t <= ea_keys.front()) {
            i_left = 0; i_right = 0;
            w_left = 1.0; w_right = 0.0;
        } else if (t >= ea_keys.back()) {
            i_left = n - 1; i_right = n - 1;
            w_left = 1.0; w_right = 0.0;
        } else {
            auto it = std::upper_bound(ea_keys.begin(), ea_keys.end(), t);
            i_right = static_cast<int>(it - ea_keys.begin());
            i_left = i_right - 1;
            const double tl = ea_keys[i_left];
            const double tr = ea_keys[i_right];
            w_left = (tr - t) / (tr - tl);
            w_right = 1.0 - w_left;
        }

        // Resolve the rotation: identity for GCRS, R_itrf_to_gcrs(t) for ITRF.
        Matrix3d R;
        if (ea_frame == forces::EmpiricalAccelFrame::ITRF) {
            R = coords::detail::get_cached_transform()
                    .itrf_to_gcrs_matrix(epoch0 + t);
        } else {
            R.setIdentity();
        }

        for (int comp = 0; comp < 3; ++comp) {
            const Vector3d Rc = R.col(comp);
            AS_plus_B.block<3, 1>(3, param_idx(comp, i_left, n_nodes)) += w_left * Rc;
            if (i_right != i_left) {
                AS_plus_B.block<3, 1>(3, param_idx(comp, i_right, n_nodes)) += w_right * Rc;
            }
        }
    };

    // 6. Observation times relative to epoch0, sorted, each with its original index.
    std::vector<std::pair<double, int>> sample_t_idx;
    sample_t_idx.reserve(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        const double t_obs = position_data[sample_indices[i]].first - epoch0;
        sample_t_idx.emplace_back(t_obs, i);
    }
    std::sort(sample_t_idx.begin(), sample_t_idx.end());
    int next_sample = 0;

    // 7. Step S over [0, t_end] with Heun's method.
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(6, n_params);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3 * n_samples, n_params);

    Eigen::Matrix<double, 6, 6> A_t;
    if (!compute_A(0.0, A_t)) {
        // No interpolable state at t = 0 (only with a degenerate trajectory). Return a zero
        // Jacobian: the caller's damped GN solve then makes no progress and its stall limit
        // ends the loop.
        std::cerr << "  [stm] WARNING: compute_A failed at t=0; returning zero J\n";
        out.J = Eigen::MatrixXd::Zero(3 * n_samples, n_params);
        return out;
    }

    double t = 0.0;
    int n_steps_taken = 0;

    while (t < t_end - 1e-12) {
        const double dt = std::min(dt_var_s, t_end - t);
        const double t_new = t + dt;

        // Predictor: dS_t = A_t · S + B_t
        Eigen::MatrixXd dS_t = A_t * S;
        add_B_to(t, dS_t);

        Eigen::MatrixXd S_pred = S + dt * dS_t;

        // Corrector: dS_tn = A_tn · S_pred + B_tn
        Eigen::Matrix<double, 6, 6> A_tn;
        if (!compute_A(t_new, A_tn)) {
            // The trajectory ended early (e.g. altitude cutoff): later observations keep zero
            // Jacobian rows.
            break;
        }
        Eigen::MatrixXd dS_tn = A_tn * S_pred;
        add_B_to(t_new, dS_tn);

        Eigen::MatrixXd S_new = S + (0.5 * dt) * (dS_t + dS_tn);

        // Record J rows for observation times within (t, t_new].
        while (next_sample < n_samples &&
               sample_t_idx[next_sample].first <= t_new + 1e-9)
        {
            const double t_obs = sample_t_idx[next_sample].first;
            const int idx = sample_t_idx[next_sample].second;
            const double frac = (dt > 1e-12) ? (t_obs - t) / dt : 0.0;
            const double f = std::clamp(frac, 0.0, 1.0);

            // S interpolated linearly between t and t_new.
            Eigen::MatrixXd S_obs = S + f * (S_new - S);

            // Rotate the position rows from GCRS to ECEF.
            const Epoch ep_obs = epoch0 + t_obs;
            const Matrix3d R = coords::detail::get_cached_transform()
                                  .gcrs_to_itrf_matrix(ep_obs);

            // J[3*idx : 3*idx+3, :] = R · S_obs.topRows(3)
            J.middleRows(3 * idx, 3) = R * S_obs.topRows(3);

            ++next_sample;
        }

        S = S_new;
        A_t = A_tn;
        t = t_new;
        ++n_steps_taken;
    }

    // Observation times past t_end (truncated arc) keep zero J rows.

    if (verbose) {
        std::cout << "  [stm] dt=" << dt_var_s
                  << " n_steps=" << n_steps_taken
                  << " t_end=" << std::fixed << std::setprecision(2) << t_end
                  << " ‖S_final‖_F=" << std::scientific << std::setprecision(2) << S.norm()
                  << " ‖J‖_F=" << J.norm()
                  << " samples_done=" << next_sample << "/" << n_samples
                  << "\n";
    }

    out.J = J;
    return out;
}

}  // namespace detail
}  // namespace calibration
}  // namespace ballistic
