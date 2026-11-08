#pragma once

// =============================================================================
// Gauss-Newton refinement of an empirical acceleration
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration fitted to
// observations and added to the force model. Gauss-Newton (GN) refinement
// takes an existing EA (from the B-spline estimator or anywhere else) and
// adjusts its values at a set of time nodes so that the orbit propagated with
// it passes as close as possible to observed positions (least squares on the
// position residuals). Each iteration linearizes "propagated positions as a
// function of the EA node values" — the Jacobian — either by finite
// differences (FD: perturb each value and propagate again) or with the state
// transition matrix (STM: sensitivity equations integrated alongside the
// orbit in one pass), then takes a damped step (Levenberg–Marquardt (LM)
// damping: a term added to the diagonal that shortens the step when the
// linearization is poor).
// =============================================================================

#include "../core/types.hpp"
#include "../time/epoch.hpp"
#include "../trajectory.hpp"
#include "../trajectory_interpolator.hpp"
#include "../forces/empirical_acceleration.hpp"
#include "../coordinates/geodetic.hpp"
#include "detail/calibration_constants.hpp"
#include "detail/gn_stm.hpp"
#include "detail/line_search.hpp"
#include "detail/run_propagation.hpp"

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace ballistic {
namespace calibration {

/// Full settings of `refine_ea_gn` as used inside the library. The public
/// estimators build it from the simpler `GNRefine` (ea_estimator.hpp).
struct GNRefineConfig {
    int max_iterations = 30;         // maximum GN iterations
    double fd_delta = 1e-2;          // finite-difference perturbation of an EA value [m/s²]
    int sample_stride = 100;         // use every N-th observation
    int ea_subsample = 4;            // keep every N-th EA node as a parameter (1 = all);
                                     // the last node is always kept
    double max_duration_s = 3600.0;  // ignore observations more than this many seconds
                                     // from epoch0 [s]
    double min_altitude_m = 0.0;     // stop propagation below this altitude [m]; 0 = no cutoff
    TrajectoryConfig trajectory_config;   // force model and integrator for each propagation

    // STM Jacobian. When true, the per-parameter finite-difference propagations
    // are replaced by one propagation that also integrates the sensitivity
    // matrix S(t) = ∂(state at t)/∂(EA node values). Requires an EA (GCRS or
    // ITRF) with linear interpolation between nodes (the EA built here is
    // linear when STM is on).
    bool use_stm = false;
    double stm_dt_s = 0.5;           // step [s] for integrating the sensitivity equations
                                     // (Heun's method, a 2nd-order Runge-Kutta)
    double stm_fd_dx = 1.0;          // position perturbation [m] for ∂a/∂r
    double stm_fd_dv = 1e-3;         // velocity perturbation [m/s] for ∂a/∂v
    bool stm_verify = false;         // on the first iteration also build the FD Jacobian
                                     // and print the difference (diagnostic)
    bool stm_verbose = false;        // print the STM computation's internal trace

    // Print one line per GN iteration (separate from stm_verbose).
    bool verbose = false;

    // Smoothness penalty (Tikhonov regularization) on the EA, the same one the
    // B-spline fit used, so refinement minimizes the same objective as the fit
    // it refines: reg_lambda · N · h^(2k+1) · Σ (k-th difference of the node
    // values / h^k)² · h, where k = reg_order, h = node spacing [s], N = number
    // of observations used. The scaling makes λ mean the same at any spacing.
    // reg_order 0 (or reg_lambda 0) turns it off. The public estimators fill
    // reg_order from GNRefine::reg_deriv_order, or the data kind's default.
    double reg_lambda = 1.0;
    int reg_order = 0;
};

/// Refine `initial_ea` so that the orbit started from (`pos_ecef` [m],
/// `vel_ecef` [m/s], ECEF/Earth-fixed, at `epoch0`) and propagated with the EA
/// best matches `position_data` (time, ECEF position [m]).
///
/// Returns a new tabulated EA on (a subsample of) the initial EA's nodes, in
/// the initial EA's frame. Returns `initial_ea` itself, unrefined, when it is
/// null or zero, when there are no observations, or when fewer than 3
/// observations remain after `sample_stride` and `max_duration_s`.
/// Stops after `max_iterations` or after repeated iterations that fail to
/// lower the objective. Exceptions from propagation (e.g. missing data files)
/// propagate.
[[nodiscard]] inline std::shared_ptr<forces::EmpiricalAcceleration> refine_ea_gn(
    const Epoch& epoch0,
    const Vector3d& pos_ecef,
    const Vector3d& vel_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& position_data,
    std::shared_ptr<forces::EmpiricalAcceleration> initial_ea,
    const GNRefineConfig& config)
{
    if (!initial_ea || initial_ea->is_zero() || position_data.empty())
        return initial_ea;

    const double fd_delta = config.fd_delta;

    // Pick the observations to fit
    std::vector<size_t> sample_indices;
    for (size_t i = 0; i < position_data.size(); i += config.sample_stride) {
        if (std::abs(position_data[i].first - epoch0) <= config.max_duration_s)
            sample_indices.push_back(i);
    }
    const int n_samples = static_cast<int>(sample_indices.size());
    if (n_samples < 3) return initial_ea;

    Eigen::VectorXd p_obs(3 * n_samples);
    for (int i = 0; i < n_samples; ++i)
        p_obs.segment<3>(3 * i) = position_data[sample_indices[i]].second;

    // Propagate with an EA and return the positions at the chosen observation
    // times (zeros where the trajectory does not reach a time).
    auto propagate_ea = [&](const std::shared_ptr<forces::EmpiricalAcceleration>& ea)
        -> Eigen::VectorXd
    {
        auto [traj, ti] = detail::run_propagation_arc(
            epoch0, pos_ecef, vel_ecef, config.trajectory_config, ea,
            config.max_duration_s, config.min_altitude_m, /*backward=*/false);
        Eigen::VectorXd positions(3 * n_samples);
        for (int i = 0; i < n_samples; ++i) {
            auto ms = ti->get_state(position_data[sample_indices[i]].first);
            positions.segment<3>(3 * i) = ms ? ms->position : Vector3d::Zero();
        }
        return positions;
    };

    // Keep every ea_subsample-th EA node to limit the number of parameters
    const auto& all_keys = initial_ea->keys();
    std::vector<double> ea_keys;
    for (size_t k = 0; k < all_keys.size(); k += config.ea_subsample)
        ea_keys.push_back(all_keys[k]);
    if (ea_keys.empty() || ea_keys.back() != all_keys.back())
        ea_keys.push_back(all_keys.back());  // Always include last point

    const int n_nodes = static_cast<int>(ea_keys.size());
    const int n_params = 3 * n_nodes;

    // Starting values: the initial EA evaluated at the kept nodes
    std::vector<Vector3d> ea_vals(n_nodes);
    for (int k = 0; k < n_nodes; ++k)
        ea_vals[k] = initial_ea->accel_at(ea_keys[k]);

    // How the refined EA interpolates between nodes depends on the Jacobian
    // method and on how smooth the starting EA is:
    //
    //   Finite differences: every integrator step samples the table's
    //     interpolation. If the true EA is smoother than that, the FD
    //     derivatives pick up kinks at the nodes. The second derivative of a
    //     cubic B-spline is piecewise linear, so Linear interpolation is exact
    //     for it; a quintic (degree 5) spline's is cubic and needs Cubic.
    //
    //   STM: the sensitivity equations assume Linear interpolation (gn_stm.hpp
    //     does not check it), so Cubic + STM is not supported. Its derivatives are

    //     computed directly, not by differencing, so Linear works there even
    //     for smoother starting EAs.
    //
    // So: Cubic when FD is used and the starting EA is a B-spline of degree > 3;
    // Linear otherwise (including tabulated starting EAs). Using Linear for FD
    // on the default degree-5 spline was measured to make it clearly worse.
    forces::EAInterp interp = forces::EAInterp::Linear;
    if (!config.use_stm) {
        if (auto bs = initial_ea->as_bspline(); bs && bs->degree > 3) {
            interp = forces::EAInterp::Cubic;
        }
    }
    const forces::EmpiricalAccelFrame frame = initial_ea->frame();
    auto build_ea = [&](const std::vector<Vector3d>& vals) {
        std::vector<forces::EmpiricalAccelEntry> entries;
        entries.reserve(static_cast<std::size_t>(n_nodes));
        for (int k = 0; k < n_nodes; ++k)
            entries.emplace_back(ea_keys[k], vals[k]);
        return std::make_shared<forces::EmpiricalAcceleration>(
            forces::EmpiricalAcceleration::from_entries(
                std::move(entries), frame, interp));
    };
    auto result_ea = build_ea(ea_vals);

    if (config.verbose) {
        std::cout << "  [gn-refine] n_nodes=" << n_nodes << " n_params=" << n_params
                  << " (subsampled from " << all_keys.size() << ")" << std::endl;
    }

    // Smoothness penalty: LᵀL built from k-th differences on the node grid,
    // separately for each of the 3 components, weighted by node spacing so
    // that θᵀLᵀLθ ≈ ∫(dᵏa/dtᵏ)² dt. Scaled by λ·N·h^(2k+1) as in the B-spline
    // fit. It enters the normal equations (H, g) AND the step-acceptance test
    // below, so refinement minimizes the same objective as the fit it refines.
    const int reg_k = config.reg_order;
    double reg_coef = 0.0;
    Eigen::MatrixXd reg_LtL;
    if (config.reg_lambda > 0.0 && reg_k > 0 && n_nodes > reg_k) {
        double h_mean = (ea_keys.back() - ea_keys.front())
                        / std::max(1, n_nodes - 1);
        reg_coef = config.reg_lambda * static_cast<double>(n_samples)
                 * std::pow(h_mean, 2 * reg_k + 1);
        reg_LtL = Eigen::MatrixXd::Zero(n_params, n_params);
        std::vector<double> stencil(static_cast<std::size_t>(reg_k) + 1);
        for (int i = 0; i <= reg_k; ++i) {
            double c = 1.0;
            for (int j = 0; j < i; ++j) c = c * (reg_k - j) / (j + 1);
            stencil[static_cast<std::size_t>(i)] =
                ((reg_k - i) % 2 == 0 ? 1.0 : -1.0) * c;
        }
        for (int c = 0; c < 3; ++c) {
            const int base = c * n_nodes;
            for (int r = 0; r + reg_k < n_nodes; ++r) {
                double h = (ea_keys[static_cast<std::size_t>(r + reg_k)]
                            - ea_keys[static_cast<std::size_t>(r)]) / reg_k;
                if (h <= 0) continue;
                double w = std::sqrt(h) / std::pow(h, reg_k);
                for (int i = 0; i <= reg_k; ++i)
                    for (int j = 0; j <= reg_k; ++j)
                        reg_LtL(base + r + i, base + r + j) +=
                            w * stencil[static_cast<std::size_t>(i)]
                            * w * stencil[static_cast<std::size_t>(j)];
            }
        }
    }
    auto flatten_vals = [&](const std::vector<Vector3d>& vals) {
        Eigen::VectorXd th(n_params);
        for (int k = 0; k < n_nodes; ++k)
            for (int c = 0; c < 3; ++c)
                th(detail::param_idx(c, k, n_nodes)) = vals[static_cast<std::size_t>(k)](c);
        return th;
    };
    auto reg_penalty = [&](const std::vector<Vector3d>& vals) -> double {
        if (reg_coef <= 0.0) return 0.0;
        Eigen::VectorXd th = flatten_vals(vals);
        return reg_coef * th.dot(reg_LtL * th);
    };

    double lm_lambda = 1e-4;
    int stall_count = 0;

    // Finite-difference Jacobian: one extra propagation per parameter. Used
    // directly when use_stm is false and as a cross-check when stm_verify is set.
    auto build_fd_jacobian = [&](const Eigen::VectorXd& p_nom_local) -> Eigen::MatrixXd {
        Eigen::MatrixXd J_fd(3 * n_samples, n_params);
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < n_params; ++j) {
            int node = j % n_nodes;
            int comp = j / n_nodes;

            std::vector<Vector3d> pert_vals = ea_vals;
            pert_vals[static_cast<std::size_t>(node)](comp) += fd_delta;
            auto pert_ea = build_ea(pert_vals);
            J_fd.col(j) = (propagate_ea(pert_ea) - p_nom_local) / fd_delta;
        }
        return J_fd;
    };

    for (int gn_iter = 0; gn_iter < config.max_iterations; ++gn_iter) {
        Eigen::VectorXd p_nom;
        Eigen::MatrixXd J;

        if (config.use_stm) {
            auto stm = detail::build_stm_jacobian_and_pnom(
                epoch0, pos_ecef, vel_ecef, config.trajectory_config, result_ea,
                sample_indices, position_data, ea_keys, n_nodes, n_params,
                frame,
                config.max_duration_s, config.min_altitude_m,
                config.stm_dt_s, config.stm_fd_dx, config.stm_fd_dv,
                config.stm_verbose && gn_iter == 0);
            p_nom = std::move(stm.p_nom);
            J = std::move(stm.J);

            if (config.stm_verify && gn_iter == 0) {
                Eigen::MatrixXd J_fd = build_fd_jacobian(p_nom);
                const double num = (J - J_fd).norm();
                const double den = std::max(J_fd.norm(), detail::COL_NORM_FLOOR);
                std::cout << "  [stm-verify] iter " << (gn_iter + 1)
                          << ": ‖J_stm-J_fd‖_F=" << std::scientific << std::setprecision(2) << num
                          << " ‖J_fd‖_F=" << J_fd.norm()
                          << " rel=" << (num / den) << "\n";
            }
        } else {
            p_nom = propagate_ea(result_ea);
            J = build_fd_jacobian(p_nom);
        }

        Eigen::VectorXd residual = p_obs - p_nom;
        double sse_before = residual.squaredNorm();

        // Scale each Jacobian column to unit length (better-conditioned solve)
        Eigen::VectorXd col_norms(n_params);
        for (int j = 0; j < n_params; ++j) {
            col_norms(j) = J.col(j).norm();
            if (col_norms(j) > detail::COL_NORM_FLOOR) J.col(j) /= col_norms(j);
        }

        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd g = J.transpose() * residual;
        if (reg_coef > 0.0) {
            // Add the penalty in the scaled coordinates (θ = D_c⁻¹ θ_s, D_c =
            // the column norms).
            Eigen::VectorXd th = flatten_vals(ea_vals);
            Eigen::VectorXd gpen = reg_coef * (reg_LtL * th);
            for (int a = 0; a < n_params; ++a) {
                double ca = (col_norms(a) > detail::COL_NORM_FLOOR) ? col_norms(a) : 1.0;
                g(a) -= gpen(a) / ca;
                for (int b = 0; b < n_params; ++b) {
                    if (reg_LtL(a, b) == 0.0) continue;
                    double cb = (col_norms(b) > detail::COL_NORM_FLOOR) ? col_norms(b) : 1.0;
                    H(a, b) += reg_coef * reg_LtL(a, b) / (ca * cb);
                }
            }
        }
        for (int j = 0; j < n_params; ++j) H(j, j) += lm_lambda;
        Eigen::VectorXd delta_scaled = H.ldlt().solve(g);

        Eigen::VectorXd delta(n_params);
        for (int j = 0; j < n_params; ++j)
            delta(j) = (col_norms(j) > detail::COL_NORM_FLOOR) ? delta_scaled(j) / col_norms(j) : 0.0;

        // Try shorter and shorter fractions of the step
        // (detail::GN_BACKTRACK_STEPS). A step is accepted if it lowers the
        // full objective (sum of squared residuals + smoothness penalty), not
        // the residuals alone — otherwise the step search would undo the
        // penalty.
        const double obj_before = sse_before + reg_penalty(ea_vals);
        bool accepted = false;
        double step_scale = 0.0;
        for (double s : detail::GN_BACKTRACK_STEPS) {
            std::vector<Vector3d> trial_vals(static_cast<std::size_t>(n_nodes));
            for (int k = 0; k < n_nodes; ++k) {
                Vector3d v = ea_vals[k];
                v.x() += s * delta(detail::param_idx(0, k, n_nodes));
                v.y() += s * delta(detail::param_idx(1, k, n_nodes));
                v.z() += s * delta(detail::param_idx(2, k, n_nodes));
                trial_vals[static_cast<std::size_t>(k)] = v;
            }
            auto trial_ea = build_ea(trial_vals);

            double trial_sse = (p_obs - propagate_ea(trial_ea)).squaredNorm()
                             + reg_penalty(trial_vals);
            if (trial_sse < obj_before) {
                for (int k = 0; k < n_nodes; ++k) {
                    ea_vals[k].x() += s * delta(detail::param_idx(0, k, n_nodes));
                    ea_vals[k].y() += s * delta(detail::param_idx(1, k, n_nodes));
                    ea_vals[k].z() += s * delta(detail::param_idx(2, k, n_nodes));
                }
                result_ea = trial_ea;
                step_scale = s;
                accepted = true;
                lm_lambda *= 0.3;
                break;
            }
        }

        double rmse = std::sqrt(sse_before / n_samples);
        double delta_norm = delta.norm();
        double trial_rmse = accepted ?
            std::sqrt((p_obs - propagate_ea(result_ea)).squaredNorm() / n_samples) : rmse;
        if (accepted) {
            stall_count = 0;
            double rel_impr = (rmse > 0) ? (rmse - trial_rmse) / rmse : 0;
            // After a big improvement reset the damping, so damping built up
            // earlier does not block the next step
            if (rel_impr > 0.15) lm_lambda = 1e-4;
            if (config.verbose) {
                std::cout << "  [gn-refine] iter " << (gn_iter + 1)
                          << ": rmse=" << std::fixed << std::setprecision(4) << rmse
                          << "->" << trial_rmse << "m (s=" << step_scale
                          << " |d|=" << std::scientific << std::setprecision(1) << delta_norm
                          << " lm=" << lm_lambda
                          << " rel=" << std::fixed << std::setprecision(1) << (rel_impr*100) << "%)" << std::endl;
            }
        } else {
            lm_lambda *= 3.0;  // no step accepted: increase damping
            stall_count++;
            if (config.verbose) {
                std::cout << "  [gn-refine] iter " << (gn_iter + 1)
                          << ": rmse=" << std::fixed << std::setprecision(4) << rmse
                          << "m (stall " << stall_count
                          << " |d|=" << std::scientific << std::setprecision(1) << delta_norm
                          << " lm=" << lm_lambda << ")" << std::endl;
            }
            if (stall_count >= detail::STALL_LIMIT) break;
        }
    }

    return result_ea;
}

// Note: GN refinement works on a tabulated EA, not directly on B-spline
// coefficients. Refining the B-spline coefficients by finite differences was
// measured to be (a) impractically slow on the reentry-glide test scenario (one
// FD propagation per coefficient, every iteration) and (b) no more accurate
// where it did converge.

} // namespace calibration
} // namespace ballistic
