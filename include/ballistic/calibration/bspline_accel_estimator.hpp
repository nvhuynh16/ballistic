#pragma once

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

// =============================================================================
// B-spline empirical-acceleration estimator (implementation)
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration fitted to
// observations and added to the force model. This estimator propagates the
// orbit with the force model, takes the residuals (observed − modeled), fits
// one smooth B-spline curve to them over the whole arc, and differentiates the
// curve analytically: twice for position residuals, once for velocity, not at
// all for acceleration. The result approximates the acceleration the force
// model is missing. Propagating again with that EA and repeating refines it.
//
// Why a spline over the whole arc: how well the fit is determined depends on
// the knot spacing (the time between spline pieces, seconds), not on the time
// between observations. Differencing neighboring observations instead would
// fail at high rates — at 300 Hz the second difference of positions divides by
// ~dt², amplifying noise by orders of magnitude — whereas one least-squares
// fit over the whole arc stays well conditioned.
//
// The least-squares normal equations are accumulated one observation at a
// time: O(N·p²) time for N observations and spline degree p, and
// O(n_basis²) memory. 180k observations at 300 Hz with 30 basis functions
// give a 30×30 system.
//
// A Tikhonov penalty (a regularization term λ·∫(dᵏf/dtᵏ)² dt added to the
// least-squares objective) on a chosen derivative of the spline keeps the EA
// smooth; λ sets how strongly.
//
// Public callers use estimate_ea_bspline (ea_estimator.hpp); everything here
// is in namespace detail.
// =============================================================================

#include "ea_decomposition.hpp"
#include "detail/bspline_time_grid.hpp"
#include "detail/bspline_eval.hpp"
#include "detail/bspline_gcv.hpp"
#include "detail/calibration_constants.hpp"
#include "detail/line_search.hpp"
#include "detail/run_propagation.hpp"
#include "../math/bspline_basis.hpp"
#include "../math/brent_minimize.hpp"
#include "../forces/empirical_acceleration.hpp"
#include "../trajectory.hpp"
#include "../trajectory_interpolator.hpp"
#include "../coordinates/geodetic.hpp"
#include "../coordinates/iau_transforms.hpp"
#include "../util/frame_transform.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <optional>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace ballistic {
namespace calibration {
namespace detail {

// =========================================================================
// Configuration (internal — public callers use EABSplineConfig, whose
// defaults differ: degree 5, knot spacing 2 s, Brent line search on)
// =========================================================================

struct BSplineAccelConfig {
    double knot_spacing_s = 20.0;       // spacing of interior knots [s]
    double lambda = 1e-4;               // Tikhonov weight λ (FIXED: used as is; DECREASING: λ₀)
    double max_duration_s = 3600.0;     // fit observations within this many seconds of epoch0
    forces::EmpiricalAccelFrame frame = forces::EmpiricalAccelFrame::GCRS;  // EA fit frame
    TrajectoryConfig trajectory_config; // force model and integrator
    double min_altitude_m = 50000.0;    // stop propagation below this altitude [m]
    bool backward = false;              // propagate backward in time from epoch0
    int max_iterations = 20;            // outer iterations (propagate, fit, update)
    double lambda_drift = 0.0;          // penalty on the spline values ‖f‖² (velocity data:
                                        // keeps the implied position from drifting)

    // B-spline degree: 3 = cubic, 5 = quintic. Quintic allows a smooth
    // penalty on the third derivative (reg_deriv_order = 3).
    int degree = 3;

    // Which derivative k of the spline f the penalty λ·∫(dᵏf/dtᵏ)² acts on:
    //  -1 = automatic: 2 for position data, 1 for velocity/acceleration data
    //   1 = ‖f'‖²  — velocity data: penalizes EA size; acceleration data: EA change
    //   2 = ‖f''‖² — position data: penalizes EA size
    //   3 = ‖f'''‖² — position data: penalizes EA change (needs degree ≥ 3; quintic advised)
    int reg_deriv_order = -1;

    // How λ is chosen at each iteration:
    //  FIXED:      config.lambda unchanged
    //  GCV:        generalized cross-validation (GCV) picks λ from the data each
    //              iteration (no noise level needed)
    //  DECREASING: iteratively regularized Gauss-Newton (IRGN) schedule
    //              λ_k = max(lambda_min, lambda · decay^k) — strong smoothing
    //              first, relaxed as the fit improves
    detail::LambdaMethod lambda_method = detail::LambdaMethod::DECREASING;
    double lambda_decay = 0.5;           // decay factor per iteration for DECREASING (< 1)
    double lambda_min = 1e-12;           // lower limit for DECREASING
    double gcv_lambda_min = 1e-12;       // GCV search interval [gcv_lambda_min,
    double gcv_lambda_max = 1e-1;        //   gcv_lambda_max] (ea::GCV{lam_min, lam_max})
    double noise_sigma = 0.0;            // observation noise std dev (data units);
                                         // > 0 enables early stopping

    // (No Gauss-Newton settings here: public callers use EABSplineConfig::gn;
    // direct callers of the detail estimators call refine_ea_gn
    // (gn_refine.hpp) on the result.)

    // Step-length search along each update: bracket the best step from 5
    // trial lengths, then refine by Brent's method, instead of trying 24 fixed
    // lengths. Off here; EABSplineConfig turns it on.
    bool brent_line_search = false;

    // Print one diagnostic line per iteration to std::cout.
    bool verbose = false;
};

// Kind of observations being fitted.
enum class BSplineObsType { POSITION, VELOCITY, ACCELERATION };

// Least-squares fit of a clamped B-spline f(t) (3 components) to residuals dp
// at times tau, with the penalty λ·N·h^(2k+1)·Σ(dᵏf/dtᵏ)² over regularly spaced
// points (k = reg_deriv_order, h = knot spacing, N = number of samples; the
// scaling keeps λ meaning the same at any knot spacing and data rate) plus an
// optional λ_drift·N·h·Σ f² penalty. tau must be sorted ascending; λ ≥ 0.
//   degree: 3 = cubic, 5 = quintic
//   reg_deriv_order 1/2/3: penalize f', f'' or f''' (see BSplineAccelConfig)
[[nodiscard]] inline BSplineFitResult fit_bspline(
    const std::vector<double>& tau,       // relative times
    const std::vector<Vector3d>& dp,      // residuals in estimation frame
    double knot_spacing_s,
    double lambda,                        // Tikhonov weight ≥ 0 (GCV runs in the caller)
    int reg_deriv_order = 2,
    double lambda_drift = 0.0,            // penalty on ‖f‖² (velocity data: position drift)
    int degree = BSPLINE_DEFAULT_DEGREE)  // B-spline degree (3 = cubic, 5 = quintic; the public
                                          // default is 5 via EABSplineConfig)
{
    assert(degree >= 1 && degree <= detail::BSPLINE_MAX_DEGREE);
    assert(lambda >= 0.0);  // automatic λ: set lambda_method in BSplineAccelConfig
    const int N = static_cast<int>(tau.size());
    const double t_start = tau.front();
    const double t_end = tau.back();
    const double arc_length = t_end - t_start;

    // Number of interior knots
    int n_interior = static_cast<int>(std::floor(arc_length / knot_spacing_s)) - 1;
    if (n_interior < 0) n_interior = 0;

    // Build knot vector (make_clamped_knot_vector is degree-generic)
    auto knots = math::make_clamped_knot_vector(t_start, t_end, n_interior, degree);
    const int n_basis = n_interior + degree + 1;
    const int n_knots = static_cast<int>(knots.size());
    const int n_idx = n_basis - 1;  // n for findspan (last basis function index)

    // Accumulate the normal equations H = BᵀB + penalty, g = Bᵀdp (B = basis
    // function values at the sample times)
    std::vector<Eigen::Triplet<double>> H_triplets;
    H_triplets.reserve(N * (degree + 1) * (degree + 1));
    Eigen::VectorXd gx = Eigen::VectorXd::Zero(n_basis);
    Eigen::VectorXd gy = Eigen::VectorXd::Zero(n_basis);
    Eigen::VectorXd gz = Eigen::VectorXd::Zero(n_basis);

    const double T = t_end - t_start;
    std::vector<double> Ni(degree + 1);
    for (int i = 0; i < N; ++i) {
        int span = math::bspline_findspan(n_idx, degree, tau[i], knots.data());
        math::bspline_basis_funs(degree, span, tau[i], knots.data(), Ni.data());
        int col0 = span - degree;
        for (int a = 0; a <= degree; ++a) {
            int ca = col0 + a;
            gx(ca) += Ni[a] * dp[i].x();
            gy(ca) += Ni[a] * dp[i].y();
            gz(ca) += Ni[a] * dp[i].z();
            for (int b = a; b <= degree; ++b) {
                int cb = col0 + b;
                double val = Ni[a] * Ni[b];
                H_triplets.emplace_back(ca, cb, val);
                if (ca != cb) H_triplets.emplace_back(cb, ca, val);
            }
        }
    }

    // Points where the penalties are evaluated: the Greville abscissae (the
    // average of each basis function's knots, roughly its center) and the
    // midpoints between them
    std::vector<double> reg_points;
    if ((lambda > 0.0 && reg_deriv_order >= 1 && reg_deriv_order <= degree) ||
        lambda_drift > 0.0) {
        std::vector<double> greville(n_basis);
        for (int j = 0; j < n_basis; ++j) {
            double sum = 0.0;
            for (int k = 1; k <= degree; ++k) {
                sum += knots[j + k];
            }
            greville[j] = sum / degree;
        }
        reg_points.reserve(2 * n_basis);
        for (int j = 0; j < n_basis; ++j) {
            reg_points.push_back(greville[j]);
        }
        for (int j = 0; j < n_basis - 1; ++j) {
            reg_points.push_back(0.5 * (greville[j] + greville[j + 1]));
        }
    }

    // Derivative penalty on dᵏf/dtᵏ (k = reg_deriv_order), scaled by N·h^(2k+1)
    // so it balances the data term BᵀB (≈ (N/T)·M, M = the basis functions'
    // overlap integrals) whatever the knot spacing and number of samples.
    const double h_pow = std::pow(knot_spacing_s, 2 * reg_deriv_order + 1);
    const double reg_scale = static_cast<double>(N) * h_pow;
    const double drift_scale = static_cast<double>(N) * knot_spacing_s;
    if (lambda > 0.0 && reg_deriv_order >= 1 && reg_deriv_order <= degree) {
        std::vector<double> ders((degree + 1) * (reg_deriv_order + 1));
        for (double t : reg_points) {
            double tc = std::clamp(t, t_start, t_end);
            int span = math::bspline_findspan(n_idx, degree, tc, knots.data());
            math::bspline_ders(degree, span, tc, knots.data(), reg_deriv_order, ders.data());

            int col0 = span - degree;
            const double* drow = &ders[reg_deriv_order * (degree + 1)];
            for (int a = 0; a <= degree; ++a) {
                int ca = col0 + a;
                for (int b = a; b <= degree; ++b) {
                    int cb = col0 + b;
                    double val = reg_scale * lambda * drow[a] * drow[b];
                    H_triplets.emplace_back(ca, cb, val);
                    if (ca != cb) H_triplets.emplace_back(cb, ca, val);
                }
            }
        }
    }

    // Drift penalty on the spline values ‖f‖²
    if (lambda_drift > 0.0) {
        std::vector<double> Ni_reg(degree + 1);
        for (double t : reg_points) {
            double tc = std::clamp(t, t_start, t_end);
            int span = math::bspline_findspan(n_idx, degree, tc, knots.data());
            math::bspline_basis_funs(degree, span, tc, knots.data(), Ni_reg.data());

            int col0 = span - degree;
            for (int a = 0; a <= degree; ++a) {
                int ca = col0 + a;
                for (int b = a; b <= degree; ++b) {
                    int cb = col0 + b;
                    double val = drift_scale * lambda_drift * Ni_reg[a] * Ni_reg[b];
                    H_triplets.emplace_back(ca, cb, val);
                    if (ca != cb) H_triplets.emplace_back(cb, ca, val);
                }
            }
        }
    }

    // Assemble the sparse H and solve for each component's coefficients
    Eigen::SparseMatrix<double> H(n_basis, n_basis);
    H.setFromTriplets(H_triplets.begin(), H_triplets.end());
    H_triplets.clear();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(H);

    BSplineFitResult result;
    result.cx = ldlt.solve(gx);
    result.cy = ldlt.solve(gy);
    result.cz = ldlt.solve(gz);
    result.knots = std::move(knots);
    result.n_basis = n_basis;
    result.t_start = t_start;
    result.t_end = t_end;
    result.degree = degree;
    result.lambda_used = lambda;
    return result;
}

// Wrap fitted coefficients as an EA stored as a B-spline. ea_deriv_order says
// what the spline represents: 0 = the acceleration itself, 1 = a velocity
// (EA = f'), 2 = a position (EA = f'').
[[nodiscard]] inline std::shared_ptr<forces::EmpiricalAcceleration> build_ea_bspline(
    const BSplineFitResult& fit,
    int ea_deriv_order,
    forces::EmpiricalAccelFrame frame)
{
    forces::BSplineCoefficients coeffs(fit.cx, fit.cy, fit.cz, fit.knots, fit.degree);

    forces::EARepresents represents = forces::EARepresents::Acceleration;
    switch (ea_deriv_order) {
        case 0: represents = forces::EARepresents::Acceleration; break;
        case 1: represents = forces::EARepresents::Velocity;     break;
        case 2: represents = forces::EARepresents::Position;     break;
        default: break;
    }

    return std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_bspline(
            std::move(coeffs), represents, frame));
}

// =========================================================================
// BSplineEstimator — the iteration loop (propagate, take residuals, fit,
// step), split into named methods with the loop state held in members.
// estimate_bspline_impl just constructs one and calls run().
// =========================================================================

template<BSplineObsType ObsType>
struct BSplineEstimator {
    // ----- compile-time config from ObsType -----
    static constexpr int default_reg_order =
        (ObsType == BSplineObsType::POSITION) ? 2 : 1;
    static constexpr int ea_deriv_order =
        (ObsType == BSplineObsType::POSITION) ? 2 :
        (ObsType == BSplineObsType::VELOCITY) ? 1 : 0;
    static constexpr const char* obs_label =
        (ObsType == BSplineObsType::POSITION) ? "pos" :
        (ObsType == BSplineObsType::VELOCITY) ? "vel" : "acc";
    static constexpr const char* obs_unit =
        (ObsType == BSplineObsType::ACCELERATION) ? "m/s2" : "m";

    // ----- inputs (held by reference; lifetime is the caller's) -----
    const Epoch& epoch0;
    const Vector3d& position_ecef;
    const Vector3d& velocity_ecef;
    const std::vector<std::pair<Epoch, Vector3d>>& obs_data;
    const BSplineAccelConfig& config;

    // ----- derived once at construction -----
    int reg_deriv_order;
    int degree;
    bool estimation_in_gcrs;
    PropagationFrame prop_frame;
    coords::ITRFRealization itrf_real;

    // ----- state carried between iterations -----
    EAResult result;
    std::shared_ptr<forces::EmpiricalAcceleration> accumulated_accel;
    std::shared_ptr<forces::EmpiricalAcceleration> best_accel;
    double best_sse = (std::numeric_limits<double>::max)();
    BSplineFitResult last_fit;
    Eigen::VectorXd accum_cx, accum_cy, accum_cz;
    double cached_lambda;
    double cached_sigma_sq = -1.0;
    int stall_count = 0;
    // Velocity data only: positions obtained by integrating the observed
    // velocities, used to judge which iteration's EA is best
    std::vector<std::pair<time::Epoch, Vector3d>> vel_integrated_positions;
    double best_pos_proxy_sse = (std::numeric_limits<double>::max)();
    std::shared_ptr<forces::EmpiricalAcceleration> best_pos_accel;

    BSplineEstimator(const Epoch& e,
                     const Vector3d& p,
                     const Vector3d& v,
                     const std::vector<std::pair<Epoch, Vector3d>>& obs,
                     const BSplineAccelConfig& c)
        : epoch0(e), position_ecef(p), velocity_ecef(v), obs_data(obs), config(c),
          reg_deriv_order(c.reg_deriv_order >= 0 ? c.reg_deriv_order : default_reg_order),
          degree(c.degree),
          estimation_in_gcrs(c.frame == forces::EmpiricalAccelFrame::GCRS),
          prop_frame(PropagationFrame::GCRS),  // propagation is GCRS-only
          itrf_real(c.trajectory_config.itrf),
          cached_lambda(c.lambda)
    {}

    // ---------------------------------------------------------------------
    // Propagation and scoring.
    // ---------------------------------------------------------------------

    // Propagate with `ea` forward (or backward) for max_duration_s, stopping
    // below min_altitude_m.
    auto propagate_arc(const std::shared_ptr<forces::EmpiricalAcceleration>& ea) const
        -> std::pair<std::unique_ptr<Trajectory>, std::unique_ptr<TrajectoryInterpolator>>
    {
        return detail::run_propagation_arc(
            epoch0, position_ecef, velocity_ecef, config.trajectory_config,
            ea, config.max_duration_s, config.min_altitude_m, config.backward);
    }

    // Sum of squared observation residuals after propagating with `ea`.
    // Returns the largest double if the propagation stopped short of
    // TRUNCATION_GUARD × max_duration_s, so such a candidate always loses the
    // step-length comparison.
    double compute_sse(const std::shared_ptr<forces::EmpiricalAcceleration>& ea) const {
        auto [traj, ti] = propagate_arc(ea);
        const double traj_span = std::abs(traj->epoch() - epoch0);
        if (traj_span < config.max_duration_s * detail::TRUNCATION_GUARD) {
            return (std::numeric_limits<double>::max)();
        }
        double sse = 0.0;
        for (const auto& [obs_epoch, obs_val] : obs_data) {
            if (std::abs(obs_epoch - epoch0) > config.max_duration_s) continue;
            auto ms = ti->get_state(obs_epoch);
            if (!ms) continue;
            if constexpr (ObsType == BSplineObsType::POSITION) {
                sse += (obs_val - ms->position).squaredNorm();
            } else if constexpr (ObsType == BSplineObsType::VELOCITY) {
                StateVector s_ecef{ms->position, ms->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, obs_epoch, itrf_real);
                sse += (obs_val - s_eci.velocity).squaredNorm();
            } else {
                StateVector s_ecef{ms->position, ms->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, obs_epoch, itrf_real);
                Vector3d a_model = traj->force_model().acceleration(obs_epoch, s_eci);
                sse += (obs_val - a_model).squaredNorm();
            }
        }
        return sse;
    }

    // EA with coefficients (accumulated + alpha × fit). Exact, because the EA
    // is linear in the coefficients.
    std::shared_ptr<forces::EmpiricalAcceleration>
    build_ea_from_increment(const BSplineFitResult& fit, double alpha) const {
        BSplineFitResult combined;
        combined.knots = fit.knots;
        combined.n_basis = fit.n_basis;
        combined.t_start = fit.t_start;
        combined.t_end = fit.t_end;
        combined.degree = fit.degree;
        if (accum_cx.size() > 0) {
            combined.cx = accum_cx + alpha * fit.cx;
            combined.cy = accum_cy + alpha * fit.cy;
            combined.cz = accum_cz + alpha * fit.cz;
        } else {
            combined.cx = alpha * fit.cx;
            combined.cy = alpha * fit.cy;
            combined.cz = alpha * fit.cz;
        }
        return build_ea_bspline(combined, ea_deriv_order, config.frame);
    }

    // ---------------------------------------------------------------------
    // Per-iteration phases.
    // ---------------------------------------------------------------------

    // Velocity data only: integrate the observed velocities (trapezoidal
    // rule, in the inertial GCRS frame) to positions, used to choose the best
    // iteration. Runs once before the loop; does nothing for other data.
    void init_vel_integrated_positions() {
        if constexpr (ObsType == BSplineObsType::VELOCITY) {
            if (obs_data.size() <= 1) return;
            Vector3d p_gcrs = coords::itrf_to_gcrs_full(position_ecef, epoch0, itrf_real);
            vel_integrated_positions.reserve(obs_data.size() / detail::VEL_PROXY_STRIDE + 1);
            vel_integrated_positions.emplace_back(epoch0, position_ecef);
            for (size_t i = 1; i < obs_data.size(); ++i) {
                const double dt = obs_data[i].first - obs_data[i - 1].first;
                p_gcrs += 0.5 * (obs_data[i - 1].second + obs_data[i].second) * dt;
                if (i % detail::VEL_PROXY_STRIDE == 0) {
                    Vector3d p_ecef = coords::gcrs_to_itrf_full(
                        p_gcrs, obs_data[i].first, itrf_real);
                    vel_integrated_positions.emplace_back(obs_data[i].first, p_ecef);
                }
            }
        }
    }

    struct ResidualsOut {
        std::vector<double> tau;
        std::vector<Vector3d> dp;
        double obs_sse;
    };

    // Residual (observed − modeled) for each observation, in the frame the
    // EA is fitted in. Returns nullopt when fewer than 4 observations fall
    // inside the propagated arc; the caller treats that as a failure.
    std::optional<ResidualsOut> compute_residuals(
        const Trajectory& ref_traj,
        TrajectoryInterpolator& interp) const
    {
        ResidualsOut out;
        out.obs_sse = 0.0;
        out.tau.reserve(obs_data.size());
        out.dp.reserve(obs_data.size());

        for (const auto& [obs_epoch, obs_val] : obs_data) {
            const double t_rel = obs_epoch - epoch0;
            if (std::abs(t_rel) > config.max_duration_s) continue;
            auto model_state = interp.get_state(obs_epoch);
            if (!model_state) continue;

            Vector3d residual;
            if constexpr (ObsType == BSplineObsType::POSITION) {
                residual = obs_val - model_state->position;  // ECEF
                out.obs_sse += residual.squaredNorm();
                if (estimation_in_gcrs) {
                    residual = coords::itrf_to_gcrs_vector_full(residual, obs_epoch);
                }
            } else if constexpr (ObsType == BSplineObsType::VELOCITY) {
                StateVector s_ecef{model_state->position, model_state->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, obs_epoch, itrf_real);
                residual = obs_val - s_eci.velocity;  // both GCRS
                out.obs_sse += residual.squaredNorm();
            } else {
                StateVector s_ecef{model_state->position, model_state->velocity};
                StateVector s_eci = util::itrf_to_gcrs(s_ecef, obs_epoch, itrf_real);
                Vector3d a_model = ref_traj.force_model().acceleration(obs_epoch, s_eci);
                residual = obs_val - a_model;  // both GCRS
                out.obs_sse += residual.squaredNorm();
            }
            out.tau.push_back(t_rel);
            out.dp.push_back(residual);
        }
        if (out.tau.size() < 4) return std::nullopt;
        if (out.tau.front() > out.tau.back()) {
            std::reverse(out.tau.begin(), out.tau.end());
            std::reverse(out.dp.begin(), out.dp.end());
        }
        return out;
    }

    // λ for this iteration: the decreasing schedule, or chosen by GCV (which
    // also updates cached_sigma_sq, the estimated noise variance).
    double effective_lambda_for_iter(int iter,
                                     const std::vector<double>& tau,
                                     const std::vector<Vector3d>& dp)
    {
        double lambda = cached_lambda;
        if (config.lambda_method == detail::LambdaMethod::DECREASING) {
            lambda = (std::max)(config.lambda_min,
                config.lambda * std::pow(config.lambda_decay, iter));
        }
        if (config.lambda_method == detail::LambdaMethod::GCV &&
            reg_deriv_order >= 1 && reg_deriv_order <= degree)
        {
            auto mats = detail::assemble_bspline_matrices(
                tau, dp, config.knot_spacing_s, reg_deriv_order, degree);
            lambda = detail::select_lambda_gcv(mats, config.gcv_lambda_min,
                                               config.gcv_lambda_max);
            // σ² at the GCV-optimal λ (Wahba 1990 §4.3)
            auto gcv_result = detail::compute_gcv(mats, lambda);
            cached_sigma_sq = gcv_result.sigma_sq;
        }
        return lambda;
    }

    struct LineSearchOut {
        double alpha = 0.0;
        std::shared_ptr<forces::EmpiricalAcceleration> trial_accel;
        double trial_sse = 0.0;
        bool accepted = false;
    };

    // Choose how far to step along the fitted update. The first iteration
    // always takes the full step; later ones search the step length (Brent or
    // a fixed set of lengths) and keep the one with the smallest residual sum,
    // or reject the update if none improves on the current one.
    LineSearchOut line_search(const BSplineFitResult& fit, double obs_sse, int iter) {
        LineSearchOut out;
        out.trial_sse = obs_sse;
        if (config.max_iterations <= 1 || iter == 0) {
            out.trial_accel = build_ea_from_increment(fit, 1.0);
            out.alpha = 1.0;
            out.accepted = true;
            return out;
        }
        double best_trial_sse = obs_sse;
        double best_alpha = 0.0;
        std::shared_ptr<forces::EmpiricalAcceleration> best_trial;
        auto try_alpha = [&](double a) {
            auto candidate = build_ea_from_increment(fit, a);
            const double candidate_sse = compute_sse(candidate);
            if (candidate_sse < best_trial_sse) {
                best_trial_sse = candidate_sse;
                best_alpha = a;
                best_trial = candidate;
            }
            return candidate_sse;
        };
        if (config.brent_line_search) {
            detail::brent_alpha_search(try_alpha);
        } else {
            detail::brute_alpha_search(try_alpha);
        }
        if (best_trial) {
            out.trial_accel = best_trial;
            out.trial_sse = best_trial_sse;
            out.alpha = best_alpha;
            out.accepted = true;
        }
        return out;
    }

    // Remember the best EA so far and count iterations without improvement.
    // Velocity data is judged against the integrated-velocity positions;
    // position/acceleration data by the residual sum.
    void update_best_tracking(bool accepted, double trial_sse) {
        if constexpr (ObsType == BSplineObsType::VELOCITY) {
            if (accepted && !vel_integrated_positions.empty()) {
                auto [proxy_traj, proxy_ti] = propagate_arc(accumulated_accel);
                double pos_sse = 0;
                for (const auto& [ep, p_int] : vel_integrated_positions) {
                    auto ms = proxy_ti->get_state(ep);
                    if (ms) pos_sse += (ms->position - p_int).squaredNorm();
                }
                if (pos_sse < best_pos_proxy_sse) {
                    best_pos_proxy_sse = pos_sse;
                    best_pos_accel = accumulated_accel;
                    stall_count = 0;
                } else {
                    ++stall_count;
                }
            } else {
                ++stall_count;
            }
        } else if (trial_sse < best_sse) {
            best_sse = trial_sse;
            best_accel = accumulated_accel;
            stall_count = 0;
        } else {
            ++stall_count;
        }
    }

    bool noise_floor_reached(double obs_rmse) const {
        return config.noise_sigma > 0.0 &&
               obs_rmse <= config.noise_sigma * detail::EARLY_STOP_SIGMA_MULT;
    }

    // ---- diagnostic prints (only when config.verbose) ----

    void log_iter(int iter, std::size_t n_obs, int n_basis, double obs_rmse,
                  double trial_rmse, bool accepted, double alpha,
                  double effective_lambda) const {
        if (!config.verbose) return;
        std::cout << "  [bspline-" << obs_label << "] iter " << (iter + 1)
                  << ": n_obs=" << n_obs
                  << " n_basis=" << n_basis
                  << " " << obs_label << "_rmse=" << std::fixed << std::setprecision(1)
                  << obs_rmse << obs_unit
                  << (accepted ? " -> " : " skip (")
                  << trial_rmse << (accepted ? obs_unit : (std::string(obs_unit) + ")"))
                  << " (a=" << std::setprecision(3) << alpha
                  << " lam=" << std::scientific << std::setprecision(1) << effective_lambda << ")"
                  << std::endl;
    }
    void log_noise_floor_stop(double obs_rmse) const {
        if (!config.verbose) return;
        std::cout << "  [bspline-" << obs_label << "] early stop: obs_rmse="
                  << std::fixed << std::setprecision(2) << obs_rmse
                  << " <= " << detail::EARLY_STOP_SIGMA_MULT << "*sigma="
                  << config.noise_sigma * detail::EARLY_STOP_SIGMA_MULT << std::endl;
    }
    void log_stall_stop(int iter) const {
        if (!config.verbose) return;
        std::cout << "  [bspline-" << obs_label << "] early stop: "
                  << stall_count << " consecutive stalls at iter " << (iter + 1) << std::endl;
    }

    // ---------------------------------------------------------------------
    // Post-loop output.
    // ---------------------------------------------------------------------

    // The EA to return: the best one recorded (for velocity data, the best by
    // integrated-velocity positions), or the latest if none was recorded yet.
    std::shared_ptr<forces::EmpiricalAcceleration> select_final_ea() const {
        if constexpr (ObsType == BSplineObsType::VELOCITY) {
            return best_pos_accel ? best_pos_accel
                                  : (best_accel ? best_accel : accumulated_accel);
        } else {
            return best_accel ? best_accel : accumulated_accel;
        }
    }

    // Fill collocation_history at the spline's knot times: for position data,
    // the propagated orbit plus the fitted residual spline (follows the data
    // without the drift a re-propagation accumulates); for velocity and
    // acceleration data, the orbit re-propagated with the final EA. Either
    // way it can be passed to decompose_to_beta. Also stores the trajectory
    // and interpolator on the result so cross-validation scoring can reuse them.
    void build_collocation_history() {
        if (!result.empirical_accel || result.empirical_accel->is_zero()) return;
        auto [final_traj_uniq, final_interp_uniq] = propagate_arc(result.empirical_accel);
        std::shared_ptr<Trajectory> final_traj = std::move(final_traj_uniq);
        std::shared_ptr<TrajectoryInterpolator> final_interp = std::move(final_interp_uniq);
        result.final_trajectory = final_traj;
        result.final_interp = final_interp;

        auto output_times = detail::knot_breakpoint_times(
            last_fit.knots, last_fit.degree, last_fit.t_start, last_fit.t_end);

        result.collocation_history.reserve(output_times.size());
        for (size_t i = 0; i < output_times.size(); ++i) {
            const double t = output_times[i];
            auto ms = final_interp->get_state(epoch0 + t);
            if (!ms) continue;
            Vector3d pos = ms->position;
            Vector3d vel = ms->velocity;
            if constexpr (ObsType == BSplineObsType::POSITION) {
                Vector3d dp = eval_d0(last_fit, t);
                Vector3d dv = eval_d1(last_fit, t);
                if (estimation_in_gcrs) {
                    dp = coords::gcrs_to_itrf_vector_full(dp, epoch0 + t);
                    dv = coords::gcrs_to_itrf_vector_full(dv, epoch0 + t);
                }
                pos += dp;
                vel += dv;
            }
            StateVector se{pos, vel};
            StateVector si = util::itrf_to_gcrs(se, epoch0 + t, itrf_real);
            Vector3d a_gcrs = final_traj->force_model().acceleration(epoch0 + t, si);
            Vector3d a_ecef = util::acceleration_to_itrf(a_gcrs, se, epoch0 + t);
            const double step =
                (i + 1 < output_times.size()) ? output_times[i + 1] - t : 0.0;
            result.collocation_history.emplace_back(epoch0 + t, t, pos, vel, a_ecef, step);
        }
    }

    // ---------------------------------------------------------------------
    // Main loop.
    // ---------------------------------------------------------------------
    EAResult run() {
        if (obs_data.size() < static_cast<size_t>(degree + 1)) {
            result.error_message = "Need at least " + std::to_string(degree + 1)
                + " observations for degree-" + std::to_string(degree) + " B-spline";
            return std::move(result);
        }

        init_vel_integrated_positions();

        for (int iter = 0; iter < config.max_iterations; ++iter) {
            // 1. Propagate with the EA accumulated so far.
            auto [ref_traj_ptr, interp_ptr] = propagate_arc(accumulated_accel);
            auto& ref_traj = *ref_traj_ptr;
            // Stop if that propagation ended early (e.g. an oscillating EA).
            if (iter > 0 && std::abs(ref_traj.epoch() - epoch0) <
                            config.max_duration_s * detail::TRUNCATION_GUARD) {
                break;
            }

            // 2. Compute residuals.
            auto residuals = compute_residuals(ref_traj, *interp_ptr);
            if (!residuals) {
                result.error_message = "Insufficient valid observations after filtering";
                return std::move(result);
            }
            auto& tau = residuals->tau;
            auto& dp  = residuals->dp;
            const double obs_sse = residuals->obs_sse;

            // 3. Fit B-spline to residuals.
            const double effective_lambda = effective_lambda_for_iter(iter, tau, dp);
            const double drift_lambda =
                (ObsType == BSplineObsType::VELOCITY) ? config.lambda_drift : 0.0;
            auto fit = fit_bspline(
                tau, dp, config.knot_spacing_s, effective_lambda, reg_deriv_order,
                drift_lambda, degree);
            last_fit = fit;

            // Start the coefficient sums at zero on the first iteration.
            if (accum_cx.size() == 0) {
                accum_cx = Eigen::VectorXd::Zero(fit.n_basis);
                accum_cy = Eigen::VectorXd::Zero(fit.n_basis);
                accum_cz = Eigen::VectorXd::Zero(fit.n_basis);
            }

            // 4. Step-length search.
            const auto ls = line_search(fit, obs_sse, iter);
            if (ls.accepted) {
                accumulated_accel = ls.trial_accel;
                accum_cx += ls.alpha * fit.cx;
                accum_cy += ls.alpha * fit.cy;
                accum_cz += ls.alpha * fit.cz;
            }

            result.iterations_used = iter + 1;
            const double obs_rmse = std::sqrt(obs_sse / tau.size());
            // EAResult::obs_rmse reports the last iteration's value.
            result.obs_rmse = obs_rmse;
            const double trial_rmse =
                ls.accepted ? std::sqrt(ls.trial_sse / tau.size()) : obs_rmse;
            log_iter(iter, tau.size(), fit.n_basis, obs_rmse, trial_rmse,
                     ls.accepted, ls.alpha, effective_lambda);

            // 5. Track the best EA and count iterations without improvement.
            update_best_tracking(ls.accepted, ls.trial_sse);

            // 6. Stop early at the noise level or after STALL_LIMIT stalls.
            if (noise_floor_reached(obs_rmse)) {
                log_noise_floor_stop(obs_rmse);
                break;
            }
            if (stall_count >= detail::STALL_LIMIT) {
                log_stall_stop(iter);
                break;
            }
        }

        // After the loop: pick the EA, record λ and σ², build the history.
        result.empirical_accel = select_final_ea();
        if (last_fit.lambda_used) result.lambda_used = last_fit.lambda_used;
        if (cached_sigma_sq >= 0.0)      result.sigma_sq_estimate = cached_sigma_sq;
        build_collocation_history();
        return std::move(result);
    }
};

template<BSplineObsType ObsType>
[[nodiscard]] inline EAResult estimate_bspline_impl(
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& obs_data,
    const BSplineAccelConfig& config)
{
    return BSplineEstimator<ObsType>(
        epoch0, position_ecef, velocity_ecef, obs_data, config
    ).run();
}

// =========================================================================
// Internal entry points. Public callers use estimate_ea_bspline
// (ea_estimator.hpp), which converts ECEF observations as needed. Start state
// in ECEF (m, m/s). Position data is ECEF [m]; velocity and acceleration data
// must already be in GCRS (m/s, m/s²). Failures are reported in
// EAResult::error_message.
// =========================================================================

[[nodiscard]] inline EAResult
estimate_empirical_acceleration_bspline(
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& position_data,
    const BSplineAccelConfig& config = {})
{
    return estimate_bspline_impl<BSplineObsType::POSITION>(
        epoch0, position_ecef, velocity_ecef, position_data, config);
}

[[nodiscard]] inline EAResult
estimate_empirical_acceleration_bspline_from_velocity(
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& velocity_data_gcrs,
    const BSplineAccelConfig& config = {})
{
    return estimate_bspline_impl<BSplineObsType::VELOCITY>(
        epoch0, position_ecef, velocity_ecef, velocity_data_gcrs, config);
}

[[nodiscard]] inline EAResult
estimate_empirical_acceleration_bspline_from_acceleration(
    const Epoch& epoch0,
    const Vector3d& position_ecef,
    const Vector3d& velocity_ecef,
    const std::vector<std::pair<Epoch, Vector3d>>& acceleration_data_gcrs,
    const BSplineAccelConfig& config = {})
{
    return estimate_bspline_impl<BSplineObsType::ACCELERATION>(
        epoch0, position_ecef, velocity_ecef, acceleration_data_gcrs, config);
}


} // namespace detail
} // namespace calibration
} // namespace ballistic
