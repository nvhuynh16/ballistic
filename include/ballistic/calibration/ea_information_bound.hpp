#pragma once

// =============================================================================
// information_bound — how well could ANY estimator do with this data?
// =============================================================================
//
// Before building or running an estimator, this computes a lower bound on the
// position error (RMS, meters) with which the trajectory can be recovered from
// a given set of sensors when an empirical acceleration (EA — a correction
// acceleration fitted to the data and added to the force model) is estimated
// along with it. Use it to design a sensor suite. Measured examples on the
// library's reentry-glide test geometry (one radar at 160 Hz + position fixes):
// one radar ≈ 10–17 m (Cramér–Rao and posterior bounds); adding a 50 µrad
// camera → ≈ 1.3 m (test_ea_problem_spec). The smoother's measured 12.7 m on that
// geometry sits at the one-radar bound: no estimator tuning can beat an
// information limit.

//
// How it is computed. The bound is a local linearization around the orbit
// propagated with zero EA (real data has no "true" EA to linearize around):
//   θ = the EA values at nodes every knot_spacing_s seconds (linear in between);
//   J = ∂(ECEF position at each observation time)/∂θ, from the state transition
//       matrix (STM: how a change at one time moves the state later);
//   H = Σ Jᵀ W² J, the Fisher information: each observation row weighted by
//       1/σ. Rows: radar range/azimuth/elevation (range rate is left out — the
//       range-rate noise the smoother can actually match is far above the
//       datasheet σ), position fixes, camera angles;

//   posterior Σ = (H + λ·L₁ᵀL₁ + λ_amp·I)⁻¹, where the added terms are a prior
//       favoring a smooth (L₁ = first differences) and small EA (λ=1, λ_amp=1e-2);
//   CRB = the Cramér–Rao bound, the inverse of H restricted to the directions
//       the data can see (eigenvalues ≥ 1e-9 of the largest), with no prior;
//   bound = √(mean over observation times of trace(S Σ Sᵀ)), S = the position
//       rows of J (evaluated at up to ~24 evenly spaced times).
//
// Limitation: accelerometers are NOT included. Their information arrives
// directly, mixed with an unknown bias, not through the trajectory; a bound
// that ignored that bias would be too optimistic. A scenario with only
// accelerometers returns an error.
// =============================================================================

#include "multisensor.hpp"
#include "detail/gn_stm.hpp"
#include "detail/ea_nodes.hpp"
#include "detail/run_propagation.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace ballistic {
namespace calibration {

/// Settings for `information_bound`.
struct EAInformationBoundOptions {
    /// Spacing of the EA nodes [s]. Default 8.0, the spacing behind the example
    /// numbers above. A finer spacing raises the bound (more unknowns, same
    /// data), so compare sensor suites at one spacing.
    double knot_spacing_s = 8.0;
    /// Prior weights: `lambda` penalizes EA changes between neighboring nodes
    /// (smoothness); `lambda_amp` penalizes EA size.
    double lambda = 1.0;
    double lambda_amp = 1e-2;
};

/// Result of `information_bound`.
struct EAInformationBound {
    double posterior_bound_m = 0.0;  ///< RMS position bound [m] with the smoothness prior
                                     ///< (what a regularized estimator can reach)
    double crb_bound_m = 0.0;        ///< RMS position bound [m] without a prior, over the
                                     ///< directions the data can see
    int n_params = 0;                ///< number of unknowns (3 per EA node)
    int n_obs_rows = 0;              ///< number of scalar observations used
    std::string error_message;       ///< non-empty = bound not computed
    /// True when the bound was computed.
    [[nodiscard]] bool ok() const noexcept { return error_message.empty(); }
};

/// Compute the information bound for the sensors of `scenario` selected by
/// `mask` (a sensor missing from the mask counts as on), propagating with
/// force model `fm_cfg`. Failures (no radar/fix/camera samples, a failed
/// propagation) are reported in `error_message`, not thrown; errors from the
/// propagator itself (e.g. missing data files) propagate.
[[nodiscard]] inline EAInformationBound information_bound(
    const MultiSensorScenario& scenario,
    const SensorMask& mask,
    const TrajectoryConfig& fm_cfg,
    const EAInformationBoundOptions& opts = {})
{
    EAInformationBound out;
    const Epoch& e0 = scenario.epoch0;
    auto active = [](const auto& maskv, size_t s) {
        return s >= maskv.size() || maskv[s];
    };

    // ---- Distinct observation times (radar / fix / camera only). ----
    std::vector<double> times;
    for (size_t s = 0; s < scenario.radars.size(); ++s)
        if (active(mask.radars, s))
            for (const auto& [t, v] : scenario.radars[s].samples)
                times.push_back(t - e0);
    for (size_t s = 0; s < scenario.fixes.size(); ++s)
        if (active(mask.fixes, s))
            for (const auto& [t, v] : scenario.fixes[s].samples)
                times.push_back(t - e0);
    for (size_t s = 0; s < scenario.optics.size(); ++s)
        if (active(mask.optics, s))
            for (const auto& [t, v] : scenario.optics[s].samples)
                times.push_back(t - e0);
    if (times.empty()) {
        out.error_message =
            "information_bound: no radar/fix/optics observations (accels "
            "are not supported — bias ambiguity would make the bound "
            "optimistic)";
        return out;
    }
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end(),
                            [](double a, double b) {
                                return std::abs(a - b) < 1e-9;
                            }),
                times.end());
    const size_t nu = times.size();

    // ---- EA nodes and the STM-based Jacobian around the zero-EA orbit. ----
    std::vector<double> t_nodes =
        detail::build_ea_keys(scenario.arc_duration_s, opts.knot_spacing_s);
    const int nn = static_cast<int>(t_nodes.size());
    const int np = 3 * nn;
    out.n_params = np;

    auto ea0 = detail::ea_from_theta(Eigen::VectorXd::Zero(np), t_nodes);
    std::vector<std::pair<Epoch, Vector3d>> position_data;
    position_data.reserve(nu);
    for (double t : times) position_data.emplace_back(e0 + t, Vector3d::Zero());
    std::vector<size_t> sample_indices(nu);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    const double prop_s = scenario.arc_duration_s + 1.0;
    auto stm = detail::build_stm_jacobian_and_pnom(
        e0, scenario.pos0_ecef, scenario.vel0_ecef, fm_cfg, ea0,
        sample_indices, position_data, t_nodes, nn, np,
        forces::EmpiricalAccelFrame::GCRS, prop_s,
        /*min_altitude_m=*/-2000.0, /*dt_var_s=*/opts.knot_spacing_s,
        /*fd_dx=*/1.0, /*fd_dv=*/1e-3, /*verbose=*/false);
    if (stm.J.size() == 0) {
        out.error_message = "information_bound: STM propagation failed";
        return out;
    }

    // Full ECEF states of the zero-EA orbit at each observation time, needed
    // for the sensors' own Jacobians.
    auto [traj, ti_ecef] = detail::run_propagation_arc(
        e0, scenario.pos0_ecef, scenario.vel0_ecef, fm_cfg, ea0, prop_s,
        /*min_alt=*/-2000.0);
    if (!traj || !ti_ecef) {
        out.error_message = "information_bound: nominal propagation failed";
        return out;
    }
    auto u_of = [&](const Epoch& t) -> long {
        const double ts = t - e0;
        auto it = std::lower_bound(times.begin(), times.end(), ts - 1e-9);
        if (it == times.end() || std::abs(*it - ts) > 1e-9) return -1;
        return it - times.begin();
    };
    std::vector<StateVector> st_u(nu);
    std::vector<char> st_ok(nu, 0);
    for (size_t u = 0; u < nu; ++u) {
        auto st = ti_ecef->get_state(e0 + times[u]);
        if (st) { st_u[u] = StateVector{st->position, st->velocity}; st_ok[u] = 1; }
    }

    // ---- Accumulate H = Σ Jᵀ W² J (only the position part of each sensor
    //      Jacobian; the EA moves velocity too, but rate rows are excluded). ----
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(np, np);
    long rows = 0;
    for (size_t s = 0; s < scenario.radars.size(); ++s) {
        if (!active(mask.radars, s)) continue;
        const auto& radar = scenario.radars[s];
        for (size_t i = 0; i < radar.samples.size(); ++i) {
            const long u = u_of(radar.samples[i].first);
            if (u < 0 || !st_ok[static_cast<size_t>(u)]) continue;
            Eigen::Vector4d r4;
            Eigen::Matrix<double, 4, 6> J6;
            RadarObs tmp = radar;
            tmp.samples.assign(1, {radar.samples[i].first,
                                   tmp.predict(st_u[static_cast<size_t>(u)])});
            tmp.residual_and_jacobian(0, st_u[static_cast<size_t>(u)], r4, J6);
            Eigen::MatrixXd dh =
                J6.block<3, 3>(0, 0) * stm.J.middleRows(3 * u, 3);
            dh.row(0) /= radar.sigma_range;
            dh.row(1) /= radar.sigma_angle;
            dh.row(2) /= radar.sigma_angle;
            H.noalias() += dh.transpose() * dh;
            rows += 3;
        }
    }
    for (size_t s = 0; s < scenario.fixes.size(); ++s) {
        if (!active(mask.fixes, s)) continue;
        const auto& fix = scenario.fixes[s];
        for (size_t i = 0; i < fix.samples.size(); ++i) {
            const long u = u_of(fix.samples[i].first);
            if (u < 0 || !st_ok[static_cast<size_t>(u)]) continue;
            Eigen::MatrixXd dh = stm.J.middleRows(3 * u, 3) / fix.sigma_pos;
            H.noalias() += dh.transpose() * dh;
            rows += 3;
        }
    }
    for (size_t s = 0; s < scenario.optics.size(); ++s) {
        if (!active(mask.optics, s)) continue;
        const auto& cam = scenario.optics[s];
        for (size_t i = 0; i < cam.samples.size(); ++i) {
            const long u = u_of(cam.samples[i].first);
            if (u < 0 || !st_ok[static_cast<size_t>(u)]) continue;
            Eigen::Vector2d r2;
            Eigen::Matrix<double, 2, 6> J6;
            OpticsObs tmp = cam;
            tmp.samples.assign(1, {cam.samples[i].first,
                                   tmp.predict(st_u[static_cast<size_t>(u)])});
            tmp.residual_and_jacobian(0, st_u[static_cast<size_t>(u)], r2, J6);
            Eigen::MatrixXd dh =
                (J6.block<2, 3>(0, 0) * stm.J.middleRows(3 * u, 3)) /
                cam.sigma_angle;
            H.noalias() += dh.transpose() * dh;
            rows += 2;
        }
    }
    out.n_obs_rows = static_cast<int>(rows);
    if (rows == 0) {
        out.error_message = "information_bound: no usable observation rows";
        return out;
    }

    // ---- Posterior covariance with the prior, and the prior-free CRB on the
    //      observable directions. ----
    Eigen::MatrixXd P_prior =
        opts.lambda *
        detail::build_first_difference_LtL(nn, opts.knot_spacing_s);
    P_prior.diagonal().array() += opts.lambda_amp;
    Eigen::MatrixXd Sigma =
        (H + P_prior).ldlt().solve(Eigen::MatrixXd::Identity(np, np));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    const auto& ev = es.eigenvalues();
    const double thr = 1e-9 * ev(np - 1);
    Eigen::MatrixXd Sobs = Eigen::MatrixXd::Zero(np, np);
    for (int i = 0; i < np; ++i)
        if (ev(i) > thr)
            Sobs += es.eigenvectors().col(i) *
                    es.eigenvectors().col(i).transpose() / ev(i);

    const size_t stride = std::max<size_t>(1, nu / 24);
    double trp = 0.0, tro = 0.0;
    long nt = 0;
    for (size_t u = 0; u + 1 < nu; u += stride) {
        Eigen::MatrixXd S = stm.J.middleRows(3 * u, 3);
        trp += (S * Sigma * S.transpose()).trace();
        tro += (S * Sobs * S.transpose()).trace();
        nt++;
    }
    out.posterior_bound_m = std::sqrt(trp / nt);
    out.crb_bound_m = std::sqrt(tro / nt);
    return out;
}

}  // namespace calibration
}  // namespace ballistic
