#pragma once

// =============================================================================
// estimate_ea_smoother — fit an empirical acceleration to several sensors
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration fitted to
// observations and added to the force model. This is the recommended EA
// estimator for data from several sensors (radar, cameras, position fixes,
// accelerometers), for radar alone and for noisy positions
// (docs/multi_sensor_estimator.md).
//
// Method ("reduced-dynamic" orbit determination, also called dynamic model
// compensation; Wu, Yunck & Thornton 1991; Myers & Tapley): the filter state is
// z = [r, v, a] (9 numbers, GCRS — the Earth-centered inertial frame): position,
// velocity and the EA itself. The EA is modeled as a first-order Gauss-Markov
// process — a random signal of typical size σ_a that stays correlated for
// about τ seconds and then wanders. An extended Kalman filter (EKF: a Kalman
// filter that linearizes the dynamics at each step) runs forward over all
// observations in time order, then a Rauch–Tung–Striebel (RTS) smoother pass
// runs backward so every estimate uses the data on both sides of it. As shown
// by Bell 1994, this gives the same answer as a batch least-squares fit with
// the equivalent prior, but in time proportional to the number of
// observations, with no iteration schedule, damping or multiple starts to tune,
// and with an uncertainty (posterior covariance) for every point.
//
// Settings are minimal. σ_a and τ are chosen automatically from a small grid
// by the normalized innovation squared (NIS): the average squared difference
// between each observation and its prediction, divided by the variance the
// filter expected; it is ≈ 1 when the noise model matches the data. The
// weight given to radar range rate is MEASURED from the data
// (σ_eff² = sensor noise² + model and numerical error), never taken from the
// datasheet. Each default is the best measured on the test scenarios.
//
// Measured on the single-radar reentry-glide test scenario (radar at 160 Hz
// plus a position fix at impact): 12.7 m position error after re-propagation,
// better than the batch fits used before and much faster; the information
// limit for that geometry is ≈ 9–17 m (test_ea_smoother). For clean data or
// velocity data alone, estimate_ea_bspline (with GN) is preferred.
//
// Sensors: radars, position fixes, cameras, accelerometers (whose constant
// bias is estimated too: the state grows to [r, v, a, b], 12 numbers).
// =============================================================================

#include "multisensor.hpp"
#include "detail/gm_rts.hpp"
#include "ea_result_core.hpp"
#include "../coordinates/earth_rotation.hpp"
#include "../coordinates/iau_transforms.hpp"
#include "../forces/empirical_acceleration.hpp"
#include "../forces/force_model.hpp"
#include "../trajectory.hpp"
#include "../util/frame_transform.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ballistic {
namespace calibration {

/// Settings for estimate_ea_smoother. Leave the optional fields empty to
/// have them chosen automatically from the data.
struct EASmootherConfig {
    /// Expected EA size σ_a [m/s²] and correlation time τ [s] of the
    /// Gauss-Markov model. Empty (default) = chosen by NIS (closest to 1) over
    /// σ_a ∈ {1, 3, 10} × τ ∈ {10, 35, 100}. Set both to skip the grid (one
    /// fit instead of nine). The grid suits
    /// reentry glides; for an ordinary orbit, values near 1e-6 m/s² and 600 s
    /// fit (see regime_aware_prior and the result's warnings).
    std::optional<double> sigma_a;
    std::optional<double> tau_s;

    /// How radar range-rate samples are weighted. The weight must reflect how
    /// well the model can actually match range rate, not the datasheet noise:
    ///   Auto (default) — fit once without range rate, measure the range-rate
    ///     residual RMS of that fit, and use σ_eff = max(that, sensor σ).
    ///     Measured: σ_eff came out well above the datasheet σ on the glide
    ///     test, and weighting by the datasheet σ instead cost accuracy.
    ///   Off — ignore range rate.
    ///   Manual — use sigma_rate_manual [m/s] as σ_eff.
    enum class RateBudget { Auto, Off, Manual };
    RateBudget rate_budget = RateBudget::Auto;
    double sigma_rate_manual = 0.0;

    /// Largest integration step [s] between observations (fixed-step
    /// 4th-order Runge-Kutta).
    double max_dt_s = 0.05;
    /// Recompute the force-model Jacobian (∂acceleration/∂position and
    /// ∂acceleration/∂velocity) every K steps; it changes slowly.
    int jac_refresh = 16;
    /// Time spacing [s] of the returned tabulated EA (GCRS).
    double output_dt_s = 0.5;
    /// Number of filter/smoother passes. 2 = run again with the first pass's
    /// EA added to the dynamics, so the filter estimates only what is left
    /// (an iterated extended Kalman smoother; per Bell 1994 this is one more
    /// Gauss-Newton iteration of the same fit).
    int passes = 1;
    /// Estimate a time-tag (clock) offset [s] as an extra state, for the
    /// sensors whose SensorClock is Shared or Own. A clock error the model
    /// does not know about is the largest measured risk with real data (in
    /// the radar test a 5 ms offset gave 316 m); with this on, the offset is
    /// absorbed and most of the accuracy comes back (52 m), although the
    /// estimated offset itself is not calibrated (see gm_rts.hpp). Harmless
    /// when the clocks are right.
    bool estimate_clock_bias = true;
    double clock_bias_prior_s = 0.02;   // expected size of the clock offset [s]
    /// Maneuver handling (off by default): after any observation whose
    /// NIS/dof exceeds `maneuver_nis_gate`, multiply the EA's process noise
    /// (how fast the EA is allowed to change) by `maneuver_q_boost`, so a
    /// sudden EA change (a thruster firing) is followed instead of smeared out.
    bool maneuver_adaptive_q = false;
    double maneuver_nis_gate = 9.0;
    double maneuver_q_boost = 25.0;
    /// Choose the σ_a candidates from the data instead of the fixed grid.
    /// The fixed grid {1, 3, 10} m/s² suits reentry glides and is ~7 orders of
    /// magnitude too large for an ordinary orbit (on a real GRACE-FO arc the
    /// EA made things worse, 61 m vs 0.157 m with no EA, while NIS looked
    /// fine). With this on, a first fit with the EA held at zero (plain orbit
    /// determination) measures the position residual Δ the dynamics cannot
    /// explain (the largest RMS among the radar-range and position-fix
    /// channels) over the arc length T, and σ_a candidates are taken
    /// around a ≈ 2Δ/T². Ignored when sigma_a is set.
    ///
    /// Measured better on the orbital arc (60.95 → 0.154 m) and no worse on
    /// the glide tests (single radar, + camera, + accelerometer; test_ea_smoother).
    ///
    /// Not the default for one measured reason: the tighter prior makes the
    /// reported uncertainty too small inside data gaps (on a test with a 40 s
    /// outage, the true error in mid-gap exceeded the acceptance limit of 2.5
    /// reported σ, against 1.6 σ with the default grid). Making it the default
    /// needs extra output points inside gaps first. Costs one extra fit.

    bool regime_aware_prior = false;

    /// Extension point (C++ only): user-defined sensors, processed exactly like
    /// the built-in ones. See detail::CustomSensor for what to provide and
    /// docs/multi_sensor_estimator.md for a worked example and its caveats.
    std::vector<detail::CustomSensor> custom_sensors;
    bool verbose = false;   // print progress (grid values, NIS) to stdout
};

/// Result of estimate_ea_smoother: the EA (`empirical_accel`, GCRS, from
/// ResultCore) and quality figures that need no true answer to compute.
/// `nis` ≈ 1 means the declared noise matches the data; far from 1 means the
/// noise levels or the EA prior are wrong.
struct EASmootherResult : ResultCore {
    double nis = 0.0;               ///< normalized innovation squared per degree of freedom
    Vector3d recovered_bias = Vector3d::Zero();  ///< estimated accelerometer bias (ECEF,
                                                 ///< m/s²), if accelerometers were used
    double sigma_a_used = 0.0;      ///< σ_a used [m/s²]
    double tau_used = 0.0;          ///< τ used [s]
    double sigma_rate_eff = 0.0;    ///< range-rate σ used [m/s] (0 = range rate ignored)
    double clock_bias_s = 0.0;      ///< clock offset [s]: the "shared" one, else the
                                    ///< first in clock_biases
    /// Estimated clock offset [s] per clock ("shared", "radar0", "optics1",
    /// …). Caveat: the scale and sign of these values are not yet calibrated
    /// (see gm_rts.hpp).
    std::vector<std::pair<std::string, double>> clock_biases;
    /// Residual RMS per channel at the final solution ("radar0_range",
    /// "radar0_rate", "fix0_pos", …) in each channel's units — the accuracy
    /// actually reached, to compare with the declared σ. A value far above
    /// the declared σ means the channel could not be fitted that well: wrong
    /// noise level, an unmodeled bias or clock error, or a model limit.
    /// Report only: the weights are not changed.
    std::vector<std::pair<std::string, double>> sigma_honored;
    /// 1σ uncertainty of the EA [m/s², GCRS axes] at each point of the
    /// returned table (same order as its keys), from the smoother's covariance.
    std::vector<Vector3d> ea_sigma;
    /// Acceleration scale 2Δ/T² [m/s²] from the zero-EA first fit (0 when
    /// regime_aware_prior is off): what the dynamics alone could not explain.
    double prior_anchor = 0.0;
    /// Plain-language warnings (nothing is changed because of them):
    ///   - the largest gap between observations is longer than τ. Inside such
    ///     a gap the EA is unobserved and errors grow unseen (measured: a 40 s
    ///     outage took the error from 12.7 to 1623 m while NIS stayed at 0.77),
    ///     so this warning is the only sign;
    ///   - the automatic grid chose an edge value (see audit_grid_edge).
    std::vector<std::string> warnings;
};

// The filter/smoother itself (state layout, sensor updates, process model,
// forward filter, backward pass, run_gm_rts) is in detail/gm_rts.hpp.

namespace detail {

// Filter options for one run: σ_a, τ and the range-rate σ as given, every
// other field copied from cfg. Used by the range-rate prepass, the grid loop
// and the second pass.
[[nodiscard]] inline RunOptions make_run_options(
    const EASmootherConfig& cfg, double sigma_a, double tau,
    double sigma_rate_eff) {
    RunOptions ro;
    ro.custom = &cfg.custom_sensors;
    ro.sigma_a = sigma_a;
    ro.tau = tau;
    ro.sigma_rate_eff = sigma_rate_eff;
    ro.max_dt = cfg.max_dt_s;
    ro.jac_refresh = cfg.jac_refresh;
    ro.est_clock = cfg.estimate_clock_bias;
    ro.clock_prior_s = cfg.clock_bias_prior_s;
    ro.adaptive_q = cfg.maneuver_adaptive_q;
    ro.nis_gate = cfg.maneuver_nis_gate;
    ro.q_boost = cfg.maneuver_q_boost;
    return ro;
}

// Add a warning when the automatically chosen σ_a or τ is at the edge of its
// grid: the best value may lie outside the grid, which is tuned for reentry
// glides. Measured reason at the call site.
inline void audit_grid_edge(
    EASmootherResult& result, const EASmootherConfig& cfg,
    const SmootherRun& best, double best_sa, double best_tau,
    size_t best_i_sa, size_t best_i_tau,
    const std::vector<double>& sas, const std::vector<double>& taus) {
    double a_ms = 0.0;
    for (const auto& a : best.a_smooth) a_ms += a.squaredNorm();
    const double ea_rms =
        best.a_smooth.empty() ? 0.0 : std::sqrt(a_ms / best.a_smooth.size());
    const double looseness =
        ea_rms > 0.0 ? best_sa / ea_rms : std::numeric_limits<double>::infinity();
    if (cfg.verbose)
        std::cout << "  [ea-smoother] sigma_a=" << best_sa
                  << " EA_rms=" << ea_rms << " looseness=" << looseness << "\n";
    const bool sa_auto = !cfg.sigma_a && sas.size() > 1;
    const bool tau_auto = !cfg.tau_s && taus.size() > 1;
    const bool sa_edge =
        sa_auto && (best_i_sa == 0 || best_i_sa == sas.size() - 1);
    const bool tau_edge =
        tau_auto && (best_i_tau == 0 || best_i_tau == taus.size() - 1);
    if (!(sa_edge || tau_edge)) return;
    std::string w = "NIS auto-grid selected a BOUNDARY value (";
    if (sa_edge)
        w += "sigma_a=" + std::to_string(best_sa) + " at the " +
             (best_i_sa == 0 ? "lower" : "upper") + " edge";
    if (sa_edge && tau_edge) w += ", ";
    if (tau_edge)
        w += "tau=" + std::to_string(best_tau) + " s at the " +
             (best_i_tau == 0 ? "lower" : "upper") + " edge";
    w += ") — the best prior may lie OUTSIDE the grid, which is "
         "tuned for the glide regime. CONFIRM YOUR REGIME MATCHES: "
         "on an orbital arc this condition accompanied a returned EA "
         "that re-propagated far worse than propagating with no EA at "
         "all, while NIS stayed inside its healthy band; "
         "pinning cfg.sigma_a / cfg.tau_s to the regime (orbital is "
         "roughly 1e-6 m/s^2 and 600 s) fixed it. NOT an error "
         "signal on its own: boundary selection also occurs on "
         "well-behaved glide-class fits, and measurement shows that "
         "neither NIS nor the prior/EA amplitude ratio distinguishes "
         "the two cases — so this is a prompt to check the regime, "
         "not evidence of a bad fit.";
    result.warnings.push_back(std::move(w));
}

// Add a warning when the largest gap between observations exceeds τ: the EA
// there is set by the prior alone and errors there do not show in NIS.
inline void audit_observation_gaps(
    EASmootherResult& result, const MultiSensorScenario& scenario,
    const SensorMask& mask, const EASmootherConfig& cfg, double best_tau) {
    auto obs_t = build_timeline(scenario, mask, &cfg.custom_sensors);
    double max_gap = 0.0, gap_t0 = 0.0;
    for (size_t i = 1; i < obs_t.size(); ++i) {
        const double dt = obs_t[i].t - obs_t[i - 1].t;
        if (dt > max_gap) { max_gap = dt; gap_t0 = obs_t[i - 1].t; }
    }
    if (max_gap > best_tau)
        result.warnings.push_back(
            "observation gap [" + std::to_string(gap_t0) + ", " +
            std::to_string(gap_t0 + max_gap) + "] s exceeds the GM "
            "correlation time τ=" + std::to_string(best_tau) +
            " s — the EA there is prior-dominated and unobserved "
            "acceleration is invisible to NIS (measured: a 40 s outage "
            "costs 128× reprop, silently); treat ea_sigma there as the "
            "only honest statement");
}

}  // namespace detail

/// Estimate an EA from the sensors of `scenario` selected by `mask` (a sensor
/// missing from the mask counts as on), with force model `fm_cfg`. Returns the
/// EA in GCRS, tabulated every cfg.output_dt_s seconds and linear in between,
/// with its 1σ (`ea_sigma`), `nis` and the other quality figures. A filter
/// failure is reported in `error_message`. Throws what the Trajectory
/// constructor throws for an implausible start state, and propagates errors
/// from the force model (e.g. missing data files).
[[nodiscard]] inline EASmootherResult estimate_ea_smoother(
    const MultiSensorScenario& scenario,
    const SensorMask& mask,
    const TrajectoryConfig& fm_cfg,
    const EASmootherConfig& cfg = {})
{
    EASmootherResult result;

    // A Trajectory built only to own the ForceModel the filter evaluates.
    Trajectory fm_holder(scenario.epoch0, scenario.pos0_ecef,
                         scenario.vel0_ecef, fm_cfg);
    fm_holder.mark_internal();
    const forces::ForceModel& fm = fm_holder.force_model();
    const coords::ITRFRealization itrf = fm_cfg.itrf;

    // Range-rate weight: measured from the data unless set by hand.
    double sigma_rate_eff = 0.0;
    const double sa0 = cfg.sigma_a.value_or(3.0);
    const double tau0 = cfg.tau_s.value_or(35.0);
    bool any_radar = false;
    for (size_t s = 0; s < scenario.radars.size(); ++s)
        if (s >= mask.radars.size() || mask.radars[s]) any_radar = true;
    if (any_radar && cfg.rate_budget == EASmootherConfig::RateBudget::Manual) {
        sigma_rate_eff = cfg.sigma_rate_manual;
    } else if (any_radar &&
               cfg.rate_budget == EASmootherConfig::RateBudget::Auto) {
        // Fit once with range rate ignored (σ = 0 means off).
        auto pre_opts = detail::make_run_options(cfg, sa0, tau0, 0.0);
        auto pre = detail::run_gm_rts(scenario, mask, fm, itrf, pre_opts);
        if (!pre.ok) {
            result.error_message = pre.error;
            return result;
        }
        double budget = detail::smoother_rate_residual_rms(scenario, mask, itrf, pre);
        double sensor_sigma = 0.0;
        for (size_t s = 0; s < scenario.radars.size(); ++s)
            if (s >= mask.radars.size() || mask.radars[s])
                sensor_sigma = std::max(sensor_sigma, scenario.radars[s].sigma_rate);
        sigma_rate_eff = std::max(budget, sensor_sigma);
        if (cfg.verbose)
            std::cout << "  [ea-smoother] measured rate budget σ_eff = "
                      << sigma_rate_eff << " m/s (sensor σ = " << sensor_sigma
                      << ")\n";
    }

    // Data-derived σ_a scale: one fit with σ_a ≈ 0, so the EA cannot move and
    // this is plain orbit determination. The residual left is what the
    // dynamics cannot explain; a constant acceleration a displaces a position
    // by a·T²/2 over T, so a ≈ 2Δ/T². No tuned constant enters.
    double anchor = 0.0;
    if (cfg.regime_aware_prior && !cfg.sigma_a) {
        detail::RunOptions ao;
        ao.custom = &cfg.custom_sensors;
        ao.sigma_a = 1e-12;                 // frozen EA
        ao.tau = std::max(1.0, scenario.arc_duration_s);
        ao.sigma_rate_eff = sigma_rate_eff;
        ao.max_dt = cfg.max_dt_s;
        ao.jac_refresh = cfg.jac_refresh;
        ao.est_clock = cfg.estimate_clock_bias;
        ao.clock_prior_s = cfg.clock_bias_prior_s;
        auto arun = detail::run_gm_rts(scenario, mask, fm, itrf, ao);
        if (arun.ok) {
            // Largest position-like residual of that fit (radar range and
            // position fixes, both in meters).
            auto honored = detail::measure_honored_sigmas(scenario, mask,
                                                          itrf, arun);
            double delta = 0.0;
            for (const auto& [ch, v] : honored)
                if (ch.size() > 4 &&
                    (ch.substr(ch.size() - 6) == "_range" ||
                     ch.substr(ch.size() - 4) == "_pos"))
                    delta = std::max(delta, v);
            const double T = std::max(1.0, scenario.arc_duration_s);
            if (delta > 0.0) anchor = 2.0 * delta / (T * T);
            if (cfg.verbose)
                std::cout << "  [ea-smoother] regime anchor: delta=" << delta
                          << " m over T=" << T << " s -> a_scale=" << anchor
                          << " m/s^2\n";
        }
    }
    result.prior_anchor = anchor;

    // σ_a and τ: as set, or chosen by NIS over the grid. The candidate lists
    // are kept so the edge check below can tell an interior choice from an
    // edge one.
    std::vector<std::pair<double, double>> grid;
    std::vector<double> sas, taus;
    if (cfg.sigma_a && cfg.tau_s) {
        grid = {{*cfg.sigma_a, *cfg.tau_s}};
    } else {
        sas = cfg.sigma_a
                  ? std::vector<double>{*cfg.sigma_a}
                  : (anchor > 0.0
                         ? std::vector<double>{0.3 * anchor, 1.0 * anchor,
                                               3.0 * anchor}
                         : std::vector<double>{1.0, 3.0, 10.0});
        taus = cfg.tau_s ? std::vector<double>{*cfg.tau_s}
                         : std::vector<double>{10.0, 35.0, 100.0};
        for (double sa : sas)
            for (double tu : taus) grid.emplace_back(sa, tu);
    }

    detail::SmootherRun best;
    double best_dist = std::numeric_limits<double>::infinity();
    double best_sa = sa0, best_tau = tau0;
    size_t best_i_sa = 0, best_i_tau = 0, i_grid = 0;
    for (auto [sa, tu] : grid) {
        auto ro = detail::make_run_options(cfg, sa, tu, sigma_rate_eff);
        auto run = detail::run_gm_rts(scenario, mask, fm, itrf, ro);
        if (!run.ok) {
            result.error_message = run.error;
            return result;
        }
        double dist = std::abs(std::log(std::max(run.nis, 1e-12)));
        if (cfg.verbose)
            std::cout << "  [ea-smoother] σ_a=" << sa << " τ=" << tu
                      << " → NIS/dof " << run.nis << "\n";
        if (dist < best_dist) {
            best_dist = dist;
            best = std::move(run);
            best_sa = sa;
            best_tau = tu;
            if (!taus.empty()) {
                best_i_sa = i_grid / taus.size();
                best_i_tau = i_grid % taus.size();
            }
        }
        ++i_grid;
    }

    // Edge check (warning only). Measured reason: on an orbital arc the grid,
    // tuned for reentry glides, chose an edge value and the EA followed radar
    // angle noise — 61 m error after re-propagation where propagating with no
    // EA gives 0.157 m — while NIS looked healthy. An edge choice means

    // the best value may lie outside the grid, and nothing else says so.
    if (grid.size() > 1)
        detail::audit_grid_edge(result, cfg, best, best_sa, best_tau,
                                best_i_sa, best_i_tau, sas, taus);

    // Second pass (cfg.passes >= 2): add the first pass's EA (every 8th point)
    // to the dynamics and estimate only the remainder.
    std::shared_ptr<forces::EmpiricalAcceleration> a_ref_ea;
    if (cfg.passes >= 2) {
        std::vector<forces::EmpiricalAccelEntry> ref_entries;
        ref_entries.reserve(best.t.size() / 8 + 2);
        for (size_t i = 0; i < best.t.size(); i += 8)
            ref_entries.emplace_back(best.t[i], best.a_smooth[i]);
        a_ref_ea = std::make_shared<forces::EmpiricalAcceleration>(
            forces::EmpiricalAcceleration::from_entries(
                std::move(ref_entries), forces::EmpiricalAccelFrame::GCRS,
                forces::EAInterp::Linear));
        auto ro2 = detail::make_run_options(cfg, best_sa, best_tau, sigma_rate_eff);
        ro2.a_ref = a_ref_ea.get();
        auto run2 = detail::run_gm_rts(scenario, mask, fm, itrf, ro2);
        if (run2.ok) {
            // Total EA = first-pass EA + remainder.
            for (size_t i = 0; i < run2.t.size(); ++i)
                run2.a_smooth[i] += a_ref_ea->accel_at(run2.t[i]);
            best = std::move(run2);
        }
    }

    // Resample the EA and its 1σ onto the output grid (linear interpolation).
    std::vector<forces::EmpiricalAccelEntry> entries;
    result.ea_sigma.clear();
    size_t j = 0;
    for (double t = best.t.front(); t <= best.t.back() + 1e-9;
         t += cfg.output_dt_s) {
        while (j + 1 < best.t.size() && best.t[j + 1] <= t) ++j;
        Vector3d a = best.a_smooth[j];
        Vector3d sig = best.a_sigma[j];
        if (j + 1 < best.t.size() && best.t[j + 1] > best.t[j]) {
            double w = std::clamp((t - best.t[j]) / (best.t[j + 1] - best.t[j]),
                                  0.0, 1.0);
            a = (1.0 - w) * best.a_smooth[j] + w * best.a_smooth[j + 1];
            sig = (1.0 - w) * best.a_sigma[j] + w * best.a_sigma[j + 1];
        }
        entries.emplace_back(t, a);
        result.ea_sigma.push_back(sig);
    }
    result.empirical_accel = std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(
            std::move(entries), forces::EmpiricalAccelFrame::GCRS,
            forces::EAInterp::Linear));
    // Gap check: inside a gap longer than τ the EA comes from the prior alone,
    // an unobserved acceleration error grows the position error with the
    // square of time, and NIS cannot see it.
    detail::audit_observation_gaps(result, scenario, mask, cfg, best_tau);

    result.nis = best.nis;
    result.recovered_bias = best.bias;
    result.clock_bias_s = best.clock_bias;
    result.clock_biases = best.clocks;
    result.sigma_honored =
        detail::measure_honored_sigmas(scenario, mask, itrf, best);
    result.sigma_a_used = best_sa;
    result.tau_used = best_tau;
    result.sigma_rate_eff = sigma_rate_eff;
    result.iterations_used = std::max(1, cfg.passes);
    return result;
}

}  // namespace calibration
}  // namespace ballistic
