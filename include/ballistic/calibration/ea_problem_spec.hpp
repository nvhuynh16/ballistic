#pragma once

// =============================================================================
// EAProblemSpec / estimate_ea — describe the problem, let the library choose
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration fitted to
// observations and added to the force model. With `estimate_ea` you describe
// the data and its errors ("these are the measurements over time, this is how
// noisy they are") and the library picks the estimator. The spec does not
// expose every setting of the two estimators; if you know what you want, pass
// an override config.
//
// Which estimator is used (docs/empirical_acceleration_estimation.md):
//   position / velocity / acceleration samples → estimate_ea_bspline, with
//                                                Gauss-Newton (GN) refinement on
//   MultiSensorScenario                        → estimate_ea_smoother
// The error model sits on the data: typed samples take `noise_sigma` here;
// scenario sensors carry their own noise levels, known biases, per-sample noise
// scaling and clock settings. The declarations are checked, not trusted: the
// result reports how well the fit actually matched each declared noise level
// and flags inconsistencies in `audit_warnings`.
// =============================================================================

#include "ea_estimator.hpp"
#include "ea_information_bound.hpp"
#include "ea_method_advisor.hpp"
#include "ea_smoother.hpp"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ballistic {
namespace calibration {

/// Input to `estimate_ea`: the observations, their noise and the force model.
struct EAProblemSpec {
    /// Start time and state, ECEF (Earth-fixed; m, m/s). Ignored for a
    /// MultiSensorScenario, which carries its own start.
    Epoch epoch0;
    Vector3d pos0_ecef = Vector3d::Zero();
    Vector3d vel0_ecef = Vector3d::Zero();

    /// The observations. The kind of data is the type, so different kinds
    /// cannot be mixed by accident.
    std::variant<PositionSamples, VelocitySamples, AccelSamples,
                 MultiSensorScenario> observations;

    /// Force model used in the fit (gravity degree, Sun/Moon, drag, ...).
    TrajectoryConfig force_model;

    /// Observation noise standard deviation σ for typed samples, in the data's
    /// units (m, m/s or m/s²). Required (> 0) for typed samples unless a
    /// `bspline` override is given — there is no "unknown noise" mode; give
    /// your best estimate and read the audit. Ignored for scenarios (each
    /// sensor carries its own).
    double noise_sigma = 0.0;

    /// Optional outlier model for typed samples: a fraction p of samples have
    /// noise k·σ instead of σ, i.e. noise = (1−p)·N(0,σ²) + p·N(0,(kσ)²).
    /// outlier_prob = p > 0 removes likely outliers with reject_heavy_tails()
    /// before fitting (the fit itself is unchanged); outlier_scale = k. Same
    /// meaning as on AccelObs. Not applied to velocity samples (see
    /// estimate_ea). Ignored for scenarios (declare it on the sensors).
    double outlier_prob = 0.0;
    double outlier_scale = 10.0;

    /// Optional estimator settings, used as given when present; otherwise each
    /// estimator's defaults (chosen by measurement) apply.
    std::optional<EABSplineConfig> bspline;
    std::optional<EASmootherConfig> smoother;
};

/// Result of `estimate_ea`: the common result fields (`ResultCore`), which
/// estimator ran and why, and the check of the declared noise model.
struct EASpecResult : ResultCore {
    EAMethod method_used = EAMethod::Smoother;   // which estimator ran
    std::string rationale;                       // why it was chosen
    /// Normalized innovation squared per degree of freedom (NIS/dof) of the
    /// smoother: the average squared prediction error divided by the error the
    /// declared noise predicts; ~1 when the declared noise is right. 0 when not
    /// available (B-spline path).
    /// `sigma_honored`: per channel, the residual RMS the fit actually reached
    /// (the B-spline path reports one "obs" entry when noise_sigma is set).
    double nis = 0.0;
    std::vector<std::pair<std::string, double>> sigma_honored;
    /// Plain-language findings of the check (empty = nothing found): NIS out of
    /// range, residuals far above the declared noise, and so on. Warnings, not
    /// errors — the estimate is still returned.
    std::vector<std::string> audit_warnings;
    /// The full result of whichever estimator ran (the other stays empty).
    std::optional<EAResult> bspline_detail;
    std::optional<EASmootherResult> smoother_detail;
};

namespace detail {

// Copy the smoother's consistency figures into `out` and add warnings when NIS
// is outside [0.3, 3] or a range-rate channel's residual exceeds 3× the
// range-rate noise level the smoother found it could match.
inline void audit_smoother(const EASmootherResult& r, EASpecResult& out) {
    out.nis = r.nis;
    out.sigma_honored = r.sigma_honored;
    for (const auto& w : r.warnings) out.audit_warnings.push_back(w);
    if (r.nis > 0.0 && (r.nis < 0.3 || r.nis > 3.0))
        out.audit_warnings.push_back(
            "NIS/dof = " + std::to_string(r.nis) +
            " outside [0.3, 3] — declared noise model inconsistent with the "
            "innovations (mis-declared σ, un-modeled bias/timing, or "
            "model-class limits)");
    if (r.sigma_rate_eff > 0.0)
        for (const auto& [ch, v] : r.sigma_honored)
            if (ch.size() > 5 && ch.substr(ch.size() - 5) == "_rate" &&
                v > 3.0 * r.sigma_rate_eff)
                out.audit_warnings.push_back(
                    "honored " + ch + " RMS " + std::to_string(v) +
                    " m/s exceeds 3x the honest budget " +
                    std::to_string(r.sigma_rate_eff) +
                    " — suspect time-tag error (see the +5 ms robustness "
                    "cell: watch sigma_rate_eff on real data)");
}

}  // namespace detail

/// Estimate an EA from `spec`, choosing the estimator from the type of the
/// observations (see the file header). The two estimators remain callable
/// directly. Does not throw for a bad spec: a missing `noise_sigma` on typed
/// samples is reported in `error_message`; errors from the estimator itself
/// (e.g. missing data files) propagate.
[[nodiscard]] inline EASpecResult estimate_ea(const EAProblemSpec& spec) {
    EASpecResult out;

    if (const auto* sc = std::get_if<MultiSensorScenario>(&spec.observations)) {
        out.method_used = EAMethod::Smoother;
        out.rationale =
            "multi-sensor scenario → GM-RTS smoother (reduced-dynamic "
            "literature answer; beats every batch method measured — "
            "docs/multi_sensor_estimator.md)";
        EASmootherConfig cfg = spec.smoother.value_or(EASmootherConfig{});
        auto r = estimate_ea_smoother(*sc, SensorMask::all_on(*sc),
                                      spec.force_model, cfg);
        static_cast<ResultCore&>(out) = static_cast<const ResultCore&>(r);
        detail::audit_smoother(r, out);
        out.smoother_detail = std::move(r);
        return out;
    }

    // Typed samples → B-spline estimator (GN on by default). noise_sigma must
    // be declared unless a bspline override is given.
    EABSplineConfig cfg = spec.bspline.value_or(EABSplineConfig{});
    if (!spec.bspline && spec.noise_sigma > 0.0)
        cfg.noise_sigma = spec.noise_sigma;
    if (spec.noise_sigma <= 0.0 && !spec.bspline) {
        out.error_message =
            "EAProblemSpec: declare noise_sigma (> 0) for typed samples — "
            "the spec path has no unknown-noise mode (pass a bspline "
            "override with ea::CV{} if you truly cannot declare one)";
        return out;
    }

    // Declared outliers: remove them before fitting and report how many.
    auto tail_filter = [&](const std::vector<std::pair<Epoch, Vector3d>>& d) {
        if (spec.outlier_prob <= 0.0) return d;
        size_t nrej = 0;
        bool oob = false;
        auto f = reject_heavy_tails(d, spec.noise_sigma, spec.outlier_prob,
                                    spec.outlier_scale, &nrej, &oob);
        if (oob)
            out.audit_warnings.push_back(
                "heavy-tail prefilter NOT applied: sample rate below the "
                "measured envelope (~4 Hz; it was validated at 100 Hz) — the "
                "local-median detrend cannot identify outliers on sparse "
                "data, and widening the window reintroduces a "
                "curvature-bias failure. Fit ran on unfiltered data");
        if (nrej > 0)
            out.audit_warnings.push_back(
                "heavy-tail prefilter rejected " + std::to_string(nrej) +
                " of " + std::to_string(d.size()) +
                " samples (declared p=" + std::to_string(spec.outlier_prob) +
                ", k=" + std::to_string(spec.outlier_scale) + ")");
        return f;
    };

    EAResult r;
    if (const auto* p = std::get_if<PositionSamples>(&spec.observations)) {
        out.method_used = EAMethod::BSplinePosition;
        out.rationale = "position samples → B-spline + GN (measured winner)";
        r = estimate_ea_bspline(spec.epoch0, spec.pos0_ecef, spec.vel0_ecef,
                                PositionSamples{tail_filter(p->ecef)},
                                spec.force_model, cfg);
    } else if (const auto* v = std::get_if<VelocitySamples>(&spec.observations)) {
        out.method_used = EAMethod::BSplineVelocity;
        out.rationale = "velocity samples → B-spline + GN (measured winner)";
        // The outlier filter is not applied to velocity data: measured, it
        // made the fit far worse than leaving the data unfiltered. The filter

        // compares each sample with a local median; velocity curves enough
        // that this comparison is biased where the acceleration changes
        // fastest (high jerk), so good samples there are thrown out in
        // clusters. It stays off until a filter that follows slope and
        // curvature is shown to fix this.
        if (spec.outlier_prob > 0.0)
            out.audit_warnings.push_back(
                "outlier_prob declared on VELOCITY samples: the heavy-tail "
                "prefilter was NOT applied and the fit ran on the "
                "unfiltered data. On velocity data the filter throws out "
                "good samples in clusters where the acceleration changes "
                "fastest, which was measured to make the fit far worse "
                "than the outliers themselves; velocity fits are also the "
                "least sensitive to outliers of the three data kinds.");
        r = estimate_ea_bspline(spec.epoch0, spec.pos0_ecef, spec.vel0_ecef,
                                *v, spec.force_model, cfg);
    } else {
        const auto& a = std::get<AccelSamples>(spec.observations);
        out.method_used = EAMethod::BSplineAcceleration;
        out.rationale =
            "acceleration samples → B-spline (direct; GN circular, skipped)";
        r = estimate_ea_bspline(spec.epoch0, spec.pos0_ecef, spec.vel0_ecef,
                                AccelSamples{tail_filter(a.ecef)},
                                spec.force_model, cfg);
    }
    static_cast<ResultCore&>(out) = static_cast<const ResultCore&>(r);
    if (r.obs_rmse && spec.noise_sigma > 0.0) {
        out.sigma_honored.emplace_back("obs", *r.obs_rmse);
        if (*r.obs_rmse > 3.0 * spec.noise_sigma)
            out.audit_warnings.push_back(
                "obs RMSE " + std::to_string(*r.obs_rmse) +
                " exceeds 3x the declared noise_sigma " +
                std::to_string(spec.noise_sigma) +
                " — declaration could not be honored");
    }
    out.bspline_detail = std::move(r);
    return out;
}

/// Information bound for a scenario spec: how accurately ANY estimator could
/// recover the trajectory from this data, computed before solving (see
/// ea_information_bound.hpp). A spec holding typed samples returns a result
/// with `error_message` set (for a single sensor the answer is set by its
/// noise and is not the sensor-design question this tool answers). Does not
/// throw for that case.
[[nodiscard]] inline EAInformationBound information_bound(
    const EAProblemSpec& spec, const EAInformationBoundOptions& opts = {})
{
    if (const auto* sc = std::get_if<MultiSensorScenario>(&spec.observations))
        return information_bound(*sc, SensorMask::all_on(*sc),
                                 spec.force_model, opts);
    EAInformationBound out;
    out.error_message =
        "information_bound: spec must hold a MultiSensorScenario";
    return out;
}

}  // namespace calibration
}  // namespace ballistic
