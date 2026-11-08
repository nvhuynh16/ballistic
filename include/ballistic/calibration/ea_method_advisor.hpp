#pragma once

// =============================================================================
// recommend_ea_method — which empirical-acceleration estimator to use
// =============================================================================
//
// An empirical acceleration (EA) is a small correction acceleration fitted to
// observations and added to the force model. Which estimator and settings fit
// best depends on the data: clean or noisy, gentle or violent motion (a stable
// orbit versus a reentry glide), and which sensors are available. This function
// encodes that choice, as found by comparing the estimators on test scenarios,
// in one testable place. It is advice only: it calls no estimator and changes
// nothing. The same guidance in prose is in
// docs/empirical_acceleration_estimation.md, "Choosing an estimator".
//
// It does not pick and run the estimator for you, because the right choice
// depends on the scenario and automatic scoring can rank close candidates the
// wrong way round; you still call the `estimate_ea_*` function yourself.
// `rationale` says why the recommendation was made.
// =============================================================================

#include <string>

namespace ballistic {
namespace calibration {

/// Kind of observations available for the EA fit.
enum class EAObsType { Position, Velocity, Acceleration, Radar, MultiSensor };

/// The recommended estimator. There are two families: the B-spline estimator
/// (`estimate_ea_bspline`, for dense data of a single kind) and the smoother
/// (`estimate_ea_smoother`, a Kalman filter run forward then backward over the
/// data, which fuses several sensors and handles radar and noisy positions).
enum class EAMethod {
    BSplinePosition, BSplineVelocity, BSplineAcceleration,
    Smoother   // estimate_ea_smoother — sensor fusion, radar, noisy positions
};

/// Description of the problem, input to `recommend_ea_method`.
struct EAScenarioHint {
    EAObsType obs = EAObsType::Position;
    double noise_sigma = 0.0;          // observation noise std dev (obs units); 0 = clean
    bool   aggressive_dynamics = false; // true for reentry/glide, false for a quiet orbit
    bool   params_known = false;        // true if the time scale of the EA's features is known,
                                        // so the fit settings need not be chosen by
                                        // cross-validation
    int    n_radars = 1;                // number of radars (radar data only)
};

/// Output of `recommend_ea_method`.
struct EARecommendation {
    EAMethod    method;
    bool        use_gn = false;  // turn on Gauss-Newton refinement (positions and velocities;
                                 // not accelerometer data, where it has nothing to fit)
    bool        use_cv = false;  // choose the fit settings by blocked cross-validation (CV):
                                 // leave out blocks of data, score the prediction of them
    std::string rationale;       // why, in one or two sentences
};

/// Recommend an EA estimator and its main switches for scenario `h`.
/// Pure function; never throws. Each branch reflects a measured comparison.
[[nodiscard]] inline EARecommendation
recommend_ea_method(const EAScenarioHint& h) {
    const bool noisy = h.noise_sigma > 0.0;
    EARecommendation r;

    switch (h.obs) {
    case EAObsType::Position:
        if (!noisy) {
            if (h.aggressive_dynamics) {
                r.method = EAMethod::BSplinePosition;
                r.rationale =
                    "Clean POS, aggressive dynamics (reentry/glide): "
                    "B-spline + GN (default ON).";
            } else {
                r.method = EAMethod::BSplinePosition;
                r.rationale =
                    "Clean POS: B-spline + GN (default ON) — reproduces "
                    "clean positions to the millimeter level on the test "
                    "fixtures.";
            }
            r.use_gn = true;
        } else if (h.params_known) {
            r.method = EAMethod::BSplinePosition;
            r.use_gn = true;
            r.rationale =
                "Noisy POS, known feature scale: B-spline + GN at "
                "knot≈feature/4 — skip CV when the scale is known. "
                "GN is the load-bearing step for noisy POS.";
        } else {
            r.method = EAMethod::BSplinePosition;
            r.use_gn = true;
            r.use_cv = true;
            r.rationale =
                "Noisy POS, no known params: B-spline + hv-block CV "
                "(ea::CV{}) + GN; CV picks the knot spacing and λ "
                "schedule, GN removes the bias of a coarse spline.";
        }
        break;

    case EAObsType::Velocity:
        r.method = EAMethod::BSplineVelocity;
        // GN helps velocity data: it improved the error over every random
        // noise seed tried.

        r.use_gn = true;
        if (noisy) {
            r.use_cv = true;
            r.rationale =
                "Noisy VEL: B-spline IRGN d5 + hv-block CV + GN "
                "(GN measured better on every noise seed tried).";
        } else {
            r.rationale =
                "Clean VEL: B-spline IRGN d5 + GN (GN measured "
                "beneficial).";
        }
        break;

    case EAObsType::Acceleration:
        r.method = EAMethod::BSplineAcceleration;
        r.use_gn = false;  // no positions to fit; the estimator ignores gn here
        r.use_cv = noisy;
        r.rationale = noisy
            ? "Noisy ACC: B-spline + hv-block CV. ACC GN is circular "
              "(SkippedAccelerationCircular); facade ignores gn."
            : "Clean ACC: B-spline alone — the accelerations are fitted "
              "directly, with no differentiation of positions. ACC GN is "
              "circular.";
        break;

    case EAObsType::Radar:
        // Measured on the radar-only test geometry (test_ea_smoother), the
        // smoother reaches 12.7 m with one radar and 10.6 m with two, using
        // the range-rate channel
        // weighted by the residual level the model can actually match (so it
        // need not be switched off). Put the RadarObs in a MultiSensorScenario
        // (radars + initial state) to call estimate_ea_smoother.
        r.method = EAMethod::Smoother;
        r.use_gn = false;
        r.use_cv = false;  // the smoother picks its own settings (filter consistency check)
        r.rationale = (h.n_radars >= 2)
            ? "Radar (2+): estimate_ea_smoother (wrap in MultiSensorScenario) "
              "— 10.6 m with two radars, measured on the test geometry; "
              "rate included at the measured honest budget."
            : "Radar (single): estimate_ea_smoother (wrap in "
              "MultiSensorScenario) — 12.7 m measured on the test "
              "geometry; no rate-off "
              "gymnastics needed (honest budget auto-measured).";
        break;

    case EAObsType::MultiSensor:
        // The smoother fuses every sensor type: 12.7 m on the single-radar
        // reentry-glide test scenario (test_ea_smoother), without tuning, multiple starts or a
        // stopping rule sensitive to rounding, and with its own consistency
        // statistics in range. Accelerometers are fused by also estimating
        // their bias (a 12-element filter state).
        r.method = EAMethod::Smoother;
        r.use_cv = false;  // the smoother picks its own settings
        r.rationale =
            "Multi-sensor fusion (radar/optics/fix/accel): "
            "estimate_ea_smoother (GM-RTS) — 12.7 m on combo 1, "
            "measured on the test geometry; "
            "auto NIS hyperparameters, measured rate budget, joint "
            "accel-bias state, posterior σ.";
        break;
    }
    return r;
}

} // namespace calibration
} // namespace ballistic
