// =============================================================================
// Calibration Module Python Bindings
// =============================================================================
//
// Bindings for empirical acceleration (EA) estimation with a B-spline fit,
// for splitting an EA into a drag part (ballistic coefficient beta) and a
// remainder, and for Gauss-Newton (GN) refinement.
//
// An empirical acceleration is a small correction acceleration fitted to
// observations of a satellite and added to the physical force model, to
// absorb what the model gets wrong (mis-modeled drag, unknown thrust, ...).
// The ballistic coefficient beta = m / (Cd * A) [kg/m^2] sets the strength of
// drag (larger beta = weaker drag). Gauss-Newton is an iterative
// least-squares method: it adjusts the EA so that the orbit propagated with
// it matches the observed positions. ECEF = Earth-fixed frame (ITRF);
// GCRS = inertial frame.
//
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>

#include <ballistic/ballistic.hpp>
#include <ballistic/calibration/ea_estimator.hpp>

namespace nb = nanobind;
using namespace ballistic;
using namespace ballistic::calibration;

void bind_calibration(nb::module_& m) {

    // =========================================================================
    // Status / result types
    // =========================================================================

    nb::enum_<ea::CV::Mode>(m, "CVMode",
        "What the cross-validation search (EABSplineConfig.set_cv) varies.")
        .value("IRGN", ea::CV::Mode::IRGN,
            "IRGN (iteratively regularized Gauss-Newton): knot spacing × λ₀ × decay × λ_min, "
            "about 495 combinations; the smoothing "
            "weight λ shrinks over the iterations. Default.")
        .value("FixedLambda", ea::CV::Mode::FixedLambda,
            "Knot spacing × a constant λ, about 99 combinations; about 5× faster, but "
            "suited to acceleration data only.");

    nb::enum_<GNStatus>(m, "GNStatus",
        "What happened to the optional Gauss-Newton (GN) refinement step.")
        .value("Disabled", GNStatus::Disabled,
            "Not requested (config.gn is None)")
        .value("Ran", GNStatus::Ran, "GN refinement ran")
        .value("SkippedAccelerationCircular",
               GNStatus::SkippedAccelerationCircular,
            "Skipped for acceleration data: there are no observed positions to fit, so GN "
            "would only fit the orbit to itself");

    nb::class_<EAResult>(m, "EAResult",
        "Result of an empirical acceleration (EA) estimation.")
        .def(nb::init<>())
        .def_ro("empirical_accel", &EAResult::empirical_accel,
            "The estimated EA (EmpiricalAcceleration), in the inertial GCRS frame; set it "
            "as TrajectoryConfig.empirical_accel to propagate with it.")
        .def_ro("collocation_history", &EAResult::collocation_history,
            "The orbit propagated with the estimated EA, sampled at the EA's time points "
            "(list of HistoryEntry, Earth-fixed). Pass it to refine_beta_from_ea.")
        .def_ro("iterations_used", &EAResult::iterations_used,
            "Number of outer iterations performed.")
        .def_ro("gn_status", &EAResult::gn_status,
            "What happened to the optional Gauss-Newton refinement (GNStatus).")
        .def_ro("lambda_used", &EAResult::lambda_used,
            "Smoothing weight λ at the end (None if not used).")
        .def_ro("sigma_sq_estimate", &EAResult::sigma_sq_estimate,
            "Noise variance per component estimated by generalized cross-validation "
            "(GCV) (None if GCV was not used).")
        .def_ro("obs_rmse", &EAResult::obs_rmse,
            "Root-mean-square (RMS) misfit to the observations at the end, in the "
            "observations' units (None if not measured).")
        .def_ro("error_message", &EAResult::error_message,
            "Error description (empty on success).")
        .def("succeeded", &EAResult::succeeded,
            "True if the estimator ran without error and produced an EA "
            "(an EA that is legitimately zero still counts as success).")
        .def("has_correction", &EAResult::has_correction,
            "True if succeeded() and the EA is not zero. Check this before using the EA "
            "for further refinement (GN or beta).");

    nb::class_<EADecomposition>(m, "EADecomposition",
        "An EA vector split into a drag-like part (along the velocity) and the rest "
        "(perpendicular to it).")
        .def(nb::init<>())
        .def_ro("beta", &EADecomposition::beta,
            "Ballistic coefficient implied by the drag part [kg/m^2]")
        .def_ro("ea_drag", &EADecomposition::ea_drag,
            "Part along the velocity, numpy (3,) [m/s^2]")
        .def_ro("ea_residual", &EADecomposition::ea_residual,
            "Part perpendicular to the velocity, numpy (3,) [m/s^2]")
        .def_ro("drag_fraction", &EADecomposition::drag_fraction,
            "|drag part| / |EA|, between 0 and 1");

    nb::class_<BetaStepEstimate>(m, "BetaStepEstimate",
        "Ballistic coefficient estimated at one point of the orbit.")
        .def(nb::init<>())
        .def_ro("time_s", &BetaStepEstimate::time_s, "Seconds after epoch0")
        .def_ro("altitude_km", &BetaStepEstimate::altitude_km,
            "Geodetic altitude (height above the WGS84 ellipsoid) [km]")
        .def_ro("beta", &BetaStepEstimate::beta, "Ballistic coefficient [kg/m^2]")
        .def_ro("w", &BetaStepEstimate::w, "1 / beta [m^2/kg]")
        .def_ro("weight", &BetaStepEstimate::weight, "Confidence weight of this estimate");

    nb::class_<BetaResult>(m, "BetaResult",
        "Ballistic coefficients estimated from an EA along an orbit.")
        .def(nb::init<>())
        .def_ro("step_estimates", &BetaResult::step_estimates,
            "Estimate at each point (list of BetaStepEstimate)")
        .def_ro("constant_beta", &BetaResult::constant_beta,
            "Median of the point estimates [kg/m^2]")
        .def_ro("node_altitudes_km", &BetaResult::node_altitudes_km,
            "Altitudes of the bins used for an altitude-dependent beta [km]")
        .def_ro("node_betas", &BetaResult::node_betas,
            "Beta in each altitude bin [kg/m^2]")
        .def_ro("steps_solved", &BetaResult::steps_solved,
            "Number of valid point estimates")
        .def_ro("error_message", &BetaResult::error_message,
            "Error description (empty if success)");

    // =========================================================================
    // Decomposition + Refinement (top-level)
    // =========================================================================

    nb::class_<BetaDecompositionConfig>(m, "BetaDecompositionConfig",
        "Settings for estimating the ballistic coefficient from an EA "
        "(refine_beta_from_ea).")
        .def(nb::init<>())
        .def_rw("min_altitude_m", &BetaDecompositionConfig::min_altitude_m,
            "Ignore points below this altitude [m] (default: 50000)")
        .def_rw("min_beta", &BetaDecompositionConfig::min_beta,
            "Discard point estimates below this beta [kg/m^2] (default: 1.0)")
        .def_rw("max_beta", &BetaDecompositionConfig::max_beta,
            "Discard point estimates above this beta [kg/m^2] (default: 1e5)")
        .def_rw("altitude_nodes_km", &BetaDecompositionConfig::altitude_nodes_km,
            "Optional altitude bins [km] for an altitude-dependent beta");


    m.def("decompose_ea",
        &decompose_ea,
        nb::arg("ea"), nb::arg("velocity"), nb::arg("density_kg_m3"),
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(Split one EA vector into a drag-like part (along the velocity)
and the rest (perpendicular), and compute the ballistic coefficient the
drag part implies, from the drag equation |a| = rho * v^2 / (2 * beta).
This is a split in the frame given, not along the satellite's own axes.
beta stays 0 when the along-velocity part does not oppose the motion, the
density is not positive, |v| < 1 m/s or the EA is zero. Never raises.

Both ea and velocity must be in the same frame; for drag, use the velocity
relative to the air (Earth-fixed, ECEF).

Args:
    ea: EA vector, numpy (3,) [m/s^2]
    velocity: velocity vector, numpy (3,) [m/s]
    density_kg_m3: air density rho [kg/m^3]

Returns:
    EADecomposition with beta, ea_drag, ea_residual, drag_fraction
)doc");

    // decompose_to_beta / compose_ea_from_beta (which take a density callback)
    // are internal (detail::); the public EA -> beta entry point is
    // refine_beta_from_ea, which takes a DragEnvironment.

    m.def("refine_beta_from_ea",
        [](const atmosphere::DragEnvironment& prior,
           const forces::EmpiricalAcceleration& ea,
           const std::vector<HistoryEntry>& trajectory,
           const Epoch& epoch0,
           const BetaDecompositionConfig& config) {
            return refine_beta_from_ea(prior, ea, trajectory, epoch0, config);
        },
        nb::arg("prior"), nb::arg("ea"),
        nb::arg("trajectory"), nb::arg("epoch0"),
        nb::arg("config") = BetaDecompositionConfig{},
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
Estimate a satellite's ballistic coefficient β from a fitted empirical
acceleration, and return a DragEnvironment that uses it.

The EA's component along the velocity is read as missing (or extra) drag;
together with the prior environment's density it gives a refined β profile.
The returned DragEnvironment keeps the prior's density and wind and replaces
only its β column.

The trajectory must be the orbit the EA was applied to, i.e. propagated
with the EA in the force model, so that its velocities match the flight the
EA describes. Every EAResult carries such an orbit as collocation_history;
pass that as `trajectory`.

Args:
    prior: DragEnvironment with density and wind filled in. Its β column is
        replaced in the returned environment.
    ea: EmpiricalAcceleration to split into drag (β) and remainder.
    trajectory: list of HistoryEntry (typically result.collocation_history).
    epoch0: start Epoch of the arc.
    config: optional BetaDecompositionConfig (β limits, altitude bins, ...).

Returns:
    DragEnvironment with the refined BetaProfile1D installed.
)doc");


    // =========================================================================
    // EA estimator, public API
    // All observations are Earth-fixed (ECEF); converted internally as needed.
    // =========================================================================

    nb::class_<GNRefine>(m, "GNRefine",
        "Settings for Gauss-Newton (GN) refinement: after the B-spline fit, the EA is "
        "adjusted so that the orbit propagated with it matches the observed positions as "
        "closely as possible. Assign to EABSplineConfig.gn (on by default).")
        .def(nb::init<>())
        .def_rw("reg_lambda", &GNRefine::reg_lambda,
            "Weight of the smoothness penalty during refinement, the same kind of penalty "
            "as in the B-spline fit, so both minimize the same objective. Default 1.0. "
            "0.0 turns it off: on noisy data the propagated orbit still matches, but the "
            "EA itself becomes badly wrong point by point.")
        .def_rw("reg_deriv_order", &GNRefine::reg_deriv_order,
            nb::for_setter(nb::arg().none()),
            "Which time derivative of the EA the smoothness penalty acts on (None = "
            "automatic: 3 for positions, 1 for velocities); same meaning as "
            "EABSplineConfig.reg_deriv_order.")
        .def_rw("max_iterations", &GNRefine::max_iterations,
            "Maximum GN iterations (default: 30)")
        .def_rw("sample_stride", &GNRefine::sample_stride,
            "Fit every N-th observation (default: 20).")
        .def_rw("ea_subsample", &GNRefine::ea_subsample,
            "Adjust every N-th EA time point (default: 1 = all of them)")
        .def_rw("fd_delta", &GNRefine::fd_delta,
            "Finite-difference perturbation of an EA value [m/s^2] (default: 1e-2), used "
            "when use_stm is False")
        .def_prop_rw("rk45_max_step",
            [](const GNRefine& g) -> nb::object {
                if (std::holds_alternative<GNRefine::MatchKnotSpacing>(g.rk45_max_step))
                    return nb::str("match_knots");
                if (const double* v = std::get_if<double>(&g.rk45_max_step))
                    return nb::float_(*v);
                return nb::none();
            },
            [](GNRefine& g, nb::object val) {
                if (val.is_none()) { g.rk45_max_step = std::monostate{}; return; }
                if (nb::isinstance<nb::str>(val)) {
                    const std::string sv = nb::cast<std::string>(val);
                    if (sv == "match_knots") { g.rk45_max_step = GNRefine::MatchKnotSpacing{}; return; }
                    throw nb::value_error("rk45_max_step: expected None, 'match_knots', or a float");
                }
                g.rk45_max_step = nb::cast<double>(val);
            },
            nb::for_setter(nb::arg().none()),
            "Largest RK45 integrator step during GN propagations: None = no override, "
            "fully adaptive (default, recommended); 'match_knots' = the estimator's knot "
            "spacing (measured to make results worse); a float = this step [s].")
        .def_rw("use_stm", &GNRefine::use_stm,
            "How GN gets its derivatives. True (default): the state transition matrix "
            "(STM), i.e. the sensitivity of the orbit to each EA value, integrated "
            "alongside the orbit in one propagation; about 2× faster. False: finite "
            "differences, one extra propagation per EA value. The STM path interpolates "
            "the EA linearly between its time points, which cost about 30% accuracy on "
            "position data with the default degree-5 spline; set False for a one-off run "
            "where accuracy matters most.")
        .def_rw("stm_dt_s", &GNRefine::stm_dt_s,
            "Step [s] for integrating the sensitivity equations (default: 0.5)")
        .def_rw("stm_fd_dx", &GNRefine::stm_fd_dx,
            "Position perturbation [m] for the derivative of acceleration with respect to "
            "position (default: 1.0)")
        .def_rw("stm_fd_dv", &GNRefine::stm_fd_dv,
            "Velocity perturbation [m/s] for the derivative of acceleration with respect "
            "to velocity (default: 1e-3)")
        .def_rw("stm_verify", &GNRefine::stm_verify,
            "First iteration only: also compute finite-difference derivatives and print the "
            "difference (default: False)")
        .def_rw("stm_verbose", &GNRefine::stm_verbose,
            "Print the STM computation's internal trace (default: False)");


    nb::class_<EABSplineConfig>(m, "EABSplineConfig",
        "Settings for estimate_ea_bspline, which fits the empirical acceleration as a "
        "B-spline (a smooth piecewise polynomial in time). A smoothness penalty with weight "
        "λ keeps the EA from chasing noise; set_irgn / set_gcv / set_cv / set_fixed_lambda "
        "choose how λ is picked (default: IRGN, iteratively regularized Gauss-Newton, "
        "where λ starts large and shrinks each iteration).")
        .def(nb::init<>())
        .def_rw("noise_sigma", &EABSplineConfig::noise_sigma,
            "Observation noise standard deviation σ in the data's units (m for positions, "
            "m/s for velocities, m/s² for accelerations). If > 0, the fit stops once the "
            "residual RMS is at most 1.5σ. Typical levels: GPS positions 1-10 m (3 m is a "
            "good default assumption), Doppler velocities 0.01-0.25 m/s, accelerometers "
            "1e-6 to 0.1 m/s².")
        .def_rw("knot_spacing_s", &EABSplineConfig::knot_spacing_s,
            "Time between spline knots [s] (default: 2.0); smaller lets the EA change faster")
        .def_rw("degree", &EABSplineConfig::degree,
            "B-spline polynomial degree: 5 = quintic (default), 3 = cubic")
        .def_rw("max_iterations", &EABSplineConfig::max_iterations,
            "Maximum outer iterations; the fit also stops after 5 iterations in a row "
            "without improvement (default: 50)")
        .def_rw("max_duration_s", &EABSplineConfig::max_duration_s,
            "Longest arc fitted [s] (default: 3600). Reduced automatically to the last "
            "observation time + 10 s; set it below the data span only to fit deliberately "
            "less of the data")
        .def_rw("reg_deriv_order", &EABSplineConfig::reg_deriv_order,
            nb::for_setter(nb::arg().none()),
            "Which time derivative of the spline the smoothness penalty acts on (None = "
            "automatic: 3 for positions, 1 for velocities and accelerations); must not "
            "exceed degree.")
        .def_rw("gn", &EABSplineConfig::gn,
            nb::for_setter(nb::arg().none()),
            "Gauss-Newton refinement settings (GNRefine), on by default; None skips it "
            "(faster, less accurate). Not applied to acceleration data. With "
            "cross-validation (set_cv), the GN settings are passed on to the GN runs inside "
            "the search.")
        .def_rw("brent_line_search", &EABSplineConfig::brent_line_search,
            "How the length of each update step is searched. True (default): bracket the "
            "best length from 5 trials, then refine with Brent's method; about 2-3× fewer "
            "propagations than False (24 fixed trial lengths), and more accurate for "
            "positions and accelerations. For velocities the mean error is the same but "
            "the spread is wider; consider False for a single velocity fit.")
        .def("set_irgn", [](EABSplineConfig& c, double l0, double q, double lm) {
            c.regularization = ea::IRGN{l0, q, lm};
        }, nb::arg("lambda_0") = 1e-4, nb::arg("decay") = 0.5,
           nb::arg("lambda_min") = 1e-12,
            "Pick λ by IRGN (the default): iteration k uses "
            "λ_k = max(lambda_min, lambda_0 * decay^k), so the first iterations fit only "
            "the broad shape and later ones the detail.")
        .def("set_gcv", [](EABSplineConfig& c, double lo, double hi) {
            c.regularization = ea::GCV{lo, hi};
        }, nb::arg("lam_min") = 1e-12, nb::arg("lam_max") = 1e-1,
            "Pick λ at each iteration by generalized cross-validation (GCV), from the data "
            "alone, searched within [lam_min, lam_max]; needs no noise level.")
        .def("set_cv", [](EABSplineConfig& c, int hv_n_blocks,
                          std::vector<int> hv_test_blocks, int hv_gap_blocks,
                          ea::CV::Mode mode) {
            c.regularization = ea::CV{hv_n_blocks,
                                       std::move(hv_test_blocks),
                                       hv_gap_blocks, mode};
        }, nb::arg("hv_n_blocks") = 18,
           nb::arg("hv_test_blocks") = std::vector<int>{},
           nb::arg("hv_gap_blocks") = 1,
           nb::arg("mode") = ea::CV::Mode::IRGN,
            "Pick the settings by a cross-validation grid search: each combination is "
            "fitted on part of the data and scored on how well it predicts the held-out "
            "rest. The data are split into hv_n_blocks contiguous time blocks; "
            "hv_test_blocks lists the held-out ones (empty = every 5th starting at block "
            "2); hv_gap_blocks blocks on each side of a held-out block are left out of "
            "training, so correlated neighbors do not leak into the score. mode=IRGN "
            "(default, about 495 combinations) or mode=FixedLambda (about 99, about 5× "
            "faster, acceleration data only). Slow, but the best automatic choice for "
            "noisy data.")
        .def("set_fixed_lambda", [](EABSplineConfig& c, double lam) {
            c.regularization = ea::FixedLambda{lam};
        }, nb::arg("lambda_") = 1e-4,
            "Use the same λ in every iteration. Suited to acceleration data only.");


    m.def("refine_ea_gn",
        [](const Epoch& epoch0, const Vector3d& pos, const Vector3d& vel,
           const std::vector<std::pair<Epoch, Vector3d>>& position_data,
           std::shared_ptr<forces::EmpiricalAcceleration> initial_ea,
           const TrajectoryConfig& force_model,
           const GNRefine& gn, double max_duration_s) {
            return refine_ea_gn(epoch0, pos, vel, position_data,
                                std::move(initial_ea), force_model, gn,
                                max_duration_s);
        },
        nb::arg("epoch"), nb::arg("position"), nb::arg("velocity"),
        nb::arg("position_data"), nb::arg("initial_ea"),
        nb::arg("force_model"), nb::arg("gn") = GNRefine{},
        nb::arg("max_duration_s") = 3600.0,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(Refine an empirical acceleration by Gauss-Newton so that the
orbit propagated with it matches observed positions.

Args:
    epoch: start Epoch
    position, velocity: start state, Earth-fixed (ECEF), numpy (3,)
        [m] and [m/s]
    position_data: observed positions, list of (Epoch, numpy (3,) ECEF
        position [m])
    initial_ea: EmpiricalAcceleration to start from
    force_model: TrajectoryConfig for the propagations
    gn: GNRefine settings (default: STM derivatives)
    max_duration_s: longest arc used [s]

Returns:
    Refined EmpiricalAcceleration (a table on the initial EA's time points,
    in its frame). The initial EA is returned unchanged if it is None or
    zero, or if fewer than 3 observations remain after sample_stride and
    max_duration_s.
)doc");

    // --- Typed observation samples ---
    // The observation modality is carried by the sample type; the estimator
    // families are single overloaded names.

    nb::class_<PositionSamples>(m, "PositionSamples",
        "Observed Earth-fixed (ECEF) positions for estimate_ea_bspline: a list of "
        "(Epoch, numpy (3,) position [m]).")
        .def(nb::init<>())
        .def("__init__", [](PositionSamples* self,
                            std::vector<std::pair<Epoch, Vector3d>> data) {
                new (self) PositionSamples{std::move(data)};
            }, nb::arg("samples"))
        .def_rw("ecef", &PositionSamples::ecef, "The samples: list of (Epoch, numpy (3,)), ECEF");

    nb::class_<VelocitySamples>(m, "VelocitySamples",
        "Observed Earth-fixed (ECEF) velocities for estimate_ea_bspline: a list of "
        "(Epoch, numpy (3,) velocity relative to the rotating Earth [m/s]).")
        .def(nb::init<>())
        .def("__init__", [](VelocitySamples* self,
                            std::vector<std::pair<Epoch, Vector3d>> data) {
                new (self) VelocitySamples{std::move(data)};
            }, nb::arg("samples"))
        .def_rw("ecef", &VelocitySamples::ecef, "The samples: list of (Epoch, numpy (3,)), ECEF");

    nb::class_<AccelSamples>(m, "AccelSamples",
        "Observed accelerations for estimate_ea_bspline: a list of (Epoch, numpy (3,) "
        "acceleration of the Earth-fixed (ECEF) coordinates [m/s²]).")
        .def(nb::init<>())
        .def("__init__", [](AccelSamples* self,
                            std::vector<std::pair<Epoch, Vector3d>> data) {
                new (self) AccelSamples{std::move(data)};
            }, nb::arg("samples"))
        .def_rw("ecef", &AccelSamples::ecef, "The samples: list of (Epoch, numpy (3,)), ECEF");

    // --- B-spline estimator: one name, modality by sample type ---

    m.def("estimate_ea_bspline",
        nb::overload_cast<const Epoch&, const Vector3d&, const Vector3d&,
                          const PositionSamples&, const TrajectoryConfig&,
                          const EABSplineConfig&>(&estimate_ea_bspline),
        nb::arg("epoch"), nb::arg("position"), nb::arg("velocity"),
        nb::arg("observations"), nb::arg("force_model"),
        nb::arg("config") = EABSplineConfig{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Estimate an empirical acceleration from observed positions (PositionSamples). "
        "epoch / position / velocity: the start state, Earth-fixed (ECEF) [m, m/s]; "
        "force_model: TrajectoryConfig; config: EABSplineConfig. Gauss-Newton refinement "
        "(config.gn) is on by default and greatly improves position fits; keep it. "
        "Returns an EAResult whose EA is in the inertial GCRS frame.");

    m.def("estimate_ea_bspline",
        nb::overload_cast<const Epoch&, const Vector3d&, const Vector3d&,
                          const VelocitySamples&, const TrajectoryConfig&,
                          const EABSplineConfig&>(&estimate_ea_bspline),
        nb::arg("epoch"), nb::arg("position"), nb::arg("velocity"),
        nb::arg("observations"), nb::arg("force_model"),
        nb::arg("config") = EABSplineConfig{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Estimate an empirical acceleration from observed velocities (VelocitySamples). "
        "Arguments as for the position form. Gauss-Newton refinement (config.gn, on by "
        "default) also helps velocity fits. Returns an EAResult (EA in GCRS).");

    m.def("estimate_ea_bspline",
        nb::overload_cast<const Epoch&, const Vector3d&, const Vector3d&,
                          const AccelSamples&, const TrajectoryConfig&,
                          const EABSplineConfig&>(&estimate_ea_bspline),
        nb::arg("epoch"), nb::arg("position"), nb::arg("velocity"),
        nb::arg("observations"), nb::arg("force_model"),
        nb::arg("config") = EABSplineConfig{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Estimate an empirical acceleration from observed accelerations (AccelSamples). "
        "Arguments as for the position form. Gauss-Newton refinement does not apply (no "
        "positions to fit); the result reports gn_status = SkippedAccelerationCircular. "
        "Returns an EAResult (EA in GCRS).");
}
