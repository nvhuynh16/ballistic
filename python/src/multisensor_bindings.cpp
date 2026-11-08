// =============================================================================
// Multi-Sensor EA Estimator Module — Python Bindings
// =============================================================================
//
// Bindings for estimating an empirical acceleration (EA: a small correction
// acceleration fitted to observations and added to the force model) from
// several sensors at once: the sensor observation types (radar, optics,
// position fixes, accelerometer with an unknown bias), the scenario and the
// sensor mask, the smoother (estimate_ea_smoother), and the one-call entry
// point estimate_ea with its sensor-design bound information_bound.
//
// The smoother is a Gauss-Markov Rauch-Tung-Striebel (GM-RTS) smoother: the
// EA is modeled as a random signal of typical size sigma_a that stays
// correlated for about tau seconds (a first-order Gauss-Markov process); an
// extended Kalman filter runs forward through the observations, then a
// Rauch-Tung-Striebel (RTS) pass runs backward so every estimate uses the
// data on both sides of it. Its quality measure is the normalized innovation
// squared (NIS): the average squared difference between each observation and
// its prediction, divided by the variance the filter expected; about 1 when
// the declared noise matches the data.
//
// Frame conventions:
//   - Positions and velocities are Earth-fixed (ECEF / ITRF).
//   - The returned `empirical_accel` is in the inertial GCRS frame.
//   - AccelObs samples are specific force (the non-gravitational acceleration
//     an accelerometer feels) in ECEF axes; rotate body-frame data
//     beforehand.
//
// See `docs/multi_sensor_estimator.md` for the user guide.
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/eigen/dense.h>

#include <ballistic/ballistic.hpp>
#include <ballistic/calibration/multisensor.hpp>
#include <ballistic/calibration/ea_smoother.hpp>
#include <ballistic/calibration/ea_problem_spec.hpp>

namespace nb = nanobind;
using namespace ballistic;
using namespace ballistic::calibration;

void bind_multisensor(nb::module_& m) {

    // =========================================================================
    // ENUMS FIRST — required by nanobind ordering rules (any enum used as
    // an `nb::arg() = default` must be registered before the function with
    // that default).
    // =========================================================================




    nb::enum_<EASmootherConfig::RateBudget>(m, "EASmootherRateBudget",
        "How the smoother weights radar range-rate samples. The weight must reflect how "
        "well the model can actually match range rate, not the datasheet noise.")
        .value("Auto", EASmootherConfig::RateBudget::Auto,
               "Default: fit once without range rate, measure its residual RMS, and use "
               "the larger of that and the sensor's sigma_rate")
        .value("Off", EASmootherConfig::RateBudget::Off, "Ignore range rate")
        .value("Manual", EASmootherConfig::RateBudget::Manual,
               "Use EASmootherConfig.sigma_rate_manual [m/s]");

    nb::enum_<SensorClock>(m, "SensorClock",
        "How the smoother treats a sensor's time tags (its clock). A clock offset makes "
        "every sample look as if taken slightly earlier or later; the smoother can "
        "estimate such an offset when EASmootherConfig.estimate_clock_bias is on.")
        .value("None_", SensorClock::None,
               "The time tags are accurate; no clock offset is estimated for this sensor")
        .value("Shared", SensorClock::Shared,
               "One offset shared by every sensor marked Shared (the radar default: all "
               "radars on one clock)")
        .value("Own", SensorClock::Own, "This sensor gets its own clock offset");

    // =========================================================================
    // Sensor types
    // =========================================================================

    nb::class_<RadarObs>(m, "RadarObs",
        "Ground radar: each sample is [range (m), azimuth (rad), elevation (rad), range "
        "rate (m/s, how fast the distance changes)] to the satellite; azimuth is measured "
        "from north toward east. Set ecef_to_enu to match position_ecef before passing it "
        "to an estimator (the identity default is only a placeholder).")
        .def(nb::init<>())
        .def_rw("position_ecef", &RadarObs::position_ecef,
            "Radar location, Earth-fixed (ECEF), numpy (3,) [m].")
        .def_rw("sigma_range", &RadarObs::sigma_range,
            "Range noise standard deviation σ [m]. Default 2.0.")
        .def_rw("sigma_angle", &RadarObs::sigma_angle,
            "Azimuth/elevation noise σ [rad]. Default 0.02° = 3.5e-4 rad.")
        .def_rw("sigma_rate", &RadarObs::sigma_rate,
            "Range-rate noise σ [m/s]. Default 0.02.")
        .def_rw("lat_rad", &RadarObs::lat_rad,
            "Radar latitude [rad]; informational only, not used by the model (set "
            "ecef_to_enu instead).")
        .def_rw("lon_rad", &RadarObs::lon_rad,
            "Radar longitude [rad]; informational only, not used by the model.")
        .def_rw("known_bias", &RadarObs::known_bias,
            "Known measurement bias [range m, az rad, el rad, rate m/s], subtracted from "
            "every observation before estimation. Default zero.")
        .def_rw("sigma_scale", &RadarObs::sigma_scale,
            "Optional per-sample multiplier of the noise σ values (one entry per sample; "
            "empty = the σ values apply unchanged).")
        .def_rw("clock", &RadarObs::clock,
            "How this radar's time tags are treated (SensorClock; default Shared).")
        .def_rw("ecef_to_enu", &RadarObs::ecef_to_enu,
            "3×3 rotation from Earth-fixed (ECEF) axes to the radar's local east-north-up "
            "(ENU) axes; must match the location.")
        .def("predict", &RadarObs::predict, nb::arg("state"),
            "Predicted [range_m, az_rad, el_rad, range_rate_mps] for a satellite state "
            "(StateVector, Earth-fixed, m and m/s); zeros if the satellite is within 1 m "
            "of the radar. Useful for making synthetic observations.")
        .def_rw("samples", &RadarObs::samples,
            "Observations: list of (Epoch, [range, az, el, range_rate]).")
        .def("set_samples",
            [](RadarObs& self, const Epoch& epoch0,
               Eigen::Ref<const Eigen::VectorXd> times_s,
               Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> obs)
            {
                const Eigen::Index N = times_s.size();
                if (obs.rows() != N) {
                    throw std::invalid_argument(
                        "RadarObs.set_samples: times_s.size() and observations.rows() must match");
                }
                self.samples.clear();
                self.samples.reserve(static_cast<size_t>(N));
                for (Eigen::Index i = 0; i < N; ++i) {
                    self.samples.emplace_back(epoch0 + times_s[i],
                        Eigen::Vector4d(obs(i, 0), obs(i, 1), obs(i, 2), obs(i, 3)));
                }
            },
            nb::arg("epoch0"), nb::arg("times_s"), nb::arg("observations"),
            "Replace the samples from numpy arrays: times_s, shape (N,), seconds after "
            "epoch0; observations, shape (N, 4), [range_m, az_rad, el_rad, range_rate_mps]. "
            "Raises ValueError if the lengths differ.");

    nb::class_<OpticsObs>(m, "OpticsObs",
        "Optical tracker that reports directions only (azimuth, elevation), already "
        "extracted from the images: a telescope, infrared tracker, processed video, "
        "anything that outputs angles. Set ecef_to_enu to match position_ecef.")
        .def(nb::init<>())
        .def_rw("position_ecef", &OpticsObs::position_ecef,
            "Tracker location, Earth-fixed (ECEF), numpy (3,) [m].")
        .def_rw("sigma_angle", &OpticsObs::sigma_angle,
            "Azimuth/elevation noise σ [rad]. Default 50 µrad.")
        .def_rw("lat_rad", &OpticsObs::lat_rad,
            "Latitude [rad]; informational only, not used by the model.")
        .def_rw("lon_rad", &OpticsObs::lon_rad,
            "Longitude [rad]; informational only, not used by the model.")
        .def_rw("ecef_to_enu", &OpticsObs::ecef_to_enu,
            "3×3 rotation from ECEF axes to local east-north-up axes; must be set.")
        .def_rw("known_bias", &OpticsObs::known_bias,
            "Known bias [az rad, el rad], subtracted from every sample before estimation.")
        .def_rw("sigma_scale", &OpticsObs::sigma_scale,
            "Optional per-sample multiplier of sigma_angle (empty = none).")
        .def_rw("clock", &OpticsObs::clock,
            "How the time tags are treated (SensorClock; default None_: assumed accurate).")
        .def("predict", &OpticsObs::predict, nb::arg("state"),
            "Predicted [az_rad, el_rad] for a satellite state (StateVector, ECEF); zeros "
            "if the satellite is within 1 m of the tracker.")
        .def_rw("samples", &OpticsObs::samples,
            "Observations: list of (Epoch, [az, el]).")
        .def("set_samples",
            [](OpticsObs& self, const Epoch& epoch0,
               Eigen::Ref<const Eigen::VectorXd> times_s,
               Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>> obs)
            {
                const Eigen::Index N = times_s.size();
                if (obs.rows() != N) {
                    throw std::invalid_argument(
                        "OpticsObs.set_samples: times_s.size() and observations.rows() must match");
                }
                self.samples.clear();
                self.samples.reserve(static_cast<size_t>(N));
                for (Eigen::Index i = 0; i < N; ++i) {
                    self.samples.emplace_back(epoch0 + times_s[i],
                        Eigen::Vector2d(obs(i, 0), obs(i, 1)));
                }
            },
            nb::arg("epoch0"), nb::arg("times_s"), nb::arg("observations"),
            "Replace the samples from numpy arrays: times_s, shape (N,), seconds after "
            "epoch0; observations, shape (N, 2), [az_rad, el_rad]. Raises ValueError if "
            "the lengths differ.");

    nb::class_<PositionFixObs>(m, "PositionFixObs",
        "Measured Earth-fixed (ECEF) positions of the object: an impact location, GPS "
        "fixes, etc.")
        .def(nb::init<>())
        .def_rw("known_bias", &PositionFixObs::known_bias,
            "Known bias, ECEF, numpy (3,) [m], subtracted before estimation.")
        .def_rw("sigma_scale", &PositionFixObs::sigma_scale,
            "Optional per-sample multiplier of sigma_pos (empty = none).")
        .def_rw("sigma_pos", &PositionFixObs::sigma_pos,
            "Position noise σ per axis [m] (default 3.0), shared by all fixes in this "
            "object.")
        .def_rw("samples", &PositionFixObs::samples,
            "Observations: list of (Epoch, ECEF position [m]).")
        .def("set_samples",
            [](PositionFixObs& self, const Epoch& epoch0,
               Eigen::Ref<const Eigen::VectorXd> times_s,
               Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> obs)
            {
                const Eigen::Index N = times_s.size();
                if (obs.rows() != N) {
                    throw std::invalid_argument(
                        "PositionFixObs.set_samples: times_s.size() and observations.rows() must match");
                }
                self.samples.clear();
                self.samples.reserve(static_cast<size_t>(N));
                for (Eigen::Index i = 0; i < N; ++i) {
                    self.samples.emplace_back(epoch0 + times_s[i],
                        ballistic::Vector3d(obs(i, 0), obs(i, 1), obs(i, 2)));
                }
            },
            nb::arg("epoch0"), nb::arg("times_s"), nb::arg("observations"),
            "Replace the samples from numpy arrays: times_s, shape (N,), seconds after "
            "epoch0; observations, shape (N, 3), ECEF positions [m]. Raises ValueError if "
            "the lengths differ.");

    nb::class_<AccelObs>(m, "AccelObs",
        "On-board accelerometer; its constant bias is estimated together with the EA. "
        "Model: reading = EA + bias, i.e. the force model is assumed to hold gravity only, "
        "so all the non-gravitational acceleration the accelerometer feels (its specific "
        "force) is attributed to the EA. Samples are specific force in Earth-fixed (ECEF) "
        "axes [m/s²]: real accelerometers report in the spacecraft body frame, so rotate "
        "each sample with the attitude at its time before storing it.")
        .def(nb::init<>())
        .def_rw("known_bias", &AccelObs::known_bias,
            "Known part of the bias, numpy (3,) [m/s²], subtracted from every sample; the "
            "remaining unknown bias is still estimated, with sigma_bias_prior as its "
            "expected size.")
        .def_rw("sigma_scale", &AccelObs::sigma_scale,
            "Optional per-sample multiplier of sigma_accel (empty = none).")
        .def_rw("outlier_prob", &AccelObs::outlier_prob,
            "Optional outlier model: the noise is mostly Gaussian with σ = sigma_accel, but "
            "a fraction p = outlier_prob of samples are outliers with σ multiplied by k = "
            "outlier_scale, i.e. (1-p)·N(0,σ²) + p·N(0,(kσ)²). With p > 0 the smoother "
            "down-weights each sample by how far it falls from the prediction (the Masreliez "
            "robust update for this mixture). Measured with 5% of samples at 10σ "
            "(test_ea_smoother): error 21.8 m without it, 11.2 m with p and k wrong by 2.5× "
            "and 2×, so declare your best estimate. Default 0 = off.")
        .def_rw("outlier_scale", &AccelObs::outlier_scale,
            "Outlier noise multiplier k (default 10); see outlier_prob.")
        .def_rw("sigma_accel", &AccelObs::sigma_accel,
            "Accelerometer noise σ per axis [m/s²]. Default 0.01.")
        .def_rw("sigma_bias_prior", &AccelObs::sigma_bias_prior,
            "Expected size (standard deviation) of the unknown bias [m/s²]. Default 1e-3 "
            "(a tactical-grade inertial measurement unit).")
        .def_rw("samples", &AccelObs::samples,
            "Observations: list of (Epoch, ECEF specific force [m/s²]).")
        .def("set_samples",
            [](AccelObs& self, const Epoch& epoch0,
               Eigen::Ref<const Eigen::VectorXd> times_s,
               Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> obs)
            {
                const Eigen::Index N = times_s.size();
                if (obs.rows() != N) {
                    throw std::invalid_argument(
                        "AccelObs.set_samples: times_s.size() and observations.rows() must match");
                }
                self.samples.clear();
                self.samples.reserve(static_cast<size_t>(N));
                for (Eigen::Index i = 0; i < N; ++i) {
                    self.samples.emplace_back(epoch0 + times_s[i],
                        ballistic::Vector3d(obs(i, 0), obs(i, 1), obs(i, 2)));
                }
            },
            nb::arg("epoch0"), nb::arg("times_s"), nb::arg("observations"),
            "Replace the samples from numpy arrays: times_s, shape (N,), seconds after "
            "epoch0; observations, shape (N, 3), ECEF specific force [m/s²]. Raises "
            "ValueError if the lengths differ.");

    // =========================================================================
    // Scenario + mask
    // =========================================================================

    nb::class_<MultiSensorScenario>(m, "MultiSensorScenario",
        "Everything a multi-sensor EA fit needs besides settings: the start time and\n"
        "state, the arc length and the sensors with their samples. The start state\n"
        "must be known exactly (it is not estimated from the observations).\n"
        "\n"
        "Two construction styles supported:\n"
        "  s = MultiSensorScenario(); s.epoch0 = ...; s.add_radar(...)\n"
        "  s = MultiSensorScenario(epoch0=..., pos0_ecef=..., vel0_ecef=...,\n"
        "                          arc_duration_s=..., radars=[...], fixes=[...])")
        .def(nb::init<>())
        .def("__init__",
            [](MultiSensorScenario* self,
               const Epoch& epoch0,
               const ballistic::Vector3d& pos0_ecef,
               const ballistic::Vector3d& vel0_ecef,
               double arc_duration_s,
               std::vector<RadarObs> radars,
               std::vector<OpticsObs> optics,
               std::vector<PositionFixObs> fixes,
               std::vector<AccelObs> accels)
            {
                new (self) MultiSensorScenario{};
                self->epoch0 = epoch0;
                self->pos0_ecef = pos0_ecef;
                self->vel0_ecef = vel0_ecef;
                self->arc_duration_s = arc_duration_s;
                self->radars = std::move(radars);
                self->optics = std::move(optics);
                self->fixes = std::move(fixes);
                self->accels = std::move(accels);
            },
            nb::arg("epoch0"), nb::arg("pos0_ecef"), nb::arg("vel0_ecef"),
            nb::arg("arc_duration_s"),
            nb::arg("radars") = std::vector<RadarObs>{},
            nb::arg("optics") = std::vector<OpticsObs>{},
            nb::arg("fixes") = std::vector<PositionFixObs>{},
            nb::arg("accels") = std::vector<AccelObs>{},
            "Build from keywords; the sensor lists default to empty.")
        .def_rw("epoch0", &MultiSensorScenario::epoch0,
            "Start time (Epoch).")
        .def_rw("pos0_ecef", &MultiSensorScenario::pos0_ecef,
            "Start position, Earth-fixed (ECEF), numpy (3,) [m]; known exactly.")
        .def_rw("vel0_ecef", &MultiSensorScenario::vel0_ecef,
            "Start velocity relative to the rotating Earth, numpy (3,) [m/s]; known "
            "exactly.")
        .def_rw("arc_duration_s", &MultiSensorScenario::arc_duration_s,
            "Length of the arc to fit [s].")
        .def_rw("radars", &MultiSensorScenario::radars, "List of RadarObs")
        .def_rw("optics", &MultiSensorScenario::optics, "List of OpticsObs")
        .def_rw("fixes", &MultiSensorScenario::fixes, "List of PositionFixObs")
        .def_rw("accels", &MultiSensorScenario::accels, "List of AccelObs")
        .def("add_radar", &MultiSensorScenario::add_radar, nb::rv_policy::reference,
             "Append a RadarObs; returns the scenario, so calls can be chained.")
        .def("add_optics", &MultiSensorScenario::add_optics, nb::rv_policy::reference,
             "Append an OpticsObs; returns the scenario.")
        .def("add_position_fix", &MultiSensorScenario::add_position_fix, nb::rv_policy::reference,
             "Append a PositionFixObs; returns the scenario.")
        .def("add_accelerometer", &MultiSensorScenario::add_accelerometer, nb::rv_policy::reference,
             "Append an AccelObs; returns the scenario.");

    nb::class_<SensorMask>(m, "SensorMask",
        "Which sensors of a scenario an estimate uses: one on/off flag per sensor, in the "
        "scenario's order. Start from `SensorMask.all_on(scenario)` and switch groups off "
        "with the chaining `.with_*(False)` methods. A sensor missing from the mask counts "
        "as on.")
        .def(nb::init<>())
        .def_static("all_on", &SensorMask::all_on, nb::arg("scenario"),
            "Mask with every sensor of `scenario` switched on")
        .def_rw("radars", &SensorMask::radars, "One flag per radar (list of bool)")
        .def_rw("optics", &SensorMask::optics, "One flag per optical tracker (list of bool)")
        .def_rw("fixes", &SensorMask::fixes, "One flag per position-fix sensor (list of bool)")
        .def_rw("accels", &SensorMask::accels, "One flag per accelerometer (list of bool)")
        .def("with_radars", &SensorMask::with_radars, nb::rv_policy::reference,
             "Set every radar flag to `on`; returns the mask.")
        .def("with_optics", &SensorMask::with_optics, nb::rv_policy::reference,
             "Set every optical-tracker flag to `on`; returns the mask.")
        .def("with_fixes", &SensorMask::with_fixes, nb::rv_policy::reference,
             "Set every position-fix flag to `on`; returns the mask.")
        .def("with_accels", &SensorMask::with_accels, nb::rv_policy::reference,
             "Set every accelerometer flag to `on`; returns the mask.");

    // =========================================================================
    // Config
    // =========================================================================



    // =========================================================================
    // Result types
    // =========================================================================



    // =========================================================================
    // Entry points
    // =========================================================================


    nb::class_<EASmootherConfig>(m, "EASmootherConfig",
        "Settings for estimate_ea_smoother. Deliberately few: leave sigma_a and tau_s "
        "unset to have them chosen from the data (by NIS), and rate_budget at Auto to have "
        "the range-rate weight measured from the data.")
        .def(nb::init<>())
        .def_rw("sigma_a", &EASmootherConfig::sigma_a,
            "Expected EA size σ_a [m/s²]; None = chosen by NIS (closest to 1) from "
            "{1, 3, 10}. That grid suits reentry glides; for an ordinary orbit, values near "
            "1e-6 m/s² fit (see regime_aware_prior).")
        .def_rw("tau_s", &EASmootherConfig::tau_s,
            "How long the EA stays correlated, τ [s]; None = chosen by NIS from "
            "{10, 35, 100}. For an ordinary orbit about 600 s fits. Setting both sigma_a "
            "and tau_s skips the grid (9× faster).")
        .def_rw("rate_budget", &EASmootherConfig::rate_budget,
            "How radar range rate is weighted (EASmootherRateBudget; default Auto)")
        .def_rw("sigma_rate_manual", &EASmootherConfig::sigma_rate_manual,
            "Range-rate σ [m/s] used when rate_budget is Manual")
        .def_rw("max_dt_s", &EASmootherConfig::max_dt_s,
            "Largest integration step [s] between observations (fixed-step 4th-order "
            "Runge-Kutta; default 0.05)")
        .def_rw("jac_refresh", &EASmootherConfig::jac_refresh,
            "Recompute the gravity gradient (how gravity changes with position) every K "
            "steps; it changes slowly (default 16)")
        .def_rw("output_dt_s", &EASmootherConfig::output_dt_s,
            "Time spacing [s] of the returned EA table (default 0.5)")
        .def_rw("passes", &EASmootherConfig::passes,
            "Number of filter/smoother passes (default 1). 2 = run again with the first "
            "pass's EA added to the dynamics (an iterated extended Kalman smoother).")
        .def_rw("estimate_clock_bias", &EASmootherConfig::estimate_clock_bias,
            "Estimate a time-tag (clock) offset [s] for sensors whose clock is Shared or "
            "Own (default True). An unmodeled clock error is the largest measured risk "
            "with real data (in the radar test a 5 ms offset gave 316 m without this state, "
            "52 m with it); harmless when the clocks are right.")
        .def_rw("clock_bias_prior_s", &EASmootherConfig::clock_bias_prior_s,
            "Expected size of the clock offset [s] (default 0.02)")
        .def_rw("regime_aware_prior", &EASmootherConfig::regime_aware_prior,
            "Choose the sigma_a candidates from the data instead of the fixed {1, 3, 10} "
            "m/s² grid, which suits reentry glides and is about 7 orders of magnitude too "
            "large for an ordinary orbit. A first fit with the EA held at zero measures the "
            "position residual the dynamics cannot explain, and the candidates are placed "
            "around the acceleration scale it implies (see prior_anchor). Measured better "
            "in every test, but the reported uncertainty inside data gaps becomes too "
            "small, so it is off by default. Costs one extra fit. Ignored when sigma_a "
            "is set.")
        .def_rw("maneuver_adaptive_q", &EASmootherConfig::maneuver_adaptive_q,
            "Maneuver handling (default False): after an observation that fits much worse "
            "than expected, let the EA change faster for a while, so a sudden change (a "
            "thruster firing) is followed instead of smeared out.")
        .def_rw("verbose", &EASmootherConfig::verbose,
            "Print progress (grid values, NIS) to stdout");

    nb::class_<EASmootherResult>(m, "EASmootherResult",
        "Result of estimate_ea_smoother: the EA (in the inertial GCRS frame) and quality "
        "figures that need no true answer to compute (nis about 1 = the declared noise "
        "matches the data; ea_sigma = the EA's 1σ uncertainty).")
        .def(nb::init<>())
        .def_ro("empirical_accel", &EASmootherResult::empirical_accel,
            "Estimated EA (GCRS), tabulated every output_dt_s seconds, linear in between.")
        .def_ro("nis", &EASmootherResult::nis,
            "Normalized innovation squared per degree of freedom: about 1 when the declared "
            "noise matches the data; far from 1 means the noise levels or the EA prior are "
            "wrong.")
        .def_ro("recovered_bias", &EASmootherResult::recovered_bias,
            "Estimated accelerometer bias (ECEF, numpy (3,) [m/s²]), if accelerometers were "
            "used.")
        .def_ro("sigma_a_used", &EASmootherResult::sigma_a_used, "σ_a used [m/s²]")
        .def_ro("tau_used", &EASmootherResult::tau_used, "τ used [s]")
        .def_ro("sigma_rate_eff", &EASmootherResult::sigma_rate_eff,
            "Range-rate σ actually used [m/s] (0 = range rate ignored).")
        .def_ro("clock_biases", &EASmootherResult::clock_biases,
            "Estimated clock offset per clock: list of (label, seconds), labels 'shared', "
            "'radar0', 'optics1', ... The scale and sign of these values are not yet "
            "calibrated.")
        .def_ro("sigma_honored", &EASmootherResult::sigma_honored,
            "Residual RMS per channel at the final solution: list of (channel, value), e.g. "
            "'radar0_range', 'radar0_rate', 'fix0_pos', in each channel's units. This is "
            "the accuracy actually reached, to compare with the declared σ; a value far "
            "above it means the channel could not be fitted that well (wrong noise level, "
            "an unmodeled bias or clock error, or a model limit). Report only.")
        .def_ro("clock_bias_s", &EASmootherResult::clock_bias_s,
            "Clock offset [s]: the 'shared' one, else the first in clock_biases.")
        .def_ro("prior_anchor", &EASmootherResult::prior_anchor,
            "Acceleration scale 2Δ/T² [m/s²] from the zero-EA first fit (Δ: the position "
            "residual the dynamics alone could not explain; T: the arc length); 0 when "
            "regime_aware_prior is off.")
        .def_ro("warnings", &EASmootherResult::warnings,
            "Plain-language warnings (list of str; nothing is changed because of them). "
            "One fires when the largest gap between observations is longer than τ: inside "
            "such a gap the EA is unobserved and errors grow unseen (measured: a 40 s "
            "outage made the error 128× larger while NIS looked normal), so this warning is "
            "the only sign. Another fires when the automatic grid chose an edge value: a "
            "prompt to check that the defaults suit your kind of trajectory.")
        .def_ro("ea_sigma", &EASmootherResult::ea_sigma,
            "1σ uncertainty of the EA [m/s², GCRS axes] at each point of the returned "
            "table, from the smoother's covariance.")
        .def_ro("iterations_used", &EASmootherResult::iterations_used,
            "Number of iterations performed")
        .def_ro("error_message", &EASmootherResult::error_message,
            "Error description (empty on success); a filter failure is reported here")
        .def("succeeded", &EASmootherResult::succeeded,
             "True if the estimator ran without error")
        .def("has_correction", &EASmootherResult::has_correction,
             "True if succeeded() and the EA is not zero");

    m.def("estimate_ea_smoother",
          &estimate_ea_smoother,
          nb::arg("scenario"), nb::arg("mask"), nb::arg("fm_cfg"),
          nb::arg("cfg") = EASmootherConfig{},
          nb::call_guard<nb::gil_scoped_release>(),
          "Estimate an empirical acceleration from several sensors with the Gauss-Markov "
          "Rauch-Tung-Striebel smoother; the recommended estimator for sensor fusion "
          "(radars, optical trackers, position fixes, accelerometers with their bias), for "
          "radar alone and for noisy positions.\n"
          "\n"
          "Args: scenario (MultiSensorScenario); mask (SensorMask: which sensors to use); "
          "fm_cfg (TrajectoryConfig: the force model); cfg (EASmootherConfig).\n"
          "\n"
          "Returns an EASmootherResult with the EA in the inertial GCRS frame, its 1σ "
          "uncertainty and NIS. The EA's size and correlation time are chosen from the data "
          "by NIS, and the range-rate weight is measured from the data. Measured on the "
          "single-radar reentry-glide test scenario (with a position fix at impact): 12.4 m "
          "position error after re-propagation, better and much faster than the batch fits "
          "used before.\n"
          "\n"
          "Raises what Trajectory raises for an implausible start state "
          "(InputPlausibilityError / FrameError), and errors from the force model (e.g. "
          "missing data files).");

    // =========================================================================
    // EAProblemSpec front door + information bound
    // =========================================================================

    nb::enum_<EAMethod>(m, "EAMethod",
        "Which estimator estimate_ea(spec) ran.")
        .value("BSplinePosition", EAMethod::BSplinePosition,
               "estimate_ea_bspline on position samples (with Gauss-Newton refinement)")
        .value("BSplineVelocity", EAMethod::BSplineVelocity,
               "estimate_ea_bspline on velocity samples (with Gauss-Newton refinement)")
        .value("BSplineAcceleration", EAMethod::BSplineAcceleration,
               "estimate_ea_bspline on acceleration samples")
        .value("Smoother", EAMethod::Smoother, "estimate_ea_smoother on a MultiSensorScenario");

    nb::class_<EAProblemSpec>(m, "EAProblemSpec",
        "Input to estimate_ea: the observations, their declared noise and the force model. "
        "The library picks the estimator and then checks the declared noise against the "
        "fit. observations is one of PositionSamples, VelocitySamples, AccelSamples or "
        "MultiSensorScenario; the first three require noise_sigma > 0.")
        .def(nb::init<>())
        .def_rw("epoch0", &EAProblemSpec::epoch0,
            "Start time (Epoch); ignored for a MultiSensorScenario, which has its own")
        .def_rw("pos0_ecef", &EAProblemSpec::pos0_ecef,
            "Start position, Earth-fixed (ECEF), numpy (3,) [m]")
        .def_rw("vel0_ecef", &EAProblemSpec::vel0_ecef,
            "Start velocity relative to the rotating Earth, numpy (3,) [m/s]")
        .def_rw("observations", &EAProblemSpec::observations,
            "PositionSamples, VelocitySamples, AccelSamples or MultiSensorScenario")
        .def_rw("force_model", &EAProblemSpec::force_model,
            "TrajectoryConfig used in the fit")
        .def_rw("noise_sigma", &EAProblemSpec::noise_sigma,
            "Declared observation noise σ for position / velocity / acceleration samples, "
            "in their units (m, m/s, m/s²). Required (> 0) for them unless `bspline` is "
            "given; ignored for scenarios (each sensor carries its own).")
        .def_rw("outlier_prob", &EAProblemSpec::outlier_prob,
            "Optional outlier model for position / acceleration samples: a fraction p of "
            "samples have noise k·σ instead of σ. p > 0 removes likely outliers with "
            "reject_heavy_tails() before fitting. Measured with 5% of samples at 10σ: "
            "acceleration data 2.85 → 0.91 m, positions 0.60 → 0.37 m. Not applied to "
            "velocity samples (measured to make velocity fits far worse; a warning says "
            "so). Ignored for scenarios (declare it on the sensors).")
        .def_rw("outlier_scale", &EAProblemSpec::outlier_scale,
            "Outlier noise multiplier k (default 10); see outlier_prob.")
        .def_rw("bspline", &EAProblemSpec::bspline,
            "Optional EABSplineConfig to use instead of the defaults")
        .def_rw("smoother", &EAProblemSpec::smoother,
            "Optional EASmootherConfig to use instead of the defaults");

    m.def("reject_heavy_tails",
          [](const std::vector<std::pair<Epoch, Vector3d>>& samples,
             double sigma, double outlier_prob, double outlier_scale) {
              size_t nrej = 0;
              bool oob = false;
              auto out = calibration::reject_heavy_tails(
                  samples, sigma, outlier_prob, outlier_scale, &nrej, &oob);
              return std::make_tuple(std::move(out), nrej, oob);
          },
          nb::arg("samples"), nb::arg("sigma"),
          nb::arg("outlier_prob") = 0.05, nb::arg("outlier_scale") = 10.0,
          nb::call_guard<nb::gil_scoped_release>(),
          "Remove outliers from dense samples before fitting.\n"
          "\n"
          "samples: list of (Epoch, numpy (3,)); sigma: the usual noise σ in the data's "
          "units; outlier_prob p and outlier_scale k: a fraction p of samples are assumed "
          "to have noise k·σ. Each sample is compared with the median of the samples within "
          "±0.25 s and dropped when any component differs by more than the point at which "
          "it is more likely an outlier than not (about 3.3σ for the defaults, and nearly "
          "independent of p and k, so a rough guess is fine).\n"
          "\n"
          "Returns (filtered samples, number dropped, out_of_envelope). out_of_envelope = "
          "True means there are fewer than about 4 samples per second, too few to judge; "
          "the input is then returned unchanged. Outliers are dropped, not clipped: "
          "pulling them toward the local median was measured to corrupt position data "
          "(a 10.3 m error). Never raises.");

    nb::class_<EASpecResult>(m, "EASpecResult",
        "Result of estimate_ea(spec): the common result fields, which estimator ran and "
        "why, and the check of the declared noise against the fit.")
        .def_ro("empirical_accel", &EASpecResult::empirical_accel,
            "The estimated EA (EmpiricalAcceleration, GCRS)")
        .def_ro("method_used", &EASpecResult::method_used, "Which estimator ran (EAMethod)")
        .def_ro("rationale", &EASpecResult::rationale, "Why it was chosen (str)")
        .def_ro("nis", &EASpecResult::nis,
            "The smoother's NIS per degree of freedom (about 1 when the declared noise is "
            "right); 0 for the B-spline path")
        .def_ro("sigma_honored", &EASpecResult::sigma_honored,
            "Residual RMS reached per channel: list of (channel, value)")
        .def_ro("audit_warnings", &EASpecResult::audit_warnings,
            "Plain-language findings of the check (empty = nothing found): NIS out of "
            "range, residuals far above the declared noise, and so on. Warnings, not "
            "errors: the estimate is still returned.")
        .def_ro("bspline_detail", &EASpecResult::bspline_detail,
            "Full EAResult if the B-spline estimator ran, else None")
        .def_ro("smoother_detail", &EASpecResult::smoother_detail,
            "Full EASmootherResult if the smoother ran, else None")
        .def_ro("error_message", &EASpecResult::error_message,
            "Error description (empty on success), e.g. a missing noise_sigma")
        .def("succeeded", &EASpecResult::succeeded,
             "True if the estimator ran without error")
        .def("has_correction", &EASpecResult::has_correction,
             "True if succeeded() and the EA is not zero");

    m.def("estimate_ea", &calibration::estimate_ea, nb::arg("spec"),
          nb::call_guard<nb::gil_scoped_release>(),
          "Estimate an empirical acceleration from an EAProblemSpec, letting the library "
          "choose the estimator: position / velocity / acceleration samples go to "
          "estimate_ea_bspline (with Gauss-Newton refinement), a MultiSensorScenario to "
          "estimate_ea_smoother. The declared noise is then checked against the fit "
          "(result.audit_warnings). Returns an EASpecResult. A bad spec (e.g. no "
          "noise_sigma) is reported in error_message, not raised; errors from the "
          "estimator itself (e.g. missing data files) are raised.");

    nb::class_<EAInformationBoundOptions>(m, "EAInformationBoundOptions",
        "Settings for information_bound.")
        .def(nb::init<>())
        .def_rw("knot_spacing_s", &EAInformationBoundOptions::knot_spacing_s,
            "Spacing of the EA time points [s] (default 8.0). A finer spacing raises the "
            "bound (more unknowns, same data), so compare sensor suites at one spacing.")
        .def_rw("lambda_", &EAInformationBoundOptions::lambda,
            "Prior weight on EA changes between neighboring time points (smoothness; "
            "default 1.0)")
        .def_rw("lambda_amp", &EAInformationBoundOptions::lambda_amp,
            "Prior weight on the EA's size (default 1e-2)");

    nb::class_<EAInformationBound>(m, "EAInformationBound",
        "Result of information_bound: the best position accuracy any estimator could "
        "reach with these sensors. Accelerometers are not included.")
        .def_ro("posterior_bound_m", &EAInformationBound::posterior_bound_m,
            "RMS position bound [m] with the smoothness prior (lambda_, lambda_amp): what "
            "a regularized estimator can reach")
        .def_ro("crb_bound_m", &EAInformationBound::crb_bound_m,
            "RMS position bound [m] without a prior (the Cramér-Rao bound), over the "
            "directions the data can see")
        .def_ro("n_params", &EAInformationBound::n_params,
            "Number of unknowns (3 per EA time point)")
        .def_ro("n_obs_rows", &EAInformationBound::n_obs_rows,
            "Number of scalar observations used")
        .def_ro("error_message", &EAInformationBound::error_message,
            "Why the bound was not computed (empty = computed)")
        .def("ok", &EAInformationBound::ok, "True when the bound was computed");

    m.def("information_bound",
          [](const MultiSensorScenario& sc, const SensorMask& mask,
             const TrajectoryConfig& fm, const EAInformationBoundOptions& o) {
              return calibration::information_bound(sc, mask, fm, o);
          },
          nb::arg("scenario"), nb::arg("mask"), nb::arg("fm_cfg"),
          nb::arg("opts") = EAInformationBoundOptions{},
          nb::call_guard<nb::gil_scoped_release>(),
          "The best position accuracy (after re-propagation) that any estimator could reach "
          "with the sensors of `scenario` selected by `mask`, propagating with `fm_cfg`; "
          "computed without running an estimator. Use it to design a sensor suite before "
          "tuning an estimator: e.g. on the single-radar reentry-glide test scenario, "
          "adding a 50 µrad camera moves the limit from about 10-17 m to about 1.3 m "
          "(test_ea_problem_spec). Returns an EAInformationBound; failures (no radar, fix "
          "or camera samples; a failed propagation) are reported in its error_message.");







}
