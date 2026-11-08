// =============================================================================
// Trajectory and TrajectoryInterpolator Python Bindings
//
// Frames used below: the ITRF (International Terrestrial Reference Frame) is
// Earth-fixed and rotates with the Earth (also called ECEF, Earth-centered
// Earth-fixed); the GCRS (Geocentric Celestial Reference System) is inertial,
// i.e. it does not rotate, and is the frame the equations of motion are
// integrated in.
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/state_transition.hpp>
#include <ballistic/ephemeris/spk_ephemeris.hpp>  // Needed for get_default_ephemeris_provider()

namespace nb = nanobind;

void bind_trajectory(nb::module_& m) {
    using namespace ballistic;

    // =========================================================================
    // Events (events.hpp): built-in event functions for propagate_until
    // =========================================================================
    auto ev = m.def_submodule("events",
        "Conditions that Trajectory.propagate_until watches for and locates in time: an "
        "altitude crossing, ground impact, perigee (lowest point of the orbit), apogee "
        "(highest point) or a given epoch.");
    nb::enum_<events::Direction>(ev, "Direction")
        .value("Rising", events::Direction::Rising,
               "The event function g goes from negative to zero or positive (e.g. climbing "
               "through an altitude)")
        .value("Falling", events::Direction::Falling,
               "The event function g goes from positive to zero or negative (e.g. descending "
               "through an altitude)")
        .value("Either", events::Direction::Either, "Either direction");
    nb::class_<events::Event>(ev, "Event",
        "A condition to watch for during propagation. Build one with altitude_crossing, "
        "ground_impact, perigee, apogee or at_epoch.")
        .def_ro("name", &events::Event::name, "Name of the event (str), used in the log")
        .def_ro("direction", &events::Event::direction,
                "Which sign change counts (Direction)")
        .def_ro("terminal", &events::Event::terminal,
                "True: the propagation stops at this event. False: it is only logged.")
        .def("__repr__", [](const events::Event& e) {
            return "Event('" + e.name + "', terminal=" + (e.terminal ? "True" : "False") + ")";
        });
    nb::class_<events::Occurrence>(ev, "Occurrence",
        "One event found during propagation, with the Earth-fixed (ITRF) state at that moment.")
        .def_ro("name", &events::Occurrence::name, "Name of the event that fired (str)")
        .def_ro("epoch", &events::Occurrence::epoch, "When it happened (Epoch)")
        .def_ro("position_ecef", &events::Occurrence::position_ecef,
                "Earth-fixed position there, numpy (3,) [m]")
        .def_ro("velocity_ecef", &events::Occurrence::velocity_ecef,
                "Velocity relative to the rotating Earth there, numpy (3,) [m/s]")
        .def_ro("terminal", &events::Occurrence::terminal,
                "True if this event ended the propagation");
    ev.def("altitude_crossing", &events::altitude_crossing, nb::arg("altitude_m"),
           nb::arg("direction") = events::Direction::Either, nb::arg("terminal") = true,
           "Event: the height above the WGS84 ellipsoid (geodetic altitude) crosses "
           "altitude_m [m]. direction picks climbing (Rising), descending (Falling) or "
           "either; terminal=True stops the propagation there.");
    ev.def("ground_impact", &events::ground_impact,
           "Event: descending through the surface of the WGS84 ellipsoid (terminal). The geoid "
           "(up to ~100 m from the ellipsoid) and terrain (up to ~9 km) are not modeled. Armed "
           "by default in "
           "propagate_until.");
    ev.def("perigee", &events::perigee, nb::arg("terminal") = false,
           "Event: perigee, the lowest point of the orbit (the radial velocity changes sign "
           "from negative to positive). Logged only unless terminal=True.");
    ev.def("apogee", &events::apogee, nb::arg("terminal") = false,
           "Event: apogee, the highest point of the orbit (the radial velocity changes sign "
           "from positive to negative). Logged only unless terminal=True.");
    ev.def("at_epoch", &events::at_epoch, nb::arg("epoch"), nb::arg("terminal") = false,
           "Event: a given time (Epoch), to log the state at that moment during a "
           "propagation.");

    nb::class_<PropagateUntilResult>(m, "PropagateUntilResult",
        "Result of Trajectory.propagate_until: where it ended, why, and every event found "
        "on the way.")
        .def_ro("epoch", &PropagateUntilResult::epoch, "Where the propagation ended (Epoch)")
        .def_ro("state_ecef", &PropagateUntilResult::state_ecef,
                "The Earth-fixed (ITRF) state there")
        .def_ro("stopped_by", &PropagateUntilResult::stopped_by,
                "The terminal event that stopped it (Occurrence), or None if the target "
                "time was reached")
        .def_ro("log", &PropagateUntilResult::log,
                "Every event found, in time order (list of Occurrence)")
        .def("reached_target", &PropagateUntilResult::reached_target,
             "True when the propagation reached its target time (no terminal event fired).");

    // =========================================================================
    // State (initial_state.hpp): a start state with its frame named, checked
    // =========================================================================
    auto opts = [](bool sub, bool frame) { return PlausibilityOptions{sub, frame}; };
    nb::class_<InitialState>(m, "State", R"doc(
Where a trajectory starts: epoch, position, velocity, and the coordinate frame
they are given in.

Frames: the ITRF is Earth-fixed (it rotates with the Earth; velocities are
measured relative to the rotating ground). The GCRS is inertial (it does not
rotate). The same position has different velocities in the two frames, and
mixing them up is a common, silent mistake, so the frame must be named.

Build a State with one of the named constructors. Each one checks the state
and raises with the fix when something looks wrong:

    State.from_itrf(epoch, r_m, v_m_s, realization=None)  # Earth-fixed velocity
    State.from_gcrs(epoch, r_m, v_m_s)                    # inertial (GCRS)
    State.from_j2000(epoch, r_m, v_m_s)                   # inertial, J2000 axes
    State.from_keplerian(epoch, a_m=..., e=..., i_deg=..., raan_deg=..., argp_deg=..., true_anomaly_deg=...)
    State.from_teme(epoch, r_m, v_m_s)                    # the frame SGP4 outputs
    State.from_tle(line1, line2, epoch=None)              # from a two-line element set (TLE)
    State.from_sp3(sp3_file, satellite, epoch)            # from a precise-orbit (SP3) file

Positions are in meters, velocities in m/s (3-element sequences or numpy
arrays).

Raises:
    ballistic.errors.InputPlausibilityError: non-finite numbers, a position
        that looks like kilometers instead of meters (|r| < 100 km) or is
        beyond 1e6 km, a position more than 1 km below the ground, or an orbit
        whose lowest point is below 100 km. A reentry, ballistic or launch
        state is legitimately below that: pass allow_suborbital=True.
    ballistic.errors.FrameError: the numbers only make sense in the other frame
        (an inertial velocity given as Earth-fixed, or the reverse). Pass
        check_frame=False to skip this check.
)doc")
        .def_static("from_itrf",
            [=](const Epoch& t, const Vector3d& r, const Vector3d& v,
                std::optional<coords::ITRFRealization> real, bool sub, bool fr) {
                return InitialState::from_itrf(t, r, v, real, opts(sub, fr));
            }, nb::arg("epoch"), nb::arg("r_m"), nb::arg("v_m_s"),
            nb::arg("realization").none() = nb::none(), nb::kw_only(),
            nb::arg("allow_suborbital") = false, nb::arg("check_frame") = true,
            "Earth-fixed (ITRF) start state: position [m] and velocity relative to the "
            "rotating Earth [m/s]. realization: which ITRF version the coordinates are in "
            "(an ITRFRealization; None = use the trajectory config's). allow_suborbital: "
            "accept an orbit that dips below 100 km. check_frame: run the Earth-fixed vs "
            "inertial mix-up check. Raises InputPlausibilityError / FrameError (see State).")
        .def_static("from_gcrs",
            [=](const Epoch& t, const Vector3d& r, const Vector3d& v, bool sub, bool fr) {
                return InitialState::from_gcrs(t, r, v, opts(sub, fr));
            }, nb::arg("epoch"), nb::arg("r_m"), nb::arg("v_m_s"), nb::kw_only(),
            nb::arg("allow_suborbital") = false, nb::arg("check_frame") = true,
            "Inertial (GCRS) start state: position [m] and velocity [m/s]. Raises "
            "InputPlausibilityError / FrameError (see State).")
        .def_static("from_j2000",
            [=](const Epoch& t, const Vector3d& r, const Vector3d& v, bool sub, bool fr) {
                return InitialState::from_j2000(t, r, v, opts(sub, fr));
            }, nb::arg("epoch"), nb::arg("r_m"), nb::arg("v_m_s"), nb::kw_only(),
            nb::arg("allow_suborbital") = false, nb::arg("check_frame") = true,
            "Inertial start state on the mean equator and equinox of J2000.0 (EME2000), "
            "position [m] and velocity [m/s]. Rotated to the GCRS by the small fixed IAU 2006 "
            "frame-bias rotation (up to about 0.8 m at low Earth orbit). Raises "
            "InputPlausibilityError / FrameError (see State).")
        .def_static("from_keplerian",
            [=](const Epoch& t, double a, double e, double i, double raan, double argp,
                double nu, bool sub, bool fr) {
                constexpr double d = 3.14159265358979323846 / 180.0;
                return InitialState::from_keplerian(
                    t, KeplerianElements{a, e, i * d, raan * d, argp * d, nu * d}, opts(sub, fr));
            }, nb::arg("epoch"), nb::kw_only(), nb::arg("a_m"), nb::arg("e"), nb::arg("i_deg"),
            nb::arg("raan_deg"), nb::arg("argp_deg"), nb::arg("true_anomaly_deg"),
            nb::arg("allow_suborbital") = false, nb::arg("check_frame") = true,
            "Start state from classical (osculating, two-body) orbital elements in the GCRS, "
            "using the WGS84 Earth gravitational constant GM. a_m: semi-major axis [m]; e: "
            "eccentricity; i_deg: inclination; raan_deg: right ascension of the ascending "
            "node; argp_deg: argument of perigee; true_anomaly_deg: true anomaly. Angles in "
            "degrees. Raises InputPlausibilityError (see State).")
        .def_static("from_teme",
            [=](const Epoch& t, const Vector3d& r, const Vector3d& v, bool sub, bool fr) {
                return InitialState::from_teme(t, r, v, opts(sub, fr));
            }, nb::arg("epoch"), nb::arg("r_m"), nb::arg("v_m_s"), nb::kw_only(),
            nb::arg("allow_suborbital") = false, nb::arg("check_frame") = true,
            "Start state in TEME (True Equator, Mean Equinox: the frame the SGP4 propagator "
            "outputs), position [m] and velocity [m/s]. Converted to the ITRF with the 1982 "
            "Greenwich mean sidereal time (GMST) and polar motion. Raises "
            "InputPlausibilityError / FrameError (see State).")
        .def_static("_from_tle",
            [=](const sgp4::Tle& tle, std::optional<Epoch> at, bool sub, bool fr) {
                return InitialState::from_tle(tle, at, opts(sub, fr));
            }, nb::arg("tle"), nb::arg("epoch").none() = nb::none(), nb::kw_only(),
            nb::arg("allow_suborbital") = false, nb::arg("check_frame") = true,
            "Internal; use State.from_tle. Evaluates SGP4 for a parsed TLE at its own epoch "
            "(epoch=None) or at `epoch`.")
        .def_prop_ro("epoch", &InitialState::epoch, "The start time (Epoch)")
        .def_prop_ro("frame", [](const InitialState& s) { return std::string(state_frame_name(s.frame())); },
            "Name of the frame position and velocity are given in (str)")
        .def_prop_ro("position", &InitialState::position,
            "Position as given, in the frame named by .frame; numpy (3,) [m]")
        .def_prop_ro("velocity", &InitialState::velocity,
            "Velocity as given, in the frame named by .frame; numpy (3,) [m/s]")
        .def_prop_ro("realization", &InitialState::realization,
            "The ITRF version of an Earth-fixed state, or None if not stated / not "
            "Earth-fixed")
        .def("itrf_state", [](const InitialState& s, coords::ITRFRealization itrf) {
                const auto sv = s.itrf_state(itrf);
                return std::make_pair(sv.position, sv.velocity);
            }, nb::arg("itrf") = coords::ITRFRealization::ITRF2020,
            "The state converted to Earth-fixed coordinates in ITRF version `itrf`: a tuple "
            "(position [m], velocity relative to the rotating Earth [m/s]), each numpy (3,).")
        .def("gcrs_state", [](const InitialState& s, coords::ITRFRealization itrf) {
                const auto sv = s.gcrs_state(itrf);
                return std::make_pair(sv.position, sv.velocity);
            }, nb::arg("itrf") = coords::ITRFRealization::ITRF2020,
            "The state converted to the inertial GCRS: a tuple (position [m], velocity "
            "[m/s]), each numpy (3,). An Earth-fixed state is read in its own ITRF version "
            "if it has one, otherwise in `itrf`.")
        .def("__repr__", [](const InitialState& s) {
            std::string r = std::string("State(") + state_frame_name(s.frame());
            if (s.realization()) r += std::string(" ") + itrf_realization_name(*s.realization());
            char b[160];
            std::snprintf(b, sizeof b, ", r=[%.3f, %.3f, %.3f] m, v=[%.6f, %.6f, %.6f] m/s)",
                          s.position().x(), s.position().y(), s.position().z(),
                          s.velocity().x(), s.velocity().y(), s.velocity().z());
            return r + b;
        });

    // =========================================================================
    // Trajectory Class
    // =========================================================================

    nb::class_<Trajectory>(m, "Trajectory",
        R"doc(
Trajectory - the orbit propagator: steps a satellite's state forward (or
backward) in time and keeps every step in a history.

Two numerical integrators are available:
- RK45 (default): Dormand-Prince adaptive-step Runge-Kutta; most accurate for
  predictions of hours.
- Yoshida (6th or 8th order): symplectic, fixed-step integrators that keep
  the orbit's energy from drifting; better for runs of many hours to days.

Forces modeled:
- Earth's gravity field (EGM2008 model, embedded up to degree 360)
- Sun, Moon and Jupiter gravity (third-body perturbations)
- Earth rotation and orientation (IAU 2006/2000A precession-nutation)
- General-relativity corrections (Schwarzschild and Lense-Thirring)
- Optional: solar radiation pressure, atmospheric drag, ocean tides

Position error one hour after the start, measured against precise orbit
products (README "Accuracy" table):
- Low Earth orbit: about 0.22 m (Sentinel satellites, with radiation
  pressure, drag and ocean tides: ballistic.recommended_config)
- Medium Earth orbit: 0.15 m (LAGEOS / Etalon laser-ranging satellites);
  0.67 m (GPS, without radiation pressure)
- Inclined geosynchronous orbit: 0.74 m (QZSS, without radiation pressure)
- Geostationary orbit: 0.80 m (BeiDou, without radiation pressure)

Example:
    import ballistic
    import numpy as np

    # Create epoch and initial state (the frame is named and checked)
    epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    # Earth-fixed (ITRF): circular 7669 m/s minus Earth rotation 494 m/s
    state = ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0])

    # Create trajectory with default config
    traj = ballistic._core.Trajectory(state)

    # Step-based propagation
    while traj.epoch() - epoch < 3600.0:
        result = traj.step()

    # Access history
    for i in range(traj.history_size()):
        entry = traj.history_at(i)
        print(f"t={entry.epoch}, pos={entry.position}")
)doc")
        .def(nb::init<const InitialState&, const TrajectoryConfig&>(),
            nb::arg("state"), nb::arg("config") = TrajectoryConfig{},
            "Build a trajectory from a State (frame named and checked) and an optional "
            "TrajectoryConfig. The recommended form.")
        // Constructor
        .def(nb::init<const Epoch&, const Vector3d&, const Vector3d&, const TrajectoryConfig&>(),
            nb::arg("epoch"), nb::arg("position"), nb::arg("velocity"),
            nb::arg("config") = TrajectoryConfig{},
            R"doc(
Build a trajectory from an Earth-fixed (ITRF) state. Only the units are
checked here; prefer Trajectory(State.from_itrf(...)), which also checks the
orbit and the frame.

Args:
    epoch: Start time (Epoch)
    position: Start position, Earth-fixed (ITRF), numpy (3,) [m]
    velocity: Start velocity relative to the rotating Earth, numpy (3,) [m/s]
    config: Optional TrajectoryConfig (the library defaults if omitted)

Raises:
    ballistic.errors.InputPlausibilityError: the position looks like
        kilometers rather than meters, or is not finite.
)doc")

        .def("propagate_until", &Trajectory::propagate_until, nb::arg("target"),
            nb::arg("events") = std::vector<events::Event>{}, nb::arg("stop_at_ground") = true,
            nb::call_guard<nb::gil_scoped_release>(),
            R"doc(
Step forward to the `target` Epoch, watching for `events` (a list of
ballistic.events.Event). When an event happens inside a step, its time is
located to 1e-4 s by re-integrating that step, so the event state is as
accurate as the integrator. Propagation ends at the first terminal event or
at the target; every event found is logged. With stop_at_ground=True (the
default) the ground-impact event is added, so a reentering object stops on
the WGS84 ellipsoid instead of integrating into the Earth. Steps are
recorded in the history as with step().

Returns:
    PropagateUntilResult (epoch, state_ecef, stopped_by, log, reached_target())
)doc")

        // Propagation methods
        .def("propagate", &Trajectory::propagate, nb::arg("target"),
            nb::call_guard<nb::gil_scoped_release>(),
            R"doc(
Propagate to the target epoch without recording history (use step() or
propagate_until to keep a history).

Args:
    target: Epoch to propagate to

Returns:
    PropagationResult with position, velocity and acceleration, Earth-fixed
    (ITRF)
)doc")

        .def("step", &Trajectory::step,
            nb::call_guard<nb::gil_scoped_release>(),
            R"doc(
Take one integrator step forward in time and append the new state to the
history.

RK45 chooses the step size from its estimate of the local error; the
symplectic integrators use the fixed step size from the config.

Returns:
    StepResult with the new state and step information
)doc")

        .def("step_backward", &Trajectory::step_backward,
            nb::call_guard<nb::gil_scoped_release>(),
            R"doc(
Take one integrator step backward in time from the earliest history entry,
and insert the new, earlier state at the front of the history. The current
(latest) state is unchanged.

Returns:
    StepResult with the new (earlier) state
)doc")

        .def("rewind", &Trajectory::rewind,
            nb::arg("count") = 1,
            R"doc(
Go back `count` entries in the history: the trajectory returns to that
earlier state and the later entries are discarded. Nothing is propagated.

Args:
    count: Number of steps to go back (default 1)

Returns:
    bool: True if successful, False if already at the initial state

Raises:
    IndexError: count is larger than the number of steps in the history.
)doc")

        // State access
        .def("epoch", &Trajectory::epoch,
            "Current time of the trajectory (Epoch)")
        .def("state_ecef", &Trajectory::state_ecef,
            "Current state in Earth-fixed (ITRF) coordinates (positions in m, velocities "
            "in m/s)")
        .def("state_gcrs", &Trajectory::state_gcrs,
            "Current state in the inertial GCRS, the frame the integration runs in")

        // History access
        .def("history_size", &Trajectory::history_size,
            "Number of entries in the history, including the initial state")
        .def("history_at", &Trajectory::history_at, nb::arg("index"),
            "History entry number `index` (0 = initial state)")
        .def("history", &Trajectory::history,
            "The whole history as a list of entries, oldest first")
        .def("clear_history", &Trajectory::clear_history,
            "Clear the history, keeping only the current state as the new initial entry")

        // Reset
        .def("reset", &Trajectory::reset,
            nb::arg("epoch"), nb::arg("position"), nb::arg("velocity"),
            "Restart from a new Earth-fixed (ITRF) state: epoch (Epoch), position [m], "
            "velocity relative to the rotating Earth [m/s]. Clears the history.")

        // Step size control
        .def("current_step_size", &Trajectory::current_step_size,
            "Step size the integrator will try next [s]")
        .def("set_step_size_hint", &Trajectory::set_step_size_hint,
            nb::arg("dt"),
            "Suggest the next step size dt [s]. RK45 may still adjust it to meet its error "
            "tolerance; a symplectic integrator uses it as its new fixed step.")

        // Configuration access
        .def("config", &Trajectory::config, nb::rv_policy::reference,
            "The TrajectoryConfig in use (read-only)")
        .def("is_symplectic", &Trajectory::is_symplectic,
            "True if a symplectic (Yoshida) integrator is used, False for RK45")
        .def("energy_drift", &Trajectory::energy_drift,
            "Relative change in orbital energy since the start, (E - E0) / |E0|. Only "
            "tracked by the symplectic integrators; RK45 returns 0.0.")
        .def("warnings", &Trajectory::warnings,
            "Notes from propagation that do not stop it, each recorded once (list of str): "
            "drag entering forecast (not yet measured) space weather, running past a "
            "tabulated drag environment (the atmosphere is then held frozen), past the time "
            "span of an empirical acceleration, or outside the loaded Earth orientation "
            "table. The ballistic.Trajectory wrapper also issues them as warnings.");

    // =========================================================================
    // TrajectoryInterpolator Class
    // =========================================================================

    nb::class_<TrajectoryInterpolator>(m, "TrajectoryInterpolator",
        R"doc(
TrajectoryInterpolator - the state at any time between a Trajectory's steps.

Fits a quintic (5th-degree) Hermite polynomial between neighboring history
entries, using the position, velocity and acceleration stored at each. The
interpolation error shrinks as the 6th power of the step size; it is below a
millimeter for typical 10-30 s steps.

Features:
- Polynomial coefficients are cached, so repeated queries are cheap
- A query outside the history propagates the trajectory to cover it
  (within InterpolatorConfig.max_propagation_time, default 3600 s)
- Output in Earth-fixed (ECEF/ITRF) or inertial (GCRS) coordinates

Example:
    # Build trajectory history
    traj = ballistic._core.Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
    while traj.epoch() - epoch < 3600.0:
        traj.step()

    # Create interpolator
    interp = ballistic.TrajectoryInterpolator(traj)

    # Query at any time within range
    state = interp.get_state(epoch + 1800.0)
    if state:
        print(f"Position: {state.position}")
        print(f"Velocity: {state.velocity}")

    # Altitude-based propagation
    result = interp.propagate_to_altitude(
        300_000,  # target altitude in meters
        ballistic.AltitudeCondition.Below,
        ballistic.PropagationDirection.Forward,
        86400.0)  # search up to one day
)doc")
        // Constructor
        // keep_alive<1, 2>: The TrajectoryInterpolator (arg 1 = self) keeps the Trajectory (arg 2) alive
        .def(nb::init<Trajectory&, const InterpolatorConfig&>(),
            nb::arg("trajectory"), nb::arg("config") = InterpolatorConfig{},
            nb::keep_alive<1, 2>(),
            R"doc(
Wrap a Trajectory for interpolation.

Args:
    trajectory: Trajectory to interpolate (must have at least one history entry)
    config: Optional InterpolatorConfig

Note:
    The interpolator holds a reference to the trajectory and keeps it alive
    as long as the interpolator exists. Out-of-range queries add steps to
    that trajectory's history.
)doc")

        // Single-point queries
        .def("get_state", &TrajectoryInterpolator::get_state, nb::arg("epoch"),
            R"doc(
The interpolated state at a given time.

Inside the history range the state is interpolated. Outside it, if automatic
propagation is allowed (InterpolatorConfig.allow_propagation, default True)
and the gap is within max_propagation_time, the history is extended first.

Args:
    epoch: Query time (Epoch)

Returns:
    InterpolatedState, or None if the time cannot be reached (propagation
    disabled or too far away)
)doc")

        .def("position", &TrajectoryInterpolator::position, nb::arg("epoch"),
            "Interpolated position at `epoch` [m], numpy (3,), or None (see get_state)")
        .def("velocity", &TrajectoryInterpolator::velocity, nb::arg("epoch"),
            "Interpolated velocity at `epoch` [m/s], numpy (3,), or None (see get_state)")
        .def("acceleration", &TrajectoryInterpolator::acceleration, nb::arg("epoch"),
            "Interpolated acceleration at `epoch` [m/s^2], numpy (3,), or None (see "
            "get_state)")

        // Batch queries
        .def("get_states",
            [](TrajectoryInterpolator& self, const std::vector<Epoch>& times) {
                return self.get_states(std::span<const Epoch>(times));
            },
            nb::arg("times"),
            R"doc(
Interpolated states at several times.

Faster than one get_state call per time because:
1. The trajectory is propagated once to cover the whole time range
2. The times are visited in sorted order, reusing cached coefficients

Args:
    times: List of Epoch

Returns:
    List of InterpolatedState in the order given (None where a time cannot be
    reached)
)doc")

        // Range information
        .def("history_start", &TrajectoryInterpolator::history_start,
            "Earliest time in the history (Epoch)")
        .def("history_end", &TrajectoryInterpolator::history_end,
            "Latest time in the history (Epoch)")
        .def("covers", &TrajectoryInterpolator::covers, nb::arg("epoch"),
            "True if `epoch` lies within the current history range")
        .def("history_size", &TrajectoryInterpolator::history_size,
            "Number of history entries")

        // History extension
        .def("extend_to", &TrajectoryInterpolator::extend_to, nb::arg("target"),
            "Propagate forward so the history covers the `target` Epoch")

        .def("extend_backward_to", &TrajectoryInterpolator::extend_backward_to, nb::arg("target"),
            "Propagate backward so the history covers the `target` Epoch")

        // Altitude-based propagation
        .def("propagate_to_altitude", &TrajectoryInterpolator::propagate_to_altitude,
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("target_altitude_m"),
            nb::arg("condition"),
            nb::arg("direction") = PropagationDirection::Forward,
            nb::arg("max_duration_s") = 7.0 * 86400.0,
            R"doc(
Propagate until the height above the WGS84 ellipsoid (geodetic altitude)
goes above or below a threshold.

Args:
    target_altitude_m: Threshold geodetic altitude [m]
    condition: AltitudeCondition.Above or AltitudeCondition.Below
    direction: PropagationDirection.Forward or Backward
    max_duration_s: How far to search [s] (default 7 days); condition_met is
        False if the altitude is not reached within it.

Forward, the history ends exactly at the crossing (located to 1e-4 s), and
ground impact also stops the search. Backward stops on the first step past
the crossing.

Returns:
    PropagateToAltitudeResult with condition_met, steps_taken, and altitudes

Note:
    If the starting point already satisfies the condition, returns
    immediately with steps_taken=0.
)doc")

        // Cache management
        .def("clear_cache", &TrajectoryInterpolator::clear_cache,
            "Drop the cached polynomial coefficients")
        .def("invalidate", &TrajectoryInterpolator::invalidate,
            "Drop the cache; call this when the trajectory's history was changed by "
            "something other than this interpolator")

        // Configuration
        .def("config", &TrajectoryInterpolator::config, nb::rv_policy::reference,
            "The InterpolatorConfig in use (read-only)")
        .def("set_output_frame", &TrajectoryInterpolator::set_output_frame,
            nb::arg("frame"),
            "Choose the coordinate frame of returned states (InterpolationFrame: ECEF or "
            "GCRS)")
        .def("set_itrf_realization", &TrajectoryInterpolator::set_itrf_realization,
            nb::arg("itrf"),
            "Choose the ITRF version used when converting between frames")
        .def("set_allow_propagation", &TrajectoryInterpolator::set_allow_propagation,
            nb::arg("allow"),
            "Allow (True) or forbid (False) propagating to answer queries outside the "
            "history")
        .def("set_max_propagation_time", &TrajectoryInterpolator::set_max_propagation_time,
            nb::arg("max_seconds"),
            "Furthest beyond the history [s] that a query may propagate");

    // =========================================================================
    // State transition matrix and covariance (state_transition.hpp)
    // =========================================================================

    nb::class_<StmOptions>(m, "StmOptions",
        "Settings for StateTransition: finite-difference offsets for the force terms without "
        "an analytic Jacobian, and the largest Runge-Kutta substep.")
        .def(nb::init<>())
        .def_rw("fd_dx", &StmOptions::fd_dx,
                "Finite-difference position offset [m] (drag's position columns)")
        .def_rw("fd_dv", &StmOptions::fd_dv,
                "Finite-difference velocity offset [m/s] (a velocity-dependent drag coefficient)")
        .def_rw("max_substep_s", &StmOptions::max_substep_s,
                "Largest Runge-Kutta substep [s] for the transition matrix");

    nb::class_<StateTransition>(m, "StateTransition", R"doc(
State transition matrix Phi(t, t0) and covariance along a propagated Trajectory.

Phi maps a small change of the state at the first recorded epoch t0 into the
change at t: dx(t) = Phi dx(t0), with x = [r (m); v (m/s)] in the inertial GCRS
frame. It is integrated along the recorded steps (dPhi/dt = A(t) Phi), so the
trajectory must have been propagated with step() or propagate_until() first.
covariance(t, P_0) = Phi P_0 Phi^T (no process noise, no consider parameters).

Caveats: radiation pressure uses a hard Earth shadow, so an arc crossing the
shadow boundary gets a Phi that misses the jump in sunlight; drag's position
derivatives come from finite differences. Most users reach this through
ballistic.Trajectory.stm / covariance.
)doc")
        .def(nb::init<const Trajectory&, StmOptions>(),
            nb::arg("trajectory"), nb::arg("options") = StmOptions{},
            nb::keep_alive<1, 2>(),
            nb::call_guard<nb::gil_scoped_release>(),
            "Integrate Phi over the trajectory's recorded history (keeps the trajectory alive). "
            "Raises ConfigError with fewer than two recorded states.")
        .def("stm", &StateTransition::stm, nb::arg("epoch"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Phi(epoch, epoch0()) as a (6, 6) array, GCRS, [r (m); v (m/s)]. Raises "
            "DataCoverageError outside [epoch0(), epoch_end()].")
        .def("covariance", &StateTransition::covariance, nb::arg("epoch"), nb::arg("p0"),
            nb::call_guard<nb::gil_scoped_release>(),
            "Phi P_0 Phi^T as a (6, 6) array: the start covariance P_0 (symmetric, GCRS, "
            "[r (m); v (m/s)]) carried to `epoch`. Raises InputPlausibilityError for a "
            "non-symmetric or non-finite P_0.")
        .def("epoch0", &StateTransition::epoch0,
            "First recorded epoch: Phi is relative to it")
        .def("epoch_end", &StateTransition::epoch_end, "Last recorded epoch covered")
        .def("node_count", &StateTransition::node_count,
            "Number of recorded steps Phi is stored at");
}
