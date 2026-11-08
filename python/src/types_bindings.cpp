// =============================================================================
// Type Bindings - Enums and Configuration Classes
//
// Frames: the GCRS (Geocentric Celestial Reference System) is inertial
// (non-rotating); the ITRF (International Terrestrial Reference Frame), also
// called ECEF (Earth-centered, Earth-fixed), rotates with the Earth.
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <ballistic/core/types.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/coordinates/earth_rotation.hpp>
#include <ballistic/coordinates/velocity_check.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/propagators/symplectic.hpp>

namespace nb = nanobind;

void bind_types(nb::module_& m) {
    using namespace ballistic;

    // =========================================================================
    // Enums
    // =========================================================================

    // PropagationFrame enum.
    // GCRS is the only supported value: sub-meter accuracy needs the full
    // IAU 2006/2000A Earth-orientation model, so no simpler inertial frame
    // (Earth rotation angle only) is offered.
    nb::enum_<PropagationFrame>(m, "PropagationFrame",
        "Frame the equations of motion are integrated in.")
        .value("GCRS", PropagationFrame::GCRS,
            "Geocentric Celestial Reference System: inertial, Earth-centered (IERS 2010 "
            "conventions). The only choice.");

    // IntegratorType enum
    nb::enum_<IntegratorType>(m, "IntegratorType",
        "Numerical integrator that steps the equations of motion.")
        .value("RK45", IntegratorType::RK45,
            "Dormand-Prince Runge-Kutta 4(5) with adaptive step size (default); most "
            "accurate for predictions of hours")
        .value("YOSHIDA6", IntegratorType::YOSHIDA6,
            "Yoshida 6th-order symplectic integrator: fixed step, no long-term energy "
            "drift; recommended for runs of 6 hours or more")
        .value("YOSHIDA8", IntegratorType::YOSHIDA8,
            "Yoshida 8th-order symplectic integrator");

    // ITRFRealization enum
    nb::enum_<coords::ITRFRealization>(m, "ITRFRealization",
        R"doc(
Versions ("realizations") of the ITRF, the International Terrestrial
Reference Frame (the Earth-fixed frame). Each version places the frame
slightly differently (millimeters to centimeters, drifting over the years).

Orbit products state which version they use; reading a product in the
wrong version costs accuracy at the sub-meter level.

Guidelines:
- ESA Sentinel (before March 4, 2025): ITRF2014
- ESA Sentinel (after March 4, 2025): ITRF2020
- GFZ CHAMP (2000-2010): ITRF2005
- IGS GPS (2017-2024): ITRF2014
- IGS GPS (2024+): ITRF2020
)doc")
        .value("ITRF2000", coords::ITRFRealization::ITRF2000)
        .value("ITRF2005", coords::ITRFRealization::ITRF2005)
        .value("ITRF2008", coords::ITRFRealization::ITRF2008)
        .value("ITRF2014", coords::ITRFRealization::ITRF2014,
            "Use for Sentinel data before March 4, 2025")
        .value("ITRF2020", coords::ITRFRealization::ITRF2020,
            "Default, use for Sentinel data after March 4, 2025");

    // InterpolationFrame enum.
    // No GAST-only inertial output frame: GCRS is the inertial choice.
    nb::enum_<InterpolationFrame>(m, "InterpolationFrame",
        "Coordinate frame of interpolated states.")
        .value("ECEF", InterpolationFrame::ECEF,
            "Earth-centered, Earth-fixed (ITRF; default)")
        .value("GCRS", InterpolationFrame::GCRS,
            "Inertial Geocentric Celestial Reference System (IAU 2006/2000A transform)");

    // AltitudeCondition enum
    nb::enum_<AltitudeCondition>(m, "AltitudeCondition",
        "Stopping condition for TrajectoryInterpolator.propagate_to_altitude.")
        .value("Above", AltitudeCondition::Above,
            "Propagate until the geodetic altitude is at or above the target")
        .value("Below", AltitudeCondition::Below,
            "Propagate until the geodetic altitude is at or below the target");

    // PropagationDirection enum
    nb::enum_<PropagationDirection>(m, "PropagationDirection",
        "Time direction for TrajectoryInterpolator.propagate_to_altitude.")
        .value("Forward", PropagationDirection::Forward,
            "Propagate forward in time from the last history entry")
        .value("Backward", PropagationDirection::Backward,
            "Propagate backward in time from the first history entry");

    // EmpiricalAccelFrame, EAInterp, EARepresents enums are registered in
    // atmosphere_bindings.cpp (bind_atmosphere runs first, so they exist before
    // the EmpiricalAcceleration default arguments that need them)

    // EmpiricalAccelEntry struct
    nb::class_<forces::EmpiricalAccelEntry>(m, "EmpiricalAccelEntry",
        "One time point of a tabulated empirical acceleration (a small correction "
        "acceleration fitted to observations).")
        .def(nb::init<>(), "Entry with key=0 and a zero acceleration")
        .def(nb::init<double, const Vector3d&>(),
            nb::arg("key"), nb::arg("accel"),
            "Entry at `key` seconds after the reference epoch with acceleration `accel`, "
            "numpy (3,) [m/s^2]")
        .def_rw("key", &forces::EmpiricalAccelEntry::key,
            "Time of the entry: seconds after the reference epoch (the trajectory's start, "
            "or TrajectoryConfig.ea_reference_epoch).")
        .def_rw("accel", &forces::EmpiricalAccelEntry::accel,
            "Acceleration vector, numpy (3,) [m/s^2]");

    // =========================================================================
    // Configuration Classes
    // =========================================================================

    // NonConservativeStrategy enum (used by SymplecticOptions::nc_strategy)
    nb::enum_<propagators::NonConservativeStrategy>(m, "NonConservativeStrategy",
        "How a symplectic integrator handles forces that do not conserve energy (drag, "
        "radiation pressure).")
        .value("NONE", propagators::NonConservativeStrategy::NONE,
            "Ignore those forces (pure symplectic integration).")
        .value("SPLITTING", propagators::NonConservativeStrategy::SPLITTING,
            "Operator splitting: symplectic steps for the energy-conserving forces, "
            "Runge-Kutta steps for the others.")
        .value("SLIMPLECTIC", propagators::NonConservativeStrategy::SLIMPLECTIC,
            "Variational ('slimplectic') integrator that includes them in the action "
            "(default).");

    // SymplecticOptions
    nb::class_<SymplecticOptions>(m, "SymplecticOptions",
        "Options for the symplectic (Yoshida) integrators.")
        .def(nb::init<>())
        .def_rw("step_size", &SymplecticOptions::step_size,
            "Fixed step size [s], or the reference step when adaptive (default: 30.0)")
        .def_rw("nc_strategy", &SymplecticOptions::nc_strategy,
            "Strategy for non-conservative forces (default: SLIMPLECTIC).")
        .def_rw("adaptive", &SymplecticOptions::adaptive,
            "Use time-reversible adaptive step sizes (Hairer, Lubich & Wanner 2006; "
            "default: False)")
        .def_rw("min_step_size", &SymplecticOptions::min_step_size,
            "Smallest adaptive step size [s] (default: 1.0)")
        .def_rw("max_step_size", &SymplecticOptions::max_step_size,
            "Largest adaptive step size [s] (default: 120.0)")
        .def_rw("adaptation_alpha", &SymplecticOptions::adaptation_alpha,
            nb::for_setter(nb::arg().none()),
            "Exponent of the adaptive step-size rule (Optional[float]; None = automatic, "
            "1/order)")
        .def_rw("symmetry_tolerance", &SymplecticOptions::symmetry_tolerance,
            "Relative tolerance for the iteration that keeps adaptive steps time-symmetric "
            "(default: 0.1)");

    // Rk45Options (must precede TrajectoryConfig: nanobind converts defaults
    // at definition time)
    nb::class_<Rk45Options>(m, "Rk45Options",
        "Options for the RK45 (Dormand-Prince) integrator.")
        .def(nb::init<>())
        .def_rw("initial_step_size", &Rk45Options::initial_step_size,
            "First step size to try [s] (default 60.0). "
            "Use about 8 s for gliding reentry or other fast-changing motion.")
        .def_rw("max_step_size", &Rk45Options::max_step_size,
            "Largest step size allowed [s] (default 3600.0). "
            "Use about 8 s for gliding reentry or other fast-changing motion.");

    // SrpConfig (must precede TrajectoryConfig)
    nb::class_<SrpConfig>(m, "SrpConfig",
        R"doc(
Radiation-pressure settings: the push of sunlight on the satellite (solar
radiation pressure, SRP), plus sunlight reflected by the Earth (albedo) and
the Earth's own infrared glow.

Assigning one to TrajectoryConfig.srp turns these forces on; leaving
srp = None keeps them off.

Args (keyword-only):
    mass_kg: satellite mass [kg], > 0
    area_m2: sunlit cross-section [m^2], >= 0
    cr: radiation pressure coefficient (default 1.3)
    albedo: include Earth albedo (default True)
    earth_ir: include Earth infrared (default True)

Raises:
    ballistic.errors.ConfigError: mass_kg not > 0 or area_m2 not >= 0.
)doc")
        .def("__init__", [](SrpConfig* self, double mass_kg, double area_m2,
                            double cr, bool albedo, bool earth_ir) {
                if (!(mass_kg > 0.0) || !(area_m2 >= 0.0))
                    throw ConfigError("SrpConfig needs the satellite's mass_kg (> 0) and "
                                      "area_m2 (>= 0)");
                new (self) SrpConfig{mass_kg, area_m2, cr, albedo, earth_ir};
            },
            nb::kw_only(), nb::arg("mass_kg"), nb::arg("area_m2"),
            nb::arg("cr") = 1.3, nb::arg("albedo") = true,
            nb::arg("earth_ir") = true)
        .def_rw("mass_kg", &SrpConfig::mass_kg,
            "Satellite mass [kg]; used by the radiation forces only (for drag, the mass "
            "enters through the DragEnvironment's ballistic coefficient beta)")
        .def_rw("area_m2", &SrpConfig::area_m2,
            "Sun-facing cross-section [m^2]")
        .def_rw("cr", &SrpConfig::cr,
            "Radiation pressure coefficient Cr (1.0 = perfect absorber, "
            "about 1.3 typical)")
        .def_rw("albedo", &SrpConfig::albedo,
            "Include sunlight reflected by the Earth (default True)")
        .def_rw("earth_ir", &SrpConfig::earth_ir,
            "Include the Earth's infrared radiation (default True)");

    // TrajectoryConfig
    nb::class_<TrajectoryConfig>(m, "TrajectoryConfig",
        R"doc(
Settings for a Trajectory: which forces to model and which integrator to use.

The defaults are the settings measured to be best in general; the measured
1-hour position errors (root-mean-square error, RMSE) per orbit type are in
docs/validation_claims.md.

Solar radiation pressure (SRP) and drag are off by default only because
they need the satellite's mass and areas (and, for drag, space weather).
They do matter: a start state taken from a precise orbit product does not
already account for them. ballistic.recommended_config() turns both on for
a named satellite. Each is switched on by setting it:
    config.srp = SrpConfig(mass_kg=2200, area_m2=38.4, cr=1.0)   # S1A
    config.drag_env = <DragEnvironment>   # make_drag_environment(...)

Example:
    config = TrajectoryConfig()
    config.gravity_degree = 70
    config.enable_third_body = True
    traj = Trajectory(State.from_itrf(epoch, pos, vel), config)
)doc")
        .def(nb::init<>())
        .def_rw("gravity_degree", &TrajectoryConfig::gravity_degree,
            "Maximum degree of the EGM2008 Earth gravity model, i.e. how much detail of the "
            "gravity field is used (70 is enough for low Earth orbit, 4-8 for higher "
            "orbits; at most 360 with the embedded model, above that raises ConfigError)")
        .def_rw("enable_third_body", &TrajectoryConfig::enable_third_body,
            "Include the gravity of the Sun, Moon and Jupiter (essential for accuracy)")
        .def_rw("enable_relativistic", &TrajectoryConfig::enable_relativistic,
            "Include general-relativity corrections, Schwarzschild and Lense-Thirring")
        .def_rw("enable_solid_tides", &TrajectoryConfig::enable_solid_tides,
            "Include tides: solid-Earth tides (IERS 2010 conventions) and, once loaded, "
            "ocean tides (decimeter-level effect)")
        .def_rw("ocean_tide_degree", &TrajectoryConfig::ocean_tide_degree,
            "Maximum degree of the FES2004 ocean-tide model (0 = off); takes effect once "
            "the tide file is loaded (ballistic.load())")
        .def_rw("itrf", &TrajectoryConfig::itrf,
            "ITRF version (ITRFRealization) used to convert between Earth-fixed and "
            "inertial coordinates")
        .def_rw("integrator", &TrajectoryConfig::integrator,
            "Integrator (IntegratorType): RK45 for hours, YOSHIDA6 for longer runs")
        .def_rw("rk45", &TrajectoryConfig::rk45,
            "Options for the RK45 integrator (ignored for symplectic)")
        .def_rw("symplectic", &TrajectoryConfig::symplectic,
            "Options for the symplectic integrators")
        .def_rw("srp", &TrajectoryConfig::srp,
            nb::for_setter(nb::arg().none()),
            "Radiation-pressure settings (Optional[SrpConfig]; None = radiation off, the "
            "default)")
        .def_rw("drag_env", &TrajectoryConfig::drag_env,
            "Atmospheric drag environment (DragEnvironment or None). Setting it turns drag "
            "on; None (the default) means no drag. Build one with make_drag_environment(...) "
            "or use ballistic.recommended_config().")
        .def_rw("empirical_accel", &TrajectoryConfig::empirical_accel,
            "Empirical acceleration (EA): a small correction acceleration, usually fitted "
            "to observations by an estimator, added to the modeled forces "
            "(EmpiricalAcceleration or None)")
        .def_rw("ea_reference_epoch", &TrajectoryConfig::ea_reference_epoch,
            nb::for_setter(nb::arg().none()),
            "Epoch from which the empirical acceleration's time keys are counted "
            "(Optional[Epoch]; None = the trajectory's start; needs empirical_accel)")
        .def_rw("report_forces", &TrajectoryConfig::report_forces,
            "True: compute each force separately and return the breakdown with results "
            "(ForceBreakdown)")
        .def_rw("include_disabled_forces", &TrajectoryConfig::include_disabled_forces,
            "True: the breakdown also includes forces that are switched off (for "
            "analysis; they still do not act on the orbit)");

    // InterpolatorConfig
    nb::class_<InterpolatorConfig>(m, "InterpolatorConfig",
        "Settings for a TrajectoryInterpolator.")
        .def(nb::init<>())
        .def_rw("output_frame", &InterpolatorConfig::output_frame,
            "Coordinate frame of returned states (InterpolationFrame; default: ECEF)")
        .def_rw("itrf", &InterpolatorConfig::itrf,
            nb::for_setter(nb::arg().none()),
            "ITRF version used when converting to the GCRS (Optional; None = the wrapped "
            "Trajectory's)")
        .def_rw("cache_coefficients", &InterpolatorConfig::cache_coefficients,
            "Cache interpolation polynomial coefficients (default: True)")
        .def_rw("allow_propagation", &InterpolatorConfig::allow_propagation,
            "Propagate the trajectory to answer queries outside its history (default: True)")
        .def_rw("max_propagation_time", &InterpolatorConfig::max_propagation_time,
            "Furthest beyond the history [s] that a query may propagate (default: 3600)");

    // =========================================================================
    // Model Classes
    // =========================================================================

    // BallisticModel
    nb::class_<BallisticModel>(m, "BallisticModel",
        R"doc(
Ballistic coefficient and optional mass and area of a body.

beta is the ballistic coefficient m / (Cd * A) [kg/m^2]: mass over drag
coefficient times frontal area. mass and drag_area are optional metadata
(mass may be used for radiation pressure, drag_area only for display). A
Trajectory's drag does not read this class: it takes beta from
TrajectoryConfig.drag_env.

Attributes:
    beta: Ballistic coefficient [kg/m^2] (default: 45.45 = 1000 / (2.2 * 10))
    mass: Mass [kg] (optional, default 0)
    drag_area: Drag cross-section [m^2] (optional, default 0)

Example:
    model = BallisticModel()              # Default beta=45.45
    model = BallisticModel(50.0)          # Explicit beta
    model = BallisticModel(50.0, 1000.0, 10.0)  # beta, mass, area
)doc")
        .def(nb::init<>(), "beta = 45.45 kg/m^2, no mass or area")
        .def(nb::init<double>(),
            nb::arg("beta"),
            "Given ballistic coefficient beta [kg/m^2]")
        .def(nb::init<double, double, double>(),
            nb::arg("beta"), nb::arg("mass"), nb::arg("area"),
            "Given beta [kg/m^2], mass [kg] and drag area [m^2]")
        .def_rw("beta", &BallisticModel::beta,
            "Ballistic coefficient [kg/m^2]")
        .def_rw("mass", &BallisticModel::mass,
            "Mass [kg] (optional metadata)")
        .def_rw("drag_area", &BallisticModel::drag_area,
            "Drag cross-section [m^2] (optional metadata)")
        .def("ballistic_coefficient", &BallisticModel::ballistic_coefficient,
            "The ballistic coefficient beta [kg/m^2]")
        .def("inverse_ballistic_coefficient", &BallisticModel::inverse_ballistic_coefficient,
            "1 / beta [m^2/kg]");

    // WeatherConditions
    nb::class_<WeatherConditions>(m, "WeatherConditions",
        R"doc(
One fixed set of space-weather values, the inputs the atmosphere model
needs. F10.7 is the Sun's radio flux at 10.7 cm wavelength, in solar flux
units (SFU = 1e-22 W/m^2/Hz), a measure of the solar ultraviolet that heats
the upper atmosphere. Ap is a daily index of geomagnetic activity (disturbance
of the Earth's magnetic field), which also heats it.

Attributes:
    f107_daily: Daily F10.7 [SFU]
    f107_average: 81-day centered average of F10.7 [SFU]
    ap_daily: Daily Ap index
    ap_history: 7 Ap values in the order the NRLMSIS 2.1 model reads them
        (daily, then 3-hourly values going back in time)

Factory methods:
    - quiet(): solar minimum (F10.7 = 70, Ap = 4)
    - active(): solar maximum (F10.7 = 200, Ap = 15)
    - storm(): geomagnetic storm (daily F10.7 = 250, 81-day mean 200, Ap = 100)
)doc")
        .def(nb::init<>(), "Moderate conditions (F10.7 = 150, Ap = 4)")
        .def_rw("f107_daily", &WeatherConditions::f107_daily,
            "Daily F10.7 solar flux [SFU]")
        .def_rw("f107_average", &WeatherConditions::f107_average,
            "81-day centered average F10.7 [SFU]")
        .def_rw("ap_daily", &WeatherConditions::ap_daily,
            "Daily Ap geomagnetic index")
        .def_rw("ap_history", &WeatherConditions::ap_history,
            "7 Ap values in NRLMSIS 2.1 order: daily, the current 3-hourly value, the three "
            "before it, then the averages of eight 3-hourly values 12-33 h and 36-57 h back")
        .def_static("quiet", &WeatherConditions::quiet,
            "Quiet Sun (solar minimum): F10.7 = 70, Ap = 4")
        .def_static("active", &WeatherConditions::active,
            "Active Sun (solar maximum): F10.7 = 200, Ap = 15")
        .def_static("storm", &WeatherConditions::storm,
            "Geomagnetic storm: daily F10.7 = 250, Ap = 100");

    // =========================================================================
    // Result Structures
    // =========================================================================

    // StateVector
    nb::class_<StateVector>(m, "StateVector",
        "Position and velocity; the frame depends on where the state came from (see the "
        "returning function).")
        .def(nb::init<>())
        .def(nb::init<const Vector3d&, const Vector3d&>(),
            nb::arg("position"), nb::arg("velocity"))
        .def_rw("position", &StateVector::position, "Position, numpy (3,) [m]")
        .def_rw("velocity", &StateVector::velocity, "Velocity, numpy (3,) [m/s]");

    // ForceBreakdown
    nb::class_<ForceBreakdown>(m, "ForceBreakdown",
        "Acceleration from each force separately, each numpy (3,) [m/s^2] (filled in "
        "when TrajectoryConfig.report_forces is True).")
        .def(nb::init<>())
        .def_ro("gravity", &ForceBreakdown::gravity, "Earth gravity")
        .def_ro("third_body", &ForceBreakdown::third_body, "Gravity of the Sun, Moon and Jupiter")
        .def_ro("drag", &ForceBreakdown::drag, "Atmospheric drag")
        .def_ro("srp", &ForceBreakdown::srp, "Solar radiation pressure")
        .def_ro("albedo", &ForceBreakdown::albedo, "Earth albedo radiation")
        .def_ro("ir", &ForceBreakdown::ir, "Earth infrared radiation")
        .def_ro("relativistic", &ForceBreakdown::relativistic, "General-relativity corrections")
        .def_ro("solid_tides", &ForceBreakdown::solid_tides, "Solid-Earth tide corrections")
        .def_ro("ocean_tides", &ForceBreakdown::ocean_tides, "Ocean tide corrections (FES2004 model, once loaded)")
        .def_ro("empirical", &ForceBreakdown::empirical, "Empirical (fitted correction) acceleration")
        .def_ro("total", &ForceBreakdown::total, "Sum of all components")
        .def("radiation", &ForceBreakdown::radiation, "Sum of the radiation forces: SRP + albedo + infrared");

    // PropagationResult
    nb::class_<PropagationResult>(m, "PropagationResult",
        "State at the end of Trajectory.propagate, Earth-fixed (ECEF).")
        .def(nb::init<>())
        .def_ro("position", &PropagationResult::position, "Position, ECEF, numpy (3,) [m]")
        .def_ro("velocity", &PropagationResult::velocity,
            "Velocity relative to the rotating Earth, numpy (3,) [m/s]")
        .def_ro("acceleration", &PropagationResult::acceleration,
            "Acceleration, ECEF axes, numpy (3,) [m/s^2]")
        .def_ro("forces", &PropagationResult::forces,
            "ForceBreakdown, or None unless TrajectoryConfig.report_forces is True");

    // HistoryEntry
    nb::class_<HistoryEntry>(m, "HistoryEntry",
        "One stored state in a Trajectory's history (Earth-fixed, ECEF).")
        .def(nb::init<>())
        .def_ro("epoch", &HistoryEntry::epoch, "Time of the state (Epoch)")
        .def_ro("t_rel_s", &HistoryEntry::t_rel_s,
            "Exact seconds elapsed since the trajectory's reference time (its start or last "
            "reset); the interpolator uses this as its time axis")
        .def_ro("position", &HistoryEntry::position, "Position, ECEF, numpy (3,) [m]")
        .def_ro("velocity", &HistoryEntry::velocity,
            "Velocity relative to the rotating Earth, numpy (3,) [m/s]")
        .def_ro("acceleration", &HistoryEntry::acceleration,
            "Acceleration, ECEF axes, numpy (3,) [m/s^2]")
        .def_ro("step_size_used", &HistoryEntry::step_size_used, "Step size that led here [s]");

    // StepResult
    nb::class_<StepResult>(m, "StepResult",
        "What Trajectory.step() and step_backward() return: the new state (ECEF) and "
        "step information.")
        .def(nb::init<>())
        .def_ro("epoch", &StepResult::epoch, "Time after the step (Epoch)")
        .def_ro("position", &StepResult::position, "New position, ECEF, numpy (3,) [m]")
        .def_ro("velocity", &StepResult::velocity,
            "New velocity relative to the rotating Earth, numpy (3,) [m/s]")
        .def_ro("acceleration", &StepResult::acceleration,
            "Acceleration at the new state, ECEF axes [m/s^2]")
        .def_ro("step_size_taken", &StepResult::step_size_taken, "Step size actually used [s]")
        .def_ro("next_step_size", &StepResult::next_step_size, "Step size suggested for the next step [s]")
        .def_ro("history_index", &StepResult::history_index, "Position of the new state in the history")
        .def_ro("forces", &StepResult::forces,
            "ForceBreakdown, or None unless TrajectoryConfig.report_forces is True");

    // InterpolatedState
    nb::class_<InterpolatedState>(m, "InterpolatedState",
        "State returned by a TrajectoryInterpolator query, in the interpolator's output "
        "frame.")
        .def(nb::init<>())
        .def_ro("epoch", &InterpolatedState::epoch, "Query time (Epoch)")
        .def_ro("position", &InterpolatedState::position, "Position, numpy (3,) [m]")
        .def_ro("velocity", &InterpolatedState::velocity, "Velocity, numpy (3,) [m/s]")
        .def_ro("acceleration", &InterpolatedState::acceleration, "Acceleration, numpy (3,) [m/s^2]")
        .def_ro("was_propagated", &InterpolatedState::was_propagated,
            "True if the history had to be extended to answer this query")
        .def_ro("segment_index", &InterpolatedState::segment_index,
            "Index of the history interval the time fell in");

    // PropagateToAltitudeResult
    nb::class_<PropagateToAltitudeResult>(m, "PropagateToAltitudeResult",
        "Result of TrajectoryInterpolator.propagate_to_altitude.")
        .def(nb::init<>())
        .def_ro("condition_met", &PropagateToAltitudeResult::condition_met,
            "True if the target altitude was reached")
        .def_ro("steps_taken", &PropagateToAltitudeResult::steps_taken,
            "Number of steps added to the history")
        .def_ro("starting_altitude_m", &PropagateToAltitudeResult::starting_altitude_m,
            "Geodetic altitude (height above the WGS84 ellipsoid) at the start [m]")
        .def_ro("ending_altitude_m", &PropagateToAltitudeResult::ending_altitude_m,
            "Geodetic altitude at the end [m]");

    // =========================================================================
    // Coordinate Frame Conversions
    // =========================================================================

    m.def("cip_drift_velocity",
        [](const Epoch& epoch, const Vector3d& pos_ecef, coords::ITRFRealization itrf) {
            return coords::cip_drift_velocity(pos_ecef, epoch, itrf);
        },
        nb::arg("epoch"), nb::arg("pos_ecef"),
        nb::arg("itrf") = coords::ITRFRealization::ITRF2020,
        R"doc(
Small Earth-fixed velocity term caused by the slow motion of the Earth's
rotation axis in space (the drift of the Celestial Intermediate Pole, CIP:
precession and nutation), about 2e-5 m/s at low Earth orbit. Returns a numpy
(3,) velocity [m/s] for the Earth-fixed position pos_ecef [m] at `epoch`.

An orbit product that wrote its Earth-fixed velocities without this term
needs vel_file - cip_drift_velocity(...). Check a product first with
check_velocity_records(): ESA Sentinel EOF files and GFZ CHAMP/GRACE-FO SP3
(standard precise-orbit format) files contain true derivatives and need no
correction.
)doc");

    nb::enum_<coords::VelocityRecordVerdict>(m, "VelocityRecordVerdict",
        "Outcome of check_velocity_records.")
        .value("TRUE_DERIVATIVE", coords::VelocityRecordVerdict::TrueDerivative,
               "Records match the time derivative of the positions: use them as written")
        .value("OMITS_CIP_DRIFT", coords::VelocityRecordVerdict::OmitsCipDrift,
               "Records leave out the rotation-axis drift term: subtract cip_drift_velocity()")
        .value("INCONSISTENT", coords::VelocityRecordVerdict::Inconsistent,
               "Neither: start from velocities derived from the positions");

    nb::class_<coords::VelocityRecordCheck>(m, "VelocityRecordCheck",
        "Result of check_velocity_records: statistics of (velocity derived from the "
        "positions) minus (velocity record).")
        .def_ro("n_compared", &coords::VelocityRecordCheck::n_compared,
                "Number of samples compared")
        .def_ro("median_difference_rtn", &coords::VelocityRecordCheck::median_difference_rtn,
                "Median (derived - record) along the radial, along-track and cross-track "
                "directions, numpy (3,) [m/s]")
        .def_ro("rms_difference", &coords::VelocityRecordCheck::rms_difference,
                "Root-mean-square size of (derived - record) [m/s]")
        .def_ro("mean_cip_drift_along", &coords::VelocityRecordCheck::mean_cip_drift_along,
                "Mean along-track size of the rotation-axis drift term (cip_drift_velocity) "
                "[m/s], for comparison with the median difference")
        .def_ro("standard_error_along", &coords::VelocityRecordCheck::standard_error_along,
                "Robust (median-absolute-deviation) standard error of the along-track median "
                "difference [m/s]")
        .def_ro("verdict", &coords::VelocityRecordCheck::verdict,
                "What to do with the records (VelocityRecordVerdict)")
        .def_ro("summary", &coords::VelocityRecordCheck::summary,
                "One-line human-readable summary (str)")
        .def("__repr__", [](const coords::VelocityRecordCheck& c) {
            return "VelocityRecordCheck(" + c.summary + ")";
        });

    m.def("check_velocity_records",
        [](const Epoch& epoch0, const std::vector<double>& offsets_s,
           Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> positions,
           Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> velocities,
           coords::ITRFRealization itrf) {
            std::vector<Vector3d> p(positions.rows()), v(velocities.rows());
            for (Eigen::Index i = 0; i < positions.rows(); ++i) p[i] = positions.row(i).transpose();
            for (Eigen::Index i = 0; i < velocities.rows(); ++i) v[i] = velocities.row(i).transpose();
            return coords::check_velocity_records(epoch0, offsets_s, p, v, itrf);
        },
        nb::arg("epoch0"), nb::arg("offsets_s"), nb::arg("positions"), nb::arg("velocities"),
        nb::arg("itrf") = coords::ITRFRealization::ITRF2020,
        nb::call_guard<nb::gil_scoped_release>(),
        R"doc(
Check whether an orbit product's Earth-fixed velocity records are the time
derivative of its own Earth-fixed positions, which is what Trajectory needs.

Args:
    epoch0: Epoch of the first sample
    offsets_s: sample times as seconds after epoch0 (list of float). Take
        them from the product's own time tags (e.g. datetime differences)
        rather than from subtracting Epoch values, so that rounding in the
        times cannot disturb a test this sensitive at low-orbit speeds.
    positions: numpy (N,3) Earth-fixed positions [m]
    velocities: numpy (N,3) velocity records [m/s]
    itrf: ITRF version of the coordinates

The positions are differentiated (9-point Lagrange polynomial, interior
samples only) and compared with the records. The returned
VelocityRecordCheck's verdict says how to use the records: as written,
minus cip_drift_velocity(), or not at all (derive the velocities from the
positions instead). Use evenly spaced samples, at most 60 s apart for low
Earth orbit.
)doc");

    m.def("gcrs_to_itrf",
        [](const Vector3d& pos, const Vector3d& vel, const Epoch& epoch,
           coords::ITRFRealization itrf) -> StateVector {
            return coords::gcrs_to_itrf_full(StateVector(pos, vel), epoch, itrf);
        },
        nb::arg("pos_gcrs"), nb::arg("vel_gcrs"), nb::arg("epoch"),
        nb::arg("itrf") = coords::ITRFRealization::ITRF2020,
        R"doc(
Convert an inertial (GCRS) state to Earth-fixed (ITRF / ECEF) coordinates
with the same IAU 2006/2000A transform the propagator uses (precession and
nutation of the Earth's axis including its drift, Earth rotation, polar
motion).

Args:
    pos_gcrs: numpy (3,) position [m]
    vel_gcrs: numpy (3,) velocity [m/s]
    epoch: time of the state (Epoch)
    itrf: ITRF version of the result

Returns a StateVector (position [m], velocity relative to the rotating Earth
[m/s]). Uses the loaded Earth orientation parameters (EOP: polar motion and
UT1-UTC); raises ballistic.errors.DataCoverageError when none are loaded
(unless ballistic.set_eop_policy('assume_zero')) or the date is more than 30
days outside the loaded table. A J2000 (EME2000) state can be passed as GCRS if a 0.7 m error at
low Earth orbit is acceptable: the frame bias between them is about 23
milliarcseconds; State.from_j2000 applies it exactly.
)doc");

    m.def("itrf_to_gcrs",
        [](const Vector3d& pos, const Vector3d& vel, const Epoch& epoch,
           coords::ITRFRealization itrf) -> StateVector {
            return coords::itrf_to_gcrs_full(StateVector(pos, vel), epoch, itrf);
        },
        nb::arg("pos_itrf"), nb::arg("vel_itrf"), nb::arg("epoch"),
        nb::arg("itrf") = coords::ITRFRealization::ITRF2020,
        R"doc(
Convert an Earth-fixed (ITRF / ECEF) state to inertial (GCRS) coordinates;
the inverse of gcrs_to_itrf. pos_itrf [m] and vel_itrf (relative to the
rotating Earth) [m/s] are numpy (3,); returns a StateVector.
)doc");
}
