// =============================================================================
// Main Python Bindings Module
// =============================================================================
//
// This file defines the _core extension module for the ballistic Python package.
// The bindings themselves are in separate files, one bind_* function each:
//   - diagnostics_bindings.cpp: exception classes and the warning handler
//   - init_bindings.cpp: initialization and data loading
//   - satellites_bindings.cpp, constants_bindings.cpp: registry and constants
//   - atmosphere_bindings.cpp: drag environment, space weather, drag coefficient
//   - ea_bindings.cpp: EmpiricalAcceleration (correction acceleration)
//   - epoch_bindings.cpp: Epoch class
//   - types_bindings.cpp: configuration types, enums, frame conversions
//   - trajectory_bindings.cpp: State, Trajectory, TrajectoryInterpolator, events
//   - batch_bindings.cpp: TrajectoryBatch
//   - calibration_bindings.cpp, multisensor_bindings.cpp: correction-acceleration
//     estimators
//   - validation_bindings.cpp, sp3_bindings.cpp: orbit-file readers
//   - sgp4_bindings.cpp: two-line element sets (TLE) and SGP4
//
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <ballistic/ballistic.hpp>  // version constants (+ umbrella headers)
// The atmosphere / EA / forces headers are included by their own binding
// units (init/atmosphere/ea_bindings.cpp); this dispatcher only needs the
// version constants + nanobind core.
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// Forward declarations for binding functions
void bind_epoch(nb::module_& m);
void bind_types(nb::module_& m);
void bind_trajectory(nb::module_& m);
void bind_batch(nb::module_& m);
void bind_calibration(nb::module_& m);
void bind_multisensor(nb::module_& m);
void bind_validation(nb::module_& m);
void bind_sp3(nb::module_& m);
void bind_sgp4(nb::module_& m);
void bind_init(nb::module_& m);
void bind_atmosphere(nb::module_& m);
void bind_ea(nb::module_& m);
void bind_diagnostics(nb::module_& m);
void bind_satellites(nb::module_& m);
void bind_constants(nb::module_& m);

NB_MODULE(_core, m) {
    // A module-level Python Trajectory whose config holds a DragEnvironment
    // outlives nanobind's leak check at interpreter shutdown, which then
    // prints "leaked instances ... reference counting issue", e.g. on the
    // README quick start. `del traj` before exit silences it and the garbage
    // collector finds no cycle, so it is the order of teardown, not a real
    // leak. Turned off so the documented recommended path exits cleanly.
    nb::set_leak_warnings(false);

    m.doc() = R"doc(
Ballistic - precise orbit prediction

The compiled core of the ballistic package (Python bindings of the Ballistic
C++ library). It predicts satellite orbits to well under a meter of position
error (root-mean-square) one hour ahead in low Earth orbit, given good force
models and data.

Key classes:
- Epoch: an instant in time, readable in several time scales (UTC, TT, TAI, ...)
- State: a start state with its coordinate frame named and checked
- TrajectoryConfig: which forces and which integrator to use
- Trajectory: the propagator, stepping forward or backward and keeping a history
- TrajectoryInterpolator: the state at any time between steps (quintic Hermite)
- TrajectoryBatch: many trajectories propagated together

Quick start:
    import ballistic
    import numpy as np

    # Load the data files (Earth orientation, leap seconds, space weather,
    # ocean tides; downloaded once into the data directory)
    ballistic.load()

    # Create an epoch
    epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)

    # The start names its frame and is checked (Earth-fixed velocity here)
    state = ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0])
    # or State.from_gcrs / from_tle(line1, line2) / from_sp3(...)

    traj = ballistic.Trajectory(state)
    traj.propagate_for(3600.0)

    # Interpolate at any time
    interp = ballistic.TrajectoryInterpolator(traj)
    state = interp.get_state(epoch + 1800.0)
    print(f"Position: {state.position}")
)doc";

    // Version info
    m.attr("__version__") = ballistic::VERSION_STRING;
    m.attr("VERSION_MAJOR") = ballistic::VERSION_MAJOR;
    m.attr("VERSION_MINOR") = ballistic::VERSION_MINOR;
    m.attr("VERSION_PATCH") = ballistic::VERSION_PATCH;


    // Registration order matters: nanobind converts default arguments when a
    // function is defined, so a type used as a default must be registered
    // first (the atmosphere enums before the EmpiricalAcceleration factories).
    bind_diagnostics(m);   // exceptions + warning handler first: everything else may raise
    bind_init(m);
    bind_satellites(m);
    bind_constants(m);
    bind_atmosphere(m);
    bind_ea(m);

    // Bind all components
    bind_epoch(m);
    bind_types(m);
    bind_trajectory(m);
    bind_batch(m);
    bind_calibration(m);
    bind_multisensor(m);
    bind_validation(m);
    bind_sp3(m);
    bind_sgp4(m);
}
