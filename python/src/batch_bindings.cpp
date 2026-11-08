// =============================================================================
// TrajectoryBatch Python Bindings
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>
#include <ballistic/ephemeris/spk_ephemeris.hpp>  // Needed for get_default_ephemeris_provider()
#include "trajectory_batch.hpp"
#include "batch_interpolator_function.hpp"

namespace nb = nanobind;

void bind_batch(nb::module_& m) {
    using namespace ballistic;
    using namespace ballistic::python;

    // =========================================================================
    // Batch Result Types
    // =========================================================================

    // BatchInterpolatedState
    nb::class_<BatchInterpolatedState>(m, "BatchInterpolatedState",
        "State of one trajectory of a TrajectoryBatch at a query time.")
        .def(nb::init<>())
        .def_ro("trajectory_index", &BatchInterpolatedState::trajectory_index,
            "Index of the trajectory in the batch")
        .def_ro("valid", &BatchInterpolatedState::valid,
            "True if the query succeeded (otherwise the state fields are meaningless)")
        .def_ro("epoch", &BatchInterpolatedState::epoch,
            "Query time (Epoch)")
        .def_ro("position", &BatchInterpolatedState::position,
            "Position in output frame (meters)")
        .def_ro("velocity", &BatchInterpolatedState::velocity,
            "Velocity in output frame (m/s)")
        .def_ro("acceleration", &BatchInterpolatedState::acceleration,
            "Acceleration in output frame (m/s^2)")
        .def_ro("was_propagated", &BatchInterpolatedState::was_propagated,
            "True if the history had to be extended to answer this query");

    // BatchAltitudeResult
    nb::class_<BatchAltitudeResult>(m, "BatchAltitudeResult",
        "Result of TrajectoryBatch.propagate_to_altitude for one trajectory.")
        .def(nb::init<>())
        .def_ro("trajectory_index", &BatchAltitudeResult::trajectory_index,
            "Index of the trajectory in the batch")
        .def_ro("condition_met", &BatchAltitudeResult::condition_met,
            "True if the target altitude was reached")
        .def_ro("steps_taken", &BatchAltitudeResult::steps_taken,
            "Number of steps added to the history")
        .def_ro("starting_altitude_m", &BatchAltitudeResult::starting_altitude_m,
            "Geodetic altitude (height above the WGS84 ellipsoid) at the start [m]")
        .def_ro("ending_altitude_m", &BatchAltitudeResult::ending_altitude_m,
            "Geodetic altitude at the end [m]");

    // =========================================================================
    // TrajectoryBatch Class
    // =========================================================================

    nb::class_<TrajectoryBatch>(m, "TrajectoryBatch",
        R"doc(
TrajectoryBatch - many trajectories propagated and queried together.

The loops over trajectories run in C++, so a batch is much faster than a
Python loop over single Trajectory objects. Every batch operation returns a
list with one result per trajectory.

Example:
    import ballistic
    import numpy as np

    ballistic.initialize()

    # Create batch from initial states
    epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    states = [
        ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0]),
        ballistic.State.from_itrf(epoch, [6878137.0, 0.0, 0.0], [0.0, 7120.0, 0.0]),
    ]
    batch = ballistic.TrajectoryBatch(states)

    # Build history for all trajectories
    batch.propagate_all(3600.0)  # 1 hour

    # Query all at a specific time
    results = batch.get_states_at(epoch + 1800.0)
    for r in results:
        if r.valid:
            print(f"Trajectory {r.trajectory_index}: pos={r.position}")

    # Propagate all to altitude
    alt_results = batch.propagate_to_altitude(
        300_000,  # meters
        ballistic.AltitudeCondition.Below)
    for r in alt_results:
        print(f"Trajectory {r.trajectory_index}: {r.ending_altitude_m/1000:.1f} km")
)doc")

        .def(nb::init<const std::vector<ballistic::InitialState>&, const TrajectoryConfig&>(),
            nb::arg("states"), nb::arg("config") = TrajectoryConfig{},
            "Build from a list of start states (ballistic.State), each with its own epoch, "
            "and one TrajectoryConfig shared by all.")

        // Size and access
        .def("__len__", &TrajectoryBatch::size,
            "Number of trajectories")
        .def("size", &TrajectoryBatch::size,
            "Number of trajectories")
        .def("empty", &TrajectoryBatch::empty,
            "True if the batch holds no trajectories")

        // Individual access
        .def("trajectory", nb::overload_cast<size_t>(&TrajectoryBatch::trajectory),
            nb::arg("index"), nb::rv_policy::reference_internal,
            "The Trajectory at position `index`")
        .def("interpolator", &TrajectoryBatch::interpolator,
            nb::arg("index"), nb::rv_policy::reference_internal,
            "The TrajectoryInterpolator of trajectory `index` (created on first use)")

        // Batch propagation
        .def("propagate_all", &TrajectoryBatch::propagate_all,
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("duration_seconds"),
            R"doc(
Propagate every trajectory by duration_seconds [s] from its own current
epoch, landing exactly there; the steps are kept in each history.
)doc")
        .def("propagate_all_to", &TrajectoryBatch::propagate_all_to,
            nb::call_guard<nb::gil_scoped_release>(), nb::arg("target"),
            "Propagate every trajectory to the epoch `target`, landing exactly on it.")

        .def("step_all", &TrajectoryBatch::step_all,
            nb::arg("n_steps") = 1,
            R"doc(
Take n_steps integrator steps on every trajectory.

Args:
    n_steps: Number of steps to take (default 1)
)doc")

        // Batch queries
        .def("get_states_at", &TrajectoryBatch::get_states_at,
            nb::arg("epoch"),
            R"doc(
The state of every trajectory at one time.

Args:
    epoch: Query time (Epoch)

Returns:
    List of BatchInterpolatedState, one per trajectory
)doc")

        .def("get_states_at_times", &TrajectoryBatch::get_states_at_times,
            nb::arg("times"),
            R"doc(
The state of every trajectory at several times.

Args:
    times: List of Epoch

Returns:
    List of lists: results[traj_idx][time_idx]
)doc")

        // Altitude propagation
        .def("propagate_to_altitude", &TrajectoryBatch::propagate_to_altitude,
            nb::call_guard<nb::gil_scoped_release>(),
            nb::arg("target_altitude_m"),
            nb::arg("condition"),
            nb::arg("direction") = PropagationDirection::Forward,
            nb::arg("max_duration_s") = 7.0 * 86400.0,
            R"doc(
Propagate every trajectory until its height above the WGS84 ellipsoid
(geodetic altitude) goes above or below a threshold; see
TrajectoryInterpolator.propagate_to_altitude.

Args:
    target_altitude_m: Threshold geodetic altitude [m]
    condition: AltitudeCondition.Above or AltitudeCondition.Below
    direction: PropagationDirection.Forward or Backward
    max_duration_s: How far to search per trajectory [s] (default 7 days)

Returns:
    List of BatchAltitudeResult, one per trajectory
)doc")

        // Configuration
        .def("config", &TrajectoryBatch::config, nb::rv_policy::reference,
            "The shared TrajectoryConfig (read-only)")
        .def("set_output_frame", &TrajectoryBatch::set_output_frame,
            nb::arg("frame"),
            "Choose the coordinate frame (InterpolationFrame) of states returned by every "
            "interpolator")
        .def("set_itrf_realization", &TrajectoryBatch::set_itrf_realization,
            nb::arg("itrf"),
            "Choose the ITRF version every interpolator uses for frame conversions")
        .def("set_allow_propagation", &TrajectoryBatch::set_allow_propagation,
            nb::arg("allow"),
            "Allow (True) or forbid (False) every interpolator to propagate to answer "
            "queries outside its history")
        .def("set_max_propagation_time", &TrajectoryBatch::set_max_propagation_time,
            nb::arg("max_seconds"),
            "Furthest beyond its history [s] that a query may propagate, for every "
            "interpolator");

    // =========================================================================
    // BatchArrayResult
    // =========================================================================

    nb::class_<BatchArrayResult>(m, "BatchArrayResult",
        R"doc(
States of all trajectories of a batch at one query time, as numpy arrays
with N rows (one per trajectory).

Attributes:
    position: (N, 3) position array in output frame (meters)
    velocity: (N, 3) velocity array in output frame (m/s)
    acceleration: (N, 3) acceleration array in output frame (m/s^2)
    epoch_s: (N,) query times in seconds after the reference epoch
    valid: (N,) 1 where the row is valid, 0 where the query failed
)doc")
        .def_ro("position", &BatchArrayResult::positions,
            "Position array (N, 3) in output frame (meters)")
        .def_ro("velocity", &BatchArrayResult::velocities,
            "Velocity array (N, 3) in output frame (m/s)")
        .def_ro("acceleration", &BatchArrayResult::accelerations,
            "Acceleration array (N, 3) in output frame (m/s^2)")
        .def_ro("epoch_s", &BatchArrayResult::epochs_s,
            "Query times (N,) in seconds after the reference epoch")
        .def_ro("valid", &BatchArrayResult::valid,
            "Validity mask (N,): 1 = valid, 0 = query failed")
        .def("__len__", &BatchArrayResult::size,
            "Number of trajectories (rows)")
        .def("__repr__", [](const BatchArrayResult& r) {
            int n_valid = r.valid.sum();
            return "<BatchArrayResult rows=" + std::to_string(r.valid.size())
                + " valid=" + std::to_string(n_valid) + ">";
        });

    // =========================================================================
    // BatchInterpolatorFunction
    // =========================================================================

    nb::class_<BatchInterpolatorFunction>(m, "BatchInterpolatorFunction",
        R"doc(
Interpolates all trajectories of a batch at once and returns numpy arrays
instead of one Python object per trajectory. Obtained by calling a batch
Trajectory with no argument, ``traj()``. Times given as floats are seconds
after the reference epoch.

Example:
    batch = ballistic.Trajectory(states_list)
    batch.propagate_for(3600.0)

    f = batch()                   # BatchInterpolatorFunction
    r = f(1800.0)                 # BatchArrayResult
    r.position                    # (N, 3) numpy array

    impact = f.at_alt(0)          # Find ground impact
    f[42].get_state(epoch + 1800) # Debug single trajectory
)doc")
        .def(nb::init<TrajectoryBatch&, const Epoch&>(),
            nb::arg("batch"), nb::arg("reference_epoch"),
            nb::keep_alive<1, 2>(),
            "Wrap a TrajectoryBatch; float times are counted from reference_epoch (Epoch)")

        // __call__: dispatch float/Epoch/list/ndarray
        .def("__call__",
            [](BatchInterpolatorFunction& self, nb::object time_arg) -> nb::object {
                // Single float → query at reference_epoch + seconds
                if (nb::isinstance<nb::float_>(time_arg) || nb::isinstance<nb::int_>(time_arg)) {
                    double seconds = nb::cast<double>(time_arg);
                    Epoch t = self.reference_epoch() + seconds;
                    return nb::cast(self.query(t));
                }

                // Single Epoch → query directly
                if (nb::isinstance<Epoch>(time_arg)) {
                    Epoch t = nb::cast<Epoch>(time_arg);
                    return nb::cast(self.query(t));
                }

                // List or ndarray → multiple times
                std::vector<Epoch> times;
                nb::object py_list = nb::steal(PySequence_List(time_arg.ptr()));
                if (!py_list.is_valid()) {
                    throw nb::type_error("times must be float, Epoch, or iterable");
                }
                Py_ssize_t n = PyList_Size(py_list.ptr());
                times.reserve(static_cast<size_t>(n));
                for (Py_ssize_t idx = 0; idx < n; ++idx) {
                    nb::object item = nb::borrow(PyList_GetItem(py_list.ptr(), idx));
                    if (nb::isinstance<Epoch>(item)) {
                        times.push_back(nb::cast<Epoch>(item));
                    } else {
                        double seconds = nb::cast<double>(item);
                        times.push_back(self.reference_epoch() + seconds);
                    }
                }
                return nb::cast(self.query_times(times));
            },
            nb::arg("times"),
            R"doc(
The states of all trajectories at one or several times.

Args:
    times: float (seconds from reference epoch), Epoch, or
           list/array of floats or Epochs

Returns:
    BatchArrayResult for single time, or list of BatchArrayResult
    for multiple times
)doc")

        // at_alt: altitude crossing search
        .def("at_alt", &BatchInterpolatorFunction::at_altitude,
            nb::arg("altitude_m"), nb::arg("descending") = true,
            nb::arg("max_time_s") = 0.0,
            R"doc(
Find where each trajectory crosses a geodetic altitude, within the history
already computed (nothing is propagated).

Scans each history for the step where the altitude crosses the target, then
bisects inside it to about 1 mm in altitude.

Args:
    altitude_m: Target geodetic altitude [m]
    descending: If True (default), find last descending crossing
                (above→below, typical for reentry/impact).
                If False, find last ascending crossing.
    max_time_s: Maximum time window in seconds from reference epoch.
                0 (default) = search entire history.

Returns:
    BatchArrayResult with states at crossing points.
    Trajectories that never cross have valid=0.
)doc")

        // Individual access: f[i] → TrajectoryInterpolator
        .def("__getitem__",
            [](BatchInterpolatorFunction& self, int64_t index)
                -> TrajectoryInterpolator&
            {
                auto n = static_cast<int64_t>(self.size());
                if (index < 0) index += n;
                if (index < 0 || index >= n) {
                    throw nb::index_error();
                }
                return self.at(static_cast<size_t>(index));
            },
            nb::arg("index"),
            nb::rv_policy::reference_internal,
            "The TrajectoryInterpolator of trajectory `index` (negative indices count from "
            "the end)")

        .def("__len__", &BatchInterpolatorFunction::size,
            "Number of trajectories")

        .def("set_frame", &BatchInterpolatorFunction::set_output_frame,
            nb::arg("frame"),
            "Choose the coordinate frame (InterpolationFrame) of all query results")

        .def("__repr__", [](const BatchInterpolatorFunction& f) {
            return "<BatchInterpolatorFunction size="
                + std::to_string(f.size()) + ">";
        });
}
