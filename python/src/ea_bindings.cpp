// =============================================================================
// Empirical Acceleration value-type bindings
// =============================================================================
//
// Called from NB_MODULE in bindings.cpp; registration order is set by the
// call order there.
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <ballistic/ballistic.hpp>  // Includes version constants and all headers
#include <ballistic/atmosphere/space_weather.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void bind_ea(nb::module_& m) {
    // =========================================================================
    // Empirical Acceleration Table
    // =========================================================================

    nb::class_<ballistic::forces::EmpiricalAcceleration>(m, "EmpiricalAcceleration",
        R"doc(
Empirical acceleration (EA): a small correction acceleration, as a function
of time, added to the modeled forces (usually fitted to observations by
estimate_ea_bspline or estimate_ea_smoother). Set it as
TrajectoryConfig.empirical_accel. Times are seconds after the reference
epoch (the trajectory's start, or TrajectoryConfig.ea_reference_epoch).

Built with the factory methods below; it cannot be changed afterwards.

Example:
    accel = ballistic.EmpiricalAcceleration.from_entries(
        [ballistic.EmpiricalAccelEntry(0.0, [0, 0, 0]),
         ballistic.EmpiricalAccelEntry(100.0, [1e-5, 2e-6, 0]),
         ballistic.EmpiricalAccelEntry(300.0, [0, 0, 0])],
        frame=ballistic.EmpiricalAccelFrame.GCRS,
    )
    print(accel.accel_at(150.0))   # interpolate at t=150 s
)doc")
        .def(nb::init<>(), "Create the zero EA (no correction).")
        .def_static("zero", &ballistic::forces::EmpiricalAcceleration::zero,
            "The zero EA (no correction).")
        .def_static("from_entries",
            &ballistic::forces::EmpiricalAcceleration::from_entries,
            nb::arg("entries"),
            nb::arg("frame") = ballistic::forces::EmpiricalAccelFrame::GCRS,
            nb::arg("interp") = ballistic::forces::EAInterp::Cubic,
            "Build from a list of EmpiricalAccelEntry (time [s], acceleration [m/s^2]), sorted "
            "internally; `frame` is the frame of the vectors, `interp` how values are "
            "interpolated between entries (EAInterp.Cubic or Linear).")
        .def_static("from_entries_hermite",
            &ballistic::forces::EmpiricalAcceleration::from_entries_hermite,
            nb::arg("entries"), nb::arg("derivatives"),
            nb::arg("frame") = ballistic::forces::EmpiricalAccelFrame::GCRS,
            "Build from a list of EmpiricalAccelEntry plus the time derivative of the "
            "acceleration at each entry [m/s^3] (as many as the entries, else ValueError), "
            "interpolated by cubic Hermite polynomials.")
        .def_static("from_bspline",
            &ballistic::forces::EmpiricalAcceleration::from_bspline,
            nb::arg("coeffs"), nb::arg("represents"),
            nb::arg("frame") = ballistic::forces::EmpiricalAccelFrame::GCRS,
            "Build from BSplineCoefficients, used directly (no conversion to a table). "
            "`represents` (EARepresents) says whether the spline is the acceleration itself "
            "or a velocity / position whose derivative gives it.")
        .def("accel_at", &ballistic::forces::EmpiricalAcceleration::accel_at,
            nb::arg("t_seconds"),
            "Acceleration at t_seconds after the reference epoch, numpy (3,) [m/s^2], in the "
            "frame the EA is stored in.")
        .def("evaluate_at",
            [](const ballistic::forces::EmpiricalAcceleration& self,
               Eigen::Ref<const Eigen::VectorXd> times_s)
            -> Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
            {
                const Eigen::Index N = times_s.size();
                Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> out(N, 3);
                for (Eigen::Index i = 0; i < N; ++i) {
                    ballistic::Vector3d a = self.accel_at(times_s[i]);
                    out.row(i) = a.transpose();
                }
                return out;
            },
            nb::arg("times_s"),
            "accel_at for a numpy (N,) array of times [s after the reference epoch]: returns "
            "a numpy (N, 3) array [m/s^2] in the stored frame.")
        .def("is_zero", &ballistic::forces::EmpiricalAcceleration::is_zero,
            "True if this is the zero EA (no correction).")
        .def("frame", &ballistic::forces::EmpiricalAcceleration::frame,
            "Frame the acceleration vectors are stored in (EmpiricalAccelFrame).")
        .def("keys", &ballistic::forces::EmpiricalAcceleration::keys,
            nb::rv_policy::reference_internal,
            "Time points [s]: the entry times of a table; the start, the distinct interior knot "
            "times and the end of a B-spline. Empty for the zero EA.")
        .def("domain", &ballistic::forces::EmpiricalAcceleration::domain,
            "Time span (t_start, t_end) [s] over which the EA is defined; (0.0, 0.0) for the "
            "zero EA.")
        .def("as_bspline", [](const ballistic::forces::EmpiricalAcceleration& ea)
                -> std::optional<std::pair<std::vector<double>, int>> {
                auto view = ea.as_bspline();
                if (!view) return std::nullopt;
                return std::make_pair(view->knots, view->degree);
            },
            "Tuple (knots, degree) if the EA was built with from_bspline(), else None. Tells "
            "whether an EA returned by estimate_ea_bspline is a B-spline and of what degree "
            "(Gauss-Newton refinement by finite differences converts it to a table with cubic "
            "interpolation for degree > 3, linear otherwise; the STM path always uses linear).");

    // There is no separate wind-table binding: wind is a column of a
    // DragEnvironment (atmosphere::WindProfile1D_ENU, atmosphere/wind_profile.hpp),
    // built by the from_models_* generators or by hand.

}
