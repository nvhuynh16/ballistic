// =============================================================================
// EOF Parser + Propagation-vs-Truth RMSE Python Bindings
// =============================================================================
//
// Exposes the validation helpers of include/ballistic/validation/
// eof_parser.hpp so Python users can read ESA Sentinel orbit files (EOF, an
// XML format; the precise orbit ephemeris, POE, files are of this type) and
// compute the root-mean-square error (RMSE) of a propagated trajectory
// against them.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>

#include <ballistic/validation/eof_parser.hpp>
#include <ballistic/core/defaults.hpp>

#include <sstream>

namespace nb = nanobind;

void bind_validation(nb::module_& m) {
    using namespace ballistic;
    using namespace ballistic::validation;

    // ----- EOFStateVector ----------------------------------------------------
    nb::class_<EOFStateVector>(m, "EOFStateVector",
        "One orbit state from an ESA EOF file (Earth-fixed coordinates, UTC time tags).")
        .def(nb::init<>())
        .def_rw("epoch", &EOFStateVector::epoch, "Time of the state (Epoch)")
        .def_rw("position", &EOFStateVector::position, "Earth-fixed (ECEF) position [m]")
        .def_rw("velocity", &EOFStateVector::velocity,
            "Velocity relative to the rotating Earth (ECEF) [m/s]")
        .def_rw("orbit_number", &EOFStateVector::orbit_number, "Absolute orbit number")
        .def_rw("quality", &EOFStateVector::quality, "Quality flag (e.g. NOMINAL)");

    // ----- EOFFile -----------------------------------------------------------
    nb::class_<EOFFile>(m, "EOFFile",
        "A parsed ESA EOF orbit file: its state vectors and metadata.")
        .def(nb::init<>())
        .def_rw("filename", &EOFFile::filename, "Name of the file read")
        .def_rw("mission", &EOFFile::mission, "e.g. 'Sentinel-1A'")
        .def_rw("reference_frame", &EOFFile::reference_frame,
            "Frame label of the file, e.g. 'EARTH_FIXED' (ECEF)")
        .def_rw("validity_start", &EOFFile::validity_start,
            "Start of the file's validity period (Epoch)")
        .def_ro("skipped_records", &EOFFile::skipped_records,
                "Number of state vectors that could not be read (reported once as a "
                "DataSkippedWarning)")
        .def_rw("validity_stop", &EOFFile::validity_stop,
            "End of the file's validity period (Epoch)")
        .def_rw("state_vectors", &EOFFile::state_vectors, "List of EOFStateVector, in time order")
        .def("find_nearest", &EOFFile::find_nearest, nb::arg("target"),
            "The state vector nearest to the Epoch `target`, or None if none is within 60 s.")
        .def("get_range", &EOFFile::get_range, nb::arg("start"), nb::arg("end"),
            "The state vectors with start <= epoch <= end.")
        .def("interpolate_position", &EOFFile::interpolate_position,
            nb::arg("target"), nb::arg("order") = defaults::LAGRANGE_DEFAULT_ORDER,
            "Earth-fixed position [m] at the Epoch `target`, by Lagrange-polynomial "
            "interpolation through `order` state vectors; None if there are too few, or "
            "outside the data (no extrapolation).");

    // ----- EOFParser ---------------------------------------------------------
    nb::class_<EOFParser>(m, "EOFParser",
        "Reader for ESA EOF orbit files.")
        .def_static("parse", &EOFParser::parse, nb::arg("filename"),
            "Read an .EOF file by path; returns an EOFFile, or None if the file cannot be "
            "opened or holds no readable state vector. Raises BallisticError if the validity "
            "period is missing or malformed.")
        .def_static("parse_content", &EOFParser::parse_content,
            nb::arg("content"), nb::arg("filename") = std::string{},
            "Read EOF XML content from a string (filename is used in messages only).");

    // ----- PropagationTestResult --------------------------------------------
    nb::class_<PropagationTestResult>(m, "PropagationTestResult",
        "Error statistics of a propagated trajectory compared with an EOF file.")
        .def(nb::init<>())
        .def_rw("start_epoch", &PropagationTestResult::start_epoch, "First compared time (Epoch)")
        .def_rw("end_epoch", &PropagationTestResult::end_epoch, "Last compared time (Epoch)")
        .def_rw("duration_seconds", &PropagationTestResult::duration_seconds,
            "Time span compared [s]")
        .def_rw("num_comparisons", &PropagationTestResult::num_comparisons,
            "Number of compared samples")
        .def_rw("position_rmse", &PropagationTestResult::position_rmse,
            "Root-mean-square position error [m]")
        .def_rw("position_max_error", &PropagationTestResult::position_max_error,
            "Largest position error [m]")
        .def_rw("position_mean_error", &PropagationTestResult::position_mean_error,
            "Mean position error [m]")
        .def_rw("velocity_rmse", &PropagationTestResult::velocity_rmse,
            "Root-mean-square velocity error [m/s]")
        .def_rw("velocity_max_error", &PropagationTestResult::velocity_max_error,
            "Largest velocity error [m/s]")
        .def_rw("along_track_rms", &PropagationTestResult::along_track_rms,
            "RMS error along the direction of motion [m]")
        .def_rw("cross_track_rms", &PropagationTestResult::cross_track_rms,
            "RMS error perpendicular to the orbit plane [m]")
        .def_rw("radial_rms", &PropagationTestResult::radial_rms,
            "RMS error in the up/down (radial) direction [m]")
        .def("__repr__", [](const PropagationTestResult& r) {
            std::ostringstream os;
            os << r;
            return os.str();
        });

    // ----- free functions ----------------------------------------------------
    m.def("parse_eof_auto", &parse_eof_auto, nb::arg("path"),
        "Read an EOF file, unpacking a .EOF.zip first if needed.");
    m.def("find_and_parse_eof", &find_and_parse_eof,
        nb::arg("directory"), nb::arg("pattern") = std::string{},
        "Find and read the first .EOF (or .EOF.zip) file in `directory` whose name contains "
        "`pattern` (empty = any; files already unpacked first, then .EOF.zip); None if "
        "nothing matches or parses.");
    m.def("compute_propagation_result", &compute_propagation_result,
        nb::arg("predicted"), nb::arg("truth"),
        nb::arg("max_time_diff") = defaults::VALIDATION_MAX_TIME_DIFF,
        R"doc(
Error statistics of a propagated trajectory against an EOF orbit file.

Args:
    predicted: list of (Epoch, StateVector), your propagated samples,
        Earth-fixed (ECEF).
    truth: an EOFFile (ECEF). Each predicted sample is compared with the
        nearest truth state within max_time_diff seconds.
    max_time_diff: largest time mismatch accepted for a comparison [s].

Returns:
    PropagationTestResult (position and velocity RMSE, and the error split
    into radial, along-track and cross-track directions).
)doc");
}
