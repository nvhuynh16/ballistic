// =============================================================================
// SP3 reader bindings (validation/sp3.hpp). Python: ballistic.sp3.read(path),
// which decompresses .gz and calls _sp3_read_string.
//
// SP3 is the standard text format for precise satellite orbits (positions,
// optionally velocities, at regular epochs), used by GNSS and laser-ranging
// services and by satellite operators.
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

#include <ballistic/validation/sp3.hpp>

#include <cctype>
#include <cstdint>
#include <vector>

namespace nb = nanobind;

namespace {

using namespace ballistic::validation::sp3;

nb::ndarray<nb::numpy, double, nb::shape<-1, 3>> rows3(const std::vector<ballistic::Vector3d>& v) {
    auto* data = new double[v.size() * 3];
    for (std::size_t i = 0; i < v.size(); ++i)
        for (int k = 0; k < 3; ++k) data[3 * i + k] = v[i][k];
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return {data, {v.size(), 3}, owner};
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::shape<-1>> vec1(std::vector<T> v) {
    auto* data = new T[v.size()];
    std::copy(v.begin(), v.end(), data);
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return {data, {v.size()}, owner};
}

}  // namespace

void bind_sp3(nb::module_& m) {
    nb::enum_<TimeSystem>(m, "SP3TimeSystem",
        "Time system an SP3 file's epochs are written in: GPS, Galileo (GAL), QZSS (QZS) "
        "and NavIC (IRN) system times, BeiDou time (BDT, GPS - 14 s), TAI, UTC, or "
        "GLONASS time (GLO).")
        .value("GPS", TimeSystem::GPS).value("GAL", TimeSystem::GAL)
        .value("QZS", TimeSystem::QZS).value("IRN", TimeSystem::IRN)
        .value("BDT", TimeSystem::BDT).value("TAI", TimeSystem::TAI)
        .value("UTC", TimeSystem::UTC).value("GLO", TimeSystem::GLO);

    nb::class_<FrameInfo>(m, "SP3FrameInfo",
        "What an SP3 file's coordinate-system label means: Earth-fixed or inertial, and for "
        "an Earth-fixed label the ITRF version it names (None for labels that do not state "
        "one, such as GFZ's 'CTS').")
        .def_ro("inertial", &FrameInfo::inertial,
                "True for an inertial label (J2000, J2K, EME00), False for Earth-fixed")
        .def_ro("itrf", &FrameInfo::itrf,
                "ITRF version (ITRFRealization) of an Earth-fixed label, or None");

    nb::class_<Header>(m, "SP3Header", "The header of an SP3 file.")
        .def_ro("version", &Header::version, "Format version letter, 'a' to 'd'")
        .def_ro("pos_or_vel", &Header::pos_or_vel,
                "'P' = positions only, 'V' = positions and velocities")
        .def_ro("frame", &Header::frame, "Coordinate-system label, e.g. 'IGS20', 'J2000', 'CTS'")
        .def_ro("orbit_type", &Header::orbit_type, "Producer's orbit-type code, e.g. 'FIT'")
        .def_ro("agency", &Header::agency, "Producing agency code")
        .def_ro("time_system", &Header::time_system, "Time system of the epochs (SP3TimeSystem)")
        .def_ro("interval_s", &Header::interval_s, "Nominal spacing between epochs [s]")
        .def_ro("declared_epochs", &Header::declared_epochs,
                "Number of epochs stated in the first header line")
        .def_ro("satellites", &Header::satellites, "Satellite IDs listed in the header");

    nb::class_<Arc>(m, "SP3Arc",
        R"doc(All records of one satellite in one SP3 file, in the file's frame.

epoch0 is the first record's epoch (converted from the file's time system);
offsets_ns (int64) / offsets_s (float) are exact times of the records
after it. position is numpy (N, 3) [m]; velocity is numpy (N, 3) [m/s]
with NaN where the file has none.)doc")
        .def_ro("satellite", &Arc::satellite, "Satellite ID, e.g. 'G01'")
        .def_ro("epoch0", &Arc::epoch0, "Epoch of the first record")
        .def_ro("missing_positions", &Arc::missing_positions,
                "Number of records skipped because the file marks the position bad or absent")
        .def_ro("interval_s", &Arc::interval_s,
                "Nominal record spacing [s] (from the header, else the median spacing)")
        .def("__len__", &Arc::size, "Number of records")
        .def_prop_ro("offsets_ns", [](const Arc& a) { return vec1<std::int64_t>(a.offset_ns); },
                     nb::rv_policy::move)
        .def_prop_ro("offsets_s", [](const Arc& a) {
            std::vector<double> t(a.size());
            for (std::size_t i = 0; i < a.size(); ++i) t[i] = a.offset_s(i);
            return vec1<double>(std::move(t));
        }, nb::rv_policy::move)
        .def_prop_ro("position", [](const Arc& a) { return rows3(a.position_m); },
                     nb::rv_policy::move)
        .def_prop_ro("velocity", [](const Arc& a) -> nb::object {
            if (a.velocity_m_s.empty()) return nb::none();
            return nb::cast(rows3(a.velocity_m_s));
        }, nb::rv_policy::move)
        .def("has_velocity", &Arc::has_velocity, "True if every record carries a velocity.")
        .def("epoch", &Arc::epoch, nb::arg("index"), "Epoch of record `index`")
        .def("segments", &Arc::segments,
             "Index ranges (first, last) of the stretches without a gap (a gap: spacing more "
             "than 1.5 times the nominal interval).")
        .def("interpolate", [](const Arc& a, double t, int order) {
                auto s = a.interpolate(t, order);
                return std::make_pair(s.position_m, s.velocity_m_s);
            }, nb::arg("t_s"), nb::arg("order") = 9,
            "Tuple (position [m], velocity [m/s]) at t_s seconds after epoch0, by "
            "Lagrange-polynomial interpolation through `order` records inside one gap-free "
            "stretch. Raises DataCoverageError outside the arc or in a gap.")
        .def("derived_velocities", [](const Arc& a, int order) {
                auto d = a.derived_velocities(order);
                std::vector<std::uint8_t> c(d.centered.begin(), d.centered.end());
                auto centered = vec1<std::uint8_t>(std::move(c));
                return nb::make_tuple(rows3(d.velocity_m_s), centered);
            }, nb::arg("order") = 9,
            "Velocities computed by differentiating the positions (Lagrange polynomial "
            "through `order` records): tuple (velocity (N, 3) [m/s], centered (N,) uint8). "
            "centered = 0 where the window is one-sided (at the ends of a gap-free stretch, "
            "about 5e-4 m/s error); do not start a prediction there.");

    nb::class_<File>(m, "SP3File", "A parsed SP3 file: its header and one SP3Arc per satellite.")
        .def_ro("source", &File::source, "Path or name of the file, for messages")
        .def_ro("header", &File::header, "The SP3Header")
        .def_ro("arcs", &File::arcs,
                "List of SP3Arc, in order of first record (satellites with records only)")
        .def_ro("missing_positions", &File::missing_positions,
                "Number of bad or absent positions skipped, all satellites")
        .def("arc", &File::arc, nb::arg("satellite"), nb::rv_policy::reference_internal,
             "The SP3Arc of `satellite` (e.g. 'G01'); raises, listing the satellites "
             "present, if the file has no records for it")
        .def("frame", &File::frame,
             "What the header's coordinate-system label means (SP3FrameInfo); raises "
             "FrameError for an unknown label.");

    m.def("_sp3_read_string", &read_string, nb::arg("text"), nb::arg("source") = "<string>",
          nb::call_guard<nb::gil_scoped_release>(),
          "Internal: parse SP3 text (use ballistic.sp3.read). Raises on a malformed line, "
          "naming the line number.");
    m.def("sp3_classify_frame", &classify_frame, nb::arg("label"),
          "What an SP3 coordinate-system label means (SP3FrameInfo). Inertial: J2000, J2K, "
          "EME00. Earth-fixed labels map to an ITRF version (IGS05, IGb08, IGS14, ITR20, ...); "
          "'CTS', 'ITRF' and 'ITRF2' are Earth-fixed without a stated version. Raises "
          "FrameError for any other label rather than guessing.");
    m.def("_sp3_initial_state",
          [](const File& f, const std::string& satellite, const ballistic::Epoch& epoch,
             std::optional<ballistic::coords::ITRFRealization> realization,
             std::optional<std::string> inertial_as, bool allow_suborbital, bool check_frame) {
              InertialAs ia = InertialAs::Unspecified;
              if (inertial_as) {
                  std::string v = *inertial_as;
                  for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                  if (v == "gcrs") ia = InertialAs::GCRS;
                  else if (v == "j2000") ia = InertialAs::J2000;
                  else throw ballistic::ConfigError("inertial_as must be 'gcrs' or 'j2000', got '" +
                                                    *inertial_as + "'");
              }
              return initial_state(f, satellite, epoch, realization, ia,
                                   {allow_suborbital, check_frame});
          },
          nb::arg("file"), nb::arg("satellite"), nb::arg("epoch"), nb::arg("realization").none(),
          nb::arg("inertial_as").none(), nb::arg("allow_suborbital"), nb::arg("check_frame"),
          "Internal (use State.from_sp3): a checked start state from the arc of `satellite` "
          "at `epoch`; the velocity is derived from the positions. realization: ITRF version "
          "for an Earth-fixed label that does not state one; inertial_as: 'gcrs' or 'j2000' "
          "for an inertial label. Raises DataCoverageError near an arc edge or gap, "
          "FrameError when the frame is unknown or not specified.");
}
