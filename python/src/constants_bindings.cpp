// Physical constants and geodetic conversions for Python, from the same
// headers C++ uses (core/constants.hpp, coordinates/geodetic.hpp), so
// scripts never need their own copies of GM (the gravitational constant times
// the Earth's mass), the Earth radius or the rotation rate (static rule
// CONST-LITERAL).

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <ballistic/core/constants.hpp>
#include <ballistic/core/types.hpp>
#include <ballistic/coordinates/geodetic.hpp>

namespace nb = nanobind;
using namespace ballistic;

void bind_constants(nb::module_& m) {
    m.def("_constants_table", [] {
        namespace C = constants;
        nb::dict earth;
        earth["GM"] = C::earth::GM;
        earth["RADIUS_EQUATORIAL"] = C::earth::RADIUS_EQUATORIAL;
        earth["RADIUS_POLAR"] = C::earth::RADIUS_POLAR;
        earth["FLATTENING"] = C::earth::FLATTENING;
        earth["ECCENTRICITY_SQ"] = C::earth::ECCENTRICITY_SQ;
        earth["ANGULAR_VELOCITY"] = C::earth::ANGULAR_VELOCITY;
        earth["EGM2008_GM"] = C::earth::EGM2008_GM;
        earth["EGM2008_RADIUS"] = C::earth::EGM2008_RADIUS;
        nb::list jn;
        for (double j : C::earth::J_COEFFS) jn.append(j);
        earth["J_COEFFS"] = jn;
        earth["J2"] = C::earth::J2;
        earth["J3"] = C::earth::J3;
        earth["J4"] = C::earth::J4;

        nb::dict sun;
        sun["GM"] = C::sun::GM;
        sun["AU"] = C::sun::AU;
        nb::dict moon;
        moon["GM"] = C::moon::GM;
        nb::dict jupiter;
        jupiter["GM"] = C::jupiter::GM;

        nb::dict top;
        top["SPEED_OF_LIGHT"] = C::SPEED_OF_LIGHT;
        top["SECONDS_PER_DAY"] = C::SECONDS_PER_DAY;
        top["DEG_TO_RAD"] = C::DEG_TO_RAD;
        top["RAD_TO_DEG"] = C::RAD_TO_DEG;
        top["J2000_JD"] = C::J2000_JD;
        top["MJD_OFFSET"] = C::MJD_OFFSET;
        top["earth"] = earth;
        top["sun"] = sun;
        top["moon"] = moon;
        top["jupiter"] = jupiter;
        return top;
    }, "Internal: the library's physical constants as nested dicts (SI units); "
       "ballistic.constants is built from this.");

    nb::class_<GeodeticCoord>(m, "GeodeticCoord",
        "Geodetic coordinates on the WGS84 ellipsoid: latitude and longitude in "
        "radians, altitude in meters above the ellipsoid.")
        .def(nb::init<double, double, double>(),
             nb::arg("latitude"), nb::arg("longitude"), nb::arg("altitude"),
             "From latitude and longitude [rad] and altitude [m]")
        .def_rw("latitude", &GeodeticCoord::latitude, "[rad], positive north")
        .def_rw("longitude", &GeodeticCoord::longitude, "[rad], positive east")
        .def_rw("altitude", &GeodeticCoord::altitude, "[m] above the WGS84 ellipsoid")
        .def_prop_ro("latitude_deg", [](const GeodeticCoord& g) { return g.latitude * constants::RAD_TO_DEG; },
                     "Latitude [deg]")
        .def_prop_ro("longitude_deg", [](const GeodeticCoord& g) { return g.longitude * constants::RAD_TO_DEG; },
                     "Longitude [deg]")
        .def("__repr__", [](const GeodeticCoord& g) {
            char buf[160];
            std::snprintf(buf, sizeof buf, "GeodeticCoord(lat=%.6f deg, lon=%.6f deg, alt=%.3f m)",
                          g.latitude * constants::RAD_TO_DEG, g.longitude * constants::RAD_TO_DEG,
                          g.altitude);
            return std::string(buf);
        });

    m.def("ecef_to_geodetic",
        [](const Vector3d& pos_ecef) { return coords::ecef_to_geodetic(pos_ecef); },
        nb::arg("pos_ecef"),
        "Convert an Earth-fixed (ECEF) position, numpy (3,) [m], to a GeodeticCoord: "
        "geodetic latitude and longitude [rad] and altitude [m] above the WGS84 ellipsoid "
        "(exact closed form).");

    m.def("geodetic_to_ecef",
        [](double latitude, double longitude, double altitude) {
            return coords::geodetic_to_ecef(latitude, longitude, altitude);
        },
        nb::arg("latitude"), nb::arg("longitude"), nb::arg("altitude"),
        "Convert geodetic latitude and longitude [rad] and altitude [m] above the WGS84 "
        "ellipsoid to an Earth-fixed (ECEF) position, numpy (3,) [m].");
}
