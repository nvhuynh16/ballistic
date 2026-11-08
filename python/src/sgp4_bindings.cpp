// =============================================================================
// TLE / SGP4 / TEME bindings (sgp4/*.hpp). Python: ballistic.tle
// (parse, Sgp4, teme_to_itrf, itrf_to_teme) and State.from_tle / from_teme.
//
// A TLE (two-line element set) is the standard public text format for a
// satellite's orbit, published e.g. by CelesTrak. Its elements are "mean"
// elements that only make sense with the SGP4 propagator (Simplified General
// Perturbations 4; SDP4 is its deep-space variant for periods over 225
// minutes). SGP4 outputs states in TEME (True Equator, Mean Equinox), a
// quasi-inertial frame of its own.
// =============================================================================

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>

#include <ballistic/core/errors.hpp>
#include <ballistic/sgp4/sgp4.hpp>
#include <ballistic/sgp4/teme.hpp>
#include <ballistic/sgp4/tle.hpp>
#include <ballistic/presets/tle_grade.hpp>
#include <nanobind/stl/optional.h>

#include <cstdio>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace ballistic;
using namespace ballistic::sgp4;

namespace {

GravityConstants gravity_from(const std::string& s) {
    if (s == "wgs72") return GravityConstants::WGS72;
    if (s == "wgs72old") return GravityConstants::WGS72Old;
    if (s == "wgs84") return GravityConstants::WGS84;
    throw ConfigError("gravity must be 'wgs72' (what element sets are fitted with), 'wgs72old' "
                      "or 'wgs84', got '" + s + "'");
}

OpsMode mode_from(const std::string& s) {
    if (s == "improved") return OpsMode::Improved;
    if (s == "afspc") return OpsMode::Afspc;
    throw ConfigError("mode must be 'improved' or 'afspc', got '" + s + "'");
}

}  // namespace

void bind_sgp4(nb::module_& m) {
    nb::class_<Tle>(m, "Tle",
        "A parsed two-line element set (TLE). Its elements are SGP4 mean elements: feed them "
        "to Sgp4, not to a numerical propagator (read as ordinary orbital elements they are "
        "kilometers off).")
        .def_static("parse", &parse_tle, "line1"_a, "line2"_a,
                    "Parse a TLE strictly: 69 columns, valid checksums, the same catalog number "
                    "on both lines, blanks where the format has separators, well-formed fields "
                    "(Alpha-5 catalog numbers accepted). Raises InputPlausibilityError naming the line "
                    "and column at fault.")
        .def_ro("line1", &Tle::line1, "Line 1 as given (trailing blanks removed)")
        .def_ro("line2", &Tle::line2, "Line 2 as given (trailing blanks removed)")
        .def_ro("catalog_number", &Tle::catalog_number,
                "Satellite catalog number (int; Alpha-5 decoded, A0001 -> 100001)")
        .def_prop_ro("classification", [](const Tle& t) { return std::string(1, t.classification); },
                "'U' unclassified, 'C' classified, 'S' secret")
        .def_ro("international_designator", &Tle::international_designator,
                "Launch year, launch number and piece, e.g. '98067A' (may be blank)")
        .def_ro("epoch_year", &Tle::epoch_year, "Epoch year, four digits")
        .def_ro("epoch_day", &Tle::epoch_day,
                "Epoch day of year plus fraction (1.0 = 1 January 00:00 UTC)")
        .def_ro("ndot_rev_day2", &Tle::ndot_rev_day2,
                "First time derivative of the mean motion, divided by 2 [rev/day^2] (not used "
                "by SGP4)")
        .def_ro("nddot_rev_day3", &Tle::nddot_rev_day3,
                "Second time derivative of the mean motion, divided by 6 [rev/day^3] (not used "
                "by SGP4)")
        .def_ro("bstar_per_er", &Tle::bstar_per_er,
                "B*, SGP4's drag term [1 / Earth radii]")
        .def_ro("ephemeris_type", &Tle::ephemeris_type,
                "Ephemeris type (0 in distributed element sets)")
        .def_ro("element_set_number", &Tle::element_set_number, "Element set number")
        .def_ro("inclination_deg", &Tle::inclination_deg,
                "Inclination: tilt of the orbit plane to the equator [deg]")
        .def_ro("raan_deg", &Tle::raan_deg, "Right ascension of the ascending node [deg]")
        .def_ro("eccentricity", &Tle::eccentricity, "Eccentricity (0 = circular)")
        .def_ro("argp_deg", &Tle::argp_deg, "Argument of perigee [deg]")
        .def_ro("mean_anomaly_deg", &Tle::mean_anomaly_deg, "Mean anomaly [deg]")
        .def_ro("mean_motion_rev_day", &Tle::mean_motion_rev_day,
                "Mean motion (Kozai convention) [revolutions per day]")
        .def_ro("revolution_number", &Tle::revolution_number,
                "Number of orbits completed at the epoch")
        .def_prop_ro("epoch", &Tle::epoch,
                "The element-set epoch (Epoch, UTC). Raises DataCoverageError for epoch years "
                "1957-1960, before UTC existed.")
        .def("epoch_days_since_1950", &Tle::epoch_days_since_1950,
             "Days since 1949 December 31 00:00 UTC (the time argument of SGP4's lunar-solar "
             "and Earth-rotation terms).")
        .def("beta_from_bstar_kg_m2", &Tle::beta_from_bstar_kg_m2,
             "Ballistic coefficient m / (Cd*A) [kg/m^2] implied by B*: 1 / (12.741621 B*). B* "
             "is a fitted term that also absorbs other model errors; it can be zero (result "
             "infinite) or negative (result negative).")
        .def("__repr__", [](const Tle& t) {
            return "Tle(catalog_number=" + std::to_string(t.catalog_number) + ", epoch=" +
                   std::to_string(t.epoch_year) + "/" + std::to_string(t.epoch_day) + ")";
        });

    m.def("tle_checksum", &tle_checksum, "line"_a,
          "The TLE checksum of a line: digits summed, each '-' counting 1, over columns 1-68, "
          "modulo 10.");

    nb::class_<Sgp4>(m, "Sgp4",
        "The SGP4/SDP4 propagator for one element set; outputs TEME states. "
        "Sgp4(tle, gravity='wgs72', mode='improved'): gravity picks the Earth constants "
        "('wgs72', what element sets are fitted with; 'wgs72old'; 'wgs84'), mode picks "
        "'improved' or the original 'afspc' operation. Raises ConfigError for other values.")
        .def("__init__",
             [](Sgp4* self, const Tle& tle, const std::string& gravity, const std::string& mode) {
                 new (self) Sgp4(tle, Sgp4Options{gravity_from(gravity), mode_from(mode)});
             },
             "tle"_a, nb::kw_only(), "gravity"_a = "wgs72", "mode"_a = "improved")
        .def_prop_ro("tle", &Sgp4::tle, "The element set (Tle)")
        .def_prop_ro("deep_space", &Sgp4::deep_space,
                "True if the deep-space model (SDP4; orbital period of 225 minutes or more) "
                "is used")
        .def("propagate_minutes",
             [](const Sgp4& s, double t) {
                 const auto o = s.propagate_minutes(t);
                 return nb::make_tuple(o.r_km, o.v_km_s, static_cast<int>(o.status));
             },
             "tsince_min"_a,
             "State tsince_min minutes after the element-set epoch: a tuple (TEME position "
             "[km], velocity [km/s], SGP4 error code). Error code 0 = ok; 1-4 = the model "
             "failed (state is NaN); 6 = the satellite has decayed (state still returned).")
        .def("propagate_minutes_array",
             [](const Sgp4& s, nb::ndarray<const double, nb::ndim<1>, nb::c_contig> t) {
                 const std::size_t n = t.shape(0);
                 auto* r = new double[3 * n];
                 auto* v = new double[3 * n];
                 auto* e = new int[n];
                 for (std::size_t i = 0; i < n; ++i) {
                     const auto o = s.propagate_minutes(t(i));
                     for (int k = 0; k < 3; ++k) { r[3 * i + k] = o.r_km[k]; v[3 * i + k] = o.v_km_s[k]; }
                     e[i] = static_cast<int>(o.status);
                 }
                 nb::capsule cr(r, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                 nb::capsule cv(v, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                 nb::capsule ce(e, [](void* p) noexcept { delete[] static_cast<int*>(p); });
                 return nb::make_tuple(
                     nb::ndarray<nb::numpy, double, nb::shape<-1, 3>>(r, {n, 3}, cr),
                     nb::ndarray<nb::numpy, double, nb::shape<-1, 3>>(v, {n, 3}, cv),
                     nb::ndarray<nb::numpy, int, nb::shape<-1>>(e, {n}, ce));
             },
             "tsince_min"_a,
             "propagate_minutes over a numpy (N,) array of minutes: returns (r_km (N,3), "
             "v_km_s (N,3), status (N,)).")
        .def("teme",
             [](const Sgp4& s, const Epoch& t) {
                 const StateVector x = s.teme(t);
                 return nb::make_tuple(x.position, x.velocity);
             },
             "epoch"_a,
             "TEME position [m] and velocity [m/s] at an Epoch, as a tuple; raises when the "
             "model fails there.");

    m.def("teme_to_itrf",
          [](const Epoch& t, const Vector3d& r, const Vector3d& v) {
              const StateVector x = teme_to_itrf(StateVector(r, v), t);
              return nb::make_tuple(x.position, x.velocity);
          },
          "epoch"_a, "position_m"_a, "velocity_m_s"_a,
          "Convert a TEME state (position [m], velocity [m/s]) to the Earth-fixed ITRF "
          "(velocity relative to the rotating Earth): rotation by the 1982 Greenwich mean "
          "sidereal time of UT1, then polar motion, both from the loaded Earth orientation "
          "data (EOP). Returns (position, velocity).");
    m.def("itrf_to_teme",
          [](const Epoch& t, const Vector3d& r, const Vector3d& v) {
              const StateVector x = itrf_to_teme(StateVector(r, v), t);
              return nb::make_tuple(x.position, x.velocity);
          },
          "epoch"_a, "position_m"_a, "velocity_m_s"_a,
          "Convert an Earth-fixed ITRF state (position [m], velocity relative to the rotating "
          "Earth [m/s]) to TEME; the inverse of teme_to_itrf.");

    // ---- TLE-grade preset (presets/tle_grade.hpp) ---------------------------
    using presets::TleGradePreset;
    nb::class_<TleGradePreset>(m, "TleGradePreset",
        "Which forces still matter when only TLE-level accuracy is needed, for one orbit and "
        "prediction horizon. A force is kept if its measured effect exceeds 10% of the "
        "typical TLE error for the orbit class (Flohrer et al. 2009). The table behind it "
        "was measured by scripts/measure_tle_grade_table.py.")
        .def_ro("regime", &TleGradePreset::regime,
                "Orbit class: 'LEO' (low Earth orbit), 'MEO' (medium Earth orbit) or 'GEO' "
                "(geostationary)")
        .def_ro("horizon_s", &TleGradePreset::horizon_s,
                "The measured horizon used [s] (at least the one requested)")
        .def_ro("rows_altitude_km_min", &TleGradePreset::rows_altitude_km_min,
                "Lowest altitude of the measured table rows merged [km]")
        .def_ro("rows_altitude_km_max", &TleGradePreset::rows_altitude_km_max,
                "Highest altitude of the measured table rows merged [km]")
        .def_ro("budget_m", &TleGradePreset::budget_m,
                "Typical TLE error for the orbit class [m] (Flohrer et al. 2009)")
        .def_ro("threshold_m", &TleGradePreset::threshold_m,
                "Keep-threshold: 10% of the budget [m]")
        .def_ro("gravity_degree", &TleGradePreset::gravity_degree,
                "Gravity-model degree to use")
        .def_ro("drag", &TleGradePreset::drag, "Keep atmospheric drag")
        .def_ro("radiation", &TleGradePreset::radiation, "Keep radiation pressure")
        .def_ro("third_body", &TleGradePreset::third_body, "Keep Sun/Moon/Jupiter gravity")
        .def_ro("tides", &TleGradePreset::tides, "Keep solid-Earth tides")
        .def_ro("ocean_tides", &TleGradePreset::ocean_tides, "Keep ocean tides")
        .def_ro("relativity", &TleGradePreset::relativity, "Keep relativistic corrections")
        .def_ro("effect_drag_m", &TleGradePreset::effect_drag_m,
                "Largest measured effect of drag over the horizon [m]")
        .def_ro("effect_radiation_m", &TleGradePreset::effect_radiation_m,
                "Largest measured effect of radiation pressure [m]")
        .def_ro("effect_third_body_m", &TleGradePreset::effect_third_body_m,
                "Largest measured effect of third-body gravity [m]")
        .def_ro("effect_tides_m", &TleGradePreset::effect_tides_m,
                "Largest measured effect of solid-Earth tides [m]")
        .def_ro("effect_ocean_tides_m", &TleGradePreset::effect_ocean_tides_m,
                "Largest measured effect of ocean tides [m]")
        .def_ro("effect_relativity_m", &TleGradePreset::effect_relativity_m,
                "Largest measured effect of relativity [m]")
        .def_ro("combined_error_m", &TleGradePreset::combined_error_m,
                "Largest measured difference between the whole preset and the full "
                "configuration [m]")
        .def_ro("speedup", &TleGradePreset::speedup,
                "Smallest measured speed-up over the full configuration (wall time)")
        .def("__repr__", [](const TleGradePreset& p) {
            char b[256];
            std::snprintf(b, sizeof b,
                          "TleGradePreset(%s, horizon %.0f s, degree %d, drag %d, radiation %d, "
                          "third_body %d, tides %d, ocean_tides %d, relativity %d, x%.1f)",
                          p.regime.c_str(), p.horizon_s, p.gravity_degree, p.drag, p.radiation,
                          p.third_body, p.tides, p.ocean_tides, p.relativity, p.speedup);
            return std::string(b);
        });
    m.def("tle_grade_preset",
          [](const InitialState& s, double horizon_s) { return presets::tle_grade(s, horizon_s); },
          "state"_a, "horizon_s"_a,
          "The TLE-grade preset (TleGradePreset) for this start State and prediction horizon "
          "[s]. Raises ConfigError outside what was measured (perigee below 300 km, "
          "geostationary-transfer or other highly elliptical orbits, medium orbits below "
          "5000 km, horizons over 7 days).");
    m.def("_tle_grade_rows", []() {
        nb::list out;
        for (const auto& r : presets::kTleGradeTable) {
            nb::dict d;
            d["regime"] = std::string(r.regime);
            d["altitude_km"] = r.altitude_km;
            d["horizon_s"] = r.horizon_s;
            d["budget_m"] = r.budget_m;
            d["threshold_m"] = r.threshold_m;
            d["gravity_degree"] = r.gravity_degree;
            d["drag"] = r.drag; d["radiation"] = r.radiation; d["third_body"] = r.third_body;
            d["tides"] = r.tides; d["ocean_tides"] = r.ocean_tides; d["relativity"] = r.relativity;
            d["effect_drag_m"] = r.effect_drag_m;
            d["effect_radiation_m"] = r.effect_radiation_m;
            d["effect_third_body_m"] = r.effect_third_body_m;
            d["effect_tides_m"] = r.effect_tides_m;
            d["effect_ocean_tides_m"] = r.effect_ocean_tides_m;
            d["effect_relativity_m"] = r.effect_relativity_m;
            d["combined_error_m"] = r.combined_error_m;
            d["speedup"] = r.speedup;
            out.append(d);
        }
        return out;
    }, "Internal: the measured TLE-grade table (presets/tle_grade_table.hpp), one dict per "
       "row.");
    m.def("_tle_grade_preset_for",
          [](double perigee_km, double apogee_km, double horizon_s) {
              return presets::tle_grade(perigee_km, apogee_km, horizon_s);
          }, "perigee_km"_a, "apogee_km"_a, "horizon_s"_a,
          "Internal: the TLE-grade preset for perigee / apogee altitudes [km] and a horizon "
          "[s].");
}
