// =============================================================================
// Atmosphere (DragEnvironment / SpaceWeatherHistory / generators) bindings
// =============================================================================
//
// Called from NB_MODULE in bindings.cpp; registration order is set by the
// call order there.
//
// Terms used below: the ballistic coefficient beta = m / (Cd * A) [kg/m^2]
// (mass over drag coefficient times area: how strongly drag slows a body);
// F10.7 (the Sun's 10.7 cm radio flux) and Ap (a geomagnetic activity index)
// are the space-weather inputs the atmosphere models need; NRLMSIS 2.1 is the
// atmospheric density/temperature model and HWM14 the wind model; an
// empirical acceleration (EA) is a small correction acceleration fitted to
// observations; ECEF / ITRF = Earth-fixed frame, GCRS = inertial frame.
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
#include <nanobind/stl/variant.h>

namespace nb = nanobind;

void bind_atmosphere(nb::module_& m) {
    // =========================================================================
    // Enums needed by types below (must be registered before default args)
    // =========================================================================

    nb::enum_<ballistic::forces::EmpiricalAccelFrame>(m, "EmpiricalAccelFrame",
        "Coordinate frame in which empirical acceleration (EA) vectors are stored.")
        .value("GCRS", ballistic::forces::EmpiricalAccelFrame::GCRS,
            "Geocentric Celestial Reference System: inertial, non-rotating (default)")
        .value("ITRF", ballistic::forces::EmpiricalAccelFrame::ITRF,
            "International Terrestrial Reference Frame: Earth-fixed, rotates with the Earth");

    nb::enum_<ballistic::forces::EAInterp>(m, "EAInterp",
        "How a tabulated empirical acceleration is interpolated between its time points.")
        .value("Cubic", ballistic::forces::EAInterp::Cubic, "Cubic spline (default)")
        .value("Linear", ballistic::forces::EAInterp::Linear, "Piecewise-linear");

    nb::enum_<ballistic::forces::EARepresents>(m, "EARepresents",
        "What quantity a B-spline empirical acceleration was fitted to, which decides how "
        "many times the spline is differentiated to give the acceleration.")
        .value("Acceleration", ballistic::forces::EARepresents::Acceleration,
            "The spline's value is the acceleration (no differentiation)")
        .value("Velocity", ballistic::forces::EARepresents::Velocity,
            "The spline's first time derivative is the acceleration")
        .value("Position", ballistic::forces::EARepresents::Position,
            "The spline's second time derivative is the acceleration");

    nb::class_<ballistic::forces::BSplineCoefficients>(m, "BSplineCoefficients",
        "B-spline coefficients for EmpiricalAcceleration.from_bspline(): one coefficient "
        "array per axis (cx, cy, cz, numpy (n,)), the knot vector (time points [s]) and the "
        "polynomial degree. Raises ValueError if the array sizes do not fit together "
        "(knots must have n + degree + 1 entries).")
        .def(nb::init<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd,
                      std::vector<double>, int>(),
            nb::arg("cx"), nb::arg("cy"), nb::arg("cz"),
            nb::arg("knots"), nb::arg("degree"))
        .def("n_basis", &ballistic::forces::BSplineCoefficients::n_basis,
            "Number of basis functions (= len(cx)).");

    // =========================================================================
    // Altitude-Dependent Cd Table — REMOVED.
    // Use BetaProfile1D (atmosphere/beta_profile.hpp) and embed it in a
    // DragEnvironment via the from_models_* generators or hand-construct.
    // =========================================================================


    // =========================================================================
    // DragEnvironment + SpaceWeatherHistory + generators
    // =========================================================================
    //
    // Variants on DragEnvironment (DensityProfile1D / WindProfile1D_ENU /
    // BetaProfile1D / etc.) are hidden behind factory methods on the
    // DragEnvironment class — same pattern as EmpiricalAcceleration. Only
    // BetaProfile1D is exposed directly because users typically need to
    // construct one (it's a satellite property, not generator output).
    // =========================================================================

    nb::class_<ballistic::atmosphere::TimeWindow>(m, "TimeWindow",
        R"doc(
Time interval [start, end) (start included, end excluded), two Epoch
values. Used to bound where a SpaceWeatherHistory is valid.
)doc")
        .def(nb::init<ballistic::Epoch, ballistic::Epoch>(),
            nb::arg("start"), nb::arg("end"))
        .def_rw("start", &ballistic::atmosphere::TimeWindow::start, "First instant (Epoch, included)")
        .def_rw("end",   &ballistic::atmosphere::TimeWindow::end, "End (Epoch, excluded)");

    nb::class_<ballistic::atmosphere::SpaceWeatherHistory>(m, "SpaceWeatherHistory",
        R"doc(
Space weather over time: daily F10.7 solar-flux values and 3-hourly Ap
geomagnetic indices, the inputs the NRLMSIS 2.1 density model and the HWM14
wind model need. Used by the drag-environment builders.

SpaceWeatherHistory.from_loaded(start, end) builds the measured history from
the loaded CelesTrak space-weather file; this is what make_drag_environment
uses by default. from_constant(WeatherConditions) turns one fixed set of
values into a history valid at every epoch (e.g. a deliberate quiet-Sun or
storm scenario); the windowed form raises outside its window.
)doc")
        .def(nb::init<>(), "Empty history (at() raises).")
        .def_static("from_constant",
            static_cast<ballistic::atmosphere::SpaceWeatherHistory (*)(
                ballistic::WeatherConditions)>(
                &ballistic::atmosphere::SpaceWeatherHistory::from_constant),
            nb::arg("w"),
            "History holding the WeatherConditions `w` at every epoch.")
        .def_static("from_constant",
            static_cast<ballistic::atmosphere::SpaceWeatherHistory (*)(
                ballistic::WeatherConditions, ballistic::atmosphere::TimeWindow)>(
                &ballistic::atmosphere::SpaceWeatherHistory::from_constant),
            nb::arg("w"), nb::arg("window"),
            "History holding the WeatherConditions `w` inside `window` (a TimeWindow); "
            "at() raises outside it.")
        .def_static("last_measured_day",
            []() -> std::optional<ballistic::Epoch> {
                auto d = ballistic::atmosphere::SpaceWeatherData::instance().last_measured_mjd();
                if (!d) return std::nullopt;
                return ballistic::Epoch::from_mjd(*d, ballistic::time::TimeScale::UTC);
            },
            "Start (UTC) of the last day with measured (not forecast) weather in the loaded "
            "file, as an Epoch, or None if nothing is loaded. Drag past the end of that day "
            "raises unless forecasts are allowed.")
        .def_static("from_loaded",
            [](const ballistic::Epoch& start, const ballistic::Epoch& end,
               bool allow_predicted) {
                bool used = false;
                auto h = ballistic::atmosphere::SpaceWeatherData::instance().history(
                    ballistic::atmosphere::TimeWindow{start, end}, allow_predicted, &used);
                if (used && PyErr_WarnEx(PyExc_RuntimeWarning,
                        "SpaceWeatherHistory.from_loaded: measured space weather runs "
                        "out inside this window; CelesTrak's forecast fills the rest", 1) != 0)
                    throw nb::python_error();
                return h;
            },
            nb::arg("start"), nb::arg("end"), nb::arg("allow_predicted") = true,
            R"doc(
Measured, time-varying space weather between the Epochs start and end, from
the file loaded by ballistic.load() / load_space_weather(): the previous
day's observed F10.7, its 81-day centered mean, and the 3-hourly Ap values in
the order NRLMSIS 2.1 reads them. This is what make_drag_environment uses
when weather=None.

Past the last measured day, CelesTrak's daily forecast (about 45 days) fills
in, with a RuntimeWarning; allow_predicted=False refuses forecasts instead.

Raises:
    IndexError: a day the window needs (the model looks back 3 days) is not
        in the file, has no 3-hourly Ap, or is a forecast while
        allow_predicted=False. No quiet default values are ever substituted.
)doc")
        .def("at", &ballistic::atmosphere::SpaceWeatherHistory::at,
            nb::arg("epoch"),
            "The WeatherConditions (F10.7, Ap, ...) in effect at `epoch`.")
        .def("empty", &ballistic::atmosphere::SpaceWeatherHistory::empty,
            "True if the history holds no values.");

    nb::class_<ballistic::atmosphere::BetaProfile1D>(m, "BetaProfile1D",
        R"doc(
Ballistic coefficient as a function of altitude. The ballistic coefficient
beta = m / (Cd * A) [kg/m^2] (mass over drag coefficient times frontal area)
sets how strongly drag decelerates a body: the larger beta, the weaker the
effect. Used as the beta column of a DragEnvironment.

Build with BetaProfile1D(altitude_km, beta_kg_m2) (two equal-length lists:
altitudes [km] and beta values [kg/m^2]), or BetaProfile1D.constant(beta) /
from_mass_cd_area(...) for the common case of a beta that does not vary with
altitude.
)doc")
        .def(nb::init<>(), "Empty profile.")
        .def(nb::init<std::vector<double>, std::vector<double>>(),
            nb::arg("altitude_km"), nb::arg("beta_kg_m2"),
            "Profile through the points (altitude_km [km], beta_kg_m2 [kg/m^2]).")
        .def_static("constant",
            &ballistic::atmosphere::BetaProfile1D::constant,
            nb::arg("beta_kg_m2"),
            nb::arg("altitude_min_km") = 0.0,
            nb::arg("altitude_max_km") = 1500.0,
            "Constant beta [kg/m^2] between altitude_min_km and altitude_max_km [km].")
        .def_static("from_mass_cd_area",
            &ballistic::atmosphere::BetaProfile1D::from_mass_cd_area,
            nb::arg("mass_kg"), nb::arg("cd"), nb::arg("area_m2"),
            nb::arg("altitude_min_km") = 0.0,
            nb::arg("altitude_max_km") = 1500.0,
            "Constant beta = mass_kg / (cd * area_m2) [kg/m^2] from a satellite's mass [kg], "
            "drag coefficient and frontal area [m^2].")
        .def("empty", &ballistic::atmosphere::BetaProfile1D::empty,
            "True if the profile holds no values.");

    // ---- Flow-regime drag coefficient (aero/flow_regime.hpp) ---------------
    {
        namespace aero = ballistic::aero;
        nb::enum_<aero::Shape>(m, "AeroShape",
            "Body shape for FlowRegimeCd. SPHERE: reference area pi D^2/4 and reference "
            "length D (diameter). FLAT_PLATE: facing the flow; reference area = plate area. "
            "CYLINDER: axis across the flow; reference area L*D, reference length D.")
            .value("SPHERE", aero::Shape::Sphere)
            .value("FLAT_PLATE", aero::Shape::FlatPlate)
            .value("CYLINDER", aero::Shape::Cylinder);

        nb::class_<aero::ConstantCd>(m, "ConstantCd",
            "A drag coefficient (dimensionless) that does not vary with the flow.")
            .def(nb::init<double>(), nb::arg("cd"), "A constant drag coefficient `cd` (> 0)")
            .def_rw("cd", &aero::ConstantCd::cd, "The drag coefficient (must be > 0)");

        nb::class_<aero::MachTable>(m, "MachTable",
            "Drag coefficient as a function of Mach number in dense air (the continuum "
            "regime), e.g. wind-tunnel data for the body. Interpolated linearly between "
            "points. MachTable(mach, cd): two equal-length lists, at least two points, Mach "
            "strictly increasing, Cd > 0, else ConfigError. Evaluating outside the table "
            "raises DataCoverageError.")
            .def("__init__", [](aero::MachTable* t, std::vector<double> mach, std::vector<double> cd) {
                new (t) aero::MachTable{std::move(mach), std::move(cd)};
                t->validate();
            }, nb::arg("mach"), nb::arg("cd"))
            .def_ro("mach", &aero::MachTable::mach, "Mach numbers of the table points")
            .def_ro("cd", &aero::MachTable::cd, "Drag coefficients at those points")
            .def("__call__", &aero::MachTable::operator(), nb::arg("mach"),
                 "Drag coefficient at Mach number `mach`");

        nb::class_<aero::FlowRegimeCd>(m, "FlowRegimeCd", R"doc(
Drag coefficient that changes with the flow regime, for reentry.

The Knudsen number Kn (the air's mean free path divided by the body's size)
decides the regime. In thin air (Kn above 10, the free-molecular regime)
Cd comes from Sentman's formulas, with the energy accommodation coefficient
(0 to 1: how fully molecules hitting the surface adopt the wall's
temperature). In dense air (Kn below 1e-3, the continuum regime) Cd comes
from Henderson (1976) for a sphere, and from modified Newtonian theory or a
MachTable for a plate or cylinder. Between the two, a smooth bridge is used.
In orbit the air is free-molecular, where Cd varies only slowly (with air
temperature and composition), so a ConstantCd is close there.

Args:
    shape: AeroShape
    accommodation: energy accommodation coefficient, 0 to 1 (default 1)
    wall_temperature_k: surface temperature [K] (default 300)
    continuum: optional MachTable for the dense-air Cd (default: the
        built-in model for the shape)
)doc")
            .def("__init__", [](aero::FlowRegimeCd* f, aero::Shape shape, double accommodation,
                                double wall_temperature_k, std::optional<aero::MachTable> continuum) {
                new (f) aero::FlowRegimeCd{shape, accommodation, wall_temperature_k,
                                           continuum ? aero::ContinuumModel(*continuum)
                                                     : aero::ContinuumModel(aero::DefaultContinuum{})};
            }, nb::arg("shape"), nb::arg("accommodation") = 1.0, nb::arg("wall_temperature_k") = 300.0,
               nb::arg("continuum") = nb::none())
            .def_rw("shape", &aero::FlowRegimeCd::shape, "Body shape (AeroShape)")
            .def_rw("accommodation", &aero::FlowRegimeCd::accommodation,
                    "Energy accommodation coefficient, 0 to 1")
            .def_rw("wall_temperature_k", &aero::FlowRegimeCd::wall_temperature_k,
                    "Surface temperature [K]");

        nb::class_<aero::AeroBeta>(m, "AeroBeta", R"doc(
Ballistic coefficient beta = mass / (Cd * reference area) [kg/m^2], with Cd
from a model (ConstantCd or FlowRegimeCd). Pass it as `beta` to
make_drag_environment.

Args:
    mass_kg: mass [kg], > 0
    reference_area_m2: reference area [m^2], > 0
    cd: ConstantCd or FlowRegimeCd
    reference_length_m: the body's size [m] for the Knudsen number (a
        sphere's diameter); required (> 0) with FlowRegimeCd

Raises:
    ballistic.errors.ConfigError: a value above is out of range.
)doc")
            .def("__init__", [](aero::AeroBeta* a, double mass_kg, double reference_area_m2,
                                std::variant<aero::ConstantCd, aero::FlowRegimeCd> cd,
                                double reference_length_m) {
                aero::CdModel model = std::visit([](const auto& c) { return aero::CdModel(c); }, cd);
                new (a) aero::AeroBeta{mass_kg, reference_area_m2, reference_length_m, std::move(model)};
                a->validate();
            }, nb::arg("mass_kg"), nb::arg("reference_area_m2"), nb::arg("cd"),
               nb::arg("reference_length_m") = 0.0)
            .def_ro("mass_kg", &aero::AeroBeta::mass_kg, "Mass [kg]")
            .def_ro("reference_area_m2", &aero::AeroBeta::reference_area_m2,
                    "Reference area [m^2]")
            .def_ro("reference_length_m", &aero::AeroBeta::reference_length_m,
                    "Reference length [m]")
            .def("depends_on_flow", &aero::AeroBeta::depends_on_flow,
                 "True if Cd changes with the flow (a FlowRegimeCd)");

        m.def("free_molecular_cd", &aero::free_molecular_cd, nb::arg("shape"), nb::arg("speed_ratio"),
              nb::arg("reemission_ratio"),
              "Drag coefficient in thin air (free-molecular flow, Sentman's formulas) of an "
              "AeroShape. speed_ratio: body speed divided by the most probable molecular "
              "speed, V / sqrt(2kT/m) (dimensionless). reemission_ratio: speed of molecules "
              "leaving the surface over their arrival speed (see reemission_ratio()).");
        m.def("reemission_ratio", &aero::reemission_ratio, nb::arg("speed_m_s"), nb::arg("molar_mass_kg_kmol"),
              nb::arg("accommodation"), nb::arg("wall_temperature_k"),
              "Speed of molecules re-emitted from a surface over their arrival speed (Sentman "
              "1961; Moe & Moe 2005), for a body moving at speed_m_s [m/s] through air of mean "
              "molar mass molar_mass_kg_kmol [kg/kmol], energy accommodation coefficient "
              "accommodation (0 to 1) and wall temperature wall_temperature_k [K].");
        m.def("henderson1976_sphere_cd", &aero::henderson1976_sphere, nb::arg("mach"), nb::arg("reynolds"),
              nb::arg("wall_to_gas_temperature"),
              "Sphere drag coefficient from Henderson's (1976) full correlation, valid from "
              "dense to thin air, as given in Singh et al. 2020, eq. B1. mach: Mach number; "
              "reynolds: Reynolds number on the diameter; wall_to_gas_temperature: wall / air "
              "temperature ratio. A reference for checking FlowRegimeCd, not used by it.");
    }

    nb::class_<ballistic::atmosphere::DragSample>(m, "DragSample",
        R"doc(
Air density, wind and ballistic coefficient at one position and time.
Returned by DragEnvironment.evaluate().
)doc")
        .def_ro("rho_kg_m3", &ballistic::atmosphere::DragSample::rho_kg_m3,
            "Atmospheric density (kg/m^3).")
        .def_ro("wind_ecef_m_s", &ballistic::atmosphere::DragSample::wind_ecef_m_s,
            "Wind velocity, Earth-fixed (ECEF) axes, numpy (3,) [m/s].")
        .def_ro("beta_kg_m2", &ballistic::atmosphere::DragSample::beta_kg_m2,
            "Ballistic coefficient m/(Cd*A) (kg/m^2).")
        .def_ro("time_clamped", &ballistic::atmosphere::DragSample::time_clamped,
            "True if the time fell outside a tabulated environment's time span, so its "
            "time-varying values were held at their first or last entry.")
        .def_ro("temperature_k", &ballistic::atmosphere::DragSample::temperature_k,
            "Air temperature (K) from the live model; NaN for a tabulated density.")
        .def_ro("molar_mass_kg_kmol", &ballistic::atmosphere::DragSample::molar_mass_kg_kmol,
            "Mean molar mass of the air (kg/kmol) from the live model; NaN for a table.")
        .def_ro("cd", &ballistic::atmosphere::DragSample::cd,
            "Drag coefficient used, when beta is an AeroBeta; NaN for a beta profile.")
        .def_ro("knudsen", &ballistic::atmosphere::DragSample::knudsen,
            "Knudsen number (mean free path / body size) for a FlowRegimeCd; NaN otherwise.")
        .def_ro("mach", &ballistic::atmosphere::DragSample::mach,
            "Mach number for a FlowRegimeCd; NaN otherwise.");

    nb::class_<ballistic::atmosphere::DragEnvironment>(m, "DragEnvironment",
        R"doc(
Everything drag needs: air density (rho), wind and the ballistic coefficient
(beta), as three columns. Set it as TrajectoryConfig.drag_env to turn drag
on (leaving it unset means no drag).

Build one with:
    ballistic.make_drag_environment(...)     # recommended: the "live" models
    ballistic.from_models_snapshot(t, lat, lon, sw, beta_prior, alt_grid_km)
    ballistic.from_models_along(epochs, positions_ecef, sw, beta_prior, alt_grid_km)
    ballistic.from_models_4d(alt, lat, lon, epochs, sw, beta_prior)
    ballistic.from_accelerometer_density(epochs, measured_density, beta_prior)

A live environment (make_drag_environment) evaluates NRLMSIS 2.1 and HWM14
at the satellite's exact position on every force evaluation. The other
builders make a table once; the propagator then only reads from the table.
)doc")
        .def(nb::init<>(), "Empty environment (no drag).")
        .def("evaluate",
            &ballistic::atmosphere::DragEnvironment::evaluate,
            nb::arg("position_ecef"), nb::arg("epoch"), nb::arg("velocity_ecef") = nb::none(),
            "Density, wind, beta and related values (a DragSample) at the Earth-fixed position "
            "position_ecef [m] (numpy (3,)) and time epoch. A FlowRegimeCd beta also needs "
            "velocity_ecef (velocity relative to the rotating Earth [m/s]), because its Cd "
            "depends on the speed through the air.")
        .def("beta_depends_on_velocity",
            &ballistic::atmosphere::DragEnvironment::beta_depends_on_velocity,
            "True when beta is an AeroBeta with a FlowRegimeCd.")
        .def("has_drag",
            &ballistic::atmosphere::DragEnvironment::has_drag,
            "True if both the density and beta columns are filled in (drag will act).")
        .def("domain", [](const ballistic::atmosphere::DragEnvironment& self)
                -> std::optional<std::pair<ballistic::Epoch, ballistic::Epoch>> {
            auto w = self.domain();
            if (!w) return std::nullopt;
            return std::make_pair(w->start, w->end);
        }, "(start, end) Epochs covered by a tabulated environment's time-varying columns, "
           "or None (altitude-only profiles, or a live environment: see weather_window). "
           "Propagating past it holds the atmosphere frozen at its last values.")
        .def("is_live", [](const ballistic::atmosphere::DragEnvironment& self) {
            return self.live_source() != nullptr;
        }, "True for an environment from make_drag_environment: NRLMSIS 2.1 / HWM14 "
           "evaluated at the satellite's position, valid wherever space weather is "
           "available (weather_window).")
        .def("weather_window", [](const ballistic::atmosphere::DragEnvironment& self)
                -> std::optional<std::pair<ballistic::Epoch, ballistic::Epoch>> {
            const auto* src = self.live_source();
            if (!src || !src->valid) return std::nullopt;
            return std::make_pair(src->valid->start, src->valid->end);
        }, "(start, end) Epochs over which a live environment has space weather, or None "
           "(a tabulated environment, or constant weather valid at any time). A Trajectory "
           "stops with DataCoverageError at its edge.")
        .def("measured_until", [](const ballistic::atmosphere::DragEnvironment& self)
                -> std::optional<ballistic::Epoch> {
            const auto* src = self.live_source();
            if (!src || !src->measured_until) return std::nullopt;
            return *src->measured_until;
        }, "The Epoch where a live environment's weather turns from measured to "
           "CelesTrak's forecast, or None (no forecast days).")
        .def("uses_predicted_weather",
            &ballistic::atmosphere::DragEnvironment::uses_predicted_weather,
            "True if the environment reaches into forecast space weather (days past the "
            "last measured day).")
        .def_static("_live",
            [](ballistic::atmosphere::BetaProfile1D beta, bool allow_predicted,
               double isa_floor_km, bool with_wind) {
                namespace atm = ballistic::atmosphere;
                atm::require_nrl_models("load_drag_env (a saved live NRLMSIS 2.1 environment)");
                const auto full = atm::SpaceWeatherData::instance().full_history(allow_predicted);
                auto src = std::make_shared<atm::LiveModelSource>();
                src->weather = full->history;
                src->valid = full->valid;
                src->measured_until = full->measured_until;
                src->loaded_weather = true;
                src->allow_predicted = allow_predicted;
                src->isa_floor_km = isa_floor_km;
                atm::DragEnvironment::WindVariant wind = atm::WindProfile1D_ENU{};
                if (with_wind) wind = atm::LiveModelWind(src);
                return std::make_shared<const atm::DragEnvironment>(
                    atm::LiveModelDensity(src), std::move(wind), std::move(beta));
            }, nb::arg("beta"), nb::arg("allow_predicted"), nb::arg("isa_floor_km"),
            nb::arg("with_wind"),
            "Internal (used by load_drag_env): rebuild a live environment on the currently "
            "loaded space weather.")
        .def("without_wind",
            &ballistic::atmosphere::DragEnvironment::without_wind,
            R"doc(
Return a copy of this environment without wind (the atmosphere co-rotates
with the Earth and has no wind of its own). Density and beta are unchanged.
Useful for comparing results with and without wind.
)doc")
        // -----------------------------------------------------------------
        // Low-level accessors for save / load (NPZ persistence).
        // Each accessor inspects the variant alternative and returns:
        //   - density: dict with kind ("1d" / "2d" / "4d") + per-kind arrays
        //   - wind   : dict with kind ("1d_enu" / "2d_ecef" / "4d_ecef") + arrays
        //   - beta   : dict with kind ("1d" / "2d" / "aero") + arrays
        // Kinds the Python persistence helper does not support (BetaProfile2D,
        // AeroBeta) return only the kind tag so save_drag_env can raise a
        // clear NotImplementedError.
        // -----------------------------------------------------------------
        .def("_density_export", [](const ballistic::atmosphere::DragEnvironment& self) {
            namespace atm = ballistic::atmosphere;
            return std::visit([](const auto& p) -> nb::dict {
                using T = std::decay_t<decltype(p)>;
                nb::dict d;
                if constexpr (std::is_same_v<T, atm::DensityProfile1D>) {
                    d["kind"] = "1d";
                    d["altitude_km"] = p.altitude_km();
                    d["density_kg_m3"] = p.density_kg_m3();
                } else if constexpr (std::is_same_v<T, atm::DensityProfile2D>) {
                    d["kind"] = "2d";
                    d["altitude_km"] = p.altitude_km();
                    std::vector<double> ep_mjd;
                    ep_mjd.reserve(p.epochs().size());
                    for (const auto& e : p.epochs())
                        ep_mjd.push_back(e.mjd(ballistic::TimeScale::UTC));
                    d["epochs_mjd_utc"] = std::move(ep_mjd);
                    d["density_grid"] = p.density_grid();
                } else if constexpr (std::is_same_v<T, atm::DensityField4D>) {
                    // Decode half-precision log(rho) → linear rho on the C++
                    // side. Storage order matches the field layout:
                    // ((t * n_lon + j) * n_lat + i) * n_alt + k.
                    d["kind"] = "4d";
                    d["altitude_km"] = p.altitude_km();
                    d["latitude_deg"] = p.latitude_deg();
                    d["longitude_deg"] = p.longitude_deg();
                    std::vector<double> ep_mjd;
                    ep_mjd.reserve(p.epochs().size());
                    for (const auto& e : p.epochs())
                        ep_mjd.push_back(e.mjd(ballistic::TimeScale::UTC));
                    d["epochs_mjd_utc"] = std::move(ep_mjd);
                    const auto& enc = p.encoded_log_rho();
                    std::vector<double> rho_grid;
                    rho_grid.reserve(enc.size());
                    for (auto h : enc) {
                        rho_grid.push_back(
                            std::exp(static_cast<double>(atm::half_to_float(h))));
                    }
                    d["density_grid"] = std::move(rho_grid);
                } else if constexpr (std::is_same_v<T, atm::LiveModelDensity>) {
                    d["kind"] = "live";
                    const auto& src = *p.source();
                    d["loaded_weather"] = src.loaded_weather;
                    d["allow_predicted"] = src.allow_predicted;
                    d["isa_floor_km"] = src.isa_floor_km;
                }
                return d;
            }, self.density());
        })
        .def("_wind_export", [](const ballistic::atmosphere::DragEnvironment& self) {
            namespace atm = ballistic::atmosphere;
            return std::visit([](const auto& p) -> nb::dict {
                using T = std::decay_t<decltype(p)>;
                nb::dict d;
                if constexpr (std::is_same_v<T, atm::WindProfile1D_ENU>) {
                    d["kind"] = p.empty() ? "1d_enu_empty" : "1d_enu";
                    if (!p.empty()) {
                        d["altitude_km"] = p.altitude_km();
                        d["meridional_ms"] = p.meridional_ms();
                        d["zonal_ms"] = p.zonal_ms();
                    }
                } else if constexpr (std::is_same_v<T, atm::WindProfile2D_ECEF>) {
                    d["kind"] = "2d_ecef";
                    d["altitude_km"] = p.altitude_km();
                    std::vector<double> ep_mjd;
                    ep_mjd.reserve(p.epochs().size());
                    for (const auto& e : p.epochs())
                        ep_mjd.push_back(e.mjd(ballistic::TimeScale::UTC));
                    d["epochs_mjd_utc"] = std::move(ep_mjd);
                    std::vector<double> wx, wy, wz;
                    wx.reserve(p.wind_grid().size());
                    wy.reserve(p.wind_grid().size());
                    wz.reserve(p.wind_grid().size());
                    for (const auto& w : p.wind_grid()) {
                        wx.push_back(w.x()); wy.push_back(w.y()); wz.push_back(w.z());
                    }
                    d["wind_x_ecef"] = std::move(wx);
                    d["wind_y_ecef"] = std::move(wy);
                    d["wind_z_ecef"] = std::move(wz);
                } else if constexpr (std::is_same_v<T, atm::WindField4D_ECEF>) {
                    // Decode 3 halves/cell to parallel x/y/z float64 arrays.
                    d["kind"] = "4d_ecef";
                    d["altitude_km"] = p.altitude_km();
                    d["latitude_deg"] = p.latitude_deg();
                    d["longitude_deg"] = p.longitude_deg();
                    std::vector<double> ep_mjd;
                    ep_mjd.reserve(p.epochs().size());
                    for (const auto& e : p.epochs())
                        ep_mjd.push_back(e.mjd(ballistic::TimeScale::UTC));
                    d["epochs_mjd_utc"] = std::move(ep_mjd);
                    const auto& packed = p.encoded_packed();
                    const std::size_t n_cells = packed.size() / 3;
                    std::vector<double> wx, wy, wz;
                    wx.reserve(n_cells);
                    wy.reserve(n_cells);
                    wz.reserve(n_cells);
                    for (std::size_t i = 0; i < n_cells; ++i) {
                        wx.push_back(static_cast<double>(atm::half_to_float(
                            packed[3 * i + 0])));
                        wy.push_back(static_cast<double>(atm::half_to_float(
                            packed[3 * i + 1])));
                        wz.push_back(static_cast<double>(atm::half_to_float(
                            packed[3 * i + 2])));
                    }
                    d["wind_x_ecef"] = std::move(wx);
                    d["wind_y_ecef"] = std::move(wy);
                    d["wind_z_ecef"] = std::move(wz);
                } else if constexpr (std::is_same_v<T, atm::LiveModelWind>) {
                    d["kind"] = "live";
                }
                return d;
            }, self.wind());
        })
        .def("_beta_export", [](const ballistic::atmosphere::DragEnvironment& self) {
            namespace atm = ballistic::atmosphere;
            return std::visit([](const auto& p) -> nb::dict {
                using T = std::decay_t<decltype(p)>;
                nb::dict d;
                if constexpr (std::is_same_v<T, atm::BetaProfile1D>) {
                    d["kind"] = "1d";
                    d["altitude_km"] = p.altitude_km();
                    d["beta_kg_m2"] = p.beta_kg_m2();
                } else if constexpr (std::is_same_v<T, atm::BetaProfile2D>) {
                    d["kind"] = "2d";
                } else {
                    d["kind"] = "aero";
                }
                return d;
            }, self.beta());
        })
        .def_static("_from_1d_components",
            [](std::vector<double> density_alt_km,
               std::vector<double> density_kg_m3,
               std::vector<double> wind_alt_km,
               std::vector<double> wind_meridional_ms,
               std::vector<double> wind_zonal_ms,
               std::vector<double> beta_alt_km,
               std::vector<double> beta_kg_m2) {
                namespace atm = ballistic::atmosphere;
                atm::DensityProfile1D density(std::move(density_alt_km),
                                                std::move(density_kg_m3));
                atm::WindProfile1D_ENU wind;
                if (!wind_alt_km.empty()) {
                    wind = atm::WindProfile1D_ENU(std::move(wind_alt_km),
                                                    std::move(wind_meridional_ms),
                                                    std::move(wind_zonal_ms));
                }
                atm::BetaProfile1D beta(std::move(beta_alt_km),
                                         std::move(beta_kg_m2));
                return atm::DragEnvironment(std::move(density),
                                              std::move(wind),
                                              std::move(beta));
            },
            nb::arg("density_altitude_km"),
            nb::arg("density_kg_m3"),
            nb::arg("wind_altitude_km"),
            nb::arg("wind_meridional_ms"),
            nb::arg("wind_zonal_ms"),
            nb::arg("beta_altitude_km"),
            nb::arg("beta_kg_m2"),
            R"doc(Internal: used by dragenv_io.load_drag_env to rebuild an
altitude-only DragEnvironment from saved arrays (altitudes in km, density
in kg/m^3, winds in m/s, beta in kg/m^2). Empty wind_* arrays mean no
wind.)doc")
        .def_static("_from_2d_components",
            [](std::vector<double> density_alt_km,
               std::vector<double> density_epochs_mjd_utc,
               std::vector<double> density_grid,
               std::string wind_kind,
               std::vector<double> wind_alt_km,
               std::vector<double> wind_epochs_mjd_utc,
               std::vector<double> wind_x_ecef,
               std::vector<double> wind_y_ecef,
               std::vector<double> wind_z_ecef,
               std::vector<double> wind_meridional_ms,
               std::vector<double> wind_zonal_ms,
               std::vector<double> beta_alt_km,
               std::vector<double> beta_kg_m2) {
                namespace atm = ballistic::atmosphere;

                // Density: always 2D for this factory.
                std::vector<ballistic::Epoch> density_epochs;
                density_epochs.reserve(density_epochs_mjd_utc.size());
                for (double m : density_epochs_mjd_utc)
                    density_epochs.push_back(
                        ballistic::Epoch::from_mjd(m, ballistic::TimeScale::UTC));
                atm::DensityProfile2D density(
                    std::move(density_alt_km), std::move(density_epochs),
                    std::move(density_grid));

                // Wind: select variant by kind.
                atm::DragEnvironment::WindVariant wind;
                if (wind_kind == "2d_ecef") {
                    std::vector<ballistic::Epoch> wind_epochs;
                    wind_epochs.reserve(wind_epochs_mjd_utc.size());
                    for (double m : wind_epochs_mjd_utc)
                        wind_epochs.push_back(
                            ballistic::Epoch::from_mjd(m, ballistic::TimeScale::UTC));
                    std::vector<ballistic::Vector3d> wind_grid;
                    wind_grid.reserve(wind_x_ecef.size());
                    for (std::size_t i = 0; i < wind_x_ecef.size(); ++i)
                        wind_grid.emplace_back(wind_x_ecef[i], wind_y_ecef[i],
                                                 wind_z_ecef[i]);
                    wind = atm::WindProfile2D_ECEF(
                        std::move(wind_alt_km), std::move(wind_epochs),
                        std::move(wind_grid));
                } else if (wind_kind == "1d_enu") {
                    wind = atm::WindProfile1D_ENU(
                        std::move(wind_alt_km),
                        std::move(wind_meridional_ms),
                        std::move(wind_zonal_ms));
                } else {
                    // "1d_enu_empty" (dragenv_io._check_pairing refuses a
                    // 4D wind alongside a 2D density before calling this).
                    wind = atm::WindProfile1D_ENU{};
                }

                atm::BetaProfile1D beta(std::move(beta_alt_km),
                                         std::move(beta_kg_m2));
                return atm::DragEnvironment(std::move(density),
                                              std::move(wind),
                                              std::move(beta));
            },
            R"doc(Internal: used by dragenv_io.load_drag_env to rebuild a
DragEnvironment whose density varies with altitude and time. `wind_kind`
selects the wind layout: "2d_ecef" / "1d_enu" / "1d_enu_empty" (no wind).
Pass empty arrays for fields that don't apply to the chosen wind kind.
Epochs are Modified Julian Dates in UTC.)doc")
        .def_static("_from_4d_components",
            [](std::vector<double> density_alt_km,
               std::vector<double> density_lat_deg,
               std::vector<double> density_lon_deg,
               std::vector<double> density_epochs_mjd_utc,
               std::vector<double> density_grid,    // linear ρ values
               std::string wind_kind,
               std::vector<double> wind_alt_km,
               std::vector<double> wind_lat_deg,
               std::vector<double> wind_lon_deg,
               std::vector<double> wind_epochs_mjd_utc,
               std::vector<double> wind_x_ecef,
               std::vector<double> wind_y_ecef,
               std::vector<double> wind_z_ecef,
               std::vector<double> beta_alt_km,
               std::vector<double> beta_kg_m2) {
                namespace atm = ballistic::atmosphere;

                // Density: encode linear ρ → log(ρ) → half.
                std::vector<ballistic::Epoch> density_epochs;
                density_epochs.reserve(density_epochs_mjd_utc.size());
                for (double m : density_epochs_mjd_utc)
                    density_epochs.push_back(
                        ballistic::Epoch::from_mjd(m, ballistic::TimeScale::UTC));
                std::vector<atm::half> log_rho_storage;
                log_rho_storage.reserve(density_grid.size());
                for (double r : density_grid) {
                    const double lr = (r > 0.0) ? std::log(r)
                                                : std::log(1e-300);
                    log_rho_storage.push_back(
                        atm::float_to_half(static_cast<float>(lr)));
                }
                atm::DensityField4D density(
                    std::move(density_alt_km),
                    std::move(density_lat_deg),
                    std::move(density_lon_deg),
                    std::move(density_epochs),
                    std::move(log_rho_storage));

                // Wind variant.
                atm::DragEnvironment::WindVariant wind;
                if (wind_kind == "4d_ecef") {
                    std::vector<ballistic::Epoch> wind_epochs;
                    wind_epochs.reserve(wind_epochs_mjd_utc.size());
                    for (double m : wind_epochs_mjd_utc)
                        wind_epochs.push_back(
                            ballistic::Epoch::from_mjd(m, ballistic::TimeScale::UTC));
                    std::vector<ballistic::Vector3d> wind_grid;
                    wind_grid.reserve(wind_x_ecef.size());
                    for (std::size_t i = 0; i < wind_x_ecef.size(); ++i)
                        wind_grid.emplace_back(wind_x_ecef[i], wind_y_ecef[i],
                                                wind_z_ecef[i]);
                    wind = atm::WindField4D_ECEF(
                        std::move(wind_alt_km),
                        std::move(wind_lat_deg),
                        std::move(wind_lon_deg),
                        std::move(wind_epochs),
                        wind_grid);
                } else {
                    // "1d_enu_empty": no wind. dragenv_io refuses every
                    // other pairing (_check_pairing) before calling this, so
                    // a populated wind never reaches this branch.
                    wind = atm::WindProfile1D_ENU{};
                }

                atm::BetaProfile1D beta(std::move(beta_alt_km),
                                         std::move(beta_kg_m2));
                return atm::DragEnvironment(std::move(density),
                                              std::move(wind),
                                              std::move(beta));
            },
            R"doc(Internal: used by dragenv_io.load_drag_env to rebuild a
DragEnvironment whose density varies with altitude, latitude, longitude and
time. `wind_kind` selects the wind layout: "4d_ecef" or "1d_enu_empty" (no
wind). Flat index of the density grid: ((t * n_lon + j) * n_lat + i) * n_alt
+ k. density_grid values are densities in kg/m³; they are stored internally
as half-precision log(ρ). Wind values are stored as 3 half-precision numbers
per cell. Epochs are Modified Julian Dates in UTC.)doc");

    nb::class_<ballistic::atmosphere::IterationDiag>(m, "IterationDiag",
        R"doc(
Report from building a drag environment along a trajectory: how many passes
were run, whether the trajectory stopped changing between passes, and by
how much it changed in the last pass. from_models_along makes a single pass
(iterations = 1); from_models_along_iterated repeats until the trajectory
settles.
)doc")
        .def(nb::init<>())
        .def_ro("iterations", &ballistic::atmosphere::IterationDiag::iterations,
                "Number of passes run")
        .def_ro("converged", &ballistic::atmosphere::IterationDiag::converged,
                "True if the trajectory change fell below the tolerance")
        .def_ro("final_traj_delta_m",
            &ballistic::atmosphere::IterationDiag::final_traj_delta_m,
            "Largest change of the trajectory in the last pass [m]");

    // Generators as free module-level functions.
    m.def("standard_leo_altitude_grid_km",
        &ballistic::atmosphere::standard_leo_altitude_grid_km,
        "Default altitude grid for low Earth orbit tables: 100-1500 km in 25 km steps "
        "(list of float, km).");

    m.def("from_models_snapshot",
        &ballistic::atmosphere::from_models_snapshot,
        nb::arg("t"), nb::arg("ref_lat_deg"), nb::arg("ref_lon_deg"),
        nb::arg("sw"), nb::arg("beta_prior"),
        nb::arg("altitude_grid_km") = std::vector<double>{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Build a DragEnvironment whose density and wind vary only with altitude, evaluated "
        "once at the time t (Epoch) and the point ref_lat_deg / ref_lon_deg [degrees]. "
        "sw: SpaceWeatherHistory; beta_prior: BetaProfile1D; altitude_grid_km: altitudes "
        "[km] (empty = standard_leo_altitude_grid_km()). Suits arcs much shorter than one "
        "orbit.");

    m.def("from_models_along",
        [](const std::vector<ballistic::Epoch>& sample_epochs,
           const std::vector<ballistic::Vector3d>& sample_positions_ecef,
           const ballistic::atmosphere::SpaceWeatherHistory& sw,
           ballistic::atmosphere::BetaProfile1D beta_prior,
           std::vector<double> altitude_grid_km)
            -> std::pair<ballistic::atmosphere::DragEnvironment,
                          ballistic::atmosphere::IterationDiag>
        {
            ballistic::atmosphere::IterationDiag diag;
            auto env = ballistic::atmosphere::from_models_along(
                sample_epochs, sample_positions_ecef, sw,
                std::move(beta_prior), std::move(altitude_grid_km), &diag);
            return {std::move(env), diag};
        },
        nb::arg("sample_epochs"), nb::arg("sample_positions_ecef"),
        nb::arg("sw"), nb::arg("beta_prior"),
        nb::arg("altitude_grid_km") = std::vector<double>{},
        nb::call_guard<nb::gil_scoped_release>(),
        "Build a DragEnvironment tabulated over altitude and time by evaluating NRLMSIS 2.1 "
        "/ HWM14 below each point of a given trajectory. sample_epochs: list of Epoch; "
        "sample_positions_ecef: matching Earth-fixed positions [m]; sw: "
        "SpaceWeatherHistory; beta_prior: BetaProfile1D; altitude_grid_km: altitudes [km] "
        "(empty = standard_leo_altitude_grid_km()). Returns (env, IterationDiag).");

    m.def("make_drag_environment",
        [](const ballistic::Epoch& t0,
           const ballistic::Vector3d& pos_ecef,
           const ballistic::Vector3d& vel_ecef,
           double duration_s,
           const ballistic::TrajectoryConfig& base_config,
           std::variant<ballistic::atmosphere::BetaProfile1D, ballistic::aero::AeroBeta> beta,
           std::optional<ballistic::atmosphere::SpaceWeatherHistory> weather,
           std::vector<double> altitude_grid_km,
           int max_iter, double tol_relative, double sample_dt_s,
           bool allow_predicted_weather)
            -> std::pair<std::shared_ptr<const ballistic::atmosphere::DragEnvironment>,
                          ballistic::atmosphere::IterationDiag>
        {
            ballistic::atmosphere::DragSetup setup;
            setup.beta = std::move(beta);
            setup.weather = std::move(weather);
            setup.altitude_grid_km = std::move(altitude_grid_km);
            setup.max_iter = max_iter;
            setup.tol_relative = tol_relative;
            setup.sample_dt_s = sample_dt_s;
            setup.allow_predicted_weather = allow_predicted_weather;
            ballistic::atmosphere::IterationDiag diag;
            auto env = ballistic::atmosphere::make_drag_environment(
                t0, pos_ecef, vel_ecef, duration_s, base_config, setup, &diag);
            return {std::move(env), diag};
        },
        nb::arg("t0"), nb::arg("pos_ecef"), nb::arg("vel_ecef"),
        nb::arg("duration_s"), nb::arg("base_config"), nb::arg("beta"),
        nb::arg("weather") = nb::none(),
        nb::arg("altitude_grid_km") = std::vector<double>{},
        nb::arg("max_iter") = 3, nb::arg("tol_relative") = 0.01,
        nb::arg("sample_dt_s") = 60.0,
        nb::arg("allow_predicted_weather") = true,
        R"doc(
Turn on drag with one call (low-level form taking the C++ TrajectoryConfig;
prefer ballistic.make_drag_environment, which also accepts the Python
TrajectoryConfig dataclass).

Returns (drag_env, IterationDiag). The environment is "live": NRLMSIS 2.1
(density) and HWM14 (wind) are evaluated at the satellite's exact position
on every force evaluation, for any time covered by the loaded space-weather
file.

Args:
    t0: start Epoch; checked to have space weather
    beta: BetaProfile1D or AeroBeta (the satellite's ballistic coefficient)
    weather: None (default) uses the measured F10.7 / Ap loaded by
        ballistic.load(); pass SpaceWeatherHistory.from_constant(...) to
        override.
    allow_predicted_weather: past the last measured day, use CelesTrak's
        forecast (default True; a Trajectory warns there). False refuses.
    pos_ecef, vel_ecef, duration_s, base_config, altitude_grid_km,
    max_iter, tol_relative, sample_dt_s: accepted for compatibility and
        not used (a live environment needs no table).

Raises:
    RuntimeError: weather=None and no space weather is loaded.
    ballistic.errors.DataCoverageError: t0 has no space weather. A
        Trajectory using the environment also stops with this error at the
        end of the weather (keeping its state and history up to there).
)doc");

    m.def("from_models_4d",
        &ballistic::atmosphere::from_models_4d,
        nb::arg("altitude_grid_km"), nb::arg("latitude_grid_deg"),
        nb::arg("longitude_grid_deg"), nb::arg("epoch_grid"),
        nb::arg("sw"), nb::arg("beta_prior"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Build a DragEnvironment tabulated over altitude [km], latitude and longitude "
        "[degrees] and time (lists of Epoch), e.g. one shared atmosphere for many "
        "satellites. sw: SpaceWeatherHistory; beta_prior: BetaProfile1D. Expensive: "
        "evaluated in parallel over the time axis (OpenMP).");

    m.def("extend_density_below",
        &ballistic::atmosphere::extend_density_below,
        nb::arg("env"), nb::arg("floor_km") = 80.0,
        R"doc(
Use the U.S. Standard Atmosphere 1976 for the density below `floor_km`
[km] (0 < floor_km <= 86; the standard is computed exactly from its
defining constants in that range), for deep reentry or sub-orbital flight.
Returns a new DragEnvironment; beta and wind are unchanged (a live
environment keeps HWM14 wind below the floor).

A live environment (make_drag_environment) already has NRLMSIS 2.1 down to
the ground, so for it this only chooses the standard atmosphere below the
floor instead. On an altitude or altitude-time table it adds table points
every 2 km from 0 km up to the floor (the standard 100-1500 km grid has no
points there to replace).

Raises:
    ballistic.errors.ConfigError: a latitude/longitude/time (4D) table, or
        floor_km out of range.
)doc");

    m.def("from_accelerometer_density",
        &ballistic::atmosphere::from_accelerometer_density,
        nb::arg("sample_epochs"),
        nb::arg("measured_density_kg_m3"),
        nb::arg("beta_prior"),
        nb::arg("altitude_grid_km") = std::vector<double>{},
        R"doc(
Build a DragEnvironment from measured air densities (derived from a
satellite's on-board accelerometer). At each time the same measured density
is used at every altitude, which is appropriate when the satellite stays
near one altitude over the arc (typical use: the GOCE satellite's final
descent, with TU Delft accelerometer densities).

Wind is empty (no HWM14). For altitude/lat/lon variation use
from_models_along or from_models_4d instead.

Args:
    sample_epochs: list of Epoch (length N).
    measured_density_kg_m3: list of float, ρ at each sample (length N,
        all values must be > 0).
    beta_prior: BetaProfile1D (satellite β = m/(Cd·A)).
    altitude_grid_km: optional altitude grid (defaults to
        standard_leo_altitude_grid_km, 100-1500 km / 25 km steps).
        Pass a tighter grid (e.g. 100-300 km for GOCE reentry) for
        storage efficiency.

Returns:
    DragEnvironment with DensityProfile2D + empty WindProfile1D_ENU +
    the provided BetaProfile1D.
)doc");

}
