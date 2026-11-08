// =============================================================================
// Initialization + data-loading Python bindings
// =============================================================================
//
// Called from NB_MODULE in bindings.cpp; registration order is set by the
// call order there.
//
// Earth orientation parameters (EOP) are the measured irregularities of the
// Earth's rotation (polar motion and UT1-UTC, the offset of Earth-rotation
// time from atomic clock time). They are needed to convert between
// Earth-fixed and inertial coordinates; without them a low-orbit prediction
// is off by 3-10 m.
// =============================================================================

#include <ballistic/core/errors.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <ballistic/ballistic.hpp>  // Includes version constants and all headers
#include <ballistic/gravity/ocean_tides.hpp>
#include <ballistic/atmosphere/space_weather.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <ballistic/core/build_info.hpp>
#include <ballistic/atmosphere/nrl_models.hpp>

namespace nb = nanobind;

void bind_init(nb::module_& m) {
    // =========================================================================
    // InitConfig - Configuration for library initialization
    // =========================================================================
    nb::class_<ballistic::InitConfig>(m, "InitConfig",
        R"doc(
Paths to external data files for initialize_with_config.

The Earth gravity model and the planetary ephemeris (JPL DE440) are built
in. The Earth orientation parameters (EOP) are not: they are needed for
sub-meter accuracy and are found automatically in the data directory below
once downloaded (ballistic.load() or ballistic.Loader().load_eop()). Set a
path here only to use a specific file. Unset paths are None.

Default data directory (platform-specific):
    Windows: %LOCALAPPDATA%\ballistic  (C:\Users\<user>\AppData\Local\ballistic)
    Linux:   ~/.local/share/ballistic
    macOS:   ~/Library/Application Support/ballistic

Example:
    config = ballistic.InitConfig()
    config.eop_path = "finals2000A.data"
    config.jpl_ephemeris_path = "de421.bsp"
    result = ballistic.initialize_with_config(config)
)doc")
        .def(nb::init<>())
        .def_prop_rw("egm2008_path",
            [](const ballistic::InitConfig& c) -> std::optional<std::string> {
                if (c.egm2008_path.has_value()) return c.egm2008_path->string();
                return std::nullopt;
            },
            [](ballistic::InitConfig& c, const std::optional<std::string>& path) {
                if (path.has_value()) c.egm2008_path = std::filesystem::path(*path);
                else c.egm2008_path = std::nullopt;
            },
            "Path to a full EGM2008 gravity coefficient file (str), needed only for "
            "gravity degrees 361-2190; the embedded model covers degrees 2-360")
        .def_prop_rw("eop_path",
            [](const ballistic::InitConfig& c) -> std::optional<std::string> {
                if (c.eop_path.has_value()) return c.eop_path->string();
                return std::nullopt;
            },
            [](ballistic::InitConfig& c, const std::optional<std::string>& path) {
                if (path.has_value()) c.eop_path = std::filesystem::path(*path);
                else c.eop_path = std::nullopt;
            },
            "Path to the IERS finals2000A.data (or .all) Earth orientation file (str)")
        .def_prop_rw("jpl_ephemeris_path",
            [](const ballistic::InitConfig& c) -> std::optional<std::string> {
                if (c.jpl_ephemeris_path.has_value()) return c.jpl_ephemeris_path->string();
                return std::nullopt;
            },
            [](ballistic::InitConfig& c, const std::optional<std::string>& path) {
                if (path.has_value()) c.jpl_ephemeris_path = std::filesystem::path(*path);
                else c.jpl_ephemeris_path = std::nullopt;
            },
            "Path to a JPL planetary ephemeris file, .bsp (SPICE) or .bin (str); replaces "
            "the embedded DE440")
        .def_prop_rw("leap_seconds_path",
            [](const ballistic::InitConfig& c) -> std::optional<std::string> {
                if (c.leap_seconds_path.has_value()) return c.leap_seconds_path->string();
                return std::nullopt;
            },
            [](ballistic::InitConfig& c, const std::optional<std::string>& path) {
                if (path.has_value()) c.leap_seconds_path = std::filesystem::path(*path);
                else c.leap_seconds_path = std::nullopt;
            },
            "Path to the IERS leap-seconds.list file (str)")
        .def_prop_rw("nrl_models_path",
            [](const ballistic::InitConfig& c) -> std::optional<std::string> {
                if (c.nrl_models_path.has_value()) return c.nrl_models_path->string();
                return std::nullopt;
            },
            [](ballistic::InitConfig& c, const std::optional<std::string>& path) {
                if (path.has_value()) c.nrl_models_path = std::filesystem::path(*path);
                else c.nrl_models_path = std::nullopt;
            },
            "Path to the NRLMSIS 2.1 / HWM14 plugin (str), needed only for model-based drag; "
            "None searches BALLISTIC_NRL_MODELS and the data directory (where "
            "ballistic.install_nrl_models installs it)");

    // =========================================================================
    // InitResult - Result of library initialization
    // =========================================================================
    nb::class_<ballistic::InitResult>(m, "InitResult",
        "What initialize() loaded and any problems found. True in a boolean context when "
        "initialization succeeded.")
        .def(nb::init<>())
        .def_ro("success", &ballistic::InitResult::success,
            "True if initialization succeeded")
        .def_ro("egm2008_loaded", &ballistic::InitResult::egm2008_loaded,
            "True if an external EGM2008 gravity file was loaded")
        .def_ro("eop_loaded", &ballistic::InitResult::eop_loaded,
            "True if Earth orientation data (EOP) was loaded")
        .def_ro("eop_auto_discovered", &ballistic::InitResult::eop_auto_discovered,
            "True if EOP was loaded via data-directory auto-discovery (not explicit eop_path)")
        .def_ro("jpl_ephemeris_loaded", &ballistic::InitResult::jpl_ephemeris_loaded,
            "True if an external JPL ephemeris file was loaded")
        .def_ro("embedded_ephemeris_loaded", &ballistic::InitResult::embedded_ephemeris_loaded,
            "True if the built-in DE440 planetary ephemeris was loaded")
        .def_ro("leap_seconds_loaded", &ballistic::InitResult::leap_seconds_loaded,
            "True if leap seconds were loaded from an external file")
        .def_ro("leap_seconds_auto_discovered", &ballistic::InitResult::leap_seconds_auto_discovered,
            "True if leap-seconds.list was loaded via data-directory auto-discovery")
        .def_ro("gravity_degree_loaded", &ballistic::InitResult::gravity_degree_loaded,
            "Highest gravity-model degree available")
        .def_ro("jpl_de_version", &ballistic::InitResult::jpl_de_version,
            "Which JPL ephemeris is in use (e.g., 'DE421' or 'DE440 (embedded)')")
        .def_ro("eop_source_path", &ballistic::InitResult::eop_source_path,
            "Path of the loaded EOP file (empty if none)")
        .def_ro("nrl_models_loaded", &ballistic::InitResult::nrl_models_loaded,
            "True if the NRLMSIS 2.1 / HWM14 plugin is loaded (model-based drag available)")
        .def_ro("nrl_models_path", &ballistic::InitResult::nrl_models_path,
            "Path of the loaded NRL models plugin (empty if none)")
        .def_ro("error_message", &ballistic::InitResult::error_message,
            "Error description if success is False")
        .def_ro("eop_table_end_mjd", &ballistic::InitResult::eop_table_end_mjd,
            "Last day of the loaded EOP table as a Modified Julian Date (MJD; 0 = no EOP "
            "loaded)")
        .def_ro("eop_table_stale", &ballistic::InitResult::eop_table_stale,
            "True if EOP is loaded but the table's end (including its ~1 year of "
            "predictions) is already in the past. Past the end the last value is held for "
            "up to 30 days with a warning (meters of error at low orbit), then the frame "
            "transforms raise DataCoverageError. Refresh with ballistic.Loader().load_eop().")
        .def_ro("leap_seconds_expiration_jd", &ballistic::InitResult::leap_seconds_expiration_jd,
            "Expiry date of the leap-second table as a Julian Date; after it a newly "
            "announced leap second could be missed (0 = unknown)")
        .def_ro("leap_seconds_expired", &ballistic::InitResult::leap_seconds_expired,
            "True if the leap-second table's expiry date is in the past. Refresh with "
            "ballistic.Loader().timescale().")
        .def_ro("warnings", &ballistic::InitResult::warnings,
            "Readable warnings about out-of-date data (list of str; empty = up to date)")
        .def("__bool__", [](const ballistic::InitResult& r) { return r.success; },
            "Same as success, so `if ballistic.initialize():` works.")
        .def("__repr__", [](const ballistic::InitResult& r) {
            std::string out = "InitResult(success=";
            out += r.success ? "True" : "False";
            out += ", eop_loaded=";
            out += r.eop_loaded ? "True" : "False";
            out += ", jpl_ephemeris_loaded=";
            out += r.jpl_ephemeris_loaded ? "True" : "False";
            out += ", leap_seconds_loaded=";
            out += r.leap_seconds_loaded ? "True" : "False";
            if (!r.error_message.empty()) out += ", error=\"" + r.error_message + "\"";
            out += ")";
            return out;
        });

    // =========================================================================
    // LeapSecondInfo - Information about leap second data
    // =========================================================================
    nb::class_<ballistic::LeapSecondInfo>(m, "LeapSecondInfo",
        "Information about the leap-second table (the list of one-second steps that keep "
        "UTC within a second of Earth-rotation time).")
        .def(nb::init<>())
        .def_ro("current_tai_utc", &ballistic::LeapSecondInfo::current_tai_utc,
            "Latest offset between atomic time (TAI) and UTC [s]")
        .def_ro("entry_count", &ballistic::LeapSecondInfo::entry_count,
            "Number of leap second entries in the table")
        .def_ro("loaded_from_file", &ballistic::LeapSecondInfo::loaded_from_file,
            "True if loaded from an external file, False if using the built-in table")
        .def_ro("expiration_jd", &ballistic::LeapSecondInfo::expiration_jd,
            "Expiry date as a Julian Date (0 if unknown)");

    // =========================================================================
    // Initialization Functions
    // =========================================================================

    // Simple initialize
    m.def("initialize", []() {
        return ballistic::initialize(ballistic::InitConfig{});
    },
    R"doc(
Initialize the library with the built-in data plus any data files found in
the data directory (get_data_directory()).

Returns:
    InitResult: full status (True in a boolean context on success, so
    `if ballistic.initialize():` works). Check `.eop_loaded`: the Earth
    orientation file (EOP) is only picked up if it has been downloaded.
    Without it, converting between Earth-fixed and inertial coordinates
    raises DataCoverageError, unless set_eop_policy('assume_zero') accepts
    a 3-10 m error at low Earth orbit.

Note:
    The EGM2008 gravity coefficients (degrees 2-360) and the DE440 planetary
    ephemeris (years 1950-2100) are built in. The only file needed for
    sub-meter accuracy is finals2000A.data (EOP): run ballistic.load() or
    ballistic.Loader().load_eop() once to put it in the data directory.
)doc");

    // Full initialize with config
    m.def("initialize_with_config", [](const ballistic::InitConfig& config) {
        return ballistic::initialize(config);
    }, nb::arg("config"),
    R"doc(
Initialize the library with explicit data-file paths.

Args:
    config: InitConfig with the file paths. Explicit paths override the
        files found automatically in the data directory.

Returns:
    InitResult: what was loaded. success is False (with error_message) if a
    named file could not be loaded.

Example:
    config = ballistic.InitConfig()
    config.eop_path = "finals2000A.data"
    config.jpl_ephemeris_path = "de421.bsp"
    result = ballistic.initialize_with_config(config)

    if result.success:
        print(f"Loaded: {result.jpl_de_version}")
        print(f"Gravity degree: {result.gravity_degree_loaded}")
)doc");

    m.def("is_initialized", &ballistic::is_initialized,
    R"doc(
Check whether any data has been loaded.

Returns:
    bool: True if an external gravity or leap-second file, an EOP table, or a
    planetary ephemeris (the built-in DE440 included) is loaded.
)doc");

    m.def("build_info", []() {
        const auto b = ballistic::build_info();
        nb::dict d;
        d["with_openmp"] = b.with_openmp;
        d["compiler"] = std::string(b.compiler);
        d["cpp_standard"] = b.cpp_standard;
        d["build_type"] = std::string(b.build_type);
        return d;
    },
    R"doc(
How this copy of the library was compiled.

Returns:
    dict: ``with_openmp``, ``compiler`` (e.g.
    "GCC 13.3.0"), ``cpp_standard`` (23, 20, ...), ``build_type``
    ("Release" when compiled with NDEBUG, else "Debug").
)doc");

    namespace nrl = ballistic::atmosphere::nrl;
    m.def("_nrl_models_status", [] {
        nrl::try_autoload();
        const auto st = nrl::status();
        nb::dict d;
        d["loaded"] = st.loaded;
        d["path"] = st.path;
        d["models"] = st.models;
        d["problem"] = st.problem;
        d["install_path"] = nrl::default_install_path().string();
        d["file_name"] = std::string(nrl::library_file_name());
        d["abi_version"] = BALLISTIC_NRL_ABI_VERSION;
        return d;
    },
    "Internal (ballistic.nrl_models): look for the NRL models plugin and report its state.");
    m.def("_load_nrl_models", [](const std::string& path) { nrl::load(path); }, nb::arg("path"),
          "Internal (ballistic.install_nrl_models): load the NRL models plugin from path "
          "(ConfigError if it cannot be loaded).");

    m.def("get_init_status", &ballistic::get_init_status,
    R"doc(
Report what data is currently loaded.

Returns:
    InitResult: current status of loaded data.
)doc");

    // =========================================================================
    // Data Loading Functions
    // =========================================================================

    m.def("load_eop", [](const std::string& path) {
        return ballistic::time::EOPData::instance().load(path);
    }, nb::arg("path"),
    R"doc(
Load Earth orientation parameters (EOP: polar motion and UT1-UTC) from an
IERS finals2000A.data (or .all) file. Replaces any table already loaded.

Args:
    path: Path to the file (str).

Returns:
    bool: True if loading succeeded.

Note:
    Download from: https://datacenter.iers.org/data/9/finals2000A.data
)doc");

    m.def("load_space_weather", [](const std::string& path) {
        return ballistic::atmosphere::SpaceWeatherData::instance().load_celestrak_csv(path);
    }, nb::arg("path"),
    R"doc(
Load a CelesTrak space-weather file: daily F10.7 (the Sun's 10.7 cm radio
flux, a measure of the ultraviolet that heats the upper atmosphere) and Ap
(geomagnetic activity) indices. Drag needs these.

Args:
    path: Path to SW-All.csv or SW-Last5Years.csv (str).

Returns:
    bool: True if loading succeeded. Non-fatal problems are listed by
    space_weather_warnings().

Raises:
    ballistic.errors.BallisticError: the file lacks a column drag cannot do without
        (observed F10.7 or daily Ap).

Note:
    Download from: https://celestrak.org/SpaceData/SW-All.csv
)doc");

    m.def("eop_loaded", []() {
        return ballistic::time::EOPData::instance().is_loaded();
    }, "True while Earth orientation data (EOP) is loaded, however it was loaded.");
    m.def("set_eop_policy", [](const std::string& policy) {
        using ballistic::time::EopPolicy;
        if (policy == "require") ballistic::time::EOPData::instance().set_policy(EopPolicy::Require);
        else if (policy == "assume_zero")
            ballistic::time::EOPData::instance().set_policy(EopPolicy::AssumeZero);
        else throw ballistic::ConfigError("set_eop_policy: 'require' or 'assume_zero', not '" +
                                          policy + "'");
    }, nb::arg("policy"), R"doc(
Choose what happens when no Earth orientation (EOP) table is loaded.
Applies to the whole process.

'require' (default): converting between Earth-fixed and inertial coordinates
raises DataCoverageError saying how to load a table (ballistic.load()).
'assume_zero': UT1-UTC and polar motion are taken as zero (3-10 m error at
low Earth orbit), with one DataCoverageWarning.

Raises:
    ballistic.errors.ConfigError: policy is neither 'require' nor
        'assume_zero'.
)doc");
    m.def("eop_coverage", []() -> nb::object {
        const auto& eop = ballistic::time::EOPData::instance();
        if (!eop.is_loaded()) return nb::none();
        const auto [m0, m1] = eop.mjd_range();
        nb::dict d;
        d["first_mjd"] = m0;
        d["last_mjd"] = m1;
        const auto measured = eop.measured_until_mjd();
        d["measured_until_mjd"] = measured ? nb::cast(*measured) : nb::none();
        d["hold_days"] = ballistic::time::kEopHoldDays;
        d["blank_rows_skipped"] = eop.blank_rows_skipped();
        d["malformed_rows_skipped"] = eop.malformed_rows_skipped();
        d["rows"] = eop.size();
        return d;
    }, R"doc(
Time span of the loaded Earth orientation table, as a dict, or None if none
is loaded. Dates are Modified Julian Dates (MJD) in UTC.

Keys:
    first_mjd, last_mjd: first and last day (the last rows are IERS
        predictions, not measurements)
    measured_until_mjd: last day with a measured UT1-UTC (or None)
    hold_days: how far beyond either end the edge value is held, with a
        warning; further out the coordinate conversions raise
        DataCoverageError and a Trajectory stops
    blank_rows_skipped: empty rows after the predictions
    malformed_rows_skipped: rows that could not be read
    rows: number of rows loaded
)doc");
    m.def("space_weather_loaded", []() {
        return ballistic::atmosphere::SpaceWeatherData::instance().is_loaded();
    }, "True while space weather (F10.7 / Ap) is loaded, however it was loaded.");

    m.def("space_weather_warnings", []() {
        return ballistic::atmosphere::SpaceWeatherData::instance().warnings();
    }, "Non-fatal problems found by the last space-weather load (missing columns that "
       "drag depends on, values that could not be read), as a list of str. Empty = clean.");

    m.def("load_jpl_ephemeris", [](const std::string& path) {
        std::filesystem::path p(path);
        if (p.extension() == ".bsp" || p.extension() == ".BSP") {
            return ballistic::ephemeris::SPKEphemerisManager::instance().load(p);
        } else {
            return ballistic::ephemeris::JPLEphemerisManager::instance().load(p);
        }
    }, nb::arg("path"),
    R"doc(
Load a JPL planetary ephemeris file (de421.bsp, de440.bsp, etc.), which
gives the Sun, Moon and planet positions; replaces the built-in DE440.

Args:
    path: Path to a .bsp (SPICE SPK format, recommended) or .bin (older
        format) file (str).

Returns:
    bool: True if loading succeeded.

Note:
    Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
)doc");

    m.def("load_ocean_tides", [](const std::string& path, int max_degree) {
        return ballistic::gravity::OceanTideModel::instance().load(path, max_degree);
    }, nb::arg("path"), nb::arg("max_degree") = 100,
    R"doc(
Load an ocean-tide model: the tide-induced changes of the gravity field's
coefficients, in the IERS Conventions chapter 6 format (e.g. the FES2004
file fes2004_Cnm-Snm.dat). Once loaded, every Trajectory adds ocean tides up
to TrajectoryConfig.ocean_tide_degree (default 50). ballistic.load() does
this for you.

Args:
    path: Path to the file (str).
    max_degree: highest degree to read (default 100).

Returns:
    bool: True if loading succeeded (False if the file cannot be opened).
)doc");
    m.def("ocean_tides_loaded", []() {
        return ballistic::gravity::OceanTideModel::instance().is_loaded();
    }, "True once an ocean-tide model is loaded.");
    m.def("unload_ocean_tides", []() {
        ballistic::gravity::OceanTideModel::instance().clear();
    }, "Unload the ocean-tide model (propagation then runs without ocean tides).");

    m.def("get_space_weather", [](const ballistic::Epoch& epoch, bool allow_predicted) {
        bool used = false;
        auto w = ballistic::atmosphere::SpaceWeatherData::instance().get_conditions(
            epoch, allow_predicted, &used);
        if (used && PyErr_WarnEx(PyExc_RuntimeWarning,
                "get_space_weather: no measurement for this epoch; returning "
                "CelesTrak's forecast", 1) != 0)
            throw nb::python_error();
        return w;
    }, nb::arg("epoch"), nb::arg("allow_predicted") = true,
    R"doc(
Space weather (F10.7 and Ap) at a given time, from the loaded file.

Args:
    epoch: Time to query (Epoch).
    allow_predicted: accept CelesTrak's forecast past the last measured day
        (default True).

Returns:
    WeatherConditions: F10.7 and Ap values for the given time.

Past the last measured day returns CelesTrak's daily forecast with a
RuntimeWarning (allow_predicted=False raises instead). Raises
DataCoverageError when nothing is loaded or the epoch is outside the file;
there is no fallback to quiet defaults or the nearest row.
)doc");

    m.def("load_leap_seconds", [](const std::string& path, bool verbose) -> nb::tuple {
        auto [success, error] = ballistic::load_leap_seconds(path, verbose);
        return nb::make_tuple(success, error);
    }, nb::arg("path"), nb::arg("verbose") = true,
    R"doc(
Load the IERS leap-second table from a leap-seconds.list file.

Args:
    path: Path to the leap-seconds.list file (str).
    verbose: If True, the error message on failure includes the download URL.

Returns:
    Tuple[bool, str]: (success, error_message). error_message is empty on success.

Note:
    Download from: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
    Alternative: https://www.ietf.org/timezones/data/leap-seconds.list

    The embedded leap seconds are current through 2017 (TAI-UTC=37s).
    No new leap seconds have been announced since then.
)doc");

    m.def("get_data_directory", []() {
        return ballistic::get_data_directory().string();
    },
    R"doc(
The default directory where Ballistic looks for, and downloads, data files.

Returns:
    str: Path to the default data directory.

Platform-specific locations:
    Windows: %LOCALAPPDATA%\ballistic  (C:\Users\<user>\AppData\Local\ballistic)
    Linux:   ~/.local/share/ballistic
    macOS:   ~/Library/Application Support/ballistic

Note:
    The directory may not exist yet; create it before saving files there.
)doc");

    m.def("get_leap_second_info", &ballistic::get_leap_second_info,
    R"doc(
Information about the leap-second table in use.

Returns:
    LeapSecondInfo: current state of the leap-second data.

Example:
    info = ballistic.get_leap_second_info()
    print(f"Current TAI-UTC: {info.current_tai_utc} seconds")
    print(f"Entries: {info.entry_count}")
    print(f"From file: {info.loaded_from_file}")
)doc");


}
