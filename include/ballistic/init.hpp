#pragma once

// =============================================================================
// Ballistic Library Global Initialization
// =============================================================================
//
// initialize() loads the data the library needs, once, at program start:
// the gravity model, the planetary ephemeris, leap seconds and, if found,
// the Earth orientation table. Call it before building any Trajectory.
//
// Embedded data (no external files required):
// - EGM2008 gravity-model coefficients: degrees 2-360 (a Trajectory uses
//   degree 70 by default, TrajectoryConfig::gravity_degree)
// - DE440 ephemeris (JPL's table of Sun/Moon/planet positions), 1950-2100,
//   ~6.6 MB embedded
// - Leap seconds: through the latest one (2017, TAI-UTC = 37 s; none since),
//   valid until the expiry date IERS announced for that table
//
// Auto-discovered from the default data directory (if present):
// - finals2000A.data / finals2000A.all: the IERS table of Earth orientation
//   parameters (EOP): polar motion (the wobble of the rotation axis) and
//   UT1-UTC (the Earth's rotation angle expressed as a clock offset).
//   Required for sub-meter conversions between the Earth-fixed (ECEF/ITRF)
//   and inertial (GCRS) frames: an error of ~0.1 arcsec is ~3 m at the
//   Earth's surface. Without a table the frame transforms raise
//   DataCoverageError unless EopPolicy::AssumeZero is set (then 3-10 m at
//   LEO). Populate the data directory by running ballistic.Loader().load_eop()
//   once (Python) or downloading the file manually; later initialize({})
//   calls pick it up automatically.
// - leap-seconds.list: a newer IERS leap-seconds table.
// - nrl/ballistic_nrl.<so|dylib|dll>: the NRLMSIS 2.1 / HWM14 plugin for
//   model-based drag (atmosphere/nrl_models.hpp). Not part of the library
//   because NRLMSIS 2.1 is licensed for non-commercial use only; also found
//   through the BALLISTIC_NRL_MODELS environment variable.
//
// External data files (optional, for extended coverage):
// - EGM2008 coefficients: degrees 361-2190 (from the NGA file)
// - JPL ephemeris: extended time range or other DE versions
// - Space weather (solar flux F10.7 and geomagnetic index Ap, for drag):
//   loaded separately, see atmosphere/space_weather.hpp
//
// Default data directory (platform-specific):
//   Windows: %LOCALAPPDATA%\ballistic  (C:\Users\<user>\AppData\Local\ballistic)
//   Linux:   ~/.local/share/ballistic
//   macOS:   ~/Library/Application Support/ballistic
//
// Usage:
//   // Simple usage - embedded data + auto-discovered EOP/leap-seconds
//   ballistic::initialize({});
//
//   // Explicit paths (override auto-discovery)
//   ballistic::InitConfig config;
//   config.egm2008_path = "path/to/EGM2008_to2190_TideFree";
//   config.eop_path = "path/to/finals2000A.data";
//   config.jpl_ephemeris_path = "path/to/de421.bsp";  // Overrides embedded DE440
//   config.leap_seconds_path = "path/to/leap-seconds.list";
//   ballistic::initialize(config);
//
// Notes:
// - All embedded EGM2008 coefficients (degrees 2-360) are always loaded.
//   The per-propagation degree used by the integrator is set on
//   TrajectoryConfig::gravity_degree (default 70 for LEO).
// - When config.egm2008_path is set, the full file is loaded
//   (degrees up to 2190 for the standard NGA distribution).
// - Embedded DE440 ephemeris (1950-2100) is always loaded unless overridden
//   by config.jpl_ephemeris_path.
// - EOP / leap-seconds files in the data directory are always auto-discovered
//   when not explicitly set. For hermetic test fixtures, point
//   XDG_DATA_HOME (Linux) at an empty directory.
//
// =============================================================================

#include "core/diagnostics.hpp"
#include "gravity/egm_gravity.hpp"
#include "time/eop_data.hpp"
#include "time/epoch.hpp"
#include "ephemeris/jpl_ephemeris.hpp"
#include "ephemeris/spk_ephemeris.hpp"
#include "core/data_paths.hpp"
#include "core/errors.hpp"
#include "atmosphere/nrl_models.hpp"
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ballistic {

/// Optional file paths for initialize(). Every path that is set is strict:
/// a missing or unparseable file makes initialize() return success = false.
struct InitConfig {
    // Path to EGM2008 coefficient file (e.g., "EGM2008_to2190_TideFree")
    // Only needed for degrees > 360. Embedded coefficients cover degrees 2-360.
    std::optional<std::filesystem::path> egm2008_path;

    // Path to IERS finals2000A.data file for EOP (polar motion, UT1-UTC)
    // Download from: https://datacenter.iers.org/data/9/finals2000A.data
    // If not set, the data directory is searched; if nothing is found, no EOP
    // is loaded and the frame transforms raise DataCoverageError (unless
    // EopPolicy::AssumeZero is set).
    std::optional<std::filesystem::path> eop_path;

    // Path to JPL DE ephemeris file - supports two formats:
    // - SPK format (.bsp): de421.bsp, de440.bsp, de441.bsp (recommended)
    // - Old binary format: de440.bin, de441.bin
    // Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    // If set, this file overrides the embedded DE440 ephemeris.
    // If not set, uses embedded DE440 (1950-2100, sub-meter accuracy).
    std::optional<std::filesystem::path> jpl_ephemeris_path;

    // Path to IERS leap-seconds.list file
    // Download from: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
    // If not set, uses embedded leap seconds (current through 2017, TAI-UTC=37s)
    std::optional<std::filesystem::path> leap_seconds_path;

    // Path to the NRL models plugin (ballistic_nrl.so / .dylib / .dll; CMake
    // target ballistic_nrl), needed only for model-based drag
    // (make_drag_environment, from_models_*). If not set, the environment
    // variable BALLISTIC_NRL_MODELS and then <data directory>/nrl/ are
    // searched; if nothing is found, model-based drag raises ConfigError.
    // NRLMSIS 2.1 is licensed for research, academic and non-profit use only.
    std::optional<std::filesystem::path> nrl_models_path;
};

/// What initialize() (or get_init_status()) found and loaded.
struct InitResult {
    bool success = true;                   // Overall success
    bool egm2008_loaded = false;           // EGM2008 coefficients loaded from file
    bool eop_loaded = false;               // EOP data loaded
    bool eop_auto_discovered = false;      // EOP came from data-dir auto-discovery (not explicit path)
    bool jpl_ephemeris_loaded = false;     // JPL ephemeris loaded from external file
    bool embedded_ephemeris_loaded = false; // Embedded DE440 ephemeris loaded
    bool leap_seconds_loaded = false;      // Leap seconds loaded from file
    bool leap_seconds_auto_discovered = false; // Leap seconds came from auto-discovery
    int gravity_degree_loaded = 0;         // Actual gravity degree available
    std::string jpl_de_version;            // "DE440 (embedded)" or file version
    std::string eop_source_path;           // Path of loaded EOP file (auto or explicit), empty if none
    bool nrl_models_loaded = false;        // NRLMSIS 2.1 / HWM14 plugin loaded (model-based drag)
    std::string nrl_models_path;           // Path of the loaded plugin, empty if none
    std::string error_message;             // Error details if success=false

    // Staleness diagnostics (success is unaffected; check the flags / warnings).
    // IERS finals files carry ~1 year of *predicted* EOP past the last
    // observation, so the staleness test is "table end < now" — i.e. even
    // the predictions are exhausted. Past the table end the lookups hold the
    // edge value for up to 30 days, with a warning (meters at LEO), and raise
    // DataCoverageError beyond. Refresh via ballistic.Loader().load_eop()
    // (Python) or a fresh finals2000A download.
    double eop_table_end_mjd = 0.0;        // Last MJD in the loaded EOP table (0 = no EOP)
    bool eop_table_stale = false;          // EOP loaded but table end < now
    // Leap-second files declare their own expiry (#@ line); the embedded
    // table carries the expiry IERS announced for it. Past the expiry a newly
    // announced leap second would be missed.
    double leap_seconds_expiration_jd = 0.0; // Julian date of the expiry; 0 = unknown
    bool leap_seconds_expired = false;     // expiry < now
    std::vector<std::string> warnings;     // Human-readable staleness warnings
};

/// Current UTC from the system clock as a Modified Julian Date (MJD: days
/// since 1858-11-17 00:00; the Unix epoch 1970-01-01 is MJD 40587). Ignores
/// leap seconds, which is fine for the day-level staleness checks it serves.
inline double current_utc_mjd() {
    const double unix_s = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return unix_s / 86400.0 + 40587.0;
}

/// Fill InitResult's staleness fields and warnings from the currently loaded
/// EOP and leap-second tables. Called at the end of initialize() and by
/// get_init_status(); never affects `success`.
inline void fill_staleness_diagnostics(InitResult& result) {
    const double now_mjd = current_utc_mjd();

    if (time::EOPData::instance().is_loaded()) {
        const auto [mjd_begin, mjd_end] = time::EOPData::instance().mjd_range();
        (void)mjd_begin;
        result.eop_table_end_mjd = mjd_end;
        if (mjd_end > 0.0 && mjd_end < now_mjd) {
            result.eop_table_stale = true;
            result.warnings.push_back(
                "EOP table ends at MJD " + std::to_string(mjd_end) +
                ", which is in the past (even the ~1-year prediction window is "
                "exhausted): the edge value is held for 30 days past it (meters at "
                "LEO), then the transforms raise and a Trajectory stops. Refresh "
                "finals2000A via ballistic.load().");
        }
    }

    const double leap_exp_jd = time::LeapSecondTable::instance().expiration_jd();
    result.leap_seconds_expiration_jd = leap_exp_jd;
    const double now_jd = now_mjd + 2400000.5;
    if (leap_exp_jd > 0.0 && leap_exp_jd < now_jd) {
        result.leap_seconds_expired = true;
        result.warnings.push_back(
            "Leap-second table expired (file-declared expiry JD " +
            std::to_string(leap_exp_jd) + " is in the past). A leap second "
            "announced since then would be missed. Refresh leap-seconds.list "
            "via ballistic.Loader().timescale().");
    }
}

/// A data file was found but not (fully) usable: recorded in the result's
/// warnings and reported once through the diagnostics channel (DataSkipped).
inline void report_init_problem(InitResult& result, const std::string& what) {
    result.warnings.push_back(what);
    diagnostics::report(diagnostics::Category::DataSkipped, "initialize: " + what,
                        "init:" + what);
}

/// Look for one of `candidate_filenames` in the platform's default data
/// directory. Returns the first existing path, or std::nullopt.
inline std::optional<std::filesystem::path> find_in_default_data_directory(
    std::initializer_list<const char*> candidate_filenames)
{
    std::error_code ec;
    const auto data_dir = get_default_data_directory();
    if (!std::filesystem::exists(data_dir, ec)) {
        return std::nullopt;
    }
    for (const char* name : candidate_filenames) {
        std::filesystem::path candidate = data_dir / name;
        if (std::filesystem::exists(candidate, ec) && !ec) {
            return candidate;
        }
    }
    return std::nullopt;
}

/// Load the library's data (see the file comment): the gravity model, the
/// ephemeris, leap seconds and EOP, from `config`'s paths or the data
/// directory. Call once at program start, before building propagators.
/// Does not throw for file problems: an explicitly given file that is missing
/// or unparseable returns success = false with error_message set; an
/// auto-discovered file that cannot be read is skipped with a warning.
inline InitResult initialize(const InitConfig& config) {
    InitResult result;
    result.success = true;

    // Load EGM2008 coefficients if path specified (loads full file, up to deg 2190).
    // Explicit paths are strict: missing or unparseable files fail init.
    if (config.egm2008_path.has_value()) {
        const auto& path = config.egm2008_path.value();
        if (std::filesystem::exists(path)) {
            if (gravity::EGM2008Gravity::instance().load_full(path)) {
                result.egm2008_loaded = true;
                result.gravity_degree_loaded = gravity::EGM2008Gravity::instance().loaded_degree();
            } else {
                result.success = false;
                result.error_message = generate_data_file_error(DataFileType::EGM2008, path, true);
                return result;
            }
        } else {
            result.success = false;
            result.error_message = generate_data_file_error(DataFileType::EGM2008, path, false);
            return result;
        }
    }

    // If EGM2008 not loaded from file, use embedded coefficients (all degrees 2-360)
    if (!result.egm2008_loaded) {
        gravity::EGM2008Gravity::instance().init();
        result.gravity_degree_loaded = gravity::EGM2008Gravity::instance().max_degree();
    }

    // Load EOP data: explicit path takes priority; otherwise auto-discover
    // finals2000A.data / finals2000A.all in the default data directory.
    // Explicit paths are strict (missing/unparseable -> init fails); an
    // auto-discovered file that does not parse is skipped and reported.
    {
        std::optional<std::filesystem::path> eop_to_load = config.eop_path;
        bool auto_discovered = false;
        if (!eop_to_load.has_value()) {
            eop_to_load = find_in_default_data_directory(
                {"finals2000A.data", "finals2000A.all"});
            auto_discovered = eop_to_load.has_value();
        }
        if (eop_to_load.has_value()) {
            const auto& path = eop_to_load.value();
            if (std::filesystem::exists(path)) {
                if (time::EOPData::instance().load(path.string())) {
                    result.eop_loaded = true;
                    result.eop_auto_discovered = auto_discovered;
                    result.eop_source_path = path.string();
                    if (const auto bad = time::EOPData::instance().malformed_rows_skipped(); bad > 0)
                        report_init_problem(result, path.string() + ": " + std::to_string(bad) +
                            " malformed rows skipped (the file may be truncated or corrupt)");
                } else if (auto_discovered) {
                    // Report it: an auto-discovered file must not be skipped silently.
                    report_init_problem(result, "found " + path.string() + " in the data "
                        "directory but could not parse it; no Earth orientation data is "
                        "loaded. Delete it and run ballistic.load() to fetch a fresh copy");
                } else {
                    result.success = false;
                    result.error_message = generate_data_file_error(DataFileType::EOP, path, true);
                    return result;
                }
            } else if (!auto_discovered) {
                result.success = false;
                result.error_message = generate_data_file_error(DataFileType::EOP, path, false);
                return result;
            }
        }
    }

    // Load leap seconds: explicit path takes priority; otherwise auto-discover
    // leap-seconds.list in the default data directory. Same strict-explicit /
    // best-effort-auto-discovery policy as EOP.
    {
        std::optional<std::filesystem::path> ls_to_load = config.leap_seconds_path;
        bool auto_discovered = false;
        if (!ls_to_load.has_value()) {
            ls_to_load = find_in_default_data_directory({"leap-seconds.list"});
            auto_discovered = ls_to_load.has_value();
        }
        if (ls_to_load.has_value()) {
            const auto& path = ls_to_load.value();
            if (std::filesystem::exists(path)) {
                if (time::LeapSecondTable::instance().load(path)) {
                    result.leap_seconds_loaded = true;
                    result.leap_seconds_auto_discovered = auto_discovered;
                } else if (auto_discovered) {
                    report_init_problem(result, "found " + path.string() + " in the data "
                        "directory but could not parse it (" +
                        time::LeapSecondTable::instance().last_error() + "); the embedded "
                        "leap-second table is used. Delete it and run ballistic.load()");
                } else {
                    result.success = false;
                    result.error_message = generate_data_file_error(DataFileType::LEAP_SECONDS, path, true,
                        time::LeapSecondTable::instance().last_error());
                    return result;
                }
            } else if (!auto_discovered) {
                result.success = false;
                result.error_message = generate_data_file_error(DataFileType::LEAP_SECONDS, path, false);
                return result;
            }
        }
    }

    // Load JPL ephemeris
    // Priority: external file (if specified) > embedded DE440
    bool external_ephemeris_loaded = false;

    if (config.jpl_ephemeris_path.has_value()) {
        const auto& path = config.jpl_ephemeris_path.value();
        if (std::filesystem::exists(path)) {
            // Try SPK format first (recommended for de421.bsp, de440.bsp, de441.bsp)
            if (path.extension() == ".bsp" || path.extension() == ".BSP") {
                if (ephemeris::SPKEphemerisManager::instance().load(path)) {
                    result.jpl_ephemeris_loaded = true;
                    external_ephemeris_loaded = true;
                    // Extract DE version from filename (e.g., "de421" from "de421.bsp")
                    std::string filename = path.stem().string();
                    if (filename.size() >= 5 && (filename.substr(0, 2) == "de" || filename.substr(0, 2) == "DE")) {
                        result.jpl_de_version = "DE" + filename.substr(2);
                    } else {
                        result.jpl_de_version = "SPK";
                    }
                }
            }

            // Fall back to old binary format
            if (!external_ephemeris_loaded) {
                if (ephemeris::JPLEphemerisManager::instance().load(path)) {
                    // get_default_ephemeris_provider() prefers a loaded SPK
                    // (the embedded DE440 loads itself on first use), so the
                    // SPK is dropped: otherwise this file would be ignored.
                    ephemeris::SPKEphemerisManager::instance().unload(
                        "replaced by the DE file " + path.string() +
                        " given to initialize()");
                    result.jpl_ephemeris_loaded = true;
                    external_ephemeris_loaded = true;
                    result.jpl_de_version = "DE" + std::to_string(
                        ephemeris::JPLEphemerisManager::instance().ephemeris().de_version());
                }
            }

            if (!external_ephemeris_loaded) {
                result.success = false;
                result.error_message = generate_data_file_error(DataFileType::JPL_EPHEMERIS, path, true);
                return result;
            }
        } else {
            result.success = false;
            result.error_message = generate_data_file_error(DataFileType::JPL_EPHEMERIS, path, false);
            return result;
        }
    }

    // Load embedded DE440 if no external file was loaded.
    if (!external_ephemeris_loaded) {
        if (ephemeris::SPKEphemerisManager::instance().load_embedded()) {
            result.embedded_ephemeris_loaded = true;
            result.jpl_de_version = "DE440 (embedded)";
        }
    }

    // NRL models plugin: an explicit path is strict; otherwise look for it
    // (BALLISTIC_NRL_MODELS, data directory) and report a file found there
    // that cannot be loaded. Absent is normal: only model-based drag needs it.
    if (config.nrl_models_path.has_value()) {
        try {
            atmosphere::nrl::load(*config.nrl_models_path);
        } catch (const ConfigError& e) {
            result.success = false;
            result.error_message = e.what();
            return result;
        }
    } else if (!atmosphere::nrl::try_autoload()) {
        if (const auto st = atmosphere::nrl::status(); !st.problem.empty()) {
            report_init_problem(result, "the NRL models plugin was found but not loaded (" +
                st.problem + "); model-based drag will raise ConfigError until it is "
                "reinstalled (Python: ballistic.install_nrl_models(accept_license=True))");
        }
    }
    {
        const auto st = atmosphere::nrl::status();
        result.nrl_models_loaded = st.loaded;
        result.nrl_models_path = st.path;
    }

    fill_staleness_diagnostics(result);

    return result;
}

/// Load an IERS leap-seconds.list file, replacing the embedded table.
/// Download from: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
/// Returns {true, ""} on success, or {false, message}; the message explains
/// how to fix it when verbose is true, and is one line otherwise.
inline std::pair<bool, std::string> load_leap_seconds(
    const std::filesystem::path& path,
    bool verbose = true)
{
    if (!std::filesystem::exists(path)) {
        std::string error = verbose
            ? generate_data_file_error(DataFileType::LEAP_SECONDS, path, false)
            : "File not found: " + path.string();
        return {false, error};
    }

    if (time::LeapSecondTable::instance().load(path)) {
        return {true, ""};
    }

    std::string error = verbose
        ? generate_data_file_error(DataFileType::LEAP_SECONDS, path, true,
            time::LeapSecondTable::instance().last_error())
        : "Failed to parse: " + path.string();
    return {false, error};
}

/// True if any data has been loaded beyond the built-in defaults: a gravity
/// or leap-second file, an EOP table, or an ephemeris (embedded DE440 included).
inline bool is_initialized() {
    return gravity::EGM2008Gravity::instance().is_loaded_from_file() ||
           time::EOPData::instance().is_loaded() ||
           time::LeapSecondTable::instance().is_loaded_from_file() ||
           ephemeris::JPLEphemerisManager::instance().is_loaded() ||
           ephemeris::SPKEphemerisManager::instance().is_loaded();
}

/// The current load state, in the same form initialize() returns (without
/// the source path and the auto-discovery flags it recorded).
inline InitResult get_init_status() {
    InitResult result;
    result.success = true;
    result.egm2008_loaded = gravity::EGM2008Gravity::instance().is_loaded_from_file();
    result.gravity_degree_loaded = gravity::EGM2008Gravity::instance().max_degree();
    result.eop_loaded = time::EOPData::instance().is_loaded();
    result.leap_seconds_loaded = time::LeapSecondTable::instance().is_loaded_from_file();
    {
        const auto st = atmosphere::nrl::status();
        result.nrl_models_loaded = st.loaded;
        result.nrl_models_path = st.path;
    }

    // Check ephemeris status
    if (ephemeris::SPKEphemerisManager::instance().is_loaded()) {
        if (ephemeris::SPKEphemerisManager::instance().is_using_embedded()) {
            result.embedded_ephemeris_loaded = true;
            result.jpl_de_version = "DE440 (embedded)";
        } else {
            result.jpl_ephemeris_loaded = true;
            result.jpl_de_version = "DE" + std::to_string(
                ephemeris::SPKEphemerisManager::instance().ephemeris().de_version());
        }
    } else if (ephemeris::JPLEphemerisManager::instance().is_loaded()) {
        result.jpl_ephemeris_loaded = true;
        result.jpl_de_version = "DE" + std::to_string(
            ephemeris::JPLEphemerisManager::instance().ephemeris().de_version());
    }

    fill_staleness_diagnostics(result);
    return result;
}

/// The platform's default data directory (see the file comment).
inline std::filesystem::path get_data_directory() {
    return get_default_data_directory();
}

/// Summary of the leap-second table in use.
struct LeapSecondInfo {
    int current_tai_utc;       // Current TAI-UTC offset (latest value)
    size_t entry_count;        // Number of leap second entries
    bool loaded_from_file;     // Whether loaded from external file
    double expiration_jd;      // Expiration date (JD) if known, 0 otherwise
};

/// Summary of the leap-second table in use.
inline LeapSecondInfo get_leap_second_info() {
    LeapSecondInfo info;
    info.current_tai_utc = time::LeapSecondTable::instance().current_tai_utc();
    info.entry_count = time::LeapSecondTable::instance().size();
    info.loaded_from_file = time::LeapSecondTable::instance().is_loaded_from_file();
    info.expiration_jd = time::LeapSecondTable::instance().expiration_jd();
    return info;
}

} // namespace ballistic
