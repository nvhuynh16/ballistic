#pragma once

// =============================================================================
// Data File Path Utilities and Error Message Generation
// =============================================================================
//
// This module provides:
// - Platform-specific default data directory detection
// - Detailed error messages with download URLs and file placement guidance
// - Data file type enumeration and metadata
//
// =============================================================================

#include <string>
#include <filesystem>
#include <sstream>

#ifdef _WIN32
// Keep <windows.h> (pulled in by <shlobj.h>) from defining min/max macros,
// which break std::min/std::max, and from dragging in rarely used APIs.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <shlobj.h>
#pragma comment(lib, "shell32.lib")
#else
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif

namespace ballistic {

/// The external data files the library can use.
enum class DataFileType {
    EOP,            // Earth Orientation Parameters (finals2000A.data): Earth rotation and pole
    JPL_EPHEMERIS,  // JPL planetary ephemeris (de421.bsp, de440.bsp)
    SPACE_WEATHER,  // CelesTrak space weather (SW-All.csv)
    EGM2008,        // Extended gravity coefficients (EGM2008_to2190_TideFree)
    LEAP_SECONDS    // IERS leap seconds (leap-seconds.list)
};

/// The directory for the library's data files (it may not exist yet):
///   BALLISTIC_DATA_DIR, when set, on every platform (the directory itself);
///   otherwise the platform's default:
///   Windows: %LOCALAPPDATA%\ballistic  (e.g., C:\Users\<user>\AppData\Local\ballistic;
///            the LOCALAPPDATA variable if set, else the system's known folder,
///            else ./ballistic_data)
///   Linux:   $XDG_DATA_HOME/ballistic, else ~/.local/share/ballistic
///   macOS:   ~/Library/Application Support/ballistic
inline std::filesystem::path get_default_data_directory() {
#ifdef _WIN32
    if (const wchar_t* dir = _wgetenv(L"BALLISTIC_DATA_DIR"); dir && *dir) {
        return std::filesystem::path(dir);
    }
    if (const wchar_t* local = _wgetenv(L"LOCALAPPDATA"); local && *local) {
        return std::filesystem::path(local) / "ballistic";
    }
    wchar_t* path = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &path))) {
        std::filesystem::path result(path);
        CoTaskMemFree(path);
        return result / "ballistic";
    }
    // Fallback to current directory
    return std::filesystem::current_path() / "ballistic_data";
#else
    if (const char* dir = getenv("BALLISTIC_DATA_DIR"); dir && *dir) {
        return std::filesystem::path(dir);
    }
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pw = getpwuid(getuid());
        home = pw ? pw->pw_dir : ".";
    }
#if defined(__APPLE__)
    return std::filesystem::path(home) / "Library" / "Application Support" / "ballistic";
#else
    const char* xdg_data = getenv("XDG_DATA_HOME");
    if (xdg_data && *xdg_data) {
        return std::filesystem::path(xdg_data) / "ballistic";
    }
    return std::filesystem::path(home) / ".local" / "share" / "ballistic";
#endif
#endif
}

/// Where to download a data file of `type` (empty string for an unknown type).
inline std::string get_download_url(DataFileType type) {
    switch (type) {
        case DataFileType::EOP:
            return "https://datacenter.iers.org/data/9/finals2000A.data";
        case DataFileType::JPL_EPHEMERIS:
            return "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/";
        case DataFileType::SPACE_WEATHER:
            return "https://celestrak.org/SpaceData/SW-All.csv";
        case DataFileType::EGM2008:
            return "https://earth-info.nga.mil/php/download.php?file=egm-08coeffs";
        case DataFileType::LEAP_SECONDS:
            return "https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list";
        default:
            return "";
    }
}

/// The file name the library looks for in the data directory.
inline std::string get_recommended_filename(DataFileType type) {
    switch (type) {
        case DataFileType::EOP:
            return "finals2000A.data";
        case DataFileType::JPL_EPHEMERIS:
            return "de421.bsp";  // or de440.bsp
        case DataFileType::SPACE_WEATHER:
            return "SW-All.csv";
        case DataFileType::EGM2008:
            return "EGM2008_to2190_TideFree";
        case DataFileType::LEAP_SECONDS:
            return "leap-seconds.list";
        default:
            return "";
    }
}

/// A human-readable name for the file type, e.g. "Earth Orientation Parameters".
inline std::string get_file_type_name(DataFileType type) {
    switch (type) {
        case DataFileType::EOP:
            return "Earth Orientation Parameters";
        case DataFileType::JPL_EPHEMERIS:
            return "JPL Planetary Ephemeris";
        case DataFileType::SPACE_WEATHER:
            return "Space Weather Data";
        case DataFileType::EGM2008:
            return "EGM2008 Gravity Coefficients";
        case DataFileType::LEAP_SECONDS:
            return "Leap Seconds Table";
        default:
            return "Unknown";
    }
}

/// A multi-line message for a data file that could not be loaded: what failed
/// (`file_exists` false = not found at `attempted_path`, true = found but unparseable,
/// with `parse_error` if given), where to download it and where to put it.
inline std::string generate_data_file_error(
    DataFileType type,
    const std::filesystem::path& attempted_path,
    bool file_exists,
    const std::string& parse_error = "")
{
    std::ostringstream oss;
    auto default_dir = get_default_data_directory();
    auto filename = get_recommended_filename(type);
    auto url = get_download_url(type);
    auto type_name = get_file_type_name(type);

    oss << "Failed to load " << type_name << ".\n\n";

    if (file_exists) {
        oss << "File found but could not be parsed:\n";
        oss << "  " << attempted_path.string() << "\n\n";
        if (!parse_error.empty()) {
            oss << "Parse error: " << parse_error << "\n\n";
        }
    } else {
        oss << "File not found:\n";
        oss << "  " << attempted_path.string() << "\n\n";
    }

    oss << "To download:\n";

    switch (type) {
        case DataFileType::EOP:
            oss << "  1. Visit: " << url << "\n";
            oss << "  2. Save to: " << (default_dir / filename).string() << "\n";
            oss << "     (or specify path via InitConfig.eop_path)\n\n";
            oss << "Without EOP data the GCRS <-> ITRF transforms raise DataCoverageError\n";
            oss << "(EopPolicy::AssumeZero accepts zero polar motion and UT1-UTC instead,\n";
            oss << "which costs 3-10 m at LEO).\n";
            break;

        case DataFileType::JPL_EPHEMERIS:
            oss << "  1. Visit: " << url << "\n";
            oss << "  2. Download one of:\n";
            oss << "     - de421.bsp (14 MB) - ESA operational, recommended for Sentinel\n";
            oss << "     - de440.bsp (114 MB) - Latest JPL ephemeris\n";
            oss << "  3. Save to: " << (default_dir / "de421.bsp").string() << "\n";
            oss << "     (or specify path via InitConfig.jpl_ephemeris_path)\n\n";
            oss << "Note: The library includes embedded DE440 (1950-2100) by default.\n";
            oss << "External files are only needed for:\n";
            oss << "  - ESA compatibility (use de421.bsp)\n";
            oss << "  - Dates outside 1950-2100 range\n";
            break;

        case DataFileType::SPACE_WEATHER:
            oss << "  1. Visit: " << url << "\n";
            oss << "  2. Save to: " << (default_dir / filename).string() << "\n";
            oss << "     (or specify path via load_space_weather())\n\n";
            oss << "Without space weather data, building a drag environment raises\n";
            oss << "DataCoverageError; there is no default F10.7/Ap.\n";
            break;

        case DataFileType::EGM2008:
            oss << "  1. Visit: " << url << "\n";
            oss << "  2. Extract and save to: " << (default_dir / filename).string() << "\n";
            oss << "     (or specify path via InitConfig.egm2008_path)\n\n";
            oss << "Note: Embedded coefficients cover degrees 2-360, sufficient for most\n";
            oss << "applications. External file only needed for degrees 361-2190.\n";
            break;

        case DataFileType::LEAP_SECONDS:
            oss << "  1. Visit: " << url << "\n";
            oss << "  2. Save to: " << (default_dir / filename).string() << "\n";
            oss << "     (or specify path via load_leap_seconds())\n\n";
            oss << "Alternative sources:\n";
            oss << "  - https://www.ietf.org/timezones/data/leap-seconds.list\n";
            oss << "  - IERS Bulletin C: https://datacenter.iers.org/productMetadata.php?id=16\n\n";
            oss << "Note: Embedded leap seconds are current through 2017 (TAI-UTC=37s).\n";
            oss << "No new leap seconds have been announced since then.\n";
            break;
    }

    return oss.str();
}

} // namespace ballistic
