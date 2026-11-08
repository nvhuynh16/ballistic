#pragma once

// Prevent Windows.h min/max macros from conflicting with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// =============================================================================
// ESA Sentinel Earth Explorer Orbit File (EOF) Parser
// =============================================================================
//
// Parses Precise Orbit Ephemerides (POE) files from ESA Sentinel missions:
// ESA's best after-the-fact orbit solutions, written as orbit state vectors
// (OSV: time, position, velocity) at 10-second intervals.
//
// File format: XML with Earth_Explorer_File structure
// Reference frame: EARTH_FIXED (ECEF, Earth-centered Earth-fixed; positions in
// meters, velocities in m/s)
// Time reference: UTC (also includes TAI and UT1; this reader uses UTC)
//
// Typical accuracy: ~5cm position, ~0.5mm/s velocity (POE products)
//
// Reference:
// ESA Earth Observation Ground Segment File Format Standard
// https://sentinels.copernicus.eu/
// =============================================================================

#include "../core/types.hpp"
#include "../core/defaults.hpp"
#include "../core/diagnostics.hpp"
#include "../core/errors.hpp"
#include "../time/epoch.hpp"
#include "../math/interpolation.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <optional>
#include <regex>
#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <array>
#include <ostream>

namespace ballistic {
namespace validation {

/// One orbit state vector (OSV) from an EOF file.
struct EOFStateVector {
    Epoch epoch;          // UTC time
    Vector3d position;    // ECEF position [m]
    Vector3d velocity;    // ECEF velocity [m/s]
    int orbit_number;     // Absolute orbit number
    std::string quality;  // Quality flag (NOMINAL, etc.)
};

/// A parsed EOF file.
struct EOFFile {
    std::string filename;          // as passed to the parser (may be empty)
    std::string mission;           // e.g., "Sentinel-1A"
    std::string reference_frame;   // e.g., "EARTH_FIXED"
    Epoch validity_start;          // validity period stated in the header (UTC)
    Epoch validity_stop;
    std::vector<EOFStateVector> state_vectors;
    int skipped_records = 0;       // OSVs that did not parse (reported once)

    /// The state vector nearest in time to `target`, or nullopt when there is
    /// none within defaults::EPOCH_PROXIMITY_SECONDS (60 s) or the file is
    /// empty. Linear search.
    [[nodiscard]] std::optional<EOFStateVector> find_nearest(const Epoch& target) const {
        if (state_vectors.empty()) return std::nullopt;

        double min_diff = (std::numeric_limits<double>::max)();
        const EOFStateVector* best = nullptr;

        for (const auto& sv : state_vectors) {
            double diff = std::abs(sv.epoch - target);
            if (diff < min_diff) {
                min_diff = diff;
                best = &sv;
            }
        }

        if (best && min_diff < defaults::EPOCH_PROXIMITY_SECONDS) {
            return *best;
        }
        return std::nullopt;
    }

    /// Copies of the state vectors with start <= epoch <= end.
    [[nodiscard]] std::vector<EOFStateVector> get_range(
        const Epoch& start, const Epoch& end) const
    {
        std::vector<EOFStateVector> result;
        for (const auto& sv : state_vectors) {
            if (sv.epoch >= start && sv.epoch <= end) {
                result.push_back(sv);
            }
        }
        return result;
    }

    /// ECEF position (m) at `target`, by Lagrange polynomial interpolation
    /// through `order` state vectors (default 9), using exact time offsets
    /// from the first state vector. Returns nullopt outside the state vectors'
    /// span (no extrapolation) or with fewer than `order` vectors.
    [[nodiscard]] std::optional<Vector3d> interpolate_position(
        const Epoch& target, int order = defaults::LAGRANGE_DEFAULT_ORDER) const
    {
        if (state_vectors.size() < static_cast<size_t>(order)) {
            return std::nullopt;
        }
        const Epoch& e0 = state_vectors.front().epoch;
        const double t = target - e0;
        if (!(t >= 0.0 && t <= state_vectors.back().epoch - e0)) {
            return std::nullopt;
        }
        auto get_time = [&](size_t i) { return state_vectors[i].epoch - e0; };
        auto get_value = [&](size_t i) { return state_vectors[i].position; };

        return math::detail::lagrange_interpolate_vector(
            get_time, get_value, state_vectors.size(), t, order);
    }
};

/// Reader for ESA Earth Explorer orbit files (EOF).
class EOFParser {
public:
    /// Reads and parses the EOF file `filename`. Returns nullopt when the file
    /// cannot be opened or holds no usable state vector; throws as
    /// parse_content.
    [[nodiscard]] static std::optional<EOFFile> parse(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return std::nullopt;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        return parse_content(content, filename);
    }

    /// Parses EOF text; `filename` is used only in messages. Returns nullopt
    /// when no state vector parses. A state vector with a malformed time,
    /// position or velocity is skipped, counted in skipped_records, and
    /// reported once as a DataSkipped diagnostic. Throws ballistic::Error when
    /// the validity period (Validity_Start / Validity_Stop) is missing or
    /// malformed.
    [[nodiscard]] static std::optional<EOFFile> parse_content(
        const std::string& content,
        const std::string& filename = "")
    {
        EOFFile result;
        result.filename = filename;

        // Parse mission
        auto mission = extract_tag(content, "Mission");
        if (mission) result.mission = *mission;

        // Parse reference frame
        auto ref_frame = extract_tag(content, "Ref_Frame");
        if (ref_frame) result.reference_frame = *ref_frame;

        // Validity period: required; a missing or malformed one raises.
        auto valid_start = extract_tag(content, "Validity_Start");
        auto valid_stop = extract_tag(content, "Validity_Stop");
        const auto vs = valid_start ? parse_utc_time(*valid_start) : std::nullopt;
        const auto ve = valid_stop ? parse_utc_time(*valid_stop) : std::nullopt;
        if (!vs || !ve) {
            throw Error("EOF " + filename + ": Validity_Start/Validity_Stop missing or "
                        "malformed; not an orbit file this reader trusts");
        }
        result.validity_start = *vs;
        result.validity_stop = *ve;

        // Parse state vectors
        std::regex osv_regex("<OSV>([\\s\\S]*?)</OSV>");
        std::sregex_iterator it(content.begin(), content.end(), osv_regex);
        std::sregex_iterator end;

        while (it != end) {
            std::string osv_content = (*it)[1].str();
            auto sv = parse_osv(osv_content);
            if (sv) {
                result.state_vectors.push_back(*sv);
            } else {
                ++result.skipped_records;
            }
            ++it;
        }
        if (result.skipped_records > 0) {
            diagnostics::report(diagnostics::Category::DataSkipped,
                "EOF " + filename + ": " + std::to_string(result.skipped_records) +
                " state vector(s) did not parse and were skipped",
                "eof-skipped:" + filename);
        }

        if (result.state_vectors.empty()) {
            return std::nullopt;
        }

        return result;
    }

private:
    // Text between the first <tag ...> and the following </tag> (attributes
    // allowed), or nullopt.
    [[nodiscard]] static std::optional<std::string> extract_tag(
        const std::string& content, const std::string& tag)
    {
        // Look for opening tag (may have attributes)
        std::string open_tag_start = "<" + tag;
        std::string close_tag = "</" + tag + ">";

        size_t start = content.find(open_tag_start);
        if (start == std::string::npos) return std::nullopt;

        // Find the end of the opening tag (the >)
        size_t tag_end = content.find('>', start);
        if (tag_end == std::string::npos) return std::nullopt;

        // Make sure this is actually the tag we want (not a prefix match)
        // e.g., we don't want <VX> when looking for <V>
        size_t after_tag_name = start + open_tag_start.length();
        if (after_tag_name < content.size()) {
            char next_char = content[after_tag_name];
            // Valid tag terminations: space (for attributes), > (direct close), / (self-closing)
            if (next_char != ' ' && next_char != '>' && next_char != '/') {
                return std::nullopt;
            }
        }

        start = tag_end + 1;

        size_t end = content.find(close_tag, start);
        if (end == std::string::npos) return std::nullopt;

        return content.substr(start, end - start);
    }

    // Parse UTC time string (format: "UTC=2016-07-01T22:59:43.000000").
    // Returns nullopt on a malformed timestamp, so a bad time is never
    // replaced by a valid-looking default epoch.
    [[nodiscard]] static std::optional<Epoch> parse_utc_time(const std::string& str) {
        // Remove "UTC=" prefix if present
        std::string time_str = str;
        if (time_str.starts_with("UTC=")) {
            time_str = time_str.substr(4);
        }

        // Parse: YYYY-MM-DDTHH:MM:SS.ffffff
        int year, month, day, hour, minute;
        double second;

        if (std::sscanf(time_str.c_str(), "%d-%d-%dT%d:%d:%lf",
                        &year, &month, &day, &hour, &minute, &second) != 6) {
            return std::nullopt;
        }

        return Epoch::from_utc(year, month, day, hour, minute, second);
    }

    // Parse single OSV (Orbit State Vector)
    [[nodiscard]] static std::optional<EOFStateVector> parse_osv(const std::string& content) {
        EOFStateVector sv;

        // Parse UTC time
        auto utc = extract_tag(content, "UTC");
        if (!utc) return std::nullopt;
        auto epoch = parse_utc_time(*utc);
        if (!epoch) return std::nullopt;   // malformed timestamp — skip this OSV
        sv.epoch = *epoch;

        // Parse position (m)
        auto x = extract_tag(content, "X");
        auto y = extract_tag(content, "Y");
        auto z = extract_tag(content, "Z");
        if (!x || !y || !z) return std::nullopt;
        auto px = parse_double(*x), py = parse_double(*y), pz = parse_double(*z);
        if (!px || !py || !pz) return std::nullopt;  // garbage truth pos — skip
        sv.position.x() = *px;
        sv.position.y() = *py;
        sv.position.z() = *pz;

        // Parse velocity (m/s)
        auto vx = extract_tag(content, "VX");
        auto vy = extract_tag(content, "VY");
        auto vz = extract_tag(content, "VZ");
        if (!vx || !vy || !vz) return std::nullopt;
        auto qvx = parse_double(*vx), qvy = parse_double(*vy), qvz = parse_double(*vz);
        if (!qvx || !qvy || !qvz) return std::nullopt;
        sv.velocity.x() = *qvx;
        sv.velocity.y() = *qvy;
        sv.velocity.z() = *qvz;

        // Parse orbit number (non-critical metadata; tolerate a bad value)
        auto orbit = extract_tag(content, "Absolute_Orbit");
        if (orbit) {
            try { sv.orbit_number = std::stoi(*orbit); } catch (const std::logic_error&) {}  // bad/overflowing integer: metadata only
        }

        // Parse quality
        auto quality = extract_tag(content, "Quality");
        if (quality) {
            sv.quality = *quality;
        }

        return sv;
    }

    // Parse a double. Returns nullopt on unparseable input (a default of 0.0
    // for a position would be the Earth's center); parse_osv then skips the
    // state vector.
    [[nodiscard]] static std::optional<double> parse_double(const std::string& str) {
        try {
            return std::stod(str);
        } catch (const std::logic_error&) {   // invalid_argument / out_of_range
            return std::nullopt;
        }
    }
};

/// Errors of one predicted trajectory against EOF truth, from
/// compute_propagation_result(). Along-track, cross-track and radial are the
/// components along the direction of motion, normal to the orbit plane, and
/// away from the Earth's center (defined by the truth state).
struct PropagationTestResult {
    Epoch start_epoch;
    Epoch end_epoch;
    double duration_seconds;
    int num_comparisons;

    // Position errors
    double position_rmse;        // meters
    double position_max_error;   // meters
    double position_mean_error;  // meters; currently set equal to position_rmse, not a true mean

    // Velocity errors (if available)
    double velocity_rmse;        // m/s
    double velocity_max_error;   // m/s

    // Along-track, cross-track, radial errors
    double along_track_rms;      // meters
    double cross_track_rms;      // meters
    double radial_rms;           // meters

    friend std::ostream& operator<<(std::ostream& os, const PropagationTestResult& r) {
        return os << "Propagation Test Results:\n"
                  << "  Duration: " << r.duration_seconds << " s ("
                  << r.duration_seconds / 60.0 << " min)\n"
                  << "  Comparisons: " << r.num_comparisons << "\n"
                  << "  Position RMSE: " << r.position_rmse << " m\n"
                  << "  Position Max Error: " << r.position_max_error << " m\n"
                  << "  Velocity RMSE: " << r.velocity_rmse << " m/s\n"
                  << "  Radial RMS: " << r.radial_rms << " m\n"
                  << "  Along-track RMS: " << r.along_track_rms << " m\n"
                  << "  Cross-track RMS: " << r.cross_track_rms << " m\n";
    }
};

/// Compares predicted states (epoch, ECEF position m / velocity m/s; must be
/// non-empty, in time order) with the nearest EOF truth state vector.
/// Predictions with no truth within max_time_diff seconds (default 5 s) are
/// skipped; if none is compared, the error fields stay zero
/// (num_comparisons = 0). The truth is not interpolated, so predictions
/// should fall on its 10 s grid.
[[nodiscard]] inline PropagationTestResult compute_propagation_result(
    const std::vector<std::pair<Epoch, StateVector>>& predicted,
    const EOFFile& truth,
    double max_time_diff = defaults::VALIDATION_MAX_TIME_DIFF)
{
    PropagationTestResult result{};
    result.start_epoch = predicted.front().first;
    result.end_epoch = predicted.back().first;
    result.duration_seconds = result.end_epoch - result.start_epoch;
    result.num_comparisons = 0;

    double sum_pos_sq = 0.0;
    double sum_vel_sq = 0.0;
    double sum_radial_sq = 0.0;
    double sum_along_sq = 0.0;
    double sum_cross_sq = 0.0;
    double max_pos_err = 0.0;
    double max_vel_err = 0.0;

    for (const auto& [epoch, pred_state] : predicted) {
        auto truth_sv = truth.find_nearest(epoch);
        if (!truth_sv || std::abs(truth_sv->epoch - epoch) > max_time_diff) {
            continue;
        }

        // Position error
        Vector3d pos_err = pred_state.position - truth_sv->position;
        double pos_err_mag = pos_err.norm();
        sum_pos_sq += pos_err_mag * pos_err_mag;
        max_pos_err = (std::max)(max_pos_err, pos_err_mag);

        // Velocity error
        Vector3d vel_err = pred_state.velocity - truth_sv->velocity;
        double vel_err_mag = vel_err.norm();
        sum_vel_sq += vel_err_mag * vel_err_mag;
        max_vel_err = (std::max)(max_vel_err, vel_err_mag);

        // RTN decomposition (Radial, Transverse/Along-track, Normal/Cross-track)
        Vector3d r_hat = truth_sv->position.normalized();
        Vector3d h = truth_sv->position.cross(truth_sv->velocity);
        Vector3d n_hat = h.normalized();  // Cross-track (normal)
        Vector3d t_hat = n_hat.cross(r_hat);  // Along-track (transverse)

        double radial_err = pos_err.dot(r_hat);
        double along_err = pos_err.dot(t_hat);
        double cross_err = pos_err.dot(n_hat);

        sum_radial_sq += radial_err * radial_err;
        sum_along_sq += along_err * along_err;
        sum_cross_sq += cross_err * cross_err;

        result.num_comparisons++;
    }

    if (result.num_comparisons > 0) {
        result.position_rmse = std::sqrt(sum_pos_sq / result.num_comparisons);
        result.position_max_error = max_pos_err;
        result.position_mean_error = result.position_rmse;  // Approximation
        result.velocity_rmse = std::sqrt(sum_vel_sq / result.num_comparisons);
        result.velocity_max_error = max_vel_err;
        result.radial_rms = std::sqrt(sum_radial_sq / result.num_comparisons);
        result.along_track_rms = std::sqrt(sum_along_sq / result.num_comparisons);
        result.cross_track_rms = std::sqrt(sum_cross_sq / result.num_comparisons);
    }

    return result;
}

// =============================================================================
// ZIP Extraction Utilities
// =============================================================================

namespace fs = std::filesystem;

/// Extracts `zip_path` (a file named <name>.EOF.zip) into its own directory
/// and returns the path of the extracted <name>.EOF; if that file already
/// exists, returns it without extracting. On Windows tries PowerShell's
/// Expand-Archive, then `tar`; elsewhere only `tar` (bsdtar reads zip, GNU tar
/// does not), all run through std::system. Returns nullopt when the zip is
/// missing, is not named *.zip, or extraction fails.
[[nodiscard]] inline std::optional<std::string> extract_eof_zip(const std::string& zip_path) {
    if (!fs::exists(zip_path)) {
        return std::nullopt;
    }

    // Get the directory containing the zip file
    fs::path zip_file(zip_path);
    fs::path parent_dir = zip_file.parent_path();

    // The EOF file name is the zip filename without .zip extension
    std::string zip_filename = zip_file.filename().string();
    if (!zip_filename.ends_with(".zip")) {
        return std::nullopt;
    }
    std::string eof_filename = zip_filename.substr(0, zip_filename.size() - 4);
    fs::path eof_path = parent_dir / eof_filename;

    // If already extracted, just return the path
    if (fs::exists(eof_path)) {
        return eof_path.string();
    }

#ifdef _WIN32
    // PowerShell's Expand-Archive first, then the tar shipped with Windows 10+.
    std::string cmd = "powershell -NoProfile -Command \"Expand-Archive -Path '" +
                      zip_path + "' -DestinationPath '" + parent_dir.string() + "' -Force\" 2>nul";
    int result = std::system(cmd.c_str());
    if (result == 0 && fs::exists(eof_path)) {
        return eof_path.string();
    }
    cmd = "tar -xf \"" + zip_path + "\" -C \"" + parent_dir.string() + "\" 2>nul";
#else
    // tar (bsdtar reads zip archives; GNU tar does not, and then this fails
    // and the caller gets nullopt). "nul" is a Windows device name: on POSIX
    // it would create a file of that name, so stderr goes to /dev/null.
    std::string cmd = "tar -xf '" + zip_path + "' -C '" + parent_dir.string() + "' 2>/dev/null";
#endif
    const int tar_result = std::system(cmd.c_str());
    if (tar_result == 0 && fs::exists(eof_path)) {
        return eof_path.string();
    }

    return std::nullopt;
}

/// Parses an EOF file, extracting it first when `path` ends in .zip. Returns
/// nullopt as EOFParser::parse or when extraction fails.
[[nodiscard]] inline std::optional<EOFFile> parse_eof_auto(const std::string& path) {
    fs::path file_path(path);
    std::string filename = file_path.filename().string();

    // If it's a .zip file, extract first
    if (filename.ends_with(".zip")) {
        auto extracted = extract_eof_zip(path);
        if (!extracted) {
            return std::nullopt;
        }
        return EOFParser::parse(*extracted);
    }

    // Otherwise parse directly
    return EOFParser::parse(path);
}

/// Parses the first EOF file found in `directory` whose name contains
/// `pattern` (empty = any): extracted *.EOF files first, then *.EOF.zip.
/// "First" is directory-iteration order, which is unspecified. Returns nullopt
/// when the directory does not exist or nothing matches or parses.
[[nodiscard]] inline std::optional<EOFFile> find_and_parse_eof(
    const std::string& directory,
    const std::string& pattern = "")
{
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return std::nullopt;
    }

    // First, look for already-extracted .EOF files
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string filename = entry.path().filename().string();

        // Check for .EOF extension (not .EOF.zip)
        bool has_eof = filename.find(".EOF") != std::string::npos;
        bool has_zip = filename.find(".zip") != std::string::npos;

        if (has_eof && !has_zip) {
            if (pattern.empty() || filename.find(pattern) != std::string::npos) {
                return EOFParser::parse(entry.path().string());
            }
        }
    }

    // If no extracted files found, try .EOF.zip files
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string filename = entry.path().filename().string();

        // Check for .EOF.zip extension
        if (filename.find(".EOF.zip") != std::string::npos) {
            if (pattern.empty() || filename.find(pattern) != std::string::npos) {
                return parse_eof_auto(entry.path().string());
            }
        }
    }

    return std::nullopt;
}

/// Subdirectories of `base_path` whose names look like a year-month period
/// ("YYYY-M" or "YYYY-MM"), sorted by name. Empty when base_path does not
/// exist.
[[nodiscard]] inline std::vector<std::string> get_period_directories(const std::string& base_path) {
    std::vector<std::string> periods;

    if (!fs::exists(base_path) || !fs::is_directory(base_path)) {
        return periods;
    }

    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            // Check if it looks like a period directory (YYYY-M or YYYY-MM format)
            if (name.size() >= 6 && name[4] == '-' && std::isdigit(name[0])) {
                periods.push_back(entry.path().string());
            }
        }
    }

    // Sort by name (chronological when months are zero-padded; "2025-9" sorts after "2025-10")
    std::sort(periods.begin(), periods.end());

    return periods;
}

} // namespace validation
} // namespace ballistic
