// =============================================================================
// SPICE SPK (DAF) Ephemeris Reader for DE440/DE441
// =============================================================================
//
// An ephemeris tells where the Sun, Moon and planets are at any time. JPL's
// Development Ephemerides (DE421, DE440, DE441, ...) store each body's position
// as Chebyshev polynomials (a smooth curve fit) over consecutive time intervals.
// This module reads them from SPICE SPK binary files (SPICE is NASA NAIF's
// toolkit; SPK is its ephemeris file kind), which use the Double-precision
// Array File (DAF) container format. It also serves the DE440 data embedded
// in the library (1950-2100), so no file is needed for ordinary use.
//
// Times inside SPK files are "ephemeris time" (ET): seconds of TDB
// (Barycentric Dynamical Time, the time scale of solar-system dynamics) past
// the J2000 epoch (2000 January 1, 12:00 TDB). Positions in the files are in
// kilometers on the ICRF/J2000 axes; this module returns meters.
//
// References:
// -----------
// [1] NAIF SPK Required Reading:
//     https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html
//
// [2] NAIF DAF Required Reading:
//     https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/daf.html
//
// [3] Park, R.S., et al. (2021). "The JPL Planetary and Lunar Ephemerides
//     DE440 and DE441". The Astronomical Journal, 161:105.
//
// File Structure:
// ---------------
// - File Record (1024 bytes): Header with format info
// - Comment Area (variable): Optional ASCII comments
// - Array Data: Chebyshev coefficients for each body
// - Summary Records: Segment descriptors (at forward pointer location)
//
// SPICE Body Codes:
// -----------------
//   1 = Mercury barycenter    199 = Mercury
//   2 = Venus barycenter      299 = Venus
//   3 = Earth-Moon barycenter 399 = Earth, 301 = Moon
//   4 = Mars barycenter       499 = Mars
//   5 = Jupiter barycenter    599 = Jupiter
//   6 = Saturn barycenter     699 = Saturn
//   7 = Uranus barycenter     799 = Uranus
//   8 = Neptune barycenter    899 = Neptune
//   9 = Pluto barycenter      999 = Pluto
//   10 = Sun
//   0 = Solar System Barycenter (SSB)
//
// =============================================================================

#pragma once

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../time/epoch.hpp"
#include "../math/chebyshev.hpp"
#include "ephemeris_provider.hpp"
#include "de440_coefficients.hpp"
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <atomic>
#include <mutex>

namespace ballistic {
namespace ephemeris {

/// One SPK segment: the position of one body (target) relative to another
/// (center) over a time span.
struct SPKSegment {
    double start_et = 0.0;      ///< start of coverage, ET [s]
    double end_et = 0.0;        ///< end of coverage, ET [s]
    int target_body = 0;        ///< target body ID (e.g. 301 for the Moon)
    int center_body = 0;        ///< center body ID (e.g. 3 for the Earth-Moon barycenter)
    int ref_frame = 0;          ///< reference frame ID (1 = J2000)
    int data_type = 0;          ///< SPK data type (2 = Chebyshev position; the only one read)
    int start_addr = 0;         ///< start address in the file (1-based, in doubles); 0 = embedded
    int end_addr = 0;           ///< end address in the file; 0 = embedded

    // Type 2 (Chebyshev) data
    double init_et = 0.0;       ///< start of the first record, ET [s]
    double interval_sec = 0.0;  ///< time span of each record [s]
    int record_size = 0;        ///< doubles per record
    int n_records = 0;          ///< number of records
    int n_coeffs = 0;           ///< Chebyshev coefficients per coordinate
    std::vector<double> coefficients;  ///< all records, back to back
};

// Earth/Moon mass ratios (EMRAT) for different DE versions.
// Source: JPL Planetary and Lunar Ephemerides documentation.
// The files give the Moon relative to the Earth-Moon barycenter (EMB, the
// center of mass of the pair); EMRAT converts that to a geocentric position.
namespace de_constants {
    // DE-421: Used by ESA Copernicus POD Service
    // Source: Folkner et al. (2009) "The Planetary and Lunar Ephemeris DE 421"
    constexpr double EMRAT_DE421 = 81.30056907;

    // DE-440 (published 2021)
    // Source: Park et al. (2021) "The JPL Planetary and Lunar Ephemerides DE440 and DE441"
    constexpr double EMRAT_DE440 = 81.3005690699;

    // DE-430: Intermediate version
    constexpr double EMRAT_DE430 = 81.30056907;
}

/// Reads JPL ephemerides from SPK files or the embedded DE440 data, and
/// provides Sun/Moon/Jupiter positions for the force model.
///
/// Only SPK type 2 (Chebyshev position) segments are evaluated, and only the
/// first summary record of a file is read (up to 25 segments), which covers
/// the standard DE files. If a file holds several segments for the same
/// (target, center) pair, the last one read is kept. Not thread-safe while
/// loading; queries on a loaded object are const.
class SPKEphemeris : public EphemerisProvider {
public:
    SPKEphemeris() = default;

    /// Load an SPK file, guessing the DE version (and so the Earth/Moon mass
    /// ratio) from the file name: "421", "430", "440" or "441"; anything else
    /// uses the DE440 value. The file's segments are added to (and replace
    /// same-pair entries of) whatever was loaded before. On failure the object
    /// is left unloaded (is_loaded() false), even if data was loaded before.
    /// @return false if the file cannot be opened, is not a DAF/SPK file, or is
    ///         truncated.
    bool load(const std::filesystem::path& spk_path) {
        // EMRAT from the file name
        std::string filename = spk_path.filename().string();
        if (filename.find("421") != std::string::npos) {
            emrat_ = de_constants::EMRAT_DE421;
            de_version_ = 421;
        } else if (filename.find("430") != std::string::npos) {
            emrat_ = de_constants::EMRAT_DE430;
            de_version_ = 430;
        } else if (filename.find("440") != std::string::npos ||
                   filename.find("441") != std::string::npos) {
            emrat_ = de_constants::EMRAT_DE440;
            de_version_ = 440;
        } else {
            // Unknown version: use the DE440 value
            emrat_ = de_constants::EMRAT_DE440;
            de_version_ = 0;
        }
        return load_impl(spk_path);
    }

    /// Load an SPK file with an explicit Earth/Moon mass ratio `emrat`.
    /// @return false as for load(path).
    bool load(const std::filesystem::path& spk_path, double emrat) {
        emrat_ = emrat;
        de_version_ = 0;  // Custom
        return load_impl(spk_path);
    }

    /// Load the embedded DE440 data (Sun, Earth-Moon barycenter, Moon and
    /// Jupiter, 1950-2100), discarding any file loaded before. Gives sub-meter
    /// accuracy without external files. Always returns true.
    bool load_embedded() {
        using namespace de440_embedded;

        // Clear any existing segments
        segments_.clear();

        // Load Sun relative to SSB
        load_embedded_segment(10, 0, SUN_SSB_INFO, SUN_SSB_DATA.data(), SUN_SSB_DATA.size());

        // Load Earth-Moon Barycenter relative to SSB
        load_embedded_segment(3, 0, EMB_SSB_INFO, EMB_SSB_DATA.data(), EMB_SSB_DATA.size());

        // Load Moon relative to EMB
        load_embedded_segment(301, 3, MOON_EMB_INFO, MOON_EMB_DATA.data(), MOON_EMB_DATA.size());

        // Load Jupiter relative to SSB
        load_embedded_segment(5, 0, JUPITER_SSB_INFO, JUPITER_SSB_DATA.data(), JUPITER_SSB_DATA.size());

        // Set DE440 metadata
        de_version_ = 440;
        emrat_ = EMRAT;
        loaded_ = true;
        using_embedded_ = true;

        return true;
    }

    /// Drop every segment (is_loaded() becomes false). `why` is added to the
    /// error a later query raises, so a force model still holding this object
    /// says what replaced it.
    void unload(std::string why = {}) {
        segments_.clear();
        loaded_ = false;
        using_embedded_ = false;
        file_path_.clear();
        unloaded_reason_ = std::move(why);
    }

    /// True if the embedded DE440 data is in use (rather than a file).
    [[nodiscard]] bool is_using_embedded() const noexcept { return using_embedded_; }

    /// DE version number (421, 430, 440), or 0 if unknown or given explicitly.
    [[nodiscard]] int de_version() const noexcept { return de_version_; }

    /// Earth/Moon mass ratio in use.
    [[nodiscard]] double emrat() const noexcept { return emrat_; }

    /// True once a file or the embedded data has loaded.
    [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }

    /// Sun position [m] on GCRS axes.
    ///
    /// Note: relative to the Earth-Moon barycenter (EMB), not the Earth's
    /// center. The ~4671 km Earth-EMB offset is about 3e-5 of the Sun's
    /// distance, negligible for the Sun's perturbation of an Earth orbit.
    /// @throws DataCoverageError if nothing is loaded or the epoch is outside
    ///         the loaded data (embedded DE440: 1950-2100).
    [[nodiscard]] Vector3d sun_position(const Epoch& epoch) const override {
        if (!loaded_) throw_not_loaded("sun_position");

        double et = epoch_to_et(epoch);

        // Sun relative to Earth-Moon Barycenter:
        // Sun_EMB = Sun_SSB - EMB_SSB

        Vector3d sun_ssb = get_position(et, 10, 0);   // Sun relative to SSB
        Vector3d emb_ssb = get_position(et, 3, 0);    // EMB relative to SSB

        if (sun_ssb.isZero() || emb_ssb.isZero()) {
            throw DataCoverageError(
                "SPKEphemeris::sun_position: epoch outside loaded SPK range "
                "(embedded DE440 covers 1950-2100). Load a wider-range file "
                "such as DE441 via config.jpl_ephemeris_path.");
        }

        // Sun position relative to Earth-Moon barycenter (not true geocentric)
        Vector3d sun_emb = sun_ssb - emb_ssb;

        // Convert from km to meters
        return sun_emb * 1000.0;
    }

    /// Moon position [m] from the Earth's center, on GCRS axes. Uses the
    /// Moon-relative-to-EMB segment scaled by the mass ratio, or a direct
    /// Moon-relative-to-Earth segment if the file has no EMB one.
    /// @throws DataCoverageError if nothing is loaded or the epoch is outside
    ///         the loaded data.
    [[nodiscard]] Vector3d moon_position(const Epoch& epoch) const override {
        if (!loaded_) throw_not_loaded("moon_position");

        double et = epoch_to_et(epoch);

        // DE files hold the Moon (301) relative to the EMB (3); the geocentric
        // position follows from the mass ratio (derivation below).

        Vector3d moon_emb = get_position(et, 301, 3);  // Moon relative to EMB

        if (moon_emb.isZero()) {
            // No EMB segment: try Moon relative to Earth (399) directly
            moon_emb = get_position(et, 301, 399);
            if (!moon_emb.isZero()) {
                return moon_emb * 1000.0;
            }
            throw DataCoverageError(
                "SPKEphemeris::moon_position: epoch outside loaded SPK range.");
        }

        // Moon position relative to Earth center
        // EMB = (Earth_mass * Earth_pos + Moon_mass * Moon_pos) / (Earth_mass + Moon_mass)
        // Let Moon_EMB = Moon_pos - EMB_pos
        // Earth_EMB = Earth_pos - EMB_pos = -Moon_mass / (Earth_mass + Moon_mass) * (Moon_pos - Earth_pos)
        //           = -1/(EMRAT+1) * Moon_geocentric
        // So: Moon_geocentric = Moon_EMB * (EMRAT + 1) / EMRAT = Moon_EMB * (1 + 1/EMRAT)

        Vector3d moon_geo = moon_emb * (1.0 + 1.0 / emrat_);

        // Convert from km to meters
        return moon_geo * 1000.0;
    }

    /// Position [m] of the Jupiter system barycenter on GCRS axes.
    ///
    /// Like sun_position(), relative to the EMB, not the Earth's center; the
    /// ~4671 km offset is negligible at Jupiter's distance (~600 million km).
    /// @throws DataCoverageError if nothing is loaded or the epoch is outside
    ///         the loaded data.
    [[nodiscard]] Vector3d jupiter_position(const Epoch& epoch) const override {
        if (!loaded_) throw_not_loaded("jupiter_position");

        double et = epoch_to_et(epoch);

        // Jupiter barycenter (body 5) relative to EMB:
        // Jupiter_EMB = Jupiter_SSB - EMB_SSB

        Vector3d jup_ssb = get_position(et, 5, 0);   // Jupiter relative to SSB
        Vector3d emb_ssb = get_position(et, 3, 0);   // EMB relative to SSB

        if (jup_ssb.isZero() || emb_ssb.isZero()) {
            throw DataCoverageError(
                "SPKEphemeris::jupiter_position: epoch outside loaded SPK range.");
        }

        Vector3d jup_emb = jup_ssb - emb_ssb;

        // Convert from km to meters
        return jup_emb * 1000.0;
    }

private:
    bool load_impl(const std::filesystem::path& spk_path) {
        std::ifstream file(spk_path, std::ios::binary);
        if (!file.is_open()) {
            loaded_ = false;
            return false;
        }

        // Read and validate file record
        if (!read_file_record(file)) {
            loaded_ = false;
            return false;
        }

        // Read this file's segment summaries and their data. Only these are
        // read from the file: segments already held (the embedded set, or an
        // earlier file) carry no addresses in it and are kept as they are.
        std::vector<SPKSegment> fresh;
        if (!read_summaries(file, fresh)) {
            loaded_ = false;
            return false;
        }
        if (!read_segment_data(file, fresh)) {
            loaded_ = false;
            return false;
        }
        for (auto& seg : fresh) {
            segments_[make_segment_key(seg.target_body, seg.center_body)] = std::move(seg);
        }

        loaded_ = true;
        using_embedded_ = false;  // Loaded from file, not embedded
        file_path_ = spk_path;
        return true;
    }

public:
    /// Raw position [km] of `target` relative to `center` at ephemeris time
    /// `et` [s TDB past J2000], straight from one segment (no chaining through
    /// other bodies). Returns (0, 0, 0) when there is no such segment, the time
    /// is outside it, or the segment is not type 2 — callers test isZero().
    [[nodiscard]] Vector3d get_position(double et, int target, int center) const {
        int64_t key = make_segment_key(target, center);
        auto it = segments_.find(key);

        if (it == segments_.end()) {
            return Vector3d::Zero();
        }

        const auto& seg = it->second;

        // Check time bounds
        if (et < seg.start_et || et > seg.end_et) {
            return Vector3d::Zero();
        }

        // Type 2: Chebyshev polynomials
        if (seg.data_type == 2) {
            return evaluate_type2(seg, et);
        }

        return Vector3d::Zero();
    }

private:
    bool loaded_ = false;
    bool using_embedded_ = false;
    std::filesystem::path file_path_;
    std::string unloaded_reason_;  // set by unload(); reported by queries

    [[noreturn]] void throw_not_loaded(const char* fn) const {
        throw DataCoverageError(
            std::string("SPKEphemeris::") + fn + ": no SPK ephemeris is loaded" +
            (unloaded_reason_.empty() ? std::string(".")
                                      : " (" + unloaded_reason_ + ").") +
            " Call ballistic::initialize({}) or load an SPK file via "
            "config.jpl_ephemeris_path, then build the force model.");
    }

    // DE version info
    int de_version_ = 0;
    double emrat_ = de_constants::EMRAT_DE440;  // Default to DE-440

    // File header info
    int nd_ = 0;  // Number of doubles per segment summary
    int ni_ = 0;  // Number of integers per segment summary
    int forward_ptr_ = 0;
    int backward_ptr_ = 0;

    // Segments keyed by (target << 16 | center)
    std::unordered_map<int64_t, SPKSegment> segments_;

    [[nodiscard]] static int64_t make_segment_key(int target, int center) {
        return (static_cast<int64_t>(target) << 16) | (center & 0xFFFF);
    }

    // Build a segment from the embedded DE440 tables
    void load_embedded_segment(int target, int center,
                               const de440_embedded::SegmentInfo& info,
                               const double* data, size_t data_size) {
        SPKSegment seg;
        seg.target_body = target;
        seg.center_body = center;
        seg.ref_frame = 1;  // J2000
        seg.data_type = 2;  // Chebyshev

        // Compute time bounds from metadata
        seg.init_et = info.init_et;
        seg.interval_sec = info.interval_sec;
        seg.n_coeffs = info.n_coeffs;
        seg.n_records = info.n_records;
        seg.record_size = info.record_size;

        // Time range covered
        seg.start_et = info.init_et;
        seg.end_et = info.init_et + info.n_records * info.interval_sec;

        // Copy coefficients
        seg.coefficients.assign(data, data + data_size);

        // Store segment
        int64_t key = make_segment_key(target, center);
        segments_[key] = std::move(seg);
    }

    // Epoch -> ET (TDB seconds past J2000). Goes through a double Julian date,
    // which is accurate to ~40 microseconds — far finer than the ephemeris needs.
    [[nodiscard]] static double epoch_to_et(const Epoch& epoch) {
        double tdb_jd = epoch.tdb_jd();
        return (tdb_jd - constants::J2000_JD) * constants::SECONDS_PER_DAY;
    }

    bool read_file_record(std::ifstream& file) {
        char buffer[1024];
        file.read(buffer, 1024);
        if (file.gcount() < 1024) return false;

        // The file must start with the "DAF/SPK" identification word
        if (std::strncmp(buffer, "DAF/SPK", 7) != 0) {
            return false;
        }

        // ND, NI: doubles and integers per segment summary (offset 8)
        std::memcpy(&nd_, buffer + 8, 4);
        std::memcpy(&ni_, buffer + 12, 4);

        // First/last summary record numbers (offset 76)
        std::memcpy(&forward_ptr_, buffer + 76, 4);
        std::memcpy(&backward_ptr_, buffer + 80, 4);

        return (nd_ == 2 && ni_ == 6);  // Expected for SPK
    }

    bool read_summaries(std::ifstream& file, std::vector<SPKSegment>& out) {
        // Summary records start at (forward_ptr - 1) * 1024 bytes
        // (DAF uses 1-based record numbering)
        size_t summary_offset = (static_cast<size_t>(forward_ptr_) - 1) * 1024;
        file.seekg(summary_offset);

        // Read summary record
        char buffer[1024];
        file.read(buffer, 1024);
        if (file.gcount() < 1024) return false;

        // First 8 bytes: next/prev record pointers
        double next_ptr, prev_ptr;
        std::memcpy(&next_ptr, buffer, 8);
        std::memcpy(&prev_ptr, buffer + 8, 8);

        // Number of summaries in this record (third double). Only this first
        // summary record is read; next_ptr is not followed.
        double n_sum_d;
        std::memcpy(&n_sum_d, buffer + 16, 8);
        int n_summaries = static_cast<int>(n_sum_d);
        // A summary record holds at most (1024 - 24) / 40 = 25 summaries; more
        // means a damaged file (reading on would run past the record)
        if (n_summaries < 0 || 24 + n_summaries * (nd_ * 8 + ni_ * 4) > 1024) return false;

        // Each summary is (nd_ * 8 + ni_ * 4) bytes = 2*8 + 6*4 = 40 bytes
        int summary_size = nd_ * 8 + ni_ * 4;
        const char* ptr = buffer + 24;  // Start of summaries

        for (int i = 0; i < n_summaries; ++i) {
            SPKSegment seg;

            // Read doubles (start_et, end_et)
            std::memcpy(&seg.start_et, ptr, 8);
            std::memcpy(&seg.end_et, ptr + 8, 8);

            // Read integers (target, center, frame, type, start_addr, end_addr)
            std::memcpy(&seg.target_body, ptr + 16, 4);
            std::memcpy(&seg.center_body, ptr + 20, 4);
            std::memcpy(&seg.ref_frame, ptr + 24, 4);
            std::memcpy(&seg.data_type, ptr + 28, 4);
            std::memcpy(&seg.start_addr, ptr + 32, 4);
            std::memcpy(&seg.end_addr, ptr + 36, 4);

            // A later segment for the same (target, center) pair replaces an
            // earlier one, as in the merged map
            std::erase_if(out, [&](const SPKSegment& o) {
                return o.target_body == seg.target_body && o.center_body == seg.center_body;
            });
            out.push_back(std::move(seg));

            ptr += summary_size;
        }

        return !out.empty();
    }

    bool read_segment_data(std::ifstream& file, std::vector<SPKSegment>& segs) {
        // Read coefficient data for each Type 2 segment of this file
        for (auto& seg : segs) {
            if (seg.data_type != 2) continue;

            // Type 2 data structure:
            // Last 4 doubles: init_et, interval_sec, record_size, n_records
            // Preceding data: Chebyshev coefficients

            // Read the last 4 values from the segment
            size_t meta_offset = (static_cast<size_t>(seg.end_addr) - 4) * 8;
            file.seekg(meta_offset);

            double meta[4];
            file.read(reinterpret_cast<char*>(meta), 32);
            if (file.gcount() < 32) return false;

            seg.init_et = meta[0];
            seg.interval_sec = meta[1];
            seg.record_size = static_cast<int>(meta[2]);
            seg.n_records = static_cast<int>(meta[3]);

            // Each record has record_size doubles
            // First 2 are mid-epoch and radius, rest are coefficients for x, y, z
            // n_coeffs = (record_size - 2) / 3
            seg.n_coeffs = (seg.record_size - 2) / 3;

            // Read all coefficient data
            size_t data_offset = (static_cast<size_t>(seg.start_addr) - 1) * 8;
            size_t data_size = static_cast<size_t>(seg.n_records) * seg.record_size * 8;

            file.seekg(data_offset);
            seg.coefficients.resize(seg.n_records * seg.record_size);
            file.read(reinterpret_cast<char*>(seg.coefficients.data()), data_size);

            if (static_cast<size_t>(file.gcount()) < data_size) {
                // Truncated file: refuse it (an emptied segment would make its
                // epochs look like "outside the file" instead of an error).
                return false;
            }
        }

        return true;
    }

    [[nodiscard]] Vector3d evaluate_type2(const SPKSegment& seg, double et) const {
        if (seg.coefficients.empty() || seg.n_records <= 0) {
            return Vector3d::Zero();
        }

        // Find which record contains this epoch
        double dt = et - seg.init_et;
        int record_idx = static_cast<int>(dt / seg.interval_sec);

        // Keep the index in range (et == end_et lands one past the last record)
        if (record_idx < 0) record_idx = 0;
        if (record_idx >= seg.n_records) record_idx = seg.n_records - 1;

        // Get pointer to record data
        const double* rec = seg.coefficients.data() + record_idx * seg.record_size;

        // First two values are mid-epoch and radius (half-interval)
        double mid_et = rec[0];
        double radius = rec[1];

        // Scale time to [-1, 1] across the record (the Chebyshev domain); the
        // clamp only absorbs round-off at the record edges
        double x = (et - mid_et) / radius;
        x = std::max(-1.0, std::min(1.0, x));

        // Coefficients layout: [mid, rad, x0, x1, ..., y0, y1, ..., z0, z1, ...]
        const double* coeffs = rec + 2;

        Vector3d pos;
        for (int i = 0; i < 3; ++i) {
            pos[i] = math::chebyshev_eval(coeffs + i * seg.n_coeffs, seg.n_coeffs, x);
        }

        return pos;  // Already in km
    }
};

/// Process-wide SPKEphemeris used by default by the force model
/// (see get_default_ephemeris_provider() in jpl_ephemeris.hpp).
///
/// Thread safety: load(), load_embedded() and ensure_embedded_loaded() are
/// serialized by one lock, so several threads may build force models at once
/// (the first one loads the embedded DE440). Loading a file while another
/// thread is evaluating the ephemeris is still a data race: load files before
/// propagating from several threads.
class SPKEphemerisManager {
public:
    /// The single shared instance.
    static SPKEphemerisManager& instance() {
        static SPKEphemerisManager mgr;
        return mgr;
    }

    /// Load an SPK file into the shared instance (see SPKEphemeris::load).
    bool load(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        loaded_.store(false, std::memory_order_release);
        const bool ok = ephemeris_.load(path);
        loaded_.store(ephemeris_.is_loaded(), std::memory_order_release);
        return ok;
    }

    /// Load the embedded DE440 data (1950-2100) into the shared instance.
    bool load_embedded() {
        std::lock_guard<std::mutex> lock(mutex_);
        return load_embedded_unlocked();
    }

    /// Load the embedded DE440 data unless something is already loaded, as one
    /// step (two threads cannot both find it empty and both load). Returns
    /// true if the ephemeris is loaded afterwards.
    bool ensure_embedded_loaded() {
        if (loaded_.load(std::memory_order_acquire)) return true;
        std::lock_guard<std::mutex> lock(mutex_);
        if (ephemeris_.is_loaded()) {
            loaded_.store(true, std::memory_order_release);
            return true;
        }
        return load_embedded_unlocked();
    }

    /// Drop the shared SPK data, so get_default_ephemeris_provider() falls
    /// through to the legacy DE reader. `why` is reported by force models that
    /// still hold the SPK object.
    void unload(std::string why = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        loaded_.store(false, std::memory_order_release);
        ephemeris_.unload(std::move(why));
    }

    /// True once something has loaded.
    [[nodiscard]] bool is_loaded() const noexcept {
        return loaded_.load(std::memory_order_acquire);
    }

    /// True if the embedded DE440 data is in use.
    [[nodiscard]] bool is_using_embedded() const noexcept {
        return ephemeris_.is_using_embedded();
    }

    /// The shared ephemeris object.
    [[nodiscard]] const SPKEphemeris& ephemeris() const noexcept {
        return ephemeris_;
    }

    [[nodiscard]] SPKEphemeris& ephemeris() noexcept {
        return ephemeris_;
    }

private:
    SPKEphemerisManager() = default;

    bool load_embedded_unlocked() {
        loaded_.store(false, std::memory_order_release);
        const bool ok = ephemeris_.load_embedded();
        loaded_.store(ephemeris_.is_loaded(), std::memory_order_release);
        return ok;
    }

    SPKEphemeris ephemeris_;
    std::atomic<bool> loaded_{false};   // ephemeris_ holds data (lock-free read)
    std::mutex mutex_;                  // serializes every load
};

/// Load an SPK file into the shared instance; false on failure.
inline bool load_spk_ephemeris(const std::filesystem::path& path) {
    return SPKEphemerisManager::instance().load(path);
}

/// The shared SPK ephemeris.
inline const SPKEphemeris& spk_ephemeris() {
    return SPKEphemerisManager::instance().ephemeris();
}

// Definitions of the functions declared in jpl_ephemeris.hpp, through which
// get_default_ephemeris_provider() reaches the shared SPK instance
namespace detail {
    inline bool is_spk_loaded() {
        return SPKEphemerisManager::instance().is_loaded();
    }

    inline bool load_embedded_spk() {
        return SPKEphemerisManager::instance().load_embedded();
    }

    inline bool ensure_embedded_spk_loaded() {
        return SPKEphemerisManager::instance().ensure_embedded_loaded();
    }

    inline std::shared_ptr<EphemerisProvider> get_spk_provider() {
        // Non-owning shared_ptr to the shared instance (it lives for the
        // whole program, so nothing must delete it)
        return std::shared_ptr<EphemerisProvider>(
            std::shared_ptr<void>{}, &SPKEphemerisManager::instance().ephemeris());
    }
}

} // namespace ephemeris
} // namespace ballistic
