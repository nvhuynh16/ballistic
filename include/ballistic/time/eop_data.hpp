#pragma once

// =============================================================================
// IERS Earth Orientation Parameters (EOP) Data Loader
// =============================================================================
//
// Loads and interpolates Earth Orientation Parameters (EOP) from the IERS
// (International Earth Rotation and Reference Systems Service) file
// finals2000A.data / finals2000A.all.
//
// Why they matter: the Earth's rotation is not perfectly regular, so turning
// a position in the Earth-fixed frame (ITRF, the International Terrestrial
// Reference Frame, rotating with the Earth) into the inertial frame (GCRS,
// the Geocentric Celestial Reference System, non-rotating) needs measured
// values of the rotation angle (through UT1 - UTC) and of the wobble of the
// rotation axis (polar motion). Without them the transform is off by
// 3-10 m at LEO (low Earth orbit).
//
// References:
// -----------
// [1] IERS Conventions (2010), IERS Technical Note No. 36
//     - Chapter 5: Transformation between ITRS and GCRS
//     - Section 5.5: Earth Orientation Parameters
//
// [2] IERS Earth Orientation Centre
//     - Data format: https://datacenter.iers.org/eop.php
//     - finals2000A.data format specification
//
// EOP Parameters:
// ---------------
// - x, y:     Polar motion: where the rotation axis meets the Earth's crust (arcseconds)
// - UT1-UTC:  Earth rotation angle expressed as a time offset from UTC (seconds)
// - dX, dY:   Celestial pole offsets: measured corrections to the IAU 2006/2000A
//             precession-nutation model of the axis direction in space (milliarcseconds)
// - LOD:      Length of day minus 86400 s (milliseconds)
//
// The finals2000A.data file format (columns):
//   Col 1-2:    Year (2 digits; not read: the MJD column dates the row)
//   Col 3-4:    Month
//   Col 5-6:    Day
//   Col 7:      Space
//   Col 8-15:   MJD
//   Col 16:     Space
//   Col 17:     Polar motion (I/P flag for x)
//   Col 18-27:  x (arcseconds)
//   Col 28-36:  x error (arcseconds)
//   Col 37:     Space
//   Col 38:     Polar motion (I/P flag for y)
//   Col 39-48:  y (arcseconds)
//   Col 49-57:  y error (arcseconds)
//   Col 58:     Space
//   Col 59:     UT1-UTC (I/P flag)
//   Col 60-68:  UT1-UTC (seconds)
//   Col 69-78:  UT1-UTC error (seconds)
//   ... additional columns for LOD, dX, dY
//
// =============================================================================

#include "../core/constants.hpp"
#include "../core/diagnostics.hpp"
#include "../core/errors.hpp"
#include "leap_seconds.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <optional>
#include <mutex>
#include <atomic>
#include <cstdint>

namespace ballistic {
namespace time {

/// One day's row of the EOP file.
struct EOPRecord {
    double mjd;         ///< Modified Julian date (MJD = JD - 2400000.5) of the row, UTC midnight
    double x;           ///< Polar motion x (arcseconds)
    double y;           ///< Polar motion y (arcseconds)
    double ut1_utc;     ///< UT1-UTC (seconds)
    double lod;         ///< Length of day minus 86400 s (milliseconds; 0 when blank)
    double dX;          ///< Celestial pole offset dX (milliarcseconds; 0 when blank)
    double dY;          ///< Celestial pole offset dY (milliarcseconds; 0 when blank)

    // Quality flags
    bool x_predicted;   ///< True if x is an IERS prediction, not a measurement
    bool y_predicted;   ///< True if y is predicted
    bool ut1_predicted; ///< True if UT1-UTC is predicted
};

/// Where a lookup's values came from.
enum class EopCoverage {
    InTable,  ///< interpolated inside the table
    Held,     ///< up to kEopHoldDays beyond the table: its edge value, held
    NoTable,  ///< nothing loaded (only under EopPolicy::AssumeZero): zeros
};

/// The Earth orientation values at one instant, as returned by EOPData::lookup (the one
/// lookup every part of the library uses).
struct EopSample {
    double xp_arcsec = 0.0;    ///< polar motion x [arcsec]
    double yp_arcsec = 0.0;    ///< polar motion y [arcsec]
    double ut1_utc_s = 0.0;    ///< UT1 - UTC [s]
    double lod_ms = 0.0;       ///< length of day minus 86400 s [ms]
    double dX_mas = 0.0;       ///< celestial pole offset dX [milliarcsec]
    double dY_mas = 0.0;       ///< celestial pole offset dY [milliarcsec]
    /// Rates of the interpolated values [per day]: what the transforms' positions follow,
    /// so a converted velocity can be the derivative of the converted positions. 0 when
    /// held, at the first row, or with no table. ut1_utc_rate is about -LOD (per day).
    double xp_rate_arcsec_per_day = 0.0;
    double yp_rate_arcsec_per_day = 0.0;
    double ut1_utc_rate_s_per_day = 0.0;
    EopCoverage coverage = EopCoverage::NoTable;
    bool predicted = false;    ///< UT1-UTC from IERS predictions, not measurements
    double days_beyond = 0.0;  ///< > 0 when Held: days past the table's edge
};

/// How far past either end of the table [days] a lookup holds the edge value (with a
/// warning) before it throws DataCoverageError. UT1-UTC drifts by ~1 ms per few days, so a
/// held value costs meters at LEO within weeks; IERS predictions already extend ~1 year,
/// so a fresh file rarely needs the hold.
inline constexpr double kEopHoldDays = 30.0;

/// What a lookup does when NO table is loaded.
enum class EopPolicy {
    Require,     ///< default: throw DataCoverageError naming how to load one
    AssumeZero,  ///< explicit opt-in: UT1-UTC, polar motion, dX/dY = 0 (3-10 m
                 ///< at LEO), reported once through the diagnostics channel
};

/// The process-wide EOP table (a singleton: one shared instance, EOPData::instance()).
///
/// Thread safety: each query takes a lock, but the table is shared global state. Load it
/// before starting threads; concurrent read-only queries are then fine; never reload while
/// other threads read. ballistic::initialize() loads it; nothing in the library reloads it
/// on its own.
class EOPData {
public:
    /// The shared instance.
    static EOPData& instance() {
        static EOPData instance;
        return instance;
    }

    // Not copyable or movable (there is one shared instance).
    EOPData(const EOPData&) = delete;
    EOPData& operator=(const EOPData&) = delete;
    EOPData(EOPData&&) = delete;
    EOPData& operator=(EOPData&&) = delete;

    /// Load (replace) the table from an IERS finals2000A.data / .all file at `file_path`.
    /// Returns false if the file cannot be opened or holds no usable row; the previous table
    /// is then kept. Rows past the end of the IERS predictions (blank x, y or UT1-UTC) and
    /// malformed rows are skipped and counted (blank_rows_skipped, malformed_rows_skipped);
    /// blank dX/dY/LOD are taken as 0; lines shorter than 68 characters are ignored
    /// (not counted). Does not throw for file content.
    bool load(const std::filesystem::path& file_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ifstream file(file_path);
        if (!file.is_open()) {
            return false;
        }

        std::vector<EOPRecord> new_data;
        std::string line;
        std::size_t blank_rows = 0, malformed_rows = 0, blank_cip = 0;

        while (std::getline(file, line)) {
            if (line.length() < 68) {
                continue;  // Skip incomplete lines
            }

            EOPRecord record;

            try {
                // Parse MJD (columns 8-15, 1-indexed -> 7-14 in 0-indexed)
                record.mjd = std::stod(line.substr(7, 8));

                // Parse polar motion x (columns 18-27 -> 17-26)
                record.x_predicted = (line[16] == 'P');
                if (line.length() >= 27) {
                    std::string x_str = line.substr(17, 10);
                    if (!x_str.empty() && x_str.find_first_not_of(" ") != std::string::npos) {
                        record.x = std::stod(x_str);
                    } else {
                        ++blank_rows;   // past the end of the predictions
                        continue;
                    }
                }

                // Parse polar motion y (columns 39-48 -> 38-47)
                if (line.length() >= 48) {
                    record.y_predicted = (line[37] == 'P');
                    std::string y_str = line.substr(38, 10);
                    if (!y_str.empty() && y_str.find_first_not_of(" ") != std::string::npos) {
                        record.y = std::stod(y_str);
                    } else {
                        ++blank_rows;
                        continue;
                    }
                }

                // Parse UT1-UTC (columns 59-68 -> 58-67)
                if (line.length() >= 68) {
                    record.ut1_predicted = (line[57] == 'P');
                    std::string ut1_str = line.substr(58, 10);
                    if (!ut1_str.empty() && ut1_str.find_first_not_of(" ") != std::string::npos) {
                        record.ut1_utc = std::stod(ut1_str);
                    } else {
                        ++blank_rows;
                        continue;
                    }
                }

                // Parse LOD (columns 80-86 -> 79-85) if available
                if (line.length() >= 86) {
                    std::string lod_str = line.substr(79, 7);
                    if (!lod_str.empty() && lod_str.find_first_not_of(" ") != std::string::npos) {
                        record.lod = std::stod(lod_str);
                    } else {
                        record.lod = 0.0;
                    }
                } else {
                    record.lod = 0.0;
                }

                // Parse dX, dY (columns 98-106 and 117-125) if available
                // These are celestial pole offsets in milliarcseconds
                if (line.length() >= 106) {
                    std::string dX_str = line.substr(97, 9);
                    if (!dX_str.empty() && dX_str.find_first_not_of(" ") != std::string::npos) {
                        record.dX = std::stod(dX_str);
                    } else {
                        record.dX = 0.0;   // CIP offsets: ~0.1-0.3 mas, cm level
                        ++blank_cip;
                    }
                } else {
                    record.dX = 0.0;
                }

                if (line.length() >= 125) {
                    std::string dY_str = line.substr(116, 9);
                    if (!dY_str.empty() && dY_str.find_first_not_of(" ") != std::string::npos) {
                        record.dY = std::stod(dY_str);
                    } else {
                        record.dY = 0.0;
                    }
                } else {
                    record.dY = 0.0;
                }

                new_data.push_back(record);
            }
            catch (const std::exception&) {
                ++malformed_rows;   // counted, reported in load_note()
                continue;
            }
        }

        if (new_data.empty()) {
            return false;
        }

        // Sort by MJD
        std::sort(new_data.begin(), new_data.end(),
            [](const EOPRecord& a, const EOPRecord& b) {
                return a.mjd < b.mjd;
            });

        data_ = std::move(new_data);
        loaded_ = true;
        ++generation_;
        blank_rows_ = blank_rows;
        malformed_rows_ = malformed_rows;
        blank_cip_ = blank_cip;
        return true;
    }

    /// Counts from the last successful load: rows skipped because x, y or UT1-UTC was blank
    /// (the file's rows past the end of the IERS predictions; loading them as zeros would
    /// count them as "in range"), rows skipped as malformed, and rows KEPT whose blank dX
    /// was taken as 0.
    [[nodiscard]] std::size_t blank_rows_skipped() const noexcept { return blank_rows_; }
    [[nodiscard]] std::size_t malformed_rows_skipped() const noexcept { return malformed_rows_; }
    [[nodiscard]] std::size_t blank_cip_rows() const noexcept { return blank_cip_; }

    /// The MJD of the last row whose UT1-UTC is measured (not an IERS prediction);
    /// nullopt if none.
    [[nodiscard]] std::optional<double> measured_until_mjd() const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = data_.rbegin(); it != data_.rend(); ++it)
            if (!it->ut1_predicted) return it->mjd;
        return std::nullopt;
    }

    /// The Earth orientation values at UTC modified Julian date `utc_mjd` [days]. Every part
    /// of the library goes through this (frame transforms, Epoch's UT1, tides, Trajectory).
    ///   - Inside the table: linear interpolation between the daily rows. UT1-UTC is
    ///     interpolated as UT1-TAI and TAI-UTC added back at the query, so the 1 s jump at
    ///     a leap second is not smeared across the day.
    ///   - Up to kEopHoldDays beyond either end: the edge row's values, held, with a
    ///     DataCoverage diagnostic (once per process).
    ///   - Further: throws DataCoverageError (a Trajectory stops and keeps its state).
    ///   - No table loaded: throws DataCoverageError under EopPolicy::Require (the
    ///     default); under EopPolicy::AssumeZero returns zeros with a DataCoverage
    ///     diagnostic once (3-10 m at LEO; the transforms cannot be right without EOP).
    /// Also throws DataCoverageError for dates before 1961 (no UTC, via the leap-second table).
    [[nodiscard]] EopSample lookup(double utc_mjd) const {
        EopSample s;
        enum class Note { None, NoTable, Held } note = Note::None;
        double edge_mjd = 0.0;
        // UT1-UTC between two rows, finished after the lock is released: the
        // leap-second lookup it needs can report a diagnostic, and a handler
        // that reads EOP data would otherwise deadlock on this mutex.
        bool ut1_pending = false;
        double ut1_a = 0.0, ut1_b = 0.0, mjd_a = 0.0, mjd_b = 0.0, frac = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (data_.empty()) {
                if (policy_ == EopPolicy::Require) {
                    throw DataCoverageError(
                        "no Earth orientation data loaded: the Earth-fixed <-> inertial "
                        "transforms need UT1-UTC and polar motion (without them, 3-10 m at "
                        "LEO). Load finals2000A (Python: ballistic.load(); C++: "
                        "ballistic::initialize with the file in the data directory or "
                        "InitConfig::eop_path), or accept zeros explicitly with "
                        "EOPData::set_policy(EopPolicy::AssumeZero).");
                }
                note = Note::NoTable;
            } else {
                const EOPRecord& first = data_.front();
                const EOPRecord& last = data_.back();
                const double before = first.mjd - utc_mjd;
                const double after = utc_mjd - last.mjd;
                if (before > 0.0 || after > 0.0) {
                    const double beyond = before > 0.0 ? before : after;
                    const EOPRecord& edge = before > 0.0 ? first : last;
                    if (beyond > kEopHoldDays) {
                        throw DataCoverageError(
                            "Earth orientation data needed for MJD " + std::to_string(utc_mjd) +
                            ", " + std::to_string(static_cast<long>(beyond)) + " days " +
                            (before > 0.0 ? "before" : "after") + " the loaded table (MJD " +
                            std::to_string(static_cast<long>(first.mjd)) + " to " +
                            std::to_string(static_cast<long>(last.mjd)) + "); beyond " +
                            std::to_string(static_cast<int>(kEopHoldDays)) + " days a held "
                            "value is too far off (UT1-UTC drifts ~1 ms per few days). Load a "
                            "finals2000A file that covers the dates (Python: ballistic.load()).");
                    }
                    fill(s, edge);
                    s.coverage = EopCoverage::Held;
                    s.days_beyond = beyond;
                    note = Note::Held;
                    edge_mjd = edge.mjd;
                } else {
                    auto it = std::lower_bound(data_.begin(), data_.end(), utc_mjd,
                        [](const EOPRecord& r, double m) { return r.mjd < m; });
                    if (it == data_.begin()) {
                        fill(s, *it);
                    } else {
                        const EOPRecord& b = *it;
                        const EOPRecord& a = *std::prev(it);
                        const double f = (utc_mjd - a.mjd) / (b.mjd - a.mjd);
                        auto lerp = [f](double va, double vb) { return (1.0 - f) * va + f * vb; };
                        s.xp_arcsec = lerp(a.x, b.x);
                        s.yp_arcsec = lerp(a.y, b.y);
                        s.lod_ms = lerp(a.lod, b.lod);
                        s.dX_mas = lerp(a.dX, b.dX);
                        s.dY_mas = lerp(a.dY, b.dY);
                        s.xp_rate_arcsec_per_day = (b.x - a.x) / (b.mjd - a.mjd);
                        s.yp_rate_arcsec_per_day = (b.y - a.y) / (b.mjd - a.mjd);
                        s.predicted = a.ut1_predicted || b.ut1_predicted;
                        ut1_pending = true;
                        ut1_a = a.ut1_utc;
                        ut1_b = b.ut1_utc;
                        mjd_a = a.mjd;
                        mjd_b = b.mjd;
                        frac = f;
                    }
                    s.coverage = EopCoverage::InTable;
                }
            }
        }
        // Outside the lock: the diagnostics handler may call back in.
        if (ut1_pending) {
            // UT1-TAI is smooth; UT1-UTC steps by 1 s at a leap second.
            auto& leaps = LeapSecondTable::instance();
            const double ja = mjd_a + constants::MJD_OFFSET;
            const double jb = mjd_b + constants::MJD_OFFSET;
            const double jq = utc_mjd + constants::MJD_OFFSET;
            const double ua = ut1_a - leaps.tai_utc(ja), ub = ut1_b - leaps.tai_utc(jb);
            const double u = (1.0 - frac) * ua + frac * ub;
            s.ut1_utc_s = u + leaps.tai_utc(jq);
            s.ut1_utc_rate_s_per_day = (ub - ua) / (mjd_b - mjd_a);
        }
        if (note == Note::NoTable) {
            diagnostics::report(diagnostics::Category::DataCoverage,
                "no Earth orientation data loaded: UT1-UTC and polar motion are taken as "
                "zero, which costs 3-10 m at LEO. Load finals2000A (Python: "
                "ballistic.load(); C++: InitConfig::eop_path or the data directory)",
                "eop-not-loaded");
        } else if (note == Note::Held) {
            diagnostics::report(diagnostics::Category::DataCoverage,
                "Earth orientation data needed " + std::to_string(s.days_beyond) +
                " days beyond the loaded table (edge MJD " +
                std::to_string(static_cast<long>(edge_mjd)) + "): its edge value is held "
                "(UT1-UTC drifts ~1 ms per few days: meters at LEO within weeks); beyond " +
                std::to_string(static_cast<int>(kEopHoldDays)) + " days it stops. Load a "
                "newer finals2000A file",
                "eop-outside-table");
        }
        return s;
    }

    /// True once a table has been loaded (and not cleared).
    [[nodiscard]] bool is_loaded() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_;
    }

    /// Number of daily rows in the table.
    [[nodiscard]] size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }

    /// First and last MJD in the table ({0, 0} when empty).
    [[nodiscard]] std::pair<double, double> mjd_range() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) {
            return {0.0, 0.0};
        }
        return {data_.front().mjd, data_.back().mjd};
    }

    /// True if `mjd` lies between the first and last row (inclusive); false when empty.
    [[nodiscard]] bool in_range(double mjd) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) return false;
        return mjd >= data_.front().mjd && mjd <= data_.back().mjd;
    }

    /// What lookups do when no table is loaded (see EopPolicy). Process-wide; set it
    /// explicitly only when running without EOP on purpose (synthetic tests, quick looks).
    void set_policy(EopPolicy p) {
        std::lock_guard<std::mutex> lock(mutex_);
        policy_ = p;
        ++generation_;
    }
    [[nodiscard]] EopPolicy policy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return policy_;
    }

    /// Single values at UTC MJD `mjd`, all through lookup() (same policy and errors):
    /// UT1-UTC [s], polar motion x / y [arcsec], dX / dY [milliarcsec], LOD [ms].
    [[nodiscard]] double ut1_utc(double mjd) const { return lookup(mjd).ut1_utc_s; }
    [[nodiscard]] double polar_x(double mjd) const { return lookup(mjd).xp_arcsec; }
    [[nodiscard]] double polar_y(double mjd) const { return lookup(mjd).yp_arcsec; }
    [[nodiscard]] double dX(double mjd) const { return lookup(mjd).dX_mas; }
    [[nodiscard]] double dY(double mjd) const { return lookup(mjd).dY_mas; }
    [[nodiscard]] double lod(double mjd) const { return lookup(mjd).lod_ms; }

    /// Polar motion (x, y) [rad] at UTC MJD `mjd`, through lookup() (same errors).
    [[nodiscard]] std::pair<double, double> polar_motion_rad(double mjd) const {
        double x_arcsec = polar_x(mjd);
        double y_arcsec = polar_y(mjd);
        return {
            x_arcsec * constants::ARCSEC_TO_RAD,
            y_arcsec * constants::ARCSEC_TO_RAD
        };
    }

    /// Celestial pole offsets (dX, dY) [rad] at UTC MJD `mjd`, through lookup() (same
    /// errors). The CIP (Celestial Intermediate Pole) is the rotation axis used by the
    /// IAU 2006/2000A transforms; dX, dY correct its modeled direction.
    [[nodiscard]] std::pair<double, double> cip_offsets_rad(double mjd) const {
        double dX_mas = dX(mjd);
        double dY_mas = dY(mjd);
        constexpr double MAS_TO_RAD = constants::ARCSEC_TO_RAD / 1000.0;
        return {
            dX_mas * MAS_TO_RAD,
            dY_mas * MAS_TO_RAD
        };
    }

    /// Remove the loaded table (lookups then follow the EopPolicy).
    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
        loaded_ = false;
        ++generation_;
    }

    /// A counter that changes whenever what lookup() returns can change (a load, clear()
    /// or set_policy()), so a cache of looked-up values knows when it is stale.
    [[nodiscard]] std::uint64_t generation() const noexcept {
        return generation_.load(std::memory_order_acquire);
    }

    /// The raw row nearest to `mjd` (the first/last row outside the table), for inspection;
    /// no interpolation and no hold limit. nullopt when no table is loaded.
    [[nodiscard]] std::optional<EOPRecord> get_record(double mjd) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) return std::nullopt;

        auto it = std::lower_bound(data_.begin(), data_.end(), mjd,
            [](const EOPRecord& r, double m) { return r.mjd < m; });

        if (it == data_.end()) {
            return data_.back();
        }
        if (it == data_.begin()) {
            return data_.front();
        }

        // Return nearest
        auto prev = std::prev(it);
        if (std::abs(it->mjd - mjd) < std::abs(prev->mjd - mjd)) {
            return *it;
        }
        return *prev;
    }

private:
    EOPData() = default;

    mutable std::mutex mutex_;
    std::vector<EOPRecord> data_;
    bool loaded_ = false;

    EopPolicy policy_ = EopPolicy::Require;
    std::atomic<std::uint64_t> generation_{0};
    std::size_t blank_rows_ = 0;
    std::size_t malformed_rows_ = 0;
    std::size_t blank_cip_ = 0;

    static void fill(EopSample& s, const EOPRecord& r) {
        s.xp_arcsec = r.x;
        s.yp_arcsec = r.y;
        s.ut1_utc_s = r.ut1_utc;
        s.lod_ms = r.lod;
        s.dX_mas = r.dX;
        s.dY_mas = r.dY;
        s.predicted = r.ut1_predicted;
    }
};

/// UT1-UTC [s] at UTC MJD `mjd` from the shared table; same policy and errors as
/// EOPData::lookup (with no table loaded it throws DataCoverageError unless
/// EopPolicy::AssumeZero is set, then it returns 0).
[[nodiscard]] inline double get_ut1_utc(double mjd) {
    return EOPData::instance().ut1_utc(mjd);
}

/// Polar motion (x, y) [rad] at UTC MJD `mjd` from the shared table; same errors as
/// EOPData::lookup.
[[nodiscard]] inline std::pair<double, double> get_polar_motion(double mjd) {
    return EOPData::instance().polar_motion_rad(mjd);
}

} // namespace time

// Bring into ballistic namespace
using time::EOPData;
using time::EOPRecord;
using time::EopSample;
using time::EopCoverage;
using time::EopPolicy;

} // namespace ballistic
