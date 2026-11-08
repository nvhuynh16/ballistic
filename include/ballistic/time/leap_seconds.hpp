#pragma once

// =============================================================================
// Leap seconds: TAI - UTC
// =============================================================================
// TAI (International Atomic Time) runs continuously; UTC (Coordinated
// Universal Time, civil time) is held within 0.9 s of the Earth's rotation by
// inserting leap seconds, so TAI - UTC grows by 1 s at each one (37 s since
// 2017). This header is separate from epoch.hpp so eop_data.hpp can use it
// too (UT1-UTC is interpolated as UT1-TAI so that a leap-second day does not
// smear a 1 s step across it).
// =============================================================================

#include "../core/constants.hpp"
#include "../core/diagnostics.hpp"
#include "../core/errors.hpp"
#include "julian_date.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace ballistic {
namespace time {

/// One row of the leap-second table.
struct LeapSecondEntry {
    double jd;      ///< UTC Julian date from which this TAI - UTC applies (a midnight)
    int tai_utc;    ///< TAI - UTC [s] from that date on (cumulative leap seconds)
};

// =============================================================================
// LeapSecondTable: TAI - UTC for any UTC date since 1961
// =============================================================================
//
// One process-wide table (a singleton). It starts with the embedded leap
// seconds (through 2017, valid until the IERS expiry date below) and can be
// replaced from a file. Lookups take a lock, so concurrent reads are safe.
//
// Supports loading from IERS leap-seconds.list format:
//   # Comments start with #
//   # File expires on: 28 December 2024
//   2272060800  10  # 1 Jan 1972
//   2287785600  11  # 1 Jul 1972
//   ...
//
// The first column is an NTP timestamp (seconds since 1900-01-01 00:00:00,
// the Network Time Protocol's epoch); the second is TAI-UTC from that date.
// A "#@" line gives the file's expiry date as an NTP timestamp.
//
// Download from:
//   https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
//   https://www.ietf.org/timezones/data/leap-seconds.list
//
// =============================================================================
class LeapSecondTable {
public:
    /// The shared instance.
    static LeapSecondTable& instance() {
        static LeapSecondTable table;
        return table;
    }

    LeapSecondTable(const LeapSecondTable&) = delete;
    LeapSecondTable& operator=(const LeapSecondTable&) = delete;

    /// Replace the table with the contents of an IERS leap-seconds.list file at `file_path`.
    /// Returns false (table unchanged; reason in last_error()) if the file cannot be opened,
    /// has no valid row, or has a line that is neither a comment ('#') nor
    /// "<NTP seconds> <TAI-UTC>" (a damaged file is refused rather than read in part: a
    /// skipped row would put a 1 s error on every date after it). Without a "#@" expiry line, the embedded expiry is used when the file's newest
    /// leap second is the embedded newest, otherwise six months past its newest entry.
    bool load(const std::filesystem::path& file_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ifstream file(file_path);
        if (!file.is_open()) {
            last_error_ = "Could not open file: " + file_path.string();
            return false;
        }

        std::vector<LeapSecondEntry> new_entries;
        std::string line;
        double expiration_ntp = 0.0;
        std::size_t line_no = 0;
        auto refuse = [&](const std::string& why) {
            last_error_ = file_path.string() + " line " + std::to_string(line_no) + ": " + why +
                          " (\"" + line + "\"); the file was not loaded. Replace it with a "
                          "current leap-seconds.list (Python: ballistic.load()).";
            return false;
        };

        while (std::getline(file, line)) {
            ++line_no;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // Skip blank lines
            if (line.find_first_not_of(" \t") == std::string::npos) continue;

            // Expiry line: "#@  NTP_timestamp"
            if (line.find("#@") == 0) {
                std::istringstream iss(line.substr(2));
                if (!(iss >> expiration_ntp) || expiration_ntp <= 0.0) {
                    return refuse("the expiry line has no NTP timestamp");
                }
                continue;
            }

            // Skip comment lines
            if (line[0] == '#') continue;

            // Data line: NTP_timestamp TAI-UTC [# comment]
            std::istringstream iss(line);
            double ntp_timestamp;
            int tai_utc;
            std::string rest;
            if (!(iss >> ntp_timestamp >> tai_utc) || ((iss >> rest) && rest[0] != '#')) {
                return refuse("not \"<NTP seconds> <TAI-UTC seconds>\"");
            }

            // Convert NTP timestamp to Julian Date
            // NTP epoch is 1900-01-01 00:00:00 UTC
            // JD of NTP epoch = 2415020.5
            constexpr double NTP_EPOCH_JD = 2415020.5;
            double jd = NTP_EPOCH_JD + ntp_timestamp / constants::SECONDS_PER_DAY;

            new_entries.push_back({jd, tai_utc});
        }

        if (new_entries.empty()) {
            last_error_ = "No valid leap second entries found in file";
            return false;
        }

        // Sort by JD
        std::sort(new_entries.begin(), new_entries.end(),
            [](const LeapSecondEntry& a, const LeapSecondEntry& b) {
                return a.jd < b.jd;
            });

        entries_ = std::move(new_entries);
        loaded_from_file_ = true;

        // Store expiration date if found
        if (expiration_ntp > 0) {
            constexpr double NTP_EPOCH_JD = 2415020.5;
            expiration_jd_ = NTP_EPOCH_JD + expiration_ntp / constants::SECONDS_PER_DAY;
        } else {
            // No '#@' expiry line. If the file's newest leap second is the
            // embedded table's newest, the embedded (IERS-sourced) expiry
            // applies; otherwise assume six months past its newest entry.
            // (Newest + 6 months alone would put the expiry in 2017 for every
            // current file and warn on every date since.)
            const double embedded_last = calendar_to_jd(2017, 1, 1);
            expiration_jd_ = std::abs(entries_.back().jd - embedded_last) < 0.5
                                 ? calendar_to_jd(2027, 6, 28)
                                 : entries_.back().jd + 182.5;
        }

        last_error_.clear();
        return true;
    }

    /// TAI - UTC [s] at UTC Julian date `utc_jd` [days].
    ///   - 1972 onwards: the leap-second table (integer seconds).
    ///   - 1961-01-01 to 1972-01-01: the US Naval Observatory (USNO) formulas of the
    ///     "rubber second" era, when UTC seconds were stretched instead of stepped
    ///     (tai-utc.dat; the table the IAU SOFA library's iauDat uses), e.g. 6.1856820 s
    ///     on 1968-02-01 (not a flat 10 s).
    ///   - Before 1961: UTC did not exist; throws DataCoverageError.
    /// Past the table's expiry date (no leap seconds announced beyond it) the last value is
    /// held and a DataCoverage diagnostic is reported once per process.
    [[nodiscard]] double tai_utc(double utc_jd) const {
        constexpr double JD_1961 = 2437300.5;
        constexpr double JD_1972 = 2441317.5;
        if (utc_jd < JD_1961) {
            throw DataCoverageError(
                "TAI-UTC is undefined before 1961-01-01: UTC did not exist. Use a TT or TAI "
                "epoch (Epoch.from_tt / from_tai) for earlier dates.");
        }
        if (utc_jd < JD_1972) {
            // {start JD, A [s], MJD0, rate [s/day]}: TAI-UTC = A + (MJD - MJD0) * rate
            static constexpr double kRubber[][4] = {
                {2437300.5, 1.4228180, 37300.0, 0.001296},
                {2437512.5, 1.3728180, 37300.0, 0.001296},
                {2437665.5, 1.8458580, 37665.0, 0.0011232},
                {2438334.5, 1.9458580, 37665.0, 0.0011232},
                {2438395.5, 3.2401300, 38761.0, 0.001296},
                {2438486.5, 3.3401300, 38761.0, 0.001296},
                {2438639.5, 3.4401300, 38761.0, 0.001296},
                {2438761.5, 3.5401300, 38761.0, 0.001296},
                {2438820.5, 3.6401300, 38761.0, 0.001296},
                {2438942.5, 3.7401300, 38761.0, 0.001296},
                {2439004.5, 3.8401300, 38761.0, 0.001296},
                {2439126.5, 4.3131700, 39126.0, 0.002592},
                {2439887.5, 4.2131700, 39126.0, 0.002592},
            };
            const double* row = kRubber[0];
            for (const auto& r : kRubber)
                if (utc_jd >= r[0]) row = r;
            const double mjd = utc_jd - constants::MJD_OFFSET;
            return row[1] + (mjd - row[2]) * row[3];
        }
        double result = 0.0;
        bool expired = false;
        double expiry = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            expired = expiration_jd_ > 0 && utc_jd > expiration_jd_;
            expiry = expiration_jd_;
            result = entries_.empty() ? 37 : entries_[0].tai_utc;
            for (const auto& entry : entries_) {
                if (utc_jd >= entry.jd) result = entry.tai_utc;
                else break;
            }
        }
        if (expired) {
            // Outside the lock: the diagnostics handler may call back in.
            const auto [y, m, d, h, mi, s] = jd_to_calendar(expiry);
            (void)h; (void)mi; (void)s;
            diagnostics::report(diagnostics::Category::DataCoverage,
                "leap-second table expires " + std::to_string(y) + "-" +
                (m < 10 ? "0" : "") + std::to_string(m) + "-" + (d < 10 ? "0" : "") +
                std::to_string(d) + "; later UTC dates assume no further leap second (each "
                "unannounced one is 1 s, ~7.5 km along-track at LEO in UTC labels). Load a "
                "current leap-seconds.list (Python: ballistic.load()).",
                "leap-seconds-expired");
        }
        return result;
    }

    /// True if the table came from load(), false if it is the embedded one.
    [[nodiscard]] bool is_loaded_from_file() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_from_file_;
    }

    /// Number of leap-second rows (1972 onwards).
    [[nodiscard]] size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    /// UTC Julian date of the newest leap second (0 if the table is empty).
    [[nodiscard]] double last_entry_jd() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.empty() ? 0.0 : entries_.back().jd;
    }

    /// The newest TAI - UTC value [s] (37 if the table is empty).
    [[nodiscard]] int current_tai_utc() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.empty() ? 37 : entries_.back().tai_utc;
    }

    /// UTC Julian date until which the table is known to be complete (0 if unknown).
    [[nodiscard]] double expiration_jd() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return expiration_jd_;
    }

    /// Why the last load() failed (empty after a success).
    [[nodiscard]] std::string last_error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_error_;
    }

    /// Go back to the embedded table and its expiry date.
    void reset_to_embedded() {
        std::lock_guard<std::mutex> lock(mutex_);
        init_embedded();               // also sets the embedded expiry
        loaded_from_file_ = false;
        last_error_.clear();
    }

private:
    LeapSecondTable() {
        init_embedded();
    }

    void init_embedded() {
        // Historical leap seconds (TAI-UTC after each date)
        // Source: IERS Bulletin C
        // Last update: Bulletin C 69 (no leap second announced through June 2025)
        entries_ = {
            {calendar_to_jd(1972, 1, 1), 10},
            {calendar_to_jd(1972, 7, 1), 11},
            {calendar_to_jd(1973, 1, 1), 12},
            {calendar_to_jd(1974, 1, 1), 13},
            {calendar_to_jd(1975, 1, 1), 14},
            {calendar_to_jd(1976, 1, 1), 15},
            {calendar_to_jd(1977, 1, 1), 16},
            {calendar_to_jd(1978, 1, 1), 17},
            {calendar_to_jd(1979, 1, 1), 18},
            {calendar_to_jd(1980, 1, 1), 19},
            {calendar_to_jd(1981, 7, 1), 20},
            {calendar_to_jd(1982, 7, 1), 21},
            {calendar_to_jd(1983, 7, 1), 22},
            {calendar_to_jd(1985, 7, 1), 23},
            {calendar_to_jd(1988, 1, 1), 24},
            {calendar_to_jd(1990, 1, 1), 25},
            {calendar_to_jd(1991, 1, 1), 26},
            {calendar_to_jd(1992, 7, 1), 27},
            {calendar_to_jd(1993, 7, 1), 28},
            {calendar_to_jd(1994, 7, 1), 29},
            {calendar_to_jd(1996, 1, 1), 30},
            {calendar_to_jd(1997, 7, 1), 31},
            {calendar_to_jd(1999, 1, 1), 32},
            {calendar_to_jd(2006, 1, 1), 33},
            {calendar_to_jd(2009, 1, 1), 34},
            {calendar_to_jd(2012, 7, 1), 35},
            {calendar_to_jd(2015, 7, 1), 36},
            {calendar_to_jd(2017, 1, 1), 37},
        };
        // Valid until the expiry of the IERS leap-seconds.list these entries
        // match ("File expires on 28 June 2027", retrieved 2026-11-02): no  // guard-exempt(PUB-DATE): data
        // leap second is announced before that date.
        expiration_jd_ = calendar_to_jd(2027, 6, 28);
    }

    mutable std::mutex mutex_;
    std::vector<LeapSecondEntry> entries_;
    bool loaded_from_file_ = false;
    double expiration_jd_ = 0.0;
    std::string last_error_;
};

/// Shorthand for LeapSecondTable::instance().
inline LeapSecondTable& get_leap_second_table() {
    return LeapSecondTable::instance();
}

} // namespace time
} // namespace ballistic
