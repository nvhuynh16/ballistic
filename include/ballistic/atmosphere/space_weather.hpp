#pragma once

// =============================================================================
// Space-weather file loader (CelesTrak SW-All.csv)
// =============================================================================
//
// Loads the measured (and forecast) daily space-weather indices that drive the
// NRLMSIS 2.1 atmosphere model, and serves them per day, per epoch, or as a
// SpaceWeatherHistory time series (space_weather_history.hpp).
//
// Data source: CelesTrak Space Weather, https://celestrak.org/SpaceData/
//   - SW-All.csv: every day since 1957, plus ~45 days of daily forecasts and
//     monthly predictions after the last measured day
//   - SW-Last5Years.csv: the last 5 years (smaller file, same layout)
//
// The indices:
// - F10.7: the Sun's 10.7 cm radio flux in solar flux units (1 SFU =
//   10^-22 W/m^2/Hz), a proxy for the solar ultraviolet that heats the upper
//   atmosphere
// - Ap: daily planetary geomagnetic index (0-400), a measure of geomagnetic
//   disturbance; also given every 3 hours
// - Kp: 3-hourly geomagnetic index (0-9), the logarithmic cousin of Ap
//
// Usage:
//   using namespace ballistic::atmosphere;
//
//   // Load space weather data
//   if (SpaceWeatherData::instance().load_celestrak_csv("SW-All.csv")) {
//       // Get conditions for a specific epoch
//       Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
//       WeatherConditions weather = SpaceWeatherData::instance().get_conditions(epoch);
//   }
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/diagnostics.hpp"
#include "../core/errors.hpp"
#include "../time/epoch.hpp"
#include "space_weather_history.hpp"
#include <vector>
#include <filesystem>
#include <optional>
#include <memory>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <string>

namespace ballistic {
namespace atmosphere {

/// One day (one row) of the CelesTrak file.
struct SpaceWeatherRecord {
    double mjd;                     // MJD of the record's date at 12:00 UTC
                                    // (date_to_mjd returns JDN - 2400000.5)
    double f107_obs;                // Observed F10.7 flux (SFU)
    double f107_adj;                // Adjusted F10.7 flux to 1 AU
    double f107_81_center;          // 81-day centered average F10.7
    double f107_81_last;            // 81-day trailing average F10.7
    double ap_daily;                // Daily Ap average
    std::array<double, 8> ap_3hr;   // 8 three-hourly Ap values (0-3h, 3-6h, ..., 21-24h)
    std::array<double, 8> kp_3hr;   // 8 three-hourly Kp values
    double kp_sum;                  // Daily Kp sum
    int sunspot_number;             // International Sunspot Number
    bool ap_observed;               // all 8 three-hourly Ap present in the file
                                    // (CelesTrak's monthly predicted rows have
                                    // none — ap_3hr is then the 4.0 default)
    bool predicted;                 // F10.7_DATA_TYPE is PRD (daily prediction)
                                    // or PRM (monthly). CelesTrak's PRD rows
                                    // DO carry 3-hourly Ap — predicted values —
                                    // so ap_observed alone cannot tell them
                                    // from measurements. OBS and INT (gap fill
                                    // in the historical record) are measured.

    /// True for measured weather: all 3-hourly Ap present and not a prediction.
    [[nodiscard]] bool measured() const noexcept { return ap_observed && !predicted; }

    SpaceWeatherRecord()
        : mjd(0.0)
        , f107_obs(150.0)
        , f107_adj(150.0)
        , f107_81_center(150.0)
        , f107_81_last(150.0)
        , ap_daily(4.0)
        , ap_3hr{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}
        , kp_3hr{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
        , kp_sum(8.0)
        , sunspot_number(0)
        , ap_observed(false)
        , predicted(false) {}
};

/// The loaded space-weather file: one instance per process (a singleton,
/// reached through instance()), thread-safe. Load it once with
/// load_celestrak_csv (Python: ballistic.load()).
class SpaceWeatherData {
public:
    /// The process-wide instance.
    static SpaceWeatherData& instance() {
        static SpaceWeatherData inst;
        return inst;
    }

    /// Non-fatal issues from the last load_celestrak_csv: values that were
    /// blank or could not be parsed and were replaced by defaults (F10.7 150,
    /// Ap 4, or the day's mean). Empty = clean load. The load still succeeds;
    /// a DataSkipped diagnostic is also reported. (A file without the
    /// F10.7_OBS or AP_AVG column is refused outright — see
    /// load_celestrak_csv.)
    [[nodiscard]] const std::vector<std::string>& warnings() const {
        return warnings_;
    }

    /// Load a CelesTrak CSV (SW-All.csv or SW-Last5Years.csv), replacing
    /// anything loaded before. Returns false if the file cannot be opened or
    /// holds no data rows; throws ballistic::Error if the header lacks the
    /// F10.7_OBS or AP_AVG column (every day would read as quiet weather).
    bool load_celestrak_csv(const std::filesystem::path& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }

        data_.clear();
        warnings_.clear();
        value_parse_failures_ = 0;
        col_f107_type_ = -1;
        std::string line;
        bool header_found = false;

        // Column indices (will be determined from header)
        int col_date = -1;
        int col_f107_obs = -1;
        int col_f107_adj = -1;
        int col_f107_81_center = -1;
        int col_f107_81_last = -1;
        int col_ap_avg = -1;
        int col_kp_sum = -1;
        int col_sunspot = -1;
        std::array<int, 8> col_ap_3hr = {-1, -1, -1, -1, -1, -1, -1, -1};
        std::array<int, 8> col_kp_3hr = {-1, -1, -1, -1, -1, -1, -1, -1};

        while (std::getline(file, line)) {
            // Skip empty lines
            if (line.empty()) continue;

            // Parse header line
            if (!header_found && line.find("DATE") != std::string::npos) {
                header_found = true;
                parse_header(line, col_date, col_f107_obs, col_f107_adj,
                           col_f107_81_center, col_f107_81_last,
                           col_ap_avg, col_ap_3hr, col_kp_3hr, col_kp_sum, col_sunspot);
                // A drag-critical column missing (renamed/reordered header)
                // would make every record quiet weather (F10.7 150, Ap 4):
                // density biased low everywhere. Such a file is refused.
                if (col_f107_obs < 0 || col_ap_avg < 0) {
                    data_.clear();
                    loaded_ = false;
                    throw Error("space weather file " + filepath.string() + ": the " +
                        std::string(col_f107_obs < 0 ? "F10.7_OBS" : "AP_AVG") +
                        " column is not in the header, so every day would read as "
                        "quiet weather. This is not a CelesTrak SW-All.csv layout.");
                }
                continue;
            }

            if (!header_found) continue;

            // Parse data line
            SpaceWeatherRecord record;
            if (parse_data_line(line, record, col_date, col_f107_obs, col_f107_adj,
                              col_f107_81_center, col_f107_81_last,
                              col_ap_avg, col_ap_3hr, col_kp_3hr, col_kp_sum, col_sunspot)) {
                data_.push_back(record);
            }
        }

        if (value_parse_failures_ > 0) {
            warnings_.push_back(std::to_string(value_parse_failures_) +
                " space weather value(s) blank or unparseable — replaced with "
                "per-field defaults");
        }

        // Sort by MJD for binary search
        std::sort(data_.begin(), data_.end(),
                  [](const SpaceWeatherRecord& a, const SpaceWeatherRecord& b) {
                      return a.mjd < b.mjd;
                  });

        loaded_ = !data_.empty();
        full_history_[0].reset();
        full_history_[1].reset();
        if (value_parse_failures_ > 0) {
            diagnostics::report(diagnostics::Category::DataSkipped,
                filepath.string() + ": " + warnings_.back() + " (F10.7 150, Ap 4 "
                "or the day's mean). Density on those days is not measured weather.");
        }
        return loaded_;
    }

    /// The row of the UTC day containing `mjd` (Modified Julian Date, UTC),
    /// or nullopt when the file has no row for that day. (Never the NEAREST
    /// row: that would be the first or last row outside the file, and the
    /// previous day at exactly 00:00 UTC, where the two noon stamps tie.)
    [[nodiscard]] std::optional<SpaceWeatherRecord> get_record(double mjd) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const SpaceWeatherRecord* r = get_record_internal(mjd);
        if (!r) return std::nullopt;
        return *r;
    }

    /// Weather at one epoch: F10.7 and daily Ap interpolated linearly between
    /// the two noon-stamped rows around it, and the 3-hourly Ap history from
    /// the rows themselves. An older single-epoch path; the drag models use
    /// history() / full_history(). Throws DataCoverageError when nothing is
    /// loaded, the epoch is more than half a day outside the file, a row has
    /// no 3-hourly Ap, or a row is a forecast and `allow_predicted` is false;
    /// std::out_of_range when a day the 57-hour Ap lookback needs is missing
    /// (from build_ap_history). Never substitutes
    /// quiet defaults. `*used_predicted` (if given) is set true when a
    /// forecast row went in.
    [[nodiscard]] WeatherConditions get_conditions(const time::Epoch& epoch,
                                                   bool allow_predicted = true,
                                                   bool* used_predicted = nullptr) const {
        std::lock_guard<std::mutex> lock(mutex_);

        WeatherConditions result;

        if (data_.empty()) {
            throw DataCoverageError(
                "SpaceWeatherData::get_conditions: no space weather loaded, so there is no "
                "F10.7/Ap for drag (no quiet defaults are substituted). Load SW-All.csv "
                "(Python: ballistic.load(); C++: load_celestrak_csv)");
        }

        double mjd = epoch.mjd(time::TimeScale::UTC);

        // Find bracketing records for interpolation
        auto it = std::lower_bound(data_.begin(), data_.end(), mjd,
                                   [](const SpaceWeatherRecord& r, double m) {
                                       return r.mjd < m;
                                   });

        const SpaceWeatherRecord* r0 = nullptr;
        const SpaceWeatherRecord* r1 = nullptr;
        double t = 0.0;  // Interpolation factor

        if (it == data_.end() || it == data_.begin()) {
            // Records are stamped at noon; within half a day of the file's
            // first/last record the nearest record is used.
            const SpaceWeatherRecord& edge = it == data_.end() ? data_.back() : data_.front();
            if (std::abs(mjd - edge.mjd) > 0.5) {
                throw DataCoverageError(
                    "SpaceWeatherData::get_conditions: " + mjd_to_date_string(std::floor(mjd)) +
                    " is outside the loaded space weather (" +
                    mjd_to_date_string(std::floor(data_.front().mjd)) + " to " +
                    mjd_to_date_string(std::floor(data_.back().mjd)) + " UTC); the nearest "
                    "day is not substituted. Load a SW-All.csv that covers the date "
                    "(Python: ballistic.load()).");
            }
            r0 = &edge;
            r1 = r0;
            t = 0.0;
        } else {
            // Interpolate between two records
            r1 = &(*it);
            r0 = &(*std::prev(it));
            double dt = r1->mjd - r0->mjd;
            t = (dt > 0.0) ? (mjd - r0->mjd) / dt : 0.0;
        }

        for (const SpaceWeatherRecord* r : {r0, r1}) {
            const bool usable = r->ap_observed && (!r->predicted || allow_predicted);
            if (!usable) {
                const auto last = last_measured_mjd_internal();
                throw DataCoverageError(
                    "SpaceWeatherData::get_conditions: space weather for " +
                    mjd_to_date_string(std::floor(r->mjd)) + " is " +
                    (r->predicted ? "a forecast" : "incomplete (no 3-hourly Ap)") +
                    ", not a measurement" +
                    (last ? "; measured through " + mjd_to_date_string(*last) : std::string()) +
                    ". Load a newer SW-All.csv.");
            }
            if (r->predicted && used_predicted) *used_predicted = true;
        }

        // Interpolate F10.7 values
        result.f107_daily = r0->f107_obs + t * (r1->f107_obs - r0->f107_obs);
        result.f107_average = r0->f107_81_center + t * (r1->f107_81_center - r0->f107_81_center);
        result.ap_daily = r0->ap_daily + t * (r1->ap_daily - r0->ap_daily);

        // Build Ap history array for NRLMSIS 2.1
        // The array should contain: [avg, current 3hr, 3hr ago, 6hr ago, 9hr ago, 12-33hr avg, 36-57hr avg]
        build_ap_history(epoch, result.ap_history);

        return result;
    }

    /// MJD stamps (noon UTC) of the first and last rows; {0, 0} if empty.
    [[nodiscard]] std::pair<double, double> mjd_range() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) {
            return {0.0, 0.0};
        }
        return {data_.front().mjd, data_.back().mjd};
    }

    /// True when a load succeeded and holds at least one row.
    [[nodiscard]] bool is_loaded() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return loaded_;
    }

    /// Number of rows loaded.
    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }

    /// Weather history over `window` for the table builders: daily F10.7 as
    /// NRLMSIS wants it — the PREVIOUS day's observed value, stamped at the
    /// start of the next day so SpaceWeatherHistory::at()'s latest-at-or-
    /// before lookup returns it — its 81-day centered mean, and every 3-hourly
    /// Ap from 3 days before window.start (NRLMSIS looks back 57 h) through
    /// the day containing window.end. Throws std::out_of_range if any of
    /// those days is missing or has no 3-hourly Ap: a silent quiet substitute
    /// would bias density low.
    ///
    /// allow_predicted (default) also accepts CelesTrak's daily forecasts
    /// (PRD rows, which carry predicted 3-hourly Ap, ~45 days past the last
    /// measured day); false accepts measurements only. Monthly predictions
    /// (PRM) have no Ap and are never used. *used_predicted (if given) reports
    /// whether any forecast day went in — callers should say so.
    [[nodiscard]] SpaceWeatherHistory history(const TimeWindow& window,
                                              bool allow_predicted = true,
                                              bool* used_predicted = nullptr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const Epoch lo = window.start < window.end ? window.start : window.end;
        const Epoch hi = window.start < window.end ? window.end : window.start;
        const double first_day = std::floor(lo.mjd(time::TimeScale::UTC)) - 3.0;
        const double last_day = std::floor(hi.mjd(time::TimeScale::UTC));
        return build_history_internal(first_day, last_day, allow_predicted, used_predicted);
    }

    /// The whole loaded file as one history — what the live drag model runs
    /// on (~25k days since 1957 is a few MB). Covers the longest gap-free run
    /// of usable days that ends at the last usable day: measured days, plus
    /// CelesTrak's daily forecasts when allow_predicted. Built once per load
    /// and shared. Throws std::out_of_range if nothing usable is loaded.
    struct FullHistory {
        SpaceWeatherHistory history;
        TimeWindow valid;                 // where at() has its full 3-day Ap lookback
        std::optional<Epoch> measured_until;  // forecasts after this (nullopt: none used)
    };
    [[nodiscard]] std::shared_ptr<const FullHistory> full_history(bool allow_predicted = true) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& slot = full_history_[allow_predicted ? 1 : 0];
        if (slot) return slot;
        auto usable = [&](const SpaceWeatherRecord& r) {
            return r.ap_observed && (!r.predicted || allow_predicted);
        };
        long last = static_cast<long>(data_.size()) - 1;
        while (last >= 0 && !usable(data_[last])) --last;
        if (last < 0) {
            throw std::out_of_range(
                "SpaceWeatherData::full_history: no usable space weather loaded "
                "(Python: ballistic.load(); C++: load_celestrak_csv)");
        }
        long first = last;
        while (first > 0 && usable(data_[first - 1]) &&
               std::floor(data_[first].mjd) - std::floor(data_[first - 1].mjd) == 1.0) {
            --first;
        }
        const double first_day = std::floor(data_[first].mjd);
        const double last_day = std::floor(data_[last].mjd);
        auto out = std::make_shared<FullHistory>();
        out->history = build_history_internal(first_day, last_day, allow_predicted, nullptr);
        out->valid = TimeWindow{day_stamp(first_day + 3.0), day_stamp(last_day + 1.0)};
        if (data_[last].predicted) {
            if (auto m = last_measured_mjd_internal())
                out->measured_until = Epoch::from_mjd(*m + 1.0, time::TimeScale::UTC);
        }
        slot = out;
        return slot;
    }

    /// MJD (start of day, UTC) of the last MEASURED row, nullopt if none.
    /// Drag environments for later dates raise unless forecasts are allowed.
    [[nodiscard]] std::optional<double> last_measured_mjd() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_measured_mjd_internal();
    }

    /// Forget the loaded file.
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
        loaded_ = false;
        full_history_[0].reset();
        full_history_[1].reset();
    }

private:
    SpaceWeatherData() = default;
    SpaceWeatherData(const SpaceWeatherData&) = delete;
    SpaceWeatherData& operator=(const SpaceWeatherData&) = delete;

    std::vector<SpaceWeatherRecord> data_;
    bool loaded_ = false;
    mutable std::mutex mutex_;

    // Non-fatal issues from the last load (see warnings()).
    std::vector<std::string> warnings_;
    size_t value_parse_failures_ = 0;   // non-empty tokens that failed to parse
    int col_f107_type_ = -1;            // F10.7_DATA_TYPE column (OBS/INT/PRD/PRM)
    mutable std::shared_ptr<const FullHistory> full_history_[2];  // [allow_predicted]

    // Parse header line to determine column indices
    void parse_header(const std::string& line,
                      int& col_date, int& col_f107_obs, int& col_f107_adj,
                      int& col_f107_81_center, int& col_f107_81_last,
                      int& col_ap_avg, std::array<int, 8>& col_ap_3hr,
                      std::array<int, 8>& col_kp_3hr, int& col_kp_sum, int& col_sunspot) {
        std::stringstream ss(line);
        std::string token;
        int col = 0;

        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos) {
                token = token.substr(start, end - start + 1);
            }

            if (token == "DATE") col_date = col;
            else if (token == "F10.7_OBS") col_f107_obs = col;
            else if (token == "F10.7_ADJ") col_f107_adj = col;
            else if (token == "F10.7_OBS_CENTER81" || token == "F10.7_81_CTR") col_f107_81_center = col;
            else if (token == "F10.7_OBS_LAST81" || token == "F10.7_81_LST") col_f107_81_last = col;
            else if (token == "AP_AVG") col_ap_avg = col;
            else if (token == "KP_SUM") col_kp_sum = col;
            else if (token == "ISN") col_sunspot = col;
            else if (token == "F10.7_DATA_TYPE") col_f107_type_ = col;
            // 3-hourly Ap values (AP1 through AP8)
            else if (token == "AP1") col_ap_3hr[0] = col;
            else if (token == "AP2") col_ap_3hr[1] = col;
            else if (token == "AP3") col_ap_3hr[2] = col;
            else if (token == "AP4") col_ap_3hr[3] = col;
            else if (token == "AP5") col_ap_3hr[4] = col;
            else if (token == "AP6") col_ap_3hr[5] = col;
            else if (token == "AP7") col_ap_3hr[6] = col;
            else if (token == "AP8") col_ap_3hr[7] = col;
            // 3-hourly Kp values (KP1 through KP8)
            else if (token == "KP1") col_kp_3hr[0] = col;
            else if (token == "KP2") col_kp_3hr[1] = col;
            else if (token == "KP3") col_kp_3hr[2] = col;
            else if (token == "KP4") col_kp_3hr[3] = col;
            else if (token == "KP5") col_kp_3hr[4] = col;
            else if (token == "KP6") col_kp_3hr[5] = col;
            else if (token == "KP7") col_kp_3hr[6] = col;
            else if (token == "KP8") col_kp_3hr[7] = col;

            col++;
        }
    }

    // Parse a data line
    bool parse_data_line(const std::string& line, SpaceWeatherRecord& record,
                         int col_date, int col_f107_obs, int col_f107_adj,
                         int col_f107_81_center, int col_f107_81_last,
                         int col_ap_avg, const std::array<int, 8>& col_ap_3hr,
                         const std::array<int, 8>& col_kp_3hr, int col_kp_sum, int col_sunspot) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos) {
                token = token.substr(start, end - start + 1);
            } else {
                token = "";
            }
            tokens.push_back(token);
        }

        if (tokens.empty()) return false;

        // Parse date (YYYY-MM-DD format)
        if (col_date >= 0 && col_date < static_cast<int>(tokens.size())) {
            int year, month, day;
            if (std::sscanf(tokens[col_date].c_str(), "%d-%d-%d", &year, &month, &day) == 3) {
                // Convert to MJD
                record.mjd = date_to_mjd(year, month, day);
            } else {
                return false;
            }
        } else {
            return false;
        }

        // Parse F10.7 values
        auto safe_parse = [&tokens, this](int col, double default_val) -> double {
            if (col >= 0 && col < static_cast<int>(tokens.size()) && !tokens[col].empty()) {
                try {
                    return std::stod(tokens[col]);
                } catch (const std::logic_error&) { ++value_parse_failures_; }  // present but garbage
            }
            return default_val;
        };

        if (col_f107_obs >= static_cast<int>(tokens.size()) || tokens[col_f107_obs].empty())
            ++value_parse_failures_;     // a blank F10.7 is a substitution too
        record.f107_obs = safe_parse(col_f107_obs, 150.0);
        record.f107_adj = safe_parse(col_f107_adj, record.f107_obs);
        record.f107_81_center = safe_parse(col_f107_81_center, record.f107_obs);
        record.f107_81_last = safe_parse(col_f107_81_last, record.f107_obs);
        record.ap_daily = safe_parse(col_ap_avg, 4.0);
        record.kp_sum = safe_parse(col_kp_sum, 8.0);
        record.sunspot_number = static_cast<int>(safe_parse(col_sunspot, 0.0));

        record.predicted =
            col_f107_type_ >= 0 && col_f107_type_ < static_cast<int>(tokens.size()) &&
            (tokens[col_f107_type_] == "PRD" || tokens[col_f107_type_] == "PRM");

        // Parse 3-hourly Ap values
        record.ap_observed = true;
        for (int i = 0; i < 8; ++i) {
            const int c = col_ap_3hr[i];
            if (c < 0 || c >= static_cast<int>(tokens.size()) || tokens[c].empty())
                record.ap_observed = false;
            record.ap_3hr[i] = safe_parse(c, record.ap_daily);
        }

        // Parse 3-hourly Kp values (stored as integers 0-90, representing 0.0-9.0)
        for (int i = 0; i < 8; ++i) {
            double kp = safe_parse(col_kp_3hr[i], 10.0);
            // CelesTrak stores Kp as integer * 10 (e.g., 23 = 2.3)
            record.kp_3hr[i] = kp / 10.0;
        }

        return true;
    }

    // Convert date to MJD
    static double date_to_mjd(int year, int month, int day) {
        // Julian day number by integer arithmetic (Fliegel-Van Flandern);
        // JDN - 2400000.5 is the MJD at noon of that date.
        int a = (14 - month) / 12;
        int y = year + 4800 - a;
        int m = month + 12 * a - 3;

        int jdn = day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
        return static_cast<double>(jdn) - 2400000.5;
    }

    // Build 7-element Ap history array for NRLMSIS 2.1
    // Array format:
    //   [0] = Daily AP
    //   [1] = 3-hour AP for current time
    //   [2] = 3-hour AP for 3 hours before current time
    //   [3] = 3-hour AP for 6 hours before current time
    //   [4] = 3-hour AP for 9 hours before current time
    //   [5] = Average of eight 3-hour AP indices from 12 to 33 hours before current time
    //   [6] = Average of eight 3-hour AP indices from 36 to 57 hours before current time
    void build_ap_history(const time::Epoch& epoch, std::array<double, 7>& history) const {
        double mjd = epoch.mjd(time::TimeScale::UTC);

        // Every day the 57-hour lookback reaches must be in the file; a
        // missing day raises rather than being filled with a guess.
        auto day = [&](double day_mjd) -> const SpaceWeatherRecord& {
            const SpaceWeatherRecord* r = get_record_internal(day_mjd);
            if (!r) {
                throw std::out_of_range(
                    "SpaceWeatherData: the 3-hourly Ap history for " +
                    mjd_to_date_string(std::floor(mjd)) + " needs " +
                    mjd_to_date_string(std::floor(day_mjd)) +
                    ", which is not in the loaded space weather (" +
                    (data_.empty() ? std::string("nothing loaded")
                                   : mjd_to_date_string(std::floor(data_.front().mjd)) + " to " +
                                     mjd_to_date_string(std::floor(data_.back().mjd))) + ").");
            }
            return *r;
        };
        const SpaceWeatherRecord* rec = &day(mjd);

        // [0] Daily Ap
        history[0] = rec->ap_daily;

        // Determine which 3-hour interval we're in (0-7)
        double day_fraction = mjd - std::floor(mjd);
        int interval = static_cast<int>(day_fraction * 8.0);
        if (interval > 7) interval = 7;

        // [1] Current 3-hour Ap
        history[1] = rec->ap_3hr[interval];

        // [2-4] Previous 3-hour periods (3h, 6h, 9h before)
        for (int i = 0; i < 3; ++i) {
            int prev_interval = interval - (i + 1);
            double prev_mjd = mjd;
            if (prev_interval < 0) {
                prev_mjd = mjd - 1.0;
                prev_interval += 8;
            }
            history[2 + i] = day(prev_mjd).ap_3hr[prev_interval];
        }

        // [5] Average 12-33 hours before (8 intervals)
        double sum_12_33 = 0.0;
        for (int i = 4; i < 12; ++i) {
            int prev_interval = interval - i;
            double prev_mjd = mjd;
            while (prev_interval < 0) {
                prev_mjd -= 1.0;
                prev_interval += 8;
            }
            sum_12_33 += day(prev_mjd).ap_3hr[prev_interval];
        }
        history[5] = sum_12_33 / 8.0;

        // [6] Average 36-57 hours before (8 intervals)
        double sum_36_57 = 0.0;
        for (int i = 12; i < 20; ++i) {
            int prev_interval = interval - i;
            double prev_mjd = mjd;
            while (prev_interval < 0) {
                prev_mjd -= 1.0;
                prev_interval += 8;
            }
            sum_36_57 += day(prev_mjd).ap_3hr[prev_interval];
        }
        history[6] = sum_36_57 / 8.0;
    }

    // A UTC MJD stamp for the weather series. UTC did not exist before
    // 1961-01-01 (MJD 37300), where Epoch refuses UTC; the file's 1957-1960
    // days are stamped in TT instead (under a minute apart, which does not
    // matter for daily and 3-hourly indices).
    [[nodiscard]] static Epoch day_stamp(double mjd) {
        return Epoch::from_mjd(mjd, mjd < 37300.0 ? time::TimeScale::TT : time::TimeScale::UTC);
    }

    // History over whole UTC days [first_day, last_day] (MJD), checking each
    // day is usable. No lock.
    SpaceWeatherHistory build_history_internal(double first_day, double last_day,
                                               bool allow_predicted,
                                               bool* used_predicted) const {
        std::vector<Epoch> times_daily, times_ap;
        std::vector<double> f107, f107_81, ap;
        for (double d = first_day; d <= last_day; d += 1.0) {
            const SpaceWeatherRecord* rec = exact_day_internal(d);
            // The 81-day mean stamped at the start of day d+1 is that day's;
            // it moves ~0.1 SFU/day, so the last day may reuse its own.
            const SpaceWeatherRecord* next = exact_day_internal(d + 1.0);
            const bool usable = rec && rec->ap_observed &&
                                (!rec->predicted || allow_predicted);
            if (!usable) {
                const std::string day = mjd_to_date_string(d);
                const auto last = last_measured_mjd_internal();
                const std::string through = last
                    ? " The loaded file has measured weather through " +
                      mjd_to_date_string(*last) + "."
                    : "";
                std::string what;
                if (rec && rec->ap_observed && rec->predicted)
                    what = " is a forecast (CelesTrak PRD row) and forecasts were"
                           " refused (allow_predicted_weather=false).";
                else if (rec)
                    what = " has no 3-hourly Ap: past CelesTrak's ~45-day daily"
                           " forecast only monthly F10.7 predictions exist, which"
                           " NRLMSIS cannot run on.";
                else
                    what = " is not in the loaded file (neither measured nor"
                           " forecast). Load a newer SW-All.csv (Python:"
                           " ballistic.load()) or pass the weather explicitly.";
                throw std::out_of_range(
                    "SpaceWeatherData::history: space weather for " + day +
                    " (needed for this arc, NRLMSIS looks back 3 days)" + what +
                    through);
            }
            if (rec->predicted && used_predicted) *used_predicted = true;
            times_daily.push_back(day_stamp(d + 1.0));
            f107.push_back(rec->f107_obs);
            f107_81.push_back(next ? next->f107_81_center : rec->f107_81_center);
            for (int k = 0; k < 8; ++k) {
                times_ap.push_back(day_stamp(d + k / 8.0));
                ap.push_back(rec->ap_3hr[k]);
            }
        }
        return SpaceWeatherHistory(std::move(times_daily), std::move(f107),
                                   std::move(f107_81), std::move(times_ap),
                                   std::move(ap));
    }

    std::optional<double> last_measured_mjd_internal() const {
        for (auto it = data_.rbegin(); it != data_.rend(); ++it)
            if (it->measured()) return std::floor(it->mjd);
        return std::nullopt;
    }

    static std::string mjd_to_date_string(double day_mjd) {
        // Civil date from MJD (Fliegel-Van Flandern via JDN).
        long jdn = static_cast<long>(std::floor(day_mjd)) + 2400001L;
        long l = jdn + 68569L;
        const long n = 4 * l / 146097L;
        l = l - (146097L * n + 3) / 4;
        const long i = 4000 * (l + 1) / 1461001L;
        l = l - 1461 * i / 4 + 31;
        const long j = 80 * l / 2447;
        const long day = l - 2447 * j / 80;
        l = j / 11;
        const long month = j + 2 - 12 * l;
        const long year = 100 * (n - 49) + i + l;
        char buf[16];
        std::snprintf(buf, sizeof buf, "%04ld-%02ld-%02ld", year, month, day);
        return buf;
    }

    // Record for the UTC day starting at integer MJD `day_mjd`, or nullptr.
    // Records are stamped at noon of their date, so the match is the record
    // in [day_mjd, day_mjd + 1). No lock.
    const SpaceWeatherRecord* exact_day_internal(double day_mjd) const {
        auto it = std::lower_bound(data_.begin(), data_.end(), day_mjd,
                                   [](const SpaceWeatherRecord& r, double m) {
                                       return r.mjd < m;
                                   });
        if (it == data_.end() || it->mjd >= day_mjd + 1.0) return nullptr;
        return &*it;
    }

    // Row of the UTC day containing mjd (rows are stamped at noon, so
    // floor(row.mjd) is the day), or nullptr. No lock.
    const SpaceWeatherRecord* get_record_internal(double mjd) const {
        const double d = std::floor(mjd);
        auto it = std::lower_bound(data_.begin(), data_.end(), d,
                                   [](const SpaceWeatherRecord& r, double m) {
                                       return std::floor(r.mjd) < m;
                                   });
        if (it == data_.end() || std::floor(it->mjd) != d) return nullptr;
        return &*it;
    }
};

/// Shorthand for SpaceWeatherData::instance().get_conditions(epoch): the
/// weather at one epoch from the loaded file (same throws).
///   auto weather = get_weather_conditions(epoch);
[[nodiscard]] inline WeatherConditions get_weather_conditions(const time::Epoch& epoch) {
    return SpaceWeatherData::instance().get_conditions(epoch);
}

} // namespace atmosphere
} // namespace ballistic
