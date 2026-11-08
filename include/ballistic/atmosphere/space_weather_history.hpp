#pragma once

// =============================================================================
// SpaceWeatherHistory — space-weather inputs of the atmosphere models over time
// =============================================================================
//
// "Space weather" here means the two inputs that drive the upper atmosphere
// in NRLMSIS 2.1 (density) and HWM14 (wind):
//   - F10.7: the Sun's radio flux at 10.7 cm wavelength, in solar flux units
//     (sfu), a daily proxy for the solar ultraviolet that heats the upper
//     atmosphere. More flux = hotter, denser air at satellite altitudes.
//   - Ap: the daily (or 3-hourly) planetary geomagnetic index, a measure of
//     geomagnetic disturbance; storms heat the upper atmosphere too.
//
// NRLMSIS takes:
//   - F10.7 of the PREVIOUS day
//   - the 81-day average of F10.7 centered on the day
//   - the daily Ap (24-hour mean), and in its storm mode a short history of
//     3-hourly Ap (see at() for the exact layout)
//
// WeatherConditions (core/types.hpp) holds those values for one moment;
// SpaceWeatherHistory holds them as a time series and returns the
// WeatherConditions for any epoch, so a drag model running over days uses
// the weather of each moment. It is the weather the live drag model and the
// table builders read.
//
// For a single snapshot, from_constant() wraps one WeatherConditions into a
// history that returns it at every epoch (or within a window).
//
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../core/constants.hpp"
#include "../core/defaults.hpp"
#include "../time/epoch.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace ballistic {
namespace atmosphere {

/// A span of time from `start` to `end`. The users of this type in the
/// library treat both ends as included.
struct TimeWindow {
    Epoch start;
    Epoch end;
};

/// Daily F10.7 and 3-hourly Ap as time series; at() gives the
/// WeatherConditions for any epoch inside the span.
class SpaceWeatherHistory {
public:
    SpaceWeatherHistory() = default;

    /// `times_daily` / `f107_daily` [sfu]: one F10.7 value per day, stamped at
    /// the moment it should take effect (at() uses the latest stamp at or
    /// before the epoch). `f107_81day_avg` [sfu]: the 81-day average at the
    /// same stamps, or empty to compute it here as a centered running mean
    /// (shorter near the series ends). `times_ap` / `ap_values`: the 3-hourly
    /// Ap index at the start of each 3-hour bin. Times must be strictly
    /// increasing. Throws std::invalid_argument on mismatched sizes or
    /// unordered times.
    SpaceWeatherHistory(std::vector<Epoch> times_daily,
                        std::vector<double> f107_daily,
                        std::vector<double> f107_81day_avg,
                        std::vector<Epoch> times_ap,
                        std::vector<double> ap_values)
        : times_daily_(std::move(times_daily))
        , f107_daily_(std::move(f107_daily))
        , f107_81day_avg_(std::move(f107_81day_avg))
        , times_ap_(std::move(times_ap))
        , ap_values_(std::move(ap_values))
    {
        validate();
        if (f107_81day_avg_.empty() && !f107_daily_.empty()) {
            f107_81day_avg_ = compute_81day_average(f107_daily_);
        } else if (f107_81day_avg_.size() != f107_daily_.size()) {
            throw std::invalid_argument(
                "SpaceWeatherHistory: f107_81day_avg size must match f107_daily "
                "or be empty for auto-compute");
        }
    }

    /// History that returns `w` at EVERY epoch (with its 3-hourly Ap history
    /// set to the daily Ap, as the windowed form does). Use when one constant
    /// snapshot should apply to an arc of any length, e.g.
    /// from_constant(WeatherConditions::quiet()) to choose quiet conditions
    /// on purpose.
    [[nodiscard]] static SpaceWeatherHistory from_constant(WeatherConditions w) {
        SpaceWeatherHistory h;
        h.flat_ = true;
        h.flat_value_ = w;
        h.flat_value_.ap_history.fill(w.ap_daily);
        return h;
    }

    /// History that returns `w` at every epoch within `window` and throws
    /// std::out_of_range outside it. Prefer the windowless form above unless
    /// the window is wanted. Throws std::invalid_argument unless
    /// window.end > window.start.
    [[nodiscard]] static SpaceWeatherHistory from_constant(WeatherConditions w,
                                                            TimeWindow window) {
        if (!(window.end > window.start)) {
            throw std::invalid_argument(
                "SpaceWeatherHistory::from_constant: window.end must be > window.start");
        }
        // Two daily samples (start and end) suffice for a flat profile;
        // f107_81day_avg supplied directly to skip auto-compute on a 2-point series.
        std::vector<Epoch> times_daily{window.start, window.end};
        std::vector<double> f107_daily{w.f107_daily, w.f107_daily};
        std::vector<double> f107_avg{w.f107_average, w.f107_average};
        // 3-hourly Ap samples covering the window.
        const double window_s = window.end - window.start;
        const double step_s = 3.0 * 3600.0;
        const std::size_t n_ap = static_cast<std::size_t>(
            std::max<double>(2.0, std::ceil(window_s / step_s) + 1.0));
        std::vector<Epoch> times_ap;
        std::vector<double> ap_values;
        times_ap.reserve(n_ap);
        ap_values.reserve(n_ap);
        for (std::size_t i = 0; i < n_ap; ++i) {
            times_ap.push_back(window.start + static_cast<double>(i) * step_s);
            ap_values.push_back(w.ap_daily);
        }
        return SpaceWeatherHistory(std::move(times_daily), std::move(f107_daily),
                                    std::move(f107_avg), std::move(times_ap),
                                    std::move(ap_values));
    }

    /// The weather the models should use at `epoch_in`:
    /// - F10.7 daily: the latest daily value stamped at or before the epoch.
    ///   NRLMSIS wants the PREVIOUS day's flux, so stamp each day's value at
    ///   the start of the NEXT day, as SpaceWeatherData::history() does.
    /// - F10.7 81-day average: linear interpolation between stamps.
    /// - Daily Ap: mean of the 3-hourly values in the 24 hours ending at the
    ///   epoch (the 8 bins up to and including the current one).
    /// - ap_history in the order of NRLMSIS's AP array: [0] daily Ap (the
    ///   24-h mean above — the ONLY Ap input in the daily-Ap mode this library
    ///   runs NRLMSIS in), [1] current 3-hourly Ap, [2..4] 3/6/9 h before,
    ///   [5] mean of the eight values 12–33 h before, [6] mean of the eight
    ///   values 36–57 h before. (Shifting this by one — the current 3-hourly
    ///   value in [0] — would feed NRLMSIS a storm's instantaneous Ap as its
    ///   daily Ap.)
    /// Throws std::out_of_range if the history is empty or the epoch is
    /// outside the span of the daily series (beyond a 1e-9 s rounding
    /// tolerance, defaults::EPOCH_EQUALITY_SECONDS), and DataCoverageError if
    /// it has daily F10.7 but no Ap.
    [[nodiscard]] WeatherConditions at(const Epoch& epoch_in) const {
        if (flat_) {
            return flat_value_;
        }
        if (empty()) {
            throw std::out_of_range("SpaceWeatherHistory::at: history is empty");
        }
        // An epoch within rounding of the window edge (a step sized to land on
        // it can overshoot by ~1e-11 s) is taken AT the edge.
        constexpr double tol = defaults::EPOCH_EQUALITY_SECONDS;
        if (times_daily_.front() - epoch_in > tol || epoch_in - times_daily_.back() > tol) {
            throw std::out_of_range(
                "SpaceWeatherHistory::at: epoch outside daily-series window");
        }
        const Epoch epoch = epoch_in < times_daily_.front() ? times_daily_.front()
                          : epoch_in > times_daily_.back()  ? times_daily_.back()
                          : epoch_in;
        WeatherConditions w;
        w.f107_daily   = nearest_prior(times_daily_, f107_daily_, epoch);
        w.f107_average = linear_interp(times_daily_, f107_81day_avg_, epoch);
        w.ap_daily     = mean_recent_ap(epoch, /*hours_window*/ 24.0);

        const auto ap_at = [&](double hours_before) {
            return nearest_prior(times_ap_, ap_values_, epoch - hours_before * 3600.0);
        };
        const auto ap_mean8 = [&](double first_hours_before) {
            double sum = 0.0;
            for (int k = 0; k < 8; ++k) sum += ap_at(first_hours_before + 3.0 * k);
            return sum / 8.0;
        };
        w.ap_history[0] = w.ap_daily;
        w.ap_history[1] = ap_at(0.0);
        w.ap_history[2] = ap_at(3.0);
        w.ap_history[3] = ap_at(6.0);
        w.ap_history[4] = ap_at(9.0);
        w.ap_history[5] = ap_mean8(12.0);
        w.ap_history[6] = ap_mean8(36.0);
        return w;
    }

    [[nodiscard]] bool empty() const noexcept { return !flat_ && f107_daily_.empty(); }

    /// True for the windowless from_constant(WeatherConditions) form, which
    /// is valid at every epoch (window() throws for it).
    [[nodiscard]] bool is_flat() const noexcept { return flat_; }

    /// First and last daily stamp. Throws std::out_of_range for an empty or
    /// windowless history.
    [[nodiscard]] TimeWindow window() const {
        if (flat_) {
            throw std::out_of_range(
                "SpaceWeatherHistory::window: flat (windowless) history is valid "
                "at every epoch and has no window");
        }
        if (empty()) {
            throw std::out_of_range("SpaceWeatherHistory::window: history is empty");
        }
        return TimeWindow{times_daily_.front(), times_daily_.back()};
    }

    [[nodiscard]] const std::vector<Epoch>& times_daily() const noexcept { return times_daily_; }
    [[nodiscard]] const std::vector<double>& f107_daily() const noexcept { return f107_daily_; }
    [[nodiscard]] const std::vector<double>& f107_81day_avg() const noexcept { return f107_81day_avg_; }
    [[nodiscard]] const std::vector<Epoch>& times_ap() const noexcept { return times_ap_; }
    [[nodiscard]] const std::vector<double>& ap_values() const noexcept { return ap_values_; }

private:
    void validate() const {
        if (times_daily_.size() != f107_daily_.size()) {
            throw std::invalid_argument(
                "SpaceWeatherHistory: times_daily / f107_daily size mismatch");
        }
        if (times_ap_.size() != ap_values_.size()) {
            throw std::invalid_argument(
                "SpaceWeatherHistory: times_ap / ap_values size mismatch");
        }
        for (std::size_t i = 1; i < times_daily_.size(); ++i) {
            if (!(times_daily_[i] > times_daily_[i - 1])) {
                throw std::invalid_argument(
                    "SpaceWeatherHistory: times_daily must be strictly increasing");
            }
        }
        for (std::size_t i = 1; i < times_ap_.size(); ++i) {
            if (!(times_ap_[i] > times_ap_[i - 1])) {
                throw std::invalid_argument(
                    "SpaceWeatherHistory: times_ap must be strictly increasing");
            }
        }
    }

    static std::vector<double> compute_81day_average(const std::vector<double>& daily) {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(daily.size());
        std::vector<double> avg(daily.size(), 0.0);
        const std::ptrdiff_t half_window = 40;  // 40 days each side -> 81-day window
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const std::ptrdiff_t lo = std::max<std::ptrdiff_t>(0, i - half_window);
            const std::ptrdiff_t hi = std::min<std::ptrdiff_t>(n - 1, i + half_window);
            double sum = 0.0;
            for (std::ptrdiff_t k = lo; k <= hi; ++k) sum += daily[k];
            avg[static_cast<std::size_t>(i)] = sum / static_cast<double>(hi - lo + 1);
        }
        return avg;
    }

    static double nearest_prior(const std::vector<Epoch>& times,
                                const std::vector<double>& values,
                                const Epoch& target) {
        // upper_bound returns the first element > target; previous is the last <= target.
        auto it = std::upper_bound(times.begin(), times.end(), target);
        if (it == times.begin()) {
            return values.front();  // before series start: clamp
        }
        if (it == times.end()) {
            return values.back();   // after series end: clamp
        }
        const std::size_t idx = static_cast<std::size_t>(std::distance(times.begin(), it)) - 1;
        return values[idx];
    }

    static double linear_interp(const std::vector<Epoch>& times,
                                const std::vector<double>& values,
                                const Epoch& target) {
        auto it = std::upper_bound(times.begin(), times.end(), target);
        if (it == times.begin()) return values.front();
        if (it == times.end()) return values.back();
        const std::size_t hi = static_cast<std::size_t>(std::distance(times.begin(), it));
        const std::size_t lo = hi - 1;
        const double dt = times[hi] - times[lo];
        if (dt <= 0.0) return values[lo];
        const double frac = (target - times[lo]) / dt;
        return values[lo] + frac * (values[hi] - values[lo]);
    }

    // Mean of the 3-hourly Ap values in the `hours_window` hours ending at
    // `epoch`. Throws DataCoverageError for a history built without any Ap
    // values: there is no honest value to return, and a quiet default would
    // bias the density low in a storm.
    double mean_recent_ap(const Epoch& epoch, double hours_window) const {
        if (times_ap_.empty()) {
            const auto [y, mo, d, h, mi, sec] = epoch.to_calendar(TimeScale::TT);
            char when[40];
            std::snprintf(when, sizeof when, "%04d-%02d-%02d %02d:%02d (TT)",
                          y, mo, d, h, mi);
            (void)sec;
            throw DataCoverageError(
                std::string("SpaceWeatherHistory: no 3-hourly Ap values, so the "
                            "geomagnetic index at ") + when +
                " cannot be formed. NRLMSIS density depends on it (without the "
                "measured Ap, the density is far too low in a geomagnetic storm). "
                "Build the history with its Ap series (SpaceWeatherData::history(), "
                "Python SpaceWeatherHistory.from_loaded), or use "
                "SpaceWeatherHistory::from_constant(WeatherConditions) to choose "
                "constant conditions on purpose.");
        }
        // Half-open (epoch - window, epoch]: exactly the 8 three-hourly bins
        // ending with the one containing epoch (9 if both ends were closed and
        // epoch fell on a bin boundary).
        const Epoch start = epoch - hours_window * 3600.0;
        // Binary search for the bins in (start, epoch]: the live drag model
        // queries a history of the whole file (~200k bins) on every force
        // evaluation, where a linear scan cost ~300 us per call.
        const auto lo = std::upper_bound(times_ap_.begin(), times_ap_.end(), start);
        const auto hi = std::upper_bound(lo, times_ap_.end(), epoch);
        double sum = 0.0;
        std::size_t count = 0;
        for (auto it = lo; it != hi; ++it) {
            sum += ap_values_[static_cast<std::size_t>(it - times_ap_.begin())];
            ++count;
        }
        if (count == 0) return nearest_prior(times_ap_, ap_values_, epoch);
        return sum / static_cast<double>(count);
    }

    std::vector<Epoch>  times_daily_;
    std::vector<double> f107_daily_;
    std::vector<double> f107_81day_avg_;  // same length as f107_daily_
    std::vector<Epoch>  times_ap_;
    std::vector<double> ap_values_;

    // Windowless-constant mode (from_constant(WeatherConditions)): at()
    // returns flat_value_ for every epoch; the series vectors stay empty.
    bool flat_ = false;
    WeatherConditions flat_value_{};
};

}  // namespace atmosphere
}  // namespace ballistic
