#pragma once

#include <cstdint>
#include <utility>

#include "../core/constants.hpp"
#include "../core/diagnostics.hpp"
#include "../core/errors.hpp"
#include "julian_date.hpp"
#include "leap_seconds.hpp"
#include "eop_data.hpp"
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <mutex>
#include <functional>
#include <filesystem>

namespace ballistic {
namespace time {

/// The time scales an Epoch can be read or written in. They tick at (almost) the same rate
/// but are offset from each other by seconds to a minute, which at orbital speeds
/// (~7.5 km/s in low Earth orbit, LEO) is tens to hundreds of kilometers of position.
enum class TimeScale {
    UTC,    ///< Coordinated Universal Time: civil time, kept near Earth rotation by leap seconds
    UT1,    ///< Universal Time: the Earth's actual rotation angle; UT1 - UTC is measured, < 0.9 s
    TAI,    ///< International Atomic Time: continuous atomic time, no leap seconds
    TT,     ///< Terrestrial Time: TT = TAI + 32.184 s exactly; the library's internal scale
    TDB,    ///< Barycentric Dynamical Time: TT plus a periodic term of < 2 ms (planet ephemerides)
    GPS     ///< GPS Time: GPS = TAI - 19 s exactly; used by GNSS receivers and orbit products
};


/// An instant in time, convertible to and from every TimeScale.
///
/// A Julian date (JD) counts days continuously; each Julian day starts at noon. The modified
/// Julian date is MJD = JD - 2400000.5 (days start at midnight). J2000.0 is the reference
/// instant 2000-01-01 12:00:00 TT (JD 2451545.0).
///
/// Storage: the instant in TT as an integer Julian day number plus seconds into that day, in
/// [0, 86400). Calendar input is therefore exact (from_utc builds the day number with integer
/// arithmetic) and `Epoch - Epoch` is exact to ~1e-11 s. A single double holding the Julian
/// date would round every instant to a ~40 microsecond grid, which corrupts anything that
/// differentiates or interpolates over short spans (trap EPOCH-JD-JITTER in
/// docs/trap_census.md). tt_jd() / jd(scale) still return a double Julian date for evaluating
/// physical models, where 40 microseconds does not matter; never subtract two of them.
///
/// Conversions that involve UTC need the leap-second table and throw DataCoverageError for
/// UTC instants before 1961 (UTC did not exist). Conversions that involve UT1 use the Earth
/// orientation parameter (EOP) table and throw DataCoverageError when no table is loaded
/// (unless EopPolicy::AssumeZero is set) or the date is more than 30 days outside it.
class Epoch {
public:
    /// J2000.0 (2000-01-01 12:00:00 TT).
    Epoch() noexcept : day_(J2000_DAY), sec_(0.0) {}

    /// From a Julian date `jd` [days] in time scale `scale`. The scale is required: a UTC
    /// Julian date read as TT is 69 s (~500 km at LEO) off (trap EPOCH-TT-DEFAULT). A single
    /// double JD resolves only ~40 microseconds at current dates; prefer the calendar
    /// constructors or from_tt_split. Throws DataCoverageError as described on the class.
    Epoch(double jd, TimeScale scale) {
        const double d = std::floor(jd);
        set_from(static_cast<std::int64_t>(d), (jd - d) * constants::SECONDS_PER_DAY, scale);
    }

    /// From a calendar date and time of day read in `scale` (exact). `month` 1-12, `day`
    /// 1-31, `second` may carry a fraction. Throws DataCoverageError for UTC before 1961 or
    /// UT1 without Earth orientation data (see the class comment).
    [[nodiscard]] static Epoch from_calendar(
        TimeScale scale, int year, int month, int day,
        int hour = 0, int minute = 0, double second = 0.0)
    {
        // Midnight of the civil date is JD (JDN - 1) + 0.5 day.
        Epoch e(Uninitialized{});
        e.set_from(civil_to_jdn(year, month, day) - 1,
                   43200.0 + hour * 3600.0 + minute * 60.0 + second, scale);
        return e;
    }

    /// Calendar constructors, one per time scale; same arguments and errors as
    /// from_calendar. GPS time is what GNSS (satellite navigation) receivers and orbit
    /// products stamp (SP3, the standard precise-orbit text format, says '%c GPS'); reading
    /// it as UTC is 18 s (~130 km at LEO) off.
    [[nodiscard]] static Epoch from_utc(int year, int month, int day,
                                        int hour = 0, int minute = 0, double second = 0.0) {
        return from_calendar(TimeScale::UTC, year, month, day, hour, minute, second);
    }
    [[nodiscard]] static Epoch from_tt(int year, int month, int day,
                                       int hour = 0, int minute = 0, double second = 0.0) {
        return from_calendar(TimeScale::TT, year, month, day, hour, minute, second);
    }
    [[nodiscard]] static Epoch from_tai(int year, int month, int day,
                                        int hour = 0, int minute = 0, double second = 0.0) {
        return from_calendar(TimeScale::TAI, year, month, day, hour, minute, second);
    }
    [[nodiscard]] static Epoch from_gps(int year, int month, int day,
                                        int hour = 0, int minute = 0, double second = 0.0) {
        return from_calendar(TimeScale::GPS, year, month, day, hour, minute, second);
    }
    [[nodiscard]] static Epoch from_tdb(int year, int month, int day,
                                        int hour = 0, int minute = 0, double second = 0.0) {
        return from_calendar(TimeScale::TDB, year, month, day, hour, minute, second);
    }

    /// From a GPS week number (counted continuously from 1980-01-06, without the 1024-week
    /// rollover of the broadcast format) and `seconds_of_week` [s] into that week.
    [[nodiscard]] static Epoch from_gps_week(int week, double seconds_of_week) {
        Epoch e(Uninitialized{});
        e.set_from(civil_to_jdn(1980, 1, 6) - 1,
                   43200.0 + week * 604800.0 + seconds_of_week, TimeScale::GPS);
        return e;
    }

    /// Same as Epoch(jd, scale).
    [[nodiscard]] static Epoch from_jd(double jd, TimeScale scale) {
        return Epoch(jd, scale);
    }

    /// From a modified Julian date `mjd` [days] in `scale`. Resolution is better than
    /// from_jd (the MJD is a smaller number), but still a double; same errors as from_jd.
    [[nodiscard]] static Epoch from_mjd(double mjd, TimeScale scale) {
        // JD = MJD + 2400000.5: split the MJD, not the (larger) JD.
        const double f = std::floor(mjd);
        Epoch e(Uninitialized{});
        e.set_from(MJD_DAY0 + static_cast<std::int64_t>(f),
                   (0.5 + (mjd - f)) * constants::SECONDS_PER_DAY, scale);
        return e;
    }

    /// From a Unix timestamp [s since 1970-01-01 00:00:00 UTC]. Like Unix time itself, the
    /// count ignores leap seconds (every day is 86400 s).
    [[nodiscard]] static Epoch from_unix(double unix_time) {
        Epoch e(Uninitialized{});
        e.set_from(UNIX_DAY0, 43200.0 + unix_time, TimeScale::UTC);
        return e;
    }

    /// Exact construction from the internal representation: `tt_day` is the TT Julian day
    /// number (the integer part of the Julian date; days start at noon) and `tt_seconds` the
    /// seconds into that day (values outside [0, 86400) are carried into the day).
    [[nodiscard]] static Epoch from_tt_split(std::int64_t tt_day, double tt_seconds) {
        Epoch e(Uninitialized{});
        e.set_from(tt_day, tt_seconds, TimeScale::TT);
        return e;
    }

    /// The reference instant J2000.0 (2000-01-01 12:00:00 TT).
    [[nodiscard]] static Epoch j2000() noexcept { return Epoch(); }

    /// The current time from the system clock (as accurate as the clock).
    [[nodiscard]] static Epoch now();

    /// This instant as a Julian date [days] in `scale`. A double with ~40 microsecond
    /// resolution: for evaluating models, not for differences (use `Epoch - Epoch`).
    /// Throws DataCoverageError as described on the class.
    [[nodiscard]] double jd(TimeScale scale) const {
        const auto [d, s] = split_in(scale);
        return static_cast<double>(d) + s / constants::SECONDS_PER_DAY;
    }

    /// This instant as a modified Julian date [days] in `scale`; same caveats as jd().
    [[nodiscard]] double mjd(TimeScale scale) const {
        const auto [d, s] = split_in(scale);
        return static_cast<double>(d - MJD_DAY0) + (s / constants::SECONDS_PER_DAY - 0.5);
    }

    /// Shorthands for jd(TimeScale::X); same caveats as jd().
    [[nodiscard]] double utc_jd() const { return jd(TimeScale::UTC); }
    [[nodiscard]] double tt_jd() const noexcept {
        return static_cast<double>(day_) + sec_ / constants::SECONDS_PER_DAY;
    }
    [[nodiscard]] double tai_jd() const { return jd(TimeScale::TAI); }
    [[nodiscard]] double tdb_jd() const { return jd(TimeScale::TDB); }
    [[nodiscard]] double ut1_jd() const { return jd(TimeScale::UT1); }
    [[nodiscard]] double gps_jd() const { return jd(TimeScale::GPS); }

    /// The GPS week number (continuous from 1980-01-06) and the seconds into that week [s].
    [[nodiscard]] std::pair<int, double> to_gps_week() const {
        const auto [d, s] = split_in(TimeScale::GPS);
        const double since = static_cast<double>(d - (civil_to_jdn(1980, 1, 6) - 1)) *
                             constants::SECONDS_PER_DAY + (s - 43200.0);
        const int week = static_cast<int>(std::floor(since / 604800.0));
        return {week, since - week * 604800.0};
    }

    /// Seconds [s] to ADD to this instant's TT reading to get its reading in `scale` (UT1
    /// through the EOP lookup). Lets a caller build a precise time argument from tt_day() /
    /// tt_seconds() without subtracting two Julian dates. Throws DataCoverageError as
    /// described on the class.
    [[nodiscard]] double seconds_from_tt(TimeScale scale) const { return offset_from_tt(scale); }

    /// The internal representation: TT Julian day number (days start at noon) and the
    /// seconds into it, in [0, 86400).
    [[nodiscard]] std::int64_t tt_day() const noexcept { return day_; }
    [[nodiscard]] double tt_seconds() const noexcept { return sec_; }

    /// Seconds since J2000.0, in TT [s]. One double of ~1e9 s, so it resolves ~1e-7 s at
    /// current dates; use `Epoch - Epoch` for differences.
    [[nodiscard]] double seconds_since_j2000() const noexcept {
        return static_cast<double>(day_ - J2000_DAY) * constants::SECONDS_PER_DAY + sec_;
    }

    /// Julian centuries (36525-day units) since J2000.0, in TT: the time argument of most
    /// astronomical series (precession, nutation, sidereal time).
    [[nodiscard]] double julian_centuries_tt() const noexcept {
        return seconds_since_j2000() /
               (constants::SECONDS_PER_DAY * constants::DAYS_PER_JULIAN_CENTURY);
    }

    /// Unix timestamp [s since 1970-01-01 00:00:00 UTC, leap seconds not counted].
    [[nodiscard]] double unix_timestamp() const {
        const auto [d, s] = split_in(TimeScale::UTC);
        return static_cast<double>(d - UNIX_DAY0) * constants::SECONDS_PER_DAY + (s - 43200.0);
    }

    /// Calendar date and time of day in `scale`: (year, month, day, hour, minute, second).
    /// Computed from the exact representation, so a whole minute prints as 0 s, not
    /// 59.99999 s. Throws DataCoverageError as described on the class.
    [[nodiscard]] std::tuple<int, int, int, int, int, double> to_calendar(
        TimeScale scale = TimeScale::UTC) const
    {
        auto [d, s] = split_in(scale);
        // Seconds after the civil midnight that precedes noon of JD day d.
        double civil = s + 43200.0;
        long long jdn = d;
        if (civil >= constants::SECONDS_PER_DAY) {
            civil -= constants::SECONDS_PER_DAY;
            jdn += 1;
        }
        const CivilDate c = jdn_to_civil(jdn);
        int hour = static_cast<int>(civil / 3600.0);
        if (hour > 23) hour = 23;
        civil -= hour * 3600.0;
        int minute = static_cast<int>(civil / 60.0);
        if (minute > 59) minute = 59;
        const double second = civil - minute * 60.0;
        return {c.year, c.month, c.day, hour, minute, second};
    }

    /// Fractional day of the year in UTC, counted from 0 (1 January 00:00 = 0.0, noon = 0.5);
    /// the form atmosphere models take.
    [[nodiscard]] double day_of_year() const {
        auto [year, month, day, hour, minute, second] = to_calendar(TimeScale::UTC);
        return fractional_day_of_year(year, month, day, hour, minute, second);
    }

    /// The calendar year in UTC.
    [[nodiscard]] int year() const {
        auto [y, m, d, h, min, sec] = to_calendar(TimeScale::UTC);
        return y;
    }

    /// Shift by `seconds` [s] (negative = earlier), or the difference between two epochs
    /// [s]. Exact to ~1e-11 s. These are SI seconds (TT), so adding 86400 s across a UTC
    /// leap second lands one second before the same UTC clock time the next day.
    [[nodiscard]] Epoch operator+(double seconds) const noexcept {
        Epoch result = *this;
        result.sec_ += seconds;
        result.normalize();
        return result;
    }

    [[nodiscard]] Epoch operator-(double seconds) const noexcept {
        return *this + (-seconds);
    }

    [[nodiscard]] double operator-(const Epoch& other) const noexcept {
        return static_cast<double>(day_ - other.day_) * constants::SECONDS_PER_DAY +
               (sec_ - other.sec_);
    }

    Epoch& operator+=(double seconds) noexcept {
        sec_ += seconds;
        normalize();
        return *this;
    }

    Epoch& operator-=(double seconds) noexcept {
        return *this += (-seconds);
    }

    /// Ordering on the exact internal representation.
    [[nodiscard]] bool operator<(const Epoch& other) const noexcept {
        return day_ < other.day_ || (day_ == other.day_ && sec_ < other.sec_);
    }
    [[nodiscard]] bool operator>(const Epoch& other) const noexcept { return other < *this; }
    [[nodiscard]] bool operator<=(const Epoch& other) const noexcept { return !(other < *this); }
    [[nodiscard]] bool operator>=(const Epoch& other) const noexcept { return !(*this < other); }
    /// Exact equality of the internal representation. Two epochs built by different
    /// arithmetic can still differ by ~1e-11 s; compare std::abs(a - b) < tol to allow
    /// for that.
    [[nodiscard]] bool operator==(const Epoch& other) const noexcept {
        return day_ == other.day_ && sec_ == other.sec_;
    }
    [[nodiscard]] bool operator!=(const Epoch& other) const noexcept { return !(*this == other); }

    /// Set a fallback UT1 - UTC [s] for this epoch, used only by the conversions to UT1 and
    /// only when no EOP table is loaded (which requires EopPolicy::AssumeZero; otherwise the
    /// conversion throws). A loaded table always takes precedence.
    void set_dut1(double dut1_seconds) noexcept {
        dut1_ = dut1_seconds;
    }

    /// The fallback UT1 - UTC [s] set by set_dut1 (0 by default).
    [[nodiscard]] double dut1() const noexcept {
        return dut1_;
    }

private:
    struct Uninitialized {};
    explicit Epoch(Uninitialized) noexcept : day_(J2000_DAY), sec_(0.0) {}

    static constexpr std::int64_t J2000_DAY = 2451545;   // JD 2451545.0 = 2000-01-01 12:00 TT
    static constexpr std::int64_t MJD_DAY0 = 2400000;    // MJD 0 = JD 2400000.5
    static constexpr std::int64_t UNIX_DAY0 = 2440587;   // Unix 0 = JD 2440587.5 UTC

    std::int64_t day_;  // TT Julian day number (the JD integer, days start at noon)
    double sec_;        // seconds into that day, [0, 86400)
    double dut1_ = 0.0; // UT1-UTC fallback in seconds, used ONLY when no EOP
                        // table is loaded (a loaded table takes precedence). Only
                        // Epoch's conversions to UT1 read it. It keeps the
                        // value it was set to on purpose: a no-data fallback, not a cache.

    void normalize() noexcept {
        if (sec_ >= constants::SECONDS_PER_DAY || sec_ < 0.0) {
            const double days = std::floor(sec_ / constants::SECONDS_PER_DAY);
            day_ += static_cast<std::int64_t>(days);
            sec_ -= days * constants::SECONDS_PER_DAY;
            if (sec_ >= constants::SECONDS_PER_DAY) { sec_ -= constants::SECONDS_PER_DAY; ++day_; }
            else if (sec_ < 0.0) { sec_ += constants::SECONDS_PER_DAY; --day_; }
        }
    }

    // (day, seconds) in `scale` -> stored TT.
    void set_from(std::int64_t day, double sec, TimeScale scale) {
        day_ = day;
        sec_ = sec;
        normalize();
        sec_ += offset_to_tt(scale, static_cast<double>(day_) + sec_ / constants::SECONDS_PER_DAY);
        normalize();
    }

    // Stored TT -> (day, seconds) in `scale`.
    [[nodiscard]] std::pair<std::int64_t, double> split_in(TimeScale scale) const {
        Epoch e = *this;
        e.sec_ += offset_from_tt(scale);
        e.normalize();
        return {e.day_, e.sec_};
    }

    // TDB - TT periodic correction in seconds: the dominant (Earth's orbital
    // eccentricity) terms of the IAU SOFA dtdb series. Used in both directions
    // (TDB -> TT and TT -> TDB). T is Julian centuries from J2000.0; T(TDB) and
    // T(TT) differ by under 2 ms, which changes the correction by far less than
    // a microsecond, so either is acceptable input.
    [[nodiscard]] static double tdb_tt_correction_seconds(double T) noexcept {
        const double g_rad = (357.53 + 35999.05 * T) * constants::DEG_TO_RAD;
        return 0.001657 * std::sin(g_rad) + 0.000014 * std::sin(2.0 * g_rad);
    }

    // Seconds to ADD to a time in `scale` (near Julian date `jd` in that
    // scale) to get TT.
    [[nodiscard]] static double offset_to_tt(TimeScale scale, double jd) {
        const auto& leap_table = get_leap_second_table();
        switch (scale) {
            case TimeScale::TT:
                return 0.0;
            case TimeScale::TAI:
                return constants::TT_TAI_OFFSET;
            case TimeScale::UTC:
                return leap_table.tai_utc(jd) + constants::TT_TAI_OFFSET;
            case TimeScale::GPS:
                return -constants::GPS_TAI_OFFSET + constants::TT_TAI_OFFSET;
            case TimeScale::TDB:
                return -tdb_tt_correction_seconds(jd_to_julian_centuries_tdb(jd));
            case TimeScale::UT1: {
                // UT1 -> UTC -> TT with the EOP table's UT1-UTC (IERS
                // Conventions 2010, Technical Note 36, Chapter 5).
                const double mjd = jd_to_mjd(jd);
                // The EOP lookup's policy applies (edge value held <= 30 days
                // past the table, then DataCoverageError; no table: EopPolicy).
                const double dut1_value = EOPData::instance().lookup(mjd).ut1_utc_s;
                const double utc_jd = jd - dut1_value / constants::SECONDS_PER_DAY;
                return -dut1_value + leap_table.tai_utc(utc_jd) + constants::TT_TAI_OFFSET;
            }
            default:
                throw std::invalid_argument("Unknown time scale");
        }
    }

    // Seconds to ADD to TT to get `scale`.
    [[nodiscard]] double offset_from_tt(TimeScale scale) const {
        const auto& leap_table = get_leap_second_table();
        const double tt = tt_jd();
        switch (scale) {
            case TimeScale::TT:
                return 0.0;
            case TimeScale::TAI:
                return -constants::TT_TAI_OFFSET;
            case TimeScale::UTC: {
                // TAI - UTC is a function of the UTC date; solve for it by
                // fixed point (converges in two steps: the 1961-1972 rates
                // are ~1e-8 s/s, and leap-second steps settle at once).
                const double tai_jd = tt - constants::TT_TAI_OFFSET / constants::SECONDS_PER_DAY;
                double dat = leap_table.tai_utc(tai_jd);
                for (int k = 0; k < 2; ++k)
                    dat = leap_table.tai_utc(tai_jd - dat / constants::SECONDS_PER_DAY);
                return -constants::TT_TAI_OFFSET - dat;
            }
            case TimeScale::GPS:
                return -constants::TT_TAI_OFFSET + constants::GPS_TAI_OFFSET;
            case TimeScale::TDB:
                return tdb_tt_correction_seconds(jd_to_julian_centuries_tt(tt));
            case TimeScale::UT1: {
                // TT -> UTC -> UT1 with the EOP table's UT1-UTC (its hold/raise
                // policy applies); the set_dut1 value only when no table is loaded.
                const double to_utc = offset_from_tt(TimeScale::UTC);
                const double mjd = jd_to_mjd(tt + to_utc / constants::SECONDS_PER_DAY);
                // No table under EopPolicy::AssumeZero: the set_dut1 value.
                const double table = EOPData::instance().lookup(mjd).ut1_utc_s;
                const double dut1_value = EOPData::instance().is_loaded() ? table : dut1_;
                return to_utc + dut1_value;
            }
            default:
                throw std::invalid_argument("Unknown time scale");
        }
    }
};

} // namespace time

// Bring Epoch into main namespace for convenience
using time::Epoch;
using time::TimeScale;

} // namespace ballistic
