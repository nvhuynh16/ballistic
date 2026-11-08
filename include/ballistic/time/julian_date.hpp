#pragma once

#include "../core/constants.hpp"
#include <cmath>
#include <tuple>

// Calendar <-> Julian date helpers. A Julian date (JD) is a continuous count
// of days (with fraction) since noon of 4713 BC January 1 (Julian calendar);
// each Julian day starts at noon. The modified Julian date is
// MJD = JD - 2400000.5, whose days start at midnight. J2000.0 is JD 2451545.0
// (2000-01-01 12:00). These functions do not know about time scales: the JD
// is in whatever scale the inputs are.

namespace ballistic {
namespace time {

/// Time of day (`hour`, `minute`, `second`) as a fraction of a day, e.g. 12:00:00 -> 0.5.
[[nodiscard]] inline double time_to_fractional_day(
    int hour, int minute, double second) noexcept {
    return (hour + minute / constants::MINUTES_PER_HOUR +
            second / constants::SECONDS_PER_HOUR) / constants::HOURS_PER_DAY;
}

/// Julian date of a Gregorian calendar date and time of day. A double: resolution
/// ~40 microseconds at current dates (Epoch keeps exact time). Algorithm from Meeus,
/// "Astronomical Algorithms", Chapter 7; this form applies the Gregorian calendar to every
/// date, so it matches historical (Julian-calendar) dates only from 1582-10-15 on.
[[nodiscard]] inline double calendar_to_jd(
    int year, int month, int day,
    int hour = 0, int minute = 0, double second = 0.0) noexcept
{
    // Handle January and February as months 13 and 14 of previous year
    int y = year;
    int m = month;
    if (m <= 2) {
        y -= 1;
        m += 12;
    }

    // Calculate A and B terms for Gregorian calendar correction
    int A = y / 100;
    int B = 2 - A + A / 4;

    // Julian date at 0h of the date (the -1524.5 puts it at midnight, not noon)
    double jdn = std::floor(365.25 * (y + 4716))
               + std::floor(30.6001 * (m + 1))
               + day + B - 1524.5;

    double fraction = time_to_fractional_day(hour, minute, second);

    return jdn + fraction;
}

/// Calendar date and time of day (year, month, day, hour, minute, second) of a Julian date
/// (Meeus, Chapter 7; Julian calendar before 1582-10-15). Rounding of the double can give
/// 59.99999 s instead of the next minute; Epoch::to_calendar does not.
[[nodiscard]] inline std::tuple<int, int, int, int, int, double> jd_to_calendar(double jd) noexcept {
    // Algorithm from Meeus, "Astronomical Algorithms", Chapter 7
    double Z = std::floor(jd + 0.5);
    double F = (jd + 0.5) - Z;

    double A;
    if (Z < 2299161) {
        A = Z;
    } else {
        double alpha = std::floor((Z - 1867216.25) / 36524.25);
        A = Z + 1 + alpha - std::floor(alpha / 4);
    }

    double B = A + 1524;
    double C = std::floor((B - 122.1) / 365.25);
    double D = std::floor(365.25 * C);
    double E = std::floor((B - D) / 30.6001);

    double day_frac = B - D - std::floor(30.6001 * E) + F;
    int day = static_cast<int>(day_frac);
    double time_frac = (day_frac - day) * constants::HOURS_PER_DAY;

    int month = (E < 14) ? static_cast<int>(E - 1) : static_cast<int>(E - 13);
    int year = (month > 2) ? static_cast<int>(C - 4716) : static_cast<int>(C - 4715);

    int hour = static_cast<int>(time_frac);
    double min_frac = (time_frac - hour) * constants::MINUTES_PER_HOUR;
    int minute = static_cast<int>(min_frac);
    double second = (min_frac - minute) * constants::SECONDS_PER_MINUTE;

    return {year, month, day, hour, minute, second};
}

/// Exact integer conversion of a Gregorian civil date to its Julian day number (JDN: the
/// Julian date at NOON of that date, an integer). Fliegel & Van Flandern (1968), Comm. ACM
/// 11:657. Integer arithmetic only, so Epoch can represent calendar input exactly.
[[nodiscard]] constexpr long long civil_to_jdn(int year, int month, int day) noexcept {
    const long long a = (14 - month) / 12;
    const long long y = static_cast<long long>(year) + 4800 - a;
    const long long m = month + 12 * a - 3;
    return day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;
}

/// A Gregorian calendar date (month 1-12, day 1-31).
struct CivilDate { int year; int month; int day; };

/// The inverse of civil_to_jdn: the Gregorian date whose noon has Julian day number `jdn`.
/// After Richards (2013), Explanatory Supplement 3rd ed., 15.11.3.
[[nodiscard]] constexpr CivilDate jdn_to_civil(long long jdn) noexcept {
    const long long a = jdn + 32044;
    const long long b = (4 * a + 3) / 146097;
    const long long c = a - 146097 * b / 4;
    const long long d = (4 * c + 3) / 1461;
    const long long e = c - 1461 * d / 4;
    const long long m = (5 * e + 2) / 153;
    return {static_cast<int>(100 * b + d - 4800 + m / 10),
            static_cast<int>(m + 3 - 12 * (m / 10)),
            static_cast<int>(e - (153 * m + 2) / 5 + 1)};
}

/// Julian date -> modified Julian date (subtracts 2400000.5).
[[nodiscard]] inline double jd_to_mjd(double jd) noexcept {
    return jd - constants::MJD_OFFSET;
}

/// Modified Julian date -> Julian date (adds 2400000.5).
[[nodiscard]] inline double mjd_to_jd(double mjd) noexcept {
    return mjd + constants::MJD_OFFSET;
}

/// Julian centuries (36525-day units) since J2000.0 of a Julian date in TT (Terrestrial
/// Time), the time argument of precession/nutation series.
[[nodiscard]] inline double jd_to_julian_centuries_tt(double jd_tt) noexcept {
    return (jd_tt - constants::J2000_JD) / constants::DAYS_PER_JULIAN_CENTURY;
}

/// Same for a Julian date in TDB (Barycentric Dynamical Time); identical arithmetic.
[[nodiscard]] inline double jd_to_julian_centuries_tdb(double jd_tdb) noexcept {
    return (jd_tdb - constants::J2000_JD) / constants::DAYS_PER_JULIAN_CENTURY;
}

/// Seconds since J2000.0 of a Julian date (same time scale as the input).
[[nodiscard]] inline double jd_to_seconds_since_j2000(double jd) noexcept {
    return (jd - constants::J2000_JD) * constants::SECONDS_PER_DAY;
}

/// Unix timestamp [s since 1970-01-01 00:00:00 UTC] -> UTC Julian date (leap seconds not counted).
[[nodiscard]] inline double unix_to_jd(double unix_time) noexcept {
    return constants::UNIX_EPOCH_JD + unix_time / constants::SECONDS_PER_DAY;
}

/// UTC Julian date -> Unix timestamp [s] (leap seconds not counted).
[[nodiscard]] inline double jd_to_unix(double jd) noexcept {
    return (jd - constants::UNIX_EPOCH_JD) * constants::SECONDS_PER_DAY;
}

/// True if `year` is a leap year in the Gregorian calendar.
[[nodiscard]] inline bool is_leap_year(int year) noexcept {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

/// Day of the year, 1 January = 1 (1-366). `month` must be 1-12 (not checked).
[[nodiscard]] inline int day_of_year(int year, int month, int day) noexcept {
    static constexpr int days_before_month[] = {
        0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334
    };
    int doy = days_before_month[month - 1] + day;
    if (month > 2 && is_leap_year(year)) {
        doy += 1;
    }
    return doy;
}

/// Day of the year counted from 0 including the time of day (1 January 00:00 = 0.0), the
/// form atmosphere models take.
[[nodiscard]] inline double fractional_day_of_year(
    int year, int month, int day,
    int hour, int minute, double second) noexcept
{
    int doy = day_of_year(year, month, day);
    double fraction = time_to_fractional_day(hour, minute, second);
    return doy + fraction - 1.0;  // 0-indexed for some atmospheric models
}

} // namespace time
} // namespace ballistic
