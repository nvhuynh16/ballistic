// =============================================================================
// Epoch Class Python Bindings
// =============================================================================

#include <nanobind/nanobind.h>
#include <ballistic/core/errors.hpp>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <ballistic/time/epoch.hpp>
#include <sstream>
#include <iomanip>

namespace nb = nanobind;

void bind_epoch(nb::module_& m) {
    using namespace ballistic;
    using namespace ballistic::time;

    // TimeScale enum
    nb::enum_<TimeScale>(m, "TimeScale",
        "Time scales an Epoch can be read or written in. They differ by offsets of up "
        "to about a minute, which matter: one second is about 7.5 km of travel in low Earth "
        "orbit.")
        .value("UTC", TimeScale::UTC,
               "Coordinated Universal Time: civil clock time, kept within a second of the "
               "Earth's rotation by leap seconds")
        .value("UT1", TimeScale::UT1,
               "Universal Time 1: time defined by the Earth's actual rotation angle (needs "
               "Earth orientation data)")
        .value("TAI", TimeScale::TAI, "International Atomic Time: uniform, no leap seconds")
        .value("TT", TimeScale::TT, "Terrestrial Time (TT = TAI + 32.184 s)")
        .value("TDB", TimeScale::TDB,
               "Barycentric Dynamical Time: the time argument of planetary ephemerides "
               "(within 2 ms of TT)")
        .value("GPS", TimeScale::GPS, "GPS Time (GPS = TAI - 19 s)");

    // Epoch class
    nb::class_<Epoch>(m, "Epoch",
        R"doc(
Epoch - an instant in time, readable in any of the time scales used for
orbits:

- UTC: Coordinated Universal Time (civil time, with leap seconds)
- TT: Terrestrial Time (TT = TAI + 32.184 s)
- TAI: International Atomic Time
- TDB: Barycentric Dynamical Time (planetary ephemerides)
- UT1: Universal Time 1 (follows the Earth's rotation)
- GPS: GPS Time (GPS = TAI - 19 s)

Internally the instant is stored in TT as a whole Julian day number plus
the seconds into that day, so labels and differences between epochs are
exact to about 1e-11 s. A Julian Date (JD) is a continuous count of days
used in astronomy; the Modified Julian Date (MJD) is JD - 2400000.5.

Examples:
    # Create from UTC calendar
    epoch = Epoch.from_utc(2024, 1, 15, 12, 0, 0)

    # Create from Julian Date
    epoch = Epoch.from_jd(2460325.0, TimeScale.TT)

    # Create from Unix timestamp
    epoch = Epoch.from_unix(1705320000.0)

    # Arithmetic
    epoch2 = epoch + 3600.0  # Add 1 hour (3600 seconds)
    dt = epoch2 - epoch      # Get difference in seconds

    # Access different time scales
    utc_jd = epoch.utc_jd
    tt_jd = epoch.tt_jd
    unix_ts = epoch.unix_timestamp
)doc")
        // Constructors
        .def(nb::init<>(), "The J2000.0 epoch (2000-01-01 12:00:00 TT)")
        .def(nb::init<double, TimeScale>(),
             nb::arg("jd"), nb::arg("scale"),
             "From a Julian Date `jd` in the time scale `scale` (both required).")
        .def("__init__", [](Epoch*, double) {
            throw RemovedApiError(
                "Epoch(jd) without a time scale was removed: it silently meant TT, and a "
                "UTC Julian date read as TT is 69 s (~500 km at LEO) off. Use "
                "Epoch.from_jd(jd, TimeScale.UTC) (or TT, TAI, GPS, TDB, UT1), or a calendar "
                "constructor such as Epoch.from_utc(...) (see docs/MIGRATION.md).");
        }, nb::arg("jd"),
            "Removed: raises RemovedApiError. Pass the time scale.")

        // Named constructors (static methods)
        .def_static("from_utc",
            [](int year, int month, int day, int hour, int minute, double second) {
                return Epoch::from_utc(year, month, day, hour, minute, second);
            },
            nb::arg("year"), nb::arg("month"), nb::arg("day"),
            nb::arg("hour") = 0, nb::arg("minute") = 0, nb::arg("second") = 0.0,
            R"doc(
Create an Epoch from a UTC calendar date and time.

Args:
    year: Year (e.g., 2024)
    month: Month (1-12)
    day: Day of month (1-31)
    hour: Hour (0-23, default 0)
    minute: Minute (0-59, default 0)
    second: Second (0-60, default 0.0, may be fractional; 60 only during a
        leap second)

Returns:
    Epoch: the given UTC instant

Raises:
    ballistic.errors.DataCoverageError: a date before 1961, when UTC is not
        defined; use from_tt or from_tai.

Example:
    epoch = Epoch.from_utc(2024, 1, 15, 12, 30, 45.5)
)doc")

        .def_static("from_jd", &Epoch::from_jd,
            nb::arg("jd"), nb::arg("scale"),
            "From a Julian Date `jd` (days) in the time scale `scale` (both required).")
        .def_static("from_jd", [](double) -> Epoch {
            throw RemovedApiError(
                "Epoch.from_jd(jd) without a time scale was removed: it silently meant TT. "
                "Pass the scale: Epoch.from_jd(jd, TimeScale.UTC) (see docs/MIGRATION.md).");
        }, nb::arg("jd"),
            "Removed: raises RemovedApiError. Pass the time scale.")

        .def_static("from_mjd", &Epoch::from_mjd,
            nb::arg("mjd"), nb::arg("scale"),
            "From a Modified Julian Date `mjd` (days; MJD = JD - 2400000.5) in the time "
            "scale `scale` (both required).")
        .def_static("from_mjd", [](double) -> Epoch {
            throw RemovedApiError(
                "Epoch.from_mjd(mjd) without a time scale was removed: it silently meant TT. "
                "Pass the scale: Epoch.from_mjd(mjd, TimeScale.UTC) (see docs/MIGRATION.md).");
        }, nb::arg("mjd"),
            "Removed: raises RemovedApiError. Pass the time scale.")

        .def_static("from_tt", &Epoch::from_tt,
            nb::arg("year"), nb::arg("month"), nb::arg("day"),
            nb::arg("hour") = 0, nb::arg("minute") = 0, nb::arg("second") = 0.0,
            "Calendar date and time in TT (Terrestrial Time).")
        .def_static("from_tai", &Epoch::from_tai,
            nb::arg("year"), nb::arg("month"), nb::arg("day"),
            nb::arg("hour") = 0, nb::arg("minute") = 0, nb::arg("second") = 0.0,
            "Calendar date and time in TAI (International Atomic Time).")
        .def_static("from_gps", &Epoch::from_gps,
            nb::arg("year"), nb::arg("month"), nb::arg("day"),
            nb::arg("hour") = 0, nb::arg("minute") = 0, nb::arg("second") = 0.0,
            "Calendar date and time in GPS time, as satellite-navigation (GNSS) products and "
            "receivers stamp it (e.g. SP3 files marked '%c GPS'). Reading GPS time as UTC "
            "is 18 s off (about 130 km along a low Earth orbit).")
        .def_static("from_tdb", &Epoch::from_tdb,
            nb::arg("year"), nb::arg("month"), nb::arg("day"),
            nb::arg("hour") = 0, nb::arg("minute") = 0, nb::arg("second") = 0.0,
            "Calendar date and time in TDB (Barycentric Dynamical Time).")
        .def_static("from_gps_week", &Epoch::from_gps_week,
            nb::arg("week"), nb::arg("seconds_of_week"),
            "From a GPS week number (counted continuously from the start of GPS time, "
            "without the 1024-week rollover) and the seconds into that week.")
        .def("to_gps_week", &Epoch::to_gps_week,
            "Tuple (GPS week number, seconds into the week).")
        .def_static("from_datetime", [](nb::handle dt) {
            if (dt.attr("utcoffset")().is_none()) {
                throw ConfigError(
                    "Epoch.from_datetime needs a timezone-aware datetime: a naive one is "
                    "ambiguous (local time? UTC?). Use datetime(..., tzinfo=timezone.utc) or "
                    "dt.replace(tzinfo=timezone.utc) if it is UTC.");
            }
            nb::object utc = nb::module_::import_("datetime").attr("timezone").attr("utc");
            nb::object u = dt.attr("astimezone")(utc);
            const double sec = nb::cast<double>(u.attr("second")) +
                               nb::cast<double>(u.attr("microsecond")) * 1e-6;
            return Epoch::from_utc(nb::cast<int>(u.attr("year")), nb::cast<int>(u.attr("month")),
                                   nb::cast<int>(u.attr("day")), nb::cast<int>(u.attr("hour")),
                                   nb::cast<int>(u.attr("minute")), sec);
        }, nb::arg("dt"),
            "From a timezone-aware datetime.datetime (converted to UTC). Raises ConfigError "
            "for a naive datetime (no tzinfo), which is ambiguous.")

        .def_static("from_unix", &Epoch::from_unix,
            nb::arg("unix_time"),
            "From a Unix timestamp: seconds since 1970-01-01 00:00:00 UTC, not counting leap "
            "seconds (as time.time() returns)")

        .def_static("j2000", &Epoch::j2000,
            "The J2000.0 reference epoch (2000-01-01 12:00:00 TT)")

        // Conversion properties
        .def_prop_ro("utc_jd", &Epoch::utc_jd,
            "Julian Date in UTC (float, days; about 40 microsecond resolution - use Epoch "
            "differences, not JD differences, for time intervals)")
        .def_static("from_tt_split", &Epoch::from_tt_split, nb::arg("tt_day"), nb::arg("tt_seconds"),
            "Exact construction from the internal representation: the TT Julian day number "
            "(int; Julian days start at noon) and the seconds into it.")
        .def_prop_ro("tt_day", &Epoch::tt_day,
            "TT Julian day number (int; part of the exact internal representation)")
        .def_prop_ro("tt_seconds", &Epoch::tt_seconds,
            "Seconds into tt_day (part of the exact internal representation)")
        .def_prop_ro("tt_jd", &Epoch::tt_jd,
            "TT (Terrestrial Time) Julian Date")
        .def_prop_ro("tai_jd", &Epoch::tai_jd,
            "TAI (International Atomic Time) Julian Date")
        .def_prop_ro("tdb_jd", &Epoch::tdb_jd,
            "TDB (Barycentric Dynamical Time) Julian Date")
        .def_prop_ro("ut1_jd", &Epoch::ut1_jd,
            "UT1 Julian Date (needs Earth orientation data, EOP; raises DataCoverageError "
            "without it unless set_eop_policy('assume_zero'))")
        .def_prop_ro("gps_jd", &Epoch::gps_jd,
            "GPS Time Julian Date")
        .def_prop_ro("unix_timestamp", &Epoch::unix_timestamp,
            "Unix timestamp (seconds since 1970-01-01 00:00:00 UTC, leap seconds not counted)")
        .def_prop_ro("seconds_since_j2000", &Epoch::seconds_since_j2000,
            "Seconds since J2000.0, in TT")
        .def_prop_ro("julian_centuries_tt", &Epoch::julian_centuries_tt,
            "Julian centuries (36525 days) since J2000.0, in TT")
        .def_prop_ro("day_of_year", &Epoch::day_of_year,
            "Fractional day of the year in UTC, counted from 0 (1 January 00:00 = 0.0, noon "
            "= 0.5), as atmosphere models take it")
        .def_prop_ro("year", &Epoch::year,
            "Calendar year (int, UTC)")

        // Generic conversion methods
        .def("jd", &Epoch::jd, nb::arg("scale"),
            "Julian Date (float, days) in the time scale `scale`")
        .def("mjd", &Epoch::mjd, nb::arg("scale"),
            "Modified Julian Date (float, days) in the time scale `scale`")

        // Calendar conversion
        .def("to_calendar", &Epoch::to_calendar,
            nb::arg("scale") = TimeScale::UTC,
            R"doc(
Calendar date and time in a chosen time scale.

Args:
    scale: Time scale for output (default UTC)

Returns:
    tuple: (year, month, day, hour, minute, second)
)doc")

        // DUT1 (UT1-UTC) offset
        .def("set_dut1", &Epoch::set_dut1, nb::arg("dut1_seconds"),
            "Set a fallback UT1-UTC [s] for this epoch, used by the conversions to UT1 only "
            "when no Earth orientation table is loaded (which requires "
            "set_eop_policy('assume_zero')). A loaded table always takes precedence.")
        .def_prop_ro("dut1", &Epoch::dut1,
            "The fallback UT1-UTC [s] set by set_dut1 (0 by default)")

        // Arithmetic operators
        .def("__add__", [](const Epoch& e, double seconds) { return e + seconds; },
            nb::arg("seconds"),
            "A new Epoch `seconds` later")
        .def("__radd__", [](const Epoch& e, double seconds) { return e + seconds; },
            nb::arg("seconds"))
        .def("__sub__", [](const Epoch& e, double seconds) { return e - seconds; },
            nb::arg("seconds"),
            "A new Epoch `seconds` earlier")
        .def("__sub__", [](const Epoch& e, const Epoch& other) { return e - other; },
            nb::arg("other"),
            "Time from `other` to this epoch, in seconds (float)")
        .def("__iadd__", [](Epoch& e, double seconds) { e += seconds; return e; },
            nb::arg("seconds"))
        .def("__isub__", [](Epoch& e, double seconds) { e -= seconds; return e; },
            nb::arg("seconds"))

        // Comparison operators
        .def("__lt__", &Epoch::operator<)
        .def("__le__", &Epoch::operator<=)
        .def("__gt__", &Epoch::operator>)
        .def("__ge__", &Epoch::operator>=)
        .def("__eq__", &Epoch::operator==)
        .def("__ne__", &Epoch::operator!=)

        // String representation
        .def("__repr__", [](const Epoch& e) {
            auto [year, month, day, hour, minute, second] = e.to_calendar(TimeScale::UTC);
            std::ostringstream oss;
            oss << "Epoch(" << year << "-"
                << std::setfill('0') << std::setw(2) << month << "-"
                << std::setw(2) << day << " "
                << std::setw(2) << hour << ":"
                << std::setw(2) << minute << ":"
                << std::setfill('0') << std::setw(6) << std::fixed << std::setprecision(3) << second
                << " UTC)";
            return oss.str();
        })

        .def("__str__", [](const Epoch& e) {
            auto [year, month, day, hour, minute, second] = e.to_calendar(TimeScale::UTC);
            std::ostringstream oss;
            oss << year << "-"
                << std::setfill('0') << std::setw(2) << month << "-"
                << std::setw(2) << day << "T"
                << std::setw(2) << hour << ":"
                << std::setw(2) << minute << ":"
                << std::setfill('0') << std::setw(6) << std::fixed << std::setprecision(3) << second
                << "Z";
            return oss.str();
        });
}
