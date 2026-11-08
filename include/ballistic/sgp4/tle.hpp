#pragma once

// =============================================================================
// Two-line element sets (TLE)
// =============================================================================
//
// A TLE is a set of MEAN orbital elements fitted by the US Space Force to
// tracking data, in the form the SGP4 model expects. They are not osculating
// elements (the instantaneous orbit) and mean nothing to any other
// propagator; see sgp4.hpp. Format: the
// NORAD/Space-Track "two-line element set", columns as documented by
// CelesTrak ("NORAD Two-Line Element Set Format", T.S. Kelso) and
// Spacetrack Report #3 (Hoots & Roehrich 1980), with the Alpha-5 catalog
// numbers (a leading letter A-Z without I and O for 100000-339999).
//
// The parser is strict: a TLE is a fixed-column format and a shifted column
// changes a value silently (a misplaced digit in the eccentricity or mean
// motion is still a number). It refuses lines that are not 69 characters,
// a wrong line number, a bad checksum, a catalog number that differs between
// the lines, a non-blank separator column, and malformed fields; each error
// names the line and the column.
//
// The epoch is UTC (the convention of the element sets), converted to an
// Epoch through the calendar date so leap seconds inside the year are
// honored.
// =============================================================================

#include "../core/errors.hpp"
#include "../time/epoch.hpp"
#include "../time/julian_date.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>

namespace ballistic::sgp4 {

/// A parsed two-line element set. Angles in degrees as written in the TLE.
struct Tle {
    std::string line1, line2;             // the two lines as given (trailing blanks removed)
    int catalog_number = 0;               // Alpha-5 decoded (A0001 -> 100001)
    char classification = 'U';            // U unclassified, C classified, S secret
    std::string international_designator; // launch year+number+piece, e.g. "98067A" (may be blank)
    int epoch_year = 0;                   // four digits (57-99 -> 19xx, 00-56 -> 20xx)
    double epoch_day = 0.0;               // day of year + fraction, 1.0 = 1 Jan 00:00 UTC
    double ndot_rev_day2 = 0.0;           // first derivative of mean motion / 2 (not used by SGP4)
    double nddot_rev_day3 = 0.0;          // second derivative / 6 (not used by SGP4)
    double bstar_per_er = 0.0;            // B*, SGP4's drag term, 1 / Earth radii
    int ephemeris_type = 0;               // 0 in distributed element sets
    int element_set_number = 0;
    double inclination_deg = 0.0;         // orbit plane tilt to the equator
    double raan_deg = 0.0;                // right ascension of the ascending node
    double eccentricity = 0.0;
    double argp_deg = 0.0;                // argument of perigee
    double mean_anomaly_deg = 0.0;
    double mean_motion_rev_day = 0.0;     // Kozai mean motion (revolutions per day)
    int revolution_number = 0;            // orbit count at epoch

    /// The element-set epoch (UTC). Throws DataCoverageError for epoch years 1957-1960,
    /// before UTC existed.
    [[nodiscard]] Epoch epoch() const {
        const int doy = static_cast<int>(std::floor(epoch_day));
        const auto c = time::jdn_to_civil(time::civil_to_jdn(epoch_year, 1, 1) + doy - 1);
        return Epoch::from_utc(c.year, c.month, c.day) + (epoch_day - doy) * 86400.0;
    }

    /// Days since 1949 December 31 00:00 UTC, the time argument of SGP4's
    /// lunar-solar and Earth-rotation terms.
    [[nodiscard]] double epoch_days_since_1950() const noexcept {
        return static_cast<double>(time::civil_to_jdn(epoch_year, 1, 1) - 2433283LL) + epoch_day;
    }

    /// Ballistic coefficient beta = m / (Cd A) [kg/m^2] implied by B* (mass m, drag
    /// coefficient Cd, area A): B* = (Cd A / m) rho0 / 2 with rho0 = 0.15696615 kg/m^2/ER
    /// (ER = Earth radius; Spacetrack Report #3), so m / (Cd A) = 1 / (12.741621 B*).
    /// B* is a fitted term that also absorbs other model errors and can be zero (result
    /// infinite) or negative (result negative); callers decide what to do with it.
    [[nodiscard]] double beta_from_bstar_kg_m2() const noexcept {
        return 1.0 / (12.741621 * bstar_per_er);
    }
};

namespace detail_tle {

[[noreturn]] inline void fail(int line, int col, const std::string& what) {
    throw InputPlausibilityError("TLE line " + std::to_string(line) + ", column " +
                                 std::to_string(col) + ": " + what);
}

inline void require_blank(std::string_view s, int line, int col) {
    if (s[col - 1] != ' ') fail(line, col, "expected a blank separator, found '" +
                                           std::string(1, s[col - 1]) + "' (shifted columns?)");
}

// Columns [a, b], 1-based inclusive, as written in the format tables.
inline std::string_view cols(std::string_view s, int a, int b) { return s.substr(a - 1, b - a + 1); }

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

inline int parse_int(std::string_view f, int line, int col, bool allow_blank = false) {
    std::size_t i = 0;
    while (i < f.size() && f[i] == ' ') ++i;
    if (i == f.size()) {
        if (allow_blank) return 0;
        fail(line, col, "empty integer field");
    }
    int v = 0;
    for (; i < f.size(); ++i) {
        if (!is_digit(f[i])) fail(line, col, "'" + std::string(f) + "' is not an integer");
        v = 10 * v + (f[i] - '0');
    }
    return v;
}

// A decimal number that fills its field, e.g. " 51.6416", "-.00002182".
inline double parse_decimal(std::string_view f, int line, int col) {
    std::string s(f);
    std::size_t a = s.find_first_not_of(' ');
    if (a == std::string::npos) fail(line, col, "empty numeric field");
    s = s.substr(a);
    // Leading sign, digits, one point, digits; no exponent, no inner blanks.
    std::size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    int points = 0, digits = 0;
    for (; i < s.size(); ++i) {
        if (s[i] == '.') ++points;
        else if (is_digit(s[i])) ++digits;
        else fail(line, col, "'" + std::string(f) + "' is not a decimal number");
    }
    if (points > 1 || digits == 0) fail(line, col, "'" + std::string(f) + "' is not a decimal number");
    return std::strtod(s.c_str(), nullptr);
}

// The packed "assumed decimal point" form: [sign]ddddd[sign]d, e.g.
// "-11606-4" = -0.11606e-4, " 00000+0" = 0.
inline double parse_packed(std::string_view f, int line, int col) {
    // 8 columns: sign, five mantissa digits, exponent sign, exponent digit.
    const char s0 = f[0];
    if (s0 != ' ' && s0 != '+' && s0 != '-') fail(line, col, "bad sign in '" + std::string(f) + "'");
    double mant = 0.0;
    for (int k = 1; k <= 5; ++k) {
        if (!is_digit(f[k])) fail(line, col, "'" + std::string(f) + "' is not a packed exponent field");
        mant = 10.0 * mant + (f[k] - '0');
    }
    const char es = f[6];
    if (es != '+' && es != '-' && es != ' ') fail(line, col, "bad exponent sign in '" + std::string(f) + "'");
    if (!is_digit(f[7])) fail(line, col, "bad exponent digit in '" + std::string(f) + "'");
    const int ex = (es == '-' ? -1 : 1) * (f[7] - '0');
    return (s0 == '-' ? -1.0 : 1.0) * mant * 1e-5 * std::pow(10.0, ex);
}

inline int parse_catalog(std::string_view f, int line, int col) {
    const char c = f[0];
    if (c >= 'A' && c <= 'Z') {
        if (c == 'I' || c == 'O') fail(line, col, "Alpha-5 catalog numbers do not use I or O");
        int lead = c - 'A' + 10;
        if (c > 'I') --lead;
        if (c > 'O') --lead;
        return lead * 10000 + parse_int(f.substr(1), line, col + 1);
    }
    return parse_int(f, line, col);
}

}  // namespace detail_tle

/// Modulo-10 checksum of columns 1-68 of a TLE line: digits count their value, '-' counts
/// 1, everything else 0. Column 69 of a valid line holds this digit.
[[nodiscard]] inline int tle_checksum(std::string_view line) noexcept {
    int sum = 0;
    for (std::size_t i = 0; i < 68 && i < line.size(); ++i) {
        if (detail_tle::is_digit(line[i])) sum += line[i] - '0';
        else if (line[i] == '-') sum += 1;
    }
    return sum % 10;
}

/// Parse the two lines of a TLE (trailing blanks and CR/LF are ignored). Throws
/// InputPlausibilityError naming the line and column for any violation listed at the top
/// of this file, and for values outside their range (inclination [0, 180] deg, angles
/// [0, 360) deg, mean motion (0, 20) rev/day, epoch day [1, 367)).
[[nodiscard]] inline Tle parse_tle(std::string_view line1, std::string_view line2) {
    using namespace detail_tle;
    auto trim = [](std::string_view s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ')) s.remove_suffix(1);
        return s;
    };
    const std::string_view l1 = trim(line1), l2 = trim(line2);
    for (int n : {1, 2}) {
        const std::string_view l = n == 1 ? l1 : l2;
        if (l.size() != 69)
            throw InputPlausibilityError("TLE line " + std::to_string(n) + " has " +
                                         std::to_string(l.size()) + " characters; a TLE line has 69");
        if (l[0] != static_cast<char>('0' + n) || l[1] != ' ')
            fail(n, 1, "line " + std::to_string(n) + " must start with '" + std::to_string(n) + " '");
        if (!is_digit(l[68])) fail(n, 69, "the checksum must be a digit");
        const int want = l[68] - '0', got = tle_checksum(l);
        if (want != got)
            fail(n, 69, "checksum is " + std::to_string(want) + " but the line sums to " +
                            std::to_string(got) + " (a corrupted or edited line)");
    }
    for (int c : {9, 18, 33, 44, 53, 62, 64}) require_blank(l1, 1, c);
    for (int c : {8, 17, 26, 34, 43, 52}) require_blank(l2, 2, c);

    Tle t;
    t.line1 = std::string(l1);
    t.line2 = std::string(l2);
    t.catalog_number = parse_catalog(cols(l1, 3, 7), 1, 3);
    if (parse_catalog(cols(l2, 3, 7), 2, 3) != t.catalog_number)
        fail(2, 3, "catalog number differs from line 1 (lines from two element sets?)");
    t.classification = l1[7];
    t.international_designator = std::string(cols(l1, 10, 17));
    while (!t.international_designator.empty() && t.international_designator.back() == ' ')
        t.international_designator.pop_back();
    const int yy = parse_int(cols(l1, 19, 20), 1, 19);
    t.epoch_year = yy < 57 ? 2000 + yy : 1900 + yy;
    t.epoch_day = parse_decimal(cols(l1, 21, 32), 1, 21);
    if (!(t.epoch_day >= 1.0 && t.epoch_day < 367.0)) fail(1, 21, "epoch day of year out of range");
    t.ndot_rev_day2 = parse_decimal(cols(l1, 34, 43), 1, 34);
    t.nddot_rev_day3 = parse_packed(cols(l1, 45, 52), 1, 45);
    t.bstar_per_er = parse_packed(cols(l1, 54, 61), 1, 54);
    t.ephemeris_type = parse_int(cols(l1, 63, 63), 1, 63, /*allow_blank=*/true);
    t.element_set_number = parse_int(cols(l1, 65, 68), 1, 65, /*allow_blank=*/true);

    t.inclination_deg = parse_decimal(cols(l2, 9, 16), 2, 9);
    t.raan_deg = parse_decimal(cols(l2, 18, 25), 2, 18);
    {
        const std::string_view e = cols(l2, 27, 33);
        for (char c : e)
            if (!is_digit(c)) fail(2, 27, "eccentricity '" + std::string(e) + "' must be 7 digits (assumed decimal point)");
        t.eccentricity = parse_int(e, 2, 27) * 1e-7;
    }
    t.argp_deg = parse_decimal(cols(l2, 35, 42), 2, 35);
    t.mean_anomaly_deg = parse_decimal(cols(l2, 44, 51), 2, 44);
    t.mean_motion_rev_day = parse_decimal(cols(l2, 53, 63), 2, 53);
    t.revolution_number = parse_int(cols(l2, 64, 68), 2, 64, /*allow_blank=*/true);

    if (!(t.inclination_deg >= 0.0 && t.inclination_deg <= 180.0)) fail(2, 9, "inclination outside [0, 180] deg");
    if (!(t.raan_deg >= 0.0 && t.raan_deg < 360.0)) fail(2, 18, "right ascension outside [0, 360) deg");
    if (!(t.argp_deg >= 0.0 && t.argp_deg < 360.0)) fail(2, 35, "argument of perigee outside [0, 360) deg");
    if (!(t.mean_anomaly_deg >= 0.0 && t.mean_anomaly_deg < 360.0)) fail(2, 44, "mean anomaly outside [0, 360) deg");
    if (!(t.mean_motion_rev_day > 0.0 && t.mean_motion_rev_day < 20.0))
        fail(2, 53, "mean motion outside (0, 20) rev/day");
    return t;
}

}  // namespace ballistic::sgp4
