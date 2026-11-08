#pragma once

// =============================================================================
// SP3 orbit files (IGS Standard Product 3, versions a-d): the one reader
// =============================================================================
//
// SP3 is the standard text format for precise satellite orbits, defined by
// the International GNSS Service (IGS): a header, then for each epoch a line
// starting with '*' followed by one 'P' (position, km) and optionally one 'V'
// (velocity, dm/s) line per satellite. This reader converts to meters and m/s.
//
// Every SP3 consumer in the library, its tests and the validation scripts
// reads through this header (Python: ballistic.sp3.read), so they all agree
// on the points below.
//
// - Time system: the %c header field is honored (GPS, GAL (Galileo), QZS
//   (QZSS), IRN (NavIC) = GPS time; BDT (BeiDou time) = GPS - 14 s; TAI;
//   UTC; GLO (GLONASS) = UTC + 3 h). SP3-a/b, which have no %c line, are GPS
//   time by definition. Anything else raises. (GPS time runs ahead of UTC by
//   the leap seconds since 1980, 18 s at present: reading a GPS-time file as
//   UTC is an 18 s epoch error.)
// - Times: each record carries an EXACT integer-nanosecond offset from the
//   arc's first epoch, so derivatives and interpolation never see Epoch
//   rounding.
// - Missing values: a position of 0.000000 or 999999.999999 (the format's
//   "bad or absent") is not a record; it is counted in Arc::missing_positions.
//   A missing velocity in a V file is NaN.
// - Failures: a malformed epoch or record line, or a record before the first
//   epoch line, raises ballistic::Error naming the line.
// - Arcs: one per satellite per file. Interpolation is range-checked
//   (DataCoverageError outside the arc) and never crosses a gap; derived
//   velocities say whether their window was centered. Merging files is the
//   caller's decision (separate files are separate solutions).
// - No static state: reading is re-entrant.
// =============================================================================

#include "../core/errors.hpp"
#include "../core/types.hpp"
#include "../initial_state.hpp"
#include "../time/epoch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ballistic {
namespace validation {
namespace sp3 {

/// The time systems an SP3 header can declare (see the file comment).
enum class TimeSystem { GPS, GAL, QZS, IRN, BDT, TAI, UTC, GLO };

/// Converts a calendar date and time of day as written in an SP3 file whose
/// time system is `ts` (year, month, day, hour, minute, second) to an Epoch.
/// Throws ballistic::Error for an unknown time system (only reachable with an
/// out-of-range enum value).
[[nodiscard]] inline Epoch to_epoch(TimeSystem ts, int y, int mo, int d, int h, int mi, double s) {
    switch (ts) {
        case TimeSystem::GPS:
        case TimeSystem::GAL:
        case TimeSystem::QZS:
        case TimeSystem::IRN: return Epoch::from_gps(y, mo, d, h, mi, s);
        case TimeSystem::BDT: return Epoch::from_gps(y, mo, d, h, mi, s) + 14.0;
        case TimeSystem::TAI: return Epoch::from_tai(y, mo, d, h, mi, s);
        case TimeSystem::UTC: return Epoch::from_utc(y, mo, d, h, mi, s);
        case TimeSystem::GLO: return Epoch::from_utc(y, mo, d, h, mi, s) - 10800.0;
    }
    throw Error("sp3: unknown time system");
}

/// What a header's coordinate-system label means: inertial (non-rotating) or
/// Earth-fixed, and for Earth-fixed labels which ITRF realization, if stated.
struct FrameInfo {
    bool inertial = false;
    std::optional<coords::ITRFRealization> itrf;   // set for Earth-fixed labels that name a realization
};

/// Classifies an SP3 coordinate-system label. Known inertial labels: J2000,
/// J2K, EME00. Known Earth-fixed labels map to an ITRF realization (IGS05,
/// IGb08, IGS14, ITR20, ...; the IGS labels are the IGS's own alignments of
/// the ITRF, treated as that ITRF). "CTS" (GFZ's "conventional terrestrial
/// system"), "ITRF" and "ITRF2" (ESA's label cut to five characters, so 2014
/// or 2020) are Earth-fixed with no stated realization: FrameInfo::itrf is
/// empty and the caller must say which. Throws FrameError for any other label
/// rather than guessing (e.g. IGb08 is Earth-fixed, not inertial).
[[nodiscard]] inline FrameInfo classify_frame(const std::string& label) {
    using coords::ITRFRealization;
    static const std::pair<const char*, ITRFRealization> earth_fixed[] = {
        {"IGS05", ITRFRealization::ITRF2005}, {"ITR05", ITRFRealization::ITRF2005},
        {"IGS08", ITRFRealization::ITRF2008}, {"IGb08", ITRFRealization::ITRF2008},
        {"ITR08", ITRFRealization::ITRF2008},
        {"IGS14", ITRFRealization::ITRF2014}, {"IGb14", ITRFRealization::ITRF2014},
        {"ITR14", ITRFRealization::ITRF2014},
        {"IGS20", ITRFRealization::ITRF2020}, {"ITR20", ITRFRealization::ITRF2020},
        {"SLR20", ITRFRealization::ITRF2020},
    };
    for (const char* l : {"J2000", "J2K", "EME00"})
        if (label == l) return {true, std::nullopt};
    for (const auto& [l, r] : earth_fixed)
        if (label == l) return {false, r};
    // "ITRF2": ESA's label cut to five characters (2014 or 2020: ambiguous).
    if (label == "CTS" || label == "ITRF2" || label == "ITRF")
        return {false, std::nullopt};
    throw FrameError("sp3: coordinate system '" + label + "' is not one this library "
                     "knows; add it to sp3::classify_frame rather than guessing");
}

/// The SP3 header fields this reader keeps.
struct Header {
    char version = ' ';               // 'a'..'d'
    char pos_or_vel = 'P';            // 'P' positions, 'V' positions + velocities
    std::string frame;                // coordinate-system label ("IGS20", "J2000", "CTS")
    std::string orbit_type;           // e.g. "FIT", "HLM" (producer's code)
    std::string agency;               // producing agency code
    TimeSystem time_system = TimeSystem::GPS;
    double interval_s = 0.0;          // nominal epoch spacing [s]
    int declared_epochs = 0;          // epoch count stated in the first header line
    std::vector<std::string> satellites;   // satellite IDs listed in the '+' lines
};

/// All records of one satellite in one file, with interpolation helpers.
/// Times are offsets in seconds from epoch0; positions in meters and
/// velocities in m/s, in the file's frame.
struct Arc {
    std::string satellite;
    Epoch epoch0;                         // first record (converted from the file's time system)
    std::vector<std::int64_t> offset_ns;  // exact, from epoch0
    std::vector<Vector3d> position_m;
    std::vector<Vector3d> velocity_m_s;   // NaN where absent
    int missing_positions = 0;            // "bad or absent" records skipped
    double interval_s = 0.0;              // nominal spacing (header, else median)

    /// Number of records.
    [[nodiscard]] std::size_t size() const noexcept { return offset_ns.size(); }
    /// Offset of record i from epoch0 (s), from the exact nanosecond count.
    [[nodiscard]] double offset_s(std::size_t i) const { return 1e-9 * static_cast<double>(offset_ns[i]); }
    /// Time of record i.
    [[nodiscard]] Epoch epoch(std::size_t i) const { return epoch0 + offset_s(i); }
    /// Offset of the last record (s); 0 for an empty arc.
    [[nodiscard]] double span_s() const { return size() ? offset_s(size() - 1) : 0.0; }
    /// True when the file has velocity records and every one is present.
    [[nodiscard]] bool has_velocity() const {
        return !velocity_m_s.empty() &&
               std::all_of(velocity_m_s.begin(), velocity_m_s.end(),
                           [](const Vector3d& v) { return v.allFinite(); });
    }

    /// Index ranges [first, last] without a gap; a gap is a spacing larger
    /// than 1.5 x the nominal interval.
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> segments() const {
        std::vector<std::pair<std::size_t, std::size_t>> out;
        if (size() == 0) return out;
        const double limit = 1.5 * interval_s;
        std::size_t start = 0;
        for (std::size_t i = 1; i < size(); ++i) {
            if (offset_s(i) - offset_s(i - 1) > limit) {
                out.emplace_back(start, i - 1);
                start = i;
            }
        }
        out.emplace_back(start, size() - 1);
        return out;
    }

    /// Position (m) and its time derivative (m/s) at offset t [s from epoch0],
    /// by Lagrange polynomial interpolation through `order` records of one
    /// gap-free segment, centered on t where the segment allows. The velocity
    /// is the derivative of the interpolated positions, not the file's
    /// velocity records. Throws DataCoverageError outside the arc, inside a
    /// gap, or in a segment shorter than `order` (never extrapolates).
    struct Sample { Vector3d position_m; Vector3d velocity_m_s; };
    [[nodiscard]] Sample interpolate(double t, int order = 9) const {
        const auto [lo, hi] = segment_containing(t);
        const std::size_t n = hi - lo + 1;
        if (n < static_cast<std::size_t>(order)) {
            throw DataCoverageError("sp3 " + satellite + ": the data segment around t = " +
                std::to_string(t) + " s has " + std::to_string(n) + " records, fewer than the " +
                std::to_string(order) + " interpolation needs");
        }
        const std::size_t start = window_start(lo, hi, t, order);
        return lagrange(start, order, t);
    }

    /// True when the `order`-point window around offset t (s) is centered, not
    /// clipped at a segment edge. A clipped window's derivative is ~5e-4 m/s
    /// off. Throws DataCoverageError outside the arc or inside a gap.
    [[nodiscard]] bool centered(double t, int order = 9) const {
        const auto [lo, hi] = segment_containing(t);
        std::size_t k = lo;
        while (k < hi && offset_s(k + 1) <= t) ++k;
        const std::size_t half = static_cast<std::size_t>(order / 2);
        return k - lo >= half && k - half + static_cast<std::size_t>(order) - 1 <= hi;
    }

    /// Velocity (m/s) at every record, differentiated from the positions with
    /// `order`-point Lagrange windows inside each gap-free segment;
    /// `centered[i]` is false where the window is one-sided (segment edges:
    /// SP3's millimeter rounding then gives ~5e-4 m/s errors, so arcs must not
    /// start there). Segments shorter than `order` are left NaN.
    struct Derived { std::vector<Vector3d> velocity_m_s; std::vector<bool> centered; };
    [[nodiscard]] Derived derived_velocities(int order = 9) const {
        Derived out{std::vector<Vector3d>(size(), Vector3d::Constant(std::nan(""))),
                    std::vector<bool>(size(), false)};
        const int half = order / 2;
        for (const auto& [lo, hi] : segments()) {
            const std::size_t n = hi - lo + 1;
            if (n < static_cast<std::size_t>(order)) continue;   // left NaN, not centered
            for (std::size_t i = lo; i <= hi; ++i) {
                std::size_t start = (i - lo >= static_cast<std::size_t>(half)) ? i - half : lo;
                if (start + order - 1 > hi) start = hi + 1 - order;
                out.velocity_m_s[i] = lagrange(start, order, offset_s(i)).velocity_m_s;
                out.centered[i] = (i - start == static_cast<std::size_t>(half));
            }
        }
        return out;
    }

private:
    [[nodiscard]] std::pair<std::size_t, std::size_t> segment_containing(double t) const {
        if (size() == 0 || !(t >= 0.0 && t <= span_s())) {
            throw DataCoverageError("sp3 " + satellite + ": t = " + std::to_string(t) +
                " s is outside the arc (0 to " + std::to_string(span_s()) + " s)");
        }
        for (const auto& seg : segments())
            if (t >= offset_s(seg.first) && t <= offset_s(seg.second)) return seg;
        throw DataCoverageError("sp3 " + satellite + ": t = " + std::to_string(t) +
                                " s falls in a data gap");
    }

    [[nodiscard]] std::size_t window_start(std::size_t lo, std::size_t hi, double t, int order) const {
        // Center the window on t, clipped to the segment.
        std::size_t k = lo;
        while (k < hi && offset_s(k + 1) <= t) ++k;
        std::size_t start = (k - lo >= static_cast<std::size_t>(order / 2)) ? k - order / 2 : lo;
        if (start + order - 1 > hi) start = hi + 1 - order;
        return start;
    }

    [[nodiscard]] Sample lagrange(std::size_t start, int order, double t) const {
        Sample s{Vector3d::Zero(), Vector3d::Zero()};
        for (int i = 0; i < order; ++i) {
            const double ti = offset_s(start + i);
            double li = 1.0, dli = 0.0;
            for (int j = 0; j < order; ++j) {
                if (j == i) continue;
                const double tj = offset_s(start + j);
                double prod = 1.0 / (ti - tj);
                for (int k = 0; k < order; ++k) {
                    if (k == i || k == j) continue;
                    const double tk = offset_s(start + k);
                    prod *= (t - tk) / (ti - tk);
                }
                dli += prod;
                li *= (t - tj) / (ti - tj);
            }
            s.position_m += li * position_m[start + i];
            s.velocity_m_s += dli * position_m[start + i];
        }
        return s;
    }
};

/// A parsed SP3 file.
struct File {
    std::string source;          // path or name, for messages
    Header header;
    std::vector<Arc> arcs;       // in order of first record (satellites with records only)
    int missing_positions = 0;   // "bad or absent" positions skipped, all satellites

    /// The arc of `satellite` (e.g. "G01"). Throws ballistic::Error, listing
    /// the satellites present, when the file has no records for it.
    [[nodiscard]] const Arc& arc(const std::string& satellite) const {
        for (const auto& a : arcs)
            if (a.satellite == satellite) return a;
        std::string have;
        for (const auto& a : arcs) have += (have.empty() ? "" : ", ") + a.satellite;
        throw Error("sp3 " + source + ": no records for satellite '" + satellite +
                    "' (has: " + have + ")");
    }
    /// The header's coordinate system, classified; throws FrameError for an
    /// unknown label (see classify_frame).
    [[nodiscard]] FrameInfo frame() const { return classify_frame(header.frame); }
};

/// What an inertial SP3 label ("J2000", "EME00") means: producers differ, and
/// the axes of the GCRS (the modern inertial frame) and of EME2000 (the mean
/// equator and equinox of J2000) differ by the frame bias, a fixed rotation of
/// a few hundredths of an arcsecond (~0.7 m at low Earth orbit, LEO).
enum class InertialAs { Unspecified, GCRS, J2000 };

/// A checked start state for propagation, taken from the arc of `satellite`
/// in `f` at `epoch` (inside the arc). The position is interpolated
/// (`order`-point Lagrange); the velocity is the derivative of the
/// interpolated positions, because producers' velocity records follow
/// different conventions (ILRS records are not the derivative of the
/// positions). The frame comes from the header; the caller must say what the
/// file does not: `realization` for an Earth-fixed label without one,
/// `inertial_as` for an inertial label. `opt` sets the plausibility checks of
/// InitialState (initial_state.hpp).
///
/// Throws: DataCoverageError when `epoch` is outside the arc, in a gap, or
/// within order/2 records of an arc edge or gap (the derived velocity would be
/// ~5e-4 m/s off); FrameError for an unknown label, an inertial label without
/// `inertial_as`, an Earth-fixed label without a known realization, or a
/// `realization` that contradicts the header; ballistic::Error when the
/// satellite has no records; and whatever the InitialState constructors throw
/// for an implausible state (e.g. InputPlausibilityError).
[[nodiscard]] inline InitialState initial_state(
    const File& f, const std::string& satellite, const Epoch& epoch,
    std::optional<coords::ITRFRealization> realization = std::nullopt,
    InertialAs inertial_as = InertialAs::Unspecified,
    const PlausibilityOptions& opt = {}, int order = 9) {
    const Arc& a = f.arc(satellite);
    const double t = epoch - a.epoch0;
    const auto s = a.interpolate(t, order);
    if (!a.centered(t, order))
        throw DataCoverageError("sp3 " + f.source + " " + satellite + ": t = " + std::to_string(t) +
                                " s from the arc start is within " + std::to_string(order / 2) + " records of an arc "
                                "edge or gap, where the derived velocity is ~5e-4 m/s off; start "
                                "further inside the arc");
    const FrameInfo fi = f.frame();
    if (fi.inertial) {
        if (inertial_as == InertialAs::Unspecified)
            throw FrameError("sp3 " + f.source + ": the coordinate system '" + f.header.frame +
                             "' is inertial but does not say whether its axes are the GCRS or "
                             "the mean equator of J2000 (they differ by ~0.7 m at LEO); pass "
                             "inertial_as");
        return inertial_as == InertialAs::GCRS
            ? InitialState::from_gcrs(epoch, s.position_m, s.velocity_m_s, opt)
            : InitialState::from_j2000(epoch, s.position_m, s.velocity_m_s, opt);
    }
    if (fi.itrf && realization && *fi.itrf != *realization)
        throw FrameError("sp3 " + f.source + ": the header says " + f.header.frame + " (" +
                         itrf_realization_name(*fi.itrf) + ") but realization " +
                         itrf_realization_name(*realization) + " was passed");
    const auto real = fi.itrf ? fi.itrf : realization;
    if (!real)
        throw FrameError("sp3 " + f.source + ": the coordinate system '" + f.header.frame +
                         "' is Earth-fixed without a stated ITRF realization; pass realization");
    return InitialState::from_itrf(epoch, s.position_m, s.velocity_m_s, real, opt);
}

namespace detail {

[[nodiscard]] inline std::string trim(std::string s) {
    const auto a = s.find_first_not_of(' ');
    if (a == std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(' ') - a + 1);
}

[[noreturn]] inline void fail(const std::string& source, long line_no, const std::string& what,
                              const std::string& line) {
    throw Error("sp3 " + source + " line " + std::to_string(line_no) + ": " + what + ": \"" +
                line.substr(0, 80) + "\"");
}

[[nodiscard]] inline TimeSystem parse_time_system(const std::string& s, char version,
                                                  const std::string& source) {
    static const std::pair<const char*, TimeSystem> names[] = {
        {"GPS", TimeSystem::GPS}, {"GAL", TimeSystem::GAL}, {"QZS", TimeSystem::QZS},
        {"IRN", TimeSystem::IRN}, {"BDT", TimeSystem::BDT}, {"TAI", TimeSystem::TAI},
        {"UTC", TimeSystem::UTC}, {"GLO", TimeSystem::GLO},
    };
    for (const auto& [n, t] : names)
        if (s == n) return t;
    if ((version == 'a' || version == 'b') && (s.empty() || s == "ccc")) return TimeSystem::GPS;
    throw Error("sp3 " + source + ": time system '" + s + "' is not one this library reads "
                "(GPS, GAL, QZS, IRN, BDT, TAI, UTC, GLO)");
}

// x, y, z from the fixed columns 5-18, 19-32, 33-46 of a P or V record. The
// clock field (47-60) can run into z with no space ("441.732191-102852.9"):
// whitespace splitting then sees four fields as three and can drop those
// records silently (e.g. 8 records in one GRACE-FO daily file).
[[nodiscard]] inline bool xyz(const std::string& line, double& x, double& y, double& z) {
    if (line.size() < 46) return false;
    const auto num = [&](std::size_t a, double& out) {
        const std::string f = line.substr(a, 14);
        char* end = nullptr;
        out = std::strtod(f.c_str(), &end);
        if (end == f.c_str()) return false;
        while (*end == ' ') ++end;
        return *end == '\0';
    };
    return num(4, x) && num(18, y) && num(32, z);
}

// Values the format uses for "bad or absent".
[[nodiscard]] inline bool absent_position(double x, double y, double z) {
    const auto bad = [](double v) { return std::abs(v) >= 999999.0; };
    return (x == 0.0 && y == 0.0 && z == 0.0) || bad(x) || bad(y) || bad(z);
}

}  // namespace detail

/// Parses SP3 text from a stream; `source` names it in messages.
/// Throws ballistic::Error for a malformed first line, epoch line or record, a
/// record before the first epoch line, or epochs that do not increase for a
/// satellite (each message names the line), and for an unknown time system or
/// a file with no epoch records.
[[nodiscard]] inline File read_stream(std::istream& in, const std::string& source) {
    File f;
    f.source = source;
    std::string line;
    long line_no = 0;
    bool have_tsys_line = false;
    std::string tsys_label;

    struct Epoch6 { int y, mo, d, h, mi; double s; };
    std::optional<Epoch6> current;
    std::optional<Epoch> current_epoch;
    // Per satellite: index into f.arcs, created lazily.
    auto arc_for = [&](const std::string& sat) -> Arc& {
        for (auto& a : f.arcs)
            if (a.satellite == sat) return a;
        f.arcs.push_back(Arc{});
        f.arcs.back().satellite = sat;
        return f.arcs.back();
    };
    // The last P record of each satellite at the current epoch, for its V line.
    std::vector<std::pair<std::string, std::size_t>> pending;

    bool in_header = true;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (in_header) {
            if (line_no == 1) {
                if (line.size() < 3 || line[0] != '#')
                    detail::fail(source, line_no, "not an SP3 first line", line);
                f.header.version = line[1];
                f.header.pos_or_vel = line[2];
                // Fixed columns (SP3-c/d spec): epochs 33-39, data used 41-45,
                // coordinate system 47-51, orbit type 53-55, agency 57-60.
                // Whitespace splitting fails when "data used" is blank (GFZ).
                const auto col = [&](std::size_t a, std::size_t n) {
                    return a < line.size() ? detail::trim(line.substr(a, n)) : std::string();
                };
                try {
                    f.header.declared_epochs = std::stoi(col(32, 7));
                } catch (const std::logic_error&) {
                    detail::fail(source, line_no, "bad epoch count in the first header line", line);
                }
                f.header.frame = col(46, 5);
                f.header.orbit_type = col(52, 3);
                f.header.agency = col(56, 4);
                continue;
            }
            if (line.rfind("##", 0) == 0) {
                std::istringstream ss(line.substr(2));
                double week, sow;
                if (!(ss >> week >> sow >> f.header.interval_s))
                    detail::fail(source, line_no, "bad second header line", line);
                continue;
            }
            if (line.rfind("+ ", 0) == 0 || (line.size() > 1 && line[0] == '+' && line[1] != '+')) {
                for (std::size_t j = 9; j + 3 <= line.size(); j += 3) {
                    const std::string id = detail::trim(line.substr(j, 3));
                    if (!id.empty() && id != "0" && id != "00")
                        f.header.satellites.push_back(id);
                }
                continue;
            }
            if (line.rfind("%c", 0) == 0) {
                if (!have_tsys_line) {
                    have_tsys_line = true;
                    tsys_label = line.size() >= 12 ? detail::trim(line.substr(9, 3)) : "";
                }
                continue;
            }
            if (line[0] == 'P' || line[0] == 'V')
                detail::fail(source, line_no, "record before the first epoch line", line);
            if (line[0] != '*') continue;   // ++, %f, %i, /* comments
            in_header = false;
            f.header.time_system = detail::parse_time_system(
                have_tsys_line ? tsys_label : "", f.header.version, source);
        }

        if (line[0] == '*') {
            Epoch6 e{};
            std::istringstream ss(line.substr(1));
            if (!(ss >> e.y >> e.mo >> e.d >> e.h >> e.mi >> e.s))
                detail::fail(source, line_no, "bad epoch line", line);
            current = e;
            current_epoch = to_epoch(f.header.time_system, e.y, e.mo, e.d, e.h, e.mi, e.s);
            pending.clear();
        } else if (line[0] == 'P') {
            if (!current_epoch) detail::fail(source, line_no, "position record before the first epoch line", line);
            const std::string sat = detail::trim(line.substr(1, 3));
            double x, y, z;
            if (!detail::xyz(line, x, y, z)) detail::fail(source, line_no, "bad position record", line);
            Arc& a = arc_for(sat);
            if (detail::absent_position(x, y, z)) {
                ++a.missing_positions;
                ++f.missing_positions;
                continue;
            }
            if (a.offset_ns.empty()) a.epoch0 = *current_epoch;
            const double dt = *current_epoch - a.epoch0;
            const auto ns = static_cast<std::int64_t>(std::llround(dt * 1e9));
            if (!a.offset_ns.empty() && ns <= a.offset_ns.back())
                detail::fail(source, line_no, "epochs not increasing for " + sat, line);
            a.offset_ns.push_back(ns);
            a.position_m.emplace_back(x * 1e3, y * 1e3, z * 1e3);
            if (f.header.pos_or_vel == 'V')
                a.velocity_m_s.push_back(Vector3d::Constant(std::nan("")));
            pending.emplace_back(sat, a.size() - 1);
        } else if (line[0] == 'V') {
            if (!current_epoch) detail::fail(source, line_no, "velocity record before the first epoch line", line);
            const std::string sat = detail::trim(line.substr(1, 3));
            double vx, vy, vz;
            if (!detail::xyz(line, vx, vy, vz)) detail::fail(source, line_no, "bad velocity record", line);
            for (const auto& [s, idx] : pending) {
                if (s != sat) continue;
                Arc& a = arc_for(sat);
                if (a.velocity_m_s.size() <= idx) a.velocity_m_s.resize(a.size(), Vector3d::Constant(std::nan("")));
                if (!(vx == 0.0 && vy == 0.0 && vz == 0.0) && std::abs(vx) < 999999.0)
                    a.velocity_m_s[idx] = Vector3d(vx * 0.1, vy * 0.1, vz * 0.1);   // dm/s
                break;
            }
        } else if (line.rfind("EOF", 0) == 0) {
            break;
        }
        // EP / EV (correlation) lines are not read.
    }
    if (in_header) throw Error("sp3 " + source + ": no epoch records");

    // Drop satellites whose every record was absent; set nominal spacing.
    f.arcs.erase(std::remove_if(f.arcs.begin(), f.arcs.end(),
                                [](const Arc& a) { return a.size() == 0; }),
                 f.arcs.end());
    for (auto& a : f.arcs) {
        if (!a.velocity_m_s.empty()) a.velocity_m_s.resize(a.size(), Vector3d::Constant(std::nan("")));
        double nominal = f.header.interval_s;
        if (!(nominal > 0.0) && a.size() > 1) {
            std::vector<double> d;
            for (std::size_t i = 1; i < a.size(); ++i) d.push_back(a.offset_s(i) - a.offset_s(i - 1));
            std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
            nominal = d[d.size() / 2];
        }
        a.interval_s = nominal;
    }
    return f;
}

/// Parses SP3 text held in a string (see read_stream for errors).
[[nodiscard]] inline File read_string(const std::string& text, const std::string& source = "<string>") {
    std::istringstream in(text);
    return read_stream(in, source);
}

/// Reads a plain-text SP3 file (decompress .gz first; Python's
/// ballistic.sp3.read does). Throws ballistic::Error when the file cannot be
/// opened, and as read_stream for its content.
[[nodiscard]] inline File read(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw Error("sp3: cannot open " + path.string());
    return read_stream(in, path.string());
}

}  // namespace sp3
}  // namespace validation
}  // namespace ballistic
