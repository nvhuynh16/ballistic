// Tests the library's SP3 reader (validation/sp3.hpp). SP3 is the standard
// text format for precise satellite orbits: an epoch line ("*") followed by a
// position ("P", km) and optionally a velocity ("V", dm/s) line per satellite.
// All inputs are synthetic strings built below (a satellite on a circle, so
// interpolated values have closed forms). Checked: time systems (GPS, UTC,
// BDT = BeiDou time), exact time offsets and unit conversion, absent values,
// malformed lines (raise with the line number), data gaps, velocities
// derived from positions, frame labels, and thread safety (re-entrancy).

#include "../test_utils.hpp"

#include <ballistic/core/errors.hpp>
#include <ballistic/validation/sp3.hpp>

#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
namespace sp3 = ballistic::validation::sp3;

namespace {

// A minimal SP3-d header for one satellite "L99"; `tsys` is the %c field.
std::string header(const std::string& tsys = "GPS", char pv = 'V', const std::string& frame = "IGS20",
                   double interval = 30.0, char version = 'd') {
    char l1[128], l2[128];
    std::snprintf(l1, sizeof l1, "#%c%c2026  1  1  0  0  0.00000000      10 ORBIT %-5s FIT  TST",
                  version, pv, frame.c_str());
    std::snprintf(l2, sizeof l2, "## 2399 345600.00000000 %14.8f 61041 0.0000000000000", interval);
    std::string h = std::string(l1) + "\n" + l2 + "\n";
    h += "+    1   L99  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n";
    h += "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n";
    if (version == 'c' || version == 'd')
        h += "%c L  cc " + tsys + " ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n";
    h += "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n";
    h += "/* test file\n";
    return h;
}

// A circle of radius r [km] at angular rate w [rad/s]: P (+ V) records every
// `dt` seconds starting 2026-01-01 00:00:00 in the file's time system.  guard-exempt(PUB-DATE): data
std::string records(int n, double dt = 30.0, bool vel = true, int skip_from = -1, int skip_to = -1,
                    double r = 6878.0, double w = 1.1e-3) {
    std::string out;
    for (int i = 0; i < n; ++i) {
        if (i >= skip_from && i < skip_to) continue;
        const double t = i * dt;
        char e[96], p[96], v[96];
        const int sec = static_cast<int>(t);
        std::snprintf(e, sizeof e, "*  2026  1  1 %2d %2d %11.8f\n", sec / 3600, (sec / 60) % 60,
                      std::fmod(t, 60.0));
        std::snprintf(p, sizeof p, "PL99%14.6f%14.6f%14.6f%14.6f\n", r * std::cos(w * t),
                      r * std::sin(w * t), 0.0, 999999.999999);
        std::snprintf(v, sizeof v, "VL99%14.6f%14.6f%14.6f%14.6f\n", -r * w * std::sin(w * t) * 1e4,
                      r * w * std::cos(w * t) * 1e4, 0.0, 999999.999999);
        out += e;
        out += p;
        if (vel) out += v;
    }
    return out + "EOF\n";
}

}  // namespace

int main() {
    TestRunner runner("SP3Reader");

    runner.add_test("gps_file_is_utc_plus_leap_seconds", []() {
        const auto g = sp3::read_string(header("GPS") + records(12));
        const auto u = sp3::read_string(header("UTC") + records(12));
        // 2026: GPS - UTC = 18 s, so the same clock reading is 18 s earlier in UTC.
        // BDT = GPS - 14 s, so a BDT reading is 14 s later than the GPS one.
        TEST_ASSERT_NEAR(u.arcs[0].epoch0 - g.arcs[0].epoch0, 18.0, 1e-9);
        TEST_ASSERT(g.header.time_system == sp3::TimeSystem::GPS);
        const auto b = sp3::read_string(header("BDT") + records(12));
        TEST_ASSERT_NEAR(b.arcs[0].epoch0 - g.arcs[0].epoch0, 14.0, 1e-9);
    });

    runner.add_test("sp3a_without_time_system_line_is_gps", []() {
        const auto a = sp3::read_string(header("", 'V', "IGS20", 30.0, 'a') + records(12));
        TEST_ASSERT(a.header.time_system == sp3::TimeSystem::GPS);
    });

    runner.add_test("unknown_time_system_raises", []() {
        TEST_ASSERT_THROWS(sp3::read_string(header("XYZ") + records(12)), Error);
    });

    runner.add_test("offsets_are_exact_and_units_converted", []() {
        const auto f = sp3::read_string(header() + records(12));
        const auto& a = f.arc("L99");
        TEST_ASSERT_EQ(a.size(), 12u);
        TEST_ASSERT_EQ(a.offset_ns[11], 330'000'000'000LL);
        TEST_ASSERT_NEAR(a.position_m[0].x(), 6878e3, 1e-6);
        TEST_ASSERT_NEAR(a.velocity_m_s[0].y(), 6878e3 * 1.1e-3, 1e-3);   // dm/s -> m/s
        TEST_ASSERT(a.has_velocity());
    });

    runner.add_test("clock_running_into_z_is_read", []() {
        // A large clock value leaves no space after z, so splitting on
        // whitespace would see three fields and drop the record; the fields
        // must be read by their fixed columns.
        std::string body = "*  2026  1  1  0  0  0.00000000\n"
                           "PL99  -5698.883987  -3747.874343    441.732191-102852.949562\n"
                           "VL99   1000.000000   2000.000000   3000.000000-102852.949562\n";
        const auto f = sp3::read_string(header() + body + "EOF\n");
        TEST_ASSERT_NEAR(f.arc("L99").position_m[0].z(), 441732.191, 1e-6);
    });

    runner.add_test("absent_positions_are_not_records", []() {
        std::string body = records(12);
        const std::string hit = "PL99";
        const auto at = body.find(hit, body.find(hit) + 1);   // second P record
        body.replace(at + 4, 42, "      0.000000      0.000000      0.000000");
        const auto f = sp3::read_string(header() + body);
        TEST_ASSERT_EQ(f.arc("L99").size(), 11u);
        TEST_ASSERT_EQ(f.arc("L99").missing_positions, 1);
    });

    runner.add_test("malformed_lines_raise_with_line_number", []() {
        try {
            (void)sp3::read_string(header() + "*  2026  1  1  0  0  0.00000000\nPL99  garbage\nEOF\n");
            TEST_ASSERT_MSG(false, "expected Error");
        } catch (const Error& e) {
            TEST_ASSERT(std::string(e.what()).find("line 9") != std::string::npos);
        }
        TEST_ASSERT_THROWS(sp3::read_string(header() + "*  2026  1  x  0  0  0.0\nEOF\n"), Error);
    });

    runner.add_test("record_before_first_epoch_raises", []() {
        const std::string body = "PL99      1.000000      2.000000      3.000000\n" + records(3);
        TEST_ASSERT_THROWS(sp3::read_string(header() + body), Error);
    });

    runner.add_test("interpolation_is_range_checked_and_gap_aware", []() {
        // 40 records with a 5-record gap at 15..19 (150 s > 1.5 x 30 s).
        const auto f = sp3::read_string(header() + records(40, 30.0, true, 15, 20));
        const auto& a = f.arc("L99");
        TEST_ASSERT_EQ(a.segments().size(), 2u);
        const double r = 6878e3, w = 1.1e-3, t = 105.0;
        const auto s = a.interpolate(t);
        TEST_ASSERT_NEAR(s.position_m.x(), r * std::cos(w * t), 1e-3);
        TEST_ASSERT_NEAR(s.velocity_m_s.y(), r * w * std::cos(w * t), 1e-5);
        TEST_ASSERT_THROWS(a.interpolate(-1.0), DataCoverageError);
        TEST_ASSERT_THROWS(a.interpolate(a.span_s() + 1.0), DataCoverageError);
        TEST_ASSERT_THROWS(a.interpolate(500.0), DataCoverageError);   // inside the gap
    });

    // Velocities from a 9-point Lagrange derivative: the 4 records at each end
    // have a one-sided (less accurate) window and must be flagged so.
    runner.add_test("derived_velocities_flag_one_sided_windows", []() {
        const auto f = sp3::read_string(header("GPS", 'P') + records(20, 30.0, false));
        const auto d = f.arc("L99").derived_velocities(9);
        TEST_ASSERT(!d.centered[0] && !d.centered[3]);
        TEST_ASSERT(d.centered[4] && d.centered[15]);
        TEST_ASSERT(!d.centered[16] && !d.centered[19]);
        TEST_ASSERT_NEAR(d.velocity_m_s[10].y(), 6878e3 * 1.1e-3 * std::cos(1.1e-3 * 300.0), 1e-4);
    });

    runner.add_test("frame_labels", []() {
        TEST_ASSERT(sp3::classify_frame("J2000").inertial);
        TEST_ASSERT(sp3::classify_frame("IGb14").itrf == coords::ITRFRealization::ITRF2014);
        TEST_ASSERT(!sp3::classify_frame("CTS").itrf.has_value());
        TEST_ASSERT_THROWS(sp3::classify_frame("WGS84"), FrameError);
        const auto f = sp3::read_string(header("GPS", 'V', "CTS") + records(3));
        TEST_ASSERT_EQ(f.header.frame, std::string("CTS"));
    });

    runner.add_test("reading_is_reentrant", []() {
        const std::string a = header() + records(200);
        const std::string b = header("UTC") + records(150, 60.0);
        std::vector<double> got(8);
        std::vector<std::thread> threads;
        for (int k = 0; k < 8; ++k) {
            threads.emplace_back([&, k] {
                const auto f = sp3::read_string(k % 2 ? b : a);
                got[k] = f.arcs[0].position_m.back().y() + f.arcs[0].span_s();
            });
        }
        for (auto& t : threads) t.join();
        for (int k = 2; k < 8; ++k) TEST_ASSERT_EQ(got[k], got[k % 2]);
    });

    runner.run();
    return print_final_summary();
}
