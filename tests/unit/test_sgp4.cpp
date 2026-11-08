// TLE parser, SGP4/SDP4 and the TEME frame (sgp4/*.hpp). The full
// verification against the SGP4-VER vectors and the oracle package is
// tests/python/test_sgp4.py (it downloads both at test time).

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/initial_state.hpp>
#include <ballistic/sgp4/sgp4.hpp>
#include <ballistic/sgp4/teme.hpp>
#include <ballistic/sgp4/tle.hpp>

#include <cmath>
#include <string>

using namespace ballistic;
using namespace ballistic::sgp4;
using namespace ballistic::test;

namespace {

// Spacetrack Report #3 test object, as carried by SGP4-VER.
const std::string STR3_L1 = "1 88888U          80275.98708465  .00073094  13844-3  66816-4 0    87";
const std::string STR3_L2 = "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  1058";
// CelesTrak's documentation example (ISS, 2008).
const std::string ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const std::string ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
// SGP4-VER 28626: geosynchronous (deep space, one-day resonance).
const std::string GEO_L1 = "1 28626U 05008A   06176.46683397 -.00000205  00000-0  10000-3 0  2190";
const std::string GEO_L2 = "2 28626   0.0019 286.9433 0000335  13.7918  55.6504  1.00270176  4891";

template <class E, class F>
bool throws(F&& f, const std::string& needle = {}) {
    try { f(); } catch (const E& e) { return needle.empty() || std::string(e.what()).find(needle) != std::string::npos; }
    catch (...) { return false; }
    return false;
}

std::string with_checksum(std::string l) {
    l[68] = static_cast<char>('0' + tle_checksum(l));
    return l;
}

}  // namespace

int main() {
    time::EOPData::instance().set_policy(time::EopPolicy::AssumeZero);   // no EOP file needed
    TestRunner runner("SGP4 / TLE / TEME");

    runner.add_test("parse_fields_and_epoch", []() {
        const Tle t = parse_tle(ISS_L1, ISS_L2);
        TEST_ASSERT_EQ(t.catalog_number, 25544);
        TEST_ASSERT(t.international_designator == "98067A");
        TEST_ASSERT_EQ(t.epoch_year, 2008);
        TEST_ASSERT_NEAR(t.ndot_rev_day2, -0.00002182, 1e-15);
        TEST_ASSERT_NEAR(t.bstar_per_er, -0.11606e-4, 1e-15);
        TEST_ASSERT_NEAR(t.eccentricity, 0.0006703, 1e-15);
        TEST_ASSERT_NEAR(t.mean_motion_rev_day, 15.72125391, 1e-12);
        TEST_ASSERT_EQ(t.revolution_number, 56353);
        // Day 264.51782528 of 2008 = 20 September 12:25:40.104 UTC.
        const auto [y, mo, d, h, mi, sec] = t.epoch().to_calendar(TimeScale::UTC);
        TEST_ASSERT_EQ(y, 2008);
        TEST_ASSERT_EQ(mo, 9);
        TEST_ASSERT_EQ(d, 20);
        TEST_ASSERT_EQ(h, 12);
        TEST_ASSERT_EQ(mi, 25);
        TEST_ASSERT_NEAR(sec, 40.104192, 1e-5);
        // A leap second inside the year counts: 2015 day 200 is after 30 June's.
        const std::string l1 = with_checksum("1 25544U 98067A   15200.00000000 -.00002182  00000-0 -11606-4 0  2920");
        const auto [y2, mo2, d2, h2, mi2, sec2] = parse_tle(l1, ISS_L2).epoch().to_calendar(TimeScale::UTC);
        TEST_ASSERT_EQ(y2, 2015);
        TEST_ASSERT_EQ(mo2, 7);
        TEST_ASSERT_EQ(d2, 19);
        TEST_ASSERT_EQ(h2, 0);
        TEST_ASSERT_EQ(mi2, 0);
        TEST_ASSERT_NEAR(sec2, 0.0, 1e-6);
    });

    runner.add_test("parser_is_strict", []() {
        TEST_ASSERT((throws<InputPlausibilityError>([] { (void)parse_tle(ISS_L1, ISS_L2.substr(0, 68)); }, "69")));
        std::string bad = ISS_L1;
        bad[68] = '0';
        TEST_ASSERT((throws<InputPlausibilityError>([&] { (void)parse_tle(bad, ISS_L2); }, "checksum")));
        // One column shifted (a digit dropped, a blank appended): checksum fixed up
        // so only the column rules can catch it.
        std::string shifted = ISS_L2.substr(0, 27) + ISS_L2.substr(28, 40) + " ";
        shifted = with_checksum(shifted.substr(0, 68) + "0");
        TEST_ASSERT((throws<InputPlausibilityError>([&] { (void)parse_tle(ISS_L1, shifted); })));
        const std::string other = with_checksum("2 25545" + ISS_L2.substr(7));
        TEST_ASSERT((throws<InputPlausibilityError>([&] { (void)parse_tle(ISS_L1, other); }, "catalog")));
        TEST_ASSERT((throws<InputPlausibilityError>([&] { (void)parse_tle(ISS_L2, ISS_L1); }, "start")));
    });

    runner.add_test("alpha5_catalog_numbers", []() {
        const std::string l1 = with_checksum("1 A0001" + ISS_L1.substr(7));
        const std::string l2 = with_checksum("2 A0001" + ISS_L2.substr(7));
        TEST_ASSERT_EQ(parse_tle(l1, l2).catalog_number, 100001);
        const std::string z1 = with_checksum("1 Z9999" + ISS_L1.substr(7));
        const std::string z2 = with_checksum("2 Z9999" + ISS_L2.substr(7));
        TEST_ASSERT_EQ(parse_tle(z1, z2).catalog_number, 339999);
        const std::string i1 = with_checksum("1 I0001" + ISS_L1.substr(7));
        const std::string i2 = with_checksum("2 I0001" + ISS_L2.substr(7));
        TEST_ASSERT((throws<InputPlausibilityError>([&] { (void)parse_tle(i1, i2); }, "I or O")));
    });

    runner.add_test("spacetrack_report3_object_matches_the_verification_output", []() {
        const Sgp4 m(parse_tle(STR3_L1, STR3_L2));
        TEST_ASSERT(!m.deep_space());
        // tcppver.out (SGP4-VER), t = 0 and 120 min.
        const auto a = m.propagate_minutes(0.0);
        TEST_ASSERT(a.status == Sgp4Status::Ok);
        TEST_ASSERT((a.r_km - Vector3d(2328.96975262, -5995.22051338, 1719.97297192)).norm() < 1e-7);
        TEST_ASSERT((a.v_km_s - Vector3d(2.912073281, -0.983417956, -7.090816210)).norm() < 1e-8);
        const auto b = m.propagate_minutes(120.0);
        TEST_ASSERT((b.r_km - Vector3d(1020.69234558, 2286.56260634, -6191.55565927)).norm() < 1e-7);
        TEST_ASSERT((b.v_km_s - Vector3d(-3.746543902, 6.467532721, 1.827985678)).norm() < 1e-8);
    });

    // SGP4's velocity is analytic but NOT the exact derivative of its position
    // (truncated theory): measured 1.0-2.0 cm/s (ISS) and 4.0-5.5 cm/s (GEO).
    // Gate it at 0.1 m/s so a real velocity bug (m/s) cannot hide.
    runner.add_test("velocity_is_close_to_the_derivative_of_position", []() {
        for (const auto& [l1, l2] : {std::pair{ISS_L1, ISS_L2}, std::pair{GEO_L1, GEO_L2}}) {
            const Sgp4 m(parse_tle(l1, l2));
            for (double t : {0.0, 333.3, 2000.0}) {
                const double h = 1e-3;  // minutes
                const Vector3d fd = (m.propagate_minutes(t + h).r_km - m.propagate_minutes(t - h).r_km) / (2 * h * 60.0);
                TEST_ASSERT((fd - m.propagate_minutes(t).v_km_s).norm() < 1e-4);   // km/s
            }
        }
        TEST_ASSERT(Sgp4(parse_tle(GEO_L1, GEO_L2)).deep_space());
    });

    runner.add_test("teme_reproduces_vallado_2006_example", []() {
        // AIAA 2006-6753 / Fundamentals 4th ed.: 2004-04-06 07:51:28.386009 UTC,
        // xp -0.140682", yp 0.333309", UT1-UTC -0.4399619 s, LOD 1.5563 ms.
        const Epoch t = Epoch::from_utc(2004, 4, 6, 7, 51, 28.386009);
        time::EopSample e;
        e.xp_arcsec = -0.140682;
        e.yp_arcsec = 0.333309;
        e.ut1_utc_s = -0.4399619;
        e.lod_ms = 1.5563;
        const StateVector teme(Vector3d(5094.18016210, 6127.64465950, 6380.34453270) * 1e3,
                               Vector3d(-4.746131487, 0.785818041, 5.531931288) * 1e3);
        const StateVector itrf = teme_to_itrf(teme, t, e);
        TEST_ASSERT((itrf.position - Vector3d(-1033.4793830, 7901.2952754, 6380.3565958) * 1e3).norm() < 1e-3);
        TEST_ASSERT((itrf.velocity - Vector3d(-3.225636520, -2.872451450, 5.531924446) * 1e3).norm() < 1e-5);
        const StateVector back = itrf_to_teme(itrf, t, e);
        TEST_ASSERT((back.position - teme.position).norm() < 1e-8);
        TEST_ASSERT((back.velocity - teme.velocity).norm() < 1e-11);
    });

    runner.add_test("state_from_tle_flies_close_to_sgp4", []() {
        const Tle tle = parse_tle(ISS_L1, ISS_L2);
        const InitialState s = InitialState::from_tle(tle);
        TEST_ASSERT(s.frame() == StateFrame::ITRF);
        const double alt = coords::ecef_to_altitude(s.position());
        TEST_ASSERT(alt > 330e3 && alt < 370e3);
        // Numerical propagation for 30 min from the SGP4 state stays within a
        // few km of SGP4 itself (different force models, same orbit).
        TrajectoryConfig c;
        c.gravity_degree = 20;
        c.ocean_tide_degree = 0;
        Trajectory tr(s, c);
        const Epoch t1 = tle.epoch() + 1800.0;
        const auto p = tr.propagate(t1);
        const StateVector ref = teme_to_itrf(Sgp4(tle).teme(t1), t1);
        TEST_ASSERT((p.position - ref.position).norm() < 5e3);
        // Far past a low, decaying object's epoch SGP4 fails; the error names it.
        const Tle str3 = parse_tle(STR3_L1, STR3_L2);
        TEST_ASSERT((throws<InputPlausibilityError>(
            [&] { (void)InitialState::from_tle(str3, str3.epoch() + 365.25 * 86400.0); }, "SGP4")));
    });

    runner.run();
    return print_final_summary();
}
