// InitialState (initial_state.hpp): a start state with its frame named,
// checked for units, the ground, a suborbital orbit and an Earth-fixed /
// inertial mix-up. Frames: ITRF = Earth-fixed (rotates with the Earth), GCRS =
// inertial (IAU 2006), J2000 = the older mean-equator inertial frame (differs
// from GCRS by a fixed ~0.02 arcsec frame bias). Python cells:
// tests/python/test_initial_state.py.

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/initial_state.hpp>
#include <ballistic/validation/sp3.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;

namespace {

const Epoch T0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
const Vector3d W(0.0, 0.0, constants::earth::ANGULAR_VELOCITY);

// Circular orbit at the given altitude and latitude: the position, the inertial
// velocity, and the Earth-fixed velocity (inertial minus Earth rotation ω×r).
struct Circ { Vector3d r, v_inertial, v_itrf; };

Circ circular(double alt_km, double lat_deg) {
    const double lat = lat_deg * constants::PI / 180.0;
    const double rn = constants::earth::RADIUS_EQUATORIAL + alt_km * 1e3;
    const Vector3d r = rn * Vector3d(std::cos(lat), 0.0, std::sin(lat));
    const Vector3d v = std::sqrt(constants::earth::GM / rn) * Vector3d::UnitY();
    return {r, v, v - W.cross(r)};
}

// A cheap force model (gravity degree 8, nothing else) for fast checks.
TrajectoryConfig light() {
    TrajectoryConfig c;
    c.gravity_degree = 8;
    c.enable_third_body = false;
    c.enable_relativistic = false;
    c.enable_solid_tides = false;
    c.ocean_tide_degree = 0;
    return c;
}

template <class E, class F>
bool throws(F&& f, const std::string& needle = {}) {
    try { f(); } catch (const E& e) { return needle.empty() || std::string(e.what()).find(needle) != std::string::npos; }
    catch (...) { return false; }
    return false;
}

}  // namespace

int main() {
    time::EOPData::instance().set_policy(time::EopPolicy::AssumeZero);   // synthetic states; no EOP file needed
    TestRunner runner("InitialState");

    // First (the hint is once per process): a config without radiation pressure or
    // drag gets one ConfigHint naming its measured cost; it never raises in strict mode.
    runner.add_test("bare_config_reports_its_measured_cost_once", []() {
        std::vector<diagnostics::Diagnostic> got;
        diagnostics::set_handler([&](const diagnostics::Diagnostic& d) { got.push_back(d); });
        const Circ c = circular(500, 20);
        TrajectoryConfig with_srp = light();
        with_srp.srp = SrpConfig{.mass_kg = 500.0, .area_m2 = 2.0};
        Trajectory a(InitialState::from_itrf(T0, c.r, c.v_itrf), with_srp);
        (void)a.propagate(T0 + 60.0);                       // radiation on: no hint
        Trajectory b(InitialState::from_itrf(T0, c.r, c.v_itrf), light());
        (void)b.propagate(T0 + 60.0);
        Trajectory d(InitialState::from_itrf(T0, c.r, c.v_itrf), light());
        (void)d.propagate(T0 + 60.0);                       // once per process
        diagnostics::set_handler(nullptr);
        std::size_t hints = 0;
        for (const auto& x : got)
            if (x.category == diagnostics::Category::ConfigHint) {
                ++hints;
                TEST_ASSERT(x.message.find("radiation pressure and drag are off") != std::string::npos);
                TEST_ASSERT(x.message.find("Measured at 500 km") != std::string::npos);
            }
        TEST_ASSERT_EQ(hints, std::size_t{1});
        // Advice, never an error: strict mode prints it instead of raising.
        diagnostics::set_strict(true);
        bool raised = false;
        try { diagnostics::report(diagnostics::Category::ConfigHint, "probe hint"); }
        catch (const StrictDiagnosticError&) { raised = true; }
        diagnostics::set_strict(false);
        TEST_ASSERT(!raised);
    });

    // Trajectory(InitialState) and the legacy (epoch, r, v) constructor give identical results.
    runner.add_test("state_ctor_is_bit_identical_to_the_itrf_ctor", []() {
        const Circ c = circular(500, 20);
        Trajectory a(InitialState::from_itrf(T0, c.r, c.v_itrf), light());
        Trajectory b(T0, c.r, c.v_itrf, light());
        const auto pa = a.propagate(T0 + 600.0).position;
        const auto pb = b.propagate(T0 + 600.0).position;
        TEST_ASSERT((pa.array() == pb.array()).all());
    });

    // A GCRS start converts to ITRF and back to < 1e-6 m / 1e-9 m/s (rounding only).
    runner.add_test("gcrs_state_round_trips", []() {
        const Circ c = circular(500, 20);
        const auto s = InitialState::from_gcrs(T0, c.r, c.v_inertial);
        const auto e = s.itrf_state(coords::ITRFRealization::ITRF2020);
        const auto g = util::itrf_to_gcrs(e, T0, coords::ITRFRealization::ITRF2020);
        TEST_ASSERT((g.position - c.r).norm() < 1e-6);
        TEST_ASSERT((g.velocity - c.v_inertial).norm() < 1e-9);
        Trajectory t(s, light());
        TEST_ASSERT((t.state_gcrs().position - c.r).norm() < 1e-6);
    });

    // Implausible starts raise a typed error whose message names the fix: an
    // inertial velocity given as Earth-fixed (or the reverse), kilometers for meters,
    // a position underground, a suborbital orbit without allow_suborbital.
    runner.add_test("plausibility_raises_with_the_fix", []() {
        const Circ c = circular(400, 0);
        TEST_ASSERT((throws<FrameError>([&] { (void)InitialState::from_itrf(T0, c.r, -c.v_inertial); },
                                        "from_gcrs")));
        TEST_ASSERT((throws<FrameError>([&] { (void)InitialState::from_gcrs(T0, c.r, c.v_itrf); },
                                        "from_itrf")));
        TEST_ASSERT((throws<InputPlausibilityError>(
            [&] { (void)InitialState::from_itrf(T0, c.r / 1e3, c.v_itrf / 1e3); }, "METERS")));
        TEST_ASSERT((throws<InputPlausibilityError>(
            [&] { (void)InitialState::from_itrf(T0, 0.9 * c.r, c.v_itrf); }, "below the ground")));
        TEST_ASSERT((throws<InputPlausibilityError>(
            [&] { (void)InitialState::from_itrf(T0, 0.97 * c.r, 0.9 * c.v_itrf); }, "allow_suborbital")));
        (void)InitialState::from_itrf(T0, 0.97 * c.r, 0.9 * c.v_itrf, std::nullopt, {.allow_suborbital = true});
        // The cheap units check also guards the ITRF constructor.
        TEST_ASSERT((throws<InputPlausibilityError>([&] { Trajectory t(T0, c.r / 1e3, c.v_itrf, light()); })));
    });

    // When both readings of the velocity give a bound orbit, the likely mix-up is
    // reported once as a FrameAssumption diagnostic, not an error.
    runner.add_test("prograde_mixup_is_a_frame_assumption_diagnostic", []() {
        const Circ c = circular(400, 45);
        std::vector<diagnostics::Diagnostic> got;
        diagnostics::set_handler([&](const diagnostics::Diagnostic& d) { got.push_back(d); });
        (void)InitialState::from_itrf(T0, c.r, c.v_inertial);
        (void)InitialState::from_itrf(T0, c.r, c.v_itrf);      // the right reading: silent
        (void)InitialState::from_gcrs(T0, c.r, c.v_inertial);
        diagnostics::set_handler(nullptr);
        TEST_ASSERT_EQ(got.size(), std::size_t{1});
        TEST_ASSERT(got[0].category == diagnostics::Category::FrameAssumption);
    });

    runner.add_test("j2000_is_the_iau2006_frame_bias", []() {
        // ERFA eraBp00 rb (GCRS -> mean J2000), its published test values, to 1e-17.
        const auto b = state_checks::frame_bias_gcrs_to_j2000();
        TEST_ASSERT_NEAR(b(0, 1), -0.7078279744199196626e-7, 1e-17);
        TEST_ASSERT_NEAR(b(0, 2), 0.8056217146976134152e-7, 1e-17);
        TEST_ASSERT_NEAR(b(1, 0), 0.7078279477857337206e-7, 1e-17);
        TEST_ASSERT_NEAR(b(1, 2), 0.3306041454222136517e-7, 1e-17);
        TEST_ASSERT_NEAR(b(2, 0), -0.8056217380986972157e-7, 1e-17);
        TEST_ASSERT_NEAR(b(2, 1), -0.3306040883980552500e-7, 1e-17);
    });

    // Keplerian elements give back their semi-major axis and inclination to 1e-12
    // (closed-form relations); e = 1 (parabolic) raises.
    runner.add_test("keplerian_elements", []() {
        const auto s = InitialState::from_keplerian(
            T0, {.a_m = 7000e3, .e = 0.01, .i_rad = 0.9, .raan_rad = 0.5, .argp_rad = 1.5,
                 .true_anomaly_rad = 0.7});
        const double mu = constants::earth::GM;
        const double energy = 0.5 * s.velocity().squaredNorm() - mu / s.position().norm();
        TEST_ASSERT_NEAR_REL(-mu / (2.0 * energy), 7000e3, 1e-12);
        const Vector3d h = s.position().cross(s.velocity());
        TEST_ASSERT_NEAR(std::acos(h.z() / h.norm()), 0.9, 1e-12);
        TEST_ASSERT((throws<ConfigError>(
            [&] { (void)InitialState::from_keplerian(T0, {.a_m = 7000e3, .e = 1.0}); })));
    });

    // An ITRF2014 start with an ITRF2020 config raises; a matching config accepts it.
    runner.add_test("realization_mismatch_raises", []() {
        const Circ c = circular(500, 20);
        const auto s = InitialState::from_itrf(T0, c.r, c.v_itrf, coords::ITRFRealization::ITRF2014);
        TEST_ASSERT((throws<ConfigError>([&] { Trajectory t(s, light()); }, "ITRF2014")));
        TrajectoryConfig cfg = light();
        cfg.itrf = coords::ITRFRealization::ITRF2014;
        Trajectory t(s, cfg);
        TEST_ASSERT((t.state_ecef().position - c.r).norm() < 1e-6);
    });

    // Solar radiation pressure without a mass and area raises (no silent default).
    runner.add_test("srp_needs_mass_and_area", []() {
        const Circ c = circular(500, 20);
        TrajectoryConfig cfg = light();
        cfg.srp = SrpConfig{};
        TEST_ASSERT((throws<ConfigError>([&] { Trajectory t(T0, c.r, c.v_itrf, cfg); }, "mass_kg")));
        cfg.srp = SrpConfig{.mass_kg = 500.0, .area_m2 = 2.0};
        Trajectory t(T0, c.r, c.v_itrf, cfg);
    });

    // A start state from an SP3 orbit file (the standard precise-orbit format)
    // needs the ITRF realization named, equals the file's interpolated state exactly,
    // and raises at the file edge (no centered window for the derived velocity).
    // Data: tests/data/sp3/champ/CHAMP_2005_182.sp3.
    runner.add_test("sp3_initial_state", []() {
        namespace sp3 = validation::sp3;
        const auto f = sp3::read(BALLISTIC_PROJECT_ROOT "/tests/data/sp3/champ/CHAMP_2005_182.sp3");
        const auto& arc = f.arcs.at(0);
        const Epoch t = arc.epoch0 + 3600.0;
        TEST_ASSERT((throws<FrameError>([&] { (void)sp3::initial_state(f, arc.satellite, t); }, "realization")));
        const auto s = sp3::initial_state(f, arc.satellite, t, coords::ITRFRealization::ITRF2005);
        const auto ref = arc.interpolate(3600.0);
        TEST_ASSERT((s.position() - ref.position_m).norm() == 0.0);
        TEST_ASSERT((s.velocity() - ref.velocity_m_s).norm() == 0.0);
        TEST_ASSERT((throws<DataCoverageError>(
            [&] { (void)sp3::initial_state(f, arc.satellite, arc.epoch0, coords::ITRFRealization::ITRF2005); },
            "edge")));
    });

    // Last: it changes the process-wide EOP and leap-second tables.
    runner.add_test("eop_hold_limit_does_not_query_leap_seconds_past_their_expiry", []() {
        // The end of the Earth orientation parameters (EOP: the IERS predictions
        // plus a 30-day hold) usually lies after the leap-second table's expiry.
        // Building that limit as a UTC epoch would query leap seconds past the
        // expiry and warn "leap-second table expires" (in strict mode: raise) on
        // every propagation, however old the start date. A table that expires
        // 2018-06-28 with the committed finals2000A (predictions into 2027)
        // sets up that case; the propagation must not raise.
        const std::string leap = "test_initial_state_leap.list";
        {
            std::ofstream f(leap);
            f << "#@\t3739132800\n3692217600\t37\t# 1 Jan 2017\n";
        }
        TEST_ASSERT(time::LeapSecondTable::instance().load(leap));
        TEST_ASSERT(time::EOPData::instance().load(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all"));
        diagnostics::set_strict(true);
        const Epoch t = Epoch::from_utc(2017, 3, 1, 0, 0, 0);
        const Circ c = circular(500, 20);
        bool raised = false;
        try {
            Trajectory tr(InitialState::from_itrf(t, c.r, c.v_itrf), light());
            (void)tr.propagate(t + 600.0);
        } catch (const StrictDiagnosticError& e) {
            raised = true;
            std::cerr << e.what() << std::endl;
        }
        diagnostics::set_strict(false);
        time::EOPData::instance().clear();
        std::filesystem::remove(leap);
        TEST_ASSERT(!raised);
    });

    runner.run();
    return print_final_summary();
}
