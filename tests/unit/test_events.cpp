// Events and propagate_until (events.hpp, Trajectory::propagate_until): an event
// (altitude crossing, ground impact, perigee, apogee) is located inside the
// integration step by root finding. Also covers three reentry traps listed in
// docs/trap_census.md: GROUND-PASS-THROUGH (without a stop at the ground, a state
// that fell through the surface would integrate toward the Earth's center and
// hang), ALT-GEOCENTRIC (altitude must be geodetic; an altitude measured from a
// geocentric-latitude radius is -21 km off at 45 deg), FORCED-STEP-SILENT (a step
// the integrator could not bring to tolerance must be reported).

#include "../test_utils.hpp"
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/events.hpp>
#include <ballistic/trajectory_interpolator.hpp>

#include <cmath>
#include <filesystem>

using namespace ballistic;
using namespace ballistic::test;
namespace atm = ballistic::atmosphere;

namespace {

Epoch t0() { return Epoch::from_utc(2024, 1, 10, 0, 0, 0); }

// Point-mass gravity only, nothing else: the radial fall has a closed form.
TrajectoryConfig vacuum_point_mass() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 0;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    cfg.ocean_tide_degree = 0;
    return cfg;
}

TrajectoryConfig fast_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 8;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    cfg.ocean_tide_degree = 0;
    return cfg;
}

// Time to fall radially from rest at r0 to r1 under GM/r^2:
// t = sqrt(r0^3 / 2GM) * (sqrt(x (1 - x)) + acos(sqrt(x))), x = r1 / r0.
double radial_fall_time(double r0, double r1) {
    const double x = r1 / r0;
    return std::sqrt(r0 * r0 * r0 / (2.0 * constants::earth::GM)) *
           (std::sqrt(x * (1.0 - x)) + std::acos(std::sqrt(x)));
}

}  // namespace

int main() {
    InitConfig init;
    init.eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
    initialize(init);
    const std::filesystem::path sw_csv = BALLISTIC_PROJECT_ROOT "/tests/data/weather/SW-All.csv";
    atm::SpaceWeatherData::instance().clear();
    if (!atm::SpaceWeatherData::instance().load_celestrak_csv(sw_csv)) {
        std::cerr << "could not load " << sw_csv << "\n";
        return 1;
    }

    TestRunner runner("events");

    runner.add_test("vacuum_drop_over_pole_matches_closed_form", []() {
        // At rest (Earth-fixed = inertial on the pole) 200 km above the north
        // pole: the geodetic altitude on the axis is |z| - b, the fall is
        // radial, and the impact time is the closed form above.
        const double b = constants::earth::RADIUS_POLAR;
        const double r0 = b + 200e3;
        Trajectory traj(t0(), Vector3d(0.0, 0.0, r0), Vector3d::Zero(), vacuum_point_mass());
        const auto r = traj.propagate_until(t0() + 3600.0);
        TEST_ASSERT(r.stopped_by.has_value());
        TEST_ASSERT(r.stopped_by->name == "ground impact");
        const double t_impact = r.epoch - t0();
        TEST_ASSERT_NEAR(t_impact, radial_fall_time(r0, b), 1e-3);
        // Ends ON the surface (never above it), within 1e-4 s of fall (~0.2 m).
        const double alt = coords::ecef_to_geodetic(r.state_ecef.position).altitude;
        TEST_ASSERT(alt <= 0.0 && alt > -0.5);
        TEST_ASSERT(traj.history().back().epoch == r.epoch);
    });

    runner.add_test("altitude_events_at_45_and_80_deg", []() {
        // A steep descent through 60 km at high latitude: the located state
        // is at 60 km GEODETIC (a geocentric approximation is -21 / -41 km off at
        // 45 / 80 deg latitude).
        for (double lat_deg : {45.0, 80.0}) {
            const double lat = lat_deg * constants::DEG_TO_RAD, lon = 0.3;
            const Vector3d p = coords::geodetic_to_ecef(lat, lon, 120e3);
            const Matrix3d enu = coords::ecef_to_enu_rotation(lat, lon);
            const Vector3d v = enu.transpose() * Vector3d(0.0, 300.0, -1500.0);   // north, down
            Trajectory traj(t0(), p, v, fast_config());
            const auto r = traj.propagate_until(
                t0() + 600.0, {events::altitude_crossing(60e3, events::Direction::Falling)});
            TEST_ASSERT(r.stopped_by.has_value());
            const auto g = coords::ecef_to_geodetic(r.state_ecef.position);
            TEST_ASSERT_NEAR(g.altitude, 60e3, 0.5);
            TEST_ASSERT_NEAR(g.latitude / constants::DEG_TO_RAD, lat_deg, 1.0);
        }
    });

    runner.add_test("non_terminal_events_are_logged", []() {
        // One LEO orbit: a perigee and an apogee each, the target reached.
        const Vector3d p(6778137.0, 0.0, 0.0);
        const Vector3d v(0.0, 1200.0, 7400.0);
        Trajectory traj(t0(), p, v, fast_config());
        const auto r = traj.propagate_until(t0() + 6000.0, {events::perigee(), events::apogee()});
        TEST_ASSERT(r.reached_target());
        TEST_ASSERT_NEAR(r.epoch - t0(), 6000.0, 1e-6);
        int n_peri = 0, n_apo = 0;
        for (const auto& o : r.log) {
            if (o.name == "perigee") ++n_peri;
            if (o.name == "apogee") ++n_apo;
            TEST_ASSERT(std::abs(o.position_ecef.dot(o.velocity_ecef)) < 1e-1 * o.position_ecef.norm());
        }
        TEST_ASSERT(n_peri >= 1 && n_apo >= 1);
    });

#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN   // needs the live NRLMSIS 2.1 drag model
    runner.add_test("decaying_200km_orbit_ends_at_ground_impact", []() {
        // Live NRLMSIS 2.1 on the committed measured weather, a low beta so
        // the orbit decays in about a day; the reentry ends exactly at the
        // ground, with no step force-accepted on the way.
        TrajectoryConfig cfg = fast_config();
        const Vector3d p = coords::geodetic_to_ecef(0.4, 0.2, 200e3);
        const double r = p.norm();
        const double v_inertial = std::sqrt(constants::earth::GM / r);
        // Prograde, inclined 50 deg; Earth-fixed velocity = inertial - w x r.
        const Vector3d east = Vector3d(-p.y(), p.x(), 0.0).normalized();
        const Vector3d north = p.normalized().cross(east);   // up x east
        const Vector3d along = std::cos(0.87) * east + std::sin(0.87) * north;
        const Vector3d w(0.0, 0.0, constants::earth::ANGULAR_VELOCITY);
        const Vector3d v = v_inertial * along.normalized() - w.cross(p);
        atm::DragSetup setup{.beta = atm::BetaProfile1D::constant(20.0)};
        cfg.drag_env = atm::make_drag_environment(t0(), p, v, 0.0, cfg, setup);
        Trajectory traj(t0(), p, v, cfg);
        const auto res = traj.propagate_until(t0() + 5.0 * 86400.0);
        TEST_ASSERT(res.stopped_by.has_value());
        TEST_ASSERT(res.stopped_by->name == "ground impact");
        const double alt = coords::ecef_to_geodetic(res.state_ecef.position).altitude;
        TEST_ASSERT(alt <= 0.0 && alt > -1.0);
        for (const auto& w_msg : traj.warnings())
            TEST_ASSERT(w_msg.find("failed its error test") == std::string::npos);
        std::cout << "(impact after " << (res.epoch - t0()) / 3600.0 << " h) ";
    });
#endif

    runner.add_test("falling_through_the_ground_raises_instead_of_hanging", []() {
        // GROUND-PASS-THROUGH: a large downward velocity, no drag, plain
        // propagate(): without the guard it would integrate toward the Earth's
        // center and hang; it must stop with an error instead.
        const Vector3d p(6778137.0, 0.0, 0.0);
        const Vector3d v(-7000.0, 500.0, 0.0);
        Trajectory traj(t0(), p, v, fast_config());
        TEST_ASSERT_THROWS((void)traj.propagate(t0() + 3600.0), DataCoverageError);
    });

    runner.add_test("forced_step_is_reported_and_strict_raises", []() {
        // FORCED-STEP-SILENT: an acceleration that jumps by 1 m/s^2 within
        // 1 us (an EA table) cannot be integrated to tolerance even at the
        // 1 ms minimum step; the integrator force-accepts the step and must
        // report a ForcedStep note (an exception in strict mode).
        auto make = [] {
            TrajectoryConfig cfg = fast_config();
            cfg.empirical_accel = std::make_shared<forces::EmpiricalAcceleration>(
                forces::EmpiricalAcceleration::from_entries(
                    {{0.0, Vector3d::Zero()}, {100.0, Vector3d::Zero()},
                     {100.0 + 1e-6, Vector3d(1.0, 0.0, 0.0)}, {1000.0, Vector3d(1.0, 0.0, 0.0)}},
                    forces::EmpiricalAccelFrame::GCRS, forces::EAInterp::Linear));
            return std::make_unique<Trajectory>(t0(), Vector3d(6778137.0, 0.0, 0.0),
                                                Vector3d(0.0, 1000.0 - 494.27, 7600.0), cfg);
        };
        auto traj = make();
        while (traj->epoch() - t0() < 200.0) (void)traj->step();
        bool noted = false;
        for (const auto& m : traj->warnings())
            noted = noted || m.find("failed its error test") != std::string::npos;
        TEST_ASSERT(noted);

        diagnostics::set_strict(true);
        auto strict_traj = make();
        bool threw = false;
        try {
            while (strict_traj->epoch() - t0() < 200.0) (void)strict_traj->step();
        } catch (const StrictDiagnosticError&) {
            threw = true;
        }
        diagnostics::set_strict(false);
        TEST_ASSERT(threw);
    });

    runner.add_test("propagate_until_refuses_a_past_target", []() {
        Trajectory traj(t0(), Vector3d(6778137.0, 0.0, 0.0), Vector3d(0.0, 1000.0 - 494.27, 7600.0), fast_config());
        TEST_ASSERT_THROWS((void)traj.propagate_until(t0() - 10.0), ConfigError);
    });

    runner.add_test("locate_brackets_the_root_on_the_far_side", []() {
        // f(t) = 1 - t^2 on [0, 2]: root at 1; the located time is within
        // the tolerance and on the far (negative) side.
        auto f = [](double t) { return 1.0 - t * t; };
        const double t = events::detail::locate(f, 2.0, f(0.0), f(2.0), 1e-6);
        TEST_ASSERT(t >= 1.0 && t - 1.0 < 1e-6);
    });

    runner.run();
    return print_final_summary();
}
