// Flow-regime drag coefficient (aero/flow_regime.hpp) and the AeroBeta beta
// column: free-molecular closed forms against quadrature, the continuum and
// bridge limits, the sphere against Henderson (1976), the U.S. Standard
// Atmosphere 1976 mean free path, bit-identity of a constant Cd with the
// beta-profile path, the Jacobian, and a reentry flown to the ground.
//
// Terms: Cd = drag coefficient; beta = m/(Cd*A), the ballistic coefficient;
// free-molecular = rarefied flow where molecules hit the body without colliding
// with each other (high orbit); continuum = ordinary dense-air flow; the bridge
// blends the two by the Knudsen number Kn (mean free path / body size).
// Tolerances: quadrature cells at 1e-6 (numerical integration accuracy), the
// sphere-vs-Henderson cell at the measured spread quoted in the cell.

#include "../test_utils.hpp"
#include <ballistic/aero/flow_regime.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN
#include <ballistic/atmosphere/internal/nrlmsis21_full.hpp>
#endif
#include <ballistic/atmosphere/ussa76.hpp>
#include <ballistic/events.hpp>
#include <ballistic/forces/drag.hpp>

#include <cmath>
#include <filesystem>
#include <numbers>

using namespace ballistic;
using namespace ballistic::test;
namespace atm = ballistic::atmosphere;

namespace {

Epoch t0() { return Epoch::from_utc(2024, 1, 10, 0, 0, 0); }

TrajectoryConfig fast_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 8;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    cfg.ocean_tide_degree = 0;
    return cfg;
}

// Composite Gauss-Legendre (5 points per panel) of f on [a, b].
template <typename F>
double integrate(F&& f, double a, double b, int panels = 2000) {
    static constexpr double x[5] = {-0.9061798459386640, -0.5384693101056831, 0.0,
                                    0.5384693101056831, 0.9061798459386640};
    static constexpr double w[5] = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                                    0.4786286704993665, 0.2369268850561891};
    const double h = (b - a) / panels;
    double sum = 0.0;
    for (int i = 0; i < panels; ++i) {
        const double c = a + (i + 0.5) * h;
        for (int k = 0; k < 5; ++k) sum += w[k] * f(c + 0.5 * h * x[k]);
    }
    return 0.5 * h * sum;
}

std::shared_ptr<const atm::DragEnvironment> live_env(atm::DragSetup setup) {
    TrajectoryConfig cfg = fast_config();
    return atm::make_drag_environment(t0(), Vector3d(7e6, 0, 0), Vector3d::Zero(), 0.0, cfg, setup);
}

aero::AeroBeta sphere(double mass_kg, double diameter_m) {
    const double area = std::numbers::pi * diameter_m * diameter_m / 4.0;
    return aero::AeroBeta{mass_kg, area, diameter_m, aero::FlowRegimeCd{aero::Shape::Sphere}};
}

// Earth-fixed state at geodetic (lat, lon, h) moving east at `speed`.
std::pair<Vector3d, Vector3d> east_moving(double h_m, double speed) {
    const Vector3d p = coords::geodetic_to_ecef(0.3, 0.2, h_m);
    const Vector3d east = Vector3d(-p.y(), p.x(), 0.0).normalized();
    return {p, speed * east};
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

    TestRunner runner("flow-regime Cd");

    runner.add_test("free_molecular_closed_forms_match_quadrature", []() {
        // The sphere (Schaaf & Chambre) and cross-flow cylinder (Bessel-function)
        // closed forms are the flat-face Sentman formula integrated over the
        // surface: sphere 2 * int_{-1}^{1} face(g) dg, cylinder
        // 1/2 * int_0^{2 pi} face(cos phi) dphi.
        for (double s : {0.5, 2.0, 7.0, 20.0}) {
            for (double vr : {0.0, 0.1, 0.7}) {
                const double q_sphere = 2.0 * integrate([&](double g) { return aero::sentman_face(g, s, vr); }, -1.0, 1.0);
                const double q_cyl = 0.5 * integrate([&](double p) { return aero::sentman_face(std::cos(p), s, vr); },
                                                     0.0, 2.0 * std::numbers::pi);
                TEST_ASSERT_NEAR(aero::free_molecular_cd(aero::Shape::Sphere, s, vr), q_sphere, 1e-6);
                TEST_ASSERT_NEAR(aero::free_molecular_cd(aero::Shape::Cylinder, s, vr), q_cyl, 1e-6);
            }
        }
        // Hypervelocity limit with no re-emission: every shape's Cd -> 2.
        for (auto sh : {aero::Shape::Sphere, aero::Shape::FlatPlate, aero::Shape::Cylinder})
            TEST_ASSERT_NEAR(aero::free_molecular_cd(sh, 1e3, 0.0), 2.0, 1e-5);
        // exp(-x) I_0(x), exp(-x) I_1(x) against 40-digit mpmath values, across
        // both branches (series to x = 50, asymptotic above).
        struct BesselRef { double x, i0e, i1e; };
        for (const BesselRef& r : {BesselRef{0.1, 0.90710092578230109644, 0.045298446808809325007},
                                   BesselRef{1.0, 0.4657596075936404365, 0.20791041534970844887},
                                   BesselRef{7.5, 0.14831583007739550284, 0.13804121154855420249},
                                   BesselRef{30.0, 0.073145946482237293929, 0.071916330598647554706},
                                   BesselRef{50.0, 0.05656162664745419253, 0.055993123892895399644},
                                   BesselRef{50.000001, 0.056561626078951446543, 0.055993123341535684485},
                                   BesselRef{200.0, 0.02822715994911191567, 0.028156503394832917822},
                                   BesselRef{700.0, 0.015081295651531357587, 0.015070519444716846949},
                                   BesselRef{5000.0, 0.005642036898744588657, 0.0056414726668388859036}}) {
            TEST_ASSERT_NEAR(aero::scaled_bessel_i(0, r.x) / r.i0e, 1.0, 5e-14);
            TEST_ASSERT_NEAR(aero::scaled_bessel_i(1, r.x) / r.i1e, 1.0, 5e-14);
        }
#if !defined(_LIBCPP_VERSION)   // libc++ has no std::cyl_bessel_i
        for (double x : {0.01, 0.5, 3.0, 12.0, 25.0, 49.9, 50.1, 120.0, 400.0, 699.0}) {
            for (int nu : {0, 1}) {
                const double ref = std::exp(-x) * std::cyl_bessel_i(static_cast<double>(nu), x);
                TEST_ASSERT_NEAR(aero::scaled_bessel_i(nu, x) / ref, 1.0, 5e-14);
            }
        }
#endif
    });

    runner.add_test("continuum_limits", []() {
        // Rayleigh pitot: Cp_max -> ((g+1)^2/(4g))^(g/(g-1)) * 4/(g+1) = 1.8394 at
        // gamma 1.4 (Anderson, Hypersonic and High-Temperature Gas Dynamics,
        // eq. 3.15), and continuous with isentropic compression at Mach 1.
        TEST_ASSERT_NEAR(aero::stagnation_cp(1e4), 1.8394, 1e-4);
        TEST_ASSERT_NEAR(aero::stagnation_cp(1.0 - 1e-9), aero::stagnation_cp(1.0 + 1e-9), 1e-6);
        // Henderson's hypersonic sphere (0.9 + 0.34/M^2) agrees with Newtonian
        // (Cp_max / 2 = 0.92) to 3%, two independent routes.
        TEST_ASSERT_NEAR_REL(aero::henderson_continuum_sphere(25.0), aero::newtonian_cd(aero::Shape::Sphere, 25.0), 0.03);
        // The continuum sphere is Henderson's full correlation at Re -> infinity.
        for (double M : {0.3, 0.9, 1.3, 2.0, 6.0, 20.0})
            TEST_ASSERT_NEAR(aero::henderson_continuum_sphere(M), aero::henderson1976_sphere(M, 1e14, 1.0), 1e-5);
    });

    runner.add_test("bridge_is_continuous_and_monotone", []() {
        TEST_ASSERT_NEAR(aero::bridge_weight(1e-3), 0.0, 1e-15);
        TEST_ASSERT_NEAR(aero::bridge_weight(1e-3 * (1 + 1e-9)), 0.0, 1e-9);
        TEST_ASSERT_NEAR(aero::bridge_weight(10.0 * (1 - 1e-9)), 1.0, 1e-9);
        TEST_ASSERT_NEAR(aero::bridge_weight(0.1), 0.5, 1e-12);
        double prev = 0.0;
        for (double lk = -3.0; lk <= 1.0; lk += 0.01) {
            const double w = aero::bridge_weight(std::pow(10.0, lk));
            TEST_ASSERT(w >= prev - 1e-15);
            prev = w;
        }
    });

    runner.add_test("mean_free_path_matches_ussa76", []() {
        // U.S. Standard Atmosphere 1976, Table I: sea-level mean free path
        // 6.6328e-8 m (hard spheres, sigma = 3.65e-10 m).
        const auto u = atm::ussa76::evaluate(0.0);
        TEST_ASSERT_NEAR_REL(aero::mean_free_path(u.density_kg_m3, u.molar_mass_kg_kmol), 6.6328e-8, 1e-3);
        // and its sea-level speed of sound, 340.29 m/s.
        TEST_ASSERT_NEAR_REL(aero::speed_of_sound(u.temperature_k, u.molar_mass_kg_kmol), 340.29, 1e-4);
    });

    runner.add_test("sphere_agrees_with_henderson_1976", []() {
        // Henderson's correlation is fitted to ballistic-range and wind-tunnel
        // data across the regimes (max 16% from Bailey & Starr per Parmar et
        // al. 2010). Measured over this grid: RMS 7.9% / 7.1%,

        // max 16% / 14% (Tw/T 1 / 0.1), largest at Kn 0.1-1 where the bridge
        // sits above Henderson; at Kn 100 Henderson's own free-molecular limit
        // differs from the exact Sentman value by up to 11%.
        for (double tw_t : {1.0, 0.1}) {
            double sum2 = 0.0, worst = 0.0;
            int n = 0;
            for (double M : {2.0, 4.0, 8.0, 15.0, 25.0}) {
                for (double kn : {1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0}) {
                    // Kn = M / Re * sqrt(gamma pi / 2) (Singh et al. 2020, eq. 2.1)
                    const double re = M / kn * std::sqrt(aero::kGammaAir * std::numbers::pi / 2.0);
                    const double s = M * std::sqrt(aero::kGammaAir / 2.0);
                    const double vr = std::sqrt(tw_t) / s;   // alpha = 1
                    const double w = aero::bridge_weight(kn);
                    const double cd_c = aero::henderson_continuum_sphere(M);
                    const double ours = cd_c + (aero::free_molecular_cd(aero::Shape::Sphere, s, vr) - cd_c) * w;
                    const double rel = ours / aero::henderson1976_sphere(M, re, tw_t) - 1.0;
                    sum2 += rel * rel; worst = std::max(worst, std::abs(rel)); ++n;
                }
            }
            const double rms = std::sqrt(sum2 / n);
            std::cout << "(Tw/T " << tw_t << ": rms " << 100 * rms << "%, max " << 100 * worst << "%) ";
            TEST_ASSERT(rms < 0.10);
            TEST_ASSERT(worst < 0.20);
        }
    });

#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN   // cells on a live NRLMSIS 2.1 environment
    runner.add_test("live_env_gives_temperature_and_composition", []() {
        // Density is bit-identical to NRLMSIS 2.1's density() (same inputs);
        // the molar mass is ~28.96 near the ground and falls with altitude.
        WeatherConditions w = WeatherConditions::quiet();
        atm::DragSetup setup{.beta = atm::BetaProfile1D::constant(100.0)};
        setup.weather = atm::SpaceWeatherHistory::from_constant(w);
        const auto env = live_env(setup);
        const atm::NRLMSIS21 msis;
        for (double h_km : {5.0, 90.0, 300.0}) {
            const auto [p, v] = east_moving(h_km * 1e3, 7000.0);
            const auto s = env->evaluate(p, t0(), v);
            const auto g = coords::ecef_to_geodetic(p);
            constexpr double R2D = 180.0 / std::numbers::pi;
            TEST_ASSERT(s.rho_kg_m3 == msis.density(g.altitude / 1000.0, g.latitude * R2D, g.longitude * R2D, t0(), w));
            TEST_ASSERT(std::isfinite(s.temperature_k) && s.temperature_k > 150.0);
            if (h_km < 50) TEST_ASSERT_NEAR(s.molar_mass_kg_kmol, 28.96, 0.05);
            if (h_km > 200) TEST_ASSERT(s.molar_mass_kg_kmol < 25.0);
            TEST_ASSERT(std::isnan(s.cd));   // a beta profile has no Cd
        }
    });

    runner.add_test("constant_cd_is_bit_identical_to_the_beta_profile", []() {
        // The default for orbits does not move: an AeroBeta with a ConstantCd
        // gives the very same acceleration as BetaProfile1D::from_mass_cd_area.
        const auto env_profile = live_env({.beta = atm::BetaProfile1D::from_mass_cd_area(1200.0, 2.2, 4.2)});
        const auto env_aero = live_env({.beta = aero::AeroBeta{1200.0, 4.2, 0.0, aero::ConstantCd{2.2}}});
        forces::AtmosphericDrag drag;
        for (double h_km : {250.0, 500.0, 700.0}) {
            const Vector3d r = coords::geodetic_to_ecef(0.5, 1.0, h_km * 1e3);
            StateVector x{coords::itrf_to_gcrs_full(r, t0()), Vector3d(1000.0, -7000.0, 2500.0)};
            const Vector3d a1 = drag.acceleration(*env_profile, t0(), x);
            const Vector3d a2 = drag.acceleration(*env_aero, t0(), x);
            TEST_ASSERT(a1 == a2);
            TEST_ASSERT(a1.norm() > 0.0);
        }
    });

    runner.add_test("flow_regime_jacobian_differentiates_cd", []() {
        // Below 120 km the sphere's Cd changes with speed; the Jacobian's
        // velocity columns (central differences) match a finer independent
        // difference of the acceleration.
        const auto env = live_env({.beta = sphere(500.0, 1.0)});
        TEST_ASSERT(env->beta_depends_on_velocity());
        forces::AtmosphericDrag drag;
        const Vector3d r = coords::geodetic_to_ecef(0.5, 1.0, 95e3);
        StateVector x{coords::itrf_to_gcrs_full(r, t0()), Vector3d::Zero()};
        x.velocity = 7400.0 * Vector3d::UnitZ().cross(x.position).normalized();
        const auto [a, J] = drag.acceleration_with_jacobian(*env, t0(), x, 1.0, 1e-2);
        for (int i = 0; i < 3; ++i) {
            const double h = 1e-3;
            StateVector xp = x, xm = x;
            xp.velocity(i) += h;
            xm.velocity(i) -= h;
            const Vector3d col = (drag.acceleration(*env, t0(), xp) - drag.acceleration(*env, t0(), xm)) / (2 * h);
            TEST_ASSERT(((J.col(3 + i) - col).norm() < 1e-5 * J.rightCols(3).norm()));
        }
        TEST_ASSERT(a.norm() > 0.0);
    });

    runner.add_test("tabulated_density_refuses_flow_regime_cd", []() {
        // A table has no temperature: refuse, never guess.
        const auto env = atm::from_models_snapshot(t0(), 0.0, 0.0,
                                                   atm::SpaceWeatherHistory::from_constant(WeatherConditions::quiet()),
                                                   atm::BetaProfile1D::constant(100.0));
        atm::DragEnvironment e = env;
        e.set_beta(sphere(500.0, 1.0));
        const auto [p, v] = east_moving(300e3, 7700.0);
        TEST_ASSERT_THROWS((void)e.evaluate(p, t0(), v), ConfigError);
        // and a live env without a velocity refuses too.
        const auto live = live_env({.beta = sphere(500.0, 1.0)});
        TEST_ASSERT_THROWS((void)live->evaluate(p, t0()), ConfigError);
    });

#endif  // BALLISTIC_TEST_NO_NRL_PLUGIN

    runner.add_test("invalid_bodies_and_tables_raise", []() {
        TEST_ASSERT_THROWS((aero::AeroBeta{500.0, 1.0, 0.0, aero::FlowRegimeCd{}}.validate()), ConfigError);
        TEST_ASSERT_THROWS((aero::AeroBeta{0.0, 1.0, 1.0, aero::ConstantCd{2.2}}.validate()), ConfigError);
        aero::MachTable t{{0.5, 2.0, 10.0}, {1.1, 1.6, 1.5}};
        t.validate();
        TEST_ASSERT_NEAR(t(1.25), 1.35, 1e-12);
        TEST_ASSERT_THROWS((void)t(12.0), DataCoverageError);
        TEST_ASSERT_THROWS((aero::MachTable{{2.0, 1.0}, {1.0, 1.0}}.validate()), ConfigError);
    });

    runner.add_test("newtonian_below_mach_3_is_reported", []() {
        diagnostics::Sink::instance().reset_dedup();
        int seen = 0;
        diagnostics::set_handler([&](const diagnostics::Diagnostic& d) {
            if (d.key == "flow-regime-newtonian-below-mach-3") ++seen;
        });
        aero::FlowRegimeCd plate{aero::Shape::FlatPlate};
        const auto u = atm::ussa76::evaluate(10e3);
        aero::FlowState f{600.0, u.density_kg_m3, u.temperature_k, u.molar_mass_kg_kmol};
        (void)aero::flow_regime_cd(plate, f, 1.0);
        f.speed_m_s = 3000.0;   // Mach ~10: no report
        (void)aero::flow_regime_cd(plate, f, 1.0);
        diagnostics::set_handler(nullptr);
        TEST_ASSERT(seen == 1);
        // a MachTable for the body replaces the theory (no report).
        diagnostics::Sink::instance().reset_dedup();
        seen = 0;
        plate.continuum = aero::MachTable{{0.1, 5.0}, {1.17, 1.8}};
        f.speed_m_s = 600.0;
        diagnostics::set_handler([&](const diagnostics::Diagnostic&) { ++seen; });
        const auto r = aero::flow_regime_cd(plate, f, 1.0);
        diagnostics::set_handler(nullptr);
        TEST_ASSERT(seen == 0);
        TEST_ASSERT(r.cd > 1.17 && r.cd < 1.8);
    });

#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN
    runner.add_test("sphere_reentry_to_the_ground", []() {
        // A 1 m, 500 kg sphere from 120 km, 7.6 km/s, 2 deg down: flown to the
        // ground with a constant Cd (2.1, its free-molecular value) and with
        // the flow-regime Cd. The regime matters only below ~100 km, where the
        // flow-regime Cd falls from ~2.05 to ~0.9 (and ~0.4 subsonic). The
        // impact time is dominated by the final fall at terminal speed,
        // sqrt(2 m g / (rho Cd A)): the lower Cd falls faster and lands EARLIER.
        const double h0 = 120e3;
        const Vector3d p = coords::geodetic_to_ecef(0.4, 0.2, h0);
        const Vector3d up = p.normalized();
        const Vector3d east = Vector3d(-p.y(), p.x(), 0.0).normalized();
        const double gam = -2.0 * std::numbers::pi / 180.0;
        const Vector3d w(0.0, 0.0, constants::earth::ANGULAR_VELOCITY);
        const Vector3d v = 7600.0 * (std::cos(gam) * east + std::sin(gam) * up) - w.cross(p);
        auto fly = [&](const aero::AeroBeta& body, double& min_cd, double& max_cd) {
            TrajectoryConfig cfg = fast_config();
            cfg.drag_env = live_env({.beta = body});
            Trajectory traj(t0(), p, v, cfg);
            const auto res = traj.propagate_until(t0() + 3600.0);
            TEST_ASSERT(res.stopped_by.has_value() && res.stopped_by->name == "ground impact");
            for (const auto& m : traj.warnings()) TEST_ASSERT(m.find("failed its error test") == std::string::npos);
            min_cd = 1e9; max_cd = 0;
            for (const auto& node : traj.history()) {
                if (coords::ecef_to_geodetic(node.position).altitude < 0.0) continue;
                const double cd = cfg.drag_env->evaluate(node.position, node.epoch, node.velocity).cd;
                min_cd = std::min(min_cd, cd); max_cd = std::max(max_cd, cd);
            }
            return res;
        };
        double lo_c, hi_c, lo_f, hi_f;
        const auto rc = fly(aero::AeroBeta{500.0, std::numbers::pi / 4, 1.0, aero::ConstantCd{2.1}}, lo_c, hi_c);
        const auto rf = fly(sphere(500.0, 1.0), lo_f, hi_f);
        const double dt = rf.epoch - rc.epoch;
        const double range = (rf.state_ecef.position - rc.state_ecef.position).norm();
        std::cout << "(impact " << (rc.epoch - t0()) << " s constant / " << (rf.epoch - t0())
                  << " s flow-regime; Cd " << lo_f << "-" << hi_f << "; impact points " << range / 1e3 << " km apart) ";
        TEST_ASSERT(hi_f > 2.0 && hi_f < 2.2);
        TEST_ASSERT(lo_f < 0.9);
        TEST_ASSERT(lo_c == 2.1 && hi_c == 2.1);
        TEST_ASSERT(dt < -30.0);
    });
#endif  // BALLISTIC_TEST_NO_NRL_PLUGIN

    runner.run();
    return print_final_summary();
}
