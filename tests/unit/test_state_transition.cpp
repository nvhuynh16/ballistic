// StateTransition (state_transition.hpp): the state transition matrix Phi and
// the covariance P = Phi P_0 Phi^T along a propagated Trajectory.
//
// Phi is checked against central differences of re-propagated perturbed
// trajectories. The arcs are dawn-dusk orbits (orbit normal along the Sun
// direction, beta angle 90 deg), which never enter the Earth's shadow; each
// cell checks that with the cylindrical shadow test at every recorded step,
// because radiation pressure is not differentiable at the shadow boundary.
// Every propagation steps at a fixed 10 s (RK45 initial = max step), so the
// nominal and perturbed runs share one step grid and their difference is
// smooth in the initial state.

#include "../test_utils.hpp"
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather.hpp>
#include <ballistic/ephemeris/jpl_ephemeris.hpp>
#include <ballistic/initial_state.hpp>
#include <ballistic/satellites.hpp>
#include <ballistic/state_transition.hpp>
#include <ballistic/trajectory_interpolator.hpp>

#include <Eigen/Eigenvalues>

#include <cmath>
#include <filesystem>
#include <iostream>

using namespace ballistic;
using namespace ballistic::test;
namespace atm = ballistic::atmosphere;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

namespace {

Epoch t0() { return Epoch::from_utc(2025, 6, 15, 0, 0, 0); }
constexpr double kArc = 6000.0;  // [s] about one orbit at 400-800 km; a multiple of the step

TrajectoryConfig conservative_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 20;
    cfg.enable_third_body = true;
    cfg.enable_relativistic = false;
    cfg.enable_solid_tides = false;
    cfg.ocean_tide_degree = 0;
    cfg.rk45.initial_step_size = 10.0;
    cfg.rk45.max_step_size = 10.0;
    return cfg;
}

TrajectoryConfig srp_config() {
    TrajectoryConfig cfg = conservative_config();
    const auto& s1 = satellites::get("S1A");
    cfg.srp = SrpConfig{.mass_kg = s1.mass_kg, .area_m2 = s1.srp_area_m2, .cr = s1.cr(),
                        .albedo = false, .earth_ir = false};
    return cfg;
}

TrajectoryConfig drag_config(const StateVector& start_gcrs) {
    TrajectoryConfig cfg = srp_config();
    cfg.drag_env = atm::make_drag_environment(
        t0(), start_gcrs.position, start_gcrs.velocity, kArc, cfg,
        {.beta = atm::BetaProfile1D::constant(satellites::get("S1A").beta_kg_m2())});
    return cfg;
}

// Circular dawn-dusk start (GCRS) at `altitude_m`: position perpendicular to
// the Sun direction, orbit normal along it.
StateVector dawn_dusk_start(double altitude_m) {
    const Vector3d s = ephemeris::get_default_ephemeris_provider()->sun_position(t0()).normalized();
    const Vector3d u = s.cross(Vector3d::UnitZ()).normalized();
    const double r = constants::earth::RADIUS_EQUATORIAL + altitude_m;
    const double v = std::sqrt(constants::earth::GM / r);
    return StateVector{r * u, v * s.cross(u)};
}

Trajectory fly(const StateVector& start_gcrs, const TrajectoryConfig& cfg, double arc = kArc) {
    Trajectory traj(InitialState::from_gcrs(t0(), start_gcrs.position, start_gcrs.velocity), cfg);
    (void)traj.propagate_until(t0() + arc, {}, false);
    return traj;
}

// Smallest distance [m] from the Sun-Earth line among the recorded states on
// the night side (r . s < 0); infinity if none is on the night side. Below the
// Earth's radius the state would be in the cylindrical shadow.
double min_night_side_distance(Trajectory& traj) {
    InterpolatorConfig ic;
    ic.output_frame = InterpolationFrame::GCRS;
    ic.allow_propagation = false;
    TrajectoryInterpolator interp(traj, ic);
    auto eph = ephemeris::get_default_ephemeris_provider();
    double dmin = std::numeric_limits<double>::infinity();
    for (const auto& h : traj.history()) {
        const auto st = interp.get_state(h.epoch);
        const Vector3d s = eph->sun_position(h.epoch).normalized();
        const double along = st->position.dot(s);
        if (along < 0.0) dmin = std::min(dmin, (st->position - along * s).norm());
    }
    return dmin;
}

// Largest relative error over the four 3x3 blocks of Phi (they differ in units).
double block_rel_err(const Matrix6& a, const Matrix6& b) {
    double worst = 0.0;
    for (int i = 0; i < 6; i += 3)
        for (int j = 0; j < 6; j += 3)
            worst = std::max(worst, (a.block<3, 3>(i, j) - b.block<3, 3>(i, j)).norm() /
                                        b.block<3, 3>(i, j).norm());
    return worst;
}

// Central-difference Phi(t0 + arc, t0) from 12 perturbed propagations, and the
// largest relative column error of `phi` against it (position and velocity rows
// compared separately).
double fd_column_error(const StateVector& start, const TrajectoryConfig& cfg, const Matrix6& phi) {
    const double delta[6] = {1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3};
    double worst = 0.0;
    for (int j = 0; j < 6; ++j) {
        Vector6 col;
        StateVector plus = start, minus = start;
        if (j < 3) { plus.position(j) += delta[j]; minus.position(j) -= delta[j]; }
        else { plus.velocity(j - 3) += delta[j]; minus.velocity(j - 3) -= delta[j]; }
        const StateVector xp = fly(plus, cfg).state_gcrs();
        const StateVector xm = fly(minus, cfg).state_gcrs();
        col.head<3>() = (xp.position - xm.position) / (2.0 * delta[j]);
        col.tail<3>() = (xp.velocity - xm.velocity) / (2.0 * delta[j]);
        const Vector6 c = phi.col(j);
        const double ep = (c.head<3>() - col.head<3>()).norm() / col.head<3>().norm();
        const double ev = (c.tail<3>() - col.tail<3>()).norm() / col.tail<3>().norm();
        std::cout << "    column " << j << ": position rows " << ep << ", velocity rows " << ev << "\n";
        worst = std::max({worst, ep, ev});
    }
    return worst;
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

    TestRunner runner("state transition");

    runner.add_test("stm_matches_central_differences_gravity_third_body_srp", []() {
        // Measured: worst column error 2.0e-7 (the out-of-plane z column's
        // velocity rows; the other columns 1e-9 to 8e-8), the level of the
        // finite-difference reference itself. Bound 2x. The analytic gravity
        // Hessian agrees with fourth-order differences of the acceleration to
        // 5e-11 (test_gravity_hessian); an error in it shows here first.
        const StateVector x0 = dawn_dusk_start(800e3);
        const auto cfg = srp_config();
        Trajectory traj = fly(x0, cfg);
        const double dmin = min_night_side_distance(traj);
        std::cout << "    closest night-side approach to the Sun-Earth line: " << dmin << " m\n";
        TEST_ASSERT(dmin > constants::earth::RADIUS_EQUATORIAL);   // never in shadow
        StateTransition st(traj);
        const double err = fd_column_error(x0, cfg, st.stm(t0() + kArc));
        std::cout << "    worst relative column error " << err << "\n";
        TEST_ASSERT(err < 2e-6);   // measured 8.8e-7 (worst column: the out-of-plane
                                   // velocity column, whose norm nearly vanishes after one orbit)
    });

#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN   // needs the live NRLMSIS 2.1 drag model
    runner.add_test("stm_matches_central_differences_with_live_drag", []() {
        // 400 km, live NRLMSIS 2.1 on the committed weather, S1A beta. Drag's
        // position columns of da/dr are finite differences.
        // Measured: worst column error 5.6e-8 to 1.0e-7 (z column velocity
        // rows). A 3e-5 change in density moved single columns by x0.4 to
        // x3.5 in both directions: that is the noise of the re-propagated
        // finite-difference reference (each perturbed run takes its own
        // adaptive steps), not STM error. Bound 3x the largest seen.
        const StateVector x0 = dawn_dusk_start(400e3);
        const auto cfg = drag_config(x0);
        Trajectory traj = fly(x0, cfg);
        TEST_ASSERT(min_night_side_distance(traj) > constants::earth::RADIUS_EQUATORIAL);
        StateTransition st(traj);
        const double err = fd_column_error(x0, cfg, st.stm(t0() + kArc));
        std::cout << "    worst relative column error " << err << "\n";
        TEST_ASSERT(err < 3e-7);
    });
#endif

    runner.add_test("conservative_forces_give_unit_determinant_and_symplectic_phi", []() {
        // Velocities in units of r n (n = mean motion), so every block of Phi
        // is of order one: D Phi D^-1 with D = diag(I, I/n) is symplectic
        // exactly when Phi is.
        const StateVector x0 = dawn_dusk_start(600e3);
        Trajectory traj = fly(x0, conservative_config());
        StateTransition st(traj);
        const double n = x0.velocity.norm() / x0.position.norm();
        Matrix6 D = Matrix6::Identity();
        D.bottomRightCorner<3, 3>() /= n;
        Matrix6 J = Matrix6::Zero();
        J.block<3, 3>(0, 3).setIdentity();
        J.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        for (double t : {1000.0, 3000.0, kArc}) {
            const Matrix6 phi = D * st.stm(t0() + t) * D.inverse();
            const double det_err = std::abs(phi.determinant() - 1.0);
            const double sym_err = (phi.transpose() * J * phi - J).cwiseAbs().maxCoeff();
            std::cout << "    t = " << t << " s: |det - 1| = " << det_err
                      << ", max |Phi^T J Phi - J| = " << sym_err << "\n";
            // Measured at 6000 s: |det - 1| 1.0e-10, symplectic 4.3e-10.
            TEST_ASSERT(det_err < 1e-9);
            TEST_ASSERT(sym_err < 1e-9);
        }
    });

    runner.add_test("composition_through_an_intermediate_epoch", []() {
        // Phi(t2, t0) against Phi(t2, t1) Phi(t1, t0). The factors come from a
        // second run of the same trajectory: Phi(t1, t0) from its history up
        // to t1, then the history is cleared (the state at t1 becomes the
        // first entry) and Phi(t2, t1) comes from the rest of the run. Both
        // runs fly the same path (restarting from a recorded state instead
        // moves the path by ~2 cm over the arc, which would dominate).
        const double t1 = 2500.0;
        const StateVector x0 = dawn_dusk_start(600e3);
        Trajectory a = fly(x0, conservative_config());
        StateTransition sa(a);
        Trajectory b = fly(x0, conservative_config(), t1);
        StateTransition s10(b);
        b.clear_history();
        (void)b.propagate_until(t0() + kArc, {}, false);
        StateTransition s21(b);
        TEST_ASSERT_NEAR(s21.epoch0() - t0(), t1, 1e-9);
        std::cout << "    end states differ by "
                  << (a.state_gcrs().position - b.state_gcrs().position).norm() << " m\n";
        const Matrix6 lhs = sa.stm(t0() + kArc);
        const Matrix6 rhs = s21.stm(t0() + kArc) * s10.stm(s21.epoch0());
        const double err = block_rel_err(rhs, lhs);
        std::cout << "    worst block relative difference " << err << "\n";
        // Measured: 3.4e-10 (the two runs differ by 7e-7 m at the end, where
        // the integrator stopped and restarted at t1).
        TEST_ASSERT(err < 1e-9);
    });

    runner.add_test("covariance_along_track_grows_radial_stays_bounded", []() {
        Trajectory traj = fly(dawn_dusk_start(600e3), conservative_config());
        StateTransition st(traj);
        Matrix6 P_0 = Matrix6::Zero();
        P_0.diagonal() << 1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6;
        InterpolatorConfig ic;
        ic.output_frame = InterpolationFrame::GCRS;
        ic.allow_propagation = false;
        TrajectoryInterpolator interp(traj, ic);
        double sigma_t_end = 0.0, sigma_r_max = 0.0;
        for (double t = 0.0; t <= kArc; t += 500.0) {
            const Matrix6 P = st.covariance(t0() + t, P_0);
            TEST_ASSERT((P - P.transpose()).cwiseAbs().maxCoeff() == 0.0);
            Eigen::SelfAdjointEigenSolver<Matrix6> es(P);
            TEST_ASSERT(es.eigenvalues().minCoeff() > -1e-9 * es.eigenvalues().maxCoeff());
            const auto s = interp.get_state(t0() + t);
            const Vector3d R = s->position.normalized();
            const Vector3d N = s->position.cross(s->velocity).normalized();
            const Vector3d T = N.cross(R);
            const Eigen::Matrix3d Prr = P.topLeftCorner<3, 3>();
            const double sigma_r = std::sqrt(R.dot(Prr * R));
            const double sigma_t = std::sqrt(T.dot(Prr * T));
            sigma_r_max = std::max(sigma_r_max, sigma_r);
            sigma_t_end = sigma_t;
        }
        // Measured: along-track 1 m -> 25.7 m over 6000 s; radial at most 4.7 m.
        std::cout << "    along-track sigma at the end " << sigma_t_end
                  << " m, largest radial sigma " << sigma_r_max << " m\n";
        TEST_ASSERT(sigma_t_end > 15.0);
        TEST_ASSERT(sigma_r_max < 8.0);
    });

    runner.add_test("errors", []() {
        Trajectory empty(InitialState::from_gcrs(t0(), dawn_dusk_start(600e3).position,
                                                 dawn_dusk_start(600e3).velocity),
                         conservative_config());
        TEST_ASSERT_THROWS(StateTransition{empty}, ConfigError);
        Trajectory traj = fly(dawn_dusk_start(600e3), conservative_config(), 600.0);
        StateTransition st(traj);
        TEST_ASSERT_THROWS((void)st.stm(t0() + 601.0), DataCoverageError);
        TEST_ASSERT_THROWS((void)st.stm(t0() - 1.0), DataCoverageError);
        Matrix6 P_0 = Matrix6::Identity();
        P_0(0, 1) = 0.5;
        TEST_ASSERT_THROWS((void)st.covariance(t0() + 300.0, P_0), InputPlausibilityError);
        TEST_ASSERT_NEAR((st.stm(t0()) - Matrix6::Identity()).norm(), 0.0, 0.0);
    });

    runner.run();
    return print_final_summary();
}
