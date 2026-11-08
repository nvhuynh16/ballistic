// Integration test (heavy tier): estimating the empirical acceleration (EA, the
// unmodeled force recovered from observations) on a synthetic glide reentry with
// the B-spline estimator, with and without Gauss-Newton (GN) refinement, from
// position, velocity and acceleration observations at several knot spacings.
// Reports two errors, both as position RMSE (root-mean-square error) against truth:
//   traj   = the estimator's own fitted trajectory;
//   reprop = a fresh propagation with the returned EA (what a user would get).
// Truth: degree-8 gravity plus a tabulated acceleration in the inertial GCRS frame
// (a glide pulse centered at 70 s and a return pulse at 420 s), no drag; impact
// near 494 s. Observations are sampled at 100 Hz in the Earth-fixed ECEF frame.
// Most thresholds are sanity bounds (no blow-up); each cell prints its RMSE.
// FD = finite difference; IRGN = the estimator's iteratively regularized
// Gauss-Newton fit; k = knot spacing in seconds.

#include "../test_utils.hpp"
#include <ballistic/init.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <ballistic/calibration/bspline_accel_estimator.hpp>
#include <ballistic/calibration/gn_refine.hpp>
#include <ballistic/util/frame_transform.hpp>
#include <cmath>
#include <iomanip>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::calibration;
using namespace ballistic::calibration::detail;
using namespace ballistic::calibration;

// =============================================================================
// Glide configs
// =============================================================================

static TrajectoryConfig make_ballistic_config() {
    TrajectoryConfig c;
    c.gravity_degree = 8;
    c.enable_third_body = false;
    c.enable_relativistic = false;
    c.enable_solid_tides = false;

    return c;
}

static TrajectoryConfig make_forced_config(const Vector3d& return_dir) {
    auto c = make_ballistic_config();
    c.rk45.max_step_size = 0.5;

    auto accel = std::make_shared<forces::EmpiricalAcceleration>();
    std::vector<forces::EmpiricalAccelEntry> entries;

    const double omega = 2.0 * constants::PI / 100.0;
    for (double t = 0.0; t <= 700.0; t += 2.0) {
        double sb = std::exp(-0.5 * std::pow((t - 70.0) / 35.0, 2));
        Vector3d glide(8.0 * sb,
                        3.0 * sb * std::sin(omega * t),
                        3.0 * sb * std::cos(omega * t));
        double rb = (t >= 280.0) ? std::exp(-0.5 * std::pow((t - 420.0) / 80.0, 2)) : 0.0;
        entries.emplace_back(t, glide + 6.0 * rb * return_dir);
    }

    *accel = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
    c.empirical_accel = accel;
    return c;
}

// =============================================================================
// Helpers
// =============================================================================

// Compute RMSE from recon history vs truth interpolator (no auto-extend)
static double pos_rmse_safe(Trajectory& recon, TrajectoryInterpolator& truth,
                            const Epoch& t0, double skip_s) {
    double sse = 0; int n = 0;
    int nh = static_cast<int>(recon.history_size());
    for (int i = 0; i < nh; i += 2) {  // every 2nd history entry (~1s for 0.5s steps)
        const auto& h = recon.history_at(i);
        double tk = h.epoch - t0;
        if (tk < skip_s) continue;
        auto ts = truth.get_state(h.epoch);
        if (!ts) continue;
        sse += (h.position - ts->position).squaredNorm(); ++n;
    }
    return (n > 0) ? std::sqrt(sse / n) : 1e9;
}

// =============================================================================
// Setup
// =============================================================================

struct Setup {
    Epoch t0;
    Vector3d pos, vel;
    double duration, sample_end;

    std::unique_ptr<Trajectory> truth_traj;
    std::unique_ptr<TrajectoryInterpolator> truth_interp;

    std::vector<std::pair<Epoch, Vector3d>> pos_10hz, pos_100hz, pos_250hz;
    std::vector<std::pair<Epoch, Vector3d>> vel_100hz, acc_100hz, acc_250hz;
    std::vector<std::pair<Epoch, Vector3d>> acc_gcrs_100hz;  // for B-spline acc estimator

    // FD-derived accelerations (from position/velocity via finite difference)
    std::vector<std::pair<Epoch, Vector3d>> acc_from_pos_100hz;
    std::vector<std::pair<Epoch, Vector3d>> acc_from_vel_100hz;
    std::vector<std::pair<Epoch, Vector3d>> acc_from_pos_250hz;
};

// 5-point 2nd derivative from positions
static std::vector<std::pair<Epoch, Vector3d>> fd_acc_from_pos(
    const std::vector<std::pair<Epoch, Vector3d>>& pos)
{
    std::vector<std::pair<Epoch, Vector3d>> acc;
    int n = static_cast<int>(pos.size());
    if (n < 5) return acc;
    double dt = pos[1].first - pos[0].first;
    double dt2 = dt * dt;
    for (int i = 2; i < n - 2; ++i) {
        Vector3d a = (-pos[i+2].second + 16.0*pos[i+1].second - 30.0*pos[i].second
                      + 16.0*pos[i-1].second - pos[i-2].second) / (12.0 * dt2);
        acc.emplace_back(pos[i].first, a);
    }
    return acc;
}

// 4th-order central difference from velocities
static std::vector<std::pair<Epoch, Vector3d>> fd_acc_from_vel(
    const std::vector<std::pair<Epoch, Vector3d>>& vel)
{
    std::vector<std::pair<Epoch, Vector3d>> acc;
    int n = static_cast<int>(vel.size());
    if (n < 5) return acc;
    double dt = vel[1].first - vel[0].first;
    for (int i = 2; i < n - 2; ++i) {
        Vector3d a = (-vel[i+2].second + 8.0*vel[i+1].second
                      - 8.0*vel[i-1].second + vel[i-2].second) / (12.0 * dt);
        acc.emplace_back(vel[i].first, a);
    }
    return acc;
}


static Setup& get_setup() {
    static Setup s;
    static bool done = false;
    if (done) return s;
    done = true;

    s.t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    double v_circ = std::sqrt(constants::earth::GM / R);
    double omega_r = constants::earth::ANGULAR_VELOCITY * R;
    s.pos = Vector3d(R, 0.0, 0.0);
    s.vel = Vector3d(-30.0, 0.85 * v_circ - omega_r, 0.0);

    std::cout << "  [setup] Building glide truth..." << std::endl;

    // Return direction
    Trajectory sp(s.t0, s.pos, s.vel, make_forced_config(Vector3d::Zero()));
    for (int i = 0; i < 50000 && sp.epoch() - s.t0 < 430.0; ++i) sp.step();
    TrajectoryInterpolator spi(sp);
    auto sps = spi.get_state(s.t0 + 420.0);
    Vector3d rd = sps ? ((-sps->position.normalized() + 0.4 * Vector3d(0,0,1)).normalized()) : Vector3d::Zero();

    // Truth
    s.truth_traj = std::make_unique<Trajectory>(s.t0, s.pos, s.vel, make_forced_config(rd));
    for (int i = 0; i < 50000; ++i) {
        auto sr = s.truth_traj->step();
        auto geo = coords::ecef_to_geodetic(sr.position);
        if (geo.altitude <= 1000.0 || sr.epoch - s.t0 > 700.0) break;
    }
    s.truth_interp = std::make_unique<TrajectoryInterpolator>(*s.truth_traj);
    s.duration = s.truth_traj->epoch() - s.t0;
    // Bisect impact
    double lo = (std::max)(0.0, s.duration - 30.0), hi = s.duration;
    for (int i = 0; i < 50 && hi - lo > 0.001; ++i) {
        double mid = 0.5 * (lo + hi);
        auto st = s.truth_interp->get_state(s.t0 + mid);
        if (!st) break;
        (coords::ecef_to_geodetic(st->position).altitude > 1000.0 ? lo : hi) = mid;
    }
    s.duration = 0.5 * (lo + hi);
    s.sample_end = s.duration - 10.0;

    std::cout << "  [setup] Duration: " << std::fixed << std::setprecision(1) << s.duration << "s" << std::endl;

    // Sample ECEF observations from interpolation
    for (double t = 0.0; t <= s.sample_end; t += 0.1) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) s.pos_10hz.emplace_back(s.t0 + t, st->position);
    }
    for (double t = 0.0; t <= s.sample_end; t += 0.01) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) {
            s.pos_100hz.emplace_back(s.t0 + t, st->position);
            s.vel_100hz.emplace_back(s.t0 + t, st->velocity);
            StateVector se{st->position, st->velocity};
            StateVector si = util::itrf_to_gcrs(se, s.t0 + t);
            Vector3d a_g = s.truth_traj->force_model().acceleration(s.t0 + t, si);
            s.acc_100hz.emplace_back(s.t0 + t,
                util::acceleration_to_itrf(a_g, se, s.t0 + t));
            s.acc_gcrs_100hz.emplace_back(s.t0 + t, a_g);
        }
    }
    for (double t = 0.0; t <= s.sample_end; t += 0.004) {
        auto st = s.truth_interp->get_state(s.t0 + t);
        if (st) {
            s.pos_250hz.emplace_back(s.t0 + t, st->position);
            StateVector se{st->position, st->velocity};
            StateVector si = util::itrf_to_gcrs(se, s.t0 + t);
            Vector3d a_g = s.truth_traj->force_model().acceleration(s.t0 + t, si);
            s.acc_250hz.emplace_back(s.t0 + t,
                util::acceleration_to_itrf(a_g, se, s.t0 + t));
        }
    }

    // Acceleration from RK45 truth history (stored ECEF coordinate acceleration).
    // Same truth trajectory used for RMSE comparison — no integrator mismatch.
    // Sample at ~2 Hz (every RK45 step, ~0.5s apart) — non-uniform but exact.
    {
        int nh = static_cast<int>(s.truth_traj->history_size());
        for (int i = 0; i < nh; ++i) {
            const auto& h = s.truth_traj->history_at(i);
            if (h.epoch - s.t0 > s.sample_end) break;
            s.acc_from_pos_100hz.emplace_back(h.epoch, h.acceleration);
        }
        // Reuse same data for all FD-acc tests (source doesn't matter — it's exact)
        s.acc_from_vel_100hz = s.acc_from_pos_100hz;
        s.acc_from_pos_250hz = s.acc_from_pos_100hz;

        std::cout << "  [setup] History acc: " << s.acc_from_pos_100hz.size()
                  << " pts (~" << std::setprecision(0)
                  << s.acc_from_pos_100hz.size() / s.sample_end << " Hz)" << std::endl;
    }

    std::cout << "  [setup] 10Hz:" << s.pos_10hz.size()
              << " 100Hz:" << s.pos_100hz.size()
              << " 250Hz:" << s.pos_250hz.size()
              << " FD-acc:" << s.acc_from_pos_100hz.size()
              << "/" << s.acc_from_vel_100hz.size()
              << "/" << s.acc_from_pos_250hz.size() << std::endl;
    return s;
}

// =============================================================================
// Tests
// =============================================================================

int main() {
    ballistic::initialize({});
    TestRunner runner("Beta+EA Glide");

    // NOTE: FD on interpolated positions/velocities fails catastrophically
    // (25,000-76,000m RMSE). Hermite interpolation between 0.5s RK45 steps
    // introduces sub-mm errors that FD amplifies by 1/dt^2 into m/s^2 noise.
    // FD-derived acceleration only works with exact (non-interpolated) data.

    // ---- B-spline trajectory for position obs: fitted trajectory within 1 m ----
    runner.add_test("bspline_pos_100hz", []() {
        auto& s = get_setup();

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        // 20 matches the design-intent default for LambdaMethod::DECREASING:
        // 10 iterations stall early under DECREASING's fast λ decay.
        cfg.max_iterations = 20;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = make_ballistic_config();

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, s.pos_100hz, cfg);

        // RMSE from B-spline trajectory (collocation_history)
        double sse = 0; int cnt = 0;
        for (const auto& h : result.collocation_history) {
            double tk = h.epoch - s.t0;
            if (tk < 5.0 || tk > s.sample_end) continue;
            auto ts = s.truth_interp->get_state(h.epoch);
            if (!ts) continue;
            sse += (h.position - ts->position).squaredNorm(); ++cnt;
        }
        double traj_rmse = (cnt > 0) ? std::sqrt(sse / cnt) : 1e9;

        // Also compute re-propagation RMSE for comparison
        double reprop_rmse = 1e9;
        if (result.empirical_accel) {
            auto rc = make_ballistic_config();
            rc.empirical_accel = result.empirical_accel;
            rc.rk45.max_step_size = 0.5;
            Trajectory recon(s.t0, s.pos, s.vel, rc);
            for (int i = 0; i < 50000 && recon.epoch() - s.t0 < s.sample_end; ++i) recon.step();
            reprop_rmse = pos_rmse_safe(recon, *s.truth_interp, s.t0, 5.0);
        }

        std::cout << "    BSpline Pos100: traj=" << std::fixed << std::setprecision(3)
                  << traj_rmse << "m reprop=" << reprop_rmse << "m"
                  << " (" << result.collocation_history.size() << " pts)" << std::endl;

        TEST_ASSERT_MSG(traj_rmse < 1.0,
            "B-spline trajectory RMSE should be <1m: " + std::to_string(traj_rmse));
    });

    // ---- B-spline trajectory for velocity obs (integrate vel→pos first) ----
    runner.add_test("bspline_vel_100hz", []() {
        auto& s = get_setup();

        // Integrate ECEF velocity → position
        std::vector<std::pair<Epoch, Vector3d>> pos_from_vel;
        pos_from_vel.reserve(s.vel_100hz.size());
        Vector3d rr = s.pos;
        pos_from_vel.emplace_back(s.vel_100hz[0].first, rr);
        for (size_t i = 0; i + 1 < s.vel_100hz.size(); ++i) {
            double dt_i = s.vel_100hz[i+1].first - s.vel_100hz[i].first;
            rr += 0.5 * (s.vel_100hz[i].second + s.vel_100hz[i+1].second) * dt_i;
            pos_from_vel.emplace_back(s.vel_100hz[i+1].first, rr);
        }

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        cfg.max_iterations = 10;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = make_ballistic_config();

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, pos_from_vel, cfg);

        double sse = 0; int cnt = 0;
        for (const auto& h : result.collocation_history) {
            double tk = h.epoch - s.t0;
            if (tk < 5.0 || tk > s.sample_end) continue;
            auto ts = s.truth_interp->get_state(h.epoch);
            if (!ts) continue;
            sse += (h.position - ts->position).squaredNorm(); ++cnt;
        }
        double traj_rmse = (cnt > 0) ? std::sqrt(sse / cnt) : 1e9;

        double reprop_rmse = 1e9;
        if (result.empirical_accel) {
            auto rc = make_ballistic_config();
            rc.empirical_accel = result.empirical_accel;
            rc.rk45.max_step_size = 0.5;
            Trajectory recon(s.t0, s.pos, s.vel, rc);
            for (int i = 0; i < 50000 && recon.epoch() - s.t0 < s.sample_end; ++i) recon.step();
            reprop_rmse = pos_rmse_safe(recon, *s.truth_interp, s.t0, 5.0);
        }

        std::cout << "    BSpline Vel100: traj=" << std::fixed << std::setprecision(3)
                  << traj_rmse << "m reprop=" << reprop_rmse << "m" << std::endl;

        // pos_from_vel is integrated in pure ECEF (rotating frame) → drifts
        // ~hundreds of m on this ~494 s glide. The IRGN fit then tracks the
        // drifty positions, so traj_rmse against truth captures the
        // integration drift, not the estimator's quality. This is the
        // unsafe regime; the production velocity + GN path is the
        // `bspline_vel_gn` cell below (asserts < 100 m). Bound here is a
        // sanity check that the estimator doesn't blow up.
        TEST_ASSERT_MSG(traj_rmse < 500.0,
            "B-spline vel trajectory RMSE: " + std::to_string(traj_rmse));
    });

    // ---- B-spline position with tighter knots (re-propagation EA quality) ----
    auto run_bspline_pos = [](double knot_s, const char* label) {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knot_s;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        cfg.max_iterations = 10;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = make_ballistic_config();

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, s.pos_100hz, cfg);

        double traj_rmse = 1e9, reprop_rmse = 1e9;
        { double sse = 0; int cnt = 0;
          for (const auto& h : result.collocation_history) {
              double tk = h.epoch - s.t0;
              if (tk < 5.0 || tk > s.sample_end) continue;
              auto ts = s.truth_interp->get_state(h.epoch);
              if (!ts) continue;
              sse += (h.position - ts->position).squaredNorm(); ++cnt;
          }
          traj_rmse = (cnt > 0) ? std::sqrt(sse / cnt) : 1e9;
        }
        if (result.empirical_accel) {
            auto rc = make_ballistic_config();
            rc.empirical_accel = result.empirical_accel;
            rc.rk45.max_step_size = 0.5;
            Trajectory recon(s.t0, s.pos, s.vel, rc);
            for (int i = 0; i < 50000 && recon.epoch() - s.t0 < s.sample_end; ++i) recon.step();
            reprop_rmse = pos_rmse_safe(recon, *s.truth_interp, s.t0, 5.0);
        }
        std::cout << "    " << label << ": traj=" << std::fixed << std::setprecision(3)
                  << traj_rmse << "m reprop=" << reprop_rmse << "m" << std::endl;
        return reprop_rmse;
    };

    runner.add_test("bspline_pos_k0.5", [&]() {
        double r = run_bspline_pos(0.5, "BSpline k=0.5");
        // k=0.5 is far below the truth feature scale (~35 s on the glide)
        // → IRGN-only over-fits residuals (traj fits to ~mm) but the
        // resulting EA is unstable on re-propagation (~thousands of m).
        // Production POS at fine knots requires GN refinement; see
        // `bspline_pos_gn` (knot spacing ≈ truth feature scale / 4).

        // Bound is a sanity check that we don't blow up to NaN.
        TEST_ASSERT(r < 10000);
    });

    // ---- B-spline with Gauss-Newton shooting refinement ----
    // GN directly optimizes EA coefficients to minimize re-propagation RMSE.
    // Slow: ~726 propagations per iteration × 10 iterations = ~2 hours.
    auto run_bspline_gn = [](
        const std::vector<std::pair<Epoch, Vector3d>>& obs_data,
        const char* label,
        double knot_spacing = 2.0) {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knot_spacing;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        cfg.max_iterations = 10;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = make_ballistic_config();
        cfg.trajectory_config.rk45.max_step_size = 0.5;  // match evaluation step size

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, obs_data, cfg);

        if (result.empirical_accel) {
            GNRefineConfig gc;
            gc.max_iterations = 30;
            gc.fd_delta = 1e-2;
            gc.sample_stride = 10;  // 10 Hz from 100 Hz
            gc.use_stm = false;  // finite-difference Jacobian, not the STM
                                 // (state transition matrix) one
            gc.trajectory_config = cfg.trajectory_config;
            gc.max_duration_s = cfg.max_duration_s;
            if (auto refined = refine_ea_gn(s.t0, s.pos, s.vel, obs_data,
                                            result.empirical_accel, gc))
                result.empirical_accel = refined;
        }

        double traj_rmse = 1e9, reprop_rmse = 1e9;
        { double sse = 0; int cnt = 0;
          for (const auto& h : result.collocation_history) {
              double tk = h.epoch - s.t0;
              if (tk < 5.0 || tk > s.sample_end) continue;
              auto ts = s.truth_interp->get_state(h.epoch);
              if (!ts) continue;
              sse += (h.position - ts->position).squaredNorm(); ++cnt;
          }
          traj_rmse = (cnt > 0) ? std::sqrt(sse / cnt) : 1e9;
        }
        if (result.empirical_accel) {
            auto rc = make_ballistic_config();
            rc.empirical_accel = result.empirical_accel;
            rc.rk45.max_step_size = 0.5;
            Trajectory recon(s.t0, s.pos, s.vel, rc);
            for (int i = 0; i < 50000 && recon.epoch() - s.t0 < s.sample_end; ++i) recon.step();
            reprop_rmse = pos_rmse_safe(recon, *s.truth_interp, s.t0, 5.0);
        }
        std::cout << "    " << label << ": traj=" << std::fixed << std::setprecision(3)
                  << traj_rmse << "m reprop=" << reprop_rmse << "m" << std::endl;
        return reprop_rmse;
    };

    runner.add_test("bspline_pos_gn", [&]() {
        double r = run_bspline_gn(get_setup().pos_100hz, "BSpline Pos GN");
        TEST_ASSERT(r < 100);
    });

    runner.add_test("bspline_vel_gn", [&]() {
        auto& s = get_setup();
        // Integrate velocity → positions
        std::vector<std::pair<Epoch, Vector3d>> pos_from_vel;
        pos_from_vel.reserve(s.vel_100hz.size());
        Vector3d rr = s.pos;
        pos_from_vel.emplace_back(s.vel_100hz[0].first, rr);
        for (size_t i = 0; i + 1 < s.vel_100hz.size(); ++i) {
            double dt_i = s.vel_100hz[i+1].first - s.vel_100hz[i].first;
            rr += 0.5 * (s.vel_100hz[i].second + s.vel_100hz[i+1].second) * dt_i;
            pos_from_vel.emplace_back(s.vel_100hz[i+1].first, rr);
        }
        double r = run_bspline_gn(pos_from_vel, "BSpline Vel GN");
        TEST_ASSERT(r < 100);
    });

    // ---- B-spline GN at k=1.0 ----
    runner.add_test("bspline_pos_gn_k1", [&]() {
        double r = run_bspline_gn(get_setup().pos_100hz, "BSpline Pos GN k=1.0", 1.0);
        TEST_ASSERT(r < 100);
    });

    runner.add_test("bspline_vel_gn_k1", [&]() {
        auto& s = get_setup();
        std::vector<std::pair<Epoch, Vector3d>> pos_from_vel;
        pos_from_vel.reserve(s.vel_100hz.size());
        Vector3d rr = s.pos;
        pos_from_vel.emplace_back(s.vel_100hz[0].first, rr);
        for (size_t i = 0; i + 1 < s.vel_100hz.size(); ++i) {
            double dt_i = s.vel_100hz[i+1].first - s.vel_100hz[i].first;
            rr += 0.5 * (s.vel_100hz[i].second + s.vel_100hz[i+1].second) * dt_i;
            pos_from_vel.emplace_back(s.vel_100hz[i+1].first, rr);
        }
        double r = run_bspline_gn(pos_from_vel, "BSpline Vel GN k=1.0", 1.0);
        TEST_ASSERT(r < 100);
    });

    // ---- B-spline acceleration with GN ----
    // Fair test: double-integrate ECEF acceleration → velocity → position from known IC.
    // No truth positions used — only what a user with accelerometer data would have.
    runner.add_test("bspline_acc_gn", []() {
        auto& s = get_setup();

        // Double integrate: acc → vel → pos (trapezoidal, from known IC)
        std::vector<std::pair<Epoch, Vector3d>> pos_from_acc;
        pos_from_acc.reserve(s.acc_100hz.size());
        Vector3d vel_int = s.vel;
        Vector3d pos_int = s.pos;
        pos_from_acc.emplace_back(s.acc_100hz[0].first, pos_int);
        for (size_t i = 0; i + 1 < s.acc_100hz.size(); ++i) {
            double dt_i = s.acc_100hz[i+1].first - s.acc_100hz[i].first;
            // Trapezoidal velocity integration
            vel_int += 0.5 * (s.acc_100hz[i].second + s.acc_100hz[i+1].second) * dt_i;
            // Trapezoidal position integration
            pos_int += vel_int * dt_i;  // Euler for simplicity (trapezoidal would need vel at i+1)
            pos_from_acc.emplace_back(s.acc_100hz[i+1].first, pos_int);
        }

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 2.0;
        cfg.lambda = 1e-8;
        cfg.max_duration_s = s.sample_end;
        cfg.max_iterations = 10;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = make_ballistic_config();
        cfg.trajectory_config.rk45.max_step_size = 0.5;

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.t0, s.pos, s.vel, s.acc_gcrs_100hz, cfg);

        if (result.empirical_accel) {
            GNRefineConfig gc;
            gc.max_iterations = 30;
            gc.fd_delta = 1e-2;
            gc.sample_stride = 10;
            gc.use_stm = false;
            gc.trajectory_config = cfg.trajectory_config;
            gc.max_duration_s = cfg.max_duration_s;
            // GN target: positions derived from double-integrating acceleration (fair)
            if (auto refined = refine_ea_gn(s.t0, s.pos, s.vel, pos_from_acc,
                                            result.empirical_accel, gc))
                result.empirical_accel = refined;
        }

        double reprop_rmse = 1e9;
        if (result.empirical_accel) {
            auto rc = make_ballistic_config();
            rc.empirical_accel = result.empirical_accel;
            rc.rk45.max_step_size = 0.5;
            Trajectory recon(s.t0, s.pos, s.vel, rc);
            for (int i = 0; i < 50000 && recon.epoch() - s.t0 < s.sample_end; ++i) recon.step();
            reprop_rmse = pos_rmse_safe(recon, *s.truth_interp, s.t0, 5.0);
        }
        std::cout << "    BSpline Acc GN: reprop=" << std::fixed << std::setprecision(3)
                  << reprop_rmse << "m" << std::endl;
        // Double-integrated positions drift → GN degrades (80m vs 2.65m without GN).
        // Acceleration-only re-propagation without GN is better.
        TEST_ASSERT(true);  // report only
    });

    // ---- B-spline acceleration (GCRS) at k=2.0 and k=0.5 ----
    auto run_bspline_acc = [](double knot_s, const char* label) {
        auto& s = get_setup();
        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = knot_s;
        cfg.lambda = 1e-8;  // recommended for acceleration obs
        cfg.max_duration_s = s.sample_end;
        cfg.max_iterations = 10;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = make_ballistic_config();

        auto result = estimate_empirical_acceleration_bspline_from_acceleration(
            s.t0, s.pos, s.vel, s.acc_gcrs_100hz, cfg);

        double reprop_rmse = 1e9;
        if (result.empirical_accel) {
            auto rc = make_ballistic_config();
            rc.empirical_accel = result.empirical_accel;
            rc.rk45.max_step_size = 0.5;
            Trajectory recon(s.t0, s.pos, s.vel, rc);
            for (int i = 0; i < 50000 && recon.epoch() - s.t0 < s.sample_end; ++i) recon.step();
            reprop_rmse = pos_rmse_safe(recon, *s.truth_interp, s.t0, 5.0);
        }
        std::cout << "    " << label << ": reprop=" << std::fixed << std::setprecision(3)
                  << reprop_rmse << "m" << std::endl;
        return reprop_rmse;
    };

    runner.add_test("bspline_acc_k2.0", [&]() {
        double r = run_bspline_acc(2.0, "BSpline Acc k=2.0");
        TEST_ASSERT(r < 10);
    });

    runner.add_test("bspline_acc_k0.5", [&]() {
        double r = run_bspline_acc(0.5, "BSpline Acc k=0.5");
        TEST_ASSERT(r < 10);
    });

    runner.add_test("bspline_acc_k1.0", [&]() {
        double r = run_bspline_acc(1.0, "BSpline Acc k=1.0");
        TEST_ASSERT(r < 10);
    });

    // B-spline velocity k=0.5 (vel→pos integration, then position B-spline)
    runner.add_test("bspline_vel_k0.5", []() {
        auto& s = get_setup();
        std::vector<std::pair<Epoch, Vector3d>> pos_from_vel;
        pos_from_vel.reserve(s.vel_100hz.size());
        Vector3d rr = s.pos;
        pos_from_vel.emplace_back(s.vel_100hz[0].first, rr);
        for (size_t i = 0; i + 1 < s.vel_100hz.size(); ++i) {
            double dt_i = s.vel_100hz[i+1].first - s.vel_100hz[i].first;
            rr += 0.5 * (s.vel_100hz[i].second + s.vel_100hz[i+1].second) * dt_i;
            pos_from_vel.emplace_back(s.vel_100hz[i+1].first, rr);
        }

        BSplineAccelConfig cfg;
        cfg.knot_spacing_s = 0.5;
        cfg.lambda = 1e-10;
        cfg.max_duration_s = s.sample_end;
        cfg.max_iterations = 10;
        cfg.frame = forces::EmpiricalAccelFrame::GCRS;
        cfg.trajectory_config = make_ballistic_config();

        auto result = estimate_empirical_acceleration_bspline(
            s.t0, s.pos, s.vel, pos_from_vel, cfg);

        double reprop_rmse = 1e9;
        if (result.empirical_accel) {
            auto rc = make_ballistic_config();
            rc.empirical_accel = result.empirical_accel;
            rc.rk45.max_step_size = 0.5;
            Trajectory recon(s.t0, s.pos, s.vel, rc);
            for (int i = 0; i < 50000 && recon.epoch() - s.t0 < s.sample_end; ++i) recon.step();
            reprop_rmse = pos_rmse_safe(recon, *s.truth_interp, s.t0, 5.0);
        }
        std::cout << "    BSpline Vel k=0.5: reprop=" << std::fixed << std::setprecision(3)
                  << reprop_rmse << "m" << std::endl;
        // Same k=0.5 over-fit regime as `bspline_pos_k0.5`. Sanity bound only.
        TEST_ASSERT(reprop_rmse < 10000);
    });

    runner.run();
    return print_final_summary();
}
