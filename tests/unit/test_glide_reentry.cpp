// A forced reentry glide from 120 km, built from an empirical-acceleration (EA)
// table (an extra acceleration added to the force model):
//   glide segment (0-250 s): lift plus a rotating lateral force;
//   return segment (280 s on): a force steering back toward the ballistic
//   impact point.
// The cells check qualitative behavior with loose bounds: the trajectory
// reaches the ground in 100-700 s, climbs or weaves at least once, lands more
// than 0.5 km off the ballistic impact point (cross- and down-range), and the
// return force brings the impact closer to the ballistic one than the glide
// alone. Writes glide_reentry_data.csv for plotting.

#include "../test_utils.hpp"
#include <ballistic/init.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/trajectory_interpolator.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;

// =============================================================================
// Glide force parameters (glide segment: 0-250s)
// =============================================================================

constexpr double LIFT_PEAK = 8.0;          // GCRS x lift [m/s^2]
constexpr double LATERAL_AMP = 3.0;        // rotating lateral amplitude [m/s^2]
constexpr double FORCE_CENTER = 70.0;      // Gaussian center [s]
constexpr double FORCE_WIDTH = 35.0;       // Gaussian sigma [s]
constexpr double LATERAL_PERIOD = 100.0;   // rotation period [s]

// Return force parameters (return segment: 280 s on)
constexpr double RETURN_MAGNITUDE = 6.0;   // m/s^2 (adds to ~2.7 effective gravity)
constexpr double RETURN_CENTER = 420.0;    // Gaussian center [s]
constexpr double RETURN_WIDTH = 80.0;      // Gaussian sigma [s]

// =============================================================================
// Config builders
// =============================================================================

static TrajectoryConfig make_ballistic_config() {
    TrajectoryConfig c;
    c.gravity_degree = 8;
    c.enable_third_body = false;
    c.enable_relativistic = false;
    c.enable_solid_tides = false;

    return c;
}

// Build forced config with glide segment + return segment toward target_dir
static TrajectoryConfig make_forced_config(const Vector3d& return_dir_gcrs) {
    TrajectoryConfig c = make_ballistic_config();

    std::vector<forces::EmpiricalAccelEntry> entries;

    auto glide_bump = [](double t) {
        double x = (t - FORCE_CENTER) / FORCE_WIDTH;
        return std::exp(-0.5 * x * x);
    };

    auto return_bump = [](double t) {
        double x = (t - RETURN_CENTER) / RETURN_WIDTH;
        return std::exp(-0.5 * x * x);
    };

    const double omega = 2.0 * constants::PI / LATERAL_PERIOD;

    // Sample both phases at 2s intervals from 0 to 700s
    for (double t = 0.0; t <= 700.0; t += 2.0) {
        // Glide segment forces
        double sb = glide_bump(t);
        double lift = LIFT_PEAK * sb;
        double lat_y = LATERAL_AMP * sb * std::sin(omega * t);
        double lat_z = LATERAL_AMP * sb * std::cos(omega * t);
        Vector3d glide_force(lift, lat_y, lat_z);

        // Return segment: force toward ballistic impact (switched on at 280 s)
        double rb = (t >= 280.0) ? return_bump(t) : 0.0;
        Vector3d return_force = RETURN_MAGNITUDE * rb * return_dir_gcrs;

        entries.emplace_back(t, glide_force + return_force);
    }

    c.empirical_accel = std::make_shared<forces::EmpiricalAcceleration>(
        forces::EmpiricalAcceleration::from_entries(
            std::move(entries), forces::EmpiricalAccelFrame::GCRS));
    return c;
}

// =============================================================================
// Helpers
// =============================================================================

struct ENU_Context {
    Vector3d launch_pos;
    double launch_lat;
    double launch_lon;
    Vector3d launch_vel;
};

static void compute_downrange_crossrange(
    const Vector3d& pos_ecef, const ENU_Context& ctx,
    double& downrange_km, double& crossrange_km)
{
    Matrix3d R_enu = coords::ecef_to_enu_rotation(ctx.launch_lat, ctx.launch_lon);
    Vector3d delta_enu = R_enu * (pos_ecef - ctx.launch_pos);

    Vector3d vel_enu = R_enu * ctx.launch_vel;
    Vector3d vel_horiz(vel_enu.x(), vel_enu.y(), 0.0);
    if (vel_horiz.norm() > 1e-6) vel_horiz.normalize();

    Vector3d delta_horiz(delta_enu.x(), delta_enu.y(), 0.0);
    downrange_km = delta_horiz.dot(vel_horiz) / 1000.0;
    Vector3d cross_vec = delta_horiz - delta_horiz.dot(vel_horiz) * vel_horiz;
    crossrange_km = cross_vec.norm() / 1000.0;
}

static double signed_crossrange_km(
    const Vector3d& pos_ecef, const ENU_Context& ctx)
{
    Matrix3d R_enu = coords::ecef_to_enu_rotation(ctx.launch_lat, ctx.launch_lon);
    Vector3d delta_enu = R_enu * (pos_ecef - ctx.launch_pos);

    Vector3d vel_enu = R_enu * ctx.launch_vel;
    Vector3d vel_horiz(vel_enu.x(), vel_enu.y(), 0.0);
    if (vel_horiz.norm() > 1e-6) vel_horiz.normalize();

    Vector3d cross_unit(-vel_horiz.y(), vel_horiz.x(), 0.0);
    Vector3d delta_horiz(delta_enu.x(), delta_enu.y(), 0.0);
    return delta_horiz.dot(cross_unit) / 1000.0;
}

// Propagate to ground (alt < 1km) or timeout, return duration and whether ground was reached
static std::pair<double, bool> propagate_to_ground(
    Trajectory& traj, const Epoch& t0, double timeout_s)
{
    for (int step = 0; step < 50000; ++step) {
        auto sr = traj.step();
        auto geo = coords::ecef_to_geodetic(sr.position);
        double dt = sr.epoch - t0;
        if (geo.altitude <= 1000.0) return {dt, true};
        if (dt > timeout_s) return {dt, false};
    }
    return {0.0, false};
}

// Bisect for precise alt=1km crossing using interpolator
static double bisect_impact(
    TrajectoryInterpolator& interp, const Epoch& t0, double duration)
{
    double t_lo = (std::max)(0.0, duration - 30.0);
    double t_hi = duration;
    for (int iter = 0; iter < 50; ++iter) {
        double t_mid = 0.5 * (t_lo + t_hi);
        auto state = interp.get_state(t0 + t_mid);
        if (!state) break;
        auto geo = coords::ecef_to_geodetic(state->position);
        if (geo.altitude > 1000.0) t_lo = t_mid;
        else t_hi = t_mid;
        if (t_hi - t_lo < 0.001) break;
    }
    return 0.5 * (t_lo + t_hi);
}

// =============================================================================
// Main test
// =============================================================================

int main() {
    ballistic::initialize({});

    TestRunner runner("Gliding Glide Reentry (120 km)");

    // Initial conditions: 120 km, sub-orbital, shallow descent
    Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    double R = constants::earth::RADIUS_EQUATORIAL + 120000.0;
    double v_circ = std::sqrt(constants::earth::GM / R);
    double omega_r = constants::earth::ANGULAR_VELOCITY * R;
    double v_tangential = 0.85 * v_circ - omega_r;
    double v_radial = -30.0;

    Vector3d pos(R, 0.0, 0.0);
    Vector3d vel(v_radial, v_tangential, 0.0);

    auto launch_geo = coords::ecef_to_geodetic(pos);
    ENU_Context enu{pos, launch_geo.latitude, launch_geo.longitude, vel};

    // =========================================================================
    // Step 1: Propagate ballistic trajectory → get impact position
    // =========================================================================
    Trajectory ball_traj(t0, pos, vel, make_ballistic_config());
    auto [ball_duration, ball_ok] = propagate_to_ground(ball_traj, t0, 600.0);
    TrajectoryInterpolator ball_interp(ball_traj);
    if (ball_ok) ball_duration = bisect_impact(ball_interp, t0, ball_duration);
    auto ball_impact = ball_interp.get_state(t0 + ball_duration);

    std::cout << "  Ballistic duration: " << std::fixed << std::setprecision(1)
              << ball_duration << "s" << std::endl;

    // =========================================================================
    // Step 2: Quick glide-only propagation → get position at return center
    // =========================================================================
    Trajectory glide_only(t0, pos, vel, make_forced_config(Vector3d::Zero()));
    for (int step = 0; step < 50000; ++step) {
        auto sr = glide_only.step();
        if ((sr.epoch - t0) >= RETURN_CENTER + 10.0) break;
    }
    TrajectoryInterpolator glide_interp(glide_only);
    auto glide_at_peak = glide_interp.get_state(t0 + RETURN_CENTER);

    // =========================================================================
    // Step 3: Compute return direction from velocity/position at peak time
    // Decelerate (oppose velocity) + descend (push toward Earth center)
    // Both vectors are Earth-fixed (ECEF); over this short arc that is close
    // enough to the inertial frame (GCRS) for a steering direction.
    // =========================================================================
    Vector3d return_dir = Vector3d::Zero();
    if (glide_at_peak) {
        // Descent: push toward Earth center
        Vector3d descent = -glide_at_peak->position.normalized();
        // Northward: push latitude back toward ballistic (equatorial) path
        // +z in GCRS ≈ celestial north ≈ geographic north for equatorial orbit
        Vector3d northward(0.0, 0.0, 1.0);
        return_dir = (descent + 0.4 * northward).normalized();
        std::cout << "  Return direction: descent + northward at t=" << RETURN_CENTER << "s"
                  << std::endl;
    }

    // =========================================================================
    // Step 4: Propagate full forced trajectory (glide + return)
    // =========================================================================
    Trajectory forced_traj(t0, pos, vel, make_forced_config(return_dir));
    auto [forced_duration, forced_ok] = propagate_to_ground(forced_traj, t0, 700.0);
    TrajectoryInterpolator forced_interp(forced_traj);
    if (forced_ok) forced_duration = bisect_impact(forced_interp, t0, forced_duration);
    auto forced_impact = forced_interp.get_state(t0 + forced_duration);

    std::cout << "  Forced duration: " << std::fixed << std::setprecision(1)
              << forced_duration << "s (glide+return)" << std::endl;

    // =========================================================================
    // Tests
    // =========================================================================

    runner.add_test("impact_reached", [&]() {
        TEST_ASSERT_MSG(forced_ok,
            "Forced trajectory did not reach ground within 700s");
    });

    runner.add_test("flight_duration_range", [&]() {
        TEST_ASSERT_MSG(forced_duration > 100.0 && forced_duration < 700.0,
            "Flight duration " + std::to_string(forced_duration) + "s outside [100, 700]");
    });

    runner.add_test("glide_detected", [&]() {
        bool glide = false;
        double prev_alt = 1e9;
        bool past_start = false;
        double prev_rate = -1e9;
        int rate_reversals = 0;

        for (double t = 0.0; t <= forced_duration; t += 1.0) {
            auto state = forced_interp.get_state(t0 + t);
            if (!state) continue;
            auto geo = coords::ecef_to_geodetic(state->position);
            double alt_km = geo.altitude / 1000.0;

            if (alt_km < 119.0) past_start = true;
            if (past_start) {
                double rate = alt_km - prev_alt;
                if (alt_km > prev_alt + 0.001 && alt_km < 120.0) glide = true;
                if (prev_rate < -0.001 && rate > 0.001) rate_reversals++;
                prev_rate = rate;
            }
            prev_alt = alt_km;
        }
        std::cout << "    Rate reversals: " << rate_reversals
                  << ", glide: " << (glide ? "yes" : "no") << std::endl;
        TEST_ASSERT_MSG(glide || rate_reversals > 0,
            "No glide evidence: rate_reversals=" + std::to_string(rate_reversals));
    });

    runner.add_test("glide_detected", [&]() {
        int sign_changes = 0;
        double prev_cr = 0.0;
        bool first = true;
        double max_cr = 0.0, min_cr = 0.0;

        for (double t = 0.0; t <= forced_duration; t += 1.0) {
            auto state = forced_interp.get_state(t0 + t);
            if (!state) continue;
            double cr = signed_crossrange_km(state->position, enu);
            if (cr > max_cr) max_cr = cr;
            if (cr < min_cr) min_cr = cr;
            if (!first && prev_cr * cr < 0.0) ++sign_changes;
            prev_cr = cr;
            first = false;
        }
        double cr_range = max_cr - min_cr;
        std::cout << "    Cross-range sign changes: " << sign_changes
                  << ", span: " << std::fixed << std::setprecision(2) << cr_range << " km"
                  << std::endl;
        TEST_ASSERT_MSG(sign_changes >= 1 || cr_range > 5.0,
            "No glide: " + std::to_string(sign_changes) + " sign changes");
    });

    runner.add_test("cross_range_deflection", [&]() {
        TEST_ASSERT(forced_impact.has_value());
        double dr_km, cr_km;
        compute_downrange_crossrange(forced_impact->position, enu, dr_km, cr_km);
        std::cout << "    Cross-range at impact: " << std::fixed << std::setprecision(2)
                  << cr_km << " km" << std::endl;
        TEST_ASSERT_MSG(cr_km > 0.5,
            "Cross-range " + std::to_string(cr_km) + " km < 0.5 km");
    });

    runner.add_test("downrange_deflection", [&]() {
        TEST_ASSERT(forced_impact.has_value() && ball_impact.has_value());
        double dr_forced, cr_forced, dr_ball, cr_ball;
        compute_downrange_crossrange(forced_impact->position, enu, dr_forced, cr_forced);
        compute_downrange_crossrange(ball_impact->position, enu, dr_ball, cr_ball);
        double diff = std::abs(dr_forced - dr_ball);
        std::cout << "    Downrange diff: " << std::fixed << std::setprecision(2)
                  << diff << " km (forced=" << dr_forced << ", ballistic=" << dr_ball << ")"
                  << std::endl;
        TEST_ASSERT_MSG(diff > 0.5,
            "Downrange deflection " + std::to_string(diff) + " km < 0.5 km");
    });

    runner.add_test("return_reduces_separation", [&]() {
        // The return segment should bring forced impact closer to ballistic impact
        // than the glide-only trajectory would
        TEST_ASSERT(forced_impact.has_value() && ball_impact.has_value());

        // Glide-only impact separation
        Trajectory glide_full(t0, pos, vel, make_forced_config(Vector3d::Zero()));
        auto [sp_dur, sp_ok] = propagate_to_ground(glide_full, t0, 700.0);
        TrajectoryInterpolator sp_interp(glide_full);
        if (sp_ok) sp_dur = bisect_impact(sp_interp, t0, sp_dur);
        auto sp_impact = sp_interp.get_state(t0 + sp_dur);

        double sep_glide = (sp_impact->position - ball_impact->position).norm() / 1000.0;
        double sep_return = (forced_impact->position - ball_impact->position).norm() / 1000.0;

        std::cout << "    Glide-only separation: " << std::fixed << std::setprecision(1)
                  << sep_glide << " km" << std::endl;
        std::cout << "    With return separation:  " << sep_return << " km" << std::endl;
        std::cout << "    Reduction: " << std::setprecision(1)
                  << ((1.0 - sep_return / sep_glide) * 100.0) << "%" << std::endl;

        TEST_ASSERT_MSG(sep_return < sep_glide,
            "Return segment did not reduce separation: glide=" + std::to_string(sep_glide)

            + " km, return=" + std::to_string(sep_return) + " km");
    });

    runner.add_test("write_csv", [&]() {
        double t_max = (std::max)(forced_duration, ball_duration);

        std::ofstream csv("glide_reentry_data.csv");
        csv << "time_s,forced_alt_km,forced_lat_deg,forced_lon_deg,"
            << "forced_downrange_km,forced_crossrange_km,"
            << "ballistic_alt_km,ballistic_lat_deg,ballistic_lon_deg\n";
        csv << std::fixed << std::setprecision(6);

        int rows = 0;
        for (double t = 0.0; t <= t_max + 1.0; t += 1.0) {
            auto sf = forced_interp.get_state(t0 + t);
            auto sb = ball_interp.get_state(t0 + t);
            if (!sf || !sb) continue;

            auto gf = coords::ecef_to_geodetic(sf->position);
            auto gb = coords::ecef_to_geodetic(sb->position);

            double dr_km, cr_km;
            compute_downrange_crossrange(sf->position, enu, dr_km, cr_km);

            csv << t << ","
                << gf.altitude / 1000.0 << ","
                << gf.latitude * constants::RAD_TO_DEG << ","
                << gf.longitude * constants::RAD_TO_DEG << ","
                << dr_km << ","
                << cr_km << ","
                << gb.altitude / 1000.0 << ","
                << gb.latitude * constants::RAD_TO_DEG << ","
                << gb.longitude * constants::RAD_TO_DEG << "\n";
            ++rows;
        }

        csv.close();
        std::cout << "    Wrote glide_reentry_data.csv (" << rows << " rows)" << std::endl;
        TEST_ASSERT(rows > 50);
    });

    runner.run();
    return print_final_summary();
}
