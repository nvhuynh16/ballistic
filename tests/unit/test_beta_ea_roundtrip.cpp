// compose_ea_from_beta() and its inverse decompose_to_beta(): turning a ballistic
// coefficient beta = m/(Cd*A) [kg/m^2] into an empirical acceleration (EA, here a
// tabulated drag acceleration) and recovering beta from it. Uses a simple
// exponential atmosphere (no data files). Tolerances: 1-5%, the error of the
// cubic-spline interpolation of the EA table.

#include "../test_utils.hpp"
#include <ballistic/init.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/calibration/ea_decomposition.hpp>
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/coordinates/geodetic.hpp>
#include <cmath>

using namespace ballistic;
using namespace ballistic::calibration;
using namespace ballistic::calibration::detail;  // compose/decompose are internal (detail::)
using namespace ballistic::test;

// Helper: wrap trajectory history + EA into EAResult for decompose_to_beta
static EAResult make_ea_result(
    const std::vector<HistoryEntry>& history,
    const std::shared_ptr<forces::EmpiricalAcceleration>& ea) {
    EAResult r;
    r.empirical_accel = ea;
    r.collocation_history = history;
    return r;
}

static TrajectoryConfig make_config() {
    TrajectoryConfig cfg;
    cfg.gravity_degree = 8;
    cfg.enable_third_body = false;
    cfg.enable_relativistic = false;
    return cfg;
}

// Simple exponential density model for testing (no external data needed)
static double test_density(double alt_km, double /*lat*/, double /*lon*/, const Epoch& /*ep*/) {
    // rho = rho0 * exp(-(h - h0) / H), scale height H=60 km, rho0 at 200km
    double rho0 = 3e-10;  // kg/m³ at 200 km
    double H = 60.0;      // scale height km
    return rho0 * std::exp(-(alt_km - 200.0) / H);
}

int main() {
    ballistic::initialize({});

    TestRunner runner("Beta-EA Round-Trip");

    // =========================================================================
    // Test 1: Constant beta → compose EA → verify magnitude
    // =========================================================================
    runner.add_test("constant_beta_ea_magnitude", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
        double vc = std::sqrt(constants::earth::GM / R);
        Vector3d pos(R, 0, 0), vel(0, vc, 0);

        // Propagate to get trajectory history
        Trajectory traj(t0, pos, vel, make_config());
        while (traj.epoch() - t0 < 300.0) traj.step();

        double beta_const = 256.4;  // a representative LEO value [kg/m^2]
        auto ea = compose_ea_from_beta(
            traj.history(), t0,
            [=](double) { return beta_const; },
            test_density);

        // Verify EA magnitude at each history point
        for (const auto& h : traj.history()) {
            double t_k = h.epoch - t0;
            auto geo = coords::ecef_to_geodetic(h.position);
            double alt_km = geo.altitude / 1000.0;
            double rho = test_density(alt_km, 0, 0, h.epoch);
            double v_mag = h.velocity.norm();

            double expected_mag = rho * v_mag * v_mag / (2.0 * beta_const);
            Vector3d ea_val = ea->accel_at((h.epoch) - (t0));
            double actual_mag = ea_val.norm();

            // Allow 1% tolerance (cubic spline interpolation of table)
            if (expected_mag > 1e-15) {
                TEST_ASSERT_NEAR(actual_mag, expected_mag, expected_mag * 0.01);
            }
        }
    });

    // =========================================================================
    // Test 2: EA opposes velocity direction
    // =========================================================================
    runner.add_test("ea_opposes_velocity", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
        double vc = std::sqrt(constants::earth::GM / R);
        Vector3d pos(R, 0, 0), vel(0, vc, 0);

        Trajectory traj(t0, pos, vel, make_config());
        while (traj.epoch() - t0 < 300.0) traj.step();

        // Use ITRF frame so EA is in same frame as h.velocity (ECEF)
        auto ea = compose_ea_from_beta(
            traj.history(), t0,
            [](double) { return 100.0; },
            test_density,
            forces::EmpiricalAccelFrame::ITRF);

        // EA (ITRF) should be anti-parallel to velocity (ECEF ≈ ITRF) at each point
        for (const auto& h : traj.history()) {
            Vector3d ea_val = ea->accel_at((h.epoch) - (t0));
            if (ea_val.norm() < 1e-20) continue;

            double cos_angle = ea_val.dot(h.velocity) / (ea_val.norm() * h.velocity.norm());
            // cos_angle should be -1 (anti-parallel)
            TEST_ASSERT_NEAR(cos_angle, -1.0, 0.01);
        }
    });

    // =========================================================================
    // Test 3: Round-trip: compose → decompose → recover beta
    // =========================================================================
    runner.add_test("roundtrip_compose_decompose", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
        double vc = std::sqrt(constants::earth::GM / R);
        Vector3d pos(R, 0, 0), vel(0, vc, 0);

        Trajectory traj(t0, pos, vel, make_config());
        while (traj.epoch() - t0 < 300.0) traj.step();

        double beta_input = 150.0;

        // Step 1: compose EA from constant beta
        auto ea = compose_ea_from_beta(
            traj.history(), t0,
            [=](double) { return beta_input; },
            test_density);

        // Step 2: decompose back to beta
        auto beta_result = decompose_to_beta(
            make_ea_result(traj.history(), ea), t0, test_density);

        // Recovered beta should match input
        TEST_ASSERT(beta_result.steps_solved > 0);
        // Median beta within 5% (cubic spline interpolation introduces small error)
        TEST_ASSERT_NEAR(beta_result.constant_beta, beta_input, beta_input * 0.05);

        // Check per-point beta recovery
        for (const auto& est : beta_result.step_estimates) {
            TEST_ASSERT_NEAR(est.beta, beta_input, beta_input * 0.05);
        }
    });

    // =========================================================================
    // Test 5: Frame consistency — GCRS EA round-trips correctly
    // =========================================================================
    runner.add_test("frame_consistency_gcrs", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
        double vc = std::sqrt(constants::earth::GM / R);
        Vector3d pos(R, 0, 0), vel(0, vc, 0);

        Trajectory traj(t0, pos, vel, make_config());
        while (traj.epoch() - t0 < 300.0) traj.step();

        double beta_input = 200.0;

        // Compose in GCRS frame (default)
        auto ea_gcrs = compose_ea_from_beta(
            traj.history(), t0,
            [=](double) { return beta_input; },
            test_density,
            forces::EmpiricalAccelFrame::GCRS);

        // Compose in ITRF frame
        auto ea_itrf = compose_ea_from_beta(
            traj.history(), t0,
            [=](double) { return beta_input; },
            test_density,
            forces::EmpiricalAccelFrame::ITRF);

        // At a non-zero time (where GCRS ≠ ECEF due to Earth rotation),
        // the GCRS and ITRF vectors should differ in direction
        // but decompose_to_beta should recover the same beta from both
        auto beta_from_gcrs = decompose_to_beta(
            make_ea_result(traj.history(), ea_gcrs), t0, test_density);
        auto beta_from_itrf = decompose_to_beta(
            make_ea_result(traj.history(), ea_itrf), t0, test_density);

        TEST_ASSERT(beta_from_gcrs.steps_solved > 0);
        TEST_ASSERT(beta_from_itrf.steps_solved > 0);

        // Both should recover the same beta (frame-independent physics)
        TEST_ASSERT_NEAR(beta_from_gcrs.constant_beta, beta_input, beta_input * 0.05);
        TEST_ASSERT_NEAR(beta_from_itrf.constant_beta, beta_input, beta_input * 0.05);
        TEST_ASSERT_NEAR(beta_from_gcrs.constant_beta, beta_from_itrf.constant_beta,
                         beta_input * 0.01);
    });

    // =========================================================================
    // Test 6: GCRS and ITRF EA vectors differ at non-zero time
    // =========================================================================
    runner.add_test("gcrs_itrf_vectors_differ", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        double R = constants::earth::RADIUS_EQUATORIAL + 400000.0;
        double vc = std::sqrt(constants::earth::GM / R);
        Vector3d pos(R, 0, 0), vel(0, vc, 0);

        Trajectory traj(t0, pos, vel, make_config());
        while (traj.epoch() - t0 < 300.0) traj.step();

        auto ea_gcrs = compose_ea_from_beta(
            traj.history(), t0,
            [](double) { return 100.0; },
            test_density,
            forces::EmpiricalAccelFrame::GCRS);

        auto ea_itrf = compose_ea_from_beta(
            traj.history(), t0,
            [](double) { return 100.0; },
            test_density,
            forces::EmpiricalAccelFrame::ITRF);

        // At t=150s, Earth has rotated ~0.06° — GCRS and ITRF vectors should differ
        Epoch t_mid = t0 + 150.0;
        Vector3d v_gcrs = ea_gcrs->accel_at((t_mid) - (t0));
        Vector3d v_itrf = ea_itrf->accel_at((t_mid) - (t0));

        // Magnitudes should be nearly equal (same physics, different frame)
        TEST_ASSERT_NEAR(v_gcrs.norm(), v_itrf.norm(), v_gcrs.norm() * 0.02);

        // But directions should differ (GCRS vs ITRF rotation)
        double cos_angle = v_gcrs.dot(v_itrf) / (v_gcrs.norm() * v_itrf.norm());
        // Should NOT be exactly 1.0 (that would mean no frame rotation applied)
        TEST_ASSERT(cos_angle < 0.9999);
    });

    // =========================================================================
    // Test 7: Empty collocation_history returns error
    // =========================================================================
    runner.add_test("empty_history_error", []() {
        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        // Create result with EA but empty history
        EAResult empty_result;
        auto ea = std::make_shared<forces::EmpiricalAcceleration>();
        std::vector<forces::EmpiricalAccelEntry> entries;
        entries.emplace_back(0.0, Vector3d(1e-5, 0, 0));
        entries.emplace_back(100.0, Vector3d(1e-5, 0, 0));
        *ea = forces::EmpiricalAcceleration::from_entries(std::move(entries), forces::EmpiricalAccelFrame::GCRS);
        empty_result.empirical_accel = ea;
        // collocation_history left empty

        auto beta_result = decompose_to_beta(empty_result, t0, test_density);
        TEST_ASSERT(!beta_result.error_message.empty());
        TEST_ASSERT(beta_result.steps_solved == 0);
    });

    runner.run();
    return print_final_summary();
}
