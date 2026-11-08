// =============================================================================
// Test: ForceModel.acceleration_with_jacobian gravity Hessian wire-up
// =============================================================================
//
// Checks that ForceModel::acceleration_with_jacobian, with only Earth gravity
// enabled, returns a 3×3 position Jacobian ∂a/∂r (the gravity Hessian, second
// derivatives of the potential) that matches the gravity engine's
// compute_with_hessian once rotated from the Earth-fixed frame (ITRF, also called
// ECEF) to the inertial frame (GCRS, called ECI below) as R·H·Rᵀ.
//
// Errors in this wiring (wrong rotation, wrong sign, wrong matrix order) would
// show up as differences from the expected ∂a/∂r. Physical checks used: a
// point-mass closed form, Laplace's equation (the Hessian's trace is zero outside
// the Earth), symmetry, and a zero velocity block (gravity does not depend on v).
// =============================================================================

#include "../test_utils.hpp"
#include "j2_hessian_closed_form.hpp"
#include "../../include/ballistic/forces/force_model.hpp"
#include "../../include/ballistic/init.hpp"

using namespace ballistic;
using namespace ballistic::test;

int main() {
    TestRunner runner("ForceModel gravity Hessian wire-up (Stage I)");

    runner.add_test("Point-mass gravity Jacobian: closed-form match", [&]() {
        // With gravity_max_degree = 1, ForceModel falls into the
        // closed-form point-mass path. Verify ∂a/∂r = -GM/r³·I + 3·GM/r⁵·r·rᵀ.
        auto fm = forces::ForceModel{};
        fm.set_gravity_max_degree(1);
        forces::ForceComponentFlags flags{};
        flags.earth_gravity = true;
        flags.third_body = false;
        flags.drag = false;
        flags.radiation = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.empirical = false;
        fm.set_force_flags(flags);

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6.778e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);
        StateVector state{pos, vel};

        auto [a, J] = fm.acceleration_with_jacobian(epoch, state);

        // Expected: a = -GM/r³ · r ;  J = -GM/r³·I + 3·GM/r⁵·r·rᵀ (3x6 with v-block zero).
        const double GM = constants::earth::GM;
        const double r_norm = pos.norm();
        const double r3 = r_norm * r_norm * r_norm;
        const double r5 = r3 * r_norm * r_norm;

        const Vector3d a_expected = -GM / r3 * pos;
        const Eigen::Matrix3d J_pos_expected =
            -GM / r3 * Eigen::Matrix3d::Identity()
            + 3.0 * GM / r5 * (pos * pos.transpose());

        // Acceleration: must match analytic point mass
        for (int i = 0; i < 3; ++i) {
            const double rel = std::abs(a(i) - a_expected(i)) / (std::abs(a_expected(i)) + 1e-30);
            TEST_ASSERT(rel < 1e-12);
        }

        // Position Jacobian (J[:3, :3]) must match analytic point mass
        const Eigen::Matrix3d J_pos = J.block<3, 3>(0, 0);
        const double scale = J_pos_expected.norm();
        const double diff = (J_pos - J_pos_expected).norm() / scale;
        TEST_ASSERT(diff < 1e-12);

        // Velocity Jacobian (J[:3, 3:]) must be zero (gravity has no v dependence)
        const Eigen::Matrix3d J_vel = J.block<3, 3>(0, 3);
        TEST_ASSERT(J_vel.norm() < 1e-15);
    });

    runner.add_test("Gravity Jacobian: Laplace equation in ECI", [&]() {
        // ∇²V = 0 outside the source ⇒ tr(∂a/∂r) = 0 in any frame (a trace is
        // unchanged by rotation).
        ballistic::initialize({});
        auto fm = forces::ForceModel{};
        fm.set_gravity_max_degree(20);  // higher degree for more interesting Hessian
        forces::ForceComponentFlags flags{};
        flags.earth_gravity = true;
        flags.third_body = false;
        flags.drag = false;
        flags.radiation = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.empirical = false;
        fm.set_force_flags(flags);

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6.778e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);
        StateVector state{pos, vel};

        auto [a, J] = fm.acceleration_with_jacobian(epoch, state);
        const Eigen::Matrix3d J_pos = J.block<3, 3>(0, 0);
        const double trace = J_pos.trace();
        const double scale = J_pos.norm();
        const double rel = std::abs(trace) / scale;
        // The Hessian is analytic (compute_with_hessian), so only rounding in
        // the recurrence and the rotation remains; the 1e-5 gate is loose.
        TEST_ASSERT(rel < 1e-5);
    });

    runner.add_test("Gravity Jacobian: symmetric in ECI", [&]() {
        // A symmetric Hessian stays symmetric under R·H·Rᵀ (R orthogonal).
        ballistic::initialize({});
        auto fm = forces::ForceModel{};
        fm.set_gravity_max_degree(20);
        forces::ForceComponentFlags flags{};
        flags.earth_gravity = true;
        flags.third_body = false;
        flags.drag = false;
        flags.radiation = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.empirical = false;
        fm.set_force_flags(flags);

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6.778e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);
        StateVector state{pos, vel};

        auto [a, J] = fm.acceleration_with_jacobian(epoch, state);
        const Eigen::Matrix3d J_pos = J.block<3, 3>(0, 0);
        const double scale = J_pos.norm();
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                const double rel = std::abs(J_pos(i, j) - J_pos(j, i)) / scale;
                // compute_with_hessian symmetrizes and R is orthogonal; only
                // rounding in R remains (gate 1e-10).
                TEST_ASSERT(rel < 1e-10);
            }
        }
    });

    runner.add_test("Velocity Jacobian: zero (gravity has no v dependence)", [&]() {
        ballistic::initialize({});
        auto fm = forces::ForceModel{};
        fm.set_gravity_max_degree(20);
        forces::ForceComponentFlags flags{};
        flags.earth_gravity = true;
        flags.third_body = false;
        flags.drag = false;
        flags.radiation = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.empirical = false;
        fm.set_force_flags(flags);

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6.778e6, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);
        StateVector state{pos, vel};

        auto [a, J] = fm.acceleration_with_jacobian(epoch, state);
        const Eigen::Matrix3d J_vel = J.block<3, 3>(0, 3);
        TEST_ASSERT(J_vel.norm() < 1e-15);
    });

    // ========================================================================
    // Mutation coverage: tests that would catch specific implementation mistakes.
    // ========================================================================

    // Mutation 4: Wrong rotation direction (R vs Rᵀ) for the ECEF→ECI Hessian transform.
    // This mutation would NOT be caught by Laplace (frame-invariant) or symmetry
    // (preserved by orthogonal similarity regardless of direction). It WOULD be
    // caught by comparing the rotated Hessian against an independent computation
    // path: the degree-2 Hessian evaluated in ECEF, then transformed via the SAME
    // R, vs the ForceModel output.
    runner.add_test("Mutation 4: ECEF→ECI rotation direction (R, not Rᵀ)", [&]() {
        ballistic::initialize({});
        auto fm = forces::ForceModel{};
        fm.set_gravity_max_degree(2);  // J2 only
        forces::ForceComponentFlags flags{};
        flags.earth_gravity = true;
        flags.third_body = false;
        flags.drag = false;
        flags.radiation = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.empirical = false;
        fm.set_force_flags(flags);

        // An epoch where Earth's rotation angle is far from 0, so R is non-trivial.
        Epoch epoch = Epoch::from_utc(2024, 6, 15, 18, 30, 0);
        // Off-equator, off-prime-meridian ECI position (so rotation matters in
        // both axes).
        Vector3d pos_eci(4.0e6, 3.0e6, 4.5e6);
        Vector3d vel(0.0, 7500.0, 0.0);
        StateVector state{pos_eci, vel};

        // ForceModel path (analytic Hessian + ECEF→ECI similarity transform)
        auto [a_engine, J_engine] = fm.acceleration_with_jacobian(epoch, state);
        const Eigen::Matrix3d H_eci_engine = J_engine.block<3, 3>(0, 0);

        // Independent path: compute in ECEF and transform via similarity.
        // 1. Convert ECI → ECEF
        Vector3d pos_ecef = coords::gcrs_to_itrf_rotation(epoch) * pos_eci;
        // 2. Engine deg=2: gravity at ECEF position
        Eigen::Matrix3d H_ecef_engine;
        {
            auto h_ecef = gravity::EGM2008Gravity::instance().compute_with_hessian(pos_ecef, 2);
            H_ecef_engine = h_ecef.hessian_ecef;
        }
        // 3. Apply ECEF→ECI rotation R · H · Rᵀ explicitly
        // (GCRS rotation, as the force model uses — not true-of-date.)
        const Eigen::Matrix3d R = coords::itrf_to_gcrs_rotation(epoch);
        Eigen::Matrix3d H_eci_ref = R * H_ecef_engine * R.transpose();

        // Both paths use the same GCRS rotation. The 1e-3 bound is loose; a
        // wrong-direction rotation changes the result by more than 1e-2
        // (asserted below), so it is still caught.
        const double scale = H_eci_ref.norm() + 1e-30;
        const double rel = (H_eci_engine - H_eci_ref).norm() / scale;
        TEST_ASSERT(rel < 1e-3);

        // ALSO verify: the wrong rotation (Rᵀ · H · R) gives a DIFFERENT result.
        // If this difference is small, our test isn't actually catching anything.
        Eigen::Matrix3d H_wrong = R.transpose() * H_ecef_engine * R;
        const double wrong_rel = (H_wrong - H_eci_ref).norm() / scale;
        TEST_ASSERT(wrong_rel > 1e-2);  // Confirms rotation matters at this position
    });

    // Mutation 5: Skip pole-singularity handling.
    // At high latitude, gravity Hessian must remain bounded and satisfy Laplace.
    // If pole guards are skipped, 1/cos²(φ) factors blow up and either NaN or
    // Laplace fails dramatically.
    runner.add_test("Mutation 5: high-latitude pole-singularity handling", [&]() {
        ballistic::initialize({});
        auto fm = forces::ForceModel{};
        fm.set_gravity_max_degree(20);
        forces::ForceComponentFlags flags{};
        flags.earth_gravity = true;
        flags.third_body = false;
        flags.drag = false;
        flags.radiation = false;
        flags.relativistic = false;
        flags.solid_earth_tides = false;
        flags.empirical = false;
        fm.set_force_flags(flags);

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

        // Sequence of positions approaching the geographic north pole.
        // (the engine's pole guards should keep all of these bounded; Laplace
        // must hold.)
        const std::vector<std::pair<std::string, Vector3d>> pole_positions = {
            {"60° lat",  Vector3d(3.39e6, 0.0, 5.87e6)},
            {"80° lat",  Vector3d(1.18e6, 0.0, 6.674e6)},
            {"89° lat",  Vector3d(1.183e5, 0.0, 6.777e6)},
            {"89.99° lat", Vector3d(1.183e3, 0.0, 6.778e6)},
        };

        for (const auto& [label, pos] : pole_positions) {
            Vector3d vel(0.0, 7500.0, 0.0);
            StateVector state{pos, vel};
            auto [a, J] = fm.acceleration_with_jacobian(epoch, state);
            const Eigen::Matrix3d H = J.block<3, 3>(0, 0);

            // (a) All entries finite (no NaN from 1/cos²(φ) blow-up)
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    TEST_ASSERT(std::isfinite(H(i, j)));
                }
            }

            // (b) Magnitude reasonable (not 10^30 from divergent terms)
            // Gravity Hessian at LEO ~ 1e-6 1/s² (GM/r³); the 1e-4 bound is a wide margin.
            TEST_ASSERT(H.norm() < 1e-4);

            // (c) Laplace's equation: tr(H) ≈ 0
            const double trace = H(0, 0) + H(1, 1) + H(2, 2);
            const double rel_trace = std::abs(trace) / (H.norm() + 1e-30);
            // At extreme latitudes the pole guards relax Laplace (some
            // longitude-derivative contributions are zeroed). The loose 1e-2
            // bound catches gross violations (NaN, missing guards) and allows
            // that small degradation; skipping the guards entirely lets the
            // 1/cos²(φ) terms blow up.
            TEST_ASSERT(rel_trace < 1e-2);
        }
    });

    // Each single harmonic (zonal m = 0, tesseral, sectoral l = m) must satisfy
    // Laplace's equation at machine precision (1e-13 relative) in the analytic
    // Hessian recurrence (compute_with_hessian_analytic). Failure mode guarded: an
    // error in the recurrence's m = 0 finalization of the longitude-derivative
    // terms violates Laplace for the sectoral cosine modes C22 and C33 and for
    // C31, while zonal modes and the full field still look fine at
    // looser (1e-9) tolerances. See docs/gravity_hessian_derivation.md.

    runner.add_test(
        "Sectoral cos modes satisfy Laplace at machine precision",
        [&]() {
            ballistic::initialize({});
            ballistic::gravity::ClenshawTriangular eng;
            eng.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 5);
            for (int n = 0; n <= 5; ++n) {
                for (int m = 0; m <= n; ++m) {
                    eng.set_coefficient(n, m, 0.0, 0.0);
                }
            }

            // Test each sectoral / tesseral cos mode individually
            const std::vector<std::tuple<int, int, const char*>> tests = {
                {2, 0, "C20 zonal"},
                {2, 1, "C21 tesseral"},
                {2, 2, "C22 sectoral"},
                {3, 0, "C30 zonal"},
                {3, 1, "C31 tesseral"},
                {3, 2, "C32 tesseral"},
                {3, 3, "C33 sectoral"},
            };
            for (const auto& [n, m, label] : tests) {
                // Reset
                for (int nn = 0; nn <= 5; ++nn) {
                    for (int mm = 0; mm <= nn; ++mm) {
                        eng.set_coefficient(nn, mm, 0.0, 0.0);
                    }
                }
                eng.set_coefficient(n, m, 1e-9, 0.0);
                const auto h = eng.compute_with_hessian_analytic(
                    Vector3d(6.778e6, 0.0, 0.0), 5);
                const auto& H = h.hessian_ecef;
                const double trace = H(0, 0) + H(1, 1) + H(2, 2);
                const double rel = std::abs(trace) / (H.norm() + 1e-30);
                // Machine precision: Laplace must hold to ~1e-13 rel
                TEST_ASSERT(rel < 1e-13);
            }
        });

    runner.run();
    return print_final_summary();
}
