// Unit tests for EmpiricalAcceleration (forces/empirical_acceleration.hpp).
// An empirical acceleration (EA) is an extra acceleration, given as a
// function of time, added to the force model to absorb forces it does not
// model; estimators produce one from observations. This file checks the
// three storage forms: a table of (time, acceleration) entries interpolated
// by cubic spline, linear or Hermite segments; a B-spline that represents the
// acceleration directly or a position / velocity whose 2nd / 1st time
// derivative is the acceleration; and the zero EA. Also: clamping outside the
// time range, frame tagging (GCRS inertial vs ITRF Earth-fixed) and that a
// Trajectory applies it. Tolerances are machine precision for exact
// interpolation properties.

#include "../test_utils.hpp"
#include <ballistic/forces/empirical_acceleration.hpp>
#include <ballistic/trajectory.hpp>
#include <ballistic/math/bspline_basis.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::forces;

namespace {

// Helper: build BSplineCoefficients from std::vector<double> coefficient lists.
BSplineCoefficients make_coeffs(
    const std::vector<double>& cx,
    const std::vector<double>& cy,
    const std::vector<double>& cz,
    std::vector<double> knots,
    int degree = 3)
{
    Eigen::VectorXd ex(cx.size());
    Eigen::VectorXd ey(cy.size());
    Eigen::VectorXd ez(cz.size());
    for (std::size_t i = 0; i < cx.size(); ++i) ex[i] = cx[i];
    for (std::size_t i = 0; i < cy.size(); ++i) ey[i] = cy[i];
    for (std::size_t i = 0; i < cz.size(); ++i) ez[i] = cz[i];
    return BSplineCoefficients(std::move(ex), std::move(ey), std::move(ez),
                               std::move(knots), degree);
}

}  // namespace

int main() {
    TestRunner runner("Empirical Acceleration");

    // =========================================================================
    // Table backend (cubic spline / linear / Hermite)
    // =========================================================================

    runner.add_test("cubic interpolation between entries", []() {
        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(0.0, Vector3d(0.0, 0.0, 0.0));
        entries.emplace_back(100.0, Vector3d(1e-5, 2e-6, 3e-7));
        auto emp = EmpiricalAcceleration::from_entries(entries);

        // With 2 points, cubic spline falls back to linear.
        Vector3d acc = emp.accel_at(50.0);
        TEST_ASSERT_NEAR(acc.x(), 0.5e-5, 1e-10);
        TEST_ASSERT_NEAR(acc.y(), 1e-6, 1e-10);
        TEST_ASSERT_NEAR(acc.z(), 1.5e-7, 1e-10);

        acc = emp.accel_at(25.0);
        TEST_ASSERT_NEAR(acc.x(), 0.25e-5, 1e-10);
    });

    runner.add_test("boundary clamping", []() {
        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(100.0, Vector3d(1e-5, 0.0, 0.0));
        entries.emplace_back(200.0, Vector3d(2e-5, 0.0, 0.0));
        auto emp = EmpiricalAcceleration::from_entries(entries);

        Vector3d acc = emp.accel_at(0.0);
        TEST_ASSERT_NEAR(acc.x(), 1e-5, 1e-15);

        acc = emp.accel_at(500.0);
        TEST_ASSERT_NEAR(acc.x(), 2e-5, 1e-15);
    });

    runner.add_test("single entry returns constant", []() {
        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(50.0, Vector3d(1e-5, 2e-6, 3e-7));
        auto emp = EmpiricalAcceleration::from_entries(entries);

        Vector3d acc = emp.accel_at(0.0);
        TEST_ASSERT_NEAR(acc.x(), 1e-5, 1e-15);
        TEST_ASSERT_NEAR(acc.y(), 2e-6, 1e-15);
    });

    runner.add_test("zero EA returns zero", []() {
        EmpiricalAcceleration emp;  // default-constructed = zero
        TEST_ASSERT(emp.is_zero());
        Vector3d acc = emp.accel_at(0.0);
        TEST_ASSERT_NEAR(acc.norm(), 0.0, 1e-20);

        // Static factory matches default ctor.
        auto z = EmpiricalAcceleration::zero();
        TEST_ASSERT(z.is_zero());
    });

    runner.add_test("from_entries with empty list returns zero", []() {
        auto emp = EmpiricalAcceleration::from_entries({});
        TEST_ASSERT(emp.is_zero());
    });

    runner.add_test("entries are sorted by key", []() {
        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(300.0, Vector3d(3e-5, 0.0, 0.0));
        entries.emplace_back(100.0, Vector3d(1e-5, 0.0, 0.0));
        entries.emplace_back(200.0, Vector3d(2e-5, 0.0, 0.0));
        auto emp = EmpiricalAcceleration::from_entries(entries);

        // At exact knots: cubic spline passes through.
        Vector3d acc = emp.accel_at(100.0);
        TEST_ASSERT_NEAR(acc.x(), 1e-5, 1e-10);
        acc = emp.accel_at(200.0);
        TEST_ASSERT_NEAR(acc.x(), 2e-5, 1e-10);
    });

    runner.add_test("multi-point cubic spline passes through knots", []() {
        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(0.0, Vector3d(0.0, 0.0, 0.0));
        entries.emplace_back(100.0, Vector3d(1e-5, 2e-6, 0.0));
        entries.emplace_back(200.0, Vector3d(1e-5, 2e-6, 0.0));
        entries.emplace_back(300.0, Vector3d(0.0, 0.0, 0.0));
        auto emp = EmpiricalAcceleration::from_entries(entries);

        TEST_ASSERT_NEAR(emp.accel_at(0.0).x(),   0.0, 1e-10);
        TEST_ASSERT_NEAR(emp.accel_at(100.0).x(), 1e-5, 1e-10);
        TEST_ASSERT_NEAR(emp.accel_at(200.0).x(), 1e-5, 1e-10);
        TEST_ASSERT_NEAR(emp.accel_at(300.0).x(), 0.0, 1e-10);
    });

    runner.add_test("cubic spline is smooth at interior", []() {
        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(0.0,   Vector3d(0.0,    0.0, 0.0));
        entries.emplace_back(100.0, Vector3d(1e-5,   0.0, 0.0));
        entries.emplace_back(200.0, Vector3d(2e-5,   0.0, 0.0));
        entries.emplace_back(300.0, Vector3d(1e-5,   0.0, 0.0));
        entries.emplace_back(400.0, Vector3d(0.0,    0.0, 0.0));
        auto emp = EmpiricalAcceleration::from_entries(entries);

        double prev = emp.accel_at(0.0).x();
        for (double t = 1.0; t <= 400.0; t += 1.0) {
            const double cur = emp.accel_at(t).x();
            const double change = std::abs(cur - prev);
            TEST_ASSERT_MSG(change < 5e-7,
                "Jump of " + std::to_string(change) + " at t=" + std::to_string(t));
            prev = cur;
        }
    });

    runner.add_test("linear interpolation factory", []() {
        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(0.0,   Vector3d(0.0,  0.0, 0.0));
        entries.emplace_back(100.0, Vector3d(1e-5, 0.0, 0.0));
        entries.emplace_back(200.0, Vector3d(0.0,  0.0, 0.0));
        auto emp = EmpiricalAcceleration::from_entries(
            entries, EmpiricalAccelFrame::GCRS, EAInterp::Linear);

        // Piecewise-linear: midpoint of [0,100] is exactly 0.5e-5
        TEST_ASSERT_NEAR(emp.accel_at(50.0).x(), 0.5e-5, 1e-15);
        TEST_ASSERT_NEAR(emp.accel_at(150.0).x(), 0.5e-5, 1e-15);
    });

    runner.add_test("hermite factory accepts derivatives", []() {
        std::vector<EmpiricalAccelEntry> entries;
        std::vector<Vector3d> derivs;
        entries.emplace_back(0.0,   Vector3d(0.0,  0.0, 0.0));
        entries.emplace_back(100.0, Vector3d(1e-5, 0.0, 0.0));
        derivs.emplace_back(0.0, 0.0, 0.0);
        derivs.emplace_back(0.0, 0.0, 0.0);
        auto emp = EmpiricalAcceleration::from_entries_hermite(entries, derivs);

        // Knots are exact; interior is a smooth Hermite cubic.
        TEST_ASSERT_NEAR(emp.accel_at(0.0).x(),   0.0,  1e-12);
        TEST_ASSERT_NEAR(emp.accel_at(100.0).x(), 1e-5, 1e-12);
    });

    // =========================================================================
    // Integration with Trajectory
    // =========================================================================

    runner.add_test("trajectory with empirical acceleration", []() {
        ballistic::initialize({});

        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        TrajectoryConfig config_base;
        config_base.gravity_degree = 4;
        config_base.enable_third_body = false;
        config_base.enable_relativistic = false;
        config_base.enable_solid_tides = false;
        Trajectory traj_base(t0, pos, vel, config_base);

        for (int i = 0; i < 20; ++i) traj_base.step();
        auto state_base = traj_base.state_ecef();

        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(0.0,     Vector3d(1e-4, 0.0, 0.0));
        entries.emplace_back(10000.0, Vector3d(1e-4, 0.0, 0.0));

        TrajectoryConfig config_emp;
        config_emp.gravity_degree = 4;
        config_emp.enable_third_body = false;
        config_emp.enable_relativistic = false;
        config_emp.enable_solid_tides = false;
        config_emp.empirical_accel = std::make_shared<EmpiricalAcceleration>(
            EmpiricalAcceleration::from_entries(entries));
        Trajectory traj_emp(t0, pos, vel, config_emp);

        while (traj_emp.epoch() - t0 < traj_base.epoch() - t0) {
            traj_emp.step();
        }
        auto state_emp = traj_emp.state_ecef();

        const double pos_diff = (state_emp.position - state_base.position).norm();
        TEST_ASSERT(pos_diff > 0.01);
    });

    runner.add_test("force breakdown includes empirical", []() {
        ballistic::initialize({});

        Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d pos(6778137.0, 0.0, 0.0);
        Vector3d vel(0.0, 7500.0, 0.0);

        std::vector<EmpiricalAccelEntry> entries;
        entries.emplace_back(0.0,     Vector3d(1e-5, 2e-6, 0.0));
        entries.emplace_back(10000.0, Vector3d(1e-5, 2e-6, 0.0));

        TrajectoryConfig config;
        config.gravity_degree = 4;
        config.enable_third_body = false;
        config.enable_relativistic = false;
        config.enable_solid_tides = false;
        config.report_forces = true;
        config.empirical_accel = std::make_shared<EmpiricalAcceleration>(
            EmpiricalAcceleration::from_entries(entries));
        Trajectory traj(t0, pos, vel, config);

        auto result = traj.step();
        TEST_ASSERT(result.forces.has_value());

        auto& fb = result.forces.value();
        TEST_ASSERT(fb.empirical.norm() > 1e-10);
    });

    // =========================================================================
    // B-Spline backend
    // =========================================================================

    runner.add_test("bspline constant EA", []() {
        // Uniform coefficients → constant value for represents=Acceleration
        auto knots = math::make_clamped_knot_vector(0.0, 100.0, 4, 3);
        const int n_basis = 4 + 3 + 1;  // n_interior + degree + 1 = 8
        std::vector<double> cx(n_basis, 1e-5);
        std::vector<double> cy(n_basis, 2e-6);
        std::vector<double> cz(n_basis, 3e-7);
        auto coeffs = make_coeffs(cx, cy, cz, knots);
        auto emp = EmpiricalAcceleration::from_bspline(
            std::move(coeffs), EARepresents::Acceleration);

        // Partition of unity: sum(N_i) = 1, so constant coefficients → constant value
        for (double t : {0.0, 10.0, 50.0, 90.0, 100.0}) {
            Vector3d acc = emp.accel_at(t);
            TEST_ASSERT_NEAR(acc.x(), 1e-5, 1e-15);
            TEST_ASSERT_NEAR(acc.y(), 2e-6, 1e-15);
            TEST_ASSERT_NEAR(acc.z(), 3e-7, 1e-15);
        }

        TEST_ASSERT(!emp.is_zero());
        TEST_ASSERT(emp.as_bspline().has_value());
    });

    runner.add_test("bspline derivative eval (Position represents → d²/dt²)", []() {
        auto knots = math::make_clamped_knot_vector(0.0, 100.0, 4, 3);
        const int n_basis = 8;
        std::vector<double> cx(n_basis), cy(n_basis, 0.0), cz(n_basis, 0.0);
        for (int i = 0; i < n_basis; ++i) cx[i] = static_cast<double>(i) * 1e-5;
        auto coeffs = make_coeffs(cx, cy, cz, knots);
        auto emp = EmpiricalAcceleration::from_bspline(
            std::move(coeffs), EARepresents::Position);

        Vector3d acc_mid = emp.accel_at(50.0);
        TEST_ASSERT(std::isfinite(acc_mid.x()));

        Vector3d acc_a = emp.accel_at(49.0);
        Vector3d acc_b = emp.accel_at(51.0);
        TEST_ASSERT((acc_a - acc_b).norm() < 1e-6);
    });

    runner.add_test("bspline boundary clamping", []() {
        auto knots = math::make_clamped_knot_vector(10.0, 90.0, 3, 3);
        const int n_basis = 3 + 3 + 1;  // 7
        std::vector<double> cx(n_basis, 1e-5);
        std::vector<double> cy(n_basis, 0.0);
        std::vector<double> cz(n_basis, 0.0);
        cx[0] = 5e-5;
        cx[n_basis - 1] = 9e-5;
        auto coeffs = make_coeffs(cx, cy, cz, knots);
        auto emp = EmpiricalAcceleration::from_bspline(
            std::move(coeffs), EARepresents::Acceleration);

        Vector3d at_start = emp.accel_at(10.0);
        Vector3d before   = emp.accel_at(0.0);
        TEST_ASSERT_NEAR((at_start - before).norm(), 0.0, 1e-10);

        Vector3d at_end = emp.accel_at(90.0);
        Vector3d after  = emp.accel_at(200.0);
        TEST_ASSERT_NEAR((at_end - after).norm(), 0.0, 1e-10);
    });

    runner.add_test("bspline ITRF frame is preserved", []() {
        auto knots = math::make_clamped_knot_vector(0.0, 100.0, 2, 3);
        const int n_basis = 6;
        std::vector<double> c(n_basis, 1e-5);
        auto coeffs = make_coeffs(c, c, c, knots);
        auto emp = EmpiricalAcceleration::from_bspline(
            std::move(coeffs), EARepresents::Acceleration,
            EmpiricalAccelFrame::ITRF);

        TEST_ASSERT(emp.frame() == EmpiricalAccelFrame::ITRF);
        auto view = emp.as_bspline();
        TEST_ASSERT(view.has_value());
        TEST_ASSERT_EQ(view->degree, 3);
    });

    runner.add_test("bspline keys() populated for refinement grid", []() {
        auto knots = math::make_clamped_knot_vector(0.0, 100.0, 4, 3);
        const int n_basis = 8;
        std::vector<double> c(n_basis, 1e-5);
        auto coeffs = make_coeffs(c, c, c, knots);
        auto emp = EmpiricalAcceleration::from_bspline(
            std::move(coeffs), EARepresents::Acceleration);

        const auto& keys = emp.keys();
        TEST_ASSERT(keys.size() >= 2);
        TEST_ASSERT_NEAR(keys.front(), 0.0, 1e-10);
        TEST_ASSERT_NEAR(keys.back(), 100.0, 1e-10);
        for (size_t i = 1; i < keys.size(); ++i) {
            TEST_ASSERT(keys[i] > keys[i - 1]);
        }
    });

    runner.add_test("BSplineCoefficients validates sizes", []() {
        // cx.size() != cy.size() should throw
        TEST_ASSERT_THROWS({
            BSplineCoefficients bad(
                Eigen::VectorXd::Zero(5),
                Eigen::VectorXd::Zero(4),
                Eigen::VectorXd::Zero(5),
                std::vector<double>(9, 0.0),
                3);
        }, std::invalid_argument);

        // knots.size() inconsistent with cx.size() should throw (expected 5+3+1=9)
        TEST_ASSERT_THROWS({
            BSplineCoefficients bad(
                Eigen::VectorXd::Zero(5),
                Eigen::VectorXd::Zero(5),
                Eigen::VectorXd::Zero(5),
                std::vector<double>(7, 0.0),
                3);
        }, std::invalid_argument);
    });

    runner.add_test("B-spline degree above the evaluator's buffers raises ConfigError", []() {
        // eval() works in fixed buffers sized for degree <= kMaxDegree (7);
        // degree 8 would write past them, so it is refused where accepted.
        const int p = BSplineCoefficients::kMaxDegree + 1;
        const int n_basis = p + 1;
        std::vector<double> knots(static_cast<std::size_t>(n_basis + p + 1), 0.0);
        for (std::size_t i = static_cast<std::size_t>(n_basis); i < knots.size(); ++i)
            knots[i] = 100.0;
        TEST_ASSERT_THROWS({
            BSplineCoefficients bad(Eigen::VectorXd::Ones(n_basis), Eigen::VectorXd::Ones(n_basis),
                                    Eigen::VectorXd::Ones(n_basis), knots, p);
        }, ConfigError);

        // A struct edited after construction is re-checked by from_bspline.
        const int q = BSplineCoefficients::kMaxDegree;
        std::vector<double> knots7(static_cast<std::size_t>(2 * (q + 1)), 0.0);
        for (std::size_t i = static_cast<std::size_t>(q + 1); i < knots7.size(); ++i)
            knots7[i] = 100.0;
        BSplineCoefficients ok(Eigen::VectorXd::Ones(q + 1), Eigen::VectorXd::Ones(q + 1),
                               Eigen::VectorXd::Ones(q + 1), knots7, q);
        // degree 7 is accepted and evaluates (partition of unity: all-ones -> 1)
        for (auto rep : {EARepresents::Acceleration, EARepresents::Position}) {
            auto ea7 = EmpiricalAcceleration::from_bspline(ok, rep, EmpiricalAccelFrame::GCRS);
            const Vector3d a = ea7.accel_at(37.0);
            TEST_ASSERT(a.allFinite());
            if (rep == EARepresents::Acceleration) TEST_ASSERT_NEAR(a.x(), 1.0, 1e-12);
        }
        BSplineCoefficients edited = ok;
        edited.degree = p;
        TEST_ASSERT_THROWS((void)EmpiricalAcceleration::from_bspline(
                               edited, EARepresents::Acceleration, EmpiricalAccelFrame::GCRS),
                           ConfigError);
    });

    runner.run();
    return print_final_summary();
}
