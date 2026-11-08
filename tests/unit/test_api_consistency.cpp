// Consistency checks on smaller library surfaces: the legacy binary DE reader's
// geocentric Sun, the gravity engine's point-mass fallback potential, a
// symplectic step-size change taking effect, and the typed errors raised by the
// math tables, the velocity-record check, RK45 and the ephemeris readers. All
// inputs are synthetic (the DE cell writes a small temporary file); the one
// Trajectory cell loads the committed EOP table.

#include "../test_utils.hpp"

#include <ballistic/ballistic.hpp>
#include <ballistic/coordinates/velocity_check.hpp>
#include <ballistic/core/diagnostics.hpp>
#include <ballistic/core/errors.hpp>
#include <ballistic/ephemeris/jpl_ephemeris.hpp>
#include <ballistic/gravity/clenshaw_engine.hpp>
#include <ballistic/math/bspline_basis.hpp>
#include <ballistic/math/cubic_spline.hpp>
#include <ballistic/math/hermite_table.hpp>
#include <ballistic/propagators/rk45.hpp>
#include <ballistic/propagators/symplectic.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;

namespace {

// A one-record DE binary file in the layout JPLEphemeris assumes (DE440's), with
// constant Chebyshev series (c_0 only) for the Earth-Moon barycenter, the Moon
// and the Sun, so every body sits at a known fixed position [km].
std::filesystem::path write_synthetic_de(const Vector3d& emb_km, const Vector3d& moon_km,
                                         const Vector3d& sun_km) {
    constexpr int kRecord = 1018;                 // doubles per record (DE440)
    std::vector<char> header(2 * kRecord * 8, 0);
    const double times[3] = {2451536.5, 2451568.5, 32.0};   // start, end, days per record
    std::memcpy(header.data() + 252, times, sizeof(times));

    std::vector<double> rec(kRecord, 0.0);
    auto put = [&](int offset1, int n_coeffs, int n_sub, const Vector3d& v) {
        for (int s = 0; s < n_sub; ++s)
            for (int c = 0; c < 3; ++c)
                rec[(offset1 - 1) + s * n_coeffs * 3 + c * n_coeffs] = v[c];
    };
    put(231, 13, 2, emb_km);   // Earth-Moon barycenter (relative to the SSB)
    put(441, 13, 8, moon_km);  // Moon (geocentric)
    put(753, 11, 2, sun_km);   // Sun (relative to the SSB)

    const auto p = std::filesystem::temp_directory_path() / "ballistic_synthetic_de.bin";
    std::ofstream f(p, std::ios::binary);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(rec.data()),
            static_cast<std::streamsize>(rec.size() * sizeof(double)));
    return p;
}

}  // namespace

int main() {
    // The Trajectory cell runs with the committed EOP table; nothing else needs it.
    TestRunner runner("API consistency");

    runner.add_test("jpl_reader_sun_is_geocentric", []() {
        // Sun - Earth with Earth = EMB - Moon / (1 + EMRAT): the reader used to return
        // the Sun relative to the solar-system barycenter (~1 AU off from geocentric).
        const Vector3d emb(1.0e8, -5.0e7, 2.0e7), moon(3.0e5, 2.0e5, -1.0e5),
                       sun(1.2e6, -3.0e5, 4.0e4);
        const auto p = write_synthetic_de(emb, moon, sun);
        ephemeris::JPLEphemeris eph;
        TEST_ASSERT(eph.load(p));
        const double emrat = 81.3005690699;
        const Vector3d earth = emb - moon / (1.0 + emrat);
        const Epoch t = Epoch::from_jd(2451550.0, time::TimeScale::TDB);
        const Vector3d got = eph.sun_position(t);
        const Vector3d want = (sun - earth) * 1000.0;
        TEST_ASSERT((got - want).norm() < 1e-6);   // [m]; a barycentric Sun is ~1e11 m off
        TEST_ASSERT((eph.moon_position(t) - moon * 1000.0).norm() < 1e-6);
        std::filesystem::remove(p);
    });

    runner.add_test("jpl_reader_errors_are_typed", []() {
        ephemeris::JPLEphemeris eph;
        TEST_ASSERT_THROWS((void)eph.sun_position(Epoch::from_jd(2451550.0, time::TimeScale::TDB)), DataCoverageError);
        TEST_ASSERT_THROWS((void)eph.moon_position(Epoch::from_jd(2451550.0, time::TimeScale::TDB)), DataCoverageError);
        const auto p = write_synthetic_de(Vector3d::Zero(), Vector3d::UnitX(), Vector3d::UnitY());
        TEST_ASSERT(eph.load(p));
        TEST_ASSERT_THROWS((void)eph.sun_position(Epoch::from_jd(2460000.0, time::TimeScale::TDB)), DataCoverageError);
        TEST_ASSERT_THROWS((void)eph.get_body_state(2451550.0, 13), ConfigError);
        std::filesystem::remove(p);
    });

    runner.add_test("clenshaw_point_mass_fallback_potential_matches_series", []() {
        // With no coefficients the engine falls back to a point mass; its potential must
        // have the series' sign (V = +GM/r), as the degree-0 series gives.
        const Vector3d pos(6.9e6, -1.2e6, 2.3e6);
        gravity::Clenshaw empty;
        gravity::Clenshaw deg0;
        deg0.init(constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 2);  // only C00 = 1
        const auto a = empty.compute(pos);
        const auto b = deg0.compute(pos);
        const double gm_r = constants::earth::GM / pos.norm();
        TEST_ASSERT_NEAR_REL(a.potential, gm_r, 1e-14);
        TEST_ASSERT_NEAR_REL(a.potential, b.potential, 1e-13);
        TEST_ASSERT((a.acceleration - b.acceleration).norm() < 1e-13 * a.acceleration.norm());
        const auto ha = empty.compute_with_hessian_analytic(pos);
        TEST_ASSERT_NEAR_REL(ha.potential, gm_r, 1e-14);
    });

    runner.add_test("symplectic_set_step_size_takes_effect", []() {
        // Two-body integrator: 60 s at the 30 s default step = 2 steps; after
        // set_step_size(10) the next 60 s take 6 steps (the change used to wait for a
        // rebuild).
        auto two_body = [](const Epoch&, const Vector3d& r) {
            return Vector3d(-constants::earth::GM / std::pow(r.norm(), 3) * r);
        };
        propagators::SymplecticIntegrator integ(two_body);
        const Epoch t0 = Epoch::from_jd(2451545.0, time::TimeScale::TT);
        const StateVector s0(Vector3d(7.0e6, 0, 0), Vector3d(0, 7546.0, 0));
        const StateVector s1 = integ.integrate(t0, s0, t0 + 60.0);
        TEST_ASSERT_EQ(integ.stats().steps_taken, static_cast<size_t>(2));
        integ.set_step_size(10.0);
        TEST_ASSERT_NEAR(integ.config().step_size, 10.0, 0.0);
        (void)integ.integrate(t0 + 60.0, s1, t0 + 120.0);
        TEST_ASSERT_EQ(integ.stats().steps_taken, static_cast<size_t>(8));
        TEST_ASSERT_THROWS(integ.set_step_size(0.0), ConfigError);
        TEST_ASSERT_THROWS(integ.set_step_size(std::nan("")), ConfigError);
    });

    runner.add_test("symplectic_trajectory_step_hint_sets_the_step", []() {
        // Trajectory::set_step_size_hint on a symplectic trajectory: each following
        // step() advances by the hinted step (it used to keep the configured one).
        TEST_ASSERT(time::EOPData::instance().load(BALLISTIC_PROJECT_ROOT
                                                   "/tests/data/eop/finals2000A.all"));
        TrajectoryConfig cfg;
        cfg.integrator = IntegratorType::YOSHIDA6;
        cfg.ocean_tide_degree = 0;
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Trajectory traj(InitialState::from_itrf(t0, Vector3d(6778137.0, 0, 0),
                                                Vector3d(0, 7175.0, 0)), cfg);
        (void)traj.step();
        TEST_ASSERT_NEAR(traj.epoch() - t0, cfg.symplectic.step_size, 1e-9);
        traj.set_step_size_hint(7.0);
        TEST_ASSERT_NEAR(traj.current_step_size(), 7.0, 0.0);
        (void)traj.step();
        (void)traj.step();
        TEST_ASSERT_NEAR(traj.epoch() - t0, cfg.symplectic.step_size + 14.0, 1e-9);
        TEST_ASSERT_THROWS(traj.set_step_size_hint(-1.0), ConfigError);
        time::EOPData::instance().clear();
    });

    runner.add_test("math_tables_raise_typed_errors", []() {
        const std::vector<double> x3{0.0, 1.0, 2.0}, y2{1.0, 2.0}, x2{0.0, 1.0}, d1{0.0};
        const std::vector<Vector3d> v1{Vector3d::Zero()};
        TEST_ASSERT_THROWS(math::CubicSplineScalar(x3, y2), InputPlausibilityError);
        TEST_ASSERT_THROWS(math::HermiteScalar(x2, y2, d1), InputPlausibilityError);
        TEST_ASSERT_THROWS(math::LinearVector3d(x2, v1), InputPlausibilityError);
        TEST_ASSERT_THROWS((void)math::make_clamped_knot_vector(1.0, 0.0, 3, 3), ConfigError);
        TEST_ASSERT_THROWS((void)math::make_clamped_knot_vector(0.0, 1.0, -1, 3), ConfigError);
        // The message gives both lengths (it must not read the moved-from arguments).
        bool named = false;
        try {
            math::CubicSplineScalar s(x3, y2);
        } catch (const InputPlausibilityError& e) {
            named = std::string(e.what()).find("3 x values but 2 y values") != std::string::npos;
        }
        TEST_ASSERT(named);
    });

    runner.add_test("velocity_check_raises_typed_errors", []() {
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 0, 0, 0);
        std::vector<double> off(5);
        std::vector<Vector3d> p(5, Vector3d(7e6, 0, 0)), v(5, Vector3d(0, 7e3, 0));
        for (int i = 0; i < 5; ++i) off[i] = 10.0 * i;
        TEST_ASSERT_THROWS((void)coords::check_velocity_records(t0, off, p, v),
                           InputPlausibilityError);   // fewer than 11 epochs
        p.pop_back();
        TEST_ASSERT_THROWS((void)coords::check_velocity_records(t0, off, p, v),
                           InputPlausibilityError);   // lengths differ
        std::vector<double> off12(12, 0.0);
        std::vector<Vector3d> p12(12, Vector3d(7e6, 0, 0)), v12(12, Vector3d(0, 7e3, 0));
        TEST_ASSERT_THROWS((void)coords::check_velocity_records(t0, off12, p12, v12),
                           InputPlausibilityError);   // not increasing
    });

    runner.add_test("rk45_budget_exhausted_raises_ballistic_error", []() {
        propagators::RK45Config cfg;
        cfg.max_iterations = 1;   // a 7-evaluation budget
        cfg.max_step_size = 10.0;
        propagators::RK45Integrator rk(
            [](const Epoch&, const StateVector& s) {
                return StateVector(s.velocity, -1e-6 * s.position);
            },
            cfg);
        const Epoch t0 = Epoch::from_jd(2451545.0, time::TimeScale::TT);
        TEST_ASSERT_THROWS((void)rk.integrate(t0, StateVector(Vector3d(1, 0, 0), Vector3d::Zero()),
                                              t0 + 1000.0),
                           Error);
    });

    runner.add_test("initialize_with_a_legacy_de_file_overrides_a_loaded_spk", []() {
        // The default provider prefers a loaded SPK, and the embedded DE440
        // loads itself on first use; a legacy DE file given to initialize()
        // afterwards must still be the one force models use.
        InitConfig embedded_cfg;
        embedded_cfg.eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
        TEST_ASSERT(initialize(embedded_cfg).success);
        TEST_ASSERT(ephemeris::SPKEphemerisManager::instance().is_loaded());
        auto old_provider = ephemeris::get_default_ephemeris_provider();

        const Vector3d emb(1.0e8, -5.0e7, 2.0e7), moon(3.0e5, 2.0e5, -1.0e5),
                       sun(1.2e6, -3.0e5, 4.0e4);
        const auto p = write_synthetic_de(emb, moon, sun);
        InitConfig de_cfg = embedded_cfg;
        de_cfg.jpl_ephemeris_path = p;
        const auto res = initialize(de_cfg);
        TEST_ASSERT(res.success);
        TEST_ASSERT(res.jpl_ephemeris_loaded);

        const Epoch t = Epoch::from_jd(2451550.0, time::TimeScale::TDB);
        const Vector3d earth = emb - moon / (1.0 + 81.3005690699);
        const Vector3d want = (sun - earth) * 1000.0;
        const Vector3d got = ephemeris::get_default_ephemeris_provider()->sun_position(t);
        TEST_ASSERT((got - want).norm() < 1e-6);
        // a provider taken before says what replaced it instead of answering
        bool named = false;
        try {
            (void)old_provider->sun_position(t);
        } catch (const DataCoverageError& e) {
            named = std::string(e.what()).find(p.string()) != std::string::npos;
        }
        TEST_ASSERT(named);

        // restore the embedded DE440 for anything that runs after this cell
        TEST_ASSERT(initialize(embedded_cfg).success);
        TEST_ASSERT(ephemeris::SPKEphemerisManager::instance().is_using_embedded());
        std::filesystem::remove(p);
    });

    runner.run();
    return print_final_summary();
}
