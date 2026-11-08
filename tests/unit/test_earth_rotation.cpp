// Unit tests for coordinates/earth_rotation.hpp and the Earth's rotation rate
// used by the GCRS <-> ITRF velocity transforms:
//   - the rotation rate is the Earth rotation angle's (2*pi*1.00273781191135448
//     / 86400 rad per UT1 second), slowed by the measured length of day (LOD);
//   - convert_itrf raises DataCoverageError before 1961 (UTC, which sets the
//     Helmert parameters' decimal year, starts then) instead of terminating;
//   - gast() keeps Meeus's low-accuracy nutation (0.5 arcsec).
// Uses the committed EOP file tests/data/eop/finals2000A.all.

#include "../test_utils.hpp"
#include <ballistic/coordinates/earth_rotation.hpp>
#include <ballistic/coordinates/iau_transforms.hpp>
#include <ballistic/time/eop_data.hpp>
#include <ballistic/util/frame_transform.hpp>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::coords;

namespace {
const double ERA_RATE = 2.0 * constants::PI * 1.00273781191135448 / 86400.0;
}

void test_rotation_rate() {
    TestRunner runner("EarthRotationRate");

    runner.add_test("rate_is_the_earth_rotation_angle_rate", []() {
        TEST_ASSERT_NEAR(EARTH_ROTATION_ANGLE_RATE, ERA_RATE, 1e-22);
        TEST_ASSERT_NEAR(earth_rotation_vector().z(), ERA_RATE, 1e-22);
        // WGS84's nominal value is 1.467e-12 rad/s smaller (~1e-5 m/s at LEO)
        TEST_ASSERT_NEAR(ERA_RATE - constants::earth::ANGULAR_VELOCITY, 1.467e-12, 1e-15);
    });

    // With the table loaded, the rate follows the table's UT1-UTC (its
    // interpolated rate, about -LOD: 0.57 ms/day on the test day, 3e-6 m/s on
    // the equator) and the velocity transform includes the polar motion's rate
    // (~1 mas/day, ~4e-7 m/s), so the velocity of a point fixed on the equator
    // is the derivative of its converted positions: measured 8e-9 m/s (the
    // 5-point difference's own error). Without the UT1 rate it would be off by
    // 3e-6 m/s; without the pole's rate by ~5e-7 m/s.
    runner.add_test("rate_includes_lod_and_matches_position_derivative", []() {
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 6, 0, 0);
        const auto sample = time::EOPData::instance().lookup(60324.25);
        TEST_ASSERT(sample.ut1_utc_rate_s_per_day < -0.4e-3);   // about -LOD
        TEST_ASSERT_NEAR(earth_rotation_rate(t0),
                         ERA_RATE * (1.0 + sample.ut1_utc_rate_s_per_day / 86400.0), 1e-20);

        IAUTransform transform;
        const Vector3d r_itrf(6378137.0, 0.0, 0.0);
        StateVector s; s.position = r_itrf; s.velocity = Vector3d::Zero();
        const Vector3d v = transform.itrf_to_gcrs(s, t0).velocity;
        const double h = 60.0;
        auto p = [&](double k) { return transform.itrf_to_gcrs(r_itrf, t0 + k * h); };
        const Vector3d v_fd = (p(-2) - 8.0 * p(-1) + 8.0 * p(1) - p(2)) / (12.0 * h);
        TEST_ASSERT((v - v_fd).norm() < 3e-8);
        // and the inverse returns zero Earth-fixed velocity
        StateVector g; g.position = transform.itrf_to_gcrs(r_itrf, t0); g.velocity = v;
        TEST_ASSERT(transform.gcrs_to_itrf(g, t0).velocity.norm() < 1e-9);
    });

    // The acceleration transforms use the same rate: a point at rest in ITRF
    // has d2r/dt2 = 0 there, so its GCRS acceleration is the centripetal one.
    runner.add_test("acceleration_transforms_use_the_same_rate", []() {
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 6, 0, 0);
        StateVector s; s.position = Vector3d(6378137.0, 0.0, 0.0); s.velocity = Vector3d::Zero();
        const Vector3d a_gcrs = util::acceleration_to_gcrs(Vector3d::Zero(), s, t0);
        const double w = earth_rotation_rate(t0);
        TEST_ASSERT_NEAR(a_gcrs.norm(), w * w * 6378137.0, 1e-12);
        const Vector3d back = util::acceleration_to_itrf(a_gcrs, s, t0);
        TEST_ASSERT(back.norm() < 1e-15);
    });

    runner.run();
}

void test_convert_itrf_before_1961() {
    TestRunner runner("ConvertItrfBefore1961");

    // convert_itrf was noexcept while its decimal-year conversion throws for
    // dates without UTC, which would have terminated the program.
    runner.add_test("pre_1961_epoch_raises_data_coverage_error", []() {
        const Epoch e = Epoch::from_tt(1955, 6, 1, 0, 0, 0);
        const Vector3d pos(6.9e6, 0.0, 0.0);
        bool caught = false;
        try {
            (void)convert_itrf(pos, ITRFRealization::ITRF2014, ITRFRealization::ITRF2020, e);
        } catch (const DataCoverageError&) {
            caught = true;
        }
        TEST_ASSERT(caught);
        // Same realization: nothing to evaluate, no error
        const Vector3d same = convert_itrf(pos, ITRFRealization::ITRF2020, ITRFRealization::ITRF2020, e);
        TEST_ASSERT_VEC3_NEAR(same, pos, 0.0);
    });

    runner.run();
}

void test_gast_low_accuracy() {
    TestRunner runner("GastLowAccuracy");

    // Meeus eq. 22.1: -17.20" sin(Om) - 1.32" sin(2L) - 0.23" sin(2L') + 0.21" sin(2 Om),
    // accurate to ~0.5". Measured against gast_full (678 terms) over 1990-2040:
    // 0.30" at most. (A version with the Sun's longitude rate 10x too large and
    // the half-month term on 2*Om was off by up to 3.1".)
    runner.add_test("gast_within_half_an_arcsecond_of_gast_full", []() {
        double worst = 0.0;
        for (double d = -3650.0; d < 14600.0; d += 3.7371) {
            const double tt = constants::J2000_JD + d;
            const double ut1 = tt - 69.0 / 86400.0;
            worst = std::max(worst, std::abs(gast(ut1, tt) - gast_full(ut1, tt)));
        }
        TEST_ASSERT(worst / constants::ARCSEC_TO_RAD < 0.5);
    });

    runner.run();
}

int main() {
    const auto eop_path = std::filesystem::path(BALLISTIC_PROJECT_ROOT) / "tests/data/eop/finals2000A.all";
    if (!time::EOPData::instance().load(eop_path)) {
        std::cerr << "cannot load " << eop_path << std::endl;
        return 1;
    }
    test_rotation_rate();
    test_convert_itrf_before_1961();
    test_gast_low_accuracy();
    return print_final_summary();
}
