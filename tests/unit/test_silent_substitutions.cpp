// Silent-substitution sweep: each place in the library that could quietly
// replace a requested model, value or record with something else (a lower
// gravity degree, a neighboring day's weather, a default, a skipped row) must
// instead raise an error or report a diagnostic. One cell per place;
// docs/trap_census.md lists them. All inputs are synthetic or small temporary
// files written by the test.


#include "../test_utils.hpp"

#include <ballistic/ballistic.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/core/diagnostics.hpp>
#include <ballistic/core/errors.hpp>
#include <ballistic/ephemeris/jpl_ephemeris.hpp>
#include <ballistic/gravity/egm2008_coefficients.hpp>
#include <ballistic/gravity/egm_gravity.hpp>
#include <ballistic/gravity/ocean_tides.hpp>
#include <ballistic/gravity/point_mass.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace ballistic;
using namespace ballistic::test;

namespace {

std::filesystem::path temp_file(const std::string& name, const std::string& text) {
    auto p = std::filesystem::temp_directory_path() / ("ballistic_p6_" + name);
    std::ofstream(p) << text;
    return p;
}

const char* kSwHeader =
    "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,"
    "AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,F10.7_OBS_CENTER81,"
    "F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n";

std::string sw_row(const std::string& date, int ap1, const std::string& f107 = "114.6") {
    return date + ",2346,17,27,20,30,23,23,27,17,13,180," + std::to_string(ap1) +
           ",9,18,12,12,15,6,5,11,0.4,2,40," + f107 + ",118.5,OBS,94.6,94.4,97.6,96.6\n";
}

}  // namespace

int main() {
    // Synthetic states only; no Earth orientation table needed.
    time::EOPData::instance().set_policy(time::EopPolicy::AssumeZero);
    TestRunner runner("SilentSubstitutions");

    const Vector3d leo(6778137.0, 0.0, 0.0);

    // ---- gravity ---------------------------------------------------------
    // Asking the EGM2008 gravity model for a degree above what is loaded (360 embedded)
    // raises ConfigError instead of quietly using a lower degree.
    runner.add_test("gravity_degree_above_model_raises", [&]() {
        TEST_ASSERT_THROWS(gravity::EGM2008Gravity::instance().acceleration(leo, 361), ConfigError);
        TEST_ASSERT_THROWS(gravity::EGM2008Gravity::instance().compute_with_hessian(leo, 400),
                           ConfigError);
        // 360 is the embedded ceiling and works.
        TEST_ASSERT(gravity::EGM2008Gravity::instance().acceleration(leo, 360).allFinite());
    });

    // A Trajectory checks its gravity and ocean-tide degrees at construction.
    runner.add_test("trajectory_config_degree_checked", [&]() {
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        TrajectoryConfig cfg;
        cfg.gravity_degree = 720;
        TEST_ASSERT_THROWS(Trajectory(t0, leo, Vector3d(0, 7175, 0), cfg), ConfigError);
        cfg.gravity_degree = 70;
        cfg.ocean_tide_degree = 1;
        TEST_ASSERT_THROWS(Trajectory(t0, leo, Vector3d(0, 7175, 0), cfg), ConfigError);
    });

    runner.add_test("embedded_init_above_360_raises", []() {
        TEST_ASSERT_THROWS(gravity::EGM2008Gravity::instance().init(400), ConfigError);
    });

    // The zonal-only model has coefficients J2..J15; degree 16 raises.
    runner.add_test("zonal_degree_above_15_raises", [&]() {
        TEST_ASSERT_THROWS(gravity::zonal_harmonics_acceleration(
                               leo, constants::earth::GM, constants::earth::RADIUS_EQUATORIAL, 16),
                           ConfigError);
    });

    // A NaN position raises instead of falling back to point-mass gravity.
    runner.add_test("non_finite_gravity_raises", []() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        TEST_ASSERT_THROWS(gravity::EGM2008Gravity::instance().acceleration(Vector3d(nan, 0, 7e6), 10),
                           Error);
    });

    runner.add_test("clenshaw_file_load_keeps_lower_degrees", []() {
        // Rows sorted by degree, model not pre-sized: growing the storage row by
        // row must keep the coefficients already read (re-allocating would zero them).
        auto p = temp_file("grav.txt",
            "# n m C S\n"
            "2 0 -0.484165143790815D-03 0.0\n"
            "2 2 0.243938357328313D-05 -0.140027370385934D-05\n"
            "3 0 0.957161207093473D-06 0.0\n"
            "4 0 0.539965866638991D-06 0.0\n");
        gravity::Clenshaw model;
        TEST_ASSERT(model.load_from_file(p));
        TEST_ASSERT_EQ(model.max_degree(), 4);
        TEST_ASSERT_NEAR(model.get_coefficient(2, 0).first, -0.484165143790815e-03, 1e-18);
        TEST_ASSERT_NEAR(model.get_coefficient(2, 2).second, -0.140027370385934e-05, 1e-18);
        std::filesystem::remove(p);
    });

    // A malformed coefficient line raises instead of being skipped.
    runner.add_test("clenshaw_file_malformed_line_raises", []() {
        auto p = temp_file("grav_bad.txt", "2 0 -0.48D-03 0.0\n2 1 garbage\n");
        gravity::Clenshaw model;
        TEST_ASSERT_THROWS(model.load_from_file(p), Error);
        std::filesystem::remove(p);
    });

    // ---- ocean tides -----------------------------------------------------
    // A malformed ocean-tide (FES2004 format) data row raises.
    runner.add_test("ocean_tide_bad_row_raises", []() {
        auto p = temp_file("tides_bad.dat",
            "Coefficients header\nDoodson Darw  l   m    DelC+     DelS+       DelC-     DelS-\n"
            " 55.565 Om1   2   0  -6.58128   0.00000    -0.00000  -0.00000\n"
            " 55.575 Om2   2   x  -0.06330   0.00000    -0.00000  -0.00000\n");
        TEST_ASSERT_THROWS(gravity::OceanTideModel::instance().load(p), Error);
        std::filesystem::remove(p);
    });

    // Degree-1 rows are skipped by convention; the degree-2 row loads.
    runner.add_test("ocean_tide_degree_one_rows_are_skipped", []() {
        auto p = temp_file("tides_ok.dat",
            "Doodson Darw  l   m    DelC+     DelS+       DelC-     DelS-\n"
            " 56.554 Sa    1   0   0.00000   0.00000     0.00000   0.00000\n"
            " 55.565 Om1   2   0  -6.58128   0.00000    -0.00000  -0.00000\n");
        TEST_ASSERT(gravity::OceanTideModel::instance().load(p));
        TEST_ASSERT_EQ(gravity::OceanTideModel::instance().max_degree(), 2);
        gravity::OceanTideModel::instance().clear();
        std::filesystem::remove(p);
    });

    // ---- space weather ---------------------------------------------------
    runner.add_test("weather_record_is_the_day_not_the_nearest", []() {
        using atmosphere::SpaceWeatherData;
        auto p = temp_file("sw.csv", std::string(kSwHeader) + sw_row("2005-06-30", 18) +
                                         sw_row("2005-07-01", 12) + sw_row("2005-07-02", 5));
        auto& sw = SpaceWeatherData::instance();
        sw.clear();
        TEST_ASSERT(sw.load_celestrak_csv(p));
        // Records are stamped at noon, so exactly 00:00 UTC on 07-01 is equally far
        // from the 06-30 and 07-01 stamps; the lookup must return the day itself
        // (07-01), not the nearest stamp.
        const double midnight = Epoch::from_utc(2005, 7, 1, 0, 0, 0).mjd(time::TimeScale::UTC);
        auto rec = sw.get_record(midnight);
        TEST_ASSERT(rec.has_value());
        TEST_ASSERT_NEAR(rec->ap_3hr[0], 12.0, 1e-12);
        // Outside the file: nothing, not the edge row.
        TEST_ASSERT(!sw.get_record(midnight + 10.0).has_value());
        TEST_ASSERT(!sw.get_record(midnight - 10.0).has_value());
        sw.clear();
        std::filesystem::remove(p);
    });

    // The geomagnetic index Ap is averaged over the previous hours; a lookback that
    // runs before the file's first day raises instead of using a default.
    runner.add_test("weather_ap_lookback_past_file_start_raises", []() {
        using atmosphere::SpaceWeatherData;
        auto p = temp_file("sw2.csv", std::string(kSwHeader) + sw_row("2005-07-01", 12) +
                                          sw_row("2005-07-02", 5));
        auto& sw = SpaceWeatherData::instance();
        sw.clear();
        TEST_ASSERT(sw.load_celestrak_csv(p));
        // The 57-hour Ap lookback from 07-01 12:00 reaches 06-29.
        TEST_ASSERT_THROWS(sw.get_conditions(Epoch::from_utc(2005, 7, 1, 12, 0, 0)),
                           std::out_of_range);
        sw.clear();
        std::filesystem::remove(p);
    });

    // A CelesTrak weather file without the Ap columns is refused and nothing is loaded.
    runner.add_test("weather_file_without_ap_column_refused", []() {
        using atmosphere::SpaceWeatherData;
        auto p = temp_file("sw3.csv", "DATE,F10.7_OBS\n2005-07-01,114.6\n");
        TEST_ASSERT_THROWS(SpaceWeatherData::instance().load_celestrak_csv(p), Error);
        TEST_ASSERT(!SpaceWeatherData::instance().is_loaded());
        std::filesystem::remove(p);
    });

    // A blank solar flux (F10.7) value is reported once as DataSkipped and kept in
    // the loader's warnings.
    runner.add_test("weather_blank_f107_reported", []() {
        using atmosphere::SpaceWeatherData;
        int reports = 0;
        diagnostics::set_handler([&](const diagnostics::Diagnostic& d) {
            if (d.category == diagnostics::Category::DataSkipped) ++reports;
        });
        auto p = temp_file("sw4.csv", std::string(kSwHeader) + sw_row("2005-07-01", 12, ""));
        TEST_ASSERT(SpaceWeatherData::instance().load_celestrak_csv(p));
        diagnostics::set_handler(nullptr);
        TEST_ASSERT_EQ(reports, 1);
        TEST_ASSERT_EQ(SpaceWeatherData::instance().warnings().size(), 1u);
        SpaceWeatherData::instance().clear();
        std::filesystem::remove(p);
    });

    // ---- beta table ------------------------------------------------------
    // The time axis of a tabulated beta profile counts in the environment's domain:
    // a query past its last time is flagged as clamped.
    runner.add_test("beta2d_time_axis_in_domain", [&]() {
        const Epoch t0 = Epoch::from_utc(2024, 1, 15, 0, 0, 0);
        atmosphere::BetaProfile2D beta({100.0, 500.0, 900.0}, {t0, t0 + 3600.0},
                                       {50.0, 50.0, 50.0, 60.0, 60.0, 60.0});
        atmosphere::DragEnvironment env(
            atmosphere::DensityProfile1D({100.0, 500.0, 900.0}, {1e-9, 1e-12, 1e-14}),
            atmosphere::WindProfile1D_ENU{}, beta);
        auto w = env.domain();
        TEST_ASSERT(w.has_value());
        TEST_ASSERT(!env.evaluate(leo, t0 + 1800.0).time_clamped);
        TEST_ASSERT(env.evaluate(leo, t0 + 7200.0).time_clamped);
    });

    // ---- ephemeris -------------------------------------------------------
    // An unloaded JPL ephemeris raises instead of returning zeros.
    runner.add_test("jpl_ephemeris_not_loaded_raises", []() {
        ephemeris::JPLEphemeris eph;
        TEST_ASSERT_THROWS(eph.get_body_state(2451545.0, 2), Error);
    });

    // Last: replaces the process-wide gravity model.
    runner.add_test("egm_file_load_reaches_active_storage", [&]() {
        // File = the embedded set to degree 360 plus one degree-361 term. The loaded
        // file must reach the storage the computation actually reads: degree 360
        // is unchanged (< 1e-15) and degree 361 differs (> 1e-12).
        auto p = std::filesystem::temp_directory_path() / "ballistic_p6_egm361.txt";
        {
            std::ofstream f(p);
            f.precision(17);
            for (const auto& c : gravity::egm2008_data::COEFFICIENTS)
                f << c.n << ' ' << c.m << ' ' << c.C << ' ' << c.S << '\n';
            f << "361 0 1.0e-2 0.0\n";   // large: (R/r)^362 ~ 1e-10 at LEO
        }
        auto& egm = gravity::EGM2008Gravity::instance();
        const Vector3d a360 = egm.acceleration(leo, 360);
        TEST_ASSERT(egm.load_from_file(p));
        TEST_ASSERT_EQ(egm.available_degree(), 361);
        TEST_ASSERT((egm.acceleration(leo, 360) - a360).norm() < 1e-15);
        TEST_ASSERT((egm.acceleration(leo, 361) - a360).norm() > 1e-12);
        std::filesystem::remove(p);
    });

    runner.run();
    return print_final_summary();
}
