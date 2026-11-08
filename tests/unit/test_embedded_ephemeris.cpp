// The embedded JPL DE440 planetary ephemeris (Sun, Moon, Jupiter positions compiled
// into the library, 1950-2100): loading, coverage edges, orbital sanity checks,
// agreement with JPL Horizons, continuity across the Chebyshev records the data is
// stored in, and (optionally) agreement with a downloaded de440.bsp SPK file
// (SPK = the SPICE binary ephemeris format). ET = ephemeris time, seconds past J2000.
//
// Reference: Park et al. (2021), "The JPL Planetary and Lunar Ephemerides DE440 and DE441",
// AJ 161:105, doi:10.3847/1538-3881/abd414

#include "../test_utils.hpp"
#include <ballistic/ephemeris/spk_ephemeris.hpp>
#include <ballistic/init.hpp>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <vector>

using namespace ballistic;
using namespace ballistic::test;
using namespace ballistic::ephemeris;
using namespace ballistic::constants;
using ballistic::time::TimeScale;

// =============================================================================
// Test Suites
// =============================================================================

void test_embedded_loading() {
    TestRunner runner("EmbeddedLoading");

    runner.add_test("load_embedded_succeeds", []() {
        SPKEphemeris eph;
        bool result = eph.load_embedded();
        TEST_ASSERT_TRUE(result);
        TEST_ASSERT_TRUE(eph.is_loaded());
        TEST_ASSERT_TRUE(eph.is_using_embedded());
    });

    runner.add_test("de_version_is_440", []() {
        SPKEphemeris eph;
        eph.load_embedded();
        TEST_ASSERT_EQ(eph.de_version(), 440);
    });

    runner.add_test("emrat_correct", []() {
        SPKEphemeris eph;
        eph.load_embedded();
        TEST_ASSERT_NEAR(eph.emrat(), 81.3005690699, 1e-10);
    });

    runner.run();
}

void test_embedded_sun_position() {
    TestRunner runner("EmbeddedSunPosition");

    runner.add_test("j2000_valid", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph.sun_position(epoch);

        // Sun distance should be ~1 AU
        double dist = sun_pos.norm();
        TEST_ASSERT_NEAR(dist, sun::AU, sun::AU * 0.02);  // Within 2%
    });

    runner.add_test("year_1950_valid", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // 1950-01-01 00:00:00 UTC
        Epoch epoch = Epoch::from_tt(1950, 1, 1, 0, 0, 0);   // TT: UTC starts in 1961
        Vector3d sun_pos = eph.sun_position(epoch);

        double dist = sun_pos.norm();
        TEST_ASSERT_NEAR(dist, sun::AU, sun::AU * 0.02);
        TEST_ASSERT(std::isfinite(sun_pos.x()));
        TEST_ASSERT(std::isfinite(sun_pos.y()));
        TEST_ASSERT(std::isfinite(sun_pos.z()));
    });

    runner.add_test("year_2099_valid", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // 2099-01-01 — inside the embedded DE440 coverage range.
        Epoch epoch = Epoch::from_utc(2099, 1, 1, 0, 0, 0);
        Vector3d sun_pos = eph.sun_position(epoch);

        double dist = sun_pos.norm();
        TEST_ASSERT_NEAR(dist, sun::AU, sun::AU * 0.02);
        TEST_ASSERT(std::isfinite(sun_pos.x()));
    });

    runner.add_test("post_2100_throws", []() {
        // Past the embedded SPK range — strict contract requires a throw.
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::from_utc(2101, 1, 1, 0, 0, 0);
        bool threw = false;
        try {
            (void)eph.sun_position(epoch);
        } catch (const ballistic::DataCoverageError&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    });

    runner.add_test("annual_period", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch_0 = Epoch::j2000();
        Epoch epoch_1yr = Epoch::j2000() + 365.25 * SECONDS_PER_DAY;

        Vector3d pos_0 = eph.sun_position(epoch_0);
        Vector3d pos_1yr = eph.sun_position(epoch_1yr);

        // After one year, should be back to similar position (eccentricity ~0.017)
        double diff = (pos_1yr - pos_0).norm();
        TEST_ASSERT(diff < sun::AU * 0.01);
    });

    runner.run();
}

void test_embedded_moon_position() {
    TestRunner runner("EmbeddedMoonPosition");

    runner.add_test("j2000_valid", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::j2000();
        Vector3d moon_pos = eph.moon_position(epoch);

        // Moon distance should be ~384,400 km
        double dist = moon_pos.norm();
        TEST_ASSERT_NEAR(dist, moon::MEAN_DISTANCE, moon::MEAN_DISTANCE * 0.1);
    });

    runner.add_test("orbital_period", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch_0 = Epoch::j2000();
        // Sidereal month ~27.3 days
        Epoch epoch_1mo = Epoch::j2000() + 27.3 * SECONDS_PER_DAY;

        Vector3d pos_0 = eph.moon_position(epoch_0);
        Vector3d pos_1mo = eph.moon_position(epoch_1mo);

        // After one sidereal month, should return to similar region
        double dot = pos_0.normalized().dot(pos_1mo.normalized());
        TEST_ASSERT(dot > -0.5);
    });

    runner.add_test("perigee_apogee_variation", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // Find approximate min and max distances over a month
        double min_dist = moon::MEAN_DISTANCE * 2;
        double max_dist = 0;

        for (double days = 0; days < 30; days += 0.5) {
            Epoch epoch = Epoch::j2000() + days * SECONDS_PER_DAY;
            Vector3d pos = eph.moon_position(epoch);
            double dist = pos.norm();

            min_dist = (std::min)(min_dist, dist);
            max_dist = (std::max)(max_dist, dist);
        }

        // Moon eccentricity ~0.055
        double mean_dist = moon::MEAN_DISTANCE;
        TEST_ASSERT(min_dist < mean_dist * 0.97);
        TEST_ASSERT(max_dist > mean_dist * 1.03);
    });

    runner.run();
}

void test_embedded_jupiter_position() {
    TestRunner runner("EmbeddedJupiterPosition");

    runner.add_test("j2000_valid", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::j2000();
        Vector3d jupiter_pos = eph.jupiter_position(epoch);

        // Jupiter distance ~5.2 AU from Sun, so ~4.2 to 6.2 AU from Earth
        double dist = jupiter_pos.norm();
        TEST_ASSERT(dist > 3.5 * sun::AU);
        TEST_ASSERT(dist < 7.0 * sun::AU);
    });

    runner.add_test("slow_motion", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // Jupiter moves slowly - position shouldn't change much in a month
        Epoch epoch_0 = Epoch::j2000();
        Epoch epoch_1mo = Epoch::j2000() + 30 * SECONDS_PER_DAY;

        Vector3d pos_0 = eph.jupiter_position(epoch_0);
        Vector3d pos_1mo = eph.jupiter_position(epoch_1mo);

        // Direction should be nearly the same
        double dot = pos_0.normalized().dot(pos_1mo.normalized());
        TEST_ASSERT(dot > 0.99);
    });

    runner.run();
}

void test_initialization_with_embedded() {
    TestRunner runner("InitializationWithEmbedded");

    runner.add_test("default_init_loads_embedded", []() {
        // Reset state (create new manager instance)
        InitConfig config;
        InitResult result = initialize(config);

        TEST_ASSERT_TRUE(result.success);
        TEST_ASSERT_TRUE(result.embedded_ephemeris_loaded);
        TEST_ASSERT_EQ(result.jpl_de_version, std::string("DE440 (embedded)"));
    });

    runner.add_test("get_init_status_reports_embedded", []() {
        InitConfig config;
        initialize(config);

        InitResult status = get_init_status();
        TEST_ASSERT_TRUE(status.embedded_ephemeris_loaded);
    });

    runner.run();
}

void test_time_range() {
    TestRunner runner("TimeRange");

    runner.add_test("within_range_1960", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::from_tt(1960, 6, 15, 12, 0, 0);   // TT: UTC starts in 1961
        Vector3d sun_pos = eph.sun_position(epoch);

        TEST_ASSERT(std::isfinite(sun_pos.x()));
        TEST_ASSERT(sun_pos.norm() > 0);  // Not zero (fallback would give zero if failed)
    });

    runner.add_test("within_range_2050", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::from_utc(2050, 6, 15, 12, 0, 0);
        Vector3d sun_pos = eph.sun_position(epoch);

        double dist = sun_pos.norm();
        TEST_ASSERT_NEAR(dist, sun::AU, sun::AU * 0.02);
    });

    runner.add_test("boundary_1950_jan_01", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // Exactly at the start boundary
        Epoch epoch = Epoch::from_tt(1950, 1, 1, 0, 0, 0);   // TT: UTC starts in 1961
        Vector3d sun_pos = eph.sun_position(epoch);

        TEST_ASSERT(std::isfinite(sun_pos.x()));
        TEST_ASSERT(sun_pos.norm() > sun::AU * 0.9);  // Should be valid
    });

    runner.add_test("boundary_late_2099", []() {
        // Late in 2099 is inside the embedded SPK coverage; verify success.
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::from_utc(2099, 12, 30, 23, 59, 59);
        Vector3d sun_pos = eph.sun_position(epoch);

        TEST_ASSERT(std::isfinite(sun_pos.x()));
        TEST_ASSERT(sun_pos.norm() > sun::AU * 0.9);
    });

    runner.run();
}

// =============================================================================
// JPL Horizons Reference Validation
// =============================================================================
// Hardcoded geocentric ICRF positions from JPL Horizons (CENTER=500@399,
// geometric, no aberrations, km). These are independent reference values.
//
// Our Sun positions are relative to EMB (not geocentric), so expect ~4671 km
// offset. Moon positions use the EMRAT conversion and should match closely.

void test_jpl_horizons_reference() {
    TestRunner runner("JPLHorizonsReference");

    runner.add_test("moon_j2000_vs_horizons", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // J2000.0 = 2000-01-01 12:00:00 TDB
        Epoch epoch = Epoch::j2000();
        Vector3d moon_pos = eph.moon_position(epoch);  // meters

        // JPL Horizons geocentric Moon position at J2000.0 TDB (km)
        // CENTER=500@399, ICRF, geometric, no aberration
        Vector3d horizons_moon_km(-291608.3841877129, -266716.8338540655, -76102.48730658790);
        Vector3d horizons_moon_m = horizons_moon_km * 1000.0;

        double diff_km = (moon_pos - horizons_moon_m).norm() / 1000.0;
        std::cout << "    Moon J2000 diff from Horizons: " << diff_km << " km" << std::endl;

        // Moon should match Horizons within 1 km
        // Small differences expected: DE440 embedded vs DE441 used by Horizons
        TEST_ASSERT_MSG(diff_km < 1.0,
            "Moon position differs from JPL Horizons by more than 1 km at J2000");
    });

    runner.add_test("moon_2024_vs_horizons", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // 2024-01-15 12:00:00 UTC  guard-exempt(PUB-DATE): data
        // Horizons values are at TDB, but UTC-TDB offset is ~69s.
        // Moon speed ~1 km/s, so UTC epoch introduces ~69 km offset.
        // We test with a 100 km tolerance to account for this.
        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d moon_pos = eph.moon_position(epoch);  // meters

        // JPL Horizons geocentric Moon position at 2024-01-15 12:00:00 TDB (km)  guard-exempt(PUB-DATE): data
        Vector3d horizons_moon_km(358785.6882446791, -54408.87710644645, -40936.18572597361);
        Vector3d horizons_moon_m = horizons_moon_km * 1000.0;

        double diff_km = (moon_pos - horizons_moon_m).norm() / 1000.0;
        std::cout << "    Moon 2024-01-15 diff from Horizons: " << diff_km << " km" << std::endl;
        std::cout << "    (Includes ~69 km from UTC-TDB offset)" << std::endl;

        // ~69 km from UTC-TDB offset + small DE440 differences
        TEST_ASSERT_MSG(diff_km < 100.0,
            "Moon position differs from JPL Horizons by more than 100 km at 2024-01-15");
    });

    runner.add_test("sun_j2000_vs_horizons", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::j2000();
        Vector3d sun_pos = eph.sun_position(epoch);  // meters

        // JPL Horizons geocentric Sun position at J2000.0 TDB (km)
        Vector3d horizons_sun_km(2.649903367743050e+07, -1.327574173383451e+08, -5.755671847054072e+07);
        Vector3d horizons_sun_m = horizons_sun_km * 1000.0;

        double diff_km = (sun_pos - horizons_sun_m).norm() / 1000.0;
        std::cout << "    Sun J2000 diff from Horizons: " << diff_km << " km" << std::endl;
        std::cout << "    (Expected ~4671 km offset: our code returns Sun-EMB, not Sun-Earth)" << std::endl;

        // Sun is relative to EMB, not geocentric — expect ~4671 km offset
        // If error exceeds 5000 km, something is wrong beyond the known EMB offset
        TEST_ASSERT_MSG(diff_km < 5000.0,
            "Sun position differs from JPL Horizons by more than 5000 km (EMB offset is ~4671 km)");
        // Verify it's in the ballpark of the expected EMB-geocentric offset
        TEST_ASSERT_MSG(diff_km > 1000.0,
            "Sun position unexpectedly close to Horizons geocentric (EMB offset should be ~4671 km)");
    });

    runner.add_test("sun_2024_vs_horizons", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
        Vector3d sun_pos = eph.sun_position(epoch);  // meters

        // JPL Horizons geocentric Sun position at 2024-01-15 12:00:00 TDB (km)  guard-exempt(PUB-DATE): data
        Vector3d horizons_sun_km(6.099755991476827e+07, -1.228667169930571e+08, -5.326173607712085e+07);
        Vector3d horizons_sun_m = horizons_sun_km * 1000.0;

        double diff_km = (sun_pos - horizons_sun_m).norm() / 1000.0;
        std::cout << "    Sun 2024-01-15 diff from Horizons: " << diff_km << " km" << std::endl;

        TEST_ASSERT_MSG(diff_km < 5000.0,
            "Sun position differs from JPL Horizons by more than 5000 km at 2024-01-15");
    });

    runner.run();
}

// =============================================================================
// Chebyshev Record Boundary Continuity
// =============================================================================
// Verifies that positions are continuous across Chebyshev record boundaries.
// A discontinuity would indicate data corruption or indexing errors.

void test_record_boundary_continuity() {
    TestRunner runner("RecordBoundaryContinuity");

    // Strategy: compare position change across a record boundary to the change
    // at a non-boundary point with the same time step. If there's a discontinuity,
    // the boundary change will be much larger than the smooth-motion change.
    // We use delta=0.01s which gives manageable motion (~300m Sun, ~10m Moon).

    runner.add_test("sun_boundary_continuity", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // SUN_SSB and EMB_SSB: init_et=-1578052800.0, interval=1382400.0s
        // Pick record boundary near J2000 for best floating-point precision
        // Record 1142: ET = -1578052800 + 1142 * 1382400 = 648000 (~7.5 days past J2000)
        double boundary_et = 648000.0;  // seconds past J2000
        double dt = 0.01;  // 10 ms

        // Change across boundary
        Vector3d pos_before = eph.sun_position(Epoch::j2000() + (boundary_et - dt));
        Vector3d pos_after  = eph.sun_position(Epoch::j2000() + (boundary_et + dt));
        double boundary_change = (pos_after - pos_before).norm();

        // Change at mid-record (should be smooth)
        double mid_et = boundary_et + 691200.0;  // half interval offset
        Vector3d pos_mid_before = eph.sun_position(Epoch::j2000() + (mid_et - dt));
        Vector3d pos_mid_after  = eph.sun_position(Epoch::j2000() + (mid_et + dt));
        double mid_change = (pos_mid_after - pos_mid_before).norm();

        // Ratio should be ~1.0 if continuous (both are just smooth motion)
        double ratio = boundary_change / mid_change;
        std::cout << "    Sun boundary/mid change ratio: " << ratio
                  << " (boundary: " << boundary_change << " m, mid: " << mid_change << " m)" << std::endl;

        // Allow 10% difference (acceleration varies slightly between points)
        TEST_ASSERT_MSG(ratio > 0.5 && ratio < 2.0,
            "Sun position shows discontinuity at record boundary");
    });

    runner.add_test("moon_boundary_continuity", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // MOON_EMB: init_et=-1578052800.0, interval=345600.0s (4 days)
        // Records 4565-4569 straddle J2000 (record 4567 starts at
        // ET = -1578052800 + 4567*345600 = 302400 s).
        double boundary_et = 345600.0;  // 1 interval past init, near J2000
        double dt = 0.01;

        double max_ratio = 0;
        // Test several boundaries
        for (int i = 4565; i < 4570; ++i) {
            double b_et = -1578052800.0 + i * 345600.0;
            Vector3d pos_before = eph.moon_position(Epoch::j2000() + (b_et - dt));
            Vector3d pos_after  = eph.moon_position(Epoch::j2000() + (b_et + dt));
            double boundary_change = (pos_after - pos_before).norm();

            double mid_et = b_et + 172800.0;  // half interval
            Vector3d pos_mid_before = eph.moon_position(Epoch::j2000() + (mid_et - dt));
            Vector3d pos_mid_after  = eph.moon_position(Epoch::j2000() + (mid_et + dt));
            double mid_change = (pos_mid_after - pos_mid_before).norm();

            double ratio = boundary_change / mid_change;
            max_ratio = (std::max)(max_ratio, ratio);
        }

        std::cout << "    Moon max boundary/mid change ratio: " << max_ratio << std::endl;

        TEST_ASSERT_MSG(max_ratio < 2.0,
            "Moon position shows discontinuity at record boundary");
    });

    runner.add_test("jupiter_boundary_continuity", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // JUPITER_SSB: init_et=-1579435200.0, interval=2764800.0s (32 days)
        // Pick a boundary near J2000: record 571 starts at
        // ET = -1579435200 + 571*2764800 = -734400 s (8.5 days before J2000).
        double boundary_et = -1579435200.0 + 571.0 * 2764800.0;
        double dt = 0.01;

        Vector3d pos_before = eph.jupiter_position(Epoch::j2000() + (boundary_et - dt));
        Vector3d pos_after  = eph.jupiter_position(Epoch::j2000() + (boundary_et + dt));
        double boundary_change = (pos_after - pos_before).norm();

        double mid_et = boundary_et + 1382400.0;  // half interval
        Vector3d pos_mid_before = eph.jupiter_position(Epoch::j2000() + (mid_et - dt));
        Vector3d pos_mid_after  = eph.jupiter_position(Epoch::j2000() + (mid_et + dt));
        double mid_change = (pos_mid_after - pos_mid_before).norm();

        double ratio = boundary_change / mid_change;
        std::cout << "    Jupiter boundary/mid change ratio: " << ratio
                  << " (boundary: " << boundary_change << " m, mid: " << mid_change << " m)" << std::endl;

        TEST_ASSERT_MSG(ratio > 0.5 && ratio < 2.0,
            "Jupiter position shows discontinuity at record boundary");
    });

    runner.run();
}

// =============================================================================
// Velocity Spot-Check via Finite Differencing
// =============================================================================
// Verifies Chebyshev derivatives by comparing finite-difference velocity
// (from two positions 1 second apart) against known Moon orbital velocity.

void test_velocity_spot_check() {
    TestRunner runner("VelocitySpotCheck");

    runner.add_test("moon_velocity_magnitude", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::j2000();
        Epoch epoch_plus_1s = Epoch::j2000() + 1.0;

        Vector3d pos0 = eph.moon_position(epoch);
        Vector3d pos1 = eph.moon_position(epoch_plus_1s);

        // Finite-difference velocity (m/s)
        Vector3d vel_fd = pos1 - pos0;
        double speed = vel_fd.norm();

        std::cout << "    Moon geocentric speed (FD): " << speed << " m/s" << std::endl;

        // Moon's geocentric velocity is ~1.0 km/s (range: 0.97 to 1.08 km/s)
        TEST_ASSERT_MSG(speed > 900.0 && speed < 1200.0,
            "Moon geocentric speed should be ~1000 m/s");
    });

    runner.add_test("moon_velocity_fd_consistency", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        // Compare forward and backward finite differences
        Epoch epoch = Epoch::j2000();
        double dt = 1.0;  // 1 second

        Vector3d pos_minus = eph.moon_position(epoch + (-dt));
        Vector3d pos_center = eph.moon_position(epoch);
        Vector3d pos_plus = eph.moon_position(epoch + dt);

        // Forward difference
        Vector3d vel_fwd = (pos_plus - pos_center) / dt;
        // Central difference (more accurate)
        Vector3d vel_central = (pos_plus - pos_minus) / (2.0 * dt);

        // They should agree to within ~0.1 m/s (second-order truncation error)
        double diff = (vel_fwd - vel_central).norm();
        std::cout << "    Moon velocity FD vs central diff: " << diff << " m/s" << std::endl;

        TEST_ASSERT_MSG(diff < 1.0,
            "Forward and central finite-difference velocities should agree within 1 m/s");
    });

    runner.add_test("sun_velocity_magnitude", []() {
        SPKEphemeris eph;
        eph.load_embedded();

        Epoch epoch = Epoch::j2000();
        Epoch epoch_plus_1s = Epoch::j2000() + 1.0;

        Vector3d pos0 = eph.sun_position(epoch);
        Vector3d pos1 = eph.sun_position(epoch_plus_1s);

        // Finite-difference velocity (m/s)
        Vector3d vel_fd = pos1 - pos0;
        double speed = vel_fd.norm();

        std::cout << "    Sun apparent speed (FD): " << speed << " m/s" << std::endl;

        // Earth's orbital velocity is ~29.8 km/s, so Sun apparent motion is similar
        TEST_ASSERT_MSG(speed > 25000.0 && speed < 35000.0,
            "Sun apparent speed should be ~30000 m/s");
    });

    runner.run();
}

// =============================================================================
// Optional: Compare with external DE440 file if available
// =============================================================================

void test_embedded_vs_external() {
    TestRunner runner("EmbeddedVsExternal");

    // Runs only if data/de440.bsp exists (a git-ignored JPL download); the path is
    // absolute (project root) because ctest runs from the build tree.
    static const std::string de440_path = BALLISTIC_PROJECT_ROOT "/data/de440.bsp";

    runner.add_test("sun_matches_external", []() {
        // Load embedded
        SPKEphemeris embedded;
        embedded.load_embedded();

        // Load external
        SPKEphemeris external;
        if (!std::filesystem::exists(de440_path)) TEST_SKIP("file not found: " << de440_path);
        external.load(de440_path);

        // Compare at J2000
        Epoch epoch = Epoch::j2000();

        Vector3d emb_sun = embedded.sun_position(epoch);
        Vector3d ext_sun = external.sun_position(epoch);

        // Should match to high precision (same data source)
        double diff = (emb_sun - ext_sun).norm();
        TEST_ASSERT(diff < 1.0);  // Within 1 meter
    });

    runner.add_test("moon_matches_external", []() {
        SPKEphemeris embedded;
        embedded.load_embedded();

        SPKEphemeris external;
        if (!std::filesystem::exists(de440_path)) TEST_SKIP("file not found: " << de440_path);
        external.load(de440_path);

        Epoch epoch = Epoch::j2000();

        Vector3d emb_moon = embedded.moon_position(epoch);
        Vector3d ext_moon = external.moon_position(epoch);

        double diff = (emb_moon - ext_moon).norm();
        TEST_ASSERT(diff < 1.0);  // Within 1 meter
    });

    runner.add_test("jupiter_matches_external", []() {
        SPKEphemeris embedded;
        embedded.load_embedded();

        SPKEphemeris external;
        if (!std::filesystem::exists(de440_path)) TEST_SKIP("file not found: " << de440_path);
        external.load(de440_path);

        Epoch epoch = Epoch::j2000();

        Vector3d emb_jup = embedded.jupiter_position(epoch);
        Vector3d ext_jup = external.jupiter_position(epoch);

        double diff = (emb_jup - ext_jup).norm();
        TEST_ASSERT(diff < 1.0);  // Within 1 meter
    });

    runner.add_test("consistency_over_time_range", []() {
        SPKEphemeris embedded;
        embedded.load_embedded();

        SPKEphemeris external;
        if (!std::filesystem::exists(de440_path)) TEST_SKIP("file not found: " << de440_path);
        external.load(de440_path);

        // Test at various epochs (all within embedded data coverage 1950-2100)
        // Note: Embedded data covers approximately 1950-01 to 2100-01
        std::vector<int> years = {1950, 1980, 2000, 2024, 2050, 2099};
        double max_sun_diff = 0;
        double max_moon_diff = 0;
        int max_sun_year = 0;
        int max_moon_year = 0;

        for (int year : years) {
            Epoch epoch = Epoch::from_tt(year, 6, 15, 12, 0, 0);  // TT: no UTC before 1961

            Vector3d emb_sun = embedded.sun_position(epoch);
            Vector3d ext_sun = external.sun_position(epoch);
            double sun_diff = (emb_sun - ext_sun).norm();
            if (sun_diff > max_sun_diff) {
                max_sun_diff = sun_diff;
                max_sun_year = year;
            }

            Vector3d emb_moon = embedded.moon_position(epoch);
            Vector3d ext_moon = external.moon_position(epoch);
            double moon_diff = (emb_moon - ext_moon).norm();
            if (moon_diff > max_moon_diff) {
                max_moon_diff = moon_diff;
                max_moon_year = year;
            }
        }

        std::cout << "    Max Sun diff: " << max_sun_diff << "m at year " << max_sun_year << std::endl;
        std::cout << "    Max Moon diff: " << max_moon_diff << "m at year " << max_moon_year << std::endl;

        // Should match to high precision (same data source)
        TEST_ASSERT_MSG(max_sun_diff < 1.0,
            "Sun position differs by more than 1m from external file");
        TEST_ASSERT_MSG(max_moon_diff < 1.0,
            "Moon position differs by more than 1m from external file");
    });

    runner.run();
}

// =============================================================================
// Main
// =============================================================================

// A minimal one-segment SPK (DAF) file: file record, one summary record, then one
// type-2 record of constant Chebyshev series (position = pos_km everywhere in
// [start_et, end_et]).
std::filesystem::path write_one_segment_spk(const std::string& name, int target, int center,
                                            double start_et, double end_et,
                                            const Vector3d& pos_km) {
    const auto p = std::filesystem::temp_directory_path() / name;
    std::vector<char> file(3 * 1024, 0);
    std::memcpy(file.data(), "DAF/SPK ", 8);
    const int nd = 2, ni = 6, fwd = 2, bwd = 2;
    std::memcpy(file.data() + 8, &nd, 4);
    std::memcpy(file.data() + 12, &ni, 4);
    std::memcpy(file.data() + 76, &fwd, 4);
    std::memcpy(file.data() + 80, &bwd, 4);

    // Data: record [mid, radius, x0, y0, z0] then [init_et, interval, rsize, n]
    const int start_addr = 2 * 128 + 1;          // first double of record 3
    const double data[9] = {0.5 * (start_et + end_et), 0.5 * (end_et - start_et),
                            pos_km.x(), pos_km.y(), pos_km.z(),
                            start_et, end_et - start_et, 5.0, 1.0};
    std::memcpy(file.data() + 2 * 1024, data, sizeof(data));
    const int end_addr = start_addr + 9 - 1;

    char* rec = file.data() + 1024;
    const double next = 0.0, prev = 0.0, nsum = 1.0;
    std::memcpy(rec, &next, 8);
    std::memcpy(rec + 8, &prev, 8);
    std::memcpy(rec + 16, &nsum, 8);
    const int ints[6] = {target, center, 1, 2, start_addr, end_addr};
    std::memcpy(rec + 24, &start_et, 8);
    std::memcpy(rec + 32, &end_et, 8);
    std::memcpy(rec + 40, ints, sizeof(ints));

    std::ofstream f(p, std::ios::binary);
    f.write(file.data(), static_cast<std::streamsize>(file.size()));
    return p;
}

void test_file_over_embedded() {
    TestRunner runner("FileOverEmbedded");

    runner.add_test("a_file_lacking_a_pair_keeps_the_embedded_pair", []() {
        // load() adds the file's segments to those already held. Only the
        // file's own segments are read from it: the embedded ones carry no
        // file addresses, and re-reading them used to fail the load.
        SPKEphemeris eph;
        TEST_ASSERT_TRUE(eph.load_embedded());
        const double et = 8.0e8;   // 2025
        const Vector3d moon_before = eph.get_position(et, 301, 3);
        const Vector3d jup_before = eph.get_position(et, 5, 0);
        TEST_ASSERT(moon_before.norm() > 3e5);

        const Vector3d sun_km(1.0e5, -2.0e5, 3.0e5);
        const auto p = write_one_segment_spk("ballistic_test_sun_only.bsp", 10, 0,
                                             7.0e8, 9.0e8, sun_km);
        TEST_ASSERT_TRUE(eph.load(p));
        TEST_ASSERT_TRUE(eph.is_loaded());
        // the file's pair comes from the file
        TEST_ASSERT_VEC3_NEAR(eph.get_position(et, 10, 0), sun_km, 1e-9);
        // the pairs it lacks still evaluate exactly as before
        TEST_ASSERT_VEC3_NEAR(eph.get_position(et, 301, 3), moon_before, 0.0);
        TEST_ASSERT_VEC3_NEAR(eph.get_position(et, 5, 0), jup_before, 0.0);
        std::filesystem::remove(p);
    });

    runner.run();
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Embedded DE440 Ephemeris Tests" << std::endl;
    std::cout << "======================================" << std::endl;

    test_embedded_loading();
    test_embedded_sun_position();
    test_embedded_moon_position();
    test_embedded_jupiter_position();
    test_initialization_with_embedded();
    test_time_range();
    test_jpl_horizons_reference();
    test_record_boundary_continuity();
    test_velocity_spot_check();
    test_embedded_vs_external();
    test_file_over_embedded();

    return print_final_summary();
}
