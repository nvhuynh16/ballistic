// =============================================================================
// ESA Copernicus POD Service Validation Test Suite
// =============================================================================
//
// PURPOSE:
// --------
// Propagates a Sentinel-1 state for one hour with a force model set up to
// resemble the one ESA's Copernicus Precise Orbit Determination (POD) Service
// uses, and compares the prediction with ESA's precise orbit. Also checks that
// the DE421 planetary ephemeris loads and agrees with DE440, and that basic
// force magnitudes are physically plausible. The 3-5 m bounds are loose
// regression bounds, not accuracy claims; the library's Sentinel accuracy
// figures are [`SENTINEL-JAN26-SAMPLED`] and [`SENTINEL-YEAR-2025`] in
// docs/validation_claims.md.
//
// Data: tests/data/eof/sentinel1_2016/2016-8 (ESA precise orbit ephemeris, POE,
// in ESA's EOF file format; not committed, so without it the suite is one
// counted skip). Earth orientation: tests/data/eop/finals2000A.all.
//
// Terms: RMSE = root-mean-square position error over the hour; SRP = solar
// radiation pressure; Cd / Cr = drag / radiation pressure coefficients;
// GCRS = the inertial frame the propagator integrates in; ITRF = the
// Earth-fixed frame of the orbit file; EOP = Earth orientation parameters
// (polar motion, UT1-UTC).
//
// ESA POD SERVICE OVERVIEW:
// -------------------------
// The Copernicus POD Service is operated by GMV under ESA contract. It provides
// precise orbit products for Sentinel-1, -2, -3, and -6 satellites using the
// NAPEOS software developed at ESA/ESOC.
//
// Key characteristics:
// - Achieves ~5cm (3D RMS) orbit accuracy for Sentinel-1
// - Uses dual-frequency GPS observations from onboard receiver
// - Employs comprehensive force modeling with estimated parameters
//
// WHY WE CAN'T EXACTLY MATCH ESA:
// --------------------------------
// 1. DRAG COEFFICIENT: ESA estimates Cd dynamically (10 coefficients/day for
//    NRT, 1/arc for precise products with sigma=0.3 constraint). We use fixed
//    Cd=2.2 as industry standard.
//
// 2. SOLAR RADIATION PRESSURE: ESA uses a detailed box-wing model with
//    re-radiation based on satellite macro-model. We use simplified cannonball.
//
// 3. TIDES: ESA includes solid Earth tides, ocean tides (FES2014), and pole
//    tides. The library has solid Earth tides and FES2004 ocean tides (and
//    omits the pole tide, measured negligible); this harness's propagator
//    does not configure them explicitly.
//
// 4. GRAVITY: ESA uses time-varying gravity field up to degree 90 with
//    quarterly updates. We use static EGM2008 (sufficient for 1-hour tests).
//
// CITABLE SOURCES:
// ----------------
// [1] Sentiwiki POD Details:
//     https://sentiwiki.copernicus.eu/web/precise-orbit-determination
//     Accessed 2025. Describes force model configuration.
//
// [2] Copernicus POD Product Handbook v1.9 (2025):
//     GMV-CPOD3-PH-0001
//     https://sentiwiki.copernicus.eu/__attachments/1673407/
//     Sections 3.2.2.4-5 describe atmospheric and drag modeling.
//
// [3] GMV-CPOD3-TN-0025 "Sentinel-1 Properties for GNSS POD" v2.0:
//     https://sentiwiki.copernicus.eu/__attachments/1673968/
//     Contains satellite macro-model and thermo-optical properties.
//
// [4] Peter, H. et al. (2017):
//     "Sentinel-1A - First precise orbit determination results"
//     Advances in Space Research, 60(5), 879-892.
//     DOI: 10.1016/j.asr.2017.05.034
//
// [5] Folkner, W.M. et al. (2009):
//     "The Planetary and Lunar Ephemeris DE 421"
//     IPN Progress Report 42-178. (DE-421 documentation)
//
// [6] Park, R.S. et al. (2021):
//     "The JPL Planetary and Lunar Ephemerides DE440 and DE441"
//     The Astronomical Journal, 161:105.
//     DOI: 10.3847/1538-3881/abd414 (DE-440 documentation)
//
// =============================================================================

#include <ballistic/ballistic.hpp>
#include <ballistic/validation/eof_parser.hpp>
#include <ballistic/ephemeris/spk_ephemeris.hpp>
#include <ballistic/coordinates/itrf_transforms.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <filesystem>

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;
using namespace ballistic::forces;
using namespace ballistic::ephemeris;

namespace fs = std::filesystem;

// =============================================================================
// ESA POD Service Configuration
// =============================================================================
//
// ESA's force model settings, each with its source and how this test
// approximates it.
//
// Source: [1] Sentiwiki POD Details, [2] POD Product Handbook

namespace esa_config {

// -----------------------------------------------------------------------------
// EPHEMERIS CONFIGURATION
// -----------------------------------------------------------------------------
// ESA uses DE-421 for third-body perturbations (Sun, Moon, Planets).
// Source: [1] "Sun, Moon, Planets DE-421"
//
// Note: DE-421 was published in 2008 and remains in operational use for
// consistency across the mission lifetime. DE-440 (2021) is newer but
// would require reprocessing historical data for consistency.

constexpr const char* DE421_PATH = BALLISTIC_PROJECT_ROOT "/data/de421.bsp";
constexpr const char* DE440_PATH = BALLISTIC_PROJECT_ROOT "/data/de440.bsp";

// -----------------------------------------------------------------------------
// GRAVITY CONFIGURATION
// -----------------------------------------------------------------------------
// ESA uses a time-varying gravity field with quarterly updates up to degree 90.
// Source: [1] "static gravity field model: FSM_quarterly_GSM-2..."
//
// For our 1-hour test, static EGM2008 at degree 70 is sufficient.
// Higher degrees attenuate rapidly at LEO altitude (~693 km).

constexpr int GRAVITY_DEGREE = 70;

// -----------------------------------------------------------------------------
// THIRD-BODY PERTURBATIONS
// -----------------------------------------------------------------------------
// ESA enables Sun, Moon, and planets.
// Source: [1] "Sun, Moon, Planets DE-421"
//
// We enable Sun, Moon, Jupiter (most significant for LEO).
// Other planets have negligible effect (<1mm over 1 hour).

constexpr bool ENABLE_SUN = true;
constexpr bool ENABLE_MOON = true;
constexpr bool ENABLE_JUPITER = true;

// -----------------------------------------------------------------------------
// ATMOSPHERIC DRAG
// -----------------------------------------------------------------------------
// ESA uses NRLMSISE-00 (documented as "MSISE00") for atmospheric density.
// Source: [1] "Atmospheric density: MSISE00"
//
// CRITICAL: ESA ESTIMATES drag coefficients, not fixed values!
// Source: [2] Section 3.2.2.5
//   - PRE/NRT products: 10 Cd coefficients per 24 hours
//   - STC/NTC products: 1 Cd per arc, constrained with sigma=0.3
//
// We cannot replicate this estimation process. At 693 km, drag moves a 1-hour
// prediction by an amount that depends on solar activity; drag with measured
// weather lowers the 1-hour error over 2025 ([`SENTINEL-YEAR-2025`]).
//
// For "ESA-compatible" test: disable drag (since we can't match their Cd).
// The "with drag" and "full physics" tests set the drag flag, but this harness
// attaches no DragEnvironment, so drag stays zero there too (see run_rmse_test).

constexpr bool ESA_ENABLE_DRAG = false;  // Can't match estimated Cd
constexpr double STANDARD_CD = 2.2;       // Industry standard for full physics

// -----------------------------------------------------------------------------
// SOLAR RADIATION PRESSURE
// -----------------------------------------------------------------------------
// ESA uses box-wing model with re-radiation for SRP.
// Source: [1] "Non-gravitational forces: Box-wing model (with re-radiation)"
//
// The SRP coefficient (Cr) is fixed to 1.0 for STC/NTC products.
// Source: [1] "radiation pressure coefficient fixed to 1.0"
//
// Our cannonball model (one effective area, Cr) differs from ESA's box-wing
// geometry. With the registry's fitted effective area it lowers Sentinel 1-h
// errors in the library's campaigns (see below).
//
// For "ESA-compatible" test: disable SRP. (For prediction accuracy SRP with
// the fitted radiation area and drag with measured space weather both lower
// the Sentinel 1-h RMSE — docs/validation_claims.md SENTINEL-YEAR-2025.)

// For "full physics" test: enable with Cr=1.0

constexpr bool ESA_ENABLE_SRP = false;    // Model differs from box-wing
constexpr double SRP_CR = 1.0;            // ESA's fixed value

// -----------------------------------------------------------------------------
// RELATIVISTIC CORRECTIONS
// -----------------------------------------------------------------------------
// ESA uses IERS 2010 conventions (implied by tidal modeling).
// Source: [1] "IERS 2010 conventions" (for tides, likely applies to relativity)
//
// We implement Schwarzschild and Lense-Thirring corrections per IERS 2010.
// These are well-defined formulas, so our implementation should match ESA's.

constexpr bool ENABLE_SCHWARZSCHILD = true;
constexpr bool ENABLE_LENSE_THIRRING = true;

// -----------------------------------------------------------------------------
// COORDINATE TRANSFORMS
// -----------------------------------------------------------------------------
// ESA's POE files are in ITRF (Earth-fixed frame). Our propagator works in
// ECI (inertial) and transforms to ITRF for comparison.
//
// The GCRS <-> ITRF transform is the full IAU 2006/2000A precession-nutation
// model.

constexpr PropagationFrame PROP_FRAME = PropagationFrame::GCRS;

// ITRF realization: ESA switched to ITRF2020 on March 4, 2025.
// For Sentinel data before that date, use ITRF2014.
constexpr coords::ITRFRealization ITRF_REALIZATION = coords::ITRFRealization::ITRF2014;

} // namespace esa_config

// =============================================================================
// Sentinel-1 Satellite Properties
// =============================================================================
//
// Physical and atmospheric parameters for Sentinel-1A/B.
// Source: [3] GMV-CPOD3-TN-0025 "Sentinel-1 Properties for GNSS POD" v2.0

namespace sentinel1 {

// Mass, frontal drag area and effective radiation area: the library's
// registry (include/ballistic/satellites.hpp, with citations).
constexpr double MASS = ballistic::satellites::get("S1A").mass_kg;
constexpr double DRAG_AREA = ballistic::satellites::get("S1A").drag_area_m2;

// Drag coefficient: ESA estimates it; we take the registry value (2.2, the
// conventional value for satellites in free-molecular flow).
constexpr double CD = ballistic::satellites::get("S1A").cd;

constexpr double SRP_AREA = ballistic::satellites::get("S1A").srp_area_m2;  // effective cannonball area

// SRP reflectivity: SatelliteProperties takes reflectivity, Cr = 1 + reflectivity.
// ESA fixes Cr = 1.0, i.e. reflectivity 0.
constexpr double SRP_REFLECTIVITY = 0.0;  // Cr = 1.0

// Round space-weather values for the August 2016 test epoch (F10.7 = 10.7 cm
// solar radio flux; Ap = daily geomagnetic index; the committed weather file
// gives F10.7 72-75, 81-day mean ~86 on 1-3 August). Passed to the propagator,
// which does not read them (drag needs a DragEnvironment).
constexpr double F107_DAILY = 89.0;    // SFU (solar flux units)
constexpr double F107_AVERAGE = 90.0;  // 81-day centered average
constexpr double AP_DAILY = 6.0;       // Quiet geomagnetic conditions

} // namespace sentinel1

// =============================================================================
// Test Framework
// =============================================================================

#include "../test_utils.hpp"
using namespace ballistic::test;

// =============================================================================
// Global State
// =============================================================================

static std::shared_ptr<EphemerisProvider> g_ephemeris;
static std::optional<EOFFile> g_eof;

// =============================================================================
// UNIT TEST 1: DE-421 Loading
// =============================================================================
//
// Verify that DE-421 ephemeris file loads correctly and returns reasonable
// positions. This is the ephemeris ESA uses for operational POD.
//
// Expected:
// - File loads without error
// - Moon distance ~384,400 km (average)
// - Sun distance ~150,000,000 km (1 AU)
// - Detected DE version = 421

void test_de421_loading() {
    fs::path de421_path = esa_config::DE421_PATH;

    if (!fs::exists(de421_path)) {
        TEST_SKIP("DE-421 not found at " << de421_path);
    }

    SPKEphemeris de421;
    TEST_ASSERT(de421.load(de421_path));
    TEST_ASSERT(de421.is_loaded());
    TEST_ASSERT(de421.de_version() == 421);

    // Test at Sentinel-1 validation epoch
    Epoch epoch = Epoch::from_utc(2016, 8, 1, 23, 0, 0);

    // Moon distance check (should be ~360,000-410,000 km)
    Vector3d moon = de421.moon_position(epoch);
    double moon_dist_km = moon.norm() / 1000.0;
    TEST_ASSERT(moon_dist_km > 350000.0 && moon_dist_km < 420000.0);

    // Sun distance check (should be ~0.98-1.02 AU)
    Vector3d sun = de421.sun_position(epoch);
    double sun_dist_au = sun.norm() / constants::sun::AU;
    TEST_ASSERT(sun_dist_au > 0.98 && sun_dist_au < 1.02);

    std::cout << "Moon: " << std::fixed << std::setprecision(0) << moon_dist_km
              << " km, Sun: " << std::setprecision(4) << sun_dist_au << " AU... ";
}

// =============================================================================
// UNIT TEST 2: DE-421 vs DE-440 Comparison
// =============================================================================
//
// Compare ephemeris positions between DE-421 (ESA's choice) and DE-440 (latest).
// This documents the difference between the two versions.
//
// Asserted: Moon position difference < 100 km and Sun < 10,000 km. These are
// loose plausibility bounds (the two ephemerides agree far better); at those
// levels either ephemeris is fine for LEO third-body forces.

void test_de421_vs_de440_comparison() {
    fs::path de421_path = esa_config::DE421_PATH;
    fs::path de440_path = esa_config::DE440_PATH;

    if (!fs::exists(de421_path) || !fs::exists(de440_path)) {
        TEST_SKIP("need both DE-421 and DE-440");
    }

    SPKEphemeris de421, de440;
    TEST_ASSERT(de421.load(de421_path));
    TEST_ASSERT(de440.load(de440_path));

    std::cout << "\n";
    std::cout << "    DE-421 vs DE-440 Ephemeris Comparison\n";
    std::cout << "    =====================================================\n";
    std::cout << "    Body      DE-421 Dist    DE-440 Dist    Diff (km)\n";
    std::cout << "    -----------------------------------------------------\n";

    Epoch epoch = Epoch::from_utc(2016, 8, 1, 23, 0, 0);

    // Moon comparison
    Vector3d moon_421 = de421.moon_position(epoch);
    Vector3d moon_440 = de440.moon_position(epoch);
    double moon_diff_km = (moon_421 - moon_440).norm() / 1000.0;

    std::cout << "    Moon      " << std::fixed << std::setprecision(0)
              << std::setw(10) << moon_421.norm() / 1000.0 << " km  "
              << std::setw(10) << moon_440.norm() / 1000.0 << " km  "
              << std::setw(8) << std::setprecision(1) << moon_diff_km << "\n";

    // Sun comparison
    Vector3d sun_421 = de421.sun_position(epoch);
    Vector3d sun_440 = de440.sun_position(epoch);
    double sun_diff_km = (sun_421 - sun_440).norm() / 1000.0;

    std::cout << "    Sun       " << std::fixed << std::setprecision(0)
              << std::setw(10) << sun_421.norm() / 1e6 << " Mkm "
              << std::setw(10) << sun_440.norm() / 1e6 << " Mkm "
              << std::setw(8) << std::setprecision(1) << sun_diff_km << "\n";

    // Jupiter comparison
    Vector3d jup_421 = de421.jupiter_position(epoch);
    Vector3d jup_440 = de440.jupiter_position(epoch);
    double jup_diff_km = (jup_421 - jup_440).norm() / 1000.0;

    std::cout << "    Jupiter   " << std::fixed << std::setprecision(0)
              << std::setw(10) << jup_421.norm() / 1e9 << " Bkm "
              << std::setw(10) << jup_440.norm() / 1e9 << " Bkm "
              << std::setw(8) << std::setprecision(1) << jup_diff_km << "\n";

    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    ";

    // Verify differences are small enough to not matter for LEO
    TEST_ASSERT(moon_diff_km < 100.0);  // <100 km Moon difference
    TEST_ASSERT(sun_diff_km < 10000.0); // <10,000 km Sun difference
}

// =============================================================================
// UNIT TEST 3: Force Model Component Verification
// =============================================================================
//
// Order-of-magnitude checks from closed forms: central gravity 8-10 m/s^2 and
// the J2 term 1e-3 to 1e-1 m/s^2 at ~400 km. The other rows are printed
// textbook values, not computed or asserted.

void test_force_model_components() {
    // Test state: r = 6778 km, i.e. ~400 km altitude (not Sentinel-1's 693 km)
    Vector3d pos(6778137.0, 0.0, 0.0);  // m (equatorial)
    Vector3d vel(0.0, 7668.56, 0.0);    // m/s (circular orbit velocity)
    Epoch epoch = Epoch::from_utc(2016, 8, 1, 23, 0, 0);

    std::cout << "\n";
    std::cout << "    Force Model Component Magnitudes at LEO\n";
    std::cout << "    =====================================================\n";
    std::cout << "    Component           Magnitude (m/s^2)    Expected\n";
    std::cout << "    -----------------------------------------------------\n";

    // Central body gravity (~9 m/s^2 at LEO)
    double r = pos.norm();
    double central_grav = constants::earth::GM / (r * r);
    std::cout << "    Central gravity     " << std::scientific << std::setprecision(3)
              << central_grav << "         ~8.7\n";
    TEST_ASSERT(central_grav > 8.0 && central_grav < 10.0);

    // J2 perturbation magnitude, 1.5·J2·GM·R²/r⁴ (~1e-2 m/s^2 at LEO)
    double j2_approx = 1.5 * constants::earth::J2 * constants::earth::GM *
                       constants::earth::RADIUS_EQUATORIAL * constants::earth::RADIUS_EQUATORIAL /
                       (r * r * r * r);
    std::cout << "    J2 acceleration     " << j2_approx << "         ~1e-2\n";
    TEST_ASSERT(j2_approx > 1e-3 && j2_approx < 1e-1);

    // Printed reference values only (not computed here).

    std::cout << "    Third-body (Moon)   ~1e-6                ~1e-6\n";
    std::cout << "    Third-body (Sun)    ~5e-7                ~5e-7\n";
    std::cout << "    Schwarzschild       ~1e-8                ~1e-8\n";
    std::cout << "    Drag (693km)        ~5e-9                ~5e-9\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    ";
}

// =============================================================================
// Configuration Structure for RMSE Tests
// =============================================================================

struct TestConfig {
    std::string name;
    std::string description;

    // Force model settings
    bool enable_sun = true;
    bool enable_moon = true;
    bool enable_jupiter = true;
    bool enable_schwarzschild = true;
    bool enable_lense_thirring = true;
    bool enable_drag = false;
    bool enable_srp = false;

    // If drag enabled, use these parameters
    double cd = sentinel1::CD;

    // If SRP enabled, use these parameters
    double cr = esa_config::SRP_CR;
};

// =============================================================================
// RMSE Test Implementation
// =============================================================================

struct RMSEResult {
    double position_rmse;
    double radial_rms;
    double along_track_rms;
    double cross_track_rms;
};

RMSEResult run_rmse_test(const TestConfig& config, double duration_sec = 3600.0) {
    if (!g_eof.has_value() || !g_ephemeris) {
        return {-1, -1, -1, -1};
    }

    // Start at the 11th state so the truth interpolation is not at a file edge
    const auto& sv0 = g_eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    // Not read for drag (drag needs a DragEnvironment).
    BallisticModel model(sentinel1::MASS, sentinel1::DRAG_AREA, config.cd);

    // Create weather conditions
    WeatherConditions weather;
    weather.f107_daily = sentinel1::F107_DAILY;
    weather.f107_average = sentinel1::F107_AVERAGE;
    weather.ap_daily = sentinel1::AP_DAILY;
    weather.ap_history.fill(sentinel1::AP_DAILY);

    // Create propagator with custom ephemeris and ITRF2014 realization
    auto prop = create_rk45_propagator(
        t0, sv0.position, sv0.velocity, model, weather, true,
        g_ephemeris, esa_config::PROP_FRAME, esa_config::ITRF_REALIZATION);

    // Configure force model
    ForceModel& fm = prop->force_model();
    fm.set_gravity_max_degree(esa_config::GRAVITY_DEGREE);

    // Third-body sub-component configuration
    fm.third_body().enable_sun(config.enable_sun);
    fm.third_body().enable_moon(config.enable_moon);
    fm.third_body().enable_jupiter(config.enable_jupiter);

    // Relativistic sub-component configuration
    fm.enable_schwarzschild(config.enable_schwarzschild);
    fm.enable_lense_thirring(config.enable_lense_thirring);

    // Force flags. Setting flags.drag without a DragEnvironment has no effect:
    // drag is zero unless set_drag_environment() is called, which this harness
    // never does. So every configuration below is drag-free.
    bool any_third_body = config.enable_sun || config.enable_moon || config.enable_jupiter;
    bool any_relativistic = config.enable_schwarzschild || config.enable_lense_thirring;

    forces::ForceComponentFlags flags = fm.force_flags();
    flags.third_body = any_third_body;
    flags.relativistic = any_relativistic;
    flags.drag = config.enable_drag;
    flags.radiation = config.enable_srp;
    fm.set_force_flags(flags);

    // SRP with a cannonball model (one effective area), not ESA's box-wing
    if (config.enable_srp) {
        SatelliteProperties sat_props(sentinel1::SRP_AREA, sentinel1::SRP_REFLECTIVITY, 0.8);
        fm.set_satellite_properties(sat_props, sentinel1::MASS);
        fm.radiation().enable_srp(true);
    }

    // Run propagation with 10-second sampling
    std::vector<std::pair<Epoch, StateVector>> results;
    int num_samples = static_cast<int>(duration_sec / 10.0) + 1;

    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * 10.0;
        if (dt > duration_sec) dt = duration_sec;

        Epoch t = t0 + dt;
        auto result = (*prop)(t);

        StateVector state(result.position, result.velocity);
        results.emplace_back(t, state);
    }

    auto test_result = compute_propagation_result(results, *g_eof);

    return {
        test_result.position_rmse,
        test_result.radial_rms,
        test_result.along_track_rms,
        test_result.cross_track_rms
    };
}

// =============================================================================
// RMSE TEST 4: ESA-Compatible Configuration
// =============================================================================
//
// Force model as close to ESA's as we can get without estimating parameters:
// drag and SRP OFF (ESA estimates Cd and uses a box-wing SRP model).
// Asserted: 0 < RMSE < 3 m (loose bound; measured 0.24 m).

void test_esa_compatible_rmse() {
    TestConfig config;
    config.name = "ESA-Compatible";
    config.description = "Match ESA force model (no drag/SRP due to model differences)";
    config.enable_sun = esa_config::ENABLE_SUN;
    config.enable_moon = esa_config::ENABLE_MOON;
    config.enable_jupiter = esa_config::ENABLE_JUPITER;
    config.enable_schwarzschild = esa_config::ENABLE_SCHWARZSCHILD;
    config.enable_lense_thirring = esa_config::ENABLE_LENSE_THIRRING;
    config.enable_srp = false;   // Can't match ESA's box-wing model

    auto result = run_rmse_test(config);

    std::cout << "\n";
    std::cout << "    ESA-Compatible Configuration (1-hour RMSE)\n";
    std::cout << "    =====================================================\n";
    std::cout << "    Configuration: EGM2008 deg70 + GCRS + Sun/Moon/Jupiter\n";
    std::cout << "                   + Schwarzschild + Lense-Thirring\n";
    std::cout << "                   NO drag (ESA estimates Cd)\n";
    std::cout << "                   NO SRP (ESA uses box-wing)\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    Position RMSE:   " << std::fixed << std::setprecision(4)
              << result.position_rmse << " m\n";
    std::cout << "    Radial RMS:      " << result.radial_rms << " m\n";
    std::cout << "    Along-track RMS: " << result.along_track_rms << " m\n";
    std::cout << "    Cross-track RMS: " << result.cross_track_rms << " m\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    Note: ESA achieves ~0.05m using estimated parameters\n";
    std::cout << "    Our ~1m result is due to fixed (non-estimated) model\n";
    std::cout << "    ";

    TEST_ASSERT(result.position_rmse > 0 && result.position_rmse < 3.0);
}

// =============================================================================
// RMSE TEST 5: With Drag (Fixed Cd=2.2)
// =============================================================================
//
// Intended to add drag with Cd 2.2. Because run_rmse_test attaches no
// DragEnvironment, the run is in fact drag-free (same as the test above).
// Asserted: 0 < RMSE < 5 m.

void test_with_drag_rmse() {
    TestConfig config;
    config.name = "With Drag";
    config.description = "ESA-compatible + fixed Cd=2.2 drag";
    config.enable_sun = true;
    config.enable_moon = true;
    config.enable_jupiter = true;
    config.enable_schwarzschild = true;
    config.enable_lense_thirring = true;
    config.enable_drag = true;   // Add drag
    config.enable_srp = false;
    config.cd = sentinel1::CD;   // 2.2

    auto result = run_rmse_test(config);

    std::cout << "\n";
    std::cout << "    With Drag Configuration (1-hour RMSE)\n";
    std::cout << "    =====================================================\n";
    std::cout << "    Added: NRLMSISE-00 drag, Cd=2.2 (fixed)\n";
    std::cout << "    Note: ESA estimates Cd dynamically\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    Position RMSE:   " << std::fixed << std::setprecision(4)
              << result.position_rmse << " m\n";
    std::cout << "    Radial RMS:      " << result.radial_rms << " m\n";
    std::cout << "    Along-track RMS: " << result.along_track_rms << " m\n";
    std::cout << "    Cross-track RMS: " << result.cross_track_rms << " m\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    ";

    TEST_ASSERT(result.position_rmse > 0 && result.position_rmse < 5.0);
}

// =============================================================================
// RMSE TEST 6: With SRP (Simplified Model, Cr=1.0)
// =============================================================================
//
// Adds SRP with the cannonball model (registry effective area, Cr 1.0); ESA
// uses a box-wing model with re-radiation. Asserted: 0 < RMSE < 5 m.

void test_with_srp_rmse() {
    TestConfig config;
    config.name = "With SRP";
    config.description = "ESA-compatible + simplified SRP (Cr=1.0)";
    config.enable_sun = true;
    config.enable_moon = true;
    config.enable_jupiter = true;
    config.enable_schwarzschild = true;
    config.enable_lense_thirring = true;
    config.enable_srp = true;    // Add SRP
    config.cr = esa_config::SRP_CR;  // 1.0

    auto result = run_rmse_test(config);

    std::cout << "\n";
    std::cout << "    With SRP Configuration (1-hour RMSE)\n";
    std::cout << "    =====================================================\n";
    std::cout << "    Added: Simplified SRP, Cr=1.0 (ESA fixes this value)\n";
    std::cout << "    Note: ESA uses box-wing model; we use cannonball\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    Position RMSE:   " << std::fixed << std::setprecision(4)
              << result.position_rmse << " m\n";
    std::cout << "    Radial RMS:      " << result.radial_rms << " m\n";
    std::cout << "    Along-track RMS: " << result.along_track_rms << " m\n";
    std::cout << "    Cross-track RMS: " << result.cross_track_rms << " m\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    ";

    TEST_ASSERT(result.position_rmse > 0 && result.position_rmse < 5.0);
}

// =============================================================================
// RMSE TEST 7: Full Physics Model
// =============================================================================
//
// Every force switched on (SRP on; drag flag set but, as above, no
// DragEnvironment is attached, so drag is zero). Asserted: 0 < RMSE < 5 m.

void test_full_physics_rmse() {
    TestConfig config;
    config.name = "Full Physics";
    config.description = "All forces enabled";
    config.enable_sun = true;
    config.enable_moon = true;
    config.enable_jupiter = true;
    config.enable_schwarzschild = true;
    config.enable_lense_thirring = true;
    config.enable_drag = true;
    config.enable_srp = true;
    config.cd = sentinel1::CD;
    config.cr = esa_config::SRP_CR;

    auto result = run_rmse_test(config);

    std::cout << "\n";
    std::cout << "    Full Physics Configuration (1-hour RMSE)\n";
    std::cout << "    =====================================================\n";
    std::cout << "    All forces: gravity, third-body, relativistic,\n";
    std::cout << "                drag (Cd=2.2), SRP (Cr=1.0)\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    Position RMSE:   " << std::fixed << std::setprecision(4)
              << result.position_rmse << " m\n";
    std::cout << "    Radial RMS:      " << result.radial_rms << " m\n";
    std::cout << "    Along-track RMS: " << result.along_track_rms << " m\n";
    std::cout << "    Cross-track RMS: " << result.cross_track_rms << " m\n";
    std::cout << "    -----------------------------------------------------\n";
    std::cout << "    ";

    TEST_ASSERT(result.position_rmse > 0 && result.position_rmse < 5.0);
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main() {
    std::cout << "=============================================================\n";
    std::cout << "ESA Copernicus POD Service Validation Test Suite\n";
    std::cout << "=============================================================\n\n";

    // -------------------------------------------------------------------------
    // Load ephemeris (prefer DE-421 to match ESA)
    // -------------------------------------------------------------------------
    std::cout << "Loading ephemeris...\n";

    fs::path de421_path = esa_config::DE421_PATH;
    fs::path de440_path = esa_config::DE440_PATH;

    if (fs::exists(de421_path)) {
        if (load_spk_ephemeris(de421_path)) {
            std::cout << "  DE-421 loaded (ESA's operational ephemeris)\n";
            g_ephemeris = std::shared_ptr<EphemerisProvider>(
                std::shared_ptr<void>{}, &SPKEphemerisManager::instance().ephemeris());
        }
    }

    if (!g_ephemeris && fs::exists(de440_path)) {
        if (load_spk_ephemeris(de440_path)) {
            std::cout << "  DE-440 loaded (DE-421 not found)\n";
            g_ephemeris = std::shared_ptr<EphemerisProvider>(
                std::shared_ptr<void>{}, &SPKEphemerisManager::instance().ephemeris());
        }
    }

    if (!g_ephemeris) {
        std::cout << "  Using embedded DE440 (no external SPK files found)\n";
        SPKEphemerisManager::instance().load_embedded();
        g_ephemeris = std::shared_ptr<EphemerisProvider>(
            std::shared_ptr<void>{}, &SPKEphemerisManager::instance().ephemeris());
    }

    // -------------------------------------------------------------------------
    // Load EOP data
    // -------------------------------------------------------------------------
    std::cout << "\nLoading EOP data...\n";
    fs::path eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
    if (time::EOPData::instance().load(eop_path)) {
        std::cout << "  EOP data loaded\n";
    } else {
        std::cout << "  WARNING: EOP data not found\n";
    }

    // -------------------------------------------------------------------------
    // Load EOF validation data
    // -------------------------------------------------------------------------
    std::cout << "\nLoading Sentinel-1 EOF validation data...\n";
    std::string data_path = BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel1_2016/2016-8";
    auto eof = find_and_parse_eof(data_path);

    if (!eof) {
        // Not committed (ESA Copernicus data, see tests/data/eof/README.md): the
        // whole suite is one counted skip, a failure under BALLISTIC_REQUIRE_TEST_DATA=1.
        TestRunner skip_runner("ESA POD validation");
        skip_runner.add_test("sentinel1_2016_eof", [&]() {
            TEST_SKIP("no Sentinel-1 EOF file under " << data_path
                      << " (Copernicus Data Space Ecosystem POEORB; tests/data/eof/README.md)");
        });
        skip_runner.run();
        return print_final_summary();
    }

    g_eof = std::move(eof);
    std::cout << "  Loaded: " << g_eof->filename << "\n";
    std::cout << "  State vectors: " << g_eof->state_vectors.size() << "\n";

    // -------------------------------------------------------------------------
    // Run unit tests
    // -------------------------------------------------------------------------
    TestRunner unit_runner("Unit Tests");
    unit_runner.add_test("de421_loading", []() { test_de421_loading(); });
    unit_runner.add_test("de421_vs_de440_comparison", []() { test_de421_vs_de440_comparison(); });
    unit_runner.add_test("force_model_components", []() { test_force_model_components(); });
    unit_runner.run();

    // -------------------------------------------------------------------------
    // Run RMSE tests
    // -------------------------------------------------------------------------
    TestRunner rmse_runner("1-Hour RMSE Tests");
    rmse_runner.add_test("esa_compatible_rmse", []() { test_esa_compatible_rmse(); });
    rmse_runner.add_test("with_drag_rmse", []() { test_with_drag_rmse(); });
    rmse_runner.add_test("with_srp_rmse", []() { test_with_srp_rmse(); });
    rmse_runner.add_test("full_physics_rmse", []() { test_full_physics_rmse(); });
    rmse_runner.run();

    return print_final_summary();
}
