// =============================================================================
// QZSS HEO (Highly Elliptical Orbit) Validation Tests
// =============================================================================
//
// Propagates QZSS (Quasi-Zenith Satellite System) satellites and compares them
// with precise orbits from the IGS multi-GNSS experiment (MGEX). Asserted:
// 1-hour RMSE (root-mean-square position error, sampled every 5 min) < 100 m
// and 6-hour RMSE < 500 m for the first satellite; two cells check the SP3
// file and the derived velocity, and all_qzss_satellites reports each
// satellite without asserting. These are loose regression bounds; the
// library's measured 1-hour QZSS figure (0.739 m over 276 starts) is
// [`GNSS-1H`] in docs/validation_claims.md (Python script).
//
// Data: tests/data/sp3/qzss/ (CODE MGEX final orbits, SP3 format = the standard
// precise-orbit file: positions every 5 min). EOP = Earth orientation
// parameters; GCRS = inertial frame of integration; ITRF2020 = Earth-fixed
// frame of the SP3 positions.
//
// QZSS Orbit Characteristics:
// - Type: inclined geosynchronous orbit (IGSO; the code calls it "HEO"),
//   mildly eccentric, Tundra-like
// - Perigee: ~32,000 km
// - Apogee: ~40,000 km
// - Inclination: ~43°
// - Period: ~23h 56m (geosynchronous)
// - Ground track: Asymmetric figure-8 over Japan
//
// Data Source:
// - CODE MGEX Final Products (COD0MGXFIN)
// - URL: http://ftp.aiub.unibe.ch/CODE_MGEX/CODE/
// - Format: SP3-d (position only, velocity derived by Lagrange interpolation)
// - Sampling: 300 seconds (5 minutes)
// - Accuracy: ~0.7m orbit consistency, ~0.15m SLR residuals
//
// QZSS satellites (SP3 ID Jnn = PRN - 192):
// - J01: QZS-1 (Michibiki 1, IGSO; since retired)
// - J02: QZS-2 (IGSO, launched 2017)
// - J03: QZS-4 (IGSO, launched 2017)
// - J04: QZS-1R (IGSO, launched 2021, replacement for QZS-1)
// - J07: QZS-3 (GEO, launched 2017; CODE MGEX files carry no GEO orbits)
// NOTE: the code below excludes J03 as "GEO"; J03 is in fact the IGSO QZS-4,
// so it is simply left out of these cells.
//
// =============================================================================

#include <ballistic/ballistic.hpp>
#include "../sp3_truth.hpp"
#include <ballistic/ephemeris/spk_ephemeris.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;
using namespace ballistic::ephemeris;

namespace fs = std::filesystem;

// =============================================================================
// Test Configuration
// =============================================================================

namespace QZSSConfig {
    // Approximate spacecraft parameters (documentation only: no cell uses the
    // mass, area or coefficients; drag and SRP are off)
    constexpr double MASS = 4000.0;           // kg (QZS-2 dry mass ~4000 kg)
    constexpr double SRP_AREA = 40.0;         // m^2 (solar array area)
    constexpr double CD = 2.2;                // Drag coefficient (irrelevant at HEO)
    constexpr double CR = 1.0;                // radiation pressure coefficient Cr

    // Orbital parameters
    constexpr double PERIGEE_ALT = 32000.0;   // km
    constexpr double APOGEE_ALT = 40000.0;    // km
    constexpr double INCLINATION = 43.0;      // degrees

    // Test configuration
    constexpr int GRAVITY_DEGREE = 8;         // Low degree sufficient at HEO altitude
}

// =============================================================================
// Test Framework
// =============================================================================

#include "../test_utils.hpp"
using namespace ballistic::test;

// Global ephemeris and SP3 data
static std::shared_ptr<EphemerisProvider> g_ephemeris;
static std::optional<ballistic::test::TruthFile> g_sp3;

// =============================================================================
// Utility Functions
// =============================================================================

std::string get_qzss_data_path() {
    std::vector<std::string> candidates = {
        BALLISTIC_PROJECT_ROOT "/tests/data/sp3/qzss",
        "../tests/data/sp3/qzss",
        "tests/data/sp3/qzss"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }

    return BALLISTIC_PROJECT_ROOT "/tests/data/sp3/qzss";
}

std::optional<ballistic::test::TruthFile> load_sp3_file() {
    std::string data_path = get_qzss_data_path();

    if (!fs::exists(data_path)) {
        return std::nullopt;
    }

    // Find SP3 file
    for (const auto& entry : fs::directory_iterator(data_path)) {
        std::string filename = entry.path().filename().string();
        if (filename.find(".SP3") != std::string::npos ||
            filename.find(".sp3") != std::string::npos) {
            auto result = ballistic::test::load_sp3_truth(entry.path().string());
            if (result) {
                std::cout << "Loaded SP3: " << filename << std::endl;
                return result;
            }
        }
    }

    return std::nullopt;
}

bool load_jpl_ephemeris() {
    fs::path de421_path = BALLISTIC_PROJECT_ROOT "/data/de421.bsp";
    fs::path de440_path = BALLISTIC_PROJECT_ROOT "/data/de440.bsp";

    if (fs::exists(de421_path)) {
        auto spk = std::make_shared<SPKEphemeris>();
        if (spk->load(de421_path)) {
            g_ephemeris = spk;
            std::cout << "Loaded JPL DE421 ephemeris" << std::endl;
            return true;
        }
    }

    if (fs::exists(de440_path)) {
        auto spk = std::make_shared<SPKEphemeris>();
        if (spk->load(de440_path)) {
            g_ephemeris = spk;
            std::cout << "Loaded JPL DE440 ephemeris" << std::endl;
            return true;
        }
    }

    std::cout << "WARNING: JPL ephemeris not found" << std::endl;
    return false;
}

// List the QZSS satellites in the file, leaving out J03 (see the file header:
// J03 is the IGSO QZS-4, not a GEO as the code assumes)
std::vector<std::string> get_qzss_heo_satellites(const ballistic::test::TruthFile& sp3) {
    std::vector<std::string> heo_sats;
    for (const auto& sat_id : sp3.header().satellites) {
        if (sat_id.size() >= 1 && sat_id[0] == 'J') {
            if (sat_id != "J03") {
                heo_sats.push_back(sat_id);
            }
        }
    }
    return heo_sats;
}

// Compute RMSE for propagation vs SP3 truth
struct QZSSValidationResult {
    double position_rmse;
    double velocity_rmse;
    double radial_rms;
    double along_track_rms;
    double cross_track_rms;
    double max_position_error;
    int num_comparisons;
};

QZSSValidationResult compute_qzss_rmse(
    const std::vector<std::pair<Epoch, StateVector>>& predicted,
    const ballistic::test::TruthFile& sp3,
    const std::string& sat_id,
    double max_time_diff = 10.0)
{
    QZSSValidationResult result = {};

    double pos_sum_sq = 0.0;
    double vel_sum_sq = 0.0;
    double radial_sum_sq = 0.0;
    double along_sum_sq = 0.0;
    double cross_sum_sq = 0.0;

    for (const auto& [pred_epoch, pred_state] : predicted) {
        // Interpolate SP3 truth at this epoch
        auto truth_state = sp3.interpolate_state(sat_id, pred_epoch);
        if (!truth_state) continue;

        Vector3d pos_error = pred_state.position - truth_state->position;
        Vector3d vel_error = pred_state.velocity - truth_state->velocity;

        double pos_err_mag = pos_error.norm();
        pos_sum_sq += pos_err_mag * pos_err_mag;
        vel_sum_sq += vel_error.squaredNorm();
        result.max_position_error = std::max(result.max_position_error, pos_err_mag);

        // Compute RTN (Radial-Transverse-Normal) components
        Vector3d r_hat = pred_state.position.normalized();
        Vector3d h = pred_state.position.cross(pred_state.velocity);
        Vector3d n_hat = h.normalized();
        Vector3d t_hat = n_hat.cross(r_hat);

        double radial_err = pos_error.dot(r_hat);
        double along_err = pos_error.dot(t_hat);
        double cross_err = pos_error.dot(n_hat);

        radial_sum_sq += radial_err * radial_err;
        along_sum_sq += along_err * along_err;
        cross_sum_sq += cross_err * cross_err;

        result.num_comparisons++;
    }

    if (result.num_comparisons > 0) {
        result.position_rmse = std::sqrt(pos_sum_sq / result.num_comparisons);
        result.velocity_rmse = std::sqrt(vel_sum_sq / result.num_comparisons);
        result.radial_rms = std::sqrt(radial_sum_sq / result.num_comparisons);
        result.along_track_rms = std::sqrt(along_sum_sq / result.num_comparisons);
        result.cross_track_rms = std::sqrt(cross_sum_sq / result.num_comparisons);
    }

    return result;
}

// =============================================================================
// Tests
// =============================================================================

// Test: SP3 file parsing and QZSS satellite detection
void test_sp3_parsing() {
    if (!g_sp3) {
        TEST_SKIP("no SP3 data");
    }

    auto heo_sats = get_qzss_heo_satellites(*g_sp3);
    std::cout << "(" << heo_sats.size() << " HEO sats: ";
    for (size_t i = 0; i < heo_sats.size(); ++i) {
        std::cout << heo_sats[i];
        if (i < heo_sats.size() - 1) std::cout << ",";
    }
    std::cout << ") ";

    TEST_ASSERT(!heo_sats.empty());
    TEST_ASSERT(g_sp3->records.size() > 0);
}

// Test: Velocity derivation from SP3 position data
void test_velocity_interpolation() {
    if (!g_sp3) {
        TEST_SKIP("no SP3 data");
    }

    auto heo_sats = get_qzss_heo_satellites(*g_sp3);
    if (heo_sats.empty()) {
        TEST_SKIP("no HEO sats");
    }

    // Get first satellite's records
    auto records = g_sp3->get_satellite_records(heo_sats[0]);
    if (records.size() < 10) {
        TEST_SKIP("insufficient records");
    }

    // Test velocity derivation at middle epoch
    const auto& mid_rec = records[records.size() / 2];
    auto vel = g_sp3->interpolate_velocity(heo_sats[0], mid_rec.epoch);

    TEST_ASSERT_MSG(vel.has_value(), "velocity interpolation returned nullopt");

    // Check velocity magnitude is reasonable for HEO (should be ~1-4 km/s)
    double vel_mag = vel->norm() / 1000.0;  // km/s
    std::cout << "(" << heo_sats[0] << " vel=" << std::fixed << std::setprecision(2)
              << vel_mag << " km/s) ";

    TEST_ASSERT(vel_mag > 0.5 && vel_mag < 5.0);
}

// Test: 1-hour HEO propagation vs SP3 truth
void test_qzss_1hour_propagation() {
    if (!g_sp3) {
        TEST_SKIP("no SP3 data");
    }

    auto heo_sats = get_qzss_heo_satellites(*g_sp3);
    if (heo_sats.empty()) {
        TEST_SKIP("no HEO sats");
    }

    std::string sat_id = heo_sats[0];
    auto records = g_sp3->get_satellite_records(sat_id);
    if (records.size() < 20) {
        TEST_SKIP("insufficient records");
    }

    // Start from 10th record, propagate for 1 hour
    const auto& start_rec = records[10];
    Epoch t0 = start_rec.epoch;

    // Start state from the SP3 interpolator (position plus a velocity derived
    // from the positions, since the file has no velocity records)
    auto init_state = g_sp3->interpolate_state(sat_id, t0);
    if (!init_state) {
        TEST_SKIP("interpolation failed");
    }

    // Gravity degree 8 + Sun/Moon/Jupiter + relativity; no drag, no SRP
    BallisticModel model;  // No drag at HEO (controlled by force flags)
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = std::make_unique<RK45Propagator>(
        t0, init_state->position, init_state->velocity,
        model, weather,
        true,  // enable_third_body (critical for HEO)
        RK45Config{},
        g_ephemeris,
        PropagationFrame::GCRS,
        coords::ITRFRealization::ITRF2020
    );

    auto& fm = prop->force_model();
    fm.set_gravity_max_degree(QZSSConfig::GRAVITY_DEGREE);
    {
        forces::ForceComponentFlags flags = fm.force_flags();
        flags.relativistic = true;
        flags.drag = false;
        fm.set_force_flags(flags);
    }

    // Propagate and sample every 5 minutes
    std::vector<std::pair<Epoch, StateVector>> results;
    results.emplace_back(t0, *init_state);

    double duration = 3600.0;  // 1 hour
    for (double dt = 300.0; dt <= duration; dt += 300.0) {
        auto result = (*prop)(t0 + dt);
        results.emplace_back(t0 + dt, StateVector(result.position, result.velocity));
    }

    // Compute RMSE
    auto validation = compute_qzss_rmse(results, *g_sp3, sat_id);

    std::cout << "\n    " << sat_id << " 1-hour propagation:";
    std::cout << "\n    Position RMSE:   " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m";
    std::cout << "\n    Velocity RMSE:   " << std::setprecision(4)
              << validation.velocity_rmse << " m/s";
    std::cout << "\n    Radial RMS:      " << std::setprecision(2) << validation.radial_rms << " m";
    std::cout << "\n    Along-track RMS: " << validation.along_track_rms << " m";
    std::cout << "\n    Cross-track RMS: " << validation.cross_track_rms << " m";
    std::cout << "\n    Max error:       " << validation.max_position_error << " m";
    std::cout << "\n    Comparisons:     " << validation.num_comparisons;

    if (g_ephemeris) {
        std::cout << "\n    (using JPL ephemeris)";
    }
    std::cout << std::endl << "    ";

    // Loose bound: without SRP the measured 1-hour error is under a meter
    // ([`GNSS-1H`]); 100 m catches a broken frame, time scale or force model.
    TEST_ASSERT(validation.position_rmse < 100.0);
    TEST_ASSERT(validation.num_comparisons >= 10);
}

// Test: 6-hour HEO propagation (quarter orbital period)
void test_qzss_6hour_propagation() {
    if (!g_sp3) {
        TEST_SKIP("no SP3 data");
    }

    auto heo_sats = get_qzss_heo_satellites(*g_sp3);
    if (heo_sats.empty()) {
        TEST_SKIP("no HEO sats");
    }

    std::string sat_id = heo_sats[0];
    auto records = g_sp3->get_satellite_records(sat_id);
    if (records.size() < 100) {
        TEST_SKIP("insufficient records for 6-hour test");
    }

    const auto& start_rec = records[10];
    Epoch t0 = start_rec.epoch;

    auto init_state = g_sp3->interpolate_state(sat_id, t0);
    if (!init_state) {
        TEST_SKIP("interpolation failed");
    }

    BallisticModel model;  // No drag (controlled by force flags)
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = std::make_unique<RK45Propagator>(
        t0, init_state->position, init_state->velocity,
        model, weather, true, RK45Config{}, g_ephemeris,
        PropagationFrame::GCRS, coords::ITRFRealization::ITRF2020
    );

    auto& fm = prop->force_model();
    fm.set_gravity_max_degree(QZSSConfig::GRAVITY_DEGREE);
    {
        forces::ForceComponentFlags flags = fm.force_flags();
        flags.relativistic = true;
        flags.drag = false;
        fm.set_force_flags(flags);
    }

    std::vector<std::pair<Epoch, StateVector>> results;
    results.emplace_back(t0, *init_state);

    double duration = 6.0 * 3600.0;  // 6 hours
    for (double dt = 300.0; dt <= duration; dt += 300.0) {
        auto result = (*prop)(t0 + dt);
        results.emplace_back(t0 + dt, StateVector(result.position, result.velocity));
    }

    auto validation = compute_qzss_rmse(results, *g_sp3, sat_id);

    std::cout << "\n    " << sat_id << " 6-hour propagation:";
    std::cout << "\n    Position RMSE:   " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m";
    std::cout << "\n    Max error:       " << validation.max_position_error << " m";
    std::cout << "\n    Comparisons:     " << validation.num_comparisons;
    std::cout << std::endl << "    ";

    // Loose bound for 6 hours (unmodeled SRP grows the error over time)
    TEST_ASSERT(validation.position_rmse < 500.0);
}

// Test: All QZSS HEO satellites
void test_all_qzss_satellites() {
    if (!g_sp3) {
        TEST_SKIP("no SP3 data");
    }

    auto heo_sats = get_qzss_heo_satellites(*g_sp3);
    if (heo_sats.empty()) {
        TEST_SKIP("no HEO sats");
    }

    std::cout << "\n";

    for (const auto& sat_id : heo_sats) {
        auto records = g_sp3->get_satellite_records(sat_id);
        if (records.size() < 20) continue;

        const auto& start_rec = records[10];
        Epoch t0 = start_rec.epoch;

        auto init_state = g_sp3->interpolate_state(sat_id, t0);
        if (!init_state) continue;

        BallisticModel model;  // No drag (controlled by force flags)
        WeatherConditions weather = WeatherConditions::quiet();

        auto prop = std::make_unique<RK45Propagator>(
            t0, init_state->position, init_state->velocity,
            model, weather, true, RK45Config{}, g_ephemeris,
            PropagationFrame::GCRS, coords::ITRFRealization::ITRF2020
        );

        prop->force_model().set_gravity_max_degree(QZSSConfig::GRAVITY_DEGREE);
        {
            forces::ForceComponentFlags flags = prop->force_model().force_flags();
            flags.relativistic = true;
            flags.drag = false;
            prop->force_model().set_force_flags(flags);
        }

        std::vector<std::pair<Epoch, StateVector>> results;
        results.emplace_back(t0, *init_state);

        for (double dt = 300.0; dt <= 3600.0; dt += 300.0) {
            auto result = (*prop)(t0 + dt);
            results.emplace_back(t0 + dt, StateVector(result.position, result.velocity));
        }

        auto validation = compute_qzss_rmse(results, *g_sp3, sat_id);

        // Get altitude range
        double r_min = std::numeric_limits<double>::max();
        double r_max = 0.0;
        for (const auto& rec : records) {
            double r = rec.position.norm();
            r_min = std::min(r_min, r);
            r_max = std::max(r_max, r);
        }
        double alt_min = (r_min - constants::earth::RADIUS_EQUATORIAL) / 1000.0;
        double alt_max = (r_max - constants::earth::RADIUS_EQUATORIAL) / 1000.0;

        std::cout << "    " << sat_id << ": RMSE=" << std::fixed << std::setprecision(1)
                  << validation.position_rmse << "m, alt="
                  << std::setprecision(0) << alt_min << "-" << alt_max << " km\n";
    }

    std::cout << "    ";
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    // Load EOP data
    fs::path eop_path = BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all";
    if (time::EOPData::instance().load(eop_path)) {
        std::cout << "EOP data loaded" << std::endl;
    }

    // Load JPL ephemeris
    load_jpl_ephemeris();

    // Load SP3 data
    g_sp3 = load_sp3_file();
    if (!g_sp3) {
        std::cout << "SP3 data not found at " << get_qzss_data_path() << std::endl;
        std::cout << "Run: python scripts/download_qzss_sp3.py --date 2024-01-15" << std::endl;
    }

    TestRunner runner("QZSS HEO Validation Tests");

    runner.add_test("sp3_parsing", []() { test_sp3_parsing(); });
    runner.add_test("velocity_interpolation", []() { test_velocity_interpolation(); });
    runner.add_test("qzss_1hour_propagation", []() { test_qzss_1hour_propagation(); });
    runner.add_test("qzss_6hour_propagation", []() { test_qzss_6hour_propagation(); });
    runner.add_test("all_qzss_satellites", []() { test_all_qzss_satellites(); });

    runner.run();
    return print_final_summary();
}
