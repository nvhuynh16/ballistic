// =============================================================================
// GEO Orbit Propagation Validation Tests (BeiDou C01, Wuhan MGEX)
// =============================================================================
//
// Validates propagation at geostationary (GEO) altitude against the committed
// precise orbits of BeiDou C01 from Wuhan University's MGEX (Multi-GNSS
// Experiment) final product, in SP3 format (the standard text format for
// precise satellite orbits): tests/data/sp3/beidou/ (public, no login; the
// same data as the GEO row of [`GNSS-1H`] in docs/validation_claims.md).
// BeiDou C01 is a true GEO (inclination < ~2 deg). QZSS J03 is NOT used: it
// is QZS-4, on an inclined geosynchronous orbit (IGSO, i = 40.6 deg).
//
// Key physics at GEO: no drag; SRP (solar radiation pressure) is the largest
// non-gravitational force (OFF here: no sourced BeiDou properties, as in
// GNSS-1H); gravity degree 4-12 suffices; third body several 1e-6 m/s^2
// (tidal terms: Moon ~7e-6, Sun ~3e-6). Start
// velocities are Lagrange derivatives of the 5-min positions at a centered
// node (this SP3 has no velocities). Errors are position RMSE (root-mean-
// square error) against the SP3 positions.
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

// Global JPL ephemeris (loaded in main)
static std::shared_ptr<EphemerisProvider> g_ephemeris;

// =============================================================================
// GEO Configuration
// =============================================================================

// Typical GEO satellite parameters (generic values; SRP is off in these tests)
struct GEOConfig {
    static constexpr double MASS = 3000.0;           // kg (typical GEO comsat)
    static constexpr double DRAG_AREA = 0.0;         // No drag at GEO
    static constexpr double SRP_AREA = 50.0;         // m^2 (large solar arrays)
    static constexpr double CD = 0.0;                // No drag
    static constexpr double CR = 1.5;                // SRP reflectivity

    // Orbit characteristics
    static constexpr double ALTITUDE = 35786000.0;   // m
    static constexpr double SEMI_MAJOR_AXIS = 42164000.0; // m
    static constexpr double PERIOD = 86164.1;        // s (sidereal day)

    // Recommended gravity degree for GEO
    static constexpr int GRAVITY_DEGREE = 12;

    // Solar activity (typical)
    static constexpr double F107_DAILY = 100.0;
    static constexpr double F107_AVERAGE = 100.0;
    static constexpr double AP_DAILY = 10.0;
};

// =============================================================================
// Test Framework
// =============================================================================

#include "../test_utils.hpp"
using namespace ballistic::test;

// =============================================================================
// Utility Functions
// =============================================================================

static const char* const kGeoSat = "C01";

std::string get_beidou_data_path() {
    return BALLISTIC_PROJECT_ROOT "/tests/data/sp3/beidou";
}

// The first daily file (sorted by name = by day).
std::optional<ballistic::test::TruthFile> load_sp3_file() {
    const fs::path dir = get_beidou_data_path();
    if (!fs::is_directory(dir)) return std::nullopt;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().find("_ORB.SP3") != std::string::npos) files.push_back(e.path());
    if (files.empty()) return std::nullopt;
    std::sort(files.begin(), files.end());
    return ballistic::test::load_sp3_truth(files.front());
}

// A centered start: SP3 carries no velocities, so the state is the 9-point
// Lagrange interpolant and its derivative at a node well inside the file.
std::optional<StateVector> start_state(const ballistic::test::TruthFile& sp3, const Epoch& t0) {
    return sp3.interpolate_state(kGeoSat, t0);
}

// Check if satellite is at GEO altitude
bool is_geo_altitude(const Vector3d& position) {
    const double altitude_km = coords::ecef_to_geodetic(position).altitude / 1000.0;
    return altitude_km > 35000 && altitude_km < 36600;
}

// Validation result structure
struct GEOValidationResult {
    double position_rmse;
    double position_max_error;
    double radial_rms;
    double along_track_rms;
    double cross_track_rms;
    int num_comparisons;
};

// Compute validation metrics
GEOValidationResult compute_geo_validation(
    const std::vector<std::pair<Epoch, StateVector>>& propagated,
    const ballistic::test::TruthFile& sp3,
    const std::string& sat_id)
{
    GEOValidationResult result = {0.0, 0.0, 0.0, 0.0, 0.0, 0};

    std::vector<double> radial_errors, along_errors, cross_errors;

    for (const auto& [epoch, state] : propagated) {
        // Truth interpolated at the exact epoch, not taken from the nearest
        // record: a record up to 2 min away is 6 km off at GEO speed.
        auto truth = sp3.interpolate_position(sat_id, epoch);
        if (!truth) continue;

        Vector3d error = state.position - *truth;

        Vector3d r_hat = truth->normalized();
        Vector3d n_hat = truth->cross(state.velocity).normalized();
        Vector3d t_hat = n_hat.cross(r_hat);

        radial_errors.push_back(error.dot(r_hat));
        along_errors.push_back(error.dot(t_hat));
        cross_errors.push_back(error.dot(n_hat));

        double err_norm = error.norm();
        if (err_norm > result.position_max_error) {
            result.position_max_error = err_norm;
        }

        result.num_comparisons++;
    }

    if (result.num_comparisons == 0) return result;

    auto rms = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double sum_sq = 0.0;
        for (double x : v) sum_sq += x * x;
        return std::sqrt(sum_sq / v.size());
    };

    result.radial_rms = rms(radial_errors);
    result.along_track_rms = rms(along_errors);
    result.cross_track_rms = rms(cross_errors);
    result.position_rmse = std::sqrt(
        result.radial_rms * result.radial_rms +
        result.along_track_rms * result.along_track_rms +
        result.cross_track_rms * result.cross_track_rms
    );

    return result;
}

// Propagate GEO satellite
std::vector<std::pair<Epoch, StateVector>> propagate_geo(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    double sample_interval = 60.0,
    int gravity_degree = GEOConfig::GRAVITY_DEGREE)
{
    std::vector<std::pair<Epoch, StateVector>> results;

    BallisticModel model;  // No drag at GEO (controlled by force flags)

    WeatherConditions weather;
    weather.f107_daily = GEOConfig::F107_DAILY;
    weather.f107_average = GEOConfig::F107_AVERAGE;
    weather.ap_daily = GEOConfig::AP_DAILY;
    weather.ap_history.fill(GEOConfig::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos0, vel0, model, weather, true);

    prop->force_model().set_gravity_max_degree(gravity_degree);
    {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.drag = false;
        prop->force_model().set_force_flags(flags);
    }

    int num_samples = static_cast<int>(duration_seconds / sample_interval) + 1;
    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * sample_interval;
        if (dt > duration_seconds) dt = duration_seconds;

        Epoch t = t0 + dt;
        auto result = (*prop)(t);

        results.emplace_back(t, StateVector(result.position, result.velocity));
    }

    return results;
}

// =============================================================================
// GEO Propagation Tests
// =============================================================================

// The C01 file has one day at 5 min (288 records), all at GEO altitude, and
// stays within 5 deg of the equator (so it is GEO, not IGSO).
void test_sp3_geo_parser() {
    auto sp3 = load_sp3_file();
    if (!sp3) TEST_SKIP("no BeiDou SP3 file in " << get_beidou_data_path());

    auto records = sp3->get_satellite_records(kGeoSat);
    TEST_ASSERT(records.size() == 288);   // one day at 5 min
    double max_lat_deg = 0.0;
    for (const auto& r : records) {
        TEST_ASSERT(is_geo_altitude(r.position));
        max_lat_deg = std::max(max_lat_deg, std::abs(coords::ecef_to_geodetic(r.position).latitude) / constants::DEG_TO_RAD);
    }
    std::cout << "(sat=" << kGeoSat << ", max |lat| " << std::fixed << std::setprecision(2)
              << max_lat_deg << " deg) ";
    TEST_ASSERT(max_lat_deg < 5.0);       // geostationary, not IGSO
}

namespace {
struct GeoCase {
    ballistic::test::TruthFile sp3;
    std::vector<ballistic::test::TruthRecord> records;
};
GeoCase require_geo_case() {
    auto sp3 = load_sp3_file();
    if (!sp3) TEST_SKIP("no BeiDou SP3 file in " << get_beidou_data_path());
    auto records = sp3->get_satellite_records(kGeoSat);
    TEST_ASSERT(records.size() > 100);
    return {std::move(*sp3), std::move(records)};
}
}  // namespace

void test_geo_1_hour_propagation() {
    auto c = require_geo_case();
    const Epoch t0 = c.records[12].epoch;
    auto s0 = start_state(c.sp3, t0);
    TEST_ASSERT(s0.has_value());

    auto results = propagate_geo(t0, s0->position, s0->velocity, 3600.0, 60.0);
    auto validation = compute_geo_validation(results, c.sp3, kGeoSat);

    std::cout << "\n    Satellite: " << kGeoSat << "\n    Duration: 1 hour\n    Position RMSE: "
              << std::fixed << std::setprecision(2) << validation.position_rmse << " m\n    ";
    TEST_ASSERT(validation.num_comparisons >= 50);
    TEST_ASSERT(validation.position_rmse < 2.0);    // measured 0.32 m; bound ~6x
}

void test_geo_day_propagation() {
    // To the end of the daily file (~23 h; truth from the same solution).
    // 2 km bound: a loose sanity check (SRP is off, so the error grows over a
    // day).
    auto c = require_geo_case();
    const Epoch t0 = c.records[12].epoch;
    auto s0 = start_state(c.sp3, t0);
    TEST_ASSERT(s0.has_value());
    const double duration = c.records.back().epoch - t0;

    auto results = propagate_geo(t0, s0->position, s0->velocity, duration, 300.0);
    auto validation = compute_geo_validation(results, c.sp3, kGeoSat);

    std::cout << "\n    Duration: " << std::fixed << std::setprecision(1) << duration / 3600.0
              << " h\n    Position RMSE: " << std::setprecision(2) << validation.position_rmse
              << " m\n    Position Max:  " << validation.position_max_error << " m\n    ";
    TEST_ASSERT(validation.num_comparisons >= 250);
    TEST_ASSERT(validation.position_rmse < 2000.0);
}

void test_geo_gravity_degree_sweep() {
    auto c = require_geo_case();
    const Epoch t0 = c.records[12].epoch;
    auto s0 = start_state(c.sp3, t0);
    TEST_ASSERT(s0.has_value());

    std::cout << "\n    Gravity Degree vs RMSE (1 hour at GEO):\n    Degree    RMSE(m)\n";
    double rmse_4 = 0.0, rmse_30 = 0.0;
    for (int degree : {4, 8, 12, 15, 20, 30}) {
        auto results = propagate_geo(t0, s0->position, s0->velocity, 3600.0, 60.0, degree);
        const double rmse = compute_geo_validation(results, c.sp3, kGeoSat).position_rmse;
        if (degree == 4) rmse_4 = rmse;
        if (degree == 30) rmse_30 = rmse;
        std::cout << "    " << std::setw(4) << degree << "      " << std::fixed
                  << std::setprecision(3) << std::setw(8) << rmse << "\n";
    }
    std::cout << "    ";
    // Degree 4 already captures what matters at GEO: degrees 4 and 30 must
    // agree to 5 cm over 1 hour.
    TEST_ASSERT(std::abs(rmse_4 - rmse_30) < 0.05);
}

// GCRS + the loaded JPL ephemeris + relativity, degree 4.
std::vector<std::pair<Epoch, StateVector>> propagate_optimal(const Epoch& t0, const StateVector& s0,
                                                             double duration) {
    BallisticModel model;  // No drag at GEO (controlled by force flags)
    auto prop = std::make_unique<RK45Propagator>(
        t0, s0.position, s0.velocity, model, WeatherConditions::quiet(),
        true, RK45Config{}, g_ephemeris, PropagationFrame::GCRS, coords::ITRFRealization::ITRF2020);
    auto& fm = prop->force_model();
    fm.set_gravity_max_degree(4);
    forces::ForceComponentFlags flags = fm.force_flags();
    flags.relativistic = true;
    flags.drag = false;
    fm.set_force_flags(flags);
    std::vector<std::pair<Epoch, StateVector>> results;
    for (double dt = 0; dt <= duration; dt += 300.0) {
        auto r = (*prop)(t0 + dt);
        results.emplace_back(t0 + dt, StateVector(r.position, r.velocity));
    }
    return results;
}

void test_geo_optimal_config() {
    auto c = require_geo_case();
    const Epoch t0 = c.records[12].epoch;
    auto s0 = start_state(c.sp3, t0);
    TEST_ASSERT(s0.has_value());
    auto validation = compute_geo_validation(propagate_optimal(t0, *s0, 3600.0), c.sp3, kGeoSat);

    std::cout << "\n    Satellite: " << kGeoSat << " (BeiDou GEO), 1 hour, "
              << (g_ephemeris ? "JPL file ephemeris" : "embedded DE440")
              << "\n    Position RMSE: " << std::fixed << std::setprecision(3) << validation.position_rmse
              << " m (radial " << validation.radial_rms << ", along " << validation.along_track_rms
              << ", cross " << validation.cross_track_rms << ")\n    ";
    TEST_ASSERT(validation.position_rmse < 5.0);   // GNSS-1H: 0.8 m at +1 h, SRP off
}

// 6 hours with the same configuration; 200 m is a loose sanity bound.
void test_geo_optimal_config_6hour() {
    auto c = require_geo_case();
    const Epoch t0 = c.records[12].epoch;
    auto s0 = start_state(c.sp3, t0);
    TEST_ASSERT(s0.has_value());
    auto validation = compute_geo_validation(propagate_optimal(t0, *s0, 6.0 * 3600.0), c.sp3, kGeoSat);

    std::cout << "\n    Duration: 6 hours\n    Position RMSE: " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m\n    Position Max:  " << validation.position_max_error << " m\n    ";
    TEST_ASSERT(validation.position_rmse < 200.0);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "==============================================\n";
    std::cout << "GEO Orbit Propagation Validation Tests\n";
    std::cout << "==============================================\n\n";

    std::cout << "BeiDou data path: " << get_beidou_data_path() << "\n";

    // Load EOP data
    fs::path eop_path = fs::path(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all");
    if (time::EOPData::instance().load(eop_path)) {
        std::cout << "EOP data loaded successfully\n";
    }

    // Load JPL ephemeris
    fs::path de421_path = BALLISTIC_PROJECT_ROOT "/data/de421.bsp";
    fs::path de440_path = BALLISTIC_PROJECT_ROOT "/data/de440.bsp";
    if (fs::exists(de421_path)) {
        auto spk = std::make_shared<SPKEphemeris>();
        if (spk->load(de421_path)) {
            g_ephemeris = spk;
            std::cout << "JPL DE421 ephemeris loaded\n";
        }
    } else if (fs::exists(de440_path)) {
        auto spk = std::make_shared<SPKEphemeris>();
        if (spk->load(de440_path)) {
            g_ephemeris = spk;
            std::cout << "JPL DE440 ephemeris loaded\n";
        }
    } else {
        std::cout << "WARNING: JPL ephemeris not found\n";
    }

    std::cout << "\n";

    TestRunner runner("GEO Validation Tests (BeiDou C01)");

    runner.add_test("sp3_geo_parser", []() { test_sp3_geo_parser(); });
    runner.add_test("geo_1_hour_propagation", []() { test_geo_1_hour_propagation(); });
    runner.add_test("geo_day_propagation", []() { test_geo_day_propagation(); });
    runner.add_test("geo_gravity_degree_sweep", []() { test_geo_gravity_degree_sweep(); });
    runner.add_test("geo_optimal_config", []() { test_geo_optimal_config(); });
    runner.add_test("geo_optimal_config_6hour", []() { test_geo_optimal_config_6hour(); });

    runner.run();
    return print_final_summary();
}
