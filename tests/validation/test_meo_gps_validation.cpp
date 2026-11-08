// =============================================================================
// MEO (GPS Satellite) Orbit Propagation Validation Tests
// =============================================================================
//
// Checks orbit propagation at Medium Earth Orbit (MEO) altitude against
// precise orbits of GPS satellites (CODE final orbits, an IGS analysis center).
//
// MEO exercises different physics from LEO:
// - Third-body perturbations (Sun/Moon) are more significant
// - Solar radiation pressure becomes dominant non-gravitational force
// - Lower gravity degree is required (attenuation at higher altitudes)
// - No atmospheric drag
//
// GPS orbit characteristics:
// - Altitude: ~20,200 km
// - Period: ~11 hours 58 minutes (half sidereal day)
// - Inclination: 55 degrees
// - Semi-major axis: ~26,560 km
//
// Data: the first uncompressed SP3 file in tests/data/sp3/gps/ (SP3 = Standard
// Product 3, the IGS text format of satellite positions; the committed CODE
// final orbits give positions every 5 minutes, source and attribution in that
// directory's README.md).
//
// Metric: RMSE (root-mean-square error) of the propagated position against
// the SP3 positions, averaged over every sample of ONE arc from ONE start.
// This arc average is smaller than the error at the end of the arc, so it
// is not comparable to the GNSS-1H claim in docs/validation_claims.md (error
// at +1 h over many starts, produced by scripts/validate_gnss.py). The
// asserted bounds below are deliberately loose regression guards, not
// accuracy claims.
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
// MEO Configuration
// =============================================================================

// GPS Block IIF satellite parameters (typical)
struct GPSConfig {
    static constexpr double MASS = 1630.0;           // kg (GPS Block IIF)
    static constexpr double DRAG_AREA = 0.0;         // No drag at MEO
    static constexpr double SRP_AREA = 25.0;         // m^2 (solar panels + body)
    static constexpr double CD = 0.0;                // No drag
    static constexpr double CR = 1.5;                // SRP reflectivity coefficient

    // Orbit characteristics
    static constexpr double ALTITUDE = 20200000.0;   // m (~20,200 km)
    static constexpr double SEMI_MAJOR_AXIS = 26560000.0; // m
    static constexpr double PERIOD = 43082.0;        // s (~11h 58m)
    static constexpr double INCLINATION = 55.0;      // degrees

    // Recommended gravity degree for MEO (higher degrees attenuate rapidly)
    static constexpr int GRAVITY_DEGREE = 30;

    // Solar activity (typical quiet conditions)
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

// Get path to GPS SP3 test data directory
std::string get_gps_data_path() {
    std::vector<std::string> candidates = {
        BALLISTIC_PROJECT_ROOT "/tests/data/sp3/gps",
        "../tests/data/sp3/gps",
        "../../tests/data/sp3/gps",
        "tests/data/sp3/gps"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }

    return BALLISTIC_PROJECT_ROOT "/tests/data/sp3/gps";  // Default
}

// Find and load SP3 file from directory
std::optional<ballistic::test::TruthFile> load_sp3_file(const std::string& directory = "") {
    std::string data_path = directory.empty() ? get_gps_data_path() : directory;

    if (!fs::exists(data_path)) {
        return std::nullopt;
    }

    // Deterministic choice: the committed CODE final orbit first, then any
    // other uncompressed SP3 file by sorted name. directory_iterator order is
    // filesystem-dependent, and a local data directory can hold other
    // products (ultra-rapid, other analysis centers) a clean checkout lacks.
    const std::string preferred = "COD0OPSFIN_20240150000_01D_05M_ORB.SP3";
    std::vector<fs::path> candidates;
    for (const auto& entry : fs::directory_iterator(data_path)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext == ".sp3" || ext == ".SP3") candidates.push_back(entry.path());
    }
    std::sort(candidates.begin(), candidates.end(), [&](const fs::path& a, const fs::path& b) {
        const bool pa = a.filename() == preferred, pb = b.filename() == preferred;
        if (pa != pb) return pa;
        return a.filename().string() < b.filename().string();
    });
    for (const auto& path : candidates) {
        auto sp3 = ballistic::test::load_sp3_truth(path.string());
        if (sp3 && !sp3->records.empty()) {
            return sp3;
        }
    }

    return std::nullopt;
}

// Get first GPS satellite ID from SP3 file
std::string get_first_gps_satellite(const ballistic::test::TruthFile& sp3) {
    for (const auto& record : sp3.records) {
        // GPS satellites have IDs starting with 'G' (e.g., G01, G02, ...)
        if (!record.satellite_id.empty() && record.satellite_id[0] == 'G') {
            return record.satellite_id;
        }
    }
    return "";
}

// =============================================================================
// Lagrange Polynomial Interpolation for Velocity
// =============================================================================
// Uses N surrounding SP3 position records to compute velocity via polynomial
// derivative. This is much more accurate than simple finite difference:
// - 2-point finite diff: O(h) error (~2% for 5-min intervals)
// - 8-point Lagrange: O(h^7) error (~0.001% for 5-min intervals)

// Compute Lagrange polynomial derivative coefficient at evaluation point
// L'_i(t) = sum_{j!=i} [ 1/(t_i - t_j) * prod_{k!=i,j} (t - t_k)/(t_i - t_k) ]
double lagrange_derivative_coeff(double t, const std::vector<double>& times, int i) {
    double result = 0.0;
    int n = static_cast<int>(times.size());

    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        double term = 1.0 / (times[i] - times[j]);
        for (int k = 0; k < n; k++) {
            if (k == i || k == j) continue;
            term *= (t - times[k]) / (times[i] - times[k]);
        }
        result += term;
    }
    return result;
}

// Compute velocity at target epoch using Lagrange interpolation of SP3 positions
// Uses n_points surrounding the target (symmetric if possible)
Vector3d compute_lagrange_velocity(
    const std::vector<ballistic::test::TruthRecord>& records,
    int center_index,
    int n_points = 8)
{
    int n = static_cast<int>(records.size());
    if (n < 2) return Vector3d::Zero();

    // Select symmetric window around center
    int half = n_points / 2;
    int start = std::max(0, center_index - half);
    int end = std::min(n, start + n_points);
    start = std::max(0, end - n_points);

    // Collect times and positions
    std::vector<double> times;
    std::vector<Vector3d> positions;
    double t0 = records[center_index].epoch.utc_jd();  // Reference time

    for (int i = start; i < end; i++) {
        // Time in seconds relative to center
        double dt = (records[i].epoch.utc_jd() - t0) * 86400.0;
        times.push_back(dt);
        positions.push_back(records[i].position);
    }

    // Compute velocity as derivative of Lagrange polynomial at t=0
    Vector3d velocity = Vector3d::Zero();
    for (size_t i = 0; i < times.size(); i++) {
        double coeff = lagrange_derivative_coeff(0.0, times, static_cast<int>(i));
        velocity += coeff * positions[i];
    }

    return velocity;
}

// Propagation result structure for MEO
struct MEOValidationResult {
    double position_rmse;       // meters
    double position_max_error;
    double radial_rms;          // RTN components
    double along_track_rms;
    double cross_track_rms;
    double velocity_rmse;       // m/s
    int num_comparisons;
};

// Compute validation metrics for MEO propagation
MEOValidationResult compute_meo_validation(
    const std::vector<std::pair<Epoch, StateVector>>& propagated,
    const ballistic::test::TruthFile& sp3,
    const std::string& sat_id)
{
    MEOValidationResult result = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0};

    std::vector<double> radial_errors, along_errors, cross_errors, vel_errors;

    for (const auto& [epoch, state] : propagated) {
        // Find closest SP3 record
        auto sp3_opt = sp3.find_nearest(sat_id, epoch);
        if (!sp3_opt || std::abs(sp3_opt->epoch - epoch) > 120.0) continue;  // 2-minute tolerance

        // Both SP3 and propagator output are in ECEF
        Vector3d error = state.position - sp3_opt->position;

        // Compute RTN components in ECEF frame
        Vector3d r_hat = sp3_opt->position.normalized();
        Vector3d n_hat = sp3_opt->position.cross(state.velocity).normalized();
        Vector3d t_hat = n_hat.cross(r_hat);

        radial_errors.push_back(error.dot(r_hat));
        along_errors.push_back(error.dot(t_hat));
        cross_errors.push_back(error.dot(n_hat));

        // Track max error
        double err_norm = error.norm();
        if (err_norm > result.position_max_error) {
            result.position_max_error = err_norm;
        }

        // Velocity error if available
        if (sp3_opt->has_velocity) {
            vel_errors.push_back((state.velocity - sp3_opt->velocity).norm());
        }

        result.num_comparisons++;
    }

    if (result.num_comparisons == 0) return result;

    // Compute RMS for each component
    auto rms = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double sum_sq = 0.0;
        for (double x : v) sum_sq += x * x;
        return std::sqrt(sum_sq / v.size());
    };

    result.radial_rms = rms(radial_errors);
    result.along_track_rms = rms(along_errors);
    result.cross_track_rms = rms(cross_errors);
    result.velocity_rmse = rms(vel_errors);

    result.position_rmse = std::sqrt(
        result.radial_rms * result.radial_rms +
        result.along_track_rms * result.along_track_rms +
        result.cross_track_rms * result.cross_track_rms
    );

    return result;
}

// Propagate MEO satellite
std::vector<std::pair<Epoch, StateVector>> propagate_meo(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    double sample_interval = 60.0,
    bool enable_srp = false,
    int gravity_degree = GPSConfig::GRAVITY_DEGREE)
{
    std::vector<std::pair<Epoch, StateVector>> results;

    // MEO ballistic model (no drag - controlled by force flags)
    BallisticModel model;

    WeatherConditions weather;
    weather.f107_daily = GPSConfig::F107_DAILY;
    weather.f107_average = GPSConfig::F107_AVERAGE;
    weather.ap_daily = GPSConfig::AP_DAILY;
    weather.ap_history.fill(GPSConfig::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos0, vel0, model, weather, true);

    // Configure force model for MEO
    prop->force_model().set_gravity_max_degree(gravity_degree);
    {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.drag = false;  // No drag at MEO
        if (enable_srp) {
            flags.radiation = true;
        }
        prop->force_model().set_force_flags(flags);
    }

    // Sample at regular intervals
    int num_samples = static_cast<int>(duration_seconds / sample_interval) + 1;
    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * sample_interval;
        if (dt > duration_seconds) dt = duration_seconds;

        Epoch t = t0 + dt;
        auto result = (*prop)(t);

        StateVector state(result.position, result.velocity);
        results.emplace_back(t, state);
    }

    return results;
}

// =============================================================================
// SP3 Parser Tests
// =============================================================================

// Test: the SP3 reader loads the IGS GPS file and the first GPS satellite sits at
// GPS altitude (19,000-21,000 km; nominal 20,200 km).
void test_sp3_parser_basic() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found in " << get_gps_data_path());
    }

    TEST_ASSERT(!sp3->records.empty());

    // Find first GPS satellite
    std::string sat_id = get_first_gps_satellite(*sp3);
    TEST_ASSERT(!sat_id.empty());

    // Verify satellite is at MEO altitude
    auto records = sp3->get_satellite_records(sat_id);
    TEST_ASSERT(!records.empty());

    double altitude_km = coords::ecef_to_geodetic(records[0].position).altitude / 1000.0;

    std::cout << "(sat=" << sat_id << ", alt=" << std::fixed << std::setprecision(0)
              << altitude_km << "km, " << records.size() << " records) ";

    // GPS altitude should be ~20,000 km
    TEST_ASSERT(altitude_km > 19000 && altitude_km < 21000);
}

// =============================================================================
// MEO Propagation Tests
// =============================================================================

// Test: 1-hour propagation from one SP3 state, arc-averaged position RMSE < 5 km.
// The start velocity is the derivative of an 8-point Lagrange fit to the SP3
// positions (SP3 GPS files carry positions only).
void test_meo_1_hour_propagation() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite in file");
    }

    auto records = sp3->get_satellite_records(sat_id);
    if (records.size() < 10) {
        TEST_SKIP("insufficient records");
    }

    // Use a record in the middle to allow symmetric Lagrange interpolation
    // Need at least 4 records on each side for 8-point Lagrange
    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    // Compute velocity using Lagrange polynomial interpolation
    // This is much more accurate than simple finite-difference
    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
        std::cout << "(SP3 velocity) ";
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
        std::cout << "(Lagrange vel) ";
    }

    // Propagate 1 hour with 5-minute sampling to match SP3 interval
    double duration = 3600.0;
    double sp3_interval = 300.0;  // SP3 files typically have 5-minute intervals
    auto results = propagate_meo(t0, sv0.position, vel0, duration, sp3_interval);

    auto validation = compute_meo_validation(results, *sp3, sat_id);

    std::cout << "\n";
    std::cout << "    Satellite: " << sat_id << "\n";
    std::cout << "    Duration: 1 hour\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m\n";
    std::cout << "    Position Max:  " << validation.position_max_error << " m\n";
    std::cout << "    Radial RMS:    " << validation.radial_rms << " m\n";
    std::cout << "    Along-trk RMS: " << validation.along_track_rms << " m\n";
    std::cout << "    Cross-trk RMS: " << validation.cross_track_rms << " m\n";
    std::cout << "    ";

    // Grade printed for the log only; the assertion below is the gate.
    if (validation.position_rmse < 50.0) {
        std::cout << "EXCELLENT (<50m)... ";
    } else if (validation.position_rmse < 100.0) {
        std::cout << "VERY GOOD (<100m)... ";
    } else if (validation.position_rmse < 500.0) {
        std::cout << "GOOD (<500m)... ";
    } else {
        std::cout << "ACCEPTABLE... ";
    }

    // Loose 5 km guard: catches a broken start state or force model, not
    // small accuracy changes.
    TEST_ASSERT(validation.position_rmse < 5000.0);  // Conservative threshold
}

// Test: 6-hour propagation (half an orbit), arc-averaged RMSE < 20 km (loose guard).
void test_meo_6_hour_propagation() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite");
    }

    auto records = sp3->get_satellite_records(sat_id);

    // Need 6 hours of data (~24 records at 15-min intervals)
    if (records.size() < 30) {
        TEST_SKIP("insufficient records for 6h");
    }

    // Use record with enough neighbors for Lagrange interpolation
    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    // Compute velocity using Lagrange polynomial interpolation
    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
    }

    double duration = 6.0 * 3600.0;
    auto results = propagate_meo(t0, sv0.position, vel0, duration, 300.0);
    auto validation = compute_meo_validation(results, *sp3, sat_id);

    std::cout << "\n";
    std::cout << "    Duration: 6 hours (half orbital period)\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m\n";
    std::cout << "    Position Max:  " << validation.position_max_error << " m\n";
    std::cout << "    ";

    // Grade printed for the log only; the assertion below is the gate.
    if (validation.position_rmse < 200.0) {
        std::cout << "EXCELLENT (<200m)... ";
    } else if (validation.position_rmse < 500.0) {
        std::cout << "VERY GOOD (<500m)... ";
    } else if (validation.position_rmse < 2000.0) {
        std::cout << "GOOD (<2km)... ";
    } else {
        std::cout << "ACCEPTABLE... ";
    }

    TEST_ASSERT(validation.position_rmse < 20000.0);  // 20 km threshold
}

// Test: 12-hour propagation (one orbit), arc-averaged RMSE < 50 km (loose guard).
void test_meo_12_hour_full_period() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite");
    }

    auto records = sp3->get_satellite_records(sat_id);

    // Need 12 hours of data (~48 records at 15-min intervals)
    if (records.size() < 55) {
        TEST_SKIP("insufficient records for 12h");
    }

    // Use record with enough neighbors for Lagrange interpolation
    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    // Compute velocity using Lagrange polynomial interpolation
    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
    }

    double duration = 12.0 * 3600.0;
    auto results = propagate_meo(t0, sv0.position, vel0, duration, 300.0);
    auto validation = compute_meo_validation(results, *sp3, sat_id);

    std::cout << "\n";
    std::cout << "    Duration: 12 hours (full orbital period)\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m\n";
    std::cout << "    Position Max:  " << validation.position_max_error << " m\n";
    std::cout << "    ";

    // Grade printed for the log only; the assertion below is the gate.
    if (validation.position_rmse < 300.0) {
        std::cout << "EXCELLENT (<300m)... ";
    } else if (validation.position_rmse < 1000.0) {
        std::cout << "VERY GOOD (<1km)... ";
    } else if (validation.position_rmse < 5000.0) {
        std::cout << "GOOD (<5km)... ";
    } else {
        std::cout << "ACCEPTABLE... ";
    }

    TEST_ASSERT(validation.position_rmse < 50000.0);  // 50 km threshold
}

// =============================================================================
// Force Model Sensitivity Tests
// =============================================================================

// Report only (no assertion): 6-hour RMSE with solar radiation pressure (SRP) off
// and on, using the generic GPSConfig areas above (not sourced satellite values).
void test_meo_srp_sensitivity() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite");
    }

    auto records = sp3->get_satellite_records(sat_id);
    if (records.size() < 30) {
        TEST_SKIP("insufficient records");
    }

    // Use record with enough neighbors for Lagrange interpolation
    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    // Compute velocity using Lagrange polynomial interpolation
    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
    }

    double duration = 6.0 * 3600.0;

    std::cout << "\n";
    std::cout << "    SRP Sensitivity (6 hours):\n";
    std::cout << "    SRP       RMSE(m)    Radial     Along-Trk\n";
    std::cout << "    ------------------------------------------------\n";

    // Without SRP - use 300s sampling to match SP3 intervals
    auto results_no_srp = propagate_meo(t0, sv0.position, vel0, duration, 300.0, false);
    auto val_no_srp = compute_meo_validation(results_no_srp, *sp3, sat_id);

    std::cout << "    OFF      " << std::fixed << std::setprecision(2) << std::setw(8)
              << val_no_srp.position_rmse
              << "   " << std::setw(8) << val_no_srp.radial_rms
              << "   " << std::setw(8) << val_no_srp.along_track_rms << "\n";

    // With SRP - use 300s sampling to match SP3 intervals
    auto results_srp = propagate_meo(t0, sv0.position, vel0, duration, 300.0, true);
    auto val_srp = compute_meo_validation(results_srp, *sp3, sat_id);

    std::cout << "    ON       " << std::setw(8) << val_srp.position_rmse
              << "   " << std::setw(8) << val_srp.radial_rms
              << "   " << std::setw(8) << val_srp.along_track_rms << "\n";

    std::cout << "    ------------------------------------------------\n";
    std::cout << "    ";
}

// Report only (no assertion): 6-hour RMSE with Sun, Moon and Jupiter gravity
// switched on one at a time.
void test_meo_third_body_contribution() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite");
    }

    auto records = sp3->get_satellite_records(sat_id);
    if (records.size() < 30) {
        TEST_SKIP("insufficient records");
    }

    // Use record with enough neighbors for Lagrange interpolation
    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    // Compute velocity using Lagrange polynomial interpolation
    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
    }

    double duration = 6.0 * 3600.0;

    std::cout << "\n";
    std::cout << "    Third-Body Contribution (6 hours at MEO):\n";
    std::cout << "    Configuration          RMSE(m)\n";
    std::cout << "    ----------------------------------\n";

    // Configurations to test
    struct Config {
        std::string name;
        bool sun;
        bool moon;
        bool jupiter;
    };

    std::vector<Config> configs = {
        {"Earth only",        false, false, false},
        {"+ Sun",             true,  false, false},
        {"+ Moon",            false, true,  false},
        {"+ Sun + Moon",      true,  true,  false},
        {"+ All (incl. Jup)", true,  true,  true}
    };

    for (const auto& cfg : configs) {
        BallisticModel model;  // No drag at MEO (controlled by force flags)

        WeatherConditions weather;
        weather.f107_daily = GPSConfig::F107_DAILY;
        weather.f107_average = GPSConfig::F107_AVERAGE;
        weather.ap_daily = GPSConfig::AP_DAILY;
        weather.ap_history.fill(GPSConfig::AP_DAILY);

        auto prop = create_rk45_propagator(t0, sv0.position, vel0, model, weather, true);

        prop->force_model().set_gravity_max_degree(GPSConfig::GRAVITY_DEGREE);
        {
            forces::ForceComponentFlags flags = prop->force_model().force_flags();
            flags.drag = false;
            prop->force_model().set_force_flags(flags);
        }
        prop->force_model().third_body().enable_sun(cfg.sun);
        prop->force_model().third_body().enable_moon(cfg.moon);
        prop->force_model().third_body().enable_jupiter(cfg.jupiter);

        std::vector<std::pair<Epoch, StateVector>> results;
        for (double dt = 0; dt <= duration; dt += 300.0) {  // Match SP3 interval
            Epoch t = t0 + dt;
            auto result = (*prop)(t);
            results.emplace_back(t, StateVector(result.position, result.velocity));
        }

        auto validation = compute_meo_validation(results, *sp3, sat_id);

        std::cout << "    " << std::left << std::setw(20) << cfg.name
                  << std::right << std::fixed << std::setprecision(2) << std::setw(10)
                  << validation.position_rmse << "\n";
    }

    std::cout << "    ----------------------------------\n";
    std::cout << "    ";
}

// Report only (no assertion): 1-hour RMSE against Earth gravity field degree 4-70.
void test_meo_gravity_degree_sweep() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite");
    }

    auto records = sp3->get_satellite_records(sat_id);
    if (records.size() < 10) {
        TEST_SKIP("insufficient records");
    }

    // Use record with enough neighbors for Lagrange interpolation
    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    // Compute velocity using Lagrange polynomial interpolation
    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
    }

    double duration = 3600.0;  // 1 hour

    std::cout << "\n";
    std::cout << "    Gravity Degree vs RMSE (1 hour at MEO):\n";
    std::cout << "    Degree    RMSE(m)\n";
    std::cout << "    -----------------------\n";

    std::vector<int> degrees = {4, 8, 15, 20, 30, 40, 50, 70};
    double best_rmse = 1e9;
    int best_degree = 4;

    for (int degree : degrees) {
        auto results = propagate_meo(t0, sv0.position, vel0, duration, 300.0, false, degree);  // Match SP3 interval
        auto validation = compute_meo_validation(results, *sp3, sat_id);

        std::cout << "    " << std::setw(4) << degree << "      "
                  << std::fixed << std::setprecision(2) << std::setw(8)
                  << validation.position_rmse << "\n";

        if (validation.position_rmse < best_rmse) {
            best_rmse = validation.position_rmse;
            best_degree = degree;
        }
    }

    std::cout << "    -----------------------\n";
    std::cout << "    Best: degree " << best_degree << " with RMSE = "
              << std::fixed << std::setprecision(2) << best_rmse << " m\n";
    std::cout << "    Note: Higher degrees have diminishing returns at MEO altitude\n";
    std::cout << "    ";
}

// =============================================================================
// Optimal Configuration Tests (GCRS + JPL Ephemeris)
// =============================================================================

// Test: 1 hour in the GCRS inertial frame (IAU 2006/2000A) with the JPL
// planetary ephemeris (if data/de421.bsp or data/de440.bsp exists; otherwise the
// analytical one), relativity on, gravity degree 4 (the QZSS test uses degree 8).
// Gate: arc-averaged RMSE < 500 m (loose guard).
void test_meo_optimal_config() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite");
    }

    auto records = sp3->get_satellite_records(sat_id);
    if (records.size() < 10) {
        TEST_SKIP("insufficient records");
    }

    // Use a record in the middle for symmetric Lagrange interpolation
    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    // Compute velocity using Lagrange polynomial interpolation
    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
    }

    // Create propagator with optimal configuration (same as QZSS HEO test)
    BallisticModel model;  // No drag at MEO (controlled by force flags)
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = std::make_unique<RK45Propagator>(
        t0, sv0.position, vel0,
        model, weather,
        true,  // enable_third_body (critical)
        RK45Config{},
        g_ephemeris,  // JPL ephemeris
        PropagationFrame::GCRS,  // GCRS frame (IAU 2006/2000A)
        coords::ITRFRealization::ITRF2020
    );

    auto& fm = prop->force_model();
    fm.set_gravity_max_degree(4);  // Low degree sufficient at MEO altitude
    {
        forces::ForceComponentFlags flags = fm.force_flags();
        flags.relativistic = true;
        flags.drag = false;
        fm.set_force_flags(flags);
    }

    // Propagate 1 hour with 5-minute sampling
    double duration = 3600.0;
    std::vector<std::pair<Epoch, StateVector>> results;
    for (double dt = 0; dt <= duration; dt += 300.0) {
        auto result = (*prop)(t0 + dt);
        results.emplace_back(t0 + dt, StateVector(result.position, result.velocity));
    }

    auto validation = compute_meo_validation(results, *sp3, sat_id);

    std::cout << "\n";
    std::cout << "    Satellite: " << sat_id << "\n";
    std::cout << "    Configuration: GCRS + JPL DE421 + Relativistic\n";
    std::cout << "    Gravity degree: 4 (sufficient for MEO)\n";
    std::cout << "    Duration: 1 hour\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m\n";
    std::cout << "    Position Max:  " << validation.position_max_error << " m\n";
    std::cout << "    Radial RMS:    " << validation.radial_rms << " m\n";
    std::cout << "    Along-trk RMS: " << validation.along_track_rms << " m\n";
    std::cout << "    Cross-trk RMS: " << validation.cross_track_rms << " m\n";
    if (g_ephemeris) {
        std::cout << "    (using JPL ephemeris)\n";
    } else {
        std::cout << "    (using analytical ephemeris - expect higher RMSE)\n";
    }
    std::cout << "    ";

    TEST_ASSERT(validation.position_rmse < 500.0);  // Conservative threshold
}

// Test: the same setup over 6 hours, RMSE < 5 km (loose guard).
void test_meo_optimal_config_6hour() {
    auto sp3 = load_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file");
    }

    std::string sat_id = get_first_gps_satellite(*sp3);
    if (sat_id.empty()) {
        TEST_SKIP("no GPS satellite");
    }

    auto records = sp3->get_satellite_records(sat_id);
    if (records.size() < 80) {  // Need enough records for 6 hours
        TEST_SKIP("insufficient records for 6h");
    }

    int start_idx = std::min(4, static_cast<int>(records.size()) - 1);
    const auto& sv0 = records[start_idx];
    Epoch t0 = sv0.epoch;

    Vector3d vel0;
    if (sv0.has_velocity) {
        vel0 = sv0.velocity;
    } else {
        vel0 = compute_lagrange_velocity(records, start_idx, 8);
    }

    BallisticModel model;  // No drag at MEO (controlled by force flags)
    WeatherConditions weather = WeatherConditions::quiet();

    auto prop = std::make_unique<RK45Propagator>(
        t0, sv0.position, vel0,
        model, weather, true, RK45Config{}, g_ephemeris,
        PropagationFrame::GCRS, coords::ITRFRealization::ITRF2020
    );

    auto& fm = prop->force_model();
    fm.set_gravity_max_degree(4);
    {
        forces::ForceComponentFlags flags = fm.force_flags();
        flags.relativistic = true;
        flags.drag = false;
        fm.set_force_flags(flags);
    }

    double duration = 6.0 * 3600.0;
    std::vector<std::pair<Epoch, StateVector>> results;
    for (double dt = 0; dt <= duration; dt += 300.0) {
        auto result = (*prop)(t0 + dt);
        results.emplace_back(t0 + dt, StateVector(result.position, result.velocity));
    }

    auto validation = compute_meo_validation(results, *sp3, sat_id);

    std::cout << "\n";
    std::cout << "    Duration: 6 hours (half orbital period)\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << validation.position_rmse << " m\n";
    std::cout << "    Position Max:  " << validation.position_max_error << " m\n";
    std::cout << "    ";

    TEST_ASSERT(validation.position_rmse < 5000.0);  // Conservative threshold
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "==============================================\n";
    std::cout << "MEO (GPS Satellite) Orbit Validation Tests\n";
    std::cout << "==============================================\n\n";

    std::cout << "GPS data path: " << get_gps_data_path() << "\n";

    // Check if data exists
    if (!fs::exists(get_gps_data_path())) {
        std::cout << "\nWARNING: GPS SP3 data directory not found.\n";
        std::cout << "To run these tests, download IGS products:\n";
        std::cout << "  1. Register at https://urs.earthdata.nasa.gov/\n";
        std::cout << "  2. Download SP3 files from https://cddis.nasa.gov/archive/gnss/products/\n";
        std::cout << "  3. Place files in: " << get_gps_data_path() << "\n\n";
    }

    // Load EOP data
    fs::path eop_path = fs::path(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all");
    if (time::EOPData::instance().load(eop_path)) {
        std::cout << "EOP data loaded successfully\n";
    } else {
        std::cout << "WARNING: Could not load EOP data\n";
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
        std::cout << "WARNING: JPL ephemeris not found (optimal config tests will use analytical)\n";
    }
    std::cout << "\n";

    TestRunner runner("MEO GPS Validation Tests");

    runner.add_test("sp3_parser_basic", []() { test_sp3_parser_basic(); });
    runner.add_test("meo_1_hour_propagation", []() { test_meo_1_hour_propagation(); });
    runner.add_test("meo_6_hour_propagation", []() { test_meo_6_hour_propagation(); });
    runner.add_test("meo_12_hour_full_period", []() { test_meo_12_hour_full_period(); });
    runner.add_test("meo_srp_sensitivity", []() { test_meo_srp_sensitivity(); });
    runner.add_test("meo_third_body_contribution", []() { test_meo_third_body_contribution(); });
    runner.add_test("meo_gravity_degree_sweep", []() { test_meo_gravity_degree_sweep(); });
    runner.add_test("meo_optimal_config", []() { test_meo_optimal_config(); });
    runner.add_test("meo_optimal_config_6hour", []() { test_meo_optimal_config_6hour(); });

    runner.run();
    return print_final_summary();
}
