// =============================================================================
// CHAMP Drag Validation Tests
// =============================================================================
//
// Checks the drag model against CHAMP (CHAllenging Minisatellite Payload)
// precise orbits from GFZ. CHAMP flew at ~450 km at launch and below 300 km in
// its last weeks, where drag is large
// enough to measure in a one-hour prediction.
//
// What it checks: the SP3 orbit file reads correctly; beta and the drag
// acceleration are physically plausible; 1-minute to 1-hour predictions
// started from an SP3 state stay within loose bounds of the SP3 truth; drag on
// differs from drag off; tabulated drag tables built at different sample
// spacings (and a coarse 4D field) give the same 1-hour error within 5% (10%
// for the 30 s table).
// The bounds here catch a broken drag path; the campaign-level accuracy
// figure is [`CHAMP-370KM`] in docs/validation_claims.md (Python script).
//
// Terms: SP3 = the standard precise-orbit file format (positions, optional
// velocities); RMSE = root-mean-square error; beta (β) = m / (Cd·A), the
// ballistic coefficient in kg/m²; GCRS = the inertial frame the propagator
// integrates in; ITRF = the Earth-fixed frame of the SP3 file (ITRF2005 for
// 2005 data); RTN = radial / along-track / cross-track error split.
//
// Data: tests/data/sp3/champ/ (GFZ RSO orbits, 2005).
//
// CHAMP characteristics:
// - Altitude: 450 km (2000) -> 310 km (2010) - decayed over mission
// - Inclination: 87.3 degrees (near-polar)
// - Mass, frontal area, Cd: the registry (include/ballistic/satellites.hpp)
// - NO drag-free control (unlike GOCE)
//
// Data source: GFZ ISDC - ftp://isdcftp.gfz.de/champ/ORBIT/CHAMP/RSO/
// SP3 satellite ID: L06 (GFZ RSO files)
// Orbit accuracy: 1-2 cm (SLR validated)
//
// Key difference from Sentinel-1:
// - Sentinel-1: 693 km altitude, drag small at 1 hour and varying with solar
//   activity (but not negligible: see [`SENTINEL-YEAR-2025`])
// - CHAMP: 350-380 km altitude, drag the largest error source at 1 hour
//   (no-drag 6.3 m vs drag-on 1.2 m in [`CHAMP-370KM`])
// =============================================================================

#include <ballistic/time/eop_data.hpp>
#include <ballistic/ballistic.hpp>
#include "../sp3_truth.hpp"
#include <ballistic/validation/eof_parser.hpp>
#include <ballistic/forces/drag.hpp>
#include <ballistic/atmosphere/internal/nrlmsis21_full.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
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

namespace fs = std::filesystem;

// Sun/Moon ephemeris used by every propagation (nullptr = the embedded DE440)
static std::shared_ptr<ephemeris::EphemerisProvider> g_ephemeris = nullptr;

// Load a JPL planetary ephemeris file (DE421 first, then DE440) if one is in data/;
// otherwise return nullptr and the library's embedded DE440 is used.
std::shared_ptr<ephemeris::EphemerisProvider> load_jpl_ephemeris() {
    std::vector<std::string> paths = {
        BALLISTIC_PROJECT_ROOT "/data/de421.bsp",
        "data/de421.bsp",
        "../data/de421.bsp",
        BALLISTIC_PROJECT_ROOT "/data/de440.bsp",
        "data/de440.bsp",
        "../data/de440.bsp"
    };

    for (const auto& path : paths) {
        if (fs::exists(path)) {
            try {
                auto spk = std::make_shared<ephemeris::SPKEphemeris>();
                if (spk->load(path)) {
                    std::cout << "  Loaded JPL ephemeris: " << path << std::endl;
                    return spk;
                }
            } catch (const std::exception& e) {
                std::cout << "  WARNING: Failed to load ephemeris from " << path
                          << ": " << e.what() << std::endl;
            } catch (...) {
                std::cout << "  WARNING: Unknown error loading ephemeris from " << path << std::endl;
            }
        }
    }
    std::cout << "  WARNING: No JPL ephemeris found, using analytical" << std::endl;
    return nullptr;
}

// =============================================================================
// Test Configuration
// =============================================================================

// CHAMP spacecraft parameters (mid-mission, ~2005)
struct CHAMPConfig {
    // Registry values (include/ballistic/satellites.hpp). Cd*A is fitted to CHAMP
    // orbits; only the product is meaningful, not the split into Cd and A.
    static constexpr double MASS = ballistic::satellites::get("CHAMP").mass_kg;
    static constexpr double DRAG_AREA = ballistic::satellites::get("CHAMP").drag_area_m2;  // frontal
    static constexpr double CD = ballistic::satellites::get("CHAMP").cd;
    static constexpr double ALTITUDE = 380.0;       // km (mid-mission nominal)
    static constexpr double INCLINATION = 87.3;     // degrees
    static constexpr const char* SP3_ID = "L06";    // SP3 satellite identifier (GFZ RSO format)

    // Solar activity for July 1, 2005
    // Source: CelesTrak SW-All.csv (https://celestrak.org/SpaceData/)
    // Line: 2005-07-01,2346,17,...,114.6,118.5,OBS,94.6,94.4,97.6,96.6
    // F10.7 = 10.7 cm solar radio flux (sfu); Ap = daily geomagnetic index.
    static constexpr double F107_DAILY = 114.6;     // Observed F10.7
    static constexpr double F107_AVERAGE = 94.6;    // 81-day centered average
    static constexpr double AP_DAILY = 14.0;        // Daily Ap average
};

// Illustrative early- and late-mission parameter sets (round values, not the
// registry). Not used by any test cell below.
struct CHAMPEarlyConfig {  // 2001-2002, solar maximum
    static constexpr double MASS = 520.0;
    static constexpr double DRAG_AREA = 3.5;
    static constexpr double CD = 2.3;
    static constexpr double ALTITUDE = 440.0;
    static constexpr double F107_DAILY = 180.0;     // Active sun
    static constexpr double F107_AVERAGE = 175.0;
    static constexpr double AP_DAILY = 15.0;
};

struct CHAMPLateConfig {  // 2009-2010, low altitude
    static constexpr double MASS = 500.0;
    static constexpr double DRAG_AREA = 3.5;
    static constexpr double CD = 2.3;
    static constexpr double ALTITUDE = 320.0;
    static constexpr double F107_DAILY = 70.0;      // Very quiet sun
    static constexpr double F107_AVERAGE = 72.0;
    static constexpr double AP_DAILY = 5.0;
};

// =============================================================================
// Test Framework
// =============================================================================

#include "../test_utils.hpp"
using namespace ballistic::test;

// =============================================================================
// Utility Functions
// =============================================================================

// Get path to CHAMP SP3 test data directory
std::string get_champ_data_path() {
    std::vector<std::string> candidates = {
        BALLISTIC_PROJECT_ROOT "/tests/data/sp3/champ",
        "../tests/data/sp3/champ",
        "../../tests/data/sp3/champ",
        "tests/data/sp3/champ"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }

    return BALLISTIC_PROJECT_ROOT "/tests/data/sp3/champ";  // Default
}

// Derive velocity from neighboring position records by central differences
// (one-sided at the ends). Not called by the cells below.
void derive_velocities_from_positions(std::vector<ballistic::test::TruthRecord>& records) {
    if (records.size() < 2) return;

    for (size_t i = 0; i < records.size(); ++i) {
        if (records[i].has_velocity) continue;  // Already has velocity

        // Use central difference where possible, forward/backward at edges
        size_t prev_idx, next_idx;
        if (i == 0) {
            prev_idx = 0;
            next_idx = 1;
        } else if (i == records.size() - 1) {
            prev_idx = i - 1;
            next_idx = i;
        } else {
            prev_idx = i - 1;
            next_idx = i + 1;
        }

        double dt = records[next_idx].epoch - records[prev_idx].epoch;
        if (dt > 0.1) {  // At least 0.1 second interval
            Vector3d dpos = records[next_idx].position - records[prev_idx].position;
            records[i].velocity = dpos / dt;
            records[i].has_velocity = true;
        }
    }
}

// Find and parse a CHAMP SP3 file
std::optional<ballistic::test::TruthFile> load_champ_sp3_file(const std::string& pattern = "", bool debug = false) {
    std::string data_path = get_champ_data_path();

    if (!fs::exists(data_path)) {
        if (debug) std::cout << "  [DEBUG] Path does not exist: " << data_path << std::endl;
        return std::nullopt;
    }
    if (!fs::is_directory(data_path)) {
        if (debug) std::cout << "  [DEBUG] Path is not a directory: " << data_path << std::endl;
        return std::nullopt;
    }

    // Deterministic choice: the committed CHAMP_2005_182.sp3 first, then the
    // other files by sorted name. directory_iterator order is
    // filesystem-dependent, and a local data directory can hold other orbit
    // files a clean checkout lacks.
    const std::string preferred = "CHAMP_2005_182.sp3";
    std::vector<fs::directory_entry> entries(fs::directory_iterator(data_path), fs::directory_iterator{});
    std::sort(entries.begin(), entries.end(), [&](const fs::directory_entry& a, const fs::directory_entry& b) {
        const bool pa = a.path().filename() == preferred, pb = b.path().filename() == preferred;
        if (pa != pb) return pa;
        return a.path().filename().string() < b.path().filename().string();
    });

    int file_count = 0;
    int sp3_count = 0;
    for (const auto& entry : entries) {
        std::string filename = entry.path().filename().string();
        file_count++;

        // Check for .sp3 extension (case-insensitive)
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".sp3") {
            sp3_count++;
            if (debug) std::cout << "  [DEBUG] Found SP3: " << filename << std::endl;
            if (pattern.empty() || filename.find(pattern) != std::string::npos) {
                auto result = ballistic::test::load_sp3_truth(entry.path().string());
                if (result) {
                    // Verify it contains CHAMP data
                    auto champ_records = result->get_satellite_records(CHAMPConfig::SP3_ID);
                    if (!champ_records.empty()) {
                        // SP3 files carry position (P) records and optional velocity (V)
                        // records. If this file has no velocities, derive them below by
                        // central differences of the positions.
                        bool has_vel = champ_records[0].has_velocity;
                        if (!has_vel) {
                            // Collect this satellite's records (all position-only here).
                            std::vector<size_t> p_indices;
                            for (size_t i = 0; i < result->records.size(); ++i) {
                                if (result->records[i].satellite_id == CHAMPConfig::SP3_ID &&
                                    !result->records[i].has_velocity) {
                                    p_indices.push_back(i);
                                }
                            }
                            // Derive velocities using central difference on P records
                            for (size_t j = 0; j < p_indices.size(); ++j) {
                                size_t i = p_indices[j];
                                size_t prev_j = (j == 0) ? 0 : j - 1;
                                size_t next_j = (j == p_indices.size() - 1) ? j : j + 1;
                                size_t prev_i = p_indices[prev_j];
                                size_t next_i = p_indices[next_j];

                                double dt = result->records[next_i].epoch - result->records[prev_i].epoch;
                                if (dt > 1.0) {  // At least 1 second
                                    Vector3d dpos = result->records[next_i].position - result->records[prev_i].position;
                                    result->records[i].velocity = dpos / dt;
                                    result->records[i].has_velocity = true;
                                }
                            }
                        }
                        if (debug) std::cout << "  [DEBUG] Parsed OK, " << champ_records.size()
                                           << " CHAMP records" << (has_vel ? "" : " (velocities derived)") << std::endl;
                        return result;
                    } else if (debug) {
                        std::cout << "  [DEBUG] No " << CHAMPConfig::SP3_ID << " records in: " << filename << std::endl;
                    }
                } else if (debug) {
                    std::cout << "  [DEBUG] Parse failed for: " << filename << std::endl;
                }
            }
        }
    }

    if (debug) {
        std::cout << "  [DEBUG] Files scanned: " << file_count << ", SP3 files: " << sp3_count << std::endl;
    }

    return std::nullopt;
}

// Create weather conditions from config
template<typename Config>
WeatherConditions make_weather() {
    WeatherConditions weather;
    weather.f107_daily = Config::F107_DAILY;
    weather.f107_average = Config::F107_AVERAGE;
    weather.ap_daily = Config::AP_DAILY;
    weather.ap_history.fill(Config::AP_DAILY);
    return weather;
}

// Create a BallisticModel from the config. Its 3-argument constructor takes
// (beta, mass, area). The propagator does not read this model for drag (drag
// comes only from a DragEnvironment); the tests use it to get beta.
template<typename Config>
BallisticModel make_ballistic_model() {
    constexpr double beta = Config::MASS / (Config::CD * Config::DRAG_AREA);
    return BallisticModel(beta, Config::MASS, Config::DRAG_AREA);
}

// Propagate from an ITRF2005 start state (integrating in GCRS) and return the
// state every `sample_interval` seconds.
template<typename Config>
std::vector<std::pair<Epoch, StateVector>> propagate_with_sampling(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    double sample_interval = 10.0,
    bool enable_drag = true)
{
    std::vector<std::pair<Epoch, StateVector>> results;

    BallisticModel model = make_ballistic_model<Config>();
    WeatherConditions weather = make_weather<Config>();

    auto prop = create_gcrs_propagator(t0, pos0, vel0, model, weather, true,
                                       g_ephemeris, coords::ITRFRealization::ITRF2005);

    // Drag comes only from a DragEnvironment. Drag on: tabulate NRLMSIS 2.1
    // density along a drag-free nominal trajectory sampled every 60 s
    // (from_models_along) and attach it. Drag off: attach nothing.
    if (enable_drag) {
        // Drag-free nominal to feed from_models_along.
        std::vector<Epoch> sample_epochs;
        std::vector<Vector3d> sample_positions_ecef;
        const double dt_sample = 60.0;
        const int n_sample = static_cast<int>(duration_seconds / dt_sample) + 1;
        sample_epochs.reserve(n_sample);
        sample_positions_ecef.reserve(n_sample);

        BallisticModel nodrag_model = model;
        auto nodrag_prop = create_gcrs_propagator(
            t0, pos0, vel0, nodrag_model, weather, true,
            g_ephemeris, coords::ITRFRealization::ITRF2005);
        // (No DragEnvironment is attached, so this nominal run is drag-free.)
        for (int i = 0; i <= n_sample; ++i) {
            double dt = i * dt_sample;
            if (dt > duration_seconds) dt = duration_seconds;
            Epoch t = t0 + dt;
            auto r = (*nodrag_prop)(t);
            sample_epochs.push_back(t);
            sample_positions_ecef.push_back(r.position);
        }

        atmosphere::SpaceWeatherHistory sw =
            atmosphere::SpaceWeatherHistory::from_constant(
                weather, atmosphere::TimeWindow{t0 - 86400.0,
                                                  t0 + duration_seconds + 86400.0});
        const double beta_kg_m2 = Config::MASS / (Config::CD * Config::DRAG_AREA);
        atmosphere::BetaProfile1D beta_prior =
            atmosphere::BetaProfile1D::constant(beta_kg_m2);
        auto env = std::make_shared<const atmosphere::DragEnvironment>(
            atmosphere::from_models_along(
                sample_epochs, sample_positions_ecef, sw, std::move(beta_prior)));
        prop->force_model().set_drag_environment(env);
    } else {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.drag = false;
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

// Compute propagation result comparing predicted positions against SP3 truth
PropagationTestResult compute_propagation_result_sp3(
    const std::vector<std::pair<Epoch, StateVector>>& predicted,
    const ballistic::test::TruthFile& truth,
    const std::string& satellite_id,
    double max_time_diff = 5.0)  // 5-second tolerance
{
    PropagationTestResult result{};

    auto truth_records = truth.get_satellite_records(satellite_id);
    if (truth_records.empty() || predicted.empty()) {
        return result;
    }

    result.start_epoch = predicted.front().first;
    result.end_epoch = predicted.back().first;
    result.duration_seconds = result.end_epoch - result.start_epoch;

    double sum_pos_sq = 0.0;
    double sum_vel_sq = 0.0;
    double sum_along_sq = 0.0;
    double sum_cross_sq = 0.0;
    double sum_radial_sq = 0.0;
    double max_pos_err = 0.0;
    double max_vel_err = 0.0;
    int count = 0;

    for (const auto& [pred_epoch, pred_state] : predicted) {
        // Find nearest truth record
        double min_diff = std::numeric_limits<double>::max();
        const ballistic::test::TruthRecord* nearest = nullptr;

        for (const auto& rec : truth_records) {
            double diff = std::abs(rec.epoch - pred_epoch);
            if (diff < min_diff) {
                min_diff = diff;
                nearest = &rec;
            }
        }

        if (nearest && min_diff <= max_time_diff) {
            // Position error
            Vector3d pos_error = pred_state.position - nearest->position;
            double pos_err_mag = pos_error.norm();
            sum_pos_sq += pos_err_mag * pos_err_mag;
            if (pos_err_mag > max_pos_err) max_pos_err = pos_err_mag;

            // Velocity error (if available)
            if (nearest->has_velocity) {
                Vector3d vel_error = pred_state.velocity - nearest->velocity;
                double vel_err_mag = vel_error.norm();
                sum_vel_sq += vel_err_mag * vel_err_mag;
                if (vel_err_mag > max_vel_err) max_vel_err = vel_err_mag;
            }

            // RTN decomposition
            Vector3d r_hat = nearest->position.normalized();
            Vector3d h = nearest->position.cross(nearest->velocity);
            Vector3d n_hat = h.normalized();
            Vector3d t_hat = n_hat.cross(r_hat);

            double radial = pos_error.dot(r_hat);
            double along = pos_error.dot(t_hat);
            double cross = pos_error.dot(n_hat);

            sum_radial_sq += radial * radial;
            sum_along_sq += along * along;
            sum_cross_sq += cross * cross;

            count++;
        }
    }

    if (count > 0) {
        result.num_comparisons = count;
        result.position_rmse = std::sqrt(sum_pos_sq / count);
        result.velocity_rmse = std::sqrt(sum_vel_sq / count);
        result.position_max_error = max_pos_err;
        result.velocity_max_error = max_vel_err;
        result.radial_rms = std::sqrt(sum_radial_sq / count);
        result.along_track_rms = std::sqrt(sum_along_sq / count);
        result.cross_track_rms = std::sqrt(sum_cross_sq / count);
    }

    return result;
}

// =============================================================================
// SP3 Parser Tests
// =============================================================================

// Test: SP3 parser can read CHAMP files
void test_sp3_parser_champ_file() {
    auto sp3 = load_champ_sp3_file("", true);
    if (!sp3) {
        TEST_SKIP("no SP3 file found in " << get_champ_data_path());
    }

    TEST_ASSERT(!sp3->records.empty());
}

// Test: SP3 parser returns only records of the requested satellite (CHAMP)
void test_sp3_satellite_id_filter() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto champ_records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    TEST_ASSERT(!champ_records.empty());

    // Verify records are for CHAMP
    for (const auto& rec : champ_records) {
        TEST_ASSERT(rec.satellite_id == CHAMPConfig::SP3_ID);
    }
}

// Test: SP3 records have reasonable values for CHAMP orbit
void test_sp3_record_values() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    TEST_ASSERT(!records.empty());

    const auto& rec = records[0];
    double r = rec.position.norm();

    // CHAMP at ~350 km altitude: radius ~6728 km
    TEST_ASSERT(r > 6.5e6 && r < 7.0e6);  // Between 6500 and 7000 km radius

    if (rec.has_velocity) {
        double v = rec.velocity.norm();
        // LEO velocity ~7.7 km/s
        TEST_ASSERT(v > 7500 && v < 8000);
    }
}

// =============================================================================
// CHAMP Configuration Tests
// =============================================================================

// Test: CHAMP ballistic coefficient is reasonable
void test_champ_ballistic_coefficient() {
    BallisticModel model = make_ballistic_model<CHAMPConfig>();

    double beta = model.ballistic_coefficient();
    // beta = m / (Cd * A) from the registry (197.9 kg/m^2, fitted Cd*A);
    // the 150-250 band would catch a total-area-for-frontal-area mix-up.

    TEST_ASSERT(beta > 150.0 && beta < 250.0);

    double inv_beta = model.inverse_ballistic_coefficient();
    TEST_ASSERT(std::abs(inv_beta - 1.0/beta) < 1e-10);

    std::cout << "(beta=" << std::fixed << std::setprecision(1) << beta << " kg/m^2) ";
}

// Test: Drag acceleration magnitude at CHAMP altitude
void test_champ_drag_magnitude() {
    // Create a state at ~350 km altitude
    double altitude = 350000.0;  // meters
    double r = constants::earth::RADIUS_EQUATORIAL + altitude;
    double v = std::sqrt(constants::earth::GM / r);  // Circular orbit velocity

    Vector3d pos_eci(r, 0.0, 0.0);
    Vector3d vel_eci(0.0, v, 0.0);
    StateVector state(pos_eci, vel_eci);

    Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
    WeatherConditions weather = make_weather<CHAMPConfig>();

    // Build a snapshot DragEnvironment at CHAMP altitude on the equator.
    auto bal = make_ballistic_model<CHAMPConfig>();
    atmosphere::TimeWindow window{epoch, epoch + 60.0};
    auto sw = atmosphere::SpaceWeatherHistory::from_constant(weather, window);
    auto beta = atmosphere::BetaProfile1D::constant(bal.beta);
    auto env = atmosphere::from_models_snapshot(
        epoch, 0.0, 0.0, sw, std::move(beta),
        std::vector<double>{200, 250, 300, 350, 400, 450, 500});

    forces::AtmosphericDrag drag;
    Vector3d accel = drag.acceleration(env, epoch, state);
    double accel_mag = accel.norm();

    // At 350 km with NRLMSIS 2.1, CHAMP's 2005 weather and the registry beta
    // this gives about 7e-7 m/s^2; the bounds below are a wide plausibility band.
    std::cout << "(a=" << std::scientific << std::setprecision(2) << accel_mag << " m/s^2) ";

    TEST_ASSERT(accel_mag > 1e-9);   // Should be measurable
    TEST_ASSERT(accel_mag < 2e-4);   // Should not be unreasonably high
}

// Test: Drag direction is anti-velocity
void test_champ_drag_direction() {
    double altitude = 350000.0;
    double r = constants::earth::RADIUS_EQUATORIAL + altitude;
    double v = std::sqrt(constants::earth::GM / r);

    Vector3d pos_eci(r, 0.0, 0.0);
    Vector3d vel_eci(0.0, v, 0.0);
    StateVector state(pos_eci, vel_eci);

    Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);
    WeatherConditions weather = make_weather<CHAMPConfig>();

    auto bal = make_ballistic_model<CHAMPConfig>();
    atmosphere::TimeWindow window{epoch, epoch + 60.0};
    auto sw = atmosphere::SpaceWeatherHistory::from_constant(weather, window);
    auto beta = atmosphere::BetaProfile1D::constant(bal.beta);
    auto env = atmosphere::from_models_snapshot(
        epoch, 0.0, 0.0, sw, std::move(beta),
        std::vector<double>{200, 250, 300, 350, 400, 450, 500});

    forces::AtmosphericDrag drag;
    Vector3d accel = drag.acceleration(env, epoch, state);

    // Drag should be roughly anti-parallel to velocity (not exactly: the
    // atmosphere co-rotates with the Earth, so the relative wind is tilted).
    double dot = accel.normalized().dot(vel_eci.normalized());
    TEST_ASSERT(dot < -0.9);  // Mostly opposite to velocity
}

// =============================================================================
// Short-Term Propagation Tests
// =============================================================================

// Test: 1-minute propagation accuracy
void test_1_minute_propagation() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 100) {
        TEST_SKIP("insufficient records");
    }

    // Start from 10th record to be safely inside valid range
    const auto& sv0 = records[10];
    Epoch t0 = sv0.epoch;
    Vector3d pos0 = sv0.position;
    Vector3d vel0 = sv0.has_velocity ? sv0.velocity : Vector3d::Zero();

    if (!sv0.has_velocity) {
        TEST_SKIP("no velocity in SP3");
    }

    double v_mag = vel0.norm();
    if (v_mag < 1000.0 || v_mag > 10000.0) {  // LEO velocity should be ~7.5 km/s
        TEST_SKIP("bad velocity: " << v_mag << " m/s");
    }

    auto results = propagate_with_sampling<CHAMPConfig>(t0, pos0, vel0, 60.0, 10.0);
    auto test_result = compute_propagation_result_sp3(results, *sp3, CHAMPConfig::SP3_ID);

    std::cout << "(RMSE=" << std::fixed << std::setprecision(2)
              << test_result.position_rmse << "m) ";

    // Loose bound that catches a broken setup, not an accuracy claim (measured
    // at the centimeter level; the 1-hour cell is at the meter level).
    TEST_ASSERT(test_result.position_rmse < 200.0);
}

// Test: 5-minute propagation accuracy
void test_5_minute_propagation() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 100) {
        TEST_SKIP("insufficient records");
    }

    const auto& sv0 = records[10];
    if (!sv0.has_velocity) {
        TEST_SKIP("no velocity");
    }

    auto results = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 300.0, 10.0);
    auto test_result = compute_propagation_result_sp3(results, *sp3, CHAMPConfig::SP3_ID);

    std::cout << "(RMSE=" << std::fixed << std::setprecision(2)
              << test_result.position_rmse << "m) ";

    // Loose bound that catches a broken setup, not an accuracy claim (measured
    // at the centimeter level).
    TEST_ASSERT(test_result.position_rmse < 500.0);
}

// Test: 10-minute propagation accuracy
void test_10_minute_propagation() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 100) {
        TEST_SKIP("insufficient records");
    }

    const auto& sv0 = records[10];
    if (!sv0.has_velocity) {
        TEST_SKIP("no velocity");
    }

    auto results = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 600.0, 10.0);
    auto test_result = compute_propagation_result_sp3(results, *sp3, CHAMPConfig::SP3_ID);

    std::cout << "(RMSE=" << std::fixed << std::setprecision(2)
              << test_result.position_rmse << "m) ";

    // Loose bound that catches a broken setup, not an accuracy claim (measured
    // about 0.1 m).
    TEST_ASSERT(test_result.position_rmse < 1000.0);
}

// =============================================================================
// One-Hour Propagation Tests (Primary Target)
// =============================================================================

// Test: 1-hour propagation accuracy
void test_1_hour_propagation() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records for 1-hour test");
    }

    const auto& sv0 = records[10];
    if (!sv0.has_velocity) {
        TEST_SKIP("no velocity");
    }

    auto results = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0);
    auto test_result = compute_propagation_result_sp3(results, *sp3, CHAMPConfig::SP3_ID);

    std::cout << "\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << test_result.position_rmse << " m\n";
    std::cout << "    Position Max:  " << test_result.position_max_error << " m\n";
    std::cout << "    Radial RMS:    " << test_result.radial_rms << " m\n";
    std::cout << "    Along-trk RMS: " << test_result.along_track_rms << " m\n";
    std::cout << "    Cross-trk RMS: " << test_result.cross_track_rms << " m\n";
    std::cout << "    Velocity RMSE: " << test_result.velocity_rmse * 1000 << " mm/s\n";
    std::cout << "    ";

    if (test_result.position_rmse < 50.0) {
        std::cout << "EXCELLENT (<50m)... ";
    } else if (test_result.position_rmse < 100.0) {
        std::cout << "GOOD (<100m)... ";
    } else if (test_result.position_rmse < 500.0) {
        std::cout << "ACCEPTABLE (<500m)... ";
    } else {
        std::cout << "NEEDS IMPROVEMENT... ";
    }

    // With the registry beta and proper configuration (GCRS + ITRF2005 + JPL
    // ephemeris) CHAMP is at the meter level (CHAMP-370KM); 50 m is a loose
    // upper bound that still catches a broken drag path.
    TEST_ASSERT_MSG(test_result.position_rmse < 50.0,
        "CHAMP 1-hour RMSE should be <50m with proper configuration");
}

// Test: Compare drag ON vs OFF to quantify drag contribution
void test_1_hour_drag_on_vs_off() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records");
    }

    const auto& sv0 = records[10];
    if (!sv0.has_velocity) {
        TEST_SKIP("no velocity");
    }

    // Propagate with drag ON
    auto results_drag_on = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0, true);
    auto rmse_drag_on = compute_propagation_result_sp3(
        results_drag_on, *sp3, CHAMPConfig::SP3_ID);

    // Propagate with drag OFF
    auto results_drag_off = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0, false);
    auto rmse_drag_off = compute_propagation_result_sp3(
        results_drag_off, *sp3, CHAMPConfig::SP3_ID);

    std::cout << "\n";
    std::cout << "    Drag ON  RMSE: " << std::fixed << std::setprecision(2)
              << rmse_drag_on.position_rmse << " m\n";
    std::cout << "    Drag OFF RMSE: " << rmse_drag_off.position_rmse << " m\n";

    double drag_contribution = rmse_drag_off.position_rmse - rmse_drag_on.position_rmse;
    std::cout << "    Drag contribution: " << drag_contribution << " m\n";
    std::cout << "    ";

    // Informational: a positive contribution means modeling drag helps.
    if (drag_contribution > 50.0) {
        std::cout << "DRAG MODELING HELPS (" << std::setprecision(0)
                  << drag_contribution << "m improvement)... ";
    } else if (drag_contribution > 0.0) {
        std::cout << "MINOR IMPROVEMENT... ";
    } else {
        std::cout << "NO IMPROVEMENT (drag may need tuning)... ";
    }

    // Asserted: both runs finite and positive, and drag changes the error by
    // more than 0.1 m (on this arc drag lowers the 1-hour RMSE by about 1.5 m).
    TEST_ASSERT_MSG(std::isfinite(rmse_drag_on.position_rmse),
        "Drag-ON RMSE should be finite");
    TEST_ASSERT_MSG(std::isfinite(rmse_drag_off.position_rmse),
        "Drag-OFF RMSE should be finite");
    TEST_ASSERT_MSG(rmse_drag_on.position_rmse > 0 && rmse_drag_off.position_rmse > 0,
        "RMSE values should be positive");
    TEST_ASSERT_MSG(std::abs(rmse_drag_on.position_rmse - rmse_drag_off.position_rmse) > 0.1,
        "Drag should have measurable effect at CHAMP altitude");
}

// =============================================================================
// DragEnvironment dual-path validation gate
// =============================================================================
//
// Propagate CHAMP for 1 hour two ways and compare each against the same SP3
// truth:
//   1. Reference: propagate_with_sampling with drag on, which tabulates
//      NRLMSIS 2.1 along a drag-free nominal sampled every 60 s. (Despite the
//      "legacy"/"live" names in the code and messages below, this harness has
//      no path that evaluates NRLMSIS inside the integration loop.)
//   2. Env path: the same kind of table built here with a chosen sample
//      spacing (sample_dt) or as a coarse 4D field, attached to the force model.
//
// Claim: the 1-hour RMSE does not depend on how the table was sampled, to
// within 5%. If it fails, the sample spacing or table layout needs revisiting.
//
// Helper: build the table at `env_sample_dt_sec` from a drag-free nominal run
// and propagate with it.
// =============================================================================
template<typename Config>
std::vector<std::pair<Epoch, StateVector>> propagate_with_env_sampling(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    double sample_interval,
    double env_sample_dt_sec)
{
    // Step 1: drag-free nominal propagation to feed from_models_along.
    auto nominal = propagate_with_sampling<Config>(
        t0, pos0, vel0, duration_seconds, env_sample_dt_sec, /*enable_drag=*/false);

    std::vector<Epoch> sample_epochs;
    std::vector<Vector3d> sample_positions_ecef;
    sample_epochs.reserve(nominal.size());
    sample_positions_ecef.reserve(nominal.size());
    for (const auto& [t, state] : nominal) {
        sample_epochs.push_back(t);
        sample_positions_ecef.push_back(state.position);  // propagator returns ECEF
    }

    // Step 2: SpaceWeatherHistory + BetaProfile1D from CHAMP config.
    WeatherConditions cw = make_weather<Config>();
    atmosphere::SpaceWeatherHistory sw =
        atmosphere::SpaceWeatherHistory::from_constant(
            cw, atmosphere::TimeWindow{t0 - 86400.0,
                                         t0 + duration_seconds + 86400.0});
    const double beta_kg_m2 = Config::MASS / (Config::CD * Config::DRAG_AREA);
    atmosphere::BetaProfile1D beta_prior =
        atmosphere::BetaProfile1D::constant(beta_kg_m2);

    // Step 3: build the table in a single pass (no re-sampling along the
    // drag-on trajectory).
    auto env = std::make_shared<const atmosphere::DragEnvironment>(
        atmosphere::from_models_along(
            sample_epochs, sample_positions_ecef, sw, std::move(beta_prior)));

    // Step 4: new propagator with the env injected.
    BallisticModel model = make_ballistic_model<Config>();
    auto prop = create_gcrs_propagator(t0, pos0, vel0, model, cw, true,
                                        g_ephemeris, coords::ITRFRealization::ITRF2005);
    prop->force_model().set_drag_environment(env);

    // Step 5: sample at the requested cadence.
    std::vector<std::pair<Epoch, StateVector>> results;
    const int num_samples = static_cast<int>(duration_seconds / sample_interval) + 1;
    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * sample_interval;
        if (dt > duration_seconds) dt = duration_seconds;
        Epoch t = t0 + dt;
        auto r = (*prop)(t);
        results.emplace_back(t, StateVector(r.position, r.velocity));
    }
    return results;
}

// Table sampled every 60 s vs the reference: 1-hour RMSE within 5%.
void test_1_hour_drag_env_dual_path_60s() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }
    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200 || !records[10].has_velocity) {
        TEST_SKIP("insufficient records");
    }
    const auto& sv0 = records[10];

    // Reference ("legacy" in the code): propagate_with_sampling with drag on.
    auto legacy = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0, /*drag*/ true);
    auto rmse_legacy = compute_propagation_result_sp3(
        legacy, *sp3, CHAMPConfig::SP3_ID);

    // Env path: from_models_along with sample_dt = 60s.
    auto env_path = propagate_with_env_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0, 60.0);
    auto rmse_env = compute_propagation_result_sp3(
        env_path, *sp3, CHAMPConfig::SP3_ID);

    const double r_old = rmse_legacy.position_rmse;
    const double r_new = rmse_env.position_rmse;
    const double delta_rel = std::abs(r_new - r_old) / r_old;

    std::cout << "\n";
    std::cout << "    [dual path] sample_dt=60s\n";
    std::cout << "    rmse_old=" << std::fixed << std::setprecision(3) << r_old
              << " m   rmse_new=" << r_new << " m   delta="
              << std::setprecision(2) << (delta_rel * 100.0) << "%\n";
    std::cout << "    ";

    TEST_ASSERT_MSG(std::isfinite(r_old) && r_old > 0.0,
        "Legacy NRLMSIS21 path RMSE should be finite and positive");
    TEST_ASSERT_MSG(std::isfinite(r_new) && r_new > 0.0,
        "Env-path RMSE should be finite and positive");
    TEST_ASSERT_MSG(delta_rel < 0.05,
        "Env-path RMSE must be within 5% of live-NRLMSIS21 path "
        "(architectural gate for tabulated drag environments). "
        "If this fails, try a tighter sample_dt or the 4D field.");
}

// Table sampled every 30 s vs the reference. The difference should stay small
// as sampling gets finer; only a loose 10% bound is asserted, because the
// number also depends on integrator step sizes.
void test_1_hour_drag_env_dual_path_30s() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }
    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200 || !records[10].has_velocity) {
        TEST_SKIP("insufficient records");
    }
    const auto& sv0 = records[10];

    auto legacy = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0, /*drag*/ true);
    auto rmse_legacy = compute_propagation_result_sp3(
        legacy, *sp3, CHAMPConfig::SP3_ID);

    auto env_path = propagate_with_env_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0, 30.0);
    auto rmse_env = compute_propagation_result_sp3(
        env_path, *sp3, CHAMPConfig::SP3_ID);

    const double r_old = rmse_legacy.position_rmse;
    const double r_new = rmse_env.position_rmse;
    const double delta_rel = std::abs(r_new - r_old) / r_old;

    std::cout << "\n";
    std::cout << "    [dual path] sample_dt=30s\n";
    std::cout << "    rmse_old=" << std::fixed << std::setprecision(3) << r_old
              << " m   rmse_new=" << r_new << " m   delta="
              << std::setprecision(2) << (delta_rel * 100.0) << "%\n";
    std::cout << "    ";

    // Sanity checks only: both finite, both positive, env path within a
    // loose 10% tolerance (the 60 s cell pins the strict 5% gate).
    TEST_ASSERT_MSG(std::isfinite(r_old) && r_old > 0.0,
        "Legacy RMSE should be finite and positive");
    TEST_ASSERT_MSG(std::isfinite(r_new) && r_new > 0.0,
        "Env-path RMSE should be finite and positive");
    TEST_ASSERT_MSG(delta_rel < 0.10,
        "Env-path RMSE at 30s sample_dt should not regress dramatically vs 60s");
}

// Coarse 4D field (altitude x latitude x longitude x time) vs the reference:
// 1-hour RMSE within 5%, so the 4D generator can stand in for the
// trajectory-following table. Opt-in (BALLISTIC_RUN_HEAVY_TESTS=1): building
// the field takes minutes.
void test_1_hour_drag_env_dual_path_4d() {
    const char* heavy = std::getenv("BALLISTIC_RUN_HEAVY_TESTS");
    if (!heavy || std::string(heavy) == "0") {
        TEST_SKIP_OPT_IN("set BALLISTIC_RUN_HEAVY_TESTS=1 to enable");
    }

    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }
    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200 || !records[10].has_velocity) {
        TEST_SKIP("insufficient records");
    }
    const auto& sv0 = records[10];

    // Reference path
    auto legacy = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0, /*drag*/ true);
    auto rmse_legacy = compute_propagation_result_sp3(
        legacy, *sp3, CHAMPConfig::SP3_ID);

    // Build a small 4D field around CHAMP's actual flight envelope.
    // CHAMP is at ~380 km in a near-polar orbit (inclination 87.3°), so: a
    // narrow altitude band, global latitude/longitude at coarse spacing.
    std::vector<double> alt_grid;
    for (double h = 300.0; h <= 500.0; h += 25.0) alt_grid.push_back(h);   // 9 alts
    std::vector<double> lat_grid;
    for (double l = -90.0; l <= 90.0; l += 30.0) lat_grid.push_back(l);     // 7 lats
    std::vector<double> lon_grid;
    for (double l = -180.0; l < 180.0; l += 60.0) lon_grid.push_back(l);    // 6 lons
    std::vector<Epoch> epoch_grid{
        sv0.epoch - 1800.0, sv0.epoch, sv0.epoch + 1800.0,
        sv0.epoch + 3600.0, sv0.epoch + 5400.0,
    };  // 5 epochs spanning the arc

    WeatherConditions cw = make_weather<CHAMPConfig>();
    atmosphere::SpaceWeatherHistory sw =
        atmosphere::SpaceWeatherHistory::from_constant(
            cw, atmosphere::TimeWindow{sv0.epoch - 86400.0,
                                         sv0.epoch + 86400.0});
    const double beta_kg_m2 = CHAMPConfig::MASS
                              / (CHAMPConfig::CD * CHAMPConfig::DRAG_AREA);
    auto env = std::make_shared<const atmosphere::DragEnvironment>(
        atmosphere::from_models_4d(
            alt_grid, lat_grid, lon_grid, epoch_grid, sw,
            atmosphere::BetaProfile1D::constant(beta_kg_m2)));

    // Propagate with the 4D env.
    BallisticModel model = make_ballistic_model<CHAMPConfig>();
    auto prop = create_gcrs_propagator(
        sv0.epoch, sv0.position, sv0.velocity, model, cw, true,
        g_ephemeris, coords::ITRFRealization::ITRF2005);
    prop->force_model().set_drag_environment(env);

    std::vector<std::pair<Epoch, StateVector>> env_path;
    const int num_samples = static_cast<int>(3600.0 / 10.0) + 1;
    for (int i = 0; i <= num_samples; ++i) {
        double dt = i * 10.0;
        if (dt > 3600.0) dt = 3600.0;
        Epoch t = sv0.epoch + dt;
        auto r = (*prop)(t);
        env_path.emplace_back(t, StateVector(r.position, r.velocity));
    }
    auto rmse_env = compute_propagation_result_sp3(
        env_path, *sp3, CHAMPConfig::SP3_ID);

    const double r_old = rmse_legacy.position_rmse;
    const double r_new = rmse_env.position_rmse;
    const double delta_rel = std::abs(r_new - r_old) / r_old;

    std::cout << "\n";
    std::cout << "    [dual path] 4D field (heavy)\n";
    std::cout << "    grid = " << alt_grid.size() << " alt × "
              << lat_grid.size() << " lat × "
              << lon_grid.size() << " lon × "
              << epoch_grid.size() << " t = "
              << (alt_grid.size() * lat_grid.size() * lon_grid.size()
                  * epoch_grid.size()) << " cells\n";
    std::cout << "    rmse_old=" << std::fixed << std::setprecision(3) << r_old
              << " m   rmse_new=" << r_new << " m   delta="
              << std::setprecision(2) << (delta_rel * 100.0) << "%\n";
    std::cout << "    ";

    TEST_ASSERT_MSG(std::isfinite(r_new) && r_new > 0.0,
        "4D-env-path RMSE should be finite and positive");
    // Same 5% bound as the 60 s cell.
    TEST_ASSERT_MSG(delta_rel < 0.05,
        "4D-env-path RMSE must be within 5% of live-NRLMSIS21 path");
}

// Test: split the 1-hour error into radial / along-track / cross-track (RTN).
// Drag errors show up mostly along-track; that pattern is printed, not
// asserted. Asserted: the three parts are non-negative and their squared
// fractions sum to 100% (within 1%).
void test_1_hour_rtn_decomposition() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records");
    }

    const auto& sv0 = records[10];
    if (!sv0.has_velocity) {
        TEST_SKIP("no velocity");
    }

    auto results = propagate_with_sampling<CHAMPConfig>(
        sv0.epoch, sv0.position, sv0.velocity, 3600.0, 10.0);
    auto test_result = compute_propagation_result_sp3(results, *sp3, CHAMPConfig::SP3_ID);

    double total_sq = test_result.radial_rms * test_result.radial_rms +
                     test_result.along_track_rms * test_result.along_track_rms +
                     test_result.cross_track_rms * test_result.cross_track_rms;

    double radial_frac = 100.0 * test_result.radial_rms * test_result.radial_rms / total_sq;
    double along_frac = 100.0 * test_result.along_track_rms * test_result.along_track_rms / total_sq;
    double cross_frac = 100.0 * test_result.cross_track_rms * test_result.cross_track_rms / total_sq;

    std::cout << "\n";
    std::cout << "    Radial:      " << std::fixed << std::setprecision(1)
              << test_result.radial_rms << " m (" << radial_frac << "%)\n";
    std::cout << "    Along-track: " << test_result.along_track_rms << " m (" << along_frac << "%)\n";
    std::cout << "    Cross-track: " << test_result.cross_track_rms << " m (" << cross_frac << "%)\n";
    std::cout << "    ";

    if (along_frac > 50.0) {
        std::cout << "ALONG-TRACK DOMINATED (drag signature)... ";
    } else if (cross_frac > 50.0) {
        std::cout << "CROSS-TRACK DOMINATED (gravity/transform issue)... ";
    } else {
        std::cout << "MIXED ERROR PATTERN... ";
    }

    TEST_ASSERT_MSG(test_result.radial_rms >= 0, "Radial RMS should be non-negative");
    TEST_ASSERT_MSG(test_result.along_track_rms >= 0, "Along-track RMS should be non-negative");
    TEST_ASSERT_MSG(test_result.cross_track_rms >= 0, "Cross-track RMS should be non-negative");
    TEST_ASSERT_MSG(std::abs(radial_frac + along_frac + cross_frac - 100.0) < 1.0,
        "RTN fractions should sum to ~100%");
}

// =============================================================================
// Atmosphere Model Comparison Tests
// =============================================================================

// Test: NRLMSIS 2.1 density at CHAMP altitude is in the physical range.
void test_atmosphere_model_comparison() {
    double altitude_km = 350.0;
    Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);

    atmosphere::NRLMSIS21 msis21;
    WeatherConditions weather = make_weather<CHAMPConfig>();
    double rho_msis21 = msis21.density(altitude_km, 0.0, 0.0, epoch, weather);

    std::cout << "\n    NRLMSIS 2.1 at " << altitude_km << " km: " << std::scientific
              << std::setprecision(2) << rho_msis21 << " kg/m^3\n    ";

    // Physical range at 350 km over the solar cycle: ~1e-12 to 1e-9 kg/m^3
    TEST_ASSERT(rho_msis21 > 1e-12 && rho_msis21 < 1e-9);
}

// Test: 1-hour error at the end point for a range of Cd values.
// NOTE: these propagators get no DragEnvironment, so drag is OFF for every Cd
// and the rows are identical (the BallisticModel is not read for drag, and
// its arguments here are not in (beta, mass, area) order either). Only
// finiteness and a < 100 km bound are asserted.
void test_1_hour_propagation_msis21_cd_sweep() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records");
    }

    // Find record with velocity
    const ballistic::test::TruthRecord* sv0 = nullptr;
    for (size_t i = 10; i < records.size() - 200; ++i) {
        if (records[i].has_velocity) {
            sv0 = &records[i];
            break;
        }
    }
    if (!sv0) {
        TEST_SKIP("no velocity");
    }

    std::cout << "\n";
    std::cout << "    NRLMSIS 2.1 Drag Coefficient Sweep (1-hour):\n";
    std::cout << "    Cd      RMSE(m)\n";
    std::cout << "    ----------------\n";

    WeatherConditions weather = make_weather<CHAMPConfig>();
    Epoch t_final = sv0->epoch + 3600.0;

    // Find truth
    const ballistic::test::TruthRecord* truth = nullptr;
    double min_diff = 1e9;
    for (const auto& rec : records) {
        double diff = std::abs(rec.epoch - t_final);
        if (diff < min_diff) {
            min_diff = diff;
            truth = &rec;
        }
    }

    if (!truth || min_diff > 5.0) {
        TEST_SKIP("no truth data for end time");
    }

    std::vector<double> cd_values = {0.5, 0.8, 1.0, 1.5, 2.0, 2.2, 2.5, 3.0};
    double best_rmse = 1e9;
    double best_cd = 2.2;

    for (double cd : cd_values) {
        BallisticModel model(CHAMPConfig::MASS, CHAMPConfig::DRAG_AREA, cd);

        auto prop = create_gcrs_propagator(sv0->epoch, sv0->position, sv0->velocity,
                                           model, weather, true, g_ephemeris,
                                           coords::ITRFRealization::ITRF2005);

        auto result = (*prop)(t_final);
        double rmse = (result.position - truth->position).norm();

        std::cout << "    " << std::fixed << std::setprecision(1) << std::setw(4) << cd
                  << "    " << std::setprecision(1) << std::setw(8) << rmse << "\n";

        if (rmse < best_rmse) {
            best_rmse = rmse;
            best_cd = cd;
        }
    }

    std::cout << "    ----------------\n";
    std::cout << "    Best: Cd=" << std::setprecision(1) << best_cd
              << " (" << best_rmse << " m)\n";
    std::cout << "    ";

    TEST_ASSERT_MSG(std::isfinite(best_rmse), "Best RMSE should be finite");
    TEST_ASSERT_MSG(best_rmse > 0 && best_rmse < 100000.0,
        "Best RMSE should be positive and reasonable (<100km)");
}

// Test: 1-hour end-point error with drag off vs "NRLMSIS 2.1 at Cd 2.2".
// NOTE: the second propagator also gets no DragEnvironment, so both runs are
// drag-free; only finiteness and positivity are asserted. The drag on/off
// comparison that does attach drag is test_1_hour_drag_on_vs_off.
void test_msis21_drag_contribution() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records");
    }

    // Find record with velocity
    const ballistic::test::TruthRecord* sv0 = nullptr;
    for (size_t i = 10; i < records.size() - 200; ++i) {
        if (records[i].has_velocity) {
            sv0 = &records[i];
            break;
        }
    }
    if (!sv0) {
        TEST_SKIP("no velocity");
    }

    WeatherConditions weather = make_weather<CHAMPConfig>();
    Epoch t_final = sv0->epoch + 3600.0;

    // Find truth
    const ballistic::test::TruthRecord* truth = nullptr;
    double min_diff = 1e9;
    for (const auto& rec : records) {
        double diff = std::abs(rec.epoch - t_final);
        if (diff < min_diff) {
            min_diff = diff;
            truth = &rec;
        }
    }

    if (!truth || min_diff > 5.0) {
        TEST_SKIP("no truth data");
    }

    // Drag OFF
    BallisticModel model_nodrag(CHAMPConfig::MASS, CHAMPConfig::DRAG_AREA, 2.2);
    auto prop_nodrag = create_gcrs_propagator(sv0->epoch, sv0->position, sv0->velocity,
                                              model_nodrag, weather, true, g_ephemeris,
                                              coords::ITRFRealization::ITRF2005);
    {
        forces::ForceComponentFlags flags = prop_nodrag->force_model().force_flags();
        flags.drag = false;
        prop_nodrag->force_model().set_force_flags(flags);
    }
    auto result_nodrag = (*prop_nodrag)(t_final);
    double rmse_nodrag = (result_nodrag.position - truth->position).norm();

    // Labeled "NRLMSIS 2.1, Cd 2.2", but no DragEnvironment is attached.
    BallisticModel model_msis21(CHAMPConfig::MASS, CHAMPConfig::DRAG_AREA, 2.2);
    auto prop_msis21 = create_gcrs_propagator(sv0->epoch, sv0->position, sv0->velocity,
                                              model_msis21, weather, true, g_ephemeris,
                                              coords::ITRFRealization::ITRF2005);
    auto result_msis21 = (*prop_msis21)(t_final);
    double rmse_msis21 = (result_msis21.position - truth->position).norm();

    std::cout << "\n";
    std::cout << "    Drag contribution (1-hour, Cd=2.2 physical):\n";
    std::cout << "    Drag OFF:     " << std::fixed << std::setprecision(1)
              << rmse_nodrag << " m\n";
    std::cout << "    NRLMSIS 2.1:  " << rmse_msis21 << " m\n";

    double improvement = rmse_nodrag - rmse_msis21;
    std::cout << "    Drag effect:  " << (improvement > 0 ? "+" : "")
              << improvement << " m\n";
    std::cout << "    ";

    TEST_ASSERT_MSG(std::isfinite(rmse_nodrag) && rmse_nodrag > 0,
        "No-drag RMSE should be finite and positive");
    TEST_ASSERT_MSG(std::isfinite(rmse_msis21) && rmse_msis21 > 0,
        "MSIS 2.1 RMSE should be finite and positive");
}

// =============================================================================
// Solar Activity Sensitivity Tests
// =============================================================================

// Test: NRLMSIS 2.1 density at 350 km rises with solar activity (quiet <
// active < storm), by at least 1.5x from F10.7 70 to 150.
void test_solar_activity_sensitivity() {
    double altitude_km = 350.0;
    Epoch epoch = Epoch::from_utc(2005, 7, 1, 12, 0, 0);

    atmosphere::NRLMSIS21 model;

    // Quiet sun (F10.7 = 70)
    WeatherConditions quiet;
    quiet.f107_daily = 70.0;
    quiet.f107_average = 72.0;
    quiet.ap_daily = 5.0;
    quiet.ap_history.fill(5.0);
    double rho_quiet = model.density(altitude_km, 0.0, 0.0, epoch, quiet);

    // Active sun (F10.7 = 150)
    WeatherConditions active;
    active.f107_daily = 150.0;
    active.f107_average = 145.0;
    active.ap_daily = 15.0;
    active.ap_history.fill(15.0);
    double rho_active = model.density(altitude_km, 0.0, 0.0, epoch, active);

    // Storm conditions (F10.7 = 200, high Ap)
    WeatherConditions storm;
    storm.f107_daily = 200.0;
    storm.f107_average = 190.0;
    storm.ap_daily = 100.0;
    storm.ap_history.fill(100.0);
    double rho_storm = model.density(altitude_km, 0.0, 0.0, epoch, storm);

    std::cout << "\n";
    std::cout << "    At " << altitude_km << " km:\n";
    std::cout << "    Quiet sun (F10.7=70):  " << std::scientific << std::setprecision(2)
              << rho_quiet << " kg/m^3\n";
    std::cout << "    Active sun (F10.7=150): " << rho_active << " kg/m^3\n";
    std::cout << "    Storm (F10.7=200):     " << rho_storm << " kg/m^3\n";

    double active_ratio = rho_active / rho_quiet;
    double storm_ratio = rho_storm / rho_quiet;
    std::cout << "    Active/Quiet ratio: " << std::fixed << std::setprecision(1)
              << active_ratio << "x\n";
    std::cout << "    Storm/Quiet ratio:  " << storm_ratio << "x\n";
    std::cout << "    ";

    TEST_ASSERT(rho_active > rho_quiet);
    TEST_ASSERT(rho_storm > rho_active);
    TEST_ASSERT(active_ratio > 1.5);  // Expect at least 1.5x increase
}

// =============================================================================
// Sensitivity Analysis Tests (Investigation)
// =============================================================================

// Report-only: 1-hour end-point error (drag off) for gravity field degrees
// 20-120 at CHAMP's ~380 km. Nothing is asserted.
void test_gravity_degree_sensitivity() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records");
    }

    // Find record with velocity
    const ballistic::test::TruthRecord* sv0 = nullptr;
    for (size_t i = 10; i < records.size() - 200; ++i) {
        if (records[i].has_velocity) {
            sv0 = &records[i];
            break;
        }
    }
    if (!sv0) {
        TEST_SKIP("no velocity");
    }

    std::cout << "\n";
    std::cout << "    Gravity Degree Sensitivity (1-hour, drag OFF):\n";
    std::cout << "    Degree   RMSE(m)\n";
    std::cout << "    -----------------\n";

    std::vector<int> degrees = {20, 50, 70, 100, 120};
    double best_rmse = 1e9;
    int best_degree = 70;

    for (int deg : degrees) {
        // Propagator with this gravity degree (no DragEnvironment: drag off)
        BallisticModel model(CHAMPConfig::MASS, CHAMPConfig::DRAG_AREA, CHAMPConfig::CD);
        WeatherConditions weather;
        weather.f107_daily = CHAMPConfig::F107_DAILY;
        weather.f107_average = CHAMPConfig::F107_AVERAGE;
        weather.ap_daily = CHAMPConfig::AP_DAILY;
        weather.ap_history.fill(CHAMPConfig::AP_DAILY);

        auto prop = create_gcrs_propagator(sv0->epoch, sv0->position, sv0->velocity,
                                           model, weather, true, g_ephemeris,
                                           coords::ITRFRealization::ITRF2005);
        {
            forces::ForceComponentFlags flags = prop->force_model().force_flags();
            flags.drag = false;
            prop->force_model().set_force_flags(flags);
        }
        prop->force_model().set_gravity_max_degree(deg);

        // Propagate 1 hour
        Epoch t_final = sv0->epoch + 3600.0;
        auto result = (*prop)(t_final);
        Vector3d pos_pred = result.position;

        // Find truth
        const ballistic::test::TruthRecord* truth = nullptr;
        double min_diff = 1e9;
        for (const auto& rec : records) {
            double diff = std::abs(rec.epoch - t_final);
            if (diff < min_diff) {
                min_diff = diff;
                truth = &rec;
            }
        }

        if (truth && min_diff < 5.0) {
            double rmse = (pos_pred - truth->position).norm();
            std::cout << "    " << std::setw(4) << deg << "     "
                      << std::fixed << std::setprecision(1) << std::setw(8) << rmse << "\n";

            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_degree = deg;
            }
        }
    }

    std::cout << "    -----------------\n";
    std::cout << "    Best: degree " << best_degree << " (" << std::setprecision(1)
              << best_rmse << " m)\n";
    std::cout << "    ";
}

// Report-only: 1-hour end-point error for several Cd values. As in the Cd
// sweep above, no DragEnvironment is attached, so every row is drag-free.
void test_cd_sensitivity() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records");
    }

    // Find record with velocity
    const ballistic::test::TruthRecord* sv0 = nullptr;
    for (size_t i = 10; i < records.size() - 200; ++i) {
        if (records[i].has_velocity) {
            sv0 = &records[i];
            break;
        }
    }
    if (!sv0) {
        TEST_SKIP("no velocity");
    }

    std::cout << "\n";
    std::cout << "    Drag Coefficient Sensitivity (1-hour):\n";
    std::cout << "    Cd      RMSE(m)\n";
    std::cout << "    ----------------\n";

    std::vector<double> cd_values = {0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0};
    double best_rmse = 1e9;
    double best_cd = 2.3;

    for (double cd : cd_values) {
        BallisticModel model(CHAMPConfig::MASS, CHAMPConfig::DRAG_AREA, cd);
        WeatherConditions weather;
        weather.f107_daily = CHAMPConfig::F107_DAILY;
        weather.f107_average = CHAMPConfig::F107_AVERAGE;
        weather.ap_daily = CHAMPConfig::AP_DAILY;
        weather.ap_history.fill(CHAMPConfig::AP_DAILY);

        auto prop = create_gcrs_propagator(sv0->epoch, sv0->position, sv0->velocity,
                                           model, weather, true, g_ephemeris,
                                           coords::ITRFRealization::ITRF2005);

        // Propagate 1 hour
        Epoch t_final = sv0->epoch + 3600.0;
        auto result = (*prop)(t_final);
        Vector3d pos_pred = result.position;

        // Find truth
        const ballistic::test::TruthRecord* truth = nullptr;
        double min_diff = 1e9;
        for (const auto& rec : records) {
            double diff = std::abs(rec.epoch - t_final);
            if (diff < min_diff) {
                min_diff = diff;
                truth = &rec;
            }
        }

        if (truth && min_diff < 5.0) {
            double rmse = (pos_pred - truth->position).norm();
            std::cout << "    " << std::fixed << std::setprecision(1) << std::setw(4) << cd
                      << "    " << std::setprecision(1) << std::setw(8) << rmse << "\n";

            if (rmse < best_rmse) {
                best_rmse = rmse;
                best_cd = cd;
            }
        }
    }

    std::cout << "    ----------------\n";
    std::cout << "    Best: Cd=" << std::setprecision(1) << best_cd
              << " (" << best_rmse << " m)\n";
    std::cout << "    ";
}

// Report-only: 1-hour end-point error for several F10.7 values (114.6 is the
// observed value for the day). No DragEnvironment is attached, so the weather
// does not reach the force model and every row is drag-free.
void test_f107_sensitivity() {
    auto sp3 = load_champ_sp3_file();
    if (!sp3) {
        TEST_SKIP("no SP3 file found");
    }

    auto records = sp3->get_satellite_records(CHAMPConfig::SP3_ID);
    if (records.size() < 200) {
        TEST_SKIP("insufficient records");
    }

    // Find record with velocity
    const ballistic::test::TruthRecord* sv0 = nullptr;
    for (size_t i = 10; i < records.size() - 200; ++i) {
        if (records[i].has_velocity) {
            sv0 = &records[i];
            break;
        }
    }
    if (!sv0) {
        TEST_SKIP("no velocity");
    }

    std::cout << "\n";
    std::cout << "    F10.7 Sensitivity (1-hour):\n";
    std::cout << "    F10.7   RMSE(m)\n";
    std::cout << "    ----------------\n";

    std::vector<double> f107_values = {70.0, 90.0, 114.6, 130.0, 150.0};

    for (double f107 : f107_values) {
        BallisticModel model(CHAMPConfig::MASS, CHAMPConfig::DRAG_AREA, CHAMPConfig::CD);
        WeatherConditions weather;
        weather.f107_daily = f107;
        weather.f107_average = CHAMPConfig::F107_AVERAGE;  // Keep 81-day avg fixed
        weather.ap_daily = CHAMPConfig::AP_DAILY;
        weather.ap_history.fill(CHAMPConfig::AP_DAILY);

        auto prop = create_gcrs_propagator(sv0->epoch, sv0->position, sv0->velocity,
                                           model, weather, true, g_ephemeris,
                                           coords::ITRFRealization::ITRF2005);

        Epoch t_final = sv0->epoch + 3600.0;
        auto result = (*prop)(t_final);
        Vector3d pos_pred = result.position;

        // Find truth
        const ballistic::test::TruthRecord* truth = nullptr;
        double min_diff = 1e9;
        for (const auto& rec : records) {
            double diff = std::abs(rec.epoch - t_final);
            if (diff < min_diff) {
                min_diff = diff;
                truth = &rec;
            }
        }

        if (truth && min_diff < 5.0) {
            double rmse = (pos_pred - truth->position).norm();
            std::string marker = (std::abs(f107 - 90.0) < 0.1) ? " (old estimate)" :
                                 (std::abs(f107 - 114.6) < 0.1) ? " (actual)" : "";
            std::cout << "    " << std::fixed << std::setprecision(1) << std::setw(5) << f107
                      << "   " << std::setprecision(1) << std::setw(8) << rmse << marker << "\n";
        }
    }

    std::cout << "    ----------------\n";
    std::cout << "    ";
}

// =============================================================================
// Main Entry Point
// =============================================================================

int main() {
    // Runs without an Earth orientation parameter (EOP) table: UT1-UTC and
    // polar motion are taken as zero. That costs meters at LEO, which the loose
    // bounds above allow for.
    ballistic::time::EOPData::instance().set_policy(ballistic::time::EopPolicy::AssumeZero);
    std::cout << "========================================\n";
    std::cout << "CHAMP Drag Validation Tests\n";
    std::cout << "========================================\n";
    std::cout << "Using GCRS frame (IAU 2006/2000A) + ITRF2005\n";
    std::cout << "\nData path: " << get_champ_data_path() << "\n";

    g_ephemeris = load_jpl_ephemeris();

    // Check for data files
    auto sp3 = load_champ_sp3_file("", true);
    if (!sp3) {
        std::cout << "\nWARNING: No CHAMP SP3 data files found.\n";
        std::cout << "Place SP3 files in: " << get_champ_data_path() << "\n";
        std::cout << "See README.md in that directory for download instructions.\n";
    }

    // SP3 Parser Tests
    TestRunner sp3_runner("SP3 Parser Tests");
    sp3_runner.add_test("sp3_parser_champ_file", []() { test_sp3_parser_champ_file(); });
    sp3_runner.add_test("sp3_satellite_id_filter", []() { test_sp3_satellite_id_filter(); });
    sp3_runner.add_test("sp3_record_values", []() { test_sp3_record_values(); });
    sp3_runner.run();

    // CHAMP Configuration Tests
    TestRunner config_runner("CHAMP Configuration Tests");
    config_runner.add_test("champ_ballistic_coefficient", []() { test_champ_ballistic_coefficient(); });
    config_runner.add_test("champ_drag_magnitude", []() { test_champ_drag_magnitude(); });
    config_runner.add_test("champ_drag_direction", []() { test_champ_drag_direction(); });
    config_runner.run();

    // Short-Term Propagation Tests
    TestRunner short_runner("Short-Term Propagation Tests");
    short_runner.add_test("1_minute_propagation", []() { test_1_minute_propagation(); });
    short_runner.add_test("5_minute_propagation", []() { test_5_minute_propagation(); });
    short_runner.add_test("10_minute_propagation", []() { test_10_minute_propagation(); });
    short_runner.run();

    // One-Hour Propagation Tests
    TestRunner hour_runner("One-Hour Propagation Tests");
    hour_runner.add_test("1_hour_propagation", []() { test_1_hour_propagation(); });
    hour_runner.add_test("1_hour_drag_on_vs_off", []() { test_1_hour_drag_on_vs_off(); });
    hour_runner.add_test("1_hour_rtn_decomposition", []() { test_1_hour_rtn_decomposition(); });
    hour_runner.add_test("1_hour_drag_env_dual_path_60s", []() { test_1_hour_drag_env_dual_path_60s(); });
    hour_runner.add_test("1_hour_drag_env_dual_path_30s", []() { test_1_hour_drag_env_dual_path_30s(); });
    hour_runner.add_test("1_hour_drag_env_dual_path_4d", []() { test_1_hour_drag_env_dual_path_4d(); });
    hour_runner.run();

    // Atmosphere Model Tests
    TestRunner atmos_runner("Atmosphere Model Tests");
    atmos_runner.add_test("atmosphere_model_comparison", []() { test_atmosphere_model_comparison(); });
    atmos_runner.add_test("solar_activity_sensitivity", []() { test_solar_activity_sensitivity(); });
    atmos_runner.run();

    // NRLMSIS 2.1 Tests
    TestRunner msis21_runner("NRLMSIS 2.1 Tests");
    msis21_runner.add_test("1_hour_propagation_msis21_cd_sweep", []() { test_1_hour_propagation_msis21_cd_sweep(); });
    msis21_runner.add_test("msis21_drag_contribution", []() { test_msis21_drag_contribution(); });
    msis21_runner.run();

    // Sensitivity Analysis Tests
    TestRunner sens_runner("Sensitivity Analysis Tests");
    sens_runner.add_test("gravity_degree_sensitivity", []() { test_gravity_degree_sensitivity(); });
    sens_runner.add_test("cd_sensitivity", []() { test_cd_sensitivity(); });
    sens_runner.add_test("f107_sensitivity", []() { test_f107_sensitivity(); });
    sens_runner.run();

    return print_final_summary();
}
