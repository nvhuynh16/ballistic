// =============================================================================
// Multi-Day Orbit Propagation Validation Tests
// =============================================================================
//
// Propagates a Sentinel-1 start state for 24 hours to 7 days and compares the
// prediction with ESA's precise orbit (POE: precise orbit ephemeris, delivered as
// EOF = Earth Explorer Orbit File) along the way. The propagator built here
// (create_rk45_propagator) ignores its satellite and weather arguments, so the
// runs have NO drag and no radiation pressure; the bounds are loose sanity
// limits, not accuracy claims. RMSE = root-mean-square position error.
//
// Data (git-ignored, see tests/data/eof/README.md; cells skip without it):
// - 24-hour and analysis cells: one Sentinel-1A POE file in
//   tests/data/eof/sentinel1_2016/2016-8/
// - 3- and 7-day cells: consecutive Sentinel-1C POE files in
//   tests/data/eof/sentinel_jan2026/, joined into one continuous arc
//
// Asserted bounds: 24 h < 500 m, 3 days < 1500 m (measured 464 m), 7 days
// < 3000 m (measured 842 m). The error-growth, orbital-phase and drag-
// coefficient cells only print tables (the drag-coefficient table's rows are
// identical, because the Cd passed in is unused).
//
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <ballistic/validation/eof_parser.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;
using namespace ballistic::test;

namespace fs = std::filesystem;

// =============================================================================
// Test Configuration
// =============================================================================

// Sentinel-1 spacecraft parameters. The weather values below are passed to
// create_rk45_propagator, which does not use them (no drag in these runs).
struct Sentinel1Config {
    static constexpr double MASS = ballistic::satellites::get("S1A").mass_kg;          // registry (satellites.hpp)
    static constexpr double DRAG_AREA = ballistic::satellites::get("S1A").drag_area_m2;  // frontal
    static constexpr double CD = ballistic::satellites::get("S1A").cd;
    static constexpr double F107_DAILY = 89.0;       // Quiet solar
    static constexpr double F107_AVERAGE = 90.0;     // 81-day average
    static constexpr double AP_DAILY = 6.0;          // Quiet geomagnetic
};

// Multi-day test configuration
struct MultiDayConfig {
    double duration_hours;
    double sample_interval;    // How often to compare (seconds)
    double expected_rmse_min;  // Too good = suspicious
    double expected_rmse_max;  // Maximum acceptable
    std::string name;
};

const std::vector<MultiDayConfig> MULTI_DAY_TESTS = {
    {24.0,  60.0,   5.0,   100.0,  "24 hours"},
    {72.0,  300.0,  30.0,  1000.0, "3 days"},
    {168.0, 600.0,  100.0, 5000.0, "7 days"}
};

// =============================================================================
// Utility Functions
// =============================================================================

std::string get_test_data_path() {
    std::vector<std::string> candidates = {
        BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel1_2016/2016-8",
        "../tests/data/eof/sentinel1_2016/2016-8",
        "../../tests/data/eof/sentinel1_2016/2016-8",
        "tests/data/eof/sentinel1_2016/2016-8"
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }

    return BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel1_2016/2016-8";
}

std::optional<EOFFile> load_eof_file(const std::string& pattern = "", bool debug = false) {
    std::string data_path = get_test_data_path();

    if (debug) std::cout << "  [DEBUG] Looking for EOF files in: " << data_path << std::endl;

    auto result = find_and_parse_eof(data_path, pattern);

    if (result && debug) {
        std::cout << "  [DEBUG] Parsed OK, " << result->state_vectors.size() << " vectors" << std::endl;
    }

    return result;
}

// A continuous multi-day arc: consecutive daily POE files of one satellite
// joined end to end (each file adds only the states after the previous
// file's last one). The 2016 folders hold scattered days (one 26-h file per
// few days), so the 3- and 7-day cells cannot run on them (and must not
// pass as "SKIPPED", trap TEST-SKIP-AS-PASS). The Sentinel-1C set has every
// day 2026-01-01..17 (git-ignored, downloaded separately).  guard-exempt(PUB-DATE): data

std::optional<EOFFile> load_continuous_eof(double hours_needed) {
    const fs::path dir = BALLISTIC_PROJECT_ROOT "/tests/data/eof/sentinel_jan2026";
    if (!fs::is_directory(dir)) return std::nullopt;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        const auto name = e.path().filename().string();
        if (name.rfind("S1C_", 0) == 0 && name.size() > 4 &&
            name.substr(name.size() - 4) == ".EOF") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        // order by validity start (the "_V" field), not by production date
        auto key = [](const fs::path& p) {
            const auto n = p.filename().string();
            return n.substr(n.find("_V") + 2);
        };
        return key(a) < key(b);
    });
    std::optional<EOFFile> joined;
    for (const auto& f : files) {
        auto part = EOFParser::parse(f.string());
        if (!part || part->state_vectors.empty()) continue;
        if (!joined) { joined = std::move(part); continue; }
        const Epoch last = joined->state_vectors.back().epoch;
        if (part->state_vectors.front().epoch - last > 60.0) break;   // a gap: stop
        for (auto& sv : part->state_vectors)
            if (sv.epoch - last > 1e-3) joined->state_vectors.push_back(sv);
        joined->validity_stop = part->validity_stop;
        if (joined->state_vectors.back().epoch - joined->state_vectors.front().epoch
                > hours_needed * 3600.0 + 7200.0) break;
    }
    return joined;
}

std::vector<std::pair<Epoch, StateVector>> propagate_extended(
    const Epoch& t0,
    const Vector3d& pos0,
    const Vector3d& vel0,
    double duration_seconds,
    double sample_interval = 60.0)
{
    std::vector<std::pair<Epoch, StateVector>> results;

    BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, Sentinel1Config::CD);

    WeatherConditions weather;
    weather.f107_daily = Sentinel1Config::F107_DAILY;
    weather.f107_average = Sentinel1Config::F107_AVERAGE;
    weather.ap_daily = Sentinel1Config::AP_DAILY;
    weather.ap_history.fill(Sentinel1Config::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos0, vel0, model, weather, true);

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

struct TimeSeriesError {
    double hours;
    double position_rmse;
    double along_track_rms;
    double cross_track_rms;
    double radial_rms;
};

std::vector<TimeSeriesError> compute_error_time_series(
    const std::vector<std::pair<Epoch, StateVector>>& propagated,
    const EOFFile& truth,
    double checkpoint_interval_hours = 1.0)
{
    std::vector<TimeSeriesError> errors;

    if (propagated.empty()) return errors;

    const Epoch& t0 = propagated[0].first;
    double total_duration = propagated.back().first - t0;
    int num_checkpoints = static_cast<int>(total_duration / (checkpoint_interval_hours * 3600.0)) + 1;

    for (int cp = 0; cp <= num_checkpoints; ++cp) {
        double checkpoint_time = cp * checkpoint_interval_hours * 3600.0;
        if (checkpoint_time > total_duration) break;

        std::vector<double> radial_errors, along_errors, cross_errors;

        for (const auto& [epoch, state] : propagated) {
            double dt = epoch - t0;
            if (dt > checkpoint_time) break;

            auto truth_opt = truth.find_nearest(epoch);
            if (!truth_opt || std::abs(truth_opt->epoch - epoch) > 60.0) continue;

            StateVector truth_state(truth_opt->position, truth_opt->velocity);
            Vector3d error = state.position - truth_state.position;

            Vector3d r_hat = truth_state.position.normalized();
            Vector3d n_hat = truth_state.position.cross(truth_state.velocity).normalized();
            Vector3d t_hat = n_hat.cross(r_hat);

            radial_errors.push_back(error.dot(r_hat));
            along_errors.push_back(error.dot(t_hat));
            cross_errors.push_back(error.dot(n_hat));
        }

        if (radial_errors.empty()) continue;

        auto rms = [](const std::vector<double>& v) {
            double sum_sq = 0.0;
            for (double x : v) sum_sq += x * x;
            return std::sqrt(sum_sq / v.size());
        };

        TimeSeriesError err;
        err.hours = checkpoint_time / 3600.0;
        err.radial_rms = rms(radial_errors);
        err.along_track_rms = rms(along_errors);
        err.cross_track_rms = rms(cross_errors);
        err.position_rmse = std::sqrt(
            err.radial_rms * err.radial_rms +
            err.along_track_rms * err.along_track_rms +
            err.cross_track_rms * err.cross_track_rms
        );

        errors.push_back(err);
    }

    return errors;
}

// =============================================================================
// Test Functions
// =============================================================================

void test_24_hour_propagation() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    if (eof->state_vectors.size() < 8640) {
        TEST_SKIP("insufficient truth data for 24h");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    double duration = 24.0 * 3600.0;
    auto results = propagate_extended(t0, sv0.position, sv0.velocity, duration, 60.0);
    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "\n";
    std::cout << "    Duration: 24 hours\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << test_result.position_rmse << " m\n";
    std::cout << "    Position Max:  " << test_result.position_max_error << " m\n";
    std::cout << "    Radial RMS:    " << test_result.radial_rms << " m\n";
    std::cout << "    Along-trk RMS: " << test_result.along_track_rms << " m\n";
    std::cout << "    Cross-trk RMS: " << test_result.cross_track_rms << " m\n";
    std::cout << "    ";

    TEST_ASSERT(test_result.position_rmse < 500.0);
}

void test_3_day_propagation() {
    auto eof = load_continuous_eof(72.0);
    if (!eof) {
        TEST_SKIP("no Sentinel-1C January 2026 EOF files");
    }

    double available_hours = (eof->state_vectors.back().epoch - eof->state_vectors.front().epoch) / 3600.0;
    if (available_hours < 72.0) {
        TEST_SKIP("only " << std::fixed << std::setprecision(1) << available_hours << " h of truth data, need 72h");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    double duration = 72.0 * 3600.0;
    auto results = propagate_extended(t0, sv0.position, sv0.velocity, duration, 300.0);
    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "\n";
    std::cout << "    Duration: 3 days (72 hours)\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << test_result.position_rmse << " m\n";
    std::cout << "    Position Max:  " << test_result.position_max_error << " m\n";
    std::cout << "    ";

    TEST_ASSERT(test_result.position_rmse < 1500.0);   // measured 464 m (S1C, 2026-01, no drag)  guard-exempt(PUB-DATE): data
}

void test_7_day_propagation() {
    auto eof = load_continuous_eof(168.0);
    if (!eof) {
        TEST_SKIP("no Sentinel-1C January 2026 EOF files");
    }

    double available_hours = (eof->state_vectors.back().epoch - eof->state_vectors.front().epoch) / 3600.0;
    if (available_hours < 168.0) {
        TEST_SKIP("only " << std::fixed << std::setprecision(1) << available_hours << " h of truth data, need 168h");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    double duration = 168.0 * 3600.0;
    auto results = propagate_extended(t0, sv0.position, sv0.velocity, duration, 600.0);
    auto test_result = compute_propagation_result(results, *eof);

    std::cout << "\n";
    std::cout << "    Duration: 7 days (168 hours)\n";
    std::cout << "    Position RMSE: " << std::fixed << std::setprecision(2)
              << test_result.position_rmse << " m\n";
    std::cout << "    Position Max:  " << test_result.position_max_error << " m\n";
    std::cout << "    ";

    TEST_ASSERT(test_result.position_rmse < 3000.0);   // measured 842 m
}

void test_error_growth_rate_24h() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    if (eof->state_vectors.size() < 8640) {
        TEST_SKIP("insufficient truth data");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    double duration = 24.0 * 3600.0;
    auto results = propagate_extended(t0, sv0.position, sv0.velocity, duration, 60.0);
    auto errors = compute_error_time_series(results, *eof, 1.0);

    std::cout << "\n";
    std::cout << "    Error Growth Analysis (24 hours):\n";
    std::cout << "    Hour    RMSE(m)    Radial    Along-Trk  Cross-Trk  Rate(m/h)\n";
    std::cout << "    ----------------------------------------------------------------\n";

    double prev_rmse = 0.0;
    for (const auto& err : errors) {
        double rate = err.hours > 0 ? (err.position_rmse - prev_rmse) : 0.0;
        std::cout << "    " << std::setw(4) << std::fixed << std::setprecision(0) << err.hours
                  << "     " << std::setw(8) << std::setprecision(2) << err.position_rmse
                  << "   " << std::setw(8) << err.radial_rms
                  << "  " << std::setw(8) << err.along_track_rms
                  << "  " << std::setw(8) << err.cross_track_rms
                  << "  " << std::setw(8) << rate << "\n";
        prev_rmse = err.position_rmse;
    }
    std::cout << "    ----------------------------------------------------------------\n";
    std::cout << "    ";
}

void test_error_by_orbital_position() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    double orbital_period = 5916.0;
    double duration = 10 * orbital_period;
    auto results = propagate_extended(t0, sv0.position, sv0.velocity, duration, 10.0);

    std::vector<double> ascending_errors, descending_errors;
    std::vector<double> perigee_errors, apogee_errors;

    for (const auto& [epoch, state] : results) {
        auto truth_opt = eof->find_nearest(epoch);
        if (!truth_opt || std::abs(truth_opt->epoch - epoch) > 60.0) continue;

        StateVector truth_state(truth_opt->position, truth_opt->velocity);
        double error = (state.position - truth_state.position).norm();

        double z_velocity = state.velocity.z();
        double radius = state.position.norm();

        if (z_velocity > 0) {
            ascending_errors.push_back(error);
        } else {
            descending_errors.push_back(error);
        }

        double mean_radius = 7.07e6;
        if (radius < mean_radius) {
            perigee_errors.push_back(error);
        } else {
            apogee_errors.push_back(error);
        }
    }

    auto compute_stats = [](const std::vector<double>& v) -> std::pair<double, double> {
        if (v.empty()) return {0.0, 0.0};
        double sum = 0.0, sum_sq = 0.0;
        for (double x : v) {
            sum += x;
            sum_sq += x * x;
        }
        double mean = sum / v.size();
        double rms = std::sqrt(sum_sq / v.size());
        return {mean, rms};
    };

    auto [asc_mean, asc_rms] = compute_stats(ascending_errors);
    auto [desc_mean, desc_rms] = compute_stats(descending_errors);
    auto [peri_mean, peri_rms] = compute_stats(perigee_errors);
    auto [apo_mean, apo_rms] = compute_stats(apogee_errors);

    std::cout << "\n";
    std::cout << "    Error by Orbital Phase (10 orbits):\n";
    std::cout << "    Phase          Count    Mean(m)    RMS(m)\n";
    std::cout << "    ------------------------------------------\n";
    std::cout << "    Ascending      " << std::setw(5) << ascending_errors.size()
              << "    " << std::fixed << std::setprecision(2) << std::setw(8) << asc_mean
              << "   " << std::setw(8) << asc_rms << "\n";
    std::cout << "    Descending     " << std::setw(5) << descending_errors.size()
              << "    " << std::setw(8) << desc_mean
              << "   " << std::setw(8) << desc_rms << "\n";
    std::cout << "    Low altitude   " << std::setw(5) << perigee_errors.size()
              << "    " << std::setw(8) << peri_mean
              << "   " << std::setw(8) << peri_rms << "\n";
    std::cout << "    High altitude  " << std::setw(5) << apogee_errors.size()
              << "    " << std::setw(8) << apo_mean
              << "   " << std::setw(8) << apo_rms << "\n";
    std::cout << "    ------------------------------------------\n";
    std::cout << "    ";
}

void test_drag_sensitivity_24h() {
    auto eof = load_eof_file();
    if (!eof) {
        TEST_SKIP("no EOF file found");
    }

    if (eof->state_vectors.size() < 8640) {
        TEST_SKIP("insufficient truth data");
    }

    const auto& sv0 = eof->state_vectors[10];
    Epoch t0 = sv0.epoch;

    double duration = 24.0 * 3600.0;

    std::cout << "\n";
    std::cout << "    Drag Coefficient Sensitivity (24 hours):\n";
    std::cout << "    Cd        RMSE(m)    Along-Trk\n";
    std::cout << "    --------------------------------\n";

    std::vector<double> cd_values = {0.0, 1.5, 2.0, 2.2, 2.5, 3.0};

    for (double cd : cd_values) {
        BallisticModel model(Sentinel1Config::MASS, Sentinel1Config::DRAG_AREA, cd);

        WeatherConditions weather;
        weather.f107_daily = Sentinel1Config::F107_DAILY;
        weather.f107_average = Sentinel1Config::F107_AVERAGE;
        weather.ap_daily = Sentinel1Config::AP_DAILY;
        weather.ap_history.fill(Sentinel1Config::AP_DAILY);

        auto prop = create_rk45_propagator(t0, sv0.position, sv0.velocity, model, weather, true);

        std::vector<std::pair<Epoch, StateVector>> results;
        for (double dt = 0; dt <= duration; dt += 60.0) {
            Epoch t = t0 + dt;
            auto result = (*prop)(t);
            results.emplace_back(t, StateVector(result.position, result.velocity));
        }

        auto test_result = compute_propagation_result(results, *eof);

        std::cout << "    " << std::fixed << std::setprecision(1) << std::setw(5) << cd
                  << "     " << std::setprecision(2) << std::setw(8) << test_result.position_rmse
                  << "    " << std::setw(8) << test_result.along_track_rms << "\n";
    }

    std::cout << "    --------------------------------\n";
    std::cout << "    Note: Along-track most sensitive to drag\n";
    std::cout << "    ";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "Test data path: " << get_test_data_path() << "\n\n";

    // Earth orientation parameters (EOP: pole position and UT1-UTC), committed file
    fs::path eop_path = fs::path(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all");
    if (time::EOPData::instance().load(eop_path)) {
        std::cout << "EOP data loaded successfully\n\n";
    } else {
        std::cout << "WARNING: Could not load EOP data\n\n";
    }

    TestRunner multiday("Multi-Day Propagation Tests");
    multiday.add_test("24_hour_propagation", test_24_hour_propagation);
    multiday.add_test("3_day_propagation", test_3_day_propagation);
    multiday.add_test("7_day_propagation", test_7_day_propagation);
    multiday.run();

    TestRunner growth("Error Growth Analysis");
    growth.add_test("error_growth_rate_24h", test_error_growth_rate_24h);
    growth.add_test("error_by_orbital_position", test_error_by_orbital_position);
    growth.run();

    TestRunner drag("Drag Sensitivity Analysis");
    drag.add_test("drag_sensitivity_24h", test_drag_sensitivity_24h);
    drag.run();

    return print_final_summary();
}
