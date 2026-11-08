// =============================================================================
// GOCE reentry env-path drag validation
// =============================================================================
//
// Checks drag configured through a DragEnvironment at very low altitude
// (140-240 km, GOCE's final decay in October-November 2013) against ESOC's
// GOCE orbits in SP3 files (SP3 = the standard precise-orbit text format),
// tests/data/sp3/goce/goce_13295.sp3 onward (year 2013, day of year 295...).
// The accuracy claim for this satellite, [`GOCE-REENTRY`], is measured by
// scripts/validate_goce_reentry.py; this C++ test is a lighter 30-minute check.
//
// GOCE:
// - mass, frontal area and drag coefficient Cd from the satellite registry
//   (include/ballistic/satellites.hpp). The ballistic coefficient
//   beta = m/(Cd*A) was fitted on GOCE's free-decay days; only beta is
//   measured, the split between A and Cd is a convention;
// - the SP3 records are in J2000 (inertial), NOT Earth-fixed (ECEF), so they
//   are converted before use;
// - SP3 satellite ID L15, 60 s cadence.
//
// Two cells: drag with NRLMSIS 2.1 density (from_models_along, constant
// weather) and drag with the density measured by GOCE's accelerometer
// (from_accelerometer_density, TU Delft file in tests/data/goce_density/).
//
// Each cell asserts only that:
//   1. the drag-on propagation completes with a finite RMSE (root-mean-square
//      position error against the SP3 orbit),
//   2. drag-on RMSE < drag-off RMSE (NRLMSIS cell: drag helps, not hurts),
//   3. the RMSE is under a loose 50 km plausibility bound for a 30-min arc
//      (far above the printed measured values; it catches blow-ups only).
// =============================================================================

#include <ballistic/ballistic.hpp>
#include "../sp3_truth.hpp"
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/coordinates/iau_transforms.hpp>
#include <ballistic/ephemeris/spk_ephemeris.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;

namespace fs = std::filesystem;

#include "../test_utils.hpp"
using namespace ballistic::test;

// =============================================================================
// Configuration — GOCE during late-mission decay (Oct 2013)
// =============================================================================

struct GOCEConfig {
    // Registry values (satellites.hpp): ESA mass, fitted beta.
    static constexpr double MASS = ballistic::satellites::get("GOCE").mass_kg;
    static constexpr double DRAG_AREA = ballistic::satellites::get("GOCE").drag_area_m2;
    static constexpr double CD = ballistic::satellites::get("GOCE").cd;
    static constexpr const char* SP3_ID = "L15";
    // October 2013: solar declining from active phase
    static constexpr double F107_DAILY = 110.0;
    static constexpr double F107_AVERAGE = 115.0;
    static constexpr double AP_DAILY = 6.0;
};

static std::shared_ptr<ephemeris::EphemerisProvider> g_ephemeris = nullptr;

// =============================================================================
// SP3 loading — handles GOCE's J2000 → ITRF conversion
// =============================================================================

std::string get_goce_data_path() {
    for (const auto& path : {
            BALLISTIC_PROJECT_ROOT "/tests/data/sp3/goce",
            "tests/data/sp3/goce",
            "../tests/data/sp3/goce"}) {
        if (fs::exists(path)) return path;
    }
    return BALLISTIC_PROJECT_ROOT "/tests/data/sp3/goce";
}

// Convert J2000 → ITRF (the Earth-fixed frame) for each record, treating J2000
// as GCRS (the IAU inertial frame). GOCE SP3 headers say "J2000 BHN ESOC";
// J2000 and GCRS differ by the frame bias (~23 milliarcseconds, under 1 m at
// LEO), well below this test's kilometer-scale bounds.
void convert_records_eci_to_itrf(ballistic::test::TruthFile& file) {
    for (auto& rec : file.records) {
        if (rec.satellite_id != GOCEConfig::SP3_ID) continue;
        StateVector st_eci{rec.position, rec.velocity};
        StateVector st_itrf = coords::gcrs_to_itrf_full(st_eci, rec.epoch,
                                                          coords::ITRFRealization::ITRF2008);
        rec.position = st_itrf.position;
        if (rec.has_velocity) rec.velocity = st_itrf.velocity;
    }
}

std::optional<ballistic::test::TruthFile> load_goce_sp3() {
    const std::string dir = get_goce_data_path();
    if (!fs::exists(dir)) return std::nullopt;
    std::vector<fs::path> sp3_paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".sp3") sp3_paths.push_back(entry.path());
    }
    std::sort(sp3_paths.begin(), sp3_paths.end());
    for (const auto& path : sp3_paths) {
        auto result = ballistic::test::load_sp3_truth(path.string());
        if (!result) continue;
        auto recs = result->get_satellite_records(GOCEConfig::SP3_ID);
        if (recs.empty()) continue;
        convert_records_eci_to_itrf(*result);
        return result;
    }
    return std::nullopt;
}

// =============================================================================
// Env-driven propagation harness
// =============================================================================

WeatherConditions goce_weather() {
    WeatherConditions w;
    w.f107_daily = GOCEConfig::F107_DAILY;
    w.f107_average = GOCEConfig::F107_AVERAGE;
    w.ap_daily = GOCEConfig::AP_DAILY;
    w.ap_history.fill(GOCEConfig::AP_DAILY);
    return w;
}

BallisticModel goce_ballistic() {
    constexpr double beta = GOCEConfig::MASS /
                           (GOCEConfig::CD * GOCEConfig::DRAG_AREA);
    return BallisticModel(beta, GOCEConfig::MASS, GOCEConfig::DRAG_AREA);
}

std::vector<std::pair<Epoch, StateVector>> propagate_goce(
    const Epoch& t0, const Vector3d& pos0, const Vector3d& vel0,
    double duration_s, double sample_interval, bool enable_drag)
{
    auto weather = goce_weather();
    auto model = goce_ballistic();
    // GOCE 2013 → ITRF2008 (the realization current in late 2013; the
    // difference to neighboring realizations is mm-level at this scale).
    auto prop = create_gcrs_propagator(t0, pos0, vel0, model, weather, true,
                                        g_ephemeris,
                                        coords::ITRFRealization::ITRF2008);

    if (enable_drag) {
        std::vector<Epoch> sample_epochs;
        std::vector<Vector3d> sample_positions;
        constexpr double dt_env = 60.0;
        const int n_env = static_cast<int>(duration_s / dt_env) + 1;
        sample_epochs.reserve(n_env + 1);
        sample_positions.reserve(n_env + 1);

        auto nodrag_prop = create_gcrs_propagator(
            t0, pos0, vel0, model, weather, true, g_ephemeris,
            coords::ITRFRealization::ITRF2008);
        for (int i = 0; i <= n_env; ++i) {
            const double dt = std::min(double(i) * dt_env, duration_s);
            const Epoch t = t0 + dt;
            auto r = (*nodrag_prop)(t);
            sample_epochs.push_back(t);
            sample_positions.push_back(r.position);
        }

        auto sw = atmosphere::SpaceWeatherHistory::from_constant(
            weather, atmosphere::TimeWindow{t0 - 86400.0,
                                              t0 + duration_s + 86400.0});
        constexpr double beta_kg_m2 = GOCEConfig::MASS /
                                       (GOCEConfig::CD * GOCEConfig::DRAG_AREA);
        auto beta_prior = atmosphere::BetaProfile1D::constant(beta_kg_m2);
        // GOCE reentry altitude grid (140-300 km in 20 km steps).
        std::vector<double> alt_grid;
        for (int alt = 140; alt <= 300; alt += 20) alt_grid.push_back(double(alt));
        auto env = std::make_shared<const atmosphere::DragEnvironment>(
            atmosphere::from_models_along(
                sample_epochs, sample_positions, sw,
                std::move(beta_prior), std::move(alt_grid)));
        prop->force_model().set_drag_environment(env);
    } else {
        auto flags = prop->force_model().force_flags();
        flags.drag = false;
        prop->force_model().set_force_flags(flags);
    }

    std::vector<std::pair<Epoch, StateVector>> results;
    const int n_samples = static_cast<int>(duration_s / sample_interval) + 1;
    for (int i = 0; i <= n_samples; ++i) {
        const double dt = std::min(double(i) * sample_interval, duration_s);
        const Epoch t = t0 + dt;
        auto r = (*prop)(t);
        results.emplace_back(t, StateVector{r.position, r.velocity});
    }
    return results;
}

double rmse_vs_sp3(const std::vector<std::pair<Epoch, StateVector>>& predicted,
                   const ballistic::test::TruthFile& truth)
{
    auto truth_records = truth.get_satellite_records(GOCEConfig::SP3_ID);
    if (truth_records.empty() || predicted.empty()) return -1.0;

    double sum_sq = 0.0;
    int count = 0;
    for (const auto& [t_pred, state_pred] : predicted) {
        const ballistic::test::TruthRecord* best = nullptr;
        double best_dt = 5.0;
        for (const auto& rec : truth_records) {
            const double dt = std::abs(rec.epoch - t_pred);
            if (dt < best_dt) { best_dt = dt; best = &rec; }
        }
        if (!best) continue;
        const double err_sq = (state_pred.position - best->position).squaredNorm();
        sum_sq += err_sq;
        ++count;
    }
    return (count > 0) ? std::sqrt(sum_sq / count) : -1.0;
}

// =============================================================================
// TU Delft GOCE density file parser (v02 format) — inline for the
// accelerometer-density cell
// =============================================================================
// Each data line is:
//   "YYYY-MM-DD HH:MM:SS.sss TIME_SYS alt_m lat lon lst arglat density …"
// Comment lines start with '#'. Density at position 7 after the datetime
// (column index 7 in the post-datetime whitespace split: 0=time_sys,
// 1=alt, 2=lat, 3=lon, 4=lst, 5=arglat, 6=density, 7+=optional fields).
// Implementation note: parse_density_table in python/ballistic/loaders.py
// handles the same format; this is a minimal C++ port so the test
// doesn't have to cross the Python boundary.
struct GoceDensitySample {
    Epoch epoch;
    double rho_kg_m3;
};
std::vector<GoceDensitySample> parse_goce_density(
    const fs::path& file_path,
    const Epoch& t_min, const Epoch& t_max)
{
    std::vector<GoceDensitySample> result;
    std::ifstream in(file_path);
    if (!in) return result;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        // First 23 chars: "YYYY-MM-DD HH:MM:SS.sss"
        if (line.size() < 27 || line[4] != '-' || line[7] != '-') continue;
        int y = std::stoi(line.substr(0, 4));
        int mo = std::stoi(line.substr(5, 2));
        int d = std::stoi(line.substr(8, 2));
        int h = std::stoi(line.substr(11, 2));
        int mi = std::stoi(line.substr(14, 2));
        double s = std::stod(line.substr(17, 6));
        Epoch t = Epoch::from_utc(y, mo, d, h, mi, s);
        if (t < t_min || t > t_max) continue;
        // Rest of line: whitespace-split, density at index 6.
        std::istringstream iss(line.substr(23));
        std::string time_sys; double alt, lat, lon, lst, arglat, rho;
        if (!(iss >> time_sys >> alt >> lat >> lon >> lst >> arglat >> rho))
            continue;
        if (rho <= 0.0 || rho >= 1.0) continue;
        result.push_back({t, rho});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.epoch < b.epoch; });
    return result;
}

// =============================================================================
// Test cases
// =============================================================================

void test_30min_drag_on_vs_off() {
    auto sp3 = load_goce_sp3();
    if (!sp3) { TEST_SKIP("no SP3"); }
    auto recs = sp3->get_satellite_records(GOCEConfig::SP3_ID);
    if (recs.size() < 50) { TEST_SKIP("insufficient records"); }
    const auto& sv0 = recs[5];
    if (!sv0.has_velocity) { TEST_SKIP("no velocity"); }

    // 30-minute arc; reentry dynamics are stiff so a shorter window
    // keeps the drag-on result inside reasonable assertion bounds.
    constexpr double dur = 1800.0;

    auto pred_on = propagate_goce(sv0.epoch, sv0.position, sv0.velocity,
                                    dur, 60.0, /*enable_drag=*/true);
    auto pred_off = propagate_goce(sv0.epoch, sv0.position, sv0.velocity,
                                     dur, 60.0, /*enable_drag=*/false);

    const double rmse_on = rmse_vs_sp3(pred_on, *sp3);
    const double rmse_off = rmse_vs_sp3(pred_off, *sp3);

    const double r0 = sv0.position.norm() / 1000.0;
    std::cout << "\n    GOCE start altitude: " << std::fixed << std::setprecision(1)
              << (r0 - 6378.137) << " km\n";
    std::cout << "    30-min drag-on  RMSE: " << std::setprecision(2)
              << rmse_on << " m\n";
    std::cout << "    30-min drag-off RMSE: " << rmse_off << " m\n";
    std::cout << "    drag improvement:     " << (rmse_off - rmse_on)
              << " m (" << std::setprecision(1)
              << (100.0 * (rmse_off - rmse_on) / std::max(rmse_off, 1.0))
              << "%)\n    ";

    TEST_ASSERT_MSG(rmse_on >= 0.0 && rmse_off >= 0.0,
                    "RMSE compute failed");
    // Loose plausibility bound (50 km) for a 30-min arc at ~250 km; it only
    // catches a blow-up. The measured value is printed above.
    TEST_ASSERT_MSG(rmse_on < 50000.0,
                    "drag-on 30-min RMSE plausible upper bound");
    // Sanity: drag-on must improve on drag-off at GOCE altitude.
    TEST_ASSERT_MSG(rmse_on < rmse_off,
                    "drag-on must outperform drag-off at GOCE altitude");
}

// Use the accelerometer-derived density (TU Delft) as the environment's
// density column instead of NRLMSIS 2.1. Asserts only a finite RMSE under the
// same 50 km plausibility bound.
void test_30min_accelerometer_density_path() {
    auto sp3 = load_goce_sp3();
    if (!sp3) { TEST_SKIP("no SP3"); }
    auto recs = sp3->get_satellite_records(GOCEConfig::SP3_ID);
    if (recs.size() < 50) { TEST_SKIP("insufficient records"); }
    const auto& sv0 = recs[5];
    if (!sv0.has_velocity) { TEST_SKIP("no velocity"); }

    constexpr double dur = 1800.0;

    // Load TU Delft GOCE accelerometer-derived density spanning the
    // test arc. The "deorb-DFACS" file covers 2013-10-22 to 2013-11-10
    // (GOCE's final reentry weeks). That matches the SP3 fixture
    // (goce_13295.sp3 = day 295 of 2013 = 2013-10-22).
    fs::path density_file;
    for (const auto& p : {
            fs::path(BALLISTIC_PROJECT_ROOT)
                / "tests/data/goce_density/goce_denswind_ac082_v2_0_deorb-DFACS.txt",
            fs::path("tests/data/goce_density/goce_denswind_ac082_v2_0_deorb-DFACS.txt")}) {
        if (fs::exists(p)) { density_file = p; break; }
    }
    if (density_file.empty()) {
        TEST_SKIP("no GOCE density file");
    }

    // Pad ±15 min so the linear-time-interp inside DensityProfile2D
    // has anchor points on both sides of the arc.
    const auto samples = parse_goce_density(
        density_file, sv0.epoch - 900.0, sv0.epoch + dur + 900.0);
    if (samples.size() < 2) {
        TEST_SKIP("density file has no overlap with SP3 arc");
    }

    std::vector<Epoch> epochs;
    std::vector<double> rhos;
    epochs.reserve(samples.size());
    rhos.reserve(samples.size());
    for (const auto& s : samples) {
        epochs.push_back(s.epoch);
        rhos.push_back(s.rho_kg_m3);
    }

    constexpr double beta_kg_m2 = GOCEConfig::MASS /
                                   (GOCEConfig::CD * GOCEConfig::DRAG_AREA);
    std::vector<double> alt_grid;
    for (int alt = 140; alt <= 300; alt += 20) alt_grid.push_back(double(alt));
    auto env = std::make_shared<const atmosphere::DragEnvironment>(
        atmosphere::from_accelerometer_density(
            epochs, rhos,
            atmosphere::BetaProfile1D::constant(beta_kg_m2),
            std::move(alt_grid)));

    // Propagate with the accel-density env attached.
    auto weather = goce_weather();
    auto model = goce_ballistic();
    auto prop = create_gcrs_propagator(
        sv0.epoch, sv0.position, sv0.velocity, model, weather, true,
        g_ephemeris, coords::ITRFRealization::ITRF2008);
    prop->force_model().set_drag_environment(env);

    std::vector<std::pair<Epoch, StateVector>> pred;
    const int n_samples = static_cast<int>(dur / 60.0) + 1;
    for (int i = 0; i <= n_samples; ++i) {
        const double dt = std::min(double(i) * 60.0, dur);
        const Epoch t = sv0.epoch + dt;
        auto r = (*prop)(t);
        pred.emplace_back(t, StateVector{r.position, r.velocity});
    }

    const double rmse = rmse_vs_sp3(pred, *sp3);
    std::cout << "\n    accel-density path samples: " << samples.size() << "\n";
    std::cout << "    30-min accel-density RMSE: " << std::fixed
              << std::setprecision(2) << rmse << " m\n    ";
    TEST_ASSERT_MSG(rmse >= 0.0, "RMSE compute failed");
    TEST_ASSERT_MSG(rmse < 50000.0,
                    "accel-density 30-min RMSE plausible upper bound");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    initialize({});

    for (const auto& path : {
            BALLISTIC_PROJECT_ROOT "/data/de440.bsp",
            BALLISTIC_PROJECT_ROOT "/data/de421.bsp"}) {
        if (fs::exists(path)) {
            try {
                auto spk = std::make_shared<ephemeris::SPKEphemeris>();
                if (spk->load(path)) {
                    g_ephemeris = spk;
                    std::cout << "  Loaded ephemeris: " << path << "\n";
                    break;
                }
            } catch (...) {}
        }
    }

    TestRunner runner("GOCE reentry env-path drag validation");
    runner.add_test("nrlmsis_drag_30min_on_vs_off", test_30min_drag_on_vs_off);
    runner.add_test("accelerometer_density_drag_30min",

                     test_30min_accelerometer_density_path);
    runner.run();
    return print_final_summary();
}
