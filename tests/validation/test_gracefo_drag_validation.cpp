// =============================================================================
// GRACE-FO env-path drag validation
// =============================================================================
//
// Propagates GRACE-FO-1 for one hour from a state in a GFZ precise orbit file
// (SP3 = the standard precise-orbit format) and compares the prediction with
// that orbit. Drag comes from a DragEnvironment: NRLMSIS 2.1 density tabulated
// by `from_models_along` along a drag-free nominal arc, with constant weather.
// Asserted: drag-on 1-hour RMSE (root-mean-square position error) < 50 m;
// drag-off < 200 m. Both are loose regression bounds; the campaign-level
// figure (0.47-0.87 m with drag, measured weather) is [`GRACEFO-500KM`] in
// docs/validation_claims.md (Python script).
//
// Terms: beta = m / (Cd·A), the ballistic coefficient; GCRS = the inertial
// frame the propagator integrates in; ITRF2020 = the Earth-fixed frame of the
// SP3 positions.
//
// GRACE-FO characteristics:
// - Altitude: ~500 km (twin-satellite gravity-field mission)
// - Mass, frontal area and Cd: the registry (include/ballistic/satellites.hpp):
//   physical mass (Level-1B), handbook area, Sentman Cd.
// - SP3 satellite ID: "L64" (GRACE-FO-1) — GFZ RSO format
//
// Data source: GFZ ISDC GRACE-FO RSO v03 (2025+; ITRF2020).
// Test fixture: tests/data/sp3/grace_fo/ holds overlapping half-day SP3 files
// for GRACE-FO-1 (L64) and GRACE-FO-2 (L65); this test uses the first L64 file.
// Span: 2025-12-31 to 2026-02-08.  guard-exempt(PUB-DATE): data
// =============================================================================

#include <ballistic/ballistic.hpp>
#include "../sp3_truth.hpp"
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <ballistic/ephemeris/spk_ephemeris.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ballistic;
using namespace ballistic::validation;
using namespace ballistic::propagators;

namespace fs = std::filesystem;

#include "../test_utils.hpp"
using namespace ballistic::test;

// =============================================================================
// Configuration
// =============================================================================

struct GRACEFOConfig {
    // Registry values (include/ballistic/satellites.hpp): physical mass, area,
    // and Cd from the Sentman free-molecular model on the panel geometry.
    static constexpr double MASS = ballistic::satellites::get("GRACE-FO-1").mass_kg;
    static constexpr double DRAG_AREA = ballistic::satellites::get("GRACE-FO-1").drag_area_m2;
    static constexpr double CD = ballistic::satellites::get("GRACE-FO-1").cd;
    static constexpr const char* SP3_ID = "L64";     // GRACE-FO-1
    // Round constant weather (F10.7 = 10.7 cm solar radio flux in sfu; Ap =
    // daily geomagnetic index), lower than on the test days: the committed
    // weather file gives F10.7 166-171, 81-day mean ~154, Ap 7-23 for
    // 31 Dec - 2 Jan. The solar cycle was past its maximum and declining.
    static constexpr double F107_DAILY = 130.0;
    static constexpr double F107_AVERAGE = 135.0;
    static constexpr double AP_DAILY = 7.0;
};

static std::shared_ptr<ephemeris::EphemerisProvider> g_ephemeris = nullptr;

// =============================================================================
// SP3 loading + velocity derivation
// =============================================================================

std::string get_gracefo_data_path() {
    for (const auto& path : {
            BALLISTIC_PROJECT_ROOT "/tests/data/sp3/grace_fo",
            "tests/data/sp3/grace_fo",
            "../tests/data/sp3/grace_fo"}) {
        if (fs::exists(path)) return path;
    }
    return BALLISTIC_PROJECT_ROOT "/tests/data/sp3/grace_fo";
}

std::optional<ballistic::test::TruthFile> load_gracefo_sp3() {
    const std::string dir = get_gracefo_data_path();
    if (!fs::exists(dir)) return std::nullopt;
    // Process the SP3 files in sorted name order and use the first one that
    // holds GRACE-FO-1 records, so the choice is deterministic. (The earliest
    // file, 2025-12-31 22:00 UTC, is not skipped.)  guard-exempt(PUB-DATE): data
    std::vector<fs::path> sp3_paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".sp3") sp3_paths.push_back(entry.path());
    }
    std::sort(sp3_paths.begin(), sp3_paths.end());
    for (const auto& path : sp3_paths) {
        auto result = ballistic::test::load_sp3_truth(path.string());
        if (!result) continue;
        auto recs = result->get_satellite_records(GRACEFOConfig::SP3_ID);
        if (recs.empty()) continue;

        // Derive velocities by central difference if SP3 has only positions.
        if (!recs[0].has_velocity) {
            std::vector<std::size_t> p_idx;
            for (std::size_t i = 0; i < result->records.size(); ++i) {
                if (result->records[i].satellite_id == GRACEFOConfig::SP3_ID &&
                    !result->records[i].has_velocity) {
                    p_idx.push_back(i);
                }
            }
            for (std::size_t j = 0; j < p_idx.size(); ++j) {
                const std::size_t i = p_idx[j];
                const std::size_t prev = (j == 0) ? 0 : j - 1;
                const std::size_t next = (j == p_idx.size() - 1) ? j : j + 1;
                const double dt = result->records[p_idx[next]].epoch
                                 - result->records[p_idx[prev]].epoch;
                if (dt > 1.0) {
                    result->records[i].velocity =
                        (result->records[p_idx[next]].position -
                         result->records[p_idx[prev]].position) / dt;
                    result->records[i].has_velocity = true;
                }
            }
        }
        return result;
    }
    return std::nullopt;
}

// =============================================================================
// Env-driven propagation harness
// =============================================================================

WeatherConditions gracefo_weather() {
    WeatherConditions w;
    w.f107_daily = GRACEFOConfig::F107_DAILY;
    w.f107_average = GRACEFOConfig::F107_AVERAGE;
    w.ap_daily = GRACEFOConfig::AP_DAILY;
    w.ap_history.fill(GRACEFOConfig::AP_DAILY);
    return w;
}

BallisticModel gracefo_ballistic() {
    constexpr double beta = GRACEFOConfig::MASS /
                           (GRACEFOConfig::CD * GRACEFOConfig::DRAG_AREA);
    return BallisticModel(beta, GRACEFOConfig::MASS, GRACEFOConfig::DRAG_AREA);
}

// Propagate a GRACE-FO arc, with or without drag, sampled every
// `sample_interval` seconds. The start state is ITRF2020 (the frame of GFZ RSO
// v03 orbits).
std::vector<std::pair<Epoch, StateVector>> propagate_gracefo(
    const Epoch& t0, const Vector3d& pos0, const Vector3d& vel0,
    double duration_s, double sample_interval, bool enable_drag)
{
    auto weather = gracefo_weather();
    auto model = gracefo_ballistic();
    auto prop = create_gcrs_propagator(t0, pos0, vel0, model, weather, true,
                                        g_ephemeris,
                                        coords::ITRFRealization::ITRF2020);

    if (enable_drag) {
        // Drag-free nominal samples for from_models_along.
        std::vector<Epoch> sample_epochs;
        std::vector<Vector3d> sample_positions;
        constexpr double dt_env = 60.0;
        const int n_env = static_cast<int>(duration_s / dt_env) + 1;
        sample_epochs.reserve(n_env + 1);
        sample_positions.reserve(n_env + 1);

        auto nodrag_prop = create_gcrs_propagator(
            t0, pos0, vel0, model, weather, true, g_ephemeris,
            coords::ITRFRealization::ITRF2020);
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
        constexpr double beta_kg_m2 = GRACEFOConfig::MASS /
                                       (GRACEFOConfig::CD * GRACEFOConfig::DRAG_AREA);
        auto beta_prior = atmosphere::BetaProfile1D::constant(beta_kg_m2);
        // Altitude grid limited to GRACE-FO's band (450-550 km, 10 km steps).
        std::vector<double> alt_grid;
        for (int alt = 450; alt <= 550; alt += 10) alt_grid.push_back(double(alt));
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

// Position RMSE against the SP3 records. The orbit file has a record every
// 30 s and predictions are made every 30 s; a prediction is scored only
// against a record within 5 s, i.e. the record at the same time, so no
// interpolation or time offset enters the error.
double rmse_vs_sp3(const std::vector<std::pair<Epoch, StateVector>>& predicted,
                   const ballistic::test::TruthFile& truth)
{
    auto truth_records = truth.get_satellite_records(GRACEFOConfig::SP3_ID);
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
// Test cases
// =============================================================================

void test_1_hour_drag_on() {
    auto sp3 = load_gracefo_sp3();
    if (!sp3) { TEST_SKIP("no SP3"); }
    auto recs = sp3->get_satellite_records(GRACEFOConfig::SP3_ID);
    if (recs.size() < 400) { TEST_SKIP("insufficient records"); }

    const auto& sv0 = recs[10];
    if (!sv0.has_velocity) { TEST_SKIP("no velocity"); }

    auto pred = propagate_gracefo(sv0.epoch, sv0.position, sv0.velocity,
                                    3600.0, 30.0, /*enable_drag=*/true);
    const double rmse = rmse_vs_sp3(pred, *sp3);

    std::cout << "\n    1-hour drag-on RMSE: " << std::fixed << std::setprecision(2)
              << rmse << " m\n    ";

    TEST_ASSERT_MSG(rmse > 0.0, "RMSE compute failed");
    // The campaign measures sub-meter to meter 1-hour errors at 500 km with
    // drag ([`GRACEFO-500KM`]). This cell uses constant weather instead of the
    // measured history, so a few meters or more is possible; 50 m is generous
    // headroom (the same bound as the CHAMP test) that still catches a broken
    // drag path.
    TEST_ASSERT_MSG(rmse < 50.0,
        "GRACE-FO 1-hour RMSE should be <50m via the env drag path");
}

void test_1_hour_drag_off_baseline() {
    auto sp3 = load_gracefo_sp3();
    if (!sp3) { TEST_SKIP("no SP3"); }
    auto recs = sp3->get_satellite_records(GRACEFOConfig::SP3_ID);
    if (recs.size() < 400) { TEST_SKIP("insufficient records"); }
    const auto& sv0 = recs[10];
    if (!sv0.has_velocity) { TEST_SKIP("no velocity"); }

    auto pred = propagate_gracefo(sv0.epoch, sv0.position, sv0.velocity,
                                    3600.0, 30.0, /*enable_drag=*/false);
    const double rmse = rmse_vs_sp3(pred, *sp3);

    std::cout << "\n    1-hour drag-off RMSE: " << std::fixed << std::setprecision(2)
              << rmse << " m\n    ";
    TEST_ASSERT_MSG(rmse >= 0.0, "RMSE compute failed");
    // Without drag the 1-hour error is several times larger than with it at
    // 500 km (the campaign measures drag cutting it by 80-88%,
    // [`GRACEFO-500KM`]). The generous 200 m bound only checks that the
    // drag-free run completes and is scored on the same records.
    TEST_ASSERT_MSG(rmse < 200.0, "drag-off 1-hour RMSE plausible bound");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    initialize({});

    // Use a JPL ephemeris file from data/ if present (DE440, DE441, DE421 in
    // that order); otherwise the library's embedded DE440.
    for (const auto& path : {
            BALLISTIC_PROJECT_ROOT "/data/de440.bsp",
            BALLISTIC_PROJECT_ROOT "/data/de441.bsp",
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

    TestRunner runner("GRACE-FO env-path drag validation");

    runner.add_test("1_hour_drag_on", test_1_hour_drag_on);
    runner.add_test("1_hour_drag_off_baseline", test_1_hour_drag_off_baseline);
    runner.run();
    return print_final_summary();
}
