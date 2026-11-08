// =============================================================================
// Tutorial example 07 — Multi-sensor EA estimation (C++)
// =============================================================================
//
// Two ground radars (from the glide fixture) track the forced glide. We
// synthesize clean [range, az, el, range-rate] observations from the truth
// trajectory, fuse them with estimate_ea_smoother (the recommended fusion
// path — a GM-RTS reduced-dynamic smoother: EKF forward pass + RTS backward
// pass over a Gauss-Markov EA state, auto NIS hyperparameters, measured
// range-rate budget), then re-propagate with the recovered EA and RMSE
// against truth. Two radars (vs one) avoids the single-radar+rate-on
// observability degeneracy documented in the multi-sensor reference.
//
// Case 2 adds a 50 µrad astrometric camera (OpticsObs, az/el only)
// co-located with radar 0 — the documented way to tighten the radar-only
// information bound (~9-17 m for this geometry) by an order of magnitude.
//
// CAVEAT (teaching point): radar -> EA is observability-limited, so the
// radar-only reprop sits at the geometry floor, not estimator error.
// `nis` ~ 1 is the truth-free health check that the noise model is
// consistent. Prefer 2+ radars, and add a camera/fix to tighten the bound.
//
//   ex07_multisensor [out.csv]

#include "glide_common.hpp"
#include <ballistic/calibration/ea_smoother.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ballistic;
using namespace ballistic::calibration;

namespace {
RadarObs make_radar(const Vector3d& site) {
    RadarObs r;
    r.position_ecef = site;
    r.sigma_range = 2.0;
    r.sigma_angle = 0.02 * constants::DEG_TO_RAD;
    r.sigma_rate = 0.02;
    auto g = coords::ecef_to_geodetic(site);
    r.lat_rad = g.latitude;
    r.lon_rad = g.longitude;
    r.ecef_to_enu = coords::ecef_to_enu_rotation(g.latitude, g.longitude);
    return r;
}

OpticsObs make_camera(const Vector3d& site) {
    OpticsObs c;                       // sigma_angle default: 50e-6 rad
    c.position_ecef = site;
    auto g = coords::ecef_to_geodetic(site);
    c.lat_rad = g.latitude;
    c.lon_rad = g.longitude;
    c.ecef_to_enu = coords::ecef_to_enu_rotation(g.latitude, g.longitude);
    return c;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string out =
        argc > 1 ? argv[1] : "docs/latex/results/multisensor_rmse.csv";
    ballistic::initialize({});

    tut::GlideTruth truth(/*with_ea=*/true);

    RadarObs r0 = make_radar(tut::gf::RADAR0_ECEF);
    RadarObs r1 = make_radar(tut::gf::RADAR1_ECEF);
    OpticsObs cam = make_camera(tut::gf::RADAR0_ECEF);  // co-located telescope

    // Synthesize clean observations from the truth trajectory.
    for (double t = 0.0; t <= truth.eval_end; t += 0.5) {
        auto st = truth.ecef_i->get_state(truth.t0 + t);
        if (!st) continue;
        StateVector sv{st->position, st->velocity};
        r0.samples.emplace_back(truth.t0 + t, r0.predict(sv));
        r1.samples.emplace_back(truth.t0 + t, r1.predict(sv));
        cam.samples.emplace_back(truth.t0 + t, cam.predict(sv));
    }

    MultiSensorScenario scenario;
    scenario.epoch0 = truth.t0;
    scenario.pos0_ecef = truth.pos;
    scenario.vel0_ecef = truth.vel;
    scenario.arc_duration_s = truth.eval_end;
    scenario.add_radar(r0).add_radar(r1);

    struct Row { std::string label; double nis; double rmse; };
    std::vector<Row> rows;

    auto run_case = [&](const std::string& label) -> bool {
        auto mask = SensorMask::all_on(scenario);
        EASmootherConfig cfg;  // minimal by design: NIS-auto prior + rate budget
        auto result = estimate_ea_smoother(scenario, mask, tut::base_config(), cfg);
        if (!result.has_correction()) {
            std::cerr << label << ": smoother failed: " << result.error_message << "\n";
            return false;
        }
        double rmse = tut::reprop_rmse(truth, result.empirical_accel);
        rows.push_back({label, result.nis, rmse});
        std::cout << std::fixed << std::setprecision(3)
                  << label << " (clean, 2 Hz, " << r0.samples.size() << " obs/sensor)\n"
                  << "  NIS (1=ideal, <1 cons.): " << result.nis << "\n"
                  << "  reprop RMSE vs truth   : " << rmse << " m\n";
        return true;
    };

    if (!run_case("2xradar")) return 1;
    scenario.add_optics(cam);          // + 50 µrad camera: tightens the bound
    if (!run_case("2xradar+camera")) return 1;

    std::ofstream f(out);
    f << "sensors,obs_hz,n_obs_per_sensor,nis,reprop_rmse_m\n"
      << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
        f << row.label << ",2," << r0.samples.size()
          << "," << row.nis << "," << row.rmse << "\n";
    std::cout << "wrote " << out << "\n";
    return 0;
}
