// Example: Sentinel satellite prediction and validation
//
// This example demonstrates:
// - Setting up a propagator for a Sentinel-1A-like orbit
// - Propagating 1 hour ahead
// - Reading positions along the way
// - How to compare against a precise orbit file (SP3, the standard text
//   format for precise satellite orbits), shown here on synthetic data
//
// To check accuracy for real, compare predictions against ESA's precise
// orbit determination (POD) products for Sentinel, published through the
// Copernicus data services; precise orbit files give positions accurate to
// centimeters. Units throughout: meters, meters/second, seconds.

#include <ballistic/ballistic.hpp>
#include <ballistic/validation/sp3.hpp>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace ballistic;

// Sentinel-1A orbit; mass, frontal area and Cd come from the library's
// registry of cited satellite properties (include/ballistic/satellites.hpp).
struct SentinelParams {
    double altitude = 693000.0;     // ~693 km nominal altitude
    double inclination = 98.18;     // Sun-synchronous, degrees
    double mass = ballistic::satellites::get("S1A").mass_kg;
    double area = ballistic::satellites::get("S1A").drag_area_m2;
    double cd = ballistic::satellites::get("S1A").cd;
};

int main() {
    std::cout << "=== Sentinel Satellite Prediction Example ===\n\n";

    // The gravity field and the Sun/Moon/planet positions (ephemeris) are
    // built in; the Earth orientation parameters (measured irregularities of
    // the Earth's rotation) come from the data directory (Python:
    // ballistic.load() fills it once).
    if (!ballistic::initialize({}).eop_loaded) {
        std::cerr << "No Earth orientation data: put finals2000A.all in " << get_data_directory()
                  << " (Python: ballistic.load()) and run again.\n";
        return 1;
    }

    SentinelParams params;

    // An illustrative start state for a Sentinel-1A-like orbit. In practice
    // it would come from a two-line element set (TLE) or a precise orbit file.
    Epoch t0 = Epoch::from_utc(2024, 6, 15, 12, 0, 0);

    // Circular orbit at the nominal altitude (Sun-synchronous inclination)
    double r = constants::earth::RADIUS_EQUATORIAL + params.altitude;
    double v_circular = std::sqrt(constants::earth::GM / r);

    // Simplified: start on the equator, moving north (ascending node)
    double inc_rad = params.inclination * constants::DEG_TO_RAD;
    // Circular speed and inclination are INERTIAL quantities: build the state
    // in GCRS (the non-rotating frame) and convert to the Earth-fixed ITRF
    // frame the propagator takes.
    const StateVector itrf = coords::gcrs_to_itrf_full(
        StateVector(Vector3d(r, 0.0, 0.0),
                    Vector3d(0.0, v_circular * std::cos(inc_rad), v_circular * std::sin(inc_rad))),
        t0);
    Vector3d pos_ecef = itrf.position;
    Vector3d vel_ecef = itrf.velocity;

    std::cout << "Sentinel-1A Configuration:\n";
    std::cout << "  Altitude: " << params.altitude / 1000.0 << " km\n";
    std::cout << "  Inclination: " << params.inclination << " deg\n";
    std::cout << "  Mass: " << params.mass << " kg\n";
    std::cout << "  Drag area: " << params.area << " m²\n";
    std::cout << "\nInitial state (ECEF):\n";
    std::cout << "  Position: [" << pos_ecef.transpose() / 1000.0 << "] km\n";
    std::cout << "  Velocity: [" << vel_ecef.transpose() << "] m/s\n\n";

    // Mass, drag area and drag coefficient of the satellite
    BallisticModel ballistic(params.mass, params.area, params.cd);

    // Fixed, moderate space weather: F10.7 is the solar radio flux index and
    // Ap the geomagnetic index, the inputs the atmosphere density model needs
    WeatherConditions weather;
    weather.f107_daily = 150.0;
    weather.f107_average = 150.0;
    weather.ap_daily = 15.0;

    // Create the propagator (adaptive Runge-Kutta 4(5), "RK45")
    auto prop = propagators::create_rk45_propagator(
        t0, pos_ecef, vel_ecef,
        ballistic, weather,
        true);  // Include Sun/Moon gravity

    // Turn on gravity, the Sun and Moon, and drag
    {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.earth_gravity = true;
        flags.third_body = true;
        flags.drag = true;
        prop->force_model().set_force_flags(flags);
    }
    prop->force_model().set_gravity_max_degree(6);  // coarse gravity field (degree 6), for speed

    // Propagate for 1 hour, printing every 5 minutes
    std::cout << "Propagating for 1 hour...\n\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Time(s) | Lat(deg) | Lon(deg) | Alt(km) | Speed(m/s)\n";
    std::cout << "--------|----------|----------|---------|----------\n";

    std::vector<PropagationResult> trajectory;

    for (int i = 0; i <= 60; i += 5) {  // Every 5 minutes
        Epoch t = t0 + i * 60.0;
        PropagationResult result = (*prop)(t);
        trajectory.push_back(result);

        // Convert to geodetic
        GeodeticCoord geo = coords::ecef_to_geodetic(result.position);

        std::cout << std::setw(7) << i * 60
                  << " | " << std::setw(8) << geo.latitude * constants::RAD_TO_DEG
                  << " | " << std::setw(8) << geo.longitude * constants::RAD_TO_DEG
                  << " | " << std::setw(7) << geo.altitude / 1000.0
                  << " | " << std::setw(9) << result.velocity.norm()
                  << "\n";
    }

    // Final state after 1 hour
    Epoch t_final = t0 + 3600.0;
    PropagationResult final_state = (*prop)(t_final);

    std::cout << "\nFinal state after 1 hour:\n";
    std::cout << "  Position (ECEF): [" << final_state.position.transpose() / 1000.0 << "] km\n";
    std::cout << "  Velocity (ECEF): [" << final_state.velocity.transpose() << "] m/s\n";
    std::cout << "  Acceleration: [" << final_state.acceleration.transpose() * 1000 << "] mm/s²\n";

    GeodeticCoord final_geo = coords::ecef_to_geodetic(final_state.position);
    std::cout << "  Geodetic: lat=" << final_geo.latitude * constants::RAD_TO_DEG
              << "°, lon=" << final_geo.longitude * constants::RAD_TO_DEG
              << "°, alt=" << final_geo.altitude / 1000.0 << " km\n";

    // Show force breakdown at final state
    StateVector state_gcrs = prop->current_state_gcrs();
    Vector3d acc_grav = prop->force_model().earth_gravity_acceleration(t_final, state_gcrs);
    Vector3d acc_3body = prop->force_model().third_body_acceleration(t_final, state_gcrs);
    Vector3d acc_drag = prop->force_model().drag_acceleration(t_final, state_gcrs);

    std::cout << "\nForce contributions at final state (GCRS):\n";
    std::cout << "  Earth gravity: " << acc_grav.norm() * 1000 << " mm/s² ("
              << acc_grav.norm() / 9.81 << " g)\n";
    std::cout << "  Third-body:    " << acc_3body.norm() * 1e9 << " nm/s²\n";
    std::cout << "  Drag:          " << acc_drag.norm() * 1e9 << " nm/s²\n";

    // Statistics
    auto stats = prop->stats();
    std::cout << "\nIntegration statistics:\n";
    std::cout << "  Steps taken: " << stats.steps_taken << "\n";
    std::cout << "  Function evaluations: " << stats.function_evaluations << "\n";

    std::cout << "\n=== SP3 Validation Example ===\n\n";

    // How reading an SP3 file and comparing against it looks (printed as
    // instructions; the RMSE below is computed on synthetic data)

    std::cout << "To validate against SP3 precise orbit data:\n\n";
    std::cout << "1. Download SP3 file from ESA Sentinel POD service\n";
    std::cout << "2. Read the file (the file's time system is honored):\n";
    std::cout << "   auto f = validation::sp3::read(\"sentinel_pod.sp3\");\n";
    std::cout << "   const auto& arc = f.arc(\"L74\");   // one satellite in one file\n\n";

    std::cout << "3. Compare predictions with the truth at each record:\n";
    std::cout << "   for (std::size_t i = 0; i < arc.size(); ++i) {\n";
    std::cout << "       PropagationResult pred = (*prop)(arc.epoch(i));\n";
    std::cout << "       err2 += (pred.position - arc.position_m[i]).squaredNorm();\n";
    std::cout << "   }\n";
    std::cout << "   (arc.interpolate(t) gives positions between records; it refuses\n";
    std::cout << "    times outside the file or inside a data gap.)\n\n";

    // Demonstrate with synthetic data
    std::cout << "Demonstrating RMSE calculation with synthetic data...\n";

    std::vector<std::pair<Epoch, Vector3d>> predicted_traj;
    std::vector<std::pair<Epoch, Vector3d>> synthetic_truth;

    // Synthetic "truth": the prediction plus a small known error
    for (int i = 0; i <= 12; ++i) {
        Epoch t = t0 + i * 300.0;  // Every 5 minutes
        PropagationResult pred = (*prop)(t);

        predicted_traj.push_back({t, pred.position});

        // A smooth offset of a few meters stands in for the prediction error
        double phase = i * 0.7;
        Vector3d truth_pos = pred.position + Vector3d(
            5.0 * std::sin(phase),
            5.0 * std::cos(phase),
            3.0 * std::sin(phase * 1.3)
        );
        synthetic_truth.push_back({t, truth_pos});
    }

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < predicted_traj.size(); ++i)
        sum_sq += (predicted_traj[i].second - synthetic_truth[i].second).squaredNorm();
    const double rmse = std::sqrt(sum_sq / static_cast<double>(predicted_traj.size()));
    std::cout << "Synthetic RMSE (with ~5m simulated truth offset): "
              << std::fixed << std::setprecision(2) << rmse << " m\n\n";

    std::cout << "Target accuracy: < 10m RMSE for 1-hour prediction\n";

    return 0;
}
