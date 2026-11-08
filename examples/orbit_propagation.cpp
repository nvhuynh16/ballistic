// Example: Basic orbit propagation
//
// Demonstrates the Trajectory + TrajectoryConfig API by flying an ISS-like
// circular orbit at 420 km for one revolution. Units: meters, meters/second,
// seconds.
//
// Run without arguments for a propagation without drag. Pass `--drag` to
// add atmospheric drag: a DragEnvironment (air density, wind and the
// satellite's ballistic coefficient) built with from_models_snapshot, a
// single altitude profile of the atmosphere computed at one place and time.

#include <ballistic/ballistic.hpp>
#include <ballistic/atmosphere/drag_environment.hpp>
#include <ballistic/atmosphere/drag_environment_generators.hpp>
#include <ballistic/atmosphere/space_weather_history.hpp>
#include <ballistic/atmosphere/beta_profile.hpp>
#include <cstring>
#include <iostream>
#include <iomanip>

using namespace ballistic;

namespace {

// Build a snapshot DragEnvironment for the point below the start position
// (equator, 0° longitude), with quiet space weather.
//
// Ballistic coefficient β = m / (Cd · A), mass over drag coefficient times
// area; ISS-like: 420000 / (2.2 · 2500) ≈ 76.4 kg/m².
//
// A narrow altitude grid (380-460 km, 10 km steps) covers the 420 km orbit
// without spending atmosphere-model evaluations on altitudes it never reaches.
std::shared_ptr<const atmosphere::DragEnvironment>
build_iss_drag_env(const Epoch& t0) {
    constexpr double beta_iss_kg_m2 = 76.36;
    constexpr double arc_duration_s = 6000.0;  // ~1.1 orbits headroom

    auto sw = atmosphere::SpaceWeatherHistory::from_constant(
        WeatherConditions::quiet(),
        atmosphere::TimeWindow{t0, t0 + arc_duration_s + 1.0});
    auto beta = atmosphere::BetaProfile1D::constant(beta_iss_kg_m2);
    std::vector<double> alt_grid{380, 390, 400, 410, 420, 430, 440, 450, 460};

    auto env = std::make_shared<const atmosphere::DragEnvironment>(
        atmosphere::from_models_snapshot(
            t0, /*ref_lat_deg=*/0.0, /*ref_lon_deg=*/0.0,
            sw, std::move(beta), std::move(alt_grid)));
    return env;
}

}  // namespace

int main(int argc, char** argv) {
    initialize({});

    bool enable_drag = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--drag") == 0) enable_drag = true;
    }

    std::cout << "=== Ballistic Orbit Propagation Example ===\n";
    std::cout << "Mode: " << (enable_drag ? "drag-on (env path)" : "drag-off")
              << "\n\n";

    // Initial epoch: January 1, 2024, 00:00:00 UTC
    Epoch t0 = Epoch::from_utc(2024, 1, 1, 0, 0, 0);

    // ISS-like circular orbit at 420 km, starting over the equator at 0° longitude.
    double altitude = 420000.0;
    double r = constants::earth::RADIUS_EQUATORIAL + altitude;
    double v_circular = std::sqrt(constants::earth::GM / r);

    // The circular speed is an INERTIAL speed. Trajectory takes Earth-fixed
    // (ITRF) states, so build the state in GCRS and convert: using v_circular
    // as an Earth-fixed velocity would add Earth's rotation (~0.5 km/s) and
    // make the orbit eccentric.
    const StateVector itrf = coords::gcrs_to_itrf_full(
        StateVector(Vector3d(r, 0.0, 0.0), Vector3d(0.0, v_circular, 0.0)), t0);
    Vector3d pos_ecef = itrf.position;
    Vector3d vel_ecef = itrf.velocity;

    std::cout << "Initial conditions:\n";
    std::cout << "  Position (ECEF): [" << pos_ecef.transpose() / 1000.0 << "] km\n";
    std::cout << "  Velocity (ECEF): [" << vel_ecef.transpose() << "] m/s\n";
    std::cout << "  Altitude: " << altitude / 1000.0 << " km\n";
    std::cout << "  Orbital velocity: " << v_circular << " m/s\n\n";

    // Build a TrajectoryConfig.
    // Drag is opt-in via config.drag_env (a shared_ptr<const DragEnvironment>).
    // Leaving it null = drag off (no atmosphere lookup in the integrator).
    TrajectoryConfig config;
    config.gravity_degree = 70;
    config.enable_third_body = true;
    config.enable_relativistic = true;
    config.itrf = coords::ITRFRealization::ITRF2020;

    if (enable_drag) {
        config.drag_env = build_iss_drag_env(t0);
        std::cout << "DragEnvironment attached (from_models_snapshot at "
                  << "equator/0°, ISS-tight 380-460 km altitude grid,\n"
                  << "β = 76.36 kg/m², quiet weather)\n\n";
    }

    Trajectory traj(t0, pos_ecef, vel_ecef, config);

    // Propagate one orbital period (~92 minutes for ISS).
    double period = 2.0 * constants::PI * std::sqrt(r * r * r / constants::earth::GM);
    std::cout << "Orbital period: " << period / 60.0 << " minutes\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Propagation results:\n";
    std::cout << "  Time (min) |   Altitude (km)  |   Speed (m/s)   \n";
    std::cout << "  -----------|------------------|------------------\n";

    for (int i = 0; i <= 10; ++i) {
        double dt = i * period / 10.0;
        Epoch t = t0 + dt;

        PropagationResult result = traj(t);

        GeodeticCoord geo = coords::ecef_to_geodetic(result.position);
        double alt_km = geo.altitude / 1000.0;
        double speed = result.velocity.norm();

        std::cout << "  " << std::setw(9) << dt / 60.0
                  << "  |  " << std::setw(14) << alt_km
                  << "  |  " << std::setw(14) << speed << "\n";
    }

    std::cout << "\n(Pass --drag to attach a DragEnvironment built at the\n"
              << "ISS sub-satellite point and compare drag-on vs drag-off.)\n";

    return 0;
}
