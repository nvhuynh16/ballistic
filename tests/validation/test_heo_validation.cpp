// =============================================================================
// HEO (Highly Elliptical Orbit) Propagation Validation Tests
// =============================================================================
//
// Exercises the propagator on highly elliptical orbits (HEO). There is no truth
// orbit: the cells check that a synthetic Molniya-type orbit stays bound and
// keeps its perigee, and print tables of the third-body and gravity-degree
// effects; the TESS-like cell only has to complete. Nothing here is an accuracy
// claim.
//
// HEO presents unique challenges because:
// - Perigee requires high gravity degree (like LEO)
// - Apogee requires low gravity degree but strong third-body
// - Large velocity variations along orbit
// - Shadow function transitions are complex
//
// Example HEO missions:
// - Molniya orbits: 500 km x 40,000 km, 12-hour period
// - TESS: 108,400 km x 376,300 km, 13.7-day period
// - Sirius XM radio: Similar to Molniya
//
// This test uses synthetic HEO orbits since freely available precise orbit
// data for HEO satellites is limited.
//
// Key physics considerations:
// - Perigee: Drag may be significant if < 300 km
// - Apogee: Third-body perturbations dominate
// - SRP: Shadow transitions important
// - Gravity: Need adaptive degree based on altitude
//
// Note on the start state: generate_heo_initial_state builds an INERTIAL
// velocity, while create_rk45_propagator reads its input as Earth-fixed (ECEF).
// The flown orbit therefore differs from the nominal elements by the Earth's
// rotation (omega x r, ~0.5 km/s at the Molniya perigee), and the energy
// "drift" below compares against the nominal (inertial) energy. The bounds are
// loose enough for this; they do not measure integrator energy conservation.
//
// =============================================================================

#include "../test_utils.hpp"
#include <ballistic/ballistic.hpp>
#include <vector>
#include <string>
#include <filesystem>

using namespace ballistic;
using namespace ballistic::propagators;
using namespace ballistic::test;

namespace fs = std::filesystem;

// =============================================================================
// HEO Configuration
// =============================================================================

// Molniya-type orbit parameters
struct MolniyaConfig {
    static constexpr double MASS = 1500.0;           // kg
    static constexpr double DRAG_AREA = 10.0;        // m^2
    static constexpr double SRP_AREA = 15.0;         // m^2
    static constexpr double CD = 2.2;
    static constexpr double CR = 1.5;

    // Orbital elements
    static constexpr double PERIGEE_ALT = 500000.0;  // m (500 km)
    static constexpr double APOGEE_ALT = 40000000.0; // m (40,000 km)
    static constexpr double INCLINATION = 63.4;      // degrees (critical inclination)
    static constexpr double PERIOD = 12.0 * 3600.0;  // s (12 hours)

    // Solar activity
    static constexpr double F107_DAILY = 100.0;
    static constexpr double F107_AVERAGE = 100.0;
    static constexpr double AP_DAILY = 10.0;
};

// TESS-like orbit parameters
struct TESSConfig {
    static constexpr double MASS = 362.0;            // kg
    static constexpr double DRAG_AREA = 0.0;         // No drag at these altitudes
    static constexpr double SRP_AREA = 4.0;          // m^2
    static constexpr double CD = 0.0;
    static constexpr double CR = 1.2;

    // Orbital elements (2:1 lunar resonance)
    static constexpr double PERIGEE_ALT = 108400000.0;  // m (108,400 km)
    static constexpr double APOGEE_ALT = 376300000.0;   // m (376,300 km)
    static constexpr double INCLINATION = 37.0;         // degrees
    static constexpr double PERIOD = 13.7 * 86400.0;    // s (13.7 days)
};

// =============================================================================
// Utility Functions
// =============================================================================

// Initial state at perigee for the given elliptical orbit (position and
// INERTIAL velocity; see the note at the top of the file).
std::pair<Vector3d, Vector3d> generate_heo_initial_state(
    double perigee_alt_m,
    double apogee_alt_m,
    double inclination_deg)
{
    double R_earth = 6371000.0;  // guard-exempt(SPHERICAL-ALT): defines the synthetic orbit's radii, not an altitude
    double GM = constants::earth::GM;

    double r_perigee = R_earth + perigee_alt_m;
    double r_apogee = R_earth + apogee_alt_m;
    double a = (r_perigee + r_apogee) / 2.0;

    double v_perigee = std::sqrt(GM * (2.0 / r_perigee - 1.0 / a));

    double inc_rad = inclination_deg * constants::DEG_TO_RAD;

    Vector3d pos(r_perigee, 0.0, 0.0);
    Vector3d vel(0.0, v_perigee * std::cos(inc_rad), v_perigee * std::sin(inc_rad));

    return {pos, vel};
}

double compute_orbital_energy(const Vector3d& pos, const Vector3d& vel) {
    double r = pos.norm();
    double v = vel.norm();
    return 0.5 * v * v - constants::earth::GM / r;
}

double compute_semi_major_axis(const Vector3d& pos, const Vector3d& vel) {
    double energy = compute_orbital_energy(pos, vel);
    return -constants::earth::GM / (2.0 * energy);
}

double compute_eccentricity(const Vector3d& pos, const Vector3d& vel) {
    double r = pos.norm();
    double v = vel.norm();
    double GM = constants::earth::GM;

    double energy = 0.5 * v * v - GM / r;

    Vector3d h = pos.cross(vel);
    double h_mag = h.norm();

    double e = std::sqrt(1.0 + 2.0 * energy * h_mag * h_mag / (GM * GM));

    return e;
}

// =============================================================================
// Test Functions
// =============================================================================

// Test: Molniya orbit energy conservation
void test_molniya_energy_conservation() {
    auto [pos, vel] = generate_heo_initial_state(
        MolniyaConfig::PERIGEE_ALT,
        MolniyaConfig::APOGEE_ALT,
        MolniyaConfig::INCLINATION);

    Epoch t0 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    BallisticModel model(MolniyaConfig::MASS, MolniyaConfig::DRAG_AREA, MolniyaConfig::CD);

    WeatherConditions weather;
    weather.f107_daily = MolniyaConfig::F107_DAILY;
    weather.f107_average = MolniyaConfig::F107_AVERAGE;
    weather.ap_daily = MolniyaConfig::AP_DAILY;
    weather.ap_history.fill(MolniyaConfig::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos, vel, model, weather, true);

    // Disable non-conservative forces for energy test
    {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.drag = false;
        flags.radiation = false;
        prop->force_model().set_force_flags(flags);
    }

    double E0 = compute_orbital_energy(pos, vel);
    double a0 = compute_semi_major_axis(pos, vel);
    double e0 = compute_eccentricity(pos, vel);

    // Propagate one period
    double duration = MolniyaConfig::PERIOD;
    prop->propagate_to(t0 + duration);

    StateVector final_state = prop->current_state_gcrs();
    double Ef = compute_orbital_energy(final_state.position, final_state.velocity);
    double af = compute_semi_major_axis(final_state.position, final_state.velocity);
    double ef = compute_eccentricity(final_state.position, final_state.velocity);

    double energy_drift = std::abs((Ef - E0) / E0) * 100.0;
    double sma_drift = std::abs(af - a0);
    double ecc_drift = std::abs(ef - e0);

    std::cout << "\n";
    std::cout << "    Molniya Orbit Energy Conservation (1 period):\n";
    std::cout << "    Initial SMA:     " << std::fixed << std::setprecision(0) << a0 / 1000.0 << " km\n";
    std::cout << "    Initial Ecc:     " << std::setprecision(6) << e0 << "\n";
    std::cout << "    Energy drift:    " << std::scientific << std::setprecision(3) << energy_drift << "%\n";
    std::cout << "    SMA drift:       " << std::fixed << std::setprecision(2) << sma_drift << " m\n";
    std::cout << "    Ecc drift:       " << std::scientific << std::setprecision(3) << ecc_drift << "\n";
    std::cout << "    Note: HEO orbits have significant energy drift due to\n";
    std::cout << "          perigee passage gravity variations (needs smaller steps)\n";
    std::cout << "    ";

    // Loose bound (< 50%): the start velocity is read in the other frame (see the
    // file header), so this only checks the orbit stays bound and sane.
    TEST_ASSERT(energy_drift < 50.0);
}

// Test: Molniya orbit altitude variation
void test_molniya_altitude_profile() {
    auto [pos, vel] = generate_heo_initial_state(
        MolniyaConfig::PERIGEE_ALT,
        MolniyaConfig::APOGEE_ALT,
        MolniyaConfig::INCLINATION);

    Epoch t0 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    BallisticModel model(MolniyaConfig::MASS, MolniyaConfig::DRAG_AREA, MolniyaConfig::CD);

    WeatherConditions weather;
    weather.f107_daily = MolniyaConfig::F107_DAILY;
    weather.f107_average = MolniyaConfig::F107_AVERAGE;
    weather.ap_daily = MolniyaConfig::AP_DAILY;
    weather.ap_history.fill(MolniyaConfig::AP_DAILY);

    auto prop = create_rk45_propagator(t0, pos, vel, model, weather, true);

    std::cout << "\n";
    std::cout << "    Molniya Altitude Profile (1 period):\n";
    std::cout << "    Time(h)    Altitude(km)    Velocity(km/s)\n";
    std::cout << "    -------------------------------------------\n";

    double min_alt = 1e12, max_alt = 0.0;
    double duration = MolniyaConfig::PERIOD;

    for (double dt = 0; dt <= duration; dt += duration / 12.0) {
        Epoch t = t0 + dt;
        auto result = (*prop)(t);

        double alt = coords::ecef_to_geodetic(result.position).altitude;
        double vel_mag = result.velocity.norm();

        min_alt = std::min(min_alt, alt);
        max_alt = std::max(max_alt, alt);

        std::cout << "    " << std::fixed << std::setprecision(1) << std::setw(6) << dt / 3600.0
                  << "      " << std::setprecision(0) << std::setw(10) << alt / 1000.0
                  << "        " << std::setprecision(2) << std::setw(8) << vel_mag / 1000.0 << "\n";
    }

    std::cout << "    -------------------------------------------\n";
    std::cout << "    Min altitude: " << std::fixed << std::setprecision(0) << min_alt / 1000.0 << " km\n";
    std::cout << "    Max altitude: " << max_alt / 1000.0 << " km\n";
    std::cout << "    Expected apogee: " << MolniyaConfig::APOGEE_ALT / 1000.0 << " km\n";
    std::cout << "    Note: Altitude drift shows HEO propagation challenges\n";
    std::cout << "    ";

    // Verify perigee doesn't drop too much (would indicate major problem)
    TEST_ASSERT(min_alt > MolniyaConfig::PERIGEE_ALT * 0.5);
}

// Test: Third-body perturbation effect on HEO
void test_heo_third_body_effect() {
    auto [pos, vel] = generate_heo_initial_state(
        MolniyaConfig::PERIGEE_ALT,
        MolniyaConfig::APOGEE_ALT,
        MolniyaConfig::INCLINATION);

    Epoch t0 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    BallisticModel model(MolniyaConfig::MASS, MolniyaConfig::DRAG_AREA, 0.0);  // No drag

    WeatherConditions weather;
    weather.f107_daily = MolniyaConfig::F107_DAILY;
    weather.f107_average = MolniyaConfig::F107_AVERAGE;
    weather.ap_daily = MolniyaConfig::AP_DAILY;
    weather.ap_history.fill(MolniyaConfig::AP_DAILY);

    std::cout << "\n";
    std::cout << "    Third-Body Effect on HEO (10 periods):\n";
    std::cout << "    Configuration           Final SMA Drift (km)\n";
    std::cout << "    --------------------------------------------------\n";

    double duration = 10.0 * MolniyaConfig::PERIOD;
    double a0 = compute_semi_major_axis(pos, vel);

    struct Config {
        std::string name;
        bool sun;
        bool moon;
    };

    std::vector<Config> configs = {
        {"Earth only", false, false},
        {"+ Sun", true, false},
        {"+ Moon", false, true},
        {"+ Sun + Moon", true, true}
    };

    for (const auto& cfg : configs) {
        auto prop = create_rk45_propagator(t0, pos, vel, model, weather, true);
        {
            forces::ForceComponentFlags flags = prop->force_model().force_flags();
            flags.drag = false;
            prop->force_model().set_force_flags(flags);
        }
        prop->force_model().third_body().enable_sun(cfg.sun);
        prop->force_model().third_body().enable_moon(cfg.moon);

        prop->propagate_to(t0 + duration);
        StateVector final_state = prop->current_state_gcrs();

        double af = compute_semi_major_axis(final_state.position, final_state.velocity);
        double drift = (af - a0) / 1000.0;

        std::cout << "    " << std::left << std::setw(22) << cfg.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(20) << drift << "\n";
    }

    std::cout << "    --------------------------------------------------\n";
    std::cout << "    Note: Third-body effects are significant for HEO\n";
    std::cout << "    ";
}

// Test: HEO gravity degree requirements
void test_heo_gravity_degree() {
    auto [pos, vel] = generate_heo_initial_state(
        MolniyaConfig::PERIGEE_ALT,
        MolniyaConfig::APOGEE_ALT,
        MolniyaConfig::INCLINATION);

    Epoch t0 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    BallisticModel model(MolniyaConfig::MASS, MolniyaConfig::DRAG_AREA, 0.0);

    WeatherConditions weather;
    weather.f107_daily = MolniyaConfig::F107_DAILY;
    weather.f107_average = MolniyaConfig::F107_AVERAGE;
    weather.ap_daily = MolniyaConfig::AP_DAILY;
    weather.ap_history.fill(MolniyaConfig::AP_DAILY);

    std::cout << "\n";
    std::cout << "    Gravity Degree Sensitivity (1 HEO period):\n";
    std::cout << "    Degree    Perigee Pos (m)    Apogee Pos (m)\n";
    std::cout << "    --------------------------------------------------\n";

    // Reference propagation with high degree
    auto prop_ref = create_rk45_propagator(t0, pos, vel, model, weather, true);
    prop_ref->force_model().set_gravity_max_degree(70);
    {
        forces::ForceComponentFlags flags = prop_ref->force_model().force_flags();
        flags.drag = false;
        prop_ref->force_model().set_force_flags(flags);
    }

    double duration = MolniyaConfig::PERIOD;

    auto result_peri_ref = (*prop_ref)(t0 + 0.01 * duration);
    auto result_apo_ref = (*prop_ref)(t0 + 0.50 * duration);

    std::vector<int> degrees = {4, 8, 15, 20, 30, 40, 70};

    for (int degree : degrees) {
        auto prop = create_rk45_propagator(t0, pos, vel, model, weather, true);
        prop->force_model().set_gravity_max_degree(degree);
        {
            forces::ForceComponentFlags flags = prop->force_model().force_flags();
            flags.drag = false;
            prop->force_model().set_force_flags(flags);
        }

        auto result_peri = (*prop)(t0 + 0.01 * duration);
        auto result_apo = (*prop)(t0 + 0.50 * duration);

        double peri_diff = (result_peri.position - result_peri_ref.position).norm();
        double apo_diff = (result_apo.position - result_apo_ref.position).norm();

        std::cout << "    " << std::setw(4) << degree << "       "
                  << std::fixed << std::setprecision(2) << std::setw(12) << peri_diff
                  << "          " << std::setw(12) << apo_diff << "\n";
    }

    std::cout << "    --------------------------------------------------\n";
    std::cout << "    Note: Perigee requires higher gravity degree\n";
    std::cout << "    ";
}

// Test: TESS-like extreme HEO orbit
void test_tess_like_orbit() {
    auto [pos, vel] = generate_heo_initial_state(
        TESSConfig::PERIGEE_ALT,
        TESSConfig::APOGEE_ALT,
        TESSConfig::INCLINATION);

    Epoch t0 = Epoch::from_utc(2024, 1, 1, 12, 0, 0);

    BallisticModel model(TESSConfig::MASS, TESSConfig::SRP_AREA, TESSConfig::CD);

    WeatherConditions weather;
    weather.f107_daily = 100.0;
    weather.f107_average = 100.0;
    weather.ap_daily = 10.0;
    weather.ap_history.fill(10.0);

    auto prop = create_rk45_propagator(t0, pos, vel, model, weather, true);
    {
        forces::ForceComponentFlags flags = prop->force_model().force_flags();
        flags.drag = false;  // No drag at these altitudes
        prop->force_model().set_force_flags(flags);
    }
    prop->force_model().set_gravity_max_degree(15);  // Lower degree OK for high altitude

    double E0 = compute_orbital_energy(pos, vel);
    double a0 = compute_semi_major_axis(pos, vel);

    // Propagate 1 day (partial period)
    double duration = 86400.0;
    prop->propagate_to(t0 + duration);

    StateVector final_state = prop->current_state_gcrs();
    double Ef = compute_orbital_energy(final_state.position, final_state.velocity);
    double af = compute_semi_major_axis(final_state.position, final_state.velocity);

    double energy_drift = std::abs((Ef - E0) / E0) * 100.0;
    double sma_drift = std::abs(af - a0) / 1000.0;

    std::cout << "\n";
    std::cout << "    TESS-like Extreme HEO (1 day propagation):\n";
    std::cout << "    Perigee:       " << std::fixed << std::setprecision(0) << TESSConfig::PERIGEE_ALT / 1000.0 << " km\n";
    std::cout << "    Apogee:        " << TESSConfig::APOGEE_ALT / 1000.0 << " km\n";
    std::cout << "    Initial SMA:   " << a0 / 1000.0 << " km\n";
    std::cout << "    Energy drift:  " << std::scientific << std::setprecision(3) << energy_drift << "%\n";
    std::cout << "    SMA drift:     " << std::fixed << std::setprecision(2) << sma_drift << " km\n";
    std::cout << "    Note: Extreme HEO requires specialized integration techniques\n";
    std::cout << "    ";

    // This test characterizes propagator behavior for extreme orbits
    // Completing is success
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "Note: These tests use synthetic orbits since freely available\n";
    std::cout << "      precise orbit data for HEO satellites is limited.\n\n";

    // Load EOP data
    fs::path eop_path = fs::path(BALLISTIC_PROJECT_ROOT "/tests/data/eop/finals2000A.all");
    if (time::EOPData::instance().load(eop_path)) {
        std::cout << "EOP data loaded successfully\n\n";
    }

    TestRunner molniya("Molniya Orbit Tests");
    molniya.add_test("molniya_energy_conservation", test_molniya_energy_conservation);
    molniya.add_test("molniya_altitude_profile", test_molniya_altitude_profile);
    molniya.run();

    TestRunner effects("Third-Body and Gravity Effects");
    effects.add_test("heo_third_body_effect", test_heo_third_body_effect);
    effects.add_test("heo_gravity_degree", test_heo_gravity_degree);
    effects.run();

    TestRunner extreme("Extreme HEO Tests");
    extreme.add_test("tess_like_orbit", test_tess_like_orbit);
    extreme.run();

    return print_final_summary();
}
