#pragma once

// =============================================================================
// The TLE-grade preset: the forces that still matter at TLE accuracy
// =============================================================================
//
// A state from a two-line element set (TLE) is uncertain by ~0.5 km in low
// Earth orbit (LEO) and geostationary orbit (GEO) and ~0.16 km in medium
// Earth orbit (MEO) (Flohrer et al. 2009, ESA SP-672, Table 1). Forces whose
// effect stays far below that buy nothing and cost run time. The table in
// tle_grade_table.hpp is MEASURED (scripts/measure_tle_grade_table.py): per
// orbit class, altitude and horizon (how far ahead the prediction goes),
// which optional terms move the trajectory by more than 10% of the class's
// error budget, and the lowest degree of the EGM2008 gravity field whose
// truncation stays below it.
//
// tle_grade(perigee, apogee, horizon) picks every measured altitude row from
// the one at or below the perigee to the one at or below the apogee, and the
// first measured horizon at or after the requested one, and merges them
// conservatively (a term is kept if any row keeps it; the highest degree).
// Outside what was measured it raises ConfigError naming the reason: perigee
// below 300 km, the transfer (GTO) / highly elliptical (HEO) classes (their
// TLE errors are kilometers and were not measured), MEO perigee below 5000 km, or a
// horizon beyond 7 days. Use the full recommended configuration there.
//
// The preset decides the gravity degree and which of third body (Sun, Moon,
// Jupiter), solid Earth tides, ocean tides, relativity, radiation pressure
// and drag to keep; radiation pressure and drag still need the object's
// properties, which the caller supplies (Python: recommended_config(state, accuracy="tle", ...)).
// =============================================================================

#include "../core/constants.hpp"
#include "../core/errors.hpp"
#include "../initial_state.hpp"
#include "tle_grade_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>

namespace ballistic::presets {

/// The result of tle_grade: which force terms to keep, and the measured numbers behind the
/// choice (distances in meters).
struct TleGradePreset {
    std::string regime;               // orbit class: "LEO", "MEO" or "GEO"
    double horizon_s = 0.0;           // the measured horizon used (>= the requested one)
    // Altitude span [km] of the measured table rows that were merged.
    double rows_altitude_km_min = 0.0, rows_altitude_km_max = 0.0;
    // The class's TLE error budget and the keep-threshold (10% of it) [m].
    double budget_m = 0.0, threshold_m = 0.0;
    int gravity_degree = 0;
    bool drag = false, radiation = false, third_body = false, tides = false, ocean_tides = false,
         relativity = false;
    // Largest measured effect of each term over the rows used [m].
    double effect_drag_m = 0.0, effect_radiation_m = 0.0, effect_third_body_m = 0.0,
           effect_tides_m = 0.0, effect_ocean_tides_m = 0.0, effect_relativity_m = 0.0;
    double combined_error_m = 0.0;    // largest measured preset-vs-full difference
    double speedup = 0.0;             // smallest measured speed-up over the rows used
};

/// The Flohrer et al. (2009) orbit class for perigee/apogee altitudes [km]: "LEO", "MEO",
/// "GEO", "GTO" (geostationary transfer) or "HEO" (other highly elliptical).
[[nodiscard]] inline const char* tle_orbit_class(double perigee_km, double apogee_km) noexcept {
    if (apogee_km < 2000.0) return "LEO";
    if (perigee_km > 33786.0 && apogee_km < 37786.0) return "GEO";
    if (perigee_km > 2000.0 && apogee_km < 33786.0) return "MEO";
    if (perigee_km < 2000.0 && apogee_km > 33786.0) return "GTO";
    return "HEO";
}

/// The TLE-grade preset for an orbit with the given perigee and apogee altitudes [km] (either
/// order) and prediction horizon [s], merged over the measured rows as described at the top
/// of this file. Throws ConfigError outside what was measured (GTO/HEO, perigee below the
/// lowest measured altitude, horizon <= 0 or beyond 7 days).
[[nodiscard]] inline TleGradePreset tle_grade(double perigee_km, double apogee_km, double horizon_s) {
    if (apogee_km < perigee_km) std::swap(perigee_km, apogee_km);
    const std::string cls = tle_orbit_class(perigee_km, apogee_km);
    auto refuse = [&](const std::string& why) -> TleGradePreset {
        throw ConfigError("TLE-grade preset: " + why + " (perigee " + std::to_string(perigee_km) +
                          " km, apogee " + std::to_string(apogee_km) + " km, horizon " +
                          std::to_string(horizon_s) + " s). Use the full recommended configuration "
                          "(recommended_config without accuracy=\"tle\").");
    };
    if (cls == "GTO" || cls == "HEO")
        return refuse(cls + " orbits were not measured (their TLE errors are kilometers and vary "
                      "strongly with the orbit)");
    if (!(horizon_s > 0.0)) return refuse("the horizon must be positive");

    double h_grid = -1.0;
    for (const auto& r : kTleGradeTable)
        if (r.horizon_s >= horizon_s - 1e-6 && (h_grid < 0.0 || r.horizon_s < h_grid)) h_grid = r.horizon_s;
    if (h_grid < 0.0) return refuse("horizons beyond 7 days were not measured");

    // Grid altitudes of this class at the horizon, ascending.
    auto row_at_or_below = [&](double alt_km) -> const TleGradeRow* {
        const TleGradeRow* best = nullptr;
        for (const auto& r : kTleGradeTable)
            if (cls == r.regime && r.horizon_s == h_grid && r.altitude_km <= alt_km + 1e-6 &&
                (!best || r.altitude_km > best->altitude_km))
                best = &r;
        return best;
    };
    const TleGradeRow* lo = row_at_or_below(perigee_km);
    if (!lo) return refuse("the perigee is below the lowest altitude measured for " + cls);
    const TleGradeRow* hi = row_at_or_below(apogee_km);

    TleGradePreset p;
    p.regime = cls;
    p.horizon_s = h_grid;
    p.rows_altitude_km_min = lo->altitude_km;
    p.rows_altitude_km_max = hi->altitude_km;
    p.speedup = 1e300;
    for (const auto& r : kTleGradeTable) {
        if (cls != r.regime || r.horizon_s != h_grid || r.altitude_km < lo->altitude_km ||
            r.altitude_km > hi->altitude_km)
            continue;
        p.budget_m = r.budget_m;
        p.threshold_m = r.threshold_m;
        p.gravity_degree = std::max(p.gravity_degree, r.gravity_degree);
        p.drag |= r.drag;
        p.radiation |= r.radiation;
        p.third_body |= r.third_body;
        p.tides |= r.tides;
        p.ocean_tides |= r.ocean_tides;
        p.relativity |= r.relativity;
        p.effect_drag_m = std::max(p.effect_drag_m, r.effect_drag_m);
        p.effect_radiation_m = std::max(p.effect_radiation_m, r.effect_radiation_m);
        p.effect_third_body_m = std::max(p.effect_third_body_m, r.effect_third_body_m);
        p.effect_tides_m = std::max(p.effect_tides_m, r.effect_tides_m);
        p.effect_ocean_tides_m = std::max(p.effect_ocean_tides_m, r.effect_ocean_tides_m);
        p.effect_relativity_m = std::max(p.effect_relativity_m, r.effect_relativity_m);
        p.combined_error_m = std::max(p.combined_error_m, r.combined_error_m);
        p.speedup = std::min(p.speedup, r.speedup);
    }
    return p;
}

/// The preset for a start state: perigee and apogee of its osculating two-body orbit (the
/// Keplerian ellipse through its current position and velocity, in the inertial GCRS frame;
/// altitudes above the equatorial radius, as in the measurement). Throws ConfigError if the
/// state is not a bound orbit or as tle_grade(perigee, apogee, horizon) does; converting an
/// Earth-fixed state to GCRS can throw DataCoverageError (no Earth orientation data).
[[nodiscard]] inline TleGradePreset tle_grade(const InitialState& state, double horizon_s) {
    const StateVector g = state.gcrs_state(state.realization().value_or(coords::ITRFRealization::ITRF2020));
    const double mu = constants::earth::GM;
    const double rn = g.position.norm(), v2 = g.velocity.squaredNorm();
    const Vector3d e_vec = ((v2 - mu / rn) * g.position - g.position.dot(g.velocity) * g.velocity) / mu;
    const double e = e_vec.norm();
    if (!(e < 1.0))
        throw ConfigError("TLE-grade preset: the state is not a bound orbit (e = " + std::to_string(e) + ")");
    const double p = g.position.cross(g.velocity).squaredNorm() / mu;
    const double re = constants::earth::RADIUS_EQUATORIAL;
    return tle_grade((p / (1.0 + e) - re) / 1e3, (p / (1.0 - e) - re) / 1e3, horizon_s);
}

/// What leaving radiation pressure and drag off costs, from bare_config_cost: position
/// effect [m] after 1 hour and after 1 day.
struct BareConfigCost {
    double altitude_km = 0.0;                 // the measured row used
    double radiation_1h_m = 0.0, drag_1h_m = 0.0, radiation_1d_m = 0.0, drag_1d_m = 0.0;
};

/// What leaving radiation pressure and drag off costs at `altitude_km` (circular orbit),
/// from the measured row at or below it: the light object of the table (area-to-mass
/// ratio A/m 0.02 m^2/kg, ballistic coefficient m/(Cd A) 50 kg/m^2). nullopt below the
/// lowest measured altitude.
[[nodiscard]] inline std::optional<BareConfigCost> bare_config_cost(double altitude_km) {
    const char* cls = altitude_km < 2000.0 ? "LEO" : altitude_km < 33786.0 ? "MEO" : "GEO";
    const TleGradeRow* h1 = nullptr;
    const TleGradeRow* d1 = nullptr;
    for (const auto& r : kTleGradeTable) {
        if (std::strcmp(r.regime, cls) != 0 || r.altitude_km > altitude_km + 1e-6) continue;
        if (r.horizon_s == 3600.0 && (!h1 || r.altitude_km > h1->altitude_km)) h1 = &r;
        if (r.horizon_s == 86400.0 && (!d1 || r.altitude_km > d1->altitude_km)) d1 = &r;
    }
    if (!h1 || !d1) return std::nullopt;
    return BareConfigCost{h1->altitude_km, h1->effect_radiation_m, h1->effect_drag_m,
                          d1->effect_radiation_m, d1->effect_drag_m};
}

}  // namespace ballistic::presets
