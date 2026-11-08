#pragma once

// =============================================================================
// U.S. Standard Atmosphere 1976 (USSA76), -5 to 86 km
// =============================================================================
//
// A fixed, average atmosphere (no weather, season or place), used near the
// ground for deep reentry (extend_density_below).
//
// Reference: NOAA/NASA/USAF, "U.S. Standard Atmosphere, 1976",
// NOAA-S/T 76-1562 (NASA-TM-X-74335), Part 1, sections 1.2-1.3:
//   - geopotential altitude H = r0 z / (r0 + z), r0 = 6 356 766 m (eq. 18):
//     the geometric height z rescaled so that gravity can be treated as
//     constant (units km', "geopotential kilometers");
//   - molecular-scale temperature TM (the temperature scaled by the change in
//     mean molar mass) linear in H on seven layers (Table 4);
//   - pressure from hydrostatic balance (eqs. 33a/33b) and density from the
//     ideal gas law, rho = P M0 / (R* TM) (eq. 42).
// Below 86 km the air is well mixed (constant composition, apart from the
// standard's 0.04% drop in mean molar mass between 80 and 86 km, Table 8), so
// the model is closed-form; this file computes it exactly from the defining constants (no
// tabulated densities). Above 86 km the 1976 standard treats each gas
// separately (diffusion); that part is NOT implemented, and evaluation there
// throws rather than extrapolate.
//
// Checked against the standard's own tables (Table I) to 0.01% in
// tests/unit/test_ussa76.cpp. It is computed from the standard's defining
// constants, not a piecewise exponential fit (Vallado Table 8-4), which is
// 3.5% high at 60 km.
// =============================================================================

#include "../core/errors.hpp"
#include <array>
#include <cmath>
#include <string>

namespace ballistic {
namespace atmosphere {
namespace ussa76 {

inline constexpr double kG0 = 9.80665;          // m/s^2, standard gravity
inline constexpr double kRStar = 8314.32;       // J/(kmol K), 1976 gas constant
inline constexpr double kM0 = 28.9644;          // kg/kmol, sea-level mean molar mass
inline constexpr double kR0 = 6356766.0;        // m, effective Earth radius (eq. 18)
inline constexpr double kGamma = 1.40;          // ratio of specific heats
inline constexpr double kP0 = 101325.0;         // Pa
inline constexpr double kMinAltitudeM = -5000.0;
inline constexpr double kMaxAltitudeM = 86000.0;

/// One layer of Table 4: TM varies linearly with H from the layer's base.
struct Layer {
    double base_km;       // geopotential altitude of the base [km']
    double base_temp_k;   // molecular-scale temperature at the base [K]
    double lapse_k_km;    // dTM/dH [K/km']
};

// Table 4 of the standard.
inline constexpr std::array<Layer, 7> kLayers{{
    {0.0, 288.15, -6.5},
    {11.0, 216.65, 0.0},
    {20.0, 216.65, 1.0},
    {32.0, 228.65, 2.8},
    {47.0, 270.65, 0.0},
    {51.0, 270.65, -2.8},
    {71.0, 214.65, -2.0},
}};

/// The atmosphere at one altitude: density [kg/m³], pressure [Pa],
/// temperature [K], mean molar mass [kg/kmol] and speed of sound [m/s].
struct Sample {
    double density_kg_m3;
    double pressure_pa;
    double temperature_k;       // kinetic temperature (= TM below 80 km; see below)
    double molar_mass_kg_kmol;  // mean molar mass
    double speed_of_sound_m_s;
};

/// Geopotential altitude [km'] from geometric altitude [m] (eq. 18).
[[nodiscard]] inline double geopotential_km(double geometric_m) {
    return kR0 * geometric_m / (kR0 + geometric_m) / 1000.0;
}

/// Mean molar mass over its sea-level value, M/M0, at geometric altitude [m]:
/// Table 8 of the standard from 80 to 86 km (0.5 km steps, linear between);
/// 1 below 80 km. The kinetic temperature is T = TM * M / M0 (eq. 21).
[[nodiscard]] inline double molar_mass_ratio(double geometric_m) {
    static constexpr std::array<double, 13> ratio{
        1.000000, 0.999996, 0.999989, 0.999971, 0.999941, 0.999909, 0.999870,
        0.999829, 0.999786, 0.999741, 0.999694, 0.999641, 0.999579};
    const double z_km = geometric_m / 1000.0;
    if (z_km <= 80.0) return 1.0;
    const double x = (z_km - 80.0) / 0.5;
    const auto i = static_cast<std::size_t>(std::min(x, 11.999999));
    const double f = x - static_cast<double>(i);
    return ratio[i] + f * (ratio[i + 1] - ratio[i]);
}

/// The atmosphere at a geometric altitude [m] (height above sea level) in
/// [-5 km, 86 km]. Throws DataCoverageError outside that range.
[[nodiscard]] inline Sample evaluate(double geometric_altitude_m) {
    if (!(geometric_altitude_m >= kMinAltitudeM && geometric_altitude_m <= kMaxAltitudeM)) {
        throw DataCoverageError(
            "U.S. Standard Atmosphere 1976 is implemented from -5 km to 86 km (the well-mixed "
            "layers); altitude " + std::to_string(geometric_altitude_m / 1000.0) +
            " km is outside. Above 86 km use NRLMSIS 2.1 (the drag environment's default).");
    }
    const double H = geopotential_km(geometric_altitude_m);
    // Pressure at each layer base, from the sea-level value (eqs. 33a/33b).
    double p_base = kP0;
    std::size_t b = 0;
    for (std::size_t i = 0; i + 1 < kLayers.size(); ++i) {
        if (H < kLayers[i + 1].base_km) break;
        const auto& L = kLayers[i];
        const double dH = kLayers[i + 1].base_km - L.base_km;
        p_base *= (L.lapse_k_km != 0.0)
            ? std::pow(L.base_temp_k / (L.base_temp_k + L.lapse_k_km * dH),
                       kG0 * kM0 / (kRStar * L.lapse_k_km * 1e-3))
            : std::exp(-kG0 * kM0 * dH * 1e3 / (kRStar * L.base_temp_k));
        b = i + 1;
    }
    const auto& L = kLayers[b];
    const double dH = H - L.base_km;
    const double tm = L.base_temp_k + L.lapse_k_km * dH;
    const double p = (L.lapse_k_km != 0.0)
        ? p_base * std::pow(L.base_temp_k / tm, kG0 * kM0 / (kRStar * L.lapse_k_km * 1e-3))
        : p_base * std::exp(-kG0 * kM0 * dH * 1e3 / (kRStar * L.base_temp_k));
    const double m_ratio = molar_mass_ratio(geometric_altitude_m);
    Sample s;
    s.pressure_pa = p;
    s.density_kg_m3 = p * kM0 / (kRStar * tm);          // eq. 42 (uses TM)
    s.temperature_k = tm * m_ratio;                     // eq. 21
    s.molar_mass_kg_kmol = kM0 * m_ratio;
    s.speed_of_sound_m_s = std::sqrt(kGamma * kRStar * tm / kM0);   // eq. 50
    return s;
}

/// Density [kg/m³] only; same range and throw as evaluate().
[[nodiscard]] inline double density(double geometric_altitude_m) {
    return evaluate(geometric_altitude_m).density_kg_m3;
}

}  // namespace ussa76
}  // namespace atmosphere
}  // namespace ballistic
