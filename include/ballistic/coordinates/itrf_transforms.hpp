#pragma once

// =============================================================================
// ITRF Realization Transformations
// =============================================================================
//
// The International Terrestrial Reference Frame (ITRF, the Earth-fixed frame)
// is re-published every few years (ITRF2000, 2005, 2008, 2014, 2020); each
// "realization" places the axes slightly differently (millimeters to
// centimeters). Orbit products state which realization they use, and
// positions must be converted before they are compared.
//
// The conversion is a Helmert transformation: a small shift, scale change and
// rotation, each with a linear rate (7 parameters + 7 rates = 14). Parameters
// from the IERS ITRS Centre:
// https://itrf.ign.fr/en/solutions/transformations
//
// The transformation from frame A to frame B is (IERS "standard model (1)"):
//   X_B = X_A + T + D*X_A + R x X_A
//
// where:
//   T = translation vector (meters; tabulated in millimeters)
//   D = scale factor (dimensionless; tabulated in ppb, parts per billion)
//   R = small-rotation vector (radians; tabulated in milliarcseconds, mas),
//       applied as a cross product
//
// Time-dependent parameters use:
//   P(t) = P(t0) + dP * (t - t0)
//
// ESA Sentinel POD Reference Frame Transition:
// =============================================
// ESA transitioned Sentinel Precise Orbit Determination (POD) products from
// ITRF2014 to ITRF2020 on March 4, 2025.
//
// - Before March 4, 2025: Use ITRFRealization::ITRF2014 for Sentinel EOF/SP3
// - After March 4, 2025:  Use ITRFRealization::ITRF2020 for Sentinel EOF/SP3
//
// The difference between ITRF2014 and ITRF2020 is ~1-3 mm at epoch, changing
// by ~0.1-0.2 mm/year (the published translation rates).
//
// =============================================================================

#include "../core/types.hpp"
#include "../core/constants.hpp"
#include <cmath>
#include <map>

namespace ballistic {
namespace coords {

// ITRFRealization enum is defined in core/types.hpp

/// Parameters of one 14-parameter Helmert transformation between two ITRF
/// realizations, in the units of the IERS tables.
struct HelmertParams {
    // Translation (millimeters, converted to meters by at_epoch)
    double Tx, Ty, Tz;
    // Scale (ppb, will be converted to dimensionless)
    double D;
    // Rotation (milliarcseconds, will be converted to radians)
    double Rx, Ry, Rz;

    // Time derivatives
    double dTx, dTy, dTz;  // mm/year
    double dD;              // ppb/year
    double dRx, dRy, dRz;   // mas/year

    // Reference epoch (decimal year)
    double t0;

    /// Evaluates the parameters at decimal year `t_year` (e.g. 2025.5) and
    /// writes them in SI units: T_out translation (m), D_out scale
    /// (dimensionless), R_out rotation vector (rad).
    void at_epoch(double t_year,
                  Vector3d& T_out,
                  double& D_out,
                  Vector3d& R_out) const {
        double dt = t_year - t0;

        // Translation in meters
        T_out(0) = (Tx + dTx * dt) * 1e-3;  // mm to m
        T_out(1) = (Ty + dTy * dt) * 1e-3;
        T_out(2) = (Tz + dTz * dt) * 1e-3;

        // Scale (ppb to dimensionless)
        D_out = (D + dD * dt) * 1e-9;

        // Rotation (mas to radians)
        const double MAS_TO_RAD = constants::PI / (180.0 * 3600.0 * 1000.0);
        R_out(0) = (Rx + dRx * dt) * MAS_TO_RAD;
        R_out(1) = (Ry + dRy * dt) * MAS_TO_RAD;
        R_out(2) = (Rz + dRz * dt) * MAS_TO_RAD;
    }
};

// IERS transformation parameters, from the ITRS Centre's tables
// "Transformation parameters from ITRF2020 to past ITRFs"
// (https://itrf.ign.fr/docs/solutions/itrf2020/Transfo-ITRF2020_TRFs.txt) and
// "... from ITRF2014 to past ITRFs"
// (https://itrf.ign.fr/doc_ITRF/Transfo-ITRF2014_ITRFs.txt).
// The tables give the transformation FROM the newer frame TO the older one,
// X_old = X_new + T + D*X_new + R x X_new, and each function's name says that
// direction. Reading a row the other way round flips every sign, an error of
// twice the shift: up to ~1.4 cm between ITRF2014 and ITRF2020 at a LEO
// radius, ~4 cm for ITRF2000.
// The ITRF2014 table supersedes the older chains (its own note), so ITRF2008,
// ITRF2005 and ITRF2000 each have a direct edge to ITRF2014.

namespace itrf_params {

/// ITRF2020 -> ITRF2014 (ITRF2020 table, row ITRF2014). Reference epoch 2015.0.
inline HelmertParams itrf2020_to_itrf2014() {
    return HelmertParams{
        // Tx, Ty, Tz (mm)
        -1.4, -0.9, 1.4,
        // D (ppb)
        -0.42,
        // Rx, Ry, Rz (mas)
        0.0, 0.0, 0.0,
        // dTx, dTy, dTz (mm/year)
        0.0, -0.1, 0.2,
        // dD (ppb/year)
        0.0,
        // dRx, dRy, dRz (mas/year)
        0.0, 0.0, 0.0,
        // Reference epoch
        2015.0
    };
}

/// ITRF2014 -> ITRF2008 (ITRF2014 table, row ITRF2008). Reference epoch 2010.0.
inline HelmertParams itrf2014_to_itrf2008() {
    return HelmertParams{
        // Tx, Ty, Tz (mm)
        1.6, 1.9, 2.4,
        // D (ppb)
        -0.02,
        // Rx, Ry, Rz (mas)
        0.0, 0.0, 0.0,
        // dTx, dTy, dTz (mm/year)
        0.0, 0.0, -0.1,
        // dD (ppb/year)
        0.03,
        // dRx, dRy, dRz (mas/year)
        0.0, 0.0, 0.0,
        // Reference epoch
        2010.0
    };
}

/// ITRF2014 -> ITRF2005 (ITRF2014 table, row ITRF2005). Reference epoch 2010.0.
inline HelmertParams itrf2014_to_itrf2005() {
    return HelmertParams{
        // Tx, Ty, Tz (mm)
        2.6, 1.0, -2.3,
        // D (ppb)
        0.92,
        // Rx, Ry, Rz (mas)
        0.0, 0.0, 0.0,
        // dTx, dTy, dTz (mm/year)
        0.3, 0.0, -0.1,
        // dD (ppb/year)
        0.03,
        // dRx, dRy, dRz (mas/year)
        0.0, 0.0, 0.0,
        // Reference epoch
        2010.0
    };
}

/// ITRF2014 -> ITRF2000 (ITRF2014 table, row ITRF2000). Reference epoch 2010.0.
inline HelmertParams itrf2014_to_itrf2000() {
    return HelmertParams{
        // Tx, Ty, Tz (mm)
        0.7, 1.2, -26.1,
        // D (ppb)
        2.12,
        // Rx, Ry, Rz (mas)
        0.0, 0.0, 0.0,
        // dTx, dTy, dTz (mm/year)
        0.1, 0.1, -1.9,
        // dD (ppb/year)
        0.11,
        // dRx, dRy, dRz (mas/year)
        0.0, 0.0, 0.0,
        // Reference epoch
        2010.0
    };
}

} // namespace itrf_params

// =============================================================================
// Transformation Graph - Lookup Table Approach
// =============================================================================
// Direct transformations are defined as edges. For indirect transformations,
// we chain through ITRF2014 as the hub.
// =============================================================================

namespace detail {

// Edge in the transformation graph: defines a direct transformation
struct TransformEdge {
    ITRFRealization from;
    ITRFRealization to;
    HelmertParams (*get_params)();  // Function pointer to parameter getter
};

// Direct transformations as published (from = the newer frame of the table).
// Each edge can be traversed in both directions (forward or inverted).
inline constexpr std::array<TransformEdge, 4> DIRECT_TRANSFORMS = {{
    {ITRFRealization::ITRF2020, ITRFRealization::ITRF2014, itrf_params::itrf2020_to_itrf2014},
    {ITRFRealization::ITRF2014, ITRFRealization::ITRF2008, itrf_params::itrf2014_to_itrf2008},
    {ITRFRealization::ITRF2014, ITRFRealization::ITRF2005, itrf_params::itrf2014_to_itrf2005},
    {ITRFRealization::ITRF2014, ITRFRealization::ITRF2000, itrf_params::itrf2014_to_itrf2000},
}};

// Find direct transformation between two frames
// Returns: {params_getter, invert_flag} or {nullptr, false} if not found
inline std::pair<HelmertParams(*)(), bool> find_direct_transform(
    ITRFRealization from, ITRFRealization to)
{
    for (const auto& edge : DIRECT_TRANSFORMS) {
        if (edge.from == from && edge.to == to) {
            return {edge.get_params, false};
        }
        if (edge.to == from && edge.from == to) {
            return {edge.get_params, true};  // Inverted
        }
    }
    return {nullptr, false};
}

// Apply single Helmert transformation
inline Vector3d apply_helmert(const Vector3d& pos, const HelmertParams& params,
                              double epoch_year, bool invert)
{
    Vector3d T;
    double D;
    Vector3d R;
    params.at_epoch(epoch_year, T, D, R);

    if (invert) {
        // Approximate inverse: negate every parameter (error is second order
        // in the parameters, far below a millimeter)
        T = -T;
        D = -D;
        R = -R;
    }

    // X_to = X_from + T + D*X_from + R×X_from
    return pos + T + D * pos + R.cross(pos);
}

} // namespace detail

/// Converts a position `pos` (m, Earth-fixed) from ITRF realization `from` to
/// `to`, with the parameters evaluated at decimal year `epoch_year`. Uses the
/// direct published transformation when one exists, otherwise chains through
/// ITRF2014 (every realization has a direct edge to it). Each step:
/// X_to = X_from + T + D*X_from + R×X_from.
[[nodiscard]] inline Vector3d transform_itrf(
    const Vector3d& pos,
    ITRFRealization from,
    ITRFRealization to,
    double epoch_year)
{
    if (from == to) {
        return pos;
    }

    // Try direct transformation first
    auto [params_fn, invert] = detail::find_direct_transform(from, to);
    if (params_fn) {
        return detail::apply_helmert(pos, params_fn(), epoch_year, invert);
    }

    // No direct path - chain through ITRF2014 as hub
    constexpr auto HUB = ITRFRealization::ITRF2014;
    return transform_itrf(transform_itrf(pos, from, HUB, epoch_year), HUB, to, epoch_year);
}

/// Size of the shift (m) that transform_itrf(from -> to, epoch_year) applies
/// to a test point on the X axis at distance `altitude_m` from the Earth's
/// center (despite the name, a geocentric radius, not a height). For
/// diagnostics.
[[nodiscard]] inline double itrf_transform_magnitude(
    ITRFRealization from,
    ITRFRealization to,
    double epoch_year,
    double altitude_m = 7e6)  // Default: a low-Earth-orbit radius
{
    Vector3d test_pos(altitude_m, 0, 0);
    Vector3d transformed = transform_itrf(test_pos, from, to, epoch_year);
    return (transformed - test_pos).norm();
}

/// Approximate decimal year for a calendar date (months taken as 1/12 year);
/// good enough for Helmert parameters, which change by at most about 2 mm
/// per year.
[[nodiscard]] inline double to_decimal_year(int year, int month, int day) {
    return year + (month - 1) / 12.0 + (day - 1) / 365.25;
}

} // namespace coords
} // namespace ballistic
