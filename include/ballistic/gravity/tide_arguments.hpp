#pragma once

// =============================================================================
// Tidal arguments and IERS 2010 Step-2 solid-tide corrections
// =============================================================================
// Every tide is a sum of sinusoids ("constituents") whose phases are integer
// combinations of six slowly varying astronomical angles, the Doodson
// fundamental arguments beta = (tau, s, h, p, N', p_s) [rad] (IERS 2010
// Conventions eqs. 5.43 + 6.8; IERS = International Earth Rotation and
// Reference Systems Service): tau, mean lunar time; s, h, the Moon's and Sun's
// mean longitudes; p, the longitude of the Moon's perigee; N', the negative
// longitude of the Moon's node; p_s, the longitude of the Sun's perigee.
// A constituent with Doodson number A.BCDEF has phase
// theta_f = sum n_i beta_i, with multipliers n = (A, B-5, ..., F-5).
// These arguments are shared by the solid-tide corrections below and the
// ocean-tide model (ocean_tides.hpp).
//
// The solid Earth tide (the Earth's own elastic deformation; solid_earth_tides.hpp)
// is computed in two steps (IERS 2010 Sec. 6.2.1). Step 1 uses one nominal
// Love number (the ratio of the deformation's potential to the tide-raising
// potential) per (n, m). Step 2, here (eq. 6.8, Tables 6.5a-c), corrects for the
// Love number k_2m varying across each tidal band (K1 sits near the
// free-core-nutation resonance: k21 = 0.257 vs nominal 0.298). The tables are
// the official in-phase / out-of-phase amplitudes in units of 1e-12 (parsed
// from the IERS Conventions Chapter 6 text; K1 = (470.9, -30.2) matches the
// chapter's worked example).
// =============================================================================

#include "../time/epoch.hpp"
#include <array>
#include <cmath>

namespace ballistic {
namespace gravity {

/// The six Doodson arguments [rad] at `epoch`.
///
/// The Moon/Sun angles use Terrestrial Time (TT, the uniform time scale of
/// the ephemerides); the Earth-rotation angle uses UT1 (time measured by the
/// Earth's actual rotation), which needs the loaded Earth orientation
/// parameters (EOP).
/// @throws DataCoverageError when UT1 is not available for the epoch (no EOP
///         table loaded under the default policy, or the epoch is too far past
///         the table's end).
[[nodiscard]] inline std::array<double, 6> doodson_arguments(const Epoch& epoch) {
    constexpr double as2r = 3.14159265358979323846 / 180.0 / 3600.0;
    const double T = epoch.julian_centuries_tt();
    const double l  = (485868.249036 + 1717915923.2178 * T + 31.8792 * T * T) * as2r;
    const double lp = (1287104.79305 + 129596581.0481 * T - 0.5532 * T * T) * as2r;
    const double F  = (335779.526232 + 1739527262.8478 * T - 12.7512 * T * T) * as2r;
    const double D  = (1072260.70369 + 1602961601.2090 * T - 6.3706 * T * T) * as2r;
    const double Om = (450160.398036 - 6962890.5431 * T + 7.4722 * T * T) * as2r;
    const double du = epoch.ut1_jd() - 2451545.0;
    const double era = 2.0 * 3.14159265358979323846 *
                       (0.7790572732640 + 1.00273781191135448 * du);
    const double gmst = era + (0.014506 + 4612.156534 * T + 1.3915817 * T * T) * as2r;
    const double s = F + Om, h = s - D;
    return {gmst + 3.14159265358979323846 - s, s, h, s - l, -Om, s - D - lp};
}

/// Phase angle sum_i n[i] * beta[i] [rad] of a constituent with Doodson
/// multipliers `n`.
[[nodiscard]] inline double doodson_angle(const std::array<int, 6>& n,
                                          const std::array<double, 6>& beta) {
    double th = 0.0;
    for (int i = 0; i < 6; ++i) th += n[i] * beta[i];
    return th;
}

/// One row of IERS Tables 6.5a-c.
struct Step2Term {
    std::array<int, 6> n;   ///< Doodson multipliers
    double ip, op;          ///< in-phase / out-of-phase amplitudes [1e-12]
};

// Table 6.5b (long period, order 0), 6.5a (diurnal, order 1), 6.5c (semidiurnal, order 2)
inline constexpr Step2Term kStep2LongPeriod[] = {
    {{0, 0, 0, 0, 1, 0}, 16.6, -6.7},
    {{0, 0, 0, 0, 2, 0}, -0.1, 0.1},
    {{0, 0, 1, 0, 0, -1}, -1.2, 0.8},
    {{0, 0, 2, 0, 0, 0}, -5.5, 4.3},
    {{0, 0, 2, 0, 1, 0}, 0.1, -0.1},
    {{0, 0, 3, 0, 0, -1}, -0.3, 0.2},
    {{0, 1, -2, 1, 0, 0}, -0.3, 0.7},
    {{0, 1, 0, -1, -1, 0}, 0.1, -0.2},
    {{0, 1, 0, -1, 0, 0}, -1.2, 3.7},
    {{0, 1, 0, -1, 1, 0}, 0.1, -0.2},
    {{0, 1, 0, 1, 0, 0}, 0.1, -0.2},
    {{0, 2, -2, 0, 0, 0}, 0.0, 0.6},
    {{0, 2, 0, -2, 0, 0}, 0.0, 0.3},
    {{0, 2, 0, 0, 0, 0}, 0.6, 6.3},
    {{0, 2, 0, 0, 1, 0}, 0.2, 2.6},
    {{0, 2, 0, 0, 2, 0}, 0.0, 0.2},
    {{0, 3, -2, 1, 0, 0}, 0.1, 0.2},
    {{0, 3, 0, -1, 0, 0}, 0.4, 1.1},
    {{0, 3, 0, -1, 1, 0}, 0.2, 0.5},
    {{0, 4, -2, 0, 0, 0}, 0.1, 0.2},
    {{0, 4, 0, -2, 0, 0}, 0.1, 0.1},
};
inline constexpr Step2Term kStep2Diurnal[] = {
    {{1, -3, 0, 2, 0, 0}, -0.1, 0.0},
    {{1, -3, 2, 0, 0, 0}, -0.1, 0.0},
    {{1, -2, 0, 1, -1, 0}, -0.1, 0.0},
    {{1, -2, 0, 1, 0, 0}, -0.7, 0.1},
    {{1, -2, 2, -1, 0, 0}, -0.1, 0.0},
    {{1, -1, 0, 0, -1, 0}, -1.3, 0.1},
    {{1, -1, 0, 0, 0, 0}, -6.8, 0.6},
    {{1, -1, 2, 0, 0, 0}, 0.1, 0.0},
    {{1, 0, -2, 1, 0, 0}, 0.1, 0.0},
    {{1, 0, 0, -1, -1, 0}, 0.1, 0.0},
    {{1, 0, 0, -1, 0, 0}, 0.4, 0.0},
    {{1, 0, 0, 1, 0, 0}, 1.3, -0.1},
    {{1, 0, 0, 1, 1, 0}, 0.3, 0.0},
    {{1, 0, 2, -1, 0, 0}, 0.3, 0.0},
    {{1, 0, 2, -1, 1, 0}, 0.1, 0.0},
    {{1, 1, -3, 0, 0, 1}, -1.9, 0.1},
    {{1, 1, -2, 0, -1, 0}, 0.5, 0.0},
    {{1, 1, -2, 0, 0, 0}, -43.4, 2.9},
    {{1, 1, -1, 0, 0, -1}, 0.6, 0.0},
    {{1, 1, -1, 0, 0, 1}, 1.6, -0.1},
    {{1, 1, 0, -2, -1, 0}, 0.1, 0.0},
    {{1, 1, 0, 0, -2, 0}, 0.1, 0.0},
    {{1, 1, 0, 0, -1, 0}, -8.8, 0.5},
    {{1, 1, 0, 0, 0, 0}, 470.9, -30.2},
    {{1, 1, 0, 0, 1, 0}, 68.1, -4.6},
    {{1, 1, 0, 0, 2, 0}, -1.6, 0.1},
    {{1, 1, 1, -1, 0, 0}, 0.1, 0.0},
    {{1, 1, 1, 0, -1, -1}, -0.1, 0.0},
    {{1, 1, 1, 0, 0, -1}, -20.6, -0.3},
    {{1, 1, 1, 0, 0, 1}, 0.3, 0.0},
    {{1, 1, 1, 0, 1, -1}, -0.3, 0.0},
    {{1, 1, 2, -2, 0, 0}, -0.2, 0.0},
    {{1, 1, 2, -2, 1, 0}, -0.1, 0.0},
    {{1, 1, 2, 0, 0, 0}, -5.0, 0.3},
    {{1, 1, 2, 0, 1, 0}, 0.2, 0.0},
    {{1, 1, 3, 0, 0, -1}, -0.2, 0.0},
    {{1, 2, -2, 1, 0, 0}, -0.5, 0.0},
    {{1, 2, -2, 1, 1, 0}, -0.1, 0.0},
    {{1, 2, 0, -1, -1, 0}, 0.1, 0.0},
    {{1, 2, 0, -1, 0, 0}, -2.1, 0.1},
    {{1, 2, 0, -1, 1, 0}, -0.4, 0.0},
    {{1, 3, -2, 0, 0, 0}, -0.2, 0.0},
    {{1, 3, 0, -2, 0, 0}, -0.1, 0.0},
    {{1, 3, 0, 0, 0, 0}, -0.6, 0.0},
    {{1, 3, 0, 0, 1, 0}, -0.4, 0.0},
    {{1, 3, 0, 0, 2, 0}, -0.1, 0.0},
    {{1, 4, 0, -1, 0, 0}, -0.1, 0.0},
    {{1, 4, 0, -1, 1, 0}, -0.1, 0.0},
};
inline constexpr Step2Term kStep2Semidiurnal[] = {
    {{2, -1, 0, 1, 0, 0}, -0.3, 0.0},
    {{2, 0, 0, 0, 0, 0}, -1.2, 0.0},
};

/// Add the Step-2 corrections to the degree-2 coefficient changes dC, dS
/// computed by Step 1 (indexed [n][m]): eq. 6.8a for (2,0), eq. 6.8b with
/// eta_1 = -i for (2,1) and eta_2 = 1 for (2,2).
/// @param beta  doodson_arguments() at the epoch [rad]
inline void add_step2_corrections(const std::array<double, 6>& beta,
                                  double dC[4][4], double dS[4][4]) {
    constexpr double u = 1e-12;
    for (const auto& t : kStep2LongPeriod) {
        const double th = doodson_angle(t.n, beta);
        dC[2][0] += u * (t.ip * std::cos(th) - t.op * std::sin(th));
    }
    for (const auto& t : kStep2Diurnal) {
        const double th = doodson_angle(t.n, beta);
        const double c = std::cos(th), sn = std::sin(th);
        dC[2][1] += u * (t.ip * sn + t.op * c);
        dS[2][1] += u * (t.ip * c - t.op * sn);
    }
    for (const auto& t : kStep2Semidiurnal) {
        const double th = doodson_angle(t.n, beta);
        dC[2][2] += u * t.ip * std::cos(th);
        dS[2][2] -= u * t.ip * std::sin(th);
    }
}

}  // namespace gravity
}  // namespace ballistic
