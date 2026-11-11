#pragma once

// =============================================================================
// NRLMSIS 2.1 Atmosphere Model - Full C++ Implementation
// =============================================================================
//
// This is a complete C++ port of the official NRLMSIS 2.1 Fortran implementation
// from the Naval Research Laboratory.
//
// Reference: Emmert, J.T., et al. (2022), NRLMSIS 2.1: An Empirical Model of
//            Nitric Oxide Incorporated into NRLMSIS 2.0, JGR Space Physics.
//
// Original Fortran source: NRL, https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.1/
// (mirror used for the translation: https://github.com/jacobwilliams/NRLMSIS2.1)
//
// This software incorporates the MSIS(R) empirical atmospheric model software
// designed and provided by NRL. Use is governed by the Open Source Academic
// Research License Agreement contained in the file nrlmsis2.1_license.txt
// (docs/licenses/nrlmsis2.1_license.txt in this repository) — research,
// academic, and non-profit use only; commercial use prohibited.
//
// Change notice (NRL-SOF-014-1 §4b): translated from the official Fortran
// (msis_*.F90, NRLMSIS 2.1) to C++ on 2026-01-25; this header comment was
// revised on 2026-07-18 and 2026-11-04. On 2026-11-11 two constants were
// corrected to the values in msis_constants.F90: the surface log mixing
// ratios (LNVMR, now log() of the CIPM-2007 mole fractions; the hard-coded
// values were off by up to 3e-4) and the anomalous-O scale height (16 g/mol).
// Model formulation and parameter values are unchanged; every output of the
// official double-precision test vectors (msis2.1_test_ref_dp.txt) matches to
// half its last printed digit (tests/unit/test_nrlmsis21_reference.cpp).
//
// =============================================================================

#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "../../time/epoch.hpp"
#include "../../core/types.hpp"
#include "nrlmsis21_parm.hpp"

namespace ballistic {
namespace atmosphere {

// Forward declaration for debug test access
class NRLMSIS21_Debug;

// =============================================================================
// Constants Module (from msis_constants.F90)
// =============================================================================
namespace msis21 {
namespace constants {

// Missing value indicator
constexpr double DMISSING = 9.999e-38;

// Mathematical constants
constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;
constexpr double DOY2RAD = 2.0 * PI / 365.0;
constexpr double LST2RAD = PI / 12.0;
constexpr double TANH1 = 0.7615941559557649;  // tanh(1.0)

// Physical constants
constexpr double KB = 1.380649e-23;           // Boltzmann constant (J/K)
constexpr double NA = 6.02214076e23;          // Avogadro constant
constexpr double G0 = 9.80665;                // Standard gravity (m/s^2)

// Species indices (1-based to match Fortran)
constexpr int SPEC_MASS = 1;  // Total mass density
constexpr int SPEC_N2 = 2;    // N2
constexpr int SPEC_O2 = 3;    // O2
constexpr int SPEC_O = 4;     // O
constexpr int SPEC_HE = 5;    // He
constexpr int SPEC_H = 6;     // H
constexpr int SPEC_AR = 7;    // Ar
constexpr int SPEC_N = 8;     // N
constexpr int SPEC_OA = 9;    // Anomalous O
constexpr int SPEC_NO = 10;   // NO

// Species molecular masses (kg/molecule) - matches Fortran specmass
constexpr std::array<double, 11> SPECMASS = {
    0.0,                         // Index 0 (unused)
    0.0,                         // Mass density (dummy)
    28.0134 / (1.0e3 * NA),      // N2
    31.9988 / (1.0e3 * NA),      // O2
    31.9988 / 2.0 / (1.0e3 * NA), // O (15.9994)
    4.0 / (1.0e3 * NA),          // He (matches Fortran exactly: 4.0)
    1.0 / (1.0e3 * NA),          // H (matches Fortran exactly: 1.0)
    39.948 / (1.0e3 * NA),       // Ar
    28.0134 / 2.0 / (1.0e3 * NA), // N (14.0067)
    31.9988 / 2.0 / (1.0e3 * NA), // Anomalous O (15.9994)
    (28.0134 + 31.9988) / 2.0 / (1.0e3 * NA)  // NO (30.0061)
};

// Dry air mean mass (kg/molecule)
constexpr double MBAR = 28.96546 / (1.0e3 * NA);

// ln of the CIPM-2007 dry-air mole fractions, as in msis_constants.F90
// (lnvmr = log(...)); std::log is not constexpr before C++26.
constexpr std::array<double, 11> LNVMR = {
    0.0,                     // Index 0 (unused)
    0.0,                     // Mass density (dummy)
    -0.24737477036295383,    // N2: ln(0.780848)
    -1.563556737177912,      // O2: ln(0.209390)
    0.0,                     // O (dummy)
    -12.166851932376892,     // He: ln(0.0000052)
    0.0,                     // H (dummy)
    -4.674305924822954,      // Ar: ln(0.009332)
    0.0,                     // N (dummy)
    0.0,                     // Anomalous O (dummy)
    0.0                      // NO (dummy)
};

// ln(P0) where P0 is global average surface pressure (Pa)
constexpr double LNP0 = 11.515614;

// Derived constants (matches Fortran exactly)
constexpr double G0DIVKB = G0 / KB * 1.0e3;           // K/(kg km)
constexpr double MBARG0DIVKB = MBAR * G0DIVKB;        // K/km (for air)

// Vertical profile parameters
constexpr int ND = 27;            // Number of temperature profile nodes
constexpr int P_ORDER = 4;        // Spline order
constexpr int NL = ND - P_ORDER;  // Last temperature profile level index = 23
constexpr int NLS = 9;            // Last parameter index for each species

// Reference altitudes (km)
constexpr double ZETA_F = 70.0;   // Fully mixed below this
constexpr double ZETA_B = 122.5;  // Bates Profile above this
constexpr double ZETA_A = 85.0;   // Reference height for active minor species
constexpr double ZETA_GAMMA = 100.0;  // Reference height for tanh taper
constexpr double H_GAMMA = 1.0 / 30.0;  // Inverse scale height of tanh taper

// Anomalous oxygen scale height (km). The Fortran (msis_constants.F90, HOA)
// uses a mass of exactly 16 g/mol here, not SPECMASS[SPEC_OA] = 15.9994.
constexpr double H_OA = 4000.0 * KB / ((16.0 / (1.0e3 * NA)) * G0) * 1.0e-3;  // ~500 km

// Spline indices
constexpr int IZFMX = 13;   // Fully mixed below this spline index
constexpr int IZFX = 14;    // Spline index at zetaF (70 km)
constexpr int IZAX = 17;    // Spline index at zetaA (85 km)
constexpr int ITEX = NL;    // Index of exospheric temperature = 23
constexpr int ITGB0 = NL - 1;  // Index of temperature gradient at zetaB = 22
constexpr int ITB0 = NL - 2;   // Index of temperature at zetaB = 21

// Temperature spline nodes (30 values, indexed 0 to 29)
constexpr std::array<double, 30> NODES_TN = {
    -15., -10., -5., 0., 5., 10., 15., 20., 25., 30.,
    35., 40., 45., 50., 55., 60., 65., 70., 75., 80.,
    85., 92.5, 102.5, 112.5, 122.5, 132.5, 142.5, 152.5, 162.5, 172.5
};

// Basis function parameters
constexpr int MBF = 383;      // Last index of linear terms
constexpr int MAXN = 6;       // Maximum Legendre degree
constexpr int MAXL = 3;       // Maximum local time order
constexpr int MAXM = 2;       // Maximum longitude order

// Index offsets for basis function terms
constexpr int CTIMEIND = 0;
constexpr int CINTANN = 7;    // ctimeind + (amaxn+1)
constexpr int CTIDE = 35;
constexpr int CSPW = 185;
constexpr int CSFX = 295;     // Starting index of linear solar flux terms
constexpr int NSFX = 5;       // Number of linear solar flux terms
constexpr int CEXTRA = 300;   // CSFX + NSFX
constexpr int NSFXMOD = 5;    // Number of nonlinear modulating solar flux terms
constexpr int NMAG = 54;      // Number of geomagnetic parameterization terms
constexpr int NUT = 12;       // Number of UT parameterization terms
constexpr int CNONLIN = 384;  // MBF + 1, starting index of nonlinear terms
constexpr int CSFXMOD = 384;  // = CNONLIN, starting index of modulating solar flux terms
constexpr int CMAG = 389;     // CSFXMOD + NSFXMOD, starting index of geomagnetic terms
constexpr int CUT = 443;      // CMAG + NMAG, starting index of UT terms

// O1 spline parameters
constexpr int NDO1 = 13;
constexpr int NSPLO1 = NDO1 - 5;  // 8 unconstrained spline parameters
constexpr double NODES_O1[14] = {
    35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 92.5, 102.5, 112.5
};
constexpr double ZETAREF_O1 = ZETA_A;  // 85 km

// NO spline parameters
constexpr int NDNO = 13;
constexpr int NSPLNO = NDNO - 5;  // 8 unconstrained spline parameters
constexpr double NODES_NO[14] = {
    47.5, 55., 62.5, 70., 77.5, 85., 92.5, 100., 107.5, 115., 122.5, 130., 137.5, 145.
};
constexpr double ZETAREF_NO = ZETA_B;  // 122.5 km
constexpr double ZETAREF_OA = ZETA_B;  // 122.5 km

// C1 continuity matrices for O1 splines (2x2, stored row-major)
// Fortran reshape fills column-major, so c1o1(1,1)=1.75, c1o1(2,1)=-2.916..., c1o1(1,2)=-1.625..., c1o1(2,2)=21.458...
constexpr double C1O1[2][2] = {
    { 1.75,               -1.624999900076852},
    {-2.916666573405061,  21.458332647194382}
};
constexpr double C1O1ADJ[2] = {0.257142857142857, -0.102857142686844};

// C1 continuity matrices for NO splines
constexpr double C1NO[2][2] = {
    { 1.5, 0.0},
    {-3.75, 15.0}
};
constexpr double C1NOADJ[2] = {0.166666666666667, -0.066666666666667};

// Weights for temperature gradient at zetaA
constexpr std::array<double, 3> WGHTAXDZ = {-0.102857142857, 0.0495238095238, 0.053333333333};

// B-spline weights at key altitudes (precomputed)
constexpr std::array<double, 4> S5ZETA_B = {0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667};
constexpr std::array<double, 5> S6ZETA_B = {0.008771929824561, 0.216228070175439, 0.550000000000000, 0.216666666666667, 0.008333333333333};
constexpr std::array<double, 3> S4ZETA_F = {0.166666666666667, 0.666666666666667, 0.166666666666667};
constexpr std::array<double, 4> S5ZETA_F = {0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667};
constexpr std::array<double, 3> S5ZETA_0 = {0.458333333333333, 0.458333333333333, 0.041666666666667};
constexpr std::array<double, 3> S4ZETA_A = {0.257142857142857, 0.653968253968254, 0.088888888888889};
constexpr std::array<double, 4> S5ZETA_A = {0.085714285714286, 0.587590187590188, 0.313020313020313, 0.013675213675214};
constexpr std::array<double, 5> S6ZETA_A = {0.023376623376623, 0.378732378732379, 0.500743700743701, 0.095538448479625, 0.001608848667672};

// C2 continuity matrix for temperature splines (top 3 coefficients)
constexpr std::array<std::array<double, 3>, 3> C2TN = {{
    {1.0, 1.0, 1.0},
    {-10.0, 0.0, 10.0},
    {33.333333333333336, -16.666666666666668, 33.333333333333336}
}};

}  // namespace constants

// =============================================================================
// Utility Functions Module (from msis_utils.F90)
// =============================================================================
namespace utils {

using namespace constants;

// Geodetic altitude to geopotential height (from msis_utils.F90)
inline double alt2gph(double lat, double alt) {
    // WGS84 Defining parameters
    constexpr double a = 6378.1370e3;           // Semi-major axis (m)
    constexpr double finv = 298.257223563;      // Reciprocal of flattening
    constexpr double w = 7.292115e-5;           // Angular velocity (rad/s)
    constexpr double GM = 398600.4418e9;        // GM (m^3/s^2)

    // WGS84 Derived parameters
    constexpr double asq = a * a;
    constexpr double wsq = w * w;
    constexpr double f = 1.0 / finv;
    constexpr double esq = 2.0 * f - f * f;
    constexpr double e = 0.08181919084262149;   // sqrt(esq)
    constexpr double Elin = a * e;              // Linear eccentricity
    constexpr double Elinsq = Elin * Elin;
    constexpr double epr = e / (1.0 - f);       // Second eccentricity
    // Pre-computed: q0 = ((1.0 + 3.0/(epr*epr))*atan(epr) - 3.0/epr)/2.0
    // Previous value 0.001001... was wrong (computed with e instead of epr)
    constexpr double q0 = 7.33462578708548e-05;
    // Pre-computed: U0 = -GM*atan(epr)/Elin - wsq*asq/3.0
    constexpr double U0 = -62636851.7149236;    // Reference potential
    constexpr double g0 = 9.80665;              // Standard gravity
    constexpr double GMdivElin = GM / Elin;

    // Centrifugal potential taper parameters
    constexpr double x0sq = 2.0e7 * 2.0e7;
    constexpr double Hsq = 1.2e7 * 1.2e7;

    // Working variables
    double altm = alt * 1000.0;
    double sinsqlat = std::sin(lat * DEG2RAD);
    sinsqlat = sinsqlat * sinsqlat;

    // Radius of curvature
    double v = a / std::sqrt(1.0 - esq * sinsqlat);

    // Squared Cartesian coordinates
    double xsq = (v + altm) * (v + altm) * (1.0 - sinsqlat);
    double zsq = (v * (1.0 - esq) + altm) * (v * (1.0 - esq) + altm) * sinsqlat;

    // Ellipsoidal coordinate u
    double rsqminElinsq = xsq + zsq - Elinsq;
    double usq = rsqminElinsq / 2.0 + std::sqrt(rsqminElinsq * rsqminElinsq / 4.0 + Elinsq * zsq);
    double cossqdelta = zsq / usq;

    // Gravitational potential
    double epru = Elin / std::sqrt(usq);
    double atanepru = std::atan(epru);
    double q = ((1.0 + 3.0 / (epru * epru)) * atanepru - 3.0 / epru) / 2.0;
    double U = -GMdivElin * atanepru - wsq * (asq * q * (cossqdelta - 1.0/3.0) / q0) / 2.0;

    // Centrifugal potential with taper
    double Vc;
    if (xsq <= x0sq) {
        Vc = (wsq / 2.0) * xsq;
    } else {
        Vc = (wsq / 2.0) * (Hsq * std::tanh((xsq - x0sq) / Hsq) + x0sq);
    }
    U = U - Vc;

    // Geopotential height (km)
    return (U - U0) / g0 / 1000.0;
}

// Dilogarithm Li_2(x) for x in [0,1), matching Fortran truncated 3rd-order approximation.
// Reference: Ginsberg & Zaborowski (1975), Commun. ACM 18, 200-202.
// Relative error < 1e-5 in domain [0,1).
inline double dilog(double x0) {
    constexpr double pi2_6 = PI * PI / 6.0;
    double x = x0;
    if (x > 0.5) {
        double lnx = std::log(x);
        x = 1.0 - x;  // Reflect into [0, 0.5]
        double xx = x * x;
        double x4 = 4.0 * x;
        return pi2_6 - lnx * std::log(x)
               - (4.0 * xx * (23.0/16.0 + x/36.0 + xx/576.0 + xx*x/3600.0)
                  + x4 + 3.0 * (1.0 - xx) * lnx) / (1.0 + x4 + xx);
    } else {
        double xx = x * x;
        double x4 = 4.0 * x;
        return (4.0 * xx * (23.0/16.0 + x/36.0 + xx/576.0 + xx*x/3600.0)
                + x4 + 3.0 * (1.0 - xx) * std::log(1.0 - x)) / (1.0 + x4 + xx);
    }
}

// B-spline evaluation result
struct BSplineResult {
    int iz;              // Spline index
    std::array<double, 6> S4;   // Order 4 spline weights
    std::array<double, 6> S5;   // Order 5 spline weights
    std::array<double, 6> S6;   // Order 6 spline weights
};

// B-spline evaluation at altitude z using precomputed eta
// Matches Fortran msis_utils.F90 bspline subroutine exactly
// Fortran uses S(-5:0, 2:6) and w(-4:0), we map:
//   Fortran S(l, k) -> C++ S[l+5] for k=6, S[l+4] for k=5, etc.
//   Fortran w(l) -> C++ w[l+4]
inline BSplineResult bspline_eval(double z, const double* nodes, int nd, const std::array<std::array<double, 30>, 5>& eta) {
    BSplineResult result;
    result.S4.fill(0.0);
    result.S5.fill(0.0);
    result.S6.fill(0.0);

    // Find spline index (same as Fortran binary search)
    if (z >= nodes[nd - 1]) {
        result.iz = nd - 1;
        return result;
    }
    if (z <= nodes[0]) {
        result.iz = -1;
        return result;
    }
    int low = 0, high = nd - 1;
    result.iz = (low + high) / 2;
    while (z < nodes[result.iz] || z >= nodes[result.iz + 1]) {
        if (z < nodes[result.iz]) {
            high = result.iz;
        } else {
            low = result.iz;
        }
        result.iz = (low + high) / 2;
    }

    const int i = result.iz;

    // Order 2 splines (k=2): S(-1:0) mapped to S2[0:1]
    std::array<double, 2> S2 = {};
    double w0 = (z - nodes[i]) * eta[0][i];
    S2[1] = w0;  // S(0,2)
    if (i > 0) S2[0] = 1.0 - w0;  // S(-1,2)
    if (i >= nd - 1) S2[1] = 0.0;

    // Order 3 splines (k=3): S(-2:0) mapped to S3[0:2]
    std::array<double, 3> S3 = {};
    std::array<double, 2> w3 = {};  // w(-1:0)
    w3[1] = (z - nodes[i]) * eta[1][i];  // w(0)
    if (i >= 1) w3[0] = (z - nodes[i-1]) * eta[1][i-1];  // w(-1)

    if (i < nd - 2) S3[2] = w3[1] * S2[1];  // S(0,3) = w(0)*S(0,2)
    if (i >= 1 && i - 1 < nd - 2) S3[1] = w3[0] * S2[0] + (1.0 - w3[1]) * S2[1];  // S(-1,3)
    if (i >= 2) S3[0] = (1.0 - w3[0]) * S2[0];  // S(-2,3)

    // Order 4 splines (k=4): S(-3:0) mapped to S4[0:3], then to result.S4[0:3]
    std::array<double, 4> S4 = {};
    std::array<double, 3> w4 = {};  // w(-2:0)
    for (int l = 0; l >= -2; l--) {
        int j = i + l;
        if (j >= 0 && j < nd - 2) w4[l + 2] = (z - nodes[j]) * eta[2][j];
    }

    if (i < nd - 3) S4[3] = w4[2] * S3[2];  // S(0,4) = w(0)*S(0,3)
    for (int l = -1; l >= -2; l--) {
        if (i + l >= 0 && i + l < nd - 3) {
            S4[l + 3] = w4[l + 2] * S3[l + 2] + (1.0 - w4[l + 3]) * S3[l + 3];
        }
    }
    if (i >= 3) S4[0] = (1.0 - w4[0]) * S3[0];  // S(-3,4)

    // Copy to result with proper offset for return
    for (int j = 0; j < 4; j++) result.S4[j] = S4[j];

    // Order 5 splines (k=5): S(-4:0) mapped to S5[0:4]
    std::array<double, 5> S5 = {};
    std::array<double, 4> w5 = {};  // w(-3:0)
    for (int l = 0; l >= -3; l--) {
        int j = i + l;
        if (j >= 0 && j < nd - 3) w5[l + 3] = (z - nodes[j]) * eta[3][j];
    }

    if (i < nd - 4) S5[4] = w5[3] * S4[3];  // S(0,5) = w(0)*S(0,4)
    for (int l = -1; l >= -3; l--) {
        if (i + l >= 0 && i + l < nd - 4) {
            S5[l + 4] = w5[l + 3] * S4[l + 3] + (1.0 - w5[l + 4]) * S4[l + 4];
        }
    }
    if (i >= 4) S5[0] = (1.0 - w5[0]) * S4[0];  // S(-4,5)

    // Copy to result
    for (int j = 0; j < 5; j++) result.S5[j] = S5[j];

    // Order 6 splines (k=6): S(-5:0) mapped to S6[0:5]
    std::array<double, 5> w6 = {};  // w(-4:0)
    for (int l = 0; l >= -4; l--) {
        int j = i + l;
        if (j >= 0 && j < nd - 4) w6[l + 4] = (z - nodes[j]) * eta[4][j];
    }

    if (i < nd - 5) result.S6[5] = w6[4] * S5[4];  // S(0,6) = w(0)*S(0,5)
    for (int l = -1; l >= -4; l--) {
        if (i + l >= 0 && i + l < nd - 5) {
            result.S6[l + 5] = w6[l + 4] * S5[l + 4] + (1.0 - w6[l + 5]) * S5[l + 5];
        }
    }
    if (i >= 5) result.S6[0] = (1.0 - w6[0]) * S5[0];  // S(-5,6)

    return result;
}

}  // namespace utils

}  // namespace msis21

// =============================================================================
// Temperature profile parameters
// =============================================================================
struct TnParm {
    std::array<double, 24> cf;  // Spline coefficients (1/T)
    double tzetaF;    // Temperature at zetaF (70 km)
    double tzetaA;    // Temperature at zetaA (85 km)
    double dlntdzA;   // d(ln T)/dz at zetaA
    double tex;       // Exospheric temperature
    double tgb0;      // Temperature gradient at zetaB
    double tb0;       // Temperature at zetaB (122.5 km)
    double sigma;     // Shape factor
    double sigmasq;   // sigma^2
    double b;         // 1 - tb0/tex
    std::array<double, 24> beta;   // 1st integration coefficients
    std::array<double, 24> gamma;  // 2nd integration coefficients
    double cVS, cVB, cWS, cWB;     // Integration constants
    double VzetaF, VzetaA, WzetaA, Vzeta0;  // Integral values at reference heights
    double lndtotF;   // ln(total number density) at zetaF
};

// =============================================================================
// Density profile parameters (from msis_dfn.F90 dnparm type)
// =============================================================================
struct DnParm {
    int ispec;                        // Species index
    double lnPhiF;                    // Log mixing ratio at zetaF
    double lndref;                    // Log reference number density
    double zref;                      // Reference height for hydrostatic integral
    double zmin;                      // Minimum altitude for this species
    double zhyd;                      // Altitude above which hydrostatic terms needed

    // Piecewise effective mass profile
    double zetaM;                     // Turbopause height
    double HML;                       // Lower scale height of mass transition
    double HMU;                       // Upper scale height of mass transition
    std::array<double, 5> zetaMi;     // Height nodes of mass profile
    std::array<double, 5> Mi;         // Effective mass at each node
    std::array<double, 4> aMi;        // Slopes of mass profile segments
    std::array<double, 5> WMi;        // 2nd integral of 1/T at mass nodes
    std::array<double, 5> XMi;        // Cumulative mass adjustment terms

    // Chapman and logistic correction parameters
    double C;                         // Chapman coefficient
    double zetaC;                     // Chapman reference height
    double HC;                        // Chapman scale height
    double R;                         // Logistic (dynamical) coefficient
    double zetaR;                     // Logistic reference height
    double HR;                        // Logistic scale height

    // Spline coefficients for chemistry-dominated species (O1, NO)
    std::array<double, 10> cf;        // cf[0:7] unconstrained, cf[8:9] C1-constrained

    // Integration terms
    double Mzref;                     // Effective mass at reference height
    double Izref;                     // Hydrostatic integral at reference height
    double Tref;                      // Temperature at reference height
};

// =============================================================================
// Main NRLMSIS 2.1 Class
// =============================================================================
class NRLMSIS21 {
    // Allow debug test access to internal computation functions
    friend class NRLMSIS21_Debug;

public:
    // Input structure
    struct Input {
        double day;        // Day of year (1-366)
        double utsec;      // UT seconds
        double alt;        // Geodetic altitude (km)
        double lat;        // Geodetic latitude (deg)
        double lon;        // Geodetic longitude (deg)
        double f107a;      // 81-day average F10.7
        double f107;       // Daily F10.7
        std::array<double, 7> ap;  // Geomagnetic indices
    };

    // Output structure
    struct Output {
        double tn;         // Temperature at altitude (K)
        double tex;        // Exospheric temperature (K)
        std::array<double, 11> dn;  // Number densities (m^-3) [1-based indexing]
        // dn[1] = total mass density (kg/m^3)
        // dn[2] = N2, dn[3] = O2, dn[4] = O, dn[5] = He
        // dn[6] = H, dn[7] = Ar, dn[8] = N, dn[9] = anomalous O, dn[10] = NO
    };

    NRLMSIS21();

    // Main calculation function
    Output msiscalc(const Input& input) const;

    // Convenience function for drag calculations
    double density(double alt_km, double lat_deg, double lon_deg,
                   const time::Epoch& epoch, const WeatherConditions& weather) const;

    // Full output including temperature and species densities
    // Used by Sentman Cd model for speed ratio computation
    Output full_output(double alt_km, double lat_deg, double lon_deg,
                       const time::Epoch& epoch, const WeatherConditions& weather) const;

private:
    // Precomputed spline eta values
    std::array<std::array<double, 30>, 5> etaTN_;
    std::array<std::array<double, 30>, 3> etaO1_;  // Orders 2-4 for O1 splines
    std::array<std::array<double, 30>, 3> etaNO_;  // Orders 2-4 for NO splines

    // Solar flux modulation flags
    std::array<bool, 24> smod_;      // Which temperature levels get sfluxmod
    std::array<bool, 384> zsfx_;     // Intra-annual terms modulated by F1
    std::array<bool, 384> tsfx_;     // Tidal terms modulated by F2
    std::array<bool, 384> psfx_;     // SPW terms modulated by F3

    // Precomputed HRfact values at reference heights
    double HRfactO1ref_, dHRfactO1ref_;
    double HRfactNOref_, dHRfactNOref_;

    // Compute horizontal basis functions
    void compute_globe(const Input& input, std::array<double, 512>& bf) const;

    // Nonlinear modulation functions
    template<std::size_t N>
    double sfluxmod(int iz, const std::array<double, 512>& gf,
                    const std::array<std::array<double, N>, 512>& beta,
                    double dffact) const;
    double sfluxmod_tn(int iz, const std::array<double, 512>& gf, double dffact) const;
    static double solzen(double doy, double lst, double lat, double lon);

    // Compute temperature profile parameters
    TnParm compute_tfnparm(const std::array<double, 512>& bf) const;

    // Compute density profile parameters for a species
    DnParm compute_dfnparm(int ispec, const std::array<double, 512>& bf, const TnParm& tpro) const;

    // Compute density at altitude for a species
    double compute_dfnx(double zeta, double tn, double lndtotz, double Vz, double Wz, double HRfact,
                        const TnParm& tpro, const DnParm& dpro) const;

    // Piecewise effective mass interpolation
    double pwmp(double z, const DnParm& dpro) const;

    // Compute temperature at altitude
    double compute_temperature(double zeta, const TnParm& tpro, int iz, const std::array<double, 6>& S4) const;

    // Compute 2nd integral W at altitude
    double compute_Wz(double zeta, const TnParm& tpro, int iz, const std::array<double, 6>& S6) const;

    // B-spline evaluation for O1/NO (order 4 only)
    struct BSplineResult4 {
        int iz;
        std::array<double, 4> S4;  // Order 4 weights
    };
    BSplineResult4 bspline_eval_o4(double z, const double* nodes, int nd,
                                    const std::array<std::array<double, 30>, 3>& eta) const;
};

// =============================================================================
// Implementation
// =============================================================================

inline NRLMSIS21::NRLMSIS21() {
    using namespace msis21::constants;
    using namespace msis21::parm;

    // Precompute eta values for TN B-spline evaluation (orders 2-6)
    for (auto& a : etaTN_) a.fill(0.0);
    for (int k = 0; k < 5; k++) {
        for (int i = 0; i <= NL; i++) {
            double denom = NODES_TN[i + k + 1] - NODES_TN[i];
            etaTN_[k][i] = (denom > 0.0) ? 1.0 / denom : 0.0;
        }
    }

    // Precompute eta values for O1 B-spline evaluation (orders 2-4)
    for (auto& a : etaO1_) a.fill(0.0);
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j <= NDO1 - k - 2; j++) {
            double denom = NODES_O1[j + k + 1] - NODES_O1[j];
            etaO1_[k][j] = (denom > 0.0) ? 1.0 / denom : 0.0;
        }
    }

    // Precompute eta values for NO B-spline evaluation (orders 2-4)
    for (auto& a : etaNO_) a.fill(0.0);
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j <= NDNO - k - 2; j++) {
            double denom = NODES_NO[j + k + 1] - NODES_NO[j];
            etaNO_[k][j] = (denom > 0.0) ? 1.0 / denom : 0.0;
        }
    }

    // Solar flux modulation flags (from msis_init.F90 initparmspace)
    zsfx_.fill(false);
    tsfx_.fill(false);
    psfx_.fill(false);
    // F1: solar flux modulation of zonal mean asymmetric annual terms
    zsfx_[9] = zsfx_[10] = true;    // Pl(1,0) annual
    zsfx_[13] = zsfx_[14] = true;   // Pl(3,0) annual
    zsfx_[17] = zsfx_[18] = true;   // Pl(5,0) annual
    // F2: solar flux modulation of tides
    for (int i = CTIDE; i < CSPW && i < 384; i++) tsfx_[i] = true;
    // F3: solar flux modulation of stationary planetary wave 1
    for (int i = CSPW; i < CSPW + 60 && i < 384; i++) psfx_[i] = true;

    // Set smod flags from TN parameter data
    // smod[ix] is true if any of the csfxmod beta values are nonzero
    smod_.fill(false);
    for (int ix = 0; ix <= NL; ix++) {
        if (TN_BETA[CSFXMOD][ix] != 0.0 ||
            TN_BETA[CSFXMOD + 1][ix] != 0.0 ||
            TN_BETA[CSFXMOD + 2][ix] != 0.0) {
            smod_[ix] = true;
        }
    }

    // Precompute HRfact values at O1 and NO reference heights (from msis_init.F90)
    double gammaterm0 = std::tanh((ZETAREF_O1 - ZETA_GAMMA) * H_GAMMA);
    HRfactO1ref_ = 0.5 * (1.0 + gammaterm0);
    dHRfactO1ref_ = (1.0 - (ZETAREF_O1 - ZETA_GAMMA) * (1.0 - gammaterm0) * H_GAMMA) / HRfactO1ref_;

    gammaterm0 = std::tanh((ZETAREF_NO - ZETA_GAMMA) * H_GAMMA);
    HRfactNOref_ = 0.5 * (1.0 + gammaterm0);
    dHRfactNOref_ = (1.0 - (ZETAREF_NO - ZETA_GAMMA) * (1.0 - gammaterm0) * H_GAMMA) / HRfactNOref_;
}

// Solar zenith angle (adapted from IRI, matches Fortran msis_gfn.F90)
inline double NRLMSIS21::solzen(double ddd, double lst, double lat, double lon) {
    using namespace msis21::constants;
    constexpr double humr = PI / 12.0;
    constexpr double p[5] = {0.017203534, 0.034407068, 0.051610602, 0.068814136, 0.103221204};

    double teqnx = ddd + 0.9369;

    double dec = 23.256 * std::sin(p[0] * (teqnx - 82.242)) + 0.381 * std::sin(p[1] * (teqnx - 44.855))
               + 0.167 * std::sin(p[2] * (teqnx - 23.355)) - 0.013 * std::sin(p[3] * (teqnx + 11.97))
               + 0.011 * std::sin(p[4] * (teqnx - 10.410)) + 0.339137;
    dec *= DEG2RAD;

    double tf = teqnx - 0.5;
    double teqt = -7.38 * std::sin(p[0] * (tf - 4.0)) - 9.87 * std::sin(p[1] * (tf + 9.0))
                + 0.27 * std::sin(p[2] * (tf - 53.0)) - 0.2 * std::cos(p[3] * (tf - 17.0));

    double phi = humr * (lst - 12.0) + teqt * DEG2RAD / 4.0;
    double rlat = lat * DEG2RAD;
    double cosx = std::sin(rlat) * std::sin(dec) + std::cos(rlat) * std::cos(dec) * std::cos(phi);
    if (std::abs(cosx) > 1.0) cosx = (cosx > 0) ? 1.0 : -1.0;
    return std::acos(cosx) / DEG2RAD;
}

// Solar flux modulation for species BETA arrays (O1, HE, H1, N1, NO)
template<std::size_t N>
inline double NRLMSIS21::sfluxmod(int iz, const std::array<double, 512>& gf,
                                   const std::array<std::array<double, N>, 512>& beta,
                                   double dffact) const {
    using namespace msis21::constants;

    // F1: intra-annual modulation factor
    double f1 = beta[CSFXMOD][iz] * gf[CSFXMOD]
              + (beta[CSFX + 2][iz] * gf[CSFXMOD + 2] + beta[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;

    // F2: tidal modulation factor
    double f2 = beta[CSFXMOD + 1][iz] * gf[CSFXMOD]
              + (beta[CSFX + 2][iz] * gf[CSFXMOD + 2] + beta[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;

    // F3: SPW modulation factor
    double f3 = beta[CSFXMOD + 2][iz] * gf[CSFXMOD];

    double sum = 0.0;
    for (int j = 0; j <= MBF; j++) {
        if (zsfx_[j]) {
            sum += beta[j][iz] * gf[j] * f1;
        } else if (tsfx_[j]) {
            sum += beta[j][iz] * gf[j] * f2;
        } else if (psfx_[j]) {
            sum += beta[j][iz] * gf[j] * f3;
        }
    }
    return sum;
}

// Solar flux modulation for TN BETA array (24 levels)
inline double NRLMSIS21::sfluxmod_tn(int iz, const std::array<double, 512>& gf, double dffact) const {
    using namespace msis21::constants;
    using namespace msis21::parm;

    double f1 = TN_BETA[CSFXMOD][iz] * gf[CSFXMOD]
              + (TN_BETA[CSFX + 2][iz] * gf[CSFXMOD + 2] + TN_BETA[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;
    double f2 = TN_BETA[CSFXMOD + 1][iz] * gf[CSFXMOD]
              + (TN_BETA[CSFX + 2][iz] * gf[CSFXMOD + 2] + TN_BETA[CSFX + 3][iz] * gf[CSFXMOD + 3]) * dffact;
    double f3 = TN_BETA[CSFXMOD + 2][iz] * gf[CSFXMOD];

    double sum = 0.0;
    for (int j = 0; j <= MBF; j++) {
        if (zsfx_[j]) {
            sum += TN_BETA[j][iz] * gf[j] * f1;
        } else if (tsfx_[j]) {
            sum += TN_BETA[j][iz] * gf[j] * f2;
        } else if (psfx_[j]) {
            sum += TN_BETA[j][iz] * gf[j] * f3;
        }
    }
    return sum;
}

// Geomagnetic activity function (from msis_gfn.F90)
// Template version that properly extracts column values from 2D beta array
// beta: 2D array of coefficients [MAXNBF][N], level: vertical level index
// bf: full basis function array (bf[CMAG:CMAG+12] used for ap, phases; bf[CMAG+13:CMAG+26] for Plg)
template<std::size_t N>
double geomag_impl(const std::array<std::array<double, N>, 512>& beta, int level,
                   const std::array<double, 512>& bf) {
    using namespace msis21::constants;

    // Extract geomag parameters for this level: p[i] = beta[CMAG+i][level]
    // p[0] = k00r, p[1] = k00s, p[2:25] = spatial/temporal coefficients
    double k00r = beta[CMAG][level];
    double k00s = beta[CMAG + 1][level];

    // Both master switches assumed on (daily Ap mode by default)
    if (k00s == 0.0) return 0.0;  // k00s == 0

    // G0fn helper - geomagnetic response function
    auto G0fn = [](double a, double k00r, double k00s) -> double {
        return a + (k00r - 1.0) * (a + (std::exp(-a * k00s) - 1.0) / k00s);
    };

    double delA = G0fn(bf[CMAG], k00r, k00s);  // bf[CMAG] = ap[0] - 4

    // Plg values are stored at bf[CMAG+13:CMAG+26]
    // Column 0 (zonal): P(0,0)..P(6,0) at indices 13-19
    // Column 1 (tesseral): P(0,1)..P(6,1) at indices 20-26
    const double* plg0 = &bf[CMAG + 13];  // plg(0:6, 0)
    const double* plg1 = &bf[CMAG + 20];  // plg(0:6, 1)

    // Extract remaining parameters
    double p2 = beta[CMAG + 2][level];
    double p3 = beta[CMAG + 3][level];
    double p4 = beta[CMAG + 4][level];
    double p5 = beta[CMAG + 5][level];
    double p6 = beta[CMAG + 6][level];
    double p7 = beta[CMAG + 7][level];
    double p8 = beta[CMAG + 8][level];  // doy phase
    double p9 = beta[CMAG + 9][level];
    double p10 = beta[CMAG + 10][level];
    double p11 = beta[CMAG + 11][level];
    double p12 = beta[CMAG + 12][level];  // lst phase
    double p13 = beta[CMAG + 13][level];
    double p14 = beta[CMAG + 14][level];
    double p15 = beta[CMAG + 15][level];
    double p16 = beta[CMAG + 16][level];
    double p17 = beta[CMAG + 17][level];  // lon phase
    double p18 = beta[CMAG + 18][level];
    double p19 = beta[CMAG + 19][level];
    double p20 = beta[CMAG + 20][level];
    double p21 = beta[CMAG + 21][level];
    double p22 = beta[CMAG + 22][level];
    double p23 = beta[CMAG + 23][level];
    double p24 = beta[CMAG + 24][level];
    double p25 = beta[CMAG + 25][level];  // UT phase

    // bf[CMAG+8] = doy_rad, bf[CMAG+9] = lst_rad, bf[CMAG+10] = lon_rad, bf[CMAG+11] = ut_rad
    double doy_rad = bf[CMAG + 8];
    double lst_rad = bf[CMAG + 9];
    double lon_rad = bf[CMAG + 10];
    double ut_rad = bf[CMAG + 11];

    double result = (p2 * plg0[0] + p3 * plg0[2] + p4 * plg0[4]                              // time independent
        + (p5 * plg0[1] + p6 * plg0[3] + p7 * plg0[5]) * std::cos(doy_rad - p8)              // doy modulation
        + (p9 * plg1[1] + p10 * plg1[3] + p11 * plg1[5]) * std::cos(lst_rad - p12)           // lst modulation
        + (1.0 + p13 * plg0[1]) *
          (p14 * plg1[2] + p15 * plg1[4] + p16 * plg1[6]) * std::cos(lon_rad - p17)          // longitude
        + (p18 * plg1[1] + p19 * plg1[3] + p20 * plg1[5]) * std::cos(lon_rad - p21)
          * std::cos(doy_rad - p8)                                                            // lon+doy
        + (p22 * plg0[1] + p23 * plg0[3] + p24 * plg0[5]) * std::cos(ut_rad - p25))          // UT
        * delA;

    return result;
}

// UT dependence function (from msis_gfn.F90)
// Template version that properly extracts column values from 2D beta array
template<std::size_t N>
double utdep_impl(const std::array<std::array<double, N>, 512>& beta, int level,
                  const std::array<double, 512>& bf) {
    using namespace msis21::constants;

    // Extract UT parameters for this level: p[i] = beta[CUT+i][level]
    double p0 = beta[CUT][level];
    double p1 = beta[CUT + 1][level];
    double p2 = beta[CUT + 2][level];
    double p3 = beta[CUT + 3][level];
    double p4 = beta[CUT + 4][level];
    double p5 = beta[CUT + 5][level];
    double p6 = beta[CUT + 6][level];
    double p7 = beta[CUT + 7][level];
    double p8 = beta[CUT + 8][level];
    double p9 = beta[CUT + 9][level];
    double p10 = beta[CUT + 10][level];
    double p11 = beta[CUT + 11][level];

    // bf[CUT:CUT+8] = UT-related basis functions
    double ut_rad = bf[CUT];         // bf(cut) = lst2rad*utsec/3600
    double doy_rad = bf[CUT + 1];    // bf(cut+1) = doy2rad*doy
    double dfa = bf[CUT + 2];        // bf(cut+2) = dfa (F10.7 avg deviation)
    double lon_rad = bf[CUT + 3];    // bf(cut+3) = deg2rad*lon
    double plg10 = bf[CUT + 4];      // bf(cut+4) = P10 = sin(lat)
    double plg30 = bf[CUT + 5];      // bf(cut+5) = P30
    double plg50 = bf[CUT + 6];      // bf(cut+6) = P50
    double plg32 = bf[CUT + 7];      // bf(cut+7) = P32
    double plg52 = bf[CUT + 8];      // bf(cut+8) = P52

    return std::cos(ut_rad - p0) *
           (1.0 + p3 * plg10 * std::cos(doy_rad - p1)) *
           (1.0 + p4 * dfa) * (1.0 + p5 * plg10) *
           (p6 * plg10 + p7 * plg30 + p8 * plg50) +
           std::cos(ut_rad - p2 + 2.0 * lon_rad) * (p9 * plg32 + p10 * plg52) * (1.0 + p11 * dfa);
}

// B-spline evaluation for order 4 only (for O1 and NO chemistry-region splines)
inline NRLMSIS21::BSplineResult4 NRLMSIS21::bspline_eval_o4(
    double z, const double* nodes, int nd,
    const std::array<std::array<double, 30>, 3>& eta) const {

    BSplineResult4 result;
    result.S4.fill(0.0);

    // Find spline index (binary search style matching Fortran)
    result.iz = 0;
    for (int i = 0; i < nd - 1; i++) {
        if (z >= nodes[i]) result.iz = i;
    }
    int iz = result.iz;

    // Order 2 (k=2): eta index 0
    double S2[4] = {};  // indices mapped as: S2[j+1] for j in {-1,0,1,2} -> idx 0,1,2,3
    double w2 = (z - nodes[iz]) * eta[0][iz];
    S2[2] = w2;          // S(0,2)
    S2[1] = 1.0 - w2;    // S(-1,2)

    // Order 3 (k=3): eta index 1
    double S3[4] = {};
    for (int l = 0; l >= -1; l--) {
        int j = iz + l;
        if (j >= 0 && j <= nd - 3) {
            double wl = (z - nodes[j]) * eta[1][j];
            int idx = l + 2;  // map l to array index
            if (l == 0) {
                S3[idx] = wl * S2[2];
            }
            if (l == -1) {
                S3[idx] = wl * S2[1] + (1.0 - ((z - nodes[iz]) * eta[1][iz])) * S2[2];
            }
        }
    }
    // S(-2,3)
    if (iz - 2 >= 0 && iz - 1 <= nd - 3) {
        double wm1 = (z - nodes[iz - 1]) * eta[1][iz - 1];
        S3[0] = (1.0 - wm1) * S2[1];
    }

    // Order 4 (k=4) using the recursive formula
    // S(l, k) = w(iz+l, k) * S(l, k-1) + (1-w(iz+l+1, k)) * S(l+1, k-1)
    // where w(j, k) = (z - nodes[j]) * eta[j, k]
    // For k=4, l ranges from -3 to 0
    result.S4.fill(0.0);
    for (int l = 0; l >= -3; l--) {
        int j = iz + l;
        double wl = 0.0;
        if (j >= 0 && j <= nd - 4) wl = (z - nodes[j]) * eta[2][j];

        double wl1 = 0.0;  // w at j+1 = iz+l+1
        int j1 = iz + l + 1;
        if (j1 >= 0 && j1 <= nd - 4) wl1 = (z - nodes[j1]) * eta[2][j1];

        // S3 values: S3[0]=S(-2,3), S3[1]=S(-1,3), S3[2]=S(0,3)
        // S3 index for l is l+2
        int s3_l = l + 2;
        int s3_l1 = l + 3;
        double s3_val = (s3_l >= 0 && s3_l < 4) ? S3[s3_l] : 0.0;
        double s3_val1 = (s3_l1 >= 0 && s3_l1 < 4) ? S3[s3_l1] : 0.0;

        // S4 index: l maps to l+3 (l=0->3, l=-1->2, l=-2->1, l=-3->0)
        int s4_idx = l + 3;
        if (s4_idx >= 0 && s4_idx < 4) {
            result.S4[s4_idx] = wl * s3_val + (1.0 - wl1) * s3_val1;
        }
    }

    return result;
}

inline void NRLMSIS21::compute_globe(const Input& input, std::array<double, 512>& bf) const {
    using namespace msis21::constants;

    bf.fill(0.0);

    double lat_rad = input.lat * DEG2RAD;
    double slat = std::sin(lat_rad);
    double clat = std::cos(lat_rad);

    // Legendre polynomials P_n^m(sin(lat))
    // IMPORTANT: Following Fortran convention where the argument is sin(lat) (= cos(colat))
    // clat = sin(lat) (the "co-latitude cosine"), slat = cos(lat) (the "co-latitude sine")
    // This matches Fortran: clat = sin(lat*deg2rad), slat = cos(lat*deg2rad)
    double clat_leg = slat;   // sin(lat) - argument to Legendre polynomials
    double slat_leg = clat;   // cos(lat) - sqrt(1-x^2) factor
    double clat2 = clat_leg * clat_leg;
    double clat4 = clat2 * clat2;
    double slat2 = slat_leg * slat_leg;

    std::array<std::array<double, 4>, 7> plg = {};
    // P(n, 0) - zonal harmonics
    plg[0][0] = 1.0;
    plg[1][0] = clat_leg;
    plg[2][0] = 0.5 * (3.0 * clat2 - 1.0);
    plg[3][0] = 0.5 * (5.0 * clat_leg * clat2 - 3.0 * clat_leg);
    plg[4][0] = (35.0 * clat4 - 30.0 * clat2 + 3.0) / 8.0;
    plg[5][0] = (63.0 * clat2 * clat2 * clat_leg - 70.0 * clat2 * clat_leg + 15.0 * clat_leg) / 8.0;
    plg[6][0] = (11.0 * clat_leg * plg[5][0] - 5.0 * plg[4][0]) / 6.0;

    // P(n, 1) - tesseral m=1
    plg[1][1] = slat_leg;
    plg[2][1] = 3.0 * clat_leg * slat_leg;
    plg[3][1] = 1.5 * (5.0 * clat2 - 1.0) * slat_leg;
    plg[4][1] = 2.5 * (7.0 * clat2 * clat_leg - 3.0 * clat_leg) * slat_leg;
    plg[5][1] = 1.875 * (21.0 * clat4 - 14.0 * clat2 + 1.0) * slat_leg;
    plg[6][1] = (11.0 * clat_leg * plg[5][1] - 6.0 * plg[4][1]) / 5.0;

    // P(n, 2) - sectoral/tesseral m=2
    plg[2][2] = 3.0 * slat2;
    plg[3][2] = 15.0 * slat2 * clat_leg;
    plg[4][2] = 7.5 * (7.0 * clat2 - 1.0) * slat2;
    plg[5][2] = 3.0 * clat_leg * plg[4][2] - 2.0 * plg[3][2];
    plg[6][2] = (11.0 * clat_leg * plg[5][2] - 7.0 * plg[4][2]) / 4.0;

    // P(n, 3) - m=3
    plg[3][3] = 15.0 * slat2 * slat_leg;
    plg[4][3] = 105.0 * slat2 * slat_leg * clat_leg;
    plg[5][3] = (9.0 * clat_leg * plg[4][3] - 7.0 * plg[3][3]) / 2.0;
    plg[6][3] = (11.0 * clat_leg * plg[5][3] - 8.0 * plg[4][3]) / 3.0;

    // Local solar time
    double lst = input.utsec / 3600.0 + input.lon / 15.0;
    lst = std::fmod(lst, 24.0);
    if (lst < 0.0) lst += 24.0;

    // Day of year harmonics
    double cdoy1 = std::cos(DOY2RAD * input.day);
    double sdoy1 = std::sin(DOY2RAD * input.day);
    double cdoy2 = std::cos(DOY2RAD * input.day * 2.0);
    double sdoy2 = std::sin(DOY2RAD * input.day * 2.0);

    // Local time harmonics
    double clst1 = std::cos(LST2RAD * lst);
    double slst1 = std::sin(LST2RAD * lst);
    double clst2 = std::cos(LST2RAD * lst * 2.0);
    double slst2 = std::sin(LST2RAD * lst * 2.0);
    double clst3 = std::cos(LST2RAD * lst * 3.0);
    double slst3 = std::sin(LST2RAD * lst * 3.0);

    // Longitude harmonics
    double clon1 = std::cos(DEG2RAD * input.lon);
    double slon1 = std::sin(DEG2RAD * input.lon);
    double clon2 = std::cos(DEG2RAD * input.lon * 2.0);
    double slon2 = std::sin(DEG2RAD * input.lon * 2.0);

    // Time-independent terms (latitude only)
    int c = CTIMEIND;
    for (int n = 0; n <= MAXN; n++) {
        bf[c++] = plg[n][0];
    }

    // Intra-annual (annual and semiannual)
    std::array<double, 2> cdoy = {cdoy1, cdoy2};
    std::array<double, 2> sdoy = {sdoy1, sdoy2};
    for (int s = 0; s < 2; s++) {
        for (int n = 0; n <= MAXN; n++) {
            bf[c++] = plg[n][0] * cdoy[s];
            bf[c++] = plg[n][0] * sdoy[s];
        }
    }

    // Tidal terms (local time)
    std::array<double, 3> clst = {clst1, clst2, clst3};
    std::array<double, 3> slst = {slst1, slst2, slst3};
    for (int l = 1; l <= MAXL; l++) {
        for (int n = l; n <= MAXN; n++) {
            bf[c++] = plg[n][l] * clst[l-1];
            bf[c++] = plg[n][l] * slst[l-1];
        }
        // Annual modulation of tides
        for (int s = 0; s < 2; s++) {
            for (int n = l; n <= MAXN; n++) {
                bf[c++] = plg[n][l] * clst[l-1] * cdoy[s];
                bf[c++] = plg[n][l] * slst[l-1] * cdoy[s];
                bf[c++] = plg[n][l] * clst[l-1] * sdoy[s];
                bf[c++] = plg[n][l] * slst[l-1] * sdoy[s];
            }
        }
    }

    // Stationary planetary waves (longitude)
    std::array<double, 2> clon = {clon1, clon2};
    std::array<double, 2> slon = {slon1, slon2};
    for (int m = 1; m <= MAXM; m++) {
        for (int n = m; n <= MAXN; n++) {
            bf[c++] = plg[n][m] * clon[m-1];
            bf[c++] = plg[n][m] * slon[m-1];
        }
        // Annual modulation of SPW
        for (int s = 0; s < 2; s++) {
            for (int n = m; n <= MAXN; n++) {
                bf[c++] = plg[n][m] * clon[m-1] * cdoy[s];
                bf[c++] = plg[n][m] * slon[m-1] * cdoy[s];
                bf[c++] = plg[n][m] * clon[m-1] * sdoy[s];
                bf[c++] = plg[n][m] * slon[m-1] * sdoy[s];
            }
        }
    }

    // Linear solar flux terms (indices CSFX to CSFX+NSFX-1 = 295-299)
    double dfa = input.f107a - 150.0;
    double df = input.f107 - input.f107a;
    bf[CSFX] = dfa;
    bf[CSFX + 1] = dfa * dfa;
    bf[CSFX + 2] = df;
    bf[CSFX + 3] = df * df;
    bf[CSFX + 4] = df * dfa;

    // Extra linear terms (indices CEXTRA = 300-314)
    // Matches Fortran msis_gfn.F90 lines 228-248
    double sza = solzen(input.day, lst, input.lat, input.lon);  // degrees
    bf[300] = -0.5 * std::tanh((sza - 98.0) / 6.0);            // SZA logistic for O, H
    bf[301] = -0.5 * std::tanh((sza - 101.5) / 20.0);          // SZA logistic for NO
    bf[302] = dfa * bf[300];                                     // F10.7 mod of SZA O,H
    bf[303] = dfa * bf[301];                                     // F10.7 mod of SZA NO
    bf[304] = dfa * plg[2][0];                                   // F10.7 mod of P(2,0)
    bf[305] = dfa * plg[4][0];                                   // F10.7 mod of P(4,0)
    bf[306] = dfa * plg[0][0] * cdoy1;                           // F10.7 mod of global AO cos
    bf[307] = dfa * plg[0][0] * sdoy1;                           // F10.7 mod of global AO sin
    bf[308] = dfa * plg[0][0] * cdoy2;                           // F10.7 mod of global SAO cos
    bf[309] = dfa * plg[0][0] * sdoy2;                           // F10.7 mod of global SAO sin
    // Truncated quadratic F10.7a function (cutoff at F10.7a = 150)
    double sfluxavg_quad_cutoff = 150.0;
    double sfluxavgref = 150.0;
    double dfa_trunc;
    if (input.f107a <= sfluxavg_quad_cutoff) {
        dfa_trunc = dfa * dfa;
    } else {
        dfa_trunc = (sfluxavg_quad_cutoff - sfluxavgref) * (2.0 * dfa - (sfluxavg_quad_cutoff - sfluxavgref));
    }
    bf[310] = dfa_trunc;                                         // Truncated quad F10.7a
    bf[311] = dfa_trunc * plg[2][0];                             // P(2,0) mod of truncated quad
    bf[312] = dfa_trunc * plg[4][0];                             // P(4,0) mod of truncated quad
    bf[313] = df * plg[2][0];                                    // P(2,0) mod of dF10.7
    bf[314] = df * plg[4][0];                                    // P(4,0) mod of dF10.7

    // Nonlinear solar flux modulation terms (indices CSFXMOD = 384-388)
    // gf[384] = dF10.7a
    // gf[385] = dF10.7a² (capped at 0 for > 150)
    // gf[386] = dF10.7
    // gf[387] = dF10.7a * dF10.7
    // gf[388] = dF10.7²
    bf[CSFXMOD] = dfa;
    bf[CSFXMOD + 1] = dfa * dfa;
    bf[CSFXMOD + 2] = df;
    bf[CSFXMOD + 3] = df * df;
    bf[CSFXMOD + 4] = df * dfa;

    // Geomagnetic terms (indices CMAG = 389-442)
    // Matches Fortran: bf(cmag:cmag+6) = ap(1:7) - 4.0
    for (int i = 0; i < 7; i++) {
        bf[CMAG + i] = input.ap[i] - 4.0;
    }
    // bf(cmag+7) is 0 (gap between ap values and phase terms)
    // (already 0 from bf.fill(0.0))

    // Phase and auxiliary terms (indices cmag+8:cmag+12)
    double doy_rad = DOY2RAD * input.day;
    double lon_rad = DEG2RAD * input.lon;
    double ut_rad = LST2RAD * (input.utsec / 3600.0);
    bf[CMAG + 8] = doy_rad;
    bf[CMAG + 9] = LST2RAD * lst;
    bf[CMAG + 10] = lon_rad;
    bf[CMAG + 11] = ut_rad;
    bf[CMAG + 12] = std::abs(input.lat);

    // Plg values for geomag parametrization (indices cmag+13:cmag+26)
    // Stored in Fortran column-major order: plg(0:6, 0:1)
    // Column 0 (zonal): P(0,0)..P(6,0)
    bf[CMAG + 13] = plg[0][0];  // plg(0,0) = P00 = 1.0
    bf[CMAG + 14] = plg[1][0];  // plg(1,0) = P10
    bf[CMAG + 15] = plg[2][0];  // plg(2,0) = P20
    bf[CMAG + 16] = plg[3][0];  // plg(3,0) = P30
    bf[CMAG + 17] = plg[4][0];  // plg(4,0) = P40
    bf[CMAG + 18] = plg[5][0];  // plg(5,0) = P50
    bf[CMAG + 19] = plg[6][0];  // plg(6,0) = P60
    // Column 1 (tesseral): P(0,1)..P(6,1)
    bf[CMAG + 20] = 0.0;        // plg(0,1) = P01 = 0 (by definition)
    bf[CMAG + 21] = plg[1][1];  // plg(1,1) = P11
    bf[CMAG + 22] = plg[2][1];  // plg(2,1) = P21
    bf[CMAG + 23] = plg[3][1];  // plg(3,1) = P31
    bf[CMAG + 24] = plg[4][1];  // plg(4,1) = P41
    bf[CMAG + 25] = plg[5][1];  // plg(5,1) = P51
    bf[CMAG + 26] = plg[6][1];  // plg(6,1) = P61

    // Storm-time mode terms (indices cmag+27:cmag+53) - zeroed for daily Ap mode
    // (kept as 0 from bf.fill(0.0))

    // UT terms (indices CUT = 443-451) - matches Fortran msis_gfn.F90 lines 282-291
    bf[CUT]     = ut_rad;               // lst2rad*utsec/3600.0
    bf[CUT + 1] = doy_rad;              // doy2rad*doy
    bf[CUT + 2] = dfa;                  // dfa (F10.7 average deviation)
    bf[CUT + 3] = lon_rad;              // deg2rad*lon
    bf[CUT + 4] = plg[1][0];            // plg(1,0) = P10
    bf[CUT + 5] = plg[3][0];            // plg(3,0) = P30
    bf[CUT + 6] = plg[5][0];            // plg(5,0) = P50
    bf[CUT + 7] = plg[3][2];            // plg(3,2) = P32
    bf[CUT + 8] = plg[5][2];            // plg(5,2) = P52
}

inline TnParm NRLMSIS21::compute_tfnparm(const std::array<double, 512>& bf) const {
    using namespace msis21::constants;
    using namespace msis21::parm;

    TnParm tpro;
    tpro.cf.fill(0.0);
    tpro.beta.fill(0.0);
    tpro.gamma.fill(0.0);

    // Compute unconstrained spline coefficients (levels 0 to ITB0-1)
    // Fortran: dffact = 1.0/TN%beta(0,ix) per level
    for (int ix = 0; ix < ITB0; ix++) {
        tpro.cf[ix] = 0.0;
        for (int j = 0; j <= MBF; j++) {
            tpro.cf[ix] += TN_BETA[j][ix] * bf[j];
        }
        // Add sfluxmod if this level has nonlinear modulation
        if (smod_[ix]) {
            double dffact_ix = 1.0 / TN_BETA[0][ix];
            tpro.cf[ix] += sfluxmod_tn(ix, bf, dffact_ix);
        }
    }

    // Exospheric temperature (level ITEX = 23)
    tpro.tex = 0.0;
    for (int j = 0; j <= MBF; j++) {
        tpro.tex += TN_BETA[j][ITEX] * bf[j];
    }
    // Fortran always calls sfluxmod for tex (no smod check)
    tpro.tex += sfluxmod_tn(ITEX, bf, 1.0 / TN_BETA[0][ITEX]);
    // Add geomag and utdep to tex (using template versions for proper column access)
    tpro.tex += geomag_impl(TN_BETA, ITEX, bf);
    tpro.tex += utdep_impl(TN_BETA, ITEX, bf);

    // Temperature gradient at zetaB (level ITGB0 = 22)
    tpro.tgb0 = 0.0;
    for (int j = 0; j <= MBF; j++) {
        tpro.tgb0 += TN_BETA[j][ITGB0] * bf[j];
    }
    if (smod_[ITGB0]) tpro.tgb0 += sfluxmod_tn(ITGB0, bf, 1.0 / TN_BETA[0][ITGB0]);
    tpro.tgb0 += geomag_impl(TN_BETA, ITGB0, bf);
    // Note: Fortran does NOT apply utdep to tgb0

    // Temperature at zetaB (level ITB0 = 21)
    tpro.tb0 = 0.0;
    for (int j = 0; j <= MBF; j++) {
        tpro.tb0 += TN_BETA[j][ITB0] * bf[j];
    }
    if (smod_[ITB0]) tpro.tb0 += sfluxmod_tn(ITB0, bf, 1.0 / TN_BETA[0][ITB0]);
    tpro.tb0 += geomag_impl(TN_BETA, ITB0, bf);
    // Note: Fortran does NOT apply utdep to tb0

    // Shape factor
    tpro.sigma = tpro.tgb0 / (tpro.tex - tpro.tb0);

    // Constrain top three spline coefficients for C2 continuity
    double bc1 = 1.0 / tpro.tb0;
    double bc2 = -tpro.tgb0 / (tpro.tb0 * tpro.tb0);
    double bc3 = -bc2 * (tpro.sigma + 2.0 * tpro.tgb0 / tpro.tb0);

    tpro.cf[ITB0] = bc1 * C2TN[0][0] + bc2 * C2TN[1][0] + bc3 * C2TN[2][0];
    tpro.cf[ITGB0] = bc1 * C2TN[0][1] + bc2 * C2TN[1][1] + bc3 * C2TN[2][1];
    tpro.cf[ITEX] = bc1 * C2TN[0][2] + bc2 * C2TN[1][2] + bc3 * C2TN[2][2];

    // Temperature at zetaF (70 km) using S4 weights at level IZFX
    double dot_zetaF = tpro.cf[IZFX] * S4ZETA_F[0] +
                       tpro.cf[IZFX + 1] * S4ZETA_F[1] +
                       tpro.cf[IZFX + 2] * S4ZETA_F[2];
    tpro.tzetaF = 1.0 / dot_zetaF;

    // Temperature at zetaA (85 km) using S4 weights at level IZAX
    double dot_zetaA = tpro.cf[IZAX] * S4ZETA_A[0] +
                       tpro.cf[IZAX + 1] * S4ZETA_A[1] +
                       tpro.cf[IZAX + 2] * S4ZETA_A[2];
    tpro.tzetaA = 1.0 / dot_zetaA;

    // d(ln T)/dz at zetaA using wghtAxdz weights (derivative of 1/T spline at zetaA)
    double dfdz_A = tpro.cf[IZAX] * WGHTAXDZ[0] +
                    tpro.cf[IZAX + 1] * WGHTAXDZ[1] +
                    tpro.cf[IZAX + 2] * WGHTAXDZ[2];
    tpro.dlntdzA = -tpro.tzetaA * dfdz_A;

    // Calculate spline coefficients for first integral (beta)
    tpro.beta[0] = tpro.cf[0] * (NODES_TN[4] - NODES_TN[0]) / 4.0;
    for (int ix = 1; ix <= NL; ix++) {
        double wbeta_ix = (NODES_TN[ix + 4] - NODES_TN[ix]) / 4.0;
        tpro.beta[ix] = tpro.beta[ix - 1] + tpro.cf[ix] * wbeta_ix;
    }

    // Calculate spline coefficients for second integral (gamma)
    tpro.gamma[0] = tpro.beta[0] * (NODES_TN[5] - NODES_TN[0]) / 5.0;
    for (int ix = 1; ix <= NL; ix++) {
        double wgamma_ix = (NODES_TN[ix + 5] - NODES_TN[ix]) / 5.0;
        tpro.gamma[ix] = tpro.gamma[ix - 1] + tpro.beta[ix] * wgamma_ix;
    }

    // Integration constants
    tpro.b = 1.0 - tpro.tb0 / tpro.tex;
    tpro.sigmasq = tpro.sigma * tpro.sigma;

    // cVS = -dot_product(beta[ITB0-1:ITB0+2], S5ZETA_B)
    tpro.cVS = -(tpro.beta[ITB0 - 1] * S5ZETA_B[0] +
                 tpro.beta[ITB0] * S5ZETA_B[1] +
                 tpro.beta[ITB0 + 1] * S5ZETA_B[2] +
                 tpro.beta[ITB0 + 2] * S5ZETA_B[3]);

    // cWS = -dot_product(gamma[ITB0-2:ITB0+2], S6ZETA_B)
    tpro.cWS = -(tpro.gamma[ITB0 - 2] * S6ZETA_B[0] +
                 tpro.gamma[ITB0 - 1] * S6ZETA_B[1] +
                 tpro.gamma[ITB0] * S6ZETA_B[2] +
                 tpro.gamma[ITB0 + 1] * S6ZETA_B[3] +
                 tpro.gamma[ITB0 + 2] * S6ZETA_B[4]);

    tpro.cVB = -std::log(1.0 - tpro.b) / (tpro.sigma * tpro.tex);
    tpro.cWB = -msis21::utils::dilog(tpro.b) / (tpro.sigmasq * tpro.tex);

    // VzetaF = dot_product(beta[IZFX-1:IZFX+2], S5ZETA_F) + cVS
    tpro.VzetaF = tpro.beta[IZFX - 1] * S5ZETA_F[0] +
                  tpro.beta[IZFX] * S5ZETA_F[1] +
                  tpro.beta[IZFX + 1] * S5ZETA_F[2] +
                  tpro.beta[IZFX + 2] * S5ZETA_F[3] + tpro.cVS;

    // VzetaA = dot_product(beta[IZAX-1:IZAX+2], S5ZETA_A) + cVS
    tpro.VzetaA = tpro.beta[IZAX - 1] * S5ZETA_A[0] +
                  tpro.beta[IZAX] * S5ZETA_A[1] +
                  tpro.beta[IZAX + 1] * S5ZETA_A[2] +
                  tpro.beta[IZAX + 2] * S5ZETA_A[3] + tpro.cVS;

    // WzetaA = dot_product(gamma[IZAX-2:IZAX+2], S6ZETA_A) + cVS*(zetaA-zetaB) + cWS
    double delzA = ZETA_A - ZETA_B;
    tpro.WzetaA = tpro.gamma[IZAX - 2] * S6ZETA_A[0] +
                  tpro.gamma[IZAX - 1] * S6ZETA_A[1] +
                  tpro.gamma[IZAX] * S6ZETA_A[2] +
                  tpro.gamma[IZAX + 1] * S6ZETA_A[3] +
                  tpro.gamma[IZAX + 2] * S6ZETA_A[4] +
                  tpro.cVS * delzA + tpro.cWS;

    // Vzeta0 = dot_product(beta[0:2], S5ZETA_0) + cVS
    tpro.Vzeta0 = tpro.beta[0] * S5ZETA_0[0] +
                  tpro.beta[1] * S5ZETA_0[1] +
                  tpro.beta[2] * S5ZETA_0[2] + tpro.cVS;

    // Total number density at zetaF
    tpro.lndtotF = LNP0 - MBARG0DIVKB * (tpro.VzetaF - tpro.Vzeta0) - std::log(KB * tpro.tzetaF);

    return tpro;
}

inline double NRLMSIS21::pwmp(double z, const DnParm& dpro) const {
    // Piecewise effective mass profile interpolation
    if (z >= dpro.zetaMi[4]) return dpro.Mi[4];
    if (z <= dpro.zetaMi[0]) return dpro.Mi[0];

    for (int i = 0; i < 4; i++) {
        if (z < dpro.zetaMi[i + 1]) {
            return dpro.Mi[i] + dpro.aMi[i] * (z - dpro.zetaMi[i]);
        }
    }
    return dpro.Mi[4];
}

inline DnParm NRLMSIS21::compute_dfnparm(int ispec, const std::array<double, 512>& bf, const TnParm& tpro) const {
    using namespace msis21::constants;
    using namespace msis21::parm;

    DnParm dpro;
    dpro.ispec = ispec;
    dpro.zetaMi.fill(0.0);
    dpro.Mi.fill(0.0);
    dpro.aMi.fill(0.0);
    dpro.WMi.fill(0.0);
    dpro.XMi.fill(0.0);
    dpro.cf.fill(0.0);

    // Helper: dot product of beta column iz with basis functions
    auto dot_bf = [&](const auto& beta, int iz) {
        double sum = 0.0;
        for (int j = 0; j <= MBF; j++) sum += beta[j][iz] * bf[j];
        return sum;
    };

    // Get species-specific parameters from BETA arrays
    switch (ispec) {
        case SPEC_N2: {  // N2
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            // Bug 5 fix: zetaM computed from basis functions, not just beta[0][1]
            dpro.zetaM = dot_bf(N2_BETA, 1);
            dpro.HML = N2_BETA[0][2];
            dpro.HMU = N2_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            // N2 R is always 0.0 in NRLMSIS 2.1 (N2Rflag not used)
            dpro.R = 0.0;
            dpro.zetaR = N2_BETA[0][8];
            dpro.HR = N2_BETA[0][9];
            break;
        }
        case SPEC_O2: {  // O2
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = O2_BETA[0][1];
            dpro.HML = O2_BETA[0][2];
            dpro.HMU = O2_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            // Bug 7 fix: add geomag to O2 R
            dpro.R = dot_bf(O2_BETA, 7);
            dpro.R += geomag_impl(O2_BETA, 7, bf);
            dpro.zetaR = O2_BETA[0][8];
            dpro.HR = O2_BETA[0][9];
            break;
        }
        case SPEC_O: {  // O1
            dpro.lnPhiF = 0.0;
            dpro.lndref = dot_bf(O1_BETA, 0);
            dpro.zref = ZETAREF_O1;  // 85 km (zetaA)
            dpro.zmin = NODES_O1[3];  // Bug 5 fix: 50.0 (not -1.0)
            dpro.zhyd = ZETAREF_O1;   // 85 km
            dpro.zetaM = O1_BETA[0][1];
            dpro.HML = O1_BETA[0][2];
            dpro.HMU = O1_BETA[0][3];
            dpro.C = dot_bf(O1_BETA, 4);
            dpro.zetaC = O1_BETA[0][5];
            dpro.HC = O1_BETA[0][6];
            // Bug 7 fix: add sfluxmod + geomag + utdep to O1 R
            dpro.R = dot_bf(O1_BETA, 7);
            dpro.R += sfluxmod(7, bf, O1_BETA, 0.0);
            dpro.R += geomag_impl(O1_BETA, 7, bf);
            dpro.R += utdep_impl(O1_BETA, 7, bf);
            dpro.zetaR = O1_BETA[0][8];
            dpro.HR = O1_BETA[0][9];

            // Bug 2 fix: Compute O1 spline coefficients (indices 10-17 in O1_BETA)
            for (int k = 0; k < NSPLO1; k++) {
                dpro.cf[k] = dot_bf(O1_BETA, 10 + k);
            }
            break;
        }
        case SPEC_HE: {  // He
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = HE_BETA[0][1];
            dpro.HML = HE_BETA[0][2];
            dpro.HMU = HE_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            // Bug 7 fix: add sfluxmod + geomag + utdep to HE R
            dpro.R = dot_bf(HE_BETA, 7);
            dpro.R += sfluxmod(7, bf, HE_BETA, 1.0);
            dpro.R += geomag_impl(HE_BETA, 7, bf);
            dpro.R += utdep_impl(HE_BETA, 7, bf);
            dpro.zetaR = HE_BETA[0][8];
            dpro.HR = HE_BETA[0][9];
            break;
        }
        case SPEC_H: {  // H
            dpro.lnPhiF = 0.0;
            dpro.lndref = dot_bf(H1_BETA, 0);
            dpro.zref = ZETA_A;
            dpro.zmin = 75.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = H1_BETA[0][1];
            dpro.HML = H1_BETA[0][2];
            dpro.HMU = H1_BETA[0][3];
            dpro.C = dot_bf(H1_BETA, 4);
            dpro.zetaC = dot_bf(H1_BETA, 5);
            dpro.HC = H1_BETA[0][6];
            // Bug 7 fix: add sfluxmod + geomag + utdep to H1 R
            dpro.R = dot_bf(H1_BETA, 7);
            dpro.R += sfluxmod(7, bf, H1_BETA, 0.0);
            dpro.R += geomag_impl(H1_BETA, 7, bf);
            dpro.R += utdep_impl(H1_BETA, 7, bf);
            dpro.zetaR = H1_BETA[0][8];
            dpro.HR = H1_BETA[0][9];
            break;
        }
        case SPEC_AR: {  // Ar
            dpro.lnPhiF = LNVMR[ispec];
            dpro.lndref = tpro.lndtotF + dpro.lnPhiF;
            dpro.zref = ZETA_F;
            dpro.zmin = -1.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = AR_BETA[0][1];
            dpro.HML = AR_BETA[0][2];
            dpro.HMU = AR_BETA[0][3];
            dpro.C = 0.0;
            dpro.zetaC = 0.0;
            dpro.HC = 1.0;
            // Bug 7 fix: add geomag + utdep to AR R
            dpro.R = dot_bf(AR_BETA, 7);
            dpro.R += geomag_impl(AR_BETA, 7, bf);
            dpro.R += utdep_impl(AR_BETA, 7, bf);
            dpro.zetaR = AR_BETA[0][8];
            dpro.HR = AR_BETA[0][9];
            break;
        }
        case SPEC_N: {  // N1
            dpro.lnPhiF = 0.0;
            // Bug 7 fix: add sfluxmod + geomag + utdep to N1 lndref
            dpro.lndref = dot_bf(N1_BETA, 0);
            dpro.lndref += sfluxmod(0, bf, N1_BETA, 0.0);
            dpro.lndref += geomag_impl(N1_BETA, 0, bf);
            dpro.lndref += utdep_impl(N1_BETA, 0, bf);
            dpro.zref = ZETA_B;
            dpro.zmin = 90.0;
            dpro.zhyd = ZETA_F;
            dpro.zetaM = N1_BETA[0][1];
            dpro.HML = N1_BETA[0][2];
            dpro.HMU = N1_BETA[0][3];
            dpro.C = N1_BETA[0][4];
            dpro.zetaC = N1_BETA[0][5];
            dpro.HC = N1_BETA[0][6];
            // N1 R has no nonlinear terms in Fortran
            dpro.R = dot_bf(N1_BETA, 7);
            dpro.zetaR = N1_BETA[0][8];
            dpro.HR = N1_BETA[0][9];
            break;
        }
        case SPEC_OA: {  // Anomalous O
            dpro.lnPhiF = 0.0;
            // Bug 6 fix: add geomag to OA lndref
            dpro.lndref = dot_bf(OA_BETA, 0);
            dpro.lndref += geomag_impl(OA_BETA, 0, bf);
            dpro.zref = ZETAREF_OA;
            dpro.zmin = 120.0;
            dpro.zhyd = 0.0;  // No hydrostatic integration for OA
            dpro.C = OA_BETA[0][4];
            dpro.zetaC = OA_BETA[0][5];
            dpro.HC = OA_BETA[0][6];
            // OA doesn't use mass profile
            return dpro;
        }
        case SPEC_NO: {  // NO
            dpro.lnPhiF = 0.0;
            // Bug 7 fix: add geomag to NO lndref
            dpro.lndref = dot_bf(NO_BETA, 0);
            dpro.lndref += geomag_impl(NO_BETA, 0, bf);
            dpro.zref = ZETAREF_NO;  // 122.5 km
            dpro.zmin = 72.5;  // Fortran: 72.5 km (cut off below due to spline artifacts at edge of domain)
            dpro.zhyd = ZETAREF_NO;
            dpro.zetaM = dot_bf(NO_BETA, 1);
            dpro.HML = dot_bf(NO_BETA, 2);
            dpro.HMU = dot_bf(NO_BETA, 3);
            // Bug 7 fix: add geomag to NO C
            dpro.C = dot_bf(NO_BETA, 4);
            dpro.C += geomag_impl(NO_BETA, 4, bf);
            dpro.zetaC = dot_bf(NO_BETA, 5);
            dpro.HC = dot_bf(NO_BETA, 6);
            dpro.R = dot_bf(NO_BETA, 7);
            dpro.zetaR = dot_bf(NO_BETA, 8);
            dpro.HR = dot_bf(NO_BETA, 9);

            // Bug 2 fix: Compute NO spline coefficients (indices 10-17 in NO_BETA)
            // Fortran adds geomag to each NO spline coefficient
            for (int k = 0; k < NSPLNO; k++) {
                dpro.cf[k] = dot_bf(NO_BETA, 10 + k);
                dpro.cf[k] += geomag_impl(NO_BETA, 10 + k, bf);
            }
            break;
        }
        default:
            return dpro;
    }

    // Compute piecewise effective mass profile (except for OA)
    if (ispec != SPEC_OA) {
        dpro.zetaMi[0] = dpro.zetaM - 2.0 * dpro.HML;
        dpro.zetaMi[1] = dpro.zetaM - dpro.HML;
        dpro.zetaMi[2] = dpro.zetaM;
        dpro.zetaMi[3] = dpro.zetaM + dpro.HMU;
        dpro.zetaMi[4] = dpro.zetaM + 2.0 * dpro.HMU;

        dpro.Mi[0] = MBAR;
        dpro.Mi[4] = SPECMASS[ispec];
        dpro.Mi[2] = (dpro.Mi[0] + dpro.Mi[4]) / 2.0;
        double delM = TANH1 * (dpro.Mi[4] - dpro.Mi[0]) / 2.0;
        dpro.Mi[1] = dpro.Mi[2] - delM;
        dpro.Mi[3] = dpro.Mi[2] + delM;

        for (int i = 0; i < 4; i++) {
            dpro.aMi[i] = (dpro.Mi[i + 1] - dpro.Mi[i]) / (dpro.zetaMi[i + 1] - dpro.zetaMi[i]);
        }

        // Compute WMi (2nd integral of 1/T at each mass node)
        for (int i = 0; i < 5; i++) {
            double delz = dpro.zetaMi[i] - ZETA_B;
            if (dpro.zetaMi[i] < ZETA_B) {
                // Spline region - need to evaluate gamma spline
                auto spl = msis21::utils::bspline_eval(dpro.zetaMi[i], NODES_TN.data(), ND + 2, etaTN_);
                double W = 0.0;
                for (int j = 0; j < 6; j++) {
                    int idx = spl.iz - 5 + j;
                    if (idx >= 0 && idx <= NL) {
                        W += tpro.gamma[idx] * spl.S6[j];
                    }
                }
                dpro.WMi[i] = W + tpro.cVS * delz + tpro.cWS;
            } else {
                // Bates region
                dpro.WMi[i] = (0.5 * delz * delz +
                    msis21::utils::dilog(tpro.b * std::exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
                    + tpro.cVB * delz + tpro.cWB;
            }
        }

        // Compute XMi (cumulative mass adjustment)
        dpro.XMi[0] = -dpro.aMi[0] * dpro.WMi[0];
        for (int i = 1; i < 4; i++) {
            dpro.XMi[i] = dpro.XMi[i - 1] - dpro.WMi[i] * (dpro.aMi[i] - dpro.aMi[i - 1]);
        }
        dpro.XMi[4] = dpro.XMi[3] + dpro.WMi[4] * dpro.aMi[3];

        // Calculate reference height terms
        if (dpro.zref == ZETA_F) {
            dpro.Mzref = MBAR;
            dpro.Tref = tpro.tzetaF;
            dpro.Izref = MBAR * tpro.VzetaF;
        } else if (dpro.zref == ZETA_A) {
            dpro.Mzref = pwmp(ZETA_A, dpro);
            dpro.Tref = tpro.tzetaA;
            double Izref_raw = dpro.Mzref * tpro.VzetaA;
            dpro.Izref = Izref_raw;
            // Apply mass profile adjustment
            if (ZETA_A > dpro.zetaMi[0] && ZETA_A < dpro.zetaMi[4]) {
                int seg = 0;
                for (int i = 1; i < 4; i++) {
                    if (ZETA_A >= dpro.zetaMi[i]) seg = i;
                }
                double adj = dpro.aMi[seg] * tpro.WzetaA + dpro.XMi[seg];
                dpro.Izref -= adj;
            } else if (ZETA_A >= dpro.zetaMi[4]) {
                dpro.Izref -= dpro.XMi[4];
            }
        } else if (dpro.zref == ZETA_B) {
            dpro.Mzref = pwmp(ZETA_B, dpro);
            dpro.Tref = tpro.tb0;
            dpro.Izref = 0.0;
            // Apply mass profile adjustment at zetaB
            if (ZETA_B > dpro.zetaMi[0] && ZETA_B < dpro.zetaMi[4]) {
                int seg = 0;
                for (int i = 1; i < 4; i++) {
                    if (ZETA_B >= dpro.zetaMi[i]) seg = i;
                }
                dpro.Izref -= dpro.XMi[seg];
            } else if (ZETA_B >= dpro.zetaMi[4]) {
                dpro.Izref -= dpro.XMi[4];
            }
        }
    }

    // C1 continuity constraint for O1 at zetaA (85 km)
    // Matches Fortran dfnparm lines 369-381
    if (ispec == SPEC_O) {
        double Cterm = dpro.C * std::exp(-(dpro.zref - dpro.zetaC) / dpro.HC);
        double Rterm0 = std::tanh((dpro.zref - dpro.zetaR) / (HRfactO1ref_ * dpro.HR));
        double Rterm = dpro.R * (1.0 + Rterm0);

        // bc(1): value constraint at reference height
        double bc0 = dpro.lndref - Cterm + Rterm
                   - dpro.cf[NSPLO1 - 1] * C1O1ADJ[0];

        // bc(2): derivative constraint at reference height
        double bc1 = -dpro.Mzref * G0DIVKB / tpro.tzetaA     // hydrostatic gradient
                   - tpro.dlntdzA                               // ideal gas gradient
                   + Cterm / dpro.HC                            // Chapman gradient
                   + Rterm * (1.0 - Rterm0) / dpro.HR * dHRfactO1ref_  // tapered logistic gradient
                   - dpro.cf[NSPLO1 - 1] * C1O1ADJ[1];        // last unconstrained spline gradient

        // matmul(bc, c1o1): result(j) = sum_i bc(i) * c1o1(i,j)
        dpro.cf[NSPLO1]     = bc0 * C1O1[0][0] + bc1 * C1O1[1][0];
        dpro.cf[NSPLO1 + 1] = bc0 * C1O1[0][1] + bc1 * C1O1[1][1];
    }

    // C1 continuity constraint for NO at zetaB (122.5 km)
    // Matches Fortran dfnparm lines 384-396
    if (ispec == SPEC_NO && dpro.lndref != 0.0) {
        double Cterm = dpro.C * std::exp(-(dpro.zref - dpro.zetaC) / dpro.HC);
        double Rterm0 = std::tanh((dpro.zref - dpro.zetaR) / (HRfactNOref_ * dpro.HR));
        double Rterm = dpro.R * (1.0 + Rterm0);

        // bc(1): value constraint at reference height
        double bc0 = dpro.lndref - Cterm + Rterm
                   - dpro.cf[NSPLNO - 1] * C1NOADJ[0];

        // bc(2): derivative constraint at reference height
        double bc1 = -dpro.Mzref * G0DIVKB / tpro.tb0         // hydrostatic gradient
                   - tpro.tgb0 / tpro.tb0                      // ideal gas gradient (-dlnT/dz)
                   + Cterm / dpro.HC                            // Chapman gradient
                   + Rterm * (1.0 - Rterm0) / dpro.HR * dHRfactNOref_  // tapered logistic gradient
                   - dpro.cf[NSPLNO - 1] * C1NOADJ[1];        // last unconstrained spline gradient

        // matmul(bc, c1no): result(j) = sum_i bc(i) * c1no(i,j)
        dpro.cf[NSPLNO]     = bc0 * C1NO[0][0] + bc1 * C1NO[1][0];
        dpro.cf[NSPLNO + 1] = bc0 * C1NO[0][1] + bc1 * C1NO[1][1];
    }

    return dpro;
}

inline double NRLMSIS21::compute_dfnx(double zeta, double tn, double lndtotz, double Vz, double Wz, double HRfact,
                                       const TnParm& tpro, const DnParm& dpro) const {
    using namespace msis21::constants;

    // Below minimum height
    if (zeta < dpro.zmin) {
        return DMISSING;
    }

    // Anomalous Oxygen (legacy MSISE-00 formulation)
    if (dpro.ispec == SPEC_OA) {
        double lnd = dpro.lndref - (zeta - dpro.zref) / H_OA;
        if (dpro.C != 0.0) {
            lnd -= dpro.C * std::exp(-(zeta - dpro.zetaC) / dpro.HC);
        }
        return std::exp(lnd);
    }

    // Check for undefined NO parameters
    if (dpro.ispec == SPEC_NO && dpro.lndref == 0.0) {
        return DMISSING;
    }

    // Chapman and logistic corrections
    double ccor = 0.0;
    switch (dpro.ispec) {
        case SPEC_N2:
        case SPEC_O2:
        case SPEC_HE:
        case SPEC_AR:
            // Logistic correction only
            ccor = dpro.R * (1.0 + std::tanh((zeta - dpro.zetaR) / (HRfact * dpro.HR)));
            break;
        case SPEC_O:
        case SPEC_H:
        case SPEC_N:
        case SPEC_NO:
            // Chapman and logistic corrections
            ccor = -dpro.C * std::exp(-(zeta - dpro.zetaC) / dpro.HC) +
                    dpro.R * (1.0 + std::tanh((zeta - dpro.zetaR) / (HRfact * dpro.HR)));
            break;
    }

    // Below hydrostatic region
    if (zeta < dpro.zhyd) {
        // Bug 3/4 fix: For mixed species (N2, O2, He, Ar), use lndtotz (computed from pressure)
        if (dpro.ispec == SPEC_N2 || dpro.ispec == SPEC_O2 ||
            dpro.ispec == SPEC_HE || dpro.ispec == SPEC_AR) {
            return std::exp(lndtotz + dpro.lnPhiF + ccor);
        }
        // For O1 below zhyd (85 km), evaluate chemistry-region spline
        if (dpro.ispec == SPEC_O) {
            auto spl = bspline_eval_o4(zeta, NODES_O1, NDO1 + 1, etaO1_);
            double lnf = 0.0;
            for (int k = 0; k < 4; k++) {
                int idx = spl.iz - 3 + k;
                if (idx >= 0 && idx < NSPLO1 + 2) {
                    lnf += dpro.cf[idx] * spl.S4[k];
                }
            }
            return std::exp(lnf);
        }
        // For NO below zhyd (122.5 km), evaluate chemistry-region spline
        if (dpro.ispec == SPEC_NO) {
            auto spl = bspline_eval_o4(zeta, NODES_NO, NDNO + 1, etaNO_);
            double lnf = 0.0;
            for (int k = 0; k < 4; k++) {
                int idx = spl.iz - 3 + k;
                if (idx >= 0 && idx < NSPLNO + 2) {
                    lnf += dpro.cf[idx] * spl.S4[k];
                }
            }
            return std::exp(lnf);
        }
        // Other species return DMISSING below zhyd
        return DMISSING;
    }

    // Calculate hydrostatic integral with piecewise effective mass
    double Mz = pwmp(zeta, dpro);
    double Ihyd = Mz * Vz - dpro.Izref;

    // Apply mass profile adjustment
    if (zeta > dpro.zetaMi[0] && zeta < dpro.zetaMi[4]) {
        int seg = 0;
        for (int i = 1; i < 4; i++) {
            if (zeta >= dpro.zetaMi[i]) seg = i;
        }
        Ihyd -= (dpro.aMi[seg] * Wz + dpro.XMi[seg]);
    } else if (zeta >= dpro.zetaMi[4]) {
        Ihyd -= dpro.XMi[4];
    }

    // Log number density
    double lnd = dpro.lndref - Ihyd * G0DIVKB + ccor;

    // Apply ideal gas law
    return std::exp(lnd) * dpro.Tref / tn;
}

inline double NRLMSIS21::compute_temperature(double zeta, const TnParm& tpro,
                                              int iz, const std::array<double, 6>& S4) const {
    using namespace msis21::constants;

    if (zeta < ZETA_B) {
        // Spline region
        int i = std::max(iz - 3, 0);
        int j_start = (iz < 3) ? -iz : -3;

        double dot = 0.0;
        for (int j = j_start; j <= 0; j++) {
            dot += tpro.cf[i + j - j_start] * S4[j + 3];
        }
        return 1.0 / dot;
    } else {
        // Bates profile region
        return tpro.tex - (tpro.tex - tpro.tb0) * std::exp(-tpro.sigma * (zeta - ZETA_B));
    }
}

inline double NRLMSIS21::compute_Wz(double zeta, const TnParm& tpro, int iz, const std::array<double, 6>& S6) const {
    using namespace msis21::constants;

    double delz = zeta - ZETA_B;
    if (zeta < ZETA_B) {
        // Spline region
        double W = 0.0;
        for (int j = 0; j < 6; j++) {
            int idx = iz - 5 + j;
            if (idx >= 0 && idx <= NL) {
                W += tpro.gamma[idx] * S6[j];
            }
        }
        return W + tpro.cVS * delz + tpro.cWS;
    } else {
        // Bates region
        return (0.5 * delz * delz +
            msis21::utils::dilog(tpro.b * std::exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
            + tpro.cVB * delz + tpro.cWB;
    }
}

inline NRLMSIS21::Output NRLMSIS21::msiscalc(const Input& input) const {
    using namespace msis21::constants;
    using namespace msis21::utils;

    Output output;
    output.dn.fill(DMISSING);

    // Convert geodetic altitude to geopotential height
    double zeta = alt2gph(input.lat, input.alt);

    // Compute basis functions
    std::array<double, 512> bf;
    compute_globe(input, bf);

    // Compute temperature profile parameters
    TnParm tpro = compute_tfnparm(bf);

    // Set exospheric temperature
    output.tex = tpro.tex;

    // Compute B-spline weights at current altitude
    BSplineResult spl;
    if (zeta < ZETA_B) {
        spl = bspline_eval(zeta, NODES_TN.data(), ND + 2, etaTN_);
    } else {
        spl.iz = NL;
        spl.S4.fill(0.0);
        spl.S5.fill(0.0);
        spl.S6.fill(0.0);
    }

    // Compute temperature at altitude
    output.tn = compute_temperature(zeta, tpro, spl.iz, spl.S4);

    // Compute integration terms (V and W) at altitude using three-region logic
    // Region 1: zeta < zetaF (70 km) - Vz from S5, Wz=0, lndtotz from pressure
    // Region 2: zetaF <= zeta < zetaB (122.5 km) - Vz from S5, Wz from S6
    // Region 3: zeta >= zetaB - Bates analytical formulas
    double delz = zeta - ZETA_B;
    double Vz, Wz, lndtotz = 0.0;
    if (zeta < ZETA_F) {
        // Region 1: Below fully-mixed boundary
        // Handle iz < 4 case (near bottom of spline range)
        int i_start = std::max(spl.iz - 4, 0);
        int j_start = (spl.iz < 4) ? -spl.iz : -4;
        Vz = 0.0;
        for (int l = j_start; l <= 0; l++) {
            int beta_idx = i_start + (l - j_start);
            Vz += tpro.beta[beta_idx] * spl.S5[l + 4];  // S5[l+4] = Sz(l, 5), l in [-4,0]
        }
        Vz += tpro.cVS;
        Wz = 0.0;
        double lnPz = LNP0 - MBARG0DIVKB * (Vz - tpro.Vzeta0);
        lndtotz = lnPz - std::log(KB * output.tn);
    } else if (zeta < ZETA_B) {
        // Region 2: zetaF <= zeta < zetaB
        Vz = 0.0;
        for (int k = 0; k < 5; k++) {
            int idx = spl.iz - 4 + k;
            if (idx >= 0 && idx <= NL) {
                Vz += tpro.beta[idx] * spl.S5[k];  // S5[k] for k in [0,4]
            }
        }
        Vz += tpro.cVS;
        Wz = compute_Wz(zeta, tpro, spl.iz, spl.S6);
    } else {
        // Region 3: Bates region (zeta >= zetaB)
        Vz = (delz + std::log(output.tn / tpro.tex) / tpro.sigma) / tpro.tex + tpro.cVB;
        Wz = (0.5 * delz * delz +
            dilog(tpro.b * std::exp(-tpro.sigma * delz)) / tpro.sigmasq) / tpro.tex
            + tpro.cVB * delz + tpro.cWB;
    }

    // Chemical/dynamical correction factor (tapers below 100 km)
    double HRfact = 0.5 * (1.0 + std::tanh(H_GAMMA * (zeta - ZETA_GAMMA)));

    // Compute density for each species
    for (int ispec = 2; ispec <= 10; ispec++) {
        DnParm dpro = compute_dfnparm(ispec, bf, tpro);
        output.dn[ispec] = compute_dfnx(zeta, output.tn, lndtotz, Vz, Wz, HRfact, tpro, dpro);
    }

    // Total mass density (kg/m^3)
    // Exclude NO (species 10) per Fortran masswgt(10) = 0.0
    output.dn[1] = 0.0;
    for (int i = 2; i <= 9; i++) {
        if (output.dn[i] != DMISSING && output.dn[i] > 0.0) {
            output.dn[1] += output.dn[i] * SPECMASS[i];
        }
    }

    return output;
}

inline double NRLMSIS21::density(double alt_km, double lat_deg, double lon_deg,
                                  const time::Epoch& epoch, const WeatherConditions& weather) const {
    Input input;

    auto [year, month, day_of_month, hour, minute, second] = epoch.to_calendar();
    input.day = static_cast<double>(epoch.day_of_year());
    input.utsec = hour * 3600.0 + minute * 60.0 + second;

    input.alt = alt_km;
    input.lat = lat_deg;
    input.lon = lon_deg;

    input.f107a = weather.f107_average;
    input.f107 = weather.f107_daily;
    input.ap = weather.ap_history;

    Output out = msiscalc(input);
    return out.dn[1];  // Total mass density (kg/m^3)
}

inline NRLMSIS21::Output NRLMSIS21::full_output(double alt_km, double lat_deg, double lon_deg,
                                                  const time::Epoch& epoch, const WeatherConditions& weather) const {
    Input input;
    auto [year, month, day_of_month, hour, minute, second] = epoch.to_calendar();
    input.day = static_cast<double>(epoch.day_of_year());
    input.utsec = hour * 3600.0 + minute * 60.0 + second;
    input.alt = alt_km;
    input.lat = lat_deg;
    input.lon = lon_deg;
    input.f107a = weather.f107_average;
    input.f107 = weather.f107_daily;
    input.ap = weather.ap_history;
    return msiscalc(input);
}

}  // namespace atmosphere
}  // namespace ballistic
