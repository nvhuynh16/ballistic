#pragma once

// =============================================================================
// Flow-regime drag coefficient: free-molecular -> transitional -> continuum
// =============================================================================
//
// The drag coefficient Cd (dimensionless) scales the drag force:
// F = ½ ρ V² Cd A, with ρ the air density, V the speed through the air and A
// the reference area. An orbiting satellite flies in free-molecular flow: the
// air is so thin that molecules hit the body without first colliding with
// each other, and Cd is set by how they reflect from the surface (Cd ~ 2-2.5
// for compact bodies).
// A reentering object descends into continuum flow, where a shock wave stands
// ahead of it and Cd depends on the Mach number (a sphere ~0.9 hypersonic).
// The Knudsen number Kn = (mean free path) / (body size) says which regime
// applies: the mean free path is the average distance a molecule travels
// between collisions. A constant Cd is right in orbit and wrong in the last
// ~100 km of a reentry; this file gives Cd as a function of the flow. The
// result enters the drag force through the ballistic coefficient
// β = m / (Cd A) [kg/m²] (AeroBeta, at the end of this file).
//
//   Free-molecular (Kn >= 10): Sentman (1961) closed forms with diffuse
//     re-emission (molecules leave the surface in random directions) and
//     energy accommodation alpha (how far they cool to the wall temperature
//     before leaving: 1 = fully), for a sphere, a flat plate
//     facing the flow, and a cylinder in cross-flow (Schaaf & Chambre 1958,
//     "Flow of Rarefied Gases", Princeton; the sphere form is also eq. B11 /
//     C6 of Singh et al. 2020, arXiv:2012.04813).
//   Continuum (Kn <= 1e-3): sphere - Henderson (1976, AIAA J. 14(6) 707) in its
//     high-Reynolds-number limit (as reproduced in Singh et al. 2020, eq. B1;
//     the Reynolds number Re measures inertia against viscosity);
//     plate and cylinder - modified Newtonian theory (surface pressure
//     coefficient Cp = Cp_max sin^2 of the angle between surface and flow),
//     Cp_max from the Rayleigh
//     pitot formula (Anderson, "Hypersonic and High-Temperature Gas
//     Dynamics", 2nd ed., sec. 3.3), valid hypersonically (M >~ 3); or a
//     user-supplied Cd(Mach) table.
//   Transitional: Cd = Cd_c + (Cd_fm - Cd_c) sin^2(pi/8 (3 + log10 Kn)) for
//     1e-3 < Kn < 10 (the bridging function of Koppenwallner & Legge; used
//     in ESA DRAMA/SCARAB, Klinkrad 2006, "Space Debris", sec. 5.4).
//
// Flow parameters from the free stream: mean free path of hard spheres,
// lambda = 1 / (sqrt(2) pi sigma^2 n), sigma = 3.65e-10 m (U.S. Standard
// Atmosphere 1976, eq. 52 and Table 7); Mach from a = sqrt(gamma R* T / M),
// gamma = 1.4 (air below ~120 km; above it the flow is free-molecular for
// any body size of interest, where Mach does not enter).
//
// Composition enters through the mean molar mass only (the free-molecular
// speed ratio s = V / sqrt(2kT/m)); a species-by-species Sentman average
// differs by under 1% at reentry speeds (s >> 1).
//
// Not modeled: heating, ablation, breakup, lift, attitude (the body's
// orientation is fixed relative to the flow), the sphere's drag crisis at
// Re > ~3e5 (Henderson's fit stops below it).
// =============================================================================

#include "../core/diagnostics.hpp"
#include "../core/errors.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ballistic {
namespace aero {

inline constexpr double kBoltzmann = 1.380649e-23;         // J/K
inline constexpr double kAvogadroKmol = 6.02214076e26;     // 1/kmol
inline constexpr double kGasConstant = 8314.462618;        // J/(kmol K)
inline constexpr double kHardSphereDiameter = 3.65e-10;    // m, USSA76 Table 7
inline constexpr double kGammaAir = 1.4;
inline constexpr double kKnContinuum = 1e-3;
inline constexpr double kKnFreeMolecular = 10.0;

/// Body shape for FlowRegimeCd; each fixes what the reference area and the
/// reference length (the size in the Knudsen number) mean.
enum class Shape {
    Sphere,      // reference area pi D^2 / 4, reference length D
    FlatPlate,   // facing the flow; reference area = plate area, length = its side
    Cylinder,    // axis across the flow; reference area L D, reference length D
};

/// Lower-case name of a shape, for messages.
[[nodiscard]] inline const char* shape_name(Shape s) noexcept {
    switch (s) {
        case Shape::Sphere: return "sphere";
        case Shape::FlatPlate: return "flat plate";
        case Shape::Cylinder: return "cylinder";
    }
    return "?";
}

// ---- Free stream -------------------------------------------------------------

/// Mean free path [m] at density rho [kg/m^3] and mean molar mass [kg/kmol]
/// (hard-sphere molecules of diameter kHardSphereDiameter).
[[nodiscard]] inline double mean_free_path(double rho, double molar_mass) {
    const double n = rho * kAvogadroKmol / molar_mass;
    return 1.0 / (std::numbers::sqrt2 * std::numbers::pi * kHardSphereDiameter * kHardSphereDiameter * n);
}

/// Speed of sound [m/s] in an ideal gas at temperature [K] and mean molar
/// mass [kg/kmol], with gamma = 1.4.
[[nodiscard]] inline double speed_of_sound(double temperature_k, double molar_mass) {
    return std::sqrt(kGammaAir * kGasConstant * temperature_k / molar_mass);
}

/// Molecular speed ratio s = V / sqrt(2 k T / m): the body's speed V [m/s]
/// over the most probable thermal speed of the molecules (temperature [K],
/// mean molar mass [kg/kmol]). In orbit s is ~5-10.
[[nodiscard]] inline double speed_ratio(double speed, double temperature_k, double molar_mass) {
    return speed / std::sqrt(2.0 * kGasConstant * temperature_k / molar_mass);
}

// ---- Free-molecular (Sentman) --------------------------------------------------

/// Ratio of the speed of molecules leaving the surface to their incoming
/// speed, for energy accommodation alpha (0..1) and a wall at wall_k [K]
/// (Sentman 1961; Moe & Moe 2005): sqrt(1/2 (1 + alpha (4kTw/(mV^2) - 1))).
[[nodiscard]] inline double reemission_ratio(double speed, double molar_mass, double alpha, double wall_k) {
    const double m = molar_mass / kAvogadroKmol;
    return std::sqrt(0.5 * (1.0 + alpha * (4.0 * kBoltzmann * wall_k / (m * speed * speed) - 1.0)));
}

/// Free-molecular Cd of one flat face (area = reference area) whose outward
/// normal makes cosine g with the direction the flow comes from (g = 1 faces
/// the flow, -1 faces away); s = speed ratio, v_ratio = reemission_ratio.
[[nodiscard]] inline double sentman_face(double g, double s, double v_ratio) {
    const double P = std::exp(-g * g * s * s) / s;
    const double Q = 1.0 + 1.0 / (2.0 * s * s);
    const double Z = 1.0 + std::erf(g * s);
    const double sqrt_pi = std::sqrt(std::numbers::pi);
    return P / sqrt_pi + g * Q * Z + 0.5 * g * v_ratio * (g * sqrt_pi * Z + P);
}

// exp(-x) I_nu(x), the scaled modified Bessel function of the first kind,
// nu = 0 or 1, x >= 0, without overflow for large x (I_nu overflows a double
// near x = 713). Written out because libc++ has no std::cyl_bessel_i.
//   x <= 50: the power series I_nu(x) = sum_k (x/2)^(2k+nu) / (k! (k+nu)!),
//            all terms positive (no cancellation), scaled by exp(-x);
//   x > 50:  the asymptotic series
//            e^-x I_nu(x) ~ (2 pi x)^-1/2 sum_k (-1)^k a_k / x^k,
//            a_k = prod_{j<=k} (4nu^2 - (2j-1)^2) / (k! 8^k)
//            (Abramowitz & Stegun 9.7.1), summed until a term no longer
//            changes the sum; its smallest term at x = 50 is ~1e-43.
// Agrees with std::cyl_bessel_i to ~1e-14 relative (test_flow_regime_cd).
[[nodiscard]] inline double scaled_bessel_i(int nu, double x) {
    if (x <= 50.0) {
        const double q = 0.25 * x * x;
        double term = (nu == 0) ? 1.0 : 0.5 * x;
        double sum = term;
        for (int k = 1; k < 500; ++k) {
            term *= q / (static_cast<double>(k) * (k + nu));
            sum += term;
            if (term < 1e-17 * sum) break;
        }
        return std::exp(-x) * sum;
    }
    const double mu = 4.0 * nu * nu;
    double term = 1.0, sum = 1.0;
    for (int k = 1; k <= 60; ++k) {
        const double next = term * -(mu - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (k * 8.0 * x);
        if (std::abs(next) >= std::abs(term)) break;   // the series starts to diverge
        term = next;
        sum += term;
        if (std::abs(term) < 1e-17 * std::abs(sum)) break;
    }
    return sum / std::sqrt(2.0 * std::numbers::pi * x);
}

/// Free-molecular Cd of a shape (Sentman / Schaaf & Chambre closed forms) at
/// speed ratio s and re-emission speed ratio v_ratio (see reemission_ratio).
[[nodiscard]] inline double free_molecular_cd(Shape shape, double s, double v_ratio) {
    const double pi = std::numbers::pi;
    const double sqrt_pi = std::sqrt(pi);
    switch (shape) {
        case Shape::Sphere: {
            const double s2 = s * s, s4 = s2 * s2;
            return (2.0 * s2 + 1.0) / (sqrt_pi * s2 * s) * std::exp(-s2) +
                   (4.0 * s4 + 4.0 * s2 - 1.0) / (2.0 * s4) * std::erf(s) +
                   2.0 * sqrt_pi / 3.0 * v_ratio;
        }
        case Shape::FlatPlate:
            return sentman_face(1.0, s, v_ratio) + sentman_face(-1.0, s, v_ratio);
        case Shape::Cylinder: {
            const double x = 0.5 * s * s;
            return sqrt_pi / s *
                       ((s * s + 1.5) * scaled_bessel_i(0, x) + (s * s + 0.5) * scaled_bessel_i(1, x)) +
                   pi * sqrt_pi / 4.0 * v_ratio;
        }
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// ---- Continuum ---------------------------------------------------------------

/// Pressure coefficient at the stagnation point (where the flow stops on the
/// body's nose) at Mach `mach`: Rayleigh pitot formula behind a normal shock
/// for M > 1, isentropic (shock-free) compression below.
[[nodiscard]] inline double stagnation_cp(double mach, double gamma = kGammaAir) {
    const double M2 = mach * mach;
    const double e = gamma / (gamma - 1.0);
    double p0_over_p;
    if (mach > 1.0) {
        p0_over_p = std::pow((gamma + 1.0) * (gamma + 1.0) * M2 / (4.0 * gamma * M2 - 2.0 * (gamma - 1.0)), e) *
                    (1.0 - gamma + 2.0 * gamma * M2) / (gamma + 1.0);
    } else {
        p0_over_p = std::pow(1.0 + 0.5 * (gamma - 1.0) * M2, e);
    }
    return 2.0 / (gamma * M2) * (p0_over_p - 1.0);
}

/// Henderson (1976) sphere drag in the continuum (Re -> infinity) limit: the
/// subsonic branch 0.38 + 0.1 M^2 + 0.2 M^8 (M < 1), the supersonic branch
/// 0.9 + 0.34 / M^2 (M > 1.75), linear in M between.
[[nodiscard]] inline double henderson_continuum_sphere(double mach) {
    auto sub = [](double M) { return 0.38 + 0.1 * M * M + 0.2 * std::pow(M, 8); };
    auto sup = [](double M) { return 0.9 + 0.34 / (M * M); };
    if (mach < 1.0) return sub(mach);
    if (mach > 1.75) return sup(mach);
    return sub(1.0) + 4.0 / 3.0 * (mach - 1.0) * (sup(1.75) - sub(1.0));
}

/// Henderson (1976) full correlation, Re- and Mach-dependent, continuum to
/// rarefied (Singh et al. 2020, eq. B1). Used as an independent check of the
/// flow-regime sphere, not by it. Re on the diameter; tw_t = wall / gas temperature.
[[nodiscard]] inline double henderson1976_sphere(double mach, double re, double tw_t) {
    const double gamma = kGammaAir;
    auto subsonic = [&](double M) {
        const double s = M * std::sqrt(gamma / 2.0);
        const double sre = std::sqrt(re);
        return 24.0 / (re + s * (4.33 + (3.65 - 1.53 * tw_t) / (1.0 + 0.353 * tw_t) * std::exp(-0.247 * re / s))) +
               std::exp(-0.5 * M / sre) *
                   ((4.5 + 0.38 * (0.03 * re + 0.48 * sre)) / (1.0 + 0.03 * re + 0.48 * sre) + 0.1 * M * M +
                    0.2 * std::pow(M, 8)) +
               (1.0 - std::exp(-M / re)) * 0.6 * s;
    };
    auto supersonic = [&](double M) {
        const double s = M * std::sqrt(gamma / 2.0);
        const double r = 1.86 * std::sqrt(M / re);
        return (0.9 + 0.34 / (M * M) + r * (2.0 + 2.0 / s + 1.058 / s * std::sqrt(tw_t) - 1.0 / std::pow(s, 4))) /
               (1.0 + r);
    };
    if (mach < 1.0) return subsonic(mach);
    if (mach > 1.75) return supersonic(mach);
    return subsonic(1.0) + 4.0 / 3.0 * (mach - 1.0) * (supersonic(1.75) - subsonic(1.0));
}

/// Continuum Cd from modified Newtonian theory, Cp = Cp_max sin^2(theta),
/// windward surface only (base pressure neglected; valid hypersonically).
[[nodiscard]] inline double newtonian_cd(Shape shape, double mach) {
    const double cp = stagnation_cp(mach);
    switch (shape) {
        case Shape::Sphere: return 0.5 * cp;
        case Shape::FlatPlate: return cp;
        case Shape::Cylinder: return 2.0 / 3.0 * cp;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/// Continuum Cd as a function of Mach, given by the user (e.g. wind-tunnel
/// data for the actual body); linear between points. validate() throws
/// ConfigError for fewer than two points, mismatched lengths, a Cd <= 0 or
/// Mach values not strictly increasing; a lookup outside the table throws
/// DataCoverageError.
struct MachTable {
    std::vector<double> mach;
    std::vector<double> cd;

    void validate() const {
        if (mach.size() < 2 || mach.size() != cd.size())
            throw ConfigError("MachTable: need at least two (mach, cd) points of equal count");
        for (std::size_t i = 0; i < mach.size(); ++i) {
            if (!(cd[i] > 0.0)) throw ConfigError("MachTable: Cd values must be > 0");
            if (i > 0 && !(mach[i] > mach[i - 1]))
                throw ConfigError("MachTable: Mach numbers must be strictly increasing");
        }
    }

    [[nodiscard]] double operator()(double m) const {
        if (m < mach.front() || m > mach.back()) {
            throw DataCoverageError("MachTable: Mach " + std::to_string(m) + " is outside the table (" +
                                    std::to_string(mach.front()) + " to " + std::to_string(mach.back()) +
                                    "); extend the table to the Mach range the flight reaches");
        }
        const auto it = std::upper_bound(mach.begin(), mach.end(), m);
        const std::size_t i = std::min<std::size_t>(static_cast<std::size_t>(it - mach.begin()), mach.size() - 1);
        const double f = (m - mach[i - 1]) / (mach[i] - mach[i - 1]);
        return cd[i - 1] + f * (cd[i] - cd[i - 1]);
    }
};

/// The continuum model used without a MachTable.
struct DefaultContinuum {};   // sphere: Henderson; plate / cylinder: modified Newtonian
using ContinuumModel = std::variant<DefaultContinuum, MachTable>;

// ---- The bridge ---------------------------------------------------------------

/// Weight of the free-molecular Cd in the transitional regime at Knudsen
/// number kn: 0 at or below kKnContinuum, 1 at or above kKnFreeMolecular,
/// sin^2(pi/8 (3 + log10 Kn)) between.
[[nodiscard]] inline double bridge_weight(double kn) {
    if (kn <= kKnContinuum) return 0.0;
    if (kn >= kKnFreeMolecular) return 1.0;
    const double x = std::sin(std::numbers::pi / 8.0 * (3.0 + std::log10(kn)));
    return x * x;
}

// ---- Cd models ------------------------------------------------------------------

/// A fixed drag coefficient (the usual choice in orbit).
struct ConstantCd {
    double cd = 2.2;
};

/// Cd computed from the flow: free-molecular, continuum and the bridge
/// between them, for the given shape. Needs the air's temperature and molar
/// mass (a live drag environment) and the body's reference length.
struct FlowRegimeCd {
    Shape shape = Shape::Sphere;
    double accommodation = 1.0;        // energy accommodation alpha (1 = fully diffuse; usual below ~300 km)
    double wall_temperature_k = 300.0;
    ContinuumModel continuum = DefaultContinuum{};
};

using CdModel = std::variant<ConstantCd, FlowRegimeCd>;

/// The air around the body at one instant: speed relative to the air [m/s],
/// density [kg/m³], temperature [K] and mean molar mass [kg/kmol].
struct FlowState {
    double speed_m_s = 0.0;            // speed relative to the air
    double density_kg_m3 = 0.0;
    double temperature_k = std::numeric_limits<double>::quiet_NaN();
    double molar_mass_kg_kmol = std::numeric_limits<double>::quiet_NaN();
};

/// The drag coefficient used, with the Knudsen and Mach numbers it was
/// computed at (NaN for a ConstantCd).
struct CdResult {
    double cd = std::numeric_limits<double>::quiet_NaN();
    double knudsen = std::numeric_limits<double>::quiet_NaN();
    double mach = std::numeric_limits<double>::quiet_NaN();
};

/// Cd for the flow `f` around a body of reference length `ref_length_m` [m].
/// Throws ConfigError when `f` has no temperature or molar mass (a tabulated
/// density), DataCoverageError from a MachTable outside its range; reports an
/// Extrapolation diagnostic when modified Newtonian theory is used below
/// Mach 3.
[[nodiscard]] inline CdResult flow_regime_cd(const FlowRegimeCd& model, const FlowState& f, double ref_length_m) {
    if (!(std::isfinite(f.temperature_k) && std::isfinite(f.molar_mass_kg_kmol))) {
        throw ConfigError(
            "FlowRegimeCd needs the air's temperature and molar mass, and this drag environment's density "
            "column gives density only (a table). Use make_drag_environment (live NRLMSIS 2.1, with U.S. "
            "Standard Atmosphere 1976 below an extend_density_below floor), or a ConstantCd.");
    }
    CdResult r;
    r.mach = f.speed_m_s / speed_of_sound(f.temperature_k, f.molar_mass_kg_kmol);
    r.knudsen = mean_free_path(f.density_kg_m3, f.molar_mass_kg_kmol) / ref_length_m;
    const double w = bridge_weight(r.knudsen);
    double cd_fm = 0.0, cd_c = 0.0;
    if (w > 0.0) {
        const double s = speed_ratio(f.speed_m_s, f.temperature_k, f.molar_mass_kg_kmol);
        const double vr = reemission_ratio(f.speed_m_s, f.molar_mass_kg_kmol, model.accommodation,
                                           model.wall_temperature_k);
        cd_fm = free_molecular_cd(model.shape, s, vr);
    }
    if (w < 1.0) {
        if (const auto* table = std::get_if<MachTable>(&model.continuum)) {
            cd_c = (*table)(r.mach);
        } else if (model.shape == Shape::Sphere) {
            cd_c = henderson_continuum_sphere(r.mach);
        } else {
            if (r.mach < 3.0) {
                diagnostics::report(
                    diagnostics::Category::Extrapolation,
                    std::string("FlowRegimeCd: modified Newtonian theory (the continuum model of a ") +
                        shape_name(model.shape) + ") is used below Mach 3, where it is not valid (it is a "
                        "hypersonic theory; a subsonic " + shape_name(model.shape) + " has a markedly "
                        "different Cd). Give a MachTable of measured Cd for the body to cover this range.",
                    "flow-regime-newtonian-below-mach-3");
            }
            cd_c = newtonian_cd(model.shape, r.mach);
        }
    }
    r.cd = cd_c + (cd_fm - cd_c) * w;
    return r;
}

/// Ballistic coefficient from the body: beta = m / (Cd A) [kg/m²], with Cd
/// from a model. One of the beta column types of DragEnvironment (alongside
/// BetaProfile1D/2D). Default-constructed (mass 0) = empty.
struct AeroBeta {
    double mass_kg = 0.0;
    double reference_area_m2 = 0.0;
    double reference_length_m = 0.0;   // for the Knudsen number (FlowRegimeCd only)
    CdModel cd = ConstantCd{};

    /// Throws ConfigError for a non-positive mass, area or Cd, a FlowRegimeCd
    /// without a positive reference length, accommodation outside [0, 1], a
    /// non-positive wall temperature, or an invalid MachTable.
    void validate() const {
        if (!(mass_kg > 0.0) || !(reference_area_m2 > 0.0))
            throw ConfigError("AeroBeta: mass and reference area must be > 0");
        if (const auto* fr = std::get_if<FlowRegimeCd>(&cd)) {
            if (!(reference_length_m > 0.0))
                throw ConfigError("AeroBeta: FlowRegimeCd needs reference_length_m > 0 (the body's size, "
                                  "for the Knudsen number: a sphere's diameter)");
            if (!(fr->accommodation >= 0.0 && fr->accommodation <= 1.0))
                throw ConfigError("FlowRegimeCd: accommodation must be in [0, 1]");
            if (!(fr->wall_temperature_k > 0.0))
                throw ConfigError("FlowRegimeCd: wall_temperature_k must be > 0");
            if (const auto* t = std::get_if<MachTable>(&fr->continuum)) t->validate();
        } else if (!(std::get<ConstantCd>(cd).cd > 0.0)) {
            throw ConfigError("AeroBeta: ConstantCd must be > 0");
        }
    }

    /// True when no body is set (mass not positive); depends_on_flow() is true
    /// for a FlowRegimeCd (β then changes with speed, density and temperature).
    [[nodiscard]] bool empty() const noexcept { return !(mass_kg > 0.0); }
    [[nodiscard]] bool depends_on_flow() const noexcept { return std::holds_alternative<FlowRegimeCd>(cd); }

    /// beta [kg/m^2] and the Cd that produced it, for the flow `f`. Same
    /// throws as flow_regime_cd for a FlowRegimeCd.
    [[nodiscard]] std::pair<double, CdResult> evaluate(const FlowState& f) const {
        CdResult r;
        if (const auto* c = std::get_if<ConstantCd>(&cd)) {
            r.cd = c->cd;
        } else {
            r = flow_regime_cd(std::get<FlowRegimeCd>(cd), f, reference_length_m);
        }
        return {mass_kg / (r.cd * reference_area_m2), r};
    }
};

}  // namespace aero
}  // namespace ballistic
