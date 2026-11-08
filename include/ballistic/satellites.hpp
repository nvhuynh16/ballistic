#pragma once

// =============================================================================
// Satellite properties: the single source of truth
// =============================================================================
//
// Every satellite the library knows by name, with mass, drag and radiation
// areas, and a citation for each value. C++ code, the Python package
// (ballistic.satellite_properties, built from this table through the
// bindings), the validation scripts, tests and examples all read these
// values; no copy may live anywhere else (tests/python/test_static_guards.py,
// rule SAT-LITERAL). Unknown names throw ConfigError: a silent default can
// hide a several-fold error in the ballistic coefficient (for example from
// confusing the frontal area with the total surface area).
//
// drag_area_m2 is the mean FRONTAL area facing the direction of flight
// ("ram-facing"), never the total or panel area. Orbit data constrain only
// the ballistic coefficient beta = m / (Cd * A) (mass over drag coefficient
// times area; a larger beta means drag slows the satellite less). Where Cd
// is fitted (CHAMP, GOCE), the split between Cd and area is a convention and
// Cd is not a physical drag coefficient
// (docs/satellite_properties_provenance.md).
//
// srp_area_m2 is the EFFECTIVE radiation area of a "cannonball" model (the
// satellite treated as a sphere of that cross-section) for solar radiation
// pressure (SRP) plus the Earth's reflected sunlight (albedo) and infrared
// (IR), at radiation coefficient Cr = 1 + srp_reflectivity. For the
// Sentinels it is a base area times a scale k fitted on all of 2025
// (kSrpFit below; SENTINEL-YEAR-2025: fitted on odd days, scored on even
// days; a refit under the current force model did not change them).
// =============================================================================

#include "core/errors.hpp"

#include <array>
#include <span>
#include <string>
#include <string_view>

namespace ballistic {
namespace satellites {

/// How a registry value was obtained, so a caller can tell a measured value
/// from one that only fits the orbits it was tuned on.
enum class FitKind {
    Physical,  ///< from mission documents or a physical model of the satellite
    Fitted,    ///< adjusted to orbit data, so it also absorbs force-model error
    Nominal,   ///< a textbook value (e.g. Cd 2.2, Cr 1.13), not checked for this satellite
    Legacy,    ///< unsourced, low confidence: pass your own values if you have them
};

/// Lower-case name of a FitKind ("physical", "fitted", "nominal", "legacy").
[[nodiscard]] constexpr std::string_view fit_kind_name(FitKind k) noexcept {
    switch (k) {
        case FitKind::Physical: return "physical";
        case FitKind::Fitted: return "fitted";
        case FitKind::Nominal: return "nominal";
        case FitKind::Legacy: return "legacy";
    }
    return "";
}

/// One satellite's physical properties, with a citation for each value.
struct SatelliteProperties {
    std::string_view name;
    double mass_kg;
    double drag_area_m2;       // mean ram-facing FRONTAL area
    double cd;                 // drag coefficient (fitted for CHAMP and GOCE)
    double srp_area_m2;        // effective cannonball radiation area
    double srp_reflectivity;   // Cr = 1 + srp_reflectivity
    double altitude_km;        // typical altitude (presets only; not used in physics)
    int gravity_degree;        // recommended EGM2008 degree
    FitKind drag_kind;         // how mass, drag area and Cd were obtained (drag columns)
    FitKind srp_kind;          // how the radiation area and Cr were obtained
    std::string_view provenance;   // where each value comes from

    /// Ballistic coefficient beta = mass / (Cd * drag area) [kg/m^2].
    [[nodiscard]] constexpr double beta_kg_m2() const noexcept {
        return mass_kg / (cd * drag_area_m2);
    }
    /// Radiation pressure coefficient Cr (1 = absorbs all light).
    [[nodiscard]] constexpr double cr() const noexcept { return 1.0 + srp_reflectivity; }
};

/// Radiation-area scale fitted per Sentinel: srp_area_m2 = base_area_m2 * k,
/// rounded to 0.1 m^2 (tests/python/test_satellite_properties.py checks it).
struct SrpFit {
    std::string_view name;
    double base_area_m2;
    double k;
};

/// The registry.
inline constexpr std::array<SatelliteProperties, 18> kSatellites{{
    {"CHAMP", 522.0, 0.9, 2.93, 0.5, 0.3,
        380.0, 70, FitKind::Fitted, FitKind::Legacy,
        "Cd FITTED 2026-11-01 under the default space weather (the drag campaigns per-arc weather"
            " snapshot had hidden ~16% of the shortfall): beta x0.75 (Cd*A 1.98 -> 2.64 m^2, beta 263"
            ".6 -> 197.9), best on the fitting day 2005-182 (1.82 -> 1.17 m; x0.75-0.8 within 2 cm). "
            "Checked on the RSO file 2005-06-30/07-01 (2.13 -> 1.03 m), which is the adjacent half-da"
            "y and overlaps the fit by 2 h (not an independent test), and on the 2010 final weeks at "
            "250-280 km near solar minimum (18.5 -> 8.8 m; same factor best there, but that set was s"
            "wept too, so not blind). NRLMSIS matches measured GOCE density on the same weather (0.97"
            "-1.09), so the factor is attributed to Cd*A, not density. VALIDATED QUANTITY = the PRODU"
            "CT Cd*A: that is what the drag campaigns constrain. The SPLIT into A = 0.9 m^2 (eoPortal"
            " CHAMP mission page, trapezoidal body cross-section, retrieved 2026-07-19) and Cd = 2.93"
            " is a DECOMPOSITION, not an independent measurement — any pair with the same product fit"
            "s every measurement equally well; errors in mass or area land in Cd, so Cd is not physic"
            "al. Mass 522 kg = launch mass (eoPortal); the 2005 mass was lower by the gas used, no ma"
            "ss history found. A drag_area of 3.5 m^2 would be a total-area-class error."},
    {"GRACE-FO-1", 595.9, 1.0, 3.56, 3.5, 0.3,
        500.0, 70, FitKind::Physical, FitKind::Legacy,
        "PHYSICAL values, not fitted to orbit data (2026-11-01). Mass 595.9 kg = launch mass 601."
            "214 kg (GRACE-FO L1 Data Product User Handbook, JPL D-56935, 2019-09-11, Table 4) minus "
            "tank gas used: 31.26 kg on 2018-05-23 -> 25.94 kg in Jan 2026 (Level-1B MAS1B, GRACE-C, "
            "GFZ ISDC). Area 1.0 m^2 = front panel 0.955 + boom 0.046 m^2 (handbook Table 5). Cd 3.56"
            " = Sentman (1961) on all nine Table-5 panels, flow along +X, mass-weighted over NRLMSIS "
            "2.1 species along the 2026-01-01..03 orbits (T ~1100 K, mean mass 15.6 amu), energy acco"
            "mmodation 0.93 (the GRACE value of the TND-IGG RL01 density product), wall 300 K (Sentm"
            "an panel model over the handbook panels). Cd is far above the flat-plate 2.2 because the 15 m^2 o"
            "f side, nadir and zenith panels lie nearly parallel to the flow and add drag at grazing "
            "incidence. Accommodation 0.85-1.0 gives Cd 3.71-3.33 (+-4%). Replaces the one-day orbit "
            "fit Cd 4.4 at an uncited 580 kg."},
    {"GRACE-FO-2", 596.1, 1.0, 3.56, 3.5, 0.3,
        500.0, 70, FitKind::Physical, FitKind::Legacy,
        "Twin of GRACE-FO-1 (identical panels, same Cd). Mass 596.1 kg = launch 601.209 kg (handb"
            "ook Table 4) minus tank gas used: 31.29 kg on 2018-05-23 -> 26.17 kg in Jan 2026 (Level-"
            "1B MAS1B, GRACE-D)."},
    {"GOCE", 1002.2, 1.1, 4.095, 2.0, 0.3,
        250.0, 120, FitKind::Fitted, FitKind::Legacy,
        "VALIDATED QUANTITY = beta 222.5 kg/m^2, FITTED 2026-10-31 (at the then-tabulated 1077 kg"
            ": drag scale 2.0 x the nominal 2.2 x 1.1 m^2), chosen on 16 free-decay days (2013-296..3"
            "04, 306..312) with both GOCE accelerometer density and NRLMSIS (best k 2.0 for both abov"
            "e 180 km), then scored on the GOCE-REENTRY days 295/305/313: NRLMSIS drag 82.5/181/891 -> 17.2/18."
            "6/159 m by band. The split into A = 1.1 m^2 frontal (ESA: \"slim 5 m satellite, cross se"
            "ction ~1 m^2\"; eoPortal GOCE) and Cd is a convention; a high Cd is plausible for a long"
            " finned body in free-molecular flow, but only beta is measured. Below ~180 km (final day"
            "s) no constant fits well (attitude-control limits). MASS 1002.2 kg (2026-11-01) = ESA GO"
            "CE Mass Properties file (earth.esa.int eogateway, public): constant from 2013-10-21 (xen"
            "on exhausted) through the GOCE-REENTRY and calibration days; 1050.8 kg at launch. With beta kept, "
            "Cd*A = 4.50 m^2 and Cd = 4.095. PREVIOUS 1077 kg (a commonly quoted launch figure, 7% ab"
            "ove the 2013 mass) and 977 kg (unsourced) are superseded."},
    {"S1A", 2200.0, 6.35, 2.2, 38.4, 0.0,
        700.0, 70, FitKind::Nominal, FitKind::Fitted,
        "frontal 6.35 m^2: GMV-GMESPOD-TN-0025 v1.4 (2019) \"Sentinel-1 properties for GPS POD\","
            " Table 5-1 (sentinels.copernicus.eu), retrieved 2026-07-19. Mass ~2200 kg mid-life (POD "
            "uses mass-history files). SRP area 38.4 m^2 = base 20 m^2 x fitted k 1.92 "
            "(SENTINEL-YEAR-2025 srp-fit, 2026-10-30)."},
    {"S1C", 2200.0, 6.35, 2.2, 38.6, 0.0,
        700.0, 70, FitKind::Nominal, FitKind::Fitted,
        "same platform as S1A (GMV-GMESPOD-TN-0025); SRP area = 20 m^2 x fitted k 1.93."},
    {"S2B", 1200.0, 4.2, 2.2, 16.6, 0.0,
        790.0, 70, FitKind::Legacy, FitKind::Fitted,
        "legacy values (unsourced frontal area; the CPOD S2 properties document is not yet obtai"
            "ned). SRP area = base 11 m^2 x fitted k 1.51."},
    {"S2C", 1200.0, 4.2, 2.2, 16.2, 0.0,
        790.0, 70, FitKind::Legacy, FitKind::Fitted,
        "same platform as S2B (Sentinel-2 twin; legacy values); SRP area = 11 m^2 x fitted k 1.47."},
    {"S3A", 1250.0, 4.8, 2.2, 22.7, 0.0,
        815.0, 70, FitKind::Legacy, FitKind::Fitted,
        "legacy values (unsourced frontal area; the CPOD S3 properties document is not yet obtai"
            "ned). SRP area = base 14 m^2 x fitted k 1.62."},
    {"S3B", 1250.0, 4.8, 2.2, 22.7, 0.0,
        815.0, 70, FitKind::Legacy, FitKind::Fitted,
        "same platform as S3A (Sentinel-3 twin; legacy values); SRP area = 14 m^2 x fitted k 1.62."},
    {"S6A", 1192.0, 3.5, 2.2, 11.4, 0.0,
        1336.0, 70, FitKind::Nominal, FitKind::Fitted,
        "mass 1192 kg = launch mass incl. 230 kg hydrazine (JPL Sentinel-6 press-kit quick facts;"
            " ESA Copernicus satellite description), current mass unknown. SRP 11.1 m^2 = 44.3/4 from"
            " the DLR macro-model, Calliess et al. (2024) Adv. Space Res. 74:3011, Table 4 (+-X 3.5, "
            "+Z 15.5, -Z 2.0, 2x solar panel 8.9, +-Y 1.0 m^2), retrieved 2026-10-30; x fitted k 1.03"
            " = 11.4 (the fit confirms the macro-model to 3%). Frontal 3.5 m^2 = body +X face of the "
            "same table (drag negligible at 1336 km)."},
    {"SWARM-A", 470.0, 1.0, 2.2, 4.0, 0.3,
        460.0, 70, FitKind::Legacy, FitKind::Legacy,
        "legacy values; UNVALIDATED in-repo (Swarm SP3 is registration-gated). Literature frontal ~0"
            ".9-1.1 m^2 class (Swarm density studies)."},
    {"SWARM-B", 470.0, 1.0, 2.2, 4.0, 0.3,
        530.0, 70, FitKind::Legacy, FitKind::Legacy,
        "legacy values; UNVALIDATED in-repo (Swarm SP3 is registration-gated). Literature frontal ~0"
            ".9-1.1 m^2 class (Swarm density studies)."},
    {"SWARM-C", 470.0, 1.0, 2.2, 4.0, 0.3,
        460.0, 70, FitKind::Legacy, FitKind::Legacy,
        "legacy values; UNVALIDATED in-repo (Swarm SP3 is registration-gated). Literature frontal ~0"
            ".9-1.1 m^2 class (Swarm density studies)."},
    {"LAGEOS-1", 406.965, 0.2827, 2.2, 0.2827, 0.13,
        5890.0, 8, FitKind::Physical, FitKind::Nominal,
        "Mass 406.965 kg and 60 cm diameter: NASA Earthdata LAGEOS-1 platform page and ILRS missi"
            "on page (retrieved 2026-11-02); an older value of 411 kg was wrong. Areas = pi (0.30 "
            "m)^2 = 0.2827 m^2 (sphere: frontal = radiation cross-section). Cr = 1.13: the convention"
            "al SLR POD value (unsourced here). Drag negligible at 5,900 km; Cd 2.2 nominal."},
    {"LAGEOS-2", 405.38, 0.2827, 2.2, 0.2827, 0.13,
        5620.0, 8, FitKind::Physical, FitKind::Nominal,
        "Mass 405.38 kg and 60 cm diameter: ILRS LAGEOS-2 mission page (retrieved 2026-11-02); th"
            "e older value of 405 kg agreed to 0.1%. Areas = pi (0.30 m)^2 = 0.2827 m^2. Cr = 1.13 "
            "conventional (unsourced here). Drag negligible; Cd 2.2 nominal."},
    {"ETALON-1", 1415.0, 1.315, 2.2, 1.315, 0.13,
        19100.0, 4, FitKind::Physical, FitKind::Nominal,
        "Mass 1415 kg and diameter 1.294 m: ILRS Etalon-1/-2 mission page (retrieved 2026-11-02)."
            " Areas = pi (0.647 m)^2 = 1.315 m^2. Cr = 1.13 (conventional, unsourced). Drag negligible a"
            "t 19,100 km."},
    {"ETALON-2", 1415.0, 1.315, 2.2, 1.315, 0.13,
        19100.0, 4, FitKind::Physical, FitKind::Nominal,
        "Same platform as ETALON-1 (ILRS Etalon-1/-2 mission page: 1415 kg, 1.294 m)."},
}};

/// The fitted radiation scales behind the Sentinel srp_area_m2 values.
inline constexpr std::array<SrpFit, 7> kSrpFit{{
    {"S1A", 20.0, 1.92},
    {"S1C", 20.0, 1.93},
    {"S2B", 11.0, 1.51},
    {"S2C", 11.0, 1.47},
    {"S3A", 14.0, 1.62},
    {"S3B", 14.0, 1.62},
    {"S6A", 11.1, 1.03},
}};

[[nodiscard]] constexpr bool valid_fit_kind(FitKind k) noexcept {
    return !fit_kind_name(k).empty();
}

/// Every row's kinds are valid enum values, and every satellite in kSrpFit is
/// in the registry with srp_kind == Fitted (its area is a fit, so it must say so).
[[nodiscard]] constexpr bool fit_kinds_consistent() noexcept {
    for (const auto& s : kSatellites)
        if (!valid_fit_kind(s.drag_kind) || !valid_fit_kind(s.srp_kind)) return false;
    for (const auto& f : kSrpFit) {
        bool found = false;
        for (const auto& s : kSatellites)
            if (s.name == f.name) found = (s.srp_kind == FitKind::Fitted);
        if (!found) return false;
    }
    return true;
}

static_assert(fit_kinds_consistent(),
              "satellite registry: a FitKind is invalid, or a kSrpFit satellite is missing "
              "or not marked srp_kind = FitKind::Fitted");

/// Every satellite in the registry.
[[nodiscard]] inline std::span<const SatelliteProperties> all() noexcept { return kSatellites; }

/// Look up a satellite by its registry name ("S1A", "CHAMP", "GRACE-FO-1",
/// ...). Throws ConfigError naming the known satellites. It is constexpr, so
/// `constexpr double m = satellites::get("S1A").mass_kg;` works and an
/// unknown name in a constant expression fails at compile time.
[[nodiscard]] constexpr const SatelliteProperties& get(std::string_view name) {
    for (const auto& s : kSatellites)
        if (s.name == name) return s;
    std::string known;
    for (const auto& s : kSatellites) {
        if (!known.empty()) known += ", ";
        known += s.name;
    }
    throw ConfigError("unknown satellite '" + std::string(name) + "' (known: " + known +
                      "). Give mass and areas explicitly for other satellites.");
}

}  // namespace satellites
}  // namespace ballistic
