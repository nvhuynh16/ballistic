#pragma once

// =============================================================================
// GNSS satellite blocks: radiation-pressure inputs per spacecraft type
// =============================================================================
//
// A GNSS satellite's radiation-pressure model depends on its block (the
// spacecraft type), which the public IGS satellite metadata file gives per
// satellite together with its in-orbit mass (scripts/_gnss_metadata.py reads
// it). That file has no geometry, so the areas live here, keyed on the IGS
// block names ("GPS-IIF", "QZS-2I", "BDS-3M-CAST", ...).
//
// The model is a cannonball: one Sun-facing cross-section per block, used as
// SrpConfig::area_m2 with the satellite's own mass. By convention the base
// area is the total solar-array area plus the body's +X face. Under the
// yaw-steering attitude of most GNSS satellites the arrays face the Sun and the
// Sun stays in the body's X-Z plane, so this is the order of the lit area; the
// rest (Z face, oblique incidence, reflection properties) is absorbed by a
// scale k fitted to orbits (kGnssSrpFit). The effective area is
// srp_base_area_m2 * srp_k.
//
// srp_kind says where the base area comes from (satellites.hpp FitKind):
// Physical = read from a published spacecraft document; Nominal = a
// placeholder or a value only partly sourced (the provenance string says
// which part is assumed); Fitted = a k from kGnssSrpFit has been applied.
// mass_kg_nominal is for orientation only: the campaigns take each
// satellite's mass from the IGS metadata file; 0 means no value is recorded
// here.
// =============================================================================

#include "core/errors.hpp"
#include "satellites.hpp"

#include <array>
#include <span>
#include <string>
#include <string_view>

namespace ballistic {
namespace gnss {

using satellites::FitKind;

/// One GNSS block (spacecraft type) with its cannonball radiation inputs.
struct Block {
    std::string_view name;        ///< IGS block name, or a prefix of several ("GPS-IIR")
    double mass_kg_nominal;       ///< typical mass [kg]; 0 = none recorded (use the IGS file)
    double srp_base_area_m2;      ///< solar arrays + body +X face [m^2]
    double cr;                    ///< radiation pressure coefficient (1.3 nominal)
    FitKind srp_kind;             ///< how the area was obtained (Fitted once a k applies)
    std::string_view provenance;  ///< source of every number in the row
    double srp_k = 1.0;           ///< fitted area scale (kGnssSrpFit), 1 when not fitted

    /// Effective cannonball area srp_base_area_m2 * srp_k [m^2].
    [[nodiscard]] constexpr double srp_area_m2() const noexcept { return srp_base_area_m2 * srp_k; }
};

/// A fitted radiation-area scale for one block (effective area = base * k).
struct GnssSrpFit {
    std::string_view block;
    double k;
};

/// The block table before any fit is applied.
inline constexpr std::array<Block, 16> kBlocksBase{{
    {"GPS-IIR", 1080.0, 17.1, 1.3, FitKind::Nominal,
        "IIR-A and IIR-B (one bus). Arrays 13.4 m^2 = 144 ft^2 and body 5 x 6.3 x 6.25 ft from the "
        "U.S. Space Force fact sheet 'GPS IIR/IIR-M' (losangeles.spaceforce.mil), as quoted by a web "
        "search; the page itself refused automated access, so the figures are not checked against "
        "it. Body +X face assumed to be the 6.3 x 6.25 ft face (3.7 m^2). Mass: IGS metadata file "
        "value for SVN G061 (IIR-B)."},
    {"GPS-IIR-M", 1080.0, 17.1, 1.3, FitKind::Nominal,
        "Same bus and arrays as GPS-IIR (the fact sheet covers IIR and IIR-M together); same caveat. "
        "Mass assumed equal to IIR's; the campaigns use the IGS file's per-satellite mass."},
    {"GPS-IIF", 1633.0, 27.7, 1.3, FitKind::Nominal,
        "Body +X face 5.72 m^2: Dilssner, Springer, Schoenemann and Enderle, 'Initial orbit "
        "determination of third-generation BeiDou MEO spacecraft', IGS Workshop 2018, slide 4 "
        "(files.igs.org). Solar-array area ASSUMED 22 m^2 (no public source read); to be refined. "
        "Mass: IGS metadata file value for SVN G063."},
    {"GPS-IIIA", 2161.0, 30.0, 1.3, FitKind::Nominal,
        "Placeholder: no public dimensions read. Base area ASSUMED 30 m^2 (arrays plus body, of the "
        "order of GPS-IIF scaled up with the mass); to be refined. Mass: IGS metadata file value "
        "for SVN G080."},
    {"QZS-1", 2281.0, 54.0, 1.3, FitKind::Physical,
        "Cabinet Office of Japan, 'QZS-1 Satellite Information' SPI_QZS1_B (qzss.go.jp): solar "
        "array panels 22.5 m^2 on each of +Y and -Y (45.0 m^2) + body +X face 9.0 m^2. Mass: "
        "beginning-of-life 2281 kg (same document, Table 1)."},
    {"QZS-2I", 2324.2, 39.9, 1.3, FitKind::Physical,
        "Cabinet Office of Japan, 'QZS-2 Satellite Information' SPI-QZS2_C (qzss.go.jp; QZS-4, "
        "SPI-QZS4_C, lists the same areas): solar array panels 14.9 m^2 on each of +Y and -Y "
        "(29.8 m^2) + body +X face 10.1 m^2. Mass: beginning-of-life 2324.2 kg (QZS-2)."},
    {"QZS-2A", 2369.3, 40.6, 1.3, FitKind::Physical,
        "QZS-1R. Cabinet Office of Japan, 'QZS-1R Satellite Information' SPI-QZS1R_A "
        "(qzss.go.jp): solar array panels 2 x 14.9 m^2 + body +X face 10.8 m^2. Mass: "
        "beginning-of-life 2369.3 kg."},
    {"QZS-2G", 2762.6, 39.9, 1.3, FitKind::Physical,
        "QZS-3 (geostationary). Cabinet Office of Japan, 'QZS-3 Satellite Information' SPI-QZS3_F "
        "(qzss.go.jp): solar array panels 2 x 14.9 m^2 + body +X face 10.1 m^2; the two "
        "reflectors on +X/-X (9.1 and 0.8 m^2 projected) are left out. Mass: beginning-of-life "
        "2762.6 kg."},
    {"QZS-3G", 0.0, 40.0, 1.3, FitKind::Nominal,
        "Placeholder: no satellite information document read for this block. Base area ASSUMED "
        "40 m^2, like QZS-2G; to be refined."},
    {"BDS-2G", 1500.0, 23.4, 1.3, FitKind::Nominal,
        "Body +X face 3.44 m^2 taken from the BDS-2 MEO row of Dilssner et al., IGS Workshop 2018, "
        "slide 4 (assumed the same bus). Solar-array area ASSUMED 20 m^2 (no public source read); "
        "to be refined. These satellites keep an orbit-normal attitude, so the lit area varies "
        "more than under yaw steering. Mass: IGS metadata file, C01-C05 carry 1486-1551 kg."},
    {"BDS-2I", 0.0, 23.4, 1.3, FitKind::Nominal,
        "Body +X face 3.44 m^2 from the BDS-2 MEO row of Dilssner et al., IGS Workshop 2018, "
        "slide 4 (assumed the same bus). Solar-array area ASSUMED 20 m^2; to be refined."},
    {"BDS-2M", 0.0, 23.4, 1.3, FitKind::Nominal,
        "Body +X face 3.44 m^2: Dilssner et al., IGS Workshop 2018, slide 4 (BDS-2M row). "
        "Solar-array area ASSUMED 20 m^2 (no public source read); to be refined."},
    {"BDS-3M-CAST", 1014.0, 28.2, 1.3, FitKind::Nominal,
        "Body +X face 1.22 m^2 and mass 1014 kg: Dilssner et al., IGS Workshop 2018, slide 4. "
        "Solar-array area ~27 m^2: the same talk's estimate from the fitted radiation "
        "acceleration (slide 11), not a manufacturer value."},
    {"BDS-3M-SECM", 1030.0, 14.7, 1.3, FitKind::Nominal,
        "SECM-A and SECM-B. Body +X face 1.24 m^2 and mass 1030 kg: Dilssner et al., IGS Workshop "
        "2018, slide 4. Solar-array area ASSUMED 13.5 m^2 = half the CAST estimate (slide 11: "
        "the CAST radiation acceleration is twice SECM's); to be refined."},
    {"BDS-3I", 0.0, 30.0, 1.3, FitKind::Nominal,
        "Placeholder: no public dimensions read. Base area ASSUMED 30 m^2; to be refined."},
    {"BDS-3G", 0.0, 30.0, 1.3, FitKind::Nominal,
        "Placeholder: no public dimensions read. Base area ASSUMED 30 m^2; to be refined."},
}};

/// Radiation-area scales fitted to orbits (scripts/validate_gnss.py --fit-srp,
/// claim GNSS-1H-SRP): grid 0.50-1.60 in steps of 0.05, one-hour RMSE
/// minimized on 2024 days 015 and 017 and scored on days 016 and 018 of the
/// committed CODE (GPS, QZSS) and Wuhan (BeiDou GEO) final orbits. A k near 1
/// says the base area with Cr 1.3 already carries the right acceleration; it
/// also absorbs the Cr and attitude simplifications, so a k belongs to this
/// model, not to the spacecraft.
inline constexpr std::array<GnssSrpFit, 7> kGnssSrpFit{{
    {"GPS-IIR", 1.10},      // 7 satellites (IIR-A and IIR-B)
    {"GPS-IIR-M", 1.05},    // 7 satellites
    {"GPS-IIF", 1.10},      // 11 satellites
    {"GPS-IIIA", 1.00},     // 6 satellites
    {"QZS-2I", 1.05},       // J02, J03
    {"QZS-2A", 1.00},       // J04
    {"BDS-2G", 1.25},       // C01-C05
}};

/// Every fitted block names a row of kBlocksBase, once, with a positive k.
[[nodiscard]] constexpr bool srp_fit_consistent() noexcept {
    for (std::size_t i = 0; i < kGnssSrpFit.size(); ++i) {
        const auto& f = kGnssSrpFit[i];
        if (!(f.k > 0.0)) return false;
        bool found = false;
        for (const auto& b : kBlocksBase) found = found || (b.name == f.block);
        if (!found) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (kGnssSrpFit[j].block == f.block) return false;
    }
    return true;
}
static_assert(srp_fit_consistent(),
              "gnss_blocks: a kGnssSrpFit block is not in kBlocksBase, appears twice, or has k <= 0");

/// The block table with the fits applied (srp_k set, srp_kind = Fitted).
inline constexpr std::array<Block, kBlocksBase.size()> kBlocks = [] {
    auto out = kBlocksBase;
    for (auto& b : out)
        for (const auto& f : kGnssSrpFit)
            if (f.block == b.name) {
                b.srp_k = f.k;
                b.srp_kind = FitKind::Fitted;
            }
    return out;
}();

/// Every block.
[[nodiscard]] inline std::span<const Block> all() noexcept { return kBlocks; }

/// The row for an IGS block name, or nullptr. An exact name wins; otherwise
/// the longest row name that is a prefix of `igs_block_name` followed by '-'
/// ("GPS-IIR-A" and "GPS-IIR-B" -> "GPS-IIR"; "BDS-3M-SECM-A" -> "BDS-3M-SECM").
[[nodiscard]] constexpr const Block* find_block(std::string_view igs_block_name) noexcept {
    const Block* best = nullptr;
    for (const auto& b : kBlocks) {
        if (b.name == igs_block_name) return &b;
        if (igs_block_name.size() > b.name.size() && igs_block_name.starts_with(b.name) &&
            igs_block_name[b.name.size()] == '-' &&
            (best == nullptr || b.name.size() > best->name.size()))
            best = &b;
    }
    return best;
}

/// Like find_block, but throws ConfigError naming the known blocks.
[[nodiscard]] constexpr const Block& get_block(std::string_view igs_block_name) {
    if (const Block* b = find_block(igs_block_name)) return *b;
    std::string known;
    for (const auto& b : kBlocks) {
        if (!known.empty()) known += ", ";
        known += b.name;
    }
    throw ConfigError("unknown GNSS block '" + std::string(igs_block_name) + "' (known: " + known +
                      "; a name may also extend a known one after '-', e.g. GPS-IIR-A). Give "
                      "the radiation area and mass explicitly (SrpConfig) for other spacecraft.");
}

}  // namespace gnss
}  // namespace ballistic
