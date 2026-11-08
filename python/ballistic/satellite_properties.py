"""Satellite physical properties, read from the library's single source.

The values live in one C++ table, include/ballistic/satellites.hpp, which
C++ code, these Python objects, the validation scripts, tests and examples
all read. Each entry carries its citation; the human-readable record is
docs/satellite_properties_provenance.md.

`drag_area_m2` is the mean FRONTAL area, the area facing the direction of
motion, never the total surface area of all panels: a total-area-sized
3.5 m² for CHAMP instead of ~0.9 m² makes the ballistic coefficient
β = m / (Cd·A) (mass over drag coefficient times area, kg/m²) 4× too small,
and an atmosphere model run with too-quiet space weather can hide the error.
Orbit data constrain only β; where Cd is fitted (CHAMP, GOCE) how β is split
between Cd and area is a convention.

`srp_area_m2` is the area used for solar radiation pressure (SRP, the push of
sunlight), treating the satellite as a sphere (a "cannonball").

Each entry says how its values were obtained, separately for the drag
columns (`drag_kind`) and the radiation columns (`srp_kind`), as one of
FIT_KINDS:

* "physical": from mission documents or a physical model of the satellite;
* "fitted": adjusted to orbit data, so it also absorbs force-model error and
  may not carry over to other dates or models;
* "nominal": a textbook value (e.g. Cd 2.2), not checked for this satellite;
* "legacy": unsourced, low confidence — pass your own values if you have
  them (``recommended_config`` says so with a ConfigHintWarning).

Unknown names raise KeyError: a silent default would hide exactly that kind
of error.
"""

from __future__ import annotations

import dataclasses

from . import _core


@dataclasses.dataclass(frozen=True)
class SatelliteProperties:
    """One satellite's physical properties (read-only).

    Attributes:
        name: registry name, e.g. "S1A".
        mass_kg: mass, kg.
        drag_area_m2: mean frontal area for drag, m².
        cd: drag coefficient (dimensionless).
        srp_area_m2: effective area for radiation pressure, m².
        srp_reflectivity: Cr - 1, where Cr is the radiation-pressure
            coefficient (1 = absorbs all light).
        altitude_km: typical altitude, km.
        gravity_degree: suggested gravity-field degree.
        drag_kind: how mass, drag area and cd were obtained, one of
            FIT_KINDS ("physical", "fitted", "nominal", "legacy"; see the
            module docstring).
        srp_kind: how srp_area_m2 and srp_reflectivity were obtained, one
            of FIT_KINDS.
        provenance: the citation the values come from.
    """
    name: str
    mass_kg: float
    drag_area_m2: float        # mean ram-facing FRONTAL area (see module doc)
    cd: float
    srp_area_m2: float         # EFFECTIVE cannonball radiation area (SRP +
                               # albedo + IR) at Cr = 1 + srp_reflectivity;
                               # FITTED for the Sentinels (see SRP_FIT)
    srp_reflectivity: float    # Cr = 1 + srp_reflectivity
    altitude_km: float
    gravity_degree: int
    drag_kind: str             # one of FIT_KINDS
    srp_kind: str              # one of FIT_KINDS
    provenance: str            # citation + retrieval date

    @property
    def beta_kg_m2(self) -> float:
        """Ballistic coefficient m / (Cd * A), kg/m²."""
        return self.mass_kg / (self.cd * self.drag_area_m2)

    def legacy_dict(self) -> dict:
        """The properties as a plain dict with short keys ('mass', 'drag_area',
        'cd', 'srp_area', 'srp_reflectivity', 'altitude_km', 'gravity_degree'),
        the form the validate_* scripts read."""
        return {
            'mass': self.mass_kg, 'drag_area': self.drag_area_m2,
            'cd': self.cd, 'srp_area': self.srp_area_m2,
            'srp_reflectivity': self.srp_reflectivity,
            'altitude_km': self.altitude_km,
            'gravity_degree': self.gravity_degree,
        }


# How a value was obtained (drag_kind / srp_kind); see the module docstring.
FIT_KINDS: tuple[str, ...] = ("physical", "fitted", "nominal", "legacy")

PROPERTIES: dict[str, SatelliteProperties] = {
    row["name"]: SatelliteProperties(**row) for row in _core._satellite_table()
}

# Radiation-area scale factor k fitted per Sentinel on the odd days of 2025
# (SENTINEL-YEAR-2025, scored on the even days): name -> (base area m^2, k), with
# srp_area_m2 = base area x k.
SRP_FIT: dict[str, tuple[float, float]] = dict(_core._srp_fit_table())


def get(name: str) -> SatelliteProperties:
    """The SatelliteProperties for ``name`` (e.g. "S1A", "CHAMP").

    Raises KeyError naming the known satellites for any other name.
    """
    try:
        return PROPERTIES[name]
    except KeyError:
        raise KeyError(f"unknown satellite {name!r} (known: {', '.join(PROPERTIES)}); "
                       "give mass and areas explicitly for other satellites") from None


# ---------------------------------------------------------------------------
# GNSS blocks (include/ballistic/gnss_blocks.hpp)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GnssBlock:
    """Cannonball radiation-pressure inputs for one GNSS block (spacecraft type).

    GNSS satellites are looked up by block, not by name: the IGS satellite
    metadata file gives each satellite's block and in-orbit mass
    (scripts/_gnss_metadata.py), and this table gives the block's area.

    Attributes:
        name: IGS block name, or a prefix of several ("GPS-IIR" covers
            GPS-IIR-A and GPS-IIR-B).
        mass_kg_nominal: typical mass, kg, for orientation only (0 = none
            recorded); use the IGS file's per-satellite mass.
        srp_base_area_m2: solar arrays plus the body's +X face, m².
        srp_k: fitted area scale (GNSS_SRP_FIT), 1.0 when not fitted.
        srp_area_m2: effective cannonball area, srp_base_area_m2 * srp_k, m².
        cr: radiation pressure coefficient (1.3 nominal).
        srp_kind: one of FIT_KINDS; "fitted" once a k applies.
        provenance: where each number comes from (placeholders say so).
    """
    name: str
    mass_kg_nominal: float
    srp_base_area_m2: float
    srp_k: float
    srp_area_m2: float
    cr: float
    srp_kind: str
    provenance: str


GNSS_BLOCKS: dict[str, GnssBlock] = {
    row["name"]: GnssBlock(**row) for row in _core._gnss_block_table()
}

# Radiation-area scale k fitted per GNSS block (GNSS-1H-SRP): block -> k.
GNSS_SRP_FIT: dict[str, float] = dict(_core._gnss_srp_fit_table())


def gnss_block(igs_block_name: str) -> GnssBlock:
    """The GnssBlock for an IGS block name, e.g. "GPS-IIF" or "GPS-IIR-A".

    An exact name wins; otherwise the longest known name followed by '-'
    matches ("GPS-IIR-A" -> "GPS-IIR"). The rule lives in C++
    (gnss::find_block). Raises ballistic.errors.ConfigError for an unknown
    block.
    """
    return GnssBlock(**_core._gnss_find_block(igs_block_name))
