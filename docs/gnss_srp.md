# Radiation pressure for GNSS satellites

Solar radiation pressure is the largest force the default configuration leaves
out at GNSS altitudes (~1e-7 m/s²); `scripts/validate_gnss.py --srp cannonball`
turns it on per satellite with a deliberately simple model.

**The model.** A cannonball (`SrpConfig`: a sphere of the given cross-section,
pushed directly away from the Sun, `cr` 1.3, with the library's Earth albedo and
infrared terms and a hard cylindrical Earth shadow). Per satellite:

- the **block** (spacecraft type) and **in-orbit mass** at the start epoch come
  from the public IGS satellite metadata file (`scripts/_gnss_metadata.py`);
- the block's **base area** (solar arrays plus the body's +X face) comes from
  `include/ballistic/gnss_blocks.hpp` (Python
  `ballistic.satellite_properties.gnss_block(name)`), where IGS names that
  extend a row after `-` resolve to it (`GPS-IIR-A` → `GPS-IIR`);
- the area used is base area × the block's fitted scale k (`kGnssSrpFit`,
  `GNSS_SRP_FIT`) × `--srp-scale`.

Left out: attitude (no yaw-steering box-wing model, no Y-bias), antenna
thrust, thermal re-radiation, penumbra. It degrades first in eclipse seasons
(shadow passes and yaw maneuvers), which the committed four days do not test.

**The blocks.** `physical` = areas from published spacecraft documents;
`nominal` = placeholder or partly assumed (each row's provenance says which
number); a fitted k makes a row `fitted`. k absorbs the simplifications above,
so it belongs to this model, not to the spacecraft.

| Block | Base area (m²) | Source of the base area | k | Status |
|---|---|---|---|---|
| GPS-IIR (IIR-A/B) | 17.1 | arrays 144 ft² and body size from the U.S. Space Force fact sheet (not checked against the page) | 1.10 | fitted |
| GPS-IIR-M | 17.1 | as GPS-IIR | 1.05 | fitted |
| GPS-IIF | 27.7 | body face 5.72 m² (Dilssner et al., IGS Workshop 2018); arrays assumed 22 m² | 1.10 | fitted |
| GPS-IIIA | 30.0 | assumed | 1.00 | fitted |
| QZS-1 | 54.0 | QZS-1 Satellite Information (Cabinet Office of Japan) | — | physical |
| QZS-2I | 39.9 | QZS-2 / QZS-4 Satellite Information | 1.05 | fitted |
| QZS-2A | 40.6 | QZS-1R Satellite Information | 1.00 | fitted |
| QZS-2G | 39.9 | QZS-3 Satellite Information (reflectors left out) | — | physical |
| QZS-3G | 40.0 | assumed (as QZS-2G) | — | nominal |
| BDS-2G / -2I / -2M | 23.4 | body face 3.44 m² (Dilssner et al.); arrays assumed 20 m² | 1.25 (2G) | fitted / nominal |
| BDS-3M-CAST | 28.2 | body face and ~27 m² array estimate (Dilssner et al.) | — | nominal |
| BDS-3M-SECM | 14.7 | body face (Dilssner et al.); arrays assumed half of CAST | — | nominal |
| BDS-3I, BDS-3G | 30.0 | assumed | — | nominal |

**Results** (`GNSS-1H-SRP`, `GNSS-24H-SRP` in
[`validation_claims.md`](validation_claims.md)): at one hour, scored on days
not used for the fit, GPS 0.089 m (0.668 m without radiation pressure), QZSS
IGSO 0.154 m (0.740 m), BeiDou GEO 0.193 m (0.808 m); at 24 hours 7.8 m
(220 m), 11 m (521 m) and 47 m (593 m).

**Refreshing the metadata.** `python scripts/download_igs_metadata.py` fetches
the current IGS file to `data/igs/igs_satellite_metadata.snx` (or pass
`--metadata PATH`). A block name the table does not know stops the campaign
with `ConfigError`; add a row with its source. **Re-fitting.** `scripts/validate_gnss.py --fit-srp --fit-json fit.json` runs
the one-hour campaign on 2024 days 015 and 017 for every k from 0.50 to 1.60
(step 0.05), picks per block the k with the smallest RMSE, and scores it on
days 016 and 018 next to radiation pressure off. Copy the printed k into
`kGnssSrpFit` and rebuild; `--srp-scale` gives the sensitivity to a common
change of all areas.
