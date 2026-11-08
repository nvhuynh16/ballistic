# Satellite Properties Provenance

Single machine-readable source: `include/ballistic/satellites.hpp` (C++; bound to
Python as `ballistic.satellite_properties`; every entry carries its provenance
string; `test_satellite_properties.py` and the static rule SAT-LITERAL gate
consistency and forbid copies). This document is the human-readable citation
record; the summary for users is [`satellites.md`](satellites.md).

**How each value was obtained** is machine-readable too: every entry
carries `drag_kind` (mass, frontal area, Cd) and `srp_kind` (radiation area,
Cr), each one of `physical` (mission documents or a physical model),
`fitted` (adjusted to orbit data, so it also absorbs force-model error),
`nominal` (a textbook value such as Cd 2.2 or Cr 1.13, not checked for this
satellite) or `legacy` (unsourced, low confidence). The "Fit (drag / SRP)"
column below shows them; the prose Confidence column says the same in more
detail. `recommended_config` warns (`ConfigHintWarning`) when it uses a
legacy value. The radiation areas of CHAMP, GRACE-FO and GOCE carry no
citation, hence `legacy`; radiation pressure is small next to drag there.

**The rule this file enforces**: `drag_area` is the mean ram-facing
**FRONTAL** cross-section, never the total/panel area. Using a total-area
class value for CHAMP (3.5 m² instead of ~0.9 m² frontal) makes β about 4×
too small, and a compensating density bias can hide it.

| Satellite | Mass (kg) | Frontal area (m²) | Cd | β (kg/m²) | Source | Confidence | Fit (drag / SRP) |
|---|---|---|---|---|---|---|---|
| CHAMP | 522 | **0.9** | **2.93** (fitted) | **197.9** | eoPortal CHAMP mission page (trapezoidal body, ~0.9 m² cross-section; mass 522), retrieved 2026-07-19. Cd·A FITTED under the default space weather: β ×0.75 of the nominal Cd 2.2, best on 2005 day 182 (×0.75–0.8 within 2 cm; `CHAMP-370KM`). Checked on the RSO file 2005-06-30/07-01 — the adjacent half-day, overlapping the fit by 2 h, so NOT an independent test — and on the 2010 final weeks at 250–280 km (same factor best; independent in altitude and solar activity, but that set was examined before the value was settled, so not blind; `CHAMP-260KM`). 522 kg is the launch mass; the 2005 mass was lower by the gas used (no mass history found). NRLMSIS ≈ measured GOCE density on the same weather (0.97–1.09), so the factor is attributed to Cd·A. **Physical cross-check**: the DSMC geometry model of March et al. (2019, Fig. 14; accommodation 1) gives Cd·A 2.44 m² along the fit day's orbit (β 214 at 522 kg, +8%); at accommodation 0.85 (TU Delft's CHAMP value, March et al. 2021) β 195 (−1.4%), 2010 days +3.6%. These come from interpolating the published DSMC Cd·A against the flow direction along the propagated orbit, at the satellite's mass. The fitted product is physical within the uncertainties. | HIGH for the product m/(Cd·A) (same factor at two altitudes and solar activities); Cd itself not physical | fitted / legacy |
| GRACE-FO 1/2 | **595.9 / 596.1** | 1.0 | **3.56** (Sentman) | **167.4** | PHYSICAL, not fitted to orbits. Mass = launch 601.21 kg ([GRACE-FO L1 Data Product User Handbook](https://isdc-data.gfz.de/grace-fo/DOCUMENTS/Level-1/GRACE-FO_L1_Data_Product_User_Handbook_20190911.pdf), JPL D-56935, Table 4) minus tank gas used since 2018-05-23 (31.26 → 25.94 kg / 31.29 → 26.17 kg; [Level-1B MAS1B](https://isdc-data.gfz.de/grace-fo/Level-1B/JPL/INSTRUMENT/RL04/), public). Area = front panel 0.955 + boom 0.046 m² (handbook Table 5). Cd = Sentman's flat-plate formula on all nine Table-5 panels, mass-weighted over the NRLMSIS 2.1 species along the 2026-01-01..03 orbits (T ≈ 1100 K, 15.6 amu), accommodation 0.93 (the GRACE value of the TND-IGG RL01 density product, chosen before scoring), wall 300 K. The side, nadir and zenith panels (15 m², nearly parallel to the flow) are why Cd is far above the flat-plate 2.2. Accommodation 0.85–1.0 → Cd 3.71–3.33 (±4%); GRACE literature values 3.5–4.0 (Mehta 2014, Sutton 2008, Bowman 2007, via ADBSat Fig. 14). Scored with true start velocities: 0.87/0.48/0.47 m (`GRACEFO-500KM`); on 01-02/03 the physical values are at the best drag scale (within 10%), on 01-01 the orbits want about 25% more drag. ESA Swarm B POD densities at the same altitude, compared point by point with NRLMSIS along Swarm's orbit, are 1.15/1.05/1.00× NRLMSIS on those days, so 01-01's extra drag is atmospheric. | HIGH for m and A (mission documents); MEDIUM for Cd (model + assumed accommodation) | physical / legacy |
| GOCE | **1002.2** (ESA mass file) | 1.1 | **4.095** (β fit at the ESA mass) | **222.5** | ESA (launch mass 1077 kg; "slim 5 m satellite, cross-section ~1 m²"), eoPortal GOCE, retrieved 2026-07-19. Cd·A FITTED (drag ×2.0 of Cd 2.2 × 1.1 m² at 1077 kg, chosen on 16 free-decay days, scored out of sample on 2013 days 295, 305 and 313: `GOCE-REENTRY`); the A/Cd split is a convention, as for CHAMP. MASS: ESA GOCE Mass Properties file ([earth.esa.int](https://earth.esa.int/eogateway/missions/data/goce-mass-properties), public) — 1050.8 kg at launch, 1002.2 kg from 2013-10-21 (xenon exhausted) through every test and calibration day; β 222.5 kept, so Cd·A = 4.50 m². **Physical cross-check**: the March et al. (2019) DSMC geometry gives Cd·A 3.73 m² on the calibration days at accommodation 1 (β 269, +21%). The density product the fit agrees with (TU Delft v2.0) was itself derived with that geometry at accommodation 0.82 (ESA GOCE+ user manual v2.0, Sect. 1.3); at 0.82 the DSMC Cd·A is 4.12 m², β 243, **+9.3% → within the 10% band** (test days +12%). | HIGH above ~180 km (out of sample); below that attitude limits dominate | fitted / legacy |
| Sentinel-1 A/C | 2200 | **6.35** | 2.2 | 157.5 | **GMV-GMESPOD-TN-0025 v1.4 (2019), Table 5-1** — the official Copernicus POD Service properties document: along-track frontal 5.57 m² (0° yaw), 6.12/8.15 (±3.9° yaw), **6.35 orbit-averaged**. Mass ≈ 2200 mid-life (POD uses mass-history files); Peter et al. (2017, Adv. Space Res. 60:879, Sect. 2) give 2158.777 kg for S1A in 2014–2015, so 2200 is ~2% high for then and the 2025 mass is lower still (kept: the SRP area was fitted as A/m at 2200 kg, and no 2025 mass is sourced). An earlier unsourced 3.9 m² made β 63% too large. | HIGH (official POD document) | nominal / fitted |
| Sentinel-2 B/C | 1200 | 4.2 | 2.2 | 129.9 | Unsourced frontal area. The orbits prefer about 2-2.5× the registry drag all year (`SENTINEL-YEAR-2025` in [`validation_claims.md`](validation_claims.md)); not adopted pending a sourced macro-model. | LOW | legacy / fitted |
| Sentinel-3 A/B | 1250 | 4.8 | 2.2 | 118.4 | Unsourced frontal area. The orbits prefer about 2-2.5× the registry drag all year (`SENTINEL-YEAR-2025` in [`validation_claims.md`](validation_claims.md)); not adopted pending a sourced macro-model. | LOW | legacy / fitted |
| Sentinel-6 A | 1192 | 3.5 | 2.2 | 154.8 | Launch mass (JPL press kit / ESA, incl. 230 kg hydrazine; current mass unknown). Frontal = body +X face; SRP cannonball 11.1 m² = total macro-model area 44.3 m² / 4 — DLR macro-model, Calliess et al. 2024, *Adv. Space Res.* 74:3011, Table 4 (retrieved 2026-10-30) — × fitted 1.03 = 11.4 m². Drag negligible at 1336 km. | MEDIUM for SRP area (published macro-model, confirmed by the fit to 3%); LOW for mass | nominal / fitted |
| Swarm A/B/C | 470 | 1.0 | 2.2 | 213.6 | Unsourced values; literature frontal ~0.9–1.1 m² class. Not validated in this repository (the SP3 orbits require registration). | LOW | legacy / legacy |
| LAGEOS-1 | **406.965** | 0.2827 | 2.2 | 654.4 | NASA Earthdata LAGEOS-1 page and ILRS mission page (406.965 kg, 60 cm sphere), retrieved 2026-11-02. Areas = π(0.30 m)²; Cr 1.13 conventional (unsourced). Drag negligible at 5,900 km. | HIGH for mass and size | physical / nominal |
| LAGEOS-2 | 405.38 | 0.2827 | 2.2 | 651.8 | ILRS LAGEOS-2 mission page (405.38 kg, 60 cm), retrieved 2026-11-02. | HIGH for mass and size | physical / nominal |
| Etalon-1/-2 | 1415 | 1.315 | 2.2 | 489.1 | ILRS Etalon-1/-2 mission page (1415 kg, 1.294 m diameter), retrieved 2026-11-02. Cr 1.13 conventional (unsourced). | HIGH for mass and size | physical / nominal |

**What is actually validated.** Drag depends only on β = m/(Cd·A), and the
drag campaigns measure only β. They cannot separate mass, area and Cd. The
fitted rows (CHAMP, GOCE) fit β by one constant factor — the
lowest 1-hour RMSE on a coarse grid — and write the result as a Cd with the
tabulated mass and area held fixed. So those **Cd values are not physical
drag coefficients**: any error in the mass (launch masses, fuel used since)
or the area (round published cross-sections, an average over attitude) lands
in them. For CHAMP, 0.74 m² × Cd 3.56 at 522 kg fits exactly as well as the
tabulated 0.9 m² × Cd 2.93. For drag prediction β is the whole story, and
the registry test gates β rather than its factors; anyone reusing the *area*
or *mass* for something else (radiation pressure, attitude) is using
unvalidated numbers. Physical Cd values need real masses and panel models;
GRACE-FO has them, so its row is physical, not fitted.

The LAGEOS/Etalon radiation areas and Cr are conventional values, validated
empirically by the `SLR-1H` campaign; do not conflate radiation areas with
the drag-area rule above.

Sources: [eoPortal CHAMP](https://www.eoportal.org/satellite-missions/champ) ·
[eoPortal GRACE-FO](https://www.eoportal.org/satellite-missions/grace-fo) ·
[eoPortal GOCE](https://www.eoportal.org/satellite-missions/goce) ·
[GMV-GMESPOD-TN-0025 (Sentinel-1 properties for GPS POD)](https://sentinels.copernicus.eu/documents/247904/3455957/Sentinel-1-properties-for-GPS-POD) ·
Doornbos-method density literature (NRLMSIS-class models over-predict ~30%
at CHAMP altitude): [SWSC 2023 CHAMP/GRACE/GRACE-FO datasets](https://www.swsc-journal.org/articles/swsc/full_html/2023/01/swsc230004/swsc230004.html)

## Radiation (SRP) areas — fitted effective values

`srp_area_m2` is the area the CANNONBALL radiation model needs (SRP +
Earth albedo + IR, at the registered Cr), not a physical panel area. For
the Sentinels it is base area × a scale k fitted on all of 2025
(`scripts/validate_sentinel_year.py srp-fit`; the 1-hour error is linear in
k to ≤ 1.2 mm, so one radiation-off run fixes k exactly; fitted on odd days,
scored on even days; [`validation_claims.md`](validation_claims.md)
`SENTINEL-YEAR-2025`):

| Satellite | Base area (m²) | k | Effective area (m²) | Held-out 60-min RMSE k=1 → k |
|---|---|---|---|---|
| S1A / S1C | 20.0 (unsourced) | 1.92 / 1.93 | 38.4 / 38.6 | 0.391 → 0.362 m (S1A) |
| S2B / S2C | 11.0 (unsourced) | 1.51 / 1.47 | 16.6 / 16.2 | 0.361 → 0.317 m (S2B) |
| S3A / S3B | 14.0 (unsourced) | 1.62 | 22.7 | 0.465 → 0.392 m (S3A) |
| S6A | 11.1 (macro-model) | 1.03 | 11.4 | 0.323 → 0.323 m |

k is stable month to month (S2/S3 within 1.3–1.7), so it is a level error
in the base areas/Cr, not seasonal chasing; S1's dawn-dusk orbit keeps its
Sun-pointing arrays lit (largest k). A refit with the full force model
(GCRS-ITRF velocity transform including the CIP drift, ocean and Step-2
solid tides) gives k* = 0.99–1.03 for all seven, so the areas above stand.
