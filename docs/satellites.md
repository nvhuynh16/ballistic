# Known satellites

`ballistic.recommended_config(state, "S1A")` configures radiation pressure
and drag from a registered satellite's properties:

| Satellite | Mass (kg) | Drag area (m²) | Cd | β (kg/m²) | Radiation area (m²) | Cr | Fit (drag / radiation) |
|-----------|-----------|----------------|-----|-----------|---------------------|-----|----|
| S1A / S1C | 2200 | 6.35 | 2.2 | 157.5 | 38.4 / 38.6 | 1.0 | nominal / fitted |
| S2B / S2C | 1200 | 4.2 | 2.2 | 129.9 | 16.6 / 16.2 | 1.0 | legacy / fitted |
| S3A / S3B | 1250 | 4.8 | 2.2 | 118.4 | 22.7 | 1.0 | legacy / fitted |
| S6A | 1192 | 3.5 | 2.2 | 154.8 | 11.4 | 1.0 | nominal / fitted |
| CHAMP | 522 | 0.9 | 2.93 (fitted) | 197.9 | 0.5 | 1.3 | fitted / legacy |
| GRACE-FO 1/2 | 595.9 / 596.1 | 1.0 | 3.56 | 167.4 | 3.5 | 1.3 | physical / legacy |
| GOCE | 1002.2 | 1.1 | 4.1 (fitted) | 222.5 | 2.0 | 1.3 | fitted / legacy |

These values live in one C++ table, `include/ballistic/satellites.hpp`
(Python: `ballistic.satellite_properties`), which cites a source for every
entry; [`satellite_properties_provenance.md`](satellite_properties_provenance.md)
gives the full record. Drag areas are mean frontal areas, the area facing
the oncoming air, not total surface area. The Sentinel radiation areas are
effective areas for a spherical ("cannonball") model: the published area
multiplied by a factor fitted to 2025 data, 1.5-1.9 for most satellites and
1.03 for S6A, whose published panel model was already close.

Each entry also says how its values were obtained, separately for drag
(`drag_kind`: mass, drag area, Cd) and radiation pressure (`srp_kind`:
radiation area, Cr): `physical` (mission documents or a physical model),
`fitted` (adjusted to orbit data, so it also absorbs force-model error and
may not carry over to other dates), `nominal` (a textbook value such as
Cd 2.2, not checked for this satellite) or `legacy` (unsourced, low
confidence). `recommended_config` issues a `ConfigHintWarning` when it uses
a legacy value, naming the arguments that replace it.

The registry also holds Swarm A, B and C, LAGEOS-1 and -2 and Etalon-1
and -2 (the laser-ranging spheres, used by `SLR-1H`); their values and
sources are in the provenance record.

For a satellite not in the table, give `mass_kg=`, `srp_area_m2=` and
`drag_area_m2=`; [`configuration.md`](configuration.md#working-the-numbers-out-from-a-shape)
shows how to derive them from a shape.

## Drag depends on a single combination of these numbers

The drag acceleration is proportional to 1/β, where β = m / (Cd A) is the
ballistic coefficient. Orbit data can measure β but cannot separate mass,
area and drag coefficient. This leads to two kinds of entry in the table.

### Physical values: GRACE-FO

For **GRACE-FO** all three are physical. The mass is the launch mass minus
the gas used since launch, from the mission's public mass records. The
frontal area (front panel plus antenna boom, 1.0 m²) comes from the panel
model in the mission's data handbook, and the drag coefficient is computed
from that same panel model with Sentman's gas-surface model, averaged over
the atmospheric composition along the orbit. Cd is well above the textbook
value of 2.2 because GRACE-FO carries 15 m² of side and bottom panels that
lie almost parallel to the airflow and still produce drag. The calculation
depends on one assumed constant, the energy accommodation coefficient; it
uses 0.93, the value adopted by a published GRACE density product, and
plausible alternatives (0.85-1.0) change Cd by ±4%. Published GRACE values
span 3.5-4.0.

On the three January 2026 test days this gives 0.87, 0.48 and 0.47 m
(`GRACEFO-500KM`). On 2 and 3 January the physical values are at the best
drag scale (within 10%); on 1 January the orbits want about 25% more drag.
ESA's Swarm satellites, whose densities are derived independently from
their own precise orbits, show the same: at GRACE-FO's altitude Swarm B
measured 1.15, 1.05 and 1.00 times the NRLMSIS density on those three days,
so on 1 January the atmosphere really was denser than the model predicted.
These figures depend on a space-weather file measured well past the test
days: a file downloaded shortly after them builds its 81-day solar-flux mean
for these days partly from forecasts (141 instead of 154), which makes the
model's density too low.

### Fitted values: CHAMP and GOCE

For **CHAMP and GOCE** there is no reliable panel model, so β was fitted to
orbit data and is written as a Cd with the listed mass and area held fixed.
These Cd values are therefore not physical drag coefficients: any error in
the mass or area ends up in them. The density model is not the cause of the
fitted factors, because NRLMSIS agrees with GOCE's measured density (median
ratio 1.00 at 156-253 km in October 2013, `GOCE-DENSITY`), and CHAMP needs
the same factor at 370 km in 2005 as at 260 km in 2010. That agreement is
not universal: at GRACE-FO's altitude in January 2026, Swarm B's measured
density was up to 1.15 times NRLMSIS, as described above, so a β fitted on
one period can absorb that period's density error.

- **GOCE** was fitted on 16 days and scored on three other days interleaved
  with them (days 295, 305 and 313 of 2013). Its mass, 1002.2 kg, is ESA's
  own record for those days; the xenon propellant ran out on 21 October 2013.
- **CHAMP** was fitted on 1 July 2005. The second 2005 orbit file (1.2 m)
  covers the adjacent half-day and overlaps the fitting file by two hours,
  so it checks a different file rather than different conditions. The
  independent check is the satellite's last weeks in 2010, at 250-280 km
  near solar minimum, where the same β is best. Those data were examined
  before the value was settled, so they confirm it rather than test it
  blind. CHAMP's 522 kg is its launch mass; no mass history was found.
