# Swarm SP3 Precise Orbit Data

ESA Swarm constellation (3 satellites) for drag validation at 460-530 km altitude.

## Why Swarm for Drag Validation

Swarm flies at 460 km (A/C) and 530 km (B), where atmospheric drag is ~100x
stronger than at Sentinel's 700 km. Our Sentinel drag test showed only +0.5%
RMSE change because drag is negligible at 700+ km. Swarm provides a meaningful
test of the NRLMSIS 2.1 atmospheric drag model.

## Data Acquisition

Swarm SP3 data requires **free registration** at ESA's Swarm dissemination server.
This is a one-time manual step (registration blocks automation).

### Step-by-Step Download

1. **Register** at https://swarm-diss.eo.esa.int/ (free ESA account)

2. **Navigate** to:
   `Level2longterm/Entire_mission_data/SP3xCOM/Sat_A/` (or Sat_B, Sat_C)

3. **Download** SP3xCOM_2_ product files (precise center-of-mass orbits)
   - Files are typically daily, with 10-second position data
   - Example filename: `SW_OPER_SP3xCOM_2__20240115T000000_20240116T000000_0201.sp3`

4. **Place** downloaded `.sp3` files in this directory

### Alternative: ESA VirES

VirES for Swarm (https://vires.services/) provides a web API, but SP3 products
are most easily accessed from the dissemination server above.

### Recommended Data Selection

- Download 7-14 days of data for statistically meaningful results
- All three satellites (A, B, C) provide complementary altitude coverage
- Recent data (2024+) uses ITRF2020 reference frame

## SP3 Satellite IDs

The satellite IDs in Swarm SP3 files may vary:
- Swarm-A: `A`, `SWA`, `L61`
- Swarm-B: `B`, `SWB`, `L62`
- Swarm-C: `C`, `SWC`, `L63`

Use `--sat-id` override if the default mapping doesn't work.

## Satellite Properties

| Property | Swarm-A | Swarm-B | Swarm-C |
|----------|---------|---------|---------|
| Mass | 470 kg | 470 kg | 470 kg |
| Drag area | 1.0 m^2 | 1.0 m^2 | 1.0 m^2 |
| SRP area | 4.0 m^2 | 4.0 m^2 | 4.0 m^2 |
| Cd | 2.2 | 2.2 | 2.2 |
| Cr | 1.3 | 1.3 | 1.3 |
| Altitude | 460 km | 530 km | 460 km |

Swarm-A and Swarm-C fly in tandem (same altitude, 1.4 deg separation).
Swarm-B is at a higher orbit with different local time sampling.

## Validation Script

```bash
python scripts/validate_swarm_drag.py --sp3 tests/data/sp3/swarm/
python scripts/validate_swarm_drag.py --sp3 tests/data/sp3/swarm/ --configs baseline drag
```

### Space Weather Data

Drag validation requires space weather data (F10.7 solar flux, Ap index) for
the NRLMSIS 2.1 atmosphere model. The validation script loads this automatically
via `ballistic.Loader().space_weather()`. Ensure `SW-All.csv` from CelesTrak is
available (downloaded automatically on first run).
