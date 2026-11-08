# LAGEOS SP3 Precise Orbit Data

LAGEOS-1 and LAGEOS-2 are geodetic retroreflector spheres at ~5900 km altitude,
tracked by the International Laser Ranging Service (ILRS).

## Why LAGEOS for SRP Validation

LAGEOS satellites are **perfect spheres** (60 cm diameter), making them ideal for
validating the cannonball SRP model. Unlike box-wing spacecraft such as Sentinel
(where a cannonball area is only an effective, fitted value), LAGEOS SRP is
genuinely isotropic.
ILRS orbit products also use a cannonball model in their orbit determination.

## Download

### Automated (EDC anonymous FTP)
```bash
python scripts/download_ilrs_sp3.py --satellite lageos1
python scripts/download_ilrs_sp3.py --satellite lageos2
python scripts/download_ilrs_sp3.py --satellite lageos1 --gps-week 2294
```

### Manual Download
1. Browse: `ftp://edc.dgfi.tum.de/pub/slr/products/orbits/`
2. Navigate to desired GPS week directory
3. Download `.sp3.gz` files
4. Decompress and place in this directory

### Alternative Source (CDDIS, requires Earthdata login)
1. Create account: https://urs.earthdata.nasa.gov/users/new
2. Browse: https://cddis.nasa.gov/archive/slr/products/orbits/
3. Download SP3 files for desired GPS week

## SP3 Satellite IDs

The satellite IDs in ILRS SP3 files (confirmed Feb 2026):
- LAGEOS-1: `L51`
- LAGEOS-2: `L52`

Use `--analyze` to inspect a downloaded file:
```bash
python scripts/download_ilrs_sp3.py --analyze path/to/file.sp3 --satellite lageos1
```

If the default IDs don't match, use `--sat-id` override in the validation script:
```bash
python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/lageos/ --sat-id L07
```

## Satellite Properties

| Property | LAGEOS-1 | LAGEOS-2 |
|----------|----------|----------|
| Mass | 406.965 kg | 405.38 kg |

| Diameter | 0.60 m | 0.60 m |
| Cross-section | 0.2827 m^2 | 0.2827 m^2 |
| Reflectivity | 0.13 | 0.13 |
| Altitude | 5890 km | 5620 km |
| Inclination | 109.8 deg | 52.6 deg |
| ILRS ID | 7603901 | 9207002 |

## Validation Script

```bash
python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/lageos/ --satellite lageos
```
