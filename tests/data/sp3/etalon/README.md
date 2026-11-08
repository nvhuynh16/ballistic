# Etalon SP3 Precise Orbit Data

Etalon-1 and Etalon-2 are Russian geodetic retroreflector spheres at ~19,100 km
altitude (GLONASS orbit), tracked by the ILRS.

## Why Etalon for SRP Validation

Like LAGEOS, Etalon satellites are **perfect spheres** (1.294 m diameter), ideal
for cannonball SRP validation. At MEO altitude (~19,100 km), SRP is a larger
fraction of total perturbation compared to LAGEOS at ~5900 km.

## Download

### Automated (EDC anonymous FTP)
```bash
python scripts/download_ilrs_sp3.py --satellite etalon1
python scripts/download_ilrs_sp3.py --satellite etalon2
```

### Manual Download
1. Browse: `ftp://edc.dgfi.tum.de/pub/slr/products/orbits/`
2. Navigate to desired GPS week directory
3. Download SP3 files containing Etalon data
4. Place in this directory

## SP3 Satellite IDs

- Etalon-1: `L53`
- Etalon-2: `L54`

Use `--analyze` to inspect:
```bash
python scripts/download_ilrs_sp3.py --analyze path/to/file.sp3 --satellite etalon1
```

## Satellite Properties

| Property | Etalon-1 | Etalon-2 |
|----------|----------|----------|
| Mass | 1415 kg | 1415 kg |
| Diameter | 1.294 m | 1.294 m |
| Cross-section | 1.315 m^2 | 1.315 m^2 |
| Reflectivity | 0.13 | 0.13 |
| Altitude | 19,100 km | 19,100 km |
| Inclination | 64.9 deg | 65.5 deg |
| ILRS ID | 8900103 | 8903903 |

## Validation Script

```bash
python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/etalon/ --satellite etalon
```
