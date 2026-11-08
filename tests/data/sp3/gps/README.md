# GPS test data

CODE (Center for Orbit Determination in Europe, AIUB, University of Bern)
operational final orbits for 2024-01-15 to 2024-01-18 (GPS week 2297, days
015-018), used by `scripts/validate_gnss.py` for the GPS row of `GNSS-1H`
and, joined across days, for predictions longer than one file.

Committed files:

| File | Day |
|------|-----|
| `COD0OPSFIN_20240150000_01D_05M_ORB.SP3` | 2024-01-15 (uncompressed) |
| `COD0OPSFIN_20240160000_01D_05M_ORB.SP3.gz` | 2024-01-16 |
| `COD0OPSFIN_20240170000_01D_05M_ORB.SP3.gz` | 2024-01-17 |
| `COD0OPSFIN_20240180000_01D_05M_ORB.SP3.gz` | 2024-01-18 |

- Content: GPS and GLONASS (79 satellites), 5-minute positions and clocks,
  no velocities; 289 epochs from 00:00 to 24:00 inclusive, so consecutive
  days share their midnight epoch.
- Format: SP3-d, GPS time, IGS20 frame (the IGS realisation of ITRF2020).
- Each daily file is an independent orbit solution; at the shared midnight
  epoch the two days' GPS positions differ by 7 mm (median; 5 cm largest).
- SHA-256 of the three compressed files, as served:
  - 016: `5859655a6f5098b443ba76a60304ba9bcb1cb08060fb26283a0da817750d55fa`
  - 017: `8ff160d44b59306872263fe68e0959dfd136eada0a36243400e9182ca242eba7`
  - 018: `da6a996c7461e606d3929d22a32ec5c2873f98f24e5689571ce2007a63a52393`

Other files that may sit in this directory (ultra-rapid, ESA, JPL and IGS
products) are not committed and not used by the tests.

## Source and attribution

CODE products are published openly by AIUB at
`https://www.aiub.unibe.ch/download/CODE/` (anonymous; the server redirects
to its object store) and through the IGS data centres. When using them,
acknowledge CODE / AIUB as the analysis centre and cite the product series
("CODE final product series for the IGS", Dach et al., Astronomical
Institute, University of Bern; AIUB's product page gives the DOI for each
year).

## Reacquire

```bash
for d in 015 016 017 018; do
  curl -L -O https://www.aiub.unibe.ch/download/CODE/2024/COD0OPSFIN_2024${d}0000_01D_05M_ORB.SP3.gz
done
gunzip COD0OPSFIN_20240150000_01D_05M_ORB.SP3.gz   # day 015 is kept uncompressed
```

The same files are on CDDIS
(`https://cddis.nasa.gov/archive/gnss/products/2297/`, Earthdata login).
