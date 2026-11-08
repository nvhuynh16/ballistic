# BeiDou GEO test data

Wuhan University (WHU) final MGEX orbits for 2024-01-15 to 2024-01-18
(GPS week 2297, days 015-018), used by `scripts/validate_gnss.py` for the
geostationary row of `GNSS-1H`.

- Files: `WUM0MGXFIN_2024{015,016,017,018}0000_01D_05M_ORB.SP3.gz`, verbatim.
- Content: GPS, GLONASS, Galileo, BeiDou and QZSS; the harness uses only the
  BeiDou GEOs (C01-C05). CODE's MGEX product, used for QZSS, omits all
  geostationary satellites, and none of these products includes QZSS's GEO
  (QZS-3, SP3 J07).
- Format: SP3-d, 5-minute positions, no velocities, GPS time, IGS20 frame.
- Source: IGS MGEX products, mirrored by WHU at
  `ftp://igs.gnsswhu.cn/pub/gnss/products/mgex/2297/` (anonymous). The same
  files are on CDDIS (`https://cddis.nasa.gov/archive/gnss/products/mgex/2297/`,
  Earthdata login required).
- Downloaded 2026-10-31.

To reacquire:

```bash
for d in 015 016 017 018; do
  curl -O ftp://igs.gnsswhu.cn/pub/gnss/products/mgex/2297/WUM0MGXFIN_2024${d}0000_01D_05M_ORB.SP3.gz
done
```

Each daily file is an independent orbit solution, so positions jump slightly
where two days meet. The harness derives velocities and picks prediction
windows within one file for that reason.
