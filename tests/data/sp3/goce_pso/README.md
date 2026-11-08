# GOCE Precise Science Orbit, reentry test days


AIUB reduced-dynamic orbits from ESA's GOCE Precise Science Orbit product
(`SST_PSO_2`) for the two `GOCE-REENTRY` test days the product covers: 2013 day 295
(22 October, baseline 0002) and day 305 (1 November, baseline 0001). The
original baselines end on 5 November 2013, so test day 313 has no PSO.
Used to check the ESOC truth in `tests/data/sp3/goce/` (they agree to
0.16-0.24 m 3D RMS).

- Files: `goce_pso_<yyddd>_<baseline>.sp3.gz`, written by
  `scripts/convert_goce_pso.py` from the ESA archives (the SST_PRD_2 section
  only; values copied unchanged).
- Format: SP3c, 10 s positions (km) and velocities (dm/s), GPS time,
  Earth-fixed (IGS05, i.e. ITRF2005/2008-aligned).
- Source: ESA GOCE Online Dissemination,
  `https://goce-ds.eo.esa.int/oads/access/collection/GOCE_Level_2/tree`
  → SST_PSO_2_ → 2013 (ESA EO Sign In account required). Archives
  `GO_CONS_SST_PSO_2__20131021T235944_20131022T235943_0002.TGZ` and
  `GO_CONS_SST_PSO_2__20131031T235944_20131101T235943_0001.TGZ`.
  Data courtesy of ESA; orbits by IAPG/AIUB.
- Downloaded 2026-10-31.

To regenerate: download the two archives, then

```bash
python scripts/convert_goce_pso.py <archives> --out <dir>
gzip -9 <dir>/goce_pso_13295_0002.sp3 <dir>/goce_pso_13305_0001.sp3
```
