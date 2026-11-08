# GRACE-FO precise orbits (GFZ RSO)

**Committed here**: `GFZOP_RSO_L64_G_20260101_100000_20260102_000000_v03.sp3`
(256 KB) — one day of GFZ Rapid Science Orbit, used as real-dynamics truth by
gate T5 (`tests/python/test_orbital_fixture.py`: the default force model
reproduces it to 0.157 m over 600 s).

Source: GFZ Potsdam ISDC, GRACE-FO RSO v03 products (`GFZOP_RSO_L64_G_*`,
L64 = GRACE-FO-1, L65 = GRACE-FO-2). Additional days used by the GRACE-FO
drag campaign are NOT committed — fetch them from the ISDC to reproduce the
campaign rows in `docs/validation_claims.md`.

ITRF realization: ITRF2020. The realizations used for the committed orbit
products:

| Data source | Period | ITRF |
|-------------|--------|------|
| ESA Sentinel | before 2025-03-04 | ITRF2014 |
| ESA Sentinel | from 2025-03-04 | ITRF2020 |
| GFZ GRACE-FO RSO v03 | 2025+ | ITRF2020 |
| GFZ CHAMP | 2000-2010 | ITRF2005 |
