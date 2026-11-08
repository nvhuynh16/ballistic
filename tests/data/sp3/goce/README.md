# GOCE decay-phase orbits (ESOC)

Daily orbits of GOCE (SP3 ID `L15`) for its free decay after the ion
thruster ran out of fuel: 2013 days 295-314 (22 October to 10 November
2013, ~236 km down to ~142 km). Truth for `scripts/validate_goce_reentry.py`
(claim `GOCE-REENTRY` in `docs/validation_claims.md`: test days 295, 305 and
313; days 296-304 and 306-312 were used to fit GOCE's drag scale) and for
`tests/validation/test_goce_reentry_validation.cpp`.

- Files: `goce_<yyddd>.sp3`, one per day (`goce_13295.sp3` = 2013 day 295).
  Not committed (ESA data, registration required).
- Format: SP3c, header `ORBIT J2000 BHN ESOC`: positions and velocities in
  J2000 (inertial), 60 s spacing, GPS time. The harness rotates them to
  ITRF2008 with the IAU 2006 GCRS-ITRF transform and honours the time
  system.
- Producer: ESA's flight dynamics team at ESOC, published by ESA for the
  decay phase (9-17 cm RMS stated).
- Source: ESA; registration (an ESA EO Sign In account) is required. ESA
  distributes GOCE products through the GOCE Online Dissemination service,
  `https://goce-ds.eo.esa.int/`.
- Cross-check: on days 295 and 305 these orbits agree with the AIUB Precise
  Science Orbit in `tests/data/sp3/goce_pso/` to 0.16-0.24 m 3D RMS.

To reacquire: download the daily ESOC GOCE SP3 orbits for 2013 days 295-314
from ESA and save them here as `goce_<yyddd>.sp3`.
