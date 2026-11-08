# Reentry and TLE starts

## Reentry

In orbit, the air is so thin that molecules strike a satellite without first
colliding with each other (free-molecular flow), and a constant drag
coefficient of about 2-2.5 describes it well. A reentering body descends into
dense air, where a shock wave stands ahead of it and the drag coefficient
depends on the Mach number: a sphere's falls from about 2.1 to about 0.9.
For a reentry, give the body's shape and size and let the coefficient follow
the flow:

```python
state = ballistic.State.from_itrf(epoch, r_m, v_m_s, allow_suborbital=True)
cfg = ballistic.recommended_config(state, mass_kg=500.0, drag_area_m2=0.785,
                                   reentry=ballistic.aero.AeroShape.SPHERE, reference_length_m=1.0)
traj = ballistic.Trajectory(state, cfg)
result = traj.propagate_until(86400.0)   # stops at the ground impact
```

The coefficient is the free-molecular value (Sentman's formulas) above a
Knudsen number of 10, the continuum value below 0.001 (for a sphere,
Henderson's 1976 correlation; for a flat plate or a cylinder, modified
Newtonian theory, which needs a table of measured values below Mach 3), and a
standard blend between. It agrees with Henderson's full sphere correlation to
8% RMS (`FR-SPHERE`). It changes a reentry mainly in its last orbit: for
GOCE's final nine days, modeled as an equivalent sphere, the predicted time
of reaching 80 km moved 29 minutes later, 26 of them accrued between 120 and
80 km (`FR-GOCE-LIFETIME`). Over a decay of days, the uncertainty of the
density (about 10%) matters far more. Heating, ablation and breakup are not
modeled.

`propagate_until` locates events inside the integration step: altitude
crossings, ground impact (armed by default), perigee, apogee and a target
epoch (`ballistic.events`). For deep reentry on a tabulated density,
`extend_density_below` adds the U.S. Standard Atmosphere 1976 below about
80 km; the default live atmosphere already runs to the ground.

## Starting from a TLE

A two-line element set (TLE) holds *mean* elements of the SGP4 theory, in
SGP4's own frame (TEME). Read as ordinary orbital elements they put the
start kilometers off, so only SGP4 reads them:

```python
state = ballistic.State.from_tle(line1, line2)        # SGP4 at the TLE epoch, converted to the ITRF
cfg = ballistic.recommended_config(state, accuracy="tle", horizon_s=6 * 3600, beta_kg_m2=100.0)
traj = ballistic.Trajectory(state, cfg)
traj.propagate_for(6 * 3600)
```

The SGP4 here is written from the published theory (Spacetrack Report #3
with the corrections of Vallado et al. 2006) and reproduces the official
verification output to 0.00003 mm (`SGP4-VER`). `ballistic.tle` parses TLE
files strictly (checksums, columns, Alpha-5 numbers) and runs SGP4 directly.

A TLE is uncertain by about 0.5 km in low orbit (Flohrer et al. 2009), so
most of the force model buys nothing. `accuracy="tle"` keeps only the terms
whose measured effect over your horizon exceeds a tenth of that: in low orbit
over an hour, gravity to degree 8-16 and drag below 500 km; at GPS and
geostationary altitude, a low degree and the Sun and Moon. The choice and
its measured cost are in `ballistic.tle.tle_grade_preset(state, horizon_s)`;
dropping drag makes a run 20-30 times faster (`TLE-PRESET`). Where the
preset keeps drag it needs a ballistic coefficient (`beta_kg_m2`, the
satellite's mass and area, or `tle=` to take it from B*).

The SGP4 state at the TLE epoch is itself imperfect: its orbit size is tens
of meters off, which makes a numerical propagation drift by kilometers per
day. Measured on GRACE-FO, 3.6-7 days from the epoch, SGP4 was 1-3 km off, a
numerical propagation from its epoch state 12-24 km, and one from a state
fitted to SGP4 over a day (`ballistic.tle.fit_state(tle, cfg)`) 0.3-1.8 km
(`TLE-GRACEFO`). Those horizons lie *before* the TLE epoch, inside the days
the TLE was fitted to; predictions after the epoch have not been measured
yet (`scripts/validate_tle.py` keeps the downloaded sets for that).

Forward horizons need an archive, because CelesTrak serves only the current
set and the rapid orbits arrive about three days late. A daily
`scripts/validate_tle.py --fetch-only` stores each new element set (one
already archived is skipped) and the recent rapid orbits under
`data/tle_validation/`, appends one line per run to its `fetch.log`, and
exits non-zero only when the CelesTrak download fails; for example, from
cron (with `/path/to/ballistic` the repository):

    0 3 * * * cd /path/to/ballistic && PYTHONPATH=./python uv run --isolated --no-project --python 3.12 --with numpy --with sgp4 python scripts/validate_tle.py --fetch-only --days 7 >> data/tle_validation/cron.log 2>&1

`--days 7` reaches back past the rapid-orbit delay and over a few missed
runs. Once the rapid orbits cover the days after an archived set,
`--tle <that set> --sp3 data/tle_validation/rso/*.sp3.gz --forward-only`
scores only the samples after its epoch; horizons are signed (negative
before the epoch, positive after) in the table and the JSON. The archive is
never committed: CelesTrak's sets derive from Space-Track data whose
redistribution terms are not clear (`data/` is git-ignored).
