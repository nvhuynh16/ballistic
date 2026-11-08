# Pitfalls

Most of the mistakes below share one property: they do not fail, they
produce a slightly different orbit, and the resulting error looks like a
force-model problem.

## Common traps and how the library stops you

Every known way to get a plausible but wrong answer from this library is
listed in [`trap_census.md`](trap_census.md), each with the guard that keeps
it from coming back. The ones a user meets most:

| Trap | What happens instead |
|------|----------------------|
| Inertial velocity passed as Earth-fixed (470 m/s at LEO) | `State.from_itrf` raises `FrameError` when the swapped reading is the plausible orbit |
| Positions in km, or a point below the ground | `State` raises `InputPlausibilityError` |
| "ECI" meaning true-of-date instead of GCRS (22 km at LEO) | The ambiguous names are gone; `gcrs_to_itrf` / `itrf_to_gcrs` name their frame |
| No Earth orientation data (meters at LEO) | A `Trajectory` raises unless you opt into zeros; past the table it holds the edge value for 30 days, with a warning, then stops |
| Drag on assumed quiet space weather (density 2-20× off) | Building the drag model raises without measured weather; forecasts are used only with a warning |
| A requested model silently replaced (gravity degree, ocean tides, a parser fallback) | `ConfigError`, or a warning through `ballistic.errors`; `ballistic.set_strict(True)` turns warnings into errors |
| TLE elements read as osculating elements (kilometers) | `State.from_tle` runs SGP4 |
| Times on a floating Julian date (tens of µs) | `Epoch` stores TT exactly; constructors require a named time scale |

## Setup

- **Earth orientation data.** Without `ballistic.load()`, a propagation
  raises; accepting zeros on purpose puts low-orbit frames 3-10 m off.
- **Radiation pressure and drag.** A bare `TrajectoryConfig` has both off,
  because both need the satellite; `recommended_config` switches them on. A
  start state taken from a precise orbit does not "already include" these
  forces, since it fixes only the starting point.
- **Tabulated drag environments.** The tabulated generators
  (`from_accelerometer_density`, `from_models_along`, `from_models_4d`)
  cover a fixed time span. Beyond it the atmosphere is frozen, and the
  trajectory warns once; over a multi-day arc a table built for one hour
  misses a large part of the drag effect. `make_drag_environment` has no
  such limit.
- **Drag area.** Drag areas are mean frontal areas, not total surface areas.
  Mixing the two puts β off by a large factor: a factor of about 4 for CHAMP
  and 63% for Sentinel-1.

## Reference frames and velocities

A state vector in the wrong frame propagates a different orbit without any
error being raised.

- **J2000 read as "ECI".** A conversion whose "ECI" frame is the true
  equator of date leaves out the precession since 2000; for ESA's J2000
  GOCE orbits of 2013 the states come out 21.7 km and 14 m/s off, and the
  one-hour GOCE error above 220 km was 181 m instead of 82.5 m with the
  right transform (before the drag coefficient was fitted; `GOCE-REENTRY`). Use `gcrs_to_itrf` and `itrf_to_gcrs` for J2000 and GCRS
  states, which differ from each other by only about 0.7 m in low orbit.
- **An Earth-fixed frame read as inertial.** The frame is named on the
  first line of an SP3 header. `IGb08`, for example, is an Earth-fixed
  realization, not an inertial frame; `ballistic.sp3.read` classifies the
  label and raises on one it does not know.
- **Correcting velocities that are already right.** The slow drift of the
  Earth's axis (the CIP drift) adds 0.6-4.6e-5 m/s to a low-orbit velocity.
  GFZ's CHAMP and GRACE-FO velocity records already include it (they match
  the derivative of GFZ's own positions to 1e-6 m/s); subtracting it again
  adds about half a meter of along-track error per hour at GRACE-FO. Run
  `check_velocity_records` on a product before choosing how to use its
  velocities (see [`configuration.md`](configuration.md#settings-that-depend-on-your-data)).
- **The CIP drift in a frame transform.** A GCRS-ITRF velocity transform
  that leaves out the CIP drift produces a ±0.3 m seasonal along-track
  swing over a year of Sentinel predictions; the library's transform
  includes it.
- **Orbital elements from an Earth-fixed velocity.** An inclination
  computed from the Earth-fixed velocity lacks the ω × r term (about
  3 km/s at GEO) and misclassifies orbits.
- **The wrong ITRF realization.** Match it to the product, as described in
  [`configuration.md`](configuration.md#settings-that-depend-on-your-data).

`Trajectory` takes position and velocity in ECEF (ITRF) through
`State.from_itrf`; inertial states go through `State.from_gcrs` or
`State.from_j2000`, or are converted with `gcrs_to_itrf`.

## Reference orbit files

- **Time systems.** SP3 files state their time system on the `%c` line, and
  GPS time runs 18 s ahead of UTC.
- **File boundaries.** Each daily SP3 file is a separate solution. Do not
  differentiate or interpolate across midnight between two files; doing so
  produced 5-26 m of false error for BeiDou GEO satellites.
- **Time differences.** When differentiating sampled positions yourself,
  take time differences from the file's time stamps, not from differences
  of `Epoch` Julian dates (`tt_jd()`, `jd(scale)`), which are doubles and
  lose microseconds.
- **File edges.** A derivative taken at the first or last sample of a file
  has to use a one-sided window, which amplifies the millimeter rounding of
  SP3 positions to about 5e-4 m/s. Do not start a prediction there from a
  derived velocity.

## Empirical acceleration

- A small repropagation error does not mean the EA is correct point by
  point; many EA shapes fit the same arc.
- The smoother's automatic prior is tuned for fast glides. On an orbital
  arc, a "boundary value" warning means you should set `sigma_a` and
  `tau_s` yourself (roughly 1e-6 m/s² and 600 s). Otherwise the EA can be
  far worse than none at all: 61 m against 0.16 m in testing.
- Long observation gaps are silent. The innovation statistics stay normal
  while the repropagation error grows quadratically, so read
  `result.warnings` and run `information_bound` on the real timeline.
- Do not apply the heavy-tail prefilter (`outlier_prob`) to velocities.
- An estimated EA holds its last value if you propagate past the arc on
  which it was estimated. That is an extrapolation, and the trajectory warns
  when it happens.

More on each of these in
[`empirical_acceleration_estimation.md`](empirical_acceleration_estimation.md).
