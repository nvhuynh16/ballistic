# Orbit Propagation: Key Concepts

Background primer for users new to orbit propagation. The API reference is
[`API_REFERENCE.md`](API_REFERENCE.md); the other guides are listed in the
README.


## What Is Orbit Propagation?

Orbit propagation takes a satellite's known position and velocity at one time (the **initial state**) and predicts where the satellite will be at a future time. This is done by numerically integrating the equations of motion, which sum up all the forces acting on the satellite: Earth's gravity, the gravitational pull of the Sun and Moon, atmospheric drag, solar radiation pressure, and relativistic effects.

The accuracy of propagation depends on:
1. **How well you model the forces** (gravity field degree, atmosphere model, etc.)
2. **How accurate your initial state is** (precise orbit products vs. TLE)
3. **How long you propagate** (errors grow over time)
4. **Whether every frame is the one you think it is** (a true-of-date frame read as GCRS puts a low orbit about 22 km off)

## Coordinate Frames: ECI vs. ECEF

Satellites orbit in inertial space, but we observe them from the rotating Earth. Two fundamental reference frames capture this:

**ECEF (Earth-Centered Earth-Fixed)** is a coordinate system that rotates with the Earth. A point on the ground has a constant ECEF coordinate. GPS receivers report positions in ECEF. Satellite state vectors from precise orbit products (SP3 files, ESA EOF files) are almost always in ECEF.

**ECI (Earth-Centered Inertial)** is a coordinate system that does not rotate with the Earth. Newton's equations of motion are simplest in an inertial frame, so propagation is done internally in ECI. The library uses **GCRS** (Geocentric Celestial Reference System) — the IAU 2006/2000A CIO-based frame with 1365-term nutation series; sub-arcsecond at LEO.

**Input and output are always in ECEF (ITRF), including the velocity.** The library converts to GCRS (the inertial frame of the IAU 2006/2000A model) internally for propagation, then converts back. An ECEF velocity is measured relative to the rotating Earth: an inertial velocity is larger by ω×r, about 0.5 km/s at LEO, so never pass an inertial velocity as an ECEF one. GCRS output is available via `state_gcrs()`, and `ballistic.itrf_to_gcrs` / `gcrs_to_itrf` convert states.

## ITRF Realizations: Why They Matter

ITRF (International Terrestrial Reference Frame) is the precise definition of ECEF used by geodesists. Different agencies publish orbit products in different ITRF "realizations" (versions), which differ by millimeters to centimeters. Using the wrong realization adds a small systematic error.

| Data Source | Period | Use ITRF |
|-------------|--------|----------|
| ESA Sentinel | Before March 4, 2025 | `ITRF2014` |
| ESA Sentinel | After March 4, 2025 | `ITRF2020` |
| GFZ GRACE-FO RSO v03 | 2025+ | `ITRF2020` |
| GFZ CHAMP | 2000-2010 | `ITRF2005` |
| ILRS LAGEOS/Etalon | Any | `ITRF2020` |
| IGS GPS | January 2017 - November 2022 | `ITRF2014` |
| IGS GPS | From November 2022 | `ITRF2020` |

If you do not know which ITRF your data uses, the default (`ITRF2020`) is a safe choice -- the error from using the wrong realization is typically 1-3 mm at epoch.

## Time Scales: UTC, UT1, TT, TAI

Precise orbit propagation requires careful treatment of time. Different physical processes use different time scales:

| Scale | Full Name | Use | Relationship |
|-------|-----------|-----|--------------|
| **UTC** | Coordinated Universal Time | Civil time, satellite timestamps, SP3/EOF epochs | Discontinuous (leap seconds) |
| **TAI** | International Atomic Time | Continuous atomic reference | TAI = UTC + leap seconds (currently 37s) |
| **TT** | Terrestrial Time | Ephemeris calculations, gravity models | TT = TAI + 32.184s (exactly) |
| **UT1** | Universal Time 1 | Earth rotation angle (ECEF-ECI conversion) | UT1 ~ UTC +/- 0.9s (DUT1 from IERS) |
| **TDB** | Barycentric Dynamical Time | Solar system ephemeris (JPL DE) | TDB ~ TT + periodic terms (~1.7ms) |
| **GPS** | GPS Time | GPS receiver timestamps | GPS = TAI - 19s |

**For most users**: Create epochs in UTC using `Epoch::from_utc(...)` -- the library handles all conversions internally. The library tracks leap seconds (embedded through 2017; no new ones have been announced since) and can load updated tables if needed.

**DUT1 (UT1-UTC)**: Earth rotation is irregular, so UT1 drifts relative to UTC by up to +/-0.9 seconds. The ECEF-to-ECI transform depends on Earth's actual rotation angle (UT1), not UTC. Without EOP data a propagation raises an error; accepting zeros on purpose (`ballistic.set_eop_policy('assume_zero')`) degrades the transform by ~3-10 m at LEO altitudes. `initialize({})` auto-loads `finals2000A.data` from the platform data directory if cached there — populate it once with `ballistic.Loader().load_eop()`. Check `InitResult.eop_loaded` to verify.

## State Vectors: What You Need to Start

To propagate an orbit, you need:
1. **Epoch**: The time at which the state is known (UTC is most common)
2. **Position**: 3D Cartesian coordinates in meters (ECEF)
3. **Velocity**: 3D Cartesian velocity in m/s (ECEF)

These come from sources like:
- **SP3 files**: Precise orbits from IGS, ILRS, GFZ (typically 15-minute or 5-minute intervals, cm-level accuracy)
- **EOF files**: ESA Sentinel precise orbit ephemerides (10-second intervals, ~2cm accuracy)
- **TLE/OMM**: NORAD two-line elements (converted to Cartesian via SGP4, km-level accuracy)
- **GPS receivers**: Onboard navigation solutions

The accuracy of your initial state directly limits propagation accuracy. A 1 cm initial position error grows to roughly 1 cm + (propagation error) after one hour.

## Forces Acting on Satellites

Orbit propagation accuracy depends on correctly modeling all significant forces. Their relative importance varies with altitude:

| Force | Typical Magnitude | Effect on 1-hr Propagation | Altitude Dependence |
|-------|-------------------|---------------------------|---------------------|
| Earth gravity (central) | ~9 m/s^2 | Dominant (defines orbit) | Decreases with altitude |
| Earth gravity (oblateness J2) | ~1e-3 m/s^2 | ~km-level | Decreases with altitude |
| Higher gravity harmonics | ~1e-5 to 1e-7 m/s^2 | ~1-100 m | Rapid decrease with altitude |
| Third-body (Sun/Moon) | ~5e-6 m/s^2 | ~10-50 m | Increases with altitude |
| Atmospheric drag | ~1e-5 (200km) to 1e-9 (700km) m/s^2 | 0-6000 m | Exponential decrease |
| Solar radiation pressure | ~1e-7 m/s^2 | ~0.5-2 m | Nearly constant |
| Solid Earth tides | ~3e-7 m/s^2 | ~1 m | Decreases with altitude |
| Relativistic effects | ~1e-9 m/s^2 | ~10-20 m | Decreases with altitude |

**Gravity degree**: At low altitudes (LEO, <2000 km), short-wavelength gravity features matter -- use degree 70. At MEO (5000+ km) and above, the gravity field is smooth -- degree 4-8 is sufficient and faster.

**Drag**: Significant below about 1000 km. Exponentially stronger at lower altitudes (the density at 300 km is roughly a thousand times that at 700 km, depending on solar activity). Requires an atmospheric density model (NRLMSIS 2.1) and space weather data (F10.7 solar flux, Ap geomagnetic index). See [Space weather](configuration.md#space-weather) for details.

**SRP**: Solar radiation pressure is nearly constant with altitude but depends on the satellite's area-to-mass ratio and reflectivity. See [The recommended configuration](configuration.md#the-recommended-configuration) for guidance.

## The State Transition Matrix

The state transition matrix Φ(t, t0) answers "if the start state were off by a small δx, how far off would the state be at t?": δx(t) = Φ δx(t0). The library orders the state as [position (m); velocity (m/s)] in the inertial GCRS frame, so Φ is 6×6 and mixes units (its position-from-velocity block is in seconds). It is obtained by integrating dΦ/dt = A(t) Φ along the propagated trajectory, where A holds the derivatives of the acceleration with respect to position and velocity. A start covariance P_0 is carried forward as P(t) = Φ P_0 Φᵀ: a position uncertainty in LEO grows mostly along the track, while the radial part stays bounded. Three limits apply. Radiation pressure switches off at the edge of a hard cylindrical Earth shadow, so the acceleration is not differentiable there and an arc that crosses the shadow boundary gets a Φ that misses the jump. Drag's dependence on position is taken by finite differences (no analytic form), good to roughly 1e-4 relative. And there are no consider parameters yet: uncertainty in β, Cr or the space weather does not enter the covariance, only the start state's.
