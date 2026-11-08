# Configuration

This guide covers how a propagation is set up: the start state, the
recommended force model and how to derive its inputs, the settings that
depend on your data, space weather, configuring every force by hand,
integrators, and interpolation. Every field is listed in
[`API_REFERENCE.md`](API_REFERENCE.md). Reentry and TLE starts have their
own guide, [`reentry_and_tle.md`](reentry_and_tle.md).

## Propagating

```python
epoch = ballistic.Epoch.from_utc(2025, 6, 1, 12, 0, 0)
# Name the frame of the start: Earth-fixed position [m] and velocity relative
# to the rotating Earth [m/s] (as GNSS receivers and POE/SP3 products give it).
state = ballistic.State.from_itrf(epoch, r_m, v_m_s)
# an inertial velocity instead: State.from_gcrs(epoch, r_m, v_m_s) (or from_j2000);
# from a file: State.from_sp3("orbit.sp3.gz", "L64", epoch)

cfg = ballistic.recommended_config(state, "S1A")
# your own satellite instead:
# cfg = ballistic.recommended_config(state, mass_kg=1500.0,
#                                    srp_area_m2=12.0, drag_area_m2=4.0)

traj = ballistic.Trajectory(state, cfg)
traj.propagate_for(3600.0)           # seconds from now
s = traj(1800.0)                     # interpolated 30 min after the start (quintic Hermite)
```

The state is checked before anything runs. Positions in km, a point below
the ground, an orbit whose perigee is below 100 km (pass
`allow_suborbital=True` for a reentry, ballistic or launch state) and an
inertial velocity given as Earth-fixed (the two differ by the Earth's
rotation, ~470 m/s at LEO) raise an error that says what to fix. The
configuration is frozen and takes keywords only, so a misspelt setting
raises; `cfg.replace(gravity_degree=100)` returns a changed copy.

States go in and come out in the Earth-fixed frame (ECEF), while the
equations of motion are integrated in the inertial GCRS frame. The drag
model evaluates the NRLMSIS 2.1 density and HWM14 wind models at the
satellite's position each time the forces are computed, using the whole
space-weather record. There is consequently no arc length or horizon to
choose: the same drag model serves any trajectory, forward or backward in
time, for as long as the weather record reaches. (The arc-duration argument
of the C++ `make_drag_environment` is not used and is kept so that older
code compiles.) Drag roughly doubles the cost of a propagation; 48 hours in
low orbit take 1.1 s with drag and 0.5 s without.

## The recommended configuration

**If you know your satellite's mass and areas, use radiation pressure, drag
and ocean tides.** `ballistic.recommended_config` switches on all three, on
top of the library defaults:

```python
cfg = ballistic.recommended_config(state, "S1A")              # a satellite the library knows
cfg = ballistic.recommended_config(state, mass_kg=1500.0,     # or your own
                                   srp_area_m2=12.0, drag_area_m2=4.0)
```

| You need | For | If you don't give it |
|----------|-----|----------------------|
| `mass_kg` | radiation pressure and drag | raises: name a satellite or give it |
| `srp_area_m2`: mean area facing the Sun | radiation pressure | raises |
| `drag_area_m2`: mean frontal area facing the air, not the total surface | drag, below 1500 km | raises |
| `cd`, `cr` | drag and radiation coefficients | 2.2 and 1.3 |
| the ocean-tide file | ocean tides | `ballistic.load()` fetches it; without it they are skipped, with a warning |

The library does not guess these: a typical area-to-mass ratio is wrong by
about 20 times for a dense satellite, and a wrong drag area quietly biases
every drag result. If you do not know them, a bare `TrajectoryConfig()`
runs with radiation pressure and drag off and ocean tides on (once loaded).
Without all three, the one-hour error in low orbit roughly doubles: 0.50 m
instead of 0.22 m on Sentinel in September 2026 (`SENTINEL-YEAR-SEP26`).
Its first propagation issues one `ConfigHintWarning` (once per process)
giving the measured cost at that altitude; it is advice and never becomes
an error. Starting from a precise orbit does not make these forces
unnecessary: a precise state fixes only where the satellite starts, and the
forces acting on it afterwards still have to be modeled. For a TLE start,
`accuracy="tle"` needs less; for a reentry, the shape and size (see
[`reentry_and_tle.md`](reentry_and_tle.md)).

| Setting | Value | Notes |
|---------|-------|-------|
| Frame | GCRS (IAU 2006/2000A) with Earth orientation data | Without the data, low-orbit error grows by 3-10 m |
| Gravity | EGM2008 to degree 70 | Degrees 100-160 were no better on 2025 Sentinel data, and slower |
| Third bodies and relativity | Sun, Moon and Jupiter; Schwarzschild and Lense-Thirring | |
| Tides | Solid Earth (IERS 2010, Steps 1 and 2) and ocean (FES2004 to degree 50) | Ocean tides and the Step 2 solid tides reduce one-hour low-orbit error by 16-28% |
| Integrator | RK45, adaptive step | |
| Radiation pressure | Solar, Earth albedo and Earth infrared | With radiation pressure, ocean tides on and drag off, Sentinel in January 2026 gives 0.218 m against 0.540 m with neither radiation pressure nor drag (`SENTINEL-JAN26-SAMPLED`). Areas for known satellites are fitted to 2025 data |
| Drag | NRLMSIS 2.1 density, HWM14 wind, measured space weather | 81% better at 370 km (`CHAMP-370KM`), 11-25% at 700-815 km over 2025, no effect above 1300 km (`SENTINEL-YEAR-2025`) |

Over all of 2025, starting from a Sentinel precise orbit every five minutes,
this configuration gives 0.16-0.26 m after one hour (0.215 m over the seven
satellites; 0.261 m without drag) and a worst case of 1.8 m outside
maneuvers (`SENTINEL-YEAR-2025`). On January 2026, which played no part in
any fitting, it gives 0.240 m, compared with 0.570 m for the bare defaults
(`SENTINEL-YEAR-JAN26`), and on September 2026, later than any data used in
development, 0.224 m against 0.502 m (`SENTINEL-YEAR-SEP26`). Drag's benefit
at 700-815 km is smaller after 2025: in January 2026 leaving drag off gave
0.218 m, and in September drag was about neutral (0.226 m without it). The
main reason is that drag itself became small as solar activity declined
(3-7 cm over an hour in January 2026), so day-to-day errors of the density
model now decide whether it helps in a given month; the 2025 orbits also
want about 2-2.5 times the registry drag on Sentinel-2 and -3, whose drag
areas are low-confidence values.

### How the error grows with the horizon

Past one hour the error is almost all along-track, and it grows between
linearly and quadratically with time. In low orbit drag sets it: a
density error changes the orbital period a little, and the position error
from that grows with the square of time. Sentinel at 700-815 km goes from
0.19 m at 1 h to 82 m at 24 h, 673 m at 72 h and 3.5 km at 7 days over
2025 (`SENTINEL-2025-MULTIDAY`), about eight times per tripling of the
horizon. The same configuration gives 27 m / 244 m / 1.3 km in September
2026, when solar activity and with it drag were lower
(`SENTINEL-SEP26-MULTIDAY`); Sentinel-6A at 1336 km, almost free of drag,
reaches only 2.5 m / 40 m / 0.26 km. Drag remains essential at every
horizon: GRACE-FO at 490 km has 0.42 km after a day and 12 km after a week
with drag against 2.3 km and 97 km without (`GRACEFO-MULTIDAY`); CHAMP at
250-280 km has 2.4 km after a day and 18 km after three days against 39 km
and 341 km (`CHAMP-2010-MULTIDAY`). For the navigation satellites the
missing force is radiation pressure, which the GNSS rows leave off: GPS
reaches 220 m at 24 h and 662 m at 72 h, QZSS and the BeiDou GEOs about
twice that (`GNSS-24H`, `GNSS-72H`), growing close to linearly. With a per-block cannonball
radiation model the 24-hour errors fall to 8 / 11 / 47 m (`GNSS-24H-SRP`;
[`gnss_srp.md`](gnss_srp.md)). Satellites
that maneuver often limit the long horizons in practice: Sentinel-1
maneuvers every few days, so no seven-day prediction of it runs
undisturbed.

## Working the numbers out from a shape

For an object without published properties (a calibration sphere, a cube,
a cone of known size, mass and surface), the four numbers come from geometry
and the surface:

- **Areas.** For a body that tumbles, use the mean area it presents, which
  for any convex body is its total surface area divided by 4 (Cauchy's
  theorem). A sphere of radius r: πr² for both areas. A cube of side a:
  1.5a² (the face-on a² only if it holds that attitude). A cone of base
  radius r and slant length l: (πr² + πrl)/4 (πr² if it flies nose
  first). A body held in a known attitude uses the area it actually
  presents: frontal for drag, Sun-facing for radiation.
- **Drag coefficient.** In orbit, molecules hit the surface without
  colliding first, and the coefficient follows from the shape:
  `ballistic.aero.free_molecular_cd(shape, speed_ratio, reemission_ratio)`
  (Sentman's formulas) gives 2.5-2.7 for a sphere at 400-800 km
  (speed ratio 7-9, reemission ratio 0.4-0.55). A cube or a cone tumbling
  lies in a similar range, above the 2.2 default; the uncertainty of the
  coefficient (about ±10%) is of the same order as that of the density.
  Pass it as `cd=`. If you have tracking, the drag scale can be measured
  instead (see
  [From EA to a drag coefficient](empirical_acceleration_estimation.md#from-ea-to-a-drag-coefficient)).
- **Radiation coefficient.** From the surface's specular and diffuse
  reflectivities ρ_s and ρ_d: Cr = 1 + ρ_s + (2/3)ρ_d for a flat surface
  facing the Sun, and Cr = 1 + (4/9)ρ_d for a sphere (its specular part
  cancels by symmetry). A black surface gives 1, a white diffuse sphere
  about 1.4. Pass it as `cr=`.
- **Mass**: weighed, or from the volume and material.

```python
import math
r = 0.5                                            # a 1 m sphere of 300 kg
area = math.pi * r**2
cd = ballistic.aero.free_molecular_cd(ballistic.aero.AeroShape.SPHERE, 7.4, 0.55)   # 2.69
cfg = ballistic.recommended_config(state, mass_kg=300.0, srp_area_m2=area,
                                   drag_area_m2=area, cd=cd, cr=1.0 + 4/9 * 0.5)
```

## Settings that depend on your data

Three settings follow from the data you start from rather than from the
physics.

**The ITRF realization** (`itrf`, default ITRF2020) must match the frame in
which your states are given. ESA Sentinel orbits before 4 March 2025 and IGS
products before November 2022 use ITRF2014, for which you would pass
`itrf=ballistic.ITRFRealization.ITRF2014`.

**Long horizons.** Beyond about six hours, the symplectic integrator
`IntegratorType.YOSHIDA6` keeps the energy error bounded and is the better
choice.

**The meaning of a product's velocity.** The library needs the start
velocity to be the true time derivative of the Earth-fixed position. Not
every orbit product writes velocities that way, and a small discrepancy is
costly: an along-track error of 5e-5 m/s grows to about half a meter in one
hour at low altitude, and nothing in the propagation will reveal it. You can
test a product with `check_velocity_records`, which differentiates the
product's own positions and compares the result with its velocity records:

```python
check = ballistic.readers.check_velocity_records(epoch0, offsets_s, positions, velocities)
print(check.verdict, check.summary)
```

Take `offsets_s` from the file's own time tags rather than from differences
of `Epoch` values. The verdict tells you to use the records as written, to
subtract `cip_drift_velocity` (for products that leave out the slow drift of
the Earth's axis), or to derive velocities from the positions instead. ESA
Sentinel EOF files and GFZ CHAMP and GRACE-FO orbits pass the test; ILRS
laser-ranging orbits do not.

## Space weather

Drag depends strongly on solar and geomagnetic activity, and the library
will not run drag on assumed conditions. If no space-weather file is loaded,
building the drag model raises an error, because a quiet Sun can
underestimate density by a factor of 2 to 20, which is enough to hide a
large error in a satellite's ballistic coefficient.

The CelesTrak file holds measured values for every day since October 1957,
so historical propagations are fully covered. For recent dates, the
measurements end on the day the file was downloaded, and `load()` reports
that day. After it, the library continues with CelesTrak's forecast, which
extends about 45 days, and the trajectory issues a warning. Beyond the
forecast there is nothing left to use, so the propagation stops with an
error, keeping the state it had reached, rather than assuming quiet
conditions. Setting `allow_predicted_weather=False` makes it stop at the
last measured day instead.

Refreshing the weather file can change results for the last weeks of
measured days in the old file: CelesTrak's 81-day centered solar-flux mean
(which NRLMSIS reads) is built partly from forecasts near the end of a file
(141 instead of 154 for 1 January 2026 in one file, `GRACEFO-500KM`).

NRLMSIS 2.1 and HWM14 are a separate plugin, not part of the package
(NRLMSIS 2.1 is licensed for research, academic and non-profit use only):
install it with `ballistic.install_nrl_models(accept_license=True)` (C++: the
`ballistic_nrl` target and `InitConfig::nrl_models_path`). Without it,
`make_drag_environment` and the `from_models_*` generators raise
`ConfigError`, and `recommended_config` leaves drag off with one
`ConfigHintWarning` (`drag=True` makes it required, `drag=False` silences
it). Drag then comes only from measured density (`parse_density_table` +
`from_accelerometer_density`), which needs neither the plugin nor space
weather; below 600 km, leaving drag out instead multiplies the 1-hour error
by 4 to 8. `ballistic.nrl_models()` says whether the plugin is installed.

## Configuring forces by hand

Every option is a field of `TrajectoryConfig`. Optional forces are switched
on by setting their field (`srp = SrpConfig{...}`, `drag_env = <env>`); the
complete list is in [`API_REFERENCE.md`](API_REFERENCE.md).

The same recommended setup in C++:

```cpp
#include <ballistic/ballistic.hpp>
using namespace ballistic;

ballistic::initialize({});   // gravity, ephemeris; EOP/leap seconds from ballistic.load()'s cache
atmosphere::SpaceWeatherData::instance().load_celestrak_csv(get_data_directory() / "SW-All.csv");
gravity::OceanTideModel::instance().load(get_data_directory() / "fes2004_Cnm-Snm.dat");

Epoch t0 = Epoch::from_utc(2025, 6, 1, 12, 0, 0);
Vector3d pos = ..., vel = ...;                                           // ECEF [m], [m/s]

TrajectoryConfig config;                                                 // measured-best defaults
config.srp = SrpConfig{.mass_kg = 2200.0, .area_m2 = 38.4, .cr = 1.0};  // S1A radiation
config.drag_env = atmosphere::make_drag_environment(                     // drag, measured weather
    t0, pos, vel, 3600.0, config,
    {.beta = atmosphere::BetaProfile1D::from_mass_cd_area(2200.0, 2.2, 6.35)});

Trajectory traj(InitialState::from_itrf(t0, pos, vel), config);   // or from_gcrs, from_j2000
auto result = traj.propagate(t0 + 3600.0);   // result.position / .velocity (ECEF)
```

The equation of motion is integrated in GCRS, and each field controls one
term of the model acceleration:

```math
\ddot{\mathbf r} = \mathbf a_\text{model}(\mathbf r, \dot{\mathbf r}, t)
  = \mathbf a_\text{grav}
  + \Delta\mathbf a_\text{solid}
  + \Delta\mathbf a_\text{ocean}
  + \mathbf a_\text{3body}
  + \mathbf a_\text{rel}
  + \mathbf a_\text{rad}
  + \mathbf a_\text{drag}
  + \hat{\boldsymbol\varepsilon}(t)
```

```math
\mathbf a_\text{drag} = -\frac{\rho}{2\beta}\,\lvert\mathbf v_\text{rel}\rvert\,\mathbf v_\text{rel},
\qquad \mathbf v_\text{rel} = \dot{\mathbf r} - \boldsymbol\omega_\oplus \times \mathbf r - \mathbf w
```

Here ρ is the NRLMSIS 2.1 density, **w** the HWM14 wind and β = m / (C_d A).

| Term | Field | Default | Notes |
|------|-------|---------|-------|
| a_grav | `gravity_degree` | 70 | EGM2008, evaluated in ITRF and rotated to GCRS. Embedded to degree 360; degrees 361-2190 through `egm2008_path`. Degree 4-8 suffices above about 5,000 km and is faster |
| Δa_solid | `enable_solid_tides` | on | IERS 2010 solid Earth tides (Steps 1 and 2), applied as corrections to the gravity coefficients |
| Δa_ocean | `ocean_tide_degree` | 50 | FES2004, once `ballistic.load()` has fetched it; 0 turns it off; degree 30 is faster for 1-2 cm more error. Requires `enable_solid_tides` |
| a_3body | `enable_third_body` | on | Sun, Moon and Jupiter, from the embedded DE440 ephemeris |
| a_rel | `enable_relativistic` | on | Schwarzschild and Lense-Thirring |
| a_rad | `srp` | off | `SrpConfig{mass_kg, area_m2, cr = 1.3, albedo, earth_ir}`: solar radiation pressure with a cylindrical Earth shadow, plus Earth albedo and infrared |
| a_drag | `drag_env` | off | `make_drag_environment(...)` or one of the generators below |
| ε̂(t) | `empirical_accel` | off | An estimated correction (see [`empirical_acceleration_estimation.md`](empirical_acceleration_estimation.md)) |
| — | `itrf` | ITRF2020 | Earth-fixed frame realization for inputs, outputs and gravity; match the data source |
| — | `integrator`, `rk45`, `symplectic` | RK45, 60/3600 s | Use steps of about 8 s for fast dynamics such as glides |

By default `make_drag_environment` evaluates ρ and **w** at the satellite's
position using the loaded space weather. Passing `weather` overrides that,
for example `SpaceWeatherHistory.from_constant(WeatherConditions.storm())`
for a scenario study. The lower-level generators, in
`atmosphere/drag_environment_generators.hpp`, build tabulated environments
(wrap the result with `share(...)`):

| Generator | Use |
|-----------|-----|
| `from_models_along` | You already have sampled (epoch, position) arcs |
| `from_models_snapshot` | Short arcs, much less than an orbit, at a fixed latitude and longitude |
| `from_models_4d` | Constellations sharing one atmosphere (heavy; uses OpenMP) |
| `from_accelerometer_density` | Measured density, as for GOCE; add `extend_density_below` (U.S. Standard Atmosphere 1976) below about 80 km |

A tabulated environment covers a fixed time span; past it the atmosphere is
frozen and the trajectory warns once; over a multi-day arc a table built
for one hour misses a large part of the drag effect.
`make_drag_environment` has no such limit.

## Integrators

| Use | Integrator | Notes |
|-----|------------|-------|
| Up to a few hours | `RK45` (default) | Dormand-Prince, adaptive step |
| Six hours or more | `YOSHIDA6` | Symplectic, bounded energy error |
| Long-term studies | `YOSHIDA6` / `YOSHIDA8` | `symplectic.nc_strategy` handles drag and radiation pressure |

## Interpolation and batch queries

```python
batch = ballistic.Trajectory([state1, state2])   # ballistic.State each
batch.propagate_for(3600.0)    # every member from its own epoch, landing exactly
f = batch()                    # BatchInterpolatorFunction
r = f(1800.0)                  # .position/.velocity/.acceleration as (N,3) numpy
impact = f.at_alt(0)           # altitude crossing via bisection (1 mm)
f[0].get_state(epoch + 1800)   # individual TrajectoryInterpolator
```

`TrajectoryInterpolator` (C++) interpolates with quintic Hermite polynomials
(error of order h⁶), extends the history automatically when a query falls
outside it, and offers `propagate_to_altitude()` in both directions.
`InterpolatorConfig.itrf` inherits the realization of the wrapped
`Trajectory` unless you set it. Vectorized Python calls return `None` for
times they cannot cover; pass `on_out_of_range='raise'` to get an error
instead.
