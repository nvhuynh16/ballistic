# Ballistic

Ballistic is a C++23 orbit propagation library with Python bindings. It
predicts where a satellite will be over the next few hours as precisely as
the force models allow (well under a meter after one hour in low Earth
orbit), and estimates the accelerations those models miss.

## Accuracy

The table gives the position error one hour after the start of a
prediction, measured against precise orbit products. Each prediction starts
from a state taken from the product itself, so the error reflects the force
model and integrator rather than the initial orbit. For the laser-ranging and
GNSS rows the start velocity is computed from the product's positions
(9-point Lagrange derivative, centered windows only): the GNSS products carry
no velocities, and the laser-ranging velocity records are not the derivative
of their own positions.

| Regime | Satellite                                               | Altitude          | 1-hour RMSE                                                                 |
| ------ | ------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------- |
| LEO    | Sentinel, 6 satellites [`SENTINEL-JAN26-SAMPLED`]                   | 700-800 km        | 0.239 m, every prediction under 1 m (0.540 m without radiation and drag)    |
| LEO    | Sentinel, January 2026, out of sample [`SENTINEL-YEAR-JAN26`] | 700-815 km        | 0.240 m (0.570 m with bare defaults)                                        |
| LEO    | Sentinel, September 2026, after all development data [`SENTINEL-YEAR-SEP26`] | 700-1336 km | 0.224 m (0.502 m with bare defaults)                     |
| LEO    | Sentinel, all of 2025, 7 satellites [`SENTINEL-YEAR-2025`] | 700-1336 km       | 0.16-0.26 m per satellite, 0.215 m overall; worst 1.8 m outside maneuvers  |
| LEO    | GRACE-FO [`GRACEFO-500KM`]                                 | 500 km            | 0.47-0.87 m with drag (80-88% better than without)                          |
| LEO    | CHAMP [`CHAMP-370KM`]                                      | 370 km            | 1.2 m with drag (81% better)                                                |
| LEO    | CHAMP, final weeks [`CHAMP-260KM`]                     | 249-283 km        | 6.5 m with drag (89% better)                                                |
| LEO    | GOCE reentry [`GOCE-REENTRY`]                                | 160-245 km        | 14-18 m above 200 km, 164 m at 160-180 km (91-95% better)                   |
| MEO    | LAGEOS-1/2, Etalon-1/2 [`SLR-1H`]                        | 5,900 / 19,100 km | 0.15 m                                                                      |
| MEO    | GPS, 31 satellites [`GNSS-1H`]                           | 20,200 km         | 0.668 m, without radiation pressure                                         |
| IGSO   | QZSS, 3 satellites [`GNSS-1H`]                           | 32,600-39,000 km  | 0.739 m, without radiation pressure                                         |
| GEO    | BeiDou, 5 satellites [`GNSS-1H`]                         | 35,800 km         | 0.805 m, without radiation pressure                                         |
| MEO / IGSO / GEO | GPS / QZSS / BeiDou GEO, with fitted cannonball radiation pressure [`GNSS-1H-SRP`] | 20,200-39,000 km | 0.089 / 0.154 / 0.193 m on days not used for the fit (0.668 / 0.740 / 0.808 m without on the same starts) |

**Longer horizons.** The same measurement after one, three and seven days
(RMSE; the drag rows give the error without drag in brackets; starts whose
window contains a maneuver are excluded):

| Satellite | Altitude | 24 h | 72 h | 7 d |
| --------- | -------- | ---- | ---- | --- |
| GPS / QZSS / BeiDou GEO, without radiation pressure [`GNSS-24H`] [`GNSS-72H`] | 20,200-39,000 km | 220 / 521 / 586 m | 662 / 1,562 / 1,705 m | — |
| GPS / QZSS / BeiDou GEO, with fitted cannonball radiation pressure [`GNSS-24H-SRP`] | 20,200-39,000 km | 7.8 / 11 / 47 m (220 / 521 / 593 m without, same starts) | — | — |
| GRACE-FO, with drag [`GRACEFO-MULTIDAY`] | 490 km | 0.42 km (2.3 km) | 2.9 km (20 km) | 12 km (97 km) |
| CHAMP, final weeks, with drag [`CHAMP-2010-MULTIDAY`] | 249-283 km | 2.4 km (39 km) | 18 km (341 km) | — |
| Sentinel, 2025, recommended configuration, 5 satellites [`SENTINEL-2025-MULTIDAY`] | 700-1336 km | 82 m | 673 m | 3.5 km |
| Sentinel, September 2026, recommended configuration [`SENTINEL-SEP26-MULTIDAY`] | 700-1336 km | 27 m | 244 m | 1.3 km |

Each bracketed ID refers to an entry in
[`docs/validation_claims.md`](docs/validation_claims.md), which records when
the number was measured, at which commit, and the command that reproduces it.

The lower orbits are dominated by drag. The drag rows use the same measured
space weather as the recommended configuration; CHAMP and GOCE use drag
coefficients fitted to orbit data, whereas GRACE-FO uses one computed from
its published panel model ([`docs/satellites.md`](docs/satellites.md)
explains the difference and how independent each score is). The GOCE row is
scored against ESA operations-center orbits, which agree with ESA's
centimeter-level precise science orbit to 0.16-0.24 m on the days both
cover, so the GOCE error comes from the model, not from the reference. The
GNSS rows without radiation pressure show what the default configuration
does; with a per-block cannonball model (mass and block from the IGS
satellite metadata, an area per block with a scale fitted on two other days,
[`docs/gnss_srp.md`](docs/gnss_srp.md)) the error drops 4-8× at one hour and
13-47× at one day.

The reference orbits are ESA Sentinel EOF files, GFZ rapid science orbits
for GRACE-FO and CHAMP, ILRS combined orbits for the laser-ranging
satellites, ESA GOCE orbits, CODE final GPS and MGEX orbits, and Wuhan
University MGEX orbits. If reference frames, time scales or ITRF
realizations are unfamiliar, [`docs/CONCEPTS.md`](docs/CONCEPTS.md)
introduces them.

## Installation

The Python package is published on PyPI as `ballistic-orbit` and is imported
as `ballistic`:

```bash
pip install ballistic-orbit
```

Wheels are published for CPython 3.12 on Linux (x86_64, glibc 2.28 or newer),
Windows (x86_64) and macOS (Apple silicon and Intel, macOS 11 or newer). On
any other platform or Python version pip has no wheel to install; build from
the repository instead, which needs a C++23 compiler (GCC 13+, clang 18 or
MSVC 2022):

```bash
pip install "ballistic-orbit @ git+https://github.com/nvhuynh16/ballistic#subdirectory=python"
```

After installing, `python -m ballistic doctor` checks the installation and
the data files.

To build the C++ library, its tests and the Python module from a checkout
you need CMake 3.20 or newer; Eigen 3.4 is downloaded automatically if it is
not installed:

```bash
git clone https://github.com/nvhuynh16/ballistic && cd ballistic
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBALLISTIC_BUILD_PYTHON=ON
cmake --build build -j$(nproc)
ctest --test-dir build -LE heavy --timeout 1500
```

### The drag models (NRLMSIS 2.1 and HWM14): a separate, non-commercial download

The package does **not** contain the atmosphere and wind models that drag is
computed from. **NRLMSIS 2.1 is licensed by the U.S. Naval Research
Laboratory for research, academic and non-profit use only; commercial use is
not permitted** (NRL-SOF-014-1, see [License](#license)). The two models are
therefore a separate plugin that you install yourself, once per machine,
after accepting that license:

```python
import ballistic
ballistic.install_nrl_models()                     # shows the license terms; installs nothing
ballistic.install_nrl_models(accept_license=True)  # downloads and installs the plugin
```

The plugin (`ballistic_nrl-<os>-<arch>.so`, `.dylib` or `.dll`, about 1 MB)
comes from the [GitHub release](https://github.com/nvhuynh16/ballistic/releases)
of your version. It is checked against the release's SHA-256 list and
installed in the data directory, under `nrl/` (for example
`~/.local/share/ballistic/nrl/` on Linux), where every later session finds it.
`ballistic.nrl_models()` says whether it is installed and where. Nothing is
shown when drag is used afterwards.

**Offline**: download the file for your platform from the release page on
another machine (`ballistic.nrl_models().download_url` names it), then either
run `ballistic.install_nrl_models(accept_license=True, source="path/to/file")`,
or copy it to `ballistic.nrl_models().install_path` (the data directory's
`nrl/` folder, renamed `ballistic_nrl.so` / `.dylib` / `.dll`), or set the
environment variable `BALLISTIC_NRL_MODELS` to its path.

**Without the plugin** everything else works, including drag from a measured
density table (`from_accelerometer_density`). `recommended_config` leaves drag
off with one warning saying what that costs: below 600 km the one-hour error
grows 4 to 8 times, while near 700 km it changed little at 2026 solar activity.
`recommended_config(..., drag=False)` leaves it off without the warning, and
`drag=True` requires it. `make_drag_environment` and the `from_models_*`
tables raise `ConfigError` with the instructions above.

**C++**: the plugin is the CMake target `ballistic_nrl`, built by default
next to the library (`build/nrl/ballistic_nrl.so`; `-DBALLISTIC_BUILD_NRL_PLUGIN=OFF`
skips it). The library never links it: point it at the file with
`InitConfig::nrl_models_path`, the `BALLISTIC_NRL_MODELS` variable, or by
copying it to the data directory's `nrl/` folder. Tests that need it carry
the ctest label `nrl`; without the plugin run `ctest -LE 'heavy|nrl'` and
`pytest -m "not nrl"`.

## Quick start

The library's defaults are the best general settings measured. What it
cannot know is your satellite: its mass and the areas that catch sunlight
and air. You supply those, or the name of a satellite the library already
knows, and it configures radiation pressure and drag from them.

<!-- quickstart: tests/python/test_quickstart.py runs this block verbatim -->
```python
import ballistic

ballistic.load()   # Earth orientation, leap seconds, space weather, ocean tides (cached)

# Where the satellite is, in a named frame. Here: Sentinel-1A from an ESA precise
# orbit, Earth-fixed (ITRF) position [m] and velocity relative to the Earth [m/s].
epoch = ballistic.Epoch.from_utc(2026, 1, 5, 11, 59, 42.0)
state = ballistic.State.from_itrf(epoch,
                                  [-858245.729566, 3986052.969605, -5790116.906923],
                                  [851.699434, 6256.268159, 4183.014931])

# The best measured configuration for this satellite: radiation pressure from its
# registered properties, and drag if the NRL drag models are installed (see
# "The drag models" above; otherwise one warning says drag is off). Your own
# satellite: mass_kg=, srp_area_m2=, drag_area_m2=. A TLE start: see below.
config = ballistic.recommended_config(state, "S1A")

traj = ballistic.Trajectory(state, config)
traj.propagate_for(3600.0)
print(traj(3600.0).position)   # Earth-fixed position one hour later [m]
```

Each step checks its input: `State` refuses kilometers for meters, a point
below the ground, or an inertial velocity passed as Earth-fixed; the
configuration refuses misspelt settings; and the propagation stops with an
error, rather than guessing, where the Earth orientation or space-weather data
run out. `ballistic.doctor()` (or `python -m ballistic doctor`) reports what
data is loaded, its coverage, whether the drag models are installed, and runs
a self-test of a few seconds.

The same in C++:

```cpp
#include <ballistic/ballistic.hpp>
using namespace ballistic;

InitConfig init;
init.nrl_models_path = "build/nrl/ballistic_nrl.so";   // drag models (CMake target ballistic_nrl)
ballistic::initialize(init);   // gravity, ephemeris; EOP/leap seconds from ballistic.load()'s cache
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

States go in and come out in the Earth-fixed frame (ECEF); the equations of
motion are integrated in the inertial GCRS frame. More in
[`docs/configuration.md`](docs/configuration.md).

## The recommended configuration

**If you know your satellite's mass and areas, use radiation pressure, drag
and ocean tides.** `ballistic.recommended_config(state, "S1A")` (a satellite
the library knows, [`docs/satellites.md`](docs/satellites.md)) or
`recommended_config(state, mass_kg=, srp_area_m2=, drag_area_m2=)` switches
on all three on top of the defaults (drag only once the NRL drag models are
installed, see [Installation](#the-drag-models-nrlmsis-21-and-hwm14-a-separate-non-commercial-download)): GCRS with Earth orientation data,
EGM2008 to degree 70, Sun, Moon and Jupiter, relativity, solid and ocean
tides, and an adaptive RK45 integrator. It raises rather than guess a
missing mass or area. Without radiation pressure and drag the one-hour error
in low orbit roughly doubles: 0.50 m instead of 0.22 m on Sentinel in
September 2026 (`SENTINEL-YEAR-SEP26`). Starting from a precise orbit does not
make these forces unnecessary: a precise state fixes only where the satellite
starts. [`docs/configuration.md`](docs/configuration.md) gives the full
settings table, how to work the numbers out from a shape, the settings that
depend on your data (ITRF realization, long horizons, a product's velocity
convention), space weather, and every force field for configuring by hand.

## Reentry and TLE starts

**Reentry.** For a body descending into dense air, give its shape and size
(`recommended_config(state, mass_kg=, drag_area_m2=, reentry=AeroShape.SPHERE,
reference_length_m=)`, with `allow_suborbital=True` on the state): the drag
coefficient then follows the flow regime, from Sentman's free-molecular
formulas to continuum correlations, and `propagate_until` stops at the
ground impact. Heating, ablation and breakup are not modeled. See
[`docs/reentry_and_tle.md`](docs/reentry_and_tle.md#reentry).

### Starting from a TLE

`State.from_tle(line1, line2)` runs the library's own SGP4, which reproduces
the official verification output to 0.00003 mm (`SGP4-VER`), and converts the
state from TEME to the ITRF; TLE elements read as ordinary orbital elements
would put the start kilometers off. `recommended_config(state, accuracy="tle",
horizon_s=...)` keeps only the force terms that matter at TLE accuracy. See
[`docs/reentry_and_tle.md`](docs/reentry_and_tle.md#starting-from-a-tle).

## Data files

Apart from four data files (Earth orientation parameters, the leap-second
table, measured space weather and the FES2004 ocean-tide model), everything
is built in: EGM2008 to degree 360, the DE440 ephemeris for 1950-2100, and
leap seconds to 2017. `ballistic.load()` fetches the four, caches them and
refreshes them when they are stale; without Earth orientation data,
low-orbit predictions are wrong by several meters. Named files and fully
offline use (`offline=True`, `BALLISTIC_OFFLINE=1`) are covered in
[`docs/data_files.md`](docs/data_files.md).

## Documentation

- [`docs/CONCEPTS.md`](docs/CONCEPTS.md): frames, time scales and ITRF realizations
- [`docs/configuration.md`](docs/configuration.md): the recommended configuration, forces, integrators, interpolation
- [`docs/satellites.md`](docs/satellites.md): registered satellites and how their properties were obtained
- [`docs/gnss_srp.md`](docs/gnss_srp.md): radiation pressure for GPS, QZSS and BeiDou satellites, per block
- [`docs/reentry_and_tle.md`](docs/reentry_and_tle.md): reentry and TLE/SGP4 starts
- [`docs/data_files.md`](docs/data_files.md): external data and running offline
- [`docs/empirical_acceleration_estimation.md`](docs/empirical_acceleration_estimation.md): estimating the accelerations a model misses
- [`docs/multi_sensor_estimator.md`](docs/multi_sensor_estimator.md): radar, camera, position-fix and accelerometer fusion
- [`docs/pitfalls.md`](docs/pitfalls.md): common traps and how the library stops you
- [`docs/out_of_range_policy.md`](docs/out_of_range_policy.md): what happens at the edge of every data table
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md): C++ and Python API
- [State transition matrix and covariance](docs/API_REFERENCE.md#state-transition-matrix-and-covariance): how a start-state error or covariance grows along a propagated trajectory (`Trajectory.stm` / `Trajectory.covariance`, C++ `StateTransition`)
- [`docs/MIGRATION.md`](docs/MIGRATION.md): removed names and their replacements
- [`docs/validation_claims.md`](docs/validation_claims.md): every accuracy claim with its reproduction command
- [`docs/layout.md`](docs/layout.md): repository layout and running the tests
- [`docs/latex/README.md`](docs/latex/README.md): C++ and Python tutorials (`bash docs/latex/build.sh`)

## License

The library's own code is released under the MIT license ([`LICENSE`](LICENSE)),
and the Python package contains only MIT code and embedded public data
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). The two drag models are
a separate plugin with their own terms; its source is in this repository and
it is installed only on request (see
[Installation](#the-drag-models-nrlmsis-21-and-hwm14-a-separate-non-commercial-download)):

- **NRLMSIS 2.1** (U.S. Naval Research Laboratory) is the atmosphere behind
  model-based drag: `make_drag_environment`, the `from_models_*`
  generators, and `recommended_config` whenever it turns drag on. It is
  licensed under NRL-SOF-014-1: **research, academic and non-profit use
  only; commercial use is not permitted**
  ([`docs/licenses/nrlmsis2.1_license.txt`](docs/licenses/nrlmsis2.1_license.txt)).
  Propagation without drag, and drag from a measured density table
  (`from_accelerometer_density`), do not use NRL code. For commercial use,
  do not install the plugin, or obtain a license from NRL (contact in the
  notices).
- **HWM14** (Drob et al. 2015, *An update to the Horizontal Wind Model (HWM):
  The quiet time thermosphere*, Earth and Space Science 2, 301-319,
  doi:10.1002/2014EA000089) is included in the plugin on the project's
  reading that it is a work of the U.S. Government (17 U.S.C. §105), with the
  caveats listed in the notices.

`ballistic.install_nrl_models` shows these terms and installs the plugin only
with `accept_license=True`; `ballistic.doctor()` reports whether it is
installed.

## Citation

If you use Ballistic in published work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
