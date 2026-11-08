# Ballistic Python Bindings

High-fidelity orbital trajectory prediction with sub-meter accuracy over one
hour (C++23 core, nanobind bindings).

## Installation

### From Source

```bash
# Prerequisites: CMake 3.20+, a C++23 compiler (GCC 13+, Clang 16+ or
# MSVC 2022; CI covers Linux today), and uv
# (Eigen 3.4 is fetched automatically if not installed system-wide.)

# Option 1: editable install via uv (recommended)
uv venv --python 3.12
uv pip install -e ./python

# Option 2: build the extension via CMake directly
cmake -S . -B build -DBALLISTIC_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Option 3: build a wheel
uv build ./python --wheel
```

The package contains no NRL code. Model-based drag needs the NRLMSIS 2.1
atmosphere and the HWM14 wind, a separate plugin licensed for research,
academic and non-profit use only (NRLMSIS 2.1, NRL-SOF-014-1; commercial use
is not permitted). Install it once, after reading the terms:

```python
ballistic.install_nrl_models()                     # prints the terms, installs nothing
ballistic.install_nrl_models(accept_license=True)  # downloads it from the GitHub release
ballistic.install_nrl_models(accept_license=True, source="ballistic_nrl-linux-x86_64.so")  # offline
```

Without it, `make_drag_environment` and the `from_models_*` generators raise
`ConfigError` saying how to install it, `recommended_config` leaves drag off
with one warning, and everything else works, including drag from measured
density (`from_accelerometer_density`). `ballistic.nrl_models()` and
`ballistic.doctor()` say whether it is installed.

## Quick Start

```python
import ballistic

ballistic.load()   # Earth orientation, leap seconds, space weather, ocean tides (cached)

# The start state names its frame. Here: Sentinel-1A from an ESA precise orbit,
# Earth-fixed (ITRF) position [m] and velocity relative to the Earth [m/s].
epoch = ballistic.Epoch.from_utc(2026, 1, 5, 11, 59, 42.0)
state = ballistic.State.from_itrf(epoch,
                                  [-858245.729566, 3986052.969605, -5790116.906923],
                                  [851.699434, 6256.268159, 4183.014931])

# The best measured configuration for this satellite (radiation pressure and
# drag from its registered properties). Your own satellite: mass_kg=,
# srp_area_m2=, drag_area_m2=.
config = ballistic.recommended_config(state, "S1A")

traj = ballistic.Trajectory(state, config)
traj.propagate_for(3600.0)
print(traj(1800.0).position)   # Earth-fixed position 30 minutes after the start [m]
```

`State` also has `from_gcrs`, `from_j2000`, `from_keplerian`, `from_sp3` and
`from_tle`. Each constructor checks its input (kilometers for meters, a point
below the ground, an inertial velocity passed as Earth-fixed).
`ballistic.doctor()` (or `python -m ballistic doctor`) reports which data
files are loaded and runs a short self-test.

## Batch Processing

```python
import ballistic

ballistic.load()
epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
states = [
    ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0]),
    ballistic.State.from_itrf(epoch, [6878137.0, 0.0, 0.0], [0.0, 7120.0, 0.0]),
]

batch = ballistic.Trajectory(states)
batch.propagate_for(3600.0)    # each member from its own epoch

f = batch()                    # vectorized queries
r = f(1800.0)                  # r.position, r.velocity: (N, 3) numpy arrays
```

## Accuracy

1-hour prediction error against precise orbits (details and reproduction
commands: `docs/validation_claims.md`):

| Regime | Satellites | 1-Hour RMSE |
|--------|------------|-------------|
| LEO | Sentinel-1/2/3/6, recommended configuration [`SENTINEL-YEAR-2025`] [`SENTINEL-YEAR-JAN26`] [`SENTINEL-YEAR-SEP26`] | 0.22-0.24 m |
| MEO | LAGEOS-1/2, Etalon-1/2 [`SLR-1H`] | 0.15 m |
| MEO | GPS (no radiation pressure model) [`GNSS-1H`] | 0.67 m |
| IGSO | QZSS (no radiation pressure model) [`GNSS-1H`] | 0.74 m |
| GEO | BeiDou (no radiation pressure model) [`GNSS-1H`] | 0.81 m |

## API Reference

### Core Classes

- `Epoch` - Time representation with named scales (UTC, TT, TAI, GPS, TDB, UT1)
- `State` - A start state with its frame named and checked
- `TrajectoryConfig` - Propagator configuration (frozen; change a copy with `replace`)
- `Trajectory` - Single or batch propagation with history and interpolation
- `recommended_config` - The best measured configuration for a satellite

### Enums

- `TimeScale` - UTC, TT, TAI, TDB, UT1, GPS
- `PropagationFrame` - GCRS
- `IntegratorType` - RK45, YOSHIDA6, YOSHIDA8
- `ITRFRealization` - ITRF2000, ITRF2005, ITRF2008, ITRF2014, ITRF2020
- `InterpolationFrame` - ECEF, GCRS
- `AltitudeCondition` - Above, Below
- `PropagationDirection` - Forward, Backward

### Modules

The top-level `ballistic` holds what a first session needs. Specialized
tools are one import away:

```python
from ballistic.estimation import estimate_ea_bspline, PositionSamples   # fit an empirical acceleration
from ballistic.estimation import estimate_ea_smoother, RadarObs        # sensor fusion
from ballistic.drag import from_accelerometer_density, parse_density_table  # measured density
from ballistic.aero import AeroBeta, AeroShape, FlowRegimeCd           # reentry drag coefficient
from ballistic.readers import parse_eof_auto, check_velocity_records   # orbit products
from ballistic.tle import parse, Sgp4, tle_grade_preset                # TLEs and SGP4
import ballistic.sp3                                                    # SP3 orbit files
```

A name that used to be top level raises `RemovedApiError` naming its new
import (`docs/MIGRATION.md`). The full reference is `docs/API_REFERENCE.md`.

## License

The package is MIT and contains no NRL code. Model-based drag
(`make_drag_environment`, `from_models_*`, and `recommended_config` when it
turns drag on) runs NRLMSIS 2.1 from the separately installed plugin, so
using it is subject to NRL's license NRL-SOF-014-1 (research, academic and
non-profit use only); propagation without drag and drag from measured density
(`from_accelerometer_density`) do not use it. HWM14, in the same plugin, is
included on the project's reading that it is a U.S. Government work.
`install_nrl_models` shows the terms before installing; `THIRD_PARTY_NOTICES.md`
has the details.
