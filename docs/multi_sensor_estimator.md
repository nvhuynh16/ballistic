# Multi-Sensor Empirical Acceleration Estimator

`estimate_ea_smoother` reconstructs an empirical acceleration (EA) from
heterogeneous sensor observations of a single ballistic, reentry or orbital
trajectory: radars (range, azimuth, elevation, range rate), optics
(post-processed azimuth/elevation), known position fixes (impact location,
GPS fixes) and onboard accelerometers with bias estimation. It is an
extended Kalman filter with a Rauch-Tung-Striebel backward pass, the EA a
first-order Gauss-Markov state alongside position and velocity. It chooses
its own prior from the innovation statistics, measures the achievable
range-rate budget, estimates clock biases, and returns a posterior σ for
every output sample. The configuration and result fields are described in
[`empirical_acceleration_estimation.md`](empirical_acceleration_estimation.md#smoother).

**Public API (C++)**: `<ballistic/calibration/ea_smoother.hpp>` (the sensor
types and `MultiSensorScenario` / `SensorMask` are in
`<ballistic/calibration/multisensor.hpp>`). **Python**: top-level
`ballistic.*` — see [Python interface](#python-interface) below.

## When to use this

| Scenario | Use |
|---|---|
| Dense ECEF positions, velocities or accelerations from one source | `estimate_ea_bspline` with typed samples (`ea_estimator.hpp`) |
| Mixed-sensor fusion (radar + optics + fixes + accelerometer) | `estimate_ea_smoother` (this document) |
| Radar alone, or noisy positions | `estimate_ea_smoother` |
| Onboard accelerometer with unknown bias | `estimate_ea_smoother` (bias is jointly estimated) |
| Not sure | `estimate_ea(EAProblemSpec)` chooses and audits the declarations |

## Frame conventions

| Quantity | Frame | Notes |
|---|---|---|
| Sensor position (`RadarObs::position_ecef`, etc.) | ECEF | ITRF2020 default |
| Initial state (`scenario.pos0_ecef`, `vel0_ecef`) | ECEF | see "Initial state" below |
| Radar / optics observations | local az/el + range, range-rate | sensor-frame; rotation cached in `ecef_to_enu` |
| `PositionFixObs::samples` | ECEF position | one observation per fix |
| `AccelObs::samples` | ECEF specific force | accelerometers natively output body frame; rotate to ECEF using spacecraft attitude before populating |
| Epochs | any scale | `Epoch` instances |
| Output `result.empirical_accel` | **GCRS** | use `EmpiricalAcceleration::accel_at(t)` to evaluate |

## Quickstart

```cpp
#include <ballistic/calibration/ea_smoother.hpp>
#include <ballistic/coordinates/geodetic.hpp>

using namespace ballistic;

// 1. Place sensors.
auto radar = calibration::RadarObs{};
constexpr double deg = constants::DEG_TO_RAD;
radar.position_ecef = coords::geodetic_to_ecef(35.0 * deg, -120.0 * deg, 0.0);
radar.sigma_range = 2.0;                       // m
radar.sigma_angle = 0.02 * deg;
radar.sigma_rate  = 0.02;                       // m/s
auto geo = coords::ecef_to_geodetic(radar.position_ecef);
radar.lat_rad = geo.latitude;
radar.lon_rad = geo.longitude;
radar.ecef_to_enu = coords::ecef_to_enu_rotation(geo.latitude, geo.longitude);

// 2. Populate observations (here from a CSV; one row per observation).
for (auto& row : my_csv_data) {
    Eigen::Vector4d obs;
    obs << row.range_m, row.az_rad, row.el_rad, row.range_rate_mps;
    radar.samples.emplace_back(row.epoch, obs);
}

// 3. Build the scenario.
calibration::MultiSensorScenario scenario;
scenario.epoch0 = my_csv_data.front().epoch;
scenario.pos0_ecef = known_initial_pos_ecef;
scenario.vel0_ecef = known_initial_vel_ecef;
scenario.arc_duration_s = my_csv_data.back().epoch - scenario.epoch0;
scenario.add_radar(std::move(radar));

// 4. Run (the defaults are meant to be used as they are).
TrajectoryConfig fm;
fm.gravity_degree = 20;
auto mask = calibration::SensorMask::all_on(scenario);
auto result = calibration::estimate_ea_smoother(scenario, mask, fm);

// 5. Use the result.
auto ea = result.empirical_accel;              // GCRS frame
Vector3d ea_at_t = ea->accel_at(30.0);          // m/s² at t=30s since epoch0
for (auto& w : result.warnings) std::cout << w << "\n";
```

## Sensor types

### `RadarObs` — range / az / el / range-rate (4 channels per sample)

- σ_range default 2 m, σ_angle 0.02° (~3.5e-4 rad), σ_rate 0.02 m/s.
- ENU frame cached in `lat_rad`/`lon_rad`/`ecef_to_enu`. Populate all three
  from the station's geodetic location, for example:

  ```cpp
  calibration::RadarObs make_radar_at_geodetic(double lat_deg, double lon_deg,
                                               double alt_m) {
      calibration::RadarObs r;
      r.position_ecef = coords::geodetic_to_ecef(
          lat_deg * constants::DEG_TO_RAD, lon_deg * constants::DEG_TO_RAD, alt_m);
      auto geo = coords::ecef_to_geodetic(r.position_ecef);
      r.lat_rad = geo.latitude;
      r.lon_rad = geo.longitude;
      r.ecef_to_enu = coords::ecef_to_enu_rotation(geo.latitude, geo.longitude);
      return r;   // σ_range / σ_angle / σ_rate keep their defaults
  }
  ```
- The smoother weights range rate by `EASmootherConfig::rate_budget`
  (`Auto` measures the achievable rate residual and uses it or the sensor
  σ, whichever is larger).

### `OpticsObs` — az / el only (2 channels per sample)

- The input is post-processed angular bearings; no pixel/intrinsic
  processing is assumed. Same frame caching as radar.
- σ_angle default 50 µrad.

### `PositionFixObs` — ECEF position fixes (3 channels per sample)

- Single σ_pos for all fixes in this object. (For per-sample σ, use
  multiple `PositionFixObs` entries with different σ values.)
- Typical use: impact location at end of arc, occasional GPS fixes.

### `AccelObs` — onboard accelerometer with bias estimation (3 channels)

- Bias is **jointly estimated** alongside the EA.
- σ_bias_prior controls how tightly bias is regularized toward zero
  (default 1e-3 m/s² ≈ tactical-grade IMU).
- Samples must be **ECEF specific force**. If your sensor outputs body
  frame, rotate using the spacecraft attitude at the sample epoch before
  populating `samples`.

## Declaring the error model

The smoother's error model is **declarative — data on the sensor structs,
not callbacks**:

| Field | On | Meaning |
|---|---|---|
| `known_bias` | all sensors | Subtracted from every observation before estimation. A known bias is data, not a state. (Accel: the *known part*; the residual unknown bias is still jointly estimated under `sigma_bias_prior`.) |
| `sigma_scale` | all sensors | Optional per-sample σ multiplier (aligned with `samples`; empty = the scalar σ's apply). |
| `clock` | radar, optics | `SensorClock::None` (tags known accurate), `Shared` (one clock state shared by all Shared sensors — radar default), `Own` (this sensor gets its own clock state). Master switch: `EASmootherConfig::estimate_clock_bias`. |

Declarations are **audited claims**, not silent trust. The result reports:

- **Closed-loop data consistency (measured and gated)**: feeding the
  returned EA back through `TrajectoryConfig::empirical_accel` and
  re-propagating lands the trajectory on the observations at the declared
  noise level — about 1.7× the declared σ in range and 1.2× for the fix,
  angles at their declared σ, rate within the honest budget (gate
  `closed_loop_data_consistency`). The position-channel inflation comes
  from the spacing of the returned EA table: a finer `output_dt_s` (0.1 s)
  brings the repropagated residuals to those of the smoothed fit.
- `sigma_honored` — per-channel residual RMS at the smoothed solution
  ("radar0_range", "fix0_pos", …). Honored ≫ declared means the declaration
  could not be honored: mis-declared noise, un-modeled bias/timing, or
  model-class limits. Measured gate cell: σ_range declared 0.2 m against
  2.0 m truth noise → NIS/dof 25.6 and honored 2.0 m. Flagged, not hidden.
- `clock_biases` — recovered time-tag bias per clock block. Gate cell: a
  +5 ms-biased radar declared `Own`, fused with a clean camera, leaves the
  camera channel at its 49.9 µrad sensor floor (unpolluted) while the
  radar's clock absorbs the timing error. Caveat: the scale and sign of the
  recovered bias are not yet calibrated — treat the value as an indicator.

Prefer `estimate_ea(EAProblemSpec)` (`ea_problem_spec.hpp`) as the front
door: it dispatches to the right family and surfaces `audit_warnings`
(NIS out of band, honored ≫ declared, measured rate budget ≫ sensor σ ⇒
suspect time tags). `information_bound(...)` answers "what could ANY
estimator do with this data" *before* you tune anything.

## Asynchronous timelines and gaps

The smoother imposes **no common time grid**: `detail::build_timeline`
merges every active sensor's samples (plus custom sensors)
into one time-ordered `ObsRef` sequence, and `ekf_forward` propagates to
each observation's own epoch before applying that sensor's update. Different
rates, unaligned epochs, and sensors that start/stop mid-arc need no
configuration. Users never call `build_timeline` directly — assembling a
`MultiSensorScenario` and calling `estimate_ea_smoother` / `estimate_ea` is
the interface; the merge is automatic.

Gaps (hardware dropout, or bad data deliberately removed — e.g. by
`reject_heavy_tails`) are expressed by omitting samples; never insert
placeholders. The GM prior bridges the gap and the filter covariance grows
across it. Measured behavior (10/40/80 s radar outages on the reference
radar-plus-fix case): mid-gap `ea_sigma` stays calibrated (z = 1.6 at the
center of a 40 s gap in the gate cell), but the reprop cost
of a long outage is CATASTROPHIC AND SILENT: 40 s → 12.7 → 1623 m (128×)
with NIS in-band at 0.77. The smoother therefore emits a report-only
prior-dominated warning (`EASmootherResult::warnings` / spec
`audit_warnings`) whenever the largest observation gap exceeds τ — gated by
`gap_is_flagged_and_covered`. A maneuver inside a gap is unobservable by
any estimator; `information_bound(...)` quantifies outage cost pre-fit.
Mis-synced *clocks* are a different problem — see `SensorClock` in the
error-model section.

## Custom sensors and non-Gaussian noise (C++ only)

`EASmootherConfig::custom_sensors` accepts `detail::CustomSensor` — obs
times plus a callback returning `{r, H, R}` against the smoother state.
The callback rides the same EKF/RTS path as built-ins (gated bit-identical
to the built-in fix adapter) and sees the same `EpochContext` (full-IAU
frame machinery — never substitute the GAST-only shortcut; it is 35 km
wrong on this problem). Build `H` via `StateLayout` indices only. This
hook is deliberately not Python-bound: a per-observation Python callback
crosses the GIL ~300k times per fit.

**Case A — the mechanism transforms (valid).** If your sensor's noise is
non-Gaussian because of a *known mechanism*, transform the observation so
the noise becomes (approximately) Gaussian and write the measurement model
in the transformed space. Worked example, multiplicative noise
`y = s·(1+ε)`, ε ~ N(0, σ²): observe `log y = log s + log(1+ε) ≈ log s + ε`.
The callback returns `r = log(y_obs) − log(h(z))`, `H = (∂ log h/∂z)`
= `(1/h)·∂h/∂z`, `R = σ²`. The transform is exact *because the mechanism
is known* — the Jacobian chain rule does the rest.

**Case B — normal-scoring additive noise (an illusion).** If the noise is
additively non-Gaussian (`y = h(x) + ε`, ε heavy-tailed), mapping the
*observations* through a normal-score/quantile transform does **not**
produce a linear-Gaussian problem: the transform is nonlinear in `h(x) + ε`
jointly, so the transformed measurement function is no longer `h`, the
noise is no longer independent of the state, and the EKF linearization
point is wrong exactly where the tails matter. This route was analyzed and
rejected — do not resurrect it. For heavy-tailed additive noise the honest
options are: declare a larger σ (cheap, robust), use `sigma_scale` to
down-weight suspect samples, or — for the ACCEL channel — declare the
tail: `AccelObs::outlier_prob` / `outlier_scale` enable the mixture-score
(Masreliez) robust update (5%×10σ contamination: the error halves, 21.8 m
→ 11.2 m in the gate cell `robust_accel_mixture_update`, with the parameters
declared 2.5×/2× too small — the ψ shape matters more than the exact
values; a sequential update works where a batch Huber loss does not,
because innovations separate outliers from signal). NIS
does NOT flag accel-tail contamination (radar rows dominate the dof) —
this is a silent failure mode without the robust update.

**NIS audits your transform.** Whatever you declare or transform, the
innovation whiteness check runs on the result: a Case-A transform that is
actually wrong, or a Case-B illusion, shows up as NIS drift out of the
[0.3, 3] band and as honored ≫ declared. Check `audit_warnings` after
every fit with a custom sensor.

## Initial state

The initial state (`pos0_ecef`, `vel0_ecef` on `MultiSensorScenario`) is
the start of the reference trajectory. Typical sources:
- External orbit determination at a recent epoch.
- Launch state from telemetry.
- Triangulation from the first ~30 s of radar observations (classical
  initial orbit determination). The polynomial second derivative of
  triangulated positions is noise-dominated at long range, so this is
  harder than it looks.

## Observability limits

The fundamental observability of EA depends on sensor geometry, not on the
estimator:
- **Radar + impact fix only**: one station measures range precisely but
  constrains transverse position only to σ_angle × range. For the reference
  glide geometry the information bound is about 10-17 m, and the smoother
  reaches 12.7 m.
- **A second radar** of the same type 40 km from the first (lines of sight
  crossing over the impact point): the smoother reaches 10.6 m, measured on
  the test geometry (`test_ea_smoother`, `two_radars_beat_one`).
- **Adding a camera**: a 50 µrad camera lowers the bound to about 1.3 m
  (`information_bound`; the smoother achieves 3.0 m).
- **With accelerometer**: directly constrains EA (5.3 m with radar, fix and
  a biased accelerometer, limited by the bias ambiguity without a camera).

When picking sensors for a deployment, run `information_bound(...)` on the
planned timeline first: it gives what any estimator could reach without
fitting anything.

## Python interface

The full multi-sensor surface is exposed at the top level (`import
ballistic`):

- **Numpy-friendly sample builders**: `sensor.set_samples(epoch0, times_s, obs)`
  takes an `(N,)` times array and `(N, K)` observations array (K = 4 radar,
  2 optics, 3 fix/accel). The `.samples` list-of-tuples accessor is
  preserved for reading.
- **Keyword-arg `MultiSensorScenario` constructor**: dataclass-style
  construction without manual default-init then field assignment.
- **Batched EA evaluation**: `result.empirical_accel.evaluate_at(times)` returns
  an `(N, 3)` numpy array.
- **Sensor forward models**: `RadarObs.predict(state)` and `OpticsObs.predict(state)`
  return the model's expected observation vector for use in synthetic data
  generation.

Example:

```python
import ballistic
import numpy as np

ballistic.load()
epoch0 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
times_s = np.linspace(0.0, 60.0, 61)
obs = ...  # (N, 4) numpy array of [range, az, el, range_rate]

radar = ballistic.estimation.RadarObs()
radar.position_ecef = station_ecef
radar.sigma_range, radar.sigma_angle, radar.sigma_rate = 2.0, 5e-4, 0.02
radar.lat_rad, radar.lon_rad, radar.ecef_to_enu = lat, lon, ecef_to_enu_mat
radar.set_samples(epoch0, times_s, obs)

fix = ballistic.estimation.PositionFixObs()
fix.sigma_pos = 1.0
fix.set_samples(epoch0, np.array([0.0]), pos0.reshape(1, 3))

scenario = ballistic.estimation.MultiSensorScenario(
    epoch0=epoch0, pos0_ecef=pos0, vel0_ecef=vel0, arc_duration_s=60.0,
    radars=[radar], fixes=[fix])
mask = ballistic.estimation.SensorMask.all_on(scenario)

fm_cfg = ballistic.TrajectoryConfig()  # dataclass accepted directly
result = ballistic.estimation.estimate_ea_smoother(
    scenario, mask, fm_cfg, ballistic.estimation.EASmootherConfig())

# Batched EA evaluation at 100 times → (100, 3) numpy array
ea_grid = result.empirical_accel.evaluate_at(np.linspace(0.0, 60.0, 100))
print(result.nis, result.sigma_honored, result.warnings)
```

End-to-end Python test: `tests/python/test_multisensor_e2e.py`.

## See also

- `tests/integration/test_ea_smoother.cpp` — the smoother's accuracy gates
  (radar plus fix, camera fusion, accelerometer fusion, gaps, clocks).
- [`empirical_acceleration_estimation.md`](empirical_acceleration_estimation.md)
  — both estimators, configuration and results.
- `include/ballistic/calibration/ea_estimator.hpp` — single-observation-type
  alternative when only one of (POS / VEL / ACC) is available.
