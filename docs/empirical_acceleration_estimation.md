# Empirical acceleration estimation

This guide covers estimating the accelerations a force model misses, using
the library's two estimators: configuration, results and pitfalls. Every
field is listed in [`API_REFERENCE.md`](API_REFERENCE.md). Sensor fusion is
covered in more depth in [`multi_sensor_estimator.md`](multi_sensor_estimator.md).

## The problem

A propagation with an incomplete force model drifts from the truth. Write
the missing part as a time-varying acceleration ε(t):

    r̈ = a_model(r, ṙ, t) + ε(t)

Given an initial state, the force model, and observations along the arc, an
estimator returns ε(t) as an `EmpiricalAcceleration`. Set it on
`TrajectoryConfig::empirical_accel` and the propagation reproduces the
observed arc. You can also split ε into a drag part and a remainder to
recover a ballistic coefficient (see the last section).

How well ε can be recovered depends on how far the observations are from it.
Accelerations measure it directly. Velocities need one differentiation,
positions two, and each differentiation amplifies noise at frequency f by
2πf. Radar adds geometry on top: one station measures range precisely but
constrains transverse position only to σ_angle × range.

| Observation | Derivatives to ε | Noise amplification | In practice |
|---|---|---|---|
| Acceleration | 0 | none | Best accuracy, improving with sample rate. Gauss-Newton refinement does not apply (it would be circular) |
| Velocity | 1 | ×2πf | Sub-meter; Gauss-Newton refinement helps |
| Position | 2 | ×(2πf)² | Ill-posed: the regularization largely determines the answer, so Gauss-Newton refinement matters |
| Radar (range, azimuth, elevation, range rate) | 2, plus the station geometry | ×(2πf)², unequal by direction | Tens of meters, limited by geometry rather than by the method |

A single radar station therefore observes transverse acceleration poorly
whatever estimator is used. Better sensors help more than a better solver:
`information_bound(...)` (below) tells you what a sensor suite can reach
before you fit anything.

## Choosing an estimator

| Data | Estimator |
|------|-----------|
| Dense positions, velocities or accelerations from one source | `estimate_ea_bspline` |
| Radar, cameras, position fixes, accelerometers, or a mix | `estimate_ea_smoother` |
| Not sure | `estimate_ea(EAProblemSpec)` chooses for you |
| Noisy data, unknown σ | `estimate_ea_bspline` with `regularization = ea::CV{}` (cross-validation chooses the regularization; takes hours), or the smoother |

The type of observation is part of the sample type, so passing velocities
to a position fit is a compile-time error.

Results on the reference glide (a 184 s reentry with an 8 m/s² forced-flight
EA). The figures are repropagation RMSE against truth, from the current test
run:

| Case | Estimator | RMSE |
|------|-----------|------|
| Clean positions, 100 Hz | B-spline | < 0.05 m |
| Positions with σ = 1 m | B-spline | 0.3 m |
| Positions with σ = 2 m | B-spline | 0.5 m |
| Clean velocities, 100 Hz | B-spline | 0.1 m |
| Clean accelerations, 100 Hz | B-spline | < 0.05 m |
| One 160 Hz radar plus an impact position fix | smoother | 12.7 m |
| Same, plus a 50 µrad camera | smoother | 3.0 m |
| Same radar and fix, plus an accelerometer with an unknown bias | smoother | 5.3 m |

The radar case is limited by geometry, not by the method. The information
bound for that geometry is about 10-17 m, which is why adding a camera helps
far more than any change to the estimator could.

## B-spline estimator

```cpp
#include <ballistic/calibration/ea_estimator.hpp>
using namespace ballistic::calibration;

EAResult estimate_ea_bspline(epoch0, pos0_ecef, vel0_ecef, PositionSamples{obs}, force_model, cfg);
EAResult estimate_ea_bspline(epoch0, pos0_ecef, vel0_ecef, VelocitySamples{obs}, force_model, cfg);
EAResult estimate_ea_bspline(epoch0, pos0_ecef, vel0_ecef, AccelSamples{obs},    force_model, cfg);
```

Each sample type wraps `std::vector<std::pair<Epoch, Vector3d>>` in its
`.ecef` field. The type tells the estimator what the data is, so mixing them
up doesn't compile. `cfg` is optional.

The estimator propagates the arc, fits a smoothing B-spline to the
residuals, and takes the EA from it: the second derivative for positions,
the first for velocities, the value itself for accelerations. It repeats
this with a regularization weight that decreases each iteration, then
optionally polishes the result with Gauss-Newton against the observations.
The work scales with the number of spline coefficients, not with the number
of observations, so high rates are cheap.

### Observations

All observations are ECEF. The estimator converts them internally and
returns the EA in GCRS.

| Type | Meaning | Conversion |
|------|---------|------------|
| Position | ECEF position [m] | none |
| Velocity | ECEF velocity, the time derivative of ECEF position [m/s] | uses integrated positions for the ω × r term, then GCRS |
| Acceleration | ECEF coordinate acceleration [m/s²] | uses integrated position and velocity for the Coriolis and centrifugal terms, then GCRS |

The velocities must be the true time derivative of the ECEF positions. ESA
Sentinel EOF and GFZ SP3 records are; ILRS SLR records are not quite (see
[the meaning of a product's velocity](configuration.md#settings-that-depend-on-your-data)).

### Configuration

`EABSplineConfig`, with the settings people actually change:

| Field | Default | Notes |
|-------|---------|-------|
| `knot_spacing_s` | 2.0 | Sets the conditioning, not the observation rate. For a feature timescale T, about T/4. For noisy accelerations, wider (~8 s at σ = 0.01 m/s²) |
| `noise_sigma` | 0 | Your observation σ. When set, iterations stop once the RMS residual reaches 1.5σ |
| `regularization` | `ea::IRGN{}` | See below |
| `gn` | `GNRefine{}` (on) | Gauss-Newton refinement. Helps positions and velocities; skipped for accelerations. `std::nullopt` (Python `None`) turns it off and halves the run time |
| `degree` | 5 | 3 also works |
| `reg_deriv_order` | auto | Order of the smoothness penalty: 3 for positions, 1 for velocities and accelerations |
| `max_iterations` | 50 | Stops early after 5 iterations without improvement |
| `max_duration_s` | 3600 | Clamped to the end of the observations plus 10 s. Set it lower only to cut the arc short |
| `brent_line_search` | true | Fewer propagations per iteration. For one-off velocity fits, `false` gives less run-to-run spread |

Regularization options:

- `ea::IRGN{lambda_0 = 1e-4, decay = 0.5, lambda_min = 1e-12}`, the default:
  the weight starts at `lambda_0` and halves each iteration down to
  `lambda_min`. With a known σ and feature scale, `lambda_min = 1e-8` for
  positions and `1e-12` for accelerations work well.
- `ea::CV{}`: hv-block cross-validation over knot spacing and the IRGN
  parameters (495 combinations). It needs no σ, but takes 1-8 hours on 16
  threads. The smoother is usually a faster choice for noisy data.
  `grid = ea::CV::Grid::Coarse` searches 12 combinations (knot spacing 1, 2,
  5, 10 s × λ_min 1e-12, 1e-8, 1e-4), about 40× faster: it finds the right
  region, not the best point in it. When the chosen knot spacing or final λ
  sits on the edge of its grid, a `ConfigHint` diagnostic names the edge (the
  best setting may lie outside the grid). The GN settings in `cfg.gn`,
  including `reg_lambda` and `reg_deriv_order`, apply inside the search too.
  The block layout must hold out at least one block (`hv_n_blocks ≥ 4` for the
  automatic pattern), otherwise `ConfigError`.
- `ea::GCV{lam_min, lam_max}`: picks the weight each iteration by generalized
  cross-validation, searched within `[lam_min, lam_max]` (default 1e-12 to
  0.1).
- `ea::FixedLambda{lambda}`: a constant weight. Only sensible for
  accelerations; it measured worse for positions and velocities.

`GNRefine` fields worth knowing:

- `use_stm` (true) computes the Jacobian from variational equations, about
  twice as fast as finite differences. For a one-off position fit where
  accuracy matters most, `false` gives about 30% lower error.
- `reg_lambda` (1.0) keeps the refinement minimizing the same regularized
  objective as the spline stage. With 0 the repropagation still looks good,
  but on noisy data the pointwise EA gets much worse.
- `max_iterations` (30), `sample_stride` (20).

### Results

`EAResult`:

- `empirical_accel`: the EA in GCRS (null if the run failed).
- `succeeded()`: the run finished without error. A zero EA still counts.
- `has_correction()`: it succeeded and the EA is non-zero.
- `gn_status`: `Disabled`, `Ran`, or `SkippedAccelerationCircular`.
- `obs_rmse`, `lambda_used`, `sigma_sq_estimate`, `iterations_used`.
- `collocation_history`: the trajectory that goes with the returned EA, used
  for the β decomposition. After GN it is the orbit propagated with the
  refined EA (one extra propagation); without GN, for positions it is the
  propagated orbit plus the fitted residual spline.
- `error_message` when it fails.

### Example

```cpp
EABSplineConfig cfg;
cfg.knot_spacing_s = 2.0;
cfg.noise_sigma = 3.0;              // GPS-class positions

TrajectoryConfig fm;                // the model the EA corrects
fm.gravity_degree = 70;

auto r = estimate_ea_bspline(t0, pos0, vel0, PositionSamples{gps_positions}, fm, cfg);
if (r.has_correction()) {
    TrajectoryConfig recon = fm;
    recon.empirical_accel = r.empirical_accel;
    Trajectory traj(t0, pos0, vel0, recon);
}
```

```python
cfg = ballistic.estimation.EABSplineConfig()
cfg.knot_spacing_s = 2.0
cfg.noise_sigma = 3.0                  # cfg.gn is on by default; None turns it off
fm = ballistic.TrajectoryConfig(gravity_degree=70)

r = ballistic.estimation.estimate_ea_bspline(epoch0, pos0, vel0,
                                  ballistic.estimation.PositionSamples(obs),   # [(Epoch, (3,) ECEF), ...]
                                  fm,                               # TrajectoryConfig dataclass
                                  cfg)
if r.succeeded() and r.has_correction():
    recon = fm.replace(empirical_accel=r.empirical_accel)
    traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch0, pos0, vel0), recon)
```

A near-zero repropagation error does not mean the EA is correct point by
point, because many EA shapes fit the same arc. If you need the pointwise
values, as the β decomposition below does, match the knot spacing to the
physics (wide knots for slowly varying drag).

### Outliers

`reject_heavy_tails(samples, sigma, outlier_prob = 0.05, outlier_scale = 10)`
removes samples that sit far from a ±0.25 s local median. The threshold is
about 3.3σ and hardly depends on the two tail parameters, so a rough guess
is fine. With 5% of samples at 10σ, it brings accelerations from 2.85 m to
0.91 m and positions from 0.60 m to 0.37 m. It needs at least ~4 Hz data.
Don't use it on velocities: there it rejects good samples around jerks and
makes things much worse. Clipping outliers toward the local trend instead of
removing them was also tried, and fails badly on positions.

## Smoother

```cpp
#include <ballistic/calibration/ea_smoother.hpp>

EASmootherResult estimate_ea_smoother(const MultiSensorScenario& scenario,
                                      const SensorMask& mask,
                                      const TrajectoryConfig& force_model,
                                      const EASmootherConfig& cfg = {});
```

This is an extended Kalman filter with a Rauch-Tung-Striebel backward pass.
The EA is a first-order Gauss-Markov state alongside position and velocity.
It handles radars (range, azimuth, elevation, range rate), cameras (azimuth,
elevation), position fixes and accelerometers, and any mix of them. Each
sensor keeps its own sample times; nothing needs to be aligned or
resampled. To express a gap, leave the samples out.

```cpp
MultiSensorScenario sc;
sc.epoch0 = t0;  sc.pos0_ecef = r0;  sc.vel0_ecef = v0;
sc.arc_duration_s = 184.0;
sc.add_radar(std::move(radar));     // RadarObs: station geometry + samples
sc.add_optics(std::move(camera));   // OpticsObs
sc.add_position_fix(std::move(fix)); // PositionFixObs

auto r = estimate_ea_smoother(sc, SensorMask::all_on(sc), fm);
// SensorMask::all_on(sc).with_optics(false) etc. to leave a sensor out
```

### Configuration

The defaults are meant to be used as they are. The fields you might change:

| Field | Default | Notes |
|-------|---------|-------|
| `sigma_a`, `tau_s` | auto | Prior amplitude [m/s²] and correlation time [s]. Auto picks, from {1, 3, 10} × {10, 35, 100}, the pair whose normalized innovations are closest to 1. Set both to skip the search (9× faster) |
| `rate_budget` | `Auto` | Range-rate weighting. Auto measures the actual rate residual and uses that or the sensor σ, whichever is larger. A sensor datasheet σ is usually far too optimistic |
| `estimate_clock_bias` | true | Estimates a radar time-tag bias. An unmodeled 5 ms tag error cost a factor of 44 in accuracy in testing |
| `output_dt_s` | 0.5 | Spacing of the returned EA table [s] |
| `passes` | 1 | 2 adds a second iteration, using the first pass's EA as a feed-forward |
| `maneuver_adaptive_q` | false | Inflates process noise after an observation with a large innovation, so impulsive EA is tracked instead of smeared |
| `regime_aware_prior` | false | See below |

The auto grid was tuned on reentry glides. On an orbital arc it is far too
loose, and the EA can make things worse: 61 m against 0.157 m with no EA on
a GRACE-FO arc, while the innovation statistics looked fine. For orbital
arcs, either pin `sigma_a ≈ 1e-6` and `tau_s ≈ 600`, or set
`regime_aware_prior = true`. That option runs one extra fit with the EA
frozen, measures what the dynamics can't explain, and scales the grid to it.
It improved every case tested (orbital 61 m → 0.15 m, camera fusion
3.0 m → 0.8 m, accelerometer fusion 5.3 m → 1.1 m). It isn't the default yet
because it makes the posterior σ inside long gaps too optimistic.

### Results

`EASmootherResult`:

- `empirical_accel` (GCRS, tabulated at `output_dt_s`), `succeeded()`,
  `has_correction()`.
- `ea_sigma`: posterior σ of the EA for each output sample.
- `nis`: normalized innovation squared per degree of freedom. Near 1 means
  the noise model fits the residuals.
- `sigma_a_used`, `tau_used`, `sigma_rate_eff`: what was actually used.
- `clock_biases`: estimated time-tag bias per clock.
- `sigma_honored`: the residual RMS each sensor channel achieved, to compare
  against what you declared.
- `warnings`: see below.

### Warnings

Both are informational; the estimate is still returned.

- **Grid edge.** The auto-selected prior is on the edge of the grid, so the
  best value may lie outside it. This fires on bad fits and on good ones
  alike, so treat it as a prompt to check that the defaults suit your
  regime (on an orbital arc, they don't).
- **Long gap.** The largest gap between observations is longer than τ.
  Inside it the EA comes from the prior alone. A 40 s outage raised the
  error from 12.7 m to 1623 m in testing while NIS stayed normal, because
  unobserved acceleration integrates quadratically. `ea_sigma` does widen
  correctly over the gap; it is the only reliable indication there.

### Error models on sensors

Each sensor struct carries its error model as data:

- `known_bias` is subtracted before fitting.
- `sigma_scale` scales σ per sample.
- `SensorClock{None, Shared, Own}` sets which clock-bias state a sensor
  uses. `Own` keeps one biased radar from contaminating the others.
- On accelerometers, `outlier_prob` / `outlier_scale` switch on a robust
  update for heavy-tailed noise. With 5% of samples at 10σ, error halves
  (22 m to 11 m in `test_ea_smoother`) even with the tail parameters
  declared 2.5× and 2× too small. NIS doesn't detect
  accelerometer outliers, so declare them if you expect them. (Robust
  updates for radars and cameras aren't implemented.)

C++ users can add their own sensor types through
`EASmootherConfig::custom_sensors`. See
[`multi_sensor_estimator.md`](multi_sensor_estimator.md).

### Asynchronous sensors and gaps

Sensors do not need a common time grid. Each keeps its own epochs, and the
smoother processes all observations in time order, so different rates,
unaligned sample times, and sensors that start or stop partway through an
arc need no special treatment.

```cpp
MultiSensorScenario sc;
sc.epoch0 = t0;  sc.pos0_ecef = r0;  sc.vel0_ecef = v0;
sc.arc_duration_s = 184.0;

RadarObs radar = /* station geometry: position_ecef, lat/lon, ecef_to_enu */;
radar.samples  = /* 160 Hz — with a 40 s hardware outage: simply OMIT the
                    missing samples; no placeholders, no NaN sentinels */;
sc.add_radar(std::move(radar));

OpticsObs cam = /* co-located camera */;
cam.samples   = /* 30 Hz, epochs unaligned with the radar's, starts late */;
sc.add_optics(std::move(cam));

auto r = estimate_ea_smoother(sc, SensorMask::all_on(sc), fm, {});
```

```python
radar = ballistic.estimation.RadarObs()
radar.position_ecef = station_ecef
radar.lat_rad, radar.lon_rad = lat, lon
radar.ecef_to_enu = R_enu                      # 3×3
radar.set_samples(t0, t_radar, radar_obs)      # (N,) s from t0; (N, 4) range, az, el, rate
                                               # a 40 s outage: just leave those rows out
cam = ballistic.estimation.OpticsObs()                    # same station fields as the radar
cam.set_samples(t0, t_cam, cam_obs)            # (M,) own epochs; (M, 2) az, el

sc = ballistic.estimation.MultiSensorScenario(epoch0=t0, pos0_ecef=r0, vel0_ecef=v0,
                                   arc_duration_s=184.0, radars=[radar], optics=[cam])
mask = ballistic.estimation.SensorMask.all_on(sc)
bound = ballistic.estimation.information_bound(sc, mask, fm._to_core())   # before fitting
r = ballistic.estimation.estimate_ea_smoother(sc, mask, fm, ballistic.estimation.EASmootherConfig())
print(bound.posterior_bound_m, r.nis, r.warnings)
```

A gap, whether a dropout or data you rejected, is represented simply by
leaving the samples out; `reject_heavy_tails` does this for you. Short gaps
make no difference, but long ones can be expensive in a way the usual health
checks do not reveal (see the long-gap warning above). Running
`information_bound(...)` on the timeline with its gaps shows what an outage
costs before any fitting. A maneuver inside a gap cannot be recovered by
any estimator.

Clocks that disagree are a different problem. Declare the situation with
`SensorClock` (`Own` isolates one suspect sensor) and watch
`sigma_rate_eff` and the audit warnings. In testing, an undeclared 5 ms
time-tag error degraded accuracy by a factor of 44; it is detected, but it
cannot be corrected after the fact.

An estimated EA holds its last value if you propagate past the arc on which
it was estimated. That is an extrapolation, and the trajectory warns when it
happens.

## Letting the library choose

```cpp
#include <ballistic/calibration/ea_problem_spec.hpp>

EAProblemSpec spec;
spec.epoch0 = t0;  spec.pos0_ecef = r0;  spec.vel0_ecef = v0;
spec.observations = PositionSamples{obs};   // or Velocity/AccelSamples, or a MultiSensorScenario
spec.force_model = fm;
spec.noise_sigma = 3.0;                     // required for typed samples

EASpecResult r = estimate_ea(spec);
// r.method_used, r.rationale, r.audit_warnings,
// r.bspline_detail / r.smoother_detail for the full result
```

Typed samples go to the B-spline estimator with Gauss-Newton; a scenario
goes to the smoother. `audit_warnings` reports when your declarations don't
match the data: NIS out of range, a channel's residuals much larger than its
declared σ, or a range-rate budget much larger than the sensor σ, which
usually means bad time tags. Setting `spec.outlier_prob` runs
`reject_heavy_tails` first (positions and accelerations only).
`spec.bspline` and `spec.smoother` override the default configs.

```python
spec = ballistic.estimation.EAProblemSpec()
spec.epoch0, spec.pos0_ecef, spec.vel0_ecef = epoch0, pos0, vel0
spec.observations = scenario          # or ballistic.estimation.PositionSamples([(Epoch, (3,) ECEF), ...])
spec.force_model = fm._to_core()      # the spec takes the core config
spec.noise_sigma = 3.0                # required for typed samples
r = ballistic.estimation.estimate_ea(spec)
print(r.method_used, r.rationale)
for w in r.audit_warnings:
    print(w)
```

## Information bound

```cpp
#include <ballistic/calibration/ea_information_bound.hpp>

EAInformationBound b = information_bound(scenario, SensorMask::all_on(scenario), fm);
// b.posterior_bound_m: what a regularized estimator can reach
// b.crb_bound_m: the prior-free Cramér-Rao bound
```

This computes how well any estimator could do with a given set of sensors,
without fitting anything. Use it to compare sensor suites, or to check what
an outage costs. For the reference radar geometry it gives about 10-17 m
(Cramér-Rao 9.8 m, posterior 16.6 m); adding a 50 µrad camera brings both to
about 1.3 m (`test_ea_problem_spec`). Accelerometers aren't included. Compare suites at the
same `knot_spacing_s` (default 8 s), since finer grids raise the bound.

## From EA to a drag coefficient

Drag acts against the velocity, so the part of ε along −v̂, combined with the
local density, gives the ballistic coefficient β = ρv²/(2|ε_drag|). The
perpendicular part holds everything else. The split is velocity-aligned
versus perpendicular in ECEF, not RTN or body axes.

```cpp
auto r = estimate_ea_bspline(t0, pos, vel, PositionSamples{obs}, drag_free_model, cfg);
auto refined = refine_beta_from_ea(*prior_env, *r.empirical_accel,
                                   r.collocation_history, t0);
config.drag_env = std::make_shared<const atmosphere::DragEnvironment>(refined);
```

```python
r = ballistic.estimation.estimate_ea_bspline(t0, pos, vel, ballistic.estimation.PositionSamples(obs),
                                  drag_free_model, cfg)
refined_env = ballistic.estimation.refine_beta_from_ea(
    prior_env, r.empirical_accel, r.collocation_history, t0)
config = config.replace(drag_env=refined_env)

d = ballistic.estimation.decompose_ea(ea_vec, velocity, density)   # one vector: d.beta, d.drag_fraction
```

`refine_beta_from_ea` keeps the prior environment's density and wind and
replaces its β. For a single vector, `decompose_ea(ea, velocity, density)`
returns `EADecomposition{beta, ea_drag, ea_residual, drag_fraction}`.

In the test case (180 km, high solar activity, a prior β three times too
large) this recovers β = 65.6 against a true 66.2. Two conditions matter:

- The drag signal must be well above the estimator's noise floor (around
  1e-5 m/s²). At 400 km in quiet conditions it isn't, and β can't be
  recovered.
- The knot spacing must suit the drag timescale. A good repropagation does
  not mean the EA is right point by point: many EA shapes fit the same arc.
  Wide knots suit slowly varying drag.

## Tests

The reference case is a 184 s glide reentry with an 8 m/s² forced-flight
acceleration observed at 100 Hz, together with a 300 s orbital arc at
400 km for the null test. The limits below are enforced by
`tests/integration/test_ea_accuracy_claims.cpp` and
`tests/integration/test_ea_smoother.cpp`.

| Check | Result | Limit | Test |
|-------|--------|-------|------|
| No EA in the data gives no EA out | ≤ 7e-8 m/s² peak | < 1e-4 | `test_ea_accuracy_claims`: `null_*` |
| Clean positions / velocities / accelerations | < 0.05 / 0.1 / < 0.05 m | < 0.30 m | `recover_*_bspline` |
| Default `max_duration_s` matches | same | < 0.30 m | `clamp_parity_*` |
| Equivalent noise: ACC < VEL ≤ POS | 0.1 / 0.1 / 0.3 m | ordering margins | `ladder_ordering` |
| Position noise σ = 1 → 2 m | 0.3 → 0.5 m | at most 3.5× | `noise_scaling_pos` |
| β through the EA loop | 0.9% error | < 10% | `beta_recovery_decompose` |
| Radar plus fix, all defaults | 12.7 m | < 25 m | `test_ea_smoother`: `smoother_combo1_*` |
| Plus camera | 3.0 m | < 6 m | `smoother_camera_fusion` |
| Plus accelerometer with bias | 5.3 m | < 8 m | `smoother_accel_fusion_with_bias` |

```bash
ctest --test-dir build -L claims                           # B-spline claims, ~1 min
ctest --test-dir build -R test_ea_smoother --timeout 900   # smoother, ~2 min
```

Other estimators (collocation, a batch multi-sensor fit, an algebraic
step-at-a-time estimator) were each measured worse than one of these two in
every case and are not offered; [`MIGRATION.md`](MIGRATION.md) maps the
collocation and multi-sensor names to the replacements.
