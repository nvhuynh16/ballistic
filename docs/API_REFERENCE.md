# Ballistic Library API Reference

High-fidelity orbital mechanics and trajectory prediction library achieving sub-meter RMSE for 1-hour orbit predictions.

## Table of Contents

0. [Errors and warnings](#errors-and-warnings)
1. [C++ API](#c-api)
   - [Initialization](#initialization)
   - [Epoch and Time](#epoch-and-time)
   - [Trajectory Propagation](#trajectory-propagation)
   - [State transition matrix and covariance](#state-transition-matrix-and-covariance)
   - [Trajectory Interpolation](#trajectory-interpolation)
   - [External Data Loading](#external-data-loading)
   - [Coordinate Transformations](#coordinate-transformations)
   - [Empirical Acceleration Estimation](#empirical-acceleration-estimation)
   - [Multi-Sensor EA Estimation](#multi-sensor-ea-estimation)
   - [Configuration Types](#configuration-types)
   - [Result Types](#result-types)
2. [Python API](#python-api)
   - [Start here: the top-level `ballistic`](#start-here-the-top-level-ballistic)
     ([Recommended setup](#recommended-setup-start-here),
     [Module Functions](#module-functions),
     [Data Loading Functions](#data-loading-functions),
     [Loader Class](#loader-class-auto-download),
     [Epoch Class](#epoch-class-python),
     [Trajectory Class (Unified API)](#trajectory-class-python---unified-api),
     [TrajectoryConfig Dataclass](#trajectoryconfig-dataclass-python),
     [InterpolatorConfig Dataclass](#interpolatorconfig-dataclass-python),
     [Configuration Classes](#configuration-classes-python),
     [Enumerations](#enumerations),
     [C++ Core Classes](#c-core-classes-python))
   - [`ballistic.estimation`](#ballisticestimation-fitting-an-empirical-acceleration)
   - [`ballistic.drag`](#ballisticdrag-atmospheres-built-by-hand)
   - [`ballistic.aero`](#ballisticaero-flow-regime-drag-coefficient)
   - [`ballistic.readers`](#ballisticreaders-orbit-products-and-velocity-checks)
   - [`ballistic.tle`](#ballistictle-tles-and-sgp4)
   - [`ballistic.sp3`](#ballisticsp3-sp3-orbit-files)
   - [`ballistic.events`](#ballisticevents-events-during-propagation)
   - [`ballistic.constants`](#ballisticconstants)
3. [External Data Files](#external-data-files)
4. [Accuracy Summary](#accuracy-summary)
5. [Constants](#constants)
6. [Version](#version)

---

## Errors and warnings

The library never substitutes a value silently. It either raises an error or
continues with an approximation and says so.

**Errors** (`core/errors.hpp`; Python `ballistic.errors`): every deliberate
error is a `ballistic::Error` (`ballistic.BallisticError`, a `RuntimeError`):

| C++ / Python | Raised when |
|--------------|-------------|
| `DataCoverageError` | required data (EOP, space weather, leap seconds, an orbit file) does not cover the request |
| `FrameError` | an input is in a different reference frame than expected |
| `InputPlausibilityError` | an input is physically implausible and was refused |
| `ConfigError` | a configuration value cannot be honored |
| `RemovedApiError` | a removed API was used; the message names the replacement (`docs/MIGRATION.md`) |
| `StrictDiagnosticError` | C++ strict mode turned a diagnostic into an error |

**Diagnostics** (`core/diagnostics.hpp`): each approximation is reported
once through a process-wide channel with a category:

- DataCoverage, ModelSubstitution, DataSkipped, ForcedStep, Extrapolation,
  FrameAssumption, ConfigHint.
- In C++ it is printed to stderr by default; `diagnostics::set_handler(...)`
  replaces that. In Python each category arrives as a warning class
  (`DataCoverageWarning`, `ModelSubstitutionWarning`, `DataSkippedWarning`,
  `ForcedStepWarning`, `ExtrapolationWarning`, `FrameAssumptionWarning`,
  `ConfigHintWarning`; all `ballistic.BallisticWarning`, a `RuntimeWarning`),
  so the usual warnings filters apply.
- Strict mode turns every diagnostic except ConfigHint (advice) into an
  error: in C++ `BALLISTIC_STRICT_WARNINGS=1` or
  `diagnostics::set_strict(true)`; in Python `ballistic.set_strict(True)`.
- A `Trajectory` also keeps its own notes in `warnings()`. Notes about
  global data (no EOP loaded, forecast space weather, ocean tides missing)
  are delivered once per process.

## C++ API

### Initialization

#### `ballistic::initialize()`

Initialize the library with configuration options. With defaults (`initialize({})`), embedded
gravity/ephemeris are loaded and EOP / leap-seconds files are **auto-discovered** in the
platform data directory if present.

```cpp
#include <ballistic/ballistic.hpp>

// Simple initialization (embedded data + auto-discovered EOP / leap-seconds)
ballistic::initialize({});

// Equivalent explicit form — every field of the default InitConfig with the
// value it carries when you call initialize({}). Anything you don't override
// stays at these values. Path comments show where auto-discovery looks on
// Linux; other platforms use:
//   Windows: %LOCALAPPDATA%\ballistic\<filename>
//   macOS:   ~/Library/Application Support/ballistic/<filename>
// The environment variable BALLISTIC_DATA_DIR, when set, names the data
// directory itself on every platform.
//
// Per-propagation gravity degree is set on TrajectoryConfig::gravity_degree
// (default 70 for LEO). All embedded EGM2008 coefficients (2-360) are loaded.
ballistic::InitConfig config;
config.egm2008_path           = std::nullopt;  // -> embedded EGM2008 (deg 2-360, no external file)
config.eop_path               = std::nullopt;  // -> auto-discovers ~/.local/share/ballistic/finals2000A.data (or .all)
config.jpl_ephemeris_path     = std::nullopt;  // -> embedded DE440 (1950-2100, no external file)
config.leap_seconds_path      = std::nullopt;  // -> auto-discovers ~/.local/share/ballistic/leap-seconds.list

auto result = ballistic::initialize(config);

if (result.success) {
    std::cout << "Gravity degree: " << result.gravity_degree_loaded << "\n";
    std::cout << "JPL version: " << result.jpl_de_version << "\n";
    if (result.eop_auto_discovered) {
        std::cout << "Auto-loaded EOP from " << result.eop_source_path << "\n";
    }
}
```

#### `InitConfig` Structure

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `egm2008_path` | `optional<path>` | none | Path to extended EGM2008 coefficients (deg 361-2190 from NGA file). Embedded deg 2-360 are always loaded; this only extends. |
| `eop_path` | `optional<path>` | none | Path to IERS `finals2000A.data` (overrides auto-discovery) |
| `jpl_ephemeris_path` | `optional<path>` | none | Path to .bsp (SPK) or .bin file (overrides embedded DE440) |
| `leap_seconds_path` | `optional<path>` | none | Path to `leap-seconds.list` file (overrides auto-discovery) |

**Loading semantics**: explicit paths (any `*_path` you set) are **strict** — if the file is
missing or fails to parse, `initialize()` returns `success=false` with `error_message`
containing the path and download URL. Auto-discovery is **best-effort** — if the file in
the data directory is missing or corrupt, init still succeeds and the corresponding
`*_loaded` flag stays `false`. This means `initialize({})` always succeeds (modulo OOM);
`initialize(cfg)` with explicit paths fails loudly on misconfiguration. An
auto-discovered file that does not parse is not skipped silently: it is
listed in `InitResult::warnings` and reported as a DataSkipped diagnostic.

#### `InitResult` Structure

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | True if initialization succeeded |
| `egm2008_loaded` | `bool` | True if external EGM2008 file loaded |
| `eop_loaded` | `bool` | True if EOP file loaded (via explicit path OR auto-discovery) |
| `eop_auto_discovered` | `bool` | True if EOP was loaded via data-dir auto-discovery (not explicit `eop_path`) |
| `jpl_ephemeris_loaded` | `bool` | True if external JPL file loaded |
| `embedded_ephemeris_loaded` | `bool` | True if embedded DE440 loaded |
| `leap_seconds_loaded` | `bool` | True if external leap-seconds file loaded |
| `leap_seconds_auto_discovered` | `bool` | True if leap-seconds was loaded via auto-discovery |
| `gravity_degree_loaded` | `int` | Actual gravity degree available |
| `jpl_de_version` | `string` | DE version string (e.g., "DE421" or "DE440 (embedded)") |
| `eop_source_path` | `string` | Filesystem path of loaded EOP file (empty if none) |
| `error_message` | `string` | Error description if failed |
| `eop_table_end_mjd` | `double` | Last MJD in the loaded EOP table (0 = no EOP) |
| `eop_table_stale` | `bool` | EOP loaded but even its predictions end before today |
| `leap_seconds_expiration_jd` | `double` | Expiry declared by the leap-seconds file (0 = unknown / embedded) |
| `leap_seconds_expired` | `bool` | The file's declared expiry is in the past |
| `warnings` | `vector<string>` | Staleness notes and data files that were found but could not be used |

**Gravity ceiling (embedded vs external).** The embedded EGM2008 set stops at degree 360
(~2 MB, sufficient for sub-meter at every validated regime LEO → HEO). To extend to degrees
361-2190, set `config.egm2008_path` to the NGA `EGM2008_to2190_TideFree` file (~75 MB). The
runtime degree used by each propagation is set on `TrajectoryConfig::gravity_degree` (default
70 for LEO) — the Clenshaw inner loop iterates only up to the evaluation degree
(`gravity/clenshaw_engine.hpp`), so a higher loaded ceiling adds memory but **no propagation-time
cost**. A `gravity_degree` above the loaded model raises `ConfigError` when the `Trajectory`
is constructed.

#### `ballistic::build_info()` (`core/build_info.hpp`)

`constexpr BuildInfo build_info()` says how the library was compiled:
`with_openmp`, `compiler`, `cpp_standard` and `build_type` (from `NDEBUG`).
The macros come from the `ballistic` CMake target (PUBLIC), so a consumer
sees the same answer as the library's own sources.

#### NRL models plugin (`atmosphere/nrl_models.hpp`)

NRLMSIS 2.1 (atmosphere) and HWM14 (wind) are not compiled into the library:
NRLMSIS 2.1 is licensed for research, academic and non-profit use only
(NRL-SOF-014-1), so both are the separate shared library `ballistic_nrl`
(CMake target, built by default into `build/nrl/`; `-DBALLISTIC_BUILD_NRL_PLUGIN=OFF`
skips it), with a plain C interface (`atmosphere/nrl_plugin_abi.h`,
`BALLISTIC_NRL_ABI_VERSION`). Model-based drag (`make_drag_environment`, the
`from_models_*` generators, the live model columns) loads it on first use
and raises `ConfigError` with installation instructions when it is absent;
drag from measured density never needs it. Results through the plugin are
bit-identical to calling the model code directly (`test_nrl_models`).

```cpp
InitConfig init;
init.nrl_models_path = "build/nrl/ballistic_nrl.so";   // or BALLISTIC_NRL_MODELS, or <data dir>/nrl/
InitResult r = ballistic::initialize(init);            // r.nrl_models_loaded, r.nrl_models_path

namespace nrl = ballistic::atmosphere::nrl;
nrl::load(path);                 // ConfigError: missing file, not a library, wrong interface version
bool ok = nrl::try_autoload();   // BALLISTIC_NRL_MODELS, then nrl::default_install_path()
nrl::Status st = nrl::status();  // loaded, path, models ("NRLMSIS 2.1, HWM14"), problem
nrl::require("my caller");       // ConfigError naming the caller and how to install
```

Lookup order: an explicit `InitConfig::nrl_models_path` (strict: a bad file
makes `initialize` fail), else the `BALLISTIC_NRL_MODELS` environment
variable, else `<data directory>/nrl/ballistic_nrl.<so|dylib|dll>`
(`nrl::default_install_path()`, where Python's `install_nrl_models` puts
it). A process uses one copy: loading a second path throws `ConfigError`.

---

### Epoch and Time

#### `ballistic::Epoch`

Time representation with multiple time scale conversions.

```cpp
#include <ballistic/time/epoch.hpp>

// Create from UTC
Epoch t = Epoch::from_utc(2024, 1, 15, 12, 0, 0);

// Other named scales: from_tt / from_tai / from_gps / from_tdb (calendar),
// from_gps_week(week, seconds_of_week), from_tt_split(tt_day, tt_seconds)
Epoch tg = Epoch::from_gps(2024, 1, 15, 12, 0, 0);

// Create from Julian Date (the time scale is required)
Epoch t2 = Epoch::from_jd(2460325.5, TimeScale::TT);

// Create from MJD
Epoch t3 = Epoch::from_mjd(60325.0, TimeScale::UTC);

// Create from Unix timestamp
Epoch t4 = Epoch::from_unix(1705320000.0);

// J2000.0 epoch
Epoch j2000 = Epoch::j2000();

// Time scale conversions
double utc_jd = t.utc_jd();
double tt_jd = t.tt_jd();
double tai_jd = t.tai_jd();
double tdb_jd = t.tdb_jd();
double ut1_jd = t.ut1_jd();  // Requires EOP data for accuracy
double gps_jd = t.gps_jd();

// Arithmetic (ALL time deltas are in SECONDS)
Epoch t5 = t + 3600.0;  // Add 3600 seconds (1 hour)
Epoch t6 = t - 60.0;    // Subtract 60 seconds (1 minute)
double dt = t5 - t;     // Difference in seconds (returns 3600.0)

// Comparison
bool earlier = t < t5;
bool same = t == t;

// Calendar conversion (UTC by default; to_calendar(TimeScale::TT) etc.)
auto [year, month, day, hour, min, sec] = t.to_calendar();

// Epoch stores TT as an integer day + seconds, so labels and differences are
// exact to ~1e-11 s; tt_jd() / jd(scale) are doubles for model evaluation
// (do not subtract them).

// Set UT1-UTC offset manually (alternative to EOP file)
t.set_dut1(-0.15);  // seconds
```

#### `TimeScale` Enumeration

| Value | Description |
|-------|-------------|
| `UTC` | Coordinated Universal Time |
| `TT` | Terrestrial Time |
| `TAI` | International Atomic Time |
| `TDB` | Barycentric Dynamical Time |
| `UT1` | Universal Time (Earth rotation based) |
| `GPS` | GPS Time |

---

### Trajectory Propagation

#### `ballistic::Trajectory`

Primary propagator class for orbital mechanics.

```cpp
#include <ballistic/trajectory.hpp>

// Create with default config (measured-best general settings; add SRP + drag
// for the recommended setup: 0.239 m Sentinel 1-h RMSE, SENTINEL-JAN26-SAMPLED)
Epoch t0 = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
Vector3d pos(6778137.0, 0.0, 0.0);  // ECEF, meters
Vector3d vel(0.0, 7175.0, 0.0);  // Earth-fixed (ITRF): circular 7669 m/s minus Earth rotation 494 m/s

// The start names its frame and is checked (units, below ground, suborbital,
// inertial-vs-Earth-fixed mix-up). Also InitialState::from_gcrs / from_j2000 /
// from_keplerian, and sp3::initial_state(file, sat, epoch).
Trajectory traj(InitialState::from_itrf(t0, pos, vel));
// (Trajectory(t0, pos_itrf, vel_itrf[, config]) is also accepted; it checks units only.)

// Step-based propagation
while (traj.epoch() - t0 < 3600.0) {
    StepResult result = traj.step();
    // result.position, result.velocity, result.acceleration (ECEF)
}

// Direct propagation to target time (forward or backward; keeps no history)
PropagationResult result = traj.propagate(t0 + 3600.0);  // 1 hour forward
PropagationResult back = traj.propagate(t0 - 60.0);      // 1 minute backward

// Forward with history and events located inside the step (events.hpp:
// altitude_crossing, ground_impact, perigee, apogee, at_epoch); the ground
// impact is armed by default. The way to fly a reentry.
PropagateUntilResult r = traj.propagate_until(t0 + 7200.0,
                                              {events::altitude_crossing(80e3)});
// r.epoch, r.state_ecef, r.stopped_by (nullopt = reached the target), r.log

// Access history
for (size_t i = 0; i < traj.history_size(); ++i) {
    const HistoryEntry& entry = traj.history_at(i);
}

// Backward propagation
StepResult back = traj.step_backward();

// Reset to new initial conditions
traj.reset(new_epoch, new_pos, new_vel);

// Symplectic integrator check
bool symp = traj.is_symplectic();
double drift = traj.energy_drift();  // For symplectic
```

**Restarting from a history entry.** `Trajectory(InitialState::from_itrf(e.epoch, e.position,
e.velocity))` starts from the same inertial state as the original run had at that entry (~1e-9 m;
the GCRS↔ITRF conversion depends on the epoch alone). To follow the same path it must also take
the same steps: set `config.rk45.initial_step_size` to the `StepResult::next_step_size` returned
with that entry, and the runs agree to ~1e-6 m. With the default first step the step sequence
differs, and so does the RK45 truncation error: ~1 mm after an hour in LEO with the default
tolerances.

**Thread safety.** Separate `Trajectory` objects (and estimator calls) may run on separate
threads at the same time, also when they share one `DragEnvironment` or `SrpConfig`; one
`Trajectory` must not be used from two threads at once. The embedded gravity model and
ephemeris load themselves safely on first use from any thread. Loading user files
(`initialize`, `load_*`, `load_celestrak_csv`, EOP files) must not run while other threads propagate.

#### `TrajectoryConfig` Structure

```cpp
TrajectoryConfig config;
config.gravity_degree = 70;           // EGM2008 degree (70 sufficient for LEO)
config.enable_third_body = true;      // Sun/Moon/Jupiter perturbations
config.enable_relativistic = true;    // Schwarzschild + Lense-Thirring
config.enable_solid_tides = true;     // IERS 2010 solid Earth tides (ON by default; -31% RMSE on LEO)
config.itrf = coords::ITRFRealization::ITRF2020;     // Use ITRF2014 for Sentinel < March 2025
config.integrator = IntegratorType::RK45;            // RK45, YOSHIDA6, or YOSHIDA8
config.rk45.initial_step_size = 60.0;                // Initial RK45 step [s]; ~8 for glides
config.rk45.max_step_size = 3600.0;                  // Max RK45 step [s]; ~8 for glides
config.symplectic.step_size = 30.0;                  // For symplectic integrators
config.symplectic.nc_strategy = propagators::NonConservativeStrategy::SLIMPLECTIC;

// Optional forces follow one idiom — presence is the switch.

// SRP (radiation): assigning an SrpConfig turns SRP + albedo + Earth IR on.
// Defaults OFF (nullopt) only because it needs the satellite's mass/area —
// turn it ON for real satellites, POE/SP3-referenced or not
// (SENTINEL-JAN26-SAMPLED: 0.218 m with SRP and no drag, 0.540 m with neither).
config.srp = SrpConfig{
    .mass_kg = 2200.0,   // satellite mass [kg] (SRP/albedo/IR only)
    .area_m2 = 38.4,     // effective radiation area [m^2] (S1A, fitted — satellite_properties)
    .cr = 1.0,           // radiation pressure coefficient Cr (1.0 = perfect absorber)
};

// Drag is opt-in via DragEnvironment. Set drag_env to turn drag on; null =
// drag off. Turn it ON for real LEO satellites (a POE/SP3 start state does
// not already include it: −11 to −25% RMSE at 700–815 km over 2025,
// SENTINEL-YEAR-2025; −81% at 370 km, CHAMP-370KM).
// One-call builder: NRLMSIS 2.1 + HWM14 evaluated live at the satellite's
// position on every force evaluation (no table). Only t0 and the setup are
// used (pos/vel/duration/config are accepted for signature compatibility).
// With no .weather it uses the MEASURED space weather loaded into
// SpaceWeatherData (forecast days past its end, with a warning) and throws if
// none is loaded:
config.drag_env = atmosphere::make_drag_environment(
    t0, pos, vel, arc_duration_s, config,
    {.beta = atmosphere::BetaProfile1D::from_mass_cd_area(2200.0, 2.2, 6.35)});  // S1 frontal
// Or build a table via a generator and wrap with share(...):
config.drag_env = atmosphere::share(
    atmosphere::from_models_along(sample_epochs, sample_positions_ecef,
                                   atmosphere::SpaceWeatherHistory::from_constant(
                                       weather,
                                       atmosphere::TimeWindow{t0, t0 + arc_duration_s}),
                                   atmosphere::BetaProfile1D::constant(157.5)));  // S1: include/ballistic/satellites.hpp

// Optional: empirical acceleration table from estimation
config.empirical_accel = result.empirical_accel;     // shared_ptr<EmpiricalAcceleration>
config.ea_reference_epoch = std::nullopt;            // Override propagation epoch for EA time-key

// Diagnostics
config.report_forces = false;            // Compute force breakdown in step/propagation results
config.include_disabled_forces = false;  // Also compute forces that are turned off (analysis only)
```

| Field | Default | Description |
|-------|---------|-------------|
| `gravity_degree` | 70 | EGM2008 spherical-harmonic degree (70 for LEO; 4-8 for MEO/GEO/HEO) |
| `enable_third_body` | true | Sun + Moon + Jupiter perturbations |
| `enable_relativistic` | true | Schwarzschild + Lense-Thirring corrections |
| `enable_solid_tides` | true | IERS 2010 tides: solid Earth Step 1 (deg 2-3) + Step 2 frequency corrections (Tables 6.5a-c), and ocean tides when loaded |
| `ocean_tide_degree` | 50 | FES2004 ocean-tide degree (0 = off); active once `gravity::OceanTideModel` is loaded (`ballistic.load()` / `load_ocean_tides(path)`). Degree 30 is ~22% faster for ~1-2 cm more 1-h error |
| `itrf` | `ITRF2020` | ITRF realization for ECEF/ITRF (propagation is always GCRS) |
| `integrator` | `RK45` | `RK45` (best 1-hour) / `YOSHIDA6` / `YOSHIDA8` (best long-term) |
| `rk45.initial_step_size` | 60.0 | Initial RK45 step [s] |
| `rk45.max_step_size` | 3600.0 | Max RK45 step [s] |
| `symplectic.step_size` | 30.0 | Fixed step for symplectic integrators [s] |
| `symplectic.nc_strategy` | `SLIMPLECTIC` | `NONE` / `SPLITTING` / `SLIMPLECTIC` for non-conservative forces |
| `symplectic.adaptive` | false | Adaptive symplectic step (`min_step_size` 1.0, `max_step_size` 120.0, `adaptation_alpha` nullopt = 1/order, `symmetry_tolerance` 0.1) |
| `srp` | `nullopt` | Optional `SrpConfig{mass_kg, area_m2, cr, albedo, earth_ir}` — presence turns the radiation force (SRP + albedo + Earth IR) on. Defaults OFF only because it needs the satellite's mass/area; enable for real satellites, POE-referenced or not (`SENTINEL-JAN26-SAMPLED`: 0.218 m with SRP, 0.540 m with neither SRP nor drag; Python: `ballistic.recommended_config`). |
| `drag_env` | `nullptr` | Optional `shared_ptr<const atmosphere::DragEnvironment>` — the **sole drag-config field**. Build via `atmosphere::make_drag_environment` (one call) or the `from_models_*` generators wrapped in `share(...)` (see DragEnvironment section). Null = drag off. |
| `empirical_accel` | nullptr | Optional `shared_ptr<EmpiricalAcceleration>` from EA estimation |
| `ea_reference_epoch` | nullopt | Optional `std::optional<Epoch>` overriding the propagation epoch for EA time-key lookup (requires `empirical_accel`) |
| `report_forces` | false | Compute force breakdown in step/propagation results |
| `include_disabled_forces` | false | Also compute disabled forces (analysis only) |

#### `SrpConfig` Structure

| Field | Default | Description |
|-------|---------|-------------|
| `mass_kg` | none (NaN) | Satellite mass [kg]; used by SRP/albedo/IR only (drag mass enters via the DragEnvironment's beta). Required: an unset value raises `ConfigError` when the `Trajectory` is constructed (Python: required keyword) |
| `area_m2` | none (NaN) | Sun-facing cross-section [m²]; required, as `mass_kg` |
| `cr` | 1.3 | Radiation pressure coefficient Cr (1.0 = perfect absorber; a = P·(A/m)·Cr·ŝ) |
| `albedo` | true | Earth-albedo component |
| `earth_ir` | true | Earth-infrared component |

#### `IntegratorType` Enumeration

| Value | Description | Best For |
|-------|-------------|----------|
| `RK45` | Dormand-Prince adaptive (default) | Short-term accuracy (1-hour) |
| `YOSHIDA6` | Yoshida 6th order symplectic | Long-term stability (6+ hours) |
| `YOSHIDA8` | Yoshida 8th order symplectic | Maximum symplectic accuracy |

---

### State transition matrix and covariance

#### `ballistic::StateTransition` (`state_transition.hpp`)

A companion to a propagated `Trajectory`: the state transition matrix
Φ(t, t0), which maps a small change of the state at the first recorded epoch
t0 into the change at t (δx(t) = Φ δx(t0)), and the covariance P(t) = Φ P_0 Φᵀ (P_0: the initial covariance).
State order [r (m); v (m/s)], frame GCRS. Φ is integrated along the recorded
history (dΦ/dt = A(t) Φ, A = ∂[v; a]/∂[r; v] from
`ForceModel::acceleration_with_jacobian`) with classical Runge-Kutta 4, so
propagate with `step()` or `propagate_until()` first (`propagate()` records
no history). The trajectory is only read: its propagation is unchanged.

```cpp
#include <ballistic/state_transition.hpp>

Trajectory traj(InitialState::from_itrf(t0, r, v), cfg);
(void)traj.propagate_until(t0 + 5400.0);

StateTransition st(traj, {.fd_dx = 1.0, .fd_dv = 1e-3, .max_substep_s = 10.0});
Eigen::Matrix<double, 6, 6> phi = st.stm(t0 + 3600.0);
Eigen::Matrix<double, 6, 6> P   = st.covariance(t0 + 3600.0, p0);   // p0: initial covariance, 6x6 symmetric
```

| Member | Meaning |
|--------|---------|
| `StmOptions::fd_dx` / `fd_dv` | Finite-difference offsets [m], [m/s] for the force terms without an analytic Jacobian (drag's position columns; the velocity columns only for a velocity-dependent drag coefficient) |
| `StmOptions::max_substep_s` | Largest RK4 substep for Φ [s] (default 10) |
| `stm(epoch)` | Φ(epoch, epoch0()); `DataCoverageError` outside [epoch0(), epoch_end()] |
| `covariance(epoch, p0)` | Φ P_0 Φᵀ; `InputPlausibilityError` for a non-symmetric or non-finite initial covariance |
| `epoch0()` / `epoch_end()` / `node_count()` | The recorded span and the number of recorded steps Φ is stored at |

Construction raises `ConfigError` when the trajectory has fewer than two
recorded states. The object reads the trajectory's force model, so the
`Trajectory` must outlive it; steps taken later are not covered (build a new
one). Caveats: radiation pressure uses a hard cylindrical shadow, so an arc
crossing the shadow boundary gets a Φ that misses the jump in sunlight; drag's
position derivatives are finite differences; no consider parameters (β, Cr)
and no process noise. Measured against central differences of re-propagated
trajectories (`tests/unit/test_state_transition.cpp`): the analytic gravity
Hessian limits Φ to about 1e-5 to 3e-4 relative per column (5e-3 for a column
whose own size nearly vanishes after one orbit).

---

### Trajectory Interpolation

#### `ballistic::TrajectoryInterpolator`

High-accuracy interpolation using quintic Hermite polynomials (O(h^6) local error).

```cpp
#include <ballistic/trajectory_interpolator.hpp>

// Build trajectory history
Trajectory traj(InitialState::from_itrf(t0, pos, vel));
while (traj.epoch() - t0 < 3600.0) {
    traj.step();
}

// Create interpolator
TrajectoryInterpolator interp(traj);

// Query at any time
auto state = interp.get_state(t0 + 1800.0);
if (state) {
    std::cout << "Position: " << state->position.transpose() << "\n";
    std::cout << "Velocity: " << state->velocity.transpose() << "\n";
}

// Individual components
auto pos_opt = interp.position(t0 + 1800.0);
auto vel_opt = interp.velocity(t0 + 1800.0);
auto acc_opt = interp.acceleration(t0 + 1800.0);

// Batch queries
std::vector<Epoch> times = {t0 + 100.0, t0 + 500.0, t0 + 1000.0};
auto states = interp.get_states(std::span<const Epoch>(times));

// Altitude-based propagation
auto result = interp.propagate_to_altitude(
    300000.0,                      // Target altitude (meters)
    AltitudeCondition::Below,      // Above or Below
    PropagationDirection::Forward, // Forward or Backward
    86400.0);                      // max_duration_s (default 7 days); forward,
                                   // the history ends at the crossing

// Explicit history extension
interp.extend_to(t0 + 7200.0);           // Extend forward
interp.extend_backward_to(t0 - 300.0);   // Extend backward

// Configuration
interp.set_output_frame(InterpolationFrame::GCRS);
interp.set_itrf_realization(coords::ITRFRealization::ITRF2014);
```

#### `InterpolatorConfig` Structure

```cpp
InterpolatorConfig config;
config.output_frame = InterpolationFrame::ECEF;  // ECEF or GCRS
config.itrf = std::nullopt;                       // default: inherit the Trajectory's realization
config.cache_coefficients = true;                 // Enable caching
config.allow_propagation = true;                  // Auto-extend for out-of-range
config.max_propagation_time = 3600.0;            // Max auto-propagation (seconds)
```

---

### External Data Loading

#### EOP Data (finals2000A.data)

```cpp
#include <ballistic/time/eop_data.hpp>

auto& eop = ballistic::EOPData::instance();
bool loaded = eop.load("finals2000A.data");

if (loaded) {
    double mjd = 60325.0;                        // UTC MJD
    EopSample s = eop.lookup(mjd);               // every consumer goes through this
    // s.xp_arcsec, s.yp_arcsec, s.ut1_utc_s, s.lod_ms, s.dX_mas, s.dY_mas,
    // s.coverage, s.predicted, s.days_beyond
    double ut1_utc = eop.ut1_utc(mjd);           // UT1-UTC in seconds
    auto [x, y] = eop.polar_motion_rad(mjd);     // Polar motion (radians)
    auto [dX, dY] = eop.cip_offsets_rad(mjd);    // CIP offsets (radians)
}
```

Outside the table the edge value is held for up to 30 days with a warning,
then `DataCoverageError` is raised. With no table loaded every lookup raises
`DataCoverageError` unless zeros are accepted explicitly with
`EOPData::set_policy(EopPolicy::AssumeZero)` (Python
`ballistic.set_eop_policy('assume_zero')`; 3-10 m at LEO). Python
`ballistic.eop_coverage()` reports the span, the last measured day and the
skipped rows.

**File Format**: IERS finals2000A.data (fixed-width ASCII, 68+ columns)

**Download**: https://datacenter.iers.org/data/9/finals2000A.data

#### JPL Ephemeris (de421.bsp, de440.bsp)

```cpp
#include <ballistic/ephemeris/spk_ephemeris.hpp>

auto& mgr = ballistic::ephemeris::SPKEphemerisManager::instance();

// Load from file (auto-detects DE version)
bool loaded = mgr.load("de421.bsp");

// Or load embedded DE440
bool embedded = mgr.load_embedded();

if (mgr.is_loaded()) {
    const auto& eph = mgr.ephemeris();
    // Used automatically by third-body force model
}
```

**Supported Files**:
- `de421.bsp` - ESA operational standard (1900-2050)
- `de440.bsp` - JPL (1550-2650)
- `de441.bsp` - JPL long-span (about −13200 to +17191)

**Download**: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

#### Space Weather (SW-All.csv)

```cpp
#include <ballistic/atmosphere/space_weather.hpp>

auto& sw = ballistic::atmosphere::SpaceWeatherData::instance();
bool loaded = sw.load_celestrak_csv("SW-All.csv");

if (loaded) {
    Epoch t = Epoch::from_utc(2024, 1, 15, 12, 0, 0);
    // One snapshot; throws outside the file. Optional arguments:
    // get_conditions(t, allow_predicted = true, bool* used_predicted = nullptr)
    WeatherConditions wx = sw.get_conditions(t);
    std::cout << "F10.7: " << wx.f107_daily << " SFU\n";
    std::cout << "Ap: " << wx.ap_daily << "\n";

    // The time-varying measured history drag uses
    auto hist = sw.history(atmosphere::TimeWindow{t, t + 86400.0});
}
```

**Download**: https://celestrak.org/SpaceData/SW-All.csv

#### Leap Seconds (leap-seconds.list)

```cpp
#include <ballistic/init.hpp>

// Load via initialize
ballistic::InitConfig config;
config.leap_seconds_path = "leap-seconds.list";
ballistic::initialize(config);

// Or load directly
auto [success, message] = ballistic::load_leap_seconds("leap-seconds.list", true);
if (success) {
    auto info = ballistic::get_leap_second_info();
    std::cout << "Entries: " << info.entry_count << "\n";
    std::cout << "Current TAI-UTC: " << info.current_tai_utc << " s\n";
    std::cout << "Loaded from file: " << info.loaded_from_file << "\n";
}
```

**File Format**: NTP leap-seconds.list

**Download**: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list

---

### Coordinate Transformations

```cpp
#include <ballistic/coordinates/geodetic.hpp>
#include <ballistic/coordinates/iau_transforms.hpp>

// Geodetic to ECEF (WGS84)
double lat = 45.0 * constants::DEG_TO_RAD;  // radians
double lon = -122.0 * constants::DEG_TO_RAD;
double alt = 1000.0;  // meters
Vector3d ecef = ballistic::coords::geodetic_to_ecef(lat, lon, alt);

// ECEF to geodetic (exact closed form, Vermeille 2011)
auto geo = ballistic::coords::ecef_to_geodetic(ecef);
// geo.latitude (rad), geo.longitude (rad), geo.altitude (m)

// Geodetic altitude from ECEF position (= ecef_to_geodetic(ecef).altitude)
double altitude = ballistic::coords::ecef_to_altitude(ecef);

StateVector state_ecef{pos, vel};

// IAU 2006/2000A transformations (ITRF <-> GCRS)
StateVector gcrs = ballistic::coords::itrf_to_gcrs_full(
    state_ecef, epoch, ITRFRealization::ITRF2020);
StateVector itrf = ballistic::coords::gcrs_to_itrf_full(
    gcrs, epoch, ITRFRealization::ITRF2020);
Matrix3d R = ballistic::coords::gcrs_to_itrf_rotation(epoch);  // free vectors
// (itrf_to_gcrs_rotation is its transpose; neither applies an ITRF
// realization change, which affects positions only)
```

The state transforms include the velocity of the CIP drift
(`coords::cip_drift_velocity`). Propagation is always in GCRS; a true-of-date
frame is not provided (J2000 and TEME start states are accepted through
`InitialState::from_j2000` / `from_teme`).

---

### Empirical Acceleration Estimation

Reconstruct unmodeled forces (drag mismodeling, maneuvers, atmospheric
density errors) by fitting an `EmpiricalAcceleration` table to
observations (position, velocity, or acceleration). The estimated EA
can then be added to `TrajectoryConfig::empirical_accel` for
high-fidelity re-propagation. The user guide is
[`empirical_acceleration_estimation.md`](empirical_acceleration_estimation.md).

#### Public Estimator API

`estimate_ea_bspline(...)` (POS / VEL / ACC) in
`<ballistic/calibration/ea_estimator.hpp>`. The observation
**modality is carried by the sample type** — wrap the data in
`PositionSamples` / `VelocitySamples` / `AccelSamples` — so modality
mix-ups are compile errors. All take **ECEF observations**; the estimator
converts internally as needed (epochs are carried inside each sample). EA
is returned in GCRS. For sensor fusion or radar, use `estimate_ea_smoother`
(see Multi-Sensor EA Estimation).

```cpp
#include <ballistic/calibration/ea_estimator.hpp>
using namespace ballistic::calibration;

EABSplineConfig cfg;                  // see fields below
cfg.knot_spacing_s = 2.0;
cfg.regularization = ea::IRGN{};      // default; or ea::CV{}, ea::GCV{}, ea::FixedLambda{}
cfg.gn = GNRefine{};                  // the default (ON); std::nullopt skips it

TrajectoryConfig fm;                  // force model used for re-propagation
fm.gravity_degree = 70;

// B-spline IRGN (noise-tolerant, general-purpose) — modality by sample type
auto pos_result = estimate_ea_bspline(epoch0, pos_ecef, vel_ecef, PositionSamples{pos_data_ecef}, fm, cfg);
auto vel_result = estimate_ea_bspline(epoch0, pos_ecef, vel_ecef, VelocitySamples{vel_data_ecef}, fm, cfg);
auto acc_result = estimate_ea_bspline(epoch0, pos_ecef, vel_ecef, AccelSamples{acc_data_ecef},    fm, cfg);

// GN refinement runs by default. For sensor fusion or radar → EA, use
// estimate_ea_smoother (Multi-Sensor EA Estimation below).

// Apply the estimated EA to a re-propagation config. Gate pipelines on
// succeeded() (ran without error; a legitimately-zero EA still counts);
// gate consumers that need a usable non-zero EA on has_correction().
if (pos_result.has_correction()) {
    TrajectoryConfig recon = fm;
    recon.empirical_accel = pos_result.empirical_accel;
    Trajectory traj(InitialState::from_itrf(epoch0, pos_ecef, vel_ecef), recon);
    // ... propagate as usual; EA is now baked in
}
```

#### `EABSplineConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `regularization` | `ea::IRGN{}` | One of: `ea::IRGN`, `ea::FixedLambda`, `ea::GCV`, `ea::CV` |
| `noise_sigma` | 0.0 | Observation σ (units match obs type); `>0` enables early stopping at 1.5σ |
| `knot_spacing_s` | 2.0 | Interior knot spacing [s] |
| `degree` | 5 | B-spline degree: 5=quintic (default), 3=cubic |
| `max_iterations` | 50 | Max IRGN outer iterations (early stops at 5 stalls) |
| `max_duration_s` | 3600.0 | Maximum arc duration [s]; clamped to the observation arc end + 10 s, so the default is safe on short arcs |
| `reg_deriv_order` | nullopt (auto) | Tikhonov derivative order (auto: 3 POS, 1 VEL/ACC) |
| `gn` | `GNRefine{}` (**ON**) | GN refinement, on by default (it improves POS and VEL fits on the synthetic test cases; see [`empirical_acceleration_estimation.md`](empirical_acceleration_estimation.md)). Set `nullopt` / Python `None` to skip. ACC ignores it (circular; `gn_status` reports `SkippedAccelerationCircular`). |
| `brent_line_search` | true | Brent's α search (5-anchor bracket); `false` reverts to 24-α brute force |
| `verbose` | false | Per-iteration estimator + GN diagnostics to stdout |

#### Regularization Variants (`cfg.regularization = ...`)

```cpp
// Iterative regularized GN with annealing schedule (default)
cfg.regularization = ea::IRGN{1e-4, 0.5, 1e-12};       // {lambda_0, decay, lambda_min}

// hv-block CV grid search (Racine 2000); best automatic selection for noisy data.
// Fields: hv_n_blocks = 18, hv_test_blocks = {} (auto), hv_gap_blocks = 1,
// mode = ea::CV::Mode::IRGN (or FixedLambda: ~5x faster, ACC only),
// grid = ea::CV::Grid::Full (495 candidates; Coarse: 12, ~40x faster).
// A choice on a grid edge (knot spacing or final λ) is reported as a ConfigHint.
// A layout that holds out no block (hv_n_blocks < 4 with the automatic
// pattern) throws ConfigError. cfg.gn (incl. reg_lambda / reg_deriv_order)
// applies to every candidate's GN run.
cfg.regularization = ea::CV{};

// Per-iteration GCV; no σ input needed. λ is searched within [lam_min, lam_max]
// (0 < lam_min < lam_max, else ConfigError)
cfg.regularization = ea::GCV{1e-12, 1e-1};             // {lam_min, lam_max}

// Constant λ; useful only for ACC. Measured worse than IRGN for POS/VEL
cfg.regularization = ea::FixedLambda{1e-4};
```

#### `GNRefine`

| Field | Default | Description |
|-------|---------|-------------|
| `max_iterations` | 30 | GN iterations |
| `sample_stride` | 20 | Observation subsample stride |
| `ea_subsample` | 1 | EA breakpoint subsample (1 = optimize all knots) |
| `fd_delta` | 1e-2 | FD perturbation [m/s²] |
| `rk45_max_step` | monostate (inherit) | `std::variant<std::monostate, GNRefine::MatchKnotSpacing, double>`: monostate = inherit `force_model.rk45` (fully adaptive), `MatchKnotSpacing{}` = auto-match the estimator's knot spacing, double = force this step [s]. Python: `None` \| `'match_knots'` \| float. |
| `use_stm` | **true** | STM (variational-equation) Jacobian path; ~2× speedup over FD |
| `stm_dt_s` | 0.5 | Heun step size for sensitivity ODE |
| `stm_fd_dx` | 1.0 | FD perturb for ∂a/∂r [m] |
| `stm_fd_dv` | 1e-3 | FD perturb for ∂a/∂v [m/s] |
| `stm_verify` | false | Iter 0: also build J via FD and log diff |
| `stm_verbose` | false | Per-call STM diagnostic prints |
| `reg_lambda` | 1.0 | Tikhonov weight of the refined EA, the same regularized objective as the B-spline stage (0 = unregularized polish, which corrupts the pointwise EA on noisy data) |
| `reg_deriv_order` | nullopt (auto) | Derivative order of that penalty (auto: 3 POS, 1 VEL) |

#### `EmpiricalAcceleration` Value Type

Variant-backed value type with three backends: `Zero`, `Table`, `BSpline`.

```cpp
#include <ballistic/forces/empirical_acceleration.hpp>
using ballistic::forces::EmpiricalAcceleration;
using ballistic::forces::EmpiricalAccelEntry;
using ballistic::forces::EmpiricalAccelFrame;
using ballistic::forces::EAInterp;
using ballistic::forces::EARepresents;
using ballistic::forces::BSplineCoefficients;

// Factories
auto zero_ea = EmpiricalAcceleration::zero();
auto table_ea = EmpiricalAcceleration::from_entries(
    entries, EmpiricalAccelFrame::GCRS, EAInterp::Cubic);
auto hermite_ea = EmpiricalAcceleration::from_entries_hermite(
    entries, derivatives, EmpiricalAccelFrame::GCRS);
auto bspline_ea = EmpiricalAcceleration::from_bspline(
    coefficients, EARepresents::Acceleration, EmpiricalAccelFrame::GCRS);

// Query
Vector3d a = bspline_ea.accel_at(t_seconds);   // in stored frame
auto frame = bspline_ea.frame();
bool empty = bspline_ea.is_zero();
```

#### EA Decomposition

Split an EA into a velocity-aligned drag component (β = m/(Cd·A)) and a
perpendicular residual (maneuver forces). Two public entry points:

```cpp
#include <ballistic/calibration/ea_decomposition.hpp>

// Single-vector decomposition: ea + velocity + density → drag/residual + beta
EADecomposition d = decompose_ea(ea_vector, velocity, density_kg_m3);

// DragEnvironment-driven EA → β calibration ("table in → refined table
// out"). Returns a new
// DragEnvironment with density + wind passed through and the β column
// replaced. `trajectory` must be the self-consistent history the EA was
// applied to — EAResult::collocation_history (estimate_ea_bspline) is
// exactly such a trajectory, also after GN (it is then re-propagated with
// the refined EA).
atmosphere::DragEnvironment refined = refine_beta_from_ea(
    prior_env, *result.empirical_accel, result.collocation_history,
    epoch0, BetaDecompositionConfig{});
// refined.beta() is now the calibrated BetaProfile1D.
// When no β can be fitted (zero EA, empty trajectory, no point where the EA
// opposes the motion) the prior comes back unchanged with a ModelSubstitution
// diagnostic; pass a BetaResult* as the last argument to read the fit itself
// (steps_solved, error_message).
```

> Use `refine_beta_from_ea` for EA → β and `decompose_ea` for a single
> vector. `EAResult` lives in `ea_result.hpp`.

#### Standalone GN Refinement

`refine_ea_gn` is the GN driver the estimator uses, also exposed standalone
for an EA built from another source. It takes the force model plus the
`GNRefine` settings and returns the refined `shared_ptr<EmpiricalAcceleration>`
(signature: `refine_ea_gn(epoch0, pos_ecef, vel_ecef, position_data,
initial_ea, force_model, gn = {}, max_duration_s = 3600.0)`).

```cpp
#include <ballistic/calibration/ea_estimator.hpp>

GNRefine gn;                           // public GN settings
gn.max_iterations = 30;
gn.use_stm = true;

auto refined = refine_ea_gn(epoch0, pos_ecef, vel_ecef, position_data,
                            initial_ea, force_model, gn,
                            /*max_duration_s=*/3600.0);
```

Trade-offs: the STM Jacobian (`use_stm = true`) is about twice as fast as
finite differences but interpolates the tabulated EA linearly; for a one-off
position fit where accuracy matters most, `use_stm = false` (finite
differences with cubic interpolation) gave about 30% lower error on the
position test case.

---

### Multi-Sensor EA Estimation

Fuses heterogeneous sensors (radar / optics / position fixes / accelerometer)
into a single EA via `estimate_ea_smoother` (a Gauss-Markov RTS smoother). Detailed
reference: [`docs/multi_sensor_estimator.md`](multi_sensor_estimator.md).

```cpp
#include <ballistic/calibration/multisensor.hpp>   // sensor types, MultiSensorScenario, SensorMask
#include <ballistic/calibration/ea_smoother.hpp>   // estimate_ea_smoother
using namespace ballistic::calibration;
```

#### Sensor types

| Type | Channels per sample | Notes |
|---|---|---|
| `RadarObs` | [range, az, el, range_rate] (4) | Station `position_ecef`, `lat_rad`/`lon_rad`, `ecef_to_enu` (az/el are in the station's ENU frame); σ defaults `sigma_range` 2 m, `sigma_angle` 0.02°, `sigma_rate` 0.02 m/s. Has `.predict(state)` forward model. |
| `OpticsObs` | [az, el] (2) | Angular bearings; `sigma_angle` 50 µrad. Has `.predict(state)`. |
| `PositionFixObs` | ECEF position (3) | Single `sigma_pos` (3 m); multiple fixes per object. |
| `AccelObs` | ECEF specific force (3) | `sigma_accel` 0.01 m/s²; bias jointly estimated, `sigma_bias_prior` (1e-3) controls its regularization. Rotate body-frame data to ECEF before filling `samples`. |

**Declarative error-model fields** (all sensors unless noted; defaults are
exact no-ops):

| Field | On | Meaning |
|---|---|---|
| `known_bias` | all | Subtracted from every observation pre-estimation (native units per sensor). Gated exactly neutralizing. |
| `sigma_scale` | all | Optional per-sample σ multiplier aligned with `samples` (empty = scalar σ). |
| `clock` | radar, optics | `SensorClock::None` / `Shared` (radar default) / `Own`. Master switch `EASmootherConfig::estimate_clock_bias`. |
| `outlier_prob`, `outlier_scale` | accel | Heavy-tail mixture declaration (1−p)·N(0,σ²)+p·N(0,(kσ)²); p>0 enables a Masreliez robust update. Default 0 = off. |

#### Scenario + mask

```cpp
MultiSensorScenario scenario;
scenario.epoch0     = csv.front().epoch;
scenario.pos0_ecef  = known_pos0;   // exactly known
scenario.vel0_ecef  = known_vel0;   // exactly known
scenario.arc_duration_s = arc_s;
scenario.add_radar(std::move(radar));
scenario.add_position_fix(std::move(fix));

auto mask = SensorMask::all_on(scenario).with_optics(false);  // disable optics
```

#### Configuration

`estimate_ea_smoother(scenario, mask, fm_cfg, cfg = {}) -> EASmootherResult`
takes an `EASmootherConfig`, minimal by design (an automatic prior chosen
from the innovation statistics and a measured range-rate budget). Fields:
`sigma_a` / `tau_s` (nullopt = chosen by NIS; pin them, e.g. ≈1e-6 m/s² /
600 s, for orbital arcs), `rate_budget` (`RateBudget::Auto` / `Off` /
`Manual` with `sigma_rate_manual`), `max_dt_s` 0.05, `jac_refresh` 16,
`output_dt_s` 0.5, `passes` 1, `estimate_clock_bias` true,
`clock_bias_prior_s` 0.02, `maneuver_adaptive_q` false
(`maneuver_nis_gate` 9, `maneuver_q_boost` 25), `regime_aware_prior` false
(anchors the prior grid on a data-derived scale from a frozen-EA prepass;
opt-in), `custom_sensors`, `verbose`. When the automatic prior lands on an
edge of its grid, `EASmootherResult::warnings` says so: a prompt to
confirm the regime, not an error. See
[`multi_sensor_estimator.md`](multi_sensor_estimator.md).

#### Result types

```cpp
// ResultCore (ea_result_core.hpp) is the shared base of EAResult and the
// smoother result; it carries the common fields + the two success predicates.
struct ResultCore {
    shared_ptr<EmpiricalAcceleration> empirical_accel;  // GCRS frame
    int iterations_used;
    string error_message;
    bool succeeded() const;       // ran without error and produced an EA
                                  // (a legitimately-zero EA is still success)
    bool has_correction() const;  // succeeded() AND the EA is non-zero
};
```

`EASmootherResult` (extends `ResultCore`): `nis` (normalized innovation² /
dof; ~1 ideal), `recovered_bias` (accelerometer bias, ECEF), `sigma_a_used`,
`tau_used`, `sigma_rate_eff` (range-rate σ used; 0 = rate off),
`clock_biases`, `clock_bias_s`, `sigma_honored`, `ea_sigma` (posterior 1σ
per output sample), `prior_anchor`, `warnings`.

---

### Declarative Front Door + Information Bound

```cpp
#include <ballistic/calibration/ea_problem_spec.hpp>       // estimate_ea(spec)
#include <ballistic/calibration/ea_information_bound.hpp>  // information_bound
```

**`estimate_ea(const EAProblemSpec&) -> EASpecResult`** — observations +
declared error model in; dispatches typed samples → `estimate_ea_bspline`
(GN default ON) and `MultiSensorScenario` → `estimate_ea_smoother`.
`EAProblemSpec`: `epoch0`/`pos0_ecef`/`vel0_ecef` (ignored for scenarios),
`observations` (variant: `PositionSamples` | `VelocitySamples` |
`AccelSamples` | `MultiSensorScenario`), `force_model`, `noise_sigma`
(REQUIRED > 0 for typed samples), optional `bspline`/`smoother` expert
overrides. `EASpecResult` (extends `ResultCore`): `method_used`
(`EAMethod`), `rationale`, `nis`, `sigma_honored`, `audit_warnings`
(NIS ∉ [0.3, 3]; honored ≫ declared; rate budget ≫ sensor σ ⇒ time-tag
suspicion), `bspline_detail`/`smoother_detail`. Python: `ballistic.
estimate_ea`, `EAProblemSpec`, `EASpecResult`, `EAMethod`.

Typed-sample heavy tails: `EAProblemSpec::outlier_prob`
/ `outlier_scale` (default 0 = off) apply the data-level prefilter
**`reject_heavy_tails(samples, sigma, p, k)`** before fitting — rejection
against a ±0.25 s local median at the mixture p_out = 0.5 boundary (~3.3σ,
nearly insensitive to (p, k)). The C++ signature is
`reject_heavy_tails(samples, sigma, outlier_prob = 0.05, outlier_scale = 10.0,
size_t* n_rejected = nullptr, bool* out_of_envelope = nullptr)`; below about
4 Hz the ±0.25 s window holds too few samples, so the input is returned
unchanged with `out_of_envelope` set (and `estimate_ea` adds an audit
warning). POS/ACC only — on VEL the detrend's curvature bias clusters false
rejections where the jerk is largest, so VEL gets a warning, not a filter.
Rejection, not winsorization: clipping toward the local trend corrupts POS
data whose curvature over the window is comparable to σ. Python:
`ballistic.estimation.reject_heavy_tails(samples, sigma, outlier_prob=0.05,
outlier_scale=10.0)` → `(filtered, n_rejected, out_of_envelope)`.

**`information_bound(scenario, mask, fm_cfg, opts = {}) -> EAInformationBound`**
(C++ also has an `EAProblemSpec` overload) — the sensor-design instrument:
posterior (`posterior_bound_m`, with a first-difference prior λ = 1 plus an
amplitude ridge λ_amp = 1e-2 on an 8 s node grid —
`EAInformationBoundOptions{knot_spacing_s, lambda, lambda_amp}`; Python
spells `lambda` as `lambda_`) and
prior-free observable-subspace CRB (`crb_bound_m`) reprop bounds, evaluated
at the zero-EA nominal; also `n_params`, `n_obs_rows`, `error_message`,
`ok()`. Compare suites at one node spacing (finer grids raise the bound).
Accelerometers are deliberately excluded (their bias ambiguity would make
the bound optimistic). Python: `ballistic.estimation.information_bound(scenario, mask,
fm_cfg, opts)`.

**Smoother clock and audit fields** (`EASmootherResult`): `clock_biases`
(label → seconds per clock block: "shared", "radar0", …; scale/sign
calibration is not yet validated) and `sigma_honored` (per-channel
residual RMS at the smoothed solution — the declared-vs-honored audit,
report-only).

**Custom sensors** (C++ only): `EASmootherConfig::custom_sensors`
takes `detail::CustomSensor` `{times_s, update(ctx, z, layout, i) ->
SensorUpdate{r, H, R}}` — rides the same EKF/RTS path as the built-in
sensors. See [`multi_sensor_estimator.md`](multi_sensor_estimator.md) for
the Case-A worked example (mechanism transforms) and the Case-B warning
(normal-scoring additive noise is an illusion).

### Radar EA Estimation

Use `estimate_ea_smoother` for radar → EA; it consumes radar natively,
range rate included at a measured noise budget (`EASmootherConfig::rate_budget`).

---

### Configuration Types

#### `BallisticModel`

```cpp
// Ctor is (beta, mass, area) — beta [kg/m^2] is the primary drag parameter;
// mass/drag_area are optional metadata (SRP / diagnostic display).
BallisticModel model(
    157.5,   // beta = m / (Cd * A_frontal)  [kg/m^2]  (Sentinel-1; include/ballistic/satellites.hpp)
    2200.0,  // mass (kg)            — optional
    6.35     // drag cross-section (m^2, frontal) — optional
);
// Single-arg form: BallisticModel(beta). Default ctor: beta = 45.45.

double beta    = model.ballistic_coefficient();          // returns stored beta
double inv_b   = model.inverse_ballistic_coefficient();  // 1 / beta
```

#### `WeatherConditions`

```cpp
// Factory methods (the default constructor gives F10.7 150, Ap 4)
auto quiet = WeatherConditions::quiet();   // F10.7=70, Ap=4
auto active = WeatherConditions::active(); // F10.7=200, Ap=15
auto storm = WeatherConditions::storm();   // F10.7=250 (81-day mean 200), Ap=100

// Custom
WeatherConditions wx;
wx.f107_daily = 150.0;
wx.f107_average = 150.0;
wx.ap_daily = 15.0;
```

#### `SatelliteProperties`

Properties for radiation pressure calculations.

```cpp
SatelliteProperties sat;
sat.srp_area = 10.0;           // Cross-sectional area for SRP [m²]
sat.srp_reflectivity = 0.3;    // Optical reflectivity [0-1]
sat.albedo_reflectivity = 0.3; // Albedo reflectivity [0-1]
sat.ir_emissivity = 0.8;       // IR absorptivity/emissivity [0-1]

// Or via the ctor: SatelliteProperties(area, refl = 0.3, ir_emis = 0.8)
// (refl sets both srp_reflectivity and albedo_reflectivity).
SatelliteProperties sat2(38.4, 0.0);  // Sentinel-1A: fitted effective area 38.4 m², Cr = 1 + 0.0
```

---

### Result Types

#### `StepResult`

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `Epoch` | Time after step |
| `position` | `Vector3d` | Position in ECEF (meters) |
| `velocity` | `Vector3d` | Velocity in ECEF (m/s) |
| `acceleration` | `Vector3d` | Acceleration in ECEF (m/s^2) |
| `step_size_taken` | `double` | Actual step size used (seconds) |
| `next_step_size` | `double` | Suggested next step size |
| `history_index` | `size_t` | Index of the new state in the history |
| `forces` | `optional<ForceBreakdown>` | Force components (if enabled) |

#### `PropagationResult`

| Field | Type | Description |
|-------|------|-------------|
| `position` | `Vector3d` | Final position in ECEF (meters) |
| `velocity` | `Vector3d` | Final velocity in ECEF (m/s) |
| `acceleration` | `Vector3d` | Final acceleration in ECEF (m/s^2) |
| `forces` | `optional<ForceBreakdown>` | Force components (if enabled) |

#### `InterpolatedState`

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `Epoch` | Query time |
| `position` | `Vector3d` | Position in output frame |
| `velocity` | `Vector3d` | Velocity in output frame |
| `acceleration` | `Vector3d` | Acceleration in output frame |
| `was_propagated` | `bool` | True if history was extended |

#### `ForceBreakdown`

| Field | Type | Description |
|-------|------|-------------|
| `gravity` | `Vector3d` | Earth gravity (m/s^2) |
| `third_body` | `Vector3d` | Sun + Moon + Jupiter (m/s^2) |
| `drag` | `Vector3d` | Atmospheric drag (m/s^2) |
| `srp` | `Vector3d` | Solar radiation pressure (m/s^2) |
| `albedo` | `Vector3d` | Earth albedo (m/s^2) |
| `ir` | `Vector3d` | Earth infrared (m/s^2) |
| `relativistic` | `Vector3d` | Relativistic corrections (m/s^2) |
| `solid_tides` | `Vector3d` | Solid Earth tides (m/s^2) |
| `ocean_tides` | `Vector3d` | Ocean tides, when the FES2004 coefficients are loaded (m/s^2) |
| `empirical` | `Vector3d` | Empirical acceleration from EA estimation (m/s^2) |
| `total` | `Vector3d` | Sum of all enabled components |
| `radiation()` | method | Helper returning `srp + albedo + ir` |

#### `EAResult` (calibration)

Result of `estimate_ea_bspline`
(`<ballistic/calibration/ea_result.hpp>`). Inherits `ResultCore`
(shared with `EASmootherResult` and `EASpecResult`).

| Field / Method | Type | Description |
|-------|------|-------------|
| `empirical_accel` | `shared_ptr<EmpiricalAcceleration>` | Estimated EA in GCRS (may be null on failure) |
| `iterations_used` | `int` | Outer iteration count |
| `error_message` | `string` | Non-empty on failure |
| `succeeded()` | `bool` | Ran without error and produced an EA — a legitimately-zero EA (nothing to correct) is still success. Gate pipelines on this. |
| `has_correction()` | `bool` | `succeeded()` AND the EA is non-zero. Gate GN / β-refinement consumers (which need a usable non-zero EA) on this. |
| `collocation_history` | `vector<HistoryEntry>` | Trajectory that goes with the returned EA, at the spline's knot times (after GN: re-propagated with the refined EA) |
| `final_trajectory` / `final_interp` | shared_ptr | Propagated trajectory + interpolator from the final pass (may be null) |
| `gn_status` | `GNStatus` | `Disabled` / `Ran` / `SkippedAccelerationCircular` |
| `lambda_used` | `optional<double>` | λ at convergence |
| `sigma_sq_estimate` | `optional<double>` | Per-component noise variance estimate (GCV) |
| `obs_rmse` | `optional<double>` | Observation RMSE at convergence |

---

## Python API

The package has a short top level and a few modules for specialized work.
Start with the top level; reach for a module when you need what it holds.

| Module | Holds |
|--------|-------|
| `ballistic` | loading data, times, start states, `Trajectory` and its configuration, `recommended_config`, `make_drag_environment`, frame conversions, errors and warnings |
| [`ballistic.estimation`](#ballisticestimation-fitting-an-empirical-acceleration) | fitting an empirical acceleration (EA) to observations, and turning it into a ballistic coefficient |
| [`ballistic.drag`](#ballisticdrag-atmospheres-built-by-hand) | tabulated and measured atmospheres, table readers, saving an environment |
| [`ballistic.aero`](#ballisticaero-flow-regime-drag-coefficient) | drag coefficients that follow the flow, for reentry |
| [`ballistic.readers`](#ballisticreaders-orbit-products-and-velocity-checks) | ESA EOF orbit files, velocity-record checks |
| [`ballistic.tle`](#ballistictle-tles-and-sgp4) | two-line element sets and SGP4 |
| [`ballistic.sp3`](#ballisticsp3-sp3-orbit-files) | SP3 precise-orbit files |
| [`ballistic.events`](#ballisticevents-events-during-propagation) | events located during propagation (reentry) |
| `ballistic.errors`, `ballistic.constants`, `ballistic.satellite_properties` | error and warning classes, physical constants, the satellite registry |

A name that used to be top level and now lives in a module raises
`RemovedApiError` naming its new import; [`MIGRATION.md`](MIGRATION.md)
lists every one.

## Start here: the top-level `ballistic`

Everything a first session needs: `ballistic.load`, `ballistic.Epoch`,
`ballistic.State`, `ballistic.Trajectory`, `ballistic.TrajectoryConfig`,
`ballistic.recommended_config`, `ballistic.make_drag_environment`,
`ballistic.satellite_properties`, `ballistic.events` and the errors.

### Recommended setup (start here)

```python
import ballistic
ballistic.load()                                   # once per session
state = ballistic.State.from_itrf(epoch, pos_ecef, vel_ecef)   # Earth-fixed velocity
cfg = ballistic.recommended_config(state, "S1A")
traj = ballistic.Trajectory(state, cfg)
traj.propagate_for(3600.0)
```

#### `ballistic.load(directory=None, verbose=True, space_weather=True, ocean_tides=True, *, eop=True, leap_seconds=True, offline=False) -> InitResult`

Loads EOP + leap seconds and — by default — CelesTrak space weather and the
FES2004 ocean-tide coefficients (IERS Conventions Ch. 6 file, ~3.7 MB; not
embedded: third-party model). Each file argument is `True` (the copy in the
data directory, downloaded if missing and refreshed if stale unless
`offline`), `False` (skip) or a path (load that file; never downloads).
`offline=True` (or `BALLISTIC_OFFLINE=1`) never downloads. A file that
cannot be fetched or loaded gives a warning (not an error); drag then raises
until space weather is loaded, and propagation runs without ocean tides.
Returns the `InitResult` (staleness flags and `warnings`, each also issued
as a `DataCoverageWarning`).

#### `ballistic.recommended_config(state, satellite=None, *, mass_kg=None, srp_area_m2=None, drag_area_m2=None, cd=None, cr=None, duration_s=3600.0, allow_predicted_weather=True, reentry=None, reference_length_m=None, accuracy="best", horizon_s=None, beta_kg_m2=None, tle=None, drag=None, **overrides) -> TrajectoryConfig`

The best-accuracy configuration (`python/ballistic/recommended.py`): the
library defaults (already the measured-best general settings) plus

- radiation pressure (SRP + albedo + Earth IR) from the satellite's
  radiation area — registry entries hold EFFECTIVE areas fitted on 2025
  Sentinel data (Cr = 1 + `srp_reflectivity`); for your own area `cr`
  defaults to 1.3;
- drag (NRLMSIS 2.1 + HWM14 evaluated live at the satellite's position)
  driven by the MEASURED space weather, built by `make_drag_environment`;
  `duration_s` is unused (there is no arc to build). Past the last measured
  day CelesTrak's forecast is used with a warning
  (`allow_predicted_weather=False` stops there instead). Drag is left off
  above 1500 km. The two models are the separate NRL plugin
  (`install_nrl_models`): `drag=None` (default) turns drag on only when it is
  installed and otherwise leaves it off with one `ConfigHintWarning` stating
  the cost; `drag=True` requires it (`ConfigError` without it); `drag=False`
  leaves drag off without a word. A reentry always requires it.

`satellite` is a registry name (`ballistic.satellite_properties`) or a
`SatelliteProperties`; otherwise pass `mass_kg`, `srp_area_m2` and (below
1500 km) `drag_area_m2` (mean frontal area; `cd` default 2.2). The state's
ITRF realization, if it states one, becomes the config's. `overrides` go to
`TrajectoryConfig` (e.g. `itrf=...`). For a reentry pass
`reentry=ballistic.aero.AeroShape.SPHERE` (or `FLAT_PLATE` / `CYLINDER`) and
`reference_length_m`: the drag coefficient then follows the flow regime
(`FlowRegimeCd`). For a TLE start pass `accuracy="tle"` and `horizon_s`
(the TLE-grade preset; see [`ballistic.tle`](#ballistictle-tles-and-sgp4)). Raises `ValueError` for missing inputs
and an error if drag is needed but no space weather is loaded (never a
quiet-atmosphere substitute).
A registry satellite whose values are `legacy` (unsourced; see below) earns
one `ConfigHintWarning` naming the arguments that replace them.

#### `ballistic.make_drag_environment(t0, pos_ecef, vel_ecef, duration_s, config, beta, weather=None, ..., allow_predicted_weather=True) -> (DragEnvironment, IterationDiag)`

Turns drag on in one call (`recommended_config` calls it for you). The
environment evaluates NRLMSIS 2.1 density and HWM14 wind at the satellite's
position on every force evaluation, on the whole loaded space-weather file;
assign it with `config.replace(drag_env=env)`. `beta` is a
`ballistic.BetaProfile1D` (`constant`, `from_mass_cd_area`) or a
`ballistic.aero.AeroBeta` (flow-regime Cd for reentry). `weather=None`
uses the measured weather loaded by `ballistic.load()` and raises if none is
loaded; pass `ballistic.SpaceWeatherHistory.from_constant(ballistic.drag.WeatherConditions.quiet())`
to model quiet conditions on purpose. `pos_ecef`, `vel_ecef`, `duration_s`
and the table arguments are accepted and unused. The C++ form and the
tabulated alternatives are described in
[DragEnvironment: unified drag / wind / β table](#dragenvironment-unified-drag--wind--β-table).

#### `ballistic.satellite_properties`

The satellite registry (`include/ballistic/satellites.hpp`, C++
`satellites::get(name)` / `satellites::all()`). `get(name)` returns a frozen
`SatelliteProperties` (unknown names raise `KeyError`); `PROPERTIES` maps
every name to one; `SRP_FIT` maps each Sentinel to its (base area, fitted
scale k). Fields: `name`, `mass_kg`, `drag_area_m2` (mean frontal area),
`cd`, `srp_area_m2` (effective cannonball radiation area),
`srp_reflectivity` (Cr − 1), `altitude_km`, `gravity_degree`, `drag_kind`,
`srp_kind`, `provenance`, and the property `beta_kg_m2`. `drag_kind` (mass,
drag area, Cd) and `srp_kind` (radiation area, Cr) say how the values were
obtained, one of `FIT_KINDS = ("physical", "fitted", "nominal", "legacy")`:
from mission documents or a physical model; adjusted to orbit data (absorbs
force-model error); a textbook value not checked for this satellite;
unsourced, low confidence. C++: `satellites::FitKind` and
`satellites::fit_kind_name(kind)`. `legacy_dict()` gives the short-key dict
the validation scripts read.

---

### Module Functions

#### `ballistic.initialize()`

Initializes the library with embedded data, **plus auto-discovered EOP and leap-seconds files**
in the platform data directory if present. Does not download anything (use `Loader` for that).

**What it loads:**
- EGM2008 gravity coefficients (degrees 2-360, embedded)
- DE440 ephemeris for Sun/Moon/Jupiter (1950-2100, ~6.6 MB embedded)
- Leap seconds (embedded, current through 2017, TAI-UTC=37s) — replaced by `leap-seconds.list` if cached in data dir
- `finals2000A.data` / `finals2000A.all` — **auto-discovered** if cached in data dir (required for sub-meter LEO accuracy)

**Sub-meter checklist:**
- Run `ballistic.Loader().load_eop()` *once* to download `finals2000A.data` into the platform
  data directory (`~/.local/share/ballistic` on Linux, `%LOCALAPPDATA%\ballistic` on Windows,
  `~/Library/Application Support/ballistic` on macOS).
- After that, `ballistic.initialize()` auto-loads it. Check `ballistic.get_init_status().eop_loaded`.
- Without EOP, Earth-fixed ↔ GCRS transforms raise `DataCoverageError`; to accept
  zeros instead (3-10 m at LEO) call `ballistic.set_eop_policy('assume_zero')`.

**What it does NOT load:**
- `SW-All.csv` (space weather) and the FES2004 ocean tides — use `ballistic.load()`, `load_space_weather()` or `Loader.space_weather()`
- External JPL ephemerides (e.g., `de421.bsp` for ESA compatibility) — use `load_jpl_ephemeris()`

```python
import ballistic

# Simple initialize - embedded data + auto-discovered EOP / leap-seconds.
# Returns the full InitResult (truthy on success, so `if ballistic.initialize():`
# works); warns if no EOP data was found and for each staleness note.
result = ballistic.initialize()

# Check initialization status
is_init = ballistic.is_initialized()
status = ballistic.get_init_status()  # Returns InitResult

# Version information
version = ballistic.__version__  # e.g., "0.1.2"
```

#### `ballistic.initialize_with_config()`

Initialize with optional external file paths. Files are only loaded if paths are specified.

```python
config = ballistic.InitConfig()
config.eop_path = "finals2000A.data"           # Optional: EOP data
config.jpl_ephemeris_path = "de421.bsp"        # Optional: external ephemeris
config.leap_seconds_path = "leap-seconds.list" # Optional: leap seconds

result = ballistic.initialize_with_config(config)

if result.success:
    print(f"Loaded: {result.jpl_de_version}")
    print(f"Gravity degree: {result.gravity_degree_loaded}")
    print(f"EOP loaded: {result.eop_loaded}")
```

**Note:** Space weather (`SW-All.csv`) cannot be loaded via `InitConfig`. Use `load_space_weather()` separately.

#### Coordinates and a health check

```python
sv = ballistic.itrf_to_gcrs(pos_itrf, vel_itrf, epoch, itrf=ballistic.ITRFRealization.ITRF2020)  # StateVector
sv = ballistic.gcrs_to_itrf(pos_gcrs, vel_gcrs, epoch)           # IAU 2006/2000A, incl. CIP drift
g = ballistic.ecef_to_geodetic(pos_ecef)                          # GeodeticCoord: latitude, longitude [rad], altitude [m]
xyz = ballistic.geodetic_to_ecef(lat_rad, lon_rad, alt_m)

report = ballistic.doctor(load=True, self_test=True, verbose=True)  # also: python -m ballistic doctor
report.ok   # False when a check failed (EOP span, leap-second expiry, data files, SGP4/propagation self-test)
```

`ballistic._core.build_info()` returns how the extension was compiled, as a
dict: `with_openmp`, `compiler` (e.g. `"GCC 13.3.0"`), `cpp_standard` (23)
and `build_type` (`"Release"` when compiled with `NDEBUG`). `doctor()` prints
it.

#### `ballistic.install_nrl_models(accept_license=False, *, source=None, timeout_s=60.0) -> Path | None`

Installs the NRLMSIS 2.1 atmosphere and HWM14 wind models, which model-based
drag needs and the package does not contain (NRLMSIS 2.1: research, academic
and non-profit use only, NRL-SOF-014-1; commercial use is not permitted).
Without `accept_license=True` it prints the license terms and installs
nothing (returns None). With it, the plugin for this platform is downloaded
from the GitHub release of this version, checked against the release's
`SHA256SUMS`, or copied from `source` (a file downloaded elsewhere: offline
installation), into `<data directory>/nrl/`, loaded, and found in every later
session; `ACCEPTED.txt` next to it records the acceptance. Returns the
installed path. Raises `ConfigError` when the download fails or
`BALLISTIC_OFFLINE=1` forbids it, the checksum differs, or the file is not a
plugin this version can load (it is then removed).

#### `ballistic.nrl_models() -> NrlModelsStatus`

Whether the plugin is installed (looks it up the way drag does and loads it
if found; true in a boolean context when installed): `installed`, `path`,
`models`, `problem` (why a plugin that was found could not be loaded),
`install_path` (where it goes) and `download_url` (this platform's file on
the GitHub release).

---

### Data Loading Functions

Load external data files after `initialize()`. These are independent of initialization.

| Function | File | Required For |
|----------|------|--------------|
| `load_eop()` | `finals2000A.data` | Polar motion, UT1-UTC (every Earth-fixed ↔ GCRS transform) |
| `load_leap_seconds()` | `leap-seconds.list` | Future dates beyond 2017 |
| `load_jpl_ephemeris()` | `de421.bsp` etc. | Alternative to embedded DE440 |
| `load_space_weather()` | `SW-All.csv` | F10.7 / Ap for drag |
| `_core.load_ocean_tides()` | `fes2004_Cnm-Snm.dat` | FES2004 ocean tides |

```python
import ballistic

# Load IERS Earth Orientation Parameters
success = ballistic.load_eop("finals2000A.data")

# Load CelesTrak space weather data
success = ballistic.load_space_weather("SW-All.csv")

# Load JPL ephemeris file
success = ballistic.load_jpl_ephemeris("de421.bsp")

# Load leap seconds file
success, message = ballistic.load_leap_seconds("leap-seconds.list", verbose=True)

# Get leap second info
info = ballistic.get_leap_second_info()
print(f"Current TAI-UTC: {info.current_tai_utc} s")
print(f"Entries: {info.entry_count}")

# Get space weather conditions at an epoch (requires load_space_weather first;
# raises outside the file, warns on forecast days unless allow_predicted=False)
epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
weather = ballistic.get_space_weather(epoch, allow_predicted=True)
print(f"F10.7: {weather.f107_daily}, Ap: {weather.ap_daily}")
```

---

### Loader Class (Auto-Download)

Automatic data downloading and caching (`python/ballistic/loader.py`).

```python
import ballistic

# Create loader (uses the platform data directory, ballistic.get_data_directory())
loader = ballistic.Loader()

# Or specify custom directory; offline=True never downloads
loader = ballistic.Loader(directory="./data", verbose=True, expire_days=30, offline=False)

# Everything the recommended setup needs in one call (EOP, leap seconds,
# space weather, ocean tides): ballistic.load(). Granular equivalents:
loader.timescale()             # EOP + leap seconds

# Load space weather for drag modeling
loader.space_weather()

# Individual loading functions
loader.load_eop(force_download=False)
loader.load_leap_seconds(force_download=False)
loader.load_space_weather(force_download=False)
loader.load_ocean_tides(force_download=False)

# Check status: one dict per file ('eop', 'leap_seconds', 'space_weather',
# 'ocean_tides') with exists, age_days, measured_until_mjd, data_age_days,
# loaded, path
status = loader.status()
print(f"EOP loaded: {status['eop']['loaded']}")
print(f"Space weather measured until MJD {status['space_weather']['measured_until_mjd']}")

# Clear cached files
loader.clear_cache()
```

#### Convenience Function

```python
import ballistic

# Quick setup: EOP, leap seconds, space weather and ocean tides
ballistic.load()

# Or with custom directory, or air-gapped from local files
ballistic.load(directory="./data", verbose=True)
ballistic.load(space_weather="SW-All.csv", eop="finals2000A.all", offline=True)
```

#### Data Sources

| Data | URL | Purpose |
|------|-----|---------|
| EOP | https://datacenter.iers.org/data/latestVersion/finals.data.iau2000.txt | Polar motion, UT1-UTC |
| Leap Seconds | https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list | TAI-UTC offsets |
| Space Weather | https://celestrak.org/SpaceData/SW-All.csv | F10.7, Ap indices |
| Ocean tides | https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat | FES2004 ocean tides |

---

### InitConfig Class

```python
config = ballistic.InitConfig()

# File paths (all optional, None by default)
config.egm2008_path = "EGM2008_to2190_TideFree"  # Extend embedded deg 2-360 up to deg 2190
config.eop_path = "finals2000A.data"              # Overrides EOP auto-discovery
config.jpl_ephemeris_path = "de421.bsp"           # JPL ephemeris file (overrides embedded DE440)
config.leap_seconds_path = "leap-seconds.list"    # Overrides leap-seconds auto-discovery
config.nrl_models_path = "ballistic_nrl.so"       # NRL drag models plugin (else BALLISTIC_NRL_MODELS, data dir)
```

Explicit `*_path` fields are strict: if the file is missing or unparseable,
`initialize_with_config()` returns a result with `result.success = False` and
`result.error_message` describing the path. Auto-discovery is best-effort.

### InitResult Class

```python
result = ballistic.initialize_with_config(config)
# or
result = ballistic.get_init_status()

# Status fields
result.success                   # bool: True if initialization succeeded
result.egm2008_loaded            # bool: External EGM2008 loaded
result.eop_loaded                # bool: EOP data loaded
result.jpl_ephemeris_loaded      # bool: External JPL ephemeris loaded
result.embedded_ephemeris_loaded # bool: Embedded DE440 loaded
result.gravity_degree_loaded     # int: Actual gravity degree available
result.jpl_de_version            # str: e.g., "DE421" or "DE440 (embedded)"
result.error_message             # str: Error description if failed
result.eop_auto_discovered       # bool; also leap_seconds_loaded, leap_seconds_auto_discovered,
                                 # eop_source_path
result.eop_table_end_mjd         # float; eop_table_stale, leap_seconds_expiration_jd,
                                 # leap_seconds_expired
result.warnings                  # list[str]: staleness and unusable-file notes
result.nrl_models_loaded         # bool: NRLMSIS 2.1 / HWM14 plugin loaded; nrl_models_path
```

### LeapSecondInfo Class

```python
info = ballistic.get_leap_second_info()

info.entry_count      # int: Number of leap second entries
info.current_tai_utc  # int: Current TAI-UTC offset in seconds
info.loaded_from_file # bool: True if loaded from external file
info.expiration_jd    # float: Expiration Julian Date
```

---

### Epoch Class (Python)

```python
import ballistic

# Create from UTC
epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)

# Other scales: from_tt / from_tai / from_gps / from_tdb (calendar),
# from_gps_week(week, sow), from_datetime(aware datetime), from_tt_split(day, sec)
epoch = ballistic.Epoch.from_gps(2024, 1, 15, 12, 0, 0)

# Create from Julian Date (the scale is required; a bare from_jd(jd) raises RemovedApiError)
epoch = ballistic.Epoch.from_jd(2460325.5, ballistic.TimeScale.TT)

# Create from MJD
epoch = ballistic.Epoch.from_mjd(60325.0, ballistic.TimeScale.UTC)

# Create from Unix timestamp
epoch = ballistic.Epoch.from_unix(1705320000.0)

# J2000.0 epoch
j2000 = ballistic.Epoch.j2000()

# Time scale access (all return Julian Date)
utc_jd = epoch.utc_jd
tt_jd = epoch.tt_jd
tai_jd = epoch.tai_jd
tdb_jd = epoch.tdb_jd
ut1_jd = epoch.ut1_jd
gps_jd = epoch.gps_jd
unix_ts = epoch.unix_timestamp

# Year property
year = epoch.year

# Arithmetic (ALL time deltas are in SECONDS)
epoch2 = epoch + 3600.0  # Add 3600 seconds (1 hour)
epoch3 = epoch - 60.0    # Subtract 60 seconds (backward in time)
dt = epoch2 - epoch      # Difference in seconds (returns 3600.0)

# Comparison
epoch < epoch2
epoch == epoch

# Calendar conversion (UTC by default; to_calendar(ballistic.TimeScale.TT) etc.)
year, month, day, hour, minute, second = epoch.to_calendar()
```

---

### Trajectory Class (Python) - Unified API

The unified `Trajectory` class provides a Pythonic interface combining propagation, interpolation, and batch processing. Use the **call operator** `traj(times, frame)` for interpolation.

```python
import ballistic
import numpy as np

# Initialize library
ballistic.initialize()

# Create initial state
epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
pos = np.array([6778137.0, 0.0, 0.0])  # ECEF, meters
vel = np.array([0.0, 7175.0, 0.0])  # Earth-fixed (ITRF): circular 7669 m/s minus Earth rotation 494 m/s

# Create trajectory with default config
traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel))

# Or with dataclass config
config = ballistic.TrajectoryConfig(
    gravity_degree=70,
    integrator=ballistic.IntegratorType.YOSHIDA6,
    symplectic_step_size=30.0,
)
traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=config)

# Step-based propagation
for _ in range(100):
    result = traj.step()  # Returns StepResult

# Direct propagation
result = traj.propagate_to(epoch + 3600.0)  # to an Epoch
result = traj.propagate_for(600.0)          # seconds from the CURRENT epoch

# === INTERPOLATION via call operator ===

# Single time (float = seconds from initial epoch)
state = traj(1800.0)  # Returns InterpolatedState

# Single time with Epoch
state = traj(epoch + 1800.0)

# Multiple times (list or array)
states = traj([100, 500, 1000, 1800])  # Returns list of InterpolatedState

# With output frame (string or enum)
state = traj(1800.0, frame='GCRS')
state = traj(1800.0, frame='ECEF')
state = traj(1800.0, frame=ballistic.InterpolationFrame.GCRS)

# Access state data
print(f"Position: {state.position}")
print(f"Velocity: {state.velocity}")
print(f"Acceleration: {state.acceleration}")

# === History & State ===
size = traj.history_size()
all_history = traj.history()
current_epoch = traj.epoch()
state_ecef = traj.state_ecef()
state_gcrs = traj.state_gcrs()

# === Events (reentry) ===
r = traj.propagate_until(7200.0, [ballistic.events.altitude_crossing(80e3)])
# r.stopped_by (None = reached the target), r.log; the ground impact is armed
# by default (stop_at_ground=True). Single mode only.

# === Altitude propagation ===
result = traj.propagate_to_altitude(
    300_000,                            # meters
    ballistic.AltitudeCondition.Below,
    ballistic.PropagationDirection.Forward,
    86400.0)                            # max_duration_s (default 7 days)

# === State transition matrix and covariance (single mode) ===
# Phi(t, t0) and Phi P_0 Phi^T (P_0: the initial covariance) along the propagated span; GCRS, [r (m); v (m/s)].
# t as for traj(t): seconds after the start or an Epoch; a list gives (N, 6, 6).
phi = traj.stm(1800.0)                       # (6, 6) numpy array
P = traj.covariance(1800.0, np.diag([1.0] * 3 + [1e-6] * 3))
# keyword options: fd_dx=1.0 (m), fd_dv=1e-3 (m/s), max_substep_s=10.0
# Batch mode raises ConfigError; outside the span raises DataCoverageError.

# === Symplectic support ===
# (traj.warnings: the trajectory's propagation notes)
is_symp = traj.is_symplectic()
drift = traj.energy_drift()

# === Low-level access ===
core_traj = traj.trajectory()  # C++ Trajectory object
interp = traj.interpolator()   # C++ TrajectoryInterpolator
```

#### Batch Mode

Pass a list of `ballistic.State` to enable batch mode:

```python
# Create batch from initial states (Earth-fixed velocities)
states = [
    ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0]),
    ballistic.State.from_itrf(epoch, [6878137.0, 0.0, 0.0], [0.0, 7120.0, 0.0]),
    ballistic.State.from_itrf(epoch, [6978137.0, 0.0, 0.0], [0.0, 7050.0, 0.0]),
]

batch = ballistic.Trajectory(states)
print(batch.is_batch)  # True
print(len(batch))      # 3

# Propagate all (each from its own epoch, landing exactly)
batch.propagate_for(3600.0)
batch.step()  # Steps all trajectories

# Interpolate all at single time - returns list of BatchInterpolatedState
results = batch(1800.0)
for r in results:
    if r.valid:
        print(f"Traj {r.trajectory_index}: {r.position}")

# Interpolate all at multiple times - returns list of lists
multi_results = batch([100, 500, 1000])

# Vectorized queries: batch() with no argument returns a BatchInterpolatorFunction
f = batch()
r = f(1800.0)                    # BatchArrayResult: r.position / r.velocity /
                                 # r.acceleration (N, 3), r.epoch_s (N,), r.valid (N,)
impact = f.at_alt(0)             # descending crossing, bisection to 1 mm (existing history only)
f.at_alt(50000, descending=False)
interp0 = f[0]                   # one member's TrajectoryInterpolator
f.set_frame(ballistic.InterpolationFrame.GCRS)

# State access returns lists
epochs = batch.epoch()           # List of Epochs
states = batch.state_ecef()      # List of StateVector
histories = batch.history()      # List of history lists
drift_list = batch.energy_drift()  # List of floats
```

---

### TrajectoryConfig Dataclass (Python)

```python
from ballistic import TrajectoryConfig

# Frozen, keyword-only dataclass: a misspelt field raises. Change a copy
# with config.replace(gravity_degree=100).
config = TrajectoryConfig(
    gravity_degree=70,                # EGM2008 degree (default 70)
    enable_third_body=True,           # Sun/Moon/Jupiter
    enable_relativistic=True,         # Schwarzschild + Lense-Thirring
    itrf=ballistic.ITRFRealization.ITRF2020,
    integrator=ballistic.IntegratorType.RK45,
    rk45_initial_step_size=60.0,      # Initial RK45 step [s]; ~8 for glides
    rk45_max_step_size=3600.0,        # Max RK45 step [s]; ~8 for glides
    symplectic_step_size=30.0,        # For YOSHIDA integrators
    srp=ballistic.SrpConfig(mass_kg=2200.0, area_m2=38.4, cr=1.0),
                                      # None (default) = radiation off; enable
                                      # for real satellites
    report_forces=False,
    include_disabled_forces=False,
)
# Other fields: enable_solid_tides=True, ocean_tide_degree=50,
# symplectic_adaptive / _min_step / _max_step / _alpha / _symmetry_tol,
# symplectic_nc_strategy, drag_env=None, empirical_accel=None,
# ea_reference_epoch=None. ballistic.recommended_config builds the
# best-accuracy config for a satellite.

# Create trajectory with config
traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=config)
```

---

### InterpolatorConfig Dataclass (Python)

Output frame is specified at call time, not in config:

```python
from ballistic import InterpolatorConfig

config = InterpolatorConfig(
    itrf=None,                 # default: inherit the Trajectory's realization
    cache_coefficients=True,
    allow_propagation=True,    # Auto-extend for out-of-range times
    max_propagation_time=3600.0,
)

traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel), interp_config=config)

# Frame specified at call time
state = traj(1800.0, frame='GCRS')
```

---

### Configuration Classes (Python)

#### TrajectoryConfig (Pythonic Dataclass)

```python
from ballistic import TrajectoryConfig

# Create with keyword arguments (dataclass)
config = TrajectoryConfig(
    gravity_degree=70,                  # 70 for LEO, 4-8 for MEO/GEO/HEO
    enable_third_body=True,             # Sun/Moon/Jupiter
    enable_relativistic=True,           # Schwarzschild + Lense-Thirring
    itrf=ballistic.ITRFRealization.ITRF2020,
    integrator=ballistic.IntegratorType.RK45,
    symplectic_step_size=30.0,          # For YOSHIDA integrators
    srp=None,                           # None (default) = radiation off; pass
                                        # SrpConfig(mass_kg=, area_m2=, cr=) for a real satellite
    report_forces=False,
    include_disabled_forces=False,
)

# Use with Trajectory
traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=config)
```

#### InterpolatorConfig (Pythonic Dataclass)

Output frame is specified at call time, not in config:

```python
from ballistic import InterpolatorConfig

config = InterpolatorConfig(
    itrf=ballistic.ITRFRealization.ITRF2020,
    cache_coefficients=True,
    allow_propagation=True,
    max_propagation_time=3600.0,
)

# Use with Trajectory
traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel), interp_config=config)

# Frame specified at interpolation time
state = traj(1800.0, frame='GCRS')  # 'ECEF' or 'GCRS'
```

#### C++ Config Classes (Advanced)

For direct C++ API access:

```python
# C++ TrajectoryConfig (mutable properties)
config = ballistic._core.TrajectoryConfig()
config.gravity_degree = 70
config.integrator = ballistic.IntegratorType.YOSHIDA6
config.symplectic.step_size = 30.0  # Access via sub-struct

# C++ InterpolatorConfig (includes output_frame)
config = ballistic._core.InterpolatorConfig()
config.output_frame = ballistic.InterpolationFrame.ECEF
config.itrf = None                       # None = inherit the Trajectory's realization
config.cache_coefficients = True
config.allow_propagation = True
config.max_propagation_time = 3600.0
```

---

### EmpiricalAcceleration (Python)

```python
# Factories
zero_ea = ballistic.EmpiricalAcceleration.zero()
table_ea = ballistic.EmpiricalAcceleration.from_entries(
    entries, ballistic.EmpiricalAccelFrame.GCRS, ballistic.estimation.EAInterp.Cubic)
hermite_ea = ballistic.EmpiricalAcceleration.from_entries_hermite(
    entries, derivatives, ballistic.EmpiricalAccelFrame.GCRS)
bspline_ea = ballistic.EmpiricalAcceleration.from_bspline(
    coefficients, ballistic.estimation.EARepresents.Acceleration, ballistic.EmpiricalAccelFrame.GCRS)

# Query
a = bspline_ea.accel_at(t_seconds)
frame = bspline_ea.frame()
empty = bspline_ea.is_zero()
```

See [`empirical_acceleration_estimation.md`](empirical_acceleration_estimation.md) for tuning guidance, validated defaults, and the comparison of the two estimators.

The tools that fit one, and the types its factories take, are in
[`ballistic.estimation`](#ballisticestimation-fitting-an-empirical-acceleration).

---

### Errors and warnings (`ballistic.errors`)

Every error class (`BallisticError` and its subclasses `DataCoverageError`,
`FrameError`, `InputPlausibilityError`, `ConfigError`, `RemovedApiError`,
`StrictDiagnosticError`), every warning class (`BallisticWarning` and one
subclass per diagnostic category) and
`set_strict` are importable from both `ballistic` and `ballistic.errors`.
What each means: [Errors and warnings](#errors-and-warnings).

---

### Enumerations

```python
# Time scales
ballistic.TimeScale.UTC
ballistic.TimeScale.TT
ballistic.TimeScale.TAI
ballistic.TimeScale.TDB
ballistic.TimeScale.UT1
ballistic.TimeScale.GPS

# Propagation frame (internal — always GCRS)
ballistic.PropagationFrame.GCRS

# Integrator type
ballistic.IntegratorType.RK45
ballistic.IntegratorType.YOSHIDA6
ballistic.IntegratorType.YOSHIDA8

# ITRF realization
ballistic.ITRFRealization.ITRF2000
ballistic.ITRFRealization.ITRF2005
ballistic.ITRFRealization.ITRF2008
ballistic.ITRFRealization.ITRF2014  # Use for Sentinel < March 2025
ballistic.ITRFRealization.ITRF2020  # Default

# Interpolation output frame
ballistic.InterpolationFrame.ECEF
ballistic.InterpolationFrame.GCRS

# Altitude condition
ballistic.AltitudeCondition.Above
ballistic.AltitudeCondition.Below

# Propagation direction
ballistic.PropagationDirection.Forward
ballistic.PropagationDirection.Backward

# (Drag is configured through TrajectoryConfig.drag_env /
# atmosphere::DragEnvironment; there is no atmosphere-type enum.)

# Symplectic non-conservative strategy (SymplecticOptions.nc_strategy)
ballistic.NonConservativeStrategy.NONE         # Pure symplectic (drop NC forces)
ballistic.NonConservativeStrategy.SPLITTING    # Symplectic for cons. + RK for NC
ballistic.NonConservativeStrategy.SLIMPLECTIC  # Default; variational integrator

# Empirical Acceleration enums
ballistic.EmpiricalAccelFrame.GCRS
ballistic.EmpiricalAccelFrame.ITRF
ballistic.estimation.EAInterp.Linear
ballistic.estimation.EAInterp.Cubic
ballistic.estimation.EARepresents.Acceleration  # EA value is the acceleration directly
ballistic.estimation.EARepresents.Velocity      # EA value is velocity; differentiate once
ballistic.estimation.EARepresents.Position      # EA value is position; differentiate twice

# Calibration result + CV mode
ballistic.estimation.GNStatus.Disabled                       # Caller did not request GN
ballistic.estimation.GNStatus.Ran                            # GN executed
ballistic.estimation.GNStatus.SkippedAccelerationCircular    # ACC obs: GN would target a re-prop of self
ballistic.estimation.CVMode.IRGN                          # Default CV inner solver
ballistic.estimation.CVMode.FixedLambda                   # Constant-λ inner solver (ACC only; measured worse for POS/VEL)
# The smoother handles range rate via EASmootherConfig.rate_budget.
```

---

### C++ Core Classes (Python)

For direct access to the C++ classes (advanced use), import them from
`ballistic._core`:

```python
import ballistic
from ballistic import _core

traj = _core.Trajectory(ballistic.State.from_itrf(epoch, pos, vel))   # optional 2nd arg: _core.TrajectoryConfig
config = _core.TrajectoryConfig()          # mutable C++ struct (cf. the frozen dataclass)
interp_config = _core.InterpolatorConfig()
interp = _core.TrajectoryInterpolator(traj)
batch = _core.TrajectoryBatch(states)
```

---

### TrajectoryInterpolator Class (Python) - C++ API

Direct access to the C++ interpolator (also available via `traj.interpolator()`):

```python
import ballistic
from ballistic import _core

# Build trajectory with history
traj = _core.Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
while traj.epoch() - epoch < 3600.0:
    traj.step()

# Create interpolator
interp = _core.TrajectoryInterpolator(traj)

# Or with config
config = _core.InterpolatorConfig()
config.output_frame = ballistic.InterpolationFrame.GCRS
interp = _core.TrajectoryInterpolator(traj, config)

# Query at any time
state = interp.get_state(epoch + 1800.0)
if state:
    print(f"Position: {state.position}")
    print(f"Velocity: {state.velocity}")

# Individual components
pos = interp.position(epoch + 1800.0)
vel = interp.velocity(epoch + 1800.0)
acc = interp.acceleration(epoch + 1800.0)

# Batch queries
times = [epoch + 100.0, epoch + 500.0, epoch + 1000.0]
states = interp.get_states(times)

# Altitude-based propagation
result = interp.propagate_to_altitude(
    300_000,                            # meters
    ballistic.AltitudeCondition.Below,
    ballistic.PropagationDirection.Forward,
    86400.0)                             # max_duration_s (default 7 days)

if result.condition_met:
    print(f"Reached altitude in {result.steps_taken} steps")
    print(f"Final altitude: {result.ending_altitude_m / 1000} km")

# Range information
start = interp.history_start()
end = interp.history_end()
is_covered = interp.covers(epoch + 1800.0)
size = interp.history_size()

# Explicit history extension
interp.extend_to(epoch + 7200.0)           # Extend forward
interp.extend_backward_to(epoch - 300.0)   # Extend backward

# Configuration
interp.set_output_frame(ballistic.InterpolationFrame.ECEF)
interp.set_itrf_realization(ballistic.ITRFRealization.ITRF2014)
interp.set_allow_propagation(True)
interp.set_max_propagation_time(7200.0)

# Cache management
interp.clear_cache()
interp.invalidate()
```

---

### TrajectoryBatch Class (Python) - C++ API

Direct access to the C++ batch processor:

```python
import ballistic
from ballistic import _core
import numpy as np

# Create batch from initial states
epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
states = [
    ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0]),
    ballistic.State.from_itrf(epoch, [6878137.0, 0.0, 0.0], [0.0, 7120.0, 0.0]),
    ballistic.State.from_itrf(epoch, [6978137.0, 0.0, 0.0], [0.0, 7050.0, 0.0]),
]

batch = _core.TrajectoryBatch(states)

# Or with config
config = _core.TrajectoryConfig()
batch = _core.TrajectoryBatch(states, config)

# Properties
size = len(batch)  # or batch.size()
is_empty = batch.empty()

# Propagation (each member from its own epoch, landing exactly)
batch.propagate_all(3600.0)  # 1 hour
batch.propagate_all_to(epoch + 7200.0)
batch.step_all(n_steps=10)

# Query all at single time
results = batch.get_states_at(epoch + 1800.0)
for r in results:
    if r.valid:
        print(f"Traj {r.trajectory_index}: {r.position}")

# Query all at multiple times
times = [epoch + 100.0, epoch + 500.0]
all_results = batch.get_states_at_times(times)

# Altitude-based batch propagation
alt_results = batch.propagate_to_altitude(
    300_000,
    ballistic.AltitudeCondition.Below,
    ballistic.PropagationDirection.Forward,
    86400.0)                             # max_duration_s

# Access individual trajectory / interpolator
traj = batch.trajectory(0)
interp = batch.interpolator(0)

# Configuration methods
batch.set_output_frame(ballistic.InterpolationFrame.GCRS)
batch.set_itrf_realization(ballistic.ITRFRealization.ITRF2014)
batch.set_allow_propagation(False)
batch.set_max_propagation_time(3600.0)
```

---

## `ballistic.estimation`: fitting an empirical acceleration

`from ballistic.estimation import ...`. An empirical acceleration is a small correction fitted to
observations, for forces the model does not know. The fitted correction is a
`ballistic.EmpiricalAcceleration` (top level, since a `TrajectoryConfig` carries it).
The wrappers accept the `ballistic.TrajectoryConfig` dataclass as the force model.

| Name | What it is |
|------|------------|
| `ballistic.estimation.estimate_ea_bspline` | fit an EA to dense positions, velocities or accelerometer data (B-spline + Gauss-Newton) |
| `ballistic.estimation.EABSplineConfig` | its settings (regularization, knot spacing, `gn`) |
| `ballistic.estimation.PositionSamples` | observed Earth-fixed positions, `[(Epoch, xyz)]` |
| `ballistic.estimation.VelocitySamples` | observed Earth-fixed velocities |
| `ballistic.estimation.AccelSamples` | observed accelerations |
| `ballistic.estimation.GNRefine` | Gauss-Newton refinement settings (on by default) |
| `ballistic.estimation.CVMode` | what the cross-validation search varies |
| `ballistic.estimation.GNStatus` | what happened to the Gauss-Newton step |
| `ballistic.estimation.EAResult` | result of `estimate_ea_bspline` |
| `ballistic.estimation.refine_ea_gn` | improve an existing EA so the orbit matches positions |
| `ballistic.estimation.estimate_ea_smoother` | fit an EA to several sensors at once (Kalman filter and smoother); accepts the `TrajectoryConfig` dataclass |
| `ballistic.estimation.EASmootherConfig` | its settings (few by design) |
| `ballistic.estimation.EASmootherResult` | its result: EA, 1-sigma uncertainty, NIS |
| `ballistic.estimation.EASmootherRateBudget` | how radar range rate is weighted |
| `ballistic.estimation.RadarObs` | ground radar samples (range, azimuth, elevation, range rate) |
| `ballistic.estimation.OpticsObs` | angles-only optical samples |
| `ballistic.estimation.PositionFixObs` | measured Earth-fixed positions (GPS fixes, an impact point) |
| `ballistic.estimation.AccelObs` | on-board accelerometer, bias estimated with the EA |
| `ballistic.estimation.MultiSensorScenario` | the start state plus all sensors |
| `ballistic.estimation.SensorMask` | which sensors of a scenario an estimate uses |
| `ballistic.estimation.SensorClock` | how a sensor's clock offset is modeled |
| `ballistic.estimation.EAProblemSpec` | observations + declared noise for `estimate_ea` |
| `ballistic.estimation.estimate_ea` | the library picks the estimator and audits the declared noise |
| `ballistic.estimation.EASpecResult` | result of `estimate_ea` |
| `ballistic.estimation.EAMethod` | which estimator `estimate_ea` ran |
| `ballistic.estimation.information_bound` | the best accuracy any estimator could reach with a sensor suite; accepts the `TrajectoryConfig` dataclass |
| `ballistic.estimation.EAInformationBound` | its result |
| `ballistic.estimation.EAInformationBoundOptions` | its settings |
| `ballistic.estimation.reject_heavy_tails` | remove outliers from dense samples before fitting |
| `ballistic.estimation.decompose_ea` | split one EA vector into a drag-like part and the rest |
| `ballistic.estimation.refine_beta_from_ea` | estimate the ballistic coefficient from a fitted EA; returns a refined `DragEnvironment` |
| `ballistic.estimation.BetaDecompositionConfig` | its settings |
| `ballistic.estimation.BetaResult` | ballistic coefficients along the orbit |
| `ballistic.estimation.BetaStepEstimate` | the coefficient at one point |
| `ballistic.estimation.EADecomposition` | result of `decompose_ea` |
| `ballistic.estimation.EmpiricalAccelEntry` | one time point of a tabulated EA (`EmpiricalAcceleration.from_entries`) |
| `ballistic.estimation.EAInterp` | how a tabulated EA is interpolated |
| `ballistic.estimation.EARepresents` | what a B-spline EA was fitted to (acceleration, velocity, position) |
| `ballistic.estimation.BSplineCoefficients` | B-spline coefficients for `EmpiricalAcceleration.from_bspline` |

### Empirical Acceleration Estimation (Python)

Python mirror of the C++ EA estimation API. All symbols are in
`ballistic.estimation`. All observations are passed as ECEF; the estimator
converts internally. The observation modality is carried by the sample
type: wrap data in `PositionSamples` / `VelocitySamples` /
`AccelSamples`. Radar and sensor fusion go to `estimate_ea_smoother`.

```python
import ballistic
import numpy as np

ballistic.initialize()

# 1. Configure the estimator
cfg = ballistic.estimation.EABSplineConfig()
cfg.knot_spacing_s = 2.0
cfg.degree = 5                       # quintic (default)
cfg.noise_sigma = 3.0                # m (POS — realistic GPS noise)
cfg.set_irgn(lambda_0=1e-4, decay=0.5, lambda_min=1e-12)  # default
# cfg.set_cv()                       # hv-block CV (auto-tuning, ~1-8 h wall)
# cfg.set_gcv(lam_min=1e-12, lam_max=1e-1)
# cfg.set_fixed_lambda(1e-4)

# GN refinement is ON by default (cfg.gn = ballistic.estimation.GNRefine()); None skips it
cfg.gn = ballistic.estimation.GNRefine()        # use_stm=True by default

# 2. Run an estimator — modality via the sample type. TrajectoryConfig is a
#    frozen dataclass: set fields in the constructor or with .replace().
fm = ballistic.TrajectoryConfig(gravity_degree=70)

result = ballistic.estimation.estimate_ea_bspline(
    epoch0, pos_ecef, vel_ecef,
    ballistic.estimation.PositionSamples(position_data_ecef), fm, cfg)

# Other modalities / estimators:
#   ballistic.estimation.estimate_ea_bspline(..., ballistic.estimation.VelocitySamples(vel_data), ...)
#   ballistic.estimation.estimate_ea_bspline(..., ballistic.estimation.AccelSamples(acc_data), ...)
#   Sensor fusion / radar → ballistic.estimation.estimate_ea_smoother (see Multi-Sensor).

# 3. Inspect the result. succeeded() = ran without error (zero EA still
#    counts); has_correction() = succeeded() AND non-zero EA.
if result.has_correction():
    print(f"GN status: {result.gn_status}")           # GNStatus.Ran / Disabled / Skipped*
    print(f"Iterations: {result.iterations_used}")
    print(f"λ used: {result.lambda_used}")            # Optional[float]

    # 4. Apply to re-propagation
    fm_recon = fm.replace(empirical_accel=result.empirical_accel)
    traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch0, pos_ecef, vel_ecef), fm_recon)

# 5. EA → β calibration. Use:
#
#    (a) refine_beta_from_ea — refine the β column of a prior
#        DragEnvironment from the estimator's self-consistent trajectory.
refined_env = ballistic.estimation.refine_beta_from_ea(
    prior_env, result.empirical_accel, result.collocation_history, epoch0,
    ballistic.estimation.BetaDecompositionConfig())
# refined_env.beta() is now the calibrated BetaProfile1D.

#    (b) decompose_ea — single-vector split: ea + velocity + density →
#        velocity-aligned drag + perpendicular residual + β.
d = ballistic.estimation.decompose_ea(np.asarray(ea_vec, float), np.asarray(velocity_vec, float), 1e-13)
print(f"β: {d.beta} kg/m²,  drag fraction: {d.drag_fraction}")

# 6. Standalone GN refinement (when EA was built from another source).
#    Takes the force model + GNRefine; returns the refined EmpiricalAcceleration.
gn = ballistic.estimation.GNRefine()
gn.max_iterations = 30
gn.use_stm = True
gn.rk45_max_step = None              # None | 'match_knots' | float [s]
refined = ballistic.estimation.refine_ea_gn(
    epoch0, pos_ecef, vel_ecef, position_data, initial_ea, fm,
    gn=gn, max_duration_s=3600.0)
```

#### Calibration Result Types

```python
# EAResult (returned by estimate_ea_bspline)
result.empirical_accel       # ballistic.EmpiricalAcceleration (or None)
result.iterations_used       # int
result.collocation_history   # list[HistoryEntry]
result.gn_status             # ballistic.estimation.GNStatus
result.lambda_used           # Optional[float]
result.sigma_sq_estimate     # Optional[float]
result.obs_rmse              # Optional[float]
result.error_message         # str (empty on success)
result.succeeded()           # bool — ran without error (zero EA still success)
result.has_correction()      # bool — succeeded() AND non-zero EA

# BetaResult
beta_result.constant_beta     # float (median β [kg/m²])
beta_result.step_estimates    # list[BetaStepEstimate]
beta_result.node_altitudes_km # list[float] (altitude binning nodes)
beta_result.node_betas        # list[float] (β at each altitude node)
beta_result.steps_solved      # int (count of valid β estimates)
beta_result.error_message     # str

# BetaStepEstimate
step.time_s, step.altitude_km, step.beta, step.w, step.weight

# EADecomposition (single-vector)
d.beta, d.ea_drag, d.ea_residual, d.drag_fraction
```

---

### Multi-Sensor EA Estimation (Python)

Top-level `ballistic.*` re-exports. The full public surface
is Pythonic: numpy bulk samplers, dataclass-style scenario construction,
batched EA evaluation. Detailed reference:
[`docs/multi_sensor_estimator.md`](multi_sensor_estimator.md).

```python
import ballistic
import numpy as np

ballistic.initialize()

# --- Sensors --------------------------------------------------------------
# Numpy bulk setters (preferred over building list-of-tuples manually).
# Shapes: times_s (N,), observations (N, K) where K = 4 radar, 2 optics, 3 fix/accel.
radar = ballistic.estimation.RadarObs()
radar.position_ecef = station_ecef
radar.sigma_range, radar.sigma_angle, radar.sigma_rate = 2.0, 5e-4, 0.02
radar.lat_rad, radar.lon_rad = lat, lon
radar.ecef_to_enu = ecef_to_enu_matrix
radar.set_samples(epoch0, times_s, obs_Nx4)

# Forward model — for synthetic data generation.
clean_obs = radar.predict(state_vector)   # ndarray [range, az, el, range_rate]

fix = ballistic.estimation.PositionFixObs()
fix.sigma_pos = 1.0
fix.set_samples(epoch0, np.array([0.0]), pos0.reshape(1, 3))

# --- Scenario (dataclass-style construction) ------------------------------
scenario = ballistic.estimation.MultiSensorScenario(
    epoch0=epoch0, pos0_ecef=pos0, vel0_ecef=vel0,
    arc_duration_s=arc_s,
    radars=[radar], fixes=[fix])
mask = ballistic.estimation.SensorMask.all_on(scenario)

# --- Config (minimal by design) -------------------------------------------
cfg = ballistic.estimation.EASmootherConfig()   # auto NIS prior + measured rate budget

# --- Run the smoother (accepts the TrajectoryConfig dataclass directly) ---
fm_cfg = ballistic.TrajectoryConfig()
result = ballistic.estimation.estimate_ea_smoother(scenario, mask, fm_cfg, cfg)

# --- Results (EASmootherResult) -------------------------------------------
result.empirical_accel         # ballistic.EmpiricalAcceleration (GCRS)
result.nis                     # normalized innovation² / dof (~1 ideal, <1 conservative)
result.recovered_bias          # ndarray (3,) — AccelObs bias (ECEF), if accels fused
result.sigma_a_used            # GM prior amplitude selected [m/s²]
result.ea_sigma                # list of per-sample 1σ of the EA (posterior)

# --- Batched EA evaluation -----------------------------------------------
times = np.linspace(0.0, arc_s, 100)
ea_grid = result.empirical_accel.evaluate_at(times)    # (100, 3) ndarray
```

#### Fusion estimator (Python)

`ballistic.estimation.estimate_ea_smoother(scenario, mask, fm_cfg, cfg)` is the single
fusion entry point (GM-RTS). It accepts the Pythonic `TrajectoryConfig`
dataclass directly and returns an `EASmootherResult`.

#### Enumerations (Python)

```python
ballistic.estimation.SensorClock.None_       # no clock-bias state for this sensor
ballistic.estimation.SensorClock.Shared      # one clock bias shared by the sensors that select it
ballistic.estimation.SensorClock.Own         # a clock bias of its own
ballistic.estimation.EASmootherRateBudget.Auto    # range-rate noise budget measured from the data (default)
ballistic.estimation.EASmootherRateBudget.Off     # range rate not used
ballistic.estimation.EASmootherRateBudget.Manual  # EASmootherConfig.sigma_rate_manual
ballistic.estimation.EAMethod.BSplinePosition / BSplineVelocity / BSplineAcceleration / Smoother  # EASpecResult.method_used
```

End-to-end Python test: `tests/python/test_multisensor_e2e.py` runs the
smoother on a synthetic radar + position-fix scenario.

---

## `ballistic.drag`: atmospheres built by hand

`from ballistic.drag import ...`. Most users turn drag on with `ballistic.recommended_config` or
`ballistic.make_drag_environment`; this module holds the tabulated and measured alternatives
and the file tools. The generators are described in
[DragEnvironment: unified drag / wind / β table](#dragenvironment-unified-drag--wind--β-table).

| Name | What it is |
|------|------------|
| `ballistic.drag.DragEnvironment` | density, wind and ballistic coefficient; assign to `TrajectoryConfig.drag_env` |
| `ballistic.drag.DragSample` | what `DragEnvironment.evaluate` returns at one position and time |
| `ballistic.drag.from_models_snapshot` | table of NRLMSIS 2.1 / HWM14 over altitude at one time and place |
| `ballistic.drag.from_models_along` | table over altitude and time below a given trajectory |
| `ballistic.drag.from_models_4d` | table over altitude, latitude, longitude and time |
| `ballistic.drag.from_accelerometer_density` | environment from measured (accelerometer-derived) densities |
| `ballistic.drag.extend_density_below` | U.S. Standard Atmosphere 1976 below a floor altitude |
| `ballistic.drag.standard_leo_altitude_grid_km` | the default altitude grid, 100-1500 km |
| `ballistic.drag.IterationDiag` | report from building a table along a trajectory |
| `ballistic.drag.TimeWindow` | a time interval `[start, end)` for space-weather histories |
| `ballistic.drag.WeatherConditions` | one fixed set of space-weather values (quiet / active / storm) |
| `ballistic.drag.BallisticModel` | ballistic coefficient with optional mass and area |
| `ballistic.drag.parse_density_table` | read a TU Delft measured-density file |
| `ballistic.drag.load_beta_table` | read a ballistic-coefficient-versus-altitude CSV |
| `ballistic.drag.load_empirical_acceleration` | read a correction-acceleration-versus-time CSV |
| `ballistic.drag.save_drag_env` | save a `DragEnvironment` as `.npz` |
| `ballistic.drag.load_drag_env` | load one back |

#### BallisticModel

```python
# Positional (beta, mass, area): beta = m / (Cd * A) [kg/m^2] is the drag
# parameter; mass [kg] and frontal area [m^2] are metadata. Also BallisticModel(beta).
model = ballistic.drag.BallisticModel(157.5, 2200.0, 6.35)   # Sentinel-1

beta = model.ballistic_coefficient()             # the stored beta
inv_beta = model.inverse_ballistic_coefficient()  # 1 / beta
```

#### WeatherConditions

```python
# Factory methods
quiet = ballistic.drag.WeatherConditions.quiet()    # F10.7=70, Ap=4
active = ballistic.drag.WeatherConditions.active()  # F10.7=200, Ap=15
storm = ballistic.drag.WeatherConditions.storm()    # F10.7=250, Ap=100

# Custom
wx = ballistic.drag.WeatherConditions()
wx.f107_daily = 150.0
wx.f107_average = 150.0
wx.ap_daily = 15.0
wx.ap_history = [15.0] * 7  # 7 values in NRLMSIS's AP-array order
```

---

### DragEnvironment NPZ Persistence (Python only)

Save and reload built `DragEnvironment` instances to a numpy `.npz`
file. A `from_models_along` env over a multi-orbit arc costs ~100 ms
of NRLMSIS21 / HWM14 evaluations; the NPZ load is ~1 ms — substantial
speedup for repeated runs.

```python
import ballistic

env, _ = ballistic.drag.from_models_along(epochs, positions, sw, beta)
ballistic.drag.save_drag_env(env, "my_env.npz")

loaded = ballistic.drag.load_drag_env("my_env.npz")
sample = loaded.evaluate(pos_ecef, epoch)   # DragSample (density, wind, beta, ...)
```

**Supported variants**:

| Density | Wind | Beta |
|---|---|---|
| `DensityProfile1D` | `WindProfile1D_ENU` (or empty) | `BetaProfile1D` |
| `DensityProfile2D` | `WindProfile2D_ECEF` | `BetaProfile1D` |
| `DensityField4D` | `WindField4D_ECEF` | `BetaProfile1D` |
| live (`make_drag_environment`) | live or none | `BetaProfile1D` |

A live environment has no table: it is saved as its settings (β, forecast
policy, the `extend_density_below` floor, wind on/off) and rebuilt on the
space weather loaded when `load_drag_env` runs; one built with explicit
weather cannot be saved.

**Not supported** (raise `NotImplementedError`): `BetaProfile2D` and an
`AeroBeta` β column.

Round-trip tolerance: 1D / 2D variants preserve to 1e-12 rel (float64
storage). 4D variants preserve to ~2e-3 rel for density (half-precision
log(ρ) floor) and ~0.5 m/s for wind (half-precision direct-value floor).

NPZ schema is documented in `python/ballistic/dragenv_io.py` (axis
arrays + parallel flat data buffers; epochs stored as UTC MJD doubles).

---

## `ballistic.aero`: flow-regime drag coefficient

`from ballistic.aero import ...`. For reentry, where the drag coefficient changes from its
free-molecular value in orbit to its continuum value in dense air. An `AeroBeta` replaces a
constant `BetaProfile1D` as the `beta` of `make_drag_environment`;
`recommended_config(state, ..., reentry=ballistic.aero.AeroShape.SPHERE, reference_length_m=...)`
builds one for you. Guide: [`reentry_and_tle.md`](reentry_and_tle.md).

```python
from ballistic.aero import AeroBeta, AeroShape, FlowRegimeCd
beta = AeroBeta(500.0, 0.785, FlowRegimeCd(AeroShape.SPHERE), reference_length_m=1.0)
cd = ballistic.aero.free_molecular_cd(AeroShape.SPHERE, speed_ratio, reemission_ratio)
```

| Name | What it is |
|------|------------|
| `ballistic.aero.AeroBeta` | mass, reference area and length, and a Cd model: a beta that `make_drag_environment` accepts |
| `ballistic.aero.FlowRegimeCd` | Cd from free-molecular to continuum flow, for a shape |
| `ballistic.aero.ConstantCd` | a Cd that does not vary |
| `ballistic.aero.AeroShape` | SPHERE, FLAT_PLATE, CYLINDER |
| `ballistic.aero.MachTable` | measured continuum Cd versus Mach number |
| `ballistic.aero.free_molecular_cd` | Sentman free-molecular Cd of a shape |
| `ballistic.aero.reemission_ratio` | speed of re-emitted molecules over their arrival speed |
| `ballistic.aero.henderson1976_sphere_cd` | Henderson's (1976) sphere Cd correlation |

---

## `ballistic.readers`: orbit products and velocity checks

`from ballistic.readers import ...`. ESA EOF orbit files (Sentinel precise orbits), scoring a
propagation against one, and checks on a product's velocity records.

```python
from ballistic.readers import parse_eof_auto, compute_propagation_result
eof = parse_eof_auto("S1A_OPER_AUX_POEORB_....EOF")
sv = eof.find_nearest(epoch)            # EOFStateVector (Earth-fixed, UTC)
```

| Name | What it is |
|------|------------|
| `ballistic.readers.parse_eof_auto` | read an ESA EOF orbit file (unpacks `.EOF.zip`) |
| `ballistic.readers.find_and_parse_eof` | find and read the first matching EOF file in a directory |
| `ballistic.readers.EOFParser` | the EOF reader |
| `ballistic.readers.EOFFile` | a parsed EOF file (`find_nearest`, `get_range`, `interpolate_position`) |
| `ballistic.readers.EOFStateVector` | one Earth-fixed state from an EOF file |
| `ballistic.readers.compute_propagation_result` | error statistics of a propagated trajectory against an EOF file |
| `ballistic.readers.PropagationTestResult` | those statistics |
| `ballistic.readers.check_velocity_records` | are a product's velocity records the true derivative of its positions? |
| `ballistic.readers.VelocityRecordCheck` | its result |
| `ballistic.readers.VelocityRecordVerdict` | its verdict |
| `ballistic.readers.cip_drift_velocity` | the Earth-axis drift term of an Earth-fixed velocity |

#### Ocean tides and velocity conventions

- `ballistic._core.load_ocean_tides(path, max_degree=100)` /
  `ocean_tides_loaded()` / `unload_ocean_tides()` — the process-wide
  ocean-tide store (C++: `gravity::OceanTideModel::instance()`).
- `ballistic.readers.cip_drift_velocity(epoch, pos_ecef, itrf=ITRF2020)` — the ECEF
  velocity contribution of the CIP drift (precession-nutation, ~2e-5 m/s at
  LEO; C++ `coords::cip_drift_velocity`). The ECEF↔GCRS velocity transform
  includes it, so start velocities must be the true time
  derivative of the ECEF position. Measured: ESA Sentinel EOF and GFZ
  CHAMP/GRACE-FO SP3 records are (use as written); ILRS SLR records differ
  from their derivative by about half this term (derive from positions).
  Check a product by differentiating its positions before correcting it.
- `ballistic.readers.check_velocity_records(epoch0, offsets_s, positions, velocities, itrf=ITRF2020)`
  — C++ `coords::check_velocity_records` (`coordinates/velocity_check.hpp`).
  Differentiates the positions (9-point Lagrange, interior samples only) and
  compares with the records; returns `VelocityRecordCheck` (`verdict`:
  `TRUE_DERIVATIVE` / `OMITS_CIP_DRIFT` / `INCONSISTENT`, `summary`,
  `median_difference_rtn`, `rms_difference`, `mean_cip_drift_along`,
  `standard_error_along`, `n_compared`). Pass exact offsets from the
  product's own time tags (an SP3 arc's `offsets_s`).

---

## `ballistic.tle`: TLEs and SGP4

C++: `include/ballistic/sgp4/`. Names: `ballistic.tle.parse`, `ballistic.tle.read`,
`ballistic.tle.Tle` (one element set), `ballistic.tle.Sgp4` (the propagator),
`ballistic.tle.tle_checksum` (a line's modulo-10 checksum),
`ballistic.tle.teme_to_itrf` / `ballistic.tle.itrf_to_teme`, `ballistic.tle.fit_state`,
and the TLE-grade force preset `ballistic.tle.tle_grade_preset`, which returns a
`ballistic.tle.TleGradePreset`.

```python
tle = ballistic.tle.parse(line1, line2)          # strict: 69 columns, checksums, Alpha-5
for name, tle in ballistic.tle.read("gp.txt"):   # 2LE / 3LE text (path or text)
    ...
m = ballistic.tle.Sgp4(tle, gravity="wgs72", mode="improved")   # or "afspc"
r_km, v_km_s, status = m.propagate_minutes(90.0)   # TEME; status: SGP4 error code (0 ok, 6 decayed)
r_m, v_m_s = m.teme(epoch)                        # raises when SGP4 fails there
ballistic.tle.teme_to_itrf(epoch, r_m, v_m_s)     # GMST 1982 of UT1 + polar motion (EOP lookup)
state = ballistic.State.from_tle(line1, line2, epoch=None)   # SGP4 state, as an ITRF State
state = ballistic.State.from_teme(epoch, r_m, v_m_s)
fitted, rms = ballistic.tle.fit_state(tle, cfg, span_s=86400.0)   # GCRS State fitted to SGP4
cfg = ballistic.recommended_config(state, accuracy="tle", horizon_s=..., beta_kg_m2=... | tle=tle)
p = ballistic.tle.tle_grade_preset(state, horizon_s)  # the measured choice, effects [m], speed-up
```

`tle.beta_from_bstar_kg_m2()` = 1 / (12.741621 B*). The preset raises
`ConfigError` outside what was measured (perigee below 300 km, GTO/HEO,
MEO below 5000 km, horizons beyond 7 days). C++: `sgp4::parse_tle`,
`sgp4::Sgp4`, `sgp4::teme_to_itrf`, `InitialState::from_tle` / `from_teme`,
`presets::tle_grade(state, horizon_s)`. Measured figures: manifest rows
`SGP4-VER`, `TLE-PRESET`, `TLE-GRACEFO`.

---

## `ballistic.sp3`: SP3 orbit files

`ballistic.sp3.read`, `ballistic.sp3.initial_state` (a checked start state from a file;
also `ballistic.State.from_sp3`), `ballistic.sp3.classify_frame`, and the types
`ballistic.sp3.SP3File`, `ballistic.sp3.SP3Arc`, `ballistic.sp3.SP3Header`,
`ballistic.sp3.SP3TimeSystem`, `ballistic.sp3.SP3FrameInfo` (what `classify_frame` returns).

- `ballistic.sp3.read(path)` — SP3 orbit files (a–d, plain or .gz); C++
  `validation::sp3::read` / `read_string` (`validation/sp3.hpp`). Returns
  `SP3File` (`header`: `frame`, `time_system`, `interval_s`, ...; `arcs`;
  `arc(satellite)`; `frame()` → `inertial`, `itrf`). An `SP3Arc` is one
  satellite in one file: `epoch0` (in the library's scales — the file's %c
  time system is honored: GPS, GAL, QZS, IRN, BDT, TAI, UTC, GLO), exact
  `offsets_ns` / `offsets_s`, `position` (N, 3) m, `velocity` (N, 3) m/s or
  NaN, `interpolate(t_s, order=9)` (range-checked, never across a gap),
  `derived_velocities()` → (velocity, centered flags; do not start an arc on
  an uncentered one), `segments()`. "Bad or absent" positions are not records
  (`missing_positions`); a malformed line raises naming it.
  `ballistic.sp3.classify_frame(label)` maps a coordinate-system label to
  inertial / Earth-fixed and its ITRF realization; unknown labels raise
  `FrameError`. Campaign scripts read through `scripts/_sp3_records.py`.

---

## `ballistic.events`: events during propagation

Built-in events for `Trajectory.propagate_until(target, events,
stop_at_ground=True)`; each is located inside the integration step.
Python callables are not accepted.

| Name | What it is |
|------|------------|
| `ballistic.events.altitude_crossing(altitude_m, direction=Either, terminal=True)` | the geodetic altitude crosses a value (`ballistic.events.Direction`: Rising, Falling, Either) |
| `ballistic.events.ground_impact()` | descending through the WGS84 ellipsoid (terminal; armed by default) |
| `ballistic.events.perigee(terminal=False)` / `ballistic.events.apogee(terminal=False)` | the lowest / highest point of the orbit |
| `ballistic.events.at_epoch(epoch, terminal=False)` | a given time, to log the state there |
| `ballistic.events.Event`, `ballistic.events.Occurrence` | an event, and one found occurrence (in `PropagateUntilResult.log`) |

---

## `ballistic.constants`

Physical constants grouped as `ballistic.constants.earth`, `.sun`, `.moon`
and `.jupiter` (the same values as C++ `constants.hpp`; WGS84 for the
Earth). The geodetic conversions `ballistic.ecef_to_geodetic` /
`ballistic.geodetic_to_ecef` are top level. The C++ values are listed in
[Constants](#constants).

---

## External Data Files

| File | Format | Size | Purpose | Download |
|------|--------|------|---------|----------|
| `finals2000A.data` / `.all` | IERS ASCII | ~2.5 / 3.7 MB | Polar motion, UT1-UTC (required: transforms raise without it) | [IERS](https://datacenter.iers.org/data/9/finals2000A.data) |
| `leap-seconds.list` | NTP text | ~10 KB | Leap seconds after the embedded table | [IERS](https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list) |
| `de421.bsp` | SPICE DAF | ~17 MB | JPL ephemeris (ESA standard) | [JPL NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/) |
| `de440.bsp` | SPICE DAF | ~115 MB | JPL ephemeris (embedded DE440 covers 1950-2100) | [JPL NAIF](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/) |
| `SW-All.csv` | CelesTrak CSV | ~3 MB | Space weather (F10.7, Ap) for drag | [CelesTrak](https://celestrak.org/SpaceData/SW-All.csv) |
| `fes2004_Cnm-Snm.dat` | IERS text | ~3.7 MB | FES2004 ocean tides | [IERS Conventions Ch. 6](https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat) |

**Note**: The library includes embedded EGM2008 (degrees 2-360), DE440 ephemeris (1950-2100) and a leap-second table. EOP must be loaded (`ballistic.load()` or a file in the data directory); the other external files are needed for:
- Drag (space weather) and ocean tides
- Higher gravity degrees (361-2190)
- Dates outside 1950-2100

Where each file is looked for: [`data_files.md`](data_files.md).

---

## DragEnvironment: unified drag / wind / β table

Drag is configured by a single optional `TrajectoryConfig::drag_env` field
(a `shared_ptr<const atmosphere::DragEnvironment>`; null = drag off). There
are no separate drag flags, β fields or atmosphere-model setters.

`make_drag_environment(t0, pos_ecef, vel_ecef, duration_s, base_config,
DragSetup{.beta, .weather, .allow_predicted_weather})` returns a live
environment (`shared_ptr<const DragEnvironment>`) that evaluates NRLMSIS 2.1
and HWM14 at the satellite's position on every force evaluation (the
production default); `beta` is a `BetaProfile1D` or an `aero::AeroBeta`
(flow-regime Cd for reentry). A `Trajectory` stops with an error at the
edge of the loaded weather and warns once on entering forecast days.
Tabulated environments come from the generators in
`atmosphere/drag_environment_generators.hpp` (wrap with `share(...)`):

- **`from_models_snapshot(t, ref_lat_deg, ref_lon_deg, sw, beta_prior,
  altitude_grid_km = {})`** — 1D altitude profile at a fixed (lat, lon, t).
  Use for short arcs (≪ 1 orbit).
- **`from_models_along(sample_epochs, sample_positions_ecef, sw,
  beta_prior, altitude_grid_km = {}, diag_out = nullptr)`** —
  2D (altitude × epoch) sampled at each
  trajectory point's sub-sat (lat, lon, t). Multi-orbit arcs.
  `from_models_along_iterated(epoch, pos, vel, duration_s, base_config, sw,
  beta_prior, ...)` propagates and rebuilds up to `max_iter = 3` times until
  the trajectory changes by less than 1%.
- **`from_models_4d(alt_grid, lat_grid, lon_grid, epoch_grid, sw,
  beta_prior)`** — full 4D field (heavy; OpenMP over the epoch dim).
  Use for constellation studies or shared atmospheres.
- **`from_accelerometer_density(sample_epochs, measured_density_kg_m3,
  beta_prior, altitude_grid_km = {})`** — accelerometer-
  derived ρ samples. Builds an altitude-degenerate `DensityProfile2D`
  (each epoch row holds the measured ρ across all altitudes) + empty
  wind column. Canonical use: GOCE-style low-LEO reentry with TU
  Delft accelerometer density (parse the `.dat` files via
  `ballistic.drag.parse_density_table(path)` and convert MJD → Epoch via
  `Epoch.from_mjd(mjd, TimeScale.UTC)`).

The first three generators take a `SpaceWeatherHistory` — normally the
MEASURED one, `SpaceWeatherData::instance().history(TimeWindow{t0, t1})`
(Python `SpaceWeatherHistory.from_loaded(t0, t1)`): previous-day observed
F10.7, 81-day centered mean, 3-hourly Ap in NRLMSIS's AP-array order; it
throws for days without observed Ap (predicted rows). This is what
`make_drag_environment` uses when `DragSetup::weather` is unset — and it
throws if no space weather is loaded (there is no quiet-weather fallback).
`SpaceWeatherHistory::from_constant(WeatherConditions[, TimeWindow])` is for
deliberate scenarios (quiet / storm studies) — and a `BetaProfile1D` (build with
`BetaProfile1D::constant(beta_kg_m2)` for constant β, or pass altitude /
beta arrays for an altitude-dependent table).
`from_accelerometer_density` skips `SpaceWeatherHistory` since the
measured ρ already encodes weather.

**`extend_density_below(env, floor_km = 80)`** adds the U.S. Standard
Atmosphere 1976 (0-86 km, `atmosphere/ussa76.hpp`) below the floor of a
tabulated environment (deep reentry; a live environment already runs
NRLMSIS to the ground).

**`DragEnvironment::without_wind()`** returns a copy of the
env with the wind column replaced by an empty `WindProfile1D_ENU`
(stationary atmosphere). Density and beta are preserved. Use for
with-wind vs without-wind comparisons (see
`scripts/validate_wind_effect.py`).

Storage: 1D and 2D density profiles hold `log(ρ)` as doubles; the 4D field
stores `log(ρ)` as IEEE 754 binary16 bit patterns (`atmosphere::half`, converted
by `float_to_half` / `half_to_float`; round-trip error in ρ up to 0.4% / 0.8% /
1.6% for |log ρ| in [8,16) / [16,32) / [32,64), i.e. ~0.8% in most of LEO). Altitudes are geodetic (WGS84,
in km).

Interpolation: cubic spline in log(ρ) over altitude, bilinear in latitude
and longitude, linear in time. Past a table's time span the atmosphere
freezes and a `Trajectory` warns once; edge behavior of every table is in
[`out_of_range_policy.md`](out_of_range_policy.md).

---

## Accuracy Summary

Measured 1-hour prediction RMSE; each row cites its entry in
[`validation_claims.md`](validation_claims.md), which records how it was
measured and how to reproduce it.

| Regime | Satellite | 1-Hour RMSE | Configuration |
|--------|-----------|-------------|---------------|
| LEO | Sentinel (6 sats) [`SENTINEL-JAN26-SAMPLED`] | **0.239 m** | Recommended (SRP + drag + ocean tides); 0.218 m without drag, 0.540 m with neither |
| LEO | Sentinel, 7 sats, all of 2025 [`SENTINEL-YEAR-2025`] | **0.215 m** | Recommended configuration |
| LEO | Sentinel, Jan 2026 out of sample [`SENTINEL-YEAR-JAN26`] | **0.240 m** | Recommended, vs 0.570 m bare defaults |
| LEO | Sentinel, Sep 2026 out of sample [`SENTINEL-YEAR-SEP26`] | **0.224 m** | Recommended, vs 0.502 m bare defaults |
| LEO | CHAMP, 370 km [`CHAMP-370KM`] | **1.17 m** | + drag (fitted β), −81% vs no drag |
| LEO | GRACE-FO, 500 km [`GRACEFO-500KM`] | **0.47–0.87 m** | + drag (physical β), −80 to −88% vs no drag |
| MEO (SLR) | LAGEOS-1/2 + Etalon-1/2 [`SLR-1H`] | **0.152 m** | GCRS + DE440 + gravity degree 8 (0.148 m with SRP) |
| MEO (GNSS) | GPS, 31 sats [`GNSS-1H`] | **0.668 m** | Library defaults, SRP off |
| IGSO | QZSS J02/J03/J04 [`GNSS-1H`] | **0.739 m** | Library defaults, SRP off |
| GEO | BeiDou C01-C05 [`GNSS-1H`] | **0.805 m** | Library defaults, SRP off |

Truth orbits: ESA/CNES Sentinel POE, GFZ CHAMP/GRACE-FO rapid orbits, ILRS
LAGEOS/Etalon SP3, CODE MGEX (GPS/QZSS) and WHU MGEX (BeiDou) SP3.

---

## Constants

Physical constants are in the `ballistic::constants` namespace (Python:
`ballistic.constants`, with `earth` / `sun` / `moon` / `jupiter`
sub-namespaces); default values are in `ballistic::defaults`.

### Mathematical

| Constant | Value | Description |
|----------|-------|-------------|
| `PI` | 3.14159... | Pi |
| `TWO_PI` | 6.28318... | 2π |
| `DEG_TO_RAD` | 0.01745... | Degrees to radians |
| `RAD_TO_DEG` | 57.2957... | Radians to degrees |
| `ARCSEC_TO_RAD` | 4.848e-6 | Arcseconds to radians |

### Time

| Constant | Value | Description |
|----------|-------|-------------|
| `SECONDS_PER_DAY` | 86400.0 | Seconds per day |
| `J2000_JD` | 2451545.0 | J2000.0 Julian Date |
| `MJD_OFFSET` | 2400000.5 | MJD = JD - MJD_OFFSET |
| `UNIX_EPOCH_JD` | 2440587.5 | Unix epoch Julian Date |
| `TT_TAI_OFFSET` | 32.184 | TT - TAI (seconds) |
| `GPS_TAI_OFFSET` | -19.0 | GPS = TAI - 19 (seconds) |

### Earth (WGS84)

| Constant | Value | Units | Description |
|----------|-------|-------|-------------|
| `earth::GM` | 3.986004418e14 | m³/s² | Gravitational parameter |
| `earth::RADIUS_EQUATORIAL` | 6378137.0 | m | Equatorial radius |
| `earth::RADIUS_POLAR` | 6356752.3142 | m | Polar radius |
| `earth::FLATTENING` | 1/298.257223563 | - | Flattening |
| `earth::ANGULAR_VELOCITY` | 7.292115e-5 | rad/s | Rotation rate |
| `earth::J2` | 1.082626173852223e-3 | - | J2 zonal harmonic (−C̄20·√5 of the embedded EGM2008; J3..J15 likewise) |

### Celestial Bodies

| Constant | Value | Units | Description |
|----------|-------|-------|-------------|
| `sun::GM` | 1.32712440041279419e20 | m³/s² | Sun gravitational parameter (JPL DE440) |
| `sun::AU` | 1.495978707e11 | m | Astronomical unit (IAU 2012, exact) |
| `moon::GM` | 4.9028001184575496e12 | m³/s² | Moon gravitational parameter (JPL DE440) |
| `jupiter::GM` | 1.26712764e17 | m³/s² | Jupiter system barycenter GM (JPL DE440) |
| `SPEED_OF_LIGHT` | 299792458.0 | m/s | Speed of light |

### Default Values (`ballistic::defaults`)

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_GRAVITY_DEGREE` | 70 | Default EGM2008 degree |
| `RK45_INITIAL_STEP_SIZE` | 60.0 | Initial RK45 step (seconds) |
| `RK45_MIN_STEP_SIZE` | 0.001 | Minimum step (seconds) |
| `RK45_MAX_STEP_SIZE` | 3600.0 | Maximum step (seconds) |
| `SYMPLECTIC_DEFAULT_STEP_SIZE` | 30.0 | Symplectic step (seconds) |
| `ATMOSPHERE_ALTITUDE_THRESHOLD_M` | 1000000.0 | Max atmosphere altitude (m) |

---

## Version

```cpp
// C++
#include <ballistic/ballistic.hpp>

int major = ballistic::VERSION_MAJOR;  // 0
int minor = ballistic::VERSION_MINOR;  // 1
int patch = ballistic::VERSION_PATCH;  // 2
std::string version = ballistic::VERSION_STRING;  // "0.1.2"
```

```python
# Python
import ballistic

version = ballistic.__version__  # "0.1.2"
major = ballistic.VERSION_MAJOR
minor = ballistic.VERSION_MINOR
patch = ballistic.VERSION_PATCH
```
