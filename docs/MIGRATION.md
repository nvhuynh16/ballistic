# Migration guide

This release removes or changes APIs that made it easy to
get a wrong answer without noticing: a function name that does not say its
frame, or a time-scale default that silently assumes TT. Nothing is removed
silently. Using a removed Python name raises `ballistic.RemovedApiError`,
and the message names the replacement. Each change is listed below with the
reason for it.

| Removed or changed | Use instead | Why |
|--------------------|-------------|-----|
| NRLMSIS 2.1 / HWM14 compiled into the library and the wheel | The plugin `ballistic_nrl`: Python `ballistic.install_nrl_models(accept_license=True)` (offline: `source=` a downloaded file); C++ the CMake target `ballistic_nrl` plus `InitConfig::nrl_models_path`, `BALLISTIC_NRL_MODELS` or `<data directory>/nrl/` | NRLMSIS 2.1 is licensed for non-commercial use only; a fresh installation now contains no NRL code, and model-based drag raises `ConfigError` with these instructions until the plugin is installed |
| `recommended_config` always turning drag on below 1500 km | The same once the plugin is installed; without it drag is off with one `ConfigHintWarning`. `drag=True` requires it, `drag=False` leaves drag off silently | Drag needs the separately installed NRL models |
| Python `LicenseWarning` and `LicenceWarning` (top level); C++ `diagnostics::Category::License` / `Licence`, `atmosphere::report_nrlmsis_license()` / `report_nrlmsis_licence()` | Nothing at use time: `install_nrl_models` shows the license terms before installing | The license is accepted once, when the plugin is installed |
| CMake option `BALLISTIC_WITHOUT_NRL`; `build_info().with_nrl` (Python `build_info()["with_nrl"]`); `atmosphere::kWithNrlModels` | The library never contains NRL code; `-DBALLISTIC_BUILD_NRL_PLUGIN=OFF` skips building the plugin; `ballistic.nrl_models()` / C++ `atmosphere::nrl::status()` report whether it is loaded | Whether the models are available is decided at run time, by the plugin |
| C++ `sp3::Arc::centred()` and the `centred` field of `derived_velocities()`'s result | `sp3::Arc::centered()` and `centered` | The library uses American spelling throughout; behavior is unchanged |
| `Epoch(jd)`, `Epoch.from_jd(jd)`, `Epoch.from_mjd(mjd)` without a scale (C++ and Python) | `Epoch.from_jd(jd, TimeScale.UTC)` etc., or a calendar constructor | The default was TT: a UTC Julian date read as TT is 69 s (~500 km at LEO) off |
| Propagating or transforming with no Earth orientation data loaded (used to run with zeros) | `ballistic.load()` (or C++ `initialize` with finals2000A); to accept zeros on purpose, `ballistic.set_eop_policy('assume_zero')` / `EOPData::set_policy(EopPolicy::AssumeZero)` | Zeros cost 3–10 m at LEO, and nothing said so in C++ |
| Dates more than 30 days outside the loaded Earth orientation table (used to run with zeros) | Load a newer finals2000A; a `Trajectory` now stops at the limit and keeps its state | UT1−UTC was silently taken as zero there: up to 0.9 s, hundreds of meters |
| UTC epochs before 1961 | `Epoch.from_tt` / `from_tai` | UTC did not exist; TAI−UTC was a guessed 10 s (1961–1972 now use the real values) |
| `LeapSecondTable::set_warning_callback` / `clear_warning_callback` (C++) | The diagnostics channel (`diagnostics::set_handler`; Python warnings) | The callback was never set, so the expiry warning could not fire |
| Python `ballistic.eci_to_ecef` / `ecef_to_eci` | `ballistic.gcrs_to_itrf` / `itrf_to_gcrs` | Their "ECI" was the true equator of date, not J2000/GCRS: ~22 km apart at LEO in 2013 |
| `Trajectory.state_eci()` (C++ and Python), `Propagator::current_state_eci()` | `state_gcrs()`, `current_state_gcrs()` | Same state; the name now says the frame |
| C++ `coords::ecef_to_eci*`, `eci_to_ecef*`, `ecef_to_eci_rotation`, `eci_velocity_to_ecef`, `TransformAccuracy` (header `coordinates/ecef_eci.hpp`) | For GCRS: `coords::itrf_to_gcrs_full` / `gcrs_to_itrf_full` / `gcrs_to_itrf_rotation` / `itrf_to_gcrs_rotation` (`iau_transforms.hpp`). The true-of-date functions were removed from the library; the GCRS transforms (`util::itrf_to_gcrs`, `coords::gcrs_to_itrf_rotation`) replace them. GMST/GAST, polar motion and `convert_itrf` are in `coordinates/earth_rotation.hpp` | The old names hid a true-of-date frame behind "ECI"; three library call sites used it as GCRS |
| C++ `util::ecef_to_eci` / `eci_to_ecef` / `acceleration_to_ecef` / `acceleration_to_eci` and their ignored `PropagationFrame` argument | `util::itrf_to_gcrs` / `gcrs_to_itrf` / `acceleration_to_itrf` / `acceleration_to_gcrs` (no frame argument) | These were always GCRS; the old names clashed with the true-of-date ones |
| `scripts/validate_goce_reentry.py --frame-transform` | — | The old GAST-of-date conversion no longer exists |
| Python `load_density_table`, `load_interpolated_density`; C++ `atmosphere::InterpolatedDensityModel` / `InterpolatedDensityData` (header `atmosphere/interpolated_density.hpp`) | `ballistic.drag.parse_density_table` + `from_accelerometer_density` (a `DragEnvironment`) | The global store they filled was read by no propagation: loading measured density changed nothing |
| C++ `SpaceWeatherData::f107_daily` / `f107_average` / `ap_daily` / `ap_history_for_nrlmsis` | `SpaceWeatherData::get_record` (exact day, nullopt outside the file) or `get_conditions` / `history()` | They returned F10.7 150 / Ap 4 with nothing loaded and the nearest row outside the file |
| `SpaceWeatherData::get_record` returned the nearest row | Same name; now the row of the UTC day containing the MJD, or nullopt | Outside the file it returned the edge row; at exactly 00:00 UTC it returned the previous day |
| A space-weather CSV without `F10.7_OBS` / `AP_AVG` loaded as all-quiet weather | — (throws) | Density would have been biased low everywhere |
| `gravity_degree` / `ocean_tide_degree` above the loaded model, `EGM2008Gravity::init(n > 360)`, zonal degree > 15 were clamped | Ask for at most the loaded degree, or load the NGA EGM2008_to2190 file (`InitConfig.egm2008_path`, which now takes effect) | A different model ran than the one requested; `ConfigError` now |
| Non-finite gravity fell back to point mass | — (throws) | It hid a diverged state |
| `ballistic.drag.load_beta_table` without a header; `load_empirical_acceleration` with an unknown `# frame:` or no rows; `parse_density_table` on the MJD-first layout or a row that does not parse | Give a header (`altitude_km,beta`); name the frame (`gcrs` / `itrf`); use the v2 TU Delft layout | Each used to assume, default (GCRS, a zero EA) or skip silently; they raise `ValueError` now. `parse_density_table` honors the file's time-system column |
| `JPLEphemeris::get_body_state` returned zeros / clamped the date | — (throws; `DataCoverageError` outside the file) | A zero Sun or Moon is a force-model error with no symptom |
| `scripts/_drag_env_helper.get_weather_for_epoch` fell back to quiet weather | — (raises) | The fallback can mask a 4.6x β error |
| C++ `validation/sp3_parser.hpp` (`SP3Parser`, `SP3File`, `SP3Record`, `compute_position_rmse`, `compute_velocity_rmse`, `ValidationResult`) | `validation/sp3.hpp`: `sp3::read(path)` / `read_string(text)` → `File` with one `Arc` per satellite (`epoch0`, exact `offset_ns`, `position_m`, `velocity_m_s` NaN when absent, `interpolate(t)`, `derived_velocities()`, `segments()`), `File::frame()`. Python: `ballistic.sp3.read(path)` (plain or .gz) | The old parser read GPS-time files as UTC, kept its epoch in a static shared across threads, accepted 0.000 positions and extrapolated |
| `math::lagrange_interpolate_vector` / `lagrange_derivative_vector` (C++) | `math::lagrange_interpolate` / `lagrange_derivative` (range-checked: `DataCoverageError` outside the data); the raw templates are `math::detail::` | They extrapolated silently |
| `EOFFile::interpolate_position` outside the state vectors; an EOF without a validity period | — (returns nothing outside; a missing validity period raises) | It extrapolated on `utc_jd` doubles; a missing validity period read as J2000 |
| `scripts/_sp3_frames.py` (`classify_sp3_frame`) | `ballistic.sp3.classify_frame(label)` (C++ `sp3::classify_frame`) → `inertial`, `itrf` | One frame table, in the library |
| `scripts/validate_goce_reentry.py --sp3-time` | — | Epochs always follow the file's time system |
| `splice_isa_density_below` (C++ and Python; `env, isa_floor_km`) | `extend_density_below(env, floor_km)` (0 < floor_km <= 86) | It only replaced existing table knots below the floor — none on the standard 100–1500 km grid, so it changed nothing — and its "US Standard Atmosphere" was a piecewise exponential (3.5% high at 60 km). The new function adds U.S. Standard Atmosphere 1976 knots from 0 km and leaves the wind unchanged |
| `atmosphere/internal/exponential.hpp` (`ExponentialAtmosphere`, `HarrisPriesterAtmosphere`, `USStandardAtmosphere`) | `atmosphere/ussa76.hpp` (`ussa76::evaluate` / `density`, 0–86 km); NRLMSIS 2.1 for everything else | Internal comparison models; the "US Standard" one was mislabeled |
| `propagate_to_altitude(..., max_steps=100)` (C++, Python single and batch) | `propagate_to_altitude(..., max_duration_s=7 days)` | It gave up after 100 steps (~1.5 h) with `condition_met = False`; forward it now ends AT the crossing (located inside the step) |
| The live drag env used USSA76 below 0 km by default | — (NRLMSIS to the ground; below −1 km raises) | A 4.5% density jump at the ground stalled the integrator |
| Python `Trajectory(epoch, position, velocity)` and the batch form `Trajectory([(epoch, pos, vel), ...])` | `Trajectory(State.from_itrf(epoch, r_m, v_m_s))` for an Earth-fixed velocity, `State.from_gcrs` / `from_j2000` for an inertial one, `State.from_keplerian`, `State.from_sp3(file, satellite, epoch)`; batch: `Trajectory([state1, state2])`. C++ keeps `Trajectory(epoch, pos_itrf, vel_itrf)` and adds `Trajectory(InitialState)` (`initial_state.hpp`) | The triple did not say its frame; an inertial velocity read as Earth-fixed is ~470 m/s off at LEO, and examples here made that mistake. A `State` is checked (units, below ground, suborbital, the frame mix-up) |
| `recommended_config(epoch, pos, vel, satellite, ...)` | `recommended_config(state, satellite, ...)`; the state's ITRF realization becomes the config's | Same reason |
| Python `Trajectory.propagate(target)` | `propagate_for(seconds)` (from the current epoch) or `propagate_to(epoch)`; `propagate_until`'s float target is now seconds from the current epoch | A float meant seconds from the INITIAL epoch, so calling it twice did not advance twice; batch mode ran member 0's duration for every member and overshot each target by up to a step |
| Mutating a Python `TrajectoryConfig` (`cfg.drag_env = env`) | `cfg = cfg.replace(drag_env=env)` (or `dataclasses.replace`); construct with keywords | It is frozen, slotted and keyword-only now: a misspelt field (`cfg.gravity_degre = 20`) used to be ignored silently |
| `SrpConfig()` (C++ and Python) defaulted to 1000 kg / 10 m² | Give `mass_kg` and `area_m2` (Python: keyword-only and required; C++: a `Trajectory` raises `ConfigError` until both are set) | A generic area-to-mass ratio is 20x wrong for some satellites |
| C++ `TrajectoryBatch::propagate_all` stepped past its target | Lands exactly; `propagate_all_to(epoch)` added | Each member overshot by up to one step |
| C++ `estimate_ea_collocation` (all overloads), `EACollocationConfig`, `CollocationBSplineConfig`; the radar overload of `estimate_ea_bspline` | `estimate_ea_bspline` (with the default `gn = GNRefine{}`) for dense single-modality data (position, velocity, acceleration); `estimate_ea_smoother` for radar, sensor fusion and noisy data | Each was measured worse than one of these two in every scenario: clean position 0.1 m vs 0.006 m for B-spline + GN; noisy position floored at 15–19 m whatever the knots and regularization |
| C++ `estimate_ea_multisensor*` (six functions), `structured_seed_grid`, `MultiSensorConfig` (with its profiles `radar_with_range_rate`, `clean_data`, `accelerometer_dominant`), `MultiSensorResult`, `MultiSensorCVResult`, `MultiSensorTermination`, `AdaptiveRefinement`, `SeedGridSpec` | `estimate_ea_smoother(scenario, mask, trajectory_config, EASmootherConfig{})` (`calibration/ea_smoother.hpp`) with the same `MultiSensorScenario`, `SensorMask` and sensor types; it returns `EASmootherResult` (EA in GCRS, posterior `ea_sigma`) | The batch fit needed seeds and continuation ladders, and its stopping point was set by floating-point noise; on the radar + impact-fix glide case the smoother reaches 12.7 m against 15.4–141 m, in about 1/40 of the time |
| C++ headers `coordinates/legacy_tod.hpp` (`coords::itrf_to_tod*` / `tod_to_itrf*`), `atmosphere/altitude_convert.hpp` (geometric ↔ geopotential altitude) and `validation/champ_accelerometer_parser.hpp` | For an inertial frame, the GCRS transforms (`coords::itrf_to_gcrs_full` / `gcrs_to_itrf_full`, `util::itrf_to_gcrs` / `gcrs_to_itrf`); for altitude, `coords::ecef_to_geodetic` (geometric, which every model in the library takes); no replacement for the CHAMP accelerometer reader | Nothing in the library used them: the true-of-date frame is not GCRS and is banned in the library, no model reads geopotential altitude, and the uncalibrated CHAMP accelerometer file cannot check drag |
| C++ `egm2008_acceleration(pos, degree)` / `egm2008_potential(pos, degree)` | `gravity::EGM2008Gravity::instance().acceleration(pos, degree)` / `.potential(pos, degree)` | One-line wrappers that nothing called |
| `GNStatus::SkippedRadarNoPositionTargets` (C++ and Python `ballistic.estimation.GNStatus`) | — (compare with `Disabled`, `Ran`, `SkippedAccelerationCircular`); radar data goes to `estimate_ea_smoother`, which has no GN step | Only the removed radar overload of `estimate_ea_bspline` returned it; a status that can never occur invites dead branches. Python raises `AttributeError` on the old name |
| C++ `SymplecticConfig::drag_correction_interval`, `SymplecticIntegrator::Stats::initial_energy` / `current_energy` | — (`SymplecticPropagator::initial_energy()` / `current_energy()` / `energy_drift()` give the energies) | Nothing read the interval and nothing filled the two counters (always 0) |
| `SymplecticPropagator::set_step_size` / `Trajectory::set_step_size_hint` on a symplectic trajectory only changed the reported step | Same names; they now set the step used from the next step on (a non-positive step raises `ConfigError`) | The integrator kept its old step until a reset, and a symplectic `Trajectory` kept its configured step |
| `JPLEphemeris::sun_position` (legacy binary DE reader) returned the Sun relative to the solar-system barycenter | Same name; now geocentric (Sun − Earth, Earth = EMB − Moon/(1 + EMRAT)) | It was off by about 1 AU from what every caller of an `EphemerisProvider` expects |
| `ClenshawEngine::compute` with no coefficients loaded returned potential −GM/r | Same name; now +GM/r, the series' convention | The fallback and the series disagreed in sign |
| `gravity::total_potential_with_j2` added −GM/r to a J2 term of the opposite convention | Same name; now U = GM/r · [1 − J2 (Re/r)² P₂(sin φ)], with a = +grad U | The sum was the potential of neither convention |
| Exceptions: `get_space_weather` / `SpaceWeatherData::get_conditions` outside the file or with nothing loaded raised `IndexError` (C++ `std::out_of_range`); `check_velocity_records` and the C++ math tables (`CubicSpline*`, `Hermite*`, `LinearVector3d`) `ValueError` (`std::invalid_argument`); `make_clamped_knot_vector` `std::invalid_argument`; the ephemeris readers and RK45's evaluation budget `std::runtime_error` | `ballistic.DataCoverageError`, `InputPlausibilityError`, `ConfigError`, `DataCoverageError` and `BallisticError` (C++ `ballistic::Error` and subclasses) respectively | One error hierarchy that says what kind of problem it is; `catch (std::runtime_error&)` still catches all of them in C++, but Python code catching `IndexError` / `ValueError` from these calls must change |
| A `leap-seconds.list` with a non-comment line that does not parse loaded the remaining lines | — (the file is refused: `load()` returns false, `last_error()` names the line; `initialize()` reports it) | A skipped row is a 1 s TAI−UTC error on every later date |
| `math::chebyshev_eval_with_derivative` returned a zero derivative for more than 64 coefficients | Same name; the derivative is computed for any length | A zero velocity without an error |
| C++ `coords::itrf_params::itrf2014_to_itrf2020`, `itrf2008_to_itrf2014`, `itrf2005_to_itrf2008`, `itrf2000_to_itrf2014` | `itrf2020_to_itrf2014`, `itrf2014_to_itrf2008`, `itrf2014_to_itrf2005`, `itrf2014_to_itrf2000` (the IERS tables' direction, newer to older; `transform_itrf` handles either direction) | The old names held the IERS rows, which go from the newer frame to the older, under older-to-newer names, so `transform_itrf` / `convert_itrf` and every realization conversion applied the shift with the wrong sign: positions move by 7–14 mm at LEO between ITRF2014 and ITRF2020, up to 4 cm for ITRF2000 |
| `coords::convert_itrf` was `noexcept` | Same function; it raises `DataCoverageError` for an epoch before 1961 when the realizations differ | The UTC decimal year it needs raises there, which terminated the program |
| `coords::gast()` (low-accuracy sidereal time) | Same function, now Meeus's four-term nutation (0.3″ from `gast_full`) | It was up to 3.1″ off (a wrong Sun-longitude rate and a misplaced term) |
| C++ `gravity::jn_acceleration(pos, n, Jn)` returned zero for n outside 2..15 (and was `noexcept`) | Same name; such a degree raises `ConfigError` (as `zonal_harmonics_acceleration` does) | A zero acceleration looks like a valid tiny term |
| C++ `forces::BSplineCoefficients` / `EmpiricalAcceleration::from_bspline` accepted any degree >= 1 | Degree 1 to `BSplineCoefficients::kMaxDegree` (7); above it `ConfigError` | The evaluator's fixed buffers hold degree + 1 <= 8 basis values; degree 8 or more overflowed them |
| `initialize()` with a legacy binary DE file after the SPK ephemeris had loaded kept the SPK | Same call; the SPK is now dropped so force models built afterwards use the file. A force model built before it raises `DataCoverageError` naming the file — build force models after `initialize()` | The file was loaded and then ignored |
| `ea::CV` with a block layout that holds out no block (`hv_n_blocks` < 4 with the automatic pattern, or `hv_test_blocks` outside `[0, hv_n_blocks)`) | Use `hv_n_blocks >= 4` or list valid held-out blocks | It fitted the first grid candidate without cross-validating; it raises `ConfigError` now |
| `ea::GCV{lam_min, lam_max}` searched 1e-12 to 0.1 whatever was given | Same type; the interval is now used (0 < lam_min < lam_max, else `ConfigError`) | The fields were never read |
| Under `ea::CV`, GN ignored `GNRefine::reg_lambda` / `reg_deriv_order`, and velocity data were refined with the third-derivative penalty | Same settings; they now reach the GN runs inside the search, and the automatic order is 3 for positions, 1 for velocities as outside CV | Velocity CV scored candidates after a different refinement than the final fit's documented one; results under `ea::CV` with velocity data change |
| `EAResult::collocation_history` after GN described the pre-GN EA; `gn_status` under `ea::CV` said `Disabled` | Same fields; after GN the history is re-propagated with the refined EA (one extra propagation), and `gn_status` is `Ran` when GN ran inside CV | `refine_beta_from_ea` pairs the history with the returned EA |

## Top-level names moved into modules

The top-level `ballistic` package now exports what a first session needs (83 names:
loading data, `Epoch`, `State`, `Trajectory` and its configuration, `recommended_config`,
`make_drag_environment`, `BetaProfile1D`, `SpaceWeatherHistory`, `EmpiricalAcceleration`,
frame conversions, errors and warnings). Specialized names moved into modules, so the top
level can be read in one sitting. Using an old name raises `ballistic.RemovedApiError`
naming its new import, for example

```
ballistic.estimate_ea_bspline moved to ballistic.estimation.estimate_ea_bspline:
use `from ballistic.estimation import estimate_ea_bspline`
```

Behavior is unchanged; only the import path moves. `ballistic.estimation.information_bound`
now accepts the `ballistic.TrajectoryConfig` dataclass as its force model, like
`estimate_ea_smoother`. The module `ballistic.estimators` is now `ballistic.estimation`.

| Module | What moved there |
|--------|------------------|
| `ballistic.estimation` | fitting an empirical acceleration: estimators, their configurations and results, sensor types, the EA-to-ballistic-coefficient tools, and the types `EmpiricalAcceleration`'s factories take (39 names) |
| `ballistic.drag` | atmospheres built by hand: `DragEnvironment` and its sample, the tabulated and measured-density generators, `extend_density_below`, space-weather and ballistic-coefficient value types, table readers, `.npz` save and load (17 names) |
| `ballistic.aero` | flow-regime drag coefficients for reentry (8 names) |
| `ballistic.readers` | ESA EOF orbit files, scoring a propagation against one, velocity-record checks (11 names) |
| `ballistic.tle` | the TLE-grade force preset, next to the TLE reader and SGP4 (2 names) |

Every moved name:

| Old | New |
|-----|-----|
| `ballistic.EmpiricalAccelEntry` | `ballistic.estimation.EmpiricalAccelEntry` |
| `ballistic.EAInterp` | `ballistic.estimation.EAInterp` |
| `ballistic.EARepresents` | `ballistic.estimation.EARepresents` |
| `ballistic.BSplineCoefficients` | `ballistic.estimation.BSplineCoefficients` |
| `ballistic.estimate_ea_bspline` | `ballistic.estimation.estimate_ea_bspline` |
| `ballistic.EABSplineConfig` | `ballistic.estimation.EABSplineConfig` |
| `ballistic.PositionSamples` | `ballistic.estimation.PositionSamples` |
| `ballistic.VelocitySamples` | `ballistic.estimation.VelocitySamples` |
| `ballistic.AccelSamples` | `ballistic.estimation.AccelSamples` |
| `ballistic.CVMode` | `ballistic.estimation.CVMode` |
| `ballistic.GNStatus` | `ballistic.estimation.GNStatus` |
| `ballistic.GNRefine` | `ballistic.estimation.GNRefine` |
| `ballistic.EAResult` | `ballistic.estimation.EAResult` |
| `ballistic.refine_ea_gn` | `ballistic.estimation.refine_ea_gn` |
| `ballistic.EADecomposition` | `ballistic.estimation.EADecomposition` |
| `ballistic.BetaStepEstimate` | `ballistic.estimation.BetaStepEstimate` |
| `ballistic.BetaResult` | `ballistic.estimation.BetaResult` |
| `ballistic.BetaDecompositionConfig` | `ballistic.estimation.BetaDecompositionConfig` |
| `ballistic.decompose_ea` | `ballistic.estimation.decompose_ea` |
| `ballistic.refine_beta_from_ea` | `ballistic.estimation.refine_beta_from_ea` |
| `ballistic.RadarObs` | `ballistic.estimation.RadarObs` |
| `ballistic.OpticsObs` | `ballistic.estimation.OpticsObs` |
| `ballistic.PositionFixObs` | `ballistic.estimation.PositionFixObs` |
| `ballistic.AccelObs` | `ballistic.estimation.AccelObs` |
| `ballistic.MultiSensorScenario` | `ballistic.estimation.MultiSensorScenario` |
| `ballistic.SensorClock` | `ballistic.estimation.SensorClock` |
| `ballistic.SensorMask` | `ballistic.estimation.SensorMask` |
| `ballistic.estimate_ea_smoother` | `ballistic.estimation.estimate_ea_smoother` |
| `ballistic.EASmootherConfig` | `ballistic.estimation.EASmootherConfig` |
| `ballistic.EASmootherResult` | `ballistic.estimation.EASmootherResult` |
| `ballistic.EASmootherRateBudget` | `ballistic.estimation.EASmootherRateBudget` |
| `ballistic.EAMethod` | `ballistic.estimation.EAMethod` |
| `ballistic.EAProblemSpec` | `ballistic.estimation.EAProblemSpec` |
| `ballistic.EASpecResult` | `ballistic.estimation.EASpecResult` |
| `ballistic.estimate_ea` | `ballistic.estimation.estimate_ea` |
| `ballistic.information_bound` | `ballistic.estimation.information_bound` |
| `ballistic.EAInformationBound` | `ballistic.estimation.EAInformationBound` |
| `ballistic.EAInformationBoundOptions` | `ballistic.estimation.EAInformationBoundOptions` |
| `ballistic.reject_heavy_tails` | `ballistic.estimation.reject_heavy_tails` |
| `ballistic.DragEnvironment` | `ballistic.drag.DragEnvironment` |
| `ballistic.DragSample` | `ballistic.drag.DragSample` |
| `ballistic.TimeWindow` | `ballistic.drag.TimeWindow` |
| `ballistic.IterationDiag` | `ballistic.drag.IterationDiag` |
| `ballistic.WeatherConditions` | `ballistic.drag.WeatherConditions` |
| `ballistic.BallisticModel` | `ballistic.drag.BallisticModel` |
| `ballistic.standard_leo_altitude_grid_km` | `ballistic.drag.standard_leo_altitude_grid_km` |
| `ballistic.from_models_snapshot` | `ballistic.drag.from_models_snapshot` |
| `ballistic.from_models_along` | `ballistic.drag.from_models_along` |
| `ballistic.from_models_4d` | `ballistic.drag.from_models_4d` |
| `ballistic.from_accelerometer_density` | `ballistic.drag.from_accelerometer_density` |
| `ballistic.extend_density_below` | `ballistic.drag.extend_density_below` |
| `ballistic.parse_density_table` | `ballistic.drag.parse_density_table` |
| `ballistic.load_beta_table` | `ballistic.drag.load_beta_table` |
| `ballistic.load_empirical_acceleration` | `ballistic.drag.load_empirical_acceleration` |
| `ballistic.save_drag_env` | `ballistic.drag.save_drag_env` |
| `ballistic.load_drag_env` | `ballistic.drag.load_drag_env` |
| `ballistic.AeroShape` | `ballistic.aero.AeroShape` |
| `ballistic.ConstantCd` | `ballistic.aero.ConstantCd` |
| `ballistic.FlowRegimeCd` | `ballistic.aero.FlowRegimeCd` |
| `ballistic.MachTable` | `ballistic.aero.MachTable` |
| `ballistic.AeroBeta` | `ballistic.aero.AeroBeta` |
| `ballistic.free_molecular_cd` | `ballistic.aero.free_molecular_cd` |
| `ballistic.reemission_ratio` | `ballistic.aero.reemission_ratio` |
| `ballistic.henderson1976_sphere_cd` | `ballistic.aero.henderson1976_sphere_cd` |
| `ballistic.EOFStateVector` | `ballistic.readers.EOFStateVector` |
| `ballistic.EOFFile` | `ballistic.readers.EOFFile` |
| `ballistic.EOFParser` | `ballistic.readers.EOFParser` |
| `ballistic.PropagationTestResult` | `ballistic.readers.PropagationTestResult` |
| `ballistic.parse_eof_auto` | `ballistic.readers.parse_eof_auto` |
| `ballistic.find_and_parse_eof` | `ballistic.readers.find_and_parse_eof` |
| `ballistic.compute_propagation_result` | `ballistic.readers.compute_propagation_result` |
| `ballistic.check_velocity_records` | `ballistic.readers.check_velocity_records` |
| `ballistic.VelocityRecordCheck` | `ballistic.readers.VelocityRecordCheck` |
| `ballistic.VelocityRecordVerdict` | `ballistic.readers.VelocityRecordVerdict` |
| `ballistic.cip_drift_velocity` | `ballistic.readers.cip_drift_velocity` |
| `ballistic.TleGradePreset` | `ballistic.tle.TleGradePreset` |
| `ballistic.tle_grade_preset` | `ballistic.tle.tle_grade_preset` |

## Additions (nothing to migrate)

- `ballistic.doctor()` / `python -m ballistic doctor`: what data is loaded
  (Earth orientation, leap seconds, space weather, ocean tides) and until
  when, the license terms of the embedded models, and a self-test (SGP4 vs
  SGP4-VER, a 10-minute Sentinel-1A prediction against its precise orbit);
  returns a `DoctorReport` (`.ok`), exit status 1 on a failed check.
- `BALLISTIC_OFFLINE=1` makes `ballistic.load()` behave as `offline=True`.
- `ea::CV::grid` (C++): `ea::CV::Grid::Coarse` searches 12 candidates (16
  with `Mode::FixedLambda`) instead of 495 (99), about 40× faster; the
  default `Grid::Full` is unchanged. Any CV run now reports a `ConfigHint`
  when the chosen knot spacing or final λ lies on the edge of its grid.
- The README quick start is executed verbatim by `tests/python/test_quickstart.py`.

- TLE input (`include/ballistic/sgp4/`, Python `ballistic.tle`): a strict
  TLE parser, a clean-room SGP4/SDP4 that reproduces SGP4-VER, TEME <-> ITRF
  (`teme_to_itrf`), `State.from_tle(line1, line2, epoch=None)` and
  `State.from_teme`; `ballistic.tle.fit_state(tle, config)` (a start fitted
  to SGP4 over a day); `recommended_config(state, accuracy="tle",
  horizon_s=...)` with the measured preset `tle_grade_preset(state,
  horizon_s)` (`presets/tle_grade.hpp`).
- A bare configuration (no radiation pressure, no drag) issues one
  `ConfigHint` per process with their measured cost. `ConfigHint` is advice:
  strict mode (`BALLISTIC_STRICT_WARNINGS=1`, `set_strict`) no longer turns it
  into an error.
- Python `Trajectory.propagate_for` / `propagate_to` (single trajectory,
  forward) now record the steps flown, so `traj(t)` interpolates them; it
  used to re-propagate on demand and returned None more than
  `max_propagation_time` (1 h) from the history it had.
- `make_drag_environment` raises `DataCoverageError` (was `std::out_of_range`
  / `IndexError`, and a misleading "before 1961" message) when the start has
  no space weather.

- `ballistic.State` / C++ `InitialState` (`initial_state.hpp`): a start
  state with its frame named. The constructors check it: positions in km,
  below the ground, a suborbital orbit (`allow_suborbital=True` for a
  reentry, ballistic or launch state), and the Earth-fixed/inertial mix-up
  (the same numbers read in the other frame are a much better orbit:
  `FrameError`, or a `FrameAssumptionWarning` when both readings are
  plausible; `check_frame=False` skips it). `from_j2000` applies the IAU
  2006 frame bias. C++ `sp3::initial_state(file, satellite, epoch, ...)`.
  C++ code that still uses the unchecked ITRF constructor can be audited
  with `BALLISTIC_AUDIT_START_STATES=1` (prints what the checks would say).

- Flow-regime drag coefficient for reentry (`aero/flow_regime.hpp`):
  `aero::AeroBeta{mass, reference area, reference length, CdModel}` is a new
  beta column of `DragEnvironment` (alongside `BetaProfile1D/2D`), with
  `CdModel` = `ConstantCd` or `FlowRegimeCd{shape, accommodation, wall
  temperature, continuum}` (sphere / flat plate / cylinder; free-molecular
  Sentman above Kn 10, continuum below 1e-3 — Henderson 1976 sphere,
  modified Newtonian plate and cylinder, or a `MachTable` — bridged
  between). `DragSetup::beta` accepts either (`{.beta = BetaProfile1D...}`
  still compiles). `DragSample` gains `temperature_k`, `molar_mass_kg_kmol`
  (live model), `cd`, `knudsen`, `mach`; `DragEnvironment::evaluate` takes an
  optional Earth-fixed velocity (a `FlowRegimeCd` needs it).
  Python: `AeroBeta`, `FlowRegimeCd`, `ConstantCd`, `MachTable`,
  `AeroShape`, `free_molecular_cd`, `henderson1976_sphere_cd`;
  `recommended_config(..., reentry=AeroShape.SPHERE, reference_length_m=D)`.
  Orbits are unchanged: beta profiles and `ConstantCd` give bit-identical
  accelerations.
- `Trajectory::propagate_until(target, events, stop_at_ground=true)`
  (`events.hpp`: `altitude_crossing`, `ground_impact`, `perigee`, `apogee`,
  `at_epoch`): events located inside the step to 1e-4 s; ground impact is
  armed by default. The force model refuses positions within 1000 km of
  the Earth's center (it used to integrate toward the center and hang).
- `coords::ecef_to_geodetic` / `ecef_to_altitude` are the exact Vermeille
  (2011) closed form (`ecef_to_altitude` was wrong by −21 km at 45°).

- `Epoch` is exact: labels and differences are good to ~1e-11 s (it used to
  round to a ~40 µs grid). One-hour propagations used to run 13.4 µs short.
- `Epoch.from_tt`, `from_tai`, `from_gps`, `from_tdb` (calendar input in
  that scale), `from_gps_week`, `from_datetime` (time-zone-aware only),
  `to_gps_week`, and the exact split `from_tt_split` / `tt_day` /
  `tt_seconds`.

- `ballistic.BallisticError` and subclasses (`DataCoverageError`,
  `FrameError`, `InputPlausibilityError`, `ConfigError`, `RemovedApiError`)
  are raised instead of plain `RuntimeError`. They derive from
  `RuntimeError`, so existing `except RuntimeError` handlers still work.
- Library warnings are `ballistic.BallisticWarning` subclasses (all
  `RuntimeWarning`), issued when the library continues with an
  approximation. `ballistic.set_strict(True)` turns them into errors.
- Notes about global data (no Earth orientation data loaded, forecast space
  weather, ocean tides not loaded) are issued once per process, not once per
  trajectory. Each trajectory still lists them in `Trajectory.warnings`.

- C++ `coords::earth_rotation_rate(epoch)` / `IAUTransform::earth_rotation_rate`
  (the Earth rotation angle's rate times 1 + d(UT1−UTC)/dt, the rate every
  ITRF↔GCRS velocity and acceleration transform now uses; the velocity
  transforms also include the polar motion's rate),
  `coords::EARTH_ROTATION_ANGLE_RATE`, the two-part
  `coords::cirs_to_tirs_matrix(ut1_day, ut1_day_fraction)`,
  `time::EopSample::xp_rate_arcsec_per_day` / `yp_rate_arcsec_per_day` /
  `ut1_utc_rate_s_per_day`, and `time::EOPData::generation()` (changes
  whenever a lookup's answer can).
  `IAUTransform` results depend on the epoch alone (no 10-minute matrix
  hold), and explicit `EarthOrientationParameters` are used in full (UT1−UTC
  and LOD included) when no table is loaded.
