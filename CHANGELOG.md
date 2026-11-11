# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.3]

Corrects the NRLMSIS 2.1 translation in the NRL plugin; the library itself is
unchanged. Reinstall the plugin after upgrading: `ballistic.install_nrl_models(accept_license=True)`.

### Fixed

- NRL plugin: two constants in the NRLMSIS 2.1 translation differed from the
  official `msis_constants.F90`. The surface log mixing ratios of N2, O2, He
  and Ar were hard-coded values off by up to 3e-4 (now `log()` of the
  CIPM-2007 mole fractions), and the anomalous-oxygen scale height used
  15.9994 g/mol instead of 16. Total mass density changes by at most 7e-5
  below 300 km and 3e-4 above 500 km. A new test checks every value of the
  official reference output (`msis2.1_test_ref_dp.txt`, now under
  `tests/data/nrlmsis/`) to half its last printed digit; the earlier check
  allowed 20% in density.

### Licensing

- The corrected `nrlmsis21_full.hpp` and the new reference test were
  delivered to the NRLMSIS authors, as §4b of the NRL-SOF-014-1 license
  requires.

## [0.1.2]

First release published on PyPI (`pip install ballistic-orbit`).

### Fixed

- `Loader`'s file-age check subtracted naive local times, an hour off across
  a daylight-saving change, and on Windows read a file written just now as
  slightly negative in age (its modification time comes from a different
  clock). It now compares POSIX timestamps; a file stamped slightly in the
  future is age 0.

### Build

- The TestPyPI install test in `wheels.yml` now sets up what a user would:
  the NRL plugin from the GitHub release (`install_nrl_models`) and the
  committed EOP and space-weather files for `ballistic doctor`. A repeat
  TestPyPI upload of the same version skips files already there.

### Licensing

- The NRLMSIS 2.1 derived files (`nrlmsis21_full.hpp`, `nrlmsis21_parm.hpp`
  and the script that generates the parameter header) were delivered to the
  NRLMSIS authors at the U.S. Naval Research Laboratory, as §4b of the
  NRL-SOF-014-1 license requires.

## [0.1.1]

Published as a GitHub release (with the NRL plugins) and on TestPyPI, not on
PyPI: its install test on Windows found the file-age bug fixed in 0.1.2.
Version 0.1.0 was never distributed.

### Changed (BREAKING)

- The Python distribution is renamed `ballistic` → `ballistic-orbit`
  (`pip install ballistic-orbit`; a git install names `ballistic-orbit @ git+…`).
  `import ballistic` is unchanged. An environment that installed the old name
  should `pip uninstall ballistic` first, since both install the same package
  directory.
- The NRLMSIS 2.1 atmosphere and HWM14 wind are no longer part of the
  library or the Python package. They are the separate plugin `ballistic_nrl`
  (CMake target; a C interface, `atmosphere/nrl_plugin_abi.h`), loaded at
  run time for model-based drag (`atmosphere/nrl_models.hpp`), because
  NRLMSIS 2.1 is licensed for non-commercial use only. Python:
  `ballistic.install_nrl_models(accept_license=True)` downloads it from the
  GitHub release (or installs a downloaded file with `source=`) after showing
  the license; `ballistic.nrl_models()` reports it. C++:
  `InitConfig::nrl_models_path`, the `BALLISTIC_NRL_MODELS` environment
  variable, or `<data directory>/nrl/`. Without it, `make_drag_environment`
  and the `from_models_*` generators raise `ConfigError` saying how to
  install it. Results through the plugin are bit-identical to the models
  called directly.
- `recommended_config` turns drag on only when the NRL plugin is installed;
  otherwise drag is off with one `ConfigHintWarning` stating the cost. New
  argument `drag`: None (that behavior), True (required; `ConfigError`
  without the plugin), False (off, no warning).
- Removed with the plugin change: the CMake option `BALLISTIC_WITHOUT_NRL`
  (use `-DBALLISTIC_BUILD_NRL_PLUGIN=OFF` to skip building the plugin),
  `build_info().with_nrl`, `atmosphere::kWithNrlModels`, the run-time license
  notice (Python `LicenseWarning`, C++ `diagnostics::Category::License`,
  `atmosphere::report_nrlmsis_license()`): the terms are shown once, by
  `install_nrl_models`. `ballistic.LicenseWarning` raises `RemovedApiError`.
- American spelling throughout the code and documentation (license, meter,
  centered, maneuver, modeled, -ize). Renamed: Python `LicenceWarning`,
  C++ `diagnostics::Category::Licence` and `report_nrlmsis_licence` (all
  three since removed, see above), `sp3::Arc::centred` → `centered` (also the
  field of `derived_velocities()`'s result). Proper names (IERS ITRS Centre, GFZ
  German Research Centre) and file-format strings (ESA's `DEGRADED-MANOEUVRE`
  flag) keep their own spelling.
- The Python package is split into tiers. The top-level `ballistic` keeps
  what a first session needs (83 names, down from 160: loading data,
  `Epoch`, `State`, `Trajectory` and its configuration, `recommended_config`,
  `make_drag_environment`, frame conversions, errors). Specialized names
  moved into `ballistic.estimation` (EA fitting; was `ballistic.estimators`),
  `ballistic.drag` (atmospheres built by hand, table readers, `.npz` files),
  `ballistic.aero` (flow-regime Cd), `ballistic.readers` (EOF files,
  velocity-record checks) and `ballistic.tle` (the TLE-grade preset). An old
  top-level name raises `RemovedApiError` naming its new import;
  `docs/MIGRATION.md` lists all 77. `ballistic.estimation.information_bound`
  now accepts the `TrajectoryConfig` dataclass, like `estimate_ea_smoother`.

### Added

- GNSS radiation pressure per block (`gnss_blocks.hpp`, Python
  `satellite_properties.gnss_block`) and `validate_gnss.py --srp cannonball`
  / `--fit-srp`: 1-h GPS 0.089 m vs 0.668 m without (`GNSS-1H-SRP`).
- `Trajectory.stm(t)` / `Trajectory.covariance(t, p0)` (C++ `StateTransition`,
  `state_transition.hpp`): the state transition matrix and the propagated
  start covariance along a finished trajectory, GCRS, [r; v].
- `ballistic::build_info()` / Python `ballistic._core.build_info()` report
  how the library was compiled, and `doctor()` prints it.
- Every satellite-registry entry says how its values were obtained
  (`drag_kind` / `srp_kind`: physical, fitted, nominal or legacy; C++
  `satellites::FitKind`), and `recommended_config` warns
  (`ConfigHintWarning`) when it uses a legacy (unsourced) value.

### Licensing

- The Python wheel contains only MIT code and embedded public data; the
  NRL plugin is installed only on request, with `accept_license=True`, and
  the GitHub release that carries it also carries `nrlmsis2.1_license.txt`
  and `THIRD_PARTY_NOTICES.md`.

### Release checklist

- Deliver every change to the NRLMSIS 2.1 derived files (the C++
  translation `nrlmsis21_full.hpp` and the converted parameter file
  `nrlmsis21_parm.hpp`) to the authors or NRL Code 7630, U.S. Naval Research
  Laboratory, as §4b of the NRL-SOF-014-1 license requires. The files as
  released in 0.1.3 have been delivered.
- Publish the draft GitHub release that `wheels.yml` creates for the tag: it
  carries the NRL plugin for each platform, `SHA256SUMS`,
  `nrlmsis2.1_license.txt` and `THIRD_PARTY_NOTICES.md`, and
  `ballistic.install_nrl_models()` downloads from it (a draft is invisible).
  Check that each platform's plugin is attached before uploading wheels.
- The PyPI distribution name is `ballistic-orbit` (the import name stays
  `ballistic`); it was unregistered when chosen, but PyPI confirms a name only
  when the first upload succeeds.
- Trusted publishing is registered for `ballistic-orbit` on TestPyPI
  (environment `testpypi`) and PyPI (environment `pypi`): owner
  `nvhuynh16`, repository `ballistic`, workflow `wheels.yml`. Each index
  needs its own registration and the GitHub environment of that name.
- Push the version tag (`v*`) and check the `Wheels` run: four wheels
  (manylinux_2_28 x86_64, Windows AMD64, macOS arm64 and x86_64), each having
  passed the doctor and quick-start tests, and the NRL plugins. Then run the
  workflow on the tag with target `testpypi` (uploads, installs from TestPyPI
  on the four platforms and runs the Python tests there), and, once that is
  green, with target `pypi`. An index never accepts the same version twice.
- Wheels are built for CPython 3.12 only (`build = "cp312-*"`); add later
  versions to `[tool.cibuildwheel]` once they are tested.
- No sdist is published: `python/pyproject.toml` builds the CMake project in
  the parent directory (`cmake.source-dir = ".."`), which an sdist of
  `python/` would not contain. Source installs use the git URL.

## [0.1.0]

First public version: a C++23 orbit propagation library with Python
bindings.

### Propagation

- `Trajectory` with Dormand-Prince RK4(5) (default) and Yoshida 6/8
  symplectic integrators, history and quintic Hermite interpolation,
  batch propagation, and event location inside the step
  (`propagate_until`: altitude crossings, ground impact, perigee, apogee,
  epochs).
- Start states with a named frame (`InitialState` / Python `State`: ITRF,
  GCRS, J2000, Keplerian, SP3, TLE/TEME), checked for units, plausibility
  and Earth-fixed/inertial mix-ups.
- Time scales UTC, UT1, TAI, TT, TDB and GPS on an exact `Epoch`; IERS
  Earth orientation parameters and leap seconds; IAU 2006/2000A
  precession-nutation (GCRS ↔ ITRF) and ITRF realizations.

### Force models

- EGM2008 gravity to degree 360 embedded (Clenshaw summation, analytic
  Hessian), higher degrees from the NGA file; Sun, Moon and Jupiter from an
  embedded JPL DE440 subset (1950–2100) or an SPK kernel; IERS 2010 solid
  Earth tides; FES2004 ocean tides (downloaded file); relativistic
  corrections; solar radiation pressure, Earth albedo and infrared.
- Atmospheric drag through `DragEnvironment`: NRLMSIS 2.1 density and HWM14
  winds evaluated along the orbit on measured space weather, measured
  (accelerometer) density, or tabulated fields; U.S. Standard Atmosphere
  1976 below 86 km; flow-regime drag coefficients for reentry
  (free-molecular, transitional, continuum).
- `recommended_config` for the most accurate setup of a known satellite, a
  satellite registry with cited properties, and a measured TLE-grade preset.

### Estimation

- Empirical-acceleration estimation from position, velocity or
  accelerometer samples (`estimate_ea_bspline`, with Gauss-Newton
  refinement), and a multi-sensor smoother for radar, optical, position-fix
  and accelerometer data (`estimate_ea_smoother`); drag ballistic
  coefficient from an estimated acceleration (`refine_beta_from_ea`).

### Interfaces and data

- Clean-room SGP4/SDP4 and a strict TLE parser.
- SP3 and Sentinel EOF readers.
- Python package `ballistic` (nanobind) with typed warnings and errors, a
  data loader (`ballistic.load()`) and an installation check
  (`ballistic.doctor()`).
- Validation against precise orbits of Sentinel-1/2/3/6, CHAMP, GRACE-FO,
  GOCE, LAGEOS, Etalon, GPS, QZSS and BeiDou; every accuracy number is
  recorded with its reproduction command in `docs/validation_claims.md`.

### Licensing

- Library code under the MIT license. NRLMSIS 2.1 is included under
  NRL-SOF-014-1 (research, academic and non-profit use only); HWM14 is a
  U.S. Government work included with attribution. See
  `THIRD_PARTY_NOTICES.md`.
