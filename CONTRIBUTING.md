# Contributing to Ballistic

Thank you for your interest. This file says how to build the library, run
the tests, and follow the rules that keep its numbers trustworthy. Issues and
pull requests go to https://github.com/nvhuynh16/ballistic.

## Building

Requirements:

- a C++23 compiler: GCC 13+, clang 18 or MSVC 2022 (the compilers CI
  builds and tests with, see below);
- CMake 3.20+;
- Eigen 3.4: found with `find_package(Eigen3 3.4)`, or downloaded into the
  build tree by `FetchContent` when no system copy exists (on Debian/Ubuntu,
  `libeigen3-dev` makes configuring faster);
- for the Python bindings, Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBALLISTIC_BUILD_PYTHON=ON
cmake --build build -j$(nproc)
```

Building the Python bindings places the `_core` extension in
`python/ballistic/`, so the package can be used from the source tree with
`PYTHONPATH=./python`. `BALLISTIC_NATIVE_ARCH` (default OFF) enables
`-march=native`; leave it off for anything you share: such binaries run
only on similar CPUs, and floating-point results can change in the last
bits.

## Running the tests

Always pass a timeout, so that a stuck test shows up instead of hanging.

```bash
# C++: the default gate (excludes the long-running `heavy` tier)
ctest --test-dir build -LE heavy --timeout 1500 -j$(nproc)

# Python, as CI runs it: every library warning is an error
BALLISTIC_STRICT_WARNINGS=1 PYTHONPATH=./python timeout 1200 \
    uv run --with numpy --with pytest --with sgp4 pytest tests/python/ -q
```

- `BALLISTIC_STRICT_WARNINGS=1` turns every library warning except
  configuration advice (`ConfigHint`) into an error.
- `BALLISTIC_REQUIRE_TEST_DATA=1` makes every C++ test that would skip for
  missing data fail instead. Use it when you have the data that is not part
  of the repository (Sentinel precise orbits and the other products listed
  in `THIRD_PARTY_NOTICES.md`; each directory's `README.md` says how to get
  them). Without the data those cells are reported as skips, never as
  passes.
- `ctest -L claims` runs the estimator accuracy claims (about a minute),
  `ctest -L heavy` the long-running tier.
- `sgp4` is a test oracle for the library's own SGP4 implementation; it is
  never a runtime dependency.
- NRLMSIS 2.1 and HWM14 are never in the library: they are the plugin
  `ballistic_nrl` (CMake target, built by default), which every test and
  diagnostic is pointed at (`tests/nrl_plugin_env.cpp`; pytest finds
  `build/nrl/` through `tests/python/conftest.py`). Two configurations are
  tested: with the plugin, and without it (`-DBALLISTIC_BUILD_NRL_PLUGIN=OFF`,
  as a fresh installation is). A test that needs model-based drag
  (`make_drag_environment`, `from_models_*`, `recommended_config` with drag,
  the NRL headers) carries the CTest label `nrl`
  (`ballistic_add_test(name path nrl)`; such tests are not built without the
  plugin) or the pytest marker `@pytest.mark.nrl`; a test with only a few
  such cells wraps them in `#ifndef BALLISTIC_TEST_NO_NRL_PLUGIN` instead.
  The no-plugin configuration runs `ctest -LE 'heavy|nrl'` and
  `pytest -m "not nrl"`; in it, `nrl` tests that are selected anyway are
  skipped by `tests/python/conftest.py`.

How the test tree is organized and how to add a test: see
[`tests/README.md`](tests/README.md). In short, C++ tests use
`tests/test_utils.hpp` and are registered in `tests/CMakeLists.txt` with
`ballistic_add_test(name unit/test_name.cpp)`; a new unit test runs in
strict mode in CI, so it must not trigger a library warning unless it is
added to the `warns` list there.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

| Job | Platform | What it runs |
|-----|----------|--------------|
| `build-test` | Linux, GCC | default ctest gate, strict-mode unit tests, the C++ examples |
| `wheel-and-pytest` | Linux, GCC | wheel build and content check, generated-stub diff, strict pytest on the installed wheel, `ballistic doctor`, skip census |
| `clang` | Linux, clang 18 | default ctest gate |
| `linux-no-nrl` | Linux, GCC | no NRL plugin (`BALLISTIC_BUILD_NRL_PLUGIN=OFF`): ctest without the `nrl` label, pytest `-m "not nrl"`, `nrl_models()` reports it absent |
| `windows-msvc` | Windows, MSVC 2022 | default ctest gate, the C++ examples, wheel, strict pytest, `ballistic doctor`, skip census |
| `heavy` | Linux, GCC | nightly: diagnostics and tutorial builds, `ctest -L heavy`, `BALLISTIC_HEAVY=1` pytest |

`.github/workflows/wheels.yml` builds binary wheels with cibuildwheel
(manylinux_2_28 x86_64, Windows AMD64, macOS arm64 and x86_64) on `v*` tags
and on manual runs; each wheel runs the doctor and the README quick start
before upload. The same workflow builds the NRL models plugin for each
platform, checks that it loads (`scripts/check_nrl_plugin.py`), and on a tag
attaches the plugins, `SHA256SUMS` and the NRLMSIS license to a draft GitHub
release, which must be published for `ballistic.install_nrl_models()` to
find them. Publishing to PyPI happens only on a manual run with
`publish` ticked. No sdist is built: the Python project
(`python/pyproject.toml`) builds the CMake project one directory up, which
an sdist of `python/` would not contain.

To reproduce a job locally, run its steps from the workflow file. The clang
job is the Linux recipe with `CC=clang-18 CXX=clang++-18`; the no-NRL job
adds `-DBALLISTIC_BUILD_NRL_PLUGIN=OFF` to the configure line. With the Visual Studio generator
(Windows) the build type is chosen at build time:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release -LE heavy --timeout 2400 --output-on-failure
```

The wheel build can be run locally with `pipx run cibuildwheel --platform
linux python` (needs Docker).

## Static guards

`tests/python/test_static_guards.py` holds rules for source patterns that
have caused bugs before (a frame mix-up, a copied satellite property, a
swallowed exception, a silent clamp, ...). Each rule closes a bug class
listed in [`docs/trap_census.md`](docs/trap_census.md), and every rule is
tested against a probe that must trigger it.

When a line legitimately needs a forbidden pattern, exempt that one line
with an inline comment naming the rule and the reason:

```cpp
const double x = legacy_value;  // guard-exempt(RULE-ID): why this line is correct
```

Files that need a pattern throughout are allowlisted in the rule itself,
with a reason. A new bug class gets a census row and, where it can be
expressed, a rule.

## Single sources

Values that have drifted apart in the past have exactly one home. Read them
from there; do not copy them.

- **Satellite properties** (mass, areas, Cd, Cr, with citations):
  `include/ballistic/satellites.hpp`, bound to Python as
  `ballistic.satellite_properties`. Static rule `SAT-LITERAL` rejects
  literals next to a real satellite's name; synthetic test objects must not
  be named after a real satellite.
- **Physical constants**: `include/ballistic/core/constants.hpp` (Python:
  `ballistic.constants`). Static rule `CONST-LITERAL`.
- **Accuracy claims**: [`docs/validation_claims.md`](docs/validation_claims.md).
  Every accuracy number in the README's validation table cites a claim ID,
  and each claim row records the value, date, commit and the exact command
  that reproduces it. `tests/python/test_validation_claims.py` checks that
  cited IDs exist, that rows are complete, and that the reproduction
  commands' scripts, data paths and command-line flags exist. A new
  accuracy number lands together with its claim row.

## Errors and warnings

Errors and warnings a user can see say what happened in the user's terms
(the input, file or date involved), how large the effect is when that is
known, and what to do about it. Raise the narrowest `ballistic::Error`
subclass (`DataCoverageError`, `FrameError`, `InputPlausibilityError`,
`ConfigError`, `RemovedApiError`); report non-fatal approximations through
`include/ballistic/core/diagnostics.hpp` with a category. Never substitute
a default, clamp a value or skip a record without a raise or a diagnostic.
A removed public name raises `RemovedApiError` naming its replacement and is
listed in [`docs/MIGRATION.md`](docs/MIGRATION.md).

## Coding style

- C++23, header-mostly: public headers in `include/ballistic/`, compiled
  code and the large embedded data tables in `src/`.
- Comments explain why: the reason for a choice, the source of a formula or
  coefficient (paper, table, equation), and the measurement that justified
  a default. Code that reproduces a published model cites it.
- Units in names or comments (`_m`, `_s`, `_kg`), frames named explicitly
  (ITRF, GCRS, TEME); time is an `Epoch` with a named scale.
- Python follows the same rules; the bindings' type stubs
  (`python/ballistic/_core.pyi`) are generated, and CI checks that they are
  current.
- Python lint: the repository is kept clean under ruff 0.14.0 with the root
  `ruff.toml` (the generated `*.pyi` stubs are excluded):
  `uvx ruff@0.14.0 check scripts python/ballistic tests/python`. CI does not
  run ruff, so run it before sending a change.

## License of contributions

The library's own code is MIT-licensed (`LICENSE`), and contributions are
accepted under the same license.

Some files are derived from third-party software and keep their own terms
(`THIRD_PARTY_NOTICES.md`). In particular the NRLMSIS 2.1 files
(`include/ballistic/atmosphere/internal/nrlmsis21_full.hpp`,
`nrlmsis21_parm.hpp`) are distributed under the NRL license NRL-SOF-014-1
(`docs/licenses/nrlmsis2.1_license.txt`), and the HWM14 files
(`hwm14.hpp`, `hwm14_coefficients.hpp`) carry an attribution notice. Do not
modify these files without preserving their notices; a change to the
NRLMSIS files must add a dated change notice to the file header (§4b of the
license), and changes that alter the model output must not be described as
MSIS.
