# Tests

## Layout

| Directory | Contents |
|-----------|----------|
| `unit/` | One component per executable: time, frames, gravity, forces, parsers, estimator building blocks |
| `integration/` | Several components together: propagation chains, the EA estimators, the accuracy-claims suite |
| `validation/` | Predictions against real orbit products (Sentinel POE, GFZ/ILRS/IGS SP3) and reference values (SOFA/ERFA, Vallado, JPL Horizons) |
| `python/` | pytest suite for the bindings, the static guards and the documentation checks |
| `data/` | Test data; each directory has a README saying where the data comes from and how to reacquire it |

Shared C++ helpers: `test_utils.hpp` (the runner), `test_helpers.hpp`
(synthetic glide and radar test beds, `IsolatedDataDir` for hermetic data
directories) and `sp3_truth.hpp` (SP3 truth on top of the library's reader).

## The C++ runner

Every test in `unit/`, `integration/` and `validation/` uses
`test_utils.hpp` (CMake refuses a file that does not include it):

```cpp
#include "test_utils.hpp"
using namespace ballistic::test;

int main() {
    TestRunner runner("My Tests");
    runner.add_test("example", []() {
        TEST_ASSERT_NEAR(1.0, 1.0, 1e-9);
    });
    runner.run();
    return print_final_summary();
}
```

- Assertions: `TEST_ASSERT`, `TEST_ASSERT_MSG`, `TEST_ASSERT_TRUE` /
  `TEST_ASSERT_FALSE`, `TEST_ASSERT_EQ`, `TEST_ASSERT_NEAR`,
  `TEST_ASSERT_NEAR_REL`, `TEST_ASSERT_THROWS`.
- Skips: `TEST_SKIP(reason)` when data or a precondition is missing,
  `TEST_SKIP_OPT_IN(reason)` for a cell the caller opts into. Skips are
  counted and listed apart from passes; a cell never "skips" by printing and
  returning.
- Expected failures: `runner.add_xfail(name, reason, fn)` records a
  documented failure. It counts as success while it fails; an unexpected
  pass (XPASS) is reported so the mark can be retired.
- Cell selection: `TEST_FILTER` and `TEST_FILTER_EXCLUDE` take
  comma-separated substrings; a cell runs if its name contains any
  `TEST_FILTER` substring (or the variable is unset) and no
  `TEST_FILTER_EXCLUDE` substring. Example:
  `TEST_FILTER=smoother_camera_fusion ./build/tests/test_ea_smoother`.

Register a new test in `tests/CMakeLists.txt`:

```cmake
ballistic_add_test(test_my_feature unit/test_my_feature.cpp)
```

## CTest labels

| Label | Meaning |
|-------|---------|
| `unit` / `integration` / `validation` | Tier, from the source directory |
| `heavy` | Long-running; excluded from the default gate, run nightly |
| `claims` | Asserts the documented accuracy of the empirical-acceleration estimators (`test_ea_accuracy_claims`) |
| `warns` | Tests that emit library warnings on purpose; excluded from the strict-mode run (`BALLISTIC_STRICT_WARNINGS=1 ctest -L unit -LE warns`) |

The default gate:

```bash
ctest --test-dir build -LE heavy --timeout 1500 --output-on-failure
```

With `BALLISTIC_REQUIRE_TEST_DATA=1` every `TEST_SKIP` for missing data
becomes a failure; use it where all test data is present. Only opt-in skips
remain skips.

## Python tests

```bash
PYTHONPATH=./python uv run --with numpy --with pytest --with sgp4 \
    pytest tests/python -q
```

`sgp4` is the test-only SGP4 oracle; it is never shipped. The session
fixtures in `conftest.py` pin the committed Earth orientation and
space-weather files and point the data directory at a temporary location, so
results do not depend on files cached in the user's data directory.

- `BALLISTIC_STRICT_WARNINGS=1` turns every `RuntimeWarning` into an error.
- `BALLISTIC_HEAVY=1` enables the heavy tier (the CHAMP drag gate also needs
  `--with pandas --with h5py`).
- A test whose data is not committed is marked with
  `conftest.require_data(path, hint)`: it skips, and the reason names how to
  reacquire the file.

## Test data

Only part of `tests/data` is committed. The README in each data directory
names the source, the format and the command or archive that reacquires the
files; the C++ and Python tests that need an uncommitted file skip with a
reason pointing there.
