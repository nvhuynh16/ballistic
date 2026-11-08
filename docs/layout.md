# Repository layout and testing

## Layout

- `include/ballistic/`: public C++ headers (header-mostly library); `src/`:
  compiled implementation and embedded data; `src/nrl_plugin/`: the separate
  NRLMSIS 2.1 / HWM14 plugin `ballistic_nrl` (non-commercial license; the
  library loads it at run time); `python/`: nanobind bindings and the
  `ballistic` Python package
- `docs/`: user guides, the API reference and the validation-claims
  manifest; `docs/latex/`: C++ and Python tutorials and the EA mathematics
  notes (`bash docs/latex/build.sh`)
- `tests/{unit,integration,validation}/`: CTest tiers; `tests/python/`:
  pytest suite; `tests/data/`: committed test data
- `scripts/`: validation campaigns and data-download scripts (the
  reproduction commands in [`validation_claims.md`](validation_claims.md))
- `examples/`: small C++ programs

## Testing

```bash
ctest --test-dir build -LE heavy              # default gate
ctest --test-dir build -L claims              # EA accuracy claims (~1 min)
ctest --test-dir build -L heavy               # long-running tests (tens of minutes)
PYTHONPATH=./python uv run --with numpy --with pytest --with sgp4 pytest tests/python/ -q
```

`sgp4` is used only as a test oracle for the library's own SGP4; it is not
a runtime dependency. Pass an explicit timeout (`ctest --timeout <s>`,
`timeout 900 pytest ...`) so a stuck test surfaces quickly.
`BALLISTIC_REQUIRE_TEST_DATA=1` makes every test that would skip for
missing data fail instead, and `BALLISTIC_STRICT_WARNINGS=1` turns every
library warning (except configuration advice) into an error.
