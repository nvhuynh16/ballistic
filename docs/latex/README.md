# docs/latex — Ballistic tutorials (C++ & Python)

The committed `results/` tables are the real output of the example
programs at the current library version. They are regenerated with
`regen.sh` (below) whenever a library change moves a number; C++ and Python
sample the same instants of one shared glide fixture and agree digit for
digit. `regen.sh` honors `BALLISTIC_BUILD_DIR` to use another build tree.

Two large, hands-on tutorial PDFs. **Every code listing is a committed
program that is compiled and run; every number in every table is that
program's real output** (no hand-typed values).

## Build the PDFs (seconds; no system TeX, no sudo)

```bash
bash docs/latex/build.sh            # tutorial_cpp.pdf, tutorial_python.pdf, ea_global_minimum.pdf
bash docs/latex/build.sh tutorial_cpp
```

`build.sh` self-bootstraps a pinned, sha256-verified Tectonic
(`0.16.9`) into `~/.local/bin` on first use. The build **never** runs the
examples — it only typesets whatever is in `results/`. First run needs
network for Tectonic's TeX bundle (cached in `~/.cache/Tectonic`).

## Regenerate the result numbers (minutes–~1 h)

```bash
cmake -S . -B build -DBALLISTIC_BUILD_TUTORIAL_EXAMPLES=ON \
      -DBALLISTIC_BUILD_PYTHON=ON
cmake --build build -j
bash docs/latex/regen.sh                 # all cells, resumable
bash docs/latex/regen.sh --status        # what's fresh/missing
bash docs/latex/regen.sh --only ex05_pos_100 --force
```

`regen.sh` is resumable (skips cells whose output is newer than the
ex05 binary and the glide fixture), writes atomically, and runs the
heavy `ex05` hv-block-CV cells `--jobs N` in parallel. Run it detached
for the POS+GN+CV cells (the slow ones); poll `--status`. Until a cell
is regenerated the PDF shows a "run regen.sh" placeholder and still
compiles.

## Layout

- `tutorial_{cpp,python}.tex` — roots; `common/{preamble,macros}.tex`
- `ea_global_minimum.tex` — a standalone note on when the global minimum
  of the EA estimation problem can be certified
- `chapters_{cpp,py}/chNN.tex` — 10 chapters each (mirrored prose)
- `examples/{cpp,py}/` — the runnable programs (verbatim in the PDFs);
  `examples/cpp/glide_common.hpp` / `examples/py/glide_common.py` the
  shared public-API glide helper; `examples/cpp/fixture/` the fixture
  generator + parity checker
- `examples/fixtures/glide_fixture.json` + `glide_fixture_data.hpp` —
  committed, generator-derived constants (C++/Python share one glide)
- `results/` — committed generated JSON/CSV the PDFs `\input`

Tracked: sources, `build.sh`, `regen.sh`, `README.md`, fixtures,
`results/`. Git-ignored: `*.pdf`, `*.aux`, `*.log`, … (build artifacts).
