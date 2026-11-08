#!/usr/bin/env bash
# =============================================================================
# regen.sh — regenerate the tutorial result files (docs/latex/results/*).
# =============================================================================
#
# The PDF build NEVER calls this; it only typesets whatever results exist.
# This driver is resumable: a cell is skipped if its output is newer than
# both the ex05 binary and the glide fixture, unless --force / --only.
# Writes atomically (.tmp -> mv) so an interrupted run is safe to resume.
#
#   ./regen.sh [--jobs N] [--only ID] [--force] [--status] [--fast-only]
#
# Heavy cells (ex05 hv-block CV) dominate runtime (~hours total). Run
# detached and poll `./regen.sh --status`.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BALLISTIC_BUILD_DIR:-$ROOT/build}/docs/latex/examples/cpp"   # BALLISTIC_BUILD_DIR: another build tree
RES="$ROOT/docs/latex/results"
FIX="$ROOT/docs/latex/examples/fixtures/glide_fixture.json"
JOBS=3 ONLY="" FORCE=0 STATUS=0 FAST_ONLY=0
while [ $# -gt 0 ]; do case "$1" in
  --jobs) JOBS="$2"; shift 2;;
  --only) ONLY="$2"; shift 2;;
  --force) FORCE=1; shift;;
  --status) STATUS=1; shift;;
  --fast-only) FAST_ONLY=1; shift;;
  *) echo "unknown arg $1"; exit 2;;
esac; done
mkdir -p "$RES"

# id | output file | command (run from $ROOT)
FAST=(
  "ex01|ex01_init_status.json|$BIN/ex01_setup $RES/ex01_init_status.json"
  "ex02|ex02_timescale_offsets.json|$BIN/ex02_timescales $RES/ex02_timescale_offsets.json"
  "ex03|ex03_sentinel_rmse.json|$BIN/ex03_sentinel_propagation tests/data/eof/sentinel1_2016/2016-1 $RES/ex03_sentinel_rmse.json"
  "ex04|ea_clean_glide.csv|$BIN/ex04_ea_clean_glide $RES/ea_clean_glide.csv"
  "ex07|multisensor_rmse.csv|$BIN/ex07_multisensor $RES/multisensor_rmse.csv"
  "ex09|ex09_pitfalls.json|$BIN/ex09_pitfalls $RES/ex09_pitfalls.json"
)
HEAVY=(
  "ex06|ea_ballistic_recovery.csv|$BIN/ex06_ballistic_recovery $RES/ea_ballistic_recovery.csv"
)
for o in pos vel acc; do for h in 100 200; do
  HEAVY+=("ex05_${o}_${h}|ea_noisy_${o}_${h}.csv|$BIN/ex05_ea_noisy_glide $o $h $RES/ea_noisy_${o}_${h}.csv")
done; done

stale() {  # $1 output file ; returns 0 (stale -> run) / 1 (fresh -> skip)
  [ "$FORCE" = 1 ] && return 0
  local f="$RES/$1"
  [ -f "$f" ] || return 0
  [ "$f" -ot "$FIX" ] && return 0
  [ "$f" -ot "$BIN/ex05_ea_noisy_glide" ] && return 0
  return 1
}

if [ "$STATUS" = 1 ]; then
  for u in "${FAST[@]}" "${HEAVY[@]}"; do
    id="${u%%|*}"; rest="${u#*|}"; outf="${rest%%|*}"
    if [ -f "$RES/$outf" ]; then printf "  %-14s FRESH  %s\n" "$id" "$outf"
    else printf "  %-14s MISSING %s\n" "$id" "$outf"; fi
  done
  exit 0
fi

run_unit() {
  local id="$1" outf="$2" cmd="$3"
  if [ -n "$ONLY" ] && [ "$id" != "$ONLY" ]; then return 0; fi
  if ! stale "$outf"; then echo "[skip fresh] $id"; return 0; fi
  echo "[run] $id -> $outf"
  ( cd "$ROOT" && eval "${cmd/$RES\/$outf/$RES/$outf.tmp}" ) \
    && mv -f "$RES/$outf.tmp" "$RES/$outf" \
    && echo "[done] $id" || { echo "[FAIL] $id"; rm -f "$RES/$outf.tmp"; }
}

# Fixture first (cheap; everything downstream depends on it). Only a
# whole-run --force regenerates it: `--force --only X` must not silently
# move the shared glide out from under every other cell.
if { [ "$FORCE" = 1 ] && [ -z "$ONLY" ]; } || [ ! -f "$FIX" ]; then
  ( cd "$ROOT" && "$BIN/gen_glide_fixture" "$FIX" ) || exit 1
fi

# The C++ examples BAKE the fixture header in at compile time. A binary
# older than the header would silently mix two different glides into the
# same results/ set (Python reads the JSON at runtime) — hard-stop instead.
FIX_HPP="$ROOT/docs/latex/examples/fixtures/glide_fixture_data.hpp"
if [ "$BIN/ex05_ea_noisy_glide" -ot "$FIX_HPP" ]; then
  echo "ERROR: example binaries are older than glide_fixture_data.hpp —"
  echo "       rebuild first (cmake --build build) so C++ and Python use"
  echo "       the same glide."
  exit 1
fi

for u in "${FAST[@]}"; do
  id="${u%%|*}"; r="${u#*|}"; outf="${r%%|*}"; cmd="${r#*|}"
  run_unit "$id" "$outf" "$cmd"
done
[ "$FAST_ONLY" = 1 ] && { echo "fast-only done"; exit 0; }

# Heavy ex05 cells: independent -> background, throttled to $JOBS.
active=0
for u in "${HEAVY[@]}"; do
  id="${u%%|*}"; r="${u#*|}"; outf="${r%%|*}"; cmd="${r#*|}"
  run_unit "$id" "$outf" "$cmd" &
  active=$((active+1))
  if [ "$active" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; active=$((active-1)); fi
done
wait
echo "regen complete"
