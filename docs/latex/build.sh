#!/usr/bin/env bash
# Build the tutorial PDFs with a self-contained Tectonic (no system TeX,
# no sudo). Bootstraps the pinned binary into ~/.local/bin on first use.
# This NEVER regenerates results — it only typesets docs/latex/results/*.
#
#   ./build.sh                 # build both tutorial_cpp.pdf + tutorial_python.pdf
#   ./build.sh tutorial_cpp    # build one
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TECTONIC_VERSION="0.16.9"
TECTONIC_SHA256="60b13a0826ae7ad9ce34b4a2df06bff2cfcfa6dda8a915477c0cbb84e1a4a902"
URL="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz"

bin="$HOME/.local/bin/tectonic"
need_boot=1
if [ -x "$bin" ] && "$bin" --version 2>/dev/null | grep -q "$TECTONIC_VERSION"; then
  need_boot=0
fi
if [ "$need_boot" = 1 ]; then
  echo "bootstrapping tectonic ${TECTONIC_VERSION} -> $bin" >&2
  mkdir -p "$HOME/.local/bin"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  curl -fsSL --max-time 120 -o "$tmp/t.tgz" "$URL"
  got="$(sha256sum "$tmp/t.tgz" | awk '{print $1}')"
  [ "$got" = "$TECTONIC_SHA256" ] || { echo "sha256 mismatch: $got" >&2; exit 1; }
  tar -xzf "$tmp/t.tgz" -C "$tmp"
  install -m 0755 "$tmp/tectonic" "$bin"
fi

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(tutorial_cpp tutorial_python ea_global_minimum)
rc=0
for t in "${targets[@]}"; do
  t="${t%.tex}"
  echo "==> $t.tex"
  if "$bin" -X compile --keep-logs --outdir "$DIR" "$DIR/$t.tex" >/dev/null 2>&1 \
     || "$bin" --keep-logs --outdir "$DIR" "$DIR/$t.tex"; then
    echo "    -> $DIR/$t.pdf"
  else
    echo "    FAILED: $t" >&2; rc=1
  fi
done
exit $rc
