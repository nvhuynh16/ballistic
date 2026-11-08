"""24-hour GNSS predictions across daily SP3 files (scripts/validate_gnss.py, joined arcs).

Each daily final-orbit file is a separate solution; a 24-hour prediction
starting inside one day ends in the next day's file, so the campaign joins a
satellite's files into one arc (scripts/_sp3_records.join_arcs). One start
each for a BeiDou GEO (C01) and a QZSS IGSO (J02) on the committed Wuhan and
CODE files, library defaults (SRP off: no sourced GNSS properties yet).

Measured: C01 660.7 m and J02 608.3 m at +24 h (start 01:00 GPS time on the first
day; 0.69 / 0.67 m at +1 h). The error is dominated by the unmodeled solar
radiation pressure (~1e-7 m/s^2 gives ~0.4 km in a day), so this gate checks
that the multi-file path runs and stays at its measured level, not that the
prediction is accurate. Limits: 1.5x the measured values.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import validate_gnss  # noqa: E402

CASES = [
    # constellation, directory, file pattern, satellite, class, limit [m]
    ("beidou-geo", "beidou", "WUM0MGXFIN_2024*_ORB.SP3.gz", "C01", "GEO", 1.5 * 660.7),
    ("qzss", "qzss", "COD0MGXFIN_2024*_ORB.SP3", "J02", "IGSO", 1.5 * 608.3),
]


def _files(directory, pattern):
    files = sorted((REPO / "tests/data/sp3" / directory).glob(pattern))
    if len(files) < 2:
        pytest.skip(f"missing test data tests/data/sp3/{directory}/{pattern} (need two "
                    f"consecutive days) — reacquire: see the README next to it")
    return [str(p) for p in files]


@pytest.mark.parametrize("constellation,directory,pattern,sat,cls,limit", CASES,
                         ids=[c[3] for c in CASES])
def test_gnss_24_hours_across_files(constellation, directory, pattern, sat, cls, limit):
    files = _files(directory, pattern)
    rows = validate_gnss.run(constellation, files, duration_s=86400.0, sats_filter={sat},
                             max_starts=1, verbose=False,
                             initialize=False)   # conftest pins EOP
    assert len(rows) == 1
    r = rows[0]
    assert r["sat"] == sat and r["class"] == cls
    print(f"\n  {sat} ({cls}) start {r['epoch']}: +24 h error {r['error_m']:.1f} m")
    assert np.isfinite(r["error_m"])
    assert r["error_m"] < limit, r["error_m"]


def test_24_hours_needs_the_join():
    # One arc per file: no start has truth 24 h later (00:00 is a file edge
    # and every later start runs past its own file).
    files = _files("qzss", "COD0MGXFIN_2024*_ORB.SP3")
    rows = validate_gnss.run("qzss", files, duration_s=86400.0, sats_filter={"J02"},
                             max_starts=1, verbose=False, initialize=False, join=False)
    assert rows == []
