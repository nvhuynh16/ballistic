"""ballistic.sp3: the one SP3 reader, on committed files.

Traps it closes (docs/trap_census.md): GPS-AS-UTC (four campaign scripts read
GPS-time files as UTC), SP3-CLOCK-OVERFLOW (a reader dropped records whose
clock field ran into z), SP3-DUPLICATE-READERS. The C++ cells are in
tests/unit/test_sp3.cpp.
"""

from pathlib import Path

import numpy as np
import pytest

import ballistic
from conftest import require_data

REPO = Path(__file__).resolve().parents[2]
GRACEFO = REPO / "tests/data/sp3/grace_fo/GFZOP_RSO_L64_G_20260101_100000_20260102_000000_v03.sp3"
CHAMP = REPO / "tests/data/sp3/champ/GFZOP_RSO_L06_G_20050701_100000_20050702_000000_v01.sp3"
SLR = sorted((REPO / "tests/data/sp3/lageos").glob("ilrsa.orb.lageos1.*.sp3"))
# Untracked (a clean checkout lacks them): reacquire from GFZ ISDC.
GRACEFO_CLOCK = REPO / "tests/data/sp3/grace_fo/GFZOP_RSO_L64_G_20260205_220000_20260206_120000_v03.sp3"
GRACEFO_SECOND = REPO / "tests/data/sp3/grace_fo/GFZOP_RSO_L64_G_20260101_220000_20260102_120000_v03.sp3"
_CHAMP_HINT = "GFZ RSO CHAMP 2005 day 182 (.sp3 and .sp3.gz) — see tests/data/sp3/champ/README.md"
_GRACEFO_HINT = "GFZ RSO GRACE-FO v03 from the ISDC — see tests/data/sp3/grace_fo/README.md"


def test_gps_time_file_is_not_utc():
    f = ballistic.sp3.read(GRACEFO)
    arc = f.arc("L64")
    assert f.header.time_system == ballistic.sp3.SP3TimeSystem.GPS
    # The first tag reads 2026-01-01 10:00:00 in GPS time = 09:59:42 UTC.  # guard-exempt(PUB-DATE): data
    assert arc.epoch0 - ballistic.Epoch.from_utc(2026, 1, 1, 10, 0, 0) == pytest.approx(-18.0, abs=1e-9)


def test_utc_file_is_utc():
    if not SLR:
        pytest.skip("missing test data ilrsa.orb.lageos1.*.sp3 — reacquire: "
                    "python scripts/download_ilrs_sp3.py --satellite lageos1")
    f = ballistic.sp3.read(SLR[0])
    assert f.header.time_system == ballistic.sp3.SP3TimeSystem.UTC


@require_data(CHAMP, _CHAMP_HINT)
@require_data(CHAMP.with_suffix(".sp3.gz"), _CHAMP_HINT)
def test_gzip_and_plain_read_identically():
    gz = CHAMP.with_suffix(".sp3.gz")
    a, b = ballistic.sp3.read(CHAMP).arcs[0], ballistic.sp3.read(gz).arcs[0]
    assert a.epoch0 - b.epoch0 == 0.0
    assert np.array_equal(a.offsets_ns, b.offsets_ns)
    assert np.array_equal(a.position, b.position)
    assert np.array_equal(a.velocity, b.velocity)


def test_offsets_exact_and_frame():
    f = ballistic.sp3.read(GRACEFO)
    arc = f.arc("L64")
    assert np.all(np.diff(arc.offsets_ns) == 30_000_000_000)
    assert f.header.frame == "CTS"
    assert f.frame().itrf is None          # GFZ states no realization
    assert arc.has_velocity()


@require_data(GRACEFO_CLOCK, _GRACEFO_HINT)
def test_clock_running_into_z_is_kept():
    # The 2026-02-05 file has records like "441.732191-102852.949562"; a  # guard-exempt(PUB-DATE): data
    # whitespace reader dropped 8 of them. Every epoch must be present.
    f = ballistic.sp3.read(GRACEFO_CLOCK)
    assert len(f.arc("L64")) == f.header.declared_epochs == 1682


def test_interpolation_is_range_checked():
    arc = ballistic.sp3.read(GRACEFO).arc("L64")
    pos, vel = arc.interpolate(1800.0)
    assert np.linalg.norm(pos - arc.position[60]) < 1e-6
    with pytest.raises(ballistic.DataCoverageError):
        arc.interpolate(arc.offsets_s[-1] + 1.0)


def test_derived_velocity_edges_flagged():
    arc = ballistic.sp3.read(GRACEFO).arc("L64")
    vel, centered = arc.derived_velocities()
    assert not centered[:4].any() and not centered[-4:].any() and centered[4:-4].all()
    # Interior derivatives match GFZ's velocity records (true derivatives).
    assert np.abs(vel[4:-4] - arc.velocity[4:-4]).max() < 1e-3


def test_malformed_file_names_the_line(tmp_path):
    text = GRACEFO.read_text(encoding="utf-8").splitlines(keepends=True)
    i = next(k for k, line in enumerate(text) if line.startswith("PL64"))  # guard-exempt(SP3-LOCAL-PARSER): test fixture edit
    text[i] = "PL64  garbage\n"
    p = tmp_path / "bad.sp3"
    p.write_text("".join(text), encoding="utf-8")
    with pytest.raises(ballistic.BallisticError, match=f"line {i + 1}"):
        ballistic.sp3.read(p)


@require_data(GRACEFO_SECOND, _GRACEFO_HINT)
def test_campaign_starts_stay_inside_one_file():
    # Two overlapping GFZ files (separate solutions, 2 h overlap): every
    # start's +1 h truth comes from the start's own file, and a start time
    # in the overlap is scored once (trap SP3-ARC-BOUNDARY).
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import _sp3_records as sp3r
    finally:
        sys.path.pop(0)
    files = sorted((REPO / "tests/data/sp3/grace_fo").glob("GFZOP_RSO_L64_G_20260101_*_v03.sp3"))[:2]
    arcs = sp3r.load_arcs(ballistic, files, quiet=True)["L64"]
    assert len(arcs) == 2
    starts = list(sp3r.starts(arcs, 1800.0, 3600.0))
    assert starts
    for s in starts:
        assert any(s["end"] is r for r in s["arc"]["records"])
        assert s["duration"] == pytest.approx(3600.0)
    labels = [s["label"] for s in starts]
    assert len(labels) == len(set(labels))
