"""Command-line contract of scripts/validate_tle.py (no network, no propagation).

The TLE archive is built by a daily cron job (--fetch-only) and scored later
for forward horizons (--forward-only); these checks pin the flags, the
forward filter and the signed horizon bins so a refactor cannot silently mix
predictions before and after the TLE epoch.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def vt():
    spec = importlib.util.spec_from_file_location("validate_tle", REPO / "scripts" / "validate_tle.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("validate_tle", mod)
    spec.loader.exec_module(mod)
    return mod


def test_new_flags_parse(vt):
    p = vt.build_parser()
    a = p.parse_args(["--fetch-only", "--days", "3", "--forward-only"])
    assert a.fetch_only and a.forward_only and a.days == 3


def test_defaults_unchanged(vt):
    a = vt.build_parser().parse_args(["--tle", "x.txt", "--sp3", "y.sp3.gz"])
    assert not a.fetch_only and not a.forward_only and not a.fetch
    assert a.days == 8 and a.step == 600.0 and a.bin_hours == 12.0
    assert a.max_horizon_days == 7.0


def test_flags_in_module_docstring(vt):
    for flag in ("--fetch-only", "--forward-only", "--days"):
        assert flag in vt.__doc__


def test_forward_only_keeps_samples_after_epoch(vt):
    epoch = 1000.0
    pairs = [(epoch - 7200.0, epoch), (epoch - 1.0, epoch), (epoch, epoch),
             (epoch + 1.0, epoch), (epoch + 86400.0, epoch), (epoch + 9 * 86400.0, epoch)]
    week = 7 * 86400.0
    kept = [t for t, e in pairs if vt.keep_horizon(t - e, week, forward_only=True)]
    assert kept == [epoch + 1.0, epoch + 86400.0]
    every = [t for t, e in pairs if vt.keep_horizon(t - e, week)]
    assert every == [t for t, _ in pairs[:-1]]   # default: both sides, within the window


def test_signed_bins(vt):
    hours = np.array([-30.5, -13.0, -0.2, 0.1, 11.9, 12.0, 25.0])
    edges = vt.bin_edges(hours, 12.0)
    assert 0.0 in edges and edges[0] <= hours.min() and edges[-1] > hours.max()
    errors = {"sgp4": {"err3d_m": np.ones_like(hours)}}
    rows = vt.binned_rms(hours, errors, 12.0)
    labels = [r["label"] for r in rows]
    assert labels == ["-36.0..-24.0 h before", "-24.0..-12.0 h before", "-12.0..+0.0 h before",
                      "+0.0..+12.0 h after", "+12.0..+24.0 h after", "+24.0..+36.0 h after"]
    assert [r["n"] for r in rows] == [1, 1, 1, 2, 1, 1]
    assert all((r["side"] == "before") == (r["hi_h"] <= 0) for r in rows)
    assert vt.bin_label(-12.0, 0.0).endswith("before") and vt.bin_label(0.0, 12.0).endswith("after")
