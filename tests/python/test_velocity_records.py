"""Velocity-record conventions of the orbit products the campaigns use.

Trajectory needs the true time derivative of the ECEF position as its start
velocity. GFZ's velocity records are true derivatives; wrongly subtracting
the CIP drift from them biases every arc by ~0.5 m per hour at GRACE-FO and
can hide a 25% drag shortfall. These tests pin the measured

convention of each committed product, check that check_velocity_records
recognizes all three cases, and check that each script defaults to the mode
its products need (the scripts also run the check on every file they load).
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import ballistic

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
V = ballistic.readers.VelocityRecordVerdict


def _records(path):
    """(epoch0, exact offsets s, positions m, velocities m/s) of the first
    satellite in an SP3, read with ballistic.sp3 (time system honored)."""
    arc = ballistic.sp3.read(path).arcs[0]
    return arc.epoch0, list(arc.offsets_s), arc.position, arc.velocity


def _one(pattern):
    files = sorted(glob.glob(str(REPO / pattern)))
    if not files:
        readme = Path(pattern).parent / "README.md"
        pytest.skip(f"missing test data {pattern} — reacquire: see {readme}")
    return files[0]


@pytest.mark.parametrize("pattern", [
    "tests/data/sp3/grace_fo/GFZOP_RSO_L64_G_20260102_100000_*.sp3",
    "tests/data/sp3/champ/CHAMP_2005_182.sp3",
    "tests/data/sp3/champ_2010/*.sp3.gz",
])
def test_gfz_records_are_true_derivatives(pattern):
    check = ballistic.readers.check_velocity_records(*_records(_one(pattern)))
    assert check.verdict == V.TRUE_DERIVATIVE, check.summary
    assert abs(check.median_difference_rtn[1]) < 5e-6


def test_ilrs_slr_records_are_not():
    check = ballistic.readers.check_velocity_records(
        *_records(_one("tests/data/sp3/lageos/ilrsa.orb.lageos1.*.sp3")))
    assert check.verdict == V.INCONSISTENT, check.summary


def test_check_recognizes_records_that_omit_the_drift():
    e0, off, pos, vel = _records(_one("tests/data/sp3/grace_fo/GFZOP_RSO_L64_G_20260102_100000_*.sp3"))
    drift = np.array([ballistic.readers.cip_drift_velocity(e0 + o, p) for o, p in zip(off, pos)])
    check = ballistic.readers.check_velocity_records(e0, off, pos, vel + drift)
    assert check.verdict == V.OMITS_CIP_DRIFT, check.summary


def test_check_flags_any_other_offset():
    e0, off, pos, vel = _records(_one("tests/data/sp3/grace_fo/GFZOP_RSO_L64_G_20260102_100000_*.sp3"))
    along = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    check = ballistic.readers.check_velocity_records(e0, off, pos, vel + 3e-5 * along)
    assert check.verdict == V.INCONSISTENT, check.summary


@pytest.mark.parametrize("script, mode", [
    ("validate_gracefo_drag.py", "file"),
    ("validate_champ_drag.py", "file"),
    ("validate_sphere_srp.py", "derived"),
])
def test_script_defaults_match_their_products(script, mode):
    text = (REPO / "scripts" / script).read_text(encoding="utf-8")
    m = re.search(r"_sp3v\.add_arg\(parser, default='(\w[\w-]*)'\)", text)
    assert m and m.group(1) == mode, f"{script} must default to --sp3-velocity {mode}"
    assert "_sp3v.verify(" in text, f"{script} must check the records it loads"
