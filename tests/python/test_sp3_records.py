"""scripts/_sp3_records.join_arcs: several SP3 files joined into one arc.

A horizon longer than one file needs the files of one satellite joined. The
committed GFZ CHAMP files are 14-h arcs starting every 12 h, so consecutive
files share 2 h (242 epochs at 30 s). The joiner must keep one record per
epoch, prefer each file's interior, record where the source changes, and
refuse to mix frame or time-system labels.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import ballistic

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import _sp3_records as sp3r  # noqa: E402

CHAMP_2010 = sorted((REPO / "tests/data/sp3/champ_2010").glob("GFZOP_RSO_L06_G_*.sp3.gz"))[:3]


@pytest.fixture(scope="module")
def champ():
    per_file = sp3r.load_arcs(ballistic, CHAMP_2010, quiet=True)["L06"]
    joined = sp3r.load_arcs(ballistic, CHAMP_2010, quiet=True, join=True)["L06"]
    return per_file, joined


def test_join_is_opt_in(champ):
    per_file, joined = champ
    assert len(per_file) == 3 and all(len(a["records"]) == 1682 for a in per_file)
    assert len(joined) == 1


def test_joined_record_count_and_time_axis(champ):
    per_file, (arc,) = champ
    # 3 x 1682 records minus the two 2-h overlaps (242 shared epochs each).
    assert len(arc["records"]) == 3 * 1682 - 2 * 242 == 4562
    off = arc["offsets"]
    assert off[0] == 0.0
    assert np.all(np.diff(off) > 0)                       # monotonic, no duplicates
    assert np.allclose(np.diff(off), 30.0, atol=1e-6)     # no gap, no dropped epoch
    assert off[-1] == pytest.approx(2 * 43200.0 + 50430.0, abs=1e-6)
    assert [r["offset_s"] for r in arc["records"]] == pytest.approx(list(off), abs=1e-9)
    assert arc["epoch0"] - per_file[0]["epoch0"] == pytest.approx(0.0, abs=1e-9)
    assert arc["sources"] == [a["source"] for a in per_file]
    assert arc["sp3_arc"] is None


def test_overlap_split_at_its_middle(champ):
    per_file, (arc,) = champ
    # Each 2-h overlap (43200..50430 s into the earlier file) is split where
    # the two files are equally deep: the interior of each file wins.
    assert arc["boundaries"] == [1561, 3001]
    assert arc["offsets"][1561] == pytest.approx(46830.0, abs=1e-6)
    assert arc["offsets"][3001] == pytest.approx(90030.0, abs=1e-6)
    src = arc["record_source"]
    assert list(src[[0, 1560, 1561, 3000, 3001, -1]]) == [0, 0, 1, 1, 2, 2]
    # The kept record is the file's own record at that epoch, bit for bit.
    i = 1561
    j = int(np.argmin(np.abs(per_file[1]["offsets"] + float(per_file[1]["epoch0"] - arc["epoch0"])
                             - arc["offsets"][i])))
    assert np.array_equal(arc["records"][i]["pos_m"], per_file[1]["records"][j]["pos_m"])


def test_starts_run_past_one_file(champ):
    _, (arc,) = champ
    s = list(sp3r.starts([arc], 3600.0, 24 * 3600.0, start_tol=1.0, end_tol=1.0))
    assert s and all(x["duration"] == pytest.approx(86400.0, abs=1e-6) for x in s)


def _tiny_arc(name, frame_label, time_system, epoch0, n=5, step=30.0):
    off = np.arange(n) * step
    recs = [{"pos_m": np.array([7.0e6 + i, 0.0, 0.0]), "vel_ms": np.zeros(3),
             "vel_from_file": False, "vel_edge": False, "offset_s": float(off[i])}
            for i in range(n)]
    return dict(source=f"{name}-{frame_label}-{time_system}.sp3", satellite=name, name=name,
                epoch0=epoch0, offsets=off, records=recs, frame=None, frame_label=frame_label,
                time_system=time_system, sp3_arc=None)


def test_mixed_labels_refused():
    e = ballistic.Epoch.from_utc(2024, 1, 15, 0, 0, 0)
    a = _tiny_arc("L99", "IGS20", "GPS", e)
    b = _tiny_arc("L99", "ITRF14", "GPS", e + 120.0)
    with pytest.raises(ValueError, match=r"frame.*'IGS20'.*'ITRF14'"):
        sp3r.join_arcs([a, b])
    c = _tiny_arc("L99", "IGS20", "UTC", e + 120.0)
    with pytest.raises(ValueError, match=r"time system.*'GPS'.*'UTC'"):
        sp3r.join_arcs([a, c])


def test_gap_refused_and_order_independent():
    e = ballistic.Epoch.from_utc(2024, 1, 15, 0, 0, 0)
    a = _tiny_arc("L99", "IGS20", "GPS", e)
    far = _tiny_arc("L99", "IGS20", "GPS", e + 2000.0)
    with pytest.raises(ValueError, match="gap"):
        sp3r.join_arcs([a, far])
    near = _tiny_arc("L99", "IGS20", "GPS", e + 120.0)      # shares 120 s with `a`
    j1 = sp3r.join_arcs([a, near])
    j2 = sp3r.join_arcs([near, a])
    assert len(j1["records"]) == 9
    assert np.array_equal(j1["offsets"], j2["offsets"])
    assert j1["sources"] == j2["sources"] == [a["source"], near["source"]]
