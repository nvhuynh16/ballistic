"""
Regression for the EOF parser + propagation-vs-truth RMSE Python binding
(python/src/validation_bindings.cpp).

Mirrors the C++ tests/validation/test_sentinel1_validation.cpp gates: parse a
Sentinel-1 POE EOF, propagate from an early state vector with ITRF2014, and
RMSE-compare against the EOF truth. Thresholds match the C++ 600 s gate
(< 50 m) so this stays robust on the embedded-DE440 path (no JPL DE421).
"""

import math
import os

import numpy as np
import pytest

import ballistic

EOF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "eof", "sentinel1_2016", "2016-1"
)


@pytest.fixture(scope="module")
def eof():
    parsed = ballistic.readers.find_and_parse_eof(os.path.abspath(EOF_DIR))
    if parsed is None:
        pytest.skip(f"missing test data: no parseable EOF under {EOF_DIR} — reacquire: "
                    "Sentinel-1 POEORB files for 2016 from the Copernicus Data Space "
                    "Ecosystem (dataspace.copernicus.eu)")
    return parsed


def test_parse_basic(eof):
    assert eof.mission.startswith("Sentinel-1")
    assert eof.reference_frame  # e.g. EARTH_FIXED
    assert len(eof.state_vectors) > 1000
    sv0 = eof.state_vectors[0]
    r = np.linalg.norm(sv0.position)
    v = np.linalg.norm(sv0.velocity)
    assert 6.9e6 < r < 7.3e6, f"LEO radius out of range: {r}"
    assert 7.0e3 < v < 8.0e3, f"LEO speed out of range: {v}"


def test_find_nearest_and_interpolate(eof):
    sv = eof.state_vectors[100]
    near = eof.find_nearest(sv.epoch)
    assert near is not None
    assert np.linalg.norm(near.position - sv.position) < 1e-6

    # midpoint between two samples should interpolate to a finite ECEF point
    mid = eof.state_vectors[100].epoch + 0.5 * (
        eof.state_vectors[101].epoch - eof.state_vectors[100].epoch
    )
    p = eof.interpolate_position(mid)
    assert p is not None
    assert np.all(np.isfinite(p))
    assert 6.9e6 < np.linalg.norm(p) < 7.3e6


def test_propagation_rmse_vs_truth(eof):
    sv0 = eof.state_vectors[10]

    cfg = ballistic.TrajectoryConfig(itrf=ballistic.ITRFRealization.ITRF2014)  # Sentinel pre-2025

    traj = ballistic.Trajectory(ballistic.State.from_itrf(sv0.epoch, sv0.position, sv0.velocity), config=cfg)

    duration_s, step_s = 600.0, 10.0
    offsets = np.arange(0.0, duration_s + step_s, step_s)
    states = traj([float(t) for t in offsets], frame="ECEF")

    predicted = [
        (sv0.epoch + float(t), ballistic.StateVector(s.position, s.velocity))
        for t, s in zip(offsets, states)
    ]

    res = ballistic.readers.compute_propagation_result(predicted, eof)

    assert res.num_comparisons > 0
    assert math.isfinite(res.position_rmse)
    assert math.isfinite(res.velocity_rmse)
    # 600 s gate from the C++ validation suite (embedded-DE440 path).
    assert res.position_rmse < 50.0, f"position RMSE too high: {res.position_rmse} m"
    # RTN components are populated and consistent with the total.
    rtn = math.sqrt(
        res.radial_rms**2 + res.along_track_rms**2 + res.cross_track_rms**2
    )
    assert rtn == pytest.approx(res.position_rmse, rel=0.05)
    assert "Position RMSE" in repr(res)
