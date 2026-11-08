"""Trajectory.stm / Trajectory.covariance (C++ cells: tests/unit/test_state_transition.cpp).

Phi(t, t0) maps a small change of the start state into the change at t, in
the GCRS frame with the state ordered [r (m); v (m/s)]; the covariance is
Phi P_0 Phi^T.
"""

import math

import numpy as np
import pytest

import ballistic
from ballistic import State
from ballistic.errors import ConfigError, DataCoverageError, InputPlausibilityError

T0 = ballistic.Epoch.from_utc(2025, 6, 15, 0, 0, 0)
GM = ballistic.constants.earth.GM
ARC = 1800.0


def _cfg():
    # Fixed 10 s steps, so a perturbed run shares the nominal run's step grid.
    return ballistic.TrajectoryConfig(gravity_degree=8, enable_relativistic=False,
                                      enable_solid_tides=False, ocean_tide_degree=0,
                                      rk45_initial_step_size=10.0, rk45_max_step_size=10.0)


def _start(dx=0.0):
    rn = ballistic.constants.earth.RADIUS_EQUATORIAL + 500e3
    return State.from_gcrs(T0, [rn + dx, 0.0, 0.0], [0.0, math.sqrt(GM / rn), 0.0])


def _flown(dx=0.0):
    traj = ballistic.Trajectory(_start(dx), _cfg())
    traj.propagate_for(ARC)
    return traj


def test_shapes_and_identity_at_start():
    traj = _flown()
    assert traj.stm(0.0).shape == (6, 6)
    np.testing.assert_array_equal(traj.stm(0.0), np.eye(6))
    assert traj.stm([0.0, 600.0, ARC]).shape == (3, 6, 6)
    assert traj.covariance(ARC, np.eye(6)).shape == (6, 6)


def test_epoch_and_seconds_agree():
    traj = _flown()
    np.testing.assert_array_equal(traj.stm(900.0), traj.stm(T0 + 900.0))


def test_covariance_is_symmetric_and_grows_along_track():
    traj = _flown()
    P_0 = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])
    P = traj.covariance(ARC, P_0)
    np.testing.assert_array_equal(P, P.T)
    assert np.linalg.eigvalsh(P).min() > -1e-9 * np.abs(P).max()
    assert np.trace(P[:3, :3]) > np.trace(P_0[:3, :3])


def test_column_matches_reprop_of_a_perturbed_start():
    # Central difference of the end state for a 1 m change of the start x.
    traj = _flown()
    plus, minus = _flown(1.0).state_gcrs(), _flown(-1.0).state_gcrs()
    col = np.concatenate([(np.asarray(plus.position) - np.asarray(minus.position)) / 2.0,
                          (np.asarray(plus.velocity) - np.asarray(minus.velocity)) / 2.0])
    phi_col = traj.stm(ARC)[:, 0]
    assert np.linalg.norm(phi_col[:3] - col[:3]) / np.linalg.norm(col[:3]) < 1e-4
    assert np.linalg.norm(phi_col[3:] - col[3:]) / np.linalg.norm(col[3:]) < 1e-4


def test_cache_follows_the_history():
    traj = _flown()
    end = traj.stm(ARC)
    with pytest.raises(DataCoverageError):
        traj.stm(ARC + 60.0)
    traj.propagate_for(60.0)
    np.testing.assert_allclose(traj.stm(ARC), end, rtol=0, atol=1e-12 * np.abs(end).max())
    assert traj.stm(ARC + 60.0).shape == (6, 6)


def test_errors():
    with pytest.raises(ConfigError):
        ballistic.Trajectory(_start(), _cfg()).stm(0.0)   # nothing recorded yet
    traj = _flown()
    with pytest.raises(DataCoverageError):
        traj.stm(-1.0)
    bad = np.eye(6)
    bad[0, 1] = 0.5
    with pytest.raises(InputPlausibilityError):
        traj.covariance(600.0, bad)
    batch = ballistic.Trajectory([_start(), _start(10.0)], _cfg())
    with pytest.raises(ConfigError):
        batch.stm(0.0)
