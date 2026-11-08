"""Trajectory.propagate_until and ballistic.events (C++ cells: tests/unit/test_events.cpp).

Trap GROUND-PASS-THROUGH: nothing used to stop a propagation at the ground.
"""

import math

import numpy as np
import pytest

import ballistic


def _vacuum_config():
    return ballistic.TrajectoryConfig(gravity_degree=0, enable_third_body=False,
                                      enable_relativistic=False, enable_solid_tides=False,
                                      ocean_tide_degree=0)


def test_vacuum_drop_ends_at_ground_impact():
    t0 = ballistic.Epoch.from_utc(2024, 1, 10, 0, 0, 0)
    b = ballistic.constants.earth.RADIUS_POLAR
    r0 = b + 200e3
    traj = ballistic.Trajectory(ballistic.State.from_itrf(t0, np.array([0.0, 0.0, r0]), np.zeros(3),
                                                       allow_suborbital=True), config=_vacuum_config())
    res = traj.propagate_until(3600.0)
    assert not res.reached_target()
    assert res.stopped_by.name == "ground impact"
    gm = ballistic.constants.earth.GM
    x = b / r0
    t_fall = math.sqrt(r0**3 / (2 * gm)) * (math.sqrt(x * (1 - x)) + math.acos(math.sqrt(x)))
    assert res.epoch - t0 == pytest.approx(t_fall, abs=1e-3)
    assert -0.5 < ballistic.ecef_to_geodetic(np.asarray(res.state_ecef.position)).altitude <= 0.0


def test_altitude_event_logged_and_terminal():
    t0 = ballistic.Epoch.from_utc(2024, 1, 10, 0, 0, 0)
    traj = ballistic.Trajectory(ballistic.State.from_itrf(
        t0, np.array([0.0, 0.0, ballistic.constants.earth.RADIUS_POLAR + 200e3]), np.zeros(3),
        allow_suborbital=True), config=_vacuum_config())
    ev = ballistic.events.altitude_crossing(100e3, ballistic.events.Direction.Falling)
    res = traj.propagate_until(3600.0, [ev])
    assert res.stopped_by.name == ev.name
    alt = ballistic.ecef_to_geodetic(np.asarray(res.stopped_by.position_ecef)).altitude
    assert alt == pytest.approx(100e3, abs=0.5)


def test_propagate_to_altitude_takes_a_duration():
    t0 = ballistic.Epoch.from_utc(2024, 1, 10, 0, 0, 0)
    traj = ballistic.Trajectory(ballistic.State.from_itrf(t0, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])))
    with pytest.raises(TypeError):
        traj.propagate_to_altitude(350e3, ballistic.AltitudeCondition.Above, max_steps=10)
