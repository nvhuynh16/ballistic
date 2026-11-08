"""Test bed for the empirical-acceleration estimator on a real orbit.

"Semi-synthetic": the true trajectory is real (a GRACE-FO orbit from GFZ),
the sensor measurements are simulated from it with random noise. An
empirical acceleration (EA) is a correction acceleration, estimated from
observations, that accounts for forces the physical model misses.

The script builds a multi-sensor scenario on a 600 s arc of a committed GFZ
GRACE-FO Rapid Science Orbit (RSO) file: one ground radar (range, azimuth,
elevation, range rate) under the middle of the arc, sampled at 2 Hz, plus
one noisy position fix (3 m sigma) near the arc's end. It runs
estimate_ea_smoother (a Kalman filter plus backward smoother) and reports:
  nis             normalized innovation squared per degree of freedom: about
                  1 when the measurement residuals match the declared noise
  sigma_a, tau    the EA model's amplitude (m/s²) and correlation time (s)
                  the estimator selected
  sigma_rate_eff  the range-rate noise level actually used (m/s; 0 = off)
  rmse_ea         root-mean-square position error (RMSE, m) of a prediction
                  that uses the estimated EA, against the true orbit
  rmse_no_ea      the same without the EA
  honored         per sensor channel, the residual RMS at the solution
It checks the estimator on real orbital motion rather than on a synthetic
gliding trajectory.

Input: tests/data/sp3/grace_fo/GFZOP_RSO_L64_G_20260101_100000_20260102_000000_v03.sp3
(SP3 = the standard precise-orbit text format), read with ballistic.sp3:
epochs in the file's time system, velocities differentiated from the
positions (9-point Lagrange). Takes no command-line options. Also imported by
tests/python/test_orbital_fixture.py.

Usage:
    PYTHONPATH=./python uv run --with numpy python scripts/orbital_fixture.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


SP3_FILE = (REPO / "tests" / "data" / "sp3" / "grace_fo" /
            "GFZOP_RSO_L64_G_20260101_100000_20260102_000000_v03.sp3")


def load_truth(ballistic, arc_s: float = 600.0, skip_s: float = 1800.0,
               sp3_file=None):
    """Read the true orbit for an arc of `arc_s` seconds starting `skip_s` into the file.

    Starting away from the file's beginning keeps the arc clear of the
    less accurate one-sided velocity derivatives at the file edge; only
    epochs with a centered derivative are kept. Read with ballistic.sp3:
    epochs in the file's time system (GFZ writes GPS time; reading it as UTC
    would be 18 s off), exact offsets, velocities differentiated from the
    positions (9-point Lagrange).

    Returns (times_s, pos_m, vel_ms, epoch0): numpy (N,) seconds since
    epoch0, numpy (N,3) Earth-fixed positions in meters, numpy (N,3)
    velocities in m/s, and the ballistic.Epoch of the arc start. `sp3_file`
    overrides the default file. Raises ValueError if fewer than 4 epochs fall
    in the window.
    """
    arc = ballistic.sp3.read(sp3_file or SP3_FILE).arcs[0]
    vel_all, centered = arc.derived_velocities()
    t_all = arc.offsets_s
    keep = (t_all >= skip_s) & (t_all <= skip_s + arc_s) & (centered == 1)
    if keep.sum() < 4:
        raise ValueError(
            f"only {int(keep.sum())} SP3 epochs in the requested window — "
            "widen arc_s/skip_s or check the file's cadence")
    times = t_all[keep] - skip_s
    return times, arc.position[keep], vel_all[keep], arc.epoch0 + skip_s


def _interp_state(times, pos, vel, t):
    """Interpolate position and velocity at time `t` (s) between SP3 epochs.

    Cubic Hermite interpolation: a cubic matching the positions and
    velocities at the two neighboring epochs. `times` is numpy (N,) seconds,
    `pos`/`vel` numpy (N,3). Returns (position (3,) m, velocity (3,) m/s).
    """
    i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
    h = times[i + 1] - times[i]
    s = (t - times[i]) / h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    p = (h00 * pos[i] + h10 * h * vel[i] + h01 * pos[i + 1]
         + h11 * h * vel[i + 1])
    d00 = 6 * s**2 - 6 * s
    d10 = 3 * s**2 - 4 * s + 1
    d01 = -6 * s**2 + 6 * s
    d11 = 3 * s**2 - 2 * s
    v = (d00 * pos[i] + d10 * h * vel[i] + d01 * pos[i + 1]
         + d11 * h * vel[i + 1]) / h
    return p, v


def build_scenario(ballistic, arc_s: float = 600.0, radar_hz: float = 2.0,
                   seed: int = 42, skip_s: float = 1800.0, sp3_file=None):
    """Build the simulated sensor scenario on the real orbit.

    One radar sits on the ground directly below the satellite's mid-arc
    position and measures at `radar_hz` (Hz) for `arc_s` seconds; one
    position fix (3 m noise) is taken 1 s before the arc's end. Noise is drawn
    from numpy's generator seeded with `seed`. `skip_s` and `sp3_file` are
    passed to load_truth.

    Returns (scenario, truth): a ballistic.estimation.MultiSensorScenario and the
    (times_s, pos_m, vel_ms, epoch0) tuple from load_truth.
    """
    times, pos, vel, epoch0 = load_truth(ballistic, arc_s, skip_s, sp3_file)
    rng = np.random.default_rng(seed)

    # Station: the ground point below the mid-arc position. A spherical-Earth
    # latitude is accurate enough to place it; the station's local
    # east-north-up (ENU) rotation below is built from the same angles.
    mid_p, _ = _interp_state(times, pos, vel, arc_s / 2.0)
    lat = math.asin(mid_p[2] / np.linalg.norm(mid_p))
    lon = math.atan2(mid_p[1], mid_p[0])
    import ballistic
    ce = ballistic.constants.earth
    site = np.array([ce.RADIUS_EQUATORIAL * math.cos(lat) * math.cos(lon),
                     ce.RADIUS_EQUATORIAL * math.cos(lat) * math.sin(lon),
                     ce.RADIUS_POLAR * math.sin(lat)])

    radar = ballistic.estimation.RadarObs()
    radar.position_ecef = site
    radar.lat_rad = lat
    radar.lon_rad = lon
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    radar.ecef_to_enu = np.array([[-so, co, 0.0],
                                  [-sl * co, -sl * so, cl],
                                  [cl * co, cl * so, sl]])
    n = int(arc_s * radar_hz)
    samples = []          # .samples is a copy on read — assign, not append
    for k in range(n):
        t = k / radar_hz
        p, v = _interp_state(times, pos, vel, t)
        st = ballistic._core.StateVector(p, v)
        obs = np.asarray(radar.predict(st))
        obs[0] += rng.normal(0.0, radar.sigma_range)
        obs[1] += rng.normal(0.0, radar.sigma_angle)
        obs[2] += rng.normal(0.0, radar.sigma_angle)
        obs[3] += rng.normal(0.0, radar.sigma_rate)
        samples.append((epoch0 + t, obs))
    radar.samples = samples

    fix = ballistic.estimation.PositionFixObs()
    fix.sigma_pos = 3.0
    tf = arc_s - 1.0
    pf, _ = _interp_state(times, pos, vel, tf)
    fix.samples = [(epoch0 + tf, pf + rng.normal(0.0, 3.0, 3))]

    p0, v0 = _interp_state(times, pos, vel, 0.0)
    sc = ballistic.estimation.MultiSensorScenario(
        epoch0=epoch0, pos0_ecef=p0, vel0_ecef=v0, arc_duration_s=arc_s)
    sc.add_radar(radar)
    sc.add_position_fix(fix)
    return sc, (times, pos, vel, epoch0)


def _reprop_rmse(ballistic, sc, truth, ea):
    """RMSE (m) of a prediction from the scenario's start state against the truth.

    Propagates over the arc with empirical acceleration `ea` (None = none)
    and compares Earth-fixed positions every 10 s. `truth` is load_truth's
    tuple.
    """
    times, pos, vel, epoch0 = truth
    cfg = ballistic.TrajectoryConfig(empirical_accel=ea)
    traj = ballistic.Trajectory(ballistic.State.from_itrf(sc.epoch0, sc.pos0_ecef, sc.vel0_ecef),
                                config=cfg)
    traj.propagate_for(sc.arc_duration_s)
    sse, n = 0.0, 0
    for t in np.arange(10.0, sc.arc_duration_s, 10.0):
        st = traj(float(t), frame="ECEF")
        p_t, _ = _interp_state(times, pos, vel, float(t))
        sse += float(np.sum((np.asarray(st.position) - p_t) ** 2))
        n += 1
    return math.sqrt(sse / n)


def measure(verbose=False):
    """Run the estimator on the default scenario and return its metrics.

    Returns a dict with the keys listed in the module docstring; prints them
    when `verbose`. Raises AssertionError if the estimator reports an error.
    """
    import ballistic
    ballistic.initialize()
    sc, truth = build_scenario(ballistic)
    mask = ballistic.estimation.SensorMask.all_on(sc)
    fm = ballistic.TrajectoryConfig()._to_core()

    r = ballistic.estimation.estimate_ea_smoother(sc, mask, fm,
                                       ballistic.estimation.EASmootherConfig())
    assert r.error_message == "", r.error_message
    out = {
        "nis": r.nis,
        "sigma_a": r.sigma_a_used,
        "tau": r.tau_used,
        "sigma_rate_eff": r.sigma_rate_eff,
        "rmse_ea": _reprop_rmse(ballistic, sc, truth, r.empirical_accel),
        "rmse_no_ea": _reprop_rmse(ballistic, sc, truth, None),
        "honored": dict(r.sigma_honored),
    }
    if verbose:
        for k, v in out.items():
            print(f"  {k:14s} {v}")
    return out


if __name__ == "__main__":
    print("Orbital semi-synthetic fixture (real GRACE-FO truth)")
    measure(verbose=True)
