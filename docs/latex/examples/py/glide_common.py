"""
Shared glide helper for the Python EA tutorial examples (ex04/05/06/07).

Reproduces the canonical "glide" reentry scenario using ONLY the public
``ballistic`` API + the committed glide_fixture.json constants (derived
bit-for-bit from the test-only bed by gen_glide_fixture). Mirrors the C++
docs/latex/examples/cpp/glide_common.hpp so the two tutorials match.
"""
import json
import math
import os

import numpy as np

import ballistic

_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                    "glide_fixture.json")


def load_fixture(path=_FIX):
    with open(os.path.abspath(path)) as f:
        return json.load(f)


FIX = load_fixture()


def glide_epoch():
    y, mo, d, h, mi, s = FIX["epoch_utc"]
    return ballistic.Epoch.from_utc(y, mo, d, h, mi, s)


def base_config():
    """Gravity-only deg-20; everything else off. No EA == ballistic."""
    return ballistic.TrajectoryConfig(gravity_degree=20, enable_third_body=False,
                                      enable_relativistic=False, enable_solid_tides=False)



def make_truth_ea():
    """Two Gaussian bumps + recoil, GCRS — the make_forced_config formula."""
    ea = FIX["ea"]
    rd = np.array(ea["recoil_direction"])
    LIFT, LAT_, FC, FW = ea["LIFT"], ea["LAT"], ea["FC"], ea["FW"]
    LP, RET, RC, RW = ea["LP"], ea["RET"], ea["RC"], ea["RW"]
    entries = []
    for i in range(701):
        t = float(i)
        g = math.exp(-0.5 * ((t - FC) / FW) ** 2)
        rg = math.exp(-0.5 * ((t - RC) / RW) ** 2)
        a = np.array([0.0,
                      LIFT * g,
                      LAT_ * g * math.sin(2 * math.pi * t / LP)]) + RET * rg * rd
        entries.append(ballistic.estimation.EmpiricalAccelEntry(t, a))
    return ballistic.EmpiricalAcceleration.from_entries(
        entries, ballistic.EmpiricalAccelFrame.GCRS)


class GlideTruth:
    """Truth trajectory + clean ECEF samplers. with_ea=False -> ballistic."""

    def __init__(self, with_ea=True):
        self.t0 = glide_epoch()
        self.pos = np.array(FIX["ic"]["pos_ecef"])
        self.vel = np.array(FIX["ic"]["vel_ecef"])
        self.eval_end = FIX["arc"]["eval_end_s"]
        cfg = base_config()
        if with_ea:
            cfg = cfg.replace(empirical_accel=make_truth_ea())
        self.traj = ballistic.Trajectory(ballistic.State.from_itrf(self.t0, self.pos, self.vel, allow_suborbital=True), config=cfg)
        while (self.traj.epoch() - self.t0) < self.eval_end + 1.0:
            self.traj.step()

    def sample(self, dt):
        """Return (times_s, pos, vel, acc) ECEF arrays; dt 0.01=100Hz.
        Grid exactly mirrors the C++ `for(t=0; t<=eval_end; t+=dt)` loop
        (strictly within the arc) so C++/Python observation sets match."""
        offs = np.arange(0.0, self.eval_end, dt)
        states = self.traj([float(t) for t in offs], frame="ECEF")
        ts, P, V, A = [], [], [], []
        for t, s in zip(offs, states):
            if s is None:
                continue
            ts.append(float(t))
            P.append(np.asarray(s.position))
            V.append(np.asarray(s.velocity))
            A.append(np.asarray(s.acceleration))
        return np.array(ts), np.array(P), np.array(V), np.array(A)

    def truth_pos(self, t):
        s = self.traj(float(t), frame="ECEF")
        return np.asarray(s.position) if s is not None else None


def to_obs(times_s, t0, arr):
    """numpy (N,) times + (N,3) → list[(Epoch, np.array)] for the estimators."""
    return [(t0 + float(t), arr[i]) for i, t in enumerate(times_s)]


def add_noise(arr, sigma, seed):
    rng = np.random.default_rng(seed)
    return arr + rng.normal(0.0, sigma, size=arr.shape)


def ea0_baseline_rmse(forced):
    """EA=0 baseline: glide IC propagated with NO empirical acceleration,
    RMSE vs the FORCED-glide truth on the same [10, eval_end] @1 Hz grid.
    The 'do nothing' control — the reference scale for every recovered RMSE."""
    ball = GlideTruth(with_ea=False)
    sse, n = 0.0, 0
    for t in np.arange(10.0, forced.eval_end, 1.0):
        a = ball.truth_pos(t)
        b = forced.truth_pos(t)
        if a is not None and b is not None:
            sse += float(np.sum((a - b) ** 2))
            n += 1
    return math.sqrt(sse / n) if n else 1e9


def reprop_rmse(truth, ea, max_step_s=0.0):
    """Re-propagate from the glide IC with `ea`; pos RMSE vs truth @1Hz."""
    if ea is None:
        return 1e9
    cfg = base_config()
    cfg = cfg.replace(empirical_accel=ea)
    if max_step_s > 0.0:
        cfg = cfg.replace(rk45_max_step_size=max_step_s)
    rt = ballistic.Trajectory(ballistic.State.from_itrf(truth.t0, truth.pos, truth.vel, allow_suborbital=True), config=cfg)
    while (rt.epoch() - truth.t0) < truth.eval_end:
        rt.step()
    sse, n = 0.0, 0
    for t in np.arange(10.0, truth.eval_end, 1.0):
        a = rt(float(t), frame="ECEF")
        b = truth.truth_pos(t)
        if a is not None and b is not None:
            sse += float(np.sum((np.asarray(a.position) - b) ** 2))
            n += 1
    return math.sqrt(sse / n) if n else 1e9
