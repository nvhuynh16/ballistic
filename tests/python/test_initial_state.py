"""ballistic.State and the start-state API (C++ cells: tests/unit/test_initial_state.cpp).

A start state names its frame and is checked before anything runs: units,
the ground, a suborbital orbit, and an Earth-fixed/inertial mix-up (the two
readings differ by omega x r, ~470 m/s at LEO). The removed forms raise
naming the replacement; the config is frozen so a typo cannot be ignored.
"""

import dataclasses
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import ballistic
from ballistic import State

REPO = Path(__file__).resolve().parents[2]
T0 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
W = np.array([0.0, 0.0, ballistic.constants.earth.ANGULAR_VELOCITY])
GM = ballistic.constants.earth.GM


def _cfg(**kw):
    return ballistic.TrajectoryConfig(gravity_degree=8, enable_third_body=False,
                                      enable_relativistic=False, enable_solid_tides=False,
                                      ocean_tide_degree=0, **kw)


def _circular(alt_km, lat_deg, east=True):
    """Inertial circular state at (lat, lon 0): r, v_inertial, v_earth_fixed."""
    lat = math.radians(lat_deg)
    rn = ballistic.constants.earth.RADIUS_EQUATORIAL + alt_km * 1e3
    r = rn * np.array([math.cos(lat), 0.0, math.sin(lat)])
    d = np.array([0.0, 1.0, 0.0]) if east else np.array([-math.sin(lat), 0.0, math.cos(lat)])
    v = math.sqrt(GM / rn) * d
    return r, v, v - np.cross(W, r)


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return f()


# --- plausibility ------------------------------------------------------------

def test_real_states_pass_in_their_own_frame():
    for alt in (400, 20200, 35786):
        for lat in (0, 45, 80):
            for east in (True, False):
                r, vi, ve = _circular(alt, lat, east)
                _quiet(lambda: State.from_itrf(T0, r, ve))
                _quiet(lambda: State.from_gcrs(T0, r, vi))
    for nu in range(0, 360, 15):   # Molniya, every anomaly, both readings declared right
        s = State.from_keplerian(T0, a_m=26600e3, e=0.74, i_deg=63.4, raan_deg=40.0,
                                 argp_deg=270.0, true_anomaly_deg=nu, check_frame=False)
        r, v = np.asarray(s.position), np.asarray(s.velocity)
        _quiet(lambda: State.from_gcrs(T0, r, v))
        _quiet(lambda: State.from_itrf(T0, r, v - np.cross(W, r)))


def test_inertial_velocity_given_as_earth_fixed_is_caught():
    r, vi, _ = _circular(400, 0)
    with pytest.warns(ballistic.FrameAssumptionWarning, match="from_gcrs"):
        State.from_itrf(T0, r, vi)               # prograde: a bound but eccentric orbit
    with pytest.raises(ballistic.FrameError, match="omega x r"):
        State.from_itrf(T0, r, -vi)              # retrograde: perigee underground
    r, vi, ve = _circular(35786, 0)
    with pytest.raises(ballistic.FrameError):
        State.from_itrf(T0, r, vi)               # GEO: hyperbolic as declared
    with pytest.raises(ballistic.FrameError, match="from_itrf"):
        State.from_gcrs(T0, r, ve)
    _quiet(lambda: State.from_itrf(T0, r, vi, check_frame=False, allow_suborbital=False))


def test_units_ground_and_suborbital():
    r, vi, ve = _circular(400, 30)
    with pytest.raises(ballistic.InputPlausibilityError, match="METERS"):
        State.from_itrf(T0, r / 1e3, ve / 1e3)
    with pytest.raises(ballistic.InputPlausibilityError, match="km/s"):
        State.from_itrf(T0, r, ve / 1e3)
    with pytest.raises(ballistic.InputPlausibilityError, match="below the ground"):
        State.from_itrf(T0, r * 0.9, ve)
    with pytest.raises(ballistic.InputPlausibilityError, match="allow_suborbital"):
        State.from_itrf(T0, r * 0.97, ve * 0.9)
    _quiet(lambda: State.from_itrf(T0, r * 0.97, ve * 0.9, allow_suborbital=True))
    with pytest.raises(ballistic.InputPlausibilityError, match="not finite"):
        State.from_gcrs(T0, [np.nan, 0, 0], vi)
    with pytest.raises(ballistic.ConfigError):
        State.from_keplerian(T0, a_m=7e6, e=1.2, i_deg=0, raan_deg=0, argp_deg=0, true_anomaly_deg=0)


def test_keplerian_and_j2000():
    s = State.from_keplerian(T0, a_m=7000e3, e=0.01, i_deg=51.6, raan_deg=30.0, argp_deg=90.0,
                             true_anomaly_deg=45.0)
    r, v = np.asarray(s.position), np.asarray(s.velocity)
    energy = 0.5 * v @ v - GM / np.linalg.norm(r)
    assert -GM / (2 * energy) == pytest.approx(7000e3, rel=1e-12)
    h = np.cross(r, v)
    assert math.degrees(math.acos(h[2] / np.linalg.norm(h))) == pytest.approx(51.6, abs=1e-10)
    assert s.frame == "GCRS" and s.realization is None
    # EME2000 -> GCRS is the IAU 2006 frame bias (ERFA eraBp00 rb, transposed).
    rb = np.array([[0.9999999999999942498, -0.7078279744199196626e-7, 0.8056217146976134152e-7],
                   [0.7078279477857337206e-7, 0.9999999999999969484, 0.3306041454222136517e-7],
                   [-0.8056217380986972157e-7, -0.3306040883980552500e-7, 0.9999999999999962084]])
    j = State.from_j2000(T0, r, v)
    assert np.allclose(np.asarray(j.position), rb.T @ r, rtol=0, atol=1e-8)
    assert np.linalg.norm(np.asarray(j.position) - r) == pytest.approx(0.58, abs=0.3)


# --- Trajectory from a State ---------------------------------------------------

def test_gcrs_and_itrf_states_fly_the_same_orbit():
    r, vi, ve = _circular(500, 20)
    g = State.from_gcrs(T0, r, vi)
    pi, vi_itrf = g.itrf_state(ballistic.ITRFRealization.ITRF2020)
    a = ballistic.Trajectory(g, _cfg())
    b = ballistic.Trajectory(State.from_itrf(T0, pi, vi_itrf), _cfg())
    pa = np.asarray(a.propagate_for(600.0).position)
    pb = np.asarray(b.propagate_for(600.0).position)
    assert np.array_equal(pa, pb)                     # same ITRF numbers in, bit for bit
    assert np.linalg.norm(np.asarray(a.state_gcrs().position)) == pytest.approx(
        np.linalg.norm(r), rel=1e-3)


@pytest.mark.nrl
def test_realization_comes_from_the_state_and_a_mismatch_raises():
    r, _, ve = _circular(500, 20)
    s = State.from_itrf(T0, r, ve, ballistic.ITRFRealization.ITRF2014)
    assert ballistic.Trajectory(s, None).config().itrf == ballistic.ITRFRealization.ITRF2014
    with pytest.raises(ballistic.ConfigError, match="ITRF2014"):
        ballistic.Trajectory(s, _cfg())               # config says ITRF2020
    cfg = ballistic.recommended_config(s, mass_kg=500.0, srp_area_m2=2.0, drag_area_m2=1.0)
    assert cfg.itrf == ballistic.ITRFRealization.ITRF2014


def test_propagate_for_advances_from_now_and_propagate_to_lands():
    r, _, ve = _circular(500, 20)
    tr = ballistic.Trajectory(State.from_itrf(T0, r, ve), _cfg())
    tr.propagate_for(300.0)
    tr.propagate_for(300.0)
    assert tr.epoch() - T0 == pytest.approx(600.0, abs=1e-9)    # twice = twice
    tr.propagate_to(T0 + 100.0)
    assert tr.epoch() - T0 == pytest.approx(100.0, abs=1e-9)
    with pytest.raises(TypeError):
        tr.propagate_to(100.0)
    res = tr.propagate_until(50.0)                               # seconds from now
    assert res.epoch - T0 == pytest.approx(150.0, abs=1e-9)


def test_batch_members_land_on_their_own_targets():
    r, _, ve = _circular(500, 20)
    r2, _, ve2 = _circular(700, -10)
    t1 = T0 + 1234.5
    batch = ballistic.Trajectory([State.from_itrf(T0, r, ve), State.from_itrf(t1, r2, ve2)], _cfg())
    batch.propagate_for(600.0)
    e = batch.epoch()
    assert e[0] - T0 == pytest.approx(600.0, abs=1e-9)
    assert e[1] - t1 == pytest.approx(600.0, abs=1e-9)          # its own epoch, not member 0's
    batch.propagate_to(T0 + 3000.0)
    assert all(abs(x - (T0 + 3000.0)) < 1e-9 for x in batch.epoch())
    assert all(n > 2 for n in batch.history_size())              # batch interpolation needs history


# --- removed forms, frozen config ---------------------------------------------

def test_removed_forms_name_their_replacement():
    r, _, ve = _circular(500, 20)
    with pytest.raises(ballistic.RemovedApiError, match="State.from_itrf"):
        ballistic.Trajectory(T0, r, ve)
    with pytest.raises(ballistic.RemovedApiError, match="State"):
        ballistic.Trajectory([(T0, r, ve)])
    with pytest.raises(ballistic.RemovedApiError, match="propagate_for"):
        ballistic.Trajectory(State.from_itrf(T0, r, ve), _cfg()).propagate(60.0)
    with pytest.raises(ballistic.RemovedApiError, match="ballistic.State"):
        ballistic.recommended_config(T0, r, ve, "S1A")


def test_config_is_frozen_and_typos_raise():
    cfg = _cfg()
    with pytest.raises(TypeError):
        ballistic.TrajectoryConfig(gravity_degre=20)            # typo
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.gravity_degree = 20
    with pytest.raises(TypeError):
        cfg.replace(gravity_degre=20)
    assert cfg.replace(gravity_degree=20).gravity_degree == 20 and cfg.gravity_degree == 8
    with pytest.raises(TypeError):
        ballistic.TrajectoryConfig(8)                           # keyword-only
    with pytest.raises(TypeError):
        ballistic.SrpConfig()                                   # no generic satellite
    with pytest.raises(ballistic.ConfigError):
        ballistic.SrpConfig(mass_kg=0.0, area_m2=1.0)


# --- SP3 ---------------------------------------------------------------------

def test_state_from_sp3():
    f = ballistic.sp3.read(REPO / "tests/data/sp3/champ/CHAMP_2005_182.sp3")
    arc = f.arcs[0]
    t = arc.epoch0 + 3600.0
    with pytest.raises(ballistic.FrameError, match="realization"):
        State.from_sp3(f, epoch=t)                               # "CTS": which ITRF?
    s = State.from_sp3(f, epoch=t, realization=ballistic.ITRFRealization.ITRF2005)
    p, v = arc.interpolate(3600.0)
    assert np.array_equal(np.asarray(s.position), np.asarray(p))
    assert np.array_equal(np.asarray(s.velocity), np.asarray(v))
    assert s.frame == "ITRF" and s.realization == ballistic.ITRFRealization.ITRF2005
    with pytest.raises(ballistic.DataCoverageError, match="edge"):
        State.from_sp3(f, epoch=arc.epoch0, realization=ballistic.ITRFRealization.ITRF2005)
    g = ballistic.sp3.read(REPO / "tests/data/sp3/goce_pso/goce_pso_13305_0001.sp3.gz")
    with pytest.raises(ballistic.FrameError, match="IGS05"):
        State.from_sp3(g, epoch=g.arcs[0].epoch0 + 600.0, realization=ballistic.ITRFRealization.ITRF2008)
