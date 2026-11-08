"""Flow-regime drag coefficient from Python (C++ cells: tests/unit/test_flow_regime_cd.cpp).

A reentering body's Cd follows the flow: free-molecular in orbit (~2.1 for a
sphere), continuum near the ground (~0.9 hypersonic). AeroBeta + FlowRegimeCd
carry it; recommended_config(reentry=...) wires it up.
"""

import math

import numpy as np
import pytest

import ballistic

T0 = ballistic.Epoch.from_utc(2024, 1, 10, 0, 0, 0)


def _cfg():
    return ballistic.TrajectoryConfig(gravity_degree=8, enable_third_body=False,
                                      enable_relativistic=False, enable_solid_tides=False,
                                      ocean_tide_degree=0)


def _sphere(mass=500.0, d=1.0):
    return ballistic.aero.AeroBeta(mass, math.pi * d * d / 4, ballistic.aero.FlowRegimeCd(ballistic.aero.AeroShape.SPHERE),
                              reference_length_m=d)


def _east(h_m, speed):
    p = np.asarray(ballistic.geodetic_to_ecef(0.3, 0.2, h_m))
    east = np.array([-p[1], p[0], 0.0])
    return p, speed * east / np.linalg.norm(east)


@pytest.mark.nrl
def test_cd_follows_the_regime():
    env, _ = ballistic.make_drag_environment(T0, np.array([7e6, 0, 0]), np.zeros(3), 0.0, _cfg(), _sphere())
    assert env.beta_depends_on_velocity()
    orbit = env.evaluate(*_east(300e3, 7700.0)[:1], T0, _east(300e3, 7700.0)[1])
    low = env.evaluate(*_east(60e3, 4000.0)[:1], T0, _east(60e3, 4000.0)[1])
    assert orbit.knudsen > 10 and 2.0 < orbit.cd < 2.2
    assert low.knudsen < 1e-3 and low.cd == pytest.approx(0.9 + 0.34 / low.mach**2, rel=1e-9)
    assert low.molar_mass_kg_kmol == pytest.approx(28.96, abs=0.05)
    assert low.beta_kg_m2 == pytest.approx(500.0 / (low.cd * math.pi / 4), rel=1e-12)


@pytest.mark.nrl
def test_velocity_is_required_and_bad_bodies_raise():
    env, _ = ballistic.make_drag_environment(T0, np.array([7e6, 0, 0]), np.zeros(3), 0.0, _cfg(), _sphere())
    with pytest.raises(ballistic.errors.ConfigError):
        env.evaluate(_east(300e3, 0)[0], T0)
    with pytest.raises(ballistic.errors.ConfigError):
        ballistic.aero.AeroBeta(500.0, 1.0, ballistic.aero.FlowRegimeCd(ballistic.aero.AeroShape.SPHERE))   # no length
    with pytest.raises(ballistic.errors.ConfigError):
        ballistic.aero.MachTable([2.0, 1.0], [1.0, 1.0])


def test_free_molecular_limit_and_henderson():
    # Hypervelocity, no re-emission: Cd -> 2 for every shape.
    for shape in (ballistic.aero.AeroShape.SPHERE, ballistic.aero.AeroShape.FLAT_PLATE, ballistic.aero.AeroShape.CYLINDER):
        assert ballistic.aero.free_molecular_cd(shape, 1e3, 0.0) == pytest.approx(2.0, abs=1e-5)
    # Henderson's continuum sphere at high Re: 0.9 + 0.34 / M^2.
    assert ballistic.aero.henderson1976_sphere_cd(10.0, 1e14, 1.0) == pytest.approx(0.9034, abs=1e-4)


@pytest.mark.nrl
def test_recommended_config_reentry_lands():
    p = np.asarray(ballistic.geodetic_to_ecef(0.4, 0.2, 120e3))
    up = p / np.linalg.norm(p)
    east = np.array([-p[1], p[0], 0.0]) / np.hypot(p[0], p[1])
    g = math.radians(-2.0)
    w = np.array([0.0, 0.0, ballistic.constants.earth.ANGULAR_VELOCITY])
    v = 7600.0 * (math.cos(g) * east + math.sin(g) * up) - np.cross(w, p)
    cfg = ballistic.recommended_config(ballistic.State.from_itrf(T0, p, v, allow_suborbital=True), mass_kg=500.0, drag_area_m2=math.pi / 4,
                                       reentry=ballistic.aero.AeroShape.SPHERE, reference_length_m=1.0,
                                       gravity_degree=8, enable_third_body=False,
                                       enable_relativistic=False, enable_solid_tides=False,
                                       ocean_tide_degree=0)
    assert cfg.srp is None
    traj = ballistic.Trajectory(ballistic.State.from_itrf(T0, p, v, allow_suborbital=True), config=cfg)
    res = traj.propagate_until(3600.0)
    assert res.stopped_by.name == "ground impact"
    # C++ measured 395.6 s for the same body (test_flow_regime_cd); same physics here.
    assert res.epoch - T0 == pytest.approx(395.6, abs=1.0)
    with pytest.raises(ValueError):
        ballistic.recommended_config(ballistic.State.from_itrf(T0, p, v, allow_suborbital=True), mass_kg=500.0, drag_area_m2=1.0,
                                     reentry=ballistic.aero.AeroShape.SPHERE)   # no reference length
