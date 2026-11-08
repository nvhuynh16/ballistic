"""ballistic.recommended_config — the one-call best-accuracy setup.

Unit cells check what it turns on; the end-to-end cell checks that it beats
the plain defaults on real Sentinel-1A POE data from January 2026 — outside
the 2025 year the radiation areas were fitted on (SENTINEL-YEAR-2025).
"""

import glob
import os

import ballistic
import numpy as np
import pytest
from ballistic import satellite_properties as sp

JAN2026 = os.path.join(os.path.dirname(__file__), "..", "data", "eof",
                       "sentinel_jan2026")


@pytest.mark.nrl
def test_named_satellite_turns_on_srp_and_drag(leo_state):
    epoch, pos, vel = leo_state
    cfg = ballistic.recommended_config(ballistic.State.from_itrf(epoch, pos, vel), "S1A")
    p = sp.get("S1A")
    assert cfg.srp.area_m2 == pytest.approx(p.srp_area_m2)
    assert cfg.srp.mass_kg == pytest.approx(p.mass_kg)
    assert cfg.srp.cr == pytest.approx(1.0 + p.srp_reflectivity)
    assert cfg.drag_env is not None
    # The general settings stay the measured-best library defaults.
    assert cfg.gravity_degree == ballistic.TrajectoryConfig().gravity_degree


@pytest.mark.nrl
def test_own_satellite_and_overrides(leo_state):
    epoch, pos, vel = leo_state
    cfg = ballistic.recommended_config(ballistic.State.from_itrf(epoch, pos, vel), mass_kg=500.0, srp_area_m2=3.0, drag_area_m2=1.2,
        itrf=ballistic._core.ITRFRealization.ITRF2014)
    assert cfg.srp.cr == pytest.approx(1.3)
    assert cfg.drag_env is not None
    assert cfg.itrf == ballistic._core.ITRFRealization.ITRF2014
    with pytest.raises(ValueError, match="drag_area_m2"):
        ballistic.recommended_config(ballistic.State.from_itrf(epoch, pos, vel), mass_kg=500.0,
                                     srp_area_m2=3.0)
    with pytest.raises(ValueError, match="name a satellite"):
        ballistic.recommended_config(ballistic.State.from_itrf(epoch, pos, vel))


def test_no_drag_above_the_atmosphere_grid():
    epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    r = ballistic.constants.earth.RADIUS_EQUATORIAL + 2.0e6   # 2000 km
    state = ballistic.State.from_gcrs(epoch, [r, 0.0, 0.0], [0.0, 6900.0, 0.0])   # circular
    cfg = ballistic.recommended_config(state,
        mass_kg=1000.0, srp_area_m2=10.0)
    assert cfg.srp is not None and cfg.drag_env is None


def _s1a_starts():
    files = sorted(glob.glob(os.path.join(JAN2026, "S1A_*POEORB*.EOF")))
    if not files:
        pytest.skip(f"missing test data S1A_*POEORB*.EOF in {JAN2026} — reacquire: "
                    "Sentinel-1A POEORB files for January 2026 from the Copernicus "
                    "Data Space Ecosystem (dataspace.copernicus.eu)")
    starts = []
    for f in files[:6]:
        sv = ballistic.readers.parse_eof_auto(f).state_vectors
        for i in (3000, 6000):          # 10 s spacing: 360 steps = 1 h
            starts.append((sv[i], sv[i + 360]))
    return starts


@pytest.mark.nrl
def test_recommended_beats_defaults_out_of_sample():
    starts = _s1a_starts()
    errs = {"defaults": [], "recommended": []}
    for s0, s1 in starts:
        for name, out in errs.items():
            cfg = (ballistic.TrajectoryConfig() if name == "defaults" else
                   ballistic.recommended_config(ballistic.State.from_itrf(s0.epoch, s0.position, s0.velocity), "S1A"))
            tr = ballistic.Trajectory(ballistic.State.from_itrf(s0.epoch, s0.position, s0.velocity),
                                      config=cfg)
            tr.propagate_for(3600.0)
            p = np.asarray(tr(3600.0).position)
            out.append(np.linalg.norm(p - np.asarray(s1.position)))
    rmse = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
    assert rmse["recommended"] < rmse["defaults"], rmse
    assert rmse["recommended"] < 1.0, rmse


@pytest.mark.parametrize("satellite", [None, "GOCE"])
def test_reentry_refuses_an_explicit_cd(satellite):
    # reentry= computes Cd from the flow; a caller's cd would be ignored, so
    # it is refused, with or without a named satellite (whose registry cd is
    # not the caller's and must not trigger the refusal on its own).
    state = ballistic.State.from_itrf(
        ballistic.Epoch.from_utc(2026, 1, 5, 0, 0, 0),
        np.asarray(ballistic.geodetic_to_ecef(0.4, 0.2, 120e3)), [0.0, 7000.0, 0.0],
        allow_suborbital=True)
    kw = {} if satellite else {"mass_kg": 500.0, "drag_area_m2": 1.0}
    with pytest.raises(ValueError, match="not both"):
        ballistic.recommended_config(state, satellite, cd=2.0,
                                     reentry=ballistic.aero.AeroShape.SPHERE,
                                     reference_length_m=1.0, **kw)
