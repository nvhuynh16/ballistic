"""Live drag model; data loads without internet.

make_drag_environment evaluates NRLMSIS 2.1 / HWM14 at the satellite's
position on the whole loaded weather file: no arc, no duration. Past the last
measured day it uses CelesTrak's forecast with a warning; past the forecast a
Trajectory stops with an error. Tabulated envs warn once when frozen.
ballistic.load() takes local paths, works offline, and falls back to the
cached file when a refresh download fails; data_files() says where they are.
"""

import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

import ballistic

REPO = Path(__file__).resolve().parents[2]
SW = REPO / "tests" / "data" / "weather" / "SW-All.csv"    # measured through 2026-11-03  # guard-exempt(PUB-DATE): data
EOP = REPO / "tests" / "data" / "eop" / "finals2000A.all"

POS = np.array([6778137.0, 0.0, 0.0])
# Earth-fixed: an inertial (0, 1000, 7600) m/s minus omega x r (given as
# Earth-fixed without the subtraction, the start trips the frame check).

VEL = np.array([0.0, 1000.0, 7600.0]) - np.cross(
    [0.0, 0.0, ballistic.constants.earth.ANGULAR_VELOCITY], POS)
_CHAMP = ballistic.satellite_properties.get("CHAMP")
BETA = ballistic.BetaProfile1D.from_mass_cd_area(_CHAMP.mass_kg, _CHAMP.cd, _CHAMP.drag_area_m2)


def _cfg():
    return ballistic.TrajectoryConfig(gravity_degree=8, enable_third_body=False,
                                      enable_relativistic=False, enable_solid_tides=False)


def _traj(t0, hint, **kw):
    cfg = _cfg()
    env, _ = ballistic.make_drag_environment(
        t0, POS, VEL, hint, cfg, BETA, altitude_grid_km=[300, 350, 400, 450, 500],
        max_iter=2, **kw)
    cfg = cfg.replace(drag_env=env)
    return ballistic.Trajectory(ballistic.State.from_itrf(t0, POS, VEL), config=cfg)


@pytest.fixture(autouse=True)
def _committed_weather():
    ballistic._core.load_space_weather(str(SW))
    yield
    ballistic._core.load_space_weather(str(SW))


@pytest.mark.nrl
def test_live_env_has_no_arc():
    t0 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    tr = _traj(t0, 0.0)
    env = tr._traj.config().drag_env
    assert env.is_live() and env.domain() is None
    start, end = env.weather_window()
    assert start.mjd(ballistic.TimeScale.TT) < 36500        # 1957 onward (TT: UTC starts in 1961)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = np.asarray(tr.propagate_to(t0 + 2 * 86400.0).position)   # two days, no rebuild
        rb = np.asarray(tr.propagate_to(t0 - 86400.0).position)   # and back before the start
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(rb))
    assert tr.warnings == []


@pytest.mark.nrl
def test_fixed_env_warns_when_frozen(tmp_path):
    t0 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    cfg = _cfg()
    env, _ = ballistic.make_drag_environment(t0, POS, VEL, 600.0, cfg, BETA,
                                             altitude_grid_km=[300, 400, 500], max_iter=1)
    cfg = cfg.replace(drag_env=ballistic.drag.from_models_along(
        [t0, t0 + 300.0, t0 + 600.0], [POS, POS, POS],
        ballistic.SpaceWeatherHistory.from_loaded(t0, t0 + 600.0), BETA,
        [300.0, 400.0, 500.0])[0])
    assert not cfg.drag_env.is_live()
    tr = ballistic.Trajectory(ballistic.State.from_itrf(t0, POS, VEL), config=cfg)
    with pytest.warns(RuntimeWarning, match="frozen"):
        tr.propagate_for(3600.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # reported once, not per call
        tr.propagate_to(t0 + 5400.0)


@pytest.mark.nrl
def test_save_and_load_a_live_env(tmp_path):
    t0 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    env, _ = ballistic.make_drag_environment(t0, POS, VEL, 0.0, _cfg(), BETA)
    ballistic.drag.save_drag_env(env, str(tmp_path / "live.npz"))
    back = ballistic.drag.load_drag_env(str(tmp_path / "live.npz"))
    assert back.is_live()
    p = POS + np.array([0.0, 1e5, 2e5])
    assert back.evaluate(p, t0).rho_kg_m3 == env.evaluate(p, t0).rho_kg_m3


@pytest.mark.nrl
def test_forecasts_are_used_with_a_warning_unless_refused():
    t0 = ballistic.Epoch.from_utc(2026, 11, 3, 20, 0, 0)
    tr = _traj(t0, 0.0)
    with pytest.warns(RuntimeWarning, match="runs out at 2026-11-04 00:00:00"):  # guard-exempt(PUB-DATE): data
        tr.propagate_for(6 * 3600.0)
    assert tr._traj.config().drag_env.uses_predicted_weather()
    refused = _traj(t0, 0.0, allow_predicted_weather=False)
    with pytest.raises(RuntimeError, match="refused"):
        refused.propagate_for(6 * 3600.0)
    assert abs(refused.epoch() - ballistic.Epoch.from_utc(2026, 11, 4, 0, 0, 0)) < 1e-3
    # The forecast ends 2026-12-18; past it there is nothing to run on.  # guard-exempt(PUB-DATE): data
    # The forecast note is about the loaded data, so the diagnostics channel
    # delivers it once per process; this trajectory still records it.
    t_end = ballistic.Epoch.from_utc(2026, 12, 18, 12, 0, 0)
    tr = _traj(t_end, 0.0)
    assert any("runs out" in w for w in tr.warnings)
    with pytest.raises(RuntimeError, match="including CelesTrak"):
        tr.propagate_for(2 * 86400.0)
    last = ballistic.SpaceWeatherHistory.last_measured_day()
    assert abs(last.mjd(ballistic.TimeScale.UTC)
               - ballistic.Epoch.from_utc(2026, 11, 3, 0, 0, 0).mjd(ballistic.TimeScale.UTC)) < 1e-6


def test_load_from_local_paths_offline():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ballistic.load(eop=EOP, leap_seconds=False, space_weather=SW,
                       ocean_tides=False, offline=True, verbose=False)
    with pytest.warns(RuntimeWarning) as caught:
        ballistic.load(eop=EOP, leap_seconds=False, space_weather="/nonexistent/SW.csv",
                       ocean_tides=False, offline=True, verbose=False)
    msgs = [str(w.message) for w in caught]
    assert any("no such file" in m for m in msgs)
    # the earlier load (session fixture) stays in memory
    assert any("still using the data already loaded" in m for m in msgs)


def test_offline_loader_uses_cache_and_reports_missing(tmp_path):
    shutil.copy(SW, tmp_path / "SW-All.csv")
    loader = ballistic.Loader(directory=str(tmp_path), verbose=False, offline=True)
    assert loader.load_space_weather()
    assert not loader.load_ocean_tides()          # not cached, never downloaded
    assert any("offline" in n for n in loader.notes)


def test_failed_refresh_falls_back_to_cached_file(tmp_path, monkeypatch):
    cached = tmp_path / "SW-All.csv"
    shutil.copy(SW, cached)
    old = time.time() - 10 * 86400                # stale: refresh is due
    os.utime(cached, (old, old))
    loader = ballistic.Loader(directory=str(tmp_path), verbose=False)
    monkeypatch.setattr(loader, "_download", lambda *a, **k: False)
    assert loader.load_space_weather()
    assert any("using the cached file" in n for n in loader.notes)


def _eop_end():
    # conftest pins tests/data/eop/finals2000A.all (predictions ~1 year ahead)
    return ballistic.Epoch.from_mjd(ballistic.eop_coverage()["last_mjd"], ballistic.TimeScale.UTC)


def test_eop_edge_value_is_held_for_30_days_with_a_warning():
    t0 = _eop_end() + 10 * 86400.0
    with pytest.warns(RuntimeWarning, match="Earth orientation"):
        ballistic.Trajectory(ballistic.State.from_itrf(t0, POS, VEL), config=_cfg()).propagate_for(600.0)


def test_trajectory_stops_at_the_eop_hold_limit_and_keeps_its_state():
    t0 = _eop_end() + 29.9 * 86400.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tr = ballistic.Trajectory(ballistic.State.from_itrf(t0, POS, VEL), config=_cfg())
        with pytest.raises(ballistic.DataCoverageError, match="Earth orientation"):
            tr.propagate_for(86400.0)
    reached = tr.epoch() - t0
    assert 0.1 * 86400.0 - 1e-6 <= reached <= 0.1 * 86400.0 + 1e-6   # stopped AT the limit


@pytest.mark.filterwarnings("ignore:leap-second table expires")  # dates past 2027-06-28  # guard-exempt(PUB-DATE): data
def test_far_past_the_eop_table_is_refused():
    t0 = ballistic.Epoch.from_utc(2045, 1, 1, 0, 0, 0)
    with pytest.raises(ballistic.DataCoverageError, match="Earth orientation"):
        ballistic.Trajectory(ballistic.State.from_itrf(t0, POS, VEL), config=_cfg()).propagate_for(600.0)


@pytest.mark.filterwarnings("ignore:leap-second table expires")  # dates past 2027-06-28  # guard-exempt(PUB-DATE): data
def test_get_space_weather_raises_instead_of_substituting():
    with pytest.raises(ballistic.DataCoverageError, match="outside the loaded"):
        ballistic.get_space_weather(ballistic.Epoch.from_utc(2050, 1, 1, 0, 0, 0))
    with pytest.warns(RuntimeWarning, match="forecast"):
        w = ballistic.get_space_weather(ballistic.Epoch.from_utc(2026, 11, 20, 0, 0, 0))
    assert w.f107_daily > 0
    with pytest.raises(ballistic.DataCoverageError, match="forecast"):
        ballistic.get_space_weather(ballistic.Epoch.from_utc(2026, 11, 20, 0, 0, 0),
                                    allow_predicted=False)


def test_failed_reload_keeps_the_data_in_memory(tmp_path):
    ballistic.load(eop=EOP, leap_seconds=False, space_weather=SW, ocean_tides=False,
                   offline=True, verbose=False)
    with pytest.warns(RuntimeWarning) as caught:
        ballistic.load(directory=str(tmp_path), eop=False, leap_seconds=False,
                       space_weather=True, ocean_tides=False, offline=True, verbose=False)
    msgs = [str(w.message) for w in caught]
    assert any("offline and no cached file" in m for m in msgs)
    assert any("still using the data already loaded from" in m and "SW-All.csv" in m
               for m in msgs)
    assert ballistic._core.space_weather_loaded()


def test_data_files_reports_what_is_present(tmp_path):
    shutil.copy(SW, tmp_path / "SW-All.csv")
    shutil.copy(EOP, tmp_path / "finals2000A.all")
    found = ballistic.data_files(str(tmp_path))
    assert found["space_weather"] == tmp_path / "SW-All.csv"
    assert found["eop"] == tmp_path / "finals2000A.all"      # alternative name
    assert found["leap_seconds"] is None and found["ocean_tides"] is None


@pytest.mark.nrl
def test_a_start_past_the_weather_says_so():
    # The committed file has daily forecasts to 2026-12-18. Formatting the  # guard-exempt(PUB-DATE): data
    # weather window (which starts in 1957) in UTC used to throw "TAI-UTC is
    # undefined before 1961" instead of this message.
    t0 = ballistic.Epoch.from_utc(2027, 3, 1, 0, 0, 0)
    with pytest.raises(ballistic.DataCoverageError, match=r"has no space weather: the loaded weather covers 1957"):
        _traj(t0, 0.0)
