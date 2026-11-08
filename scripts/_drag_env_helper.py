"""Shared helpers for validation scripts that model atmospheric drag.

Drag is switched on through one TrajectoryConfig field, `drag_env`, which
holds a DragEnvironment (None = drag off). A DragEnvironment answers three
questions at any position and time: the air density, the wind, and the
satellite's ballistic coefficient β = m / (Cd·A) in kg/m² (mass over drag
coefficient times cross-section area; a larger β means drag slows the
satellite less). The density comes from NRLMSIS 2.1 (the atmosphere density
model) and the wind from HWM14 (the wind model); both need "space weather":
F10.7 (the daily 10.7 cm solar radio flux) and Ap (the geomagnetic activity
index), read from a CelesTrak CSV file.

This module turns "an arc's start state and a satellite's β" into "a config
with a DragEnvironment attached", using ballistic.make_drag_environment, and
holds the weather options shared by the validation ("campaign") scripts:
--drag-weather and --weather (see add_weather_arg).

Typical use from a validate_*_drag.py script:

    from _drag_env_helper import (
        build_drag_env_for_arc,
        get_weather_for_epoch,
        make_arc_config,
    )

    config = make_arc_config(
        ballistic, base_config, drag_on=True,
        start_epoch=t0, pos_ecef=p, vel_ecef=v, duration_s=3600,
        beta_kg_m2=197.9,  # CHAMP: ballistic.satellite_properties (fitted Cd*A)
        altitude_grid_km=[200, 225, ..., 500],
        weather=get_weather_for_epoch(ballistic, t0))
    traj = ballistic.Trajectory(ballistic.State.from_itrf(t0, p, v), config=config)
    traj.propagate_for(3600.0)
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


# --- Reproducible space weather --------------------------------------------
# A validation script's reproduction command must give the SAME numbers on
# every run, not "whatever CelesTrak serves today". The committed
# tests/data/weather/SW-All.csv spans 1957-2041 (checked) and covers every
# validation epoch, so it is the default.
COMMITTED_WEATHER_CSV = str(
    (Path(__file__).resolve().parents[1] / "tests" / "data" / "weather"
     / "SW-All.csv"))


def load_weather_source(ballistic, source: str) -> str:
    """Load space weather into the library from `source`.

    `source` is 'live' (download the current CelesTrak file) or the path of a
    CelesTrak CSV. Returns a short description for logging. A failed load
    raises (RuntimeError for 'live', FileNotFoundError for a path); nothing
    falls back to quiet solar conditions, because a quiet-weather fallback can
    hide a several-fold error in β.
    """
    if source == "live":
        if not ballistic.Loader().space_weather():
            raise RuntimeError("live space-weather download/load failed")
        return "live CelesTrak download (NOT reproducible)"
    ok = ballistic._core.load_space_weather(source)
    if not ok:
        raise FileNotFoundError(f"space-weather file failed to load: {source}")
    from pathlib import Path
    same = Path(source).resolve() == Path(COMMITTED_WEATHER_CSV).resolve()
    return f"{'committed file' if same else 'file'} {source}"


# Weather for drag, set by --drag-weather: 'history' (the default and what
# library users get: the loaded file's measured values varying over time,
# with the previous day's F10.7 and the 3-hourly Ap) or 'snapshot' (one
# get_space_weather() value at each arc's start, held for the whole arc;
# reproduces numbers measured that way).
DRAG_WEATHER = "history"


class _DragWeatherAction(argparse.Action):
    """argparse action that also stores --drag-weather in DRAG_WEATHER."""

    def __call__(self, parser, namespace, values, option_string=None):
        global DRAG_WEATHER
        DRAG_WEATHER = values
        setattr(namespace, self.dest, values)


def add_weather_arg(parser):
    """Add --drag-weather {history,snapshot} and --weather PATH|live to a parser.

    The caller passes the parsed --weather value to load_weather_source.
    """
    parser.add_argument(
        "--drag-weather", choices=("history", "snapshot"), default="history",
        action=_DragWeatherAction,
        help="drag weather: the library default time-varying history "
             "(default) or one value per arc held for the whole arc "
             "(snapshot)")
    parser.add_argument(
        "--weather", default=COMMITTED_WEATHER_CSV,
        help="CelesTrak CSV path (default: the committed hermetic file), or "
             "'live' to download current data (not reproducible)")


def get_weather_for_epoch(ballistic, epoch):
    """Return the loaded space weather (F10.7, Ap) at `epoch` as WeatherConditions.

    Raises when no space weather is loaded or the epoch is outside it;
    there is no fallback to quiet conditions (they give a density several
    times lower than active conditions, enough to mask a several-fold error
    in β). Callers load the weather once before the validation loop.
    """
    return ballistic.get_space_weather(epoch)


def build_drag_env_for_arc(
    ballistic,
    start_epoch,
    pos_ecef,
    vel_ecef,
    weather,
    beta_kg_m2: float,
    duration_s: float,
    altitude_grid_km: Sequence[float],
    sample_dt_s: float = 60.0,
):
    """Build the DragEnvironment for one arc with ballistic.make_drag_environment.

    The environment evaluates NRLMSIS 2.1 and HWM14 at the satellite's exact
    position on every force evaluation, so it serves any arc;
    make_drag_environment only checks that `start_epoch` has space weather.

    Args:
        ballistic: the imported `ballistic` module.
        start_epoch: ballistic.Epoch at arc start.
        pos_ecef: (3,) Earth-fixed (ECEF) position at start (m).
        vel_ecef: (3,) Earth-fixed velocity at start (m/s).
        weather: WeatherConditions for the arc, held constant over it; used
            only with --drag-weather snapshot.
        beta_kg_m2: ballistic coefficient β = m / (Cd·A), kg/m².
        duration_s: arc duration (seconds).
        altitude_grid_km: altitude knots (km). Passed through, but the live
            environment does not use them (it needs no table).
        sample_dt_s: sampling step (s). Passed through; also unused by the
            live environment (as are pos_ecef, vel_ecef and duration_s:
            make_drag_environment accepts them for compatibility).

    Returns:
        ballistic.drag.DragEnvironment ready to attach to TrajectoryConfig.drag_env.
        Raises (DataCoverageError) if `start_epoch` has no space weather.
    """
    # The library's one-call helper builds the whole environment.
    env, _diag = ballistic.make_drag_environment(
        start_epoch, np.asarray(pos_ecef, float), np.asarray(vel_ecef, float),
        duration_s,
        ballistic.TrajectoryConfig(gravity_degree=70),
        beta=ballistic.BetaProfile1D.constant(beta_kg_m2),
        weather=(None if DRAG_WEATHER == "history"
                 else ballistic.SpaceWeatherHistory.from_constant(weather)),
        altitude_grid_km=list(altitude_grid_km),
        sample_dt_s=sample_dt_s,
    )
    return env


def make_arc_config(
    ballistic,
    base_config,
    drag_on: bool,
    start_epoch=None,
    pos_ecef=None,
    vel_ecef=None,
    duration_s: float = 0.0,
    beta_kg_m2: float = 0.0,
    altitude_grid_km: Optional[Sequence[float]] = None,
    weather=None,
):
    """Return a TrajectoryConfig with drag_env attached when drag_on=True.

    For drag-on arcs, builds a fresh DragEnvironment via
    build_drag_env_for_arc and returns a dataclasses.replace copy of
    `base_config` with `drag_env` set; `weather` defaults to the loaded value
    at `start_epoch`. For drag-off arcs, returns `base_config` unchanged
    (no copy). The other arguments are those of build_drag_env_for_arc.

    Raises ValueError if drag_on and altitude_grid_km is None.
    """
    if not drag_on:
        return base_config
    if altitude_grid_km is None:
        raise ValueError(
            "make_arc_config(drag_on=True): altitude_grid_km is required")
    if weather is None:
        weather = get_weather_for_epoch(ballistic, start_epoch)
    env = build_drag_env_for_arc(
        ballistic, start_epoch, pos_ecef, vel_ecef,
        weather, beta_kg_m2, duration_s, altitude_grid_km)
    return dataclasses.replace(base_config, drag_env=env)


# Altitude grids (km) matched to each satellite's altitude range, kept for the
# altitude_grid_km argument. They limited the number of NRLMSIS 2.1
# evaluations when the drag environment was a table; the live environment
# ignores them.
ALT_GRID_CHAMP_KM = [200 + 25 * i for i in range(13)]      # 200-500 km
ALT_GRID_GRACEFO_KM = [400 + 25 * i for i in range(7)]     # 400-550 km
ALT_GRID_SWARM_KM = [350 + 25 * i for i in range(8)]       # 350-525 km
ALT_GRID_SENTINEL_KM = [650 + 10 * i for i in range(11)]   # 650-750 km
ALT_GRID_GOCE_REENTRY_KM = [100 + 20 * i for i in range(11)]  # 100-300 km
