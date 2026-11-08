"""Two-line element sets (TLE) and SGP4.

A TLE is the compact public orbit format (two 69-character text lines, as
published by CelesTrak). SGP4 is the standard analytical model that turns a
TLE into a position and velocity at any time. A TLE's numbers are MEAN
elements defined inside SGP4's own theory (averaged in SGP4's particular
way), not the instantaneous ("osculating") orbit, so only SGP4 should read
them.

To start a numerical propagation from a TLE use ``ballistic.State.from_tle``,
which evaluates SGP4 at the element-set epoch and converts its output from
TEME (the "true equator, mean equinox" frame SGP4 works in) to the
Earth-fixed ITRF. Expect ~0.5 km of error at the epoch in low Earth orbit
(LEO; the Flohrer et al. 2009 TLE error budget), growing by a few km per day;
``ballistic.recommended_config(state, accuracy="tle", horizon_s=...)`` picks
the forces that still matter at that level.

    tle = ballistic.tle.parse(line1, line2)          # strict: columns, checksums
    model = ballistic.tle.Sgp4(tle)                  # WGS-72, "improved" mode
    r_km, v_km_s, status = model.propagate_minutes(90.0)   # TEME
    for name, tle in ballistic.tle.read("gp.txt"):   # CelesTrak 2LE / 3LE text
        ...

The SGP4 here is a clean-room implementation of Spacetrack Report #3 with the
corrections of Vallado et al. (2006); it reproduces the SGP4-VER verification
vectors (tests/python/test_sgp4.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _core

Tle = _core.Tle
Sgp4 = _core.Sgp4
tle_checksum = _core.tle_checksum
# The TLE-grade force preset: the forces that still matter at TLE accuracy
# (also reachable as ``recommended_config(state, accuracy="tle", ...)``).
TleGradePreset = _core.TleGradePreset
tle_grade_preset = _core.tle_grade_preset

__all__ = ["Tle", "Sgp4", "parse", "read", "tle_checksum", "teme_to_itrf", "itrf_to_teme",
           "fit_state", "TleGradePreset", "tle_grade_preset"]


def parse(line1: str, line2: str) -> Tle:
    """Parse one element set.

    Args:
        line1, line2: the two 69-column lines (str).

    Returns:
        A ``Tle``.

    Raises:
        InputPlausibilityError: wrong length, a bad checksum or an
            unreadable field (the message names the line and column).
    """
    return Tle.parse(line1, line2)


def read(source) -> list[tuple[str | None, Tle]]:
    """Read every element set in a text file or string.

    Accepts 2LE text (just the two lines per object) and 3LE text (a name
    line before each pair, CelesTrak's ``FORMAT=TLE``).

    Args:
        source: a path (str or Path) to a file, or the text itself (a str
            containing newlines).

    Returns:
        ``[(name, tle), ...]``; ``name`` is the name line of a 3LE set, or
        ``None`` for 2LE.

    Raises:
        InputPlausibilityError: a line that is not part of a set, or a
            malformed set. Blank lines are ignored.
    """
    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()) else str(source)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[tuple[str | None, Tle]] = []
    i = 0
    while i < len(lines):
        name = None
        if not lines[i].startswith("1 "):
            name = lines[i][2:].strip() if lines[i].startswith("0 ") else lines[i].strip()
            i += 1
        if i + 1 >= len(lines) or not lines[i].startswith("1 ") or not lines[i + 1].startswith("2 "):
            raise _core.InputPlausibilityError(
                f"TLE text: expected a line-1/line-2 pair after line {i} ({lines[i - 1]!r})")
        out.append((name, Tle.parse(lines[i], lines[i + 1])))
        i += 2
    return out


def teme_to_itrf(epoch, r_m, v_m_s):
    """Convert a state from TEME (SGP4's output frame) to the Earth-fixed ITRF.

    Uses the 1982 Greenwich mean sidereal time model (the Earth's rotation
    angle SGP4 assumes) and polar motion.

    Args:
        epoch: the Epoch of the state.
        r_m: position, meters, any 3-sequence.
        v_m_s: velocity, meters/second, any 3-sequence.

    Returns:
        ``(r_itrf_m, v_itrf_m_s)``; the velocity is relative to the rotating
        Earth.
    """
    return _core.teme_to_itrf(epoch, np.asarray(r_m, dtype=np.float64),
                              np.asarray(v_m_s, dtype=np.float64))


def itrf_to_teme(epoch, r_m, v_m_s):
    """Convert a state from the Earth-fixed ITRF to TEME (inverse of teme_to_itrf).

    Args and units as ``teme_to_itrf``; returns ``(r_teme_m, v_teme_m_s)``.
    """
    return _core.itrf_to_teme(epoch, np.asarray(r_m, dtype=np.float64),
                              np.asarray(v_m_s, dtype=np.float64))


def fit_state(tle, config=None, *, span_s=86400.0, step_s=300.0, max_iterations=6):
    """A start state fitted so the numerical orbit follows SGP4 for ``span_s``.

    ``State.from_tle`` takes SGP4's state AT the epoch, which carries
    SGP4's simplifications (its velocity is not even the exact derivative of
    its position: 1-5 cm/s, enough to drift kilometers in a day). This
    instead finds the inertial (GCRS) state at the epoch whose numerical
    trajectory under ``config`` best matches SGP4's positions every
    ``step_s`` seconds over ``span_s`` seconds after the epoch. The fit is
    a Gauss-Newton least-squares iteration with finite-difference
    derivatives (7 propagations per iteration).

    Args:
        tle: a parsed ``Tle``.
        config: a ``TrajectoryConfig`` (None = the defaults).
        span_s: how long after the epoch to match, seconds (default one day).
        step_s: spacing of the matched SGP4 positions, seconds (default 300).
        max_iterations: Gauss-Newton iterations at most (default 6).

    Why one day (measured, GRACE-FO 1 and 2 at 490 km, docs/validation_claims.md
    TLE-GRACEFO): SGP4's epoch state has a semi-major-axis error of tens of
    meters, so a numerical run from it drifts kilometers per day (12-24 km at
    3.6-7 days, against 1-2.9 km for SGP4 itself). Fitting fixes the mean motion
    only over a long enough span: two orbits gave 31-57 km, half a day 10-14
    km, one day 0.3-1.8 km, two days 2.4-6.4 km (the force model's drag and
    SGP4's B* drag differ, and a longer fit absorbs the difference into the
    orbit). One object class, horizons BEFORE the TLE epoch only; see the
    manifest for what has and has not been measured.

    Returns:
        ``(state, rms_m)``: the fitted ``ballistic.State`` (GCRS) and the
        root-mean-square 3-D distance between the fitted orbit and SGP4's
        positions, meters.

    Raises:
        TypeError: ``tle`` is not a parsed Tle.
    """
    from .state import State
    from .trajectory import Trajectory, TrajectoryConfig

    if not isinstance(tle, Tle):
        raise TypeError("fit_state: pass a parsed ballistic.tle.Tle")
    config = TrajectoryConfig() if config is None else config
    model = Sgp4(tle)
    t0 = tle.epoch
    times = np.arange(step_s, span_s + 1e-9, step_s)
    truth = []
    for dt in times:
        r_teme, v_teme = model.teme(t0 + float(dt))
        truth.append(np.asarray(teme_to_itrf(t0 + float(dt), r_teme, v_teme)[0]))
    truth = np.array(truth).ravel()

    def positions(x):
        tr = Trajectory(State.from_gcrs(t0, x[:3], x[3:]), config)
        tr.propagate_to(t0 + float(times[-1]))
        return np.array([np.asarray(s.position) for s in tr(list(times), frame="ECEF",
                                                             on_out_of_range="raise")]).ravel()

    r0, v0 = State.from_tle(tle).gcrs_state()
    x = np.concatenate([np.asarray(r0), np.asarray(v0)])
    steps = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])
    rms = float("inf")
    for _ in range(max_iterations):
        p0 = positions(x)
        res = p0 - truth
        rms = float(np.sqrt(np.mean(res.reshape(-1, 3) ** 2) * 3))
        jac = np.empty((res.size, 6))
        for k in range(6):
            xk = x.copy()
            xk[k] += steps[k]
            jac[:, k] = (positions(xk) - p0) / steps[k]
        dx = np.linalg.lstsq(jac, -res, rcond=None)[0]
        x = x + dx
        if np.linalg.norm(dx[:3]) < 1e-3 and np.linalg.norm(dx[3:]) < 1e-6:
            break
    res = positions(x) - truth
    rms = float(np.sqrt(np.mean(res.reshape(-1, 3) ** 2) * 3))
    return State.from_gcrs(t0, x[:3], x[3:]), rms
