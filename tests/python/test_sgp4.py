"""SGP4 verification (C++ cells: tests/unit/test_sgp4.cpp).

The reference is the SGP4-VER verification set (Vallado et al. 2006, AIAA
2006-6753): SGP4-VER.TLE (33 test objects with start/stop/step) and
tcppver.out (the published output). Both are carried by the MIT-licensed
``sgp4`` package, which is installed at TEST time only and also serves as an
oracle for the other gravity-constant sets and the AFSPC mode. Neither its
code nor its data ship with ballistic: the implementation is clean-room
(include/ballistic/sgp4/sgp4.hpp).
"""

from pathlib import Path

import numpy as np
import pytest
import sgp4 as _oracle_pkg
from sgp4.api import WGS72, WGS72OLD, WGS84, Satrec
from sgp4.io import twoline2rv as _py_twoline2rv
from sgp4.propagation import sgp4 as _py_sgp4
from sgp4 import earth_gravity as _eg

import ballistic
from ballistic import State, tle as T

_DIR = Path(_oracle_pkg.__file__).parent
# The three error-code cases were edited by hand without updating the checksum.
_HAND_EDITED = {"33333", "33334", "33335"}


def _fix_checksum(line):
    return line[:68] + str(T.tle_checksum(line))


def _vectors():
    rows = [ln.rstrip("\n") for ln in (_DIR / "SGP4-VER.TLE").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    out = []
    for k in range(0, len(rows), 2):
        l1, l2full = rows[k], rows[k + 1]
        start, stop, step = map(float, l2full[69:].split()[:3])
        out.append((l1, l2full[:69], start, stop, step))
    return out


def _published():
    """tcppver.out: {catalog: [(t_min, r_km, v_km_s), ...]} in file order (blocks may repeat)."""
    blocks = []
    for ln in (_DIR / "tcppver.out").read_text(encoding="utf-8").splitlines():
        f = ln.split()
        if len(f) == 2 and f[1] == "xx":
            blocks.append((f[0].zfill(5), []))
        elif f:
            v = list(map(float, f[:7]))
            blocks[-1][1].append((v[0], np.array(v[1:4]), np.array(v[4:7])))
    return blocks


def _times(start, stop, step):
    t = list(np.arange(start, stop, step))
    if not t or abs(t[-1] - stop) > 1e-9:
        t.append(stop)
    return t


def _parse(l1, l2):
    if l1[2:7] in _HAND_EDITED:
        with pytest.raises(ballistic.InputPlausibilityError, match="checksum"):
            T.parse(l1, l2)
        l1, l2 = _fix_checksum(l1), _fix_checksum(l2)
    return T.parse(l1, l2)


def test_reproduces_the_published_verification_output():
    vectors = _vectors()
    published = _published()
    assert len(vectors) == len(published) == 33
    worst_r = worst_v = 0.0
    previous_last = None
    for (l1, l2, *_), (cat, rows) in zip(vectors, published):
        assert l1[2:7] == cat
        m = T.Sgp4(_parse(l1, l2))
        for t, r_ref, v_ref in rows:
            r, v, status = m.propagate_minutes(t)
            if status not in (0, 6):
                # 33334 fails at epoch (error 3); the reference driver then
                # printed the PREVIOUS object's last state again.
                assert cat == "33334" and np.array_equal(r_ref, previous_last[1])
                break
            worst_r = max(worst_r, np.abs(np.asarray(r) - r_ref).max())
            worst_v = max(worst_v, np.abs(np.asarray(v) - v_ref).max())
        previous_last = rows[-1]
    # tcppver.out prints 8 decimals of km (0.01 mm) and 9 of km/s.
    assert worst_r < 1e-6, f"worst position difference {worst_r * 1e6:.4f} mm"
    assert worst_v < 1e-8, f"worst velocity difference {worst_v * 1e6:.4f} mm/s"


@pytest.mark.parametrize("gravity", ["wgs72", "wgs72old", "wgs84"])
def test_matches_the_oracle_at_every_step_with_its_error_codes(gravity):
    const = {"wgs72": WGS72, "wgs72old": WGS72OLD, "wgs84": WGS84}[gravity]
    worst = 0.0
    for l1, l2, start, stop, step in _vectors():
        tle = _parse(l1, l2)
        sat = Satrec.twoline2rv(tle.line1, tle.line2, const)
        m = T.Sgp4(tle, gravity=gravity)
        for t in [0.0] + _times(start, stop, step):
            e_ref, r_ref, v_ref = sat.sgp4_tsince(t)
            r, v, e = m.propagate_minutes(t)
            assert e == e_ref, (tle.catalog_number, t, e, e_ref)
            if e in (0, 6):
                worst = max(worst, np.abs(np.subtract(r, r_ref)).max())
                assert np.abs(np.subtract(v, v_ref)).max() < 1e-9
            if e_ref not in (0, 6):
                break
    assert worst < 1e-6, f"{worst * 1e6:.4f} mm"


def test_afspc_mode_matches_the_oracle():
    worst, compared = 0.0, 0
    for l1, l2, start, stop, step in _vectors():
        tle = _parse(l1, l2)
        sat = _py_twoline2rv(tle.line1, tle.line2, _eg.wgs72, opsmode="a")
        m = T.Sgp4(tle, mode="afspc")
        for t in _times(start, stop, step)[:6]:
            r_ref, v_ref = _py_sgp4(sat, t)
            r, v, e = m.propagate_minutes(t)
            if sat.error or e not in (0, 6):
                assert (e != 0) == bool(sat.error), (tle.catalog_number, t)
                break
            worst = max(worst, np.abs(np.subtract(r, r_ref)).max())
            compared += 1
    assert compared > 150 and worst < 1e-6, (compared, f"{worst * 1e6:.4f} mm")


def test_tle_text_reader_and_state():
    l1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    l2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    sets = T.read(f"ISS (ZARYA)\n{l1}\n{l2}\n\n{l1}\n{l2}\n")
    assert [n for n, _ in sets] == ["ISS (ZARYA)", None]
    assert sets[0][1].catalog_number == 25544
    with pytest.raises(ballistic.InputPlausibilityError):
        T.read(f"{l1}\n")
    s = State.from_tle(l1, l2)
    assert s.frame == "ITRF" and s.realization is None
    t = sets[0][1]
    assert s.epoch - t.epoch == 0.0
    # At the epoch the state is SGP4's TEME output rotated to the ITRF.
    r_teme, v_teme = T.Sgp4(t).teme(t.epoch)
    r, v = T.teme_to_itrf(t.epoch, r_teme, v_teme)
    assert np.array_equal(np.asarray(s.position), np.asarray(r))
    later = State.from_tle(t, epoch=t.epoch + 600.0)
    assert later.epoch - t.epoch == pytest.approx(600.0, abs=1e-9)
    tr = State.from_teme(t.epoch, r_teme, v_teme)
    assert np.allclose(np.asarray(tr.position), np.asarray(r), atol=1e-9)
    # B* -> beta (m / Cd A).
    assert t.beta_from_bstar_kg_m2() == pytest.approx(1.0 / (12.741621 * -0.11606e-4))


def test_tle_elements_are_not_osculating():
    # The same numbers read as osculating Keplerian elements (the trap) land
    # tens of km from SGP4's state: the elements are means in SGP4's theory,
    # in TEME, with a Kozai mean motion.
    l1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    l2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    t = T.parse(l1, l2)
    n = t.mean_motion_rev_day * 2 * np.pi / 86400.0
    a = (ballistic.constants.earth.GM / n**2) ** (1.0 / 3.0)
    e = t.eccentricity
    m = np.radians(t.mean_anomaly_deg)
    E = m
    for _ in range(20):
        E = E - (E - e * np.sin(E) - m) / (1 - e * np.cos(E))
    nu = np.degrees(2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2)))
    naive = State.from_keplerian(t.epoch, a_m=a, e=e, i_deg=t.inclination_deg, raan_deg=t.raan_deg,
                                 argp_deg=t.argp_deg, true_anomaly_deg=nu % 360.0)
    right = State.from_tle(t)
    gap = np.linalg.norm(np.asarray(naive.gcrs_state()[0]) - np.asarray(right.gcrs_state()[0]))
    print(f"naive Keplerian vs SGP4: {gap / 1e3:.1f} km")
    assert gap > 10e3, gap


# --- the TLE-grade preset (presets/tle_grade_table.hpp) ----------------------

ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


def test_preset_lookup_and_refusals():
    p = ballistic._core._tle_grade_preset_for(410.0, 430.0, 3000.0)
    assert (p.regime, p.horizon_s, p.rows_altitude_km_min) == ("LEO", 3600.0, 400.0)
    assert p.drag and p.gravity_degree >= 8 and p.combined_error_m < p.budget_m
    wide = ballistic._core._tle_grade_preset_for(400.0, 1000.0, 3600.0)   # union over the rows
    assert wide.rows_altitude_km_max == 1000.0 and wide.drag
    for args, why in [((250.0, 260.0, 3600.0), "below the lowest"),
                      ((400.0, 36000.0, 3600.0), "GTO"),
                      ((3000.0, 3100.0, 3600.0), "below the lowest"),
                      ((400.0, 410.0, 8 * 86400.0), "7 days")]:
        with pytest.raises(ballistic.ConfigError, match=why):
            ballistic._core._tle_grade_preset_for(*args)


@pytest.mark.nrl
def test_recommended_config_tle_needs_what_the_preset_keeps():
    s = State.from_tle(ISS_L1, ISS_L2)                 # 350 km: drag is kept at 1 h
    with pytest.raises(ballistic.ConfigError, match="keeps drag"):
        ballistic.recommended_config(s, accuracy="tle", horizon_s=3600.0)
    tle = T.parse(ISS_L1, ISS_L2)
    with pytest.raises(ballistic.ConfigError, match="B\\*"):     # this TLE's B* is negative
        ballistic.recommended_config(s, accuracy="tle", horizon_s=3600.0, tle=tle)
    cfg = ballistic.recommended_config(s, accuracy="tle", horizon_s=3600.0, beta_kg_m2=100.0)
    p = ballistic.tle.tle_grade_preset(s, 3600.0)
    assert cfg.gravity_degree == p.gravity_degree and cfg.drag_env is not None
    assert cfg.srp is None and cfg.enable_third_body == p.third_body
    with pytest.raises(ValueError, match="horizon_s"):
        ballistic.recommended_config(s, accuracy="tle")


def test_two_preset_cells_reproduce_at_low_cost():
    """Re-measure the radiation and third-body effects of the MEO 20200 km, 6 h
    cell (both activity epochs, as the table does): within 20% of the table.
    The reference here has no ocean tides (not loaded in the suite); they move
    MEO by millimeters."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "measure_tle_grade_table", Path(__file__).parents[2] / "scripts" / "measure_tle_grade_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    h = 6 * 3600.0
    got = {"radiation": 0.0, "third_body": 0.0}
    for act in mod.ACTIVITY:
        ref = mod._run(((0,), "MEO", 20200, 55.0, act, "reference", h))[1]
        for term in got:
            var = mod._run(((0,), "MEO", 20200, 55.0, act, term, h))[1]
            got[term] = max(got[term], mod._effect(ref, var, h))
    row = next(r for r in ballistic._core._tle_grade_rows()
               if r["regime"] == "MEO" and r["altitude_km"] == 20200.0 and r["horizon_s"] == h)
    assert got["radiation"] == pytest.approx(row["effect_radiation_m"], rel=0.2)
    assert got["third_body"] == pytest.approx(row["effect_third_body_m"], rel=0.2)


def test_propagate_for_records_the_steps_it_flew():
    # Trajectory.propagate kept no history: traj(t) re-propagated on demand
    # and returned None more than an hour from what it had.
    s = State.from_tle(ISS_L1, ISS_L2)
    tr = ballistic.Trajectory(s, ballistic.TrajectoryConfig(gravity_degree=8, ocean_tide_degree=0))
    tr.propagate_for(3 * 3600.0)
    assert tr(60.0, frame="ECEF") is not None
    assert tr._traj.history_size() > 10


def test_fit_state_returns_a_gcrs_state_close_to_sgp4():
    tle = T.parse(ISS_L1, ISS_L2)
    cfg = ballistic.TrajectoryConfig(gravity_degree=8, ocean_tide_degree=0)
    st, rms = T.fit_state(tle, cfg, span_s=3 * 3600.0, step_s=600.0, max_iterations=3)
    assert st.frame == "GCRS" and rms < 2e3
