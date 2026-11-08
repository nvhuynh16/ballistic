"""Silent-substitution sweep, Python side.


Each loader that used to skip, default or ignore something now raises or
warns; each cell is named in docs/trap_census.md.
"""

from pathlib import Path

import numpy as np
import pytest

import ballistic
from ballistic.errors import DataSkippedWarning, RemovedApiError
from ballistic.loader import Loader, measured_until_mjd

REPO = Path(__file__).resolve().parents[2]
GOCE_SLICE = REPO / "tests/data/goce_density/goce_denswind_ac082_v2_0_2013-10-15.txt"

ROW = ("2013-10-15 00:00:00.000 {sys} 238625.741  132.689  70.790  8.846 108.217 "
       "{rho}  -149.09   56.36     2.67 4.374149e-12     6.46  0  0  1  0\n")


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("name", ["load_density_table", "load_interpolated_density"])
def test_density_store_loaders_removed(name):
    # They filled a global store that no propagation read.
    with pytest.raises(RemovedApiError, match="from_accelerometer_density"):
        getattr(ballistic, name)


def test_density_table_reads_committed_slice():
    mjds, rhos = ballistic.drag.parse_density_table(str(GOCE_SLICE))
    assert len(mjds) == 8640
    assert mjds[0] == pytest.approx(56580.0, abs=1e-9)      # 2013-10-15 00:00 UTC
    assert all(0 < r < 1 for r in rhos)


def test_density_table_honors_gps_time(tmp_path):
    utc = _write(tmp_path, "u.txt", ROW.format(sys="UTC", rho="8.45e-11"))
    gps = _write(tmp_path, "g.txt", ROW.format(sys="GPS", rho="8.45e-11"))
    (m_utc,), _ = ballistic.drag.parse_density_table(utc)
    (m_gps,), _ = ballistic.drag.parse_density_table(gps)
    # GPS = UTC + 16 s in 2013: the same clock reading is 16 s earlier in UTC.
    assert (m_utc - m_gps) * 86400.0 == pytest.approx(16.0, abs=1e-6)


def test_density_table_unknown_time_system_raises(tmp_path):
    p = _write(tmp_path, "x.txt", ROW.format(sys="LOC", rho="8.45e-11"))
    with pytest.raises(ValueError, match="time system"):
        ballistic.drag.parse_density_table(p)


def test_density_table_malformed_row_raises(tmp_path):
    p = _write(tmp_path, "x.txt", ROW.format(sys="UTC", rho="8.45e-11") + "2013-10-15 garbage\n")
    with pytest.raises(ValueError, match=":2:"):
        ballistic.drag.parse_density_table(p)


def test_density_table_fill_values_counted(tmp_path):
    p = _write(tmp_path, "x.txt", ROW.format(sys="UTC", rho="8.45e-11")
               + ROW.format(sys="UTC", rho="9.999999e+99").replace("00:00:00", "00:00:10"))
    with pytest.warns(DataSkippedWarning, match="1 row"):
        mjds, _ = ballistic.drag.parse_density_table(p)
    assert len(mjds) == 1


def test_beta_table_needs_header(tmp_path):
    with pytest.raises(ValueError, match="header"):
        ballistic.drag.load_beta_table(_write(tmp_path, "b.csv", "200,40\n300,45\n"))


def test_beta_table_bad_value_raises(tmp_path):
    with pytest.raises(ValueError, match="beta"):
        ballistic.drag.load_beta_table(_write(tmp_path, "b.csv", "altitude_km,beta\n200,40\n300,-1\n"))


def test_ea_table_unknown_frame_raises(tmp_path):
    p = _write(tmp_path, "ea.csv", "# frame: teme\nkey,ax,ay,az\n0,0,0,0\n10,1e-6,0,0\n")
    with pytest.raises(ValueError, match="frame"):
        ballistic.drag.load_empirical_acceleration(p)


def test_ea_table_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="no EA rows"):
        ballistic.drag.load_empirical_acceleration(_write(tmp_path, "ea.csv", "key,ax,ay,az\n"))


def test_staleness_follows_the_data_not_the_file_time(tmp_path):
    # A freshly written copy of an old weather file must still be refreshed.
    src = REPO / "tests/data/weather/SW-All.csv"
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    head = [lines[0]] + [line for line in lines[1:] if line.startswith("2020-")]
    p = tmp_path / "SW-All.csv"
    p.write_text("".join(head), encoding="utf-8")
    assert measured_until_mjd("space_weather", p) == pytest.approx(59214.0)   # 2020-12-31
    loader = Loader(directory=str(tmp_path), offline=True)
    assert loader._file_age_days(p) < 1.0
    assert loader._needs_update(p, "space_weather")
    status = loader.status()["space_weather"]
    assert status["data_age_days"] > 365


def test_gravity_degree_above_model_raises():
    epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    cfg = ballistic.TrajectoryConfig(gravity_degree=400)
    with pytest.raises(ballistic.ConfigError, match="gravity_degree 400"):
        ballistic.Trajectory(ballistic.State.from_itrf(epoch, np.array([6778137.0, 0, 0]), np.array([0, 7175.0, 0])),
                             config=cfg)


def test_drag_helper_has_no_quiet_fallback():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import _drag_env_helper as helper
    finally:
        sys.path.pop(0)
    helper.load_weather_source(ballistic, helper.COMMITTED_WEATHER_CSV)
    # 2050 is past the committed file: used to print a warning and return quiet weather.
    with pytest.raises(Exception):
        helper.get_weather_for_epoch(ballistic, ballistic.Epoch.from_utc(2050, 1, 1, 0, 0, 0))
