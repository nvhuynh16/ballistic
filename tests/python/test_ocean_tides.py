"""Ocean tides + Step-2 solid tides: loading surface and effect.


The coefficient store is a process-wide singleton, so every cell that loads
it unloads it again (the session must stay tide-free for other tests)."""

import ballistic
import numpy as np
import pytest
from ballistic import _core

ROW = "255.555 M2    2   2   50.0   0.0   0.0   0.0\n"


@pytest.fixture
def tide_file(tmp_path):
    p = tmp_path / "tides.dat"
    p.write_text("header line\nDoodson Darw l m DelC+ DelS+ DelC- DelS-\n" + ROW, encoding="utf-8")
    yield str(p)
    _core.unload_ocean_tides()


def _propagate(cfg, epoch, pos, vel):
    tr = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=cfg)
    tr.propagate_for(3600.0)
    return np.asarray(tr(3600.0).position)


def test_session_is_tide_free():
    assert not _core.ocean_tides_loaded()
    assert ballistic.TrajectoryConfig().ocean_tide_degree == 50


def test_loaded_tides_change_the_orbit_and_degree_zero_disables(leo_state, tide_file):
    epoch, pos, vel = leo_state
    base = _propagate(ballistic.TrajectoryConfig(), epoch, pos, vel)
    assert _core.load_ocean_tides(tide_file)
    # The file stops at degree 2; the default degree 50 now raises instead of
    # being clamped to it.
    with pytest.raises(ballistic.ConfigError, match="ocean_tide_degree 50"):
        _propagate(ballistic.TrajectoryConfig(), epoch, pos, vel)
    on = _propagate(ballistic.TrajectoryConfig(ocean_tide_degree=2), epoch, pos, vel)
    off = _propagate(ballistic.TrajectoryConfig(ocean_tide_degree=0), epoch, pos, vel)
    assert np.linalg.norm(on - base) > 1e-3          # a 5e-10 C22 term moves it
    assert np.linalg.norm(off - base) == 0.0
    _core.unload_ocean_tides()
    assert np.linalg.norm(_propagate(ballistic.TrajectoryConfig(), epoch, pos, vel) - base) == 0.0
