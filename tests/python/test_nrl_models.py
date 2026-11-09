"""The NRLMSIS 2.1 / HWM14 plugin: absent from the package, installed on request.

A fresh installation has no NRL code (NRLMSIS 2.1 is licensed for
non-commercial use only). Without the plugin, model-based drag raises a
ConfigError that says how to install it, ``recommended_config`` leaves drag
off with one ConfigHintWarning, and ``install_nrl_models`` refuses to install
until the license is accepted. These run in a fresh interpreter on an empty
data directory with BALLISTIC_NRL_MODELS cleared, because a process that has
loaded the plugin keeps it. The installation itself is tested from the
plugin built in this tree (``source=``), never from the network.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ballistic
import ballistic.nrl as nrl

REPO = Path(__file__).resolve().parents[2]


def _run(tmp_path, code, **extra_env):
    """Run `code` in a fresh interpreter without the plugin; returns the last
    stdout line parsed as JSON, and the whole stdout."""
    data = tmp_path / "data"
    app = data / "ballistic"
    app.mkdir(parents=True, exist_ok=True)
    (app / "finals2000A.all").write_bytes((REPO / "tests/data/eop/finals2000A.all").read_bytes())
    env = dict(os.environ, BALLISTIC_DATA_DIR=str(app), **extra_env)
    env.pop("BALLISTIC_NRL_MODELS", None)
    env.pop("BALLISTIC_STRICT_WARNINGS", None)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve())
                                            for p in env["PYTHONPATH"].split(os.pathsep))
    prelude = textwrap.dedent(f"""
        import json, warnings
        import numpy as np
        import ballistic
        ballistic.initialize()
        ballistic._core.load_space_weather({str(REPO / "tests/data/weather/SW-All.csv")!r})
        T0 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        STATE = ballistic.State.from_itrf(T0, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0])
    """)
    r = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)],
                       capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stdout


def test_without_the_plugin_model_drag_says_how_to_install(tmp_path):
    out, _ = _run(tmp_path, """
        st = ballistic.nrl_models()
        try:
            ballistic.make_drag_environment(T0, np.array(STATE.position), np.array(STATE.velocity),
                                            0.0, ballistic.TrajectoryConfig(),
                                            ballistic.BetaProfile1D.constant(150.0))
            msg = None
        except ballistic.ConfigError as e:
            msg = str(e)
        print(json.dumps({"installed": st.installed, "msg": msg, "install_path": st.install_path}))
    """)
    assert out["installed"] is False
    msg = out["msg"]
    assert msg is not None
    for part in ("install_nrl_models(accept_license=True)", "NRL-SOF-014-1", "non-profit",
                 "commercial use is not permitted", out["install_path"], "BALLISTIC_NRL_MODELS",
                 "from_accelerometer_density", "github.com/nvhuynh16/ballistic/releases"):
        assert part in msg, (part, msg)


def test_recommended_config_leaves_drag_off_and_says_what_it_costs(tmp_path):
    out, _ = _run(tmp_path, """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = ballistic.recommended_config(STATE, "S1A")
        hints = [str(x.message) for x in w
                 if issubclass(x.category, ballistic.ConfigHintWarning) and "NRL" in str(x.message)]
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            quiet = ballistic.recommended_config(STATE, "S1A", drag=False)
        quiet_hints = [str(x.message) for x in w2 if "NRL" in str(x.message)]
        try:
            ballistic.recommended_config(STATE, "S1A", drag=True)
            required = None
        except ballistic.ConfigError as e:
            required = str(e)
        print(json.dumps({"drag": cfg.drag_env is not None, "srp": cfg.srp is not None,
                          "hints": hints, "quiet_drag": quiet.drag_env is not None,
                          "quiet_hints": quiet_hints, "required": required}))
    """)
    assert out["drag"] is False and out["srp"] is True
    assert len(out["hints"]) == 1
    hint = out["hints"][0]
    assert "4 to 8" in hint and "install_nrl_models(accept_license=True)" in hint
    assert "drag=False" in hint
    assert out["quiet_drag"] is False and out["quiet_hints"] == []
    assert out["required"] and "install_nrl_models" in out["required"]


def test_install_refuses_without_accepting_the_license(tmp_path):
    out, stdout = _run(tmp_path, """
        r = ballistic.install_nrl_models()
        print(json.dumps({"returned": r, "installed": ballistic.nrl_models().installed}))
    """)
    assert out == {"returned": None, "installed": False}
    assert "COMMERCIAL USE IS NOT PERMITTED" in stdout
    assert "NRL-SOF-014-1" in stdout
    assert "Nothing was installed" in stdout


def test_offline_install_without_a_file_is_refused(tmp_path):
    out, _ = _run(tmp_path, """
        try:
            ballistic.install_nrl_models(accept_license=True)
            msg = None
        except ballistic.ConfigError as e:
            msg = str(e)
        print(json.dumps({"msg": msg}))
    """, BALLISTIC_OFFLINE="1")
    assert out["msg"] and "BALLISTIC_OFFLINE" in out["msg"] and "source=" in out["msg"]


def test_a_file_that_is_not_the_plugin_is_refused_and_removed(tmp_path):
    bogus = tmp_path / "bogus.so"
    bogus.write_text("not a library\n", encoding="utf-8")
    out, _ = _run(tmp_path, f"""
        import pathlib
        try:
            ballistic.install_nrl_models(accept_license=True, source={str(bogus)!r})
            msg = None
        except ballistic.ConfigError as e:
            msg = str(e)
        st = ballistic.nrl_models()
        print(json.dumps({{"msg": msg, "installed": st.installed,
                          "left": pathlib.Path(st.install_path).exists()}}))
    """)
    assert out["msg"] and "not a loadable library" in out["msg"]
    assert out["installed"] is False and out["left"] is False


@pytest.mark.nrl
def test_install_from_a_file_then_drag_works_in_later_sessions(tmp_path):
    plugin = ballistic.nrl_models().path   # the one the suite uses (conftest.py)
    out, _ = _run(tmp_path, f"""
        import pathlib
        path = ballistic.install_nrl_models(accept_license=True, source={plugin!r})
        st = ballistic.nrl_models()
        accepted = (pathlib.Path(path).parent / "ACCEPTED.txt").read_text(encoding="utf-8")
        cfg = ballistic.recommended_config(STATE, "S1A")
        print(json.dumps({{"path": str(path), "st_path": st.path, "models": st.models,
                          "accepted": accepted, "drag": cfg.drag_env is not None}}))
    """)
    assert out["path"] == out["st_path"]
    assert out["models"] == "NRLMSIS 2.1, HWM14"
    assert "License accepted" in out["accepted"] and "NRL-SOF-014-1" in out["accepted"]
    assert out["drag"] is True
    # A new session finds the installed plugin by itself.
    later, _ = _run(tmp_path, """
        print(json.dumps({"installed": ballistic.nrl_models().installed,
                          "drag": ballistic.recommended_config(STATE, "S1A").drag_env is not None}))
    """)
    assert later == {"installed": True, "drag": True}


def test_release_asset_names_follow_the_platform():
    name = nrl._asset_name()
    assert name.startswith("ballistic_nrl-")
    assert name.endswith({"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so"))
    url = ballistic.nrl_models().download_url
    assert url == f"{nrl.RELEASES_URL}/download/v{ballistic._core.__version__}/{name}"


@pytest.mark.nrl
def test_loaded_plugin_is_reported():
    st = ballistic.nrl_models()
    assert st and st.models == "NRLMSIS 2.1, HWM14" and Path(st.path).is_file()
    assert ballistic.get_init_status().nrl_models_loaded


def test_license_warning_is_gone():
    for name in ("LicenseWarning", "LicenceWarning"):
        with pytest.raises(ballistic.RemovedApiError, match="install_nrl_models"):
            getattr(ballistic, name)
