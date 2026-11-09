"""The README quick start runs as written.


The block after the ``<!-- quickstart`` marker in README.md is executed
verbatim in a fresh interpreter, on a hermetic data directory holding only
the committed Earth orientation and space-weather files, with downloads
forbidden (BALLISTIC_OFFLINE=1; the ocean tides are then skipped with a
warning). Its printed position is checked against the ESA precise orbit one
hour after the start (S1A POEORB of 5 January 2026, 12:59:42 UTC).
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRUTH_1H = np.array([-1430580.257221, -6627916.892386, 2017275.123945])


def quickstart_code(readme_text):
    m = re.search(r"<!-- quickstart.*?-->\s*```python\n(.*?)```", readme_text, re.S)
    assert m, "README.md has no quick-start block after a '<!-- quickstart' marker"
    return m.group(1)


def test_readme_quickstart_runs_and_lands_on_the_precise_orbit(tmp_path):
    app = tmp_path / "data" / "ballistic"
    app.mkdir(parents=True)
    shutil.copy(REPO / "tests/data/eop/finals2000A.all", app / "finals2000A.all")
    shutil.copy(REPO / "tests/data/weather/SW-All.csv", app / "SW-All.csv")
    env = dict(os.environ, BALLISTIC_DATA_DIR=str(app), BALLISTIC_OFFLINE="1")
    env.pop("BALLISTIC_STRICT_WARNINGS", None)   # the skipped ocean tides warn, by design
    # A fresh installation has no NRL drag models (a separate, non-commercial
    # plugin): the quick start must work without them.
    env.pop("BALLISTIC_NRL_MODELS", None)
    if env.get("PYTHONPATH"):   # a source-tree run (PYTHONPATH=./python); CI uses the wheel
        env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve())
                                            for p in env["PYTHONPATH"].split(os.pathsep))
    script = tmp_path / "quickstart.py"
    script.write_text(quickstart_code((REPO / "README.md").read_text(encoding="utf-8")), encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       timeout=300, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    printed = np.array([float(x) for x in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", r.stdout)[-3:]])
    # Measured 0.05 m (out of the box: no NRL drag plugin, so no drag; no ocean tides
    # here); 2 m catches a broken path.
    assert np.linalg.norm(printed - TRUTH_1H) < 2.0, (r.stdout, np.linalg.norm(printed - TRUTH_1H))


def test_quickstart_extractor_finds_the_marked_block():
    text = "intro\n<!-- quickstart: x -->\n```python\nprint(1)\n```\n"
    assert quickstart_code(text) == "print(1)\n"
