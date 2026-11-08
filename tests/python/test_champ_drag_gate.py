"""T3: the CHAMP drag claim, as an executable gate.

Heavy tier: BALLISTIC_HEAVY=1, and the subprocess needs pandas+h5py:
  BALLISTIC_HEAVY=1 PYTHONPATH=./python uv run --with numpy --with pandas \
    --with h5py --with pytest pytest tests/python/test_champ_drag_gate.py
Runs a bounded slice (4 arcs) of the CHAMP drag campaign through the REAL
validation script (single-source properties, loaded real weather) and
gates: drag-on RMSE < 2.5 m AND >= 40% improvement vs no-drag (measured
on the full 27-arc day: 1.17 m, -80.6% with the fitted Cd*A and the
default drag weather; 1.82 m with nominal Cd; 1.50 m with

nominal Cd and the older per-arc weather snapshot).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import heavy_enabled

REPO = Path(__file__).resolve().parents[2]
SP3 = REPO / "tests" / "data" / "sp3" / "champ" / "CHAMP_2005_182.sp3"


@pytest.mark.nrl
@pytest.mark.skipif(not heavy_enabled(), reason="BALLISTIC_HEAVY!=1")
def test_champ_drag_gate(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("h5py")
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validate_champ_drag.py"),
         "--sp3", str(SP3), "--mode", "propagate", "--duration", "3600",
         "--sample-interval", "240",
         "--hdf5", str(tmp_path / "champ_gate.h5")],
        capture_output=True, text=True, timeout=560, env=env, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]

    def rmse(config):
        m = re.search(rf"^{re.escape(config)}\s+([\d.]+)", out.stdout, re.M)
        assert m, f"{config} row missing:\n{out.stdout[-2000:]}"
        return float(m.group(1))

    base = rmse("Baseline (no SRP/drag)")
    drag = rmse("Drag only")
    print(f"\n  T3: baseline {base:.2f} m  drag-on {drag:.2f} m "
          f"({(base - drag) / base * 100.0:.1f}% improvement)")
    assert drag < 2.5, (base, drag)
    assert (base - drag) / base >= 0.40, (base, drag)
