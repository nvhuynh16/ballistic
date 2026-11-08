"""ballistic.doctor(): the installation check."""


import subprocess
import sys
from pathlib import Path

import ballistic

REPO = Path(__file__).resolve().parents[2]


def _by_name(report):
    return {c.name: c for c in report.checks}


def test_doctor_passes_with_data_loaded():
    report = ballistic.doctor(load=False, verbose=False)
    checks = _by_name(report)
    assert report.ok, str(report)
    assert checks["Earth orientation"].status in ("ok", "warn")
    assert checks["self-test: SGP4"].status == "ok"
    # The embedded Sentinel-1A arc: ~0.05 m with EOP (bound 0.5 m).
    assert checks["self-test: Sentinel-1A"].status == "ok", checks["self-test: Sentinel-1A"].detail
    # The NRL drag models: installed (with their license terms) or how to install them.
    nrl = checks["NRL models (drag)"]
    assert nrl.status == "note"
    marker = "non-profit use only" if ballistic.nrl_models().installed else "install_nrl_models"
    assert marker in nrl.detail


def test_doctor_fails_without_earth_orientation(monkeypatch):
    monkeypatch.setattr(ballistic, "eop_coverage", lambda: None)
    report = ballistic.doctor(load=False, verbose=False)
    checks = _by_name(report)
    assert not report.ok
    assert checks["Earth orientation"].status == "fail"
    assert "ballistic.load()" in checks["Earth orientation"].detail
    assert checks["self-test: Sentinel-1A"].status == "warn"   # skipped, says why


def test_module_entry_point_rejects_unknown_commands():
    r = subprocess.run([sys.executable, "-m", "ballistic", "doctr"], capture_output=True,
                       text=True, timeout=120, cwd=REPO)
    assert r.returncode == 2 and "usage" in r.stderr
