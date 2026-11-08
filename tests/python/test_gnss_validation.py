"""GNSS 1-hour prediction gate (GNSS-1H): GPS MEO, QZSS IGSO, BeiDou GEO.

A slice of scripts/validate_gnss.py on committed CODE and Wuhan final orbits:
two satellites per group, the first eight hourly starts of one day each,
error at +1 hour. The full campaign measured 0.671 m GPS,

0.734 m QZSS IGSO and 0.799 m BeiDou GEO, every start under 1 m, all with
SRP off (no sourced GPS/QZSS/BeiDou properties yet). The 1.2 m limit is
~1.5x the measured RMSE.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import validate_gnss  # noqa: E402

CASES = [
    # constellation, file, satellites, expected orbit class
    ("gps", "tests/data/sp3/gps/COD0OPSFIN_20240150000_01D_05M_ORB.SP3", {"G01", "G20"}, "MEO"),
    ("qzss", "tests/data/sp3/qzss/COD0MGXFIN_20240150000_01D_05M_ORB.SP3", {"J02", "J03"}, "IGSO"),
    ("beidou-geo", "tests/data/sp3/beidou/WUM0MGXFIN_20240150000_01D_05M_ORB.SP3.gz", {"C01", "C03"}, "GEO"),
]


@pytest.mark.parametrize("constellation,path,sats,cls", CASES, ids=[c[0] for c in CASES])
def test_gnss_one_hour(constellation, path, sats, cls):
    f = REPO / path
    if not f.exists():
        pytest.skip(f"missing test data {f.name} — reacquire: see the README next to it "
                    "(tests/data/sp3/gps, tests/data/sp3/qzss, tests/data/sp3/beidou)")
    rows = validate_gnss.run(constellation, [str(f)], sats_filter=sats,
                             max_starts=8, verbose=False,
                             initialize=False)   # conftest pins EOP
    assert {r["sat"] for r in rows} == sats
    assert {r["class"] for r in rows} == {cls}
    err = np.array([r["error_m"] for r in rows])
    rmse = float(np.sqrt(np.mean(err ** 2)))
    print(f"\n  {constellation}: n={len(err)} RMSE {rmse:.3f} m max {err.max():.3f} m")
    assert len(err) == 16
    assert rmse < 1.2, rmse
    assert err.max() < 1.5, err.max()


# With the per-block cannonball radiation pressure (GNSS-1H-SRP): block and
# mass from the committed excerpt of the IGS metadata file, area from the
# block table with its fitted k. Measured on the first eight hourly starts of
# day 015: G01 (GPS-IIF) 0.091 m (0.701 m without radiation pressure), J02
# (QZS-2I) 0.153 m (0.808 m). Limits: 1.5x the measured values.
EXCERPT = REPO / "tests/data/gnss/igs_satellite_metadata_excerpt.snx"

SRP_CASES = [
    # constellation, file, satellite, block, RMSE limit [m]
    ("gps", "tests/data/sp3/gps/COD0OPSFIN_20240150000_01D_05M_ORB.SP3", "G01", "GPS-IIF", 1.5 * 0.091),
    ("qzss", "tests/data/sp3/qzss/COD0MGXFIN_20240150000_01D_05M_ORB.SP3", "J02", "QZS-2I", 1.5 * 0.153),
]


@pytest.mark.parametrize("constellation,path,sat,block,limit", SRP_CASES, ids=[c[2] for c in SRP_CASES])
def test_gnss_one_hour_with_radiation_pressure(constellation, path, sat, block, limit):
    rows = validate_gnss.run(constellation, [str(REPO / path)], sats_filter={sat},
                             max_starts=8, verbose=False, initialize=False,
                             srp="cannonball", metadata=str(EXCERPT))
    assert len(rows) == 8
    assert {r["block"] for r in rows} == {block}
    rmse = float(np.sqrt(np.mean(np.square([r["error_m"] for r in rows]))))
    print(f"\n  {sat} ({block}) with radiation pressure: RMSE {rmse:.3f} m")
    assert rmse < limit, rmse


def test_srp_off_is_the_default():
    """run() without srp= leaves radiation pressure off (GNSS-1H keeps reproducing)."""
    import inspect
    assert inspect.signature(validate_gnss.run).parameters["srp"].default == "off"
