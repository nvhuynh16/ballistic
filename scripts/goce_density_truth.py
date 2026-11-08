"""Compare the NRLMSIS 2.1 atmosphere model with GOCE's measured air density.

GOCE's accelerometers measured the drag on the satellite, from which ESA and
TU Delft derived the air density along its orbit. This script evaluates
NRLMSIS 2.1 (the atmosphere density model the library uses for drag) at the
same places and times and reports the ratio rho_NRLMSIS / rho_measured point
by point: count, altitude range, median, mean, scatter (standard deviation),
10th/90th percentiles, and the median separately for the dusk (ascending) and
dawn (descending) halves of the orbit. Used both as a measurement (run as a
script) and by tests/python/test_goce_density_truth.py (claim GOCE-DENSITY in
docs/validation_claims.md).

Inputs: the ESA GOCE v2.0 density product under tests/data/goce_density/
(the October 2013 month file if downloaded, else the committed 2013-10-15
slice) and the committed space-weather file tests/data/weather/SW-All.csv
(F10.7 solar flux and Ap geomagnetic index, which NRLMSIS needs). Takes no
command-line options.

Usage:
    PYTHONPATH=./python uv run --with numpy python scripts/goce_density_truth.py

The density file carries its own positions, densities, per-point errors and
quality flags, so no orbit file is needed. NRLMSIS is evaluated through the
public `from_models_snapshot` generator (the same NRLMSIS 2.1 model that
production drag evaluates), with weather from the committed SW-All.csv so the
result does not depend on a download.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# Resolution order: the full month if it was downloaded, else the
# committed verbatim 2013-10-15 slice (see that directory's README). The
# default test uses 10-15 and therefore works from a clean checkout; the
# heavy full-month test needs the month file and skips without it.
_DENSITY_MONTH = (REPO / "tests" / "data" / "goce_density" /
                  "goce_denswind_ac082_v2_0_2013-10.txt")
_DENSITY_SLICE = (REPO / "tests" / "data" / "goce_density" /
                  "goce_denswind_ac082_v2_0_2013-10-15.txt")
DENSITY_FILE = _DENSITY_MONTH if _DENSITY_MONTH.exists() else _DENSITY_SLICE
HAVE_FULL_MONTH = _DENSITY_MONTH.exists()
WEATHER_CSV = REPO / "tests" / "data" / "weather" / "SW-All.csv"

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float):
    """Convert latitude/longitude (degrees) and height (m) to an Earth-fixed position.

    Uses the WGS84 ellipsoid through the library; returns numpy (3,) in
    meters, Earth-centered Earth-fixed (ECEF).
    """
    import ballistic
    return np.asarray(ballistic.geodetic_to_ecef(math.radians(lat_deg), math.radians(lon_deg), alt_m))


def load_rows(days: tuple[str, ...], stride_s: float = 60.0,
              path=None):
    """Read the density file's good-quality rows for the given days.

    `days` are date strings as written in the file ('2013-10-15'); rows are
    thinned so consecutive kept rows are at least `stride_s` seconds apart;
    `path` overrides DENSITY_FILE. Rows whose outlier/filter flag is not 0
    are dropped.

    Handles BOTH product layouts: the month files carry 18 columns (density
    and wind errors), the deorbit files 17 (no error columns), so the
    quality and orbit-direction flags sit at different indices. The layout is
    detected per row from the token count rather than assumed.

    Returns a list of dicts: date, h, m, s (UTC time of day), alt_m (m),
    lon, lat (degrees), rho (measured density, kg/m³) and descending (bool).
    """
    rows = []
    last_t = {}
    with open(path or DENSITY_FILE, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            date = p[0]
            if date not in days:
                continue
            # 18 tokens = month layout (flags at 14/16); 17 = deorbit
            # layout (no error columns → flags at 12/14).
            i_flag, i_desc = (14, 16) if len(p) >= 18 else (12, 14)
            if p[i_flag] != "0":        # outlier/filter flag
                continue
            hh, mm, ss = p[1].split(":")
            t = int(hh) * 3600 + int(mm) * 60 + float(ss)
            if date in last_t and t - last_t[date] < stride_s - 1e-6:
                continue
            last_t[date] = t
            rows.append({
                "date": date, "h": int(hh), "m": int(mm), "s": float(ss),
                "alt_m": float(p[3]), "lon": float(p[4]), "lat": float(p[5]),
                "rho": float(p[8]), "descending": p[i_desc] == "1",
            })
    return rows


def measure(days=("2013-10-01", "2013-10-15"), stride_s=60.0, verbose=False,
            path=None):
    """Compute the ratio rho_NRLMSIS / rho_measured at every kept row.

    `days`, `stride_s` and `path` are passed to load_rows; `verbose` prints
    the result. Returns a dict: n, alt_km_min, alt_km_max, median, mean,
    rms_scatter (standard deviation), p10, p90, median_dusk (ascending
    passes) and median_dawn (descending passes). Raises AssertionError if the
    committed weather file fails to load.
    """
    import ballistic

    ballistic.initialize()
    ok = ballistic._core.load_space_weather(str(WEATHER_CSV))
    assert ok, "committed SW-All.csv failed to load"

    beta = ballistic.BetaProfile1D.constant(300.0)   # irrelevant to density
    ratios, desc_flags = [], []
    rows = load_rows(days, stride_s, path)
    alts = [r["alt_m"] / 1000.0 for r in rows] or [0.0]
    for r in rows:
        y, mo, d = (int(x) for x in r["date"].split("-"))
        epoch = ballistic.Epoch.from_utc(y, mo, d, r["h"], r["m"], r["s"])
        w = ballistic.get_space_weather(epoch)
        sw = ballistic.SpaceWeatherHistory.from_constant(w)
        alt_km = r["alt_m"] / 1000.0
        grid = [alt_km - 10.0, alt_km - 3.0, alt_km, alt_km + 3.0,
                alt_km + 10.0]
        env = ballistic._core.from_models_snapshot(
            epoch, r["lat"], r["lon"], sw, beta, altitude_grid_km=grid)
        s = env.evaluate(geodetic_to_ecef(r["lat"], r["lon"], r["alt_m"]),
                         epoch)
        ratios.append(s.rho_kg_m3 / r["rho"])
        desc_flags.append(r["descending"])
    ratios = np.asarray(ratios)
    desc = np.asarray(desc_flags)
    out = {
        "n": len(ratios),
        "alt_km_min": min(alts),
        "alt_km_max": max(alts),
        "median": float(np.median(ratios)),
        "mean": float(np.mean(ratios)),
        "rms_scatter": float(np.std(ratios)),
        "p10": float(np.percentile(ratios, 10)),
        "p90": float(np.percentile(ratios, 90)),
        "median_dusk": float(np.median(ratios[~desc])) if (~desc).any() else float("nan"),
        "median_dawn": float(np.median(ratios[desc])) if desc.any() else float("nan"),
    }
    if verbose:
        for k, v in out.items():
            print(f"  {k:14s} {v:.3f}" if isinstance(v, float) else f"  {k:14s} {v}")
    return out


if __name__ == "__main__":
    print("NRLMSIS 2.1 vs ESA GOCE measured density (2013-10)")
    measure(verbose=True)
