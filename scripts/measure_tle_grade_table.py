#!/usr/bin/env python3
"""Measure which forces still matter when the start state is a TLE.

A TLE (two-line element set, the public orbit format meant for the SGP4 propagator)
gives a start state that is uncertain by ~0.5 km in low Earth orbit (LEO; Flohrer et al.
2009, below). A force whose effect on the trajectory stays far below that error buys no
accuracy and costs run time. This script MEASURES, per orbit and horizon,
how far each optional term moves the trajectory, and writes the rule

    keep a term if its effect at the horizon exceeds 10% of the TLE error
    budget of the orbit class; use the lowest gravity degree whose
    truncation effect (vs degree 70) is below the same threshold

into the generated header include/ballistic/presets/tle_grade_table.hpp
(read by presets::tle_grade and recommended_config(..., accuracy="tle")).

Reference: the library's recommended configuration for a synthetic object
(ballistic coefficient m/(Cd A) = 50 kg/m^2, radiation area-to-mass A/m = 0.02 m^2/kg,
radiation coefficient Cr 1.3 -- a small, light object, so drag and radiation are
measured on the conservative side; heavier or denser objects feel them less), the
EGM2008 gravity field to degree 70, Sun/Moon/Jupiter, solid + ocean tides (FES2004 ocean
tide model to degree 50), relativity, and drag from the NRLMSIS 2.1 atmosphere model
evaluated along the orbit on the measured space weather (the F10.7 solar-flux and Ap
geomagnetic indices). Each variant removes one term (drag is not varied above 1500 km),
or lowers the gravity degree; "effect" is the largest position difference to the
reference, in the inertial frame (GCRS), at any sample up to the horizon (samples every
600 s in the first hour, then every 1800 s). Orbits are near-circular (e = 0.001) at
the altitudes and inclinations in ORBITS, each run at a low and a high solar-activity
date (ACTIVITY); the effect kept is the maximum over inclinations and dates.

Error budget: Flohrer, Krag, Klinkrad, Bastida Virgili & Frueh (2009),
"Improving ESA's collision risk estimates by an assessment of the TLE orbit
errors of the US SSN catalogue" (SSN = Space Surveillance Network), Proc. 5th
European Conference on Space Debris, ESA SP-672, Table 1 (catalog of 2008-01-01;
1-sigma over 24 h centered on the TLE epoch): LEO (apogee < 2000 km) radial 0.102,
along-track 0.471, cross-track 0.126 km; medium Earth orbit (MEO) 0.073 / 0.131 /
0.054 km; geostationary orbit (GEO) 0.359 / 0.432 / 0.086 km. The budget is their
root-sum-square (LEO 0.50 km, MEO 0.16 km, GEO 0.57 km), held constant over the
horizon -- the TLE's error grows with time, so this keeps MORE terms than strictly
needed.

Each preset (all dropped terms together) is then checked against the full configuration
and timed: combined_error_m and speedup in the table.

Options: --workers N (parallel processes, default 8), --out PATH (default
include/ballistic/presets/tle_grade_table.hpp in this repository), --json PATH (also
write the full measurement).

Run (needs ballistic.load()'s files: Earth orientation parameters (EOP), space weather,
ocean tides; loaded offline in each worker):
    PYTHONPATH=./python uv run --with numpy python scripts/measure_tle_grade_table.py \
        --workers 12 --out include/ballistic/presets/tle_grade_table.hpp
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Flohrer et al. 2009, Table 1 (km): radial, along-track, cross-track.
FLOHRER_KM = {"LEO": (0.102, 0.471, 0.126), "MEO": (0.073, 0.131, 0.054), "GEO": (0.359, 0.432, 0.086)}
THRESHOLD_FRACTION = 0.10

HORIZONS_S = (3600.0, 6 * 3600.0, 86400.0, 3 * 86400.0, 7 * 86400.0)
ACTIVITY = {"low": (2019, 7, 1), "high": (2024, 8, 1)}   # solar minimum / near cycle-25 maximum
# (regime, altitude km, inclinations deg)
ORBITS = ([("LEO", h, (51.6, 97.8)) for h in (300, 400, 500, 600, 800, 1000, 1400, 1800)] +
          [("MEO", 5000, (55.0,)), ("MEO", 10000, (55.0,)), ("MEO", 20200, (55.0,)),
           ("GEO", 35786, (0.05,))])
DEGREES = (2, 4, 8, 12, 16, 20, 30, 40, 50)
REFERENCE_DEGREE = 70
TERMS = ("drag", "radiation", "third_body", "tides", "ocean_tides", "relativity")

MASS_KG, CD, BETA = 100.0, 2.2, 50.0
DRAG_AREA = MASS_KG / (CD * BETA)
SRP_AREA, CR = 2.0, 1.3


def budget_m(regime):
    """TLE error budget of an orbit class ("LEO", "MEO", "GEO") in meters: the
    root-sum-square of the three Flohrer et al. 1-sigma errors."""
    return 1e3 * math.sqrt(sum(x * x for x in FLOHRER_KM[regime]))


def _samples():
    """Sample times in seconds after the start: every 600 s for the first hour, then
    every 1800 s up to the longest horizon."""
    s = [600.0 * k for k in range(1, 7)]
    t = 3600.0 + 1800.0
    while t <= HORIZONS_S[-1] + 1e-6:
        s.append(t)
        t += 1800.0
    return s


def _init():
    """Worker initializer: load the library's data files from the local cache (no download)."""
    import ballistic as b
    b.load(offline=True, verbose=False)


def _run(task):
    """Propagate one trajectory variant and sample it.

    task: (key, regime, altitude km, inclination deg, activity "low"/"high", variant,
    horizon s). variant is "reference", a term name from TERMS (that term removed),
    "degNN" (gravity degree NN), or "preset:" + JSON of a preset (keep flags and
    gravity_degree). Returns (key, positions numpy (N,3) in meters in the GCRS at the
    sample times up to the horizon, wall-clock seconds of the propagation).
    """
    import ballistic as b
    key, regime, alt_km, inc, act, variant, horizon = task
    y, mo, d = ACTIVITY[act]
    t0 = b.Epoch.from_utc(y, mo, d, 0, 0, 0)
    a = b.constants.earth.RADIUS_EQUATORIAL + alt_km * 1e3
    state = b.State.from_keplerian(t0, a_m=a, e=0.001, i_deg=inc, raan_deg=40.0, argp_deg=0.0,
                                   true_anomaly_deg=0.0)
    cfg = b.recommended_config(state, mass_kg=MASS_KG, srp_area_m2=SRP_AREA,
                               drag_area_m2=DRAG_AREA, cd=CD, cr=CR, gravity_degree=REFERENCE_DEGREE)
    if variant.startswith("deg"):
        cfg = cfg.replace(gravity_degree=int(variant[3:]))
    elif variant.startswith("preset:"):
        p = json.loads(variant[7:])
        cfg = cfg.replace(gravity_degree=p["gravity_degree"],
                          drag_env=cfg.drag_env if p["drag"] else None,
                          srp=cfg.srp if p["radiation"] else None,
                          enable_third_body=p["third_body"], enable_solid_tides=p["tides"],
                          ocean_tide_degree=50 if p["ocean_tides"] else 0,
                          enable_relativistic=p["relativity"])
    elif variant != "reference":
        cfg = cfg.replace(**{
            "drag": {"drag_env": None}, "radiation": {"srp": None},
            "third_body": {"enable_third_body": False}, "tides": {"enable_solid_tides": False},
            "ocean_tides": {"ocean_tide_degree": 0}, "relativity": {"enable_relativistic": False},
        }[variant])
    tr = b.Trajectory(state, cfg)
    out, t_start = [], time.perf_counter()
    for s in _samples():
        if s > horizon + 1e-6:
            break
        tr.propagate_to(t0 + s)
        out.append(np.asarray(tr.state_gcrs().position))
    return key, np.array(out), time.perf_counter() - t_start


def _effect(ref, var, horizon):
    """Largest distance in meters between two sampled trajectories (numpy (N,3)) over
    the samples up to horizon seconds."""
    n = sum(1 for s in _samples() if s <= horizon + 1e-6)
    return float(np.max(np.linalg.norm(var[:n] - ref[:n], axis=1)))


def measure(workers):
    """Run every variant on `workers` processes and return one row dict per (orbit class,
    altitude, horizon): budget, threshold, chosen gravity degree, keep flags, measured
    effects, and the preset's combined error (meters) and speed-up against the full
    configuration."""
    cells = [(regime, alt, inc, act) for regime, alt, incs in ORBITS for inc in incs for act in ACTIVITY]
    tasks = []
    for regime, alt, inc, act in cells:
        variants = ["reference"] + [t for t in TERMS if not (t == "drag" and alt > 1500)]
        variants += [f"deg{d}" for d in DEGREES]
        for v in variants:
            tasks.append(((regime, alt, inc, act, v), regime, alt, inc, act, v, HORIZONS_S[-1]))
    with mp.Pool(workers, initializer=_init) as pool:
        res = {k: (x, w) for k, x, w in pool.imap_unordered(_run, tasks, chunksize=1)}

    effects = {}   # (regime, alt, horizon) -> {term: max over inc/activity}
    for regime, alt, inc, act in cells:
        ref = res[(regime, alt, inc, act, "reference")][0]
        for h in HORIZONS_S:
            e = effects.setdefault((regime, alt, h), {})
            for v in [t for t in TERMS] + [f"deg{d}" for d in DEGREES]:
                if (regime, alt, inc, act, v) not in res:
                    e.setdefault(v, 0.0)
                    continue
                e[v] = max(e.get(v, 0.0), _effect(ref, res[(regime, alt, inc, act, v)][0], h))

    rows = []
    for (regime, alt, h), e in sorted(effects.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        thr = THRESHOLD_FRACTION * budget_m(regime)
        keep = {t: e[t] > thr for t in TERMS}
        if keep["ocean_tides"]:
            keep["tides"] = True      # ocean tides are applied through the tide model
        deg = next((d for d in DEGREES if e[f"deg{d}"] <= thr), REFERENCE_DEGREE)
        rows.append({"regime": regime, "altitude_km": alt, "horizon_s": h, "budget_m": budget_m(regime),
                     "threshold_m": thr, "gravity_degree": deg, "keep": keep,
                     "effects_m": {k: e[k] for k in TERMS},
                     "degree_effects_m": {d: e[f"deg{d}"] for d in DEGREES}})

    # Verify each preset as a whole (all dropped terms together) and time it.
    vtasks = []
    for r in rows:
        p = dict(r["keep"], gravity_degree=r["gravity_degree"])
        incs = next(i for g, a, i in ORBITS if a == r["altitude_km"])
        for inc in incs:
            for act in ACTIVITY:
                key = (r["regime"], r["altitude_km"], inc, act, "preset", r["horizon_s"])
                vtasks.append((key, r["regime"], r["altitude_km"], inc, act,
                               "preset:" + json.dumps(p), r["horizon_s"]))
                vtasks.append((key[:4] + ("full", r["horizon_s"]), r["regime"], r["altitude_km"], inc,
                               act, "reference", r["horizon_s"]))
    with mp.Pool(workers, initializer=_init) as pool:
        vres = {k: (x, w) for k, x, w in pool.imap_unordered(_run, vtasks, chunksize=1)}
    for r in rows:
        incs = next(i for g, a, i in ORBITS if a == r["altitude_km"])
        err, tf, tp = 0.0, 0.0, 0.0
        for inc in incs:
            for act in ACTIVITY:
                base = (r["regime"], r["altitude_km"], inc, act)
                xp, wp = vres[base + ("preset", r["horizon_s"])]
                xf, wf = vres[base + ("full", r["horizon_s"])]
                err = max(err, float(np.max(np.linalg.norm(xp - xf, axis=1))))
                tf += wf
                tp += wp
        r["combined_error_m"] = err
        r["speedup"] = tf / tp
    return rows


def write_header(rows, path):
    """Write the rows as the C++ table kTleGradeTable to path.

    The header carries no measurement date or commit: it is a public file, and that
    history belongs in the commit that regenerates it.
    """
    lines = [
        "#pragma once",
        "",
        "// Generated by scripts/measure_tle_grade_table.py; do not edit.",
        "//",
        "// Per orbit class, altitude and horizon: which optional forces move the",
        "// trajectory by more than 10% of the TLE error budget (Flohrer et al. 2009,",
        "// ESA SP-672, Table 1, root-sum-square of the 1-sigma radial / along-track /",
        "// cross-track errors: LEO 498 m, MEO 159 m, GEO 568 m), and the lowest",
        "// EGM2008 degree whose truncation stays below it. Effects are the largest",
        "// GCRS position change [m] up to the horizon, maximum over the inclinations",
        # The next two literals concatenate into ONE header line: the generated
        # epoch line must carry its own PUB-DATE exemption (the trailing Python
        # comment exempts only this script's line).
        "// and two solar-activity epochs (2019-07-01 low, 2024-08-01 high) measured,"  # guard-exempt(PUB-DATE): data
        "  // guard-exempt(PUB-DATE): data",
        "// for an object with m/(Cd A) = 50 kg/m^2 and A/m = 0.02 m^2/kg.",
        "// combined_error_m: the preset as a whole against the full configuration.",
        "",
        "namespace ballistic::presets {",
        "",
        "struct TleGradeRow {",
        "    const char* regime;      // \"LEO\", \"MEO\", \"GEO\" (Flohrer et al. classes)",
        "    double altitude_km;      // circular orbit measured",
        "    double horizon_s;",
        "    double budget_m;         // TLE error budget of the class",
        "    double threshold_m;      // 10% of it",
        "    int gravity_degree;",
        "    bool drag, radiation, third_body, tides, ocean_tides, relativity;",
        "    double effect_drag_m, effect_radiation_m, effect_third_body_m, effect_tides_m,",
        "        effect_ocean_tides_m, effect_relativity_m;",
        "    double combined_error_m;",
        "    double speedup;          // wall time full / preset (same machine, same load)",
        "};",
        "",
        "inline constexpr TleGradeRow kTleGradeTable[] = {",
    ]
    def b(v):
        return "true" if v else "false"

    for r in rows:
        k, e = r["keep"], r["effects_m"]
        lines.append(
            f"    {{\"{r['regime']}\", {r['altitude_km']:.1f}, {r['horizon_s']:.1f}, {r['budget_m']:.1f}, "
            f"{r['threshold_m']:.1f}, {r['gravity_degree']}, "
            f"{b(k['drag'])}, {b(k['radiation'])}, {b(k['third_body'])}, {b(k['tides'])}, "
            f"{b(k['ocean_tides'])}, {b(k['relativity'])}, "
            f"{e['drag']:.4g}, {e['radiation']:.4g}, {e['third_body']:.4g}, {e['tides']:.4g}, "
            f"{e['ocean_tides']:.4g}, {e['relativity']:.4g}, {r['combined_error_m']:.4g}, "
            f"{r['speedup']:.3g}}},")
    lines += ["};", "", "}  // namespace ballistic::presets", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    """Return the argparse parser for the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=str(REPO / "include/ballistic/presets/tle_grade_table.hpp"))
    p.add_argument("--json", default=None, help="also write the full measurement as JSON")
    return p


def main():
    """Measure the table, write the header (and optional JSON), and print one line per row."""
    args = build_parser().parse_args()
    t = time.time()
    rows = measure(args.workers)
    write_header(rows, args.out)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    for r in rows:
        kept = [t for t in TERMS if r["keep"][t]]
        print(f"{r['regime']} {r['altitude_km']:6.0f} km {r['horizon_s'] / 3600:6.1f} h  deg {r['gravity_degree']:2d}"
              f"  keep {','.join(kept) or '-':45s} combined {r['combined_error_m']:8.1f} m  x{r['speedup']:.1f}")
    print(f"done in {time.time() - t:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
