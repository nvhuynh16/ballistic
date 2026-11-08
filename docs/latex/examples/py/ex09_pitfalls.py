"""
Tutorial example 09 — Pitfalls, demonstrated (Python)

  1. Wrong ITRF realization — measured LEO divergence ITRF2014 vs ITRF2020.
  2. Missing EOP — actionable check (get_init_status().eop_loaded) + the
     documented ~3 m LEO cost when UT1-UTC defaults to 0.
  3. Stale leap seconds — embedded coverage + whether a file was loaded.

    python ex09_pitfalls.py [out.json]
"""
import json
import math
import sys

import numpy as np

import ballistic


def track(t0, p0, v0, itrf):
    c = ballistic.TrajectoryConfig(itrf=itrf)
    tr = ballistic.Trajectory(ballistic.State.from_itrf(t0, p0, v0), config=c)
    while (tr.epoch() - t0) < 3660.0:
        tr.step()
    return tr


def main(out="docs/latex/results/ex09_pitfalls.json"):
    ballistic.initialize()
    status = ballistic.get_init_status()
    leap = ballistic.get_leap_second_info()

    # --- Pitfall 1: wrong ITRF (measured) --------------------------------
    t0 = ballistic.Epoch.from_utc(2024, 6, 1, 0, 0, 0)
    p0 = np.array([6878137.0, 0.0, 0.0])
    # ~500 km circular LEO, Earth-fixed: inertial 7613 m/s minus omega x r
    v0 = np.array([0.0, 7613.0 - 501.56, 0.0])
    tr14 = track(t0, p0, v0, ballistic.ITRFRealization.ITRF2014)
    tr20 = track(t0, p0, v0, ballistic.ITRFRealization.ITRF2020)
    offs = np.arange(0.0, 3600.0 + 1.0, 60.0)
    s14 = tr14([float(t) for t in offs], frame="ECEF")
    s20 = tr20([float(t) for t in offs], frame="ECEF")
    sse, n = 0.0, 0
    for a, b in zip(s14, s20):
        if a is not None and b is not None:
            sse += float(np.sum((np.asarray(a.position) - np.asarray(b.position)) ** 2))
            n += 1
    itrf_rmse = math.sqrt(sse / n) if n else -1.0

    eop_ok = bool(status.eop_loaded)
    leap_from_file = bool(leap.loaded_from_file)
    tai_utc = int(leap.current_tai_utc)

    print("1) Wrong ITRF (ITRF2014 vs ITRF2020), 1 h LEO:")
    print(f"     position RMSE = {itrf_rmse:.6f} m  (small but real)")
    print(f"2) EOP loaded            : {eop_ok}  "
          f"(if false: ~3 m LEO error from UT1-UTC=0 — run Loader().load_eop())")
    print(f"3) leap from file        : {leap_from_file} ; "
          f"TAI-UTC = {tai_utc} s ; entries = {leap.entry_count}")
    print("     (embedded covers through 2025; refresh leap-seconds.list "
          "before querying later dates)")

    with open(out, "w") as f:
        json.dump({
            "wrong_itrf_rmse_m": round(itrf_rmse, 6),
            "eop_loaded": eop_ok,
            "missing_eop_documented_leo_error_m": 3.0,
            "leap_from_file": leap_from_file,
            "tai_utc_s": tai_utc,
            "leap_entry_count": int(leap.entry_count),
        }, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
