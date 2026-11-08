"""
Tutorial example 02 — Epoch & time scales (Python)

One instant, six time scales, all expressed as offsets against TAI:

    TT  = TAI + 32.184 s        (exact)
    GPS = TAI - 19 s            (exact)
    UTC = TAI - (leap seconds)  (currently 37 s; integer)
    UT1 = UTC + DUT1            (Earth rotation; needs EOP, else DUT1=0)
    TDB ~ TT                    (periodic, sub-ms)

``Epoch`` stores TT internally; ``.tt_jd``/``.tai_jd``/... convert on
demand. ``Epoch - Epoch`` yields SECONDS.

PITFALL demonstrated: recovering a small offset by differencing two large
Julian Dates loses precision. At JD ~ 2.46e6 one double ulp is ~4.7e-5 s,
so TT-TAI prints 32.18401.. not exactly 32.18400. The offsets ARE exact by
definition; the microsecond wobble is float cancellation, not a bug. Never
derive sub-ms time offsets by subtracting raw JDs.

    python ex02_timescales.py [out.json]
"""
import json
import sys

import ballistic

SECONDS_PER_DAY = 86400.0


def main(out="docs/latex/results/ex02_timescale_offsets.json"):
    ballistic.initialize()

    e = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    tai = e.tai_jd
    off = {
        "tt_minus_tai_s": (e.tt_jd - tai) * SECONDS_PER_DAY,
        "gps_minus_tai_s": (e.gps_jd - tai) * SECONDS_PER_DAY,
        "utc_minus_tai_s": (e.utc_jd - tai) * SECONDS_PER_DAY,
        "ut1_minus_tai_s": (e.ut1_jd - tai) * SECONDS_PER_DAY,
        "tdb_minus_tai_s": (e.tdb_jd - tai) * SECONDS_PER_DAY,
    }

    print("Epoch (UTC 2024-01-15 12:00:00)")
    print(f"  TT  - TAI = {off['tt_minus_tai_s']:.6f} s  (expect 32.184000)")
    print(f"  GPS - TAI = {off['gps_minus_tai_s']:.6f} s  (expect -19.000000)")
    print(f"  UTC - TAI = {off['utc_minus_tai_s']:.6f} s  (= -(leap seconds))")
    print(f"  UT1 - TAI = {off['ut1_minus_tai_s']:.6f} s  (UTC + DUT1; needs EOP)")
    print(f"  TDB - TAI = {off['tdb_minus_tai_s']:.6f} s  (~ TT, periodic)")

    payload = {"epoch_utc": "2024-01-15T12:00:00", **off}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}")

    # JD-precision-aware tolerance (~5e-5 s ulp floor here; 1e-3 still
    # catches a genuine wrong offset while tolerating the documented
    # float-cancellation wobble).
    JD_DIFF_TOL_S = 1e-3
    assert abs(off["tt_minus_tai_s"] - 32.184) < JD_DIFF_TOL_S, off
    assert abs(off["gps_minus_tai_s"] + 19.0) < JD_DIFF_TOL_S, off
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2])))
