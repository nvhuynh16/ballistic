"""
Tutorial example 01 — Setup & initialization (Python)

The Pythonic path is ``ballistic.Loader``: it downloads and caches EOP +
leap seconds (like Skyfield's ``load()``), then ``ballistic.initialize()``
auto-discovers them. Always inspect ``get_init_status()`` — EOP and a fresh
leap-second file are what separate meter from sub-meter work (Pitfalls ch).

    python ex01_setup.py [out.json]
        (default: docs/latex/results/ex01_init_status.json)
"""
import json
import sys

import ballistic


def main(out="docs/latex/results/ex01_init_status.json"):
    # Loader downloads/caches EOP + leap seconds on first use, then
    # initialize() picks them up via auto-discovery. (Offline-safe: if the
    # files are already cached or bundled this is a no-op.)
    try:
        ballistic.Loader().timescale()
    except Exception as e:  # pragma: no cover - network optional
        print(f"[loader skipped: {e}]")

    ballistic.initialize()
    r = ballistic.get_init_status()
    leap = ballistic.get_leap_second_info()
    data_dir = ballistic.get_data_directory()

    print(f"initialize success      : {r.success}")
    print(f"EOP loaded              : {r.eop_loaded} "
          f"(auto-discovered: {r.eop_auto_discovered})")
    print(f"leap seconds loaded     : {r.leap_seconds_loaded} "
          f"(auto-discovered: {r.leap_seconds_auto_discovered})")
    print(f"embedded ephemeris      : {r.embedded_ephemeris_loaded} "
          f"({r.jpl_de_version})")
    print(f"gravity degree available: {r.gravity_degree_loaded}")
    print(f"current TAI-UTC         : {leap.current_tai_utc} s")
    print(f"leap from file          : {leap.loaded_from_file}")
    print(f"data directory          : {data_dir}")

    status = {
        "success": bool(r.success),
        "eop_loaded": bool(r.eop_loaded),
        "eop_auto_discovered": bool(r.eop_auto_discovered),
        "leap_seconds_loaded": bool(r.leap_seconds_loaded),
        "leap_seconds_from_file": bool(leap.loaded_from_file),
        "current_tai_utc_s": int(leap.current_tai_utc),
        "leap_entry_count": int(leap.entry_count),
        "gravity_degree_loaded": int(r.gravity_degree_loaded),
        "jpl_de_version": r.jpl_de_version,
        "data_directory": str(data_dir),
    }
    with open(out, "w") as f:
        json.dump(status, f, indent=2)
    print(f"wrote {out}")
    return 0 if r.success else 1


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2])))
