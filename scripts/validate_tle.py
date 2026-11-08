#!/usr/bin/env python3
"""TLE starts against precise orbits: how good is a prediction from a TLE?

A TLE (two-line element set) is the compact orbit description published for every
tracked object; it is meant to be propagated with SGP4, the standard analytical TLE
propagator. For each public element set of an object with a public precise orbit
(GRACE-FO 1 and 2: Rapid Science Orbits from the German Research Centre for
Geosciences, GFZ, anonymous HTTPS at isdc-data.gfz.de), the positions of four
predictions are compared with the precise orbit at every ``--step`` seconds of its
coverage, within ``--max-horizon-days`` of the TLE epoch:

  sgp4         SGP4 itself (the model the TLE was fitted for)
  full         numerical, from State.from_tle, recommended_config(state, name)
  preset       numerical, from State.from_tle, recommended_config(..., accuracy="tle")
  fit+preset   numerical, from ballistic.tle.fit_state (fitted to SGP4 over the
               day after the epoch, its default span), the TLE-grade preset

CelesTrak publishes only the CURRENT element set of each object and the
rapid orbits arrive a few days late, so a run today compares the newest TLE
with the orbit of the days BEFORE its epoch (negative horizons; SGP4 and the
numerical propagator run backwards). Forward horizons need an archive:
``--fetch-only`` (run daily, e.g. from cron) stores each new element set under
``--cache`` with its download time and the recent rapid orbits beside it; once
the rapid orbits cover the days after an archived set, ``--tle <old set>
--forward-only`` scores only the samples after its epoch. No credentials:
CelesTrak GP and GFZ ISDC are public.

Inputs: --fetch downloads the current CelesTrak element sets and the last --days days
of GFZ rapid orbits (into --cache, default data/tle_validation/); --tle / --sp3 name
kept files (TLE text files; SP3 precise-orbit files, SP3 being the standard text
format for precise orbits, plain or .gz).
Output: per TLE, a table of the 3D root-mean-square (RMS) position error in meters of
each method per horizon bin (--bin-hours wide). Horizons are signed hours from the
TLE epoch: negative = before the epoch, positive = after; each row is labeled
"before" or "after" (bins are aligned on the epoch, so none straddles it). --json
writes every sample's signed horizon, 3D and along-track errors, and the binned RMS
with the same labels.

Options: --fetch, --fetch-only, --days N (default 8; rapid orbits of today and the
N days before), --cache DIR, --tle FILE ..., --sp3 FILE ..., --forward-only,
--step SECONDS (default 600), --bin-hours H (default 12), --max-horizon-days D
(default 7), --json OUT.json.

  --fetch        download, then score the downloaded sets with the given files
  --fetch-only   download only (for a daily cron job): an element set already in the
                 archive and an orbit file already present are not stored again; a
                 rapid orbit not yet published is reported, not an error; one line
                 per run is appended to <cache>/fetch.log. Exit status 2 when the
                 CelesTrak download fails (network), 0 otherwise.
  --forward-only score only truth samples after the TLE epoch (forward prediction)

    PYTHONPATH=./python uv run --with numpy python scripts/validate_tle.py --fetch --days 8
    PYTHONPATH=./python uv run --with numpy python scripts/validate_tle.py --fetch-only --days 2
    PYTHONPATH=./python uv run --with numpy python scripts/validate_tle.py \
        --tle data/tle_validation/tle/*.txt --sp3 data/tle_validation/rso/*.sp3.gz
    PYTHONPATH=./python uv run --with numpy python scripts/validate_tle.py --forward-only \
        --tle data/tle_validation/tle/<old set>.txt --sp3 data/tle_validation/rso/*.sp3.gz
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

import ballistic as b

REPO = Path(__file__).resolve().parents[1]
# NORAD catalog number -> (registry name, GFZ ISDC level-2 product directory)
TARGETS = {43476: ("GRACE-FO-1", "L64"), 43477: ("GRACE-FO-2", "L65")}
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?CATNR={cat}&FORMAT=TLE"
ISDC = "https://isdc-data.gfz.de/grace-fo/ORBIT/{level}/RSO/{year}/{doy:03d}/"
METHODS = ("sgp4", "full", "preset", "fit+preset")


def _get(url, timeout=60):
    """Return the bytes at url (timeout in seconds); raises on any HTTP or network error."""
    with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 (fixed public hosts)
        return r.read()


class TleFetchError(RuntimeError):
    """The CelesTrak element-set download failed (network or HTTP error)."""


def _tle_lines(text, rejected=None):
    """Return the set of (line 1, line 2) pairs that parse as TLEs in text (may be empty).

    CelesTrak answers a too-frequent request with a plain-text notice instead of
    element sets; such text yields no pairs. A pair that looks like a TLE but fails
    the strict parser (checksum, columns) is left out and, when rejected is a list,
    its error message is appended there so the caller can report it.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out = set()
    for a, c in zip(lines, lines[1:]):
        if a.startswith("1 ") and c.startswith("2 "):
            try:
                b.tle.parse(a, c)
            except Exception as e:   # noqa: BLE001 - reported by the caller via rejected
                if rejected is not None:
                    rejected.append(f"{a[:18]}...: {e}")
                continue
            out.add((a, c))
    return out


def _archived_tle_lines(tle_dir: Path, cat: int):
    """Every (line 1, line 2) pair already archived for catalog number cat."""
    seen = set()
    for f in sorted(tle_dir.glob(f"*_{cat}.txt")):
        seen |= _tle_lines(f.read_text(errors="replace"))
    return seen


def _write_atomic(path: Path, data: bytes):
    """Write data to path through a temporary file, so an interrupted run leaves no
    truncated file that a later run would take as complete."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def fetch(cache: Path, days: int, log=print):
    """Download the current TLEs and the rapid orbits of today and the `days` days before.

    cache: directory; TLEs go to cache/tle/<UTC time stamp>_<catalog number>.txt and
    orbits to cache/rso/. Idempotent: an element set already archived (same two lines)
    is not stored again and an orbit file already there is not downloaded again. A day
    whose directory listing fails (usually: not published yet, the rapid orbits lag a
    few days) or a file whose download fails is reported through log and skipped.
    Returns a dict: tles / sp3s (every TLE and SP3 path of this run, new or already
    present), new_tles / new_sp3s (files written by this run), missing_days (listing
    URLs that failed) and tle_errors (catalog number -> message for a failed or
    unparseable CelesTrak answer; a download error raises TleFetchError only after
    the rapid orbits were fetched, see main).
    """
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%MZ")
    (cache / "tle").mkdir(parents=True, exist_ok=True)
    (cache / "rso").mkdir(parents=True, exist_ok=True)
    rep = {"tles": [], "sp3s": [], "new_tles": [], "new_sp3s": [], "missing_days": [],
           "tle_errors": {}, "tle_network_error": False}
    today = now.date()
    for cat, (name, level) in TARGETS.items():
        try:
            text = _get(CELESTRAK.format(cat=cat)).decode(errors="replace")
        except Exception as e:      # noqa: BLE001 - recorded; main exits non-zero
            rep["tle_errors"][cat] = f"download failed: {e}"
            rep["tle_network_error"] = True
            log(f"  {name}: CelesTrak download failed: {e}")
            text = None
        if text is not None:
            rejected = []
            pairs = _tle_lines(text, rejected)
            for msg in rejected:
                log(f"  {name}: CelesTrak line pair rejected by the TLE parser: {msg}")
            if not pairs:
                msg = " ".join(text.split())[:160]
                rep["tle_errors"][cat] = f"no element set in the answer: {msg!r}"
                log(f"  {name}: CelesTrak returned no element set ({msg!r})")
            else:
                known = _archived_tle_lines(cache / "tle", cat)
                if pairs <= known:
                    match = [f for f in sorted((cache / "tle").glob(f"*_{cat}.txt"))
                             if _tle_lines(f.read_text(errors="replace")) & pairs]
                    rep["tles"].append(match[-1])
                else:
                    path = cache / "tle" / f"{stamp}_{cat}.txt"
                    _write_atomic(path, text.encode())
                    rep["tles"].append(path)
                    rep["new_tles"].append(path)
        for k in range(days, -1, -1):
            d = today - dt.timedelta(days=k)
            url = ISDC.format(level=level, year=d.year, doy=d.timetuple().tm_yday)
            try:
                listing = _get(url).decode()
            except Exception as e:      # noqa: BLE001 - a day not published yet: say so, keep going
                rep["missing_days"].append(url)
                log(f"  {name} {d.isoformat()}: no rapid orbit listing yet ({e})")
                continue
            for fname in sorted(set(re.findall(r'href="(GFZOP_RSO_[^"]+\.sp3\.gz)"', listing))):
                out = cache / "rso" / fname
                if not out.exists():
                    try:
                        _write_atomic(out, _get(url + fname))
                    except Exception as e:  # noqa: BLE001 - retried on the next run
                        log(f"  {url + fname}: download failed ({e}); the next run retries")
                        continue
                    rep["new_sp3s"].append(out)
                rep["sp3s"].append(out)
    return rep


def append_fetch_log(cache: Path, rep):
    """Append one line describing a fetch run to cache/fetch.log (UTC stamp, files added)."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    added = [p.name for p in rep["new_tles"] + rep["new_sp3s"]]
    line = (f"{stamp} added {len(rep['new_tles'])} tle, {len(rep['new_sp3s'])} rso: "
            f"{' '.join(added) if added else '-'}; rso days not yet listed "
            f"{len(rep['missing_days'])}")
    if rep["tle_errors"]:
        line += "; tle errors: " + "; ".join(f"{c} {m}" for c, m in rep["tle_errors"].items())
    with open(cache / "fetch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


def keep_horizon(offset_s, max_horizon_s, forward_only=False):
    """True when a truth sample offset_s seconds from the TLE epoch is scored.

    Within max_horizon_s of the epoch on either side; with forward_only, only
    samples strictly after the epoch (offset_s > 0).
    """
    if abs(offset_s) > max_horizon_s:
        return False
    return offset_s > 0 if forward_only else True


def bin_edges(hours, bin_hours):
    """Horizon bin edges [h] covering hours, at integer multiples of bin_hours.

    Multiples of the width put an edge on the epoch (0), so no bin mixes
    predictions before and after it.
    """
    hours = np.asarray(hours, dtype=float)
    k0 = int(np.floor(hours.min() / bin_hours))
    k1 = int(np.floor(hours.max() / bin_hours)) + 1
    return np.arange(k0, k1 + 1) * bin_hours


def bin_label(lo, hi):
    """Label of the horizon bin [lo, hi) hours: signed bounds and the side of the epoch."""
    side = "before" if hi <= 0 else "after"
    return f"{lo:+.1f}..{hi:+.1f} h {side}"


def binned_rms(hours, errors, bin_hours):
    """Bin the per-sample 3D errors by signed horizon.

    hours: (N,) signed horizons [h]; errors: {method: {"err3d_m": (N,)}}.
    Returns a list of dicts {label, lo_h, hi_h, side, n, rms_m: {method: RMS}},
    empty bins omitted, in time order.
    """
    hours = np.asarray(hours, dtype=float)
    edges = bin_edges(hours, bin_hours)
    out = []
    for a, c in zip(edges[:-1], edges[1:]):
        sel = (hours >= a) & (hours < c)
        if not sel.any():
            continue
        out.append({"label": bin_label(a, c), "lo_h": float(a), "hi_h": float(c),
                    "side": "before" if c <= 0 else "after", "n": int(sel.sum()),
                    "rms_m": {m: float(np.sqrt(np.mean(np.asarray(e["err3d_m"])[sel] ** 2)))
                              for m, e in errors.items()}})
    return out


def truth_samples(sp3_paths, step_s):
    """Read the SP3 files and return {SP3 satellite id: [sample, ...]}, about every step_s.

    Each sample is a tuple (Epoch, position (3,) meters Earth-fixed, velocity (3,) m/s
    or None when the file has no velocities, frame label of the file). Samples are
    sorted by time, and where files overlap only one sample per epoch is kept (epochs
    within 1 s count as the same).
    """
    out = {}
    for p in sp3_paths:
        f = b.sp3.read(p)
        for arc in f.arcs:
            offs = np.asarray(arc.offsets_s)
            pos = np.asarray(arc.position)
            vel = np.asarray(arc.velocity) if arc.has_velocity() else None
            last = -1e18
            for i, t in enumerate(offs):
                if t - last < step_s - 1e-6:
                    continue
                last = t
                out.setdefault(arc.satellite, []).append(
                    (arc.epoch(i), pos[i], None if vel is None else vel[i], f.header.frame))
    for k in out:
        ref = out[k][0][0]
        out[k].sort(key=lambda r: r[0] - ref)
        dedup, prev = [], None   # overlapping files: one sample per epoch
        for r in out[k]:
            if prev is None or abs(r[0] - prev) > 1.0:
                dedup.append(r)
                prev = r[0]
        out[k] = dedup
    return out


def _itrf_of(frame):
    """Return the ITRF realization (Earth-fixed frame version) for an SP3 frame label.

    Raises ballistic.FrameError when the label names no realization.
    """
    info = b.sp3.classify_frame(frame)
    if info.itrf is not None:
        return info.itrf
    if frame == "CTS":
        # GFZ RSO v03 headers say only "CTS"; GFZ processes them in ITRF2020
        # (the ITRF table in docs/CONCEPTS.md). Millimeters; the errors here
        # are km.
        return b.ITRFRealization.ITRF2020
    raise b.FrameError(f"rapid orbit frame {frame!r}: no ITRF realization")


def evaluate(tle, name, samples, max_horizon_s, forward_only=False):
    """Score the four predictions of one TLE against the truth samples.

    tle: a parsed ballistic TLE; name: satellite registry name (e.g. "GRACE-FO-1");
    samples: that satellite's list from truth_samples; max_horizon_s: only samples
    within this many seconds of the TLE epoch are used (the TLE-grade preset was
    measured up to 7 days only); forward_only: only samples after the epoch.
    Returns None when no sample is that close, otherwise a dict with catalog, epoch,
    hours (numpy array of sample times relative to the TLE epoch), fit_rms_m (RMS of
    the state fit to SGP4, meters), preset (the TLE-grade preset chosen) and errors
    ({method: {"err3d_m": 3D error, "along_m": along-track error}}, meters per sample;
    along-track is NaN where the truth has no velocity).
    """
    t0 = tle.epoch
    samples = [s for s in samples if keep_horizon(s[0] - t0, max_horizon_s, forward_only)]
    if not samples:
        return None
    times = [s[0] for s in samples]
    horizon = max(abs(t - t0) for t in times)
    state = b.State.from_tle(tle)
    itrf = _itrf_of(samples[0][3])
    full_cfg = b.recommended_config(state, name, itrf=itrf)
    preset_cfg = b.recommended_config(state, name, accuracy="tle", horizon_s=horizon, itrf=itrf)
    fitted, fit_rms = b.tle.fit_state(tle, preset_cfg)
    model = b.tle.Sgp4(tle)

    span = max(abs(t - t0) for t in times) + 3600.0

    def numerical(start, cfg):
        # Forward runs record their steps; backward times are reached by the
        # interpolator propagating back from the start (allowed over the span).
        tr = b.Trajectory(start, cfg, b.InterpolatorConfig(max_propagation_time=span))
        hi = max(times, key=lambda t: t - t0)
        if hi - t0 > 0:
            tr.propagate_to(hi)
        return np.array([np.asarray(s.position) for s in tr(times, frame="ECEF", on_out_of_range="raise")])

    pred = {
        "sgp4": np.array([np.asarray(b.tle.teme_to_itrf(t, *model.teme(t))[0]) for t in times]),
        "full": numerical(state, full_cfg),
        "preset": numerical(state, preset_cfg),
        "fit+preset": numerical(fitted, preset_cfg),
    }
    truth = np.array([s[1] for s in samples])
    vel = np.array([s[2] if s[2] is not None else np.full(3, np.nan) for s in samples])
    along = vel / np.linalg.norm(vel, axis=1, keepdims=True)
    hours = np.array([(t - t0) / 3600.0 for t in times])
    res = {}
    for m, p in pred.items():
        d = p - truth
        res[m] = {"err3d_m": np.linalg.norm(d, axis=1), "along_m": np.sum(d * along, axis=1)}
    return {"catalog": tle.catalog_number, "epoch": str(t0), "hours": hours, "fit_rms_m": fit_rms,
            "preset": b.tle.tle_grade_preset(state, horizon), "errors": res}


def build_parser():
    """Return the argparse parser for the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fetch", action="store_true",
                   help="download the current TLEs and the recent rapid orbits into --cache")
    p.add_argument("--fetch-only", action="store_true",
                   help="download into --cache and stop (no scoring; for a daily cron job); "
                        "exit status 2 if the CelesTrak download fails")
    p.add_argument("--days", type=int, default=8,
                   help="fetch rapid orbits for today and the N days before (default 8)")
    p.add_argument("--cache", default=str(REPO / "data" / "tle_validation"))
    p.add_argument("--tle", nargs="*", default=[], help="2LE/3LE text files")
    p.add_argument("--sp3", nargs="*", default=[], help="precise orbit files (.sp3 / .sp3.gz)")
    p.add_argument("--forward-only", action="store_true",
                   help="score only truth samples after the TLE epoch (forward prediction)")
    p.add_argument("--step", type=float, default=600.0, help="truth sampling [s] (default 600)")
    p.add_argument("--bin-hours", type=float, default=12.0, help="horizon bin width [h]")
    p.add_argument("--max-horizon-days", type=float, default=7.0,
                   help="score truth within this many days of each TLE epoch (default 7, "
                        "the longest horizon the TLE-grade preset was measured for)")
    p.add_argument("--json", default=None, help="write the per-sample errors as JSON")
    return p


def _fetch_only(args):
    """Run the archive download alone; return the exit status (2 = TLE download failed)."""
    cache = Path(args.cache)
    rep = fetch(cache, args.days, log=lambda m: print(m, file=sys.stderr))
    line = append_fetch_log(cache, rep)
    print(f"validate_tle --fetch-only: {line}")
    if rep["missing_days"]:
        print(f"  {len(rep['missing_days'])} satellite-days of rapid orbits are not published "
              "yet (GFZ lags a few days); a later run picks them up")
    if rep["tle_network_error"]:
        print("  CelesTrak element-set download FAILED (network); exit status 2", file=sys.stderr)
        return 2
    return 0


def main():
    """Load the library data, gather TLEs and orbits, score each TLE and print the tables."""
    args = build_parser().parse_args()
    if args.fetch_only:
        raise SystemExit(_fetch_only(args))
    b.load(offline=False, verbose=False)
    tles, sp3s = [Path(t) for t in args.tle], [Path(s) for s in args.sp3]
    if args.fetch:
        rep = fetch(Path(args.cache), args.days, log=lambda m: print(m, file=sys.stderr))
        append_fetch_log(Path(args.cache), rep)
        tles += rep["tles"]
        sp3s += rep["sp3s"]
    if not tles or not sp3s:
        raise SystemExit("validate_tle: give --tle and --sp3 files, or --fetch")
    truth = truth_samples(sorted(set(sp3s)), args.step)
    level_sat = {}
    for s in sp3s:   # GFZOP_RSO_L64_... holds one satellite
        m = re.search(r"_RSO_(L6\d)_", s.name)
        if m:
            level_sat.setdefault(m.group(1), set()).update(b.sp3.read(s).header.satellites)
    results = []
    for path in tles:
        for _, tle in b.tle.read(path):
            if tle.catalog_number not in TARGETS:
                continue
            name, level = TARGETS[tle.catalog_number]
            sats = sorted(level_sat.get(level, ()))
            if len(sats) != 1 or sats[0] not in truth:
                print(f"{name}: no rapid orbit loaded", file=sys.stderr)
                continue
            r = evaluate(tle, name, truth[sats[0]], args.max_horizon_days * 86400.0,
                         forward_only=args.forward_only)
            if r is None:
                if args.forward_only:
                    last = max(truth[sats[0]], key=lambda x: x[0] - tle.epoch)[0]
                    print(f"{name} TLE epoch {tle.epoch}: no forward samples yet (the loaded "
                          f"precise orbit ends at {last}; fetch the rapid orbits of the days "
                          f"after the epoch and re-run)", file=sys.stderr)
                else:
                    print(f"{name} TLE epoch {tle.epoch}: no precise orbit within "
                          f"{args.max_horizon_days:g} days", file=sys.stderr)
                continue
            r["bins"] = binned_rms(r["hours"], r["errors"], args.bin_hours)
            results.append(r)
            p = r["preset"]
            print(f"\n{name} TLE epoch {r['epoch']}  fit rms {r['fit_rms_m']:.0f} m  preset: degree "
                  f"{p.gravity_degree}, drag {p.drag}, radiation {p.radiation}, third body "
                  f"{p.third_body}, tides {p.tides}, relativity {p.relativity}")
            print(f"  {'horizon from epoch':>24s} " + " ".join(f"{m:>12s}" for m in METHODS)
                  + "   (3D RMS, m; - = before the epoch, + = after)")
            for row in r["bins"]:
                print(f"  {row['label']:>24s} "
                      + " ".join(f"{row['rms_m'][m]:12.0f}" for m in METHODS))
    if args.forward_only and not results:
        print("validate_tle: no forward samples yet for any TLE given")
    if args.json:
        Path(args.json).write_text(json.dumps([
            {"catalog": r["catalog"], "epoch": r["epoch"], "fit_rms_m": r["fit_rms_m"],
             "forward_only": args.forward_only, "bin_hours": args.bin_hours,
             "hours": r["hours"].tolist(),
             "bins": r["bins"],
             "errors": {m: {k: v.tolist() for k, v in e.items()} for m, e in r["errors"].items()}}
            for r in results], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
