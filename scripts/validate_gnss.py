#!/usr/bin/env python3
"""
GNSS 1-hour prediction campaign: GPS (MEO), QZSS (IGSO) and BeiDou (GEO).

GNSS = global navigation satellite systems. The orbit classes: MEO = medium Earth orbit
(GPS); IGSO = inclined geosynchronous orbit (one revolution per
sidereal day, on an inclined plane); GEO = geostationary orbit (equatorial, fixed over
one longitude). The result is claim GNSS-1H in docs/validation_claims.md.

Same metric as the other validation rows: start from a precise-orbit state,
propagate, and compare with the precise orbit at the end. Every satellite in
the files, a start every --sample-interval minutes, error (meters, 3D distance)
--duration seconds later; the summary gives the root-mean-square error (RMSE),
median and maximum per constellation and orbit class.

Truth: final orbits in SP3 files (SP3 = the standard text format for precise
orbits) in the IGS20 frame (the International GNSS Service's realization of the
Earth-fixed frame, ITRF), 5-minute positions without velocities: CODE (Center for
Orbit Determination in Europe) COD0OPSFIN for GPS, CODE COD0MGXFIN for QZSS, and
Wuhan University WUM0MGXFIN for the BeiDou GEOs (CODE's product for the Multi-GNSS
Experiment, MGEX, omits all geostationary satellites, and no product in the repo
has QZSS's GEO, J07). Start velocities are derived from the positions with 9-point
Lagrange differentiation; that is the true time derivative of the Earth-fixed
position, so no correction for the slow drift of the Earth's spin axis (the
celestial intermediate pole, CIP) applies. SP3 epochs are read in the file's own
time system (ballistic.sp3): CODE writes GPS time. Starts use only epochs whose
derivative window is centered (not the first and last four epochs of a file).

QZSS and BeiDou satellites are labeled by orbit, not by a fixed list:
inclination below 5 degrees is GEO, otherwise IGSO (or MEO for BeiDou below
30,000 km). For QZSS, IGSO is the elliptical, inclined geosynchronous
"quasi-zenith" orbit (e ~ 0.075); SP3 J03 is QZS-4, an IGSO satellite.

The force model is the library default (EGM2008 gravity field to degree 70,
Sun/Moon/Jupiter, solid Earth tides, relativity, RK45 integrator). Solar radiation
pressure (SRP), the dominant remaining force at these altitudes (~1e-7 m/s^2,
~0.5 m in an hour), is off by default (claims GNSS-1H, GNSS-24H, GNSS-72H).
--srp cannonball turns it on per satellite: the satellite's block and in-orbit
mass come from the IGS satellite metadata file (scripts/_gnss_metadata.py, for
the start epoch), the block's cannonball area and Cr from the library's block
table (include/ballistic/gnss_blocks.hpp, ballistic.satellite_properties
.gnss_block), area = base area x the block's fitted scale k x --srp-scale
(claims GNSS-1H-SRP, GNSS-24H-SRP; docs/gnss_srp.md).

--fit-srp fits k per block: for each k on a grid (0.50 to 1.60, step 0.05) the
one-hour campaign runs on the odd days of the committed files (2024 days 015
and 017) and each block takes the k with the smallest RMSE over its
satellites; the fitted k are then scored on the even days (016 and 018),
next to SRP off on the same starts. --fit-json writes the result.

Inputs: the committed SP3 files (GPS: the CODE COD0OPSFIN file of 2024 day 015
only; QZSS and BeiDou: all four committed days), or --sp3.
Output: per-satellite and summary lines on stdout; --csv writes every sample.

Usage:
    PYTHONPATH=./python uv run --with numpy python scripts/validate_gnss.py \\
        [--constellation {gps,qzss,beidou-geo,all}] (default all)
        [--sp3 FILE [FILE ...]]   (replaces the default files; one constellation only)
        [--sample-interval MINUTES] (default 60) [--duration SECONDS] (default 3600;
                                   above 3600 the daily files are joined, so a
                                   horizon may cross midnight, up to the span
                                   of all files)
        [--csv OUT.csv]
        [--srp {off,cannonball}]  (default off)
        [--metadata PATH]         (IGS satellite metadata SINEX, for --srp cannonball
                                   and --fit-srp; default
                                   data/igs/igs_satellite_metadata.snx, fetched by
                                   python scripts/download_igs_metadata.py)
        [--srp-scale K]           (default 1.0; multiplies every block's area,
                                   for sensitivity runs)
        [--fit-srp]               (fit k per block on the odd days, score on the
                                   even days; uses the committed four days, not --sp3)
        [--fit-json OUT.json]     (with --fit-srp: write the fitted k and the RMSEs)
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.normpath(os.path.join(_script_dir, '..'))
_python_dir = os.path.join(_repo, 'python')
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np

import _sp3_records as _sp3r  # noqa: E402

DEFAULT_FILES = {
    'gps': [os.path.join(_repo, 'tests/data/sp3/gps/COD0OPSFIN_20240150000_01D_05M_ORB.SP3')],
    'qzss': sorted(glob.glob(os.path.join(_repo, 'tests/data/sp3/qzss/COD0MGXFIN_*_ORB.SP3'))),
    'beidou-geo': sorted(glob.glob(os.path.join(_repo, 'tests/data/sp3/beidou/WUM0MGXFIN_*_ORB.SP3.gz'))),
}
# One file per day of the committed four days (2024 days 015-018), for --fit-srp.
DAY_PATTERNS = {
    'gps': 'tests/data/sp3/gps/COD0OPSFIN_2024{doy}0000_01D_05M_ORB.SP3*',
    'qzss': 'tests/data/sp3/qzss/COD0MGXFIN_2024{doy}0000_01D_05M_ORB.SP3*',
    'beidou-geo': 'tests/data/sp3/beidou/WUM0MGXFIN_2024{doy}0000_01D_05M_ORB.SP3*',
}
FIT_DAYS = ('015', '017')      # odd days: k is chosen here
SCORE_DAYS = ('016', '018')    # even days: k is scored here
K_GRID = tuple(round(0.50 + 0.05 * i, 2) for i in range(23))   # 0.50 ... 1.60
DEFAULT_METADATA = os.path.join(_repo, 'data/igs/igs_satellite_metadata.snx')
PREFIX = {'gps': 'G', 'qzss': 'J', 'beidou-geo': 'C'}
ONLY_CLASS = {'beidou-geo': 'GEO'}


def load(ballistic, paths, prefix, only=None, join=False):
    """Read the SP3 files and return {satellite id: [arc, ...]}, one arc per file.

    ballistic: the imported module; paths: SP3 files; prefix: the constellation letter
    of the ids to keep ('G' GPS, 'J' QZSS, 'C' BeiDou); only: optional collection of
    satellite ids to keep. Arcs are the dicts of scripts/_sp3_records.load_arcs, with
    velocities derived from the positions; satellites whose first arc has fewer than 9
    records (too few to differentiate) are dropped.

    Each daily file is an independent orbit solution, with a small jump
    where two days meet. Velocities are derived within one file and every
    prediction window stays inside one file, so neither straddles a jump.
    Epochs honor each file's time system (ballistic.sp3).

    join=True joins each satellite's daily files into one arc
    (_sp3_records.join_arcs; velocities are still derived within each file),
    so a prediction can end in a later file than it starts in. The jump
    between two daily solutions (~1 cm) then lies inside the window.
    """
    data = _sp3r.load_arcs(ballistic, paths, sorted(only) if only else None,
                           velocity='derived', quiet=True, join=join)
    return {sat: arcs for sat, arcs in data.items()
            if sat.startswith(prefix) and len(arcs[0]['records']) >= 9}


def orbit_class(constellation, rec):
    """Classify a satellite as 'MEO', 'IGSO' or 'GEO' from one record.

    constellation: 'gps', 'qzss' or 'beidou-geo'; rec: a record dict with 'pos_m' and
    'vel_ms' (Earth-fixed, meters and m/s). GPS, and anything closer than 30,000 km to
    the Earth's center, is MEO; otherwise inclination below 5 degrees is GEO, else IGSO.
    """
    if constellation == 'gps' or np.linalg.norm(rec['pos_m']) < 3.0e7:
        return 'MEO'
    # Inclination from the inertial velocity: the Earth-fixed velocity of a
    # GEO satellite is ~0, so add Earth's rotation back first.
    r = rec['pos_m']
    import ballistic
    v = rec['vel_ms'] + np.cross([0.0, 0.0, ballistic.constants.earth.ANGULAR_VELOCITY], r)
    h = np.cross(r, v)
    incl = np.degrees(np.arccos(abs(h[2]) / np.linalg.norm(h)))
    return 'GEO' if incl < 5.0 else 'IGSO'


def day_files(constellation, days):
    """The committed SP3 file of each day in ``days`` ('015' ...), one per day.

    Where a day exists both plain and gzip-compressed, the plain file is used.
    Raises FileNotFoundError naming the missing day.
    """
    out = []
    for doy in days:
        found = sorted(glob.glob(os.path.join(_repo, DAY_PATTERNS[constellation].format(doy=doy))))
        if not found:
            raise FileNotFoundError(f"no committed {constellation} SP3 file for 2024 day {doy} "
                                    f"({DAY_PATTERNS[constellation].format(doy=doy)})")
        plain = [f for f in found if not f.endswith('.gz')]
        out.append((plain or found)[0])
    return out


def srp_setup(md, sat_id, epoch, block_k=None, srp_scale=1.0):
    """The radiation-pressure inputs for one satellite at one epoch.

    md: a _gnss_metadata.GnssMetadata; sat_id: SP3 id ('G01'); epoch: a
    ballistic.Epoch; block_k: optional {block row name: k} replacing the table's
    fitted k (used while fitting); srp_scale: extra factor on the area.
    Returns (block row name, SrpConfig).
    """
    import ballistic
    from ballistic import satellite_properties as sp
    svn = md.svn_of(sat_id, epoch)
    blk = sp.gnss_block(md.block_of(svn))
    k = blk.srp_k if block_k is None else block_k[blk.name]
    cfg = ballistic.SrpConfig(mass_kg=md.mass_of(svn, epoch),
                              area_m2=blk.srp_base_area_m2 * k * srp_scale, cr=blk.cr)
    return blk.name, cfg


def run(constellation, paths, sample_interval_min=60, duration_s=3600.0,
        sats_filter=None, max_starts=None, verbose=True, initialize=True, join=None,
        *, srp='off', metadata=None, srp_scale=1.0, block_k=None):
    """Run the campaign for one constellation and return one dict per start.

    constellation: 'gps', 'qzss' or 'beidou-geo'; paths: SP3 files;
    sample_interval_min: minutes between starts; duration_s: prediction length in
    seconds; sats_filter: optional collection of satellite ids; max_starts: optional
    cap on starts per satellite; verbose: print one line per satellite.
    Each dict has keys constellation, class, sat, epoch (label of the start) and
    error_m (3D position error in meters at the end of the prediction).

    initialize=False leaves library setup (EOP etc.) to the caller, as the
    pytest session fixture does.

    join: True joins the daily files into one arc per satellite (see load), so
    horizons up to the span of all files work; False keeps one arc per file.
    None (default) joins only for horizons longer than one hour: the one-hour
    campaign (GNSS-1H) scores every window inside one daily solution, and
    joining would move the truth of the 23:00 starts into the next day's file.

    srp: 'off' (default) or 'cannonball' (per-satellite block and mass from the
    IGS metadata, area from the block table; see srp_setup). metadata: a
    GnssMetadata or a path to the SINEX file (required for 'cannonball');
    srp_scale and block_k: see srp_setup. Each row then also carries 'block'.
    """
    import ballistic

    if initialize:
        ballistic.initialize()
        if not ballistic.Loader().load_eop():
            print("WARNING: no EOP data loaded")

    if join is None:
        join = duration_s > 3600.0
    data = load(ballistic, paths, PREFIX[constellation], only=sats_filter, join=join)
    cfg = ballistic.TrajectoryConfig(itrf=ballistic.ITRFRealization.ITRF2020)
    if srp not in ('off', 'cannonball'):
        raise ValueError(f"srp must be 'off' or 'cannonball', not {srp!r}")
    md = None
    if srp == 'cannonball':
        if metadata is None:
            raise ValueError("srp='cannonball' needs metadata (the IGS satellite metadata file)")
        import _gnss_metadata
        md = metadata if hasattr(metadata, 'svn_of') else _gnss_metadata.read_sinex(metadata)

    rows = []
    for sat_id in sorted(data):
        cls = orbit_class(constellation, data[sat_id][0]['records'][0])
        if ONLY_CLASS.get(constellation, cls) != cls:
            continue
        n = 0
        # Starts on derived velocities with a CENTERED window only: the first
        # and last four epochs of a file carry larger one-sided errors
        # (00:00 is a file edge, so it is not a start).
        for s in _sp3r.starts(data[sat_id], sample_interval_min * 60.0, duration_s,
                              skip_edges=True, start_tol=1.0, end_tol=1.0):
            if max_starts is not None and n >= max_starts:
                break
            r0, r1 = s['start'], s['end']
            block, run_cfg = None, cfg
            if md is not None:
                block, srp_cfg = srp_setup(md, sat_id, s['epoch'], block_k, srp_scale)
                run_cfg = cfg.replace(srp=srp_cfg)
            traj = ballistic.Trajectory(ballistic.State.from_itrf(s['epoch'], r0['pos_m'], r0['vel_ms']),
                                        config=run_cfg)
            traj.propagate_for(s['duration'])
            err = float(np.linalg.norm(traj.state_ecef().position - r1['pos_m']))
            rows.append({'constellation': constellation, 'class': cls, 'sat': sat_id,
                         'epoch': s['label'], 'error_m': err, 'block': block})
            n += 1
        if verbose:
            e = [r['error_m'] for r in rows if r['sat'] == sat_id]
            if e:
                print(f"  {sat_id} ({cls}): n={len(e):3d}  RMSE {np.sqrt(np.mean(np.square(e))):.3f} m"
                      f"  max {max(e):.3f} m")
    return rows


def summarize(rows):
    """Print a table per (constellation, orbit class) and return {group: RMSE in meters}."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['constellation'], r['class'])].append(r['error_m'])
    print("\n" + "=" * 66)
    print(f"{'Group':<14}{'sats':>5}{'starts':>8}{'RMSE':>10}{'median':>10}{'max':>10}")
    print("-" * 66)
    for (con, cls), e in sorted(groups.items()):
        nsat = len({r['sat'] for r in rows if r['constellation'] == con and r['class'] == cls})
        e = np.asarray(e)
        print(f"{con + ' ' + cls:<14}{nsat:>5}{len(e):>8}{np.sqrt(np.mean(e**2)):>9.3f}m"
              f"{np.median(e):>9.3f}m{e.max():>9.3f}m")
    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in groups.items()}


def _rmse(errors):
    """Root-mean-square of a sequence of errors (NaN when empty)."""
    e = np.asarray(list(errors), dtype=float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float('nan')


def fit_srp(constellation, metadata, sample_interval_min=60, initialize=True, verbose=True):
    """Fit the radiation-area scale k per block for one constellation.

    Runs the one-hour campaign on the odd days (FIT_DAYS) once per k in K_GRID
    with every block at that k (each satellite depends only on its own block's
    k, so one run per k serves all blocks), picks per block the k with the
    smallest RMSE, then scores the fitted k and SRP off on the even days
    (SCORE_DAYS). Returns {'constellation', 'blocks': {block: {'k', 'n_sats',
    'fit_rmse_m', 'score_rmse_m', 'score_off_rmse_m', 'fit_starts',
    'score_starts'}}, 'score_rmse_m', 'score_off_rmse_m', 'score_starts', 'grid'}.
    """
    import ballistic
    import _gnss_metadata
    md = metadata if hasattr(metadata, 'svn_of') else _gnss_metadata.read_sinex(metadata)
    if initialize:
        ballistic.initialize()
        ballistic.Loader().load_eop()
    fit_paths, score_paths = day_files(constellation, FIT_DAYS), day_files(constellation, SCORE_DAYS)
    by_k = {}
    for k in K_GRID:
        rows = run(constellation, fit_paths, sample_interval_min, verbose=False, initialize=False,
                   srp='cannonball', metadata=md, block_k=defaultdict(lambda k=k: k))
        by_k[k] = rows
        if verbose:
            print(f"  k {k:.2f}: odd-day RMSE {_rmse(r['error_m'] for r in rows):.3f} m")
    blocks = sorted({r['block'] for r in by_k[K_GRID[0]]})
    out = {'constellation': constellation, 'blocks': {}, 'grid': list(K_GRID)}
    for b in blocks:
        curve = {k: _rmse(r['error_m'] for r in by_k[k] if r['block'] == b) for k in K_GRID}
        kbest = min(curve, key=curve.get)
        out['blocks'][b] = {'k': kbest, 'fit_rmse_m': curve[kbest],
                            'fit_starts': sum(r['block'] == b for r in by_k[kbest]),
                            'n_sats': len({r['sat'] for r in by_k[kbest] if r['block'] == b}),
                            'fit_curve': curve}
        if kbest in (K_GRID[0], K_GRID[-1]):
            print(f"  WARNING: block {b}: the best k ({kbest}) is on the edge of the grid")
    fitted = {b: v['k'] for b, v in out['blocks'].items()}
    on = run(constellation, score_paths, sample_interval_min, verbose=False, initialize=False,
             srp='cannonball', metadata=md, block_k=fitted)
    off = run(constellation, score_paths, sample_interval_min, verbose=False, initialize=False)
    off_by_key = {(r['sat'], r['epoch']): r['error_m'] for r in off}
    for b, v in out['blocks'].items():
        rows_b = [r for r in on if r['block'] == b]
        v['score_rmse_m'] = _rmse(r['error_m'] for r in rows_b)
        v['score_off_rmse_m'] = _rmse(off_by_key[(r['sat'], r['epoch'])] for r in rows_b)
        v['score_starts'] = len(rows_b)
    out['score_rmse_m'] = _rmse(r['error_m'] for r in on)
    out['score_off_rmse_m'] = _rmse(off_by_key[(r['sat'], r['epoch'])] for r in on)
    out['score_starts'] = len(on)
    return out


def print_fit(res):
    """Print one --fit-srp result (per block: k, odd-day RMSE, even-day RMSE on / off)."""
    print(f"\n{res['constellation'].upper()}: k fitted on 2024 days {'/'.join(FIT_DAYS)}, "
          f"scored on {'/'.join(SCORE_DAYS)}")
    print(f"{'block':<14}{'sats':>5}{'k':>7}{'odd RMSE':>11}{'even RMSE':>11}{'even off':>10}{'starts':>8}")
    for b, v in sorted(res['blocks'].items()):
        print(f"{b:<14}{v['n_sats']:>5}{v['k']:>7.2f}{v['fit_rmse_m']:>10.3f}m"
              f"{v['score_rmse_m']:>10.3f}m{v['score_off_rmse_m']:>9.3f}m{v['score_starts']:>8}")
    print(f"{'all':<14}{'':>5}{'':>7}{'':>11}{res['score_rmse_m']:>10.3f}m"
          f"{res['score_off_rmse_m']:>9.3f}m{res['score_starts']:>8}")


def main():
    """Parse the command line, run each requested constellation and print the summary."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--constellation', choices=['gps', 'qzss', 'beidou-geo', 'all'], default='all')
    ap.add_argument('--sp3', nargs='+', help='override the default files (one constellation only)')
    ap.add_argument('--sample-interval', type=int, default=60, help='minutes between starts')
    ap.add_argument('--duration', type=float, default=3600.0, help='prediction horizon [s]')
    ap.add_argument('--csv', help='write every sample to this CSV file')
    ap.add_argument('--srp', choices=['off', 'cannonball'], default='off',
                    help='solar radiation pressure: off (default) or the per-block cannonball')
    ap.add_argument('--metadata', default=DEFAULT_METADATA,
                    help='IGS satellite metadata SINEX (block and mass per satellite)')
    ap.add_argument('--srp-scale', type=float, default=1.0,
                    help='factor on every block area (sensitivity; default 1.0)')
    ap.add_argument('--fit-srp', action='store_true',
                    help='fit the area scale k per block on the odd days, score on the even days')
    ap.add_argument('--fit-json', help='with --fit-srp: write the fit result to this JSON file')
    args = ap.parse_args()

    if (args.srp == 'cannonball' or args.fit_srp) and not os.path.exists(args.metadata):
        sys.exit(f"no IGS satellite metadata file at {args.metadata}: fetch it with\n"
                 f"    python scripts/download_igs_metadata.py\n"
                 f"or pass --metadata PATH")
    if args.fit_json and not args.fit_srp:
        ap.error('--fit-json needs --fit-srp')

    cons = ['gps', 'qzss', 'beidou-geo'] if args.constellation == 'all' else [args.constellation]
    if args.sp3 and len(cons) > 1:
        ap.error('--sp3 needs a single --constellation')
    if args.fit_srp:
        if args.sp3:
            ap.error('--fit-srp uses the committed four days; drop --sp3')
        import ballistic
        import _gnss_metadata
        ballistic.initialize()
        if not ballistic.Loader().load_eop():
            print("WARNING: no EOP data loaded")
        md = _gnss_metadata.read_sinex(args.metadata)
        results = []
        for c in cons:
            print(f"\n{c.upper()}: fitting k on {', '.join(os.path.basename(p) for p in day_files(c, FIT_DAYS))}")
            results.append(fit_srp(c, md, args.sample_interval, initialize=False))
        for res in results:
            print_fit(res)
        if args.fit_json:
            with open(args.fit_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=1)
            print(f"\nWrote {args.fit_json}")
        return

    rows = []
    md = None
    if args.srp == 'cannonball':
        import _gnss_metadata
        md = _gnss_metadata.read_sinex(args.metadata)
    for c in cons:
        paths = args.sp3 or DEFAULT_FILES[c]
        print(f"\n{c.upper()}: {', '.join(os.path.basename(p) for p in paths)}")
        rows += run(c, paths, args.sample_interval, args.duration,
                    srp=args.srp, metadata=md, srp_scale=args.srp_scale)
    summarize(rows)
    if args.csv:
        with open(args.csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} samples to {args.csv}")


if __name__ == '__main__':
    main()
