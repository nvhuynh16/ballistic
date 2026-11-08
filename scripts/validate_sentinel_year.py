#!/usr/bin/env python3
"""
Sentinel full-year prediction-error campaign (every POE state vector).

Measures how far a prediction drifts from the true orbit after 15, 30, 45 and
60 minutes (--horizons), starting from every state of a year of Sentinel precise orbit
ephemerides (POE: the reference orbits ESA and CNES publish, as XML "EOF"
files of Earth-fixed position and velocity). Results are claims
SENTINEL-YEAR-2025, SENTINEL-YEAR-JAN26 and SENTINEL-YEAR-SEP26 in
docs/validation_claims.md.

Every state vector of an ESA POEORB file (Sentinel-1/-2, every 10 s) or a CNES
POE file (Sentinel-3/-6, every 60 s) is used as a starting state and
propagated to +15/30/45/60 min (--horizons); the prediction is compared with
the POE state at the same instant (matched in TAI, International Atomic Time,
which has no leap seconds). Starts are restricted to each file's nominal day
[D 00:00, D+1 00:00) UTC, so the 2-hour overlaps between consecutive files
are never counted twice. A file covers its day plus about an hour on either
side, so with the default horizons the truth comes from the same orbit
solution as the start. A longer horizon also reads the following files of the
same satellite (as many as max(horizons) needs): their states are appended
where the start's file has none (the start's own solution wins in the 2-hour
overlaps, matched by TAI second) and their quality flags are OR'd in, so a
maneuver flagged only in a later file still excludes the start. The configuration is the library default plus solar
radiation pressure (SRP) with the registry's radiation area, FES2004 ocean
tides (unless --ocean-tides off; a file that fails to load is an error), and
with --drag atmospheric drag.

Output: one compressed .npz per (satellite, day) in --out-dir (resumable:
existing files are skipped). Each holds
    t_start_s   (N,)      float64  seconds since 2025-01-01T00:00:00 TAI
    err_rtn     (N,H,3)   float32  predicted − truth [m], split into radial (up),
                                   along-track (direction of motion) and
                                   cross-track (normal to the orbit plane)
    horizons_s  (H,)      float64  the horizons [s]; summarize/compare/srp-fit
                                   read them from the file
    qmask       (N,)      uint8    OR of the quality-flag bits over
                                   [t, t + max(horizons)]
plus per-file metadata (Earth-fixed frame realization "itrf", the UTC/TAI
time-tag cross-check, the file name, the space weather the drag model saw).

Modes (first argument):
    propagate  - run the campaign (use --max-wall-s for chunked runs;
                 --drag for drag from the NRLMSIS 2.1 density model with
                 measured space weather, --srp-scale k to scale the radiation
                 area)
    summarize  - aggregate, print RMSE (root-mean-square error), max and
                 percentiles per satellite and horizon, write summary.csv and
                 rmse_vs_horizon.png into --out-dir
    compare    - drag OFF (--out-dir) vs ON (--drag-dir), start for start;
                 writes drag_on_vs_off.png/.csv into --drag-dir
    srp-fit    - fit a per-satellite radiation-area scale from a k=0
                 (--srp0-dir) and a k=1 (--drag-dir) run

Maneuvers (orbit-changing thruster burns) make a prediction useless across
the burn, so summarize/compare/srp-fit exclude starts whose window
[t, t + max(horizons)] contains one: from the files' flags, plus an
error-based detector for satellites whose files do not flag them reliably
(S3A, S3B, S6A).

Options (defaults in brackets):
  --sats SAT [...]      satellites, e.g. S1A S1C S2B S2C S3A S3B S6A [S1A S2B]
  --year Y              only nominal days in this year [2025]
  --eof-root DIR        directory searched recursively for POE files
                        [data/2025 Sentinel]
  --out-dir DIR         results directory [data/validation/sentinel_year_2025]
  --eop PATH            Earth orientation parameters (EOP: measured irregularities
                        of the Earth's rotation) [tests/data/eop/finals2000A.all]
  -j/--workers N        worker processes [4]
  --stride N            use every Nth state vector [1]
  --day-step N          propagate only every Nth nominal day (day of year 1, 1+N, ...) [1]
  --horizons S,S,...    prediction horizons in seconds, comma-separated
                        [900,1800,2700,3600]; above about one hour the
                        following POE files of the satellite are read too
  --drag                drag on: NRLMSIS 2.1 density + HWM14 wind along the POE track
  --srp-scale K         multiply the radiation area (SRP + albedo + IR); 0 = off [1.0]
  --weather PATH        CelesTrak space weather (F10.7 solar flux, Ap geomagnetic
                        index), used with --drag [tests/data/weather/SW-All.csv]
  --ocean-tides PATH|off  IERS FES2004 file [fes2004_Cnm-Snm.dat in the data directory]
  --drag-dir DIR        compare/srp-fit: the drag-on (k=1) results
                        [data/validation/sentinel_year_2025_drag]
  --srp0-dir DIR        srp-fit: the same configuration with --srp-scale 0
                        [data/validation/sentinel_year_2025_drag_srp0]
  --max-wall-s S        stop submitting new days after S seconds; 0 = no limit [0]
  --detect-dir DIR      summarize: take the S3A/S3B/S6A maneuver marks from this dense
                        run (horizons up to 1 h) instead of --out-dir; needed when the
                        starts are hours apart [none]

Usage:
    PYTHONPATH=./python uv run --no-project --python 3.12 --with numpy \\
        python scripts/validate_sentinel_year.py propagate --sats S1A S2B -j 4
    ... summarize --sats S1A S2B
"""

import argparse
import glob
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_script_dir, '..'))
sys.path.insert(0, os.path.join(_root, 'python'))

HORIZONS_S = (900.0, 1800.0, 2700.0, 3600.0)   # default; also assumed for files without horizons_s
# How far past its nominal day a POE file reaches (ESA: 00:59:42 UTC of the
# next day); horizons beyond it read the following files.
FILE_REACH_S = 3600.0
# Quality-flag bits per state. ESA POEORB (S1/S2) and CNES POE (S3/S6) use
# different flag names; both map onto the same bits (value 1 = maneuver).
# S6A files carry no maneuver flag at all (see summarize's error-based
# detector).
QUALITY_BITS = {'NOMINAL': 0, 'DEGRADED-MANOEUVRE': 1, 'DEGRADED-GAP': 2,
                'DEGRADED-OBSNUMBER': 4, 'DEGRADED-OBSRESIDUALS': 8,
                '3_ADJUST_NOMI': 0, '4_ADJUST_DMAN': 1, '5_INTERP_DGAP': 2,
                '6_EXTRAP_LT1D': 16}
OTHER_BIT = 128
ITRF2020_FROM = datetime(2025, 3, 4)  # ESA orbit products switched ITRF2014 -> ITRF2020
TAI_REF = datetime(2025, 1, 1)        # t_start_s origin (TAI calendar)


def _core_constants():
    """ballistic.constants (imported lazily, like the rest of the package here)."""
    import ballistic
    return ballistic.constants

_NUM = r'([-+\d.]+)'
OSV_RE = re.compile(
    r'<TAI>TAI=([\d-]+T[\d:.]+)</TAI>\s*<UTC>UTC=([\d-]+T[\d:.]+)</UTC>.*?'
    rf'<X unit="m">{_NUM}</X>\s*<Y unit="m">{_NUM}</Y>\s*'
    rf'<Z unit="m">{_NUM}</Z>\s*<VX unit="m/s">{_NUM}</VX>\s*'
    rf'<VY unit="m/s">{_NUM}</VY>\s*<VZ unit="m/s">{_NUM}</VZ>\s*'
    r'<Quality>([^<]*)</Quality>', re.DOTALL)
# File-name patterns -> (sat, validity start, production time).
# S1/S2 (and S3 'POD__'): ESA AUX_POEORB *.EOF;
# S3: CNES SR___POE_AX (gzipped tar); S6: AX____POE__AX (*.SEN6 tar or dir).
NAME_RES = (
    re.compile(r'^(S\d[A-Z])_OPER_AUX_POEORB_(?:OPOD_|POD__)(\d{8}T\d{6})_V(\d{8})T'),
    re.compile(r'^(S3[A-Z])_SR___POE_AX_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8}T\d{6})'),
    re.compile(r'^(S6[A-Z])_AX____POE__AX_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{8}T\d{6})'),
)
# Satellites whose files do not reliably flag maneuvers -> summarize excludes
# flagged OR error-detected windows. S6A flags none; S3A/S3B flag only 1-2
# of their ~25 maneuver events in 2025 (measured).
UNRELIABLE_MANEUVER_FLAGS = {'S3A', 'S3B', 'S6A'}
# CNES POE (S3/S6) write negative values in (-1, 0) as '+0000.xxxxxx' — the
# sign of the zero integer part is lost (measured: 701 '+0000.' vs 4 '-0000.'
# velocity strings in S3; every isolated km-class error traced to one). Any
# OSV with such a velocity component has an unknowable sign -> never a start.
SIGN_AMBIGUOUS_RE = re.compile(r'^\+0+\.')


def sign_ambiguous(osv):
    """True if any velocity string of a parsed state (regex groups) has a lost sign."""
    return any(SIGN_AMBIGUOUS_RE.match(x) for x in osv[5:8])


def _match_name(base):
    """Match a POE file name; return (sat, validity_start_yyyymmdd, production_key) or None."""
    m = NAME_RES[0].match(base)
    if m:  # POEORB: group 2 = production, group 3 = validity start
        return m.group(1), m.group(3), m.group(2)
    for rx in NAME_RES[1:]:
        m = rx.match(base)
        if m:
            return m.group(1), m.group(2), m.group(3)
    return None


def discover(eof_root, sats):
    """Find POE files under `eof_root` for the satellites in `sats`.

    Returns {(sat, 'YYYY-MM-DD' nominal day): path}. The nominal day is the
    day after the file's validity start; when two files cover the same day,
    the one produced latest wins.
    """
    best = {}
    for path in glob.glob(os.path.join(eof_root, '**', '*'), recursive=True):
        base = os.path.basename(path)
        if path.endswith('.EOF') and os.path.basename(os.path.dirname(path)).endswith('.SEN6'):
            continue  # inner file of an unpacked S6 dir; the dir is the entry
        r = _match_name(base)
        if not r or r[0] not in sats:
            continue
        sat, vstart, prod = r
        day = datetime.strptime(vstart, '%Y%m%d') + timedelta(days=1)
        key = (sat, day.strftime('%Y-%m-%d'))
        if key not in best or prod > best[key][0]:
            best[key] = (prod, path)
    return {k: v[1] for k, v in best.items()}


def read_eof_text(path):
    """Return the EOF XML text from a plain .EOF, a (gzipped) tar, or an unpacked directory."""
    import tarfile
    if os.path.isdir(path):
        (inner,) = glob.glob(os.path.join(path, '*.EOF'))
        path = inner
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            (m,) = [m for m in tf.getmembers() if m.name.endswith('.EOF')]
            return tf.extractfile(m).read().decode()
    with open(path, encoding='utf-8') as f:
        return f.read()


def _parse_ts(s):
    """Parse 'YYYY-MM-DDThh:mm:ss.ffffff' into a naive datetime."""
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f')


def _init_worker(eop_path, weather_csv=None, ocean_tides=None):
    """Initialize the library in a worker process (lowest CPU priority).

    Loads the EOP file `eop_path` and, when given, the space-weather CSV and
    the FES2004 ocean-tide file. Raises RuntimeError if any load fails.
    """
    os.nice(19)
    from ballistic import _core
    cfg = _core.InitConfig()
    cfg.eop_path = eop_path
    r = _core.initialize_with_config(cfg)
    if not (r.success and r.eop_loaded):
        raise RuntimeError(f'ballistic init failed (eop_loaded={r.eop_loaded})')
    if weather_csv and not _core.load_space_weather(weather_csv):
        raise RuntimeError(f'space weather failed to load: {weather_csv}')
    if ocean_tides and not _core.load_ocean_tides(ocean_tides):
        raise RuntimeError(f'ocean tides failed to load: {ocean_tides}')


def build_drag_env(utc, tai_epochs, states, beta):
    """Build a tabulated drag environment along the POE track for one file.

    Uses from_models_along: NRLMSIS 2.1 density and HWM14 wind tabulated
    against altitude at POE states at least 60 s apart over the whole file,
    on an altitude grid spanning the orbit +-20 km in 10 km steps. The space
    weather is the library's default: the measured history varying over time
    (SpaceWeatherHistory.from_loaded: previous-day F10.7, rolling 24-hour Ap).

    Parameters: `utc` (list of datetimes of the states), `tai_epochs` (list
    of ballistic Epochs of the same states), `states` (numpy (N,6) Earth-fixed
    position m and velocity m/s), `beta` (ballistic coefficient
    β = m / (Cd·A), kg/m²). Returns (env, history).
    """
    from ballistic import _core
    dt_osv = (utc[1] - utc[0]).total_seconds()
    step = max(1, round(60.0 / dt_osv))
    rnorm_km = np.linalg.norm(states[:, :3], axis=1) / 1e3
    ce = _core_constants().earth
    alt_grid = list(np.arange(np.floor(rnorm_km.min() - ce.RADIUS_EQUATORIAL / 1e3 - 20.0),
                              np.ceil(rnorm_km.max() - ce.RADIUS_POLAR / 1e3 + 20.0) + 1e-9, 10.0))
    sel = list(range(0, len(utc), step))
    sw = _core.SpaceWeatherHistory.from_loaded(tai_epochs[0], tai_epochs[-1])
    env, _diag = _core.from_models_along(
        [tai_epochs[i] for i in sel], [states[i, :3] for i in sel], sw,
        _core.BetaProfile1D.constant(beta), alt_grid)
    return env, sw


def neighbors_needed(horizons):
    """Number of following POE files a start needs for the longest of `horizons` (seconds)."""
    return int(np.ceil(max(0.0, max(horizons) - FILE_REACH_S) / 86400.0))


def _read_osvs(path):
    """Parse one POE file: (tai datetimes, utc datetimes, states (N,6), quality bits, sign-ambiguous)."""
    osvs = OSV_RE.findall(read_eof_text(path))
    tai = [_parse_ts(o[0]) for o in osvs]
    utc = [_parse_ts(o[1]) for o in osvs]
    states = np.array([[float(x) for x in o[2:8]] for o in osvs]).reshape(-1, 6)
    qbits = np.array([QUALITY_BITS.get(o[8], OTHER_BIT) for o in osvs], np.uint8)
    ambiguous = np.array([sign_ambiguous(o) for o in osvs], bool)
    return tai, utc, states, qbits, ambiguous


def _append_neighbors(own, neighbor_paths):
    """Extend one file's states with the following files' states, by TAI second.

    `own` is the tuple of _read_osvs for the start's file. A state of a
    neighbor is appended only at a TAI second the files read so far lack
    (the earlier solution wins where files overlap); at a shared second its
    quality bits are OR'd into the kept state, so a flag in either file
    counts. Returns the same tuple sorted by time, plus the sorted positions
    of the start file's states in their original order.
    """
    tai, utc, states, qbits, ambiguous = own
    tai, utc = list(tai), list(utc)
    states, qbits, ambiguous = list(states), list(qbits), list(ambiguous)
    pos = {round((t - TAI_REF).total_seconds()): i for i, t in enumerate(tai)}
    for path in neighbor_paths:
        n_tai, n_utc, n_states, n_q, n_amb = _read_osvs(path)
        for k, t in enumerate(n_tai):
            key = round((t - TAI_REF).total_seconds())
            if key in pos:
                qbits[pos[key]] |= n_q[k]
                continue
            pos[key] = len(tai)
            tai.append(t)
            utc.append(n_utc[k])
            states.append(n_states[k])
            qbits.append(n_q[k])
            ambiguous.append(n_amb[k])
    n_own = len(own[0])
    order = sorted(range(len(tai)), key=lambda i: tai[i])
    where = {old: new for new, old in enumerate(order)}
    return ([tai[i] for i in order], [utc[i] for i in order],
            np.array([states[i] for i in order]).reshape(-1, 6),
            np.array([qbits[i] for i in order], np.uint8),
            np.array([ambiguous[i] for i in order], bool),
            [where[i] for i in range(n_own)])


def process_day(sat, day, path, out_path, stride, drag=False, srp_scale=1.0,
                horizons=HORIZONS_S, neighbors=()):
    """Propagate every `stride`-th state of one POE file and write the errors to `out_path`.

    `sat` and `day` ('YYYY-MM-DD') name the file's nominal day; only starts
    inside it are used. Starts with a sign-ambiguous velocity or a missing
    truth state at any horizon are skipped and counted. `drag` attaches the
    drag environment; `srp_scale` multiplies the radiation area (0 = off).
    `horizons` are the prediction lengths in seconds; `neighbors` are the
    following POE files of the same satellite, read for truth states (and
    quality flags) past the end of `path` (see _append_neighbors).
    Writes the .npz described in the module docstring (via a temporary file,
    so an interrupted run leaves no partial result). Returns (sat, day,
    number of starts, largest UTC/TAI time-tag difference in seconds).
    """
    from ballistic import _core
    from ballistic import satellite_properties as sp

    horizons = tuple(float(h) for h in horizons)
    own = _read_osvs(path)
    n = len(own[0])                       # the start file's state count (n_osv)
    if neighbors:
        tai, utc, states, qbits, ambiguous, own_index = _append_neighbors(own, neighbors)
    else:
        (tai, utc, states, qbits, ambiguous), own_index = own, range(n)
    # Integer TAI seconds index (POE tags are whole seconds).
    tai_s = np.array([round((t - TAI_REF).total_seconds()) for t in tai], np.int64)
    idx_of = {int(s): i for i, s in enumerate(tai_s)}

    day0 = datetime.strptime(day, '%Y-%m-%d')
    itrf = (_core.ITRFRealization.ITRF2020 if day0 >= ITRF2020_FROM
            else _core.ITRFRealization.ITRF2014)
    props = sp.get(sat)
    cfg = _core.TrajectoryConfig()
    cfg.itrf = itrf
    # srp_scale multiplies the radiation area (SRP + albedo + IR); 0 = off.
    cfg.srp = (_core.SrpConfig(mass_kg=props.mass_kg, area_m2=props.srp_area_m2 * srp_scale,
                               cr=1.0 + props.srp_reflectivity) if srp_scale > 0 else None)

    def tai_epoch(t):
        """Epoch of a TAI datetime, built from the calendar fields."""
        # From the calendar, not a floating MJD: an MJD double rounds to
        # ~0.6 us, so e0 + h and the drag table's last epoch could differ by
        # a fraction of a microsecond (and the start labels jittered).
        return _core.Epoch.from_tai(t.year, t.month, t.day, t.hour, t.minute,
                                    t.second + t.microsecond * 1e-6)

    # Time-tag cross-check: each state carries both a UTC and a TAI time tag;
    # compare them for every 60th orbit state vector (OSV) of the start file. Non-zero means a
    # time-scale mislabeling in the file or in the library's leap-second table.
    sync = [abs(_core.Epoch.from_utc(u.year, u.month, u.day, u.hour, u.minute,
                                     u.second + u.microsecond * 1e-6)
                - tai_epoch(t)) for t, u in zip(own[0][::60], own[1][::60])]
    weather = np.zeros((0, 3))
    if drag:
        cfg.drag_env, sw = build_drag_env(utc, [tai_epoch(t) for t in tai],
                                          states, props.beta_kg_m2)
        # What NRLMSIS saw (F10.7, F10.7 81-day, daily Ap) at the 3-h bin
        # midpoints of the nominal day.
        mids = [day0 + timedelta(hours=1.5 + 3 * b) for b in range(8)]
        ws = [sw.at(_core.Epoch.from_utc(m.year, m.month, m.day, m.hour, m.minute, 0.0))
              for m in mids]
        weather = np.array([(w.f107_daily, w.f107_average, w.ap_history[0]) for w in ws])
    # For each start: the OR of the quality bits over [start, start + max horizon],
    # and the errors at each horizon split into radial / along-track /
    # cross-track using the truth state (velocity made inertial by adding the
    # Earth's rotation, so the orbit plane is the inertial one).
    t_out, err_out, q_out, n_missing, n_ambiguous = [], [], [], 0, 0
    for i in own_index[::stride]:
        if not (day0 <= utc[i] < day0 + timedelta(days=1)):
            continue
        if ambiguous[i]:
            n_ambiguous += 1
            continue
        js = [idx_of.get(int(tai_s[i] + h)) for h in horizons]
        if any(j is None for j in js):
            n_missing += 1
            continue
        q = np.bitwise_or.reduce(qbits[i:js[-1] + 1])
        e0 = tai_epoch(tai[i])
        tr = _core.Trajectory(e0, states[i, :3], states[i, 3:], cfg)
        errs = np.empty((len(horizons), 3))
        for k, (h, j) in enumerate(zip(horizons, js)):
            pos = np.asarray(tr.propagate(e0 + h).position)
            r, v = states[j, :3], states[j, 3:]
            v_in = v + np.cross([0.0, 0.0, _core_constants().earth.ANGULAR_VELOCITY], r)
            R = r / np.linalg.norm(r)
            N = np.cross(r, v_in)
            N /= np.linalg.norm(N)
            T = np.cross(N, R)
            d = pos - r
            errs[k] = (d @ R, d @ T, d @ N)
        t_out.append(float(tai_s[i]))
        err_out.append(errs)
        q_out.append(q)

    tmp = out_path + '.tmp.npz'
    np.savez_compressed(
        tmp, t_start_s=np.array(t_out), err_rtn=np.array(err_out, np.float32).reshape(-1, len(horizons), 3),
        qmask=np.array(q_out, np.uint8), horizons_s=np.array(horizons),
        sat=sat, day=day, eof_file=os.path.basename(path),
        itrf=str(itrf), time_tag_max_diff_s=max(sync), n_osv=n,
        n_missing_truth=n_missing, n_sign_ambiguous=n_ambiguous,
        stride=stride, drag=drag, srp_scale=srp_scale,
        weather_f107_f107avg_ap=weather,
        # Actual minus requested elapsed time of e0 + h in THIS build (zero
        # with exact Epoch storage); load_all corrects with it.
        horizon_time_offset_s=epoch_add_offsets_s(horizons),
        **({'neighbor_files': np.array([os.path.basename(x) for x in neighbors])}
           if neighbors else {}))
    os.replace(tmp, out_path)
    return sat, day, len(t_out), max(sync)


def cmd_propagate(a):
    """'propagate' mode: run process_day for every (sat, day) of --year not yet in --out-dir.

    `a` is the parsed argparse namespace. Days run in parallel on --workers
    processes; with --max-wall-s no new day is started after that many
    seconds, and a later run resumes where this one stopped.
    """
    jobs = discover(a.eof_root, set(a.sats))
    os.makedirs(a.out_dir, exist_ok=True)
    n_next = neighbors_needed(a.horizons)
    todo = []
    for (sat, day), path in sorted(jobs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if not day.startswith(str(a.year)):
            continue
        # --day-step N keeps every Nth nominal day (day of year 1, 1+N, ...):
        # multi-day horizons cost ~10 s of CPU per start, so a sample of days
        # spread over the year replaces every day.
        if (datetime.strptime(day, '%Y-%m-%d').timetuple().tm_yday - 1) % a.day_step:
            continue
        out = os.path.join(a.out_dir, f'{sat}_{day}.npz')
        if not os.path.exists(out):
            # The following days' files of the same satellite, as far as the
            # longest horizon needs (a missing day leaves those starts without
            # truth; they are counted as missing, not scored).
            d0 = datetime.strptime(day, '%Y-%m-%d')
            nxt = [jobs[(sat, (d0 + timedelta(days=k)).strftime('%Y-%m-%d'))]
                   for k in range(1, n_next + 1)
                   if (sat, (d0 + timedelta(days=k)).strftime('%Y-%m-%d')) in jobs]
            todo.append((sat, day, path, out, nxt))
    print(f'{len(jobs)} files discovered, {len(todo)} (sat,day) remaining', flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(a.workers, initializer=_init_worker,
                             initargs=(a.eop, a.weather if a.drag else None,
                                       None if a.ocean_tides == 'off' else a.ocean_tides)) as ex:
        pending = set()
        it = iter(todo)
        while True:
            while len(pending) < a.workers and (
                    a.max_wall_s <= 0 or time.time() - t0 < a.max_wall_s):
                nxt = next(it, None)
                if nxt is None:
                    break
                sat, day, path, out, neighbors = nxt
                pending.add(ex.submit(process_day, sat, day, path, out, a.stride, a.drag,
                                      a.srp_scale, tuple(a.horizons), tuple(neighbors)))
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                sat, day, n, sync = f.result()
                done += 1
                print(f'{sat} {day}: {n} starts, time-tag diff {sync*1e6:.0f} us '
                      f'[{done} done, {time.time()-t0:.0f}s]', flush=True)


# Timing offsets for results files written by library builds that stored an
# Epoch as a single double-precision Julian date (a ~40 us grid): there
# e0 + h landed off by these amounts at the 15/30/45/60-min horizons
# (measured on such a build, identical for every start in 2025-2026). Files
# that record horizon_time_offset_s use their recorded value instead.
LEGACY_HORIZON_TIME_OFFSET_S = np.array([-13.411e-6, 13.411e-6, 0.0, -13.411e-6])


def epoch_add_offsets_s(horizons=HORIZONS_S):
    """Actual minus requested elapsed time of `e0 + h` for each of `horizons`, in
    the running build. Zero with exact Epoch storage; nonzero for a
    single-double Epoch (LEGACY_HORIZON_TIME_OFFSET_S). The propagation stops
    short of the truth instant by this amount — at LEO ~0.1 m along-track
    per 13 us — so load_all(correct_time=True) removes it, using the offset
    each file recorded when it was written (not the summarizing build's).
    """
    from ballistic import _core
    _core.initialize()
    e0 = _core.Epoch.from_utc(2025, 7, 1, 0, 0, 0.0)
    return np.array([(e0 + h) - e0 - h for h in horizons])


def load_all(out_dir, sats, correct_time=True):
    """Load every results file in `out_dir` for the satellites in `sats`.

    With `correct_time`, the along-track error is corrected for each file's
    horizon timing offset (offset in seconds times the circular orbital speed
    at the satellite's registry altitude). Returns {sat: dict} with keys t
    (numpy (N,) start seconds since TAI_REF), e (numpy (N,H,3) radial /
    along-track / cross-track errors, m), q (numpy (N,) quality bits), days
    (list of day strings), drag (bool, from the first file) and sync (largest
    UTC/TAI time-tag difference, s) and horizons (tuple of seconds, from the
    files; HORIZONS_S for files that predate the field). Satellites without
    files are omitted. Raises ValueError when one satellite's files were run
    with different horizons.
    """
    from ballistic import satellite_properties as sp
    out = {}
    for sat in sats:
        files = sorted(glob.glob(os.path.join(out_dir, f'{sat}_*.npz')))
        if not files:
            continue
        parts = [np.load(f) for f in files]
        hz = {tuple(float(h) for h in (p['horizons_s'] if 'horizons_s' in p.files else HORIZONS_S))
              for p in parts}
        if len(hz) > 1:
            raise ValueError(f'{out_dir}: {sat} files were run with different horizons {sorted(hz)}; '
                             f'summarize each set in its own --out-dir')
        ce = _core_constants().earth
        v_circ = np.sqrt(ce.GM / (ce.RADIUS_EQUATORIAL + sp.get(sat).altitude_km * 1e3))
        chunks = []
        for p in parts:
            ep = p['err_rtn'].copy()
            if correct_time:
                dt = (p['horizon_time_offset_s'] if 'horizon_time_offset_s' in p.files
                      else LEGACY_HORIZON_TIME_OFFSET_S)
                ep[:, :, 1] -= (v_circ * dt).astype(np.float32)
            chunks.append(ep)
        e = np.concatenate(chunks)
        out[sat] = {
            't': np.concatenate([p['t_start_s'] for p in parts]),
            'e': e,
            'q': np.concatenate([p['qmask'] for p in parts]),
            'days': [str(p['day']) for p in parts],
            'drag': bool(parts[0]['drag']) if 'drag' in parts[0] else False,
            'sync': max(float(p['time_tag_max_diff_s']) for p in parts),
            'horizons': hz.pop()}
    return out


def common_horizons(*datasets):
    """The horizons (seconds) shared by every satellite of the load_all results given.

    Raises ValueError when they differ (results of different --horizons runs
    cannot be pooled or matched start for start).
    """
    hz = {d['horizons'] for data in datasets for d in data.values()}
    if len(hz) > 1:
        raise ValueError(f'results with different horizons cannot be combined: {sorted(hz)}')
    return hz.pop() if hz else HORIZONS_S


def _hmin(h):
    """A horizon in seconds as a minutes label ('60', '1440')."""
    return f'{h / 60:g}'


DETECT_THRESHOLD_M = 5.0  # 60-min error that marks a suspected maneuver


def _detection_column(horizons):
    """Index of the horizon the error-based detector reads: the longest one up to 60 min."""
    ok = [k for k, h in enumerate(horizons) if h <= 3600.0]
    return ok[-1] if ok else 0


def detect_maneuvers(t, e, threshold_m=DETECT_THRESHOLD_M, horizons=HORIZONS_S):
    """Error-based maneuver exclusion for files without maneuver flags.

    `t` is numpy (N,) start times in seconds, `e` numpy (N,H,3) errors in
    meters at `horizons` (seconds). The detector reads the longest horizon up
    to 60 min (h_d): a start whose error there exceeds `threshold_m` marks a
    burn somewhere in [t_b, t_b + h_d]. A start t is excluded when such a
    marked start lies in [t − h_d, t + max(horizons)], i.e. when the burn can
    fall inside its window [t, t + max(horizons)] (with the default horizons:
    any marked start within ±60 min).
    Returns a boolean numpy (N,) mask, True = exclude. It is circular by
    construction (it removes large errors), so its exclusions are checked
    against the ESA/CNES flags on the satellites that have them (summarize
    prints how many flagged events it catches and how many extra starts it
    excludes, per flagged satellite).
    """
    kd = _detection_column(horizons)
    h_d, h_max = horizons[kd], max(horizons)
    norm_d = np.linalg.norm(e[:, kd], axis=1)
    tb = np.sort(t[norm_d > threshold_m])
    if len(tb) == 0:
        return np.zeros(len(t), bool)
    # First marked start at or after t - h_d; excluded if it is not past t + h_max.
    k = np.searchsorted(tb, t - h_d, side='left')
    return (k < len(tb)) & (tb[np.minimum(k, len(tb) - 1)] <= t + h_max)


def _runs(mask, t, gap_s=HORIZONS_S[-1]):
    """Number of separate events in a boolean start mask (gap > gap_s splits)."""
    tt = np.sort(t[mask])
    return 0 if len(tt) == 0 else 1 + int(np.sum(np.diff(tt) > gap_s))


def maneuver_exclusion(sat, d):
    """Return (exclude_mask, label) for satellite `sat` and its load_all dict `d`.

    The mask is the file's maneuver flag, OR'd with detect_maneuvers for
    the satellites whose flags are known to be incomplete
    (UNRELIABLE_MANEUVER_FLAGS); the label describes the subset kept. The
    flag mask already covers [t, t + max(horizons)] (qmask is OR'd over that
    window, neighboring files included); `d['horizons']` (default
    HORIZONS_S) sizes the detector's window.
    """
    flagged = (d['q'] & 1) != 0
    if sat in UNRELIABLE_MANEUVER_FLAGS and 'burn_marks' in d:
        # Marks from a dense one-hour run (--detect-dir): sparse multi-day
        # starts rarely fall within an hour before a burn, so their own
        # errors cannot locate it.
        tb, h_d = d['burn_marks'], d['burn_h_d']
        if len(tb):
            k = np.searchsorted(tb, d['t'] - h_d, side='left')
            near = (k < len(tb)) & (tb[np.minimum(k, len(tb) - 1)] <= d['t'] + max(d['horizons']))
        else:
            near = np.zeros(len(d['t']), bool)
        return (flagged | near,
                f'no flagged/detected maneuver (>{DETECT_THRESHOLD_M:g} m, --detect-dir)')
    if sat in UNRELIABLE_MANEUVER_FLAGS:
        return (flagged | detect_maneuvers(d['t'], d['e'],
                                            horizons=d.get('horizons', HORIZONS_S)),
                f'no flagged/detected maneuver (>{DETECT_THRESHOLD_M:g} m)')
    return flagged, 'no MANEUVER'


def cmd_summarize(a):
    """'summarize' mode: print error statistics, write summary.csv and rmse_vs_horizon.png.

    For each satellite: all starts, starts whose window is all NOMINAL,
    starts outside maneuver windows, and that subset on even days only (the
    radiation areas were fitted on odd days of 2025). Then all satellites
    pooled.
    """
    data = load_all(a.out_dir, a.sats)
    horizons = common_horizons(data)
    if a.detect_dir:
        # Maneuver marks for the satellites without reliable flags, taken
        # from a dense run with horizons up to one hour over the same days.
        dense = load_all(a.detect_dir, [s for s in data if s in UNRELIABLE_MANEUVER_FLAGS])
        for sat, dd in dense.items():
            kd_ext = _detection_column(dd['horizons'])
            norm_ext = np.linalg.norm(dd['e'][:, kd_ext], axis=1)
            data[sat]['burn_marks'] = np.sort(dd['t'][norm_ext > DETECT_THRESHOLD_M])
            data[sat]['burn_h_d'] = dd['horizons'][kd_ext]
            print(f'{sat}: {_runs(norm_ext > DETECT_THRESHOLD_M, dd["t"], 3600.0)} suspected '
                  f'maneuvers from {len(dd["t"])} starts in {a.detect_dir}')
    hz = [int(h / 60) for h in horizons]
    h_max = max(horizons)
    kd = _detection_column(horizons)
    rows, pooled = [], {}
    for sat, d in data.items():
        norm = np.linalg.norm(d['e'], axis=2)
        print(f'\n== {sat}: {len(d["days"])} days, {len(norm)} starts, '
              f'max time-tag diff {d["sync"]*1e6:.0f} us')
        flagged = (d['q'] & 1) != 0
        det = detect_maneuvers(d['t'], d['e'], horizons=horizons)
        if flagged.any():
            caught = _runs(flagged & det, d['t'], h_max)
            trig = (norm[:, kd] > DETECT_THRESHOLD_M) & ~flagged
            print(f'  maneuver flags: {_runs(flagged, d["t"], h_max)} events / {flagged.sum()} starts; '
                  f'error detector catches {caught} of them, triggers on {trig.sum()} '
                  f'unflagged starts, pads {(det & ~flagged).sum()} extra starts')
        else:
            print(f'  no maneuver flags; error detector excludes {det.sum()} starts '
                  f'in {_runs(det, d["t"], h_max)} events')
        excl, mlabel = maneuver_exclusion(sat, d)
        # The SRP areas were fitted on the odd days of 2025 (srp-fit), so the
        # even days are the out-of-sample part of a 2025 run.
        even = (d['t'] // 86400).astype(int) % 2 == 0
        pooled.setdefault('all', []).append(norm[~excl])
        pooled.setdefault('even', []).append(norm[~excl & even])
        for label, mask in (('all', np.ones(len(norm), bool)),
                            ('NOMINAL window', d['q'] == 0),
                            (mlabel, ~excl),
                            (mlabel + ', even days', ~excl & even)):
            x = norm[mask]
            print(f'  {label:15s} n={mask.sum():8d}')
            if not mask.any():
                continue      # e.g. every start of a short run has a maneuver in its window
            for k, h in enumerate(hz):
                rms = np.sqrt(np.mean(x[:, k] ** 2))
                p = np.percentile(x[:, k], [50, 95, 99, 99.9])
                rtn = np.sqrt(np.mean(d['e'][mask, k] ** 2, axis=0))
                print(f'    {h:2d} min  RMSE {rms:7.3f}  p50 {p[0]:6.3f}  p95 {p[1]:6.3f}  '
                      f'p99 {p[2]:7.3f}  p99.9 {p[3]:8.3f}  max {x[:, k].max():9.2f}  '
                      f'RTN-RMS {rtn[0]:.3f}/{rtn[1]:.3f}/{rtn[2]:.3f}')
                rows.append((sat, label, h, mask.sum(), rms, *p, x[:, k].max(), *rtn))
    print(f'\n== all {len(data)} satellites pooled, maneuver windows excluded')
    for label, parts in pooled.items():
        x = np.concatenate(parts)
        if not len(x):
            print(f'  {label + " days":9s} n=       0')
            continue
        print(f'  {label + " days":9s} n={len(x):8d}  ' + '  '.join(
            f'{h} min RMSE {np.sqrt(np.mean(x[:, k] ** 2)):.3f}' for k, h in enumerate(hz))
            + f'  max({hz[-1]}) {x[:, -1].max():.2f}')
    csv = os.path.join(a.out_dir, 'summary.csv')
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('sat,subset,horizon_min,n,rmse_m,p50,p95,p99,p999,max_m,rms_r,rms_t,rms_n\n')
        f.writelines(','.join(str(x) if isinstance(x, str) else f'{x:.6g}' for x in r) + '\n' for r in rows)
    print(f'\nWrote {csv}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for sat, d in data.items():
        norm = np.linalg.norm(d['e'], axis=2)
        m = ~maneuver_exclusion(sat, d)[0]
        x = [0] + hz
        c = f'C{list(data).index(sat)}'
        if m.any():
            axes[0].plot(x, [0] + list(np.sqrt(np.mean(norm[m] ** 2, axis=0))), 'o-', color=c, label=f'{sat} RMSE')
            axes[0].plot(x, [0] + list(np.percentile(norm[m], 99, axis=0)), '--', color=c, alpha=0.6, label=f'{sat} p99')
        axes[1].plot(hz, norm.max(axis=0), 'o-', color=c, label=f'{sat} max (all)')
        if m.any():
            axes[1].plot(hz, norm[m].max(axis=0), '--', color=c, label=f'{sat} max (no maneuver)')
    axes[0].set(xlabel='Prediction horizon [min]', ylabel='3D position error [m]',
                title='RMSE / p99 (maneuver windows excluded)')
    axes[1].set(xlabel='Prediction horizon [min]', ylabel='3D position error [m]',
                title='Maximum error', yscale='log')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xticks([0] + hz)
    fig.suptitle(f'Sentinel {a.year} — every POE state vector, SRP on, '
                 f'drag {"ON (NRLMSIS 2.1, measured weather)" if any(d["drag"] for d in data.values()) else "off"}')
    fig.tight_layout()
    png = os.path.join(a.out_dir, 'rmse_vs_horizon.png')
    fig.savefig(png, dpi=130)
    print(f'Wrote {png}')


def cmd_compare(a):
    """'compare' mode: drag OFF (--out-dir) vs drag ON (--drag-dir) on identical starts.

    Prints RMSE, maximum and mean along-track error per horizon and the
    monthly 60-min RMSE; writes drag_on_vs_off.png and .csv into --drag-dir.
    """
    off_all, on_all = load_all(a.out_dir, a.sats), load_all(a.drag_dir, a.sats)
    horizons = common_horizons(off_all, on_all)
    hz = [int(h / 60) for h in horizons]
    hl = _hmin(horizons[-1])
    rows = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for sat in a.sats:
        if sat not in off_all or sat not in on_all:
            continue
        off, on = off_all[sat], on_all[sat]
        common, i_off, i_on = np.intersect1d(off['t'], on['t'], return_indices=True)
        eo, en = off['e'][i_off], on['e'][i_on]
        sub = {'t': common, 'e': eo, 'q': off['q'][i_off], 'horizons': horizons}
        keep = ~maneuver_exclusion(sat, sub)[0]
        no, nn = np.linalg.norm(eo[keep], axis=2), np.linalg.norm(en[keep], axis=2)
        print(f'\n== {sat}: {keep.sum()} matched starts (maneuver windows excluded)')
        print('    horizon   RMSE off   RMSE on   change    max off   max on   '
              'along-track mean@h off/on')
        for k, h in enumerate(hz):
            ro, rn = np.sqrt(np.mean(no[:, k] ** 2)), np.sqrt(np.mean(nn[:, k] ** 2))
            print(f'    {h:2d} min   {ro:8.3f}  {rn:8.3f}  {100*(rn/ro-1):+6.1f}%  '
                  f'{no[:, k].max():8.2f}  {nn[:, k].max():7.2f}   '
                  f'{eo[keep, k, 1].mean():+.3f} / {en[keep, k, 1].mean():+.3f}')
            rows.append((sat, h, keep.sum(), ro, rn, no[:, k].max(), nn[:, k].max()))
        # Monthly RMSE at the longest horizon (seasonal SRP signature vs drag/activity).
        month = np.array([(TAI_REF + timedelta(seconds=float(x))).month for x in common[keep]])
        mo = [np.sqrt(np.mean(no[month == m, -1] ** 2)) if (month == m).any() else np.nan for m in range(1, 13)]
        mn = [np.sqrt(np.mean(nn[month == m, -1] ** 2)) if (month == m).any() else np.nan for m in range(1, 13)]
        print(f'    monthly {hl}-min RMSE off: ' + ' '.join(f'{x:.2f}' for x in mo))
        print(f'    monthly {hl}-min RMSE on : ' + ' '.join(f'{x:.2f}' for x in mn))
        line, = axes[0].plot(range(1, 13), mo, 'o--', alpha=0.6, label=f'{sat} drag off')
        axes[0].plot(range(1, 13), mn, 'o-', color=line.get_color(), label=f'{sat} drag on')
        axes[1].plot([0] + hz, [0] + [100 * (r[4] / r[3] - 1) for r in rows if r[0] == sat],
                     'o-', label=sat)
    axes[0].set(xlabel='Month (2025)', ylabel=f'{hl}-min RMSE [m]', title=f'Monthly {hl}-min RMSE')
    axes[1].set(xlabel='Prediction horizon [min]', ylabel='RMSE change, drag on vs off [%]',
                title='Effect of drag (NRLMSIS 2.1 + measured space weather)')
    axes[1].axhline(0, color='k', lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(a.drag_dir, 'drag_on_vs_off.png')
    fig.savefig(png, dpi=130)
    csv = os.path.join(a.drag_dir, 'drag_on_vs_off.csv')
    with open(csv, 'w', encoding='utf-8') as f:
        f.write('sat,horizon_min,n,rmse_off_m,rmse_on_m,max_off_m,max_on_m\n')
        f.writelines(','.join(str(x) if isinstance(x, str) else f'{x:.6g}' for x in r) + '\n' for r in rows)
    print(f'\nWrote {png}\nWrote {csv}')


def cmd_srp_fit(a):
    """'srp-fit' mode: fit a per-satellite radiation scale k (SRP+albedo+IR area multiplier).

    The 1-h error is linear in k (k=2 predicted from k=0,1 to <=1.2 mm), so
    e(k) = e0 + k (e1 - e0) with e1 = --drag-dir (k=1) and e0 = --srp0-dir
    (same config, --srp-scale 0). k is fitted on odd days and scored on even
    days (60-min horizon, maneuver windows excluded); per-month k and the
    spread of monthly along-track means show whether a scale can absorb the
    seasonal error (measured: it cannot).
    """
    e1all, e0all = load_all(a.drag_dir, a.sats), load_all(a.srp0_dir, a.sats)
    horizons = common_horizons(e1all, e0all)

    def rmse(x):
        return np.sqrt(np.mean(np.sum(x ** 2, axis=1)))

    def kfit(e0, d):
        return -np.sum(e0 * d) / np.sum(d * d)

    print(f'sat    k*   RMSE{_hmin(horizons[-1])} even days k=0 / k=1 / k*   '
          f'swing std k=1 -> k*   monthly k*')
    for sat in a.sats:
        if sat not in e1all or sat not in e0all:
            continue
        x1, x0 = e1all[sat], e0all[sat]
        c, i1, i0 = np.intersect1d(x1['t'], x0['t'], return_indices=True)
        keep = ~maneuver_exclusion(sat, {'t': c, 'e': x1['e'][i1], 'q': x1['q'][i1],
                                          'horizons': horizons})[0]
        E1 = x1['e'][i1][keep][:, -1].astype(float)
        E0 = x0['e'][i0][keep][:, -1].astype(float)
        D, t = E1 - E0, c[keep]
        day = (t // 86400).astype(int)
        odd, even = day % 2 == 1, day % 2 == 0
        k = kfit(E0[odd], D[odd])
        Ek = E0 + k * D
        mon = np.array([(TAI_REF + timedelta(seconds=float(x))).month for x in t])
        ms = [m for m in range(1, 13) if ((mon == m) & even).sum() > 100]
        def swing(E, mon=mon, even=even, ms=ms):
            return np.std([E[(mon == m) & even, 1].mean() for m in ms])
        km = [kfit(E0[mon == m], D[mon == m]) for m in ms]
        print(f'{sat}  {k:5.2f}   {rmse(E0[even]):.3f} / {rmse(E1[even]):.3f} / {rmse(Ek[even]):.3f}'
              f'        {swing(E1):.3f} -> {swing(Ek):.3f}      ' + ' '.join(f'{x:.1f}' for x in km))


def _data_dir():
    """The library's platform data directory (where ballistic.load() stores files)."""
    from ballistic import _core
    return str(_core.get_data_directory())


def _parse_horizons(text):
    """Parse '900,1800,...' into a sorted list of distinct positive horizons in seconds."""
    try:
        hz = sorted({float(x) for x in text.split(',') if x.strip()})
    except ValueError:
        raise argparse.ArgumentTypeError(f"--horizons wants seconds separated by commas, got '{text}'")
    if not hz or hz[0] <= 0 or any(h != round(h) for h in hz):
        raise argparse.ArgumentTypeError(
            f"--horizons wants positive whole seconds (POE states are on whole seconds), got '{text}'")
    return hz


def main():
    """Parse the command line (see the module docstring) and run the chosen mode."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('mode', choices=['propagate', 'summarize', 'compare', 'srp-fit'])
    p.add_argument('--sats', nargs='+', default=['S1A', 'S2B'])
    p.add_argument('--year', type=int, default=2025)
    p.add_argument('--eof-root', default=os.path.join(_root, 'data', '2025 Sentinel'))
    p.add_argument('--out-dir', default=os.path.join(_root, 'data', 'validation', 'sentinel_year_2025'))
    p.add_argument('--eop', default=os.path.join(_root, 'tests', 'data', 'eop', 'finals2000A.all'))
    p.add_argument('-j', '--workers', type=int, default=4)
    p.add_argument('--stride', type=int, default=1, help='use every Nth state vector')
    p.add_argument('--day-step', type=int, default=1,
                   help='propagate: use every Nth nominal day of the year (day of year 1, 1+N, ...)')
    p.add_argument('--horizons', type=_parse_horizons, default=list(HORIZONS_S),
                   help='prediction horizons [s], comma-separated (default 900,1800,2700,3600)')
    p.add_argument('--drag', action='store_true',
                   help='drag ON: NRLMSIS 2.1 + HWM14 along the POE track, driven by '
                        'the measured --weather (the library default history)')
    p.add_argument('--srp-scale', type=float, default=1.0,
                   help='multiply the radiation area (SRP+albedo+IR); 0 = radiation off')
    p.add_argument('--weather', default=os.path.join(_root, 'tests', 'data', 'weather', 'SW-All.csv'))
    p.add_argument('--ocean-tides', default=os.path.join(_data_dir(), 'fes2004_Cnm-Snm.dat'),
                   help="IERS FES2004 Stokes file (ballistic.load() fetches it), or 'off'")
    p.add_argument('--drag-dir', default=os.path.join(_root, 'data', 'validation', 'sentinel_year_2025_drag'),
                   help='compare: drag-ON results (matched start-for-start against --out-dir)')
    p.add_argument('--srp0-dir', default=os.path.join(_root, 'data', 'validation', 'sentinel_year_2025_drag_srp0'),
                   help='srp-fit: same config as --drag-dir but --srp-scale 0')
    p.add_argument('--detect-dir', default=None,
                   help='summarize: locate maneuvers of S3A/S3B/S6A (no reliable flags) from this '
                        'dense run with horizons up to 1 h, instead of from --out-dir itself')
    p.add_argument('--max-wall-s', type=float, default=0, help='stop submitting after N s (0 = no limit)')
    a = p.parse_args()
    {'propagate': cmd_propagate, 'summarize': cmd_summarize,
     'compare': cmd_compare, 'srp-fit': cmd_srp_fit}[a.mode](a)


if __name__ == '__main__':
    main()
