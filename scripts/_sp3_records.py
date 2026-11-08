"""Shared SP3 reading for the validation scripts.

SP3 is the standard text format for precise satellite orbits: a list of
epochs with each satellite's position (and sometimes velocity) at each one.
Every validation script reads its reference ("truth") orbits through this
module, which wraps the library's single SP3 reader, ``ballistic.sp3``,
instead of carrying a private parser (docs/trap_census.md, trap
SP3-DUPLICATE-READERS). What it guarantees:

- Epochs honor the time system named in the file. GFZ CHAMP/GRACE-FO and
  AIUB files are in GPS time; reading them as UTC would shift every epoch by
  the GPS-UTC offset (18 s in 2026, 13-16 s in 2005-2013).
- By default each file is its own arc: a start state and the truth state one
  horizon later come from the same file, never from two overlapping files
  that are separate orbit solutions. A start time already scored from an
  earlier file is not scored again.
- Horizons longer than one file need several files joined into one arc
  (``join_arcs`` or ``load_arcs(..., join=True)``). The joined arc keeps each
  record as its own file wrote it (velocities derived within that file), drops
  the duplicate epochs where files overlap in favor of the record further
  from its file's edge, and refuses mixed frame or time-system labels and gaps
  longer than ``max_gap_s``. The jump where two independent solutions meet
  then lies inside the prediction window: measured 7 mm median between CODE
  daily GPS files (claim GNSS-24H), small next to the error of a prediction
  long enough to need the join, but not negligible for a 1-hour LEO score.
- Records whose clock field runs into the z coordinate
  ("441.732191-102852.9") are still read (8 in each GRACE-FO file of one
  February 2026 day; trap SP3-CLOCK-OVERFLOW).
- When velocities are derived from the positions, the reader's "centered"
  flag says whether the differentiation window was centered on the epoch
  (accurate) or one-sided at a file edge (less accurate).

Each record is a dict with the keys the scripts use:
  'pos_m'          position in the file's frame (Earth-fixed, or inertial
                   for a J2000 file), meters, numpy (3,)
  'vel_ms'         velocity in the same frame, meters/second, numpy (3,)
  'vel_from_file'  True if the velocity is the file's own record
  'vel_edge'       True if a derived velocity came from a one-sided window
  'offset_s'       exact seconds since the arc's first epoch (epoch0)
"""

import numpy as np

VELOCITY_SOURCES = ('file', 'derived')


def load_arcs(ballistic, paths, satellites=None, id_map=None, velocity='file', quiet=False,
              join=False, max_gap_s=900.0, dedupe_tol_s=1.0):
    """Read SP3 files into per-satellite lists of arcs (one arc per file, or joined).

    Parameters:
      ballistic   the imported ``ballistic`` module.
      paths       iterable of SP3 file paths (.sp3 or gzip-compressed).
      satellites  optional collection of SP3 satellite ids to keep (e.g.
                  'L47'); None keeps all.
      id_map      optional dict renaming SP3 ids to the names used in the
                  output (SP3 id -> name).
      velocity    'file' uses the file's velocity records when present;
                  'derived' always differentiates the positions. A file
                  without velocity records is always derived.
      quiet       if False, print one line per arc read.
      join        if True, join each satellite's arcs into ONE arc with
                  join_arcs (max_gap_s and dedupe_tol_s are passed on), so a
                  prediction can run past the end of one file.

    Returns {name: [arc, ...]} in file order (with join=True, {name: [joined
    arc]}). Each arc is a dict with keys
    source (path string), satellite (SP3 id), name, epoch0 (ballistic.Epoch of
    the first record, in the file's time system), offsets (numpy (N,) seconds
    since epoch0), records (list of record dicts, see the module docstring),
    frame (the reader's frame description), frame_label (the header's frame
    label string), time_system (name string) and sp3_arc (the underlying
    ``ballistic.sp3`` arc).

    Raises ValueError for an unknown `velocity`; the reader raises on a
    malformed file or an unknown frame label.
    """
    if velocity not in VELOCITY_SOURCES:
        raise ValueError(f"velocity must be one of {VELOCITY_SOURCES}")
    out = {}
    for path in paths:
        f = ballistic.sp3.read(path)
        frame = f.frame()
        for arc in f.arcs:
            if satellites and arc.satellite not in satellites:
                continue
            name = (id_map or {}).get(arc.satellite, arc.satellite)
            pos = arc.position
            off = arc.offsets_s
            derived, centered = arc.derived_velocities()
            from_file = velocity == 'file' and arc.has_velocity()
            vel = arc.velocity if from_file else derived
            records = [{'pos_m': pos[i], 'vel_ms': vel[i], 'vel_from_file': from_file,
                        'vel_edge': (not from_file) and not bool(centered[i]),
                        'offset_s': float(off[i])} for i in range(len(arc))]
            if not quiet:
                how = "positions+velocities" if from_file else "positions (velocities derived)"
                print(f"    {arc.satellite} ({name}) {path}: {len(arc)} {how}, "
                      f"{f.header.time_system.name} time, frame {f.header.frame}")
            out.setdefault(name, []).append(dict(
                source=str(path), satellite=arc.satellite, name=name, epoch0=arc.epoch0,
                offsets=off, records=records, frame=frame, frame_label=f.header.frame,
                time_system=f.header.time_system.name, sp3_arc=arc))
    if join:
        out = {name: [join_arcs(arcs, max_gap_s=max_gap_s, dedupe_tol_s=dedupe_tol_s)]
               for name, arcs in out.items()}
    return out


def join_arcs(arcs, *, max_gap_s=900.0, dedupe_tol_s=1.0):
    """Join one satellite's per-file arcs into a single time-ordered arc.

    Parameters:
      arcs          list of arcs from load_arcs (one satellite, any order).
      max_gap_s     largest allowed spacing, seconds, between consecutive
                    records of the joined arc.
      dedupe_tol_s  records of different files closer than this, seconds, are
                    the same epoch; only one is kept.

    Where files overlap, the record kept is the one further (in records) from
    the edge of its own file: an orbit solution is least accurate at its ends,
    and a derived velocity there comes from a one-sided window. The overlap is
    therefore split at its middle, each file contributing its interior.

    Returns an arc dict with the keys of load_arcs (epoch0 = the earliest
    epoch, offsets re-based to it, records copied with offset_s re-based,
    source = the first file, sp3_arc = None) plus sources (the files in time
    order), boundaries (record indices where the source file changes; the
    record at each index is the first from the new file) and record_source
    (numpy (N,) index into sources for every record).

    Raises ValueError for an empty list, for arcs of different satellites,
    for mixed frame labels or time systems (both labels are named), and for
    a gap longer than max_gap_s (its time and files are named).
    """
    arcs = list(arcs)
    if not arcs:
        raise ValueError("join_arcs: no arcs to join")
    if len({a['name'] for a in arcs}) > 1:
        raise ValueError(f"join_arcs: arcs of different satellites "
                         f"{sorted({a['name'] for a in arcs})}; join one satellite at a time")
    for key, what in (('frame_label', 'frame'), ('time_system', 'time system')):
        labels = {str(a.get(key)) for a in arcs}
        if len(labels) > 1:
            first = arcs[0]
            other = next(a for a in arcs if str(a.get(key)) != str(first.get(key)))
            raise ValueError(
                f"join_arcs: cannot join {first['name']} across files with different {what} "
                f"labels: '{first.get(key)}' in {first['source']} vs '{other.get(key)}' in "
                f"{other['source']}; positions in different frames or epochs in different "
                f"time systems would be mixed into one orbit")
    # Order the files by their first epoch; exact offsets of every record
    # relative to the earliest epoch (Epoch - Epoch is exact).
    ref = min((a['epoch0'] for a in arcs), key=lambda e: e - arcs[0]['epoch0'])
    shift = [float(a['epoch0'] - ref) for a in arcs]
    order = sorted(range(len(arcs)), key=lambda k: shift[k] + float(arcs[k]['offsets'][0]))
    arcs = [arcs[k] for k in order]
    shift = [shift[k] for k in order]
    cand = []    # (time, interior depth, file index, record index)
    for k, a in enumerate(arcs):
        n = len(a['records'])
        for i in range(n):
            cand.append((shift[k] + float(a['offsets'][i]), min(i, n - 1 - i), k, i))
    cand.sort(key=lambda c: (c[0], c[2]))
    kept = []
    g = 0
    while g < len(cand):
        h = g + 1
        while h < len(cand) and cand[h][0] - cand[g][0] <= dedupe_tol_s:
            h += 1
        # Deepest record wins; on a tie (both at an edge, e.g. the shared
        # midnight of two daily files) the later file, which continues past it.
        kept.append(max(cand[g:h], key=lambda c: (c[1], c[2])))
        g = h
    times = np.array([c[0] for c in kept])
    gaps = np.diff(times)
    if len(gaps) and gaps.max() > max_gap_s:
        j = int(np.argmax(gaps))
        a0, a1 = arcs[kept[j][2]], arcs[kept[j + 1][2]]
        raise ValueError(
            f"join_arcs: {a0['name']} has a {gaps[j]:.0f} s gap (limit {max_gap_s:.0f} s) after "
            f"{epoch_label(ref + float(times[j]))} between {a0['source']} and {a1['source']}; "
            f"a prediction across it would be scored against no data")
    t0 = float(times[0])
    records, src = [], []
    for c in kept:
        rec = dict(arcs[c[2]]['records'][c[3]])
        rec['offset_s'] = c[0] - t0
        records.append(rec)
        src.append(c[2])
    used = sorted(set(src), key=src.index)
    remap = {k: n for n, k in enumerate(used)}
    record_source = np.array([remap[k] for k in src], dtype=int)
    boundaries = [i for i in range(1, len(src)) if src[i] != src[i - 1]]
    first = arcs[0]
    return dict(source=first['source'], satellite=first['satellite'], name=first['name'],
                epoch0=ref + t0, offsets=times - t0, records=records, frame=first['frame'],
                frame_label=first.get('frame_label'), time_system=first['time_system'],
                sp3_arc=None, sources=[arcs[k]['source'] for k in used],
                boundaries=boundaries, record_source=record_source)


def epoch_label(epoch):
    """Return the epoch's calendar date and time as 'YYYY-MM-DDThh:mm:ss.sss'."""
    y, mo, d, h, mi, s = epoch.to_calendar()
    return f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{s:06.3f}"


def _key(epoch):
    """Hashable key for an epoch, rounded to the millisecond (duplicate-start check)."""
    return (epoch.tt_day, round(epoch.tt_seconds, 3))


def starts(arcs, interval_s, horizon_s, skip_edges=True, start_tol=None, end_tol=60.0):
    """Yield the start/truth pairs a validation script scores.

    Parameters:
      arcs        list of arcs from load_arcs (one satellite).
      interval_s  spacing of start times, seconds, counted from each arc's
                  first record.
      horizon_s   prediction length, seconds; the truth is the record nearest
                  start + horizon_s in the SAME arc.
      skip_edges  skip starts whose derived velocity came from a one-sided
                  window (file edges).
      start_tol   how far, in seconds, the nearest record may be from a grid
                  start time (default interval_s / 2).
      end_tol     how far, in seconds, the truth record may be from
                  start + horizon_s (default 60).

    Yields dicts with keys arc, start and end (record dicts), epoch and
    end_epoch (ballistic.Epoch), duration (exact seconds between the two
    records) and label (epoch_label of the start). A start time already scored
    from an earlier file is not yielded twice.
    """
    seen = set()
    start_tol = interval_s / 2 if start_tol is None else start_tol
    for arc in arcs:
        off = arc['offsets']
        if len(off) < 2:
            continue
        t = 0.0
        while t <= off[-1] - horizon_s:
            i = int(np.argmin(np.abs(off - t)))
            target = off[i] + horizon_s
            j = int(np.argmin(np.abs(off - target)))
            t += interval_s
            if abs(off[i] - (t - interval_s)) > start_tol or abs(off[j] - target) > end_tol:
                continue
            rec = arc['records'][i]
            if skip_edges and rec['vel_edge']:
                continue
            epoch = arc['epoch0'] + float(off[i])
            if _key(epoch) in seen:
                continue
            seen.add(_key(epoch))
            yield dict(arc=arc, start=rec, end=arc['records'][j], epoch=epoch,
                       end_epoch=arc['epoch0'] + float(off[j]),
                       duration=float(off[j] - off[i]), label=epoch_label(epoch))
