#!/usr/bin/env python3
"""Convert GRACE-FO Level-1B GNV1B orbits (JPL) to SP3 for the drag campaigns.

GNV1B is the precise orbit that NASA's Jet Propulsion Laboratory (JPL) computes for
each GRACE-FO satellite: one position per second, in the Earth-fixed frame, time-tagged
in GPS time (continuous seconds past 2000-01-01 12:00:00 GPS, with no leap seconds). It
is part of the daily Level-1B archive, public at the
German Research Centre for Geosciences' (GFZ) Information System and Data Center (ISDC):
    https://isdc-data.gfz.de/grace-fo/Level-1B/JPL/INSTRUMENT/RL04/YYYY/
        gracefo_1B_YYYY-MM-DD_RL04.ascii.noLRI.tgz   (GNV1B_*_C_04.txt / _D_04.txt)

Input: one extracted GNV1B text file (one satellite, one day).
Output: an SP3 file (SP3 = the standard text format for precise orbits) with positions
only (validate_gracefo_drag.py then derives velocities from them), epoch labels in GPS
time like GFZ's Rapid Science Orbit (RSO) files, and satellite IDs L64 (GRACE-C =
GRACE-FO 1) / L65 (GRACE-D = GRACE-FO 2). Epochs are kept every --step seconds from
--start to --end.

Usage:
    python scripts/convert_gracefo_gnv1b.py GNV1B_FILE OUT.sp3 \\
        [--start HH:MM (default 00:00)] [--end HH:MM (default 24:00)] [--step SECONDS (default 30)]
"""

import argparse
import datetime as dt

GPS0 = dt.datetime(2000, 1, 1, 12)
SAT_ID = {'C': 'L64', 'D': 'L65'}


def read_gnv1b(path):
    """Read the data rows of a GNV1B text file.

    The rows after the "End of YAML" header line are returned as a list of tuples
    (gps_seconds: int, satellite letter 'C' or 'D', x, y, z), positions in meters in
    the Earth-fixed frame. Raises ValueError if a row is not in the Earth-fixed frame
    (frame letter other than 'E').
    """
    rows = []
    started = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if started:
                p = line.split()
                if p[2] != 'E':
                    raise ValueError(f"{path}: frame {p[2]!r}, expected Earth-fixed 'E'")
                rows.append((int(p[0]), p[1], float(p[3]), float(p[4]), float(p[5])))
            elif 'End of YAML' in line:
                started = True
    return rows


def main():
    """Parse the command line, select the requested epochs and write the SP3 file."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('gnv1b')
    ap.add_argument('out')
    ap.add_argument('--start', default='00:00', help='first epoch, HH:MM of the file day (GPS)')
    ap.add_argument('--end', default='24:00', help='last epoch, HH:MM (24:00 = next midnight)')
    ap.add_argument('--step', type=int, default=30, help='seconds between epochs')
    args = ap.parse_args()

    rows = read_gnv1b(args.gnv1b)
    sat = SAT_ID[rows[0][1]]
    day0 = (GPS0 + dt.timedelta(seconds=rows[0][0])).replace(hour=0, minute=0, second=0)

    def at(hhmm):
        """GPS seconds (past 2000-01-01 12:00) of time HH:MM on the file's day."""
        h, m = map(int, hhmm.split(':'))
        return int((day0 - GPS0).total_seconds()) + 3600 * h + 60 * m

    t_first, t_last = at(args.start), at(args.end)
    wanted = [r for r in rows if t_first <= r[0] <= t_last and (r[0] - t_first) % args.step == 0]
    if not wanted:
        raise SystemExit("no epochs in the requested window")

    first = GPS0 + dt.timedelta(seconds=wanted[0][0])
    with open(args.out, 'w', encoding='utf-8') as f:
        # Standard fixed-column SP3 header: the reader takes the frame label
        # from columns 47-51 of the first line and the epoch interval from the
        # "##" line, so these columns must be exact.
        f.write(f"#dP{first.year:4d} {first.month:2d} {first.day:2d} {first.hour:2d} "
                f"{first.minute:2d} {first.second:11.8f} {len(wanted):7d} ORBIT IGS14 FIT  JPL\n")
        since = first - dt.datetime(1980, 1, 6)
        week, sow = since.days // 7, (since.days % 7) * 86400 + since.seconds
        mjd = (first.date() - dt.date(1858, 11, 17)).days
        frac = (first.hour * 3600 + first.minute * 60 + first.second) / 86400
        f.write(f"## {week:4d} {sow:15.8f} {float(args.step):14.8f} {mjd:5d} {frac:15.13f}\n")
        f.write(f"+    1   {sat}\n")
        f.write("%c L  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
        f.write(f"/* converted from {args.gnv1b.rsplit('/', 1)[-1]} (JPL GNV1B, Earth-fixed)\n")
        for t, _, x, y, z in wanted:
            e = GPS0 + dt.timedelta(seconds=t)
            f.write(f"*  {e.year:4d} {e.month:2d} {e.day:2d} {e.hour:2d} {e.minute:2d} {e.second:11.8f}\n")
            f.write(f"P{sat}{x / 1000:14.6f}{y / 1000:14.6f}{z / 1000:14.6f} 999999.999999\n")
        f.write("EOF\n")
    print(f"{args.out}: {len(wanted)} epochs of {sat}, {wanted[0][0]}..{wanted[-1][0]} GPS s")


if __name__ == '__main__':
    main()
