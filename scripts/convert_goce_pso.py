#!/usr/bin/env python3
"""Convert ESA GOCE Precise Science Orbit archives (SST_PSO_2) to SP3c.

GOCE was ESA's gravity-mapping satellite; its Precise Science Orbit (PSO) was computed by
the Astronomical Institute of the University of Bern (AIUB). Each
GO_CONS_SST_PSO_2__*.TGZ archive holds one XML data file (.DBL) with several AIUB products
for one day, plus a header file (.HDR). This script extracts the reduced-dynamic orbit
(product SST_PRD_2: an orbit fitted with a force model plus small estimated corrections;
one record every 10 s, in an Earth-fixed IGS frame (the header's label, IGS05 in the
committed files), GPS time, positions in km and
velocities in dm/s) and writes a plain SP3c file (SP3 = the standard text format for
precise orbits; "c" is its version) that validate_goce_reentry.py reads directly. The
kinematic orbit, covariances and attitude quaternions are skipped.

Inputs: one or more PSO archives (positional). An archive whose name contains "(" (a
browser's duplicate download, usually a saved login page) is skipped, as is any archive
that does not unpack or parse (a message goes to stderr).
Output: one SP3 file per archive in --out (created if missing).

The archive is streamed, never extracted. Output files are named after the day the orbit
covers (the archive name's second timestamp, as two-digit year + day of year) and the
archive's four-digit version: goce_pso_13295_0001.sp3 for 2013-10-22, version 0001.

Usage:
    python scripts/convert_goce_pso.py ARCHIVE.TGZ [ARCHIVE.TGZ ...] --out OUT_DIR
    e.g. python scripts/convert_goce_pso.py data/GOCE/*.TGZ --out data/GOCE/sp3
"""

import argparse
import datetime as dt
import os
import re
import sys
import tarfile

_NAME = re.compile(r"GO_CONS_SST_PSO_2__\d{8}T\d{6}_(\d{8})T\d{6}_(\d{4})\.TGZ$")
_TAG = re.compile(rb"<(Year|Month|Day_of_Month|Hour|Minute|Second|X|Y|Z)>([^<]+)<")


def read_prd(fileobj):
    """Yield the SST_PRD_2 orbit records of a PSO .DBL file, one per epoch.

    fileobj: a binary file object over the XML .DBL (read line by line).
    Yields (epoch, position, velocity): epoch is the tuple (year, month, day, hour,
    minute, second) in GPS time, second a float; position is (x, y, z) in km and
    velocity (vx, vy, vz) in dm/s, both Earth-fixed. Raises ValueError if the file has
    no SST_PRD_2 section.
    """
    inside = False
    t, p, v, block = {}, [], [], None
    for line in fileobj:
        if not inside:
            if b"<SST_PRD_2>" in line:
                inside = True
            continue
        if b"</SST_PRD_2>" in line:
            return
        if b"<Position" in line:
            block = p
        elif b"<Velocity" in line and b"/>" not in line:
            block = v
        elif b"</SP3c_Record>" in line:
            yield ((t["Year"], t["Month"], t["Day_of_Month"], t["Hour"],
                    t["Minute"], t["Second"]), tuple(p), tuple(v))
            t, p, v, block = {}, [], [], None
        m = _TAG.search(line)
        if not m:
            continue
        key, val = m.group(1).decode(), m.group(2).decode()
        if key in ("X", "Y", "Z"):
            if block is not None and len(block) < 3:
                block.append(float(val))
        elif key == "Second":
            t[key] = float(val)
        else:
            t[key] = int(val)
    raise ValueError("no SST_PRD_2 section found")


def coordinate_system(header_text):
    """Return the frame label (e.g. "IGS05") of the SST_PRD_2 block in the .HDR text.

    header_text: the whole .HDR file as a string. Reads the <Coordinate_Sys> element
    that follows the first <SST_PRD_2> tag.
    """
    sec = header_text.split("<SST_PRD_2>", 1)[1]
    return re.search(r"<Coordinate_Sys>(\S+)</Coordinate_Sys>", sec).group(1)


def write_sp3(path, recs, frame):
    """Write the records as an SP3c file for satellite L15 (GOCE).

    path: output file. recs: list of (epoch, position km, velocity dm/s) as yielded by
    read_prd; at least two records, equally spaced (the spacing of the first two is
    written as the file's epoch interval). frame: the frame label for the header.
    """
    e0 = recs[0][0]
    d0 = dt.datetime(*e0[:5], int(e0[5]))
    gps0 = dt.datetime(1980, 1, 6)
    secs = (d0 - gps0).total_seconds()
    week, sow = int(secs // 604800), secs % 604800
    mjd = (d0 - dt.datetime(1858, 11, 17)).total_seconds() / 86400.0
    step = (dt.datetime(*recs[1][0][:5], int(recs[1][0][5])) - d0).total_seconds()
    lines = [
        (f"#cV{e0[0]:4d} {e0[1]:2d} {e0[2]:2d} {e0[3]:2d} {e0[4]:2d} {e0[5]:11.8f} "
         f"{len(recs):7d} ORBIT {frame:>5s} FIT AIUB"),
        f"## {week:4d} {sow:15.8f} {step:14.8f} {int(mjd):5d} {mjd % 1:15.13f}",
        "+    1   L15" + "  0" * 14,
        "++         0" + "  0" * 14,
        "%c L  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "/* GOCE reduced-dynamic Precise Science Orbit (ESA SST_PSO_2, IAPG/AIUB)",
        "/* converted by scripts/convert_goce_pso.py",
    ]
    for (y, mo, d, h, mi, s), (x, yy, z), (vx, vy, vz) in recs:
        lines.append(f"*  {y:4d} {mo:2d} {d:2d} {h:2d} {mi:2d} {s:11.8f}")
        lines.append(f"PL15{x:14.6f}{yy:14.6f}{z:14.6f} 999999.999999")
        lines.append(f"VL15{vx:14.6f}{vy:14.6f}{vz:14.6f} 999999.999999")
    lines.append("EOF")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def convert(tgz, out_dir):
    """Convert one PSO archive to SP3.

    tgz: path of the .TGZ archive; out_dir: directory for the output file.
    Returns (output path, number of records, frame label). Raises ValueError for an
    archive name that is not a PSO name or for malformed records, and tarfile.TarError
    for an archive that does not unpack.
    """
    m = _NAME.search(os.path.basename(tgz))
    if not m:
        raise ValueError(f"not a PSO archive name: {tgz}")
    day = dt.datetime.strptime(m.group(1), "%Y%m%d")
    out = os.path.join(out_dir, f"goce_pso_{day:%y}{day.timetuple().tm_yday:03d}"
                                f"_{m.group(2)}.sp3")
    with tarfile.open(tgz, "r:gz") as tar:
        members = {os.path.splitext(x.name)[1]: x for x in tar.getmembers()}
        hdr = tar.extractfile(members[".HDR"]).read().decode("ascii", "replace")
        frame = coordinate_system(hdr)
        recs = list(read_prd(tar.extractfile(members[".DBL"])))
    if len(recs) < 2 or any(len(p) != 3 or len(v) != 3 for _, p, v in recs):
        raise ValueError(f"{tgz}: malformed SST_PRD_2 records")
    write_sp3(out, recs, frame)
    return out, len(recs), frame


def main():
    """Parse the command line and convert every archive given."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archives", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for tgz in args.archives:
        if "(" in os.path.basename(tgz):      # browser duplicate (login page)
            continue
        try:
            out, n, frame = convert(tgz, args.out)
        except (tarfile.TarError, ValueError) as e:
            print(f"skip {os.path.basename(tgz)}: {e}", file=sys.stderr)
            continue
        print(f"{os.path.basename(out)}: {n} epochs, {frame}")


if __name__ == "__main__":
    main()
