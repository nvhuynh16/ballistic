#!/usr/bin/env python3
"""Download the IGS satellite metadata file (SINEX) and list what it contains.

The file gives, for every GPS, GLONASS, Galileo, BeiDou, QZSS and IRNSS
satellite, its PRN assignments over time, its block (satellite type) and its
mass; scripts/_gnss_metadata.py reads it. It is public (no login).

Sources, tried in this order until one delivers a file whose first line
starts with %=SNX:
    https://files.igs.org/pub/station/general/igs_satellite_metadata.snx
    http://files.igs.org/pub/station/general/igs_satellite_metadata.snx
    ftp://igs-rf.ign.fr/pub/igs_satellite_metadata.snx

Inputs: --dest (output file, default data/igs/igs_satellite_metadata.snx in
this repository; the directory is created); --force re-downloads a file that
is already there; --check-excerpt PATH also checks that every data row of an
excerpt (e.g. tests/data/gnss/igs_satellite_metadata_excerpt.snx) appears
verbatim in the full file, inside the same block.

Output: the file at --dest; printed: the source that worked, the size, and
every block with its data-row count. Exits 1 when no source delivers a valid
file or the excerpt check fails.

Usage:
    python scripts/download_igs_metadata.py [--dest PATH] [--force]
        [--check-excerpt PATH]
    e.g. python scripts/download_igs_metadata.py --check-excerpt \\
        tests/data/gnss/igs_satellite_metadata_excerpt.snx
"""

import argparse
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO / "data" / "igs" / "igs_satellite_metadata.snx"
SOURCES = (
    "https://files.igs.org/pub/station/general/igs_satellite_metadata.snx",
    "http://files.igs.org/pub/station/general/igs_satellite_metadata.snx",
    "ftp://igs-rf.ign.fr/pub/igs_satellite_metadata.snx",
)


def fetch(url, timeout_s=60.0):
    """Return the bytes at url; raise OSError (incl. URLError) on failure."""
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read()


def download(dest):
    """Fetch the file from the first source that works and write it to dest.

    Returns the URL that worked. A candidate is accepted only when its first
    line starts with %=SNX; each failure is printed. Raises RuntimeError when
    every source fails.
    """
    failures = []
    for url in SOURCES:
        try:
            data = fetch(url)
        except (OSError, urllib.error.URLError) as exc:
            failures.append(f"{url}: {exc}")
            print(f"  failed: {url}: {exc}")
            continue
        if not data.startswith(b"%=SNX"):
            failures.append(f"{url}: first line is not a SINEX header ({data[:40]!r})")
            print(f"  failed: {url}: not a SINEX file")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            tmp.write(data)
        Path(tmp.name).replace(dest)
        return url
    raise RuntimeError("no source delivered the metadata file:\n  " + "\n  ".join(failures))


def block_rows(path):
    """Map each block name to its data lines (comments and blanks excluded)."""
    blocks, current = {}, None
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line.startswith("+"):
            current = line[1:].strip()
            blocks.setdefault(current, [])
        elif line.startswith("-"):
            current = None
        elif current is not None and line.strip() and not line.startswith("*"):  # guard-exempt(SP3-LOCAL-PARSER): SINEX
            blocks[current].append(line.rstrip())
    return blocks


def check_excerpt(full_path, excerpt_path):
    """List the excerpt rows that are not in the full file's same block.

    Returns a list of problem descriptions (empty when the excerpt is a subset).
    """
    full = block_rows(full_path)
    problems = []
    for block, rows in block_rows(excerpt_path).items():
        if block not in full:
            problems.append(f"block {block} is not in {full_path}")
            continue
        present = set(full[block])
        problems += [f"{block}: {row!r} not in the full file" for row in rows if row not in present]
    return problems


def main(argv=None):
    """Download (unless present), summarize, and optionally check an excerpt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"output file (default {DEFAULT_DEST})")
    parser.add_argument("--force", action="store_true",
                        help="download even if --dest already exists")
    parser.add_argument("--check-excerpt", type=Path, default=None, metavar="PATH",
                        help="check that every data row of this excerpt is in the full file")
    args = parser.parse_args(argv)

    if args.dest.exists() and not args.force:
        print(f"{args.dest} exists (use --force to re-download)")
    else:
        try:
            url = download(args.dest)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"downloaded {url}")
    if not args.dest.read_bytes().startswith(b"%=SNX"):
        print(f"{args.dest}: first line does not start with %=SNX", file=sys.stderr)
        return 1
    print(f"{args.dest}: {args.dest.stat().st_size} bytes")
    for block, rows in block_rows(args.dest).items():
        print(f"  +{block:<30s} {len(rows):5d} rows")

    if args.check_excerpt is not None:
        problems = check_excerpt(args.dest, args.check_excerpt)
        if problems:
            print(f"excerpt {args.check_excerpt}: {len(problems)} row(s) not in the full file:",
                  file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        n = sum(len(r) for r in block_rows(args.check_excerpt).values())
        print(f"excerpt {args.check_excerpt}: all {n} data rows are in the full file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
