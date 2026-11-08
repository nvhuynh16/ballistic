#!/usr/bin/env python3
"""
Download ILRS SP3 precise orbit files for geodetic satellites (LAGEOS, Etalon).

LAGEOS and Etalon are passive spheres covered in retroreflectors, tracked by satellite
laser ranging (SLR); the International Laser Ranging Service (ILRS) publishes their
precise orbits as SP3 files (SP3 = the standard text format for precise orbits). The
library validates medium-altitude prediction on them.

Data source:
  EDC, the EUROLAS Data Center at DGFI-TUM (anonymous FTP):
  ftp://edc.dgfi.tum.de/pub/slr/products/orbits/

Directory structure on EDC:
  /pub/slr/products/orbits/<satellite>/<YYMMDD>/<ac>.orb.<satellite>.<YYMMDD>.vNN.sp3.gz

Analysis centers: asi, bkg, cnes, dgfi, esa, gfz, ilrsa (combined), jcet, nsgf
We prefer 'ilrsa' (official ILRS combined product).

Satellite IDs in the SP3 files (as found in the downloaded files):
  LAGEOS-1: L51 (ILRS ID 7603901)
  LAGEOS-2: L52 (ILRS ID 9207002)
  Etalon-1: L53 (ILRS ID 8900103)
  Etalon-2: L54 (ILRS ID 8903903)

Options: --satellite {lageos1,lageos2,etalon1,etalon2} (required unless --analyze);
--date YYMMDD or YYYY-MM-DD (one weekly directory), else --weeks N (default 4) recent
weekly directories; --output DIR (default tests/data/sp3/lageos/ or
tests/data/sp3/etalon/, relative to the current directory); --ac CENTER (preferred
analysis center); --analyze FILE (report on an existing SP3 file, no download);
--list-dates (list the latest available directories, no download).
Output: decompressed SP3 files in the output directory (files already there are not
downloaded again) and a report on the first one. Exits 1 if nothing was downloaded.

Usage:
    python scripts/download_ilrs_sp3.py --satellite lageos1
    python scripts/download_ilrs_sp3.py --satellite lageos1 --date 260117
    python scripts/download_ilrs_sp3.py --satellite lageos1 --weeks 4
    python scripts/download_ilrs_sp3.py --satellite etalon1 --output tests/data/sp3/etalon/
    python scripts/download_ilrs_sp3.py --analyze tests/data/sp3/lageos/some_file.sp3
"""

import argparse
import ftplib
import gzip
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Satellite configuration
# ---------------------------------------------------------------------------

SATELLITE_INFO = {
    'lageos1': {
        'name': 'LAGEOS-1',
        'ftp_name': 'lageos1',
        'sp3_ids': ['L51', 'L07', 'L01', 'La1'],
        'default_output': 'tests/data/sp3/lageos/',
        'description': 'LAGEOS-1 (5890 km, retroreflector sphere)',
    },
    'lageos2': {
        'name': 'LAGEOS-2',
        'ftp_name': 'lageos2',
        'sp3_ids': ['L52', 'L02', 'La2'],
        'default_output': 'tests/data/sp3/lageos/',
        'description': 'LAGEOS-2 (5620 km, retroreflector sphere)',
    },
    'etalon1': {
        'name': 'Etalon-1',
        'ftp_name': 'etalon1',
        'sp3_ids': ['L53', 'E17', 'E01', 'Et1'],
        'default_output': 'tests/data/sp3/etalon/',
        'description': 'Etalon-1 (19100 km, retroreflector sphere)',
    },
    'etalon2': {
        'name': 'Etalon-2',
        'ftp_name': 'etalon2',
        'sp3_ids': ['L54', 'E07', 'E02', 'Et2'],
        'default_output': 'tests/data/sp3/etalon/',
        'description': 'Etalon-2 (19100 km, retroreflector sphere)',
    },
}

# Preferred analysis center order (ilrsa = official combined product)
PREFERRED_ACS = ['ilrsa', 'ilrsb', 'esa', 'gfz', 'dgfi', 'jcet', 'asi', 'cnes', 'bkg', 'nsgf']


# ---------------------------------------------------------------------------
# FTP download from EDC (DGFI-TUM)
# ---------------------------------------------------------------------------

def download_from_edc(satellite_key: str, date_codes: list, output_dir: Path,
                      ac_preference: str = None) -> list:
    """
    Download ILRS SP3 files from EDC anonymous FTP.

    Args:
        satellite_key: Key into SATELLITE_INFO
        date_codes: List of YYMMDD strings to download
        output_dir: Where to save files
        ac_preference: Preferred analysis center (default: ilrsa)

    For each date directory, one file is chosen: the --ac center if given, else the
    first center in PREFERRED_ACS that has a file, else the first SP3 file listed.
    Gzipped files are decompressed. A date that is unavailable or fails to download is
    reported and skipped.

    Returns list of downloaded file paths (pathlib.Path), including files that were
    already present; an empty list if the FTP connection fails.
    """
    ftp_host = 'edc.dgfi.tum.de'
    sat_info = SATELLITE_INFO[satellite_key]
    ftp_name = sat_info['ftp_name']
    base_path = f'/pub/slr/products/orbits/{ftp_name}'

    print(f"Connecting to {ftp_host}...")

    try:
        ftp = ftplib.FTP(ftp_host, timeout=30)
        ftp.login()  # anonymous
        print("Connected.")

        # List available date directories
        try:
            all_dirs = ftp.nlst(base_path)
        except ftplib.error_perm as e:
            print(f"Directory not found: {base_path}")
            print(f"  FTP error: {e}")
            ftp.quit()
            return []

        available_dates = set()
        for d in all_dirs:
            bn = os.path.basename(d)
            if len(bn) == 6 and bn.isdigit():
                available_dates.add(bn)

        print(f"Available date directories: {len(available_dates)}")

        downloaded = []
        output_dir.mkdir(parents=True, exist_ok=True)

        for date_code in date_codes:
            if date_code not in available_dates:
                print(f"  {date_code}: not available, skipping")
                continue

            dir_path = f'{base_path}/{date_code}'
            try:
                files = ftp.nlst(dir_path)
            except ftplib.error_perm:
                print(f"  {date_code}: cannot list directory, skipping")
                continue

            sp3_files = [f for f in files if '.sp3' in f.lower()]
            if not sp3_files:
                print(f"  {date_code}: no SP3 files found")
                continue

            # Select preferred analysis center
            target = None
            if ac_preference:
                for f in sp3_files:
                    if os.path.basename(f).lower().startswith(ac_preference):
                        target = f
                        break
            if target is None:
                for ac in PREFERRED_ACS:
                    for f in sp3_files:
                        if os.path.basename(f).lower().startswith(ac):
                            target = f
                            break
                    if target:
                        break
            if target is None:
                target = sp3_files[0]

            filename = os.path.basename(target)
            decompressed_name = filename[:-3] if filename.endswith('.gz') else filename
            local_decompressed = output_dir / decompressed_name

            # Skip if already downloaded
            if local_decompressed.exists():
                print(f"  {date_code}: already exists ({decompressed_name})")
                downloaded.append(local_decompressed)
                continue

            local_path = output_dir / filename
            print(f"  {date_code}: downloading {filename}...")

            try:
                with open(local_path, 'wb') as fp:
                    ftp.retrbinary(f'RETR {target}', fp.write)

                # Decompress if gzipped
                if filename.endswith('.gz'):
                    with gzip.open(local_path, 'rb') as f_in:
                        with open(local_decompressed, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(local_path)
                    downloaded.append(local_decompressed)
                else:
                    downloaded.append(local_path)

            except ftplib.all_errors as e:
                print(f"    Download failed: {e}")
                if local_path.exists():
                    os.remove(local_path)

        ftp.quit()
        return downloaded

    except (ftplib.all_errors, OSError) as e:
        print(f"FTP connection error: {e}")
        return []


def list_available_dates(satellite_key: str, n_latest: int = 20) -> list:
    """Return the latest n_latest YYMMDD date directories on the EDC FTP for a satellite
    (sorted, oldest first); an empty list if the FTP connection fails."""
    ftp_host = 'edc.dgfi.tum.de'
    ftp_name = SATELLITE_INFO[satellite_key]['ftp_name']
    base_path = f'/pub/slr/products/orbits/{ftp_name}'

    try:
        ftp = ftplib.FTP(ftp_host, timeout=30)
        ftp.login()
        all_dirs = ftp.nlst(base_path)
        ftp.quit()

        dates = []
        for d in all_dirs:
            bn = os.path.basename(d)
            if len(bn) == 6 and bn.isdigit():
                dates.append(bn)

        return sorted(dates)[-n_latest:]
    except (ftplib.all_errors, OSError) as e:
        print(f"FTP error: {e}")
        return []


def date_to_yymmdd(dt: datetime) -> str:
    """Convert datetime to YYMMDD format used by ILRS EDC."""
    return dt.strftime('%y%m%d')


def get_recent_date_codes(n_weeks: int, reference_date: datetime = None) -> list:
    """
    Return YYMMDD date codes for the n_weeks most recent weekly directories.

    ILRS products are weekly, with directories named by the Saturday start date. The
    codes are the most recent Saturday on or before reference_date (default: now) and
    the Saturdays before it, newest first.
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Go back to the most recent Saturday
    days_since_saturday = (reference_date.weekday() + 2) % 7
    last_saturday = reference_date - timedelta(days=days_since_saturday)

    codes = []
    for i in range(n_weeks):
        dt = last_saturday - timedelta(weeks=i)
        codes.append(date_to_yymmdd(dt))

    return codes


# ---------------------------------------------------------------------------
# SP3 file analysis
# ---------------------------------------------------------------------------

def analyze_sp3(sp3_path: Path, satellite_key: str = None) -> None:
    """Print a report on an SP3 file: epochs, time span, velocity records, satellite IDs
    and positions per satellite; the IDs expected for satellite_key (if given) are
    marked as targets."""
    print(f"\n{'='*60}")
    print(f"SP3 File Analysis: {sp3_path.name}")
    print(f"{'='*60}")

    sat_ids = set()
    n_epochs = 0
    n_velocity = 0
    first_epoch = None
    last_epoch = None

    with open(sp3_path, 'r', encoding='utf-8') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.startswith('+') and len(line) > 9:
                    for i in range(9, min(len(line) - 1, 60), 3):
                        sat_id = line[i:i+3].strip()
                        if sat_id and sat_id != '0' and sat_id != '00':
                            sat_ids.add(sat_id)
                elif line.startswith('*'):
                    in_header = False

            if line.startswith('*'):
                n_epochs += 1
                parts = line.split()
                if len(parts) >= 6:
                    y, m, d, h, mi = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                    epoch_str = f"{y:04d}-{m:02d}-{d:02d} {h:02d}:{mi:02d}"
                    if first_epoch is None:
                        first_epoch = epoch_str
                    last_epoch = epoch_str
            elif line.startswith('V'):
                n_velocity += 1

    print(f"Epochs: {n_epochs}")
    if first_epoch:
        print(f"Time span: {first_epoch} to {last_epoch}")
    print(f"Velocity records: {n_velocity} ({'position+velocity' if n_velocity > 0 else 'position only'})")
    print(f"\nAll satellite IDs found ({len(sat_ids)}):")

    sorted_ids = sorted(sat_ids)
    for sid in sorted_ids:
        marker = ''
        if satellite_key:
            info = SATELLITE_INFO.get(satellite_key, {})
            if sid in info.get('sp3_ids', []):
                marker = '  <-- TARGET'
        print(f"  {sid}{marker}")

    if satellite_key:
        info = SATELLITE_INFO.get(satellite_key, {})
        found = [sid for sid in info.get('sp3_ids', []) if sid in sat_ids]
        if found:
            print(f"\nTarget satellite found: {found}")
        else:
            print(f"\nWARNING: Target satellite IDs {info.get('sp3_ids', [])} not found!")
            print("  Use --sat-id to specify the correct ID from the list above")

    # Count records per satellite
    print("\nRecords per satellite:")
    sat_counts = {}
    with open(sp3_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('P') and len(line) > 3:
                sid = line[1:4].strip()
                sat_counts[sid] = sat_counts.get(sid, 0) + 1
    for sid in sorted(sat_counts.keys()):
        marker = ''
        if satellite_key:
            info = SATELLITE_INFO.get(satellite_key, {})
            if sid in info.get('sp3_ids', []):
                marker = '  <-- TARGET'
        print(f"  {sid}: {sat_counts[sid]} positions{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse the command line and analyze, list or download as requested."""
    parser = argparse.ArgumentParser(
        description='Download ILRS SP3 precise orbit files for geodetic satellites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EDC directory structure:
  /pub/slr/products/orbits/<satellite>/<YYMMDD>/
  e.g., /pub/slr/products/orbits/lageos1/260117/

Analysis centers: asi, bkg, cnes, dgfi, esa, gfz, ilrsa (combined), jcet, nsgf
Default preference: ilrsa (official ILRS combined product)

Examples:
    python scripts/download_ilrs_sp3.py --satellite lageos1
    python scripts/download_ilrs_sp3.py --satellite lageos1 --weeks 8
    python scripts/download_ilrs_sp3.py --satellite lageos1 --date 260117
    python scripts/download_ilrs_sp3.py --satellite etalon1 --output tests/data/sp3/etalon/
    python scripts/download_ilrs_sp3.py --analyze tests/data/sp3/lageos/some_file.sp3
    python scripts/download_ilrs_sp3.py --satellite lageos1 --list-dates
        """,
    )
    parser.add_argument('--satellite', choices=list(SATELLITE_INFO.keys()),
                        help='Target satellite')
    parser.add_argument('--date', type=str, default=None,
                        help='Specific date YYMMDD or YYYY-MM-DD')
    parser.add_argument('--weeks', type=int, default=4,
                        help='Number of recent weeks to download (default: 4)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: based on satellite)')
    parser.add_argument('--ac', type=str, default=None,
                        help='Preferred analysis center (default: ilrsa)')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Analyze an existing SP3 file (no download)')
    parser.add_argument('--list-dates', action='store_true',
                        help='List available dates on FTP (no download)')

    args = parser.parse_args()

    # Analyze-only mode
    if args.analyze:
        sp3_path = Path(args.analyze)
        if not sp3_path.exists():
            print(f"ERROR: File not found: {sp3_path}")
            sys.exit(1)
        analyze_sp3(sp3_path, args.satellite)
        return

    if args.satellite is None:
        parser.print_help()
        print("\nERROR: --satellite is required (unless using --analyze)")
        sys.exit(1)

    sat_info = SATELLITE_INFO[args.satellite]
    print(f"Target: {sat_info['description']}")
    print(f"Expected SP3 IDs: {sat_info['sp3_ids']}")

    # List-dates mode
    if args.list_dates:
        print("\nQuerying available dates on EDC FTP...")
        dates = list_available_dates(args.satellite, n_latest=30)
        if dates:
            print(f"Latest {len(dates)} available date directories:")
            for d in dates:
                # Convert YYMMDD to readable date
                try:
                    dt = datetime.strptime(d, '%y%m%d')
                    print(f"  {d}  ({dt.strftime('%Y-%m-%d')})")
                except ValueError:
                    print(f"  {d}")
        else:
            print("No dates found or FTP unavailable")
        return

    # Determine date codes to download
    if args.date:
        if len(args.date) == 6 and args.date.isdigit():
            date_codes = [args.date]
        else:
            dt = datetime.strptime(args.date, '%Y-%m-%d')
            date_codes = [date_to_yymmdd(dt)]
        print(f"Requesting date: {date_codes[0]}")
    else:
        date_codes = get_recent_date_codes(args.weeks)
        print(f"Requesting {args.weeks} recent weeks: {date_codes}")

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(sat_info['default_output'])

    print(f"Output directory: {output_dir}")

    # Download
    downloaded = download_from_edc(args.satellite, date_codes, output_dir, args.ac)

    if not downloaded:
        print("\nNo files downloaded.")
        print("\nManual download from:")
        print(f"  ftp://edc.dgfi.tum.de/pub/slr/products/orbits/{sat_info['ftp_name']}/")
        sys.exit(1)

    print(f"\nDownloaded {len(downloaded)} file(s)")

    # Analyze first downloaded file
    analyze_sp3(downloaded[0], args.satellite)


if __name__ == '__main__':
    main()
