#!/usr/bin/env python3
"""
Download GOCE accelerometer-derived thermospheric density data from TU Delft.

GOCE was ESA's low-orbit gravity satellite. Its accelerometers measured the drag force,
from which Delft University of Technology (TU Delft) derived the air density along the
orbit (kg/m^3). The library uses these densities as measured truth for drag at low
altitude (parse_density_table / from_accelerometer_density).

Source: TU Delft v02 dataset
  https://thermosphere.tudelft.nl/data/data/version_02/GOCE_data/

Downloads density data for GOCE's ballistic decay phase (Oct-Nov 2013)
when the spacecraft descended from ~236 km to reentry at ~142 km.

Files:
  GO_DNS_ACC_2013_10_v02.zip  — October 2013
  GO_DNS_ACC_2013_11_v02.zip  — November 2013

Output: tests/data/goce_density/ (the zip archives and their extracted files, flat;
an archive already present is not downloaded again), then a summary of every data file
found there (record count, altitude and density ranges). No command-line options.

Usage:
    python scripts/download_goce_density.py
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path

BASE_URL = "https://thermosphere.tudelft.nl/data/data/version_02/GOCE_data/"

FILES = [
    "GO_DNS_ACC_2013_10_v02.zip",
    "GO_DNS_ACC_2013_11_v02.zip",
]

def download_file(url: str, dest: Path) -> bool:
    """Download url to the file dest, printing progress.

    Returns True on success and False on any error (the error is printed); a partly
    written dest may remain after a failure.
    """
    print(f"  Downloading: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Ballistic/1.0'})
        with urllib.request.urlopen(req, timeout=120) as response:
            total = response.headers.get('Content-Length')
            if total:
                total = int(total)
            with open(dest, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end='', flush=True)
                print()
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> list:
    """Extract every file of the zip archive into dest_dir, dropping folder names.

    Returns the list of extracted file paths (pathlib.Path).
    """
    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            # Extract only data files (skip directories)
            if name.endswith('/'):
                continue
            # Extract to flat directory
            basename = os.path.basename(name)
            if not basename:
                continue
            target = dest_dir / basename
            with zf.open(name) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
            print(f"    Extracted: {basename}")
    return extracted


def parse_density_file(filepath: Path) -> dict:
    """Read a TU Delft density file and return summary statistics.

    Comment lines (starting with '#' or '%') and blank lines are skipped; densities
    outside (0, 1] kg/m^3 (fill values) are not counted. Returns a dict with 'count'
    (number of rows counted), 'alt_range' (min, max altitude, in the file's units) and
    'rho_range' (min, max density, kg/m^3); both ranges are (0, 0) when nothing is
    counted. Raises ValueError for a data row whose altitude or density does not parse.
    """
    count = 0
    alt_min = float('inf')
    alt_max = float('-inf')
    rho_min = float('inf')
    rho_max = float('-inf')

    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            # v2 rows: date time system alt(m) lon lat lst arglat density ...
            # (density is token 8, 0-based, not the longitude; a row that does
            # not parse raises)
            parts = line.split()
            try:
                alt = float(parts[3])
                rho = float(parts[8])
            except (ValueError, IndexError):
                raise ValueError(f"{filepath}: not a density row: {line[:80]!r}") from None
            if rho <= 0 or rho > 1:
                continue
            count += 1
            alt_min = min(alt_min, alt)
            alt_max = max(alt_max, alt)
            rho_min = min(rho_min, rho)
            rho_max = max(rho_max, rho)

    return {
        'count': count,
        'alt_range': (alt_min, alt_max) if count > 0 else (0, 0),
        'rho_range': (rho_min, rho_max) if count > 0 else (0, 0),
    }


def main():
    """Download and extract both monthly archives, then print the summary."""
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    dest_dir = project_root / "tests" / "data" / "goce_density"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("GOCE Accelerometer Density Download")
    print(f"{'='*50}")
    print("Source: TU Delft v02")
    print(f"Destination: {dest_dir}")
    print()

    all_data_files = []

    for filename in FILES:
        url = BASE_URL + filename
        zip_path = dest_dir / filename

        # Check if already downloaded
        if zip_path.exists():
            print(f"  Already exists: {filename}")
        else:
            if not download_file(url, zip_path):
                print(f"  FAILED: {filename}")
                continue

        # Extract
        print(f"  Extracting: {filename}")
        extracted = extract_zip(zip_path, dest_dir)
        all_data_files.extend(extracted)

    summarize(dest_dir, all_data_files)


def _print_stats(name: str, stats: dict) -> None:
    """Print one file's summary; the files store altitude in meters, shown in km."""
    print(f"\n  {name}:")
    print(f"    Records: {stats['count']}")
    print(f"    Altitude: {stats['alt_range'][0] / 1000.0:.1f} - "
          f"{stats['alt_range'][1] / 1000.0:.1f} km")
    print(f"    Density: {stats['rho_range'][0]:.2e} - {stats['rho_range'][1]:.2e} kg/m³")


def summarize(dest_dir: Path, extracted_files=()) -> int:
    """Print the summary of every density file in dest_dir and return the record count.

    Reads the .dat and .txt files, plus any file in extracted_files with another
    suffix (only when the .dat/.txt files held no records). Needs no network, so it
    can be run on the committed density slice alone.
    """
    print(f"\n{'='*50}")
    print("Validation Summary:")
    total_records = 0

    for data_file in sorted(dest_dir.glob("*.dat")) + sorted(dest_dir.glob("*.txt")):
        stats = parse_density_file(data_file)
        if stats['count'] > 0:
            total_records += stats['count']
            _print_stats(data_file.name, stats)

    # Also check for any extracted files
    for data_file in sorted(dest_dir.iterdir()):
        if data_file.suffix in ('.zip', '.py'):
            continue
        if data_file.name.startswith('.'):
            continue
        if data_file not in extracted_files and data_file.suffix not in ('.dat', '.txt'):
            continue
        stats = parse_density_file(data_file)
        if stats['count'] > 0 and total_records == 0:
            total_records += stats['count']
            _print_stats(data_file.name, stats)

    print(f"\nTotal records: {total_records}")
    if total_records == 0:
        print("WARNING: No density data found. Check file format.")
    else:
        print("SUCCESS: GOCE density data ready for validation.")
    return total_records


if __name__ == '__main__':
    main()
