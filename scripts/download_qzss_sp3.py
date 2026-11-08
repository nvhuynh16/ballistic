#!/usr/bin/env python3
"""
Download QZSS SP3 precise orbit files from IGS MGEX archive.

QZSS is Japan's Quasi-Zenith Satellite System. SP3 is the standard text format for
precise orbits; the International GNSS Service (IGS) publishes multi-constellation
orbits through its Multi-GNSS Experiment (MGEX). The script downloads one day of CODE
(Center for Orbit Determination in Europe) final MGEX orbits, decompresses it and
reports the QZSS satellites and whether the file has velocity records.

Most QZSS satellites fly inclined geosynchronous "quasi-zenith" orbits
(IGSO): perigee altitude ~32,000 km, apogee ~40,000 km, inclination
~40-43 deg, period one sidereal day; QZS-3 is geostationary.

Data sources (tried in this order):
- IGN mirror (no login): http://igs.ign.fr/pub/igs/products/mgex/
- NASA CDDIS: https://cddis.nasa.gov/archive/gnss/products/mgex/
  (requires an Earthdata login)

SP3 products containing QZSS:
- COD0MGXFIN: CODE final (GPS+GLO+GAL+BDS+QZS)
- GFZ0MGXRAP: GFZ rapid (multi-GNSS)
- WUM0MGXFIN: Wuhan final (multi-GNSS)

Inputs: a GPS week and day (--week, --day 0-6, default 0) or a date (--date
YYYY-MM-DD); with neither, GPS week 2294 day 0. Output: the decompressed SP3 file in
--output (default tests/data/sp3/qzss/ in this repository); a file already there is
not downloaded again. Exits 1 if no source delivers the file.

The CDDIS fallback needs an Earthdata login. The script never takes credentials on
the command line (they would land in the shell history and the process list): put
them in ~/.netrc, which the HTTP library reads on its own:

    machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD

and make the file private (chmod 600 ~/.netrc). Create an account at
https://urs.earthdata.nasa.gov/users/new if you have none.

Usage:
    python scripts/download_qzss_sp3.py [--week N [--day D] | --date YYYY-MM-DD]
        [--output DIR]
    e.g. python scripts/download_qzss_sp3.py --week 2294 --day 0
"""

import argparse
import os
import sys
import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'tests' / 'data' / 'sp3' / 'qzss'

try:
    import requests
except ImportError:
    print("requests library not found. Install with: uv pip install requests")
    sys.exit(1)


def gps_week_to_date(gps_week: int, day_of_week: int = 0) -> datetime:
    """Return the date of day `day_of_week` (0 = Sunday) of GPS week `gps_week`.

    GPS weeks count from the GPS time origin, Sunday 1980-01-06.
    """
    gps_epoch = datetime(1980, 1, 6)  # GPS epoch
    return gps_epoch + timedelta(weeks=gps_week, days=day_of_week)


def date_to_gps_week(dt: datetime) -> tuple:
    """Return (GPS week, day of week 0-6) of a datetime."""
    gps_epoch = datetime(1980, 1, 6)
    delta = dt - gps_epoch
    gps_week = delta.days // 7
    day_of_week = delta.days % 7
    return gps_week, day_of_week


def download_sp3_cddis(gps_week: int, day_of_week: int, output_dir: Path) -> Path:
    """
    Download one day's CODE MGEX SP3 file from CDDIS and decompress it.

    CDDIS refuses anonymous requests; the Earthdata login is taken from ~/.netrc
    (machine urs.earthdata.nasa.gov) by the HTTP library. Nothing is passed in.
    If the long-named file is missing (HTTP 404) the old short name
    cod<week><day>.sp3.Z is tried; that file is saved still compressed.
    Returns the path of the file in output_dir, or None if the download failed
    (the reason is printed).
    """
    # Try CODE final first (has QZSS since GPS week 1962)
    # File naming: COD0MGXFIN_YYYYDDDHHMM_01D_05M_ORB.SP3.gz

    dt = gps_week_to_date(gps_week, day_of_week)
    year = dt.year
    doy = dt.timetuple().tm_yday  # Day of year

    # Long file-name convention: <center><product>_<YYYYDDDHHMM>_01D_05M_ORB.SP3.gz
    filename = f"COD0MGXFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3.gz"
    url = f"https://cddis.nasa.gov/archive/gnss/products/mgex/{gps_week}/{filename}"

    print(f"Attempting to download: {filename}")
    print(f"URL: {url}")
    print(f"Date: {dt.strftime('%Y-%m-%d')}")

    output_path = output_dir / filename
    output_path_unzipped = output_dir / filename.replace('.gz', '')

    # Check if already downloaded
    if output_path_unzipped.exists():
        print(f"File already exists: {output_path_unzipped}")
        return output_path_unzipped

    # The session carries no credentials of its own: requests reads ~/.netrc
    # (trust_env is on by default) and answers the Earthdata login challenge
    # from it. A 401 therefore means the entry is missing or wrong.
    session = requests.Session()

    try:
        response = session.get(url, stream=True, timeout=60)

        if response.status_code == 401:
            print("\nCDDIS needs an Earthdata login and none was found in ~/.netrc.")
            print("Add:  machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASSWORD")
            print("(chmod 600 ~/.netrc). Create an account at https://urs.earthdata.nasa.gov/users/new")
            return None

        if response.status_code == 404:
            print(f"File not found: {filename}")
            # Try alternative naming convention (older format)
            filename_old = f"cod{gps_week}{day_of_week}.sp3.Z"
            url_old = f"https://cddis.nasa.gov/archive/gnss/products/{gps_week}/{filename_old}"
            print(f"Trying older format: {url_old}")
            response = session.get(url_old, stream=True, timeout=60)

            if response.status_code != 200:
                print(f"Download failed with status: {response.status_code}")
                return None

            output_path = output_dir / filename_old
            output_path_unzipped = output_dir / filename_old.replace('.Z', '')

        response.raise_for_status()

        # Save compressed file
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded: {output_path}")

        # Decompress
        if output_path.suffix == '.gz':
            with gzip.open(output_path, 'rb') as f_in:
                with open(output_path_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(output_path)
            print(f"Decompressed: {output_path_unzipped}")
            return output_path_unzipped

        return output_path

    except requests.exceptions.RequestException as e:
        print(f"Download error: {e}")
        return None


def download_sp3_ign(gps_week: int, day_of_week: int, output_dir: Path) -> Path:
    """
    Download one day's CODE MGEX SP3 file from the IGN mirror (no login) and decompress it.

    Returns the path of the decompressed file in output_dir, or None if the download
    failed (the reason is printed).
    """
    dt = gps_week_to_date(gps_week, day_of_week)
    year = dt.year
    doy = dt.timetuple().tm_yday

    # IGN mirror URL
    filename = f"COD0MGXFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3.gz"
    url = f"http://igs.ign.fr/pub/igs/products/mgex/{gps_week}/{filename}"

    print(f"Attempting IGN mirror: {filename}")
    print(f"URL: {url}")

    output_path = output_dir / filename
    output_path_unzipped = output_dir / filename.replace('.gz', '')

    if output_path_unzipped.exists():
        print(f"File already exists: {output_path_unzipped}")
        return output_path_unzipped

    try:
        response = requests.get(url, stream=True, timeout=60)

        if response.status_code != 200:
            print(f"IGN mirror failed: {response.status_code}")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded from IGN: {output_path}")

        # Decompress
        with gzip.open(output_path, 'rb') as f_in:
            with open(output_path_unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(output_path)
        print(f"Decompressed: {output_path_unzipped}")
        return output_path_unzipped

    except requests.exceptions.RequestException as e:
        print(f"IGN download error: {e}")
        return None


def extract_qzss_satellites(sp3_path: Path) -> list:
    """Return the QZSS satellite IDs (starting with 'J') listed in the SP3 header '+' lines."""
    qzss_sats = []
    with open(sp3_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('+') and len(line) > 9:
                # Parse satellite IDs
                for i in range(9, len(line)-2, 3):
                    sat_id = line[i:i+3].strip()
                    if sat_id.startswith('J'):  # QZSS satellites start with J
                        qzss_sats.append(sat_id)
            elif line.startswith('*'):
                break  # End of header
    return list(set(qzss_sats))


def count_velocity_records(sp3_path: Path) -> int:
    """Return the number of velocity records (lines starting with 'V') in an SP3 file."""
    count = 0
    with open(sp3_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('V'):
                count += 1
    return count


def main():
    """Parse the command line, download the file (mirror first, then CDDIS) and report on it."""
    parser = argparse.ArgumentParser(description='Download QZSS SP3 data from MGEX')
    parser.add_argument('--week', type=int, help='GPS week number')
    parser.add_argument('--day', type=int, default=0, help='Day of GPS week (0-6)')
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format (alternative to week)')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory (default: tests/data/sp3/qzss in this repository)')

    args = parser.parse_args()

    # Determine GPS week
    if args.date:
        dt = datetime.strptime(args.date, '%Y-%m-%d')
        gps_week, day_of_week = date_to_gps_week(dt)
        print(f"Date {args.date} -> GPS week {gps_week}, day {day_of_week}")
    elif args.week is not None:
        gps_week = args.week
        day_of_week = args.day
    else:
        # Default to a recent week with known good data
        # GPS week 2294 = late December 2023
        gps_week = 2294
        day_of_week = 0
        print(f"Using default GPS week {gps_week}, day {day_of_week}")

    output_dir = Path(args.output)

    # Try IGN mirror first (no auth required)
    sp3_path = download_sp3_ign(gps_week, day_of_week, output_dir)

    if sp3_path is None:
        # Fall back to CDDIS
        sp3_path = download_sp3_cddis(gps_week, day_of_week, output_dir)

    if sp3_path is None:
        print("\nFailed to download SP3 file.")
        print("\nAlternative: Download manually from:")
        print(f"  https://cddis.nasa.gov/archive/gnss/products/mgex/{gps_week}/")
        print("\nLook for files like: COD0MGXFIN_*_ORB.SP3.gz")
        sys.exit(1)

    # Analyze downloaded file
    print("\n=== SP3 File Analysis ===")
    print(f"File: {sp3_path}")

    qzss_sats = extract_qzss_satellites(sp3_path)
    if qzss_sats:
        print(f"QZSS satellites found: {', '.join(sorted(qzss_sats))}")
    else:
        print("WARNING: No QZSS satellites found in this file!")

    vel_count = count_velocity_records(sp3_path)
    print(f"Velocity records: {vel_count}")
    if vel_count == 0:
        print("WARNING: No velocity data in this file (position only)")

    print(f"\nSP3 file ready at: {sp3_path}")


if __name__ == '__main__':
    main()
