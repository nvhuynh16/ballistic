"""
Downloading and locating the external data files the library needs.

The four files, all from public sources:

- Earth orientation parameters (EOP), IERS ``finals2000A.data``: measured
  irregularities of the Earth's rotation, needed to convert between the
  Earth-fixed and inertial frames to better than a few meters.
- Leap seconds, ``leap-seconds.list``: the steps between UTC and atomic time.
- Space weather, CelesTrak ``SW-All.csv``: the daily solar-flux index F10.7
  and the geomagnetic index Ap, which the atmosphere model needs for drag.
- Ocean tides, ``fes2004_Cnm-Snm.dat``: the FES2004 ocean-tide model from the
  IERS Conventions (2010), chapter 6.

``Loader`` (modeled on Skyfield's) downloads these on first use, keeps them
in the platform data directory, refreshes them when stale and loads them into
the library. Reading tabulated products (ballistic-coefficient tables,
correction-acceleration tables, density files) is ``ballistic.loaders``.

Dates inside the files are counted as MJD, the Modified Julian Date: days
since 1858-11-17 00:00 UTC.

Example:
    >>> import ballistic
    >>>
    >>> # Create loader (uses default data directory)
    >>> loader = ballistic.Loader()
    >>>
    >>> # Load data files (downloads if not present)
    >>> loader.timescale()      # leap-seconds.list and finals2000A.data
    >>> loader.space_weather()  # SW-All.csv
    >>>
    >>> # Or load individually
    >>> loader.load_eop()
    >>> loader.load_leap_seconds()
    >>> loader.load_space_weather()
"""

import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import ssl
import time

# Import the C++ bindings
from . import _core


# Where each file is downloaded from (the *_alt entries are fallbacks)
URLS = {
    'eop': 'https://datacenter.iers.org/data/latestVersion/finals.data.iau2000.txt',
    'eop_alt': 'https://maia.usno.navy.mil/ser7/finals2000A.data',
    'leap_seconds': 'https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list',
    'leap_seconds_alt': 'https://www.ietf.org/timezones/data/leap-seconds.list',
    'space_weather': 'https://celestrak.org/SpaceData/SW-All.csv',
    # FES2004 ocean-tide gravity-coefficient (Stokes) variations, IERS
    # Conventions (2010) Ch. 6 — a third-party model, downloaded rather than
    # embedded (~3.7 MB).
    'ocean_tides': 'https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat',
}

# Where each dataset in memory came from (path), updated on every successful
# load through Loader. The data itself lives in process-wide C++ stores: a
# later load replaces it, a failed one leaves it as it was.
LOADED_FROM: dict = {}

# The name each file is saved under in the data directory
FILENAMES = {
    'eop': 'finals2000A.data',
    'leap_seconds': 'leap-seconds.list',
    'space_weather': 'SW-All.csv',
    'ocean_tides': 'fes2004_Cnm-Snm.dat',
}

# Other names a file may have in the data directory (initialize() also
# auto-discovers these).
ALT_FILENAMES = {
    'eop': ['finals2000A.all'],
}


def data_files(directory: Optional[str] = None) -> dict:
    """Where the data files ``ballistic.load()`` uses are, if present.

    Returns ``{'eop': Path | None, 'leap_seconds': ..., 'space_weather':
    ..., 'ocean_tides': ...}`` for ``directory`` (default: the platform data
    directory, ``ballistic.get_data_directory()``). None means the file is
    not there. Copy these files to run offline; see also
    ``Loader().status()`` for their age and ``ballistic.loader.LOADED_FROM``
    for what is loaded now.
    """
    d = Path(directory).expanduser() if directory else Path(_core.get_data_directory())
    out = {}
    for file_type, name in FILENAMES.items():
        found = None
        for candidate in [name, *ALT_FILENAMES.get(file_type, [])]:
            if (d / candidate).is_file():
                found = d / candidate
                break
        out[file_type] = found
    return out


# Re-download a file older than this many days
UPDATE_INTERVALS = {
    'eop': 7,            # EOP data updated weekly
    'leap_seconds': 180, # Leap seconds announced 6 months in advance
    'space_weather': 1,  # Space weather updated daily
    'ocean_tides': 36500,  # a fixed model — fetched once
}


# Measured data normally ends this many days before today even in a fresh
# download (IERS finals: UT1 measured ~1 week behind; CelesTrak: yesterday).
_PUBLICATION_LAG_DAYS = {'eop': 14, 'space_weather': 2}


def _today_mjd() -> float:
    """The current time as a Modified Julian Date (days, UTC)."""
    return datetime.now(timezone.utc).timestamp() / 86400.0 + 40587.0


def measured_until_mjd(file_type: str, path) -> Optional[float]:
    """The last MEASURED (not predicted) day in a data file, as an MJD.

    Both files continue past the last measurement with predictions; this
    reads the content to find where the measurements end.

    Args:
        file_type: 'eop' or 'space_weather' (others return None).
        path: path to the file (str or Path).

    Returns:
        The MJD (float), or None for other file types, a missing or
        unreadable file, or a file without measured rows. EOP
        (finals2000A): the last row whose UT1 flag (column 58) is 'I'
        (IERS final/measured). Space weather (CelesTrak CSV): the last row
        with F10.7_DATA_TYPE 'OBS' (observed).

    Raises:
        ValueError: a damaged EOP row with an unreadable date.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        if file_type == 'eop':
            last = None
            with open(path, 'r', errors='replace', encoding='utf-8') as f:
                for line in f:
                    if len(line) > 57 and line[57] == 'I':
                        last = float(line[7:15])     # a damaged row raises
            return last
        if file_type == 'space_weather':
            import csv
            last = None
            with open(path, newline='', errors='replace', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('F10.7_DATA_TYPE', '').strip() == 'OBS':
                        last = row.get('DATE')
            if not last:
                return None
            return float(datetime.strptime(last.strip(), '%Y-%m-%d').date().toordinal() - 678576)
    except OSError:  # guard-exempt(EXCEPT-PASS): unreadable file -> the age check decides
        return None       # unreadable: the file's age decides instead
    return None


class Loader:
    """
    Downloads, caches and loads the external data files.

    Like Skyfield's Loader, it manages a data directory and downloads each
    file the first time it is needed, or again when the copy is stale (older
    than its update interval, or holding measurements that end too long
    ago). Problems that deserve a warning (a failed download, a stale cached
    copy) are collected in ``self.notes``; ``ballistic.load()`` issues them.

    Args:
        directory: Path to data directory. If None, uses platform default:
            - Windows: %LOCALAPPDATA%\\ballistic
            - Linux: ~/.local/share/ballistic
            - macOS: ~/Library/Application Support/ballistic
            The environment variable BALLISTIC_DATA_DIR, when set, names the
            directory instead (every platform).
        verbose: If True, print download progress messages.
        expire_days: re-download files older than this many days, for every
            file type. None (default) uses each type's own interval (EOP 7,
            leap seconds 180, space weather 1 day).
        offline: never download. Cached files are loaded as they are,
            however old; a missing file makes the load return False and adds
            a note.

    Example:
        >>> loader = ballistic.Loader()
        >>> loader.timescale()  # Downloads and loads EOP + leap seconds
        >>>
        >>> # Check what's loaded
        >>> print(loader.status())
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        verbose: bool = True,
        expire_days: Optional[int] = None,
        offline: bool = False,
    ):
        if directory is None:
            self.directory = Path(_core.get_data_directory())
        else:
            self.directory = Path(directory)

        self.verbose = verbose
        self.expire_days = expire_days
        self.offline = offline
        self.notes: list = []   # problems worth a warning (stale cache, failures)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create the data directory if it does not exist."""
        self.directory.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        """Print ``message`` if verbose mode is on."""
        if self.verbose:
            print(message)

    def _file_age_days(self, path: Path) -> Optional[float]:
        """The file's age from its modification time, in days; None if it does not exist."""
        if not path.exists():
            return None
        # POSIX timestamps, not naive local datetimes (those differ by an hour
        # across a daylight-saving change). A file written just now can carry
        # a modification time a few microseconds after time.time(), because
        # Windows stamps files from a different clock: that is age 0.
        age_s = time.time() - path.stat().st_mtime
        return max(age_s, 0.0) / 86400

    def _needs_update(self, path: Path, file_type: str) -> bool:
        """True if the file is missing, older than its interval, or its
        measured data ends too long ago (allowing for publication delay)."""
        if not path.exists():
            return True

        # Check expiration
        if self.expire_days is not None:
            max_age = self.expire_days
        else:
            max_age = UPDATE_INTERVALS.get(file_type, 30)

        age = self._file_age_days(path)
        if age is not None and age > max_age:
            return True
        # A recently written file can still hold old data (a copied file, a
        # mirror that stopped updating). Judge by the data's end, not mtime.
        end = measured_until_mjd(file_type, path)
        return end is not None and _today_mjd() - end > max_age + _PUBLICATION_LAG_DAYS.get(file_type, 0)

    def _download(
        self,
        url: str,
        dest: Path,
        description: str,
        fallback_url: Optional[str] = None
    ) -> bool:
        """
        Download ``url`` to ``dest`` (through a temporary file, so a failed
        download never leaves a partial file in place).

        Args:
            url: primary download URL.
            dest: destination path.
            description: human-readable name for progress messages.
            fallback_url: URL tried if the primary one fails.

        Returns:
            True if the download succeeded, False otherwise (network and file
            errors are caught and printed in verbose mode).
        """
        self._log(f"Downloading {description}...")
        self._log(f"  URL: {url}")

        # Create SSL context that works on most systems
        ssl_context = ssl.create_default_context()

        try:
            # Download to temporary file first
            temp_path = dest.with_suffix('.tmp')

            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'ballistic-python/0.1'}
            )

            with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
                total_size = response.headers.get('Content-Length')
                if total_size:
                    total_size = int(total_size)

                downloaded = 0
                chunk_size = 65536

                with open(temp_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if self.verbose and total_size:
                            pct = downloaded * 100 // total_size
                            print(f"\r  Progress: {pct}% ({downloaded:,} / {total_size:,} bytes)", end='')

                if self.verbose and total_size:
                    print()  # Newline after progress

            # Rename to final location
            temp_path.replace(dest)
            self._log(f"  Saved to: {dest}")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            self._log(f"  Failed: {e}")

            # Try fallback URL if available
            if fallback_url:
                self._log(f"  Trying fallback URL: {fallback_url}")
                return self._download(fallback_url, dest, description, fallback_url=None)

            return False

    def _get_path(self, file_type: str) -> Path:
        """The path of a data file type inside the data directory."""
        return self.directory / FILENAMES[file_type]

    # =========================================================================
    # Public API
    # =========================================================================

    def _load_resource(self, file_type: str, description: str,
                       load_fn, force_download: bool,
                       fallback_url_key: Optional[str] = None) -> bool:
        """Download a data file if needed, then load it.

        A failed refresh keeps and loads the cached copy (with a note).

        Args:
            file_type: key into the FILENAMES / URLS dictionaries.
            description: human-readable name for progress messages.
            load_fn: callable(path_str) -> bool that loads the file.
            force_download: download even if the cached copy is fresh.
            fallback_url_key: key in URLS of a fallback URL, if any.

        Returns:
            True if the data was loaded.
        """
        path = self._get_path(file_type)

        if self.offline:
            if not path.exists():
                self.notes.append(f"{description}: offline and no cached file at {path}")
                return False
        elif force_download or self._needs_update(path, file_type):
            fallback = URLS.get(fallback_url_key) if fallback_url_key else None
            if not self._download(URLS[file_type], path, description, fallback_url=fallback):
                if not path.exists():
                    self.notes.append(f"{description}: download failed and no cached file")
                    return False
                # A failed refresh must not discard a usable cached file.
                age = self._file_age_days(path)
                self.notes.append(
                    f"{description}: download failed; using the cached file "
                    f"({age:.0f} days old, {path})")

        ok = bool(load_fn(str(path)))
        if ok:
            LOADED_FROM[file_type] = str(path)
        return ok

    def load_file(self, file_type: str, path) -> bool:
        """Load one data file from a local path; never downloads.

        Args:
            file_type: 'eop', 'leap_seconds', 'space_weather' or 'ocean_tides'.
            path: path to the file (``~`` is expanded).

        Returns:
            True if loaded; False if the file is missing or cannot be parsed
            (a note says which).

        Raises:
            ValueError: an unknown ``file_type``.
        """
        loaders = {
            'eop': _core.load_eop,
            'leap_seconds': lambda p: _core.load_leap_seconds(p, verbose=False)[0],
            'space_weather': _core.load_space_weather,
            'ocean_tides': _core.load_ocean_tides,
        }
        if file_type not in loaders:
            raise ValueError(f"unknown data file type {file_type!r}; "
                             f"use one of {sorted(loaders)}")
        path = Path(path).expanduser()
        if not path.is_file():
            self.notes.append(f"{file_type}: no such file {path}")
            return False
        ok = bool(loaders[file_type](str(path)))
        if ok:
            LOADED_FROM[file_type] = str(path)
            self._log(f"Loaded {file_type} from {path}")
        else:
            self.notes.append(f"{file_type}: could not parse {path}")
        return ok

    def load_eop(self, force_download: bool = False) -> bool:
        """Load the Earth orientation parameters, downloading if needed.

        Returns True if loaded. ``force_download`` re-downloads a fresh copy.
        """
        return self._load_resource('eop', "Earth Orientation Parameters",
                                   _core.load_eop, force_download,
                                   fallback_url_key='eop_alt')

    def load_leap_seconds(self, force_download: bool = False) -> bool:
        """Load the leap-second list, downloading if needed. Returns True if loaded."""
        def _load(path: str) -> bool:
            success, error = _core.load_leap_seconds(path, verbose=False)
            if not success:
                self._log(f"  Parse error: {error}")
            return success

        return self._load_resource('leap_seconds', "Leap Seconds",
                                   _load, force_download,
                                   fallback_url_key='leap_seconds_alt')

    def load_space_weather(self, force_download: bool = False) -> bool:
        """Load the space-weather file (F10.7 and Ap), downloading if needed.

        Returns True if loaded.
        """
        return self._load_resource('space_weather', "Space Weather Data",
                                   _core.load_space_weather, force_download)

    def load_ocean_tides(self, force_download: bool = False) -> bool:
        """Load the FES2004 ocean-tide coefficients (downloaded once).

        Returns True if loaded.
        """
        return self._load_resource('ocean_tides', "Ocean Tides (FES2004)",
                                   _core.load_ocean_tides, force_download)

    def timescale(self, force_download: bool = False) -> bool:
        """
        Load time-related data (leap seconds and EOP).

        Initializes the library, then downloads (if needed) and loads:
        - leap-seconds.list (for UTC <-> atomic time TAI)
        - finals2000A.data (Earth rotation angle UT1 and polar motion, i.e.
          the Earth orientation parameters)

        Args:
            force_download: If True, download even if files exist.

        Returns:
            True if all data was loaded successfully.
        """
        # Initialize library first
        _core.initialize()

        # Leap seconds: the built-in table covers up to its own expiry date
        leap_ok = self.load_leap_seconds(force_download)

        # EOP: needed for meter-level frame conversions
        eop_ok = self.load_eop(force_download)

        return leap_ok and eop_ok

    def space_weather(self, force_download: bool = False) -> bool:
        """
        Load space weather (F10.7 and Ap) for atmospheric drag.

        Initializes the library, then downloads (if needed) and loads
        SW-All.csv from CelesTrak.

        Args:
            force_download: If True, download even if file exists.

        Returns:
            True if data was loaded successfully.
        """
        # Initialize library first
        _core.initialize()

        return self.load_space_weather(force_download)

    def status(self) -> dict:
        """
        Report on every data file.

        Returns:
            A dictionary keyed by file type ('eop', 'leap_seconds',
            'space_weather', 'ocean_tides'), each value a dictionary:
            - exists: whether the file exists locally
            - age_days: age of the file (modification time) in days, or None
            - measured_until_mjd: last MEASURED day in the file's content
              (EOP, space weather; None otherwise)
            - data_age_days: days from that day to now — what decides a
              refresh, since a new file can hold old data
            - loaded: whether that kind of data is loaded in the library now
            - path: full path to the file
        """
        result = {}

        for file_type in FILENAMES:
            path = self._get_path(file_type)
            age = self._file_age_days(path)

            # Check if loaded in library
            if file_type == 'eop':
                loaded = _core.get_init_status().eop_loaded
            elif file_type == 'leap_seconds':
                loaded = _core.get_leap_second_info().loaded_from_file
            elif file_type == 'space_weather':
                loaded = _core.SpaceWeatherHistory.last_measured_day() is not None
            else:
                loaded = bool(_core.ocean_tides_loaded())

            end = measured_until_mjd(file_type, path)
            result[file_type] = {
                'exists': path.exists(),
                'age_days': round(age, 1) if age is not None else None,
                'measured_until_mjd': end,
                'data_age_days': round(_today_mjd() - end, 1) if end is not None else None,
                'loaded': loaded,
                'path': str(path),
            }

        return result

    def clear_cache(self) -> None:
        """Delete every cached data file from the data directory (data already
        loaded in memory stays loaded)."""
        for file_type in FILENAMES:
            path = self._get_path(file_type)
            if path.exists():
                self._log(f"Removing: {path}")
                path.unlink()
