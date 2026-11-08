"""
Reading tabulated text products into library objects.

Three kinds of file:

- a ballistic-coefficient table: beta = m / (Cd * A) (mass over drag
  coefficient times area, kg/m^2) as a function of altitude;
- an empirical-acceleration (EA) table: a small correction acceleration,
  fitted to observations, as a function of time;
- a measured air-density file (TU Delft accelerometer-derived densities for
  GOCE, CHAMP or GRACE).

These Python helpers read the files and pass arrays to C++ constructors, so
all file reading stays in Python. Downloading and locating the external data
files is ``ballistic.loader``. A row that does not parse raises ValueError
naming it (never skipped silently); deliberate filters (density fill values)
are counted and announced with a DataSkippedWarning.
"""

import warnings as _warnings

import numpy as np

from . import _core
from .errors import DataSkippedWarning

# Maximum plausible ballistic coefficient (kg/m^2)
MAX_BETA_KGM2 = 10000


def load_beta_table(path: str) -> '_core.BetaProfile1D':
    """
    Read a CSV table of ballistic coefficient versus altitude into a
    BetaProfile1D (usable as the beta column of a DragEnvironment).

    CSV format::

        altitude_km,beta
        200,40.0
        300,45.0
        400,50.0

    Supports column names: altitude_km/altitude/alt_km/alt (km),
    altitude_m/alt_m (meters), beta/ballistic_coefficient/beta_kgm2.

    A header row naming both columns is required (it is what says the
    altitude unit). Blank lines and lines starting with ``#`` are ignored.

    Args:
        path: path to the CSV file.

    Returns:
        A BetaProfile1D (altitude in km, beta in kg/m^2).

    Raises:
        ValueError: no such header, a row that does not parse, a row with
            altitude < 0 or beta outside (0, 10000] kg/m^2, or no rows.
    """
    alt_km = []
    beta = []

    col_alt = -1
    col_beta = -1
    alt_in_meters = False

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Try to parse as header
            if col_alt < 0:
                parts = [p.strip().lower() for p in line.split(',')]
                for i, p in enumerate(parts):
                    if p in ('altitude_km', 'altitude', 'alt_km', 'alt'):
                        col_alt = i
                        alt_in_meters = False
                    elif p in ('altitude_m', 'alt_m'):
                        col_alt = i
                        alt_in_meters = True
                    elif p in ('beta', 'ballistic_coefficient', 'beta_kgm2'):
                        col_beta = i
                if col_alt >= 0 and col_beta >= 0:
                    continue
                # No assumed column order: the unit of the altitude column
                # is exactly what a header must say.
                raise ValueError(
                    f"{path}: no header naming an altitude column (altitude_km / "
                    f"altitude_m / ...) and a beta column")

            # Parse data
            parts = [p.strip() for p in line.split(',')]
            try:
                a = float(parts[col_alt])
                b = float(parts[col_beta])
            except (ValueError, IndexError):
                raise ValueError(f"{path}: not a data row: {line[:80]!r}") from None
            if alt_in_meters:
                a /= 1000.0
            if not (a >= 0 and 0 < b <= MAX_BETA_KGM2):
                # Never dropped silently.
                raise ValueError(
                    f"{path}: row {line[:80]!r} has altitude {a} km / beta {b} kg/m^2; "
                    f"need altitude >= 0 and 0 < beta <= {MAX_BETA_KGM2}")
            alt_km.append(a)
            beta.append(b)

    if not alt_km:
        raise ValueError(f"{path}: no beta rows")
    return _core.BetaProfile1D(alt_km, beta)


def load_empirical_acceleration(path: str) -> '_core.EmpiricalAcceleration':
    """
    Read a CSV table of correction accelerations versus time into an
    EmpiricalAcceleration.

    CSV format::

        # frame: gcrs
        key,ax,ay,az
        0.0,0.0,0.0,0.0
        100.0,1e-5,2e-6,0.0

    The ``key`` column is seconds from the propagation start epoch (or
    ``TrajectoryConfig.ea_reference_epoch``); ``ax, ay, az`` are in m/s^2.
    The optional ``# frame:`` line says which frame the axes are in:
    ``gcrs`` (inertial; default) or ``itrf`` (Earth-fixed). An optional
    ``# key_type: time`` line is accepted.

    Args:
        path: path to the CSV file.

    Returns:
        An EmpiricalAcceleration (interpolated between rows).

    Raises:
        ValueError: an empty file, an unknown frame, a key_type other than
            time, or a malformed row.
    """
    entries = []
    frame = _core.EmpiricalAccelFrame.GCRS
    header_found = False

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#'):
                # Parse metadata
                if ':' in line:
                    before, after = line[1:].split(':', 1)
                    before = before.strip().lower()
                    after = after.strip().lower()
                    if before == 'key_type':
                        if after not in ('time', ''):
                            raise ValueError(
                                f"load_empirical_acceleration: only key_type=time is "
                                f"supported (got '{after}'); altitude-keyed EA was "
                                f"removed.")
                    elif before == 'frame':
                        if after == 'itrf':
                            frame = _core.EmpiricalAccelFrame.ITRF
                        elif after == 'gcrs':
                            frame = _core.EmpiricalAccelFrame.GCRS
                        else:
                            # Never guess the frame.
                            raise ValueError(
                                f"{path}: unknown EA frame {after!r} "
                                f"(use 'gcrs' or 'itrf')")
                continue

            # Skip header
            if not header_found:
                header_found = True
                if line[0].isalpha():
                    continue

            # Parse data: key,ax,ay,az
            parts = [p.strip() for p in line.split(',')]
            try:
                k, ax, ay, az = (float(v) for v in parts[:4])
            except ValueError:
                raise ValueError(f"{path}: not a 'key,ax,ay,az' row: {line[:80]!r}") from None
            entries.append(_core.EmpiricalAccelEntry(k, np.array([ax, ay, az])))

    if not entries:
        # A zero EA here would let a typo'd path to an empty file propagate
        # with no correction and no symptom.
        raise ValueError(f"{path}: no EA rows")
    return _core.EmpiricalAcceleration.from_entries(entries, frame)


_DENSITY_TIME_SYSTEMS = ('UTC', 'GPS', 'TT', 'TAI')


def parse_density_table(path: str) -> 'tuple[list[float], list[float]]':
    """
    Read a TU Delft GOCE/CHAMP/GRACE measured-density file (v2 layout).

    Returns ``(mjd_utc, rho_kg_m3)``: two lists, the time of each row as a
    Modified Julian Date (MJD, days since 1858-11-17 00:00) in UTC, and the
    air density in kg/m^3.

    Each data row starts ``YYYY-MM-DD HH:MM:SS.sss <time system>`` followed
    by altitude, longitude, latitude, local time, argument of latitude and
    density. The time-system column is honored (UTC, GPS, TT or TAI; the
    returned MJDs are UTC); an unknown one raises.

    Rows whose density is not in (0, 1) kg/m^3 (fill values) are dropped,
    counted, and announced with a DataSkippedWarning. A row that does not
    parse raises ValueError naming the line. A file with no rows raises. An
    MJD-first layout is not read: it does not say its time scale.

    Use Epoch.from_mjd(mjd, TimeScale.UTC) to build Epochs for
    ballistic.drag.from_accelerometer_density.

    Args:
        path: path to the density file.

    Raises:
        ValueError: a row that does not parse, an unknown time system, or
            no density rows.
    """
    keys: list[float] = []
    densities: list[float] = []
    dropped = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            try:
                date, clock, system = parts[0], parts[1], parts[2]
                y, mo, d = (int(v) for v in date.split('-'))
                hh, mi, ss = clock.split(':')
                h, mi, sec = int(hh), int(mi), float(ss)
                density = float(parts[8])
            except (ValueError, IndexError):
                raise ValueError(
                    f"{path}:{line_no}: not a density row "
                    f"('YYYY-MM-DD HH:MM:SS.sss SYS alt lon lat lst arglat rho ...'): "
                    f"{line[:80]!r}") from None
            if system not in _DENSITY_TIME_SYSTEMS:
                raise ValueError(f"{path}:{line_no}: unknown time system {system!r} "
                                 f"(expected one of {_DENSITY_TIME_SYSTEMS})")
            if not 0.0 < density < 1.0:
                dropped += 1
                continue
            if system == 'UTC':
                mjd = _calendar_to_mjd(y, mo, d, h, mi, sec)
            else:
                ctor = {'GPS': _core.Epoch.from_gps, 'TT': _core.Epoch.from_tt,
                        'TAI': _core.Epoch.from_tai}[system]
                mjd = ctor(y, mo, d, h, mi, sec).mjd(_core.TimeScale.UTC)
            keys.append(mjd)
            densities.append(density)

    if not keys:
        raise ValueError(f"{path}: no density rows")
    if dropped:
        _warnings.warn(f"{path}: {dropped} row(s) with a non-physical density "
                       f"(fill values) dropped; {len(keys)} kept",
                       DataSkippedWarning, stacklevel=2)
    return keys, densities


def _calendar_to_mjd(y: int, m: int, d: int, h: int, mi: int, s: float) -> float:
    """Convert a calendar date and time to a Modified Julian Date (Meeus algorithm)."""
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    JD = (int(365.25 * (y + 4716))
          + int(30.6001 * (m + 1))
          + d + B - 1524.5
          + h / 24.0 + mi / 1440.0 + s / 86400.0)
    return JD - 2400000.5
