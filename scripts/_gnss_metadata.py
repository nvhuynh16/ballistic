"""Read the IGS satellite metadata file (SINEX): PRN-to-SVN assignments, block, mass.

GNSS orbit validation needs to know which physical satellite (its SVN, the
space vehicle number) transmitted a given PRN on a given day, what kind of
satellite it is (its block, which decides the radiation-pressure model) and
how heavy it was. The International GNSS Service publishes all three in one
public SINEX file, igs_satellite_metadata.snx (download it with
scripts/download_igs_metadata.py; file description: Steigenberger and
Montenbruck, "IGS Satellite Metadata File Description", doi
10.57677/metadata-sinex).

Blocks in the file as downloaded (verbatim names, in file order):

    FILE/REFERENCE
    FILE/COMMENT
    SATELLITE/IDENTIFIER          SVN, COSPAR ID, NORAD number, block
    SATELLITE/PLANE               orbital plane and slot
    SATELLITE/PRN                 SVN/PRN assignment with validity interval
    SATELLITE/FREQUENCY_CHANNEL   GLONASS frequency channels
    SATELLITE/MASS                in-orbit mass with validity interval
    SATELLITE/COM                 center-of-mass offsets
    SATELLITE/ECCENTRICITY        antenna and laser-retroreflector offsets
    SATELLITE/TX_POWER            L-band transmit power

The file has NO macro-model, box-wing, panel-area or optical-property block:
it gives mass and block type, not the satellite's surfaces. Radiation-pressure
parameters therefore have to come from elsewhere, keyed on the block string.

This reader parses IDENTIFIER, PRN and MASS. The other blocks above are known
and left unread; any block not in that list is skipped and named in
GnssMetadata.ignored_blocks. A malformed row inside a parsed block raises
ValueError with the line number (nothing is silently dropped).

Times are SINEX "YYYY:DOY:SSSSS" (year, day of year, seconds of day; a
two-digit year 00-50 means 20xx, 51-99 means 19xx). "0000:000:00000" marks an
open end of an interval. They are read as UTC: assignments change on day
boundaries or at hand-over times stated to the second, so the few seconds
between UTC and GPS time never decide a lookup in practice. A validity
interval includes its start and excludes its end.

Library use (this module has no command line):

    from _gnss_metadata import read_sinex
    md = read_sinex("data/igs/igs_satellite_metadata.snx")
    svn = md.svn_of("G01", ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0))
    md.block_of(svn), md.mass_of(svn, epoch), md.cospar_of(svn)

A lookup that finds nothing raises ballistic.errors.ConfigError naming the
PRN or SVN, the epoch and the nearest intervals that do exist.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from ballistic.errors import ConfigError

# Blocks the file is documented to contain. Only the three read below are
# parsed; the others are recognized so that only genuinely unexpected blocks
# end up in ignored_blocks.
KNOWN_BLOCKS = (
    "FILE/REFERENCE",
    "FILE/COMMENT",
    "SATELLITE/IDENTIFIER",
    "SATELLITE/PLANE",
    "SATELLITE/PRN",
    "SATELLITE/FREQUENCY_CHANNEL",
    "SATELLITE/MASS",
    "SATELLITE/COM",
    "SATELLITE/ECCENTRICITY",
    "SATELLITE/TX_POWER",
)


def parse_sinex_time(text: str) -> _dt.datetime | None:
    """Turn a SINEX "YYYY:DOY:SSSSS" time into a naive UTC datetime.

    Returns None for the open-interval marker (year, day and seconds all zero).
    Raises ValueError for anything that is not three colon-separated integers
    or names an impossible day or second.
    """
    parts = text.strip().split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"not a SINEX time (YYYY:DOY:SSSSS): {text!r}")
    year, doy, sec = (int(p) for p in parts)
    if year == 0 and doy == 0 and sec == 0:
        return None
    if len(parts[0]) == 2:
        year += 2000 if year <= 50 else 1900
    if not 1 <= doy <= 366 or not 0 <= sec <= 86400:
        raise ValueError(f"SINEX time out of range (day {doy}, second {sec}): {text!r}")
    return _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy - 1, seconds=sec)


@dataclass(frozen=True)
class Interval:
    """A validity interval [start, end); None on either side means open."""

    start: _dt.datetime | None
    end: _dt.datetime | None

    def contains(self, t: _dt.datetime) -> bool:
        """True when t lies inside the interval (start included, end excluded)."""
        return (self.start is None or self.start <= t) and (self.end is None or t < self.end)

    def distance(self, t: _dt.datetime) -> float:
        """Seconds from t to the interval (0 when t is inside it)."""
        if self.start is not None and t < self.start:
            return (self.start - t).total_seconds()
        if self.end is not None and t >= self.end:
            return (t - self.end).total_seconds()
        return 0.0

    def __str__(self) -> str:
        """The interval as 'start .. end' in ISO form, 'open' for an open side."""
        a = self.start.isoformat() if self.start else "open"
        b = self.end.isoformat() if self.end else "open"
        return f"{a} .. {b}"


@dataclass(frozen=True)
class Identifier:
    """One SATELLITE/IDENTIFIER row: the satellite's catalog numbers and block."""

    svn: str
    cospar: str
    satcat: int
    block: str
    comment: str


@dataclass(frozen=True)
class PrnAssignment:
    """One SATELLITE/PRN row: SVN transmitted PRN during the interval."""

    svn: str
    prn: str
    interval: Interval


@dataclass(frozen=True)
class MassRecord:
    """One SATELLITE/MASS row: the satellite's mass [kg] during the interval."""

    svn: str
    mass_kg: float
    interval: Interval


def _to_utc(epoch) -> _dt.datetime:
    """Turn a ballistic.Epoch or a timezone-aware datetime into naive UTC.

    A naive datetime is refused: its time zone would be a guess.
    """
    if isinstance(epoch, _dt.datetime):
        if epoch.tzinfo is None or epoch.utcoffset() is None:
            raise ConfigError(
                f"the datetime {epoch.isoformat()} has no time zone; pass an aware datetime "
                "(e.g. tzinfo=datetime.timezone.utc) or a ballistic.Epoch")
        return epoch.astimezone(_dt.UTC).replace(tzinfo=None)
    if hasattr(epoch, "to_calendar"):
        y, mo, d, h, mi, s = epoch.to_calendar()
        return _dt.datetime(int(y), int(mo), int(d), int(h), int(mi)) + _dt.timedelta(seconds=s)
    raise TypeError(f"expected a ballistic.Epoch or an aware datetime, got {type(epoch).__name__}")


def _nearest(intervals, t, n=2) -> str:
    """The n intervals closest to t, formatted for an error message."""
    ranked = sorted(intervals, key=lambda item: item[1].distance(t))[:n]
    return "; ".join(f"{label} {iv}" for label, iv in ranked) or "none"


@dataclass
class GnssMetadata:
    """The parsed IDENTIFIER, PRN and MASS blocks of one metadata file."""

    path: Path
    identifiers: dict[str, Identifier] = field(default_factory=dict)
    prn_assignments: list[PrnAssignment] = field(default_factory=list)
    masses: list[MassRecord] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    ignored_blocks: list[str] = field(default_factory=list)

    def blocks_present(self) -> list[str]:
        """Every block name found in the file, in file order."""
        return list(self.blocks)

    def svn_of(self, prn: str, epoch) -> str:
        """The SVN that transmitted PRN (e.g. "G01") at the epoch.

        Raises ConfigError naming the PRN, the epoch and the nearest
        assignments when no assignment covers the epoch.
        """
        t = _to_utc(epoch)
        rows = [a for a in self.prn_assignments if a.prn == prn]
        hits = [a for a in rows if a.interval.contains(t)]
        if len(hits) == 1:
            return hits[0].svn
        if len(hits) > 1:
            raise ConfigError(
                f"PRN {prn} has {len(hits)} overlapping SVN assignments at {t.isoformat()} UTC "
                f"in {self.path}: " + "; ".join(f"{a.svn} {a.interval}" for a in hits))
        raise ConfigError(
            f"PRN {prn} has no SVN assignment at {t.isoformat()} UTC in {self.path}; "
            f"nearest assignments: {_nearest([(a.svn, a.interval) for a in rows], t)}. "
            "Refresh the file with scripts/download_igs_metadata.py --force if the epoch "
            "is recent")

    def _identifier(self, svn: str) -> Identifier:
        """The IDENTIFIER row of an SVN; ConfigError when the file has none."""
        try:
            return self.identifiers[svn]
        except KeyError:
            raise ConfigError(
                f"SVN {svn} has no SATELLITE/IDENTIFIER row in {self.path} "
                f"({len(self.identifiers)} satellites listed)") from None

    def block_of(self, svn: str) -> str:
        """The satellite block of an SVN, e.g. "GPS-IIF" or "BDS-3M-CAST"."""
        return self._identifier(svn).block

    def cospar_of(self, svn: str) -> str:
        """The COSPAR international designator of an SVN, e.g. "2011-036A"."""
        return self._identifier(svn).cospar

    def mass_of(self, svn: str, epoch) -> float:
        """The mass [kg] of SVN at the epoch, from the SATELLITE/MASS intervals.

        Raises ConfigError naming the SVN, the epoch and the nearest mass
        intervals when none covers the epoch.
        """
        t = _to_utc(epoch)
        rows = [m for m in self.masses if m.svn == svn]
        hits = [m for m in rows if m.interval.contains(t)]
        if len(hits) == 1:
            return hits[0].mass_kg
        if len(hits) > 1:
            raise ConfigError(
                f"SVN {svn} has {len(hits)} overlapping mass records at {t.isoformat()} UTC "
                f"in {self.path}: " + "; ".join(f"{m.mass_kg} kg {m.interval}" for m in hits))
        raise ConfigError(
            f"SVN {svn} has no mass record at {t.isoformat()} UTC in {self.path}; nearest: "
            f"{_nearest([(f'{m.mass_kg} kg', m.interval) for m in rows], t)}")


def _fields(line: str, n: int, lineno: int, block: str) -> list[str]:
    """Split a data row into at least n whitespace-separated fields, or raise."""
    parts = line.split()
    if len(parts) < n:
        raise ValueError(f"line {lineno} of {block} has {len(parts)} fields, expected "
                         f"at least {n}: {line.rstrip()!r}")
    return parts


def _interval(a: str, b: str, lineno: int) -> Interval:
    """Build an Interval from two SINEX times, naming the line on failure."""
    try:
        return Interval(parse_sinex_time(a), parse_sinex_time(b))
    except ValueError as exc:
        raise ValueError(f"line {lineno}: {exc}") from None


def read_sinex(path) -> GnssMetadata:
    """Read an IGS satellite metadata SINEX file (or an excerpt of one).

    Raises ValueError when the first line is not a SINEX header, a block is
    left open, or a row in a parsed block is malformed.
    """
    path = Path(path)
    lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    if not lines or not lines[0].startswith("%=SNX"):
        raise ValueError(f"{path} is not a SINEX file: the first line does not start with %=SNX")
    md = GnssMetadata(path=path)
    block = None
    for lineno, line in enumerate(lines[1:], start=2):
        if line.startswith("%ENDSNX"):
            break
        if line.startswith("+"):
            if block is not None:
                raise ValueError(f"line {lineno}: block {line[1:].strip()} opens inside {block}")
            block = line[1:].strip()
            md.blocks.append(block)
            if block not in KNOWN_BLOCKS:
                md.ignored_blocks.append(block)
            continue
        if line.startswith("-"):
            name = line[1:].strip()
            if name != block:
                raise ValueError(f"line {lineno}: -{name} closes {block or 'no open block'}")
            block = None
            continue
        if line.startswith("*") or not line.strip() or block is None:  # guard-exempt(SP3-LOCAL-PARSER): SINEX comment, not SP3
            continue
        if block == "SATELLITE/IDENTIFIER":
            svn, cospar, satcat, blk = _fields(line, 4, lineno, block)[:4]
            if not satcat.isdigit():
                raise ValueError(f"line {lineno}: NORAD number {satcat!r} is not an integer")
            comment = line.split(None, 4)[4].strip() if len(line.split(None, 4)) > 4 else ""
            md.identifiers[svn] = Identifier(svn, cospar, int(satcat), blk, comment)
        elif block == "SATELLITE/PRN":
            svn, a, b, prn = _fields(line, 4, lineno, block)[:4]
            md.prn_assignments.append(PrnAssignment(svn, prn, _interval(a, b, lineno)))
        elif block == "SATELLITE/MASS":
            svn, a, b, mass = _fields(line, 4, lineno, block)[:4]
            try:
                kg = float(mass)
            except ValueError:
                raise ValueError(f"line {lineno}: mass {mass!r} is not a number") from None
            md.masses.append(MassRecord(svn, kg, _interval(a, b, lineno)))
    if block is not None:
        raise ValueError(f"{path}: block {block} is never closed")
    return md
