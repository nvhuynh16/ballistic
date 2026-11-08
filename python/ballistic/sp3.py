"""Reading SP3 orbit files.

SP3 is the standard text format for precise satellite orbits: a header,
then for each epoch the position (and optionally velocity) of each
satellite. A file's records for one satellite form an "arc". The reader is
the C++ one in include/ballistic/validation/sp3.hpp; this module adds .gz
support and a start-state helper.

    f = ballistic.sp3.read("GFZOP_RSO_L64_..._v03.sp3.gz")
    arc = f.arc("L64")
    arc.epoch0, arc.offsets_s, arc.position, arc.velocity
    pos, vel = arc.interpolate(1800.0)             # range-checked, no gaps
    vel, centered = arc.derived_velocities()        # centered = 0 at edges

What it guarantees:
the header's time system is honored (a file in GPS time is not read as
UTC); time offsets from ``epoch0`` are exact integer nanoseconds (exposed in
seconds); positions flagged bad or absent are not records; a malformed line
raises naming it; interpolation never extrapolates or crosses a gap; each
file is its own arc (separate files are separate orbit solutions — merging
them is the caller's decision).
"""

import gzip
from pathlib import Path

from . import _core

SP3File = _core.SP3File
SP3Arc = _core.SP3Arc
SP3Header = _core.SP3Header
SP3TimeSystem = _core.SP3TimeSystem
SP3FrameInfo = _core.SP3FrameInfo
classify_frame = _core.sp3_classify_frame

__all__ = ["read", "initial_state", "classify_frame", "SP3File", "SP3Arc", "SP3Header",
           "SP3TimeSystem", "SP3FrameInfo"]


def read(path) -> "_core.SP3File":
    """Read an SP3 file, plain text or gzip-compressed (detected from content).

    Args:
        path: path to the file (str or Path).

    Returns:
        An ``SP3File`` with one ``SP3Arc`` per satellite (``.arcs``,
        ``.arc(name)``).

    Raises:
        BallisticError: a malformed line (the message gives the line
            number), an unknown time system, or no epoch records.
        UnicodeDecodeError: non-ASCII content.
    """
    path = Path(path)
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return _core._sp3_read_string(data.decode("ascii", errors="strict"), str(path))


def initial_state(file, satellite=None, epoch=None, *, realization=None, inertial_as=None,
                  allow_suborbital=False, check_frame=True) -> "_core.State":
    """A checked start state (ballistic.State) from an SP3 file at ``epoch``.

    The position is interpolated between the file's records; the velocity is
    the time derivative of the interpolated positions (the velocity records
    some producers include follow different conventions, so they are not
    used). The frame comes from the file header.

    Args:
        file: an SP3File, or a path to read.
        satellite: the satellite ID (e.g. "L64"); may be omitted when the
            file holds only one.
        epoch: the start Epoch, inside the arc.
        realization: the ITRF realization (version of the Earth-fixed
            frame), needed when the header's Earth-fixed label does not
            name one (e.g. "CTS").
        inertial_as: 'gcrs' or 'j2000', needed when the header says only
            "J2000" for an inertial frame (the two sets of axes differ by
            up to ~0.8 m in low Earth orbit).
        allow_suborbital: accept an orbit whose lowest point is below 100 km.
        check_frame: run the frame mix-up check.

    Returns:
        A ``ballistic.State``. Also available as ``ballistic.State.from_sp3``.

    Raises:
        ValueError: the file holds several satellites and none is named.
        BallisticError: no records for the named satellite.
        TypeError: ``epoch`` is not an Epoch.
        DataCoverageError: the epoch is within half an interpolation window
            of an arc edge or gap.
        FrameError: an unknown frame label, or a label that needs
            ``realization`` / ``inertial_as`` and did not get it.
    """
    if not isinstance(file, _core.SP3File):
        file = read(file)
    if satellite is None:
        if len(file.arcs) != 1:
            raise ValueError(f"sp3 {file.source}: holds {len(file.arcs)} satellites; name one "
                             f"({', '.join(a.satellite for a in file.arcs[:8])}...)")
        satellite = file.arcs[0].satellite
    if not isinstance(epoch, _core.Epoch):
        raise TypeError("initial_state needs the start epoch (an Epoch inside the arc)")
    return _core._sp3_initial_state(file, satellite, epoch, realization, inertial_as,
                                    allow_suborbital, check_frame)

