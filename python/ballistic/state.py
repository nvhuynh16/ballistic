"""ballistic.State: where a trajectory starts, with its frame named.

A State is a position (meters) and velocity (meters/second) at an Epoch,
together with the coordinate frame they are in. The frame matters: a
velocity measured in the Earth-fixed frame (ITRF, which rotates with the
Earth) differs from the same velocity in an inertial frame (GCRS or J2000,
which do not rotate) by hundreds of m/s. The constructors check the numbers
(units, below ground, an orbit that would hit the Earth, numbers that look
like they belong to the other frame) and raise ``InputPlausibilityError`` or
``FrameError`` rather than propagate a mistake.

Constructors:

- ``State.from_itrf(epoch, r_m, v_m_s)``: Earth-fixed position and velocity.
- ``State.from_gcrs(epoch, r_m, v_m_s)``: inertial GCRS.
- ``State.from_j2000(epoch, r_m, v_m_s)``: inertial J2000 (EME2000).
- ``State.from_teme(epoch, r_m, v_m_s)``: the TEME frame that TLE
  propagators output.
- ``State.from_keplerian(...)``: orbital elements.
- ``State.from_sp3(file, satellite, epoch)``: from an SP3 precise-orbit file.
- ``State.from_tle(line1, line2)``: from a two-line element set.

The constructors live in C++ (include/ballistic/initial_state.hpp); this
module lets ``r_m`` and ``v_m_s`` be any 3-element sequence (list, tuple,
numpy array) and adds ``State.from_sp3`` and ``State.from_tle``.
"""

import functools

import numpy as np

from . import _core
from . import sp3 as _sp3
from . import tle as _tle

State = _core.State


def _vectors(f):
    """Wrap a C++ constructor so r_m and v_m_s may be any 3-sequence."""
    @functools.wraps(f)
    def wrapper(epoch, r_m, v_m_s, *args, **kwargs):
        return f(epoch, np.asarray(r_m, dtype=np.float64), np.asarray(v_m_s, dtype=np.float64),
                 *args, **kwargs)
    return staticmethod(wrapper)


for _name in ("from_itrf", "from_gcrs", "from_j2000", "from_teme"):
    setattr(State, _name, _vectors(getattr(State, _name)))
State.from_sp3 = staticmethod(_sp3.initial_state)


def _from_tle(line1, line2=None, epoch=None, *, allow_suborbital=False, check_frame=True):
    """A start state from a two-line element set (TLE), as SGP4 computes it.

    A TLE is the compact public orbit format (two 69-character lines);
    SGP4 is the standard model that turns it into a position and velocity.
    This returns SGP4's position and velocity at the TLE's epoch (or at
    ``epoch``), converted from SGP4's TEME frame.

    Args:
        line1, line2: the two element lines (str), or a parsed
            ``ballistic.tle.Tle`` as ``line1`` (then pass ``epoch`` by keyword).
        epoch: an Epoch to evaluate at instead of the TLE's own epoch.
        allow_suborbital: accept an orbit whose lowest point is below 100 km.
        check_frame: run the frame mix-up check.

    Accuracy: a TLE itself is only good to ~0.5 km in low Earth orbit
    (LEO). Measured: a numerical propagation started from this state drifts
    by km per day (its orbit size, the semi-major axis, is tens of meters
    off; 12-24 km after 3.6-7 days at 490 km, where SGP4 stayed within 1-3 km).
    For horizons beyond hours start from ``ballistic.tle.fit_state(tle,
    config)`` instead (0.3-1.8 km there). See ``ballistic.tle`` and
    docs/validation_claims.md (TLE-GRACEFO).

    Raises:
        InputPlausibilityError: a malformed TLE (bad length, checksum or
            field), or a state that fails the checks.
        TypeError: a parsed Tle followed by a positional second argument.
        FrameError: the state looks like it is in a different frame.
    """
    tle = line1 if isinstance(line1, _tle.Tle) else _tle.parse(line1, line2)
    if isinstance(line1, _tle.Tle) and line2 is not None:
        raise TypeError("from_tle(tle, epoch=...): pass the epoch by keyword after a parsed Tle")
    return State._from_tle(tle, epoch, allow_suborbital=allow_suborbital, check_frame=check_frame)


State.from_tle = staticmethod(_from_tle)
