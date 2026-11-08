"""Physical constants, read from the library (include/ballistic/core/constants.hpp).

Use these instead of typing the Earth's gravitational parameter GM or its
radius into a script: hand-typed copies drift apart. Grouped as in
C++::

    ballistic.constants.earth.GM                 # WGS84, m^3/s^2
    ballistic.constants.earth.RADIUS_EQUATORIAL  # WGS84, m
    ballistic.constants.earth.ANGULAR_VELOCITY   # rad/s
    ballistic.constants.earth.J2                 # from the built-in EGM2008 gravity model
    ballistic.constants.sun.AU                   # astronomical unit, m
    ballistic.constants.SPEED_OF_LIGHT           # m/s

For altitude above the Earth's reference ellipsoid (WGS84) use
ballistic.ecef_to_geodetic, not the distance from the center minus a radius
(the Earth is flattened, so that is off by kilometers away from the equator).
"""

from types import SimpleNamespace as _NS

from . import _core

_t = _core._constants_table()

SPEED_OF_LIGHT = _t["SPEED_OF_LIGHT"]
SECONDS_PER_DAY = _t["SECONDS_PER_DAY"]
DEG_TO_RAD = _t["DEG_TO_RAD"]
RAD_TO_DEG = _t["RAD_TO_DEG"]
J2000_JD = _t["J2000_JD"]
MJD_OFFSET = _t["MJD_OFFSET"]

earth = _NS(**{k: (tuple(v) if k == "J_COEFFS" else v) for k, v in _t["earth"].items()})
sun = _NS(**_t["sun"])
moon = _NS(**_t["moon"])
jupiter = _NS(**_t["jupiter"])

del _t
