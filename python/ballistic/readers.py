"""Reading precise-orbit products and checking predictions against them.

Precise orbits published by the satellite operators are the truth this
library is validated against. This module holds the reader for ESA's EOF
orbit files (Sentinel precise orbits), the helper that propagates from one
state and scores the prediction against such a file, and checks on a
product's velocity records: whether they are the true time derivative of its
positions (``check_velocity_records``), and the small Earth-axis drift term
some products leave out of their velocities (``cip_drift_velocity``).
SP3 files have their own module, ``ballistic.sp3``.

    from ballistic.readers import parse_eof_auto
    eof = parse_eof_auto("S1A_OPER_AUX_POEORB_....EOF")
"""

from ._core import (
    EOFStateVector,
    EOFFile,
    EOFParser,
    PropagationTestResult,
    parse_eof_auto,
    find_and_parse_eof,
    compute_propagation_result,
    check_velocity_records,
    VelocityRecordCheck,
    VelocityRecordVerdict,
    cip_drift_velocity,
)

__all__ = [
    "EOFStateVector",
    "EOFFile",
    "EOFParser",
    "PropagationTestResult",
    "parse_eof_auto",
    "find_and_parse_eof",
    "compute_propagation_result",
    "check_velocity_records",
    "VelocityRecordCheck",
    "VelocityRecordVerdict",
    "cip_drift_velocity",
]
