"""Drag coefficients that follow the flow, for reentry.

In orbit the air is so thin that molecules hit a satellite one at a time
(free-molecular flow) and a constant drag coefficient Cd is a good model.
During reentry the flow passes through a transition into continuum flow,
where Cd depends on the Mach number and the shape. The types here describe
that: an ``AeroBeta`` (mass, reference area and length, and a Cd model) can
replace a constant ballistic coefficient as the beta given to
``ballistic.make_drag_environment``; ``FlowRegimeCd`` computes Cd from the
local temperature, composition and speed. The free functions give the
underlying coefficients for checks and plots.

    from ballistic.aero import AeroBeta, AeroShape, FlowRegimeCd
    beta = AeroBeta(500.0, 0.785, FlowRegimeCd(AeroShape.SPHERE),
                    reference_length_m=1.0)
"""

from ._core import (
    AeroShape,
    ConstantCd,
    FlowRegimeCd,
    MachTable,
    AeroBeta,
    free_molecular_cd,
    reemission_ratio,
    henderson1976_sphere_cd,
)

__all__ = [
    "AeroShape",
    "ConstantCd",
    "FlowRegimeCd",
    "MachTable",
    "AeroBeta",
    "free_molecular_cd",
    "reemission_ratio",
    "henderson1976_sphere_cd",
]
