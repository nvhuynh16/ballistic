"""Estimating an empirical acceleration (EA) from observations.

An empirical acceleration is a small correction acceleration, fitted to
observations of a satellite or object, that accounts for forces the model
does not know (mismodeled drag, thrust, ...). Everything needed to fit one
lives here: the estimators, their configurations and results, the sensor
types for radar, optical and mixed observations, and the tools that turn a
fitted EA into a ballistic coefficient. It is a separate module because most
users only propagate orbits and never fit a correction; the correction itself
(``ballistic.EmpiricalAcceleration``) stays in the top-level package, since a
``TrajectoryConfig`` carries it.

The wrappers here call the C++ estimators; they accept the Pythonic
:class:`ballistic.TrajectoryConfig` dataclass as well as the C++
``_core.TrajectoryConfig`` (and convert), so user code never reaches into the
private ``_to_core()``.

What was observed is said by the sample TYPE: wrap observations in
:class:`PositionSamples` / :class:`VelocitySamples` / :class:`AccelSamples`
for ``estimate_ea_bspline``. For radar, optical and mixed sensors use
``estimate_ea_smoother``; ``information_bound`` tells, before any fit, the
best accuracy a sensor suite allows.

    from ballistic.estimation import (EABSplineConfig, PositionSamples,
                                      estimate_ea_bspline)
"""

from typing import Any, Optional

from . import _core
from ._core import (
    # The correction's building blocks (EmpiricalAcceleration factories)
    EmpiricalAccelEntry,
    EAInterp,
    EARepresents,
    BSplineCoefficients,

    # Dense single-modality estimator (estimate_ea_bspline)
    EABSplineConfig,
    PositionSamples,
    VelocitySamples,
    AccelSamples,
    CVMode,
    GNStatus,
    GNRefine,
    EAResult,

    # From a fitted EA to a ballistic coefficient
    EADecomposition,
    BetaStepEstimate,
    BetaResult,
    BetaDecompositionConfig,
    decompose_ea,
    refine_beta_from_ea,

    # Sensors and the fusion estimator (estimate_ea_smoother)
    RadarObs,
    OpticsObs,
    PositionFixObs,
    AccelObs,
    MultiSensorScenario,
    SensorClock,
    SensorMask,
    EASmootherConfig,
    EASmootherResult,
    EASmootherRateBudget,

    # Declarative front door and sensor-design bound
    EAMethod,
    EAProblemSpec,
    EASpecResult,
    EAInformationBound,
    EAInformationBoundOptions,
    estimate_ea,
    reject_heavy_tails,
)


def _core_config(config: Any) -> "_core.TrajectoryConfig":
    """Return a ``_core.TrajectoryConfig`` from either config type."""
    to_core = getattr(config, "_to_core", None)
    return to_core() if callable(to_core) else config


def estimate_ea_bspline(epoch, position, velocity, observations,
                        force_model, config: Optional[Any] = None):
    """Fit an empirical acceleration to dense position, velocity or accelerometer data.

    The correction is a B-spline (a smooth piecewise polynomial in time)
    fitted by repeated regularized least squares (iteratively regularized
    Gauss-Newton, IRGN), then by default polished by a Gauss-Newton
    refinement against the observations.

    Args:
        epoch: the Epoch of the start state.
        position: start position in the Earth-fixed (ECEF) frame, meters,
            3-sequence.
        velocity: start velocity, Earth-fixed, meters/second, 3-sequence.
        observations: PositionSamples, VelocitySamples or AccelSamples (all
            Earth-fixed).
        force_model: a ballistic.TrajectoryConfig or ``_core.TrajectoryConfig``:
            the forces the correction is added to.
        config: an EABSplineConfig (None = defaults).

    Returns:
        An EAResult; its ``empirical_accel`` (in the inertial GCRS frame) can
        be put in ``TrajectoryConfig.empirical_accel``. See
        ``_core.estimate_ea_bspline`` for details.
    """
    args = (epoch, position, velocity, observations, _core_config(force_model))
    return (_core.estimate_ea_bspline(*args) if config is None
            else _core.estimate_ea_bspline(*args, config))



def refine_ea_gn(epoch, position, velocity, position_data, initial_ea,
                 force_model, gn: Optional[Any] = None,
                 max_duration_s: float = 3600.0):
    """Improve an existing empirical acceleration so the orbit matches positions.

    Runs Gauss-Newton least squares (repeatedly linearize and solve) on the
    correction ``initial_ea`` against ``position_data``.

    Args:
        epoch, position, velocity: the start state (Earth-fixed, meters and
            meters/second).
        position_data: the target positions, a list of ``(Epoch, position)``
            pairs (Earth-fixed, meters).
        initial_ea: the EmpiricalAcceleration to refine.
        force_model: a ballistic.TrajectoryConfig or ``_core.TrajectoryConfig``.
        gn: a GNRefine with the iteration settings (None = defaults).
        max_duration_s: longest arc to propagate, seconds.

    Returns:
        The refined result (see ``_core.refine_ea_gn``).
    """
    kwargs = {"max_duration_s": max_duration_s}
    if gn is not None:
        kwargs["gn"] = gn
    return _core.refine_ea_gn(epoch, position, velocity, position_data,
                              initial_ea, _core_config(force_model), **kwargs)


def _wrap_multisensor(name):
    """Wrap the C++ function ``name`` (signature ``(scenario, mask, fm_cfg,
    settings)``) so its force model may be either config type, given by
    position or as ``force_model=`` / ``fm_cfg=``; the settings may be given
    by position or as ``config=``."""
    core_fn = getattr(_core, name)

    def wrapper(scenario, mask, force_model=None, *args, fm_cfg=None, **kwargs):
        if force_model is None:
            force_model = fm_cfg
        elif fm_cfg is not None:
            raise TypeError(f"{name}(): give the force model once, as "
                            "force_model or fm_cfg, not both")
        if force_model is None:
            raise TypeError(f"{name}(): missing the force model (a "
                            "ballistic.TrajectoryConfig)")
        if "config" in kwargs:      # the estimator settings, by keyword
            if args:
                raise TypeError(f"{name}(): settings given twice")
            args = (kwargs.pop("config"),)
        return core_fn(scenario, mask, _core_config(force_model), *args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


estimate_ea_smoother = _wrap_multisensor("estimate_ea_smoother")
estimate_ea_smoother.__doc__ = """Fit an empirical acceleration to several sensors at once.

``estimate_ea_smoother(scenario, mask, force_model, config=EASmootherConfig())``
combines radar, optical (angles-only), position-fix and accelerometer data
(a MultiSensorScenario; ``mask`` is a SensorMask choosing which sensors to
use) in a Kalman-filter-and-smoother pass. ``force_model`` is a
ballistic.TrajectoryConfig or ``_core.TrajectoryConfig``. Returns an
EASmootherResult (EA in the inertial GCRS frame, its 1-sigma uncertainty and
the NIS consistency check). See ``_core.estimate_ea_smoother`` for details.
"""

information_bound = _wrap_multisensor("information_bound")
information_bound.__doc__ = """The best accuracy any estimator could reach with a sensor suite.

``information_bound(scenario, mask, force_model, opts=EAInformationBoundOptions())``
computes, without running an estimator, the position accuracy after
re-propagation that the sensors of ``scenario`` selected by ``mask`` allow.
Use it to design a sensor suite before tuning an estimator. ``force_model``
is a ballistic.TrajectoryConfig or ``_core.TrajectoryConfig``. Returns an
EAInformationBound; failures are reported in its ``error_message``. See
``_core.information_bound`` for details.
"""


__all__ = [
    # The correction's building blocks
    "EmpiricalAccelEntry",
    "EAInterp",
    "EARepresents",
    "BSplineCoefficients",

    # Dense single-modality estimator
    "estimate_ea_bspline",
    "EABSplineConfig",
    "PositionSamples",
    "VelocitySamples",
    "AccelSamples",
    "CVMode",
    "GNStatus",
    "GNRefine",
    "EAResult",
    "refine_ea_gn",

    # From a fitted EA to a ballistic coefficient
    "EADecomposition",
    "BetaStepEstimate",
    "BetaResult",
    "BetaDecompositionConfig",
    "decompose_ea",
    "refine_beta_from_ea",

    # Sensors and the fusion estimator
    "RadarObs",
    "OpticsObs",
    "PositionFixObs",
    "AccelObs",
    "MultiSensorScenario",
    "SensorClock",
    "SensorMask",
    "estimate_ea_smoother",
    "EASmootherConfig",
    "EASmootherResult",
    "EASmootherRateBudget",

    # Declarative front door and sensor-design bound
    "EAMethod",
    "EAProblemSpec",
    "EASpecResult",
    "estimate_ea",
    "information_bound",
    "EAInformationBound",
    "EAInformationBoundOptions",
    "reject_heavy_tails",
]
