"""
Ballistic - high-accuracy orbit prediction for Earth satellites.

Give it where a satellite is and how fast it is moving at one instant (its
"state"), and it computes where the satellite will be later, by integrating
the forces on it: the Earth's uneven gravity field, the Sun and Moon, the
solid-Earth and ocean tides, relativity, and, when you describe the
satellite, solar radiation pressure (SRP, the push of sunlight) and
atmospheric drag. Measured one hour after the start, against published
precise orbits, the error is (root-mean-square, RMSE; each number is a claim
in ``docs/validation_claims.md``):

- Low Earth orbit (LEO, Sentinel satellites, ``recommended_config`` with
  SRP + drag + ocean tides): 0.22 m [SENTINEL-YEAR-2025: 0.215 m]
- Medium Earth orbit, laser-ranging spheres LAGEOS / Etalon: 0.152 m [SLR-1H]
- GPS 0.668 m, QZSS inclined geosynchronous 0.739 m, BeiDou geostationary
  0.805 m, all without SRP [GNSS-1H]

Words used throughout the package:

- Frames. ITRF (also called ECEF) is the Earth-fixed frame: its axes turn
  with the Earth, so a point on the ground keeps fixed coordinates. GCRS is
  the inertial frame: its axes do not rotate. A velocity measured in one
  frame differs from the other by the Earth's rotation (hundreds of m/s;
  494 m/s in the example below), so every start state says which frame it is in.
- Epoch: an instant of time, with its time scale named (UTC, TT, GPS, ...).
- Earth orientation parameters (EOP): measured small irregularities of the
  Earth's rotation (IERS file ``finals2000A``), needed to convert between the
  two frames to better than a few meters.
- Space weather: the solar-flux index F10.7 and the geomagnetic index Ap,
  which the atmosphere model needs to compute air density (CelesTrak file
  ``SW-All.csv``).
- Ballistic coefficient beta = m / (Cd * A) in kg/m^2 (mass over drag
  coefficient times area): how strongly drag slows a satellite.
- Empirical acceleration (EA): a small correction acceleration fitted to
  observations, for forces the model does not know.

Quick start:
    >>> import ballistic
    >>> ballistic.load()   # EOP, leap seconds, space weather, ocean tides (downloads once)
    >>>
    >>> epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    >>> # A start state names its frame: Earth-fixed velocity here
    >>> # (circular 7669 m/s inertial minus the Earth's rotation, 494 m/s)
    >>> state = ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0])
    >>>
    >>> traj = ballistic.Trajectory(state)
    >>> traj.propagate_for(3600.0)                # meters, meters/second, seconds
    >>> s = traj(1800.0)                          # 30 minutes after the start
    >>> states = traj([0, 900, 1800], frame='GCRS')

Best accuracy for a known satellite (adds SRP and drag):
    >>> config = ballistic.recommended_config(state, "S1A")
    >>> traj = ballistic.Trajectory(state, config)

Several satellites at once:
    >>> batch = ballistic.Trajectory([state1, state2])
    >>> batch.propagate_for(3600.0)   # every member, from its own epoch
    >>> results = batch(1800.0)

Configuration (frozen, keyword-only; change a copy with .replace):
    >>> config = ballistic.TrajectoryConfig(
    ...     gravity_degree=70,
    ...     integrator=ballistic.IntegratorType.YOSHIDA6,
    ...     symplectic_step_size=30.0,
    ... )
    >>> traj = ballistic.Trajectory(state, config=config)

Data files (downloaded on first use, in the style of Skyfield):
    >>> loader = ballistic.Loader()
    >>> loader.timescale()      # EOP + leap seconds
    >>> loader.space_weather()  # F10.7 and Ap indices
    >>> status = ballistic.load()  # or: download and load everything

Where things live. The top-level ``ballistic`` holds what a first session
needs: loading the data files, times, start states, ``Trajectory`` and its
configuration, ``recommended_config`` and ``make_drag_environment``, frame
conversions, errors and warnings. Specialized tools are one level down, each
module with its own ``__all__``:

- ``ballistic.estimation``: fitting an empirical acceleration to
  observations (dense samples, radar, optics, position fixes,
  accelerometers), and turning it into a ballistic coefficient;
- ``ballistic.drag``: atmosphere descriptions built by hand (tabulated
  models, measured densities), table readers, saving an environment;
- ``ballistic.aero``: drag coefficients that follow the flow, for reentry;
- ``ballistic.readers``: precise-orbit EOF files and velocity-record checks;
- ``ballistic.tle``, ``ballistic.sp3``: two-line element sets with SGP4, and
  SP3 precise-orbit files;
- ``ballistic.events``, ``ballistic.errors``, ``ballistic.constants``,
  ``ballistic.satellite_properties``.

A name that used to be top level and now lives in a module raises
``RemovedApiError`` naming its new import (``docs/MIGRATION.md``).

Errors are subclasses of ``ballistic.BallisticError`` (see
``ballistic.errors``); non-fatal approximations arrive as
``ballistic.BallisticWarning`` subclasses. ``ballistic.doctor()`` checks the
installation and the data files.
"""

# Errors, warnings and the diagnostics channel come first: everything below
# may raise or warn, and library diagnostics must arrive as Python warnings.
from . import _core as _core_module
from .errors import (
    BallisticError,
    DataCoverageError,
    FrameError,
    InputPlausibilityError,
    ConfigError,
    RemovedApiError,
    StrictDiagnosticError,
    BallisticWarning,
    DataCoverageWarning,
    ModelSubstitutionWarning,
    DataSkippedWarning,
    ForcedStepWarning,
    ExtrapolationWarning,
    FrameAssumptionWarning,
    ConfigHintWarning,
    WARNING_CLASSES as _WARNING_CLASSES,
    set_strict,
)
_core_module._install_diagnostic_handler(_WARNING_CLASSES)
from ._removed import (
    REMOVED as _REMOVED, raise_removed as _raise_removed,
    MOVED as _MOVED, raise_moved as _raise_moved,
)
from . import constants
from ._core import GeodeticCoord, ecef_to_geodetic, geodetic_to_ecef
from ._core import events


def __getattr__(name):
    """Moved and removed names raise RemovedApiError naming the replacement."""
    if name in _MOVED:
        _raise_moved(name)
    if name in _REMOVED:
        _raise_removed(name)
    raise AttributeError(f"module 'ballistic' has no attribute '{name}'")


from ._core import (
    # Version
    __version__,
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,

    # Initialization
    initialize as _core_initialize,
    initialize_with_config,
    is_initialized,
    get_init_status,
    get_data_directory,
    InitConfig,
    InitResult,

    # Data Loading (low-level)
    load_eop,
    eop_coverage,
    set_eop_policy,
    load_space_weather,
    load_jpl_ephemeris,
    load_leap_seconds,
    get_space_weather,
    get_leap_second_info,
    LeapSecondInfo,

    # The empirical acceleration a TrajectoryConfig carries (fitting one:
    # ballistic.estimation)
    EmpiricalAcceleration,
    EmpiricalAccelFrame,

    # What make_drag_environment takes (more drag tools: ballistic.drag)
    SpaceWeatherHistory,
    BetaProfile1D,

    # Time
    Epoch,
    TimeScale,

    # Enums
    PropagationFrame,
    IntegratorType,
    ITRFRealization,
    InterpolationFrame,
    AltitudeCondition,
    PropagationDirection,
    NonConservativeStrategy,

    # Configuration
    SymplecticOptions,
    Rk45Options,
    SrpConfig,

    # State types
    StateVector,
    ForceBreakdown,
    PropagationResult,
    HistoryEntry,
    StepResult,
    InterpolatedState,
    PropagateToAltitudeResult,
    PropagateUntilResult,

    # Batch processing
    BatchInterpolatedState,
    BatchAltitudeResult,
    BatchArrayResult,
    BatchInterpolatorFunction,

    # Coordinate frame conversions
    gcrs_to_itrf,
    itrf_to_gcrs,
)

# Import unified Pythonic classes
from .trajectory import (
    Trajectory,
    TrajectoryConfig,
    InterpolatorConfig,
)

from . import sp3  # noqa: F401  (ballistic.sp3 submodule)
from . import tle  # noqa: F401  (ballistic.tle submodule)
from . import satellite_properties  # noqa: F401  (ballistic.satellite_properties submodule)
from . import errors  # noqa: F401  (ballistic.errors submodule)
from .doctor import doctor, DoctorReport
from .state import State  # (_core.State with list/tuple-friendly constructors added)


def make_drag_environment(t0, pos_ecef, vel_ecef, duration_s, config,
                          beta, weather=None, altitude_grid_km=(),
                          max_iter=3, tol_relative=0.01, sample_dt_s=60.0,
                          allow_predicted_weather=True):
    """Build the atmosphere description that turns drag on, in one call.

    Returns ``(drag_env, iteration_diag)``. ``drag_env`` is a
    ``DragEnvironment``: air density from the NRLMSIS 2.1 atmosphere model
    and wind from the HWM14 wind model, both evaluated at the satellite's
    exact position every time the force is computed, using the whole loaded
    space-weather file (F10.7 solar flux and Ap geomagnetic index, which the
    density depends on). ``iteration_diag`` is kept for compatibility and
    reports no iterations for this live environment. The env works for any
    trajectory, in either direction, wherever the weather reaches; a
    Trajectory stops with an error at that edge. Assign it to
    ``TrajectoryConfig.drag_env``::

        cfg = ballistic.TrajectoryConfig()
        env, diag = ballistic.make_drag_environment(
            t0, pos, vel, 3600.0, cfg,
            beta=ballistic.BetaProfile1D.from_mass_cd_area(2200.0, 2.2, 6.35))
        cfg = cfg.replace(drag_env=env)   # the config is frozen

    Args:
        config: a ballistic.TrajectoryConfig (dataclass) or
            ballistic._core.TrajectoryConfig; its drag_env is ignored.
        beta: the ballistic coefficient beta = mass / (Cd * area) in kg/m^2,
            as a BetaProfile1D (see .constant / .from_mass_cd_area), or an
            ``ballistic.aero.AeroBeta`` whose Cd follows the flow
            (FlowRegimeCd, for reentry: ``AeroBeta(mass_kg, area_m2,
            FlowRegimeCd(AeroShape.SPHERE), reference_length_m=diameter)``).
        weather: SpaceWeatherHistory, or None (default) for the MEASURED
            F10.7/Ap loaded by ballistic.load(), time-varying over the arc.
            Raises RuntimeError if none is loaded, and ``DataCoverageError``
            if ``t0`` is outside the loaded file — pass
            SpaceWeatherHistory.from_constant(
            ballistic.drag.WeatherConditions.quiet())
            to model quiet conditions on purpose.
        t0: the start Epoch; checked to have space weather, so a bad start
            fails here rather than during propagation.
        pos_ecef, vel_ecef, duration_s, altitude_grid_km, max_iter,
            tol_relative, sample_dt_s: accepted for compatibility and
            unused (the live environment needs no arc to tabulate).
        allow_predicted_weather: for days the loaded file has not measured
            yet, use CelesTrak's daily forecast (~45 days ahead; default).
            A Trajectory using the env then warns and
            ``env.uses_predicted_weather()`` is True. False raises instead.

    Most users want ballistic.recommended_config(), which calls this with
    the satellite's ballistic coefficient and turns on radiation pressure.
    """
    from . import _core as _core_mod
    core_cfg = config._to_core() if hasattr(config, '_to_core') else config
    return _core_mod.make_drag_environment(
        t0, pos_ecef, vel_ecef, duration_s, core_cfg, beta,
        weather=weather, altitude_grid_km=list(altitude_grid_km),
        max_iter=max_iter, tol_relative=tol_relative,
        sample_dt_s=sample_dt_s,
        allow_predicted_weather=allow_predicted_weather)

# The data loader and the recommended configuration
from .loader import Loader, data_files
from .recommended import recommended_config
from .nrl import install_nrl_models, nrl_models


def initialize():
    """Initialize the library without downloading anything.

    Loads the built-in data (the EGM2008 gravity field and the JPL DE440
    Sun/Moon/planet positions) and any Earth orientation parameter (EOP) and
    leap-second files already present in the data directory
    (``get_data_directory()``). Use ``ballistic.load()`` instead to also
    download missing files and load space weather and ocean tides.

    Returns:
        InitResult: what was loaded. It is truthy on success, so
        ``if ballistic.initialize():`` works.

    Warns (RuntimeWarning) when no EOP file was found. Without
    ``finals2000A.data``, propagation and Earth-fixed <-> inertial
    conversions raise ``DataCoverageError`` (or, after
    ``ballistic.set_eop_policy('assume_zero')``, run with the parameters set
    to zero, about 3-10 m off in low Earth orbit (LEO)) — run
    ``ballistic.Loader().load_eop()`` once to fix it. Also warns for each
    staleness note in ``InitResult.warnings``.
    """
    import warnings
    result = _core_initialize()
    if result.success and not result.eop_loaded:
        warnings.warn(
            "ballistic.initialize(): no EOP data (finals2000A.data) found — "
            "propagation and ECEF<->GCRS transforms raise DataCoverageError "
            "(set_eop_policy('assume_zero') runs with zeros, ~3-10 m off at LEO). "
            "Run ballistic.Loader().load_eop() once to download it.",
            RuntimeWarning, stacklevel=2)
    for msg in result.warnings:
        # Staleness notes: an EOP table that ends before today, or an expired
        # leap-second file. The data still loads, but dates past its end are
        # less accurate (or raise, see ballistic.eop_coverage()).
        warnings.warn("ballistic.initialize(): " + msg,
                      RuntimeWarning, stacklevel=2)
    return result


def load(directory=None, verbose=True, space_weather=True, ocean_tides=True, *,
         eop=True, leap_seconds=True, offline=False):
    """
    Load every data file the best-accuracy setup needs, downloading if needed.

    The files: Earth orientation parameters (EOP) and leap seconds (needed to
    convert between Earth-fixed and inertial coordinates — meters of error in
    low Earth orbit without them), CelesTrak space weather (the F10.7 and Ap
    indices the atmosphere model needs for drag) and the FES2004 ocean-tide
    coefficients.

    Each argument chooses where that file comes from, independently:

    * a path: load that file. Never downloads, whatever ``offline`` says.
    * True (default): the copy in the data directory. With ``offline=False``
      it is downloaded if missing and refreshed if stale (a failed refresh
      falls back to the cached copy, with a warning); with ``offline=True``
      the cached copy is used as it is, or the file is skipped with a
      warning if there is none.
    * False: skip it.

    So ``offline`` only matters for files left at True. Loading replaces the
    data already in memory; a file that fails to load leaves the earlier
    data in place (the warning says which). The default data directory is
    also where downloads go, so ``offline=True`` after an online load reads
    back the same files.

    Args:
        directory: Optional custom data directory path.
        verbose: If True, print download progress messages.
        space_weather, ocean_tides, eop, leap_seconds: True (default) to
            load from the data directory, False to skip, or a path to a
            local file (e.g. ``space_weather="~/data/SW-All.csv"``).
        offline: Never download. The environment variable
            ``BALLISTIC_OFFLINE=1`` has the same effect (air-gapped machines,
            CI).

    Returns:
        InitResult: what is loaded (also initializes the library), with staleness flags (``eop_table_stale``,
        ``leap_seconds_expired``) and ``warnings``; each warning is also
        issued as a ``ballistic.DataCoverageWarning``. Use
        ``ballistic.Loader()`` for granular downloads.

    Example:
        >>> import ballistic
        >>> ballistic.load()                                    # online
        >>> ballistic.load(space_weather="SW-All.csv",           # air-gapped
        ...                eop="finals2000A.all", offline=True)
    """
    import datetime as _dt
    import os
    import warnings
    from . import _core
    offline = offline or os.environ.get("BALLISTIC_OFFLINE") == "1"
    loader = Loader(directory=directory, verbose=verbose, offline=offline)
    _core.initialize()

    def _one(choice, file_type, from_cache):
        if choice is False or choice is None:
            return None
        if choice is True:
            return from_cache()
        return loader.load_file(file_type, choice)

    results = {
        "leap_seconds": _one(leap_seconds, "leap_seconds", loader.load_leap_seconds),
        "eop": _one(eop, "eop", loader.load_eop),
        "space_weather": _one(space_weather, "space_weather", loader.load_space_weather),
        "ocean_tides": _one(ocean_tides, "ocean_tides", loader.load_ocean_tides),
    }
    consequences = {
        "eop": "propagation and ECEF<->GCRS transforms raise DataCoverageError",
        "leap_seconds": "leap seconds after 2017 are unknown",
        "space_weather": "drag (recommended_config / make_drag_environment) raises",
        "ocean_tides": "propagation runs without ocean tides",
    }
    from .loader import LOADED_FROM
    in_memory = {
        "eop": _core.eop_loaded,
        "leap_seconds": lambda: _core.get_leap_second_info().loaded_from_file,
        "space_weather": _core.space_weather_loaded,
        "ocean_tides": _core.ocean_tides_loaded,
    }
    for note in loader.notes:
        warnings.warn("ballistic.load(): " + note, RuntimeWarning, stacklevel=2)
    for name, ok in results.items():
        if ok is not False:
            continue
        if in_memory[name]():
            # A failed load leaves the earlier data in place.
            src = LOADED_FROM.get(name, "an earlier load")
            warnings.warn(f"ballistic.load(): {name} not reloaded; still using "
                          f"the data already loaded from {src}.",
                          RuntimeWarning, stacklevel=2)
        else:
            warnings.warn(f"ballistic.load(): {name} not loaded — "
                          f"{consequences[name]} until it is.",
                          RuntimeWarning, stacklevel=2)
    if results["space_weather"]:
        for msg in _core.space_weather_warnings():
            warnings.warn("ballistic.load(): space weather: " + msg,
                          RuntimeWarning, stacklevel=2)
        last = _core.SpaceWeatherHistory.last_measured_day()
        if verbose and last is not None:
            day = _dt.date(1858, 11, 17) + _dt.timedelta(
                days=int(round(last.mjd(_core.TimeScale.UTC))))
            print(f"Space weather measured through {day}; drag for later "
                  "dates needs a newer file or allow_predicted_weather=True.")
    from .errors import DataCoverageWarning
    status = _core.get_init_status()
    for msg in status.warnings:
        warnings.warn("ballistic.load(): " + msg, DataCoverageWarning, stacklevel=2)
    return status


__all__ = [
    # Errors and warnings
    "BallisticError",
    "DataCoverageError",
    "FrameError",
    "InputPlausibilityError",
    "ConfigError",
    "RemovedApiError",
    "StrictDiagnosticError",
    "BallisticWarning",
    "DataCoverageWarning",
    "ModelSubstitutionWarning",
    "DataSkippedWarning",
    "ForcedStepWarning",
    "ExtrapolationWarning",
    "FrameAssumptionWarning",
    "ConfigHintWarning",
    "set_strict",

    # Constants and geodesy
    "constants",
    "GeodeticCoord",
    "ecef_to_geodetic",
    "geodetic_to_ecef",

    # Version
    "__version__",
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "VERSION_PATCH",

    # Initialization
    "initialize",
    "initialize_with_config",
    "is_initialized",
    "get_init_status",
    "get_data_directory",
    "InitConfig",
    "InitResult",

    # Data files (auto-download Loader, like Skyfield) and low-level loading
    "load",
    "Loader",
    "data_files",
    "install_nrl_models",
    "nrl_models",
    "load_eop",
    "eop_coverage",
    "set_eop_policy",
    "load_space_weather",
    "load_jpl_ephemeris",
    "load_leap_seconds",
    "get_space_weather",
    "get_leap_second_info",
    "LeapSecondInfo",

    # Time
    "Epoch",
    "TimeScale",

    # Enums
    "PropagationFrame",
    "IntegratorType",
    "ITRFRealization",
    "InterpolationFrame",
    "AltitudeCondition",
    "PropagationDirection",
    "NonConservativeStrategy",

    # Configuration (Pythonic dataclasses) and the recommended setup
    "TrajectoryConfig",
    "InterpolatorConfig",
    "SymplecticOptions",
    "Rk45Options",
    "SrpConfig",
    "recommended_config",
    "make_drag_environment",
    "BetaProfile1D",
    "SpaceWeatherHistory",
    "EmpiricalAcceleration",
    "EmpiricalAccelFrame",

    # State types
    "State",
    "StateVector",
    "ForceBreakdown",
    "PropagationResult",
    "HistoryEntry",
    "StepResult",
    "InterpolatedState",
    "PropagateToAltitudeResult",
    "PropagateUntilResult",
    "events",

    # Main class (unified Pythonic API)
    "Trajectory",

    # Batch processing
    "BatchInterpolatedState",
    "BatchAltitudeResult",
    "BatchArrayResult",
    "BatchInterpolatorFunction",

    # Coordinate frame conversions
    "gcrs_to_itrf",
    "itrf_to_gcrs",

    # Installation check
    "doctor",
    "DoctorReport",
]


# Public names of the compiled module that no ``__all__`` lists, each with
# the reason (tests/python/test_namespace_tiers.py checks every public
# ``_core`` name is exported exactly once or listed here).
_INTERNAL_CORE_NAMES: dict[str, str] = {
    "TrajectoryBatch": "wrapped: ballistic.Trajectory([states]) runs batches",
    "TrajectoryInterpolator": "reachable: returned by Trajectory() / Trajectory.interpolator",
    "StateTransition": "wrapped: behind Trajectory's state-transition methods",
    "StmOptions": "wrapped: behind Trajectory's state-transition methods",
    "sp3_classify_frame": "aliased: exported as ballistic.sp3.classify_frame",
    "build_info": "internal: read by doctor()",
    "eop_loaded": "internal: ballistic.load() and get_init_status() report it",
    "space_weather_loaded": "internal: ballistic.load() reports it",
    "space_weather_warnings": "internal: ballistic.load() issues them as warnings",
    "load_ocean_tides": "internal: ballistic.load() / Loader.load_ocean_tides",
    "ocean_tides_loaded": "internal: ballistic.load() and doctor() report it",
    "unload_ocean_tides": "internal: test hook that empties the tide store",
}


# The tiers are attributes of the package (``ballistic.estimation`` ...);
# imported last because they import from the modules above.
from . import estimation, drag, aero, readers  # noqa: E402,F401
