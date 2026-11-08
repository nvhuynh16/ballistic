"""
The ``Trajectory`` class: predict a satellite's orbit and read it back at any time.

Propagation means stepping the equations of motion forward (or backward) in
time from a known start state. ``Trajectory`` keeps every step it takes, so
you can then ask for the position and velocity at any time inside the
propagated span (interpolation). One class covers three C++ pieces:

- propagation (``_core.Trajectory``)
- interpolation between the recorded steps (``_core.TrajectoryInterpolator``)
- many satellites at once ("batch" mode, ``_core.TrajectoryBatch``)

Units are SI throughout: meters, meters per second, seconds. Frames: ECEF
(also called ITRF) is the Earth-fixed frame, which rotates with the Earth;
GCRS is the inertial (non-rotating) frame the propagation runs in.

Example usage::

    import ballistic

    ballistic.load()   # Earth orientation, leap seconds, space weather

    epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    # Earth-fixed (ITRF): circular 7669 m/s inertial minus Earth rotation 494 m/s
    state = ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0])

    traj = ballistic.Trajectory(state)
    traj.propagate_for(3600.0)

    # Interpolate with float offsets (seconds from the initial epoch)
    s = traj(1800.0)
    states = traj([0, 900, 1800, 2700, 3600], frame='GCRS')

    # Batch mode: a list of States
    batch = ballistic.Trajectory([state, state2])
    batch.propagate_for(3600.0)
"""

from dataclasses import dataclass, field, replace
from typing import Union, List, Optional, Any
import numpy as np

from . import _core
from .errors import ConfigError, RemovedApiError


# Accepted names for the ``frame`` argument (matched case-insensitively)
FRAME_MAP = {
    'ECEF': _core.InterpolationFrame.ECEF,
    'GCRS': _core.InterpolationFrame.GCRS,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryConfig:
    """
    Which forces to model and how to integrate them.

    The defaults are the most accurate general settings; only radiation
    pressure and drag are off (see the note below).

    Attributes:
        gravity_degree: how much detail of the Earth's gravity field to use
            (EGM2008 spherical-harmonic degree; default 70, enough for low
            Earth orbit (LEO); a degree above the loaded model, 360 for
            the built-in one, raises ConfigError).
        enable_third_body: include the pull of the Sun, Moon and Jupiter.
        enable_relativistic: include the general-relativity corrections
            (Schwarzschild and Lense-Thirring terms).
        enable_solid_tides: include the tides of the solid Earth (IERS 2010
            conventions) and, when loaded, the ocean tides.
        ocean_tide_degree: detail of the FES2004 ocean-tide model (0 = off;
            default 50). Active only once the coefficients are loaded
            (``ballistic.load()``).
        itrf: which ITRF realization (a dated version of the Earth-fixed
            frame, e.g. ITRF2014, ITRF2020) Earth-fixed coordinates are in.
            Must match the source of your start state.
        integrator: the numerical method. RK45 (adaptive Runge-Kutta;
            default) is the most accurate over an hour or so; YOSHIDA6 /
            YOSHIDA8 (symplectic: they keep the orbit's energy from drifting)
            are better for runs of 6 hours or more.
        rk45_initial_step_size: first RK45 step, seconds (default 60; about 8
            for a fast descent through the atmosphere).
        rk45_max_step_size: largest RK45 step, seconds (default 3600; about 8
            for a fast descent through the atmosphere).
        symplectic_step_size: fixed step of the symplectic integrators, seconds.
        symplectic_adaptive: let the symplectic step change while keeping the
            method time-reversible (Hairer et al. 2006).
        symplectic_min_step: smallest adaptive step, seconds.
        symplectic_max_step: largest adaptive step, seconds.
        symplectic_alpha: exponent that scales the adaptive step (None = auto
            = 1/order).
        symplectic_symmetry_tol: relative tolerance of the adaptive step's
            predictor-corrector iteration.
        symplectic_nc_strategy: how a symplectic integrator handles forces
            that do not conserve energy (drag, radiation pressure).
        srp: a ballistic.SrpConfig (mass, area, reflectivity). Giving one
            turns radiation pressure on: solar radiation pressure (SRP, the
            push of sunlight), sunlight reflected by the Earth (albedo) and
            the Earth's infrared glow. None (default) = off.
        drag_env: a DragEnvironment (air density, wind and ballistic
            coefficient) — the only way to turn drag on. Build one with
            ``ballistic.make_drag_environment`` or let
            ``ballistic.recommended_config`` do it. None (default) = off.
        empirical_accel: an EmpiricalAcceleration, a small correction
            acceleration (EA) fitted to observations, added to the forces.
        ea_reference_epoch: the Epoch that time 0 of ``empirical_accel``
            refers to (requires empirical_accel; default: the trajectory's
            start epoch).
        report_forces: include the breakdown of forces in each step result.
        include_disabled_forces: in that breakdown, also list switched-off
            forces (as zeros).

    The config is frozen and keyword-only, so a misspelt field raises
    instead of being ignored. Change a field with ``cfg.replace(...)``
    (or ``dataclasses.replace``), which returns a new config.

    Note:
        SRP and drag are OFF by default only because they need the
        satellite's mass and areas (and drag also needs space weather). Turn
        both on for real satellites — ``ballistic.recommended_config()`` does
        it in one call. A start state taken from a precise orbit product
        (ESA's precise orbit ephemeris (POE), or an SP3 file) does NOT
        already include these forces: it fixes only the start, and the
        forces after it must be modeled. Both lower the 1-hour
        root-mean-square error (RMSE) over the 2025 Sentinels (drag on every
        one below 1000 km; SENTINEL-YEAR-2025); drag removes 81% of it at
        CHAMP's 370 km (CHAMP-370KM).
    """
    gravity_degree: int = 70
    enable_third_body: bool = True
    enable_relativistic: bool = True
    enable_solid_tides: bool = True
    ocean_tide_degree: int = 50
    itrf: '_core.ITRFRealization' = field(
        default_factory=lambda: _core.ITRFRealization.ITRF2020
    )
    integrator: '_core.IntegratorType' = field(
        default_factory=lambda: _core.IntegratorType.RK45
    )
    symplectic_step_size: float = 30.0
    symplectic_adaptive: bool = False
    symplectic_min_step: float = 1.0
    symplectic_max_step: float = 120.0
    symplectic_alpha: Optional[float] = None
    symplectic_symmetry_tol: float = 0.1
    symplectic_nc_strategy: '_core.NonConservativeStrategy' = field(
        default_factory=lambda: _core.NonConservativeStrategy.SLIMPLECTIC
    )
    rk45_initial_step_size: float = 60.0   # Initial RK45 step [s]; ~8 for glides
    rk45_max_step_size: float = 3600.0     # Max RK45 step [s]; ~8 for glides
    srp: Optional[Any] = None              # SrpConfig; None = radiation off
    drag_env: Optional[Any] = None         # DragEnvironment; None = drag off
    empirical_accel: Optional[Any] = None  # EmpiricalAcceleration object
    ea_reference_epoch: Optional[Any] = None  # Epoch keying EA time lookups
    report_forces: bool = False
    include_disabled_forces: bool = False

    def replace(self, **changes) -> 'TrajectoryConfig':
        """Return a copy with ``changes`` applied, e.g. ``cfg.replace(gravity_degree=120)``.

        Raises TypeError for an unknown field name.
        """
        return replace(self, **changes)

    def _to_core(self) -> '_core.TrajectoryConfig':
        """Build the equivalent ``_core.TrajectoryConfig`` for the C++ layer."""
        cfg = _core.TrajectoryConfig()
        cfg.gravity_degree = self.gravity_degree
        cfg.enable_third_body = self.enable_third_body
        cfg.enable_relativistic = self.enable_relativistic
        cfg.enable_solid_tides = self.enable_solid_tides
        cfg.ocean_tide_degree = self.ocean_tide_degree
        cfg.itrf = self.itrf
        cfg.integrator = self.integrator
        cfg.symplectic.step_size = self.symplectic_step_size
        cfg.symplectic.adaptive = self.symplectic_adaptive
        cfg.symplectic.min_step_size = self.symplectic_min_step
        cfg.symplectic.max_step_size = self.symplectic_max_step
        cfg.symplectic.adaptation_alpha = self.symplectic_alpha
        cfg.symplectic.symmetry_tolerance = self.symplectic_symmetry_tol
        cfg.symplectic.nc_strategy = self.symplectic_nc_strategy
        cfg.rk45.initial_step_size = self.rk45_initial_step_size
        cfg.rk45.max_step_size = self.rk45_max_step_size
        cfg.report_forces = self.report_forces
        cfg.include_disabled_forces = self.include_disabled_forces

        # Optional forces: presence is the switch
        if self.srp is not None:
            cfg.srp = self.srp
        if self.drag_env is not None:
            cfg.drag_env = self.drag_env

        # Empirical acceleration
        if self.empirical_accel is not None:
            cfg.empirical_accel = self.empirical_accel
        if self.ea_reference_epoch is not None:
            cfg.ea_reference_epoch = self.ea_reference_epoch

        return cfg


@dataclass(slots=True, kw_only=True)
class InterpolatorConfig:
    """
    How a Trajectory answers queries for a time between (or beyond) its steps.

    The output frame is chosen per call (``traj(t, frame='GCRS')``), not here.

    Attributes:
        itrf: ITRF realization (version of the Earth-fixed frame) used when
            converting to and from the inertial GCRS frame (None = the
            Trajectory's own).
        cache_coefficients: keep the interpolation polynomials between calls
            (faster for repeated queries).
        allow_propagation: when a query falls outside the propagated span,
            propagate further to cover it instead of returning None.
        max_propagation_time: the furthest, in seconds, such an automatic
            extension may go.
    """
    itrf: Optional['_core.ITRFRealization'] = None  # None = inherit from Trajectory
    cache_coefficients: bool = True
    allow_propagation: bool = True
    max_propagation_time: float = 3600.0

    def _to_core(self, output_frame: '_core.InterpolationFrame') -> '_core.InterpolatorConfig':
        """Build the equivalent ``_core.InterpolatorConfig`` with ``output_frame``."""
        cfg = _core.InterpolatorConfig()
        cfg.output_frame = output_frame
        if self.itrf is not None:
            cfg.itrf = self.itrf
        cfg.cache_coefficients = self.cache_coefficients
        cfg.allow_propagation = self.allow_propagation
        cfg.max_propagation_time = self.max_propagation_time
        return cfg


class Trajectory:
    """
    Propagate one satellite (or several) and read the orbit back at any time.

    A Trajectory starts from a ``ballistic.State`` (position and velocity at
    an Epoch, with the frame named), steps forward or backward on request,
    records every step, and interpolates between steps when called with a
    time. A query outside the recorded span propagates further
    automatically (see InterpolatorConfig).

    Single trajectory mode::

        state = State.from_itrf(epoch, position, velocity)   # or from_gcrs, from_sp3, ...
        traj = Trajectory(state)
        traj.propagate_for(3600.0)
        s = traj(1800.0)              # interpolate at 30 minutes after the start

    Batch mode::

        batch = Trajectory([state1, state2])
        batch.propagate_for(3600.0)   # every member, from its own epoch
        results = batch(1800.0)

    Args:
        state: a ballistic.State (see State.from_itrf, from_gcrs, from_j2000,
            from_keplerian, from_sp3, from_tle), or a non-empty list of them
            for batch mode.
        config: a TrajectoryConfig (default: the library defaults, using the
            state's ITRF realization when it names one).
        interp_config: an InterpolatorConfig (default: InterpolatorConfig()).

    Raises:
        RemovedApiError: for the old ``Trajectory(epoch, position, velocity)``
            form, which did not say whether the velocity was Earth-fixed or
            inertial.
        TypeError: if ``state`` is not a State or list of States, or
            ``config`` is not a ballistic.TrajectoryConfig.
        ConfigError: if batch states name different ITRF realizations and
            no config is given, or the state's realization disagrees with
            ``config.itrf``.
        InputPlausibilityError / FrameError (from building the State) are
            raised earlier, by the State constructors.
    """

    def __init__(
        self,
        state: Union['_core.State', List['_core.State']],
        config: Optional[TrajectoryConfig] = None,
        interp_config: Optional[InterpolatorConfig] = None,
        *legacy,
        **legacy_kw,
    ):
        if isinstance(state, _core.Epoch) or legacy or (
                isinstance(state, list) and state and isinstance(state[0], tuple)):
            raise RemovedApiError(
                "Trajectory(epoch, position, velocity) was removed: the triple did not say "
                "whether the velocity is Earth-fixed or inertial (they differ by ~470 m/s at "
                "LEO). Build a State: Trajectory(ballistic.State.from_itrf(epoch, r_m, v_m_s)) "
                "for an Earth-fixed velocity, State.from_gcrs for an inertial one "
                "(see docs/MIGRATION.md).")
        if legacy_kw:
            raise TypeError(f"Trajectory: unexpected arguments {sorted(legacy_kw)}")
        states = state if isinstance(state, list) else [state]
        if not states or not all(isinstance(x, _core.State) for x in states):
            raise TypeError("Trajectory takes a ballistic.State or a non-empty list of them")
        if config is None:
            reals = {x.realization for x in states} - {None}
            if len(reals) > 1:
                raise ConfigError("Trajectory: the batch's states are in different ITRF "
                                  "realizations; give them one")
            config = TrajectoryConfig(itrf=reals.pop()) if reals else TrajectoryConfig()
        if not isinstance(config, TrajectoryConfig):
            raise TypeError("config must be a ballistic.TrajectoryConfig")
        self._config = config
        self._interp_config = interp_config or InterpolatorConfig()
        self._core_config = self._config._to_core()

        if not isinstance(state, list):
            self._batch_mode = False
            self._traj = _core.Trajectory(state, self._core_config)
            self._initial_epoch = state.epoch
            self._interp: Optional[_core.TrajectoryInterpolator] = None
            self._stm_cache = None   # (key, _core.StateTransition), see _state_transition
        else:
            self._batch_mode = True
            self._batch = _core.TrajectoryBatch(states, self._core_config)
            self._initial_epoch = states[0].epoch
            self._interps: Optional[List[_core.TrajectoryInterpolator]] = None
        self._emit_warnings()   # e.g. a drag env built on forecast weather

    def __call__(
        self,
        times: Union[float, '_core.Epoch', np.ndarray, List, None] = None,
        frame: Union[str, '_core.InterpolationFrame'] = 'ECEF',
        on_out_of_range: str = 'none',
    ):
        """
        Read the orbit at given times, or get a reusable interpolation object.

        With no arguments, returns a reusable interpolation function object:
        - **Batch mode**: Returns ``BatchInterpolatorFunction`` with numpy array
          outputs and ``at_alt()`` for altitude crossing search.
        - **Single mode**: Returns the underlying ``TrajectoryInterpolator``.

        With time arguments, interpolates at the given times:
        - float: seconds after the start epoch (batch: the first member's)
        - Epoch: an absolute time
        - numpy array: float offsets in seconds, any shape (flattened)
        - list: floats and/or Epochs

        Args:
            times: time(s) to interpolate at, or None to get the function object.
            frame: output frame, 'ECEF' (Earth-fixed; default) or 'GCRS'
                (inertial), case-insensitive, or an InterpolationFrame.
            on_out_of_range: 'none' (default) returns None for times that
                cannot be covered; 'raise' raises ValueError instead, which
                is easier to debug than a stray None in later numpy code.

        Raises:
            TypeError: a time of an unsupported type.
            ValueError: an unknown frame or on_out_of_range value, or (with
                'raise') a time that cannot be covered.

        Returns:
            No args: BatchInterpolatorFunction (batch) or TrajectoryInterpolator (single)
            With times — single mode: InterpolatedState or list of InterpolatedState
            With times — batch mode: List of BatchInterpolatedState or list of lists

        Example::

            # Vectorized batch API
            f = batch_traj()           # BatchInterpolatorFunction
            r = f(1800.0)              # BatchArrayResult with (N,3) numpy arrays
            impact = f.at_alt(0)       # Ground impact search
            f[42].get_state(epoch)     # Debug single trajectory

            # Direct interpolation
            state = single_traj(1800.0)
        """
        if times is None:
            if self._batch_mode:
                return _core.BatchInterpolatorFunction(
                    self._batch, self._initial_epoch)
            else:
                self._ensure_interp()
                return self._interp

        frame_enum = self._parse_frame(frame)
        times_list = self._normalize_times(times)
        single_time = not isinstance(times, (list, np.ndarray)) or (
            isinstance(times, np.ndarray) and times.ndim == 0
        )

        if on_out_of_range not in ('none', 'raise'):
            raise ValueError("on_out_of_range must be 'none' or 'raise'")
        if self._batch_mode:
            out = self._call_batch(times_list, frame_enum, single_time)
        else:
            out = self._call_single(times_list, frame_enum, single_time)
        if on_out_of_range == 'raise':
            flat = out if isinstance(out, list) else [out]
            for i, st in enumerate(flat):
                inner = st if isinstance(st, list) else [st]
                if any(x is None for x in inner):
                    raise ValueError(
                        f"interpolation out of range at index {i} "
                        f"(time {times_list[min(i, len(times_list)-1)]})")
        self._emit_warnings()   # interpolation may have extended the history
        return out

    def _call_single(
        self,
        times: List['_core.Epoch'],
        frame: '_core.InterpolationFrame',
        single_time: bool,
    ) -> Union['_core.InterpolatedState', List['_core.InterpolatedState']]:
        """Interpolate the single trajectory at ``times`` in ``frame``."""
        self._ensure_interp()
        self._interp.set_output_frame(frame)

        if single_time and len(times) == 1:
            return self._interp.get_state(times[0])
        return self._interp.get_states(times)

    def _call_batch(
        self,
        times: List['_core.Epoch'],
        frame: '_core.InterpolationFrame',
        single_time: bool,
    ) -> Union[List['_core.BatchInterpolatedState'], List[List['_core.BatchInterpolatedState']]]:
        """Interpolate every batch member at ``times`` in ``frame``."""
        self._batch.set_output_frame(frame)

        if single_time and len(times) == 1:
            return self._batch.get_states_at(times[0])
        return self._batch.get_states_at_times(times)

    def _ensure_interp(self) -> None:
        """Create the interpolator on first use."""
        if self._interp is None:
            core_cfg = self._interp_config._to_core(_core.InterpolationFrame.ECEF)
            self._interp = _core.TrajectoryInterpolator(self._traj, core_cfg)

    def _normalize_times(self, times: Any) -> List['_core.Epoch']:
        """Turn any accepted ``times`` value into a list of Epochs (floats are
        seconds after the start epoch); raises TypeError otherwise."""
        base_epoch = self._get_base_epoch()

        if isinstance(times, (int, float)):
            return [base_epoch + float(times)]
        elif isinstance(times, _core.Epoch):
            return [times]
        elif isinstance(times, np.ndarray):
            if times.ndim == 0:
                return [base_epoch + float(times)]
            return [base_epoch + float(t) for t in times.ravel()]
        elif isinstance(times, list):
            result = []
            for t in times:
                if isinstance(t, (int, float)):
                    result.append(base_epoch + float(t))
                elif isinstance(t, _core.Epoch):
                    result.append(t)
                else:
                    raise TypeError(f"times list item must be float or Epoch, got {type(t)}")
            return result
        else:
            raise TypeError(f"times must be float, Epoch, list, or array, got {type(times)}")

    def _parse_frame(self, frame: Union[str, '_core.InterpolationFrame']) -> '_core.InterpolationFrame':
        """Turn a frame name ('ECEF'/'GCRS', any case) or enum into the enum;
        raises ValueError for an unknown name."""
        if isinstance(frame, _core.InterpolationFrame):
            return frame
        key = frame.upper()
        if key not in FRAME_MAP:
            valid = list(FRAME_MAP.keys())
            raise ValueError(f"Unknown frame '{frame}', use one of: {valid}")
        return FRAME_MAP[key]

    def _get_base_epoch(self) -> '_core.Epoch':
        """The epoch that float time offsets are measured from (the start)."""
        return self._initial_epoch

    # =========================================================================
    # Propagation Methods
    # =========================================================================

    def step(self) -> Union['_core.StepResult', None]:
        """
        Take one integration step forward (its size is chosen by the integrator).

        Returns:
            Single mode: a StepResult describing the step.
            Batch mode: None (every member steps once; see history()).
        """
        if self._batch_mode:
            self._batch.step_all(1)
            self._emit_warnings()
            return None
        result = self._traj.step()
        self._invalidate_caches()
        self._emit_warnings()
        return result

    def step_backward(self) -> Union['_core.StepResult', None]:
        """
        Take one integration step backward in time.

        Returns:
            Single mode: a StepResult describing the step.

        Raises:
            NotImplementedError: in batch mode.
        """
        if self._batch_mode:
            raise NotImplementedError("step_backward not supported for batch mode")
        result = self._traj.step_backward()
        self._invalidate_caches()
        self._emit_warnings()
        return result

    def propagate(self, *args, **kwargs):
        """Removed; raises RemovedApiError. Use propagate_for(seconds) or propagate_to(epoch)."""
        raise RemovedApiError(
            "Trajectory.propagate was removed: a float meant seconds from the INITIAL epoch, "
            "so calling it twice did not advance twice, and batch members all ran member 0's "
            "duration. Use propagate_for(seconds) (from the current epoch) or "
            "propagate_to(epoch) (see docs/MIGRATION.md).")

    def propagate_for(self, seconds: float):
        """Propagate for ``seconds`` from the CURRENT epoch (negative = backward).

        Calling it twice advances twice. In batch mode every member runs for
        ``seconds`` from its own epoch and lands exactly there.

        Args:
            seconds: duration in seconds (a float, not an Epoch).

        Returns:
            Single mode: a PropagationResult. Batch mode: a list of each
            member's final StateVector in the Earth-fixed (ECEF) frame.

        Raises:
            TypeError: if given an Epoch (use propagate_to).
            DataCoverageError: if the run goes past the end of the loaded
                data (space weather for drag, or Earth orientation data); the
                trajectory stops at the edge and keeps its state.
        """
        if isinstance(seconds, _core.Epoch):
            raise TypeError("propagate_for takes seconds; use propagate_to(epoch)")
        seconds = float(seconds)
        if self._batch_mode:
            self._batch.propagate_all(seconds)
            return self._batch_states()
        return self._propagate_single(self._traj.epoch() + seconds)

    def propagate_to(self, epoch: '_core.Epoch'):
        """Propagate to the absolute time ``epoch`` (forward or backward).

        In batch mode every member lands on that same epoch.

        Args:
            epoch: a ballistic.Epoch.

        Returns:
            As propagate_for.

        Raises:
            TypeError: if given a number (use propagate_for).
            DataCoverageError: as propagate_for.
        """
        if not isinstance(epoch, _core.Epoch):
            raise TypeError("propagate_to takes an Epoch; use propagate_for(seconds) for a duration")
        if self._batch_mode:
            self._batch.propagate_all_to(epoch)
            return self._batch_states()
        return self._propagate_single(epoch)

    def _propagate_single(self, target):
        # Forward runs go through propagate_until, which RECORDS each step,
        # so traj(t) afterwards interpolates the trajectory that was flown.
        # The C++ propagate() keeps no history; the interpolator would then
        # re-propagate on demand and return None for times more than
        # max_propagation_time from what it had (e.g. 60 s after the start
        # of a 3-hour run). Backward runs use propagate().
        if target - self._traj.epoch() > 0:
            self._traj.propagate_until(target, [], False)
        result = self._traj.propagate(target)
        self._invalidate_caches()
        self._emit_warnings()
        return result

    def _batch_states(self):
        """Each batch member's current Earth-fixed StateVector."""
        self._emit_warnings()
        return [self._batch.trajectory(i).state_ecef() for i in range(self._batch.size())]

    def propagate_until(
        self,
        target: Union['_core.Epoch', float],
        events: Optional[list] = None,
        stop_at_ground: bool = True,
    ) -> '_core.PropagateUntilResult':
        """Propagate forward to ``target``, stopping early at chosen events.

        Events are moments to find during the run: crossing an altitude, hitting
        the ground, the lowest (perigee) or highest (apogee) point of the orbit,
        or a given time. Each is located precisely inside the integration step
        where it happens. This is the way to fly a reentry.

        Args:
            target: an Epoch, or seconds from the CURRENT epoch (forward only).
            events: a list built with ``ballistic.events.altitude_crossing``,
                ``ground_impact``, ``perigee``, ``apogee`` or ``at_epoch``.
            stop_at_ground: also stop when the satellite reaches the ground
                (default True).

        Returns:
            A PropagateUntilResult: why it stopped (``stopped_by``) and the log
            of events found.

        Raises:
            ConfigError: in batch mode (build one Trajectory per state).
        """
        if self._batch_mode:
            raise ConfigError("propagate_until runs one trajectory at a time; "
                              "build a Trajectory per state")
        if isinstance(target, (int, float)):
            target = self._traj.epoch() + float(target)
        result = self._traj.propagate_until(target, list(events or []), stop_at_ground)
        self._invalidate_caches()
        return result

    # =========================================================================
    # State transition matrix and covariance
    # =========================================================================

    def stm(self, t, *, fd_dx: float = 1.0, fd_dv: float = 1e-3,
            max_substep_s: float = 10.0) -> np.ndarray:
        """State transition matrix Phi(t, t0) along the propagated trajectory.

        Phi maps a small change of the state at the first recorded epoch t0
        (normally the start) into the change at ``t``: dx(t) = Phi dx(t0),
        with x = [r (m); v (m/s)] in the inertial GCRS frame. It is integrated
        along the recorded steps, so propagate first (``propagate_for`` /
        ``propagate_to`` / ``propagate_until``); ``t`` must lie inside the
        propagated span.

        Args:
            t: seconds after the start, or an Epoch (as for ``traj(t)``); a
                list or array gives one matrix per time.
            fd_dx: finite-difference position offset [m] for drag's position
                derivatives (the only force term without an analytic one).
            fd_dv: finite-difference velocity offset [m/s] (used only when the
                drag coefficient depends on velocity).
            max_substep_s: largest integration substep [s] for Phi.

        Returns:
            A (6, 6) array, or (N, 6, 6) for N times.

        Raises:
            ConfigError: in batch mode, or with fewer than two recorded states.
            DataCoverageError: a time outside the propagated span.
        """
        st = self._state_transition(fd_dx, fd_dv, max_substep_s)
        times = self._normalize_times(t)
        out = [np.asarray(st.stm(e)) for e in times]
        return out[0] if self._is_single_time(t) else np.stack(out)

    def covariance(self, t, p0, *, fd_dx: float = 1.0, fd_dv: float = 1e-3,
                   max_substep_s: float = 10.0) -> np.ndarray:
        """Covariance Phi P_0 Phi^T: the start-state covariance carried to ``t``.

        ``p0`` is the 6x6 covariance of the state at the first recorded epoch,
        symmetric, GCRS, order [r (m); v (m/s)] (units m^2, m^2/s, m^2/s^2).
        No process noise and no consider parameters: the growth comes from
        the dynamics alone. Arguments and errors as for ``stm``; also raises
        InputPlausibilityError for a non-symmetric or non-finite ``p0``.

        Returns:
            A (6, 6) array, or (N, 6, 6) for N times.
        """
        st = self._state_transition(fd_dx, fd_dv, max_substep_s)
        p0 = np.ascontiguousarray(p0, dtype=np.float64)
        if p0.shape != (6, 6):
            raise ValueError(f"p0 (the initial covariance) must be 6x6, got shape {p0.shape}")
        times = self._normalize_times(t)
        out = [np.asarray(st.covariance(e, p0)) for e in times]
        return out[0] if self._is_single_time(t) else np.stack(out)

    @staticmethod
    def _is_single_time(t) -> bool:
        return not isinstance(t, (list, np.ndarray)) or (
            isinstance(t, np.ndarray) and t.ndim == 0)

    def _state_transition(self, fd_dx, fd_dv, max_substep_s):
        """The cached _core.StateTransition for these options, rebuilt when the
        options or the recorded history changed."""
        if self._batch_mode:
            raise ConfigError("stm is per trajectory; index the batch (build one "
                              "Trajectory per state)")
        hist = self._traj.history_size()
        front = self._traj.history_at(0).t_rel_s if hist else None
        key = (float(fd_dx), float(fd_dv), float(max_substep_s), hist, front)
        if self._stm_cache is None or self._stm_cache[0] != key:
            opts = _core.StmOptions()
            opts.fd_dx, opts.fd_dv, opts.max_substep_s = key[:3]
            self._stm_cache = (key, _core.StateTransition(self._traj, opts))
        return self._stm_cache[1]

    def _invalidate_caches(self) -> None:
        """The history changed: drop the interpolator's coefficient cache and
        the state transition matrix."""
        if self._interp is not None:
            self._interp.invalidate()
        self._stm_cache = None

    @property
    def warnings(self) -> List[str]:
        """Notes recorded during propagation, as strings.

        Examples: the run used forecast (not yet measured) space weather, or
        went past the time span of a tabulated drag environment. Each note is
        also issued as a ``ballistic.BallisticWarning`` when it happens.
        Batch mode: notes from every member, each prefixed with "[index]".
        """
        if self._batch_mode:
            return [f"[{i}] {w}" for i in range(self._batch.size())
                    for w in self._batch.trajectory(i).warnings()]
        return list(self._traj.warnings())

    def _emit_warnings(self) -> None:
        """Does nothing on purpose: the C++ trajectory reports each note the
        moment it is recorded, and that report arrives as a
        ``ballistic.BallisticWarning`` subclass (see ballistic.errors). The
        ``warnings`` property still lists them."""

    # =========================================================================
    # History Methods
    # =========================================================================

    def history(self) -> Union[List['_core.HistoryEntry'], List[List['_core.HistoryEntry']]]:
        """The recorded steps: a list of HistoryEntry (batch: one list per member)."""
        if self._batch_mode:
            return [self._batch.trajectory(i).history() for i in range(self._batch.size())]
        return self._traj.history()

    def history_size(self) -> Union[int, List[int]]:
        """Number of recorded steps (batch: a list, one count per member)."""
        if self._batch_mode:
            return [self._batch.trajectory(i).history_size() for i in range(self._batch.size())]
        return self._traj.history_size()

    def clear_history(self) -> None:
        """Forget the recorded steps (the current state is kept)."""
        if self._batch_mode:
            for i in range(self._batch.size()):
                self._batch.trajectory(i).clear_history()
        else:
            self._traj.clear_history()
            self._invalidate_caches()

    # =========================================================================
    # State Access
    # =========================================================================

    def epoch(self) -> Union['_core.Epoch', List['_core.Epoch']]:
        """The current Epoch (batch: a list, one per member)."""
        if self._batch_mode:
            return [self._batch.trajectory(i).epoch() for i in range(self._batch.size())]
        return self._traj.epoch()

    def state_ecef(self) -> Union['_core.StateVector', List['_core.StateVector']]:
        """The current StateVector in the Earth-fixed (ECEF) frame (batch: a list)."""
        if self._batch_mode:
            return [self._batch.trajectory(i).state_ecef() for i in range(self._batch.size())]
        return self._traj.state_ecef()

    def state_gcrs(self) -> Union['_core.StateVector', List['_core.StateVector']]:
        """The current StateVector in the inertial GCRS frame, the frame the
        propagation runs in (batch: a list)."""
        if self._batch_mode:
            return [self._batch.trajectory(i).state_gcrs() for i in range(self._batch.size())]
        return self._traj.state_gcrs()

    def state_eci(self):  # guard-exempt(ECI-NAME): raises RemovedApiError
        """Removed; raises RemovedApiError. Use state_gcrs()."""
        from .errors import RemovedApiError
        raise RemovedApiError(
            "Trajectory.state_eci was removed: 'ECI' named two different frames "
            "in this library. Use Trajectory.state_gcrs() (see docs/MIGRATION.md).")

    # =========================================================================
    # Altitude Propagation
    # =========================================================================

    def propagate_to_altitude(
        self,
        altitude_m: float,
        condition: '_core.AltitudeCondition',
        direction: Optional['_core.PropagationDirection'] = None,
        max_duration_s: float = 7 * 86400.0,
    ) -> Union['_core.PropagateToAltitudeResult', List['_core.BatchAltitudeResult']]:
        """
        Propagate until the satellite is above or below a given altitude.

        Altitude is height above the WGS84 reference ellipsoid (geodetic
        altitude), in meters.

        Args:
            altitude_m: target altitude in meters.
            condition: AltitudeCondition.Above or AltitudeCondition.Below.
            direction: PropagationDirection.Forward (default) or Backward.
            max_duration_s: how far to search, seconds (default 7 days).
                Going forward, the history ends at the crossing (located to
                1e-4 s).

        Returns:
            Single mode: PropagateToAltitudeResult
            Batch mode: List of BatchAltitudeResult
        """
        if direction is None:
            direction = _core.PropagationDirection.Forward

        if self._batch_mode:
            return self._batch.propagate_to_altitude(
                altitude_m, condition, direction, max_duration_s
            )
        else:
            self._ensure_interp()
            return self._interp.propagate_to_altitude(
                altitude_m, condition, direction, max_duration_s
            )

    # =========================================================================
    # Symplectic Support
    # =========================================================================

    def is_symplectic(self) -> bool:
        """True if the integrator is symplectic (YOSHIDA6/YOSHIDA8)."""
        if self._batch_mode:
            return self._batch.trajectory(0).is_symplectic()
        return self._traj.is_symplectic()

    def energy_drift(self) -> Union[float, List[float]]:
        """How much the orbital energy has drifted since the start (symplectic
        integrators only; batch: a list, one per member)."""
        if self._batch_mode:
            return [self._batch.trajectory(i).energy_drift() for i in range(self._batch.size())]
        return self._traj.energy_drift()

    # =========================================================================
    # Configuration
    # =========================================================================

    def config(self) -> TrajectoryConfig:
        """The TrajectoryConfig this trajectory was built with."""
        return self._config

    def interp_config(self) -> InterpolatorConfig:
        """The InterpolatorConfig this trajectory uses."""
        return self._interp_config

    # =========================================================================
    # Container Protocol
    # =========================================================================

    def __len__(self) -> int:
        """
        Number of trajectories.

        Returns:
            Single mode: 1
            Batch mode: Number of trajectories
        """
        if self._batch_mode:
            return self._batch.size()
        return 1

    @property
    def is_batch(self) -> bool:
        """True if this Trajectory holds a batch of satellites."""
        return self._batch_mode

    # =========================================================================
    # Low-level Access
    # =========================================================================

    def trajectory(self, index: int = 0) -> '_core.Trajectory':
        """
        The underlying C++ Trajectory object (for advanced use).

        Args:
            index: Trajectory index (batch mode only, ignored in single mode)

        Returns:
            The C++ Trajectory object
        """
        if self._batch_mode:
            return self._batch.trajectory(index)
        return self._traj

    def interpolator(self, index: int = 0) -> Optional['_core.TrajectoryInterpolator']:
        """
        The underlying C++ TrajectoryInterpolator object (for advanced use).

        Args:
            index: Trajectory index (batch mode only, ignored in single mode)

        Returns:
            The C++ TrajectoryInterpolator object, or None if not yet created
        """
        if self._batch_mode:
            return self._batch.interpolator(index)
        return self._interp

    # =========================================================================
    # Cleanup
    # =========================================================================

    def clear_cache(self) -> None:
        """Drop the cached interpolation polynomials (they are rebuilt on demand)."""
        if self._batch_mode:
            for i in range(self._batch.size()):
                interp = self._batch.interpolator(i)
                if interp is not None:
                    interp.clear_cache()
        elif self._interp is not None:
            self._interp.clear_cache()

    def __repr__(self) -> str:
        mode = "batch" if self._batch_mode else "single"
        count = len(self)
        integrator = "symplectic" if self.is_symplectic() else "RK45"
        return f"<Trajectory mode={mode} count={count} integrator={integrator}>"
