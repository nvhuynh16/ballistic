from collections.abc import Sequence
import enum
from typing import Annotated, overload

from numpy.typing import ArrayLike

from . import events as events
from ballistic.sp3 import initial_state as from_sp3
from ballistic.state import _from_tle as from_tle


class AccelObs:
    """
    On-board accelerometer; its constant bias is estimated together with the EA. Model: reading = EA + bias, i.e. the force model is assumed to hold gravity only, so all the non-gravitational acceleration the accelerometer feels (its specific force) is attributed to the EA. Samples are specific force in Earth-fixed (ECEF) axes [m/s²]: real accelerometers report in the spacecraft body frame, so rotate each sample with the attitude at its time before storing it.
    """

    def __init__(self) -> None: ...

    @property
    def known_bias(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """
        Known part of the bias, numpy (3,) [m/s²], subtracted from every sample; the remaining unknown bias is still estimated, with sigma_bias_prior as its expected size.
        """

    @known_bias.setter
    def known_bias(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def sigma_scale(self) -> list[float]:
        """Optional per-sample multiplier of sigma_accel (empty = none)."""

    @sigma_scale.setter
    def sigma_scale(self, arg: Sequence[float], /) -> None: ...

    @property
    def outlier_prob(self) -> float:
        """
        Optional outlier model: the noise is mostly Gaussian with σ = sigma_accel, but a fraction p = outlier_prob of samples are outliers with σ multiplied by k = outlier_scale, i.e. (1-p)·N(0,σ²) + p·N(0,(kσ)²). With p > 0 the smoother down-weights each sample by how far it falls from the prediction (the Masreliez robust update for this mixture). Measured with 5% of samples at 10σ (test_ea_smoother): error 21.8 m without it, 11.2 m with p and k wrong by 2.5× and 2×, so declare your best estimate. Default 0 = off.
        """

    @outlier_prob.setter
    def outlier_prob(self, arg: float, /) -> None: ...

    @property
    def outlier_scale(self) -> float:
        """Outlier noise multiplier k (default 10); see outlier_prob."""

    @outlier_scale.setter
    def outlier_scale(self, arg: float, /) -> None: ...

    @property
    def sigma_accel(self) -> float:
        """Accelerometer noise σ per axis [m/s²]. Default 0.01."""

    @sigma_accel.setter
    def sigma_accel(self, arg: float, /) -> None: ...

    @property
    def sigma_bias_prior(self) -> float:
        """
        Expected size (standard deviation) of the unknown bias [m/s²]. Default 1e-3 (a tactical-grade inertial measurement unit).
        """

    @sigma_bias_prior.setter
    def sigma_bias_prior(self, arg: float, /) -> None: ...

    @property
    def samples(self) -> list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]:
        """Observations: list of (Epoch, ECEF specific force [m/s²])."""

    @samples.setter
    def samples(self, arg: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], /) -> None: ...

    def set_samples(self, epoch0: Epoch, times_s: Annotated[ArrayLike, dict(dtype='float64', shape=(None), writable=False)], observations: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 3), writable=False)]) -> None:
        """
        Replace the samples from numpy arrays: times_s, shape (N,), seconds after epoch0; observations, shape (N, 3), ECEF specific force [m/s²]. Raises ValueError if the lengths differ.
        """

class AccelSamples:
    """
    Observed accelerations for estimate_ea_bspline: a list of (Epoch, numpy (3,) acceleration of the Earth-fixed (ECEF) coordinates [m/s²]).
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, samples: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]) -> None: ...

    @property
    def ecef(self) -> list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]:
        """The samples: list of (Epoch, numpy (3,)), ECEF"""

    @ecef.setter
    def ecef(self, arg: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], /) -> None: ...

class AeroBeta:
    """
    Ballistic coefficient beta = mass / (Cd * reference area) [kg/m^2], with Cd
    from a model (ConstantCd or FlowRegimeCd). Pass it as `beta` to
    make_drag_environment.

    Args:
        mass_kg: mass [kg], > 0
        reference_area_m2: reference area [m^2], > 0
        cd: ConstantCd or FlowRegimeCd
        reference_length_m: the body's size [m] for the Knudsen number (a
            sphere's diameter); required (> 0) with FlowRegimeCd

    Raises:
        ballistic.errors.ConfigError: a value above is out of range.
    """

    def __init__(self, mass_kg: float, reference_area_m2: float, cd: ConstantCd | FlowRegimeCd, reference_length_m: float = 0.0) -> None: ...

    @property
    def mass_kg(self) -> float:
        """Mass [kg]"""

    @property
    def reference_area_m2(self) -> float:
        """Reference area [m^2]"""

    @property
    def reference_length_m(self) -> float:
        """Reference length [m]"""

    def depends_on_flow(self) -> bool:
        """True if Cd changes with the flow (a FlowRegimeCd)"""

class AeroShape(enum.Enum):
    """
    Body shape for FlowRegimeCd. SPHERE: reference area pi D^2/4 and reference length D (diameter). FLAT_PLATE: facing the flow; reference area = plate area. CYLINDER: axis across the flow; reference area L*D, reference length D.
    """

    SPHERE = 0

    FLAT_PLATE = 1

    CYLINDER = 2

class AltitudeCondition(enum.Enum):
    """Stopping condition for TrajectoryInterpolator.propagate_to_altitude."""

    Above = 0
    """Propagate until the geodetic altitude is at or above the target"""

    Below = 1
    """Propagate until the geodetic altitude is at or below the target"""

class BSplineCoefficients:
    """
    B-spline coefficients for EmpiricalAcceleration.from_bspline(): one coefficient array per axis (cx, cy, cz, numpy (n,)), the knot vector (time points [s]) and the polynomial degree. Raises ValueError if the array sizes do not fit together (knots must have n + degree + 1 entries).
    """

    def __init__(self, cx: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], cy: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], cz: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')], knots: Sequence[float], degree: int) -> None: ...

    def n_basis(self) -> int:
        """Number of basis functions (= len(cx))."""

class BallisticError(RuntimeError):
    pass

class BallisticModel:
    """
    Ballistic coefficient and optional mass and area of a body.

    beta is the ballistic coefficient m / (Cd * A) [kg/m^2]: mass over drag
    coefficient times frontal area. mass and drag_area are optional metadata
    (mass may be used for radiation pressure, drag_area only for display). A
    Trajectory's drag does not read this class: it takes beta from
    TrajectoryConfig.drag_env.

    Attributes:
        beta: Ballistic coefficient [kg/m^2] (default: 45.45 = 1000 / (2.2 * 10))
        mass: Mass [kg] (optional, default 0)
        drag_area: Drag cross-section [m^2] (optional, default 0)

    Example:
        model = BallisticModel()              # Default beta=45.45
        model = BallisticModel(50.0)          # Explicit beta
        model = BallisticModel(50.0, 1000.0, 10.0)  # beta, mass, area
    """

    @overload
    def __init__(self) -> None:
        """beta = 45.45 kg/m^2, no mass or area"""

    @overload
    def __init__(self, beta: float) -> None:
        """Given ballistic coefficient beta [kg/m^2]"""

    @overload
    def __init__(self, beta: float, mass: float, area: float) -> None:
        """Given beta [kg/m^2], mass [kg] and drag area [m^2]"""

    @property
    def beta(self) -> float:
        """Ballistic coefficient [kg/m^2]"""

    @beta.setter
    def beta(self, arg: float, /) -> None: ...

    @property
    def mass(self) -> float:
        """Mass [kg] (optional metadata)"""

    @mass.setter
    def mass(self, arg: float, /) -> None: ...

    @property
    def drag_area(self) -> float:
        """Drag cross-section [m^2] (optional metadata)"""

    @drag_area.setter
    def drag_area(self, arg: float, /) -> None: ...

    def ballistic_coefficient(self) -> float:
        """The ballistic coefficient beta [kg/m^2]"""

    def inverse_ballistic_coefficient(self) -> float:
        """1 / beta [m^2/kg]"""

class BatchAltitudeResult:
    """Result of TrajectoryBatch.propagate_to_altitude for one trajectory."""

    def __init__(self) -> None: ...

    @property
    def trajectory_index(self) -> int:
        """Index of the trajectory in the batch"""

    @property
    def condition_met(self) -> bool:
        """True if the target altitude was reached"""

    @property
    def steps_taken(self) -> int:
        """Number of steps added to the history"""

    @property
    def starting_altitude_m(self) -> float:
        """Geodetic altitude (height above the WGS84 ellipsoid) at the start [m]"""

    @property
    def ending_altitude_m(self) -> float:
        """Geodetic altitude at the end [m]"""

class BatchArrayResult:
    """
    States of all trajectories of a batch at one query time, as numpy arrays
    with N rows (one per trajectory).

    Attributes:
        position: (N, 3) position array in output frame (meters)
        velocity: (N, 3) velocity array in output frame (m/s)
        acceleration: (N, 3) acceleration array in output frame (m/s^2)
        epoch_s: (N,) query times in seconds after the reference epoch
        valid: (N,) 1 where the row is valid, 0 where the query failed
    """

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """Position array (N, 3) in output frame (meters)"""

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """Velocity array (N, 3) in output frame (m/s)"""

    @property
    def acceleration(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, None), order='F')]:
        """Acceleration array (N, 3) in output frame (m/s^2)"""

    @property
    def epoch_s(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C')]:
        """Query times (N,) in seconds after the reference epoch"""

    @property
    def valid(self) -> Annotated[ArrayLike, dict(dtype='int32', shape=(None), order='C')]:
        """Validity mask (N,): 1 = valid, 0 = query failed"""

    def __len__(self) -> int:
        """Number of trajectories (rows)"""

    def __repr__(self) -> str: ...

class BatchInterpolatedState:
    """State of one trajectory of a TrajectoryBatch at a query time."""

    def __init__(self) -> None: ...

    @property
    def trajectory_index(self) -> int:
        """Index of the trajectory in the batch"""

    @property
    def valid(self) -> bool:
        """
        True if the query succeeded (otherwise the state fields are meaningless)
        """

    @property
    def epoch(self) -> Epoch:
        """Query time (Epoch)"""

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Position in output frame (meters)"""

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Velocity in output frame (m/s)"""

    @property
    def acceleration(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Acceleration in output frame (m/s^2)"""

    @property
    def was_propagated(self) -> bool:
        """True if the history had to be extended to answer this query"""

class BatchInterpolatorFunction:
    """
    Interpolates all trajectories of a batch at once and returns numpy arrays
    instead of one Python object per trajectory. Obtained by calling a batch
    Trajectory with no argument, ``traj()``. Times given as floats are seconds
    after the reference epoch.

    Example:
        batch = ballistic.Trajectory(states_list)
        batch.propagate_for(3600.0)

        f = batch()                   # BatchInterpolatorFunction
        r = f(1800.0)                 # BatchArrayResult
        r.position                    # (N, 3) numpy array

        impact = f.at_alt(0)          # Find ground impact
        f[42].get_state(epoch + 1800) # Debug single trajectory
    """

    def __init__(self, batch: TrajectoryBatch, reference_epoch: Epoch) -> None:
        """
        Wrap a TrajectoryBatch; float times are counted from reference_epoch (Epoch)
        """

    def __call__(self, times: object) -> object:
        """
        The states of all trajectories at one or several times.

        Args:
            times: float (seconds from reference epoch), Epoch, or
                   list/array of floats or Epochs

        Returns:
            BatchArrayResult for single time, or list of BatchArrayResult
            for multiple times
        """

    def at_alt(self, altitude_m: float, descending: bool = True, max_time_s: float = 0.0) -> BatchArrayResult:
        """
        Find where each trajectory crosses a geodetic altitude, within the history
        already computed (nothing is propagated).

        Scans each history for the step where the altitude crosses the target, then
        bisects inside it to about 1 mm in altitude.

        Args:
            altitude_m: Target geodetic altitude [m]
            descending: If True (default), find last descending crossing
                        (above→below, typical for reentry/impact).
                        If False, find last ascending crossing.
            max_time_s: Maximum time window in seconds from reference epoch.
                        0 (default) = search entire history.

        Returns:
            BatchArrayResult with states at crossing points.
            Trajectories that never cross have valid=0.
        """

    def __getitem__(self, index: int) -> TrajectoryInterpolator:
        """
        The TrajectoryInterpolator of trajectory `index` (negative indices count from the end)
        """

    def __len__(self) -> int:
        """Number of trajectories"""

    def set_frame(self, frame: InterpolationFrame) -> None:
        """Choose the coordinate frame (InterpolationFrame) of all query results"""

    def __repr__(self) -> str: ...

class BetaDecompositionConfig:
    """
    Settings for estimating the ballistic coefficient from an EA (refine_beta_from_ea).
    """

    def __init__(self) -> None: ...

    @property
    def min_altitude_m(self) -> float:
        """Ignore points below this altitude [m] (default: 50000)"""

    @min_altitude_m.setter
    def min_altitude_m(self, arg: float, /) -> None: ...

    @property
    def min_beta(self) -> float:
        """Discard point estimates below this beta [kg/m^2] (default: 1.0)"""

    @min_beta.setter
    def min_beta(self, arg: float, /) -> None: ...

    @property
    def max_beta(self) -> float:
        """Discard point estimates above this beta [kg/m^2] (default: 1e5)"""

    @max_beta.setter
    def max_beta(self, arg: float, /) -> None: ...

    @property
    def altitude_nodes_km(self) -> list[float]:
        """Optional altitude bins [km] for an altitude-dependent beta"""

    @altitude_nodes_km.setter
    def altitude_nodes_km(self, arg: Sequence[float], /) -> None: ...

class BetaProfile1D:
    """
    Ballistic coefficient as a function of altitude. The ballistic coefficient
    beta = m / (Cd * A) [kg/m^2] (mass over drag coefficient times frontal area)
    sets how strongly drag decelerates a body: the larger beta, the weaker the
    effect. Used as the beta column of a DragEnvironment.

    Build with BetaProfile1D(altitude_km, beta_kg_m2) (two equal-length lists:
    altitudes [km] and beta values [kg/m^2]), or BetaProfile1D.constant(beta) /
    from_mass_cd_area(...) for the common case of a beta that does not vary with
    altitude.
    """

    @overload
    def __init__(self) -> None:
        """Empty profile."""

    @overload
    def __init__(self, altitude_km: Sequence[float], beta_kg_m2: Sequence[float]) -> None:
        """Profile through the points (altitude_km [km], beta_kg_m2 [kg/m^2])."""

    @staticmethod
    def constant(beta_kg_m2: float, altitude_min_km: float = 0.0, altitude_max_km: float = 1500.0) -> BetaProfile1D:
        """
        Constant beta [kg/m^2] between altitude_min_km and altitude_max_km [km].
        """

    @staticmethod
    def from_mass_cd_area(mass_kg: float, cd: float, area_m2: float, altitude_min_km: float = 0.0, altitude_max_km: float = 1500.0) -> BetaProfile1D:
        """
        Constant beta = mass_kg / (cd * area_m2) [kg/m^2] from a satellite's mass [kg], drag coefficient and frontal area [m^2].
        """

    def empty(self) -> bool:
        """True if the profile holds no values."""

class BetaResult:
    """Ballistic coefficients estimated from an EA along an orbit."""

    def __init__(self) -> None: ...

    @property
    def step_estimates(self) -> list[BetaStepEstimate]:
        """Estimate at each point (list of BetaStepEstimate)"""

    @property
    def constant_beta(self) -> float:
        """Median of the point estimates [kg/m^2]"""

    @property
    def node_altitudes_km(self) -> list[float]:
        """Altitudes of the bins used for an altitude-dependent beta [km]"""

    @property
    def node_betas(self) -> list[float]:
        """Beta in each altitude bin [kg/m^2]"""

    @property
    def steps_solved(self) -> int:
        """Number of valid point estimates"""

    @property
    def error_message(self) -> str:
        """Error description (empty if success)"""

class BetaStepEstimate:
    """Ballistic coefficient estimated at one point of the orbit."""

    def __init__(self) -> None: ...

    @property
    def time_s(self) -> float:
        """Seconds after epoch0"""

    @property
    def altitude_km(self) -> float:
        """Geodetic altitude (height above the WGS84 ellipsoid) [km]"""

    @property
    def beta(self) -> float:
        """Ballistic coefficient [kg/m^2]"""

    @property
    def w(self) -> float:
        """1 / beta [m^2/kg]"""

    @property
    def weight(self) -> float:
        """Confidence weight of this estimate"""

class CVMode(enum.Enum):
    """What the cross-validation search (EABSplineConfig.set_cv) varies."""

    IRGN = 0
    """
    IRGN (iteratively regularized Gauss-Newton): knot spacing × λ₀ × decay × λ_min, about 495 combinations; the smoothing weight λ shrinks over the iterations. Default.
    """

    FixedLambda = 1
    """
    Knot spacing × a constant λ, about 99 combinations; about 5× faster, but suited to acceleration data only.
    """

class ConfigError(BallisticError):
    pass

class ConstantCd:
    """A drag coefficient (dimensionless) that does not vary with the flow."""

    def __init__(self, cd: float) -> None:
        """A constant drag coefficient `cd` (> 0)"""

    @property
    def cd(self) -> float:
        """The drag coefficient (must be > 0)"""

    @cd.setter
    def cd(self, arg: float, /) -> None: ...

class DataCoverageError(BallisticError):
    pass

class DragEnvironment:
    """
    Everything drag needs: air density (rho), wind and the ballistic coefficient
    (beta), as three columns. Set it as TrajectoryConfig.drag_env to turn drag
    on (leaving it unset means no drag).

    Build one with:
        ballistic.make_drag_environment(...)     # recommended: the "live" models
        ballistic.from_models_snapshot(t, lat, lon, sw, beta_prior, alt_grid_km)
        ballistic.from_models_along(epochs, positions_ecef, sw, beta_prior, alt_grid_km)
        ballistic.from_models_4d(alt, lat, lon, epochs, sw, beta_prior)
        ballistic.from_accelerometer_density(epochs, measured_density, beta_prior)

    A live environment (make_drag_environment) evaluates NRLMSIS 2.1 and HWM14
    at the satellite's exact position on every force evaluation. The other
    builders make a table once; the propagator then only reads from the table.
    """

    def __init__(self) -> None:
        """Empty environment (no drag)."""

    def evaluate(self, position_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], epoch: Epoch, velocity_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')] | None = None) -> DragSample:
        """
        Density, wind, beta and related values (a DragSample) at the Earth-fixed position position_ecef [m] (numpy (3,)) and time epoch. A FlowRegimeCd beta also needs velocity_ecef (velocity relative to the rotating Earth [m/s]), because its Cd depends on the speed through the air.
        """

    def beta_depends_on_velocity(self) -> bool:
        """True when beta is an AeroBeta with a FlowRegimeCd."""

    def has_drag(self) -> bool:
        """
        True if both the density and beta columns are filled in (drag will act).
        """

    def domain(self) -> tuple[Epoch, Epoch] | None:
        """
        (start, end) Epochs covered by a tabulated environment's time-varying columns, or None (altitude-only profiles, or a live environment: see weather_window). Propagating past it holds the atmosphere frozen at its last values.
        """

    def is_live(self) -> bool:
        """
        True for an environment from make_drag_environment: NRLMSIS 2.1 / HWM14 evaluated at the satellite's position, valid wherever space weather is available (weather_window).
        """

    def weather_window(self) -> tuple[Epoch, Epoch] | None:
        """
        (start, end) Epochs over which a live environment has space weather, or None (a tabulated environment, or constant weather valid at any time). A Trajectory stops with DataCoverageError at its edge.
        """

    def measured_until(self) -> Epoch | None:
        """
        The Epoch where a live environment's weather turns from measured to CelesTrak's forecast, or None (no forecast days).
        """

    def uses_predicted_weather(self) -> bool:
        """
        True if the environment reaches into forecast space weather (days past the last measured day).
        """

    def without_wind(self) -> DragEnvironment:
        """
        Return a copy of this environment without wind (the atmosphere co-rotates
        with the Earth and has no wind of its own). Density and beta are unchanged.
        Useful for comparing results with and without wind.
        """

class DragSample:
    """
    Air density, wind and ballistic coefficient at one position and time.
    Returned by DragEnvironment.evaluate().
    """

    @property
    def rho_kg_m3(self) -> float:
        """Atmospheric density (kg/m^3)."""

    @property
    def wind_ecef_m_s(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Wind velocity, Earth-fixed (ECEF) axes, numpy (3,) [m/s]."""

    @property
    def beta_kg_m2(self) -> float:
        """Ballistic coefficient m/(Cd*A) (kg/m^2)."""

    @property
    def time_clamped(self) -> bool:
        """
        True if the time fell outside a tabulated environment's time span, so its time-varying values were held at their first or last entry.
        """

    @property
    def temperature_k(self) -> float:
        """Air temperature (K) from the live model; NaN for a tabulated density."""

    @property
    def molar_mass_kg_kmol(self) -> float:
        """
        Mean molar mass of the air (kg/kmol) from the live model; NaN for a table.
        """

    @property
    def cd(self) -> float:
        """
        Drag coefficient used, when beta is an AeroBeta; NaN for a beta profile.
        """

    @property
    def knudsen(self) -> float:
        """
        Knudsen number (mean free path / body size) for a FlowRegimeCd; NaN otherwise.
        """

    @property
    def mach(self) -> float:
        """Mach number for a FlowRegimeCd; NaN otherwise."""

class EABSplineConfig:
    """
    Settings for estimate_ea_bspline, which fits the empirical acceleration as a B-spline (a smooth piecewise polynomial in time). A smoothness penalty with weight λ keeps the EA from chasing noise; set_irgn / set_gcv / set_cv / set_fixed_lambda choose how λ is picked (default: IRGN, iteratively regularized Gauss-Newton, where λ starts large and shrinks each iteration).
    """

    def __init__(self) -> None: ...

    @property
    def noise_sigma(self) -> float:
        """
        Observation noise standard deviation σ in the data's units (m for positions, m/s for velocities, m/s² for accelerations). If > 0, the fit stops once the residual RMS is at most 1.5σ. Typical levels: GPS positions 1-10 m (3 m is a good default assumption), Doppler velocities 0.01-0.25 m/s, accelerometers 1e-6 to 0.1 m/s².
        """

    @noise_sigma.setter
    def noise_sigma(self, arg: float, /) -> None: ...

    @property
    def knot_spacing_s(self) -> float:
        """
        Time between spline knots [s] (default: 2.0); smaller lets the EA change faster
        """

    @knot_spacing_s.setter
    def knot_spacing_s(self, arg: float, /) -> None: ...

    @property
    def degree(self) -> int:
        """B-spline polynomial degree: 5 = quintic (default), 3 = cubic"""

    @degree.setter
    def degree(self, arg: int, /) -> None: ...

    @property
    def max_iterations(self) -> int:
        """
        Maximum outer iterations; the fit also stops after 5 iterations in a row without improvement (default: 50)
        """

    @max_iterations.setter
    def max_iterations(self, arg: int, /) -> None: ...

    @property
    def max_duration_s(self) -> float:
        """
        Longest arc fitted [s] (default: 3600). Reduced automatically to the last observation time + 10 s; set it below the data span only to fit deliberately less of the data
        """

    @max_duration_s.setter
    def max_duration_s(self, arg: float, /) -> None: ...

    @property
    def reg_deriv_order(self) -> int | None:
        """
        Which time derivative of the spline the smoothness penalty acts on (None = automatic: 3 for positions, 1 for velocities and accelerations); must not exceed degree.
        """

    @reg_deriv_order.setter
    def reg_deriv_order(self, arg: int | None) -> None: ...

    @property
    def gn(self) -> GNRefine | None:
        """
        Gauss-Newton refinement settings (GNRefine), on by default; None skips it (faster, less accurate). Not applied to acceleration data. With cross-validation (set_cv), the GN settings are passed on to the GN runs inside the search.
        """

    @gn.setter
    def gn(self, arg: GNRefine | None) -> None: ...

    @property
    def brent_line_search(self) -> bool:
        """
        How the length of each update step is searched. True (default): bracket the best length from 5 trials, then refine with Brent's method; about 2-3× fewer propagations than False (24 fixed trial lengths), and more accurate for positions and accelerations. For velocities the mean error is the same but the spread is wider; consider False for a single velocity fit.
        """

    @brent_line_search.setter
    def brent_line_search(self, arg: bool, /) -> None: ...

    def set_irgn(self, lambda_0: float = 0.0001, decay: float = 0.5, lambda_min: float = 1e-12) -> None:
        """
        Pick λ by IRGN (the default): iteration k uses λ_k = max(lambda_min, lambda_0 * decay^k), so the first iterations fit only the broad shape and later ones the detail.
        """

    def set_gcv(self, lam_min: float = 1e-12, lam_max: float = 0.1) -> None:
        """
        Pick λ at each iteration by generalized cross-validation (GCV), from the data alone, searched within [lam_min, lam_max]; needs no noise level.
        """

    def set_cv(self, hv_n_blocks: int = 18, hv_test_blocks: Sequence[int] = [], hv_gap_blocks: int = 1, mode: CVMode = CVMode.IRGN) -> None:
        """
        Pick the settings by a cross-validation grid search: each combination is fitted on part of the data and scored on how well it predicts the held-out rest. The data are split into hv_n_blocks contiguous time blocks; hv_test_blocks lists the held-out ones (empty = every 5th starting at block 2); hv_gap_blocks blocks on each side of a held-out block are left out of training, so correlated neighbors do not leak into the score. mode=IRGN (default, about 495 combinations) or mode=FixedLambda (about 99, about 5× faster, acceleration data only). Slow, but the best automatic choice for noisy data.
        """

    def set_fixed_lambda(self, lambda_: float = 0.0001) -> None:
        """Use the same λ in every iteration. Suited to acceleration data only."""

class EADecomposition:
    """
    An EA vector split into a drag-like part (along the velocity) and the rest (perpendicular to it).
    """

    def __init__(self) -> None: ...

    @property
    def beta(self) -> float:
        """Ballistic coefficient implied by the drag part [kg/m^2]"""

    @property
    def ea_drag(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Part along the velocity, numpy (3,) [m/s^2]"""

    @property
    def ea_residual(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Part perpendicular to the velocity, numpy (3,) [m/s^2]"""

    @property
    def drag_fraction(self) -> float:
        """|drag part| / |EA|, between 0 and 1"""

class EAInformationBound:
    """
    Result of information_bound: the best position accuracy any estimator could reach with these sensors. Accelerometers are not included.
    """

    @property
    def posterior_bound_m(self) -> float:
        """
        RMS position bound [m] with the smoothness prior (lambda_, lambda_amp): what a regularized estimator can reach
        """

    @property
    def crb_bound_m(self) -> float:
        """
        RMS position bound [m] without a prior (the Cramér-Rao bound), over the directions the data can see
        """

    @property
    def n_params(self) -> int:
        """Number of unknowns (3 per EA time point)"""

    @property
    def n_obs_rows(self) -> int:
        """Number of scalar observations used"""

    @property
    def error_message(self) -> str:
        """Why the bound was not computed (empty = computed)"""

    def ok(self) -> bool:
        """True when the bound was computed"""

class EAInformationBoundOptions:
    """Settings for information_bound."""

    def __init__(self) -> None: ...

    @property
    def knot_spacing_s(self) -> float:
        """
        Spacing of the EA time points [s] (default 8.0). A finer spacing raises the bound (more unknowns, same data), so compare sensor suites at one spacing.
        """

    @knot_spacing_s.setter
    def knot_spacing_s(self, arg: float, /) -> None: ...

    @property
    def lambda_(self) -> float:
        """Prior weight on EA changes between neighboring time points (smoothness; default 1.0)"""
    @lambda_.setter
    def lambda_(self, arg: float, /) -> None: ...

    @property
    def lambda_amp(self) -> float:
        """Prior weight on the EA's size (default 1e-2)"""

    @lambda_amp.setter
    def lambda_amp(self, arg: float, /) -> None: ...

class EAInterp(enum.Enum):
    """
    How a tabulated empirical acceleration is interpolated between its time points.
    """

    Cubic = 0
    """Cubic spline (default)"""

    Linear = 1
    """Piecewise-linear"""

class EAMethod(enum.Enum):
    """Which estimator estimate_ea(spec) ran."""

    BSplinePosition = 0
    """estimate_ea_bspline on position samples (with Gauss-Newton refinement)"""

    BSplineVelocity = 1
    """estimate_ea_bspline on velocity samples (with Gauss-Newton refinement)"""

    BSplineAcceleration = 2
    """estimate_ea_bspline on acceleration samples"""

    Smoother = 3
    """estimate_ea_smoother on a MultiSensorScenario"""

class EAProblemSpec:
    """
    Input to estimate_ea: the observations, their declared noise and the force model. The library picks the estimator and then checks the declared noise against the fit. observations is one of PositionSamples, VelocitySamples, AccelSamples or MultiSensorScenario; the first three require noise_sigma > 0.
    """

    def __init__(self) -> None: ...

    @property
    def epoch0(self) -> Epoch:
        """
        Start time (Epoch); ignored for a MultiSensorScenario, which has its own
        """

    @epoch0.setter
    def epoch0(self, arg: Epoch, /) -> None: ...

    @property
    def pos0_ecef(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Start position, Earth-fixed (ECEF), numpy (3,) [m]"""

    @pos0_ecef.setter
    def pos0_ecef(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def vel0_ecef(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Start velocity relative to the rotating Earth, numpy (3,) [m/s]"""

    @vel0_ecef.setter
    def vel0_ecef(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def observations(self) -> PositionSamples | VelocitySamples | AccelSamples | MultiSensorScenario:
        """PositionSamples, VelocitySamples, AccelSamples or MultiSensorScenario"""

    @observations.setter
    def observations(self, arg: PositionSamples | VelocitySamples | AccelSamples | MultiSensorScenario, /) -> None: ...

    @property
    def force_model(self) -> TrajectoryConfig:
        """TrajectoryConfig used in the fit"""

    @force_model.setter
    def force_model(self, arg: TrajectoryConfig, /) -> None: ...

    @property
    def noise_sigma(self) -> float:
        """
        Declared observation noise σ for position / velocity / acceleration samples, in their units (m, m/s, m/s²). Required (> 0) for them unless `bspline` is given; ignored for scenarios (each sensor carries its own).
        """

    @noise_sigma.setter
    def noise_sigma(self, arg: float, /) -> None: ...

    @property
    def outlier_prob(self) -> float:
        """
        Optional outlier model for position / acceleration samples: a fraction p of samples have noise k·σ instead of σ. p > 0 removes likely outliers with reject_heavy_tails() before fitting. Measured with 5% of samples at 10σ: acceleration data 2.85 → 0.91 m, positions 0.60 → 0.37 m. Not applied to velocity samples (measured to make velocity fits far worse; a warning says so). Ignored for scenarios (declare it on the sensors).
        """

    @outlier_prob.setter
    def outlier_prob(self, arg: float, /) -> None: ...

    @property
    def outlier_scale(self) -> float:
        """Outlier noise multiplier k (default 10); see outlier_prob."""

    @outlier_scale.setter
    def outlier_scale(self, arg: float, /) -> None: ...

    @property
    def bspline(self) -> EABSplineConfig | None:
        """Optional EABSplineConfig to use instead of the defaults"""

    @bspline.setter
    def bspline(self, arg: EABSplineConfig, /) -> None: ...

    @property
    def smoother(self) -> EASmootherConfig | None:
        """Optional EASmootherConfig to use instead of the defaults"""

    @smoother.setter
    def smoother(self, arg: EASmootherConfig, /) -> None: ...

class EARepresents(enum.Enum):
    """
    What quantity a B-spline empirical acceleration was fitted to, which decides how many times the spline is differentiated to give the acceleration.
    """

    Acceleration = 0
    """The spline's value is the acceleration (no differentiation)"""

    Velocity = 1
    """The spline's first time derivative is the acceleration"""

    Position = 2
    """The spline's second time derivative is the acceleration"""

class EAResult:
    """Result of an empirical acceleration (EA) estimation."""

    def __init__(self) -> None: ...

    @property
    def empirical_accel(self) -> EmpiricalAcceleration:
        """
        The estimated EA (EmpiricalAcceleration), in the inertial GCRS frame; set it as TrajectoryConfig.empirical_accel to propagate with it.
        """

    @property
    def collocation_history(self) -> list[HistoryEntry]:
        """
        The orbit propagated with the estimated EA, sampled at the EA's time points (list of HistoryEntry, Earth-fixed). Pass it to refine_beta_from_ea.
        """

    @property
    def iterations_used(self) -> int:
        """Number of outer iterations performed."""

    @property
    def gn_status(self) -> GNStatus:
        """What happened to the optional Gauss-Newton refinement (GNStatus)."""

    @property
    def lambda_used(self) -> float | None:
        """Smoothing weight λ at the end (None if not used)."""

    @property
    def sigma_sq_estimate(self) -> float | None:
        """
        Noise variance per component estimated by generalized cross-validation (GCV) (None if GCV was not used).
        """

    @property
    def obs_rmse(self) -> float | None:
        """
        Root-mean-square (RMS) misfit to the observations at the end, in the observations' units (None if not measured).
        """

    @property
    def error_message(self) -> str:
        """Error description (empty on success)."""

    def succeeded(self) -> bool:
        """
        True if the estimator ran without error and produced an EA (an EA that is legitimately zero still counts as success).
        """

    def has_correction(self) -> bool:
        """
        True if succeeded() and the EA is not zero. Check this before using the EA for further refinement (GN or beta).
        """

class EASmootherConfig:
    """
    Settings for estimate_ea_smoother. Deliberately few: leave sigma_a and tau_s unset to have them chosen from the data (by NIS), and rate_budget at Auto to have the range-rate weight measured from the data.
    """

    def __init__(self) -> None: ...

    @property
    def sigma_a(self) -> float | None:
        """
        Expected EA size σ_a [m/s²]; None = chosen by NIS (closest to 1) from {1, 3, 10}. That grid suits reentry glides; for an ordinary orbit, values near 1e-6 m/s² fit (see regime_aware_prior).
        """

    @sigma_a.setter
    def sigma_a(self, arg: float, /) -> None: ...

    @property
    def tau_s(self) -> float | None:
        """
        How long the EA stays correlated, τ [s]; None = chosen by NIS from {10, 35, 100}. For an ordinary orbit about 600 s fits. Setting both sigma_a and tau_s skips the grid (9× faster).
        """

    @tau_s.setter
    def tau_s(self, arg: float, /) -> None: ...

    @property
    def rate_budget(self) -> EASmootherRateBudget:
        """How radar range rate is weighted (EASmootherRateBudget; default Auto)"""

    @rate_budget.setter
    def rate_budget(self, arg: EASmootherRateBudget, /) -> None: ...

    @property
    def sigma_rate_manual(self) -> float:
        """Range-rate σ [m/s] used when rate_budget is Manual"""

    @sigma_rate_manual.setter
    def sigma_rate_manual(self, arg: float, /) -> None: ...

    @property
    def max_dt_s(self) -> float:
        """
        Largest integration step [s] between observations (fixed-step 4th-order Runge-Kutta; default 0.05)
        """

    @max_dt_s.setter
    def max_dt_s(self, arg: float, /) -> None: ...

    @property
    def jac_refresh(self) -> int:
        """
        Recompute the gravity gradient (how gravity changes with position) every K steps; it changes slowly (default 16)
        """

    @jac_refresh.setter
    def jac_refresh(self, arg: int, /) -> None: ...

    @property
    def output_dt_s(self) -> float:
        """Time spacing [s] of the returned EA table (default 0.5)"""

    @output_dt_s.setter
    def output_dt_s(self, arg: float, /) -> None: ...

    @property
    def passes(self) -> int:
        """
        Number of filter/smoother passes (default 1). 2 = run again with the first pass's EA added to the dynamics (an iterated extended Kalman smoother).
        """

    @passes.setter
    def passes(self, arg: int, /) -> None: ...

    @property
    def estimate_clock_bias(self) -> bool:
        """
        Estimate a time-tag (clock) offset [s] for sensors whose clock is Shared or Own (default True). An unmodeled clock error is the largest measured risk with real data (in the radar test a 5 ms offset gave 316 m without this state, 52 m with it); harmless when the clocks are right.
        """

    @estimate_clock_bias.setter
    def estimate_clock_bias(self, arg: bool, /) -> None: ...

    @property
    def clock_bias_prior_s(self) -> float:
        """Expected size of the clock offset [s] (default 0.02)"""

    @clock_bias_prior_s.setter
    def clock_bias_prior_s(self, arg: float, /) -> None: ...

    @property
    def regime_aware_prior(self) -> bool:
        """
        Choose the sigma_a candidates from the data instead of the fixed {1, 3, 10} m/s² grid, which suits reentry glides and is about 7 orders of magnitude too large for an ordinary orbit. A first fit with the EA held at zero measures the position residual the dynamics cannot explain, and the candidates are placed around the acceleration scale it implies (see prior_anchor). Measured better in every test, but the reported uncertainty inside data gaps becomes too small, so it is off by default. Costs one extra fit. Ignored when sigma_a is set.
        """

    @regime_aware_prior.setter
    def regime_aware_prior(self, arg: bool, /) -> None: ...

    @property
    def maneuver_adaptive_q(self) -> bool:
        """
        Maneuver handling (default False): after an observation that fits much worse than expected, let the EA change faster for a while, so a sudden change (a thruster firing) is followed instead of smeared out.
        """

    @maneuver_adaptive_q.setter
    def maneuver_adaptive_q(self, arg: bool, /) -> None: ...

    @property
    def verbose(self) -> bool:
        """Print progress (grid values, NIS) to stdout"""

    @verbose.setter
    def verbose(self, arg: bool, /) -> None: ...

class EASmootherRateBudget(enum.Enum):
    """
    How the smoother weights radar range-rate samples. The weight must reflect how well the model can actually match range rate, not the datasheet noise.
    """

    Auto = 0
    """
    Default: fit once without range rate, measure its residual RMS, and use the larger of that and the sensor's sigma_rate
    """

    Off = 1
    """Ignore range rate"""

    Manual = 2
    """Use EASmootherConfig.sigma_rate_manual [m/s]"""

class EASmootherResult:
    """
    Result of estimate_ea_smoother: the EA (in the inertial GCRS frame) and quality figures that need no true answer to compute (nis about 1 = the declared noise matches the data; ea_sigma = the EA's 1σ uncertainty).
    """

    def __init__(self) -> None: ...

    @property
    def empirical_accel(self) -> EmpiricalAcceleration:
        """
        Estimated EA (GCRS), tabulated every output_dt_s seconds, linear in between.
        """

    @property
    def nis(self) -> float:
        """
        Normalized innovation squared per degree of freedom: about 1 when the declared noise matches the data; far from 1 means the noise levels or the EA prior are wrong.
        """

    @property
    def recovered_bias(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """
        Estimated accelerometer bias (ECEF, numpy (3,) [m/s²]), if accelerometers were used.
        """

    @property
    def sigma_a_used(self) -> float:
        """σ_a used [m/s²]"""

    @property
    def tau_used(self) -> float:
        """τ used [s]"""

    @property
    def sigma_rate_eff(self) -> float:
        """Range-rate σ actually used [m/s] (0 = range rate ignored)."""

    @property
    def clock_biases(self) -> list[tuple[str, float]]:
        """
        Estimated clock offset per clock: list of (label, seconds), labels 'shared', 'radar0', 'optics1', ... The scale and sign of these values are not yet calibrated.
        """

    @property
    def sigma_honored(self) -> list[tuple[str, float]]:
        """
        Residual RMS per channel at the final solution: list of (channel, value), e.g. 'radar0_range', 'radar0_rate', 'fix0_pos', in each channel's units. This is the accuracy actually reached, to compare with the declared σ; a value far above it means the channel could not be fitted that well (wrong noise level, an unmodeled bias or clock error, or a model limit). Report only.
        """

    @property
    def clock_bias_s(self) -> float:
        """Clock offset [s]: the 'shared' one, else the first in clock_biases."""

    @property
    def prior_anchor(self) -> float:
        """
        Acceleration scale 2Δ/T² [m/s²] from the zero-EA first fit (Δ: the position residual the dynamics alone could not explain; T: the arc length); 0 when regime_aware_prior is off.
        """

    @property
    def warnings(self) -> list[str]:
        """
        Plain-language warnings (list of str; nothing is changed because of them). One fires when the largest gap between observations is longer than τ: inside such a gap the EA is unobserved and errors grow unseen (measured: a 40 s outage made the error 128× larger while NIS looked normal), so this warning is the only sign. Another fires when the automatic grid chose an edge value: a prompt to check that the defaults suit your kind of trajectory.
        """

    @property
    def ea_sigma(self) -> list[Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]:
        """
        1σ uncertainty of the EA [m/s², GCRS axes] at each point of the returned table, from the smoother's covariance.
        """

    @property
    def iterations_used(self) -> int:
        """Number of iterations performed"""

    @property
    def error_message(self) -> str:
        """
        Error description (empty on success); a filter failure is reported here
        """

    def succeeded(self) -> bool:
        """True if the estimator ran without error"""

    def has_correction(self) -> bool:
        """True if succeeded() and the EA is not zero"""

class EASpecResult:
    """
    Result of estimate_ea(spec): the common result fields, which estimator ran and why, and the check of the declared noise against the fit.
    """

    @property
    def empirical_accel(self) -> EmpiricalAcceleration:
        """The estimated EA (EmpiricalAcceleration, GCRS)"""

    @property
    def method_used(self) -> EAMethod:
        """Which estimator ran (EAMethod)"""

    @property
    def rationale(self) -> str:
        """Why it was chosen (str)"""

    @property
    def nis(self) -> float:
        """
        The smoother's NIS per degree of freedom (about 1 when the declared noise is right); 0 for the B-spline path
        """

    @property
    def sigma_honored(self) -> list[tuple[str, float]]:
        """Residual RMS reached per channel: list of (channel, value)"""

    @property
    def audit_warnings(self) -> list[str]:
        """
        Plain-language findings of the check (empty = nothing found): NIS out of range, residuals far above the declared noise, and so on. Warnings, not errors: the estimate is still returned.
        """

    @property
    def bspline_detail(self) -> EAResult | None:
        """Full EAResult if the B-spline estimator ran, else None"""

    @property
    def smoother_detail(self) -> EASmootherResult | None:
        """Full EASmootherResult if the smoother ran, else None"""

    @property
    def error_message(self) -> str:
        """Error description (empty on success), e.g. a missing noise_sigma"""

    def succeeded(self) -> bool:
        """True if the estimator ran without error"""

    def has_correction(self) -> bool:
        """True if succeeded() and the EA is not zero"""

class EOFFile:
    """A parsed ESA EOF orbit file: its state vectors and metadata."""

    def __init__(self) -> None: ...

    @property
    def filename(self) -> str:
        """Name of the file read"""

    @filename.setter
    def filename(self, arg: str, /) -> None: ...

    @property
    def mission(self) -> str:
        """e.g. 'Sentinel-1A'"""

    @mission.setter
    def mission(self, arg: str, /) -> None: ...

    @property
    def reference_frame(self) -> str:
        """Frame label of the file, e.g. 'EARTH_FIXED' (ECEF)"""

    @reference_frame.setter
    def reference_frame(self, arg: str, /) -> None: ...

    @property
    def validity_start(self) -> Epoch:
        """Start of the file's validity period (Epoch)"""

    @validity_start.setter
    def validity_start(self, arg: Epoch, /) -> None: ...

    @property
    def skipped_records(self) -> int:
        """
        Number of state vectors that could not be read (reported once as a DataSkippedWarning)
        """

    @property
    def validity_stop(self) -> Epoch:
        """End of the file's validity period (Epoch)"""

    @validity_stop.setter
    def validity_stop(self, arg: Epoch, /) -> None: ...

    @property
    def state_vectors(self) -> list[EOFStateVector]:
        """List of EOFStateVector, in time order"""

    @state_vectors.setter
    def state_vectors(self, arg: Sequence[EOFStateVector], /) -> None: ...

    def find_nearest(self, target: Epoch) -> EOFStateVector | None:
        """
        The state vector nearest to the Epoch `target`, or None if none is within 60 s.
        """

    def get_range(self, start: Epoch, end: Epoch) -> list[EOFStateVector]:
        """The state vectors with start <= epoch <= end."""

    def interpolate_position(self, target: Epoch, order: int = 9) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')] | None:
        """
        Earth-fixed position [m] at the Epoch `target`, by Lagrange-polynomial interpolation through `order` state vectors; None if there are too few, or outside the data (no extrapolation).
        """

class EOFParser:
    """Reader for ESA EOF orbit files."""

    @staticmethod
    def parse(filename: str) -> EOFFile | None:
        """
        Read an .EOF file by path; returns an EOFFile, or None if the file cannot be opened or holds no readable state vector. Raises BallisticError if the validity period is missing or malformed.
        """

    @staticmethod
    def parse_content(content: str, filename: str = '') -> EOFFile | None:
        """
        Read EOF XML content from a string (filename is used in messages only).
        """

class EOFStateVector:
    """
    One orbit state from an ESA EOF file (Earth-fixed coordinates, UTC time tags).
    """

    def __init__(self) -> None: ...

    @property
    def epoch(self) -> Epoch:
        """Time of the state (Epoch)"""

    @epoch.setter
    def epoch(self, arg: Epoch, /) -> None: ...

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Earth-fixed (ECEF) position [m]"""

    @position.setter
    def position(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Velocity relative to the rotating Earth (ECEF) [m/s]"""

    @velocity.setter
    def velocity(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def orbit_number(self) -> int:
        """Absolute orbit number"""

    @orbit_number.setter
    def orbit_number(self, arg: int, /) -> None: ...

    @property
    def quality(self) -> str:
        """Quality flag (e.g. NOMINAL)"""

    @quality.setter
    def quality(self, arg: str, /) -> None: ...

class EmpiricalAccelEntry:
    """
    One time point of a tabulated empirical acceleration (a small correction acceleration fitted to observations).
    """

    @overload
    def __init__(self) -> None:
        """Entry with key=0 and a zero acceleration"""

    @overload
    def __init__(self, key: float, accel: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> None:
        """
        Entry at `key` seconds after the reference epoch with acceleration `accel`, numpy (3,) [m/s^2]
        """

    @property
    def key(self) -> float:
        """
        Time of the entry: seconds after the reference epoch (the trajectory's start, or TrajectoryConfig.ea_reference_epoch).
        """

    @key.setter
    def key(self, arg: float, /) -> None: ...

    @property
    def accel(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Acceleration vector, numpy (3,) [m/s^2]"""

    @accel.setter
    def accel(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

class EmpiricalAccelFrame(enum.Enum):
    """
    Coordinate frame in which empirical acceleration (EA) vectors are stored.
    """

    GCRS = 0
    """
    Geocentric Celestial Reference System: inertial, non-rotating (default)
    """

    ITRF = 1
    """
    International Terrestrial Reference Frame: Earth-fixed, rotates with the Earth
    """

class EmpiricalAcceleration:
    """
    Empirical acceleration (EA): a small correction acceleration, as a function
    of time, added to the modeled forces (usually fitted to observations by
    estimate_ea_bspline or estimate_ea_smoother). Set it as
    TrajectoryConfig.empirical_accel. Times are seconds after the reference
    epoch (the trajectory's start, or TrajectoryConfig.ea_reference_epoch).

    Built with the factory methods below; it cannot be changed afterwards.

    Example:
        accel = ballistic.EmpiricalAcceleration.from_entries(
            [ballistic.EmpiricalAccelEntry(0.0, [0, 0, 0]),
             ballistic.EmpiricalAccelEntry(100.0, [1e-5, 2e-6, 0]),
             ballistic.EmpiricalAccelEntry(300.0, [0, 0, 0])],
            frame=ballistic.EmpiricalAccelFrame.GCRS,
        )
        print(accel.accel_at(150.0))   # interpolate at t=150 s
    """

    def __init__(self) -> None:
        """Create the zero EA (no correction)."""

    @staticmethod
    def zero() -> EmpiricalAcceleration:
        """The zero EA (no correction)."""

    @staticmethod
    def from_entries(entries: Sequence[EmpiricalAccelEntry], frame: EmpiricalAccelFrame = EmpiricalAccelFrame.GCRS, interp: EAInterp = EAInterp.Cubic) -> EmpiricalAcceleration:
        """
        Build from a list of EmpiricalAccelEntry (time [s], acceleration [m/s^2]), sorted internally; `frame` is the frame of the vectors, `interp` how values are interpolated between entries (EAInterp.Cubic or Linear).
        """

    @staticmethod
    def from_entries_hermite(entries: Sequence[EmpiricalAccelEntry], derivatives: Sequence[Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]], frame: EmpiricalAccelFrame = EmpiricalAccelFrame.GCRS) -> EmpiricalAcceleration:
        """
        Build from a list of EmpiricalAccelEntry plus the time derivative of the acceleration at each entry [m/s^3] (as many as the entries, else ValueError), interpolated by cubic Hermite polynomials.
        """

    @staticmethod
    def from_bspline(coeffs: BSplineCoefficients, represents: EARepresents, frame: EmpiricalAccelFrame = EmpiricalAccelFrame.GCRS) -> EmpiricalAcceleration:
        """
        Build from BSplineCoefficients, used directly (no conversion to a table). `represents` (EARepresents) says whether the spline is the acceleration itself or a velocity / position whose derivative gives it.
        """

    def accel_at(self, t_seconds: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """
        Acceleration at t_seconds after the reference epoch, numpy (3,) [m/s^2], in the frame the EA is stored in.
        """

    def evaluate_at(self, times_s: Annotated[ArrayLike, dict(dtype='float64', shape=(None), writable=False)]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, 3), order='C')]:
        """
        accel_at for a numpy (N,) array of times [s after the reference epoch]: returns a numpy (N, 3) array [m/s^2] in the stored frame.
        """

    def is_zero(self) -> bool:
        """True if this is the zero EA (no correction)."""

    def frame(self) -> EmpiricalAccelFrame:
        """Frame the acceleration vectors are stored in (EmpiricalAccelFrame)."""

    def keys(self) -> list[float]:
        """
        Time points [s]: the entry times of a table; the start, the distinct interior knot times and the end of a B-spline. Empty for the zero EA.
        """

    def domain(self) -> tuple[float, float]:
        """
        Time span (t_start, t_end) [s] over which the EA is defined; (0.0, 0.0) for the zero EA.
        """

    def as_bspline(self) -> tuple[list[float], int] | None:
        """
        Tuple (knots, degree) if the EA was built with from_bspline(), else None. Tells whether an EA returned by estimate_ea_bspline is a B-spline and of what degree (Gauss-Newton refinement by finite differences converts it to a table with cubic interpolation for degree > 3, linear otherwise; the STM path always uses linear).
        """

class Epoch:
    """
    Epoch - an instant in time, readable in any of the time scales used for
    orbits:

    - UTC: Coordinated Universal Time (civil time, with leap seconds)
    - TT: Terrestrial Time (TT = TAI + 32.184 s)
    - TAI: International Atomic Time
    - TDB: Barycentric Dynamical Time (planetary ephemerides)
    - UT1: Universal Time 1 (follows the Earth's rotation)
    - GPS: GPS Time (GPS = TAI - 19 s)

    Internally the instant is stored in TT as a whole Julian day number plus
    the seconds into that day, so labels and differences between epochs are
    exact to about 1e-11 s. A Julian Date (JD) is a continuous count of days
    used in astronomy; the Modified Julian Date (MJD) is JD - 2400000.5.

    Examples:
        # Create from UTC calendar
        epoch = Epoch.from_utc(2024, 1, 15, 12, 0, 0)

        # Create from Julian Date
        epoch = Epoch.from_jd(2460325.0, TimeScale.TT)

        # Create from Unix timestamp
        epoch = Epoch.from_unix(1705320000.0)

        # Arithmetic
        epoch2 = epoch + 3600.0  # Add 1 hour (3600 seconds)
        dt = epoch2 - epoch      # Get difference in seconds

        # Access different time scales
        utc_jd = epoch.utc_jd
        tt_jd = epoch.tt_jd
        unix_ts = epoch.unix_timestamp
    """

    @overload
    def __init__(self) -> None:
        """The J2000.0 epoch (2000-01-01 12:00:00 TT)"""

    @overload
    def __init__(self, jd: float, scale: TimeScale) -> None:
        """From a Julian Date `jd` in the time scale `scale` (both required)."""

    @overload
    def __init__(self, jd: float) -> None:
        """Removed: raises RemovedApiError. Pass the time scale."""

    @staticmethod
    def from_utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0) -> Epoch:
        """
        Create an Epoch from a UTC calendar date and time.

        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            day: Day of month (1-31)
            hour: Hour (0-23, default 0)
            minute: Minute (0-59, default 0)
            second: Second (0-60, default 0.0, may be fractional; 60 only during a
                leap second)

        Returns:
            Epoch: the given UTC instant

        Raises:
            ballistic.errors.DataCoverageError: a date before 1961, when UTC is not
                defined; use from_tt or from_tai.

        Example:
            epoch = Epoch.from_utc(2024, 1, 15, 12, 30, 45.5)
        """

    @overload
    @staticmethod
    def from_jd(jd: float, scale: TimeScale) -> Epoch:
        """
        From a Julian Date `jd` (days) in the time scale `scale` (both required).
        """

    @overload
    @staticmethod
    def from_jd(jd: float) -> Epoch:
        """Removed: raises RemovedApiError. Pass the time scale."""

    @overload
    @staticmethod
    def from_mjd(mjd: float, scale: TimeScale) -> Epoch:
        """
        From a Modified Julian Date `mjd` (days; MJD = JD - 2400000.5) in the time scale `scale` (both required).
        """

    @overload
    @staticmethod
    def from_mjd(mjd: float) -> Epoch:
        """Removed: raises RemovedApiError. Pass the time scale."""

    @staticmethod
    def from_tt(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0) -> Epoch:
        """Calendar date and time in TT (Terrestrial Time)."""

    @staticmethod
    def from_tai(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0) -> Epoch:
        """Calendar date and time in TAI (International Atomic Time)."""

    @staticmethod
    def from_gps(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0) -> Epoch:
        """
        Calendar date and time in GPS time, as satellite-navigation (GNSS) products and receivers stamp it (e.g. SP3 files marked '%c GPS'). Reading GPS time as UTC is 18 s off (about 130 km along a low Earth orbit).
        """

    @staticmethod
    def from_tdb(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0) -> Epoch:
        """Calendar date and time in TDB (Barycentric Dynamical Time)."""

    @staticmethod
    def from_gps_week(week: int, seconds_of_week: float) -> Epoch:
        """
        From a GPS week number (counted continuously from the start of GPS time, without the 1024-week rollover) and the seconds into that week.
        """

    def to_gps_week(self) -> tuple[int, float]:
        """Tuple (GPS week number, seconds into the week)."""

    @staticmethod
    def from_datetime(dt: object) -> Epoch:
        """
        From a timezone-aware datetime.datetime (converted to UTC). Raises ConfigError for a naive datetime (no tzinfo), which is ambiguous.
        """

    @staticmethod
    def from_unix(unix_time: float) -> Epoch:
        """
        From a Unix timestamp: seconds since 1970-01-01 00:00:00 UTC, not counting leap seconds (as time.time() returns)
        """

    @staticmethod
    def j2000() -> Epoch:
        """The J2000.0 reference epoch (2000-01-01 12:00:00 TT)"""

    @property
    def utc_jd(self) -> float:
        """
        Julian Date in UTC (float, days; about 40 microsecond resolution - use Epoch differences, not JD differences, for time intervals)
        """

    @staticmethod
    def from_tt_split(tt_day: int, tt_seconds: float) -> Epoch:
        """
        Exact construction from the internal representation: the TT Julian day number (int; Julian days start at noon) and the seconds into it.
        """

    @property
    def tt_day(self) -> int:
        """TT Julian day number (int; part of the exact internal representation)"""

    @property
    def tt_seconds(self) -> float:
        """Seconds into tt_day (part of the exact internal representation)"""

    @property
    def tt_jd(self) -> float:
        """TT (Terrestrial Time) Julian Date"""

    @property
    def tai_jd(self) -> float:
        """TAI (International Atomic Time) Julian Date"""

    @property
    def tdb_jd(self) -> float:
        """TDB (Barycentric Dynamical Time) Julian Date"""

    @property
    def ut1_jd(self) -> float:
        """
        UT1 Julian Date (needs Earth orientation data, EOP; raises DataCoverageError without it unless set_eop_policy('assume_zero'))
        """

    @property
    def gps_jd(self) -> float:
        """GPS Time Julian Date"""

    @property
    def unix_timestamp(self) -> float:
        """
        Unix timestamp (seconds since 1970-01-01 00:00:00 UTC, leap seconds not counted)
        """

    @property
    def seconds_since_j2000(self) -> float:
        """Seconds since J2000.0, in TT"""

    @property
    def julian_centuries_tt(self) -> float:
        """Julian centuries (36525 days) since J2000.0, in TT"""

    @property
    def day_of_year(self) -> float:
        """
        Fractional day of the year in UTC, counted from 0 (1 January 00:00 = 0.0, noon = 0.5), as atmosphere models take it
        """

    @property
    def year(self) -> int:
        """Calendar year (int, UTC)"""

    def jd(self, scale: TimeScale) -> float:
        """Julian Date (float, days) in the time scale `scale`"""

    def mjd(self, scale: TimeScale) -> float:
        """Modified Julian Date (float, days) in the time scale `scale`"""

    def to_calendar(self, scale: TimeScale = TimeScale.UTC) -> tuple[int, int, int, int, int, float]:
        """
        Calendar date and time in a chosen time scale.

        Args:
            scale: Time scale for output (default UTC)

        Returns:
            tuple: (year, month, day, hour, minute, second)
        """

    def set_dut1(self, dut1_seconds: float) -> None:
        """
        Set a fallback UT1-UTC [s] for this epoch, used by the conversions to UT1 only when no Earth orientation table is loaded (which requires set_eop_policy('assume_zero')). A loaded table always takes precedence.
        """

    @property
    def dut1(self) -> float:
        """The fallback UT1-UTC [s] set by set_dut1 (0 by default)"""

    def __add__(self, seconds: float) -> Epoch:
        """A new Epoch `seconds` later"""

    def __radd__(self, seconds: float) -> Epoch: ...

    @overload
    def __sub__(self, seconds: float) -> Epoch:
        """A new Epoch `seconds` earlier"""

    @overload
    def __sub__(self, other: Epoch) -> float:
        """Time from `other` to this epoch, in seconds (float)"""

    def __iadd__(self, seconds: float) -> Epoch: ...

    def __isub__(self, seconds: float) -> Epoch: ...

    def __lt__(self, arg: Epoch, /) -> bool: ...

    def __le__(self, arg: Epoch, /) -> bool: ...

    def __gt__(self, arg: Epoch, /) -> bool: ...

    def __ge__(self, arg: Epoch, /) -> bool: ...

    def __eq__(self, arg: Epoch, /) -> bool: ...

    def __ne__(self, arg: Epoch, /) -> bool: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

class FlowRegimeCd:
    """
    Drag coefficient that changes with the flow regime, for reentry.

    The Knudsen number Kn (the air's mean free path divided by the body's size)
    decides the regime. In thin air (Kn above 10, the free-molecular regime)
    Cd comes from Sentman's formulas, with the energy accommodation coefficient
    (0 to 1: how fully molecules hitting the surface adopt the wall's
    temperature). In dense air (Kn below 1e-3, the continuum regime) Cd comes
    from Henderson (1976) for a sphere, and from modified Newtonian theory or a
    MachTable for a plate or cylinder. Between the two, a smooth bridge is used.
    In orbit the air is free-molecular, where Cd varies only slowly (with air
    temperature and composition), so a ConstantCd is close there.

    Args:
        shape: AeroShape
        accommodation: energy accommodation coefficient, 0 to 1 (default 1)
        wall_temperature_k: surface temperature [K] (default 300)
        continuum: optional MachTable for the dense-air Cd (default: the
            built-in model for the shape)
    """

    def __init__(self, shape: AeroShape, accommodation: float = 1.0, wall_temperature_k: float = 300.0, continuum: MachTable | None = None) -> None: ...

    @property
    def shape(self) -> AeroShape:
        """Body shape (AeroShape)"""

    @shape.setter
    def shape(self, arg: AeroShape, /) -> None: ...

    @property
    def accommodation(self) -> float:
        """Energy accommodation coefficient, 0 to 1"""

    @accommodation.setter
    def accommodation(self, arg: float, /) -> None: ...

    @property
    def wall_temperature_k(self) -> float:
        """Surface temperature [K]"""

    @wall_temperature_k.setter
    def wall_temperature_k(self, arg: float, /) -> None: ...

class ForceBreakdown:
    """
    Acceleration from each force separately, each numpy (3,) [m/s^2] (filled in when TrajectoryConfig.report_forces is True).
    """

    def __init__(self) -> None: ...

    @property
    def gravity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Earth gravity"""

    @property
    def third_body(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Gravity of the Sun, Moon and Jupiter"""

    @property
    def drag(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Atmospheric drag"""

    @property
    def srp(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Solar radiation pressure"""

    @property
    def albedo(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Earth albedo radiation"""

    @property
    def ir(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Earth infrared radiation"""

    @property
    def relativistic(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """General-relativity corrections"""

    @property
    def solid_tides(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Solid-Earth tide corrections"""

    @property
    def ocean_tides(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Ocean tide corrections (FES2004 model, once loaded)"""

    @property
    def empirical(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Empirical (fitted correction) acceleration"""

    @property
    def total(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Sum of all components"""

    def radiation(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Sum of the radiation forces: SRP + albedo + infrared"""

class FrameError(BallisticError):
    pass

class GNRefine:
    """
    Settings for Gauss-Newton (GN) refinement: after the B-spline fit, the EA is adjusted so that the orbit propagated with it matches the observed positions as closely as possible. Assign to EABSplineConfig.gn (on by default).
    """

    def __init__(self) -> None: ...

    @property
    def reg_lambda(self) -> float:
        """
        Weight of the smoothness penalty during refinement, the same kind of penalty as in the B-spline fit, so both minimize the same objective. Default 1.0. 0.0 turns it off: on noisy data the propagated orbit still matches, but the EA itself becomes badly wrong point by point.
        """

    @reg_lambda.setter
    def reg_lambda(self, arg: float, /) -> None: ...

    @property
    def reg_deriv_order(self) -> int | None:
        """
        Which time derivative of the EA the smoothness penalty acts on (None = automatic: 3 for positions, 1 for velocities); same meaning as EABSplineConfig.reg_deriv_order.
        """

    @reg_deriv_order.setter
    def reg_deriv_order(self, arg: int | None) -> None: ...

    @property
    def max_iterations(self) -> int:
        """Maximum GN iterations (default: 30)"""

    @max_iterations.setter
    def max_iterations(self, arg: int, /) -> None: ...

    @property
    def sample_stride(self) -> int:
        """Fit every N-th observation (default: 20)."""

    @sample_stride.setter
    def sample_stride(self, arg: int, /) -> None: ...

    @property
    def ea_subsample(self) -> int:
        """Adjust every N-th EA time point (default: 1 = all of them)"""

    @ea_subsample.setter
    def ea_subsample(self, arg: int, /) -> None: ...

    @property
    def fd_delta(self) -> float:
        """
        Finite-difference perturbation of an EA value [m/s^2] (default: 1e-2), used when use_stm is False
        """

    @fd_delta.setter
    def fd_delta(self, arg: float, /) -> None: ...

    @property
    def rk45_max_step(self) -> object:
        """
        Largest RK45 integrator step during GN propagations: None = no override, fully adaptive (default, recommended); 'match_knots' = the estimator's knot spacing (measured to make results worse); a float = this step [s].
        """

    @rk45_max_step.setter
    def rk45_max_step(self, arg: object | None) -> None: ...

    @property
    def use_stm(self) -> bool:
        """
        How GN gets its derivatives. True (default): the state transition matrix (STM), i.e. the sensitivity of the orbit to each EA value, integrated alongside the orbit in one propagation; about 2× faster. False: finite differences, one extra propagation per EA value. The STM path interpolates the EA linearly between its time points, which cost about 30% accuracy on position data with the default degree-5 spline; set False for a one-off run where accuracy matters most.
        """

    @use_stm.setter
    def use_stm(self, arg: bool, /) -> None: ...

    @property
    def stm_dt_s(self) -> float:
        """Step [s] for integrating the sensitivity equations (default: 0.5)"""

    @stm_dt_s.setter
    def stm_dt_s(self, arg: float, /) -> None: ...

    @property
    def stm_fd_dx(self) -> float:
        """
        Position perturbation [m] for the derivative of acceleration with respect to position (default: 1.0)
        """

    @stm_fd_dx.setter
    def stm_fd_dx(self, arg: float, /) -> None: ...

    @property
    def stm_fd_dv(self) -> float:
        """
        Velocity perturbation [m/s] for the derivative of acceleration with respect to velocity (default: 1e-3)
        """

    @stm_fd_dv.setter
    def stm_fd_dv(self, arg: float, /) -> None: ...

    @property
    def stm_verify(self) -> bool:
        """
        First iteration only: also compute finite-difference derivatives and print the difference (default: False)
        """

    @stm_verify.setter
    def stm_verify(self, arg: bool, /) -> None: ...

    @property
    def stm_verbose(self) -> bool:
        """Print the STM computation's internal trace (default: False)"""

    @stm_verbose.setter
    def stm_verbose(self, arg: bool, /) -> None: ...

class GNStatus(enum.Enum):
    """What happened to the optional Gauss-Newton (GN) refinement step."""

    Disabled = 0
    """Not requested (config.gn is None)"""

    Ran = 1
    """GN refinement ran"""

    SkippedAccelerationCircular = 2
    """
    Skipped for acceleration data: there are no observed positions to fit, so GN would only fit the orbit to itself
    """

class GeodeticCoord:
    """
    Geodetic coordinates on the WGS84 ellipsoid: latitude and longitude in radians, altitude in meters above the ellipsoid.
    """

    def __init__(self, latitude: float, longitude: float, altitude: float) -> None:
        """From latitude and longitude [rad] and altitude [m]"""

    @property
    def latitude(self) -> float:
        """[rad], positive north"""

    @latitude.setter
    def latitude(self, arg: float, /) -> None: ...

    @property
    def longitude(self) -> float:
        """[rad], positive east"""

    @longitude.setter
    def longitude(self, arg: float, /) -> None: ...

    @property
    def altitude(self) -> float:
        """[m] above the WGS84 ellipsoid"""

    @altitude.setter
    def altitude(self, arg: float, /) -> None: ...

    @property
    def latitude_deg(self) -> float:
        """Latitude [deg]"""

    @property
    def longitude_deg(self) -> float:
        """Longitude [deg]"""

    def __repr__(self) -> str: ...

class HistoryEntry:
    """One stored state in a Trajectory's history (Earth-fixed, ECEF)."""

    def __init__(self) -> None: ...

    @property
    def epoch(self) -> Epoch:
        """Time of the state (Epoch)"""

    @property
    def t_rel_s(self) -> float:
        """
        Exact seconds elapsed since the trajectory's reference time (its start or last reset); the interpolator uses this as its time axis
        """

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Position, ECEF, numpy (3,) [m]"""

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Velocity relative to the rotating Earth, numpy (3,) [m/s]"""

    @property
    def acceleration(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Acceleration, ECEF axes, numpy (3,) [m/s^2]"""

    @property
    def step_size_used(self) -> float:
        """Step size that led here [s]"""

class ITRFRealization(enum.Enum):
    """
    Versions ("realizations") of the ITRF, the International Terrestrial
    Reference Frame (the Earth-fixed frame). Each version places the frame
    slightly differently (millimeters to centimeters, drifting over the years).

    Orbit products state which version they use; reading a product in the
    wrong version costs accuracy at the sub-meter level.

    Guidelines:
    - ESA Sentinel (before March 4, 2025): ITRF2014
    - ESA Sentinel (after March 4, 2025): ITRF2020
    - GFZ CHAMP (2000-2010): ITRF2005
    - IGS GPS (2017-2024): ITRF2014
    - IGS GPS (2024+): ITRF2020
    """

    ITRF2000 = 0

    ITRF2005 = 1

    ITRF2008 = 2

    ITRF2014 = 3
    """Use for Sentinel data before March 4, 2025"""

    ITRF2020 = 4
    """Default, use for Sentinel data after March 4, 2025"""

class InitConfig:
    r"""
    Paths to external data files for initialize_with_config.

    The Earth gravity model and the planetary ephemeris (JPL DE440) are built
    in. The Earth orientation parameters (EOP) are not: they are needed for
    sub-meter accuracy and are found automatically in the data directory below
    once downloaded (ballistic.load() or ballistic.Loader().load_eop()). Set a
    path here only to use a specific file. Unset paths are None.

    Default data directory (platform-specific):
        Windows: %LOCALAPPDATA%\ballistic  (C:\Users\<user>\AppData\Local\ballistic)
        Linux:   ~/.local/share/ballistic
        macOS:   ~/Library/Application Support/ballistic
        The environment variable BALLISTIC_DATA_DIR, when set, names the
        directory instead (every platform).

    Example:
        config = ballistic.InitConfig()
        config.eop_path = "finals2000A.data"
        config.jpl_ephemeris_path = "de421.bsp"
        result = ballistic.initialize_with_config(config)
    """

    def __init__(self) -> None: ...

    @property
    def egm2008_path(self) -> str | None:
        """
        Path to a full EGM2008 gravity coefficient file (str), needed only for gravity degrees 361-2190; the embedded model covers degrees 2-360
        """

    @egm2008_path.setter
    def egm2008_path(self, arg: str, /) -> None: ...

    @property
    def eop_path(self) -> str | None:
        """
        Path to the IERS finals2000A.data (or .all) Earth orientation file (str)
        """

    @eop_path.setter
    def eop_path(self, arg: str, /) -> None: ...

    @property
    def jpl_ephemeris_path(self) -> str | None:
        """
        Path to a JPL planetary ephemeris file, .bsp (SPICE) or .bin (str); replaces the embedded DE440
        """

    @jpl_ephemeris_path.setter
    def jpl_ephemeris_path(self, arg: str, /) -> None: ...

    @property
    def leap_seconds_path(self) -> str | None:
        """Path to the IERS leap-seconds.list file (str)"""

    @leap_seconds_path.setter
    def leap_seconds_path(self, arg: str, /) -> None: ...

    @property
    def nrl_models_path(self) -> str | None:
        """
        Path to the NRLMSIS 2.1 / HWM14 plugin (str), needed only for model-based drag; None searches BALLISTIC_NRL_MODELS and the data directory (where ballistic.install_nrl_models installs it)
        """

    @nrl_models_path.setter
    def nrl_models_path(self, arg: str, /) -> None: ...

class InitResult:
    """
    What initialize() loaded and any problems found. True in a boolean context when initialization succeeded.
    """

    def __init__(self) -> None: ...

    @property
    def success(self) -> bool:
        """True if initialization succeeded"""

    @property
    def egm2008_loaded(self) -> bool:
        """True if an external EGM2008 gravity file was loaded"""

    @property
    def eop_loaded(self) -> bool:
        """True if Earth orientation data (EOP) was loaded"""

    @property
    def eop_auto_discovered(self) -> bool:
        """
        True if EOP was loaded via data-directory auto-discovery (not explicit eop_path)
        """

    @property
    def jpl_ephemeris_loaded(self) -> bool:
        """True if an external JPL ephemeris file was loaded"""

    @property
    def embedded_ephemeris_loaded(self) -> bool:
        """True if the built-in DE440 planetary ephemeris was loaded"""

    @property
    def leap_seconds_loaded(self) -> bool:
        """True if leap seconds were loaded from an external file"""

    @property
    def leap_seconds_auto_discovered(self) -> bool:
        """True if leap-seconds.list was loaded via data-directory auto-discovery"""

    @property
    def gravity_degree_loaded(self) -> int:
        """Highest gravity-model degree available"""

    @property
    def jpl_de_version(self) -> str:
        """Which JPL ephemeris is in use (e.g., 'DE421' or 'DE440 (embedded)')"""

    @property
    def eop_source_path(self) -> str:
        """Path of the loaded EOP file (empty if none)"""

    @property
    def nrl_models_loaded(self) -> bool:
        """
        True if the NRLMSIS 2.1 / HWM14 plugin is loaded (model-based drag available)
        """

    @property
    def nrl_models_path(self) -> str:
        """Path of the loaded NRL models plugin (empty if none)"""

    @property
    def error_message(self) -> str:
        """Error description if success is False"""

    @property
    def eop_table_end_mjd(self) -> float:
        """
        Last day of the loaded EOP table as a Modified Julian Date (MJD; 0 = no EOP loaded)
        """

    @property
    def eop_table_stale(self) -> bool:
        """
        True if EOP is loaded but the table's end (including its ~1 year of predictions) is already in the past. Past the end the last value is held for up to 30 days with a warning (meters of error at low orbit), then the frame transforms raise DataCoverageError. Refresh with ballistic.Loader().load_eop().
        """

    @property
    def leap_seconds_expiration_jd(self) -> float:
        """
        Expiry date of the leap-second table as a Julian Date; after it a newly announced leap second could be missed (0 = unknown)
        """

    @property
    def leap_seconds_expired(self) -> bool:
        """
        True if the leap-second table's expiry date is in the past. Refresh with ballistic.Loader().timescale().
        """

    @property
    def warnings(self) -> list[str]:
        """
        Readable warnings about out-of-date data (list of str; empty = up to date)
        """

    def __bool__(self) -> bool:
        """Same as success, so `if ballistic.initialize():` works."""

    def __repr__(self) -> str: ...

class InputPlausibilityError(BallisticError):
    pass

class IntegratorType(enum.Enum):
    """Numerical integrator that steps the equations of motion."""

    RK45 = 0
    """
    Dormand-Prince Runge-Kutta 4(5) with adaptive step size (default); most accurate for predictions of hours
    """

    YOSHIDA6 = 1
    """
    Yoshida 6th-order symplectic integrator: fixed step, no long-term energy drift; recommended for runs of 6 hours or more
    """

    YOSHIDA8 = 2
    """Yoshida 8th-order symplectic integrator"""

class InterpolatedState:
    """
    State returned by a TrajectoryInterpolator query, in the interpolator's output frame.
    """

    def __init__(self) -> None: ...

    @property
    def epoch(self) -> Epoch:
        """Query time (Epoch)"""

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Position, numpy (3,) [m]"""

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Velocity, numpy (3,) [m/s]"""

    @property
    def acceleration(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Acceleration, numpy (3,) [m/s^2]"""

    @property
    def was_propagated(self) -> bool:
        """True if the history had to be extended to answer this query"""

    @property
    def segment_index(self) -> int:
        """Index of the history interval the time fell in"""

class InterpolationFrame(enum.Enum):
    """Coordinate frame of interpolated states."""

    ECEF = 0
    """Earth-centered, Earth-fixed (ITRF; default)"""

    GCRS = 1
    """
    Inertial Geocentric Celestial Reference System (IAU 2006/2000A transform)
    """

class InterpolatorConfig:
    """Settings for a TrajectoryInterpolator."""

    def __init__(self) -> None: ...

    @property
    def output_frame(self) -> InterpolationFrame:
        """
        Coordinate frame of returned states (InterpolationFrame; default: ECEF)
        """

    @output_frame.setter
    def output_frame(self, arg: InterpolationFrame, /) -> None: ...

    @property
    def itrf(self) -> ITRFRealization | None:
        """
        ITRF version used when converting to the GCRS (Optional; None = the wrapped Trajectory's)
        """

    @itrf.setter
    def itrf(self, arg: ITRFRealization | None) -> None: ...

    @property
    def cache_coefficients(self) -> bool:
        """Cache interpolation polynomial coefficients (default: True)"""

    @cache_coefficients.setter
    def cache_coefficients(self, arg: bool, /) -> None: ...

    @property
    def allow_propagation(self) -> bool:
        """
        Propagate the trajectory to answer queries outside its history (default: True)
        """

    @allow_propagation.setter
    def allow_propagation(self, arg: bool, /) -> None: ...

    @property
    def max_propagation_time(self) -> float:
        """
        Furthest beyond the history [s] that a query may propagate (default: 3600)
        """

    @max_propagation_time.setter
    def max_propagation_time(self, arg: float, /) -> None: ...

class IterationDiag:
    """
    Report from building a drag environment along a trajectory: how many passes
    were run, whether the trajectory stopped changing between passes, and by
    how much it changed in the last pass. from_models_along makes a single pass
    (iterations = 1); from_models_along_iterated repeats until the trajectory
    settles.
    """

    def __init__(self) -> None: ...

    @property
    def iterations(self) -> int:
        """Number of passes run"""

    @property
    def converged(self) -> bool:
        """True if the trajectory change fell below the tolerance"""

    @property
    def final_traj_delta_m(self) -> float:
        """Largest change of the trajectory in the last pass [m]"""

class LeapSecondInfo:
    """
    Information about the leap-second table (the list of one-second steps that keep UTC within a second of Earth-rotation time).
    """

    def __init__(self) -> None: ...

    @property
    def current_tai_utc(self) -> int:
        """Latest offset between atomic time (TAI) and UTC [s]"""

    @property
    def entry_count(self) -> int:
        """Number of leap second entries in the table"""

    @property
    def loaded_from_file(self) -> bool:
        """
        True if loaded from an external file, False if using the built-in table
        """

    @property
    def expiration_jd(self) -> float:
        """Expiry date as a Julian Date (0 if unknown)"""

class MachTable:
    """
    Drag coefficient as a function of Mach number in dense air (the continuum regime), e.g. wind-tunnel data for the body. Interpolated linearly between points. MachTable(mach, cd): two equal-length lists, at least two points, Mach strictly increasing, Cd > 0, else ConfigError. Evaluating outside the table raises DataCoverageError.
    """

    def __init__(self, mach: Sequence[float], cd: Sequence[float]) -> None: ...

    @property
    def mach(self) -> list[float]:
        """Mach numbers of the table points"""

    @property
    def cd(self) -> list[float]:
        """Drag coefficients at those points"""

    def __call__(self, mach: float) -> float:
        """Drag coefficient at Mach number `mach`"""

class MultiSensorScenario:
    """
    Everything a multi-sensor EA fit needs besides settings: the start time and
    state, the arc length and the sensors with their samples. The start state
    must be known exactly (it is not estimated from the observations).

    Two construction styles supported:
      s = MultiSensorScenario(); s.epoch0 = ...; s.add_radar(...)
      s = MultiSensorScenario(epoch0=..., pos0_ecef=..., vel0_ecef=...,
                              arc_duration_s=..., radars=[...], fixes=[...])
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, epoch0: Epoch, pos0_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], vel0_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], arc_duration_s: float, radars: Sequence[RadarObs] = [], optics: Sequence[OpticsObs] = [], fixes: Sequence[PositionFixObs] = [], accels: Sequence[AccelObs] = []) -> None:
        """Build from keywords; the sensor lists default to empty."""

    @property
    def epoch0(self) -> Epoch:
        """Start time (Epoch)."""

    @epoch0.setter
    def epoch0(self, arg: Epoch, /) -> None: ...

    @property
    def pos0_ecef(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Start position, Earth-fixed (ECEF), numpy (3,) [m]; known exactly."""

    @pos0_ecef.setter
    def pos0_ecef(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def vel0_ecef(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """
        Start velocity relative to the rotating Earth, numpy (3,) [m/s]; known exactly.
        """

    @vel0_ecef.setter
    def vel0_ecef(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def arc_duration_s(self) -> float:
        """Length of the arc to fit [s]."""

    @arc_duration_s.setter
    def arc_duration_s(self, arg: float, /) -> None: ...

    @property
    def radars(self) -> list[RadarObs]:
        """List of RadarObs"""

    @radars.setter
    def radars(self, arg: Sequence[RadarObs], /) -> None: ...

    @property
    def optics(self) -> list[OpticsObs]:
        """List of OpticsObs"""

    @optics.setter
    def optics(self, arg: Sequence[OpticsObs], /) -> None: ...

    @property
    def fixes(self) -> list[PositionFixObs]:
        """List of PositionFixObs"""

    @fixes.setter
    def fixes(self, arg: Sequence[PositionFixObs], /) -> None: ...

    @property
    def accels(self) -> list[AccelObs]:
        """List of AccelObs"""

    @accels.setter
    def accels(self, arg: Sequence[AccelObs], /) -> None: ...

    def add_radar(self, arg: RadarObs, /) -> MultiSensorScenario:
        """Append a RadarObs; returns the scenario, so calls can be chained."""

    def add_optics(self, arg: OpticsObs, /) -> MultiSensorScenario:
        """Append an OpticsObs; returns the scenario."""

    def add_position_fix(self, arg: PositionFixObs, /) -> MultiSensorScenario:
        """Append a PositionFixObs; returns the scenario."""

    def add_accelerometer(self, arg: AccelObs, /) -> MultiSensorScenario:
        """Append an AccelObs; returns the scenario."""

class NonConservativeStrategy(enum.Enum):
    """
    How a symplectic integrator handles forces that do not conserve energy (drag, radiation pressure).
    """

    NONE = 0
    """Ignore those forces (pure symplectic integration)."""

    SPLITTING = 1
    """
    Operator splitting: symplectic steps for the energy-conserving forces, Runge-Kutta steps for the others.
    """

    SLIMPLECTIC = 2
    """
    Variational ('slimplectic') integrator that includes them in the action (default).
    """

class OpticsObs:
    """
    Optical tracker that reports directions only (azimuth, elevation), already extracted from the images: a telescope, infrared tracker, processed video, anything that outputs angles. Set ecef_to_enu to match position_ecef.
    """

    def __init__(self) -> None: ...

    @property
    def position_ecef(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Tracker location, Earth-fixed (ECEF), numpy (3,) [m]."""

    @position_ecef.setter
    def position_ecef(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def sigma_angle(self) -> float:
        """Azimuth/elevation noise σ [rad]. Default 50 µrad."""

    @sigma_angle.setter
    def sigma_angle(self, arg: float, /) -> None: ...

    @property
    def lat_rad(self) -> float:
        """Latitude [rad]; informational only, not used by the model."""

    @lat_rad.setter
    def lat_rad(self, arg: float, /) -> None: ...

    @property
    def lon_rad(self) -> float:
        """Longitude [rad]; informational only, not used by the model."""

    @lon_rad.setter
    def lon_rad(self, arg: float, /) -> None: ...

    @property
    def ecef_to_enu(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3, 3), order='F')]:
        """3×3 rotation from ECEF axes to local east-north-up axes; must be set."""

    @ecef_to_enu.setter
    def ecef_to_enu(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3, 3), order='F')], /) -> None: ...

    @property
    def known_bias(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(2), order='C')]:
        """
        Known bias [az rad, el rad], subtracted from every sample before estimation.
        """

    @known_bias.setter
    def known_bias(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(2), order='C')], /) -> None: ...

    @property
    def sigma_scale(self) -> list[float]:
        """Optional per-sample multiplier of sigma_angle (empty = none)."""

    @sigma_scale.setter
    def sigma_scale(self, arg: Sequence[float], /) -> None: ...

    @property
    def clock(self) -> SensorClock:
        """
        How the time tags are treated (SensorClock; default None_: assumed accurate).
        """

    @clock.setter
    def clock(self, arg: SensorClock, /) -> None: ...

    def predict(self, state: StateVector) -> Annotated[ArrayLike, dict(dtype='float64', shape=(2), order='C')]:
        """
        Predicted [az_rad, el_rad] for a satellite state (StateVector, ECEF); zeros if the satellite is within 1 m of the tracker.
        """

    @property
    def samples(self) -> list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(2), order='C')]]]:
        """Observations: list of (Epoch, [az, el])."""

    @samples.setter
    def samples(self, arg: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(2), order='C')]]], /) -> None: ...

    def set_samples(self, epoch0: Epoch, times_s: Annotated[ArrayLike, dict(dtype='float64', shape=(None), writable=False)], observations: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 2), writable=False)]) -> None:
        """
        Replace the samples from numpy arrays: times_s, shape (N,), seconds after epoch0; observations, shape (N, 2), [az_rad, el_rad]. Raises ValueError if the lengths differ.
        """

class PositionFixObs:
    """
    Measured Earth-fixed (ECEF) positions of the object: an impact location, GPS fixes, etc.
    """

    def __init__(self) -> None: ...

    @property
    def known_bias(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Known bias, ECEF, numpy (3,) [m], subtracted before estimation."""

    @known_bias.setter
    def known_bias(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def sigma_scale(self) -> list[float]:
        """Optional per-sample multiplier of sigma_pos (empty = none)."""

    @sigma_scale.setter
    def sigma_scale(self, arg: Sequence[float], /) -> None: ...

    @property
    def sigma_pos(self) -> float:
        """
        Position noise σ per axis [m] (default 3.0), shared by all fixes in this object.
        """

    @sigma_pos.setter
    def sigma_pos(self, arg: float, /) -> None: ...

    @property
    def samples(self) -> list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]:
        """Observations: list of (Epoch, ECEF position [m])."""

    @samples.setter
    def samples(self, arg: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], /) -> None: ...

    def set_samples(self, epoch0: Epoch, times_s: Annotated[ArrayLike, dict(dtype='float64', shape=(None), writable=False)], observations: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 3), writable=False)]) -> None:
        """
        Replace the samples from numpy arrays: times_s, shape (N,), seconds after epoch0; observations, shape (N, 3), ECEF positions [m]. Raises ValueError if the lengths differ.
        """

class PositionSamples:
    """
    Observed Earth-fixed (ECEF) positions for estimate_ea_bspline: a list of (Epoch, numpy (3,) position [m]).
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, samples: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]) -> None: ...

    @property
    def ecef(self) -> list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]:
        """The samples: list of (Epoch, numpy (3,)), ECEF"""

    @ecef.setter
    def ecef(self, arg: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], /) -> None: ...

class PropagateToAltitudeResult:
    """Result of TrajectoryInterpolator.propagate_to_altitude."""

    def __init__(self) -> None: ...

    @property
    def condition_met(self) -> bool:
        """True if the target altitude was reached"""

    @property
    def steps_taken(self) -> int:
        """Number of steps added to the history"""

    @property
    def starting_altitude_m(self) -> float:
        """Geodetic altitude (height above the WGS84 ellipsoid) at the start [m]"""

    @property
    def ending_altitude_m(self) -> float:
        """Geodetic altitude at the end [m]"""

class PropagateUntilResult:
    """
    Result of Trajectory.propagate_until: where it ended, why, and every event found on the way.
    """

    @property
    def epoch(self) -> Epoch:
        """Where the propagation ended (Epoch)"""

    @property
    def state_ecef(self) -> StateVector:
        """The Earth-fixed (ITRF) state there"""

    @property
    def stopped_by(self) -> events.Occurrence | None:
        """
        The terminal event that stopped it (Occurrence), or None if the target time was reached
        """

    @property
    def log(self) -> list[events.Occurrence]:
        """Every event found, in time order (list of Occurrence)"""

    def reached_target(self) -> bool:
        """
        True when the propagation reached its target time (no terminal event fired).
        """

class PropagationDirection(enum.Enum):
    """Time direction for TrajectoryInterpolator.propagate_to_altitude."""

    Forward = 0
    """Propagate forward in time from the last history entry"""

    Backward = 1
    """Propagate backward in time from the first history entry"""

class PropagationFrame(enum.Enum):
    """Frame the equations of motion are integrated in."""

    GCRS = 0
    """
    Geocentric Celestial Reference System: inertial, Earth-centered (IERS 2010 conventions). The only choice.
    """

class PropagationResult:
    """State at the end of Trajectory.propagate, Earth-fixed (ECEF)."""

    def __init__(self) -> None: ...

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Position, ECEF, numpy (3,) [m]"""

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Velocity relative to the rotating Earth, numpy (3,) [m/s]"""

    @property
    def acceleration(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Acceleration, ECEF axes, numpy (3,) [m/s^2]"""

    @property
    def forces(self) -> ForceBreakdown | None:
        """ForceBreakdown, or None unless TrajectoryConfig.report_forces is True"""

class PropagationTestResult:
    """Error statistics of a propagated trajectory compared with an EOF file."""

    def __init__(self) -> None: ...

    @property
    def start_epoch(self) -> Epoch:
        """First compared time (Epoch)"""

    @start_epoch.setter
    def start_epoch(self, arg: Epoch, /) -> None: ...

    @property
    def end_epoch(self) -> Epoch:
        """Last compared time (Epoch)"""

    @end_epoch.setter
    def end_epoch(self, arg: Epoch, /) -> None: ...

    @property
    def duration_seconds(self) -> float:
        """Time span compared [s]"""

    @duration_seconds.setter
    def duration_seconds(self, arg: float, /) -> None: ...

    @property
    def num_comparisons(self) -> int:
        """Number of compared samples"""

    @num_comparisons.setter
    def num_comparisons(self, arg: int, /) -> None: ...

    @property
    def position_rmse(self) -> float:
        """Root-mean-square position error [m]"""

    @position_rmse.setter
    def position_rmse(self, arg: float, /) -> None: ...

    @property
    def position_max_error(self) -> float:
        """Largest position error [m]"""

    @position_max_error.setter
    def position_max_error(self, arg: float, /) -> None: ...

    @property
    def position_mean_error(self) -> float:
        """Mean position error [m]"""

    @position_mean_error.setter
    def position_mean_error(self, arg: float, /) -> None: ...

    @property
    def velocity_rmse(self) -> float:
        """Root-mean-square velocity error [m/s]"""

    @velocity_rmse.setter
    def velocity_rmse(self, arg: float, /) -> None: ...

    @property
    def velocity_max_error(self) -> float:
        """Largest velocity error [m/s]"""

    @velocity_max_error.setter
    def velocity_max_error(self, arg: float, /) -> None: ...

    @property
    def along_track_rms(self) -> float:
        """RMS error along the direction of motion [m]"""

    @along_track_rms.setter
    def along_track_rms(self, arg: float, /) -> None: ...

    @property
    def cross_track_rms(self) -> float:
        """RMS error perpendicular to the orbit plane [m]"""

    @cross_track_rms.setter
    def cross_track_rms(self, arg: float, /) -> None: ...

    @property
    def radial_rms(self) -> float:
        """RMS error in the up/down (radial) direction [m]"""

    @radial_rms.setter
    def radial_rms(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

class RadarObs:
    """
    Ground radar: each sample is [range (m), azimuth (rad), elevation (rad), range rate (m/s, how fast the distance changes)] to the satellite; azimuth is measured from north toward east. Set ecef_to_enu to match position_ecef before passing it to an estimator (the identity default is only a placeholder).
    """

    def __init__(self) -> None: ...

    @property
    def position_ecef(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Radar location, Earth-fixed (ECEF), numpy (3,) [m]."""

    @position_ecef.setter
    def position_ecef(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def sigma_range(self) -> float:
        """Range noise standard deviation σ [m]. Default 2.0."""

    @sigma_range.setter
    def sigma_range(self, arg: float, /) -> None: ...

    @property
    def sigma_angle(self) -> float:
        """Azimuth/elevation noise σ [rad]. Default 0.02° = 3.5e-4 rad."""

    @sigma_angle.setter
    def sigma_angle(self, arg: float, /) -> None: ...

    @property
    def sigma_rate(self) -> float:
        """Range-rate noise σ [m/s]. Default 0.02."""

    @sigma_rate.setter
    def sigma_rate(self, arg: float, /) -> None: ...

    @property
    def lat_rad(self) -> float:
        """
        Radar latitude [rad]; informational only, not used by the model (set ecef_to_enu instead).
        """

    @lat_rad.setter
    def lat_rad(self, arg: float, /) -> None: ...

    @property
    def lon_rad(self) -> float:
        """Radar longitude [rad]; informational only, not used by the model."""

    @lon_rad.setter
    def lon_rad(self, arg: float, /) -> None: ...

    @property
    def known_bias(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]:
        """
        Known measurement bias [range m, az rad, el rad, rate m/s], subtracted from every observation before estimation. Default zero.
        """

    @known_bias.setter
    def known_bias(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')], /) -> None: ...

    @property
    def sigma_scale(self) -> list[float]:
        """
        Optional per-sample multiplier of the noise σ values (one entry per sample; empty = the σ values apply unchanged).
        """

    @sigma_scale.setter
    def sigma_scale(self, arg: Sequence[float], /) -> None: ...

    @property
    def clock(self) -> SensorClock:
        """How this radar's time tags are treated (SensorClock; default Shared)."""

    @clock.setter
    def clock(self, arg: SensorClock, /) -> None: ...

    @property
    def ecef_to_enu(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3, 3), order='F')]:
        """
        3×3 rotation from Earth-fixed (ECEF) axes to the radar's local east-north-up (ENU) axes; must match the location.
        """

    @ecef_to_enu.setter
    def ecef_to_enu(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3, 3), order='F')], /) -> None: ...

    def predict(self, state: StateVector) -> Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]:
        """
        Predicted [range_m, az_rad, el_rad, range_rate_mps] for a satellite state (StateVector, Earth-fixed, m and m/s); zeros if the satellite is within 1 m of the radar. Useful for making synthetic observations.
        """

    @property
    def samples(self) -> list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]]]:
        """Observations: list of (Epoch, [range, az, el, range_rate])."""

    @samples.setter
    def samples(self, arg: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(4), order='C')]]], /) -> None: ...

    def set_samples(self, epoch0: Epoch, times_s: Annotated[ArrayLike, dict(dtype='float64', shape=(None), writable=False)], observations: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 4), writable=False)]) -> None:
        """
        Replace the samples from numpy arrays: times_s, shape (N,), seconds after epoch0; observations, shape (N, 4), [range_m, az_rad, el_rad, range_rate_mps]. Raises ValueError if the lengths differ.
        """

class RemovedApiError(BallisticError):
    pass

class Rk45Options:
    """Options for the RK45 (Dormand-Prince) integrator."""

    def __init__(self) -> None: ...

    @property
    def initial_step_size(self) -> float:
        """
        First step size to try [s] (default 60.0). Use about 8 s for gliding reentry or other fast-changing motion.
        """

    @initial_step_size.setter
    def initial_step_size(self, arg: float, /) -> None: ...

    @property
    def max_step_size(self) -> float:
        """
        Largest step size allowed [s] (default 3600.0). Use about 8 s for gliding reentry or other fast-changing motion.
        """

    @max_step_size.setter
    def max_step_size(self, arg: float, /) -> None: ...

class SP3Arc:
    """
    All records of one satellite in one SP3 file, in the file's frame.

    epoch0 is the first record's epoch (converted from the file's time system);
    offsets_ns (int64) / offsets_s (float) are exact times of the records
    after it. position is numpy (N, 3) [m]; velocity is numpy (N, 3) [m/s]
    with NaN where the file has none.
    """

    @property
    def satellite(self) -> str:
        """Satellite ID, e.g. 'G01'"""

    @property
    def epoch0(self) -> Epoch:
        """Epoch of the first record"""

    @property
    def missing_positions(self) -> int:
        """
        Number of records skipped because the file marks the position bad or absent
        """

    @property
    def interval_s(self) -> float:
        """Nominal record spacing [s] (from the header, else the median spacing)"""

    def __len__(self) -> int:
        """Number of records"""

    @property
    def offsets_ns(self) -> Annotated[ArrayLike, dict(dtype='int64', shape=(None))]: ...

    @property
    def offsets_s(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None))]: ...

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(None, 3))]: ...

    @property
    def velocity(self) -> object: ...

    def has_velocity(self) -> bool:
        """True if every record carries a velocity."""

    def epoch(self, index: int) -> Epoch:
        """Epoch of record `index`"""

    def segments(self) -> list[tuple[int, int]]:
        """
        Index ranges (first, last) of the stretches without a gap (a gap: spacing more than 1.5 times the nominal interval).
        """

    def interpolate(self, t_s: float, order: int = 9) -> tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]:
        """
        Tuple (position [m], velocity [m/s]) at t_s seconds after epoch0, by Lagrange-polynomial interpolation through `order` records inside one gap-free stretch. Raises DataCoverageError outside the arc or in a gap.
        """

    def derived_velocities(self, order: int = 9) -> tuple:
        """
        Velocities computed by differentiating the positions (Lagrange polynomial through `order` records): tuple (velocity (N, 3) [m/s], centered (N,) uint8). centered = 0 where the window is one-sided (at the ends of a gap-free stretch, about 5e-4 m/s error); do not start a prediction there.
        """

class SP3File:
    """A parsed SP3 file: its header and one SP3Arc per satellite."""

    @property
    def source(self) -> str:
        """Path or name of the file, for messages"""

    @property
    def header(self) -> SP3Header:
        """The SP3Header"""

    @property
    def arcs(self) -> list[SP3Arc]:
        """
        List of SP3Arc, in order of first record (satellites with records only)
        """

    @property
    def missing_positions(self) -> int:
        """Number of bad or absent positions skipped, all satellites"""

    def arc(self, satellite: str) -> SP3Arc:
        """
        The SP3Arc of `satellite` (e.g. 'G01'); raises, listing the satellites present, if the file has no records for it
        """

    def frame(self) -> SP3FrameInfo:
        """
        What the header's coordinate-system label means (SP3FrameInfo); raises FrameError for an unknown label.
        """

class SP3FrameInfo:
    """
    What an SP3 file's coordinate-system label means: Earth-fixed or inertial, and for an Earth-fixed label the ITRF version it names (None for labels that do not state one, such as GFZ's 'CTS').
    """

    @property
    def inertial(self) -> bool:
        """True for an inertial label (J2000, J2K, EME00), False for Earth-fixed"""

    @property
    def itrf(self) -> ITRFRealization | None:
        """ITRF version (ITRFRealization) of an Earth-fixed label, or None"""

class SP3Header:
    """The header of an SP3 file."""

    @property
    def version(self) -> str:
        """Format version letter, 'a' to 'd'"""

    @property
    def pos_or_vel(self) -> str:
        """'P' = positions only, 'V' = positions and velocities"""

    @property
    def frame(self) -> str:
        """Coordinate-system label, e.g. 'IGS20', 'J2000', 'CTS'"""

    @property
    def orbit_type(self) -> str:
        """Producer's orbit-type code, e.g. 'FIT'"""

    @property
    def agency(self) -> str:
        """Producing agency code"""

    @property
    def time_system(self) -> SP3TimeSystem:
        """Time system of the epochs (SP3TimeSystem)"""

    @property
    def interval_s(self) -> float:
        """Nominal spacing between epochs [s]"""

    @property
    def declared_epochs(self) -> int:
        """Number of epochs stated in the first header line"""

    @property
    def satellites(self) -> list[str]:
        """Satellite IDs listed in the header"""

class SP3TimeSystem(enum.Enum):
    """
    Time system an SP3 file's epochs are written in: GPS, Galileo (GAL), QZSS (QZS) and NavIC (IRN) system times, BeiDou time (BDT, GPS - 14 s), TAI, UTC, or GLONASS time (GLO).
    """

    GPS = 0

    GAL = 1

    QZS = 2

    IRN = 3

    BDT = 4

    TAI = 5

    UTC = 6

    GLO = 7

class SensorClock(enum.Enum):
    """
    How the smoother treats a sensor's time tags (its clock). A clock offset makes every sample look as if taken slightly earlier or later; the smoother can estimate such an offset when EASmootherConfig.estimate_clock_bias is on.
    """

    None_ = 0
    """The time tags are accurate; no clock offset is estimated for this sensor"""

    Shared = 1
    """
    One offset shared by every sensor marked Shared (the radar default: all radars on one clock)
    """

    Own = 2
    """This sensor gets its own clock offset"""

class SensorMask:
    """
    Which sensors of a scenario an estimate uses: one on/off flag per sensor, in the scenario's order. Start from `SensorMask.all_on(scenario)` and switch groups off with the chaining `.with_*(False)` methods. A sensor missing from the mask counts as on.
    """

    def __init__(self) -> None: ...

    @staticmethod
    def all_on(scenario: MultiSensorScenario) -> SensorMask:
        """Mask with every sensor of `scenario` switched on"""

    @property
    def radars(self) -> list[bool]:
        """One flag per radar (list of bool)"""

    @radars.setter
    def radars(self, arg: Sequence[bool], /) -> None: ...

    @property
    def optics(self) -> list[bool]:
        """One flag per optical tracker (list of bool)"""

    @optics.setter
    def optics(self, arg: Sequence[bool], /) -> None: ...

    @property
    def fixes(self) -> list[bool]:
        """One flag per position-fix sensor (list of bool)"""

    @fixes.setter
    def fixes(self, arg: Sequence[bool], /) -> None: ...

    @property
    def accels(self) -> list[bool]:
        """One flag per accelerometer (list of bool)"""

    @accels.setter
    def accels(self, arg: Sequence[bool], /) -> None: ...

    def with_radars(self, arg: bool, /) -> SensorMask:
        """Set every radar flag to `on`; returns the mask."""

    def with_optics(self, arg: bool, /) -> SensorMask:
        """Set every optical-tracker flag to `on`; returns the mask."""

    def with_fixes(self, arg: bool, /) -> SensorMask:
        """Set every position-fix flag to `on`; returns the mask."""

    def with_accels(self, arg: bool, /) -> SensorMask:
        """Set every accelerometer flag to `on`; returns the mask."""

class Sgp4:
    """
    The SGP4/SDP4 propagator for one element set; outputs TEME states. Sgp4(tle, gravity='wgs72', mode='improved'): gravity picks the Earth constants ('wgs72', what element sets are fitted with; 'wgs72old'; 'wgs84'), mode picks 'improved' or the original 'afspc' operation. Raises ConfigError for other values.
    """

    def __init__(self, tle: Tle, *, gravity: str = 'wgs72', mode: str = 'improved') -> None: ...

    @property
    def tle(self) -> Tle:
        """The element set (Tle)"""

    @property
    def deep_space(self) -> bool:
        """
        True if the deep-space model (SDP4; orbital period of 225 minutes or more) is used
        """

    def propagate_minutes(self, tsince_min: float) -> tuple:
        """
        State tsince_min minutes after the element-set epoch: a tuple (TEME position [km], velocity [km/s], SGP4 error code). Error code 0 = ok; 1-4 = the model failed (state is NaN); 6 = the satellite has decayed (state still returned).
        """

    def propagate_minutes_array(self, tsince_min: Annotated[ArrayLike, dict(dtype='float64', shape=(None), order='C', writable=False)]) -> tuple:
        """
        propagate_minutes over a numpy (N,) array of minutes: returns (r_km (N,3), v_km_s (N,3), status (N,)).
        """

    def teme(self, epoch: Epoch) -> tuple:
        """
        TEME position [m] and velocity [m/s] at an Epoch, as a tuple; raises when the model fails there.
        """

class SpaceWeatherHistory:
    """
    Space weather over time: daily F10.7 solar-flux values and 3-hourly Ap
    geomagnetic indices, the inputs the NRLMSIS 2.1 density model and the HWM14
    wind model need. Used by the drag-environment builders.

    SpaceWeatherHistory.from_loaded(start, end) builds the measured history from
    the loaded CelesTrak space-weather file; this is what make_drag_environment
    uses by default. from_constant(WeatherConditions) turns one fixed set of
    values into a history valid at every epoch (e.g. a deliberate quiet-Sun or
    storm scenario); the windowed form raises outside its window.
    """

    def __init__(self) -> None:
        """Empty history (at() raises)."""

    @overload
    @staticmethod
    def from_constant(w: WeatherConditions) -> SpaceWeatherHistory:
        """History holding the WeatherConditions `w` at every epoch."""

    @overload
    @staticmethod
    def from_constant(w: WeatherConditions, window: TimeWindow) -> SpaceWeatherHistory:
        """
        History holding the WeatherConditions `w` inside `window` (a TimeWindow); at() raises outside it.
        """

    @staticmethod
    def last_measured_day() -> Epoch | None:
        """
        Start (UTC) of the last day with measured (not forecast) weather in the loaded file, as an Epoch, or None if nothing is loaded. Drag past the end of that day raises unless forecasts are allowed.
        """

    @staticmethod
    def from_loaded(start: Epoch, end: Epoch, allow_predicted: bool = True) -> SpaceWeatherHistory:
        """
        Measured, time-varying space weather between the Epochs start and end, from
        the file loaded by ballistic.load() / load_space_weather(): the previous
        day's observed F10.7, its 81-day centered mean, and the 3-hourly Ap values in
        the order NRLMSIS 2.1 reads them. This is what make_drag_environment uses
        when weather=None.

        Past the last measured day, CelesTrak's daily forecast (about 45 days) fills
        in, with a RuntimeWarning; allow_predicted=False refuses forecasts instead.

        Raises:
            IndexError: a day the window needs (the model looks back 3 days) is not
                in the file, has no 3-hourly Ap, or is a forecast while
                allow_predicted=False. No quiet default values are ever substituted.
        """

    def at(self, epoch: Epoch) -> WeatherConditions:
        """The WeatherConditions (F10.7, Ap, ...) in effect at `epoch`."""

    def empty(self) -> bool:
        """True if the history holds no values."""

class SrpConfig:
    """
    Radiation-pressure settings: the push of sunlight on the satellite (solar
    radiation pressure, SRP), plus sunlight reflected by the Earth (albedo) and
    the Earth's own infrared glow.

    Assigning one to TrajectoryConfig.srp turns these forces on; leaving
    srp = None keeps them off.

    Args (keyword-only):
        mass_kg: satellite mass [kg], > 0
        area_m2: sunlit cross-section [m^2], >= 0
        cr: radiation pressure coefficient (default 1.3)
        albedo: include Earth albedo (default True)
        earth_ir: include Earth infrared (default True)

    Raises:
        ballistic.errors.ConfigError: mass_kg not > 0 or area_m2 not >= 0.
    """

    def __init__(self, *, mass_kg: float, area_m2: float, cr: float = 1.3, albedo: bool = True, earth_ir: bool = True) -> None: ...

    @property
    def mass_kg(self) -> float:
        """
        Satellite mass [kg]; used by the radiation forces only (for drag, the mass enters through the DragEnvironment's ballistic coefficient beta)
        """

    @mass_kg.setter
    def mass_kg(self, arg: float, /) -> None: ...

    @property
    def area_m2(self) -> float:
        """Sun-facing cross-section [m^2]"""

    @area_m2.setter
    def area_m2(self, arg: float, /) -> None: ...

    @property
    def cr(self) -> float:
        """
        Radiation pressure coefficient Cr (1.0 = perfect absorber, about 1.3 typical)
        """

    @cr.setter
    def cr(self, arg: float, /) -> None: ...

    @property
    def albedo(self) -> bool:
        """Include sunlight reflected by the Earth (default True)"""

    @albedo.setter
    def albedo(self, arg: bool, /) -> None: ...

    @property
    def earth_ir(self) -> bool:
        """Include the Earth's infrared radiation (default True)"""

    @earth_ir.setter
    def earth_ir(self, arg: bool, /) -> None: ...

class State:
    """
    Where a trajectory starts: epoch, position, velocity, and the coordinate frame
    they are given in.

    Frames: the ITRF is Earth-fixed (it rotates with the Earth; velocities are
    measured relative to the rotating ground). The GCRS is inertial (it does not
    rotate). The same position has different velocities in the two frames, and
    mixing them up is a common, silent mistake, so the frame must be named.

    Build a State with one of the named constructors. Each one checks the state
    and raises with the fix when something looks wrong:

        State.from_itrf(epoch, r_m, v_m_s, realization=None)  # Earth-fixed velocity
        State.from_gcrs(epoch, r_m, v_m_s)                    # inertial (GCRS)
        State.from_j2000(epoch, r_m, v_m_s)                   # inertial, J2000 axes
        State.from_keplerian(epoch, a_m=..., e=..., i_deg=..., raan_deg=..., argp_deg=..., true_anomaly_deg=...)
        State.from_teme(epoch, r_m, v_m_s)                    # the frame SGP4 outputs
        State.from_tle(line1, line2, epoch=None)              # from a two-line element set (TLE)
        State.from_sp3(sp3_file, satellite, epoch)            # from a precise-orbit (SP3) file

    Positions are in meters, velocities in m/s (3-element sequences or numpy
    arrays).

    Raises:
        ballistic.errors.InputPlausibilityError: non-finite numbers, a position
            that looks like kilometers instead of meters (|r| < 100 km) or is
            beyond 1e6 km, a position more than 1 km below the ground, or an orbit
            whose lowest point is below 100 km. A reentry, ballistic or launch
            state is legitimately below that: pass allow_suborbital=True.
        ballistic.errors.FrameError: the numbers only make sense in the other frame
            (an inertial velocity given as Earth-fixed, or the reverse). Pass
            check_frame=False to skip this check.
    """

    @staticmethod
    def from_itrf(*args, **kwargs):
        """
        from_itrf(epoch: ballistic._core.Epoch, r_m: numpy.ndarray[dtype=float64, shape=(3), order='C'], v_m_s: numpy.ndarray[dtype=float64, shape=(3), order='C'], realization: ballistic._core.ITRFRealization | None = None, *, allow_suborbital: bool = False, check_frame: bool = True) -> ballistic._core.State

        Earth-fixed (ITRF) start state: position [m] and velocity relative to the rotating Earth [m/s]. realization: which ITRF version the coordinates are in (an ITRFRealization; None = use the trajectory config's). allow_suborbital: accept an orbit that dips below 100 km. check_frame: run the Earth-fixed vs inertial mix-up check. Raises InputPlausibilityError / FrameError (see State).
        """

    @staticmethod
    def from_gcrs(*args, **kwargs):
        """
        from_gcrs(epoch: ballistic._core.Epoch, r_m: numpy.ndarray[dtype=float64, shape=(3), order='C'], v_m_s: numpy.ndarray[dtype=float64, shape=(3), order='C'], *, allow_suborbital: bool = False, check_frame: bool = True) -> ballistic._core.State

        Inertial (GCRS) start state: position [m] and velocity [m/s]. Raises InputPlausibilityError / FrameError (see State).
        """

    @staticmethod
    def from_j2000(*args, **kwargs):
        """
        from_j2000(epoch: ballistic._core.Epoch, r_m: numpy.ndarray[dtype=float64, shape=(3), order='C'], v_m_s: numpy.ndarray[dtype=float64, shape=(3), order='C'], *, allow_suborbital: bool = False, check_frame: bool = True) -> ballistic._core.State

        Inertial start state on the mean equator and equinox of J2000.0 (EME2000), position [m] and velocity [m/s]. Rotated to the GCRS by the small fixed IAU 2006 frame-bias rotation (up to about 0.8 m at low Earth orbit). Raises InputPlausibilityError / FrameError (see State).
        """

    @staticmethod
    def from_keplerian(epoch: Epoch, *, a_m: float, e: float, i_deg: float, raan_deg: float, argp_deg: float, true_anomaly_deg: float, allow_suborbital: bool = False, check_frame: bool = True) -> State:
        """
        Start state from classical (osculating, two-body) orbital elements in the GCRS, using the WGS84 Earth gravitational constant GM. a_m: semi-major axis [m]; e: eccentricity; i_deg: inclination; raan_deg: right ascension of the ascending node; argp_deg: argument of perigee; true_anomaly_deg: true anomaly. Angles in degrees. Raises InputPlausibilityError (see State).
        """

    @staticmethod
    def from_teme(*args, **kwargs):
        """
        from_teme(epoch: ballistic._core.Epoch, r_m: numpy.ndarray[dtype=float64, shape=(3), order='C'], v_m_s: numpy.ndarray[dtype=float64, shape=(3), order='C'], *, allow_suborbital: bool = False, check_frame: bool = True) -> ballistic._core.State

        Start state in TEME (True Equator, Mean Equinox: the frame the SGP4 propagator outputs), position [m] and velocity [m/s]. Converted to the ITRF with the 1982 Greenwich mean sidereal time (GMST) and polar motion. Raises InputPlausibilityError / FrameError (see State).
        """

    @property
    def epoch(self) -> Epoch:
        """The start time (Epoch)"""

    @property
    def frame(self) -> str:
        """Name of the frame position and velocity are given in (str)"""

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Position as given, in the frame named by .frame; numpy (3,) [m]"""

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Velocity as given, in the frame named by .frame; numpy (3,) [m/s]"""

    @property
    def realization(self) -> ITRFRealization | None:
        """
        The ITRF version of an Earth-fixed state, or None if not stated / not Earth-fixed
        """

    def itrf_state(self, itrf: ITRFRealization = ITRFRealization.ITRF2020) -> tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]:
        """
        The state converted to Earth-fixed coordinates in ITRF version `itrf`: a tuple (position [m], velocity relative to the rotating Earth [m/s]), each numpy (3,).
        """

    def gcrs_state(self, itrf: ITRFRealization = ITRFRealization.ITRF2020) -> tuple[Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]:
        """
        The state converted to the inertial GCRS: a tuple (position [m], velocity [m/s]), each numpy (3,). An Earth-fixed state is read in its own ITRF version if it has one, otherwise in `itrf`.
        """

    def __repr__(self) -> str: ...

class StateTransition:
    """
    State transition matrix Phi(t, t0) and covariance along a propagated Trajectory.

    Phi maps a small change of the state at the first recorded epoch t0 into the
    change at t: dx(t) = Phi dx(t0), with x = [r (m); v (m/s)] in the inertial GCRS
    frame. It is integrated along the recorded steps (dPhi/dt = A(t) Phi), so the
    trajectory must have been propagated with step() or propagate_until() first.
    covariance(t, P_0) = Phi P_0 Phi^T (no process noise, no consider parameters).

    Caveats: radiation pressure uses a hard Earth shadow, so an arc crossing the
    shadow boundary gets a Phi that misses the jump in sunlight; drag's position
    derivatives come from finite differences. Most users reach this through
    ballistic.Trajectory.stm / covariance.
    """

    def __init__(self, trajectory: Trajectory, options: StmOptions = ...) -> None:
        """
        Integrate Phi over the trajectory's recorded history (keeps the trajectory alive). Raises ConfigError with fewer than two recorded states.
        """

    def stm(self, epoch: Epoch) -> Annotated[ArrayLike, dict(dtype='float64', shape=(6, 6), order='F')]:
        """
        Phi(epoch, epoch0()) as a (6, 6) array, GCRS, [r (m); v (m/s)]. Raises DataCoverageError outside [epoch0(), epoch_end()].
        """

    def covariance(self, epoch: Epoch, p0: Annotated[ArrayLike, dict(dtype='float64', shape=(6, 6), order='F')]) -> Annotated[ArrayLike, dict(dtype='float64', shape=(6, 6), order='F')]:
        """
        Phi P_0 Phi^T as a (6, 6) array: the start covariance P_0 (symmetric, GCRS, [r (m); v (m/s)]) carried to `epoch`. Raises InputPlausibilityError for a non-symmetric or non-finite P_0.
        """

    def epoch0(self) -> Epoch:
        """First recorded epoch: Phi is relative to it"""

    def epoch_end(self) -> Epoch:
        """Last recorded epoch covered"""

    def node_count(self) -> int:
        """Number of recorded steps Phi is stored at"""

class StateVector:
    """
    Position and velocity; the frame depends on where the state came from (see the returning function).
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, position: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> None: ...

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Position, numpy (3,) [m]"""

    @position.setter
    def position(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Velocity, numpy (3,) [m/s]"""

    @velocity.setter
    def velocity(self, arg: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], /) -> None: ...

class StepResult:
    """
    What Trajectory.step() and step_backward() return: the new state (ECEF) and step information.
    """

    def __init__(self) -> None: ...

    @property
    def epoch(self) -> Epoch:
        """Time after the step (Epoch)"""

    @property
    def position(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """New position, ECEF, numpy (3,) [m]"""

    @property
    def velocity(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """New velocity relative to the rotating Earth, numpy (3,) [m/s]"""

    @property
    def acceleration(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """Acceleration at the new state, ECEF axes [m/s^2]"""

    @property
    def step_size_taken(self) -> float:
        """Step size actually used [s]"""

    @property
    def next_step_size(self) -> float:
        """Step size suggested for the next step [s]"""

    @property
    def history_index(self) -> int:
        """Position of the new state in the history"""

    @property
    def forces(self) -> ForceBreakdown | None:
        """ForceBreakdown, or None unless TrajectoryConfig.report_forces is True"""

class StmOptions:
    """
    Settings for StateTransition: finite-difference offsets for the force terms without an analytic Jacobian, and the largest Runge-Kutta substep.
    """

    def __init__(self) -> None: ...

    @property
    def fd_dx(self) -> float:
        """Finite-difference position offset [m] (drag's position columns)"""

    @fd_dx.setter
    def fd_dx(self, arg: float, /) -> None: ...

    @property
    def fd_dv(self) -> float:
        """
        Finite-difference velocity offset [m/s] (a velocity-dependent drag coefficient)
        """

    @fd_dv.setter
    def fd_dv(self, arg: float, /) -> None: ...

    @property
    def max_substep_s(self) -> float:
        """Largest Runge-Kutta substep [s] for the transition matrix"""

    @max_substep_s.setter
    def max_substep_s(self, arg: float, /) -> None: ...

class StrictDiagnosticError(BallisticError):
    pass

class SymplecticOptions:
    """Options for the symplectic (Yoshida) integrators."""

    def __init__(self) -> None: ...

    @property
    def step_size(self) -> float:
        """
        Fixed step size [s], or the reference step when adaptive (default: 30.0)
        """

    @step_size.setter
    def step_size(self, arg: float, /) -> None: ...

    @property
    def nc_strategy(self) -> NonConservativeStrategy:
        """Strategy for non-conservative forces (default: SLIMPLECTIC)."""

    @nc_strategy.setter
    def nc_strategy(self, arg: NonConservativeStrategy, /) -> None: ...

    @property
    def adaptive(self) -> bool:
        """
        Use time-reversible adaptive step sizes (Hairer, Lubich & Wanner 2006; default: False)
        """

    @adaptive.setter
    def adaptive(self, arg: bool, /) -> None: ...

    @property
    def min_step_size(self) -> float:
        """Smallest adaptive step size [s] (default: 1.0)"""

    @min_step_size.setter
    def min_step_size(self, arg: float, /) -> None: ...

    @property
    def max_step_size(self) -> float:
        """Largest adaptive step size [s] (default: 120.0)"""

    @max_step_size.setter
    def max_step_size(self, arg: float, /) -> None: ...

    @property
    def adaptation_alpha(self) -> float | None:
        """
        Exponent of the adaptive step-size rule (Optional[float]; None = automatic, 1/order)
        """

    @adaptation_alpha.setter
    def adaptation_alpha(self, arg: float | None) -> None: ...

    @property
    def symmetry_tolerance(self) -> float:
        """
        Relative tolerance for the iteration that keeps adaptive steps time-symmetric (default: 0.1)
        """

    @symmetry_tolerance.setter
    def symmetry_tolerance(self, arg: float, /) -> None: ...

class TimeScale(enum.Enum):
    """
    Time scales an Epoch can be read or written in. They differ by offsets of up to about a minute, which matter: one second is about 7.5 km of travel in low Earth orbit.
    """

    UTC = 0
    """
    Coordinated Universal Time: civil clock time, kept within a second of the Earth's rotation by leap seconds
    """

    UT1 = 1
    """
    Universal Time 1: time defined by the Earth's actual rotation angle (needs Earth orientation data)
    """

    TAI = 2
    """International Atomic Time: uniform, no leap seconds"""

    TT = 3
    """Terrestrial Time (TT = TAI + 32.184 s)"""

    TDB = 4
    """
    Barycentric Dynamical Time: the time argument of planetary ephemerides (within 2 ms of TT)
    """

    GPS = 5
    """GPS Time (GPS = TAI - 19 s)"""

class TimeWindow:
    """
    Time interval [start, end) (start included, end excluded), two Epoch
    values. Used to bound where a SpaceWeatherHistory is valid.
    """

    def __init__(self, start: Epoch, end: Epoch) -> None: ...

    @property
    def start(self) -> Epoch:
        """First instant (Epoch, included)"""

    @start.setter
    def start(self, arg: Epoch, /) -> None: ...

    @property
    def end(self) -> Epoch:
        """End (Epoch, excluded)"""

    @end.setter
    def end(self, arg: Epoch, /) -> None: ...

class Tle:
    """
    A parsed two-line element set (TLE). Its elements are SGP4 mean elements: feed them to Sgp4, not to a numerical propagator (read as ordinary orbital elements they are kilometers off).
    """

    @staticmethod
    def parse(line1: str, line2: str) -> Tle:
        """
        Parse a TLE strictly: 69 columns, valid checksums, the same catalog number on both lines, blanks where the format has separators, well-formed fields (Alpha-5 catalog numbers accepted). Raises InputPlausibilityError naming the line and column at fault.
        """

    @property
    def line1(self) -> str:
        """Line 1 as given (trailing blanks removed)"""

    @property
    def line2(self) -> str:
        """Line 2 as given (trailing blanks removed)"""

    @property
    def catalog_number(self) -> int:
        """Satellite catalog number (int; Alpha-5 decoded, A0001 -> 100001)"""

    @property
    def classification(self) -> str:
        """'U' unclassified, 'C' classified, 'S' secret"""

    @property
    def international_designator(self) -> str:
        """Launch year, launch number and piece, e.g. '98067A' (may be blank)"""

    @property
    def epoch_year(self) -> int:
        """Epoch year, four digits"""

    @property
    def epoch_day(self) -> float:
        """Epoch day of year plus fraction (1.0 = 1 January 00:00 UTC)"""

    @property
    def ndot_rev_day2(self) -> float:
        """
        First time derivative of the mean motion, divided by 2 [rev/day^2] (not used by SGP4)
        """

    @property
    def nddot_rev_day3(self) -> float:
        """
        Second time derivative of the mean motion, divided by 6 [rev/day^3] (not used by SGP4)
        """

    @property
    def bstar_per_er(self) -> float:
        """B*, SGP4's drag term [1 / Earth radii]"""

    @property
    def ephemeris_type(self) -> int:
        """Ephemeris type (0 in distributed element sets)"""

    @property
    def element_set_number(self) -> int:
        """Element set number"""

    @property
    def inclination_deg(self) -> float:
        """Inclination: tilt of the orbit plane to the equator [deg]"""

    @property
    def raan_deg(self) -> float:
        """Right ascension of the ascending node [deg]"""

    @property
    def eccentricity(self) -> float:
        """Eccentricity (0 = circular)"""

    @property
    def argp_deg(self) -> float:
        """Argument of perigee [deg]"""

    @property
    def mean_anomaly_deg(self) -> float:
        """Mean anomaly [deg]"""

    @property
    def mean_motion_rev_day(self) -> float:
        """Mean motion (Kozai convention) [revolutions per day]"""

    @property
    def revolution_number(self) -> int:
        """Number of orbits completed at the epoch"""

    @property
    def epoch(self) -> Epoch:
        """
        The element-set epoch (Epoch, UTC). Raises DataCoverageError for epoch years 1957-1960, before UTC existed.
        """

    def epoch_days_since_1950(self) -> float:
        """
        Days since 1949 December 31 00:00 UTC (the time argument of SGP4's lunar-solar and Earth-rotation terms).
        """

    def beta_from_bstar_kg_m2(self) -> float:
        """
        Ballistic coefficient m / (Cd*A) [kg/m^2] implied by B*: 1 / (12.741621 B*). B* is a fitted term that also absorbs other model errors; it can be zero (result infinite) or negative (result negative).
        """

    def __repr__(self) -> str: ...

class TleGradePreset:
    """
    Which forces still matter when only TLE-level accuracy is needed, for one orbit and prediction horizon. A force is kept if its measured effect exceeds 10% of the typical TLE error for the orbit class (Flohrer et al. 2009). The table behind it was measured by scripts/measure_tle_grade_table.py.
    """

    @property
    def regime(self) -> str:
        """
        Orbit class: 'LEO' (low Earth orbit), 'MEO' (medium Earth orbit) or 'GEO' (geostationary)
        """

    @property
    def horizon_s(self) -> float:
        """The measured horizon used [s] (at least the one requested)"""

    @property
    def rows_altitude_km_min(self) -> float:
        """Lowest altitude of the measured table rows merged [km]"""

    @property
    def rows_altitude_km_max(self) -> float:
        """Highest altitude of the measured table rows merged [km]"""

    @property
    def budget_m(self) -> float:
        """Typical TLE error for the orbit class [m] (Flohrer et al. 2009)"""

    @property
    def threshold_m(self) -> float:
        """Keep-threshold: 10% of the budget [m]"""

    @property
    def gravity_degree(self) -> int:
        """Gravity-model degree to use"""

    @property
    def drag(self) -> bool:
        """Keep atmospheric drag"""

    @property
    def radiation(self) -> bool:
        """Keep radiation pressure"""

    @property
    def third_body(self) -> bool:
        """Keep Sun/Moon/Jupiter gravity"""

    @property
    def tides(self) -> bool:
        """Keep solid-Earth tides"""

    @property
    def ocean_tides(self) -> bool:
        """Keep ocean tides"""

    @property
    def relativity(self) -> bool:
        """Keep relativistic corrections"""

    @property
    def effect_drag_m(self) -> float:
        """Largest measured effect of drag over the horizon [m]"""

    @property
    def effect_radiation_m(self) -> float:
        """Largest measured effect of radiation pressure [m]"""

    @property
    def effect_third_body_m(self) -> float:
        """Largest measured effect of third-body gravity [m]"""

    @property
    def effect_tides_m(self) -> float:
        """Largest measured effect of solid-Earth tides [m]"""

    @property
    def effect_ocean_tides_m(self) -> float:
        """Largest measured effect of ocean tides [m]"""

    @property
    def effect_relativity_m(self) -> float:
        """Largest measured effect of relativity [m]"""

    @property
    def combined_error_m(self) -> float:
        """
        Largest measured difference between the whole preset and the full configuration [m]
        """

    @property
    def speedup(self) -> float:
        """Smallest measured speed-up over the full configuration (wall time)"""

    def __repr__(self) -> str: ...

class Trajectory:
    """
    Trajectory - the orbit propagator: steps a satellite's state forward (or
    backward) in time and keeps every step in a history.

    Two numerical integrators are available:
    - RK45 (default): Dormand-Prince adaptive-step Runge-Kutta; most accurate for
      predictions of hours.
    - Yoshida (6th or 8th order): symplectic, fixed-step integrators that keep
      the orbit's energy from drifting; better for runs of many hours to days.

    Forces modeled:
    - Earth's gravity field (EGM2008 model, embedded up to degree 360)
    - Sun, Moon and Jupiter gravity (third-body perturbations)
    - Earth rotation and orientation (IAU 2006/2000A precession-nutation)
    - General-relativity corrections (Schwarzschild and Lense-Thirring)
    - Optional: solar radiation pressure, atmospheric drag, ocean tides

    Position error one hour after the start, measured against precise orbit
    products (README "Accuracy" table):
    - Low Earth orbit: about 0.22 m (Sentinel satellites, with radiation
      pressure, drag and ocean tides: ballistic.recommended_config)
    - Medium Earth orbit: 0.15 m (LAGEOS / Etalon laser-ranging satellites);
      0.67 m (GPS, without radiation pressure)
    - Inclined geosynchronous orbit: 0.74 m (QZSS, without radiation pressure)
    - Geostationary orbit: 0.80 m (BeiDou, without radiation pressure)

    Example:
        import ballistic
        import numpy as np

        # Create epoch and initial state (the frame is named and checked)
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        # Earth-fixed (ITRF): circular 7669 m/s minus Earth rotation 494 m/s
        state = ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0])

        # Create trajectory with default config
        traj = ballistic._core.Trajectory(state)

        # Step-based propagation
        while traj.epoch() - epoch < 3600.0:
            result = traj.step()

        # Access history
        for i in range(traj.history_size()):
            entry = traj.history_at(i)
            print(f"t={entry.epoch}, pos={entry.position}")
    """

    @overload
    def __init__(self, state: State, config: TrajectoryConfig = ...) -> None:
        """
        Build a trajectory from a State (frame named and checked) and an optional TrajectoryConfig. The recommended form.
        """

    @overload
    def __init__(self, epoch: Epoch, position: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], config: TrajectoryConfig = ...) -> None:
        """
        Build a trajectory from an Earth-fixed (ITRF) state. Only the units are
        checked here; prefer Trajectory(State.from_itrf(...)), which also checks the
        orbit and the frame.

        Args:
            epoch: Start time (Epoch)
            position: Start position, Earth-fixed (ITRF), numpy (3,) [m]
            velocity: Start velocity relative to the rotating Earth, numpy (3,) [m/s]
            config: Optional TrajectoryConfig (the library defaults if omitted)

        Raises:
            ballistic.errors.InputPlausibilityError: the position looks like
                kilometers rather than meters, or is not finite.
        """

    def propagate_until(self, target: Epoch, events: Sequence[events.Event] = [], stop_at_ground: bool = True) -> PropagateUntilResult:
        """
        Step forward to the `target` Epoch, watching for `events` (a list of
        ballistic.events.Event). When an event happens inside a step, its time is
        located to 1e-4 s by re-integrating that step, so the event state is as
        accurate as the integrator. Propagation ends at the first terminal event or
        at the target; every event found is logged. With stop_at_ground=True (the
        default) the ground-impact event is added, so a reentering object stops on
        the WGS84 ellipsoid instead of integrating into the Earth. Steps are
        recorded in the history as with step().

        Returns:
            PropagateUntilResult (epoch, state_ecef, stopped_by, log, reached_target())
        """

    def propagate(self, target: Epoch) -> PropagationResult:
        """
        Propagate to the target epoch without recording history (use step() or
        propagate_until to keep a history).

        Args:
            target: Epoch to propagate to

        Returns:
            PropagationResult with position, velocity and acceleration, Earth-fixed
            (ITRF)
        """

    def step(self) -> StepResult:
        """
        Take one integrator step forward in time and append the new state to the
        history.

        RK45 chooses the step size from its estimate of the local error; the
        symplectic integrators use the fixed step size from the config.

        Returns:
            StepResult with the new state and step information
        """

    def step_backward(self) -> StepResult:
        """
        Take one integrator step backward in time from the earliest history entry,
        and insert the new, earlier state at the front of the history. The current
        (latest) state is unchanged.

        Returns:
            StepResult with the new (earlier) state
        """

    def rewind(self, count: int = 1) -> bool:
        """
        Go back `count` entries in the history: the trajectory returns to that
        earlier state and the later entries are discarded. Nothing is propagated.

        Args:
            count: Number of steps to go back (default 1)

        Returns:
            bool: True if successful, False if already at the initial state

        Raises:
            IndexError: count is larger than the number of steps in the history.
        """

    def epoch(self) -> Epoch:
        """Current time of the trajectory (Epoch)"""

    def state_ecef(self) -> StateVector:
        """
        Current state in Earth-fixed (ITRF) coordinates (positions in m, velocities in m/s)
        """

    def state_gcrs(self) -> StateVector:
        """Current state in the inertial GCRS, the frame the integration runs in"""

    def history_size(self) -> int:
        """Number of entries in the history, including the initial state"""

    def history_at(self, index: int) -> HistoryEntry:
        """History entry number `index` (0 = initial state)"""

    def history(self) -> list[HistoryEntry]:
        """The whole history as a list of entries, oldest first"""

    def clear_history(self) -> None:
        """
        Clear the history, keeping only the current state as the new initial entry
        """

    def reset(self, epoch: Epoch, position: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> None:
        """
        Restart from a new Earth-fixed (ITRF) state: epoch (Epoch), position [m], velocity relative to the rotating Earth [m/s]. Clears the history.
        """

    def current_step_size(self) -> float:
        """Step size the integrator will try next [s]"""

    def set_step_size_hint(self, dt: float) -> None:
        """
        Suggest the next step size dt [s]. RK45 may still adjust it to meet its error tolerance; a symplectic integrator uses it as its new fixed step.
        """

    def config(self) -> TrajectoryConfig:
        """The TrajectoryConfig in use (read-only)"""

    def is_symplectic(self) -> bool:
        """True if a symplectic (Yoshida) integrator is used, False for RK45"""

    def energy_drift(self) -> float:
        """
        Relative change in orbital energy since the start, (E - E0) / |E0|. Only tracked by the symplectic integrators; RK45 returns 0.0.
        """

    def warnings(self) -> list[str]:
        """
        Notes from propagation that do not stop it, each recorded once (list of str): drag entering forecast (not yet measured) space weather, running past a tabulated drag environment (the atmosphere is then held frozen), past the time span of an empirical acceleration, or outside the loaded Earth orientation table. The ballistic.Trajectory wrapper also issues them as warnings.
        """

class TrajectoryBatch:
    """
    TrajectoryBatch - many trajectories propagated and queried together.

    The loops over trajectories run in C++, so a batch is much faster than a
    Python loop over single Trajectory objects. Every batch operation returns a
    list with one result per trajectory.

    Example:
        import ballistic
        import numpy as np

        ballistic.initialize()

        # Create batch from initial states
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, [6778137.0, 0.0, 0.0], [0.0, 7175.0, 0.0]),
            ballistic.State.from_itrf(epoch, [6878137.0, 0.0, 0.0], [0.0, 7120.0, 0.0]),
        ]
        batch = ballistic.TrajectoryBatch(states)

        # Build history for all trajectories
        batch.propagate_all(3600.0)  # 1 hour

        # Query all at a specific time
        results = batch.get_states_at(epoch + 1800.0)
        for r in results:
            if r.valid:
                print(f"Trajectory {r.trajectory_index}: pos={r.position}")

        # Propagate all to altitude
        alt_results = batch.propagate_to_altitude(
            300_000,  # meters
            ballistic.AltitudeCondition.Below)
        for r in alt_results:
            print(f"Trajectory {r.trajectory_index}: {r.ending_altitude_m/1000:.1f} km")
    """

    def __init__(self, states: Sequence[State], config: TrajectoryConfig = ...) -> None:
        """
        Build from a list of start states (ballistic.State), each with its own epoch, and one TrajectoryConfig shared by all.
        """

    def __len__(self) -> int:
        """Number of trajectories"""

    def size(self) -> int:
        """Number of trajectories"""

    def empty(self) -> bool:
        """True if the batch holds no trajectories"""

    def trajectory(self, index: int) -> Trajectory:
        """The Trajectory at position `index`"""

    def interpolator(self, index: int) -> TrajectoryInterpolator:
        """
        The TrajectoryInterpolator of trajectory `index` (created on first use)
        """

    def propagate_all(self, duration_seconds: float) -> None:
        """
        Propagate every trajectory by duration_seconds [s] from its own current
        epoch, landing exactly there; the steps are kept in each history.
        """

    def propagate_all_to(self, target: Epoch) -> None:
        """
        Propagate every trajectory to the epoch `target`, landing exactly on it.
        """

    def step_all(self, n_steps: int = 1) -> None:
        """
        Take n_steps integrator steps on every trajectory.

        Args:
            n_steps: Number of steps to take (default 1)
        """

    def get_states_at(self, epoch: Epoch) -> list[BatchInterpolatedState]:
        """
        The state of every trajectory at one time.

        Args:
            epoch: Query time (Epoch)

        Returns:
            List of BatchInterpolatedState, one per trajectory
        """

    def get_states_at_times(self, times: Sequence[Epoch]) -> list[list[BatchInterpolatedState]]:
        """
        The state of every trajectory at several times.

        Args:
            times: List of Epoch

        Returns:
            List of lists: results[traj_idx][time_idx]
        """

    def propagate_to_altitude(self, target_altitude_m: float, condition: AltitudeCondition, direction: PropagationDirection = PropagationDirection.Forward, max_duration_s: float = 604800.0) -> list[BatchAltitudeResult]:
        """
        Propagate every trajectory until its height above the WGS84 ellipsoid
        (geodetic altitude) goes above or below a threshold; see
        TrajectoryInterpolator.propagate_to_altitude.

        Args:
            target_altitude_m: Threshold geodetic altitude [m]
            condition: AltitudeCondition.Above or AltitudeCondition.Below
            direction: PropagationDirection.Forward or Backward
            max_duration_s: How far to search per trajectory [s] (default 7 days)

        Returns:
            List of BatchAltitudeResult, one per trajectory
        """

    def config(self) -> TrajectoryConfig:
        """The shared TrajectoryConfig (read-only)"""

    def set_output_frame(self, frame: InterpolationFrame) -> None:
        """
        Choose the coordinate frame (InterpolationFrame) of states returned by every interpolator
        """

    def set_itrf_realization(self, itrf: ITRFRealization) -> None:
        """Choose the ITRF version every interpolator uses for frame conversions"""

    def set_allow_propagation(self, allow: bool) -> None:
        """
        Allow (True) or forbid (False) every interpolator to propagate to answer queries outside its history
        """

    def set_max_propagation_time(self, max_seconds: float) -> None:
        """
        Furthest beyond its history [s] that a query may propagate, for every interpolator
        """

class TrajectoryConfig:
    """
    Settings for a Trajectory: which forces to model and which integrator to use.

    The defaults are the settings measured to be best in general; the measured
    1-hour position errors (root-mean-square error, RMSE) per orbit type are in
    docs/validation_claims.md.

    Solar radiation pressure (SRP) and drag are off by default only because
    they need the satellite's mass and areas (and, for drag, space weather).
    They do matter: a start state taken from a precise orbit product does not
    already account for them. ballistic.recommended_config() turns both on for
    a named satellite. Each is switched on by setting it:
        config.srp = SrpConfig(mass_kg=2200, area_m2=38.4, cr=1.0)   # S1A
        config.drag_env = <DragEnvironment>   # make_drag_environment(...)

    Example:
        config = TrajectoryConfig()
        config.gravity_degree = 70
        config.enable_third_body = True
        traj = Trajectory(State.from_itrf(epoch, pos, vel), config)
    """

    def __init__(self) -> None: ...

    @property
    def gravity_degree(self) -> int:
        """
        Maximum degree of the EGM2008 Earth gravity model, i.e. how much detail of the gravity field is used (70 is enough for low Earth orbit, 4-8 for higher orbits; at most 360 with the embedded model, above that raises ConfigError)
        """

    @gravity_degree.setter
    def gravity_degree(self, arg: int, /) -> None: ...

    @property
    def enable_third_body(self) -> bool:
        """
        Include the gravity of the Sun, Moon and Jupiter (essential for accuracy)
        """

    @enable_third_body.setter
    def enable_third_body(self, arg: bool, /) -> None: ...

    @property
    def enable_relativistic(self) -> bool:
        """
        Include general-relativity corrections, Schwarzschild and Lense-Thirring
        """

    @enable_relativistic.setter
    def enable_relativistic(self, arg: bool, /) -> None: ...

    @property
    def enable_solid_tides(self) -> bool:
        """
        Include tides: solid-Earth tides (IERS 2010 conventions) and, once loaded, ocean tides (decimeter-level effect)
        """

    @enable_solid_tides.setter
    def enable_solid_tides(self, arg: bool, /) -> None: ...

    @property
    def ocean_tide_degree(self) -> int:
        """
        Maximum degree of the FES2004 ocean-tide model (0 = off); takes effect once the tide file is loaded (ballistic.load())
        """

    @ocean_tide_degree.setter
    def ocean_tide_degree(self, arg: int, /) -> None: ...

    @property
    def itrf(self) -> ITRFRealization:
        """
        ITRF version (ITRFRealization) used to convert between Earth-fixed and inertial coordinates
        """

    @itrf.setter
    def itrf(self, arg: ITRFRealization, /) -> None: ...

    @property
    def integrator(self) -> IntegratorType:
        """Integrator (IntegratorType): RK45 for hours, YOSHIDA6 for longer runs"""

    @integrator.setter
    def integrator(self, arg: IntegratorType, /) -> None: ...

    @property
    def rk45(self) -> Rk45Options:
        """Options for the RK45 integrator (ignored for symplectic)"""

    @rk45.setter
    def rk45(self, arg: Rk45Options, /) -> None: ...

    @property
    def symplectic(self) -> SymplecticOptions:
        """Options for the symplectic integrators"""

    @symplectic.setter
    def symplectic(self, arg: SymplecticOptions, /) -> None: ...

    @property
    def srp(self) -> SrpConfig | None:
        """
        Radiation-pressure settings (Optional[SrpConfig]; None = radiation off, the default)
        """

    @srp.setter
    def srp(self, arg: SrpConfig | None) -> None: ...

    @property
    def drag_env(self) -> DragEnvironment:
        """
        Atmospheric drag environment (DragEnvironment or None). Setting it turns drag on; None (the default) means no drag. Build one with make_drag_environment(...) or use ballistic.recommended_config().
        """

    @drag_env.setter
    def drag_env(self, arg: DragEnvironment, /) -> None: ...

    @property
    def empirical_accel(self) -> EmpiricalAcceleration:
        """
        Empirical acceleration (EA): a small correction acceleration, usually fitted to observations by an estimator, added to the modeled forces (EmpiricalAcceleration or None)
        """

    @empirical_accel.setter
    def empirical_accel(self, arg: EmpiricalAcceleration, /) -> None: ...

    @property
    def ea_reference_epoch(self) -> Epoch | None:
        """
        Epoch from which the empirical acceleration's time keys are counted (Optional[Epoch]; None = the trajectory's start; needs empirical_accel)
        """

    @ea_reference_epoch.setter
    def ea_reference_epoch(self, arg: Epoch | None) -> None: ...

    @property
    def report_forces(self) -> bool:
        """
        True: compute each force separately and return the breakdown with results (ForceBreakdown)
        """

    @report_forces.setter
    def report_forces(self, arg: bool, /) -> None: ...

    @property
    def include_disabled_forces(self) -> bool:
        """
        True: the breakdown also includes forces that are switched off (for analysis; they still do not act on the orbit)
        """

    @include_disabled_forces.setter
    def include_disabled_forces(self, arg: bool, /) -> None: ...

class TrajectoryInterpolator:
    """
    TrajectoryInterpolator - the state at any time between a Trajectory's steps.

    Fits a quintic (5th-degree) Hermite polynomial between neighboring history
    entries, using the position, velocity and acceleration stored at each. The
    interpolation error shrinks as the 6th power of the step size; it is below a
    millimeter for typical 10-30 s steps.

    Features:
    - Polynomial coefficients are cached, so repeated queries are cheap
    - A query outside the history propagates the trajectory to cover it
      (within InterpolatorConfig.max_propagation_time, default 3600 s)
    - Output in Earth-fixed (ECEF/ITRF) or inertial (GCRS) coordinates

    Example:
        # Build trajectory history
        traj = ballistic._core.Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        while traj.epoch() - epoch < 3600.0:
            traj.step()

        # Create interpolator
        interp = ballistic.TrajectoryInterpolator(traj)

        # Query at any time within range
        state = interp.get_state(epoch + 1800.0)
        if state:
            print(f"Position: {state.position}")
            print(f"Velocity: {state.velocity}")

        # Altitude-based propagation
        result = interp.propagate_to_altitude(
            300_000,  # target altitude in meters
            ballistic.AltitudeCondition.Below,
            ballistic.PropagationDirection.Forward,
            86400.0)  # search up to one day
    """

    def __init__(self, trajectory: Trajectory, config: InterpolatorConfig = ...) -> None:
        """
        Wrap a Trajectory for interpolation.

        Args:
            trajectory: Trajectory to interpolate (must have at least one history entry)
            config: Optional InterpolatorConfig

        Note:
            The interpolator holds a reference to the trajectory and keeps it alive
            as long as the interpolator exists. Out-of-range queries add steps to
            that trajectory's history.
        """

    def get_state(self, epoch: Epoch) -> InterpolatedState | None:
        """
        The interpolated state at a given time.

        Inside the history range the state is interpolated. Outside it, if automatic
        propagation is allowed (InterpolatorConfig.allow_propagation, default True)
        and the gap is within max_propagation_time, the history is extended first.

        Args:
            epoch: Query time (Epoch)

        Returns:
            InterpolatedState, or None if the time cannot be reached (propagation
            disabled or too far away)
        """

    def position(self, epoch: Epoch) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')] | None:
        """
        Interpolated position at `epoch` [m], numpy (3,), or None (see get_state)
        """

    def velocity(self, epoch: Epoch) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')] | None:
        """
        Interpolated velocity at `epoch` [m/s], numpy (3,), or None (see get_state)
        """

    def acceleration(self, epoch: Epoch) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')] | None:
        """
        Interpolated acceleration at `epoch` [m/s^2], numpy (3,), or None (see get_state)
        """

    def get_states(self, times: Sequence[Epoch]) -> list[InterpolatedState | None]:
        """
        Interpolated states at several times.

        Faster than one get_state call per time because:
        1. The trajectory is propagated once to cover the whole time range
        2. The times are visited in sorted order, reusing cached coefficients

        Args:
            times: List of Epoch

        Returns:
            List of InterpolatedState in the order given (None where a time cannot be
            reached)
        """

    def history_start(self) -> Epoch:
        """Earliest time in the history (Epoch)"""

    def history_end(self) -> Epoch:
        """Latest time in the history (Epoch)"""

    def covers(self, epoch: Epoch) -> bool:
        """True if `epoch` lies within the current history range"""

    def history_size(self) -> int:
        """Number of history entries"""

    def extend_to(self, target: Epoch) -> bool:
        """Propagate forward so the history covers the `target` Epoch"""

    def extend_backward_to(self, target: Epoch) -> bool:
        """Propagate backward so the history covers the `target` Epoch"""

    def propagate_to_altitude(self, target_altitude_m: float, condition: AltitudeCondition, direction: PropagationDirection = PropagationDirection.Forward, max_duration_s: float = 604800.0) -> PropagateToAltitudeResult:
        """
        Propagate until the height above the WGS84 ellipsoid (geodetic altitude)
        goes above or below a threshold.

        Args:
            target_altitude_m: Threshold geodetic altitude [m]
            condition: AltitudeCondition.Above or AltitudeCondition.Below
            direction: PropagationDirection.Forward or Backward
            max_duration_s: How far to search [s] (default 7 days); condition_met is
                False if the altitude is not reached within it.

        Forward, the history ends exactly at the crossing (located to 1e-4 s), and
        ground impact also stops the search. Backward stops on the first step past
        the crossing.

        Returns:
            PropagateToAltitudeResult with condition_met, steps_taken, and altitudes

        Note:
            If the starting point already satisfies the condition, returns
            immediately with steps_taken=0.
        """

    def clear_cache(self) -> None:
        """Drop the cached polynomial coefficients"""

    def invalidate(self) -> None:
        """
        Drop the cache; call this when the trajectory's history was changed by something other than this interpolator
        """

    def config(self) -> InterpolatorConfig:
        """The InterpolatorConfig in use (read-only)"""

    def set_output_frame(self, frame: InterpolationFrame) -> None:
        """
        Choose the coordinate frame of returned states (InterpolationFrame: ECEF or GCRS)
        """

    def set_itrf_realization(self, itrf: ITRFRealization) -> None:
        """Choose the ITRF version used when converting between frames"""

    def set_allow_propagation(self, allow: bool) -> None:
        """
        Allow (True) or forbid (False) propagating to answer queries outside the history
        """

    def set_max_propagation_time(self, max_seconds: float) -> None:
        """Furthest beyond the history [s] that a query may propagate"""

VERSION_MAJOR: int = 0

VERSION_MINOR: int = 1

VERSION_PATCH: int = 2

class VelocityRecordCheck:
    """
    Result of check_velocity_records: statistics of (velocity derived from the positions) minus (velocity record).
    """

    @property
    def n_compared(self) -> int:
        """Number of samples compared"""

    @property
    def median_difference_rtn(self) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
        """
        Median (derived - record) along the radial, along-track and cross-track directions, numpy (3,) [m/s]
        """

    @property
    def rms_difference(self) -> float:
        """Root-mean-square size of (derived - record) [m/s]"""

    @property
    def mean_cip_drift_along(self) -> float:
        """
        Mean along-track size of the rotation-axis drift term (cip_drift_velocity) [m/s], for comparison with the median difference
        """

    @property
    def standard_error_along(self) -> float:
        """
        Robust (median-absolute-deviation) standard error of the along-track median difference [m/s]
        """

    @property
    def verdict(self) -> VelocityRecordVerdict:
        """What to do with the records (VelocityRecordVerdict)"""

    @property
    def summary(self) -> str:
        """One-line human-readable summary (str)"""

    def __repr__(self) -> str: ...

class VelocityRecordVerdict(enum.Enum):
    """Outcome of check_velocity_records."""

    TRUE_DERIVATIVE = 0
    """
    Records match the time derivative of the positions: use them as written
    """

    OMITS_CIP_DRIFT = 1
    """
    Records leave out the rotation-axis drift term: subtract cip_drift_velocity()
    """

    INCONSISTENT = 2
    """Neither: start from velocities derived from the positions"""

class VelocitySamples:
    """
    Observed Earth-fixed (ECEF) velocities for estimate_ea_bspline: a list of (Epoch, numpy (3,) velocity relative to the rotating Earth [m/s]).
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, samples: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]) -> None: ...

    @property
    def ecef(self) -> list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]]:
        """The samples: list of (Epoch, numpy (3,)), ECEF"""

    @ecef.setter
    def ecef(self, arg: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], /) -> None: ...

class WeatherConditions:
    """
    One fixed set of space-weather values, the inputs the atmosphere model
    needs. F10.7 is the Sun's radio flux at 10.7 cm wavelength, in solar flux
    units (SFU = 1e-22 W/m^2/Hz), a measure of the solar ultraviolet that heats
    the upper atmosphere. Ap is a daily index of geomagnetic activity (disturbance
    of the Earth's magnetic field), which also heats it.

    Attributes:
        f107_daily: Daily F10.7 [SFU]
        f107_average: 81-day centered average of F10.7 [SFU]
        ap_daily: Daily Ap index
        ap_history: 7 Ap values in the order the NRLMSIS 2.1 model reads them
            (daily, then 3-hourly values going back in time)

    Factory methods:
        - quiet(): solar minimum (F10.7 = 70, Ap = 4)
        - active(): solar maximum (F10.7 = 200, Ap = 15)
        - storm(): geomagnetic storm (daily F10.7 = 250, 81-day mean 200, Ap = 100)
    """

    def __init__(self) -> None:
        """Moderate conditions (F10.7 = 150, Ap = 4)"""

    @property
    def f107_daily(self) -> float:
        """Daily F10.7 solar flux [SFU]"""

    @f107_daily.setter
    def f107_daily(self, arg: float, /) -> None: ...

    @property
    def f107_average(self) -> float:
        """81-day centered average F10.7 [SFU]"""

    @f107_average.setter
    def f107_average(self, arg: float, /) -> None: ...

    @property
    def ap_daily(self) -> float:
        """Daily Ap geomagnetic index"""

    @ap_daily.setter
    def ap_daily(self, arg: float, /) -> None: ...

    @property
    def ap_history(self) -> list[float]:
        """
        7 Ap values in NRLMSIS 2.1 order: daily, the current 3-hourly value, the three before it, then the averages of eight 3-hourly values 12-33 h and 36-57 h back
        """

    @ap_history.setter
    def ap_history(self, arg: Sequence[float], /) -> None: ...

    @staticmethod
    def quiet() -> WeatherConditions:
        """Quiet Sun (solar minimum): F10.7 = 70, Ap = 4"""

    @staticmethod
    def active() -> WeatherConditions:
        """Active Sun (solar maximum): F10.7 = 200, Ap = 15"""

    @staticmethod
    def storm() -> WeatherConditions:
        """Geomagnetic storm: daily F10.7 = 250, Ap = 100"""

def build_info() -> dict:
    """
    How this copy of the library was compiled.

    Returns:
        dict: ``with_openmp``, ``compiler`` (e.g.
        "GCC 13.3.0"), ``cpp_standard`` (23, 20, ...), ``build_type``
        ("Release" when compiled with NDEBUG, else "Debug").
    """

def check_velocity_records(epoch0: Epoch, offsets_s: Sequence[float], positions: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 3), writable=False)], velocities: Annotated[ArrayLike, dict(dtype='float64', shape=(None, 3), writable=False)], itrf: ITRFRealization = ITRFRealization.ITRF2020) -> VelocityRecordCheck:
    """
    Check whether an orbit product's Earth-fixed velocity records are the time
    derivative of its own Earth-fixed positions, which is what Trajectory needs.

    Args:
        epoch0: Epoch of the first sample
        offsets_s: sample times as seconds after epoch0 (list of float). Take
            them from the product's own time tags (e.g. datetime differences)
            rather than from subtracting Epoch values, so that rounding in the
            times cannot disturb a test this sensitive at low-orbit speeds.
        positions: numpy (N,3) Earth-fixed positions [m]
        velocities: numpy (N,3) velocity records [m/s]
        itrf: ITRF version of the coordinates

    The positions are differentiated (9-point Lagrange polynomial, interior
    samples only) and compared with the records. The returned
    VelocityRecordCheck's verdict says how to use the records: as written,
    minus cip_drift_velocity(), or not at all (derive the velocities from the
    positions instead). Use evenly spaced samples, at most 60 s apart for low
    Earth orbit.
    """

def cip_drift_velocity(epoch: Epoch, pos_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], itrf: ITRFRealization = ITRFRealization.ITRF2020) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
    """
    Small Earth-fixed velocity term caused by the slow motion of the Earth's
    rotation axis in space (the drift of the Celestial Intermediate Pole, CIP:
    precession and nutation), about 2e-5 m/s at low Earth orbit. Returns a numpy
    (3,) velocity [m/s] for the Earth-fixed position pos_ecef [m] at `epoch`.

    An orbit product that wrote its Earth-fixed velocities without this term
    needs vel_file - cip_drift_velocity(...). Check a product first with
    check_velocity_records(): ESA Sentinel EOF files and GFZ CHAMP/GRACE-FO SP3
    (standard precise-orbit format) files contain true derivatives and need no
    correction.
    """

def compute_propagation_result(predicted: Sequence[tuple[Epoch, StateVector]], truth: EOFFile, max_time_diff: float = 5.0) -> PropagationTestResult:
    """
    Error statistics of a propagated trajectory against an EOF orbit file.

    Args:
        predicted: list of (Epoch, StateVector), your propagated samples,
            Earth-fixed (ECEF).
        truth: an EOFFile (ECEF). Each predicted sample is compared with the
            nearest truth state within max_time_diff seconds.
        max_time_diff: largest time mismatch accepted for a comparison [s].

    Returns:
        PropagationTestResult (position and velocity RMSE, and the error split
        into radial, along-track and cross-track directions).
    """

def decompose_ea(ea: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], density_kg_m3: float) -> EADecomposition:
    """
    Split one EA vector into a drag-like part (along the velocity)
    and the rest (perpendicular), and compute the ballistic coefficient the
    drag part implies, from the drag equation |a| = rho * v^2 / (2 * beta).
    This is a split in the frame given, not along the satellite's own axes.
    beta stays 0 when the along-velocity part does not oppose the motion, the
    density is not positive, |v| < 1 m/s or the EA is zero. Never raises.

    Both ea and velocity must be in the same frame; for drag, use the velocity
    relative to the air (Earth-fixed, ECEF).

    Args:
        ea: EA vector, numpy (3,) [m/s^2]
        velocity: velocity vector, numpy (3,) [m/s]
        density_kg_m3: air density rho [kg/m^3]

    Returns:
        EADecomposition with beta, ea_drag, ea_residual, drag_fraction
    """

def ecef_to_geodetic(pos_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> GeodeticCoord:
    """
    Convert an Earth-fixed (ECEF) position, numpy (3,) [m], to a GeodeticCoord: geodetic latitude and longitude [rad] and altitude [m] above the WGS84 ellipsoid (exact closed form).
    """

def eop_coverage() -> object:
    """
    Time span of the loaded Earth orientation table, as a dict, or None if none
    is loaded. Dates are Modified Julian Dates (MJD) in UTC.

    Keys:
        first_mjd, last_mjd: first and last day (the last rows are IERS
            predictions, not measurements)
        measured_until_mjd: last day with a measured UT1-UTC (or None)
        hold_days: how far beyond either end the edge value is held, with a
            warning; further out the coordinate conversions raise
            DataCoverageError and a Trajectory stops
        blank_rows_skipped: empty rows after the predictions
        malformed_rows_skipped: rows that could not be read
        rows: number of rows loaded
    """

def eop_loaded() -> bool:
    """
    True while Earth orientation data (EOP) is loaded, however it was loaded.
    """

def estimate_ea(spec: EAProblemSpec) -> EASpecResult:
    """
    Estimate an empirical acceleration from an EAProblemSpec, letting the library choose the estimator: position / velocity / acceleration samples go to estimate_ea_bspline (with Gauss-Newton refinement), a MultiSensorScenario to estimate_ea_smoother. The declared noise is then checked against the fit (result.audit_warnings). Returns an EASpecResult. A bad spec (e.g. no noise_sigma) is reported in error_message, not raised; errors from the estimator itself (e.g. missing data files) are raised.
    """

@overload
def estimate_ea_bspline(epoch: Epoch, position: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], observations: PositionSamples, force_model: TrajectoryConfig, config: EABSplineConfig = ...) -> EAResult:
    """
    Estimate an empirical acceleration from observed positions (PositionSamples). epoch / position / velocity: the start state, Earth-fixed (ECEF) [m, m/s]; force_model: TrajectoryConfig; config: EABSplineConfig. Gauss-Newton refinement (config.gn) is on by default and greatly improves position fits; keep it. Returns an EAResult whose EA is in the inertial GCRS frame.
    """

@overload
def estimate_ea_bspline(epoch: Epoch, position: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], observations: VelocitySamples, force_model: TrajectoryConfig, config: EABSplineConfig = ...) -> EAResult:
    """
    Estimate an empirical acceleration from observed velocities (VelocitySamples). Arguments as for the position form. Gauss-Newton refinement (config.gn, on by default) also helps velocity fits. Returns an EAResult (EA in GCRS).
    """

@overload
def estimate_ea_bspline(epoch: Epoch, position: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], observations: AccelSamples, force_model: TrajectoryConfig, config: EABSplineConfig = ...) -> EAResult:
    """
    Estimate an empirical acceleration from observed accelerations (AccelSamples). Arguments as for the position form. Gauss-Newton refinement does not apply (no positions to fit); the result reports gn_status = SkippedAccelerationCircular. Returns an EAResult (EA in GCRS).
    """

def estimate_ea_smoother(scenario: MultiSensorScenario, mask: SensorMask, fm_cfg: TrajectoryConfig, cfg: EASmootherConfig = ...) -> EASmootherResult:
    """
    Estimate an empirical acceleration from several sensors with the Gauss-Markov Rauch-Tung-Striebel smoother; the recommended estimator for sensor fusion (radars, optical trackers, position fixes, accelerometers with their bias), for radar alone and for noisy positions.

    Args: scenario (MultiSensorScenario); mask (SensorMask: which sensors to use); fm_cfg (TrajectoryConfig: the force model); cfg (EASmootherConfig).

    Returns an EASmootherResult with the EA in the inertial GCRS frame, its 1σ uncertainty and NIS. The EA's size and correlation time are chosen from the data by NIS, and the range-rate weight is measured from the data. Measured on the single-radar reentry-glide test scenario (with a position fix at impact): 12.4 m position error after re-propagation, better and much faster than the batch fits used before.

    Raises what Trajectory raises for an implausible start state (InputPlausibilityError / FrameError), and errors from the force model (e.g. missing data files).
    """

def extend_density_below(env: DragEnvironment, floor_km: float = 80.0) -> DragEnvironment:
    """
    Use the U.S. Standard Atmosphere 1976 for the density below `floor_km`
    [km] (0 < floor_km <= 86; the standard is computed exactly from its
    defining constants in that range), for deep reentry or sub-orbital flight.
    Returns a new DragEnvironment; beta and wind are unchanged (a live
    environment keeps HWM14 wind below the floor).

    A live environment (make_drag_environment) already has NRLMSIS 2.1 down to
    the ground, so for it this only chooses the standard atmosphere below the
    floor instead. On an altitude or altitude-time table it adds table points
    every 2 km from 0 km up to the floor (the standard 100-1500 km grid has no
    points there to replace).

    Raises:
        ballistic.errors.ConfigError: a latitude/longitude/time (4D) table, or
            floor_km out of range.
    """

def find_and_parse_eof(directory: str, pattern: str = '') -> EOFFile | None:
    """
    Find and read the first .EOF (or .EOF.zip) file in `directory` whose name contains `pattern` (empty = any; files already unpacked first, then .EOF.zip); None if nothing matches or parses.
    """

def free_molecular_cd(shape: AeroShape, speed_ratio: float, reemission_ratio: float) -> float:
    """
    Drag coefficient in thin air (free-molecular flow, Sentman's formulas) of an AeroShape. speed_ratio: body speed divided by the most probable molecular speed, V / sqrt(2kT/m) (dimensionless). reemission_ratio: speed of molecules leaving the surface over their arrival speed (see reemission_ratio()).
    """

def from_accelerometer_density(sample_epochs: Sequence[Epoch], measured_density_kg_m3: Sequence[float], beta_prior: BetaProfile1D, altitude_grid_km: Sequence[float] = []) -> DragEnvironment:
    """
    Build a DragEnvironment from measured air densities (derived from a
    satellite's on-board accelerometer). At each time the same measured density
    is used at every altitude, which is appropriate when the satellite stays
    near one altitude over the arc (typical use: the GOCE satellite's final
    descent, with TU Delft accelerometer densities).

    Wind is empty (no HWM14). For altitude/lat/lon variation use
    from_models_along or from_models_4d instead.

    Args:
        sample_epochs: list of Epoch (length N).
        measured_density_kg_m3: list of float, ρ at each sample (length N,
            all values must be > 0).
        beta_prior: BetaProfile1D (satellite β = m/(Cd·A)).
        altitude_grid_km: optional altitude grid (defaults to
            standard_leo_altitude_grid_km, 100-1500 km / 25 km steps).
            Pass a tighter grid (e.g. 100-300 km for GOCE reentry) for
            storage efficiency.

    Returns:
        DragEnvironment with DensityProfile2D + empty WindProfile1D_ENU +
        the provided BetaProfile1D.
    """

def from_models_4d(altitude_grid_km: Sequence[float], latitude_grid_deg: Sequence[float], longitude_grid_deg: Sequence[float], epoch_grid: Sequence[Epoch], sw: SpaceWeatherHistory, beta_prior: BetaProfile1D) -> DragEnvironment:
    """
    Build a DragEnvironment tabulated over altitude [km], latitude and longitude [degrees] and time (lists of Epoch), e.g. one shared atmosphere for many satellites. sw: SpaceWeatherHistory; beta_prior: BetaProfile1D. Expensive: evaluated in parallel over the time axis (OpenMP).
    """

def from_models_along(sample_epochs: Sequence[Epoch], sample_positions_ecef: Sequence[Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]], sw: SpaceWeatherHistory, beta_prior: BetaProfile1D, altitude_grid_km: Sequence[float] = []) -> tuple[DragEnvironment, IterationDiag]:
    """
    Build a DragEnvironment tabulated over altitude and time by evaluating NRLMSIS 2.1 / HWM14 below each point of a given trajectory. sample_epochs: list of Epoch; sample_positions_ecef: matching Earth-fixed positions [m]; sw: SpaceWeatherHistory; beta_prior: BetaProfile1D; altitude_grid_km: altitudes [km] (empty = standard_leo_altitude_grid_km()). Returns (env, IterationDiag).
    """

def from_models_snapshot(t: Epoch, ref_lat_deg: float, ref_lon_deg: float, sw: SpaceWeatherHistory, beta_prior: BetaProfile1D, altitude_grid_km: Sequence[float] = []) -> DragEnvironment:
    """
    Build a DragEnvironment whose density and wind vary only with altitude, evaluated once at the time t (Epoch) and the point ref_lat_deg / ref_lon_deg [degrees]. sw: SpaceWeatherHistory; beta_prior: BetaProfile1D; altitude_grid_km: altitudes [km] (empty = standard_leo_altitude_grid_km()). Suits arcs much shorter than one orbit.
    """

def gcrs_to_itrf(pos_gcrs: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], vel_gcrs: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], epoch: Epoch, itrf: ITRFRealization = ITRFRealization.ITRF2020) -> StateVector:
    """
    Convert an inertial (GCRS) state to Earth-fixed (ITRF / ECEF) coordinates
    with the same IAU 2006/2000A transform the propagator uses (precession and
    nutation of the Earth's axis including its drift, Earth rotation, polar
    motion).

    Args:
        pos_gcrs: numpy (3,) position [m]
        vel_gcrs: numpy (3,) velocity [m/s]
        epoch: time of the state (Epoch)
        itrf: ITRF version of the result

    Returns a StateVector (position [m], velocity relative to the rotating Earth
    [m/s]). Uses the loaded Earth orientation parameters (EOP: polar motion and
    UT1-UTC); raises ballistic.errors.DataCoverageError when none are loaded
    (unless ballistic.set_eop_policy('assume_zero')) or the date is more than 30
    days outside the loaded table. A J2000 (EME2000) state can be passed as GCRS if a 0.7 m error at
    low Earth orbit is acceptable: the frame bias between them is about 23
    milliarcseconds; State.from_j2000 applies it exactly.
    """

def geodetic_to_ecef(latitude: float, longitude: float, altitude: float) -> Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]:
    """
    Convert geodetic latitude and longitude [rad] and altitude [m] above the WGS84 ellipsoid to an Earth-fixed (ECEF) position, numpy (3,) [m].
    """

def get_data_directory() -> str:
    r"""
    The default directory where Ballistic looks for, and downloads, data files.

    Returns:
        str: Path to the default data directory.

    Platform-specific locations:
        Windows: %LOCALAPPDATA%\ballistic  (C:\Users\<user>\AppData\Local\ballistic)
        Linux:   ~/.local/share/ballistic
        macOS:   ~/Library/Application Support/ballistic
        The environment variable BALLISTIC_DATA_DIR, when set, names the
        directory instead (every platform).

    Note:
        The directory may not exist yet; create it before saving files there.
    """

def get_init_status() -> InitResult:
    """
    Report what data is currently loaded.

    Returns:
        InitResult: current status of loaded data.
    """

def get_leap_second_info() -> LeapSecondInfo:
    """
    Information about the leap-second table in use.

    Returns:
        LeapSecondInfo: current state of the leap-second data.

    Example:
        info = ballistic.get_leap_second_info()
        print(f"Current TAI-UTC: {info.current_tai_utc} seconds")
        print(f"Entries: {info.entry_count}")
        print(f"From file: {info.loaded_from_file}")
    """

def get_space_weather(epoch: Epoch, allow_predicted: bool = True) -> WeatherConditions:
    """
    Space weather (F10.7 and Ap) at a given time, from the loaded file.

    Args:
        epoch: Time to query (Epoch).
        allow_predicted: accept CelesTrak's forecast past the last measured day
            (default True).

    Returns:
        WeatherConditions: F10.7 and Ap values for the given time.

    Past the last measured day returns CelesTrak's daily forecast with a
    RuntimeWarning (allow_predicted=False raises instead). Raises
    DataCoverageError when nothing is loaded or the epoch is outside the file;
    there is no fallback to quiet defaults or the nearest row.
    """

def henderson1976_sphere_cd(mach: float, reynolds: float, wall_to_gas_temperature: float) -> float:
    """
    Sphere drag coefficient from Henderson's (1976) full correlation, valid from dense to thin air, as given in Singh et al. 2020, eq. B1. mach: Mach number; reynolds: Reynolds number on the diameter; wall_to_gas_temperature: wall / air temperature ratio. A reference for checking FlowRegimeCd, not used by it.
    """

def information_bound(scenario: MultiSensorScenario, mask: SensorMask, fm_cfg: TrajectoryConfig, opts: EAInformationBoundOptions = ...) -> EAInformationBound:
    """
    The best position accuracy (after re-propagation) that any estimator could reach with the sensors of `scenario` selected by `mask`, propagating with `fm_cfg`; computed without running an estimator. Use it to design a sensor suite before tuning an estimator: e.g. on the single-radar reentry-glide test scenario, adding a 50 µrad camera moves the limit from about 10-17 m to about 1.3 m (test_ea_problem_spec). Returns an EAInformationBound; failures (no radar, fix or camera samples; a failed propagation) are reported in its error_message.
    """

def initialize() -> InitResult:
    """
    Initialize the library with the built-in data plus any data files found in
    the data directory (get_data_directory()).

    Returns:
        InitResult: full status (True in a boolean context on success, so
        `if ballistic.initialize():` works). Check `.eop_loaded`: the Earth
        orientation file (EOP) is only picked up if it has been downloaded.
        Without it, converting between Earth-fixed and inertial coordinates
        raises DataCoverageError, unless set_eop_policy('assume_zero') accepts
        a 3-10 m error at low Earth orbit.

    Note:
        The EGM2008 gravity coefficients (degrees 2-360) and the DE440 planetary
        ephemeris (years 1950-2100) are built in. The only file needed for
        sub-meter accuracy is finals2000A.data (EOP): run ballistic.load() or
        ballistic.Loader().load_eop() once to put it in the data directory.
    """

def initialize_with_config(config: InitConfig) -> InitResult:
    """
    Initialize the library with explicit data-file paths.

    Args:
        config: InitConfig with the file paths. Explicit paths override the
            files found automatically in the data directory.

    Returns:
        InitResult: what was loaded. success is False (with error_message) if a
        named file could not be loaded.

    Example:
        config = ballistic.InitConfig()
        config.eop_path = "finals2000A.data"
        config.jpl_ephemeris_path = "de421.bsp"
        result = ballistic.initialize_with_config(config)

        if result.success:
            print(f"Loaded: {result.jpl_de_version}")
            print(f"Gravity degree: {result.gravity_degree_loaded}")
    """

def is_initialized() -> bool:
    """
    Check whether any data has been loaded.

    Returns:
        bool: True if an external gravity or leap-second file, an EOP table, or a
        planetary ephemeris (the built-in DE440 included) is loaded.
    """

def itrf_to_gcrs(pos_itrf: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], vel_itrf: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], epoch: Epoch, itrf: ITRFRealization = ITRFRealization.ITRF2020) -> StateVector:
    """
    Convert an Earth-fixed (ITRF / ECEF) state to inertial (GCRS) coordinates;
    the inverse of gcrs_to_itrf. pos_itrf [m] and vel_itrf (relative to the
    rotating Earth) [m/s] are numpy (3,); returns a StateVector.
    """

def itrf_to_teme(epoch: Epoch, position_m: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity_m_s: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> tuple:
    """
    Convert an Earth-fixed ITRF state (position [m], velocity relative to the rotating Earth [m/s]) to TEME; the inverse of teme_to_itrf.
    """

def load_eop(path: str) -> bool:
    """
    Load Earth orientation parameters (EOP: polar motion and UT1-UTC) from an
    IERS finals2000A.data (or .all) file. Replaces any table already loaded.

    Args:
        path: Path to the file (str).

    Returns:
        bool: True if loading succeeded.

    Note:
        Download from: https://datacenter.iers.org/data/9/finals2000A.data
    """

def load_jpl_ephemeris(path: str) -> bool:
    """
    Load a JPL planetary ephemeris file (de421.bsp, de440.bsp, etc.), which
    gives the Sun, Moon and planet positions; replaces the built-in DE440.

    Args:
        path: Path to a .bsp (SPICE SPK format, recommended) or .bin (older
            format) file (str).

    Returns:
        bool: True if loading succeeded.

    Note:
        Download from: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
    """

def load_leap_seconds(path: str, verbose: bool = True) -> tuple:
    """
    Load the IERS leap-second table from a leap-seconds.list file.

    Args:
        path: Path to the leap-seconds.list file (str).
        verbose: If True, the error message on failure includes the download URL.

    Returns:
        Tuple[bool, str]: (success, error_message). error_message is empty on success.

    Note:
        Download from: https://hpiers.obspm.fr/iers/bul/bulc/ntp/leap-seconds.list
        Alternative: https://www.ietf.org/timezones/data/leap-seconds.list

        The embedded leap seconds are current through 2017 (TAI-UTC=37s).
        No new leap seconds have been announced since then.
    """

def load_ocean_tides(path: str, max_degree: int = 100) -> bool:
    """
    Load an ocean-tide model: the tide-induced changes of the gravity field's
    coefficients, in the IERS Conventions chapter 6 format (e.g. the FES2004
    file fes2004_Cnm-Snm.dat). Once loaded, every Trajectory adds ocean tides up
    to TrajectoryConfig.ocean_tide_degree (default 50). ballistic.load() does
    this for you.

    Args:
        path: Path to the file (str).
        max_degree: highest degree to read (default 100).

    Returns:
        bool: True if loading succeeded (False if the file cannot be opened).
    """

def load_space_weather(path: str) -> bool:
    """
    Load a CelesTrak space-weather file: daily F10.7 (the Sun's 10.7 cm radio
    flux, a measure of the ultraviolet that heats the upper atmosphere) and Ap
    (geomagnetic activity) indices. Drag needs these.

    Args:
        path: Path to SW-All.csv or SW-Last5Years.csv (str).

    Returns:
        bool: True if loading succeeded. Non-fatal problems are listed by
        space_weather_warnings().

    Raises:
        ballistic.errors.BallisticError: the file lacks a column drag cannot do without
            (observed F10.7 or daily Ap).

    Note:
        Download from: https://celestrak.org/SpaceData/SW-All.csv
    """

def make_drag_environment(t0: Epoch, pos_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], vel_ecef: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], duration_s: float, base_config: TrajectoryConfig, beta: BetaProfile1D | AeroBeta, weather: SpaceWeatherHistory | None = None, altitude_grid_km: Sequence[float] = [], max_iter: int = 3, tol_relative: float = 0.01, sample_dt_s: float = 60.0, allow_predicted_weather: bool = True) -> tuple[DragEnvironment, IterationDiag]:
    """
    Turn on drag with one call (low-level form taking the C++ TrajectoryConfig;
    prefer ballistic.make_drag_environment, which also accepts the Python
    TrajectoryConfig dataclass).

    Returns (drag_env, IterationDiag). The environment is "live": NRLMSIS 2.1
    (density) and HWM14 (wind) are evaluated at the satellite's exact position
    on every force evaluation, for any time covered by the loaded space-weather
    file.

    Args:
        t0: start Epoch; checked to have space weather
        beta: BetaProfile1D or AeroBeta (the satellite's ballistic coefficient)
        weather: None (default) uses the measured F10.7 / Ap loaded by
            ballistic.load(); pass SpaceWeatherHistory.from_constant(...) to
            override.
        allow_predicted_weather: past the last measured day, use CelesTrak's
            forecast (default True; a Trajectory warns there). False refuses.
        pos_ecef, vel_ecef, duration_s, base_config, altitude_grid_km,
        max_iter, tol_relative, sample_dt_s: accepted for compatibility and
            not used (a live environment needs no table).

    Raises:
        RuntimeError: weather=None and no space weather is loaded.
        ballistic.errors.DataCoverageError: t0 has no space weather. A
            Trajectory using the environment also stops with this error at the
            end of the weather (keeping its state and history up to there).
    """

def ocean_tides_loaded() -> bool:
    """True once an ocean-tide model is loaded."""

def parse_eof_auto(path: str) -> EOFFile | None:
    """Read an EOF file, unpacking a .EOF.zip first if needed."""

def reemission_ratio(speed_m_s: float, molar_mass_kg_kmol: float, accommodation: float, wall_temperature_k: float) -> float:
    """
    Speed of molecules re-emitted from a surface over their arrival speed (Sentman 1961; Moe & Moe 2005), for a body moving at speed_m_s [m/s] through air of mean molar mass molar_mass_kg_kmol [kg/kmol], energy accommodation coefficient accommodation (0 to 1) and wall temperature wall_temperature_k [K].
    """

def refine_beta_from_ea(prior: DragEnvironment, ea: EmpiricalAcceleration, trajectory: Sequence[HistoryEntry], epoch0: Epoch, config: BetaDecompositionConfig = ...) -> DragEnvironment:
    """
    Estimate a satellite's ballistic coefficient β from a fitted empirical
    acceleration, and return a DragEnvironment that uses it.

    The EA's component along the velocity is read as missing (or extra) drag;
    together with the prior environment's density it gives a refined β profile.
    The returned DragEnvironment keeps the prior's density and wind and replaces
    only its β column.

    The trajectory must be the orbit the EA was applied to, i.e. propagated
    with the EA in the force model, so that its velocities match the flight the
    EA describes. Every EAResult carries such an orbit as collocation_history;
    pass that as `trajectory`.

    Args:
        prior: DragEnvironment with density and wind filled in. Its β column is
            replaced in the returned environment.
        ea: EmpiricalAcceleration to split into drag (β) and remainder.
        trajectory: list of HistoryEntry (typically result.collocation_history).
        epoch0: start Epoch of the arc.
        config: optional BetaDecompositionConfig (β limits, altitude bins, ...).

    Returns:
        DragEnvironment with the refined BetaProfile1D installed.
    """

def refine_ea_gn(epoch: Epoch, position: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], position_data: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], initial_ea: EmpiricalAcceleration, force_model: TrajectoryConfig, gn: GNRefine = ..., max_duration_s: float = 3600.0) -> EmpiricalAcceleration:
    """
    Refine an empirical acceleration by Gauss-Newton so that the
    orbit propagated with it matches observed positions.

    Args:
        epoch: start Epoch
        position, velocity: start state, Earth-fixed (ECEF), numpy (3,)
            [m] and [m/s]
        position_data: observed positions, list of (Epoch, numpy (3,) ECEF
            position [m])
        initial_ea: EmpiricalAcceleration to start from
        force_model: TrajectoryConfig for the propagations
        gn: GNRefine settings (default: STM derivatives)
        max_duration_s: longest arc used [s]

    Returns:
        Refined EmpiricalAcceleration (a table on the initial EA's time points,
        in its frame). The initial EA is returned unchanged if it is None or
        zero, or if fewer than 3 observations remain after sample_stride and
        max_duration_s.
    """

def reject_heavy_tails(samples: Sequence[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], sigma: float, outlier_prob: float = 0.05, outlier_scale: float = 10.0) -> tuple[list[tuple[Epoch, Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]]], int, bool]:
    """
    Remove outliers from dense samples before fitting.

    samples: list of (Epoch, numpy (3,)); sigma: the usual noise σ in the data's units; outlier_prob p and outlier_scale k: a fraction p of samples are assumed to have noise k·σ. Each sample is compared with the median of the samples within ±0.25 s and dropped when any component differs by more than the point at which it is more likely an outlier than not (about 3.3σ for the defaults, and nearly independent of p and k, so a rough guess is fine).

    Returns (filtered samples, number dropped, out_of_envelope). out_of_envelope = True means there are fewer than about 4 samples per second, too few to judge; the input is then returned unchanged. Outliers are dropped, not clipped: pulling them toward the local median was measured to corrupt position data (a 10.3 m error). Never raises.
    """

def set_eop_policy(policy: str) -> None:
    """
    Choose what happens when no Earth orientation (EOP) table is loaded.
    Applies to the whole process.

    'require' (default): converting between Earth-fixed and inertial coordinates
    raises DataCoverageError saying how to load a table (ballistic.load()).
    'assume_zero': UT1-UTC and polar motion are taken as zero (3-10 m error at
    low Earth orbit), with one DataCoverageWarning.

    Raises:
        ballistic.errors.ConfigError: policy is neither 'require' nor
            'assume_zero'.
    """

def sp3_classify_frame(label: str) -> SP3FrameInfo:
    """
    What an SP3 coordinate-system label means (SP3FrameInfo). Inertial: J2000, J2K, EME00. Earth-fixed labels map to an ITRF version (IGS05, IGb08, IGS14, ITR20, ...); 'CTS', 'ITRF' and 'ITRF2' are Earth-fixed without a stated version. Raises FrameError for any other label rather than guessing.
    """

def space_weather_loaded() -> bool:
    """
    True while space weather (F10.7 / Ap) is loaded, however it was loaded.
    """

def space_weather_warnings() -> list[str]:
    """
    Non-fatal problems found by the last space-weather load (missing columns that drag depends on, values that could not be read), as a list of str. Empty = clean.
    """

def standard_leo_altitude_grid_km() -> list[float]:
    """
    Default altitude grid for low Earth orbit tables: 100-1500 km in 25 km steps (list of float, km).
    """

def teme_to_itrf(epoch: Epoch, position_m: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')], velocity_m_s: Annotated[ArrayLike, dict(dtype='float64', shape=(3), order='C')]) -> tuple:
    """
    Convert a TEME state (position [m], velocity [m/s]) to the Earth-fixed ITRF (velocity relative to the rotating Earth): rotation by the 1982 Greenwich mean sidereal time of UT1, then polar motion, both from the loaded Earth orientation data (EOP). Returns (position, velocity).
    """

def tle_checksum(line: str) -> int:
    """
    The TLE checksum of a line: digits summed, each '-' counting 1, over columns 1-68, modulo 10.
    """

def tle_grade_preset(state: State, horizon_s: float) -> TleGradePreset:
    """
    The TLE-grade preset (TleGradePreset) for this start State and prediction horizon [s]. Raises ConfigError outside what was measured (perigee below 300 km, geostationary-transfer or other highly elliptical orbits, medium orbits below 5000 km, horizons over 7 days).
    """

def unload_ocean_tides() -> None:
    """
    Unload the ocean-tide model (propagation then runs without ocean tides).
    """
