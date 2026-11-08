"""
Unit tests for the Ballistic Python bindings.

Run with: pytest tests/python/test_ballistic.py -v
"""

import pytest
import numpy as np


# Skip all tests if the module isn't built yet
try:
    import ballistic
    BALLISTIC_AVAILABLE = True
except ImportError:
    BALLISTIC_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not BALLISTIC_AVAILABLE,
    reason="ballistic module not available - build with BALLISTIC_BUILD_PYTHON=ON"
)


# Session-scoped initialization is handled by conftest.py


class TestInitialization:
    """Tests for library initialization."""

    def test_initialize(self):
        """Test basic initialization."""
        result = ballistic.initialize()
        # initialize() now returns the full InitResult (truthy on success)
        assert result
        assert result.success is True

    def test_is_initialized(self):
        """Test initialization status check."""
        ballistic.initialize()
        # is_initialized checks if external data is loaded
        # With embedded ephemeris only, this returns True
        _ = ballistic.is_initialized()  # Just check it doesn't crash

    def test_version(self):
        """Test version information."""
        assert isinstance(ballistic.__version__, str)
        assert ballistic.VERSION_MAJOR >= 0
        assert ballistic.VERSION_MINOR >= 0
        assert ballistic.VERSION_PATCH >= 0


class TestEpoch:
    """Tests for Epoch class."""

    def test_from_utc(self):
        """Test creating epoch from UTC."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        assert epoch.year == 2024

    def test_from_jd(self):
        """Test creating epoch from Julian Date."""
        jd = 2460325.0
        epoch = ballistic.Epoch.from_jd(jd, ballistic.TimeScale.TT)
        assert abs(epoch.tt_jd - jd) < 1e-10

    def test_from_unix(self):
        """Test creating epoch from Unix timestamp."""
        unix_ts = 1705320000.0
        epoch = ballistic.Epoch.from_unix(unix_ts)
        assert abs(epoch.unix_timestamp - unix_ts) < 1.0  # Allow leap second offset

    def test_j2000(self):
        """Test J2000.0 epoch."""
        j2000 = ballistic.Epoch.j2000()
        assert abs(j2000.tt_jd - 2451545.0) < 1e-10

    def test_arithmetic(self):
        """Test epoch arithmetic."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        epoch2 = epoch + 3600.0  # Add 1 hour
        dt = epoch2 - epoch
        # Allow ~1 microsecond tolerance due to Julian Date floating-point precision
        assert abs(dt - 3600.0) < 1e-4

    def test_comparison(self):
        """Test epoch comparison operators."""
        epoch1 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        epoch2 = epoch1 + 1.0
        assert epoch1 < epoch2
        assert epoch2 > epoch1
        assert epoch1 <= epoch2
        assert epoch2 >= epoch1
        assert epoch1 != epoch2
        assert epoch1 == epoch1

    def test_time_scales(self):
        """Test time scale conversions."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        # Just verify these don't crash and return reasonable values
        assert epoch.utc_jd > 2400000
        assert epoch.tt_jd > 2400000
        assert epoch.tai_jd > 2400000
        # TT - UTC should be ~37 leap seconds + 32.184s ≈ 69s
        diff = (epoch.tt_jd - epoch.utc_jd) * 86400
        assert 60 < diff < 80

    def test_calendar_conversion(self):
        """Test calendar conversion."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 30, 45.5)
        year, month, day, hour, minute, second = epoch.to_calendar()
        assert year == 2024
        assert month == 1
        assert day == 15
        assert hour == 12
        assert minute == 30
        assert abs(second - 45.5) < 0.01

    def test_repr(self):
        """Test string representation."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        s = repr(epoch)
        assert "2024" in s
        assert "01" in s
        assert "15" in s


class TestConfiguration:
    """Tests for configuration classes (C++ core versions)."""

    def test_trajectory_config_defaults(self):
        """Test TrajectoryConfig default values (C++ version)."""
        config = ballistic._core.TrajectoryConfig()
        assert config.gravity_degree == 70
        assert config.enable_third_body is True
        assert config.srp is None
        assert config.enable_relativistic is True
        assert config.integrator == ballistic.IntegratorType.RK45
        # Drag is opt-in via drag_env (None == no drag).
        assert config.drag_env is None

    def test_trajectory_config_modification(self):
        """Test modifying TrajectoryConfig (C++ version)."""
        config = ballistic._core.TrajectoryConfig()
        config.gravity_degree = 120
        config.integrator = ballistic.IntegratorType.YOSHIDA6
        assert config.gravity_degree == 120
        assert config.integrator == ballistic.IntegratorType.YOSHIDA6

    def test_interpolator_config(self):
        """Test InterpolatorConfig (C++ version)."""
        config = ballistic._core.InterpolatorConfig()
        assert config.output_frame == ballistic.InterpolationFrame.ECEF
        assert config.allow_propagation is True

    def test_ballistic_model(self):
        """Test BallisticModel."""
        # beta = m / (Cd * A) = 1000 / (2.2 * 10) = 45.45...
        model = ballistic.drag.BallisticModel(45.454545, 1000.0, 10.0)
        assert model.mass == 1000.0
        assert model.drag_area == 10.0
        assert abs(model.beta - 45.454545) < 0.01
        assert abs(model.ballistic_coefficient() - 45.454545) < 0.01

    def test_weather_conditions(self):
        """Test WeatherConditions."""
        quiet = ballistic.drag.WeatherConditions.quiet()
        assert quiet.f107_daily == 70.0
        assert quiet.ap_daily == 4.0

        active = ballistic.drag.WeatherConditions.active()
        assert active.f107_daily == 200.0

        storm = ballistic.drag.WeatherConditions.storm()
        assert storm.f107_daily == 250.0
        assert storm.ap_daily == 100.0


class TestTrajectory:
    """Tests for Trajectory class (C++ core version)."""

    @pytest.fixture
    def initial_state(self, leo_state):
        """Create initial state for tests (delegates to shared fixture)."""
        return leo_state

    def test_construction(self, initial_state):
        """Test Trajectory construction."""
        epoch, pos, vel = initial_state
        traj = ballistic._core.Trajectory(epoch, pos, vel)
        assert traj.history_size() == 0  # History starts empty

    def test_step(self, initial_state):
        """Test single step."""
        epoch, pos, vel = initial_state
        traj = ballistic._core.Trajectory(epoch, pos, vel)
        result = traj.step()
        assert traj.history_size() == 2  # Initial + 1 step
        assert result.step_size_taken > 0
        assert np.linalg.norm(result.position) > 0

    def test_propagate(self, initial_state):
        """Test propagate method."""
        epoch, pos, vel = initial_state
        traj = ballistic._core.Trajectory(epoch, pos, vel)
        result = traj.propagate(epoch + 60.0)  # 1 minute
        assert np.linalg.norm(result.position) > 6000000  # Still in orbit
        assert np.linalg.norm(result.velocity) > 7000  # Still moving

    def test_history_access(self, initial_state):
        """Test history access."""
        epoch, pos, vel = initial_state
        traj = ballistic._core.Trajectory(epoch, pos, vel)

        # Take several steps
        for _ in range(5):
            traj.step()

        assert traj.history_size() == 6  # Initial + 5 steps
        entry = traj.history_at(0)
        assert np.allclose(entry.position, pos, rtol=1e-10)

    def test_rewind(self, initial_state):
        """Test stepping back in history."""
        epoch, pos, vel = initial_state
        traj = ballistic._core.Trajectory(epoch, pos, vel)

        for _ in range(5):
            traj.step()

        size_before = traj.history_size()
        success = traj.rewind(2)
        assert success
        assert traj.history_size() == size_before - 2

    def test_reset(self, initial_state):
        """Test reset."""
        epoch, pos, vel = initial_state
        traj = ballistic._core.Trajectory(epoch, pos, vel)

        for _ in range(5):
            traj.step()

        new_pos = np.array([7000000.0, 0.0, 0.0])
        new_vel = np.array([0.0, 7200.0, 0.0])
        traj.reset(epoch, new_pos, new_vel)

        state = traj.state_ecef()
        assert np.allclose(state.position, new_pos)
        assert np.allclose(state.velocity, new_vel)

    def test_symplectic_integrator(self, initial_state):
        """Test symplectic integrator."""
        epoch, pos, vel = initial_state
        config = ballistic._core.TrajectoryConfig()
        config.integrator = ballistic.IntegratorType.YOSHIDA6
        config.symplectic.step_size = 30.0

        traj = ballistic._core.Trajectory(epoch, pos, vel, config)
        assert traj.is_symplectic()

        # Take some steps
        for _ in range(10):
            traj.step()

        # Energy drift should be bounded (symplectic integrators have bounded energy error)
        drift = traj.energy_drift()
        assert abs(drift) < 1e-3  # Bounded energy drift for symplectic integrators


class TestTrajectoryInterpolator:
    """Tests for TrajectoryInterpolator class (C++ core version)."""

    @pytest.fixture
    def trajectory_with_history(self):
        """Create a trajectory with history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = ballistic._core.Trajectory(epoch, pos, vel)

        # Build 1 hour of history
        while traj.epoch() - epoch < 3600.0:
            traj.step()

        return traj, epoch

    def test_construction(self, trajectory_with_history):
        """Test interpolator construction."""
        traj, epoch = trajectory_with_history
        interp = ballistic._core.TrajectoryInterpolator(traj)
        assert interp.history_size() == traj.history_size()

    def test_get_state(self, trajectory_with_history):
        """Test state interpolation."""
        traj, epoch = trajectory_with_history
        interp = ballistic._core.TrajectoryInterpolator(traj)

        # Query at middle of history
        state = interp.get_state(epoch + 1800.0)
        assert state is not None
        assert np.linalg.norm(state.position) > 6000000

    def test_covers(self, trajectory_with_history):
        """Test covers method."""
        traj, epoch = trajectory_with_history
        interp = ballistic._core.TrajectoryInterpolator(traj)

        assert interp.covers(epoch + 1800.0)
        assert not interp.covers(epoch - 100.0)

    def test_propagate_to_altitude(self, trajectory_with_history):
        """Test altitude-based propagation."""
        traj, epoch = trajectory_with_history
        interp = ballistic._core.TrajectoryInterpolator(traj)

        # The satellite is at ~400km altitude
        # Propagate until below 350km (might not happen in 1 hour)
        result = interp.propagate_to_altitude(
            350000.0,
            ballistic.AltitudeCondition.Below,
            ballistic.PropagationDirection.Forward,
            600.0  # search 10 minutes
        )

        # Result should be valid (condition may or may not be met)
        assert result.starting_altitude_m > 0
        assert result.ending_altitude_m > 0


class TestTrajectoryBatch:
    """Tests for TrajectoryBatch class."""

    @pytest.fixture
    def batch_states(self):
        """Create initial states for batch tests."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6978137.0, 0.0, 0.0]), np.array([0.0, 7300.0, 0.0])),
        ]
        return states, epoch

    def test_construction(self, batch_states):
        """Test batch construction."""
        states, _ = batch_states
        batch = ballistic._core.TrajectoryBatch(states)
        assert len(batch) == 3
        assert batch.size() == 3
        assert not batch.empty()

    def test_propagate_all(self, batch_states):
        """Test propagating all trajectories."""
        states, _ = batch_states
        batch = ballistic._core.TrajectoryBatch(states)
        batch.propagate_all(600.0)  # 10 minutes

        # All trajectories should have history now
        for i in range(len(batch)):
            traj = batch.trajectory(i)
            assert traj.history_size() > 1

    def test_get_states_at(self, batch_states):
        """Test querying all trajectories at single time."""
        states, epoch = batch_states
        batch = ballistic._core.TrajectoryBatch(states)
        batch.propagate_all(600.0)

        results = batch.get_states_at(epoch + 300.0)
        assert len(results) == 3

        for r in results:
            assert r.valid
            assert np.linalg.norm(r.position) > 6000000

    def test_get_states_at_times(self, batch_states):
        """Test querying all trajectories at multiple times."""
        states, epoch = batch_states
        batch = ballistic._core.TrajectoryBatch(states)
        batch.propagate_all(600.0)

        times = [epoch + 100.0, epoch + 300.0, epoch + 500.0]
        results = batch.get_states_at_times(times)

        assert len(results) == 3  # 3 trajectories
        for traj_results in results:
            assert len(traj_results) == 3  # 3 time points
            for r in traj_results:
                assert r.valid

    def test_propagate_to_altitude(self, batch_states):
        """Test altitude-based batch propagation."""
        states, _ = batch_states
        batch = ballistic._core.TrajectoryBatch(states)

        # Propagate to build some history first
        batch.propagate_all(600.0)

        results = batch.propagate_to_altitude(
            500000.0,  # 500 km
            ballistic.AltitudeCondition.Below,
            ballistic.PropagationDirection.Forward,
            600.0
        )

        assert len(results) == 3
        for r in results:
            assert r.starting_altitude_m > 0
            assert r.ending_altitude_m > 0


class TestEnums:
    """Tests for enum values."""

    def test_time_scale_values(self):
        """Test TimeScale enum values."""
        assert ballistic.TimeScale.UTC is not None
        assert ballistic.TimeScale.TT is not None
        assert ballistic.TimeScale.TAI is not None
        assert ballistic.TimeScale.TDB is not None
        assert ballistic.TimeScale.UT1 is not None
        assert ballistic.TimeScale.GPS is not None

    def test_integrator_type_values(self):
        """Test IntegratorType enum values."""
        assert ballistic.IntegratorType.RK45 is not None
        assert ballistic.IntegratorType.YOSHIDA6 is not None
        assert ballistic.IntegratorType.YOSHIDA6 is not None
        assert ballistic.IntegratorType.YOSHIDA8 is not None

    def test_itrf_realization_values(self):
        """Test ITRFRealization enum values."""
        assert ballistic.ITRFRealization.ITRF2000 is not None
        assert ballistic.ITRFRealization.ITRF2005 is not None
        assert ballistic.ITRFRealization.ITRF2008 is not None
        assert ballistic.ITRFRealization.ITRF2014 is not None
        assert ballistic.ITRFRealization.ITRF2020 is not None


class TestEpochExtended:
    """Extended tests for Epoch class - covering additional APIs."""

    def test_from_mjd(self):
        """Test creating epoch from Modified Julian Date."""
        # MJD 60325.5 corresponds to 2024-01-15 12:00:00 approximately  # guard-exempt(PUB-DATE): data
        mjd = 60325.5
        epoch = ballistic.Epoch.from_mjd(mjd, ballistic.TimeScale.TT)
        # MJD = JD - 2400000.5
        expected_jd = mjd + 2400000.5
        assert abs(epoch.tt_jd - expected_jd) < 1e-10

    def test_gps_jd(self):
        """Test GPS Julian Date property."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        gps_jd = epoch.gps_jd
        # GPS is ~19 seconds ahead of TAI, which is ~37 seconds ahead of UTC
        # So GPS_JD should be between UTC_JD and TT_JD
        assert epoch.utc_jd < gps_jd < epoch.tt_jd

    def test_tdb_jd(self):
        """Test Barycentric Dynamical Time Julian Date."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        tdb_jd = epoch.tdb_jd
        # TDB is very close to TT (differs by <2ms)
        tt_tdb_diff_sec = abs(epoch.tt_jd - tdb_jd) * 86400
        assert tt_tdb_diff_sec < 0.002  # Less than 2 milliseconds

    def test_ut1_jd(self):
        """Test UT1 Julian Date property."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        ut1_jd = epoch.ut1_jd
        # UT1-UTC is typically within ±0.9 seconds
        ut1_utc_diff_sec = abs(ut1_jd - epoch.utc_jd) * 86400
        assert ut1_utc_diff_sec < 1.0


class TestTrajectoryConfigExtended:
    """Extended tests for TrajectoryConfig (C++ core version) - covering additional APIs."""

    def test_symplectic_substruct(self):
        """Test symplectic configuration sub-struct."""
        config = ballistic._core.TrajectoryConfig()
        config.integrator = ballistic.IntegratorType.YOSHIDA6
        # Access symplectic sub-struct
        config.symplectic.step_size = 60.0
        assert config.symplectic.step_size == 60.0

    def test_report_forces_config(self):
        """Test force reporting configuration."""
        config = ballistic._core.TrajectoryConfig()
        assert config.report_forces is False
        config.report_forces = True
        assert config.report_forces is True

    def test_include_disabled_forces(self):
        """Test disabled forces inclusion config."""
        config = ballistic._core.TrajectoryConfig()
        assert config.include_disabled_forces is False
        config.include_disabled_forces = True
        assert config.include_disabled_forces is True


class TestInterpolatorConfigExtended:
    """Extended tests for InterpolatorConfig (C++ core version)."""

    def test_max_propagation_time(self):
        """Test max propagation time setting."""
        config = ballistic._core.InterpolatorConfig()
        config.max_propagation_time = 7200.0  # 2 hours
        assert config.max_propagation_time == 7200.0


class TestTrajectoryExtended:
    """Extended tests for Trajectory class (C++ core version)."""

    @pytest.fixture
    def trajectory(self):
        """Create a basic trajectory."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        return ballistic._core.Trajectory(epoch, pos, vel)

    def test_step_size_hint(self, trajectory):
        """Test setting step size hint."""
        trajectory.set_step_size_hint(30.0)
        # Take a step and verify it respects the hint (approximately)
        result = trajectory.step()
        # Step size should be influenced by hint
        assert result.step_size_taken > 0

    def test_current_step_size(self, trajectory):
        """Test getting current step size."""
        trajectory.step()
        step_size = trajectory.current_step_size()
        assert step_size > 0


class TestTrajectoryInterpolatorExtended:
    """Extended tests for TrajectoryInterpolator (C++ core version)."""

    @pytest.fixture
    def interpolator(self):
        """Create an interpolator with history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = ballistic._core.Trajectory(epoch, pos, vel)

        # Build some history
        for _ in range(20):
            traj.step()

        return ballistic._core.TrajectoryInterpolator(traj), epoch

    def test_position_only(self, interpolator):
        """Test position-only query."""
        interp, epoch = interpolator
        pos = interp.position(epoch + 100.0)
        if pos is not None:
            assert np.linalg.norm(pos) > 6000000

    def test_velocity_only(self, interpolator):
        """Test velocity-only query."""
        interp, epoch = interpolator
        vel = interp.velocity(epoch + 100.0)
        if vel is not None:
            assert np.linalg.norm(vel) > 7000

    def test_acceleration_only(self, interpolator):
        """Test acceleration-only query."""
        interp, epoch = interpolator
        acc = interp.acceleration(epoch + 100.0)
        if acc is not None:
            # Acceleration should be around 8-9 m/s^2 for LEO
            assert np.linalg.norm(acc) > 5

    def test_extend_to(self, interpolator):
        """Test extending trajectory to a time."""
        interp, epoch = interpolator
        initial_size = interp.history_size()
        # Get the current end time
        end_time = interp.history_end()
        # Extend forward beyond current range
        target_time = end_time + 100.0
        interp.extend_to(target_time)
        # History should have grown
        assert interp.history_size() >= initial_size

    def test_clear_cache(self, interpolator):
        """Test clearing interpolation cache."""
        interp, epoch = interpolator
        # Query to populate cache
        interp.get_state(epoch + 50.0)
        # Clear cache - should not raise
        interp.clear_cache()

    def test_invalidate(self, interpolator):
        """Test invalidate method."""
        interp, _ = interpolator
        # Invalidate should not raise
        interp.invalidate()


class TestTrajectoryBatchExtended:
    """Extended tests for TrajectoryBatch."""

    @pytest.fixture
    def batch(self):
        """Create a batch for testing."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        return ballistic._core.TrajectoryBatch(states), epoch

    def test_step_all(self, batch):
        """Test step_all method."""
        b, _ = batch
        b.step_all(5)  # Take 5 steps
        # All trajectories should have history
        for i in range(len(b)):
            traj = b.trajectory(i)
            assert traj.history_size() >= 5

    def test_set_output_frame(self, batch):
        """Test setting output frame for batch."""
        b, _ = batch
        b.propagate_all(60.0)
        # Set output frame - should not raise
        b.set_output_frame(ballistic.InterpolationFrame.GCRS)

    def test_set_itrf_realization(self, batch):
        """Test setting ITRF realization for batch."""
        b, _ = batch
        b.propagate_all(60.0)
        # Set ITRF - should not raise
        b.set_itrf_realization(ballistic.ITRFRealization.ITRF2014)

    def test_set_allow_propagation(self, batch):
        """Test setting allow propagation for batch."""
        b, _ = batch
        b.propagate_all(60.0)
        # Set allow propagation - should not raise
        b.set_allow_propagation(False)

    def test_set_max_propagation_time(self, batch):
        """Test setting max propagation time for batch."""
        b, _ = batch
        b.propagate_all(60.0)
        # Set max propagation time - should not raise
        b.set_max_propagation_time(3600.0)


class TestInterpolationFrameEnum:
    """Tests for InterpolationFrame enum values."""

    def test_interpolation_frame_values(self):
        """Test InterpolationFrame enum values."""
        assert ballistic.InterpolationFrame.ECEF is not None
        assert ballistic.InterpolationFrame.GCRS is not None


class TestPropagationFrameEnum:
    """Tests for PropagationFrame enum values."""

    def test_propagation_frame_values(self):
        """Test PropagationFrame enum values."""
        assert ballistic.PropagationFrame.GCRS is not None


class TestAltitudeConditionEnum:
    """Tests for AltitudeCondition enum values."""

    def test_altitude_condition_values(self):
        """Test AltitudeCondition enum values."""
        assert ballistic.AltitudeCondition.Above is not None
        assert ballistic.AltitudeCondition.Below is not None


class TestPropagationDirectionEnum:
    """Tests for PropagationDirection enum values."""

    def test_propagation_direction_values(self):
        """Test PropagationDirection enum values."""
        assert ballistic.PropagationDirection.Forward is not None
        assert ballistic.PropagationDirection.Backward is not None


class TestInitializationExtended:
    """Tests for extended initialization API."""

    def test_init_config_defaults(self):
        """Test InitConfig default values."""
        config = ballistic.InitConfig()
        assert config.egm2008_path is None
        assert config.eop_path is None
        assert config.jpl_ephemeris_path is None
        assert config.leap_seconds_path is None

    def test_init_config_set_paths(self):
        """Test setting paths on InitConfig."""
        config = ballistic.InitConfig()
        config.eop_path = "test/finals2000A.data"
        config.jpl_ephemeris_path = "test/de421.bsp"
        assert config.eop_path == "test/finals2000A.data"
        assert config.jpl_ephemeris_path == "test/de421.bsp"

    def test_init_config_path_initial_none(self):
        """Test that paths are initially None."""
        config = ballistic.InitConfig()
        # All paths should start as None
        assert config.egm2008_path is None
        assert config.eop_path is None
        assert config.jpl_ephemeris_path is None

    def test_initialize_with_config(self):
        """Test initialize_with_config function."""
        config = ballistic.InitConfig()
        result = ballistic.initialize_with_config(config)
        assert isinstance(result, ballistic.InitResult)
        assert result.success is True
        assert result.embedded_ephemeris_loaded is True
        assert result.gravity_degree_loaded > 0

    def test_init_result_attributes(self):
        """Test InitResult has all expected attributes."""
        result = ballistic.get_init_status()
        assert hasattr(result, 'success')
        assert hasattr(result, 'egm2008_loaded')
        assert hasattr(result, 'eop_loaded')
        assert hasattr(result, 'jpl_ephemeris_loaded')
        assert hasattr(result, 'embedded_ephemeris_loaded')
        assert hasattr(result, 'gravity_degree_loaded')
        assert hasattr(result, 'jpl_de_version')
        assert hasattr(result, 'error_message')

    def test_get_init_status(self):
        """Test get_init_status function."""
        ballistic.initialize()
        result = ballistic.get_init_status()
        assert result.success is True
        # With default init, embedded ephemeris should be loaded
        assert result.embedded_ephemeris_loaded is True

    def test_load_eop_nonexistent_file(self):
        """Test load_eop with nonexistent file returns False."""
        result = ballistic.load_eop("nonexistent_file.data")
        assert result is False

    def test_load_space_weather_nonexistent_file(self):
        """Test load_space_weather with nonexistent file returns False."""
        result = ballistic.load_space_weather("nonexistent_file.csv")
        assert result is False

    def test_load_jpl_ephemeris_nonexistent_file(self):
        """Test load_jpl_ephemeris with nonexistent file returns False."""
        result = ballistic.load_jpl_ephemeris("nonexistent_file.bsp")
        assert result is False


class TestLoader:
    """Tests for the Skyfield-style Loader class."""

    def test_loader_creation(self):
        """Test Loader can be created."""
        loader = ballistic.Loader(verbose=False)
        assert loader is not None
        assert loader.directory is not None

    def test_loader_custom_directory(self, tmp_path):
        """Test Loader with custom directory."""
        loader = ballistic.Loader(directory=str(tmp_path), verbose=False)
        assert str(tmp_path) in str(loader.directory)

    def test_loader_status(self):
        """Test Loader.status() returns expected structure."""
        loader = ballistic.Loader(verbose=False)
        status = loader.status()
        assert 'eop' in status
        assert 'leap_seconds' in status
        assert 'space_weather' in status
        for key, info in status.items():
            assert 'exists' in info
            assert 'loaded' in info
            assert 'path' in info

    def test_load_function_exists(self):
        """Test ballistic.load() function exists."""
        assert hasattr(ballistic, 'load')
        assert callable(ballistic.load)


class TestDataPathsAndLeapSeconds:
    """Tests for data paths and leap second functionality."""

    def test_get_data_directory(self):
        """Test get_data_directory returns a path string."""
        path = ballistic.get_data_directory()
        assert isinstance(path, str)
        assert len(path) > 0
        # Should contain 'ballistic' in the path
        assert 'ballistic' in path.lower()

    def test_get_leap_second_info(self):
        """Test get_leap_second_info returns correct structure."""
        info = ballistic.get_leap_second_info()
        assert isinstance(info, ballistic.LeapSecondInfo)
        assert hasattr(info, 'current_tai_utc')
        assert hasattr(info, 'entry_count')
        assert hasattr(info, 'loaded_from_file')
        assert hasattr(info, 'expiration_jd')
        # Current TAI-UTC should be 37 (as of 2017)
        assert info.current_tai_utc == 37
        # Should have 28 entries (1972-2017)
        assert info.entry_count == 28
        # Using embedded data by default
        assert info.loaded_from_file is False

    def test_load_leap_seconds_nonexistent_file(self):
        """Test load_leap_seconds with nonexistent file returns error."""
        success, error = ballistic.load_leap_seconds("nonexistent.txt")
        assert success is False
        assert len(error) > 0
        # Error should contain download URL
        assert "hpiers.obspm.fr" in error

    def test_load_leap_seconds_verbose_false(self):
        """Test load_leap_seconds with verbose=False returns short error."""
        success, error = ballistic.load_leap_seconds("nonexistent.txt", verbose=False)
        assert success is False
        assert len(error) > 0
        # Short error should NOT contain download URL
        assert "hpiers.obspm.fr" not in error

    def test_init_config_leap_seconds_path(self):
        """Test InitConfig.leap_seconds_path property."""
        config = ballistic.InitConfig()
        # Should be None initially
        assert config.leap_seconds_path is None
        # Can set it
        config.leap_seconds_path = "test/leap-seconds.list"
        assert config.leap_seconds_path == "test/leap-seconds.list"

    def test_init_result_leap_seconds_loaded(self):
        """Test InitResult has leap_seconds_loaded attribute."""
        result = ballistic.get_init_status()
        assert hasattr(result, 'leap_seconds_loaded')
        # Using embedded data, so not loaded from file
        assert result.leap_seconds_loaded is False

    def test_error_message_contains_download_url(self):
        """Test that init errors contain helpful download instructions."""
        # Explicit paths are strict — missing file always fails init.
        config = ballistic.InitConfig()
        config.eop_path = "nonexistent_eop.data"

        result = ballistic.initialize_with_config(config)
        assert result.success is False
        # Verbose error always; should contain download URL
        assert "datacenter.iers.org" in result.error_message
        assert "not found" in result.error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
