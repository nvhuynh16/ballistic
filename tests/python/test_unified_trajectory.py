"""
Unit tests for the unified Pythonic Trajectory class.

Run with: pytest tests/python/test_unified_trajectory.py -v
"""

import pytest
import numpy as np


# Skip all tests if the module isn't built yet
try:
    import ballistic
    from ballistic.trajectory import Trajectory, TrajectoryConfig, InterpolatorConfig
    BALLISTIC_AVAILABLE = True
except ImportError:
    BALLISTIC_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not BALLISTIC_AVAILABLE,
    reason="ballistic module not available - build with BALLISTIC_BUILD_PYTHON=ON"
)


# Session-scoped initialization is handled by conftest.py


class TestTrajectoryConfig:
    """Tests for TrajectoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrajectoryConfig()
        assert config.gravity_degree == 70
        assert config.enable_third_body is True
        assert config.srp is None  # radiation off by default (POE/SP3 refs include SRP)
        assert config.enable_relativistic is True
        assert config.symplectic_step_size == 30.0
        assert config.report_forces is False
        assert config.include_disabled_forces is False
        # Drag opt-in via drag_env.
        assert config.drag_env is None

    def test_custom_values(self):
        """Test setting custom configuration values."""
        config = TrajectoryConfig(
            gravity_degree=120,
            symplectic_step_size=60.0,
        )
        assert config.gravity_degree == 120
        assert config.symplectic_step_size == 60.0

    def test_to_core_conversion(self):
        """Test conversion to C++ config object."""
        config = TrajectoryConfig(
            gravity_degree=100,
            integrator=ballistic.IntegratorType.YOSHIDA6,
            symplectic_step_size=45.0,
        )
        core_cfg = config._to_core()
        assert core_cfg.gravity_degree == 100
        assert core_cfg.integrator == ballistic.IntegratorType.YOSHIDA6
        assert core_cfg.symplectic.step_size == 45.0


class TestInterpolatorConfig:
    """Tests for InterpolatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InterpolatorConfig()
        assert config.cache_coefficients is True
        assert config.allow_propagation is True
        assert config.max_propagation_time == 3600.0

    def test_custom_values(self):
        """Test setting custom configuration values."""
        config = InterpolatorConfig(
            cache_coefficients=False,
            max_propagation_time=7200.0,
        )
        assert config.cache_coefficients is False
        assert config.max_propagation_time == 7200.0

    def test_to_core_conversion(self):
        """Test conversion to C++ config object."""
        config = InterpolatorConfig(
            cache_coefficients=False,
            allow_propagation=False,
        )
        core_cfg = config._to_core(ballistic.InterpolationFrame.GCRS)
        assert core_cfg.output_frame == ballistic.InterpolationFrame.GCRS
        assert core_cfg.cache_coefficients is False
        assert core_cfg.allow_propagation is False


class TestSingleTrajectoryConstruction:
    """Tests for single trajectory construction."""

    @pytest.fixture
    def initial_state(self):
        """Create initial state for tests."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        return epoch, pos, vel

    def test_basic_construction(self, initial_state):
        """Test basic trajectory construction."""
        epoch, pos, vel = initial_state
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        assert not traj.is_batch
        assert len(traj) == 1

    def test_construction_with_list_input(self, initial_state):
        """Test construction with Python lists."""
        epoch, pos, vel = initial_state
        traj = Trajectory(ballistic.State.from_itrf(epoch, list(pos), list(vel)))
        assert not traj.is_batch

    def test_construction_with_config(self, initial_state):
        """Test construction with custom config."""
        epoch, pos, vel = initial_state
        config = TrajectoryConfig(gravity_degree=50)
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=config)
        assert traj.config().gravity_degree == 50

    def test_repr(self, initial_state):
        """Test string representation."""
        epoch, pos, vel = initial_state
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        s = repr(traj)
        assert "single" in s
        assert "count=1" in s


class TestSingleTrajectoryPropagation:
    """Tests for single trajectory propagation."""

    @pytest.fixture
    def trajectory(self):
        """Create a basic trajectory."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        return Trajectory(ballistic.State.from_itrf(epoch, pos, vel))

    def test_step(self, trajectory):
        """Test single step."""
        result = trajectory.step()
        assert result is not None
        assert result.step_size_taken > 0
        assert trajectory.history_size() == 2

    def test_multiple_steps(self, trajectory):
        """Test multiple steps."""
        for _ in range(10):
            trajectory.step()
        assert trajectory.history_size() == 11

    def test_propagate_to_epoch(self, trajectory):
        """Test propagate to epoch."""
        initial_epoch = trajectory.epoch()
        target = initial_epoch + 60.0
        result = trajectory.propagate_to(target)
        assert np.linalg.norm(result.position) > 6000000

    def test_propagate_with_float(self, trajectory):
        """Test propagate with float offset."""
        result = trajectory.propagate_for(60.0)
        assert np.linalg.norm(result.position) > 6000000


class TestSingleTrajectoryInterpolation:
    """Tests for single trajectory interpolation."""

    @pytest.fixture
    def trajectory_with_history(self):
        """Create trajectory with history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        # Build some history
        while traj.epoch() - epoch < 600.0:
            traj.step()
        return traj, epoch

    def test_call_with_float(self, trajectory_with_history):
        """Test interpolation with float offset."""
        traj, epoch = trajectory_with_history
        state = traj(300.0)  # 5 minutes
        assert state is not None
        assert np.linalg.norm(state.position) > 6000000

    def test_call_with_epoch(self, trajectory_with_history):
        """Test interpolation with Epoch."""
        traj, epoch = trajectory_with_history
        state = traj(epoch + 300.0)
        assert state is not None
        assert np.linalg.norm(state.position) > 6000000

    def test_call_with_array(self, trajectory_with_history):
        """Test interpolation with numpy array."""
        traj, epoch = trajectory_with_history
        times = np.array([100.0, 200.0, 300.0])
        states = traj(times)
        assert len(states) == 3
        for state in states:
            assert state is not None

    def test_call_with_list(self, trajectory_with_history):
        """Test interpolation with list of floats."""
        traj, epoch = trajectory_with_history
        states = traj([100.0, 200.0, 300.0])
        assert len(states) == 3

    def test_call_with_mixed_list(self, trajectory_with_history):
        """Test interpolation with mixed list."""
        traj, epoch = trajectory_with_history
        states = traj([100.0, epoch + 200.0, 300.0])
        assert len(states) == 3

    def test_frame_string_ecef(self, trajectory_with_history):
        """Test frame string 'ECEF'."""
        traj, _ = trajectory_with_history
        state = traj(300.0, frame='ECEF')
        assert state is not None

    def test_frame_string_gcrs(self, trajectory_with_history):
        """Test frame string 'GCRS'."""
        traj, _ = trajectory_with_history
        state = traj(300.0, frame='GCRS')
        assert state is not None

    def test_frame_string_case_insensitive(self, trajectory_with_history):
        """Test frame string is case insensitive."""
        traj, _ = trajectory_with_history
        state1 = traj(300.0, frame='ecef')
        state2 = traj(300.0, frame='ECEF')
        assert np.allclose(state1.position, state2.position)

    def test_frame_enum(self, trajectory_with_history):
        """Test frame enum value."""
        traj, _ = trajectory_with_history
        state = traj(300.0, frame=ballistic.InterpolationFrame.GCRS)
        assert state is not None

    def test_invalid_frame_string(self, trajectory_with_history):
        """Test invalid frame string raises error."""
        traj, _ = trajectory_with_history
        with pytest.raises(ValueError, match="Unknown frame"):
            traj(300.0, frame='invalid')


class TestSingleTrajectoryAutoPropagation:
    """Tests for auto-propagation when times are outside history."""

    @pytest.fixture
    def trajectory(self):
        """Create a basic trajectory with minimal history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        # Take a few steps
        for _ in range(5):
            traj.step()
        return traj, epoch

    def test_auto_propagate_forward(self, trajectory):
        """Test auto-propagation extends forward."""
        traj, epoch = trajectory
        initial_size = traj.history_size()
        # Request time beyond current history
        state = traj(600.0)  # 10 minutes
        assert state is not None
        # History should have grown
        assert traj.history_size() > initial_size


class TestSingleTrajectoryHistory:
    """Tests for history access."""

    @pytest.fixture
    def trajectory(self):
        """Create a trajectory with history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        for _ in range(10):
            traj.step()
        return traj, epoch, pos

    def test_history(self, trajectory):
        """Test history access."""
        traj, _, pos = trajectory
        history = traj.history()
        assert len(history) == 11  # Initial + 10 steps
        # First entry should match initial position
        assert np.allclose(history[0].position, pos, rtol=1e-10)

    def test_history_size(self, trajectory):
        """Test history_size method."""
        traj, _, _ = trajectory
        assert traj.history_size() == 11

    def test_clear_history(self, trajectory):
        """Test clear_history method."""
        traj, _, _ = trajectory
        traj.clear_history()
        # History is cleared but trajectory keeps current state
        # History size is 0 or minimal after clear


class TestSingleTrajectoryState:
    """Tests for state access methods."""

    @pytest.fixture
    def trajectory(self):
        """Create a trajectory."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        traj.step()
        return traj, epoch

    def test_epoch(self, trajectory):
        """Test epoch access."""
        traj, initial_epoch = trajectory
        current = traj.epoch()
        assert current > initial_epoch

    def test_state_ecef(self, trajectory):
        """Test state_ecef access."""
        traj, _ = trajectory
        state = traj.state_ecef()
        assert np.linalg.norm(state.position) > 6000000

    def test_state_gcrs(self, trajectory):
        """Test state_gcrs access; state_eci names the replacement."""
        traj, _ = trajectory
        state = traj.state_gcrs()
        assert np.linalg.norm(state.position) > 6000000
        with pytest.raises(ballistic.RemovedApiError, match="state_gcrs"):
            traj.state_eci()


class TestBatchTrajectoryConstruction:
    """Tests for batch trajectory construction."""

    @pytest.fixture
    def batch_states(self):
        """Create batch states."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6978137.0, 0.0, 0.0]), np.array([0.0, 7300.0, 0.0])),
        ]
        return states, epoch

    def test_batch_construction(self, batch_states):
        """Test batch trajectory construction."""
        states, _ = batch_states
        traj = Trajectory(states)
        assert traj.is_batch
        assert len(traj) == 3

    def test_batch_repr(self, batch_states):
        """Test batch string representation."""
        states, _ = batch_states
        traj = Trajectory(states)
        s = repr(traj)
        assert "batch" in s
        assert "count=3" in s


class TestBatchTrajectoryPropagation:
    """Tests for batch trajectory propagation."""

    @pytest.fixture
    def batch(self):
        """Create a batch trajectory."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        return Trajectory(states), epoch

    def test_step(self, batch):
        """Test batch step."""
        traj, _ = batch
        result = traj.step()
        assert result is None  # Batch step returns None
        sizes = traj.history_size()
        assert all(s >= 2 for s in sizes)

    def test_propagate(self, batch):
        """Test batch propagate."""
        traj, _ = batch
        results = traj.propagate_for(60.0)
        assert len(results) == 2
        for state in results:
            assert np.linalg.norm(state.position) > 6000000


class TestBatchTrajectoryInterpolation:
    """Tests for batch trajectory interpolation."""

    @pytest.fixture
    def batch_with_history(self):
        """Create batch with history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        traj = Trajectory(states)
        traj.propagate_for(600.0)
        return traj, epoch

    def test_call_single_time(self, batch_with_history):
        """Test interpolation at single time."""
        traj, _ = batch_with_history
        results = traj(300.0)
        assert len(results) == 2
        for r in results:
            assert r.valid

    def test_call_multiple_times(self, batch_with_history):
        """Test interpolation at multiple times."""
        traj, _ = batch_with_history
        results = traj([100.0, 200.0, 300.0])
        assert len(results) == 2  # 2 trajectories
        for traj_results in results:
            assert len(traj_results) == 3  # 3 time points


class TestBatchTrajectoryState:
    """Tests for batch state access."""

    @pytest.fixture
    def batch(self):
        """Create a batch trajectory."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        traj = Trajectory(states)
        traj.step()
        return traj, epoch

    def test_epoch_list(self, batch):
        """Test epoch returns list."""
        traj, _ = batch
        epochs = traj.epoch()
        assert len(epochs) == 2

    def test_state_ecef_list(self, batch):
        """Test state_ecef returns list."""
        traj, _ = batch
        states = traj.state_ecef()
        assert len(states) == 2

    def test_history_list(self, batch):
        """Test history returns list of lists."""
        traj, _ = batch
        histories = traj.history()
        assert len(histories) == 2
        for h in histories:
            assert len(h) >= 2


class TestSymplecticIntegrator:
    """Tests for symplectic integrator support."""

    @pytest.fixture
    def symplectic_config(self):
        """Create symplectic configuration."""
        return TrajectoryConfig(
            integrator=ballistic.IntegratorType.YOSHIDA6,
            symplectic_step_size=30.0,
        )

    def test_single_symplectic(self, symplectic_config):
        """Test single trajectory with symplectic integrator."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=symplectic_config)
        assert traj.is_symplectic()

    def test_batch_symplectic(self, symplectic_config):
        """Test batch with symplectic integrator."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        traj = Trajectory(states, config=symplectic_config)
        assert traj.is_symplectic()

    def test_energy_drift_single(self, symplectic_config):
        """Test energy drift for single trajectory."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=symplectic_config)
        for _ in range(10):
            traj.step()
        drift = traj.energy_drift()
        assert abs(drift) < 1e-3  # Bounded energy drift

    def test_energy_drift_batch(self, symplectic_config):
        """Test energy drift for batch."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        traj = Trajectory(states, config=symplectic_config)
        for _ in range(10):
            traj.step()
        drifts = traj.energy_drift()
        assert len(drifts) == 2
        for drift in drifts:
            assert abs(drift) < 1e-3

    def test_symplectic_interpolation(self, symplectic_config):
        """Test interpolation with symplectic integrator."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=symplectic_config)
        for _ in range(20):
            traj.step()
        # Interpolation should work with symplectic history
        state = traj(300.0)
        assert state is not None


class TestAltitudePropagation:
    """Tests for altitude-based propagation."""

    @pytest.fixture
    def trajectory_with_history(self):
        """Create trajectory with history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        traj.propagate_for(600.0)
        return traj

    def test_propagate_to_altitude_single(self, trajectory_with_history):
        """Test propagate_to_altitude for single trajectory."""
        result = trajectory_with_history.propagate_to_altitude(
            350000.0,
            ballistic.AltitudeCondition.Below,
            max_duration_s=600.0,
        )
        assert result.starting_altitude_m > 0
        assert result.ending_altitude_m > 0

    def test_propagate_to_altitude_batch(self):
        """Test propagate_to_altitude for batch."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        traj = Trajectory(states)
        traj.propagate_for(600.0)
        results = traj.propagate_to_altitude(
            500000.0,
            ballistic.AltitudeCondition.Below,
            max_duration_s=600.0,
        )
        assert len(results) == 2


class TestLowLevelAccess:
    """Tests for low-level access to C++ objects."""

    @pytest.fixture
    def trajectory(self):
        """Create a trajectory."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        traj.step()
        return traj

    def test_trajectory_access(self, trajectory):
        """Test accessing underlying Trajectory object."""
        core_traj = trajectory.trajectory()
        assert core_traj is not None
        assert isinstance(core_traj, ballistic._core.Trajectory)

    def test_interpolator_access(self, trajectory):
        """Test accessing underlying interpolator."""
        # First access through call to create interpolator
        trajectory(0.0)
        interp = trajectory.interpolator()
        assert interp is not None

    def test_clear_cache(self, trajectory):
        """Test clear_cache method."""
        trajectory(0.0)  # Create interpolator
        trajectory.clear_cache()  # Should not raise


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_times_array(self):
        """Test with empty times array."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        traj.step()
        # Empty list should return empty list
        results = traj([])
        assert len(results) == 0

    def test_scalar_numpy_array(self):
        """Test with scalar numpy array."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        for _ in range(5):
            traj.step()
        # Scalar numpy array
        t = np.float64(30.0)
        state = traj(t)
        assert state is not None

    def test_invalid_construction(self):
        """Test invalid construction raises error."""
        with pytest.raises(TypeError):
            Trajectory("invalid")

    def test_batch_step_backward_not_supported(self):
        """Test step_backward raises for batch."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
        ]
        traj = Trajectory(states)
        with pytest.raises(NotImplementedError):
            traj.step_backward()


class TestAutoBackwardPropagation:
    """Tests for automatic backward propagation (C++ handles coverage)."""

    @pytest.fixture
    def trajectory(self):
        """Create a trajectory with some forward history."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        pos = np.array([6778137.0, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = Trajectory(ballistic.State.from_itrf(epoch, pos, vel))
        for _ in range(5):
            traj.step()
        return traj, epoch

    def test_auto_propagate_backward(self, trajectory):
        """Test that querying before history auto-extends backward."""
        traj, epoch = trajectory
        initial_size = traj.history_size()
        # Query 60 seconds before initial epoch
        state = traj(-60.0)
        assert state is not None
        assert np.linalg.norm(state.position) > 6000000
        # History should have grown
        assert traj.history_size() > initial_size

    def test_batch_backward_interpolation(self):
        """Test batch backward interpolation works for all trajectories."""
        epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
        states = [
            ballistic.State.from_itrf(epoch, np.array([6778137.0, 0.0, 0.0]), np.array([0.0, 7500.0, 0.0])),
            ballistic.State.from_itrf(epoch, np.array([6878137.0, 0.0, 0.0]), np.array([0.0, 7400.0, 0.0])),
        ]
        traj = Trajectory(states)
        # Build some forward history
        traj.propagate_for(300.0)

        # Query 30 seconds before initial epoch
        results = traj(-30.0)
        assert len(results) == 2
        for r in results:
            assert r.valid
            assert np.linalg.norm(r.position) > 6000000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
