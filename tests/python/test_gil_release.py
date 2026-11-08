"""GIL-release smoke test.


Long-running pure-C++ entry points (Trajectory.propagate/step, batch
propagation, the estimate_ea_* facades, from_models_* generators) release
the GIL via nb::call_guard<nb::gil_scoped_release>. Two threads propagating
concurrently after initialize() exercises both the release itself and the
EOPData read-only concurrency contract (initialize before spawning threads;
never reload concurrently).
"""

import threading

import numpy as np
import pytest

try:
    import ballistic
    BALLISTIC_AVAILABLE = True
except ImportError:
    BALLISTIC_AVAILABLE = False


@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_concurrent_propagation(leo_state):
    epoch0, pos0, vel0 = leo_state

    results = {}
    errors = []

    def propagate(tag, vel_offset):
        try:
            traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch0, pos0, np.asarray(vel0) + vel_offset))
            r = traj.propagate_to(epoch0 + 3600.0)
            results[tag] = np.asarray(r.position).copy()
        except Exception as e:  # pragma: no cover - failure diagnostics
            errors.append((tag, repr(e)))

    threads = [
        threading.Thread(target=propagate, args=("a", 0.0)),
        threading.Thread(target=propagate, args=("b", 1.0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300.0)
        assert not t.is_alive(), "propagation thread hung"

    assert not errors, f"thread errors: {errors}"
    assert set(results) == {"a", "b"}
    for tag, pos in results.items():
        assert np.all(np.isfinite(pos)), f"{tag}: non-finite position"
        r = np.linalg.norm(pos)
        assert 6.4e6 < r < 9.0e6, f"{tag}: implausible radius {r}"

    # Different initial velocities must yield different trajectories —
    # guards against results leaking across threads.
    assert np.linalg.norm(results["a"] - results["b"]) > 1.0


@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_step_releases_gil(leo_state):
    """step() with the GIL released still returns a usable StepResult."""
    epoch0, pos0, vel0 = leo_state
    traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch0, pos0, vel0))
    r = traj.step()
    assert np.all(np.isfinite(np.asarray(r.position)))
    assert r.step_size_taken > 0.0


def _recommended_like_config():
    """Gravity 70 + third body + tides, Sentinel-1A radiation pressure and a
    live drag environment on the committed weather (loaded by conftest)."""
    s1 = ballistic.satellite_properties.get("S1A")
    cfg = ballistic.TrajectoryConfig(
        gravity_degree=70,
        srp=ballistic.SrpConfig(mass_kg=s1.mass_kg, area_m2=s1.srp_area_m2,
                                cr=1.0 + s1.srp_reflectivity))
    return cfg


def _run_threads(n, target):
    errors = []

    def wrapped(i):
        try:
            target(i)
        except Exception as e:  # pragma: no cover - failure diagnostics
            errors.append((i, repr(e)))

    threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600.0)
        assert not t.is_alive(), "propagation thread hung"
    assert not errors, f"thread errors: {errors}"


@pytest.mark.nrl
@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_eight_threads_recommended_config():
    """Eight Trajectories, each on its own thread, share ONE drag environment
    and SrpConfig; every result equals the serial run bit for bit (the
    environment's model caches are per thread)."""
    epoch0 = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    pos = np.array([6778137.0, 0.0, 0.0])
    # Circular 400 km, inclined 51.6 deg; Earth-fixed velocity.
    vel = np.array([0.0, 7669.0 * 0.6211, 7669.0 * 0.7837]) - np.cross(
        [0.0, 0.0, ballistic.constants.earth.ANGULAR_VELOCITY], pos)
    cfg = _recommended_like_config()
    s1 = ballistic.satellite_properties.get("S1A")
    env, _ = ballistic.make_drag_environment(
        epoch0, pos, vel, 0.0, cfg, ballistic.BetaProfile1D.constant(s1.beta_kg_m2))
    cfg = cfg.replace(drag_env=env)

    def run(k):
        start = epoch0 + float(k)       # staggered starts: threads not in lockstep
        traj = ballistic.Trajectory(ballistic.State.from_itrf(start, pos, vel), cfg)
        r = traj.propagate_to(start + 900.0)
        return np.concatenate([np.asarray(r.position), np.asarray(r.velocity)])

    serial = [run(k) for k in range(8)]
    threaded = [None] * 8

    def target(k):
        threaded[k] = run(k)

    _run_threads(8, target)
    for k in range(8):
        assert np.array_equal(threaded[k], serial[k]), (
            f"thread {k} differs by {np.linalg.norm(threaded[k][:3] - serial[k][:3])} m")


@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_batch_and_single_on_two_threads(leo_state):
    """A 16-member batch propagating on one thread while a single trajectory
    propagates on another: both equal their serial runs bit for bit."""
    epoch0, pos0, _ = leo_state
    pos0 = np.asarray(pos0)
    vel0 = np.array([0.0, 7175.0, 0.0])   # circular 400 km, Earth-fixed
    states = [ballistic.State.from_itrf(epoch0, pos0, vel0 + [0.0, 0.5 * i, 0.2 * i])
              for i in range(16)]

    def run_batch():
        batch = ballistic.Trajectory(states)
        out = batch.propagate_for(1800.0)
        return np.array([np.concatenate([np.asarray(s.position), np.asarray(s.velocity)])
                         for s in out])

    def run_single():
        traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch0, pos0, vel0))
        r = traj.propagate_for(1800.0)
        return np.concatenate([np.asarray(r.position), np.asarray(r.velocity)])

    serial_batch, serial_single = run_batch(), run_single()
    out = {}

    def target(i):
        out[i] = run_batch() if i == 0 else run_single()

    _run_threads(2, target)
    assert out[0].shape == (16, 6)
    assert np.array_equal(out[0], serial_batch)
    assert np.array_equal(out[1], serial_single)
