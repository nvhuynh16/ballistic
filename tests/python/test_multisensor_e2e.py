"""
End-to-end Python test for the multi-sensor surface (the GM-RTS smoother is
the only public fusion estimator).

Exercises the full Python → C++ → Python path:
  1. Build a no-EA-truth scenario in Python (gravity-only propagation).
  2. Sample a synthetic radar at a ground station; inject Gaussian noise.
  3. Anchor the initial state with a position fix.
  4. Run estimate_ea_smoother through the Pythonic API
     (kwarg scenario constructor + numpy bulk set_samples).
  5. Assert the result is finite and the recovered EA is small
     (truth EA is zero; loose bound catches divergence).

Wall-time target: < 30 s. Verifies the integration of the bindings:
  - EmpiricalAcceleration.evaluate_at batched query
  - RadarObs / PositionFixObs.set_samples numpy builders
  - MultiSensorScenario kwarg constructor
  - Top-level ballistic.* re-exports
"""

import numpy as np
import pytest

try:
    import ballistic
    BALLISTIC_AVAILABLE = True
except ImportError:
    BALLISTIC_AVAILABLE = False




@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_estimation_exports():
    """Every multisensor public symbol is importable from ballistic.estimation."""
    for name in (
        "RadarObs", "OpticsObs", "PositionFixObs", "AccelObs",
        "MultiSensorScenario", "SensorMask",
        "EASmootherConfig", "EASmootherResult", "EASmootherRateBudget",
        "estimate_ea_smoother", "SensorClock",
    ):
        assert hasattr(ballistic.estimation, name), f"ballistic.estimation.{name} missing"
        assert name in ballistic.estimation.__all__


def test_python_config_parity_fixes():
    """Regression, two Python/C++ parity bugs:
      1. ballistic.estimation.estimate_ea_smoother must be the Python WRAPPER that
         accepts the Pythonic TrajectoryConfig dataclass — the raw _core
         binding (which only takes _core.TrajectoryConfig) used to shadow it.
      2. EABSplineConfig.gn is std::optional<GNRefine> in C++; the Python
         setter must accept None (to DISABLE the default-on GN), not only a
         GNRefine.
    """
    assert ballistic.estimation.estimate_ea_smoother.__module__ == "ballistic.estimation"

    c = ballistic.estimation.EABSplineConfig()
    assert c.gn is not None, "GN is default-ON"
    c.gn = None                       # must not raise (was a TypeError)
    assert c.gn is None
    c.gn = ballistic.estimation.GNRefine()
    assert c.gn is not None


@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_ea_smoother_e2e(leo_state):
    """GM-RTS smoother facade: zero-EA truth, all-auto config."""
    epoch0, pos0, vel0 = leo_state

    arc_s = 184.0
    traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch0, pos0, vel0))
    while traj.epoch() - epoch0 < arc_s:
        traj.step()

    station_ecef = np.array([ballistic.constants.earth.RADIUS_EQUATORIAL, 0.0, 0.0])
    rng = np.random.default_rng(42)
    N = int(arc_s) + 1
    times_s = np.linspace(0.0, arc_s, N)
    sigma_range, sigma_angle, sigma_rate = 10.0, 5.0e-3, 0.5

    ecef_to_enu = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    radar = ballistic.estimation.RadarObs()
    radar.position_ecef = station_ecef
    radar.sigma_range = sigma_range
    radar.sigma_angle = sigma_angle
    radar.sigma_rate = sigma_rate
    radar.lat_rad = 0.0
    radar.lon_rad = 0.0
    radar.ecef_to_enu = ecef_to_enu

    obs = np.zeros((N, 4))
    for i, t_s in enumerate(times_s):
        st = traj(epoch0 + float(t_s))
        sv = ballistic._core.StateVector()
        sv.position = np.asarray(st.position)
        sv.velocity = np.asarray(st.velocity)
        clean = np.asarray(radar.predict(sv))
        obs[i] = clean + np.array([
            sigma_range * rng.standard_normal(),
            sigma_angle * rng.standard_normal(),
            sigma_angle * rng.standard_normal(),
            sigma_rate * rng.standard_normal(),
        ])
    radar.set_samples(epoch0, times_s, obs)

    fix = ballistic.estimation.PositionFixObs()
    fix.sigma_pos = 1.0
    fix.set_samples(epoch0, np.array([0.0]), pos0.reshape(1, 3))

    scenario = ballistic.estimation.MultiSensorScenario(
        epoch0=epoch0, pos0_ecef=pos0, vel0_ecef=vel0,
        arc_duration_s=arc_s, radars=[radar], fixes=[fix])
    mask = ballistic.estimation.SensorMask.all_on(scenario)

    fm_cfg = ballistic.TrajectoryConfig()
    cfg = ballistic.estimation.EASmootherConfig()   # all-auto: NIS grid + rate budget
    # Pass the Pythonic dataclass directly (NOT _to_core()) — exercises the
    # ballistic.estimation wrapper that accepts it (regression: the raw _core binding
    # must not shadow the wrapper and reject the dataclass).
    result = ballistic.estimation.estimate_ea_smoother(scenario, mask, fm_cfg, cfg)

    assert result.error_message == "", f"smoother error: {result.error_message}"
    assert result.succeeded()
    assert 0.1 < result.nis < 10.0, f"NIS/dof {result.nis}"
    # Honest budget: measured sigma_rate_eff >= sensor sigma.
    assert result.sigma_rate_eff >= sigma_rate

    ea_grid = result.empirical_accel.evaluate_at(np.linspace(0.0, arc_s, 50))
    assert np.all(np.isfinite(ea_grid))
    ea_rms = float(np.sqrt(np.mean(ea_grid ** 2)))
    assert ea_rms < 1.0, f"zero-truth EA RMS {ea_rms:.3f} m/s2"

    # Posterior sigma present and finite.
    sig = np.asarray(result.ea_sigma)
    assert sig.shape[1] == 3 and np.all(np.isfinite(sig)) and np.all(sig >= 0)

    # Report-only warnings list (empty on this gapless fixture).
    assert isinstance(result.warnings, list)

    # Tier-1 surfaces: declared-vs-honored audit + per-block clock biases.
    honored = dict(result.sigma_honored)
    assert honored, "sigma_honored empty"
    assert all(np.isfinite(v) and v >= 0 for v in honored.values())
    assert any(k.endswith("_range") for k in honored)
    assert isinstance(result.clock_biases, list)

    # information_bound takes the same dataclass (by position or keyword)
    # and gives the same answer as the raw binding on the converted config.
    bound = ballistic.estimation.information_bound(scenario, mask, fm_cfg)
    assert bound.error_message == "", f"bound error: {bound.error_message}"
    assert np.isfinite(bound.posterior_bound_m) and bound.posterior_bound_m > 0
    by_keyword = ballistic.estimation.information_bound(scenario, mask, fm_cfg=fm_cfg)
    raw = ballistic._core.information_bound(scenario, mask, fm_cfg._to_core())
    assert by_keyword.posterior_bound_m == bound.posterior_bound_m
    assert raw.posterior_bound_m == bound.posterior_bound_m
    with pytest.raises(TypeError, match="force model"):
        ballistic.estimation.information_bound(scenario, mask)

    print(f"\n  smoother e2e: ea_rms={ea_rms:.4f} nis={result.nis:.3f} "
          f"sigma_a={result.sigma_a_used} tau={result.tau_used} "
          f"sigma_rate_eff={result.sigma_rate_eff:.3f}")


@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_tier1_declarative_fields_roundtrip():
    """Tier-1 fields are settable from Python and default to no-op values."""
    r = ballistic.estimation.RadarObs()
    assert r.clock == ballistic.estimation.SensorClock.Shared
    assert np.allclose(np.asarray(r.known_bias), 0.0)
    r.known_bias = np.array([25.0, 0.0, 0.0, 0.0])
    r.sigma_scale = [1.0, 2.0]
    r.clock = ballistic.estimation.SensorClock.Own
    assert r.known_bias[0] == 25.0 and list(r.sigma_scale) == [1.0, 2.0]
    assert r.clock == ballistic.estimation.SensorClock.Own

    cam = ballistic.estimation.OpticsObs()
    assert cam.clock == ballistic.estimation.SensorClock.None_
    fix = ballistic.estimation.PositionFixObs()
    fix.known_bias = np.array([1.0, 2.0, 3.0])
    assert fix.known_bias[2] == 3.0
    acc = ballistic.estimation.AccelObs()
    acc.sigma_scale = [0.5]
    assert list(acc.sigma_scale) == [0.5]
    assert acc.outlier_prob == 0.0 and acc.outlier_scale == 10.0
    acc.outlier_prob = 0.05
    acc.outlier_scale = 8.0
    assert acc.outlier_prob == 0.05 and acc.outlier_scale == 8.0


@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_problem_spec_python_surface(leo_state):
    """estimate_ea(spec): missing-sigma error path + typed-sample dispatch."""
    epoch, pos, vel = leo_state

    spec = ballistic.estimation.EAProblemSpec()
    spec.epoch0 = epoch
    spec.pos0_ecef = pos
    spec.vel0_ecef = vel
    spec.observations = ballistic.estimation.PositionSamples([])
    spec.force_model = ballistic.TrajectoryConfig()._to_core()
    r = ballistic.estimation.estimate_ea(spec)
    assert not r.succeeded() and "noise_sigma" in r.error_message

    # ACC dispatch: zero-truth accel samples on a short LEO arc.
    acc = ballistic.estimation.AccelSamples(
        [(epoch + float(t), np.zeros(3)) for t in np.arange(0.0, 30.0, 0.5)])
    spec.observations = acc
    spec.noise_sigma = 1e-3
    r = ballistic.estimation.estimate_ea(spec)
    assert r.error_message == "" and r.method_used == ballistic.estimation.EAMethod.BSplineAcceleration
    assert r.bspline_detail is not None
    assert isinstance(r.audit_warnings, list)

    # Heavy-tail surface: declarative tail fields + the standalone prefilter.
    assert spec.outlier_prob == 0.0 and spec.outlier_scale == 10.0
    spec.outlier_prob = 0.05
    samples = [(epoch + float(t), np.zeros(3)) for t in np.arange(0.0, 10.0, 0.1)]
    samples[50] = (samples[50][0], np.array([1.0, 0.0, 0.0]))  # 10-sigma spike
    kept, nrej, oob = ballistic.estimation.reject_heavy_tails(samples, sigma=0.1)
    assert nrej == 1 and len(kept) == len(samples) - 1 and not oob

    # Below the measured ~4 Hz envelope the prefilter refuses LOUDLY
    # (nothing dropped, out_of_envelope=True) instead of silently no-oping.
    sparse = [(epoch + float(t), np.zeros(3)) for t in np.arange(0.0, 30.0, 1.0)]
    sparse[10] = (sparse[10][0], np.array([1.0, 0.0, 0.0]))
    kept2, nrej2, oob2 = ballistic.estimation.reject_heavy_tails(sparse, sigma=0.1)
    assert oob2 and nrej2 == 0 and len(kept2) == len(sparse)


@pytest.mark.nrl
@pytest.mark.skipif(not BALLISTIC_AVAILABLE, reason="ballistic not built")
def test_make_drag_environment_wrapper(leo_state):
    """Regression: the one-call drag enabler once raised NameError (unbound
    _core) unnoticed because no test called the wrapper."""

    epoch, pos, vel = leo_state
    beta = ballistic.BetaProfile1D.constant(256.4)
    env, diag = ballistic.make_drag_environment(
        epoch, pos, vel, 120.0, ballistic.TrajectoryConfig(), beta,
        altitude_grid_km=[300.0, 400.0, 500.0, 600.0], max_iter=1)
    assert env is not None
