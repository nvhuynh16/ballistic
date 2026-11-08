"""Python smoke tests for DragEnvironment bindings.

Bindings covered: DragEnvironment, SpaceWeatherHistory, TimeWindow,
BetaProfile1D, IterationDiag, and the three from_models_* generators.

Sticks to factory-method construction (matches EmpiricalAcceleration's pattern).
"""

import numpy as np
import pytest

import ballistic


@pytest.fixture(scope="module", autouse=True)
def initialize_ballistic():
    ballistic.initialize()


@pytest.fixture
def t0():
    return ballistic.Epoch.from_utc(2024, 6, 15, 12, 0, 0)


@pytest.fixture
def quiet_sw(t0):
    w = ballistic.drag.WeatherConditions.quiet()
    window = ballistic.drag.TimeWindow(t0 - 86400.0, t0 + 86400.0)
    return ballistic.SpaceWeatherHistory.from_constant(w, window)


# ----------------------------------------------------------------------------
# SpaceWeatherHistory / TimeWindow
# ----------------------------------------------------------------------------

class TestSpaceWeatherHistory:
    def test_from_constant_at_midpoint_returns_quiet(self, t0, quiet_sw):
        w = quiet_sw.at(t0)
        assert w.f107_daily == pytest.approx(70.0)
        assert w.ap_daily == pytest.approx(4.0)

    def test_empty_history_at_throws(self, t0):
        sw = ballistic.SpaceWeatherHistory()
        with pytest.raises(Exception):
            sw.at(t0)


# ----------------------------------------------------------------------------
# BetaProfile1D
# ----------------------------------------------------------------------------

class TestBetaProfile1D:
    def test_constant_factory(self):
        b = ballistic.BetaProfile1D.constant(45.45)
        assert not b.empty()

    def test_array_constructor(self):
        b = ballistic.BetaProfile1D([200.0, 400.0, 600.0], [50.0, 60.0, 70.0])
        assert not b.empty()

    def test_empty_default(self):
        b = ballistic.BetaProfile1D()
        assert b.empty()


# ----------------------------------------------------------------------------
# DragEnvironment basic construction
# ----------------------------------------------------------------------------

class TestDragEnvironment:
    def test_default_constructed_has_no_drag(self):
        env = ballistic.drag.DragEnvironment()
        assert not env.has_drag()


# ----------------------------------------------------------------------------
# Generators (round-trip with NRLMSIS)
# ----------------------------------------------------------------------------

@pytest.mark.nrl
class TestFromModelsSnapshot:
    def test_returns_drag_capable_env(self, t0, quiet_sw):
        beta = ballistic.BetaProfile1D.constant(100.0)
        env = ballistic.drag.from_models_snapshot(t0, 0.0, 0.0, quiet_sw, beta)
        assert env.has_drag()

    def test_density_at_400km_is_physical(self, t0, quiet_sw):
        beta = ballistic.BetaProfile1D.constant(100.0)
        env = ballistic.drag.from_models_snapshot(t0, 0.0, 0.0, quiet_sw, beta)
        pos_ecef = np.array([6.778137e6, 0.0, 0.0])  # ~400 km equatorial
        s = env.evaluate(pos_ecef, t0)
        # LEO 400 km density is ~1e-12 kg/m^3 (within a couple orders).
        assert 1e-14 < s.rho_kg_m3 < 1e-10

    def test_beta_passthrough(self, t0, quiet_sw):
        beta = ballistic.BetaProfile1D.constant(75.5)
        env = ballistic.drag.from_models_snapshot(t0, 0.0, 0.0, quiet_sw, beta)
        pos_ecef = np.array([6.778137e6, 0.0, 0.0])
        s = env.evaluate(pos_ecef, t0)
        assert s.beta_kg_m2 == pytest.approx(75.5, rel=1e-6)


@pytest.mark.nrl
class TestFromModelsAlong:
    def test_returns_drag_capable_env_with_diag(self, t0, quiet_sw):
        beta = ballistic.BetaProfile1D.constant(100.0)
        epochs = [t0, t0 + 1800.0, t0 + 3600.0]
        positions = [
            np.array([6.778137e6, 0.0, 0.0]),
            np.array([0.0, 6.778137e6, 0.0]),
            np.array([-6.778137e6, 0.0, 0.0]),
        ]
        env, diag = ballistic.drag.from_models_along(
            epochs, positions, quiet_sw, beta)
        assert env.has_drag()
        assert diag.iterations == 1
        assert diag.converged


@pytest.mark.nrl
class TestFromModels4D:
    def test_small_grid_builds_env(self, t0, quiet_sw):
        beta = ballistic.BetaProfile1D.constant(100.0)
        alt = [300.0, 400.0, 500.0]
        lat = [0.0, 30.0]
        lon = [0.0, 90.0]
        epochs = [t0, t0 + 3600.0]
        env = ballistic.drag.from_models_4d(alt, lat, lon, epochs, quiet_sw, beta)
        assert env.has_drag()


# ----------------------------------------------------------------------------
# Standard altitude grid
# ----------------------------------------------------------------------------

class TestStandardGrid:
    def test_grid_is_100_to_1500_step_25(self):
        grid = ballistic.drag.standard_leo_altitude_grid_km()
        assert grid[0] == pytest.approx(100.0)
        assert grid[-1] == pytest.approx(1500.0)
        assert all(grid[i + 1] - grid[i] == pytest.approx(25.0)
                   for i in range(len(grid) - 1))


# ----------------------------------------------------------------------------
# TrajectoryConfig.drag_env round-trip
# ----------------------------------------------------------------------------

@pytest.mark.nrl
class TestTrajectoryConfigDragEnv:
    def test_drag_env_propagates_through_trajectory(self, t0, quiet_sw):
        beta = ballistic.BetaProfile1D.constant(100.0)
        env = ballistic.drag.from_models_snapshot(t0, 0.0, 0.0, quiet_sw, beta)

        config = ballistic.TrajectoryConfig(
            gravity_degree=4,
            drag_env=env,
        )

        pos = np.array([6.778137e6, 0.0, 0.0])
        vel = np.array([0.0, 7500.0, 0.0])
        traj = ballistic.Trajectory(ballistic.State.from_itrf(t0, pos, vel), config=config)

        # Short propagation through the env path should complete.
        traj.propagate_for(60.0)
        result = traj(30.0)
        assert result is not None
        # Position should have moved significantly.
        new_pos = np.array(result.position)
        assert np.linalg.norm(new_pos - pos) > 100000.0


# ----------------------------------------------------------------------------
# DragEnvironment.without_wind (G1)
# ----------------------------------------------------------------------------

@pytest.mark.nrl
class TestWithoutWind:
    def test_without_wind_zeros_wind_column(self, t0, quiet_sw):
        # Build a low-LEO env where HWM14 wind is non-trivial.
        beta = ballistic.BetaProfile1D.constant(100.0)
        env_with_wind = ballistic.drag.from_models_snapshot(
            t0, 0.0, 0.0, quiet_sw, beta,
            altitude_grid_km=[200.0, 250.0, 300.0])
        env_no_wind = env_with_wind.without_wind()

        # Both envs should still have drag (density + beta preserved).
        assert env_with_wind.has_drag()
        assert env_no_wind.has_drag()

        # Density should be identical at the same sample point.
        pos_ecef = np.array([6.378137e6 + 250e3, 0.0, 0.0])  # ~250 km alt
        s_with = env_with_wind.evaluate(pos_ecef, t0)
        s_no = env_no_wind.evaluate(pos_ecef, t0)
        assert s_with.rho_kg_m3 == pytest.approx(s_no.rho_kg_m3, rel=1e-12)
        assert s_with.beta_kg_m2 == pytest.approx(s_no.beta_kg_m2, rel=1e-12)

        # Wind in without_wind env should be exactly zero.
        wind_no = np.array(s_no.wind_ecef_m_s)
        assert np.allclose(wind_no, 0.0, atol=1e-15)

        # Wind in original env may be zero (snapshot at equator with quiet SW
        # is often near zero) but it should be derivable from the env. We
        # don't assert it's non-zero — just that without_wind() honors the
        # contract regardless.


# ----------------------------------------------------------------------------
# from_accelerometer_density (G2)
# ----------------------------------------------------------------------------

class TestFromAccelerometerDensity:
    def test_constant_density_returns_that_density_at_any_altitude(self, t0):
        # Three samples at constant ρ = 1e-11 kg/m^3 (typical ~400 km value).
        epochs = [t0, t0 + 1800.0, t0 + 3600.0]
        rho_const = 1.0e-11
        densities = [rho_const, rho_const, rho_const]
        beta = ballistic.BetaProfile1D.constant(75.0)
        alt_grid = [250.0, 300.0, 350.0, 400.0, 450.0]

        env = ballistic.drag.from_accelerometer_density(
            epochs, densities, beta, alt_grid)
        assert env.has_drag()

        # Sample at multiple altitudes — all should return rho_const since
        # every altitude row in the 2D profile holds the same ρ.
        for alt_m in (260e3, 350e3, 440e3):
            pos = np.array([6.378137e6 + alt_m, 0.0, 0.0])
            s = env.evaluate(pos, t0 + 900.0)  # mid-arc
            assert s.rho_kg_m3 == pytest.approx(rho_const, rel=1e-6)
            assert s.beta_kg_m2 == pytest.approx(75.0, rel=1e-9)

    def test_time_varying_density_interpolates_linearly(self, t0):
        # Linear ramp in density across the arc.
        rho0, rho1 = 1.0e-11, 2.0e-11
        epochs = [t0, t0 + 1000.0]
        densities = [rho0, rho1]
        beta = ballistic.BetaProfile1D.constant(75.0)
        env = ballistic.drag.from_accelerometer_density(
            epochs, densities, beta, [250.0, 350.0])

        # At midpoint, density should be ~midway.
        pos = np.array([6.378137e6 + 300e3, 0.0, 0.0])
        s_mid = env.evaluate(pos, t0 + 500.0)
        expected = 0.5 * (rho0 + rho1)
        assert s_mid.rho_kg_m3 == pytest.approx(expected, rel=1e-6)

    def test_rejects_non_positive_density(self, t0):
        epochs = [t0, t0 + 60.0]
        beta = ballistic.BetaProfile1D.constant(75.0)
        with pytest.raises(Exception):
            ballistic.drag.from_accelerometer_density(
                epochs, [1e-11, 0.0], beta, [200.0, 400.0])

    def test_rejects_length_mismatch(self, t0):
        beta = ballistic.BetaProfile1D.constant(75.0)
        with pytest.raises(Exception):
            ballistic.drag.from_accelerometer_density(
                [t0, t0 + 60.0, t0 + 120.0],
                [1e-11, 2e-11],
                beta, [200.0, 400.0])

    def test_uses_default_altitude_grid_when_omitted(self, t0):
        epochs = [t0, t0 + 60.0]
        beta = ballistic.BetaProfile1D.constant(75.0)
        env = ballistic.drag.from_accelerometer_density(
            epochs, [1e-11, 1.1e-11], beta)
        assert env.has_drag()


# ----------------------------------------------------------------------------
# refine_beta_from_ea (the Python binding of the C++ EA -> beta entry point)
# ----------------------------------------------------------------------------

@pytest.mark.nrl
class TestRefineBetaFromEa:
    """The C++ test_refine_beta_from_ea.cpp covers the algorithmic
    contract; this smoke test just verifies the Python binding is
    callable and returns a DragEnvironment whose β column is closer to
    the synthetic truth than the prior was."""

    def test_recovers_β_closer_to_truth_than_prior(self, t0, quiet_sw):
        # CHAMP-band prior with constant β = 50 (wrong).
        beta_prior_value = 50.0
        beta_truth = 100.0
        alt_grid = [200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]
        beta_prior = ballistic.BetaProfile1D.constant(beta_prior_value)
        prior_env = ballistic.drag.from_models_snapshot(
            t0, 0.0, 0.0, quiet_sw, beta_prior, alt_grid)

        # Sample a short drag-free arc to use as the trajectory.
        cfg = ballistic.TrajectoryConfig(
            gravity_degree=4,
            enable_third_body=False,
            enable_relativistic=False,
        )
        pos = np.array([6.378137e6 + 350e3, 0.0, 0.0])  # ~350 km alt
        vel = np.array([0.0, 7700.0, 0.0])                # inertial, near-circular
        traj = ballistic.Trajectory(ballistic.State.from_gcrs(t0, pos, vel), config=cfg)
        # step() accumulates history; propagate() does not.
        for _ in range(10):
            traj.step()
        history = traj.history()
        assert len(history) >= 4

        # Build a synthetic EA mimicking the drag at β_truth, evaluated
        # against the prior's density. The decomposition should then
        # recover β_truth (modulo ECEF-vs-GCRS frame artifacts).
        entries = []
        for h in history:
            pos_ecef = np.array(h.position)
            vel_ecef = np.array(h.velocity)
            sample = prior_env.evaluate(pos_ecef, h.epoch)
            v_rel = vel_ecef - np.array(sample.wind_ecef_m_s)
            v_mag = np.linalg.norm(v_rel)
            if v_mag < 1.0 or sample.rho_kg_m3 <= 0.0:
                continue
            drag_acc_ecef = (-0.5 * sample.rho_kg_m3 * v_mag / beta_truth) * v_rel
            t_k = h.epoch - t0
            entries.append(ballistic.estimation.EmpiricalAccelEntry(t_k, drag_acc_ecef))

        assert len(entries) >= 3
        ea = ballistic.EmpiricalAcceleration.from_entries(
            entries, frame=ballistic.EmpiricalAccelFrame.ITRF)

        refined = ballistic.estimation.refine_beta_from_ea(prior_env, ea, history, t0)

        # The refined env should now evaluate β closer to truth than the
        # prior's 50. Sample at the orbit altitude.
        sample_pos = np.array([6.378137e6 + 350e3, 0.0, 0.0])
        refined_beta = refined.evaluate(sample_pos, t0).beta_kg_m2
        prior_beta_eval = prior_env.evaluate(sample_pos, t0).beta_kg_m2
        assert prior_beta_eval == pytest.approx(beta_prior_value, rel=1e-6)
        # Should be closer to truth than the prior; exact match isn't
        # expected because of the GCRS↔ECEF rotation tiny over 120 s but
        # non-zero, and the altitude binning has only ~1 bin populated.
        assert abs(refined_beta - beta_truth) < abs(prior_beta_eval - beta_truth)


# ----------------------------------------------------------------------------
# save_drag_env + load_drag_env (NPZ persistence, 1D variants)
# ----------------------------------------------------------------------------

class TestDragEnvPersistence:
    @pytest.mark.nrl
    def test_1d_round_trip_at_sample_points(self, t0, quiet_sw, tmp_path):
        # Build a snapshot env (1D density + 1D wind + 1D beta).
        alt_grid = [200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]
        beta = ballistic.BetaProfile1D.constant(256.4)
        original = ballistic.drag.from_models_snapshot(
            t0, 0.0, 0.0, quiet_sw, beta, alt_grid)
        assert original.has_drag()

        # Save and load.
        path = tmp_path / "env_snapshot.npz"
        ballistic.drag.save_drag_env(original, str(path))
        loaded = ballistic.drag.load_drag_env(str(path))
        assert loaded.has_drag()

        # evaluate() at sample points must match within float64 precision.
        for alt_m in (220e3, 350e3, 480e3):
            pos = np.array([6.378137e6 + alt_m, 0.0, 0.0])
            s_orig = original.evaluate(pos, t0)
            s_load = loaded.evaluate(pos, t0)
            assert s_load.rho_kg_m3 == pytest.approx(s_orig.rho_kg_m3, rel=1e-12)
            assert s_load.beta_kg_m2 == pytest.approx(s_orig.beta_kg_m2, rel=1e-12)
            assert np.allclose(s_load.wind_ecef_m_s, s_orig.wind_ecef_m_s, atol=1e-12)

    @pytest.mark.nrl
    def test_no_wind_env_round_trips(self, t0, quiet_sw, tmp_path):
        beta = ballistic.BetaProfile1D.constant(100.0)
        env = ballistic.drag.from_models_snapshot(
            t0, 0.0, 0.0, quiet_sw, beta, [250.0, 350.0])
        env_no_wind = env.without_wind()

        path = tmp_path / "env_no_wind.npz"
        ballistic.drag.save_drag_env(env_no_wind, str(path))
        loaded = ballistic.drag.load_drag_env(str(path))

        pos = np.array([6.378137e6 + 300e3, 0.0, 0.0])
        s_load = loaded.evaluate(pos, t0)
        assert np.allclose(s_load.wind_ecef_m_s, 0.0, atol=1e-15)

    @pytest.mark.nrl
    def test_2d_from_models_along_round_trip(self, t0, quiet_sw, tmp_path):
        # 2D env (DensityProfile2D + WindProfile2D_ECEF from
        # from_models_along) round-trips through save / load.
        beta = ballistic.BetaProfile1D.constant(100.0)
        epochs = [t0, t0 + 1800.0, t0 + 3600.0]
        positions = [
            np.array([6.778137e6, 0.0, 0.0]),
            np.array([0.0, 6.778137e6, 0.0]),
            np.array([-6.778137e6, 0.0, 0.0]),
        ]
        original, _ = ballistic.drag.from_models_along(
            epochs, positions, quiet_sw, beta,
            [200.0, 300.0, 400.0, 500.0])
        assert original.has_drag()

        path = tmp_path / "env_2d.npz"
        ballistic.drag.save_drag_env(original, str(path))
        loaded = ballistic.drag.load_drag_env(str(path))
        assert loaded.has_drag()

        # evaluate() at the first sample point must match.
        pos = np.array([6.378137e6 + 350e3, 0.0, 0.0])
        s_orig = original.evaluate(pos, t0 + 60.0)
        s_load = loaded.evaluate(pos, t0 + 60.0)
        assert s_load.rho_kg_m3 == pytest.approx(s_orig.rho_kg_m3, rel=1e-12)
        assert s_load.beta_kg_m2 == pytest.approx(s_orig.beta_kg_m2, rel=1e-12)
        assert np.allclose(s_load.wind_ecef_m_s, s_orig.wind_ecef_m_s, atol=1e-9)

    @pytest.mark.nrl
    def test_isa_splice_replaces_sub_80km_density_1d(self, t0, quiet_sw):
        # Build a 1D env whose altitude grid straddles the 80 km splice.
        beta = ballistic.BetaProfile1D.constant(100.0)
        env = ballistic.drag.from_models_snapshot(
            t0, 0.0, 0.0, quiet_sw, beta,
            [40.0, 60.0, 80.0, 100.0, 200.0, 400.0])
        spliced = ballistic.drag.extend_density_below(env, floor_km=80.0)

        # At 50 km (below floor): USSA76 Table I, 1.0269e-3 kg/m³ (a knot).
        pos_50 = np.array([6.378137e6 + 50e3, 0.0, 0.0])
        s50 = spliced.evaluate(pos_50, t0)
        assert s50.rho_kg_m3 == pytest.approx(1.0269e-3, rel=1e-4)

        # At 200 km (above floor): NRLMSIS21 density (~1e-10 kg/m³ range).
        pos_200 = np.array([6.378137e6 + 200e3, 0.0, 0.0])
        s200_orig = env.evaluate(pos_200, t0)
        s200_spliced = spliced.evaluate(pos_200, t0)
        assert s200_spliced.rho_kg_m3 == pytest.approx(s200_orig.rho_kg_m3, rel=1e-12)

    @pytest.mark.nrl
    def test_isa_splice_2d_density_grid(self, t0, quiet_sw):
        # 2D env from from_models_along. After splice, every epoch row's
        # sub-80km altitudes should hold the USSA density.
        beta = ballistic.BetaProfile1D.constant(100.0)
        env2d, _ = ballistic.drag.from_models_along(
            [t0, t0 + 1800.0, t0 + 3600.0],
            [np.array([6.778137e6, 0.0, 0.0]),
             np.array([0.0, 6.778137e6, 0.0]),
             np.array([-6.778137e6, 0.0, 0.0])],
            quiet_sw, beta,
            [50.0, 80.0, 120.0, 200.0])
        spliced = ballistic.drag.extend_density_below(env2d, 80.0)

        # Sample AT an altitude knot below the splice floor. Cubic spline
        # exactly recovers the knot value, so the per-epoch result is
        # USSA-only (no upper-knot mixing). At off-knot altitudes the
        # spline interpolation pulls in the upper neighbor's value which
        # IS epoch-varying via NRLMSIS21.
        pos_50 = np.array([6.378137e6 + 50e3, 0.0, 0.0])
        s_t0 = spliced.evaluate(pos_50, t0)
        s_mid = spliced.evaluate(pos_50, t0 + 1800.0)
        s_end = spliced.evaluate(pos_50, t0 + 3600.0)
        assert s_t0.rho_kg_m3 == pytest.approx(s_mid.rho_kg_m3, rel=1e-9)
        assert s_t0.rho_kg_m3 == pytest.approx(s_end.rho_kg_m3, rel=1e-9)
        # USSA76 Table I at 50 km
        assert s_t0.rho_kg_m3 == pytest.approx(1.0269e-3, rel=1e-4)

    @pytest.mark.nrl
    def test_isa_splice_round_trips_through_npz(self, t0, quiet_sw, tmp_path):
        # ISA-spliced env preserves through save/load.
        beta = ballistic.BetaProfile1D.constant(100.0)
        env = ballistic.drag.from_models_snapshot(
            t0, 0.0, 0.0, quiet_sw, beta,
            [40.0, 60.0, 80.0, 120.0, 200.0])
        spliced = ballistic.drag.extend_density_below(env, 80.0)
        path = tmp_path / "env_spliced.npz"
        ballistic.drag.save_drag_env(spliced, str(path))
        loaded = ballistic.drag.load_drag_env(str(path))
        pos = np.array([6.378137e6 + 50e3, 0.0, 0.0])
        s_orig = spliced.evaluate(pos, t0)
        s_load = loaded.evaluate(pos, t0)
        assert s_load.rho_kg_m3 == pytest.approx(s_orig.rho_kg_m3, rel=1e-12)

    @pytest.mark.nrl
    def test_extend_below_standard_grid_adds_knots(self, t0, quiet_sw):
        # Trap USSA-MISLABEL: on a table starting at 100 km the old splice
        # changed nothing and evaluation below 100 km raised.
        beta = ballistic.BetaProfile1D.constant(100.0)
        env = ballistic.drag.from_models_snapshot(t0, 0.0, 0.0, quiet_sw, beta,
                                             [100.0, 150.0, 200.0, 400.0])
        with pytest.raises(Exception):
            env.evaluate(np.array([6.378137e6 + 30e3, 0.0, 0.0]), t0)
        ext = ballistic.drag.extend_density_below(env, floor_km=86.0)
        s30 = ext.evaluate(np.array([6.378137e6 + 30e3, 0.0, 0.0]), t0)
        assert s30.rho_kg_m3 == pytest.approx(1.8410e-2, rel=1e-4)   # USSA76 Table I
        with pytest.raises(ballistic.ConfigError):
            ballistic.drag.extend_density_below(env, floor_km=120.0)

    def test_splice_isa_density_below_is_removed(self):
        with pytest.raises(ballistic.RemovedApiError, match="extend_density_below"):
            ballistic.splice_isa_density_below

    def test_2d_from_accelerometer_density_round_trip(self, t0, tmp_path):
        # from_accelerometer_density produces a 2D-density / empty-wind env.
        # Round-trip must preserve the density time series.
        epochs = [t0, t0 + 60.0, t0 + 120.0]
        rhos = [1e-11, 1.2e-11, 1.4e-11]
        beta = ballistic.BetaProfile1D.constant(75.0)
        alt_grid = [250.0, 300.0, 350.0]

        original = ballistic.drag.from_accelerometer_density(
            epochs, rhos, beta, alt_grid)

        path = tmp_path / "env_accel.npz"
        ballistic.drag.save_drag_env(original, str(path))
        loaded = ballistic.drag.load_drag_env(str(path))

        # Sample at mid-arc — both density (interpolated linearly in
        # time) and beta should match.
        pos = np.array([6.378137e6 + 300e3, 0.0, 0.0])
        s_orig = original.evaluate(pos, t0 + 30.0)
        s_load = loaded.evaluate(pos, t0 + 30.0)
        assert s_load.rho_kg_m3 == pytest.approx(s_orig.rho_kg_m3, rel=1e-12)
        assert s_load.beta_kg_m2 == pytest.approx(s_orig.beta_kg_m2, rel=1e-12)
        # Wind is empty in both.
        assert np.allclose(s_load.wind_ecef_m_s, 0.0, atol=1e-15)
        assert np.allclose(s_orig.wind_ecef_m_s, 0.0, atol=1e-15)

    @pytest.mark.parametrize("density_kind,wind_kind", [
        ("4d", "1d_enu"), ("4d", "2d_ecef"), ("1d", "2d_ecef"),
        ("1d", "4d_ecef"), ("2d", "4d_ecef"),
    ])
    def test_unsupported_density_wind_pairing_raises(self, density_kind, wind_kind):
        # The loader's factories would rebuild these with no wind at all;
        # save and load refuse instead and name the supported pairings.
        from ballistic import dragenv_io
        with pytest.raises(NotImplementedError, match="Supported pairings"):
            dragenv_io._check_pairing(density_kind, wind_kind, "save_drag_env")

    def test_load_refuses_4d_density_with_populated_1d_wind(self, tmp_path):
        # A file pairing a 4D density with a populated 1D wind is refused
        # rather than loaded windless.
        path = tmp_path / "env_4d_1dwind.npz"
        np.savez(path,
                 density_kind=np.array("4d"), wind_kind=np.array("1d_enu"),
                 beta_kind=np.array("1d"),
                 beta_altitude_km=np.array([0.0]), beta_kg_m2=np.array([100.0]),
                 density_altitude_km=np.array([200.0, 300.0]),
                 density_latitude_deg=np.array([-10.0, 10.0]),
                 density_longitude_deg=np.array([0.0, 90.0]),
                 density_epochs_mjd_utc=np.array([60000.0, 60001.0]),
                 density_grid=np.full(16, 1e-11),
                 wind_altitude_km=np.array([200.0, 300.0]),
                 wind_meridional_ms=np.array([10.0, 20.0]),
                 wind_zonal_ms=np.array([30.0, 40.0]))
        with pytest.raises(NotImplementedError, match="4d density with a 1d_enu wind"):
            ballistic.drag.load_drag_env(str(path))

    @pytest.mark.nrl
    def test_4d_round_trip(self, t0, quiet_sw, tmp_path):
        # 4D-density + 4D-wind env from from_models_4d round-trips

        # through save / load. Half-precision storage caps the round-trip
        # tolerance at ~1e-3 rel.
        beta = ballistic.BetaProfile1D.constant(100.0)
        alt_grid = [200.0, 300.0, 400.0]
        lat_grid = [-30.0, 0.0, 30.0]
        lon_grid = [0.0, 90.0, 180.0]
        epochs = [t0, t0 + 1800.0]

        original = ballistic.drag.from_models_4d(
            alt_grid, lat_grid, lon_grid, epochs, quiet_sw, beta)
        assert original.has_drag()

        path = tmp_path / "env_4d.npz"
        ballistic.drag.save_drag_env(original, str(path))
        loaded = ballistic.drag.load_drag_env(str(path))
        assert loaded.has_drag()

        # evaluate() at a few sample points must agree within half tolerance.
        for alt_m, lat_deg, lon_deg, dt_s in [
            (250e3, 10.0,  45.0,  60.0),
            (350e3, -15.0, 100.0, 600.0),
            (380e3, 25.0,  -30.0, 1500.0),  # lon wraps to grid
        ]:
            r = 6.378137e6 + alt_m
            lat_rad = np.radians(lat_deg)
            lon_rad = np.radians(lon_deg)
            pos = np.array([
                r * np.cos(lat_rad) * np.cos(lon_rad),
                r * np.cos(lat_rad) * np.sin(lon_rad),
                r * np.sin(lat_rad),
            ])
            s_orig = original.evaluate(pos, t0 + dt_s)
            s_load = loaded.evaluate(pos, t0 + dt_s)
            assert s_load.rho_kg_m3 == pytest.approx(s_orig.rho_kg_m3, rel=2e-3)
            assert s_load.beta_kg_m2 == pytest.approx(s_orig.beta_kg_m2, rel=1e-12)
            assert np.allclose(s_load.wind_ecef_m_s, s_orig.wind_ecef_m_s, atol=0.5)
