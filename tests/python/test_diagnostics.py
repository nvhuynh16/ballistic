"""The diagnostics channel reaches Python as typed warnings and errors.

The library continues with an approximation only after saying so: each
diagnostic arrives as a ballistic.*Warning, deliberate errors are
ballistic.BallisticError subclasses, and set_strict turns warnings into
errors.
"""

import warnings

import numpy as np
import pytest

import ballistic


def test_errors_derive_from_runtime_error():
    for cls in (ballistic.DataCoverageError, ballistic.FrameError,
                ballistic.InputPlausibilityError, ballistic.ConfigError,
                ballistic.RemovedApiError):
        assert issubclass(cls, ballistic.BallisticError)
    assert issubclass(ballistic.BallisticError, RuntimeError)


def test_each_category_arrives_as_its_warning_class():
    for name, cls in ballistic.errors.WARNING_CLASSES.items():
        assert issubclass(cls, ballistic.BallisticWarning)
        assert issubclass(cls, RuntimeWarning)
        with pytest.warns(cls, match="probe " + name):
            ballistic._core._report_diagnostic(name, "probe " + name)


def test_keyed_diagnostics_arrive_once():
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        for _ in range(3):
            ballistic._core._report_diagnostic("DataCoverage", "once", "test-key")
    assert len(got) == 1


def test_set_strict_turns_warnings_into_errors():
    with warnings.catch_warnings():
        ballistic.set_strict(True)
        with pytest.raises(ballistic.DataSkippedWarning):
            ballistic._core._report_diagnostic("DataSkipped", "strict")
        # A ConfigHint is advice about the caller's own choice: still a warning.
        with pytest.warns(ballistic.ConfigHintWarning):
            ballistic._core._report_diagnostic("ConfigHint", "strict")
    # catch_warnings restored the filters


@pytest.mark.nrl
def test_trajectory_note_is_a_data_coverage_warning():
    """A drag propagation into forecast weather (committed CSV: measured
    through 3 November 2026) warns through the channel, as the class users can
    filter on."""

    t0 = ballistic.Epoch.from_utc(2026, 11, 3, 20, 0, 0)
    pos = np.array([6878137.0, 0.0, 0.0])
    vel = np.array([0.0, 7612.6, 0.0])
    cfg = ballistic.TrajectoryConfig(gravity_degree=8)
    env, _ = ballistic.make_drag_environment(
        t0, pos, vel, 0.0, cfg, beta=ballistic.BetaProfile1D.constant(100.0))
    cfg = cfg.replace(drag_env=env)
    tr = ballistic.Trajectory(ballistic.State.from_gcrs(t0, pos, vel), config=cfg)   # circular, inertial
    with pytest.warns(ballistic.DataCoverageWarning, match="forecast"):
        tr.propagate_for(6 * 3600.0)


@pytest.mark.nrl
def test_weather_edge_is_a_data_coverage_error():
    t0 = ballistic.Epoch.from_utc(2026, 11, 3, 20, 0, 0)
    pos = np.array([6878137.0, 0.0, 0.0])
    vel = np.array([0.0, 7612.6, 0.0])
    cfg = ballistic.TrajectoryConfig(gravity_degree=8)
    env, _ = ballistic.make_drag_environment(
        t0, pos, vel, 0.0, cfg, beta=ballistic.BetaProfile1D.constant(100.0),
        allow_predicted_weather=False)
    cfg = cfg.replace(drag_env=env)
    tr = ballistic.Trajectory(ballistic.State.from_gcrs(t0, pos, vel), config=cfg)   # circular, inertial
    with pytest.raises(ballistic.DataCoverageError, match="refused"):
        tr.propagate_for(6 * 3600.0)
