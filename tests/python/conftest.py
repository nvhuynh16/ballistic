"""
Shared pytest fixtures for Ballistic Python tests.

Provides session-scoped initialization and common test state (LEO orbit).

The whole suite runs against a hermetic data directory: BALLISTIC_DATA_DIR
(every platform) is pointed at a session temp dir seeded with the
checked-in, frozen tests/data/eop/finals2000A.all and nothing else. EOP loads
deterministically via auto-discovery; leap seconds stay embedded regardless of
what the user's real ~/.local/share/ballistic contains (Loader().timescale()
caching leap-seconds.list there must not change test results).

The NRLMSIS 2.1 / HWM14 plugin (model-based drag) is not part of the package:
tests marked ``nrl`` use the one built in this tree (build/nrl/, CMake target
ballistic_nrl) through BALLISTIC_NRL_MODELS, or the file that variable already
names, and are skipped when there is none (CI's no-plugin job, the wheel test).
"""

import os
import shutil
import sys
from pathlib import Path

import pytest
import numpy as np

try:
    import ballistic
    BALLISTIC_AVAILABLE = True
except ImportError:
    BALLISTIC_AVAILABLE = False

_PINNED_EOP = Path(__file__).parents[1] / "data" / "eop" / "finals2000A.all"
_REPO = Path(__file__).resolve().parents[2]
_NRL_FILE = {"win32": "ballistic_nrl.dll", "darwin": "ballistic_nrl.dylib"}.get(
    sys.platform, "ballistic_nrl.so")


def _built_nrl_plugin():
    """The plugin built in this tree (single- or multi-config generator), or None."""
    for p in (_REPO / "build" / "nrl" / _NRL_FILE,
              _REPO / "build" / "nrl" / "Release" / _NRL_FILE):
        if p.is_file():
            return p
    return None


# Before any test touches drag: the library looks the plugin up on first use.
if not os.environ.get("BALLISTIC_NRL_MODELS") and _built_nrl_plugin() is not None:
    os.environ["BALLISTIC_NRL_MODELS"] = str(_built_nrl_plugin())


@pytest.fixture(scope="session", autouse=True)
def initialize_ballistic(tmp_path_factory):
    """Initialize the ballistic library once, against a hermetic data dir."""
    if not BALLISTIC_AVAILABLE:
        return
    data_home = tmp_path_factory.mktemp("ballistic_data_home")
    app_dir = data_home / "ballistic"
    app_dir.mkdir()
    shutil.copy(_PINNED_EOP, app_dir / "finals2000A.all")
    os.environ["BALLISTIC_DATA_DIR"] = str(app_dir)
    result = ballistic.initialize()
    assert result.eop_loaded and result.eop_auto_discovered, (
        "hermetic fixture failed to pin EOP: " + str(result)
    )

    # Load the COMMITTED space-weather CSV for the whole session. The weather
    # store is a C++ global singleton, so otherwise whichever test loaded it
    # first would decide the state every later test sees (order-dependent).
    # Loading it here is cheaper and safer than isolation machinery, and
    # every test sees the same pinned 1957-2041 table.
    _sw = _PINNED_EOP.parent.parent / "weather" / "SW-All.csv"
    if _sw.exists():
        ballistic._core.load_space_weather(str(_sw))


@pytest.fixture
def leo_state():
    """Standard LEO initial state (epoch, position, velocity)."""
    epoch = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)
    pos = np.array([6778137.0, 0.0, 0.0])
    vel = np.array([0.0, 7500.0, 0.0])
    return epoch, pos, vel


def heavy_enabled() -> bool:
    """Heavy pytest tier — set BALLISTIC_HEAVY=1 to enable."""
    import os
    return os.environ.get("BALLISTIC_HEAVY", "") == "1"


def require_data(path, hint: str):
    """Skip (never ERROR) when large untracked test data is absent.

    Most of tests/data is gitignored (only part of it is tracked), so a clean

    checkout lacks the SP3/density fixtures some gates need. Every reason
    string must name the exact reacquisition command — a skip that doesn't
    say how to un-skip is a silent hole.
    """
    import pytest
    from pathlib import Path
    return pytest.mark.skipif(
        not Path(path).exists(),
        reason=f"missing test data {Path(path).name} — reacquire: {hint}")


def pytest_configure(config):
    # BALLISTIC_STRICT_WARNINGS=1 turns every RuntimeWarning into an error, to
    # catch new library notes firing where they shouldn't. (A command-line
    # `-W error` would override the ignore below; later lines here win.)
    if os.environ.get("BALLISTIC_STRICT_WARNINGS"):
        config.addinivalue_line("filterwarnings", "error::RuntimeWarning")
    # The suite runs hermetically without the (downloaded, third-party)
    # FES2004 ocean-tide file, so every Trajectory would note that ocean tides
    # are skipped. Tests that need tides load them (test_ocean_tides).
    config.addinivalue_line(
        "filterwarnings", "ignore:.*ocean tides are on.*:RuntimeWarning")
    # A bare TrajectoryConfig earns one ConfigHint (radiation and drag off,
    # with their measured cost); most tests use bare configs on purpose.
    # Advice, never an error (also not under ballistic.set_strict).
    config.addinivalue_line(
        "filterwarnings", "ignore::ballistic.errors.ConfigHintWarning")
    # Tests that build model-based drag (NRLMSIS 2.1 / HWM14 plugin). Without
    # the plugin CI runs -m "not nrl", and the hook below skips them if they
    # are selected.
    config.addinivalue_line(
        "markers", "nrl: needs the NRLMSIS 2.1 / HWM14 plugin (model-based drag)")


def _nrl_plugin_available() -> bool:
    try:
        return ballistic.nrl_models().installed
    except (NameError, AttributeError):
        return False


def pytest_collection_modifyitems(config, items):
    if not BALLISTIC_AVAILABLE or _nrl_plugin_available():
        return
    skip = pytest.mark.skip(
        reason="the NRLMSIS 2.1 / HWM14 plugin is not available: build the CMake target "
               "ballistic_nrl (build/nrl/) or set BALLISTIC_NRL_MODELS to the plugin file")
    for item in items:
        if "nrl" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_diagnostic_dedup():
    """Keyed diagnostics (no EOP loaded, forecast weather, ...) are delivered
    once per process; each test starts with a clean slate so tests that
    expect such a warning see it whatever ran before."""
    try:
        import ballistic
        ballistic._core._reset_diagnostic_dedup()
    except (ImportError, AttributeError):
        pass
    yield
