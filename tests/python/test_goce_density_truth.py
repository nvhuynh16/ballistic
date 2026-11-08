"""T4: NRLMSIS 2.1 vs measured GOCE density stays in the band.

Measured (pre-registered): median ratio 0.937, scatter 9.3%

over 1575 points. Default tier runs a thin subsample (~1 day, 900 s
stride); BALLISTIC_HEAVY=1 runs the full registered measurement. Gate is
the measured band with the registered ±0.25 margin — the library's first
absolute-density regression anchor.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from conftest import heavy_enabled, require_data  # noqa: E402
from goce_density_truth import (DENSITY_FILE, HAVE_FULL_MONTH,  # noqa: E402
                                measure)

pytestmark = require_data(
    DENSITY_FILE, "python scripts/download_goce_density.py "
                  "(or use the committed 2013-10-15 slice)")


def _check(stats, n_min):
    assert stats["n"] >= n_min, stats
    assert 0.70 <= stats["median"] <= 1.20, stats
    assert stats["rms_scatter"] < 0.25, stats


@pytest.mark.nrl
def test_nrlmsis_vs_goce_thin():
    _check(measure(days=("2013-10-15",), stride_s=900.0), n_min=50)


@pytest.mark.skipif(not heavy_enabled(), reason="BALLISTIC_HEAVY!=1")
@pytest.mark.skipif(not HAVE_FULL_MONTH,
                    reason="full month absent (committed slice only) — "
                           "python scripts/download_goce_density.py")
@pytest.mark.nrl
def test_nrlmsis_vs_goce_full():
    stats = measure(days=("2013-10-01", "2013-10-15"), stride_s=60.0)
    _check(stats, n_min=1000)
    # The registered band around the measured median 0.937.
    assert abs(stats["median"] - 0.937) < 0.25, stats
