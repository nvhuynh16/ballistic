"""T5: the orbital semi-synthetic fixture — real GRACE-FO truth.

Gates what the pre-registered run measured (default tier, ~4 s): (1) the default force model propagates REAL GRACE-FO to < 0.5 m
over 600 s with no EA (measured 0.157 m — real-dynamics validation);
(2) with a regime-appropriate GM prior the smoother is self-consistent
(NIS in band) and the EA is harmless (measured NIS 0.99, 0.15 m).
With the all-auto grid (glide-scale sigma_a) the returned EA is 61 m off,
so the cell also requires the grid-edge warning to fire there.
"""

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from conftest import require_data  # noqa: E402
from orbital_fixture import SP3_FILE  # noqa: E402

pytestmark = require_data(
    SP3_FILE, "GFZ RSO GRACE-FO SP3 — see tests/data/sp3/grace_fo/README.md")


# Measured wall 3.8 s — under the registered 15 s threshold, so this runs
# in the DEFAULT tier (the pre-stated rule for this fixture).
def test_orbital_fixture_real_truth():
    import ballistic
    from orbital_fixture import build_scenario, _reprop_rmse

    sc, truth = build_scenario(ballistic)

    rmse_dyn = _reprop_rmse(ballistic, sc, truth, None)
    print(f"\n  no-EA vs real SP3: {rmse_dyn:.3f} m")
    assert rmse_dyn < 0.5, rmse_dyn

    cfg = ballistic.estimation.EASmootherConfig()
    cfg.sigma_a = 1e-6          # orbital-regime prior (regime-appropriate arm)
    cfg.tau_s = 600.0
    r = ballistic.estimation.estimate_ea_smoother(
        sc, ballistic.estimation.SensorMask.all_on(sc),
        ballistic.TrajectoryConfig()._to_core(), cfg)
    assert r.error_message == ""
    rmse_ea = _reprop_rmse(ballistic, sc, truth, r.empirical_accel)
    print(f"  orbital-prior EA: NIS {r.nis:.2f}  reprop {rmse_ea:.3f} m")
    assert 0.3 < r.nis < 3.0, r.nis
    assert rmse_ea < 0.5, rmse_ea
    honored = dict(r.sigma_honored)
    assert honored["radar0_range"] < 2.0 * sc.radars[0].sigma_range

    # The all-auto path is the trap (measured 61 m vs 0.157 m

    # no-EA, NIS clean). It must now SAY SO — the warning is the only signal.
    r_auto = ballistic.estimation.estimate_ea_smoother(
        sc, ballistic.estimation.SensorMask.all_on(sc),
        ballistic.TrajectoryConfig()._to_core(), ballistic.estimation.EASmootherConfig())
    assert r_auto.error_message == ""
    joined = " ".join(r_auto.warnings)
    print(f"  all-auto: sigma_a={r_auto.sigma_a_used} tau={r_auto.tau_used} "
          f"warnings={len(r_auto.warnings)}")
    assert "BOUNDARY" in joined, (
        f"orbital all-auto must warn about the grid edge; got: {joined}")
