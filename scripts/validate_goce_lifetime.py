"""GOCE's last ten days: predicted vs actual reentry, constant vs flow-regime Cd.

Predicts when the GOCE satellite fell through 80 km altitude and compares it
with the observed time (claim FR-GOCE-LIFETIME in docs/validation_claims.md).
It starts from the last state of the AIUB precise science orbit (PSO)
committed for 2013-11-01 (tests/data/sp3/goce_pso/, ~220 km, 9 days before
the end; SP3 is the standard precise-orbit text format) and flies the
library's recommended configuration down to 80 km, the altitude at which
USSTRATCOM (the U.S. Strategic Command, which tracks reentries) dated the
decay: 2013-11-11 00:16 UTC +- 1 min (ESA, "GOCE completes its mission"; ESA
Rocket Science blog).

Drag acceleration depends on the drag coefficient Cd (how strongly the air
pushes on the body per unit area) and the ballistic coefficient
β = m / (Cd·A) (mass over Cd times frontal area, kg/m²). Two drag models:

  constant   the registry's fitted GOCE β (its Cd on its frontal area), as in orbit;
  flow       FlowRegimeCd: Cd follows the flow regime. In thin air the
             molecules hit the body one by one ("free-molecular" flow); lower
             down the air behaves as a continuous fluid, with a transition in
             between, judged by the Knudsen number (mean free path of the air
             molecules divided by the body's size). GOCE is modeled as an
             equivalent sphere whose free-molecular Cd·A equals the fitted one
             at the start, so the two models agree in orbit and differ only
             where the flow leaves the free-molecular regime, below ~120 km.

The error is quoted as a fraction of the remaining lifetime. Uncertainty in
the air density dominates it (NRLMSIS 2.1, the atmosphere model, scatters by
about 10% around GOCE's measured density, claim GOCE-DENSITY, and does worse
in geomagnetic storms). The flow regime acts mainly in the last orbit; this
script measures how much the two models differ (29 min, claim
FR-GOCE-LIFETIME) rather than assuming it.

Inputs (all committed, defaults shown):
  --sp3 tests/data/sp3/goce_pso/goce_pso_13305_0001.sp3.gz   start = its last state
  --weather tests/data/weather/SW-All.csv   space weather: F10.7 solar flux and
                                            Ap geomagnetic index (CelesTrak)
  --eop tests/data/eop/finals2000A.all      Earth orientation parameters (IERS)
  --reference-length M   sphere diameter for the Knudsen number, meters
                         (default: the diameter of a disc with GOCE's frontal area)

Output: printed lines, one per drag model, with the predicted 80 km crossing,
its error in hours and as a percentage of the remaining lifetime, then the
difference between the models split into the part accrued above 120 km and
the part between 120 and 80 km. Exit status 1 if a run does not reach 80 km.

Usage:
    PYTHONPATH=./python uv run --with numpy python scripts/validate_goce_lifetime.py \\
        [--sp3 PATH] [--weather PATH] [--eop PATH] [--reference-length M]
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

import ballistic

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SP3 = REPO / "tests/data/sp3/goce_pso/goce_pso_13305_0001.sp3.gz"
DEFAULT_WEATHER = REPO / "tests/data/weather/SW-All.csv"
DEFAULT_EOP = REPO / "tests/data/eop/finals2000A.all"
# USSTRATCOM decay at the 80 km interface, as published by ESA.
ACTUAL_80KM = ballistic.Epoch.from_utc(2013, 11, 11, 0, 16, 0)

def frontal_diameter_m() -> float:
    """Diameter of a disc with GOCE's frontal area (the registry's drag area)."""
    return math.sqrt(4.0 * ballistic.satellite_properties.get("GOCE").drag_area_m2 / math.pi)


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser: --sp3, --weather, --eop, --reference-length."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sp3", default=str(DEFAULT_SP3), help="GOCE PSO SP3 file; the last state is the start")
    p.add_argument("--weather", default=str(DEFAULT_WEATHER), help="CelesTrak SW-All.csv (measured 2013)")
    p.add_argument("--eop", default=str(DEFAULT_EOP), help="IERS finals2000A file")
    p.add_argument("--reference-length", type=float, default=None,
                   help="equivalent sphere diameter for the Knudsen number [m] (default: the "
                        "diameter of GOCE's frontal area, 1.18 m)")
    return p


def fly(epoch, pos, vel, *, flow: bool, ref_length: float):
    """Propagate GOCE from one state until it falls through 80 km.

    Parameters: `epoch` (ballistic.Epoch), `pos`/`vel` (numpy (3,)
    Earth-fixed ITRF2008 position in m and velocity in m/s), `flow` (False =
    constant Cd, True = flow-regime Cd on an equivalent sphere) and
    `ref_length` (sphere diameter in m, used only when `flow`).

    The run stops at the 80 km crossing, or at the latest 3 days after the
    observed decay; the 120 km crossing is logged without stopping. Returns
    (result, note, wall_s): the propagate_until result (stopped_by, epoch,
    log of events), a one-line description of the drag model, and the wall
    time in seconds.
    """
    goce = ballistic.satellite_properties.get("GOCE")
    common = dict(itrf=ballistic.ITRFRealization.ITRF2008)
    if not flow:
        cfg = ballistic.recommended_config(ballistic.State.from_itrf(epoch, pos, vel), "GOCE", **common)
        note = f"constant Cd {goce.cd} on {goce.drag_area_m2} m^2"
    else:
        # Equivalent sphere: free-molecular Cd * A at the start = the fitted Cd * A.
        probe = ballistic.recommended_config(ballistic.State.from_itrf(epoch, pos, vel), "GOCE", reentry=ballistic.aero.AeroShape.SPHERE,
                                             reference_length_m=ref_length, drag_area_m2=1.0, **common)  # guard-exempt(SAT-LITERAL): unit probe area, only its Cd is read
        cd0 = probe.drag_env.evaluate(pos, epoch, vel).cd
        area = goce.cd * goce.drag_area_m2 / cd0
        cfg = ballistic.recommended_config(ballistic.State.from_itrf(epoch, pos, vel), "GOCE", reentry=ballistic.aero.AeroShape.SPHERE,
                                           reference_length_m=ref_length, drag_area_m2=area, **common)
        note = f"flow-regime sphere, A {area:.3f} m^2 (FM Cd {cd0:.3f} at start), D {ref_length:.2f} m"
    traj = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel), config=cfg)
    ev = ballistic.events.altitude_crossing(80e3, ballistic.events.Direction.Falling)
    # 120 km (logged): for a ~1 m body the regime bridge (Kn < 10) starts
    # near 130 km but changes Cd little above 120 km, so the regime acts
    # mostly after this point.
    ev120 = ballistic.events.altitude_crossing(120e3, ballistic.events.Direction.Falling, False)
    t_wall = time.time()
    res = traj.propagate_until(ACTUAL_80KM + 3 * 86400.0, [ev, ev120])
    return res, note, time.time() - t_wall


def main(argv=None) -> int:
    """Run both drag models from the file's last state; return the exit status."""
    args = build_parser().parse_args(argv)
    if args.reference_length is None:
        args.reference_length = frontal_diameter_m()
    ballistic.load(space_weather=args.weather, eop=args.eop, offline=True)
    f = ballistic.sp3.read(args.sp3)
    arc = f.arcs[0]
    epoch = arc.epoch0 + float(arc.offsets_s[-1])
    pos = np.asarray(arc.position[-1], float)
    vel = np.asarray(arc.velocity[-1], float)
    alt0 = ballistic.ecef_to_geodetic(pos).altitude / 1e3
    lifetime = ACTUAL_80KM - epoch
    print(f"start {epoch} (last PSO state), altitude {alt0:.1f} km; actual 80 km crossing "
          f"{ACTUAL_80KM} = {lifetime / 86400:.3f} d later")
    results, at120 = {}, {}
    for flow in (False, True):
        res, note, wall = fly(epoch, pos, vel, flow=flow, ref_length=args.reference_length)
        if res.stopped_by is None or not res.stopped_by.name.startswith("altitude"):
            print(f"  {note}: did not reach 80 km ({res.stopped_by})")
            return 1
        err = res.epoch - ACTUAL_80KM
        results[flow] = res.epoch
        at120[flow] = next(o.epoch for o in res.log if o.name.startswith("altitude 120"))
        print(f"  {note}: 80 km at {res.epoch}  error {err / 3600:+.2f} h "
              f"({100 * err / lifetime:+.2f}% of the remaining lifetime)  [{wall:.0f} s]")
    d = results[True] - results[False]
    d120 = at120[True] - at120[False]
    print(f"flow-regime minus constant Cd: {d / 60:+.2f} min at 80 km ({100 * d / lifetime:+.3f}% of "
          f"lifetime), of which {d120 / 60:+.2f} min already at 120 km (free-molecular Cd varying with "
          f"temperature and composition) and {(d - d120) / 60:+.2f} min from 120 to 80 km (the regime)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
