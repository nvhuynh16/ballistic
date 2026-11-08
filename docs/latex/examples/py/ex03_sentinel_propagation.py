"""
Tutorial example 03 — Propagation vs truth: Sentinel-1, RK45 vs YOSHIDA8,
                       ITRF2014 vs ITRF2020 (Python)

Mirrors the C++ example using the new ``ballistic`` EOF/RMSE binding
(EOFFile + compute_propagation_result, exposed from the C++ validation
surface). Take an early POE state vector as the IC, propagate 1 hour,
RMSE-compare to the EOF truth (RTN-decomposed).

  * RK45 (adaptive) vs YOSHIDA8 (symplectic)
  * ITRF2014 (CORRECT for Sentinel POE before 2025-03-04) vs ITRF2020
  * SRP/drag OFF — the force-model-only baseline. (Not because the POE
    "already contains them": turning both on improves the fit — see
    ballistic.recommended_config / docs/validation_claims.md SENTINEL-YEAR-2025.)

    python ex03_sentinel_propagation.py [eof_dir] [out.json]
"""
import json
import sys

import numpy as np

import ballistic


def run_case(eof, ic, integrator, itrf, duration_s, sample_s):
    cfg = ballistic.TrajectoryConfig(itrf=itrf, integrator=integrator)
    if integrator != ballistic.IntegratorType.RK45:
        cfg = cfg.replace(symplectic_step_size=30.0)

    traj = ballistic.Trajectory(ballistic.State.from_itrf(ic.epoch, ic.position, ic.velocity), config=cfg)
    # Step to build dense history (mirrors the C++ example); a single
    # propagate() leap leaves no intermediate samples to interpolate.
    while (traj.epoch() - ic.epoch) < duration_s + sample_s:
        traj.step()

    offsets = np.arange(0.0, duration_s + sample_s / 2.0, sample_s)
    states = traj([float(t) for t in offsets], frame="ECEF")
    predicted = [
        (ic.epoch + float(t), ballistic.StateVector(s.position, s.velocity))
        for t, s in zip(offsets, states)
        if s is not None
    ]
    return ballistic.readers.compute_propagation_result(predicted, eof)


def itrf_name(r):
    return "ITRF2014" if r == ballistic.ITRFRealization.ITRF2014 else "ITRF2020"


def main(eof_dir="tests/data/eof/sentinel1_2016/2016-1",
         out="docs/latex/results/ex03_sentinel_rmse.json"):
    ballistic.initialize()

    eof = ballistic.readers.find_and_parse_eof(eof_dir)
    if eof is None or len(eof.state_vectors) < 400:
        print(f"no usable EOF in {eof_dir}", file=sys.stderr)
        return 1
    print(f"Parsed {eof.mission}  ({len(eof.state_vectors)} state vectors, "
          f"frame {eof.reference_frame})")

    ic = eof.state_vectors[10]
    DUR, DT = 3600.0, 60.0
    cases = [
        (ballistic.IntegratorType.RK45, "RK45", ballistic.ITRFRealization.ITRF2014),
        (ballistic.IntegratorType.YOSHIDA8, "YOSHIDA8", ballistic.ITRFRealization.ITRF2014),
        (ballistic.IntegratorType.RK45, "RK45", ballistic.ITRFRealization.ITRF2020),
    ]

    print("\n  integrator  ITRF      pos_RMSE   radial   along    cross   vel_RMSE")
    print("  ---------------------------------------------------------------------")
    out_cases = []
    for integ, iname, itrf in cases:
        r = run_case(eof, ic, integ, itrf, DUR, DT)
        print(f"  {iname:>9}  {itrf_name(itrf):>8}  {r.position_rmse:8.3f}  "
              f"{r.radial_rms:7.3f}  {r.along_track_rms:7.3f}  "
              f"{r.cross_track_rms:7.3f}  {r.velocity_rmse:7.3f}  "
              f"(n={r.num_comparisons})")
        out_cases.append({
            "integrator": iname, "itrf": itrf_name(itrf),
            "pos_rmse_m": round(r.position_rmse, 4),
            "radial_m": round(r.radial_rms, 4),
            "along_m": round(r.along_track_rms, 4),
            "cross_m": round(r.cross_track_rms, 4),
            "vel_rmse_mps": round(r.velocity_rmse, 4),
            "n": r.num_comparisons,
        })

    with open(out, "w") as f:
        json.dump({"mission": eof.mission, "arc_s": DUR, "sample_s": DT,
                   "cases": out_cases}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
