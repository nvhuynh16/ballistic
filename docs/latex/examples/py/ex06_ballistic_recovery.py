"""
Tutorial example 06 — The ballistic sanity check (Python)

"Ballistic" = the glide IC propagated with NO empirical acceleration. We
run the SAME EA estimator on it; a correct estimator must not hallucinate
acceleration — recovered EA ~0, re-propagation RMSE small. This validates
the pipeline before trusting it on forced data.

    python ex06_ballistic_recovery.py [out.csv]
"""
import sys

import numpy as np

import ballistic
import glide_common as gc


def main(out="docs/latex/results/ea_ballistic_recovery.csv"):
    ballistic.initialize()

    truth = gc.GlideTruth(with_ea=False)              # NO EA
    ts, P, V, A = truth.sample(0.5)                    # 2 Hz
    obs = gc.to_obs(ts, truth.t0, P)

    cfg = ballistic.estimation.EABSplineConfig()                  # IRGN, knot 2 s, deg 5
    r = ballistic.estimation.estimate_ea_bspline(
        truth.t0, truth.pos, truth.vel,
        ballistic.estimation.PositionSamples(obs), gc.base_config(), cfg)
    if not r.has_correction():
        print(f"estimator failed: {r.error_message}", file=sys.stderr)
        return 1

    peak_ea = max(
        float(np.linalg.norm(r.empirical_accel.accel_at(float(t))))
        for t in np.arange(0.0, truth.eval_end, 1.0)
    )
    rmse = gc.reprop_rmse(truth, r.empirical_accel)

    # Case 2 — deployment pipeline on noisy null data: does the train-test
    # (hv-block CV) split overfit 1 m noise into a phantom EA when there is
    # no real acceleration? POS 100 Hz, sigma=1.0, set_cv() + GN.
    tsn, Pn, _, _ = truth.sample(0.01)                 # 100 Hz
    noisy = gc.to_obs(tsn, truth.t0, gc.add_noise(Pn, 1.0, 42))
    nc = ballistic.estimation.EABSplineConfig()
    nc.set_cv()
    nc.noise_sigma = 1.0
    nc.gn = ballistic.estimation.GNRefine()
    rn = ballistic.estimation.estimate_ea_bspline(
        truth.t0, truth.pos, truth.vel,
        ballistic.estimation.PositionSamples(noisy), gc.base_config(), nc)
    peak_n = max(
        float(np.linalg.norm(rn.empirical_accel.accel_at(float(t))))
        for t in np.arange(0.0, truth.eval_end, 1.0)
    )
    rmse_n = gc.reprop_rmse(truth, rn.empirical_accel)

    print(f"clean  (2 Hz, IRGN)      : peak|EA|={peak_ea:.4f} "
          f"reprop={rmse:.4f} m  (expect ~0)")
    print(f"noisy  (100 Hz, hv-CV+GN): peak|EA|={peak_n:.4f} "
          f"reprop={rmse_n:.4f} m  (no phantom EA from noise)")

    with open(out, "w") as f:
        f.write("case,obs_hz,n_obs,sigma,peak_ea_mps2,reprop_rmse_m\n")
        f.write(f"clean_default,2,{len(obs)},0,{peak_ea:.6f},{rmse:.6f}\n")
        f.write(f"noisy_hvblock_cv,100,{len(noisy)},1.0,"
                f"{peak_n:.6f},{rmse_n:.6f}\n")
    print(f"wrote {out}")
    return 0 if (rmse < 5.0 and rmse_n < 50.0 and rn.has_correction()) else 2


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
