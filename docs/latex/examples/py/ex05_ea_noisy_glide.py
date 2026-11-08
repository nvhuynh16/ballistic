"""
Tutorial example 05 — Empirical acceleration from NOISY data, hv-block CV (Python)

Forced-glide truth + Gaussian noise (POS 1.0 m, VEL 0.25 m/s, ACC
0.1 m/s^2; seeds 42/43/44). Hyperparameters chosen by hv-block CV
(set_cv) — no truth peeking — matching deployment. GN (regularized,
consistent-objective) is default-ON and benefits POS and VEL alike;
the ACC facade ignores it (circular).

HEAVY (CV grid = hundreds of fits). One {obs,rate} cell per process.

    python ex05_ea_noisy_glide.py <pos|vel|acc> <100|200> [out.csv]
"""
import sys
import time

import ballistic
import glide_common as gc


def main(obs, hz, out=None):
    hz = int(hz)
    out = out or f"docs/latex/results/ea_noisy_{obs}_{hz}_py.csv"
    ballistic.initialize()

    dt = 0.005 if hz == 200 else 0.01
    truth = gc.GlideTruth(with_ea=True)
    ts, P, V, A = truth.sample(dt)
    fm = gc.base_config()   # wrappers accept the dataclass directly

    if obs == "pos":
        sigma, arr, seed = 1.0, P, 42
    elif obs == "vel":
        sigma, arr, seed = 0.25, V, 43
    elif obs == "acc":
        sigma, arr, seed = 0.1, A, 44
    else:
        print("obs must be pos|vel|acc", file=sys.stderr)
        return 2
    obs_data = gc.to_obs(ts, truth.t0, gc.add_noise(arr, sigma, seed))

    c = ballistic.estimation.EABSplineConfig()
    c.set_cv()                       # hv-block CV (no truth peeking)
    c.noise_sigma = sigma
    # c.gn defaults to GNRefine() (regularized GN — on for POS and VEL alike);
    # the ACC facade ignores it (circular; gn_status reports the skip).

    t0 = time.time()
    if obs == "pos":
        r = ballistic.estimation.estimate_ea_bspline(
            truth.t0, truth.pos, truth.vel,
            ballistic.estimation.PositionSamples(obs_data), fm, c)
    elif obs == "vel":
        r = ballistic.estimation.estimate_ea_bspline(
            truth.t0, truth.pos, truth.vel,
            ballistic.estimation.VelocitySamples(obs_data), fm, c)
    else:
        r = ballistic.estimation.estimate_ea_bspline(
            truth.t0, truth.pos, truth.vel,
            ballistic.estimation.AccelSamples(obs_data), fm, c)
    wall_min = (time.time() - t0) / 60.0
    rmse = gc.reprop_rmse(truth, r.empirical_accel)
    lam = r.lambda_used if r.lambda_used is not None else -1.0

    print(f"obs={obs} hz={hz} n={len(obs_data)} sigma={sigma}")
    print(f"  ok={r.has_correction()} lambda_used={lam} reprop_RMSE={rmse:.4f} m "
          f"wall={wall_min:.1f} min")

    with open(out, "w") as f:
        f.write("obs_type,obs_hz,n_obs,sigma,reg,gn,reprop_rmse_m,"
                "lambda_used,wall_min\n")
        gn = "n/a" if obs == "acc" else "on"
        f.write(f"{obs},{hz},{len(obs_data)},{sigma},hv_block_cv,{gn},"
                f"{rmse:.6f},{lam},{wall_min:.2f}\n")
    print(f"wrote {out}")
    return 0 if r.has_correction() else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:4]))
