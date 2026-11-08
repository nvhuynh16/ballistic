"""
Tutorial example 04 — Empirical acceleration from CLEAN data (Python)

Forced-glide truth, clean ECEF obs at 100 Hz. The lesson is METHOD
SELECTION — bare defaults underperform on an aggressive reentry; the
recommended config per obs type recovers sub-meter EA:

  POS  B-spline IRGN, NO GN    (cautionary: GN is load-bearing for POS)
  POS  B-spline IRGN + GN      (recommended for clean POS)
  VEL  B-spline IRGN + GN      (GN helps VEL too, measured)
  ACC  B-spline IRGN           (direct; GN circular for ACC, ignored)

    python ex04_ea_clean_glide.py [out.csv]
"""
import sys

import ballistic
import glide_common as gc


def main(out="docs/latex/results/ea_clean_glide.csv"):
    ballistic.initialize()
    truth = gc.GlideTruth(with_ea=True)
    ts, P, V, A = truth.sample(0.01)                       # 100 Hz
    fm = gc.base_config()   # wrappers accept the dataclass directly
    obsP = gc.to_obs(ts, truth.t0, P)
    obsV = gc.to_obs(ts, truth.t0, V)
    obsA = gc.to_obs(ts, truth.t0, A)

    rows = []

    # The "do nothing" control: EA=0 reprop vs forced-glide truth. Every
    # row below must be read against this reference scale.
    rows.append(("---", "no_ea_baseline", 0.0,
                 gc.ea0_baseline_rmse(truth), True, 0))

    c = ballistic.estimation.EABSplineConfig()
    c.gn = None                                            # GN is default-ON — disable for baseline
    r = ballistic.estimation.estimate_ea_bspline(truth.t0, truth.pos, truth.vel,
                                      ballistic.estimation.PositionSamples(obsP), fm, c)
    rows.append(("POS", "bspline_no_gn", c.knot_spacing_s,
                 gc.reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                 r.iterations_used))

    c = ballistic.estimation.EABSplineConfig()
    c.gn = ballistic.estimation.GNRefine()                            # load-bearing
    r = ballistic.estimation.estimate_ea_bspline(truth.t0, truth.pos, truth.vel,
                                      ballistic.estimation.PositionSamples(obsP), fm, c)
    rows.append(("POS", "bspline_gn", c.knot_spacing_s,
                 gc.reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                 r.iterations_used))

    c = ballistic.estimation.EABSplineConfig()
    c.gn = ballistic.estimation.GNRefine()                            # GN helps VEL too (measured)
    r = ballistic.estimation.estimate_ea_bspline(truth.t0, truth.pos, truth.vel,
                                      ballistic.estimation.VelocitySamples(obsV), fm, c)
    rows.append(("VEL", "bspline_gn", c.knot_spacing_s,
                 gc.reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                 r.iterations_used))

    c = ballistic.estimation.EABSplineConfig()
    r = ballistic.estimation.estimate_ea_bspline(truth.t0, truth.pos, truth.vel,
                                      ballistic.estimation.AccelSamples(obsA), fm, c)
    rows.append(("ACC", "bspline_irgn", c.knot_spacing_s,
                 gc.reprop_rmse(truth, r.empirical_accel), r.has_correction(),
                 r.iterations_used))

    print(f"obs=100 Hz  n={len(ts)}")
    print("  obs  method         knot   reprop_RMSE  ok  iters")
    print("  ---------------------------------------------------")
    with open(out, "w") as f:
        f.write("obs_type,method,knot_s,obs_hz,n_obs,reprop_rmse_m,ok,iterations\n")
        for obs, meth, knot, rmse, ok, it in rows:
            print(f"  {obs:>3}  {meth:>13}  {knot:>4}  {rmse:10.4f}  {ok}  {it}")
            f.write(f"{obs},{meth},{knot},100,{len(ts)},{rmse:.6f},"
                    f"{str(ok).lower()},{it}\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
