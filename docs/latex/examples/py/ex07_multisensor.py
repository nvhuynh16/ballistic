"""
Tutorial example 07 — Multi-sensor EA estimation (Python)

Two ground radars (from the glide fixture) track the forced glide. We
synthesize clean [range, az, el, range-rate] obs via RadarObs.predict on
the truth states, fuse with estimate_ea_smoother (the recommended fusion
path — a GM-RTS reduced-dynamic smoother: EKF forward pass + RTS backward
pass over a Gauss-Markov EA state, auto NIS hyperparameters, measured
range-rate budget), then re-propagate the recovered EA and RMSE vs truth.
Two radars avoid the single-radar+rate-on degeneracy.

Case 2 adds a 50 µrad astrometric camera (OpticsObs, az/el only)
co-located with radar 0 — the documented way to tighten the radar-only
information bound (~9-17 m for this geometry) by an order of magnitude.

CAVEAT (teaching point): radar -> EA is observability-limited, so the
radar-only reprop sits at the geometry floor, not estimator error. nis ~ 1
is the truth-free health check. Prefer 2+ radars, and add a camera/fix to
tighten the bound.

    python ex07_multisensor.py [out.csv]
"""
import math
import sys

import numpy as np

import ballistic
import glide_common as gc


def enu_rotation(lat, lon):
    """ECEF->ENU 3x3, matching coords::ecef_to_enu_rotation exactly."""
    sla, cla = math.sin(lat), math.cos(lat)
    slo, clo = math.sin(lon), math.cos(lon)
    return np.array([
        [-slo,          clo,          0.0],
        [-sla * clo,   -sla * slo,    cla],
        [ cla * clo,    cla * slo,    sla],
    ])


def make_radar(site, lat_deg, lon_deg):
    r = ballistic.estimation.RadarObs()
    r.position_ecef = np.asarray(site, dtype=float)
    r.sigma_range = 2.0
    r.sigma_angle = 0.02 * math.pi / 180.0
    r.sigma_rate = 0.02
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    r.lat_rad = lat
    r.lon_rad = lon
    r.ecef_to_enu = enu_rotation(lat, lon)
    return r


def make_camera(site, lat_deg, lon_deg):
    c = ballistic.estimation.OpticsObs()          # sigma_angle default: 50e-6 rad
    c.position_ecef = np.asarray(site, dtype=float)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    c.lat_rad = lat
    c.lon_rad = lon
    c.ecef_to_enu = enu_rotation(lat, lon)
    return c


def main(out="docs/latex/results/multisensor_rmse.csv"):
    ballistic.initialize()
    truth = gc.GlideTruth(with_ea=True)

    st = gc.FIX["radar_stations"]
    r0 = make_radar(st[0]["pos_ecef"], st[0]["lat_deg"], st[0]["lon_deg"])
    r1 = make_radar(st[1]["pos_ecef"], st[1]["lat_deg"], st[1]["lon_deg"])
    cam = make_camera(st[0]["pos_ecef"], st[0]["lat_deg"], st[0]["lon_deg"])

    ts, P, V, _ = truth.sample(0.5)  # 2 Hz clean
    o0 = np.array([r0.predict(ballistic.StateVector(P[i], V[i]))
                   for i in range(len(ts))])
    o1 = np.array([r1.predict(ballistic.StateVector(P[i], V[i]))
                   for i in range(len(ts))])
    oc = np.array([cam.predict(ballistic.StateVector(P[i], V[i]))
                   for i in range(len(ts))])
    r0.set_samples(truth.t0, ts, o0)
    r1.set_samples(truth.t0, ts, o1)
    cam.set_samples(truth.t0, ts, oc)

    rows = []
    for label, optics in [("2xradar", []), ("2xradar+camera", [cam])]:
        scenario = ballistic.estimation.MultiSensorScenario(
            epoch0=truth.t0, pos0_ecef=truth.pos, vel0_ecef=truth.vel,
            arc_duration_s=truth.eval_end, radars=[r0, r1], optics=optics)
        mask = ballistic.estimation.SensorMask.all_on(scenario)
        cfg = ballistic.estimation.EASmootherConfig()  # minimal: NIS-auto prior + rate budget

        result = ballistic.estimation.estimate_ea_smoother(
            scenario, mask, gc.base_config(), cfg)
        if not result.has_correction():
            print(f"{label}: smoother failed: {result.error_message}",
                  file=sys.stderr)
            return 1
        rmse = gc.reprop_rmse(truth, result.empirical_accel)
        rows.append((label, result.nis, rmse))

        print(f"{label} (clean, 2 Hz, {len(ts)} obs/sensor)")
        print(f"  NIS (1=ideal, <1 cons.): {result.nis:.3f}")
        print(f"  reprop RMSE vs truth   : {rmse:.3f} m")

    with open(out, "w") as f:
        f.write("sensors,obs_hz,n_obs_per_sensor,nis,reprop_rmse_m\n")
        for label, nis, rmse in rows:
            f.write(f"{label},2,{len(ts)},{nis:.6f},{rmse:.6f}\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
