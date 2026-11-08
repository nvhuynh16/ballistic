#!/usr/bin/env python3
"""
GOCE reentry validation: prediction error at 142-236 km during the final decay.

Checks the atmospheric drag model and the HWM14 wind model at very low
altitudes using GOCE's uncontrolled decay (Oct 22 - Nov 10, 2013): after its
ion-thruster fuel ran out on Oct 21, 2013, GOCE decayed from ~236 km to
reentry on Nov 11, 2013. For every start time (one per --sample-interval
minutes) the script takes the position and velocity from the precise orbit,
propagates for --duration seconds, and measures the distance to the orbit's
position at the end; results are grouped into bands of geodetic altitude
(height above the WGS84 ellipsoid). The result is claim GOCE-REENTRY in
docs/validation_claims.md.

Inputs (--sp3; a directory is scanned for *.sp3, *.SP3, *.sp3c, *.SP3c): ESOC
orbits in SP3 (the standard precise-orbit text format) in the inertial J2000
frame, rotated here to the Earth-fixed frame (ITRF) with the IAU 2006
transform, epochs in the file's GPS time; or AIUB precise science orbits
converted by convert_goce_pso.py (Earth-fixed; the ITRF version is taken from
the file's frame label). All files of one run must share one frame. The
accel_density configuration reads the ESA GOCE v2.0 density files under
tests/data/goce_density (--density-file to pick one). Space weather (F10.7
solar flux and Ap geomagnetic index, needed by the density model): --weather.

Force configurations (--configs):
  baseline       no drag, no solar radiation pressure (SRP)
  srp            SRP only (control)
  drag           drag without wind: density from NRLMSIS 2.1 (the atmosphere
                 model) evaluated at the satellite's position, registry
                 ballistic coefficient β = m / (Cd·A) (kg/m²)
  drag_wind      drag + HWM14 wind
  both           SRP + drag
  both_wind      SRP + drag + HWM14 wind
  accel_density  air density measured by GOCE's accelerometer instead of NRLMSIS

Output: a printed summary per configuration, an RMSE (root-mean-square
error) table per altitude band, the wind effect per band, the per-start errors
in an HDF5 file (--hdf5) and a two-panel plot (--output).

Options (defaults in brackets):
  --sp3 PATH [PATH ...]     SP3 files or directories; required unless --mode plot
  --mode {propagate,plot,both}   [both]; 'plot' reads --hdf5 instead of propagating
  --sample-interval MIN     minutes between start times [30]
  --duration S              prediction length, seconds [3600]
  --hdf5 PATH               results file [goce_reentry_validation.h5]
  --output PATH             plot file [goce_reentry_validation.png]
  --ephemeris PATH          external JPL planetary ephemeris (.bsp)
  --configs NAME [...]      [baseline drag drag_wind both both_wind srp]
  --density-file NAME       use only this file (basename under tests/data/goce_density)
  --drag-scale X            diagnostic: multiply drag by X (divides β) [1.0]
  --drag-weather {history,snapshot}   weather varying over the arc, or one value
                            per arc [history]
  --weather PATH|live       CelesTrak CSV [tests/data/weather/SW-All.csv]

Usage (the GOCE-REENTRY reproduction command; needs pandas and h5py):
    PYTHONPATH=./python uv run --with numpy --with pandas --with h5py \\
        python scripts/validate_goce_reentry.py --sp3 tests/data/sp3/goce/goce_13295.sp3 \\
        tests/data/sp3/goce/goce_13305.sp3 tests/data/sp3/goce/goce_13313.sp3 \\
        --mode propagate --duration 3600 --sample-interval 60 --configs baseline drag accel_density
Other examples:
    python scripts/validate_goce_reentry.py --sp3 tests/data/sp3/goce/ --configs baseline drag drag_wind
    python scripts/validate_goce_reentry.py --sp3 tests/data/sp3/goce/ --duration 7200
    python scripts/validate_goce_reentry.py --mode plot --hdf5 goce_reentry_validation.h5
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timezone
from typing import List

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.normpath(os.path.join(_script_dir, '..', 'python'))
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# GOCE Satellite Properties
# ---------------------------------------------------------------------------

from ballistic.satellite_properties import get as _sat_props
from _drag_env_helper import (add_weather_arg,  # noqa: E402
                              load_weather_source)
_WEATHER_ARG = 'live'   # set from --weather in main()
_g = _sat_props('GOCE')
GOCE_PROPERTIES = dict(_g.legacy_dict(), gravity_degree=_g.gravity_degree)

# GOCE SP3 satellite id (confirmed from the ESA products)
GOCE_SP3_ID = 'L15'

CONFIG_LABELS = {
    'baseline': 'Baseline (no SRP/drag)',
    'srp': 'SRP only',
    'drag': 'Drag only',
    'drag_wind': 'Drag + Wind',
    'both': 'SRP + Drag',
    'both_wind': 'SRP + Drag + Wind',
    'accel_density': 'Accel. Density',
}

# Geodetic altitude bands (km) for the analysis: (low, high, label)
ALTITUDE_BANDS = [
    (220, 250, '>220 km'),
    (200, 220, '200-220 km'),
    (180, 200, '180-200 km'),
    (160, 180, '160-180 km'),
    (0, 160, '<160 km'),
]


# ---------------------------------------------------------------------------
# Reference orbits: read with ballistic.sp3 through scripts/_sp3_records.py
# ---------------------------------------------------------------------------

import _sp3_records as _sp3r  # noqa: E402


def load_goce_sp3(sp3_paths: List[str]):
    """Read the GOCE SP3 files; return (arcs, frame info).

    One arc per file (see _sp3_records.load_arcs), epochs in the file's time
    system. One frame per run: the ESOC files are J2000 (inertial), the AIUB
    precise science orbit (PSO) files from convert_goce_pso.py are
    Earth-fixed (IGS05 / IGb08). Daily files are separate solutions, so a
    start's truth one horizon later comes from the same file. Raises
    SystemExit if the files hold more than one satellite or mix frames.
    """
    import ballistic as _ballistic
    data = _sp3r.load_arcs(_ballistic, sorted(sp3_paths))
    if len(data) != 1:
        raise SystemExit(f"expected one satellite in the GOCE files, found {sorted(data)}")
    arcs = next(iter(data.values()))
    frames = {(a['frame'].inertial, a['frame'].itrf) for a in arcs}
    if len(frames) != 1:
        raise SystemExit(f"mixed SP3 frames in one run: {sorted(map(str, frames))}")
    n = sum(len(a['records']) for a in arcs)
    print(f"\n  Total: {n} records in {len(arcs)} file arcs")
    return arcs, arcs[0]['frame']


# ---------------------------------------------------------------------------
# Config Builder
# ---------------------------------------------------------------------------

DRAG_CONFIGS = {
    'drag', 'both', 'drag_wind', 'both_wind',
    'accel_density',
}
SRP_CONFIGS = {'srp', 'both', 'both_wind'}
WIND_CONFIGS = {'drag_wind', 'both_wind'}
ACCEL_CONFIGS = {'accel_density'}
# Drag uses a constant β here. A drag coefficient that varies along the orbit
# (e.g. the Sentman free-molecular Cd) can be pre-evaluated into a custom
# BetaProfile1D when that fidelity is needed.


def make_config(ballistic, config_name: str):
    """Return the base TrajectoryConfig for `config_name` (one of CONFIG_LABELS).

    Earth gravity (registry degree), Sun/Moon/Jupiter gravity, relativity,
    solid Earth tides, the ITRF version chosen in run_validation, RK45
    integrator; SRP added for the SRP configs. drag_env is attached later
    (see the comment at the end).
    """
    props = GOCE_PROPERTIES

    config = ballistic.TrajectoryConfig(
        gravity_degree=props['gravity_degree'], enable_third_body=True,
        enable_relativistic=True, enable_solid_tides=True,
        itrf=getattr(ballistic.ITRFRealization, _ITRF),   # from the SP3 frame label
        integrator=ballistic.IntegratorType.RK45)

    if config_name in SRP_CONFIGS:
        config = config.replace(srp=ballistic.SrpConfig(
            mass_kg=props['mass'], area_m2=props['srp_area'],
            cr=1.0 + props['srp_reflectivity']))

    # Drag wiring: accel_density gets the shared drag environment built once
    # in run_validation from the measured densities; the NRLMSIS drag/wind
    # configs get one per start in the start loop, via
    # _drag_env_helper.build_drag_env_for_arc.
    return config


def compute_altitude_km(ballistic, pos_m: np.ndarray, epoch, is_eci: bool) -> float:
    """Geodetic altitude [km] above the WGS84 ellipsoid.

    `pos_m` is numpy (3,) in meters; with `is_eci` it is an inertial (J2000)
    position and is first rotated to Earth-fixed at `epoch`. Not the radius
    minus 6371 km: a spherical altitude shifts the altitude bands by +7 km at
    the equator to -14 km near the poles.
    """
    pos = np.asarray(pos_m, float)
    if is_eci:
        pos = np.array(_inertial_to_ecef(ballistic, pos, np.zeros(3), epoch).position)
    return ballistic.ecef_to_geodetic(pos).altitude / 1000.0


def get_altitude_band(alt_km: float) -> str:
    """Return the ALTITUDE_BANDS label for an altitude in km.

    Altitudes outside the bands go to the nearest end band.
    """
    for low, high, label in ALTITUDE_BANDS:
        if low <= alt_km < high:
            return label
    if alt_km >= 250:
        return '>220 km'
    return '<160 km'


# ---------------------------------------------------------------------------
# Propagation Engine
# ---------------------------------------------------------------------------

# Frame and time handling. The ESOC GOCE files are J2000 (inertial) with
# GPS-time epochs. They are rotated with the IAU 2006 GCRS<->ITRF transform
# (GCRS = the inertial frame the library integrates in), not a
# true-equator-of-date rotation (which misses the precession of the Earth's
# axis since 2000, ~22 km in low Earth orbit in 2013), and epochs come from
# ballistic.sp3, which honors the file's time system (reading GPS time as UTC
# would shift every epoch by the GPS-UTC offset).
_DRAG_SCALE = 1.0   # diagnostic: multiplies drag (divides beta); --drag-scale


def _goce_beta():
    """GOCE's ballistic coefficient β = m / (Cd·A) in kg/m², divided by --drag-scale."""
    return (GOCE_PROPERTIES['mass']
            / (GOCE_PROPERTIES['cd'] * GOCE_PROPERTIES['drag_area'])) / _DRAG_SCALE
# ITRF realization of the Earth-fixed states: the SP3 label for Earth-fixed
# files (IGS05 = ITRF2005, IGb08 = ITRF2008), ITRF2008 (current during GOCE)
# when the J2000 states are rotated to Earth-fixed here. Set in run_validation.
_ITRF = 'ITRF2008'


def _inertial_to_ecef(ballistic, pos, vel, epoch):
    """Rotate an inertial (J2000) state at `epoch` to Earth-fixed (ITRF version _ITRF).

    `pos` (m) and `vel` (m/s) are numpy (3,); returns the library's state
    object (.position, .velocity).
    """
    # J2000 is treated as GCRS (they differ by a fixed frame bias, ~0.7 m here)
    return ballistic.gcrs_to_itrf(pos, vel, epoch, getattr(ballistic.ITRFRealization, _ITRF))


def run_validation(
    sp3_paths: List[str],
    config_names: List[str],
    duration_sec: float = 3600.0,
    sample_interval_min: int = 30,
    ephemeris_path: str = None,
) -> pd.DataFrame:
    """
    Propagate every start in the SP3 files for each configuration and return the errors.

    Parameters: `sp3_paths` (list of SP3 paths), `config_names` (from
    CONFIG_LABELS), `duration_sec` (prediction length, s),
    `sample_interval_min` (minutes between starts) and `ephemeris_path`
    (optional JPL ephemeris).

    Inertial (J2000) files are rotated to Earth-fixed for the start state and
    the truth; Earth-fixed files are used as read. Predicted and true
    positions are compared Earth-fixed. Loads Earth orientation parameters
    (EOP: the measured irregularities of the Earth's rotation; warns without
    them) and the --weather space weather (raises if it fails). For
    accel_density, raises FileNotFoundError if a density file is missing.
    Prints the summary tables. Returns a pandas DataFrame with columns
    epoch_utc (start label), config, error_m (end-point position error, m),
    alt_km and alt_band (start altitude) and duration_sec.
    """
    import ballistic

    ballistic.initialize()

    loader = ballistic.Loader()
    if loader.load_eop():
        print("Loaded EOP data")
    else:
        print("WARNING: No EOP data loaded")

    # No fallback: drag without measured weather is not what this script measures.
    _src = load_weather_source(ballistic, globals().get('_WEATHER_ARG', 'live'))
    print(f"Loaded space weather: {_src}")

    if ephemeris_path and os.path.exists(ephemeris_path):
        ballistic.load_jpl_ephemeris(ephemeris_path)
        print(f"Loaded external ephemeris: {ephemeris_path}")

    # Load GOCE's measured densities into a drag environment if any
    # accel_density config is requested. One environment serves every start:
    # from_accelerometer_density holds the whole time series in a
    # DensityProfile2D (density against altitude and time).
    accel_density_env = None
    if any(c in ACCEL_CONFIGS for c in config_names):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        density_dir = os.path.normpath(os.path.join(script_dir, '..', 'tests', 'data', 'goce_density'))
        if os.path.isdir(density_dir):
            all_mjds = []
            all_rhos = []
            # EXPLICIT list, not a glob: a glob makes the results depend on
            # whatever files happen to be present (the committed 2013-10-15
            # slice of the month file would be counted twice). These four
            # files are exactly what the GOCE-REENTRY campaign reads
            # (2,165,858 records).
            _K7_DENSITY_FILES = [
                'goce_denswind_ac082_v2_0_2013-10.txt',        # month, ~250 km
                'goce_denswind_ac082_v2_0_deorb-DFACS.txt',    # deorbit, thruster
                'goce_denswind_ac082_v2_0_deorb-GPSonly.txt',  # deorbit, GPS-only
                'GO_DNS_ACC_2013_10_v02.txt',                  # accelerometer
            ]
            override = globals().get('_DENSITY_FILE_OVERRIDE')
            names = [override] if override else _K7_DENSITY_FILES
            density_files = [os.path.join(density_dir, n) for n in names]
            absent = [f for f in density_files if not os.path.isfile(f)]
            if absent:
                raise FileNotFoundError(
                    "GOCE density inputs missing: %s — reacquire with "
                    "`python scripts/download_goce_density.py` (results are "
                    "input-dependent, so this is an error, not a warning)"
                    % [os.path.basename(f) for f in absent])
            for f in density_files:
                mjds, rhos = ballistic.drag.parse_density_table(f)
                all_mjds.extend(mjds)
                all_rhos.extend(rhos)
                print(f"  Parsed {len(mjds)} density records from {os.path.basename(f)}")
            print(f"  Total: {len(all_mjds)} density records parsed")

            if all_mjds:
                # Sort by epoch (parse_density_table doesn't sort across files)
                order = sorted(range(len(all_mjds)), key=lambda i: all_mjds[i])
                all_mjds = [all_mjds[i] for i in order]
                all_rhos = [all_rhos[i] for i in order]
                # Drop duplicate epochs (keep first)
                dedup_mjds = []
                dedup_rhos = []
                last_mjd = None
                for mjd, rho in zip(all_mjds, all_rhos):
                    if last_mjd is None or mjd > last_mjd + 1e-9:
                        dedup_mjds.append(mjd)
                        dedup_rhos.append(rho)
                        last_mjd = mjd
                # GOCE altitudes 100-300 km in 20 km steps
                alt_grid = [100 + 20 * i for i in range(11)]
                epochs_obj = [ballistic.Epoch.from_mjd(m, ballistic.TimeScale.UTC)  # guard-exempt(FLOAT-MJD-EPOCH): density samples, interpolated
                               for m in dedup_mjds]
                beta = ballistic.BetaProfile1D.constant(
                    _goce_beta())
                accel_density_env = ballistic.drag.from_accelerometer_density(
                    epochs_obj, dedup_rhos, beta, alt_grid)
                print(f"  Built DragEnvironment from accel density: "
                      f"{len(dedup_mjds)} unique sample epochs")
        else:
            print(f"WARNING: GOCE density data not found at {density_dir}")
            print("  Download with: python scripts/download_goce_density.py")

    print("\nLoading GOCE SP3 data...")
    arcs, frame = load_goce_sp3(sp3_paths)
    is_eci = frame.inertial
    # Earth-fixed files: the label's realization; inertial (J2000) states are
    # rotated into ITRF2008, the realization current during GOCE.
    globals()['_ITRF'] = frame.itrf.name if frame.itrf is not None else 'ITRF2008'
    print(f"  Frame: {'J2000 (inertial)' if is_eci else 'Earth-fixed'}; ITRF realization {_ITRF}; "
          f"time system {arcs[0]['time_system']}; frame transform IAU 2006")

    print(f"\nConfigurations: {[CONFIG_LABELS[c] for c in config_names]}")
    print(f"Propagation duration: {duration_sec:.0f}s")
    print(f"Sample interval: {sample_interval_min} min")

    for arc in arcs:
        for r in arc['records']:
            r['alt_km'] = compute_altitude_km(ballistic, r['pos_m'],
                                              arc['epoch0'] + float(r['offset_s']), is_eci)
    alts = [r['alt_km'] for arc in arcs for r in arc['records']]
    print(f"  Altitude range: {min(alts):.0f} - {max(alts):.0f} km")

    configs = {name: make_config(ballistic, name) for name in config_names}

    # Attach the accelerometer-derived DragEnvironment to accel_density configs.
    if accel_density_env is not None:
        import dataclasses
        for name in list(configs.keys()):
            if name in ACCEL_CONFIGS:
                configs[name] = dataclasses.replace(
                    configs[name], drag_env=accel_density_env)

    rows = []
    sample_interval_sec = sample_interval_min * 60
    n_samples = 0
    n_errors = 0

    for s in _sp3r.starts(arcs, sample_interval_sec, duration_sec):
        sv_start, sv_end = s['start'], s['end']
        start_epoch, end_epoch = s['epoch'], s['end_epoch']
        actual_duration = s['duration']
        alt_km = sv_start['alt_km']
        alt_band = get_altitude_band(alt_km)
        epoch_str = s['label']

        # Rotate the J2000 state to Earth-fixed for the propagation input
        if is_eci:
            pos_j2000 = sv_start['pos_m']
            vel_j2000 = sv_start['vel_ms']
            try:
                state_ecef = _inertial_to_ecef(ballistic, pos_j2000, vel_j2000, start_epoch)
                pos_ecef = np.array(state_ecef.position)
                vel_ecef = np.array(state_ecef.velocity)
            except Exception as e:
                if n_errors < 3:
                    print(f"  WARNING: Frame conversion failed at {epoch_str}: {e}")
                n_errors += 1
                continue

            # Truth position at the end (also rotated to Earth-fixed)
            truth_pos_j2000 = sv_end['pos_m']
            truth_vel_j2000 = sv_end['vel_ms']
            try:
                truth_ecef = _inertial_to_ecef(ballistic, truth_pos_j2000, truth_vel_j2000, end_epoch)
                truth_pos = np.array(truth_ecef.position)
            except Exception as e:
                if n_errors < 3:
                    print(f"  WARNING: Truth frame conversion failed: {e}")
                n_errors += 1
                continue
        else:
            pos_ecef = sv_start['pos_m']
            vel_ecef = sv_start['vel_ms']
            truth_pos = sv_end['pos_m']

        # One NRLMSIS drag environment per start, shared by the model-drag
        # configs: the wind variants keep the HWM14 wind, the no-wind
        # variants strip it (without_wind()).
        arc_env = None
        nrlmsis_drag = [c for c in configs
                        if c in DRAG_CONFIGS and c not in ACCEL_CONFIGS]
        if nrlmsis_drag:
            try:
                from _drag_env_helper import (build_drag_env_for_arc,
                                              get_weather_for_epoch)
                arc_env = build_drag_env_for_arc(
                    ballistic, start_epoch=start_epoch, pos_ecef=pos_ecef,
                    vel_ecef=vel_ecef, duration_s=actual_duration,
                    beta_kg_m2=_goce_beta(),
                    altitude_grid_km=[120 + 20 * i for i in range(10)],
                    weather=get_weather_for_epoch(ballistic, start_epoch))
            except Exception as e:
                if n_errors < 3:
                    print(f"  WARNING: arc env build failed at {epoch_str}: {e}")
                n_errors += 1

        for config_name, config in configs.items():
            try:
                if config_name in nrlmsis_drag:
                    if arc_env is None:
                        continue
                    import dataclasses
                    env = (arc_env if config_name in WIND_CONFIGS
                           else arc_env.without_wind())
                    config = dataclasses.replace(config, drag_env=env)
                traj = ballistic.Trajectory(ballistic.State.from_itrf(start_epoch, pos_ecef, vel_ecef), config=config)
                traj.propagate_for(actual_duration)
                pred_pos = np.array(traj.state_ecef().position)
                error_m = np.linalg.norm(pred_pos - truth_pos)

                rows.append({
                    'epoch_utc': epoch_str,
                    'config': config_name,
                    'error_m': error_m,
                    'alt_km': alt_km,
                    'alt_band': alt_band,
                    'duration_sec': actual_duration,
                })
                n_samples += 1
            except Exception as e:
                n_errors += 1
                if n_errors <= 5:
                    print(f"  WARNING: {config_name} failed at {epoch_str} (alt={alt_km:.0f}km): {e}")
                elif n_errors == 6:
                    print("  (suppressing further warnings)")

        if n_samples % (len(config_names) * 10) == 0 and n_samples > 0:
            n_per_config = n_samples // len(config_names)
            print(f"  ... {n_per_config} samples per config (alt={alt_km:.0f}km)")

    n_per_config = n_samples // max(len(config_names), 1)
    print(f"\n  Total: {n_per_config} samples per config ({n_errors} errors)")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nWARNING: No propagation results!")
        return df

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=['epoch_utc', 'config'], keep='first')
    after = len(df)
    if before != after:
        print(f"Deduplicated: {before} -> {after}")

    # Print summary
    print_summary(df, config_names)
    print_altitude_analysis(df, config_names)

    return df


def print_summary(df: pd.DataFrame, config_names: List[str]) -> None:
    """Print the overall statistics per configuration, then a comparison table.

    Per configuration: sample count, mean/median/RMSE/max error, altitude
    range, and the shares of errors under 100 m and 1 km.
    """
    print(f"\n{'='*70}")
    print("GOCE REENTRY VALIDATION SUMMARY")
    print(f"{'='*70}")

    for cfg_name in config_names:
        cfg_df = df[df['config'] == cfg_name]
        if cfg_df.empty:
            continue
        rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
        print(f"\n{CONFIG_LABELS[cfg_name]}:")
        print(f"  Samples:  {len(cfg_df)}")
        print(f"  Mean:     {cfg_df['error_m'].mean():.1f} m")
        print(f"  Median:   {cfg_df['error_m'].median():.1f} m")
        print(f"  RMSE:     {rmse:.1f} m")
        print(f"  Max:      {cfg_df['error_m'].max():.1f} m")
        print(f"  Alt range: {cfg_df['alt_km'].min():.0f} - {cfg_df['alt_km'].max():.0f} km")
        print(f"  < 100m:   {(cfg_df['error_m'] < 100).sum()}/{len(cfg_df)} "
              f"({100*(cfg_df['error_m'] < 100).mean():.1f}%)")
        print(f"  < 1km:    {(cfg_df['error_m'] < 1000).sum()}/{len(cfg_df)} "
              f"({100*(cfg_df['error_m'] < 1000).mean():.1f}%)")

    # Comparison table
    if len(config_names) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Config':<28} {'RMSE':>9} {'Mean':>9} {'Median':>9} {'Max':>9} {'<100m':>7} {'<1km':>7}")
        print(f"{'-'*28} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*7}")

        baseline_rmse = None
        for cfg_name in config_names:
            cfg_df = df[df['config'] == cfg_name]
            if cfg_df.empty:
                continue
            rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
            mean = cfg_df['error_m'].mean()
            median = cfg_df['error_m'].median()
            mx = cfg_df['error_m'].max()
            sub100 = 100 * (cfg_df['error_m'] < 100).mean()
            sub1k = 100 * (cfg_df['error_m'] < 1000).mean()

            if baseline_rmse is None:
                baseline_rmse = rmse
            change = f" ({100*(rmse - baseline_rmse)/baseline_rmse:+.1f}%)" if baseline_rmse and cfg_name != 'baseline' else ""

            print(f"{CONFIG_LABELS[cfg_name]:<28} {rmse:>8.1f}m {mean:>8.1f}m {median:>8.1f}m {mx:>8.1f}m {sub100:>6.1f}% {sub1k:>6.1f}%{change}")


def print_altitude_analysis(df: pd.DataFrame, config_names: List[str]) -> None:
    """Print RMSE per altitude band and configuration, then drag vs drag+wind per band."""
    print(f"\n{'='*70}")
    print("ALTITUDE BAND ANALYSIS")
    print(f"{'='*70}")

    for low, high, label in ALTITUDE_BANDS:
        band_df = df[(df['alt_km'] >= low) & (df['alt_km'] < high)]
        if band_df.empty:
            continue

        n_per_cfg = len(band_df) // max(len(config_names), 1)
        print(f"\n--- {label} ({n_per_cfg} samples per config) ---")
        print(f"{'Config':<28} {'RMSE':>9} {'Mean':>9} {'Median':>9}")
        print(f"{'-'*28} {'-'*9} {'-'*9} {'-'*9}")

        band_baseline_rmse = None
        for cfg_name in config_names:
            cfg_df = band_df[band_df['config'] == cfg_name]
            if cfg_df.empty:
                continue
            rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
            mean = cfg_df['error_m'].mean()
            median = cfg_df['error_m'].median()

            if band_baseline_rmse is None:
                band_baseline_rmse = rmse
            change = f" ({100*(rmse - band_baseline_rmse)/band_baseline_rmse:+.1f}%)" if band_baseline_rmse and cfg_name != 'baseline' else ""

            print(f"{CONFIG_LABELS[cfg_name]:<28} {rmse:>8.1f}m {mean:>8.1f}m {median:>8.1f}m{change}")

    # Wind improvement by altitude
    if 'drag' in config_names and 'drag_wind' in config_names:
        print(f"\n{'='*70}")
        print("WIND EFFECT BY ALTITUDE (drag vs drag+wind)")
        print(f"{'='*70}")
        print(f"{'Band':<15} {'Drag RMSE':>12} {'Wind RMSE':>12} {'Change':>10} {'Samples':>8}")
        print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

        for low, high, label in ALTITUDE_BANDS:
            band_df = df[(df['alt_km'] >= low) & (df['alt_km'] < high)]
            drag_df = band_df[band_df['config'] == 'drag']
            wind_df = band_df[band_df['config'] == 'drag_wind']
            if drag_df.empty or wind_df.empty:
                continue

            drag_rmse = np.sqrt((drag_df['error_m'] ** 2).mean())
            wind_rmse = np.sqrt((wind_df['error_m'] ** 2).mean())
            change_pct = 100 * (wind_rmse - drag_rmse) / drag_rmse

            print(f"{label:<15} {drag_rmse:>11.1f}m {wind_rmse:>11.1f}m {change_pct:>+9.1f}% {len(drag_df):>8}")


# ---------------------------------------------------------------------------
# HDF5 Storage
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, path: str) -> None:
    """Write the results DataFrame to HDF5 file `path`, group 'validation' (needs h5py)."""
    import h5py
    with h5py.File(path, 'w') as f:
        grp = f.create_group('validation')
        for col in df.columns:
            vals = df[col].values
            # Decide on the VALUES dtype: pandas extension/datetime columns
            # can present a non-object dtype while .values is object ('O'),
            # which h5py rejects. Strings/objects/datetimes all go via str.
            if vals.dtype.kind in ('O', 'U', 'M'):
                grp.create_dataset(col, data=df[col].astype(str).values.astype('S'))
            else:
                grp.create_dataset(col, data=vals)
        grp.attrs['run_timestamp'] = datetime.now(timezone.utc).isoformat()
        grp.attrs['n_samples'] = len(df)
        grp.attrs['columns'] = list(df.columns)
        grp.attrs['satellite'] = 'GOCE'
        grp.attrs['phase'] = 'ballistic_decay'
    print(f"Saved {len(df)} results to {path}")


def load_results(path: str) -> pd.DataFrame:
    """Read a results DataFrame written by save_results; exits if `path` is missing."""
    import h5py
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with h5py.File(path, 'r') as f:
        grp = f['validation']
        columns = list(grp.attrs['columns'])
        data = {}
        for col in columns:
            vals = grp[col][()]
            if vals.dtype.kind in ('S', 'O', 'U'):
                data[col] = [v.decode() if isinstance(v, bytes) else v for v in vals]
            else:
                data[col] = vals
        df = pd.DataFrame(data)
    print(f"Loaded {len(df)} results from {path}")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, output_path: str, config_names: List[str]) -> None:
    """Save a two-panel plot to `output_path`: RMSE per altitude band, and error vs altitude."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if df.empty:
        print("WARNING: No data to plot")
        return

    colors = {
        'baseline': '#4C72B0',
        'srp': '#DD8452',
        'drag': '#55A868',
        'drag_wind': '#8172B2',
        'both': '#C44E52',
        'both_wind': '#CCB974',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: RMSE by altitude band
    ax = axes[0]
    band_labels = []
    bar_data = {cfg: [] for cfg in config_names}

    for low, high, label in ALTITUDE_BANDS:
        band_df = df[(df['alt_km'] >= low) & (df['alt_km'] < high)]
        if band_df.empty:
            continue
        band_labels.append(label)
        for cfg in config_names:
            cfg_df = band_df[band_df['config'] == cfg]
            if cfg_df.empty:
                bar_data[cfg].append(0)
            else:
                rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
                bar_data[cfg].append(rmse)

    if band_labels:
        x = np.arange(len(band_labels))
        width = 0.8 / len(config_names)
        for i, cfg in enumerate(config_names):
            offset = (i - len(config_names)/2 + 0.5) * width
            ax.bar(x + offset, bar_data[cfg], width,
                   label=CONFIG_LABELS[cfg], color=colors.get(cfg, '#666666'), alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=30, ha='right')
        ax.set_ylabel('RMSE (meters)')
        ax.set_title('RMSE by Altitude Band')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Plot 2: Error vs altitude scatter
    ax = axes[1]
    for cfg in config_names:
        cfg_df = df[df['config'] == cfg]
        if cfg_df.empty:
            continue
        ax.scatter(cfg_df['alt_km'], cfg_df['error_m'],
                   label=CONFIG_LABELS[cfg], color=colors.get(cfg, '#666666'),
                   alpha=0.3, s=10)
    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel('Position Error (meters)')
    ax.set_title('Error vs Altitude')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('GOCE Reentry Validation: Drag/Wind Effect at Very Low Altitude',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse the command line (see the module docstring) and run."""
    parser = argparse.ArgumentParser(
        description='Validate drag/wind modeling at very low altitude using GOCE reentry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GOCE decayed from ~236 km to reentry (~142 km) over 21 days after ion
thruster fuel exhaustion (Oct 21 - Nov 11, 2013). ESA published precise
SP3 orbits at 60-second intervals — the lowest-altitude precise orbits
ever produced.

At these altitudes, atmospheric drag dominates and thermospheric wind
(HWM14) is 5-30%% of orbital velocity, making this ideal for validating
wind effects on orbit prediction.

SP3 files are in J2000 (ECI) frame. The script converts to ECEF for
propagation and compares in ECEF.

Download data first:
    python scripts/download_goce_sp3.py

Examples:
    python scripts/validate_goce_reentry.py --sp3 tests/data/sp3/goce/
    python scripts/validate_goce_reentry.py --sp3 tests/data/sp3/goce/ --configs baseline drag drag_wind
    python scripts/validate_goce_reentry.py --sp3 tests/data/sp3/goce/ --duration 7200
    python scripts/validate_goce_reentry.py --mode plot --hdf5 goce_reentry_validation.h5
        """,
    )
    parser.add_argument('--sp3', nargs='+', help='SP3 file(s) or directory')
    parser.add_argument('--mode', choices=['propagate', 'plot', 'both'], default='both')
    parser.add_argument('--sample-interval', type=int, default=30,
                        help='Minutes between samples (default: 30)')
    parser.add_argument('--duration', type=float, default=3600.0,
                        help='Propagation duration in seconds (default: 3600)')
    parser.add_argument('--hdf5', default='goce_reentry_validation.h5',
                        help='HDF5 output path')
    parser.add_argument('--output', default='goce_reentry_validation.png',
                        help='Plot output path')
    parser.add_argument('--ephemeris', default=None,
                        help='Path to external JPL ephemeris (optional)')
    parser.add_argument(
        '--configs', nargs='+',
        default=['baseline', 'drag', 'drag_wind', 'both', 'both_wind', 'srp'],
        choices=list(CONFIG_LABELS.keys()),
        help='Configurations to compare (default: all but accel_density)',
    )

    parser.add_argument('--density-file', default=None,
                        help='Use exactly this density file (basename under '
                             'tests/data/goce_density) instead of the '
                             'GOCE-REENTRY campaign set')

    parser.add_argument('--drag-scale', type=float, default=1.0,
                        help='diagnostic: multiply drag by this factor (divides beta)')
    add_weather_arg(parser)
    args = parser.parse_args()
    globals()['_DRAG_SCALE'] = args.drag_scale
    globals()['_WEATHER_ARG'] = args.weather
    globals()['_DENSITY_FILE_OVERRIDE'] = args.density_file

    if args.mode in ('propagate', 'both'):
        if not args.sp3:
            parser.error("--sp3 is required for propagation mode")

        sp3_paths = []
        for path in args.sp3:
            if os.path.isdir(path):
                for ext in ('*.sp3', '*.SP3', '*.sp3c', '*.SP3c'):
                    sp3_paths.extend(sorted(glob.glob(os.path.join(path, ext))))
            else:
                sp3_paths.append(path)

        if not sp3_paths:
            print(f"ERROR: No SP3 files found in {args.sp3}")
            sys.exit(1)

        print(f"Found {len(sp3_paths)} SP3 file(s)")

        t0 = time.time()
        df = run_validation(
            sp3_paths, args.configs, args.duration,
            args.sample_interval, args.ephemeris,
        )
        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed/60:.1f} minutes")

        if not df.empty:
            save_results(df, args.hdf5)

    if args.mode in ('plot', 'both'):
        if args.mode == 'plot':
            df = load_results(args.hdf5)
        if not df.empty:
            make_plots(df, args.output, args.configs)


if __name__ == '__main__':
    main()
