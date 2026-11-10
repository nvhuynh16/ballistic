#!/usr/bin/env python3
"""
Wind Effect Validation: Compare Sentinel LEO accuracy with/without HWM14 wind.

Drag depends on the satellite's speed relative to the air, and the upper atmosphere
moves: HWM14 (the Horizontal Wind Model 2014) gives that wind. This script measures
whether including it improves 1-hour predictions of the Sentinel satellites in low
Earth orbit (LEO).

Runs two configurations on the Jan 2026 Sentinel orbit files (ESA's XML Earth
Observation File format, .EOF, one state every 10 s):
  1. SRP+Drag (baseline best config)
  2. SRP+Drag+Wind (HWM14 thermospheric wind correction)
SRP is solar radiation pressure; mass and areas come from the satellite registry
(ballistic.satellite_properties). Drag uses the library's live atmosphere (NRLMSIS 2.1
density, with or without HWM14 wind) on the loaded space weather, with the ballistic
coefficient beta = m/(Cd*A) in kg/m^2 from the registry.

Method: in every file, a start every 30 minutes; the state is propagated for 1 hour
and compared with the file's state at that time. Starts or ends on a non-NOMINAL state,
or within 30 minutes of one (maneuvers), are skipped.

Inputs: tests/data/eof/sentinel_jan2026/*.EOF; Earth orientation parameters and space
weather are downloaded or loaded by ballistic.Loader(). No command-line options.
Output: per configuration, the root-mean-square error (RMSE), mean, median, maximum and
share below 1 m of the 1-hour position errors, per satellite RMSE, and the change in
RMSE that the wind makes.

Usage:
    python scripts/validate_wind_effect.py
"""

import glob
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
import _source_tree  # noqa: E402
_source_tree.use_source_tree()

import numpy as np

# ---------------------------------------------------------------------------
# EOF parser
# ---------------------------------------------------------------------------

MISSION_SHORT = {
    'Sentinel-1A': 'S1A', 'Sentinel-1B': 'S1B', 'Sentinel-1C': 'S1C',
    'Sentinel-2A': 'S2A', 'Sentinel-2B': 'S2B', 'Sentinel-2C': 'S2C',
    'Sentinel-3A': 'S3A', 'Sentinel-3B': 'S3B',
}

from ballistic.satellite_properties import get as _sat_props


def parse_utc_string(utc_str):
    """Parse an EOF time "UTC=YYYY-MM-DDTHH:MM:SS.ffffff" into (year, month, day,
    hour, minute, seconds as float)."""
    s = utc_str.replace('UTC=', '')
    dt = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f')
    seconds = dt.second + dt.microsecond / 1e6
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, seconds


def parse_eof(filepath):
    """Read a Sentinel EOF orbit file.

    Returns (satellite short name such as 'S1A', list of state dicts). Each state has
    'utc_tuple' (see parse_utc_string), 'x', 'y', 'z' (Earth-fixed position, m), 'vx',
    'vy', 'vz' (Earth-fixed velocity, m/s), 'quality' (e.g. 'NOMINAL') and 'orbit'
    (absolute orbit number). Works with and without an XML namespace.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    mission_el = root.find(f'.//{ns}Mission')
    if mission_el is None:
        mission_el = root.find('.//Mission')
    mission = mission_el.text.strip() if mission_el is not None else 'Unknown'
    sat_name = MISSION_SHORT.get(mission, mission)
    osvs = root.findall(f'.//{ns}OSV')
    if not osvs:
        osvs = root.findall('.//OSV')
    states = []
    for osv in osvs:
        def txt(tag):
            el = osv.find(f'{ns}{tag}')
            if el is None:
                el = osv.find(tag)
            return el.text.strip() if el is not None else ''
        utc_tuple = parse_utc_string(txt('UTC'))
        states.append({
            'utc_tuple': utc_tuple,
            'x': float(txt('X')), 'y': float(txt('Y')), 'z': float(txt('Z')),
            'vx': float(txt('VX')), 'vy': float(txt('VY')), 'vz': float(txt('VZ')),
            'quality': txt('Quality'),
            'orbit': int(txt('Absolute_Orbit').replace('+', '')),
        })
    return sat_name, states


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_config(eof_dir, config, config_name, sample_interval_min=30,
                maneuver_buffer_min=30, *, drag=True, wind=True):
    """Run the 1-hour prediction test for one configuration over every EOF file.

    eof_dir: directory of .EOF files; config: the TrajectoryConfig (its SRP is replaced
    per file from the registry); config_name: a label (not used in the computation);
    sample_interval_min: minutes between starts; maneuver_buffer_min: minutes around a
    non-NOMINAL state in which starts are skipped.
    Returns (results, skipped): results is a list of (satellite, error in meters)
    tuples, skipped the number of starts dropped for being near a maneuver. A start
    whose propagation raises is reported and left out.

    When drag=True, a per-arc DragEnvironment is built with
    ballistic.make_drag_environment (through _drag_env_helper; live NRLMSIS 2.1
    density with HWM14 wind). When wind=False on top of drag=True,
    DragEnvironment.without_wind() is applied to isolate the wind contribution.
    drag=False bypasses the env build entirely (drag off).
    """
    import ballistic
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _drag_env_helper import (
        ALT_GRID_SENTINEL_KM,
        get_weather_for_epoch,
        build_drag_env_for_arc,
    )
    import dataclasses

    eof_files = sorted(glob.glob(os.path.join(eof_dir, '*.EOF')))
    sample_interval_sec = sample_interval_min * 60
    propagation_duration = 3600.0
    maneuver_buffer_idx = int(maneuver_buffer_min * 60 / 10)

    results = []
    skipped = 0

    for file_idx, eof_path in enumerate(eof_files):
        if file_idx % 20 == 0:
            print(f"  [{file_idx+1}/{len(eof_files)}] {os.path.basename(eof_path)}")

        sat_name, states = parse_eof(eof_path)
        if len(states) < 2:
            continue

        # Satellite properties from the registry, per file; an unknown
        # satellite raises (a silent default would reuse another satellite's
        # SRP or skip drag without a word).
        props = _sat_props(sat_name)
        file_config = dataclasses.replace(config, srp=ballistic.SrpConfig(
            mass_kg=props.mass_kg, area_m2=props.srp_area_m2,
            cr=1.0 + props.srp_reflectivity))

        first_utc = states[0]['utc_tuple']
        epoch0_ref = ballistic.Epoch.from_utc(*first_utc)
        epochs = [ballistic.Epoch.from_utc(*sv['utc_tuple']) for sv in states]
        offsets = [epochs[i] - epoch0_ref for i in range(len(states))]
        total_span = offsets[-1]

        degraded_indices = set()
        if maneuver_buffer_idx > 0:
            for i, sv in enumerate(states):
                if sv['quality'] != 'NOMINAL':
                    degraded_indices.add(i)

        max_start_offset = total_span - propagation_duration
        if max_start_offset <= 0:
            continue

        sample_time = 0.0
        while sample_time <= max_start_offset:
            start_idx = round(sample_time / 10.0)
            if start_idx >= len(states):
                break

            sv_start = states[start_idx]
            if sv_start['quality'] != 'NOMINAL':
                sample_time += sample_interval_sec
                continue

            target_offset = offsets[start_idx] + propagation_duration
            end_idx = round(target_offset / 10.0)
            if end_idx >= len(states):
                sample_time += sample_interval_sec
                continue

            sv_end = states[end_idx]
            if sv_end['quality'] != 'NOMINAL':
                sample_time += sample_interval_sec
                continue

            if degraded_indices and maneuver_buffer_idx > 0:
                too_close = False
                for d_idx in degraded_indices:
                    if (abs(d_idx - start_idx) <= maneuver_buffer_idx or
                            abs(d_idx - end_idx) <= maneuver_buffer_idx):
                        too_close = True
                        break
                if too_close:
                    skipped += 1
                    sample_time += sample_interval_sec
                    continue

            start_epoch = epochs[start_idx]
            pos_ecef = np.array([sv_start['x'], sv_start['y'], sv_start['z']])
            vel_ecef = np.array([sv_start['vx'], sv_start['vy'], sv_start['vz']])

            try:
                # Build a per-arc DragEnvironment when drag is requested.
                # `wind=False` on top swaps the HWM14 wind for an empty
                # WindProfile1D_ENU via DragEnvironment.without_wind().
                arc_config = file_config
                if drag:
                    beta_kg_m2 = props.beta_kg_m2
                    env = build_drag_env_for_arc(
                        ballistic, start_epoch, pos_ecef, vel_ecef,
                        get_weather_for_epoch(ballistic, start_epoch),
                        beta_kg_m2, propagation_duration,
                        ALT_GRID_SENTINEL_KM)
                    if not wind:
                        env = env.without_wind()
                    arc_config = dataclasses.replace(file_config, drag_env=env)
                traj = ballistic.Trajectory(ballistic.State.from_itrf(start_epoch, pos_ecef, vel_ecef), config=arc_config)
                traj.propagate_for(propagation_duration)
                pred_pos = traj.state_ecef().position
                truth_pos = np.array([sv_end['x'], sv_end['y'], sv_end['z']])
                error_m = np.linalg.norm(pred_pos - truth_pos)
                results.append((sat_name, error_m))
            except Exception as e:
                print(f"  WARNING: {sat_name} failed: {e}")

            sample_time += sample_interval_sec

    return results, skipped


def print_summary(name, results):
    """Print summary statistics for a config and return its RMSE in meters.

    results: list of (satellite, error in meters) from run_config.
    """
    errors = np.array([r[1] for r in results])
    rmse = np.sqrt(np.mean(errors ** 2))
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(errors)}")
    print(f"  RMSE:    {rmse:.3f} m")
    print(f"  Mean:    {np.mean(errors):.3f} m")
    print(f"  Median:  {np.median(errors):.3f} m")
    print(f"  Max:     {np.max(errors):.3f} m")
    print(f"  <1m:     {100*np.mean(errors < 1.0):.1f}%")

    # Per-satellite
    sats = sorted(set(r[0] for r in results))
    for sat in sats:
        sat_errors = np.array([r[1] for r in results if r[0] == sat])
        sat_rmse = np.sqrt(np.mean(sat_errors ** 2))
        print(f"  {sat}: RMSE {sat_rmse:.3f} m (n={len(sat_errors)})")

    return rmse


def main():
    """Load the library data and run both configurations, then print the comparison."""
    import ballistic

    ballistic.initialize()
    loader = ballistic.Loader()
    loader.load_eop()
    loader.space_weather()

    eof_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'data', 'eof', 'sentinel_jan2026')
    eof_dir = os.path.normpath(eof_dir)

    if not os.path.isdir(eof_dir):
        print(f"ERROR: EOF directory not found: {eof_dir}")
        sys.exit(1)

    n_files = len(glob.glob(os.path.join(eof_dir, '*.EOF')))
    print(f"Found {n_files} EOF files in {eof_dir}")

    # Config 1: SRP + Drag (no wind)
    print("\n--- Config 1: SRP + Drag (no wind) ---")
    cfg1 = ballistic.TrajectoryConfig(
        gravity_degree=70,
        enable_third_body=True,
        srp=None,                   # set per file from the registry
        enable_relativistic=True,
        itrf=ballistic.ITRFRealization.ITRF2020,
        integrator=ballistic.IntegratorType.RK45,
    )
    t0 = time.time()
    results1, skip1 = run_config(eof_dir, cfg1, "SRP+Drag(no wind)",
                                   drag=True, wind=False)
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f}s, skipped {skip1} maneuver-adjacent")

    # Config 2: SRP + Drag + HWM14 wind
    print("\n--- Config 2: SRP + Drag + Wind (HWM14) ---")
    cfg2 = ballistic.TrajectoryConfig(
        gravity_degree=70,
        enable_third_body=True,
        srp=None,                   # set per file from the registry
        enable_relativistic=True,
        itrf=ballistic.ITRFRealization.ITRF2020,
        integrator=ballistic.IntegratorType.RK45,
    )
    t2 = time.time()
    results2, skip2 = run_config(eof_dir, cfg2, "SRP+Drag+Wind",
                                   drag=True, wind=True)
    t3 = time.time()
    print(f"  Time: {t3-t2:.1f}s, skipped {skip2} maneuver-adjacent")

    # Print summaries
    rmse1 = print_summary("SRP+Drag (baseline)", results1)
    rmse2 = print_summary("SRP+Drag+Wind (HWM14)", results2)

    # Comparison
    pct_change = 100 * (rmse2 - rmse1) / rmse1
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(f"  Baseline RMSE:  {rmse1:.3f} m")
    print(f"  Wind RMSE:      {rmse2:.3f} m")
    print(f"  Change:         {pct_change:+.1f}%")
    if pct_change <= 0:
        print(f"  Result: Wind IMPROVES accuracy by {-pct_change:.1f}%")
    else:
        print(f"  Result: Wind DEGRADES accuracy by {pct_change:.1f}%")


if __name__ == '__main__':
    main()
