#!/usr/bin/env python3
"""
Sentinel SRP/drag validation: 1-hour (--duration) prediction error against ESA POE orbits.

For each start time (one per --sample-interval minutes) the script takes the
Sentinel position and velocity from a precise orbit ephemeris (POE: ESA/CNES
reference orbit files in the EOF XML format, one state every 10 s, Earth-fixed
frame), propagates for one hour (--duration), and measures the distance to the POE
position at the end. A start is skipped when its start or end state lies
within --maneuver-buffer minutes of a state not flagged NOMINAL (a maneuver
or degraded data). The result is
claim SENTINEL-JAN26-SAMPLED in docs/validation_claims.md.

Force configurations (--configs):
  - baseline: no solar radiation pressure (SRP), no drag
  - srp:      SRP with the registry's (fitted effective) radiation area
  - drag:     drag with the registry's ballistic coefficient β = m / (Cd·A)
              (kg/m²; mass over drag coefficient times frontal area). Air
              density comes from NRLMSIS 2.1 (the atmosphere density model)
              and wind from HWM14 (the wind model), evaluated at the
              satellite's position at every step
  - both:     SRP + drag (the recommended configuration)

Inputs: Sentinel POE .EOF files (--eof-dir, default the committed January 2026
set), the space-weather file (--weather: F10.7 solar flux and Ap geomagnetic
index from CelesTrak, which the density model needs), FES2004 ocean tides
(--ocean-tides), satellite properties from ballistic.satellite_properties.
Output: a printed summary per configuration (overall and per satellite) and
a comparison table; the per-start errors in an HDF5 file (--hdf5) and a violin
plot (--output).

Options (defaults in brackets):
  --mode {propagate,plot,both}   [both]; 'plot' reads --hdf5 instead of propagating
  --eof-dir DIR             directory of *.EOF files [./tests/data/eof/sentinel_jan2026]
  --hdf5 PATH               results file [sentinel_srp_drag_comparison.h5]
  --output PATH             plot file [sentinel_srp_drag_comparison.png]
  --sample-interval MIN     minutes between start times [30]
  --duration SECONDS        prediction length [3600]; the end state must lie in
                            the start's own EOF file (about 26 h of orbit)
  --ephemeris PATH          external JPL planetary ephemeris (.bsp)
  --maneuver-buffer MIN     skip starts and ends this close to a non-NOMINAL state [30]
  --configs NAME [...]      from baseline, srp, drag, both [all four]
  --drag-weather {history,snapshot}   weather varying over the arc, or one value
                            per arc [history]
  --weather PATH|live       CelesTrak CSV [tests/data/weather/SW-All.csv]
  --ocean-tides PATH|off    IERS FES2004 file [the copy ballistic.load() puts in
                            the data directory]; a file that fails to load is an error

Usage (the SENTINEL-JAN26-SAMPLED reproduction command; needs pandas and h5py):
    PYTHONPATH=./python uv run --with numpy --with pandas --with h5py \\
        python scripts/validate_sentinel_srp_drag.py --mode propagate --sample-interval 120
Other examples:
    python scripts/validate_sentinel_srp_drag.py --mode plot
    python scripts/validate_sentinel_srp_drag.py --configs baseline srp
"""

import argparse
import glob
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.normpath(os.path.join(_script_dir, '..', 'python'))
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Published Sentinel Satellite Properties
# ---------------------------------------------------------------------------

from ballistic.satellite_properties import get as _sat_props
SENTINEL_PROPERTIES = {k: _sat_props(k).legacy_dict()
                       for k in ('S1A', 'S1C', 'S2B', 'S2C', 'S3A', 'S3B')}

# Configuration labels
CONFIG_LABELS = {
    'baseline': 'Baseline (no SRP/drag)',
    'srp': 'SRP only',
    'drag': 'Drag only',
    'both': 'SRP + Drag',
}

# ---------------------------------------------------------------------------
# EOF (Earth Observation File) XML reader
# ---------------------------------------------------------------------------

MISSION_SHORT = {
    'Sentinel-1A': 'S1A', 'Sentinel-1B': 'S1B', 'Sentinel-1C': 'S1C',
    'Sentinel-2A': 'S2A', 'Sentinel-2B': 'S2B', 'Sentinel-2C': 'S2C',
    'Sentinel-3A': 'S3A', 'Sentinel-3B': 'S3B',
}


def parse_utc_string(utc_str: str) -> Tuple[int, int, int, int, int, float]:
    """Parse an EOF time 'UTC=YYYY-MM-DDThh:mm:ss.ffffff' into (y, mo, d, h, mi, seconds)."""
    s = utc_str.replace('UTC=', '')
    dt = datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f')
    seconds = dt.second + dt.microsecond / 1e6
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, seconds


def parse_eof(filepath: str) -> Tuple[str, List[Dict]]:
    """Read the orbit state vectors (OSV elements) of one Sentinel EOF file.

    Returns (satellite, states): the short name ('S1A', ...; the mission
    string if unknown) and a list of dicts with utc_tuple (y, mo, d, h, mi,
    s), x/y/z (Earth-fixed position, m), vx/vy/vz (velocity, m/s), quality
    (e.g. 'NOMINAL') and orbit (absolute orbit number). A malformed value
    raises ValueError.
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
# Drag environment (density, wind, β) helpers, shared in scripts/_drag_env_helper.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_WEATHER_ARG = 'live'   # set from --weather in main()
from _drag_env_helper import (  # noqa: E402
    ALT_GRID_SENTINEL_KM,
    get_weather_for_epoch,
    make_arc_config as _shared_make_arc_config,
    add_weather_arg,
    load_weather_source,
)

_DRAG_ON_CONFIGS = frozenset({'drag', 'both'})


# ---------------------------------------------------------------------------
# Config Builder
# ---------------------------------------------------------------------------

def make_config(ballistic, config_name: str, sat_name: str):
    """Return the base TrajectoryConfig for `config_name` and satellite `sat_name`.

    Earth gravity degree 70, Sun/Moon/Jupiter gravity, relativity, ITRF2020,
    RK45 integrator; SRP added for 'srp' and 'both'. Drag configs leave
    drag_env unset here: _make_arc_config attaches the drag environment for
    each start. An unknown `sat_name` raises KeyError.
    """
    props = SENTINEL_PROPERTIES[sat_name]   # unknown satellite: KeyError

    config = ballistic.TrajectoryConfig(
        gravity_degree=70,
        enable_third_body=True,
        enable_relativistic=True,
        itrf=ballistic.ITRFRealization.ITRF2020,
        integrator=ballistic.IntegratorType.RK45,
    )

    # Satellite properties for SRP (drag is attached per start by _make_arc_config).
    if config_name in ('srp', 'both'):
        config = config.replace(srp=ballistic.SrpConfig(
            mass_kg=props['mass'], area_m2=props['srp_area'],
            cr=1.0 + props['srp_reflectivity']))
    return config


def _make_arc_config(ballistic, base_config, config_name, sat_name,
                      start_epoch, pos_ecef, vel_ecef, duration_s, weather):
    """Return the config for one start: `base_config` plus drag when `config_name` has drag.

    β = mass / (Cd · drag area) from the registry. The start state
    (`start_epoch`, `pos_ecef` m, `vel_ecef` m/s), `duration_s` and `weather`
    (WeatherConditions, used with --drag-weather snapshot) are passed to
    _drag_env_helper.make_arc_config, which builds the drag environment.
    """
    props = SENTINEL_PROPERTIES[sat_name]   # unknown satellite: KeyError
    beta_kg_m2 = props['mass'] / (props['cd'] * props['drag_area'])
    return _shared_make_arc_config(
        ballistic, base_config,
        drag_on=(config_name in _DRAG_ON_CONFIGS),
        start_epoch=start_epoch, pos_ecef=pos_ecef, vel_ecef=vel_ecef,
        duration_s=duration_s, beta_kg_m2=beta_kg_m2,
        altitude_grid_km=ALT_GRID_SENTINEL_KM, weather=weather)


# ---------------------------------------------------------------------------
# Propagation Engine
# ---------------------------------------------------------------------------

def run_validation(
    eof_dir: str,
    config_names: List[str],
    sample_interval_min: int = 30,
    ephemeris_path: Optional[str] = None,
    maneuver_buffer_min: int = 30,
    duration_s: float = 3600.0,
) -> pd.DataFrame:
    """
    Propagate every start in the EOF files for each configuration and return the errors.

    Parameters: `eof_dir` (directory of *.EOF files), `config_names` (from
    CONFIG_LABELS), `sample_interval_min` (minutes between starts),
    `ephemeris_path` (optional JPL ephemeris) and `maneuver_buffer_min`
    (skip starts/ends this many minutes from a non-NOMINAL state; 0 = off),
    `duration_s` (prediction length, seconds; start and end in one file).

    Loads Earth orientation parameters (EOP: the measured irregularities of
    the Earth's rotation; warns without them), the --weather space weather
    and the --ocean-tides file (both raise if they fail to load). Prints the
    summary tables. Returns a pandas DataFrame with columns date, satellite,
    epoch_utc (start, UTC), config, error_m (end-point position error,
    meters) and filename; a start covered by two overlapping files is kept
    once. Exits (SystemExit) if `eof_dir` has no .EOF file.
    """
    import ballistic

    ballistic.initialize()

    loader = ballistic.Loader()
    if loader.load_eop():
        print("Loaded EOP data (polar motion + UT1-UTC)")
    else:
        print("WARNING: No EOP data loaded")

    # Measured space weather is required: a quiet-conditions fallback gives
    # a density several times too low. Drag without measured weather is not
    # what this script measures.
    _src = load_weather_source(ballistic, globals().get('_WEATHER_ARG', 'live'))
    print(f"Loaded space weather: {_src}")

    # Ocean tides: part of the recommended setup; an explicit
    # file that fails to load is an error, never a silent tide-free run.
    tides = globals().get('_OCEAN_TIDES_ARG', 'off')
    if tides != 'off':
        if not ballistic._core.load_ocean_tides(tides):
            raise FileNotFoundError(f"ocean tides failed to load: {tides}")
        print(f"Loaded ocean tides: {tides}")

    if ephemeris_path and os.path.exists(ephemeris_path):
        ballistic.load_jpl_ephemeris(ephemeris_path)
        print(f"Loaded external ephemeris: {ephemeris_path}")

    eof_files = sorted(glob.glob(os.path.join(eof_dir, '*.EOF')))
    if not eof_files:
        print(f"ERROR: No .EOF files found in {eof_dir}")
        sys.exit(1)

    print(f"Found {len(eof_files)} EOF files")
    print(f"Configurations: {[CONFIG_LABELS[c] for c in config_names]}")
    if maneuver_buffer_min > 0:
        print(f"Maneuver buffer: {maneuver_buffer_min} min")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    sample_interval_sec = sample_interval_min * 60
    propagation_duration = float(duration_s)
    maneuver_buffer_idx = int(maneuver_buffer_min * 60 / 10)

    rows = []
    skipped_maneuver = 0
    total_files = len(eof_files)

    file_iter = tqdm(eof_files, desc="EOF files") if has_tqdm else eof_files

    for file_idx, eof_path in enumerate(file_iter):
        filename = os.path.basename(eof_path)
        if not has_tqdm and (file_idx % 10 == 0 or file_idx == total_files - 1):
            print(f"  [{file_idx+1}/{total_files}] {filename}")

        sat_name, states = parse_eof(eof_path)
        if len(states) < 2:
            continue

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

        # Build configs for this satellite
        configs = {name: make_config(ballistic, name, sat_name) for name in config_names}

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
                    skipped_maneuver += 1
                    sample_time += sample_interval_sec
                    continue

            start_epoch = epochs[start_idx]
            pos_ecef = np.array([sv_start['x'], sv_start['y'], sv_start['z']])
            vel_ecef = np.array([sv_start['vx'], sv_start['vy'], sv_start['vz']])
            truth_pos = np.array([sv_end['x'], sv_end['y'], sv_end['z']])

            t = sv_start['utc_tuple']
            date_str = f"{t[0]:04d}-{t[1]:02d}-{t[2]:02d}"
            epoch_str = f"{t[0]:04d}-{t[1]:02d}-{t[2]:02d}T{t[3]:02d}:{t[4]:02d}:{t[5]:06.3f}"

            for config_name, base_config in configs.items():
                try:
                    config = _make_arc_config(
                        ballistic, base_config, config_name, sat_name,
                        start_epoch, pos_ecef, vel_ecef, propagation_duration,
                        get_weather_for_epoch(ballistic, start_epoch))
                    traj = ballistic.Trajectory(ballistic.State.from_itrf(start_epoch, pos_ecef, vel_ecef), config=config)
                    traj.propagate_for(propagation_duration)
                    pred_pos = traj.state_ecef().position
                    error_m = np.linalg.norm(pred_pos - truth_pos)

                    rows.append({
                        'date': date_str,
                        'satellite': sat_name,
                        'epoch_utc': epoch_str,
                        'config': config_name,
                        'error_m': error_m,
                        'filename': filename,
                    })
                except Exception as e:
                    print(f"  WARNING: {config_name} failed for {sat_name}: {e}")

            sample_time += sample_interval_sec

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: No propagation results!")
        return df

    if skipped_maneuver > 0:
        print(f"Skipped {skipped_maneuver} samples near maneuver flags")

    # Deduplicate overlapping EOF files
    before = len(df)
    df = df.drop_duplicates(subset=['satellite', 'epoch_utc', 'config'], keep='first')
    after = len(df)
    if before != after:
        print(f"Deduplicated: {before} -> {after} ({before - after} removed)")

    # Print summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")

    for cfg_name in config_names:
        cfg_df = df[df['config'] == cfg_name]
        if cfg_df.empty:
            continue
        rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
        print(f"\n{CONFIG_LABELS[cfg_name]}:")
        print(f"  Samples: {len(cfg_df)}")
        print(f"  Mean:    {cfg_df['error_m'].mean():.3f} m")
        print(f"  Median:  {cfg_df['error_m'].median():.3f} m")
        print(f"  RMSE:    {rmse:.3f} m")
        print(f"  Max:     {cfg_df['error_m'].max():.3f} m")
        print(f"  < 1m:    {(cfg_df['error_m'] < 1.0).sum()}/{len(cfg_df)} "
              f"({100*(cfg_df['error_m'] < 1.0).mean():.1f}%)")

        # Per-satellite
        print("  Per-satellite RMSE:")
        for sat in sorted(cfg_df['satellite'].unique()):
            sat_df = cfg_df[cfg_df['satellite'] == sat]
            sat_rmse = np.sqrt((sat_df['error_m'] ** 2).mean())
            sub1m = 100 * (sat_df['error_m'] < 1.0).mean()
            print(f"    {sat}: {sat_rmse:.3f} m ({sub1m:.0f}% sub-m, n={len(sat_df)})")

    # Comparison table
    if len(config_names) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"{'Config':<25} {'RMSE':>8} {'Mean':>8} {'Median':>8} {'Max':>8} {'<1m':>6}")
        print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for cfg_name in config_names:
            cfg_df = df[df['config'] == cfg_name]
            if cfg_df.empty:
                continue
            rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
            mean = cfg_df['error_m'].mean()
            median = cfg_df['error_m'].median()
            mx = cfg_df['error_m'].max()
            sub1m = 100 * (cfg_df['error_m'] < 1.0).mean()
            print(f"{CONFIG_LABELS[cfg_name]:<25} {rmse:>8.3f} {mean:>8.3f} {median:>8.3f} {mx:>8.3f} {sub1m:>5.1f}%")

    return df


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
# Violin Plot
# ---------------------------------------------------------------------------

def make_violin_plot(df: pd.DataFrame, output_path: str, config_names: List[str]) -> None:
    """Save a violin plot of the error distribution (log scale) to `output_path`.

    One panel per configuration in `config_names`; horizontal lines mark
    1 m and 0.5 m. Needs matplotlib and seaborn.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        print("WARNING: No data to plot")
        return

    n_configs = len(config_names)
    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 8), sharey=True)
    if n_configs == 1:
        axes = [axes]

    colors = {'baseline': '#4C72B0', 'srp': '#DD8452', 'drag': '#55A868', 'both': '#C44E52'}

    for ax, cfg_name in zip(axes, config_names):
        cfg_df = df[df['config'] == cfg_name].copy()
        if cfg_df.empty:
            continue

        satellites = sorted(cfg_df['satellite'].unique())
        cfg_df['satellite'] = pd.Categorical(cfg_df['satellite'], categories=satellites, ordered=True)
        cfg_df['log_error'] = np.log10(cfg_df['error_m'].clip(lower=1e-4))

        sns.violinplot(
            data=cfg_df, x='satellite', y='log_error', ax=ax,
            inner='box', density_norm='width', cut=0,
            color=colors.get(cfg_name, '#4C72B0'), linewidth=0.8, saturation=0.8,
        )

        ax.axhline(y=np.log10(1.0), color='green', linestyle='--', alpha=0.7, linewidth=1.2)
        ax.axhline(y=np.log10(0.5), color='green', linestyle=':', alpha=0.5, linewidth=1.0)

        y_ticks = [-2, -1, 0, 1]
        y_labels = ['0.01', '0.1', '1', '10']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
        sub1m = 100 * (cfg_df['error_m'] < 1.0).mean()
        ax.set_title(f"{CONFIG_LABELS[cfg_name]}\nRMSE: {rmse:.3f}m, <1m: {sub1m:.0f}%",
                      fontsize=11, fontweight='bold')
        ax.set_xlabel('Satellite')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Position Error (meters)')

    fig.suptitle('Sentinel 1-Hour Prediction Error: SRP/Drag Comparison',
                  fontsize=14, fontweight='bold', y=1.02)
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
        description='Compare Sentinel propagation accuracy with/without SRP and drag',
    )
    parser.add_argument('--mode', choices=['propagate', 'plot', 'both'], default='both')
    parser.add_argument('--eof-dir', default='./tests/data/eof/sentinel_jan2026')
    parser.add_argument('--hdf5', default='sentinel_srp_drag_comparison.h5')
    parser.add_argument('--output', default='sentinel_srp_drag_comparison.png')
    parser.add_argument('--sample-interval', type=int, default=30)
    parser.add_argument('--duration', type=float, default=3600.0,
                        help='prediction length [s] (default 3600)')
    parser.add_argument('--ephemeris', default=None)
    parser.add_argument('--maneuver-buffer', type=int, default=30)
    parser.add_argument(
        '--configs', nargs='+', default=['baseline', 'srp', 'drag', 'both'],
        choices=['baseline', 'srp', 'drag', 'both'],
        help='Configurations to compare (default: all four)',
    )
    add_weather_arg(parser)
    parser.add_argument(
        '--ocean-tides', default=None,
        help="IERS FES2004 Stokes file (default: the copy ballistic.load() "
             "fetches into the data directory), or 'off'")
    args = parser.parse_args()
    globals()['_WEATHER_ARG'] = args.weather
    if args.ocean_tides is None:
        import ballistic
        args.ocean_tides = os.path.join(str(ballistic.get_data_directory()), 'fes2004_Cnm-Snm.dat')
    globals()['_OCEAN_TIDES_ARG'] = args.ocean_tides

    if args.mode in ('propagate', 'both'):
        t0 = time.time()
        df = run_validation(
            args.eof_dir, args.configs, args.sample_interval,
            args.ephemeris, args.maneuver_buffer, duration_s=args.duration,
        )
        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed/60:.1f} minutes")
        if not df.empty:
            save_results(df, args.hdf5)

    if args.mode in ('plot', 'both'):
        if args.mode == 'plot':
            df = load_results(args.hdf5)
        if not df.empty:
            make_violin_plot(df, args.output, args.configs)


if __name__ == '__main__':
    main()
