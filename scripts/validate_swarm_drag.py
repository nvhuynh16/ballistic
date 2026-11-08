#!/usr/bin/env python3
"""
Swarm drag validation: 1-hour prediction error at Swarm altitudes against ESA orbits.

Checks the atmospheric drag model at Swarm's altitudes (Swarm A and C fly side
by side, Swarm B higher), where drag is a first-order force. For every start time (one per
--sample-interval minutes) the script takes the position and velocity from
the reference orbit, propagates for --duration seconds, and measures the
distance to the reference position at the end.

Force configurations (--configs):
  - baseline:  no solar radiation pressure (SRP), no drag
  - srp:       "cannonball" SRP (the satellite treated as a sphere)
  - drag:      drag with the registry's ballistic coefficient β = m / (Cd·A)
               (kg/m²; mass over drag coefficient times frontal area). Air
               density comes from NRLMSIS 2.1 (the atmosphere density model)
               and wind from HWM14 (the wind model), evaluated at the
               satellite's position at every step ("live")
  - both:      SRP + drag

The density model needs space weather: F10.7 (the daily solar radio flux)
and Ap (the geomagnetic activity index), from a CelesTrak CSV (--weather).

Inputs: Swarm precise orbit files in SP3 (the standard precise-orbit text
format; ESA product SP3xCOM_2_) or directories of them (--sp3; a directory is
scanned for *.sp3, *.SP3, *.sp3c, *.SP3c), the space-weather file
(--weather), satellite properties from ballistic.satellite_properties.
Output: a printed summary per configuration and a comparison table; the
per-start errors in an HDF5 file (--hdf5) and a violin plot (--output).

Data acquisition:
    Swarm SP3 data requires free registration at https://swarm-diss.eo.esa.int/
    Download SP3xCOM_2_ products to tests/data/sp3/swarm/

Options (defaults in brackets):
  --sp3 PATH [PATH ...]     SP3 files or directories; required unless --mode plot
  --sat-id ID               read only this SP3 satellite id
  --mode {propagate,plot,both}   [both]; 'plot' reads --hdf5 instead of propagating
  --sample-interval MIN     minutes between start times [30]
  --duration S              prediction length, seconds [3600]
  --hdf5 PATH               results file [swarm_drag_validation.h5]
  --output PATH             plot file [swarm_drag_validation.png]
  --ephemeris PATH          external JPL planetary ephemeris (.bsp)
  --configs NAME [...]      from baseline, srp, drag, both [all four]
  --drag-weather {history,snapshot}   weather varying over the arc, or one value
                            per arc [history]
  --weather PATH|live       CelesTrak CSV [tests/data/weather/SW-All.csv]
  --sp3-velocity {file-no-cip-drift,file,derived}   start velocity [file]

Usage (needs pandas and h5py):
    python scripts/validate_swarm_drag.py --sp3 tests/data/sp3/swarm/
    python scripts/validate_swarm_drag.py --sp3 tests/data/sp3/swarm/ --configs baseline drag
    python scripts/validate_swarm_drag.py --mode plot --hdf5 swarm_drag_validation.h5
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict

_script_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.normpath(os.path.join(_script_dir, '..', 'python'))
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Swarm Satellite Properties
# ---------------------------------------------------------------------------

from ballistic.satellite_properties import get as _sat_props
SWARM_PROPERTIES = {k: _sat_props(f'SWARM-{k[-1]}').legacy_dict()
                    for k in ('Swarm-A', 'Swarm-B', 'Swarm-C')}

# Swarm SP3 satellite ids -> names (the ids vary by product type)
SAT_ID_MAP = {
    # SP3xCOM product IDs
    'A': 'Swarm-A', 'B': 'Swarm-B', 'C': 'Swarm-C',
    'SWA': 'Swarm-A', 'SWB': 'Swarm-B', 'SWC': 'Swarm-C',
    'L61': 'Swarm-A', 'L62': 'Swarm-B', 'L63': 'Swarm-C',
}

CONFIG_LABELS = {
    'baseline': 'Baseline (no SRP/drag)',
    'srp': 'SRP only',
    'drag': 'Drag only',
    'both': 'SRP + Drag',
}


# ---------------------------------------------------------------------------
# Reference orbits: read with ballistic.sp3 through scripts/_sp3_records.py
# ---------------------------------------------------------------------------

import _sp3_records as _sp3r  # noqa: E402
import _sp3_velocity as _sp3v  # noqa: E402


def load_sp3_data(sp3_paths: List[str], target_sat_ids: List[str] = None,
                  sat_id_override: str = None) -> Dict[str, list]:
    """Read SP3 files into {satellite name: [arc per file]} (see _sp3_records.load_arcs).

    `sp3_paths` are file paths; `target_sat_ids` limits the SP3 ids read
    (None = all); `sat_id_override` reads one id and names it from SAT_ID_MAP
    when known. Each file stays its own arc; epochs honor the file's time
    system. Velocity records are checked against the positions
    (_sp3_velocity.verify, which raises SystemExit on a mismatch) unless
    velocities are derived.
    """
    import ballistic as _ballistic
    id_map = ({sat_id_override: SAT_ID_MAP.get(sat_id_override, sat_id_override)}
              if sat_id_override else SAT_ID_MAP)
    velocity = 'derived' if _sp3v.derive_requested() else 'file'
    data = _sp3r.load_arcs(_ballistic, sp3_paths, target_sat_ids, id_map, velocity)
    for arcs in data.values():
        for arc in arcs:
            if arc['records'][0]['vel_from_file']:
                _sp3v.verify(_ballistic, arc, f"{os.path.basename(arc['source'])} {arc['satellite']}")
    return data


# ---------------------------------------------------------------------------
# Drag environment (density, wind, β) helpers, shared in scripts/_drag_env_helper.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_WEATHER_ARG = 'live'   # set from --weather in main()
from _drag_env_helper import (  # noqa: E402
    ALT_GRID_SWARM_KM,
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

    Earth gravity (degree from the registry, else 70), Sun/Moon/Jupiter
    gravity, relativity, ITRF2020, RK45 integrator; SRP added for 'srp' and
    'both'. Drag configs leave drag_env unset here: _make_arc_config attaches
    the drag environment for each start. An unknown `sat_name` raises
    KeyError.
    """
    props = SWARM_PROPERTIES[sat_name]   # unknown satellite: KeyError

    config = ballistic.TrajectoryConfig(
        gravity_degree=props.get('gravity_degree', 70),
        enable_third_body=True,
        enable_relativistic=True,
        itrf=ballistic.ITRFRealization.ITRF2020,
        integrator=ballistic.IntegratorType.RK45,
    )

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
    props = SWARM_PROPERTIES[sat_name]   # unknown satellite: KeyError
    beta_kg_m2 = props['mass'] / (props['cd'] * props['drag_area'])
    return _shared_make_arc_config(
        ballistic, base_config,
        drag_on=(config_name in _DRAG_ON_CONFIGS),
        start_epoch=start_epoch, pos_ecef=pos_ecef, vel_ecef=vel_ecef,
        duration_s=duration_s, beta_kg_m2=beta_kg_m2,
        altitude_grid_km=ALT_GRID_SWARM_KM, weather=weather)


# ---------------------------------------------------------------------------
# Propagation Engine
# ---------------------------------------------------------------------------

def run_validation(
    sp3_paths: List[str],
    config_names: List[str],
    sample_interval_min: int = 30,
    duration_sec: float = 3600.0,
    sat_id_override: str = None,
    ephemeris_path: str = None,
) -> pd.DataFrame:
    """
    Propagate every start in the SP3 files for each configuration and return the errors.

    Parameters: `sp3_paths` (list of SP3 paths), `config_names` (from
    CONFIG_LABELS), `sample_interval_min` (minutes between starts),
    `duration_sec` (prediction length, s), `sat_id_override` (SP3 id to read
    instead of the known ones) and `ephemeris_path` (optional JPL ephemeris).

    Loads Earth orientation parameters (EOP: the measured irregularities of
    the Earth's rotation; warns without them) and the --weather space weather
    (raises if it fails; there is no quiet-weather fallback). Prints the
    summary tables. Returns a pandas DataFrame with columns satellite,
    epoch_utc (start label), config and error_m (end-point position error,
    meters). Exits (SystemExit) if no satellite is found in the files.
    """
    import ballistic

    ballistic.initialize()

    loader = ballistic.Loader()
    if loader.load_eop():
        print("Loaded EOP data")
    else:
        print("WARNING: No EOP data loaded")

    # Load space weather (needed by the NRLMSIS 2.1 density model).
    # No fallback: drag without measured weather is not what this script measures.
    _src = load_weather_source(ballistic, globals().get('_WEATHER_ARG', 'live'))
    print(f"Loaded space weather: {_src}")

    if ephemeris_path and os.path.exists(ephemeris_path):
        ballistic.load_jpl_ephemeris(ephemeris_path)
        print(f"Loaded external ephemeris: {ephemeris_path}")

    # Collect all known Swarm SP3 IDs
    target_sp3_ids = None
    if sat_id_override:
        target_sp3_ids = [sat_id_override]
    else:
        target_sp3_ids = list(SAT_ID_MAP.keys())

    print("\nLoading SP3 data...")
    print(f"SP3 IDs to search: {target_sp3_ids}")

    all_data = load_sp3_data(sp3_paths, target_sp3_ids, sat_id_override)

    if not all_data:
        print("ERROR: No satellite data found in SP3 files!")
        print("Use --sat-id to specify the correct satellite ID")
        sys.exit(1)

    # Filter to known Swarm satellites
    swarm_names = list(SWARM_PROPERTIES.keys())
    filtered_data = {k: v for k, v in all_data.items() if k in swarm_names}
    if not filtered_data and all_data:
        print(f"WARNING: No satellites matched Swarm names {swarm_names}")
        print(f"  Using all found satellites: {list(all_data.keys())}")
        filtered_data = all_data

    print(f"\nConfigurations: {[CONFIG_LABELS[c] for c in config_names]}")
    print(f"Propagation duration: {duration_sec:.0f}s")
    print(f"Sample interval: {sample_interval_min} min")

    rows = []
    sample_interval_sec = sample_interval_min * 60

    for sat_name, arcs in filtered_data.items():
        n_records = sum(len(a['records']) for a in arcs)
        print(f"\n--- {sat_name}: {n_records} SP3 records ---")

        if n_records < 2:
            print("  Insufficient data, skipping")
            continue

        total_span = sum(float(a['offsets'][-1]) for a in arcs)
        r0 = np.linalg.norm(arcs[0]['records'][0]['pos_m'])
        # geodetic altitude of the (Earth-fixed) first record
        alt_km = ballistic.ecef_to_geodetic(np.asarray(arcs[0]['records'][0]['pos_m'], float)).altitude / 1e3
        print(f"  Radius: {r0/1000:.1f} km, altitude: {alt_km:.0f} km")
        print(f"  Time span: {total_span/3600:.1f} hours")

        configs = {name: make_config(ballistic, name, sat_name) for name in config_names}

        n_samples = 0
        n_errors = 0

        for s in _sp3r.starts(arcs, sample_interval_sec, duration_sec,
                              skip_edges=False):
            sv_start, sv_end = s['start'], s['end']
            start_epoch = s['epoch']
            actual_duration = s['duration']
            pos_ecef = sv_start['pos_m']
            vel_ecef = _sp3v.start_velocity(ballistic, sv_start, start_epoch,
                                            ballistic.ITRFRealization.ITRF2020)
            truth_pos = sv_end['pos_m']
            epoch_str = s['label']

            for config_name, base_config in configs.items():
                try:
                    config = _make_arc_config(
                        ballistic, base_config, config_name, sat_name,
                        start_epoch, pos_ecef, vel_ecef, actual_duration,
                        get_weather_for_epoch(ballistic, start_epoch))
                    traj = ballistic.Trajectory(ballistic.State.from_itrf(start_epoch, pos_ecef, vel_ecef), config=config)
                    traj.propagate_for(actual_duration)
                    pred_pos = traj.state_ecef().position
                    error_m = np.linalg.norm(pred_pos - truth_pos)

                    rows.append({
                        'satellite': sat_name,
                        'epoch_utc': epoch_str,
                        'config': config_name,
                        'error_m': error_m,
                    })
                    n_samples += 1
                except Exception as e:
                    n_errors += 1
                    if n_errors <= 3:
                        print(f"  WARNING: {config_name} failed at {epoch_str}: {e}")
                    elif n_errors == 4:
                        print("  (suppressing further warnings)")


        n_per_config = n_samples // max(len(config_names), 1)
        print(f"  Propagated {n_per_config} samples per config ({n_errors} errors)")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nWARNING: No propagation results!")
        return df

    before = len(df)
    df = df.drop_duplicates(subset=['satellite', 'epoch_utc', 'config'], keep='first')
    after = len(df)
    if before != after:
        print(f"Deduplicated: {before} -> {after}")

    # Print summary
    print(f"\n{'='*70}")
    print("SWARM DRAG VALIDATION SUMMARY")
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

        baseline_rmse = None
        for cfg_name in config_names:
            cfg_df = df[df['config'] == cfg_name]
            if cfg_df.empty:
                continue
            rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
            mean = cfg_df['error_m'].mean()
            median = cfg_df['error_m'].median()
            mx = cfg_df['error_m'].max()
            sub1m = 100 * (cfg_df['error_m'] < 1.0).mean()

            if baseline_rmse is None:
                baseline_rmse = rmse
            change = f" ({100*(rmse - baseline_rmse)/baseline_rmse:+.1f}%)" if baseline_rmse and cfg_name != 'baseline' else ""

            print(f"{CONFIG_LABELS[cfg_name]:<25} {rmse:>8.3f} {mean:>8.3f} {median:>8.3f} {mx:>8.3f} {sub1m:>5.1f}%{change}")

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

        y_ticks = [-2, -1, 0, 1, 2]
        y_labels = ['0.01', '0.1', '1', '10', '100']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        rmse = np.sqrt((cfg_df['error_m'] ** 2).mean())
        sub1m = 100 * (cfg_df['error_m'] < 1.0).mean()
        ax.set_title(f"{CONFIG_LABELS[cfg_name]}\nRMSE: {rmse:.3f}m, <1m: {sub1m:.0f}%",
                      fontsize=11, fontweight='bold')
        ax.set_xlabel('Satellite')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Position Error (meters)')

    fig.suptitle('Swarm 1-Hour Prediction Error: Drag/SRP Comparison (460-530 km)',
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
        description='Validate atmospheric drag at 460 km using Swarm SP3 orbits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Swarm flies at 460-530 km where atmospheric drag is ~100x stronger than at
Sentinel's 700 km altitude. This tests whether enabling drag modeling improves
1-hour propagation accuracy.

Data acquisition:
    Swarm SP3 data requires free registration at:
    https://swarm-diss.eo.esa.int/

    Download SP3xCOM_2_ products to tests/data/sp3/swarm/
    See tests/data/sp3/swarm/README.md for detailed instructions.

Examples:
    python scripts/validate_swarm_drag.py --sp3 tests/data/sp3/swarm/
    python scripts/validate_swarm_drag.py --sp3 tests/data/sp3/swarm/ --configs baseline drag
    python scripts/validate_swarm_drag.py --mode plot --hdf5 swarm_drag_validation.h5
        """,
    )
    parser.add_argument('--sp3', nargs='+', help='SP3 file(s) or directory')
    parser.add_argument('--sat-id', type=str, default=None,
                        help='Override SP3 satellite ID')
    parser.add_argument('--mode', choices=['propagate', 'plot', 'both'], default='both')
    parser.add_argument('--sample-interval', type=int, default=30,
                        help='Minutes between samples (default: 30)')
    parser.add_argument('--duration', type=float, default=3600.0,
                        help='Propagation duration in seconds (default: 3600)')
    parser.add_argument('--hdf5', default='swarm_drag_validation.h5',
                        help='HDF5 output path')
    parser.add_argument('--output', default='swarm_drag_validation.png',
                        help='Plot output path')
    parser.add_argument('--ephemeris', default=None,
                        help='Path to external JPL ephemeris (optional)')
    parser.add_argument(
        '--configs', nargs='+', default=['baseline', 'srp', 'drag', 'both'],
        choices=['baseline', 'srp', 'drag', 'both'],
        help='Configurations to compare (default: all four)',
    )

    add_weather_arg(parser)
    _sp3v.add_arg(parser, default='file')   # records used as written
    args = parser.parse_args()
    _sp3v.set_mode(args.sp3_velocity)
    globals()['_WEATHER_ARG'] = args.weather

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
            sp3_paths, args.configs, args.sample_interval,
            args.duration, args.sat_id, args.ephemeris,
        )
        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed:.1f} seconds")

        if not df.empty:
            save_results(df, args.hdf5)

    if args.mode in ('plot', 'both'):
        if args.mode == 'plot':
            df = load_results(args.hdf5)
        if not df.empty:
            make_violin_plot(df, args.output, args.configs)


if __name__ == '__main__':
    main()
