#!/usr/bin/env python3
"""
LAGEOS & Etalon 1-hour prediction accuracy, with and without solar radiation pressure.

Solar radiation pressure (SRP) is the small push sunlight gives a satellite.
The simplest model, the "cannonball" model, treats the satellite as a sphere
of known mass, cross-section area and reflectivity coefficient Cr. LAGEOS-1/2
and Etalon-1/2 are spheres covered in laser retroreflectors in medium Earth
orbit (thousands of km up), so for them the cannonball model is
geometrically exact. Air drag is negligible at these heights and is left off.

For each start time the script takes the position and velocity from an ILRS
(International Laser Ranging Service) reference orbit file (SP3, the standard precise-orbit text format),
propagates for --duration seconds, and measures the distance to the
reference position at the end. Two configurations:
  baseline  GCRS inertial frame + EGM2008 gravity (degree from the satellite
            registry, else 8) + third-body gravity (Sun, Moon, Jupiter) +
            relativity + solid Earth tides, no SRP
  srp       the same + cannonball SRP with the registry's mass, area and Cr
The result is claim SLR-1H in docs/validation_claims.md (satellite laser
ranging, SLR).

Start velocities are by default differentiated from the positions
(--sp3-velocity derived), because ILRS velocity records are not the true
derivative of their positions (see _sp3_velocity.py).

Inputs: SP3 files or directories (--sp3); outputs: a printed summary per
configuration (sample count, mean/median/RMSE/max error, share under 1 m,
per-satellite RMSE, comparison table), the per-start errors in an HDF5 file
(--hdf5) and a violin plot of the error distributions (--output). RMSE is
the root-mean-square of the end-point position errors.

Options (defaults in brackets):
  --sp3 PATH [PATH ...]       SP3 files or directories (*.sp3, *.SP3, *.sp3c, *.SP3c);
                              required unless --mode plot
  --satellite {lageos,etalon,all}   satellite family [all]
  --sat-id ID                 use only this SP3 satellite id (e.g. L07)
  --mode {propagate,plot,both}      [both]; 'plot' reads --hdf5 instead of propagating
  --sample-interval MIN       minutes between start times [30]
  --duration S                prediction length, seconds [3600]
  --hdf5 PATH                 results file [sphere_srp_validation.h5]
  --output PATH               plot file [sphere_srp_validation.png]
  --ephemeris PATH            external JPL planetary ephemeris (.bsp) [embedded DE440]
  --configs {baseline,srp} [...]    configurations to run [baseline srp]
  --sp3-velocity {file-no-cip-drift,file,derived}   start-velocity source [derived]

Usage (the SLR-1H reproduction command; needs pandas and h5py):
    PYTHONPATH=./python uv run --with numpy --with pandas --with h5py \\
        python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/lageos tests/data/sp3/etalon \\
        --satellite all --mode propagate --duration 3600 --sample-interval 60
Other examples:
    python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/lageos/ --satellite lageos
    python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/etalon/ --satellite etalon
    python scripts/validate_sphere_srp.py --mode plot --hdf5 sphere_srp_validation.h5
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict

_script_dir = os.path.dirname(os.path.abspath(__file__))
import _source_tree  # noqa: E402
_source_tree.use_source_tree()

import numpy as np

import _sp3_velocity as _sp3v
import pandas as pd


# ---------------------------------------------------------------------------
# Satellite properties: the library registry (include/ballistic/satellites.hpp,
# cited). Script display names -> registry names.
# ---------------------------------------------------------------------------

from ballistic.satellite_properties import get as _sat_props  # noqa: E402

_REGISTRY_NAME = {
    'LAGEOS-1': 'LAGEOS-1', 'LAGEOS-2': 'LAGEOS-2',
    'Etalon-1': 'ETALON-1', 'Etalon-2': 'ETALON-2',
}
SATELLITE_PROPERTIES = {name: _sat_props(reg).legacy_dict()
                        for name, reg in _REGISTRY_NAME.items()}

# SP3 satellite ids -> names (each orbit-analysis center may use different ids)
SAT_ID_MAP = {
    'L51': 'LAGEOS-1', 'L07': 'LAGEOS-1', 'L01': 'LAGEOS-1', 'La1': 'LAGEOS-1',
    'L52': 'LAGEOS-2', 'L02': 'LAGEOS-2', 'La2': 'LAGEOS-2',
    'L53': 'Etalon-1', 'E17': 'Etalon-1', 'E01': 'Etalon-1', 'Et1': 'Etalon-1',
    'L54': 'Etalon-2', 'E07': 'Etalon-2', 'E02': 'Etalon-2', 'Et2': 'Etalon-2',
}

# Satellite family filters
SAT_FAMILIES = {
    'lageos': ['LAGEOS-1', 'LAGEOS-2'],
    'etalon': ['Etalon-1', 'Etalon-2'],
    'all': list(SATELLITE_PROPERTIES.keys()),
}

CONFIG_LABELS = {
    'baseline': 'Baseline (no SRP)',
    'srp': 'With SRP',
}


# ---------------------------------------------------------------------------
# Reference orbits: read with ballistic.sp3 through scripts/_sp3_records.py
# ---------------------------------------------------------------------------

import _sp3_records as _sp3r  # noqa: E402


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
# Config Builder
# ---------------------------------------------------------------------------

def make_config(ballistic, config_name: str, sat_name: str):
    """Return the TrajectoryConfig for `config_name` ('baseline' or 'srp') and `sat_name`.

    `sat_name` is a key of SATELLITE_PROPERTIES (e.g. 'LAGEOS-1'); an unknown
    name raises KeyError rather than using a default. The 'srp' config adds
    SrpConfig(mass, area, Cr) from the registry.
    """
    props = SATELLITE_PROPERTIES[sat_name]   # KeyError = loud: no silent default

    # LAGEOS/Etalon are MEO/high altitude — drag is negligible, leave off (default).
    config = ballistic.TrajectoryConfig(
        gravity_degree=props.get('gravity_degree', 8),
        enable_third_body=True,
        enable_relativistic=True,
        itrf=ballistic.ITRFRealization.ITRF2020,
        integrator=ballistic.IntegratorType.RK45,
    )

    if config_name == 'srp':
        config = config.replace(srp=ballistic.SrpConfig(
            mass_kg=props['mass'], area_m2=props['srp_area'],
            cr=1.0 + props['srp_reflectivity']))

    return config


# ---------------------------------------------------------------------------
# Propagation Engine
# ---------------------------------------------------------------------------

def run_validation(
    sp3_paths: List[str],
    config_names: List[str],
    satellite_filter: str = 'all',
    sample_interval_min: int = 30,
    duration_sec: float = 3600.0,
    sat_id_override: str = None,
    ephemeris_path: str = None,
) -> pd.DataFrame:
    """
    Propagate every start in the SP3 files and return the end-point errors.

    Parameters:
      sp3_paths            list of SP3 file paths.
      config_names         configurations to run ('baseline', 'srp').
      satellite_filter     'lageos', 'etalon' or 'all'.
      sample_interval_min  minutes between start times.
      duration_sec         prediction length, seconds.
      sat_id_override      read only this SP3 satellite id (None = the
                           known ids of the chosen family).
      ephemeris_path       optional JPL planetary ephemeris file (.bsp).

    Loads Earth orientation parameters (EOP: the measured irregularities of
    the Earth's rotation) with ballistic.Loader and warns without them.
    Prints the summary tables. Returns a pandas DataFrame with columns
    satellite, epoch_utc (start label), config, error_m (end-point position
    error, meters), one row per start and configuration. Exits (SystemExit)
    if no satellite is found in the files.
    """
    import ballistic

    ballistic.initialize()

    loader = ballistic.Loader()
    if loader.load_eop():
        print("Loaded EOP data")
    else:
        print("WARNING: No EOP data loaded")

    if ephemeris_path and os.path.exists(ephemeris_path):
        ballistic.load_jpl_ephemeris(ephemeris_path)
        print(f"Loaded external ephemeris: {ephemeris_path}")

    # Determine target satellite IDs for filtering
    target_family = SAT_FAMILIES.get(satellite_filter, SAT_FAMILIES['all'])
    target_sp3_ids = None
    if sat_id_override:
        target_sp3_ids = [sat_id_override]
    else:
        # Collect all known SP3 IDs for the target family
        target_sp3_ids = []
        for sp3_id, sat_name in SAT_ID_MAP.items():
            if sat_name in target_family:
                target_sp3_ids.append(sp3_id)

    print("\nLoading SP3 data...")
    print(f"Target satellites: {target_family}")
    print(f"SP3 IDs to search: {target_sp3_ids}")

    all_data = load_sp3_data(sp3_paths, target_sp3_ids, sat_id_override)

    if not all_data:
        print("ERROR: No satellite data found in SP3 files!")
        print("Use --sat-id to specify the correct satellite ID")
        sys.exit(1)

    # Filter to target family
    filtered_data = {k: v for k, v in all_data.items() if k in target_family}
    if not filtered_data and all_data:
        # If ID mapping didn't work, use whatever we found
        print(f"WARNING: No satellites matched family filter {target_family}")
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

        # Compute time offsets from first record
        total_span = sum(float(a['offsets'][-1]) for a in arcs)
        r0 = np.linalg.norm(arcs[0]['records'][0]['pos_m'])
        # geodetic altitude of the (Earth-fixed) first record
        alt_km = ballistic.ecef_to_geodetic(np.asarray(arcs[0]['records'][0]['pos_m'], float)).altitude / 1e3
        print(f"  Radius: {r0/1000:.1f} km, altitude: {alt_km:.0f} km")
        print(f"  Time span: {total_span/3600:.1f} hours")

        # Build configs
        configs = {name: make_config(ballistic, name, sat_name) for name in config_names}

        # Sample propagation windows
        max_start_offset = total_span - duration_sec
        if max_start_offset <= 0:
            print("  Time span too short for propagation duration, skipping")
            continue

        n_samples = 0
        n_errors = 0

        for s in _sp3r.starts(arcs, sample_interval_sec, duration_sec,
                              skip_edges=True):
            sv_start, sv_end = s['start'], s['end']
            start_epoch = s['epoch']
            actual_duration = s['duration']
            pos_ecef = sv_start['pos_m']
            vel_ecef = _sp3v.start_velocity(
                ballistic, sv_start, start_epoch, ballistic.ITRFRealization.ITRF2020)
            truth_pos = sv_end['pos_m']
            epoch_str = s['label']

            for config_name, config in configs.items():
                try:
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

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=['satellite', 'epoch_utc', 'config'], keep='first')
    after = len(df)
    if before != after:
        print(f"Deduplicated: {before} -> {after}")

    # Print summary
    print(f"\n{'='*70}")
    print("SPHERE SRP VALIDATION SUMMARY")
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
            print(f"    {sat}: {sat_rmse:.3f} m (n={len(sat_df)})")

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
    """Save a violin plot (error distribution per satellite, log scale) to `output_path`.

    One panel per configuration in `config_names`; dashed and dotted lines
    mark 1 m and 0.5 m. Needs matplotlib and seaborn.
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

    colors = {'baseline': '#4C72B0', 'srp': '#DD8452'}

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
        median = cfg_df['error_m'].median()
        ax.set_title(f"{CONFIG_LABELS[cfg_name]}\nRMSE: {rmse:.3f}m, Median: {median:.3f}m",
                      fontsize=11, fontweight='bold')
        ax.set_xlabel('Satellite')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Position Error (meters)')

    # Determine title from satellites present
    sats = df['satellite'].unique()
    has_lageos = any('LAGEOS' in s for s in sats)
    has_etalon = any('Etalon' in s for s in sats)
    if has_lageos and has_etalon:
        title = 'LAGEOS & Etalon 1-Hour Prediction: Cannonball SRP Validation'
    elif has_lageos:
        title = 'LAGEOS 1-Hour Prediction: Cannonball SRP Validation'
    else:
        title = 'Etalon 1-Hour Prediction: Cannonball SRP Validation'

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
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
        description='Validate cannonball SRP model on geodetic sphere satellites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
LAGEOS and Etalon are perfect spheres, making them ideal for cannonball SRP
validation. Unlike Sentinel (where cannonball SRP degrades accuracy by 48%
due to box-wing model mismatch), these satellites' SRP is genuinely isotropic.

Examples:
    python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/lageos/
    python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/lageos/ --satellite lageos
    python scripts/validate_sphere_srp.py --sp3 tests/data/sp3/etalon/ --satellite etalon
    python scripts/validate_sphere_srp.py --mode plot --hdf5 sphere_srp_validation.h5
        """,
    )
    parser.add_argument('--sp3', nargs='+', help='SP3 file(s) or directory')
    parser.add_argument('--satellite', choices=['lageos', 'etalon', 'all'], default='all',
                        help='Satellite family filter (default: all)')
    parser.add_argument('--sat-id', type=str, default=None,
                        help='Override SP3 satellite ID (e.g., L07)')
    parser.add_argument('--mode', choices=['propagate', 'plot', 'both'], default='both')
    parser.add_argument('--sample-interval', type=int, default=30,
                        help='Minutes between samples (default: 30)')
    parser.add_argument('--duration', type=float, default=3600.0,
                        help='Propagation duration in seconds (default: 3600)')
    parser.add_argument('--hdf5', default='sphere_srp_validation.h5',
                        help='HDF5 output path')
    parser.add_argument('--output', default='sphere_srp_validation.png',
                        help='Plot output path')
    parser.add_argument('--ephemeris', default=None,
                        help='Path to external JPL ephemeris (optional)')
    parser.add_argument('--configs', nargs='+', default=['baseline', 'srp'],
                        choices=['baseline', 'srp'],
                        help='Configurations to compare (default: baseline srp)')

    _sp3v.add_arg(parser, default='derived')   # ILRS records are not true derivatives
    args = parser.parse_args()
    _sp3v.set_mode(args.sp3_velocity)

    if args.mode in ('propagate', 'both'):
        if not args.sp3:
            parser.error("--sp3 is required for propagation mode")

        # Expand directories to SP3 files
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
            sp3_paths, args.configs, args.satellite,
            args.sample_interval, args.duration,
            args.sat_id, args.ephemeris,
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
