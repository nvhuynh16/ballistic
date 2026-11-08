"""How the SP3-driven validation scripts choose a start velocity.

A prediction starts from a position and velocity taken from a reference orbit
file (SP3, the standard precise-orbit text format). The library converts that
Earth-fixed state (ITRF, the frame that rotates with the Earth) into the
inertial frame it integrates in (GCRS). That conversion includes the "CIP
drift": the slow motion of the Earth's rotation axis in space (precession and
nutation), of order 1e-5 m/s at low orbits and larger farther from the
Earth. Because the conversion
already accounts for it, the start velocity must be the true time derivative
of the Earth-fixed positions.

Not every product writes its velocity records that way. Measured by
differentiating each product's own positions (9-point Lagrange derivative at
interior epochs, ballistic.readers.check_velocity_records):

- GFZ CHAMP (2005, 2010) and GRACE-FO (2026) velocity records ARE that
  derivative (mean difference <= 2.5e-6 m/s), so those scripts use the
  records as written ('file'). Subtracting the drift from them would put a
  start-velocity error of the drift's size into every arc, which grows into
  an along-track error over the hour.
- ILRS satellite laser ranging (SLR) records differ from their derivative in
  a way that matches neither convention (check_velocity_records rates them
  inconsistent), so the SLR script derives its start velocities from the positions
  ('derived', its default).

The module-level DEFAULT below only applies to a script that calls add_arg()
without naming its own default; every validation script names one.

Modes:
  'file'               records used as written.
  'derived'            differentiate the positions (accurate mid-file, but
                       millimeter rounding in the file, amplified by the
                       one-sided window, gives larger errors at the first and
                       last epochs).
  'file-no-cip-drift'  records minus the CIP drift (for a product whose
                       records omit it).
"""

import numpy as np

MODES = ('file-no-cip-drift', 'file', 'derived')
DEFAULT = 'file-no-cip-drift'
_mode = DEFAULT


def add_arg(parser, default=DEFAULT):
    """Add the --sp3-velocity option (choices MODES) to an argparse parser.

    Also makes `default` the active mode until set_mode() is called with the
    parsed value.
    """
    global _mode
    _mode = default
    parser.add_argument(
        '--sp3-velocity', choices=MODES, default=default,
        help=f"start velocity: raw records ('file'), records minus the CIP "
             f"drift ('file-no-cip-drift') or derived from positions "
             f"(default here: {default})")


def set_mode(mode):
    """Set the active start-velocity mode (one of MODES) for this process."""
    global _mode
    _mode = mode


def derive_requested():
    """True if the active mode is 'derived' (velocities from the positions)."""
    return _mode == 'derived'


def start_velocity(ballistic, record, epoch, itrf):
    """Return the start velocity for one record under the active mode.

    Parameters: `ballistic` (the module), `record` (a _sp3_records record
    dict), `epoch` (ballistic.Epoch of the record) and `itrf` (the
    ballistic.ITRFRealization of the file). Returns numpy (3,) in m/s,
    Earth-fixed. In 'file-no-cip-drift' mode a velocity read from the file has
    the CIP drift subtracted; otherwise the record's velocity is returned
    unchanged (in 'derived' mode the record already holds the derived value).
    """
    v = np.asarray(record['vel_ms'], dtype=float)
    if _mode == 'file-no-cip-drift' and record.get('vel_from_file'):
        v = v - np.asarray(ballistic._core.cip_drift_velocity(
            epoch, np.asarray(record['pos_m'], dtype=float), itrf))
    return v


def verify(ballistic, arc, label, itrf=None):
    """Stop if a file's velocity records do not suit the chosen mode.

    `arc` is a _sp3_records arc (exact offsets, epoch0 in the file's time
    system); `label` names it in messages; `itrf` is the file's
    ballistic.ITRFRealization (default ITRF2020). Runs
    ballistic.readers.check_velocity_records on the records as written: 'file' needs
    true derivatives and 'file-no-cip-drift' needs records that omit the
    drift; 'derived' ignores the records. Returns the check result, or None
    when nothing was checked ('derived' mode, no velocity records, or fewer
    than 11 records). Raises SystemExit on a mismatch instead of quietly
    biasing every arc.
    """
    if _mode == 'derived':
        return None
    sp3_arc = arc['sp3_arc']
    if not sp3_arc.has_velocity() or len(sp3_arc) < 11:
        return None
    if itrf is None:
        itrf = ballistic.ITRFRealization.ITRF2020
    check = ballistic.readers.check_velocity_records(
        sp3_arc.epoch0, list(sp3_arc.offsets_s), sp3_arc.position, sp3_arc.velocity, itrf)
    V = ballistic.readers.VelocityRecordVerdict
    expected = V.TRUE_DERIVATIVE if _mode == 'file' else V.OMITS_CIP_DRIFT
    if check.verdict != expected:
        raise SystemExit(
            f"{label}: --sp3-velocity {_mode} does not fit this file: {check.summary}. "
            f"Pass --sp3-velocity derived, or the mode the check names.")
    print(f"    velocity records checked: {check.summary}")
    return check
