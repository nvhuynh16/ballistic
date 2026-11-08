"""Removed public names and their replacements.

Some functions were removed because they let users fall into traps (an
ambiguous frame name, a silent time-scale default, a call that had no
effect). A removed name is not just deleted: using it raises
RemovedApiError naming the replacement and the reason, and
docs/MIGRATION.md lists every change. tests/python/test_removed_api.py
checks that each replacement exists.

REMOVED maps a module-level name of the ballistic package to
(replacement, why). The replacement is a dotted path relative to the
ballistic package.

MOVED maps a name that the top-level package no longer exports to its new
home, a dotted path relative to the ballistic package. The top level keeps
what a user needs in a first session (load data, describe a state,
propagate, the recommended configuration); specialized tools live in
``ballistic.estimation``, ``ballistic.drag``, ``ballistic.aero``,
``ballistic.readers`` and ``ballistic.tle``, so the top level stays short
enough to read. The old name raises RemovedApiError naming the new import.
"""

from .errors import RemovedApiError

REMOVED: dict[str, tuple[str, str]] = {
    "eci_to_ecef": ("gcrs_to_itrf",
                    "its 'ECI' (inertial) frame was the true equator of date, which "
                    "removes only the Earth's daily rotation, not the inertial "
                    "J2000/GCRS frame; the two differ by the slow wobble of the "
                    "Earth's axis (precession) since 2000 (~22 km in low Earth orbit "
                    "in 2013)."),
    "ecef_to_eci": ("itrf_to_gcrs",
                    "its 'ECI' was the true equator of date, not J2000/GCRS "
                    "(see eci_to_ecef)."),
    "load_density_table": ("drag.from_accelerometer_density",
                           "it filled a global density store that no propagation "
                           "read (drag uses a DragEnvironment), so loading measured "
                           "density silently changed nothing. Parse the file with "
                           "ballistic.drag.parse_density_table and build the environment "
                           "with from_accelerometer_density."),
    "load_interpolated_density": ("drag.from_accelerometer_density",
                                  "see load_density_table: the store it filled was "
                                  "never read by any propagation."),
    "splice_isa_density_below": ("drag.extend_density_below",
                                 "it only replaced existing table knots below the floor - "
                                 "the standard 100-1500 km grid has none, so it changed "
                                 "nothing - and its 'US Standard Atmosphere' was a "
                                 "piecewise exponential fit (3.5% high at 60 km)."),
    "LicenceWarning": ("install_nrl_models",
                       "NRLMSIS 2.1 and HWM14 are now a separate plugin, so there is "
                       "no license warning at use time: install_nrl_models("
                       "accept_license=True) shows the NRLMSIS 2.1 license terms before "
                       "installing the plugin."),
    "LicenseWarning": ("install_nrl_models",
                       "NRLMSIS 2.1 and HWM14 are now a separate plugin, so there is "
                       "no license warning at use time: install_nrl_models("
                       "accept_license=True) shows the NRLMSIS 2.1 license terms before "
                       "installing the plugin."),
}


def _tier(module: str, names: str) -> dict[str, str]:
    return {n: f"{module}.{n}" for n in names.split()}


MOVED: dict[str, str] = {
    **_tier("estimation", """
        EmpiricalAccelEntry EAInterp EARepresents BSplineCoefficients
        estimate_ea_bspline EABSplineConfig PositionSamples VelocitySamples
        AccelSamples CVMode GNStatus GNRefine EAResult refine_ea_gn
        EADecomposition BetaStepEstimate BetaResult BetaDecompositionConfig
        decompose_ea refine_beta_from_ea
        RadarObs OpticsObs PositionFixObs AccelObs MultiSensorScenario
        SensorClock SensorMask estimate_ea_smoother EASmootherConfig
        EASmootherResult EASmootherRateBudget
        EAMethod EAProblemSpec EASpecResult estimate_ea information_bound
        EAInformationBound EAInformationBoundOptions reject_heavy_tails
    """),
    **_tier("drag", """
        DragEnvironment DragSample TimeWindow IterationDiag WeatherConditions
        BallisticModel standard_leo_altitude_grid_km from_models_snapshot
        from_models_along from_models_4d from_accelerometer_density
        extend_density_below parse_density_table load_beta_table
        load_empirical_acceleration save_drag_env load_drag_env
    """),
    **_tier("aero", """
        AeroShape ConstantCd FlowRegimeCd MachTable AeroBeta
        free_molecular_cd reemission_ratio henderson1976_sphere_cd
    """),
    **_tier("readers", """
        EOFStateVector EOFFile EOFParser PropagationTestResult parse_eof_auto
        find_and_parse_eof compute_propagation_result check_velocity_records
        VelocityRecordCheck VelocityRecordVerdict cip_drift_velocity
    """),
    **_tier("tle", "TleGradePreset tle_grade_preset"),
}


def raise_moved(name: str) -> None:
    """Raise RemovedApiError for ``name`` (a key of MOVED), naming the
    module it lives in now and the import that reaches it."""
    home = MOVED[name]
    module, attr = home.rsplit(".", 1)
    raise RemovedApiError(
        f"ballistic.{name} moved to ballistic.{home}: use "
        f"`from ballistic.{module} import {attr}` (see docs/MIGRATION.md).")


def raise_removed(name: str) -> None:
    """Raise RemovedApiError for ``name`` (a key of REMOVED), naming its
    replacement and why the old name was removed."""
    replacement, why = REMOVED[name]
    raise RemovedApiError(
        f"ballistic.{name} was removed: {why} Use ballistic.{replacement} instead "
        f"(see docs/MIGRATION.md).")
