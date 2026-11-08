"""The best-accuracy configuration in one call: ``recommended_config``.

The library's defaults already are the most accurate general settings
(propagation in the inertial GCRS frame, the EGM2008 gravity field to degree
70 — degrees 100-160 measured no better — the Sun, Moon and Jupiter, solid
Earth tides, ocean tides once ``ballistic.load()`` has fetched them,
relativity, the RK45 integrator). What a default cannot know is the
satellite: solar radiation pressure (SRP, the push of sunlight) needs its
mass and area, and drag needs its ballistic coefficient beta = m / (Cd * A)
(mass over drag coefficient times area, kg/m^2) and the space weather (the
solar-flux and geomagnetic indices F10.7 and Ap). ``recommended_config``
fills exactly those in: name the satellite, or give its mass and areas, and
get the configuration that measured best on a full year of Sentinel data
(docs/validation_claims.md, SENTINEL-YEAR-2025).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from . import _core
from . import satellite_properties as _sp
from .nrl import nrl_models
from .trajectory import TrajectoryConfig

# Drag is turned on below 1500 km altitude (the top of the drag generators'
# standard altitude grid); above it the air is too thin to matter.
DRAG_CEILING_KM = 1500.0


def recommended_config(
    state,
    satellite: str | _sp.SatelliteProperties | None = None,
    *legacy,
    mass_kg: float | None = None,
    srp_area_m2: float | None = None,
    drag_area_m2: float | None = None,
    cd: float | None = None,
    cr: float | None = None,
    duration_s: float = 3600.0,
    allow_predicted_weather: bool = True,
    reentry: "_core.AeroShape | None" = None,
    reference_length_m: float | None = None,
    accuracy: str = "best",
    horizon_s: float | None = None,
    beta_kg_m2: float | None = None,
    tle=None,
    drag: bool | None = None,
    **overrides,
) -> TrajectoryConfig:
    """A TrajectoryConfig with radiation pressure and drag set up for one satellite.

    Name a satellite from ``ballistic.satellite_properties`` (``"S1A"``,
    ``"S2B"``, ``"S3A"``, ``"S6A"``, ``"CHAMP"``, ...) or give its mass and
    areas. On top of the library defaults this turns on:

    * radiation pressure (solar radiation pressure, sunlight reflected by the
      Earth (albedo) and the Earth's infrared glow) from the satellite's
      radiation area — the registry holds EFFECTIVE areas fitted on 2025
      Sentinel data; for your own area ``cr`` defaults to 1.3;
    * drag, with air density from the NRLMSIS 2.1 atmosphere model and wind
      from the HWM14 wind model, evaluated at the satellite's position and
      driven by the MEASURED space weather (``ballistic.load()`` fetches
      it). There is no arc or horizon to choose (``duration_s`` is unused).
      For days the loaded file has not measured yet it uses CelesTrak's
      forecast and the trajectory warns (``allow_predicted_weather=False``
      stops at the last measured day instead). Drag is left off above
      1500 km. The two models are a separate, non-commercial plugin
      (``ballistic.install_nrl_models``); when it is not installed, drag is
      left off with one ``ConfigHintWarning`` saying what that costs (see
      ``drag``).

    Args:
        state: the start, a ``ballistic.State``. Its ITRF realization (the
            version of the Earth-fixed frame), if it names one, becomes the
            config's.
        satellite: a registry name (str), a SatelliteProperties, or None.
            Values you pass explicitly override the registry's.
        mass_kg: mass, kg.
        srp_area_m2: area facing the Sun for radiation pressure, m^2.
        drag_area_m2: mean frontal area facing the direction of motion, m^2
            (not the total surface area). Needed below 1500 km.
        cd: drag coefficient, dimensionless (default 2.2, or the registry's
            value). Refused together with ``reentry``, which takes Cd from the
            flow.
        cr: radiation-pressure coefficient, 1 (absorbs all) to 2 (mirror);
            default 1.3, or the registry's value.
        duration_s: unused; kept for compatibility.
        allow_predicted_weather: use the space-weather forecast past the last
            measured day (default True, with a warning) instead of stopping.
        reentry: for a reentry, an ``ballistic.aero.AeroShape`` (SPHERE,
            FLAT_PLATE or CYLINDER); see below.
        reference_length_m: the body's size for ``reentry`` (a sphere's
            diameter), meters.
        accuracy: "best" (default) or "tle"; see below.
        horizon_s: with accuracy="tle", how far you will propagate, seconds.
        beta_kg_m2: with accuracy="tle", the ballistic coefficient m/(Cd*A)
            in kg/m^2.
        tle: with accuracy="tle", the parsed two-line element set (TLE)
            whose drag term B* gives the ballistic coefficient.
        drag: None (default) turns drag on where it is needed if the NRL
            models are installed, and otherwise leaves it off with a
            ``ConfigHintWarning``; True requires it (ConfigError when the
            models are not installed); False leaves drag off without a word.
        **overrides: passed on to ``TrajectoryConfig`` (e.g. ``itrf=...`` to
            match your reference data).

    Returns:
        A frozen ``ballistic.TrajectoryConfig``.

    Reentry: pass ``reentry=ballistic.aero.AeroShape.SPHERE`` (or ``FLAT_PLATE``
    / ``CYLINDER``) and ``reference_length_m``. The drag coefficient then
    follows the flow (``FlowRegimeCd``: rarefied, "free-molecular" flow in
    orbit, dense "continuum" flow deep in the atmosphere, where depends on
    the body's size) with ``drag_area_m2`` as the reference area, instead of
    a constant ``cd`` (giving ``cd`` raises when no satellite is named; with
    a named satellite its registry ``cd`` is not used). ``srp_area_m2`` becomes optional (radiation pressure is
    negligible next to drag in the final orbits; it is left off when no area
    is given). Fly it with ``traj.propagate_until(target)``, which stops at
    the ground impact.

    TLE start (``State.from_tle``): pass ``accuracy="tle"`` and the
    ``horizon_s`` you will propagate. Such a state is only good to ~0.5 km
    (Flohrer et al. 2009), so the forces and gravity detail whose MEASURED
    effect over that horizon stays below 10% of that are dropped, which is
    faster (``ballistic.tle.tle_grade_preset(state, horizon_s)`` shows the
    choice, the measured effects and the speed-up;
    scripts/measure_tle_grade_table.py measured them). Where the preset
    keeps radiation pressure it needs ``mass_kg`` and ``srp_area_m2`` (or a
    satellite); where it keeps drag, a ballistic coefficient: the
    satellite's, ``mass_kg`` + ``drag_area_m2``, ``beta_kg_m2``, or ``tle=``
    to take it from B*.

    Raises:
        RemovedApiError: the old ``recommended_config(epoch, pos, vel, ...)`` form.
        TypeError: ``state`` is not a ballistic.State.
        KeyError: an unknown satellite name.
        ConfigError: with accuracy="tle", a force the preset keeps lacks its
            inputs (the message says how large its effect is); or drag is
            required (``drag=True`` or ``reentry=``) and the NRLMSIS 2.1 /
            HWM14 models are not installed (``ballistic.nrl_models()``).
        ValueError: missing mass or areas, an unknown ``accuracy``, TLE-only
            arguments without accuracy="tle", or inconsistent reentry
            arguments.
        RuntimeError: drag is needed but no space weather is loaded — never
            a silent quiet-atmosphere substitute.
        DataCoverageError: the loaded space weather does not cover the start.

    Example::

        ballistic.load()   # once per session: EOP, leap seconds, space weather, ocean tides
        state = ballistic.State.from_itrf(epoch, r_m, v_m_s)
        cfg = ballistic.recommended_config(state, "S1A")
        traj = ballistic.Trajectory(state, cfg)
        traj.propagate_for(3600.0)
    """
    if isinstance(state, _core.Epoch) or legacy:
        from .errors import RemovedApiError
        raise RemovedApiError(
            "recommended_config(epoch, pos, vel, ...) was removed: pass a ballistic.State "
            "(State.from_itrf(epoch, r_m, v_m_s) for an Earth-fixed velocity), i.e. "
            "recommended_config(state, satellite, ...) (see docs/MIGRATION.md).")
    if not isinstance(state, _core.State):
        raise TypeError("recommended_config: the first argument is a ballistic.State")
    if state.realization is not None:
        overrides.setdefault("itrf", state.realization)
    if isinstance(satellite, str):
        satellite = _sp.get(satellite)
    # Whether the CALLER gave cd: the registry merge below fills it in for a
    # named satellite, and reentry= must still refuse a caller's cd then.
    cd_given = cd is not None
    if satellite is not None:
        _hint_legacy_values(satellite, mass_kg, srp_area_m2, drag_area_m2)
        mass_kg = satellite.mass_kg if mass_kg is None else mass_kg
        srp_area_m2 = satellite.srp_area_m2 if srp_area_m2 is None else srp_area_m2
        drag_area_m2 = satellite.drag_area_m2 if drag_area_m2 is None else drag_area_m2
        cd = satellite.cd if cd is None else cd
        if cr is None:
            cr = 1.0 + satellite.srp_reflectivity
    if accuracy == "tle":
        return _tle_grade_config(state, horizon_s, mass_kg, srp_area_m2, drag_area_m2, cd, cr,
                                 beta_kg_m2, tle, allow_predicted_weather, duration_s, drag,
                                 overrides)
    if accuracy != "best":
        raise ValueError(f"recommended_config: accuracy is 'best' or 'tle', got {accuracy!r}")
    if horizon_s is not None or beta_kg_m2 is not None or tle is not None:
        raise ValueError("recommended_config: horizon_s, beta_kg_m2 and tle belong to accuracy='tle'")
    if reentry is not None:
        if cd_given:
            raise ValueError("recommended_config: reentry= takes the drag coefficient from "
                             "the flow (FlowRegimeCd), so the cd given would be ignored; "
                             "give either reentry= or cd, not both")
        if reference_length_m is None or not reference_length_m > 0:
            raise ValueError("recommended_config: reentry= needs reference_length_m (the "
                             "body's size, e.g. a sphere's diameter) for the Knudsen number")
        if mass_kg is None or drag_area_m2 is None:
            raise ValueError("recommended_config: reentry= needs mass_kg and drag_area_m2 "
                             "(the reference area of the shape)")
    elif mass_kg is None or srp_area_m2 is None:
        raise ValueError("recommended_config: name a satellite or give mass_kg "
                         "and srp_area_m2 (and drag_area_m2 below 1500 km)")
    if cr is None:
        cr = 1.3      # SrpConfig's typical-satellite default
    if cd is None:
        cd = 2.2

    if srp_area_m2 is not None:
        overrides.setdefault("srp", _core.SrpConfig(mass_kg=mass_kg, area_m2=srp_area_m2, cr=cr))
    cfg = TrajectoryConfig(**overrides)

    # Altitude above the WGS84 ellipsoid of the given position. For an
    # inertial (GCRS) state this is still close enough: the two frames differ
    # mainly by a rotation about the polar axis, which barely changes the
    # altitude, and the drag on/off decision needs only ~km.
    altitude_km = _core.ecef_to_geodetic(np.asarray(state.position, float)).altitude / 1e3
    if reentry is not None and drag is False:
        raise ValueError("recommended_config: reentry= flies the object down through the "
                         "atmosphere and needs drag; drag=False contradicts it")
    if (altitude_km < DRAG_CEILING_KM or reentry is not None) and _use_drag(
            True if reentry is not None else drag,
            "reentry= needs drag" if reentry is not None else
            f"drag matters below {DRAG_CEILING_KM:.0f} km (this state is at "
            f"{altitude_km:.0f} km)"):
        if drag_area_m2 is None:
            raise ValueError("recommended_config: drag_area_m2 (mean frontal "
                             "area) is needed below 1500 km")
        if reentry is not None:
            beta = _core.AeroBeta(mass_kg, drag_area_m2, _core.FlowRegimeCd(reentry),
                                  reference_length_m=reference_length_m)
        else:
            beta = _core.BetaProfile1D.from_mass_cd_area(mass_kg, cd, drag_area_m2)
        env, _diag = _core.make_drag_environment(
            state.epoch, np.asarray(state.position, float), np.asarray(state.velocity, float),
            duration_s, cfg._to_core(), beta,
            allow_predicted_weather=allow_predicted_weather)
        cfg = dataclasses.replace(cfg, drag_env=env)
    return cfg


def _use_drag(drag, why):
    """Whether to add model-based drag. ``drag``: None = if the NRL models are
    installed (else off, with one ConfigHintWarning), True = required
    (ConfigError when they are not installed), False = off. ``why`` says why
    drag matters here."""
    if drag is False:
        return False
    if nrl_models().installed:
        return True
    install = ("Install them with ballistic.install_nrl_models(accept_license=True) "
               "(NRLMSIS 2.1 is licensed for research, academic and non-profit use only), "
               "or build drag from measured density (ballistic.drag.from_accelerometer_density) "
               "and pass TrajectoryConfig(drag_env=...)")
    if drag is True:
        raise _core.ConfigError(
            f"recommended_config: {why}, but the NRLMSIS 2.1 atmosphere and HWM14 wind models "
            f"are not installed, so there is no model-based drag (a reentry cannot be flown "
            f"without it). {install}.")
    from .errors import ConfigHintWarning
    warnings.warn(
        f"recommended_config: {why}, but the NRLMSIS 2.1 atmosphere and HWM14 wind models are "
        f"not installed, so drag is left off. Below 600 km that multiplies the 1-hour "
        f"prediction error by 4 to 8; near 700 km it changed little at 2026 solar activity. "
        f"{install}; drag=False leaves it off without this warning.",
        ConfigHintWarning, stacklevel=3)
    return False


def _hint_legacy_values(sat, mass_kg, srp_area_m2, drag_area_m2):
    """One ConfigHintWarning when the registry values about to be used are
    unsourced ("legacy"), naming the arguments that replace them. Values the
    caller passed explicitly are not the registry's, so they do not count."""
    from .errors import ConfigHintWarning
    parts = []
    if sat.drag_kind == "legacy" and (mass_kg is None or drag_area_m2 is None):
        parts.append("drag (mass and frontal area; pass mass_kg= and drag_area_m2=)")
    if sat.srp_kind == "legacy" and (mass_kg is None or srp_area_m2 is None):
        parts.append("radiation pressure (mass and radiation area; pass mass_kg= and "
                     "srp_area_m2=)")
    if parts:
        warnings.warn(
            f"recommended_config: the registry's {sat.name} values for "
            f"{' and '.join(parts)} are unsourced (low confidence); the prediction is only "
            f"as good as they are. See its provenance in "
            f"ballistic.satellite_properties.get({sat.name!r}).provenance.",
            ConfigHintWarning, stacklevel=3)


def _tle_grade_config(state, horizon_s, mass_kg, srp_area_m2, drag_area_m2, cd, cr, beta_kg_m2,
                      tle, allow_predicted_weather, duration_s, drag, overrides):
    """recommended_config(accuracy="tle"): apply the TLE-grade preset (forces
    whose effect over ``horizon_s`` is under 10% of a TLE's error budget are
    dropped), then add radiation pressure and drag where the preset keeps
    them. Raises ValueError for a missing horizon and ConfigError for missing
    satellite inputs."""
    if horizon_s is None or not horizon_s > 0:
        raise ValueError("recommended_config(accuracy='tle') needs horizon_s > 0: which forces "
                         "matter depends on how far the state is propagated")
    p = _core.tle_grade_preset(state, float(horizon_s))
    what = (f"{p.regime}, measured horizon {p.horizon_s / 3600:g} h, threshold {p.threshold_m:.0f} m "
            f"(10% of the {p.budget_m:.0f} m TLE error budget)")
    overrides.setdefault("gravity_degree", p.gravity_degree)
    overrides.setdefault("enable_third_body", p.third_body)
    overrides.setdefault("enable_solid_tides", p.tides)
    overrides.setdefault("ocean_tide_degree", 50 if p.ocean_tides else 0)
    overrides.setdefault("enable_relativistic", p.relativity)
    if p.radiation:
        if mass_kg is None or srp_area_m2 is None:
            raise _core.ConfigError(
                f"TLE-grade preset keeps radiation pressure here: it moves a light object (A/m "
                f"0.02 m^2/kg) by up to {p.effect_radiation_m:.0f} m ({what}). Give a satellite or "
                f"mass_kg and srp_area_m2.")
        overrides.setdefault("srp", _core.SrpConfig(mass_kg=mass_kg, area_m2=srp_area_m2,
                                                    cr=1.3 if cr is None else cr))
    else:
        overrides.setdefault("srp", None)
    cfg = TrajectoryConfig(**overrides)
    if not p.drag or not _use_drag(drag, f"the TLE-grade preset keeps drag here (up to "
                                         f"{p.effect_drag_m:.0f} m, {what})"):
        return cfg
    if mass_kg is not None and drag_area_m2 is not None:
        beta = _core.BetaProfile1D.from_mass_cd_area(mass_kg, 2.2 if cd is None else cd, drag_area_m2)
    elif beta_kg_m2 is not None:
        beta = _core.BetaProfile1D.constant(float(beta_kg_m2))
    elif tle is not None:
        if not tle.bstar_per_er > 0:
            raise _core.ConfigError(
                f"TLE-grade preset keeps drag here (up to {p.effect_drag_m:.0f} m, {what}) but this "
                f"TLE's B* is {tle.bstar_per_er:g}, which gives no ballistic coefficient. Give "
                f"beta_kg_m2 (m / (Cd A)) or mass_kg and drag_area_m2.")
        beta = _core.BetaProfile1D.constant(tle.beta_from_bstar_kg_m2())
    else:
        raise _core.ConfigError(
            f"TLE-grade preset keeps drag here: it moves an object with m/(Cd A) = 50 kg/m^2 by up "
            f"to {p.effect_drag_m:.0f} m ({what}). Give a satellite, mass_kg and drag_area_m2, "
            f"beta_kg_m2, or tle= (B*).")
    env, _diag = _core.make_drag_environment(
        state.epoch, np.asarray(state.position, float), np.asarray(state.velocity, float),
        duration_s, cfg._to_core(), beta, allow_predicted_weather=allow_predicted_weather)
    return dataclasses.replace(cfg, drag_env=env)
