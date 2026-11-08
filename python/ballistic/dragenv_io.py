"""Saving and loading a ``ballistic.drag.DragEnvironment`` as a numpy ``.npz`` file.

A DragEnvironment describes the atmosphere a satellite flies through: air
density, wind, and the satellite's ballistic coefficient beta = m / (Cd * A)
(kg/m^2), each stored as a "column" that is either a table or a live model.
Building a tabulated env with ``from_models_along`` over several orbits costs
many evaluations of the NRLMSIS 2.1 atmosphere model and the HWM14 wind
model. Validation campaigns and constellation studies build the same envs
again and again; this module saves one to a ``.npz`` file that loads much
faster than it was built.

Supported variants
------------------
``save_drag_env`` / ``load_drag_env`` handle:

- **Density**: ``DensityProfile1D`` (from ``from_models_snapshot``)
  and ``DensityProfile2D`` (from ``from_models_along`` /
  ``from_accelerometer_density``).
- **Wind**: ``WindProfile1D_ENU`` (wind as east/north components, i.e.
  zonal/meridional; empty or populated; from
  ``from_models_snapshot`` / ``from_accelerometer_density``) and
  ``WindProfile2D_ECEF`` (wind in Earth-fixed x/y/z; from ``from_models_along``).
- **Beta**: ``BetaProfile1D`` always.

- **4D fields**: ``DensityField4D`` / ``WindField4D_ECEF`` (from
  ``from_models_4d``).

``BetaProfile2D`` raises ``NotImplementedError``, and so does a density/wind
pairing the loader cannot rebuild: a 1D density takes a 1D wind, a 2D density
a 1D or 2D wind, a 4D density a 4D or empty wind.

A live env (``make_drag_environment``) has no table: it is saved as its
settings (beta, whether forecast space weather is allowed, the altitude below
which the standard atmosphere takes over (``extend_density_below``), wind on
or off) and rebuilt on the space weather loaded when ``load_drag_env`` runs.

NPZ schema
----------
Array names below are shown without their column prefix; in the file they
are ``density_...``, ``wind_...`` or ``beta_...`` (e.g.
``density_altitude_km``, ``wind_zonal_ms``, ``beta_kg_m2``).

- Common: ``density_kind``, ``wind_kind``, ``beta_kind`` (str scalars).
- Density "1d": altitude_km, density_kg_m3.
- Density "2d": altitude_km, epochs_mjd_utc, density_grid (flat row-major
  time × altitude).
- Density "4d": altitude_km, latitude_deg, longitude_deg,
  epochs_mjd_utc, density_grid (flat layout
  ``((t * n_lon + j) * n_lat + i) * n_alt + k``). Values are linear ρ;
  C++ side re-encodes to half-precision log(ρ) on load.
- Wind "1d_enu" / "1d_enu_empty": altitude_km, meridional_ms, zonal_ms
  (or absent for empty).
- Wind "2d_ecef": altitude_km, epochs_mjd_utc, wind_x_ecef, wind_y_ecef,
  wind_z_ecef (parallel flat arrays).
- Wind "4d_ecef": altitude_km, latitude_deg, longitude_deg,
  epochs_mjd_utc, wind_x_ecef, wind_y_ecef, wind_z_ecef (parallel
  flat arrays with the same 4D layout as density).
- Beta "1d": altitude_km, beta_kg_m2.

All time-axis epochs are stored as UTC Modified Julian Dates (MJD, days
since 1858-11-17 00:00) in float64, for portability.
"""

from __future__ import annotations

import numpy as np

from . import _core

# The wind layouts each density layout can be rebuilt with. load_drag_env's
# C++ factories install an empty wind for any other wind kind, so saving one
# would lose the wind without a word: both directions refuse instead.
_SUPPORTED_WIND = {
    "1d": ("1d_enu", "1d_enu_empty"),
    "2d": ("1d_enu", "1d_enu_empty", "2d_ecef"),
    "4d": ("4d_ecef", "1d_enu_empty"),
}


def _check_pairing(density_kind: str, wind_kind: str, action: str) -> None:
    """Raise NotImplementedError for a density/wind pairing the file cannot carry."""
    allowed = _SUPPORTED_WIND[density_kind]
    if wind_kind not in allowed:
        pairings = "; ".join(f"{d} density with {' / '.join(w)} wind"
                             for d, w in _SUPPORTED_WIND.items())
        raise NotImplementedError(
            f"{action}: a {density_kind} density with a {wind_kind} wind cannot be "
            f"stored; it would be rebuilt with no wind at all. Supported pairings: "
            f"{pairings}. Strip the wind with env.without_wind() if it is not "
            f"needed, or rebuild the env with a matching wind layout.")


def save_drag_env(env, path: str) -> None:
    """Save a DragEnvironment to ``path`` as a numpy ``.npz`` file.

    Supports the 1D, 2D and 4D tabulated variants listed in the module
    docstring, and live envs from ``make_drag_environment`` (saved as their
    settings).

    Args:
        env: the DragEnvironment.
        path: output file path (str); numpy appends ``.npz`` if missing.

    Raises:
        NotImplementedError: a beta column other than BetaProfile1D, an
            unsupported density or wind variant, a density/wind pairing the
            loader cannot rebuild (see ``_SUPPORTED_WIND``: a 4D density
            takes only a 4D or empty wind, a 1D density only a 1D wind), or a
            live env built with explicit (not loaded) space weather.
    """
    density = env._density_export()
    wind = env._wind_export()
    beta = env._beta_export()

    if density["kind"] == "live":
        # make_drag_environment's env holds no table, only its settings: save
        # those; load_drag_env rebuilds it on the space weather loaded then.
        if not density["loaded_weather"]:
            raise NotImplementedError(
                "save_drag_env: a live env built with explicit weather cannot "
                "be saved; rebuild it with make_drag_environment instead.")
        if beta["kind"] != "1d":
            raise NotImplementedError("save_drag_env supports 1D beta only.")
        np.savez(path,
                 density_kind=np.array("live"),
                 wind_kind=np.array("live" if wind["kind"] == "live" else "none"),
                 allow_predicted=np.array(bool(density["allow_predicted"])),
                 isa_floor_km=np.array(float(density["isa_floor_km"])),
                 beta_kind=np.array("1d"),
                 beta_altitude_km=np.asarray(beta["altitude_km"], dtype=np.float64),
                 beta_kg_m2=np.asarray(beta["beta_kg_m2"], dtype=np.float64))
        return

    if density["kind"] not in ("1d", "2d", "4d"):
        raise NotImplementedError(
            f"save_drag_env supports 1D / 2D / 4D density variants; got "
            f"density.kind={density['kind']!r}.")
    if beta["kind"] != "1d":
        raise NotImplementedError(
            f"save_drag_env supports 1D beta variants only; got "
            f"beta.kind={beta['kind']!r}. BetaProfile2D is a follow-up.")
    if wind["kind"] not in ("1d_enu", "1d_enu_empty", "2d_ecef", "4d_ecef"):
        raise NotImplementedError(
            f"save_drag_env supports 1D / 2D / 4D wind variants; got "
            f"wind.kind={wind['kind']!r}.")
    _check_pairing(density["kind"], wind["kind"], "save_drag_env")

    save_kwargs = {
        "density_kind": np.array(density["kind"]),
        "wind_kind": np.array(wind["kind"]),
        "beta_kind": np.array("1d"),
        "beta_altitude_km": np.asarray(beta["altitude_km"], dtype=np.float64),
        "beta_kg_m2": np.asarray(beta["beta_kg_m2"], dtype=np.float64),
    }

    if density["kind"] == "1d":
        save_kwargs["density_altitude_km"] = np.asarray(
            density["altitude_km"], dtype=np.float64)
        save_kwargs["density_kg_m3"] = np.asarray(
            density["density_kg_m3"], dtype=np.float64)
    elif density["kind"] == "2d":
        save_kwargs["density_altitude_km"] = np.asarray(
            density["altitude_km"], dtype=np.float64)
        save_kwargs["density_epochs_mjd_utc"] = np.asarray(
            density["epochs_mjd_utc"], dtype=np.float64)
        save_kwargs["density_grid"] = np.asarray(
            density["density_grid"], dtype=np.float64)
    else:  # "4d"
        save_kwargs["density_altitude_km"] = np.asarray(
            density["altitude_km"], dtype=np.float64)
        save_kwargs["density_latitude_deg"] = np.asarray(
            density["latitude_deg"], dtype=np.float64)
        save_kwargs["density_longitude_deg"] = np.asarray(
            density["longitude_deg"], dtype=np.float64)
        save_kwargs["density_epochs_mjd_utc"] = np.asarray(
            density["epochs_mjd_utc"], dtype=np.float64)
        save_kwargs["density_grid"] = np.asarray(
            density["density_grid"], dtype=np.float64)

    if wind["kind"] == "1d_enu":
        save_kwargs["wind_altitude_km"] = np.asarray(
            wind["altitude_km"], dtype=np.float64)
        save_kwargs["wind_meridional_ms"] = np.asarray(
            wind["meridional_ms"], dtype=np.float64)
        save_kwargs["wind_zonal_ms"] = np.asarray(
            wind["zonal_ms"], dtype=np.float64)
    elif wind["kind"] == "2d_ecef":
        save_kwargs["wind_altitude_km"] = np.asarray(
            wind["altitude_km"], dtype=np.float64)
        save_kwargs["wind_epochs_mjd_utc"] = np.asarray(
            wind["epochs_mjd_utc"], dtype=np.float64)
        save_kwargs["wind_x_ecef"] = np.asarray(wind["wind_x_ecef"], dtype=np.float64)
        save_kwargs["wind_y_ecef"] = np.asarray(wind["wind_y_ecef"], dtype=np.float64)
        save_kwargs["wind_z_ecef"] = np.asarray(wind["wind_z_ecef"], dtype=np.float64)
    elif wind["kind"] == "4d_ecef":
        save_kwargs["wind_altitude_km"] = np.asarray(
            wind["altitude_km"], dtype=np.float64)
        save_kwargs["wind_latitude_deg"] = np.asarray(
            wind["latitude_deg"], dtype=np.float64)
        save_kwargs["wind_longitude_deg"] = np.asarray(
            wind["longitude_deg"], dtype=np.float64)
        save_kwargs["wind_epochs_mjd_utc"] = np.asarray(
            wind["epochs_mjd_utc"], dtype=np.float64)
        save_kwargs["wind_x_ecef"] = np.asarray(wind["wind_x_ecef"], dtype=np.float64)
        save_kwargs["wind_y_ecef"] = np.asarray(wind["wind_y_ecef"], dtype=np.float64)
        save_kwargs["wind_z_ecef"] = np.asarray(wind["wind_z_ecef"], dtype=np.float64)
    # "1d_enu_empty" stores nothing for the wind column.

    np.savez_compressed(path, **save_kwargs)


def load_drag_env(path: str):
    """Rebuild a DragEnvironment from an ``.npz`` written by ``save_drag_env``.

    Args:
        path: the ``.npz`` file.

    Returns:
        A ``ballistic.drag.DragEnvironment``. A saved live env is rebuilt on the
        space weather loaded now (``ballistic.load()`` first).

    Raises:
        NotImplementedError: a variant or density/wind pairing this version
            cannot read.
    """
    data = np.load(path, allow_pickle=False)

    density_kind = str(data["density_kind"])
    if density_kind == "live":
        beta = _core.BetaProfile1D(list(data["beta_altitude_km"]), list(data["beta_kg_m2"]))
        return _core.DragEnvironment._live(
            beta, bool(data["allow_predicted"]), float(data["isa_floor_km"]),
            str(data["wind_kind"]) == "live")
    if density_kind not in ("1d", "2d", "4d"):
        raise NotImplementedError(
            f"load_drag_env supports 1D / 2D / 4D density; got {density_kind!r}.")
    beta_kind = str(data["beta_kind"])
    if beta_kind != "1d":
        raise NotImplementedError(
            f"load_drag_env supports 1D beta only; got {beta_kind!r}.")
    wind_kind = str(data["wind_kind"])
    if wind_kind not in ("1d_enu", "1d_enu_empty", "2d_ecef", "4d_ecef"):
        raise NotImplementedError(
            f"load_drag_env supports 1D / 2D / 4D wind; got {wind_kind!r}.")
    _check_pairing(density_kind, wind_kind, "load_drag_env")

    beta_alt = data["beta_altitude_km"].tolist()
    beta_vals = data["beta_kg_m2"].tolist()

    if density_kind == "1d":
        # 1D density: rebuild with the 1D factory.
        density_alt = data["density_altitude_km"].tolist()
        density_rho = data["density_kg_m3"].tolist()
        if wind_kind == "1d_enu":
            wind_alt = data["wind_altitude_km"].tolist()
            wind_mer = data["wind_meridional_ms"].tolist()
            wind_zon = data["wind_zonal_ms"].tolist()
        else:
            wind_alt, wind_mer, wind_zon = [], [], []
        return _core.DragEnvironment._from_1d_components(
            density_alt, density_rho,
            wind_alt, wind_mer, wind_zon,
            beta_alt, beta_vals,
        )

    if density_kind == "2d":
        # 2D density (altitude x time).
        density_alt = data["density_altitude_km"].tolist()
        density_epochs_mjd = data["density_epochs_mjd_utc"].tolist()
        density_grid = data["density_grid"].tolist()

        if wind_kind == "2d_ecef":
            wind_alt = data["wind_altitude_km"].tolist()
            wind_epochs_mjd = data["wind_epochs_mjd_utc"].tolist()
            wind_x = data["wind_x_ecef"].tolist()
            wind_y = data["wind_y_ecef"].tolist()
            wind_z = data["wind_z_ecef"].tolist()
            wind_mer, wind_zon = [], []
        elif wind_kind == "1d_enu":
            wind_alt = data["wind_altitude_km"].tolist()
            wind_mer = data["wind_meridional_ms"].tolist()
            wind_zon = data["wind_zonal_ms"].tolist()
            wind_epochs_mjd = []
            wind_x, wind_y, wind_z = [], [], []
        else:  # "1d_enu_empty"
            wind_alt, wind_mer, wind_zon = [], [], []
            wind_epochs_mjd = []
            wind_x, wind_y, wind_z = [], [], []

        return _core.DragEnvironment._from_2d_components(
            density_alt, density_epochs_mjd, density_grid,
            wind_kind,
            wind_alt, wind_epochs_mjd, wind_x, wind_y, wind_z,
            wind_mer, wind_zon,
            beta_alt, beta_vals,
        )

    # 4D density (altitude x latitude x longitude x time).
    d_alt = data["density_altitude_km"].tolist()
    d_lat = data["density_latitude_deg"].tolist()
    d_lon = data["density_longitude_deg"].tolist()
    d_epochs_mjd = data["density_epochs_mjd_utc"].tolist()
    d_grid = data["density_grid"].tolist()

    if wind_kind == "4d_ecef":
        w_alt = data["wind_altitude_km"].tolist()
        w_lat = data["wind_latitude_deg"].tolist()
        w_lon = data["wind_longitude_deg"].tolist()
        w_epochs_mjd = data["wind_epochs_mjd_utc"].tolist()
        w_x = data["wind_x_ecef"].tolist()
        w_y = data["wind_y_ecef"].tolist()
        w_z = data["wind_z_ecef"].tolist()
    else:
        # "1d_enu_empty": the only other wind _check_pairing lets through.
        w_alt, w_lat, w_lon = [], [], []
        w_epochs_mjd = []
        w_x, w_y, w_z = [], [], []

    return _core.DragEnvironment._from_4d_components(
        d_alt, d_lat, d_lon, d_epochs_mjd, d_grid,
        wind_kind,
        w_alt, w_lat, w_lon, w_epochs_mjd, w_x, w_y, w_z,
        beta_alt, beta_vals,
    )
