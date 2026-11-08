"""Building and storing atmosphere descriptions for drag, by hand.

Most users turn drag on with ``ballistic.recommended_config`` or
``ballistic.make_drag_environment``, which live in the top-level package
together with the ballistic coefficient (``ballistic.BetaProfile1D``) and the
space-weather history they take. This module holds what those calls hide: the
``DragEnvironment`` type and its sample, tabulated atmospheres (a snapshot
profile, a table along a trajectory, a full four-dimensional field), measured
accelerometer densities, the U.S. Standard Atmosphere below a chosen
altitude, the readers for density, ballistic-coefficient and
empirical-acceleration tables, and saving an environment to a file and back.

    from ballistic.drag import from_accelerometer_density, parse_density_table
    mjd_utc, rho = parse_density_table("GO_DNS_ACC_2013_10.dat")
    # turn each row's time into a ballistic.Epoch, then:
    env = from_accelerometer_density(epochs, rho, beta)
"""

from ._core import (
    DragEnvironment,
    DragSample,
    TimeWindow,
    IterationDiag,
    WeatherConditions,
    BallisticModel,
    standard_leo_altitude_grid_km,
    from_models_snapshot,
    from_models_along,
    from_models_4d,
    from_accelerometer_density,
    extend_density_below,
)
from .loaders import load_beta_table, load_empirical_acceleration, parse_density_table
from .dragenv_io import save_drag_env, load_drag_env

__all__ = [
    "DragEnvironment",
    "DragSample",
    "TimeWindow",
    "IterationDiag",
    "WeatherConditions",
    "BallisticModel",
    "standard_leo_altitude_grid_km",
    "from_models_snapshot",
    "from_models_along",
    "from_models_4d",
    "from_accelerometer_density",
    "extend_density_below",
    "parse_density_table",
    "load_beta_table",
    "load_empirical_acceleration",
    "save_drag_env",
    "load_drag_env",
]
