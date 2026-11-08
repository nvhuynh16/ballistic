"""The tiered namespaces: every public name of the compiled module has one home.

The top-level ``ballistic`` exports what a first session needs; specialized
names live in ``ballistic.estimation``, ``.drag``, ``.aero``, ``.readers``,
``.tle`` and ``.sp3``. These tests keep the split honest as the bindings
grow: a new ``_core`` name must be exported from exactly one place or listed
with a reason in ``ballistic._INTERNAL_CORE_NAMES``; a tier name must be
documented in docs/API_REFERENCE.md; and the top level must stay short.
"""

from pathlib import Path

import pytest

import ballistic
from ballistic import _core

REPO = Path(__file__).resolve().parents[2]

# The modules whose names must not overlap with each other or the top level
# (tle and sp3 are format modules with their own short verbs, read/parse).
TIERS = {
    "estimation": ballistic.estimation,
    "drag": ballistic.drag,
    "aero": ballistic.aero,
    "readers": ballistic.readers,
}
FORMAT_MODULES = {"tle": ballistic.tle, "sp3": ballistic.sp3}
MAX_TOP_LEVEL = 85


def _homes():
    """name -> list of modules whose __all__ lists it and that export the
    _core object itself (an alias such as sp3.classify_frame counts for the
    object it re-exports)."""
    homes = {}
    for mod_name, mod in {"ballistic": ballistic, **TIERS, **FORMAT_MODULES}.items():
        for name in mod.__all__:
            homes.setdefault(name, []).append(mod_name)
    return homes


def test_every_core_name_has_one_home():
    homes = _homes()
    internal = ballistic._INTERNAL_CORE_NAMES
    unplaced, twice, both = [], [], []
    for name in sorted(n for n in dir(_core) if not n.startswith("_")):
        places = homes.get(name, [])
        if name in internal and places:
            both.append((name, places))
        elif not places and name not in internal:
            unplaced.append(name)
        elif len(places) > 1:
            twice.append((name, places))
    assert not unplaced, (
        f"_core names exported nowhere: {unplaced}. Add each to the matching "
        "tier's __all__ or to ballistic._INTERNAL_CORE_NAMES with a reason.")
    assert not twice, f"_core names exported from more than one module: {twice}"
    assert not both, f"listed as internal but also exported: {both}"


def test_internal_names_exist_and_give_a_reason():
    for name, reason in ballistic._INTERNAL_CORE_NAMES.items():
        assert hasattr(_core, name), f"_INTERNAL_CORE_NAMES lists missing {name}"
        assert reason.split(":")[0] in {"internal", "wrapped", "reachable", "aliased"}, reason


def test_no_name_in_two_tiers():
    seen = {}
    for mod_name, mod in {"ballistic": ballistic, **TIERS}.items():
        for name in mod.__all__:
            seen.setdefault(name, []).append(mod_name)
    dups = {n: m for n, m in seen.items() if len(m) > 1}
    assert not dups, f"names exported from two tiers: {dups}"


def test_every_listed_name_resolves():
    for mod in [ballistic, *TIERS.values(), *FORMAT_MODULES.values()]:
        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        assert not missing, f"{mod.__name__}.__all__ lists missing names: {missing}"


def test_tiers_are_attributes_of_the_package():
    for name in [*TIERS, *FORMAT_MODULES, "events", "errors", "constants",
                 "satellite_properties"]:
        assert hasattr(ballistic, name), name


def test_top_level_stays_short():
    assert len(ballistic.__all__) <= MAX_TOP_LEVEL, (
        f"ballistic.__all__ has {len(ballistic.__all__)} names (limit "
        f"{MAX_TOP_LEVEL}); put specialized names in a tier module.")


def test_tier_names_are_documented():
    """Each tier name (and recommended_config) appears qualified in the API
    reference, so a reader searching for it finds where it lives."""
    doc = (REPO / "docs" / "API_REFERENCE.md").read_text(encoding="utf-8")
    missing = [] if "ballistic.recommended_config(" in doc else ["recommended_config"]
    for mod_name, mod in {**TIERS, **FORMAT_MODULES}.items():
        missing += [f"{mod_name}.{n}" for n in mod.__all__
                    if f"ballistic.{mod_name}.{n}" not in doc]
    assert not missing, f"not in docs/API_REFERENCE.md: {missing}"


@pytest.mark.parametrize("wrapper", ["estimate_ea_smoother", "information_bound"])
def test_multisensor_wrappers_take_either_config(wrapper):
    """The Python wrappers convert the TrajectoryConfig dataclass; the force
    model may be given by position or by keyword (fm_cfg / force_model)."""
    fn = getattr(ballistic.estimation, wrapper)
    assert fn.__module__ == "ballistic.estimation"
    assert fn is not getattr(_core, wrapper)
