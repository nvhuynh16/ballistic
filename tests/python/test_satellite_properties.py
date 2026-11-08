"""T1: satellite-properties single source stays consistent.

Guards the frontal-vs-total-area failure class: β must follow from
mass/(Cd·frontal area), every value must carry provenance, CHAMP's β must
stay in the literature band (eoPortal frontal area 0.9 m² and a blind drag
sweep optimum ≈265), and no validate script may regrow a local properties
dict (silent-fallback duplication lets a 4.6× β error survive).
"""

import re
import sys

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from ballistic.satellite_properties import PROPERTIES, get  # noqa: E402
import ballistic.satellite_properties as _sp_module  # noqa: E402


def test_beta_follows_from_parts():
    for name, p in PROPERTIES.items():
        assert abs(p.beta_kg_m2 - p.mass_kg / (p.cd * p.drag_area_m2)) < 1e-12


def test_every_entry_has_provenance():
    for name, p in PROPERTIES.items():
        assert p.provenance and len(p.provenance) > 20, name
        assert p.mass_kg > 0 and p.drag_area_m2 > 0 and p.cd > 0, name


def test_champ_beta_in_literature_band():
    # eoPortal frontal 0.9 m^2; Cd*A fitted out of sample under the default
    # space weather: beta 197.9, best within x0.7-0.8 of 263.6 (nominal Cd)
    # on three CHAMP data sets. 66.2 (the total-area error) must never
    # return.
    beta = get("CHAMP").beta_kg_m2
    assert 170.0 <= beta <= 230.0, beta
    assert get("CHAMP").drag_area_m2 < 1.5, "frontal, not total area"


def test_sentinel1_uses_official_cpod_area():
    # GMV-GMESPOD-TN-0025 Table 5-1: orbit-averaged frontal 6.35 m^2.
    assert abs(get("S1A").drag_area_m2 - 6.35) < 1e-9
    assert "GMESPOD" in get("S1A").provenance


def test_no_local_property_dicts_in_scripts():
    # Structural: the validate scripts must consume the single source.
    # A regrown literal dict with mass/drag_area keys = the failure mode.
    pat = re.compile(r"'mass':\s*[\d.]+,\s*'drag_area':\s*[\d.]+")
    for script in sorted((REPO / "scripts").glob("validate_*.py")):
        text = script.read_text(encoding="utf-8")
        assert not pat.search(text), (
            f"{script.name} regrew a local properties dict — use "
            "ballistic.satellite_properties")
        if "PROPERTIES" in text and "drag" in script.name:
            assert "satellite_properties" in text, script.name


def test_single_source_ships_in_the_package():
    """L10: library users must get the cited β, not just the scripts."""
    assert _sp_module.__name__ == "ballistic.satellite_properties"
    assert "ballistic" in str(Path(_sp_module.__file__).parts[-2:])


def test_unknown_satellite_raises():
    import pytest
    with pytest.raises(KeyError):
        get("NOT-A-SATELLITE")


def _doc_surface_files():
    """Every file a user reads to learn how to configure drag: prose docs,
    the public C++ headers (usage comments carry examples too), the LaTeX
    tutorial, and the runnable examples. Test files
    are deliberately EXCLUDED: they use arbitrary β fixtures (100.0, 66.2,
    45.45), not satellite properties."""
    files = {
        REPO / "README.md",
        REPO / "CLAUDE.md",  # guard-exempt(PUB-AI-REF): development-clone file; guard-exempt(PUB-PRIVATE-PATH): same (absent files are dropped below)
        REPO / "python" / "ballistic" / "__init__.py",
        REPO / "python" / "ballistic" / "trajectory.py",
    }
    files.update((REPO / "docs").rglob("*.md"))
    files.update((REPO / "docs" / "latex").rglob("*.tex"))
    files.update((REPO / "include").rglob("*.hpp"))
    for base in (REPO / "examples", REPO / "docs" / "latex" / "examples"):
        for ext in ("*.cpp", "*.hpp", "*.py"):
            files.update(base.rglob(ext))
    return sorted(f for f in files if f.exists())


DOC_FILES = _doc_surface_files()


def test_doc_examples_teach_real_properties():
    """Stale examples are where stale values breed (an outdated Sentinel-1
    area once survived in eight copies across docs, docstrings and a public
    header). Every from_mass_cd_area(m, cd, A) triple and every literal
    BetaProfile1D constant across the doc/header/example surface must match
    a cited entry in the single source."""
    triples = re.compile(r"from_mass_cd_area\((\d+(?:\.\d+)?),\s*"
                         r"(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")
    constants = re.compile(r"BetaProfile1D(?:::|\.)constant\((\d+(?:\.\d+)?)")
    known = {(p.mass_kg, p.cd, p.drag_area_m2) for p in PROPERTIES.values()}
    known_betas = {round(p.beta_kg_m2, 1) for p in PROPERTIES.values()}
    for f in DOC_FILES:
        rel = f.relative_to(REPO)
        text = f.read_text(encoding="utf-8")
        for m in triples.finditer(text):
            triple = tuple(float(x) for x in m.groups())
            assert triple in known, (
                f"{rel}: from_mass_cd_area{triple} matches no entry in "
                "ballistic.satellite_properties — stale example?")
        for m in constants.finditer(text):
            beta = round(float(m.group(1)), 1)
            assert beta in known_betas, (
                f"{rel}: BetaProfile1D constant {beta} matches no "
                "single-source beta — stale example?")


def test_fitted_srp_areas_match_their_record():
    """srp_area_m2 of every fitted Sentinel = base area x fitted k
    (SENTINEL-YEAR-2025 srp-fit). Guards a hand-edit of one side only."""
    from ballistic.satellite_properties import SRP_FIT
    for name, (base, k) in SRP_FIT.items():
        assert abs(get(name).srp_area_m2 - round(base * k, 1)) < 1e-9, name
        assert get(name).srp_kind == "fitted", name


def test_fit_kinds_are_valid():
    """drag_kind / srp_kind take only the four documented values."""
    for p in PROPERTIES.values():
        assert p.drag_kind in _sp_module.FIT_KINDS, (p.name, p.drag_kind)
        assert p.srp_kind in _sp_module.FIT_KINDS, (p.name, p.srp_kind)


def test_fit_kind_matches_provenance():
    """The machine-readable kinds agree with the cited record: a provenance
    that says a value was fitted (not "not fitted") has a fitted kind, every SRP-fit satellite
    is srp_kind "fitted", GRACE-FO's drag is the physical panel model, and
    CHAMP's and GOCE's drag are beta fits written as Cd."""
    from ballistic.satellite_properties import SRP_FIT
    says_fitted = re.compile(r"(?<!not )\bfitted\b", re.IGNORECASE)
    for p in PROPERTIES.values():
        if says_fitted.search(p.provenance):
            assert "fitted" in (p.drag_kind, p.srp_kind), p.name
    for name in SRP_FIT:
        assert get(name).srp_kind == "fitted", name
    assert get("GRACE-FO-1").drag_kind == "physical"
    assert get("GRACE-FO-2").drag_kind == "physical"
    assert get("CHAMP").drag_kind == "fitted"
    assert get("GOCE").drag_kind == "fitted"


@pytest.mark.nrl
def test_recommended_config_hints_on_legacy_values(leo_state):
    """A satellite with unsourced (legacy) values earns a ConfigHintWarning
    naming the arguments that replace them; a well-sourced one does not."""
    import warnings
    import ballistic
    from ballistic.errors import ConfigHintWarning
    epoch, pos, vel = leo_state
    state = ballistic.State.from_itrf(epoch, pos, vel)
    assert get("SWARM-A").drag_kind == "legacy"
    with pytest.warns(ConfigHintWarning, match="drag_area_m2=") as rec:
        ballistic.recommended_config(state, "SWARM-A")
    assert any("srp_area_m2=" in str(w.message) for w in rec)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ballistic.recommended_config(state, "S1A")
    assert not [w for w in rec if issubclass(w.category, ConfigHintWarning)
                and "unsourced" in str(w.message)]


def test_doc_srp_examples_teach_current_areas():
    """Fitted SRP areas replace nominal ones, so a `mass 2200 / area 20`
    Sentinel-1 example is stale. An SrpConfig example whose mass is a known

    satellite's must use that satellite's current radiation area (generic
    examples with other masses are fine)."""
    cpp = re.compile(r"SrpConfig\{\s*\.mass_kg\s*=\s*(\d+(?:\.\d+)?),\s*"
                     r"\.area_m2\s*=\s*(\d+(?:\.\d+)?)")
    py = re.compile(r"SrpConfig\(\s*mass_kg\s*=\s*(\d+(?:\.\d+)?),\s*"
                    r"area_m2\s*=\s*(\d+(?:\.\d+)?)")
    areas = {}
    for p in PROPERTIES.values():
        areas.setdefault(p.mass_kg, set()).add(p.srp_area_m2)
    for f in DOC_FILES:
        rel = f.relative_to(REPO)
        text = f.read_text(encoding="utf-8")
        for rx in (cpp, py):
            for m in rx.finditer(text):
                mass, area = float(m.group(1)), float(m.group(2))
                if mass in areas:
                    assert area in areas[mass], (
                        f"{rel}: SrpConfig(mass {mass}, area {area}) — "
                        f"satellites of that mass use {sorted(areas[mass])}; "
                        "stale example?")
