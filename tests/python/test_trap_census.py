"""docs/trap_census.md stays honest.

A closed trap must name a guard that exists; an open trap must name the
planned fix; every static guard rule must be listed.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CENSUS = REPO / "docs" / "trap_census.md"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Sibling test module, importable only after the path insert above.
from test_static_guards import RULES  # noqa: E402

ROW = re.compile(r"^\| ([A-Z0-9-]+) \| ([^|]+) \| ([^|]+) \| (open|closed) \| ([^|]+) \|$")


def rows():
    out = []
    for line in CENSUS.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line.strip())
        if m:
            out.append(tuple(c.strip() for c in m.groups()))
    return out


def test_census_parses():
    r = rows()
    assert len(r) >= 20, "census table looks empty or malformed"
    ids = [x[0] for x in r]
    assert len(ids) == len(set(ids)), "duplicate trap IDs"


def _guard_exists(guard):
    if guard.startswith("static:"):
        return guard[len("static:"):] in {rule.id for rule in RULES}
    path, _, name = guard.partition("::")
    p = REPO / path
    if not p.exists():
        return False
    return not name or name in p.read_text(encoding="utf-8")


def test_closed_traps_have_a_guard_that_exists():
    bad = [(tid, g) for tid, _, _, status, g in rows()
           if status == "closed" and not _guard_exists(g)]
    assert not bad, f"closed traps whose guard does not exist: {bad}"


def _names_a_fix(text):
    return bool(text.strip().strip("-—–").strip())


def test_open_traps_name_the_planned_fix():
    bad = [(tid, g) for tid, _, _, status, g in rows()
           if status == "open" and not _names_a_fix(g)]
    assert not bad, f"open rows must name the planned fix: {bad}"


def test_planned_fix_check_rejects_a_dash():
    assert not _names_a_fix(" — ") and not _names_a_fix("-")
    assert _names_a_fix("refuse inertial input labeled ECEF")


def test_every_static_rule_is_in_the_census():
    guards = {g for *_, g in rows()}
    missing = [r.id for r in RULES if f"static:{r.id}" not in guards]
    assert not missing, f"static rules not listed in the census: {missing}"
