"""Removed and moved APIs raise RemovedApiError naming a replacement that exists.

The hardening program removes names that let users fall into traps
(ballistic/_removed.py). A removal must never leave the user with a bare
AttributeError: the error names the replacement, and the replacement must
exist. Names that moved from the top level into a module (MOVED) raise
the same error naming the new import. docs/MIGRATION.md lists every entry.
"""

from pathlib import Path

import pytest

import ballistic
from ballistic import _removed

REPO = Path(__file__).resolve().parents[2]


def _resolve(dotted):
    obj = ballistic
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def test_removed_names_raise_and_name_an_existing_replacement():
    # One test over the whole table (an empty parametrization would count
    # as a skip, and CI's skip census treats skips as lost coverage).
    for name, (replacement, _why) in _removed.REMOVED.items():
        with pytest.raises(ballistic.RemovedApiError, match=replacement):
            getattr(ballistic, name)
        _resolve(replacement)   # raises if the replacement does not exist


def test_removed_names_are_in_the_migration_guide():
    guide = (REPO / "docs" / "MIGRATION.md").read_text(encoding="utf-8")
    missing = [n for n in _removed.REMOVED if f"`{n}`" not in guide]
    assert not missing, f"removed but not in docs/MIGRATION.md: {missing}"


def test_moved_names_raise_and_name_their_new_home():
    """A name that moved into a module raises RemovedApiError naming the new
    import, and the new home exists."""
    for name, home in _removed.MOVED.items():
        assert name not in ballistic.__all__, name
        module, attr = home.rsplit(".", 1)
        with pytest.raises(ballistic.RemovedApiError) as err:
            getattr(ballistic, name)
        msg = str(err.value)
        assert f"ballistic.{home}" in msg, msg
        assert f"from ballistic.{module} import {attr}" in msg, msg
        _resolve(home)
        assert attr in getattr(ballistic, module).__all__, home


def test_moved_names_are_in_the_migration_guide():
    guide = (REPO / "docs" / "MIGRATION.md").read_text(encoding="utf-8")
    missing = [n for n, home in _removed.MOVED.items()
               if f"`ballistic.{n}` | `ballistic.{home}`" not in guide]
    assert not missing, f"moved but not in docs/MIGRATION.md: {missing}"


def test_moved_and_removed_do_not_overlap():
    assert not set(_removed.MOVED) & set(_removed.REMOVED)


def test_mechanism(monkeypatch):
    """Self-test with a temporary entry, so the mechanism is exercised even
    while the table is empty."""
    monkeypatch.setitem(_removed.REMOVED, "old_name", ("Epoch", "it was ambiguous."))
    with pytest.raises(ballistic.RemovedApiError, match="Use ballistic.Epoch instead"):
        ballistic.old_name   # noqa: B018
    with pytest.raises(AttributeError):
        ballistic.never_existed   # noqa: B018
    monkeypatch.setitem(_removed.MOVED, "old_home", "drag.DragEnvironment")
    with pytest.raises(ballistic.RemovedApiError,
                       match="from ballistic.drag import DragEnvironment"):
        ballistic.old_home   # noqa: B018
