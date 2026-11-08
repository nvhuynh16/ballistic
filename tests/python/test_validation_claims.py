"""The claims manifest cannot rot silently.

`docs/validation_claims.md` records every validation number with its value,
date, commit and reproduction command. A hand-maintained manifest goes stale
unless something checks it:

  * every claim ID cited in a validation table (README, and the development
    notes when present) has a manifest row;
  * every manifest row has non-empty value/date/commit/reproduction cells;
  * every script and data path named in a reproduction command still
    EXISTS (a renamed script silently orphaning a claim is the realistic
    rot mode, and it is invisible to a human skim); data that is not
    committed must have a README saying how to reacquire it;
  * the commit column, when it looks like a hash, names a real object
    (development clone only: the public history has different commits);
  * every row says how independent its number is and whether it is stale.

Includes a self-test: a guard that cannot fail is not a guard.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "docs" / "validation_claims.md"
# The development notes (absent from the public tree) carry a copy of the table.
_DEV_NOTES = REPO / "CLAUDE.md"  # guard-exempt(PUB-AI-REF) guard-exempt(PUB-PRIVATE-PATH): optional
TABLE_FILES = [f for f in (_DEV_NOTES, REPO / "README.md") if f.exists()]

ID_IN_TABLE = re.compile(r"\[`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)`\]")
ID_IN_MANIFEST = re.compile(r"^\|\s*`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)`\s*\|", re.MULTILINE)
# Paths that look like repo files inside a reproduction command.
PATH_TOKEN = re.compile(r"(?:scripts|tests|build|docs)/[\w./{}*,-]+")


def manifest_rows():
    """[(id, [cells...])] for every ID-bearing row of the manifest table."""
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        m = ID_IN_MANIFEST.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((m.group(1), cells))
    return rows


def test_manifest_parses_and_has_rows():
    rows = manifest_rows()
    assert len(rows) >= 6, f"manifest looks empty/unparseable: {rows}"


def test_table_ids_and_manifest_ids_agree():
    table_ids = set()
    for f in TABLE_FILES:
        table_ids |= set(ID_IN_TABLE.findall(f.read_text(encoding="utf-8")))
    manifest_ids = {rid for rid, _ in manifest_rows()}
    orphaned = table_ids - manifest_ids
    assert not orphaned, (
        f"validation-table rows cite IDs with no manifest row: {orphaned}")
    # The reverse is allowed (a manifest row may record a claim not shown in
    # a summary table, e.g. the out-of-sample CHAMP day), but the IDs that
    # ARE cited must resolve.


UNREPRODUCIBLE = "UNREPRODUCIBLE-IN-REPO"


def test_every_row_has_all_cells():
    for rid, cells in manifest_rows():
        # id | claim | value | measured | commit | reproduction | gate
        assert len(cells) >= 6, (rid, cells)
        if UNREPRODUCIBLE in cells[2]:
            # A row may declare that it CANNOT be reproduced here (no
            # harness / no data). That is an explicit, auditable state —
            # unlike a silent blank, which is the rot mode this guards. It
            # must say so in the value cell and say why.
            assert len(cells[2]) > len(UNREPRODUCIBLE) + 5, (
                f"{rid}: declare WHY it is unreproducible")
            continue
        for i, name in enumerate(
                ["claim", "value", "measured", "commit", "reproduction"], 1):
            assert cells[i] and cells[i] != "—", f"{rid}: empty {name} cell"


def test_unreproducible_rows_are_declared_not_blank():
    """Rows that cannot be reproduced must SAY so — the failure mode is a
    row that quietly has nothing in it and reads as verified."""
    for rid, cells in manifest_rows():
        blanks = sum(1 for c in cells[1:6] if not c or c == "—")
        if blanks >= 3:
            assert UNREPRODUCIBLE in cells[2], (
                f"{rid}: mostly-empty row must declare {UNREPRODUCIBLE}")


def _path_resolves(tok, repo=REPO):
    """Does a path named in a reproduction command resolve?

    * ``build/tests/<name>`` is a build output: the test source
      ``tests/*/<name>.cpp`` must exist.
    * ``tests/data/...`` may be data that is not committed (too large, or
      behind a login): the path, or a README.md in its nearest existing
      ancestor under tests/data/ (the documented reacquisition), must exist.
    * anything else must exist.
    """
    if tok.startswith("build/tests/"):
        name = tok[len("build/tests/"):].split("/")[0]
        return any((repo / "tests").glob(f"*/{name}.cpp"))
    if (repo / tok).exists():
        return True
    if tok.startswith("tests/data/"):
        root = repo / "tests" / "data"
        p = (repo / tok).parent
        while p != root and root in p.parents and not p.exists():
            p = p.parent
        return (p == root or root in p.parents) and (p / "README.md").exists()
    return False


def test_reproduction_paths_exist():
    """The rot mode a human skim misses: a renamed script or moved data
    file leaves the command in place and silently unrunnable."""
    missing = []
    for rid, cells in manifest_rows():
        if UNREPRODUCIBLE in cells[2]:
            continue
        for tok in PATH_TOKEN.findall(cells[5]):
            tok = tok.rstrip("`.,;")
            if any(ch in tok for ch in "{}*"):
                continue          # shell brace/glob expansion — skip
            if not _path_resolves(tok):
                missing.append((rid, tok))
    assert not missing, f"reproduction commands name missing paths: {missing}"


def test_path_resolution_rules(tmp_path):
    """Self-test of the three path rules."""
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_x.cpp").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "data" / "sp3" / "champ").mkdir(parents=True)
    (tmp_path / "tests" / "data" / "sp3" / "champ" / "README.md").write_text("reacquire", encoding="utf-8")
    (tmp_path / "tests" / "data" / "sp3" / "README.md").write_text("reacquire: ...", encoding="utf-8")
    (tmp_path / "tests" / "data" / "sp3" / "goce").mkdir()
    (tmp_path / "tests" / "data" / "eof").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "a.py").write_text("", encoding="utf-8")
    assert _path_resolves("build/tests/test_x", tmp_path)
    assert not _path_resolves("build/tests/test_gone", tmp_path)
    assert _path_resolves("tests/data/sp3/champ/day182.sp3.gz", tmp_path)  # README beside
    assert _path_resolves("tests/data/sp3/missing_dir/f.sp3", tmp_path)    # nearest: sp3/
    assert not _path_resolves("tests/data/sp3/goce/g.sp3", tmp_path)      # goce/ has none
    assert not _path_resolves("tests/data/eof/S1A.EOF", tmp_path)         # no README
    assert _path_resolves("scripts/a.py", tmp_path)
    assert not _path_resolves("scripts/b.py", tmp_path)


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, timeout=30, check=False)


def _is_development_clone():
    """The commit column names development-history commits. They exist in
    the development clone (it has a `local` branch, or a long history), not
    in the published history or a tarball."""
    try:
        if _git("rev-parse", "-q", "--verify", "refs/heads/local").returncode == 0:
            return True
        r = _git("rev-list", "--count", "HEAD")
        return r.returncode == 0 and int(r.stdout.strip() or 0) > 100
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def test_commit_column_names_real_objects():
    if not _is_development_clone():
        pytest.skip("commit column names development-history commits; "
                    "checkable only in the development clone")
    hashes = [(rid, cells[4]) for rid, cells in manifest_rows()
              if re.fullmatch(r"[0-9a-f]{7,40}", cells[4])]
    assert hashes, "no hash-style commit cells: the Commit column looks malformed"
    for rid, h in hashes:
        r = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", h],
                           capture_output=True, timeout=30, check=False)
        assert r.returncode == 0, f"{rid}: commit {h} is not a real object"


def test_guard_detects_missing_row(tmp_path, monkeypatch):
    """Self-test: feed the parser a manifest missing a cited ID and confirm
    the agreement check would fail. A guard that can't fail is not a guard."""
    fake_manifest = tmp_path / "claims.md"
    fake_manifest.write_text(
        "| ID | Claim | Value | Measured | Commit | Reproduction | Gate |\n"
        "|---|---|---|---|---|---|---|\n"
        "| `CHAMP-370KM` | c | v | d | abc1234 | `scripts/x.py` | g |\n", encoding="utf-8")  # guard-exempt(PUB-COMMIT-HASH): synthetic
    fake_table = tmp_path / "TABLE.md"
    fake_table.write_text("| LEO | CHAMP [`CHAMP-370KM`] | 1.5 m |\n"
                          "| LEO | GHOST [`K9-GHOST`] | 9 m |\n", encoding="utf-8")
    monkeypatch.setattr("test_validation_claims.MANIFEST", fake_manifest)
    monkeypatch.setattr("test_validation_claims.TABLE_FILES", [fake_table])
    with pytest.raises(AssertionError, match="K9-GHOST"):
        test_table_ids_and_manifest_ids_agree()


# --- Sample/status columns and reproduction flags ----------------------------

SAMPLE_VOCAB = re.compile(r"^(in-sample|out-of-sample|n/a)\b|^mixed: \S")
STATUS_VOCAB = re.compile(r"^(current|STALE\([^)]+\))$")


def test_sample_and_status_vocabulary():
    """Every row says how independent its number is (Sample) and whether a
    later change has invalidated it (Status: `current` or `STALE(<reason>)`)."""
    for rid, cells in manifest_rows():
        assert len(cells) >= 9, f"{rid}: missing Sample/Status columns"
        assert SAMPLE_VOCAB.match(cells[7]), f"{rid}: bad Sample '{cells[7]}'"
        assert STATUS_VOCAB.match(cells[8]), f"{rid}: bad Status '{cells[8]}'"


def test_no_stale_rows_at_release():
    """At a release (BALLISTIC_RELEASE_GATE=1) every claim must be re-measured."""
    import os
    stale_allowed = not os.environ.get("BALLISTIC_RELEASE_GATE")
    if stale_allowed:
        return   # between releases a change may leave rows stale
    stale = [rid for rid, cells in manifest_rows() if cells[8].startswith("STALE")]
    assert not stale, f"stale claims must be re-measured before release: {stale}"


def _parser_flags(script, _seen=None):
    """Every --flag a script's argparse defines, including flags added by the
    local helper modules it imports (static: no script dependencies needed)."""
    import ast
    _seen = _seen or set()
    if script in _seen or not script.exists():
        return set()
    _seen.add(script)
    tree = ast.parse(script.read_text(encoding="utf-8"))
    flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                    flags.add(a.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""])
            for name in names:
                flags |= _parser_flags(script.parent / f"{name.split('.')[0]}.py", _seen)
    return flags


SCRIPT_ARGS = re.compile(r"(scripts/[\w-]+\.py)([^`;&|]*)")
FLAG = re.compile(r"(?<![\w-])(--[a-z][\w-]*)")


def test_reproduction_flags_exist():
    """A reproduction command must not name a flag its script lost."""
    bad = []
    for rid, cells in manifest_rows():
        for script, args in SCRIPT_ARGS.findall(cells[5]):
            defined = _parser_flags(REPO / script)
            for flag in FLAG.findall(args):
                if flag in defined or ("--" + flag[5:] in defined and flag.startswith("--no-")):
                    continue
                bad.append((rid, script, flag))
    assert not bad, f"reproduction commands use flags their scripts lack: {bad}"


def test_flag_guard_detects_a_lost_flag():
    """Self-test: a flag the script does not define is caught."""
    defined = _parser_flags(REPO / "scripts" / "validate_gracefo_drag.py")
    assert "--sp3" in defined and "--drag-weather" in defined and "--sp3-velocity" in defined
    assert "--no-such-flag" not in defined


# --- No uncited number in a validation table ---------------------------------

# The tables that state measured accuracy. Every data row of each must cite a
# claim ID: a number without one is unprovenanced and rots.
VALIDATION_TABLES = [(f, h) for f, h in ((REPO / "README.md", "## Accuracy"),
                                         (_DEV_NOTES, "## Validation Summary"))
                     if f.exists()]
_NUMBER = re.compile(r"\d+(\.\d+)?\s*(m\b|%)")


def _uncited_rows(text, heading):
    """Data rows of the first table under `heading` that state a number
    (meters or percent) but cite no [`ID`]."""
    start = text.index(heading)
    lines = text[start:].splitlines()[1:]
    rows, in_table = [], False
    for line in lines:
        if line.startswith("|"):
            in_table = True
            cells = line.strip().strip("|").split("|")
            if all(set(c.strip()) <= set("-: ") for c in cells):
                continue   # the separator row
            rows.append(line)
        elif in_table or line.startswith("#"):
            break
    return [r for r in rows[1:] if _NUMBER.search(r) and not ID_IN_TABLE.search(r)]


def test_validation_table_numbers_cite_a_claim():
    for path, heading in VALIDATION_TABLES:
        bad = _uncited_rows(path.read_text(encoding="utf-8"), heading)
        assert not bad, f"{path.name} '{heading}': rows without a claim ID:\n" + "\n".join(bad)


def test_uncited_number_guard_detects_a_row():
    text = ("## Accuracy\n\n| Regime | RMSE |\n|---|---|\n"
            "| LEO [`SENTINEL-JAN26-SAMPLED`] | 0.2 m |\n| MEO | 0.7 m |\n\nafter\n")
    assert _uncited_rows(text, "## Accuracy") == ["| MEO | 0.7 m |"]
