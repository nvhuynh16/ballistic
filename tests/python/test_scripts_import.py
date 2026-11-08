"""Every import in scripts/**/*.py must resolve (trap census: DEAD-IMPORT).

Two scripts once imported a module that never existed (validate_sentinel_2025)
and nobody noticed, because scripts are not run in CI. Importing each script
for real would pull in pandas, h5py, matplotlib and cdflib, so this check is
static: every module a script imports must be either a local module in
scripts/ or a subdirectory of it, the ballistic package, the standard library,
or a known third-party dependency, and every name imported from a local module
must be defined there.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"

# Third-party packages the scripts are allowed to need (installed per script
# with `uv run --with ...`; see each script's docstring).
THIRD_PARTY = {"numpy", "scipy", "pandas", "h5py", "matplotlib", "seaborn", "tqdm",
               "requests", "cdflib", "sgp4"}


def _scripts():
    """Every .py under scripts/, subdirectories included."""
    return sorted(SCRIPTS.rglob("*.py"))


def _local_modules():
    return {p.stem: p for p in _scripts()}


def _top_level_names(path):
    """Names a module defines at top level (functions, classes, assignments,
    imports, and names assigned inside top-level if/try blocks)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, (ast.If, ast.Try)):
                visit(node.body)
                visit(getattr(node, "orelse", []))
                for h in getattr(node, "handlers", []):
                    visit(h.body)
                visit(getattr(node, "finalbody", []))
    visit(tree.body)
    return names


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name, None
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module, [a.name for a in node.names]


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.relative_to(SCRIPTS).as_posix())
def test_script_imports_resolve(script):
    local = _local_modules()
    stdlib = set(sys.stdlib_module_names)
    problems = []
    for lineno, module, names in _imports(script):
        top = module.split(".")[0]
        if top in stdlib or top in THIRD_PARTY or top == "ballistic":
            continue
        if top not in local:
            problems.append(f"line {lineno}: imports unknown module '{module}'")
            continue
        if names:
            defined = _top_level_names(local[top])
            missing = [n for n in names if n != "*" and n not in defined]
            if missing:
                problems.append(f"line {lineno}: '{module}' does not define {missing}")
    assert not problems, f"{script.name}:\n  " + "\n  ".join(problems)


# --- Method names on library objects -----------------------------------------
#
# An API change renamed Trajectory.propagate to propagate_for/propagate_to in
# the Python wrapper; a mechanical migration then called propagate_to on a raw
# _core.Trajectory, which has no such method, and the year campaign failed at
# its first start. The scripts are not run in CI, so this is checked
# statically: for `x = <ballistic or _core>.<Class>(...)` inside a function,
# every `x.<name>` used in that function must exist on that class.


def _library_aliases(tree):
    """{local name: module object} for `import ballistic [as b]` and
    `from ballistic import _core`."""
    import ballistic
    from ballistic import _core
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "ballistic":
                    out[a.asname or "ballistic"] = ballistic
        elif isinstance(node, ast.ImportFrom) and node.module == "ballistic":
            for a in node.names:
                if a.name == "_core":
                    out[a.asname or "_core"] = _core
    return out


def _unknown_methods(source):
    tree = ast.parse(source)
    aliases = _library_aliases(tree)
    problems = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        bound = {}
        for node in ast.walk(fn):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id in aliases):
                cls = getattr(aliases[node.value.func.value.id], node.value.func.attr, None)
                if isinstance(cls, type):
                    bound[node.targets[0].id] = cls
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in bound and not hasattr(bound[node.value.id], node.attr)):
                cls = bound[node.value.id]
                problems.append(f"line {node.lineno}: {cls.__module__}.{cls.__name__} has no '{node.attr}'")
    return problems


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.relative_to(SCRIPTS).as_posix())
def test_script_methods_exist(script):
    problems = _unknown_methods(script.read_text(encoding="utf-8"))
    assert not problems, f"{script.name}:\n  " + "\n  ".join(problems)


def test_method_guard_detects_a_removed_name():
    probe = ("from ballistic import _core\n"
             "def f(e0, r, v, cfg):\n"
             "    tr = _core.Trajectory(e0, r, v, cfg)\n"
             "    return tr.propagate_to(e0)\n")
    assert len(_unknown_methods(probe)) == 1
    assert _unknown_methods(probe.replace("propagate_to", "propagate")) == []
