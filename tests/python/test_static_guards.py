"""Static guards: source patterns that have caused bugs may not reappear.

Each rule closes a bug class from docs/trap_census.md. A rule is a regex
searched line by line over a set of source trees; files that legitimately
need the pattern are allowlisted with a reason, and a single line can be
exempted with an inline comment ``guard-exempt(RULE-ID): reason``.

Every rule ID must appear in docs/trap_census.md (test_trap_census.py).

The PUB-* rules keep the public tree free of development-only material:
references to private paths, development-phase labels, dated narration and
commit hashes. The private trees they skip are the ones the publish manifest
(tools/publish/manifest.txt, development clone only) excludes; a test checks
the two lists agree.
"""

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Rule:
    id: str
    why: str                       # one line: the bug this prevents
    pattern: str                   # regex, searched per line
    roots: tuple                   # glob patterns (relative to the repo) to scan
    allow: tuple = field(default=())  # glob patterns of files allowed to match
    near: str = ""                 # optional regex: flag only if it matches within
    window: int = 0                #   +-window lines of the hit
    code_only: bool = False        # ignore comment-only lines
    comments_only: bool = False    # C/C++ files: scan only comment lines
                                   # (other file types are scanned whole)


RULES = [
    Rule(
        id="CIP-DRIFT-USE",
        why=("Subtracting the CIP drift from velocity records that were already true "
             "derivatives biased every GFZ arc by ~0.5 m/h (2026-11-01). Only the "
             "transform, the records check and the checked SP3 path may use it."),
        pattern=r"cip_drift_velocity",
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/**/*.py", "python/src/*.cpp",
               "python/src/*.hpp", "scripts/**/*.py", "examples/**/*"),
        allow=("include/ballistic/coordinates/iau_transforms.hpp",
               "include/ballistic/coordinates/velocity_check.hpp",
               "python/src/types_bindings.cpp",
               "python/ballistic/readers.py",           # re-export only
               "python/ballistic/_removed.py",          # moved-name table
               "scripts/_sp3_velocity.py"),             # gated by verify()
    ),
    Rule(
        id="SAT-LITERAL",
        why=("Satellite properties copied as literals drifted from the cited registry "
             "(Sentinel-1 at 2300 kg / 10 m^2, CHAMP at 510 kg / 3.5 m^2 / Cd 0.8 in tests "
             "long after the registry was corrected). Read include/ballistic/satellites.hpp "
             "or ballistic.satellite_properties; synthetic objects must not be named after "
             "a real satellite."),
        pattern=(r"(?i)(\b(mass|mass_kg|drag_area|drag_area_m2|area|area_m2|srp_area|srp_area_m2|cd)"
                 r"\b\s*[=:]\s*[0-9])|from_mass_cd_area\(\s*[0-9]|BetaProfile1D(::|\.)constant\(\s*[0-9]"
                 r"|SrpConfig\s*[\({]\s*\.?mass_kg\s*=\s*[0-9]"),
        roots=("tests/**/*.cpp", "tests/**/*.hpp", "tests/python/*.py", "examples/**/*.cpp",
               "examples/**/*.py", "docs/latex/examples/**/*.cpp", "docs/latex/examples/**/*.py",
               "scripts/**/*.py", "python/ballistic/*.py"),
        allow=("tests/python/test_static_guards.py",),   # the probes
        near=r"(?i)\b(sentinel|s1[ac]|s2[bc]|s3[ab]|s6a|champ|grace|goce|swarm|lageos|etalon)\b",
        window=4,
        code_only=True,
    ),
    Rule(
        id="CONST-LITERAL",
        why=("Physical constants typed into code drift apart: J2 had four values in the tree "
             "and constants.hpp's J11-J15 were mistyped by up to 10x (2026-11-02). Use "
             "core/constants.hpp (C++) or ballistic.constants / ballistic.ecef_to_geodetic "
             "(Python). C++ tests may hold independent reference values."),
        pattern=(r"3\.98600441[58]e\+?14|398600\.44|\b6378137(\.0*)?\b|6378\.137|6378136\.3"
                 r"|7\.29211[5]|1\.0826[0-9]*e-0?3"),
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp",
               "python/ballistic/*.py", "scripts/**/*.py", "tests/python/*.py",
               "examples/**/*.cpp", "examples/**/*.py",
               "docs/latex/examples/**/*.cpp", "docs/latex/examples/**/*.py"),
        allow=("include/ballistic/core/constants.hpp",
               # NRL's own WGS84 constants in the faithful port of msis_utils.F90
               "include/ballistic/atmosphere/internal/nrlmsis21_full.hpp",
               # independent high-precision references (mpmath/SymPy verification)
               "tests/python/mpmath_fd_reference.py",
               "tests/python/pole_region_hessian_reference.py",
               "tests/python/verify_gravity_hessian_stage_a.py",
               "tests/python/test_static_guards.py"),
        code_only=True,
    ),
    Rule(
        id="SGP4-ORACLE",
        why=("The MIT `sgp4` package is a TEST ORACLE only (licensing decision, hardening "
             "plan P0): the library's SGP4 is a clean-room implementation and nothing that "
             "ships or that the validation scripts run may depend on the oracle."),
        pattern=r"^\s*(import\s+sgp4\b|from\s+sgp4(\.\w+)*\s+import\b)",
        roots=("python/ballistic/*.py", "scripts/**/*.py", "examples/**/*.py",
               "docs/latex/examples/**/*.py"),
        allow=(),
        code_only=True,
    ),
    Rule(
        id="EPOCH-JD-DIFF",
        why=("Julian dates are doubles on a ~40 us grid; differences of them carry that "
             "error into derivatives, interpolation and time-scale offsets (a GPS-UTC "
             "offset computed this way was off by tens of microseconds). Use Epoch - Epoch "
             "and the per-scale constructors (Epoch.from_gps, ...)."),
        pattern=(r"(tt_jd|utc_jd|tai_jd|gps_jd|ut1_jd|tdb_jd)(\(\))?\s*-\s*[A-Za-z_][A-Za-z_0-9.]*"
                 r"\.(tt_jd|utc_jd|tai_jd|gps_jd|ut1_jd|tdb_jd|jd|mjd)"
                 r"|\.m?jd\([^)]*\)\s*-\s*[A-Za-z_][A-Za-z_0-9.]*\.m?jd\("),
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp",
               "python/ballistic/*.py", "scripts/**/*.py", "examples/**/*.cpp", "examples/**/*.py"),
        allow=("tests/python/test_static_guards.py",),
        code_only=True,
    ),
    Rule(
        id="EOP-DIRECT",
        why=("Earth orientation values must come from EOPData::lookup (or the accessors "
             "built on it), which hold the edge <= 30 days, then raise, and refuse with no "
             "table unless EopPolicy::AssumeZero. Gating a read on in_range()/is_loaded() "
             "is how the transforms silently used UT1-UTC = 0 outside the table."),
        pattern=(r"(EOPData::instance\(\)|\beop)\.in_range\("
                 r"|EOPData::instance\(\)\.is_loaded\(\)\s*&&"),
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp"),
        allow=("include/ballistic/time/eop_data.hpp",),
        code_only=True,
    ),
    Rule(
        id="TOD-IN-LIBRARY",
        why=("The true-of-date frame (R3(GAST) * ITRF) is not GCRS: it omits ~0.35 deg of "
             "precession since 2000 (22 km at LEO in 2013). Until 2026-11 it was called "
             "'ECI', and the multisensor accelerometer rotation, the CHAMP accelerometer "
             "parser and the GOCE campaign fed it to GCRS consumers. Use "
             "coords::itrf_to_gcrs_full / gcrs_to_itrf_full / gcrs_to_itrf_rotation."),
        pattern=r"legacy_tod\.hpp|\b(itrf_to_tod|tod_to_itrf|tod_velocity_to_itrf)\w*\s*\(",
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp",
               "python/ballistic/*.py", "scripts/**/*.py", "examples/**/*.cpp",
               "docs/latex/examples/**/*.cpp", "tests/**/*.cpp", "tests/**/*.hpp"),
        allow=(),
        code_only=True,
    ),
    Rule(
        id="ECI-NAME",
        why=("'ECI' named two different frames here: GCRS (util::ecef_to_eci, "
             "Trajectory::state_eci) and the true equator of date (coords::ecef_to_eci, "
             "Python eci_to_ecef). Public names say GCRS, ITRF or TOD."),
        pattern=(r"\b[A-Za-z0-9_]*(_eci|eci_)[A-Za-z0-9_]*\s*\("
                 r"|\.def\w*\(\s*\"[^\"]*\beci|\.def\w*\(\s*\"[^\"]*_eci"),
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp",
               "python/ballistic/*.py"),
        allow=(),
        code_only=True,
    ),
    Rule(
        id="CATCH-ALL",
        why=("A catch (...) that swallows turns every failure into a substitution: the "
             "CHAMP accelerometer parser dropped records on any error (including missing "
             "leap-second coverage) and a malformed EOF value became 0.0 = Earth's center. "
             "Catch the specific exception, or rethrow / report."),
        pattern=r"catch\s*\(\s*\.\.\.\s*\)",
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp"),
        allow=(),
        code_only=True,
    ),
    Rule(
        id="SP3-LOCAL-PARSER",
        why=("Nine private SP3 readers disagreed: four read GPS-time files as UTC (18 s in "
             "2026), one dropped records whose clock ran into z, merged files scored a start "
             "from one solution against truth from another. Read SP3 with "
             "validation::sp3 (Python ballistic.sp3 / scripts/_sp3_records.py)."),
        pattern=(r"def parse_sp3|def derive_velocities|lagrange_derivative_coeffs|epoch_tuple"
                 r"|startswith\(['\"][*PV]['\"]\)|line\[9:12\]|SP3Parser|sp3_parser\.hpp"),
        roots=("scripts/**/*.py", "python/ballistic/*.py", "tests/python/*.py",
               "include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp",
               "tests/**/*.cpp", "tests/**/*.hpp", "examples/**/*.cpp"),
        allow=(# download summaries: count what was fetched, produce no truth
               "scripts/download_ilrs_sp3.py", "scripts/download_qzss_sp3.py",
               "tests/python/test_static_guards.py"),
        code_only=True,
    ),
    Rule(
        id="LAGRANGE-RAW",
        why=("The raw Lagrange templates extrapolate outside the data (a degree-8 polynomial "
             "a few nodes out is wildly wrong); SP3/EOF interpolation used them on JD "
             "doubles. Use math::lagrange_interpolate / lagrange_derivative (range-checked) "
             "or validation::sp3::Arc::interpolate."),
        pattern=r"lagrange_(interpolate|derivative)_vector\s*\(",
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp",
               "tests/**/*.cpp", "examples/**/*.cpp"),
        allow=("include/ballistic/math/interpolation.hpp",
               # range-checked by the caller before the raw call
               "include/ballistic/validation/eof_parser.hpp"),
        code_only=True,
    ),
    Rule(
        id="SPHERICAL-ALT",
        why=("Altitude is geodetic (above the WGS84 ellipsoid): |r| - R is wrong by up to "
             "21 km at 45 deg latitude, and coords::ecef_to_altitude itself was r - N(geocentric "
             "latitude) until 2026-11 (-21 km at 45 deg, -41 km at 80 deg), driving "
             "propagate_to_altitude and at_alt. Use coords::ecef_to_geodetic / "
             "ballistic.ecef_to_geodetic (Earth-fixed position)."),
        pattern=(r"(\.norm\(\)|linalg\.norm\([^)]*\))\s*\)?\s*(/\s*1000\.0\s*\)?\s*)?-\s*"
                 r"([\w:.]*RADIUS\w*|R_EARTH|R_E\b|6371|6378)"
                 r"|[-+]\s*6371(\.0+)?\b(?!\.\d)|[-+]\s*6371000|[-+]\s*6371e3"),
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp",
               "python/ballistic/*.py", "scripts/**/*.py", "tests/**/*.cpp", "tests/**/*.hpp",
               "tests/python/*.py", "examples/**/*.cpp"),
        allow=("tests/python/test_static_guards.py",),   # the probe below
        code_only=True,
    ),
    Rule(
        id="SKIP-AS-PASS",
        why=("A C++ test cell that prints SKIPPED and returns is counted as PASSED: four GCRS "
             "cells, the DE440 comparison, the EOP sanity cell and the whole GEO test passed "
             "for months without running (wrong paths, absent data). Use TEST_SKIP(reason) "
             "(fails under BALLISTIC_REQUIRE_TEST_DATA=1) or TEST_SKIP_OPT_IN(reason)."),
        pattern=r'std::cout\s*<<[^;]*SKIP',
        roots=("tests/**/*.cpp", "tests/**/*.hpp"),
        allow=("tests/test_utils.hpp",
               "tests/diagnostics/*"),   # tools, not TestRunner cells
        code_only=True,
    ),
    Rule(
        id="SILENT-CLAMP",
        why=("A requested degree clamped to what is loaded runs a different model than "
             "the one asked for: gravity degree 720 silently ran 360, ocean tides above "
             "the file's degree ran the file's, J16+ ran J15. Raise ConfigError instead."),
        pattern=r"(std::min|std::clamp|\(std::min\))\s*\([^;]*\b\w*degree\w*",
        roots=("include/**/*.hpp", "src/**/*.cpp", "python/src/*.cpp", "python/src/*.hpp"),
        allow=("include/ballistic/atmosphere/internal/*",),   # NRL ports, verbatim
        code_only=True,
    ),
    Rule(
        id="FLOAT-MJD-EPOCH",
        why=("An MJD double rounds to ~0.6 us, so an Epoch built from one jitters: the "
             "Sentinel year campaign's e0 + 3600 s landed past its drag table's last epoch "
             "(a false frozen-atmosphere warning) and its start labels jittered. Build "
             "epochs from the calendar (Epoch.from_tai / from_gps / from_utc)."),
        pattern=r"Epoch\.from_mjd\(",
        roots=("scripts/**/*.py", "examples/**/*.py", "docs/latex/examples/**/*.py"),
        code_only=True,
    ),
]

# --- Publication rules -------------------------------------------------------
#
# The public tree is a filtered copy of the development tree. These trees are
# development-only and never published (must equal the manifest's excludes,
# see test_private_trees_match_the_publish_manifest).
PRIVATE_TREES = ("CLAUDE.md", ".claude/", "archive/", "tools/", "docs/dev/",
                 "docs/research/", "tests/diagnostics/", "scripts/research/")
_PRIVATE_ALLOW = tuple(t + "*" if t.endswith("/") else t for t in PRIVATE_TREES)

PUBLIC_ROOTS = ("include/**/*.hpp", "src/**/*.cpp", "python/ballistic/*.py",
                "python/ballistic/*.pyi", "python/src/*.cpp", "python/src/*.hpp",
                "python/README.md", "scripts/**/*.py", "examples/**/*",
                "tests/**/*.cpp", "tests/**/*.hpp", "tests/**/*.py",
                "tests/CMakeLists.txt", "tests/**/README.md", "docs/**/*.md",
                "docs/**/*.tex", "README.md", "CMakeLists.txt",
                ".github/workflows/*.yml", "ruff.toml")
_GUARD_SELF = ("tests/python/test_static_guards.py",)

# A 7-10 character lowercase hex word with a letter and a digit, standing
# alone (not part of a path, identifier, number, color or version; a sentence's
# full stop may follow), and not a
# scientific literal such as 42164e3.
_HEX_WORD = (r"(?<![\w./#-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])"
             r"(?![0-9]+e[0-9]+(?![\w/-]|\.\w))[0-9a-f]{7,10}(?![\w/-]|\.\w)")

RULES += [
    Rule(
        id="PUB-AI-REF",
        why=("Public files must not point at the development assistant's notes, memory "
             "or session (the development notes file, its memory index, 'user decision' "
             "narration): readers of the public tree cannot see them. State the reason "
             "itself."),
        pattern=(r"CLAUDE\.md|~/\.claude|\.claude/|\bmemory/|\bmemory `|(?i:user decision)"
                 r"|\bClaude\b|Anthropic"),
        roots=PUBLIC_ROOTS,
        allow=_PRIVATE_ALLOW + _GUARD_SELF,
    ),
    Rule(
        id="PUB-PHASE-ID",
        why=("Development-phase labels (a lettered or numbered phase, a hardening step "
             "number) mean nothing outside the development history. Say what changed and "
             "why instead."),
        pattern=(r"\bPhase[ -][A-Z0-9]\b|\bv2_phase|\bhardening P\d"
                 # a bare P0-P12, but not P2(x), pb.P0, `P0 = ...` or `Matrix P0;`
                 r"|(?<![\w/.])P([0-9]|1[0-2])\b(?![(\d])(?!\s*[=;])"),
        roots=PUBLIC_ROOTS,
        allow=_PRIVATE_ALLOW + _GUARD_SELF + (
            "include/ballistic/gravity/point_mass.hpp",   # Legendre P2, P3, ...
            "tests/unit/test_point_mass.cpp",             # its tests
            "include/ballistic/atmosphere/internal/*",    # NRL ports, verbatim
            "docs/latex/*",                               # mathematics
        ),
    ),
    Rule(
        id="PUB-DATE",
        why=("Dated narration ('fixed 2026-11-02') in public code and docs records "
             "development history, which belongs in the commit log. Data dates (an "
             "epoch, a file's coverage) carry guard-exempt(PUB-DATE): data."),
        pattern=r"\b20(2[4-9])-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b",
        roots=PUBLIC_ROOTS,
        allow=_PRIVATE_ALLOW + _GUARD_SELF + (
            # NRL ports: their own release history, verbatim
            "include/ballistic/atmosphere/internal/nrlmsis21_full.hpp",
            "include/ballistic/atmosphere/internal/nrlmsis21_parm.hpp",
            "include/ballistic/atmosphere/internal/hwm14.hpp",
            "include/ballistic/atmosphere/internal/hwm14_coefficients.hpp",
            "include/ballistic/satellites.hpp",          # provenance strings
            "tests/data/*README.md",                     # data coverage
            "docs/validation_claims.md",                 # measurement dates
            "docs/satellite_properties_provenance.md",
            "docs/third_party_provenance.md",
            "docs/latex/results/*",                      # generated tables
        ),
        comments_only=True,
    ),
    Rule(
        id="PUB-COMMIT-HASH",
        why=("A commit hash in a public comment names a development commit that the "
             "public history does not contain."),
        pattern=_HEX_WORD,
        roots=PUBLIC_ROOTS,
        allow=_PRIVATE_ALLOW + _GUARD_SELF + (
            "docs/validation_claims.md",                 # the Commit column
            "docs/latex/build.sh",                       # pinned tool checksums
        ),
        comments_only=True,
    ),
    Rule(
        id="PUB-PRIVATE-PATH",
        why=("A public file that cites a development-only path (archive/, docs/research/, "
             "docs/dev/, tests/diagnostics/, scripts/research/, tools/publish/, the "
             "development notes file) sends the reader to something that does not exist "
             "in the public tree."),
        pattern=(r"(?<![\w/.-])(CLAUDE\.md|\.claude/|archive/|docs/research/|docs/dev/"
                 r"|tests/diagnostics/|tests/benchmarks/|scripts/research/|tools/publish/)"),
        roots=PUBLIC_ROOTS,
        allow=_PRIVATE_ALLOW + _GUARD_SELF + (
            # a frozen derivation record; its citations of docs/dev/ stay as written
            "tests/python/verify_gravity_hessian_stage_a.py",
        ),
    ),
]


# Python: an except block that only passes/continues/returns a constant
# hides the failure (AST check; same allow and guard-exempt conventions,
# the comment goes on the `except` line).
EXCEPT_PASS = Rule(
    id="EXCEPT-PASS",
    why=("A silent `except ...: pass/continue` skips bad input without a trace: the "
         "GOCE density summary read the longitude column as density and skipped rows "
         "that did not parse; the Python loaders skipped malformed rows; a private SP3 "
         "reader dropped every record whose clock field ran into z. Raise, or "
         "count and warn (DataSkippedWarning)."),
    pattern="",
    roots=("python/ballistic/*.py", "scripts/**/*.py"),
    allow=(),
)


def _silent_handler(h):
    import ast
    return all(isinstance(b, (ast.Pass, ast.Continue))
               or (isinstance(b, ast.Return) and (b.value is None or isinstance(b.value, ast.Constant)))
               or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
               for b in h.body)


def except_pass_violations(text_by_file=None):
    import ast
    rule = EXCEPT_PASS
    items = (text_by_file.items() if text_by_file is not None else
             ((str(p.relative_to(REPO)), p.read_text(errors="replace", encoding="utf-8")) for p in _files(rule)))
    out = []
    for rel, text in items:
        if _allowed(rule, rel):
            continue
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ExceptHandler) and _silent_handler(node):
                line = lines[node.lineno - 1]
                if f"guard-exempt({rule.id})" not in line:
                    out.append((rel, node.lineno, line.strip()))
    return out


def test_no_silent_except():
    bad = except_pass_violations()
    assert not bad, (f"EXCEPT-PASS: {EXCEPT_PASS.why}\n" +
                     "\n".join(f"  {f}:{n}: {line}" for f, n, line in bad))


def test_no_silent_except_catches_its_bug():
    probe = "try:\n    x = float(s)\nexcept ValueError:\n    pass\n"
    assert except_pass_violations({"scripts/validate_example.py": probe})
    exempt = probe.replace("except ValueError:",
                           "except ValueError:  # guard-exempt(EXCEPT-PASS): test")
    assert not except_pass_violations({"scripts/validate_example.py": exempt})


def _files(rule):
    seen = set()
    for pat in rule.roots:
        for p in REPO.glob(pat):
            if p.is_file() and p not in seen and "build" not in p.relative_to(REPO).parts:
                seen.add(p)
                yield p


def _allowed(rule, rel):
    return any(fnmatch.fnmatch(rel, a) for a in rule.allow)


def violations(rule, text_by_file=None):
    """[(relative path, line number, line)] that break `rule`.

    `text_by_file` (rel -> text) replaces the repository scan; used by the
    self-tests below to show each rule catches its own bug.
    """
    rx = re.compile(rule.pattern)
    near = re.compile(rule.near) if rule.near else None
    items = (text_by_file.items() if text_by_file is not None else
             ((str(p.relative_to(REPO)), p.read_text(errors="replace", encoding="utf-8")) for p in _files(rule)))
    out = []
    for rel, text in items:
        if _allowed(rule, rel):
            continue
        lines = text.splitlines()
        cxx = rule.comments_only and rel.endswith((".cpp", ".hpp", ".h"))
        for i, line in enumerate(lines):
            if not rx.search(line) or f"guard-exempt({rule.id})" in line:
                continue
            if rule.code_only and line.lstrip().startswith(("//", "#", "*", "/*")):
                continue
            if cxx and not line.lstrip().startswith(("//", "/*", "*")):
                continue
            if near is not None:
                ctx = lines[max(0, i - rule.window): i + rule.window + 1]
                if not any(near.search(c) for c in ctx):
                    continue
            out.append((rel, i + 1, line.strip()))
    return out


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.id)
def test_rule_holds(rule):
    bad = violations(rule)
    assert not bad, (f"{rule.id}: {rule.why}\n" +
                     "\n".join(f"  {f}:{n}: {line}" for f, n, line in bad))


# Each rule must catch the bug it exists for (a guard that cannot fail is
# not a guard). Probe text: a line that reintroduces the bug, in a file the
# rule scans and does not allow.
PROBES = {
    "CIP-DRIFT-USE": ("scripts/validate_example.py",
                      "v = v - ballistic.readers.cip_drift_velocity(epoch, pos)"),
    "CONST-LITERAL": ("scripts/validate_example.py", "GM = 3.986004418e14"),
    "EPOCH-JD-DIFF": ("scripts/validate_example.py", "dt = (e.gps_jd - e.utc_jd) * 86400.0"),
    "SGP4-ORACLE": ("python/ballistic/example.py", "from sgp4.api import Satrec"),
    "EOP-DIRECT": ("include/ballistic/coordinates/example.hpp",
                   "if (time::EOPData::instance().is_loaded() && time::EOPData::instance().in_range(mjd)) {"),
    "TOD-IN-LIBRARY": ("include/ballistic/calibration/detail/example.hpp",
                       "Matrix3d R = coords::itrf_to_tod_rotation(epoch);"),
    "ECI-NAME": ("include/ballistic/example.hpp",
                 "[[nodiscard]] StateVector state_eci() const {"),
    "CATCH-ALL": ("include/ballistic/validation/example.hpp",
                  "    } catch (...) {}"),
    "SP3-LOCAL-PARSER": ("scripts/validate_example.py",
                         "def parse_sp3(filepath, target_ids=None):"),
    "LAGRANGE-RAW": ("include/ballistic/validation/example.hpp",
                     "return math::detail::lagrange_interpolate_vector(get_t, get_x, n, t, 9);"),
    "SPHERICAL-ALT": ("scripts/validate_example.py",
                      "alt_km = (np.linalg.norm(pos) / 1000.0) - 6371.0"),
    "SKIP-AS-PASS": ("tests/validation/test_example.cpp",
                     'if (!eof) { std::cout << "SKIPPED (no EOF file found)" << std::endl; return; }'),
    "FLOAT-MJD-EPOCH": ("scripts/validate_example.py",
                        "    e = ballistic.Epoch.from_mjd(mjd, ballistic.TimeScale.TAI)"),
    "SILENT-CLAMP": ("include/ballistic/gravity/example.hpp",
                     "const int N = std::min(degree, model.max_degree());"),
    "SAT-LITERAL": ("tests/validation/test_example.cpp",
                    "// Sentinel-1 parameters\n    static constexpr double MASS = 2300.0;"),
    "PUB-AI-REF": ("docs/example.md", "See CLAUDE.md for the background."),
    "PUB-PHASE-ID": ("include/ballistic/example.hpp",
                     "// Added in P7 together with the event locator."),
    "PUB-DATE": ("include/ballistic/example.hpp", "// fixed 2026-11-02 after the audit"),
    "PUB-COMMIT-HASH": ("include/ballistic/example.hpp",
                        "// measured at 619e7cb on identical inputs"),
    "PUB-PRIVATE-PATH": ("README.md",
                         "Details: docs/research/sentinel_drag_drift.md."),
}


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.id)
def test_rule_catches_its_bug(rule):
    assert rule.id in PROBES, f"{rule.id} has no self-test probe"
    rel, line = PROBES[rule.id]
    assert violations(rule, {rel: line}), f"{rule.id} did not flag its own probe"
    assert not violations(rule, {rel: line + f"  # guard-exempt({rule.id}): test"}), \
        f"{rule.id} ignores guard-exempt"


def _rule(rule_id):
    return next(r for r in RULES if r.id == rule_id)


def test_comments_only_scans_cxx_comments_and_other_files_whole():
    rule = _rule("PUB-DATE")
    code = 'const auto e = parse("2026-01-05T00:00:00");'
    assert not violations(rule, {"tests/unit/test_example.cpp": code})
    assert violations(rule, {"tests/unit/test_example.cpp": "/* run on 2026-01-05 */"})
    assert violations(rule, {"tests/unit/test_example.cpp": " * since 2026-01-05"})
    assert violations(rule, {"tests/python/test_example.py": 'e = "2026-01-05"'})


def test_commit_hash_pattern():
    rx = re.compile(_HEX_WORD)
    for hit in ("at 619e7cb.", "(abf71a2)", "commit 1ca61c2 merged", "deadbee5"):
        assert rx.search(hit), hit
    for miss in ("r = 42164e3;", "1.5e10", "0x7fffffff", "#1f77b4", "v1.2.3abcdef",
                 "deadbeef", "1234567", "a/619e7cb", "619e7cb-dirty", "619e7cb_x",
                 "abcdef0123456", "1e-3", "2.5e+06", "42164e3.", "619e7cb.cpp"):
        assert not rx.search(miss), miss


def test_phase_pattern_spares_legendre_and_identifiers():
    rule = _rule("PUB-PHASE-ID")
    for ok in ("double P_n = legendre(n, x);", "auto v = P2(x);", "P10(x) + P11(x)",
               "SP3 file", "P_20 term", "see ITRF2014", "x = 1.0e-P", "Eigen::Matrix2d P0;",
               "P0 = Eigen::Matrix2d::Identity();", "M = pb.Phi * pb.P0;"):
        assert not violations(rule, {"src/example.cpp": ok}), ok
    for bad in ("// Phase K measured it", "// since hardening P3", "// see P11 re-measure",
                "(P5)", "v2_phase_c"):
        assert violations(rule, {"src/example.cpp": bad}), bad


def _manifest_excludes(text):
    out = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("exclude "):
            out.add(line.split(None, 1)[1].strip())
    return out


def test_private_trees_match_the_publish_manifest():
    """The PUB-* rules skip exactly what the publish manifest leaves out; if
    the two drifted apart, a newly private tree would be scanned (noise) or a
    newly public one skipped (a leak). The manifest exists only in the
    development clone."""
    manifest = REPO / "tools" / "publish" / "manifest.txt"
    if not manifest.exists():
        return   # the public tree: nothing to compare against
    excludes = _manifest_excludes(manifest.read_text(encoding="utf-8"))
    assert excludes == set(PRIVATE_TREES), (
        f"manifest excludes {sorted(excludes)} but PRIVATE_TREES is "
        f"{sorted(PRIVATE_TREES)}")
    for rule in RULES:
        if rule.id.startswith("PUB-"):
            missing = [a for a in _PRIVATE_ALLOW if a not in rule.allow]
            assert not missing, f"{rule.id} does not allow private trees {missing}"


def test_manifest_check_detects_drift():
    text = "include docs/\nexclude docs/research/  # notes\nexclude CLAUDE.md\n"
    assert _manifest_excludes(text) == {"docs/research/", "CLAUDE.md"}
    assert _manifest_excludes(text) != set(PRIVATE_TREES)
