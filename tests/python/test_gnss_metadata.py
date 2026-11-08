"""The IGS satellite metadata reader (scripts/_gnss_metadata.py) on the committed excerpt.

The excerpt holds verbatim rows of the public IGS file for the satellites of
the committed GNSS SP3 test files (GPS G01/G02, QZSS J02-J04, BeiDou C01-C05).
The expected SVNs, blocks and masses below are the values in those rows.
Whether the excerpt is still a subset of the current full file is checked by
`scripts/download_igs_metadata.py --check-excerpt`, not here (the full file is
not committed and a skipping test would count against the CI skip census);
the check function itself is tested here on the excerpt.
"""

import datetime as dt
import sys
from pathlib import Path

import ballistic
import pytest
from ballistic.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

# The reader lives in scripts/ (not a package); the path insert above must run first.
import _gnss_metadata as gm  # noqa: E402
import download_igs_metadata as dl  # noqa: E402

EXCERPT = REPO / "tests" / "data" / "gnss" / "igs_satellite_metadata_excerpt.snx"
EPOCH = ballistic.Epoch.from_utc(2024, 1, 15, 12, 0, 0)   # day of the committed SP3 files


@pytest.fixture(scope="module")
def md():
    """The parsed excerpt."""
    return gm.read_sinex(EXCERPT)


def test_blocks_and_nothing_ignored(md):
    """The excerpt has the three parsed blocks and no unknown ones."""
    assert md.blocks_present() == ["SATELLITE/IDENTIFIER", "SATELLITE/PRN", "SATELLITE/MASS"]
    assert md.ignored_blocks == []
    assert len(md.identifiers) == 12


@pytest.mark.parametrize("prn, svn, block, mass_kg, cospar", [
    ("G01", "G063", "GPS-IIF", 1633.0, "2011-036A"),
    ("G02", "G061", "GPS-IIR-B", 1080.0, "2004-045A"),
    ("J02", "J002", "QZS-2I", 2233.0, "2017-028A"),
    ("J03", "J004", "QZS-2I", 2243.0, "2017-062A"),
    ("J04", "J005", "QZS-2A", 2327.0, "2021-096A"),
    ("C01", "C020", "BDS-2G", 1500.0, "2019-027A"),
    ("C02", "C016", "BDS-2G", 1551.0, "2012-059A"),
    ("C03", "C018", "BDS-2G", 1486.0, "2016-037A"),
    ("C04", "C006", "BDS-2G", 1536.0, "2010-057A"),
    ("C05", "C011", "BDS-2G", 1502.0, "2012-008A"),
])
def test_lookup_on_the_sp3_day(md, prn, svn, block, mass_kg, cospar):
    """PRN -> SVN -> block, mass and COSPAR id on the day of the committed SP3 files."""
    assert md.svn_of(prn, EPOCH) == svn
    assert md.block_of(svn) == block
    assert md.mass_of(svn, EPOCH) == mass_kg
    assert md.cospar_of(svn) == cospar


def test_aware_datetime_equals_epoch(md):
    """A timezone-aware datetime gives the same answer; a naive one is refused."""
    t = dt.datetime(2024, 1, 15, 12, tzinfo=dt.UTC)
    assert md.svn_of("G01", t) == "G063"
    with pytest.raises(ConfigError, match="no time zone"):
        md.svn_of("G01", dt.datetime(2024, 1, 15, 12))


def test_open_interval_and_reassignment(md):
    """G01 passed G063 -> G049 -> G080; G080's row has an open end."""
    assert gm.parse_sinex_time("0000:000:00000") is None
    assert gm.parse_sinex_time("2024:352:00000") == dt.datetime(2024, 12, 17)
    assert gm.parse_sinex_time("26:295:28054") == dt.datetime(2026, 10, 22, 7, 47, 34)
    assert md.svn_of("G01", ballistic.Epoch.from_utc(2024, 6, 1, 0, 0, 0)) == "G049"
    # the start of an interval belongs to it, the end does not
    assert md.svn_of("G01", ballistic.Epoch.from_utc(2024, 12, 17, 0, 0, 0)) == "G080"
    assert md.svn_of("G01", ballistic.Epoch.from_utc(2027, 1, 1, 0, 0, 0)) == "G080"
    assert md.mass_of("G080", ballistic.Epoch.from_utc(2027, 1, 1, 0, 0, 0)) == 2161.0


def test_unassigned_prn_message(md):
    """A gap names the PRN, the epoch and the nearest assignments."""
    gap = ballistic.Epoch.from_utc(2024, 4, 15, 0, 0, 0)   # between G063's end and G049's start
    with pytest.raises(ConfigError) as err:
        md.svn_of("G01", gap)
    msg = str(err.value)
    assert "G01" in msg and "2024-04-15T00:00:00" in msg
    assert "G063" in msg and "G049" in msg
    with pytest.raises(ConfigError, match="G99.*nearest assignments: none"):
        md.svn_of("G99", EPOCH)


def test_unknown_svn_and_mass_gap(md):
    """An SVN without an identifier row, or a mass gap, raises ConfigError."""
    with pytest.raises(ConfigError, match="G999"):
        md.block_of("G999")
    with pytest.raises(ConfigError, match="J002.*nearest"):
        md.mass_of("J002", ballistic.Epoch.from_utc(2025, 1, 1, 0, 0, 0))


def test_malformed_input_raises(tmp_path):
    """A non-SINEX file, a bad time and an unclosed block are refused with the line."""
    bad = tmp_path / "x.snx"
    bad.write_text("hello\n")
    with pytest.raises(ValueError, match="%=SNX"):
        gm.read_sinex(bad)
    text = EXCERPT.read_text().replace("2011:197:00000 2024:103:00000", "2011:197:00000 2024:1x3:00000")
    bad.write_text(text)
    with pytest.raises(ValueError, match="line"):
        gm.read_sinex(bad)
    bad.write_text("%=SNX 2.02\n+SATELLITE/NEW_BLOCK\n x\n")
    with pytest.raises(ValueError, match="never closed"):
        gm.read_sinex(bad)


def test_unknown_block_is_listed(tmp_path):
    """A block the reader does not know is skipped and named."""
    f = tmp_path / "x.snx"
    f.write_text("%=SNX 2.02\n+SATELLITE/BOX_WING\n G001 1 2 3\n-SATELLITE/BOX_WING\n%ENDSNX\n")
    assert gm.read_sinex(f).ignored_blocks == ["SATELLITE/BOX_WING"]


def test_excerpt_check_function(tmp_path):
    """check_excerpt passes on itself and names a row the full file lacks."""
    assert dl.check_excerpt(EXCERPT, EXCERPT) == []
    edited = tmp_path / "edited.snx"
    edited.write_text(EXCERPT.read_text().replace("1633.000", "1634.000"))
    problems = dl.check_excerpt(EXCERPT, edited)
    assert len(problems) == 1 and "1634.000" in problems[0]


# --- The GNSS block table (include/ballistic/gnss_blocks.hpp) ----------------

from ballistic import satellite_properties as sp  # noqa: E402


def test_gnss_block_fields():
    """GPS-IIF: base area from the table, Cr nominal, effective area = base x k."""
    b = sp.gnss_block("GPS-IIF")
    assert b.name == "GPS-IIF"
    assert b.srp_base_area_m2 > 0 and b.cr == 1.3
    assert b.srp_area_m2 == pytest.approx(b.srp_base_area_m2 * b.srp_k, rel=1e-15)
    assert b.srp_kind in sp.FIT_KINDS
    assert b.srp_k == sp.GNSS_SRP_FIT.get("GPS-IIF", 1.0)
    assert (b.srp_kind == "fitted") == ("GPS-IIF" in sp.GNSS_SRP_FIT)
    assert sp.GNSS_BLOCKS["GPS-IIF"] == b


def test_gnss_block_prefix_rule():
    """IGS names that extend a row after '-' resolve to that row; exact names win."""
    assert sp.gnss_block("GPS-IIR-A").name == "GPS-IIR"
    assert sp.gnss_block("GPS-IIR-B").name == "GPS-IIR"
    assert sp.gnss_block("GPS-IIR-M").name == "GPS-IIR-M"
    assert sp.gnss_block("BDS-3M-SECM-A").name == "BDS-3M-SECM"


def test_gnss_block_unknown_raises():
    with pytest.raises(ConfigError, match="unknown GNSS block 'GAL-2'"):
        sp.gnss_block("GAL-2")
    with pytest.raises(ConfigError):
        sp.gnss_block("GPS-IIRX")      # a prefix counts only at a '-' boundary


def test_every_block_of_the_excerpt_resolves(md):
    """Every satellite of the committed SP3 files has a block row."""
    for prn in ("G01", "G02", "J02", "J03", "J04", "C01", "C02", "C03", "C04", "C05"):
        sp.gnss_block(md.block_of(md.svn_of(prn, EPOCH)))


def test_fit_table_names_real_blocks():
    for block, k in sp.GNSS_SRP_FIT.items():
        assert block in sp.GNSS_BLOCKS and k > 0
        assert sp.GNSS_BLOCKS[block].srp_kind == "fitted"
