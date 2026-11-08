"""``ballistic.doctor()``: is this installation ready for precise work?

Reports what a precise prediction depends on and what is missing (Earth
orientation parameters (EOP), leap seconds, space weather, ocean tides,
the optional NRL drag models), then runs a self-test of a few seconds on data embedded
below:

* the library's own SGP4 (the standard model for two-line element sets,
  written from the public specification) against the published
  verification output for the Spacetrack Report #3 test object
  (tcppver.out, Vallado et al. 2006);
* a 10-minute prediction from an ESA Sentinel-1A precise orbit state (the
  precise orbit ephemeris file, POEORB, of 5 January 2026) against the same
  file's state 10 minutes later. With Earth orientation data loaded the default configuration lands within
  about 5 cm (measured 0.052 m); without it the error is meters.

Also ``python -m ballistic doctor`` (exit status 1 if a check fails).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import _core

OK, NOTE, WARN, FAIL = "ok", "note", "warn", "fail"

# ESA S1A_OPER_AUX_POEORB_OPOD_20260125T070655_V20260104T225942_20260106T005942
# (tests/data/eof/sentinel_jan2026): ITRF2020, UTC, meters and m/s.
_S1A_T0_UTC = (2026, 1, 5, 11, 59, 42.0)
_S1A_START = ((-858245.729566, 3986052.969605, -5790116.906923),
              (851.699434, 6256.268159, 4183.014931))
_S1A_AFTER_600S = (-83908.560475, 6689918.724326, -2314209.349909)
_S1A_BOUND_M = 0.5

# Spacetrack Report #3 test object; position at t = 0 and 120 min [km]
# from SGP4-VER tcppver.out (also tests/unit/test_sgp4.cpp).
_STR3 = ("1 88888U          80275.98708465  .00073094  13844-3  66816-4 0    87",
         "2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518  1058")
_STR3_R_KM = {0.0: (2328.96975262, -5995.22051338, 1719.97297192),
              120.0: (1020.69234558, 2286.56260634, -6191.55565927)}

# Third-party data and models whose terms a user must know (THIRD_PARTY_NOTICES.md).
_LICENSE_NOTES = (
    ("FES2004 (ocean tides)", "downloaded at run time from the IERS Conventions, not redistributed"),
)


@dataclass
class Check:
    """One line of the report: what was checked, its status
    ("ok" / "note" / "warn" / "fail") and a human-readable detail."""
    name: str
    status: str          # ok / note / warn / fail
    detail: str


@dataclass
class DoctorReport:
    """The result of ``ballistic.doctor()``: a list of Check entries.

    ``str(report)`` formats it as the printed table; ``report.ok`` is True
    when no check failed.
    """
    checks: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No check failed (warnings and notes allowed)."""
        return all(c.status != FAIL for c in self.checks)

    def add(self, name, status, detail):
        """Append a Check."""
        self.checks.append(Check(name, status, detail))

    def __str__(self):
        width = max(len(c.name) for c in self.checks)
        lines = [f"[{c.status:4s}] {c.name:{width}s}  {c.detail}" for c in self.checks]
        verdict = "ready" if self.ok else "NOT ready: fix the failed checks"
        n_warn = sum(c.status == WARN for c in self.checks)
        return "\n".join(lines + [f"\n{verdict}" + (f" ({n_warn} warning(s))" if n_warn else "")])


def _today_mjd():
    """The current time as a Modified Julian Date (MJD, days, UTC)."""
    return datetime.now(timezone.utc).timestamp() / 86400.0 + 40587.0


def _ago(days):
    """'today' or 'N days ago', for display."""
    return "today" if days < 1 else f"{days:.0f} day{'s' if days >= 1.5 else ''} ago"


def _date(mjd):
    """Calendar date of a UTC MJD (whole days only, for display)."""
    d = datetime.fromtimestamp((mjd - 40587.0) * 86400.0, tz=timezone.utc)
    return d.strftime("%Y-%m-%d")


def _data_checks(report):
    """Add the version, data-file and license checks to ``report``."""
    import ballistic

    report.add("version", OK, f"ballistic {ballistic.__version__} (core {_core.__version__})")

    bi = _core.build_info()
    report.add("build", OK,
               f"{bi['compiler']}, C++{bi['cpp_standard']}, "
               f"OpenMP: {'yes' if bi['with_openmp'] else 'no'}, {bi['build_type']}")

    cov = ballistic.eop_coverage()
    if cov is None:
        report.add("Earth orientation", FAIL,
                   "no EOP table loaded: frame transforms are wrong by meters at LEO and a "
                   "Trajectory refuses to run. Call ballistic.load()")
    else:
        lag = _today_mjd() - cov["measured_until_mjd"]
        status = WARN if lag > 30 else OK
        report.add("Earth orientation", status,
                   f"{_date(cov['first_mjd'])} to {_date(cov['last_mjd'])}, measured until "
                   f"{_date(cov['measured_until_mjd'])} ({_ago(lag)}); the edge value is "
                   f"held {cov['hold_days']:.0f} days past the table, then propagation stops"
                   + (" - refresh with ballistic.load()" if status == WARN else ""))

    li = _core.get_leap_second_info()
    days_left = li.expiration_jd - 2400000.5 - _today_mjd()
    status = WARN if days_left < 90 else OK
    report.add("leap seconds", status,
               f"TAI-UTC {li.current_tai_utc:.0f} s, table valid until {_date(li.expiration_jd - 2400000.5)}"
               + (" (from file)" if li.loaded_from_file else " (embedded table)")
               + (f" - {'expired' if days_left < 0 else 'expires soon'}: ballistic.load()"
                  if status == WARN else ""))

    last = _core.SpaceWeatherHistory.last_measured_day()
    if last is None:
        report.add("space weather", WARN,
                   "none loaded: drag cannot be configured (it never runs on assumed "
                   "conditions). Call ballistic.load()")
    else:
        lag = _today_mjd() - last.jd(_core.TimeScale.UTC) + 2400000.5
        status = WARN if lag > 7 else OK
        report.add("space weather", status,
                   f"measured until {str(last)[:10]} ({_ago(lag)}); CelesTrak's ~45-day "
                   "forecast is used past that, with a warning")

    if _core.ocean_tides_loaded():
        report.add("ocean tides", OK, "FES2004 loaded (applied to degree 50 by default)")
    else:
        report.add("ocean tides", WARN,
                   "not loaded: skipped, with a warning (they cut 1-h LEO error by 16-28%). "
                   "Call ballistic.load()")

    report.add("embedded models", NOTE,
               "EGM2008 gravity to degree 360, DE440 ephemeris 1950-2100, IERS 2010 tides")
    st = ballistic.nrl_models()
    if st.installed:
        report.add("NRL models (drag)", NOTE,
                   f"{st.models} installed ({st.path}); NRLMSIS 2.1 is licensed for research, "
                   "academic and non-profit use only (NRL-SOF-014-1)")
    elif st.problem:
        report.add("NRL models (drag)", WARN,
                   f"found but not loaded: {st.problem}. Reinstall with "
                   "ballistic.install_nrl_models(accept_license=True)")
    else:
        report.add("NRL models (drag)", NOTE,
                   "not installed: model-based drag (make_drag_environment, recommended_config "
                   "with drag) is unavailable; propagation without drag and drag from measured "
                   "density work. Install with ballistic.install_nrl_models(accept_license=True) "
                   "(NRLMSIS 2.1: non-commercial use only)")
    for name, text in _LICENSE_NOTES:
        report.add(name, NOTE, text)


def _self_test(report):
    """Run the SGP4 and Sentinel-1A self-tests and add their results to ``report``."""
    import numpy as np

    import ballistic

    t = time.perf_counter()
    model = _core.Sgp4(_core.Tle.parse(*_STR3))
    worst = max(np.linalg.norm(np.asarray(model.propagate_minutes(m)[0]) - r)
                for m, r in _STR3_R_KM.items())
    report.add("self-test: SGP4", OK if worst < 1e-6 else FAIL,
               f"Spacetrack Report #3 object vs SGP4-VER: {worst * 1e6:.2g} mm")

    if ballistic.eop_coverage() is None:
        report.add("self-test: Sentinel-1A", WARN, "skipped: needs Earth orientation data")
        return
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")   # report, do not raise, in strict mode
        state = ballistic.State.from_itrf(ballistic.Epoch.from_utc(*_S1A_T0_UTC), *_S1A_START,
                                          ballistic.ITRFRealization.ITRF2020)
        traj = ballistic.Trajectory(state)
        traj.propagate_for(600.0)
        err = float(np.linalg.norm(np.asarray(traj(600.0, frame="ECEF").position) - _S1A_AFTER_600S))
    report.add("self-test: Sentinel-1A", OK if err < _S1A_BOUND_M else FAIL,
               f"10-min prediction from an ESA precise orbit: {err:.3f} m "
               f"(bound {_S1A_BOUND_M} m; ~0.05 m expected) in {time.perf_counter() - t:.2f} s")


def doctor(load: bool = True, self_test: bool = True, verbose: bool = True) -> DoctorReport:
    """Check that this installation is ready for precise work.

    Args:
        load: first load the cached data files (``ballistic.load(offline=True)``;
            never downloads). False reports what is loaded now.
        self_test: run the SGP4 and Sentinel-1A checks (a few seconds).
        verbose: print the report.

    Returns:
        DoctorReport: ``.checks`` (name, status, detail) and ``.ok`` (no
        check failed). Statuses: ok, note, warn (works, less accurate or
        incomplete), fail (precise work will not run or will be wrong).
    """
    import ballistic

    report = DoctorReport()
    if load:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ballistic.load(offline=True, verbose=False)
    _data_checks(report)
    if self_test:
        _self_test(report)
    if verbose:
        print(report)
    return report
