"""Epochs are built in a named time scale, and exactly.

Trap EPOCH-TT-DEFAULT: Epoch(jd) / from_jd(jd) / from_mjd(mjd) used to mean
TT silently, so a UTC Julian date was 69 s (~500 km at LEO) off. Trap
GPS-AS-UTC: GPS-time stamps read with from_utc are 18 s off. These tests
pin the explicit constructors and the refusal of the ambiguous forms.
"""

import datetime as dt

import pytest

import ballistic

E = ballistic.Epoch
TS = ballistic.TimeScale


def test_scale_is_required_for_julian_dates():
    for call in (lambda: E(2460000.5), lambda: E.from_jd(2460000.5),
                 lambda: E.from_mjd(61000.0)):
        with pytest.raises(ballistic.RemovedApiError, match="TimeScale"):
            call()
    assert E.from_jd(2460000.5, TS.UTC) - E.from_utc(2023, 2, 25) == 0.0


def test_calendar_constructors_differ_by_the_scale_offsets():
    utc = E.from_utc(2026, 1, 1, 12, 0, 0.0)
    assert E.from_gps(2026, 1, 1, 12, 0, 18.0) - utc == 0.0      # GPS = UTC + 18 s
    assert E.from_tai(2026, 1, 1, 12, 0, 37.0) - utc == 0.0      # TAI = UTC + 37 s
    assert E.from_tt(2026, 1, 1, 12, 1, 9.184) - utc == pytest.approx(0.0, abs=1e-9)


def test_gps_week_round_trip():
    e = E.from_gps_week(2399, 381600.0)
    assert e - E.from_gps(2026, 1, 1, 10, 0, 0.0) == 0.0
    week, sow = e.to_gps_week()
    assert week == 2399 and sow == pytest.approx(381600.0, abs=1e-9)


def test_datetime_must_be_timezone_aware():
    aware = dt.datetime(2026, 1, 1, 12, 0, 0, 250000, tzinfo=dt.timezone.utc)
    assert E.from_datetime(aware) - E.from_utc(2026, 1, 1, 12, 0, 0.25) == pytest.approx(0.0, abs=1e-9)
    plus2 = dt.datetime(2026, 1, 1, 14, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert E.from_datetime(plus2) - E.from_utc(2026, 1, 1, 12) == 0.0
    with pytest.raises(ballistic.ConfigError, match="timezone-aware"):
        E.from_datetime(dt.datetime(2026, 1, 1, 12))


def test_differences_are_exact():
    t0 = E.from_utc(2026, 6, 15, 7, 31, 12.0)
    assert (t0 + 1e-6) - t0 == pytest.approx(1e-6, abs=1e-11)
    assert (t0 + 3600.0) - t0 == 3600.0      # was 3599.9999866 on the old grid
    back = E.from_tt_split(t0.tt_day, t0.tt_seconds)
    assert back == t0
