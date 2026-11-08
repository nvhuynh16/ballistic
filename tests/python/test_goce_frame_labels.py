"""SP3 coordinate-system labels are read explicitly (ballistic.sp3.classify_frame).

Trap FRAME-LABEL-DEFAULT: an unknown label used to fall through to
Earth-fixed, and IGb08 was once listed as inertial. The ITRF realization
follows the label (trap ITRF-REALIZATION): the campaign scripts take it from
each file's arc, so a JPL GNV1B conversion labeled IGS14 is ITRF2014.
The classification lives in the library's one SP3 reader.
"""

import pytest

import ballistic

R = ballistic.ITRFRealization


@pytest.mark.parametrize("label, inertial, itrf", [
    ("J2000", True, None), ("IGb08", False, R.ITRF2008), ("IGS05", False, R.ITRF2005),
    ("IGS14", False, R.ITRF2014), ("IGb14", False, R.ITRF2014), ("IGS20", False, R.ITRF2020),
    ("SLR20", False, R.ITRF2020), ("CTS", False, None),
])
def test_known_labels_map_to_frame_and_realization(label, inertial, itrf):
    f = ballistic.sp3.classify_frame(label)
    assert f.inertial == inertial
    assert f.itrf == itrf


@pytest.mark.parametrize("label", ["UNKNOWN", "XYZ", "ECEF", ""])
def test_unknown_label_is_refused(label):
    with pytest.raises(ballistic.FrameError, match="not one this library"):
        ballistic.sp3.classify_frame(label)
