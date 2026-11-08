"""gcrs_to_itrf / itrf_to_gcrs match the propagator's IAU 2006 transform.

The old eci_to_ecef / ecef_to_eci worked in the true equator of date, not
GCRS or J2000: used on J2000 states they miss ~22 km of precession at LEO
(2013). They now raise, naming the GCRS replacement.

"""

import numpy as np
import pytest

import ballistic


def _state():
    epoch = ballistic.Epoch.from_utc(2013, 11, 1, 0, 0, 0.0)
    pos = np.array([6578e3, 0.0, 0.0])
    vel = np.array([0.0, 1000.0, 7700.0])
    return epoch, pos, vel


def test_itrf_to_gcrs_matches_trajectory_and_round_trips():
    epoch, pos, vel = _state()
    ref = ballistic.Trajectory(ballistic.State.from_itrf(epoch, pos, vel))(0.0, frame="GCRS")
    g = ballistic.itrf_to_gcrs(pos, vel, epoch)
    assert np.linalg.norm(np.asarray(g.position) - np.asarray(ref.position)) < 1e-6
    assert np.linalg.norm(np.asarray(g.velocity) - np.asarray(ref.velocity)) < 1e-9
    back = ballistic.gcrs_to_itrf(g.position, g.velocity, epoch)
    assert np.linalg.norm(np.asarray(back.position) - pos) < 1e-6
    assert np.linalg.norm(np.asarray(back.velocity) - vel) < 1e-9


@pytest.mark.parametrize("old,new", [("eci_to_ecef", "gcrs_to_itrf"),
                                     ("ecef_to_eci", "itrf_to_gcrs")])
def test_true_of_date_names_are_removed(old, new):
    with pytest.raises(ballistic.RemovedApiError, match=new):
        getattr(ballistic, old)
