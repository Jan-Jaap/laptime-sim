import pytest

import laptime_sim.track as track
from laptime_sim.track import Track


def test_track_creation():
    assert Track(name=None, geodataframe=track.GeoDataFrame())

