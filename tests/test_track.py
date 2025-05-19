import pytest
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import laptime_sim
from main import TRACK_LIST

track_list = laptime_sim.track_list(TRACK_LIST)


@pytest.mark.parametrize("track", track_list, ids=lambda x: x.name)
class TestTrack:
    def test_track_properties(self, track: laptime_sim.Track):
        assert isinstance(track, laptime_sim.Track)
        assert isinstance(track.get_width, np.ndarray)
        assert isinstance(track.slope, np.ndarray)
        assert isinstance(track.left, GeoSeries)
        assert isinstance(track.right, GeoSeries)
        assert isinstance(track.is_circular, np.bool)
        assert isinstance(track.name, str)
        assert isinstance(track.layout, GeoDataFrame)
        assert isinstance(track.start_finish, GeoSeries)
        assert isinstance(track.divisions, GeoSeries)

    def test_track_layout(self, track: laptime_sim.Track):
        assert len(track.left) == 1
        assert len(track.right) == 1

    def test_circular_track(self, track: laptime_sim.Track):
        assert track.layout.is_closed.all()

    # def test_track_position(self, track: laptime_sim.Track):
    #     coords = track.coordinates_from_pos([0.5])
    #     pos = track.pos_from_coordinates(coords)
    #     np.testing.assert_almost_equal(coords, track.coordinates_from_pos(pos))
