
import geopandas
# import pandas as pd
import laptime_sim.file_operations as fo
from shapely.geometry import Polygon

PATH_TRACK_FILES = './tracks/'
SUPPORTED_FILETYPES = ('.csv', '.geojson', '.parquet')


def test_trackfiles_has_geometry():
    for filename in fo.filename_iterator(PATH_TRACK_FILES):
        track_layout = fo.load_trackdata_from_file(filename)
        assert isinstance(track_layout, geopandas.GeoDataFrame)


def test_inner_outer():
    for filename in fo.filename_iterator(PATH_TRACK_FILES):
        track_layout = geopandas.read_parquet(filename)
        right = Polygon(track_layout.right[0])
        left = Polygon(track_layout.left[0])
        assert left.contains_properly(right) or right.contains_properly(left)


def test_is_ring():
    for filename in fo.filename_iterator(PATH_TRACK_FILES):
        geo = fo.load_trackdata_from_file(filename)
        assert geo.left[0].is_ring
        assert geo.right[0].is_ring
