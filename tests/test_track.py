
import geopandas
import laptime_sim.file_operations as fo
from shapely.geometry import Polygon

PATH_TRACK_FILES = './tracks/'
SUPPORTED_FILETYPES = ('.csv', '.geojson', '.parquet')


def test_trackfiles_have_geometry():
    for filename in fo.filename_iterator(PATH_TRACK_FILES):
        track_layout = fo.load_trackdata_from_file(filename)
        assert isinstance(track_layout, geopandas.GeoSeries)


def test_inner_outer():
    for filename in fo.filename_iterator(PATH_TRACK_FILES):
        track_layout = fo.load_trackdata_from_file(filename)

        inner = Polygon(track_layout['inner'])
        outer = Polygon(track_layout['outer'])
        assert outer.contains_properly(inner)
        assert not inner.contains_properly(outer)


def test_is_ring():
    for filename in fo.filename_iterator(PATH_TRACK_FILES):
        geo, _ = fo.load_trackdata_from_file(filename)
        assert geo.is_ring
