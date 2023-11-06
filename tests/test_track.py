
import geopandas
from laptime_sim.file_operations import filename_iterator, load_trackdata_from_file

from icecream import ic

# from shapely import Polygon, LineString
from shapely.geometry import mapping, Polygon

PATH_TRACK_FILES = './tracks/'
SUPPORTED_FILETYPES = ('.csv', '.geojson', '.parquet')


def test_trackfiles_have_geometry():
    for filename in filename_iterator():
        track_layout, _ = load_trackdata_from_file(filename)
        assert isinstance(track_layout, geopandas.GeoSeries)


def test_inner_outer():
    for filename in filename_iterator():
        geo, _ = load_trackdata_from_file(filename)

        inner = Polygon(geo['inner'])
        outer = Polygon(geo['outer'])
        assert outer.contains(inner)
        assert not inner.contains(outer)

