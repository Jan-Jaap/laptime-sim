
import os 
from laptime_sim.file_operations import track_from_parquet
from laptime_sim.track import TrackSession

from icecream import ic

# from shapely import Polygon, LineString
from shapely.geometry import mapping, Polygon

filename = "./tracks/20191030_Circuit_Zandvoort.parquet"
test_track =  track_from_parquet(filename)
gdf = test_track.track_layout


def test_if_track_file_exists():
    assert os.path.isfile(filename)


def test_track_creation():
    assert TrackSession(track_layout=gdf)


def test_inner_outer():
    gdf['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gdf.geometry]
    inner = gdf.geometry.loc['inner']
    outer = gdf.geometry.loc['outer']
    
    assert outer.contains(inner)
    assert not inner.contains(outer)

