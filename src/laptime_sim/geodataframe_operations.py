from shapely import MultiLineString
from geopandas import GeoSeries, GeoDataFrame

import shapely
import geopandas

geopandas.options.io_engine = "pyogrio"

# from shapelysmooth import chaikin_smooth
# smoothed_geometry = chaikin_smooth(geometry, iters, keep_ends)


def drop_z(geom: shapely.LineString):
    return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))


def divisions(geo) -> shapely.MultiLineString:
    '''return track GeoSeries with divisions added'''
    border_left = geo.left.get_coordinates(include_z=False).to_numpy(na_value=0)
    border_right = geo.right.get_coordinates(include_z=False).to_numpy(na_value=0)

    lines = []
    for point_left, point_right in zip(border_left, border_right):
        lines.append(([(point_left), (point_right)]))
    # if geo['outer'].is_ring:
    #     lines = lines[1:]  # first entry is double for rings.
    return MultiLineString(lines=lines)


def get_divisions(geo: GeoDataFrame) -> GeoSeries:
    return GeoSeries(divisions(geo), index=['divisions'], crs=geo.crs)


def get_intersections(geo: GeoDataFrame) -> GeoSeries:
    line = geo.line.geometry.values[0]
    intersection = shapely.intersection_all([divisions(geo),  drop_z(line)])
    return GeoSeries(intersection, index=['intersections'], crs=geo.crs)
