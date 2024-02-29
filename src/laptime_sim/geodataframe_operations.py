from shapely import MultiLineString, LinearRing, Point, LineString
from geopandas import GeoSeries, GeoDataFrame

import shapely
import geopandas
import pandas as pd

geopandas.options.io_engine = "pyogrio"

# from shapelysmooth import chaikin_smooth
# smoothed_geometry = chaikin_smooth(geometry, iters, keep_ends)


def df_to_geo(df, crs=None) -> geopandas.GeoSeries:

    border_right = df.filter(regex="inner_").values
    border_left = df.filter(regex="outer_").values
    race_line = df.filter(regex="line_").values

    geo = GeoSeries({
            'outer': LinearRing(border_left.tolist()),
            'inner': LinearRing(border_right.tolist())
            }, crs=crs)

    if race_line.size == 0:
        return geo

    return pd.concat([geo, GeoSeries([LinearRing(race_line.tolist())], ['line'], crs=crs)])


def drop_z(geom: shapely.LineString):
    return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))


def to_multipoints(df: GeoDataFrame):
    df.geometry = df.geometry.apply(lambda x: shapely.geometry.MultiPoint(list(x.coords)))
    return df


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
