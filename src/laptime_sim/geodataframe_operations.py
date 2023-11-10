from shapely import MultiLineString, LineString
from geopandas import GeoSeries

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
            'outer': LineString(border_left.tolist()),
            'inner': LineString(border_right.tolist())
            }, crs=crs)

    if race_line.size == 0:
        return geo

    return pd.concat([geo, GeoSeries([LineString(race_line.tolist())], ['line'], crs=crs)])


def drop_z(geom: GeoSeries) -> GeoSeries:
    return shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))


def get_divisions(geo: GeoSeries) -> shapely.MultiLineString:
    '''return track GeoSeries with divisions added'''
    border_left = geo[['outer']].get_coordinates(include_z=False).to_numpy(na_value=0)
    border_right = geo[['inner']].get_coordinates(include_z=False).to_numpy(na_value=0)

    lines = []
    for point_left, point_right in zip(border_left, border_right):
        lines.append(([(point_left), (point_right)]))
    return MultiLineString(lines=lines)


def add_divisions(geo: GeoSeries) -> GeoSeries:
    divisions = get_divisions(geo)
    return pd.concat([geo, GeoSeries(divisions, index=['divisions'], crs=geo.crs)])


def get_intersections(geo: GeoSeries) -> shapely.MultiPoint:
    return shapely.intersection_all([get_divisions(geo),  drop_z(geo['line'])])


def add_intersections(geo: GeoSeries) -> GeoSeries:
    intersection = get_intersections(geo)
    return pd.concat([geo, GeoSeries(intersection, index=['intersections'], crs=geo.crs)])
