from shapely import LineString, LinearRing, MultiLineString
from geopandas import GeoSeries

import shapely
import geopandas
import pandas as pd
# import numpy as np

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


def drop_z(geo: GeoSeries) -> GeoSeries:
    for i, geom in enumerate(geo.geometry):
        geo.geometry[i] = shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geo


def get_divisions(geo: GeoSeries) -> GeoSeries:

    # border_left = geo.geometry.loc[['outer']].get_coordinates(include_z=True).to_numpy(na_value=0)
    border_left = geo[['outer']].get_coordinates(include_z=True).to_numpy(na_value=0)
    border_right = geo[['inner']].get_coordinates(include_z=True).to_numpy(na_value=0)

    lines = []
    for point_left, point_right in zip(border_left, border_right):
        lines.append(([(point_left), (point_right)]))

    lines = MultiLineString(lines=lines)

    points = geo['line'].intersection(lines)

    lines = GeoSeries(lines, index=['divisions'], crs=geo.crs)
    points = GeoSeries(points, index=['intersections'], crs=geo.crs)

    return merge_geometry([lines.geometry, points.geometry])


def merge_geometry(list):
    return pd.concat(list)
