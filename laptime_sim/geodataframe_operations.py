import shapely
import geopandas
import pandas as pd
import numpy as np

geopandas.options.io_engine = "pyogrio"
from shapely import LineString, LinearRing

from geopandas import GeoSeries

# from shapelysmooth import chaikin_smooth
# smoothed_geometry = chaikin_smooth(geometry, iters, keep_ends)

def df_to_geo(df, crs=None):

    border_right = df.filter(regex="inner_").values
    border_left = df.filter(regex="outer_").values
    race_line = df.filter(regex="line_").values
   
    geo = GeoSeries({
    'outer' : LineString(border_left.tolist()),
    'inner' : LineString(border_right.tolist())
    }, crs=crs)

    if not race_line.size == 0:
        geo = pd.concat([geo, 
                         GeoSeries([LineString(race_line.tolist())],['line'], crs=crs)
                         ])

    return geo.to_crs(geo.estimate_utm_crs())

def drop_z(geo: GeoSeries) -> GeoSeries:
    for i, geom in enumerate(geo.geometry):
        geo.geometry[i] = shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return geo
   

def add_lines(geo: GeoSeries) -> GeoSeries:

    
    border_left = geo.geometry.loc[['outer']].get_coordinates(include_z=True).to_numpy(na_value=0)
    border_right = geo.geometry.loc[['inner']].get_coordinates(include_z=True).to_numpy(na_value=0)
    lines=[]
    for i, point_left in enumerate(border_left):
        point_right = border_right[i]
        lines.append(([(point_left), (point_right)   ] ))
    lines = shapely.MultiLineString(lines=lines)
    lines = GeoSeries(lines, index=['divisions'], crs=geo.crs)

    return pd.concat([geo.geometry, lines.geometry])
