import shapely
import geopandas
import pandas as pd
import numpy as np

geopandas.options.io_engine = "pyogrio"
from shapely import LineString, LinearRing

from geopandas import GeoDataFrame, GeoSeries

# from shapelysmooth import chaikin_smooth
# smoothed_geometry = chaikin_smooth(geometry, iters, keep_ends)



def gdf_from_df(df, crs=None):

    border_right = df.filter(regex="inner_").values
    border_left = df.filter(regex="outer_").values
    race_line = df.filter(regex="line_").values
   
    track_dict = {'geometry':{
    'outer' : LineString(border_left.tolist()),
    'inner' : LineString(border_right.tolist())
    }}
                  

    if not race_line.size == 0:
        track_dict['geometry']['line'] = LineString(race_line.tolist())
    
    # gdf = GeoDataFrame(
    #     dict(
    #         name=list(track_dict.keys()),
    #         geometry=list(track_dict.values())
    #         ), 
    #     crs=crs)
    gdf = GeoDataFrame.from_dict(track_dict, crs=crs)
    
    
    crs = gdf.estimate_utm_crs()
    # convert from arbitrary crs to UTM (unit=m)
    return gdf.to_crs(crs)

def drop_z(gdf: GeoDataFrame) -> GeoDataFrame:
    for i, geom in enumerate(gdf.geometry):
        gdf.geometry[i] = shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return gdf
   

def add_lines(gdf: GeoDataFrame) -> GeoDataFrame:

    
    border_left = gdf.geometry.loc[['outer']].get_coordinates(include_z=True).to_numpy(na_value=0)
    border_right = gdf.geometry.loc[['inner']].get_coordinates(include_z=True).to_numpy(na_value=0)
    lines=[]
    for i, point_left in enumerate(border_left):
        point_right = border_right[i]
        lines.append(([(point_left), (point_right)   ] ))
    lines = shapely.MultiLineString(lines=lines)
    lines = GeoSeries(lines, index=['divisions'], crs=gdf.crs)
    lines = GeoDataFrame(geometry=lines)
    
    joined = pd.concat([gdf, lines])
    return joined
