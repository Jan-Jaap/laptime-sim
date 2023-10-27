import shapely
import geopandas

def gdf_from_df(df, crs=None):

    border_left = df.filter(regex="outer_").values
    border_right = df.filter(regex="inner_").values
    race_line = df.filter(regex="line_").values
   
    track_dict = dict(
    inner = shapely.LinearRing(border_left.tolist()),
    outer = shapely.LinearRing(border_right.tolist())
    )

    if not race_line.size == 0:
        track_dict['line'] = shapely.LinearRing(race_line.tolist())
    
    gdf = geopandas.GeoDataFrame(
        dict(
            name=list(track_dict.keys()),
            geometry=list(track_dict.values())
            ), 
        crs=crs)
    
    crs = gdf.estimate_utm_crs()
    # convert from arbitrary crs to UTM (unit=m)
    return gdf.to_crs(crs)

def drop_z(data: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    for i, geom in enumerate(data.geometry):
        data.geometry[i] = shapely.wkb.loads(shapely.wkb.dumps(geom, output_dimension=2))
    return data
