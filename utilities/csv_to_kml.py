# import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import fiona

fiona.supported_drivers['KML'] = 'rw'

FILE_NAME   = './simulated/Peugeot_205RFS_20191211_Bilsterberg_simulated'
CRS         = 32632

def main():
        
    df = pd.read_csv(f'{FILE_NAME}.csv')

    inner = LineString(df.filter(regex='inner_').values.tolist())
    outer = LineString(df.filter(regex='outer_').values.tolist())

    if any(df.columns.str.startswith('line_')):
        print('adding raceline')
        line = LineString(df.filter(regex='line_').values.tolist())
        d = {'name':['inner','outer','line'], 'geometry':[inner, outer, line]}
    else:
        d = {'name':['inner','outer'], 'geometry':[inner, outer]}
    
    track = gpd.GeoDataFrame(d, crs=CRS)

    utm_crs = track.estimate_utm_crs()
    track.to_crs(utm_crs).to_file(f'{FILE_NAME}.kml', driver='KML')
    

if __name__ == '__main__':
    main()

# border_left         = Feature(geometry=Polygon([data.filter(regex="outer_").values.tolist()]))
# border_right        = Feature(geometry=Polygon([data.filter(regex="inner_").values.tolist()]))


# crs = {
#     "type": "name",
#     "properties": {
#         "name": "EPSG:32631"
#     }
# }

# geo = FeatureCollection( [border_left, border_right], name="zandvoort_coded", crs=crs)
 

# with open('./tracks/20191030_Circuit_Zandvoort.geojson', 'w') as f:
#     dump(geo, fp=f, indent=2)
