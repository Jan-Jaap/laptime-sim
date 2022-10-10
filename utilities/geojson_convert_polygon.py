# import streamlit as st
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium

from shapely.geometry import Polygon

FILE_NAME = '202209022_Circuit_Meppen'
DIR_TRACK = './tracks/'
def main():
        
    df = pd.read_csv(f'{DIR_TRACK}{FILE_NAME}.csv')
    inner  = gpd.GeoSeries.from_xy(df[['inner_x']], df[['inner_y']], z=df[['inner_z']])
    outer  = gpd.GeoSeries.from_xy(df[['outer_x']], df[['outer_y']], z=df[['outer_z']])
           
    inner = gpd.GeoDataFrame(geometry=gpd.GeoSeries(Polygon([(p.x, p.y) for p in inner]), crs="EPSG:32632"))
    outer = gpd.GeoDataFrame(geometry=gpd.GeoSeries(Polygon([(p.x, p.y) for p in outer]), crs="EPSG:32632"))
    # outer = gpd.GeoDataFrame(geometry=outer)
    
    gdf = outer.overlay(inner, how='symmetric_difference')
        
    # st_data = st_folium(gdf.explore(style_kwds=dict(color="black")))


    if input('save file?  [y/n]') in ('y', 'Y'):
        print('saving file')
        utm_crs = gdf.estimate_utm_crs()
        gdf.to_crs(utm_crs).to_file(f'{DIR_TRACK}{FILE_NAME}.geojson', driver='GeoJSON')
    

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
