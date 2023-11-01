# import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, LinearRing

FILE_NAME   = './tracks/20191220_Spa_Francorchamp'
input_CRS         = 32631

def main():
        
    df = pd.read_csv(f'{FILE_NAME}.csv')

    track_dict = dict(
    inner = LinearRing(df.filter(regex='inner_').values[:,:2].tolist()),
    outer = LinearRing(df.filter(regex='outer_').values[:,:2].tolist())
    )


    if any(df.columns.str.startswith('line_')):
        print('adding raceline')
        track_dict['line'] = LinearRing(df.filter(regex='line_').values.tolist())
    
    track = gpd.GeoDataFrame(
        dict(
            name=list(track_dict.keys()),
            geometry=list(track_dict.values())
            ), 
        crs=input_CRS)

    local_utm_crs = track.estimate_utm_crs()
    track.to_crs(local_utm_crs).to_file(f'{FILE_NAME}.geojson', driver='GeoJSON')
    

if __name__ == '__main__':
    main()

