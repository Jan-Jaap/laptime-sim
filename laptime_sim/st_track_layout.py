import os
import pandas as pd
import streamlit as st
import geopandas
geopandas.options.io_engine = "pyogrio"

from streamlit_folium import st_folium
import file_operations as file_operations
import geodataframe_operations

SUPPORTED_FILETYPES = ('.csv', '.geojson', '.parquet')
PATH_TRACK_FILES = './tracks/'
PATH_RESULTS_FILES = './simulated/'
PATH_CAR_FILES = './cars/'

def st_select_track(path):
    
    tracks_in_dir = [s for s in sorted(os.listdir(path)) if s.endswith(SUPPORTED_FILETYPES)]
        
    if track_name := st.radio(label='select track', options=tracks_in_dir):
        return path + track_name
    return None


def load_track(file_path) -> geopandas.GeoDataFrame:
    
    match file_path:
        case None:
            st.stop()
        case s if s.endswith('.csv'):
            crs = st.number_input('EPSG', value=32631, label_visibility='collapsed')
            return file_operations.gdf_from_df(pd.read_csv(s), crs)
        case s if s.endswith('.geojson'):
            return geopandas.read_file(file_path)
        case s if  s.endswith('.parquet'):
            return geopandas.read_parquet(file_path)
    return None

 
if __name__ == '__main__':
    st.set_page_config(
    page_title='HSR Webracing',
    layout='wide')

    tab1, tab2 = st.tabs(['Select track','Display results'])\
    
    with tab1:
        st.header('Race track display')
        
        path = st.radio('select directory', [PATH_TRACK_FILES, PATH_RESULTS_FILES])
        file_name = st_select_track(path)
        track = load_track(file_name)
        
        cols = st.columns(3, gap='small')
        file_name = os.path.splitext(file_name)[0]
        
        with cols[0]:
                
            if st.button('save parquet', use_container_width=True):
                track.to_parquet(file_name +'.parquet')
    
        with cols[1]:
            if st.button('save shape file', use_container_width=True):
                track.to_file(f'{file_name}.shp')
    
        track = geodataframe_operations.add_lines(track)
        
        st_folium(track.explore(style_kwds=dict(color="black")), use_container_width=True)

        with st.expander('GeoDataFrame'):
            st.write(track)
    with tab2:
        st.header('test')
        