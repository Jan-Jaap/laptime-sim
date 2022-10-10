import os
# import pandas as pd
import streamlit as st
# import plotly.graph_objects as go
import geopandas as gpd
from streamlit_folium import st_folium
from track_sim.track import Track


def get_track_filename():
    match st.radio('select filetype', options=['csv tracks', 'geojson tracks', 'simulation results']):
        case 'csv tracks':
            dir = './tracks/'
            endswith = '.csv'
        case 'geojson tracks':
            dir = './tracks/'
            endswith = '.geojson'
        case  'simulation results':
            dir = './simulated/'
            endswith = '.csv'

    files_in_dir = [s for s in sorted(os.listdir(dir)) if s.endswith(endswith)]

    if filename_track := st.radio(label='select track', options=files_in_dir):
        return dir + filename_track
    return None

def display_track(track):
    
    if laptime:=track.track_record:
        st.write(f'Simulated laptime = {laptime%3600//60:02.0f}:{laptime%60:06.03f}')
    st.plotly_chart(track.figure(), use_container_width=True)

def display_track_geojson(gdf):
    st_data = st_folium(gdf.explore(style_kwds=dict(color="black")),  width = 725)
    with st.expander("Expand to see data returned to Python"):
        st_data

if __name__ == '__main__':
    st.set_page_config(
    page_title='HSR Webracing',
    layout='wide')

    tab1, tab2 = st.tabs(['Select track','Display results'])\
    
    with tab1:
        st.header('Race track display')
        
        file_name=get_track_filename()
            
        if file_name is None:
            st.stop()
        if file_name.endswith('.csv'):
            track = Track.from_csv(file_name)
            display_track(track)
        if file_name.endswith('.geojson'):
            df_track = gpd.read_file(file_name)
            display_track_geojson(df_track)
        
    with tab2:
        st.header('test')
        