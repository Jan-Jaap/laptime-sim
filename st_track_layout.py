import os
import streamlit as st
import geopandas as gpd
from streamlit_folium import st_folium
from track_sim.track import Track

SUPPORTED_FILETYPES = ('.csv', '.geojson')

def get_track_filename():
    match st.radio('select directory', options=['tracks', 'simulation results']):
        case 'tracks':
            dir = './tracks/'
        case  'simulation results':
            dir = './simulated/'
        
    files_in_dir = [s for s in sorted(os.listdir(dir)) if s.endswith(SUPPORTED_FILETYPES)]

    if filename_track := st.radio(label='select track', options=files_in_dir):
        return dir + filename_track
    return None


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
            track = Track.from_csv(file_name, crs=st.number_input('EPSG', value=32631, label_visibility='collapsed'))
            if tr:=track.track_record:
                st.write(f'Simulated laptime = {tr%3600//60:02.0f}:{tr%60:06.03f}')
            gdf = track.to_geojson()

        if file_name.endswith('.geojson'):
            gdf = gpd.read_file(file_name)


        st_folium(gdf.explore(style_kwds=dict(color="black")),  width = 725)
        
    with tab2:
        st.header('test')
        