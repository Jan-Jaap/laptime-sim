import os
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

def st_select_track(path=PATH_TRACK_FILES):
    
    tracks_in_dir = [s for s in sorted(os.listdir(path)) if s.endswith(SUPPORTED_FILETYPES)]
        
    if track_name := st.radio(label='select track', options=tracks_in_dir):
        return  os.path.join(path, track_name)
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
        track_layout, best_line = file_operations.load_trackdata_from_file(file_name)
        
        if track_layout.crs is None:
            track_layout = track_layout.set_crs(st.number_input('set crs to', value=32631))
        
        cols = st.columns(3, gap='small')
        file_name = os.path.splitext(file_name)[0]
        
        with cols[0]:
                
            if st.button('save parquet', use_container_width=True):
                geopandas.GeoDataFrame(geometry=track_layout.geometry, crs=track_layout.crs).to_parquet(file_name +'.parquet')
    
        with cols[1]:
            if st.button('save shape file', use_container_width=True):
                track_layout.to_file(f'{file_name}.shp')
    
        if st.toggle('Show divisions'):
            track_layout = geodataframe_operations.add_lines(track_layout)
        


        map = track_layout.explore(style_kwds=dict(color="black"))
        
        st_folium(map, use_container_width=True)

        with st.expander('GeoDataFrame'):
            st.write(f'{track_layout.crs=}')
            st.json(track_layout.geometry.to_json())
    with tab2:
        st.header('test')
        