"""This module creates a streamlit app"""

import os
from datetime import datetime

from streamlit_folium import st_folium
import streamlit as st
import geopandas

import file_operations
import geodataframe_operations
from car import Car
import race_lap
from track import TrackSession

geopandas.options.io_engine = "pyogrio"

SUPPORTED_FILETYPES = (".csv", ".geojson", ".parquet")
PATH_TRACK_FILES = "./tracks/"
PATH_RESULTS_FILES = "./simulated/"
PATH_CAR_FILES = "./cars/"


def st_select_track(path=PATH_TRACK_FILES):
    tracks_in_dir = [
        s for s in sorted(os.listdir(path)) if s.endswith(SUPPORTED_FILETYPES)
    ]
    if track_name := st.radio(label="select track", options=tracks_in_dir):
        return os.path.join(path, track_name)
    return None


def time_to_str(seconds: float) -> str:
    return "{:02.0f}:{:06.03f}".format(seconds % 3600 // 60, seconds % 60)


if __name__ == "__main__":
    st.set_page_config(page_title="HSR Webracing", layout="wide")

    if 'optimization_running' not in st.session_state:
        st.session_state['optimization_running'] = False

    tab1, tab2, tab3 = st.tabs(["Select Track", "Select Car", "Simulate and optimize"])
    with tab1:
        st.header("Race track display")

        path = st.radio("select directory", [PATH_TRACK_FILES, PATH_RESULTS_FILES])

        tracks_in_dir = [f for f in sorted(os.listdir(path)) if f.endswith(SUPPORTED_FILETYPES)]
        file_name = os.path.join(path, st.radio(label="select track", options=tracks_in_dir))

        track_layout, best_line = file_operations.load_trackdata_from_file(file_name)

        if track_layout is None:
            st.error("No track selected for optimization")

        if track_layout.crs is None:
            track_layout = track_layout.set_crs(
                st.number_input("set crs to", value=32631)
            )

        save_parquet_button, *_ = st.columns(3, gap="small")
        file_name = os.path.splitext(file_name)[0]

        with save_parquet_button:
            if st.button("save parquet", use_container_width=True):
                geopandas.GeoDataFrame(geometry=track_layout).to_parquet(file_name + ".parquet")

        track_display = track_layout

        if st.toggle("Show divisions"):
            track_display = geodataframe_operations.add_divisions(track_display)

        if st.toggle("Show intersections"):
            if 'line' not in track_display.index:
                st.error('Line not found in track')
            else:
                track_display = geodataframe_operations.add_intersections(track_display)

        with st.expander("GeoDataFrame"):
            st.write(track_display.to_dict())
            st.write(track_display.is_ring.rename('is_ring'))
            st.write(f"{track_layout.crs=}")

        map = track_display.explore(style_kwds=dict(color="black"))
        st_folium(map, use_container_width=True)

    with tab2:
        race_car = Car.from_toml("./cars/Peugeot_205RFS.toml")
        st.write(race_car)

    with tab3:

        dir_name = os.path.dirname(file_name)
        file_name = os.path.join(PATH_RESULTS_FILES, os.path.basename(file_name)+'.csv')

        track_session = TrackSession(
            track_layout=track_layout, car=race_car, line_pos=best_line
        )

        def intermediate_results(time, itereration, track_session):
            placeholder_laptime.write(f"Laptime = {time_to_str(time)}  (iteration:{itereration})")
            st.session_state['track_session'] = track_session

        def save_results(track_session) -> None:
            results = race_lap.sim(track_session, verbose=True)
            file_operations.save_results(results, file_name)
            st.write(
                f"{datetime.now().strftime('%H:%M:%S')}: results saved to {file_name=}"
            )
        if st.button('save track to results'):
            # results = race_lap.sim(track_session, verbose=True)
            save_results(track_session)

        with st.status("Raceline optimization", state='error', expanded=True) as status:
            placeholder_laptime = st.empty()
            if st.button("Start/Stop - Optimize raceline"):
                if not st.session_state.optimization_running:  # if not running start te optimization
                    st.session_state.optimization_running = True
                    status.update(state='running')
                    placeholder_laptime.write('optimization is started')

                    # this is a blocking function... no execution after this line, when optimizing...
                    st.session_state['track_session'] = race_lap.optimize_laptime(
                        track_session, intermediate_results, save_results)

                if st.session_state.optimization_running:  # if running stop te optimization
                    st.session_state.optimization_running = False
                    # status.update(state='running')
                    placeholder_laptime.write('optimization is stopped')
