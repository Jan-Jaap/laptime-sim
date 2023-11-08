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

    tab1, tab2 = st.tabs(["Select track", "Display results"])
    with tab1:
        st.header("Race track display")

        path = st.radio("select directory", [PATH_TRACK_FILES, PATH_RESULTS_FILES])
        file_name = st_select_track(path)
        track_layout, best_line = file_operations.load_trackdata_from_file(file_name)

        if track_layout.crs is None:
            track_layout = track_layout.set_crs(
                st.number_input("set crs to", value=32631)
            )

        save_parquet_button, save_shp_button, *_ = st.columns(3, gap="small")
        file_name = os.path.splitext(file_name)[0]

        with save_parquet_button:
            if st.button("save parquet", use_container_width=True):
                geopandas.GeoDataFrame(geometry=track_layout).to_parquet(
                    file_name + ".parquet"
                )

        with save_shp_button:
            if st.button("save shape file", use_container_width=True):
                track_layout.to_file(f"{file_name}.shp")

        if st.toggle("Show divisions"):
            track_divisions = geodataframe_operations.get_divisions(track_layout)
            track_display = geodataframe_operations.merge_geometry(
                [track_layout, track_divisions]
            )
        else:
            track_display = track_layout

        map = track_display.explore(style_kwds=dict(color="black"))
        st_folium(map, use_container_width=True)

        with st.expander("GeoDataFrame"):
            st.write(f"{track_layout.crs=}")
            st.json(track_layout.geometry.to_json())
    with tab2:
        race_car = Car.from_file("./cars/Peugeot_205RFS.json")
        filename_results = "output_simulated.csv"

        track_session = TrackSession(
            track_layout=track_layout, car=race_car, line_pos=best_line
        )
        if track_session is None:
            st.error("No track selected for optimization")

        best_time = race_lap.sim(track_session=track_session)

        st.write(f"Track has {track_session.len} datapoints")
        st.write(f"{race_car.name} - Simulated laptime = {time_to_str(best_time)}")

        # def print_results(time, iter) -> None:
        #     st.write(f"Laptime = {time_to_str(time)}  (iteration:{iter})")

        def print_results(time, iter):
            placeholder_laptime.write(f"Laptime = {time_to_str(time)}  (iteration:{iter})")

        def save_results(df) -> None:
            geopandas.GeoDataFrame(geometry=track_session.track_layout).to_parquet(
                file_name + ".parquet"
            )
            st.write(
                f"{datetime.now().strftime('%H:%M:%S')}: results saved to {filename_results=}"
            )

        with st.expander("Raceline optimization", expanded=True):
            placeholder_laptime = st.empty()
            if st.toggle("Optimize raceline"):
                track_session = race_lap.optimize_laptime(
                    track_session, print_results, save_results
                )

        # except KeyboardInterrupt:
        #     print('Interrupted by CTRL+C, saving progress')
        file_operations.save_results(
            race_lap.sim(track_session=track_session, verbose=True), filename_results
        )
        # geopandas.GeoDataFrame(geometry=track_session.track_layout).to_parquet(file_name +'.parquet')

        st.write(f"final results saved to {filename_results=}")
        #     best_time = race_lap.sim(track_session=track_session)

        #     print(f'{race_car.name} - Simulated laptime = {time_to_str(best_time)}')
