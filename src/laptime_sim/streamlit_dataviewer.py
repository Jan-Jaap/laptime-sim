"""This module creates a streamlit app"""

import os
from streamlit_folium import st_folium
import streamlit as st
import geopandas

import file_operations
import geodataframe_operations
from car import Car

geopandas.options.io_engine = "pyogrio"

SUPPORTED_FILETYPES = (".csv", ".geojson", ".parquet")
PATH_TRACKS = ["./tracks/", "./simulated/"]
PATH_CARS = ["./cars/"]


def app():
    st.set_page_config(page_title="HSR Webracing", layout="wide")

    with st.sidebar:
        dir_selected = st.radio("select directory", PATH_TRACKS+PATH_CARS)

    if dir_selected in PATH_TRACKS:

        st.header("Race track display")

        tracks_in_dir = [f for f in sorted(os.listdir(dir_selected)) if f.endswith(SUPPORTED_FILETYPES)]
        file_name = os.path.join(dir_selected, st.radio(label="select track", options=tracks_in_dir))

        track_layout = file_operations.load_trackdata_from_file(file_name)

        if track_layout is None:
            st.error("No track selected for optimization")

        if track_layout.crs is None:
            track_layout = track_layout.set_crs(
                st.number_input("CRS not in file. Set track crs to", value=32631)
            )

        file_name = os.path.splitext(file_name)[0]

        if st.button("save parquet"):
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

    elif dir_selected in PATH_CARS:

        filename = os.path.join(dir_selected, "Peugeot_205RFS.toml")
        race_car = Car.from_toml(filename)
        st.write(race_car)


if __name__ == "__main__":
    app()
