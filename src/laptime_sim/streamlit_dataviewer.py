"""This module creates a streamlit app"""

import os
import numpy as np
import plotly.express as px
from streamlit_folium import st_folium
import streamlit as st

import file_operations
import geodataframe_operations
from car import Car, DriverExperience

import race_lap
from tracksession import TrackSession

SUPPORTED_FILETYPES = (".csv", ".geojson", ".parquet")
PATH_TRACKS = ["./tracks/", "./simulated/"]
PATH_CARS = "./cars/"


def st_select_file(dir, extensions):
    files_in_dir = [f for f in sorted(os.listdir(dir)) if f.endswith(extensions)]
    return os.path.join(dir, st.radio(label="select track", options=files_in_dir))


def app():
    st.set_page_config(page_title="HSR Webracing", layout="wide")

    with st.sidebar:
        dir_selected = st.radio("select directory", PATH_TRACKS + [PATH_CARS])

    if dir_selected in PATH_TRACKS:

        st.header("Race track display")

        file_name = st_select_file(dir_selected, SUPPORTED_FILETYPES)
        track_layout = file_operations.load_trackdata_from_file(file_name)

        if track_layout is None:
            st.error("No track selected for optimization")

        if track_layout.crs is None:
            track_layout = track_layout.set_crs(
                st.number_input("CRS not in file. Set track crs to", value=32631)
            )

        file_name = os.path.splitext(file_name)[0]

        track_display = track_layout

        racecar = Car.from_toml("./cars/Peugeot_205RFS.toml")
        track_session = TrackSession(track_layout, racecar)
        results = race_lap.sim(track_session, verbose=True).to_csv().encode('utf-8')

        st_download_button = st.empty()
        if st_download_button.button("save results"):
            st_download_button.download_button(
                label='Download results',
                data=results,
                file_name='test.csv',
                mime='text/csv'
                )

        track_map = None
        if st.toggle("Show divisions"):
            divisions = geodataframe_operations.get_divisions(track_layout)
            track_map = divisions.explore(m=track_map, style_kwds=dict(color="grey"))

        # if any(idx := track_session.track_layout['type'] == 'line'):
        if track_session.has_line:
            track_map = track_session.line.explore(m=track_map, style_kwds=dict(color="blue"))

        if st.toggle("Show intersections"):
            try:
                intersections = geodataframe_operations.get_intersections(track_layout)
                track_map = intersections.explore(m=track_map, style_kwds=dict(color="red"))
            except IndexError:
                st.error('no line')

        with st.expander("GeoDataFrame"):
            st.write(track_display.to_dict())
            st.write(track_display.is_ring.rename('is_ring'))
            st.write(f"{track_layout.crs=}")

        idx = track_session.track_layout['type'].isin(['inner', 'outer'])
        track_map = (track_session
                     .track_layout
                     .geometry
                     .loc[idx]
                     .explore(m=track_map, style_kwds=dict(color="black"))
                     )
        st_folium(track_map, use_container_width=True)

    elif dir_selected in PATH_CARS:
        file_name = st_select_file(dir_selected, ('toml'))
        race_car = Car.from_toml(file_name)

        f = st.selectbox('Select driver experience', DriverExperience._member_names_, index=3)
        race_car.trail_braking = st.slider('Trail braking', min_value=30, max_value=100, value=DriverExperience[f])
        v = np.linspace(0, 250, 100)
        v1 = st.slider('Velocity in km/h', min_value=v.min(), max_value=v.max())

        col1, col2 = st.columns(2)
        fig = px.line(dict(
            v=v,
            acc=[race_car.get_max_acc(v0/3.6, 0)/9.81 for v0 in v],
            dec=[-race_car.get_min_acc(v0/3.6, 0)/9.81 for v0 in v],
            ), x=['acc', 'dec'], y='v')
        fig.add_vline(x=0, )
        fig.add_hline(y=v1)
        col1.plotly_chart(fig, use_container_width=True, )

        x = np.linspace(-race_car.acc_grip_max, race_car.acc_grip_max, 100)

        fig = px.line(dict(
            x=x,
            acc=[race_car.get_max_acc(v1/3.6, lat)/9.81 for lat in x],
            dec=[-race_car.get_min_acc(v1/3.6, lat)/9.81 for lat in x],
            ), x='x', y=['acc', 'dec'])
        fig.add_vline(x=0, )
        fig.add_hline(y=0)
        col2.plotly_chart(fig, use_container_width=True, )

        with st.expander('Car parameters'):
            st.write(race_car)


if __name__ == "__main__":
    app()
