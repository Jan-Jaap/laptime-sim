"""This module creates a streamlit app"""

import os
import numpy as np
import plotly.express as px
from streamlit_folium import st_folium
import streamlit as st

import geopandas as gpd
import geodataframe_operations
from car import Car, DriverExperience

from tracksession import TrackSession

from icecream import ic

SUPPORTED_FILETYPES = (".csv", ".geojson", ".parquet")
PATH_TRACKS = "./tracks/"
PATH_LINES = "./simulated/"
PATH_CARS = "./cars/"


def st_select_file(dir, extensions):
    files_in_dir = [f for f in sorted(os.listdir(dir)) if f.endswith(extensions)]
    return os.path.join(dir, st.radio(label="select file", options=files_in_dir))


def app():
    st.set_page_config(page_title="HSR Webracing", layout="wide")

    tab1, tab2 = st.tabs(['Track definitions', 'Car definitions'])

    with tab1:

        st.header("Race track display")

        filename_track = st_select_file(PATH_TRACKS, 'parquet')
        if filename_track is None:
            st.error("No track selected")
            return

        track_layout = gpd.read_parquet(filename_track)
        track_name = track_layout.name[0]

        # this results in wrong crs for some paths
        track_racelines = gpd.read_parquet(PATH_LINES, filters=[('track', '==', track_name)])
        # correct crs
        track_racelines.set_crs(track_racelines.crs_backup[0], inplace=True, allow_override=True)

        with st.sidebar:
            idx_selected_line = st.radio(
                label='raceline',
                options=track_racelines.index,
                format_func=lambda x: track_racelines.car[x]
                )
            selected_line = track_racelines[track_racelines.index == idx_selected_line]
        # if raceline_idx is None:
        #     track_raceline = None
        # else:
        #     track_raceline = track_raceline[track_raceline.index == raceline_idx]

        # racecar = Car.from_toml("./cars/Peugeot_205RFS.toml")
        # results = race_lap.sim(track_session, verbose=True).to_csv().encode('utf-8')

        # st_download_button = st.empty()
        # if st_download_button.button("save results"):
        #     st_download_button.download_button(
        #         label='Download results',
        #         data=results,
        #         file_name='test.csv',
        #         mime='text/csv'
        #         )
        track_map = None
        if st.toggle("Show divisions"):
            divisions = geodataframe_operations.get_divisions(track_layout)
            track_map = divisions.explore(m=track_map, style_kwds=dict(color="grey"))

        track_map = track_layout.inner.explore(m=track_map, style_kwds=dict(color="blue"))
        track_map = track_layout.outer.explore(m=track_map, style_kwds=dict(color="blue"))

        if len(track_racelines.index) > 0:
            track_map = track_racelines.explore(m=track_map, style_kwds=dict(color="grey"))
            track_map = selected_line.explore(m=track_map, style_kwds=dict(color="black", dashArray='1 4'))

        st_folium(track_map, use_container_width=True, returned_objects=[])

        with st.expander("GeoDataFrame"):
            st.write(track_layout.to_dict())
            st.write(track_layout.is_ring.rename('is_ring'))
            st.write(f"{track_layout.crs=}")

        with st.expander('TrackSession object description'):
            st.write(TrackSession(track_layout.outer, track_layout.inner))

    with tab2:
        file_name = st_select_file(PATH_CARS, ('toml'))
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
