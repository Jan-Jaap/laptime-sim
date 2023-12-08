"""This module creates a streamlit app"""

import os
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import streamlit as st

import geopandas as gpd
import geodataframe_operations
import car
import race_lap
from tracksession import TrackSession

# from icecream import ic

SUPPORTED_FILETYPES = (".csv", ".geojson", ".parquet")
PATH_TRACKS = "./tracks/"
PATH_LINES = "./simulated/"
PATH_CARS = "./cars/"

G = 9.81  # m/s²


def st_select_file(dir, extensions):
    files_in_dir = [f for f in sorted(os.listdir(dir)) if f.endswith(extensions)]
    return os.path.join(dir, st.radio(label="select file", options=files_in_dir))


def folium_track_map(track_layout, track_racelines, idx_selected_line):

    track_map = None
    divisions = geodataframe_operations.get_divisions(track_layout)
    track_map = divisions.explore(m=track_map, name='divisions', show=False, style_kwds=dict(color="grey"))

    track_map = track_layout.right.explore(m=track_map, name='right', control=False, style_kwds=dict(color="black"))
    track_map = track_layout.left.explore(m=track_map, name='left', control=False, style_kwds=dict(color="black"))

    for i in track_racelines.index:
        # only boolean indexing maintains correct type: GeoDataFrame
        race_line = track_racelines[track_racelines.index == i]
        name = race_line.car.values[0]

        if i == idx_selected_line:
            # race_line = geodataframe_operations.to_multipoints(race_line)
            style_kwds = dict(color="blue")
            name += ' [blue]'
        else:
            style_kwds = dict(color="grey", dashArray='2 5', opacity=0.6)
        track_map = race_line.explore(m=track_map, name=name, style_kwds=style_kwds)

    folium.LayerControl().add_to(track_map)
    return track_map


def app():
    st.set_page_config(page_title="HSR Webracing", layout="wide")

    tab1, tab2 = st.tabs(['Track definitions', 'Car definitions'])

    with tab1:

        st.header("Race track display")
        col1, col2 = st.columns(2)
        with col1:
            filename_track = st_select_file(PATH_TRACKS, 'parquet')
        if filename_track is None:
            st.error("No track selected")
            return

        track_layout = gpd.read_parquet(filename_track)
        track_name = track_layout.name[0]

        # this results in wrong crs for some paths (if not all crs are the same)
        # we correct for this by saving the crs in columns crs_backup
        track_racelines = gpd.read_parquet(PATH_LINES, filters=[('track', '==', track_name)])
        track_racelines.set_crs(track_racelines.pop('crs_backup')[0], inplace=True, allow_override=True)

        with col2:
            idx_selected_line = st.radio(
                label='raceline',
                options=track_racelines.index,
                format_func=lambda x: track_racelines.car[x]
                )

        race_line = track_racelines[track_racelines.index == idx_selected_line]
        race_car = car.Car.from_toml(f"./cars/{race_line.car.iloc[0]}.toml")
        track_session = TrackSession.from_layout(track_layout, race_line)

        sim_results = race_lap.simulate(race_car, track_session.line_coords(), track_session.slope, verbose=True)
        st.write(race_lap.time_to_str(sim_results.time[-1]))
        df = race_lap.results_dataframe(track_session, sim_results)

        st_download_button = st.empty()
        if st_download_button.button("save results"):
            st_download_button.download_button(
                label='Download results',
                data=df.to_csv().encode('utf-8'),
                file_name='test.csv',
                mime='text/csv'
                )
        track_map = folium_track_map(track_layout, track_racelines, idx_selected_line)
        st_folium(track_map, returned_objects=[], use_container_width=True)

        with st.expander("track_layout: GeoDataFrame"):
            st.write(track_layout.to_dict())
            st.write(track_layout.is_ring.rename('is_ring'))
            st.write(f"{track_layout.crs=}")

        with st.expander("results of simulation"):
            st.write()

    with tab2:
        file_name = st_select_file(PATH_CARS, ('toml'))
        race_car = car.Car.from_toml(file_name)

        f = st.selectbox('Trailbraking driver experience', car.Trailbraking._member_names_, index=3)
        race_car.trail_braking = st.slider('Trail braking', min_value=30, max_value=100, value=car.Trailbraking[f])

        f = st.selectbox('Select corner acceleration', car.CornerAcceleration._member_names_, index=3)
        race_car.corner_acc = st.slider(
            label='Corner acceleration',
            min_value=30,
            max_value=100,
            value=car.CornerAcceleration[f]
            )

        v = np.linspace(0, 250, 100)
        v1 = st.slider('Velocity in km/h', min_value=v.min(), max_value=v.max())

        col1, col2 = st.columns(2)
        fig = px.line(dict(
            v=v,
            acc=[race_lap.get_acceleration_wrapper(race_car, v=v0/3.6, acc_lat=0)/G for v0 in v],
            dec=[-race_lap.get_acceleration_wrapper(race_car, v=v0/3.6, acc_lat=0, braking=True)/G for v0 in v],
            ), x=['acc', 'dec'], y='v')
        fig.add_vline(x=0)
        fig.add_hline(y=v1)
        col1.plotly_chart(fig, use_container_width=True, )

        x = np.linspace(-race_car.acc_grip_max, race_car.acc_grip_max, 100)

        fig = px.line(dict(
            x=x/G,
            acc=[race_lap.get_acceleration_wrapper(race_car, v=v1/3.6, acc_lat=lat)/G for lat in x],
            dec=[-race_lap.get_acceleration_wrapper(race_car, v=v1/3.6, acc_lat=lat, braking=True)/G for lat in x],
            ), x='x', y=['acc', 'dec'])
        fig.add_vline(x=0)
        fig.add_vline(x=race_car.acc_grip_max/G)
        fig.add_vline(x=-race_car.acc_grip_max/G)
        fig.add_hline(y=0)
        fig.add_hline(y=0)
        col2.plotly_chart(fig, use_container_width=True, )

        with st.expander('Car parameters'):
            st.write(race_car)


if __name__ == "__main__":
    app()
