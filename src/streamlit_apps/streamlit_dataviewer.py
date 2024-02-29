"""This module creates a streamlit app"""

import streamlit as st
import os
import numpy as np
import plotly.express as px

import folium
from streamlit_folium import st_folium

import geopandas as gpd
from laptime_sim import geodataframe_operations

from laptime_sim.car import Car, Trailbraking, CornerAcceleration
from laptime_sim import race_lap
from laptime_sim.tracksession import TrackSession
from laptime_sim import file_operations as fo

PATH_TRACKS = './tracks/'


# from st_graphs.st_racetrack_display import st_racetrack_display
# from st_graphs.st_car_ggv_display import st_car_ggv_display

PATH_TRACKS = "./tracks/"
PATH_LINES = "./simulated/"
PATH_CARS = "./cars/"


G = 9.81  # m/s²


def plot_car_lon(race_car, v1):
    v = np.linspace(0, 300, 100)
    fig = px.line(dict(
        v=v,
        acc=[race_lap.get_max_acceleration(race_car, v=v0/3.6, acc_lat=0)/G for v0 in v],
        dec=[-race_lap.get_max_deceleration(race_car, v=v0/3.6, acc_lat=0)/G for v0 in v],
        ), x=['acc', 'dec'], y='v')
    fig.add_vline(x=0)
    fig.add_hline(y=v1)
    return fig


def plot_car_lat(race_car, v1):

    x = np.linspace(-race_car.acc_grip_max, race_car.acc_grip_max, 100)

    fig = px.line(dict(
        x=x/G,
        acc=[race_lap.get_max_acceleration(race_car, v=v1/3.6, acc_lat=lat)/G for lat in x],
        dec=[-race_lap.get_max_deceleration(race_car, v=v1/3.6, acc_lat=lat)/G for lat in x],
        ), x='x', y=['acc', 'dec'])
    fig.add_vline(x=0)
    fig.add_vline(x=race_car.acc_grip_max/G)
    fig.add_vline(x=-race_car.acc_grip_max/G)
    fig.add_hline(y=0)
    fig.add_hline(y=0)
    return fig


def filename_from_name(track_name, path_tracks=PATH_TRACKS):
    for filename in fo.filename_iterator(path_tracks, ('parquet')):
        if track_name in filename:
            return filename


def st_select_track_layout(path_tracks):
    track_names = gpd.read_parquet(path_tracks).name.to_list()
    track_selected = st.selectbox(label="select track", options=track_names)

    for filename_track in fo.filename_iterator(path_tracks, ('parquet')):
        if track_selected in filename_track:
            return filename_track


def st_select_raceline(path_racelines, track_name):
    track_racelines = gpd.read_parquet(path_racelines, filters=[('track', '==', track_name)])
    if track_racelines.empty:
        return track_racelines, None
    track_racelines.set_crs(track_racelines.pop('crs_backup')[0], inplace=True, allow_override=True)

    idx_selected_line = st.selectbox(
        label="Select car to show raceline",
        options=track_racelines.index,
        format_func=lambda x: f"{track_racelines.car[x]} - {track_name}"
        )
    return track_racelines, idx_selected_line


def st_racetrack_display(track_layout, track_racelines, idx_selected_line):

    st_trackmap = st.empty()

    if not track_racelines.empty:

        race_line = track_racelines[track_racelines.index == idx_selected_line]
        race_car_name = race_line.car.iloc[0]
        race_car = Car.from_toml(f"./cars/{race_car_name}.toml")
        track_session = TrackSession.from_layout(track_layout, race_line)

        sim_results = race_lap.simulate(race_car, track_session.line_coords(), track_session.slope, verbose=True)
        st.subheader(f'Simulated laptime for {race_car_name}: {race_lap.time_to_str(sim_results.time[-1])}')
        df = race_lap.results_dataframe(track_session, sim_results)
        st_trackmap = st.empty()

        with st.expander("results of simulation"):
            st.write(df)
            st_download_button = st.empty()
            if st_download_button.button("generate results csv"):
                st_download_button.download_button(
                    label='Download results csv',
                    data=df.to_csv().encode('utf-8'),
                    file_name='test.csv',
                    mime='text/csv'
                    )

        with st.expander('curvature'):
            st.write(sim_results.Nk)

    track_map = folium_track_map(track_layout, track_racelines, idx_selected_line)

    with st_trackmap:
        st_folium(track_map, returned_objects=[], use_container_width=True)

    with st.expander("track_layout: GeoDataFrame"):
        st.write(track_layout.to_dict())
        st.write(track_layout.is_ring.rename('is_ring'))
        st.write(f"{track_layout.crs=}")


def folium_track_map(track_layout, track_racelines, idx_selected_line):

    track_map = None
    divisions = geodataframe_operations.get_divisions(track_layout)
    track_map = divisions.explore(m=track_map, name='divisions', show=False, style_kwds=dict(color="grey"))

    track_map = track_layout.right.explore(m=track_map, name='right', control=False, style_kwds=dict(color="black"))
    track_map = track_layout.left.explore(m=track_map, name='left', control=False, style_kwds=dict(color="black"))

    for i in track_racelines.index:
        # only boolean indexing maintains correct type: GeoDataFrame
        race_line = track_racelines[track_racelines.index == i]
        # race_line = geodataframe_operations.to_multipoints(race_line)

        name = race_line.car.values[0]

        if i == idx_selected_line:
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
        header = st.empty()

        filename_track = st_select_track_layout(PATH_TRACKS)
        if filename_track is None:
            return st.error("No track selected")

        track_layout = gpd.read_parquet(filename_track)
        track_name = track_layout.name[0]
        track_racelines, idx_selected_line = st_select_raceline(PATH_LINES, track_name)

        header.header(f"Race track display - {track_name}")
        st_racetrack_display(track_layout, track_racelines, idx_selected_line)

    with tab2:
        files_in_dir = [f for f in sorted(os.listdir(PATH_CARS)) if f.endswith('toml')]
        filename_car = st.radio(label="select file", options=files_in_dir)
        race_car = Car.from_toml(os.path.join(PATH_CARS, filename_car))

        f = st.selectbox('Trailbraking driver experience', Trailbraking._member_names_, index=3)
        race_car.trail_braking = Trailbraking[f]

        f = st.selectbox('Select corner acceleration', CornerAcceleration._member_names_, index=3)
        race_car.corner_acc = CornerAcceleration[f]

        v1 = st.slider('Velocity in km/h', min_value=0, max_value=300)

        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_car_lon(race_car, v1), use_container_width=True, )
        col2.plotly_chart(plot_car_lat(race_car, v1), use_container_width=True, )

        with st.expander('Car parameters'):
            st.write(race_car)


if __name__ == "__main__":
    app()
