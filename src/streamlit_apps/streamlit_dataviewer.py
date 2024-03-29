"""This module creates a streamlit app"""

import streamlit as st
import os
import numpy as np
import plotly.express as px

import folium
from streamlit_folium import st_folium
import geopandas as gpd

from laptime_sim import geodataframe_operations
from laptime_sim.car import Trailbraking, CornerAcceleration
from laptime_sim import file_operations as fo
from laptime_sim import race_lap, Car, Track

PATH_TRACKS = "./tracks/"
PATH_LINES = "./simulated/"
PATH_CARS = "./cars/"

G = 9.81  # m/s²


def plot_car_lon(race_car, v1):
    v = np.linspace(0, 300, 100)
    fig = px.line(
        dict(
            v=v,
            acc=[
                race_lap.get_max_acceleration(race_car, v=v0 / 3.6, acc_lat=0) / G
                for v0 in v
            ],
            dec=[
                -race_lap.get_max_deceleration(race_car, v=v0 / 3.6, acc_lat=0) / G
                for v0 in v
            ],
        ),
        x=["acc", "dec"],
        y="v",
    )
    fig.add_vline(x=0)
    fig.add_hline(y=v1)
    return fig


def plot_car_lat(race_car, v1):

    x = np.linspace(-race_car.acc_grip_max, race_car.acc_grip_max, 100)

    fig = px.line(
        dict(
            x=x / G,
            acc=[
                race_lap.get_max_acceleration(race_car, v=v1 / 3.6, acc_lat=lat) / G
                for lat in x
            ],
            dec=[
                -race_lap.get_max_deceleration(race_car, v=v1 / 3.6, acc_lat=lat) / G
                for lat in x
            ],
        ),
        x="x",
        y=["acc", "dec"],
    )
    fig.add_vline(x=0)
    fig.add_vline(x=race_car.acc_grip_max / G)
    fig.add_vline(x=-race_car.acc_grip_max / G)
    fig.add_hline(y=0)
    fig.add_hline(y=0)
    return fig


def st_get_car(path=PATH_CARS):
    files_in_dir = [f for f in sorted(os.listdir(path)) if f.endswith("toml")]
    filename_car = st.selectbox(
        label="Select car to show raceline",
        options=files_in_dir,
        format_func=lambda x: f"{fo.strip_filename(x)}",
    )
    race_car = Car.from_toml(os.path.join(path, filename_car))
    return fo.strip_filename(filename_car), race_car


def filename_from_name(track_name, path_tracks=PATH_TRACKS):
    for filename in fo.filename_iterator(path_tracks, ("parquet")):
        if track_name in filename:
            return filename


def st_select_track_layout(path_tracks):
    track_names = gpd.read_parquet(path_tracks).name.to_list()
    track_selected = st.selectbox(label="select track", options=track_names)
    return track_selected
    # for filename_track in fo.filename_iterator(path_tracks, ("parquet")):
    #     if track_selected in filename_track:
    #         return filename_track


def folium_track_map(track_layout, all_track_racelines, selected_line):

    track_map = None
    divisions = geodataframe_operations.get_divisions(track_layout)
    track_map = divisions.explore(
        m=track_map, name="divisions", show=False, style_kwds=dict(color="grey")
    )

    track_map = track_layout.right.explore(
        m=track_map, name="right", control=False, style_kwds=dict(color="black")
    )
    track_map = track_layout.left.explore(
        m=track_map, name="left", control=False, style_kwds=dict(color="black")
    )

    for i in all_track_racelines.index:
        # only boolean indexing maintains correct type: GeoDataFrame
        race_line = all_track_racelines[all_track_racelines.index == i]
        name = race_line.car.values[0]
        track_map = race_line.explore(
            m=track_map,
            name=name,
            style_kwds=dict(color="grey", dashArray="2 5", opacity=0.6),
        )
    track_map = selected_line.explore(
        m=track_map, name=selected_line.car.values[0], style_kwds=dict(color="blue")
    )

    folium.LayerControl().add_to(track_map)
    return track_map


def st_tab1():

    header = st.empty()

    track_name = st_select_track_layout(PATH_TRACKS)
    if track_name is None:
        return st.error("No track selected")
    track = Track.from_parquet(fo.find_track_filename(track_name))

    all_racelines_gdf = gpd.read_parquet(
        PATH_LINES, filters=[("track", "==", track.name)]
    )
    all_racelines_gdf.set_crs(
        all_racelines_gdf.pop("crs_backup")[0], inplace=True, allow_override=True
    )

    filename_car, race_car = st_get_car()
    filename_raceline = fo.find_raceline_filename(track.name, filename_car)
    raceline_gdf = gpd.read_parquet(filename_raceline)

    header.header(f"Race track display - {track.name}")

    st.write(f"Simulated laptime - {race_lap.time_to_str(raceline_gdf.best_time[0])}")

    track_map = folium_track_map(track.layout, all_racelines_gdf, raceline_gdf)
    st_folium(track_map, returned_objects=[], use_container_width=True)

    with st.expander("track_layout: GeoDataFrame"):
        st.write(track.layout.to_dict())
        st.write(track.layout.is_ring.rename("is_ring"))
        st.write(f"{track.layout.crs=}")


def st_tab2():
    files_in_dir = [f for f in sorted(os.listdir(PATH_CARS)) if f.endswith("toml")]
    filename_car = st.radio(label="select file", options=files_in_dir)
    race_car = Car.from_toml(os.path.join(PATH_CARS, filename_car))

    f = st.selectbox(
        "Trailbraking driver experience", Trailbraking._member_names_, index=3
    )
    race_car.trail_braking = Trailbraking[f]

    f = st.selectbox(
        "Select corner acceleration", CornerAcceleration._member_names_, index=3
    )
    race_car.corner_acc = CornerAcceleration[f]

    v1 = st.slider("Velocity in km/h", min_value=0, max_value=300)

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        plot_car_lon(race_car, v1),
        use_container_width=True,
    )
    col2.plotly_chart(
        plot_car_lat(race_car, v1),
        use_container_width=True,
    )

    with st.expander("Car parameters"):
        st.write(race_car)


def app():
    st.set_page_config(page_title="HSR Webracing", layout="wide")

    tab1, tab2 = st.tabs(["Track definitions", "Car definitions"])
    with tab1:
        st_tab1()
    with tab2:
        st_tab2()


if __name__ == "__main__":
    app()
