"""This module creates a streamlit app"""

import streamlit as st
import os

import folium
import xyzservices.providers as xyz
from streamlit_folium import st_folium
import geopandas as gpd

import laptime_sim
import laptime_sim.file_operations as fo

PATH_TRACKS = "./tracks/"
PATH_LINES = "./simulated/"
PATH_CARS = "./cars/"


def st_get_car(path=PATH_CARS):
    files_in_dir = [f for f in sorted(os.listdir(path)) if f.endswith("toml")]
    filename_car = st.selectbox(
        label="Select car to show raceline",
        options=files_in_dir,
        format_func=lambda x: f"{fo.strip_filename(x)}",
    )
    race_car = laptime_sim.Car.from_toml(os.path.join(path, filename_car))
    return fo.strip_filename(filename_car), race_car


def st_select_track_layout(path_tracks):
    track_names = gpd.read_parquet(path_tracks).name.to_list()
    track_selected = st.selectbox(label="select track", options=track_names)
    return track_selected


def folium_track_map(track: laptime_sim.Track, all_track_racelines, selected_line):

    my_map = None
    my_map = track.divisions.explore(
        m=my_map, name="divisions", show=False, style_kwds=dict(color="grey")
    )
    my_map = track.right.explore(
        m=my_map, name="right", control=False, style_kwds=dict(color="black")
    )
    my_map = track.left.explore(
        m=my_map, name="left", control=False, style_kwds=dict(color="black")
    )

    for i in all_track_racelines.index:
        # only boolean indexing maintains correct type: GeoDataFrame
        race_line = all_track_racelines[all_track_racelines.index == i]
        my_map = race_line.explore(
            m=my_map,
            name=race_line.car.values[0],
            style_kwds=dict(color="grey", dashArray="2 5", opacity=0.6),
        )
    my_map = selected_line.explore(
        m=my_map, name=selected_line.car.values[0], style_kwds=dict(color="blue")
    )

    folium.TileLayer(xyz.Esri.WorldImagery).add_to(my_map)
    folium.TileLayer("openstreetmap").add_to(my_map)
    folium.LayerControl().add_to(my_map)

    return my_map


def main() -> None:

    st.header("Race track display")

    # st.write(xyz)
    track_name = st_select_track_layout(PATH_TRACKS)
    if track_name is None:
        return st.error("No track selected")
    track = laptime_sim.Track.from_parquet(fo.find_track_filename(track_name))

    all_racelines_gdf = gpd.read_parquet(
        PATH_LINES, filters=[("track", "==", track.name)]
    )
    all_racelines_gdf.set_crs(
        all_racelines_gdf.pop("crs_backup")[0], inplace=True, allow_override=True
    )

    filename_car, race_car = st_get_car()
    filename_raceline = fo.find_raceline_filename(track.name, filename_car)
    raceline_gdf = gpd.read_parquet(filename_raceline)

    st.write(
        f"Simulated laptime - {laptime_sim.time_to_str(raceline_gdf.best_time[0])}"
    )

    track_map = folium_track_map(track, all_racelines_gdf, raceline_gdf)
    st_folium(track_map, returned_objects=[], use_container_width=True)

    with st.expander("track_layout: GeoDataFrame"):
        st.write(track.layout.to_dict())
        st.write(track.layout.is_ring.rename("is_ring"))
        st.write(f"{track.layout.crs=}")


if __name__ == "__main__":
    main()
