"""This module creates a streamlit app"""

import streamlit as st

import folium
import xyzservices.providers as xyz
from streamlit_folium import st_folium
import geopandas as gpd

import laptime_sim
from matplotlib import pyplot as plt

from laptime_sim.raceline import Raceline

PATH_RESULTS = "./simulated/"
PATH_TRACKS = "./tracks/"
PATH_CARS = "./cars/"


def folium_track_map(track: laptime_sim.Track, all_track_racelines: gpd.GeoDataFrame, results):

    my_map = track.divisions.explore(name="divisions", show=False, style_kwds=dict(color="grey"))

    style_kwds = dict(color="black")
    track.layout.explore(m=my_map, name="border", control=False, style_kwds=style_kwds)
    gpd.GeoSeries(track.start_finish(), crs=track.crs).explore(
        m=my_map,
        name="start_finish",
        style_kwds=dict(
            color="red",
            weight=5,
        ),
    )

    if not all_track_racelines.empty:
        style_kwds = dict(color="black", dashArray="2 5", opacity=0.6)
        my_map = all_track_racelines.explore(m=my_map, name="results", control=False, style_kwds=style_kwds)

    if results is not None and not results.empty:
        results.explore(m=my_map, name=results.car.iloc[0], style_kwds=dict(color="blue"))

    folium.TileLayer(xyz.Esri.WorldImagery).add_to(my_map)
    folium.TileLayer("openstreetmap").add_to(my_map)
    folium.LayerControl().add_to(my_map)
    return my_map


def format_results(x):
    best_time = x["best_time"]
    return f"{x['car']} ({best_time % 3600 // 60:02.0f}:{best_time % 60:06.03f})"


def main() -> None:

    st.header("Race track display")

    with st.sidebar:
        track = st.radio("select track", options=laptime_sim.get_all_tracks(PATH_TRACKS), format_func=lambda x: x.name)

    all_racelines = gpd.read_parquet(PATH_RESULTS, filters=[("track_name", "==", track.name)])

    if all_racelines.empty:
        st.warning("No results found for this track")
        selected_raceline = None
    else:
        d = st.radio("select result", options=all_racelines.to_dict(orient="records"), format_func=format_results)
        selected_raceline = all_racelines.from_dict([d]).set_crs(epsg=4326)
        raceline = Raceline.from_geodataframe(selected_raceline, path_tracks=PATH_TRACKS, path_cars=PATH_CARS)

    track_map = folium_track_map(track, all_racelines, selected_raceline)
    st_folium(track_map, returned_objects=[], use_container_width=True)

    sim_results = raceline.simulate()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(211)
    ax.plot(sim_results.distance, sim_results.speed)

    st.pyplot(fig)
    with st.expander("Selected Raceline"):
        st.write(selected_raceline)
    with st.expander("SimResults"):
        st.write(sim_results)
    with st.expander("Race Car"):
        st.write(raceline.car)
    with st.expander("Race Track Layout"):
        st.write(track.layout)


if __name__ == "__main__":
    main()
