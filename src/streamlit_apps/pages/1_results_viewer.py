"""This module creates a streamlit app"""

import streamlit as st

import folium
import xyzservices.providers as xyz
from streamlit_folium import st_folium
import geopandas as gpd

import laptime_sim
from laptime_sim.simulate import simulate
from matplotlib import pyplot as plt

PATH_LINES = "./simulated/"


def folium_track_map(track: laptime_sim.Track, all_track_racelines: gpd.GeoDataFrame, results):

    my_map = None
    my_map = track.divisions.explore(m=my_map, name="divisions", show=False, style_kwds=dict(color="grey"))

    style_kwds = dict(color="black")
    my_map = track.right.explore(m=my_map, name="right", control=False, style_kwds=style_kwds)
    my_map = track.left.explore(m=my_map, name="left", control=False, style_kwds=style_kwds)

    if not all_track_racelines.empty:
        style_kwds = dict(color="black", dashArray="2 5", opacity=0.6)
        my_map = all_track_racelines.explore(m=my_map, name="results", control=False, style_kwds=style_kwds)

    if not results.empty:
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
        track = st.radio("select track", options=laptime_sim.get_all_tracks(), format_func=lambda x: x.name)

    all_racelines = gpd.read_parquet(PATH_LINES, filters=[("track_name", "==", track.name)])
    d = st.radio("select result", options=all_racelines.to_dict(orient="records"), format_func=format_results)
    selected_raceline = all_racelines.from_dict([d]).set_crs(epsg=4326)
    selected_raceline.to_crs(track.crs, inplace=True)

    track_map = folium_track_map(track, all_racelines, selected_raceline)
    st_folium(track_map, returned_objects=[], use_container_width=True)

    race_car = [f for f in laptime_sim.get_all_cars() if f.name == selected_raceline.iloc[0].car][0]

    coords = selected_raceline.get_coordinates(include_z=True).to_numpy(na_value=0)
    sim_results = simulate(race_car, coords, track.slope)

    # st.write(saved_raceline.best_time_str)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.subplot(211)
    ax.plot(sim_results.distance, sim_results.speed)
    # ax2 = plt.subplot(223)
    # ax2.plot(saved_raceline.distance, saved_raceline.speed)
    # ax3 = plt.subplot(224)
    # ax3.plot(track.left_coords())

    st.pyplot(fig)
    with st.expander("Selected Raceline"):
        st.write(selected_raceline)
    with st.expander("SimResults"):
        st.write(sim_results)
    with st.expander("Race Car"):
        st.write(race_car)
    with st.expander("Race Track"):
        st.write(track)


if __name__ == "__main__":
    main()
