"""This module creates a streamlit app"""

from pathlib import Path
import folium
import geopandas as gpd
import streamlit as st
import xyzservices.providers as xyz
from matplotlib import pyplot as plt
from streamlit_folium import st_folium

import laptime_sim
from laptime_sim.main import PATH_CARS, PATH_RESULTS, PATH_TRACKS


def folium_track_map(track: laptime_sim.Track, all_track_racelines: gpd.GeoDataFrame, race_line_df):
    my_map = track.divisions.explore(name="divisions", show=False, style_kwds=dict(color="grey"))
    style_kwds = dict(color="black")
    track.layout.explore(m=my_map, name="border", control=False, style_kwds=style_kwds)
    gpd.GeoSeries(track.start_finish, crs=track.crs).explore(
        m=my_map,
        name="start_finish",
        style_kwds=dict(
            color="red",
            weight=5,
        ),
    )
    if not all_track_racelines.empty:
        style_kwds = dict(color="black", dashArray="2 5", opacity=0.6)
        my_map = all_track_racelines.explore(m=my_map, tooltip="car", name="results", control=True, style_kwds=style_kwds)

    if race_line_df is not None and not race_line_df.empty:
        race_line_df.explore(m=my_map, name=race_line_df.car.iloc[0], style_kwds=dict(color="blue"))

    folium.TileLayer(xyz.Esri.WorldImagery).add_to(my_map)
    folium.TileLayer("openstreetmap").add_to(my_map)
    folium.LayerControl().add_to(my_map)
    return my_map


def main() -> None:
    with st.sidebar:
        track = st.radio("select track", options=laptime_sim.track_list(PATH_TRACKS), format_func=lambda x: x.name)
        race_car = st.radio("select car", options=laptime_sim.car_list(PATH_CARS), format_func=lambda x: x.name)

    st.header(f"Racetrack - {track.name}")

    PATH_RESULTS.mkdir(exist_ok=True)
    file_path = PATH_RESULTS / Path(f"{race_car.file_name}_{track.name}_simulated.parquet")

    try:
        all_racelines = gpd.read_parquet(PATH_RESULTS, filters=[("track_name", "==", track.name)])
    except ValueError:
        all_racelines = gpd.GeoDataFrame()

    try:
        raceline = laptime_sim.Raceline.from_file(track, file_path)
    except FileNotFoundError:
        st.warning("No results found for this car.")
        raceline = laptime_sim.Raceline(track=track)

    raceline.simulate(race_car)
    st.info(f"{race_car.name}, Best time = {raceline.best_time_str()}")

    track_map = folium_track_map(track, all_racelines, raceline.dataframe(track_name=track.name, car_name=race_car.name))
    st_folium(track_map, returned_objects=[], use_container_width=True)

    with st.expander("Raceline speed", expanded=True):
        sim_results = raceline.simulate(race_car)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_results.distance, sim_results.speed_kph)
        ax.set_ylabel("Speed in km/h")
        ax.set_xlabel("Track distance in m")
        st.pyplot(fig)

    with st.expander("Track definition"):
        st.write(track.__dict__)


if __name__ == "__main__":
    main()
