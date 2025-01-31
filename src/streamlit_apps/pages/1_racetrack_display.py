"""This module creates a streamlit app"""

from pathlib import Path
import folium
import geopandas as gpd
import streamlit as st
import xyzservices.providers as xyz
from matplotlib import pyplot as plt
from streamlit_folium import st_folium
import laptime_sim
from main import PATH_CARS, PATH_TRACKS, PATH_RESULTS


style_divisions = dict(color="grey")
style_track_border = dict(color="black")
style_start_finish = dict(color="red", weight=5)
style_all_racelines = dict(color="black", dashArray="2 5", opacity=0.6)
style_selected_raceline = dict(color="blue")


def folium_track_map(track: laptime_sim.Track, all_track_racelines: gpd.GeoDataFrame, race_line_df: gpd.GeoDataFrame = None):
    """
    Create a folium map of a track with its divisions, start finish line, the border of the track, and all racelines.

    Parameters
    ----------
    track: laptime_sim.Track
        The track to display
    all_track_racelines: gpd.GeoDataFrame
        A geopandas dataframe with all racelines of all cars
    race_line_df: gpd.GeoDataFrame, default None
        A geopandas dataframe with the raceline of the selected car

    Returns
    -------
    folium.Map
        The generated map
    """
    my_map = track.divisions.explore(name="divisions", show=False, style_kwds=style_divisions)
    track.layout.explore(m=my_map, name="border", control=False, style_kwds=style_track_border)
    track.start_finish.explore(m=my_map, name="start_finish", style_kwds=style_start_finish)
    if not all_track_racelines.empty:
        all_track_racelines.explore(m=my_map, tooltip="car", name="results", style_kwds=style_all_racelines)

    if race_line_df is not None and not race_line_df.empty:
        race_line_df.explore(m=my_map, name=race_line_df.car.iloc[0], style_kwds=style_selected_raceline)

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

    sim_results = raceline.update(race_car)
    st.info(f"{race_car.name}, Best time = {raceline.best_time_str()}")

    with st.expander("Track Map", expanded=True):
        track_map = folium_track_map(track, all_racelines, raceline.dataframe(track_name=track.name, car_name=race_car.name))
        st_folium(track_map, returned_objects=[], use_container_width=True)

    with st.expander("Raceline speed", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_results.distance, sim_results.speed_kph)
        ax.set_ylabel("Speed in km/h")
        ax.set_xlabel("Track distance in m")
        st.pyplot(fig)

    with st.expander("Track slope and elevation", expanded=True):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        ax1.plot(sim_results.distance, track.slope * 100, label="Slope in %", color="C0")
        ax2.plot(sim_results.distance, raceline.coordinates()[:, 2], label="Elevation in m", color="C1")
        ax1.set_ylabel("Slope in %")
        ax2.set_ylabel("Elevation in m")
        ax1.set_xlabel("Track distance in m")
        ax1.tick_params(axis="y", colors="C0")
        ax2.tick_params(axis="y", colors="C1")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        ax2.legend(loc="upper left", bbox_to_anchor=(1.05, 0.95))
        st.pyplot(fig)

    with st.expander("Raceline acceleration", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_results.distance, sim_results.a_lon / 9.81, label="Longitudinal")
        ax.plot(sim_results.distance, sim_results.a_lat / 9.81, label="Lateral")
        ax.set_ylabel("Acceleration in g")
        ax.set_xlabel("Track distance in m")
        ax.legend()
        st.pyplot(fig)

    acc_lat, acc = race_car.performance_envelope(0)

    with st.expander("Raceline g-g plot", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(acc_lat, acc, label="Car performance envelope", color="k")
        ax.plot(sim_results.a_lat, sim_results.a_lon, label="Raceline")
        ax.set_ylabel("Longitudinal acceleration in m/s2")
        ax.set_xlabel("Lateral acceleration in m/s2")
        ax.legend()
        st.pyplot(fig)

    # with st.expander("Track definition"):
    #     st.write(track.__dict__)


if __name__ == "__main__":
    main()
