"""This module creates a streamlit app"""

from pathlib import Path
import folium
import geopandas as gpd
import streamlit as st
import xyzservices.providers as xyz
from matplotlib import pyplot as plt
from streamlit_folium import st_folium
import laptime_sim
from main import CAR_LIST, TRACK_LIST, PATH_RESULTS


style_divisions = dict(color="grey")
style_track_border = dict(color="black")
style_start_finish = dict(color="red", weight=5)
style_all_racelines = dict(color="black", dashArray="2 5", opacity=0.6)
style_selected_raceline = dict(color="blue")


def create_track_map(track: laptime_sim.Track) -> folium.Map:
    """Add track border to the map"""
    return track.layout.explore(m=None, name="border", tooltip=False, control=False, style_kwds=style_track_border)


def add_divisions(my_map: folium.Map | None, track: laptime_sim.Track) -> None:
    """Add track divisions to the map"""
    track.divisions.explore(m=my_map, name="divisions", show=False, style_kwds=style_divisions)


def add_start_finish(my_map: folium.Map | None, track: laptime_sim.Track) -> None:
    """Add start/finish line to the map"""
    track.start_finish.explore(m=my_map, name="start_finish", style_kwds=style_start_finish)


def add_racelines(my_map: folium.Map | None, racelines: gpd.GeoDataFrame, style_kwds=style_all_racelines) -> None:
    """Add all racelines to the map"""
    if not racelines.empty:
        racelines.explore(m=my_map, tooltip=True, name="results", style_kwds=style_kwds)


def add_points(my_map: folium.Map | None, point: gpd.GeoSeries) -> None:
    """Add points to the map"""
    point.explore(m=my_map, tooltip=True, style_kwds=style_start_finish)


def add_base_layers(my_map: folium.Map) -> None:
    """Add base map layers and layer control"""
    folium.TileLayer(xyz.Esri.WorldImagery).add_to(my_map)  # type: ignore
    folium.TileLayer("openstreetmap").add_to(my_map)
    folium.LayerControl().add_to(my_map)


def main() -> None:
    with st.sidebar:
        track = st.radio("select track", options=TRACK_LIST, format_func=lambda x: x.name)
        race_car = st.radio("select car", options=CAR_LIST, format_func=lambda x: x.name)

    st.header(f"Racetrack - {track.name}")

    PATH_RESULTS.mkdir(exist_ok=True)
    file_path = PATH_RESULTS / Path(f"{race_car.file_name}_{track.name}_simulated.parquet")

    try:
        all_racelines = gpd.read_parquet(PATH_RESULTS, filters=[("track_name", "==", track.name)])
    except ValueError:
        all_racelines = gpd.GeoDataFrame()

    raceline = laptime_sim.Raceline(track)

    try:
        raceline.load_file(file_path)
    except FileNotFoundError:
        st.warning("No results found for this car.")

    sim_results = raceline.simulate(race_car)
    st.info(f"{race_car.name}, Best time = {raceline.best_time_str()}")

    race_line_df = raceline.dataframe(track_name=track.name, car_name=race_car.name)

    expander_track_map = st.expander("Track map", expanded=True)

    point_index = st.slider("Select point on raceline", min_value=0, max_value=track.len - 1, value=0)
    selected_point = raceline.get_point(point_index)

    with expander_track_map:
        # track_map = folium_track_map(track, all_racelines, raceline.dataframe(track_name=track.name, car_name=race_car.name))
        my_map = create_track_map(track)
        add_divisions(my_map, track)
        add_start_finish(my_map, track)
        add_racelines(my_map, all_racelines)
        add_racelines(my_map, race_line_df, style_kwds=style_selected_raceline)
        add_points(my_map, selected_point)
        add_base_layers(my_map)
        st_folium(my_map, returned_objects=[], use_container_width=True)

    with st.expander("Raceline data", expanded=True):
        raceline_datapoint = sim_results.get_dataframe().iloc[point_index]
        raceline_datapoint["slope"] = track.slope[point_index] * 100
        st.write(raceline_datapoint, use_container_width=True)
        # st.write(

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
        ax2.plot(sim_results.distance, raceline.get_coordinates()[:, 2], label="Elevation in m", color="C1")
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


if __name__ == "__main__":
    main()
