"""This module creates a streamlit app"""

import itertools
import os
from pathlib import Path
import folium
import geopandas as gpd
import streamlit as st
import xyzservices.providers as xyz
from matplotlib import pyplot as plt
from streamlit_folium import st_folium

import laptime_sim
from laptime_sim.main import PATH_CARS, PATH_RESULTS, PATH_TRACKS
from laptime_sim import Raceline


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


def format_results(data):
    car = list(data.keys())[0]
    if best_time := data[car]:
        return f"{car} ({best_time % 3600 // 60:02.0f}:{best_time % 60:06.03f})"
    return car


def optimize_raceline(raceline: laptime_sim.Raceline):
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        filename_results = PATH_RESULTS / raceline.filename

        if filename_results.exists():
            st.warning(f"Filename {raceline.filename} exists and will be overwritten")
            raceline.load_line(filename_results)
        else:
            st.warning(f"Filename {raceline.filename} does not exist. New raceline will be created.")

        placeholder_saved = st.empty()
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_saved.write("optimization is started")

                timer1 = laptime_sim.Timer(1)
                timer2 = laptime_sim.Timer(30)

                raceline.save_line(filename_results)

                placeholder_laptime.write(f"Laptime = {raceline.best_time_str}")

                timer3 = laptime_sim.Timer()
                for itereration in itertools.count():
                    raceline.simulate_new_line()

                    if timer1.triggered:
                        placeholder_laptime.write(
                            f"Laptime = {raceline.best_time_str}. iteration:{itereration}, iteration_rate:{itereration / timer3.elapsed_time.total_seconds():.0f}"
                        )
                        timer1.reset()

                    if timer2.triggered:
                        raceline.save_line(filename_results)
                        placeholder_saved.write(f"Results: {raceline.best_time_str} saved. iteration:{itereration}")
                        timer2.reset()

                raceline.save_line(filename_results)
                st.write(f"Results: {raceline.best_time_str} saved. iteration:{itereration}")

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_saved.write("optimization is stopped")


def main() -> None:
    header = st.header("Race track display")

    path_results = Path(PATH_RESULTS)
    path_results.mkdir(exist_ok=True)

    with st.sidebar:
        track = st.radio("select track", options=laptime_sim.track_list(PATH_TRACKS), format_func=lambda x: x.name)
        race_car = st.radio("select car", options=laptime_sim.car_list(PATH_CARS), format_func=lambda x: x.name)

    if not os.listdir(path_results):
        all_racelines = gpd.GeoDataFrame()
    else:
        all_racelines = gpd.read_parquet(path_results, filters=[("track_name", "==", track.name)])

    selected_raceline = all_racelines.query(f"car == '{race_car.name}'")

    if selected_raceline.empty:
        st.warning("No results found for this car.")
    else:
        header.header(f"Race track display - {race_car.name}")
        best_time = selected_raceline.best_time.values[0]

        st.info(f"{track.name}, Best time = {best_time % 3600 // 60:02.0f}:{best_time % 60:06.03f}")

    track_map = folium_track_map(track, all_racelines, selected_raceline)
    st_folium(track_map, returned_objects=[], use_container_width=True)

    with st.expander("Race Track Layout"):
        st.write(track.layout)
        st.write(track)
    with st.expander("Race Car"):
        st.write(race_car)

    raceline = Raceline(track=track, car=race_car)

    with st.expander("Selected Raceline"):
        sim_results = raceline.run_sim(raceline.car)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_results.distance, sim_results.speed_kph)
        ax.set_ylabel("Speed in km/h")
        ax.set_xlabel("Track distance in m")
        st.pyplot(fig)
        st.write(raceline)

    optimize_raceline(raceline)

    with st.expander("SimResults"):
        st.write(sim_results)


if __name__ == "__main__":
    main()
