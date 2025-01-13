"""This module creates a streamlit app"""

import itertools
from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
import xyzservices.providers as xyz
from matplotlib import pyplot as plt
from streamlit_folium import st_folium

import laptime_sim
from laptime_sim import Raceline

PATH_RESULTS = Path("./simulated/")
PATH_TRACKS = Path("./tracks/")
PATH_CARS = Path("./cars/")


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
    best_time = data["best_time"]
    return f"{data['car']} ({best_time % 3600 // 60:02.0f}:{best_time % 60:06.03f})"


def optimize_raceline(raceline: laptime_sim.Raceline):
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        filename_results = Path(PATH_RESULTS, f"{raceline.car.file_name}_{raceline.track.name}_simulated.parquet")

        if filename_results.exists():
            st.warning(f"Filename {filename_results} exists and will be overwritten")
            raceline.load_line(filename_results)

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
                            f"Laptime = {raceline.best_time_str}. iteration:{itereration}, iteration_rate:{itereration / timer3.elapsed_time:.0f}"
                        )
                        timer1.reset()

                    if timer2.triggered:
                        raceline.save_line(filename_results)
                        placeholder_saved.write(f"Results: {raceline.best_time_str} saved. iteration:{itereration}")
                        timer2.reset()

                raceline.save_line(filename_results)
                st.write(f"Results: {raceline.best_time_str} saved. iteration:{itereration}")

                # show_laptime_and_iterations(raceline, itereration, saved=True)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_saved.write("optimization is stopped")


def main() -> None:
    st.header("Race track display")

    all_cars = laptime_sim.car_list(PATH_CARS)
    all_tracks = laptime_sim.track_list(PATH_TRACKS)

    with st.sidebar:
        track = st.radio("select track", options=all_tracks, format_func=lambda x: x.name)

    # TODO add car selection to the sidebar and load the car properties from the car.toml file in the cars folder (see car.py)

    all_racelines = gpd.read_parquet(PATH_RESULTS, filters=[("track_name", "==", track.name)])

    if all_racelines.empty:
        st.warning("No results found for this track")
        selected_raceline = None
    else:
        d = st.radio("select raceline", options=all_racelines.to_dict(orient="records"), format_func=format_results)
        selected_raceline = all_racelines.from_dict([d]).set_crs(epsg=4326)
        raceline = Raceline.from_geodataframe(selected_raceline, all_cars=all_cars, all_tracks=all_tracks)

    track_map = folium_track_map(track, all_racelines, selected_raceline)
    st_folium(track_map, returned_objects=[], use_container_width=True)

    optimize_raceline(raceline)

    sim_results = raceline.simulate()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sim_results.distance, sim_results.speed_kph)
    ax.set_ylabel("Speed in km/h")
    ax.set_xlabel("Track distance in m")
    st.pyplot(fig)

    with st.expander("Selected Raceline"):
        st.write(raceline)
    with st.expander("SimResults"):
        st.write(sim_results)
    with st.expander("Race Car"):
        st.write(raceline.car)
    with st.expander("Race Track Layout"):
        st.write(raceline.track.layout)
        st.write(raceline.track)


if __name__ == "__main__":
    main()
