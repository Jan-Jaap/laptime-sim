"""This module creates a streamlit app"""

import itertools
import os
import geopandas as gpd
import streamlit as st
from matplotlib import pyplot as plt

import laptime_sim
from laptime_sim.main import PATH_CARS, PATH_RESULTS, PATH_TRACKS
from laptime_sim import Raceline, Car


def format_results(data):
    car = list(data.keys())[0]
    if best_time := data[car]:
        return f"{car} ({best_time % 3600 // 60:02.0f}:{best_time % 60:06.03f})"
    return car


def optimize_raceline(raceline: Raceline, car: Car):
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        filename_results = PATH_RESULTS / raceline.filename(car.file_name)

        if filename_results.exists():
            st.warning(f"Filename {filename_results.name} exists and will be overwritten")
            raceline.load_line(filename_results)
        else:
            st.warning(f"Filename {filename_results} does not exist. New raceline will be created.")

        placeholder_saved = st.empty()
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_saved.write("optimization is started")

                timer1 = laptime_sim.Timer(1)
                timer2 = laptime_sim.Timer(30)

                raceline.save_line(filename_results, car.name)

                placeholder_laptime.write(f"Laptime = {raceline.best_time % 3600 // 60:02.0f}:{raceline.best_time % 60:06.03f}")

                timer3 = laptime_sim.Timer()
                for itereration in itertools.count():
                    raceline.simulate_new_line(car)
                    if timer1.triggered:
                        placeholder_laptime.write(
                            f"Laptime = {raceline.best_time_str()}. iteration:{itereration}, iteration_rate:{itereration / timer3.elapsed_time.total_seconds():.0f}"
                        )
                        timer1.reset()

                    if timer2.triggered:
                        raceline.save_line(filename_results, car.name)
                        placeholder_saved.write(f"Results: {raceline.best_time_str()} saved. iteration:{itereration}")
                        timer2.reset()

                raceline.save_line(filename_results, car.name)
                st.write(f"Results: {raceline.best_time_str()} saved. iteration:{itereration}")

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_saved.write("optimization is stopped")


def main() -> None:
    PATH_RESULTS.mkdir(exist_ok=True)

    with st.sidebar:
        track = st.radio("select track", options=laptime_sim.track_list(PATH_TRACKS), format_func=lambda x: x.name)
        race_car = st.radio("select car", options=laptime_sim.car_list(PATH_CARS), format_func=lambda x: x.name)

    if not os.listdir(PATH_RESULTS):
        all_racelines = gpd.GeoDataFrame()
    else:
        all_racelines = gpd.read_parquet(PATH_RESULTS, filters=[("track_name", "==", track.name)])

    selected_raceline = all_racelines.query(f"car == '{race_car.name}'")

    st.header(f"Racetrack - {track.name}")

    if selected_raceline.empty:
        st.warning("No results found for this car.")
        raceline = Raceline(track=track)
    else:
        raceline = Raceline.from_geodataframe(selected_raceline, track)
        raceline.simulate(race_car)
        # header.header(f"Race track display - {race_car.name}")
        st.info(f"{race_car.name}, Best time = {raceline.best_time_str()}")

    optimize_raceline(raceline, race_car)

    with st.expander("Race Car"):
        st.write(race_car)
    with st.expander("Selected Raceline"):
        sim_results = raceline.simulate(race_car)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_results.distance, sim_results.speed_kph)
        ax.set_ylabel("Speed in km/h")
        ax.set_xlabel("Track distance in m")
        st.pyplot(fig)
        st.write(raceline)
    with st.expander("SimResults"):
        st.write(sim_results)


if __name__ == "__main__":
    main()