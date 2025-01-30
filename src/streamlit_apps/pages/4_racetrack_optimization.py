"""This module creates a streamlit app"""

import itertools
from pathlib import Path
import streamlit as st
from matplotlib import pyplot as plt

import laptime_sim
from laptime_sim.main import PATH_CARS, PATH_RESULTS, PATH_TRACKS
from laptime_sim import Raceline, Car, Track


def format_results(data):
    """Format a single result for display in a selectbox.

    Args:
        data: A mapping with a single key-value pair, where the key is the car name and the value is the best time.

    Returns:
        A string that displays the car name and best time in the format "Car Name (minutes:seconds.milliseconds)".
    """
    car = list(data.keys())[0]
    if best_time := data[car]:
        return f"{car} ({best_time % 3600 // 60:02.0f}:{best_time % 60:06.03f})"
    return car


def optimize_raceline(track: Track, car: Car, raceline: Raceline):
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        file_path = PATH_RESULTS / results_filename(track, car)

        placeholder_saved = st.empty()
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_saved.write("optimization is started")

                timer1 = laptime_sim.Timer(1)
                timer2 = laptime_sim.Timer(30)

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
                        raceline.save_line(file_path, car.name)
                        placeholder_saved.write(f"Results: {raceline.best_time_str()} saved. iteration:{itereration}")
                        timer2.reset()

                st.write(f"Results: {raceline.best_time_str()} saved. iteration:{itereration}")

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_saved.write("optimization is stopped")

        raceline.save_line(file_path, car.name)


def main() -> None:
    with st.sidebar:
        track = st.radio("select track", options=laptime_sim.track_list(PATH_TRACKS), format_func=lambda x: x.name)
        race_car = st.radio("select car", options=laptime_sim.car_list(PATH_CARS), format_func=lambda x: x.name)

    st.header(f"Racetrack - {track.name}")

    PATH_RESULTS.mkdir(exist_ok=True)
    file_path = PATH_RESULTS / results_filename(track, race_car)

    try:
        raceline = laptime_sim.Raceline.from_file(track, file_path)
        st.warning(f"Filename {file_path.name} exists and will be overwritten")

    except FileNotFoundError:
        st.warning("No results found for this car. A new raceline will be created.")
        raceline = laptime_sim.Raceline(track=track)

    raceline.update(race_car)
    st.info(f"{race_car.name}, Best time = {raceline.best_time_str()}")

    optimize_raceline(track, race_car, raceline)

    with st.expander("Race Car"):
        st.write(race_car)
    with st.expander("Selected Raceline"):
        sim_results = raceline.update(race_car)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_results.distance, sim_results.speed_kph)
        ax.set_ylabel("Speed in km/h")
        ax.set_xlabel("Track distance in m")
        st.pyplot(fig)
        st.write(raceline)
    with st.expander("SimResults"):
        st.write(sim_results)


def results_filename(track, car):
    return Path(f"{car.file_name}_{track.name}_simulated.parquet")


if __name__ == "__main__":
    main()
