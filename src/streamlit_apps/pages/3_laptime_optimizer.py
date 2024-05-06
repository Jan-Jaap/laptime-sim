import os
import streamlit as st
import laptime_sim
from laptime_sim.raceline import Raceline

PATH_TRACKS = "./tracks/"
PATH_CARS = "./cars/"
PATH_RESULTS = "./simulated/"


def main():
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    track = st.radio("select track", options=laptime_sim.get_all_tracks(PATH_TRACKS), format_func=lambda x: x.name)
    race_car = st.radio(label="Select Car", options=laptime_sim.get_all_cars(PATH_CARS), format_func=lambda x: x.name)
    simulator = laptime_sim.RacelineSimulator(race_car)

    filename_results = os.path.join(PATH_RESULTS, f"{race_car.file_name}_{track.name}_simulated.parquet")

    raceline = laptime_sim.Raceline(track, race_car, simulator).load_results(filename_results)

    if os.path.exists(filename_results):
        st.warning(f"Filename {filename_results} exists and will be overwritten")

    def show_laptime_and_nr_iterations(raceline: Raceline, itererations: int, saved: bool) -> None:
        placeholder_laptime.write(f"Laptime = {raceline.best_time_str}  (iteration:{itererations})")
        if saved:
            placeholder_saved.write(f"Results: {raceline.best_time_str} saved. iteration:{itererations}")

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        placeholder_saved = st.empty()
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_laptime.write("optimization is started")

                # this is a blocking function... no execution after this line, when optimizing...
                laptime_sim.optimize_raceline(raceline, show_laptime_and_nr_iterations, filename_results)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_laptime.write("optimization is stopped")


if __name__ == "__main__":
    main()
