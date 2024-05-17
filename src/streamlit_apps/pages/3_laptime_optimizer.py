import itertools
from pathlib import Path
import streamlit as st
import laptime_sim
from laptime_sim.main import get_all_cars, get_all_tracks
from laptime_sim.raceline import Raceline
from laptime_sim.timer import Timer

PATH_TRACKS = "./tracks/"
PATH_CARS = "./cars/"
PATH_RESULTS = "./simulated/"


def show_laptime_and_iterations(raceline: Raceline, itererations: int, saved: bool) -> None:
    st.write(f"Results: {raceline.best_time_str} saved. iteration:{itererations}")


def main():
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    track = st.radio("select track", options=get_all_tracks(PATH_TRACKS), format_func=lambda x: x.name)
    race_car = st.radio(label="Select Car", options=get_all_cars(PATH_CARS), format_func=lambda x: x.name)
    raceline = laptime_sim.Raceline(track, race_car)
    filename_results = Path(PATH_RESULTS, f"{race_car.file_name}_{track.name}_simulated.parquet")

    if filename_results.exists():
        st.warning(f"Filename {filename_results} exists and will be overwritten")
        raceline.load_line(filename_results)

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        placeholder_saved = st.empty()
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_saved.write("optimization is started")

                timer1 = Timer(1)
                timer2 = Timer(30)

                raceline.save_line(filename_results)

                placeholder_laptime.write(f"Laptime = {raceline.best_time_str}")

                for itereration in itertools.count():

                    raceline.simulate_new_line()

                    if timer1.triggered:
                        placeholder_laptime.write(f"Laptime = {raceline.best_time_str}. iteration:{itereration}")
                        timer1.reset()

                    if timer2.triggered:
                        raceline.save_line(filename_results)
                        placeholder_saved.write(f"Results: {raceline.best_time_str} saved. iteration:{itereration}")

                        timer2.reset()

                raceline.save_line(filename_results)
                show_laptime_and_iterations(raceline, itereration, saved=True)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_saved.write("optimization is stopped")


if __name__ == "__main__":
    main()
