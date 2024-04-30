import os
import streamlit as st
import laptime_sim

PATH_TRACKS = "./tracks/"
PATH_CARS = "./cars/"


def main():
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    track = st.radio("select track", options=laptime_sim.get_all_tracks(PATH_TRACKS), format_func=lambda x: x.name)
    race_car = st.radio(label="Select Car", options=laptime_sim.get_all_cars(PATH_CARS), format_func=lambda x: x.name)

    raceline = laptime_sim.Raceline(track, race_car).load_raceline()

    if os.path.exists(raceline.filename_results):
        st.warning(f"Filename {raceline.filename_results} exists and will be overwritten")

    def show_laptime_and_nr_iterations(laptime: str, itererations: int, saved: bool) -> None:
        placeholder_saved = st.empty()
        placeholder_laptime.write(f"Laptime = {laptime}  (iteration:{itererations})")
        if saved:
            placeholder_saved.write(f"Results: {laptime} saved. iteration:{itererations}")

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        placeholder_saved = st.empty()
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_laptime.write("optimization is started")

                # this is a blocking function... no execution after this line, when optimizing...
                laptime_sim.optimize_raceline(raceline, show_laptime_and_nr_iterations)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_laptime.write("optimization is stopped")


if __name__ == "__main__":
    main()
