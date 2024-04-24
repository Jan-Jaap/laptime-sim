from datetime import datetime
import os
import streamlit as st

import laptime_sim
from geopandas import GeoDataFrame


def main():
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    track = st.radio("select track", options=laptime_sim.get_all_tracks(), format_func=lambda x: x.name)
    race_car = st.radio(label="Select Car", options=laptime_sim.get_all_cars(), format_func=lambda x: x.name)

    raceline = laptime_sim.Raceline.from_results(track, race_car)

    if os.path.exists(raceline.filename_results):
        st.warning(f"Filename {raceline.filename_results} exists and will be overwritten")

    def intermediate_results(time, itereration):
        placeholder_laptime.write(f"Laptime = {laptime_sim.time_to_str(time)}  (iteration:{itereration})")

    def save_results(raceline_gdf: GeoDataFrame) -> None:
        raceline_gdf.to_parquet(raceline.filename_results)
        now = datetime.now().strftime("%H:%M")
        lap_time = laptime_sim.time_to_str(raceline_gdf.best_time[0])
        placeholder_savefile.write(f"{now}: results {lap_time} saved to {raceline.filename_results}")

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        placeholder_laptime = st.empty()
        placeholder_savefile = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_laptime.write("optimization is started")

                # this is a blocking function... no execution after this line, when optimizing...
                laptime_sim.race_lap.optimize_laptime(raceline, intermediate_results, save_results)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_laptime.write("optimization is stopped")


if __name__ == "__main__":
    main()
