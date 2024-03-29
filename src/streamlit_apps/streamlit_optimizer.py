import os
from datetime import datetime
import streamlit as st

from laptime_sim import Car, Track
import laptime_sim.file_operations as io
import geopandas as gpd
from geopandas import GeoDataFrame


from laptime_sim import race_lap
from laptime_sim.raceline import Raceline

PATH_TRACK_FILES = "./tracks/"
PATH_RESULTS_FILES = "./simulated/"
PATH_CAR_FILES = "./cars/"


def st_select_file(label, dir, extensions):
    files_in_dir = [f for f in sorted(os.listdir(dir)) if f.endswith(extensions)]
    return os.path.join(dir, st.radio(label=label, options=files_in_dir))


def main():
    # with st.sidebar:
    #     dir_selected = st.radio("select directory", [PATH_TRACK_FILES])

    if 'optimization_running' not in st.session_state:
        st.session_state['optimization_running'] = False

    filename_track = st_select_file('Select Track', PATH_TRACK_FILES, "parquet")
    # track_layout = gpd.read_parquet(filename_track)
    # name_track = io.get_trackname_from_filename(filename_track)
    track = Track.from_parquet(filename_track)

    filename_car = st_select_file('Select Car', PATH_CAR_FILES, "toml")
    race_car = Car.from_toml(filename_car)
    name_car = io.strip_filename(filename_car)

    raceline = Raceline(track=track, car=race_car)

    filename_results = io.find_raceline_filename(track.name, name_car)
    if filename_results is not None:
        st.warning(f'Filename {filename_results} exists and will be overwritten')
        raceline_gdf = gpd.read_parquet(filename_results)
        raceline = raceline.parametrize_raceline(raceline_gdf=raceline_gdf)
    else:
        filename_results = os.path.join(PATH_RESULTS_FILES, '_'.join([name_car, track.name, 'simulated']) + '.parquet')

    def intermediate_results(time, itereration):
        placeholder_laptime.write(f"Laptime = {race_lap.time_to_str(time)}  (iteration:{itereration})")

    def save_results(raceline_gdf: GeoDataFrame) -> None:
        raceline_gdf.to_parquet(filename_results)
        placeholder_savefile.write(
            f"{datetime.now().strftime('%H:%M')}: results {raceline_gdf.best_time[0]} saved to {filename_results=}"
        )

    with st.status("Raceline optimization", state='error', expanded=True) as status:
        placeholder_laptime = st.empty()
        placeholder_savefile = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state='running')
                placeholder_laptime.write('optimization is started')

                # this is a blocking function... no execution after this line, when optimizing...
                race_lap.optimize_laptime(raceline, intermediate_results, save_results)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_laptime.write('optimization is stopped')


if __name__ == "__main__":
    main()
