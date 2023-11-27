import os
from datetime import datetime
import streamlit as st
from car import Car
import file_operations as io
import geopandas as gpd
from geopandas import GeoDataFrame

from tracksession import TrackSession
import race_lap
from race_lap import time_to_str

from icecream import install
install()

PATH_TRACK_FILES = "./tracks/"
PATH_RESULTS_FILES = "./simulated/"
PATH_CAR_FILES = "./cars/"


def st_select_file(label, dir, extensions):
    files_in_dir = [f for f in sorted(os.listdir(dir)) if f.endswith(extensions)]
    return os.path.join(dir, st.radio(label=label, options=files_in_dir))


def main():
    with st.sidebar:
        dir_selected = st.radio("select directory", [PATH_TRACK_FILES]+[PATH_RESULTS_FILES])

    if 'optimization_running' not in st.session_state:
        st.session_state['optimization_running'] = False

    filename_track = st_select_file('Select Track', dir_selected, "parquet")
    track_layout = gpd.read_parquet(filename_track)

    name_track = io.get_trackname_from_filename(filename_track)

    filename_car = st_select_file('Select Car', PATH_CAR_FILES, "toml")
    race_car = Car.from_toml(filename_car)
    name_car = io.strip_filename(filename_car)

    filename_results = io.find_raceline_filename(name_track, name_car)
    if filename_results is not None:
        st.warning(f'Filename {filename_results} exists and will be overwritten')
    else:
        filename_results = os.path.join(PATH_RESULTS_FILES, '_'.join([name_car, name_track, 'simulated']) + '.parquet')

    track_session = TrackSession(
        track_border_left=track_layout.outer.geometry,
        track_border_right=track_layout.inner.geometry,
        )

    def intermediate_results(time, itereration):
        placeholder_laptime.write(f"Laptime = {time_to_str(time)}  (iteration:{itereration})")

    def save_results(track_layout: GeoDataFrame) -> None:
        track_layout.to_parquet(filename_results)
        placeholder_savefile.write(
            f"{datetime.now().strftime('%H:%M:%S')}: results saved to {filename_results=}"
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
                race_lap.optimize_laptime(track_session, race_car, intermediate_results, save_results)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_laptime.write('optimization is stopped')


if __name__ == "__main__":
    main()
