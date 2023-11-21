import os
from datetime import datetime
import streamlit as st
from car import Car
import file_operations as io

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
    track_layout = io.load_trackdata_from_file(filename_track)

    name_track = io.get_trackname_from_filename(filename_track)

    filename_car = st_select_file('Select Car', PATH_CAR_FILES, "toml")
    race_car = Car.from_toml(filename_car)
    name_car = io.strip_filename(filename_car)

    filename_results = '_'.join([name_car, name_track, 'simulated'])

    f = io.find_filename(name_track, name_car)
    if f is not None:
        st.warning(f'Filename {f} exists and will be overwritten')

    track_session = TrackSession(track_layout, race_car)

    def intermediate_results(time, itereration):
        placeholder_laptime.write(f"Laptime = {time_to_str(time)}  (iteration:{itereration})")

    def save_results(track_layout) -> None:
        filename = filename_results + '.parquet'
        io.save_parquet(track_layout, filename)
        placeholder_savefile.write(
            f"{datetime.now().strftime('%H:%M:%S')}: results saved to {filename=}"
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
                # st.session_state['track_session'] =
                race_lap.optimize_laptime(track_session, intermediate_results, save_results)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                # status.update(state='running')
                placeholder_laptime.write('optimization is stopped')


if __name__ == "__main__":
    main()
