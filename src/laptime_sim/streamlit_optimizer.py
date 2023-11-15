import os
import streamlit as st
import file_operations

from tracksession import TrackSession

SUPPORTED_FILETYPES = (".parquet")
PATH_TRACK_FILES = "./tracks/"
PATH_RESULTS_FILES = "./simulated/"
PATH_CAR_FILES = "./cars/"


def main():

    if 'optimization_running' not in st.session_state:
        st.session_state['optimization_running'] = False

    tracks_in_dir = [f for f in sorted(os.listdir(PATH_TRACK_FILES)) if f.endswith(SUPPORTED_FILETYPES)]
    file_name = os.path.join(PATH_TRACK_FILES, st.radio(label="select track", options=tracks_in_dir))

    track_layout = file_operations.load_trackdata_from_file(file_name)

    # dir_name = os.path.dirname(file_name)
    # file_name = os.path.join(PATH_RESULTS_FILES, os.path.basename(file_name)+'.csv')

    track_session = TrackSession(track_layout, race_car)

    def intermediate_results(time, itereration, track_session):
        placeholder_laptime.write(f"Laptime = {time_to_str(time)}  (iteration:{itereration})")
        st.session_state['track_session'] = track_session

    def save_results(track_session) -> None:
        results = race_lap.sim(track_session, verbose=True)
        file_operations.save_csv(results, file_name)
        st.write(
            f"{datetime.now().strftime('%H:%M:%S')}: results saved to {file_name=}"
        )
    if st.button('save track to results'):
        # results = race_lap.sim(track_session, verbose=True)
        save_results(track_session)

    with st.status("Raceline optimization", state='error', expanded=True) as status:
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state='running')
                placeholder_laptime.write('optimization is started')

                # this is a blocking function... no execution after this line, when optimizing...
                st.session_state['track_session'] = race_lap.optimize_laptime(
                    track_session, intermediate_results, save_results)

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                # status.update(state='running')
                placeholder_laptime.write('optimization is stopped')


if __name__ == "__main__":
    main()
