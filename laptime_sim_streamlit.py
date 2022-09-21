
import pandas as pd

from track_sim.track import Track
import track_sim.streamlit_ui as ui

NAME_CAR = "Peugeot_205RFS"
NAME_TRACK = "20191030_Circuit_Zandvoort"
RESULTS_PATH = './simulated/'

filename_results = f'{RESULTS_PATH}{NAME_CAR}_{NAME_TRACK}_simulated.csv'

def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)


#%% main scripts
def main():
    ui.page_config()

    filename_results = ui.upload_results()
    df_track = pd.read_csv(filename_results)
    laptime = df_track.Timestamp.iloc[-1]

    best_known_raceline = df_track['Race line'].values
    
    track = Track(
        name                = NAME_TRACK,
        border_left         = df_track.filter(regex="outer_").values,
        border_right        = df_track.filter(regex="inner_").values,
        )

    ui.write(f'{NAME_CAR} - Simulated laptime = {laptime_str(laptime)}')
    ui.plot_track(track, track.get_line_coordinates(best_known_raceline))

if __name__ == '__main__':
    main()