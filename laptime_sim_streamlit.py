import pandas as pd
from track_sim.sim import Track
import track_sim.streamlit_ui as ui

#%% main scripts
def main():
    ui.page_config()

    if filename_results := ui.upload_results():

        df_track = pd.read_csv(filename_results)

        if 'Timestamp' in df_track.columns:
            laptime = df_track.Timestamp.iloc[-1]
            ui.write(f'Simulated laptime = {laptime%3600//60:02.0f}:{laptime%60:06.03f}')
  
        track = Track(
            name                = 'uploaded track',
            border_left         = df_track.filter(regex="outer_").values,
            border_right        = df_track.filter(regex="inner_").values,
            )

        for column in ['Race line','Optimized line','initial_position']:
            if column in df_track.columns:
                best_known_raceline = df_track[column].values
                race_line = track.get_line_coordinates(best_known_raceline)
            else:
                # df_track['Race line'] = 0.5
                # race_line = track.get_line_coordinates(df_track['Race line'])
                race_line = None
            break
        
        ui.plot_track(track, race_line)

if __name__ == '__main__':
    main()