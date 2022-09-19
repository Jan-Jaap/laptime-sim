
import pandas as pd
import json, os, time

from track_sim.track import Track
from track_sim.car import Car
import track_sim.streamlit_ui as ui

import threading

FILENAME_CAR_PROPERTIES = './cars/Peugeot_205RFS.json'
FILENAME_TRACK = './tracks/20191030_Circuit_Zandvoort copy.csv'
RESULTS_PATH = './simulated/'
#Racechrono csv.v2 Headers
rc_header = dict(
        speed  = 'Speed (m/s)',
        distance = 'Distance (m)',
        time = 'Time (s)',
        a_lat = 'Lateral acceleration (m/s2)',
        a_lon = 'Longitudinal acceleration (m/s2)',
        )
 

def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)


def return_dataframe(df, results):
    df['Distance (m)']=results.distance
    df['line_x']=results.line[:,0]
    df['line_y']=results.line[:,1]
    df['Speed (m/s)'] = results.speed
    df['Longitudinal acceleration (m/s2)'] = results.a_lon
    df['Lateral acceleration (m/s2)'] = results.a_lat
    df['Race line'] = results.race_line_position
    df['Timestamp'] = results.time
    return df


def thread_target(iterations, seconds):
    for i in range(iterations):
        print(f"thread_target ({i})")
        time.sleep(seconds)
    print('=== END (Refresh or CTRL-C) ===')

#%% main scripts
def main():

    with open(FILENAME_CAR_PROPERTIES, 'r') as fp:
        race_car = Car(json.load(fp))

    filename_results = f'{RESULTS_PATH}{race_car.name}_Zandvoort_simulated.csv'

    try:
        df_track = pd.read_csv(filename_results)
    except FileNotFoundError:
        df_track = pd.read_csv(FILENAME_TRACK)

    for column in ['Race line','Optimized line','initial_position']:
        if column in df_track.columns:
            best_known_raceline = df_track[column].values
        else:
            df_track['Race line'] = 0.5
            best_known_raceline = df_track['Race line'].values
        break
    

    track = Track(
        name                = "Zandvoort",
        outside             = df_track.filter(regex="outer_").values,
        inside              = df_track.filter(regex="inner_").values,
        initial_position    = best_known_raceline
        )

    laptime = track.race(race_car, best_known_raceline)
    ui.write(f'{race_car.name} - Simulated laptime = {laptime_str(laptime)}')
    ui.plot_track(track, track.get_line_coordinates(best_known_raceline))

if __name__ == '__main__':
    main()