
import pandas as pd
import json, os, time

from track_sim.track import Track
from track_sim.car import Car
from utilities.dotdict import DotDict

FILENAME_CAR_PROPERTIES = './cars/BMW_Z3M.json'
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

#%% main scripts
def main():

    with open(FILENAME_CAR_PROPERTIES, 'r') as fp:
        race_car = Car(json.load(fp))

    filename_results = f'{RESULTS_PATH}{race_car.name}_Zandvoort_simulated.csv'
    nr_iterations = 0

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
        df_track.filter(regex="outer_").values,
        df_track.filter(regex="inner_").values,
        initial_position=best_known_raceline
        )

    laptime = track.race(race_car, best_known_raceline)
    print(f'{race_car.name} - Simulated laptime = {laptime_str(laptime)}')

    optimize_yn = input('Start line optimization? [y/N]')
    start_time = time.time()

    try:
        while optimize_yn in ['y', 'Y']:
            new_race_line = track.new_line(best_known_raceline)
            new_laptime = track.race(race_car, new_race_line)

            nr_iterations += 1

            if new_laptime < laptime:
                laptime = new_laptime
                best_known_raceline = new_race_line

                if time.time() - start_time > 5:
                    start_time = time.time()
                    print(f"Laptime = {laptime_str(laptime)}  (iteration:{nr_iterations})")

    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')


#%% save results
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    results = track.race(race_car, best_known_raceline, verbose=True)
    results['speed'] *= 3.6  #convert speed from m/s to km/h

    return_dataframe(df_track, DotDict(results)).to_csv(filename_results, index = None, header=True)
    print(f'{race_car.name} - Simulated laptime = {laptime%3600//60:02.0f}:{laptime%60:05.02f}')


if __name__ == '__main__':
    main()