import pandas as pd
import json, os

from track_sim.sim import Car, Track
from utilities.timer import Timer

PATH_RESULTS_   = './simulated/'
PATH_TRACKS     = './tracks/'
PATH_CARS       = './cars/'

NAME_CAR = "Peugeot_205RFS"
NAME_TRACK = "20191030_Circuit_Zandvoort"

OUTPUT_COLUMNS_NAMES = dict(
    distance = 'Distance (m)',
    line_x = 'line_x',
    line_y='line_y',
    speed='Speed (m/s)',
    a_lon='Longitudinal acceleration (m/s2)',
    a_lat='Lateral acceleration (m/s2)',
    race_line_position='Race line',
    time='Timestamp',
)

filename_car_properties = f"{PATH_CARS}{NAME_CAR}.json"
filename_track = f"{PATH_TRACKS}{NAME_TRACK}.csv"
filename_results = f'{PATH_RESULTS_}{NAME_CAR}_{NAME_TRACK}_simulated.csv'

def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)

def save_results(df, results):
    results = results.rename(columns = OUTPUT_COLUMNS_NAMES)
    df.drop(columns=results.columns, errors='ignore').join(results).to_csv(filename_results, index = None, header=True)


def main():

    with open(filename_car_properties, 'r') as fp:
        race_car = Car(json.load(fp))

    try:
        df_track = pd.read_csv(filename_results)
    except FileNotFoundError:
        df_track = pd.read_csv(filename_track)

    for column in ['Race line','Optimized line','initial_position']:
        if column in df_track.columns:
            best_known_raceline = df_track[column].values
        else:
            df_track['Race line'] = 0.5
            best_known_raceline = df_track['Race line'].values
        break
    
    track = Track(
        name = NAME_TRACK,
        border_left     = df_track.filter(regex="outer_").values,
        border_right    = df_track.filter(regex="inner_").values,
        min_clearance   = 0.85,
        )

    laptime = track.race(race_car, best_known_raceline)
    print(f'{race_car.name} - Simulated laptime = {laptime_str(laptime)}')

    nr_iterations = 0

    timer1 = Timer()
    timer2 = Timer()

    try:
        while True:
            new_race_line = track.new_line(best_known_raceline)
            new_laptime = track.race(race_car, new_race_line)

            nr_iterations += 1

            if new_laptime < laptime:
                laptime = new_laptime
                best_known_raceline = new_race_line

                if timer1.elapsed_time > 3:
                    print(f"Laptime = {laptime_str(laptime)}  (iteration:{nr_iterations})")
                    timer1.reset()
            
            if timer2.elapsed_time > 10:
                results = track.race(race_car, best_known_raceline, verbose=True)

                
                save_results(df_track, results)#.to_csv(filename_results, index = None, header=True)
                print(f'intermediate results saved to {filename_results=}')
                timer2.reset()
                
    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')


    if not os.path.exists(PATH_RESULTS_):
        os.makedirs(PATH_RESULTS_)

    results = track.race(race_car, best_known_raceline, verbose=True)
    # return_dataframe(df_track, results).to_csv(filename_results, index = None, header=True)
    save_results(df_track,  results)#.rename(columns = OUTPUT_COLUMNS_NAMES)).to_csv(filename_results, index = None, header=True)

    print(f'{race_car.name} - Simulated laptime = {laptime_str(laptime)}')


if __name__ == '__main__':
    main()