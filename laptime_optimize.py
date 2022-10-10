import pandas as pd
import json, os
import geopandas as gpd

from track_sim.car import Car
from track_sim.track import Track
from track_sim.driver import Driver

from utilities.timer import Timer


PATH_RESULTS_   = './simulated/'
PATH_TRACKS     = './tracks/'
PATH_CARS       = './cars/'

NAME_CAR = "Peugeot_205RFS"
NAME_TRACK = "20191211_Bilsterberg"

OUTPUT_COLUMNS_NAMES = dict(
    distance            = 'Distance (m)',
    line_x              = 'line_x',
    line_y              = 'line_y',
    line_z              = 'line_z',
    speed               = 'Speed (m/s)',
    a_lon               = 'Longitudinal acceleration (m/s2)',
    a_lat               = 'Lateral acceleration (m/s2)',
    race_line_position  = 'Race line',
    time                = 'Timestamp',
)


def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)

def save_results(df, results, filename_results):
    results = pd.DataFrame(
            data = results,
            columns=(('race_line_position', 'distance', 'line_x', 'line_y', 'line_z', 'speed', 'time', 'a_lat', 'a_lon' ))
            ).rename(columns = OUTPUT_COLUMNS_NAMES)

    df.drop(columns=results.columns, errors='ignore').join(results).to_csv(filename_results, index = None, header=True)


def get_track_data(track):
       
    filename = f'{PATH_RESULTS_}{NAME_CAR}_{track}_simulated.csv'
    if os.path.isfile(filename):
        print(f'Loading track data from {filename}')
        return pd.read_csv(filename)
    
    filename = f"{PATH_TRACKS}{track}.geojson"
    if os.path.isfile(filename):
        print(f'Loading geojson track from {filename}')
        return gpd.read_file(filename)

    filename = f"{PATH_TRACKS}{track}.csv"
    if os.path.isfile(filename):
        print(f'Loading csv track from {filename}')
        return pd.read_csv(filename)

    print('No track data found')
    return False


def main():

    filename_car_properties = f"{PATH_CARS}{NAME_CAR}.json"
    filename_results = f'{PATH_RESULTS_}{NAME_CAR}_{NAME_TRACK}_simulated.csv'

    with open(filename_car_properties, 'r') as fp:
        race_car = Car(json.load(fp))

    df_track = get_track_data(NAME_TRACK)
    
    for column in ['Race line','Optimized line','initial_position']:
        if column in df_track.columns:
            best_known_raceline = df_track[column].values
        else:
            best_known_raceline = None
        break
     
    track = Track(
        name = NAME_TRACK,
        border_left     = df_track.filter(regex="outer_").values,
        border_right    = df_track.filter(regex="inner_").values,
        best_known_raceline = best_known_raceline,
        min_clearance   = 0.85,
        )

    driver = Driver(race_car, track, best_known_raceline)
    
    print(f'{race_car.name} - Simulated laptime = {laptime_str(driver.pr)}')

    nr_iterations = 0

    timer1 = Timer()
    timer2 = Timer()

    try:
        while True:
            nr_iterations += 1
            driver.try_new_line()

            if timer1.elapsed_time > 3:
                print(f"Laptime = {laptime_str(driver.pr)}  (iteration:{nr_iterations})")
                timer1.reset()
            
            if timer2.elapsed_time > 10:
                print(f'intermediate results saved to {filename_results=}')
                timer2.reset()
                save_results(df_track, driver.race_results(), filename_results)
                
    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')


    if not os.path.exists(PATH_RESULTS_):
        os.makedirs(PATH_RESULTS_)

    save_results(df_track, driver.race_results(), filename_results)

    print(f'{race_car.name} - Simulated laptime = {laptime_str(driver.pr)}')


if __name__ == '__main__':
    main()