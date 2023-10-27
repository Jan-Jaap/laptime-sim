import json
import os
import time
import itertools

import geopandas as gpd
import numpy as np
import pandas as pd

from car import Car
from track import Track
import race_lap
from geopanda_utils import gdf_from_df

from icecream import install
install()

PATH_RESULTS_   = './simulated/'
PATH_TRACKS     = './tracks/'
PATH_CARS       = './cars/'

NAME_CAR = "Peugeot_205RFS"
NAME_TRACK = "2020_zandvoort"
# NAME_TRACK = "20191030_Circuit_Zandvoort"

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

class Timer:
    def __init__(self):
        self.time = time.time()
    def reset(self):
        self.time = time.time()
    @property
    def elapsed_time(self):
        return time.time() - self.time


def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)


def save_results(data: np.ndarray, filename_results):
       
    results = pd.DataFrame(
            data = data,
            columns=(('outer_x','outer_y','outer_z','inner_x','inner_y','inner_z','race_line_position', 'distance', 'line_x', 'line_y', 'line_z', 'speed', 'time', 'a_lat', 'a_lon' ))
            ).rename(columns = OUTPUT_COLUMNS_NAMES)

    results.to_csv(filename_results, index = None, header=True)


def get_best_known_raceline(df) -> np.ndarray:
    col_options = ['Race line','Optimized line','initial_position']
    for col in col_options:
        if col in df.columns:
            return df[col].values
    return None
    

def get_track_data(track_name) -> Track:
       
    filename = f'{PATH_RESULTS_}{NAME_CAR}_{track_name}_simulated.csv'
    if os.path.isfile(filename):
        print(f'Loading track data from {filename}')
        df = pd.read_csv(filename)
        return Track(
            name=track_name,
            geodataframe=gdf_from_df(df, crs=32631),
            best_line=get_best_known_raceline(df),
            min_clearance=0.85)
            

    
    filename = f"{PATH_TRACKS}{track_name}.csv"
    if os.path.isfile(filename):
        print(f'Loading csv track from {filename}')
        df = pd.read_csv(filename)
        return Track(
            name=track_name,
            geodataframe=gdf_from_df(df, crs=32631),
            best_line=get_best_known_raceline(df),
            min_clearance=0.85)
            

    
    filename = f"{PATH_TRACKS}{track_name}.parquet"
    if os.path.isfile(filename):
        print(f'Loading geojson track from {filename}')
        gdf = gpd.read_parquet(filename)
        return Track(
            name=track_name, 
            geodataframe=gdf,
            best_line=None,
            min_clearance=0.85
            )

    print('No track data found')
    

def main():

    filename_car_properties = f"{PATH_CARS}{NAME_CAR}.json"
    filename_results = f'{PATH_RESULTS_}{NAME_CAR}_{NAME_TRACK}_simulated.csv'

    with open(filename_car_properties, 'r') as fp:
        race_car = Car(**json.load(fp))

    track = get_track_data(NAME_TRACK)
    if track is None: return
    
    best_time = race_lap.race(track=track, car=race_car)

    
    
    
    print(f'{race_car.name} - Simulated laptime = {laptime_str(best_time)}')


    timer1 = Timer()
    timer2 = Timer()

    try:
        for nr_iterations in itertools.count():
            new_line = race_lap.get_new_line(track=track)
            laptime = race_lap.race(track=track, car=race_car, raceline=new_line)
            
                
            if laptime < best_time:
                best_time = laptime
                track.best_line = new_line
                

            if timer1.elapsed_time > 3:
                print(f"Laptime = {laptime_str(laptime)}  (iteration:{nr_iterations})")
                timer1.reset()
            
            if timer2.elapsed_time > 10:
                print(f'intermediate results saved to {filename_results=}')
                timer2.reset()
                save_results(race_lap.race(track=track, car=race_car, verbose=True), filename_results)
                
    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')


    if not os.path.exists(PATH_RESULTS_):
        os.makedirs(PATH_RESULTS_)

    save_results(race_lap.race(track=track, car=race_car, verbose=True), filename_results)

    print(f'{race_car.name} - Simulated laptime = {laptime_str(best_time)}')


if __name__ == '__main__':
    main()
