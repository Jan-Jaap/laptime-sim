import geopandas
import pandas as pd
import numpy as np

from laptime_sim.geodataframe_operations import gdf_from_df, interpolate
geopandas.options.io_engine = "pyogrio"


import os
from track import Track

PATH_RESULTS_   = './simulated/'
PATH_TRACKS     = './tracks/'

if not os.path.exists(PATH_RESULTS_):
    os.makedirs(PATH_RESULTS_)

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

def get_best_known_raceline(df) -> np.ndarray:
    col_options = ['Race line','Optimized line','initial_position']
    for col in col_options:
        if col in df.columns:
            return df[col].values
    return None
    
def track_from_csv(filename: str, track_name:str) -> Track:
    print(f'Loading track data from {filename}')
    df = pd.read_csv(filename)
    return Track(
        name=track_name,
        geodataframe=gdf_from_df(df, crs=32631),
        best_line=get_best_known_raceline(df),
        min_clearance=0.85)

def track_from_parquet(filename: str, track_name:str) -> Track:
    print(f'Loading track from {filename}')
    gdf = geopandas.read_parquet(filename)
    return Track(
        name=track_name, 
        geodataframe=gdf,
        best_line=None,
        min_clearance=0.85
        )


def get_track_data(track_name, name_car) -> Track:

    filename = f'{PATH_RESULTS_}{name_car}_{track_name}_simulated.csv'
    if os.path.isfile(filename):
        return track_from_csv(filename, track_name)
    
    filename = f"{PATH_TRACKS}{track_name}.csv"
    if os.path.isfile(filename):
        return track_from_csv(filename, track_name)

    filename = f"{PATH_TRACKS}{track_name}.parquet"
    if os.path.isfile(filename):
        print(f'Loading geojson track from {filename}')
        gdf = geopandas.read_parquet(filename)
        return Track(
            name=track_name, 
            geodataframe=gdf,
            best_line=None,
            min_clearance=0.85
            )

    print('No track data found')
    

def save_results(data: np.ndarray, filename_results:str):

    results = pd.DataFrame(
            data = data,
            columns=(('inner_x','inner_y','inner_z','outer_x','outer_y','outer_z','race_line_position', 'distance', 'line_x', 'line_y', 'line_z', 'speed', 'time', 'a_lat', 'a_lon' ))
            ).rename(columns = OUTPUT_COLUMNS_NAMES)

    results.to_csv(PATH_RESULTS_+filename_results, index = None, header=True)


