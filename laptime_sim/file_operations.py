import os
import geopandas
import pandas as pd
import numpy as np

from geodataframe_operations import df_to_geo
geopandas.options.io_engine = "pyogrio"

from track import TrackSession

PATH_RESULTS_   = './simulated/'
PATH_TRACKS     = './tracks/'

if not os.path.exists(PATH_RESULTS_):
    os.makedirs(PATH_RESULTS_)

OUTPUT_COLUMNS_NAMES = dict(
    distance            = 'Distance (m)',
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
    
def track_from_csv(filename: str) -> TrackSession:
    df = pd.read_csv(filename)
    return TrackSession(
        track_layout=df_to_geo(df, crs=32631),
        best_line=get_best_known_raceline(df),
        min_clearance=0.85)

def track_from_parquet(filename: str) -> TrackSession:
    gdf = geopandas.read_parquet(filename)
    return TrackSession(
        track_layout=gdf,
        best_line=None,
        min_clearance=0.85
        )


def get_track_data(track_name, name_car) -> TrackSession:

    filename = f'{PATH_RESULTS_}{name_car}_{track_name}_simulated.csv'
    if os.path.isfile(filename):
        print(f'Loading track data from {filename}')
        return track_from_csv(filename)
    
    filename = f"{PATH_TRACKS}{track_name}.parquet"
    if os.path.isfile(filename):
        print(f'Loading track from {filename}')
        return track_from_parquet(filename)
    
    filename = f"{PATH_TRACKS}{track_name}.csv"
    if os.path.isfile(filename):
        print(f'Loading track data from {filename}')
        return track_from_csv(filename)

    print('No track data found')
    

def save_results(pd: pd.DataFrame, filename_results:str):

    results = pd.rename(columns = OUTPUT_COLUMNS_NAMES)
    results.to_csv(PATH_RESULTS_+filename_results, index = None, header=True)


