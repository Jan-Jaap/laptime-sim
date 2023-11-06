import os
import geopandas
import pandas as pd
import numpy as np

from geodataframe_operations import df_to_geo
geopandas.options.io_engine = "pyogrio"

PATH_RESULTS_   = './simulated/'
PATH_TRACKS     = './tracks/'
SUPPORTED_FILETYPES = ('.csv', '.geojson', '.parquet')

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


def load_trackdata_from_file(file_name: str) -> tuple[geopandas.GeoSeries, np.ndarray | None]:
    
    match file_name:
        case s if s.endswith('.csv'):
            df = pd.read_csv(file_name)
            return df_to_geo(df), get_best_known_raceline(df)
        case s if s.endswith('.geojson'):
            return geopandas.read_file(file_name).geometry, None
        case s if  s.endswith('.parquet'):
            return geopandas.read_parquet(file_name).geometry, None
    return None

def filename_iterator(path=PATH_TRACKS) -> str:
    tracks_in_dir = [ os.path.join(path, s) for s in os.listdir(path) if s.endswith(SUPPORTED_FILETYPES)]
    for track_name in tracks_in_dir:
        yield track_name
        
def find_filename(track_name, name_car) -> str:
    
    #first try to restart an existing simulation
    for filename in filename_iterator(PATH_RESULTS_):
        match filename:
            case s if track_name in s and name_car in s and '.csv' in s:
                return s

    #find track data from different file sources.
    for filename in filename_iterator():
        match filename:
            case s if track_name in s:
                return s

    print('No track data found')
    

def save_results(pd: pd.DataFrame, filename_results:str):

    results = pd.rename(columns = OUTPUT_COLUMNS_NAMES)
    results.to_csv(PATH_RESULTS_+filename_results, index = None, header=True)


