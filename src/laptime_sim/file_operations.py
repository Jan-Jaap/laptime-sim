import os
import geopandas
import pandas as pd

from geodataframe_operations import df_to_geo

geopandas.options.io_engine = "pyogrio"

PATH_RESULTS_ = './simulated/'
PATH_TRACKS = './tracks/'
SUPPORTED_FILETYPES = ('.csv', '.geojson', '.parquet')

if not os.path.exists(PATH_RESULTS_):
    os.makedirs(PATH_RESULTS_)


def get_trackname_from_filename(filename: str) -> str:
    for n in filename_iterator(PATH_TRACKS):
        name_track = strip_filename(n)
        if name_track in filename:
            # name_track = strip_filename(filename_track)
            return name_track
    return None


def load_trackdata_from_file(filename: str) -> geopandas.GeoSeries:

    match filename:
        case s if s.endswith('.csv'):
            df = pd.read_csv(filename)
            return df_to_geo(df)
        case s if s.endswith('.geojson'):
            return geopandas.read_file(filename)
        case s if s.endswith('.parquet'):
            return geopandas.read_parquet(filename)
    return None


def filename_iterator(path, extensions=SUPPORTED_FILETYPES):
    tracks_in_dir = [os.path.join(path, s) for s in os.listdir(path) if s.endswith(extensions)]
    for track_name in tracks_in_dir:
        yield track_name


def strip_filename(filename: str) -> str:
    filename = os.path.basename(filename).replace('_simulated', '')
    return strip_extension(filename)


def strip_extension(path: str) -> str:
    return os.path.splitext(path)[0]


def find_filename(track_name, name_car) -> str:

    # first try to restart an existing simulation
    for filename in filename_iterator(PATH_RESULTS_, ('parquet')):
        match filename:
            case f if track_name in f and name_car in f:
                return f

    # find track data from different file sources.
    for filename in filename_iterator(PATH_TRACKS, ('parquet')):
        match filename:
            case f if track_name in f and name_car is None:
                return f

    print('No track data found')


def save_csv(df: pd.DataFrame, filename_results: str):
    f = os.path.join(PATH_RESULTS_, os.path.basename(filename_results))
    df.to_csv(strip_extension(f)+'.csv', index=None, header=True)


def save_parquet(track_layout, filename_results):
    f = os.path.join(PATH_RESULTS_, os.path.basename(filename_results))
    geopandas.GeoDataFrame(geometry=track_layout).to_parquet(strip_extension(f)+'.parquet')


def load_parquet(track_dir):
    return geopandas.read_parquet(track_dir)
