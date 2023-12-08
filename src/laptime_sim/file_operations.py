import os
import geopandas
from pandas import DataFrame
from geopandas import GeoDataFrame

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


def load_trackdata_from_file(filename: str) -> GeoDataFrame | None:

    match filename:
        case s if s.endswith('.parquet'):
            return geopandas.read_parquet(filename)
        case s if s.endswith('.geojson'):
            return geopandas.read_file(filename)

    return None


def filename_iterator(path, extensions=SUPPORTED_FILETYPES):
    for filename in [os.path.join(path, s) for s in os.listdir(path) if s.endswith(extensions)]:
        yield filename


def strip_filename(filename: str) -> str:
    filename = os.path.basename(filename).replace('_simulated', '')
    return strip_extension(filename)


def strip_extension(path: str) -> str:
    return os.path.splitext(path)[0]


def find_raceline_filename(track_name, name_car) -> str:

    # first try to restart an existing simulation
    for filename in filename_iterator(PATH_RESULTS_, ('parquet')):
        match filename:
            case f if track_name in f and name_car in f:
                return f


def find_track_filename(track_name, path=PATH_TRACKS) -> str:
    # find track data from different file sources.
    for filename in filename_iterator(path, ('parquet')):
        match filename:
            case f if track_name in f:
                return f


def save_csv(df: DataFrame, filename_results: str):
    f = os.path.join(PATH_RESULTS_, os.path.basename(filename_results))
    df.to_csv(strip_extension(f)+'.csv', index=None, header=True)
