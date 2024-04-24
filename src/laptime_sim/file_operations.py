import os
import geopandas

geopandas.options.io_engine = "pyogrio"

PATH_RESULTS = "./simulated/"
PATH_TRACKS = "./tracks/"

if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)


def strip_filename(filename: str) -> str:
    filename = os.path.basename(filename).replace("_simulated", "")
    return strip_extension(filename)


def strip_extension(path: str) -> str:
    return os.path.splitext(path)[0]
