
import os
from icecream import ic
import pandas as pd
from laptime_sim.geodataframe_operations import df_to_geo

from laptime_sim import get_best_known_raceline

from tracksession import Track

PATH_TRACKS = './tracks/'
NAME_TRACK = '20191030_Circuit_Zandvoort'


filename = f"{PATH_TRACKS}{NAME_TRACK}.csv"
if os.path.isfile(filename):
    print(f'Loading csv track from {filename}')
    df = pd.read_csv(filename)
    track = Track(
        name=NAME_TRACK,
        geodataframe=df_to_geo(df, crs=32631),
        best_line=get_best_known_raceline(df),
        min_clearance=0.85)
            

inner = track.geodataframe.geometry.iloc[[0]]
ic(type(inner.get_coordinates()))
ic(inner.get_coordinates().to_numpy())