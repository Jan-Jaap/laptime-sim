# import pytest
import numpy as np
from laptime_sim.car import Car
import laptime_sim.race_lap as race_lap
import laptime_sim.file_operations as file_operations
import pandas as pd
from icecream import ic

PATH_TRACK = "./tracks/20191211_Bilsterberg.parquet"
PATH_CAR = "./cars/Peugeot_205RFS.json"

test_track =  file_operations.track_from_parquet(PATH_TRACK)
race_car = Car.from_file(PATH_CAR)

def test_mag():
    test_data = np.array([[3.0, 4.0]])
    assert race_lap.mag(test_data) != None
    assert race_lap.mag(test_data) == 5.0


def test_full_race():
    race_output =  race_lap.race(track=test_track, car=race_car)
    assert isinstance(race_output, float) and race_output > 0
    
def test_full_race_verbose():
    race_output =  race_lap.race(track=test_track, car=race_car, verbose=True)
    assert isinstance(race_output, pd.DataFrame)