# import pytest
import numpy as np
from laptime_sim.car import Car
from laptime_sim import race_lap
from laptime_sim import file_operations
import pandas as pd

from tracksession import TrackSession

PATH_TRACK = "./tracks/20191211_Bilsterberg.parquet"
PATH_CAR = "./cars/Peugeot_205RFS.toml"

track_layout = file_operations.load_trackdata_from_file(PATH_TRACK)
race_car = Car.from_toml(PATH_CAR)
track_session = TrackSession(track_layout=track_layout, car=race_car)


def test_mag():
    test_data = np.array([[3.0, 4.0]])
    assert race_lap.mag(test_data) is not None
    assert race_lap.mag(test_data) == 5.0


def test_full_race():
    race_output = race_lap.sim(track_session=track_session)
    assert isinstance(race_output, float)
    assert race_output > 0


def test_full_race_verbose():
    race_output = race_lap.sim(track_session=track_session, verbose=True)
    assert isinstance(race_output, pd.DataFrame)
