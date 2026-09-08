import pytest

import laptime_sim
from laptime_sim.registry import cars, tracks

CAR_LIST = cars()
TRACK_LIST = tracks()


@pytest.fixture(scope="module")
def car_list() -> list[laptime_sim.Car]:
    return CAR_LIST


@pytest.fixture(scope="module")
def track_list() -> list[laptime_sim.Track]:
    return TRACK_LIST


@pytest.fixture(scope="module")
def race_car(car_list) -> laptime_sim.Car:
    return car_list[0]


@pytest.fixture(scope="module")
def track(track_list) -> laptime_sim.Track:
    return track_list[0]
