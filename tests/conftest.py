import pytest
import laptime_sim
from main import PATH_CARS, PATH_TRACKS


@pytest.fixture(scope="module")
def car_list() -> list[laptime_sim.Car]:
    return laptime_sim.car_list(PATH_CARS)


@pytest.fixture(scope="module")
def track_list() -> list[laptime_sim.Track]:
    return laptime_sim.track_list(PATH_TRACKS)


@pytest.fixture(scope="module")
def race_car(car_list) -> laptime_sim.Car:
    return car_list[0]


@pytest.fixture(scope="module")
def track(track_list) -> laptime_sim.Track:
    return track_list[0]
