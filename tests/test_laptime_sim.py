import itertools
from pathlib import Path
import pytest

import laptime_sim
from laptime_sim.main import PATH_CARS, PATH_TRACKS


@pytest.fixture
def car_list() -> list[laptime_sim.Car]:
    return laptime_sim.car_list(PATH_CARS)


@pytest.fixture
def car(car_list) -> laptime_sim.Car:
    return car_list[0]


@pytest.fixture
def track_list() -> list[laptime_sim.Track]:
    return laptime_sim.track_list(PATH_TRACKS)


@pytest.fixture
def track(track_list) -> laptime_sim.Track:
    return track_list[0]


@pytest.fixture
def raceline(track, car):
    return laptime_sim.Raceline(track=track, car=car)


def test_car_loading(car_list):
    assert isinstance(car_list, list)
    for car in car_list:
        assert isinstance(car, laptime_sim.Car)


def test_track_loading(track_list):
    assert isinstance(track_list, list)
    for track in track_list:
        assert isinstance(track, laptime_sim.Track)


def test_raceline_creation(track_list, car_list):
    for car, track in itertools.product(car_list, track_list):
        raceline = laptime_sim.Raceline(track=track, car=car)
        assert isinstance(raceline, laptime_sim.Raceline)


def test_raceline_filename(raceline):
    assert isinstance(raceline.filename, Path)
