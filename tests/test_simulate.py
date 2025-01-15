import pytest
from laptime_sim.main import PATH_CARS, PATH_TRACKS
import laptime_sim
import laptime_sim.simresults
from laptime_sim.simulate import simulate


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
def raceline(track: laptime_sim.Track, car: laptime_sim.Car):
    return laptime_sim.Raceline(track=track, car=car, simulate=simulate)


def test_raceline_creation(raceline: laptime_sim.Raceline):
    assert isinstance(raceline, laptime_sim.Raceline)


def test_simulate(track: laptime_sim.Track, car: laptime_sim.Car):
    raceline = laptime_sim.Raceline(track=track, car=car, simulate=simulate)
    sim_results = raceline.run_sim(car)
    assert isinstance(sim_results, laptime_sim.simresults.SimResults)
    assert len(sim_results.distance) == len(track.width)
    assert len(sim_results.speed_kph) == len(track.width)
