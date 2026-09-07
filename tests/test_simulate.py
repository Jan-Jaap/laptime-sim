import numpy as np

import laptime_sim
from laptime_sim.simulate import simulate
from main import CAR_LIST


def test_car_list():
    assert CAR_LIST
    for car in CAR_LIST:
        assert isinstance(car, laptime_sim.Car)


def test_sim(race_car, track):
    raceline = laptime_sim.Raceline(track=track)
    line_coordinates = raceline.get_coordinates()

    sim_results = simulate(race_car, line_coordinates, track.slope)

    assert sim_results.speed.shape == sim_results.dt.shape == track.slope.shape
    assert np.all(np.isfinite(sim_results.speed))
    assert np.all(sim_results.speed > 0)
    assert sim_results.laptime > 0
