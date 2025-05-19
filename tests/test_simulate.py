import numpy as np
import laptime_sim
from main import CAR_LIST


from laptime_sim.simulate import simulate
from laptime_sim.simulate_old import simulate as simulate_old


def test_car_list():
    for car in laptime_sim.car_list(CAR_LIST):
        assert isinstance(car, laptime_sim.Car)


def test_sim(race_car, track):
    raceline = laptime_sim.Raceline(track=track)  # .simulate(car)
    line_coordinates = track.coordinates_from_position(raceline.line_position)

    sim_results = simulate(race_car, line_coordinates, track.slope)
    sim_results_old = simulate_old(race_car, line_coordinates, track.slope)

    np.testing.assert_array_equal(sim_results.line_coordinates, sim_results_old.line_coordinates)
    np.testing.assert_array_equal(sim_results.ds, sim_results_old.ds)
    np.testing.assert_array_equal(sim_results.Nk, sim_results_old.Nk)
    np.testing.assert_almost_equal(sim_results.speed, sim_results_old.speed)
    np.testing.assert_almost_equal(sim_results.dt, sim_results_old.dt)
