import pytest
import laptime_sim
from laptime_sim.main import PATH_CARS, PATH_TRACKS


@pytest.mark.parametrize("track", laptime_sim.track_list(PATH_TRACKS), ids=lambda track: track.name)
# @pytest.mark.parametrize("car", laptime_sim.car_list(PATH_CARS), ids=lambda car: car.name)
def test_simulate(track: laptime_sim.Track):
    raceline = laptime_sim.Raceline(track=track)  # .simulate(car)
    assert isinstance(raceline, laptime_sim.Raceline)
    # sim_results = raceline.simulate(car)
    # assert isinstance(sim_results, laptime_sim.SimResults)
    # assert len(sim_results.distance) == len(track.width)
    # assert len(sim_results.speed_kph) == len(track.width)
