import os
from time import time
import functools
import numpy as np
from geopandas import GeoDataFrame, read_parquet

from shapely import LineString
from dataclasses import dataclass

from laptime_sim.car import Car
from laptime_sim.track import Track
from laptime_sim.simulate import RacelineSimulator

import itertools
from typing import Callable


@dataclass(frozen=True)
class LineParameters:
    location: int
    length: int
    deviation: float

    @classmethod
    def from_heatmap(cls, p):
        location = np.random.choice(len(p), p=p / sum(p))
        length = np.random.randint(1, 60)
        deviation = np.random.randn() / 10
        return cls(location, length, deviation)


# annealing factor
f_anneal = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class Raceline:
    track: Track
    car: Car
    simulator: RacelineSimulator
    # filename_results: os.PathLike | str = None
    line_pos: np.ndarray = None
    heatmap: np.ndarray = None
    clearance_meter: float = 0.85
    progress: float = 1.0
    best_time: float = None

    def __post_init__(self):

        if self.line_pos is None:
            # TODO: here we could use a line from a different car or run
            self.line_pos = np.zeros_like(self.track.width) + 0.5

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_pos)

    @classmethod
    def from_geodataframe(cls, results: GeoDataFrame, path_tracks, path_cars):

        track = Track.from_name(results.iloc[0].track_name, path_tracks)
        results = results.to_crs(track.crs)
        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)

        car = Car.from_name(results.iloc[0].car, path_cars)
        simulator = RacelineSimulator(car)

        # results = results.to_crs(self.track.crs)
        # results = results.iloc[0]

        return cls(
            track=track,
            car=car,
            simulator=simulator,
            line_pos=track.parametrize_line_coords(line_coords),
            best_time=results.iloc[0].best_time,
        )

    def load_results(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} doesn't exist")
        results = read_parquet(filename)
        results = results.to_crs(self.track.crs)

        assert self.track.name == results.iloc[0].track_name
        self.best_time = results.iloc[0].best_time
        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)
        self.line_pos = self.track.parametrize_line_coords(line_coords)

        return self

    def save_results(self, filename) -> None:
        if filename is None:
            raise FileNotFoundError("no output filename provided")
        self.get_dataframe().to_parquet(filename)

    def get_dataframe(self) -> GeoDataFrame:
        geom = LineString(self.track.line_coords(self.line_pos).tolist())
        data = dict(
            track_name=self.track.name,
            car=self.car.name,
            best_time=self.best_time,
        )
        return GeoDataFrame.from_dict(data=[data], geometry=[geom], crs=self.track.crs).to_crs(epsg=4326)

    @property
    def best_time_str(self):
        return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

    @functools.cached_property
    def _position_clearance(self):
        return self.track.normalize_distance(self.clearance_meter)

    def simulate(self):
        line_coordinates = self.track.line_coords(self.line_pos)
        sim_results = self.simulator.run(line_coordinates=line_coordinates, slope=self.track.slope)
        if self.best_time != sim_results.laptime:
            self.best_time = sim_results.laptime
        return sim_results

    def simulate_new_line(self, new_line=None) -> None:

        if new_line is None:
            new_line = self.line_pos

        sim_results = self.simulator.run(line_coordinates=self.track.line_coords(new_line), slope=self.track.slope)

        if self.best_time is None:
            self.best_time = sim_results.laptime

        if sim_results.laptime < self.best_time:
            improvement = self.best_time - sim_results.laptime
            self.best_time = sim_results.laptime
            self.line_pos = new_line
            deviation = np.abs(self.line_pos - new_line)
            max_deviation = max(deviation)
            if max_deviation > 0:
                self.heatmap += deviation / max_deviation * improvement * 1e3
            self.progress += improvement

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= f_anneal  # slowly to zero
        return sim_results

    def get_new_line(self):
        line_param = LineParameters.from_heatmap(self.heatmap)
        line_adjust = 1 - np.cos(np.linspace(0, 2 * np.pi, line_param.length))
        position = np.zeros_like(self.line_pos)
        position[: line_param.length] = line_adjust * line_param.deviation
        position = np.roll(position, line_param.location - line_param.length // 2)
        test_line = self.line_pos + position / self.track.width
        return np.clip(
            test_line,
            a_min=self._position_clearance,
            a_max=1 - self._position_clearance,
        )


def optimize_raceline(
    raceline: Raceline,
    display_callback: Callable[[Raceline, int, bool], None],
    filename_save=None,
    tolerance=0.005,
):

    timer1 = Timer()
    timer2 = Timer()

    raceline.simulate()
    raceline.save_results(filename_save)

    if not display_callback:

        def display_callback(*args, **kwargs):
            return None

    display_callback(raceline, 0, saved=True),

    for nr_iterations in itertools.count():

        new_line = raceline.get_new_line()
        raceline.simulate_new_line(new_line=new_line)

        if raceline.progress < tolerance:
            break

        if timer1.elapsed_time > 3:
            display_callback(raceline, nr_iterations, saved=False)
            timer1.reset()

        if timer2.elapsed_time > 30:
            display_callback(raceline, nr_iterations, saved=True)
            raceline.save_results(filename_save)
            timer2.reset()

    raceline.save_results(filename_save)
    display_callback(raceline, nr_iterations, saved=True)
    return raceline


class Timer:
    def __init__(self):
        self.time = time()

    def reset(self):
        self.time = time()

    @property
    def elapsed_time(self):
        return time() - self.time
