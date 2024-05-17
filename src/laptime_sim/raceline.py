import os

import functools
import numpy as np
from geopandas import GeoDataFrame, read_parquet

from shapely import LineString
from dataclasses import dataclass, field

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults
from laptime_sim.track import Track
from laptime_sim.simulate import RacelineSimulator


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
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class Raceline:
    track: Track
    car: Car
    simulator: RacelineSimulator = field(default_factory=RacelineSimulator)
    # filename_results: os.PathLike | str = None
    line_pos: np.ndarray = None
    heatmap: np.ndarray = None
    clearance_meter: float = 0.85
    progress: float = 1.0
    best_time: float = None

    def __post_init__(self):

        if self.line_pos is None:
            # TODO: here we could use a line_pos from a different car or run
            self.line_pos = np.zeros_like(self.track.width) + 0.5

        if self.simulator is not None and self.best_time is None:
            self.best_time = self.simulate().laptime

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_pos)

    @classmethod
    def from_geodataframe(cls, results: GeoDataFrame, all_cars, all_tracks):

        track = [track for track in all_tracks if track.name == results.iloc[0].track_name][0]
        car = [car for car in all_cars if car.name == results.iloc[0].car][0]

        results = results.to_crs(track.crs)
        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)

        simulator = RacelineSimulator()

        return cls(
            track=track,
            car=car,
            simulator=simulator,
            line_pos=track.parametrize_line_coords(line_coords),
            best_time=results.iloc[0].best_time,
        )

    def load_line(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} doesn't exist")
        results = read_parquet(filename)
        results = results.to_crs(self.track.crs)

        assert self.track.name == results.iloc[0].track_name
        assert self.car.name == results.iloc[0].car
        self.best_time = results.iloc[0].best_time
        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)
        self.line_pos = self.track.parametrize_line_coords(line_coords)

        return self

    def save_line(self, filename) -> None:
        """
        Saves the results to a parquet file.

        Parameters:
            filename: str - The name of the output file to save the results to.

        Returns:
            None
        """
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
        """
        Property method that formats the best lap time in hours, minutes, and seconds.
        Returns a formatted string of the best lap time.
        """
        if self.best_time:
            return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

    @functools.cached_property
    def _position_clearance(self):
        return self.track.normalize_distance(self.clearance_meter)

    def simulate(self, line_pos: np.ndarray = None) -> SimResults:

        if line_pos is None:
            return self.simulator.run(
                car=self.car,
                line_coordinates=self.track.line_coords(self.line_pos),
                slope=self.track.slope,
            )

        sim_results = self.simulator.run(
            car=self.car,
            line_coordinates=self.track.line_coords(line_pos),
            slope=self.track.slope,
        )

        if sim_results.laptime < self.best_time:
            improvement = self.best_time - sim_results.laptime
            deviation = np.abs(self.line_pos - line_pos)
            max_deviation = max(deviation)
            if max_deviation > 0:
                self.heatmap += deviation / max_deviation * improvement * 1e3
            self.best_time = sim_results.laptime
            self.progress += improvement
            self.line_pos = line_pos

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= F_ANNEAL  # slowly to zero

        return sim_results

    def simulate_new_line(self, line_param=None) -> None:
        if line_param is None:
            line_param = LineParameters.from_heatmap(self.heatmap)

        new_line = self.get_new_line(line_param=line_param)

        return self.simulate(new_line)

    def get_new_line(self, line_param: LineParameters):
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


# def optimize_raceline(
#     raceline: Raceline,
#     display_callback: Callable[[Raceline, int, bool], None],
#     filename_save=None,
#     tolerance=0.005,
# ):

#     timer1 = Timer()
#     timer2 = Timer()

#     raceline.save_line(filename_save)

#     if not display_callback:

#         def display_callback(*args, **kwargs):
#             return None

#     display_callback(raceline, 0, saved=True),

#     for nr_iterations in itertools.count():

#         raceline.simulate_new_line()

#         if raceline.progress < tolerance:
#             break

#         if timer1.elapsed_time > 3:
#             display_callback(raceline, nr_iterations, saved=False)
#             timer1.reset()

#         if timer2.elapsed_time > 30:
#             display_callback(raceline, nr_iterations, saved=True)
#             raceline.save_line(filename_save)
#             timer2.reset()

#     raceline.save_line(filename_save)
#     display_callback(raceline, nr_iterations, saved=True)
#     return raceline


# class Timer:
#     def __init__(self):
#         self.time = time()

#     def reset(self):
#         self.time = time()

#     @property
#     def elapsed_time(self):
#         return time() - self.time
