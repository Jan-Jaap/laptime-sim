import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from geopandas import GeoDataFrame, read_parquet
from shapely import LineString

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults
from laptime_sim.simulate import simulate
from laptime_sim.track import Track

MAX_DEVIATION_LENGTH = 60
MAX_DEVIATION = 0.1
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class Raceline:
    track: Track
    car: Car
    simulator: Callable[[Car, np.ndarray, np.ndarray], SimResults] = simulate
    line_position: np.ndarray = None
    heatmap: np.ndarray = None
    clearance_meter: float = 0.85
    progress: float = 1.0
    best_time: float = None

    def __post_init__(self):
        if self.line_position is None:
            # TODO: here we could use a line_pos from a different car or run
            self.line_position = self.track.initialize_line(smoothing_window=40, poly_order=5)

        if self.best_time is None:
            self.simulate()

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_position)

    @classmethod
    def from_geodataframe(cls, data: GeoDataFrame, all_cars, all_tracks):
        track: Track = [track for track in all_tracks if track.name == data.iloc[0].track_name][0]
        car: Car = [car for car in all_cars if car.name == data.iloc[0].car][0]
        data = data.to_crs(track.crs)
        line_coords = data.get_coordinates(include_z=False).to_numpy(na_value=0)

        # simulator = RacelineSimulator()

        return cls(
            track=track,
            car=car,
            # simulator=simulator,
            line_position=track.parameterize_line_coordinates(line_coords),
            # best_time=data.iloc[0].best_time,
        )

    def filename(self, PATH_RESULTS) -> Path:
        return Path(PATH_RESULTS, f"{self.car.file_name}_{self.track.name}_simulated.parquet")

    # filename_results = Path(PATH_RESULTS, f"{car.file_name}_{track.name}_simulated.parquet")

    def load_line(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} doesn't exist")
        results = read_parquet(filename)
        results = results.to_crs(self.track.crs)

        assert self.track.name == results.iloc[0].track_name
        assert self.car.name == results.iloc[0].car
        # self.best_time = results.iloc[0].best_time
        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)
        self.line_position = self.track.parameterize_line_coordinates(line_coords)
        self.best_time = self.simulate().laptime
        return self

    def save_line(self, filename) -> None:
        if filename is None:
            raise FileNotFoundError("no output filename provided")
        self.get_dataframe().to_parquet(filename)

    def get_dataframe(self) -> GeoDataFrame:
        geom = LineString(self.track.line_coordinates(self.line_position).tolist())
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

    def simulate(self, new_line_position: np.ndarray | None = None) -> SimResults:
        """
        Simulates a lap on the track with the current line position or a new line position.
        If new_line_position is provided it will be used instead of the current line position.

        Returns a SimResults object containing the results of the simulation.
        """
        line_coords = self.track.line_coordinates(new_line_position if new_line_position is not None else self.line_position)
        sim_results: SimResults = self.simulator(car=self.car, line_coordinates=line_coords, slope=self.track.slope)

        if new_line_position is None:
            self.best_time = sim_results.laptime
            return sim_results

        if sim_results.laptime < self.best_time:
            improvement = self.best_time - sim_results.laptime
            deviation = np.abs(self.line_position - new_line_position)
            deviation /= np.max(deviation)

            self.heatmap += deviation * improvement * 1e3
            self.best_time = sim_results.laptime
            self.progress += improvement
            self.line_position = new_line_position

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= F_ANNEAL  # slowly to zero
        return sim_results

    def simulate_new_line(self) -> SimResults:
        location = np.random.choice(len(self.heatmap), p=self.heatmap / sum(self.heatmap))
        length = np.random.randint(1, MAX_DEVIATION_LENGTH)
        deviation = np.random.randn() * MAX_DEVIATION

        line_adjust = 1 - np.cos(np.linspace(0, 2 * np.pi, length))
        position = np.zeros_like(self.line_position)
        position[:length] = line_adjust * deviation
        position = np.roll(position, location - length // 2)
        new_line_position = self.line_position + position / self.track.width
        new_line_position = np.clip(
            new_line_position,
            a_min=self._position_clearance,
            a_max=1 - self._position_clearance,
        )

        return self.simulate(new_line_position)
