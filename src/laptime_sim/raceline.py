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
    simulate: Callable[[Car, np.ndarray, np.ndarray], SimResults] = simulate
    # line_position: np.ndarray = None
    heatmap: np.ndarray = None
    progress_rate: float = 1.0
    _clearance_meter: float = 0.85
    _results_path = None

    def __post_init__(self):
        self.heatmap = np.ones_like(self.line_position)

    @functools.cached_property
    def best_time(self) -> float:
        return self.run_sim(self.car).laptime

    @functools.cached_property
    def line_position(self) -> np.ndarray:
        return self.track.initialize_line(smoothing_window=40, poly_order=5)

    @classmethod
    def from_geodataframe(cls, data: GeoDataFrame, all_cars, all_tracks):
        track: Track = [track for track in all_tracks if track.name == data.iloc[0].track_name][0]
        car: Car = [car for car in all_cars if car.name == data.iloc[0].car][0]
        data = data.to_crs(track.crs)
        line_coords = data.get_coordinates(include_z=False).to_numpy(na_value=0)

        return cls(
            track=track,
            car=car,
            line_position=track.parameterize_line_coordinates(line_coords),
        )

    @property
    def filename(self) -> Path:
        return Path(f"{self.car.file_name}_{self.track.name}_simulated.parquet")

    def load_line(self, file_path: Path) -> bool:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exist")

        results = read_parquet(file_path)
        results = results.to_crs(self.track.crs)

        assert self.track.name == results.iloc[0].track_name
        assert self.car.name == results.iloc[0].car

        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)
        self.line_position = self.track.parameterize_line_coordinates(line_coords)
        self.update()
        return True

    def save_line(self, file_path: Path) -> None:
        if file_path is None:
            raise FileNotFoundError("no output filename provided")
        self.get_dataframe().to_parquet(file_path)

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
    def position_clearance(self):
        return self.track.normalize_distance(self._clearance_meter)

    def update(self, new_line_position: np.ndarray | None = None) -> bool:
        """
        Simulates a lap on the track with the current line position or a new line position.
        If new_line_position is provided it will be used instead of the current line position.power

        Returns True if the new line position is better than the current one, False otherwise.
        """

        sim_results = self.run_sim(self.car, new_line_position)

        if new_line_position is None:
            self.best_time = sim_results.laptime
            return True

        if sim_results.laptime < self.best_time:
            improvement = self.best_time - sim_results.laptime
            deviation = np.abs(self.line_position - new_line_position)
            deviation /= np.max(deviation)

            self.heatmap += deviation * improvement * 1e3
            self.best_time = sim_results.laptime
            self.progress_rate += improvement
            self.line_position = new_line_position
            return True

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress_rate *= F_ANNEAL  # slowly to zero

        return False

    def run_sim(self, car: Car, new_line_position: np.ndarray | None = None) -> SimResults:
        line_position = new_line_position if new_line_position is not None else self.line_position
        return self.simulate(
            car=car,
            line_coordinates=self.track.line_coordinates(line_position),
            slope=self.track.slope,
        )

    def simulate_new_line(self) -> None:
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
            a_min=self.position_clearance,
            a_max=1 - self.position_clearance,
        )
        self.update(new_line_position)
