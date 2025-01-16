import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

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
    best_time: float = np.inf
    heatmap: np.ndarray = None
    progress_rate: float = 1.0
    _line_position: np.ndarray = None
    _clearance_meter: float = 0.85
    _results_path = None

    def __post_init__(self):
        self.heatmap = np.ones_like(self.line_position)

    @property
    def line_position(self) -> np.ndarray:
        if self._line_position is None:
            self._line_position = self.track.initialize_line(smoothing_window=40, poly_order=5)
        return self._line_position

    @line_position.setter
    def line_position(self, value):
        self._line_position = self.clip_line(value)

    def best_time_str(self) -> str:
        return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

    @classmethod
    def from_geodataframe(cls, data: GeoDataFrame, track: Track) -> Self:
        data = data.to_crs(track.crs)
        line_coords = data.get_coordinates(include_z=False).to_numpy(na_value=0)
        raceline = cls(track=track)
        raceline.line_position = track.parameterize_line_coordinates(line_coords)
        return raceline

    def filename(self, car_name) -> Path:
        return Path(f"{car_name}_{self.track.name}_simulated.parquet")

    def load_line(self, file_path: Path) -> Self:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exist")

        results = read_parquet(file_path)
        results = results.to_crs(self.track.crs)

        assert self.track.name == results.iloc[0].track_name
        # assert car.name == results.iloc[0].car

        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)
        self.line_position = self.track.parameterize_line_coordinates(line_coords)
        return self

    def save_line(self, file_path: Path, car_name: str) -> None:
        if file_path is None:
            raise FileNotFoundError("no output filename provided")

        geom = LineString(self.track.line_coordinates(self.line_position).tolist())
        data = dict(track_name=self.track.name, car=car_name)
        GeoDataFrame.from_dict(data=[data], geometry=[geom], crs=self.track.crs).to_crs(epsg=4326).to_parquet(file_path)

    @functools.cached_property
    def position_clearance(self):
        return self.track.normalize_distance(self._clearance_meter)

    def clip_line(self, line_position: np.ndarray | None = None) -> np.ndarray:
        if line_position is None:
            line_position = self.line_position
        return np.clip(line_position, a_min=self.position_clearance, a_max=1 - self.position_clearance)

    def simulate(self, car: Car) -> SimResults:
        sim_results = run_sim(self.track, car, self.line_position)

        if sim_results.laptime < self.best_time:
            self.best_time = sim_results.laptime
        return sim_results

    def simulate_new_line(self, car: Car) -> None:
        location = np.random.choice(len(self.heatmap), p=self.heatmap / sum(self.heatmap))
        length = np.random.randint(1, MAX_DEVIATION_LENGTH)
        deviation = np.random.randn() * MAX_DEVIATION

        line_adjust = 1 - np.cos(np.linspace(0, 2 * np.pi, length))
        position = np.zeros_like(self.line_position)
        position[:length] = line_adjust * deviation
        position = np.roll(position, location - length // 2)
        new_line_position = self.clip_line(self.line_position + position / self.track.width)
        laptime = run_sim(self.track, car, new_line_position).laptime

        if laptime < self.best_time:
            improvement = self.best_time - laptime
            deviation = np.abs(self.line_position - new_line_position)
            deviation /= np.max(deviation)

            self.heatmap += deviation * improvement * 1e3
            self.best_time = laptime
            self.progress_rate += improvement
            self.line_position = new_line_position
            return True

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress_rate *= F_ANNEAL  # slowly to zero


def run_sim(track: Track, car: Car, line_position: np.ndarray) -> SimResults:
    return simulate(
        car=car,
        line_coordinates=track.line_coordinates(line_position),
        slope=track.slope,
    )
