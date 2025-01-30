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

MAX_DEVIATION = 0.1
MAX_DEVIATION_LENGTH = 60
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement
BORDER_CLEARANCE_M: float = 0.85

random_generator = np.random.default_rng()


@dataclass
class Raceline:
    track: Track
    best_time: float = np.inf
    progress_rate: float = 1.0
    _heatmap: np.ndarray = None
    line_position: np.ndarray = None

    def __post_init__(self):
        if self.line_position is None:
            initial_line = self.track.initial_line()
            line_position = self.track.position_from_coordinates(initial_line)
            self.line_position = self.clip_line(line_position)
        self._heatmap = np.ones_like(self.line_position)

    @classmethod
    def from_coordinates(cls, track: Track, line_coords: np.ndarray) -> Self:
        return cls(track=track, line_position=track.position_from_coordinates(line_coords))

    @classmethod
    def from_geodataframe(cls, track: Track, data: GeoDataFrame) -> Self:
        data = data.to_crs(data.estimate_utm_crs())
        line_coords = data.get_coordinates(include_z=False).to_numpy(na_value=0)
        return cls.from_coordinates(track=track, line_coords=line_coords)

    @classmethod
    def from_file(cls, track: Track, file_path: Path) -> Self:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exist")

        data = read_parquet(file_path)
        assert track.name == data.iloc[0].track_name
        return cls.from_geodataframe(track, data)

    def best_time_str(self) -> str:
        return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

    def filename(self, car_name) -> Path:
        return Path(f"{car_name}_{self.track.name}_simulated.parquet")

    def save_line(self, file_path: Path, car_name: str) -> None:
        if file_path is None:
            raise FileNotFoundError("no output filename provided")

        self.dataframe(track_name=self.track.name, car_name=car_name).to_parquet(file_path)

    def dataframe(self, track_name, car_name) -> GeoDataFrame:
        geom = LineString(self.track.coordinates_from_position(self.line_position).tolist())
        data = dict(track_name=track_name, car=car_name)
        return GeoDataFrame.from_dict(data=[data], geometry=[geom], crs=self.track.crs).to_crs(epsg=4326)

    @functools.cached_property
    def width(self):
        if self.track.is_circular:
            return self.track.width[:-1]

        return self.track.width

    @functools.cached_property
    def position_clearance(self):
        return BORDER_CLEARANCE_M / self.width

    def clip_line(self, line_position: np.ndarray) -> np.ndarray:
        return np.clip(line_position, a_min=self.position_clearance, a_max=1 - self.position_clearance)

    def update(self, car: Car) -> SimResults:
        sim_results = run_sim(self.track, car, self.line_position)

        if sim_results.laptime < self.best_time:
            self.best_time = sim_results.laptime
        return sim_results

    def simulate_new_line(self, car: Car) -> None:
        location = random_generator.choice(len(self._heatmap), p=self._heatmap / sum(self._heatmap))
        length = random_generator.integers(3, MAX_DEVIATION_LENGTH, endpoint=True)
        deviation = random_generator.uniform(-1, 1) * max(length / MAX_DEVIATION_LENGTH, MAX_DEVIATION)
        line_adjust = (1 + np.cos(np.linspace(-np.pi, np.pi, length))) / 2
        position = np.zeros(len(self.line_position))

        if self.track.is_circular:  # ensure first and last position value are the same
            position[:length] = line_adjust
            position = np.roll(position, location - length // 2)
        else:  # Start position is fixed.  Finish position can change.
            if location + length > len(self._line_position):
                length = len(self._line_position) - location
            position[location : location + length] = line_adjust[:length]

        new_line_position = self.clip_line(self.line_position + position * deviation / self.width)
        laptime = run_sim(self.track, car, new_line_position).laptime

        if laptime < self.best_time:
            improvement = self.best_time - laptime
            self._heatmap += position * improvement * 1e3

            self.best_time = laptime
            self.progress_rate += improvement
            self.line_position = new_line_position
            return True

        self._heatmap = (self._heatmap + 0.00015) / 1.00015  # slowly to one
        self.progress_rate *= F_ANNEAL  # slowly to zero


def run_sim(track: Track, car: Car, line_position: np.ndarray) -> SimResults:
    return simulate(
        car=car,
        line_coordinates=track.coordinates_from_position(line_position),
        slope=track.slope,
    )
