import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from geopandas import GeoDataFrame, read_parquet
from shapely import LineString
import shapely

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults
from laptime_sim.simulate import simulate
from laptime_sim.track import Track

MAX_DEVIATION = 0.5
MAX_DEVIATION_LENGTH = 60
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class Raceline:
    track: Track
    best_time: float = np.inf
    progress_rate: float = 1.0
    _heatmap: np.ndarray = None
    _line_position: np.ndarray = None
    _clearance_meter: float = 0.85

    def __post_init__(self):
        if self._line_position is None:
            initial_line = self.track.initial_line(smoothing_window=40, poly_order=5)
            self.line_position = parameterize_line_coordinates(self.track, initial_line)
        self._heatmap = np.ones_like(self.line_position)

    @property
    def line_position(self) -> np.ndarray:
        if self.track.is_circular:
            return np.append(self._line_position, self._line_position[0])
        return self._line_position

    @line_position.setter
    def line_position(self, value) -> None:
        if self.track.is_circular:
            self._line_position = self.clip_line(value)[:-1]
        else:
            self._line_position = self.clip_line(value)

    def best_time_str(self) -> str:
        return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

    @classmethod
    def from_geodataframe(cls, data: GeoDataFrame, track: Track) -> Self:
        data = data.to_crs(track.crs)
        line_coords = data.get_coordinates(include_z=False).to_numpy(na_value=0)
        raceline = cls(track=track)
        raceline.line_position = parameterize_line_coordinates(track, line_coords)
        return raceline

    def filename(self, car_name) -> Path:
        return Path(f"{car_name}_{self.track.name}_simulated.parquet")

    def load_line(self, file_path: Path) -> Self:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exist")

        results = read_parquet(file_path)
        results = results.to_crs(self.track.crs)

        assert self.track.name == results.iloc[0].track_name

        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)
        self.line_position = parameterize_line_coordinates(self.track, line_coords)
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

    def clip_line(self, line_position: np.ndarray) -> np.ndarray:
        return np.clip(line_position, a_min=self.position_clearance, a_max=1 - self.position_clearance)

    def simulate(self, car: Car) -> SimResults:
        sim_results = run_sim(self.track, car, self.line_position)

        if sim_results.laptime < self.best_time:
            self.best_time = sim_results.laptime
        return sim_results

    def simulate_new_line(self, car: Car) -> None:
        rng = np.random.default_rng()
        location = rng.choice(len(self._heatmap), p=self._heatmap / sum(self._heatmap))

        length = rng.integers(3, MAX_DEVIATION_LENGTH)
        # length = int(rng.exponential(scale=50) + 3) % len(self._line_position)
        deviation = rng.random() * MAX_DEVIATION
        line_adjust = (1 + np.cos(np.linspace(-np.pi, np.pi, length))) / 2

        position = np.zeros(len(self._line_position))

        if self.track.is_circular:  # ensure first and last position value are the same
            position[:length] = line_adjust
            position = np.roll(position, location - length // 2)
            position = np.append(position, position[0])
        else:  # Start position is fixed.  Finish position can change.
            if location + length > len(self._line_position):
                length = len(self._line_position) - location
            position[location : location + length] = line_adjust[:length]

        new_line_position = self.clip_line(self.line_position + position * deviation / self.track.width)
        laptime = run_sim(self.track, car, new_line_position).laptime

        if laptime < self.best_time:
            improvement = self.best_time - laptime
            self._heatmap += position * improvement * 1e3

            self.best_time = laptime
            self.progress_rate += improvement
            self.line_position = new_line_position
            return True

        # self._heatmap = (self._heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress_rate *= F_ANNEAL  # slowly to zero


def run_sim(track: Track, car: Car, line_position: np.ndarray) -> SimResults:
    return simulate(
        car=car,
        line_coordinates=track.line_coordinates(line_position),
        slope=track.slope,
    )


def loc_line(point_left, point_right, point_line):
    division = shapely.LineString([(point_left), (point_right)])
    intersect = shapely.Point(point_line)
    return division.project(intersect, normalized=True)


def parameterize_line_coordinates(track: Track, line_coords: np.ndarray) -> np.ndarray:
    return np.array(
        [
            loc_line(pl, pr, loc)
            for pl, pr, loc in zip(
                track.left_coords_2d,
                track.right_coords_2d,
                line_coords,
            )
        ]
    )
