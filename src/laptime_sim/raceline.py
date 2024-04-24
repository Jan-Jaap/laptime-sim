import os
from typing import NamedTuple
import functools
import numpy as np
from geopandas import GeoDataFrame, read_parquet

from shapely import LineString
from dataclasses import dataclass

from laptime_sim.car import Car
from laptime_sim.track import Track


PATH_RESULTS = "./simulated/"

param_type = NamedTuple("RandomLineParameters", [("location", int), ("length", int), ("deviation", float)])

# annealing factor
f_anneal = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class Raceline:
    track: Track
    car: Car
    line_pos: np.ndarray = None
    heatmap: np.ndarray = None
    min_clearance: float = 0.85
    progress: float = 1.0
    best_time: float = None

    def __post_init__(self):

        if self.line_pos is None:
            self.line_pos = np.zeros_like(self.track.width) + 0.5

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_pos)

    @classmethod
    def from_results(cls, track: Track, car):
        raceline = cls(track, car)
        if os.path.exists(raceline.filename_results):
            results = read_parquet(raceline.filename_results)
            raceline.line_pos = track.parametrize_line_coords(results)
        return raceline

    @functools.cached_property
    def _position_clearance(self):
        return self.min_clearance / self.track.width

    @property
    def len(self):
        return len(self._position_clearance)

    @property
    def slope(self):
        return self.track.slope

    @property
    def filename_results(self):
        return os.path.join(PATH_RESULTS, f"{self.car.file_name}_{self.track.name}_simulated.parquet")

    def line_coords(self, position: np.ndarray = None, include_z=True) -> np.ndarray:
        if position is None:
            position = self.line_pos
        left = self.track.left_coords(include_z=include_z)
        right = self.track.right_coords(include_z=include_z)
        return left + (right - left) * np.expand_dims(position, axis=1)

    def get_dataframe(self) -> GeoDataFrame:
        geom = LineString(self.line_coords().tolist())
        data = dict(
            track_name=self.track.name,
            car=self.car.name,
            best_time=self.best_time,
        )

        return GeoDataFrame.from_dict(data=[data], geometry=[geom], crs=self.track.crs).to_crs(epsg=4326)

    def update(self, position, laptime: float) -> None:

        if self.best_time is None:
            self.best_time = laptime

        if laptime < self.best_time:
            improvement = self.best_time - laptime
            self.best_time = laptime
            self.line_pos = position
            deviation = np.abs(self.line_pos - position)
            max_deviation = max(deviation)
            if max_deviation > 0:
                self.heatmap += deviation / max_deviation * improvement * 1e3
            self.progress += improvement

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= f_anneal  # slowly to zero

    def get_new_line(self):
        line_param = get_new_line_parameters(self.heatmap)
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


def get_new_line_parameters(p) -> param_type:
    location = np.random.choice(len(p), p=p / sum(p))
    length = np.random.randint(1, 60)
    deviation = np.random.randn() / 10
    return param_type(location, length, deviation)
