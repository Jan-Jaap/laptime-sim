from typing import NamedTuple
from geopandas import GeoSeries, GeoDataFrame
from dataclasses import dataclass
import functools
import numpy as np
from shapely import LineString, Point
import pandas as pd

from car import Car

param_type = NamedTuple("RandomLineParameters", [("location", int), ("length", int), ("deviation", float)])


@dataclass
class TrackSession:
    track_layout: GeoSeries
    car: Car
    line_pos: np.ndarray = None
    heatmap: np.ndarray = None
    min_clearance: float = 0.85
    progress: float = 1.0

    def __post_init__(self):
        if self.line_pos is None:
            if self.has_line:
                self.line_pos = parametrize_raceline(
                    self.get_coordinates('outer', include_z=False),
                    self.get_coordinates('inner', include_z=False),
                    self.get_coordinates('line', include_z=False),
                    )
            else:
                self.line_pos = np.zeros_like(self.width) + 0.5

            self.line_pos = self.clip_raceline(self.line_pos)

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_pos)

    @functools.cached_property
    def _position_clearance(self):
        return self.min_clearance / self.width

    @functools.cached_property
    def width(self):
        return np.sum((self.border_left - self.border_right)**2, 1) ** 0.5

    @functools.cached_property
    def slope(self):
        return (self.border_right[:, 2] - self.border_left[:, 2]) / self.width

    @functools.cached_property
    def border_left(self):
        return self.get_coordinates('outer')

    @functools.cached_property
    def border_right(self):
        return self.get_coordinates('inner')

    def get_coordinates(self, type_name, include_z=True):
        idx = self.track_layout['type'] == type_name
        return self.track_layout.loc[idx].geometry.get_coordinates(include_z=include_z).to_numpy(na_value=0)

    @functools.cached_property
    def len(self):
        return len(self._position_clearance)

    @property
    def line(self):
        # return self.get_coordinates('line')
        idx = self.track_layout['type'] == 'line'
        return self.track_layout.loc[idx].geometry

    @property
    def has_line(self):
        return any(self.track_layout['type'] == 'line')

    def line_coords(self, position: np.ndarray = None) -> np.ndarray:
        if position is None:
            position = self.line_pos
        return position_to_coordinates(self.border_left, self.border_right, position)

    def clip_raceline(self, raceline: np.ndarray) -> np.ndarray:
        return np.clip(raceline,  a_min=self._position_clearance, a_max=1 - self._position_clearance)

    def update_line(self, position):
        self.line_pos = self.clip_raceline(position)
        line_coords = self.line_coords(self.line_pos)

        # remove existing raceline
        if any(idx := self.track_layout['type'] == 'line'):
            self.track_layout = self.track_layout.loc[~idx]

        # create net raceline
        track_layout = GeoDataFrame(
            data={
                'geometry': LineString(line_coords.tolist()),
                'type': 'line',
                'track': [self.track_layout.track.iloc[0]],
                },
            crs=self.track_layout.crs
            )
        self.track_layout = pd.concat([self.track_layout, track_layout]).reset_index(drop=True)

    def update(self, position, improvement) -> None:
        f = 0.01 ** (1/10000)  # from 1 to 0.01 in 10000 iterations without improvement

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= f                               # slowly to zero

        if improvement > 0:
            deviation = np.abs(self.line_pos - position)
            self.heatmap += deviation / max(deviation) * improvement * 1e3
            self.progress += improvement
            self.update_line(position)

    def get_new_line(self, line_param: param_type = None):
        if line_param is None:
            line_param = get_new_line_parameters(self.heatmap)

        line_adjust = 1 - np.cos(np.linspace(0, 2*np.pi, line_param.length))
        position = self.line_pos * 0
        position[:line_param.length] = line_adjust * line_param.deviation
        position = np.roll(position, line_param.location - line_param.length//2)
        test_line = self.line_pos + position / self.width
        return self.clip_raceline(test_line)


def get_new_line_parameters(p) -> param_type:
    location = np.random.choice(len(p), p=p / sum(p))
    length = np.random.randint(1, 60)
    deviation = np.random.randn() / 10
    return param_type(location, length, deviation)


def position_to_coordinates(left_coords, right_coords, position: np.ndarray) -> np.ndarray:
    return left_coords + (right_coords - left_coords) * np.expand_dims(position, axis=1)


def parametrize_raceline(left_coords, right_coords, line_coords):
    def loc_line(point_left, point_right, point_line):
        division = LineString([(point_left), (point_right)])
        intersect = Point(point_line)
        return division.project(intersect, normalized=True)
    return [loc_line(pl, pr, loc) for pl, pr, loc in zip(left_coords, right_coords, line_coords)]
