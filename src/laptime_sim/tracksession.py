from typing import NamedTuple
from geopandas import GeoSeries
from dataclasses import dataclass
import functools
import numpy as np
from shapely import LineString, Point

param_type = NamedTuple("RandomLineParameters", [("location", int), ("length", int), ("deviation", float)])

# annealing factor
f_anneal = 0.01 ** (1/10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class TrackSession:
    # track_layout: GeoSeries
    track_border_left: GeoSeries
    track_border_right: GeoSeries
    track_raceline: GeoSeries = None
    line_pos: np.ndarray = None
    heatmap: np.ndarray = None
    min_clearance: float = 0.85
    progress: float = 1.0

    def __post_init__(self):
        if self.line_pos is None and self.track_raceline is None:
            self.line_pos = np.zeros_like(self.width) + 0.5

        if self.line_pos is None:
            outer_2dcoords = self.track_border_left.get_coordinates(include_z=False).to_numpy(na_value=0)
            inner_2dcoords = self.track_border_right.get_coordinates(include_z=False).to_numpy(na_value=0)
            line_2dcoords = self.track_raceline.get_coordinates(include_z=False).to_numpy(na_value=0)
            self.line_pos = parametrize_raceline(outer_2dcoords, inner_2dcoords, line_2dcoords)

        self.line_pos = self.clip_raceline(self.line_pos)

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_pos)

    @classmethod
    def from_layout(cls, layout, track_raceline=None, cw=True):
        track_border_left = layout.outer.geometry
        track_border_right = layout.inner.geometry
        return cls(
            track_border_left=track_border_left,
            track_border_right=track_border_right,
            track_raceline=track_raceline)

    @functools.cached_property
    def _position_clearance(self):
        return self.min_clearance / self.width

    @functools.cached_property
    def width(self):
        return np.sum((self.left_coords - self.right_coords)**2, 1) ** 0.5

    @functools.cached_property
    def slope(self):
        return (self.right_coords[:, 2] - self.left_coords[:, 2]) / self.width

    @functools.cached_property
    def left_coords(self):
        return self.track_border_left.get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def right_coords(self):
        return self.track_border_right.get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def len(self):
        return len(self._position_clearance)

    def line_coords(self, position: np.ndarray = None) -> np.ndarray:
        if position is None:
            position = self.line_pos
        return self.left_coords + (self.right_coords - self.left_coords) * np.expand_dims(position, axis=1)

    def clip_raceline(self, raceline: np.ndarray) -> np.ndarray:
        return np.clip(raceline,  a_min=self._position_clearance, a_max=1 - self._position_clearance)

    def update_line(self):
        line_coords = self.line_coords()
        track_raceline = GeoSeries(data=LineString(line_coords.tolist()), name='line', crs=self.track_border_left.crs)
        self.track_raceline = track_raceline

    def update(self, position, improvement) -> None:
        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= f_anneal                               # slowly to zero

        if improvement > 0:
            self.line_pos = self.clip_raceline(position)
            deviation = np.abs(self.line_pos - position)
            max_deviation = max(deviation)
            if max_deviation > 0:
                self.heatmap += deviation / max_deviation * improvement * 1e3
            self.progress += improvement
            self.update_line()

    def get_new_line(self):
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


def parametrize_raceline(left_coords, right_coords, line_coords):
    def loc_line(point_left, point_right, point_line):
        division = LineString([(point_left), (point_right)])
        intersect = Point(point_line)
        return division.project(intersect, normalized=True)
    return [loc_line(pl, pr, loc) for pl, pr, loc in zip(left_coords, right_coords, line_coords)]
