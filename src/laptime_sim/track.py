from typing import NamedTuple
from geopandas import GeoSeries, GeoDataFrame
# from laptime_sim.geodataframe_operations import parametrize_raceline
from dataclasses import dataclass
import functools
import numpy as np
from shapely import LineString, Point

# from icecream import ic

param_type = NamedTuple("RandomLineParameters", [("location", int), ("length", int), ("deviation", float)])

# annealing factor
f_anneal = 0.01 ** (1/10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class TrackInterface:
    track_border_left: GeoSeries
    track_border_right: GeoSeries
    line_pos: np.ndarray = None
    heatmap: np.ndarray = None
    min_clearance: float = 0.85
    progress: float = 1.0
    name_track: str = None

    def __post_init__(self):

        if self.line_pos is None:
            self.line_pos = np.zeros_like(self.width) + 0.5

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_pos)

    @classmethod
    def from_layout(cls, layout: GeoDataFrame, track_raceline: GeoSeries = None):
        if track_raceline is not None:
            line_pos = parametrize_raceline(layout.left.geometry, layout.right.geometry, track_raceline)
        else:
            line_pos = None

        return cls(
            track_border_left=layout.left.geometry,
            track_border_right=layout.right.geometry,
            line_pos=line_pos,
            name_track=layout.name[0])

    @functools.cached_property
    def _position_clearance(self):
        return self.min_clearance / self.width

    @functools.cached_property
    def width(self):
        return np.sum((self.left_coords() - self.right_coords())**2, 1) ** 0.5

    @functools.cached_property
    def slope(self):
        return (self.right_coords()[:, 2] - self.left_coords()[:, 2]) / self.width

    def left_coords(self, include_z=True):
        return self.track_border_left.get_coordinates(include_z=include_z).to_numpy(na_value=0)

    def right_coords(self, include_z=True):
        return self.track_border_right.get_coordinates(include_z=include_z).to_numpy(na_value=0)

    @property
    def len(self):
        return len(self._position_clearance)

    def line_coords(self, position: np.ndarray = None, include_z=True) -> np.ndarray:
        if position is None:
            position = self.line_pos
        left = self.left_coords(include_z=include_z)
        right = self.right_coords(include_z=include_z)
        return left + (right - left) * np.expand_dims(position, axis=1)

    def clip_raceline(self, raceline: np.ndarray) -> np.ndarray:
        return np.clip(raceline,  a_min=self._position_clearance, a_max=1 - self._position_clearance)

    def get_raceline(self) -> GeoDataFrame:
        line_coords = self.line_coords()
        # return GeoSeries(data=LineString(line_coords.tolist()), name='line', crs=self.track_border_left.crs)
        track_raceline = GeoSeries(data=LineString(line_coords.tolist()), name='line', crs=self.track_border_left.crs)
        gdf_raceline = GeoDataFrame(geometry=track_raceline, crs=track_raceline.crs)
        return gdf_raceline

    def update(self, position, improvement: bool) -> None:
        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= f_anneal                               # slowly to zero

        if improvement:
            self.line_pos = position
            deviation = np.abs(self.line_pos - position)
            max_deviation = max(deviation)
            if max_deviation > 0:
                self.heatmap += deviation / max_deviation * improvement * 1e3
            self.progress += improvement
            # self.update_line()

    def get_new_line(self):
        line_param = get_new_line_parameters(self.heatmap)
        line_adjust = 1 - np.cos(np.linspace(0, 2*np.pi, line_param.length))
        position = np.zeros_like(self.line_pos)
        position[:line_param.length] = line_adjust * line_param.deviation
        position = np.roll(position, line_param.location - line_param.length//2)
        test_line = self.line_pos + position / self.width
        return self.clip_raceline(test_line)


def get_new_line_parameters(p) -> param_type:
    location = np.random.choice(len(p), p=p / sum(p))
    length = np.random.randint(1, 60)
    deviation = np.random.randn() / 10
    return param_type(location, length, deviation)


def parametrize_raceline(
        track_border_left: GeoSeries,
        track_border_right: GeoSeries,
        track_raceline: GeoSeries
        ):

    left_coords = track_border_left.get_coordinates(include_z=False).to_numpy(na_value=0)
    right_coords = track_border_right.get_coordinates(include_z=False).to_numpy(na_value=0)
    line_coords = track_raceline.get_coordinates(include_z=False).to_numpy(na_value=0)

    def loc_line(point_left, point_right, point_line):
        division = LineString([(point_left), (point_right)])
        intersect = Point(point_line)
        return division.project(intersect, normalized=True)

    return [loc_line(pl, pr, loc) for pl, pr, loc in zip(left_coords, right_coords, line_coords)]
