from typing import NamedTuple
from geodataframe_operations import GeoSeries
from geopandas import GeoDataFrame
from dataclasses import dataclass
import functools
import numpy as np
from shapely import LineString, Point


from car import Car
import geodataframe_operations

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
            try:
                self.line_pos = parametrize_raceline(self.track_layout)
            except KeyError:
                self.line_pos = np.zeros_like(self.width) + 0.5
            finally:
                self.line_pos = self.clip_raceline(self.line_pos)
                self.update_line(self.line_pos)

        self.track_layout.index = self.track_layout['type']

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

    def get_coordinates(self, type_name):
        idx = self.track_layout['type'] == type_name
        return self.track_layout.loc[idx].geometry.get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def len(self):
        return len(self._position_clearance)

    @property
    def line(self):
        return self.get_coordinates('line')
        # return self.track_layout.geometry.loc[['line']].get_coordinates(include_z=True).to_numpy(na_value=0)

    def line_coords(self, position: np.ndarray = None) -> np.ndarray:
        if position is None:
            position = self.line_pos
        return get_line_coordinates(self.border_left, self.border_right, position)

    def clip_raceline(self, raceline: np.ndarray) -> np.ndarray:
        return np.clip(raceline,  a_min=self._position_clearance, a_max=1 - self._position_clearance)

    def update_line(self, position):
        # crs = self.track_layout.crs
        self.line_pos = self.clip_raceline(position)
        line_coords = self.line_coords(self.line_pos)
        # line = GeoSeries(LineString(line_coords.tolist()), crs=self.track_layout.crs)
        # line = GeoDataFrame({'geometry': line,
        #                      'type': 'line',
        #                      'track': self.track_layout.track[0],
        #                      })
        # self.track_layout = geodataframe_operations.append([self.track_layout,  line])

        self.track_layout.geometry['line'] = LineString(line_coords.tolist())

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


def get_line_coordinates(left_coords, right_coords, position: np.ndarray) -> np.ndarray:
    return left_coords + (right_coords - left_coords) * np.expand_dims(position, axis=1)


def parametrize_raceline(geo: GeoSeries):
    def loc_line(point_left, point_right, point_line):
        division = LineString([(point_left), (point_right)])
        intersect = Point(point_line)
        return division.project(intersect, normalized=True)

    # border_left = geo.loc[geo.type == 'outer'].geometry.get_coordinates(include_z=False).to_numpy(na_value=0)
    border_left = geo.loc[['outer']].get_coordinates(include_z=False).to_numpy(na_value=0)
    border_right = geo.loc[['inner']].get_coordinates(include_z=False).to_numpy(na_value=0)
    raceline = geo[['line']].get_coordinates(include_z=False).to_numpy(na_value=0)
    return [loc_line(pl, pr, loc) for pl, pr, loc in zip(border_left, border_right, raceline)]
