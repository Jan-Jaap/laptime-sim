from dataclasses import dataclass
import functools
import numpy as np
from shapely import LineString

from geodataframe_operations import GeoSeries
from car import Car


@dataclass
class TrackSession:
    track_layout: GeoSeries
    car: Car
    line_pos: np.ndarray = None
    heatmap: np.ndarray = None
    min_clearance: float = 0.85

    def __post_init__(self):
        if self.line_pos is None:
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
        return self.track_layout.geometry.loc[['outer']].get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def border_right(self):
        return self.track_layout.geometry.loc[['inner']].get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def len(self):
        return len(self._position_clearance)

    @property
    def line(self):
        return self.track_layout.geometry.loc[['line']].get_coordinates(include_z=True).to_numpy(na_value=0)

    def line_coords(self, position: np.ndarray = None) -> np.ndarray:
        if position is None:
            position = self.line_pos
        return get_line_coordinates(self.border_left, self.border_right, position)

    # def set_line(self, new_line):
    #     line_coords = self.line_coords(new_line)
    #     self.track_layout.geometry['line'] = LineString(line_coords.tolist())
    #     # return self.line

    def clip_raceline(self, raceline: np.ndarray) -> np.ndarray:
        return np.clip(raceline,  a_min=self._position_clearance, a_max=1 - self._position_clearance)

    def update_best_line(self, new_line, improvement) -> None:
        deviation = np.abs(self.line_pos - new_line)
        self.heatmap += deviation / max(deviation) * improvement * 1e3
        self.line_pos = new_line
        line_coords = self.line_coords(new_line)
        self.track_layout.geometry['line'] = LineString(line_coords.tolist())
        return self

    def get_new_line(self, parameters):

        location, length, deviation = parameters
        line_adjust = 1 - np.cos(np.linspace(0, 2*np.pi, length))
        new_line = self.line_pos * 0
        new_line[:length] = line_adjust * deviation
        new_line = np.roll(new_line, location - length//2)
        test_line = self.line_pos + new_line / self.width
        return self.clip_raceline(test_line)


def get_line_coordinates(left_coords, right_coords, position: np.ndarray) -> np.ndarray:
    return left_coords + (right_coords - left_coords) * np.expand_dims(position, axis=1)
