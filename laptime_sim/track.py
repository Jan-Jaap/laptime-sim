from dataclasses import dataclass
import functools
import numpy as np
from geopandas import GeoSeries
from shapely import LineString

@dataclass
class TrackSession:
    track_layout: GeoSeries
    best_line: np.ndarray = None
    heatmap: np.ndarray = None
    min_clearance: float = 0.0
    
    def __post_init__(self):
        if self.best_line is None:
            self.best_line = self._position_clearance
                    
        if self.heatmap is None:
            self.heatmap = np.ones(shape=np.shape(self.best_line))

    @functools.cached_property    
    def _position_clearance(self):
        return self.min_clearance / self.width
    @functools.cached_property
    def width(self):
        return np.sum((self.border_left - self.border_right)**2, 1) ** 0.5
    @functools.cached_property 
    def slope(self):
        return (self.border_right[:,2] - self.border_left[:,2]) / self.width
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
            position = self.best_line
        return get_line_coordinates( self.border_left, self.border_right, position)

    def set_line(self, new_line):
        line_coords = self.line_coords(new_line)
        self.track_layout.geometry['line'] = LineString(line_coords.tolist())
        # return self.line

    def clip_raceline(self, raceline:np.ndarray) -> np.ndarray:
        return np.clip(raceline,  a_min=self._position_clearance, a_max=1 - self._position_clearance)

    def update_best_line(self, new_line, improvement) -> None:
        deviation = np.abs(self.best_line - new_line)
        self.heatmap += deviation / max(deviation) * improvement * 10
        self.best_line = new_line
        line_coords = self.line_coords(new_line)
        self.track_layout.geometry['line'] = LineString(line_coords.tolist())

def get_line_coordinates(left_coords, right_coords, position: np.ndarray) -> np.ndarray:
    return left_coords + (right_coords - left_coords) * np.expand_dims(position, axis=1)
