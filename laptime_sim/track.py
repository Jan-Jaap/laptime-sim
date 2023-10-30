from dataclasses import dataclass
import functools
import numpy as np
from geopandas import GeoDataFrame

@dataclass
class Track:
    name: str
    geodataframe: GeoDataFrame
    best_line: np.ndarray = None
    min_clearance: float = 0

    def __post_init__(self):

        if self.best_line is None:
            self.best_line = self.position_clearance

    @functools.cached_property    
    def position_clearance(self):
        return self.min_clearance / self.width
    @functools.cached_property
    def width(self):
        return np.sum((self.border_left - self.border_right)**2, 1) ** 0.5
    @functools.cached_property 
    def slope(self):
        return (self.border_right[:,2] - self.border_left[:,2]) / self.width
    @functools.cached_property
    def border_right(self):
        return self.geodataframe.geometry.loc[[0]].get_coordinates(include_z=True).to_numpy(na_value=0)
    @functools.cached_property
    def border_left(self):
        return self.geodataframe.geometry.loc[[1]].get_coordinates(include_z=True).to_numpy(na_value=0)
    @functools.cached_property
    def len(self):
        return len(self.position_clearance)

    def get_line_coordinates(self, position: np.ndarray = None) -> np.ndarray:
        # return self.border_left + np.multiply(self.border_right - self.border_left, position)
        return self.border_left + (self.border_right - self.border_left) * np.expand_dims(position, axis=1)

    def check_clearance(self, position):
        return np.clip(position, self.position_clearance, 1 - self.position_clearance)


def interpolate_track(track: Track, nr_datapoint: int) -> Track:
    
    pass