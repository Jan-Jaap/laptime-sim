from dataclasses import dataclass
import functools
import numpy as np
import geopandas

@dataclass
class Track:
    name: str
    geodataframe: geopandas.GeoDataFrame
    best_line: np.ndarray = None
    min_clearance: float = 0

    def __post_init__(self):
        # if np.shape(self.border_left)[1] == 2:
        #     self.border_left = np.c_[self.border_left, np.zeros(self.len)]
        # if np.shape(self.border_right)[1] == 2:
        #     self.border_right = np.c_[self.border_right, np.zeros(self.len)]
        if self.best_line is None:
            self.best_line = self.position_clearance

    @functools.cached_property    
    def position_clearance(self):
        return self.min_clearance / self.width
    @functools.cached_property
    def width(self):
        # inner = np.array(self.inner.get_coordinates(include_z=True))
        # outer = np.array(self.inner.get_coordinates(include_z=False))
        return np.sum((self.border_left - self.border_right)**2, 1) ** 0.5
    @functools.cached_property 
    def slope(self):
        return (self.border_right[:,2] - self.border_left[:,2]) / self.width
    @functools.cached_property
    def inner(self):
        return self.geodataframe.geometry.loc[[0]]
    @functools.cached_property
    def outer(self):
        return self.geodataframe.geometry.loc[[1]]
    @functools.cached_property
    def border_right(self):
        border_right = np.array(self.inner.get_coordinates(include_z=True))
        if np.shape(border_right)[1] == 2:
            border_right = np.c_[border_right, np.zeros(self.len)]
        return border_right
    @functools.cached_property
    def border_left(self):
        border_left = np.array(self.outer.get_coordinates(include_z=True))
        if np.shape(border_left)[1] == 2:
            border_left = np.c_[border_left, np.zeros(self.len)]
        return border_left

                # if np.shape(self.border_left)[1] == 2:
        #     self.border_left = np.c_[self.border_left, np.zeros(self.len)]
        return np.array(self.outer.get_coordinates(include_z=True))
    @functools.cached_property
    def len(self):
        return len(self.position_clearance)

            
    def get_line_coordinates(self, position: np.ndarray = None) -> np.ndarray:
        # return self.border_left + np.multiply(self.border_right - self.border_left, position)
        return self.border_left + (self.border_right - self.border_left) * np.expand_dims(position, axis=1)

    def check_clearance(self, position):
        return np.clip(position, self.position_clearance, 1 - self.position_clearance)
