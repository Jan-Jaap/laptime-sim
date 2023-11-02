from dataclasses import dataclass
import functools
import numpy as np
from geopandas import GeoDataFrame

@dataclass
class TrackSession:
    track_layout: GeoDataFrame
    best_line: np.ndarray = None
    heatmap: np.ndarray = None
    min_clearance: float = 0

    def __post_init__(self):
        if self.best_line is None:
            self.best_line = self.position_clearance
                    
        if self.heatmap is None:
            self.heatmap = np.zeros(shape=np.shape(self.best_line))

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
    def border_left(self):
        return self.track_layout.geometry.loc[['outer']].get_coordinates(include_z=True).to_numpy(na_value=0)
    @functools.cached_property
    def border_right(self):
        return self.track_layout.geometry.loc[['inner']].get_coordinates(include_z=True).to_numpy(na_value=0)
    @functools.cached_property
    def len(self):
        return len(self.position_clearance)

    def get_line_coordinates(self, position: np.ndarray = None) -> np.ndarray:
        if position is None:
            position = self.best_line
        return self.border_left + (self.border_right - self.border_left) * np.expand_dims(position, axis=1)

    def clip_raceline(self, raceline:np.ndarray) -> np.ndarray:
        return np.clip(raceline,  a_min=self.position_clearance, a_max=1 - self.position_clearance)

    def update_best_line(self, new_line) -> None:
        
        self.heatmap = (self.heatmap + 0.05) / (1.05)
        self.heatmap += abs(self.best_line - new_line)
        
        self.best_line = new_line
