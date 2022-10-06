from dataclasses import dataclass
import numpy as np

@dataclass
class Track:
    name: str
    border_left: np.ndarray
    border_right: np.ndarray
    best_known_raceline: np.ndarray = None
    min_clearance: float = 0
    
    def __post_init__(self):
        self.position_clearance = self.min_clearance / self.width

        if self.best_known_raceline is None:
            self.best_known_raceline = self.position_clearance  #hugging left edge

    @property
    def width(self):
        return np.sum((self.border_right[:,:2] - self.border_left[:,:2])**2, 1) ** 0.5
    @property
    def slope(self):
        return (self.border_right[:,2] - self.border_left[:,2]) / self.width
    @property
    def left_x(self):
        return self.border_left[:,0]
    @property
    def left_y(self):
        return self.border_left[:,1]
    @property
    def right_x(self):
        return self.border_right[:,0]
    @property
    def right_y(self):
        return self.border_right[:,1]
            
    def get_line_coordinates(self, position: np.ndarray = None) -> np.ndarray:
        return self.border_left + (self.border_right - self.border_left) * np.expand_dims(position, axis=1)
