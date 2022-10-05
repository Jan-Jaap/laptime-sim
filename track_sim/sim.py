from dataclasses import dataclass
from utilities.dotdict import DotDict
import numpy as np
import pandas as pd

class Car(DotDict):
    trail_braking = 70
    
    def get_max_acc(self,v, acc_lat):
        '''maximum possible acceleration (flooring)'''
        acc_lon_max = self.acc_limit / self.acc_grip_max * (self.acc_grip_max**2 - acc_lat**2)**0.5   #grip circle (no downforce accounted for)
        acc_lon = (self.force_engine(v) - (v**2 * self.c_drag) ) / self.mass                        
        acc_lon -=  self.c_roll * 9.81                               #rolling resistance
        return min(acc_lon_max, acc_lon)

    def get_min_acc(self,v, acc_lat):
        '''maximum possible deceleration (braking)'''
        n = self.trail_braking / 50
        acc_lon = self.dec_limit * (1 - (np.abs(acc_lat) / self.acc_grip_max)**n)**(1/n)
        acc_lon +=  v**2 * self.c_drag / self.mass
        acc_lon +=  self.c_roll * 9.81 #rolling resistance
        return acc_lon

    def force_engine(self, v):
        P_engine = self.P_engine / 1.3410 * 1000  # from hp to Watt
        return P_engine / v   #tractive force (limited by engine power)

    def get_gear(self, v):
        return v*0


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

    def get_track_borders(self):
        # return pd.DataFrame([[self.left_x, self.left_y, self.right_x, self.right_y]], columns=['left_x','left_y','right_x','right_y'])

        return pd.DataFrame(
            data = np.column_stack([self.left_x, self.left_y, self.right_x, self.right_y]),
            columns=['left_x','left_y','right_x','right_y'],
            )
