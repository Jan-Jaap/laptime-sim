from dataclasses import dataclass
from functools import cached_property
import numpy as np
import json

import toml


@dataclass
class Car:
    """
        Args:
        name:           Name of car
        mass:           Mass of car [kg]
        P_engine:       Engine power [HP]
        acc_limit:      maximal longitudinal acceleration limited by tire grip  [m/s²]
        dec_limit:      maximal longitudinal deceleration limited by tire grip = limit braking
        acc_grip_max:   maximal lateral accelaration (lateral g-force)
        c_drag:         aerodynamic drag
        c_roll:         rolling resistance
        trail_braking:  float = 70
    """
    mass:           float
    P_engine:       float
    acc_limit:      float
    dec_limit:      float
    acc_grip_max:   float
    c_drag:         float
    c_roll:         float
    name:           str = None
    trail_braking:  float = 70

    @classmethod
    def from_json(cls, filename):
        '''load car parameters from JSON file'''
        with open(filename, 'r') as fp:
            return cls(**json.load(fp))

    @classmethod
    def from_toml(cls, filename):
        '''load car parameters from TOML file'''
        return cls(**toml.load(filename))

    def get_max_acc(self, v, acc_lat):
        '''maximum possible acceleration (flooring it)
        args:
            v:          velocity in m/s
            acc_lat:    lateral acceleration in m/s²
        '''
        # grip circle (no downforce accounted for)
        acc_lon_max = self.acc_limit / self.acc_grip_max * (self.acc_grip_max**2 - acc_lat**2)**0.5
        # max lateral accceleration due to engine
        acc_lon = (self.force_engine(v) - (v**2 * self.c_drag)) / self.mass
        acc_lon -= self.c_roll * 9.81       # rolling resistance
        return min(acc_lon_max, acc_lon)

    def get_min_acc(self, v, acc_lat):
        '''maximum possible deceleration (limit braking)'''
        n = self.trail_braking / 50
        acc_lon = self.dec_limit * (1 - (np.abs(acc_lat) / self.acc_grip_max)**n)**(1/n)
        acc_lon += v**2 * self.c_drag / self.mass
        acc_lon += self.c_roll * 9.81      # rolling resistance
        return acc_lon

    @cached_property
    def P_engine_in_watt(self):
        return self.P_engine / 1.3410 * 1000  # from hp to Watt

    def force_engine(self, v):
        '''return available engine force at given velocity'''
        return self.P_engine_in_watt / v   # tractive force (limited by engine power)

    def get_gear(self, v):
        '''not implemented'''
        pass
