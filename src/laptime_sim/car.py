from dataclasses import dataclass, asdict
from enum import IntEnum
import functools
import json

import toml
# from numba import njit


class Trailbraking(IntEnum):
    '''Enum with trailbraking parameter corresponding with driver experience'''
    AMATEUR = 30
    BEGINNER = 50
    EXPERIENCED = 70
    PROFESSIONAL = 100


class CornerAcceleration(IntEnum):
    '''Enum with trailbraking parameter corresponding with driver experience'''
    NO_DIFFERENTIAL_FWD = 40
    NO_DIFFERENTIAL_RWD = 60
    DIFFERENTIAL_FWD = 80
    DIFFERENTIAL_RWD = 100


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
        trail_braking:  Trailbraking, int = 0..100
        corner_acc:     CornerAcceleration, int = 0..100

    """
    mass:           float
    P_engine:       float
    acc_limit:      float
    dec_limit:      float
    acc_grip_max:   float
    c_drag:         float
    c_roll:         float
    trail_braking:  Trailbraking
    corner_acc:     CornerAcceleration
    name:           str = None

    @classmethod
    def from_json(cls, filename):
        '''load car parameters from JSON file'''
        with open(filename, 'r') as fp:
            return cls(**json.load(fp))

    @classmethod
    def from_toml(cls, filename):
        '''load car parameters from TOML file'''
        return cls(**toml.load(filename))

    @functools.cached_property
    def P_engine_in_watt(self):
        return self.P_engine / 1.3410 * 1000  # from hp to Watt

    def force_engine(self, v):
        '''return available engine force at given velocity'''
        return v and self.P_engine_in_watt / v or 0  # tractive force (limited by engine power)

    def get_gear(self, v):
        '''not implemented'''
        pass

    def dict(self):
        return asdict(self)
