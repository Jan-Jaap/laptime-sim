import os
import functools
from enum import IntEnum
from pathlib import Path
from pydantic import BaseModel


import numpy as np
import toml


class DriverExperience(IntEnum):
    """Enum with trailbraking parameter corresponding with driver experience"""

    AMATEUR = 30
    BEGINNER = 50
    EXPERIENCED = 70
    PROFESSIONAL = 100


class CornerAcceleration(IntEnum):
    """Enum with trailbraking parameter corresponding with driver experience"""

    NO_DIFFERENTIAL_FWD = 50
    DIFFERENTIAL_FWD = 80
    NO_DIFFERENTIAL_RWD = 60
    DIFFERENTIAL_RWD = 90


# @dataclass


class Car(BaseModel):
    mass: float
    P_engine: float
    acc_limit: float
    dec_limit: float
    lat_limit: float
    c_drag: float
    c_roll: float
    trail_braking: DriverExperience
    corner_acc: CornerAcceleration
    name: str = None

    @classmethod
    def from_toml(cls, file_name):
        """load car parameters from TOML file"""
        return cls(**toml.load(file_name))

    @functools.cached_property
    def file_name(self):
        return self.name.replace(" ", "_")

    @functools.cached_property
    def P_engine_in_watt(self):
        return self.P_engine / 1.3410 * 1000

    @functools.cached_property
    def rolling_drag(self):
        return self.c_roll * 9.81

    def get_acceleration(self, v, acc_lat):
        n = self.corner_acc / 50
        acc_max = (self.acc_limit) * (1 - (np.abs(acc_lat) / self.lat_limit) ** n) ** (1 / n)
        force_engine = v and self.P_engine_in_watt / v or 0
        acc_max = force_engine and min(acc_max, force_engine / self.mass) or acc_max
        aero_drag = v**2 * self.c_drag / 2 / self.mass  #  F=ma -> a=F/m
        rolling_drag = self.c_roll * 9.81
        return acc_max - aero_drag - rolling_drag

    def get_deceleration(self, v, acc_lat):
        n = self.trail_braking / 50
        max_dec_grip = (self.dec_limit) * (1 - (np.abs(acc_lat) / self.lat_limit) ** n) ** (1 / n)
        aero_drag = v**2 * self.c_drag / 2 / self.mass  #  F=ma -> a=F/m
        rolling_drag = self.c_roll * 9.81
        return max_dec_grip + aero_drag + rolling_drag


def strip_filename(filename: str) -> str:
    filename = os.path.basename(filename).replace("_simulated", "")
    return strip_extension(filename)


def strip_extension(path: str) -> str:
    return os.path.splitext(path)[0]


def car_list(path_cars: Path | str) -> list[Car]:
    path_cars = Path(path_cars)
    return [Car.from_toml(file) for file in sorted(path_cars.glob("*.toml"))]
