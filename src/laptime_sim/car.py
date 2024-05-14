import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
import json
import toml
import os


class Trailbraking(IntEnum):
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

    mass: float
    P_engine: float
    acc_limit: float
    dec_limit: float
    acc_grip_max: float
    c_drag: float
    c_roll: float
    trail_braking: Trailbraking
    corner_acc: CornerAcceleration
    name: str = None
    P_engine_in_watt: float = field(init=False)
    file_name: str = None

    def __post_init__(self):
        self.P_engine_in_watt = self.P_engine / 1.3410 * 1000
        if self.file_name is None:
            self.file_name = self.name.replace(" ", "_")

    @classmethod
    def from_name(cls, name, path):
        return [f for f in get_all_cars(path) if f.name == name][0]

    @classmethod
    def from_json(cls, filename):
        """load car parameters from JSON file"""
        with open(filename, "r") as fp:
            return cls(**json.load(fp))

    @classmethod
    def from_toml(cls, file_name):
        """load car parameters from TOML file"""
        return cls(**toml.load(file_name), file_name=strip_filename(file_name))

    def force_engine(self, v):
        """return available engine force at given velocity"""
        return v and self.P_engine_in_watt / v or 0  # tractive force (limited by engine power)

    def get_gear(self, v):
        """not implemented"""
        pass

    def get_acceleration(self, v, acc_lat):

        n = self.corner_acc / 50
        max_acc_grip = (self.acc_limit) * (1 - (np.abs(acc_lat) / self.acc_grip_max) ** n) ** (1 / n)
        force_engine = v and self.P_engine_in_watt / v or 0
        acceleration_max = force_engine and min(max_acc_grip, force_engine / self.mass) or max_acc_grip
        aero_drag = v**2 * self.c_drag / self.mass
        rolling_drag = self.c_roll * 9.81
        return acceleration_max - aero_drag - rolling_drag

    def get_deceleration(self, v, acc_lat):

        n = self.trail_braking / 50
        max_dec_grip = (self.dec_limit) * (1 - (np.abs(acc_lat) / self.acc_grip_max) ** n) ** (1 / n)
        aero_drag = v**2 * self.c_drag / self.mass
        rolling_drag = self.c_roll * 9.81
        return max_dec_grip + aero_drag + rolling_drag


def strip_filename(filename: str) -> str:
    filename = os.path.basename(filename).replace("_simulated", "")
    return strip_extension(filename)


def strip_extension(path: str) -> str:
    return os.path.splitext(path)[0]


def get_all_cars(path: str) -> list[Car]:
    """
    Returns a list of all cars in the given directory, sorted by filename.

    :param path: The path to the directory containing the car definition files
    :return: A list of all cars in the given directory
    """
    car_files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith("toml")]
    return [Car.from_toml(f) for f in car_files]
