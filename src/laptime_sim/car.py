import functools
import tomllib
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class DriverExperience(IntEnum):
    """Enum with trailbraking parameter corresponding with driver experience"""

    AMATEUR = 30
    BEGINNER = 50
    EXPERIENCED = 70
    PROFESSIONAL = 100


class CornerAcceleration(IntEnum):
    """Enum representing corner acceleration factors based on drivetrain configuration"""

    NO_DIFFERENTIAL_FWD = 50
    DIFFERENTIAL_FWD = 80
    NO_DIFFERENTIAL_RWD = 60
    DIFFERENTIAL_RWD = 90


@dataclass
class Car:
    """
    A class representing a car with its physical properties and capabilities.
    """

    name: str
    mass: float  # Mass of the car in kilograms
    P_engine: float  # Engine power in horsepower
    acc_limit: float  # Longitudinal acceleration limit (grip limit) in m/s²
    dec_limit: float  # Deceleration limit (grip limit under braking) in m/s²
    lat_limit: float  # Lateral acceleration limit (grip limit) in m/s²
    c_drag: float  # Drag coefficient
    c_roll: float  # Rolling resistance coefficient
    trail_braking: DriverExperience  # Trailbraking factor
    corner_acc: CornerAcceleration  # Corner acceleration factor

    def __post_init__(self):
        for field_name in ("mass", "P_engine", "acc_limit", "dec_limit", "lat_limit", "c_drag", "c_roll"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive, got {getattr(self, field_name)}")
        if not 0 <= self.trail_braking <= 100:
            raise ValueError(f"trail_braking must be in the range 0-100, got {self.trail_braking}")
        if not 0 <= self.corner_acc <= 100:
            raise ValueError(f"corner_acc must be in the range 0-100, got {self.corner_acc}")
        self.trail_braking = DriverExperience(self.trail_braking)
        self.corner_acc = CornerAcceleration(self.corner_acc)

    @classmethod
    def from_toml(cls, file_name):
        """Load car parameters from TOML file."""
        with open(file_name, "rb") as f:
            return cls(**tomllib.load(f))

    @functools.cached_property
    def file_name(self):
        """Return the file name derived from the car's name."""
        return self.name.replace(" ", "_")

    @functools.cached_property
    def P_engine_in_watt(self):
        """Return the engine power in watts."""
        return self.P_engine / 1.3410 * 1000

    @functools.cached_property
    def rolling_drag(self):
        """Return the rolling drag force."""
        return self.c_roll * 9.81

    def get_acceleration(self, v, acc_lat):
        """
        Calculate the maximum possible acceleration given the current speed and lateral acceleration.

        Args:
            v (float): The current speed.
            acc_lat (float): The current lateral acceleration.

        Returns:
            float: The calculated longitudinal acceleration.
        """
        n = self.corner_acc / 50
        acc_max = (self.acc_limit) * (1 - (np.abs(acc_lat) / self.lat_limit) ** n) ** (1 / n)
        force_engine = v and self.P_engine_in_watt / v or 0
        acc_max = force_engine and min(acc_max, force_engine / self.mass) or acc_max
        aero_drag = v**2 * self.c_drag / 2 / self.mass
        return acc_max - aero_drag - self.rolling_drag

    def get_deceleration(self, v, acc_lat):
        """
        Calculate the maximum possible deceleration given the current speed and lateral acceleration.

        Args:
            v (float): The current speed.
            acc_lat (float): The current lateral acceleration.

        Returns:
            float: The calculated longitudinal deceleration.
        """
        n = self.trail_braking / 50
        max_dec_grip = (self.dec_limit) * (1 - (np.abs(acc_lat) / self.lat_limit) ** n) ** (1 / n)
        aero_drag = v**2 * self.c_drag / 2 / self.mass
        return max_dec_grip + aero_drag + self.rolling_drag

    def performance_envelope(self, v):
        """
        Calculate the performance envelope of the car at a given speed.

        Args:
            v (float): The current speed.

        Returns:
            tuple: A tuple containing the lateral accelerations and corresponding
                   longitudinal accelerations and decelerations.
        """
        acc_lat = np.linspace(-self.lat_limit, self.lat_limit, 100)
        acc = self.get_acceleration(v=0, acc_lat=acc_lat)
        dec = self.get_deceleration(v=v, acc_lat=acc_lat)
        return acc_lat, np.column_stack((acc, -dec))
