import os
import functools
from enum import IntEnum
from pathlib import Path
from pydantic import BaseModel, Field, PositiveFloat
import numpy as np
import toml


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


class Car(BaseModel):
    """
    A class representing a car with its physical properties and capabilities.
    """

    name: str = Field(None, description="Name of the car")
    mass: PositiveFloat = Field(..., description="Mass of the car in kilograms")
    P_engine: PositiveFloat = Field(..., description="Engine power in horsepower")
    acc_limit: PositiveFloat = Field(..., description="Longitudinal acceleration limit (grip limit) in m/s²")
    dec_limit: PositiveFloat = Field(..., description="Deceleration limit (grip limit under braking) in m/s²")
    lat_limit: PositiveFloat = Field(..., description="Lateral acceleration limit (grip limit) in m/s²")
    c_drag: PositiveFloat = Field(..., description="Drag coefficient")
    c_roll: PositiveFloat = Field(..., description="Rolling resistance coefficient")
    trail_braking: DriverExperience = Field(ge=0, le=100, description="Trailbraking factor")
    corner_acc: CornerAcceleration = Field(ge=0, le=100, description="Corner acceleration factor")

    @classmethod
    def from_toml(cls, file_name):
        """Load car parameters from TOML file."""
        return cls(**toml.load(file_name))

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


def car_list(path_cars: Path | str) -> list[Car]:
    path_cars = Path(path_cars)
    return [Car.from_toml(file) for file in sorted(path_cars.glob("*.toml"))]
