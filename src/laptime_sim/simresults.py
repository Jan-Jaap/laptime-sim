import dataclasses
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
import pandas as pd

OUTPUT_COLUMNS_NAMES = dict(
    distance="Distance (m)",
    speed="Speed (m/s)",
    speed_kph="Speed (km/h)",
    a_lon="Longitudinal acceleration (m/s2)",
    a_lat="Lateral acceleration (m/s2)",
    time="Timestamp",
)


@dataclasses.dataclass(frozen=True)
class SimResults:
    """
    Represents the results of a simulation.

    Attributes:
        line_coordinates (NDArray): The coordinates of the raceline.
        dt (NDArray): The time differences between consecutive points.
        speed (NDArray): The speed at each point.
        Nk (NDArray): The normal vectors in car frame with magnitude 1/R. Shape is (N, 3).
        ds (NDArray): The distances between consecutive points.

    Properties:
        laptime (float): The total time of the simulation in seconds.
    """

    line_coordinates: NDArray
    dt: NDArray
    speed: NDArray
    Nk: NDArray  # should be (N, 3) array of normal vectors in car frame with magnitude 1/R
    ds: NDArray

    @cached_property
    def laptime(self) -> float:
        """
        The total time of the simulation in seconds.

        Returns:
            float: The total time of the simulation in seconds.
        """
        return self.dt.squeeze().sum()

    def __str__(self) -> str:
        """
        Returns a string representing the simulation results.

        Returns:
            str: The string representing the simulation results.
        """
        return f"{self.laptime % 3600 // 60:02.0f}:{self.laptime % 60:06.03f}"

    @cached_property
    def a_lat(self) -> NDArray:
        """
        The lateral acceleration at each point.

        Returns:
            NDArray: The lateral acceleration at each point.
        """
        return -(self.speed**2) * self.Nk[:, 0]

    @cached_property
    def a_lon(self) -> NDArray:
        """
        The longitudinal acceleration at each point.

        Returns:
            NDArray: The longitudinal acceleration at each point.
        """
        return np.gradient(self.speed, self.distance) * self.speed

    @cached_property
    def distance(self) -> NDArray:
        """
        The cumulative sum of the distances between consecutive points.

        Returns:
            NDArray: The cumulative sum of the distances between consecutive points.
        """
        return self.ds.cumsum() - self.ds[0]

    @cached_property
    def speed_kph(self) -> NDArray:
        """
        The speed in kilometers per hour at each point.

        Returns:
            NDArray: The speed in kilometers per hour at each point.
        """
        return self.speed * 3.6

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame containing the simulation results.

        Returns:
            pd.DataFrame: The DataFrame with the following columns:
                - time: The cumulative sum of the time differences between consecutive timestamps.
                - distance: The cumulative sum of the distances between consecutive points.
                - a_lat: The lateral acceleration.
                - a_lon: The longitudinal acceleration.
                - speed: The speed at each point.
                - speed_kph: The speed in kilometers per hour at each point.

        Note:
            The DataFrame is indexed by the 'time' column. The column names are renamed using the OUTPUT_COLUMNS_NAMES dict.

        """
        df = pd.DataFrame()
        df["time"] = self.dt.cumsum()
        df["distance"] = self.distance
        df["a_lat"] = self.a_lat
        df["a_lon"] = self.a_lon
        df["speed"] = self.speed
        df["speed_kph"] = self.speed * 3.6
        return df.set_index("time").rename(columns=OUTPUT_COLUMNS_NAMES)
