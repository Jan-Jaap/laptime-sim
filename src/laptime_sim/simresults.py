import dataclasses
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

OUTPUT_COLUMNS_NAMES = dict(
    distance="Distance (m)",
    speed="Speed (m/s)",
    speed_kph="Speed (km/h)",
    a_lon="Longitudinal acceleration (m/s2)",
    a_lat="Lateral acceleration (m/s2)",
    curvature_radius_x="Curvature radius (m)",
    time="Timestamp",
)


def format_laptime(seconds: float) -> str:
    """
    Formats a laptime in seconds as "MM:SS.mmm".

    Parameters:
    - seconds: float - The laptime in seconds.

    Returns:
    - str: The laptime in "MM:SS.mmm" format.
    """
    return f"{seconds % 3600 // 60:02.0f}:{seconds % 60:06.03f}"


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

    line_coordinates: NDArray[np.float64]
    dt: NDArray[np.float64]
    speed: NDArray[np.float64]
    Nk: NDArray[np.float64]  # should be (N, 3) array of normal vectors in car frame with magnitude 1/R
    ds: NDArray[np.float64]

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
            str: The best laptime in "MM:SS.mmm" format.
        """
        return format_laptime(self.laptime)

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

    @cached_property
    def curvature_radius_x(self) -> NDArray:
        """
        The curvature in the x direction.

        Returns:
            NDArray: The curvature in the x direction.
        """
        return 1 / self.Nk[:, 0]

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
            The DataFrame is indexed by the 'time' column. The column names are
            renamed using the OUTPUT_COLUMNS_NAMES dict.

        """
        import pandas as pd

        df = pd.DataFrame()
        df["time"] = self.dt.cumsum()
        df["distance"] = self.distance
        df["a_lat"] = self.a_lat
        df["a_lon"] = self.a_lon
        df["curvature_radius_x"] = self.curvature_radius_x
        df["speed"] = self.speed
        df["speed_kph"] = self.speed * 3.6
        return df.set_index("time").rename(columns=OUTPUT_COLUMNS_NAMES)
