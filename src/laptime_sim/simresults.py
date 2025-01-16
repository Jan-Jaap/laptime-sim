import dataclasses
from functools import cached_property

import numpy as np
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
    line_coordinates: np.ndarray
    dt: np.ndarray
    speed: np.ndarray
    Nk: np.ndarray  # should be (N, 3) array of normal vectors in car frame with magnitude 1/R
    ds: np.ndarray

    @cached_property
    def laptime(self) -> float:
        return self.dt.squeeze().sum()

    def __str__(self) -> str:
        return f"{self.laptime % 3600 // 60:02.0f}:{self.laptime % 60:06.03f}"

    @cached_property
    def a_lat(self):
        return -(self.speed**2) * self.Nk[:, 0]

    @cached_property
    def a_lon(self):
        return np.gradient(self.speed, self.distance) * self.speed

    @cached_property
    def distance(self):
        return self.ds.cumsum() - self.ds[0]

    @cached_property
    def speed_kph(self):
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
