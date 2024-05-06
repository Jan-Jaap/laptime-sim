import dataclasses
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


@dataclasses.dataclass
class SimResults:
    line_coordinates: np.ndarray
    dt: np.ndarray
    speed: np.ndarray
    Nk: np.ndarray
    ds: np.ndarray

    @property
    def laptime(self):
        return sum(self.dt)

    def __str__(self) -> str:
        return f"{self.laptime % 3600 // 60:02.0f}:{self.laptime % 60:06.03f}"

    @property
    def a_lat(self):
        return -(self.speed**2) * self.Nk[:, 0]

    @property
    def a_lon(self):
        return np.gradient(self.speed, self.distance) * self.speed

    @property
    def distance(self):
        return self.ds.cumsum() - self.ds[0]

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["time"] = self.dt.cumsum()
        # df1 = pd.DataFrame(data=track_session.left_coords(), columns=["x", "y", "z"]).add_prefix("left_")
        # df2 = pd.DataFrame(data=track_session.right_coords(), columns=["x", "y", "z"]).add_prefix("right_")
        # df3 = pd.DataFrame(data=track_session.line_coords(), columns=["x", "y", "z"]).add_prefix("line_")
        # df = pd.concat([df, df1, df2, df3], axis=1)
        df["distance"] = self.distance
        df["a_lat"] = self.a_lat
        df["a_lon"] = self.a_lon
        df["speed"] = self.speed
        df["speed_kph"] = self.speed * 3.6
        return df.set_index("time").rename(columns=OUTPUT_COLUMNS_NAMES)
