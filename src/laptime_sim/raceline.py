import os

import functools
import numpy as np
from geopandas import GeoDataFrame, read_parquet

from shapely import LineString
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
from laptime_sim.car import Car
from laptime_sim.simresults import SimResults
from laptime_sim.track import Track
from laptime_sim.simulate import RacelineSimulator


SMOOTHING_WINDOW = 40
MAX_DEVIATION_LENGTH = 40
MAX_DEVIATION = 0.1


@dataclass(frozen=True)
class LineParameters:
    location: int
    length: int
    deviation: float

    @classmethod
    def from_heatmap(cls, p):
        location = np.random.choice(len(p), p=p / sum(p))
        length = np.random.randint(1, MAX_DEVIATION_LENGTH)
        deviation = np.random.randn() * MAX_DEVIATION
        return cls(location, length, deviation)


# annealing factor
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement


@dataclass
class Raceline:
    track: Track
    car: Car
    simulator: RacelineSimulator = field(default_factory=RacelineSimulator)
    # filename_results: os.PathLike | str = None
    line_position: np.ndarray = None
    heatmap: np.ndarray = None
    clearance_meter: float = 0.85
    progress: float = 1.0
    best_time: float = None

    def __post_init__(self):

        if self.line_position is None:
            # TODO: here we could use a line_pos from a different car or run
            self.initialize_line(self.track, smoothing_window=SMOOTHING_WINDOW, poly_order=5)
            # self.line_pos = np.zeros_like(self.track.width) + 0.5

        if self.simulator is not None and self.best_time is None:
            self.best_time = self.simulate().laptime

        if self.heatmap is None:
            self.heatmap = np.ones_like(self.line_position)

    @classmethod
    def from_geodataframe(cls, data: GeoDataFrame, all_cars, all_tracks):

        track: Track = [track for track in all_tracks if track.name == data.iloc[0].track_name][0]
        car: Car = [car for car in all_cars if car.name == data.iloc[0].car][0]

        data = data.to_crs(track.crs)
        line_coords = data.get_coordinates(include_z=False).to_numpy(na_value=0)

        simulator = RacelineSimulator()

        return cls(
            track=track,
            car=car,
            simulator=simulator,
            line_position=track.parameterize_line_coordinates(line_coords),
            best_time=data.iloc[0].best_time,
        )

    def load_line(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} doesn't exist")
        results = read_parquet(filename)
        results = results.to_crs(self.track.crs)

        assert self.track.name == results.iloc[0].track_name
        assert self.car.name == results.iloc[0].car
        self.best_time = results.iloc[0].best_time
        line_coords = results.get_coordinates(include_z=False).to_numpy(na_value=0)
        self.line_position = self.track.parameterize_line_coordinates(line_coords)

        return self

    def save_line(self, filename) -> None:
        """
        Saves the results to a parquet file.

        Parameters:
            filename: str - The name of the output file to save the results to.

        Returns:
            None
        """
        if filename is None:
            raise FileNotFoundError("no output filename provided")
        self.get_dataframe().to_parquet(filename)

    def get_dataframe(self) -> GeoDataFrame:
        geom = LineString(self.track.line_coordinates(self.line_position).tolist())
        data = dict(
            track_name=self.track.name,
            car=self.car.name,
            best_time=self.best_time,
        )
        return GeoDataFrame.from_dict(data=[data], geometry=[geom], crs=self.track.crs).to_crs(epsg=4326)

    @property
    def best_time_str(self):
        """
        Property method that formats the best lap time in hours, minutes, and seconds.
        Returns a formatted string of the best lap time.
        """
        if self.best_time:
            return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

    @functools.cached_property
    def _position_clearance(self):
        return self.track.normalize_distance(self.clearance_meter)

    def simulate(self, line_pos: np.ndarray = None) -> SimResults:

        if line_pos is None:
            return self.simulator.run(
                car=self.car,
                line_coordinates=self.track.line_coordinates(self.line_position),
                slope=self.track.slope,
            )

        sim_results = self.simulator.run(
            car=self.car,
            line_coordinates=self.track.line_coordinates(line_pos),
            slope=self.track.slope,
        )

        if sim_results.laptime < self.best_time:
            improvement = self.best_time - sim_results.laptime
            deviation = np.abs(self.line_position - line_pos)
            max_deviation = max(deviation)
            if max_deviation > 0:
                self.heatmap += deviation / max_deviation * improvement * 1e3
            self.best_time = sim_results.laptime
            self.progress += improvement
            self.line_position = line_pos

        self.heatmap = (self.heatmap + 0.0015) / 1.0015  # slowly to one
        self.progress *= F_ANNEAL  # slowly to zero

        return sim_results

    def simulate_new_line(self, line_param=None) -> None:
        if line_param is None:
            line_param = LineParameters.from_heatmap(self.heatmap)

        new_line = self.get_new_line(line_param=line_param)

        return self.simulate(new_line)

    def get_new_line(self, line_param: LineParameters):
        line_adjust = 1 - np.cos(np.linspace(0, 2 * np.pi, line_param.length))
        position = np.zeros_like(self.line_position)
        position[: line_param.length] = line_adjust * line_param.deviation
        position = np.roll(position, line_param.location - line_param.length // 2)
        test_line = self.line_position + position / self.track.width
        return np.clip(
            test_line,
            a_min=self._position_clearance,
            a_max=1 - self._position_clearance,
        )

    def initialize_line(self, track: Track, smoothing_window: int = 20, poly_order: int = 5):
        """
        Initializes the raceline by generating a smoothed line of coordinates
        along the track.

        Parameters:
        - track: Track - The track to initialize the raceline on.
        """
        coordinates = track.line_coordinates(np.full_like(self.track.width, 0.5), include_z=False)
        x, y = coordinates.T
        smoothed_x = savgol_filter(x, smoothing_window, poly_order, mode="wrap")
        smoothed_y = savgol_filter(y, smoothing_window, poly_order, mode="wrap")
        self.line_position = track.parameterize_line_coordinates(np.array([smoothed_x, smoothed_y]).T)
