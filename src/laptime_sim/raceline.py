import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray
from geopandas import GeoDataFrame, read_parquet
from shapely import LineString

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults
from laptime_sim.simulate import simulate
from laptime_sim.track import Track

MAX_DEVIATION = 0.1
MAX_DEVIATION_LENGTH = 60
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement
BORDER_CLEARANCE_M: float = 0.85

random_generator = np.random.default_rng()


def empty_array() -> NDArray:
    """Returns an empty NumPy array of type float64."""
    return np.empty(0, dtype=np.float64)


@dataclass
class Raceline:
    track: Track
    line_position: NDArray = field(default_factory=empty_array)  # np.empty(0)
    best_time: float = np.inf
    progress_rate: float = 1.0
    _heatmap: NDArray = field(init=False)

    def __post_init__(self):
        """
        Initializes the Raceline class by setting the initial line position and heatmap.
        """
        if self.line_position.size == 0:
            initial_line = self.track.initial_line()
            line_position = self.track.position_from_coordinates(initial_line)
            self.line_position = self.clip_line(line_position)
        self._heatmap = np.ones_like(self.line_position)

    @classmethod
    def from_coordinates(cls, track: Track, line_coords: NDArray) -> Self:
        """
        Creates a Raceline instance from track and line coordinates.

        Parameters:
        - track: Track - The track to initialize the raceline on.
        - line_coords: NDArray - The coordinates of the raceline.

        Returns:
        - Raceline: An instance of the Raceline class.
        """
        return cls(track=track, line_position=track.position_from_coordinates(line_coords))

    @classmethod
    def from_geodataframe(cls, track: Track, data: GeoDataFrame) -> Self:
        """
        Creates a Raceline instance from a track and a GeoDataFrame.

        Parameters:
        - track: Track - The track to initialize the raceline on.
        - data: GeoDataFrame - The GeoDataFrame containing raceline data.

        Returns:
        - Raceline: An instance of the Raceline class.
        """
        data = data.to_crs(data.estimate_utm_crs())
        line_coords = data.get_coordinates(include_z=False).to_numpy(na_value=0)
        return cls.from_coordinates(track=track, line_coords=line_coords)

    @classmethod
    def from_file(cls, track: Track, file_path: Path) -> Self:
        """
        Creates a Raceline instance from a file.

        Parameters:
        - track: Track - The track to initialize the raceline on.
        - file_path: Path - The path to the file containing raceline data.

        Returns:
        - Raceline: An instance of the Raceline class.

        Raises:
        - FileNotFoundError: If the provided file path does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exist")

        data = read_parquet(file_path)
        assert track.name == data.iloc[0].track_name
        return cls.from_geodataframe(track, data)

    def coordinates(self) -> NDArray:
        """
        Converts raceline position to coordinates.

        Parameters:
        - position: NDArray  - The position array to convert.

        Returns:
        - NDArray: The coordinates corresponding to the position.
        """

        return self.track.coordinates_from_position(self.line_position)

    def best_time_str(self) -> str:
        """
        Returns the best laptime as a formatted string.

        Returns:
        - str: The best laptime in "MM:SS.mmm" format.
        """
        return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

    def filename(self, car_name) -> Path:
        """
        Generates a file name for the raceline data.

        Parameters:
        - car_name: str - The name of the car.

        Returns:
        - Path: The path with the generated file name.
        """
        return Path(f"{car_name}_{self.track.name}_simulated.parquet")

    def save_line(self, file_path: Path, car_name: str) -> None:
        """
        Saves the raceline data to a file.

        Parameters:
        - file_path: Path - The path where the data will be saved.
        - car_name: str - The name of the car.

        Raises:
        - FileNotFoundError: If no file path is provided.
        """
        if file_path is None:
            raise FileNotFoundError("no output filename provided")

        self.dataframe(track_name=self.track.name, car_name=car_name).to_parquet(file_path)

    def dataframe(self, track_name, car_name) -> GeoDataFrame:
        """
        Converts raceline data to a GeoDataFrame.

        Parameters:
        - track_name: str - The name of the track.
        - car_name: str - The name of the car.

        Returns:
        - GeoDataFrame: The raceline data as a GeoDataFrame.
        """
        coordinates: list = self.coordinates().tolist()
        geom = LineString(coordinates)

        data = dict(track_name=[track_name], car=[car_name], geometry=[geom])
        return GeoDataFrame.from_dict(data, crs=self.track.crs).to_crs(epsg=4326)

    @functools.cached_property
    def width(self):
        """
        Returns the track width, excluding the last point if the track is circular.

        Returns:
        - NDArray: The width of the track.
        """
        if self.track.is_circular:
            return self.track.width[:-1]

        return self.track.width

    @functools.cached_property
    def position_clearance(self):
        """
        Computes the position clearance based on border clearance and track width.

        Returns:
        - NDArray: The position clearance values.
        """
        return BORDER_CLEARANCE_M / self.width

    def clip_line(self, line_position: NDArray) -> NDArray:
        """
        Clips the line position within the track boundaries.

        Parameters:
        - line_position: NDArray - The line position array to clip.

        Returns:
        - NDArray: The clipped line position.
        """
        return np.clip(line_position, a_min=self.position_clearance, a_max=1 - self.position_clearance)

    def update(self, car: Car) -> SimResults:
        """
        Updates the raceline with a new simulation for the given car.

        Parameters:
        - car: Car - The car to simulate the raceline for.

        Returns:
        - SimResults: The simulation results.
        """
        sim_results = simulate(car, self.coordinates(), self.track.slope)

        if sim_results.laptime < self.best_time:
            self.best_time = sim_results.laptime
        return sim_results

    def simulate_new_line(self, car: Car) -> None:
        """
        Simulates a new line with a random deviation and length.

        Parameters:
        - car: Car - The car to simulate the raceline for.

        Returns:
        - None
        """

        location = random_generator.choice(len(self._heatmap), p=self._heatmap / sum(self._heatmap))
        length = random_generator.integers(3, MAX_DEVIATION_LENGTH, endpoint=True)
        deviation = random_generator.uniform(-1, 1) * max(length / MAX_DEVIATION_LENGTH, MAX_DEVIATION)
        line_adjust = (1 + np.cos(np.linspace(-np.pi, np.pi, length))) / 2
        position = np.zeros(len(self.line_position))

        if self.track.is_circular:  # ensure first and last position value are the same
            position[:length] = line_adjust
            position = np.roll(position, location - length // 2)
        else:  # Start position is fixed.  Finish position can change.
            if location + length > len(self.line_position):
                length = len(self.line_position) - location
            position[location : location + length] = line_adjust[:length]

        new_line_position = self.clip_line(self.line_position + position * deviation / self.width)
        new_coordinates = self.track.coordinates_from_position(new_line_position)
        laptime = simulate(car, new_coordinates, self.track.slope).laptime
        if laptime < self.best_time:
            improvement = self.best_time - laptime
            self._heatmap += position * improvement * 1e3

            self.best_time = laptime
            self.progress_rate += improvement
            self.line_position = new_line_position
            return

        self._heatmap = (self._heatmap + 0.00015) / 1.00015  # slowly to one
        self.progress_rate *= F_ANNEAL  # slowly to zero
