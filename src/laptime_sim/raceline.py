import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from geopandas import GeoDataFrame, GeoSeries, read_parquet
from shapely import LineString, Point

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults
from laptime_sim.simulate import simulate
from laptime_sim.track import Track

MAX_DEVIATION = 0.1
MAX_DEVIATION_LENGTH = 60
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement
BORDER_CLEARANCE_M: float = 0.85

random_generator = np.random.default_rng()


@dataclass
class Raceline:
    track: Track
    best_time: float = np.inf
    progress_rate: float = 1.0
    _position: NDArray = field(init=False, repr=False)
    _heatmap: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._position = np.zeros_like(self.width)
        self._heatmap = np.ones_like(self.width)

    def load_file(self, file_path: Path) -> None:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exist")

        data = read_parquet(file_path)
        assert self.track.name == data.iloc[0].track_name
        data = data.to_crs(data.estimate_utm_crs())
        coordinates = data.get_coordinates(include_z=False).to_numpy(na_value=0)
        # check if first and last coordinates are the same

        assert len(coordinates) == self.track.len, f"Track length {self.track.len} != coordinates length {len(coordinates)}"

        if self.track.is_circular:
            assert np.array_equal(coordinates[0], coordinates[-1]), "First and last coordinates are not the same"

        self.set_coordinates(coordinates)

    def set_coordinates(self, coordinates: NDArray) -> None:
        self._position = self.track.position_from_coordinates(coordinates)

    def get_coordinates(self) -> NDArray:
        """Returns the coordinates of the raceline."""
        return self.track.coordinates_from_position(self._position)

    def best_time_str(self) -> str:
        """
        Returns the best laptime as a formatted string.

        Returns:
        - str: The best laptime in "MM:SS.mmm" format.
        """
        return f"{self.best_time % 3600 // 60:02.0f}:{self.best_time % 60:06.03f}"

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
        coordinates: list = self.get_coordinates().tolist()
        geom = LineString(coordinates)

        data = dict(track_name=[track_name], car=[car_name], geometry=[geom])
        return GeoDataFrame.from_dict(data, crs=self.track.crs).to_crs(epsg=4326)

    def get_point(self, index: int) -> GeoSeries:
        """
        Returns the coordinates of the raceline at a given index.
        Parameters:
        - index: int - The index of the raceline point.
        Returns:
        - GeoSeries: The coordinates of the raceline at the given index.
        """
        coordinates = self.get_coordinates()

        return GeoSeries(Point([coordinates[index]]), crs=self.track.crs)

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

    def simulate(self, car: Car) -> SimResults:
        """
        Updates the raceline with a new simulation for the given car.

        Parameters:
        - car: Car - The car to simulate the raceline for.

        Returns:
        - SimResults: The simulation results.
        """
        sim_results = simulate(car, self.get_coordinates(), self.track.slope)

        if sim_results.laptime < self.best_time:
            self.best_time = sim_results.laptime
        return sim_results

    def try_random_line(self, car: Car) -> None:
        """
        Simulates a new line with a random deviation and length.

        Parameters:
        - car: Car - The car to simulate the raceline for.

        Returns:
        - None
        """

        idx_len = len(self.width)

        # Choose a random location and length for the deviation
        location_index = random_generator.choice(len(self._heatmap), p=self._heatmap / sum(self._heatmap))
        length = random_generator.integers(3, MAX_DEVIATION_LENGTH, endpoint=True)
        deviation = random_generator.uniform(-1, 1) * max(length / MAX_DEVIATION_LENGTH, MAX_DEVIATION)

        # Create a position array with the deviation
        position = np.zeros(idx_len)
        line_adjust = (1 + np.cos(np.linspace(-np.pi, np.pi, length))) / 2
        if self.track.is_circular:
            position[:length] = line_adjust
            position = np.roll(position, location_index - length // 2)
        else:
            if location_index + length > idx_len:
                length = idx_len - location_index
            position[location_index : location_index + length] = line_adjust[:length]

        # Calculate the new line position and coordinates
        new_line_position = self.clip_line(self._position + position * deviation / self.width)
        new_coordinates = self.track.coordinates_from_position(new_line_position)

        # Simulate the new line and check if it's better
        laptime = simulate(car, new_coordinates, self.track.slope).laptime
        if improvement := self.best_time - laptime > 0:
            self._position = new_line_position
            self.best_time = laptime

            self._heatmap += position * improvement * 1e3
            self.progress_rate += improvement
            return

        # If the new line is not better, slowly decay the heatmap and progress rate
        self._heatmap = (self._heatmap + 0.00015) / 1.00015
        self.progress_rate *= F_ANNEAL
