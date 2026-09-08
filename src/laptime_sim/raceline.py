import functools
from dataclasses import dataclass, field

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from numpy.typing import NDArray
from shapely import LineString, Point

from laptime_sim.car import Car
from laptime_sim.optimizer import BatchGeometry, run_batch
from laptime_sim.simresults import SimResults, format_laptime
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
    _geometry: BatchGeometry = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._position = np.zeros_like(self.width)
        self._heatmap = np.ones_like(self.width)
        self._heatmap_sum = float(self._heatmap.sum())
        self._geometry = BatchGeometry(
            width=self.width,
            pos_min=self.position_clearance,
            pos_max=1 - self.position_clearance,
            is_circular=bool(self.track.is_circular),
            idx_len=len(self.width),
            n_coords=self.track.len,
            left_coords=self.track.left_coords,
            right_coords=self.track.right_coords,
            slope=self.track.slope,
        )

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
        return format_laptime(self.best_time)

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
        - NDArray: The track width.
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
        self.try_random_lines(car, 1)

    def try_random_lines(self, car: Car, batch_size: int) -> None:
        """
        Simulates a batch of random candidate lines in parallel on threads and keeps
        the best one if it improves on the current best line.

        Parameters:
        - car: Car - The car to simulate the raceline for.
        - batch_size: int - Number of random candidate lines to test.

        Returns:
        - None
        """
        idx_len = len(self.width)

        # Choose random locations and lengths for the deviations
        cdf = self._heatmap.cumsum()
        locations = np.minimum(
            cdf.searchsorted(random_generator.random(batch_size) * cdf[-1], side="right"),
            idx_len - 1,
        )
        lengths = random_generator.integers(3, MAX_DEVIATION_LENGTH, size=batch_size, endpoint=True)
        deviations = random_generator.uniform(-1, 1, batch_size) * np.maximum(
            lengths / MAX_DEVIATION_LENGTH, MAX_DEVIATION
        )

        laptimes, positions, perturbations = run_batch(
            self._geometry, self._position, locations, lengths, deviations, car
        )

        try:
            winner = int(np.nanargmin(laptimes))
        except ValueError:
            winner = -1

        if winner >= 0 and np.isfinite(improvement := self.best_time - laptimes[winner]) and improvement > 0:
            self._position = positions[winner].copy()
            self.best_time = laptimes[winner]

            self._heatmap += perturbations[winner] * improvement * 1e3
            self._heatmap_sum += improvement * 1e3 * float(perturbations[winner].sum())
            self.progress_rate += improvement
            return

        # If the new line is not better, slowly decay the heatmap and progress rate
        self._heatmap += 0.00015
        self._heatmap /= 1.00015
        self._heatmap_sum += idx_len * 0.00015
        self._heatmap_sum /= 1.00015
        self.progress_rate *= F_ANNEAL
