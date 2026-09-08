import functools
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
from geopandas import GeoDataFrame, GeoSeries, read_parquet
from numpy.typing import NDArray
from shapely import LineString, Point

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults
from laptime_sim.simulate import _simulate_kernel, simulate
from laptime_sim.track import Track, _interpolate_positions

MAX_DEVIATION = 0.1
MAX_DEVIATION_LENGTH = 60
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement
BORDER_CLEARANCE_M: float = 0.85
N_THREADS = os.cpu_count() or 4

random_generator = np.random.default_rng()

_thread_pool: ThreadPoolExecutor | None = None


def _get_thread_pool() -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=N_THREADS)
    return _thread_pool


@nb.njit(cache=True, nogil=True)
def _batch_kernel(  # noqa: PLR0913, PLR0914
    current_position,
    width,
    pos_min,
    pos_max,
    is_circular,
    idx_len,
    n_coords,
    left_coords,
    right_coords,
    slope,
    locations,
    lengths,
    deviations,
    lat_limit,
    mass,
    acc_limit,
    dec_limit,
    c_drag,
    c_roll,
    corner_acc,
    trail_braking,
    P_engine_in_watt,
):
    """
    Builds, clips, interpolates and simulates a chunk of random candidate lines
    without holding the GIL, so chunks can run in parallel on threads.

    Returns: (laptimes, positions, perturbations)
    """
    m = locations.shape[0]
    laptimes = np.empty(m)
    positions = np.empty((m, idx_len))
    perturbations = np.empty((m, idx_len))
    coords = np.empty((n_coords, 3))
    for k in range(m):
        location = locations[k]
        length = lengths[k]
        deviation = deviations[k]
        perturbation = perturbations[k]
        for j in range(idx_len):
            perturbation[j] = 0.0
        line_adjust = (1 + np.cos(np.linspace(-np.pi, np.pi, length))) / 2
        if is_circular:
            shift = location - length // 2
            for a in range(length):
                perturbation[(a + shift) % idx_len] = line_adjust[a]
        else:
            if location + length > idx_len:
                length = idx_len - location
            for a in range(length):
                perturbation[location + a] = line_adjust[a]

        new_line = positions[k]
        for j in range(idx_len):
            v = current_position[j] + perturbation[j] * deviation / width[j]
            if v < pos_min[j]:
                v = pos_min[j]
            elif v > pos_max[j]:
                v = pos_max[j]
            new_line[j] = v

        _interpolate_positions(left_coords, right_coords, new_line[:idx_len], coords[:idx_len])
        if is_circular:
            for j in range(3):
                coords[n_coords - 1, j] = coords[0, j]

        dt, speed, Nk, ds = _simulate_kernel(
            coords,
            slope,
            lat_limit,
            mass,
            acc_limit,
            dec_limit,
            c_drag,
            c_roll,
            corner_acc,
            trail_braking,
            P_engine_in_watt,
        )
        laptimes[k] = dt.sum()
    return laptimes, positions, perturbations


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
        self._heatmap_sum = float(self._heatmap.sum())

    def load_file(self, file_path: Path) -> None:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} doesn't exist")

        data = read_parquet(file_path)
        assert self.track.name == data.iloc[0].track_name
        data = data.to_crs(data.estimate_utm_crs())
        coordinates = data.get_coordinates(include_z=False).to_numpy(na_value=0)
        # check if first and last coordinates are the same

        assert len(coordinates) == self.track.len, (
            f"Track length {self.track.len} != coordinates length {len(coordinates)}"
        )

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

        tail_args = (
            self.width,
            self.position_clearance,
            1 - self.position_clearance,
            self.track.is_circular,
            idx_len,
            self.track.len,
            self.track.left_coords,
            self.track.right_coords,
            self.track.slope,
        )
        car_args = (
            car.lat_limit,
            car.mass,
            car.acc_limit,
            car.dec_limit,
            car.c_drag,
            car.c_roll,
            car.corner_acc,
            car.trail_braking,
            car.P_engine_in_watt,
        )

        def run_chunk(chunk_args):
            locs, lens, devs = chunk_args
            return _batch_kernel(self._position, *tail_args, locs, lens, devs, *car_args)

        if batch_size == 1:
            laptimes, positions, perturbations = run_chunk((locations, lengths, deviations))
        else:
            chunk_size = -(-batch_size // N_THREADS)
            chunks = []
            for start in range(0, batch_size, chunk_size):
                chunk_slice = slice(start, start + chunk_size)
                chunks.append((locations[chunk_slice], lengths[chunk_slice], deviations[chunk_slice]))
            results = list(_get_thread_pool().map(run_chunk, chunks))
            laptimes = np.concatenate([result[0] for result in results])
            positions = np.vstack([result[1] for result in results])
            perturbations = np.vstack([result[2] for result in results])

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
