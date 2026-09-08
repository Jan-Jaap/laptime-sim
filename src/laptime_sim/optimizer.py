import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numba as nb
import numpy as np
from numpy.typing import NDArray

from laptime_sim.car import Car
from laptime_sim.simulate import simulate_kernel
from laptime_sim.track import interpolate_positions

N_THREADS = os.cpu_count() or 4

_thread_pool: ThreadPoolExecutor | None = None


def _get_thread_pool() -> ThreadPoolExecutor:
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=N_THREADS)
    return _thread_pool


@dataclass(frozen=True)
class BatchGeometry:
    """
    Immutable snapshot of the track arrays needed to build and simulate
    candidate lines, so the batch kernel does not need the Track object.
    """

    width: NDArray
    pos_min: NDArray
    pos_max: NDArray
    is_circular: bool
    idx_len: int
    n_coords: int
    left_coords: NDArray
    right_coords: NDArray
    slope: NDArray


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

        interpolate_positions(left_coords, right_coords, new_line[:idx_len], coords[:idx_len])
        if is_circular:
            for j in range(3):
                coords[n_coords - 1, j] = coords[0, j]

        dt, speed, Nk, ds = simulate_kernel(
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


def run_batch(
    geometry: BatchGeometry,
    current_position: NDArray,
    locations: NDArray,
    lengths: NDArray,
    deviations: NDArray,
    car: Car,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Simulates a batch of random candidate lines in parallel on threads.

    Parameters:
    - geometry: BatchGeometry - Immutable track arrays snapshot.
    - current_position: NDArray - The current line positions.
    - locations: NDArray - Candidate deviation locations.
    - lengths: NDArray - Candidate deviation lengths.
    - deviations: NDArray - Candidate deviation magnitudes.
    - car: Car - The car to simulate the candidates for.

    Returns:
    - tuple: (laptimes, positions, perturbations)
    """
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
        return _batch_kernel(
            current_position,
            geometry.width,
            geometry.pos_min,
            geometry.pos_max,
            geometry.is_circular,
            geometry.idx_len,
            geometry.n_coords,
            geometry.left_coords,
            geometry.right_coords,
            geometry.slope,
            locs,
            lens,
            devs,
            *car_args,
        )

    batch_size = locations.shape[0]
    if batch_size <= N_THREADS:
        return run_chunk((locations, lengths, deviations))

    chunk_size = -(-batch_size // N_THREADS)
    chunks = []
    for start in range(0, batch_size, chunk_size):
        chunk_slice = slice(start, start + chunk_size)
        chunks.append((locations[chunk_slice], lengths[chunk_slice], deviations[chunk_slice]))
    results = list(_get_thread_pool().map(run_chunk, chunks))
    laptimes = np.concatenate([result[0] for result in results])
    positions = np.vstack([result[1] for result in results])
    perturbations = np.vstack([result[2] for result in results])
    return laptimes, positions, perturbations
