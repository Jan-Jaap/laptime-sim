import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from numpy.typing import NDArray

from laptime_sim.car import Car
from laptime_sim.simulate import simulate_kernel
from laptime_sim.track import interpolate_positions

if TYPE_CHECKING:
    from laptime_sim.raceline import Raceline

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


MAX_DEVIATION = 0.1
MAX_DEVIATION_LENGTH = 60
F_ANNEAL = 0.01 ** (1 / 10000)  # from 1 to 0.01 in 10000 iterations without improvement

random_generator = np.random.default_rng()


def anneal_step(raceline: Raceline, car: Car, batch_size: int, rng: np.random.Generator | None = None) -> None:
    """
    Simulates a batch of random candidate lines in parallel on threads and keeps
    the best one if it improves on the current best line. Decays the raceline's
    heatmap and progress rate otherwise.

    Operates on the raceline's package-internal state (_position/_heatmap/_geometry).
    """
    rng = random_generator if rng is None else rng
    idx_len = len(raceline.width)

    # Choose random locations and lengths for the deviations
    cdf = raceline._heatmap.cumsum()
    locations = np.minimum(
        cdf.searchsorted(rng.random(batch_size) * cdf[-1], side="right"),
        idx_len - 1,
    )
    lengths = rng.integers(3, MAX_DEVIATION_LENGTH, size=batch_size, endpoint=True)
    deviations = rng.uniform(-1, 1, batch_size) * np.maximum(lengths / MAX_DEVIATION_LENGTH, MAX_DEVIATION)

    laptimes, positions, perturbations = run_batch(
        raceline._geometry, raceline._position, locations, lengths, deviations, car
    )

    try:
        winner = int(np.nanargmin(laptimes))
    except ValueError:
        winner = -1

    if winner >= 0 and np.isfinite(improvement := raceline.best_time - laptimes[winner]) and improvement > 0:
        raceline._position = positions[winner].copy()
        raceline.best_time = laptimes[winner]

        raceline._heatmap += perturbations[winner] * improvement * 1e3
        raceline.progress_rate += improvement
        return

    # If the new line is not better, slowly decay the heatmap and progress rate
    raceline._heatmap += 0.00015
    raceline._heatmap /= 1.00015
    raceline.progress_rate *= F_ANNEAL


def _status(raceline: Raceline, candidates: int, pending: int, baseline: float, finished: bool) -> dict:
    return {
        "candidates": candidates,
        "pending": pending,
        "best_time": raceline.best_time,
        "best_time_str": raceline.best_time_str(),
        "improvement": raceline.best_time - baseline,
        "progress_rate": raceline.progress_rate,
        "finished": finished,
    }


def optimize(
    raceline: Raceline,
    car: Car,
    *,
    batch_size: int = 100,
    tolerance: float = 0.1,
    baseline: float = 0.0,
    update_every: float = 1.0,
    save_every: float = 10.0,
    on_save: Callable[[], None] | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[dict]:
    """
    Anneals candidate lines until the raceline converges (progress_rate < tolerance).

    The raceline must have been simulated for this car first (`raceline.simulate(car)`),
    so best_time is finite. Yields a status dict every update_every seconds and calls
    on_save every save_every seconds — always before the corresponding yield, so saving
    still happens when the consumer breaks out of the loop.
    """
    if not np.isfinite(raceline.best_time):
        raise ValueError("raceline.best_time is not finite — call raceline.simulate(car) before optimize()")

    start = time.monotonic()
    last_update = start
    last_save = start
    pending = 0
    candidates = 0
    while True:
        anneal_step(raceline, car, batch_size, rng)
        pending += batch_size
        candidates += batch_size
        now = time.monotonic()
        finished = raceline.progress_rate < tolerance
        if on_save is not None and now - last_save >= save_every:
            on_save()
            last_save = now
        if finished or now - last_update >= update_every:
            yield _status(raceline, candidates, pending, baseline, finished=finished)
            pending = 0
            last_update = now
        if finished:
            return
