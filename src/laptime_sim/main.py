import functools
import logging
import os
from typing import Union

from laptime_sim.car import Car
from laptime_sim.timer import Timer
from laptime_sim.track import Track
from laptime_sim.raceline import Raceline

from pathlib import Path

import itertools


logging.basicConfig(level=logging.INFO)


PATH_TRACKS = "./tracks/"
PATH_CARS = "./cars/"
PATH_RESULTS = "./simulated/"
TOLERANCE = 0.001


def print_results(raceline: Raceline, **kwargs) -> None:
    print(
        f"{raceline.track.name}-{raceline.car.name}:",
        f"laptime={raceline.best_time_str}",
        f"{kwargs}",
        # f"({raceline.progress:.3f})",
        # f"({speed=:.3f})",
        end="                                                       \r",
    )


@functools.lru_cache()
def get_all_cars(path: str) -> list[Car]:
    """
    Returns a list of all cars in the given directory, sorted by filename.

    :param path: The path to the directory containing the car definition files
    :return: A list of all cars in the given directory
    """
    car_files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith("toml")]
    return [Car.from_toml(f) for f in car_files]


@functools.lru_cache()
def get_all_tracks(path: Union[str, os.PathLike]) -> list[Track]:
    """
    Returns a list of all tracks in the given directory

    :param path: the path to the directory containing the tracks
    :return: a list of all tracks in the given directory
    """
    track_files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith("parquet")]
    return [Track.from_parquet(f) for f in track_files]


def optimize_raceline(raceline: Raceline, filename_save=None) -> Raceline:

    start_timer = Timer()
    display_timer = Timer(1)
    save_timer = Timer(10)

    raceline.save_line(filename_save)

    print_results(raceline, filename_save=filename_save),

    for iterations in itertools.count():

        raceline.simulate_new_line()

        if raceline.progress < TOLERANCE:
            break

        if display_timer.triggered:
            print_results(raceline, iterations=iterations, iteration_speed=iterations / start_timer.elapsed_time)
            display_timer.reset()

        if save_timer.triggered:
            raceline.save_line(filename_save)
            print_results(raceline, filename_save=filename_save)
            save_timer.reset()

    raceline.save_line(filename_save)
    print_results(raceline, iterations=iterations)
    return raceline


def main() -> None:

    cars = get_all_cars(PATH_CARS)
    tracks = get_all_tracks(PATH_TRACKS)

    logging.info("started optimzation")
    for car in cars:
        for track in tracks:

            raceline = Raceline(track=track, car=car)
            logging.info(f"Loaded track data for {track.name} has {track.len} datapoints.")
            filename_results = Path(PATH_RESULTS, f"{car.file_name}_{track.name}_simulated.parquet")

            if filename_results.exists():
                logging.warning(f"Filename {filename_results} exists and will be overwritten")
                raceline.load_line(filename_results)
                loaded_best_time = raceline.best_time
                logging.info(
                    f"Track: {track.name} has {track.len} datapoints. Current best time for track = {raceline.best_time_str} "
                )

            try:
                raceline = optimize_raceline(raceline, filename_results)
            except KeyboardInterrupt:
                print("")
                logging.warning("Interrupted by CTRL+C")
                exit()
            finally:
                print("")
                raceline.save_line(filename_results)
                logging.info(f"final results saved to {filename_results=}")
                print(f"{filename_results} - Saved laptime = {raceline.best_time_str}")

            logging.info(f"Optimization finished. {car.name}:{raceline.best_time_str}")
            print(f"Optimization improvement {loaded_best_time - raceline.best_time}")


if __name__ == "__main__":
    main()
