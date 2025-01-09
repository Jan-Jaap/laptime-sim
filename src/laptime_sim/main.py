import functools
import logging
from pathlib import Path

from tqdm import tqdm

from laptime_sim.car import Car
from laptime_sim.raceline import Raceline
from laptime_sim.track import Track

logging.basicConfig(level=logging.INFO)

PATH_TRACKS = Path("./tracks/")
PATH_CARS = Path("./cars/")
PATH_RESULTS = Path("./simulated/")
TOLERANCE = 0.005


@functools.lru_cache()
def get_all_cars(cars_path: Path) -> list[Car]:
    """Returns a list of all cars in the given directory, sorted by filename

    :param cars_path: The path to the directory containing the car definition files
    :return: A list of all cars in the given directory
    """
    return [Car.from_toml(file) for file in cars_path.glob("*.toml")]


@functools.lru_cache()
def get_all_tracks(tracks_path: Path) -> list[Track]:
    """Returns a list of all tracks in the given directory

    :param tracks_path: the path to the directory containing the tracks
    :return: a list of all tracks in the given directory
    """
    return [Track.from_parquet(file) for file in tracks_path.glob("*.parquet")]


def main() -> None:
    PATH_RESULTS.mkdir(exist_ok=True)

    cars = get_all_cars(PATH_CARS)
    tracks = get_all_tracks(PATH_TRACKS)

    logging.info("started optimzation")
    for car in cars:
        for track in tracks:
            raceline = Raceline(track=track, car=car)
            logging.info(f"Loaded track data for {track.name} has {track.len} datapoints.")
            filename_results = raceline.filename(PATH_RESULTS)

            if filename_results.exists():
                raceline.load_line(filename_results)
                logging.warning(f"Filename {filename_results} exists and will be overwritten")
                logging.info(
                    f"Track: {track.name} has {track.len} datapoints. Current best time for track = {raceline.best_time_str}"
                )
            loaded_best_time = raceline.best_time

            try:
                with tqdm(leave=True, desc=f"{raceline.track.name}-{raceline.car.name}", mininterval=100) as bar:
                    while raceline.progress > TOLERANCE:
                        raceline.simulate_new_line()
                        bar.set_postfix(laptime=raceline.best_time_str, progress_speed=f"{raceline.progress:.3f}")
                        bar.update()

            except KeyboardInterrupt:
                logging.warning("Interrupted by CTRL+C")
                exit()
            finally:
                raceline.save_line(filename_results)
                logging.info(
                    f"final results saved to {filename_results.absolute()} \n"
                    f"improvement {loaded_best_time - raceline.best_time}"
                )

            logging.info(
                f"Optimization finished. {car.name}:{raceline.best_time_str} \n"
                f"improvement {loaded_best_time - raceline.best_time}"
            )


if __name__ == "__main__":
    main()
