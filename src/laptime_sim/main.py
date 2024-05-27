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
PATH_RESULTS = Path("./simulated2/")
TOLERANCE = 0.5


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
            filename_results = Path(PATH_RESULTS, f"{car.file_name}_{track.name}_simulated.parquet")

            if filename_results.exists():
                raceline.load_line(filename_results)
                logging.warning(f"Filename {filename_results} exists and will be overwritten")
                logging.info(
                    f"Track: {track.name} has {track.len} datapoints. Current best time for track = {raceline.best_time_str} "
                )
            loaded_best_time = raceline.best_time

            bar = tqdm(leave=True, desc=f"{raceline.track.name}-{raceline.car.name}", mininterval=100)

            try:
                while raceline.progress > TOLERANCE:
                    raceline.simulate_new_line()
                    bar.set_postfix(laptime=raceline.best_time_str, progress=f"{raceline.progress:.3f}")
                    bar.update()
                bar.close()

            except KeyboardInterrupt:
                # logging.warning("Interrupted by CTRL+C")
                exit()
            finally:
                bar.close()

                raceline.save_line(filename_results)
                logging.info(f"final results saved to {filename_results.absolute()}")

            logging.info(
                f"Optimization finished. {car.name}:{raceline.best_time_str} "
                f"improvement {loaded_best_time - raceline.best_time}"
            )


if __name__ == "__main__":
    main()
