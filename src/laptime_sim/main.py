import itertools
import logging
from pathlib import Path

from tqdm import tqdm

import laptime_sim
from laptime_sim import car_list, track_list


logging.basicConfig(level=logging.INFO)

PATH_TRACKS = Path("./tracks/")
PATH_CARS = Path("./cars/")
PATH_RESULTS = Path("./simulated/")
TOLERANCE = 0.005


def main() -> None:
    PATH_RESULTS.mkdir(exist_ok=True)

    logging.info("started optimzation")
    for car, track in itertools.product(car_list(PATH_CARS), track_list(PATH_TRACKS)):
        raceline = laptime_sim.Raceline(track=track, car=car)
        file_path = PATH_RESULTS / raceline.filename
        try:
            raceline.load_line(file_path)
            logging.info(f"Loading {raceline.filename} from {PATH_RESULTS.absolute()}")
            logging.warning("Filename exists and will be overwritten")
            logging.info(
                f"Track: {track.name} has {track.len} datapoints. Current best time for track = {raceline.best_time_str}"
            )
        except FileNotFoundError:
            logging.info(f"File not found. Creating new file: {file_path.absolute()}")
            raceline.update()
            raceline.save_line(file_path)

        loaded_best_time = raceline.best_time

        try:
            with tqdm(leave=True, desc=f"{raceline.track.name}-{raceline.car.name}", mininterval=0.5) as bar:
                while raceline.progress_rate > TOLERANCE:
                    raceline.simulate_new_line()
                    bar.set_postfix(
                        laptime=raceline.best_time_str,
                        progress=f"{raceline.best_time - loaded_best_time:.4f}",
                        progress_rate=f"{raceline.progress_rate:.3f}",
                    )
                    bar.update()

        except KeyboardInterrupt:
            logging.warning("Interrupted by CTRL+C")
            exit()
        finally:
            raceline.save_line(file_path)
            logging.info(f"final results saved to {file_path.absolute()}")

        logging.info(f"Optimization finished. {car.name}:{raceline.best_time_str}")


if __name__ == "__main__":
    main()
