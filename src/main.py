import itertools
import logging
from pathlib import Path

from tqdm import tqdm

import laptime_sim
from laptime_sim import car_list, track_list

logging.basicConfig(level=logging.INFO)

PATH_TRACKS = Path("./resources/tracks/")
PATH_CARS = Path("./resources/cars/")
PATH_RESULTS = Path("./resources/simulated/")
TOLERANCE = 0.0005


def main() -> None:
    PATH_RESULTS.mkdir(exist_ok=True)

    logging.info("started optimization")
    for race_car, track in itertools.product(car_list(PATH_CARS), track_list(PATH_TRACKS)):
        file_path = PATH_RESULTS / Path(f"{race_car.file_name}_{track.name}_simulated.parquet")

        try:
            raceline = laptime_sim.Raceline.from_file(track, file_path)
            logging.info(f"Loading {raceline.filename(race_car.file_name)} from {PATH_RESULTS.absolute()}")
            logging.warning("Filename exists and will be overwritten")
            logging.info(f"Track: {track.name} has {track.len} datapoints.")
        except FileNotFoundError:
            logging.info(f"File not found. Creating new file: {file_path.absolute()}")
            raceline = laptime_sim.Raceline(track=track)
            raceline.save_line(file_path, race_car.name)

        raceline.update(race_car)
        loaded_best_time = raceline.best_time

        try:
            with tqdm(leave=True, desc=f"{raceline.track.name}-{race_car.name}") as bar:
                update_interval = 500
                for i in itertools.count():
                    raceline.simulate_new_line(race_car)
                    if i % update_interval == 0:
                        bar.set_postfix(
                            laptime=f"{raceline.best_time_str()}",
                            improvement=f"{raceline.best_time - loaded_best_time:.4f}",
                            progress=f"{1 / (1 + (raceline.progress_rate - TOLERANCE) * 100):.3%}",
                        )
                        bar.update(update_interval)

                    if i % 10000 == 0:
                        raceline.save_line(file_path, race_car.name)

                    if raceline.progress_rate < TOLERANCE:
                        bar.set_postfix(
                            laptime=f"{raceline.best_time_str()}",
                            improvement=f"{raceline.best_time - loaded_best_time:.4f}",
                            status="finished",
                        )
                        bar.update(update_interval)
                        break
        except KeyboardInterrupt:
            logging.warning("Interrupted by CTRL+C")
            exit()
        finally:
            raceline.save_line(file_path, race_car.name)
            logging.info(f"final results saved to {file_path.absolute()}")

        logging.info(f"Optimization finished. {race_car.name}:{raceline.best_time_str()}")


if __name__ == "__main__":
    main()
