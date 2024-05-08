import logging
import laptime_sim
from laptime_sim.raceline import Raceline, optimize_raceline
from laptime_sim.simulate import RacelineSimulator

from pathlib import Path

logging.basicConfig(level=logging.INFO)


PATH_TRACKS = "./tracks/"
PATH_CARS = "./cars/"
PATH_RESULTS = "./simulated/"
TOLERANCE = 0.01


def print_results(raceline: Raceline, nr_iterations, saved) -> None:
    print(
        f"{raceline.track.name}-{raceline.car.name}:",
        f"laptime={raceline.best_time_str}",
        f"({nr_iterations=})",
        f"({raceline.progress=:.3f})",
        end="                 \r",
    )


def main() -> None:

    logging.info("started optimzation")
    for car in laptime_sim.get_all_cars(PATH_CARS):
        simulator = RacelineSimulator(car=car)
        for track in laptime_sim.get_all_tracks(PATH_TRACKS):

            raceline = Raceline(track=track, car=car, simulator=simulator)
            logging.info(f"Loaded track data for {track.name}")
            logging.info(f"Track has {track.len} datapoints.")
            filename_results = Path(PATH_RESULTS, f"{car.file_name}_{track.name}_simulated.parquet")

            if filename_results.exists():
                logging.warning(f"Filename {filename_results} exists and will be overwritten")
                raceline.load_results(filename_results)
                loaded_best_time = raceline.best_time
                logging.info(
                    f"Track: {track.name} has {track.len} datapoints. Current best time for track = {raceline.best_time_str} "
                )

            try:
                raceline = optimize_raceline(raceline, print_results, filename_results, TOLERANCE)
            except KeyboardInterrupt:
                print("")
                logging.warning("Interrupted by CTRL+C")
                exit()
            finally:
                print("")
                raceline.save_results(filename_results)
                logging.info(f"final results saved to {filename_results=}")
                print(f"{filename_results} - Saved laptime = {raceline.best_time_str}")

            logging.info(f"Optimization finished. {car.name}:{raceline.best_time_str}")
            print(f"Optimization improvement {loaded_best_time - raceline.best_time}")


if __name__ == "__main__":
    main()
