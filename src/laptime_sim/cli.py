import functools
import itertools
import logging
import sys

from tqdm import tqdm

from laptime_sim.optimizer import optimize
from laptime_sim.registry import cars, load_result, results_path, save_result, tracks

logging.basicConfig(level=logging.INFO)

TOLERANCE = 0.1


def main() -> None:
    CAR_LIST = cars()
    TRACK_LIST = tracks()

    logging.info("started optimization")
    for race_car, track in itertools.product(CAR_LIST, TRACK_LIST):
        file_path = results_path(race_car, track)

        raceline, found = load_result(track, race_car)

        if found:
            logging.info(f"Loading {file_path.name} from {file_path.parent}")
            logging.warning("Filename exists and will be overwritten")
        else:
            logging.info(f"File not found. Creating new file: {file_path.absolute()}")
            save_result(raceline, race_car)

        raceline.simulate(race_car)
        loaded_best_time = raceline.best_time

        try:
            with tqdm(leave=True, desc=f"{raceline.track.name}-{race_car.name}") as bar:
                for status in optimize(
                    raceline,
                    race_car,
                    tolerance=TOLERANCE,
                    baseline=loaded_best_time,
                    on_save=functools.partial(save_result, raceline, race_car),
                ):
                    bar.update(status["pending"])
                    postfix = {
                        "laptime": status["best_time_str"],
                        "improvement": f"{status['improvement']:.4f}",
                    }
                    if status["finished"]:
                        postfix["status"] = "finished"
                    else:
                        postfix["progress"] = f"{1 / (1 + (status['progress_rate'] - TOLERANCE) * 100):.3%}"
                    bar.set_postfix(postfix)
                    if status["finished"]:
                        break
        except KeyboardInterrupt:
            logging.warning("Interrupted by CTRL+C")
            sys.exit(130)
        except SystemError as exc:
            logging.warning(f"Interrupted during numba cache load: {exc}")
            sys.exit(1)
        finally:
            save_result(raceline, race_car)
            logging.info(f"final results saved to {file_path.absolute()}")

        logging.info(f"Optimization finished. {race_car.name}:{raceline.best_time_str()}")


if __name__ == "__main__":
    main()
