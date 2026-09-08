import itertools
import logging
import time

from tqdm import tqdm

from laptime_sim.registry import cars, load_result, results_path, save_result, tracks

logging.basicConfig(level=logging.INFO)

TOLERANCE = 0.1
BATCH_SIZE = 100


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
            logging.info(f"Track: {track.name} has {track.len} datapoints.")
        else:
            logging.info(f"File not found. Creating new file: {file_path.absolute()}")
            save_result(raceline, race_car)

        raceline.simulate(race_car)
        loaded_best_time = raceline.best_time

        try:
            with tqdm(leave=True, desc=f"{raceline.track.name}-{race_car.name}") as bar:
                update_seconds = 1.0
                last_update = time.monotonic()
                save_seconds = 10.0
                last_save = time.monotonic()
                pending = 0
                bar.set_postfix(
                    laptime=f"{raceline.best_time_str()}",
                    improvement=f"{raceline.best_time - loaded_best_time:.4f}",
                )
                for _ in itertools.count():
                    raceline.try_random_lines(race_car, BATCH_SIZE)
                    pending += 1

                    if time.monotonic() - last_update >= update_seconds:
                        bar.set_postfix(
                            laptime=f"{raceline.best_time_str()}",
                            improvement=f"{raceline.best_time - loaded_best_time:.4f}",
                            progress=f"{1 / (1 + (raceline.progress_rate - TOLERANCE) * 100):.3%}",
                        )
                        bar.update(pending * BATCH_SIZE)
                        pending = 0
                        last_update = time.monotonic()

                    if time.monotonic() - last_save >= save_seconds:
                        save_result(raceline, race_car)
                        last_save = time.monotonic()

                    if raceline.progress_rate < TOLERANCE:
                        bar.set_postfix(
                            laptime=f"{raceline.best_time_str()}",
                            improvement=f"{raceline.best_time - loaded_best_time:.4f}",
                            status="finished",
                        )
                        bar.update(pending * BATCH_SIZE)
                        break
        except KeyboardInterrupt:
            logging.warning("Interrupted by CTRL+C")
            exit()
        except SystemError as exc:
            logging.warning(f"Interrupted during numba cache load: {exc}")
            exit()
        finally:
            save_result(raceline, race_car)
            logging.info(f"final results saved to {file_path.absolute()}")

        logging.info(f"Optimization finished. {race_car.name}:{raceline.best_time_str()}")


if __name__ == "__main__":
    main()
