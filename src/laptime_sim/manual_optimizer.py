import laptime_sim
from laptime_sim.raceline import Raceline, optimize_raceline
from laptime_sim.simulate import SimResults

PATH_TRACKS = "./tracks/"
PATH_CARS = "./cars/"

TOLERANCE = 0.001


def optimize(raceline: Raceline):

    def print_results(time, iteration, saved) -> None:
        print(f"Laptime = {laptime_sim.time_to_str(time)}  (iteration:{iteration}) (progress:{raceline.progress:.4f})")

    def save_results(sim_results: SimResults) -> None:
        # raceline_dataframe.to_parquet(raceline.filename_results)
        print(f"intermediate results saved to {raceline.filename_results}")

    try:
        raceline = optimize_raceline(raceline, print_results, tolerance=TOLERANCE)
        print(f"optimization finished. {raceline.progress=}")

    except KeyboardInterrupt:
        print("Interrupted by CTRL+C, saving progress")
        print(f"final results saved to {raceline.filename_results=}")
        save_results(raceline.get_dataframe())
        print(f"{raceline.car.name} - Saved laptime = {laptime_sim.time_to_str(raceline.best_time)}")


def main() -> None:

    for track in laptime_sim.get_all_tracks(PATH_TRACKS):
        for car in laptime_sim.get_all_cars(PATH_CARS):

            print(f"Loaded track data for {track.name}")
            print(f"Track has {track.len} datapoints")

            optimize(Raceline(track=track, car=car).load_raceline())


if __name__ == "__main__":
    main()
