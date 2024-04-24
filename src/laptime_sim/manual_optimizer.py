import geopandas

from laptime_sim.race_lap import optimize_laptime
import laptime_sim
from laptime_sim import Raceline

TOLERANCE = 0.001


def load_racecar(name):
    return laptime_sim.Car.from_toml(f"./cars/{name}.toml")


def optimize(raceline: Raceline):

    def print_results(time, iteration) -> None:
        print(f"Laptime = {laptime_sim.time_to_str(time)}  (iteration:{iteration}) (progress:{raceline.progress:.4f})")

    def save_results(raceline_dataframe: geopandas.GeoDataFrame) -> None:
        raceline_dataframe.to_parquet(raceline.filename_results)
        print(f"intermediate results saved to {raceline.filename_results}")

    try:
        raceline = optimize_laptime(raceline, print_results, save_results, tolerance=TOLERANCE)
        print(f"optimization finished. {raceline.progress=}")

    except KeyboardInterrupt:
        print("Interrupted by CTRL+C, saving progress")
        print(f"final results saved to {raceline.filename_results=}")
        save_results(raceline.get_dataframe())
        print(f"{raceline.car.name} - Saved laptime = {laptime_sim.time_to_str(raceline.best_time)}")


def main() -> None:

    for car in laptime_sim.get_all_cars():
        for track in laptime_sim.get_all_tracks():

            print(f"Loaded track data for {track.name}")
            print(f"Track has {track.len} datapoints")

            optimize(Raceline(track=track, car=car).load_results())


if __name__ == "__main__":
    main()
