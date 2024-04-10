import os
import geopandas

from laptime_sim import file_operations
from laptime_sim.race_lap import optimize_laptime
import laptime_sim

CARS = ["Peugeot_205RFS", "BMW_Z3M"]
PATH_RESULTS = "./simulated/"
PATH_TRACKS = "./tracks/"
TOLERANCE = 0.001


def load_racecar(name):
    return laptime_sim.Car.from_toml(f"./cars/{name}.toml")


def optimize(track: laptime_sim.Track, name_car: str):

    race_car = load_racecar(name_car)
    raceline = laptime_sim.Raceline(track=track, car=race_car)

    filename_raceline = file_operations.find_raceline_filename(track.name, name_car)
    if filename_raceline is not None:
        raceline_gdf = geopandas.read_parquet(filename_raceline)
        raceline = raceline.parametrize_raceline(raceline_gdf)

    filename_output = os.path.join(
        PATH_RESULTS, f"{name_car}_{track.name}_simulated.parquet"
    )

    print(f"Loaded track data for {track.name}")
    print(f"Track has {track.len} datapoints")

    def print_results(time, iteration) -> None:
        print(
            f"Laptime = {laptime_sim.time_to_str(time)}  (iteration:{iteration}) (progress:{raceline.progress:.4f})"
        )

    def save_results(raceline_dataframe: geopandas.GeoDataFrame) -> None:
        raceline_dataframe.to_parquet(filename_output)
        print(f"intermediate results saved to {filename_output=}")

    try:
        raceline = optimize_laptime(raceline, print_results, save_results, tolerance=TOLERANCE)
        print(f"optimization finished. {raceline.progress=}")

    except KeyboardInterrupt:
        print("Interrupted by CTRL+C, saving progress")
        print(f"final results saved to {filename_output=}")
        save_results(raceline.get_dataframe())
        print(f"{race_car.name} - Saved laptime = {laptime_sim.time_to_str(raceline.best_time)}")


def main2() -> None:
    filename = file_operations.find_track_filename(
        track_name="20191030_Circuit_Zandvoort", path=PATH_TRACKS
    )
    track_layout = file_operations.load_trackdata_from_file(filename)
    optimize(track_layout, name_car="BMW_Z3M")


def main() -> None:
    for filename_racetrack in file_operations.filename_iterator(
        PATH_TRACKS, ("parquet")
    ):
        track = laptime_sim.Track.from_parquet(filename_racetrack)
        for car in CARS:
            optimize(track, car)


if __name__ == "__main__":
    main()
