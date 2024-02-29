import os
import geopandas as gpd
from geopandas import GeoDataFrame

from laptime_sim import file_operations
from laptime_sim.car import Car
from laptime_sim.track import TrackInterface

from laptime_sim import race_lap
from laptime_sim.race_lap import time_to_str

# from icecream import ic

CARS = ['Peugeot_205RFS', 'BMW_Z3M']
TRACKS = [
    # "2020_zandvoort",
    "20191030_Circuit_Zandvoort",
    "202209022_Circuit_Meppen",
    "20191211_Bilsterberg",
    "20191128_Circuit_Assen",
    "20191220_Spa_Francorchamp",
    ]
PATH_RESULTS = "./simulated/"
TOLERANCE = 0.01


def load_track(name_track):
    filename_track = file_operations.find_track_filename(name_track)
    if filename_track is None:
        return None
    return gpd.read_parquet(filename_track)


def load_raceline(name_track, name_car) -> GeoDataFrame:
    filename_raceline = file_operations.find_raceline_filename(name_track, name_car)
    if filename_raceline is None:
        return None
    return gpd.read_parquet(filename_raceline)


def load_racecar(name):
    return Car.from_toml(f"./cars/{name}.toml")


def main(track_layout: GeoDataFrame, name_car: str):

    name_track = track_layout.name[0]
    gdf_raceline = load_raceline(name_track=name_track, name_car=name_car)
    race_car = load_racecar(name_car)
    track_session = TrackInterface.from_layout(track_layout, gdf_raceline)

    filename_output = os.path.join(PATH_RESULTS, f"{name_car}_{name_track}_simulated.parquet")

    print(f'Loaded track data for {name_track}')
    print(f'Track has {track_session.len} datapoints')

    def print_results(time, iteration) -> None:
        print(f"Laptime = {time_to_str(time)}  (iteration:{iteration}) (progress:{track_session.progress:.4f})")

    def save_results(track_raceline: GeoDataFrame) -> None:
        # track_raceline.set_geometry(track_raceline, inplace=True)
        track_raceline['crs_backup'] = track_raceline.crs.to_epsg()
        track_raceline['track'] = name_track
        track_raceline['car'] = name_car
        track_raceline.to_parquet(filename_output)
        print(f'intermediate results saved to {filename_output=}')

    try:
        track_session = race_lap.optimize_laptime(
            track_session,
            race_car,
            print_results,
            save_results,
            tolerance=TOLERANCE
            )
        print(f'optimization finished. {track_session.progress=}')

    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')
        print(f'final results saved to {filename_output=}')
        save_results(track_session.get_raceline())
        best_time = race_lap.simulate(race_car, track_session.line_coords(), track_session.slope)
        # results = race_lap.(track_session, verbose=True).to_csv().encode('utf-8')
        best_time = race_lap.simulate(race_car, track_session.line_coords(), track_session.slope)

        print(f'{race_car.name} - Simulated laptime = {time_to_str(best_time)}')


if __name__ == '__main__':
    track_layout = load_track('20191030_Circuit_Zandvoort')
    main(track_layout, 'BMW_Z3M')

    # for racetrack in TRACKS:
    #     for car in CARS:
    #         # ic(racetrack, car)
    #         track_layout = load_track(racetrack)
    #         main(track_layout, car)
