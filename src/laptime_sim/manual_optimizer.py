import os
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries

import file_operations
from car import Car
from tracksession import TrackSession

import race_lap
from race_lap import time_to_str

name_car = 'Peugeot_205RFS'
# name_car = 'BMW_Z3M'
# NAME_TRACK = "2020_zandvoort"
# name_track = "20191030_Circuit_Zandvoort"
# NAME_TRACK = "202209022_Circuit_Meppen"
# NAME_TRACK = "20191211_Bilsterberg"
# NAME_TRACK = "20191128_Circuit_Assen"
NAME_TRACK = "20191220_Spa_Francorchamp"
PATH_RESULTS = "./simulated/"


def load_track(name_track):
    filename_track = file_operations.find_track_filename(name_track)
    if filename_track is None:
        return None
    return gpd.read_parquet(filename_track)


def load_raceline(name_track, name_car):

    filename_raceline = file_operations.find_raceline_filename(name_track, name_car)
    if filename_raceline is None:
        track_layout = load_track(name_track)
        track_session = TrackSession.from_layout(track_layout)
        track_session.update_line()

        gdf_raceline = GeoDataFrame(geometry=track_session.track_raceline)
        gdf_raceline['car'] = name_car
        gdf_raceline['track'] = name_track
        return gdf_raceline
    return gpd.read_parquet(filename_raceline)


def load_racecar(name):
    return Car.from_toml(f"./cars/{name}.toml")


def main():

    track_layout = load_track(NAME_TRACK)
    if track_layout is None:
        FileNotFoundError('No track files found')

    gdf_raceline = load_raceline(name_track=NAME_TRACK, name_car=name_car)
    print(gdf_raceline)
    race_car = load_racecar(name_car)
    track_session = TrackSession.from_layout(track_layout, gdf_raceline)

    filename_output = os.path.join(PATH_RESULTS, f"{name_car}_{NAME_TRACK}_simulated.parquet")

    print(f'Loaded track data for {NAME_TRACK}')
    print(f'Track has {track_session.len} datapoints')

    def print_results(time, iteration) -> None:
        print(f"Laptime = {time_to_str(time)}  (iteration:{iteration}) (progress:{track_session.progress:.4f})")

    def save_results(track_raceline: GeoSeries) -> None:
        gdf_raceline.set_geometry(track_raceline, inplace=True)
        gdf_raceline.to_parquet(filename_output)
        print(f'intermediate results saved to {filename_output=}')

    try:
        track_session = race_lap.optimize_laptime(track_session, race_car, print_results, save_results)
        print(f'optimization finished. {track_session.progress=}')

    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')
        print(f'final results saved to {filename_output=}')
        save_results(track_session.track_raceline)
        best_time = race_lap.simulate(race_car, track_session.line_coords(), track_session.slope)

        print(f'{race_car.name} - Simulated laptime = {time_to_str(best_time)}')


if __name__ == '__main__':
    main()
