import os
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries

import file_operations
from car import Car
from tracksession import TrackSession

import race_lap
from race_lap import time_to_str

# NAME_CAR = 'Peugeot_205RFS'
NAME_CAR = 'BMW_Z3M'


# NAME_TRACK = "2020_zandvoort"
# NAME_TRACK = "20191030_Circuit_Zandvoort"
# NAME_TRACK = "202209022_Circuit_Meppen"
NAME_TRACK = "20191211_Bilsterberg"
# NAME_TRACK = "20191128_Circuit_Assen"
# NAME_TRACK = "20191220_Spa_Francorchamp"
PATH_RESULTS = "./simulated/"


def load_track(name_track):
    filename_track = file_operations.find_track_filename(name_track)
    if filename_track is None:
        return None
    return gpd.read_parquet(filename_track)


def load_raceline(name_track, name_car):
    filename_raceline = file_operations.find_raceline_filename(name_track, name_car)
    if filename_raceline is None:
        return init_raceline(name_track, name_car)
    return gpd.read_parquet(filename_raceline)


def init_raceline(name_track, name_car):
    track_layout = load_track(name_track)
    track_session = TrackSession.from_layout(track_layout)
    track_session.update_line()

    gdf_raceline = GeoDataFrame(geometry=track_session.track_raceline, crs=track_session.track_raceline.crs)
    gdf_raceline['car'] = name_car
    gdf_raceline['track'] = name_track
    return gdf_raceline


def load_racecar(name):
    return Car.from_toml(f"./cars/{name}.toml")


def main(name_track, name_car):

    name_track = NAME_TRACK
    name_car = NAME_CAR
    track_layout = load_track(name_track)
    if track_layout is None:
        FileNotFoundError('No track files found')

    gdf_raceline = load_raceline(name_track=name_track, name_car=name_car)
    race_car = load_racecar(name_car)
    track_session = TrackSession.from_layout(track_layout, gdf_raceline)

    filename_output = os.path.join(PATH_RESULTS, f"{name_car}_{name_track}_simulated.parquet")

    print(f'Loaded track data for {name_track}')
    print(f'Track has {track_session.len} datapoints')

    def print_results(time, iteration) -> None:
        print(f"Laptime = {time_to_str(time)}  (iteration:{iteration}) (progress:{track_session.progress:.4f})")

    def save_results(track_raceline: GeoSeries) -> None:
        gdf_raceline.set_geometry(track_raceline, inplace=True)
        gdf_raceline['crs_backup'] = gdf_raceline.crs.to_epsg()
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
    # main(NAME_TRACK, NAME_CAR)
    cars = ['Peugeot_205RFS', 'BMW_Z3M']
    tracks = [
        "2020_zandvoort",
        "20191030_Circuit_Zandvoort",
        "202209022_Circuit_Meppen",
        "20191211_Bilsterberg",
        "20191128_Circuit_Assen",
        "20191220_Spa_Francorchamp",
    ]

    for car in cars:
        for racetrack in tracks:
            main(racetrack, car)
