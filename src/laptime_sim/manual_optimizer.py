import file_operations
from car import Car
import geodataframe_operations
from tracksession import TrackSession
import race_lap

NAME_CAR = 'Peugeot_205RFS'
# NAME_TRACK = "2020_zandvoort"
# NAME_TRACK = "20191030_Circuit_Zandvoort"
# NAME_TRACK = "202209022_Circuit_Meppen"
NAME_TRACK = "20191211_Bilsterberg"
# NAME_TRACK = "20191128_Circuit_Assen"
# NAME_TRACK = "20191220_Spa_Francorchamp"


def time_to_str(seconds: float) -> str:
    return "{:02.0f}:{:06.03f}".format(seconds % 3600//60, seconds % 60)


def print_results(time, iteration, _) -> None:
    print(f"Laptime = {time_to_str(time)}  (iteration:{iteration})")


def save_results(track_session, filename_results) -> None:
    results = race_lap.sim(track_session, verbose=True)
    file_operations.save_results(results, filename_results)
    print(f'intermediate results saved to {filename_results=}')


def main():

    # filename_track = file_operations.find_filename(NAME_TRACK, NAME_CAR)
    filename_track = './simulated/20191211_Bilsterberg.parquet'
    race_car = Car.from_toml(f"./cars/{NAME_CAR}.toml")
    filename_results = f"{NAME_CAR}_{NAME_TRACK}_simulated.csv"

    track_layout, best_line = file_operations.load_trackdata_from_file(filename_track)
    best_line = geodataframe_operations.parametrize_race_line(track_layout)
    track_session = TrackSession(track_layout=track_layout, car=race_car, line_pos=best_line)

    if track_session is None:
        return

    best_time = race_lap.sim(track_session=track_session)

    print(f'Loaded track data from {filename_track}')
    print(f'Track has {track_session.len} datapoints')
    print(f'{race_car.name} - Simulated laptime = {time_to_str(best_time)}')

    def save_results(track_session) -> None:
        results = race_lap.sim(track_session, verbose=True)
        file_operations.save_results(results, filename_results)
        print(f'intermediate results saved to {filename_results=}')

    try:

        track_session = race_lap.optimize_laptime(
            track_session=track_session,
            display_intermediate_results=print_results,
            save_intermediate_results=save_results
            )

    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')
        results_dataframe = race_lap.sim(track_session=track_session, verbose=True)
        file_operations.save_results(results_dataframe, filename_results)

        best_time = race_lap.sim(track_session=track_session)

        print(f'final results saved to {filename_results=}')
        print(f'{race_car.name} - Simulated laptime = {time_to_str(best_time)}')


if __name__ == '__main__':
    main()
