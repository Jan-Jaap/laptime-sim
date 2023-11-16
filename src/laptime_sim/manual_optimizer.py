import file_operations
from car import Car
from tracksession import TrackSession
import race_lap
from race_lap import time_to_str

NAME_CAR = 'Peugeot_205RFS'
# NAME_TRACK = "2020_zandvoort"
# NAME_TRACK = "20191030_Circuit_Zandvoort"
NAME_TRACK = "202209022_Circuit_Meppen"
# NAME_TRACK = "20191211_Bilsterberg"
# NAME_TRACK = "20191128_Circuit_Assen"
# NAME_TRACK = "20191220_Spa_Francorchamp"


def main():

    filename_track = file_operations.find_filename(NAME_TRACK, NAME_CAR)
    if filename_track is None:
        filename_track = file_operations.find_filename(NAME_TRACK, None)

    race_car = Car.from_toml(f"./cars/{NAME_CAR}.toml")
    filename_results = f"{NAME_CAR}_{NAME_TRACK}_simulated"

    track_layout = file_operations.load_trackdata_from_file(filename_track)
    track_session = TrackSession(track_layout=track_layout, car=race_car)

    if track_session is None:
        return

    print(f'Loaded track data from {filename_track}')
    print(f'Track has {track_session.len} datapoints')

    def print_results(time, iteration) -> None:
        print(f"Laptime = {time_to_str(time)}  (iteration:{iteration}) (progress:{track_session.progress:.4f})")

    def save_results(track_layout) -> None:
        filename = filename_results + '.parquet'
        file_operations.save_parquet(track_layout, filename)
        print(f'intermediate results saved to {filename=}')

    try:

        track_session = race_lap.optimize_laptime(
            track_session=track_session,
            display_intermediate_results=print_results,
            save_intermediate_results=save_results,
            )
        print(f'optimization finished. {track_session.progress=}')

    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')
        # filename = filename_results + '.csv'
        # results_dataframe = race_lap.sim(track_session=track_session, verbose=True)
        # file_operations.save_csv(results_dataframe, filename)
        # print(f'final results saved to {filename=}')

        filename = filename_results + '.parquet'
        file_operations.save_parquet(track_session.track_layout, filename)
        print(f'final results saved to {filename=}')

        best_time = race_lap.sim(track_session=track_session)

        print(f'{race_car.name} - Simulated laptime = {time_to_str(best_time)}')


if __name__ == '__main__':
    main()
