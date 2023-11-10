
# from track_data import get_track_data, interpolate_track,

import file_operations as file_operations
from car import Car
import race_lap
import shapely

from icecream import ic

NAME_CAR = 'Peugeot_205RFS'
# NAME_TRACK = "2020_zandvoort"
# NAME_TRACK = "20191030_Circuit_Zandvoort"
NAME_TRACK = "20191128_Circuit_Assen"


def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds % 3600 // 60, seconds % 60)


def main():

    race_car = Car.from_json(f"./cars/{NAME_CAR}.json")
    track = file_operations.get_track_data(NAME_TRACK, NAME_CAR)

    print(f'Track has {track.len} datapoints')
    print(f'{race_car.name} - Simulated laptime = {laptime_str(race_lap.race(track=track, car=race_car))}')

    track = file_operations.interpolate_track(track, 1000)

    point_list = []
    for i, point_left in enumerate(track.border_left):
        point_right = track.border_right[i]
        point_list.append(([(point_left), (point_right)]))
    lines = shapely.MultiLineString(lines=point_list)
    ic(lines)
    print(f'Track has {track.len} datapoints')
    print(f'{race_car.name} - Simulated laptime = {laptime_str(race_lap.race(track=track, car=race_car))}')

    track = file_operations.add_lines(track)


if __name__ == '__main__':
    main()
