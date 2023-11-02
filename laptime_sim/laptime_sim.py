import time
import itertools

import file_operations
from car import Car
import race_lap


NAME_CAR = 'Peugeot_205RFS'
# NAME_TRACK = "2020_zandvoort"
# NAME_TRACK = "20191030_Circuit_Zandvoort"
NAME_TRACK = "202209022_Circuit_Meppen"


class Timer:
    def __init__(self):
        self.time = time.time()
    def reset(self):
        self.time = time.time()
    @property
    def elapsed_time(self):
        return time.time() - self.time


def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)


def main():
    
    race_car = Car.from_file(f"./cars/{NAME_CAR}.json")
    track = file_operations.get_track_data(NAME_TRACK, NAME_CAR)
    
    if track is None: return
    
    filename_results = f"{NAME_CAR}_{NAME_TRACK}_simulated.csv"
    
    print(f'Track has {track.len} datapoints')
    
    best_time = race_lap.race(track=track, car=race_car)
    
    print(f'{race_car.name} - Simulated laptime = {laptime_str(best_time)}')

    timer1 = Timer()
    timer2 = Timer()

    try:
        for nr_iterations in itertools.count():
            new_line = race_lap.get_new_line(track=track)
            laptime = race_lap.race(track=track, car=race_car, raceline=new_line)
            
            if laptime < best_time:
                best_time = laptime
                track.best_line = new_line
            
            if timer1.elapsed_time > 3:
                print(f"Laptime = {laptime_str(best_time)}  (iteration:{nr_iterations})")
                timer1.reset()
            
            if timer2.elapsed_time > 30:
                timer2.reset()
                file_operations.save_results(race_lap.race(track=track, car=race_car, verbose=True), filename_results)
                print(f'intermediate results saved to {filename_results=}')
                
    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')

    file_operations.save_results(race_lap.race(track=track, car=race_car, verbose=True), filename_results)

    print(f'{race_car.name} - Simulated laptime = {laptime_str(best_time)}')


if __name__ == '__main__':
    main()
