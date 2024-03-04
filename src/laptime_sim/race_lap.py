import itertools
import time
from typing import Callable, NamedTuple
import numpy as np
import pandas as pd

from geopandas import GeoDataFrame

from laptime_sim.car import Car
from laptime_sim.raceline import Raceline

OUTPUT_COLUMNS_NAMES = dict(
    distance='Distance (m)',
    speed='Speed (m/s)',
    speed_kph='Speed (km/h)',
    a_lon='Longitudinal acceleration (m/s2)',
    a_lat='Lateral acceleration (m/s2)',
    time='Timestamp',
    )

sim_results_type = NamedTuple('SimResults', [('time', np.ndarray), ('speed', np.ndarray), ('Nk', np.ndarray)])


def mag(vector: np.ndarray) -> np.ndarray:
    return np.sum(vector**2, 1) ** 0.5


def dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", u, v)


def simulate(car: Car, line_coordinates: np.ndarray, slope: np.ndarray, verbose=False) -> float | sim_results_type:
    # distance between nodes
    ds = mag(np.diff(line_coordinates.T, 1, prepend=np.c_[line_coordinates[-1]]).T)

    # Calculate the first and second derivative of the points
    dX = np.gradient(line_coordinates, axis=0)
    ddX = np.gradient(dX, axis=0)

    k = mag(np.cross(dX, ddX)) / mag(dX) ** 3  # magnitude of curvature

    T = dX / mag(dX)[:, None]  # unit tangent (direction of travel)
    B = np.cross(dX, ddX)  # binormal
    B = B / mag(B)[:, None]  # unit binormal
    N = np.cross(B, T)  # unit normal vector
    Nk = N * k[:, None]  # direction of curvature  (normal vector with magnitude 1/R)
    Tt = T  # car and raceline share tangent vector. We're not flying

    # Rotate Tt 90deg CW in xy-plane
    Bt = Tt[:, [1, 0, 2]]
    Bt[:, 1] *= -1
    Bt[:, 2] = slope  # align Bt with the track and normalize
    Bt = Bt / mag(Bt)[:, None]
    Nt = np.cross(Bt, Tt)

    # curvature projected in car frame [lon, lat, z]
    k_car = np.c_[dot(Nk, Tt), dot(Nk, Bt), dot(Nk, Nt)]
    # direction of gravity in car frame [lon, lat, z]
    g = np.array([0, 0, 9.81])[None, :]
    g_car = np.c_[dot(g, Tt), dot(g, Bt), dot(g, Nt)]

    k_car = k_car[:, 1]
    k_car = np.sign(k_car) * np.abs(k_car).clip(1e-3)

    v_max = np.abs((car.acc_grip_max - np.sign(k_car) * g_car[:, 1]) / k_car) ** 0.5

    v_a = np.ones(
        len(v_max)
    )  # simulated speed maximum acceleration (ones to avoid division by zero)
    v_b = np.ones(len(v_max))  # simulated speed maximum braking

    for i in range(-90, len(v_max)):  # negative index to simulate running start....
        j = i - 1  # index to previous timestep

        # max possible speed accelerating out of corners
        if v_a[j] < v_max[j]:  # check if previous speed was lower than max
            acc_lat = v_a[j] ** 2 * k_car[j] + g_car[j, 1]  # calc lateral acceleration based on

            # grip circle (no downforce accounted for)
            n = car.corner_acc / 50
            max_acc_grip = (car.acc_limit) * (1 - (np.abs(acc_lat) / car.acc_grip_max)**n)**(1/n)

            force_engine = v_a[j] and car.P_engine_in_watt / v_a[j] or 0
            # max_acc_engine = force_engine / mass
            acc_lon = force_engine and min(max_acc_grip, force_engine / car.mass) or max_acc_grip
            aero_drag = v_a[j]**2 * car.c_drag / car.mass
            rolling_drag = car.c_roll * 9.81       # rolling resistance
            acc_lon -= aero_drag + rolling_drag + g_car[j, 0]

            v1 = (v_a[j] ** 2 + 2 * acc_lon * ds[i]) ** 0.5
            v_a[i] = min(v1, v_max[i])
        else:  # if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
            acc_lon = g_car[j, 0]  # no grip available for longitudinal acceleration
            v_a[i] = min(v_a[j], v_max[i])  # speed remains the same

        # max possible speed braking into corners (backwards lap)
        v0 = v_b[j]

        if v0 < v_max[::-1][j]:
            acc_lat = v0**2 * k_car[::-1][j] + g_car[::-1][j, 1]

            # grip circle (no downforce accounted for)
            n = car.trail_braking / 50
            max_acc_grip = (car.dec_limit) * (1 - (np.abs(acc_lat) / car.acc_grip_max)**n)**(1/n)

            force_engine = 0
            # max_acc_engine = force_engine / mass
            acc_lon = max_acc_grip
            aero_drag = v0**2 * car.c_drag / car.mass
            rolling_drag = car.c_roll * 9.81       # rolling resistance
            acc_lon += aero_drag + rolling_drag + g_car[::-1][j, 0]

            # acc_lon += g_car[::-1][j, 0]
            v1 = (v0**2 + 2 * acc_lon * ds[::-1][i]) ** 0.5
            v_b[i] = min(v1, v_max[::-1][i])

        else:
            acc_lon = g_car[::-1][j, 0]
            v_b[i] = min(v0, v_max[::-1][i])

    v_b = v_b[::-1]  # flip the braking matrix
    speed = np.fmin(v_a, v_b)
    dt = 2 * ds / (speed + np.roll(speed, 1))
    t_lap = dt.cumsum()

    if not verbose:
        return t_lap[-1]

    return sim_results_type(t_lap, speed, Nk)


def results_dataframe(track_session, sim: sim_results_type) -> pd.DataFrame:

    line_coordinates = track_session.line_coords()
    ds = mag(
        np.diff(line_coordinates.T, 1, prepend=np.c_[line_coordinates[-1]]).T
    )  # distance from previous

    df = pd.DataFrame()
    df["time"] = sim.time
    df1 = pd.DataFrame(data=track_session.left_coords(), columns=["x", "y", "z"]).add_prefix("left_")
    df2 = pd.DataFrame(data=track_session.right_coords(), columns=["x", "y", "z"]).add_prefix("right_")
    df3 = pd.DataFrame(data=track_session.line_coords(), columns=["x", "y", "z"]).add_prefix("line_")
    df = pd.concat([df, df1, df2, df3], axis=1)
    df["distance"] = ds.cumsum() - ds[0]
    df["a_lat"] = -(sim.speed**2) * sim.Nk[:, 0]
    df["a_lon"] = np.gradient(sim.speed, df.distance) * sim.speed
    df["speed"] = sim.speed
    df["speed_kph"] = sim.speed * 3.6
    return df.set_index('time').rename(columns=OUTPUT_COLUMNS_NAMES)


def optimize_laptime(
        raceline: Raceline,
        display_intermediate_results: Callable[[float, int], None],
        save_intermediate_results: Callable[[GeoDataFrame], None],
        tolerance=0.005,
        ):

    timer1 = Timer()
    timer2 = Timer()

    racecar = raceline.car

    best_time = simulate(racecar, raceline.line_coords(), raceline.slope)
    display_intermediate_results(best_time, 0)
    # track_raceline = track_session.get_raceline()  # update raceline (create one if not present)
    save_intermediate_results(raceline.get_dataframe())

    for nr_iterations in itertools.count():

        new_line = raceline.get_new_line()
        laptime = simulate(racecar, raceline.line_coords(new_line), raceline.slope)
        raceline.update(new_line, laptime)

        if timer1.elapsed_time > 3:
            display_intermediate_results(best_time, nr_iterations)
            timer1.reset()

        if timer2.elapsed_time > 30:
            display_intermediate_results(best_time, nr_iterations)
            save_intermediate_results(raceline.get_dataframe())
            timer2.reset()

        if raceline.progress < tolerance:
            display_intermediate_results(best_time, nr_iterations)
            save_intermediate_results(raceline.get_dataframe())
            return raceline

    return raceline


class Timer:
    def __init__(self):
        self.time = time.time()

    def reset(self):
        self.time = time.time()

    @property
    def elapsed_time(self):
        return time.time() - self.time


def time_to_str(seconds: float) -> str:
    return "{:02.0f}:{:06.03f}".format(seconds % 3600 // 60, seconds % 60)


def get_max_acceleration(car, v: float, acc_lat):
    '''Return the acceleration limit using the performance envelope

    Args:
        car (Car): Car parameters
        v (float): velocity [m/s]
        acc_lat (float): lateral acceleration, a_y [m/s²]

    Returns:
        float: maximum longitudinal acceleration, a_x [m/s²]
    '''
    return get_acceleration(P_engine_in_watt=car.P_engine_in_watt, **car.dict(), v=v, acc_lat=acc_lat)


def get_max_deceleration(car, v, acc_lat):
    '''Return the deceleration limit using the performance envelope

    Args:
        car (Car): Car parameters
        v (float): velocity [m/s]
        acc_lat (float): lateral acceleration, a_y [m/s²]

    Returns:
        float: maximum longitudinal deceleration, -a_x [m/s²]
    '''
    return get_acceleration(0, -car.c_roll, car.mass, -car.c_drag,
                            car.acc_grip_max, car.dec_limit, car.trail_braking, v, acc_lat)


def get_acceleration(P_engine_in_watt, c_roll, mass, c_drag, acc_grip_max, acc_limit, corner_acc, v, acc_lat, **kwargs):

    n = corner_acc / 50
    max_acc_grip = (acc_limit) * (1 - (np.abs(acc_lat) / acc_grip_max)**n)**(1/n)
    force_engine = v and P_engine_in_watt / v or 0
    acceleration_max = force_engine and min(max_acc_grip, force_engine / mass) or max_acc_grip
    aero_drag = v**2 * c_drag / mass
    rolling_drag = c_roll * 9.81
    return acceleration_max - aero_drag - rolling_drag
