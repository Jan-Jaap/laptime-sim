import itertools
import time
from typing import Callable, NamedTuple
import numpy as np
import pandas as pd
from car import Car

from tracksession import TrackSession
from geopandas import GeoDataFrame

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


def sim(track_session: TrackSession, raceline=None, verbose=False) -> float | pd.DataFrame:
    results = simulate(
            car=track_session.car,
            line_coordinates=track_session.line_coords(raceline),
            slope=track_session.slope,
            verbose=verbose)

    if not verbose:
        return results

    return results_dataframe(track_session, results)


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

    for i in range(-100, len(v_max)):  # negative index to simulate running start....
        j = i - 1  # index to previous timestep

        # max possible speed accelerating out of corners
        if v_a[j] < v_max[j]:  # check if previous speed was lower than max
            acc_lat = (
                v_a[j] ** 2 * k_car[j] + g_car[j, 1]
            )  # calc lateral acceleration based on
            acc_lon = car.get_max_acc(v_a[j], acc_lat)
            acc_lon -= g_car[j, 0]
            v1 = (v_a[j] ** 2 + 2 * acc_lon * ds[i]) ** 0.5
            v_a[i] = min(v1, v_max[i])
        else:  # if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
            acc_lon = 0  # no grip available for longitudinal acceleration
            v_a[i] = min(v_a[j], v_max[i])  # speed remains the same

        # max possible speed braking into corners (backwards lap)
        v0 = v_b[j]
        if v0 < v_max[::-1][j]:
            acc_lat = v0**2 * k_car[::-1][j] + g_car[::-1][j, 1]
            acc_lon = car.get_min_acc(v0, acc_lat)
            acc_lon += g_car[::-1][j, 0]
            v1 = (v0**2 + 2 * acc_lon * ds[::-1][i]) ** 0.5
            v_b[i] = min(v1, v_max[::-1][i])

        else:
            acc_lon = 0
            v_b[i] = min(v0, v_max[::-1][i])

    v_b = v_b[::-1]  # flip the braking matrix
    speed = np.fmin(v_a, v_b)
    dt = 2 * ds / (speed + np.roll(speed, 1))
    time = dt.cumsum()

    if not verbose:
        return time[-1]

    return sim_results_type(time, speed, Nk)


def results_dataframe(track_session: TrackSession, sim: sim_results_type) -> pd.DataFrame:
    line_coordinates = track_session.line_coords()
    ds = mag(
        np.diff(line_coordinates.T, 1, prepend=np.c_[line_coordinates[-1]]).T
    )  # distance from previous

    df = pd.DataFrame()
    df["time"] = sim.time
    df1 = pd.DataFrame(data=track_session.border_right, columns=["x", "y", "z"]).add_prefix("inner_")
    df2 = pd.DataFrame(data=track_session.border_left, columns=["x", "y", "z"]).add_prefix("outer_")
    df3 = pd.DataFrame(data=line_coordinates, columns=["x", "y", "z"]).add_prefix("line_")
    df = pd.concat([df, df1, df2, df3], axis=1)
    df["distance"] = ds.cumsum() - ds[0]
    df["a_lat"] = -(sim.speed**2) * sim.Nk[:, 0]
    df["a_lon"] = np.gradient(sim.speed, df.distance) * sim.speed
    df["speed"] = sim.speed
    df["speed_kph"] = sim.speed * 3.6
    return df.set_index('time').rename(columns=OUTPUT_COLUMNS_NAMES)


def optimize_laptime(
    track_session: TrackSession,
    display_intermediate_results: Callable[[float, int], None],
    save_intermediate_results: Callable[[GeoDataFrame], None],
):
    class Timer:
        def __init__(self):
            self.time = time.time()

        def reset(self):
            self.time = time.time()

        @property
        def elapsed_time(self):
            return time.time() - self.time

    timer1 = Timer()
    timer2 = Timer()

    best_time = sim(track_session=track_session)
    display_intermediate_results(best_time, 0)
    save_intermediate_results(track_session.track_layout)

    for nr_iterations in itertools.count():

        if track_session.progress < 0.01:
            display_intermediate_results(best_time, nr_iterations)
            save_intermediate_results(track_session.track_layout)
            return track_session

        new_line = track_session.get_new_line()
        laptime = sim(track_session=track_session, raceline=new_line)
        improvement = best_time - laptime

        if improvement > 0:
            best_time = laptime

        track_session.update(new_line, improvement=improvement)

        if timer1.elapsed_time > 3:
            display_intermediate_results(best_time, nr_iterations)
            timer1.reset()

        if timer2.elapsed_time > 30:
            display_intermediate_results(best_time, nr_iterations)
            save_intermediate_results(track_session.track_layout)
            timer2.reset()

    return track_session


def time_to_str(seconds: float) -> str:
    return "{:02.0f}:{:06.03f}".format(seconds % 3600 // 60, seconds % 60)
