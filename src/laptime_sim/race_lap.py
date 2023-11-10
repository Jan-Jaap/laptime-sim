import itertools
import time
from typing import Callable
import numpy as np
import pandas as pd

from track import TrackSession


def mag(vector):
    return np.sum(vector**2, 1) ** 0.5


def dot(u, v):
    return np.einsum("ij,ij->i", u, v)


def get_new_line_parameters(session: TrackSession) -> tuple[int, int, float]:
    location = np.random.choice(session.len, p=session.heatmap / sum(session.heatmap))
    length = np.random.randint(1, 60)
    deviation = np.random.randn() / 10
    return location, length, deviation


def sim(track_session: TrackSession, raceline=None, verbose=False) -> float | np.ndarray:
    car = track_session.car
    line_coordinates = track_session.line_coords(raceline)
    ds = mag(
        np.diff(line_coordinates.T, 1, prepend=np.c_[line_coordinates[-1]]).T
    )  # distance from previous

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
    Bt[:, 2] = track_session.slope  # align Bt with the track and normalize
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

    return results_dataframe(track_session, time, speed, Nk)


def results_dataframe(track_session: TrackSession, time, speed, Nk) -> pd.DataFrame:
    line_coordinates = track_session.line_coords()
    ds = mag(
        np.diff(line_coordinates.T, 1, prepend=np.c_[line_coordinates[-1]]).T
    )  # distance from previous

    df = pd.DataFrame()
    df["time"] = time
    df1 = pd.DataFrame(
        data=track_session.border_right, columns=["x", "y", "z"]
    ).add_prefix("inner_")
    df2 = pd.DataFrame(
        data=track_session.border_left, columns=["x", "y", "z"]
    ).add_prefix("outer_")
    df3 = pd.DataFrame(
        data=track_session.line_coords(), columns=["x", "y", "z"]
    ).add_prefix("line_")
    # df3 = pd.DataFrame(data=track.line, columns=['x','y','z']).add_prefix('line_')
    df = pd.concat([df, df1, df2, df3], axis=1)
    df["race_line_position"] = track_session.line_pos
    df["distance"] = ds.cumsum() - ds[0]
    df["a_lat"] = -(speed**2) * Nk[:, 0]
    df["a_lon"] = np.gradient(speed, df.distance) * speed
    df["speed"] = speed
    return df


def optimize_laptime(
    track_session: TrackSession,
    display_intermediate_results: Callable[[float, int], None],
    save_intermediate_results: Callable[[pd.DataFrame], None],
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
    save_intermediate_results(track_session)

    for nr_iterations in itertools.count():
        new_line_parameters = get_new_line_parameters(session=track_session)
        new_line = track_session.get_new_line(parameters=new_line_parameters)
        laptime = sim(track_session=track_session, raceline=new_line)

        dt = best_time - laptime

        if laptime < best_time:
            track_session.update_best_line(new_line, improvement=dt)
            best_time = laptime

        track_session.heatmap = (track_session.heatmap + 0.0015) / 1.0015  # slowly to one

        if timer1.elapsed_time > 3:
            # pd.DataFrame(data=heatmap).to_csv('./simulated/optimized_results.csv', index=False)
            display_intermediate_results(best_time, nr_iterations, track_session)
            timer1.reset()

        if timer2.elapsed_time > 30:
            # save_intermediate_results(sim(track_session=track_session, verbose=True))
            save_intermediate_results(track_session)
            timer2.reset()

    return track_session
