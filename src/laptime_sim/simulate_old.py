import numpy as np
from numpy.typing import NDArray
from laptime_sim.simresults import SimResults


def simulate(car, line_coordinates: NDArray, slope: NDArray):
    # distance between nodes
    # ds = mag(np.diff(line_coordinates.T, 1, prepend=np.c_[line_coordinates[-1]]).T)

    # Calculate the first and second derivative of the points
    dX = np.gradient(line_coordinates, axis=0)
    ddX = np.gradient(dX, axis=0)
    ds = mag(dX)

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

    # g_car_lon = np.einsum("ij,ij->i", g, Tt)
    # np.testing.assert_array_almost_equal(g_car[:, 0], g_car_lon)

    k_car = k_car[:, 1]
    k_car_clipped = np.sign(k_car) * np.abs(k_car).clip(1e-3)

    v_max = np.abs((car.lat_limit - np.sign(k_car) * g_car[:, 1]) / k_car_clipped) ** 0.5

    v_a = np.zeros_like(v_max)  # simulated speed maximum acceleration
    v_b = np.zeros_like(v_max)  # simulated speed maximum braking

    for i in range(-90, len(v_max) - 1):  # negative index to simulate running start....
        j = i + 1  # index to previous timestep

        # max possible speed accelerating out of corners
        if v_a[i] < v_max[i]:  # check if previous speed was lower than max
            acc_lat = v_a[i] ** 2 * k_car[i] + g_car[i, 1]  # calc lateral acceleration based on
            # grip circle (no downforce accounted for)
            n = car.corner_acc / 50
            max_acc_grip = (car.acc_limit) * (1 - (np.abs(acc_lat) / car.lat_limit) ** n) ** (1 / n)

            force_engine = v_a[i] and car.P_engine_in_watt / v_a[i] or 0

            acc_lon = force_engine and min(max_acc_grip, force_engine / car.mass) or max_acc_grip
            aero_drag = v_a[i] ** 2 * car.c_drag / 2 / car.mass
            rolling_drag = car.c_roll * 9.81  # rolling resistance
            acc_lon -= aero_drag + rolling_drag + g_car[i, 0]  # <--  ISSUE!!!!!!!!!!!!!!!!!!!!!!!!

            v1 = (v_a[i] ** 2 + 2 * acc_lon * ds[i]) ** 0.5
            v_a[j] = min(v1, v_max[j])

        else:  # if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
            # acc_lon = g_car[j, 0]  # no grip available for longitudinal acceleration
            v_a[j] = min(v_a[i], v_max[j])  # speed remains the same

        # max possible speed braking into corners (backwards lap)
        v0 = v_b[i]

        if v0 < v_max[::-1][i]:
            acc_lat = v0**2 * k_car[::-1][i] + g_car[::-1][i, 1]

            # grip circle (no downforce accounted for)
            n = car.trail_braking / 50
            max_acc_grip = (car.dec_limit) * (1 - (np.abs(acc_lat) / car.lat_limit) ** n) ** (1 / n)

            # force_engine = 0
            # max_acc_engine = force_engine / mass
            # acc_lon = max_acc_grip
            aero_drag = v0**2 * -car.c_drag / 2 / car.mass
            rolling_drag = -car.c_roll * 9.81  # rolling resistance
            acc_lon = max_acc_grip - aero_drag - rolling_drag + g_car[::-1][i, 0]

            # acc_lon += g_car[::-1][j, 0]
            v1 = (v0**2 + 2 * acc_lon * ds[::-1][i]) ** 0.5
            v_b[j] = min(v1, v_max[::-1][j])

        else:
            # acc_lon = g_car[::-1][j, 0]
            v_b[j] = min(v0, v_max[::-1][j])

    v_b = v_b[::-1]  # flip the braking matrix
    speed = np.fmin(v_a, v_b)
    dt = 2 * ds / (speed + np.roll(speed, 1))

    return SimResults(line_coordinates, dt, speed, Nk, ds)


def mag(vector: NDArray) -> NDArray:
    return np.sum(vector**2, 1) ** 0.5


def dot(u: NDArray, v: NDArray) -> NDArray:
    return np.einsum("ij,ij->i", u, v)
