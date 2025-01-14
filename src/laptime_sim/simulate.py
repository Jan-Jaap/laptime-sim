import numpy as np

from laptime_sim.simresults import SimResults
from laptime_sim.car import Car

"""Race simulator for determining the laptime when racing optimal speed"""


def simulate(car: Car, line_coordinates: np.ndarray, slope: np.ndarray) -> SimResults:
    """
    Run the race simulator to determine the laptime when racing optimal speed.

    Args:
        car (Car): The car object representing the vehicle.
        line_coordinates (np.ndarray): The array of coordinates representing the raceline.
        slope (np.ndarray): The array of slopes at each point in the raceline.

    Returns:
        SimResults: The simulation results containing the line coordinates, timestep, speed,
        curvature direction, and distance.
    """
    # distance between nodes
    # ds = mag(np.diff(line_coordinates.T, 1, prepend=np.c_[line_coordinates[-1]]).T)

    # Calculate the first and second derivative of the points
    dX = np.gradient(line_coordinates, axis=0)
    ddX = np.gradient(dX, axis=0)
    ds = mag(dX)
    T = dX / ds[:, None]  # unit tangent (direction of travel)
    B = np.cross(dX, ddX)  # binormal

    k = mag(B) / ds**3  # magnitude of curvature

    B = B / mag(B)[:, None]  # unit binormal
    N = np.cross(B, T)  # unit normal vector
    Nk = N * k[:, None]  # curvature normal vector
    # T = T  # car velocity and raceline are tangent. We're not flying

    # Rotate Tt 90deg CW in xy-plane
    Bt = T[:, [1, 0, 2]]
    Bt[:, 1] *= -1
    Bt[:, 2] = slope  # align Bt with the track and normalize
    Bt = Bt / mag(Bt)[:, None]

    # lateral curvature in car frame
    k_car_lat = np.einsum("ij,ij->i", Nk, Bt)
    k_car_lat = np.sign(k_car_lat) * np.abs(k_car_lat).clip(1e-3)

    # gravity vector
    g = np.array([[0, 0, 9.81]])  # [None, :]
    g_car_lon = np.einsum("ij,ij->i", g, T)
    g_car_lat = np.einsum("ij,ij->i", g, Bt)
    v_max = np.abs((car.acc_grip_max * np.sign(k_car_lat) - g_car_lat) / k_car_lat) ** 0.5

    v_a = np.zeros_like(v_max)  # simulated speed maximum acceleration
    v_b = np.zeros_like(v_max)  # simulated speed maximum braking

    for i in range(-90, len(v_max) - 1):  # negative index to simulate running start....
        # max possible speed accelerating out of corners
        v_a[i + 1] = min(
            calc_speed(
                car.get_acceleration,
                ds[i],
                k_car_lat[i],
                g_car_lon[i],
                g_car_lat[i],
                v_max[i],
                v_a[i],
            ),
            v_max[i + 1],
        )

        v_b[i + 1] = min(
            calc_speed(
                car.get_deceleration,
                ds[::-1][i],
                k_car_lat[::-1][i],
                g_car_lon[::-1][i],
                g_car_lat[::-1][i],
                v_max[::-1][i],
                v_b[i],
            ),
            v_max[::-1][i],
        )

    v_b = v_b[::-1]  # flip the braking matrix

    speed = np.fmin(v_a, v_b)
    dt = 2 * ds / (speed + np.roll(speed, 1))
    return SimResults(line_coordinates, dt, speed, Nk, ds)


def calc_speed(get_acceleration, ds, k_car_lat, g_car_lon, g_car_lat, v_max0, v0):
    # v0 = v_a[i - 1]

    if v0 >= v_max0:  # if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
        return v0  # speed remains the same

    # calc lateral acceleration based on grip circle (no downforce accounted for)
    acc_lat = v0**2 * k_car_lat + g_car_lat
    acc_lon = get_acceleration(v0, acc_lat) + g_car_lon
    return (v0**2 + 2 * acc_lon * ds) ** 0.5


def mag(vector: np.ndarray) -> np.ndarray:
    return np.sum(vector**2, 1) ** 0.5
