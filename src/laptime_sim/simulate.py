import numpy as np
from numba import njit
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

    g = np.array([[0, 0, 9.81]])
    g_car_lon = np.einsum("ij,ij->i", g, T)
    g_car_lat = np.einsum("ij,ij->i", g, Bt)
    k_car_lat = np.einsum("ij,ij->i", Nk, Bt)

    v_max = np.where(
        np.abs(k_car_lat) > 1e-3,
        np.abs((car.lat_limit - g_car_lat * np.sign(k_car_lat)) / k_car_lat) ** 0.5,
        500,
    )

    v_a = np.zeros_like(v_max)  # simulated speed maximum acceleration
    v_b = np.zeros_like(v_max)  # simulated speed maximum braking

    # data = np.column_stack((ds, k_car_lat, g_car_lon, g_car_lat, v_max))
    calc_speed(
        v_a,
        ds,
        k_car_lat,
        g_car_lon,
        g_car_lat,
        v_max,
        car.mass,
        car.acc_limit,
        car.lat_limit,
        car.c_drag,
        car.c_roll,
        car.corner_acc,
        car.P_engine_in_watt,
    )

    calc_speed(
        v_b,
        ds[::-1],
        k_car_lat[::-1],
        g_car_lon[::-1],
        g_car_lat[::-1],
        v_max[::-1],
        car.mass,
        car.dec_limit,
        car.lat_limit,
        -car.c_drag,
        -car.c_roll,
        car.trail_braking,
        0,
    )

    v_b = v_b[::-1]  # flip the braking matrix
    speed = np.fmin(v_a, v_b)
    dt = 2 * ds / (speed + np.roll(speed, 1))
    return SimResults(line_coordinates, dt, speed, Nk, ds)


@njit
def calc_speed(
    speed,
    ds,
    k_car_lat,
    g_car_lon,
    g_car_lat,
    v_max,
    mass: float,
    acc_limit: float,
    acc_grip_max: float,
    c_drag: float,
    c_roll: float,
    corner_acc: int,
    P_engine_in_watt: float,
):
    for i in range(-90, len(v_max) - 1):  # negative index to simulate running start....
        if not speed[i] < v_max[i]:  # if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
            speed[i + 1] = min(v_max[i], v_max[i + 1])
            continue  # speed remains the same

        # calc lateral acceleration based on grip circle (no downforce accounted for)
        acc_lat = speed[i] ** 2 * k_car_lat[i] + g_car_lat[i]
        n = corner_acc / 50
        a_max = (acc_limit) * (1 - (np.abs(acc_lat) / acc_grip_max) ** n) ** (1 / n)

        force_engine = speed[i] and P_engine_in_watt / speed[i] or 0
        a_max = force_engine and min(a_max, force_engine / mass) or a_max

        aero_drag = speed[i] ** 2 * c_drag / mass
        rolling_drag = c_roll * 9.81
        acc_lon = a_max - aero_drag - rolling_drag
        acc_lon += g_car_lon[i]

        v1 = (speed[i] ** 2 + 2 * acc_lon * ds[i]) ** 0.5
        speed[i + 1] = min(v1, v_max[i + 1])


def mag(vector: np.ndarray) -> np.ndarray:
    return np.sum(vector**2, 1) ** 0.5
