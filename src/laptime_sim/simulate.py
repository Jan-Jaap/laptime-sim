import numpy as np
import numba as nb
from laptime_sim.simresults import SimResults
from laptime_sim.car import Car

"""Race simulator for determining the laptime when racing optimal speed"""

# gravity vector


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
    ds = mag(dX)  # traveled distance
    T = dX / ds[:, None]  # unit tangent (direction of travel)
    B = np.cross(dX, ddX)  # binormal
    k = mag(B) / ds**3  # magnitude of curvature

    B = B / mag(B)[:, None]  # unit binormal
    N = np.cross(B, T)  # unit normal vector
    Nk = N * k[:, None]  # curvature normal vector

    # Rotate Tt 90deg CW in xy-plane
    Bt = T[:, [1, 0, 2]]
    Bt[:, 1] *= -1
    Bt[:, 2] = slope  # align Bt with the track and normalize
    Bt = Bt / mag(Bt)[:, None]

    g = np.array([[0, 0, 9.81]])
    g_car_lon = np.einsum("ij,ij->i", g, T, optimize=True)
    g_car_lat = np.einsum("ij,ij->i", g, Bt, optimize=True)
    k_car_lat = np.einsum("ij,ij->i", Nk, Bt, optimize=True)

    v_max = np.sqrt(np.fmax(0, car.lat_limit - g_car_lat * np.sign(k_car_lat)) / np.fmax(1e-3, np.abs(k_car_lat)))

    car_acc = {
        "mass": car.mass,
        "acc_limit": car.acc_limit,
        "lat_limit": car.lat_limit,
        "c_drag": car.c_drag,
        "c_roll": car.c_roll,
        "corner_acc": car.corner_acc,
        "P_engine_in_watt": car.P_engine_in_watt,
    }

    car_dec = {
        "mass": car.mass,
        "acc_limit": car.dec_limit,
        "lat_limit": car.lat_limit,
        "c_drag": -car.c_drag,
        "c_roll": -car.c_roll,
        "corner_acc": car.trail_braking,
        "P_engine_in_watt": 0,
    }

    v_a = np.zeros_like(v_max)  # simulated speed maximum acceleration
    v_b = np.zeros_like(v_max)  # simulated speed maximum braking

    # run the simulation in forward direction for acceleration
    calc_speed(v_a, ds, k_car_lat, g_car_lon, g_car_lat, v_max, **car_acc)
    # run the simulation in reverse direction for braking
    calc_speed(v_b[::-1], ds[::-1], k_car_lat[::-1], -g_car_lon[::-1], g_car_lat[::-1], v_max[::-1], **car_dec)
    # v_b = v_b[::-1]  # flip the braking array not required since we pass in the pointer backwards

    speed = np.fmin(v_a, v_b)
    dt = 2 * ds / (speed + np.roll(speed, 1))
    return SimResults(line_coordinates, dt, speed, Nk, ds)


@nb.njit
def calc_speed(
    speed,
    ds,
    k_car_lat,
    g_car_lon,
    g_car_lat,
    v_max,
    mass: float,
    acc_limit: float,
    lat_limit: float,
    c_drag: float,
    c_roll: float,
    corner_acc: int,
    P_engine_in_watt: float,
):
    for i in range(-90, len(v_max) - 1):  # negative index to simulate running start....
        j = i + 1

        if not speed[i] < v_max[i]:  # if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
            speed[j] = min(speed[i], v_max[j])
            continue  # speed remains the same

        # calc lateral acceleration based on grip circle (no downforce accounted for)
        acc_lat = speed[i] ** 2 * k_car_lat[i] + g_car_lat[i]
        n = corner_acc / 50
        max_acc_grip = (acc_limit) * (1 - (np.abs(acc_lat) / lat_limit) ** n) ** (1 / n)

        # calc longitudinal acceleration. It's not the most readable code, but it's really fast
        force_engine = speed[i] and P_engine_in_watt / speed[i] or 0
        acc_lon = force_engine and min(max_acc_grip, force_engine / mass) or max_acc_grip

        aero_drag = speed[i] ** 2 * c_drag / 2 / mass  # F=ma -> a=F/m
        rolling_drag = c_roll * 9.81
        acc_lon -= aero_drag + rolling_drag + g_car_lon[i]

        speed[j] = min(
            (speed[i] ** 2 + 2 * acc_lon * ds[i]) ** 0.5,
            v_max[j],
        )

    return speed


@nb.njit
def mag(vector: np.ndarray) -> np.ndarray:
    return np.sum(vector**2, 1) ** 0.5
