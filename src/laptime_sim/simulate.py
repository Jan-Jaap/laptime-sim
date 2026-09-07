import numba as nb
import numpy as np
from numpy.typing import NDArray

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults

"""Race simulator for determining the laptime when racing optimal speed"""


def simulate(car: Car, line_coordinates: NDArray, slope: NDArray) -> SimResults:
    """
    Run the race simulator to determine the laptime when racing optimal speed.

    Args:
        car (Car): The car object representing the vehicle.
        line_coordinates (NDArray): The array of coordinates representing the raceline.
        slope (NDArray): The array of slopes at each point in the raceline.

    Returns:
        SimResults: The simulation results containing the line coordinates, timestep, speed,
        curvature direction, and distance.
    """
    dt, speed, Nk, ds = _simulate_kernel(
        line_coordinates,
        slope,
        lat_limit=car.lat_limit,
        mass=car.mass,
        acc_limit=car.acc_limit,
        dec_limit=car.dec_limit,
        c_drag=car.c_drag,
        c_roll=car.c_roll,
        corner_acc=car.corner_acc,
        trail_braking=car.trail_braking,
        P_engine_in_watt=car.P_engine_in_watt,
    )
    return SimResults(line_coordinates, dt, speed, Nk, ds)


@nb.njit(cache=True)
def _simulate_kernel(  # noqa: PLR0913, PLR0914
    line_coordinates,
    slope,
    lat_limit,
    mass,
    acc_limit,
    dec_limit,
    c_drag,
    c_roll,
    corner_acc,
    trail_braking,
    P_engine_in_watt,
):
    """
    Compiled equivalent of the original numpy precompute (gradient, curvature,
    gravity projection) followed by the forward acceleration and reverse
    braking passes. Bit-for-bit equivalent to the original formula's semantics.

    Returns: (dt, speed, Nk, ds)
    """
    n = line_coordinates.shape[0]

    dX = np.empty((n, 3))
    ddX = np.empty((n, 3))
    for i in range(n):
        for j in range(3):
            if i == 0:
                dX[i, j] = line_coordinates[1, j] - line_coordinates[0, j]
            elif i == n - 1:
                dX[i, j] = line_coordinates[n - 1, j] - line_coordinates[n - 2, j]
            else:
                dX[i, j] = 0.5 * (line_coordinates[i + 1, j] - line_coordinates[i - 1, j])
    for i in range(n):
        for j in range(3):
            if i == 0:
                ddX[i, j] = dX[1, j] - dX[0, j]
            elif i == n - 1:
                ddX[i, j] = dX[n - 1, j] - dX[n - 2, j]
            else:
                ddX[i, j] = 0.5 * (dX[i + 1, j] - dX[i - 1, j])

    ds = np.empty(n)
    Nk = np.empty((n, 3))
    g_car_lon = np.empty(n)
    g_car_lat = np.empty(n)
    k_car_lat = np.empty(n)
    v_max = np.empty(n)

    for i in range(n):
        m = np.sqrt(dX[i, 0] ** 2 + dX[i, 1] ** 2 + dX[i, 2] ** 2)
        ds[i] = m
        tx = dX[i, 0] / m
        ty = dX[i, 1] / m
        tz = dX[i, 2] / m

        bx = dX[i, 1] * ddX[i, 2] - dX[i, 2] * ddX[i, 1]
        by = dX[i, 2] * ddX[i, 0] - dX[i, 0] * ddX[i, 2]
        bz = dX[i, 0] * ddX[i, 1] - dX[i, 1] * ddX[i, 0]
        bm = np.sqrt(bx * bx + by * by + bz * bz)
        k = bm / (m * m * m)
        bx /= bm
        by /= bm
        bz /= bm

        nx = by * tz - bz * ty
        ny = bz * tx - bx * tz
        nz = bx * ty - by * tx
        Nk[i, 0] = nx * k
        Nk[i, 1] = ny * k
        Nk[i, 2] = nz * k

        # Rotate T 90deg CW in the xy-plane and align with the track slope
        btx = ty
        bty = -tx
        btz = slope[i]
        btm = np.sqrt(btx * btx + bty * bty + btz * btz)
        btx /= btm
        bty /= btm
        btz /= btm

        g_car_lon[i] = 9.81 * tz
        g_car_lat[i] = 9.81 * btz
        k_car_lat[i] = Nk[i, 0] * btx + Nk[i, 1] * bty + Nk[i, 2] * btz

        if k_car_lat[i] > 0:
            sgn = 1.0
        elif k_car_lat[i] < 0:
            sgn = -1.0
        else:
            sgn = 0.0
        kk = abs(k_car_lat[i])
        if kk < 1e-3:
            kk = 1e-3
        num = lat_limit - g_car_lat[i] * sgn
        if num < 0:
            num = 0.0
        v_max[i] = np.sqrt(num / kk)

    v_a = np.zeros(n)  # simulated speed maximum acceleration
    v_b = np.zeros(n)  # simulated speed maximum braking

    # run the simulation in forward direction for acceleration
    calc_speed(
        v_a,
        ds,
        k_car_lat,
        g_car_lon,
        g_car_lat,
        v_max,
        mass,
        acc_limit,
        lat_limit,
        c_drag,
        c_roll,
        corner_acc,
        P_engine_in_watt,
    )
    # run the simulation in reverse direction for braking
    calc_speed(
        v_b[::-1],
        ds[::-1],
        k_car_lat[::-1],
        -g_car_lon[::-1],
        g_car_lat[::-1],
        v_max[::-1],
        mass,
        dec_limit,
        lat_limit,
        -c_drag,
        -c_roll,
        trail_braking,
        0.0,
    )

    speed = np.empty(n)
    for i in range(n):
        if v_a[i] < v_b[i]:
            speed[i] = v_a[i]
        else:
            speed[i] = v_b[i]

    dt = np.empty(n)
    for i in range(n):
        im = i - 1 if i > 0 else n - 1  # np.roll(speed, 1) semantics
        dt[i] = 2.0 * ds[i] / (speed[i] + speed[im])

    return dt, speed, Nk, ds


@nb.njit(cache=True)
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
    """
    Run the race simulator in forward or reverse direction to determine the laptime when racing optimal speed.

    The function calculates the speed for each point in the raceline and returns the array of speeds.
    The function is optimized for speed and uses numba's njit decorator to compile the function to machine code.

    Parameters:
        speed (NDArray): The array of speeds to be calculated.
        ds (NDArray): The array of distances between points in the raceline.
        k_car_lat (NDArray): The array of lateral curvature of the raceline.
        g_car_lon (NDArray): The array of longitudinal gravity components.
        g_car_lat (NDArray): The array of lateral gravity components.
        v_max (NDArray): The array of maximum speeds for each point in the raceline.
        mass (float): The mass of the car.
        acc_limit (float): The acceleration limit of the car.
        lat_limit (float): The lateral acceleration limit of the car.
        c_drag (float): The drag coefficient of the car.
        c_roll (float): The rolling resistance coefficient of the car.
        corner_acc (int): The acceleration factor during cornering (0-100).
        P_engine_in_watt (float): The power of the engine in watt.

    Returns:
        NDArray: The array of calculated speeds.
    """
    for i in range(-90, len(v_max) - 1):  # negative index to simulate running start....
        j = i + 1

        if (
            not speed[i] < v_max[i]
        ):  # if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
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


@nb.njit(cache=True)
def mag(vector: NDArray) -> NDArray:
    """
    Calculate the magnitude of vectors along the second axis of the input array.

    Parameters:
        vector (NDArray): A numpy array where each row represents a vector.

    Returns:
        NDArray: A 1D numpy array containing the magnitudes of the vectors.
    """

    return np.sum(vector**2, 1) ** 0.5
