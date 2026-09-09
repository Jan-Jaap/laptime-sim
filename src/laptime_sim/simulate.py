"""Race simulator for determining the laptime when racing optimal speed."""

import numba as nb
import numpy as np
from numpy.typing import NDArray

from laptime_sim.car import Car
from laptime_sim.simresults import SimResults

GRAVITY = 9.81


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
    dt, speed, Nk, ds = simulate_kernel(
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


@nb.njit(cache=True, nogil=True)
def simulate_kernel(
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
    Determine the speed profile along the line with a forward acceleration pass
    and a reverse braking pass, taking the minimum of the two at every point.

    Returns: (dt, speed, Nk, ds)
    """
    n = line_coordinates.shape[0]
    ds, Nk, k_car_lat, g_car_lon, g_car_lat, v_max = _track_frame(line_coordinates, slope, lat_limit)

    # acceleration pass along the line
    v_a = np.zeros(n)
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

    # braking pass over the reversed line: deceleration limit instead of engine power,
    # gravity/drag/roll resistance push the car forward (hence the flipped signs)
    v_b = np.zeros(n)
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

    speed = np.minimum(v_a, v_b)

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

        # engine force, capped by the remaining grip; at standstill (or zero power)
        # the grip limit itself is the ceiling
        force_engine = P_engine_in_watt / speed[i] if speed[i] > 0.0 else 0.0
        acc_lon = min(max_acc_grip, force_engine / mass) if force_engine > 0.0 else max_acc_grip

        aero_drag = speed[i] ** 2 * c_drag / 2 / mass  # F=ma -> a=F/m
        rolling_drag = c_roll * GRAVITY
        acc_lon -= aero_drag + rolling_drag + g_car_lon[i]

        speed[j] = min(
            (speed[i] ** 2 + 2 * acc_lon * ds[i]) ** 0.5,
            v_max[j],
        )

    return speed


@nb.njit(cache=True)
def _track_frame(line_coordinates, slope, lat_limit):
    """
    Precompute the per-point track frame that drives the physics passes.

    From the line geometry (first and second central-difference gradients) it
    derives, for every point:
    - ds:      distance step along the line
    - Nk:      curvature vector (unit normal, pointing to the turn center, scaled by curvature)
    - k_lat:   curvature projected on the car's lateral axis (signed)
    - g_lon:   gravity component along the car's forward axis
    - g_lat:   gravity component along the car's lateral axis (from the track slope)
    - v_max:   cornering speed limit from the lateral grip budget
    """
    n = line_coordinates.shape[0]
    dX = _gradient(line_coordinates)
    ddX = _gradient(dX)

    ds = np.empty(n)
    Nk = np.empty((n, 3))
    k_car_lat = np.empty(n)
    g_car_lon = np.empty(n)
    g_car_lat = np.empty(n)
    v_max = np.empty(n)

    for i in range(n):
        dxi = (dX[i, 0], dX[i, 1], dX[i, 2])
        ddxi = (ddX[i, 0], ddX[i, 1], ddX[i, 2])

        tangent = v_normalize(dxi)  # car's forward axis, global frame
        ds[i] = v_norm(dxi)

        curvature = v_cross(dxi, ddxi)
        k = v_norm(curvature) / (ds[i] * ds[i] * ds[i])
        normal = v_cross(v_normalize(curvature), tangent)  # points to the turn center
        nk = v_scale(normal, k)
        Nk[i, 0] = nk[0]
        Nk[i, 1] = nk[1]
        Nk[i, 2] = nk[2]

        # car's lateral axis: tangent rotated 90 deg CW in the xy-plane, aligned with the track slope
        lateral = v_normalize((tangent[1], -tangent[0], slope[i]))

        g_lon = GRAVITY * tangent[2]
        g_lat = GRAVITY * lateral[2]
        k_lat = v_dot(nk, lateral)
        g_car_lon[i] = g_lon
        g_car_lat[i] = g_lat
        k_car_lat[i] = k_lat

        # lateral grip budget -> cornering speed limit (clamps written explicitly
        # so a 0/0 NaN curvature propagates instead of being clamped away)
        sgn = 1.0 if k_lat > 0 else -1.0 if k_lat < 0 else 0.0
        kk = abs(k_lat)
        if kk < 1e-3:
            kk = 1e-3
        budget = lat_limit - g_lat * sgn
        if budget < 0.0:
            budget = 0.0
        v_max[i] = np.sqrt(budget / kk)

    return ds, Nk, k_car_lat, g_car_lon, g_car_lat, v_max


@nb.njit(cache=True)
def _gradient(src):
    """Central differences along axis 0 with one-sided differences at the ends (np.gradient semantics)."""
    n = src.shape[0]
    out = np.empty_like(src)
    for i in range(n):
        for j in range(src.shape[1]):
            if i == 0:
                out[i, j] = src[1, j] - src[0, j]
            elif i == n - 1:
                out[i, j] = src[n - 1, j] - src[n - 2, j]
            else:
                out[i, j] = 0.5 * (src[i + 1, j] - src[i - 1, j])
    return out


@nb.njit(cache=True, inline="always")
def v_normalize(a):
    """Scale a 3-vector (x, y, z) tuple to unit length."""
    m = v_norm(a)
    return (a[0] / m, a[1] / m, a[2] / m)


@nb.njit(cache=True, inline="always")
def v_norm(a):
    """Length of a 3-vector (x, y, z) tuple."""
    return np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


@nb.njit(cache=True, inline="always")
def v_cross(a, b):
    """Cross product of two 3-vectors given as (x, y, z) tuples."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@nb.njit(cache=True, inline="always")
def v_dot(a, b):
    """Dot product of two 3-vectors given as (x, y, z) tuples."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@nb.njit(cache=True, inline="always")
def v_scale(a, s):
    """Multiply a 3-vector (x, y, z) tuple by a scalar."""
    return (a[0] * s, a[1] * s, a[2] * s)
