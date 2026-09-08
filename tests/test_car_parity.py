import numpy as np

from laptime_sim.registry import cars

CAR_LIST = cars()


def kernel_reference_acceleration(car, v, acc_lat):
    """Frozen copy of the acceleration math inline in simulate.calc_speed."""
    n = car.corner_acc / 50
    max_acc_grip = (car.acc_limit) * (1 - (np.abs(acc_lat) / car.lat_limit) ** n) ** (1 / n)
    force_engine = v and car.P_engine_in_watt / v or 0
    acc_lon = force_engine and min(max_acc_grip, force_engine / car.mass) or max_acc_grip
    aero_drag = v**2 * car.c_drag / 2 / car.mass
    rolling_drag = car.c_roll * 9.81
    return acc_lon - aero_drag - rolling_drag


def kernel_reference_deceleration(car, v, acc_lat):
    """Frozen copy of the braking math inline in simulate.calc_speed."""
    n = car.trail_braking / 50
    max_dec_grip = (car.dec_limit) * (1 - (np.abs(acc_lat) / car.lat_limit) ** n) ** (1 / n)
    aero_drag = v**2 * car.c_drag / 2 / car.mass
    rolling_drag = car.c_roll * 9.81
    return max_dec_grip + aero_drag + rolling_drag


def test_car_matches_kernel_math():
    """Car.get_acceleration/get_deceleration (used by the UI) must stay in sync
    with the physics inline in simulate.calc_speed (used by the optimizer)."""
    speeds = [0.0, 1.0, 5.0, 20.0, 50.0, 80.0]
    for car in CAR_LIST:
        acc_lats = np.linspace(-car.lat_limit, car.lat_limit, 21)
        for v in speeds:
            for acc_lat in acc_lats:
                expected_acc = kernel_reference_acceleration(car, v, acc_lat)
                np.testing.assert_allclose(
                    car.get_acceleration(v=v, acc_lat=acc_lat),
                    expected_acc,
                    rtol=1e-9,
                    err_msg=f"{car.name}: acceleration mismatch at v={v}, acc_lat={acc_lat}",
                )
                expected_dec = kernel_reference_deceleration(car, v, acc_lat)
                np.testing.assert_allclose(
                    car.get_deceleration(v=v, acc_lat=acc_lat),
                    expected_dec,
                    rtol=1e-9,
                    err_msg=f"{car.name}: deceleration mismatch at v={v}, acc_lat={acc_lat}",
                )
