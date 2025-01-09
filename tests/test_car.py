import pytest
from laptime_sim.car import Car


@pytest.fixture()
def car():
    return Car(
        name="Ferrari",
        P_engine=200,
        mass=1000,
        acc_limit=10,
        dec_limit=10,
        acc_grip_max=10,
        c_drag=0.1,
        c_roll=0.1,
        trail_braking=100,
        corner_acc=100,
    )


def test_car(car):
    assert car.name == "Ferrari"


def test_car_from_toml():
    car = Car.from_toml("./cars/BMW_Z3M.toml")
    assert car.file_name == "BMW_Z3M"
