import pytest
import numpy as np


from laptime_sim.race_lap import *


def test_mag():
    test_data = np.array([[3.0, 4.0]])
    assert mag(test_data) != None
    assert mag(test_data) == 5.0
