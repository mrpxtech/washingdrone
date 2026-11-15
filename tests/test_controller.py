import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from washsim.controller import Controller, allocation_matrix
from washsim.params import Params


def test_hover_mix_is_equal_positive_thrust():
    params = Params()
    controller = Controller(params)
    command, saturated = controller.mix(params.hover_thrust_n, np.zeros(3))
    assert not saturated
    assert np.allclose(command, params.hover_thrust_n / 4.0, atol=1e-6)
    assert np.all(command > 0.0)


def test_positive_roll_torque_loads_the_left_rotor():
    params = Params()
    controller = Controller(params)
    hover, _ = controller.mix(params.hover_thrust_n, np.zeros(3))
    rolled, _ = controller.mix(params.hover_thrust_n, np.array([0.2, 0.0, 0.0]))
    # tau_x = L (T_left - T_right), and the left rotor is index 1.
    assert rolled[1] > hover[1]
    assert rolled[3] < hover[3]


def test_allocation_matches_wrench():
    params = Params()
    matrix = allocation_matrix(params)
    controller = Controller(params)
    wrench = np.array([params.hover_thrust_n, 0.15, -0.1, 0.02])
    command, _ = controller.mix(*_split(wrench))
    assert np.allclose(matrix @ command, wrench, atol=1e-6)


def test_huge_thrust_saturates_every_rotor():
    params = Params()
    controller = Controller(params)
    command, saturated = controller.mix(1000.0, np.zeros(3))
    assert saturated
    assert np.allclose(command, params.thrust_max_n)


def _split(wrench):
    return float(wrench[0]), wrench[1:]
