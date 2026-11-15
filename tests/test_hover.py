import os

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

from washsim.controller import Controller
from washsim.model import body_id, load_model, read_state
from washsim.params import Params
from washsim.simulate import _apply_rotors, initialize


def _fly(reference, seconds, start):
    params = Params()
    model, data = load_model(params)
    bid = body_id(model)
    initialize(model, data, start)
    controller = Controller(params)
    zeros = np.zeros(3)
    steps = int(seconds / params.timestep_s)
    position = start
    tilt = 0.0
    for _ in range(steps):
        position, rotation, velocity, omega = read_state(model, data, bid)
        thrusts, _ = controller.step(
            position, velocity, rotation, omega, reference, zeros, zeros, 0.0, params.timestep_s
        )
        data.qfrc_applied[:] = 0.0
        _apply_rotors(model, data, bid, rotation, thrusts, params)
        import mujoco

        mujoco.mj_step(model, data)
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
    position, rotation, _, _ = read_state(model, data, bid)
    return position, tilt


def test_hover_holds_position_and_attitude():
    params = Params()
    start = np.array([0.0, params.body_standoff_m, 1.9])
    position, _ = _fly(start, 4.0, start)
    assert np.linalg.norm(position - start) < 0.05
    assert position[2] > 1.7


def test_same_hover_is_deterministic():
    params = Params()
    start = np.array([0.2, params.body_standoff_m, 1.7])
    first, _ = _fly(start, 1.5, start)
    second, _ = _fly(start, 1.5, start)
    assert np.allclose(first, second, atol=1e-9)


def test_lateral_step_moves_the_right_way_without_falling():
    params = Params()
    start = np.array([0.0, params.body_standoff_m, 1.9])
    reference = start + np.array([0.4, 0.0, 0.0])
    position, tilt = _fly(reference, 3.0, start)
    assert position[0] > start[0] + 0.15
    assert abs(position[2] - start[2]) < 0.2
    assert tilt < np.deg2rad(35)
