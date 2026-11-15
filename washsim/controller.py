"""Cascaded position and attitude controller with rotor allocation.

Position loop, in the world frame:
    a = Kp e + Kd e_dot + Ki integral(e) + a_ref
    thrust direction = normalize(a + (0, 0, g))
Horizontal and vertical accelerations are limited. The integrator freezes and
decays while any rotor command is saturated.

Attitude loop, in the body frame (Lee et al. geometric tracking, yaw rate 0):
    e_R = 0.5 * vee(R_des^T R - R^T R_des)
    tau = -Kr e_R - Kw omega
Desired yaw is zero, so body -Y keeps facing the glass when the vehicle is level.

Allocation uses the same geometry as the force application in simulate.py:
    T     = sum Ti
    tau_x = L (T1 - T3)
    tau_y = L (T2 - T0)
    tau_z = -kappa (s0 T0 + s1 T1 + s2 T2 + s3 T3)
"""

from __future__ import annotations

import numpy as np

from washsim.params import Params

SPIN = np.array([1.0, -1.0, 1.0, -1.0])


def allocation_matrix(params: Params) -> np.ndarray:
    length = params.arm_m
    kappa = params.kappa_m
    signs = SPIN
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, length, 0.0, -length],
            [-length, 0.0, length, 0.0],
            -kappa * signs,
        ],
        dtype=float,
    )


def vee(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=float,
    )


def desired_rotation(thrust_direction: np.ndarray, yaw: float) -> np.ndarray:
    z_body = thrust_direction / np.linalg.norm(thrust_direction)
    x_hint = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_body = np.cross(z_body, x_hint)
    norm = np.linalg.norm(y_body)
    if norm < 1e-6:
        x_hint = np.array([0.0, 1.0, 0.0])
        y_body = np.cross(z_body, x_hint)
        norm = np.linalg.norm(y_body)
    y_body = y_body / norm
    x_body = np.cross(y_body, z_body)
    return np.column_stack((x_body, y_body, z_body))


class Controller:
    def __init__(self, params: Params):
        self.params = params
        self.alloc_inv = np.linalg.inv(allocation_matrix(params))
        self.integral = np.zeros(3)

    def reset(self) -> None:
        self.integral[:] = 0.0

    def mix(self, thrust_n: float, torque_nm: np.ndarray) -> tuple[np.ndarray, bool]:
        wrench = np.array([thrust_n, torque_nm[0], torque_nm[1], torque_nm[2]], dtype=float)
        command = self.alloc_inv @ wrench
        saturated = bool(np.any(command < self.params.thrust_min_n - 1e-9) or np.any(command > self.params.thrust_max_n + 1e-9))
        command = np.clip(command, self.params.thrust_min_n, self.params.thrust_max_n)
        return command, saturated

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        rotation: np.ndarray,
        omega_body: np.ndarray,
        ref_position: np.ndarray,
        ref_velocity: np.ndarray,
        ref_acceleration: np.ndarray,
        yaw: float,
        dt: float,
    ) -> tuple[np.ndarray, bool]:
        params = self.params
        error = ref_position - position
        velocity_error = ref_velocity - velocity
        kp = np.array(params.kp)
        kd = np.array(params.kd)
        ki = np.array(params.ki)
        accel = kp * error + kd * velocity_error + ki * self.integral + ref_acceleration
        accel[0] = float(np.clip(accel[0], -params.a_xy_max_mps2, params.a_xy_max_mps2))
        accel[1] = float(np.clip(accel[1], -params.a_xy_max_mps2, params.a_xy_max_mps2))
        accel[2] = float(np.clip(accel[2], -params.a_z_max_mps2, params.a_z_max_mps2))

        specific = accel + np.array([0.0, 0.0, params.gravity_mps2])
        thrust = float(params.mass_kg * np.linalg.norm(specific))
        thrust = float(np.clip(thrust, 0.0, 4.0 * params.thrust_max_n))
        direction = specific / max(np.linalg.norm(specific), 1e-8)
        rotation_des = desired_rotation(direction, yaw)
        attitude_error = 0.5 * vee(rotation_des.T @ rotation - rotation.T @ rotation_des)
        torque = -np.array(params.kr) * attitude_error - np.array(params.kw) * omega_body
        command, saturated = self.mix(thrust, torque)
        if saturated:
            self.integral *= 0.98
        else:
            self.integral += error * dt
            limit = np.array(params.i_limit)
            self.integral = np.clip(self.integral, -limit, limit)
        return command, saturated
