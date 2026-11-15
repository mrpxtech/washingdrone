"""Overlapping horizontal passes, with a settle and a retreat.

Moves use rest-to-rest quintic polynomials. Duration is long enough that the
peak speed stays at or below speed_limit_mps and the peak acceleration stays
at or below accel_limit_mps2. The quintic peak-speed factor is 15/8 = 1.875,
and the peak of d²s/dτ² is about 5.77.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from washsim.params import Params


@dataclass
class Reference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float
    spraying: bool
    phase: str


@dataclass
class Segment:
    p0: np.ndarray
    p1: np.ndarray
    duration: float
    phase: str
    spraying: bool


def move_duration(distance: float, params: Params) -> float:
    if distance < 1e-6:
        return 0.4
    speed_time = 1.875 * distance / params.speed_limit_mps
    accel_time = (5.77 * distance / params.accel_limit_mps2) ** 0.5
    return max(speed_time, accel_time, 0.4)


def _sample_quintic(segment: Segment, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = float(np.clip(tau, 0.0, 1.0))
    position_blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    velocity_blend = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    accel_blend = 60 * tau - 180 * tau**2 + 120 * tau**3
    delta = segment.p1 - segment.p0
    duration = segment.duration
    position = segment.p0 + delta * position_blend
    velocity = delta * (velocity_blend / duration)
    acceleration = delta * (accel_blend / duration**2)
    return position, velocity, acceleration


class Mission:
    def __init__(self, segments: list[Segment], scan_start_s: float, scan_end_s: float):
        self.segments = segments
        self.scan_start_s = scan_start_s
        self.scan_end_s = scan_end_s
        self.duration = float(sum(segment.duration for segment in segments))
        self._ends = np.cumsum([segment.duration for segment in segments])

    def sample(self, time_s: float) -> Reference:
        if time_s >= self.duration:
            segment = self.segments[-1]
            position, velocity, acceleration = _sample_quintic(segment, 1.0)
            velocity[:] = 0.0
            acceleration[:] = 0.0
            return Reference(position, velocity, acceleration, 0.0, False, segment.phase)
        index = int(np.searchsorted(self._ends, time_s, side="right"))
        previous = 0.0 if index == 0 else float(self._ends[index - 1])
        segment = self.segments[index]
        tau = (time_s - previous) / segment.duration
        position, velocity, acceleration = _sample_quintic(segment, tau)
        return Reference(position, velocity, acceleration, 0.0, segment.spraying, segment.phase)


def _pass_heights(params: Params) -> np.ndarray:
    radius = params.footprint_radius_m
    margin = 0.85 * radius
    z0 = params.panel_z0_m + margin
    z1 = params.panel_z0_m + params.panel_height_m - margin
    span = max(z1 - z0, 1e-6)
    intervals = int(np.ceil(span / params.pass_spacing_m))
    return np.linspace(z0, z1, intervals + 1)


def build_mission(params: Params) -> Mission:
    x0, x1, _, _ = params.panel_bounds()
    heights = _pass_heights(params)
    body_y = params.body_standoff_m
    start = np.array([x0, body_y + 0.15, heights[0] + 0.20])
    hover = np.array([x0, body_y, heights[0]])
    segments: list[Segment] = []

    def add_move(p0: np.ndarray, p1: np.ndarray, phase: str, spraying: bool) -> None:
        distance = float(np.linalg.norm(p1 - p0))
        segments.append(Segment(p0.copy(), p1.copy(), move_duration(distance, params), phase, spraying))

    def add_dwell(point: np.ndarray, duration: float, phase: str) -> None:
        segments.append(Segment(point.copy(), point.copy(), duration, phase, False))

    add_move(start, hover, "settle", False)
    add_dwell(hover, params.dwell_s, "settle")
    scan_start = float(sum(segment.duration for segment in segments))

    point = hover.copy()
    for index, height in enumerate(heights):
        going_right = index % 2 == 0
        begin_x = x0 if going_right else x1
        end_x = x1 if going_right else x0
        line_start = np.array([begin_x, body_y, height])
        line_end = np.array([end_x, body_y, height])
        if np.linalg.norm(line_start - point) > 1e-6:
            add_move(point, line_start, "scan", True)
        add_move(line_start, line_end, "scan", True)
        point = line_end

    scan_end = float(sum(segment.duration for segment in segments))
    retreat = point + np.array([0.0, params.retreat_m, 0.0])
    add_move(point, retreat, "retreat", False)
    add_dwell(retreat, params.hold_s, "hold")
    return Mission(segments, scan_start, scan_end)


def initial_position(params: Params) -> np.ndarray:
    mission = build_mission(params)
    origin = mission.sample(0.0).position
    return origin + np.array(params.initial_offset_m, dtype=float)
