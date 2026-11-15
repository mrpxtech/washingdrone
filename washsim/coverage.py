"""Estimated spray coverage on the glass plane.

The nozzle is a circular cone of half-angle alpha. Let u be the unit nozzle
axis (body -Y in the world) and n = (0, 1, 0) the glass normal. If the axis
hits the plane y = 0 at slant range t:

    hit = p + t u
    cos(theta) = |u · n|
    minor radius b = t * tan(alpha)
    major radius a = b / cos(theta)

The major axis lies along the projection of u onto the glass. A cell is marked
when its center is inside that ellipse and inside the panel. This is a binary
geometric approximation: it ignores droplet size, a finite disk's perspective
beyond the cone, intensity falloff, and anything the glass does to the water.

Gating, all required:
    spraying is enabled by the mission phase
    u · n < -spray_axis_alignment (the nozzle points at the glass)
    slant range is positive and at most spray_slant_max_m
    normal distance p_y is inside [spray_range_min_m, spray_range_max_m]
"""

from __future__ import annotations

import numpy as np

from washsim.params import Params


class CoverageMap:
    def __init__(self, params: Params):
        self.params = params
        self.grid = np.zeros((params.grid_n, params.grid_n), dtype=np.uint8)
        x0, x1, z0, z1 = params.panel_bounds()
        self.x_centers = np.linspace(x0, x1, params.grid_n, endpoint=False)
        self.x_centers += 0.5 * (x1 - x0) / params.grid_n
        self.z_centers = np.linspace(z0, z1, params.grid_n, endpoint=False)
        self.z_centers += 0.5 * (z1 - z0) / params.grid_n
        self.xx, self.zz = np.meshgrid(self.x_centers, self.z_centers)

    @property
    def fraction(self) -> float:
        return float(self.grid.mean())

    def footprint(self, nozzle_position: np.ndarray, axis: np.ndarray) -> dict | None:
        params = self.params
        direction = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        direction = direction / norm
        alignment = float(direction[1])
        if alignment > -params.spray_axis_alignment:
            return None
        slant = -float(nozzle_position[1]) / alignment
        if slant <= 0.05 or slant > params.spray_slant_max_m:
            return None
        normal_distance = float(nozzle_position[1])
        if normal_distance < params.spray_range_min_m or normal_distance > params.spray_range_max_m:
            return None
        hit = nozzle_position + slant * direction
        cos_theta = min(1.0, abs(alignment))
        minor = slant * np.tan(params.half_angle_rad)
        major = minor / max(cos_theta, 0.25)
        tilt = np.array([direction[0], direction[2]])
        tilt_norm = np.linalg.norm(tilt)
        if tilt_norm < 1e-6:
            tilt_hat = np.array([1.0, 0.0])
        else:
            tilt_hat = tilt / tilt_norm
        return {
            "center": hit,
            "major_m": float(major),
            "minor_m": float(minor),
            "tilt": tilt_hat,
            "normal_distance_m": normal_distance,
            "slant_m": float(slant),
        }

    def accumulate(self, nozzle_position: np.ndarray, axis: np.ndarray, spraying: bool) -> dict | None:
        if not spraying:
            return None
        footprint = self.footprint(nozzle_position, axis)
        if footprint is None:
            return None
        delta_x = self.xx - footprint["center"][0]
        delta_z = self.zz - footprint["center"][2]
        tilt_x, tilt_z = footprint["tilt"]
        along = delta_x * tilt_x + delta_z * tilt_z
        across = -delta_x * tilt_z + delta_z * tilt_x
        inside = (along / footprint["major_m"]) ** 2 + (across / footprint["minor_m"]) ** 2 <= 1.0
        self.grid[inside] = 1
        return footprint
