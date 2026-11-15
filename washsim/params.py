"""Physical parameters, frames, and documented simplifications.

World frame
    +X  along the window width, right when the glass is faced from the drone side
    +Y  normal to the glass, away from the facade. The glass plane is y = 0.
    +Z  up
    Units are meters, seconds, radians, and newtons.

Body frame (level, yaw = 0)
    Aligned with the world frame.
    The nozzle points along body -Y, toward the glass.
    Rotors spin about body +Z.

Rotor order and spin, viewed from above
    0 front (+X)  counterclockwise  spin sign +1
    1 left  (+Y)  clockwise         spin sign -1
    2 back  (-X)  counterclockwise  spin sign +1
    3 right (-Y)  clockwise         spin sign -1
    A positive spin sign is counterclockwise. The reaction torque on the body
    is opposite the rotor spin: tau_z = -kappa * spin_sign * thrust.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Params:
    """Illustrative 2 kg washing quadrotor and a 1.8 m square pane."""

    mass_kg: float = 2.0
    gravity_mps2: float = 9.81
    # Arm length is the horizontal distance from the center of mass to each rotor.
    arm_m: float = 0.16
    # Nozzle tip is this far from the center of mass along body -Y.
    nozzle_length_m: float = 0.20
    # Reaction torque per unit thrust. kappa = k_tau / k_f, in meters.
    kappa_m: float = 0.018
    thrust_min_n: float = 0.0
    thrust_max_n: float = 12.0
    # diaginertia (Ixx, Iyy, Izz) in kg m^2. See inertia_notes().
    inertia_kgm2: tuple[float, float, float] = (0.014, 0.016, 0.026)

    panel_width_m: float = 1.8
    panel_height_m: float = 1.8
    panel_z0_m: float = 1.05
    nozzle_standoff_m: float = 0.8
    min_clearance_m: float = 0.3

    # Spray cone half-angle. At 0.8 m a circular section has radius d * tan(alpha).
    half_angle_rad: float = math.radians(14.0)
    overlap: float = 0.30
    # Peak speed of a rest-to-rest quintic. Average speed on a long move is lower
    # by the factor 8/15, because the quintic velocity hump peaks at 1.875x the mean.
    speed_limit_mps: float = 0.70
    accel_limit_mps2: float = 1.2

    timestep_s: float = 0.002
    seed: int = 0

    # Coverage is accumulated only inside this normal-distance window.
    spray_range_min_m: float = 0.45
    spray_range_max_m: float = 1.15
    spray_slant_max_m: float = 1.35
    # Require the nozzle axis to point toward the glass (axis · normal < -this).
    spray_axis_alignment: float = 0.25

    grid_n: int = 90
    visual_tiles: int = 18

    wind_force_n: float = 3.5
    wind_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # Wind begins this long after the scanning phase starts.
    wind_start_offset_s: float = 8.0
    wind_duration_s: float = 4.0

    kp: tuple[float, float, float] = (4.8, 4.8, 6.5)
    kd: tuple[float, float, float] = (4.0, 4.0, 4.8)
    ki: tuple[float, float, float] = (1.4, 1.4, 1.8)
    i_limit: tuple[float, float, float] = (1.0, 1.0, 0.6)
    kr: tuple[float, float, float] = (2.4, 2.6, 0.40)
    kw: tuple[float, float, float] = (0.32, 0.36, 0.08)
    a_xy_max_mps2: float = 3.5
    a_z_max_mps2: float = 4.0

    dwell_s: float = 2.0
    hold_s: float = 2.5
    retreat_m: float = 0.75

    # Vehicle starts this far from the t = 0 reference so the hover settle is visible.
    initial_offset_m: tuple[float, float, float] = (0.0, 0.10, 0.08)

    @property
    def hover_thrust_n(self) -> float:
        return self.mass_kg * self.gravity_mps2

    @property
    def body_standoff_m(self) -> float:
        """Center-of-mass distance from the glass when the nozzle is at the target."""
        return self.nozzle_standoff_m + self.nozzle_length_m

    @property
    def footprint_radius_m(self) -> float:
        return self.nozzle_standoff_m * math.tan(self.half_angle_rad)

    @property
    def pass_spacing_m(self) -> float:
        """Center spacing from a circular footprint and the configured overlap."""
        diameter = 2.0 * self.footprint_radius_m
        return diameter * (1.0 - self.overlap)

    def panel_bounds(self) -> tuple[float, float, float, float]:
        half_w = 0.5 * self.panel_width_m
        return (-half_w, half_w, self.panel_z0_m, self.panel_z0_m + self.panel_height_m)

    def to_dict(self) -> dict:
        raw = asdict(self)
        raw["half_angle_deg"] = math.degrees(self.half_angle_rad)
        raw["footprint_radius_m"] = self.footprint_radius_m
        raw["pass_spacing_m"] = self.pass_spacing_m
        raw["body_standoff_m"] = self.body_standoff_m
        raw["hover_thrust_n"] = self.hover_thrust_n
        return raw


def inertia_notes() -> str:
    return (
        "Inertia is an estimate, not a measured vehicle. A 1.2 kg central body "
        "of 0.18 x 0.12 x 0.06 m contributes about (0.0018, 0.0036, 0.0047) kg m^2. "
        "Four 0.20 kg arm-and-motor masses at 0.16 m add about 0.010 kg m^2 to Ixx "
        "and Iyy and 0.020 kg m^2 to Izz. The values (0.014, 0.016, 0.026) kg m^2 "
        "round that sum up to leave room for the nozzle, skids, and rotor disks."
    )


def modeling_assumptions() -> list[str]:
    return [
        "The vehicle is one free rigid body. Rotor thrust and reaction torque are applied at four sites; rotor blades are not articulated.",
        "Gravity is MuJoCo's world gravity. No hose tension, spray momentum, or wall-suction force is applied.",
        "Air density and viscosity are zero, so there is no profile drag, rotor inflow, or wall-effect ground cushion. The wind case adds one external force at the center of mass.",
        "Water, droplets, and dirt are not simulated. Estimated spray coverage is the fraction of panel cells whose centers fall inside a geometric footprint.",
        "The footprint is the intersection of a circular cone with the glass plane, approximated as an ellipse. Cells are binary: reached or not. There is no intensity falloff.",
        "Coverage accumulates only while the mission enables spraying, the nozzle axis points at the glass, and the normal distance is inside the operating window.",
        "The controller reads the simulated state directly. There is no estimator, camera, or range-sensor noise.",
        "Contacts are unintentional. The mission never commands the vehicle to touch the glass.",
        "Pose writes are used only to initialize an episode and, later, to pose the recorded replay for pictures and video. The mission itself advances only by rotor forces and MuJoCo integration.",
    ]
