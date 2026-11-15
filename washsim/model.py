"""Original MuJoCo geometry for one quadrotor and one window.

Visual tiles on the glass are not collision shapes. Collision geometry is the
glass pane plus the vehicle parts named veh_*.
"""

from __future__ import annotations

import mujoco
import numpy as np

from washsim.params import Params


def build_xml(params: Params) -> str:
    x0, x1, z0, z1 = params.panel_bounds()
    cx = 0.5 * (x0 + x1)
    cz = 0.5 * (z0 + z1)
    half_w = 0.5 * params.panel_width_m
    half_h = 0.5 * params.panel_height_m
    tiles = params.visual_tiles
    tile_w = params.panel_width_m / tiles
    tile_h = params.panel_height_m / tiles
    tile_xml = []
    for row in range(tiles):
        for col in range(tiles):
            px = x0 + (col + 0.5) * tile_w
            pz = z0 + (row + 0.5) * tile_h
            tile_xml.append(
                f'<geom name="tile_{row}_{col}" type="box" '
                f'pos="{px:.4f} 0.012 {pz:.4f}" '
                f'size="{0.46 * tile_w:.4f} 0.0015 {0.46 * tile_h:.4f}" '
                f'rgba="0.78 0.88 0.93 0.55" contype="0" conaffinity="0"/>'
            )
    tiles_joined = "\n      ".join(tile_xml)
    inertia = " ".join(f"{value:.5f}" for value in params.inertia_kgm2)
    length = params.arm_m
    return f"""
<mujoco model="window_wash">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{params.timestep_s}" gravity="0 0 -{params.gravity_mps2}"
          integrator="implicitfast" density="0" viscosity="0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.38 0.38 0.38" ambient="0.48 0.50 0.52" specular="0.12 0.12 0.12"/>
    <rgba haze="0.93 0.94 0.96 1"/>
    <map znear="0.05" zfar="40"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.93 0.94 0.95" rgb2="0.84 0.86 0.88"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <default>
    <geom mass="0" contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light directional="true" pos="0 2 6" dir="0 -0.15 -1" diffuse="0.85 0.85 0.88"/>
    <light directional="true" pos="-2 1 4" dir="0.4 0.1 -1" diffuse="0.35 0.40 0.48"/>
    <camera name="overview" mode="targetbody" target="window_anchor" pos="-2.35 3.70 3.15" fovy="50"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="grid" contype="2" conaffinity="1"/>
    <geom name="facade" type="box" pos="{cx:.3f} -0.09 {cz:.3f}" size="{half_w + 0.28:.3f} 0.025 {half_h + 0.38:.3f}"
          rgba="0.87 0.88 0.90 1"/>
    <geom name="frame_bottom" type="box" pos="{cx:.3f} 0.0 {z0 - 0.045:.3f}" size="{half_w + 0.06:.3f} 0.025 0.035"
          rgba="0.28 0.30 0.33 1"/>
    <geom name="frame_top" type="box" pos="{cx:.3f} 0.0 {z1 + 0.045:.3f}" size="{half_w + 0.06:.3f} 0.025 0.035"
          rgba="0.28 0.30 0.33 1"/>
    <geom name="frame_left" type="box" pos="{x0 - 0.045:.3f} 0.0 {cz:.3f}" size="0.035 0.025 {half_h + 0.08:.3f}"
          rgba="0.28 0.30 0.33 1"/>
    <geom name="frame_right" type="box" pos="{x1 + 0.045:.3f} 0.0 {cz:.3f}" size="0.035 0.025 {half_h + 0.08:.3f}"
          rgba="0.28 0.30 0.33 1"/>
    <geom name="glass" type="box" pos="{cx:.3f} 0.0 {cz:.3f}" size="{half_w:.3f} 0.008 {half_h:.3f}"
          rgba="0.72 0.86 0.93 0.28" contype="2" conaffinity="1"/>
    <body name="window_anchor" pos="{cx:.3f} 0.0 {cz:.3f}"/>
    <site name="standoff_mark" pos="0 {params.nozzle_standoff_m:.3f} 0.02" size="0.02" rgba="0.15 0.55 0.60 1"/>
      {tiles_joined}
    <body name="footprint" mocap="true" pos="0 0.02 {cz:.3f}">
      <geom name="footprint_vis" type="ellipsoid" size="0.20 0.004 0.20" rgba="0.12 0.62 0.66 0.35"/>
    </body>
    <body name="quadrotor" pos="0 {params.body_standoff_m:.3f} {cz:.3f}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{params.mass_kg}" diaginertia="{inertia}"/>
      <geom name="veh_body" type="box" size="0.09 0.06 0.025" rgba="0.90 0.46 0.16 1" contype="1" conaffinity="2"/>
      <geom name="arm_fx" type="capsule" fromto="{length:.3f} 0 0.01 {-length:.3f} 0 0.01" size="0.012"
            rgba="0.22 0.24 0.27 1"/>
      <geom name="arm_fy" type="capsule" fromto="0 {length:.3f} 0.01 0 {-length:.3f} 0.01" size="0.012"
            rgba="0.22 0.24 0.27 1"/>
      <geom name="rotor_vis_0" type="cylinder" pos="{length:.3f} 0 0.028" size="0.075 0.004" rgba="0.16 0.17 0.19 0.9"/>
      <geom name="rotor_vis_1" type="cylinder" pos="0 {length:.3f} 0.028" size="0.075 0.004" rgba="0.16 0.17 0.19 0.9"/>
      <geom name="rotor_vis_2" type="cylinder" pos="{-length:.3f} 0 0.028" size="0.075 0.004" rgba="0.16 0.17 0.19 0.9"/>
      <geom name="rotor_vis_3" type="cylinder" pos="0 {-length:.3f} 0.028" size="0.075 0.004" rgba="0.16 0.17 0.19 0.9"/>
      <geom name="veh_rotor_0" type="sphere" pos="{length:.3f} 0 0.028" size="0.04" rgba="0 0 0 0" contype="1" conaffinity="2"/>
      <geom name="veh_rotor_1" type="sphere" pos="0 {length:.3f} 0.028" size="0.04" rgba="0 0 0 0" contype="1" conaffinity="2"/>
      <geom name="veh_rotor_2" type="sphere" pos="{-length:.3f} 0 0.028" size="0.04" rgba="0 0 0 0" contype="1" conaffinity="2"/>
      <geom name="veh_rotor_3" type="sphere" pos="0 {-length:.3f} 0.028" size="0.04" rgba="0 0 0 0" contype="1" conaffinity="2"/>
      <geom name="veh_skid_l" type="capsule" fromto="-0.10 0.045 -0.055 0.10 0.045 -0.055" size="0.01"
            rgba="0.18 0.19 0.21 1" contype="1" conaffinity="2"/>
      <geom name="veh_skid_r" type="capsule" fromto="-0.10 -0.045 -0.055 0.10 -0.045 -0.055" size="0.01"
            rgba="0.18 0.19 0.21 1" contype="1" conaffinity="2"/>
      <geom name="veh_nozzle" type="capsule" fromto="0 -0.04 -0.01 0 {-params.nozzle_length_m:.3f} -0.01" size="0.012"
            rgba="0.10 0.66 0.70 1" contype="1" conaffinity="2"/>
      <geom name="nozzle_vis" type="capsule" fromto="0 -0.02 0.0 0 {-params.nozzle_length_m - 0.03:.3f} 0.0" size="0.018"
            rgba="0.05 0.78 0.82 1"/>
      <site name="rotor_0" pos="{length:.3f} 0 0.028" size="0.01"/>
      <site name="rotor_1" pos="0 {length:.3f} 0.028" size="0.01"/>
      <site name="rotor_2" pos="{-length:.3f} 0 0.028" size="0.01"/>
      <site name="rotor_3" pos="0 {-length:.3f} 0.028" size="0.01"/>
      <site name="nozzle" pos="0 {-params.nozzle_length_m:.3f} -0.01" size="0.012" rgba="0.05 0.8 0.85 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def load_model(params: Params) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(build_xml(params))
    data = mujoco.MjData(model)
    return model, data


def body_id(model: mujoco.MjModel, name: str = "quadrotor") -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def site_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, position: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = position
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)


def read_state(model: mujoco.MjModel, data: mujoco.MjData, bid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = data.xmat[bid].reshape(3, 3).copy()
    position = data.xpos[bid].copy()
    spatial = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, spatial, 0)
    omega_world = spatial[:3]
    linear = spatial[3:].copy()
    omega_body = rotation.T @ omega_world
    return position, rotation, linear, omega_body


def paint_coverage(model: mujoco.MjModel, grid: np.ndarray, tiles: int) -> None:
    """Color the coarse glass tiles from the fine coverage grid."""
    rows, cols = grid.shape
    for row in range(tiles):
        for col in range(tiles):
            r0 = int(row * rows / tiles)
            r1 = int((row + 1) * rows / tiles)
            c0 = int(col * cols / tiles)
            c1 = int((col + 1) * cols / tiles)
            fraction = float(grid[r0:r1, c0:c1].mean()) if r1 > r0 and c1 > c0 else 0.0
            gid = geom_id(model, f"tile_{row}_{col}")
            if fraction <= 0.0:
                model.geom_rgba[gid] = (0.78, 0.88, 0.93, 0.55)
            else:
                model.geom_rgba[gid] = (
                    0.78 * (1.0 - fraction) + 0.10 * fraction,
                    0.88 * (1.0 - fraction) + 0.62 * fraction,
                    0.93 * (1.0 - fraction) + 0.66 * fraction,
                    0.55 + 0.40 * fraction,
                )


def place_footprint(data: mujoco.MjData, center: np.ndarray | None) -> None:
    if center is None:
        data.mocap_pos[0] = (0.0, 0.02, -2.0)
        return
    data.mocap_pos[0] = (center[0], 0.018, center[2])
