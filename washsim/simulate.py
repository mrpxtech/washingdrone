"""Advance one washing episode with rotor forces and MuJoCo.

Direct pose writes happen only in initialize(). After that, mj_step integrates
thrust at the rotor sites, reaction torques, gravity, and an optional wind force.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from washsim.controller import SPIN, Controller
from washsim.coverage import CoverageMap
from washsim.metrics import summarize
from washsim.mission import Mission, build_mission, initial_position
from washsim.model import body_id, load_model, read_state, site_id
from washsim.params import Params

ROTOR_NAMES = ("rotor_0", "rotor_1", "rotor_2", "rotor_3")


def initialize(model: mujoco.MjModel, data: mujoco.MjData, position: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.qpos[0:3] = position
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)


def _apply_rotors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bid: int,
    rotation: np.ndarray,
    thrusts: np.ndarray,
    params: Params,
) -> None:
    z_axis = rotation[:, 2]
    for index, name in enumerate(ROTOR_NAMES):
        force = thrusts[index] * z_axis
        torque = (-SPIN[index] * params.kappa_m * thrusts[index]) * z_axis
        point = data.site_xpos[site_id(model, name)]
        mujoco.mj_applyFT(model, data, force, torque, point, bid, data.qfrc_applied)


def _apply_force_at_com(model: mujoco.MjModel, data: mujoco.MjData, bid: int, force: np.ndarray) -> None:
    if np.linalg.norm(force) == 0.0:
        return
    mujoco.mj_applyFT(model, data, force, np.zeros(3), data.xipos[bid], bid, data.qfrc_applied)


def wind_force(time_s: float, mission: Mission, params: Params, scenario: str) -> np.ndarray:
    if scenario != "wind":
        return np.zeros(3)
    start = mission.scan_start_s + params.wind_start_offset_s
    end = start + params.wind_duration_s
    if time_s < start or time_s > end:
        return np.zeros(3)
    direction = np.array(params.wind_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return direction * params.wind_force_n


def vehicle_clearance(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    closest = np.inf
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith("veh_"):
            continue
        kind = int(model.geom_type[gid])
        position = data.geom_xpos[gid]
        rotation = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for sz in (-1.0, 1.0):
                        corner = position + rotation @ (np.array([sx, sy, sz]) * size)
                        closest = min(closest, float(corner[1]))
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            closest = min(closest, float(position[1] - size[0]))
        elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            axis = rotation[:, 2]
            end_a = position + axis * size[1]
            end_b = position - axis * size[1]
            closest = min(closest, float(end_a[1] - size[0]), float(end_b[1] - size[0]))
    return float(closest)


def glass_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    glass = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "glass")
    for index in range(data.ncon):
        geom1 = int(data.contact[index].geom1)
        geom2 = int(data.contact[index].geom2)
        if glass not in (geom1, geom2):
            continue
        other = geom2 if geom1 == glass else geom1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
        if name.startswith("veh_"):
            return True
    return False


def _tilt(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))


def run_episode(scenario: str, run_dir: Path, params: Params | None = None, run_id: str | None = None) -> dict:
    params = params or Params()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(params.seed)
    model, data = load_model(params)
    bid = body_id(model)
    mission = build_mission(params)
    controller = Controller(params)
    coverage = CoverageMap(params)
    position0 = initial_position(params)
    initialize(model, data, position0)
    controller.reset()

    columns = {
        "time_s": [],
        "position": [],
        "ref_position": [],
        "velocity": [],
        "ref_velocity": [],
        "quaternion": [],
        "omega_body": [],
        "thrust_n": [],
        "saturated": [],
        "nozzle_position": [],
        "nozzle_distance_m": [],
        "clearance_m": [],
        "coverage": [],
        "spraying": [],
        "contact": [],
        "wind_force_n": [],
        "phase": [],
        "tilt_rad": [],
    }
    failed = False
    fail_reason = ""
    steps = int(np.ceil(mission.duration / params.timestep_s))
    for step in range(steps):
        time_s = step * params.timestep_s
        reference = mission.sample(time_s)
        position, rotation, velocity, omega = read_state(model, data, bid)
        thrusts, saturated = controller.step(
            position,
            velocity,
            rotation,
            omega,
            reference.position,
            reference.velocity,
            reference.acceleration,
            reference.yaw,
            params.timestep_s,
        )
        data.qfrc_applied[:] = 0.0
        _apply_rotors(model, data, bid, rotation, thrusts, params)
        wind = wind_force(time_s, mission, params, scenario)
        _apply_force_at_com(model, data, bid, wind)
        mujoco.mj_step(model, data)

        position, rotation, velocity, omega = read_state(model, data, bid)
        nozzle = data.site_xpos[site_id(model, "nozzle")].copy()
        axis = -rotation[:, 1]
        clearance = vehicle_clearance(model, data)
        contacting = glass_contact(model, data) or clearance <= 0.0
        coverage.accumulate(nozzle, axis, reference.spraying)
        tilt = _tilt(rotation)
        sample_time = (step + 1) * params.timestep_s
        columns["time_s"].append(sample_time)
        columns["position"].append(position.copy())
        columns["ref_position"].append(reference.position.copy())
        columns["velocity"].append(velocity.copy())
        columns["ref_velocity"].append(reference.velocity.copy())
        columns["quaternion"].append(data.qpos[3:7].copy())
        columns["omega_body"].append(omega.copy())
        columns["thrust_n"].append(thrusts.copy())
        columns["saturated"].append(1.0 if saturated else 0.0)
        columns["nozzle_position"].append(nozzle.copy())
        columns["nozzle_distance_m"].append(float(nozzle[1]))
        columns["clearance_m"].append(clearance)
        columns["coverage"].append(coverage.fraction)
        columns["spraying"].append(1.0 if reference.spraying else 0.0)
        columns["contact"].append(1.0 if contacting else 0.0)
        columns["wind_force_n"].append(wind.copy())
        columns["phase"].append(reference.phase)
        columns["tilt_rad"].append(tilt)

        if not np.isfinite(position).all() or not np.isfinite(data.qvel).all():
            failed = True
            fail_reason = "non-finite state"
            break
        if position[2] < 0.25:
            failed = True
            fail_reason = "altitude below 0.25 m"
            break
        if tilt > np.deg2rad(75.0):
            failed = True
            fail_reason = "tilt above 75 deg"
            break

    rows = {key: np.asarray(value) for key, value in columns.items()}
    wind_end = None
    if scenario == "wind":
        wind_end = mission.scan_start_s + params.wind_start_offset_s + params.wind_duration_s
    metrics = summarize(rows, params.nozzle_standoff_m, wind_end, failed, fail_reason)
    metrics.update(
        {
            "scenario": scenario,
            "run_id": run_id or run_dir.name,
            "seed": params.seed,
            "timestep_s": params.timestep_s,
            "scan_start_s": mission.scan_start_s,
            "scan_end_s": mission.scan_end_s,
            "wind_start_s": None if scenario != "wind" else mission.scan_start_s + params.wind_start_offset_s,
            "wind_end_s": wind_end,
            "wind_force_n": params.wind_force_n if scenario == "wind" else 0.0,
            "wind_direction": list(params.wind_direction),
        }
    )
    _write_csv(run_dir / "telemetry.csv", rows)
    np.savez_compressed(run_dir / "telemetry.npz", **rows)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(run_dir / "coverage_grid.npy", coverage.grid)
    (run_dir / "params.json").write_text(json.dumps(params.to_dict(), indent=2))
    return {"metrics": metrics, "rows": rows, "coverage": coverage, "mission": mission, "model": model, "params": params}


def load_rows(run_dir: Path) -> dict[str, np.ndarray]:
    stored = np.load(Path(run_dir) / "telemetry.npz")
    return {key: stored[key] for key in stored.files}


def _write_csv(path: Path, rows: dict[str, np.ndarray]) -> None:
    count = len(rows["time_s"])
    header = [
        "time_s",
        "ref_x_m", "ref_y_m", "ref_z_m",
        "ref_vx_mps", "ref_vy_mps", "ref_vz_mps",
        "x_m", "y_m", "z_m",
        "qw", "qx", "qy", "qz",
        "vx_mps", "vy_mps", "vz_mps",
        "wx_radps", "wy_radps", "wz_radps",
        "nozzle_x_m", "nozzle_y_m", "nozzle_z_m",
        "nozzle_distance_m",
        "clearance_m",
        "coverage",
        "spraying",
        "contact",
        "thrust_0_n", "thrust_1_n", "thrust_2_n", "thrust_3_n",
        "saturated",
        "wind_fx_n", "wind_fy_n", "wind_fz_n",
        "tilt_rad",
        "phase",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(count):
            position = rows["position"][index]
            reference = rows["ref_position"][index]
            velocity = rows["velocity"][index]
            ref_velocity = rows["ref_velocity"][index]
            quat = rows["quaternion"][index]
            omega = rows["omega_body"][index]
            nozzle = rows["nozzle_position"][index]
            thrust = rows["thrust_n"][index]
            wind = rows["wind_force_n"][index]
            writer.writerow(
                [
                    f"{rows['time_s'][index]:.4f}",
                    *[f"{value:.6f}" for value in reference],
                    *[f"{value:.6f}" for value in ref_velocity],
                    *[f"{value:.6f}" for value in position],
                    *[f"{value:.6f}" for value in quat],
                    *[f"{value:.6f}" for value in velocity],
                    *[f"{value:.6f}" for value in omega],
                    *[f"{value:.6f}" for value in nozzle],
                    f"{rows['nozzle_distance_m'][index]:.6f}",
                    f"{rows['clearance_m'][index]:.6f}",
                    f"{rows['coverage'][index]:.6f}",
                    int(rows["spraying"][index]),
                    int(rows["contact"][index]),
                    *[f"{value:.6f}" for value in thrust],
                    int(rows["saturated"][index]),
                    *[f"{value:.6f}" for value in wind],
                    f"{rows['tilt_rad'][index]:.6f}",
                    rows["phase"][index],
                ]
            )
