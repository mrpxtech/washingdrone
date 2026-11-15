"""Plots, dashboard replay, and the portable bundle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from washsim.params import Params


def save_plots(run_dir: Path, rows: dict[str, np.ndarray], params: Params) -> None:
    time_s = rows["time_s"]
    error = np.linalg.norm(rows["position"] - rows["ref_position"], axis=1)
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(time_s, error, color="#0f6f78", lw=1.4, label="Position error")
    axes[0].axhline(0.15, color="#b45309", ls="--", lw=1, label="0.15 m target")
    axes[0].set_ylabel("m")
    axes[0].set_title("Position tracking error")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(time_s, rows["nozzle_distance_m"], color="#0f6f78", lw=1.4, label="Nozzle distance")
    axes[1].axhline(params.nozzle_standoff_m, color="#b45309", ls="--", lw=1, label="0.8 m target")
    axes[1].set_ylabel("m")
    axes[1].set_title("Nozzle-to-glass distance")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(time_s, rows["clearance_m"], color="#0f6f78", lw=1.4, label="Clearance")
    axes[2].axhline(params.min_clearance_m, color="#b45309", ls="--", lw=1, label="0.30 m minimum")
    axes[2].set_ylabel("m")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Closest vehicle collision geometry to the glass")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, color="#e2e8ee", lw=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(run_dir / "tracking.png", dpi=120)
    plt.close(figure)

    grid = np.load(run_dir / "coverage_grid.npy")
    heat, heat_axis = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    image = heat_axis.imshow(
        grid,
        origin="lower",
        cmap="GnBu",
        vmin=0,
        vmax=1,
        extent=[-0.9, 0.9, params.panel_z0_m, params.panel_z0_m + params.panel_height_m],
        interpolation="nearest",
    )
    heat_axis.set_xlabel("Window x (m)")
    heat_axis.set_ylabel("Window z (m)")
    heat_axis.set_title("Estimated spray coverage")
    heat_axis.set_aspect("equal")
    heat.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04)
    heat.savefig(run_dir / "coverage.png", dpi=120)
    plt.close(heat)


def save_replay(run_dir: Path, rows: dict[str, np.ndarray], metrics: dict, params: Params, rate_hz: float = 20.0) -> None:
    stride = max(1, int(round((1.0 / rate_hz) / params.timestep_s)))
    frames = []
    for index in range(0, len(rows["time_s"]), stride):
        frames.append(
            {
                "t": round(float(rows["time_s"][index]), 4),
                "p": [round(float(value), 4) for value in rows["position"][index]],
                "q": [round(float(value), 5) for value in rows["quaternion"][index]],
                "ref": [round(float(value), 4) for value in rows["ref_position"][index]],
                "nozzle": [round(float(value), 4) for value in rows["nozzle_position"][index]],
                "coverage": round(float(rows["coverage"][index]), 4),
                "clearance": round(float(rows["clearance_m"][index]), 4),
                "spraying": int(rows["spraying"][index]),
                "phase": str(rows["phase"][index]),
                "wind": [round(float(value), 3) for value in rows["wind_force_n"][index]],
                "tilt_deg": round(float(np.degrees(rows["tilt_rad"][index])), 2),
            }
        )
    grid = np.load(run_dir / "coverage_grid.npy")
    payload = {
        "run_id": metrics["run_id"],
        "scenario": metrics["scenario"],
        "label": "Trajectory replay of recorded MuJoCo states",
        "dt_s": round(stride * params.timestep_s, 4),
        "frames": frames,
        "coverage_grid": grid.astype(int).tolist(),
        "metrics": metrics,
        "panel": {
            "width_m": params.panel_width_m,
            "height_m": params.panel_height_m,
            "z0_m": params.panel_z0_m,
            "standoff_m": params.nozzle_standoff_m,
        },
    }
    (run_dir / "replay.json").write_text(json.dumps(payload))


def write_manifest(run_dir: Path, metrics: dict, params: Params, extra: dict | None = None) -> None:
    payload = {
        "run_id": metrics["run_id"],
        "scenario": metrics["scenario"],
        "seed": params.seed,
        "timestep_s": params.timestep_s,
        "mujoco_gl": "osmesa",
        "metrics": metrics,
        "parameters": params.to_dict(),
    }
    if extra:
        payload.update(extra)
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2))


def build_bundle(root: Path, run_dirs: list[Path]) -> Path:
    bundle = root / "outputs" / "replay-bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    static = bundle / "static"
    shutil.copytree(root / "dashboard" / "static", static)
    data = bundle / "data"
    data.mkdir(parents=True)
    listing = []
    for run_dir in run_dirs:
        replay = json.loads((run_dir / "replay.json").read_text())
        target = data / f"{replay['run_id']}.json"
        shutil.copyfile(run_dir / "replay.json", target)
        listing.append({"run_id": replay["run_id"], "scenario": replay["scenario"], "file": target.name})
    (data / "index.json").write_text(json.dumps(listing, indent=2))
    (bundle / "SERVE.txt").write_text(
        "\n".join(
            [
                "Portable trajectory replay. This bundle does not run physics.",
                "From this directory:",
                "  python3 -m http.server 8765 --bind 127.0.0.1",
                "Open http://127.0.0.1:8765/static/replay.html",
                "On the VM, bind the Tailscale address instead of 127.0.0.1 if a phone on the tailnet should open it.",
                "",
            ]
        )
    )
    return bundle
