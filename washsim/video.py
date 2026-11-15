"""Stream MuJoCo replay frames into ffmpeg.

Frames are rendered from recorded states. The physics episode is not re-integrated.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from washsim.coverage import CoverageMap
from washsim.model import load_model, paint_coverage, place_footprint
from washsim.params import Params


class ReplayRenderer:
    def __init__(self, model: mujoco.MjModel, width: int, height: int):
        self.model = model
        self.width = width
        self.height = height
        self.context = mujoco.GLContext(max(width, 64), max(height, 64))
        self.context.make_current()
        self.scene = mujoco.MjvScene(model, maxgeom=4000)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.mjr = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = camera_id

    def render(self, data: mujoco.MjData) -> np.ndarray:
        mujoco.mjv_updateScene(
            self.model,
            data,
            self.option,
            self.perturb,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        mujoco.mjr_render(self.viewport, self.scene, self.mjr)
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(image, None, self.viewport, self.mjr)
        return np.flipud(image)


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(20 if frame.shape[1] >= 1000 else 15)
    line_h = 24 if frame.shape[1] >= 1000 else 20
    height = 8 + line_h * len(lines) + 8
    draw.rectangle((0, 0, frame.shape[1], height), fill=(15, 23, 32, 220))
    for index, line in enumerate(lines):
        draw.text((12, 8 + line_h * index), line, fill=(248, 250, 252, 255), font=font)
    return np.asarray(image.convert("RGB"))


def _card(width: int, height: int, lines: list[str]) -> np.ndarray:
    image = Image.new("RGB", (width, height), (244, 246, 248))
    draw = ImageDraw.Draw(image)
    title = _font(28 if width >= 1000 else 20)
    body = _font(18 if width >= 1000 else 14)
    draw.text((36, 36), lines[0], fill=(15, 61, 66), font=title)
    y = 90
    for line in lines[1:]:
        draw.text((36, y), line, fill=(30, 41, 59), font=body)
        y += 32
    return np.asarray(image)


def _pose(model: mujoco.MjModel, data: mujoco.MjData, position: np.ndarray, quaternion: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = position
    data.qpos[3:7] = quaternion
    mujoco.mj_forward(model, data)


def _encoder(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _write(process: subprocess.Popen, frame: np.ndarray) -> None:
    process.stdin.write(frame.tobytes())


def render_episode_video(
    rows: dict,
    params: Params,
    destination: Path,
    width: int,
    height: int,
    fps: int,
    scenario_title: str,
    stills: dict[str, Path] | None = None,
) -> None:
    model, data = load_model(params)
    renderer = ReplayRenderer(model, width, height)
    coverage = CoverageMap(params)
    stride = max(1, int(round((1.0 / fps) / params.timestep_s)))
    process = _encoder(destination, width, height, fps)
    saved: set[str] = set()
    try:
        cursor = 0
        for index in range(0, len(rows["time_s"]), stride):
            while cursor <= index:
                if rows["spraying"][cursor]:
                    coverage.accumulate(rows["nozzle_position"][cursor], _axis_from_quat(rows["quaternion"][cursor]), True)
                cursor += 1
            paint_coverage(model, coverage.grid, params.visual_tiles)
            _pose(model, data, rows["position"][index], rows["quaternion"][index])
            place_footprint(data, rows["nozzle_position"][index] if rows["spraying"][index] else None)
            # mj_forward inside _pose, then footprint mocap needs another forward for the marker.
            mujoco.mj_forward(model, data)
            frame = renderer.render(data)
            phase = str(rows["phase"][index])
            wind = np.linalg.norm(rows["wind_force_n"][index])
            lines = [
                f"{scenario_title}    trajectory replay",
                f"t = {rows['time_s'][index]:5.1f} s    {phase}    estimated spray coverage {100 * rows['coverage'][index]:5.1f}%",
                f"clearance {rows['clearance_m'][index]:.2f} m    nozzle {rows['nozzle_distance_m'][index]:.2f} m"
                + (f"    wind {wind:.1f} N" if wind > 0.1 else ""),
            ]
            image = _annotate(frame, lines)
            _write(process, image)
            if stills:
                _maybe_still(stills, saved, phase, wind, image, float(rows["coverage"][index]))
        process.stdin.close()
        code = process.wait(timeout=120)
        if code != 0:
            error = process.stderr.read().decode("utf-8", "replace")[-2000:]
            raise RuntimeError(f"ffmpeg exited {code}: {error}")
    finally:
        if process.poll() is None:
            process.kill()


def _maybe_still(
    stills: dict[str, Path],
    saved: set[str],
    phase: str,
    wind: float,
    image: np.ndarray,
    coverage: float,
) -> None:
    stills["hover"].parent.mkdir(parents=True, exist_ok=True)
    if phase == "settle" and "hover" in stills:
        Image.fromarray(image).save(stills["hover"])
        saved.add("hover")
        return
    if phase == "retreat" and "retreat" in stills:
        Image.fromarray(image).save(stills["retreat"])
        saved.add("retreat")
        return
    key = None
    if phase == "scan" and wind > 0.1 and "disturbance" not in saved and coverage >= 0.25:
        key = "disturbance"
    elif phase == "scan" and wind <= 0.1 and "scan" not in saved and coverage >= 0.20:
        key = "scan"
    if key and key in stills:
        Image.fromarray(image).save(stills[key])
        saved.add(key)


def _axis_from_quat(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return -rotation[:, 1]


def write_demo_video(
    episodes: list[tuple[str, dict, dict]],
    params: Params,
    destination: Path,
    width: int = 640,
    height: int = 360,
    fps: int = 15,
    stills_dir: Path | None = None,
) -> None:
    """episodes is a list of (title, rows, metrics)."""
    process = _encoder(destination, width, height, fps)
    model, data = load_model(params)
    renderer = ReplayRenderer(model, width, height)
    try:
        for title, rows, metrics in episodes:
            for _ in range(fps):
                _write(
                    process,
                    _card(
                        width,
                        height,
                        [
                            title,
                            "Trajectory replay of recorded MuJoCo states",
                            f"Run {metrics['run_id']}",
                        ],
                    ),
                )
            coverage = CoverageMap(params)
            stride = max(1, int(round((1.0 / fps) / params.timestep_s)))
            cursor = 0
            saved: set[str] = set()
            stills = None
            if stills_dir is not None:
                stills = {name: stills_dir / f"{name}.png" for name in ("hover", "scan", "disturbance", "retreat")}
            for index in range(0, len(rows["time_s"]), stride):
                while cursor <= index:
                    if rows["spraying"][cursor]:
                        coverage.accumulate(rows["nozzle_position"][cursor], _axis_from_quat(rows["quaternion"][cursor]), True)
                    cursor += 1
                paint_coverage(model, coverage.grid, params.visual_tiles)
                _pose(model, data, rows["position"][index], rows["quaternion"][index])
                place_footprint(data, rows["nozzle_position"][index] if rows["spraying"][index] else None)
                mujoco.mj_forward(model, data)
                frame = renderer.render(data)
                wind = float(np.linalg.norm(rows["wind_force_n"][index]))
                phase = str(rows["phase"][index])
                image = _annotate(
                    frame,
                    [
                        f"{title}  ·  trajectory replay",
                        f"t = {rows['time_s'][index]:5.1f} s   {phase}   estimated spray coverage {100 * rows['coverage'][index]:4.1f}%",
                        f"clearance {rows['clearance_m'][index]:.2f} m   nozzle {rows['nozzle_distance_m'][index]:.2f} m"
                        + (f"   wind {wind:.1f} N along +X" if wind > 0.1 else ""),
                    ],
                )
                _write(process, image)
                if stills:
                    _maybe_still(stills, saved, phase, wind, image, float(rows["coverage"][index]))
            outcome = [
                f"{title} result",
                f"Estimated spray coverage  {100 * metrics['estimated_spray_coverage']:.1f}%",
                f"Position tracking RMSE    {metrics['position_rmse_m']:.3f} m",
                f"Nozzle distance RMSE      {metrics['nozzle_distance_rmse_m']:.3f} m",
                f"Minimum clearance         {metrics['min_clearance_m']:.3f} m",
                f"Contact episodes          {metrics['contact_episodes']}",
                f"Completed                 {metrics['completed']}",
            ]
            if metrics.get("recovery_time_s") is not None:
                outcome.append(f"Recovery time             {metrics['recovery_time_s']:.2f} s")
            for _ in range(fps * 2):
                _write(process, _card(width, height, outcome))
        process.stdin.close()
        code = process.wait(timeout=180)
        if code != 0:
            error = process.stderr.read().decode("utf-8", "replace")[-2000:]
            raise RuntimeError(f"ffmpeg exited {code}: {error}")
    finally:
        if process.poll() is None:
            process.kill()
