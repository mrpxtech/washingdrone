"""Command line: python -m washsim run|demo|video|verify."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

from washsim.metrics import baseline_targets_met
from washsim.params import Params, modeling_assumptions
from washsim.report import build_bundle, save_plots, save_replay, write_manifest
from washsim.simulate import load_rows, run_episode
from washsim.video import write_demo_video

ROOT = Path(__file__).resolve().parents[1]


def _run_id(scenario: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario}-{stamp}"


def _one(scenario: str, run_id: str, params: Params) -> dict:
    run_dir = ROOT / "runs" / run_id
    print(f"running {scenario} -> {run_dir}", flush=True)
    result = run_episode(scenario, run_dir, params, run_id=run_id)
    save_plots(run_dir, result["rows"], params)
    save_replay(run_dir, result["rows"], result["metrics"], params)
    write_manifest(
        run_dir,
        result["metrics"],
        params,
        {"assumptions": modeling_assumptions(), "hostname": socket.gethostname()},
    )
    metrics = result["metrics"]
    print(
        f"  coverage={metrics['estimated_spray_coverage']:.3f} "
        f"rmse={metrics['position_rmse_m']:.3f} m "
        f"clearance={metrics['min_clearance_m']:.3f} m "
        f"contacts={metrics['contact_episodes']} "
        f"completed={metrics['completed']} {metrics['fail_reason']}",
        flush=True,
    )
    return result


def demo() -> int:
    params = Params()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    first = _one("baseline", _run_id("baseline"), params)
    second = _one("baseline", _run_id("baseline"), params)
    wind = _one("wind", _run_id("wind"), params)
    preview = ROOT / "outputs" / "demo.mp4"
    stills = ROOT / "outputs" / "frames"
    print("rendering preview video", flush=True)
    write_demo_video(
        [
            ("Baseline", first["rows"], first["metrics"]),
            ("Wind disturbance", wind["rows"], wind["metrics"]),
        ],
        params,
        preview,
        width=640,
        height=360,
        fps=15,
        stills_dir=stills,
    )
    bundle = build_bundle(ROOT, [ROOT / "runs" / item["metrics"]["run_id"] for item in (first, second, wind)])
    summary = {
        "baseline": first["metrics"],
        "baseline_repeat": second["metrics"],
        "wind": wind["metrics"],
        "baseline_targets": baseline_targets_met(first["metrics"]),
        "reproducibility": _reproducibility(first["metrics"], second["metrics"]),
        "preview_video": str(preview.relative_to(ROOT)),
        "bundle": str(bundle.relative_to(ROOT)),
        "presentation_runs": [first["metrics"]["run_id"], wind["metrics"]["run_id"]],
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"targets": summary["baseline_targets"], "reproducibility": summary["reproducibility"]}, indent=2))
    ok = all(summary["baseline_targets"].values()) and summary["reproducibility"]["within_tolerance"]
    return 0 if ok else 1


def _reproducibility(first: dict, second: dict) -> dict:
    coverage_delta = abs(first["estimated_spray_coverage"] - second["estimated_spray_coverage"])
    rmse_delta = abs(first["position_rmse_m"] - second["position_rmse_m"])
    return {
        "coverage_abs_delta": coverage_delta,
        "position_rmse_abs_delta_m": rmse_delta,
        "coverage_tolerance": 0.005,
        "rmse_tolerance_m": 0.01,
        "within_tolerance": coverage_delta <= 0.005 and rmse_delta <= 0.01,
    }


def video() -> int:
    summary_path = ROOT / "results" / "summary.json"
    if not summary_path.exists():
        print("results/summary.json is missing. Run make demo first.", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text())
    params = Params()
    episodes = []
    titles = {"baseline": "Baseline", "wind": "Wind disturbance"}
    for key in ("baseline", "wind"):
        run_id = summary[key]["run_id"]
        rows = load_rows(ROOT / "runs" / run_id)
        episodes.append((titles[key], rows, summary[key]))
    destination = ROOT / "outputs" / "presentation.mp4"
    print(f"rendering {destination}", flush=True)
    write_demo_video(episodes, params, destination, width=1280, height=720, fps=24)
    return 0


def verify() -> int:
    summary_path = ROOT / "results" / "summary.json"
    if not summary_path.exists():
        print("No results/summary.json yet. Unit tests can still pass; demo results are pending.")
        return 0
    summary = json.loads(summary_path.read_text())
    print(json.dumps({"targets": summary["baseline_targets"], "reproducibility": summary["reproducibility"]}, indent=2))
    ok = all(summary["baseline_targets"].values()) and summary["reproducibility"]["within_tolerance"]
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "demo"
    if command == "run":
        scenario = argv[2] if len(argv) > 2 else "baseline"
        if scenario not in ("baseline", "wind"):
            print("scenario must be baseline or wind", file=sys.stderr)
            return 2
        run_id = argv[3] if len(argv) > 3 else _run_id(scenario)
        _one(scenario, run_id, Params())
        return 0
    if command == "demo":
        return demo()
    if command == "video":
        return video()
    if command == "verify":
        return verify()
    print("usage: python -m washsim run [baseline|wind] | demo | video | verify", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
