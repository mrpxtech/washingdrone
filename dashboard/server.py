"""Small dashboard server.

The browser never steps MuJoCo. Run simulation starts one backend episode at a
time and the page loads that episode's replay.json.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "runs"

_job = {
    "state": "idle",
    "run_id": None,
    "scenario": None,
    "error": None,
}
_lock = threading.Lock()
_runner = None


def tailscale_ipv4() -> str | None:
    try:
        text = subprocess.check_output(["tailscale", "ip", "-4"], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None


def free_port(host: str, start: int = 8765) -> int:
    for port in range(start, start + 30):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, port))
        except OSError:
            continue
        finally:
            probe.close()
        return port
    raise RuntimeError(f"no free port on {host} from {start} to {start + 29}")


def default_runner(scenario: str, run_id: str) -> int:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    env["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [sys.executable, "-m", "washsim", "run", scenario, run_id],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    log = RUNS / run_id / "log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
    return completed.returncode


def _finish_run_artifacts(run_id: str) -> None:
    """Plots and replay are written by the CLI when the subprocess is the real runner."""
    return


def start_job(scenario: str) -> tuple[bool, str]:
    if scenario not in ("baseline", "wind"):
        return False, "scenario must be baseline or wind"
    with _lock:
        if _job["state"] in ("queued", "running"):
            return False, "A simulation is already queued or running. Wait for it to finish."
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{scenario}-{stamp}"
        _job.update(state="queued", run_id=run_id, scenario=scenario, error=None)
    thread = threading.Thread(target=_worker, args=(scenario, run_id), daemon=True)
    thread.start()
    return True, run_id


def _worker(scenario: str, run_id: str) -> None:
    with _lock:
        _job["state"] = "running"
    runner = _runner or default_runner
    try:
        code = runner(scenario, run_id)
    except Exception as exc:  # noqa: BLE001 - surface the failure on the dashboard
        with _lock:
            _job["state"] = "failed"
            _job["error"] = str(exc)
        return
    replay = RUNS / run_id / "replay.json"
    metrics_path = RUNS / run_id / "metrics.json"
    with _lock:
        if code == 0 and replay.exists() and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("failed"):
                _job["state"] = "failed"
                _job["error"] = metrics.get("fail_reason") or "episode failed"
            else:
                _job["state"] = "completed"
                _job["error"] = None
        else:
            _job["state"] = "failed"
            log = RUNS / run_id / "log.txt"
            detail = log.read_text()[-800:] if log.exists() else f"simulation exited {code}"
            _job["error"] = detail.strip() or f"simulation exited {code}"


def list_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    found = []
    for path in RUNS.iterdir():
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        found.append(
            {
                "run_id": metrics.get("run_id", path.name),
                "scenario": metrics.get("scenario", "unknown"),
                "completed": metrics.get("completed"),
                "estimated_spray_coverage": metrics.get("estimated_spray_coverage"),
                "position_rmse_m": metrics.get("position_rmse_m"),
            }
        )
    found.sort(key=lambda item: item["run_id"])
    return found


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict | list) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/status":
            with _lock:
                self._json(200, dict(_job))
            return
        if path == "/api/runs":
            self._json(200, list_runs())
            return
        if path.startswith("/api/runs/"):
            run_id = path.split("/")[-1]
            if not run_id.replace("-", "").isalnum():
                self._json(400, {"error": "invalid run id"})
                return
            replay = RUNS / run_id / "replay.json"
            if not replay.exists():
                self._json(404, {"error": f"no replay for {run_id}"})
                return
            self._send(200, replay.read_bytes(), "application/json")
            return
        if path == "/":
            path = "/index.html"
        relative = path.lstrip("/")
        file_path = (STATIC / relative).resolve()
        if not str(file_path).startswith(str(STATIC.resolve())) or not file_path.is_file():
            self._json(404, {"error": "not found"})
            return
        types = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json",
            ".png": "image/png",
        }
        self._send(200, file_path.read_bytes(), types.get(file_path.suffix, "application/octet-stream"))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/jobs":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        ok, message = start_job(str(payload.get("scenario", "")))
        if not ok:
            self._json(409, {"error": message})
            return
        self._json(202, {"state": "queued", "run_id": message})


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"dashboard http://{host}:{port}/", flush=True)
    print("Open that URL on a laptop or phone that is on this tailnet.", flush=True)
    server.serve_forever()


def main() -> None:
    host = tailscale_ipv4() or "127.0.0.1"
    if len(sys.argv) > 1:
        host = sys.argv[1]
    port = free_port(host)
    serve(host, port)


if __name__ == "__main__":
    main()
