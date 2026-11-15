import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from dashboard import server


def _replay(run_id, scenario, coverage):
    return {
        "run_id": run_id,
        "scenario": scenario,
        "label": "Trajectory replay of recorded MuJoCo states",
        "dt_s": 0.5,
        "frames": [
            {
                "t": 0.0,
                "p": [0, 1, 1],
                "q": [1, 0, 0, 0],
                "ref": [0, 1, 1],
                "nozzle": [0, 0.8, 1],
                "coverage": coverage,
                "clearance": 0.8,
                "spraying": 0,
                "phase": "settle",
                "wind": [0, 0, 0],
                "tilt_deg": 0,
            }
        ],
        "coverage_grid": [[0, 1], [1, 0]],
        "metrics": {
            "run_id": run_id,
            "scenario": scenario,
            "estimated_spray_coverage": coverage,
            "position_rmse_m": 0.01,
            "nozzle_distance_rmse_m": 0.01,
            "min_clearance_m": 0.8,
            "contact_episodes": 0,
            "max_tilt_deg": 1,
            "tilt_rms_deg": 0.2,
            "actuator_saturation_fraction": 0,
            "completed": True,
            "failed": False,
            "fail_reason": "",
            "recovered": None,
            "recovery_time_s": None,
        },
        "panel": {"width_m": 1.8, "height_m": 1.8, "z0_m": 1.05, "standoff_m": 0.8},
    }


def test_run_selection_returns_the_chosen_episode(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    for run_id, scenario, coverage in (("baseline-a", "baseline", 0.91), ("wind-a", "wind", 0.7)):
        folder = runs / run_id
        folder.mkdir()
        payload = _replay(run_id, scenario, coverage)
        (folder / "metrics.json").write_text(json.dumps(payload["metrics"]))
        (folder / "replay.json").write_text(json.dumps(payload))
    monkeypatch.setattr(server, "RUNS", runs)
    listed = server.list_runs()
    assert [item["run_id"] for item in listed] == ["baseline-a", "wind-a"]
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        import urllib.request

        baseline = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/baseline-a"))
        wind = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs/wind-a"))
        assert baseline["metrics"]["estimated_spray_coverage"] == 0.91
        assert wind["scenario"] == "wind"
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
        assert b"Run simulation" in page
        script = urllib.request.urlopen(f"http://127.0.0.1:{port}/vendor/three.min.js").read()
        assert b"THREE" in script
    finally:
        httpd.shutdown()


def test_second_job_is_rejected_while_one_is_queued(monkeypatch):
    monkeypatch.setattr(server, "_job", {"state": "running", "run_id": "baseline-x", "scenario": "baseline", "error": None})
    ok, message = server.start_job("wind")
    assert not ok
    assert "already" in message
