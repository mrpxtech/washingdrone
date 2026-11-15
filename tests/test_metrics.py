import numpy as np

from washsim.metrics import contact_episodes, recovery_time, summarize


def test_contact_episode_merges_short_gaps_and_counts_two_hits():
    time_s = np.arange(0, 1.0, 0.01)
    contacting = np.zeros(len(time_s))
    contacting[10:20] = 1
    contacting[25:30] = 1  # 0.05 s gap, still one episode
    contacting[80:90] = 1
    assert contact_episodes(time_s, contacting) == 2


def test_tracking_rmse_on_a_known_offset():
    count = 20
    rows = {
        "time_s": np.linspace(0, 1, count),
        "position": np.tile(np.array([0.3, 0.0, 0.0]), (count, 1)),
        "ref_position": np.zeros((count, 3)),
        "nozzle_distance_m": np.full(count, 0.8),
        "clearance_m": np.full(count, 0.5),
        "contact": np.zeros(count),
        "tilt_rad": np.zeros(count),
        "saturated": np.zeros(count),
        "coverage": np.linspace(0, 0.5, count),
        "phase": np.array(["scan"] * count),
    }
    metrics = summarize(rows, 0.8, None, False, "")
    assert abs(metrics["position_rmse_m"] - 0.3) < 1e-9
    assert metrics["nozzle_distance_rmse_m"] == 0.0
    assert metrics["contact_episodes"] == 0
    # Completion requires the final position error to be at or below 0.15 m.
    assert not metrics["completed"]


def test_recovery_window():
    time_s = np.linspace(0, 6, 601)
    error = np.full_like(time_s, 0.2)
    error[time_s >= 3.5] = 0.05
    assert abs(recovery_time(time_s, error, 2.0) - 1.5) < 0.05
    assert recovery_time(time_s, np.full_like(time_s, 0.2), 2.0) is None
