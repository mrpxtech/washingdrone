"""Run metrics. Definitions used by the dashboard and the README.

Position tracking RMSE
    sqrt(mean(||p - p_ref||^2)) on the recorded samples. The reference is the
    controller reference at the same timestamp.

Nozzle distance RMSE
    sqrt(mean((nozzle_y - nozzle_standoff)^2)) during the scanning phase only.
    Settle and retreat change the standoff on purpose.

Minimum vehicle clearance
    The smallest distance from any vehicle collision geom to the plane y = 0.
    Positive means the geom is entirely on the drone side of the glass.

Contact episode
    A sample is in contact when clearance is <= 0 or MuJoCo reports a contact
    between a veh_* geom and the glass. An episode starts at the first such
    sample and ends after 0.10 s continuously out of contact. Every episode is
    unintended.

Recovery
    For the wind case, recovery succeeds when some 1.0 s window after the wind
    ends keeps the position error at or below 0.08 m. recovery_time_s is the
    delay from the end of the wind to the start of that window.

Completion
    The episode reached its planned duration, the state stayed finite, tilt
    stayed at or below 75 degrees, altitude stayed above 0.25 m, and the final
    position error is at or below 0.15 m.
"""

from __future__ import annotations

import numpy as np


RELEASE_S = 0.10
RECOVERY_ERROR_M = 0.08
RECOVERY_WINDOW_S = 1.0


def contact_episodes(time_s: np.ndarray, contacting: np.ndarray) -> int:
    episodes = 0
    in_contact = False
    release_start = None
    for time, flag in zip(time_s, contacting):
        if flag:
            if not in_contact:
                episodes += 1
                in_contact = True
            release_start = None
        elif in_contact:
            if release_start is None:
                release_start = float(time)
            elif float(time) - release_start >= RELEASE_S:
                in_contact = False
                release_start = None
    return episodes


def recovery_time(time_s: np.ndarray, error_m: np.ndarray, wind_end_s: float | None) -> float | None:
    if wind_end_s is None:
        return None
    mask = time_s >= wind_end_s
    times = time_s[mask]
    errors = error_m[mask]
    if len(times) == 0:
        return None
    for start_index, start in enumerate(times):
        window = errors[(times >= start) & (times <= start + RECOVERY_WINDOW_S)]
        span = times[(times >= start) & (times <= start + RECOVERY_WINDOW_S)]
        if len(span) == 0:
            continue
        if span[-1] - span[0] < RECOVERY_WINDOW_S - 0.05:
            continue
        if np.all(window <= RECOVERY_ERROR_M):
            return float(start - wind_end_s)
    return None


def summarize(rows: dict[str, np.ndarray], standoff_m: float, wind_end_s: float | None, failed: bool, fail_reason: str) -> dict:
    position_error = rows["position"] - rows["ref_position"]
    error_norm = np.linalg.norm(position_error, axis=1)
    rmse = float(np.sqrt(np.mean(error_norm**2))) if len(error_norm) else float("nan")
    scan = rows["phase"] == "scan"
    if np.any(scan):
        nozzle_error = rows["nozzle_distance_m"][scan] - standoff_m
        nozzle_rmse = float(np.sqrt(np.mean(nozzle_error**2)))
    else:
        nozzle_rmse = float("nan")
    tilt = rows["tilt_rad"]
    contacting = rows["contact"] > 0.5
    recovered_after = recovery_time(rows["time_s"], error_norm, wind_end_s)
    final_error = float(error_norm[-1]) if len(error_norm) else float("nan")
    completed = (not failed) and np.isfinite(final_error) and final_error <= 0.15
    return {
        "estimated_spray_coverage": float(rows["coverage"][-1]) if len(rows["coverage"]) else 0.0,
        "position_rmse_m": rmse,
        "nozzle_distance_rmse_m": nozzle_rmse,
        "min_clearance_m": float(np.min(rows["clearance_m"])) if len(rows["clearance_m"]) else float("nan"),
        "contact_episodes": contact_episodes(rows["time_s"], contacting),
        "max_tilt_deg": float(np.degrees(np.max(tilt))) if len(tilt) else float("nan"),
        "tilt_rms_deg": float(np.degrees(np.sqrt(np.mean(tilt**2)))) if len(tilt) else float("nan"),
        "actuator_saturation_fraction": float(np.mean(rows["saturated"])) if len(rows["saturated"]) else 0.0,
        "completed": bool(completed),
        "failed": bool(failed),
        "fail_reason": fail_reason,
        "final_position_error_m": final_error,
        "recovered": None if wind_end_s is None else recovered_after is not None,
        "recovery_time_s": recovered_after,
        "duration_s": float(rows["time_s"][-1]) if len(rows["time_s"]) else 0.0,
        "samples": int(len(rows["time_s"])),
    }


def baseline_targets_met(metrics: dict) -> dict[str, bool]:
    return {
        "coverage_at_least_0.90": metrics["estimated_spray_coverage"] >= 0.90,
        "position_rmse_at_most_0.15_m": metrics["position_rmse_m"] <= 0.15,
        "min_clearance_at_least_0.30_m": metrics["min_clearance_m"] >= 0.30,
        "zero_contacts": metrics["contact_episodes"] == 0,
        "completed": bool(metrics["completed"]),
    }
