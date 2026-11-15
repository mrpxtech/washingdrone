import numpy as np

from washsim.mission import build_mission
from washsim.params import Params


def test_reference_speed_stays_inside_the_limit_and_passes_overlap():
    params = Params()
    mission = build_mission(params)
    peak = 0.0
    spraying = False
    for step in range(int(mission.duration / 0.02)):
        sample = mission.sample(step * 0.02)
        peak = max(peak, float(np.linalg.norm(sample.velocity)))
        spraying = spraying or sample.spraying
    assert peak <= params.speed_limit_mps + 1e-6
    assert spraying
    assert mission.scan_end_s > mission.scan_start_s
    heights = sorted(
        {
            round(float(segment.p0[2]), 4)
            for segment in mission.segments
            if segment.phase == "scan" and abs(float(segment.p0[2] - segment.p1[2])) < 1e-6
        }
    )
    assert len(heights) >= 4
    gaps = np.diff(heights)
    assert np.max(gaps) <= params.pass_spacing_m + 1e-6
