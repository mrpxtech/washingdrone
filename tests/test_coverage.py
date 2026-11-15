import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from washsim.coverage import CoverageMap
from washsim.params import Params


def _at(y, axis):
    params = Params()
    coverage = CoverageMap(params)
    nozzle = np.array([0.0, y, params.panel_z0_m + 0.5 * params.panel_height_m])
    coverage.accumulate(nozzle, np.array(axis, dtype=float), True)
    return coverage


def test_spraying_disabled_adds_nothing():
    params = Params()
    coverage = CoverageMap(params)
    nozzle = np.array([0.0, 0.8, params.panel_z0_m + 0.9])
    coverage.accumulate(nozzle, np.array([0.0, -1.0, 0.0]), False)
    assert coverage.fraction == 0.0


def test_far_nozzle_is_outside_the_operating_window():
    coverage = _at(2.0, [0.0, -1.0, 0.0])
    assert coverage.fraction == 0.0


def test_axis_pointing_away_is_rejected():
    coverage = _at(0.8, [0.0, 1.0, 0.0])
    assert coverage.fraction == 0.0


def test_centered_spray_covers_a_bounded_patch():
    coverage = _at(0.8, [0.0, -1.0, 0.0])
    assert 0.01 < coverage.fraction < 0.2
    # Cells outside the ellipse stay dry, so the patch is clipped by geometry.
    assert coverage.grid[0, 0] == 0
    assert coverage.grid.mean() > 0


def test_oblique_spray_still_marks_cells():
    straight = _at(0.8, [0.0, -1.0, 0.0]).fraction
    oblique = _at(0.8, [0.45, -0.9, 0.0]).fraction
    assert oblique > 0.0
    assert oblique != straight
