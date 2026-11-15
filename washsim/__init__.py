"""Window-washing quadrotor simulation.

MUJOCO_GL is set before any MuJoCo import. Submodules import MuJoCo themselves.
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

__version__ = "1.0.0"
