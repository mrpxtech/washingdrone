# Window-washing drone

A MuJoCo demonstration of one quadrotor washing a vertical glass panel. Physics runs on the VM. The browser replays recorded states and the estimated spray coverage. It does not integrate the model.

The working tree on the interview VM is `/home/drone/washingdrone` (host `drone`, user `drone`, Tailscale `100.115.231.106`). That is this repository. The path `~/projects/drone-window-wash-sim` was not used.

## Install

From the repository root, with `~/.local/bin` on `PATH` if `make` is the user-local binary:

```bash
make setup
make demo
make verify
make serve
make video
```

`make setup` creates `.venv`, installs the pinned Python packages, and extracts OSMesa into `.deps` when the system linker does not already provide it. This VM's CPU is a QEMU virtual CPU without AVX, so the PyPI `mujoco` wheel raises `SIGILL`. The environment here uses MuJoCo 3.12.0 built from the 3.12.0 tag with `-march=x86-64 -mno-avx -mno-avx2`. `make setup` keeps that install when `import mujoco` already works, and only tries the wheel when it does not.

If a normal package install is available, the system packages are:

```bash
sudo apt-get install -y python3.12-venv make ffmpeg libosmesa6 libllvm20
```

Set `MUJOCO_GL=osmesa` before importing MuJoCo. The Makefile does that.

## Architecture

| Piece | Role |
| --- | --- |
| `washsim/model.py` | MJCF for the window, frame, floor, and quadrotor. Loads the model and reads simulated state. |
| `washsim/controller.py` | Position and attitude control, then the rotor allocation. |
| `washsim/mission.py` | Settle, overlapping horizontal passes, retreat, and hold. |
| `washsim/coverage.py` | Estimated spray coverage on the glass. |
| `washsim/simulate.py` | One episode: rotor forces, optional wind, MuJoCo steps, telemetry. |
| `washsim/metrics.py` | RMSE, clearance, contacts, recovery, completion. |
| `washsim/video.py` | Trajectory replay rendered with OSMesa and streamed to ffmpeg. |
| `dashboard/server.py` | One physics job at a time, then the recorded replay. |

A run directory under `runs/<run-id>/` holds `telemetry.csv`, `metrics.json`, `coverage.png`, `tracking.png`, `replay.json`, and `manifest.json`. `results/summary.json` is the compact measured-results record. Videos and the portable bundle are generated under `outputs/` and are not committed.

## Coordinate frames and the vehicle

World frame: +X along the window width, +Y away from the glass (the pane is the plane `y = 0`), +Z up. Units are meters, seconds, radians, and newtons. Body yaw 0 aligns the body axes with the world. The nozzle points along body −Y. Rotors spin about body +Z.

Rotor order, viewed from above: 0 front (+X) counterclockwise, 1 left (+Y) clockwise, 2 back (−X) counterclockwise, 3 right (−Y) clockwise. The reaction torque on the body is opposite the spin, `tau_z = -kappa * spin_sign * thrust`, with `kappa = 0.018 m`. Each rotor thrust is limited to 0–12 N. Hover thrust for the 2 kg vehicle is `mg = 19.62 N`.

The panel is 1.8 by 1.8 m, with its bottom at `z = 1.05 m`. The nozzle target distance is 0.8 m, so the center of mass sits at 1.00 m (`nozzle length 0.20 m + 0.8 m`). The arm length is 0.16 m. Inertia `(0.014, 0.016, 0.026) kg m²` is an estimate: a 1.2 kg central body plus four 0.20 kg motor masses at 0.16 m, rounded up for the nozzle, skids, and disks. The vehicle is one free rigid body. Collision shapes are the parts named `veh_*` and the glass. Pose writes happen only when an episode is initialized, and again when a picture or video poses a recorded frame. The mission itself advances only by rotor forces and `mj_step`.

## Controller

The position loop, in the world frame, is

```
a = Kp e + Kd ė + Ki ∫e + a_ref
thrust direction = normalize(a + (0, 0, g))
```

Horizontal and vertical accelerations are limited. The integrator freezes and decays while any rotor command is saturated. The attitude loop is the Lee geometric tracker at yaw 0, so body −Y keeps facing the glass when the vehicle is level:

```
e_R = 0.5 vee(R_des^T R - R^T R_des)
tau = -Kr e_R - Kw omega
```

Allocation matches the force application:

```
T     = sum Ti
tau_x = L (T1 - T3)
tau_y = L (T2 - T0)
tau_z = -kappa * sum(si Ti)
```

Thrust is applied at the four rotor sites along body +Z. Roll and pitch come from `r × F`. Reaction torque is applied about the body vertical only. `qfrc_applied` is cleared every step.

## Path

The vehicle starts 0.10 m farther from the glass and 0.08 m above the `t = 0` reference. It moves to the first pass, dwells, flies overlapping horizontal passes, retreats 0.75 m, and holds. Moves are rest-to-rest quintics. The peak speed factor is 1.875 and the peak of `d²s/dτ²` is about 5.77, so each segment is long enough that the peak speed stays at or below 0.70 m/s and the peak acceleration stays at or below 1.2 m/s².

Pass spacing uses the circular footprint at the 0.8 m standoff. The half-angle is 14°, so the radius is `0.8 tan(14°) ≈ 0.199 m`. Spacing is the diameter times `(1 - 0.30 overlap)`.

## Estimated spray coverage

This is the fraction of a 90 by 90 panel grid whose cell centers have been reached. A cell is marked only while spraying is enabled, the nozzle axis points at the glass (`axis · n < -0.25`), the slant range is inside 1.35 m, and the normal distance is inside 0.45–1.15 m. The footprint is the cone-plane intersection, approximated as an ellipse: minor radius `t tan(alpha)`, major radius `minor / cos(theta)`, clipped to the panel.

Not modeled, on purpose: water, droplets, dirt removal, hose forces, spray reaction force, wall-effect aerodynamics, and sensing. The controller reads the simulated state directly. Wind, when selected, is one external force at the center of mass: +X, 3.5 N, starting 8 s after the scan begins and lasting 4 s. Air density and viscosity are zero otherwise.

## Evaluation

Tracking RMSE is `sqrt(mean(||p - p_ref||²))` against the reference at the same timestamp. Nozzle-distance RMSE uses the scanning phase only. Clearance is the distance from the closest `veh_*` collision corner or cap to the plane `y = 0`. A contact sample is clearance `<= 0` or a MuJoCo contact between a `veh_*` geom and the glass. An episode ends after 0.10 s continuously clear. Recovery, for the wind case, is the delay from the end of the wind to a 1.0 s window whose position error stays at or below 0.08 m. Completion means the planned duration was reached, the state stayed finite, tilt stayed at or below 75°, altitude stayed above 0.25 m, and the final position error is at or below 0.15 m.

Timestep is 0.002 s, seed is 0, integrator is `implicitfast`. A repeated baseline is within tolerance when coverage changes by at most 0.005 and position RMSE by at most 0.01 m.

## Measured results

Recorded 15 Nov 2026 on `drone`. Both baselines are `baseline-20261115T010526Z` and `baseline-20261115T010534Z`. The wind run is `wind-20261115T010541Z`. The repeat matched the first baseline exactly.

| | Baseline | Wind disturbance |
| --- | ---: | ---: |
| Estimated spray coverage | 100% | 99.9% |
| Position RMSE | 0.015 m | 0.087 m |
| Nozzle distance RMSE | 0.003 m | 0.003 m |
| Minimum clearance | 0.771 m | 0.771 m |
| Contact episodes | 0 | 0 |
| Actuator saturation | 0 | 0 |
| Max tilt | 8.0° | 11.8° |
| Completed | yes | yes |
| Recovery after the wind | — | 5.30 s |

Baseline targets (coverage ≥ 90%, RMSE ≤ 0.15 m, clearance ≥ 0.30 m, no contacts, completed) are met. The wind case uses the same metrics: coverage and clearance hold, and the position RMSE rises by about 0.07 m during the gust and then returns.

## Dashboard

`make serve` binds the first free port from 8765 on the Tailscale IPv4 address and prints the URL. On this machine that is:

```text
http://100.115.231.106:8765/
```

Open it from a laptop or phone that is on the same tailnet. Run simulation starts one new MuJoCo episode. Play, pause, reset, the scrubber, and the speed control only move through the selected recording. The page labels that view "Trajectory replay of recorded MuJoCo states" and shows the scenario and run id.

The portable bundle is `outputs/replay-bundle/`. From that directory, with no physics:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/static/replay.html`. To show it on a phone, bind the Tailscale address instead of `127.0.0.1`.

## 90-second demo

1. Open the dashboard and name the loaded run: scenario, run id, and "trajectory replay."
2. Play the baseline. Point at the orange body, the cyan nozzle aimed at the glass, the footprint, and the coverage tiles filling the pane.
3. Read the cards: about 100% estimated spray coverage, 0.015 m position RMSE, 0.77 m clearance, no contacts.
4. Switch to the wind run. Around 11 s the overlay shows 3.5 N along +X. The path bends, the error plot rises, and the vehicle returns. Coverage stays near 100% and clearance does not change.
5. Close on the limitation: coverage is a geometric footprint, not simulated water or dirt.

## Five-minute code walkthrough

1. **Physics** (`washsim/simulate.py`, `washsim/model.py`). One free body, gravity, and `mj_step`. Thrust and reaction torque are applied with `mj_applyFT` at the rotor sites. Pose writes are limited to initialization and to posing a recorded frame for a picture.
2. **Allocation** (`controller.allocation_matrix`). Four thrusts produce total thrust, roll, pitch, and yaw. The inverse of that map is the mixer, then each command is clipped to 0–12 N.
3. **Controller** (`Controller.step`). World-frame PD with a limited integrator and gravity compensation, then the Lee attitude error at yaw 0 so the nozzle keeps facing the glass.
4. **Path** (`washsim/mission.py`). Quintic segments under the 0.70 m/s peak. Pass heights come from the footprint diameter and the 30% overlap.
5. **Coverage** (`washsim/coverage.py`). Ellipse on the glass, gated by spray enable, aim, and distance. The reported number is the fraction of grid cells reached.
6. **Disturbance** (`wind_force`). A timed force at the center of mass, saved in the telemetry. It is not a wind field.
7. **Evaluation** (`washsim/metrics.py`). Time-aligned RMSE, scan-only nozzle RMSE, closest collision clearance, contact episodes with a 0.10 s release, and the 1.0 s recovery window.

## Verification already completed

- Physics step and a visible OSMesa frame before the mission.
- Hover holds within 5 cm, the same hover repeats, and a +X step moves the correct way (`tests/test_hover.py`).
- Mixer signs, saturation, coverage gating, contact counting, and mission speed limits.
- Baseline twice and the wind case, with the targets above and an exact repeat.
- Telemetry is finite, timestamps step by 0.002 s, and RMSE, coverage, and clearance match `metrics.json`.
- `outputs/demo.mp4` decodes as 640×360 H.264 at 15 fps. Representative frames for hover, scan, the gust, and retreat were inspected.
- Dashboard run selection is covered by `tests/test_dashboard.py`.

`make video` writes `outputs/presentation.mp4` at 1280×720 and 24 fps from the same run ids.

## Limitations

The coverage number is not a washed-glass claim. There is no water, dirt, hose, spray momentum, or wall cushion. Wind is one force. State feedback is perfect. Inertia and `kappa` are plausible illustrations, not a measured airframe. The PyPI MuJoCo wheel does not run on this VM's CPU; the source build described above is the one that does.
