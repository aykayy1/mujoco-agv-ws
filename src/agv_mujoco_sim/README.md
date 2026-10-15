# AGV MuJoCo + Nav2 MPPI + SAC supervisor

SAC is a bounded supervisory layer. It proposes a common speed scale for MPPI;
it never publishes `cmd_vel` and does not control wheel torque or motor state.

This package is synchronized with the real robot frame convention:

- `base_foot_link` -> `base_link`
- `left_wheel_jt`, `right_wheel_jt`
- `imu`, `lidar`

Runtime baseline:

- Motor command and wheel odometry: 20 Hz
- IMU: 50 Hz
- EKF output: 35 Hz
- LiDAR: 10 Hz
- Nav2/MPPI controller: 20 Hz
- SAC supervisory action: 5 Hz

The simulation is LiDAR + AMCL only. Camera geometry is removed from MuJoCo
and the simulation URDF, while its mass and inertia remain merged into
`base_link`; total robot mass remains 80 kg.

## Build

Place this directory in the workspace `src` folder, then run:

```bash
cd ~/mujoco_agv_ws
colcon build --symlink-install --packages-select agv_mujoco_sim
source install/setup.bash
```

## Start simulation and EKF

```bash
ros2 launch agv_mujoco_sim mujoco_sim.launch.py
```

The bridge publishes `/wheel/odom`, `/imu`, `/scan`, `/joint_states`, and
`/clock`. EKF is the only publisher of `odom -> base_foot_link`.

## Start AMCL and Nav2

```bash
ros2 launch agv_mujoco_sim localization.launch.py \
  map:=/absolute/path/to/map.yaml
ros2 launch agv_mujoco_sim navigation.launch.py
```

Before RL training, verify topic rates, the TF tree, forward LiDAR direction,
wheel odometry, EKF, and AMCL using the real deployment map.

## Measure the fixed-MPPI baseline

Start MuJoCo, localization, and Nav2 without an RL policy, then measure how
much of MPPI's configured `vx_max` is actually used on the clear straight
from `(0, 0)` to `(9, 0)`:

```bash
ros2 run agv_mujoco_sim benchmark_mppi_baseline.py --trials 3
```

The command writes a JSON summary and per-sample CSV under
`benchmark_results/`. `mean_cruise_speed_utilization >= 0.8` is the initial
go criterion for testing an RL speed-limit-only policy. Lower utilization
means the fixed MPPI baseline should be tuned and measured again first.

Run the three-route decision suite (clear straight, upper obstacle detour,
and lower zig-zag) before changing the RL interface:

```bash
ros2 run agv_mujoco_sim benchmark_mppi_baseline.py \
  --suite decision --trials 3 --timeout-s 75
```

For a reproducible A/B test, `--vx-std-override VALUE` temporarily changes
`FollowPath.vx_std` and restores its original value when the benchmark exits.
The decision suite also reports whole-visible-arc and central-forward clearance,
near-obstacle speed, command variation, success, collision, and travel time.

Evaluate the tuned fixed MPPI against an obstacle that is absent from the
static map and crosses the route only after the robot reaches cruise speed:

```bash
ros2 run agv_mujoco_sim benchmark_dynamic_obstacle.py --trials 3
```

This benchmark records detection and deceleration delay, front clearance,
estimated body clearance, collisions, success, and completion time. The
runtime obstacle is controlled through `/dynamic_obstacle/position` and is
returned outside the arena on every simulator reset.

Use `--reactive-speed-limit-m-s 0.25` as the deployable non-learning safety
baseline. Detection and release use only LiDAR range, goal distance, and
hysteresis; obstacle ground truth is reserved for scenario generation and
evaluation. This uses Nav2's coupled velocity-envelope `SpeedLimit` API and
does not reset the MPPI optimizer. `--reactive-limit-m-s` remains available
only as a dynamic-parameter comparison arm.

Run the reproducible scenario matrix after the Nav2/MuJoCo stack is active:

```bash
ros2 run agv_mujoco_sim run_dynamic_benchmark_matrix.py \
  --trials 3 --arm mppi --arm speedlimit025
```

For this fixed-controller A/B matrix, start `phase2_finetune_bringup.launch.py`
with `enable_rl_supervisor:=false`; otherwise the independent TTC shield would
intentionally intervene and contaminate both benchmark arms. Keep the default
`true` for SAC training and deployment.

The Phase-2 MuJoCo RL launch also enables a simulation-only front stop at the
wheel-command boundary. It uses the same forward LiDAR scan, a speed-dependent
stopping envelope and lateral corridor for crossing obstacles, and a stale-scan
fail-safe. This can command zero wheel
speed; the SAC/MPPI `SpeedLimit` path cannot. The stop publishes
`/sim_front_safety_stop` and `/sim_front_safety_diagnostics` for evaluation and is disabled with
`enable_rl_supervisor:=false` so baseline benchmark arms stay unchanged. It is
not a real-robot safety certification or proof against a moving obstacle
colliding with a stationary robot.
An approaching object at the edge of the forward LiDAR view is held for a
short interval after it disappears, so an obstacle cutting across the robot's
front corner does not immediately release the stop. Edge closing is compared
with wheel-odometry ego motion to avoid interpreting an ordinary static wall
as an approaching obstacle. The sixth diagnostic value is a stop-reason bitmask:
1 = missing/stale scan, 2 = forward corridor, 4 = edge hold, 8 = invalid command.

When the simulator is launched with domain randomization enabled, add
`--paired-domain-randomization`. Every matrix cell then resets the sampler and
uses the same seed/generation sequence, so MPPI and SpeedLimit are compared on
matching noise, bias, odometry-scale, and latency samples.

The `all` stage also randomizes actuator velocity-response scale. Current
light/medium/strong priors use LiDAR noise up to 1/2/3 cm, dropout up to
0.5/1/3%, odometry scale error up to 1/2/5%, command latency up to
10/25/50 ms, and velocity-response scale ranges of 0.98--1.0, 0.90--1.0,
and 0.80--1.0 respectively. These are behavioral-interface priors, not a
replacement for later wheel, friction, mass, and motor calibration.

The matrix includes clear, forward/reverse crossing, obstacle-speed,
early/late trigger, near-miss, and stop-on-path cases. Use `--scenario ID` to
run a subset and `--dry-run` to inspect all commands without starting trials.
Each cell writes JSON/CSV plus `manifest.json` and a compact `summary.csv`
under `benchmark_results/`.
The benchmark moves the obstacle from an independent ROS timer so blocking
parameter calls cannot pause its trajectory.

For long three-map benchmarks, `run_mppi_three_maps.py --tf-guarded` watches
the controller's own `map`→`odom` TF-age errors. A sustained lag of at least
2 s invalidates only the affected trial: the runner saves the attempt logs and
bag, restarts Nav2/MuJoCo, and replays the same seeded scenario. Collisions and
timeouts with healthy TF still count normally. The default allows one TF
restart; a second fault stops the run rather than silently dropping trials.
`benchmark.progress.json` contains only valid completed trials, while
`tf_faults.json` records system faults. Each stack attempt has its own rosbag
under `attempt_NNN/`; the final `benchmark.json` and samples CSV combine exactly
the requested number of valid trials.

To diagnose SAC-only speed-cap latency without a new 100-trial run, pass
`--tf-guarded --map MAP --replay-report OLD_MAP/benchmark.json
--replay-trials 2,94,100`. The runner uses those original trial IDs and the
exact stored obstacle scenarios, and writes a fresh rosbag and report. Use
`--supervisor-params-file config/rl_supervisor_sac_only_fast_fall.yaml` for
the separate diagnostic arm: SAC and MPPI are unchanged; only downward scale
filtering is faster. Upward filtering remains at baseline settings, with TTC
and the independent front stop disabled. This small replay checks a mechanism,
not a collision-rate estimate.

For SAC-only training with `--retain-crossing-obstacle`, the environment also
recovers from a motion stall without relying on the disabled front stop: less
than 0.20 m displacement across 30 one-second steps truncates the episode and
requests a hard simulator reset. The end reason distinguishes sustained
rotation (`rotation_stall`), stale Nav2 commands (`command_stall`), and other
no-translation cases (`motion_stall`). Every 180-step episode with no reached
goal now hard-resets, including non-retained training arms. Normal episodes
that have reached a goal retain the soft-reset path. The training summary
separately reports zero-goal episodes, collision episodes, moving-obstacle
time, and retained-obstacle time; goal count alone is not a safety metric.
The retained arm uses a new replay contract; resume from compatible model
weights without loading a replay buffer collected before these changes.

## Start Nav2 with the RL supervisor

```bash
ros2 launch agv_mujoco_sim navigation_rl.launch.py
```

The supervisor consumes exactly one value on `/rl_velocity_limits`, clamps it
to `[0.15, 1.00]`, filters/slew-limits normal RL changes, and uses Nav2's
`/speed_limit` API. An independent LiDAR TTC shield can reduce immediately to
the emergency scale, then hands control back to the smooth RL path after the
hazard clears. If SAC becomes stale, the watchdog removes the adaptive limit
and returns to scale `1.0` (the verified `vx_max=0.70 m/s` MPPI baseline).
Persistent blockage remains Nav2's responsibility: its progress checker and
Behavior Tree own replanning, costmap clearing, spin/wait and recovery;
the SAC policy is not trained to escape a stuck state.

`AgvRlEnv` uses a 54D temporal observation (24 forward-sector ranges across
the usable 120-degree LiDAR field, 24 matching TTC values, goal
distance/bearing, measured `vx/wz`, applied scale, hazard persistence)
and exactly one action, `[speed_scale]`. Older 26D/4D and 28D/4D checkpoints
and replay buffers are intentionally incompatible and must not be resumed.
Earlier 54D full-circle checkpoints are also incompatible: the front-only
observation has the same shape but different angular meaning. MPPI reverse
motion and Nav2's automatic BackUp recovery are disabled because the fixed
LiDAR mount cannot observe behind the robot.
The forward +/-60-degree arc is the initial simulation assumption; measure the
actual usable arc on the physical scanner before transferring a policy.

The SAC speed-scale policy runs at 1 Hz while MPPI remains the high-rate local
controller and the independent TTC shield runs at 5 Hz.  The default episode
horizon is 180 steps (about 180 simulated seconds).  Dense rewards are scaled
by the action interval; progress, goal, abort and collision event rewards are
not.  The supervisor watchdog remains at 2.5 seconds, so it detects a genuinely
stale SAC process without releasing the limit between normal 1 Hz updates.
The TTC shield compares forward LiDAR sectors over 0.35 seconds after aligning
the older scan for robot yaw.  It requires adjacent sectors and three
consecutive scans before limiting speed from TTC; a dangerously close return
or a blind scan still activates the shield immediately.  The 24 TTC values in
the SAC observation come from this rotation-compensated sector estimate.  This change
uses the v4 contract, so earlier v3 checkpoints and replay buffers are rejected.

## Issue 02: headless and accelerated simulation

The physics timestep remains `0.002 s`. Simulation speed is controlled only by
wall-time pacing, so sensor and controller rates remain defined in simulation
time.

GUI baseline at 1x:

```bash
ros2 launch agv_mujoco_sim mujoco_sim.launch.py \
  headless:=false real_time_factor:=1.0
```

Headless at 1x or 4x:

```bash
ros2 launch agv_mujoco_sim mujoco_sim.launch.py \
  headless:=true real_time_factor:=1.0

ros2 launch agv_mujoco_sim mujoco_sim.launch.py \
  headless:=true real_time_factor:=4.0
```

Uncapped throughput test:

```bash
ros2 launch agv_mujoco_sim mujoco_sim.launch.py \
  headless:=true real_time_factor:=0.0
```

Inspect effective RTF and timestamp progression directly from `/clock`:

```bash
ros2 topic hz /clock
ros2 topic echo /clock --once
```

`real_time_factor` must be finite and non-negative. A value of `0` means no
wall-time sleeping; actual throughput is then limited by MuJoCo, LiDAR ray
casting, ROS 2 communication, and the other nodes consuming `/clock`.

## Issue 03: behavior-level domain randomization

Issue 03 is scoped to mismatches visible to the SAC/MPPI interface:

- LiDAR range noise and dropout;
- linear/angular wheel-odometry scale;
- command latency;
- aggregate velocity-response scale.

Detailed motor torque, caster contact, wheel hardness, driver mode, and battery
parameters are intentionally not part of this issue. The shipped configuration
contains conservative provisional ranges but keeps every group and the master
switch disabled. Therefore the nominal simulator remains unchanged until a
training launch explicitly opts in.

Pure-Python validation:

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

Inspect the active domain-randomization state after launch:

```bash
ros2 topic echo /domain_randomization/state --once
```
