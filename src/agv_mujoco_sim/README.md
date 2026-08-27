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

## Start Nav2 with the RL supervisor

```bash
ros2 launch agv_mujoco_sim navigation_rl.launch.py
```

The supervisor consumes exactly one value on `/rl_velocity_limits`, clamps it
to `[0.40, 1.00]`, filters/slew-limits it, and uses Nav2's `/speed_limit` API.
If SAC becomes stale, the watchdog removes the adaptive limit and returns to
the verified MPPI baseline.

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

Measure effective RTF, topic rates, and timestamp monotonicity:

```bash
ros2 run agv_mujoco_sim measure_issue02_baseline.py \
  --duration-wall 15 \
  --label headless_4x \
  --output ~/issue02_headless_4x.json
```

Include a reset during the measurement:

```bash
ros2 run agv_mujoco_sim measure_issue02_baseline.py \
  --duration-wall 15 \
  --reset-at-wall 5 \
  --label headless_4x_reset \
  --output ~/issue02_headless_4x_reset.json
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

Runtime reset/diagnostic validation:

```bash
ros2 run agv_mujoco_sim measure_issue03_sampling.py --resets 3
```
