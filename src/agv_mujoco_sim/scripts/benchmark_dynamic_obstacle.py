#!/usr/bin/env python3
"""Evaluate fixed MPPI against an initially unknown crossing obstacle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

from agv_rl_env import AgvRosInterface, normalize_angle
from benchmark_mppi_baseline import (
    CommandRecorder,
    TERMINAL_STATUSES,
    forward_clearance,
    hard_reset,
    mean_or_none,
    percentile,
    read_double_parameter,
    set_double_parameter,
    wait_for_costmap,
)


HIDDEN_OBSTACLE_POSITION = (50.0, 50.0, 0.5)
ROBOT_HALF_LENGTH_M = 0.422
ROBOT_HALF_WIDTH_M = 0.357
OBSTACLE_RADIUS_M = 0.45


def moving_encounter_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure proximity while the obstacle moves, not at its final parked pose."""
    active = [
        row for row in samples
        if row["obstacle_active"] and row["obstacle_center_distance_m"] is not None
    ]
    moving = [row for row in active if row["obstacle_moving"]]
    closest = min(active, key=lambda row: float(row["obstacle_center_distance_m"])) if active else None
    moving_closest = min(
        moving, key=lambda row: float(row["obstacle_center_distance_m"])
    ) if moving else None
    moving_clearances = [
        float(row["estimated_body_clearance_m"])
        for row in moving if row["estimated_body_clearance_m"] is not None
    ]
    return {
        "moving_at_closest_approach": bool(closest and closest["obstacle_moving"]),
        "moving_encounter": bool(
            moving_closest is not None
            and float(moving_closest["obstacle_center_distance_m"]) <= 2.0
        ),
        "closest_approach_time_s": None if closest is None else closest["elapsed_sim_s"],
        "obstacle_y_at_closest_approach_m": None if closest is None else closest["obstacle_y"],
        "moving_closest_approach_time_s": (
            None if moving_closest is None else moving_closest["elapsed_sim_s"]
        ),
        "minimum_moving_obstacle_center_distance_m": (
            None if moving_closest is None
            else float(moving_closest["obstacle_center_distance_m"])
        ),
        "minimum_moving_estimated_body_clearance_m": (
            None if not moving_clearances else min(moving_clearances)
        ),
    }


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Benchmark MPPI with an unknown obstacle crossing its path."
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--scenario-profile", choices=("fixed", "warehouse", "corridor"),
        default="fixed", help="fixed crossing or reproducible varied encounters",
    )
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--goal-x", type=float, default=9.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--trigger-x", type=float, default=2.5)
    parser.add_argument("--obstacle-x", type=float, default=5.0)
    parser.add_argument("--obstacle-start-y", type=float, default=-1.5)
    parser.add_argument("--obstacle-end-y", type=float, default=1.5)
    parser.add_argument("--obstacle-speed-m-s", type=float, default=0.40)
    parser.add_argument("--detection-range-m", type=float, default=3.0)
    parser.add_argument("--release-range-m", type=float, default=3.5)
    parser.add_argument("--release-hold-s", type=float, default=0.5)
    parser.add_argument("--goal-clearance-margin-m", type=float, default=0.5)
    parser.add_argument(
        "--reactive-limit-m-s",
        type=float,
        default=None,
        help="temporary vx_max applied at obstacle detection; disabled by default",
    )
    parser.add_argument(
        "--reactive-speed-limit-m-s",
        type=float,
        default=None,
        help="temporary Nav2 SpeedLimit applied at obstacle detection",
    )
    parser.add_argument(
        "--sham-vx-write",
        action="store_true",
        help="write unchanged vx_max at detection to measure optimizer-reset effects",
    )
    parser.add_argument(
        "--fixed-speed-limit-m-s",
        type=float,
        default=None,
        help="apply a Nav2 SpeedLimit for the complete trial",
    )
    parser.add_argument("--sample-period-s", type=float, default=0.05)
    parser.add_argument(
        "--sac-model", type=Path, default=None,
        help="frozen SAC checkpoint; policy only publishes a speed scale",
    )
    parser.add_argument(
        "--sac-only", action="store_true",
        help="label SAC-only ablation with TTC override and front stop disabled",
    )
    parser.add_argument(
        "--shield-only", action="store_true",
        help="label the full-speed MPPI arm with the same supervisor/safety layers",
    )
    parser.add_argument(
        "--paired-domain-randomization",
        action="store_true",
        help="reset the deterministic DR sequence and resample once per trial",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results") / f"mppi_dynamic_{stamp}.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = (
        args.timeout_s,
        args.goal_x,
        args.goal_y,
        args.trigger_x,
        args.obstacle_x,
        args.obstacle_start_y,
        args.obstacle_end_y,
        args.obstacle_speed_m_s,
        args.detection_range_m,
        args.release_range_m,
        args.release_hold_s,
        args.goal_clearance_margin_m,
        args.sample_period_s,
    )
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("All numeric arguments must be finite")
    if args.timeout_s <= 0.0 or args.sample_period_s <= 0.0:
        raise ValueError("Timeout and sample period must be positive")
    if args.obstacle_speed_m_s <= 0.0 or args.detection_range_m <= 0.0:
        raise ValueError("Obstacle speed and detection range must be positive")
    if args.release_range_m <= args.detection_range_m:
        raise ValueError("--release-range-m must exceed --detection-range-m")
    if args.release_hold_s <= 0.0:
        raise ValueError("--release-hold-s must be positive")
    if args.goal_clearance_margin_m < 0.0:
        raise ValueError("--goal-clearance-margin-m must be non-negative")
    if args.reactive_limit_m_s is not None and (
        not math.isfinite(args.reactive_limit_m_s)
        or args.reactive_limit_m_s <= 0.0
    ):
        raise ValueError("--reactive-limit-m-s must be finite and positive")
    if args.reactive_speed_limit_m_s is not None and (
        not math.isfinite(args.reactive_speed_limit_m_s)
        or args.reactive_speed_limit_m_s <= 0.0
    ):
        raise ValueError("--reactive-speed-limit-m-s must be finite and positive")
    if args.fixed_speed_limit_m_s is not None and (
        not math.isfinite(args.fixed_speed_limit_m_s)
        or args.fixed_speed_limit_m_s <= 0.0
    ):
        raise ValueError("--fixed-speed-limit-m-s must be finite and positive")
    intervention_count = sum(
        (
            args.reactive_limit_m_s is not None,
            args.reactive_speed_limit_m_s is not None,
            args.sham_vx_write,
            args.fixed_speed_limit_m_s is not None,
        )
    )
    if intervention_count > 1:
        raise ValueError("Select only one reactive intervention")
    if args.sac_model is not None and (intervention_count or args.shield_only):
        raise ValueError("SAC cannot be combined with another intervention")
    if args.sac_only and args.sac_model is None:
        raise ValueError("--sac-only requires --sac-model")
    if args.obstacle_end_y == args.obstacle_start_y:
        raise ValueError("Obstacle start and end y must differ")


def stratified_values(
    rng: random.Random, count: int, minimum: float, maximum: float
) -> List[float]:
    """Cover the full range without relying on 100 near-identical draws."""
    values = [
        minimum + (maximum - minimum) * (index + rng.random()) / count
        for index in range(count)
    ]
    rng.shuffle(values)
    return values


def trial_scenarios(args: argparse.Namespace) -> List[Dict[str, float]]:
    if args.scenario_profile == "fixed":
        return [{
            "trigger_x": args.trigger_x,
            "obstacle_x": args.obstacle_x,
            "obstacle_start_y": args.obstacle_start_y,
            "obstacle_end_y": args.obstacle_end_y,
            "obstacle_speed_m_s": args.obstacle_speed_m_s,
        } for _ in range(args.trials)]

    rng = random.Random(args.scenario_seed)
    count = args.trials
    obstacle_x = stratified_values(rng, count, 4.7, 5.3)
    if args.scenario_profile == "warehouse":
        gaps = stratified_values(rng, count, 1.7, 2.8)
        negative_y = stratified_values(rng, count, -1.5, -1.0)
        positive_y = stratified_values(rng, count, 1.0, 1.5)
        speeds = stratified_values(rng, count, 0.25, 0.55)
    else:
        # The floor-2 corridor's free strip at x=5 is about y=-1.1..0.8.
        # Keep the 0.45 m-radius obstacle wholly inside that strip.
        gaps = stratified_values(rng, count, 1.5, 2.1)
        negative_y = stratified_values(rng, count, -0.62, -0.48)
        positive_y = stratified_values(rng, count, 0.18, 0.32)
        speeds = stratified_values(rng, count, 0.18, 0.35)
    directions = [index % 2 == 0 for index in range(count)]
    rng.shuffle(directions)
    return [{
        "trigger_x": obstacle_x[index] - gaps[index],
        "obstacle_x": obstacle_x[index],
        "obstacle_start_y": (
            negative_y[index] if directions[index] else positive_y[index]
        ),
        "obstacle_end_y": (
            positive_y[index] if directions[index] else negative_y[index]
        ),
        "obstacle_speed_m_s": speeds[index],
    } for index in range(count)]


class DynamicObstaclePublisher:
    def __init__(self, ros: AgvRosInterface) -> None:
        self.ros = ros
        self.lock = threading.Lock()
        self.active = False
        self.start_sim = 0.0
        self.obstacle_x = HIDDEN_OBSTACLE_POSITION[0]
        self.start_y = HIDDEN_OBSTACLE_POSITION[1]
        self.end_y = HIDDEN_OBSTACLE_POSITION[1]
        self.speed_m_s = 0.0
        self.current_y = HIDDEN_OBSTACLE_POSITION[1]
        self.publisher = ros.create_publisher(
            Point,
            "/dynamic_obstacle/position",
            10,
        )
        self.timer = ros.create_timer(0.02, self._publish_trajectory)
        self.speed_limit_publisher = ros.create_publisher(
            SpeedLimit,
            "/speed_limit",
            10,
        )

    def move(self, x: float, y: float, z: float = 0.5) -> None:
        message = Point()
        message.x = float(x)
        message.y = float(y)
        message.z = float(z)
        self.publisher.publish(message)

    def hide(self) -> None:
        with self.lock:
            self.active = False
            self.current_y = HIDDEN_OBSTACLE_POSITION[1]
        self.move(*HIDDEN_OBSTACLE_POSITION)

    def start_crossing(
        self,
        start_sim: float,
        obstacle_x: float,
        start_y: float,
        end_y: float,
        speed_m_s: float,
    ) -> None:
        with self.lock:
            self.active = True
            self.start_sim = float(start_sim)
            self.obstacle_x = float(obstacle_x)
            self.start_y = float(start_y)
            self.end_y = float(end_y)
            self.speed_m_s = float(speed_m_s)
            self.current_y = float(start_y)
        self.move(obstacle_x, start_y)

    def snapshot(self) -> Tuple[bool, float, float]:
        with self.lock:
            return self.active, self.obstacle_x, self.current_y

    def _publish_trajectory(self) -> None:
        with self.lock:
            if not self.active:
                return
            elapsed = max(0.0, self.ros.current_sim_time() - self.start_sim)
            direction = 1.0 if self.end_y > self.start_y else -1.0
            travelled = self.speed_m_s * elapsed
            distance = abs(self.end_y - self.start_y)
            self.current_y = self.start_y + direction * min(travelled, distance)
            obstacle_x = self.obstacle_x
            obstacle_y = self.current_y
        self.move(obstacle_x, obstacle_y)

    def set_speed_limit(self, limit_m_s: Optional[float], vx_max: float) -> None:
        message = SpeedLimit()
        message.percentage = True
        message.speed_limit = (
            0.0
            if limit_m_s is None
            else 100.0 * min(float(limit_m_s), vx_max) / vx_max
        )
        self.speed_limit_publisher.publish(message)


class GroundTruthRecorder:
    def __init__(self, ros: AgvRosInterface) -> None:
        self.lock = threading.Lock()
        self.pose: Optional[Tuple[float, float, float]] = None
        self.subscription = ros.create_subscription(
            Odometry,
            "/ground_truth/odom",
            self._callback,
            10,
        )

    def _callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        rotation = message.pose.pose.orientation
        sin_yaw = 2.0 * (rotation.w * rotation.z + rotation.x * rotation.y)
        cos_yaw = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
        with self.lock:
            self.pose = (
                float(position.x),
                float(position.y),
                math.atan2(sin_yaw, cos_yaw),
            )

    def snapshot(self) -> Tuple[float, float, float]:
        with self.lock:
            if self.pose is None:
                raise RuntimeError("Ground-truth odometry is not ready")
            return self.pose


def estimated_body_clearance(
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    obstacle_x: float,
    obstacle_y: float,
) -> float:
    """Estimate circle clearance to the robot's oriented box footprint."""
    dx = obstacle_x - robot_x
    dy = obstacle_y - robot_y
    local_x = math.cos(robot_yaw) * dx + math.sin(robot_yaw) * dy
    local_y = -math.sin(robot_yaw) * dx + math.cos(robot_yaw) * dy
    outside_x = max(abs(local_x) - ROBOT_HALF_LENGTH_M, 0.0)
    outside_y = max(abs(local_y) - ROBOT_HALF_WIDTH_M, 0.0)
    return math.hypot(outside_x, outside_y) - OBSTACLE_RADIUS_M


def write_outputs(
    output_path: Path,
    report: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output_path.with_suffix(".samples.csv")
    fieldnames = list(samples[0].keys()) if samples else ["trial"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    return output_path, csv_path


def resample_domain_randomization(ros: AgvRosInterface) -> Dict[str, Any]:
    before = ros.current_state()
    future = ros.randomization_client.call_async(Trigger.Request())
    response = ros._wait_future(future, 3.0, "domain randomization")
    if not response.success:
        raise RuntimeError(response.message)

    deadline = time.monotonic() + 3.0
    while rclpy.ok() and time.monotonic() < deadline:
        state = ros.current_state()
        if (
            state["scan_seq"] > before["scan_seq"]
            and state["odom_seq"] > before["odom_seq"]
        ):
            return json.loads(response.message)
        time.sleep(0.005)
    raise TimeoutError("No fresh sensor samples after domain randomization")


def run_trial(
    ros: AgvRosInterface,
    commands: CommandRecorder,
    obstacle: DynamicObstaclePublisher,
    ground_truth: GroundTruthRecorder,
    args: argparse.Namespace,
    trial_index: int,
    vx_max: float,
    policy_env=None,
    policy_model=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hard_reset(ros)
    randomization_state = (
        resample_domain_randomization(ros)
        if args.paired_domain_randomization
        else None
    )
    obstacle.hide()
    if policy_env is not None:
        policy_env.ros.publish_speed_scale(1.0)
    ros.send_navigation_goal(args.goal_x, args.goal_y, 0.0)
    if args.fixed_speed_limit_m_s is not None:
        obstacle.set_speed_limit(args.fixed_speed_limit_m_s, vx_max)
    start_sim = ros.current_sim_time()
    next_sample_sim = start_sim
    next_fixed_limit_publish_sim = start_sim
    next_sac_update_sim = start_sim
    requested_scale = 1.0
    sac_action_count = 0
    obstacle_start_sim: Optional[float] = None
    detection_sim: Optional[float] = None
    braking_onset_sim: Optional[float] = None
    deceleration_sim: Optional[float] = None
    reactive_limit_active = False
    reactive_limit_was_applied = False
    clear_since_sim: Optional[float] = None
    intervention_count = 0
    false_intervention_count = 0
    samples: List[Dict[str, Any]] = []
    status = "timeout"

    while rclpy.ok():
        now_sim = ros.current_sim_time()
        elapsed = now_sim - start_sim
        if (
            args.fixed_speed_limit_m_s is not None
            and now_sim >= next_fixed_limit_publish_sim
        ):
            obstacle.set_speed_limit(args.fixed_speed_limit_m_s, vx_max)
            next_fixed_limit_publish_sim = now_sim + 0.1
        if elapsed >= args.timeout_s:
            break

        if policy_model is not None and now_sim >= next_sac_update_sim:
            observation, _ = policy_env._observation()
            action, _ = policy_model.predict(observation, deterministic=True)
            requested_scale = float(policy_env._apply_action(action)[0])
            sac_action_count += 1
            next_sac_update_sim = now_sim + policy_env.STEP_WAIT_DURATION

        state = ros.current_state()
        try:
            nav_x, nav_y, nav_yaw = ros.get_map_pose()
            x, y, yaw = ground_truth.snapshot()
        except Exception:
            time.sleep(0.005)
            continue

        if obstacle_start_sim is None and x >= args.trigger_x:
            obstacle_start_sim = now_sim
            obstacle.start_crossing(
                start_sim=now_sim,
                obstacle_x=args.obstacle_x,
                start_y=args.obstacle_start_y,
                end_y=args.obstacle_end_y,
                speed_m_s=args.obstacle_speed_m_s,
            )

        obstacle_active, obstacle_x, obstacle_y = obstacle.snapshot()

        front_range = forward_clearance(state["lidar"])
        distance_to_goal = math.hypot(args.goal_x - x, args.goal_y - y)
        obstacle_precedes_goal = (
            front_range + args.goal_clearance_margin_m < distance_to_goal
        )
        hazard_detected = (
            front_range <= args.detection_range_m and obstacle_precedes_goal
        )
        if detection_sim is None and hazard_detected:
            detection_sim = now_sim
        can_react = (
            args.reactive_limit_m_s is not None
            or args.reactive_speed_limit_m_s is not None
            or (args.sham_vx_write and not reactive_limit_was_applied)
        )
        if hazard_detected and can_react and not reactive_limit_active:
            intervention_count += 1
            if obstacle_start_sim is None:
                false_intervention_count += 1
            if args.reactive_limit_m_s is not None:
                set_double_parameter(
                    ros,
                    ros.mppi_parameter_client,
                    "FollowPath.vx_max",
                    min(args.reactive_limit_m_s, vx_max),
                )
                reactive_limit_active = True
                reactive_limit_was_applied = True
            elif args.reactive_speed_limit_m_s is not None:
                obstacle.set_speed_limit(args.reactive_speed_limit_m_s, vx_max)
                reactive_limit_active = True
                reactive_limit_was_applied = True
            elif args.sham_vx_write and not reactive_limit_was_applied:
                set_double_parameter(
                    ros,
                    ros.mppi_parameter_client,
                    "FollowPath.vx_max",
                    vx_max,
                )
                reactive_limit_was_applied = True
        if reactive_limit_active and args.fixed_speed_limit_m_s is None:
            clear_detected = (
                front_range >= args.release_range_m
                or not obstacle_precedes_goal
            )
            if clear_detected:
                if clear_since_sim is None:
                    clear_since_sim = now_sim
                elif now_sim - clear_since_sim >= args.release_hold_s:
                    if args.reactive_speed_limit_m_s is not None:
                        obstacle.set_speed_limit(None, vx_max)
                    else:
                        set_double_parameter(
                            ros,
                            ros.mppi_parameter_client,
                            "FollowPath.vx_max",
                            vx_max,
                        )
                    reactive_limit_active = False
                    clear_since_sim = None
            else:
                clear_since_sim = None
        if (
            detection_sim is not None
            and braking_onset_sim is None
        ):
            command = commands.snapshot()
            final_vx = command["final_cmd_vx"]
            if final_vx is not None and float(final_vx) <= 0.95 * vx_max:
                braking_onset_sim = now_sim
        if (
            detection_sim is not None
            and deceleration_sim is None
            and float(state["linear_velocity"]) <= 0.8 * vx_max
        ):
            deceleration_sim = now_sim

        nav_status = state["nav_status"]
        if nav_status in TERMINAL_STATUSES:
            status = TERMINAL_STATUSES[nav_status]

        if now_sim >= next_sample_sim:
            dx = args.goal_x - x
            dy = args.goal_y - y
            command = commands.snapshot()
            obstacle_center_distance = (
                math.hypot(obstacle_x - x, obstacle_y - y)
                if obstacle_active
                else None
            )
            body_clearance = (
                estimated_body_clearance(
                    x,
                    y,
                    yaw,
                    obstacle_x,
                    obstacle_y,
                )
                if obstacle_active
                else None
            )
            samples.append(
                {
                    "trial": trial_index,
                    "elapsed_sim_s": elapsed,
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                    "nav_x": nav_x,
                    "nav_y": nav_y,
                    "nav_yaw": nav_yaw,
                    "distance_to_goal_m": math.hypot(dx, dy),
                    "goal_bearing_rad": normalize_angle(
                        math.atan2(dy, dx) - yaw
                    ),
                    "actual_vx": float(state["linear_velocity"]),
                    "actual_wz": float(state["angular_velocity"]),
                    "sac_requested_scale": (
                        requested_scale if policy_model is not None else None
                    ),
                    "supervisor_applied_scale": float(state["applied_speed_scale"]),
                    "safety_override_active": bool(state["safety_override_active"]),
                    "front_safety_stop_active": bool(state["front_safety_stop_active"]),
                    "mppi_cmd_vx": command["mppi_cmd_vx"],
                    "mppi_cmd_wz": command["mppi_cmd_wz"],
                    "final_cmd_vx": command["final_cmd_vx"],
                    "final_cmd_wz": command["final_cmd_wz"],
                    "front_lidar_m": front_range,
                    "min_lidar_m": float(np.min(state["lidar"])),
                    "obstacle_active": obstacle_active,
                    "obstacle_moving": bool(
                        obstacle_active
                        and obstacle_start_sim is not None
                        and now_sim - obstacle_start_sim
                        < abs(args.obstacle_end_y - args.obstacle_start_y)
                        / args.obstacle_speed_m_s
                    ),
                    "obstacle_x": (
                        obstacle_x if obstacle_active else None
                    ),
                    "obstacle_y": obstacle_y if obstacle_active else None,
                    "obstacle_center_distance_m": obstacle_center_distance,
                    "estimated_body_clearance_m": body_clearance,
                }
            )
            next_sample_sim += args.sample_period_s

        if bool(state["collision"]):
            status = "collision"
            break
        if nav_status in TERMINAL_STATUSES:
            break
        time.sleep(0.005)

    if status == "timeout":
        ros.cancel_navigation()
    obstacle.hide()
    if policy_env is not None:
        policy_env.ros.publish_speed_scale(1.0)
    if args.fixed_speed_limit_m_s is not None:
        obstacle.set_speed_limit(None, vx_max)
    if reactive_limit_active:
        if args.reactive_speed_limit_m_s is not None:
            obstacle.set_speed_limit(None, vx_max)
        else:
            set_double_parameter(
                ros,
                ros.mppi_parameter_client,
                "FollowPath.vx_max",
                vx_max,
            )
    state = ros.current_state()
    active_samples = [row for row in samples if row["obstacle_active"]]
    actual_speeds = [max(0.0, float(row["actual_vx"])) for row in samples]
    clear_speeds = [
        max(0.0, float(row["actual_vx"])) for row in samples
        if 0.5 <= float(row["x"]) < args.trigger_x
        and not row["obstacle_active"]
    ]
    encounter_metrics = moving_encounter_metrics(samples)
    center_distances = [
        float(row["obstacle_center_distance_m"])
        for row in active_samples
        if row["obstacle_center_distance_m"] is not None
    ]
    minimum_center_distance = (
        None if not center_distances else min(center_distances)
    )
    body_clearances = [
        float(row["estimated_body_clearance_m"])
        for row in active_samples
        if row["estimated_body_clearance_m"] is not None
    ]
    minimum_body_clearance = (
        None if not body_clearances else min(body_clearances)
    )
    summary = {
        "trial": trial_index,
        "status": status,
        "success": status == "succeeded",
        "collision": bool(state["collision"]) or status == "collision",
        "elapsed_sim_s": 0.0 if not samples else samples[-1]["elapsed_sim_s"],
        "obstacle_triggered": obstacle_start_sim is not None,
        "reactive_limit_was_applied": reactive_limit_was_applied,
        "intervention_count": intervention_count,
        "false_intervention_count": false_intervention_count,
        "sac_action_count": sac_action_count,
        "mean_sac_requested_scale": mean_or_none([
            float(row["sac_requested_scale"]) for row in samples
            if row["sac_requested_scale"] is not None
        ]),
        "mean_supervisor_applied_scale": mean_or_none([
            float(row["supervisor_applied_scale"]) for row in samples
        ]),
        "front_safety_stop_sample_count": sum(
            bool(row["front_safety_stop_active"]) for row in samples
        ),
        "domain_randomization": randomization_state,
        "obstacle_trigger_time_s": (
            None if obstacle_start_sim is None else obstacle_start_sim - start_sim
        ),
        "detection_delay_from_trigger_s": (
            None
            if detection_sim is None or obstacle_start_sim is None
            else detection_sim - obstacle_start_sim
        ),
        "deceleration_delay_from_detection_s": (
            None
            if deceleration_sim is None or detection_sim is None
            else max(0.0, deceleration_sim - detection_sim)
        ),
        "braking_onset_delay_from_detection_s": (
            None
            if braking_onset_sim is None or detection_sim is None
            else max(0.0, braking_onset_sim - detection_sim)
        ),
        "actual_speed_mean_m_s": mean_or_none(actual_speeds),
        "actual_speed_p95_m_s": percentile(actual_speeds, 95.0),
        "clear_segment_speed_mean_m_s": mean_or_none(clear_speeds),
        **encounter_metrics,
        "minimum_front_clearance_m": (
            None
            if not active_samples
            else min(float(row["front_lidar_m"]) for row in active_samples)
        ),
        "minimum_obstacle_center_distance_m": minimum_center_distance,
        "minimum_estimated_body_clearance_m": minimum_body_clearance,
    }
    return summary, samples


def aggregate_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [trial for trial in trials if trial["success"]]
    moving_encounters = [trial for trial in trials if trial["moving_encounter"]]
    braking_delays = [
        float(trial["braking_onset_delay_from_detection_s"])
        for trial in trials
        if trial["braking_onset_delay_from_detection_s"] is not None
    ]
    response_delays = [
        float(trial["deceleration_delay_from_detection_s"])
        for trial in trials
        if trial["deceleration_delay_from_detection_s"] is not None
    ]
    clearances = [
        float(trial["minimum_estimated_body_clearance_m"])
        for trial in trials
        if trial["minimum_estimated_body_clearance_m"] is not None
    ]
    return {
        "trial_count": len(trials),
        "success_rate": len(successful) / len(trials),
        "collision_count": sum(bool(trial["collision"]) for trial in trials),
        "timeout_count": sum(trial["status"] == "timeout" for trial in trials),
        "moving_encounter_count": len(moving_encounters),
        "moving_encounter_collision_count": sum(
            bool(trial["collision"]) for trial in moving_encounters
        ),
        "mean_clear_segment_speed_m_s": mean_or_none([
            float(trial["clear_segment_speed_mean_m_s"])
            for trial in trials
            if trial["clear_segment_speed_mean_m_s"] is not None
        ]),
        "mean_success_time_s": mean_or_none(
            [float(trial["elapsed_sim_s"]) for trial in successful]
        ),
        "mean_sac_requested_scale": mean_or_none([
            float(trial["mean_sac_requested_scale"])
            for trial in trials if trial["mean_sac_requested_scale"] is not None
        ]),
        "mean_supervisor_applied_scale": mean_or_none([
            float(trial["mean_supervisor_applied_scale"])
            for trial in trials
            if trial["mean_supervisor_applied_scale"] is not None
        ]),
        "front_safety_stop_sample_count": sum(
            int(trial["front_safety_stop_sample_count"]) for trial in trials
        ),
        "mean_braking_onset_delay_s": mean_or_none(braking_delays),
        "mean_deceleration_delay_s": mean_or_none(response_delays),
        "minimum_estimated_body_clearance_m": (
            None if not clearances else min(clearances)
        ),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    scenarios = trial_scenarios(args)
    policy_env = policy_model = model_contract = None
    if args.sac_model is not None:
        from stable_baselines3 import SAC
        from agv_rl_env import AgvRlEnv
        from train_sac import validate_model

        model_path = args.sac_model.expanduser().resolve(strict=True)
        policy_model = SAC.load(str(model_path), device="cpu")
        model_contract = validate_model(policy_model, sac_only=args.sac_only)
        policy_env = AgvRlEnv(
            goal=(args.goal_x, args.goal_y, 0.0),
            goal_sampling="fixed",
            include_ttc=not args.sac_only,
        )
        ros = policy_env.ros
        executor = spin_thread = None
    else:
        rclpy.init()
        ros = AgvRosInterface()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(ros)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
    commands = CommandRecorder(ros)
    obstacle = DynamicObstaclePublisher(ros)
    ground_truth = GroundTruthRecorder(ros)
    parameter_client = ros.create_client(
        GetParameters,
        "/controller_server/get_parameters",
    )
    trials: List[Dict[str, Any]] = []
    all_samples: List[Dict[str, Any]] = []
    nominal_vx_max: Optional[float] = None
    try:
        ros.wait_until_ready(
            timeout=30.0, require_supervisor=policy_model is not None
        )
        wait_for_costmap(ros)
        if not ros.is_position_free(args.goal_x, args.goal_y):
            raise RuntimeError("Dynamic benchmark goal is occupied")
        if not parameter_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Controller parameter service unavailable")
        vx_max = read_double_parameter(
            ros, parameter_client, "FollowPath.vx_max"
        )
        nominal_vx_max = vx_max
        vx_std = read_double_parameter(
            ros, parameter_client, "FollowPath.vx_std"
        )
        print(
            f"[dynamic] MPPI vx_max={vx_max:.3f}, vx_std={vx_std:.3f}",
            flush=True,
        )
        if args.paired_domain_randomization:
            future = ros.randomization_sequence_reset_client.call_async(
                Trigger.Request()
            )
            response = ros._wait_future(
                future,
                3.0,
                "domain-randomization sequence reset",
            )
            if not response.success:
                raise RuntimeError(response.message)
        for trial_index in range(1, args.trials + 1):
            scenario = scenarios[trial_index - 1]
            trial_args = argparse.Namespace(**vars(args))
            vars(trial_args).update(scenario)
            validate_args(trial_args)
            print(
                f"[dynamic] Trial {trial_index}/{args.trials}: {scenario}",
                flush=True,
            )
            summary, samples = run_trial(
                ros,
                commands,
                obstacle,
                ground_truth,
                trial_args,
                trial_index,
                vx_max,
                policy_env,
                policy_model,
            )
            summary["scenario"] = scenario
            trials.append(summary)
            all_samples.extend(samples)
            progress_path = args.output.expanduser().resolve().with_suffix(
                ".progress.json"
            )
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps({
                "scenario_profile": args.scenario_profile,
                "scenario_seed": args.scenario_seed,
                "completed_trials": len(trials),
                "planned_trials": args.trials,
                "trials": trials,
                "aggregate_so_far": aggregate_trials(trials),
            }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(
                f"[dynamic] status={summary['status']} "
                f"time={summary['elapsed_sim_s']:.2f}s "
                f"response={summary['deceleration_delay_from_detection_s']} "
                f"clearance={summary['minimum_estimated_body_clearance_m']}",
                flush=True,
            )
        report = {
            "schema_version": 1,
            "mode": (
                "mppi_sac_speed_cap_only" if args.sac_only
                else "mppi_sac_speed_cap" if policy_model is not None
                else "mppi_with_shared_shield" if args.shield_only
                else "fixed_mppi_dynamic_crossing"
            ),
            "sac_model": (
                None if args.sac_model is None else str(model_path)
            ),
            "sac_model_contract": model_contract,
            "controller": {"vx_max_m_s": vx_max, "vx_std_m_s": vx_std},
            "scenario": {
                "profile": args.scenario_profile,
                "seed": args.scenario_seed,
                "goal": [args.goal_x, args.goal_y, 0.0],
                "trigger_x": args.trigger_x,
                "obstacle_start": [
                    args.obstacle_x,
                    args.obstacle_start_y,
                ],
                "obstacle_end_y": args.obstacle_end_y,
                "obstacle_speed_m_s": args.obstacle_speed_m_s,
                "reactive_limit_m_s": args.reactive_limit_m_s,
                "reactive_speed_limit_m_s": args.reactive_speed_limit_m_s,
                "detection_range_m": args.detection_range_m,
                "release_range_m": args.release_range_m,
                "release_hold_s": args.release_hold_s,
                "goal_clearance_margin_m": args.goal_clearance_margin_m,
                "sham_vx_write": args.sham_vx_write,
                "fixed_speed_limit_m_s": args.fixed_speed_limit_m_s,
                "speed_limit_subscriber_count": (
                    obstacle.speed_limit_publisher.get_subscription_count()
                ),
                "paired_domain_randomization": (
                    args.paired_domain_randomization
                ),
            },
            "trials": trials,
            "aggregate": aggregate_trials(trials),
        }
        json_path, csv_path = write_outputs(args.output, report, all_samples)
        print(json.dumps(report["aggregate"], indent=2), flush=True)
        print(f"[dynamic] JSON: {json_path}", flush=True)
        print(f"[dynamic] samples: {csv_path}", flush=True)
        return 0
    finally:
        try:
            obstacle.hide()
            if (
                args.fixed_speed_limit_m_s is not None
                or args.reactive_speed_limit_m_s is not None
            ):
                obstacle.set_speed_limit(None, nominal_vx_max or 1.0)
            ros.cancel_navigation()
            if nominal_vx_max is not None and (
                args.reactive_limit_m_s is not None or args.sham_vx_write
            ):
                set_double_parameter(
                    ros,
                    ros.mppi_parameter_client,
                    "FollowPath.vx_max",
                    nominal_vx_max,
                )
        except Exception:
            pass
        if policy_env is not None:
            policy_env.close()
        else:
            executor.shutdown(timeout_sec=2.0)
            spin_thread.join(timeout=2.0)
            ros.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
