#!/usr/bin/env python3
"""Measure how well fixed-parameter MPPI uses its configured speed ceiling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from rcl_interfaces.srv import GetParameters
from rclpy.executors import MultiThreadedExecutor

from agv_rl_env import AgvRosInterface, normalize_angle


TERMINAL_STATUSES = {
    GoalStatus.STATUS_SUCCEEDED: "succeeded",
    GoalStatus.STATUS_ABORTED: "aborted",
    GoalStatus.STATUS_CANCELED: "canceled",
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Run repeatable fixed-MPPI trials and measure actual speed "
            "utilization in straight, clear-space cruise samples."
        )
    )
    parser.add_argument("--goal-x", type=float, default=9.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--sample-period-s", type=float, default=0.05)
    parser.add_argument("--cruise-min-elapsed-s", type=float, default=2.0)
    parser.add_argument("--cruise-min-clearance-m", type=float, default=2.0)
    parser.add_argument("--cruise-min-goal-distance-m", type=float, default=2.0)
    parser.add_argument("--cruise-max-bearing-deg", type=float, default=10.0)
    parser.add_argument("--cruise-max-angular-rad-s", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results") / f"mppi_baseline_{stamp}.json",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    finite_values = (
        args.goal_x,
        args.goal_y,
        args.goal_yaw,
        args.timeout_s,
        args.sample_period_s,
        args.cruise_min_elapsed_s,
        args.cruise_min_clearance_m,
        args.cruise_min_goal_distance_m,
        args.cruise_max_bearing_deg,
        args.cruise_max_angular_rad_s,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("All numeric arguments must be finite")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.timeout_s <= 0.0 or args.sample_period_s <= 0.0:
        raise ValueError("Timeout and sample period must be positive")
    if args.cruise_min_clearance_m < 0.0:
        raise ValueError("Cruise clearance must be non-negative")
    if args.cruise_min_elapsed_s < 0.0:
        raise ValueError("Cruise elapsed time must be non-negative")
    if args.cruise_min_goal_distance_m < 0.0:
        raise ValueError("Cruise goal distance must be non-negative")
    if args.cruise_max_bearing_deg < 0.0:
        raise ValueError("Cruise bearing threshold must be non-negative")
    if args.cruise_max_angular_rad_s < 0.0:
        raise ValueError("Cruise angular threshold must be non-negative")


class CommandRecorder:
    """Capture MPPI input and final smoothed velocity commands."""

    def __init__(self, ros: AgvRosInterface) -> None:
        self._lock = threading.Lock()
        self._raw: Optional[Tuple[float, float]] = None
        self._final: Optional[Tuple[float, float]] = None
        self._raw_count = 0
        self._final_count = 0
        self._raw_subscription = ros.create_subscription(
            Twist,
            "/cmd_vel_nav",
            self._raw_callback,
            20,
        )
        self._final_subscription = ros.create_subscription(
            Twist,
            "/cmd_vel",
            self._final_callback,
            20,
        )

    def _raw_callback(self, msg: Twist) -> None:
        with self._lock:
            self._raw = (float(msg.linear.x), float(msg.angular.z))
            self._raw_count += 1

    def _final_callback(self, msg: Twist) -> None:
        with self._lock:
            self._final = (float(msg.linear.x), float(msg.angular.z))
            self._final_count += 1

    def snapshot(self) -> Dict[str, Optional[float]]:
        with self._lock:
            return {
                "mppi_cmd_vx": None if self._raw is None else self._raw[0],
                "mppi_cmd_wz": None if self._raw is None else self._raw[1],
                "final_cmd_vx": None if self._final is None else self._final[0],
                "final_cmd_wz": None if self._final is None else self._final[1],
                "mppi_cmd_count": self._raw_count,
                "final_cmd_count": self._final_count,
            }


def read_double_parameter(
    ros: AgvRosInterface,
    client,
    name: str,
    timeout: float = 5.0,
    attempts: int = 3,
) -> float:
    last_error: Optional[Exception] = None
    response = None
    for attempt in range(1, attempts + 1):
        request = GetParameters.Request()
        request.names = [name]
        try:
            response = ros._wait_future(  # Reuse the bounded ROS wait.
                client.call_async(request),
                timeout,
                f"/controller_server/get_parameters ({name}), "
                f"attempt {attempt}/{attempts}",
            )
            break
        except TimeoutError as error:
            last_error = error
            if attempt < attempts:
                print(
                    f"[benchmark] Parameter request timed out; retrying "
                    f"({attempt}/{attempts})",
                    flush=True,
                )
    if response is None:
        raise RuntimeError(
            f"Controller parameter request failed after {attempts} attempts"
        ) from last_error
    if len(response.values) != 1:
        raise RuntimeError(f"Controller did not return parameter {name}")
    value = float(response.values[0].double_value)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"Invalid controller parameter {name}={value}")
    return value


def hard_reset(ros: AgvRosInterface) -> None:
    ros.cancel_navigation()
    ros.reset_simulation()
    ros.reset_ekf()
    ros.reset_amcl()
    barrier = ros.current_state()
    ros.wait_for_reset_state(
        old_scan_seq=barrier["scan_seq"],
        old_odom_seq=barrier["odom_seq"],
    )
    ros.reset_episode_flags()
    ros.clear_costmaps(timeout=10.0)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_none(values: List[float]) -> Optional[float]:
    return None if not values else float(np.mean(values))


def summarize_trial(
    trial_index: int,
    status: str,
    samples: List[Dict[str, Any]],
    vx_max: float,
    goal: Tuple[float, float, float],
    final_pose: Tuple[float, float, float],
    collision: bool,
    path_length: float,
) -> Dict[str, Any]:
    actual_speeds = [max(0.0, float(row["actual_vx"])) for row in samples]
    cruise_speeds = [
        max(0.0, float(row["actual_vx"]))
        for row in samples
        if bool(row["is_cruise_sample"])
    ]
    cruise_mppi_commands = [
        max(0.0, float(row["mppi_cmd_vx"]))
        for row in samples
        if bool(row["is_cruise_sample"]) and row["mppi_cmd_vx"] is not None
    ]
    cruise_final_commands = [
        max(0.0, float(row["final_cmd_vx"]))
        for row in samples
        if bool(row["is_cruise_sample"]) and row["final_cmd_vx"] is not None
    ]
    elapsed = 0.0 if not samples else float(samples[-1]["elapsed_sim_s"])
    gx, gy, gyaw = goal
    fx, fy, fyaw = final_pose
    final_pose_is_valid = all(math.isfinite(value) for value in final_pose)
    final_position_error = (
        math.hypot(gx - fx, gy - fy) if final_pose_is_valid else None
    )
    final_yaw_error = (
        abs(normalize_angle(gyaw - fyaw)) if final_pose_is_valid else None
    )
    min_clearance = (
        None
        if not samples
        else float(min(float(row["min_lidar_m"]) for row in samples))
    )
    cruise_mean = mean_or_none(cruise_speeds)
    return {
        "trial": trial_index,
        "status": status,
        "success": status == "succeeded",
        "collision": collision,
        "elapsed_sim_s": elapsed,
        "sample_count": len(samples),
        "cruise_sample_count": len(cruise_speeds),
        "actual_speed_mean_m_s": mean_or_none(actual_speeds),
        "actual_speed_p95_m_s": percentile(actual_speeds, 95.0),
        "actual_speed_max_m_s": None if not actual_speeds else max(actual_speeds),
        "cruise_speed_mean_m_s": cruise_mean,
        "cruise_speed_p95_m_s": percentile(cruise_speeds, 95.0),
        "cruise_mppi_command_mean_m_s": mean_or_none(cruise_mppi_commands),
        "cruise_final_command_mean_m_s": mean_or_none(cruise_final_commands),
        "cruise_speed_utilization": (
            None if cruise_mean is None else cruise_mean / vx_max
        ),
        "path_length_m": path_length,
        "minimum_clearance_m": min_clearance,
        "final_position_error_m": final_position_error,
        "final_yaw_error_rad": final_yaw_error,
        "final_pose": [fx, fy, fyaw] if final_pose_is_valid else None,
    }


def run_trial(
    ros: AgvRosInterface,
    commands: CommandRecorder,
    args: argparse.Namespace,
    trial_index: int,
    vx_max: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hard_reset(ros)
    goal = (args.goal_x, args.goal_y, args.goal_yaw)
    ros.send_navigation_goal(*goal)
    start_sim = ros.current_sim_time()
    next_sample_sim = start_sim
    samples: List[Dict[str, Any]] = []
    previous_pose: Optional[Tuple[float, float, float]] = None
    path_length = 0.0
    status = "timeout"

    while rclpy.ok():
        now_sim = ros.current_sim_time()
        elapsed = now_sim - start_sim
        if elapsed >= args.timeout_s:
            status = "timeout"
            break

        state = ros.current_state()
        nav_status = state["nav_status"]
        if nav_status in TERMINAL_STATUSES:
            status = TERMINAL_STATUSES[nav_status]

        if now_sim >= next_sample_sim:
            try:
                x, y, yaw = ros.get_map_pose()
            except Exception:
                time.sleep(0.005)
                continue
            if previous_pose is not None:
                path_length += math.hypot(x - previous_pose[0], y - previous_pose[1])
            previous_pose = (x, y, yaw)

            dx = args.goal_x - x
            dy = args.goal_y - y
            distance = math.hypot(dx, dy)
            bearing = normalize_angle(math.atan2(dy, dx) - yaw)
            command = commands.snapshot()
            min_lidar = float(np.min(state["lidar"]))
            is_cruise = (
                elapsed >= args.cruise_min_elapsed_s
                and distance >= args.cruise_min_goal_distance_m
                and min_lidar >= args.cruise_min_clearance_m
                and abs(math.degrees(bearing)) <= args.cruise_max_bearing_deg
                and abs(float(state["angular_velocity"]))
                <= args.cruise_max_angular_rad_s
            )
            samples.append(
                {
                    "trial": trial_index,
                    "elapsed_sim_s": elapsed,
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                    "distance_to_goal_m": distance,
                    "relative_goal_bearing_rad": bearing,
                    "min_lidar_m": min_lidar,
                    "actual_vx": float(state["linear_velocity"]),
                    "actual_wz": float(state["angular_velocity"]),
                    "mppi_cmd_vx": command["mppi_cmd_vx"],
                    "mppi_cmd_wz": command["mppi_cmd_wz"],
                    "final_cmd_vx": command["final_cmd_vx"],
                    "final_cmd_wz": command["final_cmd_wz"],
                    "is_cruise_sample": is_cruise,
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
    try:
        final_pose = ros.get_map_pose()
    except Exception:
        final_pose = (float("nan"), float("nan"), float("nan"))
    state = ros.current_state()
    summary = summarize_trial(
        trial_index=trial_index,
        status=status,
        samples=samples,
        vx_max=vx_max,
        goal=goal,
        final_pose=final_pose,
        collision=bool(state["collision"]),
        path_length=path_length,
    )
    return summary, samples


def aggregate_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [trial for trial in trials if trial["success"]]
    utilizations = [
        float(trial["cruise_speed_utilization"])
        for trial in trials
        if trial["cruise_speed_utilization"] is not None
    ]
    elapsed = [float(trial["elapsed_sim_s"]) for trial in successful]
    return {
        "trial_count": len(trials),
        "success_count": len(successful),
        "success_rate": len(successful) / len(trials),
        "collision_count": sum(bool(trial["collision"]) for trial in trials),
        "mean_success_time_s": mean_or_none(elapsed),
        "mean_cruise_speed_utilization": mean_or_none(utilizations),
        "speed_limit_only_recommendation": (
            "insufficient_cruise_samples"
            if not utilizations
            else (
                "go"
                if float(np.mean(utilizations)) >= 0.80
                else "tune_mppi_then_retest"
            )
        ),
    }


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


def main() -> int:
    args = parse_args()
    validate_args(args)
    rclpy.init()
    ros = AgvRosInterface()
    commands = CommandRecorder(ros)
    get_parameters_client = ros.create_client(
        GetParameters,
        "/controller_server/get_parameters",
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    trial_summaries: List[Dict[str, Any]] = []
    all_samples: List[Dict[str, Any]] = []
    try:
        ros.wait_until_ready(timeout=30.0)
        if not get_parameters_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Required service unavailable: get_parameters")
        vx_max = read_double_parameter(
            ros,
            get_parameters_client,
            "FollowPath.vx_max",
        )
        print(f"[benchmark] Fixed MPPI vx_max={vx_max:.3f} m/s", flush=True)

        for trial_index in range(1, args.trials + 1):
            print(
                f"[benchmark] Trial {trial_index}/{args.trials}: "
                f"goal=({args.goal_x:.2f}, {args.goal_y:.2f}, "
                f"{args.goal_yaw:.2f})",
                flush=True,
            )
            summary, samples = run_trial(
                ros,
                commands,
                args,
                trial_index,
                vx_max,
            )
            trial_summaries.append(summary)
            all_samples.extend(samples)
            print(
                "[benchmark] "
                f"status={summary['status']} "
                f"time={summary['elapsed_sim_s']:.2f}s "
                f"eta_v={summary['cruise_speed_utilization']} "
                f"clearance={summary['minimum_clearance_m']}",
                flush=True,
            )

        report = {
            "schema_version": 1,
            "mode": "fixed_mppi_baseline",
            "goal": [args.goal_x, args.goal_y, args.goal_yaw],
            "controller": {
                "vx_max_m_s": vx_max,
                "command_topics": commands.snapshot(),
            },
            "cruise_filter": {
                "minimum_elapsed_s": args.cruise_min_elapsed_s,
                "minimum_clearance_m": args.cruise_min_clearance_m,
                "minimum_goal_distance_m": args.cruise_min_goal_distance_m,
                "maximum_bearing_deg": args.cruise_max_bearing_deg,
                "maximum_angular_velocity_rad_s": args.cruise_max_angular_rad_s,
            },
            "trials": trial_summaries,
            "aggregate": aggregate_trials(trial_summaries),
        }
        json_path, csv_path = write_outputs(args.output, report, all_samples)
        print(json.dumps(report["aggregate"], indent=2), flush=True)
        print(f"[benchmark] JSON: {json_path}", flush=True)
        print(f"[benchmark] samples: {csv_path}", flush=True)
        return 0
    finally:
        try:
            ros.cancel_navigation()
        except Exception:
            pass
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        ros.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
