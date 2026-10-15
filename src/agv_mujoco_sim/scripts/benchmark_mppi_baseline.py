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
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParameters
from rclpy.executors import MultiThreadedExecutor

from agv_rl_env import AgvRosInterface, normalize_angle


TERMINAL_STATUSES = {
    GoalStatus.STATUS_SUCCEEDED: "succeeded",
    GoalStatus.STATUS_ABORTED: "aborted",
    GoalStatus.STATUS_CANCELED: "canceled",
}

DECISION_SCENARIOS = (
    ("clear_straight", (9.0, 0.0, 0.0)),
    ("upper_obstacle_detour", (9.0, 5.0, 0.0)),
    ("lower_zigzag", (8.5, -8.5, 0.0)),
)


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
    parser.add_argument(
        "--suite",
        choices=("single", "decision"),
        default="single",
        help="single goal or the three-route MPPI/RL decision suite",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="number of repetitions per scenario",
    )
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
    parser.add_argument(
        "--vx-std-override",
        type=float,
        default=None,
        help="temporary FollowPath.vx_std value, restored on exit",
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
    if args.vx_std_override is not None and (
        not math.isfinite(args.vx_std_override)
        or args.vx_std_override <= 0.0
    ):
        raise ValueError("--vx-std-override must be finite and positive")


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


def set_double_parameter(
    ros: AgvRosInterface,
    client,
    name: str,
    value: float,
    timeout: float = 5.0,
) -> None:
    request = SetParameters.Request()
    request.parameters.append(
        Parameter(
            name=name,
            value=ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE,
                double_value=float(value),
            ),
        )
    )
    response = ros._wait_future(
        client.call_async(request),
        timeout,
        f"/controller_server/set_parameters ({name})",
    )
    if len(response.results) != 1 or not response.results[0].successful:
        reason = (
            response.results[0].reason
            if response.results
            else "missing result"
        )
        raise RuntimeError(f"Controller rejected {name}={value}: {reason}")


def hard_reset(ros: AgvRosInterface, attempts: int = 3) -> None:
    ros.cancel_navigation()
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
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
            return
        except Exception as error:
            last_error = error
            if attempt < attempts:
                print(
                    f"[benchmark] Hard reset failed; retrying "
                    f"({attempt}/{attempts}): {error}",
                    flush=True,
                )
                time.sleep(0.25)
    raise RuntimeError(
        f"Hard reset failed after {attempts} attempts: {last_error}"
    ) from last_error


def wait_for_costmap(ros: AgvRosInterface, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        with ros.lock:
            if ros.costmap is not None:
                return
        time.sleep(0.02)
    raise RuntimeError("No global costmap received for scenario validation")


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_none(values: List[float]) -> Optional[float]:
    return None if not values else float(np.mean(values))


def forward_clearance(lidar: np.ndarray) -> float:
    """Minimum range in the central +/-30 degrees of the forward 120-degree scan."""
    beam_count = int(lidar.size)
    if beam_count < 4:
        return float(np.min(lidar))
    center = beam_count // 2
    half_width = beam_count // 4
    return float(np.min(lidar[center - half_width:center + half_width]))


def summarize_trial(
    trial_index: int,
    scenario: str,
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
    min_front_clearance = (
        None
        if not samples
        else float(min(float(row["front_lidar_m"]) for row in samples))
    )
    cruise_mean = mean_or_none(cruise_speeds)
    near_obstacle_speeds = [
        max(0.0, float(row["actual_vx"]))
        for row in samples
        if float(row["min_lidar_m"]) <= 1.0
    ]
    near_forward_obstacle_speeds = [
        max(0.0, float(row["actual_vx"]))
        for row in samples
        if float(row["front_lidar_m"]) <= 1.0
    ]
    final_commands = [
        float(row["final_cmd_vx"])
        for row in samples
        if row["final_cmd_vx"] is not None
    ]
    command_total_variation = sum(
        abs(current - previous)
        for previous, current in zip(final_commands, final_commands[1:])
    )
    return {
        "trial": trial_index,
        "scenario": scenario,
        "goal": [gx, gy, gyaw],
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
        "near_obstacle_sample_count": len(near_obstacle_speeds),
        "near_obstacle_speed_mean_m_s": mean_or_none(near_obstacle_speeds),
        "near_forward_obstacle_sample_count": len(
            near_forward_obstacle_speeds
        ),
        "near_forward_obstacle_speed_mean_m_s": mean_or_none(
            near_forward_obstacle_speeds
        ),
        "command_total_variation_m_s": command_total_variation,
        "command_variation_rate_m_s2": (
            None if elapsed <= 0.0 else command_total_variation / elapsed
        ),
        "cruise_speed_utilization": (
            None if cruise_mean is None else cruise_mean / vx_max
        ),
        "path_length_m": path_length,
        "minimum_clearance_m": min_clearance,
        "minimum_front_clearance_m": min_front_clearance,
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
    scenario: str,
    goal: Tuple[float, float, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hard_reset(ros)
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

            dx = goal[0] - x
            dy = goal[1] - y
            distance = math.hypot(dx, dy)
            bearing = normalize_angle(math.atan2(dy, dx) - yaw)
            command = commands.snapshot()
            min_lidar = float(np.min(state["lidar"]))
            front_lidar = forward_clearance(state["lidar"])
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
                    "scenario": scenario,
                    "elapsed_sim_s": elapsed,
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                    "distance_to_goal_m": distance,
                    "relative_goal_bearing_rad": bearing,
                    "min_lidar_m": min_lidar,
                    "front_lidar_m": front_lidar,
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
        scenario=scenario,
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
    all_scenarios_passed = (
        len(successful) == len(trials)
        and not any(bool(trial["collision"]) for trial in trials)
    )
    aggregate = {
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
                "fix_scenario_failures_before_rl"
                if not all_scenarios_passed
                else (
                    "go"
                    if float(np.mean(utilizations)) >= 0.80
                    else "tune_mppi_then_retest"
                )
            )
        ),
    }
    scenario_names = dict.fromkeys(str(trial["scenario"]) for trial in trials)
    aggregate["by_scenario"] = {
        name: aggregate_trials_without_groups(
            [trial for trial in trials if trial["scenario"] == name]
        )
        for name in scenario_names
    }
    return aggregate


def aggregate_trials_without_groups(
    trials: List[Dict[str, Any]],
) -> Dict[str, Any]:
    successful = [trial for trial in trials if trial["success"]]
    elapsed = [float(trial["elapsed_sim_s"]) for trial in successful]
    clearance = [
        float(trial["minimum_clearance_m"])
        for trial in trials
        if trial["minimum_clearance_m"] is not None
    ]
    front_clearance = [
        float(trial["minimum_front_clearance_m"])
        for trial in trials
        if trial["minimum_front_clearance_m"] is not None
    ]
    near_speed = [
        float(trial["near_obstacle_speed_mean_m_s"])
        for trial in trials
        if trial["near_obstacle_speed_mean_m_s"] is not None
    ]
    near_forward_speed = [
        float(trial["near_forward_obstacle_speed_mean_m_s"])
        for trial in trials
        if trial["near_forward_obstacle_speed_mean_m_s"] is not None
    ]
    variation = [
        float(trial["command_variation_rate_m_s2"])
        for trial in trials
        if trial["command_variation_rate_m_s2"] is not None
    ]
    return {
        "trial_count": len(trials),
        "success_rate": len(successful) / len(trials),
        "collision_count": sum(bool(trial["collision"]) for trial in trials),
        "mean_success_time_s": mean_or_none(elapsed),
        "minimum_clearance_m": None if not clearance else min(clearance),
        "minimum_front_clearance_m": (
            None if not front_clearance else min(front_clearance)
        ),
        "mean_near_obstacle_speed_m_s": mean_or_none(near_speed),
        "mean_near_forward_obstacle_speed_m_s": mean_or_none(
            near_forward_speed
        ),
        "mean_command_variation_rate_m_s2": mean_or_none(variation),
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
    scenarios = (
        DECISION_SCENARIOS
        if args.suite == "decision"
        else (("single", (args.goal_x, args.goal_y, args.goal_yaw)),)
    )
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
    original_vx_std: Optional[float] = None
    try:
        ros.wait_until_ready(timeout=30.0, require_supervisor=False)
        wait_for_costmap(ros)
        invalid_scenarios = [
            name for name, goal in scenarios
            if not ros.is_position_free(goal[0], goal[1])
        ]
        if invalid_scenarios:
            raise RuntimeError(
                "Scenario goals are occupied in the global costmap: "
                + ", ".join(invalid_scenarios)
            )
        if not get_parameters_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Required service unavailable: get_parameters")
        vx_max = read_double_parameter(
            ros,
            get_parameters_client,
            "FollowPath.vx_max",
        )
        original_vx_std = read_double_parameter(
            ros,
            get_parameters_client,
            "FollowPath.vx_std",
        )
        if args.vx_std_override is not None:
            set_double_parameter(
                ros,
                ros.mppi_parameter_client,
                "FollowPath.vx_std",
                args.vx_std_override,
            )
        active_vx_std = read_double_parameter(
            ros,
            get_parameters_client,
            "FollowPath.vx_std",
        )
        print(
            f"[benchmark] Fixed MPPI vx_max={vx_max:.3f} m/s, "
            f"vx_std={active_vx_std:.3f}",
            flush=True,
        )

        for scenario, goal in scenarios:
            for trial_index in range(1, args.trials + 1):
                print(
                    f"[benchmark] {scenario} trial "
                    f"{trial_index}/{args.trials}: "
                    f"goal=({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})",
                    flush=True,
                )
                summary, samples = run_trial(
                    ros,
                    commands,
                    args,
                    trial_index,
                    vx_max,
                    scenario,
                    goal,
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
            "schema_version": 2,
            "mode": "fixed_mppi_baseline",
            "suite": args.suite,
            "scenarios": [
                {"name": name, "goal": list(goal)}
                for name, goal in scenarios
            ],
            "controller": {
                "vx_max_m_s": vx_max,
                "vx_std_m_s": active_vx_std,
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
        if args.vx_std_override is not None and original_vx_std is not None:
            try:
                set_double_parameter(
                    ros,
                    ros.mppi_parameter_client,
                    "FollowPath.vx_std",
                    original_vx_std,
                )
            except Exception as error:
                print(
                    f"[benchmark] WARNING: could not restore vx_std: {error}",
                    flush=True,
                )
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        ros.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
