#!/usr/bin/env python3
"""Evaluate a frozen 56D SAC checkpoint with moving obstacles in MuJoCo.

This command never calls ``learn`` or changes the checkpoint. It requires the
MuJoCo/Nav2 training stack, including /reset_simulation, to be running.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from stable_baselines3 import SAC
from std_msgs.msg import Bool, Float32MultiArray, String

from agv_rl_env import AgvRlEnv
from train_sac import validate_model


class DiagnosticTrace(Node):
    """Keep timestamped, sub-second telemetry for the end of each episode."""

    FIELDS = (
        "episode", "sample_time_s", "source", "scan_stamp_s",
        "front_min_range_m", "sector_min_ttc_s", "safety_override_active",
        "applied_scale", "hard_stop_active", "cmd_vx_m_s", "cmd_wz_rad_s", "odom_vx_m_s",
        "hard_stop_clearance_m", "hard_stop_distance_m", "hard_stop_corridor_half_width_m",
        "hard_stop_edge_hold_remaining_s",
        "hard_stop_reason",
        "obstacle_x", "obstacle_y", "collision_event",
    )

    def __init__(self) -> None:
        super().__init__(
            "sac_dynamic_diagnostic_trace",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.lock = threading.RLock()
        self.episode = 0
        self.active = False
        self.samples = deque(maxlen=3000)
        self.latest = {
            "scan_stamp_s": "", "front_min_range_m": "",
            "sector_min_ttc_s": "", "safety_override_active": "",
            "applied_scale": "", "hard_stop_active": "",
            "cmd_vx_m_s": "", "cmd_wz_rad_s": "",
            "odom_vx_m_s": "", "obstacle_x": "", "obstacle_y": "",
            "hard_stop_clearance_m": "", "hard_stop_distance_m": "",
            "hard_stop_corridor_half_width_m": "",
            "hard_stop_edge_hold_remaining_s": "",
            "hard_stop_reason": "",
        }
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(LaserScan, "/scan", self._scan, sensor_qos)
        self.create_subscription(
            Float32MultiArray, "/rl_velocity_limits_applied",
            self._status, 10,
        )
        self.create_subscription(Twist, "/cmd_vel", self._cmd, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self._odom, 10)
        self.create_subscription(Point, "/dynamic_obstacle/position", self._obstacle, 10)
        self.create_subscription(String, "/collision_event", self._collision, 10)
        self.create_subscription(Bool, "/sim_front_safety_stop", self._hard_stop, 10)
        self.create_subscription(
            Float32MultiArray, "/sim_front_safety_diagnostics", self._hard_stop_diagnostics, 10,
        )

    def start_episode(self, episode: int) -> None:
        with self.lock:
            self.episode = episode
            self.samples.clear()
            self.latest = {key: "" for key in self.latest}
            self.active = True

    def stop_episode(self) -> list[dict]:
        with self.lock:
            self.active = False
            return list(self.samples)

    def _record(self, source: str, collision_event: bool = False) -> None:
        if not self.active:
            return
        self.samples.append({
            "episode": self.episode,
            "sample_time_s": self.get_clock().now().nanoseconds * 1.0e-9,
            "source": source,
            **self.latest,
            "collision_event": collision_event,
        })

    def _scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = float(msg.angle_min) + np.arange(ranges.size) * float(msg.angle_increment)
        front = (angles >= -math.pi / 3) & (angles <= math.pi / 3)
        valid = front & np.isfinite(ranges) & (ranges > 0.0)
        with self.lock:
            self.latest["scan_stamp_s"] = (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
            )
            self.latest["front_min_range_m"] = (
                float(np.min(ranges[valid])) if np.any(valid) else ""
            )
            self._record("scan")

    def _status(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 29:
            return
        with self.lock:
            self.latest["applied_scale"] = float(msg.data[0])
            self.latest["safety_override_active"] = bool(msg.data[3])
            self.latest["sector_min_ttc_s"] = float(min(msg.data[5:29]))
            self._record("supervisor")

    def _cmd(self, msg: Twist) -> None:
        with self.lock:
            self.latest["cmd_vx_m_s"] = float(msg.linear.x)
            self.latest["cmd_wz_rad_s"] = float(msg.angular.z)
            self._record("cmd_vel")

    def _odom(self, msg: Odometry) -> None:
        with self.lock:
            self.latest["odom_vx_m_s"] = float(msg.twist.twist.linear.x)

    def _obstacle(self, msg: Point) -> None:
        with self.lock:
            self.latest["obstacle_x"] = float(msg.x)
            self.latest["obstacle_y"] = float(msg.y)

    def _collision(self, _msg: String) -> None:
        with self.lock:
            self._record("collision", collision_event=True)

    def _hard_stop(self, msg: Bool) -> None:
        with self.lock:
            previous = self.latest["hard_stop_active"]
            self.latest["hard_stop_active"] = bool(msg.data)
            if previous != bool(msg.data):
                self._record("hard_stop")

    def _hard_stop_diagnostics(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            return
        with self.lock:
            self.latest["hard_stop_clearance_m"] = float(msg.data[1])
            self.latest["hard_stop_distance_m"] = float(msg.data[2])
            self.latest["hard_stop_corridor_half_width_m"] = float(msg.data[3])
            if len(msg.data) >= 5:
                self.latest["hard_stop_edge_hold_remaining_s"] = float(msg.data[4])
            if len(msg.data) >= 6:
                self.latest["hard_stop_reason"] = int(msg.data[5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--dynamic-obstacle-probability", type=float, default=0.75)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmark_results")
        / f"sac_dynamic_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument(
        "--inspect-only", action="store_true",
        help="Validate the checkpoint without connecting to ROS",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.max_episode_steps <= 0:
        raise ValueError("episodes and max-episode-steps must be positive")
    if not math.isfinite(args.dynamic_obstacle_probability) or not (
        0.0 <= args.dynamic_obstacle_probability <= 1.0
    ):
        raise ValueError("dynamic-obstacle-probability must be in [0, 1]")

    model_path = args.model.expanduser().resolve(strict=True)
    model = SAC.load(str(model_path), device="cpu")
    contract = validate_model(model)
    print(json.dumps(contract, indent=2), flush=True)
    if args.inspect_only:
        print("SAC_DYNAMIC_EVAL_PREFLIGHT=PASS", flush=True)
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "metadata.json", {
        "model": str(model_path),
        "model_timesteps": int(model.num_timesteps),
        "contract": contract,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "dynamic_obstacle_probability": args.dynamic_obstacle_probability,
        "policy_deterministic": True,
        "hard_reset_every_episode": True,
        "diagnostic_trace": "last 3000 sub-second samples per episode",
    })

    step_fields = (
        "episode", "step", "reward", "robot_map_x", "robot_map_y",
        "goal_distance_m", "velocity_m_s", "requested_scale",
        "applied_scale", "front_min_range_m", "sector_min_ttc_s",
        "sector_min_ttc_episode_s", "safety_override_active",
        "front_safety_stop_active", "front_safety_stop_events",
        "dynamic_obstacle_spawns", "collision_event_count",
        "collision_event_detail_count", "terminated", "truncated",
    )
    episode_fields = (
        "episode", "steps", "reward", "termination_reason", "is_success",
        "goals_reached", "collision_events", "collision_event_details",
        "dynamic_obstacle_spawns", "minimum_clearance_m",
        "minimum_sector_ttc_s",
        "front_safety_stop_events",
    )
    results = []
    pair_counts: Counter[str] = Counter()
    environment = None
    trace = None
    try:
        environment = AgvRlEnv(
            goals=AgvRlEnv.COMBINED_GOALS,
            goal_sampling="random_costmap",
            max_episode_steps=args.max_episode_steps,
            dynamic_obstacles=True,
            dynamic_obstacle_probability=args.dynamic_obstacle_probability,
            require_front_safety_stop=True,
        )
        environment._goal_rng = np.random.default_rng(args.seed)
        trace = DiagnosticTrace()
        environment.executor.add_node(trace)
        with (
            (output_dir / "steps.csv").open("w", newline="", encoding="utf-8") as step_stream,
            (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_stream,
            (output_dir / "collision_events.jsonl").open("w", encoding="utf-8") as event_stream,
            (output_dir / "diagnostic_trace.csv").open("w", newline="", encoding="utf-8") as trace_stream,
        ):
            step_writer = csv.DictWriter(step_stream, fieldnames=step_fields)
            episode_writer = csv.DictWriter(episode_stream, fieldnames=episode_fields)
            step_writer.writeheader()
            episode_writer.writeheader()
            trace_writer = csv.DictWriter(trace_stream, fieldnames=DiagnosticTrace.FIELDS)
            trace_writer.writeheader()
            for episode_index in range(args.episodes):
                environment._needs_hard_reset = True
                observation, _ = environment.reset(seed=args.seed + episode_index)
                trace.start_episode(episode_index + 1)
                reward_total = 0.0
                for step_index in range(1, args.max_episode_steps + 1):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = environment.step(action)
                    reward_total += float(reward)
                    step_writer.writerow({
                        "episode": episode_index + 1,
                        "step": step_index,
                        "reward": float(reward),
                        "robot_map_x": info["x"],
                        "robot_map_y": info["y"],
                        "goal_distance_m": info["distance_to_goal"],
                        "velocity_m_s": info["linear_velocity"],
                        "requested_scale": float(action[0]),
                        "applied_scale": info["applied_speed_scale"],
                        "front_min_range_m": info["min_lidar_step"],
                        "sector_min_ttc_s": info["min_sector_ttc"],
                        "sector_min_ttc_episode_s": info["min_sector_ttc_episode"],
                        "safety_override_active": info["safety_override_active"],
                        "front_safety_stop_active": info["front_safety_stop_active"],
                        "front_safety_stop_events": info["front_safety_stop_events"],
                        "dynamic_obstacle_spawns": info["dynamic_obstacle_spawns"],
                        "collision_event_count": info["collision_event_count"],
                        "collision_event_detail_count": info["collision_event_detail_count"],
                        "terminated": terminated,
                        "truncated": truncated,
                    })
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("Environment did not truncate at max_episode_steps")

                trace_writer.writerows(trace.stop_episode())
                events = info["collision_events"]
                for event in events:
                    event_stream.write(json.dumps({
                        "episode": episode_index + 1,
                        **event,
                    }, ensure_ascii=False) + "\n")
                    for pair in event.get("contact_pairs", []):
                        pair_counts[
                            f"{pair.get('robot_geom', '?')}<->"
                            f"{pair.get('environment_geom', '?')}"
                        ] += 1
                row = {
                    "episode": episode_index + 1,
                    "steps": step_index,
                    "reward": reward_total,
                    "termination_reason": info["termination_reason"],
                    "is_success": bool(info.get("is_success", False)),
                    "goals_reached": info["goals_reached_this_episode"],
                    "collision_events": info["collision_event_count"],
                    "collision_event_details": len(events),
                    "dynamic_obstacle_spawns": info["dynamic_obstacle_spawns"],
                    "minimum_clearance_m": info["min_lidar_episode"],
                    "minimum_sector_ttc_s": info["min_sector_ttc_episode"],
                    "front_safety_stop_events": info["front_safety_stop_events"],
                }
                episode_writer.writerow(row)
                results.append(row)
                step_stream.flush()
                episode_stream.flush()
                event_stream.flush()
                trace_stream.flush()
                print(
                    f"EVAL episode={episode_index + 1}/{args.episodes} "
                    f"reason={row['termination_reason']} "
                    f"spawns={row['dynamic_obstacle_spawns']} "
                    f"collision_details={len(events)}",
                    flush=True,
                )
    except Exception as exc:
        write_json(output_dir / "failure.json", {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "completed_episodes": len(results),
        })
        raise
    finally:
        if environment is not None:
            if trace is not None:
                environment.executor.remove_node(trace)
                trace.destroy_node()
            environment.close()

    if int(model.num_timesteps) != int(contract["num_timesteps"]):
        raise RuntimeError("Evaluation changed model timesteps unexpectedly")
    summary = {
        "model": str(model_path),
        "model_timesteps": int(model.num_timesteps),
        "episodes": len(results),
        "collisions": sum(row["collision_events"] for row in results),
        "collision_episodes": sum(row["collision_events"] > 0 for row in results),
        "success_episodes": sum(row["is_success"] for row in results),
        "dynamic_obstacle_spawns": sum(row["dynamic_obstacle_spawns"] for row in results),
        "front_safety_stop_events": sum(row["front_safety_stop_events"] for row in results),
        "collision_pair_counts": dict(pair_counts),
        "collision_events_with_details": sum(row["collision_event_details"] for row in results),
        "minimum_sector_ttc_s": min(row["minimum_sector_ttc_s"] for row in results),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"SAC_DYNAMIC_EVAL_DONE {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
