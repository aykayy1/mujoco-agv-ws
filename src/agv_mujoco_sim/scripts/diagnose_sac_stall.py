#!/usr/bin/env python3
"""Frozen SAC-only rollout with per-step telemetry; never calls learn()."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import numpy as np
from action_msgs.msg import GoalStatusArray
from nav_msgs.msg import Path as NavPath
from stable_baselines3 import SAC

from train_sac import make_environment, validate_model


FIELDS = (
    "x", "y", "yaw", "goal", "distance_to_goal", "nav_status",
    "goal_event", "previous_goal_nav_status", "termination_reason",
    "linear_velocity", "angular_velocity", "cmd_vx_m_s", "cmd_wz_rad_s",
    "cmd_vel_age_s", "requested_action", "applied_speed_scale",
    "front_corridor_clearance_m", "min_lidar", "front_scan_valid",
    "conservative_lateral_risk", "front_proactive_risk",
    "dynamic_obstacle_active", "dynamic_obstacle_position",
    "dynamic_obstacle_parked_after_crossing", "dynamic_obstacle_spawns",
    "front_safety_stop_active", "safety_override_active",
    "collision_event_count", "goals_reached_this_episode",
    "hard_reset_next_episode",
)


class Nav2Trace:
    """Capture published plans and action status without changing Nav2."""

    PLAN_TOPICS = ("/plan", "/transformed_global_plan", "/local_plan")

    def __init__(self, node, output: Path) -> None:
        self.lock = threading.RLock()
        self.stream = output.open("w", encoding="utf-8")
        self.plan_counts = {topic: 0 for topic in self.PLAN_TOPICS}
        self.statuses = []
        self.subscriptions = [
            node.create_subscription(
                NavPath, topic,
                lambda message, topic=topic: self._plan(topic, message), 10,
            )
            for topic in self.PLAN_TOPICS
        ]
        self.subscriptions.append(node.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status", self._status, 10,
        ))

    def _plan(self, topic: str, message: NavPath) -> None:
        row = {
            "topic": topic,
            "stamp_s": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
            "frame": message.header.frame_id,
            "poses": [[pose.pose.position.x, pose.pose.position.y]
                      for pose in message.poses],
        }
        with self.lock:
            self.plan_counts[topic] += 1
            row["sequence"] = self.plan_counts[topic]
            self.stream.write(json.dumps(row) + "\n")

    def _status(self, message: GoalStatusArray) -> None:
        with self.lock:
            self.statuses = [int(item.status) for item in message.status_list]

    def snapshot(self) -> dict:
        with self.lock:
            return {"plan_counts": dict(self.plan_counts),
                    "navigate_to_pose_statuses": list(self.statuses)}

    def close(self) -> None:
        with self.lock:
            self.stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dynamic-obstacle-probability", type=float, default=0.75)
    parser.add_argument("--suite", choices=("combined", "single"), default="combined")
    parser.add_argument("--goal-sampling", default="random_costmap")
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    args.model = args.model.expanduser().resolve()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    # Preflight before ROS startup. A 20k checkpoint is intentionally used;
    # neither the early-stopped 22k model nor its replay buffer is loaded.
    policy = SAC.load(str(args.model), device="cpu")
    validate_model(policy, sac_only=True)
    args.run_dir = args.run_dir.expanduser().resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "step_trace.jsonl").exists():
        raise FileExistsError(
            f"Choose a new --run-dir: {args.run_dir / 'step_trace.jsonl'} already exists"
        )
    args.dynamic_obstacles = True
    args.sac_only = True
    args.conservative_crossing = True
    args.retain_crossing_obstacle = True
    environment = None
    nav_trace = None
    episodes = 0
    stalls = 0
    steps = 0
    try:
        environment = make_environment(args)
        nav_trace = Nav2Trace(environment.unwrapped.ros, args.run_dir / "plans.jsonl")
        observation, _ = environment.reset(seed=args.seed)
        with (args.run_dir / "step_trace.jsonl").open("w", encoding="utf-8") as trace:
            for steps in range(1, args.max_steps + 1):
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(
                    np.asarray(action, dtype=np.float32)
                )
                row = {"step": steps, "episode": episodes + 1}
                row.update({key: info.get(key) for key in FIELDS})
                row.update(nav_trace.snapshot())
                trace.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                if terminated or truncated:
                    episodes += 1
                    reason = info.get("termination_reason")
                    print(
                        f"EPISODE {episodes} step={steps} reason={reason} "
                        f"goals={info.get('goals_reached_this_episode')} "
                        f"pose=({info.get('x'):.2f},{info.get('y'):.2f})",
                        flush=True,
                    )
                    trace.flush()
                    if reason in ("motion_stall", "safety_deadlock"):
                        stalls += 1
                        break
                    observation, _ = environment.reset()
    finally:
        if environment is not None:
            environment.close()
        if nav_trace is not None:
            nav_trace.close()
        summary = {
            "source_model": str(args.model),
            "frozen_policy": True,
            "deterministic": True,
            "steps": steps,
            "episodes": episodes,
            "stalls": stalls,
            "trace": str(args.run_dir / "step_trace.jsonl"),
            "plans": str(args.run_dir / "plans.jsonl"),
            "published_plan_counts": (
                nav_trace.snapshot()["plan_counts"] if nav_trace is not None else {}
            ),
        }
        (args.run_dir / "diagnostic_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
