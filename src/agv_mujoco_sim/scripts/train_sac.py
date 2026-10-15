#!/usr/bin/env python3
"""Train the temporal-observation, one-dimensional SpeedLimit SAC policy.

The default environment uses a 56D observation; ``--sac-only`` uses a
31D observation without TTC. Both use a 1D ``[speed_scale]`` action.
This breaks every earlier 26D/4D and 28D/4D checkpoint/replay contract:
- Thêm chế độ --from-scratch: tạo SAC(...) HOÀN TOÀN MỚI, KHÔNG gọi
  SAC.load() — dùng khi không còn checkpoint nào tương thích observation
  hiện tại (đúng tình huống hiện tại: checkpoint Gazebo cũ là 26D).
- Model and replay preflight derives both shapes and action bounds directly
  from AgvRlEnv and rejects incompatible artifacts before training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from stable_baselines3 import SAC, __version__ as sb3_runtime_version
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from agv_rl_env import AgvRlEnv


def default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path.cwd()
        / "phase2_finetune_runs"
        / f"run_{stamp}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train AgvRlEnv (56D default or 31D no-TTC SAC-only / 1D speed_scale) "
            "against MuJoCo, "
            "AMCL and Nav2 MPPI — either fine-tuning an existing compatible "
            "checkpoint (--model), or starting a brand new model "
            "(--from-scratch) when no compatible checkpoint exists."
        )
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help=(
            "Tạo SAC(...) HOÀN TOÀN MỚI thay vì SAC.load() — dùng khi "
            "observation/action contract của env vừa đổi và không còn "
            "checkpoint cũ nào tương thích (bỏ qua --model, --inspect-only)."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Explicit compatible checkpoint for the selected observation mode. "
            "Required unless --from-scratch is used."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate the model contract without creating the ROS env",
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    parser.add_argument(
        "--stop-after-consecutive-deadlocks", type=int, default=0,
        help="Probe circuit breaker: stop after this many consecutive safety or motion-stall episodes (0 disables).",
    )
    parser.add_argument(
        "--stop-on-collision", action="store_true",
        help="Probe circuit breaker: stop after the first collision episode.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=250,
        help=(
            "New MuJoCo transitions collected before the first gradient "
            "update when no replay buffer is supplied. Với --from-scratch, "
            "đây chính là learning_starts của SAC."
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument(
        "--dynamic-obstacles", action="store_true",
        help="Move the MuJoCo obstacle through the front field of view during training",
    )
    parser.add_argument(
        "--sac-only", action="store_true",
        help="Ablation without the simulation front safety stop",
    )
    parser.add_argument(
        "--conservative-crossing", action="store_true",
        help=(
            "Fine-tune no-TTC SAC with the same 31D observation, closer "
            "warehouse-style crossings and a current-scan lateral-risk reward"
        ),
    )
    parser.add_argument(
        "--retain-crossing-obstacle", action="store_true",
        help=(
            "Keep the crossing obstacle at its endpoint until goal completion "
            "or reset, matching the warehouse benchmark"
        ),
    )
    parser.add_argument(
        "--dynamic-obstacle-probability", type=float, default=0.75,
        help="Probability of a front crossing at each eligible ~20-step interval",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--replay-buffer",
        type=Path,
        default=None,
        help="Optional replay buffer saved by an earlier MuJoCo run",
    )
    parser.add_argument(
        "--suite",
        choices=("single", "combined"),
        default="combined",
    )
    parser.add_argument(
        "--goal-sampling",
        choices=("fixed", "cycle", "random_list", "random_costmap", "random"),
        default="random_costmap",
        help=(
            "Default: sample across the global costmap. fixed/cycle/random_list "
            "use the --suite goal list. random is a legacy alias for random_list."
        ),
    )
    parser.add_argument("--goal-x", type=float, default=9.5)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    return parser.parse_args()


def resolve_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() and resolved.suffix != ".zip":
        candidate = resolved.with_suffix(".zip")
        if candidate.is_file():
            resolved = candidate
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def saved_sb3_version(model_path: Path) -> str:
    try:
        with zipfile.ZipFile(model_path) as archive:
            return archive.read("_stable_baselines3_version").decode().strip()
    except (KeyError, OSError, zipfile.BadZipFile):
        return "unknown"


def validate_model(model: SAC, *, sac_only: bool = False) -> Dict[str, Any]:
    # Derive the contract from the environment's single source of truth.
    expected_observation_shape = (
        AgvRlEnv.OBSERVATION_DIM_NO_TTC if sac_only else AgvRlEnv.OBSERVATION_DIM,
    )
    expected_contract = (
        AgvRlEnv.CONTRACT_VERSION_NO_TTC if sac_only else AgvRlEnv.CONTRACT_VERSION
    )
    expected_action_shape = tuple(AgvRlEnv.ACTION_LOW.shape)
    problems = []
    saved_contract = getattr(model, "observation_contract_version", None)
    if saved_contract != expected_contract:
        problems.append(
            f"observation contract={saved_contract!r}, "
            f"expected={expected_contract!r}; "
            "checkpoints with different LiDAR or timing semantics "
            "cannot be reused"
        )
    if tuple(model.observation_space.shape) != expected_observation_shape:
        problems.append(
            f"observation={model.observation_space.shape}, "
            f"expected={expected_observation_shape}"
        )
    if tuple(model.action_space.shape) != expected_action_shape:
        problems.append(
            f"action={model.action_space.shape}, "
            f"expected={expected_action_shape}"
        )
    if tuple(model.action_space.shape) == expected_action_shape:
        if not np.allclose(model.action_space.low, AgvRlEnv.ACTION_LOW):
            problems.append(
                f"action_low={model.action_space.low}, "
                f"expected={AgvRlEnv.ACTION_LOW}"
            )
        if not np.allclose(model.action_space.high, AgvRlEnv.ACTION_HIGH):
            problems.append(
                f"action_high={model.action_space.high}, "
                f"expected={AgvRlEnv.ACTION_HIGH}"
            )
    if problems:
        raise ValueError("Incompatible Phase-2 checkpoint: " + "; ".join(problems))

    probe_observation = np.ones(expected_observation_shape, dtype=np.float32)
    probe_action, _state = model.predict(
        probe_observation,
        deterministic=True,
    )
    probe_action = np.asarray(probe_action, dtype=np.float32)
    if probe_action.shape != expected_action_shape:
        raise ValueError(f"Invalid prediction shape: {probe_action.shape}")
    if not np.all(np.isfinite(probe_action)):
        raise ValueError("Probe action contains NaN or Inf")
    if not model.action_space.contains(probe_action):
        raise ValueError("Probe action is outside checkpoint bounds")

    return {
        "contract_version": expected_contract,
        "observation_shape": list(expected_observation_shape),
        "action_shape": list(expected_action_shape),
        "action_low": AgvRlEnv.ACTION_LOW.tolist(),
        "action_high": AgvRlEnv.ACTION_HIGH.tolist(),
        "action_interval_seconds": AgvRlEnv.STEP_WAIT_DURATION,
        "probe_action": probe_action.tolist(),
        "num_timesteps": int(model.num_timesteps),
    }


def validate_replay_buffer_shape(model: SAC, environment: Monitor) -> None:
    """Reject replay data from older observation, action or reward semantics."""
    saved_contract = getattr(
        model.replay_buffer, "observation_contract_version", None
    )
    base_environment = getattr(environment, "unwrapped", environment)
    expected_contract = getattr(
        base_environment, "contract_version", AgvRlEnv.CONTRACT_VERSION
    )
    expected_reward_contract = getattr(
        base_environment, "reward_contract_version",
        AgvRlEnv.REWARD_CONTRACT_VERSION,
    )
    if saved_contract != expected_contract:
        raise ValueError(
            f"Replay buffer contract={saved_contract!r}, "
            f"expected={expected_contract!r}. "
            "A buffer recorded with different LiDAR or timing semantics "
            "cannot train this policy."
        )
    saved_reward_contract = getattr(
        model.replay_buffer, "reward_contract_version", None
    )
    if saved_reward_contract != expected_reward_contract:
        raise ValueError(
            f"Replay reward contract={saved_reward_contract!r}, "
            f"expected={expected_reward_contract!r}. "
            "Old stop rewards cannot be mixed with this run; omit "
            "--replay-buffer and collect fresh transitions."
        )
    buffer_obs_shape = tuple(model.replay_buffer.observations.shape[2:])
    env_obs_shape = tuple(environment.observation_space.shape)
    if buffer_obs_shape != env_obs_shape:
        raise ValueError(
            f"Replay buffer KHÔNG khớp observation shape với env hiện tại: "
            f"buffer={buffer_obs_shape}, env={env_obs_shape}. "
            f"Không dùng buffer này — rất có thể nó được ghi từ 1 phiên bản "
            f"observation_space cũ. Bỏ --replay-buffer để bắt đầu buffer trống."
        )
    buffer_action_shape = tuple(model.replay_buffer.actions.shape[2:])
    env_action_shape = tuple(environment.action_space.shape)
    if buffer_action_shape != env_action_shape:
        raise ValueError(
            "Replay buffer KHÔNG khớp action shape với env hiện tại: "
            f"buffer={buffer_action_shape}, env={env_action_shape}. "
            "Replay 4D cũ không thể dùng cho policy SpeedLimit 1D."
        )


class FineTuneCallback(BaseCallback):
    """Save recoverable checkpoints and compact episode diagnostics."""

    def __init__(
        self,
        run_dir: Path,
        starting_timesteps: int,
        checkpoint_freq: int,
        stop_after_consecutive_deadlocks: int = 0,
        stop_on_collision: bool = False,
    ) -> None:
        super().__init__(verbose=1)
        self.run_dir = run_dir
        self.starting_timesteps = int(starting_timesteps)
        self.checkpoint_freq = int(checkpoint_freq)
        self.stop_after_consecutive_deadlocks = int(stop_after_consecutive_deadlocks)
        self.stop_on_collision = bool(stop_on_collision)
        self.consecutive_deadlocks = 0
        self.early_stop_reason: Optional[str] = None
        self.next_checkpoint = self.checkpoint_freq
        self.episode_count = 0
        self.reason_counts: Counter[str] = Counter()
        self.collision_events = 0
        self.collision_pair_counts: Counter[str] = Counter()
        self.collision_details_written = 0
        self.goals_reached = 0
        self.goals_reached_completed_episodes = 0
        self.goals_reached_partial_episode = 0
        self.front_stop_penalty_sum = 0.0
        self.front_risk_penalty_sum = 0.0
        self.front_risk_steps = 0
        self.lateral_risk_penalty_sum = 0.0
        self.lateral_risk_steps = 0
        self.minimum_front_braking_margin = math.inf
        self.successful_episodes = 0
        self.zero_goal_episodes = 0
        self.collision_episodes = 0
        self.episodes_with_goal = 0
        self.observed_steps = 0
        self.safety_override_steps = 0
        self.front_safety_stop_steps = 0
        self.front_safety_stop_events = 0
        self.front_safety_stop_reason_steps: Counter[str] = Counter()
        self.minimum_scale_steps = 0
        self.requested_scale_sum = 0.0
        self.applied_scale_sum = 0.0
        self.minimum_clearance = math.inf
        self.minimum_sector_ttc = math.inf
        self.dynamic_obstacle_steps = 0
        self.parked_obstacle_steps = 0
        self.dynamic_obstacle_episodes = 0
        self.dynamic_obstacle_spawns = 0
        self.started = time.monotonic()

    @property
    def additional_steps(self) -> int:
        return int(self.num_timesteps - self.starting_timesteps)

    def _on_training_start(self) -> None:
        # SAC creates a fresh buffer inside learn() when resuming without one.
        if self.model.replay_buffer is not None:
            self.model.replay_buffer.observation_contract_version = (
                self.model.observation_contract_version
            )
            self.model.replay_buffer.reward_contract_version = (
                getattr(
                    self.model, "reward_contract_version",
                    AgvRlEnv.REWARD_CONTRACT_VERSION,
                )
            )

    def _save(self, stem: str) -> None:
        self.model.save(str(self.run_dir / stem))
        self.model.save(str(self.run_dir / "latest_model"))
        try:
            self.model.save_replay_buffer(
                str(self.run_dir / "latest_replay_buffer.pkl")
            )
        except Exception as exc:
            print(f"WARNING: replay buffer save failed: {exc}", flush=True)

    def _on_step(self) -> bool:
        if self.additional_steps >= self.next_checkpoint:
            self._save(
                f"checkpoint_{self.num_timesteps:09d}_total_steps"
            )
            print(
                f"CHECKPOINT total={self.num_timesteps} "
                f"additional={self.additional_steps}",
                flush=True,
            )
            while self.next_checkpoint <= self.additional_steps:
                self.next_checkpoint += self.checkpoint_freq

        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            self.observed_steps += 1
            self.goals_reached += int(info.get("goal_event") == "succeeded")
            self.goals_reached_partial_episode = int(
                info.get("goals_reached_this_episode", 0)
            )
            self.front_stop_penalty_sum += float(
                info.get("reward_front_stop", 0.0)
            )
            self.front_risk_penalty_sum += float(
                info.get("reward_front_risk", 0.0)
            )
            proactive_risk = float(info.get("front_proactive_risk", 0.0))
            self.front_risk_steps += int(proactive_risk > 0.0)
            self.lateral_risk_penalty_sum += float(
                info.get("reward_lateral_risk", 0.0)
            )
            self.lateral_risk_steps += int(
                float(info.get("conservative_lateral_risk", 0.0)) > 0.0
            )
            braking_margin = float(
                info.get("front_braking_margin_m", math.inf)
            )
            if math.isfinite(braking_margin):
                self.minimum_front_braking_margin = min(
                    self.minimum_front_braking_margin, braking_margin
                )
            self.dynamic_obstacle_steps += int(
                bool(info.get("dynamic_obstacle_active", False))
            )
            self.parked_obstacle_steps += int(
                bool(info.get("dynamic_obstacle_parked_after_crossing", False))
            )
            step_ttc = float(info.get("min_sector_ttc", math.inf))
            if math.isfinite(step_ttc):
                self.minimum_sector_ttc = min(self.minimum_sector_ttc, step_ttc)
            self.safety_override_steps += int(
                bool(info.get("safety_override_active", False))
            )
            self.front_safety_stop_steps += int(
                bool(info.get("front_safety_stop_active", False))
            )
            if bool(info.get("front_safety_stop_active", False)):
                self.front_safety_stop_reason_steps[
                    str(int(info.get("front_safety_stop_reason", 0)))
                ] += 1
            applied_scale = float(info.get("applied_speed_scale", math.nan))
            requested = info.get("requested_action", [])
            if math.isfinite(applied_scale):
                self.applied_scale_sum += applied_scale
                self.minimum_scale_steps += int(applied_scale <= 0.16)
            if requested:
                self.requested_scale_sum += float(requested[0])
            if not done:
                continue
            self.episode_count += 1
            reason = str(info.get("termination_reason") or "unknown")
            self.reason_counts[reason] += 1
            self.consecutive_deadlocks = (
                self.consecutive_deadlocks + 1
                if reason in (
                    "safety_deadlock", "motion_stall", "rotation_stall",
                    "command_stall",
                ) else 0
            )
            self.collision_events += int(info.get("collision_event_count", 0))
            self.front_safety_stop_events += int(info.get("front_safety_stop_events", 0))
            collision_details = info.get("collision_events", [])
            if collision_details:
                with (self.run_dir / "collision_events.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    for detail in collision_details:
                        stream.write(json.dumps({
                            "episode": self.episode_count,
                            "model_timesteps": int(self.num_timesteps),
                            **detail,
                        }, ensure_ascii=False) + "\n")
                        self.collision_details_written += 1
                        for pair in detail.get("contact_pairs", []):
                            label = (
                                f"{pair.get('robot_geom', '?')}<->"
                                f"{pair.get('environment_geom', '?')}"
                            )
                            self.collision_pair_counts[label] += 1
            goals_reached = int(info.get("goals_reached_this_episode", 0))
            self.zero_goal_episodes += int(goals_reached == 0)
            self.episodes_with_goal += int(goals_reached > 0)
            self.collision_episodes += int(reason == "collision")
            is_success = bool(info.get("is_success", False))
            self.goals_reached_completed_episodes += goals_reached
            self.goals_reached_partial_episode = 0
            self.successful_episodes += int(is_success)
            clearance = float(info.get("min_lidar_episode", math.inf))
            self.minimum_clearance = min(self.minimum_clearance, clearance)
            spawns = int(info.get("dynamic_obstacle_spawns", 0))
            self.dynamic_obstacle_spawns += spawns
            self.dynamic_obstacle_episodes += int(spawns > 0)
            episode_ttc = float(info.get("min_sector_ttc_episode", math.inf))
            if math.isfinite(episode_ttc):
                self.minimum_sector_ttc = min(self.minimum_sector_ttc, episode_ttc)
            episode = info.get("episode", {})
            print(
                f"EPISODE n={self.episode_count} "
                f"additional_steps={self.additional_steps} "
                f"length={episode.get('l', '?')} "
                f"reward={float(episode.get('r', math.nan)):+.3f} "
                f"reason={reason} "
                f"distance={float(info.get('distance_to_goal', math.nan)):.3f} "
                f"min_obstacle={clearance:.3f} "
                f"min_ttc_episode={episode_ttc:.3f} "
                f"dynamic_spawns={spawns} "
                f"goals_reached={goals_reached} "
                f"success={is_success} "
                f"collision_events={int(info.get('collision_event_count', 0))} "
                f"front_stop_events={int(info.get('front_safety_stop_events', 0))}",
                flush=True,
            )
            if self.stop_on_collision and reason == "collision":
                self.early_stop_reason = "collision"
            elif (
                self.stop_after_consecutive_deadlocks > 0
                and self.consecutive_deadlocks >= self.stop_after_consecutive_deadlocks
            ):
                self.early_stop_reason = (
                    f"{self.consecutive_deadlocks}_consecutive_deadlocks"
                )
            if self.early_stop_reason is not None:
                print(
                    f"PROBE_EARLY_STOP reason={self.early_stop_reason} "
                    f"total={self.num_timesteps}; trigger transition is not "
                    "stored by SB3, so do not resume from this replay buffer",
                    flush=True,
                )
                return False
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "starting_timesteps": self.starting_timesteps,
            "final_timesteps": int(self.num_timesteps),
            "additional_timesteps": self.additional_steps,
            "early_stopped": self.early_stop_reason is not None,
            "early_stop_reason": self.early_stop_reason,
            "episodes": self.episode_count,
            "termination_reasons": dict(self.reason_counts),
            "collision_events": self.collision_events,
            "front_safety_stop_events": self.front_safety_stop_events,
            "front_safety_stop_reason_step_counts": dict(
                self.front_safety_stop_reason_steps
            ),
            "front_safety_stop_reason_bits": {
                "1": "missing or stale scan",
                "2": "forward stopping corridor",
                "4": "approaching edge hazard held after visibility loss",
                "8": "invalid command",
            },
            "front_safety_stop_step_fraction": (
                self.front_safety_stop_steps / self.observed_steps
                if self.observed_steps else None
            ),
            "collision_events_with_details": self.collision_details_written,
            "collision_pair_counts": dict(self.collision_pair_counts),
            "goals_reached": self.goals_reached,
            "goals_reached_completed_episodes": self.goals_reached_completed_episodes,
            "goals_reached_partial_episode": self.goals_reached_partial_episode,
            "front_stop_penalty_sum": self.front_stop_penalty_sum,
            "front_risk_penalty_sum": self.front_risk_penalty_sum,
            "front_risk_step_fraction": (
                self.front_risk_steps / self.observed_steps
                if self.observed_steps else None
            ),
            "lateral_risk_penalty_sum": self.lateral_risk_penalty_sum,
            "lateral_risk_step_fraction": (
                self.lateral_risk_steps / self.observed_steps
                if self.observed_steps else None
            ),
            "minimum_front_braking_margin_m": (
                self.minimum_front_braking_margin
                if math.isfinite(self.minimum_front_braking_margin)
                else None
            ),
            "successful_episodes": self.successful_episodes,
            "zero_goal_episodes": self.zero_goal_episodes,
            "episodes_with_goal": self.episodes_with_goal,
            "collision_episodes": self.collision_episodes,
            "collision_episode_rate": (
                self.collision_episodes / self.episode_count
                if self.episode_count else None
            ),
            "episode_success_rate": (
                self.successful_episodes / self.episode_count
                if self.episode_count
                else None
            ),
            "safety_override_step_fraction": (
                self.safety_override_steps / self.observed_steps
                if self.observed_steps else None
            ),
            "minimum_scale_step_fraction": (
                self.minimum_scale_steps / self.observed_steps
                if self.observed_steps else None
            ),
            "mean_requested_scale": (
                self.requested_scale_sum / self.observed_steps
                if self.observed_steps else None
            ),
            "mean_applied_scale": (
                self.applied_scale_sum / self.observed_steps
                if self.observed_steps else None
            ),
            "minimum_clearance_m": (
                self.minimum_clearance
                if math.isfinite(self.minimum_clearance)
                else None
            ),
            "minimum_sector_ttc_s": (
                self.minimum_sector_ttc
                if math.isfinite(self.minimum_sector_ttc)
                else None
            ),
            "dynamic_obstacle_spawns": self.dynamic_obstacle_spawns,
            "dynamic_obstacle_episodes": self.dynamic_obstacle_episodes,
            "dynamic_obstacle_active_step_fraction": (
                self.dynamic_obstacle_steps / self.observed_steps
                if self.observed_steps else None
            ),
            "retained_obstacle_step_fraction": (
                self.parked_obstacle_steps / self.observed_steps
                if self.observed_steps else None
            ),
            "wall_seconds": time.monotonic() - self.started,
        }


def make_environment(args: argparse.Namespace) -> Monitor:
    if args.suite == "combined":
        goals = AgvRlEnv.COMBINED_GOALS
    else:
        goal = (args.goal_x, args.goal_y, args.goal_yaw)
        if not all(math.isfinite(value) for value in goal):
            raise ValueError("Goal values must be finite")
        goals = (goal,)
    environment = AgvRlEnv(
        goals=goals,
        goal_sampling=args.goal_sampling,
        max_episode_steps=args.max_episode_steps,
        dynamic_obstacles=args.dynamic_obstacles,
        dynamic_obstacle_probability=args.dynamic_obstacle_probability,
        require_front_safety_stop=args.dynamic_obstacles and not args.sac_only,
        include_ttc=not args.sac_only,
        conservative_crossing=args.conservative_crossing,
        retain_dynamic_obstacle_after_crossing=args.retain_crossing_obstacle,
    )
    if args.sac_only:
        try:
            for node, name in (
                ("/rl_velocity_supervisor", "safety_override_enabled"),
                ("/rl_velocity_supervisor", "ttc_observation_enabled"),
                ("/mujoco_ros_bridge", "front_safety_stop_enabled"),
            ):
                client = environment.ros.create_client(
                    GetParameters, f"{node}/get_parameters"
                )
                if not client.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError(f"Cannot verify SAC-only parameter: {node} {name}")
                request = GetParameters.Request()
                request.names = [name]
                response = environment.ros._wait_future(
                    client.call_async(request), 5.0, f"{node} {name}"
                )
                if (
                    len(response.values) != 1
                    or response.values[0].type != ParameterType.PARAMETER_BOOL
                    or response.values[0].bool_value
                ):
                    raise RuntimeError(f"SAC-only requires {node} {name}=false")
        except Exception:
            environment.close()
            raise
    environment._goal_rng = np.random.default_rng(args.seed)
    environment.action_space.seed(args.seed)
    return Monitor(
        environment,
        filename=str(args.run_dir / "training_monitor"),
        info_keywords=(
            "termination_reason",
            "x",
            "y",
            "yaw",
            "goal",
            "distance_to_goal",
            "relative_goal_bearing",
            "linear_velocity",
            "angular_velocity",
            "cmd_vx_m_s",
            "cmd_wz_rad_s",
            "cmd_vel_age_s",
            "requested_action",
            "min_lidar_episode",
            "min_sector_ttc",
            "min_sector_ttc_episode",
            "dynamic_obstacle_spawns",
            "dynamic_obstacle_active",
            "dynamic_obstacle_parked_after_crossing",
            "hazard_persistence",
            "applied_speed_scale",
            "safety_override_active",
            "front_safety_stop_active",
            "front_safety_stop_events",
            "front_safety_stop_reason",
            "front_safety_stop_clearance_m",
            "front_safety_stop_distance_m",
            "front_corridor_clearance_m",
            "front_braking_margin_m",
            "front_proactive_risk",
            "conservative_lateral_risk",
            "front_scan_valid",
            "reward_front_risk",
            "reward_lateral_risk",
            "safety_deadlock_steps",
            "contact_count",
            "collision_event_count",
            "collision_event_detail_count",
            "goals_reached_this_episode",
            "is_success",
        ),
    )


def build_fresh_model(environment: Monitor, args: argparse.Namespace) -> SAC:
    """[MIGRATION] Tạo SAC(...) HOÀN TOÀN MỚI — dùng cho --from-scratch.
    Hyperparameter mặc định khớp với các script train khác trong dự án
    (buffer_size=1e6, batch_size=256, tau=0.005, gamma=0.99)."""
    return SAC(
        "MlpPolicy",
        environment,
        learning_rate=args.learning_rate,
        buffer_size=1_000_000,
        learning_starts=args.warmup_steps,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        device=args.device,
        seed=args.seed,
        verbose=1,
    )


def main() -> int:
    args = parse_args()
    args.sac_only = bool(getattr(args, "sac_only", False))
    args.conservative_crossing = bool(getattr(args, "conservative_crossing", False))
    args.retain_crossing_obstacle = bool(
        getattr(args, "retain_crossing_obstacle", False)
    )
    if args.goal_sampling == "random":
        args.goal_sampling = "random_list"
    if args.goal_sampling == "random_costmap":
        print("GOAL SAMPLING: whole global costmap; --suite/--goal-* do not select goals.", flush=True)
    if args.total_steps <= 0:
        raise ValueError("total-steps must be positive")
    if args.checkpoint_freq <= 0:
        raise ValueError("checkpoint-freq must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps cannot be negative")
    if args.stop_after_consecutive_deadlocks < 0:
        raise ValueError("stop-after-consecutive-deadlocks cannot be negative")
    if args.max_episode_steps <= 0:
        raise ValueError("max-episode-steps must be positive")
    if not math.isfinite(args.dynamic_obstacle_probability) or not (
        0.0 <= args.dynamic_obstacle_probability <= 1.0
    ):
        raise ValueError("dynamic-obstacle-probability must be in [0, 1]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be finite and positive")
    if args.conservative_crossing and not (args.sac_only and args.dynamic_obstacles):
        raise ValueError(
            "--conservative-crossing requires --sac-only and --dynamic-obstacles"
        )
    if args.retain_crossing_obstacle and not args.conservative_crossing:
        raise ValueError(
            "--retain-crossing-obstacle requires --conservative-crossing"
        )
    if args.from_scratch and args.inspect_only:
        raise ValueError(
            "--inspect-only chỉ dùng để kiểm tra 1 checkpoint CÓ SẴN — "
            "không tương thích với --from-scratch (không có checkpoint nào "
            "để kiểm tra)."
        )
    if args.from_scratch and args.replay_buffer is not None:
        print(
            "LƯU Ý: --from-scratch kèm --replay-buffer nghĩa là tạo model "
            "MỚI nhưng tái sử dụng buffer transition CŨ — chỉ hợp lệ nếu "
            "buffer đó được ghi dưới ĐÚNG observation_space hiện tại "
            "(sẽ tự kiểm tra shape trước khi dùng).",
            flush=True,
        )
    if args.sac_only:
        print(
            "SAC-ONLY ABLATION: launch with front_safety_stop_enabled:=false "
            "and rl_supervisor_sac_only.yaml; TTC channels are removed.",
            flush=True,
        )

    model_path: Optional[Path] = None
    stored_version = "n/a (from-scratch)"

    if not args.from_scratch:
        if args.model is None:
            raise ValueError(
                "Phải dùng --from-scratch cho contract SpeedLimit mới, hoặc "
                "truyền rõ --model tới checkpoint tương thích với chế độ đã chọn."
            )
        model_path = resolve_file(args.model, "Model checkpoint")
        stored_version = saved_sb3_version(model_path)
        print(
            f"Model SB3={stored_version}; runtime SB3={sb3_runtime_version}",
            flush=True,
        )
        if stored_version != "unknown" and stored_version != sb3_runtime_version:
            print(
                "WARNING: SB3 versions differ; contract preflight must pass.",
                flush=True,
            )

        if args.inspect_only:
            model = SAC.load(str(model_path), device=args.device)
            contract = validate_model(model, sac_only=args.sac_only)
            print(json.dumps(contract, indent=2), flush=True)
            print("GAZEBO_PHASE2_FINETUNE_PREFLIGHT=PASS", flush=True)
            return 0

        # Reject obsolete observation contracts before creating ROS clients or
        # resetting the simulator. This is especially important after the v5
        # 56D risk-channel migration: a v4 54D checkpoint must not start a run.
        preflight_model = SAC.load(str(model_path), device=args.device)
        validate_model(preflight_model, sac_only=args.sac_only)
        del preflight_model

    args.run_dir = args.run_dir.expanduser().resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    environment: Optional[Monitor] = None
    model: Optional[SAC] = None
    callback: Optional[FineTuneCallback] = None
    interrupted = False
    training_error: Optional[str] = None
    replay_loaded = False
    contract: Dict[str, Any] = {}

    try:
        environment = make_environment(args)
        base_environment = getattr(environment, "unwrapped", environment)
        active_contract = getattr(
            base_environment, "contract_version", AgvRlEnv.CONTRACT_VERSION
        )
        active_reward_contract = getattr(
            base_environment, "reward_contract_version",
            AgvRlEnv.REWARD_CONTRACT_VERSION,
        )

        if args.from_scratch:
            print(
                "FRESH START: tạo SAC(...) HOÀN TOÀN MỚI, KHÔNG load "
                "checkpoint nào (đúng theo kế hoạch: observation contract "
                "vừa đổi, checkpoint cũ không còn tương thích).",
                flush=True,
            )
            model = build_fresh_model(environment, args)
            model.observation_contract_version = active_contract
            model.reward_contract_version = active_reward_contract
            model.replay_buffer.observation_contract_version = (
                active_contract
            )
            model.replay_buffer.reward_contract_version = (
                active_reward_contract
            )
            starting_timesteps = 0
            contract = {
                "contract_version": active_contract,
                "observation_shape": list(environment.observation_space.shape),
                "action_shape": list(environment.action_space.shape),
                "action_low": AgvRlEnv.ACTION_LOW.tolist(),
                "action_high": AgvRlEnv.ACTION_HIGH.tolist(),
                "action_interval_seconds": AgvRlEnv.STEP_WAIT_DURATION,
                "probe_action": None,
                "num_timesteps": 0,
            }
        else:
            model = SAC.load(
                str(model_path),
                env=environment,
                device=args.device,
                learning_rate=args.learning_rate,
            )
            contract = validate_model(model, sac_only=args.sac_only)
            starting_timesteps = int(model.num_timesteps)

        if args.replay_buffer is not None:
            replay_path = resolve_file(args.replay_buffer, "Replay buffer")
            model.load_replay_buffer(str(replay_path))
            # [MIGRATION] Kiểm tra shape ngay sau khi load — chặn sớm nếu
            # buffer được ghi dưới observation_space KHÁC (ví dụ 26D cũ).
            validate_replay_buffer_shape(model, environment)
            replay_loaded = True
            print(f"REPLAY BUFFER loaded: {replay_path}", flush=True)
        else:
            model.learning_starts = starting_timesteps + args.warmup_steps
            print(
                "REPLAY BUFFER: starting empty; "
                f"warmup={args.warmup_steps} MuJoCo steps",
                flush=True,
            )
        model.reward_contract_version = active_reward_contract
        model.replay_buffer.observation_contract_version = active_contract
        model.replay_buffer.reward_contract_version = active_reward_contract

        callback = FineTuneCallback(
            run_dir=args.run_dir,
            starting_timesteps=starting_timesteps,
            checkpoint_freq=args.checkpoint_freq,
            stop_after_consecutive_deadlocks=args.stop_after_consecutive_deadlocks,
            stop_on_collision=args.stop_on_collision,
        )
        print(json.dumps(contract, indent=2), flush=True)
        print("GAZEBO_PHASE2_FINETUNE_PREFLIGHT=PASS", flush=True)
        print(
            f"FINE_TUNE_START model={model_path if model_path else '(from-scratch)'} "
            f"suite={args.suite} sampling={args.goal_sampling} "
            f"additional_steps={args.total_steps} "
            f"learning_rate={args.learning_rate}",
            flush=True,
        )
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            log_interval=1,
            reset_num_timesteps=False,
            progress_bar=args.progress,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("INTERRUPT: saving recoverable state", flush=True)
    except Exception as exc:
        training_error = f"{type(exc).__name__}: {exc}"
        print(f"TRAINING_ERROR: {training_error}", flush=True)
        raise
    finally:
        if model is not None and callback is not None:
            final_stem = (
                "failed_model" if training_error else
                "interrupted_model" if interrupted else
                "stopped_model" if callback.early_stop_reason else "final_model"
            )
            model.save(str(args.run_dir / final_stem))
            model.save(str(args.run_dir / "latest_model"))
            replay_path = args.run_dir / "latest_replay_buffer.pkl"
            try:
                model.save_replay_buffer(str(replay_path))
                replay_value: Optional[str] = str(replay_path)
            except Exception as exc:
                replay_value = None
                print(f"WARNING: replay buffer save failed: {exc}", flush=True)
            summary = callback.summary()
            # Callback._on_step has not run when env.reset() fails inside
            # model.learn(); its counter is then zero although the loaded
            # checkpoint still has its original timesteps.
            summary["final_timesteps"] = int(model.num_timesteps)
            summary["additional_timesteps"] = int(model.num_timesteps) - starting_timesteps
            summary["requested_additional_steps"] = args.total_steps
            summary["trigger_transition_in_replay"] = (
                False if callback.early_stop_reason else None
            )
            summary["replay_safe_to_resume"] = not bool(callback.early_stop_reason)
            summary.update(
                {
                    "interrupted": interrupted,
                    "training_error": training_error,
                    "source_model": str(model_path) if model_path else "from-scratch",
                    "saved_model": str(args.run_dir / f"{final_stem}.zip"),
                    "replay_buffer": replay_value,
                    "replay_buffer_loaded": replay_loaded,
                    "suite": args.suite,
                    "goal_sampling": args.goal_sampling,
                    "seed": args.seed,
                    "learning_rate": args.learning_rate,
                    "max_episode_steps": args.max_episode_steps,
                    "dynamic_obstacles": args.dynamic_obstacles,
                    "sac_only": args.sac_only,
                    "conservative_crossing": args.conservative_crossing,
                    "retain_crossing_obstacle": args.retain_crossing_obstacle,
                    "dynamic_obstacle_probability": args.dynamic_obstacle_probability,
                    "action_interval_seconds": AgvRlEnv.STEP_WAIT_DURATION,
                    "contract": contract,
                    "reward_contract_version": active_reward_contract,
                    "saved_sb3_version": stored_version,
                    "runtime_sb3_version": sb3_runtime_version,
                    "from_scratch": args.from_scratch,
                }
            )
            atomic_write_json(args.run_dir / "training_summary.json", summary)
            print(
                f"FINE_TUNE_END model={summary['saved_model']} "
                f"additional_steps={summary['additional_timesteps']} "
                f"episodes={summary['episodes']} "
                f"reasons={summary['termination_reasons']} "
                f"early_stop={summary['early_stop_reason']}",
                flush=True,
            )
            if summary["early_stopped"]:
                print(
                    "PROBE_REJECTED: stopped_model and latest_model are "
                    "diagnostic artifacts, not approved checkpoints; "
                    "do not resume from latest_replay_buffer.pkl",
                    flush=True,
                )
        if environment is not None:
            environment.close()

    return 130 if interrupted else 0


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
