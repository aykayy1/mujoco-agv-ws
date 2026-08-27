#!/usr/bin/env python3
"""Fine-tune the Gazebo Phase-2 SAC checkpoint on the MuJoCo ROS graph."""

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
from ament_index_python.packages import get_package_share_directory
from stable_baselines3 import SAC, __version__ as sb3_runtime_version
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from agv_rl_env import AgvRlEnv


def default_model_path() -> Path:
    return (
        Path(get_package_share_directory("agv_mujoco_sim"))
        / "transfer_models"
        / "sac_vmax_interrupted_model.zip"
    )


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
            "Resume the 26D/4D Gazebo SAC checkpoint and fine-tune it "
            "against MuJoCo, AMCL and Nav2 MPPI."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Gazebo checkpoint or a later MuJoCo fine-tune checkpoint",
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
        "--warmup-steps",
        type=int,
        default=50,
        help=(
            "New MuJoCo transitions collected before the first gradient "
            "update when no replay buffer is supplied"
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=35)
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
        choices=("cycle", "random"),
        default="cycle",
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


def validate_model(model: SAC) -> Dict[str, Any]:
    expected_observation_shape = (26,)
    expected_action_shape = (4,)
    problems = []
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
        "observation_shape": list(expected_observation_shape),
        "action_shape": list(expected_action_shape),
        "action_low": AgvRlEnv.ACTION_LOW.tolist(),
        "action_high": AgvRlEnv.ACTION_HIGH.tolist(),
        "probe_action": probe_action.tolist(),
        "num_timesteps": int(model.num_timesteps),
    }


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


class FineTuneCallback(BaseCallback):
    """Save recoverable checkpoints and compact episode diagnostics."""

    def __init__(
        self,
        run_dir: Path,
        starting_timesteps: int,
        checkpoint_freq: int,
    ) -> None:
        super().__init__(verbose=1)
        self.run_dir = run_dir
        self.starting_timesteps = int(starting_timesteps)
        self.checkpoint_freq = int(checkpoint_freq)
        self.next_checkpoint = self.checkpoint_freq
        self.episode_count = 0
        self.reason_counts: Counter[str] = Counter()
        self.collision_events = 0
        self.minimum_clearance = math.inf
        self.started = time.monotonic()

    @property
    def additional_steps(self) -> int:
        return int(self.num_timesteps - self.starting_timesteps)

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
            if not done:
                continue
            self.episode_count += 1
            reason = str(info.get("termination_reason") or "unknown")
            self.reason_counts[reason] += 1
            self.collision_events += int(info.get("collision_event_count", 0))
            clearance = float(info.get("min_lidar_episode", math.inf))
            self.minimum_clearance = min(self.minimum_clearance, clearance)
            episode = info.get("episode", {})
            print(
                f"EPISODE n={self.episode_count} "
                f"additional_steps={self.additional_steps} "
                f"length={episode.get('l', '?')} "
                f"reward={float(episode.get('r', math.nan)):+.3f} "
                f"reason={reason} "
                f"distance={float(info.get('distance_to_goal', math.nan)):.3f} "
                f"min_obstacle={clearance:.3f} "
                f"collision_events={int(info.get('collision_event_count', 0))}",
                flush=True,
            )
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "starting_timesteps": self.starting_timesteps,
            "final_timesteps": int(self.num_timesteps),
            "additional_timesteps": self.additional_steps,
            "episodes": self.episode_count,
            "termination_reasons": dict(self.reason_counts),
            "collision_events": self.collision_events,
            "minimum_clearance_m": (
                self.minimum_clearance
                if math.isfinite(self.minimum_clearance)
                else None
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
    )
    environment.action_space.seed(args.seed)
    return Monitor(
        environment,
        filename=str(args.run_dir / "training_monitor"),
        info_keywords=(
            "termination_reason",
            "distance_to_goal",
            "min_lidar_episode",
            "contact_count",
            "collision_event_count",
        ),
    )


def main() -> int:
    args = parse_args()
    if args.total_steps <= 0:
        raise ValueError("total-steps must be positive")
    if args.checkpoint_freq <= 0:
        raise ValueError("checkpoint-freq must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps cannot be negative")
    if args.max_episode_steps <= 0:
        raise ValueError("max-episode-steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be finite and positive")

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
        contract = validate_model(model)
        print(json.dumps(contract, indent=2), flush=True)
        print("GAZEBO_PHASE2_FINETUNE_PREFLIGHT=PASS", flush=True)
        return 0

    args.run_dir = args.run_dir.expanduser().resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    environment: Optional[Monitor] = None
    model: Optional[SAC] = None
    callback: Optional[FineTuneCallback] = None
    interrupted = False
    replay_loaded = False
    contract: Dict[str, Any] = {}

    try:
        environment = make_environment(args)
        model = SAC.load(
            str(model_path),
            env=environment,
            device=args.device,
            learning_rate=args.learning_rate,
        )
        contract = validate_model(model)
        starting_timesteps = int(model.num_timesteps)

        if args.replay_buffer is not None:
            replay_path = resolve_file(args.replay_buffer, "Replay buffer")
            model.load_replay_buffer(str(replay_path))
            replay_loaded = True
            print(f"REPLAY BUFFER loaded: {replay_path}", flush=True)
        else:
            model.learning_starts = starting_timesteps + args.warmup_steps
            print(
                "REPLAY BUFFER: starting empty; "
                f"warmup={args.warmup_steps} MuJoCo steps",
                flush=True,
            )

        callback = FineTuneCallback(
            run_dir=args.run_dir,
            starting_timesteps=starting_timesteps,
            checkpoint_freq=args.checkpoint_freq,
        )
        print(json.dumps(contract, indent=2), flush=True)
        print("GAZEBO_PHASE2_FINETUNE_PREFLIGHT=PASS", flush=True)
        print(
            f"FINE_TUNE_START model={model_path} "
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
    finally:
        if model is not None and callback is not None:
            final_stem = "interrupted_model" if interrupted else "final_model"
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
            summary.update(
                {
                    "interrupted": interrupted,
                    "source_model": str(model_path),
                    "saved_model": str(args.run_dir / f"{final_stem}.zip"),
                    "replay_buffer": replay_value,
                    "replay_buffer_loaded": replay_loaded,
                    "suite": args.suite,
                    "goal_sampling": args.goal_sampling,
                    "seed": args.seed,
                    "learning_rate": args.learning_rate,
                    "contract": contract,
                    "saved_sb3_version": stored_version,
                    "runtime_sb3_version": sb3_runtime_version,
                }
            )
            atomic_write_json(args.run_dir / "training_summary.json", summary)
            print(
                f"FINE_TUNE_END model={summary['saved_model']} "
                f"additional_steps={summary['additional_timesteps']} "
                f"episodes={summary['episodes']} "
                f"reasons={summary['termination_reasons']}",
                flush=True,
            )
        if environment is not None:
            environment.close()

    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
