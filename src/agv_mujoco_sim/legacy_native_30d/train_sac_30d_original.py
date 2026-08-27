#!/usr/bin/env python3

"""Train and evaluate SAC for the verified AGV RL environment.

This is the Stage-1 pipeline check. It intentionally uses one ROS/Gym
environment at a time because a second Nav2 goal client would interfere with
the active episode.

Examples:
    python3 -u train_sac.py train --total-steps 2000
    python3 -u train_sac.py train --total-steps 20000 --resume runs/sac_agv/latest_model.zip
    python3 -u train_sac.py evaluate --model runs/sac_agv/final_model.zip --episodes 10
    python3 -u train_sac.py baseline --episodes 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise SystemExit(
        "Missing Stable-Baselines3. Install it with:\n"
        "  python3 -m pip install --user 'stable-baselines3[extra]>=2.3,<3'\n"
        f"Original import error: {exc}"
    ) from exc

from agv_rl_env import AgvRlEnv


DEFAULT_RUN_DIR = Path("runs/sac_agv")


def atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


class SacTrainingCallback(BaseCallback):
    """Checkpoint models and print compact episode summaries."""

    def __init__(
        self,
        run_dir: Path,
        checkpoint_freq: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_dir = run_dir
        self.checkpoint_freq = max(1, int(checkpoint_freq))
        self.episode_count = 0
        self.reason_counts: Counter[str] = Counter()
        self.start_wall_time = time.monotonic()
        self.last_checkpoint_step = 0

    def _save_checkpoint(self, label: str) -> None:
        checkpoint_path = self.run_dir / label
        self.model.save(str(checkpoint_path))
        self.model.save(str(self.run_dir / "latest_model"))
        self.model.save_replay_buffer(
            str(self.run_dir / "latest_replay_buffer.pkl")
        )
        if self.verbose:
            print(
                f"CHECKPOINT: step={self.num_timesteps} "
                f"path={checkpoint_path}.zip",
                flush=True,
            )

    def _on_step(self) -> bool:
        if (
            self.num_timesteps - self.last_checkpoint_step
            >= self.checkpoint_freq
        ):
            self._save_checkpoint(
                f"checkpoint_{self.num_timesteps:09d}_steps"
            )
            self.last_checkpoint_step = self.num_timesteps

        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        if dones is None or infos is None:
            return True

        for index, done in enumerate(dones):
            if not done:
                continue
            self.episode_count += 1
            info = infos[index]
            reason = str(info.get("termination_reason") or "unknown")
            self.reason_counts[reason] += 1
            episode = info.get("episode", {})
            episode_reward = episode.get(
                "r",
                float(rewards[index]) if rewards is not None else math.nan,
            )
            episode_length = episode.get("l", "?")
            print(
                f"EPISODE: n={self.episode_count} "
                f"steps={self.num_timesteps} "
                f"length={episode_length} "
                f"reward={float(episode_reward):+.3f} "
                f"reason={reason} "
                f"distance={float(info.get('distance_to_goal', math.nan)):.3f} "
                f"contacts={int(info.get('contact_count', 0))}",
                flush=True,
            )

        return True

    def training_summary(self) -> Dict[str, Any]:
        return {
            "timesteps": int(self.num_timesteps),
            "episodes_this_run": int(self.episode_count),
            "termination_reasons": dict(self.reason_counts),
            "wall_seconds": float(time.monotonic() - self.start_wall_time),
        }


def make_monitored_env(
    run_dir: Path,
    seed: int,
    monitor_name: str,
) -> Monitor:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = AgvRlEnv()
    env.action_space.seed(seed)
    return Monitor(
        env,
        filename=str(run_dir / monitor_name),
        info_keywords=(
            "termination_reason",
            "distance_to_goal",
            "min_lidar",
            "contact_count",
        ),
    )


def new_sac_model(env: Monitor, args: argparse.Namespace) -> SAC:
    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=None,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(args.run_dir / "tensorboard"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )


def run_train(args: argparse.Namespace) -> int:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = make_monitored_env(
        args.run_dir,
        args.seed,
        "training_monitor",
    )
    callback = SacTrainingCallback(
        run_dir=args.run_dir,
        checkpoint_freq=args.checkpoint_freq,
    )

    interrupted = False
    model: Optional[SAC] = None
    try:
        if args.resume is not None:
            print(f"RESUME: {args.resume}", flush=True)
            model = SAC.load(
                str(args.resume),
                env=env,
                device=args.device,
            )
            replay_path = (
                args.replay_buffer
                if args.replay_buffer is not None
                else args.run_dir / "latest_replay_buffer.pkl"
            )
            if replay_path.exists():
                model.load_replay_buffer(str(replay_path))
                print(f"REPLAY BUFFER: loaded {replay_path}", flush=True)
            else:
                print(
                    "REPLAY BUFFER: not found; continuing with an empty buffer",
                    flush=True,
                )
            reset_num_timesteps = False
        else:
            model = new_sac_model(env, args)
            reset_num_timesteps = True

        print(
            "TRAIN START:",
            f"additional_steps={args.total_steps}",
            f"seed={args.seed}",
            f"device={args.device}",
            flush=True,
        )
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            log_interval=1,
            tb_log_name="SAC",
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=args.progress,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("INTERRUPT: saving the current model safely", flush=True)
    finally:
        if model is not None:
            final_stem = "interrupted_model" if interrupted else "final_model"
            model.save(str(args.run_dir / final_stem))
            model.save(str(args.run_dir / "latest_model"))
            replay_path = args.run_dir / "replay_buffer.pkl"
            try:
                model.save_replay_buffer(str(replay_path))
                model.save_replay_buffer(
                    str(args.run_dir / "latest_replay_buffer.pkl")
                )
                replay_value: Optional[str] = str(replay_path)
            except Exception as exc:
                replay_value = None
                print(
                    f"WARNING: could not save replay buffer: {exc}",
                    flush=True,
                )
            summary = callback.training_summary()
            summary.update(
                {
                    "interrupted": interrupted,
                    "model": str(args.run_dir / f"{final_stem}.zip"),
                    "replay_buffer": replay_value,
                }
            )
            atomic_json_write(args.run_dir / "training_summary.json", summary)
            print(
                "TRAIN END:",
                f"model={summary['model']}",
                f"timesteps={summary['timesteps']}",
                f"episodes={summary['episodes_this_run']}",
                f"reasons={summary['termination_reasons']}",
                flush=True,
            )
        env.close()

    return 130 if interrupted else 0


def evaluate_controller(
    args: argparse.Namespace,
    model: Optional[SAC],
) -> int:
    mode = "baseline" if model is None else "sac"
    env = make_monitored_env(
        args.run_dir,
        args.seed,
        f"{mode}_evaluation_monitor",
    )
    records: List[Dict[str, Any]] = []

    try:
        for episode_index in range(args.episodes):
            observation, reset_info = env.reset(
                seed=args.seed + episode_index
            )
            total_reward = 0.0
            step_count = 0
            info: Dict[str, Any] = dict(reset_info)

            while True:
                if model is None:
                    action = np.asarray([1.0], dtype=np.float32)
                else:
                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )

                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                total_reward += float(reward)
                step_count += 1

                if terminated or truncated:
                    break

            record = {
                "episode": episode_index + 1,
                "reward": total_reward,
                "steps": step_count,
                "reason": str(
                    info.get("termination_reason") or "unknown"
                ),
                "distance_to_goal": float(
                    info.get("distance_to_goal", math.nan)
                ),
                "min_lidar": float(info.get("min_lidar", math.nan)),
                "contact_count": int(info.get("contact_count", 0)),
            }
            records.append(record)
            print(
                f"EVAL: mode={mode} "
                f"episode={record['episode']:03d} "
                f"steps={record['steps']:03d} "
                f"reward={record['reward']:+.3f} "
                f"reason={record['reason']} "
                f"distance={record['distance_to_goal']:.3f} "
                f"min_lidar={record['min_lidar']:.3f} "
                f"contacts={record['contact_count']}",
                flush=True,
            )
    finally:
        env.close()

    reasons = Counter(record["reason"] for record in records)
    summary = {
        "mode": mode,
        "episodes": len(records),
        "success_rate": (
            reasons.get("success", 0) / len(records) if records else 0.0
        ),
        "mean_reward": (
            float(np.mean([record["reward"] for record in records]))
            if records
            else math.nan
        ),
        "mean_steps": (
            float(np.mean([record["steps"] for record in records]))
            if records
            else math.nan
        ),
        "termination_reasons": dict(reasons),
        "records": records,
    }
    output_path = args.run_dir / f"{mode}_evaluation.json"
    atomic_json_write(output_path, summary)
    print(
        "EVAL SUMMARY:",
        f"mode={mode}",
        f"success_rate={summary['success_rate']:.1%}",
        f"mean_reward={summary['mean_reward']:+.3f}",
        f"mean_steps={summary['mean_steps']:.2f}",
        f"reasons={summary['termination_reasons']}",
        f"output={output_path}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAC training/evaluation for AgvRlEnv",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
    )
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:0, ...",
    )

    train_parser = subparsers.add_parser(
        "train",
        parents=[common],
        help="Train a new SAC model or resume a checkpoint",
    )
    train_parser.add_argument("--total-steps", type=int, default=2000)
    train_parser.add_argument("--checkpoint-freq", type=int, default=1000)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--replay-buffer",
        type=Path,
        help=(
            "Replay buffer used with --resume; defaults to "
            "<run-dir>/latest_replay_buffer.pkl"
        ),
    )
    train_parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    train_parser.add_argument("--buffer-size", type=int, default=100_000)
    train_parser.add_argument("--learning-starts", type=int, default=500)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show the tqdm progress bar",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        parents=[common],
        help="Evaluate a trained model deterministically",
    )
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--episodes", type=int, default=10)

    baseline_parser = subparsers.add_parser(
        "baseline",
        parents=[common],
        help="Evaluate the fixed maximum-speed MPPI baseline",
    )
    baseline_parser.add_argument("--episodes", type=int, default=10)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "total_steps", 1) <= 0:
        raise SystemExit("--total-steps must be positive")
    if getattr(args, "checkpoint_freq", 1) <= 0:
        raise SystemExit("--checkpoint-freq must be positive")
    if getattr(args, "episodes", 1) <= 0:
        raise SystemExit("--episodes must be positive")

    model_path = getattr(args, "model", None)
    if model_path is not None and not model_path.exists():
        zip_path = Path(str(model_path) + ".zip")
        if zip_path.exists():
            args.model = zip_path
        else:
            raise SystemExit(f"Model does not exist: {model_path}")

    resume_path = getattr(args, "resume", None)
    if resume_path is not None and not resume_path.exists():
        zip_path = Path(str(resume_path) + ".zip")
        if zip_path.exists():
            args.resume = zip_path
        else:
            raise SystemExit(f"Checkpoint does not exist: {resume_path}")

    replay_path = getattr(args, "replay_buffer", None)
    if replay_path is not None and not replay_path.exists():
        raise SystemExit(f"Replay buffer does not exist: {replay_path}")


def main() -> int:
    # SIGTERM follows the same safe-save path as Ctrl-C.
    signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: signal.raise_signal(signal.SIGINT),
    )
    args = build_parser().parse_args()
    validate_args(args)

    if args.command == "train":
        return run_train(args)
    if args.command == "evaluate":
        model = SAC.load(str(args.model), device=args.device)
        return evaluate_controller(args, model)
    if args.command == "baseline":
        return evaluate_controller(args, model=None)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
