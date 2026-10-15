"""Checks for reward compatibility and goals in unfinished episodes."""

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from agv_rl_env import AgvRlEnv
from train_sac import FineTuneCallback, validate_replay_buffer_shape
import train_sac


def test_old_reward_replay_is_rejected():
    model = SimpleNamespace(replay_buffer=SimpleNamespace(
        observation_contract_version=AgvRlEnv.CONTRACT_VERSION,
        observations=np.zeros((1, 1, AgvRlEnv.OBSERVATION_DIM), dtype=np.float32),
        actions=np.zeros((1, 1, 1), dtype=np.float32),
    ))
    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(AgvRlEnv.OBSERVATION_DIM,)),
        action_space=SimpleNamespace(shape=(1,)),
    )
    with pytest.raises(ValueError, match="Replay reward contract"):
        validate_replay_buffer_shape(model, env)


def test_summary_counts_goals_before_episode_ends(tmp_path):
    callback = FineTuneCallback(tmp_path, starting_timesteps=0, checkpoint_freq=100)
    callback.num_timesteps = 1
    callback.locals = {
        "dones": [False],
        "infos": [{"goal_event": "succeeded", "goals_reached_this_episode": 1}],
    }
    callback._on_step()
    assert callback.summary()["goals_reached"] == 1
    assert callback.summary()["goals_reached_partial_episode"] == 1

    callback.num_timesteps = 2
    callback.locals = {
        "dones": [True],
        "infos": [{"goal_event": None, "goals_reached_this_episode": 1}],
    }
    callback._on_step()
    summary = callback.summary()
    assert summary["goals_reached"] == 1
    assert summary["goals_reached_completed_episodes"] == 1
    assert summary["goals_reached_partial_episode"] == 0


def test_probe_stops_after_two_consecutive_deadlocks(tmp_path):
    callback = FineTuneCallback(
        tmp_path, starting_timesteps=5400, checkpoint_freq=100,
        stop_after_consecutive_deadlocks=2,
    )
    for total, reason, should_continue in (
        (5401, "safety_deadlock", True),
        (5402, "max_steps", True),
        (5403, "safety_deadlock", True),
        (5404, "safety_deadlock", False),
    ):
        callback.num_timesteps = total
        callback.locals = {
            "dones": [True],
            "infos": [{"termination_reason": reason}],
        }
        assert callback._on_step() is should_continue
    summary = callback.summary()
    assert summary["early_stopped"] is True
    assert summary["early_stop_reason"] == "2_consecutive_deadlocks"
    assert summary["additional_timesteps"] == 4


def test_probe_counts_motion_stall_as_deadlock(tmp_path):
    callback = FineTuneCallback(
        tmp_path, starting_timesteps=20000, checkpoint_freq=100,
        stop_after_consecutive_deadlocks=2,
    )
    for total, reason, should_continue in (
        (20001, "motion_stall", True),
        (20002, "safety_deadlock", False),
    ):
        callback.num_timesteps = total
        callback.locals = {
            "dones": [True],
            "infos": [{"termination_reason": reason}],
        }
        assert callback._on_step() is should_continue
    assert callback.summary()["early_stop_reason"] == "2_consecutive_deadlocks"


def test_probe_distinguishes_rotation_and_command_stalls(tmp_path):
    callback = FineTuneCallback(
        tmp_path, starting_timesteps=20000, checkpoint_freq=100,
        stop_after_consecutive_deadlocks=2,
    )
    for total, reason, keep_running in (
        (20001, "rotation_stall", True),
        (20002, "command_stall", False),
    ):
        callback.num_timesteps = total
        callback.locals = {"dones": [True], "infos": [{
            "termination_reason": reason, "goals_reached_this_episode": 0,
        }]}
        assert callback._on_step() is keep_running
    report = callback.summary()
    assert report["zero_goal_episodes"] == 2
    assert report["collision_episode_rate"] == 0.0


def test_collision_probe_stop_is_opt_in(tmp_path):
    off = FineTuneCallback(tmp_path, starting_timesteps=0, checkpoint_freq=100)
    on = FineTuneCallback(
        tmp_path, starting_timesteps=0, checkpoint_freq=100,
        stop_on_collision=True,
    )
    for callback, should_continue in ((off, True), (on, False)):
        callback.num_timesteps = 1
        callback.locals = {
            "dones": [True],
            "infos": [{"termination_reason": "collision"}],
        }
        assert callback._on_step() is should_continue
    assert off.summary()["early_stopped"] is False
    assert on.summary()["early_stop_reason"] == "collision"


def test_early_stopped_probe_saves_rejected_artifact_not_final_model(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        goal_sampling="fixed", total_steps=10, checkpoint_freq=100,
        warmup_steps=0, stop_after_consecutive_deadlocks=2,
        stop_on_collision=False, max_episode_steps=180,
        dynamic_obstacle_probability=0.0, learning_rate=1e-4,
        from_scratch=True, inspect_only=False, replay_buffer=None,
        model=None, run_dir=tmp_path, device="cpu", seed=1,
        progress=False, suite="single", dynamic_obstacles=False,
    )

    class FakeModel:
        def __init__(self):
            self.num_timesteps = 0
            self.replay_buffer = SimpleNamespace()

        def learn(self, *, callback, **kwargs):
            callback.model = self
            for step in (1, 2):
                self.num_timesteps = step
                callback.num_timesteps = step
                callback.locals = {
                    "dones": [True],
                    "infos": [{"termination_reason": "safety_deadlock"}],
                }
                if not callback._on_step():
                    break

        def save(self, path):
            Path(str(path) + ".zip").write_bytes(b"fake model")

        def save_replay_buffer(self, path):
            Path(path).write_bytes(b"fake replay")

    environment = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(AgvRlEnv.OBSERVATION_DIM,)),
        action_space=SimpleNamespace(shape=(1,)),
        close=lambda: None,
    )
    monkeypatch.setattr(train_sac, "parse_args", lambda: args)
    monkeypatch.setattr(train_sac, "make_environment", lambda parsed: environment)
    monkeypatch.setattr(train_sac, "build_fresh_model", lambda env, parsed: FakeModel())

    assert train_sac.main() == 0

    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["early_stopped"] is True
    assert summary["early_stop_reason"] == "2_consecutive_deadlocks"
    assert summary["saved_model"].endswith("stopped_model.zip")
    assert summary["replay_safe_to_resume"] is False
    assert summary["trigger_transition_in_replay"] is False
    assert (tmp_path / "stopped_model.zip").exists()
    assert not (tmp_path / "final_model.zip").exists()
