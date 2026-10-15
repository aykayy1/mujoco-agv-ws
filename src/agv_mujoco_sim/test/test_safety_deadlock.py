"""Regression checks for persistent front-stop recovery during training."""

import sys
from collections import deque
from pathlib import Path

import numpy as np
from action_msgs.msg import GoalStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from agv_rl_env import AgvRlEnv


class FakeRos:
    def begin_action_interval(self):
        pass

    def wait_for_sim_duration(self, duration):
        pass


def make_env():
    env = AgvRlEnv.__new__(AgvRlEnv)
    env.ros = FakeRos()
    env.goal = (8.0, 0.0, 0.0)
    env.episode_step_count = 0
    env.max_episode_steps = 180
    env.previous_distance = 8.0
    env.previous_applied_action = np.array([0.9], dtype=np.float32)
    env.goals_reached_this_episode = 0
    env.safety_deadlock_steps = 0
    env.safety_deadlock_history = deque(maxlen=env.SAFETY_DEADLOCK_STEPS)
    env._needs_hard_reset = False
    env._maybe_start_training_obstacle = lambda: None
    env._apply_action = lambda action: np.asarray(action, dtype=np.float32)
    state = {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "relative_goal_bearing": 0.0,
        "distance_to_goal": 8.0,
        "min_lidar_step": 0.62,
        "collision_state": False,
        "contact_count": 0,
        "linear_velocity": 0.0,
        "angular_velocity": 0.0,
        "applied_speed_scale": 0.9,
        "nav_status": None,
        "front_safety_stop_active": True,
        "front_safety_stop_reason": 2,
        "front_safety_stop_clearance_m": 0.62,
        "front_safety_stop_distance_m": 0.75,
        "dynamic_obstacle_active": False,
    }
    env._observation = lambda: (
        np.zeros(env.OBSERVATION_DIM, dtype=np.float32), state.copy()
    )
    env._select_next_goal = lambda: None
    env._send_selected_goal = lambda: None
    return env, state


def test_persistent_corridor_stop_ends_episode_and_requests_hard_reset():
    env, _ = make_env()
    for _ in range(env.SAFETY_DEADLOCK_STEPS - 1):
        _, _, terminated, truncated, info = env.step([0.9])
        assert not terminated and not truncated

    _, reward, terminated, truncated, info = env.step([0.9])
    assert terminated and not truncated
    assert info["termination_reason"] == "safety_deadlock"
    assert info["safety_deadlock_steps"] == env.SAFETY_DEADLOCK_STEPS
    assert info["is_success"] is False
    assert env._needs_hard_reset
    assert reward < -25.0


def test_movement_and_transient_stop_clear_deadlock_counter():
    env, state = make_env()
    for _ in range(8):
        env.step([0.9])
    state["x"] = 0.15
    for _ in range(12):
        _, _, terminated, _, _ = env.step([0.9])
        assert not terminated

    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    for _ in range(20):
        _, _, terminated, _, _ = env.step([0.9])
        assert not terminated
    assert env.safety_deadlock_steps == 0


def test_turning_and_sensor_fault_do_not_trigger_corridor_deadlock():
    env, state = make_env()
    for n in range(25):
        state["yaw"] = 0.02 * n
        _, _, terminated, _, _ = env.step([0.9])
        assert not terminated

    state["angular_velocity"] = 0.0
    state["front_safety_stop_reason"] = 1
    for _ in range(25):
        _, _, terminated, _, _ = env.step([0.9])
        assert not terminated
    assert env.safety_deadlock_steps == 0


def test_intermittent_corridor_stop_still_triggers_deadlock():
    env, state = make_env()
    for n in range(env.SAFETY_DEADLOCK_STEPS):
        state["front_safety_stop_active"] = n % 4 != 0
        state["front_safety_stop_reason"] = 2 if n % 4 != 0 else 0
        _, _, terminated, _, info = env.step([0.9])
    assert terminated
    assert info["termination_reason"] == "safety_deadlock"
    assert info["safety_deadlock_steps"] == 15


def test_nav_abort_does_not_hide_persistent_safety_deadlock():
    env, state = make_env()
    for n in range(env.SAFETY_DEADLOCK_STEPS):
        state["nav_status"] = (
            GoalStatus.STATUS_ABORTED if n in (5, 11, 17) else None
        )
        _, _, terminated, _, info = env.step([0.9])
    assert terminated
    assert info["termination_reason"] == "safety_deadlock"


def test_blocked_high_speed_request_has_larger_penalty_than_low_speed():
    high, _ = make_env()
    low, _ = make_env()
    _, high_reward, _, _, high_info = high.step([0.9])
    _, low_reward, _, _, low_info = low.step([0.15])
    assert high_info["reward_front_stop"] < low_info["reward_front_stop"] < 0.0
    assert high_reward < low_reward


def test_sac_only_motion_stall_truncates_and_requests_hard_reset():
    env, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    env.enable_motion_stall_recovery = True
    env.motion_stall_history = deque(maxlen=env.MOTION_STALL_STEPS)
    for _ in range(env.MOTION_STALL_STEPS - 1):
        _, _, terminated, truncated, _ = env.step([0.9])
        assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step([0.9])
    assert not terminated and truncated
    assert info["termination_reason"] == "motion_stall"
    assert info["hard_reset_next_episode"] is True
    assert info["is_success"] is False


def test_nav_abort_does_not_reset_motion_stall_window():
    env, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    env.enable_motion_stall_recovery = True
    env.motion_stall_history = deque(maxlen=env.MOTION_STALL_STEPS)
    for n in range(env.MOTION_STALL_STEPS):
        state["nav_status"] = (
            GoalStatus.STATUS_ABORTED if n % 7 == 0 else None
        )
        _, _, _, truncated, info = env.step([0.9])
    assert truncated
    assert info["termination_reason"] == "motion_stall"


def test_motion_stall_does_not_trigger_with_meaningful_displacement():
    env, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    env.enable_motion_stall_recovery = True
    env.motion_stall_history = deque(maxlen=env.MOTION_STALL_STEPS)
    for n in range(env.MOTION_STALL_STEPS + 5):
        state["x"] = 0.02 * n
        _, _, terminated, truncated, _ = env.step([0.9])
        assert not terminated and not truncated
    assert not env._needs_hard_reset


def test_max_steps_without_goal_hard_resets_in_every_arm():
    env, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    env.max_episode_steps = 1
    env.enable_motion_stall_recovery = True
    env.motion_stall_history = deque(maxlen=env.MOTION_STALL_STEPS)
    _, _, _, truncated, info = env.step([0.9])
    assert truncated and info["termination_reason"] == "max_steps"
    assert info["hard_reset_next_episode"] is True

    with_goal, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    with_goal.max_episode_steps = 1
    with_goal.goals_reached_this_episode = 1
    with_goal.enable_motion_stall_recovery = True
    with_goal.motion_stall_history = deque(maxlen=with_goal.MOTION_STALL_STEPS)
    _, _, _, truncated, info = with_goal.step([0.9])
    assert truncated and info["hard_reset_next_episode"] is False

    baseline, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    baseline.max_episode_steps = 1
    _, _, _, truncated, info = baseline.step([0.9])
    assert truncated and info["hard_reset_next_episode"] is True


def test_rotation_and_missing_command_stalls_have_distinct_reasons():
    rotating, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    rotating.enable_motion_stall_recovery = True
    rotating.motion_stall_history = deque(maxlen=rotating.MOTION_STALL_STEPS)
    for n in range(rotating.MOTION_STALL_STEPS):
        state["yaw"] = 0.08 * n
        _, _, _, _, info = rotating.step([0.9])
    assert info["termination_reason"] == "rotation_stall"

    silent, state = make_env()
    state["front_safety_stop_active"] = False
    state["front_safety_stop_reason"] = 0
    state["cmd_vel_age_s"] = 2.0
    silent.enable_motion_stall_recovery = True
    silent.motion_stall_history = deque(maxlen=silent.MOTION_STALL_STEPS)
    for _ in range(silent.MOTION_STALL_STEPS):
        _, _, _, _, info = silent.step([0.9])
    assert info["termination_reason"] == "command_stall"
