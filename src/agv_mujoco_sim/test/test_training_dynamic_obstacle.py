"""Checks for the optional moving obstacle used during SAC training."""

import threading
from types import SimpleNamespace

import numpy as np

from agv_rl_env import AgvRlEnv, AgvRosInterface


class FakeRos:
    def __init__(self, free=True):
        self.free = free
        self.started = []

    def current_state(self):
        return {"dynamic_obstacle_active": False}

    def get_map_pose(self):
        return (0.0, 0.0, 0.0)

    def is_position_free(self, x, y):
        return self.free

    def start_dynamic_obstacle(self, start, end, speed):
        self.started.append((start, end, speed))


def make_env(free=True, enabled=True):
    env = AgvRlEnv.__new__(AgvRlEnv)
    env.ros = FakeRos(free=free)
    env.np_random = np.random.default_rng(42)
    env.dynamic_obstacles = enabled
    env.dynamic_obstacle_probability = 1.0
    env.episode_step_count = 1
    env.next_dynamic_obstacle_step = 1
    return env


def test_front_crossing_starts_only_when_enabled():
    env = make_env()
    env._maybe_start_training_obstacle()
    assert len(env.ros.started) == 1
    start, end, speed = env.ros.started[0]
    assert 3.2 <= start[0] <= 3.8
    assert abs(start[1]) == 1.5
    assert end[0] == start[0]
    assert end[1] == -start[1]
    assert 0.45 <= speed <= 0.60
    assert env.next_dynamic_obstacle_step == 21

    disabled = make_env(enabled=False)
    disabled._maybe_start_training_obstacle()
    assert disabled.ros.started == []


def test_blocked_path_is_skipped_and_retried():
    env = make_env(free=False)
    env._maybe_start_training_obstacle()
    assert env.ros.started == []
    assert env.next_dynamic_obstacle_step == 6


def test_probability_zero_does_not_spawn():
    env = make_env()
    env.dynamic_obstacle_probability = 0.0
    env._maybe_start_training_obstacle()
    assert env.ros.started == []


def test_retained_obstacle_blocks_new_spawn_until_goal_reset():
    env = make_env()
    env.ros.dynamic_obstacle_parked_after_crossing = True
    env._maybe_start_training_obstacle()
    assert env.ros.started == []


def test_crossing_endpoint_is_retained_only_when_opted_in():
    for retain in (False, True):
        published = []
        backend = SimpleNamespace(
            lock=threading.RLock(), sim_time=10.0,
            dynamic_obstacle_trajectory=((1.0, -1.0), (1.0, 1.0), 0.0, 5.0),
            retain_dynamic_obstacle_after_crossing=retain,
            dynamic_obstacle_parked_after_crossing=False,
            DYNAMIC_OBSTACLE_PARK=(20.0, 20.0, 0.5),
            _publish_dynamic_obstacle=published.append,
        )
        AgvRosInterface._dynamic_obstacle_tick(backend)
        assert backend.dynamic_obstacle_trajectory is None
        assert backend.dynamic_obstacle_parked_after_crossing is retain
        assert published == [(1.0, 1.0, 0.5) if retain else (20.0, 20.0, 0.5)]


def test_conservative_crossing_matches_warehouse_encounter_ranges():
    env = make_env()
    env.conservative_crossing = True
    env._maybe_start_training_obstacle()
    assert len(env.ros.started) == 1
    start, end, speed = env.ros.started[0]
    assert 1.7 <= start[0] <= 2.8
    assert 1.0 <= abs(start[1]) <= 1.5
    assert end[0] == start[0]
    assert end[1] == -start[1]
    assert 0.25 <= speed <= 0.55


def test_conservative_risk_uses_current_lateral_scan_only():
    state = {
        "front_scan_valid": True,
        "front_scan_age_s": 0.0,
        "front_scan_x": np.array([2.0, 2.02, 2.04], dtype=np.float32),
        "front_scan_y": np.array([1.0, 1.02, 1.04], dtype=np.float32),
    }
    assert 0.4 < AgvRlEnv._conservative_lateral_risk(state) < 0.5
    state["front_scan_y"] = np.array([0.1, 0.12, 0.14], dtype=np.float32)
    assert AgvRlEnv._conservative_lateral_risk(state) == 0.0
    state["front_scan_y"] = np.array([-1.0, -1.02, -1.04], dtype=np.float32)
    assert AgvRlEnv._conservative_lateral_risk(state) > 0.0
    state["front_scan_age_s"] = 1.0
    assert AgvRlEnv._conservative_lateral_risk(state) == 0.0
