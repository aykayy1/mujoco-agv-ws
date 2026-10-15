"""Goal candidates must have free space for the whole robot clearance disk."""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from agv_rl_env import AgvRlEnv, AgvRosInterface


def make_costmap(*, blocked_index=None, blocked_value=100):
    data = [0] * 25
    if blocked_index is not None:
        data[blocked_index] = blocked_value
    info = SimpleNamespace(
        width=5,
        height=5,
        resolution=1.0,
        origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
    )
    return SimpleNamespace(info=info, data=data)


def make_ros(costmap):
    ros = AgvRosInterface.__new__(AgvRosInterface)
    ros.lock = threading.RLock()
    ros.costmap = costmap
    ros.costmap_occupied_threshold = 50
    return ros


@pytest.mark.parametrize("blocked_value", [-1, 100])
def test_clearance_disk_rejects_neighboring_unknown_or_occupied_cell(blocked_value):
    # The center cell (2, 2) is free. The neighboring (3, 2) cell intersects
    # the 0.8 m clearance disk centered at (2.5, 2.5).
    ros = make_ros(make_costmap(blocked_index=2 * 5 + 3, blocked_value=blocked_value))
    assert ros.is_position_free(2.5, 2.5, clearance_radius_m=0.0)
    assert not ros.is_position_free(2.5, 2.5, clearance_radius_m=0.8)


def test_clearance_disk_rejects_outside_map_even_when_center_is_free():
    ros = make_ros(make_costmap())
    assert ros.is_position_free(0.5, 0.5, clearance_radius_m=0.0)
    assert not ros.is_position_free(0.5, 0.5, clearance_radius_m=0.8)
    assert not ros.is_position_free(0.1, 2.5, clearance_radius_m=0.2)


def test_clearance_disk_uses_actual_position_within_center_cell():
    # A small disk close to a cell boundary reaches the neighboring obstacle
    # even though the occupied cell's centre is 1 m from the centre cell.
    ros = make_ros(make_costmap(blocked_index=2 * 5 + 3))
    assert not ros.is_position_free(2.9, 2.5, clearance_radius_m=0.2)


def test_clearance_disk_accepts_free_interior_and_rejects_invalid_radius():
    ros = make_ros(make_costmap())
    assert ros.is_position_free(2.5, 2.5, clearance_radius_m=0.8)
    with pytest.raises(ValueError, match="clearance_radius_m"):
        ros.is_position_free(2.5, 2.5, clearance_radius_m=-0.1)
    with pytest.raises(ValueError, match="clearance_radius_m"):
        ros.is_position_free(2.5, 2.5, clearance_radius_m=float("nan"))


class GoalRos:
    def __init__(self, free=True):
        self.lock = threading.RLock()
        self.costmap = make_costmap()
        self.free = free
        self.calls = []

    def get_map_pose(self):
        return 0.0, 0.0, 0.0

    def is_position_free(self, x, y, clearance_radius_m=0.0):
        self.calls.append((x, y, clearance_radius_m))
        return self.free


def make_goal_env(free=True):
    env = AgvRlEnv.__new__(AgvRlEnv)
    env.ros = GoalRos(free=free)
    env._goal_rng = np.random.default_rng(13)
    env.goal = (4.0, 4.0, 0.0)
    env.min_goal_distance = 1.0
    env.max_goal_generation_attempts = 1
    return env


def test_generated_goal_requests_robot_clearance():
    env = make_goal_env()
    env._generate_new_goal()
    assert env.ros.calls
    assert all(
        radius == env.GOAL_CLEARANCE_M
        for _, _, radius in env.ros.calls
    )


def test_no_valid_goal_fails_closed_without_replacing_previous_goal():
    env = make_goal_env(free=False)
    previous_goal = env.goal
    with pytest.raises(RuntimeError, match="goal|Goal|free|clearance"):
        env._generate_new_goal()
    assert env.goal == previous_goal
    assert env.ros.calls
    assert all(
        radius == env.GOAL_CLEARANCE_M
        for _, _, radius in env.ros.calls
    )
