"""Reset action semantics and raw command telemetry before the physical gate."""

import sys
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Twist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import agv_rl_env
from agv_rl_env import AgvRlEnv, AgvRosInterface


class ResetRos:
    def __init__(self, applied_scale=0.42):
        self.applied_scale = applied_scale
        self.sequence = 0
        self.randomization_client = SimpleNamespace(call_async=lambda request: object())

    def current_state(self):
        self.sequence += 1
        return {
            "odom_seq": self.sequence,
            "scan_seq": self.sequence,
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
            "applied_speed_scale": self.applied_scale,
        }

    def park_dynamic_obstacle(self):
        pass

    def cancel_navigation(self):
        pass

    def _wait_future(self, future, timeout, description):
        return SimpleNamespace(success=True, message="nominal")

    def reset_episode_flags(self):
        pass

    def clear_costmaps(self, timeout):
        pass


@pytest.mark.parametrize("hard_reset", [False, True])
def test_reset_preserves_soft_scale_and_uses_applied_acknowledgement(
    monkeypatch, hard_reset
):
    monkeypatch.setattr(agv_rl_env.time, "sleep", lambda duration: None)
    env = AgvRlEnv.__new__(AgvRlEnv)
    env.ros = ResetRos(applied_scale=0.42)
    env._needs_hard_reset = hard_reset
    hard_reset_calls = []
    env._perform_hard_reset = lambda: hard_reset_calls.append(True)
    requested = []
    env._apply_action = lambda action: requested.append(float(action[0]))
    env._select_next_goal = lambda: None
    env._send_selected_goal = lambda: None
    env._observation = lambda: (
        np.zeros(env.OBSERVATION_DIM, dtype=np.float32),
        {"distance_to_goal": 2.0},
    )
    env.safety_deadlock_history = deque(maxlen=env.SAFETY_DEADLOCK_STEPS)

    _, info = env.reset(seed=1)

    assert info["reset_ok"]
    assert requested == ([1.0] if hard_reset else [])
    assert hard_reset_calls == ([True] if hard_reset else [])
    np.testing.assert_allclose(env.previous_applied_action, [0.42], atol=1e-6)


def test_raw_cmd_vel_callback_records_pre_gate_command():
    ros = AgvRosInterface.__new__(AgvRosInterface)
    ros.lock = threading.RLock()
    ros.sim_time = 1.0
    ros.cmd_vx_m_s = 0.0
    ros.cmd_wz_rad_s = 0.0
    message = Twist()
    message.linear.x = 0.71
    message.angular.z = -0.18
    ros._cmd_vel_callback(message)
    assert ros.cmd_vx_m_s == pytest.approx(0.71)
    assert ros.cmd_wz_rad_s == pytest.approx(-0.18)
    assert ros.cmd_vel_time_s == pytest.approx(1.0)


def test_observation_info_exposes_raw_cmd_separately_from_measured_velocity():
    state = {
        "lidar": np.full(24, 3.0, dtype=np.float32),
        "front_scan_x": np.asarray([1.4], dtype=np.float32),
        "front_scan_y": np.asarray([0.0], dtype=np.float32),
        "front_scan_valid": True,
        "front_scan_age_s": 0.0,
        "sector_ttc": np.full(24, 5.0, dtype=np.float32),
        "hazard_persistence": 0.0,
        "applied_speed_scale": 0.35,
        "safety_override_active": False,
        "front_safety_stop_active": True,
        "front_safety_stop_events": 1,
        "front_safety_stop_reason": 2,
        "front_safety_stop_clearance_m": 0.6,
        "front_safety_stop_distance_m": 0.9,
        "supervisor_min_ttc": 5.0,
        "min_sector_ttc_episode": 5.0,
        "dynamic_obstacle_spawns": 0,
        "dynamic_obstacle_active": False,
        "linear_velocity": 0.0,
        "angular_velocity": 0.0,
        "cmd_vx_m_s": 0.71,
        "cmd_wz_rad_s": -0.18,
        "cmd_vel_age_s": 0.05,
        "minimum_lidar": 3.0,
        "minimum_lidar_step": 3.0,
        "contact_count": 0,
        "collision": False,
        "collision_event_count": 0,
        "collision_events": [],
        "nav_status": None,
    }
    env = AgvRlEnv.__new__(AgvRlEnv)
    env.ros = SimpleNamespace(TTC_MAX=5.0, current_state=lambda: state)
    env.goal = (4.0, 0.0, 0.0)
    env.goals_reached_this_episode = 0
    env._pose_and_distance = lambda: (0.0, 0.0, 0.0, 4.0, 0.0)

    _, info = env._observation()

    assert info["cmd_vx_m_s"] == pytest.approx(0.71)
    assert info["cmd_wz_rad_s"] == pytest.approx(-0.18)
    assert info["cmd_vel_age_s"] == pytest.approx(0.05)
    assert info["linear_velocity"] == 0.0
    assert info["front_safety_stop_active"] is True
