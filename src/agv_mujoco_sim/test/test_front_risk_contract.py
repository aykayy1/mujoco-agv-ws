"""Regression checks for the proactive 56D front-risk contract."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import LaserScan

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from agv_rl_env import AgvRlEnv
from front_safety_stop import FrontSafetyStop
from train_sac import validate_model, validate_replay_buffer_shape


def risk_state(clearance=1.4, valid=True, applied_scale=1.0):
    return {
        "front_scan_valid": valid,
        "front_scan_age_s": 0.0,
        "front_scan_x": np.asarray([clearance], dtype=np.float32),
        "front_scan_y": np.asarray([0.0], dtype=np.float32),
        "linear_velocity": 0.7,
        "applied_speed_scale": applied_scale,
    }


def test_proactive_risk_appears_before_physical_gate_stop():
    clearance, margin, risk = AgvRlEnv._front_risk_features(risk_state())
    assert clearance == pytest.approx(1.4, abs=1e-6)
    assert 0.0 < margin < AgvRlEnv.FRONT_RISK_LOOKAHEAD_M
    assert 0.0 < risk < 1.0

    gate = FrontSafetyStop()
    scan = np.full(241, math.inf)
    scan[120] = 1.4
    gate.update_scan(
        scan, angle_min=-math.pi / 3, angle_increment=math.pi / 360,
        range_min=0.12, range_max=12.0, sim_time_s=1.0,
    )
    assert gate.limit_command(0.7, 0.0, measured_vx=0.7, sim_time_s=1.0) == (
        0.7, 0.0
    )
    assert not gate.active


def test_invalid_front_scan_has_finite_fail_closed_features():
    for state in (
        risk_state(valid=False),
        {**risk_state(), "front_scan_age_s": math.inf},
        {**risk_state(), "front_scan_age_s": 0.30},
        {**risk_state(), "front_scan_x": np.asarray([math.nan])},
        {**risk_state(), "front_scan_y": np.asarray([])},
    ):
        clearance, margin, risk = AgvRlEnv._front_risk_features(state)
        assert all(math.isfinite(value) for value in (clearance, margin, risk))
        assert clearance == 0.0
        assert margin < 0.0
        assert risk == 1.0


@pytest.mark.parametrize("distance", [0.12, 0.13, 0.15])
def test_raw_front_returns_near_range_min_are_not_erased(distance):
    import threading
    from agv_rl_env import AgvRosInterface

    ros = AgvRosInterface.__new__(AgvRosInterface)
    ros.lock = threading.RLock()
    ros.sim_time = 1.0
    ros.minimum_lidar_since_reset = 10.0
    ros.minimum_lidar_since_step = 10.0
    ros.scan_seq = 0
    ros.scan_ready = threading.Event()
    msg = LaserScan()
    msg.angle_min = -math.pi / 3
    msg.angle_increment = math.pi / 360
    msg.range_min = 0.12
    msg.range_max = 12.0
    msg.ranges = [math.inf] * 241
    msg.ranges[120] = distance

    ros._scan_callback(msg)

    assert ros.front_scan_valid
    assert min(ros.front_scan_x) == pytest.approx(distance, abs=1e-5)
    assert min(ros.lidar) == pytest.approx(distance, abs=1e-5)


def test_fresh_clear_scan_is_safe_but_stale_same_scan_fails_closed():
    fresh = risk_state(clearance=1.8)
    stale = {**fresh, "front_scan_age_s": 0.3}
    _, fresh_margin, fresh_risk = AgvRlEnv._front_risk_features(fresh)
    _, stale_margin, stale_risk = AgvRlEnv._front_risk_features(stale)
    assert fresh_margin > AgvRlEnv.FRONT_RISK_LOOKAHEAD_M
    assert fresh_risk == 0.0
    assert stale_margin < 0.0
    assert stale_risk == 1.0


def test_front_risk_drops_when_applied_envelope_is_lower():
    high = risk_state(clearance=0.9, applied_scale=1.0)
    low = risk_state(clearance=0.9, applied_scale=0.15)
    high["linear_velocity"] = 0.0
    low["linear_velocity"] = 0.0
    _, high_margin, high_risk = AgvRlEnv._front_risk_features(high)
    _, low_margin, low_risk = AgvRlEnv._front_risk_features(low)
    assert high_margin < low_margin
    assert high_risk > low_risk


def test_old_54d_checkpoint_and_replay_are_rejected():
    old_model = SimpleNamespace(
        observation_contract_version="speedlimit-v4-front120-54x1-1hz-ttc",
        observation_space=SimpleNamespace(shape=(54,)),
        action_space=SimpleNamespace(
            shape=(1,), low=AgvRlEnv.ACTION_LOW, high=AgvRlEnv.ACTION_HIGH
        ),
    )
    with pytest.raises(ValueError, match="observation contract|observation"):
        validate_model(old_model)

    old_model.replay_buffer = SimpleNamespace(
        observation_contract_version="speedlimit-v4-front120-54x1-1hz-ttc",
        reward_contract_version="front-corridor-request-penalty-v1",
        observations=np.zeros((1, 1, 54), dtype=np.float32),
        actions=np.zeros((1, 1, 1), dtype=np.float32),
    )
    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(AgvRlEnv.OBSERVATION_DIM,)),
        action_space=SimpleNamespace(shape=(1,)),
    )
    with pytest.raises(ValueError, match="Replay buffer contract"):
        validate_replay_buffer_shape(old_model, env)


def test_sac_only_checkpoint_has_separate_31d_contract():
    action_space = SimpleNamespace(
        shape=(1,), low=AgvRlEnv.ACTION_LOW, high=AgvRlEnv.ACTION_HIGH,
        contains=lambda action: bool(
            np.all(action >= AgvRlEnv.ACTION_LOW)
            and np.all(action <= AgvRlEnv.ACTION_HIGH)
        ),
    )
    model = SimpleNamespace(
        observation_contract_version=AgvRlEnv.CONTRACT_VERSION_NO_TTC,
        observation_space=SimpleNamespace(shape=(AgvRlEnv.OBSERVATION_DIM_NO_TTC,)),
        action_space=action_space,
        predict=lambda _obs, deterministic: (np.asarray([0.5], dtype=np.float32), None),
        num_timesteps=100,
    )
    contract = validate_model(model, sac_only=True)
    assert contract["observation_shape"] == [31]
    with pytest.raises(ValueError, match="observation contract|observation"):
        validate_model(model)


def test_observation_appends_risk_features_after_unchanged_54d_prefix():
    state = {
        **risk_state(clearance=1.4, applied_scale=0.35),
        "lidar": np.full(24, 3.0, dtype=np.float32),
        "sector_ttc": np.full(24, 5.0, dtype=np.float32),
        "hazard_persistence": 0.0,
        "angular_velocity": 0.0,
        "safety_override_active": False,
        "front_safety_stop_active": False,
        "front_safety_stop_events": 0,
        "front_safety_stop_reason": 0,
        "front_safety_stop_clearance_m": 1.4,
        "front_safety_stop_distance_m": 0.9,
        "supervisor_min_ttc": 5.0,
        "min_sector_ttc_episode": 5.0,
        "dynamic_obstacle_spawns": 0,
        "dynamic_obstacle_active": False,
        "cmd_vx_m_s": 0.7,
        "cmd_wz_rad_s": 0.0,
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

    observation, info = env._observation()

    assert observation.shape == (56,)
    np.testing.assert_allclose(observation[:24], 0.3)
    np.testing.assert_allclose(observation[24:48], 1.0)
    np.testing.assert_allclose(
        observation[48:54],
        [0.2, 0.0, 0.7 / env.MAX_EXPECTED_LINEAR_VELOCITY, 0.0, 0.35, 0.0],
    )
    assert observation[54] == pytest.approx(info["front_corridor_clearance_m"] / 10.0)
    assert observation[55] == pytest.approx(info["front_braking_margin_m"] / 2.0)
    assert np.all(np.isfinite(observation))

    env.include_ttc = False
    no_ttc_observation, _ = env._observation()
    assert no_ttc_observation.shape == (AgvRlEnv.OBSERVATION_DIM_NO_TTC,)
    np.testing.assert_allclose(no_ttc_observation[:24], observation[:24])
    np.testing.assert_allclose(no_ttc_observation[24:29], observation[48:53])
    np.testing.assert_allclose(no_ttc_observation[29:], observation[54:])
