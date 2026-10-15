"""Regression checks for false TTC alarms and approaching obstacles."""

import math
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_velocity_supervisor import RlVelocitySupervisor


class FakeSupervisor:
    TTC_SECTORS = RlVelocitySupervisor.TTC_SECTORS
    TTC_LOOKBACK_SECONDS = RlVelocitySupervisor.TTC_LOOKBACK_SECONDS
    TTC_HISTORY_SECONDS = RlVelocitySupervisor.TTC_HISTORY_SECONDS
    TTC_CONFIRM_SCANS = RlVelocitySupervisor.TTC_CONFIRM_SCANS
    TTC_MAX_CLOSING_SPEED = RlVelocitySupervisor.TTC_MAX_CLOSING_SPEED
    _sector_ranges = RlVelocitySupervisor._sector_ranges
    _align_previous_scan = staticmethod(RlVelocitySupervisor._align_previous_scan)

    def __init__(self):
        self.scan_history = deque(maxlen=16)
        self.last_scan_time = None
        self.ttc_candidate_scans = 0
        self.latest_yaw = 0.0
        self.min_closing_speed = 0.05
        self.emergency_ttc = 1.0
        self.safety_override_enabled = True
        self.ttc_observation_enabled = True
        self.emergency_distance = 0.35
        self.safety_hold_time = 0.75
        self.safety_active = False
        self.safety_release_after = 0.0
        self.safety_min_ttc = math.inf
        self.controller_verified = False
        self.applied_scale = 1.0
        self.emergency_scale = 0.15
        self.now = 0.0
        self.warnings = []

    def _control_time(self):
        return self.now

    def get_logger(self):
        return SimpleNamespace(warning=self.warnings.append)

    def publish_scale(self, value):
        self.applied_scale = value

    def feed(self, t, ranges, yaw=0.0):
        self.now = t
        self.latest_yaw = yaw
        scan = SimpleNamespace(
            ranges=ranges,
            angle_min=-math.pi,
            angle_increment=math.pi / 360,
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=int(t), nanosec=round((t % 1) * 1e9))
            ),
        )
        RlVelocitySupervisor.scan_callback(self, scan)


def make_scan(distance):
    scan = np.full(720, np.nan, dtype=np.float32)
    scan[240:481] = distance
    return scan


class TtcShieldTest(unittest.TestCase):
    def test_rotating_past_static_wall_does_not_trigger(self):
        supervisor = FakeSupervisor()
        angles = np.arange(720) * math.pi / 360 - math.pi
        for index in range(10):
            yaw = index * 0.05
            scan = make_scan(2.0)
            scan[240:481] = np.clip(
                1.1 + 3.0 * (angles[240:481] + yaw),
                0.5,
                5.0,
            )
            supervisor.feed(1.0 + index * 0.1, scan, yaw=yaw)
        self.assertFalse(supervisor.safety_active)

    def test_single_ray_jump_does_not_trigger(self):
        supervisor = FakeSupervisor()
        for index in range(10):
            scan = make_scan(2.0)
            if index >= 4:
                scan[360] = 0.8
            supervisor.feed(1.0 + index * 0.1, scan)
        self.assertFalse(supervisor.safety_active)
        self.assertEqual(supervisor.warnings, [])

    def test_persistent_wide_approach_triggers(self):
        supervisor = FakeSupervisor()
        for index in range(10):
            scan = make_scan(float("inf"))
            scan[350:371] = 1.2 - index * 0.1
            supervisor.feed(1.0 + index * 0.1, scan)
        self.assertTrue(supervisor.safety_active)
        self.assertEqual(len(supervisor.warnings), 1)
        self.assertLess(supervisor.safety_min_ttc, 1.0)

    def test_approaching_obstacle_while_turning_triggers(self):
        supervisor = FakeSupervisor()
        for index in range(10):
            yaw = index * 0.02
            scan = make_scan(float("inf"))
            center = round(360 - yaw / (math.pi / 360))
            scan[center - 10:center + 11] = 1.2 - index * 0.1
            supervisor.feed(1.0 + index * 0.1, scan, yaw=yaw)
        self.assertTrue(supervisor.safety_active)

    def test_close_obstacle_triggers_immediately(self):
        supervisor = FakeSupervisor()
        scan = make_scan(2.0)
        scan[360] = 0.3
        supervisor.feed(1.0, scan)
        self.assertTrue(supervisor.safety_active)

    def test_blind_scan_triggers_immediately(self):
        supervisor = FakeSupervisor()
        supervisor.feed(1.0, np.full(720, np.nan, dtype=np.float32))
        self.assertTrue(supervisor.safety_active)

    def test_observation_only_ttc_never_overrides_speed(self):
        supervisor = FakeSupervisor()
        supervisor.safety_override_enabled = False
        for index in range(10):
            scan = make_scan(float("inf"))
            scan[350:371] = 1.2 - index * 0.1
            supervisor.feed(1.0 + index * 0.1, scan)
        self.assertLess(supervisor.safety_min_ttc, 1.0)
        self.assertFalse(supervisor.safety_active)
        self.assertEqual(supervisor.applied_scale, 1.0)
        self.assertEqual(supervisor.warnings, [])

    def test_no_ttc_mode_does_not_estimate_or_override(self):
        supervisor = FakeSupervisor()
        supervisor.safety_override_enabled = False
        supervisor.ttc_observation_enabled = False
        supervisor.feed(1.0, make_scan(0.2))
        self.assertTrue(math.isinf(supervisor.safety_min_ttc))
        self.assertFalse(supervisor.safety_active)
        self.assertEqual(supervisor.applied_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
