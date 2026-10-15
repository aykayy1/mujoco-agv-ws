"""Regression tests for the simulator's final wheel-command stop gate."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from front_safety_stop import FrontSafetyStop


def scan_with_hit(distance=math.inf, angle=0.0):
    scan = np.full(241, math.inf)
    index = round((angle + math.pi / 3) / (math.pi / 360))
    scan[index] = distance
    return scan


def update(gate, scan, t=1.0, ego_vx=0.0, ego_wz=0.0):
    gate.update_scan(
        scan,
        angle_min=-math.pi / 3,
        angle_increment=math.pi / 360,
        range_min=0.12,
        range_max=12.0,
        sim_time_s=t,
        ego_vx_m_s=ego_vx,
        ego_wz_rad_s=ego_wz,
    )


class FrontSafetyStopTest(unittest.TestCase):
    def test_stale_or_blind_scan_fails_closed(self):
        gate = FrontSafetyStop()
        self.assertEqual(gate.limit_command(0.5, 0.2, measured_vx=0.5, sim_time_s=1.0), (0.0, 0.0))
        update(gate, np.full(241, math.nan))
        self.assertEqual(gate.limit_command(0.5, 0.2, measured_vx=0.5, sim_time_s=1.1), (0.0, 0.0))
        update(gate, scan_with_hit(), t=2.0)
        self.assertEqual(gate.limit_command(0.5, 0.2, measured_vx=0.5, sim_time_s=2.3), (0.0, 0.0))

    def test_approaching_front_obstacle_stops_before_contact(self):
        gate = FrontSafetyStop()
        update(gate, scan_with_hit(0.75))
        self.assertEqual(gate.limit_command(0.55, 0.2, measured_vx=0.55, sim_time_s=1.0), (0.0, 0.0))
        self.assertTrue(gate.active)

    def test_side_obstacle_outside_swept_corridor_does_not_stop(self):
        gate = FrontSafetyStop()
        update(gate, scan_with_hit(0.8, angle=math.pi / 3))
        self.assertEqual(gate.limit_command(0.3, 0.1, measured_vx=0.3, sim_time_s=1.0), (0.3, 0.1))

    def test_fast_crossing_obstacle_enters_speed_expanded_corridor(self):
        gate = FrontSafetyStop()
        # Surface point is laterally outside the robot's fixed 0.45 m strip,
        # but crosses into its path while the robot is braking from 0.67 m/s.
        update(gate, scan_with_hit(0.94, angle=math.radians(42)))
        self.assertEqual(gate.limit_command(0.67, 0.0, measured_vx=0.67, sim_time_s=1.0), (0.0, 0.0))

    def test_approaching_edge_obstacle_is_held_after_leaving_field_of_view(self):
        gate = FrontSafetyStop()
        for t, distance in ((1.0, 1.55), (1.1, 1.42), (1.2, 1.28)):
            update(gate, scan_with_hit(distance, angle=math.radians(55)), t=t)
        self.assertEqual(gate.limit_command(0.65, 0.0, measured_vx=0.65, sim_time_s=1.2), (0.0, 0.0))
        update(gate, scan_with_hit(), t=1.3)
        self.assertEqual(gate.limit_command(0.65, 0.0, measured_vx=0.4, sim_time_s=1.3), (0.0, 0.0))
        update(gate, scan_with_hit(), t=2.5)
        self.assertEqual(gate.limit_command(0.65, 0.0, measured_vx=0.0, sim_time_s=2.5), (0.65, 0.0))

    def test_static_edge_obstacle_does_not_start_hold(self):
        gate = FrontSafetyStop()
        for t in (1.0, 1.1, 1.2):
            update(gate, scan_with_hit(1.1, angle=math.radians(55)), t=t)
        self.assertEqual(gate.limit_command(0.65, 0.0, measured_vx=0.65, sim_time_s=1.2), (0.65, 0.0))

    def test_ego_motion_does_not_mark_static_edge_wall_as_moving(self):
        gate = FrontSafetyStop()
        for t, distance in ((1.0, 1.25), (1.1, 1.213), (1.2, 1.176)):
            update(
                gate, scan_with_hit(distance, angle=math.radians(55)),
                t=t, ego_vx=0.65,
            )
        self.assertEqual(gate.limit_command(0.65, 0.0, measured_vx=0.65, sim_time_s=1.2), (0.65, 0.0))
        self.assertEqual(gate.stop_reason, 0)

    def test_extra_edge_closing_beyond_ego_motion_starts_hold(self):
        gate = FrontSafetyStop()
        for t, distance in ((1.0, 1.40), (1.1, 1.34), (1.2, 1.28)):
            update(
                gate, scan_with_hit(distance, angle=math.radians(55)),
                t=t, ego_vx=0.65,
            )
        self.assertEqual(gate.limit_command(0.65, 0.0, measured_vx=0.65, sim_time_s=1.2), (0.0, 0.0))
        self.assertEqual(gate.stop_reason, gate.REASON_EDGE_HOLD)

    def test_release_has_distance_hysteresis(self):
        gate = FrontSafetyStop()
        update(gate, scan_with_hit(0.7))
        self.assertEqual(gate.limit_command(0.5, 0.0, measured_vx=0.5, sim_time_s=1.0), (0.0, 0.0))
        update(gate, scan_with_hit(0.8), t=1.1)
        self.assertEqual(gate.limit_command(0.5, 0.0, measured_vx=0.5, sim_time_s=1.1), (0.0, 0.0))
        update(gate, scan_with_hit(1.0), t=1.2)
        self.assertEqual(gate.limit_command(0.5, 0.0, measured_vx=0.5, sim_time_s=1.2), (0.5, 0.0))

    def test_clear_scan_allows_motion(self):
        gate = FrontSafetyStop()
        update(gate, scan_with_hit())
        self.assertEqual(gate.limit_command(0.6, 0.2, measured_vx=0.6, sim_time_s=1.0), (0.6, 0.2))


if __name__ == "__main__":
    unittest.main()
