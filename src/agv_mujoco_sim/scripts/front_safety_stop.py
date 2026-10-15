"""Simulation-only, LiDAR-based last-resort stop before wheel commands.

This gate never uses MuJoCo obstacle ground truth. It is separate from the
54D SAC observation/action contract and the Nav2 SpeedLimit supervisor.
"""

from __future__ import annotations

import math

import numpy as np


class FrontSafetyStop:
    REASON_SENSOR = 1
    REASON_CORRIDOR = 2
    REASON_EDGE_HOLD = 4
    REASON_INVALID_COMMAND = 8

    def __init__(
        self,
        *,
        corridor_half_width_m: float = 0.45,
        clearance_m: float = 0.40,
        reaction_time_s: float = 0.40,
        assumed_deceleration_m_s2: float = 0.80,
        release_margin_m: float = 0.15,
        scan_timeout_s: float = 0.25,
        edge_trigger_range_m: float = 1.30,
        edge_hold_s: float = 1.25,
    ) -> None:
        values = (
            corridor_half_width_m, clearance_m, reaction_time_s,
            assumed_deceleration_m_s2, release_margin_m, scan_timeout_s,
            edge_trigger_range_m, edge_hold_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("FrontSafetyStop parameters must be finite and positive")
        self.corridor_half_width_m = corridor_half_width_m
        self.clearance_m = clearance_m
        self.reaction_time_s = reaction_time_s
        self.assumed_deceleration_m_s2 = assumed_deceleration_m_s2
        self.release_margin_m = release_margin_m
        self.scan_timeout_s = scan_timeout_s
        self.edge_trigger_range_m = edge_trigger_range_m
        self.edge_hold_s = edge_hold_s
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.scan_time_s = None
        self.clearance_observed_m = math.inf
        self.scan_valid = False
        self.scan_x = np.empty(0, dtype=np.float64)
        self.scan_y = np.empty(0, dtype=np.float64)
        self.stop_distance_m = math.inf
        self.corridor_half_width_current_m = self.corridor_half_width_m
        self.edge_last_ranges_m = (math.inf, math.inf)
        self.edge_last_scan_time_s = None
        self.edge_closing_scans = [0, 0]
        self.edge_hold_until_s = -math.inf
        self.stop_reason = 0

    def update_scan(
        self,
        ranges,
        *,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        sim_time_s: float,
        ego_vx_m_s: float = 0.0,
        ego_wz_rad_s: float = 0.0,
    ) -> None:
        data = np.asarray(ranges, dtype=np.float64)
        angles = angle_min + np.arange(data.size) * angle_increment
        front = (angles >= -math.pi / 3.0) & (angles <= math.pi / 3.0)
        observed = front & (
            np.isposinf(data)
            | (np.isfinite(data) & (data >= range_min) & (data <= range_max))
        )
        self.scan_valid = bool(np.any(observed))
        finite = observed & np.isfinite(data)
        edge_ranges = []
        edge_angles = []
        for side in (-1, 1):
            edge = finite & (side * angles >= math.pi / 6.0)
            if np.any(edge):
                edge_indices = np.flatnonzero(edge)
                closest = edge_indices[int(np.argmin(data[edge_indices]))]
                edge_ranges.append(float(data[closest]))
                edge_angles.append(float(angles[closest]))
            else:
                edge_ranges.append(math.inf)
                edge_angles.append(0.0)
        scan_dt = (
            sim_time_s - self.edge_last_scan_time_s
            if self.edge_last_scan_time_s is not None else math.inf
        )
        for index, edge_range in enumerate(edge_ranges):
            old_range = self.edge_last_ranges_m[index]
            observed_closing_rate = (
                (old_range - edge_range) / scan_dt
                if 0.0 < scan_dt <= self.scan_timeout_s
                and math.isfinite(old_range)
                and math.isfinite(edge_range)
                else -math.inf
            )
            ego_closing_rate = max(0.0, ego_vx_m_s) * max(
                0.0, math.cos(edge_angles[index])
            )
            closing = (
                math.isfinite(ego_vx_m_s)
                and math.isfinite(ego_wz_rad_s)
                and abs(ego_wz_rad_s) <= 0.10
                and observed_closing_rate - ego_closing_rate >= 0.12
            )
            self.edge_closing_scans[index] = (
                self.edge_closing_scans[index] + 1 if closing else 0
            )
            if self.edge_closing_scans[index] >= 2 and edge_range <= self.edge_trigger_range_m:
                self.edge_hold_until_s = float(sim_time_s) + self.edge_hold_s
        self.edge_last_ranges_m = tuple(edge_ranges)
        self.edge_last_scan_time_s = float(sim_time_s)
        x = data[finite] * np.cos(angles[finite])
        y = data[finite] * np.sin(angles[finite])
        self.scan_x = x
        self.scan_y = y
        self.scan_time_s = float(sim_time_s)

    def limit_command(
        self,
        vx: float,
        wz: float,
        *,
        measured_vx: float,
        sim_time_s: float,
    ) -> tuple[float, float]:
        if not all(math.isfinite(value) for value in (vx, wz, measured_vx, sim_time_s)):
            self.active = True
            self.stop_reason = self.REASON_INVALID_COMMAND
            return 0.0, 0.0
        scan_stale = (
            self.scan_time_s is None
            or sim_time_s < self.scan_time_s
            or sim_time_s - self.scan_time_s > self.scan_timeout_s
        )
        if scan_stale or not self.scan_valid:
            self.active = True
            self.stop_reason = self.REASON_SENSOR
            return 0.0, 0.0
        edge_hold_active = sim_time_s < self.edge_hold_until_s
        if vx <= 0.0 and measured_vx <= 0.0 and not edge_hold_active:
            self.active = False
            self.stop_reason = 0
            return vx, wz
        speed = max(vx, measured_vx, 0.0)
        # A crossing obstacle can enter the vehicle's path during the time
        # needed to react and brake. Widen the monitored corridor with speed;
        # a fixed-width cone only saw the previous collision 0.1 s in advance.
        corridor_half_width = (
            self.corridor_half_width_m + self.reaction_time_s * speed
        )
        self.corridor_half_width_current_m = corridor_half_width
        corridor = (
            (self.scan_x > 0.0)
            & (np.abs(self.scan_y) <= corridor_half_width)
        )
        self.clearance_observed_m = (
            float(np.min(self.scan_x[corridor])) if np.any(corridor) else math.inf
        )
        stop_distance = (
            self.clearance_m
            + self.reaction_time_s * speed
            + speed * speed / (2.0 * self.assumed_deceleration_m_s2)
        )
        if self.active:
            stop_distance += self.release_margin_m
        self.stop_distance_m = stop_distance
        self.stop_reason = (
            (self.REASON_CORRIDOR if self.clearance_observed_m <= stop_distance else 0)
            | (self.REASON_EDGE_HOLD if edge_hold_active else 0)
        )
        self.active = bool(self.stop_reason)
        return (0.0, 0.0) if self.active else (vx, wz)
