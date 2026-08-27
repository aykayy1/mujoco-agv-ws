#!/usr/bin/env python3

"""Episode-level randomization at the ROS/behavior interface.

SAC is a bounded supervisor for Nav2 MPPI, not a low-level robot controller.
The randomizer therefore perturbs only effects visible at that interface:
LiDAR quality, wheel-odometry scale, command latency, and command response.
All groups are opt-in and the shipped configuration keeps the nominal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ParameterGroup:
    """Configuration and nominal value for one behavioral perturbation."""

    name: str
    enabled: bool
    low: float
    high: float
    nominal: float
    status: str
    source: str


class DomainRandomizer:
    """Sample deterministic per-episode behavioral perturbations."""

    GROUP_SPECS: Dict[str, Tuple[float, float, float]] = {
        # name: (nominal, absolute minimum, absolute maximum)
        "lidar_noise_std_m": (0.0, 0.0, 0.50),
        "lidar_dropout_probability": (0.0, 0.0, 1.0),
        "odom_linear_scale": (1.0, 0.50, 1.50),
        "odom_angular_scale": (1.0, 0.50, 1.50),
        "command_latency_s": (0.0, 0.0, 0.50),
        "velocity_response_scale": (1.0, 0.50, 1.20),
    }
    GROUP_NAMES: Tuple[str, ...] = tuple(GROUP_SPECS)

    def __init__(
        self,
        *,
        enabled: bool,
        seed: int,
        groups: Dict[str, ParameterGroup],
    ) -> None:
        self.enabled = bool(enabled)
        self.seed = int(seed)
        self.groups = groups
        self.sensor_rng = np.random.default_rng(self.seed)
        self.last_state: Dict[str, Any] = {}

        self.lidar_noise_std_m = 0.0
        self.lidar_dropout_probability = 0.0
        self.odom_linear_scale = 1.0
        self.odom_angular_scale = 1.0
        self.command_latency_s = 0.0
        self.velocity_response_scale = 1.0

        self._validate_configuration()

    @classmethod
    def declare_ros_parameters(cls, node: Any) -> None:
        node.declare_parameter("domain_randomization.enabled", False)
        node.declare_parameter("domain_randomization.seed", 42)

        for group_name, (nominal, _minimum, _maximum) in cls.GROUP_SPECS.items():
            prefix = f"domain_randomization.{group_name}"
            node.declare_parameter(f"{prefix}.enabled", False)
            node.declare_parameter(f"{prefix}.low", nominal)
            node.declare_parameter(f"{prefix}.high", nominal)
            node.declare_parameter(f"{prefix}.status", "disabled")
            node.declare_parameter(f"{prefix}.source", "nominal")

    @classmethod
    def from_ros_node(cls, node: Any) -> "DomainRandomizer":
        groups: Dict[str, ParameterGroup] = {}
        for group_name, (nominal, _minimum, _maximum) in cls.GROUP_SPECS.items():
            prefix = f"domain_randomization.{group_name}"
            groups[group_name] = ParameterGroup(
                name=group_name,
                enabled=bool(node.get_parameter(f"{prefix}.enabled").value),
                low=float(node.get_parameter(f"{prefix}.low").value),
                high=float(node.get_parameter(f"{prefix}.high").value),
                nominal=nominal,
                status=str(node.get_parameter(f"{prefix}.status").value),
                source=str(node.get_parameter(f"{prefix}.source").value),
            )

        return cls(
            enabled=bool(
                node.get_parameter("domain_randomization.enabled").value
            ),
            seed=int(node.get_parameter("domain_randomization.seed").value),
            groups=groups,
        )

    def _validate_configuration(self) -> None:
        if self.seed < 0:
            raise ValueError("domain_randomization.seed must be non-negative")

        missing = set(self.GROUP_NAMES) - set(self.groups)
        extra = set(self.groups) - set(self.GROUP_NAMES)
        if missing or extra:
            raise ValueError(
                f"Invalid domain-randomization groups: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        for name, group in self.groups.items():
            _nominal, absolute_minimum, absolute_maximum = self.GROUP_SPECS[name]
            if not np.isfinite(group.low) or not np.isfinite(group.high):
                raise ValueError(f"Randomization range for {name} must be finite")
            if group.high < group.low:
                raise ValueError(
                    f"Invalid range for {name}: low={group.low}, high={group.high}"
                )
            if group.low < absolute_minimum or group.high > absolute_maximum:
                raise ValueError(
                    f"Range for {name} must stay within "
                    f"[{absolute_minimum}, {absolute_maximum}]"
                )
            if group.enabled and group.status not in {"provisional", "calibrated"}:
                raise ValueError(
                    f"Refusing enabled group {name!r} with status={group.status!r}"
                )

        if self.enabled and not any(group.enabled for group in self.groups.values()):
            raise ValueError(
                "domain_randomization.enabled=true but no group is enabled"
            )

    def _sample_groups(self, generation: int) -> Dict[str, float]:
        # Episode values depend only on seed + generation, not on how many
        # sensor samples the previous episode produced.
        episode_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(generation), 0])
        )
        values: Dict[str, float] = {}
        for name, group in self.groups.items():
            if self.enabled and group.enabled:
                values[name] = float(
                    episode_rng.uniform(group.low, group.high)
                )
            else:
                values[name] = group.nominal
        return values

    def reset_episode(self, *, generation: int, sim_time: float) -> Dict[str, Any]:
        """Sample and apply one deterministic episode configuration."""
        values = self._sample_groups(generation)
        self.sensor_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(generation), 1])
        )
        for name, value in values.items():
            setattr(self, name, value)

        groups_state = {
            name: {
                "enabled": group.enabled,
                "low": group.low,
                "high": group.high,
                "nominal": group.nominal,
                "status": group.status,
                "source": group.source,
                "applied_value": values[name],
            }
            for name, group in self.groups.items()
        }
        behavior_randomized = any(
            self.enabled
            and group.enabled
            and not np.isclose(values[name], group.nominal)
            for name, group in self.groups.items()
        )

        state: Dict[str, Any] = {
            "config_phase": "issue03_behavioral_randomization",
            "enabled": self.enabled,
            "seed": self.seed,
            "generation": int(generation),
            "sim_time": float(sim_time),
            "behavior_randomized": behavior_randomized,
            # Kept for compatibility with Issue 03A diagnostics.
            "physics_randomized": False,
            "groups": groups_state,
        }
        self.last_state = state
        return state

    def initial_state(self, *, sim_time: float = 0.0) -> Dict[str, Any]:
        return self.reset_episode(generation=0, sim_time=sim_time)

    def apply_command(self, vx: float, wz: float) -> Tuple[float, float]:
        scale = self.velocity_response_scale
        return float(vx) * scale, float(wz) * scale

    def apply_lidar(
        self,
        ranges: np.ndarray,
        *,
        range_min: float,
        range_max: float,
    ) -> np.ndarray:
        """Apply noise/dropout only to finite LiDAR hits."""
        randomized = np.asarray(ranges, dtype=np.float64).copy()
        valid = np.isfinite(randomized)

        if self.lidar_noise_std_m > 0.0 and np.any(valid):
            randomized[valid] += self.sensor_rng.normal(
                0.0,
                self.lidar_noise_std_m,
                size=int(np.count_nonzero(valid)),
            )
            randomized[valid] = np.clip(
                randomized[valid],
                float(range_min),
                float(range_max),
            )

        if self.lidar_dropout_probability > 0.0 and np.any(valid):
            dropout = (
                self.sensor_rng.random(randomized.shape)
                < self.lidar_dropout_probability
            )
            randomized[valid & dropout] = np.inf

        return randomized
