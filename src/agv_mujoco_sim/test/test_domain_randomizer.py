"""Unit tests for Issue 03 behavioral randomization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from domain_randomizer import DomainRandomizer, ParameterGroup  # noqa: E402


def make_groups(*, enabled=(), status="provisional"):
    enabled_names = set(enabled)
    ranges = {
        "lidar_noise_std_m": (0.0, 0.03),
        "lidar_dropout_probability": (0.0, 0.03),
        "odom_linear_scale": (0.98, 1.02),
        "odom_angular_scale": (0.98, 1.02),
        "command_latency_s": (0.02, 0.15),
        "velocity_response_scale": (0.80, 1.00),
    }
    return {
        name: ParameterGroup(
            name=name,
            enabled=name in enabled_names,
            low=ranges[name][0],
            high=ranges[name][1],
            nominal=DomainRandomizer.GROUP_SPECS[name][0],
            status=status,
            source="unit_test",
        )
        for name in DomainRandomizer.GROUP_NAMES
    }


class DomainRandomizerTest(unittest.TestCase):
    def test_shipped_yaml_matches_group_schema_and_stays_disabled(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / (
            "domain_randomization.yaml"
        )
        parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
            "mujoco_ros_bridge"
        ]["ros__parameters"]

        self.assertFalse(parameters["domain_randomization.enabled"])
        configured_groups = {
            key.split(".")[1]
            for key in parameters
            if key.startswith("domain_randomization.")
            and key.count(".") == 2
        }
        self.assertEqual(configured_groups, set(DomainRandomizer.GROUP_NAMES))

    def test_disabled_configuration_is_exactly_nominal(self):
        randomizer = DomainRandomizer(
            enabled=False,
            seed=42,
            groups=make_groups(),
        )
        state = randomizer.initial_state()

        self.assertFalse(state["behavior_randomized"])
        self.assertFalse(state["physics_randomized"])
        for group in state["groups"].values():
            self.assertEqual(group["applied_value"], group["nominal"])

    def test_sampling_is_reproducible_and_bounded(self):
        enabled = {
            "lidar_noise_std_m",
            "odom_linear_scale",
            "command_latency_s",
        }
        first = DomainRandomizer(
            enabled=True,
            seed=7,
            groups=make_groups(enabled=enabled),
        )
        second = DomainRandomizer(
            enabled=True,
            seed=7,
            groups=make_groups(enabled=enabled),
        )

        first_state = first.reset_episode(generation=1, sim_time=1.0)
        first.apply_lidar(
            np.asarray([1.0, 2.0, 3.0]),
            range_min=0.12,
            range_max=12.0,
        )
        second_state = second.reset_episode(generation=1, sim_time=1.0)
        self.assertEqual(first_state, second_state)
        self.assertTrue(first_state["behavior_randomized"])

        # Parameter sampling is independent from sensor RNG consumption.
        self.assertEqual(
            first.reset_episode(generation=2, sim_time=2.0),
            second.reset_episode(generation=2, sim_time=2.0),
        )

        for name in enabled:
            group = first_state["groups"][name]
            self.assertGreaterEqual(group["applied_value"], group["low"])
            self.assertLessEqual(group["applied_value"], group["high"])

    def test_lidar_perturbation_preserves_missing_returns(self):
        randomizer = DomainRandomizer(
            enabled=True,
            seed=1,
            groups=make_groups(
                enabled={
                    "lidar_noise_std_m",
                    "lidar_dropout_probability",
                }
            ),
        )
        randomizer.reset_episode(generation=1, sim_time=0.0)
        ranges = np.asarray([1.0, 2.0, np.inf, 11.9], dtype=np.float64)
        result = randomizer.apply_lidar(
            ranges,
            range_min=0.12,
            range_max=12.0,
        )

        self.assertTrue(np.isinf(result[2]))
        finite = result[np.isfinite(result)]
        self.assertTrue(np.all(finite >= 0.12))
        self.assertTrue(np.all(finite <= 12.0))

    def test_enabled_group_requires_explicit_status(self):
        with self.assertRaisesRegex(ValueError, "status"):
            DomainRandomizer(
                enabled=True,
                seed=1,
                groups=make_groups(
                    enabled={"command_latency_s"},
                    status="disabled",
                ),
            )


if __name__ == "__main__":
    unittest.main()
