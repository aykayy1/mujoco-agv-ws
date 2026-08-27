#!/usr/bin/env python3

"""Verify nominal Issue 03 reset behavior and diagnostic state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Issue03SamplingCheck(Node):
    def __init__(self) -> None:
        super().__init__("issue03_sampling_check")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.latest_state: Optional[Dict[str, Any]] = None
        self.create_subscription(
            String,
            "/domain_randomization/state",
            self._state_callback,
            qos,
        )
        self.reset_client = self.create_client(Trigger, "/reset_simulation")

    def _state_callback(self, message: String) -> None:
        self.latest_state = json.loads(message.data)

    def wait_for_state(self, timeout_s: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_state is not None:
                return self.latest_state
        raise TimeoutError("No /domain_randomization/state message received")

    def reset_once(self, timeout_s: float) -> Dict[str, Any]:
        if not self.reset_client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError("/reset_simulation is unavailable")

        previous_generation = int((self.latest_state or {}).get("generation", -1))
        future = self.reset_client.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout_s

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if future.done():
                break
        if not future.done():
            raise TimeoutError("Reset service timed out")

        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                "Reset failed: " + (response.message if response else "no response")
            )

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_state is not None:
                generation = int(self.latest_state.get("generation", -1))
                if generation > previous_generation:
                    return self.latest_state
        raise TimeoutError("No new domain-randomization generation received")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resets", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rclpy.init()
    node = Issue03SamplingCheck()
    states = []

    try:
        states.append(node.wait_for_state(args.timeout))
        for _ in range(args.resets):
            states.append(node.reset_once(args.timeout))

        checks = {
            "all_physics_randomized_false": all(
                not bool(state.get("physics_randomized", True)) for state in states
            ),
            "all_behavior_randomized_false": all(
                not bool(state.get("behavior_randomized", True)) for state in states
            ),
            "all_global_enabled_false": all(
                not bool(state.get("enabled", True)) for state in states
            ),
            "generations_strictly_increasing": all(
                int(current["generation"]) > int(previous["generation"])
                for previous, current in zip(states, states[1:])
            ),
            "all_applied_values_nominal": all(
                float(group["applied_value"]) == float(group["nominal"])
                for state in states
                for group in state["groups"].values()
            ),
        }
        result = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "states": states,
        }

        print(json.dumps(result, indent=2, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return 0 if result["all_checks_passed"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
