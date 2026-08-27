#!/usr/bin/env python3

"""Measure simulation speed, topic rates, and timestamp monotonicity.

Run this node while ``mujoco_sim.launch.py`` is active.  The resulting JSON is
suitable for comparing GUI 1x, headless 1x, accelerated, and uncapped runs.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_srvs.srv import Trigger


EXPECTED_SIM_RATES = {
    "/clock": 100.0,
    "/joint_states": 20.0,
    "/wheel/odom": 20.0,
    "/imu": 50.0,
    "/scan": 10.0,
    "/scan_raw": 10.0,
}


def ros_time_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


@dataclass
class TopicStats:
    count: int = 0
    first_stamp: Optional[float] = None
    last_stamp: Optional[float] = None
    first_wall: Optional[float] = None
    last_wall: Optional[float] = None
    backward_jumps: int = 0
    largest_backward_jump_s: float = 0.0

    def record(self, stamp: float, wall_time: float) -> None:
        if self.last_stamp is not None and stamp < self.last_stamp:
            jump = self.last_stamp - stamp
            self.backward_jumps += 1
            self.largest_backward_jump_s = max(
                self.largest_backward_jump_s,
                jump,
            )

        if self.first_stamp is None:
            self.first_stamp = stamp
            self.first_wall = wall_time

        self.last_stamp = stamp
        self.last_wall = wall_time
        self.count += 1

    def summarize(self, expected_rate_hz: float) -> Dict[str, Any]:
        stamp_elapsed = 0.0
        wall_elapsed = 0.0

        if self.first_stamp is not None and self.last_stamp is not None:
            stamp_elapsed = self.last_stamp - self.first_stamp
        if self.first_wall is not None and self.last_wall is not None:
            wall_elapsed = self.last_wall - self.first_wall

        samples_minus_one = max(0, self.count - 1)
        sim_rate = (
            samples_minus_one / stamp_elapsed
            if stamp_elapsed > 0.0
            else None
        )
        wall_rate = (
            samples_minus_one / wall_elapsed
            if wall_elapsed > 0.0
            else None
        )
        error_percent = (
            100.0 * (sim_rate - expected_rate_hz) / expected_rate_hz
            if sim_rate is not None and expected_rate_hz > 0.0
            else None
        )

        result = asdict(self)
        result.update(
            {
                "expected_rate_hz_sim": expected_rate_hz,
                "stamp_elapsed_s": stamp_elapsed,
                "arrival_wall_elapsed_s": wall_elapsed,
                "observed_rate_hz_sim": sim_rate,
                "observed_rate_hz_wall": wall_rate,
                "rate_error_percent": error_percent,
            }
        )
        return result


class Issue02Measurement(Node):

    def __init__(self) -> None:
        super().__init__("issue02_measurement")
        self.stats = {
            topic: TopicStats()
            for topic in EXPECTED_SIM_RATES
        }

        clock_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            Clock,
            "/clock",
            self._clock_callback,
            clock_qos,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            lambda msg: self._header_callback("/joint_states", msg),
            10,
        )
        self.create_subscription(
            Odometry,
            "/wheel/odom",
            lambda msg: self._header_callback("/wheel/odom", msg),
            10,
        )
        self.create_subscription(
            Imu,
            "/imu",
            lambda msg: self._header_callback("/imu", msg),
            10,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            lambda msg: self._header_callback("/scan", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan,
            "/scan_raw",
            lambda msg: self._header_callback("/scan_raw", msg),
            qos_profile_sensor_data,
        )

        self.reset_client = self.create_client(
            Trigger,
            "/reset_simulation",
        )
        self.reset_future = None
        self.reset_requested_wall: Optional[float] = None

    def _clock_callback(self, msg: Clock) -> None:
        self.stats["/clock"].record(
            ros_time_to_seconds(msg.clock),
            time.perf_counter(),
        )

    def _header_callback(self, topic: str, msg: Any) -> None:
        self.stats[topic].record(
            ros_time_to_seconds(msg.header.stamp),
            time.perf_counter(),
        )

    def request_reset(self) -> bool:
        if not self.reset_client.service_is_ready():
            return False
        self.reset_requested_wall = time.perf_counter()
        self.reset_future = self.reset_client.call_async(Trigger.Request())
        return True

    def reset_result(self) -> Optional[Dict[str, Any]]:
        if self.reset_future is None:
            return None
        if not self.reset_future.done():
            return {"completed": False}

        try:
            response = self.reset_future.result()
            return {
                "completed": True,
                "success": bool(response.success),
                "message": str(response.message),
            }
        except Exception as exc:  # pragma: no cover - runtime ROS failure
            return {
                "completed": True,
                "success": False,
                "message": str(exc),
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-wall", type=float, default=15.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("issue02_measurement.json"),
    )
    parser.add_argument("--label", default="issue02")
    parser.add_argument(
        "--reset-at-wall",
        type=float,
        default=-1.0,
        help=(
            "Request /reset_simulation after this many wall seconds; "
            "negative disables it"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.duration_wall) or args.duration_wall <= 0.0:
        raise SystemExit("--duration-wall must be finite and > 0")

    rclpy.init(args=[])
    node = Issue02Measurement()
    wall_start = time.perf_counter()
    reset_attempted = False

    try:
        while rclpy.ok():
            elapsed = time.perf_counter() - wall_start
            if elapsed >= args.duration_wall:
                break

            if (
                not reset_attempted
                and args.reset_at_wall >= 0.0
                and elapsed >= args.reset_at_wall
            ):
                reset_attempted = True
                if not node.request_reset():
                    node.get_logger().warn(
                        "/reset_simulation is not ready; reset test skipped"
                    )

            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        wall_end = time.perf_counter()

    clock_stats = node.stats["/clock"]
    clock_sim_elapsed = 0.0
    clock_wall_elapsed = 0.0
    if (
        clock_stats.first_stamp is not None
        and clock_stats.last_stamp is not None
    ):
        clock_sim_elapsed = clock_stats.last_stamp - clock_stats.first_stamp
    if (
        clock_stats.first_wall is not None
        and clock_stats.last_wall is not None
    ):
        clock_wall_elapsed = clock_stats.last_wall - clock_stats.first_wall

    achieved_rtf = (
        clock_sim_elapsed / clock_wall_elapsed
        if clock_wall_elapsed > 0.0
        else None
    )

    report = {
        "label": args.label,
        "measurement_wall_duration_s": wall_end - wall_start,
        "clock_sim_elapsed_s": clock_sim_elapsed,
        "clock_arrival_wall_elapsed_s": clock_wall_elapsed,
        "achieved_real_time_factor": achieved_rtf,
        "reset": node.reset_result(),
        "topics": {
            topic: node.stats[topic].summarize(expected)
            for topic, expected in EXPECTED_SIM_RATES.items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved: {args.output.resolve()}")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
