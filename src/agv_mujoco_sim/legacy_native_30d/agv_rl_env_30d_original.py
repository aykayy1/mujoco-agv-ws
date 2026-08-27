#!/usr/bin/env python3

"""Gymnasium environment for RL-assisted Nav2 MPPI on the MuJoCo AGV.

The policy does not publish cmd_vel. Its one normalized action is mapped to a
common MPPI speed scale and published on /rl_velocity_limits:

    action[0] -> speed_scale in [0.40, 1.00]

With MPPI base constraints vx_max=0.40 m/s and wz_max=0.50 rad/s, this gives:

    effective vx_max in [0.16, 0.40] m/s
    effective wz_max in [0.20, 0.50] rad/s

Run this file directly for a short random-action smoke test before training.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from robot_localization.srv import SetPose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray, Int32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_from_yaw(yaw: float) -> Tuple[float, float]:
    return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def yaw_from_quaternion(z: float, w: float) -> float:
    return 2.0 * math.atan2(z, w)


class AgvRosInterface(Node):
    """Asynchronous ROS interface used by the synchronous Gym environment."""

    LIDAR_BEAMS = 24
    LIDAR_FOV_MIN = -math.pi / 3.0
    LIDAR_FOV_MAX = math.pi / 3.0

    def __init__(self) -> None:
        super().__init__(
            "agv_rl_environment",
            parameter_overrides=[
                Parameter("use_sim_time", value=True),
            ],
        )

        self.lock = threading.RLock()
        self.scan_ready = threading.Event()
        self.odom_ready = threading.Event()

        self.scan_seq = 0
        self.odom_seq = 0
        self.lidar_ranges = np.full(
            self.LIDAR_BEAMS,
            12.0,
            dtype=np.float32,
        )
        self.lidar_range_max = 12.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.collision_since_reset = False
        self.max_contact_count = 0
        self.applied_speed_scale = 1.0
        self.applied_vx_limit = 0.40
        self.applied_wz_limit = 0.50

        self.goal_handle = None
        self.goal_result_status: Optional[int] = None
        self.goal_result_future = None
        self.goal_token = 0

        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        initial_pose_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self._scan_callback,
            scan_qos,
        )
        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self._odom_callback,
            20,
        )
        self.create_subscription(
            Bool,
            "/collision_state",
            self._collision_callback,
            20,
        )
        self.create_subscription(
            Int32,
            "/contact_count",
            self._contact_callback,
            20,
        )

        self.rl_limits_publisher = self.create_publisher(
            Float32MultiArray,
            "/rl_velocity_limits",
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/rl_velocity_limits_applied",
            self._applied_limits_callback,
            10,
        )
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            initial_pose_qos,
        )

        self.reset_simulation_client = self.create_client(
            Trigger,
            "/reset_simulation",
        )
        self.set_ekf_pose_client = self.create_client(
            SetPose,
            "/set_pose",
        )
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
        )
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
        )
        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

    def _scan_callback(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return

        angles = (
            float(msg.angle_min)
            + np.arange(ranges.size, dtype=np.float32)
            * float(msg.angle_increment)
        )
        fov_mask = (
            (angles >= self.LIDAR_FOV_MIN - 1.0e-5)
            & (angles <= self.LIDAR_FOV_MAX + 1.0e-5)
        )
        fov_ranges = ranges[fov_mask]
        if fov_ranges.size == 0:
            return

        range_min = max(float(msg.range_min), 0.0)
        range_max = max(float(msg.range_max), range_min + 1.0e-3)
        reduced = np.empty(self.LIDAR_BEAMS, dtype=np.float32)

        # Minimum pooling preserves small obstacles better than sampling one ray.
        for index, sector in enumerate(
            np.array_split(fov_ranges, self.LIDAR_BEAMS)
        ):
            valid = sector[
                np.isfinite(sector)
                & (sector >= range_min)
                & (sector <= range_max)
            ]
            reduced[index] = (
                float(np.min(valid))
                if valid.size > 0
                else range_max
            )

        with self.lock:
            self.lidar_ranges = reduced
            self.lidar_range_max = range_max
            self.scan_seq += 1
            self.scan_ready.set()

    def _odom_callback(self, msg: Odometry) -> None:
        with self.lock:
            self.linear_velocity = float(msg.twist.twist.linear.x)
            self.angular_velocity = float(msg.twist.twist.angular.z)
            self.odom_seq += 1
            self.odom_ready.set()

    def _collision_callback(self, msg: Bool) -> None:
        if msg.data:
            with self.lock:
                self.collision_since_reset = True

    def _contact_callback(self, msg: Int32) -> None:
        with self.lock:
            self.max_contact_count = max(
                self.max_contact_count,
                int(msg.data),
            )

    def _applied_limits_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) != 3:
            return
        scale, vx_limit, wz_limit = (float(value) for value in msg.data)
        if not all(
            math.isfinite(value)
            for value in (scale, vx_limit, wz_limit)
        ):
            return
        with self.lock:
            self.applied_speed_scale = scale
            self.applied_vx_limit = vx_limit
            self.applied_wz_limit = wz_limit

    def reset_episode_flags(self) -> None:
        with self.lock:
            self.collision_since_reset = False
            self.max_contact_count = 0
            self.goal_result_status = None

    def publish_speed_scale(self, speed_scale: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(speed_scale)]
        self.rl_limits_publisher.publish(msg)

    def current_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "lidar": self.lidar_ranges.copy(),
                "lidar_range_max": float(self.lidar_range_max),
                "linear_velocity": float(self.linear_velocity),
                "angular_velocity": float(self.angular_velocity),
                "collision": bool(self.collision_since_reset),
                "contact_count": int(self.max_contact_count),
                "scan_seq": int(self.scan_seq),
                "odom_seq": int(self.odom_seq),
                "nav_status": self.goal_result_status,
                "speed_scale": float(self.applied_speed_scale),
                "vx_max": float(self.applied_vx_limit),
                "wz_max": float(self.applied_wz_limit),
            }

    def get_map_pose(self) -> Tuple[float, float, float]:
        transform = self.tf_buffer.lookup_transform(
            "map",
            "base_foot_link",
            Time(),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = yaw_from_quaternion(
            float(rotation.z),
            float(rotation.w),
        )
        return float(translation.x), float(translation.y), yaw

    @staticmethod
    def _wait_future(future, timeout: float, description: str):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timeout while waiting for {description}")
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError(f"ROS stopped while waiting for {description}")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(f"{description} failed: {exception}")
        return future.result()

    def wait_until_ready(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        required_services = (
            (self.reset_simulation_client, "/reset_simulation"),
            (self.set_ekf_pose_client, "/set_pose"),
            (
                self.clear_global_costmap_client,
                "/global_costmap/clear_entirely_global_costmap",
            ),
            (
                self.clear_local_costmap_client,
                "/local_costmap/clear_entirely_local_costmap",
            ),
        )

        for client, name in required_services:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not client.wait_for_service(
                timeout_sec=remaining
            ):
                raise RuntimeError(f"Required service is unavailable: {name}")

        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not self.navigation_client.wait_for_server(
            timeout_sec=remaining
        ):
            raise RuntimeError(
                "Required action server is unavailable: /navigate_to_pose"
            )

        remaining = max(0.0, deadline - time.monotonic())
        if not self.scan_ready.wait(timeout=remaining):
            raise RuntimeError("No LaserScan received on /scan")

        remaining = max(0.0, deadline - time.monotonic())
        if not self.odom_ready.wait(timeout=remaining):
            raise RuntimeError(
                "No filtered odometry received on /odometry/filtered"
            )

        while (
            self.rl_limits_publisher.get_subscription_count() < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if self.rl_limits_publisher.get_subscription_count() < 1:
            raise RuntimeError(
                "RL supervisor is not subscribed to /rl_velocity_limits"
            )

    def cancel_navigation(self, timeout: float = 3.0) -> None:
        goal_handle = self.goal_handle
        if goal_handle is None:
            return
        with self.lock:
            self.goal_token += 1
        cancel_future = goal_handle.cancel_goal_async()
        self._wait_future(
            cancel_future,
            timeout,
            "NavigateToPose cancellation",
        )
        self.goal_handle = None
        self.goal_result_future = None

    def reset_simulation(self, timeout: float = 3.0) -> None:
        future = self.reset_simulation_client.call_async(Trigger.Request())
        response = self._wait_future(
            future,
            timeout,
            "/reset_simulation",
        )
        if not response.success:
            raise RuntimeError(response.message)

    def _zero_pose_message(
        self,
        frame_id: str,
    ) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        covariance = [0.0] * 36
        covariance[0] = 0.01
        covariance[7] = 0.01
        covariance[14] = 1.0e6
        covariance[21] = 1.0e6
        covariance[28] = 1.0e6
        covariance[35] = math.radians(2.0) ** 2
        msg.pose.covariance = covariance
        return msg

    def reset_ekf(self, timeout: float = 3.0) -> None:
        request = SetPose.Request()
        request.pose = self._zero_pose_message("odom")
        future = self.set_ekf_pose_client.call_async(request)
        self._wait_future(future, timeout, "/set_pose")

    def reset_amcl(self) -> None:
        # Publish more than once because AMCL may be transitioning immediately
        # after the previous goal cancellation.
        for _ in range(3):
            self.initial_pose_publisher.publish(
                self._zero_pose_message("map")
            )
            time.sleep(0.05)

    def clear_costmaps(self, timeout: float = 10.0) -> None:
        for client, name in (
            (
                self.clear_global_costmap_client,
                "global costmap clear",
            ),
            (
                self.clear_local_costmap_client,
                "local costmap clear",
            ),
        ):
            future = client.call_async(ClearEntireCostmap.Request())
            self._wait_future(future, timeout, name)

    def wait_for_reset_state(
        self,
        old_scan_seq: int,
        old_odom_seq: int,
        timeout: float = 6.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "waiting for fresh sensor data"

        fresh_scan_seen = False
        last_checked_odom_seq = old_odom_seq
        stable_samples = 0
        required_stable_samples = 3

        while rclpy.ok() and time.monotonic() < deadline:
            state = self.current_state()

            if state["scan_seq"] > old_scan_seq:
                fresh_scan_seen = True

            # Phải có LiDAR và odometry phát sinh sau reset_barrier.
            if (
                not fresh_scan_seen
                or state["odom_seq"] <= old_odom_seq
            ):
                time.sleep(0.02)
                continue

            # Chỉ đếm một lần cho mỗi mẫu odometry mới.
            if state["odom_seq"] == last_checked_odom_seq:
                time.sleep(0.01)
                continue

            last_checked_odom_seq = state["odom_seq"]

            try:
                x, y, yaw = self.get_map_pose()
            except TransformException as exc:
                stable_samples = 0
                last_error = str(exc)
                time.sleep(0.02)
                continue

            position_error = math.hypot(x, y)
            state_is_stable = (
                position_error <= 0.05
                and abs(normalize_angle(yaw)) <= math.radians(10.0)
                and abs(state["linear_velocity"]) <= 0.05
                and abs(state["angular_velocity"]) <= 0.10
            )

            if state_is_stable:
                stable_samples += 1
                if stable_samples >= required_stable_samples:
                    return
            else:
                stable_samples = 0

            last_error = (
                f"pose=({x:.3f}, {y:.3f}, "
                f"{math.degrees(yaw):.1f} deg), "
                f"velocity=({state['linear_velocity']:.3f}, "
                f"{state['angular_velocity']:.3f}), "
                f"stable={stable_samples}/{required_stable_samples}"
            )
            time.sleep(0.02)

        raise RuntimeError(
            "Reset did not converge to the initial state: " + last_error
        )

    def send_navigation_goal(
        self,
        x: float,
        y: float,
        yaw: float,
        timeout: float = 5.0,
    ) -> None:
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = (
            quaternion_from_yaw(yaw)
        )

        send_future = self.navigation_client.send_goal_async(goal)
        goal_handle = self._wait_future(
            send_future,
            timeout,
            "NavigateToPose goal acceptance",
        )
        if not goal_handle.accepted:
            raise RuntimeError("NavigateToPose rejected the episode goal")

        with self.lock:
            self.goal_handle = goal_handle
            self.goal_result_status = None
            self.goal_token += 1
            goal_token = self.goal_token
            self.goal_result_future = goal_handle.get_result_async()
            self.goal_result_future.add_done_callback(
                lambda future: self._navigation_result_callback(
                    future,
                    goal_token,
                )
            )

    def _navigation_result_callback(self, future, goal_token: int) -> None:
        try:
            wrapped_result = future.result()
            status = int(wrapped_result.status)
        except Exception as exc:
            self.get_logger().error(
                f"NavigateToPose result failed: {exc}"
            )
            status = GoalStatus.STATUS_ABORTED

        with self.lock:
            if goal_token == self.goal_token:
                self.goal_result_status = status
                self.goal_handle = None

    def wait_sim_duration(
        self,
        duration: float,
        wall_timeout: float,
    ) -> None:
        start_ns = self.get_clock().now().nanoseconds
        target_ns = start_ns + int(duration * 1.0e9)
        deadline = time.monotonic() + wall_timeout

        while rclpy.ok():
            if self.get_clock().now().nanoseconds >= target_ns:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"/clock did not advance {duration:.2f} s "
                    f"within {wall_timeout:.2f} wall seconds"
                )
            time.sleep(0.01)


class AgvRlEnv(gym.Env):
    """One-goal RL environment around the verified Nav2 MPPI stack."""

    metadata = {"render_modes": []}

    SPEED_SCALE_MIN = 0.40
    SPEED_SCALE_MAX = 1.00
    VX_MIN = 0.16
    VX_MAX = 0.40
    WZ_MIN = 0.20
    WZ_MAX = 0.50
    BASELINE_VX = 0.40
    BASELINE_WZ = 0.50

    MAX_GOAL_DISTANCE = 12.0
    CONTROL_DT = 1.0
    EPISODE_TIMEOUT = 120.0
    STUCK_WINDOW = 15.0
    STUCK_MIN_PROGRESS = 0.05

    GOAL_DISTANCE_TOLERANCE = 0.30
    GOAL_YAW_TOLERANCE = math.radians(15.0)
    STOP_LINEAR_TOLERANCE = 0.05
    STOP_ANGULAR_TOLERANCE = 0.10

    def __init__(
        self,
        goal: Tuple[float, float, float] = (8.5, 0.0, 0.0),
    ) -> None:
        super().__init__()

        if not rclpy.ok():
            rclpy.init()

        self.ros = AgvRosInterface()
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.ros)
        self.spin_thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self.spin_thread.start()

        self.goal = tuple(float(value) for value in goal)
        self.current_vx_limit = self.BASELINE_VX
        self.current_wz_limit = self.BASELINE_WZ
        self.current_speed_scale = self.SPEED_SCALE_MAX
        self.last_action = np.ones(1, dtype=np.float32)
        self.previous_distance = 0.0
        self.episode_start_ns = 0
        self.progress_history = deque()
        self.closed = False

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # 24 lidar sectors + distance + heading error + vx + wz +
        # current vx_max + current wz_max = 30 values.
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(30,),
            dtype=np.float32,
        )

        self.ros.wait_until_ready()

    @staticmethod
    def _map_action(
        normalized: float,
        lower: float,
        upper: float,
    ) -> float:
        clipped = float(np.clip(normalized, -1.0, 1.0))
        return lower + 0.5 * (clipped + 1.0) * (upper - lower)

    @staticmethod
    def _normalize_limit(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        if upper <= lower:
            return 0.0
        return float(
            np.clip(
                2.0 * (value - lower) / (upper - lower) - 1.0,
                -1.0,
                1.0,
            )
        )

    def _pose_and_goal_metrics(
        self,
    ) -> Tuple[float, float, float, float, float]:
        x, y, yaw = self.ros.get_map_pose()
        goal_x, goal_y, goal_yaw = self.goal
        dx = goal_x - x
        dy = goal_y - y
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        heading_error = normalize_angle(bearing - yaw)
        final_yaw_error = normalize_angle(goal_yaw - yaw)
        return x, y, distance, heading_error, final_yaw_error

    def _observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state = self.ros.current_state()
        x, y, distance, heading_error, final_yaw_error = (
            self._pose_and_goal_metrics()
        )

        lidar_physical = state["lidar"]
        lidar_normalized = np.clip(
            lidar_physical / max(state["lidar_range_max"], 1.0e-3),
            0.0,
            1.0,
        )

        observation = np.concatenate(
            (
                lidar_normalized.astype(np.float32),
                np.asarray(
                    [
                        np.clip(
                            distance / self.MAX_GOAL_DISTANCE,
                            0.0,
                            1.0,
                        ),
                        np.clip(heading_error / math.pi, -1.0, 1.0),
                        np.clip(
                            state["linear_velocity"] / self.VX_MAX,
                            -1.0,
                            1.0,
                        ),
                        np.clip(
                            state["angular_velocity"] / self.WZ_MAX,
                            -1.0,
                            1.0,
                        ),
                        self._normalize_limit(
                            state["vx_max"],
                            self.VX_MIN,
                            self.VX_MAX,
                        ),
                        self._normalize_limit(
                            state["wz_max"],
                            self.WZ_MIN,
                            self.WZ_MAX,
                        ),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)

        info = {
            "x": x,
            "y": y,
            "distance_to_goal": distance,
            "heading_error": heading_error,
            "final_yaw_error": final_yaw_error,
            "linear_velocity": state["linear_velocity"],
            "angular_velocity": state["angular_velocity"],
            "min_lidar": float(np.min(lidar_physical)),
            "collision": state["collision"],
            "contact_count": state["contact_count"],
            "nav_status": state["nav_status"],
            "speed_scale": state["speed_scale"],
            "vx_max": state["vx_max"],
            "wz_max": state["wz_max"],
            "requested_speed_scale": self.current_speed_scale,
        }
        return observation, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        if options and "goal" in options:
            goal = options["goal"]
            if len(goal) != 3:
                raise ValueError("goal must be (x, y, yaw)")
            self.goal = tuple(float(value) for value in goal)

        # Phải nằm ngoài khối if options.
        self.ros.cancel_navigation()
        self.ros.reset_episode_flags()
        self.ros.reset_simulation()
        self.ros.reset_ekf()
        self.ros.reset_amcl()

        # Chỉ chấp nhận sensor và TF phát sinh sau chuỗi reset.
        reset_barrier = self.ros.current_state()

        self.ros.wait_for_reset_state(
            old_scan_seq=reset_barrier["scan_seq"],
            old_odom_seq=reset_barrier["odom_seq"],
        )

        self.ros.clear_costmaps(timeout=10.0)

        self.current_speed_scale = self.SPEED_SCALE_MAX
        self.current_vx_limit = self.BASELINE_VX
        self.current_wz_limit = self.BASELINE_WZ
        self.ros.publish_speed_scale(self.current_speed_scale)

        self.last_action = np.asarray([1.0], dtype=np.float32)
        self.progress_history.clear()
        self.ros.send_navigation_goal(*self.goal)

        observation, info = self._observation()
        self.previous_distance = float(info["distance_to_goal"])
        self.episode_start_ns = self.ros.get_clock().now().nanoseconds
        self.progress_history.append((0.0, self.previous_distance))
        info["reset_ok"] = True
        return observation, info

    def step(self, action):
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (1,):
            raise ValueError(
                f"action must have shape (1,), got {action_array.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action contains NaN or Inf")

        clipped_action = np.clip(action_array, -1.0, 1.0)
        speed_scale = self._map_action(
            clipped_action[0],
            self.SPEED_SCALE_MIN,
            self.SPEED_SCALE_MAX,
        )
        vx_limit = self.BASELINE_VX * speed_scale
        wz_limit = self.BASELINE_WZ * speed_scale
        self.current_speed_scale = speed_scale
        self.current_vx_limit = vx_limit
        self.current_wz_limit = wz_limit
        self.ros.publish_speed_scale(speed_scale)
        self.ros.wait_sim_duration(
            self.CONTROL_DT,
            wall_timeout=max(10.0, 10.0 * self.CONTROL_DT),
        )

        observation, info = self._observation()
        distance = float(info["distance_to_goal"])
        progress = self.previous_distance - distance
        action_change = float(
            np.linalg.norm(clipped_action - self.last_action)
        )

        reward = 10.0 * progress - 0.05 - 0.10 * action_change

        safe_distance = 0.60
        if info["min_lidar"] < safe_distance:
            reward -= 0.50 * (
                safe_distance - info["min_lidar"]
            ) / safe_distance

        elapsed = (
            self.ros.get_clock().now().nanoseconds
            - self.episode_start_ns
        ) * 1.0e-9
        self.progress_history.append((elapsed, distance))
        while (
            self.progress_history
            and elapsed - self.progress_history[0][0] > self.STUCK_WINDOW
        ):
            self.progress_history.popleft()

        # Kiểm tra robot đã nằm đúng vùng goal hay chưa.
        pose_in_goal_gate = (
            distance <= self.GOAL_DISTANCE_TOLERANCE
            and abs(info["final_yaw_error"]) <= self.GOAL_YAW_TOLERANCE
        )

        # Kiểm tra robot đã dừng hẳn hay chưa.
        robot_stopped = (
            abs(info["linear_velocity"]) <= self.STOP_LINEAR_TOLERANCE
            and abs(info["angular_velocity"]) <= self.STOP_ANGULAR_TOLERANCE
        )

        # Chỉ công nhận success khi vừa đúng pose, vừa đã dừng.
        success = pose_in_goal_gate and robot_stopped
        collision = bool(info["collision"])

        stuck = False
        if (
            len(self.progress_history) >= 2
            and elapsed - self.progress_history[0][0]
            >= self.STUCK_WINDOW - self.CONTROL_DT
        ):
            window_progress = (
                self.progress_history[0][1]
                - self.progress_history[-1][1]
            )
            stuck = (
                window_progress < self.STUCK_MIN_PROGRESS
                and abs(info["linear_velocity"]) < 0.03
                and abs(info["angular_velocity"]) < 0.05
            )

        nav_status = info["nav_status"]
        nav_failed = nav_status in (
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        )
        nav_succeeded_outside_gate = (
            nav_status == GoalStatus.STATUS_SUCCEEDED and not pose_in_goal_gate
        )
        timed_out = elapsed >= self.EPISODE_TIMEOUT

        terminated = success or collision or nav_failed
        truncated = stuck or timed_out or nav_succeeded_outside_gate

        termination_reason = ""
        if success:
            reward += 100.0
            termination_reason = "success"
        elif collision:
            reward -= 100.0
            termination_reason = "collision"
        elif nav_failed:
            reward -= 50.0
            termination_reason = "nav2_failed"
        elif stuck:
            reward -= 20.0
            termination_reason = "stuck"
        elif timed_out:
            reward -= 30.0
            termination_reason = "timeout"
        elif nav_succeeded_outside_gate:
            reward -= 30.0
            termination_reason = "nav2_success_outside_metric_gate"

        info.update(
            {
                "progress": progress,
                "action_change": action_change,
                "elapsed_sim_time": elapsed,
                "success": success,
                "stuck": stuck,
                "termination_reason": termination_reason,
                "pose_in_goal_gate": pose_in_goal_gate,
                "robot_stopped": robot_stopped,
            }
        )

        self.previous_distance = distance
        self.last_action = clipped_action.copy()
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        try:
            self.ros.cancel_navigation()
            self.ros.publish_speed_scale(self.SPEED_SCALE_MAX)
        except Exception:
            pass

        self.executor.shutdown(timeout_sec=2.0)
        self.ros.destroy_node()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=2.0)
        if rclpy.ok():
            rclpy.shutdown()


def smoke_test(episodes: int = 2, max_steps: int = 10) -> int:
    env = AgvRlEnv()
    try:
        for episode_index in range(episodes):
            observation, info = env.reset(seed=7 + episode_index)
            print(
                f"EPISODE {episode_index + 1} RESET OK:",
                f"obs={observation.shape}",
                f"distance={info['distance_to_goal']:.3f}",
                f"min_lidar={info['min_lidar']:.3f}",
            )

            for step_index in range(max_steps):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                print(
                    f"episode={episode_index + 1}",
                    f"step={step_index + 1:03d}",
                    f"action=({action[0]:+.2f})",
                    f"scale={info['speed_scale']:.3f}",
                    f"limits=({info['vx_max']:.3f},{info['wz_max']:.3f})",
                    f"distance={info['distance_to_goal']:.3f}",
                    f"min_scan={info['min_lidar']:.3f}",
                    f"reward={reward:+.3f}",
                    f"done={terminated or truncated}",
                    f"reason={info['termination_reason'] or '-'}",
                )
                if terminated or truncated:
                    break
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(smoke_test())
