#!/usr/bin/env python3
"""MuJoCo Gym environment for Gazebo Phase-2 SAC fine-tuning.

This is the active environment in the original workspace. It preserves the
checkpoint contract validated by the isolated A/B workspace.
Updated with:
- Continuing Task mode (episodes do not terminate on goal reach/abort).
- Random & Safe goal generation using Global Costmap across the map.
- Soft Reset on Truncation: Never reset to (0,0) unless there's a collision.
- [PATCH] info["is_success"] now populated at episode end so SB3's
  ep_success_buffer / rollout/success_rate are no longer empty.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry, OccupancyGrid
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from robot_localization.srv import SetPose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


Goal = Tuple[float, float, float]


def quaternion_from_yaw(yaw: float) -> Tuple[float, float]:
    return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def yaw_from_quaternion(z: float, w: float) -> float:
    return 2.0 * math.atan2(z, w)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class AgvRosInterface(Node):
    """ROS backend connecting the 26D/4D Gym contract to MuJoCo/Nav2."""

    LIDAR_BEAMS = 24
    LIDAR_ANGLE_MIN = -2.0943951
    LIDAR_ANGLE_MAX = 2.0943951
    LIDAR_REPLACEMENT_RANGE = 10.0

    def __init__(self) -> None:
        super().__init__(
            "gazebo_phase2_finetune_environment",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.lock = threading.RLock()
        self.scan_ready = threading.Event()
        self.odom_ready = threading.Event()
        self.scan_seq = 0
        self.odom_seq = 0
        self.lidar = np.full(
            self.LIDAR_BEAMS,
            self.LIDAR_REPLACEMENT_RANGE,
            dtype=np.float32,
        )
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.collision_since_reset = False
        self.collision_active = False
        self.collision_event_count = 0
        self.max_contact_count = 0
        self.minimum_lidar_since_reset = self.LIDAR_REPLACEMENT_RANGE

        self.goal_handle = None
        self.goal_result_status: Optional[int] = None
        self.goal_result_future = None
        self.goal_token = 0

        # Lưu trữ Costmap
        self.costmap = None
        self.costmap_occupied_threshold = 50

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        initial_pose_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        costmap_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            LaserScan,
            "/scan_raw",
            self._scan_callback,
            sensor_qos,
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
        self.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            self._costmap_callback,
            costmap_qos,
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
        self.set_ekf_pose_client = self.create_client(SetPose, "/set_pose")
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
        )
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
        )
        self.mppi_parameter_client = self.create_client(
            SetParameters,
            "/controller_server/set_parameters",
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
        mask = (
            (angles >= self.LIDAR_ANGLE_MIN - 1.0e-5)
            & (angles <= self.LIDAR_ANGLE_MAX + 1.0e-5)
        )
        filtered = ranges[mask]
        if filtered.size < self.LIDAR_BEAMS:
            return
        filtered = np.where(
            (~np.isfinite(filtered)) | (filtered < 0.15),
            self.LIDAR_REPLACEMENT_RANGE,
            filtered,
        )
        reduced = np.asarray(
            [
                np.min(sector)
                for sector in np.array_split(filtered, self.LIDAR_BEAMS)
            ],
            dtype=np.float32,
        )
        with self.lock:
            self.lidar = reduced
            self.minimum_lidar_since_reset = min(
                self.minimum_lidar_since_reset,
                float(np.min(reduced)),
            )
            self.scan_seq += 1
            self.scan_ready.set()

    def _odom_callback(self, msg: Odometry) -> None:
        with self.lock:
            self.linear_velocity = float(msg.twist.twist.linear.x)
            self.angular_velocity = float(msg.twist.twist.angular.z)
            self.odom_seq += 1
            self.odom_ready.set()

    def _collision_callback(self, msg: Bool) -> None:
        with self.lock:
            if msg.data and not self.collision_active:
                self.collision_event_count += 1
            if msg.data:
                self.collision_since_reset = True
            self.collision_active = bool(msg.data)

    def _contact_callback(self, msg: Int32) -> None:
        with self.lock:
            self.max_contact_count = max(
                self.max_contact_count,
                int(msg.data),
            )

    def _costmap_callback(self, msg: OccupancyGrid) -> None:
        with self.lock:
            self.costmap = msg

    def is_position_free(self, x: float, y: float) -> bool:
        """Check if position is free on the global costmap."""
        with self.lock:
            if self.costmap is None:
                return False

            info = self.costmap.info
            mx = int((x - info.origin.position.x) / info.resolution)
            my = int((y - info.origin.position.y) / info.resolution)

            if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
                return False

            index = my * info.width + mx
            value = self.costmap.data[index]

            if value < 0:
                return False
            return value < self.costmap_occupied_threshold

    def current_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "lidar": self.lidar.copy(),
                "scan_seq": int(self.scan_seq),
                "odom_seq": int(self.odom_seq),
                "linear_velocity": float(self.linear_velocity),
                "angular_velocity": float(self.angular_velocity),
                "collision": bool(self.collision_since_reset),
                "collision_event_count": int(self.collision_event_count),
                "contact_count": int(self.max_contact_count),
                "minimum_lidar": float(self.minimum_lidar_since_reset),
                "nav_status": self.goal_result_status,
            }

    def reset_episode_flags(self) -> None:
        with self.lock:
            self.collision_since_reset = False
            self.collision_active = False
            self.collision_event_count = 0
            self.max_contact_count = 0
            self.minimum_lidar_since_reset = self.LIDAR_REPLACEMENT_RANGE
            self.goal_result_status = None

    def get_map_pose(self) -> Tuple[float, float, float]:
        transform = self.tf_buffer.lookup_transform(
            "map",
            "base_foot_link",
            Time(),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion(float(rotation.z), float(rotation.w)),
        )

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

    def wait_until_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        clients = (
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
            (
                self.mppi_parameter_client,
                "/controller_server/set_parameters",
            ),
        )
        for client, name in clients:
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
            raise RuntimeError("No LaserScan received on /scan_raw")
        remaining = max(0.0, deadline - time.monotonic())
        if not self.odom_ready.wait(timeout=remaining):
            raise RuntimeError(
                "No filtered odometry received on /odometry/filtered"
            )

    def cancel_navigation(self, timeout: float = 3.0) -> None:
        goal_handle = self.goal_handle
        if goal_handle is None:
            return
        with self.lock:
            self.goal_token += 1
        future = goal_handle.cancel_goal_async()
        self._wait_future(future, timeout, "NavigateToPose cancellation")
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

    def _zero_pose_message(self, frame_id: str) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
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
        for _ in range(3):
            self.initial_pose_publisher.publish(
                self._zero_pose_message("map")
            )
            time.sleep(0.05)

    def wait_for_reset_state(
        self,
        old_scan_seq: int,
        old_odom_seq: int,
        timeout: float = 6.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        stable_samples = 0
        last_odom_seq = old_odom_seq
        last_error = "waiting for fresh sensor data"
        while rclpy.ok() and time.monotonic() < deadline:
            state = self.current_state()
            if (
                state["scan_seq"] <= old_scan_seq
                or state["odom_seq"] <= old_odom_seq
                or state["odom_seq"] == last_odom_seq
            ):
                time.sleep(0.02)
                continue
            last_odom_seq = state["odom_seq"]
            try:
                x, y, yaw = self.get_map_pose()
            except TransformException as exc:
                stable_samples = 0
                last_error = str(exc)
                time.sleep(0.02)
                continue
            stable = (
                math.hypot(x, y) <= 0.05
                and abs(normalize_angle(yaw)) <= math.radians(10.0)
                and abs(state["linear_velocity"]) <= 0.05
                and abs(state["angular_velocity"]) <= 0.10
            )
            if stable:
                stable_samples += 1
                if stable_samples >= 3:
                    return
            else:
                stable_samples = 0
            last_error = (
                f"pose=({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f} deg), "
                f"velocity=({state['linear_velocity']:.3f}, "
                f"{state['angular_velocity']:.3f}), "
                f"stable={stable_samples}/3"
            )
        raise RuntimeError(
            "Reset did not converge to the initial state: " + last_error
        )

    def clear_costmaps(self, timeout: float = 10.0) -> None:
        for client, name in (
            (self.clear_global_costmap_client, "global costmap clear"),
            (self.clear_local_costmap_client, "local costmap clear"),
        ):
            future = client.call_async(ClearEntireCostmap.Request())
            self._wait_future(future, timeout, name)

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
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = (
            quaternion_from_yaw(yaw)
        )
        future = self.navigation_client.send_goal_async(goal)
        handle = self._wait_future(
            future,
            timeout,
            "NavigateToPose goal acceptance",
        )
        if not handle.accepted:
            raise RuntimeError("NavigateToPose rejected the training goal")
        with self.lock:
            self.goal_handle = handle
            self.goal_result_status = None
            self.goal_token += 1
            token = self.goal_token
            self.goal_result_future = handle.get_result_async()
            self.goal_result_future.add_done_callback(
                lambda result_future: self._navigation_result_callback(
                    result_future,
                    token,
                )
            )

    def _navigation_result_callback(self, future, token: int) -> None:
        try:
            status = int(future.result().status)
        except Exception as exc:
            self.get_logger().error(f"NavigateToPose result failed: {exc}")
            status = GoalStatus.STATUS_ABORTED
        with self.lock:
            if token == self.goal_token:
                self.goal_result_status = status
                self.goal_handle = None

    def set_mppi_parameters(
        self,
        vx_max: float,
        path_weight: float,
        goal_weight: float,
        goal_angle_weight: float,
        timeout: float = 3.0,
    ) -> None:
        request = SetParameters.Request()
        values = (
            ("FollowPath.vx_max", vx_max),
            ("FollowPath.PathAlignCritic.cost_weight", path_weight),
            ("FollowPath.GoalCritic.cost_weight", goal_weight),
            (
                "FollowPath.GoalAngleCritic.cost_weight",
                goal_angle_weight,
            ),
        )
        for name, value in values:
            parameter = ParameterMsg()
            parameter.name = name
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE,
                double_value=float(value),
            )
            request.parameters.append(parameter)
        future = self.mppi_parameter_client.call_async(request)
        response = self._wait_future(
            future,
            timeout,
            "/controller_server/set_parameters",
        )
        failures = [
            result.reason or "unspecified failure"
            for result in response.results
            if not result.successful
        ]
        if failures:
            raise RuntimeError(
                "MPPI rejected Phase-2 action: " + "; ".join(failures)
            )


class AgvRlEnv(gym.Env):
    """Phase-2 environment used to fine-tune the Gazebo checkpoint."""

    metadata = {"render_modes": []}

    ACTION_LOW = np.asarray([0.05, 0.5, 5.0, 1.0], dtype=np.float32)
    ACTION_HIGH = np.asarray([1.0, 15.0, 25.0, 15.0], dtype=np.float32)
    SOURCE_NOMINAL_ACTION = np.asarray(
        [0.25, 5.0, 14.0, 8.0],
        dtype=np.float32,
    )
    NATIVE_MUJOCO_ACTION = np.asarray(
        [0.40, 14.0, 5.0, 3.0],
        dtype=np.float32,
    )

    # KHÔI PHỤC COMBINED_GOALS để train_sac.py có thể đọc được
    COMBINED_GOALS: Tuple[Goal, ...] = (
        (9.5, 0.0, 0.0),
        (9.5, 5.0, 0.0),
        (9.5, -5.0, 0.0),
    )

    MAX_SCAN_RANGE = 10.0
    MAX_EXPECTED_GOAL_DISTANCE = 20.0
    STEP_WAIT_DURATION = 5.05
    SAFE_DISTANCE = 0.60
    COLLISION_DISTANCE = 0.25
    SAFETY_REWARD_CLIP = -20.0
    COLLISION_REWARD = -50.0
    GOAL_REWARD = 100.0
    NAV_ABORT_REWARD = -30.0

    # [PATCH] Nếu True: chỉ tính episode là "thành công" khi đã tới >=1 goal
    # VÀ episode không kết thúc do va chạm. Nếu False: chỉ cần đã từng tới
    # goal là tính thành công (kể cả nếu sau đó va chạm). Xem giải thích ở
    # cuối phương thức step().
    SUCCESS_REQUIRES_NO_COLLISION = True

    def __init__(
        self,
        goal: Goal = (9.5, 0.0, 0.0),
        goals: Optional[Sequence[Goal]] = None,
        goal_sampling: str = "cycle",
        max_episode_steps: int = 35,
    ) -> None:
        super().__init__()
        # Giữ nguyên khung kiểm tra đối số gốc để tương thích với script ngoài
        if goal_sampling not in ("fixed", "cycle", "random"):
            raise ValueError(
                "goal_sampling must be fixed, cycle or random"
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        candidate_goals = tuple(goals) if goals is not None else (goal,)
        if not candidate_goals:
            raise ValueError("At least one training goal is required")
        validated_goals = []
        for candidate in candidate_goals:
            if len(candidate) != 3:
                raise ValueError("Each goal must be (x, y, yaw)")
            converted = tuple(float(value) for value in candidate)
            if not all(math.isfinite(value) for value in converted):
                raise ValueError("Goal values must be finite")
            validated_goals.append(converted)

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

        self.goals: Tuple[Goal, ...] = tuple(validated_goals)
        self.goal_sampling = goal_sampling
        self.goal_cursor = 0
        self.goal = self.goals[0]
        self.max_episode_steps = int(max_episode_steps)
        self.episode_step_count = 0
        self.previous_distance = 0.0
        self.closed = False

        # Chỉ Reset Hard (về 0,0) ở lần chạy đầu tiên hoặc khi có va chạm
        self._needs_hard_reset = True

        # Các biến bổ sung cho Random Goal Generation
        self.min_goal_distance = 1.0
        self.max_goal_generation_attempts = 20
        self.goals_reached_this_episode = 0

        self.action_space = gym.spaces.Box(
            low=self.ACTION_LOW.copy(),
            high=self.ACTION_HIGH.copy(),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(26,),
            dtype=np.float32,
        )
        try:
            self.ros.wait_until_ready()
        except Exception:
            self.close()
            raise

    def _generate_new_goal(self) -> None:
        """Sinh đích đến ngẫu nhiên trên toàn bản đồ, đảm bảo không chạm vật cản."""
        try:
            robot_x, robot_y, _ = self.ros.get_map_pose()
        except Exception:
            robot_x, robot_y = 0.0, 0.0

        costmap_info = None
        with self.ros.lock:
            if self.ros.costmap is not None:
                costmap_info = self.ros.costmap.info

        if costmap_info is not None:
            origin_x = costmap_info.origin.position.x
            origin_y = costmap_info.origin.position.y
            width_m = costmap_info.width * costmap_info.resolution
            height_m = costmap_info.height * costmap_info.resolution

            max_global_attempts = self.max_goal_generation_attempts * 5

            for _ in range(max_global_attempts):
                cand_x = float(self.np_random.uniform(origin_x, origin_x + width_m))
                cand_y = float(self.np_random.uniform(origin_y, origin_y + height_m))

                if math.hypot(cand_x - robot_x, cand_y - robot_y) < self.min_goal_distance:
                    continue

                if self.ros.is_position_free(cand_x, cand_y):
                    self.goal = (cand_x, cand_y, 0.0)
                    print(f"[Goal] Đã sinh đích toàn map thành công: ({cand_x:.2f}, {cand_y:.2f})")
                    return

        print("[Goal] Không tìm được điểm toàn map, dùng fallback bán kính xung quanh robot.")
        for r_max in (10.0, 6.0, 3.0, 1.5):
            for _ in range(self.max_goal_generation_attempts):
                dx = float(self.np_random.uniform(-r_max, r_max))
                dy = float(self.np_random.uniform(-r_max, r_max))
                if math.hypot(dx, dy) < self.min_goal_distance:
                    continue

                cand_x = robot_x + dx
                cand_y = robot_y + dy
                if self.ros.is_position_free(cand_x, cand_y):
                    self.goal = (cand_x, cand_y, 0.0)
                    return

        for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            cand_x = float(robot_x + self.min_goal_distance * math.cos(ang))
            cand_y = float(robot_y + self.min_goal_distance * math.sin(ang))
            if self.ros.is_position_free(cand_x, cand_y):
                self.goal = (cand_x, cand_y, 0.0)
                return

        self.goal = (float(robot_x + self.min_goal_distance), float(robot_y), 0.0)

    def _pose_and_distance(self) -> Tuple[float, float, float, float]:
        x, y, _yaw = self.ros.get_map_pose()
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        return x, y, math.hypot(dx, dy), math.atan2(dy, dx)

    def _observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state = self.ros.current_state()
        x, y, distance, global_goal_bearing = self._pose_and_distance()
        scan_normalized = np.clip(
            state["lidar"] / self.MAX_SCAN_RANGE,
            0.0,
            1.0,
        )
        observation = np.concatenate(
            (
                scan_normalized,
                np.asarray(
                    [
                        np.clip(
                            distance / self.MAX_EXPECTED_GOAL_DISTANCE,
                            0.0,
                            1.0,
                        ),
                        np.clip(
                            global_goal_bearing / math.pi,
                            -1.0,
                            1.0,
                        ),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)
        if observation.shape != (26,) or not np.all(
            np.isfinite(observation)
        ):
            raise RuntimeError(
                f"Invalid Phase-2 observation: shape={observation.shape}"
            )
        info = {
            "x": x,
            "y": y,
            "goal": list(self.goal),
            "distance_to_goal": distance,
            "global_goal_bearing": global_goal_bearing,
            "min_lidar": float(np.min(state["lidar"])),
            "min_lidar_episode": float(state["minimum_lidar"]),
            "contact_count": int(state["contact_count"]),
            "collision_state": bool(state["collision"]),
            "collision_event_count": int(state["collision_event_count"]),
            "nav_status": state["nav_status"],
            "goals_reached_this_episode": self.goals_reached_this_episode,
        }
        return observation, info

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (4,):
            raise ValueError(f"action must have shape (4,), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("action contains NaN or Inf")
        clipped = np.clip(array, self.ACTION_LOW, self.ACTION_HIGH)
        applied = np.asarray(
            [round(float(value), 2) for value in clipped],
            dtype=np.float32,
        )
        self.ros.set_mppi_parameters(*[float(value) for value in applied])
        return applied

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        self.ros.cancel_navigation()

        if self._needs_hard_reset:
            print("\n[Môi trường] Khởi tạo hoặc Va chạm: Dịch chuyển robot về (0,0)...")
            self.ros.reset_simulation()
            self.ros.reset_ekf()
            self.ros.reset_amcl()
            barrier = self.ros.current_state()
            self.ros.wait_for_reset_state(
                old_scan_seq=barrier["scan_seq"],
                old_odom_seq=barrier["odom_seq"],
            )
            self._needs_hard_reset = False
        else:
            print("\n[Môi trường] Hết giờ (Truncated): Giữ nguyên vị trí robot, chạy tiếp...")
            time.sleep(0.5)  # Đợi nhẹ 0.5s để cảm biến đồng bộ

        self.ros.reset_episode_flags()
        self.ros.clear_costmaps(timeout=10.0)
        self._apply_action(self.SOURCE_NOMINAL_ACTION)

        # Luôn tự động random goal mới ngay khi reset
        self.goals_reached_this_episode = 0
        self._generate_new_goal()
        self.ros.send_navigation_goal(*self.goal)

        observation, info = self._observation()
        self.previous_distance = float(info["distance_to_goal"])
        self.episode_step_count = 0
        info["reset_ok"] = True
        return observation, info

    def step(self, action):
        self.episode_step_count += 1
        applied = self._apply_action(action)
        deadline = time.monotonic() + self.STEP_WAIT_DURATION
        while rclpy.ok() and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if not rclpy.ok():
            raise RuntimeError("ROS stopped during the action interval")

        observation, info = self._observation()
        distance = float(info["distance_to_goal"])
        # [PATCH] Dùng min_lidar_episode (tích luỹ liên tục từ mọi bản tin
        # lidar nhận được kể từ lần reset gần nhất) thay vì min_lidar (chỉ
        # là 1 lát cắt tức thời tại đúng thời điểm đọc). Nếu robot chạm
        # tường thoáng qua rồi bị đẩy bật ra trước khi step() kịp đọc,
        # min_lidar sẽ bỏ sót va chạm đó - min_lidar_episode thì không.
        minimum_obstacle_distance = float(info["min_lidar_episode"])
        progress = self.previous_distance - distance

        reward_progress = 15.0 * progress
        reward_safety = 0.0
        reward_speed = 0.0
        reward_step = -0.05
        reward_goal = 0.0
        terminated = False
        truncated = False
        reason = ""

        physical_collision = bool(info["collision_state"])
        lidar_collision = minimum_obstacle_distance < self.COLLISION_DISTANCE
        # [PATCH] Thêm contact_count làm nguồn thứ 3, độc lập, đến trực
        # tiếp từ physics engine MuJoCo (topic /contact_count). Trước đây
        # info["contact_count"] được track nhưng KHÔNG được dùng ở đâu cả
        # trong quyết định va chạm - nếu /collision_state bị cấu hình sai
        # hoặc không publish đúng lúc, đây là lưới an toàn dự phòng.
        contact_collision = int(info["contact_count"]) > 0
        if physical_collision or lidar_collision or contact_collision:
            reward_safety = self.COLLISION_REWARD
            terminated = True
            reason = "collision"
            self._needs_hard_reset = True  # CHỈ KHI NÀY MỚI BẬT CỜ RESET VỀ 0,0
            # [PATCH] Log rõ nguồn nào phát hiện va chạm - hữu ích để debug
            # xem /collision_state có đang publish đúng không, hay chỉ có
            # lidar/contact_count bắt được.
            print(
                "💥 VA CHẠM PHÁT HIỆN BỞI: "
                f"physical_collision={physical_collision}, "
                f"lidar_collision={lidar_collision} "
                f"(min={minimum_obstacle_distance:.3f}m), "
                f"contact_collision={contact_collision} "
                f"(contact_count={info['contact_count']})"
            )

        elif minimum_obstacle_distance <= self.SAFE_DISTANCE:
            reward_safety = max(
                -5.0
                * (
                    self.SAFE_DISTANCE
                    / max(minimum_obstacle_distance, 1.0e-3)
                    - 1.0
                ),
                self.SAFETY_REWARD_CLIP,
            )
        else:
            reward_speed = 2.0 * float(applied[0])

        nav_status = info["nav_status"]
        if not terminated and nav_status == GoalStatus.STATUS_SUCCEEDED:
            reward_goal = self.GOAL_REWARD
            self.goals_reached_this_episode += 1
            print(f"✅ ĐÃ TỚI ĐÍCH! (Tổng: {self.goals_reached_this_episode}) -> Đang sinh goal mới...")

            self._generate_new_goal()
            self.ros.send_navigation_goal(*self.goal)

            _, _, distance, _ = self._pose_and_distance()
            reason = "success_but_continue"

        elif not terminated and nav_status == GoalStatus.STATUS_ABORTED:
            reward_goal = self.NAV_ABORT_REWARD
            print("⚠️ NAV2 BỊ KẸT (ABORTED) -> Đang sinh goal mới...")

            self._generate_new_goal()
            self.ros.send_navigation_goal(*self.goal)

            _, _, distance, _ = self._pose_and_distance()
            reason = "aborted_but_continue"

        if not terminated and self.episode_step_count >= self.max_episode_steps:
            truncated = True
            reason = "max_steps"

        reward = (
            reward_progress
            + reward_safety
            + reward_speed
            + reward_step
            + reward_goal
        )
        self.previous_distance = distance

        # [PATCH] Populate info["is_success"] khi episode thực sự kết thúc,
        # để SB3 (ep_success_buffer / rollout/success_rate) đọc được.
        # Đây là continuing task: robot có thể tới nhiều goal trong 1
        # episode (goals_reached_this_episode đếm số lần). Ta coi episode
        # là "thành công" nếu đã tới được >=1 goal.
        #
        # SUCCESS_REQUIRES_NO_COLLISION=True (mặc định): nếu robot tới goal
        # rồi SAU ĐÓ vẫn đâm trước khi episode kết thúc, KHÔNG tính là
        # thành công - vì mục tiêu là điều hướng an toàn, không chỉ là
        # "đã từng chạm goal".
        if terminated or truncated:
            reached_at_least_one_goal = self.goals_reached_this_episode > 0
            ended_in_collision = reason == "collision"
            if self.SUCCESS_REQUIRES_NO_COLLISION:
                info["is_success"] = (
                    reached_at_least_one_goal and not ended_in_collision
                )
            else:
                info["is_success"] = reached_at_least_one_goal

        info.update(
            {
                "termination_reason": reason,
                "progress": progress,
                "applied_action": [float(value) for value in applied],
                "physical_collision": physical_collision,
                "lidar_collision": lidar_collision,
                "contact_collision": contact_collision,  # [PATCH] để debug/log riêng nguồn này
                "reward_progress": reward_progress,
                "reward_safety": reward_safety,
                "reward_speed": reward_speed,
                "reward_goal": reward_goal,
            }
        )
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.ros.cancel_navigation()
            self.ros.set_mppi_parameters(
                *[float(value) for value in self.NATIVE_MUJOCO_ACTION]
            )
        except Exception:
            pass
        self.executor.shutdown(timeout_sec=2.0)
        self.ros.destroy_node()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=2.0)
        if rclpy.ok():
            rclpy.shutdown()


# Compatibility aliases for existing Phase-2 utilities.
TransferRosInterface = AgvRosInterface
GazeboMujocoTransferEnv = AgvRlEnv


if __name__ == "__main__":
    environment = AgvRlEnv(
        goals=AgvRlEnv.COMBINED_GOALS,
        goal_sampling="cycle",
        max_episode_steps=300,
    )
    try:
        observation, reset_info = environment.reset(seed=42)
        print("reset", observation.shape, reset_info)
        for _ in range(3):
            action = environment.action_space.sample()
            observation, reward, terminated, truncated, info = (
                environment.step(action)
            )
            print(reward, terminated, truncated, info["termination_reason"])
            if terminated or truncated:
                break
    finally:
        environment.close()