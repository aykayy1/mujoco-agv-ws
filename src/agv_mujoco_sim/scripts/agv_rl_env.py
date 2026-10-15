#!/usr/bin/env python3
"""MuJoCo Gym environment for the one-dimensional Nav2 speed supervisor.

Updated with:
- Continuing Task mode (episodes do not terminate on goal reach/abort).
- Random & Safe goal generation using Global Costmap across the map.
- Soft reset on ordinary time limits; hard reset after collision or a
  sustained front safety stop with no robot motion.
- [PATCH] info["is_success"] now populated at episode end so SB3's
  ep_success_buffer / rollout/success_rate are no longer empty.
- [PATCH] Hard reset after collision now RETRIES automatically instead of
  crashing the whole training process if one ROS2 call times out.
- Goal bearing is egocentric and measured velocity comes from filtered odom.
- RL action is exactly ``[speed_scale]``.  The environment publishes it on
  ``/rl_velocity_limits`` and observes the supervisor acknowledgement on
  ``/rl_velocity_limits_applied``.  It never changes MPPI parameters.
- Observation includes per-sector TTC, the actually applied scale and hazard
  persistence so lateral/crossing motion is visible to the policy.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from robot_localization.srv import SetPose
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray, Int32, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


Goal = Tuple[float, float, float]


class NavigationGoalRejected(RuntimeError):
    """Nav2 action server explicitly refused a NavigateToPose goal."""


def quaternion_from_yaw(yaw: float) -> Tuple[float, float]:
    return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def yaw_from_quaternion(z: float, w: float) -> float:
    return 2.0 * math.atan2(z, w)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class AgvRosInterface(Node):
    """ROS backend connecting the front-sector observation to Nav2."""

    LIDAR_BEAMS = 24
    # Match the usable forward field of view of the fixed real LiDAR mount.
    LIDAR_ANGLE_MIN = -math.pi / 3.0
    LIDAR_ANGLE_MAX = math.pi / 3.0
    LIDAR_REPLACEMENT_RANGE = 10.0
    TTC_MAX = 5.0
    DYNAMIC_OBSTACLE_PARK = (20.0, 20.0, 0.5)

    def __init__(self, retain_dynamic_obstacle_after_crossing: bool = False) -> None:
        super().__init__(
            "gazebo_phase2_finetune_environment",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.lock = threading.RLock()
        self.scan_ready = threading.Event()
        self.odom_ready = threading.Event()
        self.clock_ready = threading.Event()
        self.front_safety_stop_ready = threading.Event()
        self.front_safety_diagnostics_ready = threading.Event()
        self.scan_seq = 0
        self.odom_seq = 0
        self.sim_time = 0.0
        self.lidar = np.full(
            self.LIDAR_BEAMS,
            self.LIDAR_REPLACEMENT_RANGE,
            dtype=np.float32,
        )
        self.front_scan_x = np.empty(0, dtype=np.float32)
        self.front_scan_y = np.empty(0, dtype=np.float32)
        self.front_scan_valid = False
        self.front_scan_time_s: Optional[float] = None
        self.sector_ttc = np.full(
            self.LIDAR_BEAMS,
            self.TTC_MAX,
            dtype=np.float32,
        )
        self.last_supervisor_status_time = None
        self.hazard_persistence = 0.0
        self.applied_speed_scale = 1.0
        self.safety_override_active = False
        self.front_safety_stop_active = False
        self.front_safety_stop_events = 0
        self.front_safety_stop_reason = 0
        self.front_safety_stop_clearance_m = math.inf
        self.front_safety_stop_distance_m = math.inf
        self.supervisor_min_ttc = math.inf
        self.minimum_sector_ttc_since_reset = self.TTC_MAX
        self.dynamic_obstacle_spawns_since_reset = 0
        self.dynamic_obstacle_trajectory = None
        self.dynamic_obstacle_position = self.DYNAMIC_OBSTACLE_PARK
        self.retain_dynamic_obstacle_after_crossing = bool(
            retain_dynamic_obstacle_after_crossing
        )
        self.dynamic_obstacle_parked_after_crossing = False
        self.applied_speed_scale_ready = threading.Event()
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        # Nav2 command before the simulation-only last-resort front gate.
        # Keep this as telemetry; it is not part of the policy observation.
        self.cmd_vx_m_s = 0.0
        self.cmd_wz_rad_s = 0.0
        self.cmd_vel_time_s: Optional[float] = None
        self.collision_since_reset = False
        self.collision_active = False
        self.collision_event_count = 0
        self.collision_events = []
        self.max_contact_count = 0
        self.minimum_lidar_since_reset = self.LIDAR_REPLACEMENT_RANGE
        self.minimum_lidar_since_step = self.LIDAR_REPLACEMENT_RANGE

        self.goal_handle = None
        self.goal_result_status: Optional[int] = None
        self.goal_result_future = None
        self.goal_token = 0

        # Lưu trữ Costmap
        self.costmap = None
        self.costmap_occupied_threshold = 50
        self.costmap_lethal_threshold = 99

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
            "/scan",
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
            Twist,
            "/cmd_vel",
            self._cmd_vel_callback,
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
            String,
            "/collision_event",
            self._collision_event_callback,
            20,
        )
        self.create_subscription(
            Bool,
            "/sim_front_safety_stop",
            self._front_safety_stop_callback,
            20,
        )
        self.create_subscription(
            Float32MultiArray,
            "/sim_front_safety_diagnostics",
            self._front_safety_diagnostics_callback,
            20,
        )
        self.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            self._costmap_callback,
            costmap_qos,
        )
        applied_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Float32MultiArray,
            "/rl_velocity_limits_applied",
            self._applied_speed_scale_callback,
            applied_qos,
        )
        self.speed_scale_publisher = self.create_publisher(
            Float32MultiArray,
            "/rl_velocity_limits",
            10,
        )
        self.dynamic_obstacle_publisher = self.create_publisher(
            Point, "/dynamic_obstacle/position", 10,
        )
        self.create_timer(0.05, self._dynamic_obstacle_tick)
        
        clock_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            Clock,
            "/clock",
            self._clock_callback,
            clock_qos,
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
        self.randomization_client = self.create_client(
            Trigger, "/resample_domain_randomization")
        self.randomization_sequence_reset_client = self.create_client(
            Trigger,
            "/reset_domain_randomization_sequence",
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
            with self.lock:
                self.front_scan_valid = False
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
            with self.lock:
                self.front_scan_valid = False
            return
        sectors = np.array_split(filtered, self.LIDAR_BEAMS)
        # A masked sector is unknown, not free space at maximum range.
        if any(np.all(np.isnan(sector)) for sector in sectors):
            with self.lock:
                self.front_scan_valid = False
            return
        # The v5 risk channels use the same raw front returns on simulator
        # and robot; they never consume simulation-only gate diagnostics.
        range_min_raw = float(msg.range_min)
        range_max_raw = float(msg.range_max)
        if (
            not math.isfinite(range_min_raw)
            or math.isnan(range_max_raw)
            or range_min_raw < 0.0
            or range_max_raw <= range_min_raw
        ):
            with self.lock:
                self.front_scan_valid = False
            return
        range_min = range_min_raw
        range_max = min(self.LIDAR_REPLACEMENT_RANGE, range_max_raw)
        valid_return = (
            np.isfinite(filtered)
            & (filtered >= range_min)
            & (filtered <= range_max)
        )
        front_x = filtered[valid_return] * np.cos(angles[mask][valid_return])
        front_y = filtered[valid_return] * np.sin(angles[mask][valid_return])
        filtered = np.where(
            (~np.isfinite(filtered))
            | (filtered < range_min)
            | (filtered > range_max),
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
            self.front_scan_x = front_x.astype(np.float32)
            self.front_scan_y = front_y.astype(np.float32)
            self.front_scan_valid = True
            self.front_scan_time_s = float(getattr(self, "sim_time", math.nan))
            self.minimum_lidar_since_reset = min(
                self.minimum_lidar_since_reset,
                float(np.min(reduced)),
            )
            self.minimum_lidar_since_step = min(
                self.minimum_lidar_since_step, float(np.min(reduced))
            )
            self.scan_seq += 1
            self.scan_ready.set()

    def _applied_speed_scale_callback(self, msg: Float32MultiArray) -> None:
        # The v4 contract requires the supervisor's confirmed 24-sector TTC.
        if len(msg.data) < 5 + self.LIDAR_BEAMS:
            return
        sector_ttc = np.asarray(
            msg.data[5:5 + self.LIDAR_BEAMS], dtype=np.float32
        )
        if (
            not math.isfinite(float(msg.data[0]))
            or not np.all(np.isfinite(sector_ttc))
        ):
            return
        with self.lock:
            self.applied_speed_scale = float(msg.data[0])
            self.safety_override_active = bool(msg.data[3])
            self.supervisor_min_ttc = float(msg.data[4])
            self.sector_ttc = np.clip(sector_ttc, 0.0, self.TTC_MAX)
            self.minimum_sector_ttc_since_reset = min(
                self.minimum_sector_ttc_since_reset,
                float(np.min(self.sector_ttc)),
            )
            now = self.get_clock().now().nanoseconds * 1.0e-9
            dt = (
                min(now - self.last_supervisor_status_time, 1.0)
                if self.last_supervisor_status_time is not None
                and now > self.last_supervisor_status_time
                else 0.0
            )
            self.hazard_persistence = (
                min(self.hazard_persistence + dt, 10.0)
                if self.safety_override_active
                else 0.0
            )
            self.last_supervisor_status_time = now
            self.applied_speed_scale_ready.set()

    def _odom_callback(self, msg: Odometry) -> None:
        with self.lock:
            self.linear_velocity = float(msg.twist.twist.linear.x)
            self.angular_velocity = float(msg.twist.twist.angular.z)
            self.odom_seq += 1
            self.odom_ready.set()

    def _cmd_vel_callback(self, msg: Twist) -> None:
        with self.lock:
            self.cmd_vx_m_s = float(msg.linear.x)
            self.cmd_wz_rad_s = float(msg.angular.z)
            self.cmd_vel_time_s = float(getattr(self, "sim_time", math.nan))

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

    def _collision_event_callback(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Invalid /collision_event: {exc}")
            return
        if not isinstance(event, dict):
            self.get_logger().warning("Invalid /collision_event: expected object")
            return
        with self.lock:
            self.collision_events.append(event)

    def _front_safety_stop_callback(self, msg: Bool) -> None:
        with self.lock:
            if msg.data and not self.front_safety_stop_active:
                self.front_safety_stop_events += 1
            self.front_safety_stop_active = bool(msg.data)
            self.front_safety_stop_ready.set()

    def _front_safety_diagnostics_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 6:
            with self.lock:
                self.front_safety_stop_clearance_m = float(msg.data[1])
                self.front_safety_stop_distance_m = float(msg.data[2])
                self.front_safety_stop_reason = int(msg.data[5])
            self.front_safety_diagnostics_ready.set()

    def _costmap_callback(self, msg: OccupancyGrid) -> None:
        with self.lock:
            self.costmap = msg

    def _clock_callback(self, msg: Clock) -> None:
        stamp = msg.clock
        sim_time = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        with self.lock:
            self.sim_time = sim_time
            self.clock_ready.set()

    def current_sim_time(self) -> float:
        with self.lock:
            return float(self.sim_time)

    def _publish_dynamic_obstacle(self, position: Tuple[float, float, float]) -> None:
        with self.lock:
            self.dynamic_obstacle_position = tuple(float(value) for value in position)
        self.dynamic_obstacle_publisher.publish(
            Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        )

    def park_dynamic_obstacle(self) -> None:
        with self.lock:
            self.dynamic_obstacle_trajectory = None
            self.dynamic_obstacle_parked_after_crossing = False
        self._publish_dynamic_obstacle(self.DYNAMIC_OBSTACLE_PARK)

    def start_dynamic_obstacle(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        speed_m_s: float,
    ) -> None:
        distance = math.dist(start, end)
        if not math.isfinite(distance) or not math.isfinite(speed_m_s) or speed_m_s <= 0.0:
            raise ValueError("Dynamic-obstacle trajectory must be finite with positive speed")
        with self.lock:
            self.dynamic_obstacle_trajectory = (
                tuple(start), tuple(end), self.sim_time, distance / speed_m_s,
            )
            self.dynamic_obstacle_parked_after_crossing = False
            self.dynamic_obstacle_spawns_since_reset += 1
        self._publish_dynamic_obstacle((start[0], start[1], 0.5))

    def _dynamic_obstacle_tick(self) -> None:
        with self.lock:
            trajectory = self.dynamic_obstacle_trajectory
            sim_time = self.sim_time
            if trajectory is None:
                return
            start, end, start_time, duration = trajectory
            fraction = (sim_time - start_time) / duration
            if fraction < 0.0 or fraction >= 1.0:
                self.dynamic_obstacle_trajectory = None
                self.dynamic_obstacle_parked_after_crossing = (
                    self.retain_dynamic_obstacle_after_crossing
                )
                position = (
                    (end[0], end[1], 0.5)
                    if self.retain_dynamic_obstacle_after_crossing
                    else self.DYNAMIC_OBSTACLE_PARK
                )
            else:
                position = (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                    0.5,
                )
        self._publish_dynamic_obstacle(position)

    def wait_for_sim_duration(
        self,
        duration: float,
        stall_timeout: float = 5.0,
    ) -> None:
        """Wait for a duration measured by MuJoCo's monotonic /clock."""
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("Simulation duration must be finite and positive")
        if not math.isfinite(stall_timeout) or stall_timeout <= 0.0:
            raise ValueError("Clock stall timeout must be finite and positive")
        if not self.clock_ready.wait(timeout=stall_timeout):
            raise TimeoutError("No simulation time received on /clock")

        start_sim_time = self.current_sim_time()
        target_sim_time = start_sim_time + duration
        latest_sim_time = start_sim_time
        last_progress_wall_time = time.monotonic()

        while rclpy.ok():
            sim_time = self.current_sim_time()
            if sim_time >= target_sim_time:
                return
            if sim_time < latest_sim_time:
                raise RuntimeError(
                    "/clock moved backwards during the action interval"
                )
            if sim_time > latest_sim_time:
                latest_sim_time = sim_time
                last_progress_wall_time = time.monotonic()
            elif time.monotonic() - last_progress_wall_time >= stall_timeout:
                raise TimeoutError(
                    "/clock stopped advancing during the action interval"
                )
            time.sleep(0.005)

        raise RuntimeError("ROS stopped during the action interval")

    def is_position_free(
        self,
        x: float,
        y: float,
        clearance_radius_m: float = 0.0,
    ) -> bool:
        """Check a point or circular footprint against the global costmap.

        Unknown cells and circles extending outside the costmap fail closed.
        The default radius preserves point checks used by obstacle spawning.
        """
        if not math.isfinite(clearance_radius_m) or clearance_radius_m < 0.0:
            raise ValueError("clearance_radius_m must be finite and non-negative")
        with self.lock:
            if self.costmap is None:
                return False

            info = self.costmap.info
            mx = int((x - info.origin.position.x) / info.resolution)
            my = int((y - info.origin.position.y) / info.resolution)

            if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
                return False

            # Occupancy values represent square cells. Test exact circle/AABB
            # intersection so a candidate near a cell boundary cannot skip a
            # neighboring lethal or unknown cell.
            center_value = self.costmap.data[my * info.width + mx]
            if center_value < 0 or center_value >= self.costmap_occupied_threshold:
                return False
            if clearance_radius_m == 0.0:
                return True

            origin_x = info.origin.position.x
            origin_y = info.origin.position.y
            min_cell_x = math.floor(
                (x - clearance_radius_m - origin_x) / info.resolution
            )
            max_cell_x = math.floor(
                (x + clearance_radius_m - origin_x) / info.resolution
            )
            min_cell_y = math.floor(
                (y - clearance_radius_m - origin_y) / info.resolution
            )
            max_cell_y = math.floor(
                (y + clearance_radius_m - origin_y) / info.resolution
            )
            if (
                min_cell_x < 0 or min_cell_y < 0
                or max_cell_x >= info.width or max_cell_y >= info.height
            ):
                return False
            lethal_threshold = getattr(self, "costmap_lethal_threshold", 99)
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cell_bottom = origin_y + cell_y * info.resolution
                cell_top = cell_bottom + info.resolution
                closest_y = min(max(y, cell_bottom), cell_top)
                for cell_x in range(min_cell_x, max_cell_x + 1):
                    cell_left = origin_x + cell_x * info.resolution
                    cell_right = cell_left + info.resolution
                    closest_x = min(max(x, cell_left), cell_right)
                    if math.hypot(x - closest_x, y - closest_y) > clearance_radius_m:
                        continue
                    value = self.costmap.data[cell_y * info.width + cell_x]
                    if value < 0 or value >= lethal_threshold:
                        return False
            return True

    def current_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "lidar": self.lidar.copy(),
                "front_scan_x": self.front_scan_x.copy(),
                "front_scan_y": self.front_scan_y.copy(),
                "front_scan_valid": bool(self.front_scan_valid),
                "front_scan_age_s": (
                    max(0.0, float(self.sim_time - self.front_scan_time_s))
                    if self.front_scan_time_s is not None
                    and self.sim_time >= self.front_scan_time_s
                    else math.inf
                ),
                "sector_ttc": self.sector_ttc.copy(),
                "hazard_persistence": float(self.hazard_persistence),
                "applied_speed_scale": float(self.applied_speed_scale),
                "safety_override_active": bool(self.safety_override_active),
                "front_safety_stop_active": bool(self.front_safety_stop_active),
                "front_safety_stop_events": int(self.front_safety_stop_events),
                "front_safety_stop_reason": int(self.front_safety_stop_reason),
                "front_safety_stop_clearance_m": float(self.front_safety_stop_clearance_m),
                "front_safety_stop_distance_m": float(self.front_safety_stop_distance_m),
                "supervisor_min_ttc": float(self.supervisor_min_ttc),
                "min_sector_ttc_episode": float(self.minimum_sector_ttc_since_reset),
                "dynamic_obstacle_spawns": int(self.dynamic_obstacle_spawns_since_reset),
                "dynamic_obstacle_active": self.dynamic_obstacle_trajectory is not None,
                "dynamic_obstacle_position": list(self.dynamic_obstacle_position),
                "dynamic_obstacle_parked_after_crossing": bool(
                    self.dynamic_obstacle_parked_after_crossing
                ),
                "scan_seq": int(self.scan_seq),
                "odom_seq": int(self.odom_seq),
                "linear_velocity": float(self.linear_velocity),
                "angular_velocity": float(self.angular_velocity),
                "cmd_vx_m_s": float(self.cmd_vx_m_s),
                "cmd_wz_rad_s": float(self.cmd_wz_rad_s),
                "cmd_vel_age_s": (
                    max(0.0, float(self.sim_time - self.cmd_vel_time_s))
                    if self.cmd_vel_time_s is not None
                    and self.sim_time >= self.cmd_vel_time_s
                    else math.inf
                ),
                "collision": bool(self.collision_since_reset),
                "collision_event_count": int(self.collision_event_count),
                "collision_events": list(self.collision_events),
                "contact_count": int(self.max_contact_count),
                "minimum_lidar": float(self.minimum_lidar_since_reset),
                "minimum_lidar_step": float(self.minimum_lidar_since_step),
                "nav_status": self.goal_result_status,
            }

    def begin_action_interval(self) -> None:
        with self.lock:
            self.minimum_lidar_since_step = float(np.min(self.lidar))

    def reset_episode_flags(self) -> None:
        with self.lock:
            self.collision_since_reset = False
            self.collision_active = False
            self.collision_event_count = 0
            self.collision_events.clear()
            self.front_safety_stop_active = False
            self.front_safety_stop_events = 0
            self.front_safety_stop_reason = 0
            self.front_safety_stop_clearance_m = math.inf
            self.front_safety_stop_distance_m = math.inf
            self.max_contact_count = 0
            self.minimum_lidar_since_reset = self.LIDAR_REPLACEMENT_RANGE
            self.minimum_lidar_since_step = self.LIDAR_REPLACEMENT_RANGE
            self.goal_result_status = None
            self.last_supervisor_status_time = None
            self.sector_ttc.fill(self.TTC_MAX)
            self.minimum_sector_ttc_since_reset = self.TTC_MAX
            self.dynamic_obstacle_spawns_since_reset = 0
            self.hazard_persistence = 0.0

    def publish_speed_scale(self, scale: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(scale)]
        self.speed_scale_publisher.publish(msg)

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

    def wait_until_ready(
        self, timeout: float = 20.0, require_supervisor: bool = True
    ) -> None:
        deadline = time.monotonic() + timeout
        clients = (
            (self.randomization_client, "/resample_domain_randomization"),
            (
                self.randomization_sequence_reset_client,
                "/reset_domain_randomization_sequence",
            ),
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
            raise RuntimeError("No LaserScan received on /scan")
        remaining = max(0.0, deadline - time.monotonic())
        if not self.odom_ready.wait(timeout=remaining):
            raise RuntimeError(
                "No filtered odometry received on /odometry/filtered"
            )
        remaining = max(0.0, deadline - time.monotonic())
        if not self.clock_ready.wait(timeout=remaining):
            raise RuntimeError("No simulation time received on /clock")
        if not require_supervisor:
            return
        remaining = max(0.0, deadline - time.monotonic())
        while (
            self.speed_scale_publisher.get_subscription_count() < 1
            and remaining > 0.0
        ):
            time.sleep(0.05)
            remaining = max(0.0, deadline - time.monotonic())
        if self.speed_scale_publisher.get_subscription_count() < 1:
            raise RuntimeError("No RL supervisor subscribed to /rl_velocity_limits")
        remaining = max(0.0, deadline - time.monotonic())
        if not self.applied_speed_scale_ready.wait(timeout=remaining):
            raise RuntimeError(
                "No supervisor status received on /rl_velocity_limits_applied"
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
            raise NavigationGoalRejected(
                f"NavigateToPose rejected goal ({x:.2f}, {y:.2f}, "
                f"{math.degrees(yaw):.1f} deg)"
            )
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

class AgvRlEnv(gym.Env):
    """Train a policy that only selects the Nav2 velocity-envelope scale."""

    metadata = {"render_modes": []}
    # Timing and reward semantics are part of the contract even though the
    # tensor shapes are unchanged.  V4 uses the supervisor's rotation-aligned
    # TTC sectors, so v3 replay and checkpoints have different semantics.
    CONTRACT_VERSION = "speedlimit-v5-front120-56x1-1hz-proactive-risk"
    CONTRACT_VERSION_NO_TTC = "speedlimit-v5-front120-31x1-1hz-no-ttc"
    # Model weights remain transferable; replay rewards must match this tag.
    REWARD_CONTRACT_VERSION = "front-corridor-proactive-risk-v2"
    REWARD_CONTRACT_VERSION_NO_TTC = "front-corridor-sac-only-v1"
    REWARD_CONTRACT_VERSION_CONSERVATIVE = "front-lateral-sac-only-v1"
    REWARD_CONTRACT_VERSION_CONSERVATIVE_RETAINED = (
        "front-lateral-sac-only-v4-classified-stall-reset"
    )

    ACTION_LOW = np.asarray([0.15], dtype=np.float32)
    ACTION_HIGH = np.asarray([1.0], dtype=np.float32)
    SOURCE_NOMINAL_ACTION = np.asarray([1.0], dtype=np.float32)

    # KHÔI PHỤC COMBINED_GOALS để train_sac.py có thể đọc được
    COMBINED_GOALS: Tuple[Goal, ...] = (
        (9.5, 0.0, 0.0),
        (9.5, 5.0, 0.0),
        (9.5, -5.0, 0.0),
    )

    MAX_SCAN_RANGE = 10.0
    MAX_EXPECTED_GOAL_DISTANCE = 20.0
    STEP_WAIT_DURATION = 1.0
    DENSE_REWARD_REFERENCE_SECONDS = 5.05
    SAFE_DISTANCE = 0.60
    COLLISION_DISTANCE = 0.25
    SAFETY_REWARD_CLIP = -20.0
    COLLISION_REWARD = -50.0
    GOAL_REWARD = 100.0
    NAV_ABORT_REWARD = -30.0
    FRONT_STOP_REQUEST_PENALTY_WEIGHT = 5.0
    FRONT_RISK_PENALTY_WEIGHT = 1.0
    FRONT_RISK_LOOKAHEAD_M = 0.40
    FRONT_RISK_MARGIN_NORM_M = 2.0
    FRONT_RISK_NOMINAL_VX_M_S = 0.70
    FRONT_RISK_CORRIDOR_HALF_WIDTH_M = 0.45
    FRONT_RISK_REACTION_S = 0.40
    FRONT_RISK_CLEARANCE_M = 0.40
    FRONT_RISK_DECELERATION_M_S2 = 0.80
    FRONT_RISK_RELEASE_RESERVE_M = 0.15
    FRONT_RISK_SCAN_TIMEOUT_S = 0.25
    CONSERVATIVE_LATERAL_REWARD_WEIGHT = 20.0
    CONSERVATIVE_LATERAL_MIN_X_M = 0.75
    CONSERVATIVE_LATERAL_MAX_X_M = 3.0
    CONSERVATIVE_LATERAL_MIN_Y_M = 0.65
    CONSERVATIVE_LATERAL_MAX_Y_M = 1.75
    # Robot half-diagonal is about 0.56 m including Nav2 padding. Keep random
    # goal centres farther away so the terminal pose has room to rotate.
    GOAL_CLEARANCE_M = 0.80
    SAFETY_DEADLOCK_STEPS = 20
    SAFETY_DEADLOCK_MIN_BLOCKED_STEPS = 14
    SAFETY_DEADLOCK_MAX_DISPLACEMENT_M = 0.10
    SAFETY_DEADLOCK_MAX_YAW_RAD = math.radians(15.0)
    MOTION_STALL_STEPS = 30
    MOTION_STALL_MAX_DISPLACEMENT_M = 0.20

    # Previous 54 channels retain their order. The final two are computed
    # solely from front LaserScan, odometry and applied SpeedLimit, all of
    # which are available through the same path during training and deploy.
    OBSERVATION_DIM = 56
    OBSERVATION_DIM_NO_TTC = 31
    MAX_HAZARD_PERSISTENCE = 5.0
    # Normalization ceilings leave headroom for transient odometry overshoot.
    MAX_EXPECTED_LINEAR_VELOCITY = 1.2
    MAX_EXPECTED_ANGULAR_VELOCITY = 1.0

    # Penalize abrupt changes in the scale actually applied by the supervisor.
    ACTION_SMOOTHNESS_PENALTY_WEIGHT = 0.1

    # [PATCH] Nếu True: chỉ tính episode là "thành công" khi đã tới >=1 goal
    # VÀ episode không kết thúc do va chạm. Nếu False: chỉ cần đã từng tới
    # goal là tính thành công (kể cả nếu sau đó va chạm). Xem giải thích ở
    # cuối phương thức step().
    SUCCESS_REQUIRES_NO_COLLISION = True

    # [PATCH] Số lần thử lại tối đa cho quy trình hard reset sau va chạm,
    # và thời gian chờ giữa các lần thử. Trước đây nếu BẤT KỲ bước nào
    # (reset_simulation/reset_ekf/wait_for_reset_state...) timeout, lỗi sẽ
    # bay thẳng lên và làm CRASH TOÀN BỘ script training - không hề quay
    # lại (0,0,0) như mong đợi, mà dừng hẳn không chạy tiếp.
    HARD_RESET_MAX_ATTEMPTS = 5
    HARD_RESET_RETRY_DELAY = 2.0

    def __init__(
        self,
        goal: Goal = (9.5, 0.0, 0.0),
        goals: Optional[Sequence[Goal]] = None,
        goal_sampling: str = "random_costmap",
        max_episode_steps: int = 180,
        dynamic_obstacles: bool = False,
        dynamic_obstacle_probability: float = 0.75,
        require_front_safety_stop: bool = False,
        include_ttc: bool = True,
        conservative_crossing: bool = False,
        retain_dynamic_obstacle_after_crossing: bool = False,
    ) -> None:
        super().__init__()

        # RNG độc lập chỉ dùng cho việc sinh random goal.
        # Không chịu ảnh hưởng bởi seed của Gymnasium/SB3.
        self._goal_rng = np.random.default_rng()

        # Giữ nguyên khung kiểm tra đối số gốc để tương thích với script ngoài
        # Legacy 'random' means random selection from the supplied list.
        if goal_sampling == "random":
            goal_sampling = "random_list"
        if goal_sampling not in ("fixed", "cycle", "random_list", "random_costmap"):
            raise ValueError(
                "goal_sampling must be fixed, cycle, random_list or random_costmap"
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not 0.0 <= dynamic_obstacle_probability <= 1.0:
            raise ValueError("dynamic_obstacle_probability must be in [0, 1]")

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
        if retain_dynamic_obstacle_after_crossing and not conservative_crossing:
            raise ValueError(
                "retained crossing obstacle requires conservative_crossing"
            )
        self.ros = AgvRosInterface(
            retain_dynamic_obstacle_after_crossing=retain_dynamic_obstacle_after_crossing
        )
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
        self.dynamic_obstacles = bool(dynamic_obstacles)
        self.dynamic_obstacle_probability = float(dynamic_obstacle_probability)
        self.include_ttc = bool(include_ttc)
        self.conservative_crossing = bool(conservative_crossing)
        self.retain_dynamic_obstacle_after_crossing = bool(
            retain_dynamic_obstacle_after_crossing
        )
        # This experimental arm keeps the obstacle after crossing, so a
        # no-progress episode must not soft-reset at the same blocked pose.
        self.enable_motion_stall_recovery = self.retain_dynamic_obstacle_after_crossing
        if self.conservative_crossing and (self.include_ttc or not self.dynamic_obstacles):
            raise ValueError("conservative_crossing requires no-TTC dynamic-obstacle training")
        self.contract_version = (
            self.CONTRACT_VERSION if self.include_ttc else self.CONTRACT_VERSION_NO_TTC
        )
        if self.retain_dynamic_obstacle_after_crossing:
            self.reward_contract_version = (
                self.REWARD_CONTRACT_VERSION_CONSERVATIVE_RETAINED
            )
        elif self.conservative_crossing:
            self.reward_contract_version = self.REWARD_CONTRACT_VERSION_CONSERVATIVE
        elif self.include_ttc:
            self.reward_contract_version = self.REWARD_CONTRACT_VERSION
        else:
            self.reward_contract_version = self.REWARD_CONTRACT_VERSION_NO_TTC
        self.observation_dim = (
            self.OBSERVATION_DIM if self.include_ttc else self.OBSERVATION_DIM_NO_TTC
        )
        self.next_dynamic_obstacle_step = 1
        self.safety_deadlock_steps = 0
        self.safety_deadlock_history = deque(maxlen=self.SAFETY_DEADLOCK_STEPS)
        self.motion_stall_history = deque(maxlen=self.MOTION_STALL_STEPS)
        self.previous_distance = 0.0
        self.closed = False

        # [GROUP2] Action ĐÃ ÁP DỤNG ở step trước — dùng để tính phạt thay
        # đổi đột ngột. None nghĩa là "chưa có step nào" (đầu episode).
        self.previous_applied_action: Optional[np.ndarray] = None

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
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        try:
            self.ros.wait_until_ready()
            if require_front_safety_stop and not self.ros.front_safety_stop_ready.wait(timeout=5.0):
                raise RuntimeError(
                    "No /sim_front_safety_stop status: restart the Phase-2 "
                    "MuJoCo launch with enable_rl_supervisor:=true"
                )
            if require_front_safety_stop and not self.ros.front_safety_diagnostics_ready.wait(timeout=5.0):
                raise RuntimeError(
                    "No current /sim_front_safety_diagnostics status: restart "
                    "the Phase-2 MuJoCo launch after rebuilding"
                )
        except Exception:
            self.close()
            raise

    def _select_next_goal(self) -> None:
        """Select a goal consistently on reset, success and abort."""
        if self.goal_sampling == "random_costmap":
            self._generate_new_goal()
        elif self.goal_sampling == "fixed":
            self.goal = self.goals[0]
        elif self.goal_sampling == "cycle":
            self.goal = self.goals[self.goal_cursor]
            self.goal_cursor = (self.goal_cursor + 1) % len(self.goals)
        else:  # random_list
            index = int(self.np_random.integers(len(self.goals)))
            self.goal = self.goals[index]

    def _send_selected_goal(self) -> None:
        """Retry explicit Nav2 rejections without hiding transport/timeouts."""
        attempts = 5
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                self.ros.send_navigation_goal(*self.goal)
                return
            except NavigationGoalRejected as exc:
                last_error = exc
                print(
                    f"[Goal] Nav2 từ chối lần {attempt}/{attempts}: {exc}",
                    flush=True,
                )
                if attempt == attempts:
                    break
                time.sleep(0.5)
                if self.goal_sampling in ("random_costmap", "random_list"):
                    self._select_next_goal()
        raise NavigationGoalRejected(
            f"Nav2 rejected {attempts} consecutive goals; check bt_navigator "
            "lifecycle and the sampled goal locations"
        ) from last_error

    def _maybe_start_training_obstacle(self) -> None:
        """Place a lateral crossing far enough ahead to permit slowing/avoidance.

        A head-on diagonal approach proved unsuitable for the 1D speed-limit
        policy: the obstacle could enter the robot footprint even at minimum
        commanded speed. Keep longitudinal position fixed in this first
        dynamic curriculum and let Nav2 steer around the crossing.
        """
        if not self.dynamic_obstacles or self.episode_step_count < self.next_dynamic_obstacle_step:
            return
        self.next_dynamic_obstacle_step = self.episode_step_count + 20
        if (
            self.ros.current_state()["dynamic_obstacle_active"]
            or getattr(self.ros, "dynamic_obstacle_parked_after_crossing", False)
        ):
            return
        if self.np_random.random() >= self.dynamic_obstacle_probability:
            return
        try:
            robot_x, robot_y, yaw = self.ros.get_map_pose()
        except Exception as exc:
            print(f"[Dynamic obstacle] No robot pose: {exc!r}", flush=True)
            return
        cosine, sine = math.cos(yaw), math.sin(yaw)

        def to_world(forward: float, lateral: float) -> Tuple[float, float]:
            return (
                robot_x + cosine * forward - sine * lateral,
                robot_y + sine * forward + cosine * lateral,
            )

        for _ in range(16):
            side = -1.0 if self.np_random.random() < 0.5 else 1.0
            if getattr(self, "conservative_crossing", False):
                forward = float(self.np_random.uniform(1.7, 2.8))
                lateral = float(self.np_random.uniform(1.0, 1.5))
                speed = float(self.np_random.uniform(0.25, 0.55))
            else:
                forward = float(self.np_random.uniform(3.2, 3.8))
                lateral = 1.5
                speed = float(self.np_random.uniform(0.45, 0.60))
            start = to_world(forward, side * lateral)
            end = to_world(forward, -side * lateral)
            # Avoid teleporting the obstacle into known static geometry.
            if not all(
                self.ros.is_position_free(
                    start[0] + alpha * (end[0] - start[0]),
                    start[1] + alpha * (end[1] - start[1]),
                )
                for alpha in np.linspace(0.0, 1.0, 7)
            ):
                continue
            self.ros.start_dynamic_obstacle(start, end, speed)
            print(
                f"[Dynamic obstacle] start={start} end={end} speed={speed:.2f} m/s",
                flush=True,
            )
            return
        self.next_dynamic_obstacle_step = self.episode_step_count + 5
        print("[Dynamic obstacle] No free front crossing; retrying later", flush=True)

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
                cand_x = float(self._goal_rng.uniform(origin_x, origin_x + width_m))
                cand_y = float(self._goal_rng.uniform(origin_y, origin_y + height_m))

                if math.hypot(cand_x - robot_x, cand_y - robot_y) < self.min_goal_distance:
                    continue

                if self.ros.is_position_free(
                    cand_x, cand_y, self.GOAL_CLEARANCE_M
                ):
                    self.goal = (cand_x, cand_y, 0.0)
                    print(f"[Goal] Đã sinh đích toàn map thành công: ({cand_x:.2f}, {cand_y:.2f})")
                    return

        print("[Goal] Không tìm được điểm toàn map, dùng fallback bán kính xung quanh robot.")
        for r_max in (10.0, 6.0, 3.0, 1.5):
            for _ in range(self.max_goal_generation_attempts):
                dx = float(self._goal_rng.uniform(-r_max, r_max))
                dy = float(self._goal_rng.uniform(-r_max, r_max))
                if math.hypot(dx, dy) < self.min_goal_distance:
                    continue

                cand_x = robot_x + dx
                cand_y = robot_y + dy
                if self.ros.is_position_free(
                    cand_x, cand_y, self.GOAL_CLEARANCE_M
                ):
                    self.goal = (cand_x, cand_y, 0.0)
                    return

        for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            cand_x = float(robot_x + self.min_goal_distance * math.cos(ang))
            cand_y = float(robot_y + self.min_goal_distance * math.sin(ang))
            if self.ros.is_position_free(
                cand_x, cand_y, self.GOAL_CLEARANCE_M
            ):
                self.goal = (cand_x, cand_y, 0.0)
                return

        raise RuntimeError(
            "Không tìm được goal có đủ khoảng hở trên global costmap; "
            "giữ nguyên goal cũ và dừng fail-closed"
        )

    def _pose_and_distance(self) -> Tuple[float, float, float, float, float]:
        x, y, yaw = self.ros.get_map_pose()
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        distance = math.hypot(dx, dy)
        global_bearing = math.atan2(dy, dx)
        # [GROUP1] Đổi từ bearing TUYỆT ĐỐI (theo trục X của map) sang
        # bearing TƯƠNG ĐỐI so với hướng mũi robot (yaw) — quan sát
        # "egocentric" giúp policy tổng quát hoá tốt hơn nhiều: cùng 1
        # tình huống hình học (goal ở bên trái/phải/phía trước robot) sẽ
        # luôn cho ra cùng 1 giá trị quan sát, bất kể robot đang quay mặt
        # hướng nào trên bản đồ tuyệt đối. Trước đây (bearing tuyệt đối),
        # CÙNG một tình huống tương đối nhưng robot đứng quay hướng khác
        # sẽ cho ra observation hoàn toàn khác nhau, buộc policy phải học
        # lại quan hệ này riêng cho từng hướng quay có thể -> tốn dữ liệu
        # hơn nhiều, khó tổng quát hoá.
        relative_bearing = normalize_angle(global_bearing - yaw)
        return x, y, yaw, distance, relative_bearing

    @classmethod
    def _front_risk_features(cls, state: Dict[str, Any]) -> Tuple[float, float, float]:
        """Return clearance, signed braking margin and proactive risk.

        This is a conservative *proxy*, not the state of the simulation-only
        front gate. Invalid/unknown front scans fail closed. The reserve is
        always included so risk appears before the physical gate latches.
        """
        if (
            not state["front_scan_valid"]
            or not math.isfinite(float(state.get("front_scan_age_s", math.inf)))
            or float(state.get("front_scan_age_s", math.inf))
            > cls.FRONT_RISK_SCAN_TIMEOUT_S
        ):
            return 0.0, -cls.FRONT_RISK_MARGIN_NORM_M, 1.0
        x = np.asarray(state["front_scan_x"], dtype=np.float32)
        y = np.asarray(state["front_scan_y"], dtype=np.float32)
        if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return 0.0, -cls.FRONT_RISK_MARGIN_NORM_M, 1.0
        measured_vx = float(state["linear_velocity"])
        applied_scale = float(state["applied_speed_scale"])
        if not math.isfinite(measured_vx) or not math.isfinite(applied_scale):
            return 0.0, -cls.FRONT_RISK_MARGIN_NORM_M, 1.0
        v_ref = max(
            0.0,
            measured_vx,
            cls.FRONT_RISK_NOMINAL_VX_M_S * float(np.clip(applied_scale, 0.15, 1.0)),
        )
        half_width = cls.FRONT_RISK_CORRIDOR_HALF_WIDTH_M + cls.FRONT_RISK_REACTION_S * v_ref
        corridor = (x > 0.0) & (np.abs(y) <= half_width)
        clearance = float(np.min(x[corridor])) if np.any(corridor) else cls.MAX_SCAN_RANGE
        brake_distance = (
            cls.FRONT_RISK_CLEARANCE_M
            + cls.FRONT_RISK_REACTION_S * v_ref
            + v_ref * v_ref / (2.0 * cls.FRONT_RISK_DECELERATION_M_S2)
            + cls.FRONT_RISK_RELEASE_RESERVE_M
        )
        margin = clearance - brake_distance
        risk = float(np.clip(
            (cls.FRONT_RISK_LOOKAHEAD_M - margin) / cls.FRONT_RISK_LOOKAHEAD_M,
            0.0, 1.0,
        ))
        return clearance, margin, risk

    @classmethod
    def _conservative_lateral_risk(cls, state: Dict[str, Any]) -> float:
        """Current-scan risk for an object ahead and to either side.

        This deliberately does not infer motion: a static object in the same
        location receives the same risk. The benchmark measures that cost.
        """
        scan_age = float(state.get("front_scan_age_s", math.inf))
        if (
            not state["front_scan_valid"]
            or not math.isfinite(scan_age)
            or scan_age > cls.FRONT_RISK_SCAN_TIMEOUT_S
        ):
            return 0.0
        x = np.asarray(state["front_scan_x"], dtype=np.float32)
        y = np.asarray(state["front_scan_y"], dtype=np.float32)
        if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return 0.0
        mask = (
            (x >= cls.CONSERVATIVE_LATERAL_MIN_X_M)
            & (x <= cls.CONSERVATIVE_LATERAL_MAX_X_M)
            & (np.abs(y) >= cls.CONSERVATIVE_LATERAL_MIN_Y_M)
            & (np.abs(y) <= cls.CONSERVATIVE_LATERAL_MAX_Y_M)
        )
        if np.count_nonzero(mask) < 3:
            return 0.0
        nearest_x = float(np.min(x[mask]))
        return float(np.clip(
            (cls.CONSERVATIVE_LATERAL_MAX_X_M - nearest_x)
            / (cls.CONSERVATIVE_LATERAL_MAX_X_M - cls.CONSERVATIVE_LATERAL_MIN_X_M),
            0.0, 1.0,
        ))

    def _observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state = self.ros.current_state()
        front_clearance, front_margin, front_risk = self._front_risk_features(state)
        lateral_risk = (
            self._conservative_lateral_risk(state)
            if getattr(self, "conservative_crossing", False) else 0.0
        )
        x, y, yaw, distance, relative_bearing = self._pose_and_distance()
        scan_normalized = np.clip(
            state["lidar"] / self.MAX_SCAN_RANGE,
            0.0,
            1.0,
        )
        include_ttc = getattr(self, "include_ttc", True)
        ttc_normalized = (
            np.clip(state["sector_ttc"] / self.ros.TTC_MAX, 0.0, 1.0)
            if include_ttc else np.empty(0, dtype=np.float32)
        )

        # Measured velocity is required to interpret closing risk and braking.
        linear_velocity_norm = float(
            np.clip(
                state["linear_velocity"] / self.MAX_EXPECTED_LINEAR_VELOCITY,
                -1.0,
                1.0,
            )
        )
        angular_velocity_norm = float(
            np.clip(
                state["angular_velocity"] / self.MAX_EXPECTED_ANGULAR_VELOCITY,
                -1.0,
                1.0,
            )
        )

        observation = np.concatenate(
            (
                scan_normalized,
                ttc_normalized,
                np.asarray(
                    [
                        np.clip(
                            distance / self.MAX_EXPECTED_GOAL_DISTANCE,
                            0.0,
                            1.0,
                        ),
                        np.clip(
                            relative_bearing / math.pi,
                            -1.0,
                            1.0,
                        ),
                        linear_velocity_norm,
                        angular_velocity_norm,
                        np.clip(state["applied_speed_scale"], 0.0, 1.0),
                        *(
                            [np.clip(
                                state["hazard_persistence"]
                                / self.MAX_HAZARD_PERSISTENCE,
                                0.0,
                                1.0,
                            )] if include_ttc else []
                        ),
                        np.clip(front_clearance / self.MAX_SCAN_RANGE, 0.0, 1.0),
                        np.clip(front_margin / self.FRONT_RISK_MARGIN_NORM_M, -1.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)
        expected_dim = (
            self.OBSERVATION_DIM if include_ttc else self.OBSERVATION_DIM_NO_TTC
        )
        if observation.shape != (expected_dim,) or not np.all(
            np.isfinite(observation)
        ):
            raise RuntimeError(
                f"Invalid Phase-2 observation: shape={observation.shape}"
            )
        info = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "goal": list(self.goal),
            "distance_to_goal": distance,
            # [GROUP1] đổi tên field cho đúng bản chất (trước là
            # global_goal_bearing, tuyệt đối theo map)
            "relative_goal_bearing": relative_bearing,
            "linear_velocity": state["linear_velocity"],
            "angular_velocity": state["angular_velocity"],
            "cmd_vx_m_s": state.get("cmd_vx_m_s", math.nan),
            "cmd_wz_rad_s": state.get("cmd_wz_rad_s", math.nan),
            "cmd_vel_age_s": state.get("cmd_vel_age_s", math.inf),
            "applied_speed_scale": state["applied_speed_scale"],
            "safety_override_active": state["safety_override_active"],
            "front_safety_stop_active": state["front_safety_stop_active"],
            "front_safety_stop_events": state["front_safety_stop_events"],
            "front_safety_stop_reason": state["front_safety_stop_reason"],
            "front_safety_stop_clearance_m": state["front_safety_stop_clearance_m"],
            "front_safety_stop_distance_m": state["front_safety_stop_distance_m"],
            "supervisor_min_ttc": state["supervisor_min_ttc"],
            "min_sector_ttc": float(np.min(state["sector_ttc"])),
            "min_sector_ttc_episode": state["min_sector_ttc_episode"],
            "dynamic_obstacle_spawns": state["dynamic_obstacle_spawns"],
            "dynamic_obstacle_active": state["dynamic_obstacle_active"],
            "dynamic_obstacle_position": state.get(
                "dynamic_obstacle_position", [20.0, 20.0, 0.5]
            ),
            "dynamic_obstacle_parked_after_crossing": state.get(
                "dynamic_obstacle_parked_after_crossing", False
            ),
            "hazard_persistence": state["hazard_persistence"],
            "min_lidar": float(np.min(state["lidar"])),
            "front_corridor_clearance_m": front_clearance,
            "conservative_lateral_risk": lateral_risk,
            "front_braking_margin_m": front_margin,
            "front_proactive_risk": front_risk,
            "front_scan_valid": bool(state["front_scan_valid"]),
            "min_lidar_episode": float(state["minimum_lidar"]),
            "min_lidar_step": float(state["minimum_lidar_step"]),
            "contact_count": int(state["contact_count"]),
            "collision_state": bool(state["collision"]),
            "collision_event_count": int(state["collision_event_count"]),
            "collision_event_detail_count": len(state["collision_events"]),
            "collision_events": state["collision_events"],
            "nav_status": state["nav_status"],
            "goals_reached_this_episode": self.goals_reached_this_episode,
        }
        return observation, info

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (1,):
            raise ValueError(f"action must have shape (1,), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("action contains NaN or Inf")
        clipped = np.clip(array, self.ACTION_LOW, self.ACTION_HIGH)
        requested = np.asarray([float(clipped[0])], dtype=np.float32)
        self.ros.publish_speed_scale(float(requested[0]))
        return requested

    def _perform_hard_reset(self) -> None:
        """[PATCH] Thực hiện chuỗi hard-reset (về 0,0,0) với cơ chế thử lại.

        Trước đây, nếu bất kỳ bước nào trong reset_simulation/reset_ekf/
        reset_amcl/wait_for_reset_state bị timeout (rất dễ xảy ra khi CPU
        tải cao hoặc MuJoCo/EKF chưa kịp ổn định), exception sẽ bay thẳng
        lên trên và làm CRASH TOÀN BỘ script training - dừng hẳn, không hề
        quay lại (0,0,0) như mong đợi. Giờ đây, nếu 1 lần thử thất bại, sẽ
        tự động thử lại tối đa HARD_RESET_MAX_ATTEMPTS lần trước khi thực
        sự báo lỗi.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.HARD_RESET_MAX_ATTEMPTS + 1):
            try:
                print(
                    f"\n[Môi trường] Khởi tạo hoặc Va chạm: Dịch chuyển robot "
                    f"về (0,0,0)... (lần thử {attempt}/{self.HARD_RESET_MAX_ATTEMPTS})"
                )
                self.ros.reset_simulation()
                self.ros.reset_ekf()
                self.ros.reset_amcl()
                barrier = self.ros.current_state()
                self.ros.wait_for_reset_state(
                    old_scan_seq=barrier["scan_seq"],
                    old_odom_seq=barrier["odom_seq"],
                )
                print(f"[Môi trường] Hard reset THÀNH CÔNG (lần thử {attempt}).")
                return
            except Exception as exc:  # noqa: BLE001 - cố tình bắt rộng để không crash
                last_exc = exc
                print(
                    f"[Môi trường] Hard reset THẤT BẠI ở lần thử {attempt}: "
                    f"{exc!r}"
                )
                if attempt < self.HARD_RESET_MAX_ATTEMPTS:
                    print(
                        f"[Môi trường] Chờ {self.HARD_RESET_RETRY_DELAY}s rồi "
                        f"thử lại toàn bộ chuỗi reset..."
                    )
                    time.sleep(self.HARD_RESET_RETRY_DELAY)

        # Hết số lần thử cho phép - lúc này mới thực sự báo lỗi, kèm thông
        # tin rõ ràng để dễ chẩn đoán (thay vì traceback mơ hồ như trước).
        raise RuntimeError(
            f"Hard reset thất bại sau {self.HARD_RESET_MAX_ATTEMPTS} lần thử "
            f"liên tiếp. Lỗi cuối cùng: {last_exc!r}. Kiểm tra lại ROS2/MuJoCo "
            f"có đang chạy ổn định không (CPU quá tải, service không phản hồi...)."
        ) from last_exc

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        self.ros.park_dynamic_obstacle()
        self.ros.cancel_navigation()

        performed_hard_reset = self._needs_hard_reset
        if performed_hard_reset:
            # [PATCH] Gọi qua hàm có retry thay vì gọi trực tiếp inline.
            self._perform_hard_reset()
            self._needs_hard_reset = False
        else:
            print("\n[Môi trường] Hết giờ (Truncated): Giữ nguyên vị trí robot, chạy tiếp...")
            time.sleep(0.5)  # Đợi nhẹ 0.5s để cảm biến đồng bộ

        # Resample only at stationary episode boundaries, never on goal reach.
        deadline = time.monotonic() + 10.0
        stable = 0
        last_seq = -1
        while time.monotonic() < deadline:
            state = self.ros.current_state()
            if state["odom_seq"] != last_seq:
                last_seq = state["odom_seq"]
                stopped = (abs(state["linear_velocity"]) < 0.02
                           and abs(state["angular_velocity"]) < 0.03)
                stable = stable + 1 if stopped else 0
                if stable >= 3:
                    break
            time.sleep(0.01)
        else:
            raise TimeoutError("Robot did not stop before randomization")
        future = self.ros.randomization_client.call_async(Trigger.Request())
        response = self.ros._wait_future(future, 3.0, "domain randomization")
        if not response.success:
            raise RuntimeError(response.message)
        # Wait for sensor samples generated with the new parameters.
        barrier = self.ros.current_state()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            state = self.ros.current_state()
            if (state["scan_seq"] >= barrier["scan_seq"] + 2
                    and state["odom_seq"] >= barrier["odom_seq"] + 2):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError("No fresh sensors after randomization")
        self.ros.reset_episode_flags()
        self.ros.clear_costmaps(timeout=10.0)
        # A soft reset preserves pose, so forcing scale=1.0 near an obstacle
        # can recreate the latch that ended the previous episode. Only request
        # the nominal envelope after a hard reset to the known-clear origin.
        if performed_hard_reset:
            self._apply_action(self.SOURCE_NOMINAL_ACTION)
        # Smoothness is defined against the supervisor acknowledgement, not
        # the asynchronous request just published above.
        applied_scale = float(self.ros.current_state()["applied_speed_scale"])
        self.previous_applied_action = np.asarray(
            [applied_scale], dtype=np.float32
        )

        # Select according to the configured mode; default is whole-map sampling.
        self.goals_reached_this_episode = 0
        self._select_next_goal()
        self._send_selected_goal()

        observation, info = self._observation()
        self.previous_distance = float(info["distance_to_goal"])
        self.episode_step_count = 0
        self.next_dynamic_obstacle_step = 1
        self.safety_deadlock_steps = 0
        self.safety_deadlock_history.clear()
        if getattr(self, "enable_motion_stall_recovery", False):
            self.motion_stall_history.clear()
        info["reset_ok"] = True
        info["domain_randomization"] = response.message
        return observation, info

    def _update_safety_deadlock(self, info: Dict[str, Any], goal_event: Optional[str]) -> bool:
        """Detect near-zero motion with frequent corridor stops over 20 s."""
        if goal_event == "succeeded":
            self.safety_deadlock_history.clear()
            self.safety_deadlock_steps = 0
            return False
        blocked = (
            bool(info["front_safety_stop_active"])
            and (int(info["front_safety_stop_reason"]) & 2) != 0
        )
        self.safety_deadlock_history.append((
            float(info["x"]),
            float(info["y"]),
            float(info["yaw"]),
            blocked,
        ))
        history = self.safety_deadlock_history
        self.safety_deadlock_steps = sum(sample[3] for sample in history)
        if (
            not blocked
            or len(history) < self.SAFETY_DEADLOCK_STEPS
            or self.safety_deadlock_steps < self.SAFETY_DEADLOCK_MIN_BLOCKED_STEPS
        ):
            return False
        origin_x, origin_y, origin_yaw, _ = history[0]
        return all(
            math.hypot(x - origin_x, y - origin_y)
            <= self.SAFETY_DEADLOCK_MAX_DISPLACEMENT_M
            and abs(normalize_angle(yaw - origin_yaw))
            <= self.SAFETY_DEADLOCK_MAX_YAW_RAD
            for x, y, yaw, _ in history
        )

    def _update_motion_stall(self, info: Dict[str, Any], goal_event: Optional[str]) -> Optional[str]:
        """Classify no-translation episodes without conflating spin and stale cmd."""
        if not getattr(self, "enable_motion_stall_recovery", False):
            return None
        history = self.motion_stall_history
        if goal_event == "succeeded":
            history.clear()
            return None
        history.append((
            float(info["x"]), float(info["y"]), float(info["yaw"]),
            float(info.get("cmd_vel_age_s", 0.0)),
        ))
        if len(history) < self.MOTION_STALL_STEPS:
            return None
        origin_x, origin_y = history[0][:2]
        stationary = all(
            math.hypot(x - origin_x, y - origin_y)
            <= self.MOTION_STALL_MAX_DISPLACEMENT_M
            for x, y, _, _ in history
        )
        if not stationary:
            return None
        yaw_travel = sum(
            abs(normalize_angle(current[2] - previous[2]))
            for previous, current in zip(history, list(history)[1:])
        )
        if yaw_travel >= math.radians(30.0):
            return "rotation_stall"
        if all(sample[3] >= 1.0 for sample in list(history)[-3:]):
            return "command_stall"
        return "motion_stall"

    def step(self, action):
        self.episode_step_count += 1
        self._maybe_start_training_obstacle()
        self.ros.begin_action_interval()
        requested = self._apply_action(action)
        self.ros.wait_for_sim_duration(self.STEP_WAIT_DURATION)

        observation, info = self._observation()
        distance = float(info["distance_to_goal"])
        # Capture transient proximity within this action interval only.
        # An old noisy reading must not penalize every subsequent step.
        minimum_obstacle_distance = float(info["min_lidar_step"])
        progress = self.previous_distance - distance

        # Preserve the dense-reward objective per unit of simulated time when
        # moving from the old 5.05 s decision interval to 1 Hz.  Progress and
        # terminal/event rewards are already independent of the sampling rate.
        dense_reward_scale = (
            self.STEP_WAIT_DURATION / self.DENSE_REWARD_REFERENCE_SECONDS
        )
        reward_progress = 15.0 * progress
        reward_safety = 0.0
        reward_speed = 0.0
        reward_step = -0.05 * dense_reward_scale
        reward_goal = 0.0
        reward_front_stop = 0.0
        reward_front_risk = 0.0
        reward_lateral_risk = 0.0
        terminated = False
        truncated = False
        reason = ""

        physical_collision = bool(info["collision_state"])
        lidar_collision = minimum_obstacle_distance < self.COLLISION_DISTANCE
        # Noisy LiDAR remains a proximity signal; physics decides termination.
        contact_collision = int(info["contact_count"]) > 0
        if physical_collision or contact_collision:
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
            reward_safety = dense_reward_scale * max(
                -5.0
                * (
                    self.SAFE_DISTANCE
                    / max(minimum_obstacle_distance, 1.0e-3)
                    - 1.0
                ),
                self.SAFETY_REWARD_CLIP,
            )
        else:
            # Reward measured motion, never the requested/applied limit itself;
            # otherwise the policy can earn reward merely by asking for 1.0.
            # Reverse motion is not counted as forward-speed reward.
            reward_speed = (
                dense_reward_scale
                * 2.0
                * max(0.0, float(info["linear_velocity"]))
            )

        # A high requested speed can keep the independent front gate latched
        # while the robot is stationary. Lower SpeedLimit requests can release
        # it without relaxing the physical stop threshold.
        if (
            info["front_safety_stop_active"]
            and (int(info["front_safety_stop_reason"]) & 2) != 0
        ):
            reward_front_stop = (
                -dense_reward_scale
                * self.FRONT_STOP_REQUEST_PENALTY_WEIGHT
                * float(requested[0])
            )

        # Penalize a high request *before* the independent gate closes. Risk
        # comes only from front scan/odom/applied envelope, so this signal is
        # defined on both simulator and robot. At 1 Hz it is bounded by -1
        # per step, compared with +100 per reached goal and -30 per deadlock.
        request_fraction = float(np.clip(
            (float(requested[0]) - float(self.ACTION_LOW[0]))
            / float(self.ACTION_HIGH[0] - self.ACTION_LOW[0]),
            0.0, 1.0,
        ))
        reward_front_risk = (
            -self.FRONT_RISK_PENALTY_WEIGHT
            * float(info.get("front_proactive_risk", 0.0))
            * request_fraction
        )
        if getattr(self, "conservative_crossing", False):
            reward_lateral_risk = (
                -self.CONSERVATIVE_LATERAL_REWARD_WEIGHT
                * float(info["conservative_lateral_risk"])
                * request_fraction
            )

        # Penalize changes in the scale Nav2 actually received, not merely
        # changes in the raw policy request (the supervisor may smooth or
        # override that request).
        actual_applied = np.asarray(
            [float(info["applied_speed_scale"])],
            dtype=np.float32,
        )
        if self.previous_applied_action is not None:
            action_range = self.ACTION_HIGH - self.ACTION_LOW
            normalized_delta = (
                actual_applied - self.previous_applied_action
            ) / action_range
            reward_smoothness = (
                -dense_reward_scale
                * self.ACTION_SMOOTHNESS_PENALTY_WEIGHT
                * float(np.sum(normalized_delta ** 2))
            )
        else:
            reward_smoothness = 0.0
        self.previous_applied_action = actual_applied

        # Rewards describe the action interval against the old goal. Keep
        # that context separate from the next state after a goal handover.
        nav_status = info["nav_status"]
        previous_goal = list(self.goal)
        previous_goal_distance = distance
        goal_event = None
        if not terminated and nav_status == GoalStatus.STATUS_SUCCEEDED:
            reward_goal = self.GOAL_REWARD
            self.goals_reached_this_episode += 1
            print(f"✅ ĐÃ TỚI ĐÍCH! (Tổng: {self.goals_reached_this_episode}) -> Đang sinh goal mới...")

            if getattr(self, "retain_dynamic_obstacle_after_crossing", False):
                self.ros.park_dynamic_obstacle()
            self._select_next_goal()
            self._send_selected_goal()

            goal_event = "succeeded"
            reason = "success_but_continue"

        elif not terminated and nav_status == GoalStatus.STATUS_ABORTED:
            reward_goal = self.NAV_ABORT_REWARD
            print("⚠️ NAV2 BỊ KẸT (ABORTED) -> Đang sinh goal mới...")

            if getattr(self, "retain_dynamic_obstacle_after_crossing", False):
                self.ros.park_dynamic_obstacle()
            self._select_next_goal()
            self._send_selected_goal()

            goal_event = "aborted"
            reason = "aborted_but_continue"

        if goal_event is not None:
            # SAC must choose its next action from the NEW goal, not the
            # observation sampled before send_navigation_goal(). Refresh
            # info as well so its goal/distance/count match that observation.
            observation, info = self._observation()
            distance = float(info["distance_to_goal"])

        info.update(
            {
                "goal_event": goal_event,
                "previous_goal": previous_goal,
                "previous_goal_distance": previous_goal_distance,
                "previous_goal_nav_status": nav_status,
                "new_goal": list(self.goal) if goal_event is not None else None,
            }
        )

        if not terminated and self._update_safety_deadlock(info, goal_event):
            terminated = True
            reason = "safety_deadlock"
            reward_goal = self.NAV_ABORT_REWARD
            self._needs_hard_reset = True
            print(
                "[Safety deadlock] Front corridor stop during "
                f"{self.safety_deadlock_steps}/{self.SAFETY_DEADLOCK_STEPS} "
                "recent steps with no meaningful movement at "
                f"({info['x']:.2f}, {info['y']:.2f}); "
                f"clearance={info['front_safety_stop_clearance_m']:.2f} m, "
                f"stop_distance={info['front_safety_stop_distance_m']:.2f} m, "
                f"dynamic_active={info['dynamic_obstacle_active']}; "
                "hard reset next episode",
                flush=True,
            )
        stall_reason = (
            self._update_motion_stall(info, goal_event) if not terminated else None
        )
        if stall_reason is not None:
            truncated = True
            reason = stall_reason
            self._needs_hard_reset = True
            print(
                f"[{stall_reason}] No displacement >"
                f"{self.MOTION_STALL_MAX_DISPLACEMENT_M:.2f} m over "
                f"{self.MOTION_STALL_STEPS} steps at "
                f"({info['x']:.2f}, {info['y']:.2f}); hard reset next episode",
                flush=True,
            )
        if not terminated and not truncated and self.episode_step_count >= self.max_episode_steps:
            truncated = True
            reason = "max_steps"
            if self.goals_reached_this_episode == 0:
                self._needs_hard_reset = True
                print(
                    "[Max steps] No goal reached; hard reset next episode",
                    flush=True,
                )

        reward = (
            reward_progress
            + reward_safety
            + reward_speed
            + reward_step
            + reward_goal
            + reward_front_stop
            + reward_front_risk
            + reward_lateral_risk
            + reward_smoothness
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
            ended_in_failure = reason in (
                "collision", "safety_deadlock", "motion_stall",
                "rotation_stall", "command_stall"
            )
            if self.SUCCESS_REQUIRES_NO_COLLISION:
                info["is_success"] = (
                    reached_at_least_one_goal and not ended_in_failure
                )
            else:
                info["is_success"] = reached_at_least_one_goal

        info.update(
            {
                "termination_reason": reason,
                "progress": progress,
                "requested_action": [float(value) for value in requested],
                "applied_speed_scale": float(info["applied_speed_scale"]),
                "physical_collision": physical_collision,
                "lidar_collision": lidar_collision,
                "contact_collision": contact_collision,  # [PATCH] để debug/log riêng nguồn này
                "reward_progress": reward_progress,
                "reward_safety": reward_safety,
                "reward_speed": reward_speed,
                "reward_smoothness": reward_smoothness,  # [GROUP2] để debug/log riêng
                "reward_goal": reward_goal,
                "reward_front_stop": reward_front_stop,
                "reward_front_risk": reward_front_risk,
                "reward_lateral_risk": reward_lateral_risk,
                "goals_reached_this_episode": self.goals_reached_this_episode,
                "safety_deadlock_steps": self.safety_deadlock_steps,
                "hard_reset_next_episode": self._needs_hard_reset,
            }
        )
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.ros.cancel_navigation()
            self.ros.park_dynamic_obstacle()
            self.ros.publish_speed_scale(1.0)
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
        goal_sampling="random_costmap",
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
