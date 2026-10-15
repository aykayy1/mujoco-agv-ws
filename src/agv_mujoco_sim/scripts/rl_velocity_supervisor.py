#!/usr/bin/env python3
"""Apply RL speed scaling to Nav2 MPPI without dynamic parameter writes.

The node accepts one value on ``/rl_velocity_limits``:

    [speed_scale]

``speed_scale`` is clamped to [0.15, 1.00].  It is published to the Nav2
Controller Server through ``nav2_msgs/msg/SpeedLimit`` in percentage mode.
Nav2 Humble MPPI applies that one ratio to all velocity constraints, so with
the verified base limits vx_max=0.70 m/s and wz_max=0.50 rad/s the effective
range is:

    vx_max: 0.105 .. 0.70 m/s
    wz_max: 0.075 .. 0.50 rad/s

This path calls MPPI's setSpeedLimit() API and therefore does not invoke the
dynamic-parameter callback that resets the optimizer.
"""

import math
import time
from collections import deque

import numpy as np
import rclpy
from nav2_msgs.msg import SpeedLimit
from rcl_interfaces.msg import ParameterType, SetParametersResult
from rcl_interfaces.srv import GetParameters
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


class RlVelocitySupervisor(Node):
    """Safely apply one-dimensional RL speed scaling to Nav2 MPPI."""

    EXPECTED_PLUGIN = "nav2_mppi_controller::MPPIController"
    PLUGIN_PARAM = "FollowPath.plugin"

    VERIFY_PERIOD = 0.5
    SERVICE_RETRY_DELAY = 0.5
    VALUE_EPSILON = 1.0e-4
    TTC_SECTORS = 24
    TTC_LOOKBACK_SECONDS = 0.35
    TTC_HISTORY_SECONDS = 0.85
    TTC_CONFIRM_SCANS = 3
    TTC_MAX_CLOSING_SPEED = 3.0

    MUTABLE_LOCAL_PARAMETERS = {
        "baseline_vx",
        "baseline_wz",
        "scale_min",
        "scale_max",
        "watchdog_timeout",
        "scale_slew_rate",
        "lowpass_alpha",
        "scale_fall_slew_rate",
        "fall_lowpass_alpha",
        "request_timeout",
        "emergency_ttc",
        "emergency_distance",
        "emergency_scale",
        "safety_hold_time",
        "min_closing_speed",
    }

    def __init__(self):
        super().__init__("rl_velocity_supervisor")

        self._declare_local_parameters()
        self._load_local_parameters()
        config_error = self._validate_config(self._config_as_dict())
        if config_error:
            raise ValueError(
                f"Invalid supervisor configuration: {config_error}"
            )

        self.target_scale = self.scale_max
        self.applied_scale = self.scale_max
        self.last_apply_time = self._control_time()
        self.last_rl_message = None
        self.watchdog_active = True
        self.safety_active = False
        self.safety_release_after = 0.0
        self.safety_min_ttc = math.inf
        self.safety_sector_ttc = np.full(self.TTC_SECTORS, 5.0, dtype=np.float32)
        self.scan_history = deque(maxlen=16)
        self.last_scan_time = None
        self.ttc_candidate_scans = 0
        self.latest_yaw = None

        self.controller_verified = False
        self.verification_future = None
        self.verification_token = 0
        self.verification_deadline = 0.0
        self.verification_retry_after = 0.0

        self.last_warning_time = 0.0
        self.last_unverified_action_warning = 0.0

        service_prefix = self.controller_node
        self.get_parameters_client = self.create_client(
            GetParameters,
            f"{service_prefix}/get_parameters",
        )

        self.speed_limit_publisher = self.create_publisher(
            SpeedLimit,
            self.speed_limit_topic,
            10,
        )
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.applied_status_publisher = self.create_publisher(
            Float32MultiArray,
            "/rl_velocity_limits_applied",
            status_qos,
        )
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "/rl_velocity_limits",
            self.rl_callback,
            10,
        )
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            sensor_qos,
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.odom_callback,
            sensor_qos,
        )

        self.local_parameter_callback_handle = (
            self.add_on_set_parameters_callback(
                self.local_parameter_callback
            )
        )
        self.verification_timer = self.create_timer(
            self.VERIFY_PERIOD,
            self.verify_controller,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )
        self.update_timer = self.create_timer(
            1.0 / self.update_hz,
            self.update_callback,
        )

        self.get_logger().info(
            f"Waiting to verify MPPI on {self.controller_node} and a "
            f"subscriber on {self.speed_limit_topic}"
        )

    def _declare_local_parameters(self):
        self.declare_parameter("controller_node", "/controller_server")
        self.declare_parameter("speed_limit_topic", "/speed_limit")
        self.declare_parameter("baseline_vx", 0.70)
        self.declare_parameter("baseline_wz", 0.50)
        self.declare_parameter("scale_min", 0.15)
        self.declare_parameter("scale_max", 1.00)
        self.declare_parameter("update_hz", 5.0)
        self.declare_parameter("watchdog_timeout", 2.5)
        self.declare_parameter("scale_slew_rate", 0.25)
        self.declare_parameter("lowpass_alpha", 0.35)
        self.declare_parameter("scale_fall_slew_rate", 0.25)
        self.declare_parameter("fall_lowpass_alpha", 0.35)
        self.declare_parameter("request_timeout", 1.0)
        self.declare_parameter("safety_override_enabled", True)
        self.declare_parameter("ttc_observation_enabled", True)
        self.declare_parameter("emergency_ttc", 1.0)
        self.declare_parameter("emergency_distance", 0.35)
        self.declare_parameter("emergency_scale", 0.15)
        self.declare_parameter("safety_hold_time", 0.75)
        self.declare_parameter("min_closing_speed", 0.05)

    def _load_local_parameters(self):
        controller_node = str(
            self.get_parameter("controller_node").value
        ).strip()
        if not controller_node:
            raise ValueError("controller_node must not be empty")
        if not controller_node.startswith("/"):
            controller_node = "/" + controller_node
        self.controller_node = controller_node.rstrip("/")

        speed_limit_topic = str(
            self.get_parameter("speed_limit_topic").value
        ).strip()
        if not speed_limit_topic:
            raise ValueError("speed_limit_topic must not be empty")
        if not speed_limit_topic.startswith("/"):
            speed_limit_topic = "/" + speed_limit_topic
        self.speed_limit_topic = speed_limit_topic

        self.baseline_vx = float(
            self.get_parameter("baseline_vx").value
        )
        self.baseline_wz = float(
            self.get_parameter("baseline_wz").value
        )
        self.scale_min = float(
            self.get_parameter("scale_min").value
        )
        self.scale_max = float(
            self.get_parameter("scale_max").value
        )
        self.update_hz = float(
            self.get_parameter("update_hz").value
        )
        self.watchdog_timeout = float(
            self.get_parameter("watchdog_timeout").value
        )
        self.scale_slew_rate = float(
            self.get_parameter("scale_slew_rate").value
        )
        self.lowpass_alpha = float(
            self.get_parameter("lowpass_alpha").value
        )
        self.scale_fall_slew_rate = float(
            self.get_parameter("scale_fall_slew_rate").value
        )
        self.fall_lowpass_alpha = float(
            self.get_parameter("fall_lowpass_alpha").value
        )
        self.request_timeout = float(
            self.get_parameter("request_timeout").value
        )
        self.safety_override_enabled = bool(
            self.get_parameter("safety_override_enabled").value
        )
        self.ttc_observation_enabled = bool(
            self.get_parameter("ttc_observation_enabled").value
        )
        self.emergency_ttc = float(self.get_parameter("emergency_ttc").value)
        self.emergency_distance = float(
            self.get_parameter("emergency_distance").value
        )
        self.emergency_scale = float(
            self.get_parameter("emergency_scale").value
        )
        self.safety_hold_time = float(
            self.get_parameter("safety_hold_time").value
        )
        self.min_closing_speed = float(
            self.get_parameter("min_closing_speed").value
        )

    def _config_as_dict(self):
        return {
            "baseline_vx": self.baseline_vx,
            "baseline_wz": self.baseline_wz,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "update_hz": self.update_hz,
            "watchdog_timeout": self.watchdog_timeout,
            "scale_slew_rate": self.scale_slew_rate,
            "lowpass_alpha": self.lowpass_alpha,
            "scale_fall_slew_rate": self.scale_fall_slew_rate,
            "fall_lowpass_alpha": self.fall_lowpass_alpha,
            "request_timeout": self.request_timeout,
            "emergency_ttc": self.emergency_ttc,
            "emergency_distance": self.emergency_distance,
            "emergency_scale": self.emergency_scale,
            "safety_hold_time": self.safety_hold_time,
            "min_closing_speed": self.min_closing_speed,
        }

    @staticmethod
    def _validate_config(config):
        for name, value in config.items():
            if not math.isfinite(float(value)):
                return f"{name} must be finite"

        if config["baseline_vx"] <= 0.0:
            return "baseline_vx must be > 0"
        if config["baseline_wz"] <= 0.0:
            return "baseline_wz must be > 0"
        if not 0.0 < config["scale_min"] <= config["scale_max"]:
            return "scale range must satisfy 0 < scale_min <= scale_max"
        if config["scale_max"] > 1.0:
            return "scale_max must be <= 1.0"
        if config["update_hz"] <= 0.0:
            return "update_hz must be > 0"
        if config["watchdog_timeout"] <= 0.0:
            return "watchdog_timeout must be > 0"
        if config["scale_slew_rate"] <= 0.0:
            return "scale_slew_rate must be > 0"
        if not 0.0 < config["lowpass_alpha"] <= 1.0:
            return "lowpass_alpha must be in (0, 1]"
        if config["scale_fall_slew_rate"] <= 0.0:
            return "scale_fall_slew_rate must be > 0"
        if not 0.0 < config["fall_lowpass_alpha"] <= 1.0:
            return "fall_lowpass_alpha must be in (0, 1]"
        if config["request_timeout"] <= 0.0:
            return "request_timeout must be > 0"
        if config["emergency_ttc"] <= 0.0:
            return "emergency_ttc must be > 0"
        if config["emergency_distance"] <= 0.0:
            return "emergency_distance must be > 0"
        if not config["scale_min"] <= config["emergency_scale"] <= config["scale_max"]:
            return "emergency_scale must be inside the configured scale range"
        if config["safety_hold_time"] < 0.0:
            return "safety_hold_time must be >= 0"
        if config["min_closing_speed"] <= 0.0:
            return "min_closing_speed must be > 0"
        return ""

    def local_parameter_callback(self, parameters):
        candidate = self._config_as_dict()

        for parameter in parameters:
            name = parameter.name
            if name in (
                "controller_node",
                "speed_limit_topic",
                "update_hz",
                "safety_override_enabled",
                "ttc_observation_enabled",
            ):
                return SetParametersResult(
                    successful=False,
                    reason=f"{name} requires restarting the supervisor",
                )
            if name not in self.MUTABLE_LOCAL_PARAMETERS:
                continue
            try:
                candidate[name] = float(parameter.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False,
                    reason=f"{name} must be a number",
                )

        config_error = self._validate_config(candidate)
        if config_error:
            return SetParametersResult(
                successful=False,
                reason=config_error,
            )

        for name in self.MUTABLE_LOCAL_PARAMETERS:
            setattr(self, name, candidate[name])

        self.target_scale = self.clamp(
            self.target_scale,
            self.scale_min,
            self.scale_max,
        )
        self.applied_scale = self.clamp(
            self.applied_scale,
            self.scale_min,
            self.scale_max,
        )
        if self.watchdog_active:
            self.target_scale = self.scale_max

        return SetParametersResult(successful=True)

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(float(value), maximum))

    @staticmethod
    def move_towards(current, target, max_step):
        error = target - current
        if abs(error) <= max_step:
            return target
        return current + math.copysign(max_step, error)

    @staticmethod
    def smoothed_scale_step(current, target, dt, rise_rate, rise_alpha,
                            fall_rate, fall_alpha):
        falling = target < current
        alpha = fall_alpha if falling else rise_alpha
        rate = fall_rate if falling else rise_rate
        filtered = current + alpha * (target - current)
        return RlVelocitySupervisor.move_towards(current, filtered, rate * dt)

    def _control_time(self):
        """ROS time in simulation and system time on the real robot."""
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _warn_throttled(self, message, period=5.0):
        now = time.monotonic()
        if now - self.last_warning_time >= period:
            self.get_logger().warning(message)
            self.last_warning_time = now

    def verify_controller(self):
        if self.controller_verified:
            return

        now = time.monotonic()
        if (
            self.verification_future is not None
            and now > self.verification_deadline
        ):
            self.verification_token += 1
            self.verification_future = None
            self.verification_retry_after = (
                now + self.SERVICE_RETRY_DELAY
            )
            self._warn_throttled(
                f"Timeout while reading {self.PLUGIN_PARAM}"
            )

        if (
            self.verification_future is not None
            or now < self.verification_retry_after
        ):
            return

        if not self.get_parameters_client.service_is_ready():
            self._warn_throttled(
                f"Waiting for {self.controller_node}/get_parameters"
            )
            return

        if self.speed_limit_publisher.get_subscription_count() < 1:
            self._warn_throttled(
                f"Waiting for Controller Server subscription on "
                f"{self.speed_limit_topic}"
            )
            return

        request = GetParameters.Request()
        request.names = [self.PLUGIN_PARAM]
        self.verification_token += 1
        token = self.verification_token
        future = self.get_parameters_client.call_async(request)
        self.verification_future = future
        self.verification_deadline = now + self.request_timeout
        future.add_done_callback(
            lambda completed: self.plugin_response_callback(
                completed,
                token,
            )
        )

    def plugin_response_callback(self, future, token):
        if (
            self.verification_future is None
            or token != self.verification_token
        ):
            return
        self.verification_future = None

        try:
            response = future.result()
        except Exception as error:
            self.verification_retry_after = (
                time.monotonic() + self.SERVICE_RETRY_DELAY
            )
            self._warn_throttled(
                f"Could not read {self.PLUGIN_PARAM}: {error}"
            )
            return

        if response is None or len(response.values) != 1:
            self.verification_retry_after = (
                time.monotonic() + self.SERVICE_RETRY_DELAY
            )
            self._warn_throttled(
                f"Incomplete response for {self.PLUGIN_PARAM}"
            )
            return

        plugin_value = response.values[0]
        actual_plugin = (
            plugin_value.string_value
            if plugin_value.type == ParameterType.PARAMETER_STRING
            else "<parameter not set or wrong type>"
        )
        if actual_plugin != self.EXPECTED_PLUGIN:
            self.verification_retry_after = (
                time.monotonic() + self.SERVICE_RETRY_DELAY
            )
            self._warn_throttled(
                "RL supervisor blocked: expected "
                f"{self.EXPECTED_PLUGIN}, got {actual_plugin}"
            )
            return

        self.controller_verified = True
        self.verification_retry_after = 0.0
        self.publish_scale(self.scale_max)
        self.get_logger().info(
            "MPPI READY: using nav2_msgs/SpeedLimit; dynamic parameter "
            "writes are disabled"
        )
        self.get_logger().info(
            "Listening on /rl_velocity_limits for exactly [speed_scale]"
        )

    def rl_callback(self, msg):
        if not self.controller_verified:
            now = time.monotonic()
            if now - self.last_unverified_action_warning >= 2.0:
                self.get_logger().warning(
                    "RL action ignored: MPPI SpeedLimit path is not READY"
                )
                self.last_unverified_action_warning = now
            return

        if len(msg.data) != 1:
            self.get_logger().warning(
                "RL action rejected: expected exactly [speed_scale]"
            )
            return

        scale = float(msg.data[0])
        if not math.isfinite(scale):
            self.get_logger().warning(
                "RL action rejected: NaN or infinity"
            )
            return

        self.target_scale = self.clamp(
            scale,
            self.scale_min,
            self.scale_max,
        )
        self.last_rl_message = self._control_time()

        if self.watchdog_active:
            self.get_logger().info(
                "Valid RL actions received; adaptive speed enabled"
            )
        self.watchdog_active = False

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.latest_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    @classmethod
    def _sector_ranges(cls, ranges):
        sectors = np.full(cls.TTC_SECTORS, np.inf, dtype=np.float32)
        for index, rays in enumerate(np.array_split(ranges, cls.TTC_SECTORS)):
            finite = rays[np.isfinite(rays)]
            if finite.size >= 3:
                sectors[index] = float(np.quantile(finite, 0.2))
        return sectors

    @staticmethod
    def _align_previous_scan(previous, angle_increment, yaw_delta):
        """Compare rays aimed at the same world bearing during robot turns."""
        indices = np.arange(previous.size, dtype=np.float64)
        shifted = indices + yaw_delta / angle_increment
        left = np.floor(shifted).astype(np.int64)
        right = left + 1
        valid = (left >= 0) & (right < previous.size)
        aligned = np.full(previous.shape, np.inf, dtype=np.float32)
        if not np.any(valid):
            return aligned
        output_indices = np.flatnonzero(valid)
        left_values = previous[left[valid]]
        right_values = previous[right[valid]]
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        fraction = shifted[valid][finite] - left[valid][finite]
        aligned[output_indices[finite]] = (
            (1.0 - fraction) * left_values[finite]
            + fraction * right_values[finite]
        )
        return aligned

    def scan_callback(self, msg):
        """Estimate TTC for observations; optionally apply an independent shield."""
        if not self.ttc_observation_enabled:
            return
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        if ranges.size == 0:
            return
        angles = (
            float(msg.angle_min)
            + np.arange(ranges.size) * float(msg.angle_increment)
        )
        front = (
            (angles >= -math.pi / 3.0 - 1.0e-5)
            & (angles <= math.pi / 3.0 + 1.0e-5)
        )
        # NaN is an unobserved/masked ray; +Inf is a valid no-return ray.
        observed = front & (
            np.isposinf(ranges)
            | (np.isfinite(ranges) & (ranges > 0.0))
        )
        valid = observed & np.isfinite(ranges) & (ranges > 0.0)
        current = np.where(valid, ranges, np.inf)
        stamp = msg.header.stamp
        scan_time = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        if scan_time <= 0.0:
            scan_time = self._control_time()

        min_range = float(np.min(current))
        min_ttc = math.inf
        if self.last_scan_time is not None and (
            scan_time <= self.last_scan_time
            or scan_time - self.last_scan_time > self.TTC_HISTORY_SECONDS
        ):
            self.scan_history.clear()
            self.ttc_candidate_scans = 0
        self.last_scan_time = scan_time

        front_ranges = current[front]
        sectors = self._sector_ranges(front_ranges)

        while self.scan_history and (
            scan_time - self.scan_history[0][0] > self.TTC_HISTORY_SECONDS
        ):
            self.scan_history.popleft()
        reference = next(
            (
                (old_time, old_ranges, old_yaw)
                for old_time, old_ranges, old_yaw in reversed(self.scan_history)
                if scan_time - old_time >= self.TTC_LOOKBACK_SECONDS
            ),
            None,
        )
        sector_ttc = np.full(self.TTC_SECTORS, np.inf, dtype=np.float32)
        if reference is not None and self.latest_yaw is not None:
            old_time, old_ranges, old_yaw = reference
            dt = scan_time - old_time
            if old_yaw is not None and front_ranges.shape == old_ranges.shape:
                yaw_delta = math.atan2(
                    math.sin(self.latest_yaw - old_yaw),
                    math.cos(self.latest_yaw - old_yaw),
                )
                if abs(yaw_delta) <= 0.5:
                    aligned_old_ranges = self._align_previous_scan(
                        old_ranges, float(msg.angle_increment), yaw_delta
                    )
                    old_sectors = self._sector_ranges(aligned_old_ranges)
                    finite_pair = np.isfinite(sectors) & np.isfinite(old_sectors)
                    closing_speed = np.zeros_like(sectors)
                    np.subtract(
                        old_sectors, sectors, out=closing_speed,
                        where=finite_pair,
                    )
                    closing_speed /= dt
                    closing = (
                        finite_pair
                        & (closing_speed > self.min_closing_speed)
                        & (closing_speed <= self.TTC_MAX_CLOSING_SPEED)
                    )
                    if np.any(closing):
                        sector_ttc[closing] = sectors[closing] / closing_speed[closing]
                        min_ttc = float(np.min(sector_ttc))
        self.scan_history.append((scan_time, front_ranges.copy(), self.latest_yaw))

        hazardous_sectors = sector_ttc <= self.emergency_ttc
        spatially_supported = bool(
            np.any(hazardous_sectors[:-1] & hazardous_sectors[1:])
        )
        if spatially_supported:
            self.ttc_candidate_scans += 1
        else:
            self.ttc_candidate_scans = 0
        ttc_confirmed = self.ttc_candidate_scans >= self.TTC_CONFIRM_SCANS
        # Give SAC the aligned estimate before the shield's persistence gate,
        # so it can act early rather than only after an emergency override.
        self.safety_sector_ttc = np.clip(sector_ttc, 0.0, 5.0)
        self.safety_min_ttc = min_ttc
        if not self.safety_override_enabled:
            return
        emergency = (
            not np.any(observed)
            or min_range <= self.emergency_distance
            or ttc_confirmed
        )
        if emergency:
            self.safety_release_after = self._control_time() + self.safety_hold_time
            if not self.safety_active:
                self.get_logger().warning(
                    "TTC safety override active: applying emergency speed scale "
                    f"(range={min_range:.3f} m, TTC={min_ttc:.3f} s, "
                    f"confirmed_scans={self.ttc_candidate_scans})"
                )
            self.safety_active = True
            if self.controller_verified and self.applied_scale > self.emergency_scale:
                self.publish_scale(self.emergency_scale)

    def update_callback(self):
        control_now = self._control_time()
        wall_now = time.monotonic()
        if not self.controller_verified:
            return

        if control_now < self.last_apply_time:
            # MuJoCo episode reset can rewind /clock.  Do not carry an RL
            # lease or an old safety hold across that discontinuity.
            self.last_apply_time = control_now
            self.last_rl_message = None
            self.safety_active = False
            self.safety_release_after = control_now
            self.scan_history.clear()
            self.last_scan_time = None
            self.ttc_candidate_scans = 0
            self.latest_yaw = None
            self.safety_sector_ttc.fill(5.0)

        if self.speed_limit_publisher.get_subscription_count() < 1:
            self.controller_verified = False
            self.verification_retry_after = (
                wall_now + self.SERVICE_RETRY_DELAY
            )
            self._warn_throttled(
                f"Controller Server subscription disappeared from "
                f"{self.speed_limit_topic}"
            )
            return

        command_age = (
            math.inf
            if self.last_rl_message is None
            else control_now - self.last_rl_message
        )
        command_stale = command_age < 0.0 or command_age > self.watchdog_timeout
        if command_stale:
            desired_scale = self.scale_max
            watchdog_just_activated = not self.watchdog_active
            if not self.watchdog_active:
                self.get_logger().warning(
                    "RL watchdog timeout: removing the MPPI speed limit"
                )
                self.watchdog_active = True
        else:
            desired_scale = self.target_scale
            watchdog_just_activated = False

        if self.safety_active:
            if control_now < self.safety_release_after:
                desired_scale = min(desired_scale, self.emergency_scale)
            else:
                self.safety_active = False
                self.get_logger().info(
                    "TTC safety override released; returning to smoothed RL control"
                )

        if watchdog_just_activated and not self.safety_active:
            self.publish_scale(self.scale_max)
            return

        dt = min(max(control_now - self.last_apply_time, 0.001), 0.5)
        if self.safety_active and desired_scale < self.applied_scale:
            next_scale = desired_scale
        else:
            next_scale = self.smoothed_scale_step(
                self.applied_scale, desired_scale, dt,
                self.scale_slew_rate, self.lowpass_alpha,
                self.scale_fall_slew_rate, self.fall_lowpass_alpha,
            )

        if abs(next_scale - self.applied_scale) < self.VALUE_EPSILON:
            self.publish_status()
            return

        self.publish_scale(next_scale)

    def publish_scale(self, scale):
        scale = self.clamp(scale, self.scale_min, self.scale_max)

        speed_msg = SpeedLimit()
        if abs(scale - 1.0) < self.VALUE_EPSILON:
            # Nav2 defines zero as NO_SPEED_LIMIT.
            speed_msg.speed_limit = 0.0
            speed_msg.percentage = False
        else:
            speed_msg.speed_limit = 100.0 * scale
            speed_msg.percentage = True
        self.speed_limit_publisher.publish(speed_msg)

        self.applied_scale = scale
        self.last_apply_time = self._control_time()

        self.publish_status()

    def publish_status(self):
        """Publish a latched acknowledgement and safety diagnostics."""

        status_msg = Float32MultiArray()
        status_msg.data = [
            float(self.applied_scale),
            float(self.baseline_vx * self.applied_scale),
            float(self.baseline_wz * self.applied_scale),
            float(self.safety_active),
            float(self.safety_min_ttc),
        ] + self.safety_sector_ttc.astype(float).tolist()
        self.applied_status_publisher.publish(status_msg)

    def restore_baseline(self):
        if self.speed_limit_publisher.get_subscription_count() > 0:
            self.publish_scale(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = RlVelocitySupervisor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.restore_baseline()
            except Exception:
                pass
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
