#!/usr/bin/env python3
"""Apply RL speed scaling to Nav2 MPPI without dynamic parameter writes.

The node accepts one value on ``/rl_velocity_limits``:

    [speed_scale]

``speed_scale`` is clamped to [0.40, 1.00].  It is published to the Nav2
Controller Server through ``nav2_msgs/msg/SpeedLimit`` in percentage mode.
Nav2 Humble MPPI applies that one ratio to all velocity constraints, so with
the verified base limits vx_max=0.40 m/s and wz_max=0.50 rad/s the effective
range is:

    vx_max: 0.16 .. 0.40 m/s
    wz_max: 0.20 .. 0.50 rad/s

This path calls MPPI's setSpeedLimit() API and therefore does not invoke the
dynamic-parameter callback that resets the optimizer.
"""

import math
import time

import rclpy
from nav2_msgs.msg import SpeedLimit
from rcl_interfaces.msg import ParameterType, SetParametersResult
from rcl_interfaces.srv import GetParameters
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RlVelocitySupervisor(Node):
    """Safely apply one-dimensional RL speed scaling to Nav2 MPPI."""

    EXPECTED_PLUGIN = "nav2_mppi_controller::MPPIController"
    PLUGIN_PARAM = "FollowPath.plugin"

    VERIFY_PERIOD = 0.5
    SERVICE_RETRY_DELAY = 0.5
    VALUE_EPSILON = 1.0e-4

    MUTABLE_LOCAL_PARAMETERS = {
        "baseline_vx",
        "baseline_wz",
        "scale_min",
        "scale_max",
        "watchdog_timeout",
        "scale_slew_rate",
        "lowpass_alpha",
        "request_timeout",
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
        self.applied_status_publisher = self.create_publisher(
            Float32MultiArray,
            "/rl_velocity_limits_applied",
            10,
        )
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "/rl_velocity_limits",
            self.rl_callback,
            10,
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
        self.declare_parameter("baseline_vx", 0.40)
        self.declare_parameter("baseline_wz", 0.50)
        self.declare_parameter("scale_min", 0.40)
        self.declare_parameter("scale_max", 1.00)
        self.declare_parameter("update_hz", 5.0)
        self.declare_parameter("watchdog_timeout", 2.5)
        self.declare_parameter("scale_slew_rate", 0.25)
        self.declare_parameter("lowpass_alpha", 0.35)
        self.declare_parameter("request_timeout", 1.0)

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
        self.request_timeout = float(
            self.get_parameter("request_timeout").value
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
            "request_timeout": self.request_timeout,
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
        if config["request_timeout"] <= 0.0:
            return "request_timeout must be > 0"
        return ""

    def local_parameter_callback(self, parameters):
        candidate = self._config_as_dict()

        for parameter in parameters:
            name = parameter.name
            if name in (
                "controller_node",
                "speed_limit_topic",
                "update_hz",
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

    def update_callback(self):
        control_now = self._control_time()
        wall_now = time.monotonic()
        if not self.controller_verified:
            return

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

        command_stale = (
            self.last_rl_message is None
            or control_now - self.last_rl_message > self.watchdog_timeout
        )
        if command_stale:
            desired_scale = self.scale_max
            if not self.watchdog_active:
                self.get_logger().warning(
                    "RL watchdog timeout: removing the MPPI speed limit"
                )
                self.watchdog_active = True
        else:
            desired_scale = self.target_scale

        dt = min(max(control_now - self.last_apply_time, 0.001), 0.5)
        filtered_scale = (
            self.applied_scale
            + self.lowpass_alpha
            * (desired_scale - self.applied_scale)
        )
        next_scale = self.move_towards(
            self.applied_scale,
            filtered_scale,
            self.scale_slew_rate * dt,
        )

        if abs(next_scale - self.applied_scale) < self.VALUE_EPSILON:
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

        status_msg = Float32MultiArray()
        status_msg.data = [
            float(scale),
            float(self.baseline_vx * scale),
            float(self.baseline_wz * scale),
        ]
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
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
