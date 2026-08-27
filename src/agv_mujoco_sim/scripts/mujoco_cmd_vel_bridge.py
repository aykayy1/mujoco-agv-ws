#!/usr/bin/env python3

import json
import math
import os
import sys
import time
from collections import deque
from contextlib import nullcontext
from typing import Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import TransformStamped, Twist
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger

try:
    import mujoco
except ImportError as exc:
    print("Không import được MuJoCo.")
    print("Chạy: python3 -m pip install --user mujoco")
    raise SystemExit(1) from exc

from domain_randomizer import DomainRandomizer


PACKAGE_NAME = "agv_mujoco_sim"
MODEL_FILE = "agv_spawn_transfer_arena.xml"

WHEEL_RADIUS = 0.1

WHEEL_SEPARATION_COMMAND = 0.636
WHEEL_SEPARATION_ODOM = 0.636

MAX_WHEEL_SPEED = 8.3775804096

CMD_TIMEOUT = 0.5

CLOCK_RATE = 100.0
MOTOR_COMMAND_RATE = 20.0
STATE_RATE = 20.0
VIEWER_RATE = 60.0

SETTLE_TIME = 1.0

ODOM_FRAME = "odom"
BASE_FRAME = "base_foot_link"
GROUND_TRUTH_ODOM_TOPIC = "/ground_truth/odom"
WHEEL_ODOM_TOPIC = "/wheel/odom"

LEFT_JOINT = "left_wheel_jt"
RIGHT_JOINT = "right_wheel_jt"

LEFT_ACTUATOR = "left_wheel_motor"
RIGHT_ACTUATOR = "right_wheel_motor"

BASE_FOOTPRINT_SITE = "base_foot_link"
IMU_RATE = 50.0

IMU_FRAME = "imu"

IMU_GYRO_SENSOR = "imu_gyro"
IMU_ACCEL_SENSOR = "imu_accelerometer"
IMU_ORIENTATION_SENSOR = "imu_orientation"

LIDAR_RATE = 10.0
LIDAR_FRAME = "lidar"
LIDAR_SITE = "lidar_origin"
LIDAR_NUM_RAYS = 720
LIDAR_ANGLE_MIN = -math.pi
LIDAR_ANGLE_INCREMENT = math.radians(0.5)
LIDAR_RANGE_MIN = 0.12
LIDAR_RANGE_MAX = 12.0

# Khớp filter đang chạy trên robot thật:
#   LaserScanAngularBoundsFilterInPlace, -60...+60 độ.
# /scan_raw vẫn là 720 mẫu toàn vòng. /scan giữ nguyên metadata và kích thước
# 720 mẫu giống filter InPlace; các tia ngoài FOV 120 độ được thay bằng NaN.
# Với scan gốc -180...+179.5 độ, bước 0.5 độ, vùng hợp lệ là index
# 240...480 (241 tia, gồm cả hai biên).
LIDAR_FILTER_ANGLE_MIN = -math.pi / 3.0
LIDAR_FILTER_ANGLE_MAX = math.pi / 3.0
LIDAR_FILTER_START_INDEX = int(round(
    (LIDAR_FILTER_ANGLE_MIN - LIDAR_ANGLE_MIN)
    / LIDAR_ANGLE_INCREMENT
))
LIDAR_FILTER_END_INDEX = int(round(
    (LIDAR_FILTER_ANGLE_MAX - LIDAR_ANGLE_MIN)
    / LIDAR_ANGLE_INCREMENT
))

# Geom môi trường dùng group 0/1; geom robot dùng group 2/3.
# Chỉ các tiếp xúc robot-môi trường theo phương ngang mới được xem là
# va chạm điều hướng. Tiếp xúc thẳng đứng với sàn bị loại bỏ.
ENVIRONMENT_GEOM_GROUPS = frozenset((0, 1))
ROBOT_GEOM_GROUPS = frozenset((2, 3))
GROUND_NORMAL_Z_THRESHOLD = 0.70
GROUND_GEOM_NAMES = frozenset(
    (
        "floor",
        "ground",
        "ground_plane",
        "world_floor",
        "floor_geom",
    )
)

# Chỉ ray-cast các geom môi trường thuộc group 0 hoặc 1.
# Robot hiện dùng group 2 cho visual và group 3 cho collision, nhờ đó
# LiDAR không nhìn thấy chính thân và bánh xe của robot.
LIDAR_GEOM_GROUPS = np.array(
    [1, 1, 0, 0, 0, 0],
    dtype=np.uint8,
)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def seconds_to_ros_time(seconds_float: float) -> RosTime:
    seconds = int(math.floor(seconds_float))
    nanoseconds = int(round((seconds_float - seconds) * 1_000_000_000))

    if nanoseconds >= 1_000_000_000:
        seconds += 1
        nanoseconds -= 1_000_000_000

    msg = RosTime()
    msg.sec = seconds
    msg.nanosec = nanoseconds
    return msg


def get_named_id(
    model: mujoco.MjModel,
    object_type,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(
            f"Không tìm thấy MuJoCo object: {name}"
        )

    return object_id


def site_yaw(
    data: mujoco.MjData,
    site_id: int,
) -> float:
    rotation_matrix = (
        data.site_xmat[site_id]
        .reshape(3, 3)
    )

    return math.atan2(
        float(rotation_matrix[1, 0]),
        float(rotation_matrix[0, 0]),
    )


def cmd_vel_to_wheels(
    vx: float,
    wz: float,
) -> Tuple[float, float]:
    omega_left = (
        vx - 0.5 * WHEEL_SEPARATION_COMMAND * wz
    ) / WHEEL_RADIUS

    omega_right = (
        vx + 0.5 * WHEEL_SEPARATION_COMMAND * wz
    ) / WHEEL_RADIUS

    omega_left = max(
        -MAX_WHEEL_SPEED,
        min(MAX_WHEEL_SPEED, omega_left),
    )

    omega_right = max(
        -MAX_WHEEL_SPEED,
        min(MAX_WHEEL_SPEED, omega_right),
    )

    return omega_left, omega_right


def integrate_differential_drive(
    x: float,
    y: float,
    yaw: float,
    delta_left_angle: float,
    delta_right_angle: float,
    linear_scale: float = 1.0,
    angular_scale: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """Tích phân encoder hai bánh trong mặt phẳng.

    Trả về x, y, yaw mới cùng quãng đường tâm robot và delta yaw.
    Hàm chỉ dùng góc quay bánh, không đọc pose thân MuJoCo.
    """
    delta_left = WHEEL_RADIUS * delta_left_angle
    delta_right = WHEEL_RADIUS * delta_right_angle
    delta_distance = 0.5 * (delta_left + delta_right) * linear_scale
    delta_yaw = (
        (delta_right - delta_left)
        / WHEEL_SEPARATION_ODOM
        * angular_scale
    )
    midpoint_yaw = yaw + 0.5 * delta_yaw

    x += delta_distance * math.cos(midpoint_yaw)
    y += delta_distance * math.sin(midpoint_yaw)
    yaw = normalize_angle(yaw + delta_yaw)

    return x, y, yaw, delta_distance, delta_yaw


def create_pose_covariance() -> list:
    covariance = [0.0] * 36

    covariance[0] = 1.0e-4       # x
    covariance[7] = 1.0e-4       # y
    covariance[14] = 1.0e6       # z
    covariance[21] = 1.0e6       # roll
    covariance[28] = 1.0e6       # pitch
    covariance[35] = 1.0e-4      # yaw

    return covariance


def create_twist_covariance() -> list:
    covariance = [0.0] * 36

    covariance[0] = 1.0e-3       # vx
    covariance[7] = 1.0e3        # vy
    covariance[14] = 1.0e6       # vz
    covariance[21] = 1.0e6       # wx
    covariance[28] = 1.0e6       # wy
    covariance[35] = 1.0e-3      # wz

    return covariance


def create_wheel_pose_covariance() -> list:
    """Covariance khởi tạo; sẽ tuning lại từ residual encoder thực tế."""
    covariance = [0.0] * 36
    covariance[0] = 4.0e-4       # x: (0.02 m)^2
    covariance[7] = 4.0e-4       # y: (0.02 m)^2
    covariance[14] = 1.0e6       # z không quan sát
    covariance[21] = 1.0e6       # roll không quan sát
    covariance[28] = 1.0e6       # pitch không quan sát
    covariance[35] = 4.0e-4      # yaw: (0.02 rad)^2
    return covariance


def create_wheel_twist_covariance() -> list:
    """Covariance ban đầu cho vx, vy=0 và wz đưa vào EKF."""
    covariance = [0.0] * 36
    covariance[0] = 4.0e-4       # vx: (0.02 m/s)^2
    covariance[7] = 1.0e-4       # vy=0: ràng buộc nonholonomic
    covariance[14] = 1.0e6       # vz không quan sát
    covariance[21] = 1.0e6       # wx không quan sát
    covariance[28] = 1.0e6       # wy không quan sát
    covariance[35] = 4.0e-4      # wz: (0.02 rad/s)^2
    return covariance


def read_mujoco_sensor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_id: int,
) -> np.ndarray:
    """Đọc đúng vùng dữ liệu của một MuJoCo sensor."""
    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])

    return np.asarray(
        data.sensordata[address:address + dimension],
        dtype=np.float64,
    ).copy()


def geom_name(
    model: mujoco.MjModel,
    geom_id: int,
) -> str:
    """Trả về tên geom ổn định để chẩn đoán collision."""
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )
    return name if name else f"geom_{geom_id}"


def detect_navigation_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> Tuple[bool, int, list]:
    """Đếm tiếp xúc robot-vật cản, bỏ qua robot-sàn.

    `contact_count` là số contact point trong physics step hiện tại. Danh sách
    pair chỉ dùng để log khi trạng thái collision thay đổi.
    """
    contact_count = 0
    contact_pairs = set()

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)

        group1 = int(model.geom_group[geom1_id])
        group2 = int(model.geom_group[geom2_id])

        if (
            group1 in ROBOT_GEOM_GROUPS
            and group2 in ENVIRONMENT_GEOM_GROUPS
        ):
            robot_geom_id = geom1_id
            environment_geom_id = geom2_id
        elif (
            group2 in ROBOT_GEOM_GROUPS
            and group1 in ENVIRONMENT_GEOM_GROUPS
        ):
            robot_geom_id = geom2_id
            environment_geom_id = geom1_id
        else:
            continue

        environment_name = geom_name(
            model,
            environment_geom_id,
        )

        environment_type = int(
            model.geom_type[environment_geom_id]
        )

        # Plane luôn là sàn trong các scenario AGV hiện tại.
        if environment_type == int(
            mujoco.mjtGeom.mjGEOM_PLANE
        ):
            continue

        if environment_name.lower() in GROUND_GEOM_NAMES:
            continue

        # MuJoCo lưu normal của contact ở ba phần tử đầu contact.frame.
        # Normal gần trục Z là tiếp xúc đỡ tải với mặt sàn, không phải va chạm
        # ngang cần đánh giá cho Nav2.
        normal_z = abs(float(contact.frame[2]))
        if normal_z >= GROUND_NORMAL_Z_THRESHOLD:
            continue

        contact_count += 1
        contact_pairs.add(
            (
                geom_name(model, robot_geom_id),
                environment_name,
            )
        )

    return (
        contact_count > 0,
        contact_count,
        sorted(contact_pairs),
    )


def create_imu_orientation_covariance() -> list:
    covariance = [0.0] * 9
    covariance[0] = 1.0e-4
    covariance[4] = 1.0e-4
    covariance[8] = 1.0e-4
    return covariance


def create_imu_angular_velocity_covariance() -> list:
    covariance = [0.0] * 9
    covariance[0] = 1.0e-5
    covariance[4] = 1.0e-5
    covariance[8] = 1.0e-5
    return covariance


def create_imu_linear_acceleration_covariance() -> list:
    covariance = [0.0] * 9
    covariance[0] = 1.0e-3
    covariance[4] = 1.0e-3
    covariance[8] = 1.0e-3
    return covariance


class MujocoRosBridge(Node):

    def __init__(self) -> None:
        super().__init__("mujoco_ros_bridge")

        # Load the shared Sim2Real baseline from agv_physical_params.yaml.
        # The module-level values remain safe defaults when the node is run
        # directly without the launch file.
        global WHEEL_RADIUS
        global WHEEL_SEPARATION_COMMAND
        global WHEEL_SEPARATION_ODOM
        global MAX_WHEEL_SPEED
        global MOTOR_COMMAND_RATE
        global STATE_RATE
        global IMU_RATE
        global LIDAR_RATE
        global BASE_FRAME
        global IMU_FRAME
        global LIDAR_FRAME
        global WHEEL_ODOM_TOPIC

        parameter_defaults = {
            "model_file": MODEL_FILE,
            "wheel_radius": WHEEL_RADIUS,
            "wheel_separation_command": WHEEL_SEPARATION_COMMAND,
            "wheel_separation_odom": WHEEL_SEPARATION_ODOM,
            "max_wheel_speed_rad_s": MAX_WHEEL_SPEED,
            "motor_command_rate_hz": MOTOR_COMMAND_RATE,
            "wheel_odom_rate_hz": STATE_RATE,
            "imu_rate_hz": IMU_RATE,
            "lidar_rate_hz": LIDAR_RATE,
            "base_frame": BASE_FRAME,
            "imu_frame": IMU_FRAME,
            "lidar_frame": LIDAR_FRAME,
            "wheel_odom_topic": WHEEL_ODOM_TOPIC,
        }
        for parameter_name, default_value in parameter_defaults.items():
            self.declare_parameter(parameter_name, default_value)

        self.declare_parameter("headless", False)
        self.declare_parameter("real_time_factor", 1.0)
        DomainRandomizer.declare_ros_parameters(self)

        self.headless = bool(self.get_parameter("headless").value)
        self.real_time_factor = float(
            self.get_parameter("real_time_factor").value
        )
        self.model_file = str(
            self.get_parameter("model_file").value
        ).strip()

        if (
            not self.model_file
            or os.path.basename(self.model_file) != self.model_file
            or not self.model_file.endswith(".xml")
        ):
            raise ValueError(
                "model_file must be an XML filename inside the package "
                "models directory"
            )

        if (
            not math.isfinite(self.real_time_factor)
            or self.real_time_factor < 0.0
        ):
            raise ValueError(
                "real_time_factor must be finite and >= 0.0"
            )

        WHEEL_RADIUS = float(self.get_parameter("wheel_radius").value)
        WHEEL_SEPARATION_COMMAND = float(
            self.get_parameter("wheel_separation_command").value
        )

        WHEEL_SEPARATION_ODOM = float(
            self.get_parameter("wheel_separation_odom").value
        )
        MAX_WHEEL_SPEED = float(
            self.get_parameter("max_wheel_speed_rad_s").value
        )
        MOTOR_COMMAND_RATE = float(
            self.get_parameter("motor_command_rate_hz").value
        )
        STATE_RATE = float(
            self.get_parameter("wheel_odom_rate_hz").value
        )
        IMU_RATE = float(self.get_parameter("imu_rate_hz").value)
        LIDAR_RATE = float(self.get_parameter("lidar_rate_hz").value)
        BASE_FRAME = str(self.get_parameter("base_frame").value)
        IMU_FRAME = str(self.get_parameter("imu_frame").value)
        LIDAR_FRAME = str(self.get_parameter("lidar_frame").value)
        WHEEL_ODOM_TOPIC = str(
            self.get_parameter("wheel_odom_topic").value
        )

        self.domain_randomizer = DomainRandomizer.from_ros_node(self)

        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )

        clock_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.clock_publisher = self.create_publisher(
            Clock,
            "/clock",
            clock_qos,
        )

        self.joint_state_publisher = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.ground_truth_odom_publisher = self.create_publisher(
            Odometry,
            GROUND_TRUTH_ODOM_TOPIC,
            10,
        )

        self.wheel_odom_publisher = self.create_publisher(
            Odometry,
            WHEEL_ODOM_TOPIC,
            10,
        )

        self.imu_publisher = self.create_publisher(
            Imu,
            "/imu",
            10,
        )

        scan_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.scan_publisher = self.create_publisher(
            LaserScan,
            "/scan",
            scan_qos,
        )

        self.raw_scan_publisher = self.create_publisher(
            LaserScan,
            "/scan_raw",
            scan_qos,
        )

        self.collision_state_publisher = self.create_publisher(
            Bool,
            "/collision_state",
            10,
        )

        self.contact_count_publisher = self.create_publisher(
            Int32,
            "/contact_count",
            10,
        )

        domain_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.domain_randomization_state_publisher = self.create_publisher(
            String,
            "/domain_randomization/state",
            domain_state_qos,
        )

        # The simulator objects are attached only after the MuJoCo model has
        # loaded and the robot has settled.  The Trigger service is available
        # immediately, but safely rejects requests until that point.
        self.simulation_ready = False
        self.sim_model = None
        self.sim_data = None
        self.initial_qpos = None
        self.sim_left_joint_id = None
        self.sim_right_joint_id = None
        self.reset_generation = 0

        self.reset_simulation_service = self.create_service(
            Trigger,
            "/reset_simulation",
            self.reset_simulation_callback,
        )

        self.declare_parameter(
            "publish_odom_tf",
            False,
        )

        self.publish_odom_tf_enabled = bool(
            self.get_parameter(
                "publish_odom_tf"
            ).value
        )

        self.tf_broadcaster = TransformBroadcaster(
            self
        )

        self.get_logger().info(
            f"Publish ground-truth odom TF: "
            f"{self.publish_odom_tf_enabled}"
        )

        if self.publish_odom_tf_enabled:
            self.get_logger().warn(
                "publish_odom_tf=true chỉ dùng để tái tạo baseline cũ. "
                "Khi chạy EKF phải đặt false để tránh hai nguồn TF odom."
            )

        self.target_vx = 0.0
        self.target_wz = 0.0
        self.pending_commands = deque()

        self.wheel_odom_x = 0.0
        self.wheel_odom_y = 0.0
        self.wheel_odom_yaw = 0.0
        self.last_left_wheel_angle = 0.0
        self.last_right_wheel_angle = 0.0
        self.last_wheel_odom_time = 0.0
        self.wheel_odom_initialized = False

        self.received_cmd = False
        self.cmd_timeout_reported = False
        self.current_sim_time = 0.0
        self.last_cmd_sim_time = 0.0

        # Latch collision giữa hai lần publish để không bỏ sót contact chỉ tồn
        # tại một vài physics step.
        self.collision_latched = False
        self.max_contact_count = 0
        self.collision_pairs = set()
        self.last_published_collision = False

        self.get_logger().info("Subscribe: /cmd_vel")
        self.get_logger().info("Publish: /clock")
        self.get_logger().info("Publish: /joint_states")
        self.get_logger().info(
            f"Publish: {GROUND_TRUTH_ODOM_TOPIC} (evaluation only)"
        )
        self.get_logger().info(
            f"Publish: {WHEEL_ODOM_TOPIC} (EKF input)"
        )
        self.get_logger().info("Publish: /imu")
        self.get_logger().info(
            "Publish: /scan_raw (360 deg) and "
            "/scan (InPlace -60...+60 deg)"
        )
        self.get_logger().info("Publish: /collision_state")
        self.get_logger().info("Publish: /contact_count")
        self.get_logger().info("Publish: /domain_randomization/state")
        self.get_logger().info("Service: /reset_simulation")
        pacing_label = (
            "uncapped"
            if self.real_time_factor == 0.0
            else f"{self.real_time_factor:.3f}x"
        )
        self.get_logger().info(
            f"Simulation mode: headless={self.headless}, "
            f"pacing={pacing_label}"
        )

    def update_sim_time(self, sim_time: float) -> None:
        """Expose the latest MuJoCo time to callback-side safety logic."""
        self.current_sim_time = float(sim_time)

    def attach_simulation(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        left_joint_id: int,
        right_joint_id: int,
    ) -> None:
        """Attach the settled simulator state used for episode resets."""
        self.sim_model = model
        self.sim_data = data
        self.initial_qpos = np.asarray(
            data.qpos,
            dtype=np.float64,
        ).copy()
        self.sim_left_joint_id = int(left_joint_id)
        self.sim_right_joint_id = int(right_joint_id)
        self.update_sim_time(float(data.time))
        self.last_cmd_sim_time = self.current_sim_time

        initial_domain_state = self.domain_randomizer.initial_state(
            sim_time=float(data.time),
        )
        self.publish_domain_randomization_state(initial_domain_state)

        self.simulation_ready = True

        self.get_logger().info(
            "Episode reset state captured; /reset_simulation is ready"
        )

    def publish_domain_randomization_state(
        self,
        state: dict,
    ) -> None:
        message = String()
        message.data = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.domain_randomization_state_publisher.publish(message)

    def reset_simulation_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        """Reset MuJoCo and bridge state without moving ROS time backwards."""
        del request

        if (
            not self.simulation_ready
            or self.sim_model is None
            or self.sim_data is None
            or self.initial_qpos is None
            or self.sim_left_joint_id is None
            or self.sim_right_joint_id is None
        ):
            response.success = False
            response.message = "MuJoCo is not ready for reset"
            return response

        try:
            model = self.sim_model
            data = self.sim_data

            # ROS nodes using /clock do not tolerate a backward time jump well.
            # Reset all MuJoCo state, restore the settled robot pose, then keep
            # the monotonically increasing simulation timestamp.
            preserved_sim_time = float(data.time)
            mujoco.mj_resetData(model, data)
            data.qpos[:] = self.initial_qpos
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            data.time = preserved_sim_time

            next_generation = self.reset_generation + 1
            domain_state = self.domain_randomizer.reset_episode(
                generation=next_generation,
                sim_time=preserved_sim_time,
            )

            mujoco.mj_forward(model, data)

            self.target_vx = 0.0
            self.target_wz = 0.0
            self.pending_commands.clear()
            self.received_cmd = False
            self.cmd_timeout_reported = False
            self.update_sim_time(preserved_sim_time)
            self.last_cmd_sim_time = preserved_sim_time

            self.collision_latched = False
            self.max_contact_count = 0
            self.collision_pairs.clear()
            self.last_published_collision = False

            self.reset_wheel_odometry(
                model=model,
                data=data,
                left_joint_id=self.sim_left_joint_id,
                right_joint_id=self.sim_right_joint_id,
                sim_time=preserved_sim_time,
            )

            self.reset_generation = next_generation
            self.publish_domain_randomization_state(domain_state)
            response.success = True
            response.message = (
                "MuJoCo, wheel odometry, command, and collision state reset; "
                f"generation={self.reset_generation}"
            )
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = f"Simulation reset failed: {exc}"
            self.get_logger().error(response.message)

        return response

    def cmd_vel_callback(self, msg: Twist) -> None:
        vx = float(msg.linear.x)
        wz = float(msg.angular.z)

        if not math.isfinite(vx) or not math.isfinite(wz):
            self.get_logger().error(
                "Bỏ qua /cmd_vel vì có NaN hoặc Inf"
            )
            return

        received_at = self.current_sim_time
        apply_at = received_at + self.domain_randomizer.command_latency_s
        self.pending_commands.append((apply_at, received_at, vx, wz))
        self.cmd_timeout_reported = False

    def get_command(self) -> Tuple[float, float]:
        while (
            self.pending_commands
            and self.pending_commands[0][0] <= self.current_sim_time
        ):
            _apply_at, received_at, vx, wz = self.pending_commands.popleft()
            self.target_vx = vx
            self.target_wz = wz
            self.received_cmd = True
            self.last_cmd_sim_time = received_at

        if not self.received_cmd:
            return 0.0, 0.0

        command_age = max(
            0.0,
            self.current_sim_time - self.last_cmd_sim_time,
        )

        if command_age > CMD_TIMEOUT:
            if not self.cmd_timeout_reported:
                self.get_logger().warn(
                    f"/cmd_vel timeout sau "
                    f"{command_age:.3f} s. Dừng robot."
                )
                self.cmd_timeout_reported = True

            return 0.0, 0.0

        return self.domain_randomizer.apply_command(
            self.target_vx,
            self.target_wz,
        )

    def publish_clock(self, sim_time: float) -> None:
        msg = Clock()
        msg.clock = seconds_to_ros_time(sim_time)
        self.clock_publisher.publish(msg)

    def update_collision_monitor(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        collision, contact_count, contact_pairs = (
            detect_navigation_contacts(model, data)
        )

        if collision:
            self.collision_latched = True
            self.max_contact_count = max(
                self.max_contact_count,
                contact_count,
            )
            self.collision_pairs.update(contact_pairs)

    def publish_collision_monitor(self) -> None:
        collision_msg = Bool()
        collision_msg.data = bool(self.collision_latched)
        self.collision_state_publisher.publish(collision_msg)

        contact_msg = Int32()
        contact_msg.data = int(self.max_contact_count)
        self.contact_count_publisher.publish(contact_msg)

        if (
            self.collision_latched
            and not self.last_published_collision
        ):
            pair_text = ", ".join(
                f"{robot}<->{environment}"
                for robot, environment in sorted(
                    self.collision_pairs
                )
            )
            self.get_logger().warn(
                "Phát hiện va chạm Nav2: "
                f"contacts={self.max_contact_count}; "
                f"pairs={pair_text or 'không rõ'}"
            )
        elif (
            not self.collision_latched
            and self.last_published_collision
        ):
            self.get_logger().info(
                "Robot đã hết tiếp xúc với vật cản."
            )

        self.last_published_collision = bool(
            self.collision_latched
        )
        self.collision_latched = False
        self.max_contact_count = 0
        self.collision_pairs.clear()

    def publish_joint_states(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stamp: RosTime,
        left_joint_id: int,
        right_joint_id: int,
    ) -> None:
        left_qpos_address = int(
            model.jnt_qposadr[left_joint_id]
        )
        right_qpos_address = int(
            model.jnt_qposadr[right_joint_id]
        )

        left_dof_address = int(
            model.jnt_dofadr[left_joint_id]
        )
        right_dof_address = int(
            model.jnt_dofadr[right_joint_id]
        )

        msg = JointState()
        msg.header.stamp = stamp

        msg.name = [
            LEFT_JOINT,
            RIGHT_JOINT,
        ]

        msg.position = [
            float(data.qpos[left_qpos_address]),
            float(data.qpos[right_qpos_address]),
        ]

        msg.velocity = [
            float(data.qvel[left_dof_address]),
            float(data.qvel[right_dof_address]),
        ]

        # Moment thực tế tác dụng lên DOF của từng bánh.
        msg.effort = [
            float(data.qfrc_actuator[left_dof_address]),
            float(data.qfrc_actuator[right_dof_address]),
        ]

        self.joint_state_publisher.publish(msg)

    def reset_wheel_odometry(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        left_joint_id: int,
        right_joint_id: int,
        sim_time: float,
    ) -> None:
        left_address = int(model.jnt_qposadr[left_joint_id])
        right_address = int(model.jnt_qposadr[right_joint_id])

        self.wheel_odom_x = 0.0
        self.wheel_odom_y = 0.0
        self.wheel_odom_yaw = 0.0
        self.last_left_wheel_angle = float(data.qpos[left_address])
        self.last_right_wheel_angle = float(data.qpos[right_address])
        self.last_wheel_odom_time = float(sim_time)
        self.wheel_odom_initialized = True

        self.get_logger().info(
            "Wheel odometry reset tại x=0, y=0, yaw=0"
        )

    def publish_wheel_odometry(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stamp: RosTime,
        sim_time: float,
        left_joint_id: int,
        right_joint_id: int,
    ) -> None:
        if not self.wheel_odom_initialized:
            self.reset_wheel_odometry(
                model=model,
                data=data,
                left_joint_id=left_joint_id,
                right_joint_id=right_joint_id,
                sim_time=sim_time,
            )

        left_address = int(model.jnt_qposadr[left_joint_id])
        right_address = int(model.jnt_qposadr[right_joint_id])
        left_angle = float(data.qpos[left_address])
        right_angle = float(data.qpos[right_address])
        delta_left_angle = left_angle - self.last_left_wheel_angle
        delta_right_angle = right_angle - self.last_right_wheel_angle
        delta_time = float(sim_time) - self.last_wheel_odom_time

        if delta_time <= 0.0:
            return

        (
            self.wheel_odom_x,
            self.wheel_odom_y,
            self.wheel_odom_yaw,
            delta_distance,
            delta_yaw,
        ) = integrate_differential_drive(
            x=self.wheel_odom_x,
            y=self.wheel_odom_y,
            yaw=self.wheel_odom_yaw,
            delta_left_angle=delta_left_angle,
            delta_right_angle=delta_right_angle,
            linear_scale=self.domain_randomizer.odom_linear_scale,
            angular_scale=self.domain_randomizer.odom_angular_scale,
        )

        self.last_left_wheel_angle = left_angle
        self.last_right_wheel_angle = right_angle
        self.last_wheel_odom_time = float(sim_time)

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = ODOM_FRAME
        msg.child_frame_id = BASE_FRAME

        msg.pose.pose.position.x = self.wheel_odom_x
        msg.pose.pose.position.y = self.wheel_odom_y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(
            0.5 * self.wheel_odom_yaw
        )
        msg.pose.pose.orientation.w = math.cos(
            0.5 * self.wheel_odom_yaw
        )
        msg.pose.covariance = create_wheel_pose_covariance()

        msg.twist.twist.linear.x = delta_distance / delta_time
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = delta_yaw / delta_time
        msg.twist.covariance = create_wheel_twist_covariance()

        self.wheel_odom_publisher.publish(msg)


    def publish_imu(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stamp: RosTime,
        gyro_sensor_id: int,
        accelerometer_sensor_id: int,
        orientation_sensor_id: int,
    ) -> None:
        gyro = read_mujoco_sensor(
            model,
            data,
            gyro_sensor_id,
        )

        acceleration = read_mujoco_sensor(
            model,
            data,
            accelerometer_sensor_id,
        )

        quaternion = read_mujoco_sensor(
            model,
            data,
            orientation_sensor_id,
        )

        if gyro.shape != (3,):
            raise RuntimeError(
                f"imu_gyro có shape sai: {gyro.shape}"
            )

        if acceleration.shape != (3,):
            raise RuntimeError(
                "imu_accelerometer có shape sai: "
                f"{acceleration.shape}"
            )

        if quaternion.shape != (4,):
            raise RuntimeError(
                "imu_orientation có shape sai: "
                f"{quaternion.shape}"
            )

        # MuJoCo quaternion: w, x, y, z
        # ROS quaternion:     x, y, z, w
        qw = float(quaternion[0])
        qx = float(quaternion[1])
        qy = float(quaternion[2])
        qz = float(quaternion[3])

        quaternion_norm = math.sqrt(
            qw * qw
            + qx * qx
            + qy * qy
            + qz * qz
        )

        if quaternion_norm <= 1.0e-12:
            self.get_logger().error(
                "Quaternion IMU không hợp lệ"
            )
            return

        qw /= quaternion_norm
        qx /= quaternion_norm
        qy /= quaternion_norm
        qz /= quaternion_norm

        msg = Imu()

        msg.header.stamp = stamp
        msg.header.frame_id = IMU_FRAME

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        msg.orientation_covariance = (
            create_imu_orientation_covariance()
        )

        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])
        msg.angular_velocity_covariance = (
            create_imu_angular_velocity_covariance()
        )

        msg.linear_acceleration.x = float(
            acceleration[0]
        )
        msg.linear_acceleration.y = float(
            acceleration[1]
        )
        msg.linear_acceleration.z = float(
            acceleration[2]
        )
        msg.linear_acceleration_covariance = (
            create_imu_linear_acceleration_covariance()
        )

        self.imu_publisher.publish(msg)

    def publish_laser_scan(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stamp: RosTime,
        lidar_site_id: int,
    ) -> None:
        """Ray-cast một vòng quét 2D và publish sensor_msgs/LaserScan."""
        ray_origin = np.asarray(
            data.site_xpos[lidar_site_id],
            dtype=np.float64,
        ).copy()

        # site_xmat biến vector từ frame lidar sang world MuJoCo.
        lidar_rotation = np.asarray(
            data.site_xmat[lidar_site_id],
            dtype=np.float64,
        ).reshape(3, 3)

        raw_ranges = []
        hit_geom_id = np.array([-1], dtype=np.int32)
        ray_api_has_normal = True

        for ray_index in range(LIDAR_NUM_RAYS):
            angle = (
                LIDAR_ANGLE_MIN
                + ray_index * LIDAR_ANGLE_INCREMENT
            )

            direction_lidar = np.array(
                [math.cos(angle), math.sin(angle), 0.0],
                dtype=np.float64,
            )
            direction_world = lidar_rotation @ direction_lidar

            hit_geom_id[0] = -1

            # MuJoCo 3.x có thêm đối số normal; một số bản 2.x chưa có.
            if ray_api_has_normal:
                try:
                    distance = mujoco.mj_ray(
                        model,
                        data,
                        ray_origin,
                        direction_world,
                        LIDAR_GEOM_GROUPS,
                        1,
                        -1,
                        hit_geom_id,
                        None,
                    )
                except TypeError:
                    ray_api_has_normal = False
                    distance = mujoco.mj_ray(
                        model,
                        data,
                        ray_origin,
                        direction_world,
                        LIDAR_GEOM_GROUPS,
                        1,
                        -1,
                        hit_geom_id,
                    )
            else:
                distance = mujoco.mj_ray(
                    model,
                    data,
                    ray_origin,
                    direction_world,
                    LIDAR_GEOM_GROUPS,
                    1,
                    -1,
                    hit_geom_id,
                )

            distance = float(distance)

            if (
                distance < LIDAR_RANGE_MIN
                or distance > LIDAR_RANGE_MAX
            ):
                raw_ranges.append(float("inf"))
            else:
                raw_ranges.append(distance)

        raw_ranges = self.domain_randomizer.apply_lidar(
            np.asarray(raw_ranges, dtype=np.float64),
            range_min=LIDAR_RANGE_MIN,
            range_max=LIDAR_RANGE_MAX,
        ).tolist()

        scan_time = float(1.0 / LIDAR_RATE)
        time_increment = float(scan_time / LIDAR_NUM_RAYS)

        raw_msg = LaserScan()
        raw_msg.header.stamp = stamp
        raw_msg.header.frame_id = LIDAR_FRAME
        raw_msg.angle_min = float(LIDAR_ANGLE_MIN)
        raw_msg.angle_increment = float(LIDAR_ANGLE_INCREMENT)
        raw_msg.angle_max = float(
            LIDAR_ANGLE_MIN
            + (LIDAR_NUM_RAYS - 1) * LIDAR_ANGLE_INCREMENT
        )
        raw_msg.scan_time = scan_time
        raw_msg.time_increment = time_increment
        raw_msg.range_min = float(LIDAR_RANGE_MIN)
        raw_msg.range_max = float(LIDAR_RANGE_MAX)
        raw_msg.ranges = raw_ranges
        raw_msg.intensities = []
        self.raw_scan_publisher.publish(raw_msg)

        # Mô phỏng đúng LaserScanAngularBoundsFilterInPlace trên robot thật:
        # giữ nguyên 720 phần tử và metadata scan gốc; các tia ngoài
        # -60...+60 độ được đánh dấu NaN.
        filtered_ranges = [float("nan")] * LIDAR_NUM_RAYS
        filtered_ranges[
            LIDAR_FILTER_START_INDEX:LIDAR_FILTER_END_INDEX + 1
        ] = raw_ranges[
            LIDAR_FILTER_START_INDEX:LIDAR_FILTER_END_INDEX + 1
        ]

        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = LIDAR_FRAME
        msg.angle_min = float(LIDAR_ANGLE_MIN)
        msg.angle_increment = float(LIDAR_ANGLE_INCREMENT)
        msg.angle_max = float(
            LIDAR_ANGLE_MIN
            + (LIDAR_NUM_RAYS - 1) * LIDAR_ANGLE_INCREMENT
        )

        msg.scan_time = scan_time
        # AngularBoundsFilter giữ time_increment của scan 720 mẫu gốc.
        msg.time_increment = time_increment
        msg.range_min = float(LIDAR_RANGE_MIN)
        msg.range_max = float(LIDAR_RANGE_MAX)
        msg.ranges = filtered_ranges
        msg.intensities = []

        self.scan_publisher.publish(msg)

    def broadcast_odom_transform(
        self,
        stamp: RosTime,
        odom_x: float,
        odom_y: float,
        odom_yaw: float,
    ) -> None:
        if not self.publish_odom_tf_enabled:
            return

        transform = TransformStamped()

        transform.header.stamp = stamp
        transform.header.frame_id = ODOM_FRAME
        transform.child_frame_id = BASE_FRAME

        transform.transform.translation.x = odom_x
        transform.transform.translation.y = odom_y
        transform.transform.translation.z = 0.0

        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = math.sin(
            odom_yaw * 0.5
        )
        transform.transform.rotation.w = math.cos(
            odom_yaw * 0.5
        )

        self.tf_broadcaster.sendTransform(
            transform
        )
    

    def publish_ground_truth_odometry(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        stamp: RosTime,
        base_site_id: int,
        origin_x: float,
        origin_y: float,
        origin_yaw: float,
    ) -> None:
        site_position = data.site_xpos[base_site_id]

        current_x = float(site_position[0])
        current_y = float(site_position[1])
        current_yaw = site_yaw(data, base_site_id)

        delta_x_world = current_x - origin_x
        delta_y_world = current_y - origin_y

        cos_origin = math.cos(origin_yaw)
        sin_origin = math.sin(origin_yaw)

        odom_x = (
            cos_origin * delta_x_world
            + sin_origin * delta_y_world
        )

        odom_y = (
            -sin_origin * delta_x_world
            + cos_origin * delta_y_world
        )

        odom_yaw = normalize_angle(
            current_yaw - origin_yaw
        )

        # MuJoCo trả velocity 6D theo thứ tự:
        # angular xyz, linear xyz.
        # flg_local=1: biểu diễn trong frame local của site.
        velocity_6d = np.zeros(6, dtype=np.float64)

        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_SITE,
            base_site_id,
            velocity_6d,
            1,
        )

        angular_x = float(velocity_6d[0])
        angular_y = float(velocity_6d[1])
        angular_z = float(velocity_6d[2])

        linear_x = float(velocity_6d[3])
        linear_y = float(velocity_6d[4])
        linear_z = float(velocity_6d[5])

        msg = Odometry()

        msg.header.stamp = stamp
        msg.header.frame_id = ODOM_FRAME
        msg.child_frame_id = BASE_FRAME

        msg.pose.pose.position.x = odom_x
        msg.pose.pose.position.y = odom_y
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(
            odom_yaw * 0.5
        )
        msg.pose.pose.orientation.w = math.cos(
            odom_yaw * 0.5
        )

        msg.pose.covariance = create_pose_covariance()

        msg.twist.twist.linear.x = linear_x
        msg.twist.twist.linear.y = linear_y
        msg.twist.twist.linear.z = 0.0

        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = angular_z

        msg.twist.covariance = create_twist_covariance()

        self.ground_truth_odom_publisher.publish(msg)
        self.broadcast_odom_transform(
            stamp=stamp,
            odom_x=odom_x,
            odom_y=odom_y,
            odom_yaw=odom_yaw,
        )


def main(args=None) -> int:
    rclpy.init(args=args)

    try:
        node = MujocoRosBridge()
    except Exception as exc:
        print(f"Invalid simulator configuration: {exc}")
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    try:
        package_share = get_package_share_directory(
            PACKAGE_NAME
        )

        model_path = os.path.join(
            package_share,
            "models",
            node.model_file,
        )

        model = mujoco.MjModel.from_xml_path(
            model_path
        )

        data = mujoco.MjData(model)

        left_actuator_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            LEFT_ACTUATOR,
        )

        right_actuator_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            RIGHT_ACTUATOR,
        )

        left_joint_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            LEFT_JOINT,
        )

        right_joint_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            RIGHT_JOINT,
        )

        base_site_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            BASE_FOOTPRINT_SITE,
        )

        gyro_sensor_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            IMU_GYRO_SENSOR,
        )

        accelerometer_sensor_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            IMU_ACCEL_SENSOR,
        )

        orientation_sensor_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            IMU_ORIENTATION_SENSOR,
        )

        lidar_site_id = get_named_id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            LIDAR_SITE,
        )

    except Exception as exc:
        node.get_logger().error(str(exc))
        node.destroy_node()
        rclpy.shutdown()
        return 1

    node.get_logger().info(
        f"Đã load model: {model_path}"
    )

    node.get_logger().info(
        f"IMU sensor IDs: "
        f"gyro={gyro_sensor_id}, "
        f"accel={accelerometer_sensor_id}, "
        f"quat={orientation_sensor_id}"
    )

    node.get_logger().info(
        f"LiDAR site ID: {lidar_site_id}, "
        f"rays={LIDAR_NUM_RAYS}, rate={LIDAR_RATE:.1f} Hz"
    )

    # Cho robot rơi xuống và ổn định trước khi đặt gốc odom.
    node.get_logger().info(
        f"Ổn định robot trong {SETTLE_TIME:.1f} s..."
    )

    while data.time < SETTLE_TIME:
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

    # Xóa vận tốc dư sau quá trình rơi.
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.time = 0.0

    mujoco.mj_forward(model, data)

    initial_site_position = (
        data.site_xpos[base_site_id].copy()
    )

    origin_x = float(initial_site_position[0])
    origin_y = float(initial_site_position[1])
    origin_yaw = site_yaw(data, base_site_id)

    node.get_logger().info(
        f"Odom origin: "
        f"x={origin_x:.4f}, "
        f"y={origin_y:.4f}, "
        f"yaw={math.degrees(origin_yaw):.3f} deg"
    )

    node.reset_wheel_odometry(
        model=model,
        data=data,
        left_joint_id=left_joint_id,
        right_joint_id=right_joint_id,
        sim_time=0.0,
    )

    node.attach_simulation(
        model=model,
        data=data,
        left_joint_id=left_joint_id,
        right_joint_id=right_joint_id,
    )

    physics_timestep = float(model.opt.timestep)

    clock_step_interval = max(
        1,
        round(1.0 / (CLOCK_RATE * physics_timestep)),
    )

    motor_step_interval = max(
        1,
        round(1.0 / (MOTOR_COMMAND_RATE * physics_timestep)),
    )

    state_step_interval = max(
        1,
        round(1.0 / (STATE_RATE * physics_timestep)),
    )

    imu_step_interval = max(
        1,
        round(1.0 / (IMU_RATE * physics_timestep)),
    )

    lidar_step_interval = max(
        1,
        round(1.0 / (LIDAR_RATE * physics_timestep)),
    )

    step_count = 0
    last_log_wall_time = 0.0
    last_viewer_sync_wall_time = 0.0
    vx = 0.0
    wz = 0.0
    omega_left = 0.0
    omega_right = 0.0
    observed_reset_generation = node.reset_generation

    node.get_logger().info(
        f"Physics timestep: {physics_timestep:.6f} s"
    )

    node.get_logger().info(
        f"Publish intervals: "
        f"clock={clock_step_interval} steps, "
        f"motor={motor_step_interval} steps, "
        f"state={state_step_interval} steps, "
        f"imu={imu_step_interval} steps, "
        f"lidar={lidar_step_interval} steps; "
        f"viewer capped at {VIEWER_RATE:.1f} Hz wall time"
    )

    try:
        viewer_context = nullcontext(None)

        if not node.headless:
            try:
                import mujoco.viewer as mujoco_viewer
            except ImportError as exc:
                raise RuntimeError(
                    "MuJoCo viewer is unavailable; run with headless:=true "
                    "or install the viewer dependencies"
                ) from exc

            viewer_context = mujoco_viewer.launch_passive(
                model,
                data,
            )

        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = [0.0, 0.0, 0.15]
                viewer.cam.distance = 2.2
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -25.0

            wall_anchor = time.perf_counter()
            sim_anchor = float(data.time)
            last_log_wall_time = wall_anchor
            last_viewer_sync_wall_time = (
                wall_anchor - 1.0 / VIEWER_RATE
            )
            node.update_sim_time(sim_anchor)

            while rclpy.ok():
                if viewer is not None and not viewer.is_running():
                    break

                # Process ROS 2 callbacks without tying physics to wall time.
                rclpy.spin_once(
                    node,
                    timeout_sec=0.0,
                )

                if node.reset_generation != observed_reset_generation:
                    # Do not let a wheel command sampled before reset survive
                    # until the next 20 Hz motor command boundary.
                    omega_left = 0.0
                    omega_right = 0.0
                    observed_reset_generation = node.reset_generation

                # Match the real driver cycle: sample the latest command at
                # 20 Hz simulation time and hold it between two updates.
                if step_count % motor_step_interval == 0:
                    vx, wz = node.get_command()
                    omega_left, omega_right = (
                        cmd_vel_to_wheels(vx, wz)
                    )

                data.ctrl[left_actuator_id] = omega_left
                data.ctrl[right_actuator_id] = omega_right

                mujoco.mj_step(model, data)

                node.update_collision_monitor(
                    model=model,
                    data=data,
                )

                step_count += 1
                sim_time = float(data.time)
                node.update_sim_time(sim_time)

                if step_count % clock_step_interval == 0:
                    node.publish_clock(sim_time)

                if step_count % state_step_interval == 0:
                    stamp = seconds_to_ros_time(sim_time)

                    node.publish_joint_states(
                        model=model,
                        data=data,
                        stamp=stamp,
                        left_joint_id=left_joint_id,
                        right_joint_id=right_joint_id,
                    )

                    node.publish_ground_truth_odometry(
                        model=model,
                        data=data,
                        stamp=stamp,
                        base_site_id=base_site_id,
                        origin_x=origin_x,
                        origin_y=origin_y,
                        origin_yaw=origin_yaw,
                    )

                    node.publish_wheel_odometry(
                        model=model,
                        data=data,
                        stamp=stamp,
                        sim_time=sim_time,
                        left_joint_id=left_joint_id,
                        right_joint_id=right_joint_id,
                    )

                    node.publish_collision_monitor()

                if step_count % imu_step_interval == 0:
                    imu_stamp = seconds_to_ros_time(sim_time)

                    node.publish_imu(
                        model=model,
                        data=data,
                        stamp=imu_stamp,
                        gyro_sensor_id=gyro_sensor_id,
                        accelerometer_sensor_id=(
                            accelerometer_sensor_id
                        ),
                        orientation_sensor_id=(
                            orientation_sensor_id
                        ),
                    )

                if step_count % lidar_step_interval == 0:
                    scan_stamp = seconds_to_ros_time(sim_time)

                    node.publish_laser_scan(
                        model=model,
                        data=data,
                        stamp=scan_stamp,
                        lidar_site_id=lidar_site_id,
                    )

                current_wall_time = time.perf_counter()

                if (
                    viewer is not None
                    and current_wall_time - last_viewer_sync_wall_time
                    >= 1.0 / VIEWER_RATE
                ):
                    viewer.sync()
                    last_viewer_sync_wall_time = current_wall_time

                if current_wall_time - last_log_wall_time >= 1.0:
                    wall_elapsed = current_wall_time - wall_anchor
                    achieved_rtf = (
                        (sim_time - sim_anchor) / wall_elapsed
                        if wall_elapsed > 0.0
                        else 0.0
                    )
                    node.get_logger().info(
                        f"sim_t={sim_time:7.2f} "
                        f"rtf={achieved_rtf:6.2f} "
                        f"cmd=({vx:5.2f}, {wz:5.2f}) "
                        f"wheel=({omega_left:5.2f}, "
                        f"{omega_right:5.2f})"
                    )
                    last_log_wall_time = current_wall_time

                # Pace against an absolute anchor to avoid accumulated sleep
                # error.  A factor of 0 disables pacing (maximum throughput).
                if node.real_time_factor > 0.0:
                    target_wall_time = (
                        wall_anchor
                        + (sim_time - sim_anchor)
                        / node.real_time_factor
                    )
                    sleep_time = target_wall_time - time.perf_counter()

                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

    except KeyboardInterrupt:
        node.get_logger().info(
            "Dừng bằng Ctrl+C."
        )

    except Exception as exc:
        node.get_logger().error(
            f"Lỗi vòng lặp MuJoCo: {exc}"
        )
        return_code = 1

    else:
        return_code = 0

    finally:
        data.ctrl[:] = 0.0
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
