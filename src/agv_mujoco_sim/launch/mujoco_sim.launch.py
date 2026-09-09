#!/usr/bin/env python3

import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    model_file = LaunchConfiguration("model_file")
    headless = LaunchConfiguration("headless")
    real_time_factor = LaunchConfiguration("real_time_factor")
    domain_randomization = LaunchConfiguration(
        "domain_randomization"
    )
    domain_randomization_seed = LaunchConfiguration(
        "domain_randomization_seed"
    )

    package_share = get_package_share_directory(
        "agv_mujoco_sim"
    )

    urdf_path = os.path.join(
        package_share,
        "urdf",
        "agv0509test6.urdf",
    )

    ekf_params_path = os.path.join(
        package_share,
        "config",
        "ekf_mujoco.yaml",
    )

    physical_params_path = os.path.join(
        package_share,
        "config",
        "agv_physical_params.yaml",
    )

    domain_randomization_params_path = os.path.join(
        package_share,
        "config",
        "domain_randomization.yaml",
    )

    with open(
        urdf_path,
        "r",
        encoding="utf-8",
    ) as urdf_file:
        robot_description = urdf_file.read()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    mujoco_bridge = Node(
        package="agv_mujoco_sim",
        executable="mujoco_cmd_vel_bridge",
        name="mujoco_ros_bridge",
        output="screen",
        parameters=[
            physical_params_path,
            domain_randomization_params_path,
            {
                # EKF is the only publisher of odom -> base_foot_link.
                "publish_odom_tf": False,
                "model_file": ParameterValue(
                    model_file,
                    value_type=str,
                ),
                "headless": ParameterValue(
                    headless,
                    value_type=bool,
                ),
                "real_time_factor": ParameterValue(
                    real_time_factor,
                    value_type=float,
                ),
                "domain_randomization.enabled": ParameterValue(
                    domain_randomization,
                    value_type=bool,
                ),
                "domain_randomization.seed": ParameterValue(
                    domain_randomization_seed,
                    value_type=int,
                ),
            }
        ],
    )

    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_params_path],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_file",
            default_value="warehouse_20x20.xml",
            description=(
                "MuJoCo XML filename installed in the package models directory"
            ),
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Disable the MuJoCo viewer for server/training runs",
        ),
        DeclareLaunchArgument(
            "real_time_factor",
            default_value="1.0",
            description=(
                "Simulation speed relative to wall time; 0 means uncapped"
            ),
        ),
        DeclareLaunchArgument(
            "domain_randomization",
            default_value="false",
            description=(
                "Issue 03 behavior-randomization master switch; individual "
                "groups must also be enabled in the parameter file"
            ),
        ),
        DeclareLaunchArgument(
            "domain_randomization_seed",
            default_value="42",
            description="Deterministic RNG seed for episode sampling",
        ),
        robot_state_publisher,
        mujoco_bridge,
        ekf,
    ])
