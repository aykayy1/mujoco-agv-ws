"""Start the self-contained MuJoCo Phase-2 fine-tune arena and Nav2 stack."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_share = get_package_share_directory("agv_mujoco_sim")

    simulator_launch = os.path.join(
        package_share,
        "launch",
        "mujoco_sim.launch.py",
    )
    localization_launch = os.path.join(
        package_share,
        "launch",
        "localization.launch.py",
    )
    navigation_launch = os.path.join(
        package_share,
        "launch",
        "navigation.launch.py",
    )
    map_file = os.path.join(
        package_share,
        "maps",
        "transfer_combined_arena.yaml",
    )
    params_file = os.path.join(
        package_share,
        "config",
        "nav2_gazebo_transfer_params.yaml",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    real_time_factor = LaunchConfiguration("real_time_factor")

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(simulator_launch),
        launch_arguments={
            "model_file": "agv_spawn_transfer_arena.xml",
            "headless": headless,
            "real_time_factor": real_time_factor,
            "domain_randomization": "false",
        }.items(),
    )
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch),
        launch_arguments={
            "map": map_file,
            "params_file": params_file,
            "use_sim_time": use_sim_time,
            "autostart": "true",
        }.items(),
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch),
        launch_arguments={
            "params_file": params_file,
            "use_sim_time": use_sim_time,
            "autostart": "true",
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("real_time_factor", default_value="1.0"),
            simulator,
            TimerAction(period=2.0, actions=[localization]),
            TimerAction(period=4.0, actions=[navigation]),
        ]
    )
