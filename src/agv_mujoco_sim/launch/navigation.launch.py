"""Start the Nav2 navigation servers using the package's Humble parameters."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("agv_mujoco_sim")
    nav2_bringup_share = FindPackageShare("nav2_bringup")

    default_params_file = PathJoinSubstitution(
        [package_share, "config", "nav2_params.yaml"]
    )
    stock_navigation_launch = PathJoinSubstitution(
        [nav2_bringup_share, "launch", "navigation_launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="Nav2 parameter file.",
            ),
            DeclareLaunchArgument("use_composition", default_value="False"),
            DeclareLaunchArgument("use_respawn", default_value="False"),
            DeclareLaunchArgument("log_level", default_value="info"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(stock_navigation_launch),
                launch_arguments={
                    "namespace": LaunchConfiguration("namespace"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "autostart": LaunchConfiguration("autostart"),
                    "params_file": LaunchConfiguration("params_file"),
                    "use_composition": LaunchConfiguration("use_composition"),
                    "use_respawn": LaunchConfiguration("use_respawn"),
                    "log_level": LaunchConfiguration("log_level"),
                }.items(),
            ),
        ]
    )
