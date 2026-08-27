"""Start Nav2 plus the bounded SAC-to-MPPI speed supervisor."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("agv_mujoco_sim")
    navigation_launch = PathJoinSubstitution(
        [package_share, "launch", "navigation.launch.py"]
    )
    default_nav2_params = PathJoinSubstitution(
        [package_share, "config", "nav2_params.yaml"]
    )
    default_supervisor_params = PathJoinSubstitution(
        [package_share, "config", "rl_supervisor.yaml"]
    )

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    use_composition = LaunchConfiguration("use_composition")
    use_respawn = LaunchConfiguration("use_respawn")
    log_level = LaunchConfiguration("log_level")

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_nav2_params,
                description="Nav2 MPPI parameter file.",
            ),
            DeclareLaunchArgument("use_composition", default_value="False"),
            DeclareLaunchArgument("use_respawn", default_value="False"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument(
                "enable_rl_supervisor",
                default_value="true",
                description="Start the bounded SAC-to-MPPI speed adapter.",
            ),
            DeclareLaunchArgument(
                "supervisor_params_file",
                default_value=default_supervisor_params,
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(navigation_launch),
                launch_arguments={
                    "namespace": namespace,
                    "use_sim_time": use_sim_time,
                    "autostart": autostart,
                    "params_file": params_file,
                    "use_composition": use_composition,
                    "use_respawn": use_respawn,
                    "log_level": log_level,
                }.items(),
            ),
            Node(
                condition=IfCondition(
                    LaunchConfiguration("enable_rl_supervisor")
                ),
                package="agv_mujoco_sim",
                executable="rl_velocity_supervisor",
                name="rl_velocity_supervisor",
                namespace=namespace,
                output="screen",
                parameters=[
                    LaunchConfiguration("supervisor_params_file"),
                    {
                        "use_sim_time": ParameterValue(
                            use_sim_time,
                            value_type=bool,
                        )
                    },
                ],
                arguments=["--ros-args", "--log-level", log_level],
            ),
        ]
    )
