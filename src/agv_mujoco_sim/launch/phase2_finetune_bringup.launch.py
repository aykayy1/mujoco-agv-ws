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
        "navigation_rl.launch.py",
    )
    default_map_file = os.path.join(package_share, "maps", "rl_trainingv2.yaml")
    params_file = os.path.join(
        package_share,
        "config",
        "nav2_params.yaml",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    real_time_factor = LaunchConfiguration("real_time_factor")
    enable_rl_supervisor = LaunchConfiguration("enable_rl_supervisor")
    front_safety_stop_enabled = LaunchConfiguration("front_safety_stop_enabled")

    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(simulator_launch),
        launch_arguments={
            "model_file": LaunchConfiguration("model_file"),
            "headless": headless,
            "real_time_factor": real_time_factor,
            "front_safety_stop_enabled": front_safety_stop_enabled,
            "domain_randomization": LaunchConfiguration("domain_randomization"),
            "domain_randomization_seed": LaunchConfiguration("domain_randomization_seed"),
            "dr_profile": LaunchConfiguration("dr_profile"),
            "dr_stage": LaunchConfiguration("dr_stage"),
            "dr_nominal_probability": LaunchConfiguration("dr_nominal_probability"),
        }.items(),
    )
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch),
        launch_arguments={
            "map": LaunchConfiguration("map_file"),
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
            "enable_rl_supervisor": enable_rl_supervisor,
            "supervisor_params_file": LaunchConfiguration("supervisor_params_file"),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("model_file", default_value="rl_trainingworld.xml"),
            DeclareLaunchArgument("map_file", default_value=default_map_file),
            DeclareLaunchArgument("domain_randomization", default_value="false"),
            DeclareLaunchArgument("domain_randomization_seed", default_value="42"),
            DeclareLaunchArgument("dr_profile", default_value="light", choices=["light", "medium", "strong", "custom"]),
            DeclareLaunchArgument("dr_stage", default_value="lidar", choices=["lidar", "lidar_odom", "all"]),
            DeclareLaunchArgument("dr_nominal_probability", default_value="0.2"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("real_time_factor", default_value="1.0"),
            DeclareLaunchArgument(
                "enable_rl_supervisor",
                default_value="true",
                description=(
                    "Enable for 1D SAC training/deploy; disable for isolated "
                    "MPPI/SpeedLimit benchmark arms."
                ),
            ),
            DeclareLaunchArgument(
                "front_safety_stop_enabled",
                default_value=enable_rl_supervisor,
                description="Simulation-only last-resort stop; set false for SAC-only ablation.",
            ),
            DeclareLaunchArgument(
                "supervisor_params_file",
                default_value=os.path.join(package_share, "config", "rl_supervisor.yaml"),
            ),
            simulator,
            TimerAction(period=2.0, actions=[localization]),
            TimerAction(period=4.0, actions=[navigation]),
        ]
    )
