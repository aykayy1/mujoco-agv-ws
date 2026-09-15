"""Start Nav2 map server and AMCL using the package's Humble parameters."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("agv_mujoco_sim")
    nav2_bringup_share = FindPackageShare("nav2_bringup")

    # Đường dẫn tới file config
    default_params_file = PathJoinSubstitution(
        [package_share, "config", "nav2_params.yaml"]
    )
    
    # THÊM MỚI: Đường dẫn mặc định tới file bản đồ trong thư mục "map"
    default_map_file = PathJoinSubstitution(
        [package_share, "maps", "rl_trainingv2.yaml"]
    )

    stock_localization_launch = PathJoinSubstitution(
        [nav2_bringup_share, "launch", "localization_launch.py"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=default_map_file,  # Cập nhật default_value tại đây
                description="Path to the map YAML file.",
            ),
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
                PythonLaunchDescriptionSource(stock_localization_launch),
                launch_arguments={
                    "map": LaunchConfiguration("map"),
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