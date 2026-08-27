cmake_minimum_required(VERSION 3.10)
project(agv_mujoco_sim LANGUAGES NONE)


# Register this plain-CMake package in the ROS 2 ament resource index.
# This does not use ament_cmake; it only makes the installed package
# discoverable by ros2 run/launch and ament_index_python.
install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Pure-CMake packages do not automatically prepend their install prefix to
# AMENT_PREFIX_PATH.  colcon.pkg registers this installed .dsv file as an
# environment hook, so sourcing install/setup.bash makes ROS 2 discover the
# package through the ament resource index.
install(
  FILES hooks/ament_prefix_path.dsv
  DESTINATION share/${PROJECT_NAME}/hook
)

install(
  DIRECTORY
    models
    meshes
    urdf
    launch
    config
    maps
    transfer_models
  DESTINATION share/${PROJECT_NAME}
)

# ROS runtime nodes. Keep stable executable names for ros2 run/launch.
install(
  PROGRAMS scripts/mujoco_cmd_vel_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME mujoco_cmd_vel_bridge
)

install(
  PROGRAMS scripts/rl_velocity_supervisor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME rl_velocity_supervisor
)

# Keep these files together and preserve the .py names because train_sac.py
# imports AgvRlEnv from agv_rl_env.py in the same directory.
install(
  PROGRAMS
    scripts/agv_rl_env.py
    scripts/train_sac.py
    scripts/check_runtime.py
    scripts/measure_issue02_baseline.py
    scripts/domain_randomizer.py
    scripts/measure_issue03_sampling.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES requirements.txt
  DESTINATION share/${PROJECT_NAME}
)
