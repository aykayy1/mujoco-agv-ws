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
    behavior_trees
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
    scripts/front_safety_stop.py
    scripts/benchmark_mppi_baseline.py
    scripts/benchmark_dynamic_obstacle.py
    scripts/run_dynamic_benchmark_matrix.py
    scripts/run_mppi_three_maps.py
    scripts/train_sac.py
    scripts/evaluate_sac_dynamic.py
    scripts/resume_train.py
    scripts/deploy_sac1.py
    scripts/check_runtime.py
    scripts/domain_randomizer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES requirements.txt
  DESTINATION share/${PROJECT_NAME}
)

# This package uses plain CMake, so colcon does not discover the Python tests
# unless they are explicitly registered with CTest.
include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME agv_mujoco_sim_pytest
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts:$ENV{PYTHONPATH}"
      "PYTHONDONTWRITEBYTECODE=1"
      ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider
      ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
endif()
