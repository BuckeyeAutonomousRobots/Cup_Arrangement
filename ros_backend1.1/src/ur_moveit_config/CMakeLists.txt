cmake_minimum_required(VERSION 3.5)
project(ur_moveit_config)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(self_collision_inspector src/self_collision_inspector.cpp)
target_compile_features(self_collision_inspector PUBLIC cxx_std_17)
ament_target_dependencies(
  self_collision_inspector
  moveit_core
  moveit_ros_planning
  rclcpp
  sensor_msgs
)

install(DIRECTORY config launch rviz srdf
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS self_collision_inspector
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
ament_python_install_module(${PROJECT_NAME}/launch_common.py)

ament_package()
