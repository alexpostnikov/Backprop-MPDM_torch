cmake_minimum_required(VERSION 2.8.3)
project(mpdm)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS_DEBUG}")
##############################################################################
# Find Packages
##############################################################################

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  rospy
  std_msgs
  geometry_msgs
  message_generation
  grid_map_core
  grid_map_ros
  rosmsg
  nav_msgs
  # OpenCV REQUIRED
)

add_service_files(
  FILES
  Path.srv
)

add_message_files(
  FILES
  Ped.msg
  Peds.msg
  Propagation.msg
  Learning.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
)

##############################################################################
# Catkin
##############################################################################

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
#  INCLUDE_DIRS include
#  LIBRARIES mpdm
#  CATKIN_DEPENDS geometry_msgs rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(path_planner 
src/path_planner.cpp 
src/path_simplifier.cpp
src/global_planner.cpp
src/develop.cpp
)

target_link_libraries(path_planner 
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############

## Copy launch files
# install(DIRECTORY launch/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
#   FILES_MATCHING PATTERN "*.launch"
# )
