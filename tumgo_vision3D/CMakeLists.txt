cmake_minimum_required(VERSION 2.8.3)
project(tumgo_vision3D)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
  /usr/local/lib/cmake 
  /opt/ros/indigo/share/cmake_modules/cmake/Modules 
  ${PROJECT_SOURCE_DIR}/cmake
)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  std_srvs
  cv_bridge
  image_transport
  image_geometry
  message_generation
  sensor_msgs
  tumgo_vision
  tf
  pcl_ros
  pcl_conversions
)

find_package(Eigen REQUIRED)
find_package(PCL 1.7 REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   DIRECTORY
#    msg/tumgo_vision3D
#   FILES
# )
#
### Generate services in the 'srv' folder
# add_service_files(
#   DIRECTORY
#    srv/tumgo_vision3D
#   FILES
# )
### Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   sensor_msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES tutorial_reasoning
 CATKIN_DEPENDS roscpp 
    rospy 
    std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  include
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# include the headers of the library
file(GLOB_RECURSE KNOWLEDGE_INC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS 
  include/*.h)

### Declare a cpp library

file(GLOB SRC_HELPERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
src/src_helpers/*.cpp)

## Declare a cpp executable
add_executable(tumgo_vision3D src/tumgo_vision3D.cpp)
target_link_libraries(tumgo_vision3D    
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  )