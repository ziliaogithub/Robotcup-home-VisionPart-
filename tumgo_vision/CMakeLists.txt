cmake_minimum_required(VERSION 2.8.3)
project(tumgo_vision)

set(CMAKE_CXX_FLAGS "-std=c++11")

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
  tf
)

find_package(Eigen REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)

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
 add_message_files(
   DIRECTORY
    msg/tumgo_vision
   FILES
    msgDetectObject.msg
 )

## Generate services in the 'srv' folder
 add_service_files(
   DIRECTORY
    srv/tumgo_vision
   FILES
    srvCloud.srv
    srvImage.srv
    srvLabel.srv
    srvDetectColorBlob.srv
    srvDetectFace.srv
    srvDetectObject3D.srv
    srvGetCloud.srv
    srvGetImage.srv
    srvSegmentObjects.srv
    srvSegmentPlanes.srv
    srvRecognition.srv
    srvRecognition2D.srv
    srvRecognition3D.srv
    srvGetFacePosition.srv
    srvLabel.srv
    srvDetectHuman.srv
    srvDetectObject2D.srv
    srvTrackObject.srv
 )
## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
 )

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
        message_runtime
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

## Declare a cpp executable

add_executable(tumgo_vision2D src/tumgo_vision2D.cpp)
target_link_libraries(tumgo_vision2D
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${CV_BRIDGE_LIBRARY}
  )

add_executable(tumgo_vision src/tumgo_vision.cpp)
target_link_libraries(tumgo_vision
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CV_BRIDGE_LIBRARY}
  )

## Declare a C++ library
add_library(tumgo_human_detection
   src/${PROJECT_NAME}/tumgo_human_detection.cpp
)
target_link_libraries(
    tumgo_human_detection
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
## Declare a C++ executable
add_executable(tumgo_human_detection_node
    src/main.cpp
)
target_link_libraries(tumgo_human_detection_node
   ${catkin_LIBRARIES}
   tumgo_human_detection
)

catkin_install_python(PROGRAMS scripts/darknet.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

 ## Mark other files for installation (e.g. launch and bag files, etc.)
 install(DIRECTORY launch config
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

