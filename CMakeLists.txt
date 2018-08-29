cmake_minimum_required(VERSION 2.8.3)
project(pose_3d_ros)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_repository
  roslib
)

catkin_package(
  CATKIN_DEPENDS 
  rospy 
  std_msgs
  sensor_msgs
  message_repository
  roslib
)


# for python scripts
catkin_install_python(PROGRAMS scripts/pose_3d_ros scripts/pose_3d_ros_multi_person
					  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
					  

catkin_python_setup()
