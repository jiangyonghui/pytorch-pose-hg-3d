#!/usr/bin/env python
import rospy
from pose_extractor import PoseExtractor

def main():
  rospy.init_node('pose_3d_ros_multi_person')
  poseExtractor = PoseExtractor()
  rospy.spin()
  print("Shutting down 3D pose node!")
