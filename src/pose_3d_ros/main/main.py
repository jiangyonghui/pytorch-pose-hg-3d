#!/usr/bin/env python
import sys
from absl import flags
import rospy
from pose_3d_ros.pose_extractor import PoseExtractor


"""
usage: FLAGS.flag_name

"""
flags.DEFINE_boolean('save_pose_image', False, 'set True to enable pose image saving')
flags.DEFINE_boolean('save_pose_file', False,  'set True to enable pose keypoints saving')

FLAGS = flags.FLAGS
FLAGS(sys.argv)


def main():
  rospy.init_node('pose_3d_ros', log_level=rospy.DEBUG)
  poseExtractor = PoseExtractor(FLAGS.save_pose_image, FLAGS.save_pose_file)
  rospy.spin()
  print("Shutting down 3D pose node!")
