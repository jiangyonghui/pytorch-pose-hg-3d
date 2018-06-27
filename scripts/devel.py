#!/usr/bin/env python
import torch
from utils.debugger import Debugger
from utils.eval import getPreds
import cv2
import numpy as np
import time
import rospy
import roslib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

##Class for image subscription
class PoseExtractor:
    def __init__(self):
      rospy.loginfo("Initializing 3D Pose Extractor")
      self.frame_id = 0
      self.n_time = 0
      self.bridge = CvBridge()
      self.cv_image = None
      self.input_image = None
      self.debugger = Debugger()
      self.image_topic = rospy.get_param('~image_topic', '/image_reader/image_raw')
      self.model_name = rospy.get_param('~pose_model', dir_path+'/models/hgreg-3d.pth')
      self.save_pose_image = rospy.get_param('~save_pose_image', False)
      self.initModel()
      rospy.loginfo("Waiting for coming image message ...")
   
      #self.pose_2d_pub = rospy.Publisher('/pose_3d_ros_devel/pose_2d', Image, queue_size=1)
      #self.pose_3d_pub = rospy.Publisher('/pose_3d_ros_devel/pose_3d', Image, queue_size=1)
      self.image_sub = rospy.Subscriber(self.image_topic, Image, self.callback, queue_size=1)
      
    
    def callback(self, data):
      t_0 = time.time()
      rospy.loginfo("[Frame ID: %d]", self.frame_id)

      try:
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
        print(e)  
        
      rospy.loginfo("recieved image shape: %s", str(self.cv_image.shape))
      
      # Resize input image
      rospy.loginfo("reshaping input image")
      self.input_image = cv2.resize(self.cv_image, (256,256))
      rospy.loginfo("input image shape: %s", str(self.input_image.shape)) 
      
      rospy.loginfo("feeding image to model") 
      input = torch.from_numpy(self.input_image.transpose(2, 0, 1)).float() / 256.
      input = input.view(1, input.size(0), input.size(1), input.size(2))
      input_var = torch.autograd.Variable(input).float().cuda()
      output = self.model(input_var)
     
      rospy.loginfo("got output from model")
      
      # Get 2D pose from output and converting it to image msg using cv_bridge
      rospy.loginfo("Rendering 2D pose")
      pose2D = getPreds((output[-2].data).cpu().numpy())[0] * 4
#      msg_2d = self.bridge.cv2_to_imgmsg(pose2D)
#      self.pose_2d_pub.publish(msg_2d)
      print(pose2D)
      
      # Get 3D pose and converting it to image msg using cv_bridge
      rospy.loginfo("Rendering 3D pose")
      reg = (output[-1].data).cpu().numpy().reshape(pose2D.shape[0], 1)
      pose3D = np.concatenate([pose2D, (reg + 1) / 2. * 256], axis = 1)
#      msg_3d = self.bridge.cv2_to_imgmsg(pose3D)
#      self.pose_3d_pub.publish(msg_3d)
      print(pose3D)
      
      # Estimate fps
      self.n_time += time.time() - t_0
      fps = (self.frame_id+1)/self.n_time
      rospy.loginfo("Average fps: %s", str(fps))
      
      # Save pose image 
      if self.save_pose_image:
        cv2.imwrite(dir_path+'/debug/original/original_'+str(self.frame_id)+'.png', self.cv_image)
        cv2.imwrite(dir_path+'/debug/devel/resized_'+str(self.frame_id)+'.png', self.input_image)
        self.debugger.addImg(self.input_image, imgId=self.frame_id)
        self.debugger.addPoint2D(pose2D, (255, 0, 0), imgId=self.frame_id)
        self.debugger.saveImg(path=dir_path+'/debug/pose/pose_'+str(self.frame_id)+'.png', imgId=self.frame_id)
      
      # Update frame_id
      self.frame_id += 1
      
      print(" ")
      print("-------------------------------------------")
      
    # Initialize model  
    def initModel(self):
      rospy.loginfo("=====> Loading and Initializing Model")
      self.model = torch.load(self.model_name).cuda()
      img = np.zeros((256,256,3))
      input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
      input = input.view(1, input.size(0), input.size(1), input.size(2))
      input_var = torch.autograd.Variable(input).float().cuda()
      output = self.model(input_var)
      rospy.loginfo("Model Initialization Done")
   

def main():
  rospy.init_node('pose_3d_ros_devel')
  poseExtractor = PoseExtractor()
  rospy.spin() 
  print("Shutting down pose_3d_ros_devel node")
    

if __name__ == '__main__':
  main()
  
  
  
  
  
  
  
  
  
  
