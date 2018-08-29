#!/usr/bin/env python
import torch
import cv2
import numpy as np
import time

import rospy
import rospkg
from message_repository.msg import DetectedPerson
from cv_bridge import CvBridge, CvBridgeError

from utils.debugger import Debugger
from utils.eval import getPreds
from utils.img import Crop


class PoseExtractor:
  def __init__(self):
    rospy.loginfo("Initializing 3D Pose Extractor")
 
    self.bridge = CvBridge()
    self.image_shape = (256,256,3)
    self.debugger = Debugger()
    self.person_image_topic = rospy.get_param('~person_image_topic', '/data_manager_multi_person/person_image')
    self.model_name = rospy.get_param('~pose_model', 'hgreg-3d.pth')
    self.model = {}
    self.save_pose_image = False
    self.save_pose_file = False
    self.initModel()
    rospy.loginfo("Waiting for coming image message ...")

    self.pose_3d_pub = rospy.Publisher('~pose_3d', DetectedPerson, queue_size=1)
    self.person_image_sub = rospy.Subscriber(self.person_image_topic, DetectedPerson, self.callback, queue_size=1)
    

  def callback(self, detected_person):
    t_0 = time.time()
    frame_id = detected_person.frame_id
    person_id = detected_person.person_id
    rospy.loginfo("Got person {0} image at frame {1} ".format(person_id, frame_id))
    
    try:
      person_image = self.bridge.imgmsg_to_cv2(detected_person.person_image)
    except CvBridgeError as e:
      rospy.logerr(e)

    # Resize input image
    if person_image.shape != self.image_shape:
      h, w = person_image.shape[0], person_image.shape[1]
      center = torch.FloatTensor((w/2, h/2))
      scale = 1.0 * max(h, w)
      res = 256
      input_image = Crop(person_image, center, scale, 0, res)
    else:
      input_image = person_image

    # Feed input image to model
    rospy.loginfo("feeding image to model")
    input = torch.from_numpy(input_image.transpose(2, 0, 1)).float() / 256.
    input = input.view(1, input.size(0), input.size(1), input.size(2))
    input_var = torch.autograd.Variable(input).float().cuda()
    output = self.model(input_var)

    rospy.loginfo("got output from model")

    # Get 2D pose from output and converting it to image msg using cv_bridge
    rospy.loginfo("Rendering 2D pose")
    pose2D = getPreds((output[-2].data).cpu().numpy())[0] * 4

    # Get 3D pose and converting it to image msg using cv_bridge
    rospy.loginfo("Rendering 3D pose")
    reg = (output[-1].data).cpu().numpy().reshape(pose2D.shape[0], 1)
    pose3D = np.concatenate([pose2D, (reg + 1) / 2. * 256], axis = 1)
 
    detected_person.person_pose = self.bridge.cv2_to_imgmsg(pose3D)
    self.pose_3d_pub.publish(detected_person)
    
    rospy.loginfo("Publishing 3D Pose")
    rospy.loginfo("Pose 3D: \n{}".format(pose3D))
    
    # Estimate fps
    rospy.loginfo("Pose estimation elapsed time: {}".format(time.time() - t_0))
    
    # Save pose image
    if self.save_pose_image:
      cv2.imwrite(pkg_path+'/scripts/debug/original/ogImg_'+str(self.frame_id)+'.png', self.cv_image)
      cv2.imwrite(pkg_path+'/scripts/debug/input/inputImg_'+str(self.frame_id)+'.png', input_image)
      self.debugger.addImg(input_image, imgId=self.frame_id)
      self.debugger.addPoint2D(pose2D, (255, 0, 0), imgId=self.frame_id)
      self.debugger.saveImg(pkg_path+'/scripts/debug/pose/poseImg_'+str(self.frame_id)+'.png', imgId=self.frame_id)

    if self.save_pose_file:
      file_name = pkg_path + '/pose_file/pose_{:04d}.txt'.format(self.frame_id)
      with file(file_name, 'w') as outfile:
        np.savetxt(outfile, pose3D, fmt='%-7.2f')
    
    rospy.loginfo("----------------")

  # Initialize model
  def initModel(self):
    rospy.loginfo("=====> Loading and Initializing Model")
    model_path = rospkg.RosPack().get_path('pose_3d_ros') + '/models/' + self.model_name
    self.model = torch.load(model_path).cuda()
    img = np.zeros((256,256,3))
    input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
    input = input.view(1, input.size(0), input.size(1), input.size(2))
    input_var = torch.autograd.Variable(input).float().cuda()
    output = self.model(input_var)
    rospy.loginfo("Model Initialization Done")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
