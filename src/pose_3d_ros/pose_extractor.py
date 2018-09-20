#!/usr/bin/env python
import rospy
import rospkg
from geometry_msgs.msg import Point
from message_repository.msg import Person, FrameInfo
from cv_bridge import CvBridge, CvBridgeError

import torch
import cv2
import numpy as np
import time
from threading import Lock
from functools import partial
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

from utils.debugger import Debugger
from utils.eval import getPreds
from utils.img import Crop


class PoseExtractor:
  def __init__(self, flag_save_pose_image, flag_save_pose_file):
    rospy.loginfo("Initializing 3D Pose Extractor")
    
    self.frameInfo = FrameInfo()
    self.bridge = CvBridge()
    self.lock = Lock()
    self.image_shape = (256,256,3)
    self.debugger = Debugger()
    self.tracking_info_topic = rospy.get_param('~tracking_info_topic', '/person_tracker/tracking_info')
    self.model_name = rospy.get_param('~pose_model', 'hgreg-3d.pth')
    self.model = {}
    self.save_pose_image = flag_save_pose_image
    self.save_pose_file = flag_save_pose_file
    self.initModel()

    self.pose_3d_pub = rospy.Publisher('~pose_3d', Person, queue_size=1)
    self.tracking_info_sub = rospy.Subscriber(self.tracking_info_topic, FrameInfo, self.trackingInfoCallback, queue_size=1)
    

  def trackingInfoCallback(self, tracking_info_msg):
    begin = time.time()
    self.frameInfo = tracking_info_msg
    rospy.loginfo("Frame ID: {}".format(self.frameInfo.frame_id))
    
    tracked_persons = self.frameInfo.persons
    num_persons = len(tracked_persons)

    if num_persons != 0:
      for person in tracked_persons:
        rospy.loginfo("Person {} is detected".format(person.person_id))

      try:
        # multithreading
        p = ThreadPool(num_persons)
        p.map(self.poseEstimation, tracked_persons)
        
        # multiprocessing
#        curImage = self.bridge.imgmsg_to_cv2(self.frameInfo.image_frame)
#        p = ProcessPool(1)
#        p.map(partial(poseEstimationMp, cur_img=curImage, image_shape=self.image_shape, model=self.model, 
#                      pose_pub=self.pose_3d_pub), tracked_persons)  
        p.close()
        p.join()                   
      except BaseException as e:
        rospy.logerr(e)

    else:
      rospy.logwarn("No person is detected!")

    rospy.loginfo("FPS: {}".format(1 / (time.time() - begin)))
    
    
  def poseEstimation(self, tracked_person):
    person_id = tracked_person.person_id
    
    try:
      curImage = self.bridge.imgmsg_to_cv2(self.frameInfo.image_frame)
      person_image = curImage[int(tracked_person.bbox.top):int(tracked_person.bbox.top + tracked_person.bbox.height),
                              int(tracked_person.bbox.left):int(tracked_person.bbox.left + tracked_person.bbox.width)]
    except CvBridgeError as e:
      rospy.logerr(e)

    # Resize input image
    rospy.logdebug("person image shape: {}".format(person_image.shape))
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
    
    # lock when using model to estimate pose
    self.lock.acquire()
    try:
      output = self.model(input_var)
    finally:
      self.lock.release()
      
    rospy.logdebug("got output from model")

    # Get 2D pose from output and converting it to image msg using cv_bridge
    rospy.logdebug("Rendering 2D pose")
    pose2D = getPreds((output[-2].data).cpu().numpy())[0] * 4

    # Get 3D pose and converting it to image msg using cv_bridge
    rospy.logdebug("Rendering 3D pose")
    reg = (output[-1].data).cpu().numpy().reshape(pose2D.shape[0], 1)
    pose3D = np.concatenate([pose2D, (reg + 1) / 2. * 256], axis = 1)
    rospy.logdebug("pose 3d shape: {}".format(pose3D.shape))
    
    
    for pose in pose3D:
      joint = Point()
      joint.x = pose[0] + tracked_person.bbox.left
      joint.y = pose[1] + tracked_person.bbox.top
      joint.z = pose[2]
      tracked_person.person_pose.append(joint)
      
    self.pose_3d_pub.publish(tracked_person)
    
    rospy.logdebug("Publishing 3D Pose")
    rospy.logdebug("pose3D: \n {}".format(pose3D))
    #rospy.logdebug("Pose 3D msg: \n {}".format(tracked_person.person_pose))
    
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
    
    rospy.loginfo("Person {} processing finished".format(person_id))

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
    
    
    
    
## for multi processing
def poseEstimationMp(tracked_person, cur_img, image_shape, model, pose_pub):
  person_id = tracked_person.person_id
  person_image = cur_img[int(tracked_person.bbox.top):int(tracked_person.bbox.top + tracked_person.bbox.height),
                         int(tracked_person.bbox.left):int(tracked_person.bbox.left + tracked_person.bbox.width)]
 
  # Resize input image
  rospy.logdebug("person image shape: {}".format(person_image.shape))
  if person_image.shape != image_shape:
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
  output = model(input_var)  
  rospy.logdebug("got output from model")

  # Get 2D pose from output and converting it to image msg using cv_bridge
  rospy.logdebug("Rendering 2D pose")
  pose2D = getPreds((output[-2].data).cpu().numpy())[0] * 4

  # Get 3D pose and converting it to image msg using cv_bridge
  rospy.logdebug("Rendering 3D pose")
  reg = (output[-1].data).cpu().numpy().reshape(pose2D.shape[0], 1)
  pose3D = np.concatenate([pose2D, (reg + 1) / 2. * 256], axis = 1)
  rospy.logdebug("pose 3d shape: {}".format(pose3D.shape))
  
  for pose in pose3D:
    joint = Point()
    joint.x = pose[0] + tracked_person.bbox.left
    joint.y = pose[1] + tracked_person.bbox.top
    joint.z = pose[2]
    tracked_person.person_pose.append(joint)
    
  pose_pub.publish(tracked_person)
  
  rospy.logdebug("3D pose published")
  rospy.logdebug("pose3D: \n {}".format(pose3D))
  rospy.loginfo("Person {} processing finished".format(person_id))
  

    
    
    
    
    
    
    
    
    
    
    
    
