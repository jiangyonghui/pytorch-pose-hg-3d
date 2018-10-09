#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import rospkg
import torch
import cv2
import glob
import time
import numpy as np
from pose_3d_ros.tools.opts import opts
from pose_3d_ros.tools import ref
from pose_3d_ros.utils.debugger import Debugger
from pose_3d_ros.utils.eval import getPreds
from pose_3d_ros.utils.img import Crop

def main():
  print 'loading model'
  model_path = rospkg.RosPack().get_path('pose_3d_ros') + '/models/hgreg-3d.pth'
  model = torch.load(model_path).cuda()
  print 'model loaded!'
  
  #image_path = rospkg.RosPack().get_path('pose_3d_ros') + '/images/h36m_1214.png'
  img = cv2.imread('./00000.png')
  print 'image shape: ', img.shape
  
  if img.shape != (256,256,3):
    h, w = img.shape[0], img.shape[1]
    center = torch.FloatTensor((w/2, h/2))
    size = 1.0 * max(h, w)
    res = 256
    img = Crop(img, center, size, 0, res)
  
  input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
  input = input.view(1, input.size(0), input.size(1), input.size(2))
  input_var = torch.autograd.Variable(input).float().cuda()
  begin = time.time()
  output = model(input_var)
  pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
  reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
  pose2D = pred
  pose3D = np.concatenate([pred, (reg+1) / 2. * 256], axis = 1)
  print "3D Pose:\n", pose3D
  print 'eclapsed time: ', time.time() - begin
  
  #debugger = Debugger()
  #debugger.addImg(img)
  #debugger.addPoint2D(pred, (255, 0, 0))
  #debugger.addPoint3D(pose3D)
  #debugger.saveImg("debug_1_0.png")
  #debugger.showImg(pause = True)
  #debugger.show3D()  
    

if __name__ == '__main__':
  main()
  
  
  
  
  




