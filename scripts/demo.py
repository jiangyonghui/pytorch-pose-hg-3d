#!/usr/bin/env python
import torch
import cv2
import glob
import numpy as np
from pose_3d_ros.tools.opts import opts
from pose_3d_ros.tools import ref
from pose_3d_ros.utils.debugger import Debugger
from pose_3d_ros.utils.eval import getPreds
from pose_3d_ros.utils.img import Crop

def main():
  opt = opts().parse()
  
  if opt.loadModel != 'none':
    model = torch.load(opt.loadModel).cuda()
  else:
    model = torch.load('../models/hgreg-3d.pth').cuda()
  
  if img.shape != (256,256,3):
    h, w = img.shape[0], img.shape[1]
    center = torch.FloatTensor((w/2, h/2))
    size = 1.0 * max(h, w)
    res = 256
    img = Crop(img, center, size, 0, res)
    
  input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
  input = input.view(1, input.size(0), input.size(1), input.size(2))
  input_var = torch.autograd.Variable(input).float().cuda()
  output = model(input_var)
  pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
  reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
  pose2D = pred
  pose3D = np.concatenate([pred, (reg+1) / 2. * 256], axis = 1)
  print "3D Pose:\n", pose3D
  
  debugger = Debugger()
  debugger.addImg(img)
  debugger.addPoint2D(pred, (255, 0, 0))
  debugger.addPoint3D(pose3D)
  #debugger.saveImg("debug_1_0.png")
  debugger.showImg(pause = True)
  debugger.show3D()   
    

if __name__ == '__main__':
  main()
  
  
  
  
  




