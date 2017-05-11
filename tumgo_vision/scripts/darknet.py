#!/usr/bin/env python

# -*- encoding: UTF-8 -*-

##  @file darknet.py
#   @brief File storing information about the darknet services.

##  @package darknet
#   @brief Package storing information about the darknet services.

import sys
import os
import re
import subprocess
import time
import math
import numpy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import rospy
from threading import Lock
from tumgo_vision.srv import srvRecognition2D, srvRecognition2DResponse

class tumgo_label:
  def __init__(self):
    self.darknet_dir = rospy.get_param('darknet_dir', 'NONE')
    self.bridge = CvBridge()

  def predict_label(self, req):
    labels = []
    percent = []
    bb_minx = []
    bb_maxx = []
    bb_miny = []
    bb_maxy = []
    if self.darknet_dir == 'NONE':
      os.chdir(os.path.join(os.path.expanduser("~"), "ros/workspaces/project_ws/src/tumgo_vision/darknet/"))
    else:
      os.chdir(self.darknet_dir)
    try:
      cv_image = self.bridge.imgmsg_to_cv2(req.frame, "bgr8")
    except CvBridgeError as e:
      print(e)
    cv2.imwrite(os.path.join(self.darknet_dir, 'Img.png'),cv_image)
    command = './darknet detect cfg/yolo.cfg yolo.weights Img.png'
    val = subprocess.check_output(command.split(' '))

    # postprocess the results and return
    val = val.split('\n')
    for ret in val:
      if ret == '':
        continue
      elif ret.startswith("Bounding box:"):
        coords = ret[14:].split(',')
        bb_minx.append(int(coords[0]))
        bb_maxx.append(int(coords[1]))
        bb_miny.append(int(coords[2]))
        bb_maxy.append(int(coords[3]))
      else:
        label = ret.split(':')
        labels.append(label[0])
        percent.append(float(label[1][:-1])/100)
    return srvRecognition2DResponse(labels,percent,bb_minx,bb_maxx,bb_miny,bb_maxy)

  def predict_server(self):
    s = rospy.Service('/tumgo_vision/recognition2D', srvRecognition2D, self.predict_label)
    rospy.loginfo("Ready to label.")

if __name__ == "__main__":
    rospy.init_node('tumgo_label', anonymous=True)
    rospy.loginfo("Recognition node running...")
    node = tumgo_label();
    node.predict_server();
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")