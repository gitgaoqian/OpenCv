# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 10:46:57 2016

@author: redsun
"""

import numpy as np
import cv2
intr_left=cv2.cv.Load("Intrinsics_camera_left.xml")
intr_right=cv2.cv.Load("Intrinsics_camera_right.xml")
distortion_left=cv2.cv.Load("distortion_left.xml")
distortion_right=cv2.cv.Load("distortion_right.xml")
translation=cv2.cv.Load("translational_vector.xml")
#rot_rodg=cv2.cv.Load("rot_rodgus_vector.xml")
rot_mat = cv2.cv.Load("rot_mat.xml")

#rot_mat = np.zeros((3,3), np.float64)  
#cv2.Rodrigues(np.asarray(rot_rodg),rot_mat)

R1 = cv2.cv.fromarray(np.zeros((3,3), np.float64))  
R2 = cv2.cv.fromarray(np.zeros((3,3), np.float64))  
P1 = cv2.cv.fromarray(np.zeros((3,4), np.float64))  
P2 = cv2.cv.fromarray(np.zeros((3,4), np.float64)) 
Q  = cv2.cv.fromarray(np.zeros((4,4), np.float64)) 

cv2.cv.StereoRectify(
  intr_left, 
  intr_right, 
  distortion_left, 
  distortion_right, 
  (640,480), 
  cv2.cv.fromarray( rot_mat), 
  translation,
  R1, 
  R2, 
  P1, 
  P2,
  Q)

mapx1,mapy1=cv2.initUndistortRectifyMap(
  np.asarray(intr_left),
  np.asarray(distortion_left),
  np.asarray(R1),
  np.asarray(P1),
  (640,480),
  cv2.CV_32FC1
  )

mapx2,mapy2=cv2.initUndistortRectifyMap(
  np.asarray(intr_right),
  np.asarray(distortion_right),
  np.asarray(R2),
  np.asarray(P2),
  (640,480),
  cv2.CV_32FC1
  )
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    img_L = frame[0:480,0:640]
    img_R=frame[0:480,640:1280]  
    
    
    img_L_recti=cv2.remap(img_L,mapx1,mapy1,cv2.cv.CV_INTER_LINEAR)
    img_R_recti=cv2.remap(img_R,mapx2,mapy2,cv2.cv.CV_INTER_LINEAR)
    vis = np.concatenate((img_L_recti, img_R_recti), axis=1)
    for j in range(0,480,16):
      cv2.line(vis, (0,j), (1280,j), (0, 255, 0));
    cv2.imshow('img', vis )
    
    
    stereo = cv2.StereoSGBM(minDisparity = 16,
        numDisparities = 128,
        SADWindowSize = 3,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 16,
        disp12MaxDiff = -1,
        P1 = 8*3*3**2,
        P2 = 32*3*3**2,
        fullDP = False
    )
    
    disparity = stereo.compute(img_L_recti, img_R_recti).astype(np.float32) / 16.0
    disp=cv2.normalize(disparity,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    
    cv2.imshow('disparity', disp)
    cv2.waitKey(10)


cv2.destroyAllWindows()
