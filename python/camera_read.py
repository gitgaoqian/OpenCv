# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:56:03 2017

@author: ros
"""

import numpy as np
import cv2
import time
utc=time.time()
interval = 2
cap = cv2.VideoCapture(0)
count=0
while(True):
	# Capture frame-by-frame
         ret, frame = cap.read()
         left_frame = frame[0:480,0:320]
         right_frame=frame[0:480,320:640]
	# Our operations on the frame come here
         
	# Display the resulting frame
         cv2.imshow('left_frame',left_frame)
         cv2.imshow('right_frame',right_frame)
         now=time.time()
        
         if (now-utc)>2 and count < 30:
             cv2.imwrite('left_img/'+'left_img'+str(count)+'.png',left_frame)
             cv2.imwrite('right_img/'+'right_img'+str(count)+'.png',right_frame)
             count+=1
             utc=time.time()
        
         if cv2.waitKey(1) & 0xFF == ord('q'):
             break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

