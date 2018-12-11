# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 18:03:03 2017

@author: ros
"""
#对于一副图像像素值的改变,可以用opencv函数或者numpy进行数组的处理
import numpy 
import cv2
#读取图像
img1 = cv2.imread('test1.jpg')#img的格式就是三维数组(BGR图像)
img2 = cv2.imread('test2.jpg')
#获取图像属性
print img2.shape
#获取一点像素值
print img1[100,100]
#更改一点像素值
img1[100,100]=[255,255,255]
#拷贝某片图像到图像其他区域
area=img1[350:440,400:490]
img1[100:190,100:190]=area
#图像颜色的转换:BGR↔Gray 和 BGR↔HSV。
img1_color = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#canny边缘检测
img1_edges = cv2.Canny(img1_gray,240,245)

#图像的尺寸改变,利用了插值
img1_size = cv2.resize(img1,(512,512),interpolation=cv2.INTER_LINEAR)
#图像的旋转
M=cv2.getRotationMatrix2D((320,512),45,0.6)#这里的第一个参数为旋转中心,第二个为旋转角度,第三个为旋转后的缩放因子
img1_rotation=cv2.warpAffine(img1,M,(1024,640))# 第三个参数是输出图像的尺寸中心
#两幅图像执行加法运算
img_add = cv2.add(img1,img2)
#两幅图像混合,出现透明现象,图像1加权值0.7,图像2占0.3
img_merge=cv2.addWeighted(img1,0.7,img2,0.3,0)
#显示图像
cv2.imshow('image',img1_edges)
#等待键盘输入,当输入esc,图像窗口关闭
k = cv2.waitKey(0)&0xFF
if k == 27:
    cv2.destroyAllWindows()
