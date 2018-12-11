#include <iostream>  
#include </usr/include/opencv/cv.h>
#include </usr/include/opencv/cxcore.h>
#include </usr/include/opencv/highgui.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

//定义全局变量


Mat frame;
Mat Image;
Mat Image_L;
Mat Image_R;
int i=1;

int main()
{
   
    VideoCapture capture(0); //读取摄像头
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
 
    capture>>frame;
    Image = frame;
    printf("%d,%d",Image.cols,Image.rows);
    imwrite("/home/ros/opencv_files/c++/face_reco/test.jpg",Image);
    //int c = waitKey(10000);
    return 0;


    //Image_L = frame(Rect(0,0,640,480));
    //Image_R = frame(Rect(640,0,640,480));
	//imshow("Display left_Image", Image_L);
	//imshow("Display right_Image",Image_R);
	//save 10 picture of me

	
}


