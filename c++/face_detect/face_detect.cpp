

#include </usr/include/opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "/home/ros/opencv_files/c++/face_detect/haarcascades/haarcascade_frontalface_default.xml";
String eyes_cascade_name = "/home/ros/opencv_files/c++/face_detect/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;   //定义人脸分类器
CascadeClassifier eyes_cascade;   //定义人眼分类器
String window_name = "Capture - Face detection";

/** @function main */
int main(void)
{
	Mat frame = imread("/home/ros/opencv_files/c++/face_detect/boy.jpg");

	//VideoCapture capture;
	//Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };

	//-- 2. Read the video stream
	//capture.open(0);
	//if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	//while (capture.read(frame))
	//{
	//	if (frame.empty())
	//	{
	//		printf(" --(!) No captured frame -- Break!");
	//		break;
	//	}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);

		int c = waitKey(0);
		if ((char)c == 27) { return 0; } // escape
	//}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, Size(70, 70),Size(100,100));
        // mark the face and eyes
	for (size_t i = 0; i < faces.size(); i++)
	{
		//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		rectangle(frame, faces[i],Scalar(255,0,0),2,8,0);
		//shrink the area of myface 
		Mat faceROI = frame_gray(faces[i]);
		Mat MyFace;
		if (faceROI.cols > 100)  
    		{  
        		resize(faceROI, MyFace, Size(92, 112));   
        		imwrite(("/home/ros/opencv_files/c++/face_reco/11.jpg"), MyFace);  
        
    		}  
		

		//-- In each face, detect eyes
		
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 1, CV_HAAR_DO_ROUGH_SEARCH, Size(3, 3));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Rect rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height);
			
			//Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			//int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			//circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
			rectangle(frame, rect, Scalar(0, 255, 0), 2, 8, 0);
		}
	}

	//-- Show what you got
	namedWindow(window_name, 2);
	imshow(window_name, frame);
}
