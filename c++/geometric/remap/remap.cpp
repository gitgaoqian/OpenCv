#include <iostream>  
#include <opencv2/opencv.hpp>  


using namespace std;
using namespace cv;

//定义全局变量

Mat map1_x, map2_y;
Mat frame;




int main()
{
   
    VideoCapture capture(0); //读取摄像头
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

 
    capture>>frame;
    Mat srcImage = frame;;

    if(!srcImage.data) 
       return -1;
    // 输出矩阵定义
    Mat resultImage(srcImage.size(), srcImage.type());
    // X与Y方向矩阵
    Mat xMapImage(srcImage.size(), CV_32FC1);
    Mat yMapImage(srcImage.size(), CV_32FC1);
    int rows = srcImage.rows;
    int cols = srcImage.cols;
    for( int j = 0; j < rows; j++ )
    {
      for( int i = 0; i < cols; i++ )
       {
            // x与y均翻转
            xMapImage.at<float>(j,i) = cols - i ;
            yMapImage.at<float>(j,i) = j ;
       }
    }
    // 重映射操作
    remap( srcImage, resultImage, xMapImage, yMapImage,
           CV_INTER_LINEAR, cv::BORDER_CONSTANT,
           cv::Scalar(0,0, 0) );
    // 输出结果
    imshow("srcImage", srcImage);
    imshow("resultImage", resultImage);
    waitKey(0);
    return 0;
}


