/******************************/
/*        立体匹配和测距        */
/******************************/

#include <opencv2/opencv.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;


Mat rectifyImageL, rectifyImageR;
int blockSize = 0, uniquenessRatio =0, numDisparities=0;
//Ptr<StereoBM> bm = StereoBM::create(16, 9);
StereoSGBM sgbm;

/*****立体匹配*****/
void stereo_match(int,void*)
{

    sgbm.preFilterCap=63;
    sgbm.SADWindowSize=2*blockSize+3;//SAD窗口大小，3~11之间
    sgbm.numberOfDisparities=numDisparities*16+16;
    sgbm.P1=8*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2=4*sgbm.P1;
    sgbm.minDisparity=0;
    sgbm.uniquenessRatio=uniquenessRatio;
    sgbm.speckleWindowSize=100;
    sgbm.speckleRange=32;
    sgbm.disp12MaxDiff=1;
    Mat disp,disp8;
    sgbm(rectifyImageL, rectifyImageR, disp);
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式,转换成CV_8U
    
    imshow("disparity", disp8);//显示视差图
}



/*****主函数*****/
int main()
{
    rectifyImageL=imread("01.jpg");
    rectifyImageR=imread("02.jpg");

    /*
    立体匹配
    */
    namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity",&blockSize, 4, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);

    stereo_match(0,0);

    waitKey(0);
    return 0;
}
