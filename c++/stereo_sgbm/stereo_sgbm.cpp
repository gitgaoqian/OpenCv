/******************************/
/*        立体匹配和测距        */
/******************************/

#include <opencv2/opencv.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;

const int imageWidth = 320;           //摄像头的分辨率  
const int imageHeight = 240;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;


Mat frame;
Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 0, uniquenessRatio =0, numDisparities=0;
//Ptr<StereoBM> bm = StereoBM::create(16, 9);
StereoSGBM sgbm;

/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 213.437818, 0.000000, 172.803103,
0.000000, 214.075437 ,127.027571,
0.000000, 0.000000, 1.000000
);
Mat distCoeffL = (Mat_<double>(5, 1) <<-0.374243, 0.118810 ,0.001474, 0.001239, 0.000000);

Mat cameraMatrixR = (Mat_<double>(3, 3) <<210.689344, 0.000000, 168.144027,
0.000000, 211.281242, 123.774837,
0.000000, 0.000000, 1.000000
);
Mat distCoeffR = (Mat_<double>(5, 1) <<-0.353463, 0.098363 ,0.002089, 0.003900, 0.000000);

Mat T = (Mat_<double>(3, 1) << -58.9993788834643,-0.166484432134018,1.80514541799934);//T平移向量
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
Mat R = (Mat_<double>(3, 3) << 0.999987, -0.001475, 0.004724 ,
0.01476 ,0.999980 ,-0.00014896, 
-0.00014711, 0.005906, 0.999972);//R 旋转矩阵


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

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        cout << origin <<"in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;//输出鼠标所点击的位置在世界坐标系下的三维坐标
        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
        break;
    }
}


/*****主函数*****/
int main()
{
    /*
    立体校正:畸变矫正和平行校正.产生
    */
    //Rodrigues(rec, R); //Rodrigues变换，将旋转向量转化为旋转矩阵Ｒ：利用的是罗德里格斯变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
       0, imageSize);//立体校正，产生新的校正旋转矩阵和投影矩阵以及重投影矩阵
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);//校正映射，产生映射表,
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /*
    读取图片
    */
    VideoCapture capture(0); //读取摄像头
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    //分割图片 
    capture>>frame;
    rgbImageL = frame(Rect(0,0,320,240));
    cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);//转为灰度图
    rgbImageR = frame(Rect(320,0,320,240));
    cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

   // imshow("ImageL Before Rectify", grayImageL);//输出校正前的图片
    //imshow("ImageR Before Rectify", grayImageR);
    //合并到同一张画布
    Mat concat1;
    hconcat(rgbImageL,rgbImageR,concat1);
    //画上对应的线条
    for (int i = 0; i < concat1.rows; i += 16)
        line(concat1, Point(0, i), Point(concat1.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("origin", concat1);


    /*
    经过remap之后，左右相机的图像已经共面并且行对准了
    */
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);//对图像进行校正,同时实现了畸变矫正和平行校正
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

   
    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //颜色空间转换：从灰度到RGB
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);

    //输出校正后并且转换成彩色的图像
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    //imshow("ImageL After Rectify", rgbRectifyImageL);
    //imshow("ImageR After Rectify", rgbRectifyImageR);

    //显示在同一张图上，创建画布
    Mat concat2;
    hconcat(rgbRectifyImageL,rgbRectifyImageR,concat2);


    //画上对应的线条
    for (int i = 0; i < concat2.rows; i += 8)
        line(concat2, Point(0, i), Point(concat2.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", concat2);

    /*
    立体匹配
    */
    namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity",&blockSize, 8, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);
    stereo_match(0,0);

    waitKey(0);
    return 0;
}
