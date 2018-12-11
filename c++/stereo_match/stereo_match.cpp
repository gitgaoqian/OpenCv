#include <opencv2/opencv.hpp> 
#include <stdio.h> 
#include <iostream>  

using namespace std;
using namespace cv;
  
  
  
using namespace cv;  
  
static void print_help()  
{  
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");  
    printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"  
        "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"  
        "[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");  
}  
  
static void saveXYZ(const char* filename, const Mat& mat)  
{  
    const double max_z = 1.0e4;  
    FILE* fp = fopen(filename, "wt");  
    for (int y = 0; y < mat.rows; y++)  
    {  
        for (int x = 0; x < mat.cols; x++)  
        {  
            Vec3f point = mat.at<Vec3f>(y, x);  
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;  
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);  
        }  
    }  
    fclose(fp);  
}  
  
int main(int argc, char** argv)
{  
    const char* algorithm_opt = "--algorithm=";  
    const char* maxdisp_opt = "--max-disparity=";  
    const char* blocksize_opt = "--blocksize=";  
    const char* nodisplay_opt = "--no-display";  
    const char* scale_opt = "--scale=";  
  
    //if (argc < 3)  
    //{  
    //  print_help();  
    //  return 0;  
    //}  
    const char* img1_filename = 0;  
    const char* img2_filename = 0;  
    const char* intrinsic_filename = 0;  
    const char* extrinsic_filename = 0;  
    const char* disparity_filename = 0;  
    const char* point_cloud_filename = 0;  
  
    enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3 };  
    int alg = STEREO_VAR;  
    int SADWindowSize = 0, numberOfDisparities = 0;  
    bool no_display = false;  
    float scale = 1.f;  
  
    StereoBM bm;  
    StereoSGBM sgbm;  
    StereoVar var;  
  
    //------------------------------  
  
    /*img1_filename = "tsukuba_l.png"; 
    img2_filename = "tsukuba_r.png";*/  
  
    img1_filename = "01.jpg";  
    img2_filename = "02.jpg";  
  
    int color_mode = alg == STEREO_BM ? 0 : -1;  
    Mat img1 = imread(img1_filename, color_mode);  
    Mat img2 = imread(img2_filename, color_mode);  
  
  
    Size img_size = img1.size();  
  
    Rect roi1, roi2;  
    Mat Q;  
  
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;  
  
    bm.state->roi1 = roi1;  
    bm.state->roi2 = roi2;  
    bm.state->preFilterCap = 31;  
    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;  
    bm.state->minDisparity = 0;  
    bm.state->numberOfDisparities = numberOfDisparities;  
    bm.state->textureThreshold = 10;  
    bm.state->uniquenessRatio = 15;  
    bm.state->speckleWindowSize = 100;  
    bm.state->speckleRange = 32;  
    bm.state->disp12MaxDiff = 1;  
  
    sgbm.preFilterCap = 63;  
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;  
  
    int cn = img1.channels();  
  
    sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;  
    sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;  
    sgbm.minDisparity = 0;  
    sgbm.numberOfDisparities = numberOfDisparities;  
    sgbm.uniquenessRatio = 10;  
    sgbm.speckleWindowSize = bm.state->speckleWindowSize;  
    sgbm.speckleRange = bm.state->speckleRange;  
    sgbm.disp12MaxDiff = 1;  
    sgbm.fullDP = alg == STEREO_HH;  
  
    var.levels = 3;                                 // ignored with USE_AUTO_PARAMS  
    var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS  
    var.nIt = 25;  
    var.minDisp = -numberOfDisparities;  
    var.maxDisp = 0;  
    var.poly_n = 3;  
    var.poly_sigma = 0.0;  
    var.fi = 15.0f;  
    var.lambda = 0.03f;  
    var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS  
    var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS  
    var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING;  
  
    Mat disp, disp8;  
    //Mat img1p, img2p, dispp;  
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);  
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);  
  
    int64 t = getTickCount();  
    if (alg == STEREO_BM)  
        bm(img1, img2, disp);  
    else if (alg == STEREO_VAR) {  
        var(img1, img2, disp);  
    }  
    else if (alg == STEREO_SGBM || alg == STEREO_HH)  
        sgbm(img1, img2, disp);//------  
  
    t = getTickCount() - t;  
    printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());  
  
    //disp = dispp.colRange(numberOfDisparities, img1p.cols);  
    if (alg != STEREO_VAR)  
        disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));  
    else  
        disp.convertTo(disp8, CV_8U);  
    if (!no_display)  
    {  
        namedWindow("left", 1);  
        imshow("left", img1);  
  
        namedWindow("right", 1);  
        imshow("right", img2);  
  
        namedWindow("disparity", 0);  
        imshow("disparity", disp8);  
  
        imwrite("result.bmp", disp8);  
        printf("press any key to continue...");  
        fflush(stdout);  
        waitKey();  
        printf("\n");  
    }  
  
  
    return 0;  
}  