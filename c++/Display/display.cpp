#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char* argv[]) {

  // 檢查是否有指定輸入影像檔案
  if ( argc != 2 ) {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }

  // 讀取影像檔案
  Mat image;
Mat left_image;
Mat right_image;
  image = imread( argv[1], 1 );
left_image = image(Rect(0,0,512,640));
right_image = image(Rect(512,0,512,640));
  // 檢查影像是否正確讀入
  if ( !image.data ) {
    printf("No image data \n");
    return -1;
  }

  // 建立視窗
  //namedWindow("Display Image", WINDOW_AUTOSIZE);

  // 用視窗顯示影像
  imshow("Display left_Image", left_image);
imshow("Display right_Image", right_image);

  // 顯示視窗，直到任何鍵盤輸入後才離開
  waitKey(0);

  return 0;
}
