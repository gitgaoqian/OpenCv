// ���ܣ�����  5-31 ���������������ȡ
// ���ߣ���ΰ zhu1988wei@163.com
// ��Դ����OpenCVͼ������ʵ����
// ���ͣ�http://blog.csdn.net/zhuwei1988
// ���£�2016-8-1
// ˵������Ȩ���У����û�ժ¼����ϵ���ߣ������������ʽע��������лл��//
	// hsv �޶���ΧԪ����ȡ
	cv::Mat bw_blue = ((img_h>0.45) &
		(img_h<0.75) &
		(img_s>0.15) &
		(img_v>0.25));
	int height = bw_blue.rows;
	int width = bw_blue.cols;
	cv::Mat bw_blue_edge = cv::Mat::zeros(bw_blue.size(), bw_blue.type());
	cv::imshow("bw_blue", bw_blue);
	cv::waitKey(0);
	// ��������������ȡ
	for (int k = 1; k != height - 2; ++k)
	{
		for (int l = 1; l != width - 2; ++l)
		{
			cv::Rect rct;
			rct.x = l - 1;
			rct.y = k - 1;
			rct.height = 3;
			rct.width = 3;
			if ((sobelMat.at<uchar>(k, l) == 255) && (cv::countNonZero(bw_blue(rct)) >= 1))
				bw_blue_edge.at<uchar>(k, l) = 255;
		}
	}
	