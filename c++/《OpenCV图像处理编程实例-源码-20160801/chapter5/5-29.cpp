// ���ܣ����� 5-29 HSV ͨ������
// ���ߣ���ΰ zhu1988wei@163.com
// ��Դ����OpenCVͼ������ʵ����
// ���ͣ�http://blog.csdn.net/zhuwei1988
// ���£�2016-8-1
// ˵������Ȩ���У����û�ժ¼����ϵ���ߣ������������ʽע��������лл��    // 
	// ת����hsv 
	cv::Mat img_h, img_s, img_v, imghsv;
	std::vector<cv::Mat> hsv_vec;
	cv::cvtColor(srcImage, imghsv, CV_BGR2HSV);
	cv::imshow("hsv", imghsv);
	cv::waitKey(0);
	// �ָ�hsvͨ��
	cv::split(imghsv, hsv_vec);
	img_h = hsv_vec[0];
	img_s = hsv_vec[1];
	img_v = hsv_vec[2];
	img_h.convertTo(img_h, CV_32F);
	img_s.convertTo(img_s, CV_32F);
	img_v.convertTo(img_v, CV_32F);
	double max_s, max_h, max_v;
	cv::minMaxIdx(img_h, 0, &max_h);
	cv::minMaxIdx(img_s, 0, &max_s);
	cv::minMaxIdx(img_v, 0, &max_v);
	// ����ͨ����һ��
	img_h /= max_h;
	img_s /= max_s;
	img_v /= max_v;