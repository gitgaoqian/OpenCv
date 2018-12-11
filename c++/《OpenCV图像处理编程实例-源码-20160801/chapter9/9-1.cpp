// ���ܣ����� 9-1 �����˶������
// ���ߣ���ΰ zhu1988wei@163.com
// ��Դ����OpenCVͼ������ʵ����
// ���ͣ�http://blog.csdn.net/zhuwei1988
// ���£�2016-8-1
// ˵������Ȩ���У����û�ժ¼����ϵ���ߣ������������ʽע��������лл��// 
// good ���������
cv::Mat getRansacMat(const std::vector<cv::DMatch>& matches,
const std::vector<cv::KeyPoint>& keypoints1,
const std::vector<cv::KeyPoint>& keypoints2,
std::vector<cv::DMatch>& outMatches)
{
	// ת���������ʽ

	std::vector<cv::Point2f> points1, points2;
	cv::Mat featureMat;
	for (std::vector<cv::DMatch>::const_iterator it =
	matches.begin(); it!= matches.end(); ++it)
	{
		// ��ȡ����������λ��
		float x= keypoints1[it->queryIdx].pt.x;
		float y= keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x,y));
		// ��ȡ��������Ҳ�λ��
		x= keypoints2[it->trainIdx].pt.x;
		y= keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x,y));
	}
	// ���� good ��������
	std::vector<uchar> inliers(points1.size(),0);
	if (points1.size()>0 &&points2.size()>0)
	{
		// ��������ͼ��Ķ�Ӧ�����������
		cv::Mat featureMat= cv::findFundamentalMat(
		cv::Mat(points1),cv::Mat(points2), inliers,
		CV_FM_RANSAC,distance,confidence);
		// ��ȡ����������ƥ��
		std::vector<uchar>::const_iterator
		itIn= inliers.begin();
		std::vector<cv::DMatch>::const_iterator
		itM= matches.begin();
		// ��������������
		for ( ;itIn!= inliers.end(); ++itIn, ++itM)
		{
			if (*itIn)
			outMatches.push_back(*itM);
		}
		if (refineF)
		{
			points1.clear();
			points2.clear();
			for (std::vector<cv::DMatch>::
			const_iterator it= outMatches.begin();
			it!= outMatches.end(); ++it)
			{
				float x= keypoints1[it->queryIdx].pt.x;
				float y= keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x,y));
				x= keypoints2[it->trainIdx].pt.x;
				y= keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x,y));
			}
			// ��������ͼ�����������
			if (points1.size()>0 && points2.size()>0)
			{

				featureMat= cv::findFundamentalMat(
				cv::Mat(points1),cv::Mat(points2),
				CV_FM_8POINT);
			}
		}
	}
	return featureMat;
 }