// ���ܣ����� 5-30 ϸ�� Sobel ��Ե��ȡ
// ���ߣ���ΰ zhu1988wei@163.com
// ��Դ����OpenCVͼ������ʵ����
// ���ͣ�http://blog.csdn.net/zhuwei1988
// ���£�2016-8-1
// ˵������Ȩ���У����û�ժ¼����ϵ���ߣ������������ʽע��������лл��// 
// ��ȡ��ֱ��sobel��Ե
bool SobelVerEdge(cv::Mat srcImage, cv::Mat& resultImage)
{
	CV_Assert(srcImage.channels() == 1);
	srcImage.convertTo(srcImage, CV_32FC1);
	// ˮƽ����� Sobel ����
	cv::Mat sobelx = (cv::Mat_<float>(3, 3) << -0.125, 0, 0.125,
		-0.25, 0, 0.25,
		-0.125, 0, 0.125);
	cv::Mat ConResMat;
	// �������
	cv::filter2D(srcImage, ConResMat, srcImage.type(), sobelx);
	// �����ݶȵķ���
	cv::Mat graMagMat;
	cv::multiply(ConResMat, ConResMat, graMagMat);
	// �����ݶȷ��ȼ�����������ֵ
	int scaleVal = 4;
	double thresh = scaleVal * cv::mean(graMagMat).val[0];
	cv::Mat resultTempMat = cv::Mat::zeros(
		graMagMat.size(), graMagMat.type());
	float* pDataMag = (float*)graMagMat.data;
	float* pDataRes = (float*)resultTempMat.data;
	const int nRows = ConResMat.rows;
	const int nCols = ConResMat.cols;
	for (int i = 1; i != nRows - 1; ++i) {
		for (int j = 1; j != nCols - 1; ++j) {
			// ����õ��ݶ���ˮƽ��ֱ�ݶ�ֵ��С�ȽϽ��
			bool b1 = (pDataMag[i * nCols + j] > pDataMag[i *
				nCols + j - 1]);
			bool b2 = (pDataMag[i * nCols + j] > pDataMag[i *
				nCols + j + 1]);
			bool b3 = (pDataMag[i * nCols + j] > pDataMag[(i - 1)
				* nCols + j]);
			bool b4 = (pDataMag[i * nCols + j] > pDataMag[(i + 1)
				* nCols + j]);
			// �ж������ݶ��Ƿ��������ˮƽ��ֱ�ݶ�
			// ����������Ӧ��ֵ�������ж�ֵ��
			pDataRes[i * nCols + j] = 255 * ((pDataMag[i *
				nCols + j] > thresh) &&
				((b1 && b2) || (b3 && b4)));
		}
	}
	resultTempMat.convertTo(resultTempMat, CV_8UC1);
	resultImage = resultTempMat.clone();
	return true;
}