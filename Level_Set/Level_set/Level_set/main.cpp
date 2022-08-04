﻿#include<iostream>
#include<opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include"LevelSet.h"

using namespace std;
using namespace cv;

Mat Gabor7(Mat img_1);
Mat butterworth_high_kernel(cv::Mat &scr, float sigma, int n);
Mat butterworth_high_pass_filter(cv::Mat &src, float d0, int n);
void fftshift(cv::Mat &plane0, cv::Mat &plane1);
Mat frequency_filter(cv::Mat &scr, cv::Mat &blur);
void getcart(int rows, int cols, cv::Mat &x, cv::Mat &y);
Mat image_make_border(cv::Mat &src);

// 巴特沃斯高通滤波核函数
cv::Mat butterworth_high_kernel(cv::Mat &scr, float sigma, int n)
{
	cv::Mat butterworth_high_pass(scr.size(), CV_32FC1); //，CV_32FC1
	float D0 = (float)sigma;  // 半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			float d = sqrt(pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2));//分子,计算pow必须为float型
			butterworth_high_pass.at<float>(i, j) = 1.0f - 1.0f / (1.0f + pow(d / D0, 2 * n));
		}
	}
	return butterworth_high_pass;
}

// 巴特沃斯高通滤波
cv::Mat butterworth_high_pass_filter(cv::Mat &src, float d0, int n)
{
	cv::Mat padded = image_make_border(src);
	cv::Mat butterworth_kernel = butterworth_high_kernel(padded, d0, n);
	cv::Mat result = frequency_filter(padded, butterworth_kernel);
	return result;
}

// 频率域滤波
cv::Mat frequency_filter(cv::Mat &scr, cv::Mat &blur)
{
	cv::Mat mask = scr == scr;
	scr.setTo(0.0f, ~mask);

	//创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
	cv::Mat plane[] = { scr.clone(), cv::Mat::zeros(scr.size() , CV_32FC1) };

	cv::Mat complexIm;
	cv::merge(plane, 2, complexIm); // 合并通道 （把两个矩阵合并为一个2通道的Mat类容器）
	cv::dft(complexIm, complexIm); // 进行傅立叶变换，结果保存在自身

	// 分离通道（数组分离）
	cv::split(complexIm, plane);

	// 以下的操作是频域迁移
	fftshift(plane[0], plane[1]);

	// *****************滤波器函数与DFT结果的乘积****************
	cv::Mat blur_r, blur_i, BLUR;
	cv::multiply(plane[0], blur, blur_r);  // 滤波（实部与滤波器模板对应元素相乘）
	cv::multiply(plane[1], blur, blur_i);  // 滤波（虚部与滤波器模板对应元素相乘）
	cv::Mat plane1[] = { blur_r, blur_i };

	// 再次搬移回来进行逆变换
	fftshift(plane1[0], plane1[1]);
	cv::merge(plane1, 2, BLUR); // 实部与虚部合并

	cv::idft(BLUR, BLUR);       // idft结果也为复数
	BLUR = BLUR / BLUR.rows / BLUR.cols;

	cv::split(BLUR, plane);//分离通道，主要获取通道

	return plane[0];
}

// 图像边界处理
cv::Mat image_make_border(cv::Mat &src)
{
	int w = cv::getOptimalDFTSize(src.cols); // 获取DFT变换的最佳宽度
	int h = cv::getOptimalDFTSize(src.rows); // 获取DFT变换的最佳高度

	cv::Mat padded;
	// 常量法扩充图像边界，常量 = 0
	cv::copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);

	return padded;
}

// 实现频域滤波器的网格函数
void getcart(int rows, int cols, cv::Mat &x, cv::Mat &y) {
	x.create(rows, cols, CV_32FC1);
	y.create(rows, cols, CV_32FC1);
	//设置边界

	//计算其他位置的值
	for (int i = 0; i < rows; ++i) {
		if (i <= rows / 2) {
			x.row(i) = i;
		}
		else {
			x.row(i) = i - rows;
		}
	}
	for (int i = 0; i < cols; ++i) {
		if (i <= cols / 2) {
			y.col(i) = i;
		}
		else {
			y.col(i) = i - cols;
		}
	}
}

// fft变换后进行频谱搬移
void fftshift(cv::Mat &plane0, cv::Mat &plane1)
{
	// 以下的操作是移动图像  (零频移到中心)
	int cx = plane0.cols / 2;
	int cy = plane0.rows / 2;
	cv::Mat part1_r(plane0, cv::Rect(0, 0, cx, cy));  // 元素坐标表示为(cx, cy)
	cv::Mat part2_r(plane0, cv::Rect(cx, 0, cx, cy));
	cv::Mat part3_r(plane0, cv::Rect(0, cy, cx, cy));
	cv::Mat part4_r(plane0, cv::Rect(cx, cy, cx, cy));

	cv::Mat temp;
	part1_r.copyTo(temp);  //左上与右下交换位置(实部)
	part4_r.copyTo(part1_r);
	temp.copyTo(part4_r);

	part2_r.copyTo(temp);  //右上与左下交换位置(实部)
	part3_r.copyTo(part2_r);
	temp.copyTo(part3_r);

	cv::Mat part1_i(plane1, cv::Rect(0, 0, cx, cy));  //元素坐标(cx,cy)
	cv::Mat part2_i(plane1, cv::Rect(cx, 0, cx, cy));
	cv::Mat part3_i(plane1, cv::Rect(0, cy, cx, cy));
	cv::Mat part4_i(plane1, cv::Rect(cx, cy, cx, cy));

	part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
	part4_i.copyTo(part1_i);
	temp.copyTo(part4_i);

	part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
	part3_i.copyTo(part2_i);
	temp.copyTo(part3_i);
}


LevelSet::LevelSet()
{
	m_iterNum = 300;
	m_lambda1 = 1;
	m_nu = 0.00001 * 255 * 255;
	m_mu = 1.0;
	m_timestep = 0.1;
	m_epsilon = 1.0;
}


LevelSet::~LevelSet()
{
}

void LevelSet::initializePhi(Mat img, int iterNum, Rect boxPhi)
{
	//boxPhi是前景区域  
	m_iterNum = iterNum;
	//cvtColor(img, m_mImage, CV_BGR2GRAY);
	m_mImage = img;
	m_iCol = img.cols;
	m_iRow = img.rows;
	m_depth = CV_32FC1;

	//显式分配内存  
	m_mPhi = Mat::zeros(m_iRow, m_iCol, m_depth);
	m_mDirac = Mat::zeros(m_iRow, m_iCol, m_depth);
	m_mHeaviside = Mat::zeros(m_iRow, m_iCol, m_depth);

	//初始化惩罚性卷积核  
	m_mK = (Mat_<float>(3, 3) << 0.5, 1, 0.5,
		1, -6, 1,
		0.5, 1, 0.5);

	int c = 2;
	for (int i = 0; i < m_iRow; i++)
	{
		for (int j = 0; j < m_iCol; j++)
		{
			if (i<boxPhi.y || i>boxPhi.y + boxPhi.height || j<boxPhi.x || j>boxPhi.x + boxPhi.width)
			{
				m_mPhi.at<float>(i, j) = -c;
			}
			else
			{
				m_mPhi.at<float>(i, j) = c;
			}
		}
	}
}

void LevelSet::Dirac()
{
	//狄拉克函数  
	float k1 = m_epsilon / CV_PI;
	float k2 = m_epsilon * m_epsilon;
	for (int i = 0; i < m_iRow; i++)
	{
		float *prtDirac = &(m_mDirac.at<float>(i, 0));
		float *prtPhi = &(m_mPhi.at<float>(i, 0));

		for (int j = 0; j < m_iCol; j++)
		{
			float *prtPhi = &(m_mPhi.at<float>(i, 0));
			prtDirac[j] = k1 / (k2 + prtPhi[j] * prtPhi[j]);
		}
	}
}

void LevelSet::Heaviside()
{
	//海氏函数  
	float k3 = 2 / CV_PI;
	for (int i = 0; i < m_iRow; i++)
	{
		float *prtHeaviside = (float *)m_mHeaviside.ptr(i);
		float *prtPhi = (float *)m_mPhi.ptr(i);

		for (int j = 0; j < m_iCol; j++)
		{
			prtHeaviside[j] = 0.5 * (1 + k3 * atan(prtPhi[j] / m_epsilon));
		}
	}
}

void LevelSet::Curvature()
{
	//计算曲率  
	Mat dx, dy;
	Sobel(m_mPhi, dx, m_mPhi.depth(), 1, 0, 1);
	Sobel(m_mPhi, dy, m_mPhi.depth(), 0, 1, 1);

	for (int i = 0; i < m_iRow; i++)
	{
		float *prtdx = (float *)dx.ptr(i);
		float *prtdy = (float *)dy.ptr(i);
		for (int j = 0; j < m_iCol; j++)
		{
			float val = sqrtf(prtdx[j] * prtdx[j] + prtdy[j] * prtdy[j] + 1e-10);
			prtdx[j] = prtdx[j] / val;
			prtdy[j] = prtdy[j] / val;
		}
	}
	Mat ddx, ddy;
	Sobel(dx, ddy, m_mPhi.depth(), 0, 1, 1);
	Sobel(dy, ddx, m_mPhi.depth(), 1, 0, 1);
	m_mCurv = ddx + ddy;
}

void LevelSet::BinaryFit()
{
	//先计算海氏函数  
	Heaviside();

	//计算前景与背景灰度均值  
	float sumFG = 0;
	float sumBK = 0;
	float sumH = 0;
	//float sumFH = 0;  
	Mat temp = m_mHeaviside;
	Mat temp2 = m_mImage;
	float fHeaviside;
	float fFHeaviside;
	float fImgValue;
	for (int i = 1; i < m_iRow; i++)
	{
		float *prtHeaviside = &(m_mHeaviside.at<float>(i, 0));
		uchar *prtImgValue = &(m_mImage.at<uchar>(i, 0));
		for (int j = 1; j < m_iCol; j++)
		{
			fImgValue = prtImgValue[j];
			fHeaviside = prtHeaviside[j];
			fFHeaviside = 1 - fHeaviside;

			sumFG += fImgValue * fHeaviside;
			sumBK += fImgValue * fFHeaviside;
			sumH += fHeaviside;
		}
	}
	m_FGValue = sumFG / (sumH + 1e-10);         //前景灰度均值  
	m_BKValue = sumBK / (m_iRow*m_iCol - sumH + 1e-10); //背景灰度均值  
}
Mat showIMG;
void LevelSet::EVolution()
{
	float fCurv;
	float fDirac;
	float fPenalize;
	float fImgValue;

	for (int i = 0; i < m_iterNum; i++)
	{
		Dirac();
		Curvature();
		BinaryFit();
		filter2D(m_mPhi, m_mPenalize, m_depth, m_mK, Point(1, 1));//惩罚项的△φ  
		for (int i = 0; i < m_iRow; i++)
		{
			float *prtCurv = &(m_mCurv.at<float>(i, 0));
			float *prtDirac = &(m_mDirac.at<float>(i, 0));
			float *prtPenalize = &(m_mPenalize.at<float>(i, 0));
			uchar *prtImgValue = &(m_mImage.at<uchar>(i, 0));
			for (int j = 0; j < m_iCol; j++)
			{
				fCurv = prtCurv[j];
				fDirac = prtDirac[j];
				fPenalize = prtPenalize[j];
				fImgValue = prtImgValue[j];

				float lengthTerm = m_nu * fDirac * fCurv;                    //长度约束  
				float penalizeTerm = m_mu * (fPenalize - fCurv);                  //惩罚项  
				float areaTerm = fDirac * m_lambda1 *                       //全局项  
					(-((fImgValue - m_FGValue)*(fImgValue - m_FGValue))
						+ ((fImgValue - m_BKValue)*(fImgValue - m_BKValue)));

				m_mPhi.at<float>(i, j) = m_mPhi.at<float>(i, j) + m_timestep * (lengthTerm + penalizeTerm + areaTerm);
			}
		}

		//显示每一次演化的结果  

		cvtColor(m_mImage, showIMG, CV_GRAY2BGR);
		Mat Mask = m_mPhi >= 0;   //findContours的输入是二值图像  
		dilate(Mask, Mask, Mat(), Point(-1, -1), 3);
		erode(Mask, Mask, Mat(), Point(-1, -1), 3);
		vector<vector<Point> > contours;
		findContours(Mask,
			contours,// 轮廓点  
			RETR_EXTERNAL,// 只检测外轮廓  
			CHAIN_APPROX_NONE);// 提取轮廓所有点  
		drawContours(showIMG, contours, -1, Scalar(255, 0, 0), 2);
		namedWindow("Level Set后图像");
		imshow("Level Set后图像", showIMG);
		waitKey(1);
		//return showIMG;
	}
}
void main()
{
	//Mat img_IN = imread("D:\\Postgraduate\\09_project\\01_image\\07_Git\\BLCB_VS\\Level_Set\\B0ROI.bmp", -1);
	Mat img_IN = imread("D:\\test\\verify\\Carell\\B03-0804-膜材打折 膜拱样本\\1138BG\\BRROI.bmp", -1);
	Mat img_R = img_IN(Rect(0, 0, img_IN.cols, 500)).clone();
	Mat img_R_IN = img_IN(Rect(0, 0, img_IN.cols, 500)).clone();
	//Mat img_R = img_IN.clone();
	//img_R = butterworth_high_pass_filter(img_R,5,4);
	GaussianBlur(img_R, img_R, Size(513, 513), 120);        //白底滤波
	
	Mat img_IN_New ;
	divide(img_R_IN, img_R, img_IN_New,255);

	threshold(img_IN_New, img_IN_New,250, 255, CV_THRESH_BINARY);
	//img_IN_New = Gabor7(img_IN_New);
	medianBlur(img_IN_New, img_IN_New, 3);
	threshold(img_IN_New, img_IN_New, 250, 255, CV_THRESH_BINARY);
	Ptr<CLAHE> clahe = createCLAHE(2, Size(30, 30));
	Mat img;
	clahe->apply(img_R, img);   //整图增强
	//imshow("原图", img);
	img = Gabor7(img);
	adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 171, -1);//21   -1
	img = img_IN_New;
	Mat struct1 = getStructuringElement(0, Size(3, 3));  //矩形结构元素
	erode(img, img, struct1);	//腐蚀 减少噪声
	//img = ~img;
	
	Rect rec(0, 0, img.cols, img.rows);
	LevelSet ls;
	ls.initializePhi(img, 10, rec);
	ls.EVolution();
	//imshow("Level Set后图像", showIMG);
	waitKey(0);
}

/*=========================================================
* 函 数 名: Gabor7
* 功能描述: gabor滤波
=========================================================*/
Mat Gabor7(Mat img_1)
{                               //(核       ，𝜎    ，𝜃，     𝜆，   𝛾  𝜑，数据类型 )
	Mat kernel1 = getGaborKernel(Size(7, 7), 2.7, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//求卷积核
	float sum = 0.0;
	for (int i = 0; i < kernel1.rows; i++)
	{
		for (int j = 0; j < kernel1.cols; j++)
		{
			sum = sum + kernel1.ptr<float>(i)[j];
		}
	}
	Mat mmm = kernel1 / sum;
	Mat kernel2 = getGaborKernel(Size(7, 7), 2.7, 0, 1.0, 1.0, 0, CV_32F);
	float sum2 = 0.0;
	for (int i = 0; i < kernel2.rows; i++)
	{
		for (int j = 0; j < kernel2.cols; j++)
		{
			sum2 = sum2 + kernel2.ptr<float>(i)[j];
		}
	}
	Mat mmm2 = kernel2 / sum2;
	Mat img_4, img_5;
	filter2D(img_1, img_4, CV_8UC3, mmm);//卷积运算
	filter2D(img_4, img_5, CV_8UC3, mmm2);
	return img_5;
}

void ssr(Mat src, Mat& dst, double sigma) {
	Mat src_log, gauss, gauss_log, dst_log;
	src_log = Mat(src.size(), CV_32FC3);
	gauss_log = Mat(src.size(), CV_32FC3);
	dst_log = Mat(src.size(), CV_32FC3);
	dst = Mat(src.size(), CV_32FC3);
	int height = dst_log.rows;
	int width = dst_log.cols;
	int ksize = (int)(sigma * 3 / 2);
	ksize = ksize * 2 + 1;
	//求Log(S(x,y)
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < 3; k++) {
				float value = src.at<Vec3b>(i, j)[k];
				if (value <= 0.01) value = 0.01;
				src_log.at<Vec3f>(i, j)[k] = log10(value);
			}
		}
	}
	GaussianBlur(src, gauss, Size(ksize, ksize), sigma, sigma, 4);
	//求Log（L(x,y)）
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < 3; k++) {
				float value = gauss.at<Vec3b>(i, j)[k];
				if (value <= 0.01) value = 0.01;
				gauss_log.at<Vec3f>(i, j)[k] = log10(value);
			}
		}
	}
	//求Log（S（x,y）)-Log(L(x,y))
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < 3; k++) {
				float value1 = src_log.at<Vec3f>(i, j)[k];
				float value2 = gauss_log.at<Vec3f>(i, j)[k];
				dst_log.at<Vec3f>(i, j)[k] = value1 - value2;
			}
		}
	}
	float min[3] = { dst_log.at<Vec3f>(0, 0)[0], dst_log.at<Vec3f>(0, 0)[1],dst_log.at<Vec3f>(0, 0)[2] };
	float max[3] = { dst_log.at<Vec3f>(0, 0)[0], dst_log.at<Vec3f>(0, 0)[1],dst_log.at<Vec3f>(0, 0)[2] };
	//求R/G/B三通道的min,max
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < 3; k++) {
				float value = dst_log.at<Vec3f>(i, j)[k];
				if (value > max[k]) max[k] = value;
				if (value < min[k]) min[k] = value;
			}
		}
	}
	//量化处理
	cout << min[0] << " " << min[1] << " " << min[2] << endl;
	cout << max[0] << " " << max[1] << " " << max[2] << endl;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < 3; k++) {
				float value = dst_log.at<Vec3f>(i, j)[k];
				dst.at<Vec3f>(i, j)[k] = (saturate_cast<float>(255 * (value - min[k]) / (max[k] - min[k])));
			}
		}
	}
	dst.convertTo(dst, CV_8UC3);
	return;
}
