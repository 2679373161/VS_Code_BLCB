#include<iostream>
#include<opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include"LevelSet.h"
using namespace std;
using namespace cv;


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
	//boxPhi��ǰ������  
	m_iterNum = iterNum;
	//cvtColor(img, m_mImage, CV_BGR2GRAY);
	m_mImage = img;
	m_iCol = img.cols;
	m_iRow = img.rows;
	m_depth = CV_32FC1;

	//��ʽ�����ڴ�  
	m_mPhi = Mat::zeros(m_iRow, m_iCol, m_depth);
	m_mDirac = Mat::zeros(m_iRow, m_iCol, m_depth);
	m_mHeaviside = Mat::zeros(m_iRow, m_iCol, m_depth);

	//��ʼ���ͷ��Ծ����  
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
	//�����˺���  
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
	//���Ϻ���  
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
	//��������  
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
	//�ȼ��㺣�Ϻ���  
	Heaviside();

	//����ǰ���뱳���ҶȾ�ֵ  
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
	m_FGValue = sumFG / (sumH + 1e-10);         //ǰ���ҶȾ�ֵ  
	m_BKValue = sumBK / (m_iRow*m_iCol - sumH + 1e-10); //�����ҶȾ�ֵ  
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
		filter2D(m_mPhi, m_mPenalize, m_depth, m_mK, Point(1, 1));//�ͷ���ġ���  
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

				float lengthTerm = m_nu * fDirac * fCurv;                    //����Լ��  
				float penalizeTerm = m_mu * (fPenalize - fCurv);                  //�ͷ���  
				float areaTerm = fDirac * m_lambda1 *                       //ȫ����  
					(-((fImgValue - m_FGValue)*(fImgValue - m_FGValue))
						+ ((fImgValue - m_BKValue)*(fImgValue - m_BKValue)));

				m_mPhi.at<float>(i, j) = m_mPhi.at<float>(i, j) + m_timestep * (lengthTerm + penalizeTerm + areaTerm);
			}
		}

		//��ʾÿһ���ݻ��Ľ��  

		cvtColor(m_mImage, showIMG, CV_GRAY2BGR);
		Mat Mask = m_mPhi >= 0;   //findContours�������Ƕ�ֵͼ��  
		dilate(Mask, Mask, Mat(), Point(-1, -1), 3);
		erode(Mask, Mask, Mat(), Point(-1, -1), 3);
		vector<vector<Point> > contours;
		findContours(Mask,
			contours,// ������  
			RETR_EXTERNAL,// ֻ���������  
			CHAIN_APPROX_NONE);// ��ȡ�������е�  
		drawContours(showIMG, contours, -1, Scalar(255, 0, 0), 2);
		namedWindow("Level Set��ͼ��");
		imshow("Level Set��ͼ��", showIMG);
		waitKey(1);
		//return showIMG;
	}
}
void main()
{
	Mat img_IN = imread("D:\\Postgraduate\\09_project\\01_image\\07_Git\\BLCB_VS\\Level_Set\\B0ROI.bmp", -1);
	Mat img_R = img_IN(Rect(1000, 0, img_IN.cols-1500, 500)).clone();

	Ptr<CLAHE> clahe = createCLAHE(2, Size(30, 30));
	Mat img;
	clahe->apply(img_R, img);   //��ͼ��ǿ

	imshow("ԭͼ", img);
	Rect rec(0, 0, img.cols, img.rows);
	LevelSet ls;
	ls.initializePhi(img, 50, rec);
	ls.EVolution();
	//imshow("Level Set��ͼ��", showIMG);
	waitKey(0);
}