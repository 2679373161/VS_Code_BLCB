#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <string.h>
//#include "PersTrans.h"
using namespace cv;
using namespace std;

bool Ext_Result_Left_Right;
bool Ext_Result_Front_Back;
bool isArea_1, isArea_2;														//��ʾ�쳣��־λ
String Screen_Type = "R��ˮ����";

Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB);
int lis(vector<int> arr, int len);
void convex_edg(Mat src, Mat &dst, int min_area);
void predict_edg(Mat& dst, vector<int> &value, vector<int>value_num, bool is_reverse);
void predict_num(Mat& dst, vector<int>& value);
void DeleteElem(vector <int> &vec, int elem);

/*========================================================
 *@�� �� ����             convexSetPretreatment
 *@����������             ��ͼ����͹������ȡ��͹�����壬�ı�������ɵ�͸�ӱ任�и׼ȷ�����
 *@Mat                    ԭͼ��
 *@Mat                    ��Ӧ͹��ͼ��
 *@�޸�ʱ�䣺             2021/03/16
 *=======================================================*/
void convexSetPretreatment(Mat& src) 
{
	//�ն�Ԥ���� ����������
	Mat src_copy = src.clone();
	Mat threshold_output;
	vector<vector<Point> > preContours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	//threshold(src, threshold_output, 30, 255, THRESH_BINARY);
	
	//threshold(src_copy, threshold_output, 30, 255, THRESH_BINARY);
	double meanGray = mean(src_copy)[0];
	threshold(src_copy, threshold_output, meanGray * 0.6, 255, THRESH_BINARY);
	
	/// Find contours
	findContours(threshold_output, preContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(preContours.size());

	for (size_t i = 0; i < preContours.size(); i++)
	{
		convexHull(Mat(preContours[i]), hull[i], false);
	}

	/// Draw contours + hull results
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC1);
	for (size_t i = 0; i < preContours.size(); i++)
	{
		double area = contourArea(preContours[i]);
		if (area > 100000)
		{
			drawContours(drawing, preContours, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
			drawContours(drawing, hull, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
		}
	}
	src =  drawing;
}
/*=========================================================
* �� �� ��: toushi_white
* ��������: ͸�ӱ任ͼ�����
=========================================================*/
Mat toushi_white(Mat image, Mat M, int border, int length, int width)
{
	Mat perspective;
	cv::warpPerspective(image, perspective, M, cv::Size(length, width), cv::INTER_LINEAR);
	return perspective;
}


/*=========================================================
*@�� �� ��:              f_MainCam_PersTransMatCal
*@��������:              ���ڰ�/��ɫ���R����Ļ��͸�ӱ任�������
*@param _src             ����Ҷ�/��ɫͼ��
*@param _dst             �����ʾ���ͻ���ͼ��
*@param border_white     ��׵�ͼ��Ե��������ֵ
*@param border_black     ��ڵ�ͼ��Ե��������ֵ
*@param border_lightleak ��©��ͼ��Ե��������ֵ
*@param Mwhite           �׵�͸�ӱ任����
*@param Mblack           �ڵ�͸�ӱ任����
*@param Mlightleak       ©��͸�ӱ任����
*@param M_white_abshow   ��ʾ�쳣�任����
*@param ID               ��λID��(����)
*@ScreenType_Flag        ��Ļ����
*@����ʱ�䣺		     2020��8��17��
*@��ע˵��
=========================================================*/
//bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat* Mwhite, Mat* Mbiankuang, Mat* M_white_abshow, int ID, String ScreenType_Flag, int leftRightWhiteFlag)
//{
//	//    double screen_long=size_long/size_width;
//	//    int screen_long=size_long/size_width;
//	bool isArea_1, isArea_2;														//��ʾ�쳣��־λ
//	Mat src = _src.getMat();                                                        //����Դͼ��
//	if (src.type() == CV_8UC1)														//������8λͼ
//		src = src.clone();															//����ԭͼ
//	else
//		cvtColor(src, src, CV_BGR2GRAY);										    //�ҶȻ���ɫͼ
//	if (leftRightWhiteFlag == 1)
//	{
//		try
//		{
//			convexSetPretreatment(src);
//		}
//		catch (const std::exception& e)
//		{
//			std::cout << e.what();
//		}
//
//	}
//	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
//	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��
//	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
//	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
//	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
//	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
//	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
//	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
//	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
//	vector<Point2f> src_corner(4), src_corner_biankuang(4), src_corner_abshow(4);   //�ĸ����ཻ�õ��ǵ����꣬©��ǵ㣬��ʾ�쳣�ǵ�
//	Rect rect;																        //��С����Ӿ���
//	int x1, y1, x2, y2, x3, y3, x4, y4;			//���Ӿ����������Ϣ
//	for (vector<int>::size_type i = 0; i < contours.size(); i++)
//	{
//		double area = contourArea(contours[i]);
//
//		Mat temp_mask = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC1);
//		drawContours(temp_mask, contours, i, 255, FILLED, 8);
//
//		if (area > 250000 && area < 5000000)
//		{
//
//			displayError_Areasignal++;
//			rect = boundingRect(contours[i]);
//			x1 = rect.tl().x;//���Ͻ�
//			y1 = rect.tl().y;//���Ͻ�
//			x2 = rect.tl().x;//���½�
//			y2 = rect.br().y;//���½�
//			x3 = rect.br().x;//���½�
//			y3 = rect.br().y;//���½�
//			x4 = rect.br().x;//���Ͻ�
//			y4 = rect.tl().y;//���Ͻ�
//			//int radianEliminate = 230;
//			//int deviation = 160;
//
//			int radianEliminate = 23;
//			int deviation = 16; //����
//			for (int j = 0; j < contours[i].size(); j++)
//			{
//				//���㼯
//				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1) * 0.3 && abs(contours[i][j].x - x1) < deviation ||
//					contours[i][j].y > y1 + (y2 - y1) * 0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
//					leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				//�Ҳ�㼯
//				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
//					rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				//�ϲ�㼯
//				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y1) < deviation)
//					upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				//�²�㼯
//				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y2) < deviation)
//					downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//			}
//			break;
//		}
//	}
//	if (leftLinePoint.size() == 0 || rightLinePoint.size() == 0 || upLinePoint.size() == 0 || downLinePoint.size() == 0)
//		displayError_Areasignal = 0;
//	//������������ж���ʾ�쳣
//	if (displayError_Areasignal > 0 && ID == 1)
//		isArea_1 = false;
//	if (displayError_Areasignal == 0 && ID == 1)
//		isArea_1 = true;
//	if (displayError_Areasignal > 0 && ID == 2)
//		isArea_2 = false;
//	if (displayError_Areasignal == 0 && ID == 2)
//		isArea_2 = true;
//	//δ��ȡ����Ļ�ж���ʾ�쳣��ȡ��Ե����
//	if (displayError_Areasignal == 0)
//	{
//		vector<Point2f> src_points(4);
//		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
//		vector<Point2f> dst_points(4);
//		if (ScreenType_Flag == "������")
//			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
//		else        //pixel_num
//			//dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
//			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
//		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
//		//*Mlightleak = cv::getPerspectiveTransform(src_points, dst_points);
//		*Mbiankuang = cv::getPerspectiveTransform(src_points, dst_points);
//		*M_white_abshow = cv::getPerspectiveTransform(src_points, dst_points);
//	}
//	//������Ļ��ȡ��Ļ���ĸ��ǵ�
//	else
//	{
//		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
//		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
//		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
//		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�²����ֱ��
//
//		/*Mat img = _src.getMat();
//		for (int i = 0; i < leftLinePoint.size(); i++)
//		{
//			circle(img, Point(leftLinePoint[i].x, leftLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//		}
//		for (int i = 0; i < rightLinePoint.size(); i++)
//		{
//			circle(img, Point(rightLinePoint[i].x, rightLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//		}
//		for (int i = 0; i < upLinePoint.size(); i++)
//		{
//			circle(img, Point(upLinePoint[i].x, upLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//		}
//		for (int i = 0; i < downLinePoint.size(); i++)
//		{
//			circle(img, Point(downLinePoint[i].x, downLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//		}*/
//
//		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
//		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
//		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
//		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�
//
//																						//��4���ǵ������λ�ý���΢�����׵�ͼ�Լ��ڵ�ͼ��
//		src_corner[0].x = src_corner[0].x - border_white;
//		src_corner[0].y = src_corner[0].y - border_white;
//		src_corner[1].x = src_corner[1].x - border_white;
//		src_corner[1].y = src_corner[1].y + border_white;
//		src_corner[2].x = src_corner[2].x + border_white;
//		src_corner[2].y = src_corner[2].y + border_white;
//		src_corner[3].x = src_corner[3].x + border_white;
//		src_corner[3].y = src_corner[3].y - border_white;
//		//��4���ǵ������λ�ý���΢����©����ͼ��
//		src_corner_biankuang[0].x = src_corner[0].x - border_biankuang;
//		src_corner_biankuang[0].y = src_corner[0].y - border_biankuang;
//		src_corner_biankuang[1].x = src_corner[1].x - border_biankuang;
//		src_corner_biankuang[1].y = src_corner[1].y + border_biankuang;
//		src_corner_biankuang[2].x = src_corner[2].x + border_biankuang;
//		src_corner_biankuang[2].y = src_corner[2].y + border_biankuang;
//		src_corner_biankuang[3].x = src_corner[3].x + border_biankuang;
//		src_corner_biankuang[3].y = src_corner[3].y - border_biankuang;
//		//��ʾ�쳣(�׵�ͼ)
//		src_corner_abshow[0].x = src_corner[0].x - border_white + 10;
//		src_corner_abshow[0].y = src_corner[0].y - border_white + 10;
//		src_corner_abshow[1].x = src_corner[1].x - border_white + 10;
//		src_corner_abshow[1].y = src_corner[1].y + border_white - 10;
//		src_corner_abshow[2].x = src_corner[2].x + border_white - 10;
//		src_corner_abshow[2].y = src_corner[2].y + border_white - 10;
//		src_corner_abshow[3].x = src_corner[3].x + border_white - 10;
//		src_corner_abshow[3].y = src_corner[3].y - border_white + 10;
//
//		vector<Point2f> dst_corner(4);
//		if (ScreenType_Flag == "������")
//		{
//			dst_corner[0] = Point(0, 0);
//			dst_corner[1] = Point(0, 1775);
//			dst_corner[2] = Point(3000, 1775);
//			dst_corner[3] = Point(3000, 0);
//		}
//		else
//		{
//			dst_corner[0] = Point(0, 0);
//			dst_corner[1] = Point(0, 1500);
//			//            dst_corner[2] = Point(3000, 1500);
//			//            dst_corner[3] = Point(3000, 0);
//			dst_corner[2] = Point(3000, 1500);
//			dst_corner[3] = Point(3000, 0);
//		}
//		*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
//		//*Mlightleak = cv::getPerspectiveTransform(src_corner_lightleak, dst_corner);
//		*Mbiankuang = cv::getPerspectiveTransform(src_corner_biankuang, dst_corner);
//		*M_white_abshow = cv::getPerspectiveTransform(src_corner_abshow, dst_corner);
//	}
//	if (ID == 1)
//		return isArea_1;
//	else
//		return isArea_2;
//}
//
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat* Mwhite, Mat* Mbiankuang, Mat* M_white_abshow, int ID, String ScreenType_Flag, int leftRightWhiteFlag)
{
	//    double screen_long=size_long/size_width;
	//    int screen_long=size_long/size_width;
	bool isArea_1, isArea_2;														//��ʾ�쳣��־λ
	Mat src = _src.getMat();                                                        //����Դͼ��
	if (src.type() == CV_8UC1)														//������8λͼ
		src = src.clone();															//����ԭͼ
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //�ҶȻ���ɫͼ
	//if (leftRightWhiteFlag == 1)
	{
		try
		{
			convexSetPretreatment(src);
		}
		catch (const std::exception& e)
		{
			std::cout << e.what();
		}

	}
	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
	vector<Point2f> src_corner(4), src_corner_biankuang(4), src_corner_abshow(4);   //�ĸ����ཻ�õ��ǵ����꣬©��ǵ㣬��ʾ�쳣�ǵ�
	Rect rect;																        //��С����Ӿ���
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 250000 && area < 5000000)
		{
			rect = boundingRect(contours[i]);
			Mat temp_mask = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC1);
			drawContours(temp_mask, contours, i, 255, FILLED, 8);
			displayError_Areasignal++;
			Mat sra_canny;
			Canny(temp_mask, sra_canny, 0, 255);
			//dilate(sra_canny, sra_canny, cv::getStructuringElement(0, cv::Size(3, 3)), Point(-1, -1));
			for (int i = 1; i < sra_canny.cols; i++) {
				for (int j = 1; j < sra_canny.rows; j++) {
					//���㼯
					for (int k = 1; k < 4; k++) {
						if (i < (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							leftLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//�Ҳ�㼯
						if (i > (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							rightLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//�ϲ�㼯
						if (j < (rect.y + rect.height / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + k, i + 7 * k) > 100 || sra_canny.at<uchar>(j - k, i + 7 * k) > 100 || sra_canny.at<uchar>(j, i + 5) > 100)) {
							upLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//�²�㼯
						if (j > (rect.y + rect.height / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + k, i + 7 * k) > 100 || sra_canny.at<uchar>(j - k, i + 7 * k) > 100 || sra_canny.at<uchar>(j, i + 5) > 100)) {
							downLinePoint.push_back(Point(i, j));
						}
					}
				}
			}

			break;
		}
	}
	if (leftLinePoint.size() == 0 || rightLinePoint.size() == 0 || upLinePoint.size() == 0 || downLinePoint.size() == 0)
		displayError_Areasignal = 0;
	//������������ж���ʾ�쳣
	if (displayError_Areasignal > 0 && ID == 1)
		isArea_1 = false;
	if (displayError_Areasignal == 0 && ID == 1)
		isArea_1 = true;
	if (displayError_Areasignal > 0 && ID == 2)
		isArea_2 = false;
	if (displayError_Areasignal == 0 && ID == 2)
		isArea_2 = true;
	//δ��ȡ����Ļ�ж���ʾ�쳣��ȡ��Ե����
	if (displayError_Areasignal == 0)
	{
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "������")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else        //pixel_num
			//dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
		//*Mlightleak = cv::getPerspectiveTransform(src_points, dst_points);
		*Mbiankuang = cv::getPerspectiveTransform(src_points, dst_points);
		*M_white_abshow = cv::getPerspectiveTransform(src_points, dst_points);
	}
	//������Ļ��ȡ��Ļ���ĸ��ǵ�
	else
	{
		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�²����ֱ��

		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�

																						//��4���ǵ������λ�ý���΢�����׵�ͼ�Լ��ڵ�ͼ��
		src_corner[0].x = src_corner[0].x - border_white;
		src_corner[0].y = src_corner[0].y - border_white;
		src_corner[1].x = src_corner[1].x - border_white;
		src_corner[1].y = src_corner[1].y + border_white;
		src_corner[2].x = src_corner[2].x + border_white;
		src_corner[2].y = src_corner[2].y + border_white;
		src_corner[3].x = src_corner[3].x + border_white;
		src_corner[3].y = src_corner[3].y - border_white;
		//��4���ǵ������λ�ý���΢����©����ͼ��
		src_corner_biankuang[0].x = src_corner[0].x - border_biankuang;
		src_corner_biankuang[0].y = src_corner[0].y - border_biankuang;
		src_corner_biankuang[1].x = src_corner[1].x - border_biankuang;
		src_corner_biankuang[1].y = src_corner[1].y + border_biankuang;
		src_corner_biankuang[2].x = src_corner[2].x + border_biankuang;
		src_corner_biankuang[2].y = src_corner[2].y + border_biankuang;
		src_corner_biankuang[3].x = src_corner[3].x + border_biankuang;
		src_corner_biankuang[3].y = src_corner[3].y - border_biankuang;
		//��ʾ�쳣(�׵�ͼ)
		src_corner_abshow[0].x = src_corner[0].x - border_white + 10;
		src_corner_abshow[0].y = src_corner[0].y - border_white + 10;
		src_corner_abshow[1].x = src_corner[1].x - border_white + 10;
		src_corner_abshow[1].y = src_corner[1].y + border_white - 10;
		src_corner_abshow[2].x = src_corner[2].x + border_white - 10;
		src_corner_abshow[2].y = src_corner[2].y + border_white - 10;
		src_corner_abshow[3].x = src_corner[3].x + border_white - 10;
		src_corner_abshow[3].y = src_corner[3].y - border_white + 10;

		vector<Point2f> dst_corner(4);
		if (ScreenType_Flag == "������")
		{
			dst_corner[0] = Point(0, 0);
			dst_corner[1] = Point(0, 1775);
			dst_corner[2] = Point(3000, 1775);
			dst_corner[3] = Point(3000, 0);
		}
		else
		{
			dst_corner[0] = Point(0, 0);
			dst_corner[1] = Point(0, 1500);
			//            dst_corner[2] = Point(3000, 1500);
			//            dst_corner[3] = Point(3000, 0);
			dst_corner[2] = Point(3000, 1500);
			dst_corner[3] = Point(3000, 0);
		}
		*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
		//*Mlightleak = cv::getPerspectiveTransform(src_corner_lightleak, dst_corner);
		*Mbiankuang = cv::getPerspectiveTransform(src_corner_biankuang, dst_corner);
		*M_white_abshow = cv::getPerspectiveTransform(src_corner_abshow, dst_corner);
	}
	if (ID == 1)
		return isArea_1;
	else
		return isArea_2;
}


/*=========================================================
*@�� �� ��:     getPointSlopeCrossPoint
*@��������:     �����бʽ����ֱ�ߵĽ���
*@param LineA   ƽ������
*@param LineB   ��ֱ����
*@����ʱ�䣺    2020��8��17��
*@��ע˵��
=========================================================*/
Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB)
{
	const double PI = 3.1415926535897;
	Point2f crossPoint;
	double kA = LineA[1] / LineA[0];
	double kB = LineB[1] / LineB[0];
	double theta = atan2(LineB[1], LineB[0]);
	if (theta == PI * 0.5)
	{
		crossPoint.x = LineB[0];
		crossPoint.y = kA * LineB[0] + LineA[3] - kA * LineA[2];
		return crossPoint;
	}
	double bA = LineA[3] - kA * LineA[2];
	double bB = LineB[3] - kB * LineB[2];
	crossPoint.x = (bB - bA) / (kA - kB);
	crossPoint.y = (kA * bB - kB * bA) / (kA - kB);
	return crossPoint;
}


/*=========================================================
*@�� �� ��:              f_FrontBackCam_PersTransMatCal
*@��������:              ǰ�����R��͸�ӱ任������㺯��
*@param _src             ����Ҷ�/��ɫͼ��
*@param Mwhite           �׵�͸�ӱ任����
*@ScreenType_Flag        ��Ļ����
*@����ʱ�䣺		     2020��8��21��
*@��ע˵��
=========================================================*/
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat* Mwhite, String ScreenType_Flag)
{
	bool Ext_Result_Front_Back;                                                     //��ȡ��Ļ�ɹ���־λ
	Mat src = _src.getMat();                                                        //����Դͼ��
	if (src.type() == CV_8UC1)														//������8λͼ
		src = src.clone();															//����ԭͼ
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //�ҶȻ���ɫͼ
	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
	vector<Point2f> src_corner(4);                                                  //�ĸ����ཻ�õ��ǵ�����
	Rect rect;																        //��С����Ӿ���
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //���Ӿ����������Ϣ
	vector<Point2f> dst_corner(4);                                                  //͸�ӱ任��ĵ����Ϣ
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 2000000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = cv::boundingRect(contours[i]);                                           //��С��Ӿ�����ȡ
			Ext_Result_Front_Back = false;                                                             //��ȡ����Ļ
			//���������С1/3�����õ��µľ��ζ���
			int PixelGap1 = rect.tl().x;
			int PixelGap2 = src.cols - (rect.tl().x + rect.width);
			//�������ȡ
			if (PixelGap1 > PixelGap2)
			{
				x1 = rect.tl().x;
				x2 = x1;
				x3 = (rect.br().x - x1) * 3 / 5 + x1;
				x4 = x3;
			}
			else
			{
				x3 = rect.br().x;
				x4 = x3;
				x1 = x3 - (x3 - rect.tl().x) * 3 / 5;
				x2 = x1;
			}
			//�������ȡ
			y1 = rect.tl().y;
			y2 = rect.br().y;
			if (y2 >= src.rows)
				y2 = src.rows - 1;
			y3 = y2;
			y4 = y1;
			//ȡֱ�ߵĲ�������
			int radianEliminate = 350;//(R��)����ʹ��
			int radianEliminate1 = 480;//(R��)����ʹ��
			int radianEliminate2 = 230;//(R��)����ʹ��
			int deviation = 120;//(б�ߴ��������)����ʹ��
			int deviation2 = 200;//(б�ߴ��������)����ʹ��
			if (PixelGap1 > PixelGap2)
			{
				//��Ӿ�����С
				while (binaryImage.at<uchar>(y3, x3) == 255 || binaryImage.at<uchar>(y4, x4) == 255)
				{
					x3 = x3 - 1;
					x4 = x4 - 1;

					if (x3 == 0 || x4 == 0)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y3, x3) != 255)
				{
					y3 = y3 - 1;
					y2 = y3;

					if (y2 == 0 || y3 == 0)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y4, x4) != 255)
				{
					y4 = y4 + 1;
					y1 = y4;

					if (y4 == src.rows - 1 || y1 == src.rows - 1)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				if (displayError_Areasignal != 0)
				{
					for (int j = 0; j < contours[i].size(); j++)
					{
						//���㼯
						if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate1 && abs(contours[i][j].x - x1) < deviation)
							leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�ϲ�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�²�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//�Ҳ�㼯
					rightLinePoint.push_back(Point((x3 + x4) / 2, (y3 + y4) / 2));
					if (leftLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//ֱ�����
						fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
						rightLine_Fit[0] = leftLine_Fit[0];
						rightLine_Fit[1] = leftLine_Fit[1];
						rightLine_Fit[2] = rightLinePoint[0].x;
						rightLine_Fit[3] = rightLinePoint[0].y;                                         //�Ҳ����ֱ��
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //�²����ֱ��
						//�ǵ���ȡ
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�
						//͸�ӱ任�������
						if (ScreenType_Flag == "������")
							dst_corner = { Point(0, 0), Point(0, 1775), Point(2000, 1775), Point(2000, 0) };
						else
							dst_corner = { Point(0, 0), Point(0, 1500), Point(2000, 1500), Point(2000, 0) };
						*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
					}
					else
					{
						displayError_Areasignal = 0;
						break;
					}
				}
			}
			else
			{
				//��Ӿ�����С
				while (binaryImage.at<uchar>(y1, x1) == 255 || binaryImage.at<uchar>(y2, x2) == 255)
				{
					x1 = x1 + 1;
					x2 = x2 + 1;

					if (x1 == src.cols - 1 || x1 == src.cols - 1)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y1, x1) != 255)
				{
					y1 = y1 + 1;
					y4 = y1;

					if (y1 == src.rows - 1 || y4 == src.rows - 1)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				while (binaryImage.at<uchar>(y2, x2) != 255)
				{
					y2 = y2 - 1;
					y3 = y2;

					if (y2 == 0 || y3 == 0)
					{
						displayError_Areasignal = 0;
						break;
					}
				}
				if (displayError_Areasignal != 0)
				{
					for (int j = 0; j < contours[i].size(); j++)
					{
						//�Ҳ�㼯
						if (contours[i][j].y > y1 + radianEliminate1 && contours[i][j].y < y1 + (y2 - y1) * 0.3 && abs(contours[i][j].x - x3) < deviation || contours[i][j].y > y1 + (y2 - y1) * 0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
							rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�ϲ�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//�²�㼯
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//���㼯
					leftLinePoint.push_back(Point((x1 + x2) / 2, (y1 + y2) / 2));
					if (rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//���ֱ��
						fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
						leftLine_Fit[0] = rightLine_Fit[0];
						leftLine_Fit[1] = rightLine_Fit[1];
						leftLine_Fit[2] = leftLinePoint[0].x;
						leftLine_Fit[3] = leftLinePoint[0].y;                                           //������ֱ��
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				    //�ϲ����ֱ��
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�²����ֱ��
						//�ǵ���ȡ
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�
						//͸�ӱ任�������
						if (ScreenType_Flag == "������")
							dst_corner = { Point(0, 0), Point(0, 1775), Point(2000, 1775), Point(2000, 0) };
						else
							dst_corner = { Point(0, 0), Point(0, 1500), Point(2000, 1500), Point(2000, 0) };
						*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
					}
					else
					{
						displayError_Areasignal = 0;
						break;
					}
				}
			}
		}
	}
	//û����ȡ����Ļ
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Front_Back = true; //û����ȡ����Ļ
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "������")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //͸�ӱ任������ȡ
	}

	return Ext_Result_Front_Back;
}

/*=========================================================
*@�� �� ��:              f_LeftRightCam_PersTransMatCal
*@��������:              �������R��͸�ӱ任������㺯��
*@param _src             ����Ҷ�/��ɫͼ��
*@param Mwhite           �׵�͸�ӱ任����
*@ScreenType_Flag        ��Ļ����
*@leftRightWhiteFlag     �׵����������־λ
*@����ʱ�䣺		     2021��03��15��
*@��ע˵��              use
=========================================================*/
//bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat* Mwhite, Mat* M_R_1_E, String ScreenType_Flag, int leftRightWhiteFlag, int border_white)
//{
//	bool Ext_Result_Left_Right;                                                     //��ȡ��Ļ�ɹ���־λ
//	Mat src = _src.getMat();                                                        //����Դͼ��
//	if (src.type() == CV_8UC1)														//������8λͼ
//		src = src.clone();															//����ԭͼ
//	else
//		cvtColor(src, src, CV_BGR2GRAY);                                            //�ҶȻ���ɫͼ
//
//	if (leftRightWhiteFlag == 1)
//	{
//		convexSetPretreatment(src);
//	}
//
//	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
//	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��
//	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
//	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
//	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
//	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
//	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
//	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
//	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
//	vector<Point2f> src_corner(4);                                                  //�ĸ����ཻ�õ��ǵ�����
//	vector<Point2f> src_corner_enlarge(4);
//	Rect rect;																        //��С����Ӿ���
//	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //���Ӿ����������Ϣ
//	vector<Point2f> dst_corner(4);                                                  //͸�ӱ任��ĵ����Ϣ
//	for (vector<int>::size_type i = 0; i < contours.size(); i++)
//	{
//		double area = contourArea(contours[i]);
//		if (area > 150000 && area < 600000)
//		{
//
//
//			displayError_Areasignal++;
//			rect = cv::boundingRect(contours[i]);                                           //��С��Ӿ�����ȡ
//			Ext_Result_Left_Right = false;                                                             //��ȡ����Ļ
//
//			//cv::rectangle(src, rect, Scalar(255, 0, 0), 5, LINE_8, 0);
//			x1 = rect.tl().x;//���Ͻ�
//			y1 = rect.tl().y;//���Ͻ�
//			x2 = rect.tl().x;//���½�
//			y2 = rect.br().y;//���½�
//			x3 = rect.br().x;//���½�
//			y3 = rect.br().y;//���½�
//			x4 = rect.br().x;//���Ͻ�
//			y4 = rect.tl().y;//���Ͻ�
//																									   //���������С1/3�����õ��µľ��ζ���
//			//ȡֱ�ߵĲ�������
//			//int radianEliminate = 230;//(R��)����ʹ��
//			//int radianEliminate2 = 360;//(R��)����ʹ��
//			int radianEliminate = 0;//(R��)����ʹ��
//			int radianEliminate2 = 0;//(R��)����ʹ��
//			int deviation = 200;//(б�ߴ��������)����ʹ��
//			int deviation2 = 120;//(б�ߴ��������)����ʹ��
//
//			if (displayError_Areasignal != 0)
//			{
//				for (int j = 0; j < contours[i].size(); j++)
//				{
//					//���㼯
//					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1) * 0.3 && abs(contours[i][j].x - x1) < deviation ||
//						contours[i][j].y > y1 + (y2 - y1) * 0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
//						leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//					//�Ҳ�㼯
//					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
//						rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//					//�ϲ�㼯
//					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
//						upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
//						downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				}
//				//�²�㼯
//				//downLinePoint.push_back(Point((x2 + x3) / 2, (y2 + y3) / 2));
//				if (leftLinePoint.size() != 0 && rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
//				{
//					//ֱ�����
//					fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
//					fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
//					fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
//					downLine_Fit[0] = upLine_Fit[0];
//					downLine_Fit[1] = upLine_Fit[1];
//					downLine_Fit[2] = downLinePoint[0].x;
//					downLine_Fit[3] = downLinePoint[0].y;                                           //�²�ֱ��ȷ��
//
//					/*Mat img = _src.getMat();
//					for (int i = 0; i < leftLinePoint.size(); i++)
//					{
//						circle(img, Point(leftLinePoint[i].x, leftLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//					}
//					for (int i = 0; i < rightLinePoint.size(); i++)
//					{
//						circle(img, Point(rightLinePoint[i].x, rightLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//					}
//					for (int i = 0; i < upLinePoint.size(); i++)
//					{
//						circle(img, Point(upLinePoint[i].x, upLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//					}
//					for (int i = 0; i < downLinePoint.size(); i++)
//					{
//						circle(img, Point(downLinePoint[i].x, downLinePoint[i].y), 8, Scalar(0, 0, 255), -1);
//					}*/
//					//�ǵ���ȡ
//					src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
//					src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
//					src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
//					src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�
//
//
//					//src_corner_enlarge[0] = Point2f(xcoordinate1 + tl.x - border_white, ycoordinate1 + tl.y - border_white);	                         //���Ͻ�
//					//src_corner_enlarge[1] = Point2f(xcoordinate2 + bl.x - border_white, ycoordinate2 - height / 3 + bl.y + border_white);              //���½�
//					//src_corner_enlarge[2] = Point2f(xcoordinate3 - width / 4 + br.x + border_white, ycoordinate3 - height / 3 + br.y + border_white);	 //���½�
//					//src_corner_enlarge[3] = Point2f(xcoordinate4 - width / 4 + tr.x + border_white, ycoordinate4 + tr.y - border_white);	             //���Ͻ�
//
//					src_corner_enlarge[0].y = src_corner[0].y - border_white;
//					src_corner_enlarge[0].x = src_corner[0].x - border_white;
//					src_corner_enlarge[1].y = src_corner[1].y + border_white;
//					src_corner_enlarge[1].x = src_corner[1].x - border_white;
//					src_corner_enlarge[2].y = src_corner[2].y + border_white;
//					src_corner_enlarge[2].x = src_corner[2].x + border_white;
//					src_corner_enlarge[3].y = src_corner[3].y - border_white;
//					src_corner_enlarge[3].x = src_corner[3].x + border_white;
//					Mat temp_mask = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC1);
//					drawContours(temp_mask, contours, i, 255, FILLED, 8);
//					//͸�ӱ任�������
//					if (ScreenType_Flag == "������")
//						dst_corner = { Point(0, 0), Point(0, 1183), Point(3000, 1183), Point(3000, 0) };
//					else
//						dst_corner = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
//					*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
//					*M_R_1_E = cv::getPerspectiveTransform(src_corner_enlarge, dst_corner);
//				}
//				else
//				{
//					displayError_Areasignal = 0;
//					break;
//				}
//			}
//		}
//	}
//	//û����ȡ����Ļ
//	if (displayError_Areasignal == 0)
//	{
//		Ext_Result_Left_Right = true; //û����ȡ����Ļ
//		vector<Point2f> src_points(4);
//		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
//		vector<Point2f> dst_points(4);
//		if (ScreenType_Flag == "������")
//			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
//		else
//			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
//		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //͸�ӱ任������ȡ
//		*M_R_1_E = cv::getPerspectiveTransform(src_points, dst_points);
//	}
//
//	return Ext_Result_Left_Right;
//}
//

bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat* Mwhite, Mat* M_R_1_E, String ScreenType_Flag, int leftRightWhiteFlag, int border_white)
{
	bool Ext_Result_Left_Right = true;                                                     //��ȡ��Ļ�ɹ���־λ
	Mat src = _src.getMat();                                                        //����Դͼ��
	if (src.type() == CV_8UC1)														//������8λͼ
		src = src.clone();															//����ԭͼ
	else
		cvtColor(src, src, CV_BGR2GRAY);                                            //�ҶȻ���ɫͼ

	convexSetPretreatment(src);
	CV_Assert(src.depth() == CV_8U);                                                //8λ�޷���
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //��ֵͼ��

	threshold(src, binaryImage, mean(src)[0] * 0.6, 255, THRESH_BINARY);
	//threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//��ֵ��(������)
	medianBlur(binaryImage, binaryImage, 5);										//��ֵ�˲�ȥ�����
	int displayError_Areasignal = 0;												//������������ж���ʾ�쳣��־λ
	vector<vector<Point>> contours;													//contours��ŵ㼯��Ϣ
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL����������CV_CHAIN_APPROX_NONE������������Ϣ
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//�������Ҳ�㼯����
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //�����������ֱ������
	vector<Point2f> src_corner(4);                                                  //�ĸ����ཻ�õ��ǵ�����
	vector<Point2f> src_corner_enlarge(4);
	Rect rect;																        //��С����Ӿ���
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //���Ӿ����������Ϣ
	vector<Point2f> dst_corner(4);                                                  //͸�ӱ任��ĵ����Ϣ
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 150000 && area < 6000000)
		{
			rect = boundingRect(contours[i]);
			Mat temp_mask = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC1);
			drawContours(temp_mask, contours, i, 255, FILLED, 8);
			displayError_Areasignal++;
			Mat sra_canny;
			Canny(temp_mask, sra_canny, 0, 255);
			//dilate(sra_canny, sra_canny, cv::getStructuringElement(0, cv::Size(3, 3)), Point(-1, -1));
			for (int i = 1; i < sra_canny.cols; i++) {
				for (int j = 1; j < sra_canny.rows; j++) {
					//���㼯
					for (int k = 1; k < 4; k++) {
						if (i < (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							leftLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//�Ҳ�㼯
						if (i > (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							rightLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//�ϲ�㼯
						if (j < (rect.y + rect.height / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + k, i + 7 * k) > 100 || sra_canny.at<uchar>(j - k, i + 7 * k) > 100 || sra_canny.at<uchar>(j, i + 5) > 100)) {
							upLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//�²�㼯
						if (j > (rect.y + rect.height / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + k, i + 7 * k) > 100 || sra_canny.at<uchar>(j - k, i + 7 * k) > 100 || sra_canny.at<uchar>(j, i + 5) > 100)) {
							downLinePoint.push_back(Point(i, j));
						}
					}
				}
			}

			if (leftLinePoint.size() != 0 && rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
			{
				//ֱ�����
				fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //������ֱ��
				fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�Ҳ����ֱ��
				fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//�ϲ����ֱ��
				fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//�²����ֱ��

				//�ǵ���ȡ
				src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //���Ͻǵ�
				src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //���½ǵ�
				src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //���½ǵ�
				src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //���Ͻǵ�

				src_corner_enlarge[0].y = src_corner[0].y - border_white;
				src_corner_enlarge[0].x = src_corner[0].x - border_white;
				src_corner_enlarge[1].y = src_corner[1].y + border_white;
				src_corner_enlarge[1].x = src_corner[1].x - border_white;
				src_corner_enlarge[2].y = src_corner[2].y + border_white;
				src_corner_enlarge[2].x = src_corner[2].x + border_white;
				src_corner_enlarge[3].y = src_corner[3].y - border_white;
				src_corner_enlarge[3].x = src_corner[3].x + border_white;
				//͸�ӱ任�������
				if (ScreenType_Flag == "������")
					dst_corner = { Point(0, 0), Point(0, 1183), Point(3000, 1183), Point(3000, 0) };
				else
					dst_corner = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
				*Mwhite = cv::getPerspectiveTransform(src_corner, dst_corner);
				*M_R_1_E = cv::getPerspectiveTransform(src_corner_enlarge, dst_corner);
				Ext_Result_Left_Right = false;
				break;
			}
			else
			{
				displayError_Areasignal = 0;
				break;
			}
		}
	}
	//û����ȡ����Ļ
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Left_Right = true; //û����ȡ����Ļ
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "������")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //͸�ӱ任������ȡ
		*M_R_1_E = cv::getPerspectiveTransform(src_points, dst_points);
	}
	return Ext_Result_Left_Right;
}
/// <summary>
/// ��Ļ��͹������ȡ���ڿס�ˮ�β��֣�
/// </summary>�����Ƿ��͹����Ϊ��͹�������͹���������͹λ�ö�ֵ��ͼ��
/// <param name="scr"></param>͸�ӱ任��İ׵�ͼ
/// <param name="feitu"></param>���صķ�͹ͼ
/// <param name="feitu_rect"></param>���صķ�͹����λ��
/// <param name="feitu_radio"></param>��͹��������ֵ����
/// <param name="feitu_lowerArea"></param>��͹�����½�
/// <param name="feitu_higtherArea"></param>��͹�����Ͻ�
/// <returns></returns>
bool get_feitu_area(Mat scr, Mat& feitu,Rect& feitu_rect,double feitu_radio,int feitu_lowerArea,int feitu_higherArea)
{
	//double feitu_radio=6;
	//int feitu_lowerArea=4000,feitu_higherArea=300*500;
	Mat binary,dst;
	threshold(scr, binary, mean(scr) [0]* 0.6, 255, THRESH_BINARY);//��ֵ��
	dst = binary.clone();
	convexSetPretreatment(binary);
	dst = binary - dst;
	//�����ṹԪ��
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(-1, -1));
	//ִ�п�����
	morphologyEx(dst, dst, MORPH_OPEN, kernel);

	/// Find contours
	vector<vector<Point>> contours;
	findContours(dst, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	for(int i=0;i<contours.size();i++)
	{
		double area = contourArea(contours[i]);
		if (area >= feitu_lowerArea && area < feitu_higherArea) //
		{
			Rect boundRect;
			boundRect = boundingRect(Mat(contours[i]));
			int w = boundRect.width;
			int h = boundRect.height;
			double radio = max(w / h, h / w);

			//������ų�
			if (radio > feitu_radio)
			{
				continue;
			}
			else
			{
				feitu = dst(boundRect);
				feitu_rect = boundRect;
				return true;
			}
		}
	}
	return false;
}


///
/// \brief find_camera_roi�������ROI
/// \param src����ͼ��
/// \param roi_rect���ص�ROI����
/// \param roi_radio����ĳ����Ҫ��
/// \param roi_lowerArea�������
/// \param roi_expand��ԭʼROI����������Ŀ��
/// \return�Ƿ���ȡ��ROI
///

bool find_camera_roi(Mat src, Rect& roi_rect, float roi_radio, int roi_lowerArea, int roi_expand)
{
	if (src.type() == CV_8UC1)														//������8λͼ
		src = src.clone();															//����ԭͼ
	else
		cvtColor(src, src, CV_BGR2GRAY);                                            //�ҶȻ���ɫͼ
	Mat dst;
	threshold(src, dst, mean(src)[0] * 0.8, 255, THRESH_BINARY);//��ֵ��
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(5, 5), Point(-1, -1));
	//ִ�п�����
	morphologyEx(dst, dst, MORPH_OPEN, kernel);
	/// Find contours
	vector<vector<Point>> contours;
	findContours(dst, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area >= roi_lowerArea) //
		{
			Rect boundRect;
			boundRect = boundingRect(Mat(contours[i]));
			int w = boundRect.width;
			int h = boundRect.height;
			double radio = max(w / h, h / w);

			//������ų�
			if (radio > roi_radio)
			{
				continue;
			}
			else
			{
				roi_rect = boundRect;
				boundRect.x - roi_expand < 0 ? roi_rect.x = 0 : roi_rect.x = (boundRect.x - roi_expand) / 16 * 16;
				boundRect.y - roi_expand < 0 ? roi_rect.y = 0 : roi_rect.y = (boundRect.y - roi_expand) / 16 * 16;
				boundRect.x + roi_expand + boundRect.width > src.cols - 1 ? roi_rect.width = (src.cols - 1 - boundRect.x) / 16 * 16 : roi_rect.width = (boundRect.width + 2 * roi_expand) / 16 * 16;
				boundRect.y + roi_expand + boundRect.height > src.rows - 1 ? roi_rect.height = (src.rows - 1 - boundRect.y) / 16 * 16 : roi_rect.height = (boundRect.height + 2 * roi_expand) / 16 * 16;
				return true;
			}
		}
	}
	return false;
}

void make_screen_border(Mat scr, Mat& dst, int border_size,Rect& shap_rect) {
	
	
	
	//������,�ֺ��ڲ��ն�,�������ܽ�������
	Mat dst1, dst2, dst3,th1, img_gray, img_gray2,th2;
	adaptiveThreshold(scr, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, border_size, -1);
	threshold(scr, th1, 0.9*mean(scr)[0], 255, CV_THRESH_BINARY);
	threshold(scr, th2, 0.9 * mean(scr)[0], 255, CV_THRESH_BINARY_INV);

	bitwise_and(th1, scr, img_gray);

	Mat element = getStructuringElement(MORPH_RECT, Size(border_size, border_size));//�ղ����ṹԪ��
	Mat element1 = getStructuringElement(MORPH_CROSS, Size(5, 5));//�ղ����ṹԪ��
	morphologyEx(scr, dst1, CV_MOP_CLOSE, element);   //��������̬ѧ���������Լ������
    dilate(scr, dst2, element);//����
//    //imwrite("D:pengzhang.bmp",th_result);
    erode(scr, dst3, element);//����
	bitwise_and(dst2, th2, img_gray2);
	bitwise_or(img_gray, img_gray2, dst2);

	adaptiveThreshold(dst2, dst3, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, border_size, -1);


	
	///ͨ����ֵ�жϱ߽���������
	mean(scr(Rect(0, 0,  scr.cols-1,scr.rows / 2)));
	mean(scr(Rect( 0,scr.rows / 2-1, scr.cols-1, scr.rows / 2)));
	mean(scr(Rect(0, 0, scr.cols/2, scr.rows-1)));
	mean(scr(Rect( scr.cols / 2-1,0, scr.cols/2, scr.rows-1)));
	Canny(scr.clone(), dst,  mean(scr)[0], mean(scr)[0]);
	/// ���߽紹ֱ��������/�����������е�һ�׶��׵���
	Mat grad_x, grad_y, grad_xx, grad_yy;
	int scale = 1, delta = 1;
	Sobel(scr, grad_x, -1, 1, 0, CV_SCHARR, scale, delta, BORDER_DEFAULT);
	Sobel(scr, grad_y, -1, 0, 1, CV_SCHARR, scale, delta, BORDER_DEFAULT);
	Sobel(grad_x, grad_xx, -1, 1, 0, CV_SCHARR, scale, delta, BORDER_DEFAULT);
	Sobel(grad_y, grad_yy, -1, 0, 1, CV_SCHARR, scale, delta, BORDER_DEFAULT);
	

	Mat kernel=(Mat_<char>(2, 1) << 1, -1);
	Mat kernel2 = (Mat_<char>(1, 2) << 1, -1);
	Mat kernel3 = (Mat_<int>(3, 3) << 0, 0, 0, -2, 0, 2, 0, 0, 0);
	Mat kernel4 = (Mat_<int>(3, 3) << 0, -2, 0, 0, 0, 0, 0, 2, 0);
	Mat kernel30 = (Mat_<int>(3, 3) << 0, 0, 0, 2, 0, -2, 0, 0, 0);
	Mat kernel40 = (Mat_<int>(3, 3) << 0, 2, 0, 0, 0, 0, 0, -2, 0);
	Mat dx, dy,dx1,dy1,dx2,dy2,dxy;
	/// ��ͼ������˲�����
	filter2D(scr, dx1, scr.depth(), kernel3);
	filter2D(scr, dx2, scr.depth(), kernel30);
	bitwise_or(dx1, dx2, dx);

	filter2D(scr, dy1, scr.depth(), kernel4);
	filter2D(scr, dy2, scr.depth(), kernel40);
	bitwise_or(dy1, dy2, dy);
	dst = scr.clone();

	dxy = dy.clone();
	///����һ�׶��׵��������Ļ�߽�
	for (int i = 0; i < scr.rows; i++) {
		float mean_col = mean(dxy)[0];//���ݶȾ�ֵ
		mean_col = 6;
		bool up_flag = false, down_flag = false;
		for (int j = 0; j < scr.cols/2-1; j++) {
			if(dxy.at<uchar>(i, j)<= mean_col&& dxy.at<uchar>(i, j+1) > mean_col){
				up_flag = true;
			}
			if(up_flag&& dxy.at<uchar>(i, j) > mean_col && dxy.at<uchar>(i, j + 1) <= mean_col)
			{
				down_flag = true;
				int k;
				j - border_size*2 < 0 ? k = 0 : k = j - border_size*2;//��ֹԽ��
				for (k; k < j + 1;k++) 
				{
					dst.at<uchar>(i, k) = scr.at<uchar>(i, j + 1);
				}


				dxy.at<uchar>(i, j + 1) = 255;
				break;
			}
	}

	}
	dxy = dy.clone();
	///����һ�׶��׵��������Ļ�߽�
	for (int i = 0; i < scr.rows; i++) {
		float mean_col = mean(dxy)[0];//���ݶȾ�ֵ
		mean_col = 6;
		bool up_flag = false, down_flag = false;
		for (int j = scr.cols-1; j > scr.cols / 2 - 1; j--) {
			if (dxy.at<uchar>(i, j) <= mean_col && dxy.at<uchar>(i, j - 1) > mean_col) {
				up_flag = true;
			}
			if (up_flag && dxy.at<uchar>(i, j) > mean_col && dxy.at<uchar>(i, j - 1) <= mean_col)
			{
				down_flag = true;
				int k;
				j + border_size * 2 > scr.cols - 1 ? k = scr.cols - 1 : k = j + border_size * 2;//��ֹԽ��
				for (k; k > j - 1; k--)
				{
					dst.at<uchar>(i, k) = scr.at<uchar>(i, j - 1);
				}


				dxy.at<uchar>(i, j - 1) = 255;
				break;
			}
		}

	}




	dxy = dy.clone();
	///����һ�׶��׵��������Ļ�߽�
	for (int j = 0; j < scr.cols ; j++) {
		float mean_col = mean(dxy)[0];//���ݶȾ�ֵ
		mean_col = 6;
		bool up_flag = false, down_flag = false;
		for (int i = 0; i < scr.rows/2 - 1; i++) {
			if (dxy.at<uchar>(i, j) <= mean_col && dxy.at<uchar>(i+1, j) > mean_col) {
				up_flag = true;
			}
			if (up_flag && dxy.at<uchar>(i, j) > mean_col && dxy.at<uchar>(i+1, j ) <= mean_col)
			{
				down_flag = true;
				int k;
				i - border_size*2 < 0 ? k = 0 : k = i - border_size*2;//��ֹԽ��
				for (k; k < i + 1; k++)
				{
					dst.at<uchar>(k, j) = scr.at<uchar>(i+1, j );
				}
				dxy.at<uchar>(i+1, j) = 255;
				break;
			}
		}

	}


	dxy = dy.clone();
	///����һ�׶��׵��������Ļ�߽�
	for (int j = 0; j < scr.cols; j++) {
		float mean_col = mean(dxy)[0];//���ݶȾ�ֵ
		mean_col = 6;
		bool up_flag = false, down_flag = false;
		for (int i = scr.rows-1; i > scr.rows / 2 - 1; i--) {
			if (dxy.at<uchar>(i, j) <= mean_col && dxy.at<uchar>(i - 1, j) > mean_col) {
				up_flag = true;
			}
			if (up_flag && dxy.at<uchar>(i, j) > mean_col && dxy.at<uchar>(i - 1, j) <= mean_col)
			{
				down_flag = true;
				int k;
				i + border_size*2 > scr.rows-1 ? k = scr.rows-1 : k = i + border_size*2;//��ֹԽ��
				for (k; k > i - 1; k--)
				{
					dst.at<uchar>(k, j) = scr.at<uchar>(i - 1, j);
				}
				dxy.at<uchar>(i - 1, j) = 255;
				break;
			}
		}

	}
	/// ����Ӧ��ֵ�������غ�ԭͼ��Сһ��ͼ
}




/// <summary>
/// ��1*x�ľ���תΪ�á���ʼλ�á�+�����ݡ�+��������������ʾ������
/// </summary>����1,1,1,2,2,3,��ʾΪ��0����1����3������3����2����2������5����3����1��
/// <param name="scr"></param>ԭ����
/// <returns></returns>
vector<vector<int>> Mat_to_vector(Mat scr) {
	vector<vector<int>> dst_vec;
	int last_data = scr.at<uchar>(0, 0);//�ϴε�ֵ
	int last_location = 0;//�ϴε���ֹλ��
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 1; j < scr.cols; j++) {
			if (scr.at<uchar>(i, j) != last_data && scr.at<uchar>(i, j) != -1 || j == scr.cols - 1)
			{
				vector<int>mat_data;
				mat_data.push_back(last_location);//��ʼλ��
				mat_data.push_back(last_data);//����
				mat_data.push_back(j - last_location);//��������ͬ���ݸ���
				dst_vec.push_back(mat_data);
				last_data = scr.at<uchar>(i, j);//��������
				last_location = j;//����λ��
			}
		}
	}
	return dst_vec;
}

bool is_deformation(vector<vector<int>> scr_vec, int length, int border) {
	vector<vector<int>> scr_data, scr_noise, noise;
	int avg = 35;//length / scr_vec.size();//30
	vector<int> value, start_index, value_num;
	vector<int> value2, start_index2, value_num2;

	for (int i = 0; i < scr_vec.size(); i++) {
		vector<int > data_unit;

		for (int j = 0; j < scr_vec[i].size(); j++) {
			data_unit.push_back(scr_vec[i][j]);
		}
		scr_vec[i][2] < avg ? scr_noise.push_back(data_unit) : scr_data.push_back(data_unit);
		if (scr_vec[i][2] >= avg) {
			//start_index.push_back(data_unit[0]);
			//value.push_back(data_unit[1]);
			//value_num.push_back(data_unit[2]);
		}
	}

	for (int i = 0; i < scr_data.size(); i++) {
		//��ȫ������ϵ�����Ԫ��
		int last_end_index= scr_data[i][2]+ scr_data[i][0];//������ַ=��ʼ��ַ+��������
		int start= scr_data[i][0];

			for (int j = i+1; j < scr_data.size(); j++) {
				//��ȫ������ϵ�����Ԫ��
				if (scr_data[i][1] == scr_data[j][1] && scr_data[j][0] - (last_end_index) < avg)
				{
					last_end_index = scr_data[j][2] + scr_data[j][0];//������ַ=��ʼ��ַ+��������
					//change_num++;

					i = j;
				}
				else {
					break;
				}
			}
			
			if (scr_data[i][1] != 0|| scr_data.size()==1) {
				start_index.push_back(start);
				value.push_back(scr_data[i][1]);
				value_num.push_back(last_end_index - start);
			}

			start_index2.push_back(start);
			value2.push_back(scr_data[i][1]);
			value_num2.push_back(last_end_index- start);

			
	}
	

	
	vector<int> value0(value);
	//findLIS(LIST_STRICTLY, value, value.size(), value0 ,num );
	int a = lis(value, value.size());
	//int b = lis(value_num, value_num.size());
	cout << "��������" << scr_data.size() << "��������" << scr_noise.size() << endl;
	cout << "��������" << value.size() << "��������" << a << "��ֵ��" << value.size() -a<< endl;

	std::vector<int>::iterator max = std::max_element(std::begin(value), std::end(value));
	//	vector<double>::iterator max=max_element(arr.begin(),arr.end());// ����Ҳ����������ʾ������max
	//std::cout << "Max element is " << *max << " at position " << std::distance(std::begin(value), max) << std::endl;
	//���ֵһ���Ǵ�*  ���*max����ʾ������ 
	auto min = std::min_element(std::begin(value), std::end(value));
	//std::cout << "min element is " << *min << " at position " << std::distance(std::begin(value), min) << std::endl;

	Mat data(*max, value.size(), CV_8UC1, Scalar(0));//����һ��ͼƬ�����ڷ����͹�����������ų�
	for (int i = 0; i < value.size(); i++) {
		for (int j = 0; j < value[i];j++) {
			data.at<uchar>(j, i) = 255;
		}
	}
	Mat convex_data = data.clone();
	Mat data_contrary, convex_data_contrary,error, error_contrary;
	vector<int> predict_value(value), predict_value_contrary(value);
	//convex_edg(data,convex_data,1);
	//convex_edg(~data,convex_data_contrary,1);
	/*predict_edg(convex_data, predict_value, value_num, false);
	predict_edg(convex_data_contrary, predict_value_contrary, value_num, true);*/

	predict_edg(convex_data, predict_value, false);
	predict_edg(convex_data_contrary, predict_value_contrary, true);


	error = abs(convex_data - data);
	error_contrary = abs(convex_data_contrary - ~data);
	bool result_error = false, result_contrary = false,result_down=false;
	int num = 2;//ƫ���Ĵ�С

	///ԭʼ�����б�
	int change_lenght=0;
	//for (int i = 0; i < error.cols; i++) {
	//	for (int j = error.rows-1; j > 0; j--) {
	//		if (error.at<uchar>(j, i) >100) {
	//			if (abs(j - value[i])+1 >= num) 
	//			{
	//				result_error = true; }
	//			change_lenght = change_lenght + value_num[i];//��ı�����ݿ�ȣ���Ӧ��Ļ�����ؿ�ȣ�
	//			break;
	//		}			
	//	}
	//}
	for (int i = 0; i < error.cols; i++) {	
		if (abs(predict_value[i] - value[i]) >= num) {
			change_lenght = change_lenght + value_num[i];//��ı�����ݿ�ȣ���Ӧ��Ļ�����ؿ�ȣ�
			result_error = true;
			}
	}
	if (change_lenght > length / 2) {
		//result_error = false;
	}
	///ȡ��ͼ�б�
	change_lenght = 0;
	//for (int i = 0; i < error.cols; i++) {
	//	for (int j = 0; j < error_contrary.rows; j++) {
	//		if (error_contrary.at<uchar>(j, i) > 100) {
	//			if (abs(j - value[i]) >= num)
	//			{
	//				result_contrary = true;
	//			}
	//			change_lenght = change_lenght + value_num[i];//��ı�����ݿ�ȣ���Ӧ��Ļ�����ؿ�ȣ�
	//			break;
	//		}
	//	}
	//}
	for (int i = 0; i < error.cols; i++) {
		if (abs(predict_value_contrary[i] -*max+ value[i]) >= num) {
			change_lenght = change_lenght + value_num[i];//��ı�����ݿ�ȣ���Ӧ��Ļ�����ؿ�ȣ�
			result_contrary = true;
		}
	}

	if (change_lenght > length / 2) {
		//result_contrary = false;
	}
	///���򳬹�һ������ֱ����Ϊ����
	//if(value.size() > 3 && a!= value.size())//a<= value.size()/2
	//{
	//	vector<int> value_contrary(value);
	//	int down_num = 0;
	//	int max_postion = distance(std::begin(value), max);
	//	for (int i = max_postion; i < value.size(); i++)
	//	{
	//		value_contrary[i] = *max- value[i];
	//		down_num = down_num + value_num[i];
	//	}
	//	int c = lis(value_contrary, value_contrary.size());
	//	down_num-value_num[max_postion]> value_num[max_postion]&&c >=2 ? result_down = true : result_down = false;

	//}
	
	if (value.size() > 3) {
		Mat  predict_num_data;
		Mat num_data(*max_element(value_num.begin(), value_num.end()), value_num.size(), CV_8UC1, Scalar(0));//����һ��ͼƬ�����ڷ����͹�����������ų�	
		for (int i = 0; i < value.size(); i++) {
			for (int j = 0; j < value_num[i]; j++) {
				num_data.at<uchar>(j, i) = 255;
			}
		}
		vector<int> value_predict_num(value_num);
		predict_num(predict_num_data, value_predict_num);
		Mat error_num = abs(predict_num_data - num_data);
		for (int i = 0; i < error_num.cols; i++) {
			if (abs(value_predict_num[i] - value_num[i]) >= cv::max(value_num[i], value_predict_num[i]) >> 1) {
				//change_lenght = change_lenght + value_num[i];//��ı�����ݿ�ȣ���Ӧ��Ļ�����ؿ�ȣ�
				result_down = true;
			}
		}
	}


	if (value.size() > 3 && a < value.size()/3.0*2.0)//a<= value.size()/2
	{
		vector<int> value_contrary(value);
		int down_num = 0;
		int max_postion = distance(std::begin(value), max);
		for (int i = max_postion; i < value.size(); i++)
		{
			value_contrary[i] = *max - value[i];
			down_num = down_num + value_num[i];
		}
		int c = lis(value_contrary, value_contrary.size());
		//c >= 2 ? result_down = true : result_down = false;

	}
	cout << "ԭͼ�����" << result_error << "��ͼ�����" << result_contrary << "�����Խ����" << result_down << endl;
	cout << "�ܽ��" <<( result_error || result_contrary|| result_down )<< endl;
	return result_contrary|| result_error|| result_down;
}


void convex_edg(Mat src, Mat& dst, int min_area)
{

	{
		//�ն�Ԥ���� ����ڱ߽��㷨
		Mat src_copy = src.clone();
		Mat threshold_output;
		vector<vector<Point> > preContours;
		vector<Vec4i> hierarchy;

		/// Detect edges using Threshold

		double meanGray = mean(src_copy)[0];
		threshold(src_copy, threshold_output, meanGray * 0.6, 255, THRESH_BINARY);

		/// Find contours
		findContours(threshold_output, preContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Find the convex hull object for each contour
		vector<vector<Point> >hull(preContours.size());

		for (size_t i = 0; i < preContours.size(); i++)
		{
			convexHull(Mat(preContours[i]), hull[i], false);
		}

		/// Draw contours + hull results
		Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC1);
		for (size_t i = 0; i < preContours.size(); i++)
		{
			double area = contourArea(preContours[i]);
			//if (area > min_area)
			{
				drawContours(drawing, preContours, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
				drawContours(drawing, hull, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
			}
		}
		dst = drawing;
	}

}


void predict_edg( Mat& dst, vector<int> & value, vector<int>value_num,bool is_reverse)
{
	int max = *std::max_element(std::begin(value), std::end(value));

	///Ϊ�˸��ã�ȡ��ͼ�ȷ�ת
	if (is_reverse)
	{
		///�Ȱ�����ȡ�����ٷ�ת
		int temp_max =* std::max_element(std::begin(value), std::end(value));
		for (int i = 0; i < value.size(); i++)
		{
			value[i] = temp_max - value[i];
		}
		reverse(value.begin(), value.end());
		reverse(value_num.begin(), value_num.end());
	}
	std::vector<int>::iterator num_max = std::max_element(std::begin(value_num), std::end(value_num));

	int num_max_index = distance(begin(value_num), num_max);//��������������λ��
	std::vector<int>::iterator local_max = std::max_element(std::begin(value), std::end(value));
	int local_max_index= distance(begin(value), local_max);//��������������λ��
	
	if (value.size() > 1)
	{
		if (*max_element(begin(value), begin(value) + num_max_index) >= value[num_max_index])
		{
			for (int i = num_max_index; i >= 0; i--) {
				//��ԭ����߱��ұ�ֵ��Ĳ���ɾ��
				if (value[num_max_index] < value[i] && i >= local_max_index) {
					for (int j = num_max_index; j >= local_max_index; j--) {
						value[j] = value[num_max_index];
					}
					break;
				}
			}
		}

		for (int i = num_max_index - 1; i >= 0; i--) {
			//��֤��಻��
			if (value[i] > value[i + 1]) {
				value[i] = value[i + 1];
			}
		}

		for (int i = num_max_index; i < value.size() - 1; i++) {
			//��֤�Ҳ಻��
			if (value[i] > value[i + 1]) {
				value[i + 1] = value[i];
			}
		}

	}

	//////Ϊ�˸��ã�ȡ��ͼ�ȷ�ת������ͼʱ�ٷ�ת��ȥ
	Mat data(max, value.size(), CV_8UC1, Scalar(0));//����һ��ͼƬ�����ڷ����͹�����������ų�

	if (is_reverse)
	{
		reverse(value.begin(), value.end());
		reverse(value_num.begin(), value_num.end());
		for (int i = 0; i < value.size(); i++) {
			for (int j = 0; j < value[i]; j++) {
				data.at<uchar>(data.rows-1-j, i) = 255;
			}
		}
	}
	else {
		for (int i = 0; i < value.size(); i++) {
			for (int j = 0; j < value[i]; j++) {
				data.at<uchar>(j, i) = 255;
			}
		}
	}
	

	dst = data.clone();


}

void predict_edg(Mat& dst, vector<int>& value,  bool is_reverse)
{
	int max = *std::max_element(std::begin(value), std::end(value));

	///Ϊ�˸��ã�ȡ��ͼ�ȷ�ת
	if (is_reverse)
	{
		///�Ȱ�����ȡ�����ٷ�ת
		int temp_max = *std::max_element(std::begin(value), std::end(value));
		for (int i = 0; i < value.size(); i++)
		{
			value[i] = temp_max - value[i];
		}
	}
	sort(value.begin(), value.end());//��С������

	//////Ϊ�˸��ã�ȡ��ͼ�ȷ�ת������ͼʱ�ٷ�ת��ȥ
	Mat data(max, value.size(), CV_8UC1, Scalar(0));//����һ��ͼƬ�����ڷ����͹�����������ų�

	if (is_reverse)
	{
		reverse(value.begin(), value.end());
		for (int i = 0; i < value.size(); i++) {
			for (int j = 0; j < value[i]; j++) {
				data.at<uchar>(data.rows - 1 - j, i) = 255;
			}
		}
	}
	else {
		for (int i = 0; i < value.size(); i++) {
			for (int j = 0; j < value[i]; j++) {
				data.at<uchar>(j, i) = 255;
			}
		}
	}


	dst = data.clone();


}

/// <summary>
/// /������������
/// </summary>


int lis(vector<int> arr, int len)
{
	vector<int> longest(arr);
	for (int i = 0; i < len; i++)
		longest[i] = 1;

	for (int j = 1; j < len; j++) {
		for (int i = 0; i < j; i++) {
			if (arr[j] > arr[i] && longest[j] < longest[i] + 1) { //ע��longest[j]<longest[i]+1�������������ʡ�ԡ�  
				longest[j] = longest[i] + 1; //������arr[j]��β�����е�����������г���  
			}
		}
	}

	int max = 0;
	for (int j = 0; j < len; j++) {
		//cout << "longest[" << j << "]=" << longest[j] << endl;
		if (longest[j] > max) max = longest[j];  //��longest[j]���ҳ����ֵ  
	}
	return max;
}

bool is_deformation_A(Mat scr,Mat big_scr,Mat mask, Rect rect,int boder, int size, int radio) {
	
		double minVal = 0, maxVal = 0;
	cv::Point minPt, maxPt;
	
	minMaxLoc(scr, &minVal, &maxVal, &minPt, &maxPt);
	Mat data = big_scr(Rect(rect.width  - maxVal, boder, maxVal, rect.height- boder - 1));
	Mat stander_data(data.rows, data.cols, CV_8UC1, Scalar(0));
	line(stander_data, Point(minVal,0), Point(maxVal, data.rows-1), Scalar( 255), 1);
	return true;
}

vector<Point2i> get_acqureline(Mat scr,Rect rect,int max,int min) {
	int midblur_size = 9;
	int border = 120;
	vector<Point2i>distance_vec; vector<Point2i>distance_vec_e, distance_vec_s, distance_vec_n;
	Rect we_rect(border, 0, scr.rows - 2 * border, 1);
	Mat dst_west(1, scr.rows, CV_16UC1, Scalar(0));
	int max_distance = 0;
	for (int i = scr.rows - border - 1; i >= border; i--) {
		int distance=0;
		for (int j = 0; j < scr.cols / 2 - 1; j++) {
			if (scr.at<uchar>(i, j) > max)
			{
				//dst_west.at<uchar>(0, i) = j;
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_west.at<uchar>(0, i) = j;
				distance = distance + 150;
			}
			if (scr.at<uchar>(i, j) > min&& scr.at<uchar>(i, j) < max)
			{
				//dst_west.at<uchar>(0, i) = j;
				distance = distance + 150 - scr.at<uchar>(i, j);
			}
		}
		distance_vec.push_back(Point2i(i,distance));
		dst_west.at<ushort>(0, i) = distance;
		max_distance < distance ? max_distance = distance : distance=0;
	}
	medianBlur(dst_west, dst_west, midblur_size);

	Mat west(max_distance+1,scr.rows, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec.size()+ border; i++) {
		//circle(west, distance_vec[i], 1, Scalar(255));
		west.at<uchar>(dst_west.at<ushort>(0, i), i) = 255;

	}
	Mat dst_east(1, scr.rows, CV_16UC1, Scalar(0));
	for (int i = border - 1; i < scr.rows - border; i++) {
		int distance = 0;
		for (int j = scr.cols - 1; j > scr.cols / 2 - 1; j--) 
		{
		
			if (scr.at<uchar>(i, j) > max)
			{
				//dst_east.at<uchar>(0, i) = j;
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_east.at<uchar>(0, i) = j;
				distance = distance + 170;
			}
			if (scr.at<uchar>(i, j) > min && scr.at<uchar>(i, j) < max)
			{
				//dst_east.at<uchar>(0, i) = j;
				distance = distance + 170 - scr.at<uchar>(i, j);
			}
		}
		distance_vec_e.push_back(Point2i(i, distance));
		dst_east.at<ushort>(0, i) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;

	}
	medianBlur(dst_east, dst_east, midblur_size);

	Mat east(max_distance+1, scr.rows, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_e.size()+ border; i++) {
		//circle(east, distance_vec_e[i], 1, Scalar(255));
		east.at<uchar>(dst_east.at<ushort>(0, i), i) = 255;

	}

	Mat dst_south(1, scr.cols, CV_16UC1, Scalar(0));
	for (int j = scr.cols - 1 - border; j >= border; j--) {
		int distance = 0, i;
		for (i = scr.rows - 1; i > scr.rows / 2; i--)
		{

			if (scr.at<uchar>(i, j) > max)
			{
				//dst_south.at<uchar>(0, i) = j;
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_south.at<uchar>(0, i) = j;
				distance = distance + 170;
			}
			if (scr.at<uchar>(i, j) > min && scr.at<uchar>(i, j) < max)
			{
				//dst_south.at<uchar>(0, i) = j;
				distance = distance + 170 - scr.at<uchar>(i, j);
			}
		}
		distance_vec_s.push_back(Point2i(j, distance));
		dst_south.at<ushort>(0, j) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;
	}
	medianBlur(dst_south, dst_south, midblur_size);

	Mat south(max_distance+1, scr.cols, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_s.size()+ border; i++) {
		//circle(south, distance_vec_s[i], 1, Scalar(255));
		south.at<uchar>(dst_south.at<ushort>(0, i), i) = 255;
	}

	Mat dst_north(1, scr.cols, CV_16UC1, Scalar(0));
	for (int j = scr.cols - 1 - border; j >= border; j--) {
		int distance = 0, i;
		for ( i = 0; i < scr.rows / 2; i++)
		{

			if (scr.at<uchar>(i, j) > max)
			{
				//dst_north.at<uchar>(0, i) = j;
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_north.at<uchar>(0, i) = j;
				distance = distance + 170;
			}
			if (scr.at<uchar>(i, j) > min && scr.at<uchar>(i, j) < max)
			{
				//dst_north.at<uchar>(0, i) = j;
				distance = distance + 170 - scr.at<uchar>(i, j);
			}
		}
		distance_vec_n.push_back(Point2i(j, distance));
		dst_north.at<ushort>(0, j) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;

	}
	medianBlur(dst_north, dst_north, midblur_size);
	Mat north(max_distance+1, scr.cols, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_n.size()+ border; i++) {
		//circle(north, distance_vec_n[i], 1, Scalar(255));
		north.at<uchar>(dst_north.at<ushort>(0, i), i) = 255;
	}
	
	Vec4f line_para;
	fitLine(distance_vec_n, line_para, DIST_L1, 0, 0.01, 0.01);
	//��ȡ��бʽ�ĵ��б��
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];

	double k = line_para[1] / line_para[0];

	//����ֱ�ߵĶ˵�(y = k(x - x0) + y0)
	cv::Point start_point, end_point;
	start_point.x = 0;
	start_point.y = k * (0 - point0.x) + point0.y;
	end_point.x = north.cols;
	end_point.y = k * (north.cols - point0.x) + point0.y;

	cv::line(north, start_point, end_point, cv::Scalar(255), 2, 8, 0);
	cv::line(north, Point(border,0), Point(north.cols- border, north.rows), cv::Scalar(255), 2, 8, 0);
	cv::line(south, Point(border, south.rows), Point(south.cols - border, 0), cv::Scalar(255), 2, 8, 0);
	cv::line(east, Point(border, 0), Point(east.cols - border, east.rows), cv::Scalar(255), 2, 8, 0);
	cv::line(west, Point(border, west.rows), Point(west.cols - border, 0), cv::Scalar(255), 2, 8, 0);

	return distance_vec;
}
vector<Point2i> get_acqureline(Mat scr, Rect rect, int max, int min,int border,bool is_clockwise_direction) {
	int midblur_size = 9; Mat binary;
	//int border = 120;
	vector<Point2i>distance_vec; vector<Point2i>distance_vec_e, distance_vec_s, distance_vec_n;
	Rect we_rect(border, 0, scr.rows - 2 * border, 1);
	Mat dst_west(1, scr.rows, CV_16UC1, Scalar(0));
	int max_distance = 0;
	for (int i = scr.rows - border - 1; i >= border; i--) {
		int distance = 0;
		int num = 0;//����߽�������
		for (int j = 0; j < scr.cols / 2 - 1; j++) {
			if (scr.at<uchar>(i, j) > max)
			{
				//dst_west.at<uchar>(0, i) = j;
				//distance = distance + num * max;
				distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
				//distance = distance + num * scr.at<uchar>(i, j);
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_west.at<uchar>(0, i) = j;
				num++;
				// distance = distance + 150;
			}
			if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_west.at<uchar>(0, i) = j;
				num++;
				distance = distance - scr.at<uchar>(i, j);
			}
		}
		distance_vec.push_back(Point2i(i, distance));
		dst_west.at<ushort>(0, i) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;
	}
	medianBlur(dst_west, dst_west, midblur_size);

	Mat west(max_distance + 1, scr.rows, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec.size() + border; i++) {
		//circle(west, distance_vec[i], 1, Scalar(255));
		for (int j = 0; j < dst_west.at<ushort>(0, i); j++) {
			west.at<uchar>(j, i) = 255;
		}
		//west.at<uchar>(dst_west.at<ushort>(0, i), i) = 255;

	}
	Mat dst_east(1, scr.rows, CV_16UC1, Scalar(0));
	for (int i = border - 1; i < scr.rows - border; i++) {
		int distance = 0;
		int num = 0;//����߽�������
		for (int j = scr.cols - 1; j > scr.cols / 2 - 1; j--)
		{

			if (scr.at<uchar>(i, j) > max)
			{
				//dst_east.at<uchar>(0, i) = j;
				//distance = distance + num * max;
				distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
				//distance = distance + num * scr.at<uchar>(i, j);
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_east.at<uchar>(0, i) = j;
				num++;
				//distance = distance + 170;
			}
			if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_east.at<uchar>(0, i) = j;
				num++;
				distance = distance  - scr.at<uchar>(i, j);
			}
		}
		distance_vec_e.push_back(Point2i(i, distance));
		dst_east.at<ushort>(0, i) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;

	}
	medianBlur(dst_east, dst_east, midblur_size);

	Mat east(max_distance + 1, scr.rows, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_e.size() + border; i++) {
		//circle(east, distance_vec_e[i], 1, Scalar(255));
		for (int j = 0; j < dst_east.at<ushort>(0, i); j++) {
			east.at<uchar>(j, i) = 255;
		}
		//east.at<uchar>(dst_east.at<ushort>(0, i), i) = 255;
	}

	Mat dst_south(1, scr.cols, CV_16UC1, Scalar(0));
	for (int j = scr.cols - 1 - border; j >= border; j--) {
		int distance = 0, i;
		int num = 0;//����߽�������
		for (i = scr.rows - 1; i > scr.rows / 2; i--)
		{

			if (scr.at<uchar>(i, j) > max)
			{
				//distance = distance + num * scr.at<uchar>(i, j);
				//distance = distance + num * max;
				distance = distance /double(scr.at<uchar>(i, j))*max + int(num) * max;
				//dst_south.at<uchar>(0, i) = j;
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_south.at<uchar>(0, i) = j;
				num++;
				//distance = distance + 170;
			}
			if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_south.at<uchar>(0, i) = j;
				num++;
				distance = distance  - scr.at<uchar>(i, j);
			}
		}
		distance_vec_s.push_back(Point2i(j, distance));
		dst_south.at<ushort>(0, j) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;
	}
	medianBlur(dst_south, dst_south, midblur_size);

	Mat south(max_distance + 1, scr.cols, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_s.size() + border; i++) {
		//circle(south, distance_vec_s[i], 1, Scalar(255));
		for (int j = 0; j < dst_south.at<ushort>(0, i); j++) {
			south.at<uchar>(j, i) = 255;
		}
		//south.at<uchar>(dst_south.at<ushort>(0, i), i) = 255;
	}

	Mat dst_north(1, scr.cols, CV_16UC1, Scalar(0));
	for (int j = scr.cols - 1 - border; j >= border; j--) {
		int distance = 0, i;
		int num = 0;//����߽�������
		for (i = 0; i < scr.rows / 2; i++)
		{

			if (scr.at<uchar>(i, j) > max)
			{
				//distance = distance + num * max;
				distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
				//distance = distance + num * scr.at<uchar>(i, j);
				//dst_north.at<uchar>(0, i) = j;
				break;
			}
			if (scr.at<uchar>(i, j) < min)
			{
				//dst_north.at<uchar>(0, i) = j;
				num++;
				//distance = distance + 170;
			}
			if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_north.at<uchar>(0, i) = j;
				num++;
				distance = distance  - scr.at<uchar>(i, j);
			}
		}
		distance_vec_n.push_back(Point2i(j, distance));
		dst_north.at<ushort>(0, j) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;

	}
	medianBlur(dst_north, dst_north, midblur_size);
	Mat north(max_distance + 1, scr.cols, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_n.size() + border; i++) {
		//circle(north, distance_vec_n[i], 1, Scalar(255));
		for (int j = 0; j < dst_north.at<ushort>(0, i); j++) {
			north.at<uchar>(j, i) = 255;
		}
		//north.at<uchar>(dst_north.at<ushort>(0, i), i) = 255;
	}
	
	Mat mask_we(1, dst_west.cols, CV_8UC1, Scalar::all(0));
	mask_we(Rect(border, 0, dst_west.cols-2*border, 1)) = 255;//�趨��������
	double min_w, min_e;
	minMaxLoc(dst_west, &min_w, 0, 0, 0, mask_we);
	minMaxLoc(dst_east, &min_e, 0, 0, 0, mask_we);

	Mat mask_ns(1, dst_north.cols, CV_8UC1, Scalar::all(0));
	mask_ns(Rect(border, 0, dst_north.cols - 2 * border, 1)) = 255;//�趨��������
	double min_n, min_s;
	minMaxLoc(dst_south, &min_s, 0, 0, 0, mask_ns);
	minMaxLoc(dst_north, &min_n, 0, 0, 0, mask_ns);

	Mat south_a, south_b, s_error_a, s_error_b, s_error;
	Mat north_a, north_b, n_error_a, n_error_b, n_error;
	Mat east_a, east_b, e_error_a, e_error_b, e_error;
	Mat west_a, west_b, w_error_a, w_error_b, w_error;
	

	//get_predict_line( border, south.rows, south.cols, Point(border, south.rows), Point(south.cols - border, min_s), south_a, south_b);
	//get_predict_line(border, north.rows, north.cols, Point(border, min_n), Point(north.cols - border, north.rows), north_a, north_b);
	//get_predict_line(border, east.rows, east.cols, Point(border, min_e), Point(east.cols - border, east.rows), east_a, east_b);
	//get_predict_line(border, west.rows, west.cols, Point(border, west.rows), Point(west.cols - border, min_w), west_a, west_b);
	Mat south_pridict, north_pridict, east_pridict, west_pridict;
	vector<double> feature_s, feature_n, feature_e, feature_w;
	cout << "south:" << endl;
	feature_s=Feature_calculation(Point(border, south.rows), Point(south.cols - border, min_s), dst_south,0 ,is_clockwise_direction);
	cout << "north:" << endl;
	feature_n=Feature_calculation(Point(border, min_n), Point(north.cols - border, north.rows), dst_north,0 ,is_clockwise_direction);
	cout << "east:" << endl;
	feature_e=Feature_calculation(Point(border, min_e), Point(east.cols - border, east.rows), dst_east,0, is_clockwise_direction);
	cout << "west:" << endl;
	feature_w=Feature_calculation(Point(border, west.rows), Point(west.cols - border, min_w), dst_west,0, is_clockwise_direction);
	
	//get_predict_line(Point(border, south.rows), Point(south.cols - border, min_s), south_pridict);
	//get_predict_line(Point(border, min_n), Point(north.cols - border, north.rows), north_pridict);
	//get_predict_line(Point(border, min_e), Point(east.cols - border, east.rows), east_pridict);
	//get_predict_line(Point(border, west.rows), Point(west.cols - border, min_w), west_pridict);
	//Mat error_n, error_s, error_w, error_e;
	//error_s = south_pridict - dst_south;
	//error_n = north_pridict - dst_north;
	//error_w = west_pridict - dst_west;
	//error_e = east_pridict - dst_east;





	//s_error_a= south- south_b;
	//s_error_b = south_b - south;
	//n_error_a = north - north_b;
	//n_error_b = north_b - north;
	//e_error_a = east - east_b;
	//e_error_b = east_b - east;
	//w_error_a = west - west_b;
	//w_error_b = west_b - west;
	//s_error = (s_error_a)+(s_error_b);
	//n_error = (n_error_a)+(n_error_b);
	//w_error = (w_error_a)+(w_error_b);
	//e_error = (e_error_a)+(e_error_b);

	//cv::line(north, Point(border, min_n), Point(north.cols - border, north.rows), cv::Scalar(255), 2, 8, 0);
	//cv::line(south, Point(border, south.rows), Point(south.cols - border, min_s), cv::Scalar(255), 2, 8, 0);
	//cv::line(east, Point(border, min_e), Point(east.cols - border, east.rows), cv::Scalar(255), 2, 8, 0);
	//cv::line(west, Point(border, west.rows), Point(west.cols - border, min_w), cv::Scalar(255), 2, 8, 0);
	return distance_vec;
}


bool get_acqureline(Mat scr,  int max, int min, int border,string &result,bool is_clockwise_direction) {
	int midblur_size = 5; Mat binary;
	//adaptiveThreshold(scr, binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, -2);
	int scale = 1, delta = 1;
	Mat grad_x(scr.rows, scr.cols, CV_8SC1, Scalar(0)),grad_y(scr.rows, scr.cols, CV_8SC1, Scalar(0)), binaryImage;
	
	Sobel(scr, grad_x, CV_16SC1, 1, 0, 3);
	Sobel(scr, grad_y, CV_16SC1, 0, 1, 3);

	cv::addWeighted(abs(grad_x), 1, abs(grad_y), 1, 0, binaryImage);
	binaryImage.convertTo(binary,CV_8UC1);
	//Canny(scr, binary, min,255);
	//int border = 120;
	vector<Point2i>distance_vec; vector<Point2i>distance_vec_e, distance_vec_s, distance_vec_n;
	Rect we_rect(border, 0, scr.rows - 2 * border, 1);
	Mat dst_west(1, scr.rows, CV_16UC1, Scalar(0));
	int max_distance = 0;
	for (int i = scr.rows - border - 1; i >= border; i--) {
		int distance = 0;
		int num = 0;//����߽�������
		bool edge_flag = false;
		for (int j = 0; j < scr.cols / 2 - 1; j++) {
			if(binary.at<uchar>(i, j) > 100) edge_flag = true ;
			if (edge_flag && binary.at<uchar>(i, j) <100 && scr.at<uchar>(i, j)>0.6*max)
			{
				edge_flag = true;
				//distance = distance + int(num) * scr.at<uchar>(i, j);
				//int a = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
				distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;

				//distance = double(distance)* (max  / double(scr.at<uchar>(i, j))) * (max / double(scr.at<uchar>(i, j))) + int(num) * max;
				//distance = distance + num * scr.at<uchar>(i, j);
				break;
			}
			//if (scr.at<uchar>(i, j) > max)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	//distance = distance + num * max;
			//	distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
			//	//distance = distance + num * scr.at<uchar>(i, j);
			//	break;
			//}
			//if (scr.at<uchar>(i, j) < min)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	num++;
			//	// distance = distance + 150;
			//}
			//if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_west.at<uchar>(0, i) = j;
				num++;
				distance = distance - scr.at<uchar>(i, j);
			}
		}
		distance_vec.push_back(Point2i(i, distance));
		dst_west.at<ushort>(0, i) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;
	}
	medianBlur(dst_west, dst_west, midblur_size);

	Mat west(max_distance + 1, scr.rows, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec.size() + border; i++) {
		//circle(west, distance_vec[i], 1, Scalar(255));
		for (int j = 0; j < dst_west.at<ushort>(0, i); j++) {
			west.at<uchar>(j, i) = 255;
		}
		//west.at<uchar>(dst_west.at<ushort>(0, i), i) = 255;

	}
	Mat dst_east(1, scr.rows, CV_16UC1, Scalar(0));
	for (int i = border - 1; i < scr.rows - border; i++) {
		int distance = 0;
		int num = 0;//����߽�������
		bool edge_flag = false;
		for (int j = scr.cols - 1; j > scr.cols / 2 - 1; j--)
		{
			if (binary.at<uchar>(i, j) > 100) edge_flag = true;
			if (edge_flag && binary.at<uchar>(i, j) < 100 && scr.at<uchar>(i, j) > 0.6 * max)
			{
				edge_flag = true;
				distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
				//distance = distance + num * scr.at<uchar>(i, j);
				break;
			}
			//if (scr.at<uchar>(i, j) > max)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	//distance = distance + num * max;
			//	distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
			//	//distance = distance + num * scr.at<uchar>(i, j);
			//	break;
			//}
			//if (scr.at<uchar>(i, j) < min)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	num++;
			//	// distance = distance + 150;
			//}
			//if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_west.at<uchar>(0, i) = j;
				num++;
				distance = distance - scr.at<uchar>(i, j);
			}
		}
		distance_vec_e.push_back(Point2i(i, distance));
		dst_east.at<ushort>(0, i) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;

	}
	medianBlur(dst_east, dst_east, midblur_size);

	Mat east(max_distance + 1, scr.rows, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_e.size() + border; i++) {
		//circle(east, distance_vec_e[i], 1, Scalar(255));
		for (int j = 0; j < dst_east.at<ushort>(0, i); j++) {
			east.at<uchar>(j, i) = 255;
		}
		//east.at<uchar>(dst_east.at<ushort>(0, i), i) = 255;
	}

	Mat dst_south(1, scr.cols, CV_16UC1, Scalar(0));
	for (int j = scr.cols - 1 - border; j >= border; j--) {
		int distance = 0, i;
		int num = 0;//����߽�������
		bool edge_flag = false;
		for (i = scr.rows - 1; i > scr.rows / 2; i--)
		{
			if (binary.at<uchar>(i, j) > 100) edge_flag = true;
			if (edge_flag && binary.at<uchar>(i, j) < 100 && scr.at<uchar>(i, j) > 0.6 * max)
			{
				edge_flag = true;
				distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
				//distance = distance + num * scr.at<uchar>(i, j);
				break;
			}
			//if (scr.at<uchar>(i, j) > max)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	//distance = distance + num * max;
			//	distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
			//	//distance = distance + num * scr.at<uchar>(i, j);
			//	break;
			//}
			//if (scr.at<uchar>(i, j) < min)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	num++;
			//	// distance = distance + 150;
			//}
			//if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_west.at<uchar>(0, i) = j;
				num++;
				distance = distance - scr.at<uchar>(i, j);
			}
		}
		distance_vec_s.push_back(Point2i(j, distance));
		dst_south.at<ushort>(0, j) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;
	}
	medianBlur(dst_south, dst_south, midblur_size);

	Mat south(max_distance + 1, scr.cols, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_s.size() + border; i++) {
		//circle(south, distance_vec_s[i], 1, Scalar(255));
		for (int j = 0; j < dst_south.at<ushort>(0, i); j++) {
			south.at<uchar>(j, i) = 255;
		}
		//south.at<uchar>(dst_south.at<ushort>(0, i), i) = 255;
	}

	Mat dst_north(1, scr.cols, CV_16UC1, Scalar(0));
	for (int j = scr.cols - 1 - border; j >= border; j--) {
		int distance = 0, i;
		int num = 0;//����߽�������
		bool edge_flag = false;
		for (i = 0; i < scr.rows / 2; i++)
		{
			if (binary.at<uchar>(i, j) > 100) edge_flag = true;
			if (edge_flag && binary.at<uchar>(i, j) < 100 && scr.at<uchar>(i, j) > 0.6 * max)
			{
				edge_flag = true;
				distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
				//distance = distance + num * scr.at<uchar>(i, j);
				break;
			}
			//if (scr.at<uchar>(i, j) > max)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	//distance = distance + num * max;
			//	distance = distance / double(scr.at<uchar>(i, j)) * max + int(num) * max;
			//	//distance = distance + num * scr.at<uchar>(i, j);
			//	break;
			//}
			//if (scr.at<uchar>(i, j) < min)
			//{
			//	//dst_west.at<uchar>(0, i) = j;
			//	num++;
			//	// distance = distance + 150;
			//}
			//if (scr.at<uchar>(i, j) >= min && scr.at<uchar>(i, j) <= max)
			{
				//dst_west.at<uchar>(0, i) = j;
				num++;
				distance = distance - scr.at<uchar>(i, j);
			}
		}
		distance_vec_n.push_back(Point2i(j, distance));
		dst_north.at<ushort>(0, j) = distance;
		max_distance < distance ? max_distance = distance : distance = 0;

	}
	medianBlur(dst_north, dst_north, midblur_size);
	Mat north(max_distance + 1, scr.cols, CV_8UC1, Scalar(0));
	for (int i = border; i < distance_vec_n.size() + border; i++) {
		//circle(north, distance_vec_n[i], 1, Scalar(255));
		for (int j = 0; j < dst_north.at<ushort>(0, i); j++) {
			north.at<uchar>(j, i) = 255;
		}
		//north.at<uchar>(dst_north.at<ushort>(0, i), i) = 255;
	}

	Mat mask_we(1, dst_west.cols, CV_8UC1, Scalar::all(0));
	mask_we(Rect(border, 0, dst_west.cols - 2 * border, 1)) = 255;//�趨��������
	double min_w, min_e, max_w, max_e;
	minMaxLoc(dst_west, &min_w, &max_w, 0, 0, mask_we);
	minMaxLoc(dst_east, &min_e, &max_e, 0, 0, mask_we);

	Mat mask_ns(1, dst_north.cols, CV_8UC1, Scalar::all(0));
	mask_ns(Rect(border, 0, dst_north.cols - 2 * border, 1)) = 255;//�趨��������
	double min_n, min_s, max_n, max_s;
	minMaxLoc(dst_south, &min_s, &max_s, 0, 0, mask_ns);
	minMaxLoc(dst_north, &min_n, &max_n, 0, 0, mask_ns);

	Mat south_a, south_b, s_error_a, s_error_b, s_error;
	Mat north_a, north_b, n_error_a, n_error_b, n_error;
	Mat east_a, east_b, e_error_a, e_error_b, e_error;
	Mat west_a, west_b, w_error_a, w_error_b, w_error;
	vector<Point2i> point_w, point_s, point_n, point_e;
	//point_w= Mat_to_pointlist(dst_west, mask_we);
	//point_e = Mat_to_pointlist(dst_east, mask_we);
	//point_s = Mat_to_pointlist(dst_south, mask_ns);
	//point_n = Mat_to_pointlist(dst_north, mask_ns);
	point_w = point_cal(dst_west, mask_we);
	point_e = point_cal(dst_east, mask_we);
	point_s = point_cal(dst_south, mask_ns);
	point_n = point_cal(dst_north, mask_ns);
	cv::line(north, point_n[0], point_n[1], cv::Scalar(255), 2, 8, 0);
	cv::line(south, point_s[0], point_s[1], cv::Scalar(255), 2, 8, 0);
	cv::line(east, point_e[0], point_e[1], cv::Scalar(255), 2, 8, 0);
	cv::line(west, point_w[0], point_w[1], cv::Scalar(255), 2, 8, 0);

	Mat south_pridict, north_pridict, east_pridict, west_pridict;
	vector<double> feature_s, feature_n, feature_e, feature_w;

	//get_predict_line( border, south.rows, south.cols, Point(border, south.rows), Point(south.cols - border, min_s), south_a, south_b);
	//get_predict_line(border, north.rows, north.cols, Point(border, min_n), Point(north.cols - border, north.rows), north_a, north_b);
	//get_predict_line(border, east.rows, east.cols, Point(border, min_e), Point(east.cols - border, east.rows), east_a, east_b);
	//get_predict_line(border, west.rows, west.cols, Point(border, west.rows), Point(west.cols - border, min_w), west_a, west_b);

	//cout << "south:" << endl;
	//feature_s = Feature_calculation(Point(border, max_s), Point(south.cols - border, min_s), dst_south, 0, is_clockwise_direction);
	//cout << "north:" << endl;
	//feature_n = Feature_calculation(Point(border, min_n), Point(north.cols - border, max_n), dst_north, 0,is_clockwise_direction);
	//cout << "east:" << endl;
	//feature_e = Feature_calculation(Point(border, min_e), Point(east.cols - border, max_e), dst_east, 0, is_clockwise_direction);
	//cout << "west:" << endl;
	//feature_w = Feature_calculation(Point(border, max_w), Point(west.cols - border, min_w), dst_west, 0, is_clockwise_direction);
	
	//cout << "south:" << endl;
	//feature_s = Feature_calculation(point_s[0], point_s[1], dst_south, 0, is_clockwise_direction);
	//cout << "north:" << endl;
	//feature_n = Feature_calculation(point_n[0], point_n[1], dst_north, 0, is_clockwise_direction);
	//cout << "east:" << endl;
	//feature_e = Feature_calculation(point_e[0], point_e[1], dst_east, 0, is_clockwise_direction);
	//cout << "west:" << endl;
	//feature_w = Feature_calculation(point_w[0], point_w[1], dst_west, 0, is_clockwise_direction);


	feature_w = Feature_calculation(dst_west, mask_we,0, is_clockwise_direction);
	feature_e = Feature_calculation(dst_east, mask_we, 0, is_clockwise_direction);
	feature_s = Feature_calculation(dst_south, mask_ns, 0, is_clockwise_direction);
	feature_n = Feature_calculation(dst_north, mask_ns, 0, is_clockwise_direction);





	//feature_s[3] > 350 ? result = result + "��߿����" : result = result + "";///�����������б�����
	//feature_n[3] > 350 ? result = result + "�ұ߿����" : result = result + "";
	//feature_e[3] > 350 ? result = result + "�±߿����" : result = result + "";
	//feature_w[3] > 180 ? result = result + "�ϱ߿����" : result = result + "";


	//get_predict_line(Point(border, south.rows), Point(south.cols - border, min_s), south_pridict);
	//get_predict_line(Point(border, min_n), Point(north.cols - border, north.rows), north_pridict);
	//get_predict_line(Point(border, min_e), Point(east.cols - border, east.rows), east_pridict);
	//get_predict_line(Point(border, west.rows), Point(west.cols - border, min_w), west_pridict);
	//Mat error_n, error_s, error_w, error_e;
	//error_s = south_pridict - dst_south;
	//error_n = north_pridict - dst_north;
	//error_w = west_pridict - dst_west;
	//error_e = east_pridict - dst_east;

	//s_error_a= south- south_b;
	//s_error_b = south_b - south;
	//n_error_a = north - north_b;
	//n_error_b = north_b - north;
	//e_error_a = east - east_b;
	//e_error_b = east_b - east;
	//w_error_a = west - west_b;
	//w_error_b = west_b - west;
	//s_error = (s_error_a)+(s_error_b);
	//n_error = (n_error_a)+(n_error_b);
	//w_error = (w_error_a)+(w_error_b);
	//e_error = (e_error_a)+(e_error_b);

	///------------
	//cv::line(north, Point(border, min_n), Point(north.cols - border, max_n), cv::Scalar(255), 2, 8, 0);
	//cv::line(south, Point(border, max_s), Point(south.cols - border, min_s), cv::Scalar(255), 2, 8, 0);
	//cv::line(east, Point(border, min_e), Point(east.cols - border, max_e), cv::Scalar(255), 2, 8, 0);
	//cv::line(west, Point(border, max_w), Point(west.cols - border, min_w), cv::Scalar(255), 2, 8, 0);
	if (result != "") {
		return true;
	}
	else {
			return false;
	}
	
}


void predict_num(Mat& dst, vector<int>& value)
{

	///�ж���������ֹԽ��
	if (value.size() > 3)
	{
		int max = *std::max_element(std::begin(value), std::end(value));

		vector<int> value_fliter(value);

		for (int i = 0; i < value.size() - 2; i++)
		{
			value[i + 1] = (value_fliter[i] + value_fliter[i + 1] + value_fliter[i + 2]) / 3;
		}


		sort(value.begin() + 1, value.end() - 1);//��С������
		reverse(value.begin()+1, value.end()-1);//��ɴӴ�С
	//////Ϊ�˸��ã�ȡ��ͼ�ȷ�ת������ͼʱ�ٷ�ת��ȥ
		Mat data(max, value.size() , CV_8UC1, Scalar(0));//����һ��ͼƬ�����ڷ����͹�����������ų�
		{
			for (int i = 0; i < value.size(); i++) {
				for (int j = 0; j < value[i]; j++) {
					data.at<uchar>(j, i) = 255;
				}
			}
		}
		dst = data.clone();
	}


}


void DeleteElem(vector <int> &vec, int elem) {
	vector<int>::iterator it = vec.begin();
	for (; it != vec.end();)
	{
		if (*it == elem)
			//ɾ��ָ��Ԫ�أ�����ָ��ɾ��Ԫ�ص���һ��Ԫ�ص�λ�õĵ�����
			it = vec.erase(it);
		else
			//������ָ����һ��Ԫ��λ��
			++it;
	}
}


bool is_deformation(Mat scr, int length, int border) {
	Mat out;
	int histSize[1] = { 256 };  //�Ҷ�ֵSize��256��
	float hrange[2] = { 0,255 }; //�Ҷȷ�Χ[0-255]
	const float* ranges[1] = { hrange }; //�����Ҷȷ�Χ[0-255]
	int channels = 0;
	calcHist(&scr, 1, &channels, Mat(), out, 1, histSize, ranges, true, false);
	double maxVal = 0;
	minMaxLoc(scr, NULL, &maxVal,NULL,NULL);
	Mat data_scr(maxVal + 1, scr.cols, CV_8UC1, Scalar(0));//����һ��ͼƬ����������Ԥ��ı߽�
	for (int i = 0; i < scr.cols; i++) {
		for (int j = 0; j < scr.at<uchar>(0,i); j++) {
			data_scr.at<uchar>(j, i) = 255;
		}
	}


	Mat data(maxVal+1, scr.cols, CV_8UC1, Scalar(0));//����һ��ͼƬ����������Ԥ��ı߽�
	{	
		int start_index = 0;//��ʼ������
		for (int i = 0; i < histSize[0]; i++) {
			out.type();
			int num = out.at<float>(i, 0);
			for (int j = 0; j < i; j++) {
				for (int k = start_index; k < num+ start_index; k++) {
					data.at<uchar>(j, k) = 255;
				}
				
			}
			start_index = start_index + out.at<float>(i, 0);
		}
	}
	Mat error = data_scr - data;
	Mat error_contrary = data - data_scr;

	return false;
}
bool is_deformation(Mat scr_a, Mat scr_b, int border) {
	Mat sum = scr_a + scr_b; 
	double maxVal = 0;
	minMaxLoc(sum, NULL, &maxVal, NULL, NULL);
	int max = maxVal;
	Mat data(maxVal + 1, sum.cols, CV_8UC1, Scalar(0));//����һ��ͼƬ����������Ԥ��ı߽�
	for (int j = 0; j < sum.cols; j++) {
		for (int k = 0; k < sum.at<uchar>(0, j); k++) {
			data.at<uchar>(k, j) = 255;
		}
	}
	flip(scr_a, scr_a, 1);
	Mat error_ab = scr_a - scr_b;
	Mat error_ba = scr_b - scr_a;
	Mat data_errorab(maxVal + 1, sum.cols, CV_8UC1, Scalar(0));//����һ��ͼƬ����������Ԥ��ı߽�
	for (int j = 0; j < sum.cols; j++) {
		for (int k = 0; k < error_ab.at<uchar>(0, j); k++) {
			data_errorab.at<uchar>(k, j) = 255;
		}
	}
	Mat data_errorba(maxVal + 1, sum.cols, CV_8UC1, Scalar(0));//����һ��ͼƬ����������Ԥ��ı߽�
	for (int j = 0; j < sum.cols; j++) {
		for (int k = 0; k < error_ba.at<uchar>(0, j); k++) {
			data_errorba.at<uchar>(k, j) = 255;
		}
	}
	return false;
}
void get_predict_line(int border,int rows,int cols, Point start, Point end,Mat& dst_a, Mat& dst_b) {
	Mat dst(rows, cols, CV_8UC1, Scalar(0));
	line(dst, start, end, Scalar(255), 1, 8, 0);
	line(dst, Point(start.x, rows -1), Point(end.x, rows -1), Scalar(255), 1, 8, 0);
	start.y<end.y? line(dst, start, Point(start.x, rows - 1), Scalar(255), 1, 8, 0): line(dst, end, Point(end.x, rows - 1), Scalar(255), 1, 8, 0);
	convexSetPretreatment(dst);
	dst_a = dst.clone();
	dst_b = ~dst.clone();
	start.x>end.x? dst(Rect(end.x, 0, start.x-end.x,rows)) = 255: dst(Rect(start.x, 0, end.x - start.x, rows)) = 255;//�趨��������
	bitwise_and(dst_b, dst, dst_b);

}
vector<Point2i> get_predict_line( Point start, Point end,Mat &predictline, bool is_clockwise_direction)
{
		vector<Point2i>predict_line;
		//if (start.x>end.x)
		//{
		//	Point swap= end;//���ڽ���start��end���н����
		//	end = start;
		//	start = swap;
		//}
		//if (!is_clockwise_direction) {
		//	Point swap = end;//���ڽ���start��end���н����
		//	end = start;
		//	start = swap;
		//}
		Mat dst(1, start.x+end.x+1, CV_16UC1, Scalar(0));
		double dy_dx = ((end.y-start.y)*1.0/ (end.x - start.x));
		for (int i = start.x; i <= end.x;i++) {
			int predict_y = start.y + (i - start.x) * dy_dx;
			predict_line.push_back(Point(i, predict_y));
			dst.at<ushort>(0, i) = predict_y;
		}
		predictline = dst.clone();
		return predict_line;
}
vector<double> Feature_calculation(Point start, Point end,Mat scr,double feature, bool is_clockwise_direction) {
	vector<double> feature_result;
	//if (start.x > end.x)
	//{
	//	Point swap = end;//���ڽ���start��end���н����
	//	end = start;
	//	start = swap;
	//}
	//if (!is_clockwise_direction) {
	//	int swap = end.y;//���ڽ���start��end���н����
	//	end.y = start.y;
	//	start.y = swap;
	//}
	Mat predictline;//����ֱ��
	get_predict_line(start,end, predictline,  is_clockwise_direction);
	scr.convertTo(scr, CV_16SC1);
	predictline.convertTo(predictline, CV_16SC1);
	Mat error = scr - predictline;

	Mat mask_we(1, start.x + end.x, CV_8UC1, Scalar::all(0));
	mask_we(Rect(start.x, 0, end.x - start.x, 1)) = 255;//�趨��������
	double min,max;
	Point  max_index, min_index;
	minMaxLoc(error, &min,  &max, &min_index, &max_index, mask_we);
	double sin_predict = abs(end.x - start.x) / sqrt((end.x - start.x)* (end.x - start.x) + (end.y - start.y)* (end.y - start.y));

	feature_result.push_back(max-min);//��ѡ����1
	feature_result.push_back((max - min)/ abs(end.y - start.y));//��ѡ����2
	feature_result.push_back((max - min) / abs(end.y - start.y)*sin_predict);//��ѡ����3
	feature_result.push_back((max - min)  * sin_predict);//��ѡ����4

	cout << " ��ֵ�� " << feature_result[0] << " ��һ��ֵ�� " << feature_result[1] << " ��һ�������ֵ�� " << feature_result[2] << " �������ֵ�� " << feature_result[3] << endl;

	return feature_result;
}

/// <summary>
/// ��������
/// </summary>
/// <param name="scr"></param>�������˲���ľ�ȷ�������(1*x��С)
/// <param name="mask"></param>�˱߶�ӦҪ�����Ĥ�����Ϊ255
/// <param name="feature"></param>��������ֵ
/// <param name="is_clockwise_direction"></param>�Ƿ�˳ʱ�뵥������
/// <returns></returns>������������
vector<double> Feature_calculation( Mat scr,Mat mask, double feature, bool is_clockwise_direction) {
	
	vector<double> feature_result; 
	vector<Point2i> point_list;
	//point_list = Mat_to_pointlist(scr, mask); //point_cal
	point_list = point_cal(scr, mask);

	Point start= point_list[0], end= point_list[1];

	//if (start.x > end.x)
	//{
	//	Point swap = end;//���ڽ���start��end���н����
	//	end = start;
	//	start = swap;
	//}
	//if (!is_clockwise_direction) {
	//	int swap = end.y;//���ڽ���start��end���н����
	//	end.y = start.y;
	//	start.y = swap;
	//}
	Mat predictline;//����ֱ��
	get_predict_line(start, end, predictline, is_clockwise_direction);
	scr.convertTo(scr, CV_16SC1);
	predictline.convertTo(predictline, CV_16SC1);
	Mat error = scr - predictline;

	Mat mask_we(1, start.x + end.x, CV_8UC1, Scalar::all(0));
	mask_we(Rect(start.x, 0, end.x - start.x, 1)) = 255;//�趨��������
	double min, max;
	Point  max_index, min_index;
	minMaxLoc(error, &min, &max, &min_index, &max_index, mask);
	double sin_predict = abs(end.x - start.x) / sqrt((end.x - start.x) * (end.x - start.x) + (end.y - start.y) * (end.y - start.y));

	cv::Scalar     mean;
	cv::Scalar     dev;
	cv::meanStdDev(abs(error), mean, dev,mask);
	double       m = mean.val[0];
	double       std = dev.val[0];

	double std_0 = 0;
	for (int i = 0; i < error.cols; i++)
	{
		std_0 = std_0 + error.at<short>(0, i) / 1.0 / error.cols * error.at<short>(0, i);
	}

	

	feature_result.push_back(max - min);//��ѡ����1
	feature_result.push_back((max - min) / abs(end.y - start.y));//��ѡ����2
	feature_result.push_back((max - min) / abs(end.y - start.y) * sin_predict);//��ѡ����3
	feature_result.push_back((max - min) * sin_predict);//��ѡ����4
	feature_result.push_back(std);//��ѡ����5
	feature_result.push_back(m);//��ѡ����6
	feature_result.push_back(std_0);//��ѡ����6

	cout << " ��ֵ��" << feature_result[0] << " ��һ��ֵ��" << feature_result[1] << " ��һ�������ֵ��" << feature_result[2] << " �������ֵ��" << feature_result[3]<< " �ܷ��" << feature_result[4]<<" ��ֵ��" << feature_result[5]<<" ��ֵ��" << feature_result[6] << endl;

	return feature_result;
}

vector<Point2i> Mat_to_pointlist(Mat scr,Mat mask) {
	vector<Point2i>dst,output_point;
	int last_x = scr.cols / 4 * 3;
	for (int i = 0; i < mask.cols; i++) {
		if (mask.at<uchar>(0, i) > 100) {
			dst.push_back(Point(i, scr.at<ushort>(0, i)));
			last_x = i;
		}
	}

	Vec4f line_para;
	fitLine(dst, line_para, DIST_L1, 0, 0.01, 0.01);
	//��ȡ��бʽ�ĵ��б��
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];

	double k = line_para[1] / line_para[0];

	//����ֱ�ߵĶ˵�(y = k(x - x0) + y0)
	cv::Point start_point, end_point;
	start_point.x = scr.cols- last_x;
	start_point.y = k * (start_point.x - point0.x) + point0.y;
	end_point.x = last_x;
	end_point.y = k * (last_x - point0.x) + point0.y;
	output_point.push_back(start_point);
	output_point.push_back(end_point);

	return output_point;
}

/// <summary>
/// ����������ֱ����ʼ�����
/// </summary>
/// <param name="scr"></param>ʵ�ʱ߾�ȷ����
/// <param name="mask"></param>��Ĥ��ȷ���ǿ�Ҫ����ǿ鲻�ü��
/// <returns></returns>����һ����ķ��ͣ���һλ����ʼ�㣬�ڶ�λ����ֹ��
vector<Point2i> point_cal(Mat scr, Mat mask) {
	vector<Point2i> output_point;
	Point start_point, end_point;

	int start_y=0 ,end_y=0;
	int start_x = 0, end_x = scr.cols / 4 * 3;
	int size = 4;
	bool start_flag = false,end_flag = false;
	for (int i = 0; i < mask.cols; i++) {
		if (mask.at<uchar>(0, i) > 100) {
			if (!start_flag) {
				start_x = i;
				start_flag = true;
			}
			if (i < start_x+ size) {
				start_y = scr.at<ushort>(0, i) / size + start_y;
			}
			else {
				break;
			}

		}
	}


	for (int i = mask.cols-1; i > start_x; i--) {
		if (mask.at<uchar>(0, i) > 100) {
			if (!end_flag) {
				end_x = i;
				end_flag = true;
			}
			if (end_x < i + size) {
				end_y = scr.at<ushort>(0, i) / size + end_y;
			}
			else {
				break;
			}
		}
	}

	start_point.x = start_x;
	start_point.y = start_y;
	end_point.x = end_x;
	end_point.y = end_y;

	output_point.push_back(start_point);
	output_point.push_back(end_point);

	return output_point;
}

/// <summary>
/// �����߱����жϺ���
/// </summary>
/// <param name="scr"></param>ԭʼ��ȷ����
/// <param name="mask"></param>��Ĥ
/// <param name="feature_threshold"></param>������ֵ
/// <param name="result"></param>-
/// <param name="is_clockwise_direction"></param>�Ƿ�˳ʱ�뵥������
/// <returns></returns>�˱��Ƿ����
bool Deformation_judgment(Mat scr,Mat mask, double feature_threshold,string& result, bool is_clockwise_direction){
vector<double> feature= Feature_calculation(scr, mask, 0, is_clockwise_direction);
if (feature[3] > feature_threshold) {
	return true;
}
else {
	return false;
}
}

/// <summary>
/// �����Ǳ����жϺ���
/// </summary>
/// <param name="scr"></param>�ǵĲü��Ҷ�ͼ
/// <param name="mask"></param>��Ĥ
/// <param name="feature_threshold"></param>������ֵ
bool R_Deformation_judgment(Mat src,  double feature_threshold) {
	Mat background = src.clone();//����
	Mat error = src.clone();//��ֽ��

	//�ն�Ԥ���� ����������
	Mat src_copy = src.clone();
	//Mat threshold_output;
	vector<vector<Point> > preContours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	double meanGray = mean(src_copy)[0];
	threshold(src_copy, background, meanGray * 0.6, 255, THRESH_BINARY);

	/// Find contours
	findContours(background, preContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(preContours.size());

	for (size_t i = 0; i < preContours.size(); i++)
	{
		convexHull(Mat(preContours[i]), hull[i], false);
	}

	/// Draw contours + hull results
	Mat drawing = Mat::zeros(background.size(), CV_8UC1);
	for (size_t i = 0; i < preContours.size(); i++)
	{
		double area = contourArea(preContours[i]);
		//if (area > 100000)
		{
			drawContours(drawing, preContours, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
			drawContours(drawing, hull, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
		}
	}
	src = drawing;
	error = src - background;
	int feature = countNonZero(error);
	cout << feature << endl;
	if (feature > feature_threshold) {
		return true;
	}
	else {
		return false;
	}
}


Mat reshape(Mat cameraMatrix,Mat distCoeffs, Mat map1, Mat map2,Mat frame,Mat frameCalibration) {

	// cameraMatrix = Mat::eye(3, 3, CV_64F);
	//cameraMatrix.at<double>(0, 0) = 1.012265946977986e+05;
	//cameraMatrix.at<double>(0, 1) = -3.231430910165207e+04;
	//cameraMatrix.at<double>(0, 2) = -7.440040622814023e+04;
	//cameraMatrix.at<double>(1, 1) = 1.218123687050294e+05;
	//cameraMatrix.at<double>(1, 2) = -5.193803185637342e+04;

	// distCoeffs = Mat::zeros(5, 1, CV_64F);
	//distCoeffs.at<double>(0, 0) = -0.016461187353794;
	//distCoeffs.at<double>(1, 0) = 5.135636379585712e-05;
	//distCoeffs.at<double>(2, 0) = -0.007742335803243;
	//distCoeffs.at<double>(3, 0) = 0.015372841468934;
	//distCoeffs.at<double>(4, 0) = 0;
	 cameraMatrix = Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = 4.450537506243416e+02;
	cameraMatrix.at<double>(0, 1) = 0.192095145445498;
	cameraMatrix.at<double>(0, 2) = 3.271489590204837e+02;
	cameraMatrix.at<double>(1, 1) = 4.473690628394497e+02;
	cameraMatrix.at<double>(1, 2) = 2.442734958206504e+02;

	 distCoeffs = Mat::zeros(5, 1, CV_64F);
	distCoeffs.at<double>(0, 0) = -0.320311439187776;
	distCoeffs.at<double>(1, 0) = 0.117708464407889;
	distCoeffs.at<double>(2, 0) = -0.00548954846049678;
	distCoeffs.at<double>(3, 0) = 0.00141925006352090;
	distCoeffs.at<double>(4, 0) = 0;
	Size imageSize = frame.size();
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), cameraMatrix,imageSize, CV_16SC2, map1, map2);
	remap(frame, frameCalibration, map1, map2, INTER_LINEAR);

	
	
	
	
	undistort(frame, frameCalibration, cameraMatrix, distCoeffs);

	return frameCalibration;
}