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
bool isArea_1, isArea_2;														//显示异常标志位
String Screen_Type = "R角水滴屏";

Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB);
int lis(vector<int> arr, int len);
void convex_edg(Mat src, Mat &dst, int min_area);
void predict_edg(Mat& dst, vector<int> &value, vector<int>value_num, bool is_reverse);
void predict_num(Mat& dst, vector<int>& value);
void DeleteElem(vector <int> &vec, int elem);

/*========================================================
 *@函 数 名：             convexSetPretreatment
 *@功能描述：             求图像中凸集，获取其凸多面体，改变刘海造成的透视变换切割不准确的情况
 *@Mat                    原图像
 *@Mat                    对应凸集图像
 *@修改时间：             2021/03/16
 *=======================================================*/
void convexSetPretreatment(Mat& src) 
{
	//空洞预处理 针对于主相机
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
* 函 数 名: toushi_white
* 功能描述: 透视变换图像矫正
=========================================================*/
Mat toushi_white(Mat image, Mat M, int border, int length, int width)
{
	Mat perspective;
	cv::warpPerspective(image, perspective, M, cv::Size(length, width), cv::INTER_LINEAR);
	return perspective;
}


/*=========================================================
*@函 数 名:              f_MainCam_PersTransMatCal
*@功能描述:              主黑白/彩色相机R角屏幕的透视变换矩阵计算
*@param _src             输入灰度/彩色图像
*@param _dst             输出显示到客户用图像
*@param border_white     提白底图边缘调整参数值
*@param border_black     提黑底图边缘调整参数值
*@param border_lightleak 提漏光图边缘调整参数值
*@param Mwhite           白底透视变换矩阵
*@param Mblack           黑底透视变换矩阵
*@param Mlightleak       漏光透视变换矩阵
*@param M_white_abshow   显示异常变换矩阵
*@param ID               工位ID号(弃用)
*@ScreenType_Flag        屏幕类型
*@编制时间：		     2020年8月17日
*@备注说明
=========================================================*/
//bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat* Mwhite, Mat* Mbiankuang, Mat* M_white_abshow, int ID, String ScreenType_Flag, int leftRightWhiteFlag)
//{
//	//    double screen_long=size_long/size_width;
//	//    int screen_long=size_long/size_width;
//	bool isArea_1, isArea_2;														//显示异常标志位
//	Mat src = _src.getMat();                                                        //输入源图像
//	if (src.type() == CV_8UC1)														//若输入8位图
//		src = src.clone();															//拷贝原图
//	else
//		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
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
//	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
//	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
//	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
//	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
//	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
//	vector<vector<Point>> contours;													//contours存放点集信息
//	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
//	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
//	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
//	vector<Point2f> src_corner(4), src_corner_biankuang(4), src_corner_abshow(4);   //四个边相交得到角点坐标，漏光角点，显示异常角点
//	Rect rect;																        //最小正外接矩形
//	int x1, y1, x2, y2, x3, y3, x4, y4;			//正接矩阵坐标点信息
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
//			x1 = rect.tl().x;//左上角
//			y1 = rect.tl().y;//左上角
//			x2 = rect.tl().x;//左下角
//			y2 = rect.br().y;//右下角
//			x3 = rect.br().x;//右下角
//			y3 = rect.br().y;//右下角
//			x4 = rect.br().x;//右上角
//			y4 = rect.tl().y;//右上角
//			//int radianEliminate = 230;
//			//int deviation = 160;
//
//			int radianEliminate = 23;
//			int deviation = 16; //可以
//			for (int j = 0; j < contours[i].size(); j++)
//			{
//				//左侧点集
//				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1) * 0.3 && abs(contours[i][j].x - x1) < deviation ||
//					contours[i][j].y > y1 + (y2 - y1) * 0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
//					leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				//右侧点集
//				if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
//					rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				//上侧点集
//				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y1) < deviation)
//					upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				//下侧点集
//				if (contours[i][j].x > x1 + radianEliminate && contours[i][j].x < x4 - radianEliminate && abs(contours[i][j].y - y2) < deviation)
//					downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//			}
//			break;
//		}
//	}
//	if (leftLinePoint.size() == 0 || rightLinePoint.size() == 0 || upLinePoint.size() == 0 || downLinePoint.size() == 0)
//		displayError_Areasignal = 0;
//	//根据轮廓面积判定显示异常
//	if (displayError_Areasignal > 0 && ID == 1)
//		isArea_1 = false;
//	if (displayError_Areasignal == 0 && ID == 1)
//		isArea_1 = true;
//	if (displayError_Areasignal > 0 && ID == 2)
//		isArea_2 = false;
//	if (displayError_Areasignal == 0 && ID == 2)
//		isArea_2 = true;
//	//未提取到屏幕判定显示异常提取边缘角落
//	if (displayError_Areasignal == 0)
//	{
//		vector<Point2f> src_points(4);
//		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
//		vector<Point2f> dst_points(4);
//		if (ScreenType_Flag == "矩形屏")
//			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
//		else        //pixel_num
//			//dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
//			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
//		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
//		//*Mlightleak = cv::getPerspectiveTransform(src_points, dst_points);
//		*Mbiankuang = cv::getPerspectiveTransform(src_points, dst_points);
//		*M_white_abshow = cv::getPerspectiveTransform(src_points, dst_points);
//	}
//	//正常屏幕提取屏幕的四个角点
//	else
//	{
//		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
//		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
//		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
//		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线
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
//		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
//		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
//		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
//		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点
//
//																						//对4个角点的坐标位置进行微调（白底图以及黑底图）
//		src_corner[0].x = src_corner[0].x - border_white;
//		src_corner[0].y = src_corner[0].y - border_white;
//		src_corner[1].x = src_corner[1].x - border_white;
//		src_corner[1].y = src_corner[1].y + border_white;
//		src_corner[2].x = src_corner[2].x + border_white;
//		src_corner[2].y = src_corner[2].y + border_white;
//		src_corner[3].x = src_corner[3].x + border_white;
//		src_corner[3].y = src_corner[3].y - border_white;
//		//对4个角点的坐标位置进行微调（漏光检测图）
//		src_corner_biankuang[0].x = src_corner[0].x - border_biankuang;
//		src_corner_biankuang[0].y = src_corner[0].y - border_biankuang;
//		src_corner_biankuang[1].x = src_corner[1].x - border_biankuang;
//		src_corner_biankuang[1].y = src_corner[1].y + border_biankuang;
//		src_corner_biankuang[2].x = src_corner[2].x + border_biankuang;
//		src_corner_biankuang[2].y = src_corner[2].y + border_biankuang;
//		src_corner_biankuang[3].x = src_corner[3].x + border_biankuang;
//		src_corner_biankuang[3].y = src_corner[3].y - border_biankuang;
//		//显示异常(白底图)
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
//		if (ScreenType_Flag == "矩形屏")
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
	bool isArea_1, isArea_2;														//显示异常标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
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
	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4), src_corner_biankuang(4), src_corner_abshow(4);   //四个边相交得到角点坐标，漏光角点，显示异常角点
	Rect rect;																        //最小正外接矩形
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
					//左侧点集
					for (int k = 1; k < 4; k++) {
						if (i < (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							leftLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//右侧点集
						if (i > (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							rightLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//上侧点集
						if (j < (rect.y + rect.height / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + k, i + 7 * k) > 100 || sra_canny.at<uchar>(j - k, i + 7 * k) > 100 || sra_canny.at<uchar>(j, i + 5) > 100)) {
							upLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//下侧点集
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
	//根据轮廓面积判定显示异常
	if (displayError_Areasignal > 0 && ID == 1)
		isArea_1 = false;
	if (displayError_Areasignal == 0 && ID == 1)
		isArea_1 = true;
	if (displayError_Areasignal > 0 && ID == 2)
		isArea_2 = false;
	if (displayError_Areasignal == 0 && ID == 2)
		isArea_2 = true;
	//未提取到屏幕判定显示异常提取边缘角落
	if (displayError_Areasignal == 0)
	{
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else        //pixel_num
			//dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);
		//*Mlightleak = cv::getPerspectiveTransform(src_points, dst_points);
		*Mbiankuang = cv::getPerspectiveTransform(src_points, dst_points);
		*M_white_abshow = cv::getPerspectiveTransform(src_points, dst_points);
	}
	//正常屏幕提取屏幕的四个角点
	else
	{
		fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
		fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
		fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
		fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线

		src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
		src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
		src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
		src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点

																						//对4个角点的坐标位置进行微调（白底图以及黑底图）
		src_corner[0].x = src_corner[0].x - border_white;
		src_corner[0].y = src_corner[0].y - border_white;
		src_corner[1].x = src_corner[1].x - border_white;
		src_corner[1].y = src_corner[1].y + border_white;
		src_corner[2].x = src_corner[2].x + border_white;
		src_corner[2].y = src_corner[2].y + border_white;
		src_corner[3].x = src_corner[3].x + border_white;
		src_corner[3].y = src_corner[3].y - border_white;
		//对4个角点的坐标位置进行微调（漏光检测图）
		src_corner_biankuang[0].x = src_corner[0].x - border_biankuang;
		src_corner_biankuang[0].y = src_corner[0].y - border_biankuang;
		src_corner_biankuang[1].x = src_corner[1].x - border_biankuang;
		src_corner_biankuang[1].y = src_corner[1].y + border_biankuang;
		src_corner_biankuang[2].x = src_corner[2].x + border_biankuang;
		src_corner_biankuang[2].y = src_corner[2].y + border_biankuang;
		src_corner_biankuang[3].x = src_corner[3].x + border_biankuang;
		src_corner_biankuang[3].y = src_corner[3].y - border_biankuang;
		//显示异常(白底图)
		src_corner_abshow[0].x = src_corner[0].x - border_white + 10;
		src_corner_abshow[0].y = src_corner[0].y - border_white + 10;
		src_corner_abshow[1].x = src_corner[1].x - border_white + 10;
		src_corner_abshow[1].y = src_corner[1].y + border_white - 10;
		src_corner_abshow[2].x = src_corner[2].x + border_white - 10;
		src_corner_abshow[2].y = src_corner[2].y + border_white - 10;
		src_corner_abshow[3].x = src_corner[3].x + border_white - 10;
		src_corner_abshow[3].y = src_corner[3].y - border_white + 10;

		vector<Point2f> dst_corner(4);
		if (ScreenType_Flag == "矩形屏")
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
*@函 数 名:     getPointSlopeCrossPoint
*@功能描述:     计算点斜式两条直线的交点
*@param LineA   平行线条
*@param LineB   垂直线条
*@编制时间：    2020年8月17日
*@备注说明
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
*@函 数 名:              f_FrontBackCam_PersTransMatCal
*@功能描述:              前后相机R角透视变换矩阵计算函数
*@param _src             输入灰度/彩色图像
*@param Mwhite           白底透视变换矩阵
*@ScreenType_Flag        屏幕类型
*@编制时间：		     2020年8月21日
*@备注说明
=========================================================*/
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat* Mwhite, String ScreenType_Flag)
{
	bool Ext_Result_Front_Back;                                                     //提取屏幕成功标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);										    //灰度化彩色图
	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4);                                                  //四个边相交得到角点坐标
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //正接矩阵坐标点信息
	vector<Point2f> dst_corner(4);                                                  //透视变换后的点的信息
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 2000000 && area < 5000000)
		{
			displayError_Areasignal++;
			rect = cv::boundingRect(contours[i]);                                           //最小外接矩形提取
			Ext_Result_Front_Back = false;                                                             //提取到屏幕
			//矩形面积缩小1/3，并得到新的矩形顶点
			int PixelGap1 = rect.tl().x;
			int PixelGap2 = src.cols - (rect.tl().x + rect.width);
			//横坐标获取
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
			//纵坐标获取
			y1 = rect.tl().y;
			y2 = rect.br().y;
			if (y2 >= src.rows)
				y2 = src.rows - 1;
			y3 = y2;
			y4 = y1;
			//取直线的参数设置
			int radianEliminate = 350;//(R角)左右使用
			int radianEliminate1 = 480;//(R角)左右使用
			int radianEliminate2 = 230;//(R角)上下使用
			int deviation = 120;//(斜线带来的误差)左右使用
			int deviation2 = 200;//(斜线带来的误差)上下使用
			if (PixelGap1 > PixelGap2)
			{
				//外接矩形缩小
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
						//左侧点集
						if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate1 && abs(contours[i][j].x - x1) < deviation)
							leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//上侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//下侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//右侧点集
					rightLinePoint.push_back(Point((x3 + x4) / 2, (y3 + y4) / 2));
					if (leftLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//直线拟合
						fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
						rightLine_Fit[0] = leftLine_Fit[0];
						rightLine_Fit[1] = leftLine_Fit[1];
						rightLine_Fit[2] = rightLinePoint[0].x;
						rightLine_Fit[3] = rightLinePoint[0].y;                                         //右侧拟合直线
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //下侧拟合直线
						//角点提取
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点
						//透视变换矩阵计算
						if (ScreenType_Flag == "矩形屏")
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
				//外接矩形缩小
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
						//右侧点集
						if (contours[i][j].y > y1 + radianEliminate1 && contours[i][j].y < y1 + (y2 - y1) * 0.3 && abs(contours[i][j].x - x3) < deviation || contours[i][j].y > y1 + (y2 - y1) * 0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
							rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//上侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
							upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
						//下侧点集
						if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
							downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
					}
					//左侧点集
					leftLinePoint.push_back(Point((x1 + x2) / 2, (y1 + y2) / 2));
					if (rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
					{
						//拟合直线
						fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
						leftLine_Fit[0] = rightLine_Fit[0];
						leftLine_Fit[1] = rightLine_Fit[1];
						leftLine_Fit[2] = leftLinePoint[0].x;
						leftLine_Fit[3] = leftLinePoint[0].y;                                           //左侧拟合直线
						fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				    //上侧拟合直线
						fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线
						//角点提取
						src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
						src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
						src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
						src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点
						//透视变换矩阵计算
						if (ScreenType_Flag == "矩形屏")
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
	//没有提取到屏幕
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Front_Back = true; //没有提取到屏幕
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //透视变换矩阵提取
	}

	return Ext_Result_Front_Back;
}

/*=========================================================
*@函 数 名:              f_LeftRightCam_PersTransMatCal
*@功能描述:              左右相机R角透视变换矩阵计算函数
*@param _src             输入灰度/彩色图像
*@param Mwhite           白底透视变换矩阵
*@ScreenType_Flag        屏幕类型
*@leftRightWhiteFlag     白底左右相机标志位
*@编制时间：		     2021年03月15日
*@备注说明              use
=========================================================*/
//bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat* Mwhite, Mat* M_R_1_E, String ScreenType_Flag, int leftRightWhiteFlag, int border_white)
//{
//	bool Ext_Result_Left_Right;                                                     //提取屏幕成功标志位
//	Mat src = _src.getMat();                                                        //输入源图像
//	if (src.type() == CV_8UC1)														//若输入8位图
//		src = src.clone();															//拷贝原图
//	else
//		cvtColor(src, src, CV_BGR2GRAY);                                            //灰度化彩色图
//
//	if (leftRightWhiteFlag == 1)
//	{
//		convexSetPretreatment(src);
//	}
//
//	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
//	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像
//	threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
//	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
//	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
//	vector<vector<Point>> contours;													//contours存放点集信息
//	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
//	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
//	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
//	vector<Point2f> src_corner(4);                                                  //四个边相交得到角点坐标
//	vector<Point2f> src_corner_enlarge(4);
//	Rect rect;																        //最小正外接矩形
//	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //正接矩阵坐标点信息
//	vector<Point2f> dst_corner(4);                                                  //透视变换后的点的信息
//	for (vector<int>::size_type i = 0; i < contours.size(); i++)
//	{
//		double area = contourArea(contours[i]);
//		if (area > 150000 && area < 600000)
//		{
//
//
//			displayError_Areasignal++;
//			rect = cv::boundingRect(contours[i]);                                           //最小外接矩形提取
//			Ext_Result_Left_Right = false;                                                             //提取到屏幕
//
//			//cv::rectangle(src, rect, Scalar(255, 0, 0), 5, LINE_8, 0);
//			x1 = rect.tl().x;//左上角
//			y1 = rect.tl().y;//左上角
//			x2 = rect.tl().x;//左下角
//			y2 = rect.br().y;//左下角
//			x3 = rect.br().x;//右下角
//			y3 = rect.br().y;//右下角
//			x4 = rect.br().x;//右上角
//			y4 = rect.tl().y;//右上角
//																									   //矩形面积缩小1/3，并得到新的矩形顶点
//			//取直线的参数设置
//			//int radianEliminate = 230;//(R角)左右使用
//			//int radianEliminate2 = 360;//(R角)上下使用
//			int radianEliminate = 0;//(R角)左右使用
//			int radianEliminate2 = 0;//(R角)上下使用
//			int deviation = 200;//(斜线带来的误差)左右使用
//			int deviation2 = 120;//(斜线带来的误差)上下使用
//
//			if (displayError_Areasignal != 0)
//			{
//				for (int j = 0; j < contours[i].size(); j++)
//				{
//					//左侧点集
//					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y1 + (y2 - y1) * 0.3 && abs(contours[i][j].x - x1) < deviation ||
//						contours[i][j].y > y1 + (y2 - y1) * 0.7 && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x1) < deviation)
//						leftLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//					//右侧点集
//					if (contours[i][j].y > y1 + radianEliminate && contours[i][j].y < y2 - radianEliminate && abs(contours[i][j].x - x3) < deviation)
//						rightLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//					//上侧点集
//					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y1) < deviation2)
//						upLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//					if (contours[i][j].x > x1 + radianEliminate2 && contours[i][j].x < x4 - radianEliminate2 && abs(contours[i][j].y - y2) < deviation2)
//						downLinePoint.push_back(Point(contours[i][j].x, contours[i][j].y));
//				}
//				//下侧点集
//				//downLinePoint.push_back(Point((x2 + x3) / 2, (y2 + y3) / 2));
//				if (leftLinePoint.size() != 0 && rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
//				{
//					//直线拟合
//					fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
//					fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
//					fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
//					downLine_Fit[0] = upLine_Fit[0];
//					downLine_Fit[1] = upLine_Fit[1];
//					downLine_Fit[2] = downLinePoint[0].x;
//					downLine_Fit[3] = downLinePoint[0].y;                                           //下侧直线确定
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
//					//角点提取
//					src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
//					src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
//					src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
//					src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点
//
//
//					//src_corner_enlarge[0] = Point2f(xcoordinate1 + tl.x - border_white, ycoordinate1 + tl.y - border_white);	                         //左上角
//					//src_corner_enlarge[1] = Point2f(xcoordinate2 + bl.x - border_white, ycoordinate2 - height / 3 + bl.y + border_white);              //左下角
//					//src_corner_enlarge[2] = Point2f(xcoordinate3 - width / 4 + br.x + border_white, ycoordinate3 - height / 3 + br.y + border_white);	 //右下角
//					//src_corner_enlarge[3] = Point2f(xcoordinate4 - width / 4 + tr.x + border_white, ycoordinate4 + tr.y - border_white);	             //右上角
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
//					//透视变换矩阵计算
//					if (ScreenType_Flag == "矩形屏")
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
//	//没有提取到屏幕
//	if (displayError_Areasignal == 0)
//	{
//		Ext_Result_Left_Right = true; //没有提取到屏幕
//		vector<Point2f> src_points(4);
//		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
//		vector<Point2f> dst_points(4);
//		if (ScreenType_Flag == "矩形屏")
//			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
//		else
//			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
//		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //透视变换矩阵提取
//		*M_R_1_E = cv::getPerspectiveTransform(src_points, dst_points);
//	}
//
//	return Ext_Result_Left_Right;
//}
//

bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat* Mwhite, Mat* M_R_1_E, String ScreenType_Flag, int leftRightWhiteFlag, int border_white)
{
	bool Ext_Result_Left_Right = true;                                                     //提取屏幕成功标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);                                            //灰度化彩色图

	convexSetPretreatment(src);
	CV_Assert(src.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(src.size(), CV_8UC1);                              //二值图像

	threshold(src, binaryImage, mean(src)[0] * 0.6, 255, THRESH_BINARY);
	//threshold(src, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4);                                                  //四个边相交得到角点坐标
	vector<Point2f> src_corner_enlarge(4);
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //正接矩阵坐标点信息
	vector<Point2f> dst_corner(4);                                                  //透视变换后的点的信息
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
					//左侧点集
					for (int k = 1; k < 4; k++) {
						if (i < (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							leftLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//右侧点集
						if (i > (rect.x + rect.width / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 5 * k, i + k) > 100 || sra_canny.at<uchar>(j - 4, i) > 100)) {
							rightLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//上侧点集
						if (j < (rect.y + rect.height / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + k, i + 7 * k) > 100 || sra_canny.at<uchar>(j - k, i + 7 * k) > 100 || sra_canny.at<uchar>(j, i + 5) > 100)) {
							upLinePoint.push_back(Point(i, j));
						}
					}
					for (int k = 1; k < 4; k++) {
						//下侧点集
						if (j > (rect.y + rect.height / 2) && sra_canny.at<uchar>(j, i) > 100 && (sra_canny.at<uchar>(j + k, i + 7 * k) > 100 || sra_canny.at<uchar>(j - k, i + 7 * k) > 100 || sra_canny.at<uchar>(j, i + 5) > 100)) {
							downLinePoint.push_back(Point(i, j));
						}
					}
				}
			}

			if (leftLinePoint.size() != 0 && rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
			{
				//直线拟合
				fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
				fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
				fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
				fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线

				//角点提取
				src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
				src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
				src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
				src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点

				src_corner_enlarge[0].y = src_corner[0].y - border_white;
				src_corner_enlarge[0].x = src_corner[0].x - border_white;
				src_corner_enlarge[1].y = src_corner[1].y + border_white;
				src_corner_enlarge[1].x = src_corner[1].x - border_white;
				src_corner_enlarge[2].y = src_corner[2].y + border_white;
				src_corner_enlarge[2].x = src_corner[2].x + border_white;
				src_corner_enlarge[3].y = src_corner[3].y - border_white;
				src_corner_enlarge[3].x = src_corner[3].x + border_white;
				//透视变换矩阵计算
				if (ScreenType_Flag == "矩形屏")
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
	//没有提取到屏幕
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Left_Right = true; //没有提取到屏幕
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //透视变换矩阵提取
		*M_R_1_E = cv::getPerspectiveTransform(src_points, dst_points);
	}
	return Ext_Result_Left_Right;
}
/// <summary>
/// 屏幕非凸部分提取（挖孔、水滴部分）
/// </summary>返回是否非凸，若为非凸，计算非凸区域并输出非凸位置二值化图像
/// <param name="scr"></param>透视变换后的白底图
/// <param name="feitu"></param>返回的非凸图
/// <param name="feitu_rect"></param>返回的非凸部分位置
/// <param name="feitu_radio"></param>非凸部分最大比值限制
/// <param name="feitu_lowerArea"></param>非凸部分下界
/// <param name="feitu_higtherArea"></param>非凸部分上界
/// <returns></returns>
bool get_feitu_area(Mat scr, Mat& feitu,Rect& feitu_rect,double feitu_radio,int feitu_lowerArea,int feitu_higherArea)
{
	//double feitu_radio=6;
	//int feitu_lowerArea=4000,feitu_higherArea=300*500;
	Mat binary,dst;
	threshold(scr, binary, mean(scr) [0]* 0.6, 255, THRESH_BINARY);//二值化
	dst = binary.clone();
	convexSetPretreatment(binary);
	dst = binary - dst;
	//创建结构元素
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3), Point(-1, -1));
	//执行开操作
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

			//长宽比排除
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
/// \brief find_camera_roi计算相机ROI
/// \param src输入图像
/// \param roi_rect返回的ROI矩阵
/// \param roi_radio满足的长宽比要求
/// \param roi_lowerArea面积下限
/// \param roi_expand在原始ROI基础上扩大的宽度
/// \return是否提取到ROI
///

bool find_camera_roi(Mat src, Rect& roi_rect, float roi_radio, int roi_lowerArea, int roi_expand)
{
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);                                            //灰度化彩色图
	Mat dst;
	threshold(src, dst, mean(src)[0] * 0.8, 255, THRESH_BINARY);//二值化
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(5, 5), Point(-1, -1));
	//执行开操作
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

			//长宽比排除
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
	
	
	
	//闭运算,弥合内部空洞,连接相距很近的区域
	Mat dst1, dst2, dst3,th1, img_gray, img_gray2,th2;
	adaptiveThreshold(scr, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, border_size, -1);
	threshold(scr, th1, 0.9*mean(scr)[0], 255, CV_THRESH_BINARY);
	threshold(scr, th2, 0.9 * mean(scr)[0], 255, CV_THRESH_BINARY_INV);

	bitwise_and(th1, scr, img_gray);

	Mat element = getStructuringElement(MORPH_RECT, Size(border_size, border_size));//闭操作结构元素
	Mat element1 = getStructuringElement(MORPH_CROSS, Size(5, 5));//闭操作结构元素
	morphologyEx(scr, dst1, CV_MOP_CLOSE, element);   //闭运算形态学操作。可以减少噪点
    dilate(scr, dst2, element);//膨胀
//    //imwrite("D:pengzhang.bmp",th_result);
    erode(scr, dst3, element);//膨胀
	bitwise_and(dst2, th2, img_gray2);
	bitwise_or(img_gray, img_gray2, dst2);

	adaptiveThreshold(dst2, dst3, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, border_size, -1);


	
	///通过均值判断边界所在区域
	mean(scr(Rect(0, 0,  scr.cols-1,scr.rows / 2)));
	mean(scr(Rect( 0,scr.rows / 2-1, scr.cols-1, scr.rows / 2)));
	mean(scr(Rect(0, 0, scr.cols/2, scr.rows-1)));
	mean(scr(Rect( scr.cols / 2-1,0, scr.cols/2, scr.rows-1)));
	Canny(scr.clone(), dst,  mean(scr)[0], mean(scr)[0]);
	/// 按边界垂直方向逐行/列求所在行列的一阶二阶导数
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
	/// 对图像进行滤波操作
	filter2D(scr, dx1, scr.depth(), kernel3);
	filter2D(scr, dx2, scr.depth(), kernel30);
	bitwise_or(dx1, dx2, dx);

	filter2D(scr, dy1, scr.depth(), kernel4);
	filter2D(scr, dy2, scr.depth(), kernel40);
	bitwise_or(dy1, dy2, dy);
	dst = scr.clone();

	dxy = dy.clone();
	///根据一阶二阶导数填充屏幕边界
	for (int i = 0; i < scr.rows; i++) {
		float mean_col = mean(dxy)[0];//列梯度均值
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
				j - border_size*2 < 0 ? k = 0 : k = j - border_size*2;//防止越界
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
	///根据一阶二阶导数填充屏幕边界
	for (int i = 0; i < scr.rows; i++) {
		float mean_col = mean(dxy)[0];//列梯度均值
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
				j + border_size * 2 > scr.cols - 1 ? k = scr.cols - 1 : k = j + border_size * 2;//防止越界
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
	///根据一阶二阶导数填充屏幕边界
	for (int j = 0; j < scr.cols ; j++) {
		float mean_col = mean(dxy)[0];//列梯度均值
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
				i - border_size*2 < 0 ? k = 0 : k = i - border_size*2;//防止越界
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
	///根据一阶二阶导数填充屏幕边界
	for (int j = 0; j < scr.cols; j++) {
		float mean_col = mean(dxy)[0];//列梯度均值
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
				i + border_size*2 > scr.rows-1 ? k = scr.rows-1 : k = i + border_size*2;//防止越界
				for (k; k > i - 1; k--)
				{
					dst.at<uchar>(k, j) = scr.at<uchar>(i - 1, j);
				}
				dxy.at<uchar>(i - 1, j) = 255;
				break;
			}
		}

	}
	/// 自适应二值化，返回和原图大小一样图
}




/// <summary>
/// 把1*x的矩阵转为用【起始位置】+【数据】+【连续个数】表示的向量
/// </summary>例如1,1,1,2,2,3,表示为【0】【1】【3】，【3】【2】【2】，【5】【3】【1】
/// <param name="scr"></param>原矩阵
/// <returns></returns>
vector<vector<int>> Mat_to_vector(Mat scr) {
	vector<vector<int>> dst_vec;
	int last_data = scr.at<uchar>(0, 0);//上次的值
	int last_location = 0;//上次的终止位置
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 1; j < scr.cols; j++) {
			if (scr.at<uchar>(i, j) != last_data && scr.at<uchar>(i, j) != -1 || j == scr.cols - 1)
			{
				vector<int>mat_data;
				mat_data.push_back(last_location);//起始位置
				mat_data.push_back(last_data);//数据
				mat_data.push_back(j - last_location);//连续的相同数据个数
				dst_vec.push_back(mat_data);
				last_data = scr.at<uchar>(i, j);//更新数据
				last_location = j;//更新位置
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
		//补全被噪点打断的相邻元素
		int last_end_index= scr_data[i][2]+ scr_data[i][0];//结束地址=开始地址+持续数量
		int start= scr_data[i][0];

			for (int j = i+1; j < scr_data.size(); j++) {
				//补全被噪点打断的相邻元素
				if (scr_data[i][1] == scr_data[j][1] && scr_data[j][0] - (last_end_index) < avg)
				{
					last_end_index = scr_data[j][2] + scr_data[j][0];//结束地址=开始地址+持续数量
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
	cout << "数据数：" << scr_data.size() << "噪声数：" << scr_noise.size() << endl;
	cout << "数据数：" << value.size() << "升序数：" << a << "差值：" << value.size() -a<< endl;

	std::vector<int>::iterator max = std::max_element(std::begin(value), std::end(value));
	//	vector<double>::iterator max=max_element(arr.begin(),arr.end());// 或者也可以这样表示，计算max
	//std::cout << "Max element is " << *max << " at position " << std::distance(std::begin(value), max) << std::endl;
	//输出值一定是带*  输出*max，表示解引用 
	auto min = std::min_element(std::begin(value), std::end(value));
	//std::cout << "min element is " << *min << " at position " << std::distance(std::begin(value), min) << std::endl;

	Mat data(*max, value.size(), CV_8UC1, Scalar(0));//建立一张图片，用于分离非凸区域进行噪点排除
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
	int num = 2;//偏离点的大小

	///原始数据判别
	int change_lenght=0;
	//for (int i = 0; i < error.cols; i++) {
	//	for (int j = error.rows-1; j > 0; j--) {
	//		if (error.at<uchar>(j, i) >100) {
	//			if (abs(j - value[i])+1 >= num) 
	//			{
	//				result_error = true; }
	//			change_lenght = change_lenght + value_num[i];//求改变的数据宽度（对应屏幕的像素宽度）
	//			break;
	//		}			
	//	}
	//}
	for (int i = 0; i < error.cols; i++) {	
		if (abs(predict_value[i] - value[i]) >= num) {
			change_lenght = change_lenght + value_num[i];//求改变的数据宽度（对应屏幕的像素宽度）
			result_error = true;
			}
	}
	if (change_lenght > length / 2) {
		//result_error = false;
	}
	///取反图判别
	change_lenght = 0;
	//for (int i = 0; i < error.cols; i++) {
	//	for (int j = 0; j < error_contrary.rows; j++) {
	//		if (error_contrary.at<uchar>(j, i) > 100) {
	//			if (abs(j - value[i]) >= num)
	//			{
	//				result_contrary = true;
	//			}
	//			change_lenght = change_lenght + value_num[i];//求改变的数据宽度（对应屏幕的像素宽度）
	//			break;
	//		}
	//	}
	//}
	for (int i = 0; i < error.cols; i++) {
		if (abs(predict_value_contrary[i] -*max+ value[i]) >= num) {
			change_lenght = change_lenght + value_num[i];//求改变的数据宽度（对应屏幕的像素宽度）
			result_contrary = true;
		}
	}

	if (change_lenght > length / 2) {
		//result_contrary = false;
	}
	///逆序超过一定长度直接判为变形
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
		Mat num_data(*max_element(value_num.begin(), value_num.end()), value_num.size(), CV_8UC1, Scalar(0));//建立一张图片，用于分离非凸区域进行噪点排除	
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
				//change_lenght = change_lenght + value_num[i];//求改变的数据宽度（对应屏幕的像素宽度）
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
	cout << "原图结果：" << result_error << "反图结果：" << result_contrary << "单调性结果：" << result_down << endl;
	cout << "总结果" <<( result_error || result_contrary|| result_down )<< endl;
	return result_contrary|| result_error|| result_down;
}


void convex_edg(Mat src, Mat& dst, int min_area)
{

	{
		//空洞预处理 针对于边界算法
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

	///为了复用，取反图先反转
	if (is_reverse)
	{
		///先把序列取反，再反转
		int temp_max =* std::max_element(std::begin(value), std::end(value));
		for (int i = 0; i < value.size(); i++)
		{
			value[i] = temp_max - value[i];
		}
		reverse(value.begin(), value.end());
		reverse(value_num.begin(), value_num.end());
	}
	std::vector<int>::iterator num_max = std::max_element(std::begin(value_num), std::end(value_num));

	int num_max_index = distance(begin(value_num), num_max);//最多的数所在索引位置
	std::vector<int>::iterator local_max = std::max_element(std::begin(value), std::end(value));
	int local_max_index= distance(begin(value), local_max);//最多的数所在索引位置
	
	if (value.size() > 1)
	{
		if (*max_element(begin(value), begin(value) + num_max_index) >= value[num_max_index])
		{
			for (int i = num_max_index; i >= 0; i--) {
				//把原来左边比右边值多的部分删掉
				if (value[num_max_index] < value[i] && i >= local_max_index) {
					for (int j = num_max_index; j >= local_max_index; j--) {
						value[j] = value[num_max_index];
					}
					break;
				}
			}
		}

		for (int i = num_max_index - 1; i >= 0; i--) {
			//保证左侧不减
			if (value[i] > value[i + 1]) {
				value[i] = value[i + 1];
			}
		}

		for (int i = num_max_index; i < value.size() - 1; i++) {
			//保证右侧不减
			if (value[i] > value[i + 1]) {
				value[i + 1] = value[i];
			}
		}

	}

	//////为了复用，取反图先反转、生成图时再反转回去
	Mat data(max, value.size(), CV_8UC1, Scalar(0));//建立一张图片，用于分离非凸区域进行噪点排除

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

	///为了复用，取反图先反转
	if (is_reverse)
	{
		///先把序列取反，再反转
		int temp_max = *std::max_element(std::begin(value), std::end(value));
		for (int i = 0; i < value.size(); i++)
		{
			value[i] = temp_max - value[i];
		}
	}
	sort(value.begin(), value.end());//从小到大排

	//////为了复用，取反图先反转、生成图时再反转回去
	Mat data(max, value.size(), CV_8UC1, Scalar(0));//建立一张图片，用于分离非凸区域进行噪点排除

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
/// /求最大递增序列
/// </summary>


int lis(vector<int> arr, int len)
{
	vector<int> longest(arr);
	for (int i = 0; i < len; i++)
		longest[i] = 1;

	for (int j = 1; j < len; j++) {
		for (int i = 0; i < j; i++) {
			if (arr[j] > arr[i] && longest[j] < longest[i] + 1) { //注意longest[j]<longest[i]+1这个条件，不能省略。  
				longest[j] = longest[i] + 1; //计算以arr[j]结尾的序列的最长递增子序列长度  
			}
		}
	}

	int max = 0;
	for (int j = 0; j < len; j++) {
		//cout << "longest[" << j << "]=" << longest[j] << endl;
		if (longest[j] > max) max = longest[j];  //从longest[j]中找出最大值  
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
	//获取点斜式的点和斜率
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];

	double k = line_para[1] / line_para[0];

	//计算直线的端点(y = k(x - x0) + y0)
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
		int num = 0;//距离边界像素数
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
		int num = 0;//距离边界像素数
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
		int num = 0;//距离边界像素数
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
		int num = 0;//距离边界像素数
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
	mask_we(Rect(border, 0, dst_west.cols-2*border, 1)) = 255;//设定搜索区域
	double min_w, min_e;
	minMaxLoc(dst_west, &min_w, 0, 0, 0, mask_we);
	minMaxLoc(dst_east, &min_e, 0, 0, 0, mask_we);

	Mat mask_ns(1, dst_north.cols, CV_8UC1, Scalar::all(0));
	mask_ns(Rect(border, 0, dst_north.cols - 2 * border, 1)) = 255;//设定搜索区域
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
		int num = 0;//距离边界像素数
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
		int num = 0;//距离边界像素数
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
		int num = 0;//距离边界像素数
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
		int num = 0;//距离边界像素数
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
	mask_we(Rect(border, 0, dst_west.cols - 2 * border, 1)) = 255;//设定搜索区域
	double min_w, min_e, max_w, max_e;
	minMaxLoc(dst_west, &min_w, &max_w, 0, 0, mask_we);
	minMaxLoc(dst_east, &min_e, &max_e, 0, 0, mask_we);

	Mat mask_ns(1, dst_north.cols, CV_8UC1, Scalar::all(0));
	mask_ns(Rect(border, 0, dst_north.cols - 2 * border, 1)) = 255;//设定搜索区域
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





	//feature_s[3] > 350 ? result = result + "左边框变形" : result = result + "";///各个边最终判别改这个
	//feature_n[3] > 350 ? result = result + "右边框变形" : result = result + "";
	//feature_e[3] > 350 ? result = result + "下边框变形" : result = result + "";
	//feature_w[3] > 180 ? result = result + "上边框变形" : result = result + "";


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

	///判断数量，防止越界
	if (value.size() > 3)
	{
		int max = *std::max_element(std::begin(value), std::end(value));

		vector<int> value_fliter(value);

		for (int i = 0; i < value.size() - 2; i++)
		{
			value[i + 1] = (value_fliter[i] + value_fliter[i + 1] + value_fliter[i + 2]) / 3;
		}


		sort(value.begin() + 1, value.end() - 1);//从小到大排
		reverse(value.begin()+1, value.end()-1);//变成从大到小
	//////为了复用，取反图先反转、生成图时再反转回去
		Mat data(max, value.size() , CV_8UC1, Scalar(0));//建立一张图片，用于分离非凸区域进行噪点排除
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
			//删除指定元素，返回指向删除元素的下一个元素的位置的迭代器
			it = vec.erase(it);
		else
			//迭代器指向下一个元素位置
			++it;
	}
}


bool is_deformation(Mat scr, int length, int border) {
	Mat out;
	int histSize[1] = { 256 };  //灰度值Size：256个
	float hrange[2] = { 0,255 }; //灰度范围[0-255]
	const float* ranges[1] = { hrange }; //单个灰度范围[0-255]
	int channels = 0;
	calcHist(&scr, 1, &channels, Mat(), out, 1, histSize, ranges, true, false);
	double maxVal = 0;
	minMaxLoc(scr, NULL, &maxVal,NULL,NULL);
	Mat data_scr(maxVal + 1, scr.cols, CV_8UC1, Scalar(0));//建立一张图片，用于生成预测的边界
	for (int i = 0; i < scr.cols; i++) {
		for (int j = 0; j < scr.at<uchar>(0,i); j++) {
			data_scr.at<uchar>(j, i) = 255;
		}
	}


	Mat data(maxVal+1, scr.cols, CV_8UC1, Scalar(0));//建立一张图片，用于生成预测的边界
	{	
		int start_index = 0;//起始列索引
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
	Mat data(maxVal + 1, sum.cols, CV_8UC1, Scalar(0));//建立一张图片，用于生成预测的边界
	for (int j = 0; j < sum.cols; j++) {
		for (int k = 0; k < sum.at<uchar>(0, j); k++) {
			data.at<uchar>(k, j) = 255;
		}
	}
	flip(scr_a, scr_a, 1);
	Mat error_ab = scr_a - scr_b;
	Mat error_ba = scr_b - scr_a;
	Mat data_errorab(maxVal + 1, sum.cols, CV_8UC1, Scalar(0));//建立一张图片，用于生成预测的边界
	for (int j = 0; j < sum.cols; j++) {
		for (int k = 0; k < error_ab.at<uchar>(0, j); k++) {
			data_errorab.at<uchar>(k, j) = 255;
		}
	}
	Mat data_errorba(maxVal + 1, sum.cols, CV_8UC1, Scalar(0));//建立一张图片，用于生成预测的边界
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
	start.x>end.x? dst(Rect(end.x, 0, start.x-end.x,rows)) = 255: dst(Rect(start.x, 0, end.x - start.x, rows)) = 255;//设定搜索区域
	bitwise_and(dst_b, dst, dst_b);

}
vector<Point2i> get_predict_line( Point start, Point end,Mat &predictline, bool is_clockwise_direction)
{
		vector<Point2i>predict_line;
		//if (start.x>end.x)
		//{
		//	Point swap= end;//用于交换start、end的中介变量
		//	end = start;
		//	start = swap;
		//}
		//if (!is_clockwise_direction) {
		//	Point swap = end;//用于交换start、end的中介变量
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
	//	Point swap = end;//用于交换start、end的中介变量
	//	end = start;
	//	start = swap;
	//}
	//if (!is_clockwise_direction) {
	//	int swap = end.y;//用于交换start、end的中介变量
	//	end.y = start.y;
	//	start.y = swap;
	//}
	Mat predictline;//理论直线
	get_predict_line(start,end, predictline,  is_clockwise_direction);
	scr.convertTo(scr, CV_16SC1);
	predictline.convertTo(predictline, CV_16SC1);
	Mat error = scr - predictline;

	Mat mask_we(1, start.x + end.x, CV_8UC1, Scalar::all(0));
	mask_we(Rect(start.x, 0, end.x - start.x, 1)) = 255;//设定搜索区域
	double min,max;
	Point  max_index, min_index;
	minMaxLoc(error, &min,  &max, &min_index, &max_index, mask_we);
	double sin_predict = abs(end.x - start.x) / sqrt((end.x - start.x)* (end.x - start.x) + (end.y - start.y)* (end.y - start.y));

	feature_result.push_back(max-min);//待选特征1
	feature_result.push_back((max - min)/ abs(end.y - start.y));//待选特征2
	feature_result.push_back((max - min) / abs(end.y - start.y)*sin_predict);//待选特征3
	feature_result.push_back((max - min)  * sin_predict);//待选特征4

	cout << " 差值： " << feature_result[0] << " 归一差值： " << feature_result[1] << " 归一法方向差值： " << feature_result[2] << " 法方向差值： " << feature_result[3] << endl;

	return feature_result;
}

/// <summary>
/// 特征计算
/// </summary>
/// <param name="scr"></param>单个边滤波后的精确距离矩阵(1*x大小)
/// <param name="mask"></param>此边对应要检测掩膜，检测为255
/// <param name="feature"></param>特征的阈值
/// <param name="is_clockwise_direction"></param>是否顺时针单调递增
/// <returns></returns>返回特征序列
vector<double> Feature_calculation( Mat scr,Mat mask, double feature, bool is_clockwise_direction) {
	
	vector<double> feature_result; 
	vector<Point2i> point_list;
	//point_list = Mat_to_pointlist(scr, mask); //point_cal
	point_list = point_cal(scr, mask);

	Point start= point_list[0], end= point_list[1];

	//if (start.x > end.x)
	//{
	//	Point swap = end;//用于交换start、end的中介变量
	//	end = start;
	//	start = swap;
	//}
	//if (!is_clockwise_direction) {
	//	int swap = end.y;//用于交换start、end的中介变量
	//	end.y = start.y;
	//	start.y = swap;
	//}
	Mat predictline;//理论直线
	get_predict_line(start, end, predictline, is_clockwise_direction);
	scr.convertTo(scr, CV_16SC1);
	predictline.convertTo(predictline, CV_16SC1);
	Mat error = scr - predictline;

	Mat mask_we(1, start.x + end.x, CV_8UC1, Scalar::all(0));
	mask_we(Rect(start.x, 0, end.x - start.x, 1)) = 255;//设定搜索区域
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

	

	feature_result.push_back(max - min);//待选特征1
	feature_result.push_back((max - min) / abs(end.y - start.y));//待选特征2
	feature_result.push_back((max - min) / abs(end.y - start.y) * sin_predict);//待选特征3
	feature_result.push_back((max - min) * sin_predict);//待选特征4
	feature_result.push_back(std);//待选特征5
	feature_result.push_back(m);//待选特征6
	feature_result.push_back(std_0);//待选特征6

	cout << " 差值：" << feature_result[0] << " 归一差值：" << feature_result[1] << " 归一法方向差值：" << feature_result[2] << " 法方向差值：" << feature_result[3]<< " 总方差：" << feature_result[4]<<" 均值：" << feature_result[5]<<" 均值：" << feature_result[6] << endl;

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
	//获取点斜式的点和斜率
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];

	double k = line_para[1] / line_para[0];

	//计算直线的端点(y = k(x - x0) + y0)
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
/// 单个边理论直线起始点计算
/// </summary>
/// <param name="scr"></param>实际边精确距离
/// <param name="mask"></param>掩膜，确定那块要检测那块不用检测
/// <returns></returns>返回一个点的泛型，第一位是起始点，第二位是终止点
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
/// 单个边变形判断函数
/// </summary>
/// <param name="scr"></param>原始精确距离
/// <param name="mask"></param>掩膜
/// <param name="feature_threshold"></param>特征阈值
/// <param name="result"></param>-
/// <param name="is_clockwise_direction"></param>是否顺时针单调递增
/// <returns></returns>此边是否变形
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
/// 单个角变形判断函数
/// </summary>
/// <param name="scr"></param>角的裁剪灰度图
/// <param name="mask"></param>掩膜
/// <param name="feature_threshold"></param>特征阈值
bool R_Deformation_judgment(Mat src,  double feature_threshold) {
	Mat background = src.clone();//背景
	Mat error = src.clone();//差分结果

	//空洞预处理 针对于主相机
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