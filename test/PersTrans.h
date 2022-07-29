#pragma once
/*=========================================================
* 文 件 名：PersTrans.h
* 功能描述：透视变换算法头文件（包括主，前后，左右相机）
=========================================================*/

#ifndef PERSTRANS_H
#define PERSTRANS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

Mat toushi_white(Mat image, Mat M, int border, int length, int width);
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat* Mwhite, Mat* Mbiankuang, Mat* M_white_abshow, int ID, String ScreenType_Flag, int leftRightWhiteFlag);
bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat* Mwhite, Mat* M_R_1_E, String ScreenType_Flag, int leftRightWhiteFlag, int border_white);
bool f_FrontBackCam_PersTransMatCal(InputArray _src, Mat* Mwhite, String ScreenType_Flag);
void convexSetPretreatment(Mat& src);
bool get_feitu_area(Mat scr, Mat& feitu, Rect& feitu_rect, double feitu_radio, int feitu_lowerArea, int feitu_higtherArea);
bool find_camera_roi(Mat src, Rect& roi_rect, float roi_radio, int roi_lowerArea, int roi_expand);
void make_screen_border(Mat scr, Mat& dst, int border_size, Rect& shap_rect);

vector<vector<int>> Mat_to_vector(Mat scr);
bool is_deformation(vector<vector<int>> scr_vec, int length, int border);

bool is_deformation_A(Mat scr, Mat big_scr, Mat mask, Rect rect, int boder, int size, int radio);
vector<Point2i> get_acqureline(Mat scr, Rect rect, int max, int min);
void predict_edg(Mat& dst, vector<int>& value, bool is_reverse);
bool is_deformation(Mat scr, int length, int border);
bool is_deformation(Mat scr_a, Mat scr_b, int border);
vector<Point2i> get_acqureline(Mat scr, Rect rect, int max, int min, int border);
void get_predict_line(int border, int rows, int cols, Point start, Point end, Mat& dst_a, Mat& dst_b);
vector<Point2i> get_predict_line(Point start, Point end, Mat& predictline, bool is_clockwise_direction);
vector<double> Feature_calculation(Point start, Point end, Mat scr, double feature, bool is_clockwise_direction);
bool get_acqureline(Mat scr, int max, int min, int border, string& result, bool is_clockwise_direction);
vector<Point2i> Mat_to_pointlist(Mat scr, Mat mask);
vector<double> Feature_calculation(Mat scr, Mat mask, double feature, bool is_clockwise_direction);
vector<Point2i> point_cal(Mat scr, Mat mask);
bool Deformation_judgment(Mat scr, Mat mask, double feature, string& result, bool is_clockwise_direction);
Mat reshape(Mat cameraMatrix, Mat distCoeffs, Mat map1, Mat map2, Mat frame, Mat frameCalibration);
bool R_Deformation_judgment(Mat scr, double feature_threshold);
#endif
