#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <io.h>
#include <time.h>
#include <string.h>
#include "PersTrans.h"
#include <map>
#include "direct.h"

#define BatchOperate
bool Dead_light0(Mat white, Mat ceguang, Mat* mresult, String* causecolor);
Mat max_fliter(Mat scr, int size);
Mat mid_adaptive_threshold(Mat scr, int size, int mid_off_set, int method, int maxmin_remove,int siqu);
bool ForeignBodyDeep(Mat white_yiwu, Mat ceguang, Mat Original, Mat* mresult, String* causecolor);//灰度检测S
bool Frame(Mat src_img, Mat ceguang, Mat* mresult, String* causecolor);

//#define LpPath
void sharpen2D(const Mat& image, Mat& result, int size, bool five);
void adaptiveThresholdCustom_whitedot(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double ratio);
Mat slipe_threshold(Mat scr, int row_size, int col_size, float ratio, float c);

Mat image_make_border(cv::Mat& src);
Mat frequency_filter(Mat& scr, Mat& blur);
Mat ideal_low_kernel(Mat& scr, float sigma);
Mat ideal_low_pass_filter(Mat& src, float sigma);
Mat butterworth_low_kernel(Mat& scr, float sigma, int n);
Mat butterworth_low_paass_filter(Mat& src, float d0, int n);
Mat gaussian_low_pass_kernel(Mat scr, float sigma);
Mat gaussian_low_pass_filter(Mat& src, float d0);
Mat ideal_high_kernel(Mat& scr, float sigma);
Mat ideal_high_pass_filter(Mat& src, float sigma);
Mat butterworth_high_kernel(Mat& scr, float sigma, int n);
Mat butterworth_high_paass_filter(Mat& src, float d0, int n);
Mat gaussian_high_pass_kernel(Mat scr, float sigma);
Mat gaussian_high_pass_filter(Mat& src, float d0);

//锐化，size奇数
void sharpen2D(const Mat& image, Mat& result, int size, bool five)
{
	// 首先构造一个内核
	Mat kernel(size, size, CV_32F, Scalar(0));
	/// 对 对应内核进行赋值
	///
	///
	if (five) {
		for (int i = 0; i < size; i++)
		{
			kernel.at<float>((size - 1) / 2, i) = -1.0;
			kernel.at<float>(i, (size - 1) / 2) = -1.0;
		}

		kernel.at<float>((size - 1) / 2, (size - 1) / 2) = 2 * size - 1;
	}
	else {
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				kernel.at<float>(i, j) = -1.0;

			}
		}

		kernel.at<float>((size - 1) / 2, (size - 1) / 2) = size * size - 1;
	}
	/// 对图像进行滤波操作
	filter2D(image, result, image.depth(), kernel);
}

using namespace cv;
using namespace std;


#define InputSampleNum             6
#define PlaceNum				   3
#define GroudingNum				   3

#define whitePoint_step          37       //白点自适应分割领域块大小
#define whitePoint_threshold     -3  //白点自适应分割领阈值
#define whitePoint_lowerArea     4  //面积下限
#define whitePoint_higherArea    600 //面积上限
#define scratchth                2.5             //白点排除划痕阈值大于则认为贴膜划痕
#define bubbleth                 19.5          //白点排除气泡阈值大于则认为贴膜气泡
#define dotdomainnum             5    //白点连通域个数大于则认为噪点
#define defectouterth            6.4   //白点缺陷处灰度均值与缺陷外围灰度差
#define corewholeth              99    //白点缺陷中心点灰度值与整块疑似缺陷灰度均值差
#define spotpeak                 6  //白点缺陷最亮点与最暗点灰度差
#define siecevariance            5.5   //侧光图中排除划痕参数
#define whitePoint_w_h           6.65 //白点缺陷长宽比阈值


#define scratch__bolckSize      57     //划伤自适应分割领域块大小
#define scratch__delta          -3.95        //划伤自适应分割领阈值
#define scratch_higherArea      8000     //划伤面积上限
#define scratch_lowerArea       200      //划伤面积下限
#define scratch_lowerLongShort  1  //划伤长宽比下限
#define scratch_higherLongShort 13//划伤长宽比上限
#define scratch_lowerWidth      9      //划伤宽度下限
#define scratch_higherWidth     28    //划伤宽度上限
#define scratch_lowerLength     40    //划伤长度下限
#define scratch_higherLength    400   //划伤长度上限限
#define filmscratch             30    //贴膜划痕系数//0.8
#define scratchbubbleth         20    //贴膜排除气泡系数1
#define scratchbubbleth1        20    //贴膜排除气泡系数2
#define scratchbrighth          4    //划伤亮度阈值
#define minscratchbrighth       2     //最小划伤亮度阈值


void batchOperate(string path, string defectName);
void algorithmTest(string fileName, string path, string defectName, ushort& sampleNum, char sampleNumBuf[]);
string reName(struct _finddata_t& fileinfo, string path, string defectName, ushort& sampleNum, char sampleNumBuf[]);

bool defectDection(int mode, map<string, Mat> inputSample, Mat* mresult, String* causecolor, String indexBuf);
bool ForeignBody(Mat src_white, Mat white_yiwu, Mat ceguang, Mat* mresult, String* causecolor);
bool WhiteDotLeft(Mat white_yiwu, Mat ceguang, Mat Original, Mat* mresult, String* causecolor);//灰度检测  Mat white_middle
bool boom_light(Mat white, Mat* mresult, String* causecolor);
bool Brightedge(Mat src_white, Mat photomainwhite, Mat* Mwhite, String* causecolor);
bool Dead_light(Mat white, Mat* mresult, String* causecolor);
bool Shifting(Mat white, Mat* mresult, String* causecolor, int num, Mat& left_white, Mat& right_white);
bool Mura_Decter(Mat imageGray, Mat* mresult, String* causecolor);
bool Scratch(Mat white, Mat ceguang, Mat* mresult, String* causecolor);
bool LpParamStandardize(Mat& white_Main1, Mat& ceL1, Mat& ceR1, Mat& SideLight_Main,  Mat& LeftCeGuang, Mat& RightCeGuang, String indexbuf);
void adaptiveThresholdCustom(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double ratio);
Mat Gabor7(Mat img_1);
bool compareContourAreas(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2);
void adaptiveThreshold1(InputArray _src, OutputArray _dst, double maxValue, int method, int type, int blockSize, double delta, int tianchong);
#ifdef BatchOperate
//string sampleLibPath = "C:\\Users\\wsc\\Desktop\\20200927165\\0307";// "D:\\graduateStudent\\Project\\背光源项目相关\\背光源样本\\样本库";
//string sampleLibPath = "E:\\项目\\背光源\\良品\\良品";//E:\项目\背光源\死灯
//string sampleLibPath = "E:\\项目\\背光源\\白点漏检\\矩形误检";//E:\项目\背光源\死灯
//string sampleLibPath = "E:\\项目\\背光源\\死灯";//\\1

//string sampleLibPath = "E:\\项目\\背光源\\死灯\\矩形死灯";//\\1
//string sampleLibPath = "E:\\项目\\背光源\\死灯\\1";//\\1
//string sampleLibPath = "E:\\beiguang\\white_point1";
//string sampleLibPath = "E:\\项目\\背光源\\白点漏检\\误检";
//string sampleLibPath = "E:\\项目\\背光源\\死灯\\误检";
string sampleLibPath = "E:\\项目\\背光源\\死灯";
//string sampleLibPath = "E:\\项目\\背光源\\变形";

string fileDic;

std::ofstream outFile;    //用于统计批量测试数据
#else
                                    
	#ifdef LpPath
	std::string SRC_PATH = "C:\\Users\\wsc\\Desktop\\20200927165\\0317yiweiwujian\\1674\\01_20200927165_1674_";
	#else
	std::string SRC_PATH = "E:\\项目\\背光源\\xianyi\\xianyi1";
	#endif // LpPath
#endif
	int main()
	{
#ifdef BatchOperate
		//批量测试类型
		string defectName = "SD";
		//合成对应类型的路径
		string path = sampleLibPath + "\\" + defectName;

		//若测试为良品类型，则建立.csv文件，用于批量计算当前目录下良品的灰度均值等参数
		if (defectName == "LP") {
			//获取当前时间，从1970年1月1日0点开始，此处使用time.h中的函数
			time_t now = time(0);
			tm tmPtr;
			localtime_s(&tmPtr, &now);

			//获取时间字符串，用于生成文件名
			const int timeBufLength = 30;
			char timeBuf[timeBufLength];
			sprintf_s(timeBuf, timeBufLength, "%04u-%02u-%02u %02u-%02u-%02u", tmPtr.tm_year + 1900,
				tmPtr.tm_mon + 1, tmPtr.tm_mday, tmPtr.tm_hour, tmPtr.tm_min, tmPtr.tm_sec);

			string outPath = sampleLibPath + "\\" + defectName + "\\" + string(timeBuf) + "_data.csv";
			outFile.open(outPath, ios::out); // 打开模式可省略

			if (outFile.is_open())
			{
				outFile << "样本索引, 整体灰度均值, 整体灰度均方差, 灯头灰度均值, 灯头灰度均方差" << endl;
				batchOperate(path, defectName);

				outFile.close();
			}
			else
			{
				cout << "文件创建失败！";
				return 0;
			}
		}
		else {
			fileDic = sampleLibPath + "\\" + defectName + "TestRes\\";

			//若目录不存在，则创建目录
			if (0 != _access(fileDic.c_str(), 00))
			{
				// if this folder not exist, create a new one.
				_mkdir(fileDic.c_str());   // 返回 0 表示创建成功，-1 表示失败
			}
			else //删除目录中所有文件
			{
				long long hFile;
				struct _finddata_t fileinfo;
				if ((hFile = _findfirst(fileDic.append("*").c_str(), &fileinfo)) != -1)
				{
					do
					{
						//如果是目录,迭代之
						//如果不是,加入列表
						if (!(fileinfo.attrib & _A_SUBDIR))
						{
							// 删除多余文件
							remove((fileDic + string(fileinfo.name)).c_str());
						}

					} while (_findnext(hFile, &fileinfo) == 0);
					_findclose(hFile);
				}
			}


			batchOperate(path, defectName);
		}
#else
		map<string, Mat> inputSample;
		//缺陷标定图
		Mat mresult;
		//缺陷结果字符串
		String causecolor;

		//defectDection(6, inputSample, &mresult, &causecolor);
		bool Res[8] = { true, true, true, true, true, true, true, true };
		for (int i = 5; i < 8; i++)
		{
			Res[i] = defectDection(i + 1, inputSample, &mresult, &causecolor, "001");
		}
		for (int i = 0; i < 8; i++)
		{
			std::cout << Res[i];
		}
#endif 
		system("pause");
	}

#ifdef BatchOperate
	//批量测试
	void batchOperate(string path, string defectName)
	{
		//文件句柄
		long long hFile = 0;
		//文件信息
		struct _finddata_t fileinfo;
		//路径字符串
		string p;

		//当前样本序号
		static ushort sampleNum = 0;
		//样本序号字符串长度
		const int sampleNumBufCnt = 4;
		//样本序号字符串
		char sampleNumBuf[sampleNumBufCnt] = "001";


		if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		{
			do
			{
				//如果是目录,迭代之
				//如果不是,加入列表
				if ((fileinfo.attrib & _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
						batchOperate(p.assign(path).append("\\").append(fileinfo.name), defectName);
				}
				else {
					string fileName = reName(fileinfo, path, defectName, sampleNum, sampleNumBuf);


					algorithmTest(fileName, path, defectName, sampleNum, sampleNumBuf);
				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}
	}
	//算法测试
	void algorithmTest(string fileName, string path, string defectName, ushort& sampleNum, char sampleNumBuf[])
	{
		const string sampleStrs[InputSampleNum]
			= { "_212.", "_112.", "_012.", "_210.", "_110.", "_010." };

		const string placeGroundingStrs[InputSampleNum]
			= { "M_W_", "L_W_", "R_W_", "M_C_", "L_C_", "R_C_" };

		static map<string, Mat> inputSample;
		//缺陷模式标识
		map<string, int> defectMode;//1-爆灯,2-亮边,3-白点,4-死灯,5-移位,6-白印,7-背光异物,8-划伤
		defectMode.insert(pair<string, int>("BL", 1));
		defectMode.insert(pair<string, int>("LB", 2));
		defectMode.insert(pair<string, int>("BD", 3));
		defectMode.insert(pair<string, int>("SD", 4));
		defectMode.insert(pair<string, int>("YW", 5));
		defectMode.insert(pair<string, int>("BY", 6));
		defectMode.insert(pair<string, int>("BG", 7));
		defectMode.insert(pair<string, int>("HS", 8));
		defectMode.insert(pair<string, int>("LP", 9));
		defectMode.insert(pair<string, int>("BX", 10));

		//缺陷标定图
		Mat mresult;
		//缺陷结果字符串
		String causecolor;
		//缺陷返回结果
		bool result;
		//保存文件名
		string saveRes;
		//样本名已经修改 当前类型命名如带有"SD"的样本才可测试SD算法，或者带有"LP"样本测试是否漏检
		if (fileName != "NULL" && strstr(fileName.c_str(), defectName.c_str()) != NULL || strstr(fileName.c_str(), "LP") != NULL)
		{
			//循环对比该文件是否是标准名的文件
			for (int i = 0; i < InputSampleNum; i++)
			{
				//若是标准文件则保存在map集合中
				if (strstr(fileName.c_str(), placeGroundingStrs[i].c_str()) != NULL)
				{
					//当前路径
					string curPath;
					Mat tempMat = imread(curPath.assign(path).append("\\").append(fileName.c_str()), -1);
					inputSample.insert(pair<string, Mat>(placeGroundingStrs[i], tempMat));
					break;
				}
			}
			if (sampleNum % InputSampleNum == 0)
			{
				sprintf_s(sampleNumBuf, 4, "%03d", sampleNum / InputSampleNum);
				//缺陷检测
				result = defectDection(defectMode[defectName], inputSample, &mresult, &causecolor, sampleNumBuf);

				std::cout << "第" + string((sampleNumBuf)) + "组" + defectName + "测试结果：" + (result ? "true" : "false") << endl;

				//清空map中图片
				inputSample.erase(inputSample.begin(), inputSample.end());

				//非良品目录测试，则输出测试结果
				if (defectName != "LP")
				{
					saveRes.append(defectName).append(sampleNumBuf).append(result ? "_true" : "_false").append(".jpg");
					string outPath = fileDic + saveRes;
					//cv::imwrite(outPath, mresult);
				}
			}
		}
		//else
		//{
		//	throw new std::exception;
		//}
	}
	//样本名修改
	string reName(struct _finddata_t& fileinfo, string path, string defectName, ushort& sampleNum, char sampleNumBuf[]) {

		//改名目标字符串
		string destStr;
		//样本字符串数组
		const string sampleStrs[InputSampleNum]
			= { "_B0.", "_BL.", "_BR.", "_C0.", "_CL.", "_CR." };

		//第二种样本字符串
		const string sampleInputStrs[InputSampleNum]
			= { "src_white1.", "src_L1.", "src_R1.", "src_ceguang1.", "src_ceguang_left.", "src_ceguang_right." };

		//目标样本字符串
		const string placeGroundingStrs[InputSampleNum]
			= { "M_W_", "L_W_", "R_W_", "M_C_", "L_C_", "R_C_" };

		//是否重复修改
		bool repeatModify = true;

		if ((!repeatModify) && (strstr(fileinfo.name, defectName.c_str()) != NULL || strstr(fileinfo.name, "LP") != NULL))
		{ //样本名已修改，直接跳过
			sampleNum++;

		}
		else {
			//循环查询属于那种类型图片，作对应类型名字修改
			for (int i = 0; i < InputSampleNum; i++)
			{
				//若寻找到任意一种原文件名或修改后文件名，则改名
				bool res = (strstr(fileinfo.name, sampleStrs[i].c_str()) != NULL && strstr(fileinfo.name, "S") == NULL && strstr(fileinfo.name, "ROI") == NULL);
				bool res2 = strstr(fileinfo.name, "ROI") == NULL;
				/*	if (((strstr(fileinfo.name, sampleStrs[i].c_str()) != NULL && strstr(fileinfo.name, "S") == NULL && strstr(fileinfo.name, "ROI")) == NULL

						|| (strstr(fileinfo.name, sampleInputStrs[i].c_str())!=NULL)
						|| (strstr(fileinfo.name, placeGroundingStrs[i].c_str()) != NULL))&& (strstr(fileinfo.name, ".jpg") == NULL))*/
				if ((strstr(fileinfo.name, sampleStrs[i].c_str()) != NULL && strstr(fileinfo.name, "S") == NULL && strstr(fileinfo.name, "ROI") == NULL)
					|| (strstr(fileinfo.name, placeGroundingStrs[i].c_str()) != NULL)
					|| (strstr(fileinfo.name, sampleInputStrs[i].c_str()) != NULL)
					)
				{
					//样本个数字符串
					sprintf_s(sampleNumBuf, 4, "%03d", sampleNum / InputSampleNum + 1);
					sampleNum++;

					//更换的文件名
					string fileName = defectName.append("_").append(placeGroundingStrs[i]).append(sampleNumBuf).append("S.bmp");

					destStr = ""; //首先置为空
					destStr.assign(path).append("\\").append(fileName);
					if (0 == rename(path.append("\\").append(fileinfo.name).c_str(), destStr.c_str())) {

						//std::cout << fileName << "-->" << destStr << endl;
						return fileName;
					}
				}
			}
		}
		return "NULL";
	}
#endif // BatchOperate

	bool defectDection(int mode, map<string, Mat> inputSample, Mat* mresult, String* causecolor, String indexBuf) {
		//1-爆灯,2-亮边,3-白点,4-死灯,5-移位,6-白印,7-背光异物,8-划伤

		Mat Mwhite, Mblack, Mlouguang, Mabshow, Mceguang;
		bool result = false;
		String causeColor_1_white;
		Mat M_L_1, M_R_1, M_L_1_E, M_R_1_E;

		Mat src_White, ceguang, src_L1, src_R1, src_ceguang_left, src_ceguang_right;

		if (!inputSample["M_W_"].empty())
		{
			//主相机图片处理
			src_White = inputSample["M_W_"];
			ceguang = inputSample["M_C_"];
			//左右相机图片处理
			src_L1 = inputSample["L_W_"];
			src_R1 = inputSample["R_W_"];
			src_ceguang_left = inputSample["L_C_"];;
			src_ceguang_right = inputSample["R_C_"];;
		}
#ifndef BatchOperate
		else
		{
#ifdef LpPath

			////主相机图片处理
			//src_White = cv::imread(SRC_PATH + "B0.bmp", -1);
			//ceguang = cv::imread(SRC_PATH + "C0.bmp", -1);
			////左右相机图片处理
			//src_R1 = cv::imread(SRC_PATH + "BR.bmp", -1);
			//src_L1 = cv::imread(SRC_PATH + "BL.bmp", -1);

			//src_ceguang_right = cv::imread(SRC_PATH + "CR.bmp", -1);
			//src_ceguang_left = cv::imread(SRC_PATH + "CL.bmp", -1);
			//主相机图片处理
			src_White = cv::imread(SRC_PATH + "212.bmp", -1);
			ceguang = cv::imread(SRC_PATH + "210.bmp", -1);
			//左右相机图片处理
			src_R1 = cv::imread(SRC_PATH + "112.bmp", -1);
			src_L1 = cv::imread(SRC_PATH + "012.bmp", -1);

			src_ceguang_right = cv::imread(SRC_PATH + "112.bmp", -1);
			src_ceguang_left = cv::imread(SRC_PATH + "010.bmp", -1);
#else

			////主相机图片处理
			//src_White = cv::imread(SRC_PATH + "\\src_white1.bmp", -1);
			//ceguang = cv::imread(SRC_PATH + "\\src_ceguang1.bmp", -1);
			////左右相机图片处理
			//src_L1 = cv::imread(SRC_PATH + "\\src_L1.bmp", -1);
			//src_R1 = cv::imread(SRC_PATH + "\\src_R1.bmp", -1);

			//src_ceguang_right = cv::imread(SRC_PATH + "\\src_ceguang_right.bmp", -1);
			//src_ceguang_left = cv::imread(SRC_PATH + "\\src_ceguang_left.bmp", -1);

					//主相机图片处理
			src_White = cv::imread(SRC_PATH + "\\mainwhite.bmp", -1);
			ceguang = cv::imread(SRC_PATH + "\\mainceguang.bmp", -1);
			//左右相机图片处理
			src_L1 = cv::imread(SRC_PATH + "\\leftwhite.bmp", -1);
			src_R1 = cv::imread(SRC_PATH + "\\rightwhite.bmp", -1);

			src_ceguang_right = cv::imread(SRC_PATH + "\\leftceguang.bmp", -1);
			src_ceguang_left = cv::imread(SRC_PATH + "\\rightceguang.bmp", -1);
#endif // LpPath
		}
#endif // !BatchOperate

		if (src_White.channels() == 3)
			cvtColor(src_White, src_White, CV_BGR2GRAY);
		if (ceguang.channels() == 3)
			cvtColor(ceguang, ceguang, CV_BGR2GRAY);

		if (src_ceguang_right.channels() == 3)
			cvtColor(src_ceguang_right, src_ceguang_right, CV_BGR2GRAY);
		if (src_ceguang_left.channels() == 3)
			cvtColor(src_ceguang_left, src_ceguang_left, CV_BGR2GRAY);
		//主黑白相机处理
		if (src_L1.channels() == 3)
			cvtColor(src_L1, src_L1, CV_BGR2GRAY);
		if (src_R1.channels() == 3)
			cvtColor(src_R1, src_R1, CV_BGR2GRAY);

		try
		{
			double meanV = mean(src_White)[0];
		}
		catch (const std::exception& e)
		{
			std::cout << "测试·异常 ++++++++++";
		}

		//白底透视变换
		bool Ext_Result_BlackWhite = f_MainCam_PersTransMatCal(src_White, 0, 70, &Mwhite, &Mblack, &Mabshow, 1, "R角水滴屏", -1);
		double meanV1 = mean(src_White)[0];


		//主相机白底图
		Mat white_Main = toushi_white(src_White, Mwhite, -1, 3000, 1500);
		double meanV2 = mean(white_Main)[0];
		//主相机侧光图
		Mat SideLight_Main = toushi_white(ceguang, Mwhite, -1, 3000, 1500);
		double meanV3 = mean(src_White)[0];
		//主相机gabor滤波
		Mat mainfilter = Gabor7(white_Main);           //滤波去除水平和竖直方向的纹理
		//Mat mainfilter = white_Main;           //滤波去除水平和竖直方向的纹理
		double meanVal = mean(mainfilter)[0];


		bool Ext_Result_Left = f_LeftRightCam_PersTransMatCal(src_L1, &M_L_1, &M_L_1_E, "R角水滴屏", 1, 15);
		bool Ext_Result_Right = f_LeftRightCam_PersTransMatCal(src_R1, &M_R_1, &M_R_1_E, "R角水滴屏", 1, 15);
		//左右相机透视变换图
		Mat ceL1 = toushi_white(src_L1, M_L_1_E, -5, 3000, 1500);
		Mat ceR1 = toushi_white(src_R1, M_R_1_E, -5, 3000, 1500);

		Mat LeftCeGuang = toushi_white(src_ceguang_left, M_L_1_E, -5, 3000, 1500);      //左相机侧光校正图
		Mat RightCeGuang = toushi_white(src_ceguang_right, M_R_1_E, -5, 3000, 1500);    //右相机侧光校正图

		Mat src_L1_gray, src_R1_gray;
		src_L1_gray = src_L1.clone();
		src_R1_gray = src_R1.clone();
		Mat th2, th3;
		threshold(src_L1_gray, th2, 20, 255, CV_THRESH_BINARY);
		threshold(src_R1_gray, th3, 20, 255, CV_THRESH_BINARY);
		Mat left_mask = toushi_white(th2, M_L_1, -1, 3000, 1500);
		Mat right_mask = toushi_white(th3, M_R_1, -1, 3000, 1500);

		bitwise_and(left_mask, LeftCeGuang, LeftCeGuang);
		bitwise_and(right_mask, RightCeGuang, RightCeGuang);
		Mat leftfilter = Gabor7(ceL1);       //左侧白底滤波
		Mat rightfilter = Gabor7(ceR1);     //右侧白底滤波

		switch (mode) {
		case 1:

			result = boom_light(mainfilter, mresult, &causeColor_1_white);
			break;
		case 2:
			result = Brightedge(mainfilter, src_White, mresult, &causeColor_1_white);
			break;
		case 3:
			result = WhiteDotLeft(mainfilter, SideLight_Main, white_Main, mresult, &causeColor_1_white);//主相机白点检测
			break;
		case 4:
			//result = Dead_light0(mainfilter, SideLight_Main, mresult, &causeColor_1_white);
			//Dead_light(mainfilter, mresult, &causeColor_1_white);
			result=Frame(src_White, ceguang, mresult, causecolor);
			break;
		case 5:
			//result = Shifting(leftfilter, mresult, &causeColor_1_white, 1, left_mask, right_mask, src_L1);
			if (result == false) {
				//result = Shifting(rightfilter, mresult, &causeColor_1_white, 1, left_mask, right_mask, src_R1);
			}
			break;
		case 6:
			result = Mura_Decter(mainfilter, mresult, &causeColor_1_white);//主相机白印检测
			break;
		case 7:
			//result = ForeignBody(mainfilter, SideLight_Main, white_Main, mresult, &causeColor_1_white);//ForeignBodyDeep
			result = ForeignBodyDeep(mainfilter, SideLight_Main, white_Main, mresult, &causeColor_1_white);//ForeignBodyDeep
			break;
		case 8:
			result = Scratch(leftfilter, LeftCeGuang, mresult, &causeColor_1_white);
			if (!result)
			{
				result = Scratch(rightfilter, RightCeGuang, mresult, &causeColor_1_white);
			}
			break;
			//良品处理程序，用于降低算法的误检率
		case 9:
			result = LpParamStandardize(white_Main, ceL1, ceR1, SideLight_Main, LeftCeGuang, RightCeGuang, indexBuf);
			break;
		case 10:
			result = Frame(src_White, SideLight_Main,mresult,  causecolor);
			;
			break;

		}
		return result;
	}
	//良品参数标准化函数
	bool LpParamStandardize(Mat& white_Main1, Mat& ceL1, Mat& ceR1, Mat& SideLight_Main, Mat& LeftCeGuang, Mat& RightCeGuang, String indexbuf) {


		double m = 0.0;
		double std = 0.0;
		double lightM = 0.0;
		double lightStd = 0.0;
		static double mSum = 0.0, stdSum = 0.0;
		static int sampleNum = 0;
		static int invaildNum = 0;
		if (mean(white_Main1)[0] > 0)
		{
			sampleNum++;
			vector<vector<Point>> contours;
			findContours(white_Main1, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			Mat mask, tem_m, tem_s;
			cv::threshold(white_Main1, mask, 10, 255, CV_THRESH_BINARY);
			cv::meanStdDev(white_Main1, tem_m, tem_s, mask);
			m = tem_m.at<double>(0, 0);
			std = tem_s.at<double>(0, 0);
			cout << "当前样本索引：" + indexbuf << endl;
			//cout << "全局特征：灰度均值2: " << m << endl;
			//cout << "全局特征：标准差2:   " << std << endl;
			//cout << "全局特征：变异系数2：" << std / m << endl;

			Mat whiteTemp = white_Main1.clone();
			Rect rect(0, 0, 100, 1500);
			Mat lightBorArea = white_Main1(rect);
			Mat lightBorMask;
			threshold(lightBorArea, lightBorMask, 15, 255, CV_THRESH_BINARY);
			cv::meanStdDev(lightBorArea, tem_m, tem_s, lightBorMask);
			lightM = tem_m.at<double>(0, 0);
			mSum += lightM;
			lightStd = tem_s.at<double>(0, 0);
			stdSum += lightStd;
			//cout << "灯条特征：灰度均值2:  " << m << endl;
			//cout << "灯条特征：标准差2:    " << std << endl;
#ifdef BatchOperate
			std::cout << outFile.is_open();
			outFile << to_string(sampleNum) + "," + to_string(m) + "," + to_string(std) + "," + to_string(lightM) + "," + to_string(lightStd) << endl;
#endif // BatchOperate



		}
		else
		{
			invaildNum++;

		}
		if (indexbuf == "060")
		{
			cout << "灯条特征：批次灰度总值:   " << mSum << endl;
			cout << "灯条特征：批次标准差总值: " << stdSum << endl;
			cout << "无效样本数：              " << invaildNum << endl;
		}
		return true;
	}

	/*====================================================================
	* 函 数 名: boom_light
	* 功能描述: 爆灯检测  屏幕左侧发光LED管损坏，呈现局部发亮
	* 输入：主相机白底图像
	* 输出：主相机白底下检测结果图和result
	* 其他：
	======================================================================*/
	bool boom_light(Mat white, Mat* mresult, String* causecolor)
	{
		int Boom_TopLength = 0;
		int Boom_BottomLength = 0;

		int boder = 5;
		int decter_length = 200;
		bool result = false;
		Mat img_gray = white.clone();

		Mat strong_result;
		Ptr<CLAHE> clahe = createCLAHE(5.0, Size(3, 3));
		clahe->apply(img_gray, strong_result);

		//Mat edge_img = strong_result(Rect(strong_result.cols - 100, 0, 100, strong_result.rows)).clone();
		Mat edge_img = strong_result(Rect(0, 0, decter_length, strong_result.rows)).clone();
		medianBlur(edge_img, edge_img, 3);
		//Mat ad_result;
		//adaptiveThreshold(edge_img, ad_result, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 15, 3);

		Mat edge_img_X;
		Mat edge_img_Y;
		Mat edge_img_result;
		Mat sobel_result;
		Mat structure_element = getStructuringElement(MORPH_RECT, Size(5, 5));
		Sobel(edge_img, edge_img_X, CV_16S, 0, 1, 3, 1, 2, BORDER_DEFAULT);
		convertScaleAbs(edge_img_X, edge_img_X);
		Sobel(edge_img, edge_img_Y, CV_16S, 1, 0, 3, 1, 2, BORDER_DEFAULT);
		convertScaleAbs(edge_img_Y, edge_img_Y);
		addWeighted(edge_img_X, 0.5, edge_img_Y, 0.5, 0, edge_img_result);
		clahe->apply(edge_img_result, edge_img_result);
		threshold(edge_img_result, sobel_result, 30, 255, CV_THRESH_BINARY);
		Mat F_result = ~sobel_result;
		Mat edge_thresold;
		erode(F_result, edge_thresold, structure_element);
		Mat th_result = Mat::zeros(img_gray.size(), img_gray.type());
		edge_thresold.copyTo(th_result(Rect(0, 0, decter_length, th_result.rows)));
		th_result(Rect(0, 0, 10, th_result.rows)) = uchar(0);            //屏蔽右侧10行，防止误检

		th_result(Rect(0, 0, Boom_TopLength, th_result.rows)) = uchar(0);
		th_result(Rect(th_result.cols - Boom_BottomLength, 0, Boom_BottomLength, th_result.rows)) = uchar(0);

		vector<vector<Point>> contours;
		findContours(th_result, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		std::sort(contours.begin(), contours.end(), compareContourAreas);

		vector<Rect> boundRect(contours.size());
		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			if (area > 200 && area < 150000)
			{
				Mat temp_mask = Mat::zeros(th_result.rows, th_result.cols, CV_8UC1);
				drawContours(temp_mask, contours, i, 255, FILLED, 8);
				boundRect[i] = boundingRect(Mat(contours[i]));
				float w = boundRect[i].width;
				float h = boundRect[i].height;
				RotatedRect rect = minAreaRect(contours[i]);  //包覆轮廓的最小斜矩形 划伤缺陷有旋转特点
				Point p = rect.center;
				double w1 = rect.size.height;
				double h1 = rect.size.width;
				int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
				int x_1 = X_1;//矩形左上角X坐标值
				int y_1 = Y_1;//矩形左上角Y坐标值
				int x_2 = X_2;//矩形右下角X坐标值
				int y_2 = Y_2;//矩形右下角Y坐标值
				int x_point = X_1 + round(w / 2);
				int y_point = Y_1 + round(h / 2);
				if (max(w / h, h / w) <= 14 && max(w, h) >= 10)//4
				{
					X_1 = X_1 - boder - int(w);
					Y_1 = Y_1 - boder - int(h);
					X_2 = X_2 + boder + int(w);
					Y_2 = Y_2 + boder + int(h);
					if (X_1 < 0)
					{
						X_1 = 0;
					}
					if (Y_1 < 0)
					{
						Y_1 = 0;
					}
					if (X_2 > decter_length - 1)
					{
						X_2 = decter_length - 1;
					}
					if (Y_2 > th_result.rows - 1)
					{
						Y_2 = th_result.rows - 1;
					}

					Mat imagedoubt = img_gray(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
					Mat mask = th_result(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
					double mean_in_gray = mean(imagedoubt, mask)[0];
					if (mean_in_gray <= 45)
						continue;
					Mat temp_mask1;
					Mat temp_mask2;
					threshold(imagedoubt, temp_mask1, 30, 255, CV_THRESH_BINARY);
					double mean_all = mean(imagedoubt, temp_mask1)[0];
					threshold(imagedoubt, temp_mask1, mean_all - 20, 255, CV_THRESH_BINARY);
					bitwise_and(temp_mask1, mask, temp_mask1);
					mean_in_gray = mean(imagedoubt, temp_mask1)[0];
					if (mean_in_gray <= 70)
					{
						continue;
					}
					threshold(imagedoubt, temp_mask1, 30, 255, CV_THRESH_BINARY);
					bitwise_and(temp_mask1, ~mask, temp_mask2);
					double mean_out_gray = mean(imagedoubt, temp_mask2)[0];
					double intensity = mean_in_gray - mean_out_gray;
					Mat TempImage_Binary;
					if (mean_in_gray >= 200 && mean_out_gray >= 200 && intensity <= 23 && intensity >= -10)
					{
						Mat tempImage = img_gray(Rect(img_gray.cols - 500, 0, 300, img_gray.rows)).clone();
						threshold(tempImage, TempImage_Binary, 30, 255, CV_THRESH_BINARY);
						mean_out_gray = mean(tempImage, TempImage_Binary)[0];
						intensity = mean_out_gray - mean_in_gray;
					}
					if (intensity > 23)
					{
						result = true;
						CvPoint top_lef4 = cvPoint(x_1, y_1);
						CvPoint bottom_right4 = cvPoint(x_2, y_2);
						rectangle(white, top_lef4, bottom_right4, Scalar(0), 5, 8, 0);
						break;
					}
				}
			}
		}
		if (result == true)
		{
			*mresult = white;
			*causecolor = "爆灯";
			result = true;
		}
		return result;

	}

	/*=========================================================
	* 函 数 名: Brightedge
	* 功能描述：亮边
	* 函数输入：
	* 备注说明：2021年5月11日修改
	 =========================================================*/
	bool Brightedge(Mat src_white, Mat photomainwhite, Mat* Mwhite, String* causecolor)
	{

		int Flag_L_R = 2;

		double enlarge_num = 1.12;    //刘海部分的拉伸
		int reduce_num = 40;  //往里缩的像素值
		int reduce_num_out = 20;//计算外围灰度时向外扩的像素行数

		double canny_low_limit = 8;//8
		//int reduce_num = 50;  //往里缩的像素值
		int Border_remove_highlimit = 1200; //去除边框的轮廓size大小
		int contours_min_limit1 = 50;  //边缘ROI检测的轮廓最小size
		int contours_min_limit2 = 30;  //候选框内检测的轮廓最小size
		int small_rect_filter = 25;//最小矩形框长宽限制
		int mean_value_defect = 160;//缺陷处的亮度最低阈值
		int mean_value_defect_sum_out = 33;//候选框内缺陷处与框内其他区域的灰度差值
		int mean_value_defect_sum_out_middle = 35;//中间部分灰度值修正
		int mean_value_defect_sum_rect_out = 15;//候选框内缺陷处与框外其他区域的灰度差值
		int mean_value_defect_sum_rect_out_nearlight = 25; //靠近灯部的候选框内缺陷处与框外其他区域的灰度差值

		Mat Mask_img;
		Mat img1;
		Mat M;
		vector<vector<Point>> contours;
		vector<vector<Point>> contours1;
		Mat img = src_white.clone();
		Mat src = src_white.clone();
		//Canny(img, img1, 20, 60, 3);
		//threshold(img, Mask_img, 20, 255, CV_THRESH_BINARY);
		adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 57, -3.95);
		findContours(img, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//clock_t start_time = clock();
		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			if (area > 100000)
			{
				drawContours(img, contours, i, Scalar(255), -1, 8);
				break;
			}
			else
				drawContours(img, contours, i, Scalar(0), -1, 8);
		}

		int y_part1 = int(img.rows / 5 - reduce_num);
		int y_part2 = y_part1 + reduce_num;
		int y_part3 = int(img.rows / 5 * 4 - reduce_num);
		int y_part4 = y_part3 + reduce_num;

		int x_part1 = int(img.cols / 2);
		int x_part2 = x_part1 + 2 * reduce_num;

		Mat desImg1(img.rows, img.cols, CV_8UC1, Scalar(0, 0, 0));  //创建和原图相同大小的黑色图片
		Mat desImg = desImg1(Rect(reduce_num, reduce_num, img.cols - 2 * reduce_num, img.rows - 2 * reduce_num)); //长和宽缩小固定像素后的区域

		//将原图分分为6部分
		Mat desImgROI1 = desImg(Rect(0, 0, x_part1, y_part1));
		Mat desImgROI2 = desImg(Rect(x_part1, 0, desImg.cols - x_part1, y_part1));
		Mat desImgROI3 = desImg(Rect(0, y_part1, x_part1, y_part3 - y_part2));
		Mat desImgROI4 = desImg(Rect(x_part1, y_part1, desImg.cols - x_part1, y_part3 - y_part2));
		Mat desImgROI5 = desImg(Rect(0, y_part1 + y_part3 - y_part2, x_part1, desImg.rows + 2 * reduce_num - y_part4));
		Mat desImgROI6 = desImg(Rect(x_part1, y_part1 + y_part3 - y_part2, desImg.cols - x_part1 - 1, desImg.rows + 2 * reduce_num - y_part4 - 1));

		Mat imgROI1 = img(Rect(0, 0, x_part1, y_part1));
		Mat imgROI2 = img(Rect(x_part2, 0, desImg.cols - x_part1, y_part1));
		Mat imgROI3 = img(Rect(0, y_part2, x_part1, y_part3 - y_part2));
		Mat imgROI4 = img(Rect(x_part2, y_part2, desImg.cols - x_part1, y_part3 - y_part2));
		Mat imgROI5 = img(Rect(0, y_part4, x_part1, desImg.rows + 2 * reduce_num - y_part4));
		Mat imgROI6 = img(Rect(x_part2, y_part4, desImg.cols - x_part1 - 1, desImg.rows + 2 * reduce_num - y_part4 - 1));

		//调整刘海部分边缘
		Mat imgROI3_Enlarge;
		resize(imgROI3, imgROI3_Enlarge, Size(imgROI3.cols, imgROI3.rows * enlarge_num), 0, 0, INTER_LINEAR);
		Mat imgROI3_Enlarge_ROI = imgROI3_Enlarge(Rect(0, imgROI3.rows * abs(1 - enlarge_num) / 2, imgROI3.cols, imgROI3.rows));

		Mat imgROI4_Enlarge;
		resize(imgROI4, imgROI4_Enlarge, Size(imgROI4.cols, imgROI4.rows * enlarge_num), 0, 0, INTER_LINEAR);
		Mat imgROI4_Enlarge_ROI = imgROI4_Enlarge(Rect(0, imgROI4.rows * abs(1 - enlarge_num) / 2, imgROI4.cols, imgROI4.rows));

		imgROI1.copyTo(desImgROI1);
		imgROI2.copyTo(desImgROI2);
		imgROI3_Enlarge_ROI.copyTo(desImgROI3);
		imgROI4_Enlarge_ROI.copyTo(desImgROI4);
		imgROI5.copyTo(desImgROI5);
		imgROI6.copyTo(desImgROI6);

		//得到边缘区域的掩膜
		Mat img2;
		bitwise_not(desImg1, desImg1);
		bitwise_and(img, desImg1, img2);

		//在原图上得到边缘区域
		Mat img3 = src_white.clone();
		Mat img4;
		bitwise_and(img3, img2, img4);

		double mean_s = mean(img4, img4)[0];
		Mat mask_mean4;
		threshold(img4, mask_mean4, mean_s, 255, CV_THRESH_BINARY);
		bitwise_and(img4, mask_mean4, img4);
		int start_row = 0;
		int end_row = 0;
		if (Flag_L_R == 0)//右相机屏蔽对应不聚焦区域
		{
			start_row = 70;
			end_row = img4.rows - 550;
			img4(Rect(0, img4.rows - 100, img4.cols - 1, 100)) = uchar(0);
		}
		else if (Flag_L_R == 1)//左相机屏蔽对应不聚焦区域
		{
			start_row = 550;
			end_row = img4.rows - 70;
			img4(Rect(0, 0, img4.cols - 1, 100)) = uchar(0);
		}

		Mat img_result = img4.clone();

		double m, n;
		Canny(img_result, img_result, canny_low_limit, canny_low_limit * 3, 3);
		findContours(img_result, contours, CV_RETR_TREE, CHAIN_APPROX_NONE);

		for (vector<int>::size_type i = 0; i < contours.size(); i++)  // 去掉边框的轮廓
		{
			if (contours[i].size() > Border_remove_highlimit)
			{
				drawContours(img_result, contours, i, Scalar(0), 2, 8);
			}
		}
		//clock_t end_time = clock();
		//double time_all = (end_time - start_time) / CLOCKS_PER_SEC * 1000;

		//去掉边框轮廓后再膨胀
		Mat element = getStructuringElement(MORPH_RECT, Size(6, 6));
		morphologyEx(img_result, img_result, CV_MOP_CLOSE, element);
		//dilate(img_result, img_result, element);

		findContours(img_result, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
		std::sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Rect> boundRect(contours.size());//Rect类型的vector容器boundRect存放正外接矩形
		vector<RotatedRect> roRect(contours.size());  //定义Rect类型的vector容器roRect存放最小外接矩形

		int num = 0;
		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() > contours_min_limit1)
			{
				boundRect[i] = boundingRect(Mat(contours[i]));
				int X_1 = boundRect[i].tl().x; //矩形左上角X坐标值
				int Y_1 = boundRect[i].tl().y; //矩形左上角Y坐标值
				int X_2 = boundRect[i].br().x; //矩形右下角X坐标值
				int Y_2 = boundRect[i].br().y; //矩形右下角Y坐标值
				Mat boundrect;
				Mat b_2;
				int b_x1;
				int b_y1;
				int b_wdith;
				int b_height;

				if ((X_2 - X_1) < small_rect_filter && (Y_2 - Y_1) < small_rect_filter)  //滤掉一些干扰点
					continue;


				//需要在侧光图上排掉是否是膜贴歪出现的边缘
				int Ce_x1;
				int Ce_y1;
				int Ce_wdith;
				int Ce_height;

				Ce_x1 = X_1;
				Ce_y1 = Y_1;
				Ce_wdith = X_2 - X_1;
				Ce_height = Y_2 - Y_1;

				int mean_value_sum_out = mean_value_defect_sum_rect_out;
				/*	if (X_1 < 0)
					{
						X_1 = 0;
					}
					if (Y_1 < 0)
					{
						Y_1 = 0;
					}*/

				if (X_1 < 200)  //头部
				{
					//Ce_x1 = Ce_x1 + reduce_num_out;
					Ce_x1 = Ce_x1 - reduce_num_out;
					b_x1 = 0;
					b_y1 = Y_1;
					b_wdith = X_2;
					b_height = Y_2 - Y_1;
				}
				else if (X_1 > 2800) //尾部
				{
					Ce_x1 = Ce_x1 - reduce_num_out;
					b_x1 = X_1;
					b_y1 = Y_1;
					b_wdith = img4.cols - 1 - X_1;
					b_height = Y_2 - Y_1;
				}
				else  //两侧
				{
					if (X_1 > 2500 && X_1 < 2800)
						mean_value_sum_out = mean_value_defect_sum_rect_out_nearlight;
					if (Flag_L_R == 0)
					{
						Ce_y1 = Ce_y1 + reduce_num_out;
						b_x1 = X_1;
						b_y1 = 0;
						b_wdith = X_2 - X_1;
						b_height = Y_2;
					}
					else
					{
						Ce_y1 = Ce_y1 - reduce_num_out;
						b_x1 = X_1;
						b_y1 = Y_1;
						b_wdith = X_2 - X_1;
						b_height = img4.rows - Y_1 - 1;
					}
				}

				boundrect = img4(Rect(b_x1, b_y1, b_wdith, b_height));
				Mat b_1 = boundrect.clone();
				b_2 = boundrect.clone();
				Mat b_3 = boundrect.clone();

				if (Ce_x1 < 0)
				{
					Ce_x1 = 0;
				}
				if (Ce_y1 < 0)
				{
					Ce_y1 = 0;
				}

				Mat img_out = src_white(Rect(Ce_x1, Ce_y1, Ce_wdith, Ce_height));
				Mat img_out_mask = img(Rect(Ce_x1, Ce_y1, Ce_wdith, Ce_height));
				double mean_out = mean(img_out, img_out_mask)[0];

				//查找候选框内的轮廓形状
				m = mean(b_1, b_1)[0];  //计算候选框的灰度均值
				threshold(b_1, b_1, m, 255, CV_THRESH_BINARY);  //用该灰度均值为阈值进行二值化

				//求取割出的轮廓之外的灰度均值
				threshold(b_3, b_3, m, 255, CV_THRESH_TOZERO_INV);  //计算除分割出的轮廓之外的其余部分的灰度均值CV_THRESH_BINARY  CV_THRESH_TOZERO_INV
				n = mean(b_3, b_3)[0];
				threshold(b_3, b_3, n, 255, CV_THRESH_BINARY);
				double m_anti = mean(b_2, b_3)[0];
				if (m_anti < 100) {
					m_anti = n + 3;
				}

				findContours(b_1, contours1, CV_RETR_LIST, CHAIN_APPROX_NONE);
				std::sort(contours1.begin(), contours1.end(), compareContourAreas);
				vector<Rect> boundRect1(contours1.size());//Rect类型的vector容器boundRect存放正外接矩形
				for (vector<int>::size_type i1 = 0; i1 < contours1.size(); i1++)
				{
					if (contours1[i1].size() > contours_min_limit2)
					{
						boundRect1[i1] = boundingRect(Mat(contours1[i1]));
						int X_1_lo = boundRect1[i1].tl().x; //矩形左上角X坐标值
						int Y_1_lo = boundRect1[i1].tl().y; //矩形左上角Y坐标值
						int X_2_lo = boundRect1[i1].br().x; //矩形右下角X坐标值
						int Y_2_lo = boundRect1[i1].br().y; //矩形右下角Y坐标值

						Mat boundrect_local = b_2(Rect(X_1_lo, Y_1_lo, X_2_lo - X_1_lo, Y_2_lo - Y_1_lo));
						Mat mask_lo = b_1(Rect(X_1_lo, Y_1_lo, X_2_lo - X_1_lo, Y_2_lo - Y_1_lo));
						double m_lo = mean(boundrect_local, mask_lo)[0];
						//if (m_lo > mean_value_defect && (m_lo - m_anti) > mean_value_defect_sum_out && (m_lo - mean_out) > mean_value_sum_out)
					/*	if (m_lo > mean_value_defect && (m_lo - m_anti) > mean_value_defect_sum_out)
						{
							num++;
							rectangle(src, boundRect[i].tl(), boundRect[i].br(), Scalar(255), 5, 8, 0);
							break;
						}*/
						if ((X_1 > 1000 && X_1 < 2000 && Y_1 < 50) || (X_1 > 1000 && X_1 < 2000 && Y_1 >(img.rows - 50)))
						{
							if (m_lo > mean_value_defect && (m_lo - m_anti) > mean_value_defect_sum_out_middle || m_lo > 230)
							{
								num++;
								rectangle(src, boundRect[i].tl(), boundRect[i].br(), Scalar(255), 5, 8, 0);
								break;
							}
						}
						else
						{
							if (m_lo > mean_value_defect && (m_lo - m_anti) > mean_value_defect_sum_out || m_lo > 230)
							{
								num++;
								rectangle(src, boundRect[i].tl(), boundRect[i].br(), Scalar(255), 5, 8, 0);
								break;
							}
						}
					}
				}

				if (num > 0)
				{
					break;
				}
			}

		}
		if (num == 0)
		{
			//获取原图
			Mat yuantu = photomainwhite.clone();
			//获取自适应二值化图
			adaptiveThreshold(yuantu, yuantu, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 57, -3.95);
			//寻找其轮廓
			findContours(yuantu, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			//clock_t start_time = clock();
			//获取其最大最外围轮廓绘制到原图上
			for (vector<int>::size_type i = 0; i < contours.size(); i++)
			{
				//
				double area = contourArea(contours[i]);
				if (area > 100000)
				{
					drawContours(yuantu, contours, i, Scalar(0), -1, 8);
					break;
				}
			}
			findContours(yuantu, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			for (vector<int>::size_type i = 0; i < contours.size(); i++)
			{
				double area = contourArea(contours[i]);
				if (area > 100)
				{
					num++;
					break;
				}
			}
		}
		if (num != 0)
		{
			*Mwhite = src;
			*causecolor = "亮边";

			//cout << "Running time is: " << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
			return true;

		}

		else
			return false;
	}

	/*=========================================================
	 * 函 数 名: WhiteDotLeft
	 * 功能描述: 白色底下间检测圆形白点状缺陷,白点缺陷在主黑白相机中不可见
	 * 输入: 左右侧相机拍摄白点图片和对应侧光图片，侧光图片排除划痕和气泡
	 * 输出：白底下检测结果图和result
	 * 修改时间：2020年11月10日
	 * 其他：
	 =========================================================*/
	bool WhiteDotLeft(Mat white_yiwu, Mat ceguang, Mat Original, Mat* mresult, String* causecolor)//灰度检测  Mat white_middle
	{
		bool result = false;
		Mat img_gray = Original.clone();
		Mat img_ceguang = ceguang.clone();
		//    Mat gaussion_low = gaussian_low_pass_filter(img_gray, 25);
		//    imwrite("D:gaussion_low.bmp",gaussion_low);
		//    sharpen2D(gaussion_low, gaussion_low, 5,false);

		//    adaptiveThreshold(gaussion_low, gaussion_low, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 39, -3);
		//    imwrite("D:gaussion_lowadaptiveThreshold.bmp",gaussion_low);


		medianBlur(img_gray, img_gray, 3); //中值滤波滤除椒盐噪声,缺点耗时26毫秒 奇数半径越大效果越强
		//sharpen2D(img_gray, img_gray, 3,false);
		//imwrite("D:sharp.bmp",img_gray);

		//imshow("mid", img_gray);

		//medianBlur(img_gray, img_gray, 3); //中值滤波滤除椒盐噪声,缺点耗时26毫秒 奇数半径越大效果越强
		//6.11屏蔽最大值滤波
		//Mat maximg = max_fliter(img_gray, 2);

		Mat th_result;
		Mat th_result1;
		//Mat a;
		//threshold(img_gray, a, 100, 255, CV_THRESH_BINARY);
		//bitwise_and(img_gray, a, img_gray);
		//adaptiveThresholdCustom_whitedot(img_gray, th_result, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, whitePoint_step, -3, 1);
		Mat img_top11 = img_gray(Rect(10, 10, 2980, 1480));
		//adaptiveThreshold(img_top11, th_result1, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 39, -3);
		//imwrite("D:th_result1.bmp",th_result1);
		//imshow("gaosi th_result1", th_result1);
		//adaptiveThreshold(img_gray, th_result, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 59, -3);

		//6.11屏蔽最大值自适应二值化
		//adaptiveThreshold(maximg, th_result, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 49, -3.25);
		Mat th_result2;//中值滤波自适应二值化结果
		//adaptiveThreshold(img_gray, th_result2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 49, -3.25);

		//adaptiveThreshold(img_gray, th_result2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 43, -4);
		adaptiveThreshold(img_gray, th_result2, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 45, -3.5);

		//bitwise_or(th_result2, th_result, th_result);//最大值滤波与中值滤波的自适应二值化结果相或，降低漏检
		th_result = th_result2;//屏蔽或的结果
		//imwrite("D:th_result.bmp",th_result);
		//imshow("均值 th_result1", th_result);

		//针对边界位置取原图的边界
		Mat img_top = img_gray(Rect(0, 0, img_gray.cols - 1, 40));
		Mat img_bottom = img_gray(Rect(0, img_gray.rows - 40, img_gray.cols - 1, 40));
		Mat img_left = img_gray(Rect(0, 0, 60, img_gray.rows - 1));
		//Mat img_right = img_gray(Rect(img_gray.cols - 40, 0, 40, img_gray.rows - 1));
		Mat img_right = img_gray(Rect(img_gray.cols - 200, 0, 200, img_gray.rows - 1));

		Mat img_right_light = img_gray(Rect(img_gray.cols - 15, 0, 15, img_gray.rows - 1));
		Mat img_tl_R = img_gray(Rect(0, 0, 200, 150));
		Mat img_bl_R = img_gray(Rect(0, 1350, 200, 150));
		Mat img_tr_R = img_gray(Rect(2850, 0, 150, 150));
		Mat img_br_R = img_gray(Rect(2849, 1349, 150, 150));
		Mat img_tm = img_gray(Rect(500, 0, 2300, 40));
		Mat img_bm = img_gray(Rect(500, 1460, 2300, 40));

		Mat top_th, top_th1, bottom_th, bottom_th1, left_th, right_th, img_tl_R_th, img_bl_R_th, img_tr_R_th, img_br_R_th, img_right_light_th, img_tm_th, img_bm_th;

		//针对边界位置设置参数
		adaptiveThreshold(img_top, top_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, -1);
		adaptiveThreshold(img_bottom, bottom_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, -1);
		//adaptiveThreshold(img_left, left_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, -1);
		adaptiveThreshold(img_left, left_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_right, right_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_right_light, img_right_light_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
		adaptiveThreshold(img_tl_R, img_tl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_bl_R, img_bl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_tr_R, img_tr_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);
		adaptiveThreshold(img_br_R, img_br_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);
		adaptiveThreshold(img_tm, img_tm_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, -3);
		adaptiveThreshold(img_bm, img_bm_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 11, -3);

		threshold(img_top, top_th1, 30, 255, CV_THRESH_BINARY);
		double meanTop = mean(img_top, top_th1)[0];
		threshold(img_top, top_th1, meanTop, 255, CV_THRESH_BINARY);
		bitwise_and(top_th1, top_th, top_th);

		threshold(img_bottom, bottom_th1, 30, 255, CV_THRESH_BINARY);
		double meanBottom = mean(img_bottom, bottom_th1)[0];
		threshold(img_bottom, bottom_th1, meanBottom, 255, CV_THRESH_BINARY);
		bitwise_and(bottom_th1, bottom_th, bottom_th);

		//针对边界位置深拷贝
		top_th.copyTo(th_result(Rect(0, 0, th_result.cols - 1, 40)));                    //上边界
		bottom_th.copyTo(th_result(Rect(0, th_result.rows - 40, th_result.cols - 1, 40)));     //下边界
		left_th.copyTo(th_result(Rect(0, 0, 60, th_result.rows - 1)));                   //左边界
		img_tl_R_th.copyTo(th_result(Rect(0, 0, 200, 150)));                    //上边界
		img_bl_R_th.copyTo(th_result(Rect(0, 1350, 200, 150)));     //下边界
		img_tr_R_th.copyTo(th_result(Rect(2850, 0, 150, 150)));                   //左边界
		img_br_R_th.copyTo(th_result(Rect(2849, 1349, 150, 150)));      //右边界
		//th_result1.copyTo(th_result(Rect(10, 10, 2980, 1480)));

		//right_th.copyTo(th_result(Rect(th_result.cols - 40, 0, 40, th_result.rows - 1)));      //右边界

		right_th.copyTo(th_result(Rect(th_result.cols - 200, 0, 200, th_result.rows - 1)));      //右边界


		img_right_light_th.copyTo(th_result(Rect(img_gray.cols - 15, 0, 15, img_gray.rows - 1)));      //右边界
		//img_tm_th.copyTo(th_result(Rect(500, 0, 2300, 40)));
		//img_bm_th.copyTo(th_result(Rect(500, 1460, 2300, 40)));

		//th_result(Rect(0, 0, 20, th_result.rows)) = uchar(0);            //屏蔽右侧15行，防止灯口误检白点
	   // th_result(Rect(th_result.cols - 10, 0, 10, th_result.rows)) = uchar(0);            //屏蔽左侧10行，防止头部亮边误检为白点
		//th_result(Rect(0, 0, th_result.cols, 10)) = uchar(0);
		//th_result(Rect(0, th_result.rows - 10, th_result.cols, 10)) = uchar(0);

		//Mat img_top1, img_bottom1;
		//threshold(img_tm, img_top1, 110, 255, CV_THRESH_BINARY);
		//threshold(img_bm, img_bottom1, 110, 255, CV_THRESH_BINARY);
		//bitwise_and(img_tm, img_top1, img_tm);
		//bitwise_and(img_bm, img_bottom1, img_bm);

		Mat th1;
		//做掩膜
		threshold(img_gray, th1, 25, 255, CV_THRESH_BINARY);
		bitwise_and(th1, img_gray, img_gray);
		//闭运算,弥合内部空洞,连接相距很近的区域
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//闭操作结构元素
		Mat element1 = getStructuringElement(MORPH_CROSS, Size(5, 5));//闭操作结构元素
		morphologyEx(th_result, th_result, CV_MOP_CLOSE, element);   //闭运算形态学操作。可以减少噪点
	//    dilate(th_result, th_result, element);//膨胀
	//    //imwrite("D:pengzhang.bmp",th_result);
	//    erode(th_result, th_result, element);//膨胀

																	 //th_result(Rect(th_result.cols - 261, th_result.rows - 351, 260, 350)) = uchar(0);//易撕贴部分设置右下角
		//imwrite("D:fushi.bmp",th_result);
		//imshow("biyunsuan", th_result);

		vector<vector<Point>> contours;
		findContours(th_result, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
		std::sort(contours.begin(), contours.end(), compareContourAreas);

		//    cv::Mat drawing = cv::Mat::zeros(th_result.size(), CV_8UC3);
		//    for (int i = 0; i < contours.size(); i++) {
		//        cv::Scalar color(0,0, 255);
		//        cv::drawContours(drawing, contours, i, color, 2, 8);
		//        if (i % 50 == 0&&i!=0) {
		//            continue;
		//        }
		//    }

		vector<Rect> boundRect(contours.size());
		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() < 3)
			{
				break;
			}

			double area = contourArea(contours[i]);
			Mat temp_mask = Mat::zeros(th_result.rows, th_result.cols, CV_8UC1);
			drawContours(temp_mask, contours, i, 255, FILLED, 8);
			if (area >= whitePoint_lowerArea && area < whitePoint_higherArea) // 2021.5.22 jsw 下限改为80
			{
				boundRect[i] = boundingRect(Mat(contours[i]));
				int w = boundRect[i].width;
				int h = boundRect[i].height;
				int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
	//            if ((X_1 <= 25 && Y_1 <= 15) || (X_1 <= 25 && Y_2 > 1485) || (X_2 >= 2985 && Y_1 <= 15) || (X_2 > 2985 && Y_2 > 1485) || (Y_1 < 12 && Y_2 > 16) || (Y_1 > 1485 && Y_2 >= 1489) || (X_1 < 22 && X_2 < 28) || (X_1 > 2985 && X_2 <= 2990)) {
				//边上误检太多，多屏蔽一点张之航5.29
	//            if ((X_1 <= 40 && Y_1 <= 40) || (X_1 <= 40 && Y_2 > 1460) || (X_2 >= 2960 && Y_1 <= 40) || (X_2 > 2960 && Y_2 > 1460) || (Y_1 < 30 && Y_2 > 40) || (Y_1 > 1460 && Y_2 >= 1480) || (X_1 < 30 && X_2 < 40) || (X_1 > 2960 && X_2 <= 3000)||X_2>2960) {
				if ((X_1 <= 40 && Y_1 <= 40) || (X_1 <= 40 && Y_2 > 1460) || (X_2 >= 2960 && Y_1 <= 40) || (X_2 > 2960 && Y_2 > 1460) || (Y_1 < 30 && Y_2 > 40) || (Y_1 > 1460 && Y_2 >= 1480) || (X_1 < 30 && X_2 < 40) || (X_1 > 2960 && X_2 <= 3000)) {

					continue;
				}

				///6.23反光板裁剪，屏蔽裁剪区域，其他版本要删除此处
	//            if (((X_1 >= 128 && Y_1 >= 0) && X_2 <= 720 && Y_2 <70)||(X_1 >= 128 && Y_1 >= 1430 && X_2 <= 720 && Y_2 <1500)) {

	//                continue;
	//            }
				/// 6.23反光板裁剪，屏蔽裁剪区域，其他版本要删除此处


				RotatedRect rect = minAreaRect(contours[i]);
				double mw = rect.size.height;
				double mh = rect.size.width;
				double radio = max(mw / mh, mh / mw);

				//长宽比排除
				if (radio > whitePoint_w_h)
				{
					continue;
				}

				int x = boundRect[i].x + w / 2;
				int y = boundRect[i].y + h / 2;

				//Moments m = moments(contours[i]);//查找轮廓的重心
				//x_point = int(m.m10 / m.m00);
				//y_point = int(m.m01 / m.m00);
				//if (x_point > 840 && x_point < 930 && y_point > 730 && y_point < 830)
				//{
				// int A = 0;
				//}
				if (true)
				{
					//粗筛选白点缺陷
					int border = 20;//15
					int x_lt = X_1 - border;
					if (x_lt < 0)
					{
						x_lt = 0;
					}
					int y_lt = Y_1 - border;
					if (y_lt < 0)
					{
						y_lt = 0;
					}
					int x_rt = X_2 + border;
					if (x_rt > img_gray.size[1] - 1)
					{
						x_rt = img_gray.size[1] - 1;
					}
					int y_rt = Y_2 + border;
					if (y_rt > img_gray.size[0] - 1)
					{
						y_rt = img_gray.size[0] - 1;
					}
					//排除屏幕上划痕的干扰,侧光图上检测
					Mat temp_ceguang = img_ceguang(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));
					int Qnum1 = 0;
					double mean_temp1 = mean(temp_ceguang)[0];
					double ceguang_th, stddev;

					//上下边缘时排灰尘与气泡
					if (y_rt < 40 || (img_gray.rows - y_lt < 40))
					{
						Mat  col, row;
						double m, n, p;
						for (int col_line = 0; col_line < temp_ceguang.cols; col_line++)//列进行编列  行遍历
						{
							col = temp_ceguang.colRange(col_line, col_line + 1).clone();
							m = mean(col)[0];

							p = m - mean_temp1;

							if (p > 4)
								Qnum1++;
						}
					}

					//左右边缘时排灰尘与气泡
					if (x_rt < 40 || (img_gray.cols - x_rt < 40))
					{
						Mat  col, row;
						double m, n, p;
						for (int row_line = 0; row_line < temp_ceguang.rows; row_line++)//列进行编列  行遍历
						{
							row = temp_ceguang.rowRange(row_line, row_line + 1).clone();
							m = mean(row)[0];

							p = m - mean_temp1;

							if (p > 4)
								Qnum1++;
						}
					}

					//屏幕内部时排灰尘与气泡
					if ((y_rt >= 40 && (img_gray.rows - y_lt >= 40 && x_rt >= 40 && (img_gray.cols - x_rt >= 40))) || (Qnum1 > 0))
					{
						Mat mask_ceguang = th_result(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));
						double mean_out_ceguang = mean(temp_ceguang, mask_ceguang)[0];
						double mean_in_ceguang = mean(temp_ceguang, ~mask_ceguang)[0];//2021.5.22 jsw mean_out_ceguang mean_in_ceguang 交换顺序
						ceguang_th = mean_in_ceguang - mean_out_ceguang;

						//侧光图上的灰度均值方差排除气泡等干扰
						cv::Mat meanGray;
						cv::Mat stdDev;
						cv::meanStdDev(temp_ceguang, meanGray, stdDev);
						double avg = meanGray.at<double>(0, 0);
						stddev = stdDev.at<double>(0, 0);
					}
					else
					{
						ceguang_th = 0;
						stddev = 0;
					}


					//排除贴膜划痕划痕跳过
					if (ceguang_th > scratchth)
					{
						continue;
					}
					////侧光图上的灰度均值方差排除气泡等干扰
					//cv::Mat meanGray;
					//cv::Mat stdDev;
					//cv::meanStdDev(temp_ceguang, meanGray, stdDev);
					//double avg = meanGray.at<double>(0, 0);
					//double stddev = stdDev.at<double>(0, 0);
					//排除贴膜表面气泡等跳过
					if (stddev > bubbleth)
					{
						continue;
					}

					//颜色深浅判断2个指标  1:缺陷区域与周围灰度差值(整体性)  2:缺陷中心点与一次缺陷区域灰度差值(局部性)
					Mat temp_gray = img_gray(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));//2021.5.25 jsw gai
					//Mat temp_gray = Original(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));
					Mat mask = th_result(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));

					double minp;			//最小灰度值
					double maxp;			//最大灰度值
					Point low_gray, high_gray;	//正常取时灰度最大最小点

					minMaxLoc(temp_gray, &minp, &maxp, &low_gray, &high_gray, mask);	//求最大最小灰度点
					double spotpeak_temp = maxp - minp;

					double mean_temp = mean(temp_gray)[0];

					double meanOut;//缺陷外围灰度均值
					double meanIn;//缺陷区域灰度均值
					double meanAll;//整个区域的灰度均值
					int Qnum = 0;

					//防止普通边缘误检——上下边缘
					if (y_rt < 40 || (img_gray.rows - y_lt < 40))
					{
						Mat  col, row;
						double m, n, p;
						for (int col_line = 0; col_line < temp_gray.cols; col_line++)//列进行编列  行遍历
						{
							col = temp_gray.colRange(col_line, col_line + 1).clone();
							m = mean(col)[0];

							p = m - mean_temp;

							if (p > 4)
								Qnum++;	//真实白点区域
						}
					}
					//左右边缘
					if (x_rt < 40 || (img_gray.cols - x_lt < 40))
					{
						Mat  col, row;
						double m, n, p;
						for (int row_line = 0; row_line < temp_gray.rows; row_line++)//列进行编列  行遍历
						{
							row = temp_gray.rowRange(row_line, row_line + 1).clone();
							m = mean(row)[0];

							p = m - mean_temp;

							if (p > 4)
								Qnum++;	//真实白点区域
						}
					}

					if (Qnum > 0 || (y_rt >= 40 && (img_gray.rows - y_lt >= 40 && x_rt >= 40 && (img_gray.cols - x_lt >= 40))))//边界处真实白点或屏内区域
					{
						//防止R角边缘误检
						int grayValueSum = 0;
						int pixelsNum = 0;
						Mat maskGray;
						bitwise_and(temp_gray, mask, maskGray);
						for (int i = 0; i < maskGray.cols; i++)
						{
							for (int j = 0; j < maskGray.rows; j++)
							{
								if (maskGray.at<uchar>(j, i) > 100)
								{
									grayValueSum += maskGray.at<uchar>(j, i);
									pixelsNum++;
								}
							}
						}
						if (pixelsNum == 0) {
							meanIn = 0;
						}
						else
						{
							meanIn = grayValueSum / (float)pixelsNum;
						}

						grayValueSum = 0;
						pixelsNum = 0;
						bitwise_and(temp_gray, ~mask, maskGray);
						for (int i = 0; i < maskGray.cols; i++)
						{
							for (int j = 0; j < maskGray.rows; j++)
							{
								if (maskGray.at<uchar>(j, i) > 100)
								{
									grayValueSum += maskGray.at<uchar>(j, i);
									pixelsNum++;
								}
							}
						}
						if (pixelsNum == 0) {
							meanIn = 0;
						}
						else
						{
							meanOut = grayValueSum / (float)pixelsNum;
						}
						grayValueSum = 0;
						pixelsNum = 0;
						for (int i = 0; i < temp_gray.cols; i++)
						{
							for (int j = 0; j < temp_gray.rows; j++)
							{
								if (temp_gray.at<uchar>(j, i) > 100)
								{
									grayValueSum += temp_gray.at<uchar>(j, i);
									pixelsNum++;
								}
							}
						}
						if (pixelsNum == 0) {
							meanIn = 0;
						}
						else
						{
							meanAll = grayValueSum / (float)pixelsNum;
						}

						double  defect_areath = meanIn - meanOut;//缺陷区域与周围灰度差值(整体性)
						//if (img_gray.cols - x_rt < corewholeth)
						{
							//灰度差限制
							//if (defect_areath >= 4.5 && spotpeak_temp >= 3 && area <= 60 || area > 60 && defect_areath >= 4 && spotpeak_temp >= 4)//这里的参数先写成定值

							if (defect_areath >= 4.5 && spotpeak_temp >= 3 && area <= 100 && area >= 25 || area > 60 && defect_areath >= 4 && spotpeak_temp >= 4)//这里的参数先写成定值
							{
								result = true;
								cout << " 00灰度差： " << defect_areath << " 最大值-均值： " << spotpeak_temp << " 面积： " << area << endl;

								CvPoint top_lef4 = cvPoint(X_1 - 10, Y_1 - 10);
								CvPoint bottom_right4 = cvPoint(X_2 + 20, Y_2 + 20);
								rectangle(white_yiwu, top_lef4, bottom_right4, Scalar(0, 0, 0), 5, 8, 0);
								/*Point p3(x_point, y_point);
								circle(white_yiwu, p3, 12, Scalar(0, 0, 255), 1, 8, 0);*/
								string ceghaung = "cegaung:" + to_string(ceguang_th) + " " + "th:" + to_string(defect_areath) + " " /*+ "ceth:" + to_string(coreth) + " "*/ + "area:" + to_string(area) + " " + "stddev:" + to_string(stddev);
								putText(white_yiwu, ceghaung, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, 8, 0);
								//break;//去掉break，显示所有缺陷位置
							}
						}
						//else
						{
							//灰度差限制
							//if (defect_areath >= 4.2 && spotpeak_temp >= 3.2 && area <= 30 || area > 30 && defect_areath >= 3 && spotpeak_temp >= 4/*|| defect_areath >= 8|| spotpeak_temp >= 8*/)//这里的参数先写成定值

							if (defect_areath >= 4.2 && spotpeak_temp >= 3.2 && area >= 30 && area <= 100/*|| defect_areath >= 8|| spotpeak_temp >= 8*/)//这里的参数先写成定值
							{
								cout << " 11灰度差： " << defect_areath << " 最大值-均值： " << spotpeak_temp << " 面积： " << area << endl;

								result = true;
								CvPoint top_lef4 = cvPoint(X_1 - 10, Y_1 - 10);
								CvPoint bottom_right4 = cvPoint(X_2 + 20, Y_2 + 20);
								rectangle(white_yiwu, top_lef4, bottom_right4, Scalar(0, 0, 0), 5, 8, 0);
								/*Point p3(x_point, y_point);
								circle(white_yiwu, p3, 12, Scalar(0, 0, 255), 1, 8, 0);*/
								string ceghaung = "cegaung:" + to_string(ceguang_th) + " " + "th:" + to_string(defect_areath) + " " /*+ "ceth:" + to_string(coreth) + " "*/ + "area:" + to_string(area) + " " + "stddev:" + to_string(stddev);
								putText(white_yiwu, ceghaung, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, 8, 0);
								//break;//去掉break，显示所有缺陷位置
							}
						}
					}
				}
			}
			if (area < whitePoint_lowerArea)
			{
				break;
			}
		}
		if (result == true)
		{
			*causecolor = "白点";
			*mresult = white_yiwu;
		}
		return result;
	}

	/*====================================================================
	* 函 数 名: Dead_light
	* 功能描述: 死灯检测
	* 输入：主相机白底图像
	* 输出：主相机白底下检测结果图和result
	* 其他：
	======================================================================*/
	bool Dead_light(Mat white, Mat* mresult, String* causecolor)
	{
		bool result = false;
		if (result == true)
		{
			*mresult = white;
			*causecolor = "死灯";
			result = true;
		}
		return result;
	}


	bool Dead_light0(Mat white, Mat ceguang, Mat* mresult, String* causecolor)
	{
		bool result = false;
		int light_width = 200;
		//double meanValue = mean(white)[0]; //134 //143 144
		int boder = 15;
		Mat mask;
		ceguang(Rect(0, 0, boder, ceguang.rows)) = uchar(0);
		threshold(ceguang(Rect(0, 0, light_width, ceguang.rows)), mask, 0.5 * mean(ceguang(Rect(0, 0, light_width, ceguang.rows)))[0], 255, CV_THRESH_BINARY_INV);
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));//闭操作结构元素
		Mat element1 = getStructuringElement(MORPH_CROSS, Size(29, 29));//闭操作结构元素
		dilate(mask, mask, element);//膨胀
	//    //imwrite("D:pengzhang.bmp",th_result);
	//    erode(th_result, th_result, element);//膨胀



		Mat img_gray = white(Rect(0, 0, light_width, white.rows)).clone();
		img_gray(Rect(0, 0, boder, img_gray.rows)) = uchar(0);

		medianBlur(img_gray, img_gray, 3);

		Mat out; 
		int histSize[1] = { 256 };  //灰度值Size：256个
		float hrange[2] = { 0,255 }; //灰度范围[0-255]
		const float* ranges[1] = { hrange }; //单个灰度范围[0-255]
		int channels = 0;
		calcHist(&img_gray, 1, &channels, ~mask, out, 1, histSize, ranges,true, false);
		int dark_num = 0, light_num=0;
		int dark_th = 0, light_th = 0;//亮暗区二值化阈值
		for (int i = 0; i <out.rows; i++) {
			dark_num = dark_num + out.at<float>(i,0);
			if (dark_num > light_width / 10 * img_gray.rows * 1) {
				dark_th = i;
				break;
			}
		}
		for (int i = out.rows-1; i >=0; i--) {
			light_num = light_num + out.at<float>(i, 0);
			if (light_num > light_width / 10 * img_gray.rows * 2) {
				light_th = i;
				break;
			}
		}
		Mat dark_img1, light_img1;
		threshold(img_gray, dark_img1, dark_th, 255, CV_THRESH_BINARY_INV);
		threshold(img_gray, light_img1, light_th , 255, CV_THRESH_BINARY);
		Mat mask_mid = ~dark_img1 - light_img1;

		//////多裁一部分
		erode(dark_img1, dark_img1, element1);//膨胀
		erode(light_img1, light_img1, element1);//膨胀
		dark_img1 = dark_img1 - mask;
		double mean_sum = mean(img_gray(Rect(0, 0, light_width, white.rows)), mask_mid)[0];

		Mat std_sum0,mean_sum0;
		meanStdDev(img_gray(Rect(0, 0, light_width, white.rows)), mean_sum0, std_sum0);
		double mean_dark = mean(img_gray(Rect(0, 0, light_width, white.rows)), dark_img1)[0];
		double mean_light = mean(img_gray(Rect(0, 0, light_width, white.rows)), light_img1)[0];


		//Mat mat_queryThreshold = white(Rect(0, 0, light_width, white.rows));
		//double stdThreshold = mean(mat_queryThreshold)[0];

		//Mat strong_result;
		//Ptr<CLAHE> clahe = createCLAHE(5.0, Size(3, 3));
		//clahe->apply(img_gray, strong_result);

		//Mat edge_img = strong_result(Rect(0, 0, light_width, strong_result.rows)).clone();

		//Mat edge_img_dark, edge_img_light, edge_img0;
		//medianBlur(edge_img, edge_img, 3);

		//threshold(img_gray(Rect(0, 0, light_width, white.rows)), edge_img, 0.95*mean(img_gray(Rect(0, 0, light_width, ceguang.rows)), ~mask)[0], 255, CV_THRESH_BINARY_INV);
		//threshold(strong_result(Rect(0, 0, light_width, white.rows)), edge_img_light, 0,255, THRESH_OTSU);
		//edge_img_light(Rect(0, 0, 10, edge_img_light.rows)) = uchar(0);
		////threshold(img_gray(Rect(0, 0, light_width, white.rows)), edge_img, mean(img_gray(Rect(0, 0, light_width, ceguang.rows)), ~mask)[0], 255, CV_THRESH_BINARY_INV);
		////edge_img_light = ~edge_img;
		////erode(edge_img, edge_img_dark, element1);//膨胀
		////erode(edge_img_light, edge_img_light, element1);//膨胀
		////bitwise_xor(edge_img_dark, mask, edge_img_dark);


		//////多裁一部分
		//erode(edge_img, edge_img_dark, element1);//膨胀
		//erode(edge_img_light, edge_img_light, element1);//膨胀
		////bitwise_xor(edge_img_dark, mask, edge_img_dark);
		//edge_img_dark = edge_img_dark - mask;
		//double mean_sum = mean(img_gray(Rect(0, 0, light_width, white.rows)))[0];
		//double mean_dark = mean(img_gray(Rect(0, 0, light_width, white.rows)), edge_img_dark)[0];
		//double mean_light = mean(img_gray(Rect(0, 0, light_width, white.rows)), edge_img_light)[0];
		double error = 120 - mean_sum;
		cout << " 整体均值： "  << mean_sum << " 暗处均值： " << mean_dark << " 亮处均值： " << mean_light <<  endl;
		cout << " 亮处-整体均值： " << mean_light - mean_sum << " 整体-暗处均值： " << mean_sum- mean_dark << " 亮处-暗处均值： " << mean_light- mean_dark <<  endl;
		cout << " 亮处-整体均值比： " << (mean_light - mean_sum)/ mean_sum*120 << " 整体-暗处均值： " << (mean_sum - mean_dark) / mean_sum*120 << " 亮处-暗处均值： " << (mean_light - mean_dark) / mean_sum*120 << endl;
		cout << " 亮处-整体均值cha： " << mean_light - mean_sum+ error << " 整体-暗处均值： " << mean_sum - mean_dark + error << " 亮处-暗处均值： " << mean_light - mean_dark + error << endl;
		cout << " 总方差  " << std_sum0.at<double>(0, 0)/ mean_sum0.at<double>(0, 0)*120 <<"方差与差值和"<< std_sum0.at<double>(0, 0) / mean_sum0.at<double>(0, 0) * 120+
			(mean_sum - mean_dark) / mean_sum * 120 << endl;

		///根据亮处均值-暗处均值大小判死灯
		mean_light - mean_dark > 13.8 ? result = true : result = false;




		
		
		if (result == true)
		{
			Mat img_dark,bian;//死灯暗区边框图
			dilate(dark_img1, img_dark, element);//膨胀
			//显示暗区轮廓
			bian = (img_dark - dark_img1) + white(Rect(0, 0, light_width, dark_img1.rows));
			bian.copyTo(white(Rect(0, 0, light_width, dark_img1.rows)));
			*mresult = white.clone();
			*causecolor = "死灯";
			result = true;
		}
		return result;
	}

	/*====================================================================
	* 函 数 名: Shifting
	* 功能描述:移位，表现为白底图象有一条亮线
	* 输入：主相机白底图像
	* 输出：主相机白底下检测结果图和result
	* 其他：
	======================================================================*/
	//bool Shifting(Mat white, Mat* mresult, String* causecolor, int num, Mat& left_white, Mat& right_white, Mat& src_L1)
	//{
	//	bool result = false;

	//	vector<vector<Point>> doubtContours;
	//	findContours(left_white, doubtContours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	//	std::sort(doubtContours.begin(), doubtContours.end(), compareContourAreas);

	//	vector<Rect> doubtBoundRect(doubtContours.size());

	//	for (int i = 0; i < doubtContours.size(); i++)
	//	{
	//		double area = contourArea(doubtContours[i]); //363
	//		doubtBoundRect[i] = boundingRect(Mat(doubtContours[i]));

	//		//获取区域中心
	//		int centerX = doubtBoundRect[i].tl().x + doubtBoundRect[i].width / 2;
	//		int centerY = doubtBoundRect[i].tl().y + doubtBoundRect[i].height / 2;

	//		Mat temp_mask = Mat::zeros(white.rows, white.cols, CV_8UC1);
	//		drawContours(temp_mask, doubtContours, i, 255, FILLED, 8);
	//		//判断边角出现较大像素区域则认定为出现移位
	//		if ((centerX < 100 || centerX>2900) && (centerY < 100 || centerY>1400) && (area > YWPara->edgeAreaDownLimit && area < YWPara->edgeAreaUpperLimit) ||
	//			centerY > 1480 && area > 500 && area < 5000)
	//		{
	//			CvPoint top_lef4 = cvPoint(doubtBoundRect[i].tl().x, doubtBoundRect[i].tl().y);
	//			CvPoint bottom_right4 = cvPoint(doubtBoundRect[i].br().x, doubtBoundRect[i].br().y);
	//			rectangle(left_white, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
	//			*mresult = left_white;
	//			*causecolor = "移位";
	//			return true;
	//		}
	//	}

	//	Mat binaryImage = Mat::zeros(src_L1.size(), CV_8UC1);                              //二值图像
	//	threshold(src_L1, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化

	//	vector<vector<Point>> originDtContours;
	//	findContours(binaryImage, originDtContours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	//	std::sort(originDtContours.begin(), originDtContours.end(), compareContourAreas);

	//	int sum = 0;
	//	vector<Rect> originDtBoundRect(originDtContours.size());
	//	for (int i = 0; i < originDtContours.size(); i++)
	//	{
	//		double area = contourArea(originDtContours[i]); // 35112
	//		Mat temp_mask = Mat::zeros(src_L1.rows, src_L1.cols, CV_8UC1);
	//		drawContours(temp_mask, originDtContours, i, 255, FILLED, 8);
	//		if (area > 300) {
	//			sum++;
	//		}
	//		if (sum >= 2)
	//		{
	//			*mresult = src_L1;
	//			*causecolor = "移位";
	//			return true;
	//		}
	//	}


	//	//拷贝一张主相机白底图
	//	Mat img_gray = white.clone();

	//	Mat ad_result, th1, th2, th3, th4, th5, ImageBinary;

	//	adaptiveThresholdCustom(img_gray, th1, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 27, 5.5, 1);


	//	//threshold(img_gray, th4, 80, 255, 0);  
	//	// 2021/05/26
	//	threshold(img_gray, th4, 70, 255, 0);

	//	cv::Mat whiteMean;
	//	cv::Mat whiteStdDev;
	//	cv::meanStdDev(white, whiteMean, whiteStdDev);

	//	threshold(img_gray, th5, 70, whiteMean.at<double>(0, 0), THRESH_BINARY_INV);

	//	//屏蔽灯口10行
	//	th1(Rect(0, 0, 10, th1.rows)) = uchar(0);           //wsc 2021/03/05 防止漏检
	//	//创建2400*1100的纯黑模板
	//	Mat th_result = Mat::zeros(th1.rows - 2 * 200, th1.cols - 2 * 500, img_gray.type());
	//	//将自适应二值化的中间位置2000*1100涂黑
	//	th_result.copyTo(th1(Rect(500, 200, th1.cols - 2 * 500, th1.rows - 2 * 200)));
	//	vector<vector<Point>> contours;

	//	//二值化操作
	//	threshold(img_gray, ImageBinary, 100, 255, CV_THRESH_BINARY);
	//	bitwise_and(ImageBinary, th1, th2);

	//	//涂黑上下
	//	th1(Rect(0, 0, th1.cols - 1, YWPara->lightPortShieldWidth)) = uchar(0);
	//	th1(Rect(0, th1.rows - 10, th1.cols - 1, 10)) = uchar(0);

	//	th1(Rect(0, 0, 100, 100)) = uchar(0);
	//	th1(Rect(0, th1.rows - 100, 100, 100)) = uchar(0);
	//	th1(Rect(th1.cols - 100, 0, 100, 100)) = uchar(0);
	//	th1(Rect(th1.cols - 100, th1.rows - 100, 100, 100)) = uchar(0);


	//	bitwise_and(th1, th4, th1);
	//	threshold(img_gray, th3, 190, 255, 0);
	//	th3(Rect(10, 0, 2990, th3.rows)) = uchar(0);
	//	bitwise_or(th1, th3, th1);

	//	findContours(th1, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	//	std::sort(contours.begin(), contours.end(), compareContourAreas);

	//	vector<Rect> boundRect(contours.size());
	//	vector<RotatedRect>box(contours.size());
	//	Point2f rect[4];
	//	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	//	{
	//		double area = contourArea(contours[i]);
	//		Mat temp_mask = Mat::zeros(th1.rows, th1.cols, CV_8UC1);
	//		drawContours(temp_mask, contours, i, 255, FILLED, 8);
	//		if (area > YWPara->areaDownLimit && area < YWPara->areaUpperLimit)
	//		{
	//			Mat temp_mask = Mat::zeros(th1.rows, th1.cols, CV_8UC1);
	//			drawContours(temp_mask, contours, i, 255, FILLED, 8);

	//			boundRect[i] = boundingRect(Mat(contours[i]));

	//			cv::Mat tempGray = white(boundRect[i]);
	//			cv::Mat tempGrayClone = tempGray.clone();
	//			cv::Mat th5_1 = th5(boundRect[i]);
	//			bitwise_or(tempGrayClone, th5_1, tempGrayClone);
	//			cv::Mat meanGray1;
	//			cv::Mat stdDev1;
	//			cv::meanStdDev(tempGrayClone, meanGray1, stdDev1);

	//			//获取当前区域长宽比，用于得到相对于长宽比的标准差
	//			double virtualRadio = max(boundRect[i].width / boundRect[i].height, boundRect[i].height / boundRect[i].width);
	//			double stddev1 = stdDev1.at<double>(0, 0);    // 65
	//			double val = stddev1 / virtualRadio * 9;
	//			// 44.2
	//			if (stddev1 >= 44.2 && (stddev1 / virtualRadio * 9) >= YWPara->standardDevThres2) //将长宽比相对标准差对应系数改为9
	//			{
	//				continue;
	//			}


	//			box[i] = minAreaRect(Mat(contours[i]));
	//			box[i].points(rect);
	//			float Width = sqrt(abs(rect[0].x - rect[1].x) * abs(rect[0].x - rect[1].x) + abs(rect[0].y - rect[1].y) * abs(rect[0].y - rect[1].y));
	//			float Height = sqrt(abs(rect[1].x - rect[2].x) * abs(rect[1].x - rect[2].x) + abs(rect[1].y - rect[2].y) * abs(rect[1].y - rect[2].y));
	//			float w = boundRect[i].width;
	//			float h = boundRect[i].height;
	//			//            RotatedRect rect = minAreaRect(contours[i]);  //包覆轮廓的最小斜矩形 划伤缺陷有旋转特点
	//			int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
	//			int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
	//			int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
	//			int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值

	//			double HeWid = max(Height / Width, Width / Height);
	//			if ((w < 5 && X_1 >= th1.cols - 4) || (w < 5 && X_2 <= 4))
	//			{
	//				continue;
	//			}

	//			if (HeWid >= YWPara->lengthWidthRadioULimit && max(Height, Width) >= 40)  // 03/04 wsc HeWid 3.2 -> 10
	//			{
	//				int border = 25;
	//				X_1 = X_1 - border;
	//				Y_1 = Y_1 - border;
	//				X_2 = X_2 + border;
	//				Y_2 = Y_2 + border;
	//				if (X_1 < 0)
	//				{
	//					X_1 = 0;
	//				}
	//				if (Y_1 < 0)
	//				{
	//					Y_1 = 0;
	//				}
	//				if (X_2 > th1.cols - 1)
	//				{
	//					X_2 = th1.cols - 1;
	//				}
	//				if (Y_2 > th1.rows - 1)
	//				{
	//					Y_2 = th1.rows - 1;
	//				}
	//				Mat ImageOutBinary;
	//				Mat tempImage = ImageBinary(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
	//				Mat tempBinary1 = temp_mask(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1)).clone();
	//				Mat tempBinary2 = th2(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1)).clone();
	//				Mat tempGray = img_gray(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));

	//				double mean_all = mean(tempGray, tempImage)[0];
	//				threshold(tempGray, ImageOutBinary, mean_all - 10, 255, CV_THRESH_BINARY);
	//				double mean_In;
	//				mean_In = mean(tempGray, tempBinary1)[0];
	//				bitwise_and(ImageOutBinary, ~tempBinary2, tempBinary2);
	//				double mean_Out = mean(tempGray, tempBinary2)[0];
	//				double differ = mean_Out - mean_In;
	//				double MinmeanIn;
	//				if (mean_all >= 150)
	//					MinmeanIn = mean_all / 2 + 20;
	//				else if (mean_all >= 100)
	//					MinmeanIn = mean_all / 2 + 10;
	//				else
	//					MinmeanIn = mean_all / 2;
	//				if (differ < 0)
	//				{
	//					differ = -differ;
	//				}
	//				if (((HeWid <= YWPara->lengthWidthRadio && differ >= YWPara->doubtAreaIntensity1)
	//					|| (HeWid > YWPara->lengthWidthRadio && differ >= YWPara->doubtAreaIntensity2)) && mean_In >= MinmeanIn)   //日期：2021/3/2 6504:   HeWid :8-->9   differ 13 --> 19
	//				{
	//					result = true;
	//					CvPoint top_lef4 = cvPoint(X_1, Y_1);
	//					CvPoint bottom_right4 = cvPoint(X_2, Y_2);
	//					rectangle(white, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
	//					break;
	//				}
	//			}
	//		}
	//	}
	//	if (result == true)
	//	{
	//		*mresult = white;
	//		*causecolor = "移位";
	//		result = true;
	//	}
	//	return result;
	//}
	//
	/*=========================================================
	 * 函 数 名: Mura
	 * 功能描述: 白印检测
	 * 输入: 左右侧相机拍摄白点图片和对应侧光图片，侧光图片排除划痕和气泡
	 * 输出：白底下检测结果图和result
	 * 修改时间：2021年1月12日
	 * 其他：
	 =========================================================*/


	bool Mura_Decter(Mat imageGray, Mat* mresult, String* causecolor)
	{
		bool result = false;

		int length = 80;
		int left_length = 50; //wsc 0312 改变左侧边区域使用不同自适应阈值的宽度
		Mat th_result;
		Mat img_gray = imageGray.clone();
		adaptiveThreshold(img_gray, th_result, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 107, -3);//2021.5.21 97-107

		int shuidi_length = 160;
		int part = img_gray.rows / 2 - shuidi_length;
		//针对边界位置取原图的边界
		Mat img_top = img_gray(Rect(0, 0, img_gray.cols - 1, length));
		Mat img_bottom = img_gray(Rect(0, img_gray.rows - length, img_gray.cols - 1, length));
		Mat img_left = img_gray(Rect(0, 0, length + left_length, img_gray.rows - 1));
		Mat img_right = img_gray(Rect(img_gray.cols - length, 0, length, img_gray.rows - 1));
		Mat img_right_light = img_gray(Rect(img_gray.cols - 150, part, 150, shuidi_length * 2));
		Mat img_tl_R = img_gray(Rect(0, 0, 150, 150));
		Mat img_bl_R = img_gray(Rect(0, 1350, 150, 150));
		Mat img_tr_R = img_gray(Rect(2850, 0, 150, 150));
		Mat img_br_R = img_gray(Rect(2849, 1349, 150, 150));

		Mat top_th, top_th1, bottom_th, bottom_th1, left_th, right_th, img_tl_R_th, img_bl_R_th, img_tr_R_th, img_br_R_th, img_right_light_th;

		//针对边界位置设置参数
		adaptiveThreshold(img_top, top_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -1);
		adaptiveThreshold(img_bottom, bottom_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -1);
		//adaptiveThreshold(img_left, left_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, -1);
		adaptiveThreshold(img_left, left_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_right, right_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_right_light, img_right_light_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
		adaptiveThreshold(img_tl_R, img_tl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_bl_R, img_bl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_tr_R, img_tr_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);
		adaptiveThreshold(img_br_R, img_br_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);

		threshold(img_top, top_th1, 30, 255, CV_THRESH_BINARY);
		double meanTop = mean(img_top, top_th1)[0];
		threshold(img_top, top_th1, meanTop, 255, CV_THRESH_BINARY);
		bitwise_and(top_th1, top_th, top_th);

		threshold(img_bottom, bottom_th1, 30, 255, CV_THRESH_BINARY);
		double meanBottom = mean(img_bottom, bottom_th1)[0];
		threshold(img_bottom, bottom_th1, meanBottom, 255, CV_THRESH_BINARY);
		bitwise_and(bottom_th1, bottom_th, bottom_th);

		//针对边界位置深拷贝
		top_th.copyTo(th_result(Rect(0, 0, th_result.cols - 1, length)));                    //上边界
		bottom_th.copyTo(th_result(Rect(0, th_result.rows - length, th_result.cols - 1, length)));     //下边界
		left_th.copyTo(th_result(Rect(0, 0, length + left_length, th_result.rows - 1)));                   //左边界
		right_th.copyTo(th_result(Rect(th_result.cols - length, 0, length, th_result.rows - 1)));      //右边界
		img_right_light_th.copyTo(th_result(Rect(img_gray.cols - 150, part, 150, shuidi_length * 2)));      //右边界
		img_tl_R_th.copyTo(th_result(Rect(0, 0, 150, 150)));                    //上边界
		img_bl_R_th.copyTo(th_result(Rect(0, 1350, 150, 150)));     //下边界
		img_tr_R_th.copyTo(th_result(Rect(2850, 0, 150, 150)));                   //左边界
		img_br_R_th.copyTo(th_result(Rect(2849, 1349, 150, 150)));      //右边界

		th_result(Rect(0, 0, 15, th_result.rows)) = uchar(0);            //屏蔽右侧15行，防止灯口误检白点
		th_result(Rect(th_result.cols - 15, 0, 15, th_result.rows)) = uchar(0);            //屏蔽左侧10行，防止头部亮边误检为白点

		th_result(Rect(0, 0, th_result.cols, 10)) = uchar(0);  // 屏蔽上下亮边误检   //2021/03/03 wsc_ChangeCode 添加两行
		th_result(Rect(0, th_result.rows - 10, th_result.cols, 10)) = uchar(0);

		//2021.5.22 jsw/ 增
		th_result(Rect(0, 0, 40, 40)) = uchar(0);
		th_result(Rect(th_result.cols - 40, 0, 40, 40)) = uchar(0);
		th_result(Rect(0, th_result.rows - 40, 40, 40)) = uchar(0);
		th_result(Rect(th_result.cols - 40, th_result.rows - 40, 40, 40)) = uchar(0);

		Mat th1;
		//做掩膜
		threshold(img_gray, th1, 25, 255, CV_THRESH_BINARY);
		bitwise_and(th1, img_gray, img_gray);
		//闭运算,弥合内部空洞,连接相距很近的区域
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//闭操作结构元素
		morphologyEx(th_result, th_result, CV_MOP_CLOSE, element);   //闭运算形态学操作。可以减少噪点

		vector<vector<Point>> contours;
		double meanV = mean(th_result)[0];
		//qDebug() << meanV;
		findContours(th_result, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
		std::sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Rect> boundRect(contours.size());

		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);

			if (area >= 200 && area < 60000) //2021.5.22 jsw 100-200
			{
				boundRect[i] = boundingRect(Mat(contours[i]));
				int w = boundRect[i].width;
				int h = boundRect[i].height;
				int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值

				RotatedRect rect = minAreaRect(contours[i]);
				double mw = rect.size.height;
				double mh = rect.size.width;
				double radio = max(mw / mh, mh / mw);

				//长宽比排除
				if (radio > 7 && (Y_1 >= th_result.rows - 40 || Y_2 <= 40) || radio > 10 && (X_1 >= th_result.cols - 40 || X_2 <= 40))   //日期：2021/3/2 6504: 10 -> 7 2021.5.21增加条件
				{
					continue;
				}

				int boder = 10;
				int x_lt = X_1 - boder - int(w);
				int y_lt = Y_1 - boder - int(h);
				int x_rt = X_2 + boder + int(w);
				int y_rt = Y_2 + boder + int(h);
				if (x_lt < 0)
				{
					x_lt = 0;
				}
				if (y_lt < 0)
				{
					y_lt = 0;
				}
				if (x_rt > img_gray.size[1] - 1)
				{
					x_rt = img_gray.size[1] - 1;
				}
				if (y_rt > img_gray.size[0] - 1)
				{
					y_rt = img_gray.size[0] - 1;
				}

				Mat tempImage_Binary;
				Mat tempGray = img_gray(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt)).clone();
				Mat tempBinary = th_result(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt)).clone();
				Mat tempBinary0 = th_result(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1)).clone();

				if (X_1 >= th_result.cols - 300 || Y_1 >= th_result.rows - 50 || X_2 <= 300 || Y_2 <= 50)
				{
					double Width_Average = 0.0;
					int row_line;
					for (row_line = 0; row_line < tempBinary0.rows; row_line++)//列进行编列  行遍历
					{
						Mat row;
						row = tempBinary0.rowRange(row_line, row_line + 1).clone();
						double num = countNonZero(row);
						Width_Average = Width_Average + num;
					}
					Width_Average = Width_Average / row_line;
					if (Width_Average < tempBinary0.cols / 5)
						continue;

					int col_line;
					Width_Average = 0.0;
					for (col_line = 0; col_line < tempBinary0.cols; col_line++)//列进行编列  行遍历
					{
						Mat col;
						col = tempBinary0.colRange(col_line, col_line + 1).clone();
						double num = countNonZero(col);
						Width_Average = Width_Average + num;
					}
					Width_Average = Width_Average / row_line;
					if (Width_Average < tempBinary0.rows / 5)
						continue;
				}

				Mat temp_Binary1, temp_Binary2;
				double mean_In = mean(tempGray, tempBinary)[0];
				threshold(tempGray, tempImage_Binary, 30, 255, CV_THRESH_BINARY);
				double mean_All = mean(tempGray, tempImage_Binary)[0];
				//两步操作排除过亮和过暗区域
				threshold(tempGray, temp_Binary1, mean_All - 25, 255, CV_THRESH_BINARY);
				if (mean_All >= 180)//2021.5.21 180-130
				{
					threshold(tempGray, temp_Binary2, mean_All + 25, 255, CV_THRESH_BINARY);
				}
				else
				{
					threshold(tempGray, temp_Binary2, mean_All + 40, 255, CV_THRESH_BINARY);
				}
				bitwise_and(temp_Binary1, ~tempBinary, tempBinary);
				bitwise_and(tempBinary, ~temp_Binary2, tempBinary);
				double mean_Out = mean(tempGray, tempBinary)[0];
				double differ = mean_In - mean_Out;

				//if ((mean_All >= 160 && area >= 500 && differ >= 5.3) || differ >= 6.4)
				if ((area >= 300 && differ >= 5.3) || (area >= 500 && differ >= 4.8) || (area >= 800 && differ >= 4.5) || differ >= 6.4)
				{
					result = true;
					CvPoint top_lef4 = cvPoint(X_1, Y_1);
					CvPoint bottom_right4 = cvPoint(X_2, Y_2);
					rectangle(imageGray, top_lef4, bottom_right4, Scalar(0, 0, 0), 5, 8, 0);
				}
				else if (differ < 6.4 && differ >= 0)
				{
					if (w <= h)
					{
						Mat tempGray_Width = img_gray(Rect(x_lt, Y_1, x_rt - x_lt, Y_2 - Y_1)).clone();
						Mat tempBinary_Width = th_result(Rect(x_lt, Y_1, x_rt - x_lt, Y_2 - Y_1)).clone();
						threshold(tempGray_Width, tempImage_Binary, 30, 255, CV_THRESH_BINARY);
						double mean_All = mean(tempGray_Width, tempImage_Binary)[0];
						threshold(tempGray_Width, temp_Binary1, mean_All - 20, 255, CV_THRESH_BINARY);
						threshold(tempGray_Width, temp_Binary2, mean_All + 25, 255, CV_THRESH_BINARY);//2021.5.21 40-25
						bitwise_and(temp_Binary1, ~tempBinary_Width, tempBinary_Width);
						bitwise_and(tempBinary_Width, ~temp_Binary2, tempBinary_Width);
						double meanOut_Width = mean(tempGray_Width, tempBinary_Width)[0];
						differ = mean_In - meanOut_Width;
					}
					else if (w > h)
					{
						Mat tempGray_Height = img_gray(Rect(X_1, y_lt, X_2 - X_1, y_rt - y_lt)).clone();
						Mat tempBinary_Height = th_result(Rect(X_1, y_lt, X_2 - X_1, y_rt - y_lt)).clone();
						threshold(tempGray_Height, tempImage_Binary, 30, 255, CV_THRESH_BINARY);
						mean_All = mean(tempGray_Height, tempImage_Binary)[0];
						threshold(tempGray_Height, temp_Binary1, mean_All - 20, 255, CV_THRESH_BINARY);
						threshold(tempGray_Height, temp_Binary2, mean_All + 25, 255, CV_THRESH_BINARY);//2021.5.21 40-25
						bitwise_and(temp_Binary1, ~tempBinary_Height, tempBinary_Height);
						bitwise_and(tempBinary_Height, ~temp_Binary2, tempBinary_Height);
						double meanOut_Height = mean(tempGray_Height, tempBinary_Height)[0];
						differ = mean_In - meanOut_Height;
					}

					//if ((mean_All >= 160 && area >= 500 && differ >= 5.3) || differ >= 6.4)
					if ((area >= 300 && differ >= 5.3) || (area >= 500 && differ >= 4.8) || (area >= 800 && differ >= 4.5) || differ >= 6.4)
					{
						result = true;
						CvPoint top_lef4 = cvPoint(X_1, Y_1);
						CvPoint bottom_right4 = cvPoint(X_2, Y_2);
						rectangle(imageGray, top_lef4, bottom_right4, Scalar(0, 0, 0), 5, 8, 0);
					}
				}

			}
		}
		if (result == true)
		{
			*causecolor = "白印";
			*mresult = imageGray;
		}
		return result;
	}

	/*=========================================================
	  * 函 数 名: ForeignBody
	  * 功能描述: 异物缺陷检测
	  * 函数输入：主相机白底图像和主相机拍摄侧光图像
	  * 备注说明：2020年12月17日修改
	  =========================================================*/
	bool ForeignBody(Mat white_yiwu, Mat ceguang, Mat Original, Mat* mresult, String* causecolor)//灰度检测S
	{
		double yiwu_pre_size = 51;//原51
		double yiwu_pre_th = 5.5;
		double yiwu_area_lower = 6;//原25
		double yiwu_area_upper = 2200;//原6000
		double yiwu_sec_size = 47;
		double yiwu_sec_th = 6;
		double yiwu_sec_area_lower = 9;
		double yiwu_sec_area_upper = 2200;
		double yiwu_lighth_BigArea = 5.9;       //暂时没用
		double yiwu_lighth_SmallArea = 10;      //暂时没用

		double yiwu_MaxLight = 50;
		if (yiwu_MaxLight >= 100)
			yiwu_MaxLight = 100;
		if (yiwu_MaxLight <= 10)
			yiwu_MaxLight = 10;
		double yiwu_light1 = 0.118 * yiwu_MaxLight;
		double yiwu_light2 = 0.15 * yiwu_MaxLight;
		double yiwu_light3 = 0.2 * yiwu_MaxLight;
		double yiwu_light4 = 0.26 * yiwu_MaxLight;
		//yiwu_area_lower = 0.24*yiwu_MaxLight;
		bool result = false;
		//int length = 200;
		Mat gray_ceguang = ceguang.clone();
		Mat img_gray = white_yiwu.clone();
		Mat th1, th2, binaryImage;

		adaptiveThresholdCustom(img_gray, th1, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, yiwu_pre_size, yiwu_pre_th, 1);

		Mat element = getStructuringElement(MORPH_RECT, Size(11, 11));//闭操作结构元素
		morphologyEx(th1, th1, CV_MOP_CLOSE, element);   //闭运算形态学操作。可以减少噪点


		//Mat yanmo(th1.size(), th1.type(), Scalar(255));//全白图
		//threshold(img_gray, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
		//Mat left_up = binaryImage(Rect(0,0, length, length));
		//left_up.copyTo(yanmo(Rect(0,0, length, length)));
		//Mat left_down = binaryImage(Rect(0, th1.rows - length, length, th1.rows-1));
		//left_down.copyTo(yanmo(Rect(0, th1.rows - length, length, th1.rows-1)));
		//Mat right_up = binaryImage(Rect(th1.cols - length, 0, th1.cols-1, length));
		//right_up.copyTo(yanmo(Rect(th1.cols - length, 0, th1.cols-1, length)));
		//Mat right_down = binaryImage(Rect(th1.cols - length, th1.rows - length, th1.cols-1, th1.rows-1));
		//right_down.copyTo(yanmo(Rect(th1.cols - length, th1.rows - length, th1.cols-1, th1.rows-1)));

		//bitwise_and(th1, yanmo,th1);
		//针对边界位置取原图的边界
		//针对边界位置取原图的边界
	//Mat img_left = img_gray(Rect(0, 0, 20, img_gray.rows - 1));
	//Mat img_right = img_gray(Rect(img_gray.cols - 20, 0, 20, img_gray.rows - 1));
	//Mat img_tl_R = img_gray(Rect(0, 0, 150, 150));
	//Mat img_bl_R = img_gray(Rect(0, 1350, 150, 150));
	//Mat img_tr_R = img_gray(Rect(2850, 0, 150, 150));
	//Mat img_br_R = img_gray(Rect(2849, 1349, 150, 150));

	//Mat top_th, bottom_th, left_th, right_th, img_tl_R_th, img_bl_R_th, img_tr_R_th, img_br_R_th, img_right_light_th;

	////针对边界位置设置参数
	//adaptiveThreshold(img_left, left_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, -1);
	//adaptiveThreshold(img_right, right_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
	//adaptiveThreshold(img_tl_R, img_tl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
	//adaptiveThreshold(img_bl_R, img_bl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
	//adaptiveThreshold(img_tr_R, img_tr_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);
	//adaptiveThreshold(img_br_R, img_br_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);

	////针对边界位置深拷贝
	//left_th.copyTo(th1(Rect(0, 0, 20, th1.rows - 1)));                   //左边界
	//right_th.copyTo(th1(Rect(th1.cols - 20, 0, 20, th1.rows - 1)));      //右边界
	//img_tl_R_th.copyTo(th1(Rect(0, 0, 150, 150)));                    //上边界
	//img_bl_R_th.copyTo(th1(Rect(0, 1350, 150, 150)));     //下边界
	//img_tr_R_th.copyTo(th1(Rect(2850, 0, 150, 150)));                   //左边界
	//img_br_R_th.copyTo(th1(Rect(2849, 1349, 150, 150)));      //右边界

	//Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//闭操作结构元素
	//morphologyEx(th1, th1, CV_MOP_CLOSE, element);   //闭运算形态学操作。可以减少噪点

		Mat th1_R;
		//做掩膜
		threshold(Original, th1_R, 25, 255, CV_THRESH_BINARY);
		//bitwise_and(th1_R, img_gray, img_gray);
		bitwise_and(th1_R, Original, Original);

		//       Mat left_yamo(th1.rows, 70, th1.type(), Scalar(0));//全白图
		//       left_yamo.copyTo(th1(Rect(0,0,70, th1.rows)));      //右边界
			   //left_yamo.copyTo(th1(Rect(th1.cols-71, 0, th1.cols, th1.rows)));      //右边界

		vector<vector<Point>> contours;
		findContours(th1, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		vector<Rect> boundRect(contours.size());
		vector<Rect> boundRect_area(contours.size());

		//主相机侧光图全局特征，防止良品误检
		//       double cgm = 0.0, cgstd = 0.0;
		//       vector<vector<Point>> cgcontours;
		//       findContours(gray_ceguang, cgcontours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//       Mat cgmask, cgtem_m, cgtem_s;
		//       cv::threshold(gray_ceguang, cgmask, 10, 255, CV_THRESH_BINARY);
		//       cv::meanStdDev(gray_ceguang, cgtem_m, cgtem_s);
		//       cgm = cgtem_m.at<double>(0, 0);
		//       cgstd = cgtem_s.at<double>(0, 0);
		//       cout << cgm << endl;//主相机侧光图全局特征：灰度均值1
		//       cout  << cgstd << endl;//主相机侧光图全局特征：标准差1
		//       cout << cgstd / cgm << endl;//主相机侧光图全局特征：变异系数1
		//       if (cgm < 170 && cgstd < 65)
		//           return result;

		float w, h;
		int X_1, Y_1, X_2, Y_2;//矩形左上角X坐标值

		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			if (area > yiwu_area_lower && area < yiwu_area_upper/*&&i == 8*/)
			{
				boundRect_area[i] = boundingRect(Mat(contours[i]));
				w = boundRect_area[i].width;
				h = boundRect_area[i].height;
				X_1 = boundRect_area[i].tl().x;//矩形左上角X坐标值
				Y_1 = boundRect_area[i].tl().y;//矩形左上角Y坐标值
				X_2 = boundRect_area[i].br().x;//矩形右下角X坐标值
				Y_2 = boundRect_area[i].br().y;//矩形右下角Y坐标值

				//防止R角位置误检
				if ((X_1 < 10 && Y_1 < 10) || (X_1 < 10 && Y_2> 1490) || (X_2 > 2990 && Y_1 < 10) || (X_2 > 2990 && Y_2 > 1490))
				{
					continue;
				}

				double longShortRatio = max(h / w, w / h);
				if (longShortRatio < 5 && min(w, h) >= 2 && max(w, h) < 120)	//对异物最大最小直径,长宽之比做限制50
				{
					if (area > 300)
					{
						if ((X_1 == 0 && Y_1 == 0) || (X_1 == 0 && Y_2 >= th1.rows - 1) || (Y_1 == 0 && X_2 >= th1.cols - 1) || (X_2 >= th1.cols - 1 && Y_2 >= th1.rows - 1))
							continue;
					}
					if (X_2 <= 4 || (th1.cols - X_1) <= 4 || Y_2 <= 4 || (th1.rows - Y_1 <= 4))
						continue;
					int Ceguang_Expand = 1;
					int x_lt = X_1 - Ceguang_Expand;
					if (x_lt < 0)
					{
						x_lt = 0;
					}
					int y_lt = Y_1 - Ceguang_Expand;
					if (y_lt < 0)
					{
						y_lt = 0;
					}
					int x_rt = X_2 + Ceguang_Expand;
					if (x_rt > gray_ceguang.cols - 1)
					{
						x_rt = gray_ceguang.cols - 1;
						//x_rt = gray_ceguang.cols;
					}
					int y_rt = Y_2 + Ceguang_Expand;
					if (y_rt > gray_ceguang.rows - 1)
					{
						//y_rt = gray_ceguang.rows;
						y_rt = gray_ceguang.rows - 1;
					}


					//计算侧光图像缺陷中心和缺陷外围灰度差
					Mat temp_ceguang = gray_ceguang(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));
					double ceguangth_huahen, ceguangth_huichen, stddev;
					int Qnum = 0;
					double mean_temp = mean(temp_ceguang)[0];

					//上下边缘时排灰尘与气泡
					if (y_rt < 20 || (img_gray.rows - y_lt < 20))
					{
						Mat  col, row;
						double m, n, p;
						for (int col_line = 0; col_line < temp_ceguang.cols; col_line++)//列进行编列  行遍历
						{
							col = temp_ceguang.colRange(col_line, col_line + 1).clone();
							m = mean(col)[0];

							p = m - mean_temp;

							if (p > 4)
								Qnum++;
						}
					}

					//左右边缘时排灰尘与气泡
					if (x_rt < 20 || (img_gray.cols - x_lt < 20))
					{
						Mat  col, row;
						double m, n, p;
						for (int row_line = 0; row_line < temp_ceguang.rows; row_line++)//列进行编列  行遍历
						{
							row = temp_ceguang.rowRange(row_line, row_line + 1).clone();
							m = mean(row)[0];

							p = m - mean_temp;

							if (p > 4)
								Qnum++;
						}
					}

					//屏幕内部时排灰尘与气泡
					if ((y_rt >= 20 && (img_gray.rows - y_lt >= 20 && x_rt >= 20 && (img_gray.cols - x_lt >= 20))) || (Qnum > 0))
					{
						Mat mask = th1(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));
						double mean_in = mean(temp_ceguang, mask)[0];
						double mean_out = mean(temp_ceguang, ~mask)[0];
						ceguangth_huahen = mean_in - mean_out;
						ceguangth_huichen = mean_out - mean_in;
						//侧光图上的灰度均值方差排除气泡等干扰
						cv::Mat meanGray;
						cv::Mat stdDev;
						cv::meanStdDev(temp_ceguang, meanGray, stdDev);
						stddev = stdDev.at<double>(0, 0);
					}
					else
					{
						ceguangth_huahen = 0;
						stddev = 0;
					}

					//if (ceguangth <-5&&area>30)
					//{
					   // continue;
					//}
					//if ((ceguangth <1.5 && stddev < 50) || (ceguangth <-5 && area>30))

					if ((ceguangth_huahen >= 0 && ceguangth_huahen < 1.5 && area <= 50)
						|| (ceguangth_huahen >= 0 && ceguangth_huahen < 2.3 && area >50)
						|| (ceguangth_huichen >= 3 && area > 30)
						|| (ceguangth_huichen >= 0 && ceguangth_huichen < 3)
						|| (ceguangth_huichen >= 3 && area > 15))
					{
						int Secscreen_Expand = 20;
						int x_lt_sec = X_1 - Secscreen_Expand;
						if (x_lt_sec < 0)
						{
							x_lt_sec = 0;
						}
						int y_lt_sec = Y_1 - Secscreen_Expand;
						if (y_lt_sec < 0)
						{
							y_lt_sec = 0;
						}
						int x_rt_sec = X_2 + Secscreen_Expand;
						if (x_rt_sec > gray_ceguang.cols - 1)
						{
							//x_rt_sec = gray_ceguang.cols;
							x_rt_sec = gray_ceguang.cols - 1;
						}
						int y_rt_sec = Y_2 + Secscreen_Expand;
						if (y_rt_sec > gray_ceguang.rows - 1)
						{
							//y_rt_sec = gray_ceguang.rows;
							y_rt_sec = gray_ceguang.rows - 1;
						}

						//Mat imageSuspected = img_gray(Rect(x_lt_sec, y_lt_sec, x_rt_sec - x_lt_sec, y_rt_sec - y_lt_sec));
						Mat imageSuspected = Original(Rect(x_lt_sec, y_lt_sec, x_rt_sec - x_lt_sec, y_rt_sec - y_lt_sec));
						Mat TempCeguang = gray_ceguang(Rect(x_lt_sec, y_lt_sec, x_rt_sec - x_lt_sec, y_rt_sec - y_lt_sec));
						double mean_temp1 = mean(imageSuspected)[0];

						int num_wu = 0;
						//防止普通边缘误检——上下边缘
						if (y_lt_sec < 20 || (img_gray.rows - y_rt_sec < 20))
						{
							Mat  col, row;
							double m, n, p;
							for (int col_line = 0; col_line < imageSuspected.cols; col_line++)//列进行编列  行遍历
							{
								col = imageSuspected.colRange(col_line, col_line + 1).clone();
								m = mean(col)[0];

								p = m - mean_temp1;

								if (p > 4 || p < -4)
									num_wu++;	//真实白点区域
							}
						}
						//左右边缘
						if (x_lt_sec < 20 || (img_gray.cols - x_rt_sec < 20))
						{
							Mat  col, row;
							double m, n, p;
							for (int row_line = 0; row_line < imageSuspected.rows; row_line++)//列进行编列  行遍历
							{
								row = imageSuspected.rowRange(row_line, row_line + 1).clone();
								m = mean(row)[0];

								p = m - mean_temp1;

								if (p > 3.3 || p < -3.3)
									num_wu++;	//真实白点区域
							}
						}


						//if (y_rt_sec < 20 || (img_gray.rows - y_lt_sec<20))
						//{
						   // Mat  col, row;
						   // double m, n, p, o, r;
						   // for (int col_line = 0; col_line < imageSuspected.cols - 2; col_line++)//列进行编列
						   // {
							  //  col = imageSuspected.colRange(col_line, col_line + 1).clone();
							  //  m = mean(col)[0];

							  //  col = imageSuspected.colRange(col_line + 1, col_line + 2).clone();
							  //  n = mean(col)[0];

							  //  p = abs(n - m);

							  //  if (p > 1)
								 //   num_wu++;
						   // }
						//}

						//if (x_rt_sec < 20 || (img_gray.cols - x_lt_sec<20))
						//{
						   // Mat  col, row;
						   // double m, n, p, o, r;
						   // for (int col_line = 0; col_line < imageSuspected.rows - 2; col_line++)//列进行编列
						   // {
							  //  col = imageSuspected.rowRange(col_line, col_line + 1).clone();
							  //  m = mean(col)[0];

							  //  col = imageSuspected.rowRange(col_line + 1, col_line + 2).clone();
							  //  n = mean(col)[0];

							  //  p = abs(n - m);

							  //  if (p > 1)
								 //   num_wu++;
						   // }
						//}

						//int num_wu = 1;
						if (num_wu > 0 || (y_lt_sec >= 20 && (img_gray.rows - y_rt_sec >= 20 && x_lt_sec >= 20 && (img_gray.cols - x_rt_sec >= 20))))
						{
							double Luminaceth;
							Mat imageBinary;
							adaptiveThreshold(imageSuspected, imageBinary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, yiwu_sec_size, yiwu_sec_th);
							if (x_lt < 200 || img_gray.cols - x_rt <= 200 || y_lt_sec <= 20 || img_gray.rows - y_rt <= 20)
							{
								//防止R角边缘误检
								int grayValueSum = 0;
								int pixelsNum = 0;
								double meanIn, meanOut, meanAll;
								Mat maskGray;
								Mat mask1 = th1(Rect(x_lt_sec, y_lt_sec, x_rt_sec - x_lt_sec, y_rt_sec - y_lt_sec));
								bitwise_and(imageSuspected, mask1, maskGray);
								for (int i = 0; i < maskGray.cols; i++)
								{
									for (int j = 0; j < maskGray.rows; j++)
									{
										if (maskGray.at<uchar>(j, i) > 0)
										{
											grayValueSum += maskGray.at<uchar>(j, i);
											pixelsNum++;
										}
									}
								}
								meanIn = grayValueSum / (float)pixelsNum;

								grayValueSum = 0;
								pixelsNum = 0;
								bitwise_and(imageSuspected, ~mask1, maskGray);
								for (int i = 0; i < maskGray.cols; i++)
								{
									for (int j = 0; j < maskGray.rows; j++)
									{
										if (maskGray.at<uchar>(j, i) > 0)
										{
											grayValueSum += maskGray.at<uchar>(j, i);
											pixelsNum++;
										}
									}
								}
								meanOut = grayValueSum / (float)pixelsNum;

								grayValueSum = 0;
								pixelsNum = 0;
								for (int i = 0; i < imageSuspected.cols; i++)
								{
									for (int j = 0; j < imageSuspected.rows; j++)
									{
										if (imageSuspected.at<uchar>(j, i) > 0)
										{
											grayValueSum += imageSuspected.at<uchar>(j, i);
											pixelsNum++;
										}
									}
								}
								meanAll = grayValueSum / (float)pixelsNum;

								Luminaceth = meanOut - meanIn;
							}
							//double  defect_areath = meanIn - meanOut;//缺陷区域与周围灰度差值(整体性)
							else
							{
								//异物缺陷亮度阈值
								Mat tt;
								Mat mask1 = th1(Rect(x_lt_sec, y_lt_sec, x_rt_sec - x_lt_sec, y_rt_sec - y_lt_sec));
								double meanGrayin_Suspect = mean(imageSuspected, mask1)[0];
								bitwise_and(imageSuspected, mask1, tt);
								double meanGrayout_Suspect = mean(imageSuspected, ~mask1)[0];
								bitwise_and(imageSuspected, ~mask1, tt);
								Luminaceth = meanGrayout_Suspect - meanGrayin_Suspect;
							}

							if (area < 25 && Luminaceth < 9 && ceguangth_huichen > 1.5)		//如果缺陷本身面积与灰度差都很大，不用排侧光黑色区域，否则进行侧光黑色区域排除
								continue;

							vector<vector<Point>> contours7;
							vector<vector<Point>> contours8;
							findContours(imageBinary, contours7, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
							vector<Rect> BoundRect_Area_Sec(contours7.size());

							Mat tempBinary;
							if (imageBinary.rows > 4 && imageBinary.cols > 4)
								tempBinary = imageBinary(Rect(2, 2, imageBinary.cols - 4, imageBinary.rows - 4));
							else
								tempBinary = imageBinary;
							findContours(tempBinary, contours8, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);

							if (area <= 300 && (contours7.size() < 15 || contours8.size() < 15) || area <= 500 && area > 300 && (contours7.size() < 20 || contours8.size() < 20) || area > 500 && (contours7.size() < 25 || contours8.size() < 25))//彩色相机网格效应对此处有影响
							{
								for (vector<int>::size_type i7 = 0; i7 < contours7.size(); i7++)
								{
									BoundRect_Area_Sec[i7] = boundingRect(Mat(contours7[i7]));
									int w_last = BoundRect_Area_Sec[i7].width;
									int h_last = BoundRect_Area_Sec[i7].height;
									int X_1_last = BoundRect_Area_Sec[i7].tl().x;//矩形左上角X坐标值
									int Y_1_last = BoundRect_Area_Sec[i7].tl().y;//矩形左上角Y坐标值
									int X_2_last = BoundRect_Area_Sec[i7].br().x;//矩形右下角X坐标值
									int Y_2_last = BoundRect_Area_Sec[i7].br().y;//矩形右下角Y坐标值

									Mat Crop_Image_last = imageBinary(Rect(X_1_last, Y_1_last, X_2_last - X_1_last, Y_2_last - Y_1_last));

									//Mat TempImage = imageBinary.clone();
									//for (int i = 0; i < TempImage.cols; i++)//cols是列，rows是行
									//{
									//	for (int j = 0; j < TempImage.rows; j++)
									//	{
									//		if (i < X_2_last && i >= X_1_last && j < Y_2_last && j >= Y_1_last)
									//		{
									//			TempImage.at<uchar>(j, i) = imageBinary.at<uchar>(j, i);
									//		}
									//		else {
									//			TempImage.at<uchar>(j, i) = 0;
									//		}
									//	}
									//}
									int Secscreen_Expand = 1;
									int X_1_end = X_1_last - Secscreen_Expand;
									if (X_1_end < 0)
									{
										X_1_end = 0;
									}
									int Y_1_end = Y_1_last - Secscreen_Expand;
									if (Y_1_end < 0)
									{
										Y_1_end = 0;
									}
									int X_2_end = X_2_last + Secscreen_Expand;
									if (X_2_end > imageBinary.cols - 1)
									{
										//x_rt_sec = gray_ceguang.cols;
										X_2_end = imageBinary.cols - 1;
									}
									int Y_2_end = Y_2_last + Secscreen_Expand;
									if (Y_2_end > imageBinary.rows - 1)
									{
										//y_rt_sec = gray_ceguang.rows;
										Y_2_end = imageBinary.rows - 1;
									}
									Mat TempCeguang0 = TempCeguang(Rect(X_1_end, Y_1_end, X_2_end - X_1_end, Y_2_end - Y_1_end));
									Mat TempImage0 = imageBinary(Rect(X_1_end, Y_1_end, X_2_end - X_1_end, Y_2_end - Y_1_end));
									//                                   Mat TempCeguang1;
									//                                   bitwise_and(TempCeguang0, TempImage0, TempCeguang1);
									double ceguang1_In = mean(TempCeguang0, TempImage0)[0];
									//                                   bitwise_and(TempCeguang0, ~TempImage0, TempCeguang1);
									double ceguang1_Out = mean(TempCeguang0, ~TempImage0)[0];
									double differ = ceguang1_Out - ceguang1_In;

									if ((area < 130 && area >= 50 && differ <= 6.6)
										|| (area < 50 && area >= 10 && differ < 3.1)
										|| (area > 0 && area < 10 && differ < 1.8)
										|| (area >= 130)
										|| (Luminaceth > 11.2 && area > 80)
										|| (area < 50 && area >= 10 && differ > 6 && differ < 11.2))
									{
										//double area7 = countNonZero(Crop_Image_last);

										//double meanGrayin_Suspect = mean(imageSuspected, TempImage)[0];
										//double meanGrayout_Suspect = mean(imageSuspected, ~TempImage)[0];
										//Luminaceth = meanGrayout_Suspect - meanGrayin_Suspect;

										//double area7 = contourArea(contours7[i7]);
										if (area > 25 && area < yiwu_sec_area_upper && Luminaceth >= 5.9 || area >15 && area <= 25 & Luminaceth >= 7.5 || area <= 15 && area >= 12 && Luminaceth >= 10 || area < 12 && area >= yiwu_sec_area_lower && Luminaceth >= 13)
											//if (area7 > 25 && area7 < yiwu_sec_area_upper && Luminaceth >= yiwu_lighth_BigArea || area7 <= 25 && area7 >= yiwu_sec_area_lower && Luminaceth >= yiwu_lighth_SmallArea)
											//if (area7 >= yiwu_sec_area_lower && area7 < yiwu_sec_area_upper && Luminaceth >= yiwu_lighth_BigArea)
										{
											result = true;

											CvPoint top_lef44 = cvPoint(X_1, Y_1);
											CvPoint bottom_right44 = cvPoint(X_2, Y_2);
											rectangle(white_yiwu, top_lef44, bottom_right44, Scalar(0, 255, 0), 1, 8, 0);

											int x_lt_high = top_lef44.x - 50;
											if (x_lt_high < 0)
											{
												x_lt_high = 0;
											}
											int y_lt_high = top_lef44.y - 50;
											if (y_lt_high < 0)
											{
												y_lt_high = 0;
											}
											int x_rt_high = bottom_right44.x + 50;
											if (x_rt_high > white_yiwu.cols - 1)
											{
												x_rt_high = white_yiwu.cols;
											}
											int y_rt_high = bottom_right44.y + 50;
											if (y_rt_high > white_yiwu.rows - 1)
											{
												y_rt_high = white_yiwu.rows;
											}

											CvPoint top_lef4 = cvPoint(x_lt_high, y_lt_high);
											CvPoint bottom_right4 = cvPoint(x_rt_high, y_rt_high);

											rectangle(white_yiwu, top_lef4, bottom_right4, Scalar(0, 255, 0), 5, 8, 0);

											string information = "th:" + to_string(Luminaceth) + "area " + to_string(area);	//信息数据查看
											putText(white_yiwu, information, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);
										}
									}
								}
							}

						}

					}
				}
			}
		}

		if (result == true)
		{
			*causecolor = "背光异物";
			*mresult = white_yiwu;
		}
		return result;
	}

	bool ForeignBodyZZH(Mat white_yiwu, Mat ceguang, Mat Original, Mat* mresult, String* causecolor) {
		double yiwu_pre_size = 51;
		double yiwu_pre_th = 5.5;
		double yiwu_area_lower = 0.6;//原25
		double yiwu_area_upper = 2200;//原6000
		double yiwu_sec_size = 47;
		double yiwu_sec_th = 6;
		double yiwu_sec_area_lower = 9;
		double yiwu_sec_area_upper = 2200;
		double yiwu_lighth_BigArea = 5.9;       //暂时没用
		double yiwu_lighth_SmallArea = 10;      //暂时没用

		double yiwu_MaxLight = 50;
		if (yiwu_MaxLight >= 100)
			yiwu_MaxLight = 100;
		if (yiwu_MaxLight <= 10)
			yiwu_MaxLight = 10;
		double yiwu_light1 = 0.118 * yiwu_MaxLight;
		double yiwu_light2 = 0.15 * yiwu_MaxLight;
		double yiwu_light3 = 0.2 * yiwu_MaxLight;
		double yiwu_light4 = 0.26 * yiwu_MaxLight;
		bool result = false;
		Mat imageOriginC = ceguang.clone();	  //侧光源图像
		Mat imageOriginB = Original.clone();//背光源图像
		Mat imageBinaryB;					  //背光二值图
		Mat imageRectC, imageRectB;			//测光和背光的特征矩形区域

		int RectExpand = 3;					//矩形区域
		int boundary = 40;					//边界气泡排除
		adaptiveThresholdCustom(imageOriginB, imageBinaryB, 255, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, yiwu_pre_size, yiwu_pre_th, 1);//自适应阈值二值化
		Mat imageBinaryori = imageBinaryB.clone();
		Mat imageBinaryLook;
		//Mat  kernel = (Mat_<char>(3, 3) << 1, 1, 1, 1, 1, 1, -1, -1, -1);
		Mat element3_3 = getStructuringElement(MORPH_RECT, Size(3, 3));		//闭操作结构元素（3*11）
		Mat element1_1 = getStructuringElement(MORPH_RECT, Size(2, 2));
		erode(imageBinaryB, imageBinaryLook, element3_3);//dilate是膨胀，erode是腐蚀
		erode(imageBinaryB, imageBinaryB, element3_3);//dilate是膨胀，erode是腐蚀


	  //morphologyEx(imageBinaryB, imageBinaryLook, CV_MOP_CLOSE, element);

		//morphologyEx(imageBinaryB, imageBinaryB, CV_MOP_CLOSE, element);   //闭运算形态学操作。可以减少噪点

		///遍历处理后的图像所有轮廓
		vector<vector<Point>> contours;			//保存寻找的轮廓变量
		findContours(imageBinaryB, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);	//寻找二值图轮廓
		vector<Rect> boundRect(contours.size());										//保存轮廓
		vector<Rect> boundRect_area(contours.size());									//保存轮廓的面积
		vector<int> importContours(contours.size());									//保存疑似异物区域的索引
		vector<double> importArea(contours.size());										//保存疑似异物区域的面积
		vector<double> grayDifferB(contours.size()), grayDifferC(contours.size());										//保存疑似异物区域灰度差


		float w, h;
		vector<int> X_left(contours.size()), Y_left(contours.size()), X_right(contours.size()), Y_right(contours.size());//保存疑似异物区域外接矩形坐标值
		int  contorIndex = 0;				//索引值

		cv::Mat stdDevC, stdDevB;			//B为背光方差 C为侧光方差矩阵
		cv::Mat meanGrayC, meanGrayB;
		vector<double>  stdDevCnum(contours.size()), stdDevBnum(contours.size());		//侧光和背光方差值
		Mat maskBinaryB;					//二值化区域掩膜	
		double meanGrayInC, meanGrayInB, meanGrayOutC, meanGrayOutB;//背、测光轮廓内外区域平均灰度值,背、侧内外灰度差

		double minInB = 0, minOutB = 0, maxInB = 0, maxOutB = 0;
		double minInC = 0, minOutC = 0, maxInC = 0, maxOutC = 0;
		double minRectC = 0, maxRectC = 0, minRectB = 0, maxRectB = 0;
		vector<double> normolationB(contours.size()), normolationC(contours.size());
		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);						//轮廓面积获取

			if (area > yiwu_area_lower && area < yiwu_sec_area_upper) //(轮廓面积在6-2200之间)读取
			{

				boundRect_area[i] = boundingRect(Mat(contours[i]));//圈出区域的矩形

				w = boundRect_area[i].width;//获得外接区域的长宽
				h = boundRect_area[i].height;
				int X_1, X_2, Y_1, Y_2;
				X_1 = boundRect_area[i].tl().x - RectExpand;//矩形左上角X坐标值
				if (X_1 < 0) { continue; };
				Y_1 = boundRect_area[i].tl().y - RectExpand;//矩形左上角Y坐标值
				if (Y_1 < 0) { continue; };
				X_2 = boundRect_area[i].br().x + RectExpand;//矩形右下角X坐标值
				if (X_2 > imageOriginC.cols - 1) { continue; };//cols是行=3000，y方向
				Y_2 = boundRect_area[i].br().y + RectExpand;//矩形右下角Y坐标值
				if (Y_2 > imageOriginC.rows - 1) { continue; };//rows是列 =1500列

				double longShortRatio = max(h / w, w / h);//最大长宽比率

				//if (longShortRatio < 5 && min(w, h) >= 2 && max(w, h) < 120)	//长宽比小于5且 最小宽度大于2个像素 最大长度小于120像素 
				if (longShortRatio < 5)//长宽比小于3
				{
					//计算侧光图像缺陷中心和缺陷外围灰度差
					Mat imageRectC = imageOriginC(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
					Mat imageRectB = imageOriginB(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
					maskBinaryB = imageBinaryB(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));//获取缺陷位置的二值图//内部比外部灰度值大
					//删除了排除上下气泡等方法
					//屏幕内部时排灰尘与气泡
					if (X_1 < boundary || Y_1 < boundary || X_2>imageOriginC.cols - boundary || Y_2>imageOriginB.rows - boundary) {
						continue;					//当缺陷轮廓在边界时候跳出本次轮廓循环
					}
					Mat imageGetInB, imageGetInC, imageGetOutB, imageGetOutC;

					//获取目标矩形区域的图像
					bitwise_and(imageRectC, maskBinaryB, imageGetInC);
					bitwise_and(imageRectC, ~maskBinaryB, imageGetOutC);
					bitwise_and(imageRectB, maskBinaryB, imageGetInB);
					bitwise_and(imageRectB, ~maskBinaryB, imageGetOutB);


					//获取灰度值

					meanGrayInC = mean(imageRectC, maskBinaryB)[0];//取测光轮廓区域的灰度值
					meanGrayOutC = mean(imageRectC, ~maskBinaryB)[0];
					meanGrayInB = mean(imageRectB, maskBinaryB)[0];//背光灰度值内部
					meanGrayOutB = mean(imageRectB, ~maskBinaryB)[0];//背光灰度值外部

					//获取极值
					minMaxLoc(imageGetInC, &minInC, &maxInC);//得到背光区域最大的值
					minMaxLoc(imageGetOutC, &minOutC, &maxOutC);
					minMaxLoc(imageGetInB, &minInB, &maxInB);//得到背光区域最大的值	
					minMaxLoc(imageGetOutB, &minOutB, &maxOutB);


					//侧光图上的灰度均值方差排除气泡等干扰
					cv::meanStdDev(imageRectC, meanGrayC, stdDevC);//侧光
					cv::meanStdDev(imageRectB, meanGrayB, stdDevB);//背光
					//获取极值
					minMaxLoc(imageRectC, &minRectC, &maxRectC);
					minMaxLoc(imageRectB, &minRectB, &maxRectB);
					double grayDifferMidC, grayDifferMidB;

					grayDifferMidB = meanGrayInB - meanGrayOutB;
					grayDifferMidC = meanGrayInC - meanGrayOutC;
					if (abs(grayDifferMidC) > abs(grayDifferMidB)		//侧光灰度差大于背光且侧光灰度差(加不加侧光大于0后续看)
						|| stdDevC.at<double>(0, 0) > stdDevB.at<double>(0, 0)
						)//侧光灰度差大于背光
					{
						continue;//该条件直接退出
					}

					if (true)
					{
						//获取背光图下的特征区域轮廓和矩形
						stdDevBnum[contorIndex] = stdDevB.at<double>(0, 0);
						stdDevCnum[contorIndex] = stdDevC.at<double>(0, 0);
						grayDifferC[contorIndex] = grayDifferMidC;
						grayDifferB[contorIndex] = grayDifferMidB;
						normolationB[contorIndex] = grayDifferMidB / (maxRectB - minRectB);
						normolationC[contorIndex] = grayDifferMidC / (maxRectC - minRectC);
						importContours[contorIndex] = i;
						importArea[contorIndex] = area;
						X_left[contorIndex] = X_1;
						Y_left[contorIndex] = Y_1;
						X_right[contorIndex] = X_2;
						Y_right[contorIndex] = Y_2;

						//灰度差显示

						contorIndex = contorIndex + 1;
					}
				}
				else {
				}
			}
		}
		Mat MainBImgShow, MainCImgShow;
		//画出感兴趣的区域
		MainBImgShow = imageOriginB.clone();
		MainCImgShow = ceguang.clone();
		drawContours(MainCImgShow, contours, 1, (255, 255, 255), 1);
		drawContours(MainBImgShow, contours, 1, (255, 0, 0), 1);

		//////////////////////////////排除R角误检////////////////////////////////////////
		for (int index = 0; index < contorIndex; index++) {
			int x_lt = X_left[index];
			int x_rt = X_right[index];
			int y_lt = Y_left[index];
			int y_rt = Y_right[index];
			Mat imageRectO = Original(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));//获得这个区域的特征		imageRectB
		//减去了

			int area = importArea[index];
			if (grayDifferC[index] > 0 && abs(grayDifferC[index]) > 2 * abs(grayDifferB[index]) || grayDifferC[index] >= 5) {
				continue;
			}//大于5

			if (stdDevCnum[index] < stdDevBnum[index] && abs(grayDifferC[index]) / 10 <= abs(grayDifferB[index]) / 10
				|| grayDifferC[index]>0 && grayDifferC[index] / area < stdDevCnum[index])//侧光灰度差小于1 背光方差大于侧光
			{
				result = true;

				CvPoint top_lef44 = cvPoint(X_left[index], Y_left[index]);
				CvPoint bottom_right44 = cvPoint(X_right[index], Y_right[index]);
				rectangle(white_yiwu, top_lef44, bottom_right44, Scalar(0, 255, 0), 1, 8, 0);

				int x_lt_high = top_lef44.x - 50;
				if (x_lt_high < 0)
				{
					x_lt_high = 0;
				}
				int y_lt_high = top_lef44.y - 50;
				if (y_lt_high < 0)
				{
					y_lt_high = 0;
				}
				int x_rt_high = bottom_right44.x + 50;
				if (x_rt_high > white_yiwu.cols - 1)
				{
					x_rt_high = white_yiwu.cols;
				}
				int y_rt_high = bottom_right44.y + 50;
				if (y_rt_high > white_yiwu.rows - 1)
				{
					y_rt_high = white_yiwu.rows;
				}

				CvPoint top_lef4 = cvPoint(x_lt_high, y_lt_high);
				CvPoint bottom_right4 = cvPoint(x_rt_high, y_rt_high);

				rectangle(white_yiwu, top_lef4, bottom_right4, Scalar(0, 255, 0), 5, 8, 0);

				string information = "th:" + to_string(grayDifferB[index]) + "area " + to_string(area);	//信息数据查看
				putText(white_yiwu, information, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);

			}
		}
		if (result == true)
		{
			*causecolor = "背光异物";
			*mresult = white_yiwu;
		}
		return result;
	}

	void unevenLightCompensate(Mat& image, Mat& imageOut, int blockSize)
	{
		if (image.channels() == 3) cvtColor(image, image, 7);
		double average = mean(image)[0];
		int rows_new = ceil(double(image.rows) / double(blockSize));
		int cols_new = ceil(double(image.cols) / double(blockSize));
		Mat blockImage;
		blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
		for (int i = 0; i < rows_new; i++)
		{
			for (int j = 0; j < cols_new; j++)
			{
				int rowmin = i * blockSize;
				int rowmax = (i + 1) * blockSize;
				if (rowmax > image.rows) rowmax = image.rows;
				int colmin = j * blockSize;
				int colmax = (j + 1) * blockSize;
				if (colmax > image.cols) colmax = image.cols;
				Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
				double temaver = mean(imageROI)[0];
				blockImage.at<float>(i, j) = temaver;//为每一个区域获取均值
			}
		}
		blockImage = blockImage - average;//获取全局均值和分块均值的差值
		Mat blockImage2;
		resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
		Mat image2;
		image.convertTo(image2, CV_32FC1);
		Mat dst = image2 - blockImage2;

		dst.convertTo(imageOut, CV_8UC1);

	}
	void improHist(Mat& imporImage, Mat& img_grayHist, Mat imageOrigin) {

		//对ROI取二值化图7
		Mat thMat;
		threshold(imageOrigin, thMat, 40, 255, CV_THRESH_BINARY);
		//获取全局均值
		double meanVal = mean(imporImage, thMat)[0];

		// 获取线性变换参数
		double alpha1 = 0.8;//0.8
		double beta1 = 0;

		// 增强 srcMean - 10 与 230 
		double r1 = 100;//第一个灰度分级//130-1 2
		double r2 = meanVal;//第二个分度分级
		double s1 = alpha1 * r1;//拉伸后中间距离的下限
		double s2 = 255;//拉伸后中间长度的上限 
		double alpha2 = (255 - alpha1 * r1) / (r2 - r1);//
		double beta2 = -alpha2 * r1 + alpha1 * r1;
		img_grayHist = imporImage.clone();

		for (int r = 0; r < imporImage.rows; r++)
		{
			for (int c = 0; c < imporImage.cols; c++) {
				uchar temp = imporImage.at<uchar>(r, c);
				if (temp <= r1)
				{
					img_grayHist.at<uchar>(r, c) = saturate_cast<uchar>(temp * alpha1); //alpha = 0.5, beta = 0
				}
				else if (r1 < temp && temp < r2)
				{
					img_grayHist.at<uchar>(r, c) = saturate_cast<uchar>(temp * alpha2 + beta2);
				}
				else
				{
					img_grayHist.at<uchar>(r, c) = saturate_cast<uchar>(255);
				}
			}
		}
	}

	bool ForeignBodyDeep(Mat white_yiwu, Mat ceguang, Mat Original, Mat* mresult, String* causecolor)//灰度检测S
	{
		double yiwu_pre_size = 11;
		double yiwu_pre_th = 5.5;
		double yiwu_area_lower = 0.6;//原25
		double yiwu_area_upper = 2200;//原6000
		double yiwu_sec_size = 47;
		double yiwu_sec_th = 6;
		double yiwu_sec_area_lower = 9;
		double yiwu_sec_area_upper = 2200;
		double yiwu_lighth_BigArea = 5.9;       //暂时没用
		double yiwu_lighth_SmallArea = 10;      //暂时没用

		double yiwu_MaxLight = 50;
		if (yiwu_MaxLight >= 100)
			yiwu_MaxLight = 100;
		if (yiwu_MaxLight <= 10)
			yiwu_MaxLight = 10;
		double yiwu_light1 = 0.118 * yiwu_MaxLight;
		double yiwu_light2 = 0.15 * yiwu_MaxLight;
		double yiwu_light3 = 0.2 * yiwu_MaxLight;
		double yiwu_light4 = 0.26 * yiwu_MaxLight;
		bool result = false;
		Mat imageOriginC = ceguang.clone();	  //侧光源图像
		Mat imageOriginB = Original.clone();//背光源图像
		int RectExpand = 5;					//外接矩形区域
		int boundary = 5;					//边界气泡排除
		//灰度增强
		Mat imageMeamB;

		unevenLightCompensate(imageOriginB, imageMeamB, 12);

		//图像增强
	//使用直方图增强
		Mat imangMeanImproB;
		improHist(imageMeamB, imangMeanImproB, imageOriginB);

		//灰度直方图显示
		//Histogram1D hsit;
		//Mat histOriImageB = hsit.getHistogramImage(imangOriImproB);//画出原始图像灰度分布



		Mat filterGas;
		GaussianBlur(imangMeanImproB, filterGas, Size(5, 5), 1, 1);
		Mat imageBinaryB;
		adaptiveThresholdCustom(filterGas, imageBinaryB, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 10, 1);//自适应阈值二值化CV_THRESH_BINARY_INV



		///遍历处理后的图像所有轮廓
		vector<vector<Point>> contours;			//保存寻找的轮廓变量
		findContours(imageBinaryB, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);	//寻找二值图轮廓
		vector<Rect> boundRect(contours.size());										//保存轮廓
		vector<Rect> boundRect_area(contours.size());									//保存轮廓的面积
		vector<int> importContours(contours.size());									//保存疑似异物区域的索引
		vector<double> importArea(contours.size());										//保存疑似异物区域的面积
		vector<double> grayDifferB(contours.size()), grayDifferC(contours.size());										//保存疑似异物区域灰度差
		vector<Mat> d;

		float w, h;
		vector<int> X_left(contours.size()), Y_left(contours.size()), X_right(contours.size()), Y_right(contours.size());//保存疑似异物区域外接矩形坐标值
		int  contorIndex = 0;				//索引值

		cv::Mat stdDevC, stdDevB;			//B为背光方差 C为侧光方差矩阵
		cv::Mat meanGrayC, meanGrayB;
		vector<double>  stdDevCnum(contours.size()), stdDevBnum(contours.size());		//侧光和背光方差值
		Mat maskBinaryB;					//二值化区域掩膜	
		double meanGrayInC, meanGrayInB, meanGrayOutC, meanGrayOutB;//背、测光轮廓内外区域平均灰度值,背、侧内外灰度差

		double minInB = 0, minOutB = 0, maxInB = 0, maxOutB = 0;
		double minInC = 0, minOutC = 0, maxInC = 0, maxOutC = 0;
		double minRectC = 0, maxRectC = 0, minRectB = 0, maxRectB = 0;
		vector<double> normolationB(contours.size()), normolationC(contours.size());
		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);						//轮廓面积获取

			if (area > yiwu_area_lower && area < yiwu_sec_area_upper) //(轮廓面积在6-2200之间)读取
			{
				boundRect_area[i] = boundingRect(Mat(contours[i]));//圈出区域的矩形
				w = boundRect_area[i].width;//获得外接区域的长宽
				h = boundRect_area[i].height;
				int X_1, X_2, Y_1, Y_2;
				X_1 = boundRect_area[i].tl().x - RectExpand;//矩形左上角X坐标值
				if (X_1 < 0) { continue; };
				Y_1 = boundRect_area[i].tl().y - RectExpand;//矩形左上角Y坐标值
				if (Y_1 < 0) { continue; };
				X_2 = boundRect_area[i].br().x + RectExpand;//矩形右下角X坐标值
				if (X_2 > 2998) { continue; };//cols是行=3000，y方向列数
				Y_2 = boundRect_area[i].br().y + RectExpand;//矩形右下角Y坐标值
				if (Y_2 > 1498) { continue; };//rows是列 =1500列

				double longShortRatio = max(h / w, w / h);//最大长宽比率
				//if (longShortRatio < 5 && min(w, h) >= 2 && max(w, h) < 120)	//长宽比小于5且 最小宽度大于2个像素 最大长度小于120像素 
				if (longShortRatio < 5)//长宽比小于3
				{
					//计算侧光图像缺陷中心和缺陷外围灰度差
					Mat imageRectC = ceguang(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
					Mat imageRectB = Original(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));//取增强后的区域
					maskBinaryB = imageBinaryB(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));//获取缺陷位置的二值图//内部比外部灰度值大
					//删除了排除上下气泡等方法
					//屏幕内部时排灰尘与气泡
					if (X_1 < boundary || Y_1 < boundary || X_2>imageOriginC.cols - boundary || Y_2>imageOriginB.rows - boundary) {
						continue;					//当缺陷轮廓在边界时候跳出本次轮廓循环
					}
					Mat imageGetInB, imageGetInC, imageGetOutB, imageGetOutC;

					//获取目标矩形区域的图像
					bitwise_and(imageRectC, maskBinaryB, imageGetInC);
					bitwise_and(imageRectC, ~maskBinaryB, imageGetOutC);
					bitwise_and(imageRectB, maskBinaryB, imageGetInB);
					bitwise_and(imageRectB, ~maskBinaryB, imageGetOutB);


					//获取灰度值

					meanGrayInC = mean(imageRectC, maskBinaryB)[0];//取测光轮廓区域的灰度值
					meanGrayOutC = mean(imageRectC, ~maskBinaryB)[0];
					meanGrayInB = mean(imageRectB, maskBinaryB)[0];//背光灰度值内部
					meanGrayOutB = mean(imageRectB, ~maskBinaryB)[0];//背光灰度值外部

					//获取极值
					minMaxLoc(imageGetInC, &minInC, &maxInC);//得到背光区域最大的值
					minMaxLoc(imageGetOutC, &minOutC, &maxOutC);
					minMaxLoc(imageGetInB, &minInB, &maxInB);//得到背光区域最大的值	
					minMaxLoc(imageGetOutB, &minOutB, &maxOutB);


					//侧光图上的灰度均值方差排除气泡等干扰
					cv::meanStdDev(imageRectC, meanGrayC, stdDevC);//侧光
					cv::meanStdDev(imageRectB, meanGrayB, stdDevB);//背光
					//获取极值
					minMaxLoc(imageRectC, &minRectC, &maxRectC);
					minMaxLoc(imageRectB, &minRectB, &maxRectB);

					double grayDifferMidC, grayDifferMidB;
					grayDifferMidB = meanGrayInB - meanGrayOutB;
					grayDifferMidC = meanGrayInC - meanGrayOutC;
					if (maxRectC - minRectC > 2 * (maxRectB - minRectB)) { continue; }
					if (grayDifferMidB > -2) { continue; }

					if (abs(grayDifferMidC) <= 1)
					{
						if (area < 30) { continue; }
					}//灰度差绝对值小于1，判定良品
					else if (grayDifferMidC > 1) {//侧光灰度值大于1 的层次
						if (grayDifferMidC > abs(grayDifferMidB)		//侧光灰度差大于背光绝对值
							|| stdDevC.at<double>(0, 0) - stdDevB.at<double>(0, 0) > 0.3//侧光方差比背光方差大0.3
							|| maxRectC - minRectC > maxRectB - minRectB//侧光极值大于背光极值
							|| area / 10 <= grayDifferMidC
							|| grayDifferMidB < 30
							)//侧光灰度差大于背光
						{
							continue;//该条件直接退出
						}
					}
					else {//侧光小于-1

					}



					if (true)
					{
						//获取背光图下的特征区域轮廓和矩形
						stdDevBnum[contorIndex] = stdDevB.at<double>(0, 0);
						stdDevCnum[contorIndex] = stdDevC.at<double>(0, 0);
						grayDifferC[contorIndex] = grayDifferMidC;
						grayDifferB[contorIndex] = grayDifferMidB;
						normolationB[contorIndex] = grayDifferMidB / (maxRectB - minRectB);
						normolationC[contorIndex] = grayDifferMidC / (maxRectC - minRectC);
						importContours[contorIndex] = i;
						importArea[contorIndex] = area;
						X_left[contorIndex] = X_1;
						Y_left[contorIndex] = Y_1;
						X_right[contorIndex] = X_2;
						Y_right[contorIndex] = Y_2;

						//灰度差显示

						contorIndex = contorIndex + 1;
					}
				}
				else {
				}
			}
		}
		Mat MainBImgShow, MainCImgShow;
		//画出感兴趣的区域  
		MainBImgShow = Original.clone();
		MainCImgShow = ceguang.clone();
		drawContours(MainCImgShow, contours, 1, (255, 255, 255), 1);
		drawContours(MainBImgShow, contours, 1, (255, 0, 0), 1);


		//////////////////////////////排除R角误检////////////////////////////////////////
		for (int index = 0; index < contorIndex; index++)
		{
			int x_lt = X_left[index];
			int x_rt = X_right[index];
			int y_lt = Y_left[index];
			int y_rt = Y_right[index];
			Mat imageRectO = Original(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));//获得这个区域的特征		imageRectB
		//减去了

			int area = importArea[index];
			if (true)
			{
				result = true;

				CvPoint top_lef44 = cvPoint(X_left[index], Y_left[index]);
				CvPoint bottom_right44 = cvPoint(X_right[index], Y_right[index]);
				rectangle(white_yiwu, top_lef44, bottom_right44, Scalar(0, 255, 0), 1, 8, 0);

				int x_lt_high = top_lef44.x - 50;
				if (x_lt_high < 0)
				{
					x_lt_high = 0;
				}
				int y_lt_high = top_lef44.y - 50;
				if (y_lt_high < 0)
				{
					y_lt_high = 0;
				}
				int x_rt_high = bottom_right44.x + 50;
				if (x_rt_high > white_yiwu.cols - 1)
				{
					x_rt_high = white_yiwu.cols;
				}
				int y_rt_high = bottom_right44.y + 50;
				if (y_rt_high > white_yiwu.rows - 1)
				{
					y_rt_high = white_yiwu.rows;
				}

				CvPoint top_lef4 = cvPoint(x_lt_high, y_lt_high);
				CvPoint bottom_right4 = cvPoint(x_rt_high, y_rt_high);

				rectangle(white_yiwu, top_lef4, bottom_right4, Scalar(0, 255, 0), 5, 8, 0);
				string information = "th:" + to_string(grayDifferB[index]) + "area " + to_string(area);	//信息数据查看
				putText(white_yiwu, information, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2, 8, 0);

			}
		}
		if (result == true)
		{
			*causecolor = "背光异物";
			*mresult = white_yiwu;
		}
		return result;
	}





	/*=========================================================
	* 函 数 名: Scratch
	* 功能描述: 屏幕划伤缺陷判断
	* 输入：左右侧面相机白底图像及侧光整张图像
	* 输出：划伤缺陷检测结果图像和检测Result
	* 时间：2021年2月25日
	* 其他：
	=========================================================*/
	bool Scratch(Mat white, Mat ceguang, Mat* mresult, String* causecolor)
	{

		bool result = false;
		double val1 = mean(white)[0];
		double val2 = mean(ceguang)[0];
		int length = 100;
		Mat img_gray = white.clone();
		Mat img_ceguang = ceguang.clone();
		Mat Filer;
		medianBlur(img_gray, Filer, 3);//中值滤波滤除椒盐噪声,缺点耗时26毫秒 奇数半径越大效果越强
		Mat th1;
		adaptiveThresholdCustom(Filer, th1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 83, -3, 1);

		int shuidi_length = 160;
		int part = img_gray.rows / 2 - shuidi_length;
		//针对边界位置取原图的边界
		Mat img_top = img_gray(Rect(0, 0, img_gray.cols - 1, length));
		Mat img_bottom = img_gray(Rect(0, img_gray.rows - length, img_gray.cols - 1, length));
		Mat img_left = img_gray(Rect(0, 0, length, img_gray.rows - 1));
		Mat img_right = img_gray(Rect(img_gray.cols - length, 0, length, img_gray.rows - 1));
		Mat img_right_light = img_gray(Rect(img_gray.cols - 150, part, 150, shuidi_length * 2));
		Mat img_tl_R = img_gray(Rect(0, 0, 150, 150));
		Mat img_bl_R = img_gray(Rect(0, 1350, 150, 150));
		Mat img_tr_R = img_gray(Rect(2850, 0, 150, 150));
		Mat img_br_R = img_gray(Rect(2849, 1349, 150, 150));

		Mat top_th, bottom_th, left_th, right_th, img_tl_R_th, img_bl_R_th, img_tr_R_th, img_br_R_th, img_right_light_th;

		//针对边界位置设置参数
		adaptiveThreshold(img_top, top_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
		adaptiveThreshold(img_bottom, bottom_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
		//adaptiveThreshold(img_left, left_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, -1);
		adaptiveThreshold(img_left, left_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
		adaptiveThreshold(img_right, right_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
		adaptiveThreshold(img_right_light, img_right_light_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
		adaptiveThreshold(img_tl_R, img_tl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_bl_R, img_bl_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3);
		adaptiveThreshold(img_tr_R, img_tr_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);
		adaptiveThreshold(img_br_R, img_br_R_th, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -3);

		//针对边界位置深拷贝
		top_th.copyTo(th1(Rect(0, 0, th1.cols - 1, length)));                    //上边界
		bottom_th.copyTo(th1(Rect(0, th1.rows - length, th1.cols - 1, length)));     //下边界
		left_th.copyTo(th1(Rect(0, 0, length, th1.rows - 1)));                   //左边界
		right_th.copyTo(th1(Rect(th1.cols - length, 0, length, th1.rows - 1)));      //右边界
		img_right_light_th.copyTo(th1(Rect(img_gray.cols - 150, part, 150, shuidi_length * 2)));      //右边界
		img_tl_R_th.copyTo(th1(Rect(0, 0, 150, 150)));                    //上边界
		img_bl_R_th.copyTo(th1(Rect(0, 1350, 150, 150)));     //下边界
		img_tr_R_th.copyTo(th1(Rect(2850, 0, 150, 150)));                   //左边界
		img_br_R_th.copyTo(th1(Rect(2849, 1349, 150, 150)));      //右边界

		//Mat th_mask = th1.clone();
		//Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
		//morphologyEx(th1, th1, CV_MOP_OPEN, element);

		//dilate(th1, th1, element);   //膨胀

		th1(Rect(0, 0, 15, th1.rows)) = uchar(0);            //屏蔽右侧15行，防止灯口误检白点
		th1(Rect(th1.cols - 10, 0, 10, th1.rows)) = uchar(0);            //屏蔽左侧10行，防止头部亮边误检为白点

		Mat th_result;
		//做掩膜
		threshold(img_gray, th_result, 25, 255, CV_THRESH_BINARY);
		bitwise_and(th_result, img_gray, img_gray);
		//闭运算,弥合内部空洞,连接相距很近的区域
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//闭操作结构元素
		morphologyEx(th1, th1, CV_MOP_CLOSE, element);   //闭运算形态学操作。可以减少噪点
		dilate(th1, th1, element);   //膨胀

		erode(th1, th1, element);

		vector<vector<Point>> contours;
		findContours(th1, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
		std::sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Rect> boundRect(contours.size());

		for (vector<int>::size_type i = 0; i < contours.size(); i++)
		{
			Mat temp_mask = Mat::zeros(th_result.rows, th_result.cols, CV_8UC1);
			drawContours(temp_mask, contours, i, 255, FILLED, 8);
			double area = contourArea(contours[i]);
			if (area >= 100 && area < 60000)
			{
				boundRect[i] = boundingRect(Mat(contours[i]));
				int w = boundRect[i].width;
				int h = boundRect[i].height;

				int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
				int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
				int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
				int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值

				RotatedRect rect = minAreaRect(contours[i]);
				double mw = rect.size.height;
				double mh = rect.size.width;
				double radio = max(mw / mh, mh / mw);


				int boder = 10;

				//长宽比排除
				if (radio < 2.7 || radio > 9.0 && (X_1 < boder || X_2 >(th1.cols - boder) || Y_1 < boder || Y_2 >(th1.rows - boder)))
				{
					continue;
				}


				int x_lt = X_1 - boder - int(w);
				int y_lt = Y_1 - boder - int(h);
				int x_rt = X_2 + boder + int(w);
				int y_rt = Y_2 + boder + int(h);
				if (x_lt < 0)
				{
					x_lt = 0;
				}
				if (y_lt < 0)
				{
					y_lt = 0;
				}
				if (x_rt > img_gray.size[1] - 1)
				{
					x_rt = img_gray.size[1] - 1;
				}
				if (y_rt > img_gray.size[0] - 1)
				{
					y_rt = img_gray.size[0] - 1;
				}

				Mat temp_ceguang = img_ceguang(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));
				Mat tempBinary = th1(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt)).clone();

				double mean_in_ceguang = mean(temp_ceguang, tempBinary)[0];
				double mean_out_ceguang = mean(temp_ceguang, ~tempBinary)[0];
				double ceguang_th = mean_in_ceguang - mean_out_ceguang;
				//排除划痕干扰限定条件1
				//if (abs(ceguang_th) >= filmscratch)
				//{
				//	continue;
				//}
				//排除白底图像上的明显气泡，计算缺陷外围的灰度方差，若存在气泡则灰度方差明显较大
				Mat tempgray_small = img_gray(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
				Mat tempbinary_small = th1(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));

				Mat tempGray = img_gray(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt)).clone();
				//Mat mask_bubble = th1(Rect(x_lt1, y_lt1, x_rt1 - x_lt1, y_rt1 - y_lt1));
				Mat mask_bubble = th1(Rect(x_lt, y_lt, x_rt - x_lt, y_rt - y_lt));
				cv::Mat meanGray1;
				cv::Mat stdDev1;
				cv::meanStdDev(tempGray, meanGray1, stdDev1, ~mask_bubble);
				//获取当前区域长宽比，用于得到相对于长宽比的标准差
				double virtualRadio = max((x_rt - x_lt) / (y_rt - y_lt), (y_rt - y_lt) / (x_rt - x_lt));
				double stddev1 = stdDev1.at<double>(0, 0);    // 30
				double val = stddev1 / radio * virtualRadio;
				if (stddev1 >= scratchbubbleth1 && (stddev1 / virtualRadio * 9) >= scratchbubbleth1) //将长宽比相对标准差对应系数改为9
				{
					continue;
				}

				Mat tempImage_Binary;
				double mean_In = mean(tempGray, tempBinary)[0];
				threshold(tempGray, tempImage_Binary, 30, 255, CV_THRESH_BINARY);
				double mean_All = mean(tempGray, tempImage_Binary)[0];
				threshold(tempGray, tempImage_Binary, mean_All - 20, 255, CV_THRESH_BINARY);
				bitwise_and(tempImage_Binary, ~tempBinary, tempBinary);
				double mean_Out = mean(tempGray, tempBinary)[0];
				double differ = mean_In - mean_Out;

				tempgray_small = img_gray(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
				tempbinary_small = th1(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
				double meanIn_small = mean(tempgray_small, tempbinary_small)[0];
				double meanOut_small = mean(tempgray_small, ~tempbinary_small)[0];
				double differ_small = meanIn_small - meanOut_small;
				if ((mean_All >= 160 && area >= 500 && differ >= 5.3) || differ >= 6.4 || differ_small >= 4.1)
				{
					result = true;
					CvPoint top_lef4 = cvPoint(X_1, Y_1);
					CvPoint bottom_right4 = cvPoint(X_2, Y_2);
					rectangle(white, top_lef4, bottom_right4, Scalar(0, 0, 0), 5, 8, 0);
					break;
				}
			}
		}
		if (result == true)
		{
			*causecolor = "划伤";
			*mresult = white;
		}
		return result;
	}

	//比较函数对象
	bool compareContourAreas(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2)
	{
		return (cv::contourArea(contour1) > cv::contourArea(contour2));
	}


	/*=========================================================
	 *@函 数 名: adaptiveThresholdCustom
	 *@功能描述: 自适应阈值分割实现图像二值化
	 *@param src          输入灰度图像
	 *@param dst          输出二值图像
	 *@param maxValue     输入满足阈值条件时像素取值
	 *@param method       计算局部均值方法
	 *@param type         输入阈值判断类型
	 *@param blockSize    卷积窗口大小(奇数)
	 *@param delta        输入偏移常量
	 *@param ratio        输入均值比例系数
	 *@备注说明：
	 =========================================================*/
	void adaptiveThresholdCustom(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double boundary)
	{
		CV_Assert(src.type() == CV_8UC1);               // 原图必须是单通道无符号8位,CV_Assert（）若括号中的表达式值为false，则返回一个错误信息
		CV_Assert(blockSize % 2 == 1 && blockSize > 1);	// 块大小必须大于1，并且是奇数,卷积核
		CV_Assert(maxValue > 0);                        //二值图像最大值

		Size size = src.size();							//源图像的尺寸
		Mat _dst(size, src.type());						//目标图像的尺寸
		Mat imagemean;	                                    //存放均值图像
		if (src.data != _dst.data)
			imagemean = _dst;

		int expend = (blockSize - 1) * 0.5;
		int expend_2 = expend * 2;

		int border_type = BORDER_REFLECT_101; //边界填充方式
		Mat src_Expand;	                   //对原图像进行边界扩充

		Mat topImage = src(Rect(0, 0, src.cols, 1));//上边界一行图像


		///cv::Scalar color = cv::mean(topImage)*0.5;//35-80之间均可以  该值需要确定
		//Scalar color = Scalar(50);//35-80之间均可以


		copyMakeBorder(src, src_Expand, expend, expend, expend, expend, border_type);

		if (method == ADAPTIVE_THRESH_MEAN_C)//
		{
			/*
			@param src 单通道灰度图
			@param dst 单通道处理后的图
			@param int类型的ddepth，输出图像的深度
			@param Size类型的ksize，内核的大小
			@param Point类型的anchor，表示锚点
			@param bool类型的normaliz,即是否归一化
			@param borderType 图像外部像素的某种边界模式
			*/
			boxFilter(src_Expand, imagemean, -1, Size(blockSize, blockSize));
			//	boxFilter(src_Expand, mean, src.type(), Size(blockSize, blockSize), Point(-1, -1), true, BORDER_CONSTANT);
		}
		else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
		{
			GaussianBlur(src_Expand, imagemean, Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
		}
		else
			CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");
		Rect ma = cv::Rect(expend, expend, src_Expand.cols - expend_2, src_Expand.rows - expend_2);
		imagemean = imagemean(ma); //删除扩充的图像边界


		int i, j;
		uchar imaxval = saturate_cast<uchar>(maxValue);	                       //将maxValue由double类型转换为uchar型 
		double idelta = delta;   //将idelta由double类型转换为int型
		if (src.isContinuous() && imagemean.isContinuous() && _dst.isContinuous())//判断数据是否连续 两个图像均连续
		{
			size.width *= size.height;
			size.height = 1;
		}

		for (i = 0; i < size.height; i++)
		{
			const uchar* sdata = src.data + src.step * i;		   //指向源图像  .step是返回内存参数 .data是返回数据首地址
			const uchar* mdata = imagemean.data + imagemean.step * i;		   //指向均值图
			uchar* ddata = _dst.data + _dst.step * i;	           //指向输出图
			for (j = 0; j < size.width; j++)
			{
				double Thresh = idelta;	        //阈值
				if (CV_THRESH_BINARY == type)	                    //S>T时为imaxval
				{
					ddata[j] = mdata[j] - sdata[j] > delta ? imaxval : 0;//如果均值大于原图像5个像素点，
				}
				else if (CV_THRESH_BINARY_INV == type)	            //S<T时为imaxval
				{
					ddata[j] = mdata[j] - sdata[j] > delta ? imaxval : 0;//如果均值大于原图像5个像素点，
				}
				else
					CV_Error(CV_StsBadFlag, "Unknown/unsupported threshold type");
			}
		}
		//	_dst = imagemean - src;
		dst = _dst.clone();
	}

	/*=========================================================
	* 函 数 名: Gabor7
	* 功能描述: gabor滤波
	=========================================================*/
	Mat Gabor7(Mat img_1)
	{
		Mat kernel1 = getGaborKernel(Size(5, 5), 1.1, CV_PI / 2, 1.0, 1.0, 0, CV_32F);//求卷积核
		float sum = 0.0;
		for (int i = 0; i < kernel1.rows; i++)
		{
			for (int j = 0; j < kernel1.cols; j++)
			{
				sum = sum + kernel1.ptr<float>(i)[j];
			}
		}
		Mat mmm = kernel1 / sum;
		Mat kernel2 = getGaborKernel(Size(5, 5), 1.1, 0, 1.0, 1.0, 0, CV_32F);
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
/** @brief 自适应二值化
*@param _src     要二值化的灰度图
*@param _dst     二值化后的图
*@param maxValue    二值化后要设置的那个值
*@param method 块计算的方法（ADAPTIVE_THRESH_MEAN_C 平均值，ADAPTIVE_THRESH_GAUSSIAN_C 高斯分布加权和）
*@param type     二值化类型（CV_THRESH_BINARY 大于为最大值，CV_THRESH_BINARY_INV 小于为最大值）
*@param blockSize    块大小（奇数，大于1）
*@param delta 差值（负值也可以）
*/
void adaptiveThreshold1(InputArray _src, OutputArray _dst, double maxValue, int method, int type, int blockSize, double delta, int tianchong)
{
	Mat src = _src.getMat();

	// 原图必须是单通道无符号8位
	CV_Assert(src.type() == CV_8UC1);

	// 块大小必须大于1，并且是奇数
	CV_Assert(blockSize % 2 == 1 && blockSize > 1);
	Size size = src.size();

	// 构建与原图像相同的图像
	_dst.create(size, src.type());
	Mat dst = _dst.getMat();

	if (maxValue < 0)
	{
		// 二值化后值小于0，图像都为0
		dst = Scalar(0);
		return;
	}
	// 用于比较的值
	Mat mean;
	if (src.data != dst.data)
		mean = dst;
	if (method == ADAPTIVE_THRESH_MEAN_C)
		// 计算平均值作为比较值
		boxFilter(src, mean, src.type(), Size(blockSize, blockSize),
			Point(-1, -1), true, tianchong);
	else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
		// 计算高斯分布和作为比较值
		GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, tianchong);
	else if(method== 3)
		medianBlur(src, mean, blockSize);
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");
	int i, j;
	// 将maxValue夹到[0,255]的uchar范围区间，用作二值化后的值
	uchar imaxval = saturate_cast<uchar>(maxValue);

	// 根据二值化类型计算delta值
	int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);

	// 计算生成每个像素差对应的值表格，以后查表就可以。但像素差范围为什么是768，我确实认为512已经够了
	uchar tab[768];

	if (type == CV_THRESH_BINARY)
		for (i = 0; i < 768; i++)
			// i = src[j] - mean[j] + 255
			// i - 255 > -idelta ? imaxval : 0
			// = src[j] - mean[j] + 255 -255 > -idelta ? imaxval : 0
			// = src[j] > mean[j] - idelta ? imaxval : 0
			tab[i] = (uchar)(i - 255 > -idelta ? imaxval : 0);
	else if (type == CV_THRESH_BINARY_INV)
		for (i = 0; i < 768; i++)
			// i = src[j] - mean[j] + 255
			// i - 255 <= -idelta ? imaxval : 0
			// = src[j] - mean[j] + 255 - 255 <= -idelta ? imaxval : 0
			// = src[j] <= mean[j] - idelta ? imaxval : 0
			tab[i] = (uchar)(i - 255 <= -idelta ? imaxval : 0);
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported threshold type");
	// 如果连续，加速运算
	if (src.isContinuous() && mean.isContinuous() && dst.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	// 逐像素计算src[j] - mean[j] + 255，并查表得到结果
	for (i = 0; i < size.height; i++)
	{
		const uchar* sdata = src.data + src.step * i;
		const uchar* mdata = mean.data + mean.step * i;
		uchar* ddata = dst.data + dst.step * i;
		for (j = 0; j < size.width; j++)
			// 将[-255, 255] 映射到[0, 510]然后查表
			ddata[j] = tab[sdata[j] - mdata[j] + 255];
	}
}

/// <summary>
/// 最大值滤波器
/// </summary>
/// <param name="scr"></param>图
/// <param name="size"></param>大小
/// <returns></returns>
Mat max_fliter(Mat scr,int size) 
{
	Mat dst = scr.clone();
	for (int i = size/2; i < scr.rows - 1-size/2; i++)
	{
		for (int j = size/2; j < scr.cols - 1 - size/2; j++)
		{
			double max;
			//dst= scr(Range(i, size+i), Range(j, size+j));
			minMaxIdx(scr(Range(i, size + i), Range(j, size + j)), NULL, &max,NULL, NULL);
			dst.at<uchar>(i,j)=max;
		}
	}
	return dst;
}

cv::Mat image_make_border(cv::Mat& src)
{
	int w = getOptimalDFTSize(src.cols);
	int h = getOptimalDFTSize(src.rows);
	Mat padded;
	copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, BORDER_CONSTANT, Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);
	return padded;
}

//频率域滤波
Mat frequency_filter(Mat& scr, Mat& blur)
{
	//***********************DFT*******************
	Mat plane[] = { scr, Mat::zeros(scr.size() , CV_32FC1) }; //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
	Mat complexIm;
	merge(plane, 2, complexIm);//合并通道 （把两个矩阵合并为一个2通道的Mat类容器）
	dft(complexIm, complexIm);//进行傅立叶变换，结果保存在自身

	//***************中心化********************
	split(complexIm, plane);//分离通道（数组分离）
//    plane[0] = plane[0](Rect(0, 0, plane[0].cols & -2, plane[0].rows & -2));//这里为什么&上-2具体查看opencv文档
//    //其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
	int cx = plane[0].cols / 2; int cy = plane[0].rows / 2;//以下的操作是移动图像  (零频移到中心)
	Mat part1_r(plane[0], Rect(0, 0, cx, cy));  //元素坐标表示为(cx,cy)
	Mat part2_r(plane[0], Rect(cx, 0, cx, cy));
	Mat part3_r(plane[0], Rect(0, cy, cx, cy));
	Mat part4_r(plane[0], Rect(cx, cy, cx, cy));

	Mat temp;
	part1_r.copyTo(temp);  //左上与右下交换位置(实部)
	part4_r.copyTo(part1_r);
	temp.copyTo(part4_r);

	part2_r.copyTo(temp);  //右上与左下交换位置(实部)
	part3_r.copyTo(part2_r);
	temp.copyTo(part3_r);

	Mat part1_i(plane[1], Rect(0, 0, cx, cy));  //元素坐标(cx,cy)
	Mat part2_i(plane[1], Rect(cx, 0, cx, cy));
	Mat part3_i(plane[1], Rect(0, cy, cx, cy));
	Mat part4_i(plane[1], Rect(cx, cy, cx, cy));

	part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
	part4_i.copyTo(part1_i);
	temp.copyTo(part4_i);

	part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
	part3_i.copyTo(part2_i);
	temp.copyTo(part3_i);

	//*****************滤波器函数与DFT结果的乘积****************
	Mat blur_r, blur_i, BLUR;
	multiply(plane[0], blur, blur_r); //滤波（实部与滤波器模板对应元素相乘）
	multiply(plane[1], blur, blur_i);//滤波（虚部与滤波器模板对应元素相乘）
	Mat plane1[] = { blur_r, blur_i };
	merge(plane1, 2, BLUR);//实部与虚部合并

	  //*********************得到原图频谱图***********************************
	magnitude(plane[0], plane[1], plane[0]);//获取幅度图像，0通道为实部通道，1为虚部，因为二维傅立叶变换结果是复数
	plane[0] += Scalar::all(1);  //傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
	log(plane[0], plane[0]);    // float型的灰度空间为[0，1])
	normalize(plane[0], plane[0], 1, 0, CV_MINMAX);  //归一化便于显示

	idft(BLUR, BLUR);    //idft结果也为复数
	split(BLUR, plane);//分离通道，主要获取通道
	magnitude(plane[0], plane[1], plane[0]);  //求幅值(模)
	normalize(plane[0], plane[0], 1, 0, CV_MINMAX);  //归一化便于显示
	return plane[0];//返回参数
}

//*****************理想低通滤波器***********************
Mat ideal_low_kernel(Mat& scr, float sigma)
{
	Mat ideal_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
	float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//分子,计算pow必须为float型
			if (d <= d0) {
				ideal_low_pass.at<float>(i, j) = 1;
			}
			else {
				ideal_low_pass.at<float>(i, j) = 0;
			}
		}
	}

	return ideal_low_pass;
}

//理想低通滤波器
cv::Mat ideal_low_pass_filter(Mat& src, float sigma)
{
	Mat padded = image_make_border(src);
	Mat ideal_kernel = ideal_low_kernel(padded, sigma);
	Mat result = frequency_filter(padded, ideal_kernel);
	return result;
}


Mat butterworth_low_kernel(Mat& scr, float sigma, int n)
{
	Mat butterworth_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
	double D0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//分子,计算pow必须为float型
			butterworth_low_pass.at<float>(i, j) = 1.0 / (1 + pow(d / D0, 2 * n));
		}
	}


	return butterworth_low_pass;
}

//巴特沃斯低通滤波器
Mat butterworth_low_paass_filter(Mat& src, float d0, int n)
{
	//H = 1 / (1+(D/D0)^2n)    n表示巴特沃斯滤波器的次数
	//阶数n=1 无振铃和负值    阶数n=2 轻微振铃和负值  阶数n=5 明显振铃和负值   阶数n=20 与ILPF相似
	Mat padded = image_make_border(src);
	Mat butterworth_kernel = butterworth_low_kernel(padded, d0, n);
	Mat result = frequency_filter(padded, butterworth_kernel);
	return result;
}

Mat gaussian_low_pass_kernel(Mat scr, float sigma)
{
	Mat gaussianBlur(scr.size(), CV_32FC1); //，CV_32FC1
	float d0 = 2 * sigma * sigma;//高斯函数参数，越小，频率高斯滤波器越窄，滤除高频成分越多，图像就越平滑
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			float d = pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2);//分子,计算pow必须为float型
			gaussianBlur.at<float>(i, j) = expf(-d / d0);//expf为以e为底求幂（必须为float型）
		}
	}
	//    Mat show = gaussianBlur.clone();
	//    //归一化到[0,255]供显示
	//    normalize(show, show, 0, 255, NORM_MINMAX);
	//    //转化成CV_8U型
	//    show.convertTo(show, CV_8U);
	//    std::string pic_name = "gaussi" + std::to_string((int)sigma) + ".jpg";
	//    imwrite( pic_name, show);


	return gaussianBlur;
}

//高斯低通
Mat gaussian_low_pass_filter(Mat& src, float d0)
{
	Mat padded = image_make_border(src);
	Mat gaussian_kernel = gaussian_low_pass_kernel(padded, d0);//理想低通滤波器
	Mat result = frequency_filter(padded, gaussian_kernel);
	return result;
}

Mat ideal_high_kernel(Mat& scr, float sigma)
{
	Mat ideal_high_pass(scr.size(), CV_32FC1); //，CV_32FC1
	float d0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//分子,计算pow必须为float型
			if (d <= d0) {
				ideal_high_pass.at<float>(i, j) = 0;
			}
			else {
				ideal_high_pass.at<float>(i, j) = 1;
			}
		}
	}
	string name = "理想高通滤波器d0=" + std::to_string(sigma);
	imshow(name, ideal_high_pass);
	return ideal_high_pass;
}

//理想高通滤波器
cv::Mat ideal_high_pass_filter(Mat& src, float sigma)
{
	Mat padded = image_make_border(src);
	Mat ideal_kernel = ideal_high_kernel(padded, sigma);
	Mat result = frequency_filter(padded, ideal_kernel);
	return result;
}

Mat butterworth_high_kernel(Mat& scr, float sigma, int n)
{
	Mat butterworth_low_pass(scr.size(), CV_32FC1); //，CV_32FC1
	double D0 = sigma;//半径D0越小，模糊越大；半径D0越大，模糊越小
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//分子,计算pow必须为float型
			butterworth_low_pass.at<float>(i, j) = 1.0 / (1 + pow(D0 / d, 2 * n));
		}
	}


	return butterworth_low_pass;
}

//巴特沃斯高通滤波器
Mat butterworth_high_paass_filter(Mat& src, float d0, int n)
{
	//H = 1 / (1+(D0/D)^2n)    n表示巴特沃斯滤波器的次数
	Mat padded = image_make_border(src);
	Mat butterworth_kernel = butterworth_high_kernel(padded, d0, n);
	Mat result = frequency_filter(padded, butterworth_kernel);
	return result;
}

Mat gaussian_high_pass_kernel(Mat scr, float sigma)
{
	Mat gaussianBlur(scr.size(), CV_32FC1); //，CV_32FC1
	float d0 = 2 * sigma * sigma;
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			float d = pow(float(i - scr.rows / 2), 2) + pow(float(j - scr.cols / 2), 2);//分子,计算pow必须为float型
			gaussianBlur.at<float>(i, j) = 1 - expf(-d / d0);
		}
	}

	return gaussianBlur;
}

//高斯高通
Mat gaussian_high_pass_filter(Mat& src, float d0)
{
	Mat padded = image_make_border(src);
	Mat gaussian_kernel = gaussian_high_pass_kernel(padded, d0);//理想低通滤波器
	Mat result = frequency_filter(padded, gaussian_kernel);
	return result;
}


/*=========================================================
*@函 数 名: adaptiveThresholdCustom
*@功能描述: 自适应阈值分割实现图像二值化
*@param src          输入灰度图像
*@param dst          输出二值图像
*@param maxValue     输入满足阈值条件时像素取值
*@param method       计算局部均值方法
*@param type         输入阈值判断类型
*@param blockSize    卷积窗口大小(奇数)
*@param delta        输入偏移常量
*@param ratio        输入均值比例系数
*@修改时间：			2020年09月26日
=========================================================*/
void adaptiveThresholdCustom_whitedot(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double ratio)
{
	CV_Assert(src.type() == CV_8UC1);               // 原图必须是单通道无符号8位,CV_Assert（）若括号中的表达式值为false，则返回一个错误信息
	CV_Assert(blockSize % 2 == 1 && blockSize > 1);	// 块大小必须大于1，并且是奇数
	CV_Assert(maxValue > 0);                        //二值图像最大值
	CV_Assert(ratio > DBL_EPSILON);	                //输入均值比例系数
	Size size = src.size();							//源图像的尺寸
	Mat _dst(size, src.type());						//目标图像的尺寸
	Mat mean;	                                    //存放均值图像
	if (src.data != _dst.data)
		mean = _dst;


	int top = (blockSize - 1) * 0.5;     //填充的上边界行数
	int bottom = (blockSize - 1) * 0.5;  //填充的下边界行数
	int left = (blockSize - 1) * 0.5;	   //填充的左边界行数
	int right = (blockSize - 1) * 0.5;   //填充的右边界行数
	int border_type = BORDER_CONSTANT; //边界填充方式——常量填充
	//int border_type = BORDER_REPLICATE; //边界填充方式——复制最边缘像素
	//int border_type = BORDER_REFLECT_101;//边界填充方式——以最边缘像素为轴，对称
	//int border_type = BORDER_REFLECT;
	//int border_type = BORDER_WRAP;
	Mat src_Expand;	                   //对原图像进行边界扩充

	Mat topImage = src(Rect(src.cols / 4, 0, src.cols/2, bottom));//上边界一行图像

	//cv::Scalar color = cv::mean(topImage)*0.5;//35-80之间均可以  该值需要确定

	//cv::Scalar color = cv::mean(topImage);//35-80之间均可以  该值需要确定

	Scalar color = Scalar(200);//35-80之间均可以
	copyMakeBorder(src, src_Expand, top, bottom, left, right, border_type, color);

	if (method == ADAPTIVE_THRESH_MEAN_C)
	{
		/*
		@param src 单通道灰度图
		@param dst 单通道处理后的图
		@param int类型的ddepth，输出图像的深度
		@param Size类型的ksize，内核的大小
		@param Point类型的anchor，表示锚点
		@param bool类型的normaliz,即是否归一化
		@param borderType 图像外部像素的某种边界模式
		*/
		boxFilter(src_Expand, mean, src.type(), Size(blockSize, blockSize), Point(-1, -1), true, BORDER_CONSTANT);
	}
	else if (method == ADAPTIVE_THRESH_GAUSSIAN_C)
	{
		GaussianBlur(src, mean, Size(blockSize, blockSize), 0, 0, BORDER_DEFAULT);
	}
	else
		CV_Error(CV_StsBadFlag, "Unknown/unsupported adaptive threshold method");

	mean = mean(cv::Rect(top, top, src_Expand.cols - top * 2, src_Expand.rows - top * 2)); //删除扩充的图像边界

	int i, j;
	uchar imaxval = saturate_cast<uchar>(maxValue);	                       //将maxValue由double类型转换为uchar型
	int idelta = type == THRESH_BINARY ? cvCeil(delta) : cvFloor(delta);   //将idelta由double类型转换为int型
	if (src.isContinuous() && mean.isContinuous() && _dst.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	for (i = 0; i < size.height; i++)
	{
		const uchar* sdata = src.data + src.step * i;		   //指向源图像
		const uchar* mdata = mean.data + mean.step * i;		   //指向均值图
		uchar* ddata = _dst.data + _dst.step * i;	           //指向输出图
		for (j = 0; j < size.width; j++)
		{
			double Thresh = mdata[j] * ratio - idelta;	        //阈值
			if (CV_THRESH_BINARY == type)	                    //S>T时为imaxval
			{
				ddata[j] = sdata[j] > Thresh ? imaxval : 0;
			}
			else if (CV_THRESH_BINARY_INV == type)	            //S<T时为imaxval
			{
				ddata[j] = sdata[j] > Thresh ? 0 : imaxval;
			}
			else
				CV_Error(CV_StsBadFlag, "Unknown/unsupported threshold type");
		}
	}
	dst = _dst.clone();
}
/// <summary>
/// 滑窗二值化，
/// </summary>二值化阈值=滑窗均值*ratio+c
/// <param name="scr"></param>图像
/// <param name="row_size"></param>滑窗高度
/// <param name="col_size"></param>滑窗宽度
/// <param name="ratio"></param>均值*比例
/// <param name="c"></param>均值+c
/// <returns></returns>
Mat slipe_threshold(Mat scr,int row_size,int col_size,float ratio,float c)
{
		Mat black_gray;
	    black_gray= scr.clone();
	    Mat gray_clone=black_gray.clone();
	    Mat mask_black;
	    //threshold(black_gray, mask_black,30, 255, CV_THRESH_BINARY);//100->60
	    Scalar tempVal1;
	    int co,ro;
	    for( int row=0;row<black_gray.rows;row=row+ row_size)//滑动窗口分块检测
	    {
	        for( int col=0;col<black_gray.cols;col=col+ col_size)
	        {  co=col;
	            ro=row;
	            if(ro+ row_size >black_gray.rows-1)
	            {
	                ro=black_gray.rows- row_size -1;
	                row=black_gray.rows;
	            }
	            if(co+ col_size >black_gray.cols-1)
	            {
	                co=black_gray.cols- col_size -1;
	                col=black_gray.cols;
	            }
				Rect rect_wp(co, ro, col_size, row_size);

	            //Rect rect_wp(co,ro, size, black_gray.rows-1);
	            Mat temp_wp=black_gray(rect_wp);
	            tempVal1 =mean( temp_wp );
	            double matMean_wp = tempVal1.val[0];
	            Mat th_wp;
	            threshold(temp_wp, th_wp,matMean_wp* ratio +c, 255, CV_THRESH_BINARY);//10->20
	            th_wp.copyTo(gray_clone(rect_wp));
	        }
	    }

		return gray_clone;
}

Mat mid_adaptive_threshold(Mat scr, int size, int mid_off_set, int method, int maxmin_remove, int siqu)
{
	Mat black_gray;
	black_gray = scr.clone();
	Mat gray_clone = black_gray.clone();
	Mat mask_black;
	//threshold(black_gray, mask_black,30, 255, CV_THRESH_BINARY);//100->60
	Scalar tempVal1;
	int co, ro;
	for (int row = 0; row < black_gray.rows; row = row + size)//滑动窗口分块检测
	{
		for (int col = 0; col < black_gray.cols; col = col + size)
		{
			co = col;
			ro = row;
			if (ro + size > black_gray.rows - 1)
			{
				ro = black_gray.rows - size - 1;
				row = black_gray.rows;
			}
			if (co + size > black_gray.cols - 1)
			{
				co = black_gray.cols - size - 1;
				col = black_gray.cols;
			}
			Rect rect_wp(co, ro, size, size);
			
			Mat temp_wp = black_gray(rect_wp);
			vector<int> nums;
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < size; j++) {
					nums.push_back(temp_wp.at<uchar>(i, j));
				}
			}


			vector<int> index0(nums.size(), 0);
			for (int i = 0; i != index0.size(); i++) {
				index0[i] = i;
			}
			sort(index0.begin(), index0.end(), [&](const int& a, const int& b)
				{
					return (nums[a] < nums[b]);
				});

			int threshold_value = nums[(size * size - 1) / 2 + mid_off_set];
			nums[(size * size - 1) / 2 + mid_off_set] < nums[(size * size - 1)] - siqu ? threshold_value = nums[(size * size - 1) / 2 + mid_off_set] : threshold_value = nums[(size * size - 1)];
			//Rect rect_wp(co,ro, size, black_gray.rows-1);
			tempVal1 = mean(temp_wp);
			double matMean_wp = tempVal1.val[0];
			threshold_value > tempVal1.val[0] ? threshold_value : tempVal1.val[0];
			Mat th_wp;
			threshold(temp_wp, th_wp, threshold_value, 255, CV_THRESH_BINARY);//10->20
			th_wp.copyTo(gray_clone(rect_wp));
		}
	}

	return gray_clone;
}
bool Frame(Mat src_img,Mat ceguang, Mat* mresult, String* causecolor)
{
	bool TestResult_Frame = false;
	//Mat se = Gabor7(ceguang);
	Mat grad_x, grad_y;
	int scale = 1, delta = 1;
	Rect rect;
	bool is_clockwise_direction = true;
	Mat binaryImage = Mat::zeros(src_img.size(), CV_8UC1);                              //二值图像
	threshold(src_img, binaryImage, 0.6 * mean(src_img)[0], 255, CV_THRESH_BINARY);							//二值化(有问题)
	vector<vector<Point>> contours5;
	//flip(binaryImage, binaryImage, 0);

	findContours(binaryImage, contours5, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (vector<int>::size_type i = 0; i < contours5.size(); i++)
	{
		double area = contourArea(contours5[i]);
		if (area > 300000 && area < 5000000)
		{
			Point2f rect_points[4];
			Rect Rect = boundingRect(contours5[i]);
			RotatedRect minRect = minAreaRect(contours5[i]);                                   //最小外接矩形提取

			minRect.angle < -45 ? is_clockwise_direction = true : is_clockwise_direction = false;//判断矩形角度，保持序列递增


			if (minRect.size.height >= 7 && minRect.size.width >= 7)
			{
				minRect.points(rect_points);
				for (int j = 0; j < 4; j++)
				{
					//line(se, rect_points[j], rect_points[(j + 1) % 4], Scalar(255), 1, 8);
					//line(binaryImage, rect_points[j], rect_points[(j + 1) % 4], Scalar(255), 1, 8);
				}

			}
			if (Rect.height >= 7 && Rect.width >= 7)
			{

				//rectangle(se, Rect.tl(), Rect.br(), Scalar(255), 1, 8, 0);
				//rectangle(binaryImage, Rect.tl(), Rect.br(), Scalar(255), 1, 8, 0);
				rect = Rect;
			}
			break;
		}
	}
	//未提取轮廓直接跳出
	if (rect.empty()) {
		return false;
	}
	int midblur_size = 5;
	Mat scr = binaryImage.clone()(rect);
	int border = 70;
	Rect we_rect(border, 0, scr.rows - 2 * border, 1);
	Mat dst_west(1, scr.rows, CV_8UC1, Scalar(0));
	for (int i = scr.rows - border - 1; i >= border; i--) {
		for (int j = 0; j < scr.cols / 2 - 1; j++) {
			if (scr.at<uchar>(i, j) > 100)
			{
				dst_west.at<uchar>(0, i) = j;
				break;
			}
		}
	}
	Mat west;
	if (is_clockwise_direction) {
	flip(dst_west, dst_west, 1);
	}
	medianBlur(dst_west, west, midblur_size);
	Mat dst_east(1, scr.rows, CV_8UC1, Scalar(0));
	for (int i = border - 1; i < scr.rows - border; i++) {
		for (int j = scr.cols - 1; j > scr.cols / 2 - 1; j--) {
			if (scr.at<uchar>(i, j) > 100)
			{
				dst_east.at<uchar>(0, i) = scr.cols - j;
				break;
			}
		}
	}
	Mat east;
	if (is_clockwise_direction!=true) {
		flip(dst_east, dst_east, 1);
	}
	medianBlur(dst_east, east, midblur_size);

	Rect sn_rect(border, 0, scr.cols - 2 * border, 1);

	Mat dst_north(1, scr.cols, CV_8UC1, Scalar(0));
	for (int j = border - 1; j < scr.cols - border; j++) {
		for (int i = 0; i < scr.rows / 2; i++) {
			if (scr.at<uchar>(i, j) > 100)
			{
				dst_north.at<uchar>(0, j) = i;
				break;
			}
		}
	}
	Mat north;
	if (is_clockwise_direction != true) {
		flip(dst_north, dst_north, 1);
	}
	medianBlur(dst_north, north, midblur_size);
	Mat dst_south(1, scr.cols, CV_8UC1, Scalar(0));
	for (int j = scr.cols - 1 - border; j >= border; j--) {
		for (int i = scr.rows - 1; i > scr.rows / 2; i--) {
			if (scr.at<uchar>(i, j) > 100)
			{
				dst_south.at<uchar>(0, j) = scr.rows - i;
				break;
			}
		}
	}
	if (is_clockwise_direction) {
		flip(dst_south, dst_south, 1);
	}
	Mat south;
	medianBlur(dst_south, south, midblur_size);
	east = east(we_rect);
	west = west(we_rect);
	south = south(sn_rect);
	north = north(sn_rect);
	//if(is_clockwise_direction)
	//{ 
	//	flip(south, south, 1);
	//	flip(west, west, 1);
	//}
	//else {
	//	flip(north, north, 1);
	//	flip(east, east, 1);
	//}



	vector<vector<int>>east_sample = Mat_to_vector(east);
	vector<vector<int>>west_sample = Mat_to_vector(west);
	vector<vector<int>>south_sample = Mat_to_vector(south);
	vector<vector<int>>north_sample = Mat_to_vector(north);
	bool frame_east=is_deformation(east_sample, east.cols, border);
	bool frame_west = is_deformation(west_sample, west.cols, border);
	bool frame_south = is_deformation(south_sample, south.cols, border);
	bool frame_north = is_deformation(north_sample, north.cols, border);
	

	if (frame_east|| frame_west || frame_south || frame_north)
	{
		TestResult_Frame = true;
		*causecolor = "边框";
		//*mresult = img_FinalResult;
	}

	return TestResult_Frame;
}

		
