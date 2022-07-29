#include "direct.h"
#include <cmath>
#include <fstream>
#include <io.h>
#include <iostream>
#include <map>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string.h>
#include "windows.h"
#include <time.h>
#include <iostream>
#include "math.h"
using namespace cv;
using namespace std;

void getSubdirs(std::string path, std::vector<std::string>& files);
Mat toushi_white(Mat image, Mat M, int border, int length, int width);
Mat Gabor7(Mat img_1);
bool Crease(Mat frontSideLight, Mat leftSideLight, Mat rightSideLight, Mat front, Mat* mresult, string* causecolor);
bool Creases(Mat frontSideLight, Mat front, Mat* mresult, string* causecolor);
void adaptiveThresholdCustom(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double ratio, double fillValCoeffi);
bool compareContourAreas(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2);
bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat* Mwhite, Mat* Mbiankuang, Mat* M_white_abshow, int ID, String ScreenType_Flag, int leftRightWhiteFlag);
bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat* Mwhite, Mat* M_R_1_E, String ScreenType_Flag, int leftRightWhiteFlag, int border_white);
bool f_FrontCam_PersTransMatCal(InputArray _src, Mat* Mwhite, string ScreenType_Flag, int leftRightWhiteFlag);
Point2f getPointSlopeCrossPoint(Vec4f LineA, Vec4f LineB);
void convexSetPretreatment(Mat& src);
Mat Ployfit_Col7(Mat img_col, int poly_n, bool isSaveOrNot, double Scoral);
bool Scratch(Mat white, Mat ceguang, Mat* mresult, string* causecolor, int camera);
Mat gamma(Mat src, double g);
bool Crease(Mat frontSideLight, Mat front, Mat left, Mat* mresult, string* causecolor, bool leftflag);
bool edgeCrease(Mat white_F1, Mat* mresult, string* causecolor);
bool Creasemain(Mat white, Mat* mresult, string* causecolor);
bool Crease_L(Mat frontSideLight, Mat front, Mat* mresult, string* causecolor);
bool Crease1(Mat frontSideLight, Mat leftSideLight, Mat rightSideLight, string* causecolor);
bool Dead_light0(Mat white, Mat sideLight, Mat* mresult, string* causecolor);
bool sideLightDetection(Mat sideLight);
bool leftRightCrease(Mat white_L1, Mat* mresult, string* causecolor);
bool blackLine(Mat white, Mat* mresult, string* causecolor);

int bSums(Mat src);


Mat Process(Mat& A, double sig1, double sig2, Size Ksize)
{
	Mat AF, out, out1, out2;
	A.convertTo(AF, CV_32FC1);
	GaussianBlur(AF, out1, Ksize, sig1, 0);
	GaussianBlur(AF, out2, Ksize, sig2, 0);
	subtract(out1, out2, out);
	normalize(out, out, 0, 255, NORM_MINMAX);
	out.convertTo(out, CV_8UC1);
	return out;
}
void num2string(double num, string& str)
{
	stringstream ss;
	ss << num;
	str = ss.str();
}
string rootPath = "D:\\test\\verify\\LONG\\7_28\\打折漏检样本\\435";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
string rootPath1 = "D:\\test\\verify\\膜材折痕固定位置漏检样本";//文件根目录  1\\2.8寸样本总览\\少线\\2.8寸少线\\2.8寸少线\\108SX_
//ofstream  csvFile("F:\\photoScreen\\手机屏项目出差\\检测结果.csv");//保存结果路径




using namespace std;
string cyk;
string cy1;
string Blankimage;
string outname = "\\新轮廓.csv";
string outname1 = "\\结果.csv";
string Sample_number = "HF-ACA03-LP";
string Sample_number1;
FILE* pOutFile;                               //输出文件指针
FILE* pOutFile1;                               //输出文件指针
string path1;
string namenum;
double timeLength;
int sampleNum;
bool Ext_Result_Left_Right;
DWORD t1, t2;



void getSubdirs(std::string path, std::vector<std::string>& files)
{
	long long hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))//是目录
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//
				{
					getSubdirs(p.assign(path).append("\\").append(fileinfo.name), files);
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}
			}
			else//是文件不是目录
			{
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				if (std::find(files.begin(), files.end(), p.assign(path)) == files.end())
					files.push_back(p.assign(path));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

vector<string> split(string str, char del) {
	stringstream ss(str);
	string temp;
	vector<string> ret;
	while (getline(ss, temp, del)) {
		ret.push_back(temp);
	}
	return ret;
}
Mat AreaGrow(Mat mat)
{
	//mat为输入图像
	//定义第一个种子点位置为图片最中心处
	//注：opencv里面图像的x坐标是列数，y坐标是行数

	int firstSeed_x = round(mat.cols / 2);    //四舍五入取整，防止坐标为小数
	int firstSeed_y = round(mat.rows / 2);
	Point firstSeed = Point(firstSeed_x, firstSeed_y);

	Mat growArea = Mat::zeros(mat.size(), CV_8UC1);    //生长区域
	 //为第一个生长点赋值
	growArea.at<uchar>(Point(firstSeed.x, firstSeed.y)) = mat.at<uchar>(Point(firstSeed.x, firstSeed.y));
	Point waitSeed;    //待生长种子点
	int waitSeed_value = 0;    //待生长种子点像素值
	int opp_waitSeed_value = 0;   //mat_thresh中对应待生长种子点处的像素值
	vector<Point> seedVector;     //种子栈
	seedVector.push_back(firstSeed);    //将种子放入栈中最后一个位置
	int direct[4][2] = { {0,-1},{1,0}, {0,1}, {-1,0} };   //4邻域,应该用4邻域减小时间复杂度
	//int direct[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };  //8邻域

	while (!seedVector.empty())     //种子栈不为空则生长，即遍历栈中所有元素后停止生长
	{
		Point seed = seedVector.back();     //取出最后一个元素
		seedVector.pop_back();         //删除栈中最后一个元素,防止重复扫描
		for (int i = 0; i < 4; i++)    //遍历种子点的4邻域
		{
			waitSeed.x = seed.x + direct[i][0];    //第i个坐标0行，即x坐标值
			waitSeed.y = seed.y + direct[i][1];    //第i个坐标1行，即y坐标值

			//检查是否是边缘点
			if (waitSeed.x < 0 || waitSeed.y < 0 ||
				waitSeed.x >(mat.cols - 1) || (waitSeed.y > mat.rows - 1))
				continue;

			waitSeed_value = growArea.at<uchar>(Point(waitSeed.x, waitSeed.y));   //为待生长种子点赋对应位置的像素值
			opp_waitSeed_value = mat.at<uchar>(Point(waitSeed.x, waitSeed.y));
			if (waitSeed_value == 0)     //判断waitSeed是否已经被生长，避免重复生长造成死循环
			{
				if (opp_waitSeed_value != 0)     //区域生长条件
				{
					growArea.at<uchar>(Point(waitSeed.x, waitSeed.y)) = mat.at<uchar>(Point(waitSeed.x, waitSeed.y));
					seedVector.push_back(waitSeed);    //将满足生长条件的待生长种子点放入种子栈中
				}
			}
		}
	}
	return growArea;
}


Mat RegionGrow(Mat src, Point2i pt, int th)
{
	Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0;								//标记是否生长过
	int nSrcValue = 0;								//生长起点灰度值
	int nCurValue = 0;								//当前生长点灰度值
	Mat matDst = Mat::zeros(src.size(), CV_8UC1);	//创建一个空白区域，填充为黑色
	//生长方向顺序数据
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	vector<Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push_back(pt);							//将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;				//标记生长点
	nSrcValue = src.at<uchar>(pt.y, pt.x);			//记录生长点的灰度值

	while (!vcGrowPt.empty())						//生长栈不为空则生长
	{
		pt = vcGrowPt.back();						//取出一个生长点
		vcGrowPt.pop_back();

		//分别对八个方向上的点进行生长
		for (int i = 0; i < 9; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;

			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值

			if (nGrowLable == 0)					//如果标记点还没有被生长
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th)					//在阈值范围内则生长
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//标记为白色
					vcGrowPt.push_back(ptGrowing);					//将下一个生长点压入栈中
				}
			}
		}
	}
	return matDst.clone();
}


int main() {

	vector<vector<string>> results;
	vector<string> filenames;//用来存储文件名	
	cyk.append(rootPath1).append(outname);
	cy1.append(rootPath1).append(outname1);
	/*	Sample_number1=Sample_number.data;*/
	const char* cyk1 = cyk.data();
	const char* cy2 = cy1.data();

	if ((fopen_s(&pOutFile, cyk1, "w")) != NULL)
	{
		printf("can not write on file");
		exit(0);
	}
	if ((fopen_s(&pOutFile1, cy2, "w")) != NULL)
	{
		printf("can not write on file1");
		exit(0);
	}
	getSubdirs(rootPath, filenames);//获得根目录下所有图片路径
	fprintf(pOutFile, "样本名,轮廓编码,轮廓面积,长度,长宽比,侧光灰底差,白底灰度差,侧光标准差,白底标准差,结果\n");
	fprintf(pOutFile1, "样本名,结果\n");
	if (filenames.size() == 0)
		filenames.push_back(rootPath);
	for (int g = 0; g < filenames.size(); g++)
	{
		string Path = filenames[g] + "\\";
		cout << Path << endl;
		vector<string> res = split(Path, '\\');
		namenum = res[res.size() - 1];
		Mat src_white = cv::imread(Path + "212.bmp", -1);
		Mat src_white_L = cv::imread(Path + "012.bmp", -1);
		Mat src_white_R = cv::imread(Path + "112.bmp", -1);
		Mat src_white_F = cv::imread(Path + "front.bmp", -1);
		Mat src_ceguang = cv::imread(Path + "210.bmp", -1);
		Mat src_ceguang_L = cv::imread(Path + "010.bmp", -1);
		Mat src_ceguang_R = cv::imread(Path + "110.bmp", -1);
		Mat src_ceguang_F = cv::imread(Path + "FrontCeguang.bmp", -1);

		Mat src_White_F_MY = cv::imread(Path + "frontROI.bmp", -1);
		Mat M_white;
		Mat M_white_2;
		Mat M_black;
		Mat M_black_2;
		Mat M_louguang;
		Mat M_louguang_2;
		Mat M_white_abshow;
		Mat M_white_abshow_2;
		Mat M_L_1, M_louguang_left, M_R_1, M_louguang_right, M_F_1, M_B_1;
		Mat M_L_1_E, M_R_1_E;
		Mat Mresult_1_white;
		string causeColor_1_white = "";

		bool Ext_Result_BlackWhite = f_MainCam_PersTransMatCal(src_white, 0, 70, &M_white, &M_black, &M_white_abshow, 1, "R角水滴屏", -1);

		bool Ext_Result_Left = f_LeftRightCam_PersTransMatCal(src_white_L, &M_L_1, &M_L_1_E, "R角水滴屏", 1, 3);
		bool Ext_Result_Right = f_LeftRightCam_PersTransMatCal(src_white_R, &M_R_1, &M_R_1_E, "R角水滴屏", 1, 15);

		string Model_Mod_type;
		bool Ext_Result_Front = f_FrontCam_PersTransMatCal(src_white_F, &M_F_1, Model_Mod_type, 1);

		//f_FrontBackCam_PersTransMatCal(src_white_B, &M_B_1, "后");

		Mat white = toushi_white(src_white, M_white, -1, 3000, 1500);
		Mat white_L1 = toushi_white(src_white_L, M_L_1_E, -5, 3000, 1500);
		Mat white_R1 = toushi_white(src_white_R, M_R_1, -5, 3000, 1500);
		Mat white_F1 = toushi_white(src_white_F, M_F_1, -1, 3000, 1500);


		//Mat white_B1 = toushi_white(src_white_B, M_B_1, -5, 3000, 1500);
		Mat ceguang = toushi_white(src_ceguang, M_white, -1, 3000, 1500);
		Mat ceguang_L1 = toushi_white(src_ceguang_L, M_L_1, -5, 3000, 1500);
		Mat ceguang_R1 = toushi_white(src_ceguang_R, M_R_1, -5, 3000, 1500);

		Mat ceguang_F1 = toushi_white(src_ceguang_F, M_F_1, -1, 3000, 1500);
		Mat frontSideLight = ceguang_F1;
		Mat leftSideLight = ceguang_L1;
		Mat rightSideLight = ceguang_R1;
		Mat mresult;
		string causecolor;
		white = Gabor7(white);
		white_F1 = Gabor7(white_F1);
		white_L1 = Gabor7(white_L1);
		frontSideLight = Gabor7(frontSideLight);
		//Scratch(white_F1, ceguang_F1, &mresult, &causecolor, 1);
		Mat cha = ceguang_F1 - white_F1;

		//主相机侧光异常检测
		//sideLightDetection(white);
		bool result = false;

		//黑线算法
		//result = blackLine(white, &mresult, &causecolor);

		//死灯算法
		//result = Dead_light0(white, ceguang, &mresult, &causecolor);

		//边缘折痕检测
		//result = edgeCrease(white_F1, &mresult, &causecolor);

		//左相机检测
		//result = leftRightCrease(white_L1, &mresult, &causecolor);

		//右相机检测
		//result = leftRightCrease(white_R1, &mresult, &causecolor);

		//小折痕检测
		src_White_F_MY = Gabor7(src_White_F_MY);
		result = Crease(frontSideLight, src_White_F_MY, white_L1, &mresult, &causecolor, true);//white_F1

		//膜材打折检测
		//result = Creasemain(white, &mresult, &causecolor);

		if (result) {
			cout << causecolor << endl;
		}
		else {
			cout << causecolor << endl;
		}

	}
	/*Mat frontSideLight = imread("D:\\Backlight\\折痕测试\\FrontCeguangROI.bmp");
	Mat leftSideLight = imread("D:\\Backlight\\折痕测试\\CLROI.bmp");
	Mat rightSideLight = imread("D:\\Backlight\\折痕测试\\CRROI.bmp");
	if (frontSideLight.empty() || leftSideLight.empty() || rightSideLight.empty()) {
		cout << "could not find image file" << endl;
		return 0;
	}

	cvtColor(frontSideLight, frontSideLight, CV_BGR2GRAY);
	cvtColor(leftSideLight, leftSideLight, CV_BGR2GRAY);
	cvtColor(rightSideLight, rightSideLight, CV_BGR2GRAY);

	Crease(frontSideLight, leftSideLight, rightSideLight);*/
	waitKey();

	return 0;
}
Mat gammaTransform(Mat& srcImage, float kFactor) {

	unsigned char LUT[256];
	for (int i = 0; i < 256; i++) {
		//压缩到0-1区间
		float f = (i + 0.5f) / 255;
		f = (float)(pow(f, kFactor));
		LUT[i] = saturate_cast<uchar>(f * 255.0f - 0.5f);
	}
	Mat resultImage = srcImage.clone();
	cout << srcImage.channels() << endl;


	MatIterator_<uchar> iterator = resultImage.begin<uchar>();
	MatIterator_<uchar> iteratorEnd = resultImage.end<uchar>();
	for (; iterator != iteratorEnd; iterator++) {
		*iterator = LUT[(*iterator)];
	}


	return resultImage;
}


bool blackLine(Mat white, Mat* mresult, string* causecolor) {
	ofstream lbd;
	lbd.open("D:\\lbd.txt", ios::app);
	bool result = false;

	//CLAHE
	Ptr<CLAHE> clahe = createCLAHE(2, Size(40, 40));
	Mat claheImg;
	clahe->apply(white, claheImg);   //整图增强
	Mat enhanceImg;
	enhanceImg = claheImg + (claheImg - white);//frontWhite ++(img_gray_clahe-frontWhite)

	//遍历列像素
	int shield = 150;
	Mat colMean = Mat::zeros(1, 3000 - 2 * shield, CV_32FC1);
	for (int i = shield; i < 3000 - shield; i++) {
		//const uchar* t = sideLight.ptr<uchar>(i);
		int sum = 0;
		for (int n = shield; n < 1500 - shield; n++) {
			int dx = enhanceImg.at<uchar>(n, i);
			sum = sum + dx;
		}
		colMean.at<float>(0, i - shield) = sum / double(1500 - 2 * shield);
	}

	//移动均值
	Mat dst;
	copyMakeBorder(colMean, dst, 0, 0, 50, 50, BORDER_REPLICATE);
	Mat  moveMean = Mat::zeros(colMean.size(), colMean.type());
	for (int i = 0; i < colMean.cols; i++) {
		double m = 0;
		m = mean(dst(Rect(i, 0, 100, 1)))[0];
		moveMean.at<float>(i) = m;
	}

	//cout << " " << endl;

	//计算偏差
	Mat gap;
	gap = moveMean - colMean;
	int pos = 0;
	int count = 0;
	float area = 0;
	for (int i = 0; i < gap.cols; i++) {
		if (gap.at<float>(0, i) >= 3) {
			count++;
			area += gap.at<float>(0, i);
			if (count > 3 && area > 25.0) {
				pos = i;
				break;
			}
		}
		else {
			count = 0;
			area = 0;
		}
	}

	for (int i = 0; i < colMean.cols; i++) {
		lbd << moveMean.at<float>(0, i) << " " << colMean.at<float>(0, i) << endl;
	}
	Mat  smallSrcTh = Mat(white.size(), white.type(), 255);

	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("01").value().find("01").value(), smallSrcTh, *creasePara->m_shieldshape.find("01").value().find("01").value());




	//计算偏差
	int shelid = smallSrcTh.at<uchar>(750, pos + 150);
	if (pos != 0 && area > 25.0 && shelid == 255) {
		*causecolor = "黑线";
		result = true;
		//矩形框标记
		CvPoint small_lt = cvPoint(pos + shield - 50, 150); //pos - 50 + 150
		CvPoint small_br = cvPoint(pos + shield + 50, 1350);  //pos + 50 + 150
		rectangle(white, small_lt, small_br, Scalar(255, 255, 255), 5, 8, 0);
		*mresult = white;
		return true;
	}
	return false;
}

bool colMean(Mat white) {
	ofstream lbd;
	lbd.open("D:\\lbd.txt", ios::app);


	for (int i = 11; i < 12; i++) {
		//const uchar* t = sideLight.ptr<uchar>(i);
		int sum = 0;
		for (int n = 150; n < 1350; n++) {
			int dx = white.at<uchar>(n, i);
			//            sum = sum + dx;
			lbd << dx << endl;
		}
	}
	return false;
}
double getPSNR(const Mat& I1)
{
	Mat s1;
	Mat I2 = Mat::zeros(I1.size(), CV_8UC1);

	absdiff(I1, I2, s1);
	s1 = s1.mul(s1);
	Scalar s = sum(s1);
	double sse = s.val[0];
	if (sse <= 1e-10)
		return 0;
	else {
		double  mse = sse / (double)(I1.channels() * I1.total());	// MSE
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

double getGradient(Mat img) {

	double tmp = 0;
	int rows = img.rows - 1;
	int cols = img.cols - 1;
	for (int m = 0; m < rows; m++) {
		const uchar* t = img.ptr<uchar>(m);
		for (int n = 0; n < cols; n++) {
			int dx = t[n + 1] - t[n];
			int dy = img.ptr<uchar>(m + 1)[n] - t[n];
			//int dy = (t + 1)[n] - t[n];
			double ds = std::sqrt((dx * dx + dy * dy) / 2);
			tmp += ds;
		}
	}
	double imageAvG = tmp / (rows * cols);
	return imageAvG;
}

//计算图像信息熵
double getEntropy(Mat img)
{
	double temp[256] = { 0.0 };

	// 计算每个像素的累积值
	for (int m = 0; m < img.rows; m++) {
		const uchar* t = img.ptr<uchar>(m);
		for (int n = 0; n < img.cols; n++) {
			int i = t[n];
			temp[i] = temp[i] + 1;
		}
	}

	// 计算每个像素的概率
	for (int i = 0; i < 256; i++)
	{
		temp[i] = temp[i] / (img.rows * img.cols);
	}

	double result = 0;
	// 计算图像信息熵
	for (int i = 0; i < 256; i++)
	{
		if (temp[i] == 0.0)
			result = result;
		else
			result = result - temp[i] * (log(temp[i]) / log(2.0));
	}

	return result;

}

double SSIM(vector<double> x, vector<double> y) {

	int m = x.size();
	int n = y.size();
	if (m != n || m == 0) {
		return 0;
	}

	double molecule = 0;
	double denominator = 0;
	double xSum = 0;
	double ySum = 0;
	for (int i = 0; i < m; i++) {
		//分子计算
		molecule += x[i] * y[i];

		//分母计算
		xSum += pow(x[i], 2);
		ySum += pow(y[i], 2);
	}
	denominator = sqrt(xSum) * sqrt(ySum);

	if (denominator == 0) {
		return 0;
	}

	return molecule / denominator;
}



bool sideLightDetection(Mat sideLight) {
	ofstream lbd;
	lbd.open("D:\\lbd.txt", ios::app);

	Mat temp = sideLight(Rect(500, 1450, 2000, 30));
	double m = mean(temp)[0];
	//cout << m << endl;
	//cout << sideLight.rows << endl;
	//像素均值统计
	for (int i = 0; i < 3000; i++) {
		//const uchar* t = sideLight.ptr<uchar>(i);
		int sum = 0;
		for (int n = 150; n < 1350; n++) {
			int dx = sideLight.at<uchar>(n, i);
			sum = sum + dx;
		}
		//cout << sum / 1200 << endl;
		lbd << sum / 1200 << endl;
	}

	//for (int i = 0; i < edge.cols; i++) {
	//	cout << int(mm.at<uchar>(0,i)) << endl;
	//}


	if (m < 20) {
		return true;
	}

	return false;
}
bool edgeCrease(Mat frontMat, Mat* mresult, string* causecolor) {

	bool result = false;
	Mat borderImg;

	Mat front = frontMat.clone();
	Mat leftEdge = front(Rect(0, 0, 200, front.rows));
	Mat rightEdge = front(Rect(2800, 0, 200, front.rows));
	Mat img1 = front(Rect(2800, 0, 183, front.rows)).clone();

	//    qDebug()<<"111111"<<endl;
		//边界镜像扩充，自适应分割
	copyMakeBorder(img1, borderImg, 0, 0, 0, 50, BORDER_REFLECT_101);
	//Mat thLeftEdge;
	Mat thRightEdge;
	//adaptiveThreshold(leftEdge, thLeftEdge, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, 2);
	adaptiveThreshold(borderImg, thRightEdge, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 21, -1);//21   -1

	//屏蔽R角区域
	thRightEdge(Rect(0, 0, thRightEdge.cols, 125)) = uchar(0);
	thRightEdge(Rect(0, 1375, thRightEdge.cols, 125)) = uchar(0);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(thRightEdge, thRightEdge, MORPH_OPEN, element);   //开运算形态学操作。可以减少噪点

	Mat thRightEdge1;
	thRightEdge1 = thRightEdge(Rect(0, 0, 183, thRightEdge.rows));


	//轮廓遍历
	Mat frontSrcTh2 = Mat::ones(front.size(), CV_8UC1);
	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("02").value().find("04").value(), frontSrcTh2, *creasePara->m_shieldshape.find("02").value().find("04").value());
	Mat img2 = frontSrcTh2(Rect(2800, 0, 183, front.rows));
	bitwise_and(img2, thRightEdge1, thRightEdge1);
	//  imwrite("D:\\ssss.bmp",thRightEdge1);
	vector<vector<Point>>contoursEdge;
	findContours(thRightEdge1, contoursEdge, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contoursEdge.begin(), contoursEdge.end(), compareContourAreas);
	vector<Rect> smallBoundRect(contoursEdge.size());
	for (vector<int> ::size_type i = 0; i < contoursEdge.size(); i++) {
		//        Mat smallMask = Mat::zeros(Size(235, 1500), CV_8UC1);
		//        drawContours(smallMask, contoursEdge, i, 255, -1, 8);
		double edgeArea = contourArea(contoursEdge[i]);

		//  qDebug()<<edgeArea<<
		  //面积判定
		if (edgeArea < 240) {  //905,801,1493,390,450,647,
			//break;
		}

		//正交外接矩形
		smallBoundRect[i] = boundingRect(Mat(contoursEdge[i]));
		int smallX_1 = smallBoundRect[i].tl().x;//矩形左上角X坐标值
		int smallY_1 = smallBoundRect[i].tl().y;//矩形左上角Y坐标值
		int smallX_2 = smallBoundRect[i].br().x;//矩形右下角X坐标值
		int smallY_2 = smallBoundRect[i].br().y;//矩形右下角Y坐标值


		Mat whitelightSuspect = img1(Rect(smallX_1, smallY_1, smallX_2 - smallX_1 - 1, smallY_2 - smallY_1 - 1));//白底图像疑似反光划痕图像
		cv::Mat meanGray;
		cv::Mat stdDev;
		cv::meanStdDev(whitelightSuspect, meanGray, stdDev);
		//长宽比判定
		RotatedRect  smallRotRect = minAreaRect(Mat(contoursEdge[i]));
		double smallHei = smallRotRect.size.height;
		double smallWid = smallRotRect.size.width;
		double smallRatio = max(smallHei / smallWid, smallWid / smallHei);

		// double smallRatio1 = max(smallBoundRect[i].width / smallBoundRect[i].height, smallBoundRect[i].height / smallBoundRect[i].width);
 //        qDebug()<<edgeArea<<"  "<<meanGray.at<double>(0, 0)<<"w:"<<stdDev.at<double>(0, 0)<<""<<smallBoundRect[i].height<<endl;

 //           qDebug()<<edgeArea<<"  "<<smallRatio<<"w1:"<<smallWid<<""<<smallHei<<endl;
		if (smallRatio > 2.0 && smallRatio < 7.0 && meanGray.at<double>(0, 0) < 100 && stdDev.at<double>(0, 0) > 2.4) {
			*causecolor = "边缘折痕";
			result = true;
			CvPoint small_lt = cvPoint(smallX_1 + 2800, smallY_1);
			CvPoint small_br = cvPoint(smallX_2 + 2800, smallY_2);
			rectangle(front, small_lt, small_br, Scalar(255, 255, 255), 5, 8, 0);
			*mresult = front;
			break;
		}
	}

	return result;
}

int bSums(Mat src)
{

	int counter = 0;
	//迭代器访问像素点
	
	vector<int>rows_pixel;
	vector<int>cols_pixel;
	for (int temp_index_rows = 0; temp_index_rows < src.rows; temp_index_rows++)
	{//行遍历
		int counter_rows = 0;
		for (int temp_index_cols = 0; temp_index_cols < src.cols; temp_index_cols++)
		{//列遍历
			if (temp_index_rows == 0)
			{
				cols_pixel.push_back(0);
			}
			if (src.at<uchar>(temp_index_rows, temp_index_cols) == 255)
			{
				counter_rows++;
				cols_pixel[temp_index_cols] = (cols_pixel[temp_index_cols] + 1);
			}
		}
		rows_pixel.push_back(counter_rows);

	}

	for (int i = 0; i < rows_pixel.size(); ++i)
	{
		if (i >= 20)
		{
			break;
		}
		if (rows_pixel[i]<1300)
		{
			counter = i + 3;
		}
	}//给出边界索引

	

	return counter;
}
bool Creasemain(Mat white, Mat* mresult, string* causecolor) {
	bool result = false;
	//flip(front, front, -1);
	//2022.3.28 am
	/**********屏蔽模板**********/
	Mat shieldMask;
	threshold(white, shieldMask, 0.3 * mean(white)[0], 255, CV_THRESH_BINARY_INV);
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(29, 29));
	dilate(shieldMask, shieldMask, dilateElement);
	shieldMask(Rect(0, 0, shieldMask.cols, 30)) = uchar(255);
	shieldMask(Rect(0, shieldMask.rows - 30, shieldMask.cols, 30)) = uchar(255);
	shieldMask(Rect(0, 0, 25, shieldMask.rows)) = uchar(255);
	shieldMask(Rect(shieldMask.cols - 25, 0, 25, shieldMask.rows)) = uchar(255);
	/**********屏蔽模板**********/
	/**********图像拷贝********/
	Mat srcWhite = white.clone();
	double meanfront = mean(white)[0];
	//srcWhite = Gabor7Crease(srcWhite);

	/*********高斯滤波*********/
	GaussianBlur(srcWhite, srcWhite, Size(25, 25), 5);        //白底滤波  *creasePara->GaussianStv

	Mat frontSrcTh;  //前相机阈值分割
	Mat frontSrcTh1;  //前相机阈值分

	//屏幕类型
	int border = 100;
	//if (ScreenMode == 1 || ScreenMode == 2 || ScreenMode == 3) {
	//	border = 200;
	//}
	//else {
	//	border = 80;
	//}

	Mat th1;
	/********边缘屏蔽模板*****/
	Mat th_img_gray;
	adaptiveThreshold(srcWhite, th_img_gray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 157, -3);//157 -3  //中部折痕检测四周屏蔽
	Mat element4 = getStructuringElement(MORPH_RECT, Size(7, 7)); //闭操作结构元素
	Mat elementb = getStructuringElement(MORPH_RECT, Size(17, 17)); //闭操作结构元素
	morphologyEx(th_img_gray, th_img_gray, CV_MOP_CLOSE, elementb);   //闭运算形态学操作。可以减少噪点
	dilate(th_img_gray, th_img_gray, element4);
	Mat tempMasklun = Mat::zeros(th_img_gray.size(), CV_8UC1);
	//bitwise_or(th_img_gray, ~tempMasklun, th_img_gray);
	frontSrcTh = Mat::zeros(Size(3000 - 2 * border, 1500 - 2 * border), CV_8UC1);//中部完全屏蔽
	frontSrcTh.copyTo(th_img_gray(Rect(border, border, srcWhite.cols - 2 * border, srcWhite.rows - 2 * border)));

	//---------------图像增强-------------//
	Ptr<CLAHE> clahe = createCLAHE(2, Size(40, 40));
	Ptr<CLAHE> clahebian = createCLAHE(2, Size(60, 60));
	Mat img_gray_clahe;
	clahe->apply(srcWhite, img_gray_clahe);   //整图增强

	Mat th_img_gray_clahe;
	Mat th_img_gray_clahebian;
	Mat th_img_gray_clahebian1;

	Mat c5;
	if (meanfront > 10) {
		c5 = img_gray_clahe + (img_gray_clahe - srcWhite);//frontWhite ++(img_gray_clahe-frontWhite)
	}
	else {
		c5 = srcWhite;//frontWhite +
	}

	//  c5 = Gabor7Crease(c5);
	  //---------------最终分割-------------//
	adaptiveThreshold(c5, th_img_gray_clahe, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 71, -3);  //-3.8
	morphologyEx(th_img_gray_clahe, th_img_gray_clahe, MORPH_OPEN, element4);   //开运算形态学操作。可以减少噪点
   // th_img_gray_clahebian.copyTo(th_img_gray_clahe(Rect(0, 0, 200, th_img_gray_clahe.rows)));//尾部替代
  //  th_img_gray_clahebian1.copyTo(th_img_gray_clahe(Rect(2800, 0, 200, th_img_gray_clahe.rows)));//尾部替代
	Mat element21 = getStructuringElement(MORPH_RECT, Size(27, 27));//闭操作结构元素
	Mat frontSrcTh2;
	medianBlur(th_img_gray_clahe, th_img_gray_clahe, 17);//去除白色噪点
	//---------------中部折痕边缘屏蔽-------------//
	int boder = 22;//5-15
	bitwise_and(th_img_gray_clahe, ~th_img_gray, th_img_gray_clahe);
	/************************************暂时*/
//    th_img_gray_clahe(Rect(0, 0, creasePara->blackSideLeft, th_img_gray_clahe.rows)) = uchar(0);
//    th_img_gray_clahe(Rect(0, 0, th_img_gray_clahe.cols, creasePara->blackSideUp)) = uchar(0);
//    th_img_gray_clahe(Rect(0, th_img_gray_clahe.rows - creasePara->blackSideDown, th_img_gray_clahe.cols, creasePara->blackSideDown)) = uchar(0);
//    th_img_gray_clahe(Rect(th_img_gray_clahe.cols - creasePara->blackSideRight, 0, creasePara->blackSideRight, th_img_gray_clahe.rows)) = uchar(0);
	Mat c6;
	c6 = c5(Rect(150, 150, 2700, 1300));

	//  morphologyEx(th_img_gray_clahe, frontSrcTh2, CV_MOP_CLOSE, element21);   //闭运算形态学操作。可以减少噪点
	frontSrcTh2 = th_img_gray_clahe - shieldMask;
	//---------------边缘折痕检测处理-------------//

	imwrite("D:\\s.bmp", c5);

	//手动屏蔽//
	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("02").value().find("01").value(), frontSrcTh2, *creasePara->m_shieldshape.find("02").value().find("01").value());
	//---------------筛选轮廓-------------//
	//imwrite(rootPath1 + "//result4//" + namenum.c_str() + ".bmp", th_result);
	vector<vector<Point>>contours;
	findContours(frontSrcTh2, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);
	vector<Rect> boundRect(contours.size());
	//     imwrite( "D://6.bmp", frontSrcTh2);
	for (vector<int> ::size_type i = 0; i < contours.size(); i++) {

		Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
		drawContours(tempMask, contours, i, 255, 1, 8);
		double area = contourArea(contours[i]);
		if (area < 1200) {
			break;
		}

		Point2f RectPoint[4];
		RotatedRect  externrect = minAreaRect(Mat(contours[i]));
		double mw2 = externrect.size.height;
		double mh2 = externrect.size.width;
		Point2f center = externrect.center;  //中心
		//特征一： 长宽比
		double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
		//特征二： 最大长度
		double length = max(mw2, mh2);
		double width = min(mw2, mh2);
		boundRect[i] = boundingRect(Mat(contours[i]));
		float w = boundRect[i].width;
		float h = boundRect[i].height;
		float w1 = max(h, w);;
		int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
		int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
		int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
		int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
		double longShortRatio = max(h / w, w / h);
		if (radio2 > 2 && length > 200 && width > 3 && w1 > 150)//长宽比，长度，宽度限制  radio2 > 2 && length > 200 && width > 3 && w1 > 150
		{
			int border_x = 10;//选定框边界宽度
			int border_y = 10;//选定框边界宽度
			if (area > 10000) {

				if (w > h) {
					border_x = 5;
					border_y = 10;
				}
				else {
					border_x = 10;
					border_y = 5;
				}
			}
			else {

				if (w > h) {
					border_x = 3;
					border_y = 5;
				}
				else {
					border_x = 5;
					border_y = 3;
				}
			}

			int x_lt = X_1 - border_x;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border_y;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border_x;
			if (x_rt > frontSrcTh2.size[1] - 1)
			{
				x_rt = frontSrcTh2.size[1] - 1;
			}
			int y_rt = Y_2 + border_y;
			if (y_rt > frontSrcTh2.size[0] - 1)
			{
				y_rt = frontSrcTh2.size[0] - 1;
			}
			//Mat sidelightSuspect = frontSrc(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//侧光图像疑似贴膜划痕图像
			Mat whitelightSuspect = white(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
			Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜

			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
			double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
			cv::Mat meanGray;
			cv::Mat stdDev;
			cv::meanStdDev(whitelightSuspect, meanGray, stdDev);


			if (area > 4500) {
				if (stdDev.at<double>(0, 0) > 1 && stdDev.at<double>(0, 0) < 15)
				{
					*causecolor = "打折主";
					result = true;
					CvPoint top_lef4 = cvPoint(x_lt, y_lt);
					CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
					rectangle(white, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
					//                    imwrite( "D://1.bmp", frontSrcTh2);
					*mresult = white;
					break;
				}

			}


		}
	}
	return result;
}

bool leftRightCrease(Mat left, Mat* mresult, string* causecolor) {

	bool result = false;
	//    left = Gabor7Crease(left);
	Mat leftshieldMask;
	threshold(left, leftshieldMask, 0.3 * mean(left)[0], 255, CV_THRESH_BINARY_INV);
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(45, 45));
	dilate(leftshieldMask, leftshieldMask, dilateElement);
	leftshieldMask(Rect(0, 0, leftshieldMask.cols, 30)) = uchar(255);
	leftshieldMask(Rect(0, leftshieldMask.rows - 40, leftshieldMask.cols, 40)) = uchar(255);
	leftshieldMask(Rect(0, 0, 30, leftshieldMask.rows)) = uchar(255);
	leftshieldMask(Rect(leftshieldMask.cols - 40, 0, 40, leftshieldMask.rows)) = uchar(255);




	Mat leftWhite = left.clone();
	GaussianBlur(leftWhite, leftWhite, Size(25, 25), 3);        //白底滤波
	Mat leftSrcTh;  //左相机阈值分割
	Mat leftSrcTh1;  //左相机阈值分



	//截取边框区域
	Mat imgTopTh;
	Mat imgButtomTh;
	Mat imgLeftTh;
	Mat imgRightTh;
	Mat imgTop = leftWhite(Rect(0, 0, leftWhite.cols - 1, 200));
	Mat imgButtom = leftWhite(Rect(0, 1300, leftWhite.cols - 1, 200));
	Mat imgLeft = leftWhite(Rect(0, 0, 200, leftWhite.rows));
	Mat imgRight = leftWhite(Rect(2800, 0, 200, leftWhite.rows));
	//---------------边界二值化-------------//
	adaptiveThreshold(imgTop, imgTopTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);  //0.8
	adaptiveThreshold(imgButtom, imgButtomTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);
	adaptiveThreshold(imgLeft, imgLeftTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);
	adaptiveThreshold(imgRight, imgRightTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);


	Mat th_img_gray;
	adaptiveThreshold(leftWhite, th_img_gray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 71, -2);//157 -3  //中部折痕检测四周屏蔽
	Mat element4 = getStructuringElement(MORPH_RECT, Size(7, 7)); //闭操作结构元素
	Mat elementb = getStructuringElement(MORPH_RECT, Size(17, 17)); //闭操作结构元素
	morphologyEx(th_img_gray, th_img_gray, CV_MOP_CLOSE, elementb);   //闭运算形态学操作。可以减少噪点
	leftSrcTh = Mat::zeros(Size(2800, 1300), CV_8UC1);//中部完全屏蔽
	leftSrcTh.copyTo(th_img_gray(Rect(100, 100, left.cols - 200, left.rows - 200)));

	//---------------图像增强-------------//
//    Ptr<CLAHE> clahe = createCLAHE(2, Size(40, 40));
//    Mat img_gray_clahe;
//    clahe->apply(leftWhite, img_gray_clahe);   //整图增强
	Mat th_img_gray_clahe;
	Mat th_img_gray_clahebian;
	//    medianBlur(img_gray_clahe, img_gray_clahe, 7);
	//    Mat c5 = (img_gray_clahe-leftWhite)*0.9 + img_gray_clahe;//frontWhite +

		//---------------最终分割-------------//
	adaptiveThreshold(leftWhite, th_img_gray_clahe, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 45, -2);
	//    morphologyEx(th_img_gray_clahe, th_img_gray_clahe, MORPH_OPEN, element4);   //开运算形态学操作。可以减少噪点
	Mat element21 = getStructuringElement(MORPH_RECT, Size(27, 27));//闭操作结构元素
	Mat leftSrcTh2;
	medianBlur(th_img_gray_clahe, th_img_gray_clahe, 7);//去除白色噪点   此处 7 17
	//---------------中部折痕边缘屏蔽-------------//
//        int boder = 35;//5-15
	bitwise_and(th_img_gray_clahe, ~th_img_gray, th_img_gray_clahe);
	th_img_gray_clahe = th_img_gray_clahe - leftshieldMask; //2022.3.28am
	morphologyEx(th_img_gray_clahe, leftSrcTh2, CV_MOP_CLOSE, element21);   //闭运算形态学操作。可以减少噪点

   //---------------边缘折痕检测处理-------------//
	Mat leftSrcThb = Mat::zeros(Size(3000, 1500), CV_8UC1);
	imgTopTh.copyTo(leftSrcThb(Rect(0, 0, th_img_gray_clahe.cols - 1, 200)));
	imgButtomTh.copyTo(leftSrcThb(Rect(0, 1300, th_img_gray_clahe.cols - 1, 200)));
	imgLeftTh.copyTo(leftSrcThb(Rect(0, 0, 200, th_img_gray_clahe.rows)));
	imgRightTh.copyTo(leftSrcThb(Rect(2800, 0, 200, th_img_gray_clahe.rows)));
	//===========================================R角================================================//
	Mat th_result = leftSrcThb;
	//    imwrite("D:\\leftSrcThb.bmp", leftSrcThb);
	th_result = th_result - leftshieldMask; //2.22.3.28 am //模板生成有问题
	morphologyEx(th_result, th_result, MORPH_OPEN, element4);   //闭运算形态学操作。可以减少噪点
	morphologyEx(th_result, th_result, CV_MOP_CLOSE, element4);   //闭运算形态学操作。可以减少噪点
//    imwrite("D:\\th_result.bmp", th_result);

	//手动屏蔽//
	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("02").value().find("02").value(), leftSrcTh2, *creasePara->m_shieldshape.find("02").value().find("02").value());
	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("02").value().find("02").value(), th_result, *creasePara->m_shieldshape.find("02").value().find("02").value());
	//---------------筛选轮廓-------------//
	//imwrite(rootPath1 + "//result4//" + namenum.c_str() + ".bmp", th_result);
	vector<vector<Point>>contours;
	findContours(leftSrcTh2, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	//imwrite("D://My_debug//verify//leftSrcTh2.bmp", leftSrcTh2);
	sort(contours.begin(), contours.end(), compareContourAreas);
	vector<Rect> boundRect(contours.size());
	for (vector<int> ::size_type i = 0; i < contours.size(); i++) {

		Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
		drawContours(tempMask, contours, i, 255, 1, 8);
		double area = contourArea(contours[i]);
		if (area < 1000) {
			break;
		}

		Point2f RectPoint[4];
		RotatedRect  externrect = minAreaRect(Mat(contours[i]));
		double mw2 = externrect.size.height;
		double mh2 = externrect.size.width;
		Point2f center = externrect.center;  //中心
		//特征一： 长宽比
		double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
		//特征二： 最大长度
		double length = max(mw2, mh2);
		double width = min(mw2, mh2);
		boundRect[i] = boundingRect(Mat(contours[i]));
		float w = boundRect[i].width;
		float h = boundRect[i].height;
		float w1 = max(h, w);;
		int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
		int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
		int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
		int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
		double longShortRatio = max(h / w, w / h);
		if (radio2 > 1.8 && radio2 < 24 && length > 100 && width > 4 && w1 > 100)//长宽比，长度，宽度限制  radio2 > 2 && length > 200 && width > 3 && w1 > 150
		{
			int border_x = 10;//选定框边界宽度
			int border_y = 10;//选定框边界宽度
			if (area > 10000) {

				if (w > h) {
					border_x = 5;
					border_y = 10;
				}
				else {
					border_x = 10;
					border_y = 5;
				}
			}
			else {

				if (w > h) {
					border_x = 3;
					border_y = 5;
				}
				else {
					border_x = 5;
					border_y = 3;
				}
			}

			int x_lt = X_1 - border_x;
			//越界保护
			if (x_lt < 0)
			{
				x_lt = 0;
			}
			int y_lt = Y_1 - border_y;
			if (y_lt < 0)
			{
				y_lt = 0;
			}
			int x_rt = X_2 + border_x;
			if (x_rt > leftSrcTh2.size[1] - 1)
			{
				x_rt = leftSrcTh2.size[1] - 1;
			}
			int y_rt = Y_2 + border_y;
			if (y_rt > leftSrcTh2.size[0] - 1)
			{
				y_rt = leftSrcTh2.size[0] - 1;
			}

			Mat whitelightSuspect = left(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
			Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
			double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
			double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
			double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
			cv::Mat meanGray;
			cv::Mat stdDev;
			cv::meanStdDev(whitelightSuspect, meanGray, stdDev);
			/*string c, c1;
			num2string(center.x, c);
			num2string(center.y, c1);
			string c2 = "(";
			c2 = c2 + c + "和" + c1 + ")";*/
			//fprintf(pOutFile, "%s,%d,  %f,  %f, %f, %f,%f, %f,%f,%s\n", namenum.c_str(), i + 1, area, length, radio2, removeScratch, removeScratch1, stdDev1.at<double>(0, 0), stdDev.at<double>(0, 0), c2.c_str());
			if (area > 2400) {
				if (stdDev.at<double>(0, 0) > 1 && stdDev.at<double>(0, 0) < 28 && abs(removeScratch1) > 0.9)
				{
					*causecolor = "折痕左";

					/*QList<QString> creaseList;

					if (Flag_Running_State == "Offline") {
						QString fname = QString::fromStdString(filenames[currentSample]);
						fileInfod = QFileInfo(fname);
						creaseList.append(fileInfod.fileName());
					}
					else {
						creaseList.append(QString::number(dianliang_num));
					}
					creaseList.append(QString::number(radio2));
					creaseList.append(QString::number(length));
					creaseList.append(QString::number(area));
					creaseList.append(QString::number(stdDev.at<double>(0, 0)));*/
					//creaseList.append(*causecolor);
					//CreaseFeatures.append(creaseList);

					//                    shielddisplay6.push_back("11");
					//                    shielddisplay6.push_back("02");
					//                    shielddisplay6.push_back("02");
					/*shielddisplay6["11"].push_back("11");
					shielddisplay6["11"].push_back("02");
					shielddisplay6["11"].push_back("02");*/

					result = true;
					CvPoint top_lef4 = cvPoint(x_lt, y_lt);
					CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
					rectangle(left, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
					*mresult = left;
					break;
				}
			}
		}
	}
	if (result == false) {
		vector<vector<Point>>contours;
		findContours(th_result, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		//bSums(imgTopTh);
		//findContours(imgTopTh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Rect> boundRect(contours.size());
		for (vector<int> ::size_type i = 0; i < contours.size(); i++) {

			Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
			drawContours(tempMask, contours, i, 255, 1, 8);
			double area = contourArea(contours[i]);
			if (area < 150) {
				break;
			}

			Point2f RectPoint[4];
			RotatedRect  externrect = minAreaRect(Mat(contours[i]));
			double mw2 = externrect.size.height;
			double mh2 = externrect.size.width;
			Point2f center = externrect.center;  //中心
			//特征一： 长宽比
			double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
			//特征二： 最大长度
			double length = max(mw2, mh2);
			double width = min(mw2, mh2);
			boundRect[i] = boundingRect(Mat(contours[i]));
			float w = boundRect[i].width;
			float h = boundRect[i].height;
			float w1 = max(h, w);;
			int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
			double longShortRatio = max(h / w, w / h);

			if (radio2 > 2 && radio2 < 15 && length > 40 && width > 3)//长宽比，长度，宽度限制  radio2 > 2 && length > 200 && width > 3 && w1 > 150
			{
				int border_x = 10;//选定框边界宽度
				int border_y = 10;//选定框边界宽度
				if (area > 4000) {

					if (w > h) {
						border_x = 5;
						border_y = 10;
					}
					else {
						border_x = 10;
						border_y = 5;
					}
				}
				else {

					if (w > h) {
						border_x = 3;
						border_y = 5;
					}
					else {
						border_x = 5;
						border_y = 3;
					}
				}

				int x_lt = X_1 - border_x;
				//越界保护
				if (x_lt < 0)
				{
					x_lt = 0;
				}
				int y_lt = Y_1 - border_y;
				if (y_lt < 0)
				{
					y_lt = 0;
				}
				int x_rt = X_2 + border_x;
				if (x_rt > leftSrcTh2.size[1] - 1)
				{
					x_rt = leftSrcTh2.size[1] - 1;
				}
				int y_rt = Y_2 + border_y;
				if (y_rt > leftSrcTh2.size[0] - 1)
				{
					y_rt = leftSrcTh2.size[0] - 1;
				}

				Mat whitelightSuspect = left(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
				Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜
				double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
				double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
				double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
				double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
				cv::Mat meanGray;
				cv::Mat stdDev;
				cv::meanStdDev(whitelightSuspect, meanGray, stdDev);

				if (area > 150) {
					if (stdDev.at<double>(0, 0) > 2.5 && abs(removeScratch1) > 0.9)
					{
						*causecolor = "折痕边";
						//QList<QString> creaseList;

						/*if (Flag_Running_State == "Offline") {
							QString fname = QString::fromStdString(filenames[currentSample]);
							fileInfod = QFileInfo(fname);
							creaseList.append(fileInfod.fileName());
						}
						else {
							creaseList.append(QString::number(dianliang_num));
						}
						creaseList.append(QString::number(radio2));
						creaseList.append(QString::number(length));
						creaseList.append(QString::number(area));
						creaseList.append(QString::number(stdDev.at<double>(0, 0)));
						creaseList.append(*causecolor);
						CreaseFeatures.append(creaseList);*/
						result = true;
						CvPoint top_lef4 = cvPoint(x_lt, y_lt);
						CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
						rectangle(left, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
						*mresult = left;
						break;
					}

				}


			}



		}
	}

	if (result == false) {

		vector<vector<Point>>contours;
		//Mat Top_End_Find = imgTopTh;
		Mat Top_End_Find = imgButtomTh;
		imwrite("D:\\Test_result\\V_H\\imgTopTh.bmp", imgTopTh);
		int Index_My_Bound =  bSums(imgTopTh);
		//Top_End_Find(Rect(0, Index_My_Bound, Top_End_Find.cols, Index_My_Bound)) = uchar(0);
		findContours(Top_End_Find, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		//bSums(imgTopTh);
		//findContours(imgTopTh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Rect> boundRect(contours.size());
		for (vector<int> ::size_type i = 0; i < contours.size(); i++) {

			Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
			drawContours(tempMask, contours, i, 255, 1, 8);
			double area = contourArea(contours[i]);
			if (area < 100) {
				break;
			}

			Point2f RectPoint[4];
			RotatedRect  externrect = minAreaRect(Mat(contours[i]));
			double mw2 = externrect.size.height;
			double mh2 = externrect.size.width;
			Point2f center = externrect.center;  //中心
			//特征一： 长宽比
			double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
			//特征二： 最大长度
			double length = max(mw2, mh2);
			double width = min(mw2, mh2);
			boundRect[i] = boundingRect(Mat(contours[i]));
			float w = boundRect[i].width;
			float h = boundRect[i].height;
			float w1 = max(h, w);;
			int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
			double longShortRatio = max(h / w, w / h);

			if (radio2 > 1.9 && radio2 < 15 && length > 40 && width > 3)//长宽比，长度，宽度限制  radio2 > 2 && length > 200 && width > 3 && w1 > 150
			{
				int border_x = 10;//选定框边界宽度
				int border_y = 10;//选定框边界宽度
				if (area > 4000) {

					if (w > h) {
						border_x = 5;
						border_y = 10;
					}
					else {
						border_x = 10;
						border_y = 5;
					}
				}
				else {

					if (w > h) {
						border_x = 3;
						border_y = 5;
					}
					else {
						border_x = 5;
						border_y = 3;
					}
				}

				int x_lt = X_1 - border_x;
				//越界保护
				if (x_lt < 0)
				{
					x_lt = 0;
				}
				int y_lt = Y_1 - border_y;
				if (y_lt < 0)
				{
					y_lt = 0;
				}
				int x_rt = X_2 + border_x;
				if (x_rt > leftSrcTh2.size[1] - 1)
				{
					x_rt = leftSrcTh2.size[1] - 1;
				}
				int y_rt = Y_2 + border_y;
				if (y_rt > leftSrcTh2.size[0] - 1)
				{
					y_rt = leftSrcTh2.size[0] - 1;
				}

				Mat whitelightSuspect = left(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
				Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜
				double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
				double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
				double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
				double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
				cv::Mat meanGray;
				cv::Mat stdDev;
				cv::meanStdDev(whitelightSuspect, meanGray, stdDev);

				if (area > 150) {
					if (stdDev.at<double>(0, 0) > 2.5 && abs(removeScratch1) > 0.9)
					{
						*causecolor = "折痕边";
						//QList<QString> creaseList;

						/*if (Flag_Running_State == "Offline") {
							QString fname = QString::fromStdString(filenames[currentSample]);
							fileInfod = QFileInfo(fname);
							creaseList.append(fileInfod.fileName());
						}
						else {
							creaseList.append(QString::number(dianliang_num));
						}
						creaseList.append(QString::number(radio2));
						creaseList.append(QString::number(length));
						creaseList.append(QString::number(area));
						creaseList.append(QString::number(stdDev.at<double>(0, 0)));
						creaseList.append(*causecolor);
						CreaseFeatures.append(creaseList);*/
						result = true;
						CvPoint top_lef4 = cvPoint(x_lt, y_lt);
						CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
						rectangle(left, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
						*mresult = left;
						break;
					}

				}


			}



		}
	}
	return result;
}
//bool leftRightCrease(Mat left, Mat* mresult, string* causecolor) {
//
//	bool result = false;
//	//    left = Gabor7Crease(left);
//	Mat leftshieldMask;
//	threshold(left, leftshieldMask, 0.3 * mean(left)[0], 255, CV_THRESH_BINARY_INV);
//	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(45, 45));
//	dilate(leftshieldMask, leftshieldMask, dilateElement);
//	leftshieldMask(Rect(0, 0, leftshieldMask.cols, 30)) = uchar(255);
//	leftshieldMask(Rect(0, leftshieldMask.rows - 40, leftshieldMask.cols, 40)) = uchar(255);
//	leftshieldMask(Rect(0, 0, 30, leftshieldMask.rows)) = uchar(255);
//	leftshieldMask(Rect(leftshieldMask.cols - 40, 0, 40, leftshieldMask.rows)) = uchar(255);
//
//
//
//	Mat leftWhite = left.clone();
//	GaussianBlur(leftWhite, leftWhite, Size(25, 25), 3);        //白底滤波
//	Mat leftSrcTh;  //左相机阈值分割
//	Mat leftSrcTh1;  //左相机阈值分
//
//
//
//	//截取边框区域
//	Mat imgTopTh;
//	Mat imgButtomTh;
//	Mat imgLeftTh;
//	Mat imgRightTh;
//	Mat imgTop = leftWhite(Rect(0, 0, leftWhite.cols - 1, 200));
//	Mat imgButtom = leftWhite(Rect(0, 1300, leftWhite.cols - 1, 200));
//	Mat imgLeft = leftWhite(Rect(0, 0, 200, leftWhite.rows));
//	Mat imgRight = leftWhite(Rect(2800, 0, 200, leftWhite.rows));
//	//---------------边界二值化-------------//
//	adaptiveThreshold(imgTop, imgTopTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);  //0.8
//	adaptiveThreshold(imgButtom, imgButtomTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);
//	adaptiveThreshold(imgLeft, imgLeftTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);
//	adaptiveThreshold(imgRight, imgRightTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 19, 1);
//
//
//	Mat th_img_gray;
//	adaptiveThreshold(leftWhite, th_img_gray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 71, -2);//157 -3  //中部折痕检测四周屏蔽
//	Mat element4 = getStructuringElement(MORPH_RECT, Size(7, 7)); //闭操作结构元素
//	Mat elementb = getStructuringElement(MORPH_RECT, Size(17, 17)); //闭操作结构元素
//	morphologyEx(th_img_gray, th_img_gray, CV_MOP_CLOSE, elementb);   //闭运算形态学操作。可以减少噪点
//	leftSrcTh = Mat::zeros(Size(2800, 1300), CV_8UC1);//中部完全屏蔽
//	leftSrcTh.copyTo(th_img_gray(Rect(100, 100, left.cols - 200, left.rows - 200)));
//
//	//---------------图像增强-------------//
//	//Ptr<CLAHE> clahe = createCLAHE(2, Size(40, 40));
//	//Mat img_gray_clahe;
//	//clahe->apply(leftWhite, img_gray_clahe);   //整图增强
//	Mat th_img_gray_clahe;
//	Mat th_img_gray_clahebian;
//	//medianBlur(img_gray_clahe, img_gray_clahe, 7);
//	//Mat c5 = (img_gray_clahe - leftWhite) * 0.9 + img_gray_clahe;//frontWhite +
//
//	//---------------最终分割-------------//
//	adaptiveThreshold(leftWhite, th_img_gray_clahe, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 33, -1);
//	morphologyEx(th_img_gray_clahe, th_img_gray_clahe, MORPH_OPEN, element4);   //开运算形态学操作。可以减少噪点
//	Mat element21 = getStructuringElement(MORPH_RECT, Size(27, 27));//闭操作结构元素
//	Mat leftSrcTh2;
//	//medianBlur(th_img_gray_clahe, th_img_gray_clahe, 17);//去除白色噪点
//	//---------------中部折痕边缘屏蔽-------------//
////        int boder = 35;//5-15
//	bitwise_and(th_img_gray_clahe, ~th_img_gray, th_img_gray_clahe);
//	th_img_gray_clahe = th_img_gray_clahe - leftshieldMask; //2022.3.28am
//	morphologyEx(th_img_gray_clahe, leftSrcTh2, CV_MOP_CLOSE, element21);   //闭运算形态学操作。可以减少噪点
//
//   //---------------边缘折痕检测处理-------------//
//	Mat leftSrcThb = Mat::zeros(Size(3000, 1500), CV_8UC1);
//	imgTopTh.copyTo(leftSrcThb(Rect(0, 0, th_img_gray_clahe.cols - 1, 200)));
//	imgButtomTh.copyTo(leftSrcThb(Rect(0, 1300, th_img_gray_clahe.cols - 1, 200)));
//	imgLeftTh.copyTo(leftSrcThb(Rect(0, 0, 200, th_img_gray_clahe.rows)));
//	imgRightTh.copyTo(leftSrcThb(Rect(2800, 0, 200, th_img_gray_clahe.rows)));
//	//===========================================R角================================================//
//	Mat th_result = leftSrcThb;
//	//    imwrite("D:\\leftSrcThb.bmp", leftSrcThb);
//	th_result = th_result - leftshieldMask; //2.22.3.28 am
//	morphologyEx(th_result, th_result, MORPH_OPEN, element4);   //闭运算形态学操作。可以减少噪点
//	morphologyEx(th_result, th_result, CV_MOP_CLOSE, element4);   //闭运算形态学操作。可以减少噪点
////    imwrite("D:\\th_result.bmp", th_result);
//
//	//手动屏蔽//
//	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("02").value().find("02").value(), leftSrcTh2, *creasePara->m_shieldshape.find("02").value().find("02").value());
//	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("02").value().find("02").value(), th_result, *creasePara->m_shieldshape.find("02").value().find("02").value());
//	//---------------筛选轮廓-------------//
//	//imwrite(rootPath1 + "//result4//" + namenum.c_str() + ".bmp", th_result);
//	vector<vector<Point>>contours;
//	findContours(leftSrcTh2, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
//	sort(contours.begin(), contours.end(), compareContourAreas);
//	vector<Rect> boundRect(contours.size());
//	for (vector<int> ::size_type i = 0; i < contours.size(); i++) {
//
//		Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
//		drawContours(tempMask, contours, i, 255, 1, 8);
//		double area = contourArea(contours[i]);
//		if (area < 2400) {
//			break;
//		}
//
//		Point2f RectPoint[4];
//		RotatedRect  externrect = minAreaRect(Mat(contours[i]));
//		double mw2 = externrect.size.height;
//		double mh2 = externrect.size.width;
//		Point2f center = externrect.center;  //中心
//		//特征一： 长宽比
//		double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
//		//特征二： 最大长度
//		double length = max(mw2, mh2);
//		double width = min(mw2, mh2);
//		boundRect[i] = boundingRect(Mat(contours[i]));
//		float w = boundRect[i].width;
//		float h = boundRect[i].height;
//		float w1 = max(h, w);;
//		int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
//		int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
//		int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
//		int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
//		double longShortRatio = max(h / w, w / h);
//		if (radio2 > 1.8 && radio2 < 7 && length > 100 && width > 4 && w1 > 100)//长宽比，长度，宽度限制  radio2 > 2 && length > 200 && width > 3 && w1 > 150
//		{
//			int border_x = 10;//选定框边界宽度
//			int border_y = 10;//选定框边界宽度
//			if (area > 10000) {
//
//				if (w > h) {
//					border_x = 5;
//					border_y = 10;
//				}
//				else {
//					border_x = 10;
//					border_y = 5;
//				}
//			}
//			else {
//
//				if (w > h) {
//					border_x = 3;
//					border_y = 5;
//				}
//				else {
//					border_x = 5;
//					border_y = 3;
//				}
//			}
//
//			int x_lt = X_1 - border_x;
//			//越界保护
//			if (x_lt < 0)
//			{
//				x_lt = 0;
//			}
//			int y_lt = Y_1 - border_y;
//			if (y_lt < 0)
//			{
//				y_lt = 0;
//			}
//			int x_rt = X_2 + border_x;
//			if (x_rt > leftSrcTh2.size[1] - 1)
//			{
//				x_rt = leftSrcTh2.size[1] - 1;
//			}
//			int y_rt = Y_2 + border_y;
//			if (y_rt > leftSrcTh2.size[0] - 1)
//			{
//				y_rt = leftSrcTh2.size[0] - 1;
//			}
//
//			Mat whitelightSuspect = left(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
//			Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜
//			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
//			double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
//			double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
//			double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
//			cv::Mat meanGray;
//			cv::Mat stdDev;
//			cv::meanStdDev(whitelightSuspect, meanGray, stdDev);
//			/*string c, c1;
//			num2string(center.x, c);
//			num2string(center.y, c1);
//			string c2 = "(";
//			c2 = c2 + c + "和" + c1 + ")";*/
//			//fprintf(pOutFile, "%s,%d,  %f,  %f, %f, %f,%f, %f,%f,%s\n", namenum.c_str(), i + 1, area, length, radio2, removeScratch, removeScratch1, stdDev1.at<double>(0, 0), stdDev.at<double>(0, 0), c2.c_str());
//			if (area > 2400) {
//				if (stdDev.at<double>(0, 0) > 1 && stdDev.at<double>(0, 0) < 10 && abs(removeScratch1) > 0.9)
//				{
//					*causecolor = "折痕左";
//
//					//QList<QString> creaseList;
//					//creaseList.append(QString::number(radio2));
//					//creaseList.append(QString::number(length));
//					//creaseList.append(QString::number(area));
//					//creaseList.append(QString::number(stdDev.at<double>(0, 0)));
//					//creaseList.append(*causecolor);
//					//CreaseFeatures.append(creaseList);
//
//
//					result = true;
//					CvPoint top_lef4 = cvPoint(x_lt, y_lt);
//					CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
//					rectangle(left, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
//					*mresult = left;
//					break;
//				}
//			}
//		}
//	}
//	if (result == false) {
//		vector<vector<Point>>contours;
//		findContours(th_result, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
//		sort(contours.begin(), contours.end(), compareContourAreas);
//		vector<Rect> boundRect(contours.size());
//		for (vector<int> ::size_type i = 0; i < contours.size(); i++) {
//
//			Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
//			drawContours(tempMask, contours, i, 255, 1, 8);
//			double area = contourArea(contours[i]);
//			if (area < 150) {
//				break;
//			}
//
//			Point2f RectPoint[4];
//			RotatedRect  externrect = minAreaRect(Mat(contours[i]));
//			double mw2 = externrect.size.height;
//			double mh2 = externrect.size.width;
//			Point2f center = externrect.center;  //中心
//			//特征一： 长宽比
//			double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
//			//特征二： 最大长度
//			double length = max(mw2, mh2);
//			double width = min(mw2, mh2);
//			boundRect[i] = boundingRect(Mat(contours[i]));
//			float w = boundRect[i].width;
//			float h = boundRect[i].height;
//			float w1 = max(h, w);;
//			int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
//			int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
//			int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
//			int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
//			double longShortRatio = max(h / w, w / h);
//
//			if (radio2 > 2 && radio2 < 7 && length > 40 && width > 3)//长宽比，长度，宽度限制  radio2 > 2 && length > 200 && width > 3 && w1 > 150
//			{
//				int border_x = 10;//选定框边界宽度
//				int border_y = 10;//选定框边界宽度
//				if (area > 4000) {
//
//					if (w > h) {
//						border_x = 5;
//						border_y = 10;
//					}
//					else {
//						border_x = 10;
//						border_y = 5;
//					}
//				}
//				else {
//
//					if (w > h) {
//						border_x = 3;
//						border_y = 5;
//					}
//					else {
//						border_x = 5;
//						border_y = 3;
//					}
//				}
//
//				int x_lt = X_1 - border_x;
//				//越界保护
//				if (x_lt < 0)
//				{
//					x_lt = 0;
//				}
//				int y_lt = Y_1 - border_y;
//				if (y_lt < 0)
//				{
//					y_lt = 0;
//				}
//				int x_rt = X_2 + border_x;
//				if (x_rt > leftSrcTh2.size[1] - 1)
//				{
//					x_rt = leftSrcTh2.size[1] - 1;
//				}
//				int y_rt = Y_2 + border_y;
//				if (y_rt > leftSrcTh2.size[0] - 1)
//				{
//					y_rt = leftSrcTh2.size[0] - 1;
//				}
//
//				Mat whitelightSuspect = left(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
//				Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜
//				double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
//				double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
//				double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
//				double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
//				cv::Mat meanGray;
//				cv::Mat stdDev;
//				cv::meanStdDev(whitelightSuspect, meanGray, stdDev);
//
//				if (area > 150) {
//					if (stdDev.at<double>(0, 0) > 2.5 && abs(removeScratch1) > 0.9)
//					{
//						*causecolor = "折痕边";
//						//QList<QString> creaseList;
//						//creaseList.append(QString::number(radio2));
//						//creaseList.append(QString::number(length));
//						//creaseList.append(QString::number(area));
//						//creaseList.append(QString::number(stdDev.at<double>(0, 0)));
//						//creaseList.append(*causecolor);
//						//CreaseFeatures.append(creaseList);
//						result = true;
//						CvPoint top_lef4 = cvPoint(x_lt, y_lt);
//						CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
//						rectangle(left, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
//						*mresult = left;
//						break;
//					}
//
//				}
//
//
//			}
//
//
//
//		}
//	}
//
//
//	return result;
//}
bool Crease(Mat frontSideLight, Mat front, Mat left, Mat* mresult, string* causecolor, bool leftflag) {
	bool result = false;
	//flip(front, front, -1);
	//2022.3.28 am
	/**********屏蔽模板**********/
	Mat shieldMask;
	double temp = mean(front)[0];
	threshold(front, shieldMask, 0.3 * mean(front)[0], 255, CV_THRESH_BINARY_INV);
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(29, 29));
	dilate(shieldMask, shieldMask, dilateElement);
	shieldMask(Rect(0, 0, shieldMask.cols, 0)) = uchar(255);
	shieldMask(Rect(0, shieldMask.rows - 0, shieldMask.cols, 0)) = uchar(255);
	shieldMask(Rect(0, 0, 0, shieldMask.rows)) = uchar(255);
	shieldMask(Rect(shieldMask.cols - 0, 0, 0, shieldMask.rows)) = uchar(255);



	Mat frontWhite = front.clone();
	//frontWhite = Gabor7Crease(frontWhite);
	Mat smallSrc = frontWhite.clone();

	GaussianBlur(frontWhite, frontWhite, Size(25, 25), 5);        //白底滤波  *creasePara->GaussianStv

	Mat frontSrcTh;  //前相机阈值分割
	Mat frontSrcTh1;  //前相机阈值分


	/******************相似度计算***********/
	Mat test;
	adaptiveThreshold(front, test, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -1);
	Mat mean1;
	Mat std1;
	meanStdDev(front, mean1, std1);
	double psnr;
	double grad;
	double entro;
	psnr = getPSNR(test);
	grad = getGradient(front);
	entro = getEntropy(front);

	//cout << grad << " " << entro << " " << psnr << " " << std1.at<double>(0, 0) << " " << mean1.at<double>(0, 0) << endl;
	vector<double> stand;
	vector<double> var;

	stand.push_back(0.1269656);
	stand.push_back(6.368142);
	stand.push_back(40.54397);
	stand.push_back(31.10499);
	stand.push_back(77.34243);

	var.push_back(grad);
	var.push_back(entro);
	var.push_back(psnr);
	var.push_back(std1.at<double>(0, 0));
	var.push_back(mean1.at<double>(0, 0));
	double ssim = SSIM(stand, var);
	cout << ssim << endl;

	/*******************相似度计算**********/


	Mat smallSrcTh;

	Mat shieldMask1;
	threshold(front, shieldMask1, 0.3 * mean(front)[0], 255, CV_THRESH_BINARY_INV);
	Mat dilateElement1 = getStructuringElement(MORPH_RECT, Size(35, 35));
	dilate(shieldMask1, shieldMask1, dilateElement1);
	shieldMask1(Rect(0, 0, shieldMask1.cols, 40)) = uchar(255);
	shieldMask1(Rect(0, shieldMask1.rows - 40, shieldMask1.cols, 40)) = uchar(255);
	shieldMask1(Rect(0, 0, 35, shieldMask1.rows)) = uchar(255);
	shieldMask1(Rect(shieldMask1.cols - 35, 0, 35, shieldMask1.rows)) = uchar(255);

	//边缘小核
	Mat smallTopTh;
	Mat smallButtomTh;
	Mat smallLeftTh;
	Mat smallRightTh;


	Mat smallTop = smallSrc(Rect(0, 0, smallSrc.cols, 100));
	Mat smallButtom = smallSrc(Rect(0, 1400, smallSrc.cols, 100));
	Mat smallLeft = smallSrc(Rect(0, 0, 200, smallSrc.rows));
	Mat smallRight = smallSrc(Rect(2800, 0, 200, smallSrc.rows));


	adaptiveThreshold(smallTop, smallTopTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, -2);
	adaptiveThreshold(smallButtom, smallButtomTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, -2);
	adaptiveThreshold(smallLeft, smallLeftTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, 2.5);//19
	adaptiveThreshold(smallRight, smallRightTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, 2);//19

	adaptiveThreshold(smallSrc, smallSrcTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 39, -2);

	//边缘替换
	smallLeftTh = ~smallLeftTh;
	smallRightTh = ~smallRightTh;

	smallTopTh.copyTo(smallSrcTh(Rect(0, 0, smallSrc.cols, 100)));
	smallButtomTh.copyTo(smallSrcTh(Rect(0, 1400, smallSrc.cols, 100)));
	smallLeftTh.copyTo(smallSrcTh(Rect(0, 0, 200, smallSrc.rows)));
	smallRightTh.copyTo(smallSrcTh(Rect(2800, 0, 200, smallSrc.rows)));


	Mat smallElement = getStructuringElement(MORPH_RECT, Size(7, 7));
	morphologyEx(smallSrcTh, smallSrcTh, MORPH_CLOSE, smallElement);
	smallSrcTh = smallSrcTh - shieldMask;


	//DefectAlgorithm::shieldingOperate(*creasePara->m_shieldingArea.find("02").value().find("04").value(), smallSrcTh, *creasePara->m_shieldshape.find("02").value().find("04").value());
	vector<vector<Point>>contoursSmall;
	findContours(smallSrcTh, contoursSmall, RETR_LIST, CHAIN_APPROX_SIMPLE);


	sort(contoursSmall.begin(), contoursSmall.end(), compareContourAreas);
	vector<Rect> smallBoundRect(contoursSmall.size());
	for (vector<int> ::size_type i = 0; i < contoursSmall.size(); i++) {
		Mat smallMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
		drawContours(smallMask, contoursSmall, i, 255, -1, 8);
		double smallArea = contourArea(contoursSmall[i]);
		//qDebug() << smallArea << endl;
		//面积判定
		if (smallArea < 80) {  //280
			break;
		}

		//正交外接矩形
		smallBoundRect[i] = boundingRect(Mat(contoursSmall[i]));
		int smallX_1 = smallBoundRect[i].tl().x;//矩形左上角X坐标值
		int smallY_1 = smallBoundRect[i].tl().y;//矩形左上角Y坐标值
		int smallX_2 = smallBoundRect[i].br().x;//矩形右下角X坐标值
		int smallY_2 = smallBoundRect[i].br().y;//矩形右下角Y坐标值

		//长宽比判定
		RotatedRect  smallRotRect = minAreaRect(Mat(contoursSmall[i]));
		double smallHei = smallRotRect.size.height;
		double smallWid = smallRotRect.size.width;
		double smallRatio = max(smallHei / smallWid, smallWid / smallHei);

		//边界扩充->2022.5.9测试
		smallX_1 = smallX_1 - 10 < 0 ? 0 : smallX_1 - 10;
		smallY_1 = smallY_1 - 10 < 0 ? 0 : smallY_1 - 10;
		smallX_2 = smallX_2 + 10 > 2999 ? 2999 : smallX_2 + 10;
		smallY_2 = smallY_2 + 10 > 1499 ? 1499 : smallY_2 + 10;

		//限制条件 ->2022.4.27
		Mat whitelightSuspect = front(Rect(smallX_1, smallY_1, smallX_2 - smallX_1 - 1, smallY_2 - smallY_1 - 1));//白底图像疑似反光划痕图像
		Mat mask = smallMask(Rect(smallX_1, smallY_1, smallX_2 - smallX_1 - 1, smallY_2 - smallY_1 - 1));             //侧光图像疑似贴膜划痕掩膜

		double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
		double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
		double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
		cv::Mat meanGray;
		cv::Mat stdDev;
		cv::meanStdDev(whitelightSuspect, meanGray, stdDev);
		double coefficient = stdDev.at<double>(0) / meanGray.at<double>(0) * 100;    //&&meanGray.at<double>(0)<170&&coefficient>4&&stdDev.at<double>(0)>5

		//统计信息
		double psnr = getPSNR(mask);
		double gradient = getGradient(whitelightSuspect);
		double entropy = getEntropy(whitelightSuspect);


		//cout << smallArea << " " << smallRatio << " " << meanGray.at<double>(0) << " " << removeScratch1 << " " << stdDev.at<double>(0) << " " << psnr << " " << gradient << " " << entropy << endl;

		if (smallRatio > 2.8 && smallRatio < 20.0 && meanGray.at<double>(0) < 180 && (abs(removeScratch1) > 3 || stdDev.at<double>(0) > 2.5)) {
			*causecolor = "小折痕";
			result = true;
			CvPoint small_lt = cvPoint(smallX_1, smallY_1);
			CvPoint small_br = cvPoint(smallX_2, smallY_2);
			rectangle(front, small_lt, small_br, Scalar(255, 255, 255), 5, 8, 0);
			*mresult = front;
			//break;
		}
	}

	return result;
}
bool Dead_light0(Mat white, Mat sideLight, Mat* mresult, string* causecolor)
{

	//基本参数
	bool result = false;
	int detectionWidth = 200;
	int shieldEdge = 12;
	double sideLightTh = 0.5;
	int dilateSize = 7;
	int erodeSize = 29;
	double darkRatio = 0.05;
	double lightRatio = 0.2;

	//侧光图处理
	Mat mask;
	sideLight = sideLight(Rect(0, 0, detectionWidth, sideLight.rows));
	sideLight(Rect(0, 0, shieldEdge, sideLight.rows)) = uchar(0);
	sideLight(Rect(0, 0, sideLight.cols, 10)) = uchar(0);
	sideLight(Rect(0, sideLight.rows - 10, sideLight.cols, 10)) = uchar(0);
	threshold(sideLight, mask, sideLightTh * mean(sideLight)[0], 255, CV_THRESH_BINARY_INV);
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(dilateSize, dilateSize));
	dilate(mask, mask, dilateElement);

	//白底图处理
	Mat src = white.clone();
	Mat grayImg = white(Rect(0, 0, detectionWidth, white.rows)).clone();
	grayImg(Rect(0, 0, shieldEdge, white.rows)) = uchar(0);
	medianBlur(grayImg, grayImg, 5);


	//计算灰度直方图
	int histSize[1] = { 256 };          //灰度值Size：256个
	float hrange[2] = { 0, 255 };       //灰度范围[0-255]
	const float* ranges[1] = { hrange };//单个灰度范围[0-255]
	int channels = 0;
	Mat histMat;
	Mat out;
	calcHist(&grayImg, 1, &channels, ~mask, histMat, 1, histSize, ranges, true, false);

	//计算阈值
	int darkNum = 0, lightNum = 0;
	int darkTh = 0, lightTh = 0;
	for (int i = 0; i < histMat.rows; i++) {
		darkNum = darkNum + histMat.at<float>(i, 0);
		if (darkNum > detectionWidth * darkRatio * grayImg.rows) {
			darkTh = i;
			break;
		}
	}
	for (int i = histMat.rows - 1; i >= 0; i--) {
		lightNum = lightNum + histMat.at<float>(i, 0);
		if (lightNum > detectionWidth * lightRatio * grayImg.rows) {//0.05-0.2
			lightTh = i;
			break;
		}
	}

	//缺陷特征计算
	Mat normalImg, darkImg, lightImg;
	threshold(grayImg, darkImg, darkTh, 255, CV_THRESH_BINARY_INV);
	threshold(grayImg, lightImg, lightTh, 255, CV_THRESH_BINARY);
	normalImg = ~darkImg - lightImg;
	//腐蚀掉边缘
	Mat openElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat erodeElement = getStructuringElement(MORPH_CROSS, Size(erodeSize, erodeSize));
	morphologyEx(darkImg, darkImg, MORPH_CLOSE, openElement, Point(-1, -1), 1);
	erode(darkImg, darkImg, erodeElement);
	erode(lightImg, lightImg, erodeElement);
	darkImg = darkImg - mask;
	double normalMean = mean(grayImg(Rect(0, 0, detectionWidth, white.rows)), normalImg)[0];
	double darkMean = mean(grayImg(Rect(0, 0, detectionWidth, white.rows)), darkImg)[0];
	double dist = abs(normalMean - darkMean) / sqrt(2);

	//判断结果
	dist > 12 ? result = true : result = false;

	//cout << normalMean << " " << darkMean << endl;

	//dist > SDPara->Normalized_mean_difference ? result = true : result = false;//14.6--21
	if (result == true)
	{


		Mat img_dark, bian;//死灯暗区边框图
		dilate(darkImg, img_dark, erodeElement);//膨胀
		//显示暗区轮廓
		bian = (img_dark - darkImg) + white(Rect(0, 0, detectionWidth, darkImg.rows));
		bian.copyTo(white(Rect(0, 0, detectionWidth, darkImg.rows)));
		*mresult = white.clone();
		*causecolor = "死灯";
		result = true;
	}
	return result;
}

bool Crease_L(Mat left, Mat* mresult, string* causecolor) {
	bool result = false;


	return result;
}
bool Crease(Mat frontSideLight, Mat leftSideLight, Mat rightSideLight, Mat front, Mat* mresult, string* causecolor) {

	Mat frontSrc = frontSideLight.clone();
	Mat frontWhite = front.clone();

	GaussianBlur(frontSrc, frontSrc, Size(25, 25), 3);           //侧光滤波
	GaussianBlur(frontWhite, frontWhite, Size(25, 25), 3);        //白底滤波
	Mat a = front - frontWhite;
	bool result = false;
	Mat frontSrcTh;  //前相机阈值分割
	Mat frontSrcTh1;  //前相机阈值分割
	Mat dst, abs_dst;
	Mat binaryImage;
	medianBlur(front, front, 5);										//中值滤波去除锯齿


	//截取边框区域

	if (result == false) {

		medianBlur(frontWhite, frontWhite, 5);
		Mat src;
		//	截取边框区域
		Mat imgTopTh;
		Mat imgButtomTh;
		Mat imgLeftTh;
		Mat imgRightTh;
		Mat th1;
		Mat imgTop = frontWhite(Rect(0, 0, frontWhite.cols - 1, 200));
		Mat imgButtom = frontWhite(Rect(0, 1300, frontWhite.cols - 1, 200));
		Mat imgLeft = frontWhite(Rect(0, 0, 200, frontWhite.rows));
		Mat imgRight = frontWhite(Rect(2800, 0, 200, frontWhite.rows));
		adaptiveThreshold(imgTop, imgTopTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, -0.8);
		adaptiveThreshold(imgButtom, imgButtomTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, -0.8);
		adaptiveThreshold(imgLeft, imgLeftTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, -0.8);
		adaptiveThreshold(imgRight, imgRightTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, -0.8);
		//	边界进行深拷贝
			/*frontSrcTh = Mat::zeros(frontWhite.size(), CV_8UC1);
			imgTopTh.copyTo(frontSrcTh(Rect(0, 0, frontSrcTh.cols - 1, 200)));
			imgButtomTh.copyTo(frontSrcTh(Rect(0, 1300, frontSrc.cols - 1, 200)));
			imgLeftTh.copyTo(frontSrcTh(Rect(0, 0, 200, frontSrcTh.rows)));
			imgRightTh.copyTo(frontSrcTh(Rect(2800, 0, 200, frontSrc.rows)));*/
		frontSrcTh = Mat::zeros(Size(2600, 1100), CV_8UC1);

		Mat th_img_gray;
		adaptiveThreshold(frontWhite, th_img_gray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 157, -2);//157 -3
		Mat element4 = getStructuringElement(MORPH_RECT, Size(5, 5)); //闭操作结构元素
		morphologyEx(th_img_gray, th_img_gray, CV_MOP_CLOSE, element4);   //闭运算形态学操作。可以减少噪点
		frontSrcTh.copyTo(th_img_gray(Rect(200, 200, frontSrc.cols - 400, frontSrc.rows - 400)));

		Ptr<CLAHE> clahe = createCLAHE(2, Size(40, 40));
		Mat img_gray_clahe;
		clahe->apply(frontWhite, img_gray_clahe);
		Mat th_img_gray_clahe;

		adaptiveThreshold(img_gray_clahe, th_img_gray_clahe, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 101, -3);
		//bitwise_and(img_gray_clahe, ~th_img_gray, img_gray_clahe);
		Mat c5 = frontWhite + img_gray_clahe;//frontWhite +
		medianBlur(c5, c5, 5);
		adaptiveThreshold(c5, th_img_gray_clahe, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 101, -3);
		morphologyEx(th_img_gray_clahe, th_img_gray_clahe, MORPH_OPEN, element4);   //闭运算形态学操作。可以减少噪点

		Mat element21 = getStructuringElement(MORPH_RECT, Size(35, 35));//闭操作结构元素
		Mat frontSrcTh21;

		medianBlur(th_img_gray_clahe, th_img_gray_clahe, 17);
		int boder = 22;//5-15
		bitwise_and(th_img_gray_clahe, ~th_img_gray, th_img_gray_clahe);
		th_img_gray_clahe(Rect(0, 0, boder, th_img_gray_clahe.rows)) = uchar(0);
		th_img_gray_clahe(Rect(0, 0, th_img_gray_clahe.cols, boder)) = uchar(0);
		//frontSrcTh2(Rect(0, 430, 150, 580)) = uchar(0);//刘海屏幕
		//frontSrcTh21(Rect(th_img_gray_clahe.cols - 200, 0, 199, frontSrcTh.rows)) = uchar(0);
		th_img_gray_clahe(Rect(0, th_img_gray_clahe.rows - boder, th_img_gray_clahe.cols, boder)) = uchar(0);
		th_img_gray_clahe(Rect(th_img_gray_clahe.cols - boder, 0, boder, th_img_gray_clahe.rows)) = uchar(0);
		morphologyEx(th_img_gray_clahe, frontSrcTh21, CV_MOP_CLOSE, element21);   //闭运算形态学操作。可以减少噪点
		//dilate(th_img_gray_clahe, frontSrcTh21, element21);//膨胀
		imgTopTh.copyTo(frontSrcTh21(Rect(0, 0, frontSrcTh21.cols - 1, 200)));
		imgButtomTh.copyTo(frontSrcTh21(Rect(0, 1300, frontSrcTh21.cols - 1, 200)));
		imgLeftTh.copyTo(frontSrcTh21(Rect(0, 0, 200, frontSrcTh21.rows)));
		imgRightTh.copyTo(frontSrcTh21(Rect(2800, 0, 200, frontSrcTh21.rows)));


		imwrite(rootPath1 + "//result4//" + namenum.c_str() + ".bmp", frontSrcTh21);


		//---------------筛选轮廓-------------//
		vector<vector<Point>>contours;
		findContours(frontSrcTh21, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
		sort(contours.begin(), contours.end(), compareContourAreas);
		vector<Rect> boundRect(contours.size());
		double bian = 0.0;
		double bian1 = 0.0;

		for (vector<int> ::size_type i = 0; i < contours.size(); i++) {

			Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
			drawContours(tempMask, contours, i, 255, 1, 8);
			double area = contourArea(contours[i]);
			if (area < 2000) {
				break;
			}

			Point2f RectPoint[4];
			RotatedRect  externrect = minAreaRect(Mat(contours[i]));
			double mw2 = externrect.size.height;
			double mh2 = externrect.size.width;
			Point2f center = externrect.center;  //中心
			//特征一： 长宽比
			double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
			//特征二： 最大长度
			double length = max(mw2, mh2);
			boundRect[i] = boundingRect(Mat(contours[i]));
			float w = boundRect[i].width;
			float h = boundRect[i].height;
			int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
			int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
			int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
			int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
			double longShortRatio = max(h / w, w / h);
			if (radio2 > 1.5 && length > 200)//长宽比，长度，宽度限制
			{
				int border_x = 10;//选定框边界宽度
				int border_y = 10;//选定框边界宽度
				if (area > 10000) {

					if (w > h) {
						border_x = 5;
						border_y = 10;
					}
					else {
						border_x = 10;
						border_y = 5;
					}
				}
				else {

					if (w > h) {
						border_x = 3;
						border_y = 5;
					}
					else {
						border_x = 5;
						border_y = 3;
					}
				}

				int x_lt = X_1 - border_x;
				//越界保护
				if (x_lt < 0)
				{
					x_lt = 0;
				}
				int y_lt = Y_1 - border_y;
				if (y_lt < 0)
				{
					y_lt = 0;
				}
				int x_rt = X_2 + border_x;
				if (x_rt > frontSrcTh21.size[1] - 1)
				{
					x_rt = frontSrcTh21.size[1] - 1;
				}
				int y_rt = Y_2 + border_y;
				if (y_rt > frontSrcTh21.size[0] - 1)
				{
					y_rt = frontSrcTh21.size[0] - 1;
				}
				Mat sidelightSuspect = frontSrc(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//侧光图像疑似贴膜划痕图像
				Mat whitelightSuspect = front(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
				Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜
				Mat mask1;
				Mat mask2;
				/*mask2 = binaryImage(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1)).clone();
				mask1 = binaryImage(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1)).clone();
				bitwise_and(~mask, mask1, mask1);*/
				double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
				double meanGrayout_Suspect = mean(sidelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
				double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                        //排除侧光图贴膜划痕的参数
				double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
				double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
				double meanGrayout_Suspect1 = mean(whitelightSuspect, ~mask)[0];                          //缺陷外围灰度均值
				double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
				cv::Mat meanGray;
				cv::Mat stdDev;
				cv::meanStdDev(whitelightSuspect, meanGray, stdDev);
				bian = stdDev.at<double>(0, 0) / meanGray.at<double>(0, 0);
				cv::Mat meanGray1;
				cv::Mat stdDev1;
				cv::meanStdDev(sidelightSuspect, meanGray1, stdDev1);
				bian1 = stdDev1.at<double>(0, 0) / meanGray1.at<double>(0, 0);
				double removeScratchFlag;
				double removeStvflag;
				/*	if (removeScratchArea > 1000) {
						if (radio2 > 1.5) {
							removeScratchFlag = 1.25;
							removeStvflag = 2.2;
						}
						else {
							removeScratchFlag = 2.0;
							removeStvflag = 2.5;
						}

					}
					else {
						removeScratchFlag = 1.8;
						removeStvflag = 2.2;
					}*/
				string c, c1;
				num2string(center.x, c);
				num2string(center.y, c1);
				string c2 = "(";
				c2 = c2 + c + "和" + c1 + ")";
				fprintf(pOutFile, "%s,%d,  %f,  %f, %f, %f,%f, %f,%f,%s\n", namenum.c_str(), i + 1, area, length, radio2, removeScratch, removeScratch1, stdDev1.at<double>(0, 0), stdDev.at<double>(0, 0), c2.c_str());
				if (area > 4500) {
					if (stdDev.at<double>(0, 0) > 1 && stdDev.at<double>(0, 0) < 10)
					{


						result = true;
						CvPoint top_lef4 = cvPoint(x_lt, y_lt);
						CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
						rectangle(front, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
						*mresult = front;
						break;
					}

				}


			}



		}
	}

	if (result == true)
	{
		*causecolor = "折痕";


		imwrite(rootPath1 + "//result4//" + namenum.c_str() + "1.bmp", *mresult);
	}
	else {
		*causecolor = "良品";
		*mresult = frontSideLight;
	}
	fprintf(pOutFile1, "%s,%s,%f,%f\n", namenum.c_str(), (*causecolor).c_str());
	return result;
}
bool compareContourSize(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2) {

	return contour1.size() > contour2.size();
}
bool compareContourRadio(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2) {

	//当两个轮廓有任一个小于30时 比较轮廓像素个数，否则使用长宽比比较
	if (contour1.size() < 30 || contour2.size() < 30)
	{
		return compareContourSize(contour1, contour2);
	}
	RotatedRect box;
	box = minAreaRect(Mat(contour1));
	double mw = box.size.height;
	double mh = box.size.width;
	double radio1 = (mw == 0 || mh == 0) ? 0 : max(mw / mh, mh / mw);

	box = minAreaRect(Mat(contour2));
	mw = box.size.height;
	mh = box.size.width;
	double radio2 = (mw == 0 || mh == 0) ? 0 : max(mw / mh, mh / mw);

	return radio1 > radio2;
}

void myEnhanceHist(cv::Mat& img_gray, cv::Mat& img_grayHist)
{
	// 获取线性变换参数
	double alpha1 = 0.5;
	double beta1 = 0;

	// 增强 srcMean - 10 与 230
	double srcMean = mean(img_gray)[0];
	double r1 = srcMean - 5;
	double r2 = srcMean + 80; //145 224
	double s1 = alpha1 * r1;
	double s2 = 255;
	double alpha2 = (255 - alpha1 * r1) / (r2 - r1);
	double beta2 = -alpha2 * r1 + alpha1 * r1;

	for (int r = 0; r < img_gray.rows; r++)
	{
		for (int c = 0; c < img_gray.cols; c++) {
			uchar temp = img_gray.at<uchar>(r, c);
			if (temp <= r1)
			{
				img_grayHist.at<uchar>(r, c) = saturate_cast<uchar>(temp * alpha1);	//alpha = 0.5, beta = 0
			}
			else if (r1 < temp && temp < r2)
			{
				img_grayHist.at<uchar>(r, c) = saturate_cast<uchar>(temp * alpha2 + beta2);	//alpha = 3.6, beta = -310
			}
			else
			{
				img_grayHist.at<uchar>(r, c) = saturate_cast<uchar>(255);	//alpha = 0.238, beta = 194
			}
		}
	}
}

bool Scratch(Mat white, Mat ceguang, Mat* mresult, string* causecolor, int camera)
{
	bool result = false;
	double val1 = mean(white)[0];
	double val2 = mean(ceguang)[0];

	int length = 100;
	int left_right = 100;//左右屏蔽宽度
	int up_down = 122;//或者122（10号漏检）
	Mat img_gray = white.clone();
	Mat img_ceguang = ceguang.clone();
	Mat Filer = img_gray.clone();


	//去除黑边影响
	Mat img_gray_temp;
	img_gray_temp = img_gray(Rect(20, 20, img_gray.size().width - 45, img_gray.size().height - 45));
	resize(img_gray_temp, img_gray_temp, img_gray.size());
	img_gray = img_gray_temp.clone();
	white = img_gray.clone();




	//侧光相机去除黑边影响
	Mat ceguang_temp;
	ceguang_temp = ceguang(Rect(20, 20, img_gray.size().width - 45, img_gray.size().height - 45));
	resize(ceguang_temp, ceguang_temp, ceguang.size());
	ceguang = ceguang_temp.clone();

	//中值滤波
	medianBlur(img_gray_temp, Filer, 11);//中值滤波滤除椒盐噪声,缺点耗时26毫秒 奇数半径越大效果越强

	img_gray = Filer.clone();


	//线性变换增强
	Size size = img_gray.size();
	Mat img_grayHist(size, img_gray.type());
	myEnhanceHist(img_gray, img_grayHist);

	//    //将原图拆分为3部分
	//    Mat img_grayPart1;
	//    Mat img_grayPart2;
	//    Mat img_grayPart3;
	//    img_grayPart1 = img_gray(Rect(0, 0, 600, 1500)).clone();
	//    img_grayPart2 = img_gray(Rect(600, 0, 1800, 1500)).clone();
	//    img_grayPart3 = img_gray(Rect(2400, 0, 600, 1500)).clone();

	//    //对3部分分别进行线性增强
	//    Mat img_grayHistPart1(img_grayPart1.size(), img_grayPart1.type());
	//    Mat img_grayHistPart2(img_grayPart2.size(), img_grayPart2.type());
	//    Mat img_grayHistPart3(img_grayPart3.size(), img_grayPart3.type());

	//    myEnhanceHist(img_grayPart1, img_grayHistPart1);
	//    myEnhanceHist(img_grayPart2, img_grayHistPart2);
	//    myEnhanceHist(img_grayPart3, img_grayHistPart3);


	//    //对3部分进行分别进行自适应二值化
	Mat th2, th4, th5, th6, th7;
	//    adaptiveThreshold(img_grayHistPart1, th4, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 107, -3);
	//    adaptiveThreshold(img_grayHistPart2, th5, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 107, -3);
	//    adaptiveThreshold(img_grayHistPart3, th7, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 107, -3);

	//    //将3部分的自适应二值化图组合到一起
	//    Mat thResult(img_gray.rows, img_gray.cols, img_gray.type(), Scalar(0));
	//    Mat thResultPart1 = thResult(Rect(0, 0, img_grayHistPart1.cols, img_grayHistPart2.rows));
	//    Mat thResultPart2 = thResult(Rect(img_grayHistPart1.cols, 0, img_grayHistPart2.cols, img_grayHistPart2.rows));
	//    Mat thResultPart3 = thResult(Rect(img_grayHistPart1.cols + img_grayHistPart2.cols, 0,
	//                                      img_grayHistPart1.cols, img_grayHistPart2.rows));
	//    th4.copyTo(thResultPart1);
	//    th5.copyTo(thResultPart2);
	//    th7.copyTo(thResultPart3);

		//th6 = thResult.clone();


	//    Mat img_grayPart;
	//    Mat img_grayHistPart(img_grayPart.size(), img_grayPart.type());
	//    myEnhanceHist(img_grayPart, img_grayPart);
	   // cv::imwrite("E:\\Results\\img_grayPart.bmp",img_grayPart);
	adaptiveThreshold(white(Rect(0, 0, 1500, img_gray.rows)), th7, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 51, -1.5);
	//adaptiveThreshold(white(Rect(1000, 0, 1000, img_gray.rows)), th6, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -0.5);
	//adaptiveThresholdCustom(white(Rect(1000, 0, 1000, img_gray.rows)), th6, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 51, -3, 1);
	adaptiveThresholdCustom(white, th6, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 51, -1, 1, 0.5);//21 5  LQY:17 3
	th6 = th7.clone();

	//腐蚀结果图像
	//Mat adThresholdDilate, adThresholdErode;
	////使用腐蚀将噪点滤除
	////cv::Mat elementdilate = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(10, 10));
	//cv::Mat elementErode = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(3, 3));
	////dilate(Filer, adThresholdDilate, elementdilate);
	//erode(th6, adThresholdErode, elementErode);


	medianBlur(th6, Filer, 7);//中值滤波滤除椒盐噪声,缺点耗时26毫秒 奇数半径越大效果越强


	//chao_thinimage(Filer);

	th2 = Filer.clone();

	int shuidi_length = 160;
	int part = img_gray.rows / 2 - shuidi_length;

	//屏蔽th1左、右的一小列，为了将边缘划伤提取出来
	//th1(Rect(th1.cols - 50, 0, 3, th1.rows - 1)) = uchar(0);//屏蔽右边150
	//th1(Rect(th1.cols - 52, 0, 5, th1.rows - 1)) = uchar(0);
	//屏蔽th2上下110像素，左右150像素
	//    th2(Rect(0, 0, left_right, th2.rows - 1)) = uchar(0);//屏蔽左边150
	//    th2(Rect(th2.cols - left_right, 0, left_right, th2.rows - 1)) = uchar(0);//屏蔽右边150
	//    th2(Rect(0, 0, th2.cols - 1, up_down)) = uchar(0);//上边屏蔽110
	//    th2(Rect(0, th2.rows - up_down, th2.cols - 1, up_down)) = uchar(0);//下边屏蔽110




	vector<vector<Point>> contours2;
	findContours(th2, contours2, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
	std::sort(contours2.begin(), contours2.end(), compareContourRadio);
	vector<Rect> boundRect2(contours2.size());
	cout << contours2.size() << endl;
	//for (vector<int>::size_type i = 0; contourArea(contours2[i]) >= 750 && contourArea(contours2[i]) < 45000; i++)//800改成750
	//{
	for (vector<int>::size_type i = 0; i < 20 && i < contours2.size(); i++)//800改成750
	{
		double area = contourArea(contours2[i]);
		Mat currentContours2 = Mat::zeros(th2.size(), CV_8UC1);
		drawContours(currentContours2, contours2, i, 255, FILLED, 8);

		double area2 = contourArea(contours2[i]);
		vector<RotatedRect>box2(contours2.size());

		Point2f RectPoint[4];
		box2[i] = minAreaRect(Mat(contours2[i]));

		//绘制矩形框，观察缺陷位置
		Point2f vertices[4];      //定义4个点的数组
		box2[i].points(vertices);   //将四个点存储到vertices数组中
		for (int j = 0; j < 4; j++)
			// 注意Scala中存储顺序 BGR
			line(currentContours2, vertices[j], vertices[(j + 1) % 4], Scalar(255));

		double mw2 = box2[i].size.height;
		double mh2 = box2[i].size.width;
		//特征一： 长宽比
		double radio2 = max(mw2 / mh2, mh2 / mw2); // 80
		//特征二： 最大长度
		double length = max(mw2, mh2);
		box2[i].points(RectPoint);
		std::cout << "长宽比:  " << radio2;
		std::cout << "最大长度:  " << length;

		boundRect2[i] = boundingRect(Mat(contours2[i]));
		int X_1 = boundRect2[i].tl().x;//矩形左上角X坐标值
		int Y_1 = boundRect2[i].tl().y;//矩形左上角Y坐标值
		int X_2 = boundRect2[i].br().x;//矩形右下角X坐标值
		int Y_2 = boundRect2[i].br().y;//矩形右下角Y坐标值

		//长宽比识别
		if ((radio2 > 1 && radio2 < 20 && length > 20
			&& contours2[i].size() > 30 && (X_2 <= 2850 && X_1 >= 200))
			|| radio2 > 1 && radio2 < 30 && length > 120 && contours2[i].size() > 50 && (X_2 > 2850 || X_1 < 200))//16.5改成16.6
		{

			//特征3 内外灰度差
			Mat temp2, tempimg_gray;
			temp2 = th2(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
			tempimg_gray = img_gray(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
			double mean_in_gray = mean(tempimg_gray, temp2)[0];
			double mean_out_gray = mean(tempimg_gray, ~temp2)[0];
			double differ = mean_in_gray - mean_out_gray;//划伤 为2.73

			//特征4 侧光内外灰度差
			Mat ceguang_gray;
			ceguang_gray = ceguang(Rect(X_1, Y_1, X_2 - X_1, Y_2 - Y_1));
			double mean_in_ceguang = mean(ceguang_gray, temp2)[0];
			double mean_out_ceguang = mean(ceguang_gray, ~temp2)[0];
			double ceguang_differ = mean_in_ceguang - mean_out_ceguang;

			if (ceguang_differ >= 100)
			{
				continue;
			}
			std::cout << "侧光内外灰度差:  " << differ << endl;
			std::cout << "内外灰度差:  " << differ << endl;
			if (differ > 2)
			{


				result = true;
				Point2f vertices[4];      //定义4个点的数组
				box2[i].points(vertices);   //将四个点存储到vertices数组中
				for (int j = 0; j < 4; j++)
					// 注意Scala中存储顺序 BGR
					line(white, vertices[j], vertices[(j + 1) % 4], Scalar(0), 3);
				if (result == true)
				{
					*causecolor = "划伤";
					*mresult = white;
				}
				return result;
			}
		}
	}

	return result;
}


bool Creases(Mat frontSideLight, Mat front, Mat* mresult, string* causecolor) {
	bool result = false;
	Mat frontSrcTh;  //前相机阈值分割
	Mat frontSrcTh1;  //前相机阈值分割
	Mat white_L = frontSideLight.clone();
	Mat white_R = front.clone();
	medianBlur(white_L, white_L, 5);										//中值滤波去除锯齿
	Mat frontSrc = white_L;
	Mat binaryImage;

	double meanGray = mean(white_L)[0];
	threshold(white_L, binaryImage, meanGray * 0.65, 255, CV_THRESH_BINARY);							//二值化(有问题)
	meanGray = mean(white_L, binaryImage)[0];
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	Mat th1;

	//Mat yk = Process(white_L, 0, 0.4, Size(51, 51));
	//lacklineEenhance1 = 255 - lacklineEenhance1;

	Mat imgLeft = white_L(Rect(0, 0, 200, frontSrc.rows)).clone();
	Mat imgLeft1 = white_L(Rect(0, 0, 200, frontSrc.rows / 2));
	Mat imgLeft2 = white_L(Rect(0, frontSrc.rows / 2, 200, frontSrc.rows / 2));
	Mat imgRight1 = white_L(Rect(2799, 0, 200, frontSrc.rows)).clone();
	Mat imgMiddle = white_L(Rect(200, 0, frontSrc.cols - 400, frontSrc.rows));
	Mat imageBinary;
	Mat imgLeftTh;
	Mat imgLeftTh1;
	Mat imgRightTh;
	Mat c2;
	Mat lacklineEenhance1 = gamma(imgLeft, 0.2);

	Mat canny_output;
	int x = imgLeft1.cols / 2;
	int y = imgLeft1.rows / 2;
	Point pt(x, y); //待生长点位置
	int th = 2;

	equalizeHist(lacklineEenhance1, c2);
	Canny(c2, canny_output, 20, 20 * 3, 3);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;

	//求x方向梯度
	Sobel(imgLeft, grad_x, CV_16S, 1, 0, 3, 3, 3, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//imshow("x方向soble", abs_grad_x);

	//求y方向梯度
	Sobel(imgLeft, grad_y, CV_16S, 0, 1, 3, 3, 3, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//imshow("y向soble", abs_grad_y);

	//合并梯度
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	//imshow("Sobel算法轮廓提取效果", dst);
	Ptr<CLAHE> clahe = createCLAHE(2, Size(21, 21));
	clahe->apply(imgLeft, imgLeft);
	double meanGray1 = mean(imgLeft)[0];
	threshold(imgLeft, binaryImage, meanGray1, 255, CV_THRESH_BINARY);							//二值化(有问题)
	adaptiveThresholdCustom(imgLeft, imgLeftTh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -1, 1, 0.5);//21 5  LQY:17 3
	adaptiveThresholdCustom(imgLeft, imgLeftTh1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3, 1, 1);//21 5  LQY:17 3
	//bitwise_and(imgLeftTh, ~imgLeftTh1, imgLeftTh);
	adaptiveThresholdCustom(imgRight1, imgRightTh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -2, 1, 1);//21 5  LQY:17 3
	adaptiveThresholdCustom(imgMiddle, th1, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 17, -3, 1, 1);//21 5  LQY:17 3
	Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));//腐蚀连通区域
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));

	morphologyEx(imgLeftTh, imgLeftTh, MORPH_OPEN, element, Point(-1, -1), 1);

	Mat frontSrcTh2;
	Mat element1 = getStructuringElement(MORPH_RECT, Size(5, 5));//闭操作结构元素
	dilate(imgLeftTh, imgLeftTh, element1);//膨胀
	//morphologyEx(imageBinary, imageBinary, MORPH_DILATE, kernel);
	//边界进行深拷贝
	frontSrcTh = Mat::zeros(white_L.size(), CV_8UC1);
	imgLeftTh.copyTo(frontSrcTh(Rect(0, 0, 200, frontSrcTh.rows)));
	imgRightTh.copyTo(frontSrcTh(Rect(2800, 0, 200, frontSrc.rows)));
	th1.copyTo(frontSrcTh(Rect(200, 0, frontSrc.cols - 400, frontSrc.rows)));


	morphologyEx(frontSrcTh, frontSrcTh, MORPH_OPEN, element1, Point(-1, -1), 1);

	vector<vector<Point>>contours;
	findContours(frontSrcTh, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);
	vector<Rect> boundRect(contours.size());
	double bian = 0.0;
	double bian1 = 0.0;
	for (vector<int> ::size_type i = 0; i < contours.size(); i++) {

		Mat tempMask = Mat::zeros(Size(3000, 1500), CV_8UC1);
		drawContours(tempMask, contours, i, 255, FILLED, 8);
		double area = contourArea(contours[i]);

		if (area < 500) {
			continue;
		}

		RotatedRect rect = minAreaRect(contours[i]);
		boundRect[i] = boundingRect(Mat(contours[i]));

		int x1 = boundRect[i].tl().x;
		int y1 = boundRect[i].tl().y;
		int x2 = boundRect[i].br().x;
		int y2 = boundRect[i].br().y;
		float w = boundRect[i].width;
		float h = boundRect[i].height;
		int X_1 = boundRect[i].tl().x;//矩形左上角X坐标值
		int Y_1 = boundRect[i].tl().y;//矩形左上角Y坐标值
		int X_2 = boundRect[i].br().x;//矩形右下角X坐标值
		int Y_2 = boundRect[i].br().y;//矩形右下角Y坐标值
		double longShortRatio = max(h / w, w / h);
		if (w > 1000 || h > 200) {
			continue;
		}

		if (longShortRatio > 1.2)//长宽比，长度，宽度限制
		{
			int border = 20;//选定框边界宽度
			if (area > 15000) {
				border = 20;
			}
			else {

				border = 5;
			}

			int x_lt = X_1 - border;
			//越界保护
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
			if (x_rt > white_L.size[1] - 1)
			{
				x_rt = white_L.size[1] - 1;
			}
			int y_rt = Y_2 + border;
			if (y_rt > white_L.size[0] - 1)
			{
				y_rt = white_L.size[0] - 1;
			}
			Mat sidelightSuspect = frontSideLight(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//侧光图像疑似贴膜划痕图像
			//Mat whitelightSuspect = frontSideLight(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));//白底图像疑似反光划痕图像
			Mat mask = tempMask(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1));             //侧光图像疑似贴膜划痕掩膜
			Mat mask1;
			Mat mask2;

			{
				mask2 = binaryImage(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1)).clone();
				mask1 = binaryImage(Rect(x_lt, y_lt, x_rt - x_lt - 1, y_rt - y_lt - 1)).clone();

				bitwise_and(~mask, mask1, mask1);
				//mask1 = cy;
			}
			/*else {
				mask1 = Mat::zeros(mask.size(), CV_8UC1);
				mask1 = ~mask;
			}*/
			double meanGrayin_Suspect = mean(sidelightSuspect, mask)[0];                            //缺陷中心灰度均值
			//mask1 = mask1 + ~mask;
			double meanGrayout_Suspect = mean(sidelightSuspect, mask1)[0];                          //缺陷外围灰度均值
			double removeScratch = meanGrayin_Suspect - meanGrayout_Suspect;                        //排除侧光图贴膜划痕的参数
			double removeScratchArea = (x_rt - x_lt - 2) * (y_rt - y_lt - 2);
			//double meanGrayin_Suspect1 = mean(whitelightSuspect, mask)[0];                            //缺陷中心灰度均值
		//	double meanGrayout_Suspect1 = mean(whitelightSuspect, mask1)[0];                          //缺陷外围灰度均值
		//	double removeScratch1 = meanGrayin_Suspect1 - meanGrayout_Suspect1;                        //排除侧光图贴膜划痕的参数
		/*	cv::Mat meanGray;
			cv::Mat stdDev;
			cv::meanStdDev(whitelightSuspect, meanGray, stdDev, mask2);
			bian = stdDev.at<double>(0, 0) / meanGray.at<double>(0, 0);*/

			cv::Mat meanGray1;
			cv::Mat stdDev1;
			cv::meanStdDev(sidelightSuspect, meanGray1, stdDev1, mask2);
			bian1 = stdDev1.at<double>(0, 0) / meanGray1.at<double>(0, 0);
			if (removeScratch > 1) {
				if (stdDev1.at<double>(0, 0) > 2.4)
				{
					fprintf(pOutFile, "%s,%d, %f, %f,%f\n", namenum.c_str(), i + 1, removeScratchArea);
					result = true;
					CvPoint top_lef4 = cvPoint(x_lt, y_lt);
					CvPoint bottom_right4 = cvPoint(x_rt, y_rt);
					rectangle(frontSideLight, top_lef4, bottom_right4, Scalar(255, 255, 255), 5, 8, 0);
					break;
				}

			}


		}



	}
	if (result == false) {


	}

	if (result == true)
	{
		*causecolor = "折痕";
		*mresult = frontSideLight;

		imwrite(rootPath1 + "//result//" + namenum.c_str() + ".bmp", *mresult);
	}
	else {
		*causecolor = "良品";
		*mresult = frontSideLight;
	}
	fprintf(pOutFile1, "%s,%s,%f,%f\n", namenum.c_str(), (*causecolor).c_str(), meanGray, bian1);
	return result;
}

bool Crease1(Mat frontSideLight, Mat leftSideLight, Mat rightSideLight, string* causecolor) {

	bool result = false;
	Mat frontSrc = frontSideLight.clone();
	//medianBlur(frontSrc, frontSrc, 25);
	GaussianBlur(frontSrc, frontSrc, Size(25, 25), 3);

	Mat frontSrcTh;  //前相机阈值分割
	Mat frontSrcTh1;  //前相机阈值分割

	//Mat frontSrc1;
	//copyMakeBorder(frontSrc, frontSrc, 40, 40, 40, 40, BORDER_REPLICATE);

	//截取边框区域
	Mat imgTopTh;
	Mat imgButtomTh;
	Mat imgLeftTh;
	Mat imgRightTh;
	Mat imgTop = frontSrc(Rect(0, 0, frontSrc.cols - 1, 20));
	Mat imgButtom = frontSrc(Rect(0, 1480, frontSrc.cols - 1, 20));
	Mat imgLeft = frontSrc(Rect(0, 0, 20, frontSrc.rows));
	Mat imgRight = frontSrc(Rect(2980, 0, 20, frontSrc.rows));
	adaptiveThreshold(imgTop, imgTopTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -3);
	adaptiveThreshold(imgButtom, imgButtomTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -3);
	adaptiveThreshold(imgLeft, imgLeftTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -3);
	adaptiveThreshold(imgRight, imgRightTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -3);


	//屏幕中间的部分
	adaptiveThreshold(frontSrc, frontSrcTh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 19, -3);//21
	//adaptiveThresholdCustom(frontSrc, frontSrcTh1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 255, -3, 1);

	//屏蔽边界，去除边界的影响
	//frontSrcTh(Rect(0, 0, frontSrc.cols, 10)) = uchar(0);
	//frontSrcTh(Rect(0, 0, 20, frontSrcTh.rows)) = uchar(0);
	//frontSrcTh(Rect(2990, 0, 10, frontSrcTh.rows)) = uchar(0);
	//frontSrcTh(Rect(0, 1490, frontSrcTh.cols, 10)) = uchar(0);

	////边界进行深拷贝
	imgTopTh.copyTo(frontSrcTh(Rect(0, 0, frontSrcTh.cols - 1, 20)));
	imgButtomTh.copyTo(frontSrcTh(Rect(0, 1480, frontSrc.cols - 1, 20)));
	imgLeftTh.copyTo(frontSrcTh(Rect(0, 0, 20, frontSrcTh.rows)));
	imgRightTh.copyTo(frontSrcTh(Rect(2980, 0, 20, frontSrc.rows)));

	Mat closeElement = getStructuringElement(MORPH_RECT, Size(7, 7));
	//Mat openElement = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(frontSrcTh, frontSrcTh1, MORPH_CLOSE, closeElement, Point(-1, -1), 1);
	//morphologyEx(frontSrcTh1, frontSrcTh1, MORPH_OPEN, openElement, Point(-1, -1), 1);

	//---------------筛选轮廓-------------//
	vector<vector<Point>>contours;
	findContours(frontSrcTh1, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	sort(contours.begin(), contours.end(), compareContourAreas);
	vector<Rect> boundRect(contours.size());

	for (vector<int> ::size_type i = 0; i < contours.size(); i++) {

		/****************轮廓面积计算****************/
		Mat tempMask = Mat::zeros(frontSrcTh.size(), CV_8UC1);
		drawContours(tempMask, contours, i, 255, FILLED, 8);
		double area = contourArea(contours[i]);
		//cout << "面积：" << area << endl;
		//对轮廓面积较小的灰尘等干扰进行滤除
		if (area < 160) {
			continue;
		}

		/****************轮廓长宽比计算****************/
		RotatedRect rotRect = minAreaRect(contours[i]);
		double minWidth = rotRect.size.width;
		double minHeight = rotRect.size.height;
		double ratio = max(minWidth / minHeight, minHeight / minWidth);
		if (ratio < 2.0 || ratio > 8) {
			continue;
		}
		//cout << "ratio" << ratio << endl;
		//Point2f pts[4];                                     //通过四个顶点画矩形
		//rotRect.points(pts);
		//for (int i = 0; i < 4; i++)
		//{
		//	line(tempMask, pts[i], pts[(i + 1) % 4], Scalar(255), 1, 8);
		//}


		/****************轮廓内外灰度差计算****************/
		Mat rectMask = tempMask.clone();
		boundRect[i] = boundingRect(Mat(contours[i]));
		int x1 = boundRect[i].tl().x;
		int y1 = boundRect[i].tl().y;
		int x2 = boundRect[i].br().x;
		int y2 = boundRect[i].br().y;

		//防止越界处理
		if ((x1 - 10) < 0) {
			x1 = 0;
		}
		if ((y1 - 10) < 0) {
			y1 = 0;
		}
		if ((x2 + 10) > 3000) {
			x2 = 3000;
		}
		if ((y2 + 10) > 1500) {
			y2 = 1500;
		}


		Rect rect(x1 - 10, y1 - 10, x2 - x1 + 20, y2 - y1 + 20);
		rectangle(rectMask, rect, Scalar(255), 1, 8);
		rectangle(rectMask, rect, Scalar(255), -1, 8);
		Mat outSide = rectMask - tempMask;
		double meanOut = mean(frontSrc, outSide)[0];
		double meanIn = mean(frontSrc, tempMask)[0];
		double gap = meanIn - meanOut;

		if (gap > 16) {
			result = true;
			break;

		}

		//cout << area << " " << ratio << " " << gap << endl;






		//cout << "轮廓：" << i + 1 << endl;


	}
	if (result == true) {
		*causecolor = "折痕";
	}
	else {
		*causecolor = "良品";
	}
	return result;

}


/*=========================================================
* 函 数 名: Vec2Mat
* 功能描述: 将Vector二维数组转换为Mat数据格式
* 输入变量: 1.二维数组Vector 2.图片像素行数 3.图片像素列数
=========================================================*/
Mat Vec2Mat7(vector<vector<uchar>> array, int row, int col)
{
	Mat img(row, col, CV_8UC1);
	uchar* ptmp = NULL;
	for (int i = 0; i < row; i++)
	{
		vector<uchar> z_i = array[i];
		ptmp = img.ptr<uchar>(i);
		for (int j = 0; j < col; j++)
			ptmp[j] = z_i[j];
	}
	return img;
}

/*========================================================================
* 函 数 名: Read_Mat
* 功能描述: 将单通道图像像素读到Vector中
* 输入变量: 1.二维数组Vector 2.输出的图片 3.图片像素行数 4.图片的列数
==========================================================================*/
vector<vector<int>> Read_Mat7(vector<vector<int>> array, Mat d_Img, int nRows, int nCols)
{
	for (int i = 0; i < nRows; i++)
	{
		uchar* p = d_Img.ptr<uchar>(i);
		for (int j = 0; j < nCols; j++)
			array[i][j] = p[j];
	}
	return array;
}
class Fit {
	std::vector<double> factor; ///<   Ϻ  ķ   ϵ    
	double ssr;                 ///< ع ƽ      
	double sse;                 ///<(ʣ  ƽ    )  
	double rmse;                ///<RMSE            
	std::vector<double> fitedYs;///<       Ϻ   yֵ        ʱ      Ϊ        ʡ ڴ   
public:
	Fit() :ssr(0), sse(0), rmse(0) { factor.resize(2, 0); }
	~Fit() {}
	///  
	/// \brief ֱ      -һԪ ع ,   ϵĽ       ʹ  getFactor  ȡ      ʹ  getSlope  ȡб ʣ getIntercept  ȡ ؾ   
	/// \param x  ۲ ֵ  x  
	/// \param y  ۲ ֵ  y  
	/// \param isSaveFitYs    Ϻ        Ƿ񱣴棬Ĭ Ϸ   
	///  
	template<typename T>
	bool linearFit(const std::vector< T>& x, const std::vector< T>& y, bool isSaveFitYs = false)
	{
		return linearFit(&x[0], &y[0], getSeriesLength(x, y), isSaveFitYs);
	}
	template<typename T>
	bool linearFit(const T* x, const T* y, size_t length, bool isSaveFitYs = false)
	{
		factor.resize(2, 0);
		T t1 = 0, t2 = 0, t3 = 0, t4 = 0;
		for (int i = 0; i < length; ++i)
		{
			t1 += x[i] * x[i];
			t2 += x[i];
			t3 += x[i] * y[i];
			t4 += y[i];
		}
		factor[1] = (t3 * length - t2 * t4) / (t1 * length - t2 * t2);
		factor[0] = (t1 * t4 - t2 * t3) / (t1 * length - t2 * t2);
		//////////////////////////////////////////////////////////////////////////  
		//          
		calcError(x, y, length, this->ssr, this->sse, this->rmse, isSaveFitYs);
		return true;
	}
	///  
	/// \brief     ʽ   ϣ     y=a0+a1*x+a2*x^2+    +apoly_n*x^poly_n  
	/// \param x  ۲ ֵ  x  
	/// \param y  ۲ ֵ  y  
	/// \param poly_n        ϵĽ       poly_n=2    y=a0+a1*x+a2*x^2  
	/// \param isSaveFitYs    Ϻ        Ƿ񱣴棬Ĭ      
	///   
	template<typename T>
	void polyfit(const std::vector<T>& x
		, const std::vector<T>& y
		, int poly_n
		, bool isSaveFitYs = true)
	{
		polyfit(&x[0], &y[0], getSeriesLength(x, y), poly_n, isSaveFitYs);
	}
	template<typename T>
	void polyfit(const T* x, const T* y, size_t length, int poly_n, bool isSaveFitYs = true)
	{
		factor.resize(poly_n + 1, 0);
		int i, j;
		//double *tempx,*tempy,*sumxx,*sumxy,*ata;  
		std::vector<double> tempx(length, 1.0);

		std::vector<double> tempy(y, y + length);

		std::vector<double> sumxx(poly_n * 2 + 1);
		std::vector<double> ata((poly_n + 1) * (poly_n + 1));
		std::vector<double> sumxy(poly_n + 1);
		for (i = 0; i < 2 * poly_n + 1; i++) {
			for (sumxx[i] = 0, j = 0; j < length; j++)
			{
				sumxx[i] += tempx[j];
				tempx[j] *= x[j];
			}
		}
		for (i = 0; i < poly_n + 1; i++) {
			for (sumxy[i] = 0, j = 0; j < length; j++)
			{
				sumxy[i] += tempy[j];
				tempy[j] *= x[j];
			}
		}
		for (i = 0; i < poly_n + 1; i++)
			for (j = 0; j < poly_n + 1; j++)
				ata[i * (poly_n + 1) + j] = sumxx[i + j];
		gauss_solve(poly_n + 1, ata, factor, sumxy);
		//       Ϻ      ݲ           
		//fitedYs.reserve(length);
		//calcError(&x[0], &y[0], length, this->ssr, this->sse, this->rmse, isSaveFitYs);  //12.11

	}
	///   
	/// \brief   ȡϵ    
	/// \param     ϵ          
	///  
	void getFactor(std::vector<double>& factor) { factor = this->factor; }
	///   
	/// \brief   ȡ   Ϸ  ̶ Ӧ  yֵ  ǰ        ʱ    isSaveFitYsΪtrue  
	///  
	void getFitedYs(std::vector<double>& fitedYs) { fitedYs = this->fitedYs; }

	///   
	/// \brief     x  ȡ   Ϸ  ̵ yֵ  
	/// \return     x  Ӧ  yֵ  
	///  
	template<typename T>
	double getY(const T x) const
	{
		double ans(0);
		for (size_t i = 0; i < factor.size(); ++i)
		{
			ans += factor[i] * pow((double)x, (int)i);
		}
		return ans;
	}
	///   
	/// \brief   ȡб    
	/// \return б  ֵ  
	///  
	double getSlope() { return factor[1]; }
	///   
	/// \brief   ȡ ؾ   
	/// \return  ؾ ֵ  
	///  
	double getIntercept() { return factor[0]; }
	///   
	/// \brief ʣ  ƽ      
	/// \return ʣ  ƽ      
	///  
	double getSSE() { return sse; }
	///   
	/// \brief  ع ƽ      
	/// \return  ع ƽ      
	///  
	double getSSR() { return ssr; }
	///   
	/// \brief             
	/// \return             
	///  
	double getRMSE() { return rmse; }
	///   
	/// \brief ȷ  ϵ    ϵ    0~1֮                 ж      Ŷȵ һ      
	/// \return ȷ  ϵ    
	///  
	double getR_square() { return 1 - (sse / (ssr + sse)); }
	///   
	/// \brief   ȡ    vector İ ȫsize  
	/// \return   С  һ        
	///  
	template<typename T>
	size_t getSeriesLength(const std::vector< T>& x
		, const std::vector<T>& y)
	{
		return (x.size() > y.size() ? y.size() : x.size());
	}
	///   
	/// \brief       ֵ  
	/// \return   ֵ  
	///  
	template <typename T>
	static T Mean(const std::vector<T>& v)
	{
		return Mean(&v[0], v.size());
	}
	template <typename T>
	static T Mean(const T* v, size_t length)
	{
		T total(0);
		for (size_t i = 0; i < length; ++i)
		{
			total += v[i];
		}
		return (total / length);
	}
	///   
	/// \brief   ȡ   Ϸ   ϵ   ĸ     
	/// \return    Ϸ   ϵ   ĸ     
	///  
	size_t getFactorSize() { return factor.size(); }
	///   
	/// \brief    ݽ״λ ȡ   Ϸ  ̵ ϵ      
	///   getFactor(2),   ǻ ȡy=a0+a1*x+a2*x^2+    +apoly_n*x^poly_n  a2  ֵ  
	/// \return    Ϸ  ̵ ϵ    
	///  
	double getFactor(size_t i) { return factor.at(i); }
private:
	template<typename T>
	void calcError(const T* x
		, const T* y
		, size_t length
		, double& r_ssr
		, double& r_sse
		, double& r_rmse
		, bool isSaveFitYs = true
	)
	{
		T mean_y = Mean<T>(y, length);
		T yi(0);
		fitedYs.reserve(length);
		for (int i = 0; i < length; ++i)
		{
			yi = getY(x[i]);
			r_ssr += ((yi - mean_y) * (yi - mean_y));//     ع ƽ      
			r_sse += ((yi - y[i]) * (yi - y[i]));// в ƽ      
			if (isSaveFitYs)
			{
				fitedYs.push_back(double(yi));
			}
		}
		r_rmse = sqrt(r_sse / (double(length)));
	}
	template<typename T>
	void gauss_solve(int n
		, std::vector< T>& A
		, std::vector<T>& x
		, std::vector<T>& b)
	{
		gauss_solve(n, &A[0], &x[0], &b[0]);
	}
	template<typename T>
	void gauss_solve(int n
		, T* A
		, T* x
		, T* b)
	{
		int i, j, k, r;
		double max;
		for (k = 0; k < n - 1; k++)
		{
			max = fabs(A[k * n + k]); /*find maxmum*/
			r = k;
			for (i = k + 1; i < n - 1; i++) {
				if (max < fabs(A[i * n + i]))
				{
					max = fabs(A[i * n + i]);
					r = i;
				}
			}
			if (r != k) {
				for (i = 0; i < n; i++)         /*change array:A[k]&A[r] */
				{
					max = A[k * n + i];
					A[k * n + i] = A[r * n + i];
					A[r * n + i] = max;
				}
			}
			max = b[k];                    /*change array:b[k]&b[r]     */
			b[k] = b[r];
			b[r] = max;
			for (i = k + 1; i < n; i++)
			{
				for (j = k + 1; j < n; j++)
					A[i * n + j] -= A[i * n + k] * A[k * n + j] / A[k * n + k];
				b[i] -= A[i * n + k] * b[k] / A[k * n + k];
			}
		}

		for (i = n - 1; i >= 0; x[i] /= A[i * n + i], i--)
			for (j = i + 1, x[i] = b[i]; j < n; j++)
				x[i] -= A[i * n + j] * x[j];
	}
};

/*========================================================================================================
* 函 数 名: Ployfit_Col
* 功能描述: 对”列“数据进行多项式拟合，并返回拟合结果
* 输入变量: 1.输入图片 2.拟合的阶数（若poly_n=2，则y=a0+a1*x+a2*x^2 ）3.是否保存数据（FLASE）
4.图片行数 5.图片列数 6.行拟合阈值
==========================================================================================================*/
Mat Ployfit_Col7(Mat img_col, int poly_n, bool isSaveOrNot, double Scoral)
{
	Fit fit1;
	int nRows = img_col.rows;
	int nCols = img_col.cols;
	int sigma = 0;

	vector<vector<int>> array(nRows);
	vector<vector<int>> array_col_diff(nRows);
	vector<vector<uchar>> array_clone2(nRows);
	vector<int> array_clone_cols(nRows);
	vector<int> array_rows(nRows);

	for (int i = 0; i < nRows; i++) {
		array_col_diff[i].resize(nCols);
		array_clone2[i].resize(nCols);
		array[i].resize(nCols);
	}
	array = Read_Mat7(array, img_col, nRows, nCols);  //将图像数据存入vector
	for (int b = 1; b < nRows + 1; b++) {
		array_rows[b - 1] = b;
	}
	for (int i = 0; i < nCols; i++)
	{
		for (int m = 0; m < nRows; m++)
		{
			if (array[m][i] > 100)
			{
				array_clone_cols[m] = array[m][i];
			}
			else {
				array_clone_cols[m] = 100;
			}
			if (m >= 1)
			{
				if (abs(array[m][i] - array_clone_cols[m - 1]) > 20)
				{
					array_clone_cols[m] = array_clone_cols[m - 1];
				}
			}
		}

		fit1.polyfit(array_rows, array_clone_cols, poly_n, isSaveOrNot);
		for (int j = 0; j < nRows; j++)
		{
			array_col_diff[j][i] = array[j][i] - (fit1.getFactor(0) + fit1.getFactor(1) * (j + 1) + fit1.getFactor(2) * pow(j + 1, 2) + fit1.getFactor(3) * pow(j + 1, 3));;
			if (array_col_diff[j][i] > sigma) {
				sigma = array_col_diff[j][i];
			}
		}
	}
	/* 对列列拟合进行阈值选择 */
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
			//if (array_col_diff[i][j] < sigma1 - 2)
		{
			if (abs(array_col_diff[i][j]) < Scoral)
				array_clone2[i][j] = 0;
			else
				array_clone2[i][j] = 255;
		}
	}
	Mat img = Vec2Mat7(array_clone2, nRows, nCols);
	return img;
}

/*=========================================================
* 函 数 名: adaptiveThresholdCustom
* 功能描述: 自适应阈值分割实现图像二值化
 =========================================================*/

void adaptiveThresholdCustom(const cv::Mat& src, cv::Mat& dst, double maxValue, int method, int type, int blockSize, double delta, double ratio, double fillValCoeffi)
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


	int top = (blockSize - 1) * 1;     //填充的上边界行数
	int bottom = (blockSize - 1) * 1;  //填充的下边界行数
	int left = (blockSize - 1) * 1;	   //填充的左边界行数
	int right = (blockSize - 1) * 1;   //填充的右边界行数
	int border_type = BORDER_CONSTANT; //边界填充方式
	Mat src_Expand;	                   //对原图像进行边界扩充

	Mat topImage = src(Rect(0, 0, src.cols, 3));//上边界一行图像

	cv::Scalar color = cv::mean(topImage) * fillValCoeffi;//35-80之间均可以  该值需要确定

											  //Scalar color = Scalar(50);//35-80之间均可以
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
bool compareContourAreas(std::vector< cv::Point> contour1, std::vector< cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}
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
		if (area > 10000)
		{
			drawContours(drawing, preContours, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
			drawContours(drawing, hull, i, Scalar(255), -1, 8, vector<Vec4i>(), 0, Point());
		}
	}
	src = drawing;
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

bool f_MainCam_PersTransMatCal(InputArray _src, int border_white, int border_biankuang, Mat* Mwhite, Mat* Mbiankuang, Mat* M_white_abshow, int ID, String ScreenType_Flag, int leftRightWhiteFlag)
{

	bool isArea_1, isArea_2;														//显示异常标志位
	Mat src = _src.getMat();                                                        //输入源图像

	//ROI加速2022.4.13
	Mat dst;
	resize(src, dst, Size(src.cols / 4, src.rows / 4), 0, 0, INTER_AREA);

	if (dst.type() == CV_8UC1)														//若输入8位图
		dst = dst.clone();															//拷贝原图
	else
		cvtColor(dst, dst, CV_BGR2GRAY);										    //灰度化彩色图

	convexSetPretreatment(dst);

	CV_Assert(dst.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(dst.size(), CV_8UC1);                              //二值图像
	threshold(dst, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4), src_corner_biankuang(4), src_corner_abshow(4);   //四个边相交得到角点坐标，漏光角点，显示异常角点
	Rect rect;																        //最小正外接矩形
	Mat three_channel = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC3);
	vector<Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binaryImage);
	}
	merge(channels, three_channel);
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 15000 && area < 200000000)
		{
			rect = boundingRect(contours[i]);
			Mat temp_mask = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC1);
			drawContours(temp_mask, contours, i, 255, FILLED, 8);
			displayError_Areasignal++;
			Mat sra_canny;
			Canny(temp_mask, sra_canny, 0, 255);
			dilate(sra_canny, sra_canny, cv::getStructuringElement(0, cv::Size(2, 2)), Point(-1, -1));

			for (int i = rect.y; i < rect.y + rect.height; i++) {
				const uchar* col = sra_canny.ptr<uchar>(i);
				for (int j = rect.x; j < rect.x + rect.width; j++) {
					//左侧点集
					for (int k = 1; k < 4; k++) {
						if (j < (rect.x + rect.width / 2) && col[j] > 100 && (sra_canny.ptr<uchar>(i + 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 4)[j] > 100)) {
							leftLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 2, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//右侧点集
						if (j > (rect.x + rect.width / 2) && col[j] > 100 && (sra_canny.ptr<uchar>(i + 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 4)[j] > 100)) {
							rightLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 2, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//上侧点集
						if (i < (rect.y + rect.height / 2) && col[j] > 100 && (sra_canny.ptr<uchar>(i + k)[j + 7 * k] > 100 || sra_canny.ptr<uchar>(i - k)[j + 7 * k] > 100 || col[j + 5] > 100)) {
							upLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 2, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//下侧点集
						if (i > (rect.y + rect.height / 2) && col[j] > 100 && (sra_canny.ptr<uchar>(i + k)[j + 7 * k] > 100 || sra_canny.ptr<uchar>(i - k)[j + 7 * k] > 100 || col[j + 5] > 100)) {
							downLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 2, 8, 0);
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
		if (false)//(ScreenType_Flag == "矩形屏")
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
		src_corner[0].x = 4 * src_corner[0].x - border_white;
		src_corner[0].y = 4 * src_corner[0].y - border_white;
		src_corner[1].x = 4 * src_corner[1].x - border_white;
		src_corner[1].y = 4 * src_corner[1].y + border_white;
		src_corner[2].x = 4 * src_corner[2].x + border_white;
		src_corner[2].y = 4 * src_corner[2].y + border_white;
		src_corner[3].x = 4 * src_corner[3].x + border_white;
		src_corner[3].y = 4 * src_corner[3].y - border_white;
		//对4个角点的坐标位置进行微调（漏光检测图）
		src_corner_biankuang[0].x = 4 * src_corner[0].x - border_biankuang;
		src_corner_biankuang[0].y = 4 * src_corner[0].y - border_biankuang;
		src_corner_biankuang[1].x = 4 * src_corner[1].x - border_biankuang;
		src_corner_biankuang[1].y = 4 * src_corner[1].y + border_biankuang;
		src_corner_biankuang[2].x = 4 * src_corner[2].x + border_biankuang;
		src_corner_biankuang[2].y = 4 * src_corner[2].y + border_biankuang;
		src_corner_biankuang[3].x = 4 * src_corner[3].x + border_biankuang;
		src_corner_biankuang[3].y = 4 * src_corner[3].y - border_biankuang;
		//显示异常(白底图)
		src_corner_abshow[0].x = 4 * src_corner[0].x - border_white + 10;
		src_corner_abshow[0].y = 4 * src_corner[0].y - border_white + 10;
		src_corner_abshow[1].x = 4 * src_corner[1].x - border_white + 10;
		src_corner_abshow[1].y = 4 * src_corner[1].y + border_white - 10;
		src_corner_abshow[2].x = 4 * src_corner[2].x + border_white - 10;
		src_corner_abshow[2].y = 4 * src_corner[2].y + border_white - 10;
		src_corner_abshow[3].x = 4 * src_corner[3].x + border_white - 10;
		src_corner_abshow[3].y = 4 * src_corner[3].y - border_white + 10;





		vector<Point2f> dst_corner(4);
		if (false)//(ScreenType_Flag == "矩形屏")
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


bool f_FrontCam_PersTransMatCal(InputArray _src, Mat* Mwhite, string ScreenType_Flag, int leftRightWhiteFlag)
{


	bool Ext_Result_Front = true;                                                     //提取屏幕成功标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);                                            //灰度化彩色图

	//ROI加速2022.4.13
	Mat dst;
	resize(src, dst, Size(src.cols / 4, src.rows / 4), 0, 0, INTER_AREA);


	if (leftRightWhiteFlag == 1)
	{
		convexSetPretreatment(dst);
	}


	CV_Assert(dst.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(dst.size(), CV_8UC1);                              //二值图像
	threshold(dst, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4), src_corner1(4);                                                  //四个边相交得到角点坐标
	vector<Point2f> src_corner_enlarge(4);
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //正接矩阵坐标点信息
	vector<Point2f> dst_corner(4);                                                  //透视变换后的点的信息
	Mat three_channel = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC3);
	vector<Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binaryImage);
	}
	merge(channels, three_channel);
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 25000 && area < 5000000)
		{
			rect = boundingRect(contours[i]);
			Mat temp_mask = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC1);
			drawContours(temp_mask, contours, i, 255, FILLED, 8);
			displayError_Areasignal++;
			Mat sra_canny;
			Canny(temp_mask, sra_canny, 0, 255);
			dilate(sra_canny, sra_canny, cv::getStructuringElement(0, cv::Size(2, 2)), Point(-1, -1));

			for (int i = rect.y; i < rect.y + rect.height; i++) {
				const uchar* col = sra_canny.ptr<uchar>(i);
				for (int j = rect.x; j < rect.x + rect.width; j++) {
					//左侧点集
					for (int k = 1; k < 4; k++) {
						if (j < (rect.x + rect.width / 2) && i<(rect.y + rect.height * 0.9) && i >(rect.y + rect.height * 0.1) && col[j] > 100 && (sra_canny.ptr<uchar>(i + 7 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 6)[j] > 100 || sra_canny.ptr<uchar>(i + 6)[j] > 100)) {
							leftLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 1, Scalar(0, 0, 255), 2, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//右侧点集
						if (j > (rect.x + rect.width / 2) && i <(rect.y + rect.height * 0.9) && i >(rect.y + rect.height * 0.1) && col[j] > 100 && (sra_canny.ptr<uchar>(i + 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 4)[j] > 100)) {
							rightLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 1, Scalar(0, 0, 255), 2, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//上侧点集
						if (i < (rect.y + rect.height / 2) && col[j] > 100 && (sra_canny.ptr<uchar>(i + k)[j + 3 * k] > 100 || sra_canny.ptr<uchar>(i - k)[j + 3 * k] > 100 || col[j + 4] > 100)) {
							upLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 1, Scalar(0, 0, 255), 2, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//下侧点集
						if (i > (rect.y + rect.height / 2) && col[j] > 100 && (sra_canny.ptr<uchar>(i + k)[j + 7 * k] > 100 || sra_canny.ptr<uchar>(i - k)[j + 7 * k] > 100 || col[j + 5] > 100)) {
							downLinePoint.push_back(Point(j, i));
							circle(three_channel, Point(j, i), 1, Scalar(0, 0, 255), 2, 8, 0);
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
				//line(three_channel,Point(leftLine_Fit[2]-1000, leftLine_Fit[3] - 1000 * leftLine_Fit[1] / leftLine_Fit[0]),Point(leftLine_Fit[2]+1000,leftLine_Fit[3]+1000*leftLine_Fit[1]/leftLine_Fit[0]),Scalar(0,255,0),3);
				//line(three_channel,Point(rightLine_Fit[2]-1000, rightLine_Fit[3] - 1000 * rightLine_Fit[1] / rightLine_Fit[0]),Point(rightLine_Fit[2]+1000, rightLine_Fit[3]+1000* rightLine_Fit[1]/ rightLine_Fit[0]),Scalar(0,255,0),3);
				//line(three_channel,Point(upLine_Fit[2]-1000, upLine_Fit[3] - 1000 * upLine_Fit[1] / upLine_Fit[0]),Point(upLine_Fit[2]+1000, upLine_Fit[3]+1000* upLine_Fit[1]/ upLine_Fit[0]),Scalar(0,255,0),3);
				//line(three_channel,Point(downLine_Fit[2]-1000, downLine_Fit[3] - 1000 * downLine_Fit[1] / downLine_Fit[0]),Point(downLine_Fit[2]+1000, downLine_Fit[3]+1000* downLine_Fit[1]/ downLine_Fit[0]),Scalar(0,255,0),3);
				//角点提取
				src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
				src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
				src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
				src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点

//                imwrite("D:\\roi.bmp",three_channel);
				//角点映射
				src_corner1[0].y = 4 * src_corner[0].y;
				src_corner1[0].x = 4 * src_corner[0].x;
				src_corner1[1].y = 4 * src_corner[1].y;
				src_corner1[1].x = 4 * src_corner[1].x;
				src_corner1[2].y = 4 * src_corner[2].y;
				src_corner1[2].x = 4 * src_corner[2].x;
				src_corner1[3].y = 4 * src_corner[3].y;
				src_corner1[3].x = 4 * src_corner[3].x;


				//透视变换矩阵计算
				if (false)//(ScreenType_Flag == "矩形屏")
					dst_corner = { Point(0, 0), Point(0, 1183), Point(3000, 1183), Point(3000, 0) };
				else
					dst_corner = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
				*Mwhite = cv::getPerspectiveTransform(src_corner1, dst_corner);
				Ext_Result_Front = false;
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
		Ext_Result_Front = true; //没有提取到屏幕
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (false)//(ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //透视变换矩阵提取
	}
	return Ext_Result_Front;
}


bool f_LeftRightCam_PersTransMatCal(InputArray _src, Mat* Mwhite, Mat* M_R_1_E, String ScreenType_Flag, int leftRightWhiteFlag, int border_white)
{

	//    clock_t start, end;
	//    start = clock();

	bool Ext_Result_Left_Right = true;                                                     //提取屏幕成功标志位
	Mat src = _src.getMat();                                                        //输入源图像
	if (src.type() == CV_8UC1)														//若输入8位图
		src = src.clone();															//拷贝原图
	else
		cvtColor(src, src, CV_BGR2GRAY);                                            //灰度化彩色图

	//ROI加速2022.4.13
	Mat dst;
	resize(src, dst, Size(src.cols / 4, src.rows / 4), 0, 0, INTER_AREA);


	if (leftRightWhiteFlag == 1)
	{
		convexSetPretreatment(dst);
	}
	CV_Assert(dst.depth() == CV_8U);                                                //8位无符号
	Mat binaryImage = Mat::zeros(dst.size(), CV_8UC1);                              //二值图像
	threshold(dst, binaryImage, 40, 255, CV_THRESH_BINARY);							//二值化(有问题)
	medianBlur(binaryImage, binaryImage, 5);										//中值滤波去除锯齿
	int displayError_Areasignal = 0;												//根据轮廓面积判定显示异常标志位
	vector<vector<Point>> contours;													//contours存放点集信息
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);    //CV_RETR_EXTERNAL最外轮廓，CV_CHAIN_APPROX_NONE所有轮廓点信息
	vector<Point> upLinePoint, leftLinePoint, downLinePoint, rightLinePoint;		//上左下右侧点集数据
	Vec4f upLine_Fit, leftLine_Fit, downLine_Fit, rightLine_Fit;				    //上左下右拟合直线数据
	vector<Point2f> src_corner(4);                                                  //四个边相交得到角点坐标
	vector<Point2f> src_corner_enlarge(4);
	vector<Point2f> src_corner1(4);
	Rect rect;																        //最小正外接矩形
	int x1, y1, x2, y2, x3, y3, x4, y4;			                                    //正接矩阵坐标点信息
	vector<Point2f> dst_corner(4);                                                  //透视变换后的点的信息
	Mat three_channel = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC3);
	vector<Mat> channels;



	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binaryImage);
	}
	merge(channels, three_channel);
	for (vector<int>::size_type i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 15000 && area < 6000000)
		{
			rect = boundingRect(contours[i]);
			Mat temp_mask = Mat::zeros(binaryImage.rows, binaryImage.cols, CV_8UC1);
			drawContours(temp_mask, contours, i, 255, FILLED, 8);
			displayError_Areasignal++;
			Mat sra_canny;
			Canny(temp_mask, sra_canny, 0, 255);
			dilate(sra_canny, sra_canny, cv::getStructuringElement(0, cv::Size(2, 2)), Point(-1, -1));
			//            rectangle(three_channel, rect, Scalar(255), 2, 8, 0);

			for (int i = rect.y; i < rect.y + rect.height; i++) {
				const uchar* col = sra_canny.ptr<uchar>(i);
				for (int j = rect.x; j < rect.x + rect.width; j++) {

					for (int k = 1; k < 4; k++) {
						//左侧点集
						if (j < (rect.x + rect.width / 2) && i<(rect.y + rect.height * 0.9) && i >(rect.y + rect.height * 0.1) && col[j] > 100 && (sra_canny.ptr<uchar>(i + 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 4)[j] > 100)) {
							leftLinePoint.push_back(Point(j, i));
							//                            circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 1, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//右侧点集
						if (j > (rect.x + rect.width / 2) && i<(rect.y + rect.height * 0.9) && i >(rect.y + rect.height * 0.1) && (col[j] > 100) && (sra_canny.ptr<uchar>(i + 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 5 * k)[j + k] > 100 || sra_canny.ptr<uchar>(i - 4)[j] > 100)) {
							rightLinePoint.push_back(Point(j, i));
							//                            circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 1, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//上侧点集
						if (i < (rect.y + rect.height / 2) && col[j] > 100 && ((col + k)[j + 7 * k] > 100 || (col - k)[j + 7 * k] > 100 || col[j + 5] > 100)) {
							upLinePoint.push_back(Point(j, i));
							//                            circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 1, 8, 0);
						}
					}
					for (int k = 1; k < 4; k++) {
						//下侧点集
						if (i > (rect.y + rect.height / 2) && col[j] > 100 && ((col + k)[j + 7 * k] > 100 || (col - k)[j + 7 * k] > 100 || col[j + 5] > 100)) {
							downLinePoint.push_back(Point(j, i));
							//                            circle(three_channel, Point(j, i), 3, Scalar(0, 0, 255), 1, 8, 0);
						}
					}
				}
			}
			//            imwrite("D:\\binary.bmp",three_channel);//2022.4.12调试删

			if (leftLinePoint.size() != 0 && rightLinePoint.size() != 0 && upLinePoint.size() != 0 && downLinePoint.size() != 0)
			{
				//直线拟合
				fitLine(leftLinePoint, leftLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);               //左侧拟合直线
				fitLine(rightLinePoint, rightLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//右侧拟合直线
				fitLine(upLinePoint, upLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);					//上侧拟合直线
				fitLine(downLinePoint, downLine_Fit, cv::DIST_L2, 0, 1e-2, 1e-2);				//下侧拟合直线
				//line(three_channel,Point(leftLine_Fit[2]-1000, leftLine_Fit[3] - 1000 * leftLine_Fit[1] / leftLine_Fit[0]),Point(leftLine_Fit[2]+1000,leftLine_Fit[3]+1000*leftLine_Fit[1]/leftLine_Fit[0]),Scalar(0,255,0),3);
				//line(three_channel,Point(rightLine_Fit[2]-1000, rightLine_Fit[3] - 1000 * rightLine_Fit[1] / rightLine_Fit[0]),Point(rightLine_Fit[2]+1000, rightLine_Fit[3]+1000* rightLine_Fit[1]/ rightLine_Fit[0]),Scalar(0,255,0),3);
				//line(three_channel,Point(upLine_Fit[2]-1000, upLine_Fit[3] - 1000 * upLine_Fit[1] / upLine_Fit[0]),Point(upLine_Fit[2]+1000, upLine_Fit[3]+1000* upLine_Fit[1]/ upLine_Fit[0]),Scalar(0,255,0),3);
				//line(three_channel,Point(downLine_Fit[2]-1000, downLine_Fit[3] - 1000 * downLine_Fit[1] / downLine_Fit[0]),Point(downLine_Fit[2]+1000, downLine_Fit[3]+1000* downLine_Fit[1]/ downLine_Fit[0]),Scalar(0,255,0),3);
				//角点提取
				src_corner[0] = getPointSlopeCrossPoint(upLine_Fit, leftLine_Fit);              //左上角点
				src_corner[1] = getPointSlopeCrossPoint(downLine_Fit, leftLine_Fit);		    //左下角点
				src_corner[2] = getPointSlopeCrossPoint(downLine_Fit, rightLine_Fit);	        //右下角点
				src_corner[3] = getPointSlopeCrossPoint(upLine_Fit, rightLine_Fit);		        //右上角点


				src_corner1[0].y = 4 * src_corner[0].y;
				src_corner1[0].x = 4 * src_corner[0].x;
				src_corner1[1].y = 4 * src_corner[1].y;
				src_corner1[1].x = 4 * src_corner[1].x;
				src_corner1[2].y = 4 * src_corner[2].y;
				src_corner1[2].x = 4 * src_corner[2].x;
				src_corner1[3].y = 4 * src_corner[3].y;
				src_corner1[3].x = 4 * src_corner[3].x;

				src_corner_enlarge[0].y = 4 * src_corner[0].y - border_white;
				src_corner_enlarge[0].x = 4 * src_corner[0].x - border_white;
				src_corner_enlarge[1].y = 4 * src_corner[1].y + border_white;
				src_corner_enlarge[1].x = 4 * src_corner[1].x - border_white;
				src_corner_enlarge[2].y = 4 * src_corner[2].y + border_white;
				src_corner_enlarge[2].x = 4 * src_corner[2].x + border_white;
				src_corner_enlarge[3].y = 4 * src_corner[3].y - border_white;
				src_corner_enlarge[3].x = 4 * src_corner[3].x + border_white;
				//透视变换矩阵计算
				if (false)//(ScreenType_Flag == "矩形屏")
					dst_corner = { Point(0, 0), Point(0, 1183), Point(3000, 1183), Point(3000, 0) };
				else
					dst_corner = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
				*Mwhite = cv::getPerspectiveTransform(src_corner1, dst_corner);
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

	//    end = clock();
	//    qDebug() << end- start << endl;



		//没有提取到屏幕
	if (displayError_Areasignal == 0)
	{
		Ext_Result_Left_Right = true; //没有提取到屏幕
		vector<Point2f> src_points(4);
		src_points = { Point(0, 0), Point(0, 10), Point(10, 10), Point(10, 0) };
		vector<Point2f> dst_points(4);
		if (false)//(ScreenType_Flag == "矩形屏")
			dst_points = { Point(0, 0), Point(0, 1775), Point(3000, 1775), Point(3000, 0) };
		else
			dst_points = { Point(0, 0), Point(0, 1500), Point(3000, 1500), Point(3000, 0) };
		*Mwhite = cv::getPerspectiveTransform(src_points, dst_points);                        //透视变换矩阵提取
		*M_R_1_E = cv::getPerspectiveTransform(src_points, dst_points);
	}
	return Ext_Result_Left_Right;
}
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

Mat gamma(Mat src, double g)
{
	Mat temp;
	src.convertTo(temp, CV_32FC3, 1 / 255.0);
	cv::Mat temp1;
	cv::pow(temp, g, temp1);
	Mat dst;
	temp1.convertTo(dst, CV_8UC1, 255);
	return dst;
}