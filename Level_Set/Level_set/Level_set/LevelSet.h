#include <vector>  
#include <opencv2/opencv.hpp> 
using namespace cv;
using namespace std;



class LevelSet
{
public:
	LevelSet();
	~LevelSet();

	//��������  
	int m_iterNum;      //��������  
	float m_lambda1;    //ȫ����ϵ��  
	float m_nu;     //����Լ��ϵ����  
	float m_mu;     //�ͷ���ϵ����  
	float m_timestep;   //�ݻ�������t  
	float m_epsilon;    //���򻯲�����  

						//��������  
	Mat m_mImage;       //Դͼ��  

	int m_iCol;     //ͼ����  
	int m_iRow;     //ͼ��߶�  
	int m_depth;        //ˮƽ���������  
	float m_FGValue;    //ǰ��ֵ  
	float m_BKValue;    //����ֵ  

						//��ʼ��ˮƽ��  
	void initializePhi(Mat img,  //����ͼ��  
		int iterNum, //��������  
		Rect boxPhi);//ǰ������  
	void EVolution();   //�ݻ�  

	Mat m_mPhi;     //ˮƽ������  
protected:
	Mat m_mDirac;       //�����˴����ˮƽ�����ģ��գ�  
	Mat m_mHeaviside;   //���Ϻ��������ˮƽ���������գ�  
	Mat m_mCurv;        //ˮƽ�����ʦ�=div(����/|����|)  
	Mat m_mK;       //�ͷ�������  
	Mat m_mPenalize;    //�ͷ����еĨ�<sup>2</sup>��  

	void Dirac();       //�����˺���  
	void Heaviside();   //���Ϻ���  
	void Curvature();   //����  
	void BinaryFit();   //����ǰ���뱳��ֵ  
}; 
