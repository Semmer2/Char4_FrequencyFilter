#include"iostream"
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>//dtf函数
#include"MyNSp.h"

using namespace std;
using namespace cv;
using namespace MyNSP;

Mat FouriorTransit(Mat image)
{
	//imshow("Original", image);
	int R = getOptimalDFTSize(image.rows);
	int C = getOptimalDFTSize(image.cols);

	Mat dst;
	copyMakeBorder(image, dst, 0, R - image.rows, 0, C - image.cols, BORDER_CONSTANT, Scalar::all(255));
	Mat planes[] = { Mat_<float>(dst),Mat::zeros(dst.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);

	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magnitudeImage = planes[0];

	magnitudeImage += 1;
	log(magnitudeImage, magnitudeImage);
	magnitudeImage = magnitudeImage(Rect(0, 0, image.cols&-2, image.rows&-2));
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	Mat tmp;
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy));
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy));
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy));
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy));

	//对角象限交换
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q2.copyTo(tmp);
	q1.copyTo(q2);
	tmp.copyTo(q1);

	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
	imshow("MagIm final", magnitudeImage);
	return complexI;
}

Mat  InvertFouriorTransit(Mat image,Size size)//逆傅里叶变换，将频域图转为空域图
{
	Mat IDFTImage;
	dft(image, IDFTImage, DFT_INVERSE | DFT_REAL_OUTPUT);
	//imshow("magTmp", IDFTImage);
	normalize(IDFTImage, IDFTImage, 0, 1, CV_MINMAX);
	//imshow("IDFT", IDFTImage);
	IDFTImage = IDFTImage(Rect(0, 0, size.width, size.height));

	//waitKey();
	return IDFTImage;
}

void LowPassFilter(Mat *src, double D0,int OpCode)
{
	int state = -1;
	double tmpD;
	long width, height;
	width = src->cols;
	height = src->rows;
	long x = width / 2, y = height / 2;
	Mat H_mat(height, width, CV_64FC2);

	(*src).convertTo(*src, CV_64FC2);

	int i, j;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (i > y&&j > x)
				state = 3;
			else if (j > x)
				state = 2;
			else if (i > y)
				state = 1;
			else state = 0;
			switch (state)//计算各点到中心的长度？
			{
			case 0:
			{
				tmpD = (double)sqrt((i*i + j*j));
				break;
			}
			case 1:
			{
				tmpD = (double)sqrt((height - i)*(height - i) + j*j);
				break;
			}
			case 2:
			{
				tmpD = (double)sqrt(i*i + (j - width)*(j - width));
				break;
			}
			case 3:
			{
				tmpD = (double)sqrt((height - i)*(height - i) + (j - width)*(j - width));
				break;
			}
			}

			switch (OpCode)
			{
			case IdeaLPF:
			{
				if (tmpD < D0)
				{
					((double*)(H_mat.data + H_mat.step*i))[j * 2] = 1.0;
				}
				else
				{
					((double*)(H_mat.data + H_mat.step*i))[j * 2] = 0.0;
				}
				((double*)(H_mat.data + H_mat.step*i))[j * 2 + 1] = 0.0;//双通道的原因，其中偶数通道储存实部（第一个通道），奇数通道储存虚部（第二个通道）
				break;
			}
			case TrapeLPF:
			{
				int D1 = D0 + 20;
				if (tmpD < D0)
				{
					((double*)(H_mat.data + H_mat.step*i))[j * 2] = 1.0;
				}
				else if (tmpD<D1)
				{
					tmpD = (tmpD - D1) / (D0 - D1);
					((double*)(H_mat.data + H_mat.step*i))[j * 2] = tmpD;
				}
				else
				{
					((double*)(H_mat.data + H_mat.step*i))[j * 2] = 0.0;
				}
				((double*)(H_mat.data + H_mat.step*i))[j * 2 + 1] = 0.0;
				break;
			}
			case ButterworthLPF:
			{
				tmpD = 1 / (1 + pow(tmpD / D0, 2 * 2));
				((double*)(H_mat.data + H_mat.step*i))[j * 2] = tmpD;
				((double*)(H_mat.data + H_mat.step*i))[j * 2 + 1] = 0.0;
				break;
			}
			case ExpLPF:
			{
				tmpD = exp(-pow(tmpD / D0, 2));
				((double*)(H_mat.data + H_mat.step*i))[j * 2] = tmpD;
				((double*)(H_mat.data + H_mat.step*i))[j * 2 + 1] = 0.0;
				break;
			}
			default:
				break;
			}
		}
	}

	mulSpectrums(*src, H_mat, *src, DFT_ROWS);//进行两个傅里叶频谱乘法
}

void HighPassFilter(Mat *src, double D0, int OpCode)
{

}