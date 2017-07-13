#pragma once
#include"iostream"
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>//dtf函数

using namespace std;
using namespace cv;

Mat FouriorTransit(Mat image);//输出未处理的原始傅里叶频域图，返回的是双通道图像
Mat InvertFouriorTransit(Mat image,Size size);//输出傅里叶频域图的原始图,因变换时经过填充，固需要知道原始图像大小
void LowPassFilter(Mat *src, double D0,int OpCode);//输入未处理的傅里叶频谱图，D0为临界值
void HighPassFilter(Mat *src, double D0, int OpCode);
void FilterTest(Mat image,int D0);