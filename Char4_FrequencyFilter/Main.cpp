#include"iostream"
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>//dtfº¯Êý

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("test.jpg");

	int R = getOptimalDFTSize(image.rows);
	int C = getOptimalDFTSize(image.cols);

	Mat dst;
	copyMakeBorder(image, dst, 0, R - image.rows, 0, C - image.cols, BORDER_CONSTANT, Scalar::all(255));

	imshow("Border", dst);

	//dft();
	waitKey();
	return 0;
}