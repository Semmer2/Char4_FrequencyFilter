#include"iostream"
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>//dtfº¯Êý
#include"Fourior.h"
#include"MyNSp.h"

using namespace std;
using namespace cv;
using namespace MyNSP;

int main()
{
	Mat image = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	FilterTest(image,100);

	waitKey();

	return 0;
}