#include"iostream"
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>//dtf����
#include"Fourior.h"

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	
	FouriorTransit(image);

	return 0;
}