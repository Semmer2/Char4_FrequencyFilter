#include"iostream"
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>//dtfº¯Êý
#include"Fourior.h"

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Ori Image", image);
	
	Mat Newimage(FouriorTransit(image));
	//imshow("DFT Image", Newimage);

	image = InvertFouriorTransit(Newimage,image.size());
	imshow("IDFT Image", image);

	waitKey();
	return 0;
}